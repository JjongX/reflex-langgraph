"""
The store submodule keeps the server's list of conversations.

LangGraph's checkpointer stores each conversation's graph state; this store keeps what the
server needs around it: who owns a thread, its title, and whether it is waiting for an
approval. Both must be durable for conversations to survive a restart::

    graph = build_graph(config, registry, checkpointer=sqlite_checkpointer("data/graph.db"))
    app = create_server(graph, threads=SqliteThreadStore("data/threads.db"), ...)
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Sequence, Union


@dataclass
class ThreadRecord:
    """
    What the server keeps about one conversation, besides its graph state.

    Attributes
    ----------
    id : str
        The thread id.
    owner : str, optional
        Id of the owner; ``None`` without an auth hook.
    created_at : str
        When it was created (ISO 8601, UTC).
    title : str
        The first line of its first message; empty until then.
    owner_principal : dict, optional
        The owner's id and display name. Never roles: permissions are looked up when a
        run starts.
    pending : bool
        Whether it waits for an approval decision.
    updated_at : str
        Its last message or decision; lists are ordered by it, newest first.

    Examples
    --------

    >>> from reflex_langgraph.server import MemoryThreadStore, ThreadRecord
    >>> store = MemoryThreadStore()
    >>> store.create(ThreadRecord(id='t1', owner=None, created_at='2026-10-01T09:00:00',
    ...                           title='Plan the run'))
    >>> store.create(ThreadRecord(id='t2', owner=None, created_at='2026-10-02T09:00:00',
    ...                           title='Check the results'))
    >>> [r.title for r in store.list()]
    ['Check the results', 'Plan the run']
    """

    id: str
    owner: Optional[str]
    created_at: str
    title: str = ""
    owner_principal: Optional[dict] = None
    pending: bool = False
    updated_at: str = ""

    @property
    def last_activity(self) -> str:
        """str : Its last message or decision, else when it was created."""
        return self.updated_at or self.created_at


class ThreadStore(Protocol):
    """What :func:`~reflex_langgraph.server.create_server` needs to keep conversations."""

    def create(self, record: ThreadRecord) -> None:
        """
        Add a conversation.

        Parameters
        ----------
        record : ThreadRecord
            The new conversation.
        """

    def get(self, thread_id: str) -> Optional[ThreadRecord]:
        """
        Return a conversation.

        Parameters
        ----------
        thread_id : str
            The thread id.

        Returns
        -------
        ThreadRecord or None
            The conversation; ``None`` when unknown.
        """

    def update(self, record: ThreadRecord, fields: Optional[Sequence[str]] = None) -> None:
        """
        Save changes to a conversation.

        Nothing happens when it was deleted, so a run that ends after a delete does not
        bring the conversation back.

        Parameters
        ----------
        record : ThreadRecord
            The conversation, changed.
        fields : sequence of str, optional
            Save only these fields, e.g. ``["pending"]``; by default, all of them. The
            server always names its fields: it may hold a copy read before someone else
            changed another field (a rename while a reply runs), which must survive.
        """

    def delete(self, thread_id: str) -> None:
        """
        Remove a conversation; nothing happens when it is unknown.

        Parameters
        ----------
        thread_id : str
            The thread id.
        """

    def list(self) -> List[ThreadRecord]:
        """
        Return every conversation.

        Returns
        -------
        list of ThreadRecord
            The conversations, newest activity first.
        """


class MemoryThreadStore(ThreadStore):
    """
    Keeps conversations in memory; the default.

    They are lost on restart, like the default in-memory checkpointer's graph state.

    Examples
    --------

    >>> from reflex_langgraph.server import MemoryThreadStore, ThreadRecord
    >>> store = MemoryThreadStore()
    >>> record = ThreadRecord(id='t1', owner=None, created_at='2026-10-01T09:00:00')
    >>> store.create(record)
    >>> store.delete('t1')
    >>> record.title = 'Plan the run'
    >>> store.update(record)   # deleted: not brought back
    >>> store.list()
    []
    """

    def __init__(self):
        self._items: Dict[str, ThreadRecord] = {}

    def create(self, record: ThreadRecord) -> None:
        self._items[record.id] = record

    def get(self, thread_id: str) -> Optional[ThreadRecord]:
        return self._items.get(thread_id)

    def update(self, record: ThreadRecord, fields: Optional[Sequence[str]] = None) -> None:
        stored = self._items.get(record.id)
        if stored is None:
            return
        if fields is None:
            self._items[record.id] = record
        else:
            for name in fields:
                setattr(stored, name, getattr(record, name))

    def delete(self, thread_id: str) -> None:
        self._items.pop(thread_id, None)

    def list(self) -> List[ThreadRecord]:
        return sorted(self._items.values(), key=lambda r: r.last_activity, reverse=True)


class SqliteThreadStore(ThreadStore):
    """
    Keeps conversations in one SQLite file, across restarts.

    Fine for a single server process. Pair it with
    :func:`~reflex_langgraph.agents.sqlite_checkpointer`: both must be durable for
    conversations to survive a restart. Files from older versions are upgraded in place.

    Parameters
    ----------
    path : str or pathlib.Path
        The SQLite file; its folder is created if needed.
    """

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS threads (
                id TEXT PRIMARY KEY, owner TEXT, created_at TEXT NOT NULL,
                title TEXT NOT NULL DEFAULT '', owner_principal TEXT,
                pending INTEGER NOT NULL DEFAULT 0, updated_at TEXT)""")
            # Files written before updated_at existed.
            columns = {row[1] for row in db.execute("PRAGMA table_info(threads)")}
            if "updated_at" not in columns:
                db.execute("ALTER TABLE threads ADD COLUMN updated_at TEXT")

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=10)

    _COLUMNS = "id, owner, created_at, title, owner_principal, pending, updated_at"

    @staticmethod
    def _row(row) -> ThreadRecord:
        return ThreadRecord(id=row[0], owner=row[1], created_at=row[2], title=row[3],
                            owner_principal=json.loads(row[4]) if row[4] else None,
                            pending=bool(row[5]), updated_at=row[6] or "")

    @staticmethod
    def _values(record: ThreadRecord) -> tuple:
        """The record's columns after ``id``, in ``_COLUMNS`` order."""
        d = asdict(record)
        return (d["owner"], d["created_at"], d["title"],
                json.dumps(d["owner_principal"], default=str) if d["owner_principal"] else None,
                int(d["pending"]), d["updated_at"] or None)

    def create(self, record: ThreadRecord) -> None:
        with self._lock, self._connect() as db:
            db.execute(
                f"INSERT OR REPLACE INTO threads ({self._COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (record.id, *self._values(record)))

    _UPDATABLE = ("owner", "created_at", "title", "owner_principal", "pending", "updated_at")

    def update(self, record: ThreadRecord, fields: Optional[Sequence[str]] = None) -> None:
        names = list(self._UPDATABLE if fields is None else fields)
        unknown = set(names) - set(self._UPDATABLE)
        if unknown:
            raise ValueError(f"unknown fields: {sorted(unknown)}")
        values = dict(zip(self._UPDATABLE, self._values(record)))
        with self._lock, self._connect() as db:
            db.execute(f"UPDATE threads SET {', '.join(f'{n} = ?' for n in names)} WHERE id = ?",
                       (*(values[n] for n in names), record.id))

    def delete(self, thread_id: str) -> None:
        with self._lock, self._connect() as db:
            db.execute("DELETE FROM threads WHERE id = ?", (thread_id,))

    def get(self, thread_id: str) -> Optional[ThreadRecord]:
        with self._connect() as db:
            row = db.execute(f"SELECT {self._COLUMNS} FROM threads WHERE id = ?",
                             (thread_id,)).fetchone()
        return self._row(row) if row else None

    def list(self) -> List[ThreadRecord]:
        with self._connect() as db:
            rows = db.execute(f"SELECT {self._COLUMNS} FROM threads "
                              "ORDER BY COALESCE(updated_at, created_at) DESC").fetchall()
        return [self._row(r) for r in rows]
