"""
The checkpoint submodule keeps conversations' graph state durable.

:func:`~reflex_langgraph.agents.build_graph` accepts any LangGraph checkpointer. :func:`sqlite_checkpointer` is the
one-line durable choice. Any other LangGraph checkpointer (e.g. Postgres) works too, but the
server itself runs as one process: active runs live in its memory, so a shared database does
not make several workers safe.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Union

from langgraph.checkpoint.base import BaseCheckpointSaver


class _LazyAsyncSqliteSaver(BaseCheckpointSaver):
    """``AsyncSqliteSaver`` created on first use.

    ``AsyncSqliteSaver`` binds to the running event loop when constructed, but the graph is
    usually built at import time, before the server's loop exists. This wrapper defers the
    construction to the first call, inside the loop that serves requests.
    """

    def __init__(self, path: Path):
        super().__init__()
        self.path = path
        self._inner = None
        self._loop = None

    def _saver(self):
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        import aiosqlite

        loop = asyncio.get_running_loop()
        if self._inner is None or self._loop is not loop:
            if self._inner is not None:
                self._inner.conn.stop()  # bound to a loop that is gone; stop its thread
            # Not awaited here: AsyncSqliteSaver.setup() opens the connection on first use.
            self._inner = AsyncSqliteSaver(aiosqlite.connect(str(self.path)), serde=self.serde)
            self._loop = loop
        return self._inner

    async def aclose(self) -> None:
        """Close the connection. aiosqlite runs it on a non-daemon thread, which would
        otherwise keep the process from exiting; ``create_server`` calls this on shutdown."""
        inner, self._inner = self._inner, None
        if inner is not None:
            await inner.conn.close()

    async def aget_tuple(self, config):
        return await self._saver().aget_tuple(config)

    async def alist(self, config, *, filter=None, before=None, limit=None):
        async for item in self._saver().alist(config, filter=filter, before=before, limit=limit):
            yield item

    async def aput(self, config, checkpoint, metadata, new_versions):
        return await self._saver().aput(config, checkpoint, metadata, new_versions)

    async def aput_writes(self, config, writes, task_id, task_path=""):
        return await self._saver().aput_writes(config, writes, task_id, task_path)

    async def adelete_thread(self, thread_id):
        return await self._saver().adelete_thread(thread_id)

    async def aget_delta_channel_history(self, *, config, channels):
        return await self._saver().aget_delta_channel_history(config=config, channels=channels)

    def get_next_version(self, current: Any, channel: None) -> str:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        return AsyncSqliteSaver.get_next_version(self, current, channel)


def sqlite_checkpointer(path: Union[str, Path]) -> BaseCheckpointSaver:
    """
    Create a checkpointer that keeps every conversation's graph state in one SQLite file.

    The connection opens on first use, inside the server's event loop, and closes when
    the server stops. Fine for a single server process.

    Parameters
    ----------
    path : str or pathlib.Path
        The SQLite file; its folder is created if needed.

    Returns
    -------
    BaseCheckpointSaver
        The checkpointer, for :func:`~reflex_langgraph.agents.build_graph`.

    Raises
    ------
    ImportError
        If ``langgraph-checkpoint-sqlite`` or ``aiosqlite`` is missing.
    """
    try:
        import aiosqlite  # noqa: F401
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver  # noqa: F401
    except ImportError:
        raise ImportError("sqlite_checkpointer needs langgraph-checkpoint-sqlite and aiosqlite: "
                          "pip install reflex-langgraph") from None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return _LazyAsyncSqliteSaver(path)
