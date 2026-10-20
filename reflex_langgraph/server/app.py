"""
The app submodule builds the FastAPI application that serves a graph to the UI.

Routes (JSON bodies are the models in :mod:`.schemas`; the three run routes stream SSE
events from :mod:`.events`)::

    GET    /health                                        -> Health
    GET    /session                                       -> SessionInfo
    POST   /threads                                       -> ThreadCreated
    GET    /threads?limit=100&q=                          -> [ThreadSummary], newest first
    GET    /threads/{id}                                  -> ThreadHistory
    PATCH  /threads/{id}            {"title"}             -> ThreadSummary
    DELETE /threads/{id}                                  -> ThreadDeleted
    POST   /threads/{id}/files      (multipart "file")    -> FileInfo
    GET    /threads/{id}/files                            -> [FileInfo]
    DELETE /threads/{id}/files/{file_id}
    POST   /threads/{id}/messages   {"content", "files"}  -> SSE
    POST   /threads/{id}/regenerate                       -> SSE
    POST   /threads/{id}/decisions  {"interrupt_id", "decisions"}  -> SSE
    POST   /threads/{id}/cancel                           -> Cancelled
    GET    /approvals                                     -> [QueuedApproval]

``q`` searches titles and the text of messages. ``regenerate`` answers the conversation's
last message again: the run starts from the checkpoint where that message had just been
added, so the reply it replaces (and an approval it waited for) drops out of the
conversation. Nothing that already ran is undone, and the new run calls its tools again:
ordinary tools run again, tools behind an approval ask again.

There is no login. Without ``auth`` (the default) everyone who reaches the server is the same
anonymous user. An optional ``auth`` callable can identify callers from the request (a header
set by a proxy or gateway); whatever it returns is the principal. Authorisation is the ``policy``
(:class:`~reflex_langgraph.policy.AccessPolicy`). Tools receive the principal who triggered
the run as ``config["configurable"]["user"]`` and the conversation's owner as
``config["configurable"]["thread_owner"]``, so a tool that needs to know *who approved*
reads it from there rather than trusting an argument the model wrote.

Permissions are always the current ones: ``thread_owner`` is the signed-in principal when
the owner triggered the run, and otherwise only the owner's id and name, with no roles. A
conversation never stores roles, so taking a role away takes effect in old conversations
too.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Annotated, Any, Awaitable, Callable, Dict, List, Literal, Optional, Set, Union

from fastapi import APIRouter, Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.types import Command
from pydantic import BaseModel, Field, StringConstraints
from sse_starlette.sse import EventSourceResponse
from starlette.concurrency import run_in_threadpool

from ..agents.blocked import block_info
from ..files import ATTACHMENTS_KEY, FILES_KEY, FileError, FileStore, FileTooLarge
from ..policy import AccessPolicy, Principal, principal_id
from ..version import __version__
from . import events as ev
from . import schemas as sc
from .store import MemoryThreadStore, ThreadRecord, ThreadStore
from .stream import (MAX_TOOL_OUTPUT, interrupt_events, message_reasoning, message_text,
                     run_events)

log = logging.getLogger("reflex_langgraph.server")

#: ``auth(request) -> principal or None``, sync or async; ``None`` means not authenticated.
AuthHook = Callable[[Request], Union[Optional[Principal], Awaitable[Optional[Principal]]]]

#: Set on a router to declare a capability in ``/health`` (``jobs_router`` sets "jobs").
CAPABILITY_ATTR = "rlg_capability"


class NewMessage(BaseModel):
    """
    A message (``POST /threads/{id}/messages``).

    Attributes
    ----------
    content : str
        The message, up to 20,000 characters; it may be empty when files are attached.
    files : list of str
        Ids of files uploaded to the conversation (``POST /threads/{id}/files``).
    """

    content: str = Field("", max_length=20000)
    files: List[str] = Field(default_factory=list)


class RenameThread(BaseModel):
    """
    A conversation's new title (``PATCH /threads/{id}``).

    Attributes
    ----------
    title : str
        The title, 1 to 200 characters once trimmed.
    """

    title: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]


class DecisionIn(BaseModel):
    """
    One decision on a tool call waiting for approval.

    Attributes
    ----------
    type : {'approve', 'edit', 'reject', 'respond'}
        The decision.
    edited_action : dict, optional
        For ``edit``: ``{"name": tool, "args": {...}}``.
    message : str, optional
        For ``reject`` and ``respond``: shown to the agent.
    """

    type: Literal["approve", "edit", "reject", "respond"]
    edited_action: Optional[Dict[str, Any]] = None
    message: Optional[str] = None


class Decisions(BaseModel):
    """
    The answer to an approval (``POST /threads/{id}/decisions``).

    Attributes
    ----------
    interrupt_id : str
        The approval.
    decisions : list of DecisionIn
        One decision per tool call, in order.
    """

    interrupt_id: str
    decisions: List[DecisionIn] = Field(min_length=1)


@dataclass
class _Run:
    """One agent run (a message or a resumed approval).

    The server owns it: the graph runs in its own task, and the HTTP streams that started or
    follow it only *watch* it. A dropped connection or a restarted UI stops the watching,
    never the run; only ``POST /threads/{id}/cancel`` stops a run.
    """
    task: Optional[asyncio.Task] = None
    #: SSE payloads so far, so a watcher that starts late still sees the whole run.
    events: List[dict] = field(default_factory=list)
    done: bool = False
    changed: asyncio.Condition = field(default_factory=asyncio.Condition)

    async def emit(self, event: ev._Event) -> None:
        async with self.changed:
            self.events.append(_sse(event))
            self.changed.notify_all()

    async def finish(self) -> None:
        async with self.changed:
            self.done = True
            self.changed.notify_all()

    async def watch(self):
        """Yield every event of the run, from the first, until it ends."""
        seen = 0
        while True:
            async with self.changed:
                await self.changed.wait_for(lambda: seen < len(self.events) or self.done)
                batch, done = self.events[seen:], self.done
            for item in batch:
                yield item
            seen += len(batch)
            if done and seen == len(self.events):
                return


@dataclass
class _Runtime:
    """Per-thread state that only makes sense inside this process. Kept only while the
    thread has a live run."""
    run: Optional[_Run] = None

    @property
    def running(self) -> bool:
        return self.run is not None and not self.run.done


def _now() -> str:
    # Microseconds: conversation lists sort by these, and messages can come within a second.
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _owner_record(user: Optional[Principal]) -> Optional[dict]:
    """What a thread keeps about its owner: id and name, never roles (see run_config)."""
    uid = principal_id(user)
    return {"id": uid, "name": _display_name(user) or uid} if uid is not None else None


def _display_name(user: Optional[Principal]) -> Optional[str]:
    if isinstance(user, dict):
        return str(user.get("name") or user.get("id") or "") or None
    return principal_id(user)


def auth_mode(auth: Optional[AuthHook]) -> str:
    """
    Return how a server identifies callers, as ``/health`` reports it.

    Parameters
    ----------
    auth : callable, optional
        The server's auth hook.

    Returns
    -------
    str
        ``none`` without a hook, ``external`` with one.
    """
    return "none" if auth is None else "external"


def create_server(graph, *, title: str = "reflex-langgraph", auth: Optional[AuthHook] = None,
                  policy: Optional[AccessPolicy] = None, threads: Optional[ThreadStore] = None,
                  cors_origins: Optional[List[str]] = None,
                  routers: Optional[List[APIRouter]] = None,
                  recursion_limit: int = 100, max_tool_output: int = MAX_TOOL_OUTPUT,
                  expose_errors: bool = False, files: Optional[FileStore] = None,
                  max_file_mb: int = 100, max_files: int = 30) -> FastAPI:
    """
    Create the FastAPI application that serves a graph.

    Runs belong to the server: a run executes in its own task and HTTP streams only watch
    it, so a dropped connection or a restarted UI never stops it. Serve it from **one
    process** (one uvicorn worker): which runs are going on, cancel and the one-run-per-
    conversation guard live in that process's memory. A shared database alone does not
    make several workers safe.

    Parameters
    ----------
    graph : CompiledStateGraph
        The entry agent's graph (see :func:`~reflex_langgraph.agents.build_graph`). A graph
        built another way must offer what the conversation features rely on:

        * a ``messages`` state key with LangGraph's ``add_messages`` reducer: a run's input
          is ``{"messages": [user message]}``, and history, search and Try again read it;
        * a checkpointer, which also supports ``adelete_thread`` (delete) and running from
          an earlier checkpoint (Try again);
        * approval pauses in the format of LangChain's ``HumanInTheLoopMiddleware``
          (``action_requests`` and ``review_configs``), resumed with
          ``Command(resume={interrupt_id: {"decisions": [...]}})``;
        * chat models whose replies stream as message chunks, with the agent's name in the
          run metadata (``lc_agent_name``), so replies are labelled.
    title : str, default: 'reflex-langgraph'
        Shown by the UI, which reads it from ``/health``.
    auth : callable, optional
        ``auth(request) -> principal or None`` identifies callers (e.g. from a header set
        by a proxy). Without it, everyone is one anonymous user; the built-in UI has no
        login and needs it unset.
    policy : AccessPolicy, optional
        Who may read conversations, answer approvals and use the approvals queue; pass
        the same object to your :class:`~reflex_langgraph.jobs.JobRunner`. By default,
        :class:`~reflex_langgraph.policy.AccessPolicy`.
    threads : ThreadStore, optional
        Keeps the conversation list. By default, in memory (see
        :class:`~reflex_langgraph.server.SqliteThreadStore`).
    cors_origins : list of str, optional
        Origins allowed to call the API from a browser.
    routers : list of fastapi.APIRouter, optional
        Mounted as they are, e.g. :func:`~reflex_langgraph.jobs.jobs_router`.
    recursion_limit : int, default: 100
        Graph steps allowed per run.
    max_tool_output : int, default: 4000
        Characters of a tool result streamed to the UI.
    expose_errors : bool, default: False
        Send a crashed run's exception text to the user instead of a reference id.
        Development only: it can contain server paths.
    files : FileStore, optional
        Where attached files are kept, e.g.
        :class:`~reflex_langgraph.files.LocalFileStore`. Without it, users cannot attach
        files (the UI hides the paperclip).
    max_file_mb : int, default: 100
        The largest file accepted, in MB.
    max_files : int, default: 30
        The most files one message may carry.

    Returns
    -------
    fastapi.FastAPI
        The application, e.g. for ``uvicorn backend:app``.

    Examples
    --------

    >>> from fastapi.testclient import TestClient
    >>> from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    >>> from reflex_langgraph import build_graph, create_server, load_config
    >>> config = load_config({'model': 'x:y', 'agents': [{'name': 'assistant'}]})
    >>> model = GenericFakeChatModel(messages=iter([]))  # no API key; /health calls no model
    >>> graph = build_graph(config, models={'assistant': model})
    >>> app = create_server(graph, title='My agents')
    >>> with TestClient(app) as client:
    ...     client.get('/health').json()['title']
    'My agents'
    """
    policy = policy or AccessPolicy()
    store: ThreadStore = threads if threads is not None else MemoryThreadStore()
    runtimes: Dict[str, _Runtime] = {}
    deleting: Set[str] = set()  # threads being deleted: no run may start on them
    capabilities = {getattr(r, CAPABILITY_ATTR, None) for r in routers or []}
    health_body = sc.Health(version=__version__, title=title, auth=auth_mode(auth),
                            jobs="jobs" in capabilities, files=files is not None,
                            max_file_mb=max_file_mb if files else 0,
                            max_files=max_files if files else 0,
                            accept=files.accepted() if files else {})

    @asynccontextmanager
    async def lifespan(_app):
        yield
        # Stop runs still going, before closing what they write to.
        live = [rt.run.task for rt in runtimes.values() if rt.running and rt.run.task]
        for task in live:
            task.cancel()
        await asyncio.gather(*live, return_exceptions=True)
        # e.g. sqlite_checkpointer's connection thread
        close = getattr(getattr(graph, "checkpointer", None), "aclose", None)
        if close is not None:
            await close()

    app = FastAPI(title=title, lifespan=lifespan)
    if cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=cors_origins, allow_credentials=True,
                           allow_methods=["*"], allow_headers=["*"])

    async def current_user(request: Request) -> Optional[Principal]:
        if auth is None:
            return None
        result = auth(request)
        if inspect.isawaitable(result):
            result = await result
        if result is None:
            raise HTTPException(401, "not authenticated")
        return result

    app.state.current_user = current_user
    app.state.policy = policy
    app.state.runtimes = runtimes  # threads with a live run (diagnostics, tests)

    def runtime(thread_id: str) -> _Runtime:
        return runtimes.setdefault(thread_id, _Runtime())

    def running(thread_id: str) -> bool:
        rt = runtimes.get(thread_id)
        return bool(rt and rt.running)

    def thread_for_view(thread_id: str, user) -> ThreadRecord:
        rec = store.get(thread_id)
        if rec is None or not policy.can_view_thread(user, rec.owner):
            raise HTTPException(404, "thread not found")
        return rec

    def thread_owner(rec: ThreadRecord, user) -> Optional[Principal]:
        """The owner with their current permissions when they are the one acting; else only
        their id and name, so no stored or borrowed role ever widens what a tool shows."""
        if principal_id(user) == rec.owner:
            return user
        return {"id": rec.owner, "name": _display_name(rec.owner_principal) or rec.owner}

    def run_config(rec: ThreadRecord, user) -> dict:
        configurable = {"thread_id": rec.id, "user": user, "thread_owner": thread_owner(rec, user)}
        if files is not None:
            configurable[FILES_KEY] = files  # for the attachments middleware and tools
        return {"configurable": configurable, "recursion_limit": recursion_limit}

    def file_store() -> FileStore:
        if files is None:
            raise HTTPException(404, "this server does not accept files")
        return files

    async def pending_for(rec: ThreadRecord, user) -> List[ev.ApprovalRequired]:
        """Pending approvals of a thread, as ``user`` may act on them. Refreshes the flag."""
        state = await graph.aget_state(run_config(rec, user))
        pending = interrupt_events(state.interrupts, {})
        if rec.pending != bool(pending):
            rec.pending = bool(pending)
            store.update(rec, ["pending"])
        decide = policy.can_decide(user, rec.owner)
        for p in pending:
            p.can_decide = decide
        return pending

    def summary(rec: ThreadRecord, snippet: str = "") -> sc.ThreadSummary:
        return sc.ThreadSummary(thread_id=rec.id, title=rec.title, created_at=rec.created_at,
                                updated_at=rec.last_activity, running=running(rec.id),
                                pending=rec.pending, snippet=snippet)

    async def matching_text(rec: ThreadRecord, needle: str) -> Optional[str]:
        """The text around ``needle`` (lower case) in the first message that contains it."""
        state = await graph.aget_state({"configurable": {"thread_id": rec.id}})
        for m in (state.values or {}).get("messages", []):
            if isinstance(m, (HumanMessage, AIMessage)):
                text = message_text(m.content)
                at = text.lower().find(needle)
                if at >= 0:
                    return _snippet(text, at, len(needle))
        return None

    async def execute(run: _Run, rec: ThreadRecord, graph_input, user, config: dict) -> None:
        """The run itself, in its own task: unaffected by who is watching."""
        decide = policy.can_decide(user, rec.owner)
        try:
            await run.emit(ev.RunStarted(thread_id=rec.id, run_id=str(uuid.uuid4())))
            async for item in run_events(graph, graph_input, config,
                                         max_tool_output=max_tool_output):
                if isinstance(item, ev.ApprovalRequired):
                    item.can_decide = decide
                elif isinstance(item, ev.RunFinished):
                    rec.pending = item.status == "interrupted"
                    store.update(rec, ["pending"])
                await run.emit(item)
        except asyncio.CancelledError:  # POST /cancel, or server shutdown
            await run.emit(ev.RunFinished(status="cancelled"))
        except Exception as exc:  # surfaced to the UI, not swallowed
            ref = uuid.uuid4().hex[:8]
            log.exception("run failed on thread %s [ref %s]", rec.id, ref)
            await run.emit(ev.Error(message=f"{type(exc).__name__}: {exc}" if expose_errors
                                    else f"Something went wrong (ref {ref}). The details "
                                         "are in the server log."))
        finally:
            await run.finish()
            # Forget a finished run: its events stay only as long as a stream still reads
            # them (the stream holds the run), so memory does not grow with every thread.
            rt = runtimes.get(rec.id)
            if rt is not None and rt.run is run:
                del runtimes[rec.id]

    def stream(rec: ThreadRecord, graph_input, user, *,
               from_checkpoint: Optional[str] = None) -> EventSourceResponse:
        # Deleted (or being deleted) while the request awaited the graph's state.
        if rec.id in deleting or store.get(rec.id) is None:
            raise HTTPException(404, "thread not found")
        rt = runtime(rec.id)
        if rt.running:
            raise HTTPException(409, "a run is already in progress on this thread")
        config = run_config(rec, user)
        if from_checkpoint:
            config["configurable"]["checkpoint_id"] = from_checkpoint
        # No await between the checks and here: two requests cannot both start a run.
        rt.run = run = _Run()
        run.task = asyncio.create_task(execute(run, rec, graph_input, user, config),
                                       name=f"run-{rec.id}")
        return EventSourceResponse(run.watch(), ping=15)

    @app.get("/health", response_model=sc.Health)
    async def health():
        # Also tells `reflex-langgraph ui` how to build the page for this server.
        return health_body

    @app.get("/session", response_model=sc.SessionInfo)
    async def session(user=Depends(current_user)):
        return sc.SessionInfo(user=user, can_review=policy.can_review(user))

    @app.post("/threads", response_model=sc.ThreadCreated)
    async def create_thread(user=Depends(current_user)):
        rec = ThreadRecord(id=str(uuid.uuid4()), owner=principal_id(user), created_at=_now(),
                           owner_principal=_owner_record(user))
        store.create(rec)
        return sc.ThreadCreated(thread_id=rec.id)

    @app.get("/threads", response_model=List[sc.ThreadSummary])
    async def list_threads(user=Depends(current_user),
                           limit: int = Query(100, ge=1, le=1000),
                           q: str = Query("", max_length=200)):
        """The caller's conversations, newest activity first; with ``q``, those whose title
        or messages contain it (ignoring case)."""
        needle = q.strip().lower()
        out = []
        for r in store.list():
            if len(out) >= limit:
                break
            if not policy.can_view_thread(user, r.owner):
                continue
            snippet = ""
            if needle:
                if not r.title:
                    continue  # no message yet
                if needle not in r.title.lower():
                    # Reads the conversation: fine for the hundreds of chats a person keeps.
                    snippet = await matching_text(r, needle)
                    if snippet is None:
                        continue
            out.append(summary(r, snippet))
        return out

    @app.get("/threads/{thread_id}", response_model=sc.ThreadHistory)
    async def get_history(thread_id: str, user=Depends(current_user)):
        rec = thread_for_view(thread_id, user)
        state = await graph.aget_state(run_config(rec, user))
        return sc.ThreadHistory(
            thread_id=rec.id, title=rec.title, running=running(rec.id),
            messages=[_history_item(m) for m in (state.values or {}).get("messages", [])],
            pending=await pending_for(rec, user))

    @app.patch("/threads/{thread_id}", response_model=sc.ThreadSummary)
    async def rename_thread(thread_id: str, body: RenameThread, user=Depends(current_user)):
        """Rename a conversation; its place in the list does not change."""
        rec = thread_for_view(thread_id, user)
        rec.title = body.title
        store.update(rec, ["title"])
        return summary(rec)

    @app.delete("/threads/{thread_id}", response_model=sc.ThreadDeleted)
    async def delete_thread(thread_id: str, user=Depends(current_user)):
        """Delete a conversation: stop its run, then remove its graph state and record."""
        rec = thread_for_view(thread_id, user)
        deleting.add(rec.id)  # from here on, no run starts on it
        try:
            rt = runtimes.get(rec.id)
            if rt is not None and rt.running:
                rt.run.task.cancel()
                await asyncio.gather(rt.run.task, return_exceptions=True)
            saver = getattr(graph, "checkpointer", None)
            if isinstance(saver, BaseCheckpointSaver):
                try:
                    await saver.adelete_thread(rec.id)
                except NotImplementedError:
                    raise HTTPException(501, "this server's checkpointer cannot delete "
                                             "conversations") from None
            if files is not None:
                await run_in_threadpool(files.delete_thread, rec.id)
            store.delete(rec.id)
        finally:
            deleting.discard(rec.id)
        return sc.ThreadDeleted(thread_id=rec.id)

    @app.post("/threads/{thread_id}/files", response_model=sc.FileInfo)
    async def upload_file(thread_id: str, file: UploadFile = File(...),
                          user=Depends(current_user)):
        """Keep a file for the conversation; send its id with a message to attach it."""
        rec, store_ = thread_for_view(thread_id, user), file_store()
        try:
            record = await run_in_threadpool(store_.save, rec.id, file.filename or "file",
                                             file.file, max_file_mb << 20)
        except FileTooLarge as exc:
            raise HTTPException(413, str(exc)) from None
        except FileError as exc:
            raise HTTPException(415, str(exc)) from None
        return _file_info(record.ref())

    @app.get("/threads/{thread_id}/files", response_model=List[sc.FileInfo])
    async def list_files(thread_id: str, user=Depends(current_user)):
        rec, store_ = thread_for_view(thread_id, user), file_store()
        return [_file_info(r.ref()) for r in await run_in_threadpool(store_.list, rec.id)]

    @app.delete("/threads/{thread_id}/files/{file_id}", status_code=204)
    async def delete_file(thread_id: str, file_id: str, user=Depends(current_user)):
        """Remove a file, e.g. one taken off a message before it was sent."""
        rec, store_ = thread_for_view(thread_id, user), file_store()
        await run_in_threadpool(store_.delete, rec.id, file_id)

    @app.post("/threads/{thread_id}/messages")
    async def post_message(thread_id: str, body: NewMessage, user=Depends(current_user)):
        rec = thread_for_view(thread_id, user)
        if not body.content.strip() and not body.files:
            raise HTTPException(422, "a message needs text or files")
        refs = []
        if body.files:
            if len(body.files) > max_files:
                raise HTTPException(422, f"at most {max_files} files per message")
            store_ = file_store()
            for file_id in body.files:
                record = store_.get(rec.id, file_id)
                if record is None:
                    raise HTTPException(422, f"no file {file_id!r} in this conversation")
                refs.append(record.ref())
        state = await graph.aget_state(run_config(rec, user))
        if state.interrupts:
            raise HTTPException(409, "the thread is waiting for an approval decision")
        changed = ["updated_at"]
        if not rec.title:  # the first message (or its first file) names the conversation
            first_line = (body.content.strip().splitlines() or [""])[0]
            rec.title = (first_line or (refs[0]["name"] if refs else ""))[:80]
            changed.append("title")
        rec.updated_at = _now()
        store.update(rec, changed)
        message = HumanMessage(body.content,
                               additional_kwargs={ATTACHMENTS_KEY: refs} if refs else {})
        return stream(rec, {"messages": [message]}, user)

    @app.post("/threads/{thread_id}/regenerate")
    async def regenerate(thread_id: str, user=Depends(current_user)):
        """Answer the last message again, on a new branch from where it was sent."""
        rec = thread_for_view(thread_id, user)
        if running(rec.id):
            raise HTTPException(409, "a run is already in progress on this thread")
        start = await _turn_start(graph, run_config(rec, user))
        if start is None:
            raise HTTPException(409, "there is no message to answer again")
        rec.updated_at = _now()
        store.update(rec, ["updated_at"])
        return stream(rec, None, user, from_checkpoint=start)

    @app.post("/threads/{thread_id}/decisions")
    async def post_decisions(thread_id: str, body: Decisions, user=Depends(current_user)):
        rec = store.get(thread_id)
        if rec is None:
            raise HTTPException(404, "thread not found")
        if not policy.can_decide(user, rec.owner):
            # Do not reveal threads the caller may not even see.
            if policy.can_view_thread(user, rec.owner):
                raise HTTPException(403, "you are not allowed to answer approval requests")
            raise HTTPException(404, "thread not found")
        state = await graph.aget_state(run_config(rec, user))
        pending = {i.id: i for i in state.interrupts}
        if body.interrupt_id not in pending:
            raise HTTPException(409, "no such pending approval")
        _check_decisions(pending[body.interrupt_id].value, body.decisions)
        rec.updated_at = _now()
        store.update(rec, ["updated_at"])
        resume = {body.interrupt_id: {
            "decisions": [d.model_dump(exclude_none=True) for d in body.decisions]}}
        return stream(rec, Command(resume=resume), user)

    @app.post("/threads/{thread_id}/cancel", response_model=sc.Cancelled)
    async def cancel(thread_id: str, user=Depends(current_user)):
        rec = thread_for_view(thread_id, user)
        rt = runtimes.get(rec.id)
        if rt is None or not rt.running:
            raise HTTPException(409, "nothing is running on this thread")
        rt.run.task.cancel()
        return sc.Cancelled()

    @app.get("/approvals", response_model=List[sc.QueuedApproval])
    async def approvals(user=Depends(current_user)):
        """Approvals the caller may answer: across all conversations for reviewers, in their
        own conversations otherwise."""
        uid, review = principal_id(user), policy.can_review(user)
        out: List[sc.QueuedApproval] = []
        for rec in store.list():
            if not rec.pending or running(rec.id):
                continue
            if not policy.can_decide(user, rec.owner) or not (review or rec.owner == uid):
                continue
            for p in await pending_for(rec, user):
                out.append(sc.QueuedApproval(
                    **p.model_dump(), thread_id=rec.id, thread_title=rec.title,
                    requested_by=rec.owner, requested_by_name=_display_name(rec.owner_principal)))
        return out

    for router in routers or []:
        app.include_router(router)
    return app


def _check_decisions(value: Any, decisions: List[DecisionIn]) -> None:
    value = value if isinstance(value, dict) else {}
    requests = value.get("action_requests", [])
    configs = {c.get("action_name"): c.get("allowed_decisions", [])
               for c in value.get("review_configs", [])}
    if len(decisions) != len(requests):
        raise HTTPException(422, f"expected {len(requests)} decisions, got {len(decisions)}")
    for req, dec in zip(requests, decisions):
        allowed = configs.get(req.get("name"), [])
        if dec.type not in allowed:
            raise HTTPException(422, f"{dec.type!r} is not allowed for {req.get('name')!r}")
        if dec.type == "edit" and not (dec.edited_action and "args" in dec.edited_action):
            raise HTTPException(422, "edit requires edited_action with args")


async def _turn_start(graph, config: dict) -> Optional[str]:
    """The checkpoint where the last user message had just been added, if there is one.

    A run from it answers that message again on a new branch (LangGraph's time travel), and
    the thread's latest state becomes that branch. The walk follows parent checkpoints, so it
    stays on the current branch after earlier regenerations.
    """
    snapshot = await graph.aget_state(config)
    messages = (snapshot.values or {}).get("messages", [])
    last = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
    if last is None:
        return None
    found = snapshot
    while any(m.id == last.id for m in (snapshot.values or {}).get("messages", [])):
        found = snapshot
        if snapshot.parent_config is None:
            break
        snapshot = await graph.aget_state(snapshot.parent_config)
    return found.config["configurable"]["checkpoint_id"]


def _snippet(text: str, at: int, length: int, width: int = 90) -> str:
    """The text around a match, on one line, with an ellipsis where it was cut."""
    start, end = max(0, at - width // 3), min(len(text), at + length + width)
    out = " ".join(text[start:end].split())
    return ("…" if start > 0 else "") + out + ("…" if end < len(text) else "")


def _history_item(m) -> sc.HistoryMessage:
    if isinstance(m, HumanMessage):
        return sc.HistoryMessage(role="user", text=message_text(m.content),
                                 files=[_file_info(r) for r in
                                        m.additional_kwargs.get(ATTACHMENTS_KEY) or []])
    if isinstance(m, AIMessage):
        return sc.HistoryMessage(
            role="assistant", agent=m.name, id=m.id, text=message_text(m.content),
            tool_calls=[sc.HistoryToolCall(id=c["id"], tool=c["name"], args=c.get("args", {}))
                        for c in m.tool_calls or []],
            reasoning=message_reasoning(m), blocked=block_info(m))
    if isinstance(m, ToolMessage):
        text = m.content if isinstance(m.content, str) else message_text(m.content)
        return sc.HistoryMessage(role="tool", call_id=m.tool_call_id, tool=m.name, text=text)
    return sc.HistoryMessage(role=getattr(m, "type", "other"),
                             text=message_text(getattr(m, "content", "")))


def _file_info(ref: dict) -> sc.FileInfo:
    return sc.FileInfo(file_id=ref["id"], name=ref["name"], type=ref["type"], size=ref["size"],
                       kind=ref.get("kind") or "data")


def _sse(event: ev._Event) -> dict:
    return {"event": event.type, "data": json.dumps(event.model_dump(), default=str)}
