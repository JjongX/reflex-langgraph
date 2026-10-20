"""
The client submodule is an async client for the server's API, used by the Reflex states
and usable on its own.

Every response and streamed event is validated against :mod:`reflex_langgraph.server.schemas`
and :mod:`reflex_langgraph.server.events` and returned as plain dicts, so a server that
drifts from the contract fails here with a clear error instead of rendering wrongly.
"""

from __future__ import annotations

import json
import os
from typing import Any, AsyncIterator, BinaryIO, Dict, List, Optional, Tuple

import httpx
from httpx_sse import aconnect_sse
from pydantic import TypeAdapter, ValidationError

from ..server import schemas as sc
from ..server.events import EVENT

#: The backend's URL: ``$RLG_API_URL`` when this module is imported, else
#: ``http://localhost:8100``.
DEFAULT_API_URL = os.environ.get("RLG_API_URL", "http://localhost:8100")


class ApiError(RuntimeError):
    """
    Raised when the server answers with an error.

    Parameters
    ----------
    status : int
        The HTTP status.
    detail : str
        The server's explanation.
    """

    def __init__(self, status: int, detail: str):
        super().__init__(f"{status}: {detail}")
        self.status = status
        self.detail = detail


class ContractError(ApiError):
    """
    Raised when the server answers with something the contract does not allow.

    Parameters
    ----------
    what : str
        What was being read, e.g. ``"health"`` or ``"event"``.
    exc : pydantic.ValidationError
        How it failed validation.
    """

    def __init__(self, what: str, exc: ValidationError):
        super().__init__(502, f"unexpected {what} from the server: {exc.errors(include_url=False)}")


_ADAPTERS: Dict[Any, TypeAdapter] = {}


def _check(model, data: Any, what: str):
    """Validate ``data`` against a schema (a model, ``List[model]`` or a TypeAdapter) and
    return plain dicts."""
    if isinstance(model, TypeAdapter):
        adapter = model
    else:
        adapter = _ADAPTERS.get(model) or _ADAPTERS.setdefault(model, TypeAdapter(model))
    try:
        value = adapter.validate_python(data)
    except ValidationError as exc:
        raise ContractError(what, exc) from None
    return adapter.dump_python(value, mode="json")


def _detail(resp: httpx.Response) -> str:
    try:
        return str(resp.json().get("detail", resp.text))
    except (ValueError, AttributeError):
        return resp.text


class BackendClient:
    """
    Async client for the server's API.

    Every response and streamed event is validated against the contract and returned as
    plain data.

    Parameters
    ----------
    base_url : str, default: DEFAULT_API_URL
        The backend's URL.
    timeout : float, default: 30
        Seconds to wait for a response (streams have no read timeout).
    headers : dict, optional
        Sent with every request, e.g. what a backend's auth hook reads.
    transport : httpx.AsyncBaseTransport, optional
        E.g. ``httpx.ASGITransport(app=app)`` to call an app in-process, as tests do.
    """

    def __init__(self, base_url: str = DEFAULT_API_URL, timeout: float = 30,
                 *, headers: Optional[Dict[str, str]] = None,
                 transport: Optional[httpx.AsyncBaseTransport] = None):
        self.base_url = base_url.rstrip("/")
        self.headers = dict(headers or {})
        self.timeout = timeout
        self.transport = transport

    def _client(self, timeout: Optional[httpx.Timeout] = None) -> httpx.AsyncClient:
        return httpx.AsyncClient(base_url=self.base_url, headers=self.headers,
                                 timeout=timeout or self.timeout, transport=self.transport)

    async def request(self, method: str, path: str, **kw) -> Any:
        """
        Send a request and return its JSON body, unvalidated.

        Parameters
        ----------
        method : str
            HTTP method.
        path : str
            Path under the base URL.
        **kw
            Passed to ``httpx.AsyncClient.request`` (e.g. ``json``, ``params``).

        Returns
        -------
        Any
            The JSON body.

        Raises
        ------
        ApiError
            If the server answers with an error.
        """
        async with self._client() as c:
            resp = await c.request(method, path, **kw)
        if resp.status_code >= 400:
            raise ApiError(resp.status_code, _detail(resp))
        return resp.json()

    async def stream(self, path: str, body: Dict[str, Any]) -> AsyncIterator[Tuple[str, dict]]:
        """
        Send a POST and read its SSE stream until the server closes it.

        Parameters
        ----------
        path : str
            Path under the base URL.
        body : dict
            The JSON body.

        Yields
        ------
        (str, dict)
            Each event's type and data, validated.

        Raises
        ------
        ApiError
            If the server answers with an error.
        ContractError
            If an event breaks the contract.
        """
        # No read timeout: a tool call (e.g. a long preflight) may be silent for minutes.
        timeout = httpx.Timeout(self.timeout, read=None)
        async with self._client(timeout) as c:
            async with aconnect_sse(c, "POST", path, json=body) as source:
                if source.response.status_code >= 400:
                    await source.response.aread()
                    raise ApiError(source.response.status_code, _detail(source.response))
                async for sse in source.aiter_sse():
                    if sse.event == "ping" or not sse.data:
                        continue
                    data = _check(EVENT, {"type": sse.event, **json.loads(sse.data)}, "event")
                    yield data.pop("type"), data

    async def health(self) -> dict:
        """
        Read ``GET /health``.

        Returns
        -------
        dict
            See :class:`~reflex_langgraph.server.schemas.Health`.
        """
        return _check(sc.Health, await self.request("GET", "/health"), "health")

    async def session(self) -> dict:
        """
        Read ``GET /session``.

        Returns
        -------
        dict
            See :class:`~reflex_langgraph.server.schemas.SessionInfo`.
        """
        return _check(sc.SessionInfo, await self.request("GET", "/session"), "session")

    async def new_thread(self) -> str:
        """
        Create a conversation.

        Returns
        -------
        str
            Its thread id.
        """
        return _check(sc.ThreadCreated, await self.request("POST", "/threads"),
                      "thread")["thread_id"]

    async def threads(self, limit: int = 100, q: str = "") -> list:
        """
        List the conversations the caller may see, newest activity first.

        Parameters
        ----------
        limit : int, default: 100
            At most this many.
        q : str, default: ''
            Only those whose title or messages contain this text (ignoring case).

        Returns
        -------
        list of dict
            See :class:`~reflex_langgraph.server.schemas.ThreadSummary`.
        """
        params = {"limit": limit, **({"q": q} if q else {})}
        return _check(List[sc.ThreadSummary],
                      await self.request("GET", "/threads", params=params), "threads")

    async def history(self, thread_id: str) -> dict:
        """
        Read a conversation.

        Parameters
        ----------
        thread_id : str
            The thread id.

        Returns
        -------
        dict
            See :class:`~reflex_langgraph.server.schemas.ThreadHistory`.
        """
        return _check(sc.ThreadHistory, await self.request("GET", f"/threads/{thread_id}"),
                      "history")

    async def rename_thread(self, thread_id: str, title: str) -> dict:
        """
        Rename a conversation.

        Parameters
        ----------
        thread_id : str
            The conversation.
        title : str
            Its new title.

        Returns
        -------
        dict
            See :class:`~reflex_langgraph.server.schemas.ThreadSummary`.
        """
        return _check(sc.ThreadSummary, await self.request(
            "PATCH", f"/threads/{thread_id}", json={"title": title}), "thread")

    async def delete_thread(self, thread_id: str) -> dict:
        """
        Delete a conversation, stopping its run first.

        Parameters
        ----------
        thread_id : str
            The conversation.

        Returns
        -------
        dict
            See :class:`~reflex_langgraph.server.schemas.ThreadDeleted`.
        """
        return _check(sc.ThreadDeleted, await self.request("DELETE", f"/threads/{thread_id}"),
                      "deleted thread")

    async def approvals(self) -> list:
        """
        List the approvals the caller may answer.

        Returns
        -------
        list of dict
            See :class:`~reflex_langgraph.server.schemas.QueuedApproval`.
        """
        return _check(List[sc.QueuedApproval], await self.request("GET", "/approvals"),
                      "approvals")

    def send(self, thread_id: str, content: str, files: Optional[List[str]] = None):
        """
        Send a message and stream the run it starts.

        Parameters
        ----------
        thread_id : str
            The conversation.
        content : str
            The message; it may be empty when files are attached.
        files : list of str, optional
            Ids of files uploaded to the conversation (:meth:`upload_file`).

        Returns
        -------
        async iterator of (str, dict)
            The run's events (see :meth:`stream`).
        """
        return self.stream(f"/threads/{thread_id}/messages",
                           {"content": content, "files": list(files or [])})

    async def upload_file(self, thread_id: str, name: str, data: BinaryIO) -> dict:
        """
        Upload a file to a conversation, to attach to a message.

        Parameters
        ----------
        thread_id : str
            The conversation.
        name : str
            The file's name; its extension decides its type.
        data : file object
            Its contents, sent in chunks.

        Returns
        -------
        dict
            See :class:`~reflex_langgraph.server.schemas.FileInfo`.

        Raises
        ------
        ApiError
            If the server refuses it (too large: 413; type not accepted: 415; uploads
            off: 404).
        """
        # Large files take a while: no read timeout.
        async with self._client(httpx.Timeout(self.timeout, read=None, write=None)) as c:
            resp = await c.post(f"/threads/{thread_id}/files", files={"file": (name, data)})
        if resp.status_code >= 400:
            raise ApiError(resp.status_code, _detail(resp))
        return _check(sc.FileInfo, resp.json(), "file")

    async def delete_file(self, thread_id: str, file_id: str) -> None:
        """
        Remove a file from a conversation, e.g. one taken off a message before sending.

        Parameters
        ----------
        thread_id : str
            The conversation.
        file_id : str
            The file.
        """
        async with self._client() as c:
            resp = await c.delete(f"/threads/{thread_id}/files/{file_id}")
        if resp.status_code >= 400:
            raise ApiError(resp.status_code, _detail(resp))

    def regenerate(self, thread_id: str):
        """
        Answer a conversation's last message again and stream that run.

        The reply it replaces drops out of the conversation.

        Parameters
        ----------
        thread_id : str
            The conversation.

        Returns
        -------
        async iterator of (str, dict)
            The run's events (see :meth:`stream`).
        """
        return self.stream(f"/threads/{thread_id}/regenerate", {})

    def decide(self, thread_id: str, interrupt_id: str, decisions: list):
        """
        Answer an approval and stream the run it resumes.

        Parameters
        ----------
        thread_id : str
            The conversation.
        interrupt_id : str
            The approval.
        decisions : list of dict
            One decision per request, e.g. ``{"type": "approve"}``.

        Returns
        -------
        async iterator of (str, dict)
            The run's events (see :meth:`stream`).
        """
        return self.stream(f"/threads/{thread_id}/decisions",
                           {"interrupt_id": interrupt_id, "decisions": decisions})

    async def cancel_run(self, thread_id: str) -> dict:
        """
        Stop the run going on in a conversation.

        Parameters
        ----------
        thread_id : str
            The conversation.

        Returns
        -------
        dict
            See :class:`~reflex_langgraph.server.schemas.Cancelled`.
        """
        return await self.request("POST", f"/threads/{thread_id}/cancel")

    async def jobs(self) -> list:
        """
        List the jobs the caller may see.

        Returns
        -------
        list of dict
            See :class:`~reflex_langgraph.server.schemas.JobRow`.
        """
        return _check(List[sc.JobRow], await self.request("GET", "/jobs"), "job list")

    async def job(self, job_id: str) -> dict:
        """
        Read a job.

        Parameters
        ----------
        job_id : str
            The job id.

        Returns
        -------
        dict
            See :class:`~reflex_langgraph.server.schemas.JobSummary`.
        """
        return _check(sc.JobSummary, await self.request("GET", f"/jobs/{job_id}"), "job")

    async def cancel_job(self, job_id: str) -> dict:
        """
        Stop a running job.

        Parameters
        ----------
        job_id : str
            The job id.

        Returns
        -------
        dict
            See :class:`~reflex_langgraph.server.schemas.JobCancelled`.
        """
        return await self.request("POST", f"/jobs/{job_id}/cancel")
