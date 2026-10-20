"""
The schemas submodule defines the server's HTTP response bodies.

With :mod:`~reflex_langgraph.server.events` (the SSE stream) this is the whole contract
between the server and any client. :class:`~reflex_langgraph.ui.BackendClient` validates
against these models, so a mismatch fails loudly instead of rendering wrongly.
:data:`API_VERSION` changes when a change would break an existing client.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from .events import ApprovalRequired

#: Version of the contract; ``reflex-langgraph ui`` refuses a backend with another one.
API_VERSION = 1

#: ``none``: no auth hook, anyone who reaches the server may use it. ``external``: an auth
#: hook identifies callers (a proxy header, an API gateway...). The built-in UI has no
#: login and sends no credentials, so it works with ``none`` only.
AuthMode = Literal["none", "external"]


class Health(BaseModel):
    """
    What the server is (``GET /health``), read by ``reflex-langgraph ui``.

    Attributes
    ----------
    ok : bool
        Always true.
    api : int
        The contract's version (:data:`API_VERSION`).
    version : str
        The package's version.
    title : str
        The app's title.
    auth : {'none', 'external'}
        How callers are identified.
    jobs : bool
        Whether the server has the jobs routes.
    approvals : bool
        Whether the server has the approvals queue.
    files : bool
        Whether users may attach files to their messages.
    max_file_mb : int
        The largest file accepted, in MB.
    max_files : int
        The most files one message may carry.
    accept : dict
        The file types accepted, each with its extensions (for the UI's file picker).
    """

    ok: bool = True
    api: int = API_VERSION
    version: str
    title: str
    auth: AuthMode
    jobs: bool
    approvals: bool = True
    files: bool = False
    max_file_mb: int = 0
    max_files: int = 0
    accept: Dict[str, List[str]] = Field(default_factory=dict)


class FileInfo(BaseModel):
    """
    A file attached in a conversation (``POST /threads/{id}/files``).

    Attributes
    ----------
    file_id : str
        Its id, to send with a message.
    name : str
        Its name.
    type : str
        Its type, e.g. ``application/pdf``.
    size : int
        Its size in bytes.
    kind : str
        How a model takes it: ``image``, ``pdf``, ``text`` or ``data``.
    """

    file_id: str
    name: str
    type: str
    size: int
    kind: str = "data"


class SessionInfo(BaseModel):
    """
    The caller as the server sees them (``GET /session``).

    Attributes
    ----------
    user : dict, optional
        The principal; ``None`` without auth.
    can_review : bool
        Whether the caller works from the approvals queue.
    """

    user: Optional[Dict[str, Any]] = None
    can_review: bool = False


class ThreadCreated(BaseModel):
    """
    A new conversation (``POST /threads``).

    Attributes
    ----------
    thread_id : str
        Its id.
    """

    thread_id: str


class ThreadSummary(BaseModel):
    """
    One conversation in ``GET /threads``, which lists them newest activity first.

    Attributes
    ----------
    thread_id : str
        Its id.
    title : str
        The first line of its first message, or the title it was renamed to; empty until
        a message is sent.
    created_at : str
        When it was created.
    updated_at : str
        Its last message or decision.
    running : bool
        Whether a run is going on in it.
    pending : bool
        Whether it waits for an approval decision.
    snippet : str
        In search results (``GET /threads?q=``), the text around the match when it is in
        a message rather than the title; otherwise empty.
    """

    thread_id: str
    title: str = ""
    created_at: str
    updated_at: str
    running: bool = False
    pending: bool = False
    snippet: str = ""


class ThreadDeleted(BaseModel):
    """
    A deleted conversation (``DELETE /threads/{id}``).

    Its run, if any, was stopped; its messages and pending approvals are gone. Jobs it
    started are not touched.

    Attributes
    ----------
    thread_id : str
        Its id.
    deleted : bool
        Always true.
    """

    thread_id: str
    deleted: bool = True


class HistoryToolCall(BaseModel):
    """
    A tool call in a conversation's history.

    Attributes
    ----------
    id : str
        The call, matched by the tool message with the same ``call_id``.
    tool : str
        The tool's name.
    args : dict
        Its arguments.
    """

    id: str
    tool: str
    args: Dict[str, Any] = Field(default_factory=dict)


class BlockInfo(BaseModel):
    """
    How the provider blocked a reply; see :class:`~reflex_langgraph.server.events.Blocked`.

    Attributes
    ----------
    reason : str
        The provider's code.
    scope : {'prompt', 'response'}
        What was blocked.
    message : str, optional
        The provider's explanation, if any.
    """

    reason: str
    scope: Literal["prompt", "response"]
    message: Optional[str] = None


class HistoryMessage(BaseModel):
    """
    One message in a conversation's history.

    Attributes
    ----------
    role : str
        ``user``, ``assistant``, ``tool`` or ``other``.
    text : str
        Its text.
    agent : str, optional
        The agent, for an assistant message.
    id : str, optional
        The message id.
    tool_calls : list of HistoryToolCall
        The tools an assistant message called.
    call_id : str, optional
        The call a tool message answers.
    tool : str, optional
        The tool, for a tool message.
    reasoning : str
        An assistant message's reasoning, when the model returned it.
    blocked : BlockInfo, optional
        Set when the provider blocked this reply.
    files : list of FileInfo
        The files a user message carried.
    """

    role: str
    text: str = ""
    agent: Optional[str] = None
    id: Optional[str] = None
    tool_calls: List[HistoryToolCall] = Field(default_factory=list)
    call_id: Optional[str] = None
    tool: Optional[str] = None
    reasoning: str = ""
    blocked: Optional[BlockInfo] = None
    files: List[FileInfo] = Field(default_factory=list)


class ThreadHistory(BaseModel):
    """
    A conversation, to show it again (``GET /threads/{id}``).

    Attributes
    ----------
    thread_id : str
        Its id.
    title : str
        Its title.
    running : bool
        Whether a run is going on in it.
    messages : list of HistoryMessage
        Its messages.
    pending : list of ApprovalRequired
        Approvals it waits for.
    """

    thread_id: str
    title: str = ""
    running: bool = False
    messages: List[HistoryMessage] = Field(default_factory=list)
    pending: List[ApprovalRequired] = Field(default_factory=list)


class QueuedApproval(ApprovalRequired):
    """
    An approval waiting in someone's conversation, as ``GET /approvals`` lists it.

    Attributes
    ----------
    thread_id : str
        The conversation.
    thread_title : str
        Its title.
    requested_by : str, optional
        Id of the conversation's owner.
    requested_by_name : str, optional
        Their display name.
    """

    thread_id: str
    thread_title: str = ""
    requested_by: Optional[str] = None
    requested_by_name: Optional[str] = None


class Cancelled(BaseModel):
    """The run was told to stop (``POST /threads/{id}/cancel``)."""

    cancelled: bool = True


# --- jobs -----------------------------------------------------------------------------

class JobRow(BaseModel):
    """
    One job in ``GET /jobs``.

    Attributes
    ----------
    job_id : str
        Its id.
    spec : str, optional
        Its kind.
    managed : bool
        Whether the runner created it.
    status : str
        E.g. ``running``, ``succeeded``, ``completed_with_failures``, ``cancelled``.
    submitted_at : str, optional
        When it was submitted.
    finished_at : str, optional
        When it ended.
    requested_by : str, optional
        Who asked for it.
    approved_by : str, optional
        Who approved it.
    """

    job_id: str
    spec: Optional[str] = None
    managed: bool = True
    status: str
    submitted_at: Optional[str] = None
    finished_at: Optional[str] = None
    requested_by: Optional[str] = None
    approved_by: Optional[str] = None


class JobStep(BaseModel):
    """One step of a job's progress; see :class:`~reflex_langgraph.jobs.Step`."""

    key: str = ""
    label: str
    status: str
    started: Optional[str] = None
    returncode: Optional[int] = None


class JobProgress(BaseModel):
    """A job's progress; see :meth:`~reflex_langgraph.jobs.Progress.to_dict`."""

    finished: bool = False
    counts: Dict[str, int] = Field(default_factory=dict)
    current: Optional[str] = None
    steps: List[JobStep] = Field(default_factory=list)


class JobSummary(BaseModel):
    """
    A job as the policy lets the caller see it (``GET /jobs/{id}``).

    The spec's public view and the policy's job view may drop or add keys; unknown keys
    pass through.

    Attributes
    ----------
    job_id : str
        Its id.
    status : str
        Its status.
    spec : str, optional
        Its kind.
    managed : bool, optional
        Whether the runner created it.
    state : dict
        Its recorded state.
    manifest : dict
        Its manifest: parameters, requester, approver, fingerprints.
    progress : JobProgress, optional
        Its step-by-step progress.
    """

    model_config = ConfigDict(extra="allow")

    job_id: str
    status: str
    spec: Optional[str] = None
    managed: Optional[bool] = None
    state: Dict[str, Any] = Field(default_factory=dict)
    manifest: Dict[str, Any] = Field(default_factory=dict)
    progress: Optional[JobProgress] = None


class JobCancelled(BaseModel):
    """
    The job was told to stop (``POST /jobs/{id}/cancel``).

    Attributes
    ----------
    job_id : str
        Its id.
    signalled : bool
        Whether a stop signal was sent.
    """

    job_id: str
    signalled: bool = True
