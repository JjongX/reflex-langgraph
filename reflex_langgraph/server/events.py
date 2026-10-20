"""
The events submodule defines the events a run streams from the server to the UI (SSE).

Every event has a ``type``, which is also its SSE event name. The events are the contract
between :mod:`reflex_langgraph.server` and :mod:`reflex_langgraph.ui` (or any other client),
so they are versioned with the package (``/health`` reports the contract's version).

A run streams ``run_started``, then any of ``token``, ``reasoning``, ``message``,
``blocked``, ``tool_call``, ``tool_result`` and ``approval_required``, and ends with
``run_finished`` (or ``error``).
"""

from __future__ import annotations

from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, TypeAdapter


class _Event(BaseModel):
    type: str


class RunStarted(_Event):
    """
    A run began.

    Attributes
    ----------
    thread_id : str
        The conversation.
    run_id : str
        This run.
    """

    type: Literal["run_started"] = "run_started"
    thread_id: str
    run_id: str


class Token(_Event):
    """
    A streamed fragment of an agent's reply.

    Attributes
    ----------
    agent : str, optional
        The agent replying.
    message_id : str, optional
        The reply it belongs to.
    text : str
        The fragment.
    """

    type: Literal["token"] = "token"
    agent: Optional[str]
    message_id: Optional[str]
    text: str


class Reasoning(_Event):
    """
    A streamed fragment of an agent's reasoning.

    It is the provider's thinking summary, when the model is configured to return it.

    Attributes
    ----------
    agent : str, optional
        The agent reasoning.
    message_id : str, optional
        The reply it belongs to.
    text : str
        The fragment.
    """

    type: Literal["reasoning"] = "reasoning"
    agent: Optional[str]
    message_id: Optional[str]
    text: str


class Message(_Event):
    """
    An agent's complete reply.

    It supersedes the tokens, and the streamed reasoning, with the same ``message_id``.

    Attributes
    ----------
    agent : str, optional
        The agent replying.
    message_id : str, optional
        The reply.
    text : str
        The answer; empty for a turn that only calls tools.
    reasoning : str
        Its reasoning, when the model returned it.
    """

    type: Literal["message"] = "message"
    agent: Optional[str]
    message_id: Optional[str]
    text: str
    reasoning: str = ""


class Blocked(_Event):
    """
    The model provider refused a request or withheld a reply.

    The cause is a safety filter, a policy, or a refusal.

    Attributes
    ----------
    agent : str, optional
        The agent whose model call was blocked.
    message_id : str, optional
        The blocked reply.
    reason : str
        The provider's code, e.g. ``SAFETY`` or ``refusal``.
    scope : {'prompt', 'response'}
        ``prompt``: the request was refused; ``response``: the reply was withheld.
    message : str, optional
        The provider's explanation, if any.
    """

    type: Literal["blocked"] = "blocked"
    agent: Optional[str]
    message_id: Optional[str]
    reason: str
    scope: Literal["prompt", "response"]
    message: Optional[str] = None


class ToolCall(_Event):
    """
    An agent called a tool (or handed a task to a worker).

    Attributes
    ----------
    agent : str, optional
        The calling agent.
    call_id : str
        The call, matched by its ``tool_result``.
    tool : str
        The tool's name.
    args : dict
        Its arguments.
    delegate_to : str, optional
        The worker, when the tool hands the task to another agent.
    """

    type: Literal["tool_call"] = "tool_call"
    agent: Optional[str]
    call_id: str
    tool: str
    args: Dict[str, Any]
    delegate_to: Optional[str] = None


class ToolResult(_Event):
    """
    A tool's result.

    Attributes
    ----------
    agent : str, optional
        The agent that called it.
    call_id : str
        The call.
    tool : str
        The tool's name.
    output : str
        The result, cut at ``max_tool_output`` characters.
    truncated : bool
        Whether it was cut.
    """

    type: Literal["tool_result"] = "tool_result"
    agent: Optional[str]
    call_id: str
    tool: str
    output: str
    truncated: bool = False


class ActionRequest(BaseModel):
    """
    One tool call waiting for a decision.

    Attributes
    ----------
    tool : str
        The tool's name.
    args : dict
        The arguments, as proposed.
    description : str
        Why it needs a decision.
    allowed : list of str
        The decisions offered (``approve``, ``edit``, ``reject``, ``respond``).
    """

    tool: str
    args: Dict[str, Any]
    description: str = ""
    allowed: List[str] = Field(default_factory=list)


class ApprovalRequired(_Event):
    """
    The run paused, with tool calls waiting for a person's decision.

    Attributes
    ----------
    interrupt_id : str
        What the decisions answer (``POST /threads/{id}/decisions``).
    agent : str, optional
        The agent whose tool calls wait.
    requests : list of ActionRequest
        The calls.
    can_decide : bool
        Whether the user receiving the event may answer it (the access policy decides).
    """

    type: Literal["approval_required"] = "approval_required"
    interrupt_id: str
    agent: Optional[str]
    requests: List[ActionRequest]
    can_decide: bool = True


class RunFinished(_Event):
    """
    The run ended.

    Attributes
    ----------
    status : {'completed', 'interrupted', 'cancelled'}
        ``interrupted`` means it is paused, waiting for an approval decision.
    """

    type: Literal["run_finished"] = "run_finished"
    status: Literal["completed", "interrupted", "cancelled"]


class Error(_Event):
    """
    The run failed.

    Attributes
    ----------
    message : str
        What to show the user: a reference to the server log, unless the server exposes
        errors.
    """

    type: Literal["error"] = "error"
    message: str


#: Any event, told apart by its ``type``.
Event = Annotated[Union[RunStarted, Token, Reasoning, Message, Blocked, ToolCall, ToolResult,
                        ApprovalRequired, RunFinished, Error], Field(discriminator="type")]

#: Validates one event: ``EVENT.validate_python({"type": name, **data})``.
EVENT = TypeAdapter(Event)
