"""
The stream submodule translates a LangGraph run into the events the server streams.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, List, Optional

from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

from ..agents.blocked import block_info
from ..agents.builder import DELEGATE_PREFIX
from . import events as ev

#: Characters of a tool result streamed to the UI, by default.
MAX_TOOL_OUTPUT = 4000


def message_text(content: Any) -> str:
    """
    Return the plain text of a message's content.

    Providers return content as a string or as a list of parts; only text parts count
    (not reasoning, images or tool calls).

    Parameters
    ----------
    content : str or list
        A message's ``content``.

    Returns
    -------
    str
        The text.

    Examples
    --------

    >>> from reflex_langgraph.server.stream import message_text
    >>> message_text([{'type': 'thinking', 'thinking': 'Hmm.'}, {'type': 'text', 'text': 'Hi!'}])
    'Hi!'
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, str):
                out.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                out.append(part.get("text", ""))
        return "".join(out)
    return ""


def message_reasoning(message) -> str:
    """
    Return the reasoning in a message or a streamed chunk.

    It comes from LangChain's standard ``reasoning`` content blocks (Gemini thoughts,
    Claude thinking and OpenAI reasoning summaries all map to them), or from
    ``additional_kwargs["reasoning_content"]`` (DeepSeek-style APIs).

    Parameters
    ----------
    message : BaseMessage
        A message or chunk.

    Returns
    -------
    str
        The reasoning; empty when there is none.

    Examples
    --------

    >>> from langchain_core.messages import AIMessage
    >>> from reflex_langgraph.server.stream import message_reasoning
    >>> reply = AIMessage([{'type': 'thinking', 'thinking': 'Check the units first.'},
    ...                    {'type': 'text', 'text': '42 km'}],
    ...                   response_metadata={'model_provider': 'google_genai'})
    >>> message_reasoning(reply)
    'Check the units first.'
    """
    try:
        blocks = message.content_blocks
    except Exception:  # noqa: BLE001 - an unexpected provider shape: no reasoning
        blocks = []
    text = "".join(b.get("reasoning") or "" for b in blocks
                   if isinstance(b, dict) and b.get("type") == "reasoning")
    if not text:
        extra = (getattr(message, "additional_kwargs", None) or {}).get("reasoning_content")
        text = extra if isinstance(extra, str) else ""
    return text


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str)
    except TypeError:
        return str(value)


def interrupt_events(interrupts,
                     tool_agents: Dict[str, Optional[str]]) -> List[ev.ApprovalRequired]:
    """
    Turn a paused graph's interrupts into approval events.

    Parameters
    ----------
    interrupts : sequence of Interrupt
        The interrupts of the graph's state.
    tool_agents : dict
        Tool name to the agent that last called it, to label each approval.

    Returns
    -------
    list of ApprovalRequired
        One event per interrupt.
    """
    out = []
    for intr in interrupts:
        value = intr.value if isinstance(intr.value, dict) else {}
        allowed = {c.get("action_name"): c.get("allowed_decisions", [])
                   for c in value.get("review_configs", [])}
        requests = [
            ev.ActionRequest(tool=r.get("name", ""), args=r.get("args", {}) or {},
                             description=r.get("description", ""),
                             allowed=list(allowed.get(r.get("name"), [])))
            for r in value.get("action_requests", [])
        ]
        agent = tool_agents.get(requests[0].tool) if requests else None
        out.append(ev.ApprovalRequired(interrupt_id=intr.id, agent=agent, requests=requests))
    return out


async def run_events(graph, graph_input, config, *,
                     max_tool_output: int = MAX_TOOL_OUTPUT) -> AsyncIterator[ev._Event]:
    """
    Stream one run of a graph as events.

    A run answers a new message, resumes after decisions, or answers a message again.

    Parameters
    ----------
    graph : CompiledStateGraph
        The entry agent's graph.
    graph_input : dict, Command or None
        The new message; the decisions to resume with; or ``None`` to run again from the
        checkpoint named in ``config``.
    config : dict
        The run's config: ``thread_id``, and ``user`` and ``thread_owner`` for tools;
        ``checkpoint_id`` to run from an earlier checkpoint, which starts a new branch of
        the conversation (LangGraph's time travel).
    max_tool_output : int, default: 4000
        Characters of a tool result to stream.

    Yields
    ------
    Event
        ``token``, ``reasoning``, ``message``, ``blocked``, ``tool_call``,
        ``tool_result`` and ``approval_required`` events, then ``run_finished``.
    """
    call_agents: Dict[str, Optional[str]] = {}   # tool call id -> calling agent
    tool_agents: Dict[str, Optional[str]] = {}   # tool name -> last agent that called it

    async for ns, mode, chunk in graph.astream(graph_input, config,
                                               stream_mode=["messages", "updates"],
                                               subgraphs=True):
        if mode == "messages":
            msg, meta = chunk
            if isinstance(msg, AIMessageChunk):
                agent = meta.get("lc_agent_name")
                reasoning = message_reasoning(msg)
                if reasoning:  # the thinking summary streams before the answer
                    yield ev.Reasoning(agent=agent, message_id=msg.id, text=reasoning)
                text = message_text(msg.content)
                if text:
                    yield ev.Token(agent=agent, message_id=msg.id, text=text)
            continue

        # mode == "updates": {node: update}
        for node, update in (chunk or {}).items():
            if node == "__interrupt__":
                continue  # read once from the final state, below
            if not isinstance(update, dict):
                continue
            for msg in update.get("messages", []) or []:
                if isinstance(msg, AIMessage):
                    agent = msg.name
                    text, reasoning = message_text(msg.content), message_reasoning(msg)
                    if text or reasoning:
                        yield ev.Message(agent=agent, message_id=msg.id, text=text,
                                         reasoning=reasoning)
                    blocked = block_info(msg)
                    if blocked:
                        yield ev.Blocked(agent=agent, message_id=msg.id, **blocked)
                    for call in msg.tool_calls or []:
                        call_agents[call["id"]] = agent
                        tool_agents[call["name"]] = agent
                        name = call["name"]
                        yield ev.ToolCall(
                            agent=agent, call_id=call["id"], tool=name, args=call.get("args", {}),
                            delegate_to=name[len(DELEGATE_PREFIX):]
                            if name.startswith(DELEGATE_PREFIX) else None)
                elif isinstance(msg, ToolMessage):
                    output = _stringify(msg.content)
                    yield ev.ToolResult(
                        agent=call_agents.get(msg.tool_call_id), call_id=msg.tool_call_id,
                        tool=msg.name or "", output=output[:max_tool_output],
                        truncated=len(output) > max_tool_output)

    # The thread's latest state, which is the new branch after a run from a checkpoint.
    head = {**config, "configurable": {k: v for k, v in (config.get("configurable") or {}).items()
                                       if k != "checkpoint_id"}}
    state = await graph.aget_state(head)
    pending = interrupt_events(state.interrupts, tool_agents)
    for item in pending:
        yield item
    yield ev.RunFinished(status="interrupted" if pending else "completed")
