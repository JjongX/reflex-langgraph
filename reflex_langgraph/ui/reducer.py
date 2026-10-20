"""
The reducer submodule folds the server's events into the items the chat shows.

It is pure (no Reflex, no I/O), so the UI's behaviour can be tested without a browser.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import List, Optional


@dataclass
class ChatItem:
    """
    One item of the chat.

    Attributes
    ----------
    kind : str
        ``user``, ``assistant``, ``tool``, ``blocked``, ``error`` or ``notice``.
    agent : str
        The agent, for assistant, tool and blocked items.
    text : str
        What it says.
    reasoning : str
        An assistant item's reasoning, when the model returned it.
    thinking : bool
        Whether the reasoning is still streaming and the answer has not started.
    thought_seconds : int
        How long the model reasoned, when measured live; 0 when unknown (history).
    reasoning_started : float
        When the reasoning began (epoch seconds), to measure it.
    show_agent : bool
        Whether to name the agent: only for agents other than the one the user talks to.
    message_id : str
        The reply it shows (assistant and blocked items).
    call_id : str
        The tool call it shows (tool items).
    tool : str
        The tool's name (tool items).
    args : str
        The tool's arguments, as indented JSON.
    output : str
        The tool's result.
    delegate_to : str
        The worker, when the tool handed a task to another agent.
    status : str
        A tool item's ``running`` or ``done``.
    files : list of str
        The names of the files a user message carried.
    """

    kind: str = ""
    agent: str = ""
    text: str = ""
    reasoning: str = ""
    thinking: bool = False
    thought_seconds: int = 0
    reasoning_started: float = 0.0
    show_agent: bool = False
    message_id: str = ""
    call_id: str = ""
    tool: str = ""
    args: str = ""
    output: str = ""
    delegate_to: str = ""
    status: str = ""
    files: List[str] = field(default_factory=list)


@dataclass
class RequestView:
    """
    One tool call on an approval card.

    Attributes
    ----------
    tool : str
        The tool's name.
    description : str
        Why it needs a decision.
    args : str
        The arguments as proposed, as indented JSON.
    edited_args : str
        What the reviewer typed, when editing.
    allowed : list of str
        The decisions offered.
    """

    tool: str = ""
    description: str = ""
    args: str = ""
    edited_args: str = ""
    allowed: List[str] = field(default_factory=list)


@dataclass
class ApprovalView:
    """
    An approval card.

    Attributes
    ----------
    interrupt_id : str
        The approval.
    agent : str
        The agent whose tool calls wait.
    requests : list of RequestView
        The calls.
    can_decide : bool
        False when this user must wait for someone else (a reviewer) to answer.
    thread_id : str
        The conversation (set in the approvals queue).
    thread_title : str
        Its title (set in the approvals queue).
    requested_by : str
        Who asked (set in the approvals queue).
    """

    interrupt_id: str = ""
    agent: str = ""
    requests: List[RequestView] = field(default_factory=list)
    can_decide: bool = True
    thread_id: str = ""
    thread_title: str = ""
    requested_by: str = ""


def pretty(value) -> str:
    """
    Format a value as indented JSON for display.

    Parameters
    ----------
    value : Any
        The value.

    Returns
    -------
    str
        Indented JSON, or ``str(value)`` when it is not JSON.

    Examples
    --------

    >>> from reflex_langgraph.ui.reducer import pretty
    >>> print(pretty({'job_id': 'demo1'}))
    {
      "job_id": "demo1"
    }
    """
    try:
        return json.dumps(value, indent=2, ensure_ascii=False, default=str)
    except TypeError:
        return str(value)


def _find(items: List[ChatItem], **match) -> Optional[ChatItem]:
    for item in reversed(items):
        if all(getattr(item, k) == v for k, v in match.items()):
            return item
    return None


def approval_from(data: dict) -> ApprovalView:
    """
    Make an approval card from an ``approval_required`` event or a queued approval.

    Parameters
    ----------
    data : dict
        The event's data.

    Returns
    -------
    ApprovalView
        The card.
    """
    return ApprovalView(
        interrupt_id=data["interrupt_id"], agent=data.get("agent") or "",
        requests=[RequestView(tool=r["tool"], description=r.get("description", ""),
                              args=pretty(r.get("args", {})), edited_args=pretty(r.get("args", {})),
                              allowed=list(r.get("allowed", [])))
                  for r in data.get("requests", [])],
        can_decide=bool(data.get("can_decide", True)),
        thread_id=data.get("thread_id") or "", thread_title=data.get("thread_title") or "",
        requested_by=data.get("requested_by_name") or data.get("requested_by") or "")


def blocked_text(data: dict) -> str:
    """
    Say what the provider blocked, for the chat.

    Parameters
    ----------
    data : dict
        A ``blocked`` event's data, or a history message's ``blocked``.

    Returns
    -------
    str
        The notice.

    Examples
    --------

    >>> from reflex_langgraph.ui.reducer import blocked_text
    >>> blocked_text({'reason': 'SAFETY', 'scope': 'response'})
    'The model provider blocked this response (SAFETY). Rephrase it to continue.'
    """
    what = "your request" if data.get("scope") == "prompt" else "this response"
    text = f"The model provider blocked {what} ({data.get('reason') or 'no reason given'})."
    if data.get("message"):
        text += f" {data['message']}"
    return text + " Rephrase it to continue."


def history_group(updated_at: str, now: Optional[datetime] = None,
                  tz_offset_minutes: int = 0) -> str:
    """
    Name the group a conversation falls in, by its last activity, as ChatGPT's list does.

    Parameters
    ----------
    updated_at : str
        Its last activity (ISO 8601).
    now : datetime, optional
        The current time. By default, now.
    tz_offset_minutes : int, default: 0
        The browser's ``Date.getTimezoneOffset()`` (UTC minus local time, in minutes), so
        days start at the user's midnight.

    Returns
    -------
    str
        ``Today``, ``Yesterday``, ``Previous 7 days``, ``Previous 30 days`` or ``Older``.

    Examples
    --------

    >>> from datetime import datetime, timezone
    >>> from reflex_langgraph.ui.reducer import history_group
    >>> now = datetime(2026, 10, 14, 1, 0, tzinfo=timezone.utc)
    >>> history_group('2026-10-13T20:00:00+00:00', now)
    'Yesterday'
    >>> history_group('2026-10-13T20:00:00+00:00', now, tz_offset_minutes=-540)  # Seoul
    'Today'
    >>> history_group('2026-09-01T09:00:00+00:00', now)
    'Older'
    """
    try:
        then = datetime.fromisoformat(updated_at)
    except (TypeError, ValueError):
        return "Older"
    if then.tzinfo is None:
        then = then.replace(tzinfo=timezone.utc)
    shift = timedelta(minutes=tz_offset_minutes)
    now = now or datetime.now(timezone.utc)
    days = ((now - shift).date() - (then - shift).date()).days
    if days <= 0:
        return "Today"
    if days == 1:
        return "Yesterday"
    if days < 7:
        return "Previous 7 days"
    if days < 30:
        return "Previous 30 days"
    return "Older"


def _show_agent(items: List[ChatItem], agent: str) -> bool:
    """Name an agent unless it is the one the user talks to: the first that spoke."""
    main = next((i.agent for i in items if i.agent and i.kind in ("assistant", "tool")), "")
    return bool(agent and main and agent != main)


def _end_thinking(items: List[ChatItem], now: float) -> None:
    for item in items:
        if item.thinking:
            item.thinking = False
            if item.reasoning_started:
                item.thought_seconds = max(1, round(now - item.reasoning_started))


def _blocked_item(items: List[ChatItem], agent: str, message_id: str, info: dict) -> ChatItem:
    # Right under the same message, the agent's label is already shown.
    if items and items[-1].kind == "assistant" and items[-1].message_id == message_id:
        agent = ""
    return ChatItem(kind="blocked", agent=agent, message_id=message_id, text=blocked_text(info),
                    show_agent=_show_agent(items, agent))


def _assistant(items: List[ChatItem], data: dict) -> ChatItem:
    """The assistant item for ``data["message_id"]``, created on its first event."""
    item = _find(items, kind="assistant", message_id=data.get("message_id") or "")
    if item is None:
        agent = data.get("agent") or ""
        item = ChatItem(kind="assistant", agent=agent, message_id=data.get("message_id") or "",
                        show_agent=_show_agent(items, agent))
        items.append(item)
    return item


def apply_event(items: List[ChatItem], pending: List[ApprovalView], event: str,
                data: dict, now: Optional[float] = None) -> Optional[str]:
    """
    Apply one event to the chat.

    Parameters
    ----------
    items : list of ChatItem
        The chat's items; changed in place.
    pending : list of ApprovalView
        The approval cards; changed in place.
    event : str
        The event's type.
    data : dict
        The event's data.
    now : float, optional
        The time (epoch seconds), to measure reasoning. By default, now.

    Returns
    -------
    str or None
        The run's status when the event ends it, else ``None``.

    Examples
    --------

    >>> from reflex_langgraph.ui.reducer import apply_event
    >>> items, pending = [], []
    >>> apply_event(items, pending, 'reasoning', {'message_id': 'm1', 'text': 'Plan.'}, now=100)
    >>> items[0].thinking
    True
    >>> apply_event(items, pending, 'token', {'message_id': 'm1', 'text': 'Hel'}, now=112)
    >>> apply_event(items, pending, 'token', {'message_id': 'm1', 'text': 'lo'}, now=113)
    >>> [(i.kind, i.reasoning, i.text, i.thinking, i.thought_seconds) for i in items]
    [('assistant', 'Plan.', 'Hello', False, 12)]
    >>> apply_event(items, pending, 'run_finished', {'status': 'completed'})
    'completed'
    """
    now = time.time() if now is None else now
    if event in ("token", "message", "tool_call", "approval_required", "blocked", "error",
                 "run_finished"):
        _end_thinking(items, now)   # the answer started, or the model call ended
    if event == "token":
        _assistant(items, data).text += data["text"]
    elif event == "reasoning":   # streams before the answer, into the same item
        item = _assistant(items, data)
        if not item.reasoning:
            item.reasoning_started = now
        item.reasoning += data["text"]
        item.thinking = not item.text
    elif event == "message":
        item = _assistant(items, data)
        if data.get("agent") and data["agent"] != item.agent:   # the complete reply knows best
            item.agent = data["agent"]
            item.show_agent = _show_agent([i for i in items if i is not item], item.agent)
        item.text = data["text"]
        if data.get("reasoning"):
            item.reasoning = data["reasoning"]
    elif event == "blocked":
        mid = data.get("message_id") or ""
        if not (mid and _find(items, kind="blocked", message_id=mid)):
            items.append(_blocked_item(items, data.get("agent") or "", mid, data))
    elif event == "tool_call":
        # A resume after approval re-sends the same call (possibly with edited args): update it.
        item = _find(items, kind="tool", call_id=data["call_id"])
        if item is None:
            agent = data.get("agent") or ""
            items.append(ChatItem(kind="tool", agent=agent, show_agent=_show_agent(items, agent),
                                  call_id=data["call_id"], tool=data["tool"],
                                  args=pretty(data.get("args", {})),
                                  delegate_to=data.get("delegate_to") or "", status="running"))
        else:
            item.args, item.status = pretty(data.get("args", {})), "running"
    elif event == "tool_result":
        item = _find(items, kind="tool", call_id=data["call_id"])
        if item is None:
            item = ChatItem(kind="tool", agent=data.get("agent") or "", call_id=data["call_id"],
                            tool=data.get("tool", ""))
            items.append(item)
        item.output = data.get("output", "") + ("\n… (truncated)" if data.get("truncated") else "")
        item.status = "done"
    elif event == "approval_required":
        if not any(p.interrupt_id == data["interrupt_id"] for p in pending):
            pending.append(approval_from(data))
    elif event == "error":
        items.append(ChatItem(kind="error", text=data.get("message", "error")))
    elif event == "run_finished":
        status = data.get("status", "completed")
        if status == "cancelled":
            items.append(ChatItem(kind="notice", text="Stopped."))
        for item in items:
            if item.kind == "tool" and item.status == "running" and status != "interrupted":
                item.status = "done"
        return status
    return None


def drop_reply(items: List[ChatItem]) -> List[ChatItem]:
    """
    Return the chat up to its last user message, before that message is answered again.

    Parameters
    ----------
    items : list of ChatItem
        The chat's items.

    Returns
    -------
    list of ChatItem
        The items up to and including the last user message; all of them when there is
        none.

    Examples
    --------

    >>> from reflex_langgraph.ui.reducer import ChatItem, drop_reply
    >>> chat = [ChatItem(kind='user', text='hi'), ChatItem(kind='assistant', text='Hello.'),
    ...         ChatItem(kind='notice', text='Stopped.')]
    >>> [i.text for i in drop_reply(chat)]
    ['hi']
    """
    last = max((n for n, i in enumerate(items) if i.kind == "user"), default=None)
    return list(items) if last is None else items[:last + 1]


def items_from_history(messages: List[dict]) -> List[ChatItem]:
    """
    Make the chat's items from a conversation's history.

    Parameters
    ----------
    messages : list of dict
        The history's messages (see
        :class:`~reflex_langgraph.server.schemas.HistoryMessage`).

    Returns
    -------
    list of ChatItem
        The items.
    """
    items: List[ChatItem] = []
    for m in messages:
        role = m.get("role")
        if role == "user":
            items.append(ChatItem(kind="user", text=m.get("text", ""),
                                  files=[f["name"] for f in m.get("files") or []]))
        elif role == "assistant":
            if m.get("text") or m.get("reasoning"):
                items.append(ChatItem(kind="assistant", agent=m.get("agent") or "",
                                      message_id=m.get("id") or "", text=m.get("text") or "",
                                      reasoning=m.get("reasoning") or ""))
            if m.get("blocked"):
                items.append(_blocked_item(items, m.get("agent") or "", m.get("id") or "",
                                           m["blocked"]))
            for c in m.get("tool_calls", []):
                tool = c["tool"]
                items.append(ChatItem(kind="tool", agent=m.get("agent") or "", call_id=c["id"],
                                      tool=tool, args=pretty(c.get("args", {})), status="running",
                                      delegate_to=tool[4:] if tool.startswith("ask_") else ""))
        elif role == "tool":
            item = _find(items, kind="tool", call_id=m.get("call_id", ""))
            if item is not None:
                item.output, item.status = m.get("text", ""), "done"
    return items


def build_decisions(approval: ApprovalView, decision: str, reject_note: str = ""):
    """
    Make the decisions that answer an approval card.

    Parameters
    ----------
    approval : ApprovalView
        The card.
    decision : str
        ``approve``, ``edit`` or ``reject``.
    reject_note : str, default: ''
        Shown to the agent when rejecting.

    Returns
    -------
    (list of dict or None, str)
        The decisions and an empty error, or ``None`` and the error (e.g. edited
        arguments that are not valid JSON).
    """
    decisions = []
    for req in approval.requests:
        if decision == "edit":
            try:
                args = json.loads(req.edited_args)
            except json.JSONDecodeError as exc:
                return None, f"Edited arguments for {req.tool} are not valid JSON: {exc}"
            decisions.append({"type": "edit", "edited_action": {"name": req.tool, "args": args}})
        elif decision == "reject":
            decisions.append({"type": "reject", "message": reject_note or "Rejected by the reviewer."})
        else:
            decisions.append({"type": decision})
    return decisions, ""
