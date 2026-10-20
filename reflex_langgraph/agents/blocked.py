"""
The blocked submodule recognizes replies the model provider blocked, and keeps them out of
later model calls.

When a provider's safety systems intervene, the reply arrives as a normal response with a
marker, not as an exception:

* Gemini: ``prompt_feedback.block_reason`` (the prompt was refused; no reply at all), or a
  ``finish_reason`` such as ``SAFETY``, ``PROHIBITED_CONTENT`` or ``RECITATION`` (the reply
  was withheld; "the content that was blocked is not returned");
* Anthropic: ``stop_reason == "refusal"``, with ``stop_details`` (category, explanation);
* OpenAI: ``finish_reason == "content_filter"``, or a ``refusal`` message.

:class:`BlockedResponseMiddleware`, on every agent, records what it finds on the reply as
``response_metadata["rlg_blocked"]`` (``reason``, ``scope``, ``message``); the server
streams it as a ``blocked`` event and history keeps it. The middleware also leaves blocked
turns out of later model calls: sending the refused content again gets the conversation
refused again (Anthropic: "you must reset the conversation context before continuing").
The chat itself still shows the turn, with the block notice.
"""

from __future__ import annotations

from dataclasses import replace
from enum import Enum
from typing import List, Optional

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

#: Where the middleware records a block on the reply's ``response_metadata``.
BLOCK_KEY = "rlg_blocked"

#: Gemini ``finish_reason`` values that mean the reply was withheld by a filter or policy.
GEMINI_BLOCKED = frozenset({
    "SAFETY", "RECITATION", "LANGUAGE", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII",
    "IMAGE_SAFETY", "IMAGE_PROHIBITED_CONTENT", "IMAGE_RECITATION", "ESCALATION",
    "PUP_LIMITED_DISABLED",
})


def _name(value) -> Optional[str]:
    """An enum-ish provider value as an upper-case name; ``None`` for unset/unspecified."""
    if isinstance(value, Enum):
        value = value.value
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return None if value == 0 else f"UNKNOWN_{value}"
    name = str(value).strip().upper()
    if not name or name.endswith("_UNSPECIFIED"):
        return None
    return name


def _info(reason: str, scope: str, message=None) -> dict:
    return {"reason": reason, "scope": scope,
            "message": message.strip() if isinstance(message, str) and message.strip() else None}


def detect_block(message: AIMessage) -> Optional[dict]:
    """
    Read a provider's block signal from a reply.

    Parameters
    ----------
    message : AIMessage
        The model's reply.

    Returns
    -------
    dict or None
        ``reason`` (the provider's code, e.g. ``SAFETY``), ``scope`` (``prompt`` when the
        request was refused, ``response`` when the reply was withheld) and ``message``
        (the provider's explanation, if any); ``None`` when it was not blocked.

    Examples
    --------

    >>> from langchain_core.messages import AIMessage
    >>> from reflex_langgraph.agents.blocked import detect_block
    >>> detect_block(AIMessage('', response_metadata={'finish_reason': 'SAFETY'}))
    {'reason': 'SAFETY', 'scope': 'response', 'message': None}
    >>> detect_block(AIMessage('', response_metadata={'stop_reason': 'refusal'}))
    {'reason': 'refusal', 'scope': 'response', 'message': None}
    >>> detect_block(AIMessage('Hello.', response_metadata={'finish_reason': 'STOP'})) is None
    True
    """
    meta = message.response_metadata or {}
    feedback = meta.get("prompt_feedback")
    if isinstance(feedback, dict):
        reason = _name(feedback.get("block_reason"))
        if reason:
            return _info(reason, "prompt", feedback.get("block_reason_message"))
    finish = _name(meta.get("finish_reason"))
    if finish in GEMINI_BLOCKED:
        return _info(finish, "response")
    if finish == "CONTENT_FILTER":
        return _info("content_filter", "response")
    if meta.get("stop_reason") == "refusal":
        details = meta.get("stop_details")
        details = details if isinstance(details, dict) else {}
        return _info(details.get("category") or "refusal", "response", details.get("explanation"))
    refusal = message.additional_kwargs.get("refusal")
    if isinstance(refusal, str) and refusal.strip():
        return _info("refusal", "response", refusal)
    return None


def block_info(message) -> Optional[dict]:
    """
    Return how the provider blocked a message, if it did.

    Works for replies the middleware marked and for raw provider replies.

    Parameters
    ----------
    message : BaseMessage
        Any message; only replies (``AIMessage``) can be blocked.

    Returns
    -------
    dict or None
        As in :func:`detect_block`.
    """
    if not isinstance(message, AIMessage):
        return None
    marked = (message.response_metadata or {}).get(BLOCK_KEY)
    return marked if isinstance(marked, dict) else detect_block(message)


def without_blocked_turns(messages: List[BaseMessage]) -> List[BaseMessage]:
    """
    Drop each blocked turn from a conversation.

    A turn runs from the user message that started it to the blocked reply. Whole turns
    go, so tool calls keep their results and the user/model alternation holds.

    Parameters
    ----------
    messages : list of BaseMessage
        The conversation.

    Returns
    -------
    list of BaseMessage
        The conversation without blocked turns (the same list when there are none).

    Examples
    --------

    >>> from langchain_core.messages import AIMessage, HumanMessage
    >>> from reflex_langgraph.agents.blocked import without_blocked_turns
    >>> blocked = AIMessage('', response_metadata={'finish_reason': 'SAFETY'})
    >>> kept = without_blocked_turns([HumanMessage('hi'), AIMessage('Hello.'),
    ...                               HumanMessage('...'), blocked, HumanMessage('next')])
    >>> [m.text for m in kept]
    ['hi', 'Hello.', 'next']
    """
    drop = set()
    for i, message in enumerate(messages):
        if block_info(message):
            start = i
            while start > 0 and not isinstance(messages[start], HumanMessage):
                start -= 1
            drop.update(range(start, i + 1))
    return [m for j, m in enumerate(messages) if j not in drop] if drop else messages


class BlockedResponseMiddleware(AgentMiddleware):
    """
    Agent middleware for replies the model provider blocked.

    It marks a blocked reply, and keeps blocked turns out of later model calls.
    :func:`~reflex_langgraph.agents.build_graph` adds it to every agent. A marked reply
    carries the block under ``response_metadata["rlg_blocked"]`` (see
    :func:`detect_block`).
    """

    def _clean(self, request):
        messages = without_blocked_turns(request.messages)
        return request if messages is request.messages else request.override(messages=messages)

    @staticmethod
    def _mark(response):
        def mark(message):
            info = detect_block(message) if isinstance(message, AIMessage) else None
            if info is None:
                return message
            return message.model_copy(update={
                "response_metadata": {**message.response_metadata, BLOCK_KEY: info}})

        if isinstance(response, AIMessage):
            return mark(response)
        result = getattr(response, "result", None)
        if isinstance(result, list):
            return replace(response, result=[mark(m) for m in result])
        return response

    def wrap_model_call(self, request, handler):
        """Call the model without blocked turns, and mark its reply if it is blocked."""
        return self._mark(handler(self._clean(request)))

    async def awrap_model_call(self, request, handler):
        """Async form of :meth:`wrap_model_call`."""
        return self._mark(await handler(self._clean(request)))
