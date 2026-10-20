"""A chat model that replays a script, so the tests run real agents without an API key.

``call()`` scripts a tool call, ``calls()`` several at once, ``think()`` reasoning before a
turn; a step can also be any ``AIMessage`` (e.g. one carrying a provider's block signal).
"""

import uuid
from typing import Any, Iterator, List, Sequence, Union

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ConfigDict, PrivateAttr

Step = Union[str, AIMessage]


def _call_id() -> str:
    return f"call_{uuid.uuid4().hex[:12]}"


def call(tool: str, **args: Any) -> AIMessage:
    """A turn that calls one tool."""
    return AIMessage(content="", tool_calls=[{"name": tool, "args": args, "id": _call_id()}])


def calls(*pairs: Sequence) -> AIMessage:
    """A turn that calls several tools at once: ``(tool, args)`` pairs."""
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": _call_id()}
                                             for name, args in pairs])


def think(reasoning: str, step: Step = "") -> AIMessage:
    """A turn with a standard ``reasoning`` block before it."""
    msg = AIMessage(content=step) if isinstance(step, str) else step.model_copy(deep=True)
    content = msg.content if isinstance(msg.content, list) else (
        [{"type": "text", "text": msg.content}] if msg.content else [])
    msg.content = [{"type": "reasoning", "reasoning": reasoning}, *content]
    return msg


class ScriptedChatModel(BaseChatModel):
    """Answers each call with the script's next step; raises once the script runs out.

    ``inputs`` holds the messages each call was given; ``calls_made`` counts the calls.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    script: List[Step]
    _script_iter: Iterator[Step] = PrivateAttr()
    _script_calls: int = PrivateAttr(default=0)
    _inputs: List[list] = PrivateAttr(default_factory=list)

    def model_post_init(self, __context: Any) -> None:
        self._script_iter = iter(self.script)

    @property
    def inputs(self) -> List[list]:
        return self._inputs

    @property
    def _llm_type(self) -> str:
        return "scripted"

    @property
    def calls_made(self) -> int:
        return self._script_calls

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self._inputs.append(list(messages))
        try:
            step = next(self._script_iter)
        except StopIteration:
            raise RuntimeError(f"ScriptedChatModel script exhausted after {self._script_calls} calls")
        self._script_calls += 1
        msg = AIMessage(content=step) if isinstance(step, str) else step.model_copy(deep=True)
        # fresh ids so a replayed message is never mistaken for the previous one
        msg.id = f"msg_{uuid.uuid4().hex[:12]}"
        return ChatResult(generations=[ChatGeneration(message=msg)])
