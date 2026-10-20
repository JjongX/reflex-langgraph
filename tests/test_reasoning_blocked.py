"""Reasoning (thinking summaries) and replies the model provider blocked."""

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.chat_models import generate_from_stream
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGenerationChunk

from reflex_langgraph.agents import ToolRegistry, build_graph
from reflex_langgraph.agents.blocked import BLOCK_KEY, detect_block, without_blocked_turns
from reflex_langgraph.config import load_config
from reflex_langgraph.server import create_server
from scripted import ScriptedChatModel, call, think
from reflex_langgraph.ui.reducer import apply_event, items_from_history

from conftest import client_for, sse_events

pytestmark = pytest.mark.anyio

ONE_AGENT = {"model": "x:y", "agents": [{"name": "a"}]}
TEAM = {"model": "x:y", "entry": "boss", "agents": [
    {"name": "boss", "workers": ["worker"]},
    {"name": "worker", "description": "Does the work."}]}


def app_for(config, **models):
    return create_server(build_graph(load_config(config), ToolRegistry(), models=models))


async def turns(app, *texts):
    """Send each text in one new thread; return (thread id, [events per turn], history)."""
    async with client_for(app) as c:
        tid = (await c.post("/threads")).json()["thread_id"]
        out = [await sse_events(c, f"/threads/{tid}/messages", {"content": t}) for t in texts]
        return tid, out, (await c.get(f"/threads/{tid}")).json()


def of(events, kind):
    return [d for e, d in events if e == kind]


def replay(events):
    items, pending = [], []
    for e, d in events:
        apply_event(items, pending, e, d)
    return items


# --- reasoning ---------------------------------------------------------------------------

async def test_reasoning_reaches_the_chat_and_stays_in_history():
    _, (events,), history = await turns(
        app_for(ONE_AGENT, a=ScriptedChatModel(script=[think("Plan it first.", "Done.")])), "go")
    assert [(m["text"], m["reasoning"]) for m in of(events, "message")] == [("Done.", "Plan it first.")]
    [item] = [i for i in replay(events) if i.kind == "assistant"]
    assert (item.text, item.reasoning) == ("Done.", "Plan it first.")
    last = history["messages"][-1]
    assert (last["text"], last["reasoning"]) == ("Done.", "Plan it first.")
    [item] = [i for i in items_from_history(history["messages"]) if i.kind == "assistant"]
    assert item.reasoning == "Plan it first."


class GeminiShaped(BaseChatModel):
    """Streams the way langchain-google-genai does with include_thoughts=True."""

    @property
    def _llm_type(self):
        return "gemini-shaped"

    def bind_tools(self, tools, **kwargs):
        return self

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        for part in ({"type": "thinking", "thinking": "First, "},
                     {"type": "thinking", "thinking": None, "signature": "c2ln"},  # no text
                     {"type": "thinking", "thinking": "check."},
                     {"type": "text", "text": "Answer."}):
            chunk = ChatGenerationChunk(message=AIMessageChunk(
                content=[part], response_metadata={"model_provider": "google_genai"}))
            if run_manager:
                run_manager.on_llm_new_token("", chunk=chunk)
            yield chunk

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return generate_from_stream(self._stream(messages, stop, run_manager, **kwargs))


async def test_streamed_reasoning_arrives_before_the_answer():
    _, (events,), _ = await turns(app_for(ONE_AGENT, a=GeminiShaped()), "go")
    kinds = [e for e, _ in events if e in ("reasoning", "token", "message")]
    assert kinds == ["reasoning", "reasoning", "token", "message"]
    assert "".join(d["text"] for d in of(events, "reasoning")) == "First, check."
    [item] = [i for i in replay(events) if i.kind == "assistant"]
    assert (item.reasoning, item.text) == ("First, check.", "Answer.")


async def test_a_worker_hands_back_its_answer_without_its_reasoning():
    app = app_for(TEAM, boss=ScriptedChatModel(script=[call("ask_worker", task="t"), "Ok."]),
                  worker=ScriptedChatModel(script=[think("private plan", "the answer")]))
    _, (events,), _ = await turns(app, "go")
    assert [d["output"] for d in of(events, "tool_result")] == ["the answer"]
    assert {(d["agent"], d["reasoning"]) for d in of(events, "message")} >= {("worker", "private plan")}


# --- blocked replies ---------------------------------------------------------------------

def test_provider_block_signals():
    from google.genai.types import BlockedReason

    def info(content="", **meta):
        extra = meta.pop("additional_kwargs", {})
        return detect_block(AIMessage(content=content, response_metadata=meta,
                                      additional_kwargs=extra))

    for not_blocked in ({"finish_reason": "STOP"}, {"finish_reason": "MAX_TOKENS"},
                        {"prompt_feedback": {"block_reason": 0}},
                        {"prompt_feedback": {"block_reason": None}},
                        {"prompt_feedback": {"block_reason": "BLOCKED_REASON_UNSPECIFIED"}},
                        {"stop_reason": "end_turn"}, {"finish_reason": "stop"}):
        assert info(**not_blocked) is None, not_blocked
    assert info(finish_reason="SAFETY") == {"reason": "SAFETY", "scope": "response", "message": None}
    assert info(prompt_feedback={"block_reason": BlockedReason.PROHIBITED_CONTENT,
                                 "block_reason_message": "Not allowed."}) == {
        "reason": "PROHIBITED_CONTENT", "scope": "prompt", "message": "Not allowed."}
    assert info(finish_reason="content_filter")["reason"] == "content_filter"       # OpenAI
    assert info(additional_kwargs={"refusal": "I can't help with that."})["message"] == \
        "I can't help with that."                                                   # OpenAI
    assert info(stop_reason="refusal", stop_details={"category": "cyber",           # Anthropic
                                                     "explanation": "Declined."}) == {
        "reason": "cyber", "scope": "response", "message": "Declined."}
    assert info(stop_reason="refusal", stop_details={"category": None})["reason"] == "refusal"


async def test_a_blocked_reply_is_marked_shown_and_kept_in_history():
    # Thinking first, then the safety filter cut the reply: still a block, not an answer.
    blocked = AIMessage(content=[{"type": "reasoning", "reasoning": "Considering..."}],
                        response_metadata={"finish_reason": "SAFETY"})
    _, (events,), history = await turns(
        app_for(ONE_AGENT, a=ScriptedChatModel(script=[blocked])), "go")
    [blocked_event] = of(events, "blocked")
    assert {k: blocked_event[k] for k in ("agent", "reason", "scope", "message")} == {
        "agent": "a", "reason": "SAFETY", "scope": "response", "message": None}
    assert of(events, "run_finished")[0]["status"] == "completed"
    kinds = [i.kind for i in replay(events)]
    assert kinds[-1] == "blocked" and "blocked this response (SAFETY)" in replay(events)[-1].text
    last = history["messages"][-1]
    assert last["blocked"] == {"reason": "SAFETY", "scope": "response", "message": None}
    assert [i.kind for i in items_from_history(history["messages"])][-1] == "blocked"


async def test_a_blocked_turn_is_left_out_of_later_model_calls():
    model = ScriptedChatModel(script=[
        "Hello.",
        AIMessage("", response_metadata={"prompt_feedback": {
            "block_reason": "PROHIBITED_CONTENT", "block_reason_message": "Not allowed."}}),
        "Fine.",
    ])
    _, (_, second, third), history = await turns(app_for(ONE_AGENT, a=model),
                                                 "hi", "the refused message", "next")
    [blocked_event] = of(second, "blocked")
    assert {k: blocked_event[k] for k in ("agent", "reason", "scope", "message")} == {
        "agent": "a", "reason": "PROHIBITED_CONTENT", "scope": "prompt", "message": "Not allowed."}
    assert "blocked your request (PROHIBITED_CONTENT). Not allowed." in replay(second)[-1].text
    # The model never sees the refused turn again; the user still does.
    assert [m.text for m in model.inputs[2]] == ["hi", "Hello.", "next"]
    assert "the refused message" in [m["text"] for m in history["messages"]]
    assert [d["text"] for d in of(third, "message")] == ["Fine."]


async def test_a_blocked_worker_reports_back_to_its_supervisor():
    refused = AIMessage("", response_metadata={
        "stop_reason": "refusal", "stop_details": {"category": "cyber", "explanation": "Declined."}})
    app = app_for(TEAM, boss=ScriptedChatModel(script=[call("ask_worker", task="t"),
                                                       "I can't help with that one."]),
                  worker=ScriptedChatModel(script=[refused]))
    _, (events,), _ = await turns(app, "go")
    [blocked] = of(events, "blocked")
    assert (blocked["agent"], blocked["reason"], blocked["message"]) == ("worker", "cyber", "Declined.")
    [output] = [d["output"] for d in of(events, "tool_result")]
    assert output.startswith("BLOCKED: the model provider blocked the worker agent's response (cyber)")
    assert of(events, "message")[-1]["text"] == "I can't help with that one."


def test_without_blocked_turns_drops_whole_turns():
    blocked = AIMessage("", response_metadata={BLOCK_KEY: {"reason": "SAFETY", "scope": "response"}})
    h0, a0, h1, h2 = (HumanMessage("h0"), AIMessage("a0"), HumanMessage("h1"), HumanMessage("h2"))
    tool_turn = AIMessage("", tool_calls=[{"name": "t", "args": {}, "id": "c1"}])
    result = ToolMessage("r", tool_call_id="c1")
    assert without_blocked_turns([h0, a0, h1, tool_turn, result, blocked, h2]) == [h0, a0, h2]
    unchanged = [h0, a0]
    assert without_blocked_turns(unchanged) is unchanged


async def test_a_workers_streamed_output_is_labelled_with_the_worker():
    # The supervisor's config carries its own agent name; the worker's chunks must not.
    app = app_for(TEAM, boss=ScriptedChatModel(script=[call("ask_worker", task="t"), "Done."]),
                  worker=GeminiShaped())
    _, (events,), _ = await turns(app, "go")
    streamed = {(e, d["agent"]) for e, d in events if e in ("reasoning", "token")}
    assert streamed == {("reasoning", "worker"), ("token", "worker")}
