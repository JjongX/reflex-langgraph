
import pytest
from langchain_core.runnables import RunnableConfig

from reflex_langgraph.agents import ToolRegistry, build_graph
from reflex_langgraph.config import load_config
from reflex_langgraph.server import create_server
from scripted import ScriptedChatModel, call

from conftest import client_for, sse_events

pytestmark = pytest.mark.anyio

CONFIG = """
model: openai:unused
entry: supervisor
agents:
  - name: supervisor
    prompt: Coordinate.
    workers: [operator]
  - name: operator
    description: Starts jobs.
    tools: [submit]
    approve:
      submit: [approve, reject, edit]
"""


def make_app(supervisor_script, operator_script, seen=None, auth=None, **server_kw):
    registry = ToolRegistry()

    @registry.tool
    def submit(job: str, config: RunnableConfig) -> str:
        """Submit a job."""
        if seen is not None:
            seen.append({"job": job, "user": config["configurable"].get("user")})
        return f"submitted {job}"

    models = {"supervisor": ScriptedChatModel(script=supervisor_script),
              "operator": ScriptedChatModel(script=operator_script)}
    graph = build_graph(load_config(CONFIG), registry, models=models)
    return create_server(graph, auth=auth, **server_kw), models


# --- config -----------------------------------------------------------------------------

@pytest.mark.parametrize("bad, match", [
    ({"agents": []}, "at least one"),
    ({"model": "x:y", "agents": [{"name": "a", "workers": ["b"]}]}, "undefined workers"),
    ({"model": "x:y", "agents": [{"name": "a", "workers": ["b"]}, {"name": "b", "workers": ["a"]}]},
     "cycle"),
    ({"agents": [{"name": "a"}]}, "no model"),
    ({"model": "x:y", "agents": [{"name": "a", "approve": ["t"]}]}, "does not have"),
])
def test_config_rejects(bad, match):
    with pytest.raises(ValueError, match=match):
        load_config(bad)


def test_approve_shorthand():
    cfg = load_config({"model": "x:y", "agents": [{"name": "a", "tools": ["t"], "approve": ["t"]}]})
    assert cfg.agent("a").approve["t"] == ["approve", "edit", "reject", "respond"]
    assert cfg.entry == "a"


# --- end to end over HTTP ---------------------------------------------------------------

async def test_nested_approval_flow_records_approver():
    seen = []
    app, models = make_app(
        [call("ask_operator", task="run j1"), "All done."],
        [call("submit", job="j1"), "Submitted."],
        seen=seen, auth=lambda req: {"id": req.headers.get("x-user", "anon")},
    )
    async with client_for(app) as c:
        h = {"x-user": "alice"}
        tid = (await c.post("/threads", headers=h)).json()["thread_id"]

        first = await sse_events(c, f"/threads/{tid}/messages", {"content": "go"}, h)
        types = [e for e, _ in first]
        assert types[0] == "run_started" and types[-1] == "run_finished"
        assert first[-1][1]["status"] == "interrupted"
        delegation = next(d for e, d in first if e == "tool_call" and d["tool"] == "ask_operator")
        assert delegation["delegate_to"] == "operator" and delegation["agent"] == "supervisor"
        approval = next(d for e, d in first if e == "approval_required")
        assert approval["agent"] == "operator"
        assert approval["requests"][0]["tool"] == "submit"
        assert approval["requests"][0]["allowed"] == ["approve", "reject", "edit"]
        assert seen == []  # nothing ran before approval

        # history shows the pending approval; new messages are refused meanwhile
        hist = (await c.get(f"/threads/{tid}", headers=h)).json()
        assert hist["pending"][0]["interrupt_id"] == approval["interrupt_id"]
        assert (await c.post(f"/threads/{tid}/messages", json={"content": "x"}, headers=h)).status_code == 409

        # a decision the reviewer was not offered is refused
        bad = await c.post(f"/threads/{tid}/decisions", headers=h, json={
            "interrupt_id": approval["interrupt_id"], "decisions": [{"type": "respond"}]})
        assert bad.status_code == 422

        second = await sse_events(c, f"/threads/{tid}/decisions", {
            "interrupt_id": approval["interrupt_id"], "decisions": [{"type": "approve"}]}, h)
        assert second[-1][1]["status"] == "completed"
        assert any(e == "message" and d["text"] == "All done." and d["agent"] == "supervisor"
                   for e, d in second)

    assert seen == [{"job": "j1", "user": {"id": "alice"}}]
    # the worker resumed from its checkpoint rather than re-planning
    assert models["operator"].calls_made == 2


async def test_reject_does_not_run_tool():
    seen = []
    app, _ = make_app([call("ask_operator", task="run j1"), "Understood."],
                      [call("submit", job="j1"), "Not submitted."], seen=seen)
    async with client_for(app) as c:
        tid = (await c.post("/threads")).json()["thread_id"]
        first = await sse_events(c, f"/threads/{tid}/messages", {"content": "go"})
        iid = next(d for e, d in first if e == "approval_required")["interrupt_id"]
        second = await sse_events(c, f"/threads/{tid}/decisions", {
            "interrupt_id": iid, "decisions": [{"type": "reject", "message": "not today"}]})
        assert second[-1][1]["status"] == "completed"
    assert seen == []


async def test_edit_changes_tool_args():
    seen = []
    app, _ = make_app([call("ask_operator", task="run j1"), "Done."],
                      [call("submit", job="j1"), "Submitted."], seen=seen)
    async with client_for(app) as c:
        tid = (await c.post("/threads")).json()["thread_id"]
        first = await sse_events(c, f"/threads/{tid}/messages", {"content": "go"})
        iid = next(d for e, d in first if e == "approval_required")["interrupt_id"]
        await sse_events(c, f"/threads/{tid}/decisions", {"interrupt_id": iid, "decisions": [
            {"type": "edit", "edited_action": {"name": "submit", "args": {"job": "j2"}}}]})
    assert [s["job"] for s in seen] == ["j2"]


async def test_threads_are_private_to_their_owner():
    app, _ = make_app(["hi"], [], auth=lambda req: {"id": req.headers.get("x-user")} if req.headers.get("x-user") else None)
    async with client_for(app) as c:
        assert (await c.post("/threads")).status_code == 401
        tid = (await c.post("/threads", headers={"x-user": "alice"})).json()["thread_id"]
        assert (await c.get(f"/threads/{tid}", headers={"x-user": "bob"})).status_code == 404
        assert (await c.get("/threads", headers={"x-user": "bob"})).json() == []


@pytest.mark.parametrize("expose", [False, True])
async def test_run_error_reaches_client_as_event(expose, caplog):
    # operator script empty -> the run raises
    app, _ = make_app([call("ask_operator", task="x")], [], expose_errors=expose)
    async with client_for(app) as c:
        tid = (await c.post("/threads")).json()["thread_id"]
        events = await sse_events(c, f"/threads/{tid}/messages", {"content": "go"})
    kind, data = events[-1]
    assert kind == "error"
    if expose:
        assert "exhausted" in data["message"]
    else:  # users get a reference; the exception text stays in the server log
        assert "exhausted" not in data["message"] and "ref " in data["message"]
        ref = data["message"].split("ref ")[1].split(")")[0]
        assert any(ref in r.getMessage() and "exhausted" in r.exc_text for r in caplog.records
                   if r.exc_text)


async def test_registry_runs_async_functions():
    registry = ToolRegistry()

    async def fetch(x: int) -> str:
        """Fetch x."""
        return f"got {x}"

    registry.add(fetch)

    @registry.tool
    async def other(y: int) -> str:
        """Other."""
        return f"other {y}"

    assert await registry.get(["fetch"])[0].ainvoke({"x": 2}) == "got 2"
    assert await registry.get("other")[0].ainvoke({"y": 3}) == "other 3"   # one name
    with pytest.raises(KeyError, match=r"unregistered tools: \['nope'\]"):  # not ['n','o',...]
        registry.get("nope")


async def test_finished_runs_are_not_kept_in_memory():
    app, _ = make_app(["Hello."] * 4, [])
    async with client_for(app) as c:
        for _ in range(3):
            tid = (await c.post("/threads")).json()["thread_id"]
            events = await sse_events(c, f"/threads/{tid}/messages", {"content": "hi"})
            assert events[-1][1]["status"] == "completed"
        assert app.state.runtimes == {}
        # ... and a finished thread can run again
        events = await sse_events(c, f"/threads/{tid}/messages", {"content": "again"})
        assert events[-1][1]["status"] == "completed"


async def test_conversation_list_is_newest_first_with_titles():
    app, _ = make_app(["One.", "Two.", "Three."], [])
    async with client_for(app) as c:
        async def say(tid, text):
            await sse_events(c, f"/threads/{tid}/messages", {"content": text})

        t1 = (await c.post("/threads")).json()["thread_id"]
        await say(t1, "first chat\nmore lines")
        t2 = (await c.post("/threads")).json()["thread_id"]
        await say(t2, "second chat")

        rows = (await c.get("/threads")).json()
        assert [(r["thread_id"], r["title"]) for r in rows] == [(t2, "second chat"),
                                                                 (t1, "first chat")]
        assert all(r["updated_at"] >= r["created_at"] and not r["running"] for r in rows)

        await say(t1, "back to the first")          # activity moves it to the top
        assert [r["thread_id"] for r in (await c.get("/threads")).json()] == [t1, t2]
        assert len((await c.get("/threads", params={"limit": 1})).json()) == 1
        assert (await c.get("/threads", params={"limit": 0})).status_code == 422
