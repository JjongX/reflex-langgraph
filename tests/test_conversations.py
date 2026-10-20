"""Managing conversations like in ChatGPT: rename, delete and search them, and answer the
last message again."""

import asyncio
import sqlite3

import httpx
import pytest
from langchain_core.runnables import RunnableConfig

from reflex_langgraph import (SqliteThreadStore, ToolRegistry, build_graph, create_server,
                              load_config, sqlite_checkpointer)
from reflex_langgraph.server import MemoryThreadStore, ThreadRecord
from reflex_langgraph.ui import BackendClient

from conftest import client_for, collect, sse_events
from scripted import ScriptedChatModel, call

pytestmark = pytest.mark.anyio

TEAM = """
model: x:y
entry: supervisor
agents:
  - name: supervisor
    workers: [operator]
  - name: operator
    description: Starts jobs.
    tools: [submit]
    approve:
      submit: [approve, reject]
"""


def chat_app(script, **server_kw):
    graph = build_graph(load_config({"model": "x:y", "agents": [{"name": "a"}]}),
                        models={"a": ScriptedChatModel(script=script)})
    return create_server(graph, **server_kw), graph


async def say(c, tid, text, headers=None):
    return await sse_events(c, f"/threads/{tid}/messages", {"content": text}, headers)


async def new_thread(c, headers=None):
    return (await c.post("/threads", headers=headers or {})).json()["thread_id"]


# --- rename ------------------------------------------------------------------------------

async def test_rename_keeps_the_conversation_in_its_place():
    app, _ = chat_app(["One.", "Two.", "Three."])
    async with client_for(app) as c:
        t1 = await new_thread(c)
        await say(c, t1, "first chat")
        t2 = await new_thread(c)
        await say(c, t2, "second chat")

        resp = await c.patch(f"/threads/{t1}", json={"title": "  Sequencing plan  "})
        assert resp.status_code == 200 and resp.json()["title"] == "Sequencing plan"
        rows = (await c.get("/threads")).json()
        assert [(r["thread_id"], r["title"]) for r in rows] == [(t2, "second chat"),
                                                                 (t1, "Sequencing plan")]
        await say(c, t1, "a later message")              # the new title stays
        assert (await c.get(f"/threads/{t1}")).json()["title"] == "Sequencing plan"

        for bad in ("   ", "x" * 201):
            assert (await c.patch(f"/threads/{t1}", json={"title": bad})).status_code == 422
        assert (await c.patch("/threads/nope", json={"title": "x"})).status_code == 404


async def test_only_who_may_see_a_conversation_renames_or_deletes_it():
    alice, bob = {"x-user": "alice"}, {"x-user": "bob"}
    app, _ = chat_app(["Hi."], auth=lambda req: {"id": req.headers.get("x-user", "anon")})
    async with client_for(app) as c:
        tid = await new_thread(c, alice)
        await say(c, tid, "alice's plan", alice)
        assert (await c.patch(f"/threads/{tid}", json={"title": "mine"}, headers=bob)).status_code == 404
        assert (await c.delete(f"/threads/{tid}", headers=bob)).status_code == 404
        assert (await c.get("/threads", params={"q": "plan"}, headers=bob)).json() == []
        assert (await c.delete(f"/threads/{tid}", headers=alice)).status_code == 200


# --- delete ------------------------------------------------------------------------------

async def test_delete_removes_the_conversation_and_its_graph_state():
    app, graph = chat_app(["Hello."])
    async with client_for(app) as c:
        tid = await new_thread(c)
        await say(c, tid, "hi")
        resp = await c.delete(f"/threads/{tid}")
        assert resp.json() == {"thread_id": tid, "deleted": True}

        assert (await c.get("/threads")).json() == []
        assert (await c.get(f"/threads/{tid}")).status_code == 404
        assert (await c.post(f"/threads/{tid}/messages", json={"content": "x"})).status_code == 404
        assert (await c.delete(f"/threads/{tid}")).status_code == 404
    state = await graph.aget_state({"configurable": {"thread_id": tid}})
    assert state.values == {}


def store_of(kind, tmp_path):
    return MemoryThreadStore() if kind == "memory" else SqliteThreadStore(tmp_path / "t.db")


@pytest.mark.parametrize("kind", ["memory", "sqlite"])
async def test_a_rename_while_a_reply_runs_survives_the_end_of_the_run(kind, tmp_path):
    # The run holds the record it read when it started; saving its end must not write that
    # copy's old title back.
    started, release = asyncio.Event(), asyncio.Event()
    registry = ToolRegistry()

    async def hold() -> str:
        """Wait until released."""
        started.set()
        await release.wait()
        return "done"

    registry.add(hold, name="hold", description="Wait until released.")
    graph = build_graph(load_config({"model": "x:y", "agents": [{"name": "a", "tools": ["hold"]}]}),
                        registry, models={"a": ScriptedChatModel(script=[call("hold"), "Done."])})
    app = create_server(graph, threads=store_of(kind, tmp_path))
    async with client_for(app) as c:
        tid = await new_thread(c)
        run = asyncio.create_task(say(c, tid, "Original title"))
        await asyncio.wait_for(started.wait(), 10)
        renamed = await c.patch(f"/threads/{tid}", json={"title": "Renamed during run"})
        assert renamed.status_code == 200
        release.set()
        assert (await run)[-1][1]["status"] == "completed"
        assert [r["title"] for r in (await c.get("/threads")).json()] == ["Renamed during run"]


@pytest.mark.parametrize("kind", ["memory", "sqlite"])
def test_stores_save_only_the_fields_named(kind, tmp_path):
    store = store_of(kind, tmp_path)
    store.create(ThreadRecord(id="t1", owner=None, created_at="2026-10-15T09:00:00", title="A"))
    stale = ThreadRecord(id="t1", owner=None, created_at="2026-10-15T09:00:00", title="A")
    fresh = store.get("t1")
    fresh.title = "B"
    store.update(fresh, ["title"])
    stale.pending = True
    store.update(stale, ["pending"])             # the old title is not written back
    assert (store.get("t1").title, store.get("t1").pending) == ("B", True)


@pytest.mark.parametrize("kind", ["memory", "sqlite"])
def test_stores_delete_and_never_bring_a_deleted_record_back(kind, tmp_path):
    store = MemoryThreadStore() if kind == "memory" else SqliteThreadStore(tmp_path / "t.db")
    record = ThreadRecord(id="t1", owner=None, created_at="2026-10-14T09:00:00", title="A")
    store.create(record)
    record.title = "B"
    store.update(record)
    assert store.get("t1").title == "B"
    store.delete("t1")
    store.update(record)            # e.g. a run that ended after the delete
    assert store.get("t1") is None and store.list() == []
    store.delete("t1")              # unknown: nothing happens


# --- search ------------------------------------------------------------------------------

async def test_search_matches_titles_and_messages_ignoring_case():
    app, _ = chat_app(["Use lane 3 for the new samples.", "About 4,000 USD."])
    async with client_for(app) as c:
        t1 = await new_thread(c)
        await say(c, t1, "Plan the sequencing run")
        t2 = await new_thread(c)
        await say(c, t2, "Budget for October")
        await new_thread(c)                               # no message yet: never a result

        async def search(q):
            return [(r["thread_id"], r["snippet"])
                    for r in (await c.get("/threads", params={"q": q})).json()]

        assert await search("budget") == [(t2, "")]      # in the title: no snippet
        assert await search("new samples") == [(t1, "Use lane 3 for the new samples.")]
        assert await search("LANE 3") == await search("new samples")
        assert await search("nothing like it") == []
        assert len(await search("")) == 3                 # no query: the plain list


def test_snippets_are_cut_around_the_match():
    from reflex_langgraph.server.app import _snippet
    text = "word " * 40 + "NEEDLE" + " tail" * 40
    snippet = _snippet(text, text.index("NEEDLE"), 6)
    assert snippet.startswith("…") and snippet.endswith("…") and "NEEDLE" in snippet
    assert len(snippet) < 140 and "\n" not in _snippet("a\nb", 0, 1)


# --- regenerate --------------------------------------------------------------------------

async def test_regenerate_replaces_the_last_reply():
    app, _ = chat_app(["First.", "Second.", "Third.", "Fourth."])
    async with client_for(app) as c:
        tid = await new_thread(c)
        assert (await c.post(f"/threads/{tid}/regenerate")).status_code == 409   # nothing yet

        await say(c, tid, "hi")
        events = await sse_events(c, f"/threads/{tid}/regenerate", {})
        assert [d["text"] for e, d in events if e == "message"] == ["Second."]
        assert events[-1][1]["status"] == "completed"

        await say(c, tid, "more")                         # goes on from the new reply
        await sse_events(c, f"/threads/{tid}/regenerate", {})
        history = (await c.get(f"/threads/{tid}")).json()
        assert [m["text"] for m in history["messages"]] == ["hi", "Second.", "more", "Fourth."]


async def test_regenerate_drops_a_pending_approval_and_asks_again():
    registry, ran = ToolRegistry(), []

    @registry.tool
    def submit(job: str, config: RunnableConfig) -> str:
        """Submit a job."""
        ran.append(job)
        return f"submitted {job}"

    graph = build_graph(load_config(TEAM), registry, models={
        "supervisor": ScriptedChatModel(script=[call("ask_operator", task="run j1"),
                                                call("ask_operator", task="run j2"), "Done."]),
        "operator": ScriptedChatModel(script=[call("submit", job="j1"),
                                              call("submit", job="j2"), "Submitted j2."])})
    app = create_server(graph)
    async with client_for(app) as c:
        tid = await new_thread(c)
        first = await say(c, tid, "go")
        old = next(d for e, d in first if e == "approval_required")

        again = await sse_events(c, f"/threads/{tid}/regenerate", {})
        new = next(d for e, d in again if e == "approval_required")
        assert new["interrupt_id"] != old["interrupt_id"]
        assert new["requests"][0]["args"] == {"job": "j2"}
        stale = await c.post(f"/threads/{tid}/decisions", json={
            "interrupt_id": old["interrupt_id"], "decisions": [{"type": "approve"}]})
        assert stale.status_code == 409

        done = await sse_events(c, f"/threads/{tid}/decisions", {
            "interrupt_id": new["interrupt_id"], "decisions": [{"type": "approve"}]})
        assert done[-1][1]["status"] == "completed"
        texts = [m["text"] for m in (await c.get(f"/threads/{tid}")).json()["messages"]]
        assert texts[0] == "go" and texts[-1] == "Done."
    assert ran == ["j2"]                                  # the dropped request never ran


async def test_regenerate_and_delete_with_durable_storage(tmp_path):
    saver = sqlite_checkpointer(tmp_path / "graph.db")
    graph = build_graph(load_config({"model": "x:y", "agents": [{"name": "a"}]}),
                        checkpointer=saver,
                        models={"a": ScriptedChatModel(script=["First.", "Second."])})
    app = create_server(graph, threads=SqliteThreadStore(tmp_path / "threads.db"))
    client = BackendClient("http://t", transport=httpx.ASGITransport(app=app))
    try:
        tid = await client.new_thread()
        await collect(client.send(tid, "hi"))
        events = await collect(client.regenerate(tid))
        assert [d["text"] for e, d in events if e == "message"] == ["Second."]
        assert [m["text"] for m in (await client.history(tid))["messages"]] == ["hi", "Second."]

        renamed = await client.rename_thread(tid, "Greeting")
        assert renamed["title"] == "Greeting"
        assert [r["title"] for r in await client.threads(q="second")] == ["Greeting"]

        assert await client.delete_thread(tid) == {"thread_id": tid, "deleted": True}
        assert await client.threads() == []
    finally:
        await saver.aclose()
    with sqlite3.connect(tmp_path / "graph.db") as db:
        assert db.execute("SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?",
                          (tid,)).fetchone() == (0,)
