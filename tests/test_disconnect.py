"""An agent run belongs to the server, not to the HTTP connection that started it.

Runs a real uvicorn server: the in-process ASGI transport buffers whole responses, so it
cannot drop a connection halfway through a stream.
"""

import asyncio
import socket
import threading
import time

import httpx
import pytest
import uvicorn

from reflex_langgraph import ToolRegistry, build_graph, create_server, load_config
from scripted import ScriptedChatModel, call

CONFIG = {"model": "x:y", "agents": [{"name": "a", "tools": ["slow"]}]}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def server():
    """Yield (base_url, finished) for an API whose one tool takes ~1.5 s."""
    finished = threading.Event()
    registry = ToolRegistry()

    async def slow(seconds: float) -> str:
        """Wait, then report."""
        await asyncio.sleep(seconds)
        finished.set()
        return "slept"

    registry.add(slow, name="slow", description="Wait, then report.")
    graph = build_graph(load_config(CONFIG), registry, models={
        "a": ScriptedChatModel(script=[call("slow", seconds=1.5), "Done after the wait."])})
    port = _free_port()
    srv = uvicorn.Server(uvicorn.Config(create_server(graph), host="127.0.0.1", port=port,
                                        log_level="warning"))
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not srv.started and time.time() < deadline:
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}", finished
    srv.should_exit = True
    thread.join(timeout=10)


def _history(base, tid):
    return httpx.get(f"{base}/threads/{tid}", timeout=5).json()


def test_run_survives_client_disconnect(server):
    base, finished = server
    tid = httpx.post(f"{base}/threads").json()["thread_id"]

    # Start a run and hang up as soon as the tool call is announced.
    with httpx.Client(timeout=10) as c:
        with c.stream("POST", f"{base}/threads/{tid}/messages", json={"content": "go"}) as r:
            for line in r.iter_lines():
                if line.startswith("event: tool_call"):
                    break
    assert _history(base, tid)["running"] is True

    # The run keeps going without anyone listening, and finishes.
    assert finished.wait(10), "the tool was cancelled with the connection"
    deadline = time.time() + 10
    while _history(base, tid)["running"] and time.time() < deadline:
        time.sleep(0.1)
    hist = _history(base, tid)
    assert hist["running"] is False
    assert hist["messages"][-1]["text"] == "Done after the wait."


def test_cancel_route_still_stops_a_run(server):
    base, finished = server
    tid = httpx.post(f"{base}/threads").json()["thread_id"]
    events = []

    def listen():
        with httpx.stream("POST", f"{base}/threads/{tid}/messages", json={"content": "go"},
                          timeout=10) as r:
            events.extend(line for line in r.iter_lines() if line.startswith("event:"))

    t = threading.Thread(target=listen)
    t.start()
    deadline = time.time() + 5
    while not _history(base, tid)["running"] and time.time() < deadline:
        time.sleep(0.05)
    time.sleep(0.3)
    assert httpx.post(f"{base}/threads/{tid}/cancel").json() == {"cancelled": True}
    t.join(timeout=10)
    assert "event: run_finished" in events
    assert not finished.wait(2), "the tool kept running after cancel"
    assert _history(base, tid)["running"] is False


def test_delete_stops_the_run_and_removes_the_conversation(server):
    base, finished = server
    tid = httpx.post(f"{base}/threads").json()["thread_id"]
    events = []

    def listen():
        with httpx.stream("POST", f"{base}/threads/{tid}/messages", json={"content": "go"},
                          timeout=10) as r:
            events.extend(line for line in r.iter_lines() if line.startswith("data:"))

    t = threading.Thread(target=listen)
    t.start()
    deadline = time.time() + 5
    while not _history(base, tid)["running"] and time.time() < deadline:
        time.sleep(0.05)
    time.sleep(0.3)  # the tool is waiting now
    assert httpx.delete(f"{base}/threads/{tid}", timeout=10).json() == {
        "thread_id": tid, "deleted": True}
    t.join(timeout=10)
    assert '"status": "cancelled"' in events[-1]       # the watcher saw the run stop
    assert not finished.wait(2), "the tool kept running after the delete"
    assert httpx.get(f"{base}/threads/{tid}").status_code == 404
    assert httpx.get(f"{base}/threads").json() == []
