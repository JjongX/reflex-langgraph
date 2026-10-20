"""An app adds its own agent middleware with ``build_graph(middleware=...)``."""

import pytest
from langchain.agents.middleware import AgentMiddleware

from reflex_langgraph import LocalFileStore, build_graph, create_server, load_config

from conftest import client_for, sse_events
from scripted import ScriptedChatModel, call

pytestmark = pytest.mark.anyio

TEAM = {"model": {"id": "x:y", "reads": ["image"]}, "entry": "boss", "agents": [
    {"name": "boss", "workers": ["helper"]}, {"name": "helper", "description": "Helps."}]}


class Recorder(AgentMiddleware):
    """Records which agent's model was called, and what it was sent last."""

    def __init__(self, label, seen):
        super().__init__()
        self.label, self.seen = label, seen

    def wrap_model_call(self, request, handler):
        self.seen.append((self.label, request.messages[-1].content))
        return handler(request)

    async def awrap_model_call(self, request, handler):
        self.seen.append((self.label, request.messages[-1].content))
        return await handler(request)


def team(middleware):
    models = {"boss": ScriptedChatModel(script=[call("ask_helper", task="t"), "Done."]),
              "helper": ScriptedChatModel(script=["Helped."])}
    return build_graph(load_config(TEAM), models=models, middleware=middleware)


async def run(graph):
    async with client_for(create_server(graph)) as c:
        tid = (await c.post("/threads")).json()["thread_id"]
        return await sse_events(c, f"/threads/{tid}/messages", {"content": "go"})


async def test_one_list_for_every_agent_a_dict_for_some_or_a_factory():
    seen = []
    await run(team([Recorder("all", seen)]))
    assert [label for label, _ in seen] == ["all", "all", "all"]      # boss, helper, boss

    seen.clear()
    await run(team({"helper": [Recorder("helper only", seen)]}))
    assert [label for label, _ in seen] == ["helper only"]

    seen.clear()
    await run(team(lambda spec: [Recorder(spec.name, seen)]))        # fresh per agent
    assert [label for label, _ in seen] == ["boss", "helper", "boss"]

    with pytest.raises(ValueError, match="unknown agents"):
        team({"nobody": [Recorder("x", seen)]})


async def test_app_middleware_sees_the_attached_files(tmp_path):
    seen = []
    graph = build_graph(load_config({"model": {"id": "x:y", "reads": ["image"]},
                                     "agents": [{"name": "a"}]}),
                        models={"a": ScriptedChatModel(script=["Ok."])},
                        middleware=[Recorder("app", seen)])
    app = create_server(graph, files=LocalFileStore(tmp_path))
    async with client_for(app) as c:
        tid = (await c.post("/threads")).json()["thread_id"]
        png = (await c.post(f"/threads/{tid}/files", files={
            "file": ("chart.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)})).json()
        await sse_events(c, f"/threads/{tid}/messages", {"content": "Look", "files": [png["file_id"]]})
    content = seen[0][1]
    assert [b["type"] for b in content] == ["text", "image"]          # already expanded
