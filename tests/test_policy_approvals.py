"""Access policy, the cross-user approval queue, and persistence across a restart.

Cast: carol and dave are clients; rita is a reviewer (role "approver").
"""

import json

import httpx
import pytest
from pydantic import BaseModel

from reflex_langgraph import (AccessPolicy, JobRunner, JobSpec, RolePolicy, SqliteThreadStore,
                              ToolRegistry, build_graph, create_server, job_tools, jobs_router,
                              load_config, sqlite_checkpointer)
from scripted import ScriptedChatModel, call
from reflex_langgraph.ui.client import BackendClient

from conftest import collect, wait

pytestmark = pytest.mark.anyio

USERS = {"carol": {"id": "carol", "name": "Carol", "roles": []},
         "dave": {"id": "dave", "name": "Dave", "roles": []},
         "rita": {"id": "rita", "name": "Rita", "roles": ["approver"]}}
H = {name: {"x-user": name} for name in USERS}

CONFIG = {"model": "x:y", "entry": "operator", "agents": [{
    "name": "operator", "tools": ["echo_submit", "list_jobs", "job_status", "cancel_job"],
    "approve": {"echo_submit": ["approve", "edit", "reject"]}}]}


def header_auth(request):
    return USERS.get(request.headers.get("x-user", ""))


class EchoParams(BaseModel):
    text: str = "hi"
    sleep: float = 0
    fail_prepare: bool = False


class Echo(JobSpec):
    name = "echo"
    description = "Prints its text."
    params_model = EchoParams

    def prepare(self, p, job_dir):
        if p.fail_prepare:
            raise OSError(f"cannot link inputs into {job_dir}")  # a message with a server path

    def command(self, p, job_dir):
        return ["bash", "-c", f"sleep {p.sleep}; echo {p.text!r}"]

    def public_view(self, summary):
        return {k: v for k, v in summary.items() if k != "state"}


@pytest.fixture
def runner(tmp_path):
    return JobRunner(tmp_path / "jobs", [Echo()], max_concurrent=5, policy=RolePolicy("approver"))


def make_app(runner, script, **kw):
    graph = build_graph(load_config(CONFIG), ToolRegistry(job_tools(runner)),
                        models={"operator": ScriptedChatModel(script=script)},
                        checkpointer=kw.pop("checkpointer", None))
    return create_server(graph, auth=header_auth, policy=runner.policy,
                         routers=[jobs_router(runner)], **kw)


def client(app, user) -> BackendClient:
    return BackendClient("http://t", headers=H[user], transport=httpx.ASGITransport(app=app))


# --- the cross-user approval flow ------------------------------------------------------

async def test_client_requests_reviewer_approves(runner):
    app = make_app(runner, [call("echo_submit", job_id="c1", text="from carol"), "Started c1."])
    carol, dave, rita = (client(app, u) for u in ("carol", "dave", "rita"))

    tid = await carol.new_thread()
    events = await collect(carol.send(tid, "please run echo"))
    approval = next(d for e, d in events if e == "approval_required")
    assert approval["can_decide"] is False          # carol must wait for a reviewer
    assert (await carol.session())["can_review"] is False
    assert (await rita.session())["can_review"] is True

    # carol cannot answer; dave cannot even see the thread or the request
    for who, status in ((carol, 403), (dave, 404)):
        with pytest.raises(Exception) as exc:
            await collect(who.decide(tid, approval["interrupt_id"], [{"type": "approve"}]))
        assert exc.value.status == status
    assert await dave.approvals() == []
    assert await carol.approvals() == []
    with pytest.raises(Exception) as exc:
        await dave.history(tid)
    assert exc.value.status == 404

    # rita finds it in the queue -- but may not read carol's chat
    queue = await rita.approvals()
    assert len(queue) == 1
    item = queue[0]
    assert (item["thread_id"], item["requested_by"], item["requested_by_name"]) == (tid, "carol", "Carol")
    assert item["can_decide"] is True and item["requests"][0]["tool"] == "echo_submit"
    with pytest.raises(Exception) as exc:
        await rita.history(tid)
    assert exc.value.status == 404

    events = await collect(rita.decide(tid, item["interrupt_id"], [{"type": "approve"}]))
    assert events[-1] == ("run_finished", {"status": "completed"})
    manifest = wait(runner, "c1")["manifest"]
    assert (manifest["requested_by"], manifest["approved_by"]) == ("carol", "rita")

    # the conversation continued in carol's thread; nothing is pending any more
    hist = await carol.history(tid)
    assert hist["pending"] == [] and hist["messages"][-1]["text"] == "Started c1."
    assert await rita.approvals() == []


async def test_jobs_visibility_same_on_routes_and_tools(runner):
    runner.submit("echo", "carol_job", {"text": "c"}, approved_by="rita", requested_by="carol")
    runner.submit("echo", "dave_job", {"text": "d"}, approved_by="rita", requested_by="dave")
    wait(runner, "carol_job"), wait(runner, "dave_job")
    app = make_app(runner, [])
    carol, dave, rita = (client(app, u) for u in ("carol", "dave", "rita"))

    # HTTP
    assert [j["job_id"] for j in await carol.jobs()] == ["carol_job"]
    assert sorted(j["job_id"] for j in await rita.jobs()) == ["carol_job", "dave_job"]
    with pytest.raises(Exception) as exc:
        await dave.job("carol_job")
    assert exc.value.status == 404
    assert (await carol.job("carol_job"))["state"] == {}         # spec.public_view for clients
    assert (await rita.job("carol_job"))["state"]["returncode"] == 0  # reviewers see everything
    with pytest.raises(Exception) as exc:                        # clients may not cancel
        await carol.request("POST", "/jobs/carol_job/cancel")
    assert exc.value.status == 403

    # agent tools, same policy
    tools = {t.name: t for t in job_tools(runner)}

    def cfg(user, owner):
        return {"configurable": {"user": USERS[user], "thread_owner": USERS[owner]}}

    out = await tools["job_status"].ainvoke({"job_id": "carol_job"}, config=cfg("dave", "dave"))
    assert out.startswith("REFUSED") and "no such job" in out
    # A reviewer resuming carol's conversation must not leak dave's job into her chat.
    listed = json.loads(await tools["list_jobs"].ainvoke({}, config=cfg("rita", "carol")))
    assert [j["job_id"] for j in listed] == ["carol_job"]


async def test_default_policy_owner_answers_own_approval(runner):
    runner.policy = AccessPolicy()
    app = make_app(runner, [call("echo_submit", job_id="own", text="x"), "ok"])
    carol = client(app, "carol")
    tid = await carol.new_thread()
    events = await collect(carol.send(tid, "go"))
    approval = next(d for e, d in events if e == "approval_required")
    assert approval["can_decide"] is True
    assert [a["thread_id"] for a in await carol.approvals()] == [tid]  # own queue
    await collect(carol.decide(tid, approval["interrupt_id"], [{"type": "approve"}]))
    assert wait(runner, "own")["manifest"]["approved_by"] == "carol"


# --- persistence -----------------------------------------------------------------------

async def test_pending_approval_survives_restart(runner, tmp_path):
    savers = []

    def start(script):
        savers.append(sqlite_checkpointer(tmp_path / "graph.db"))
        return make_app(runner, script, checkpointer=savers[-1],
                        threads=SqliteThreadStore(tmp_path / "threads.db"))

    first = start([call("echo_submit", job_id="durable", text="x")])
    carol = client(first, "carol")
    tid = await carol.new_thread()
    await collect(carol.send(tid, "run it after the restart"))

    # "Restart": a new graph and server over the same files; the model continues its script.
    second = start(["Started after restart."])
    carol, rita = client(second, "carol"), client(second, "rita")
    hist = await carol.history(tid)
    assert hist["title"] == "run it after the restart"
    assert hist["messages"][0] == {**hist["messages"][0], "role": "user"}
    assert len(hist["pending"]) == 1
    queue = await rita.approvals()
    assert [q["thread_id"] for q in queue] == [tid]

    events = await collect(rita.decide(tid, queue[0]["interrupt_id"], [{"type": "approve"}]))
    assert events[-1] == ("run_finished", {"status": "completed"})
    assert wait(runner, "durable")["manifest"]["requested_by"] == "carol"
    assert (await carol.history(tid))["messages"][-1]["text"] == "Started after restart."
    for saver in savers:  # what create_server's shutdown does under a real server
        await saver.aclose()


# --- self-service: each client approves and cancels only their own ---------------------

async def test_self_service_clients_are_isolated(runner):
    runner.policy = RolePolicy("approver", owners_can_decide=True, owners_can_cancel=True)
    app = make_app(runner, [call("echo_submit", job_id="mine", text="sleep"), "Started mine."])
    carol, dave = client(app, "carol"), client(app, "dave")

    tid = await carol.new_thread()
    approval = next(d for e, d in await collect(carol.send(tid, "run it")) if e == "approval_required")
    assert approval["can_decide"] is True                       # carol approves her own
    assert [a["thread_id"] for a in await carol.approvals()] == [tid]
    assert await dave.approvals() == []
    with pytest.raises(Exception) as exc:                        # dave cannot touch it
        await collect(dave.decide(tid, approval["interrupt_id"], [{"type": "approve"}]))
    assert exc.value.status == 404

    await collect(carol.decide(tid, approval["interrupt_id"], [{"type": "approve"}]))
    manifest = wait(runner, "mine")["manifest"]
    assert (manifest["requested_by"], manifest["approved_by"]) == ("carol", "carol")

    # jobs: each sees and cancels only their own
    runner.submit("echo", "daves", {"text": "d"}, approved_by="dave", requested_by="dave")
    assert [j["job_id"] for j in await carol.jobs()] == ["mine"]
    assert [j["job_id"] for j in await dave.jobs()] == ["daves"]
    for who, job, status in ((dave, "mine", 404), (carol, "daves", 404)):
        with pytest.raises(Exception) as exc:
            await who.request("POST", f"/jobs/{job}/cancel")
        assert exc.value.status == status
    wait(runner, "daves")
    with pytest.raises(Exception) as exc:                        # allowed, but finished
        await dave.request("POST", "/jobs/daves/cancel")
    assert exc.value.status == 409
    tools = {t.name: t for t in job_tools(runner)}
    out = await tools["cancel_job"].ainvoke(
        {"job_id": "mine"}, config={"configurable": {"user": USERS["dave"],
                                                     "thread_owner": USERS["dave"]}})
    assert "no such job" in out


# --- nothing a user sees names another user's jobs or a server path ---------------------

async def test_tool_outputs_never_leak_other_jobs_or_paths(runner, monkeypatch):
    runner.policy = RolePolicy("approver", owners_can_decide=True, owners_can_cancel=True)
    runner.max_concurrent = 1
    secret, root = "carol_secret_run", str(runner.root)
    runner.submit("echo", secret, {"sleep": 30}, approved_by="carol", requested_by="carol")
    tools = {t.name: t for t in job_tools(runner)}
    dave = {"configurable": {"user": USERS["dave"], "thread_owner": USERS["dave"]}}
    outputs = []

    async def run(tool, **args):
        out = await tools[tool].ainvoke(args, config=dave)
        outputs.append(out)
        return out

    try:
        assert "server is busy" in await run("echo_preflight")
        assert "server is busy" in await run("echo_submit", job_id="dave1")
        assert json.loads(await run("list_jobs")) == []
        # Asking about carol's job by id answers exactly as for an id that never existed.
        for tool in ("job_status", "cancel_job"):
            real = await tools[tool].ainvoke({"job_id": secret}, config=dave)
            fake = await tools[tool].ainvoke({"job_id": "never_existed"}, config=dave)
            assert "no such job" in real
            assert real.replace(secret, "X") == fake.replace("never_existed", "X")
    finally:
        runner.cancel(secret, requested_by="test")
    wait(runner, secret)

    started = await run("echo_submit", job_id="dave1")
    assert json.loads(started)["started"] is True
    wait(runner, "dave1")
    failed = await run("echo_submit", job_id="dave2", fail_prepare=True)
    assert "could not be prepared" in failed
    assert "cannot link inputs" in (runner.root / "dave2" / "error.txt").read_text()
    outputs.append(await run("job_status", job_id="dave2"))

    def boom(user):
        raise RuntimeError(f"disk error under {root}")
    monkeypatch.setattr(runner, "list_for", boom)
    assert "internal error occurred (ref " in await run("list_jobs")

    for out in outputs:
        assert secret not in out, out
        assert root not in out, out


async def test_a_job_that_fails_to_prepare_still_belongs_to_its_requester(runner):
    runner.policy = RolePolicy("approver", owners_can_decide=True, owners_can_cancel=True)
    root = str(runner.root)
    with pytest.raises(Exception, match="could not be prepared"):
        runner.submit("echo", "broken", {"fail_prepare": True}, approved_by="dave",
                      requested_by="dave")
    # The requester sees the failure, without the detail; nobody else sees the job.
    assert [(j["job_id"], j["status"]) for j in runner.list_for(USERS["dave"])] == [
        ("broken", "failed_to_start")]
    assert runner.list_for(USERS["carol"]) == []
    with pytest.raises(Exception, match="no such job"):
        runner.summary_for(USERS["carol"], "broken")
    state = runner.summary("broken", public=False)["state"]
    assert "ref " in state["error"] and root not in state["error"]
    assert "cannot link inputs" in (runner.root / "broken" / "error.txt").read_text()

    tools = {t.name: t for t in job_tools(runner)}
    as_ = lambda who: {"configurable": {"user": USERS[who], "thread_owner": USERS[who]}}
    mine = await tools["job_status"].ainvoke({"job_id": "broken"}, config=as_("dave"))
    assert "failed_to_start" in mine and root not in mine
    assert "no such job" in await tools["job_status"].ainvoke({"job_id": "broken"},
                                                                config=as_("carol"))


async def test_removing_a_role_takes_effect_in_old_conversations(runner, monkeypatch):
    from reflex_langgraph import MemoryThreadStore
    runner.submit("echo", "carols", {"text": "c"}, approved_by="carol", requested_by="carol")
    wait(runner, "carols")
    threads = MemoryThreadStore()
    app = make_app(runner, [call("list_jobs"), "Listed.", call("list_jobs"), "Listed."],
                   threads=threads)
    rita = client(app, "rita")
    tid = await rita.new_thread()
    # Only id and name are kept with the conversation, never roles.
    assert threads.get(tid).owner_principal == {"id": "rita", "name": "Rita"}

    def listed(events):
        return [d for e, d in events if e == "tool_result" and d.get("tool") == "list_jobs"]

    first = listed(await collect(rita.send(tid, "list jobs")))
    assert "carols" in json.dumps(first)                  # a reviewer sees every job

    monkeypatch.setitem(USERS, "rita", {"id": "rita", "name": "Rita", "roles": []})
    assert await rita.jobs() == []                        # the jobs routes: revoked
    second = listed(await collect(rita.send(tid, "list jobs again")))
    assert second and "carols" not in json.dumps(second)  # the old conversation: revoked too


def test_sqlite_thread_store_upgrades_files_without_updated_at(tmp_path):
    import sqlite3
    from reflex_langgraph.server.store import ThreadRecord
    path = tmp_path / "threads.db"
    with sqlite3.connect(path) as db:          # the schema before updated_at existed
        db.execute("""CREATE TABLE threads (id TEXT PRIMARY KEY, owner TEXT,
            created_at TEXT NOT NULL, title TEXT NOT NULL DEFAULT '', owner_principal TEXT,
            pending INTEGER NOT NULL DEFAULT 0)""")
        db.execute("INSERT INTO threads VALUES ('old', NULL, '2026-10-01T00:00:00+00:00', "
                   "'old chat', NULL, 0)")
    store = SqliteThreadStore(path)
    old = store.get("old")
    assert (old.title, old.updated_at, old.last_activity) == (
        "old chat", "", "2026-10-01T00:00:00+00:00")
    store.create(ThreadRecord(id="new", owner=None, created_at="2026-10-02T00:00:00+00:00",
                              title="new chat"))
    assert [r.id for r in store.list()] == ["new", "old"]
    old.updated_at = "2026-10-03T00:00:00+00:00"   # activity moves it to the top
    store.update(old)
    assert [r.id for r in SqliteThreadStore(path).list()] == ["old", "new"]
