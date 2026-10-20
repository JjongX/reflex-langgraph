import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest
from pydantic import BaseModel

from reflex_langgraph.agents import ToolRegistry, build_graph
from reflex_langgraph.config import load_config
from reflex_langgraph.jobs import (JobError, JobRunner, JobSpec, Preflight, StepMarkerParser,
                                   job_tools, jobs_router)
from reflex_langgraph.server import create_server
from scripted import ScriptedChatModel, call

from conftest import sse_events, wait

SCRIPT = """
for s in 01 02 03; do
  echo "=== [$(date +%H:%M:%S)] step $s ==="
  sleep "${SLEEP:-0}"
  if [ "$s" = "${FAIL:-x}" ]; then R+=("FAIL(4)  $s"); else R+=("PASS  $s"); fi
done
echo "== RESULT =="
printf '%s\\n' "${R[@]}"
cat "$INPUT" > copied.txt
"""


class Params(BaseModel):
    input_file: str
    fail_step: str = "x"
    sleep: float = 0


class Demo(JobSpec):
    name = "demo"
    description = "A three-step demo job."
    params_model = Params
    log_name = "run.log"
    progress = StepMarkerParser(
        plan=[("01", "01"), ("02", "02"), ("03", "03")],
        start=r"^=== \[(?P<time>[\d:]+)\] step (?P<key>\d+) ===$",
        summary_start=r"^== RESULT ==$",
        result=r"^(?P<status>PASS|FAIL\((?P<rc>\d+)\))\s+(?P<label>\S+)$",
    )

    def __init__(self, script_dir: Path, protected=()):
        self.script = script_dir / "job.sh"
        self.script.write_text(SCRIPT)
        self.protected_paths = protected

    def preflight(self, p):
        r = Preflight(info={"input": p.input_file})
        if not Path(p.input_file).is_file():
            r.blockers.append("input file does not exist")
        return r

    def command(self, p, job_dir):
        return ["bash", str(self.script)]

    def env(self, p, job_dir):
        return {"INPUT": p.input_file, "FAIL": p.fail_step, "SLEEP": str(p.sleep)}

    def fingerprint(self, p, job_dir):
        return {"script": self.script, "input": Path(p.input_file)}

    def version_commands(self, p, job_dir):
        return {"bash": ["bash", "--version"], "missing": ["no-such-tool-xyz"]}

    def public_view(self, summary):
        summary["manifest"].pop("fingerprints", None)
        return summary


@pytest.fixture
def env(tmp_path):
    root = tmp_path / "jobs"
    spec = Demo(tmp_path, protected=[root / "archive"])
    runner = JobRunner(root, [spec])
    inp = tmp_path / "in.txt"
    inp.write_text("hello")
    yield runner, inp
    for j in runner.active():
        try:
            runner.cancel(j, requested_by="pytest")
        except JobError:
            pass


def test_success_records_provenance(env):
    runner, inp = env
    runner.submit("demo", "j1", {"input_file": str(inp)}, approved_by="alice")
    s = wait(runner, "j1")
    assert s["status"] == "succeeded"
    assert [st["status"] for st in s["progress"]["steps"]] == ["pass"] * 3
    m = s["manifest"]
    assert m["approved_by"] == "alice" and m["params"]["input_file"] == str(inp)
    assert m["fingerprints"]["input"]["sha256"]
    assert m["versions"]["bash"].startswith("GNU bash")
    assert m["versions"]["missing"].startswith("unavailable")
    assert (runner.root / "j1" / "copied.txt").read_text() == "hello"


def test_public_view_filters(env):
    runner, inp = env
    runner.submit("demo", "j1", {"input_file": str(inp)}, approved_by="a")
    wait(runner, "j1")
    assert "fingerprints" not in runner.summary("j1")["manifest"]
    assert "fingerprints" in runner.summary("j1", public=False)["manifest"]


def test_exit_zero_with_failed_step_is_not_success(env):
    runner, inp = env
    runner.submit("demo", "j1", {"input_file": str(inp), "fail_step": "02"}, approved_by="a")
    s = wait(runner, "j1")
    assert s["status"] == "completed_with_failures"
    step = s["progress"]["steps"][1]
    assert step["status"] == "fail" and step["returncode"] == 4


def test_guards(env):
    runner, inp = env
    with pytest.raises(JobError, match="approver"):
        runner.submit("demo", "j1", {"input_file": str(inp)}, approved_by=" ")
    with pytest.raises(JobError, match="invalid job id"):
        runner.submit("demo", "../x", {"input_file": str(inp)}, approved_by="a")
    with pytest.raises(JobError, match="reserved"):
        runner.submit("demo", "archive", {"input_file": str(inp)}, approved_by="a")
    with pytest.raises(JobError, match="preflight failed"):
        runner.submit("demo", "j1", {"input_file": "/nope"}, approved_by="a")
    with pytest.raises(JobError, match="invalid parameters"):
        runner.submit("demo", "j1", {}, approved_by="a")
    assert not (runner.root / "j1").exists()

    runner.submit("demo", "j1", {"input_file": str(inp)}, approved_by="a")
    wait(runner, "j1")
    with pytest.raises(JobError, match="not available"):
        runner.submit("demo", "j1", {"input_file": str(inp)}, approved_by="a")


def test_concurrency_limit_and_cancel(env):
    runner, inp = env
    runner.submit("demo", "slow", {"input_file": str(inp), "sleep": 5}, approved_by="a")
    wait(runner, "slow", done=("running",))
    # "running" is set when the command launches, a moment before it prints step 01's
    # banner; cancel only once step 01 has really started, so it reads as interrupted.
    end = time.time() + 10
    while runner.summary("slow")["progress"]["steps"][0]["status"] != "running":
        assert time.time() < end, runner.summary("slow")
        time.sleep(0.05)
    assert any("server is busy" in b for b in runner.preflight("demo", {"input_file": str(inp)}).blockers)
    with pytest.raises(JobError, match="server is busy"):
        runner.submit("demo", "j2", {"input_file": str(inp)}, approved_by="a")

    runner.cancel("slow", requested_by="bob")
    s = wait(runner, "slow")
    assert s["status"] == "cancelled" and s["state"]["cancelled_by"] == "bob"
    assert [st["status"] for st in s["progress"]["steps"]] == ["interrupted", "pending", "pending"]
    assert runner.active() == []


def test_progress_mid_run_is_done_not_pass():
    p = Demo.progress.parse("=== [1] step 01 ===\n=== [2] step 02 ===\n", alive=True)
    assert [s.status for s in p.steps] == ["done", "running", "pending"]
    assert [s.status for s in Demo.progress.parse("=== [1] step 01 ===", alive=False).steps] == \
        ["interrupted", "pending", "pending"]


def test_unmanaged_dirs_hidden_unless_enabled(env, tmp_path):
    runner, _ = env
    (runner.root).mkdir(parents=True, exist_ok=True)
    (runner.root / "old_run").mkdir()
    assert runner.list() == []
    with pytest.raises(JobError):
        runner.summary("old_run")
    runner.include_unmanaged = True
    assert runner.list()[0]["status"] == "unmanaged"


# --- through the agent, with approval -----------------------------------------------

@pytest.mark.anyio
@pytest.mark.parametrize("auth, recorded", [
    (None, "anonymous"),                                   # no login: the default
    (lambda r: {"id": "u1", "name": "Alice"}, "u1"),       # an auth hook identifies callers
])
async def test_agent_submits_only_after_approval(env, auth, recorded):
    runner, inp = env
    registry = ToolRegistry(job_tools(runner))
    cfg = load_config({"model": "x:y", "agents": [{
        "name": "operator", "tools": ["demo_preflight", "demo_submit", "job_status"],
        "approve": {"demo_submit": ["approve", "reject"]}}]})
    model = ScriptedChatModel(script=[
        call("demo_preflight", input_file=str(inp)),
        call("demo_submit", job_id="via_agent", input_file=str(inp), approved_by="mallory"),
        "Started.",
    ])
    graph = build_graph(cfg, registry, models={"operator": model})
    app = create_server(graph, auth=auth, routers=[jobs_router(runner)])
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        h = {}
        tid = (await c.post("/threads")).json()["thread_id"]
        first = await sse_events(c, f"/threads/{tid}/messages", {"content": "run it"}, h)
        pre = next(d for e, d in first if e == "tool_result" and d["tool"] == "demo_preflight")
        assert json.loads(pre["output"])["ok"] is True
        approval = next(d for e, d in first if e == "approval_required")
        assert not (runner.root / "via_agent").exists()

        await sse_events(c, f"/threads/{tid}/decisions",
                   {"interrupt_id": approval["interrupt_id"], "decisions": [{"type": "approve"}]}, h)
        s = wait(runner, "via_agent")
        # the model's "approved_by" argument is ignored; the server's identity is recorded
        assert s["manifest"]["approved_by"] == recorded
        assert s["manifest"]["requested_by"] == recorded

        jobs = (await c.get("/jobs")).json()
        assert jobs[0]["job_id"] == "via_agent"
        assert (await c.get("/jobs/via_agent")).json()["status"] == "succeeded"
        assert (await c.get("/jobs/via_agent/log")).status_code == 404  # logs not exposed


def test_concurrent_state_updates_are_not_lost(tmp_path):
    """The web server and the worker update state.json from different processes."""
    import subprocess
    import sys
    from reflex_langgraph.jobs.runner import STATE
    (tmp_path / STATE).write_text("{}")
    code = ("import sys; from pathlib import Path; from reflex_langgraph.jobs.runner import _update\n"
            "for i in range(150): _update(Path(sys.argv[1]), **{f'{sys.argv[2]}{i}': i})")
    procs = [subprocess.Popen([sys.executable, "-c", code, str(tmp_path), w]) for w in "ab"]
    assert [p.wait(60) for p in procs] == [0, 0]
    state = json.loads((tmp_path / STATE).read_text())
    assert {k for k in state if k != "updated_at"} == {f"{w}{i}" for w in "ab" for i in range(150)}
    assert not list(tmp_path.glob("*.tmp"))


def test_a_job_is_alive_while_its_worker_or_command_is(env):
    """The worker records the outcome a moment after the command exits; a status read in
    that moment used to say "died" (seen as a flaky test)."""
    import subprocess
    from reflex_langgraph.jobs.runner import START_GRACE
    runner, _ = env
    gone = subprocess.Popen(["true"])
    gone.wait()
    me, dead = os.getpid(), gone.pid
    long_ago = (datetime.now(timezone.utc) - START_GRACE * 2).isoformat()
    cases = [
        ({"status": "running", "worker_pid": me, "command_pid": dead}, "running"),  # recording
        ({"status": "running", "worker_pid": dead, "command_pid": me}, "running"),  # orphan
        ({"status": "running", "worker_pid": dead, "command_pid": dead}, "died"),
        ({"status": "starting", "updated_at": datetime.now(timezone.utc).isoformat()},
         "starting"),                                       # submit is starting the worker
        ({"status": "starting", "updated_at": long_ago}, "died"),
        ({"status": "failed", "worker_pid": me}, "failed"),
    ]
    assert [runner._status(state) for state, _ in cases] == [want for _, want in cases]
