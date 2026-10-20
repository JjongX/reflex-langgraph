"""Example 05: agents that start a long-running job behind an approval.

    cd examples
    reflex-langgraph init-env                                   # once: paste GOOGLE_API_KEY into .env
    uvicorn backend:app --app-dir 05_jobs_with_approval --port 8100
    reflex-langgraph ui                                         # another shell, also in examples/

Settings come from ``examples/.env`` (see ``examples/.env.example``): GOOGLE_API_KEY, and
JOBS_EXAMPLE_DATA_DIR to keep conversations across restarts.

There is no login: whoever opens the UI asks for a job, approves or rejects it on the
approval card, and follows it in the Jobs panel.
"""

import os
from pathlib import Path

from pydantic import BaseModel, Field

from reflex_langgraph import SqliteThreadStore, env_path, load_env_file, sqlite_checkpointer
from reflex_langgraph.agents import ToolRegistry, build_graph
from reflex_langgraph.config import load_config
from reflex_langgraph.jobs import JobRunner, JobSpec, Preflight, StepMarkerParser, job_tools, jobs_router
from reflex_langgraph.server import create_server

HERE = Path(__file__).parent
CONFIG = HERE / "agents.yaml"
load_env_file(HERE.parent / ".env")  # examples/.env; shell variables win over the file


class CountdownParams(BaseModel):
    steps: int = Field(5, ge=1, le=20, description="How many steps to run.")
    seconds_per_step: int = Field(3, ge=1, le=60, description="Seconds each step takes.")


class Countdown(JobSpec):
    name = "countdown"
    description = "A demo job that runs N timed steps and prints progress markers."
    params_model = CountdownParams
    log_name = "countdown.log"

    def preflight(self, p: CountdownParams) -> Preflight:
        r = Preflight(info={"estimated minutes": round(p.steps * p.seconds_per_step / 60, 1)},
                      required_bytes=1_000_000)
        if p.steps * p.seconds_per_step > 600:
            r.blockers.append("demo jobs must finish within 10 minutes")
        return r

    def command(self, p: CountdownParams, job_dir: Path):
        script = (f'echo "TOTAL {p.steps}"; for i in $(seq 1 {p.steps}); do '
                  f'printf "STEP %02d start\\n" "$i"; sleep {p.seconds_per_step}; '
                  f'echo "$i" >> done.txt; done; echo "ALL DONE"')
        return ["bash", "-c", script]

    def version_commands(self, p, job_dir):
        return {"bash": ["bash", "--version"]}


# The step plan depends on params, so the parser is built per job from the manifest.
class CountdownProgress:
    def parse(self, log_text: str, alive: bool):
        first = log_text.split("\n", 1)[0]
        total = int(first.split()[1]) if first.startswith("TOTAL ") else 1
        parser = StepMarkerParser(plan=[(f"{i:02d}", f"step {i:02d}") for i in range(1, total + 1)],
                                  start=r"^STEP (?P<key>\d+) start$")
        progress = parser.parse(log_text, alive)
        if "ALL DONE" in log_text:
            progress.finished = True
            for s in progress.steps:
                s.status = "pass"
        return progress


Countdown.progress = CountdownProgress()

if not os.environ.get("GOOGLE_API_KEY"):
    raise SystemExit("GOOGLE_API_KEY is not set: add it to examples/.env")
# Relative to examples/, where the .env file is.
data = (env_path("JOBS_EXAMPLE_DATA_DIR", "", HERE.parent)
        if os.environ.get("JOBS_EXAMPLE_DATA_DIR") else None)

runner = JobRunner(HERE / ".jobs", [Countdown()], max_concurrent=1)
registry = ToolRegistry(job_tools(runner))
graph = build_graph(load_config(CONFIG), registry,
                    checkpointer=sqlite_checkpointer(Path(data) / "graph.db") if data else None)

app = create_server(graph, title="Example 05 · jobs with approval",
                    threads=SqliteThreadStore(Path(data) / "threads.db") if data else None,
                    routers=[jobs_router(runner)])
