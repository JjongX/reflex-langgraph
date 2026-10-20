# reflex-langgraph

Build LangGraph multi-agent apps with a **FastAPI** backend and a ChatGPT-style **Reflex**
UI, with **human approval** of tool calls and **long-running jobs** that survive restarts.

```text
browser ──▶ Reflex server ──HTTP + SSE──▶ your API (create_server) ──▶ LangGraph agents
            (reflex-langgraph ui,          conversations, approvals,          │
             no frontend code)             access policy, jobs                ▼
                                                                    detached job processes
```

> Status: alpha (0.1). The API may change before 1.0.

The UI holds only screen state and never sees API keys; the agents, approvals and jobs live
in your API, which works without the UI too. A run belongs to the API: closing the browser
or restarting the UI never stops it. **There is no login**: anyone who can open the UI can
ask, approve and cancel, so keep it private (an SSH tunnel, or a firewall open to known
addresses).

## Features

- **Agents from one YAML file.** An agent with `workers` is a supervisor that delegates to
  them; nesting gives hierarchies of any depth. Any LangChain chat model
  (`google_genai:…`, `openai:…`, `anthropic:…`).
- **A chat page like ChatGPT's.** Streaming replies with "Thinking", tool calls, the
  conversation list (search, rename, delete), Try again. Each reply is labelled with the
  agent that wrote it.
- **Human-in-the-loop.** List a tool under `approve:` and every call to it waits for
  approve / edit / reject, even inside a worker.
- **Long-running jobs.** Describe a job once (`JobSpec`) and agents get preflight / submit /
  status / cancel tools. Jobs run detached, with a concurrency limit, step-by-step progress,
  and a record of who approved them and the hashes of their inputs.
- **Blocked replies.** When a provider refuses a prompt or withholds a reply, the chat says
  so, and the blocked turn is left out of later model calls.
- **Attached files** (opt-in: `create_server(files=LocalFileStore(...))`, or
  `S3FileStore("bucket")` with the `s3` extra, or your own `FileStore` subclass). Images,
  PDFs and text files go with a message: paperclip or drag and drop, up to 100 MB and 30 files.
  An app adds its own formats (e.g. YAML or compressed CSV) with `LocalFileStore(root,
  formats=[*FORMATS, FileFormat(...)])`, and the paperclip follows. The model reads what it
  can (binary data is described to it); agents hand files to workers and tools read them
  by id (`attached_file`).
- **Optional:** durable conversations (SQLite), and an access policy with reviewers when an
  `auth=` hook identifies callers.

## Install

```bash
pip install "reflex-langgraph[ui,server,google]"   # extras: ui, server, sqlite, s3, google, openai, anthropic
```

## Quick start

`agents.yaml`:

```yaml
model: google_genai:gemini-2.5-flash
entry: supervisor
agents:
  - name: supervisor
    prompt: You coordinate the team.
    workers: [operator]
  - name: operator
    description: Starts jobs. Every start needs approval.
    tools: [train_preflight, train_submit, job_status]
    approve:
      train_submit: [approve, edit, reject]
```

`backend.py`:

```python
from pathlib import Path
from pydantic import BaseModel
from reflex_langgraph import (JobRunner, JobSpec, ToolRegistry, build_graph, create_server,
                              job_tools, jobs_router, load_config)

HERE = Path(__file__).parent

class TrainParams(BaseModel):
    dataset: str

class Train(JobSpec):
    name = "train"
    description = "Train a model on a dataset directory."
    params_model = TrainParams

    def command(self, p, job_dir):
        # Jobs run inside their own new folder: give the script's full path.
        return ["python", str(HERE / "train.py"), "--data", p.dataset, "--out", str(job_dir)]

runner = JobRunner(HERE / "jobs", [Train()])
graph = build_graph(load_config(HERE / "agents.yaml"), ToolRegistry(job_tools(runner)))
app = create_server(graph, routers=[jobs_router(runner)])
```

Run the API (as one process: active runs live in its memory):

```bash
uvicorn backend:app --port 8100
```

Then the UI, in one of two modes (both open at http://localhost:3000):

```bash
# Production: to try it or show it. Fast (about 0.2 s, 14 files); one port (3000).
reflex-langgraph ui --prod --port 3000 --backend-port 3000

# Development: while changing UI code. Reloads on save, but the page takes seconds
# (about 250 files); uses two ports, 3000 and 8000.
reflex-langgraph ui
```

In production the two port flags must be the same number. In development the browser must
reach both ports: forward both through a tunnel, or pass `--public-url
http://<host>:8000`. `reflex-langgraph ui -h` lists the other options (title, greeting,
accent, ...).

API keys and settings go in a git-ignored `.env` (`reflex-langgraph init-env` creates it
from `.env.example`). The backend loads it with `load_env_file`. The UI reads only
`RLG_API_URL` and `RLG_UI_*` from it, so the keys never reach the UI.

For your own page layout, build a Reflex app from the components
(`from reflex_langgraph.ui import add_agent_page, agent_page, chat_panel, ...`).

## Examples

Five runnable apps, simplest first, in [`examples/`](examples) (how to run them is there):
basic chat, a supervisor team, a three-level hierarchy, human in the loop, and jobs with
approval.

## Documentation

The API reference, the command line and the changelog are built from the code:

```bash
pip install -e ".[docs,ui,server,sqlite,google]"
make -C docs html          # -> docs/_build/html/index.html
```

## Development

```bash
pip install -e ".[dev,ui,google]"
pytest -q                  # tests/test_docs.py also checks the docstring style
```

## License

MIT
