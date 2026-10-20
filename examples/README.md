# Examples

Each example is an `agents.yaml` and a `backend.py`; the UI is the generic
`reflex-langgraph ui`.

| Example | Shows |
|---|---|
| [01 basic chat](01_basic_chat) | One agent with plain Python tools. |
| [02 supervisor team](02_supervisor_team) | A supervisor delegating to a researcher and a writer; per-agent model settings. |
| [03 hierarchy](03_hierarchy) | Three levels: manager -> analysis lead -> profiler / aggregator, over a CSV. |
| [04 human in the loop](04_human_in_the_loop) | Approve / edit / reject refunds and emails. |
| [05 jobs with approval](05_jobs_with_approval) | Long-running jobs behind an approval, with live progress in the Jobs panel. |

## Running

From this folder, with one shared settings file:

```bash
cd examples
reflex-langgraph init-env                                        # once; then paste GOOGLE_API_KEY into .env
uvicorn backend:app --app-dir 02_supervisor_team --port 8100     # pick an example
```

Then the UI, in another shell (also here), in one of two modes -> http://localhost:3000:

```bash
reflex-langgraph ui --prod --port 3000 --backend-port 3000   # production: fast, port 3000 only
reflex-langgraph ui                                          # development: for UI changes; slow to load, ports 3000 and 8000
```

Run one backend at a time; restart the UI after switching, so its title updates.
