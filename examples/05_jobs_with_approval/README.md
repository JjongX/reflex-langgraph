# Example 05: jobs with approval

Everything from the earlier examples, plus long-running work: a supervisor delegates to an
`inspector` (read-only) and an `operator` whose `countdown_submit` calls need approval. Jobs
run detached, outlive the backend, and stream step-by-step progress to the Jobs panel on the
right.

Run it as in [Running](../README.md#running), with `--app-dir 05_jobs_with_approval`.

Its own setting is in the shared `examples/.env`, under "05_jobs_with_approval only":
`JOBS_EXAMPLE_DATA_DIR=data` keeps conversations and pending approvals across backend
restarts, in `examples/data/` (`pip install "../..[sqlite]"`).

Try: *"Start a 4-step countdown job named demo1."* An approval card appears in the chat;
**Approve** it, and the job starts and shows in the Jobs panel.

What to look at:

- A `JobSpec` (the `Countdown` class) describes the job once; `job_tools` gives the agents
  preflight / submit / status / cancel tools, and `jobs_router` gives the UI its Jobs panel.
- Jobs are separate processes under `.jobs/`, so restarting the backend never stops them.
