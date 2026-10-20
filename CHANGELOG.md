# Changelog

## 0.1.0 (unreleased)

First version.

- **Agents:** one YAML file declares every agent, its model, tools and workers;
  supervisors and hierarchies of any depth (`load_config`, `build_graph`). Approval pauses
  inside workers surface on the entry graph and resume in place. Apps add their own agent
  middleware with `build_graph(middleware=...)` (a list, per agent, or a factory).
- **Server:** `create_server`, a FastAPI app with conversations, SSE streaming, approval
  decisions, cancel, rename, delete, search and regenerate. Runs belong to the server, so a
  dropped connection never stops one. A typed contract (`server/schemas.py`,
  `server/events.py`), validated by `BackendClient`.
- **UI:** `reflex-langgraph ui` runs a ChatGPT-style page for any backend, with no frontend
  code. It has a conversation list (search, rename, delete), streaming replies with
  "Thinking", one-line tool calls, approval cards, Try again and a jobs panel. Its
  components (`add_agent_page`, ...) also work in your own Reflex app. There is no login.
- **Jobs:** `JobSpec` and `JobRunner` run detached jobs with a concurrency limit, cancel,
  step-by-step progress (`StepMarkerParser`) and provenance (approver, input hashes, tool
  versions). `job_tools` gives agents preflight / submit / status / cancel, and
  `jobs_router` serves the jobs panel.
- **Reasoning and blocked replies:** thinking summaries stream and stay in history. Replies
  a provider blocked are shown as such and left out of later model calls.
- **Access policy** (`AccessPolicy`, `RolePolicy`): one policy for the routes and the job
  tools, used when an `auth=` hook identifies callers. Includes an approvals queue for
  reviewers.
- **Attached files** (opt-in, `create_server(files=...)`: `LocalFileStore`, `S3FileStore`
  with the `s3` extra, or a `FileStore` subclass that implements five storage methods; tools
  get a local copy wherever files are kept): upload routes,
  file chips in the message box and in history, and `AttachmentMiddleware`, which shows the
  model the files of the message it answers (images and PDFs as content when it reads
  them, text as text, the rest described). Accepts images, PDF and text by default
  (`files.FORMATS`); an app adds formats as `FileFormat` entries, which `/health` passes on
  to the paperclip. Every file is checked against its first bytes. Workers receive files
  by id, and tools read them with `attached_file`.
- **Persistence:** `SqliteThreadStore` and `sqlite_checkpointer` (`[sqlite]` extra).
- **Settings:** `.env` files (`load_env_file`, `env_path`, `reflex-langgraph init-env`).
- **Docs:** five examples, NumPy-style docstrings with examples that run, and a Sphinx site
  built from the code.
