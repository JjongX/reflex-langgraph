# Guide for coding agents

reflex-langgraph is a Python package: LangGraph agents declared in YAML, a FastAPI server
(conversations, SSE streaming, approvals, jobs, attached files) and a ChatGPT-style Reflex
UI. People install it from PyPI. `README.md` says what it does; this file says how to work on it.

## Layout

| Path | What |
|---|---|
| `reflex_langgraph/agents/` | `build_graph`: agents from the config, middleware, checkpointers |
| `reflex_langgraph/server/` | `create_server`: routes, streaming, thread stores, the API's types |
| `reflex_langgraph/ui/` | Reflex states and components; `launcher.py` generates the app `reflex-langgraph ui` runs |
| `reflex_langgraph/jobs/` | detached jobs: `JobSpec`, `JobRunner`, progress, agent tools, routes |
| `reflex_langgraph/files.py` | attached files: formats and stores |
| `reflex_langgraph/config.py`, `env.py`, `policy.py`, `cli/` | YAML config, `.env` files, access policy, commands |
| `tests/` | pytest; `scripted.py` is a chat model that replays a script |
| `docs/` | Sphinx; `create.py` writes `api.rst` and `cli.rst` from the code |
| `examples/` | five runnable apps; they need `GOOGLE_API_KEY` in `examples/.env` |

## Commands

```bash
pip install -e ".[dev,google]"
pytest -q                                   # all tests; no API key needed
pip install -e ".[docs,google]"
make -C docs html SPHINXOPTS="-W"           # docs; warnings are errors
python -m build && twine check dist/*       # the package
```

A change is done when the tests pass and, if a docstring or a command's help changed, the
docs build passes with `-W`.

## Code

- Python 3.11 or later. Match the surrounding code: its naming, idioms and comment density
  (few comments; they say why, not what).
- Public names are exported from `reflex_langgraph/__init__.py` or their subpackage's
  `__init__.py`. Every public module, class, function and method has a NumPy-style
  docstring whose examples run as doctests; `tests/test_docs.py` lists and checks the rules.
- Tests never call a real model provider: use `tests/scripted.py` and the helpers in
  `tests/conftest.py`. Each fix or feature gets a test, and a fix's test fails without it.
- Add dependencies sparingly; optional ones go in an extra in `setup.py`. Reflex stays
  pinned to 0.10.x.

## Design rules

- The UI holds only screen state. API keys and agents live in the API; the UI reads only
  `RLG_API_URL` and `RLG_UI_*` settings.
- A run belongs to the server: closing the browser or restarting the UI never stops it.
- The server runs as one process: active runs live in its memory.
- Messages carry a file's id and name, never its bytes; files stay in the `FileStore`.
- Save a thread with only the fields that changed (`ThreadStore.update(record, fields)`),
  so concurrent updates cannot overwrite each other.
- A job's worker process imports no user code: everything it runs is frozen into
  `state.json` when the job is submitted.
- The Reflex app is generated and rewritten on every start. Never patch Reflex or any other
  dependency.
- Keep the package generic: no company, product, internal project, host or domain names in
  code, docs, tests or examples. Default file formats stay general; apps add their own with
  `FileFormat`.

## Writing

- Plain, short English, one idea per sentence. The README and docs stay concise; detail
  goes in docstrings, which the API reference is built from.
- Every user-visible change gets a line in `CHANGELOG.md` under the next version.

## Git

- Never commit secrets. `.env` files are git-ignored; `examples/.env.example` keeps its
  values empty.
- Commit messages are short, capitalized and imperative, with no full stop (e.g. "Add S3
  file storage").
- Do not push, tag, publish or force-push unless the maintainer asks.

## Releases

1. Work on a `<version>-dev` branch (e.g. `0.2.0-dev`). The middle number goes up for new
   features, the last for fixes.
2. Set the version in `reflex_langgraph/version.py`, and date the CHANGELOG heading:
   `## 0.2.0 (YYYY-MM-DD)`.
3. Open a pull request into `main` and merge it.
4. Tag the merge commit with the bare version (`git tag 0.2.0`, no `v`) and push the tag.
5. `rm -rf dist build *.egg-info && python -m build && twine check dist/* && twine upload dist/*`
6. Create the GitHub release from the tag, named after the version.

PyPI never accepts the same version twice, so a fix after an upload is a new version.
