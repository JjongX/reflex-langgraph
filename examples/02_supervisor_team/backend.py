"""Example 02: a supervisor and its team (researcher -> writer).

    cd examples
    reflex-langgraph init-env                                  # once: paste GOOGLE_API_KEY into .env
    uvicorn backend:app --app-dir 02_supervisor_team --port 8100
    reflex-langgraph ui                                        # another shell, also in examples/
"""

import os
from pathlib import Path

from reflex_langgraph import (LocalFileStore, ToolRegistry, build_graph, create_server,
                              load_config, load_env_file)

HERE = Path(__file__).parent
CONFIG = HERE / "agents.yaml"
NOTES = HERE / "notes"
load_env_file(HERE.parent / ".env")

registry = ToolRegistry()


@registry.tool
def search_notes(query: str) -> str:
    """Find lines in the team's notes that contain any of the query's words."""
    words = [w for w in query.lower().split() if len(w) > 2]
    hits = []
    for note in sorted(NOTES.glob("*.md")):
        for line in note.read_text().splitlines():
            if line.startswith("- ") and any(w in line.lower() for w in words):
                hits.append(f"[{note.stem}] {line[2:]}")
    return "\n".join(hits) or f"no notes mention {query!r}. Notes: {list_names()}"


@registry.tool
def read_note(name: str) -> str:
    """The full text of one note, by name (e.g. "solar")."""
    path = NOTES / f"{Path(name).stem}.md"
    return path.read_text() if path.is_file() else f"no note {name!r}. Notes: {list_names()}"


def list_names() -> str:
    return ", ".join(p.stem for p in sorted(NOTES.glob("*.md")))


if not os.environ.get("GOOGLE_API_KEY"):
    raise SystemExit("GOOGLE_API_KEY is not set: add it to examples/.env")
graph = build_graph(load_config(CONFIG), registry)
# Users may attach files (images, PDF, text); they are kept in .files/ here.
app = create_server(graph, title="Example 02 · supervisor team", files=LocalFileStore(HERE / ".files"))
