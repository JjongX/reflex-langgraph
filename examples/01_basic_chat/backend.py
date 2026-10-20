"""Example 01: one agent with tools.

    cd examples
    reflex-langgraph init-env                             # once: paste GOOGLE_API_KEY into .env
    uvicorn backend:app --app-dir 01_basic_chat --port 8100
    reflex-langgraph ui                                   # another shell, also in examples/
"""

import ast
import operator
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from reflex_langgraph import (LocalFileStore, ToolRegistry, build_graph, create_server,
                              load_config, load_env_file)

HERE = Path(__file__).parent
CONFIG = HERE / "agents.yaml"
load_env_file(HERE.parent / ".env")

registry = ToolRegistry()

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
        ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        if isinstance(node.op, ast.Pow) and abs(_eval(node.right)) > 100:
            raise ValueError("exponent too large")
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("only numbers and + - * / // % ** are allowed")


@registry.tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression, e.g. "(12.5 * 4) / 3" or "2 ** 10"."""
    try:
        return str(_eval(ast.parse(expression, mode="eval").body))
    except (ValueError, SyntaxError, ZeroDivisionError) as exc:
        return f"error: {exc}"


@registry.tool
def current_time(timezone: str = "UTC") -> str:
    """The current date and time in an IANA time zone, e.g. "Asia/Seoul" or "Europe/London"."""
    try:
        return datetime.now(ZoneInfo(timezone)).strftime("%A %Y-%m-%d %H:%M %Z")
    except ZoneInfoNotFoundError:
        return f"error: unknown time zone {timezone!r}"


if not os.environ.get("GOOGLE_API_KEY"):
    raise SystemExit("GOOGLE_API_KEY is not set: add it to examples/.env")
graph = build_graph(load_config(CONFIG), registry)
# Users may attach files (images, PDF, text); they are kept in .files/ here.
app = create_server(graph, title="Example 01 · basic chat", files=LocalFileStore(HERE / ".files"))
