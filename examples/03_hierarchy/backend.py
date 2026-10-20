"""Example 03: a hierarchy of agents over a CSV file.

    cd examples
    reflex-langgraph init-env                             # once: paste GOOGLE_API_KEY into .env
    uvicorn backend:app --app-dir 03_hierarchy --port 8100
    reflex-langgraph ui                                   # another shell, also in examples/
"""

import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Literal, Optional

from reflex_langgraph import ToolRegistry, build_graph, create_server, load_config, load_env_file

HERE = Path(__file__).parent
CONFIG = HERE / "agents.yaml"
DATA = HERE / "sales.csv"
load_env_file(HERE.parent / ".env")

Measure = Literal["units", "revenue"]
Column = Literal["month", "region", "product"]


def rows() -> list:
    """The data, with revenue = units * unit_price added."""
    with DATA.open(newline="") as fh:
        out = []
        for r in csv.DictReader(fh):
            r["units"], r["unit_price"] = int(r["units"]), float(r["unit_price"])
            r["revenue"] = round(r["units"] * r["unit_price"], 2)
            out.append(r)
        return out


def _totals(measure: str, by: str, where_column: Optional[str], where_value: Optional[str]):
    totals = defaultdict(float)
    for r in rows():
        if where_column and str(r[where_column]) != where_value:
            continue
        totals[r[by]] += r[measure]
    return {k: round(v, 2) for k, v in sorted(totals.items())}


registry = ToolRegistry()


@registry.tool
def list_columns() -> str:
    """Columns of sales.csv, the number of rows, and the values of each text column."""
    data = rows()
    return json.dumps({
        "rows": len(data),
        "columns": {"month": sorted({r["month"] for r in data}),
                    "region": sorted({r["region"] for r in data}),
                    "product": sorted({r["product"] for r in data}),
                    "units": "integer", "unit_price": "number",
                    "revenue": "number, computed as units * unit_price"}}, indent=1)


@registry.tool
def preview_rows(n: int = 5) -> str:
    """The first n rows (at most 20)."""
    return json.dumps(rows()[:max(1, min(n, 20))], indent=1)


@registry.tool
def total_by(measure: Measure, by: Column, where_column: Optional[Column] = None,
             where_value: Optional[str] = None) -> str:
    """Sum units or revenue per value of a column, optionally only rows where
    where_column == where_value (e.g. by="month", where_column="region", where_value="North")."""
    return json.dumps(_totals(measure, by, where_column, where_value), indent=1)


@registry.tool
def top_n(measure: Measure, by: Column, n: int = 3) -> str:
    """The n values of a column with the highest total units or revenue."""
    ranked = sorted(_totals(measure, by, None, None).items(), key=lambda kv: kv[1], reverse=True)
    return json.dumps(ranked[:max(1, n)], indent=1)


if not os.environ.get("GOOGLE_API_KEY"):
    raise SystemExit("GOOGLE_API_KEY is not set: add it to examples/.env")
graph = build_graph(load_config(CONFIG), registry)
app = create_server(graph, title="Example 03 · agent hierarchy")
