"""Example 04: human approval of actions (refunds and emails), no jobs.

    cd examples
    reflex-langgraph init-env                                   # once: paste GOOGLE_API_KEY into .env
    uvicorn backend:app --app-dir 04_human_in_the_loop --port 8100
    reflex-langgraph ui                                         # another shell, also in examples/

The shop is an in-memory dict: refunds and sent emails reset when the backend restarts.
"""

import json
import os
from pathlib import Path

from reflex_langgraph import ToolRegistry, build_graph, create_server, load_config, load_env_file

HERE = Path(__file__).parent
CONFIG = HERE / "agents.yaml"
load_env_file(HERE.parent / ".env")

ORDERS = {
    "A100": {"customer": "kim@example.com", "item": "Desk lamp", "total": 39.0, "refunded": 0.0},
    "A101": {"customer": "lee@example.com", "item": "Office chair", "total": 180.0, "refunded": 0.0},
    "A102": {"customer": "park@example.com", "item": "Monitor arm", "total": 65.0, "refunded": 20.0},
}
OUTBOX: list = []

registry = ToolRegistry()


@registry.tool
def lookup_order(order_id: str) -> str:
    """An order's customer, item, total and how much was already refunded."""
    order = ORDERS.get(order_id.strip().upper())
    return json.dumps(order) if order else f"no order {order_id!r}; known: {sorted(ORDERS)}"


@registry.tool
def issue_refund(order_id: str, amount: float, reason: str) -> str:
    """Refund part or all of an order. Needs human approval."""
    order = ORDERS.get(order_id.strip().upper())
    if order is None:
        return f"REFUSED: no order {order_id!r}"
    left = round(order["total"] - order["refunded"], 2)
    if not 0 < amount <= left:
        return f"REFUSED: amount must be more than 0 and at most {left} (the unrefunded part)"
    order["refunded"] = round(order["refunded"] + amount, 2)
    return f"refunded {amount:.2f} on {order_id} ({reason}); {left - amount:.2f} left refundable"


@registry.tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to a customer. Needs human approval. (Here it goes to an in-memory outbox.)"""
    OUTBOX.append({"to": to, "subject": subject, "body": body})
    return f"email sent to {to}: {subject!r}"


if not os.environ.get("GOOGLE_API_KEY"):
    raise SystemExit("GOOGLE_API_KEY is not set: add it to examples/.env")
graph = build_graph(load_config(CONFIG), registry)
app = create_server(graph, title="Example 04 · human in the loop")
