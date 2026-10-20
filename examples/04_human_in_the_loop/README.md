# Example 04: human in the loop

A support agent for a small shop (three orders, in memory). Looking orders up is free;
**every refund and every email waits for a person** to approve, edit or reject it.

Run it as in [Running](../README.md#running), with `--app-dir 04_human_in_the_loop`.

Try: *"Order A101 arrived broken, please refund it."* An approval card shows the refund's
arguments.

- **Approve**: the refund runs as proposed.
- **Edit**: change the arguments (e.g. the amount) before it runs.
- **Reject** with a note: the tool does not run; the agent reads your note and continues.

What to look at:

- `approve:` in `agents.yaml` lists the tools that need a decision and which decisions are
  allowed. Nothing else in the code changes.
- Approvals need no jobs: any tool can be gated. (Example 05 gates long-running jobs.)
- The shop resets when the backend restarts.
