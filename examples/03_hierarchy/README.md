# Example 03: agent hierarchy

Three levels over a small sales table (`sales.csv`: 6 months x 4 regions x 2 products):

```text
manager ─┬─ analysis_lead ─┬─ profiler     columns, values, sample rows
         │                 └─ aggregator   totals and top-N of units or revenue
         └─ reporter                       a short summary for a manager
```

Run it as in [Running](../README.md#running), with `--app-dir 03_hierarchy`.

Try: *"Which region had the most revenue?"*, *"How did Pro units in the North change month
by month?"* or *"Write me a short sales report."*

What to look at:

- A worker can have workers of its own: `analysis_lead` is both a worker (of the manager)
  and a supervisor (of profiler and aggregator). Nesting has no depth limit; cycles are
  rejected when the config loads.
- The figures come from tools, not from the model: `total_by` and `top_n` compute them
  from the CSV, with arguments restricted to known columns (`Literal` types).
