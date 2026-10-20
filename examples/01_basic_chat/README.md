# Example 01: basic chat

One agent with two tools, `calculator` and `current_time`. The smallest app: an
`agents.yaml` and a `backend.py` whose tools are plain Python functions.

Run it as in [Running](../README.md#running), with `--app-dir 01_basic_chat`.

Try: *"What is 17.5% of 2,340?"* or *"What time is it in Seoul and in London?"*

You can also attach files (the paperclip, or drop them on the message box): images, PDFs
and text files, kept in `.files/` beside the backend.

What to look at:

- `@registry.tool` turns a typed, documented function into a tool; the docstring is what the
  model reads. Tools return text, including errors, so the agent can recover.
- The calculator parses the expression and allows only arithmetic, never `eval`.
