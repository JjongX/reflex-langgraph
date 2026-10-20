# Example 02: supervisor team

An `editor` supervises two workers: a `researcher` that searches the team's notes
(`notes/*.md`) and a `writer` with no tools. The editor asks for facts first, then hands
them to the writer.

Run it as in [Running](../README.md#running), with `--app-dir 02_supervisor_team`.

Try: *"Write a short paragraph on why solar needs storage."* or *"What do the notes say
about offshore wind?"*

You can also attach files (the paperclip, or drop them on the message box): images, PDFs
and text files, kept in `.files/` beside the backend.

What to look at:

- `workers: [researcher, writer]` gives the editor an `ask_researcher` and an
  `ask_writer` tool. The chat shows each delegation and, inside it, the worker's own tool
  calls.
- The `writer` has its own `model:` block (a warmer temperature). Any agent can use a
  different model or provider this way.
- Add your own Markdown files to `notes/` (lines starting with `- ` are searchable).
