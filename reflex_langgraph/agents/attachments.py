"""
The attachments submodule shows the model the files users attached to their messages.

Messages keep only their files' ids and names (see :mod:`reflex_langgraph.files`).
:class:`AttachmentMiddleware`, on every agent, rewrites each model call:

* the message being answered (the last user message) gets its files as content: images
  and PDFs as standard ``image`` / ``file`` blocks when the agent's model reads that kind,
  text files as text;
* every message with files also gets one label per file (name, type, size, id), so the
  agent can name a file to a tool or hand it to a worker;
* older messages keep only the labels, so a file is not paid for again on every turn.

A file the model cannot take (a binary or compressed data format such as Parquet or
``.csv.gz``, a kind its model does not read, or one over the size limits) is described in
its label instead, with a hint to use a tool.
"""

from __future__ import annotations

import base64
from typing import FrozenSet, Iterable, List, Optional

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import HumanMessage

from ..files import ATTACHMENTS_KEY, FILES_KEY, FileRecord, file_kind

#: Kinds of file the models of these providers read, by default (text always works).
PROVIDER_READS = {
    "google_genai": frozenset({"image", "pdf"}),
    "google_vertexai": frozenset({"image", "pdf"}),
    "anthropic": frozenset({"image", "pdf"}),
    "openai": frozenset({"image", "pdf"}),
    "azure_openai": frozenset({"image", "pdf"}),
    "bedrock_converse": frozenset({"image", "pdf"}),
}


def default_reads(model_id: str) -> FrozenSet[str]:
    """
    Return the kinds of file a model reads, by its provider.

    Parameters
    ----------
    model_id : str
        ``"<provider>:<model>"``.

    Returns
    -------
    frozenset of str
        ``image`` and ``pdf`` for providers known to take them; empty otherwise (text
        files always go in as text).

    Examples
    --------

    >>> from reflex_langgraph.agents.attachments import default_reads
    >>> sorted(default_reads('google_genai:gemini-2.5-flash'))
    ['image', 'pdf']
    >>> sorted(default_reads('ollama:llama3'))
    []
    """
    return PROVIDER_READS.get(model_id.split(":", 1)[0], frozenset())


def _size(n: int) -> str:
    for unit in ("bytes", "KB", "MB"):
        if n < 1024 or unit == "MB":
            return f"{n} {unit}" if unit == "bytes" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} MB"


def file_label(ref: dict, note: str = "") -> str:
    """
    Describe an attached file for the model, with the id tools take.

    Parameters
    ----------
    ref : dict
        The file as the message keeps it: ``id``, ``name``, ``type``, ``size``.
    note : str, default: ''
        Why its contents are not included, if they are not.

    Returns
    -------
    str
        One line.

    Examples
    --------

    >>> from reflex_langgraph.agents.attachments import file_label
    >>> file_label({'id': 'f_1', 'name': 'q3.pdf', 'type': 'application/pdf', 'size': 2048})
    '[Attached file: q3.pdf (application/pdf, 2.0 KB), file id f_1]'
    """
    text = (f"[Attached file: {ref['name']} ({ref['type']}, {_size(ref['size'])}), "
            f"file id {ref['id']}")
    return text + (f". {note}]" if note else "]")


class AttachmentMiddleware(AgentMiddleware):
    """
    Agent middleware that shows the model the files attached to messages.

    :func:`~reflex_langgraph.agents.build_graph` adds it to every agent. The files come
    from the server's store, which it puts in the run's config.

    Parameters
    ----------
    reads : iterable of str, optional
        Kinds of file the agent's model reads as content: ``image``, ``pdf``. By
        default, none (text files are always included as text).
    max_inline_mb : float, default: 15
        Images and PDFs, together, sent with one model call; larger ones are described
        instead (providers cap a request's size, e.g. about 20 MB for Gemini).
    max_text_chars : int, default: 50000
        Characters of one text file included as text; a longer one is described instead.
    """

    def __init__(self, reads: Optional[Iterable[str]] = None, max_inline_mb: float = 15,
                 max_text_chars: int = 50_000):
        super().__init__()
        self.reads = frozenset(reads or ())
        self.max_inline_bytes = int(max_inline_mb * (1 << 20))
        self.max_text_chars = max_text_chars

    @staticmethod
    def _store():
        """The server's file store and the conversation, from the run's config."""
        try:
            from langgraph.config import get_config
            configurable = get_config().get("configurable") or {}
        except RuntimeError:  # called outside a run
            return None, None
        return configurable.get(FILES_KEY), configurable.get("thread_id")

    def _content(self, message: HumanMessage, refs: List[dict], include: bool,
                 store, thread_id) -> list:
        text = message.content if isinstance(message.content, str) else ""
        labels, blocks, budget = [], [], self.max_inline_bytes
        for ref in refs:
            if not include:
                labels.append(file_label(ref))
                continue
            record: Optional[FileRecord] = store.get(thread_id, ref["id"]) if store else None
            kind = ref.get("kind") or file_kind(ref["type"])
            if record is None:
                labels.append(file_label(ref, "Its contents are not available"))
            elif kind == "data":
                labels.append(file_label(ref, "A data file the model does not read directly; "
                                              "tools can read it by its id"))
            elif kind == "text":
                body = store.read(record).decode("utf-8", errors="replace")
                if len(body) > self.max_text_chars:
                    labels.append(file_label(ref, "Too long to include here; read it with a "
                                                  "tool if one is available"))
                else:
                    labels.append(file_label(ref, "Contents below"))
                    name = ref["name"]
                    blocks.append({"type": "text",
                                   "text": f"--- {name} ---\n{body}\n--- end of {name} ---"})
            elif kind not in self.reads:
                labels.append(file_label(ref, f"This model cannot read {kind} files directly"))
            elif record.size > budget:
                labels.append(file_label(ref, "Too large to send to the model with this message"))
            else:
                budget -= record.size
                data = base64.b64encode(store.read(record)).decode("ascii")
                labels.append(file_label(ref))
                blocks.append({"type": "image" if kind == "image" else "file",
                               "base64": data, "mime_type": ref["type"]})
        intro = "\n".join(labels)
        return [{"type": "text", "text": f"{text}\n\n{intro}" if text else intro}, *blocks]

    def _expand(self, request):
        messages = request.messages
        marked = [i for i, m in enumerate(messages)
                  if isinstance(m, HumanMessage) and m.additional_kwargs.get(ATTACHMENTS_KEY)]
        if not marked:
            return request
        store, thread_id = self._store()
        # Contents only for the message being answered: the last user message.
        current = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
        out = list(messages)
        for i in marked:
            message = messages[i]
            refs = message.additional_kwargs[ATTACHMENTS_KEY]
            out[i] = message.model_copy(update={"content": self._content(
                message, refs, include=(i == current), store=store, thread_id=thread_id)})
        return request.override(messages=out)

    def wrap_model_call(self, request, handler):
        """Call the model with the newest message's files as content, the rest as labels."""
        return handler(self._expand(request))

    async def awrap_model_call(self, request, handler):
        """Async form of :meth:`wrap_model_call`."""
        return await handler(self._expand(request))
