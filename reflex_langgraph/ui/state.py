"""
The state submodule holds the Reflex states behind the page: the conversation and its list,
the jobs panel, and the approvals queue.

Every call to the backend happens in the Reflex server, never in the browser.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import List

import reflex as rx

from .client import DEFAULT_API_URL, ApiError, BackendClient
from .reducer import (ApprovalView, ChatItem, apply_event, approval_from, build_decisions,
                      drop_reply, history_group, items_from_history)


#: Shown when the backend cannot be reached.
UNREACHABLE = "The server is not reachable."
#: HTML id of the chat's scrolling area.
CHAT_SCROLL_ID = "rlg-chat-scroll"
#: HTML id of the button that jumps to the end of the chat.
JUMP_BUTTON_ID = "rlg-jump"
#: Script that scrolls the chat to its end (after sending, or opening a conversation).
SCROLL_TO_BOTTOM = (f"requestAnimationFrame(() => {{ const e = document.getElementById("
                    f"'{CHAT_SCROLL_ID}'); if (e) {{ e.dataset.stick = '1'; "
                    "e.scrollTop = e.scrollHeight; } })")
#: Script that keeps up with a streaming reply, unless the user scrolled up to read.
FOLLOW_BOTTOM = (f"(() => {{ const e = document.getElementById('{CHAT_SCROLL_ID}'); "
                 "if (e && e.dataset.stick !== '0') e.scrollTop = e.scrollHeight; })()")
#: Script, run once when the chat appears: remembers whether the user is at the end of the
#: chat (``data-stick``), and shows the jump button only when they are not.
WATCH_SCROLL = f"""(() => {{
  const e = document.getElementById('{CHAT_SCROLL_ID}');
  if (!e || e.dataset.watched) return;
  e.dataset.watched = '1';
  e.dataset.stick = '1';
  const away = () => e.scrollHeight - e.scrollTop - e.clientHeight;
  const button = (show) => {{
    const b = document.getElementById('{JUMP_BUTTON_ID}');
    if (b) b.style.display = show ? 'flex' : 'none';
  }};
  e.addEventListener('scroll', () => {{
    e.dataset.stick = away() < 40 ? '1' : '0';
    button(away() > 160);
  }}, {{passive: true}});
  new MutationObserver(() => button(e.dataset.stick === '0' && away() > 160))
    .observe(e, {{childList: true, subtree: true, characterData: true}});
}})()"""
#: Script that returns the browser's ``Date.getTimezoneOffset()``.
TIMEZONE_OFFSET = "new Date().getTimezoneOffset()"
#: Script, run once when the chat appears: Enter in the message box sends it, Shift+Enter
#: starts a new line, and an Enter that only finishes a syllable in an input method (Korean,
#: Japanese...) only finishes it. The box keeps its own text: nothing is sent to the server
#: while typing, so input methods compose as usual.
SEND_ON_ENTER = """(() => {
  if (window.__rlgEnterToSend) return;
  window.__rlgEnterToSend = true;
  document.addEventListener('keydown', (e) => {
    const box = e.target;
    if (e.key !== 'Enter' || e.shiftKey || !(box instanceof HTMLTextAreaElement)
        || box.name !== 'message') return;
    if (e.isComposing || e.keyCode === 229) return;
    e.preventDefault();
    if (!e.repeat && box.form) box.form.requestSubmit();
  });
})()"""
#: Script that empties the message box once its message is sent (resetting the form also
#: shrinks the box back to one line).
CLEAR_MESSAGE_BOX = ("(() => { const e = document.querySelector(\"textarea[name='message']\"); "
                     "if (e && e.form) e.form.reset(); })()")
#: HTML id of the field that renames a conversation in the list.
RENAME_INPUT_ID = "rlg-rename"
#: Script that focuses the rename field, with its text selected, once it has rendered.
FOCUS_RENAME = (f"(() => {{ let n = 0; const go = () => {{ const e = document.getElementById("
                f"'{RENAME_INPUT_ID}'); if (e) {{ e.focus(); e.select(); }} "
                "else if (n++ < 30) requestAnimationFrame(go); }; go(); })()")
#: HTML id of the list's "Search chats" entry.
SEARCH_BUTTON_ID = "rlg-search-open"
#: Script, run once when the list appears: Ctrl+K (Cmd+K on a Mac) opens the search, as in
#: ChatGPT.
SEARCH_SHORTCUT = f"""(() => {{
  if (window.__rlgSearchKey) return;
  window.__rlgSearchKey = true;
  document.addEventListener('keydown', (e) => {{
    if ((e.ctrlKey || e.metaKey) && !e.altKey && !e.shiftKey && e.key.toLowerCase() === 'k') {{
      const b = document.getElementById('{SEARCH_BUTTON_ID}');
      if (b) {{ e.preventDefault(); b.click(); }}
    }}
  }});
}})()"""


class SessionState(rx.State):
    """
    The backend's address and what its policy allows, shared by every page.

    There is no login: the UI sends no credentials.

    Attributes
    ----------
    api_url : str
        The backend's URL.
    can_review : bool
        Whether the server's policy lets this user work from the approvals queue.
    tz_offset_minutes : int
        The browser's timezone offset (``Date.getTimezoneOffset()``), so the conversation
        list's days start at the user's midnight.
    """

    api_url: str = DEFAULT_API_URL
    can_review: bool = False
    tz_offset_minutes: int = 0

    def client(self) -> BackendClient:
        """
        Return a client for the backend.

        Returns
        -------
        BackendClient
            The client.
        """
        return BackendClient(self.api_url)

    async def _load_session(self) -> None:
        """What the server's policy allows this (anonymous) user."""
        try:
            info = await self.client().session()
        except Exception:
            return
        self.can_review = bool(info.get("can_review"))

    @rx.event
    async def check_session(self):
        """Ask the server what its policy allows this user (``/session``)."""
        await self._load_session()

    @rx.event
    def set_tz_offset(self, value):
        """
        Keep the browser's timezone offset, then regroup the conversation list.

        Parameters
        ----------
        value : int
            ``Date.getTimezoneOffset()``: UTC minus local time, in minutes.
        """
        try:
            self.tz_offset_minutes = int(value)
        except (TypeError, ValueError):
            return
        return ChatState.refresh_threads


@dataclass
class Attachment:
    """
    A file on the message being written, shown as a chip.

    Attributes
    ----------
    key : str
        The chip's own id.
    file_id : str
        The file's id on the server, once uploaded.
    name : str
        Its name.
    size : str
        Its size, for display.
    status : str
        ``uploading`` or ``ready``.
    """

    key: str = ""
    file_id: str = ""
    name: str = ""
    size: str = ""
    status: str = ""


def _size_label(n) -> str:
    n = n or 0
    if n < 1024:
        return f"{n} B"
    if n < 1 << 20:
        return f"{n / 1024:.0f} KB"
    return f"{n / (1 << 20):.1f} MB"


@dataclass
class ThreadRow:
    """
    One conversation in the sidebar.

    Attributes
    ----------
    thread_id : str
        Its id.
    title : str
        Its title; ``New chat`` until its first message.
    updated_at : str
        Its last message or decision.
    running : bool
        Whether a run is going on in it.
    pending : bool
        Whether it waits for an approval decision.
    group : str
        Its heading in the list: ``Today``, ``Yesterday``, ``Previous 7 days``, ...
    first_in_group : bool
        Whether the heading goes above it.
    snippet : str
        In search results, the text around the match when it is in a message.
    """

    thread_id: str = ""
    title: str = ""
    updated_at: str = ""
    running: bool = False
    pending: bool = False
    group: str = ""
    first_in_group: bool = False
    snippet: str = ""


def thread_rows(rows: List[dict], tz_offset_minutes: int = 0) -> List[ThreadRow]:
    """
    Make the list's rows from conversations, with a heading above each day's first one.

    Parameters
    ----------
    rows : list of dict
        Conversations, newest activity first (see
        :class:`~reflex_langgraph.server.schemas.ThreadSummary`).
    tz_offset_minutes : int, default: 0
        The browser's ``Date.getTimezoneOffset()``.

    Returns
    -------
    list of ThreadRow
        The rows; an untitled conversation (no message yet) is called ``New chat``.

    Examples
    --------

    >>> from reflex_langgraph.ui.state import thread_rows
    >>> rows = [{'thread_id': 't2', 'title': 'Budget', 'updated_at': '2026-10-14T09:00:00'},
    ...         {'thread_id': 't1', 'title': 'Plan', 'updated_at': '2026-10-14T08:00:00'}]
    >>> [(r.title, r.first_in_group) for r in thread_rows(rows)]
    [('Budget', True), ('Plan', False)]
    """
    out, last_group = [], None
    for r in rows:
        group = history_group(r["updated_at"], tz_offset_minutes=tz_offset_minutes)
        out.append(ThreadRow(thread_id=r["thread_id"], title=r.get("title") or "New chat",
                             updated_at=r["updated_at"], running=r.get("running", False),
                             pending=r.get("pending", False), group=group,
                             first_in_group=group != last_group,
                             snippet=r.get("snippet") or ""))
        last_group = group
    return out


class ChatState(SessionState):
    """
    The conversation on the screen, and the list of conversations.

    Attributes
    ----------
    thread_id : str
        The conversation shown; empty for a new chat, created by its first message. Kept
        in the browser, so a reload returns to it.
    threads : list of ThreadRow
        The conversation list, newest activity first.
    threads_error : str
        Why the list could not be loaded, if it could not.
    items : list of ChatItem
        The chat.
    pending : list of ApprovalView
        Approvals waiting in this conversation.
    running : bool
        Whether this page is streaming a run.
    remote_running : bool
        Whether a run this page did not start is going on (e.g. after a reviewer's
        decision).
    error : str
        Shown above the message box.
    reject_note : str
        The note to send with a rejection.
    renaming_id : str
        The conversation whose title is being edited in the list.
    delete_id : str
        The conversation waiting for the user to confirm its deletion.
    delete_title : str
        Its title, for the confirmation.
    search_open : bool
        Whether the search is open.
    search_query : str
        What is being searched for.
    search_results : list of ThreadRow
        The conversations found; the recent ones while the query is empty.
    search_error : str
        Why the search failed, if it did.
    attachments : list of Attachment
        Files attached to the message being written.
    waiting_for_reviewer : bool
        Whether an approval here waits for someone else to answer it (computed).
    can_retry : bool
        Whether the last message can be answered again (computed).
    retry_on_reply : bool
        Whether that offer goes on the last reply rather than under the chat (computed).
    """

    thread_id: str = rx.LocalStorage("", name="rlg_thread", sync=True)
    threads: List[ThreadRow] = []
    threads_error: str = ""
    items: List[ChatItem] = []
    pending: List[ApprovalView] = []
    running: bool = False
    remote_running: bool = False
    error: str = ""
    reject_note: str = ""
    renaming_id: str = ""
    delete_id: str = ""
    delete_title: str = ""
    search_open: bool = False
    search_query: str = ""
    search_results: List[ThreadRow] = []
    search_error: str = ""
    attachments: List[Attachment] = []
    #: Which screen a stream belongs to. Bumped by New chat; a stream started under an older
    #: value stops updating the screen (its run goes on on the server). Without it, an old
    #: reply keeps writing into the new conversation.
    _gen: int = 0

    @rx.var
    def waiting_for_reviewer(self) -> bool:
        """bool : Whether an approval here waits for someone else to answer it."""
        return any(not p.can_decide for p in self.pending)

    @rx.var
    def awaiting_reply(self) -> bool:
        """bool : Whether a run goes on and nothing is streaming into the chat right now."""
        if not (self.running or self.remote_running):
            return False
        last = self.items[-1] if self.items else None
        return not (last and last.kind == "assistant" and (last.text or last.thinking))

    @rx.var
    def can_retry(self) -> bool:
        """bool : Whether the last message can be answered again (nothing runs or waits)."""
        return bool(self.thread_id) and not (self.running or self.remote_running
                                             or self.pending) \
            and any(i.kind == "user" for i in self.items)

    @rx.var
    def retry_on_reply(self) -> bool:
        """bool : Whether the offer to answer again goes on the last reply."""
        last = self.items[-1] if self.items else None
        return self.can_retry and bool(last and last.kind == "assistant" and last.text)

    @rx.event
    def set_reject_note(self, value: str):
        """
        Keep the note to send with a rejection.

        Parameters
        ----------
        value : str
            The note.
        """
        self.reject_note = value

    def _reset(self) -> None:
        self._gen += 1
        self.thread_id, self.items, self.pending, self.error = "", [], [], ""
        self.running = self.remote_running = False
        self.attachments = []

    async def _reload(self) -> None:
        if not self.thread_id:
            self.items, self.pending = [], []
            return
        try:
            hist = await self.client().history(self.thread_id)
        except ApiError as exc:
            if exc.status == 404:  # e.g. the backend restarted and lost in-memory threads
                return self._reset()
            return self._handle_api_error(exc)
        except Exception:
            self.error = UNREACHABLE
            return
        self.items = items_from_history(hist["messages"])
        self.pending = [approval_from(p) for p in hist["pending"]]
        if self.error == UNREACHABLE or self.error.startswith("Connection lost"):
            self.error = ""  # the server answered; a connectivity error is stale now
        if not self.running:
            self.remote_running = hist.get("running", False)

    @rx.event
    def new_chat(self):
        """
        Start a new chat, on an empty screen.

        The conversation is created when its first message is sent, so the list never
        fills with empty chats.
        """
        self._reset()
        return ChatState.refresh_threads

    @rx.event
    async def open_thread(self, thread_id: str):
        """
        Show another conversation.

        A reply still streaming in the current one keeps running on the server; it just
        stops updating this screen.

        Parameters
        ----------
        thread_id : str
            The conversation to show.
        """
        if thread_id == self.thread_id:
            return
        self._reset()
        self.thread_id = thread_id
        await self._reload()
        return rx.call_script(SCROLL_TO_BOTTOM)

    @rx.event
    async def refresh_threads(self):
        """Reload the conversation list."""
        try:
            rows = await self.client().threads()
        except Exception:  # noqa: BLE001 - the chat shows connectivity errors already
            self.threads_error = "Could not load conversations."
            return
        self.threads_error = ""
        # Untitled ones have no message yet (e.g. left over from an older UI): only the
        # current one is worth showing.
        shown = [r for r in rows if r["title"] or r["thread_id"] == self.thread_id]
        self.threads = thread_rows(shown, self.tz_offset_minutes)

    # --- renaming, deleting and searching conversations -----------------------------

    @rx.event
    def start_rename(self, thread_id: str):
        """
        Edit a conversation's title in the list.

        Parameters
        ----------
        thread_id : str
            The conversation.
        """
        self.renaming_id = thread_id
        return rx.call_script(FOCUS_RENAME)

    @rx.event
    async def save_rename(self, form: dict):
        """
        Save the title typed in the list (Enter).

        Parameters
        ----------
        form : dict
            The submitted form, with the ``title``.
        """
        await self._rename((form or {}).get("title", ""))

    @rx.event
    async def rename_blur(self, value: str):
        """
        Save the title typed in the list when the field loses focus.

        Parameters
        ----------
        value : str
            The title.
        """
        await self._rename(value)

    @rx.event
    def rename_key(self, key: str):
        """
        Stop renaming on Escape, keeping the old title.

        Parameters
        ----------
        key : str
            The key pressed in the field.
        """
        if key == "Escape":
            self.renaming_id = ""

    async def _rename(self, title: str) -> None:
        thread_id, self.renaming_id = self.renaming_id, ""
        title = (title or "").strip()[:200]
        row = next((t for t in self.threads if t.thread_id == thread_id), None)
        if not (thread_id and title) or (row is not None and row.title == title):
            return  # Enter and the blur that follows both arrive; the first one saves
        try:
            await self.client().rename_thread(thread_id, title)
        except Exception:  # noqa: BLE001 - shown in the list
            self.threads_error = "Could not rename the chat."
            return
        for t in self.threads:
            if t.thread_id == thread_id:
                t.title = title
        self.threads = list(self.threads)

    @rx.event
    def ask_delete(self, thread_id: str):
        """
        Ask the user to confirm deleting a conversation.

        Parameters
        ----------
        thread_id : str
            The conversation.
        """
        row = next((t for t in self.threads if t.thread_id == thread_id), None)
        self.delete_id, self.delete_title = thread_id, row.title if row else "this chat"

    @rx.event
    def close_delete(self, value: bool):
        """
        Close the confirmation without deleting.

        Parameters
        ----------
        value : bool
            Whether the confirmation is open; it is closed when false.
        """
        if not value:
            self.delete_id = ""

    @rx.event
    async def confirm_delete(self, thread_id: str):
        """
        Delete a conversation; a run going on in it stops.

        Parameters
        ----------
        thread_id : str
            The conversation.
        """
        self.delete_id = ""
        if not thread_id:
            return
        try:
            await self.client().delete_thread(thread_id)
        except ApiError as exc:
            if exc.status != 404:  # already gone is what the user wanted
                self.threads_error = exc.detail
                return
        except Exception:  # noqa: BLE001 - shown in the list
            self.threads_error = "Could not delete the chat."
            return
        if thread_id == self.thread_id:
            self._reset()
        self.threads = [t for t in self.threads if t.thread_id != thread_id]
        return ChatState.refresh_threads

    @rx.event
    def open_search(self):
        """Open the search, listing the recent conversations."""
        self.search_open, self.search_query, self.search_error = True, "", ""
        self.search_results = list(self.threads)

    @rx.event
    def set_search_open(self, value: bool):
        """
        Open or close the search.

        Parameters
        ----------
        value : bool
            Whether it is open.
        """
        self.search_open = bool(value)

    @rx.event
    async def search(self, query: str):
        """
        Find the conversations whose title or messages contain the query.

        Parameters
        ----------
        query : str
            What to look for; empty lists the recent conversations.
        """
        self.search_query, self.search_error = query, ""
        if not query.strip():
            self.search_results = list(self.threads)
            return
        try:
            rows = await self.client().threads(limit=50, q=query.strip())
        except Exception:  # noqa: BLE001 - shown in the search
            self.search_results, self.search_error = [], "Could not search the chats."
            return
        self.search_results = thread_rows(rows, self.tz_offset_minutes)

    @rx.event
    def open_search_result(self, thread_id: str):
        """
        Close the search and show a conversation it found.

        Parameters
        ----------
        thread_id : str
            The conversation.
        """
        self.search_open = False
        return ChatState.open_thread(thread_id)

    @rx.event
    def search_new_chat(self):
        """Close the search and start a new chat."""
        self.search_open = False
        return ChatState.new_chat

    @rx.event
    async def reload(self):
        """Reload the conversation shown (or clear the screen for a new chat)."""
        await self._reload()

    @rx.event
    async def poll(self):
        """
        Follow what someone else does in this conversation.

        That is a reviewer answering its approval, and the run that follows; otherwise
        nothing happens.
        """
        if self.running or not self.thread_id:
            return
        if self.waiting_for_reviewer or self.remote_running:
            await self._reload()
            return JobsState.refresh

    def _handle_api_error(self, exc: ApiError) -> None:
        if exc.status == 401:
            self.error = ("The backend requires authentication, which this UI does not "
                          "have; run it without an auth hook.")
        else:
            self.error = exc.detail

    @rx.event
    def set_edited_args(self, interrupt_id: str, index: int, value: str):
        """
        Keep the arguments typed to edit a request.

        Parameters
        ----------
        interrupt_id : str
            The approval.
        index : int
            The request, within the approval.
        value : str
            The arguments, as JSON.
        """
        for p in self.pending:
            if p.interrupt_id == interrupt_id and 0 <= index < len(p.requests):
                p.requests[index].edited_args = value
        self.pending = list(self.pending)

    @rx.event
    def send(self, form: dict | None = None):
        """
        Send the message and stream the run it starts.

        The message box keeps its text until the message is accepted here, so a message
        sent while a reply is still running stays in the box.

        Parameters
        ----------
        form : dict, optional
            The submitted form, with the ``message``.
        """
        text = ((form or {}).get("message") or "").strip()
        if any(a.status == "uploading" for a in self.attachments):
            self.error = "Wait until the files finish uploading."
            return
        files = [a for a in self.attachments if a.file_id]
        if not (text or files) or self.running or self.pending:
            return
        self.items.append(ChatItem(kind="user", text=text, files=[a.name for a in files]))
        self.attachments = []
        self.running, self.error = True, ""
        return [rx.call_script(CLEAR_MESSAGE_BOX), rx.call_script(SCROLL_TO_BOTTOM),
                ChatState.stream_message(text, [a.file_id for a in files], self._gen)]

    @rx.event
    async def attach(self, files: list[rx.UploadFile]):
        """
        Upload files to the conversation, as chips on the message being written.

        The conversation is created first if it is new. Each file shows as a chip while it
        uploads; a file the server refuses is dropped, with the reason shown.

        Parameters
        ----------
        files : list of rx.UploadFile
            The files chosen or dropped.
        """
        if not files or self.pending:
            return
        if not self.thread_id:
            try:
                self.thread_id = await self.client().new_thread()
            except Exception as exc:  # noqa: BLE001 - shown to the user
                self.error = str(exc)
                return
        chips = [Attachment(key=uuid.uuid4().hex, name=f.name or "file",
                            size=_size_label(f.size), status="uploading") for f in files]
        self.attachments = [*self.attachments, *chips]
        self.error = ""
        yield
        client, errors = self.client(), []
        for upload, chip in zip(files, chips):
            try:
                info = await client.upload_file(self.thread_id, chip.name, upload.file)
            except ApiError as exc:
                errors.append(f"{chip.name}: {exc.detail}")
                self.attachments = [a for a in self.attachments if a.key != chip.key]
            except Exception as exc:  # noqa: BLE001 - shown to the user
                errors.append(f"{chip.name}: {exc}")
                self.attachments = [a for a in self.attachments if a.key != chip.key]
            else:
                for a in self.attachments:
                    if a.key == chip.key:
                        a.file_id, a.status = info["file_id"], "ready"
                self.attachments = list(self.attachments)
            yield
        self.error = "; ".join(errors)

    @rx.event
    async def remove_attachment(self, key: str):
        """
        Take a file off the message being written (and off the server).

        Parameters
        ----------
        key : str
            The chip.
        """
        chip = next((a for a in self.attachments if a.key == key), None)
        if chip is None or chip.status == "uploading":
            return
        self.attachments = [a for a in self.attachments if a.key != key]
        if chip.file_id and self.thread_id:
            try:
                await self.client().delete_file(self.thread_id, chip.file_id)
            except Exception:  # noqa: BLE001 - the file just stays unused on the server
                pass

    @rx.event(background=True)
    async def stream_message(self, text: str, files: list, gen: int):
        """
        Stream a message's run (started by :meth:`send`).

        Parameters
        ----------
        text : str
            The message.
        files : list of str
            Ids of the files it carries.
        gen : int
            The screen it belongs to; it stops updating the screen when that changes.
        """
        async with self:
            if self._gen != gen:
                return
            if not self.thread_id:
                try:
                    self.thread_id = await self.client().new_thread()
                except Exception as exc:  # noqa: BLE001 - shown to the user
                    self.running, self.error = False, str(exc)
                    return
            client, thread_id = self.client(), self.thread_id
        async for script in self._consume(client.send(thread_id, text, files), gen):
            yield script
        yield JobsState.refresh
        yield ChatState.refresh_threads

    @rx.event
    def regenerate(self):
        """Answer the last message again, in place of the reply it got, and stream that run."""
        if not self.thread_id or self.running or self.remote_running or self.pending \
                or not any(i.kind == "user" for i in self.items):
            return
        self.items = drop_reply(self.items)
        self.running, self.error = True, ""
        return [rx.call_script(SCROLL_TO_BOTTOM), ChatState.stream_regenerate(self._gen)]

    @rx.event(background=True)
    async def stream_regenerate(self, gen: int):
        """
        Stream the run that answers the last message again (started by :meth:`regenerate`).

        Parameters
        ----------
        gen : int
            The screen it belongs to.
        """
        async with self:
            if self._gen != gen:
                return
            client, thread_id = self.client(), self.thread_id
        async for script in self._consume(client.regenerate(thread_id), gen):
            yield script
        yield JobsState.refresh
        yield ChatState.refresh_threads

    @rx.event
    def decide(self, interrupt_id: str, decision: str):
        """
        Answer an approval in this conversation and stream the run it resumes.

        Parameters
        ----------
        interrupt_id : str
            The approval.
        decision : str
            ``approve``, ``edit`` or ``reject``.
        """
        approval = next((p for p in self.pending if p.interrupt_id == interrupt_id), None)
        if approval is None or self.running or not approval.can_decide:
            return
        decisions, error = build_decisions(approval, decision, self.reject_note)
        if error:
            self.error = error
            return
        self.pending = [p for p in self.pending if p.interrupt_id != interrupt_id]
        self.items.append(ChatItem(kind="notice", text=f"Decision: {decision}"))
        self.reject_note = ""
        self.running, self.error = True, ""
        return ChatState.stream_decision(interrupt_id, decisions, self._gen)

    @rx.event(background=True)
    async def stream_decision(self, interrupt_id: str, decisions: list, gen: int):
        """
        Stream the run a decision resumes (started by :meth:`decide`).

        Parameters
        ----------
        interrupt_id : str
            The approval.
        decisions : list of dict
            The decisions.
        gen : int
            The screen it belongs to.
        """
        async with self:
            if self._gen != gen:
                return
            client, thread_id = self.client(), self.thread_id
        async for script in self._consume(client.decide(thread_id, interrupt_id, decisions),
                                          gen):
            yield script
        yield JobsState.refresh
        yield ChatState.refresh_threads

    async def _consume(self, events, gen: int):
        """Apply streamed events while the screen is still the one that started the stream
        (``gen``); yields a scroll-to-bottom script at most every 250 ms."""
        last_scroll = 0.0
        try:
            async for event, data in events:
                async with self:
                    if self._gen != gen:
                        return  # the screen moved on; stop watching (the run continues)
                    apply_event(self.items, self.pending, event, data)
                    self.items = list(self.items)
                    self.pending = list(self.pending)
                if event == "run_started":
                    yield ChatState.refresh_threads  # a new title, or the running mark
                now = time.monotonic()
                if now - last_scroll > 0.25 or event in ("approval_required", "run_finished"):
                    last_scroll = now
                    yield rx.call_script(FOLLOW_BOTTOM)
        except ApiError as exc:
            async with self:
                if self._gen == gen:
                    self._handle_api_error(exc)
                    if exc.status != 401:
                        await self._reload()
        except Exception as exc:  # noqa: BLE001 - connection loss etc., shown to the user
            async with self:
                if self._gen == gen:
                    self.error = f"Connection lost: {exc}"
        finally:
            await events.aclose()
            async with self:
                if self._gen == gen:  # else the flag belongs to a newer stream
                    self.running = False

    @rx.event
    async def stop(self):
        """Stop the run going on in this conversation."""
        if not self.thread_id:
            return
        try:
            await self.client().cancel_run(self.thread_id)
        except ApiError as exc:
            self.error = exc.detail


@dataclass
class JobRow:
    """
    One job in the jobs panel.

    Attributes
    ----------
    job_id : str
        Its id.
    spec : str
        Its kind.
    status : str
        Its status.
    submitted_at : str
        When it was submitted.
    """

    job_id: str = ""
    spec: str = ""
    status: str = ""
    submitted_at: str = ""


@dataclass
class StepRow:
    """
    One step of the selected job.

    Attributes
    ----------
    label : str
        The step.
    status : str
        Its status.
    started : str
        When it started.
    """

    label: str = ""
    status: str = ""
    started: str = ""


@dataclass
class JobDetail:
    """
    The selected job.

    Attributes
    ----------
    job_id : str
        Its id.
    status : str
        Its status.
    current : str
        The step running now.
    done : int
        Steps finished.
    total : int
        Steps in all.
    approved_by : str
        Who approved it.
    submitted_at : str
        When it was submitted.
    steps : list of StepRow
        Its steps.
    """

    job_id: str = ""
    status: str = ""
    current: str = ""
    done: int = 0
    total: int = 0
    approved_by: str = ""
    submitted_at: str = ""
    steps: List[StepRow] = field(default_factory=list)


class JobsState(SessionState):
    """
    The jobs panel.

    Attributes
    ----------
    jobs : list of JobRow
        The jobs this user may see, newest first.
    selected : str
        The job shown in detail.
    detail : JobDetail
        Its detail.
    jobs_error : str
        Why the jobs could not be loaded, if they could not.
    """

    jobs: List[JobRow] = []
    selected: str = ""
    detail: JobDetail = JobDetail()
    jobs_error: str = ""

    @rx.event
    async def refresh(self):
        """Reload the jobs, and the selected job's detail."""
        try:
            rows = await self.client().jobs()
        except ApiError as exc:
            self.jobs_error = "" if exc.status in (401, 404) else exc.detail
            return
        except Exception:
            self.jobs_error = UNREACHABLE
            return
        self.jobs_error = ""
        self.jobs = [JobRow(job_id=r["job_id"], spec=r.get("spec") or "",
                            status=r.get("status") or "", submitted_at=r.get("submitted_at") or "")
                     for r in sorted(rows, key=lambda r: r.get("submitted_at") or "", reverse=True)]
        if not self.selected and self.jobs:
            self.selected = self.jobs[0].job_id
        if self.selected:
            await self._load_detail()

    @rx.event
    async def select(self, job_id: str):
        """
        Show a job in detail.

        Parameters
        ----------
        job_id : str
            The job.
        """
        self.selected = job_id
        await self._load_detail()

    async def _load_detail(self):
        try:
            s = await self.client().job(self.selected)
        except Exception:
            self.detail = JobDetail(job_id=self.selected, status="unavailable")
            return
        progress = s.get("progress") or {}
        steps = progress.get("steps", [])
        self.detail = JobDetail(
            job_id=s["job_id"], status=s.get("status", ""), current=progress.get("current") or "",
            done=sum(1 for st in steps if st["status"] in ("pass", "done", "fail")),
            total=len(steps),
            approved_by=(s.get("manifest") or {}).get("approved_by", ""),
            submitted_at=(s.get("manifest") or {}).get("submitted_at", ""),
            steps=[StepRow(label=st["label"], status=st["status"], started=st.get("started") or "")
                   for st in steps])

    @rx.event
    async def cancel(self, job_id: str):
        """
        Stop a running job.

        Parameters
        ----------
        job_id : str
            The job.
        """
        try:
            await self.client().cancel_job(job_id)
        except ApiError as exc:
            self.jobs_error = exc.detail
        await self.refresh()


class ApprovalsState(SessionState):
    """
    The approvals queue, with requests waiting in other people's conversations.

    It is for users the server's policy lets review.

    Attributes
    ----------
    items : list of ApprovalView
        The approvals waiting.
    running : bool
        Whether a decision is being sent.
    reject_note : str
        The note to send with a rejection.
    notice : str
        How the last decision went.
    error : str
        What went wrong, if anything.
    """

    items: List[ApprovalView] = []
    running: bool = False
    reject_note: str = ""
    notice: str = ""
    error: str = ""

    @rx.event
    async def refresh(self):
        """Reload the queue, keeping what the reviewer is editing."""
        if not self.can_review or self.running:
            if not self.can_review:
                self.items = []
            return
        try:
            rows = await self.client().approvals()
        except ApiError as exc:
            self.error = "" if exc.status == 401 else exc.detail
            return
        except Exception:
            self.error = UNREACHABLE
            return
        self.error = ""
        # Keep what the reviewer is editing across polls.
        edits = {(a.interrupt_id, i): r.edited_args
                 for a in self.items for i, r in enumerate(a.requests)}
        items = [approval_from(row) for row in rows]
        for a in items:
            for i, r in enumerate(a.requests):
                r.edited_args = edits.get((a.interrupt_id, i), r.edited_args)
        self.items = items

    @rx.event
    def set_reject_note(self, value: str):
        """
        Keep the note to send with a rejection.

        Parameters
        ----------
        value : str
            The note.
        """
        self.reject_note = value

    @rx.event
    def set_edited_args(self, interrupt_id: str, index: int, value: str):
        """
        Keep the arguments typed to edit a request.

        Parameters
        ----------
        interrupt_id : str
            The approval.
        index : int
            The request, within the approval.
        value : str
            The arguments, as JSON.
        """
        for a in self.items:
            if a.interrupt_id == interrupt_id and 0 <= index < len(a.requests):
                a.requests[index].edited_args = value
        self.items = list(self.items)

    @rx.event
    def decide(self, interrupt_id: str, decision: str):
        """
        Answer a queued approval.

        Parameters
        ----------
        interrupt_id : str
            The approval.
        decision : str
            ``approve``, ``edit`` or ``reject``.
        """
        approval = next((a for a in self.items if a.interrupt_id == interrupt_id), None)
        if approval is None or self.running:
            return
        decisions, error = build_decisions(approval, decision, self.reject_note)
        if error:
            self.error = error
            return
        self.running, self.error, self.reject_note = True, "", ""
        self.notice = f"Sending {decision} for {approval.requested_by or 'the requester'}…"
        return ApprovalsState.send_decision(approval.thread_id, interrupt_id, decision, decisions)

    @rx.event(background=True)
    async def send_decision(self, thread_id: str, interrupt_id: str, decision: str,
                            decisions: list):
        """
        Send a queued approval's decisions (started by :meth:`decide`).

        The run resumes in the requester's conversation; here only its outcome shows.

        Parameters
        ----------
        thread_id : str
            The requester's conversation.
        interrupt_id : str
            The approval.
        decision : str
            The decision, to report.
        decisions : list of dict
            The decisions.
        """
        async with self:
            client = self.client()
        status, errors = "", []
        try:
            # The run resumes in the requester's conversation; they see it there. Here we
            # only need to know how it ended.
            async for event, data in client.decide(thread_id, interrupt_id, decisions):
                if event == "run_finished":
                    status = data.get("status", "")
                elif event == "error":
                    errors.append(data.get("message", "error"))
        except ApiError as exc:
            errors.append(exc.detail)
        except Exception as exc:  # noqa: BLE001 - shown to the reviewer
            errors.append(f"Connection lost: {exc}")
        async with self:
            self.running = False
            if errors:
                self.error = "; ".join(errors)
                self.notice = ""
            else:
                outcome = {"completed": "the agent finished",
                           "interrupted": "the agent is waiting for another approval",
                           "cancelled": "the run was stopped"}.get(status, status)
                self.notice = f"{decision.capitalize()} sent; {outcome}."
        yield ApprovalsState.refresh
        yield JobsState.refresh
