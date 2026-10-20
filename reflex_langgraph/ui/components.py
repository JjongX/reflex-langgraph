"""
The components submodule builds the page: the conversation list (with rename, delete and
search), the chat, approval cards, the approvals queue and the jobs panel, and
:func:`agent_page`, which puts them together.
"""

from __future__ import annotations

from typing import Optional

import reflex as rx

from ..files import FORMATS, accepted_types
from .reducer import ApprovalView, ChatItem, RequestView
from .state import (CHAT_SCROLL_ID, JUMP_BUTTON_ID, RENAME_INPUT_ID, SCROLL_TO_BOTTOM,
                    SEARCH_BUTTON_ID, SEARCH_SHORTCUT, SEND_ON_ENTER, TIMEZONE_OFFSET,
                    WATCH_SCROLL, ApprovalsState, Attachment, ChatState, JobDetail, JobRow,
                    JobsState, SessionState, StepRow, ThreadRow)

#: Upload id (and HTML id of its drop zone) of the message box's files.
UPLOAD_ID = "rlg-files"
#: Script that opens the file picker of the message box's drop zone (the paperclip).
OPEN_FILE_PICKER = (f"document.querySelector('#{UPLOAD_ID} input[type=file]')?.click()")

#: CSS for what moves or folds: the shimmering "Thinking", the waiting dot, folding
#: sections, the message box, and actions shown when a reply or a conversation is hovered.
PAGE_CSS = """
@keyframes rlg-shimmer { from { background-position: 150% 0; } to { background-position: -50% 0; } }
.rlg-shimmer {
  background: linear-gradient(90deg, var(--gray-10) 30%, var(--gray-12) 50%, var(--gray-10) 70%);
  background-size: 200% 100%; -webkit-background-clip: text; background-clip: text;
  color: transparent; animation: rlg-shimmer 1.8s linear infinite;
}
@keyframes rlg-pulse { 0%, 100% { opacity: 0.4; transform: scale(0.8); } 50% { opacity: 1; transform: scale(1); } }
.rlg-dot {
  width: 0.8rem; height: 0.8rem; border-radius: 50%; background: var(--gray-12);
  animation: rlg-pulse 1.3s ease-in-out infinite;
}
.rlg-fold > summary { list-style: none; cursor: pointer; width: fit-content; }
.rlg-fold > summary::-webkit-details-marker { display: none; }
.rlg-fold .rlg-chevron { transition: transform 0.15s ease; }
.rlg-fold[open] > summary .rlg-chevron { transform: rotate(90deg); }
.rlg-msg .rlg-actions { opacity: 0; transition: opacity 0.15s ease; }
.rlg-msg:hover .rlg-actions, .rlg-msg .rlg-actions.rlg-shown { opacity: 1; }
.rlg-row .rlg-row-menu { opacity: 0; }
.rlg-row:hover .rlg-row-menu, .rlg-row.rlg-current .rlg-row-menu,
.rlg-row .rlg-row-menu[data-state="open"] { opacity: 1; }
@media (hover: none) { .rlg-msg .rlg-actions, .rlg-row .rlg-row-menu { opacity: 1; } }
.rlg-rename {
  width: 100%; font: inherit; font-size: var(--font-size-2); color: var(--gray-12);
  padding: 0.4rem 0.6rem; border-radius: 0.5rem; border: 1px solid var(--accent-8);
  background: var(--color-background); outline: none;
}
.rlg-input {
  border: none; outline: none; background: transparent; resize: none; width: 100%;
  font: inherit; font-size: 1rem; line-height: 1.5; color: var(--gray-12);
  max-height: 30vh; padding: 0.4rem 0;
}
.rlg-input::placeholder { color: var(--gray-9); }
.rlg-composer:not(.rlg-has-files):has(textarea:placeholder-shown) .rlg-send {
  opacity: 0.35; pointer-events: none;
}
.rlg-answer { line-height: 1.7; }
"""

#: Badge colour for each job and step status.
STATUS_COLORS = {
    "pass": "green", "succeeded": "green", "done": "blue", "running": "blue",
    "starting": "blue", "fingerprinting": "blue", "fail": "red", "failed": "red",
    "error": "red", "died": "red", "completed_with_failures": "orange",
    "interrupted": "orange", "cancelled": "gray", "not_reached": "gray", "pending": "gray",
    "unmanaged": "gray",
}


def status_badge(status) -> rx.Component:
    """
    Show a job's or step's status as a coloured badge.

    Parameters
    ----------
    status : str or rx.Var
        The status.

    Returns
    -------
    rx.Component
        The badge.
    """
    return rx.badge(status, color_scheme=rx.match(
        status, *[(k, v) for k, v in STATUS_COLORS.items()], "gray"), variant="soft")


# --- chat ------------------------------------------------------------------------------

def _shown(condition) -> rx.Var:
    """CSS display for a condition. Inside rx.foreach, prefer this to rx.cond around
    state-bound components: it keeps the hook count identical for every item."""
    return rx.cond(condition, "flex", "none")


def _agent_label(agent) -> rx.Component:
    return rx.cond(agent != "", rx.badge(agent, variant="outline", size="1"), rx.fragment())


def _agent_name(item: ChatItem) -> rx.Component:
    """The agent's name above its reply: only for agents other than the main one."""
    return rx.text(item.agent, size="1", weight="medium", color=rx.color("gray", 10),
                   display=rx.cond(item.show_agent, "block", "none"))


def _file_chip(name, *extra) -> rx.Component:
    return rx.hstack(rx.icon("file_text", size=15, color=rx.color("gray", 10), flex_shrink="0"),
                     rx.text(name, size="1", overflow="hidden", text_overflow="ellipsis",
                             white_space="nowrap", max_width="14rem"),
                     *extra, align="center", spacing="1", padding="0.3rem 0.6rem",
                     border=f"1px solid {rx.color('gray', 5)}", border_radius="0.75rem",
                     background=rx.color("gray", 1))


def _user_message(item: ChatItem) -> rx.Component:
    return rx.vstack(
        rx.hstack(rx.foreach(item.files, lambda name: _file_chip(name)), wrap="wrap",
                  justify="end", spacing="2",
                  display=rx.cond(item.files.length() > 0, "flex", "none")),
        rx.box(rx.text(item.text, white_space="pre-wrap"), background=rx.color("gray", 3),
               padding="0.6rem 1.1rem", border_radius="1.5rem", max_width="75%",
               display=rx.cond(item.text != "", "block", "none")),
        align="end", spacing="2", width="100%")


@rx.memo
def markdown_text(text: rx.Var[str]) -> rx.Component:
    """
    Render Markdown, as a component of its own.

    ``rx.markdown`` calls React hooks, which must not run conditionally inside a parent's
    ``rx.foreach`` or ``rx.match``; its own component keeps that safe.

    Parameters
    ----------
    text : rx.Var of str
        The Markdown.

    Returns
    -------
    rx.Component
        The rendered text.
    """
    return rx.markdown(text)


def _fold_summary(*children) -> rx.Component:
    return rx.el.summary(
        *children, rx.icon("chevron_right", size=15, class_name="rlg-chevron",
                           color=rx.color("gray", 10)),
        display="flex", align_items="center", gap="0.35rem")


def _reasoning(item: ChatItem) -> rx.Component:
    """The model's thinking: open while it streams, folded to "Thought for Ns" after."""
    label = rx.cond(
        item.thinking,
        rx.text("Thinking", as_="span", class_name="rlg-shimmer", weight="medium"),
        rx.text(rx.cond(item.thought_seconds > 0,
                        "Thought for " + item.thought_seconds.to_string() + "s", "Thought"),
                as_="span", color=rx.color("gray", 11)))
    return rx.el.details(
        _fold_summary(label),
        rx.box(markdown_text(text=item.reasoning), color=rx.color("gray", 11),
               font_size="0.9rem", border_left=f"2px solid {rx.color('gray', 5)}",
               padding_left="1rem", margin="0.4rem 0 0.25rem 0.3rem"),
        open=item.thinking, class_name="rlg-fold",
        display=rx.cond(item.reasoning != "", "block", "none"))


def _retry_icon() -> rx.Component:
    return rx.tooltip(
        rx.icon_button(rx.icon("rotate_ccw", size=15), variant="ghost", color_scheme="gray",
                       size="1", on_click=ChatState.regenerate, aria_label="Try again"),
        content="Try again")


def _retry_button() -> rx.Component:
    """Try again, under a chat that ended without a reply (an error, a stop, a block)."""
    return rx.button(rx.icon("rotate_ccw", size=14), "Try again", size="1", variant="soft",
                     color_scheme="gray", radius="full", on_click=ChatState.regenerate)


def _assistant_message(item: ChatItem, retry) -> rx.Component:
    # A turn that only calls tools has reasoning but no text: only the reasoning shows.
    # The last reply keeps its actions in view, with Try again, as in ChatGPT.
    return rx.box(
        _agent_name(item),
        _reasoning(item),
        rx.box(markdown_text(text=item.text), class_name="rlg-answer",
               display=rx.cond(item.text != "", "block", "none")),
        rx.hstack(
            rx.tooltip(
                rx.icon_button(rx.icon("copy", size=15), variant="ghost", color_scheme="gray",
                               size="1", on_click=[rx.set_clipboard(item.text),
                                                   rx.toast("Copied", duration=1500)],
                               aria_label="Copy"),
                content="Copy"),
            rx.box(_retry_icon(), display=rx.cond(retry, "flex", "none")),
            class_name=rx.cond(retry, "rlg-actions rlg-shown", "rlg-actions"),
            display=rx.cond(item.text != "", "flex", "none"), spacing="1",
            margin_top="0.2rem"),
        class_name="rlg-msg", width="100%")


def _blocked(item: ChatItem) -> rx.Component:
    return rx.box(
        _agent_name(item),
        rx.callout(item.text, icon="shield_alert", color_scheme="orange", width="100%"),
        width="100%")


def _tool_item(item: ChatItem) -> rx.Component:
    """One line per tool call, like "Used countdown_submit"; it unfolds to its details."""
    running = item.status == "running"
    label = rx.cond(
        item.delegate_to != "",
        rx.cond(running, "Asking " + item.delegate_to + "…", "Asked " + item.delegate_to),
        rx.cond(running, "Using " + item.tool + "…", "Used " + item.tool))
    return rx.el.details(
        _fold_summary(
            rx.cond(running, rx.spinner(size="1"),
                    rx.icon("check", size=14, color=rx.color("gray", 10))),
            rx.text(rx.cond(item.show_agent, item.agent + " · ", ""), label, as_="span",
                    size="2", color=rx.color("gray", 11))),
        rx.vstack(
            rx.text("Arguments", size="1", color_scheme="gray"),
            rx.code_block(item.args, language="json", width="100%"),
            rx.cond(item.output != "", rx.vstack(
                rx.text("Result", size="1", color_scheme="gray"),
                rx.code_block(item.output, language="log", width="100%", wrap_long_lines=True),
                width="100%", spacing="1")),
            width="100%", spacing="1", padding="0.5rem 0 0 1.4rem"),
        class_name="rlg-fold", width="100%")


def chat_item(item: ChatItem, retry=False) -> rx.Component:
    """
    Show one item of the chat, by its kind.

    Parameters
    ----------
    item : ChatItem
        The item.
    retry : bool or rx.Var of bool, default: False
        Whether it is the last reply, which offers to answer the message again.

    Returns
    -------
    rx.Component
        A message (with its reasoning), a tool call, a block notice, an error or a notice.
    """
    return rx.match(
        item.kind,
        ("user", _user_message(item)),
        ("assistant", _assistant_message(item, retry)),
        ("tool", _tool_item(item)),
        ("blocked", _blocked(item)),
        ("error", rx.callout(item.text, icon="triangle_alert", color_scheme="red", width="100%")),
        rx.text(item.text, size="1", color_scheme="gray", align="center", width="100%"),
    )


def _request_view(approval: ApprovalView, req: RequestView, index, state) -> rx.Component:
    editable = req.allowed.contains("edit") & approval.can_decide
    return rx.vstack(
        rx.hstack(rx.text("Tool"), rx.code(req.tool), spacing="2", align="center"),
        rx.cond(req.description != "", rx.text(req.description, size="2", color_scheme="gray")),
        # Both are always rendered (toggled with CSS) so every list item calls the same hooks.
        rx.text_area(value=req.edited_args, width="100%", rows="8", font_family="monospace",
                     on_change=lambda v: state.set_edited_args(approval.interrupt_id, index, v),
                     display=_shown(editable)),
        rx.box(rx.code_block(req.args, language="json", width="100%"), width="100%",
               display=_shown(~editable)),
        width="100%", spacing="2")


def approval_card(approval: ApprovalView, state=ChatState) -> rx.Component:
    """
    Show one approval, with its decisions.

    Parameters
    ----------
    approval : ApprovalView
        The approval.
    state : type, default: ChatState
        Who answers it: :class:`~reflex_langgraph.ui.ChatState` in the conversation,
        :class:`~reflex_langgraph.ui.ApprovalsState` in the reviewers' queue.

    Returns
    -------
    rx.Component
        The card.
    """
    allowed = approval.requests[0].allowed

    def button(decision: str, label: str, color: str, icon: str):
        return rx.button(rx.icon(icon, size=16), label, color_scheme=color,
                         on_click=state.decide(approval.interrupt_id, decision),
                         disabled=state.running,
                         display=_shown(allowed.contains(decision) & approval.can_decide))

    return rx.card(
        rx.vstack(
            rx.hstack(rx.icon("shield_check"), rx.heading("Approval required", size="4"),
                      _agent_label(approval.agent), align="center", spacing="2", wrap="wrap"),
            rx.text(rx.cond(approval.requested_by != "", "Requested by " + approval.requested_by,
                            ""),
                    rx.cond(approval.thread_title != "", " · " + approval.thread_title, ""),
                    size="1", color_scheme="gray", display=_shown(approval.thread_id != "")),
            rx.foreach(approval.requests, lambda r, i: _request_view(approval, r, i, state)),
            rx.input(placeholder="Reason, if rejecting (optional)", value=state.reject_note,
                     on_change=state.set_reject_note, width="100%",
                     display=_shown(allowed.contains("reject") & approval.can_decide)),
            rx.hstack(button("approve", "Approve", "green", "check"),
                      button("edit", "Approve with edits", "blue", "pencil"),
                      button("reject", "Reject", "red", "x"), spacing="2", wrap="wrap"),
            rx.callout("Waiting for a reviewer to approve this. You can leave this page; the "
                       "conversation continues once they decide.", icon="hourglass", size="1",
                       color_scheme="amber", width="100%", display=_shown(~approval.can_decide)),
            width="100%", spacing="3"),
        width="100%", border=f"1px solid {rx.color('amber', 7)}")


def _pending(approval: ApprovalView) -> rx.Component:
    # One-argument wrappers: rx.foreach passes the index to a second parameter.
    return approval_card(approval, ChatState)


def _attachment_chip(chip: Attachment) -> rx.Component:
    uploading = chip.status == "uploading"
    return _file_chip(
        chip.name,
        rx.text(chip.size, size="1", color=rx.color("gray", 10), white_space="nowrap"),
        rx.cond(uploading, rx.spinner(size="1"),
                rx.icon_button(rx.icon("x", size=12), size="1", variant="ghost",
                               color_scheme="gray", type="button", aria_label="Remove",
                               on_click=ChatState.remove_attachment(chip.key))))


def _composer(placeholder: str, attachments: bool = False, max_file_mb: int = 100,
              max_files: int = 30, file_types: Optional[dict] = None) -> rx.Component:
    """The message box: one rounded field with a round send (or stop) button.

    The browser keeps what is typed until it is sent: no keystroke goes to the server, so
    typing never waits for it and input methods (Korean, Japanese...) compose as usual. Enter
    sends, Shift+Enter starts a new line (:data:`~reflex_langgraph.ui.state.SEND_ON_ENTER`),
    and the send button fades while the box is empty. With ``attachments``, a paperclip
    (or dropping files on the box) uploads files, shown as chips until the message is sent.
    """
    waiting = ChatState.pending.length() > 0
    has_files = ChatState.attachments.length() > 0
    send = rx.icon_button(rx.icon("arrow_up", size=18), type="submit", radius="full",
                          size="2", high_contrast=True, class_name="rlg-send",
                          aria_label="Send")
    stop = rx.icon_button(rx.icon("square", size=13, fill="currentColor"), type="button",
                          radius="full", size="2", high_contrast=True, on_click=ChatState.stop,
                          aria_label="Stop")
    clip = rx.icon_button(rx.icon("paperclip", size=17), type="button", variant="ghost",
                          color_scheme="gray", radius="full", size="2", disabled=waiting,
                          aria_label="Attach files", on_click=rx.call_script(OPEN_FILE_PICKER))
    field = rx.el.textarea(name="message",
                           placeholder=rx.cond(waiting, "Answer the approval above to continue.",
                                               placeholder),
                           auto_height=True, rows="1", class_name="rlg-input", disabled=waiting)
    box = rx.vstack(
        rx.hstack(rx.foreach(ChatState.attachments, _attachment_chip), wrap="wrap", spacing="2",
                  width="100%", display=rx.cond(has_files, "flex", "none"))
        if attachments else rx.fragment(),
        rx.hstack(clip if attachments else rx.fragment(), field,
                  rx.cond(ChatState.running, stop, send), align="end", spacing="2",
                  width="100%"),
        class_name=rx.cond(has_files, "rlg-composer rlg-has-files", "rlg-composer"),
        spacing="2", width="100%", background=rx.color("gray", 2),
        border=f"1px solid {rx.color('gray', 5)}", border_radius="1.75rem",
        padding=f"0.55rem 0.55rem 0.55rem {'0.55rem' if attachments else '1.25rem'}",
        box_shadow="0 2px 12px rgba(0, 0, 0, 0.06)")
    form = rx.form(box, on_submit=ChatState.send, reset_on_submit=False, width="100%")
    if not attachments:
        return form
    accept = file_types or accepted_types(FORMATS)
    # Dropping files on the box uploads them; the paperclip opens the same zone's picker.
    return rx.upload.root(
        form, id=UPLOAD_ID, multiple=True, no_click=True, no_keyboard=True,
        max_files=max_files, max_size=max_file_mb << 20, accept=accept,
        on_drop=ChatState.attach(rx.upload_files(upload_id=UPLOAD_ID)), width="100%")


def _error_banner() -> rx.Component:
    return rx.cond(ChatState.error != "",
                   rx.callout(ChatState.error, icon="info", color_scheme="red", width="100%",
                              margin_bottom="0.75rem"))


def chat_panel(placeholder: str = "Ask anything", greeting: str = "What can I help with?",
               attachments: bool = False, max_file_mb: int = 100, max_files: int = 30,
               file_types: Optional[dict] = None) -> rx.Component:
    """
    Show the chat, with its messages, pending approvals, errors and the message box.

    An empty chat shows only a greeting with the message box in the middle; once it has
    messages, they scroll above the box. A reply streaming in follows the end of the chat
    while the user is there, and a button jumps back down when they scrolled up.

    Parameters
    ----------
    placeholder : str, default: 'Ask anything'
        Hint text in the message box.
    greeting : str, default: 'What can I help with?'
        Shown above the message box in an empty chat.
    attachments : bool, default: False
        Let users attach files (the backend must keep them; its ``/health`` says so).
    max_file_mb : int, default: 100
        The largest file the box accepts, in MB.
    max_files : int, default: 30
        The most files it accepts at once.
    file_types : dict, optional
        Type to extensions, as the backend's ``/health`` reports them (``accept``). By
        default, the package's own formats (images, PDF, text).

    Returns
    -------
    rx.Component
        The panel.
    """
    def composer():
        return _composer(placeholder, attachments, max_file_mb, max_files, file_types)

    empty = (ChatState.items.length() == 0) & (ChatState.pending.length() == 0) \
        & ~ChatState.running & ~ChatState.remote_running
    start = rx.box(
        rx.vstack(
            rx.heading(greeting, size="7", weight="medium", align="center"),
            rx.box(_error_banner(), composer(), width="100%"),
            width="100%", max_width="48rem", spacing="6", align="center"),
        display="flex", align_items="center", justify_content="center", height="100%",
        width="100%", padding="0 1rem 12vh")
    jump = rx.icon_button(
        rx.icon("arrow_down", size=18), id=JUMP_BUTTON_ID, radius="full", size="2",
        variant="surface", color_scheme="gray", on_click=rx.call_script(SCROLL_TO_BOTTOM),
        position="absolute", bottom="1rem", left="50%", transform="translateX(-50%)",
        display="none", box_shadow="0 2px 10px rgba(0, 0, 0, 0.15)")
    conversation = rx.box(
        rx.box(
            rx.box(
                rx.vstack(
                    rx.foreach(ChatState.items, lambda item, i: chat_item(
                        item, (i == ChatState.items.length() - 1) & ChatState.retry_on_reply)),
                    rx.foreach(ChatState.pending, _pending),
                    rx.cond(ChatState.awaiting_reply, rx.box(class_name="rlg-dot")),
                    rx.cond(ChatState.can_retry & ~ChatState.retry_on_reply, _retry_button()),
                    width="100%", max_width="48rem", margin="0 auto", spacing="5",
                    padding="1.5rem 1rem 2rem"),
                id=CHAT_SCROLL_ID, on_mount=rx.call_script(WATCH_SCROLL), height="100%",
                width="100%", overflow_y="auto"),
            jump,
            position="relative", flex="1", min_height="0", width="100%"),
        rx.box(_error_banner(), composer(), width="100%", max_width="48rem",
               margin="0 auto", padding="0 1rem 1.25rem"),
        display="flex", flex_direction="column", height="100%", width="100%")
    return rx.box(
        # Picks up a reviewer's decision and the run that follows it (no-op otherwise).
        rx.moment(interval=3000, on_change=ChatState.poll, display="none"),
        rx.cond(empty, start, conversation),
        on_mount=rx.call_script(SEND_ON_ENTER), height="100%", width="100%")


# --- conversation list -----------------------------------------------------------------

def _row_menu(thread: ThreadRow) -> rx.Component:
    """The row's "..." menu: Rename, Delete."""
    return rx.menu.root(
        rx.menu.trigger(
            rx.icon_button(rx.icon("ellipsis", size=16), variant="ghost", color_scheme="gray",
                           size="1", class_name="rlg-row-menu", aria_label="Chat options")),
        rx.menu.content(
            rx.menu.item(rx.icon("pencil", size=14), "Rename",
                         on_select=ChatState.start_rename(thread.thread_id)),
            rx.menu.item(rx.icon("trash_2", size=14), "Delete", color_scheme="red",
                         on_select=ChatState.ask_delete(thread.thread_id)),
            # Keep the focus off the trigger, or the rename field would lose it at once.
            on_close_auto_focus=rx.prevent_default, size="2"))


def _rename_field(thread: ThreadRow) -> rx.Component:
    """The title as a field: Enter or leaving it saves, Escape keeps the old title."""
    return rx.form(
        rx.el.input(name="title", default_value=thread.title, id=RENAME_INPUT_ID,
                    max_length=200, class_name="rlg-rename", on_blur=ChatState.rename_blur,
                    on_key_down=ChatState.rename_key),
        on_submit=ChatState.save_rename, reset_on_submit=False, width="100%")


def thread_row(thread: ThreadRow) -> rx.Component:
    """
    Show one conversation in the list, under its group's heading; a click opens it.

    Its "..." menu renames it in place or deletes it, after a confirmation.

    Parameters
    ----------
    thread : ThreadRow
        The conversation.

    Returns
    -------
    rx.Component
        The row: its title, whether it runs or waits for an approval, and its menu.
    """
    current = thread.thread_id == ChatState.thread_id
    row = rx.hstack(
        rx.hstack(
            rx.text(thread.title, size="2", overflow="hidden", text_overflow="ellipsis",
                    white_space="nowrap", flex="1", min_width="0"),
            rx.cond(thread.running, rx.spinner(size="1")),
            rx.cond(thread.pending & ~thread.running,
                    rx.tooltip(rx.icon("circle_pause", size=14, color=rx.color("orange", 9)),
                               content="Waiting for approval")),
            on_click=ChatState.open_thread(thread.thread_id), flex="1", min_width="0",
            padding="0.45rem 0 0.45rem 0.6rem", align="center", spacing="2",
            cursor="pointer"),
        _row_menu(thread),
        class_name=rx.cond(current, "rlg-row rlg-current", "rlg-row"),
        background=rx.cond(current, rx.color("gray", 4), "transparent"),
        _hover={"background": rx.color("gray", 4)}, border_radius="0.5rem",
        padding_right="0.35rem", width="100%", align="center", spacing="1")
    return rx.box(
        rx.text(thread.group, size="1", weight="medium", color=rx.color("gray", 10),
                padding="1rem 0.6rem 0.35rem",
                display=rx.cond(thread.first_in_group, "block", "none")),
        rx.cond(thread.thread_id == ChatState.renaming_id, _rename_field(thread), row),
        width="100%")


def _nav_item(icon: str, label: str, on_click, **props) -> rx.Component:
    style = {"_hover": {"background": rx.color("gray", 4)}, "border_radius": "0.5rem",
             "padding": "0.5rem 0.6rem", "width": "100%", "align": "center", "spacing": "2",
             "cursor": "pointer", **props}
    return rx.hstack(rx.icon(icon, size=18), rx.text(label, size="2", weight="medium"),
                     on_click=on_click, **style)


def _delete_dialog() -> rx.Component:
    return rx.alert_dialog.root(
        rx.alert_dialog.content(
            rx.alert_dialog.title("Delete chat?"),
            rx.alert_dialog.description(
                "This will delete ", rx.text(ChatState.delete_title, as_="span", weight="bold"),
                ".", size="2"),
            rx.hstack(
                rx.alert_dialog.cancel(rx.button("Cancel", variant="soft", color_scheme="gray",
                                                 radius="full")),
                rx.alert_dialog.action(rx.button(
                    "Delete", color_scheme="red", radius="full",
                    on_click=ChatState.confirm_delete(ChatState.delete_id))),
                justify="end", spacing="3", margin_top="1.25rem"),
            max_width="26rem"),
        open=ChatState.delete_id != "", on_open_change=ChatState.close_delete)


def _search_result(row: ThreadRow) -> rx.Component:
    return rx.box(
        rx.text(row.group, size="1", weight="medium", color=rx.color("gray", 10),
                padding="0.75rem 0.75rem 0.3rem",
                display=rx.cond(row.first_in_group, "block", "none")),
        rx.hstack(
            rx.icon("message_circle", size=16, color=rx.color("gray", 10), flex_shrink="0"),
            rx.box(
                rx.text(row.title, size="2", overflow="hidden", text_overflow="ellipsis",
                        white_space="nowrap"),
                rx.text(row.snippet, size="1", color=rx.color("gray", 10), overflow="hidden",
                        text_overflow="ellipsis", white_space="nowrap",
                        display=rx.cond(row.snippet != "", "block", "none")),
                flex="1", min_width="0"),
            on_click=ChatState.open_search_result(row.thread_id),
            _hover={"background": rx.color("gray", 3)}, border_radius="0.6rem",
            padding="0.6rem 0.75rem", width="100%", align="center", spacing="3",
            cursor="pointer"),
        width="100%")


def _search_dialog() -> rx.Component:
    """Search chats, as in ChatGPT: the recent ones until something is typed."""
    searching = ChatState.search_query.strip() != ""
    return rx.dialog.root(
        rx.dialog.content(
            rx.dialog.title("Search chats", display="none"),
            rx.dialog.description("Find a chat by its title or messages.", display="none"),
            rx.hstack(
                rx.icon("search", size=18, color=rx.color("gray", 10), flex_shrink="0"),
                rx.debounce_input(
                    rx.el.input(placeholder="Search chats...", value=ChatState.search_query,
                                on_change=ChatState.search, class_name="rlg-input"),
                    debounce_timeout=250),
                rx.dialog.close(rx.icon_button(rx.icon("x", size=18), variant="ghost",
                                               color_scheme="gray", aria_label="Close")),
                align="center", spacing="3", padding="0.75rem 1rem",
                border_bottom=f"1px solid {rx.color('gray', 4)}"),
            rx.box(
                rx.cond(searching, rx.fragment(),
                        _nav_item("square_pen", "New chat", ChatState.search_new_chat,
                                  padding="0.6rem 0.75rem")),
                rx.foreach(ChatState.search_results, _search_result),
                rx.cond(ChatState.search_error != "",
                        rx.text(ChatState.search_error, size="2", color_scheme="red",
                                padding="1rem 0.75rem")),
                rx.cond(searching & (ChatState.search_results.length() == 0)
                        & (ChatState.search_error == ""),
                        rx.text("No chats found.", size="2", color=rx.color("gray", 10),
                                padding="1rem 0.75rem")),
                max_height="60vh", overflow_y="auto", padding="0.5rem"),
            padding="0", max_width="40rem", width="calc(100vw - 2rem)"),
        open=ChatState.search_open, on_open_change=ChatState.set_search_open)


def threads_panel(poll_ms: int = 5000) -> rx.Component:
    """
    Show New chat, Search chats and the conversations, grouped by day, newest first.

    Only the conversations this user may see are listed. Search (also Ctrl+K, or Cmd+K)
    looks through titles and messages; each conversation's menu renames or deletes it.

    Parameters
    ----------
    poll_ms : int, default: 5000
        How often it refreshes, in milliseconds.

    Returns
    -------
    rx.Component
        The panel.
    """
    return rx.vstack(
        _nav_item("square_pen", "New chat", ChatState.new_chat),
        _nav_item("search", "Search chats", ChatState.open_search, id=SEARCH_BUTTON_ID),
        rx.cond(ChatState.threads_error != "",
                rx.text(ChatState.threads_error, size="1", color_scheme="red",
                        padding="0 0.6rem")),
        # A plain box, not scroll_area: its viewport lets rows grow wider than the column,
        # which defeats the ellipsis on long titles.
        rx.box(
            rx.vstack(rx.foreach(ChatState.threads, thread_row), width="100%", spacing="0"),
            flex="1", min_height="0", width="100%", overflow_y="auto", overflow_x="hidden"),
        # Picks up conversations started elsewhere and runs that end in the background.
        rx.moment(interval=poll_ms, on_change=ChatState.refresh_threads, display="none"),
        _search_dialog(),
        _delete_dialog(),
        on_mount=rx.call_script(SEARCH_SHORTCUT),
        width="100%", height="100%", spacing="1", padding="0.75rem 0.5rem")


# --- approvals queue -------------------------------------------------------------------

def _queued(approval: ApprovalView) -> rx.Component:
    return approval_card(approval, ApprovalsState)


def approvals_panel(poll_ms: int = 5000) -> rx.Component:
    """
    Show the approvals queue, with requests waiting in other people's conversations.

    Render it for reviewers only (``SessionState.can_review``); the server refuses
    everyone else anyway.

    Parameters
    ----------
    poll_ms : int, default: 5000
        How often it refreshes, in milliseconds.

    Returns
    -------
    rx.Component
        The panel.
    """
    return rx.vstack(
        rx.hstack(rx.heading("Approvals", size="4"),
                  rx.cond(ApprovalsState.items.length() > 0,
                          rx.badge(ApprovalsState.items.length(), color_scheme="amber")),
                  rx.spacer(),
                  rx.icon_button(rx.icon("refresh_cw", size=14), size="1", variant="ghost",
                                 on_click=ApprovalsState.refresh),
                  width="100%", align="center"),
        rx.moment(interval=poll_ms, on_change=ApprovalsState.refresh, display="none"),
        rx.cond(ApprovalsState.error != "",
                rx.callout(ApprovalsState.error, color_scheme="red", size="1", width="100%")),
        rx.cond(ApprovalsState.notice != "",
                rx.text(ApprovalsState.notice, size="1", color_scheme="gray")),
        rx.cond(ApprovalsState.items.length() > 0,
                rx.vstack(rx.foreach(ApprovalsState.items, _queued), width="100%", spacing="3"),
                rx.text("No pending approvals.", size="2", color_scheme="gray")),
        width="100%", spacing="3", padding="1em")


# --- jobs ------------------------------------------------------------------------------

def _job_row(job: JobRow) -> rx.Component:
    return rx.table.row(
        rx.table.cell(rx.text(job.job_id, weight="medium")),
        rx.table.cell(status_badge(job.status)),
        on_click=JobsState.select(job.job_id), cursor="pointer",
        background=rx.cond(JobsState.selected == job.job_id, rx.color("accent", 3), "transparent"))


def _step_row(step: StepRow) -> rx.Component:
    return rx.hstack(status_badge(step.status), rx.text(step.label, size="2"),
                     rx.spacer(), rx.text(step.started, size="1", color_scheme="gray"),
                     width="100%", align="center")


def job_detail(detail: JobDetail = JobsState.detail) -> rx.Component:
    """
    Show the selected job, with its status, progress and steps.

    Parameters
    ----------
    detail : JobDetail, default: JobsState.detail
        The job.

    Returns
    -------
    rx.Component
        The detail.
    """
    return rx.cond(
        detail.job_id != "",
        rx.vstack(
            rx.hstack(rx.heading(detail.job_id, size="3"), status_badge(detail.status),
                      rx.spacer(),
                      rx.cond(rx.Var.create(["running", "starting", "fingerprinting"])
                              .contains(detail.status),
                              rx.button("Cancel", size="1", color_scheme="red", variant="soft",
                                        on_click=JobsState.cancel(detail.job_id))),
                      width="100%", align="center"),
            rx.cond(detail.total > 0, rx.vstack(
                rx.progress(value=detail.done, max=detail.total, width="100%"),
                rx.text(detail.done.to_string() + " / " + detail.total.to_string() + " steps",
                        rx.cond(detail.current != "", " · now: " + detail.current, ""),
                        size="1", color_scheme="gray"),
                width="100%", spacing="1")),
            rx.cond(detail.approved_by != "",
                    rx.text("Approved by ", detail.approved_by, size="1", color_scheme="gray")),
            rx.scroll_area(rx.vstack(rx.foreach(detail.steps, _step_row), width="100%",
                                     spacing="1"),
                           max_height="45vh", type="auto"),
            width="100%", spacing="2"),
        rx.text("Select a job.", size="2", color_scheme="gray"))


def jobs_panel(poll_ms: int = 5000) -> rx.Component:
    """
    Show the jobs this user may see, and the selected one in detail.

    Parameters
    ----------
    poll_ms : int, default: 5000
        How often it refreshes, in milliseconds.

    Returns
    -------
    rx.Component
        The panel.
    """
    return rx.vstack(
        rx.hstack(rx.heading("Jobs", size="4"), rx.spacer(),
                  rx.icon_button(rx.icon("refresh_cw", size=14), size="1", variant="ghost",
                                 on_click=JobsState.refresh),
                  width="100%", align="center"),
        rx.moment(interval=poll_ms, on_change=JobsState.refresh, display="none"),
        rx.cond(JobsState.jobs_error != "",
                rx.callout(JobsState.jobs_error, color_scheme="red", size="1")),
        rx.cond(JobsState.jobs.length() > 0,
                rx.scroll_area(rx.table.root(rx.table.body(rx.foreach(JobsState.jobs, _job_row)),
                                             size="1", width="100%"),
                               max_height="25vh", type="auto"),
                rx.text("No jobs yet.", size="2", color_scheme="gray")),
        rx.separator(),
        job_detail(),
        width="100%", spacing="3", padding="1em")


# --- page --------------------------------------------------------------------------------

def agent_page(title: str, *, subtitle: str = "", show_jobs: bool = True,
               show_threads: bool = True, placeholder: str = "Ask anything",
               greeting: str = "What can I help with?", poll_ms: int = 5000,
               header_extra: Optional[rx.Component] = None, attachments: bool = False,
               max_file_mb: int = 100, max_files: int = 30,
               file_types: Optional[dict] = None) -> rx.Component:
    """
    Build the whole page, laid out like ChatGPT.

    The conversations on the left, the chat in the middle, and a side column with the
    approvals queue (reviewers only) and the jobs panel on the right. On a phone they stack.

    .. code-block:: python

        import reflex as rx
        from reflex_langgraph.ui import agent_page, on_load_events

        app = rx.App()
        app.add_page(lambda: agent_page("My agents", show_jobs=False),
                     on_load=on_load_events())

    Parameters
    ----------
    title : str
        Shown at the top.
    subtitle : str, default: ''
        Shown next to the title.
    show_jobs : bool, default: True
        Show the jobs panel.
    show_threads : bool, default: True
        Show the conversation list.
    placeholder : str, default: 'Ask anything'
        Hint text in the message box.
    greeting : str, default: 'What can I help with?'
        Shown above the message box in an empty chat.
    poll_ms : int, default: 5000
        How often it refreshes, in milliseconds.
    header_extra : rx.Component, optional
        Shown at the top right.
    attachments, max_file_mb, max_files, file_types
        As in :func:`chat_panel`.

    Returns
    -------
    rx.Component
        The page.
    """
    border = f"1px solid {rx.color('gray', 4)}"
    header = rx.hstack(
        rx.hstack(rx.text(title, size="4", weight="medium", white_space="nowrap"),
                  rx.text(subtitle, size="2", color=rx.color("gray", 10), overflow="hidden",
                          text_overflow="ellipsis", white_space="nowrap")
                  if subtitle else rx.fragment(),
                  align="baseline", spacing="2", min_width="0"),
        rx.spacer(),
        header_extra or rx.fragment(),
        rx.color_mode.button(size="1"),
        width="100%", padding="0.75rem 1rem", align="center")
    main = rx.box(
        header,
        rx.box(chat_panel(placeholder, greeting, attachments, max_file_mb, max_files,
                                   file_types), flex="1", min_height="0", width="100%"),
        display="flex", flex_direction="column", flex="1", min_width="0",
        height=rx.breakpoints(initial="85vh", md="100%"))
    threads = rx.box(
        threads_panel(poll_ms), background=rx.color("gray", 2), flex_shrink="0",
        width=rx.breakpoints(initial="100%", md="260px"),
        height=rx.breakpoints(initial="30vh", md="100%"),
        border_bottom=rx.breakpoints(initial=border, md="none"))
    side = rx.box(
        rx.vstack(
            rx.cond(SessionState.can_review,
                    rx.vstack(approvals_panel(poll_ms), rx.separator(), width="100%",
                              spacing="0")),
            jobs_panel(poll_ms) if show_jobs else rx.fragment(),
            width="100%", spacing="0"),
        flex_shrink="0", width=rx.breakpoints(initial="100%", md="380px"),
        border_left=rx.breakpoints(initial="none", md=border),
        border_top=rx.breakpoints(initial=border, md="none"), overflow_y="auto",
        height=rx.breakpoints(initial="auto", md="100%"))
    body = rx.flex(
        threads if show_threads else rx.fragment(),
        main,
        # Without jobs, the side column exists only for reviewers.
        side if show_jobs else rx.cond(SessionState.can_review, side),
        direction=rx.breakpoints(initial="column", md="row"),
        height=rx.breakpoints(initial="auto", md="100vh"), width="100%")
    return rx.fragment(rx.el.style(PAGE_CSS), body)


def on_load_events():
    """
    Return the events the page needs when it loads.

    Returns
    -------
    list
        Load the session, the conversation, its list, the jobs and the approvals queue,
        and read the browser's timezone (to group the list by day).
    """
    return [SessionState.check_session, ChatState.reload, ChatState.refresh_threads,
            JobsState.refresh, ApprovalsState.refresh,
            rx.call_script(TIMEZONE_OFFSET, callback=SessionState.set_tz_offset)]


def add_agent_page(app: rx.App, title: str, *, route: str = "/", **kwargs) -> None:
    """
    Add :func:`agent_page` to an app, with the events it needs when it loads.

    .. code-block:: python

        import reflex as rx
        from reflex_langgraph.ui import add_agent_page

        app = rx.App()
        add_agent_page(app, "My agents", subtitle="Ask about your jobs")

    Parameters
    ----------
    app : rx.App
        The app.
    title : str
        The page's title.
    route : str, default: '/'
        Where the page is served.
    **kwargs
        Passed to :func:`agent_page`.
    """
    app.add_page(lambda: agent_page(title, **kwargs), route=route, title=title,
                 on_load=on_load_events())
