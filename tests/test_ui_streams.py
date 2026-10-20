"""A stream belongs to the screen that started it: after New chat, its events no longer
touch the screen. (The Reflex state is replaced by a small stand-in; the
method under test is the real one.)"""

import pytest

from reflex_langgraph.ui.state import ChatState

pytestmark = pytest.mark.anyio

consume = ChatState.__dict__["_consume"]


class Screen:
    """Just what ``ChatState._consume`` touches."""

    def __init__(self):
        self.items, self.pending, self.running, self.error, self._gen = [], [], True, "", 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def reset(self):  # what ChatState._reset does to these fields
        self._gen += 1
        self.items, self.pending, self.running, self.error = [], [], False, ""


def token(text):
    return "token", {"message_id": "m1", "agent": "a", "text": text}


async def test_events_after_a_reset_do_not_reach_the_new_screen():
    screen, closed = Screen(), []

    async def events():
        try:
            yield token("old reply, ")
            screen.reset()            # New chat while the reply streams
            screen.running = True     # ... and a new message was sent on the new screen
            yield token("still old")
            yield "run_finished", {"status": "completed"}
        finally:
            closed.append(True)

    async for _ in consume(screen, events(), 0):
        pass
    assert screen.items == []          # nothing from the old reply
    assert screen.running is True      # the new stream's flag is left alone
    assert closed == [True]            # the old stream is closed, not left open


async def test_a_current_stream_updates_the_screen():
    screen = Screen()

    async def events():
        yield token("hello")
        yield "run_finished", {"status": "completed"}

    async for _ in consume(screen, events(), 0):
        pass
    assert [i.text for i in screen.items] == ["hello"] and screen.running is False
