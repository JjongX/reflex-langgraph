from reflex_langgraph.ui.reducer import apply_event, items_from_history


def test_tokens_then_message_merge_into_one_bubble():
    items, pending = [], []
    apply_event(items, pending, "token", {"agent": "a", "message_id": "m1", "text": "Hel"})
    apply_event(items, pending, "token", {"agent": "a", "message_id": "m1", "text": "lo"})
    assert [i.text for i in items] == ["Hello"]
    apply_event(items, pending, "message", {"agent": "a", "message_id": "m1", "text": "Hello!"})
    assert [(i.kind, i.text) for i in items] == [("assistant", "Hello!")]


def test_resumed_tool_call_updates_instead_of_duplicating():
    items, pending = [], []
    call = {"agent": "op", "call_id": "c1", "tool": "submit", "args": {"job": "j1"}}
    apply_event(items, pending, "tool_call", call)
    apply_event(items, pending, "approval_required", {
        "interrupt_id": "i1", "agent": "op",
        "requests": [{"tool": "submit", "args": {"job": "j1"}, "allowed": ["approve"]}]})
    apply_event(items, pending, "approval_required", {"interrupt_id": "i1", "requests": []})
    assert len(pending) == 1 and pending[0].requests[0].allowed == ["approve"]

    apply_event(items, pending, "tool_call", {**call, "args": {"job": "j2"}})  # edited on resume
    apply_event(items, pending, "tool_result", {"call_id": "c1", "tool": "submit", "output": "ok"})
    tools = [i for i in items if i.kind == "tool"]
    assert len(tools) == 1
    assert '"j2"' in tools[0].args and tools[0].output == "ok" and tools[0].status == "done"


def test_run_finished_statuses():
    items, pending = [], []
    apply_event(items, pending, "tool_call", {"call_id": "c", "tool": "t", "args": {}})
    assert apply_event(items, pending, "run_finished", {"status": "interrupted"}) == "interrupted"
    assert items[0].status == "running"  # still waiting on approval
    assert apply_event(items, pending, "run_finished", {"status": "cancelled"}) == "cancelled"
    assert items[0].status == "done" and items[-1].text == "Stopped."


def test_history_rebuild():
    items = items_from_history([
        {"role": "user", "text": "go"},
        {"role": "assistant", "agent": "sup", "id": "m1", "text": "",
         "tool_calls": [{"id": "c1", "tool": "ask_op", "args": {"task": "x"}}]},
        {"role": "tool", "call_id": "c1", "text": "done"},
        {"role": "assistant", "agent": "sup", "id": "m2", "text": "All set", "tool_calls": []},
    ])
    assert [(i.kind, i.delegate_to, i.output or i.text) for i in items] == [
        ("user", "", "go"), ("tool", "op", "done"), ("assistant", "", "All set")]


def test_queued_approval_and_decisions():
    from reflex_langgraph.ui.reducer import approval_from, build_decisions
    a = approval_from({"interrupt_id": "i", "agent": "op", "can_decide": False, "thread_id": "t",
                       "thread_title": "run it", "requested_by": "carol",
                       "requested_by_name": "Carol",
                       "requests": [{"tool": "submit", "args": {"n": 1}, "allowed": ["edit"]}]})
    assert (a.can_decide, a.thread_id, a.requested_by) == (False, "t", "Carol")

    assert build_decisions(a, "approve") == ([{"type": "approve"}], "")
    assert build_decisions(a, "reject", "no") == ([{"type": "reject", "message": "no"}], "")
    a.requests[0].edited_args = '{"n": 2}'
    assert build_decisions(a, "edit")[0] == [{"type": "edit",
                                              "edited_action": {"name": "submit", "args": {"n": 2}}}]
    a.requests[0].edited_args = "{not json"
    decisions, error = build_decisions(a, "edit")
    assert decisions is None and "not valid JSON" in error


def test_reasoning_folds_when_a_tool_call_starts_and_is_timed():
    items, pending = [], []
    apply_event(items, pending, "reasoning", {"message_id": "m1", "agent": "boss",
                                              "text": "Ask the operator."}, now=10.0)
    assert items[0].thinking and items[0].thought_seconds == 0
    apply_event(items, pending, "tool_call", {"call_id": "c1", "tool": "ask_operator",
                                              "agent": "boss", "args": {},
                                              "delegate_to": "operator"}, now=13.4)
    assert (items[0].thinking, items[0].thought_seconds) == (False, 3)


def test_only_agents_other_than_the_main_one_are_named():
    items, pending = [], []
    apply_event(items, pending, "tool_call", {"call_id": "c1", "tool": "ask_operator",
                                              "agent": "boss", "args": {},
                                              "delegate_to": "operator"})
    apply_event(items, pending, "message", {"message_id": "w1", "agent": "operator",
                                            "text": "Started."})
    apply_event(items, pending, "message", {"message_id": "m2", "agent": "boss",
                                            "text": "It has started."})
    assert [(i.agent, i.show_agent) for i in items] == [
        ("boss", False), ("operator", True), ("boss", False)]
