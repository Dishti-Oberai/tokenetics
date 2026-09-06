from dataclasses import replace
from itertools import combinations

from tokenetics.core.logger import InMemoryCostLogger
from tokenetics.core.request import Message, RequestMeta, from_api_kwargs
from tokenetics.stages.context_scheduler import (
    ContextSchedulerStage,
    _classify_turn,
    _dp_knapsack,
    _fix_alternation,
    _greedy_select,
)


def _request(messages, task_type=None, model="claude-sonnet-5", max_tokens=100):
    request = from_api_kwargs(model=model, max_tokens=max_tokens, messages=messages)
    return replace(request, meta=RequestMeta(task_type=task_type, confidence=0.8))


def _long_text(word, n=60):
    return " ".join([word] * n)


# --- per-turn classification ---


def test_classifies_tool_result_structurally():
    m = Message(
        role="user", content=[{"type": "tool_result", "tool_use_id": "1", "content": "ok"}]
    )
    assert _classify_turn(m) == "tool_result"


def test_classifies_error_trace():
    m = Message(
        role="user", content="Traceback (most recent call last):\n  File x\nValueError: bad"
    )
    assert _classify_turn(m) == "error"


def test_classifies_decision_phrasing():
    m = Message(role="assistant", content="Let's go with option B for the migration.")
    assert _classify_turn(m) == "decision"


def test_classifies_code():
    m = Message(role="user", content="```python\ndef f():\n    pass\n```")
    assert _classify_turn(m) == "code"


def test_classifies_small_talk_as_fallback():
    m = Message(role="user", content="Thanks, that's helpful!")
    assert _classify_turn(m) == "small_talk"


# --- pinning + resolution ---


def test_decision_always_pinned_regardless_of_budget():
    stage = ContextSchedulerStage()
    messages = [
        {"role": "assistant", "content": "Let's go with option B for the migration."},
        {"role": "user", "content": _long_text("filler")},
    ]
    request = _request(messages, max_tokens=100)
    result = stage.run(request, {"token_budget": 1}, InMemoryCostLogger())
    assert result.messages[0].content == "Let's go with option B for the migration."


def test_unresolved_error_stays_pinned():
    stage = ContextSchedulerStage()
    messages = [
        {"role": "user", "content": "Traceback (most recent call last):\nValueError: bad"},
        {"role": "assistant", "content": _long_text("filler")},
        {"role": "user", "content": _long_text("padding")},
    ]
    request = _request(messages, max_tokens=100)
    stage.run(request, {"token_budget": 50}, InMemoryCostLogger())
    assert stage.extra.get("pinned_turns") == 1


def test_error_resolved_by_explicit_user_signal_is_unpinned():
    stage = ContextSchedulerStage()
    messages = [
        {"role": "user", "content": "Traceback (most recent call last):\nValueError: bad"},
        {"role": "assistant", "content": "Try X."},
        {"role": "user", "content": "That fixed it, thanks!"},
        {"role": "assistant", "content": _long_text("filler")},
        {"role": "user", "content": _long_text("padding")},
    ]
    request = _request(messages, max_tokens=100)
    # Budget high enough that resolving the error (dropping pinned_turns to
    # 0) still leaves room for the knapsack to keep something -- a budget
    # tight enough to empty the result entirely would trip the alternation
    # fix's fail-open path instead, which is a different behavior (tested
    # separately) from what this test is checking.
    stage.run(request, {"token_budget": 300}, InMemoryCostLogger())
    assert stage.extra.get("pinned_turns") == 0


def test_resolution_from_assistant_does_not_count():
    stage = ContextSchedulerStage()
    messages = [
        {"role": "user", "content": "Traceback (most recent call last):\nValueError: bad"},
        {"role": "assistant", "content": "That fixed it! Great."},
        {"role": "assistant", "content": _long_text("filler")},
        {"role": "user", "content": _long_text("padding")},
    ]
    request = _request(messages, max_tokens=100)
    stage.run(request, {"token_budget": 50}, InMemoryCostLogger())
    assert stage.extra.get("pinned_turns") == 1


# --- token budget source ---


def test_uses_caller_supplied_token_budget():
    stage = ContextSchedulerStage()
    messages = [{"role": "user", "content": "hi"}]
    request = _request(messages, max_tokens=100)
    stage.run(request, {"token_budget": 10}, InMemoryCostLogger())
    assert stage.extra["token_budget"] == 10
    assert stage.extra["token_budget_estimated"] is False


def test_falls_back_to_estimated_budget_when_not_supplied():
    stage = ContextSchedulerStage()
    messages = [{"role": "user", "content": "hi"}]
    request = _request(messages, task_type="code", max_tokens=100)
    stage.run(request, {}, InMemoryCostLogger())
    assert stage.extra["token_budget_estimated"] is True
    assert stage.extra["token_budget"] > 0


# --- DP knapsack correctness ---


def _brute_force_best_value(costs, values, capacity):
    n = len(costs)
    best = 0.0
    for r in range(n + 1):
        for combo in combinations(range(n), r):
            total_cost = sum(costs[i] for i in combo)
            if total_cost <= capacity:
                best = max(best, sum(values[i] for i in combo))
    return best


def test_dp_knapsack_matches_brute_force_on_small_cases():
    indices = [0, 1, 2, 3, 4, 5]
    costs = [3, 2, 4, 1, 5, 2]
    values = [4.0, 3.0, 5.0, 1.0, 6.0, 2.5]
    capacity = 7

    chosen = _dp_knapsack(indices, costs, values, capacity)
    dp_value = sum(values[i] for i in chosen)
    dp_cost = sum(costs[i] for i in chosen)

    assert dp_cost <= capacity
    assert dp_value == _brute_force_best_value(costs, values, capacity)


def test_dp_knapsack_respects_capacity_zero():
    assert _dp_knapsack([0, 1], [1, 2], [1.0, 2.0], 0) == []


# --- greedy fallback ---


def test_greedy_select_respects_capacity():
    indices = [0, 1, 2, 3]
    costs = [3, 3, 3, 3]
    values = [10.0, 1.0, 1.0, 1.0]
    chosen = _greedy_select(indices, costs, values, capacity=5)
    assert chosen == [0]  # best value-per-cost item picked first, then out of room


def test_greedy_fallback_engages_above_the_size_threshold():
    stage = ContextSchedulerStage()
    # Many long, distinct small-talk turns -- pushes candidates * capacity
    # buckets well past _DP_SIZE_THRESHOLD.
    messages = [{"role": "user", "content": _long_text(f"word{i}", n=80)} for i in range(600)]
    request = _request(messages, max_tokens=100)
    stage.run(request, {"token_budget": 60_000}, InMemoryCostLogger())
    assert stage.extra.get("used_greedy_fallback") is True


# --- degraded mode ---


def _alternating_messages(n):
    return [{"role": "user" if i % 2 == 0 else "assistant", "content": f"turn {i}"} for i in range(n)]


def test_degraded_fallback_truncates_to_last_n():
    stage = ContextSchedulerStage()
    messages = _alternating_messages(30)
    request = _request(messages, max_tokens=100)
    result = stage.degraded_fallback(request, {}, InMemoryCostLogger())
    assert len(result.messages) == 20
    assert result.messages[-1].content == "turn 29"
    assert stage.extra["degraded"] is True
    assert stage.extra["mode"] == "last_n_truncation"


def test_degraded_fallback_is_a_no_op_under_the_last_n_limit():
    stage = ContextSchedulerStage()
    messages = _alternating_messages(5)
    request = _request(messages, max_tokens=100)
    result = stage.degraded_fallback(request, {}, InMemoryCostLogger())
    assert len(result.messages) == 5


def test_disabled_stage_engages_degraded_fallback_through_orchestrator(stub_client):
    from tokenetics import Tokenetics

    stage = ContextSchedulerStage(enabled=False)
    tk = Tokenetics(stages=[stage], client=stub_client)
    messages = _alternating_messages(30)
    tk.prepare(model="claude-sonnet-5", max_tokens=100, messages=messages)

    entry = tk.logger.entries[0]  # type: ignore[attr-defined]
    assert entry.extra["degraded"] is True
    assert entry.enabled is False


# --- empty history edge case ---


def test_empty_history_is_a_no_op():
    stage = ContextSchedulerStage()
    request = _request([], max_tokens=100)
    result = stage.run(request, {}, InMemoryCostLogger())
    assert result.messages == []


# --- role-alternation safety net ---


def test_fix_alternation_merges_adjacent_same_role_survivors():
    messages = [
        Message(role="user", content="first"),
        Message(role="user", content="second"),
        Message(role="assistant", content="reply"),
    ]
    fixed, dropped_lead = _fix_alternation(messages)
    assert [m.role for m in fixed] == ["user", "assistant"]
    assert fixed[0].content == "first\n\nsecond"
    assert dropped_lead is False  # merging is not "dropped content"


def test_fix_alternation_merges_list_content_by_concatenating_blocks():
    messages = [
        Message(role="user", content=[{"type": "text", "text": "a"}]),
        Message(role="user", content=[{"type": "text", "text": "b"}]),
    ]
    fixed, dropped_lead = _fix_alternation(messages)
    assert len(fixed) == 1
    assert fixed[0].content == [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
    assert dropped_lead is False


def test_fix_alternation_drops_a_lone_leading_assistant_turn():
    messages = [
        Message(role="assistant", content="orphaned reply"),
        Message(role="user", content="a real question"),
    ]
    fixed, dropped_lead = _fix_alternation(messages)
    assert [m.role for m in fixed] == ["user"]
    assert fixed[0].content == "a real question"
    assert dropped_lead is True


def test_fix_alternation_of_only_an_assistant_turn_is_empty():
    messages = [Message(role="assistant", content="orphaned")]
    fixed, dropped_lead = _fix_alternation(messages)
    assert fixed == []
    assert dropped_lead is True


def test_fix_alternation_leaves_already_valid_conversation_untouched():
    messages = [
        Message(role="user", content="hi"),
        Message(role="assistant", content="hello"),
        Message(role="user", content="bye"),
    ]
    fixed, dropped_lead = _fix_alternation(messages)
    assert fixed == messages
    assert dropped_lead is False


def test_run_does_not_log_a_lossless_merge_as_a_dropped_lead_turn():
    # Regression test: found via dev_demo.py's long_history scenario, where
    # two pinned same-role turns (a decision and an unresolved error, both
    # role="user") correctly merged into one message -- but the note used
    # to fire `alternation_fix_dropped_lead_turn=True` for this, which is
    # wrong: nothing was dropped, two messages were losslessly combined
    # into one. Only an actual lead-turn drop should set that flag.
    stage = ContextSchedulerStage()
    # Two pinned "decision" turns that both survive with the same role --
    # nothing between them also survives, so they merge into one message.
    messages = [
        {"role": "user", "content": "Let's go with option A, final answer."},
        {"role": "assistant", "content": "Got it."},
        {"role": "user", "content": "Actually, let's go with option B, final answer."},
    ]
    request = _request(messages, max_tokens=100)
    stage.run(request, {"token_budget": 1}, InMemoryCostLogger())
    assert stage.extra.get("alternation_fix_dropped_lead_turn") is not True


def test_dropped_turns_does_not_count_a_lossless_merge_as_a_drop():
    # Regression test, found via code review (2026-09-06): the exact
    # scenario from test_run_does_not_log_a_lossless_merge_as_a_dropped_
    # lead_turn above also mis-logged the NUMERIC dropped_turns count --
    # two pinned same-role turns surviving selection but merging into one
    # message used to report dropped_turns=2 out of only 3 total turns,
    # when only the one unpinned "Got it" turn was actually excluded.
    stage = ContextSchedulerStage()
    messages = [
        {"role": "user", "content": "Let's go with option A, final answer."},
        {"role": "assistant", "content": "Got it."},
        {"role": "user", "content": "Actually, let's go with option B, final answer."},
    ]
    request = _request(messages, max_tokens=100)
    stage.run(request, {"token_budget": 1}, InMemoryCostLogger())
    assert stage.extra["dropped_turns"] == 1
    assert stage.extra["pinned_turns"] == 2


def test_run_fails_open_when_the_alternation_fix_would_empty_the_result():
    # A budget so tight nothing survives once pinning drops to zero --
    # rather than send an empty history, the stage should skip its own
    # effect and pass the original request through unmodified.
    stage = ContextSchedulerStage()
    messages = [
        {"role": "user", "content": "Traceback (most recent call last):\nValueError: bad"},
        {"role": "assistant", "content": "Try X."},
        {"role": "user", "content": "That fixed it, thanks!"},
    ]
    request = _request(messages, max_tokens=100)
    result = stage.run(request, {"token_budget": 1}, InMemoryCostLogger())
    assert result.messages == request.messages
    assert stage.extra["alternation_fix_emptied_result"] is True
