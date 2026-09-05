from dataclasses import replace

from tokenetics.core.logger import InMemoryCostLogger
from tokenetics.core.request import RequestMeta, from_api_kwargs
from tokenetics.stages.adaptive_budget import AdaptiveBudgetStage


def _request(task_type, max_tokens=50, model="claude-sonnet-5", **extra):
    request = from_api_kwargs(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": "hi"}],
        **extra,
    )
    return replace(request, meta=RequestMeta(task_type=task_type, confidence=0.8))


def test_widens_max_tokens_for_a_known_task_type():
    request = _request("code", max_tokens=50)
    result = AdaptiveBudgetStage().run(request, {}, InMemoryCostLogger())
    assert result.max_tokens > 50


def test_never_narrows_below_the_callers_value():
    request = _request("conversational", max_tokens=100_000)
    result = AdaptiveBudgetStage().run(request, {}, InMemoryCostLogger())
    assert result.max_tokens == 100_000


def test_leaves_max_tokens_untouched_when_unclassified():
    request = _request(None, max_tokens=50)
    result = AdaptiveBudgetStage().run(request, {}, InMemoryCostLogger())
    assert result.max_tokens == 50


def test_truncation_stats_widen_further():
    request = _request("conversational", max_tokens=50)
    baseline = AdaptiveBudgetStage().run(request, {}, InMemoryCostLogger())

    request2 = _request("conversational", max_tokens=50)
    widened = AdaptiveBudgetStage().run(
        request2, {"truncation_stats": {"conversational": 0.05}}, InMemoryCostLogger()
    )
    assert widened.max_tokens > baseline.max_tokens


_OPT_IN = {"enable_thinking_effort": True}


def test_thinking_effort_not_set_by_default_without_opt_in():
    # The core new-behavior test (2026-09-04, after a real $-cost benchmark
    # showed the old automatic default made "code"/"conversational"
    # requests 2-3x more expensive with no caller signal requesting that
    # trade): a classified, model-supported request must NOT get thinking
    # effort set unless the caller explicitly opts in via config.
    request = _request("code", model="claude-sonnet-5")
    result = AdaptiveBudgetStage().run(request, {}, InMemoryCostLogger())
    assert "thinking" not in result.extra
    assert "output_config" not in result.extra


def test_notes_skip_reason_when_opt_in_not_enabled():
    stage = AdaptiveBudgetStage()
    request = _request("code", model="claude-sonnet-5")
    stage.run(request, {}, InMemoryCostLogger())
    assert stage.extra["thinking_effort_skipped"] == "opt_in_not_enabled"


def test_sets_adaptive_thinking_effort_when_opted_in_for_supported_model():
    request = _request("code", model="claude-sonnet-5")
    result = AdaptiveBudgetStage().run(request, _OPT_IN, InMemoryCostLogger())
    assert result.extra["thinking"] == {"type": "adaptive", "display": "omitted"}
    assert result.extra["output_config"]["effort"] == "high"


def test_effort_level_varies_by_task_type_when_opted_in():
    result = AdaptiveBudgetStage().run(
        _request("extraction", model="claude-sonnet-5"), _OPT_IN, InMemoryCostLogger()
    )
    assert result.extra["output_config"]["effort"] == "low"


def test_leaves_thinking_untouched_when_unclassified_even_if_opted_in():
    request = _request(None, model="claude-sonnet-5")
    result = AdaptiveBudgetStage().run(request, _OPT_IN, InMemoryCostLogger())
    assert "thinking" not in result.extra


def test_respects_callers_existing_thinking_config_even_if_opted_in():
    request = _request("code", model="claude-sonnet-5", thinking={"type": "adaptive"})
    result = AdaptiveBudgetStage().run(request, _OPT_IN, InMemoryCostLogger())
    assert result.extra["thinking"] == {"type": "adaptive"}
    assert "output_config" not in result.extra


def test_fails_open_on_model_without_adaptive_thinking_support_even_if_opted_in():
    request = _request("code", model="claude-haiku-4-5")
    result = AdaptiveBudgetStage().run(request, _OPT_IN, InMemoryCostLogger())
    assert "thinking" not in result.extra


def test_notes_max_tokens_widen_and_thinking_effort_when_opted_in():
    stage = AdaptiveBudgetStage()
    request = _request("code", max_tokens=50, model="claude-sonnet-5")
    stage.run(request, _OPT_IN, InMemoryCostLogger())
    assert "max_tokens_widened_to" in stage.extra
    assert stage.extra["thinking_effort"] == "high"


def test_notes_skip_reason_on_unsupported_model_when_opted_in():
    stage = AdaptiveBudgetStage()
    request = _request("code", model="claude-haiku-4-5")
    stage.run(request, _OPT_IN, InMemoryCostLogger())
    assert stage.extra["thinking_effort_skipped"] == "model_unsupported"


def test_external_budget_estimate_can_widen_further_than_the_heuristic():
    # "conversational" heuristic widens 50 -> 360; a much larger external
    # (TALE) estimate should win instead, proving it's actually used as an
    # estimate source rather than being ignored entirely.
    request = _request("conversational", max_tokens=50)
    result = AdaptiveBudgetStage().run(
        request, {"external_budget_estimate": 1000}, InMemoryCostLogger()
    )
    assert result.max_tokens == round(1000 * 1.2)


def test_external_budget_estimate_works_even_when_unclassified():
    # The whole point of TALE: it doesn't need task_type classification at
    # all, unlike the free heuristic which requires a known task_type.
    request = _request(None, max_tokens=50)
    result = AdaptiveBudgetStage().run(
        request, {"external_budget_estimate": 500}, InMemoryCostLogger()
    )
    assert result.max_tokens == round(500 * 1.2)


def test_external_budget_estimate_still_never_narrows_below_callers_value():
    request = _request("conversational", max_tokens=100_000)
    result = AdaptiveBudgetStage().run(
        request, {"external_budget_estimate": 10}, InMemoryCostLogger()
    )
    assert result.max_tokens == 100_000


def test_external_budget_estimate_notes_reason():
    stage = AdaptiveBudgetStage()
    request = _request("conversational", max_tokens=50)
    stage.run(request, {"external_budget_estimate": 1000}, InMemoryCostLogger())
    assert stage.extra["max_tokens_widen_reason"] == "external_estimate"


def test_external_budget_estimate_never_undercuts_the_heuristic_for_a_classified_task_type():
    # Regression test for a real bug caught via a live `--tale` run
    # (2026-08-10): TALE estimated 150 tokens for a `code` question's
    # visible answer with no way to know adaptive_budget was about to also
    # switch on high-effort thinking for the same request -- thinking draws
    # from the same max_tokens budget, so the 180-token widened cap left
    # ZERO room for visible text once thinking consumed it (a real reply
    # came back with no TextBlock at all). The heuristic's 960-token code
    # estimate would have had headroom for both. A too-small external
    # estimate must never leave the caller worse off than the heuristic
    # alone would have -- the wider of the two must always win.
    request = _request("code", max_tokens=50)
    result = AdaptiveBudgetStage().run(
        request, {"external_budget_estimate": 150}, InMemoryCostLogger()
    )
    heuristic_only = AdaptiveBudgetStage().run(_request("code", max_tokens=50), {}, InMemoryCostLogger())
    assert result.max_tokens == heuristic_only.max_tokens
    assert result.max_tokens == round(800 * 1.2)


def test_external_budget_estimate_still_wins_when_it_genuinely_widens_further():
    # A LARGER external estimate than the heuristic should still take
    # effect -- the fix caps how far TALE can narrow the budget, it
    # doesn't disable TALE from widening it further when warranted.
    request = _request("code", max_tokens=50)
    result = AdaptiveBudgetStage().run(
        request, {"external_budget_estimate": 2000}, InMemoryCostLogger()
    )
    assert result.max_tokens == round(2000 * 1.2)


def test_reason_reflects_whichever_source_actually_won():
    stage = AdaptiveBudgetStage()
    request = _request("code", max_tokens=50)
    stage.run(request, {"external_budget_estimate": 150}, InMemoryCostLogger())
    assert stage.extra["max_tokens_widen_reason"] == "task_type_estimate"

    stage2 = AdaptiveBudgetStage()
    request2 = _request("code", max_tokens=50)
    stage2.run(request2, {"external_budget_estimate": 2000}, InMemoryCostLogger())
    assert stage2.extra["max_tokens_widen_reason"] == "external_estimate"
