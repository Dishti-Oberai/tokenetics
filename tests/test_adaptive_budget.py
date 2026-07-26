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


def test_sets_adaptive_thinking_effort_for_supported_model():
    request = _request("code", model="claude-sonnet-5")
    result = AdaptiveBudgetStage().run(request, {}, InMemoryCostLogger())
    assert result.extra["thinking"] == {"type": "adaptive", "display": "omitted"}
    assert result.extra["output_config"]["effort"] == "high"


def test_effort_level_varies_by_task_type():
    result = AdaptiveBudgetStage().run(
        _request("extraction", model="claude-sonnet-5"), {}, InMemoryCostLogger()
    )
    assert result.extra["output_config"]["effort"] == "low"


def test_leaves_thinking_untouched_when_unclassified():
    request = _request(None, model="claude-sonnet-5")
    result = AdaptiveBudgetStage().run(request, {}, InMemoryCostLogger())
    assert "thinking" not in result.extra


def test_respects_callers_existing_thinking_config():
    request = _request("code", model="claude-sonnet-5", thinking={"type": "adaptive"})
    result = AdaptiveBudgetStage().run(request, {}, InMemoryCostLogger())
    assert result.extra["thinking"] == {"type": "adaptive"}
    assert "output_config" not in result.extra


def test_fails_open_on_model_without_adaptive_thinking_support():
    request = _request("code", model="claude-haiku-4-5")
    result = AdaptiveBudgetStage().run(request, {}, InMemoryCostLogger())
    assert "thinking" not in result.extra


def test_notes_max_tokens_widen_and_thinking_effort():
    stage = AdaptiveBudgetStage()
    request = _request("code", max_tokens=50, model="claude-sonnet-5")
    stage.run(request, {}, InMemoryCostLogger())
    assert "max_tokens_widened_to" in stage.extra
    assert stage.extra["thinking_effort"] == "high"


def test_notes_skip_reason_on_unsupported_model():
    stage = AdaptiveBudgetStage()
    request = _request("code", model="claude-haiku-4-5")
    stage.run(request, {}, InMemoryCostLogger())
    assert stage.extra["thinking_effort_skipped"] == "model_unsupported"
