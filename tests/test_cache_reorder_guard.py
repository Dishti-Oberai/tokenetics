import pytest

from tokenetics.core.errors import CacheSafetyError
from tokenetics.core.logger import InMemoryCostLogger
from tokenetics.core.request import from_api_kwargs
from tokenetics.stages.cache_reorder_guard import CacheReorderGuardStage

_TOOL = {
    "name": "get_weather",
    "description": "Get the current weather for a city",
    "input_schema": {"type": "object", "properties": {}},
}


def _kwargs(system="You are a helpful assistant.", tools=None, user_text="hi"):
    return {
        "model": "claude-sonnet-5",
        "max_tokens": 100,
        "system": system,
        "tools": tools,
        "messages": [{"role": "user", "content": user_text}],
    }


# --- the standalone hard-raise test, per CLAUDE.md's explicit requirement ---


def test_hard_raises_when_system_prompt_changed():
    stage = CacheReorderGuardStage()
    request = from_api_kwargs(**_kwargs(system="New system prompt."))
    config = {"previous_request": _kwargs(system="Old system prompt.")}
    with pytest.raises(CacheSafetyError):
        stage.run(request, config, InMemoryCostLogger())


def test_hard_raises_when_tools_changed():
    stage = CacheReorderGuardStage()
    other_tool = {**_TOOL, "name": "different_tool"}
    request = from_api_kwargs(**_kwargs(tools=[_TOOL]))
    config = {"previous_request": _kwargs(tools=[other_tool])}
    with pytest.raises(CacheSafetyError):
        stage.run(request, config, InMemoryCostLogger())


def test_passes_silently_when_stable_prefix_is_byte_identical():
    stage = CacheReorderGuardStage()
    request = from_api_kwargs(**_kwargs(system="Same.", tools=[_TOOL]))
    config = {"previous_request": _kwargs(system="Same.", tools=[_TOOL])}
    result = stage.run(request, config, InMemoryCostLogger())
    assert result == request
    assert stage.extra["guard_checked"] is True
    assert stage.extra["guard_result"] == "unchanged"


def test_no_previous_request_skips_the_guard_entirely():
    stage = CacheReorderGuardStage()
    request = from_api_kwargs(**_kwargs())
    result = stage.run(request, {}, InMemoryCostLogger())
    assert result == request
    assert "guard_checked" not in stage.extra


def test_reorder_is_a_noop_given_the_current_stable_first_request_shape():
    stage = CacheReorderGuardStage()
    request = from_api_kwargs(**_kwargs(system="Sys.", tools=[_TOOL]))
    result = stage.run(request, {}, InMemoryCostLogger())
    assert result.system == "Sys."
    assert result.tools == request.tools
    assert stage.extra["reorder_noop"] is True


# --- regression test: the guard's raise must NOT be swallowed by the
# orchestrator's generic fail-open framework ---


def test_orchestrator_propagates_cache_safety_error_instead_of_fail_opening(stub_client):
    from tokenetics import Tokenetics

    stage = CacheReorderGuardStage()
    tk = Tokenetics(
        stages=[stage],
        client=stub_client,
        stage_config={"cache_reorder_guard": {"previous_request": _kwargs(system="Old.")}},
    )
    with pytest.raises(CacheSafetyError):
        tk.prepare(**_kwargs(system="New."))
