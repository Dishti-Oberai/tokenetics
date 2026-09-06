from dataclasses import replace

from tokenetics.core.logger import InMemoryCostLogger
from tokenetics.core.request import RequestMeta, from_api_kwargs
from tokenetics.stages.brevity_injector import (
    AGGRESSIVE_INSTRUCTION,
    MODERATE_INSTRUCTION,
    BrevityInjectorStage,
)


def _request(task_type, system=None, bounded_shape=None):
    request = from_api_kwargs(
        model="claude-sonnet-5",
        max_tokens=100,
        messages=[{"role": "user", "content": "hi"}],
        system=system,
    )
    meta = RequestMeta(task_type=task_type, confidence=0.8, bounded_shape=bounded_shape)
    return replace(request, meta=meta)


def test_injects_moderate_instruction_for_extraction():
    request = _request("extraction")
    result = BrevityInjectorStage().run(request, {}, InMemoryCostLogger())
    assert result.system == MODERATE_INSTRUCTION


def test_appends_to_existing_system_prompt():
    request = _request("extraction", system="You are a helpful assistant.")
    result = BrevityInjectorStage().run(request, {}, InMemoryCostLogger())
    assert result.system == f"You are a helpful assistant.\n\n{MODERATE_INSTRUCTION}"


def test_off_for_code_when_shape_is_not_bounded():
    # bounded_shape defaults to None (ambiguous) -- conservative, stays off
    # even though "code" is in the default AGGRESSIVE set.
    request = _request("code")
    result = BrevityInjectorStage().run(request, {}, InMemoryCostLogger())
    assert result.system is None


def test_off_for_conversational_when_shape_is_not_bounded():
    request = _request("conversational")
    result = BrevityInjectorStage().run(request, {}, InMemoryCostLogger())
    assert result.system is None


def test_off_for_code_when_shape_is_explicitly_unbounded():
    request = _request("code", bounded_shape=False)
    result = BrevityInjectorStage().run(request, {}, InMemoryCostLogger())
    assert result.system is None


def test_off_when_unclassified():
    request = _request(None)
    result = BrevityInjectorStage().run(request, {}, InMemoryCostLogger())
    assert result.system is None


def test_notes_the_selected_bucket():
    stage = BrevityInjectorStage()
    request = _request("extraction")
    stage.run(request, {}, InMemoryCostLogger())
    assert stage.extra["brevity_bucket"] == "moderate"


def test_no_note_when_bucket_is_off():
    stage = BrevityInjectorStage()
    request = _request("code")
    stage.run(request, {}, InMemoryCostLogger())
    assert stage.extra == {}


def test_aggressive_is_off_without_bounded_shape_even_when_opted_in():
    # The opt-in exists (config["aggressive_for_task_types"]), but the
    # bounded_shape gate applies regardless -- ambiguous shape stays off.
    request = _request("code")
    config = {"aggressive_for_task_types": ["code"]}
    result = BrevityInjectorStage().run(request, config, InMemoryCostLogger())
    assert result.system is None


def test_aggressive_fires_by_default_for_code_when_bounded():
    # "code" is in the default AGGRESSIVE set -- no config needed once
    # bounded_shape is True.
    request = _request("code", bounded_shape=True)
    result = BrevityInjectorStage().run(request, {}, InMemoryCostLogger())
    assert result.system == AGGRESSIVE_INSTRUCTION


def test_aggressive_fires_by_default_for_conversational_when_bounded():
    request = _request("conversational", bounded_shape=True)
    result = BrevityInjectorStage().run(request, {}, InMemoryCostLogger())
    assert result.system == AGGRESSIVE_INSTRUCTION


def test_aggressive_fires_by_default_for_tool_heavy_when_bounded():
    request = _request("tool-heavy", bounded_shape=True)
    result = BrevityInjectorStage().run(request, {}, InMemoryCostLogger())
    assert result.system == AGGRESSIVE_INSTRUCTION


def test_aggressive_does_not_default_for_extraction_even_when_bounded():
    # extraction's real economics came out break-even -- deliberately left
    # out of the default set, stays MODERATE unless explicitly opted in.
    request = _request("extraction", bounded_shape=True)
    result = BrevityInjectorStage().run(request, {}, InMemoryCostLogger())
    assert result.system == MODERATE_INSTRUCTION


def test_aggressive_opt_in_does_not_affect_other_task_types():
    # Opting in "extraction" must not accidentally turn on AGGRESSIVE for a
    # task type not in the list and not in the default set.
    request = _request(None, bounded_shape=True)
    config = {"aggressive_for_task_types": ["extraction"]}
    result = BrevityInjectorStage().run(request, config, InMemoryCostLogger())
    assert result.system is None


def test_aggressive_opt_in_overrides_the_default_moderate_bucket():
    request = _request("extraction", bounded_shape=True)
    config = {"aggressive_for_task_types": ["extraction"]}
    result = BrevityInjectorStage().run(request, config, InMemoryCostLogger())
    assert result.system == AGGRESSIVE_INSTRUCTION


def test_aggressive_opt_in_for_extraction_still_requires_bounded_shape():
    request = _request("extraction", bounded_shape=False)
    config = {"aggressive_for_task_types": ["extraction"]}
    result = BrevityInjectorStage().run(request, config, InMemoryCostLogger())
    assert result.system == MODERATE_INSTRUCTION


def test_notes_aggressive_bucket_when_selected():
    stage = BrevityInjectorStage()
    request = _request("code", bounded_shape=True)
    stage.run(request, {}, InMemoryCostLogger())
    assert stage.extra["brevity_bucket"] == "aggressive"
