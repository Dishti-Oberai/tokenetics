from dataclasses import replace

from tokenetics.core.logger import InMemoryCostLogger
from tokenetics.core.request import RequestMeta, from_api_kwargs
from tokenetics.stages.brevity_injector import MODERATE_INSTRUCTION, BrevityInjectorStage


def _request(task_type, system=None):
    request = from_api_kwargs(
        model="claude-sonnet-5",
        max_tokens=100,
        messages=[{"role": "user", "content": "hi"}],
        system=system,
    )
    return replace(request, meta=RequestMeta(task_type=task_type, confidence=0.8))


def test_injects_moderate_instruction_for_extraction():
    request = _request("extraction")
    result = BrevityInjectorStage().run(request, {}, InMemoryCostLogger())
    assert result.system == MODERATE_INSTRUCTION


def test_appends_to_existing_system_prompt():
    request = _request("extraction", system="You are a helpful assistant.")
    result = BrevityInjectorStage().run(request, {}, InMemoryCostLogger())
    assert result.system == f"You are a helpful assistant.\n\n{MODERATE_INSTRUCTION}"


def test_off_by_default_for_code():
    request = _request("code")
    result = BrevityInjectorStage().run(request, {}, InMemoryCostLogger())
    assert result.system is None


def test_off_by_default_for_conversational():
    request = _request("conversational")
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
