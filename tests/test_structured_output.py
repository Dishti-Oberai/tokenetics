from dataclasses import replace

from tokenetics.core.logger import InMemoryCostLogger
from tokenetics.core.request import RequestMeta, from_api_kwargs
from tokenetics.stages.structured_output import StructuredOutputStage

_EXTRACT_TOOL = {
    "name": "extract_contact",
    "description": "Extract a contact's name, email, and phone number",
    "input_schema": {"type": "object", "properties": {}},
}
_UNRELATED_TOOL = {
    "name": "get_weather",
    "description": "Get the current weather for a city",
    "input_schema": {"type": "object", "properties": {}},
}


def _request(*, task_type, tools=None, messages=None, extra=None):
    request = from_api_kwargs(
        model="claude-sonnet-5",
        max_tokens=100,
        messages=messages or [{"role": "user", "content": "hi"}],
        tools=tools,
        **(extra or {}),
    )
    return replace(request, meta=RequestMeta(task_type=task_type, confidence=0.8))


def test_forces_tool_choice_on_matching_extraction_request():
    request = _request(
        task_type="extraction",
        tools=[_EXTRACT_TOOL, _UNRELATED_TOOL],
        messages=[
            {
                "role": "user",
                "content": "Extract the contact name, email, and phone from this email.",
            }
        ],
    )
    result = StructuredOutputStage().run(request, {}, InMemoryCostLogger())
    assert result.extra["tool_choice"] == {"type": "tool", "name": "extract_contact"}


def test_no_op_when_task_type_is_not_extraction():
    request = _request(task_type="conversational", tools=[_EXTRACT_TOOL])
    request = replace(request, meta=RequestMeta(task_type="conversational", confidence=0.4))
    result = StructuredOutputStage().run(request, {}, InMemoryCostLogger())
    assert "tool_choice" not in result.extra


def test_no_op_when_no_tools_registered():
    request = _request(task_type="extraction", tools=None)
    result = StructuredOutputStage().run(request, {}, InMemoryCostLogger())
    assert "tool_choice" not in result.extra


def test_no_op_when_no_tool_clears_the_match_threshold():
    request = _request(
        task_type="extraction",
        tools=[_UNRELATED_TOOL],
        messages=[{"role": "user", "content": "Extract the key data points from this."}],
    )
    result = StructuredOutputStage().run(request, {}, InMemoryCostLogger())
    assert "tool_choice" not in result.extra


def test_respects_callers_existing_tool_choice():
    request = _request(
        task_type="extraction",
        tools=[_EXTRACT_TOOL],
        messages=[{"role": "user", "content": "Extract the contact name and email."}],
        extra={"tool_choice": {"type": "auto"}},
    )
    result = StructuredOutputStage().run(request, {}, InMemoryCostLogger())
    assert result.extra["tool_choice"] == {"type": "auto"}


def test_notes_forced_tool_and_match_score():
    stage = StructuredOutputStage()
    request = _request(
        task_type="extraction",
        tools=[_EXTRACT_TOOL],
        messages=[{"role": "user", "content": "Extract the contact name and email."}],
    )
    stage.run(request, {}, InMemoryCostLogger())
    assert stage.extra["tool_choice_forced"] == "extract_contact"
    assert stage.extra["match_score"] > 0
