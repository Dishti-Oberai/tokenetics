from tokenetics.core.logger import InMemoryCostLogger
from tokenetics.core.request import from_api_kwargs
from tokenetics.stages.schema_minification import SchemaMinificationStage


def _request(tools, user_text="I need to look stuff up, can you help?"):
    return from_api_kwargs(
        model="claude-sonnet-5",
        max_tokens=100,
        messages=[{"role": "user", "content": user_text}],
        tools=tools,
    )


def test_strips_inert_metadata_keys():
    request = _request(
        [
            {
                "name": "lookup",
                "description": "look stuff up",
                "input_schema": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "$comment": "internal note",
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            }
        ]
    )
    result = SchemaMinificationStage().run(request, {}, InMemoryCostLogger())
    schema = result.tools[0].input_schema
    assert "$schema" not in schema
    assert "$comment" not in schema
    assert schema["type"] == "object"
    assert schema["required"] == ["query"]
    assert schema["properties"]["query"]["type"] == "string"


def test_collapses_redundant_whitespace_in_descriptions():
    request = _request(
        [
            {
                "name": "lookup",
                "description": "look   stuff\n\n  up",
                "input_schema": {"type": "object", "properties": {}},
            }
        ]
    )
    result = SchemaMinificationStage().run(request, {}, InMemoryCostLogger())
    assert result.tools[0].description == "look stuff up"


def test_preserves_functionally_important_fields():
    request = _request(
        [
            {
                "name": "lookup",
                "description": "look stuff up",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string", "enum": ["a", "b"]}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
            }
        ]
    )
    result = SchemaMinificationStage().run(request, {}, InMemoryCostLogger())
    schema = result.tools[0].input_schema
    assert schema["properties"]["query"]["enum"] == ["a", "b"]
    assert schema["additionalProperties"] is False


def test_no_tools_is_a_no_op():
    request = _request([])
    result = SchemaMinificationStage().run(request, {}, InMemoryCostLogger())
    assert result == request


_LOOKUP_TOOL = {
    "name": "lookup",
    "description": "look stuff up",
    "input_schema": {"type": "object", "properties": {}},
}
_WEATHER_TOOL = {
    "name": "get_weather",
    "description": "Get the current weather for a city",
    "input_schema": {"type": "object", "properties": {}},
}


def test_drops_tool_with_zero_word_overlap():
    request = _request([_WEATHER_TOOL], user_text="I need to look stuff up, can you help?")
    result = SchemaMinificationStage().run(request, {}, InMemoryCostLogger())
    assert result.tools == []


def test_keeps_tool_with_any_word_overlap():
    request = _request(
        [_WEATHER_TOOL], user_text="What's the weather like in Boston this weekend?"
    )
    result = SchemaMinificationStage().run(request, {}, InMemoryCostLogger())
    assert [t.name for t in result.tools] == ["get_weather"]


def test_keeps_all_tools_when_no_user_text_to_compare_against():
    request = from_api_kwargs(
        model="claude-sonnet-5",
        max_tokens=100,
        messages=[{"role": "assistant", "content": "hello"}],
        tools=[_LOOKUP_TOOL, _WEATHER_TOOL],
    )
    result = SchemaMinificationStage().run(request, {}, InMemoryCostLogger())
    assert len(result.tools) == 2


def test_notes_dropped_tools_with_score():
    stage = SchemaMinificationStage()
    request = _request([_WEATHER_TOOL], user_text="I need to look stuff up, can you help?")
    stage.run(request, {}, InMemoryCostLogger())
    assert stage.extra["dropped_tools"] == [{"tool": "get_weather", "score": 0.0}]
