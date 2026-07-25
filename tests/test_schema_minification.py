from tokenetics.core.logger import InMemoryCostLogger
from tokenetics.core.request import from_api_kwargs
from tokenetics.stages.schema_minification import SchemaMinificationStage


def _request(tools):
    return from_api_kwargs(
        model="claude-sonnet-5",
        max_tokens=100,
        messages=[{"role": "user", "content": "hi"}],
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
