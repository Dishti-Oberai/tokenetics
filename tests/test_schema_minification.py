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


def test_a_tool_already_used_earlier_is_exempt_from_the_relevance_drop():
    # Regression test (caught via Phase 9's benchmark corpus, 2026-08-02): a
    # tool with a real tool_use block earlier in the conversation must not
    # be dropped just because a later follow-up doesn't happen to share any
    # words with its name/description -- e.g. "What about a large?" after
    # already checking medium-size stock shares zero words with
    # "check_inventory", but the model may still need to call it again.
    request = from_api_kwargs(
        model="claude-sonnet-5",
        max_tokens=100,
        messages=[
            {"role": "user", "content": "Do we have medium in stock?"},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "1", "name": "get_weather", "input": {}}
                ],
            },
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "1", "content": "0"}]},
            {"role": "user", "content": "What about a large?"},
        ],
        tools=[_WEATHER_TOOL],
    )
    stage = SchemaMinificationStage()
    result = stage.run(request, {}, InMemoryCostLogger())
    assert [t.name for t in result.tools] == ["get_weather"]
    assert "dropped_tools" not in stage.extra


def _messages_with_n_followups_since_tool_use(n: int):
    messages = [
        {"role": "user", "content": "Do we have medium in stock?"},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "1", "name": "get_weather", "input": {}}],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "1", "content": "0"}]},
    ]
    for i in range(n):
        messages.append({"role": "assistant", "content": f"reply {i}"})
        messages.append({"role": "user", "content": f"unrelated follow-up {i}"})
    return messages


def test_a_tool_used_within_the_reuse_window_stays_exempt():
    # Exactly at the window boundary (3 genuine follow-ups since use) --
    # still exempt.
    request = from_api_kwargs(
        model="claude-sonnet-5",
        max_tokens=100,
        messages=_messages_with_n_followups_since_tool_use(3),
        tools=[_WEATHER_TOOL],
    )
    stage = SchemaMinificationStage()
    result = stage.run(request, {}, InMemoryCostLogger())
    assert [t.name for t in result.tools] == ["get_weather"]
    assert "dropped_tools" not in stage.extra


def test_a_tool_used_beyond_the_reuse_window_reverts_to_relevance_scoring():
    # One follow-up past the window (4) -- the exemption has expired, and
    # since none of the unrelated follow-ups share any words with
    # "get_weather", it's correctly dropped again, reclaiming the tokens.
    request = from_api_kwargs(
        model="claude-sonnet-5",
        max_tokens=100,
        messages=_messages_with_n_followups_since_tool_use(4),
        tools=[_WEATHER_TOOL],
    )
    stage = SchemaMinificationStage()
    result = stage.run(request, {}, InMemoryCostLogger())
    assert result.tools == []
    assert stage.extra["dropped_tools"] == [{"tool": "get_weather", "score": 0.0}]


def test_tool_result_messages_do_not_consume_the_reuse_window():
    # Several tool round-trips (each tool_result is mechanically role="user"
    # but isn't a genuine follow-up question) before the real next question
    # -- none of those should count against the window.
    request = from_api_kwargs(
        model="claude-sonnet-5",
        max_tokens=100,
        messages=[
            {"role": "user", "content": "Check the weather."},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "1", "name": "get_weather", "input": {}}],
            },
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "1", "content": "a"}]},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "2", "name": "get_weather", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "2", "content": "b"}]},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "3", "name": "get_weather", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "3", "content": "c"}]},
            {"role": "user", "content": "Totally unrelated question."},
        ],
        tools=[_WEATHER_TOOL],
    )
    stage = SchemaMinificationStage()
    result = stage.run(request, {}, InMemoryCostLogger())
    assert [t.name for t in result.tools] == ["get_weather"]
    assert "dropped_tools" not in stage.extra


def test_a_never_used_irrelevant_tool_is_still_dropped_alongside_an_exempt_one():
    request = from_api_kwargs(
        model="claude-sonnet-5",
        max_tokens=100,
        messages=[
            {"role": "user", "content": "Check the weather in Boston."},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "1", "name": "get_weather", "input": {}}
                ],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "1", "content": "58F"}],
            },
            {"role": "user", "content": "What about tomorrow?"},
        ],
        tools=[_WEATHER_TOOL, _LOOKUP_TOOL],
    )
    stage = SchemaMinificationStage()
    result = stage.run(request, {}, InMemoryCostLogger())
    # get_weather exempt (already used); lookup still has zero overlap and
    # was never used, so it's still correctly dropped.
    assert [t.name for t in result.tools] == ["get_weather"]
    assert stage.extra["dropped_tools"] == [{"tool": "lookup", "score": 0.0}]
