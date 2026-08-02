import json

from tokenetics.core.delta import apply_delta, is_delta_wire_text
from tokenetics.core.logger import InMemoryCostLogger
from tokenetics.core.request import from_api_kwargs
from tokenetics.orchestrator import Tokenetics
from tokenetics.stages.delta_compression import DeltaCompressionStage

_LARGE_PREVIOUS = json.dumps({"status": "pending", "id": 42, "note": "x" * 300})
_LARGE_NEW = json.dumps({"status": "done", "id": 42, "note": "x" * 300})


def _request_with_tool_result(content: str, tool_use_id: str = "toolu_1"):
    return from_api_kwargs(
        model="claude-sonnet-5",
        max_tokens=100,
        messages=[
            {"role": "user", "content": "check status"},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": tool_use_id, "name": "check_status", "input": {}}],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": tool_use_id, "content": content}],
            },
        ],
    )


def test_no_previous_payloads_is_a_noop():
    stage = DeltaCompressionStage()
    request = _request_with_tool_result(_LARGE_NEW)
    result = stage.run(request, {}, InMemoryCostLogger())
    assert result is request
    assert stage.extra["compressed_count"] == 0
    assert stage.extra["reason"] == "no_previous_payloads_supplied"


def test_matching_tool_result_gets_compressed():
    stage = DeltaCompressionStage()
    request = _request_with_tool_result(_LARGE_NEW, tool_use_id="toolu_1")
    config = {"previous_payloads": {"toolu_1": _LARGE_PREVIOUS}}
    result = stage.run(request, config, InMemoryCostLogger())

    assert stage.extra["compressed_count"] == 1
    tool_result_block = result.messages[-1].content[0]
    assert is_delta_wire_text(tool_result_block["content"])
    # The delta-compressed content, applied against the previous payload,
    # reconstructs exactly what was there before compression.
    reconstructed = apply_delta(_LARGE_PREVIOUS, tool_result_block["content"])
    assert json.loads(reconstructed) == json.loads(_LARGE_NEW)


def test_non_matching_tool_use_id_is_left_untouched():
    stage = DeltaCompressionStage()
    request = _request_with_tool_result(_LARGE_NEW, tool_use_id="toolu_1")
    config = {"previous_payloads": {"some_other_id": _LARGE_PREVIOUS}}
    result = stage.run(request, config, InMemoryCostLogger())

    assert stage.extra["compressed_count"] == 0
    assert result.messages[-1].content[0]["content"] == _LARGE_NEW


def test_short_content_with_no_meaningfully_smaller_delta_is_left_full():
    stage = DeltaCompressionStage()
    request = _request_with_tool_result("done", tool_use_id="toolu_1")
    config = {"previous_payloads": {"toolu_1": "pending"}}
    result = stage.run(request, config, InMemoryCostLogger())

    assert stage.extra["compressed_count"] == 0
    assert stage.extra["skipped_not_meaningfully_smaller"] == 1
    assert result.messages[-1].content[0]["content"] == "done"


def test_non_string_tool_result_content_is_skipped_not_guessed_at():
    stage = DeltaCompressionStage()
    request = from_api_kwargs(
        model="claude-sonnet-5",
        max_tokens=100,
        messages=[
            {"role": "user", "content": "look at this"},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "toolu_1", "name": "get_image", "input": {}}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": [{"type": "image", "source": {"type": "base64", "data": "..."}}],
                    }
                ],
            },
        ],
    )
    config = {"previous_payloads": {"toolu_1": "some previous text"}}
    result = stage.run(request, config, InMemoryCostLogger())

    assert stage.extra["compressed_count"] == 0
    assert stage.extra["skipped_non_string_content"] == 1
    assert result is request


def test_plain_string_message_content_is_untouched():
    stage = DeltaCompressionStage()
    request = from_api_kwargs(
        model="claude-sonnet-5",
        max_tokens=100,
        messages=[{"role": "user", "content": "just plain text, no blocks"}],
    )
    config = {"previous_payloads": {"toolu_1": "anything"}}
    result = stage.run(request, config, InMemoryCostLogger())
    assert result is request
    assert stage.extra["compressed_count"] == 0


def test_multiple_matching_tool_results_all_compressed():
    stage = DeltaCompressionStage()
    request = from_api_kwargs(
        model="claude-sonnet-5",
        max_tokens=100,
        messages=[
            {"role": "user", "content": "check two things"},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "toolu_1", "name": "check_status", "input": {}},
                    {"type": "tool_use", "id": "toolu_2", "name": "check_status", "input": {}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": _LARGE_NEW},
                    {"type": "tool_result", "tool_use_id": "toolu_2", "content": _LARGE_NEW},
                ],
            },
        ],
    )
    config = {
        "previous_payloads": {"toolu_1": _LARGE_PREVIOUS, "toolu_2": _LARGE_PREVIOUS}
    }
    result = stage.run(request, config, InMemoryCostLogger())
    assert stage.extra["compressed_count"] == 2
    blocks = result.messages[-1].content
    assert is_delta_wire_text(blocks[0]["content"])
    assert is_delta_wire_text(blocks[1]["content"])


def test_deliberately_broken_content_block_fails_open_through_orchestrator(stub_client):
    # A malformed content block -- a plain string where a content-block dict
    # is expected -- makes `block.get(...)` raise AttributeError deep inside
    # the stage. The orchestrator's fail-open framework must catch it and
    # pass the request through unmodified, not crash the whole prepare() call.
    tk = Tokenetics(
        client=stub_client,
        stage_config={"delta_compression": {"previous_payloads": {"toolu_1": "previous"}}},
    )
    broken_content = ["not a real content block"]
    result = tk.prepare(
        model="claude-sonnet-5",
        max_tokens=100,
        messages=[
            {"role": "user", "content": "check status"},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "toolu_1", "name": "check_status", "input": {}}],
            },
            {"role": "user", "content": broken_content},
        ],
    )
    assert result["messages"][-1]["content"] == broken_content

    entries = tk.logger.entries  # type: ignore[attr-defined]
    delta_entry = next(e for e in entries if e.stage_name == "delta_compression")
    assert delta_entry.extra.get("error") is True
