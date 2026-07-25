from tokenetics.core.logger import InMemoryCostLogger
from tokenetics.core.request import Message, from_api_kwargs
from tokenetics.stages.near_dup import (
    NearDupStage,
    _estimate_similarity,
    _minhash_signature,
    _shingles,
    _threshold_for,
)

_LONG_TEXT = "the quick brown fox jumps over the lazy dog near the riverbank"
_OTHER_TEXT = "completely unrelated sentence about something else entirely today"


def _request(messages):
    return from_api_kwargs(model="claude-sonnet-5", max_tokens=100, messages=messages)


def test_shingles_splits_into_overlapping_word_windows():
    shingles = _shingles("the quick brown fox jumps", k=3)
    assert "the quick brown" in shingles
    assert "quick brown fox" in shingles
    assert "brown fox jumps" in shingles


def test_identical_text_has_similarity_one():
    sig = _minhash_signature(_shingles(_LONG_TEXT))
    assert _estimate_similarity(sig, sig) == 1.0


def test_very_different_text_has_low_similarity():
    sig_a = _minhash_signature(_shingles(_LONG_TEXT))
    sig_b = _minhash_signature(_shingles(_OTHER_TEXT))
    assert _estimate_similarity(sig_a, sig_b) < 0.5


def test_identical_messages_are_merged_older_dropped():
    request = _request(
        [
            {"role": "assistant", "content": _LONG_TEXT},
            {"role": "user", "content": "ok"},
            {"role": "assistant", "content": _LONG_TEXT},
        ]
    )
    result = NearDupStage().run(request, {}, InMemoryCostLogger())
    assert len(result.messages) == 2
    assert result.messages[-1].content == _LONG_TEXT


def test_most_recent_message_is_never_dropped():
    request = _request(
        [
            {"role": "assistant", "content": _LONG_TEXT},
            {"role": "assistant", "content": _LONG_TEXT},
        ]
    )
    result = NearDupStage().run(request, {}, InMemoryCostLogger())
    assert len(result.messages) == 1
    assert result.messages[0].content == _LONG_TEXT


def test_dissimilar_messages_are_not_merged():
    request = _request(
        [
            {"role": "assistant", "content": _LONG_TEXT},
            {"role": "user", "content": "ok"},
            {"role": "assistant", "content": _OTHER_TEXT},
        ]
    )
    result = NearDupStage().run(request, {}, InMemoryCostLogger())
    assert len(result.messages) == 3


def test_merge_is_noted_with_similarity_score():
    stage = NearDupStage()
    request = _request(
        [
            {"role": "assistant", "content": _LONG_TEXT},
            {"role": "assistant", "content": _LONG_TEXT},
        ]
    )
    stage.run(request, {}, InMemoryCostLogger())
    assert stage.extra["merges"][0]["similarity"] == 1.0


def test_threshold_for_tool_output_is_looser_than_conversation():
    conversation_pair = (
        Message(role="user", content="hi"),
        Message(role="assistant", content="hello"),
    )
    tool_pair = (
        Message(role="user", content=[{"type": "tool_result", "content": "x"}]),
        Message(role="assistant", content="ok"),
    )
    assert _threshold_for(*conversation_pair) > _threshold_for(*tool_pair)
