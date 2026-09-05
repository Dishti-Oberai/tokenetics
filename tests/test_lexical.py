from tokenetics.core.lexical import latest_user_text, overlap_score
from tokenetics.core.request import Message


def test_overlap_score_identical_text_is_one():
    assert overlap_score("hello world", "hello world") == 1.0


def test_overlap_score_disjoint_text_is_zero():
    assert overlap_score("hello world", "goodbye moon") == 0.0


def test_overlap_score_partial_overlap_is_between_zero_and_one():
    score = overlap_score("get weather for city", "what is the weather today")
    assert 0.0 < score < 1.0


def test_overlap_score_empty_string_is_zero():
    assert overlap_score("", "hello") == 0.0
    assert overlap_score("hello", "") == 0.0


def test_overlap_score_is_case_insensitive():
    assert overlap_score("Hello World", "hello world") == 1.0


def test_overlap_score_ignores_contraction_remnants():
    # Regression test for a real bug caught via a live MCP demo request
    # (2026-09-06): "user's" and "what's" both fragment (on the apostrophe)
    # into a real word plus the stray token "s" -- two texts about
    # completely different topics, each containing an unrelated
    # contraction, must not register as overlapping purely because of that.
    tool_text = "send_email Send an email on the user's behalf with a subject and body."
    question = "What's the weather like in Paris today? It's been raining a lot."
    assert overlap_score(tool_text, question) == 0.0


def test_latest_user_text_returns_last_user_message():
    messages = [
        Message(role="user", content="first"),
        Message(role="assistant", content="reply"),
        Message(role="user", content="second"),
    ]
    assert latest_user_text(messages) == "second"


def test_latest_user_text_joins_text_blocks():
    messages = [
        Message(
            role="user",
            content=[{"type": "text", "text": "hello"}, {"type": "text", "text": "world"}],
        ),
    ]
    assert latest_user_text(messages) == "hello world"


def test_latest_user_text_ignores_non_text_blocks():
    messages = [
        Message(
            role="user",
            content=[{"type": "tool_result", "tool_use_id": "1", "content": "42"}],
        ),
    ]
    assert latest_user_text(messages) == ""


def test_latest_user_text_empty_when_no_user_messages():
    messages = [Message(role="assistant", content="hi")]
    assert latest_user_text(messages) == ""
