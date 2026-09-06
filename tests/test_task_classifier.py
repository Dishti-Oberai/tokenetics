from tokenetics.core.logger import InMemoryCostLogger
from tokenetics.core.request import from_api_kwargs
from tokenetics.stages.task_classifier import TaskClassifierStage


def _request(messages, tools=None):
    return from_api_kwargs(
        model="claude-sonnet-5", max_tokens=100, messages=messages, tools=tools
    )


def test_detects_code():
    request = _request(
        [{"role": "user", "content": "```python\ndef f():\n    pass\n```\nWhat does this do?"}]
    )
    result = TaskClassifierStage().run(request, {}, InMemoryCostLogger())
    assert result.meta.task_type == "code"


def test_detects_extraction():
    request = _request(
        [{"role": "user", "content": "Extract all email addresses from this text."}]
    )
    result = TaskClassifierStage().run(request, {}, InMemoryCostLogger())
    assert result.meta.task_type == "extraction"


def test_detects_tool_heavy_from_tool_result_content():
    request = _request(
        [
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "1", "name": "get_weather", "input": {}}],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "1", "content": "58F"}],
            },
        ]
    )
    result = TaskClassifierStage().run(request, {}, InMemoryCostLogger())
    assert result.meta.task_type == "tool-heavy"


def test_detects_conversational_as_fallback():
    request = _request([{"role": "user", "content": "What's the capital of France?"}])
    result = TaskClassifierStage().run(request, {}, InMemoryCostLogger())
    assert result.meta.task_type == "conversational"


def test_empty_message_is_unclassified_not_guessed():
    request = _request([{"role": "user", "content": ""}])
    result = TaskClassifierStage().run(request, {}, InMemoryCostLogger())
    assert result.meta.task_type is None


def test_confidence_is_always_set():
    request = _request([{"role": "user", "content": "hi"}])
    result = TaskClassifierStage().run(request, {}, InMemoryCostLogger())
    assert result.meta.confidence is not None


def test_runs_once_does_not_mutate_messages():
    request = _request([{"role": "user", "content": "What's the capital of France?"}])
    result = TaskClassifierStage().run(request, {}, InMemoryCostLogger())
    assert result.messages == request.messages


def test_notes_task_type_and_confidence_for_the_cost_logger():
    stage = TaskClassifierStage()
    request = _request([{"role": "user", "content": "What's the capital of France?"}])
    stage.run(request, {}, InMemoryCostLogger())
    assert stage.extra["task_type"] == "conversational"
    assert stage.extra["confidence"] > 0


def test_bounded_shape_true_for_short_single_question():
    request = _request([{"role": "user", "content": "Why is the sky blue?"}])
    result = TaskClassifierStage().run(request, {}, InMemoryCostLogger())
    assert result.meta.bounded_shape is True


def test_bounded_shape_true_for_short_question_with_code_fence():
    # A long code snippet shouldn't count against the prose length -- only
    # the actual question text should.
    request = _request(
        [
            {
                "role": "user",
                "content": (
                    "```python\n"
                    "def f(x):\n"
                    "    return x * 2\n"
                    "def g(x):\n"
                    "    return f(x) + 1\n"
                    "```\n"
                    "What does this return for x=3?"
                ),
            }
        ]
    )
    result = TaskClassifierStage().run(request, {}, InMemoryCostLogger())
    assert result.meta.bounded_shape is True


def test_bounded_shape_false_for_open_ended_marker():
    request = _request(
        [
            {
                "role": "user",
                "content": "Can you design a caching layer for our microservices architecture?",
            }
        ]
    )
    result = TaskClassifierStage().run(request, {}, InMemoryCostLogger())
    assert result.meta.bounded_shape is False


def test_bounded_shape_false_for_multiple_questions():
    request = _request(
        [
            {
                "role": "user",
                "content": "What does this function do? Also, why is it slow? And how do I fix it?",
            }
        ]
    )
    result = TaskClassifierStage().run(request, {}, InMemoryCostLogger())
    assert result.meta.bounded_shape is False


def test_bounded_shape_false_for_long_prose():
    request = _request([{"role": "user", "content": "explain " * 45 + "?"}])
    result = TaskClassifierStage().run(request, {}, InMemoryCostLogger())
    assert result.meta.bounded_shape is False


def test_bounded_shape_false_for_empty_message():
    request = _request([{"role": "user", "content": ""}])
    result = TaskClassifierStage().run(request, {}, InMemoryCostLogger())
    assert result.meta.bounded_shape is False


def test_notes_bounded_shape_for_the_cost_logger():
    stage = TaskClassifierStage()
    request = _request([{"role": "user", "content": "Why is the sky blue?"}])
    stage.run(request, {}, InMemoryCostLogger())
    assert stage.extra["bounded_shape"] is True
