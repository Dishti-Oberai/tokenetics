from tokenetics.core.logger import InMemoryCostLogger
from tokenetics.core.request import from_api_kwargs
from tokenetics.stages.dedup import DedupStage


def _request(messages):
    return from_api_kwargs(model="claude-sonnet-5", max_tokens=100, messages=messages)


def test_removes_exact_duplicate_keeps_first_occurrence():
    request = _request(
        [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
            {"role": "user", "content": "hello"},
        ]
    )
    result = DedupStage().run(request, {}, InMemoryCostLogger())
    assert [m.content for m in result.messages] == ["hello", "hi there"]


def test_keeps_non_duplicate_messages_untouched():
    request = _request(
        [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
    )
    result = DedupStage().run(request, {}, InMemoryCostLogger())
    assert result == request


def test_same_text_different_role_is_not_deduped():
    request = _request(
        [
            {"role": "user", "content": "same text"},
            {"role": "assistant", "content": "same text"},
        ]
    )
    result = DedupStage().run(request, {}, InMemoryCostLogger())
    assert len(result.messages) == 2


def test_logs_dropped_duplicate_count():
    logger = InMemoryCostLogger()
    request = _request(
        [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "hello"},
        ]
    )
    DedupStage().run(request, {}, logger)
    assert len(logger.entries) == 1
    assert logger.entries[0].extra["dropped_duplicates"] == 1


def test_no_log_entry_when_nothing_dropped():
    logger = InMemoryCostLogger()
    request = _request([{"role": "user", "content": "hello"}])
    DedupStage().run(request, {}, logger)
    assert len(logger.entries) == 0
