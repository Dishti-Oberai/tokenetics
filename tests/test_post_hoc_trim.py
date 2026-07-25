from tokenetics.core.logger import InMemoryCostLogger
from tokenetics.stages.post_hoc_trim import PostHocTrimStage


def test_strips_trailing_boilerplate():
    text = "Here's the answer: 42.\n\nLet me know if you have any questions!"
    result = PostHocTrimStage().run(text, {}, InMemoryCostLogger())
    assert result == "Here's the answer: 42."


def test_strips_stacked_boilerplate():
    text = "The answer is 42. I hope this helps! Let me know if you need anything else."
    result = PostHocTrimStage().run(text, {}, InMemoryCostLogger())
    assert result == "The answer is 42."


def test_no_false_positive_on_legitimate_content():
    text = "To read user input, call input(). Let the value flow into the next step."
    result = PostHocTrimStage().run(text, {}, InMemoryCostLogger())
    assert result == text


def test_logs_when_boilerplate_removed():
    logger = InMemoryCostLogger()
    text = "Done. Let me know if you have any questions!"
    PostHocTrimStage().run(text, {}, logger)
    assert len(logger.entries) == 1
    assert logger.entries[0].extra["boilerplate_removed"] >= 1


def test_no_log_entry_when_nothing_removed():
    logger = InMemoryCostLogger()
    PostHocTrimStage().run("Plain answer, nothing to trim.", {}, logger)
    assert len(logger.entries) == 0
