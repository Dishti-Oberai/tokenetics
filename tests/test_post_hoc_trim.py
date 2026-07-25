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


def test_notes_when_boilerplate_removed():
    stage = PostHocTrimStage()
    text = "Done. Let me know if you have any questions!"
    stage.run(text, {}, InMemoryCostLogger())
    assert stage.extra["boilerplate_removed"] >= 1


def test_no_note_when_nothing_removed():
    stage = PostHocTrimStage()
    stage.run("Plain answer, nothing to trim.", {}, InMemoryCostLogger())
    assert stage.extra == {}
