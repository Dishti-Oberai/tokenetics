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


def test_falls_back_to_original_text_when_trim_would_empty_the_result():
    # Regression test, found via code review (2026-09-06): a reply that is
    # ENTIRELY boilerplate trims to an empty string, and the real Messages
    # API rejects empty assistant content the same way it rejects trailing
    # whitespace (already fixed once in Tokenetics.finalize()) -- an empty
    # `stored` value here would break re-injecting it as history on a later
    # turn. Conservative: keep the original rather than emit unusable
    # content.
    stage = PostHocTrimStage()
    text = "Happy to help!"
    result = stage.run(text, {}, InMemoryCostLogger())
    assert result == text
    assert stage.extra["would_have_emptied_result"] is True
