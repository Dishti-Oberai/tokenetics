from tokenetics.core.model_compatibility import thinking_support_for


def test_current_generation_models_are_adaptive():
    for model in [
        "claude-fable-5",
        "claude-opus-4-8",
        "claude-opus-4-7",
        "claude-sonnet-5",
        "claude-sonnet-4-6",
    ]:
        assert thinking_support_for(model) == "adaptive"


def test_unknown_model_conservatively_unsupported():
    assert thinking_support_for("some-model-nobody-has-heard-of") == "unsupported"


def test_unconfirmed_model_is_unsupported_not_guessed():
    # Haiku 4.5's thinking support isn't confirmed against live docs -- must
    # resolve to unsupported (fail-open, skip) rather than a guessed "adaptive".
    assert thinking_support_for("claude-haiku-4-5") == "unsupported"
