from tokenetics.core.context_window import context_window_for, default_token_budget


def test_context_window_for_known_model():
    assert context_window_for("claude-sonnet-5") == 1_000_000


def test_context_window_for_unknown_model_is_conservative():
    # Smallest known window, never assume more room than guaranteed.
    assert context_window_for("some-model-nobody-has-heard-of") == 200_000


def test_default_budget_scales_by_task_type():
    code_budget = default_token_budget("claude-sonnet-5", "code")
    extraction_budget = default_token_budget("claude-sonnet-5", "extraction")
    assert code_budget > extraction_budget


def test_default_budget_for_unclassified_task_type_matches_conversational():
    unclassified = default_token_budget("claude-sonnet-5", None)
    conversational = default_token_budget("claude-sonnet-5", "conversational")
    assert unclassified == conversational


def test_default_budget_is_a_fraction_of_the_context_window():
    budget = default_token_budget("claude-sonnet-5", "code")
    assert 0 < budget < context_window_for("claude-sonnet-5")
