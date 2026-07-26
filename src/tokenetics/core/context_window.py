"""Versioned model context-window lookup table + task-type budget scaling,
for stage 5's fallback history token budget.

Used only when the caller doesn't supply `config["token_budget"]`
themselves -- see `default_token_budget`'s docstring. Never hardcode a
context-window size inline in stage logic -- read it from here, same
"last verified" discipline as the cache-pricing and model-compatibility
tables.
"""

from __future__ import annotations

LAST_VERIFIED = "2026-07-26"

CONTEXT_WINDOW_TOKENS: dict[str, int] = {
    "claude-fable-5": 1_000_000,
    "claude-mythos-5": 1_000_000,
    "claude-opus-4-8": 1_000_000,
    "claude-opus-4-7": 1_000_000,
    "claude-opus-4-6": 1_000_000,
    "claude-sonnet-5": 1_000_000,
    "claude-sonnet-4-6": 1_000_000,
    "claude-haiku-4-5": 200_000,
}
# Conservative fallback for an unlisted model -- the smallest window among
# known current-generation models. Never assume more room than guaranteed.
_UNKNOWN_MODEL_CONTEXT_WINDOW = 200_000

# What share of the context window the history budget gets, by task type --
# heuristic defaults confirmed with the user 2026-07-26, not yet benchmarked
# (that's Phase 9's job). Code-heavy conversations retain more history than
# short extraction/small-talk ones.
_TASK_TYPE_BUDGET_FRACTION: dict[str, float] = {
    "code": 0.5,
    "tool-heavy": 0.35,
    "conversational": 0.3,
    "extraction": 0.2,
}
_DEFAULT_BUDGET_FRACTION = 0.3  # unclassified task_type -- same as conversational


def context_window_for(model: str) -> int:
    return CONTEXT_WINDOW_TOKENS.get(model, _UNKNOWN_MODEL_CONTEXT_WINDOW)


def default_token_budget(model: str, task_type: str | None) -> int:
    """Fallback history token budget when the caller doesn't supply one.

    Derived fresh from the model's context window on every call -- no
    state, no persistence. Callers should tag this value `estimated`, not
    `measured`, when logging it, since it's a heuristic default rather than
    a value the caller actually supplied.
    """
    fraction = _TASK_TYPE_BUDGET_FRACTION.get(task_type or "", _DEFAULT_BUDGET_FRACTION)
    return round(context_window_for(model) * fraction)
