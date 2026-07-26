"""Versioned model-compatibility table for stage 9c's thinking-effort knob.

Maps a model ID to the thinking API surface it accepts. Never hardcode a
per-model version check inline in stage logic -- read it from here, same
"last verified" discipline as the cache-pricing config table.

Unknown/unlisted models conservatively resolve to UNSUPPORTED: stage 9c
skips the thinking sub-stage silently (info-level log) rather than guessing
at an API shape that could 400 the request. Entries below are current-
generation models Tokenetics actually targets; models confirmed here as
UNSUPPORTED are cases where a live check couldn't be done and skipping is
the safe default, not necessarily a permanent "never supports thinking."
"""

from __future__ import annotations

from typing import Literal

ThinkingSupport = Literal["adaptive", "legacy_budget", "unsupported"]

LAST_VERIFIED = "2026-07-26"

MODEL_THINKING_SUPPORT: dict[str, ThinkingSupport] = {
    "claude-fable-5": "adaptive",
    "claude-mythos-5": "adaptive",
    "claude-opus-4-8": "adaptive",
    "claude-opus-4-7": "adaptive",
    "claude-opus-4-6": "adaptive",
    "claude-sonnet-5": "adaptive",
    "claude-sonnet-4-6": "adaptive",
    # Haiku 4.5's extended-thinking support isn't confirmed against live docs
    # as of LAST_VERIFIED -- conservatively unsupported rather than guessed.
    "claude-haiku-4-5": "unsupported",
}


def thinking_support_for(model: str) -> ThinkingSupport:
    return MODEL_THINKING_SUPPORT.get(model, "unsupported")
