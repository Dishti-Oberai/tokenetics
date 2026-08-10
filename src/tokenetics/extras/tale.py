"""Tier 2c: TALE-style budget estimation (opt-in, Phase 10).

Per the brief: stage 9c's `max_tokens` heuristic (a flat per-task-type
estimate) is coarse. TALE-style estimation asks the model itself, via a
small, cheap side-call, how many tokens the *specific* upcoming request
will actually need -- then feeds that number into stage 9c's existing
safety-margin/truncation-widening machinery via
`config["external_budget_estimate"]` (see `adaptive_budget.py`).

Deliberately NOT part of Tier 0: the core pipeline (`Tokenetics.prepare()`)
must stay stateless and network-free (CLAUDE.md), so this module is a
separate, explicitly opt-in helper the caller runs themselves, before
`prepare()`, and feeds the result into `stage_config` -- the same shape
every other caller-supplied stage input already uses (`truncation_stats`,
`cache_usage_history`, `previous_payloads`).

Fail-open (per CLAUDE.md's "hard error" flavor): any failure in the
estimation call itself -- network error, malformed response, an
unparseable answer -- returns `None` rather than raising. The caller (or
`adaptive_budget`) then falls back to the free heuristic untouched; a
failed *estimate* must never block or corrupt the real request.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

import anthropic

_log = logging.getLogger(__name__)

_ESTIMATION_MODEL_DEFAULT = "claude-haiku-4-5"
_ESTIMATION_MAX_TOKENS = 20
_ESTIMATION_SYSTEM_PROMPT = (
    "You estimate how many output tokens a response to the given request will "
    "need. Reply with ONLY a single integer -- no words, no punctuation, no "
    "explanation. Round up generously; underestimating causes truncation."
)
_INTEGER_RE = re.compile(r"\d+")


@dataclass(frozen=True)
class TaleEstimate:
    """Result of a TALE-style estimation call.

    `estimation_input_tokens`/`estimation_output_tokens` are the real,
    measured cost of the side-call itself (from `response.usage`) -- tagged
    `measured`, per CLAUDE.md, so callers/benchmarks can honestly weigh this
    against what the tighter budget actually saved, not guess at it.
    """

    budget_tokens: int
    estimation_input_tokens: int
    estimation_output_tokens: int


def estimate_token_budget(
    client: anthropic.Anthropic,
    *,
    messages: list[dict[str, Any]],
    model: str = "claude-sonnet-5",
    system: str | list[dict[str, Any]] | None = None,
    estimation_model: str = _ESTIMATION_MODEL_DEFAULT,
) -> TaleEstimate | None:
    """Ask a small, cheap model to estimate the output-token budget the real
    request (`model`, `messages`, `system`) will need. Returns `None` on any
    failure -- fail-open, never raises.

    `estimation_model` defaults to a cheaper model than `model` itself
    (Haiku), since the estimation call's own cost has to stay small relative
    to what it might save -- see the module docstring and the head-to-head
    benchmark this is meant to justify itself against.
    """
    prompt_lines = [
        f"The request will be sent to model {model!r}.",
        "Conversation so far:",
        _render_messages(messages),
    ]
    if system:
        prompt_lines.insert(1, f"System prompt: {_render_system(system)}")

    try:
        response = client.messages.create(
            model=estimation_model,
            max_tokens=_ESTIMATION_MAX_TOKENS,
            system=_ESTIMATION_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": "\n".join(prompt_lines)}],
        )
    except Exception:
        _log.info("TALE estimation call failed; falling back to the free heuristic", exc_info=True)
        return None

    text = "".join(
        getattr(block, "text", "") for block in response.content if getattr(block, "type", None) == "text"
    )
    match = _INTEGER_RE.search(text)
    if match is None:
        _log.info("TALE estimation response had no parseable integer (%r); falling back", text)
        return None

    usage = getattr(response, "usage", None)
    return TaleEstimate(
        budget_tokens=int(match.group()),
        estimation_input_tokens=getattr(usage, "input_tokens", 0) if usage else 0,
        estimation_output_tokens=getattr(usage, "output_tokens", 0) if usage else 0,
    )


def _render_messages(messages: list[dict[str, Any]]) -> str:
    lines = []
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                block.get("text", f"[{block.get('type', 'block')}]")
                for block in content
                if isinstance(block, dict)
            )
        lines.append(f"{message.get('role', 'user')}: {content}")
    return "\n".join(lines)


def _render_system(system: str | list[dict[str, Any]]) -> str:
    if isinstance(system, str):
        return system
    return " ".join(block.get("text", "") for block in system)
