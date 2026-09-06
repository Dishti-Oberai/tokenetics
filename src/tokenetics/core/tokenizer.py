"""The shared tokenizer utility -- the single source of every token count in
the project.

Wraps Anthropic's real token-counting endpoint (`messages.count_tokens`), not
a local approximation -- see CLAUDE.md's "Logging and measurement" rule.

**A real, confirmed gap found and fixed 2026-09-06**: `_COUNT_TOKENS_KEYS`
used to be `{"model", "messages", "system", "tools"}` only -- `tool_choice`
was excluded on the assumption it was a "generation-only param like
max_tokens or temperature" that doesn't affect token count. That assumption
was wrong. Caught via a real `structured_output`-plus-`brevity_injector`
A/B (extraction-shaped requests, where structured_output forces
`tool_choice` to a specific tool): the REAL measured `usage.input_tokens`
jumped ~171-172 tokens between two calls, while this project's own
before/after counting showed only ~43 tokens (matching just the brevity
instruction's own length) -- a ~128-129 token gap, consistent across two
independent samples. Confirmed against the real API reference for `POST
/v1/messages/count_tokens`: `tool_choice` IS a documented, accepted
parameter ("How the model should use the provided tools" -- explicitly
affects the count), and this magnitude matches Anthropic's own documented
tool-use system-prompt overhead difference between `auto`/`none` and
`any`/`tool` choice modes for Sonnet 5. This means every REQUEST-side
token count this project has ever logged for a `structured_output`-forced
call (dashboards, benchmark corpora, quality-check estimates) has been
UNDER-counting real input tokens whenever `tool_choice` forces a specific
tool -- not a new bug introduced by any recent stage, a measurement gap
that predates it and was simply never exercised by a real completion-based
A/B until now. `thinking` and `output_config` (which `adaptive_budget`'s
thinking-effort knob sets) are ALSO documented, accepted parameters of this
same endpoint -- included here on the same principle (forward what the
real endpoint documents as count-affecting), not because either has been
individually confirmed empirically the way `tool_choice` was.
"""

from __future__ import annotations

from typing import Any

import anthropic

from tokenetics.core.request import TokeneticsRequest, from_api_kwargs, to_api_kwargs

_COUNT_TOKENS_KEYS = {"model", "messages", "system", "tools", "tool_choice", "thinking", "output_config"}


def count_tokens(request: TokeneticsRequest, client: anthropic.Anthropic) -> int:
    """Count tokens for `request` via the real Anthropic API.

    Forwards every field the real `count_tokens` endpoint documents as
    accepted and count-affecting (see module docstring for the real gap
    this fixed) -- genuinely generation-only params like max_tokens or
    temperature, which the endpoint doesn't accept at all, still aren't
    part of it.
    """
    kwargs = to_api_kwargs(request)
    count_kwargs: dict[str, Any] = {k: v for k, v in kwargs.items() if k in _COUNT_TOKENS_KEYS}
    response = client.messages.count_tokens(**count_kwargs)
    return response.input_tokens


def count_text_tokens(text: str, model: str, client: anthropic.Anthropic) -> int:
    """Count tokens in a plain string (e.g. a model's reply) by wrapping it in
    a minimal synthetic request -- used for response-side stages, which don't
    have a full TokeneticsRequest to measure directly. Reuses the same real,
    measured count_tokens path rather than introducing a separate estimator.
    """
    request = from_api_kwargs(
        model=model, max_tokens=1, messages=[{"role": "assistant", "content": text}]
    )
    return count_tokens(request, client)
