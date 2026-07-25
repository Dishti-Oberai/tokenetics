"""The shared tokenizer utility -- the single source of every token count in
the project.

Wraps Anthropic's real token-counting endpoint (`messages.count_tokens`), not
a local approximation -- see CLAUDE.md's "Logging and measurement" rule.
"""

from __future__ import annotations

from typing import Any

import anthropic

from tokenetics.core.request import TokeneticsRequest, from_api_kwargs, to_api_kwargs

_COUNT_TOKENS_KEYS = {"model", "messages", "system", "tools"}


def count_tokens(request: TokeneticsRequest, client: anthropic.Anthropic) -> int:
    """Count tokens for `request` via the real Anthropic API.

    Only the fields that actually affect a token count are forwarded --
    generation-only params like max_tokens or temperature aren't part of it.
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
