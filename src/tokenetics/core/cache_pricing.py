"""Versioned prompt-cache pricing config table.

Never hardcode a cache price or multiplier inline in stage logic -- read it
from here. Sourced from a live fetch of Anthropic's prompt-caching pricing
docs (https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
on LAST_VERIFIED below, not a remembered/guessed figure -- a wrong cache
price actively costs the caller money on every cache write, not just
mis-optimizes like a stale model-capability entry would.

The write/read multipliers are universal (stated as multipliers on top of
each model's own base input price, not separate per-model figures), so this
table only needs to track the multipliers once plus each model's base
input price.
"""

from __future__ import annotations

LAST_VERIFIED = "2026-08-02"
SOURCE = "https://platform.claude.com/docs/en/build-with-claude/prompt-caching"

# Cache write tokens cost this multiple of the base input token price,
# per TTL tier. Cache read (hit/refresh) tokens cost this multiple of the
# base input token price, regardless of tier.
WRITE_MULTIPLIER_5M = 1.25
WRITE_MULTIPLIER_1H = 2.0
READ_MULTIPLIER = 0.1

# Base input price in USD per million tokens. Used only to compute the
# breakeven math for breakpoint/TTL choice -- Tokenetics never bills
# anyone, this is purely a relative-cost comparison.
BASE_INPUT_PRICE_PER_MTOK: dict[str, float] = {
    "claude-fable-5": 10.00,
    "claude-mythos-5": 10.00,
    "claude-opus-4-8": 5.00,
    "claude-opus-4-7": 5.00,
    "claude-opus-4-6": 5.00,
    "claude-sonnet-5": 3.00,
    "claude-sonnet-4-6": 3.00,
    "claude-haiku-4-5": 1.00,
}
# Conservative fallback for an unlisted model -- the highest known current-
# generation base price, never assume a cheaper price than guaranteed (that
# would understate the write premium's real cost in the breakeven math).
_UNKNOWN_MODEL_BASE_PRICE = 10.00

TTL_5M_SECONDS = 5 * 60
TTL_1H_SECONDS = 60 * 60

# Minimum prompt-prefix length (tokens) below which the API silently skips
# caching a cache_control breakpoint entirely -- no error, no cache_creation/
# cache_read tokens in usage, just processed as an ordinary uncached request.
# A stage that places a breakpoint without checking this can confidently log
# "breakpoint_placed=True" for a breakpoint the real API will just ignore.
MIN_CACHEABLE_TOKENS: dict[str, int] = {
    "claude-opus-5": 512,
    "claude-fable-5": 512,
    "claude-mythos-5": 512,
    "claude-opus-4-8": 1024,
    "claude-sonnet-5": 1024,
    "claude-sonnet-4-6": 1024,
    "claude-opus-4-7": 2048,
    "claude-opus-4-6": 4096,
    "claude-haiku-4-5": 4096,
}
# Conservative fallback for an unlisted model -- the highest known minimum,
# never assume a smaller prefix is cacheable than guaranteed.
_UNKNOWN_MODEL_MIN_CACHEABLE_TOKENS = 4096


def base_input_price_for(model: str) -> float:
    return BASE_INPUT_PRICE_PER_MTOK.get(model, _UNKNOWN_MODEL_BASE_PRICE)


def min_cacheable_tokens_for(model: str) -> int:
    return MIN_CACHEABLE_TOKENS.get(model, _UNKNOWN_MODEL_MIN_CACHEABLE_TOKENS)
