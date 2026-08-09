"""Versioned prompt-cache and base model pricing config table.

Never hardcode a cache price, output price, or multiplier inline in stage
logic -- read it from here. Sourced from a live fetch of Anthropic's pricing
docs on LAST_VERIFIED below, not a remembered/guessed figure -- a wrong price
actively costs the caller money (or, for the benchmark suite's cost-ceiling
estimate, under/overestimates real spend before a billed run), not just
mis-optimizes like a stale model-capability entry would.

The write/read multipliers are universal (stated as multipliers on top of
each model's own base input price, not separate per-model figures), so this
table only needs to track the multipliers once plus each model's base
input/output price.

**Caught stale while sourcing OUTPUT_PRICE_PER_MTOK for Phase 9 (2026-08-02)**:
`BASE_INPUT_PRICE_PER_MTOK["claude-sonnet-5"]` was hardcoded to $3.00, but the
live-verified current price is $2.00/MTok -- Sonnet 5 has introductory
pricing in effect through 2026-08-31 (step-up to $3.00/$15.00 on 2026-09-01,
per https://platform.claude.com/docs/en/about-claude/pricing). This table
does not model the step-up automatically (no date-based logic, deliberately
-- that's real complexity for one scheduled change); whoever touches this
file after 2026-09-01 should re-verify and update both entries, not just
trust LAST_VERIFIED blindly past that date.
"""

from __future__ import annotations

LAST_VERIFIED = "2026-08-02"
# Two source pages, both re-verified on LAST_VERIFIED: the model pricing page
# covers base input/output prices and the cache multipliers (it documents
# both); the prompt-caching page is the source for MIN_CACHEABLE_TOKENS,
# which isn't on the pricing page at all.
SOURCE_PRICING = "https://platform.claude.com/docs/en/about-claude/pricing"
SOURCE_PROMPT_CACHING = "https://platform.claude.com/docs/en/build-with-claude/prompt-caching"

# Cache write tokens cost this multiple of the base input token price,
# per TTL tier. Cache read (hit/refresh) tokens cost this multiple of the
# base input token price, regardless of tier.
WRITE_MULTIPLIER_5M = 1.25
WRITE_MULTIPLIER_1H = 2.0
READ_MULTIPLIER = 0.1

# Base input price in USD per million tokens. Used both for the cache
# breakeven math (Tokenetics never bills anyone -- purely a relative-cost
# comparison there) and, as of Phase 9, the benchmark suite's real
# cost-ceiling estimate before a billed run.
BASE_INPUT_PRICE_PER_MTOK: dict[str, float] = {
    "claude-fable-5": 10.00,
    "claude-mythos-5": 10.00,
    "claude-opus-5": 5.00,
    "claude-opus-4-8": 5.00,
    "claude-opus-4-7": 5.00,
    "claude-opus-4-6": 5.00,
    "claude-sonnet-5": 2.00,  # introductory price through 2026-08-31 -- see module docstring
    "claude-sonnet-4-6": 3.00,
    "claude-haiku-4-5": 1.00,
}
# Conservative fallback for an unlisted model -- the highest known current-
# generation base price, never assume a cheaper price than guaranteed (that
# would understate the write premium's real cost in the breakeven math, or
# underestimate a benchmark run's real cost).
_UNKNOWN_MODEL_BASE_PRICE = 10.00

# Output price in USD per million tokens -- same models, same source.
OUTPUT_PRICE_PER_MTOK: dict[str, float] = {
    "claude-fable-5": 50.00,
    "claude-mythos-5": 50.00,
    "claude-opus-5": 25.00,
    "claude-opus-4-8": 25.00,
    "claude-opus-4-7": 25.00,
    "claude-opus-4-6": 25.00,
    "claude-sonnet-5": 10.00,  # introductory price through 2026-08-31 -- see module docstring
    "claude-sonnet-4-6": 15.00,
    "claude-haiku-4-5": 5.00,
}
_UNKNOWN_MODEL_OUTPUT_PRICE = 50.00

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


def output_price_for(model: str) -> float:
    return OUTPUT_PRICE_PER_MTOK.get(model, _UNKNOWN_MODEL_OUTPUT_PRICE)


def min_cacheable_tokens_for(model: str) -> int:
    return MIN_CACHEABLE_TOKENS.get(model, _UNKNOWN_MODEL_MIN_CACHEABLE_TOKENS)
