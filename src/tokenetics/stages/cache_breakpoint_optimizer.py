"""Stage 8: cache breakpoint optimizer (headline feature).

Decides *where* to place a `cache_control` marker and *which* TTL tier
(5-minute vs 1-hour) to use, instead of the common habit of one breakpoint
at the end of the system prompt with no cost model behind it.

Scoped to the same two anchors stage 7's guard protects: "system" (if
present -- covers tools+system, see below) or "tools" (only when there's
no system to fold in). Per-message breakpoints are a natural future
extension once delta compression's payload model exists; not attempted
here.

Anchor choice depends on Anthropic's documented cache hierarchy -- tools,
then system, then messages, each breakpoint caching everything up to AND
INCLUDING the block it's placed on. A marker on the last tool caches ONLY
the tools array; system (which comes after tools in that hierarchy) is
NOT included unless the marker is placed on system instead. An earlier
version of this stage anchored to "tools" whenever tools were present,
on the mistaken assumption that a tools-anchored marker also covered the
system prompt before it -- this was backwards (confusing JSON key order,
which Anthropic doesn't care about, with the actual cache hierarchy, which
it does). Caught 2026-08-02 via `--measure-usage` against the real API:
the `cache_hit` demo scenario logged `breakpoint_placed=True` but real
`usage.cache_creation_input_tokens`/`cache_read_input_tokens` both stayed
0 -- the tools-only prefix being cached (~30 tokens) was far below the
model's minimum cacheable length, even though system+tools combined
cleared it comfortably. Confirmed against a live fetch of
https://platform.claude.com/docs/en/build-with-claude/prompt-caching.

Usage-history shape (resolved with the user 2026-07-27): caller-supplied
`config["cache_usage_history"]`, a list of small
`{"timestamp": float, "content_hash": str}` records -- not full past
request bodies. `content_hash` must be computed with
`core.cache_hash.stable_prefix_hash` so it matches what this stage
computes for the current request; entries with a different hash (a
different stable prefix, from a different task) are ignored.

Conservative by design: with fewer than 2 matching history entries there's
no repeat interval to estimate, so no breakpoint is placed rather than
guessed at. Same conservatism applies to prompt size: below each model's
documented minimum cacheable prefix length (`core.cache_pricing.MIN_CACHEABLE_TOKENS`),
the real API silently skips caching a `cache_control` marker entirely (no
error) -- confirmed live against the prompt-caching docs on 2026-08-02 after
a demo scenario's real `usage.cache_creation_input_tokens`/`cache_read_input_tokens`
both came back 0 despite this stage reporting `breakpoint_placed=True`. This
stage checks that minimum before placing a breakpoint, rather than claiming
a benefit the API will never actually deliver.

TTL choice: a cache *read* refreshes that cache entry's own TTL on
Anthropic's side, so once a tier's window comfortably covers the observed
repeat gap, the cache never actually expires between repeats -- it stays
continuously warm off a single amortized write. Read pricing (0.1x) is
identical across tiers, so once a tier is warm the only real cost
difference left is the one-time write premium (1.25x for 5-minute, 2x for
1-hour). That means the cheaper tier should always win when it's viable;
the 1-hour tier is only worth its larger premium when the 5-minute window
*wouldn't* stay warm between repeats. (An earlier version of this stage
compared each tier's *total* expected benefit over its own window
independently -- which, since a longer window always nets more total
reads, meant the 1-hour tier's larger window mathematically dominated even
at gaps well under 5 minutes, and the 5-minute tier could never actually
be selected. Caught while writing this stage's tests.) `core.cache_pricing`
is still used, for the informational net-benefit estimate logged alongside
the decision -- never hardcoded inline.
"""

from __future__ import annotations

from dataclasses import replace

from tokenetics.core.cache_hash import stable_prefix_hash
from tokenetics.core.cache_pricing import (
    READ_MULTIPLIER,
    TTL_1H_SECONDS,
    TTL_5M_SECONDS,
    WRITE_MULTIPLIER_1H,
    WRITE_MULTIPLIER_5M,
    base_input_price_for,
    min_cacheable_tokens_for,
)
from tokenetics.core.logger import CostLogger
from tokenetics.core.plugin import Stage, StageConfig
from tokenetics.core.request import CacheAnchor, CacheBreakpoint, CacheTTL, TokeneticsRequest

_TTL_WINDOWS: list[tuple[CacheTTL, int, float]] = [
    ("5m", TTL_5M_SECONDS, WRITE_MULTIPLIER_5M),
    ("1h", TTL_1H_SECONDS, WRITE_MULTIPLIER_1H),
]


def _estimate_tokens(text: str) -> int:
    # Same fast, local, ~4-chars/token estimate used elsewhere (context
    # scheduler) for internal sizing -- stages have no client/network
    # access, so this can't call the real tokenizer.
    return max(1, len(text) // 4)


def _choose_ttl(avg_gap_seconds: float) -> CacheTTL | None:
    if avg_gap_seconds <= 0:
        return None
    for ttl, window_seconds, _write_multiplier in _TTL_WINDOWS:
        if avg_gap_seconds <= window_seconds:
            return ttl  # cheapest tier that would stay continuously warm
    return None  # wider than even the 1h window -- would expire before reuse


def _estimated_net_benefit_per_token(ttl: CacheTTL, avg_gap_seconds: float, model: str) -> float:
    """Informational only -- does not drive the tier choice above, just
    logged alongside it so the decision has a real number behind it.
    """
    window_seconds, write_multiplier = next(
        (w, m) for t, w, m in _TTL_WINDOWS if t == ttl
    )
    base = base_input_price_for(model) / 1_000_000
    savings_per_read = base - base * READ_MULTIPLIER
    expected_reads = window_seconds / avg_gap_seconds
    write_premium = (write_multiplier - 1) * base
    return expected_reads * savings_per_read - write_premium


class CacheBreakpointOptimizerStage(Stage):
    name = "cache_breakpoint_optimizer"

    def run(
        self, request: TokeneticsRequest, config: StageConfig, logger: CostLogger
    ) -> TokeneticsRequest:
        if not request.system and not request.tools:
            return request  # nothing stable to cache

        # Anthropic's cache hierarchy is tools -> system -> messages (a
        # breakpoint caches everything up to AND INCLUDING the block it's on,
        # in that order) -- so a marker on "system" covers tools+system
        # together, while a marker on "tools" covers ONLY the tools array,
        # since system comes after it and is excluded. Anchor to "system"
        # whenever one exists; "tools" is only correct when there's no
        # system to fold in.
        anchor: CacheAnchor = "system" if request.system else "tools"
        current_hash = stable_prefix_hash(request.system, request.tools)

        history: list[dict[str, object]] = config.get("cache_usage_history", [])
        matching_timestamps = sorted(
            float(h["timestamp"])  # type: ignore[arg-type]
            for h in history
            if h.get("content_hash") == current_hash
        )
        if len(matching_timestamps) < 2:
            # Not enough data to estimate a repeat interval -- conservative:
            # don't guess at a breakpoint.
            self.note(breakpoint_placed=False, reason="insufficient_usage_history")
            return request

        gaps = [b - a for a, b in zip(matching_timestamps, matching_timestamps[1:])]
        avg_gap = sum(gaps) / len(gaps)

        size_text = (request.system or "") + "".join(
            t.name + t.description for t in request.tools
        )
        size_tokens = _estimate_tokens(size_text)
        min_required = min_cacheable_tokens_for(request.model)
        if size_tokens < min_required:
            # The API silently skips caching below this length -- no error,
            # just an ordinary uncached request. Placing a breakpoint here
            # would claim a benefit that can never materialize.
            self.note(
                breakpoint_placed=False,
                reason="prefix_below_minimum_cacheable_size",
                estimated_size_tokens=size_tokens,
                minimum_required_tokens=min_required,
            )
            return request

        ttl = _choose_ttl(avg_gap)
        if ttl is None:
            self.note(
                breakpoint_placed=False,
                reason="not_cost_effective",
                avg_repeat_gap_seconds=round(avg_gap, 1),
            )
            return request

        net_benefit_per_token = _estimated_net_benefit_per_token(ttl, avg_gap, request.model)

        self.note(
            breakpoint_placed=True,
            cache_anchor=anchor,
            cache_ttl=ttl,
            avg_repeat_gap_seconds=round(avg_gap, 1),
            estimated_size_tokens=size_tokens,
            estimated_net_benefit_per_token=round(net_benefit_per_token, 8),
        )
        return replace(request, cache_breakpoint=CacheBreakpoint(anchor=anchor, ttl=ttl))
