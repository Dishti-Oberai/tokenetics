#!/usr/bin/env python3
"""Synthetic-traffic benchmark for stage 8 (cache breakpoint optimizer).

NOT part of pytest or CI -- but unlike dev_demo.py, this makes NO network
calls at all and needs no API key. It's synthetic on purpose: CLAUDE.md's
build order calls for a "synthetic-traffic benchmark (the headline number)"
as part of Phase 6 itself, distinct from the real, measured, held-out-corpus
benchmark suite that's Phase 9's job. This script simulates a stream of
calls with realistic repeat patterns and reports the cache-hit-rate and
estimated cost improvement CacheBreakpointOptimizerStage actually achieves
-- using the real stage, the real pricing table, and a simulated clock
standing in for Anthropic's actual cache expiry behavior (which nothing
Tokenetics can call locally reproduces).

Run it:

    uv run python scripts/cache_benchmark.py
    uv run python scripts/cache_benchmark.py --calls 2000 --seed 7
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass

from tokenetics.core.cache_hash import stable_prefix_hash
from tokenetics.core.cache_pricing import (
    READ_MULTIPLIER,
    TTL_1H_SECONDS,
    TTL_5M_SECONDS,
    WRITE_MULTIPLIER_1H,
    WRITE_MULTIPLIER_5M,
    base_input_price_for,
)
from tokenetics.core.logger import NullLogger
from tokenetics.core.request import ToolSpec, from_api_kwargs
from tokenetics.stages.cache_breakpoint_optimizer import CacheBreakpointOptimizerStage

_MODEL = "claude-sonnet-5"
_BASE_PRICE_PER_TOKEN = base_input_price_for(_MODEL) / 1_000_000
_TTL_WINDOW_SECONDS = {"5m": TTL_5M_SECONDS, "1h": TTL_1H_SECONDS}
_WRITE_MULTIPLIER = {"5m": WRITE_MULTIPLIER_5M, "1h": WRITE_MULTIPLIER_1H}


@dataclass
class Profile:
    """One simulated "agent config" -- a distinct stable prefix with its
    own typical reuse interval. Real traffic is a mix of hot configs
    (called constantly, e.g. a chat app's standing system prompt) and cold
    ones (called rarely, e.g. a one-off script).
    """

    name: str
    system: str
    tools: list[ToolSpec]
    typical_gap_seconds: float
    weight: float
    prefix_tokens: int


def _make_profiles() -> list[Profile]:
    # System prompts sized well above claude-sonnet-5's real 1024-token
    # minimum cacheable prefix (core.cache_pricing.MIN_CACHEABLE_TOKENS) --
    # CacheBreakpointOptimizerStage now declines to place a breakpoint below
    # that, same as the real API would silently ignore one. prefix_tokens
    # (used only for this script's own cost arithmetic) tracks each system
    # prompt's actual approximate length so the two stay consistent.
    hot_tool = ToolSpec(
        name="search_kb",
        description="Search the internal knowledge base for relevant articles",
        input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
    )
    return [
        Profile(
            "hot_chat_agent",
            system="You are a customer support agent. " * 200,
            tools=[hot_tool],
            typical_gap_seconds=30,
            weight=0.5,
            prefix_tokens=1800,
        ),
        Profile(
            "warm_batch_job",
            system="You are a data extraction pipeline. " * 160,
            tools=[],
            typical_gap_seconds=420,
            weight=0.3,
            prefix_tokens=1450,
        ),
        Profile(
            "cold_one_off",
            system="You are a one-off analysis assistant. " * 140,
            tools=[],
            typical_gap_seconds=5000,
            weight=0.2,
            prefix_tokens=1400,
        ),
    ]


def _simulate(num_calls: int, seed: int) -> None:
    rng = random.Random(seed)
    profiles = _make_profiles()
    weights = [p.weight for p in profiles]

    stage = CacheBreakpointOptimizerStage()
    history: list[dict[str, object]] = []
    # per-hash simulated cache state: (expires_at, ttl)
    cache_state: dict[str, tuple[float, str]] = {}

    now = 0.0
    total_cost_with_optimizer = 0.0
    total_cost_never_cache = 0.0
    total_cost_naive_5m = 0.0
    naive_cache_state: dict[str, float] = {}
    reads = 0
    writes = 0
    full_price_calls = 0

    for _ in range(num_calls):
        profile = rng.choices(profiles, weights=weights, k=1)[0]
        now += rng.expovariate(1 / profile.typical_gap_seconds)

        request = from_api_kwargs(
            model=_MODEL,
            max_tokens=100,
            system=profile.system,
            tools=[
                {"name": t.name, "description": t.description, "input_schema": t.input_schema}
                for t in profile.tools
            ],
            messages=[{"role": "user", "content": "hi"}],
        )
        content_hash = stable_prefix_hash(profile.system, profile.tools)

        config = {"cache_usage_history": [h for h in history if h["content_hash"] == content_hash]}
        result = stage.run(request, config, NullLogger())
        history.append({"timestamp": now, "content_hash": content_hash})

        # -- with-optimizer billing, using our simulated cache clock --
        if result.cache_breakpoint is None:
            total_cost_with_optimizer += profile.prefix_tokens * _BASE_PRICE_PER_TOKEN
            full_price_calls += 1
        else:
            ttl = result.cache_breakpoint.ttl
            expiry, _ = cache_state.get(content_hash, (-1.0, ttl))
            if now <= expiry:
                total_cost_with_optimizer += (
                    profile.prefix_tokens * _BASE_PRICE_PER_TOKEN * READ_MULTIPLIER
                )
                reads += 1
            else:
                total_cost_with_optimizer += (
                    profile.prefix_tokens * _BASE_PRICE_PER_TOKEN * _WRITE_MULTIPLIER[ttl]
                )
                writes += 1
            cache_state[content_hash] = (now + _TTL_WINDOW_SECONDS[ttl], ttl)

        # -- baseline: never cache, always full price --
        total_cost_never_cache += profile.prefix_tokens * _BASE_PRICE_PER_TOKEN

        # -- baseline: naive habit of always placing a 5m breakpoint --
        naive_expiry = naive_cache_state.get(content_hash, -1.0)
        if now <= naive_expiry:
            total_cost_naive_5m += profile.prefix_tokens * _BASE_PRICE_PER_TOKEN * READ_MULTIPLIER
        else:
            total_cost_naive_5m += (
                profile.prefix_tokens * _BASE_PRICE_PER_TOKEN * WRITE_MULTIPLIER_5M
            )
        naive_cache_state[content_hash] = now + TTL_5M_SECONDS

    decided_calls = reads + writes
    hit_rate = reads / decided_calls if decided_calls else 0.0
    savings_vs_never_cache = 1 - (total_cost_with_optimizer / total_cost_never_cache)
    savings_vs_naive = 1 - (total_cost_with_optimizer / total_cost_naive_5m)

    print(f"calls simulated: {num_calls} (seed={seed})")
    print(f"breakpoint placed: {decided_calls}/{num_calls}  (no-breakpoint/full-price: {full_price_calls})")
    print(f"cache hit rate (reads / decided calls): {hit_rate:.1%}")
    print()
    print("estimated relative cost (arbitrary units, pricing-table-derived):")
    print(f"  never cache (baseline):        {total_cost_never_cache:,.2f}")
    print(f"  naive always-5m breakpoint:    {total_cost_naive_5m:,.2f}")
    print(f"  Tokenetics breakpoint optimizer: {total_cost_with_optimizer:,.2f}")
    print()
    print(f"HEADLINE: {savings_vs_never_cache:.1%} cheaper than never caching")
    print(f"          {savings_vs_naive:.1%} cheaper than a naive always-5m breakpoint habit")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calls", type=int, default=1000, help="Number of synthetic calls to simulate.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed, for reproducible runs.")
    args = parser.parse_args()
    _simulate(args.calls, args.seed)


if __name__ == "__main__":
    main()
