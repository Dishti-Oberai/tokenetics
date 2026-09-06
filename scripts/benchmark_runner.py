#!/usr/bin/env python3
"""Phase 9 benchmark suite runner.

NOT part of pytest or CI -- this makes real, billed API calls (`quality-check`
and `usage` do; `corpus` is free, prepare()-only). See CLAUDE.md's Phase 9
build-order entry and `src/tokenetics/core/benchmark.py` for the corpus/
quality-check design.

    uv run python scripts/benchmark_runner.py corpus
    uv run python scripts/benchmark_runner.py quality-check --dry-run
    uv run python scripts/benchmark_runner.py quality-check --confirm-spend
    uv run python scripts/benchmark_runner.py usage --dry-run
    uv run python scripts/benchmark_runner.py usage --confirm-spend

Three subcommands:

- `corpus`: runs every sample in benchmarks/corpus/ through prepare() only
  (free -- no completions, same cost model as dev_demo.py's --all-scenarios)
  and reports token savings as a MIN-MAX range per task-type category, never
  a single flat percentage, per CLAUDE.md's honest-benchmarking discipline.
  Request-side only -- structurally blind to output-token and cache-tier
  cost, by design (see `usage` below for those).

- `quality-check`: runs every sample in benchmarks/quality_checks/ as a real
  completion, once with its named stage enabled and once with it disabled,
  and checks the response against that sample's `required_elements` (the
  authoritative pass/fail gate).

- `usage` (added after Phase 11, per a real finding): runs the SAME 82+
  corpus samples `corpus` uses, but as real completions (baseline vs.
  pipeline-enabled, `response.usage`-measured), computing the REAL dollar
  cost of each via `core.cache_pricing`'s live-verified table -- input,
  output, and cache tokens all correctly weighted by their real, different
  per-token prices. This exists because `dev_demo.py --measure-usage`
  revealed a real finding the free `corpus` numbers structurally can't see:
  `adaptive_budget`'s wider max_tokens/higher thinking effort can generate
  enough EXTRA output tokens (priced ~5x higher than input) to outweigh
  real input-side savings in dollar terms -- but `dev_demo.py`'s own
  scenario set is a handful of stage-demonstration fixtures deliberately
  biased toward triggering every mechanism at once, not a representative
  sample. The corpus (already vetted for "genuine variety, not padding",
  20+ samples per category) is a much better base for this than that.

  Applies `_REALISTIC_MAX_TOKENS_FLOOR` to BOTH the baseline and pipeline
  calls (see that constant's own comment for the full story): the corpus's
  own max_tokens values are deliberately tiny, chosen for Phase 9 to
  exercise adaptive_budget's widening mechanism, not to represent a
  realistic caller's cap. A real diagnostic run (2026-09-04) confirmed
  every completion in an unmodified run hit `stop_reason="max_tokens"` --
  both sides were truncated, and the pipeline just wrote more before ALSO
  getting cut off, which is not a fair test of real efficiency.

All three subcommands that spend real money are gated behind an upfront
cost estimate: refuse to run past --cost-ceiling (default $1.00) even with
--confirm-spend, and refuse to run AT ALL without --confirm-spend
regardless of estimate size.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import anthropic

from tokenetics import Tokenetics
from tokenetics.core.cache_hash import stable_prefix_hash
from tokenetics.core.request import ToolSpec
from tokenetics.core.benchmark import (
    CorpusSample,
    QualityCheckSample,
    RequiredElementsResult,
    SavingsRecord,
    aggregate_savings_by_category,
    check_required_elements,
    check_tool_calls,
    estimate_cost_usd,
    load_corpus,
    load_quality_checks,
    response_text_with_tool_inputs,
)
from tokenetics.core.cache_pricing import (
    READ_MULTIPLIER,
    WRITE_MULTIPLIER_1H,
    WRITE_MULTIPLIER_5M,
    base_input_price_for,
    output_price_for,
)
from tokenetics.core.request import from_api_kwargs
from tokenetics.core.tokenizer import count_tokens

_BENCHMARKS_DIR = Path(__file__).resolve().parent.parent / "benchmarks"
_DEFAULT_COST_CEILING_USD = 1.00

# `usage` subcommand only: the corpus's own max_tokens values (e.g. 400 for
# code, 300 for conversational) are deliberately tiny -- chosen for Phase 9
# to exercise adaptive_budget's WIDENING mechanism under a tight cap, not
# to represent what a realistic caller would set. Reusing them unmodified
# for a $-cost benchmark produced a real, misleading result (2026-09-04): a
# diagnostic run confirmed EVERY completion, baseline and pipeline alike,
# hit `stop_reason="max_tokens"` -- both sides were truncated, incomplete
# replies, and the pipeline was only "more expensive" because it wrote more
# BEFORE also getting cut off, not because it produced a wastefully long
# complete answer. That's not a fair test of real cost. This floor is
# applied identically to both the baseline and pipeline calls (never just
# one side), so both get a fair chance to finish naturally -- 2048 is
# chosen from real observed reply lengths in this project's own dev_demo.py
# runs (thorough code/explanation answers commonly ran 700-960 output
# tokens), with generous headroom, not a guess. With this floor in place,
# adaptive_budget's own widening will often no-op (the caller's cap already
# exceeds its heuristic) -- correct, and it isolates whether the OTHER
# mechanisms (dedup, near_dup, schema_minification, context_scheduler,
# delta_compression, caching) save money once truncation stops confounding
# the comparison.
_REALISTIC_MAX_TOKENS_FLOOR = 2048


def _with_realistic_max_tokens(kwargs: dict[str, Any]) -> dict[str, Any]:
    kwargs = dict(kwargs)
    kwargs["max_tokens"] = max(kwargs.get("max_tokens", 0), _REALISTIC_MAX_TOKENS_FLOOR)
    return kwargs

# Not one of task_classifier's 4 real categories -- a deliberately separate,
# explicitly-labeled bucket (deferred from Phase 9, added Phase 10) for a
# single realistic multi-turn session spanning several task types in one
# conversation. Reported on its own, never blended into the 4 categorized
# ranges or their "clears the floor" claim below: CLAUDE.md's honest-
# benchmarking rule requires savings tied to A stated workload assumption,
# and this category is intentionally several workloads at once.
_MIXED_WORKLOAD_CATEGORY = "mixed_workload"


def _by_category(samples: list[CorpusSample]) -> dict[str, list[CorpusSample]]:
    by_category: dict[str, list[CorpusSample]] = {}
    for sample in samples:
        by_category.setdefault(sample.category, []).append(sample)
    return by_category


def _print_nonzero_stage_deltas(tk: Tokenetics) -> None:
    """Traces which stage(s) actually produced a sample's token reduction --
    the corpus summary alone only shows before/after, not which of the 10
    pipeline stages did the work. Only prints stages with a real (nonzero)
    token delta or a logged note; a 0-delta stage that fired silently (e.g.
    task_classifier, which only annotates) is omitted as noise.
    """
    entries = getattr(tk.logger, "entries", None) or []
    for entry in entries:
        delta = (entry.tokens_before or 0) - (entry.tokens_after or 0)
        notes = {k: v for k, v in entry.extra.items() if k not in ("error", "timing_seconds")}
        if delta == 0 and not notes:
            continue
        sign = "saved" if delta > 0 else "cost" if delta < 0 else "0 tokens"
        detail = f"{sign} {abs(delta)}" if delta != 0 else sign
        notes_str = f" {notes}" if notes else ""
        print(f"    {entry.stage_name}: {detail}{notes_str}")


def _usage_dict(response: Any) -> dict[str, int]:
    """Same shape as dev_demo.py's own `_usage_dict` -- kept as a separate
    copy rather than a shared import since dev_demo.py is a standalone
    script, not a package module, but the formula must stay IDENTICAL to
    dev_demo.py's, since this is meant to be the more rigorous version of
    the exact same measurement, not a divergent one.
    """
    usage = response.usage
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", 0) or 0,
        "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0) or 0,
    }


def _thinking_tokens(response: Any) -> int:
    # Same formula as dev_demo.py's own `_thinking_tokens` (see that
    # module for the real field this reads -- `usage.output_tokens_details.
    # thinking_tokens`, confirmed 2026-09-06 against the installed SDK).
    # Returns 0 (not None) here since the session benchmark sums this
    # across many turns -- a per-turn None would need extra plumbing this
    # rigor-focused script doesn't need; 0 and "genuinely no thinking data"
    # are indistinguishable here by design, unlike dev_demo.py's own
    # None-preserving version.
    details = getattr(response.usage, "output_tokens_details", None)
    if details is None:
        return 0
    return int(details.thinking_tokens)


def _extract_cache_ttl(prepared: dict[str, Any]) -> str | None:
    system = prepared.get("system")
    if isinstance(system, list) and system and "cache_control" in system[-1]:
        ttl = system[-1]["cache_control"].get("ttl")
        return str(ttl) if ttl is not None else None
    tools = prepared.get("tools")
    if tools and "cache_control" in tools[-1]:
        ttl = tools[-1]["cache_control"].get("ttl")
        return str(ttl) if ttl is not None else None
    return None


def _full_cost_usd(usage: dict[str, int], model: str, ttl: str | None) -> float:
    """The REAL, full dollar cost of one completion's measured usage --
    input, output, and cache tokens each weighted by their own real,
    different per-token price from core.cache_pricing's live-verified
    table. This is the piece `dev_demo.py`'s own cost verdict never
    computed: it only ever priced the cache portion of a call (write vs.
    read), never combined that with the regular input/output cost into one
    total -- which is exactly what's needed to see whether input-side
    savings actually survive being weighed against output-token cost
    (output is priced ~5x higher than input for Sonnet 5), the real
    finding that prompted building this subcommand at all.
    """
    write_multiplier = WRITE_MULTIPLIER_1H if ttl == "1h" else WRITE_MULTIPLIER_5M
    input_price = base_input_price_for(model) / 1_000_000
    output_price = output_price_for(model) / 1_000_000
    return (
        usage["input_tokens"] * input_price
        + usage["cache_creation_input_tokens"] * input_price * write_multiplier
        + usage["cache_read_input_tokens"] * input_price * READ_MULTIPLIER
        + usage["output_tokens"] * output_price
    )


def _estimate_usage_cost(samples: list[CorpusSample], model: str) -> float:
    """Pre-flight estimate: 2 completions per sample (baseline + pipeline),
    input size from message length, output assumed at ~half of max_tokens
    -- same deliberately-conservative-ish style as
    core.benchmark.estimate_cost_usd, explicitly an ESTIMATE, never logged
    as measured.
    """
    input_price = base_input_price_for(model) / 1_000_000
    output_price = output_price_for(model) / 1_000_000
    total = 0.0
    for sample in samples:
        kwargs = _with_realistic_max_tokens(sample.kwargs)
        input_tokens = len(json.dumps(kwargs)) // 4
        output_tokens = kwargs["max_tokens"] * 0.5
        cost_per_completion = input_tokens * input_price + output_tokens * output_price
        total += cost_per_completion * 2
    return total


@dataclass
class _UsageResult:
    sample_id: str
    category: str
    baseline_cost_usd: float
    pipeline_cost_usd: float
    baseline_usage: dict[str, int]
    pipeline_usage: dict[str, int]
    repeat_cost_usd: float | None = None  # steady-state cache-read cost, if applicable

    @property
    def pct_cheaper(self) -> float:
        if self.baseline_cost_usd == 0:
            return 0.0
        return (1 - self.pipeline_cost_usd / self.baseline_cost_usd) * 100


def run_usage_benchmark(args: argparse.Namespace) -> None:
    samples = load_corpus(_BENCHMARKS_DIR / "corpus")
    if args.sample:
        samples = [s for s in samples if s.id == args.sample]
    elif args.category:
        samples = [s for s in samples if s.category == args.category]
    if not samples:
        print("No corpus samples found.")
        return
    if args.limit:
        by_cat = _by_category(samples)
        samples = [s for cat_samples in by_cat.values() for s in cat_samples[: args.limit]]

    model = args.model or samples[0].kwargs.get("model", "claude-sonnet-5")
    estimate = _estimate_usage_cost(samples, model)
    print(
        f"Estimated cost: ${estimate:.4f} for {len(samples)} sample(s) x 2 completions "
        f"(baseline + pipeline-enabled) -- ESTIMATED, not measured. A sample that writes "
        f"to cache adds one more (cheap) repeat completion."
    )

    if args.dry_run:
        print("--dry-run: no completions made.")
        return
    if estimate > args.cost_ceiling:
        print(f"REFUSED: estimated cost ${estimate:.4f} exceeds --cost-ceiling ${args.cost_ceiling:.2f}.")
        return
    if not args.confirm_spend:
        print("REFUSED: real billed calls require --confirm-spend.")
        return

    client = anthropic.Anthropic()
    results: list[_UsageResult] = []
    total_baseline = 0.0
    total_pipeline = 0.0

    for sample in samples:
        kwargs = _with_realistic_max_tokens(sample.kwargs)
        tk = Tokenetics(client=client, stage_config=sample.stage_config)
        prepared = tk.prepare(**kwargs)

        baseline_response = client.messages.create(**kwargs)
        pipeline_response = client.messages.create(**prepared)
        baseline_usage = _usage_dict(baseline_response)
        pipeline_usage = _usage_dict(pipeline_response)

        baseline_cost = _full_cost_usd(baseline_usage, kwargs["model"], None)
        pipeline_cost = _full_cost_usd(pipeline_usage, kwargs["model"], _extract_cache_ttl(prepared))
        total_baseline += baseline_cost
        total_pipeline += pipeline_cost

        result = _UsageResult(
            sample_id=sample.id,
            category=sample.category,
            baseline_cost_usd=baseline_cost,
            pipeline_cost_usd=pipeline_cost,
            baseline_usage=baseline_usage,
            pipeline_usage=pipeline_usage,
        )

        repeat_note = ""
        if pipeline_usage["cache_creation_input_tokens"] > 0:
            # Steady-state signal: what does the SECOND (repeat, still
            # inside the TTL) call cost once the cache write has already
            # happened? Not blended into the main totals -- reported
            # separately, since real traffic is mostly repeat calls, not
            # perpetual first-calls, and blending would misrepresent both.
            repeat_response = client.messages.create(**prepared)
            repeat_usage = _usage_dict(repeat_response)
            result.repeat_cost_usd = _full_cost_usd(repeat_usage, kwargs["model"], _extract_cache_ttl(prepared))
            repeat_note = f", repeat(steady-state)=${result.repeat_cost_usd:.6f}"

        results.append(result)
        print(
            f"{sample.id} ({sample.category}): baseline=${baseline_cost:.6f} "
            f"pipeline=${pipeline_cost:.6f} ({result.pct_cheaper:+.1f}%){repeat_note}"
        )

    print("\n=== Real $ cost by category (range, not a single flat percentage) ===")
    by_category: dict[str, list[_UsageResult]] = {}
    for r in results:
        by_category.setdefault(r.category, []).append(r)
    for category, cat_results in sorted(by_category.items()):
        pcts = [r.pct_cheaper for r in cat_results]
        cat_baseline = sum(r.baseline_cost_usd for r in cat_results)
        cat_pipeline = sum(r.pipeline_cost_usd for r in cat_results)
        print(
            f"  {category}: {min(pcts):.1f}% - {max(pcts):.1f}% cheaper "
            f"(${cat_baseline:.6f} -> ${cat_pipeline:.6f}, n={len(cat_results)})"
        )

    repeat_results = [r for r in results if r.repeat_cost_usd is not None]
    if repeat_results:
        print("\n=== Steady-state (cache-read) cost for samples that wrote to cache ===")
        for r in repeat_results:
            never_cache = r.baseline_cost_usd  # first-call-shaped baseline, as a rough per-call reference
            print(
                f"  {r.sample_id}: first pipeline call=${r.pipeline_cost_usd:.6f}, "
                f"repeat (cache read)=${r.repeat_cost_usd:.6f} "
                f"(vs. a never-cached call at roughly ${never_cache:.6f})"
            )

    overall_pct = (1 - total_pipeline / total_baseline) * 100 if total_baseline else 0.0
    print(
        f"\n=== TOTAL across {len(results)} samples (equal per-sample weighting -- "
        f"NOT a claim about real traffic proportions) ==="
    )
    print(f"baseline: ${total_baseline:.6f}")
    print(f"pipeline: ${total_pipeline:.6f}")
    print(f"net: {overall_pct:+.1f}% ({'cheaper' if overall_pct >= 0 else 'MORE EXPENSIVE'})")

    total_output_delta = sum(
        r.pipeline_usage["output_tokens"] - r.baseline_usage["output_tokens"] for r in results
    )
    total_input_delta = sum(
        r.baseline_usage["input_tokens"] - r.pipeline_usage["input_tokens"] for r in results
    )
    print(
        f"\nfor context: input tokens {'saved' if total_input_delta >= 0 else 'cost'} "
        f"{abs(total_input_delta)} total; output tokens {'saved' if total_output_delta <= 0 else 'cost'} "
        f"{abs(total_output_delta)} total (output is priced ~5x input for {model} -- "
        "this is why a token-count-only view can disagree with the $ verdict above)"
    )

    if args.save:
        _save_results(
            "usage",
            {
                "date": date.today().isoformat(),
                "sample_count": len(results),
                "total_baseline_usd": total_baseline,
                "total_pipeline_usd": total_pipeline,
                "overall_pct_cheaper": overall_pct,
                "by_sample": [
                    {
                        "id": r.sample_id,
                        "category": r.category,
                        "baseline_usd": r.baseline_cost_usd,
                        "pipeline_usd": r.pipeline_cost_usd,
                        "repeat_usd": r.repeat_cost_usd,
                    }
                    for r in results
                ],
            },
        )


_SESSION_PATH = _BENCHMARKS_DIR / "session" / "realistic_session.json"


def _estimate_session_cost(data: dict[str, Any], model: str) -> float:
    """Pre-flight estimate for `session`: 2 completions per turn (baseline +
    pipeline), input growing roughly linearly as history accumulates,
    output assumed at ~half of max_tokens per turn -- same deliberately
    conservative-ish style as the other estimators here, explicitly an
    ESTIMATE, never logged as measured.
    """
    input_price = base_input_price_for(model) / 1_000_000
    output_price = output_price_for(model) / 1_000_000
    system = data["system"] * data["system_repeat"]
    max_tokens = data["max_tokens"]
    running_chars = len(system)
    total = 0.0
    for turn in data["user_turns"]:
        running_chars += len(turn)
        input_tokens = running_chars // 4
        output_tokens = max_tokens * 0.5
        running_chars += int(output_tokens * 4)  # the reply becomes part of the next turn's history
        cost_per_completion = input_tokens * input_price + output_tokens * output_price
        total += cost_per_completion * 2  # baseline + pipeline
    return total


def run_session_benchmark(args: argparse.Namespace) -> None:
    """Simulates one real, growing multi-turn conversation -- the thing a
    one-shot-per-sample benchmark (`usage`) structurally can't measure:
    does Tokenetics save money over the LIFE of a session, where the same
    system prompt/tools repeat across many real calls (letting caching
    actually amortize past a single write) and genuine redundancy
    accumulates in history (letting dedup/near_dup/context_scheduler have
    real material to prune)? Two parallel real threads -- `baseline`
    (raw request/reply every turn, no Tokenetics, never cached) and
    `pipeline` (`prepare()`/`finalize()` every turn, with a real,
    incrementally-grown `cache_usage_history` fed forward turn to turn,
    matching how a real caller would track their own repeat pattern) --
    diverge in reply content turn by turn (they're separate real
    completions), which is expected and correct: this measures "what would
    my bill look like if I'd used Tokenetics from turn one," not a replay
    of identical content.
    """
    data = json.loads(_SESSION_PATH.read_text())
    model = args.model or data["model"]
    estimate = _estimate_session_cost(data, model)
    n_turns = len(data["user_turns"])
    print(
        f"Estimated cost: ${estimate:.4f} for {n_turns} turns x 2 threads "
        f"(baseline + pipeline) = {n_turns * 2} completions -- ESTIMATED, not measured."
    )

    if args.dry_run:
        print("--dry-run: no completions made.")
        return
    if estimate > args.cost_ceiling:
        print(f"REFUSED: estimated cost ${estimate:.4f} exceeds --cost-ceiling ${args.cost_ceiling:.2f}.")
        return
    if not args.confirm_spend:
        print("REFUSED: real billed calls require --confirm-spend.")
        return

    client = anthropic.Anthropic()
    system = data["system"] * data["system_repeat"]
    tools = data["tools"]
    max_tokens = data["max_tokens"]
    tool_specs = [ToolSpec(name=t["name"], description=t["description"], input_schema=t["input_schema"]) for t in tools]
    prefix_hash = stable_prefix_hash(system, tool_specs)

    baseline_messages: list[dict[str, Any]] = []
    pipeline_messages: list[dict[str, Any]] = []
    cache_usage_history: list[dict[str, Any]] = []
    total_baseline = 0.0
    total_pipeline = 0.0
    # Per-dimension real breakdown (added 2026-09-06, after the user asked
    # for input/output/thinking savings specifically, not just one blended
    # $ verdict per turn) -- same three dimensions the dashboard already
    # tracks for isolated single-turn samples, now measured across a real,
    # cache-amortizing, redundancy-accumulating SESSION for the first
    # time. This session predates neither AGGRESSIVE brevity's default nor
    # adaptive_budget's bounded-shape thinking-effort default (both landed
    # earlier today) -- both fire automatically here with no stage_config
    # changes needed, since they're real pipeline defaults now, not
    # something this script has to opt into.
    total_baseline_input = 0
    total_pipeline_input = 0
    total_baseline_output = 0
    total_pipeline_output = 0
    total_baseline_thinking = 0
    total_pipeline_thinking = 0

    for turn_idx, user_text in enumerate(data["user_turns"], start=1):
        baseline_messages.append({"role": "user", "content": user_text})
        pipeline_messages.append({"role": "user", "content": user_text})

        baseline_kwargs: dict[str, Any] = {
            "model": model, "max_tokens": max_tokens, "system": system, "tools": tools,
            "messages": baseline_messages,
        }
        baseline_response = client.messages.create(**baseline_kwargs)
        baseline_usage = _usage_dict(baseline_response)
        baseline_cost = _full_cost_usd(baseline_usage, model, None)
        total_baseline += baseline_cost
        total_baseline_input += baseline_usage["input_tokens"]
        total_baseline_output += baseline_usage["output_tokens"]
        baseline_thinking = _thinking_tokens(baseline_response)
        total_baseline_thinking += baseline_thinking
        baseline_reply = "".join(
            getattr(b, "text", "") for b in baseline_response.content if getattr(b, "type", None) == "text"
        )
        baseline_messages.append({"role": "assistant", "content": baseline_reply})

        tk = Tokenetics(
            client=client,
            stage_config={"cache_breakpoint_optimizer": {"cache_usage_history": list(cache_usage_history)}},
        )
        pipeline_kwargs: dict[str, Any] = {
            "model": model, "max_tokens": max_tokens, "system": system, "tools": tools,
            "messages": pipeline_messages,
        }
        prepared = tk.prepare(**pipeline_kwargs)
        logged_entries = getattr(tk.logger, "entries", None) or []
        cache_entry = next(e for e in logged_entries if e.stage_name == "cache_breakpoint_optimizer")
        pipeline_response = client.messages.create(**prepared)
        pipeline_usage = _usage_dict(pipeline_response)
        pipeline_cost = _full_cost_usd(pipeline_usage, model, _extract_cache_ttl(prepared))
        total_pipeline += pipeline_cost
        total_pipeline_input += pipeline_usage["input_tokens"]
        total_pipeline_output += pipeline_usage["output_tokens"]
        pipeline_thinking = _thinking_tokens(pipeline_response)
        total_pipeline_thinking += pipeline_thinking
        stored_reply = tk.finalize(pipeline_response)
        pipeline_messages.append({"role": "assistant", "content": stored_reply})

        cache_usage_history.append({"timestamp": time.time(), "content_hash": prefix_hash})

        # Tier/gap visibility (added 2026-09-05): without this, a re-write
        # a few turns after the first one is indistinguishable from "the
        # tier-selection fix isn't working" vs. "no prior evidence existed
        # yet to be conservative about" -- printing the actual decision
        # basis removes the guesswork.
        if cache_entry.extra.get("breakpoint_placed"):
            tier_info = (
                f"ttl={cache_entry.extra.get('cache_ttl')} "
                f"avg_gap={cache_entry.extra.get('avg_repeat_gap_seconds')}s "
                f"max_gap={cache_entry.extra.get('max_repeat_gap_seconds')}s"
            )
        else:
            tier_info = f"no breakpoint (reason={cache_entry.extra.get('reason')})"

        turn_pct = (1 - pipeline_cost / baseline_cost) * 100 if baseline_cost else 0.0
        turn_thinking = pipeline_thinking - baseline_thinking
        print(
            f"turn {turn_idx}: baseline=${baseline_cost:.6f} "
            f"(cache_read={baseline_usage['cache_read_input_tokens']}) "
            f"pipeline=${pipeline_cost:.6f} "
            f"(cache_write={pipeline_usage['cache_creation_input_tokens']}, "
            f"cache_read={pipeline_usage['cache_read_input_tokens']}) "
            f"({turn_pct:+.1f}%) [{tier_info}] "
            f"input: {baseline_usage['input_tokens']}->{pipeline_usage['input_tokens']}, "
            f"output: {baseline_usage['output_tokens']}->{pipeline_usage['output_tokens']}, "
            f"thinking delta: {turn_thinking:+d}"
        )

    overall_pct = (1 - total_pipeline / total_baseline) * 100 if total_baseline else 0.0
    input_pct = (
        (total_baseline_input - total_pipeline_input) / total_baseline_input * 100 if total_baseline_input else 0.0
    )
    output_pct = (
        (total_baseline_output - total_pipeline_output) / total_baseline_output * 100
        if total_baseline_output
        else 0.0
    )
    print(f"\n=== TOTAL over {n_turns} real turns (one growing session) ===")
    print(f"baseline: ${total_baseline:.6f}")
    print(f"pipeline: ${total_pipeline:.6f}")
    print(f"net: {overall_pct:+.1f}% ({'cheaper' if overall_pct >= 0 else 'MORE EXPENSIVE'})")
    print(
        f"input tokens:    {total_baseline_input} -> {total_pipeline_input} "
        f"(saved {total_baseline_input - total_pipeline_input}, {input_pct:+.1f}%)"
    )
    print(
        f"output tokens:   {total_baseline_output} -> {total_pipeline_output} "
        f"(saved {total_baseline_output - total_pipeline_output}, {output_pct:+.1f}%)"
    )
    print(
        f"thinking tokens: {total_baseline_thinking} -> {total_pipeline_thinking} "
        f"(real, measured, from usage.output_tokens_details.thinking_tokens -- a SUBSET of "
        f"output tokens above, not additive to them)"
    )

    if args.save:
        _save_results(
            "session",
            {
                "date": date.today().isoformat(),
                "turns": n_turns,
                "total_baseline_input_tokens": total_baseline_input,
                "total_pipeline_input_tokens": total_pipeline_input,
                "total_baseline_output_tokens": total_baseline_output,
                "total_pipeline_output_tokens": total_pipeline_output,
                "total_baseline_thinking_tokens": total_baseline_thinking,
                "total_pipeline_thinking_tokens": total_pipeline_thinking,
                "total_baseline_usd": total_baseline,
                "total_pipeline_usd": total_pipeline,
                "overall_pct_cheaper": overall_pct,
            },
        )


def run_corpus(args: argparse.Namespace) -> None:
    samples = load_corpus(_BENCHMARKS_DIR / "corpus")
    if args.sample:
        samples = [s for s in samples if s.id == args.sample]
        if not samples:
            print(f"No corpus sample with id {args.sample!r} found.")
            return
    elif args.category:
        samples = [s for s in samples if s.category == args.category]
    if not samples:
        print("No corpus samples found in benchmarks/corpus/.")
        return

    client = anthropic.Anthropic()
    records: list[SavingsRecord] = []
    for sample in samples:
        before = count_tokens(from_api_kwargs(**sample.kwargs), client)
        tk = Tokenetics(client=client, stage_config=sample.stage_config)
        prepared = tk.prepare(**sample.kwargs)
        after = count_tokens(from_api_kwargs(**prepared), client)
        record = SavingsRecord(
            category=sample.category, sample_id=sample.id, before_tokens=before, after_tokens=after
        )
        records.append(record)
        print(f"{sample.id} ({sample.category}): {before} -> {after} tokens ({record.pct_saved:.1f}%)")
        if not args.quiet:
            _print_nonzero_stage_deltas(tk)

    ranges = aggregate_savings_by_category(records)
    print("\n=== Savings by category (range, not a single flat percentage) ===")
    for r in ranges:
        note = ""
        if r.category == _MIXED_WORKLOAD_CATEGORY:
            note = (
                " -- NOT a task-type category; a single realistic multi-turn session "
                "spanning several task types in one conversation, reported separately "
                "and never blended into the 4 categorized ranges above/below"
            )
        print(f"  {r.category}: {r.min_pct:.1f}% - {r.max_pct:.1f}% (n={r.sample_count} samples){note}")

    if args.sample or args.category:
        print(f"\n{len(samples)} sample(s) shown -- filtered run, not the full corpus.")
    else:
        task_type_samples = [s for s in samples if s.category != _MIXED_WORKLOAD_CATEGORY]
        by_category = _by_category(task_type_samples)
        print(
            f"\n{len(task_type_samples)} task-type samples total, "
            f"{min(len(v) for v in by_category.values())}+ per category -- clears CLAUDE.md's "
            "15-20+/category floor. Still worth growing further before treating these as final "
            "published numbers; see ROADMAP.md's Phase 9 section."
        )
        mixed_count = len(samples) - len(task_type_samples)
        if mixed_count:
            print(
                f"(+{mixed_count} mixed-workload sample(s), reported separately above -- "
                "see ROADMAP.md's Phase 10 section)"
            )

    if args.save:
        _save_results(
            "corpus",
            {
                "date": date.today().isoformat(),
                "sample_count": len(samples),
                "ranges": [
                    {
                        "category": r.category,
                        "min_pct": r.min_pct,
                        "max_pct": r.max_pct,
                        "sample_count": r.sample_count,
                    }
                    for r in ranges
                ],
                "measured": True,
            },
        )


def _grade(response: object, sample: QualityCheckSample) -> RequiredElementsResult:
    """Combines both grading signals -- required TEXT/TOOL-INPUT elements
    and required TOOL CALLS -- into one pass/fail, since a sample may
    specify either or both. A correct tool_use-only response has no text to
    check, so the two checks are independent, not stacked as "must have
    both kinds"; a sample that only sets one of the two fields effectively
    skips the other (an unset list means nothing is required there).

    `required_elements` is checked against `response_text_with_tool_inputs`
    (text + any tool_use `input`, JSON-serialized), not text alone -- an
    extraction-shaped sample where `structured_output` forces `tool_choice`
    puts its real answer in the tool call's arguments, and a text-only
    check can never find it there. See that function's docstring for the
    real run that caught this.
    """
    content = list(getattr(response, "content", []))
    text_result = check_required_elements(
        response_text_with_tool_inputs(content), sample.required_elements
    )
    tool_result = check_tool_calls(content, sample.required_tool_calls or [])
    return RequiredElementsResult(
        passed=text_result.passed and tool_result.passed,
        missing=text_result.missing + tool_result.missing,
    )


def run_quality_check(args: argparse.Namespace) -> None:
    samples = load_quality_checks(_BENCHMARKS_DIR / "quality_checks")
    if args.stage:
        samples = [s for s in samples if s.stage == args.stage]
    if not samples:
        print("No matching quality-check samples found.")
        return

    model = args.model or samples[0].kwargs.get("model", "claude-sonnet-5")
    estimate = estimate_cost_usd(samples, model, completions_per_sample=2)
    print(
        f"Estimated cost: ${estimate:.4f} for {len(samples)} sample(s) x 2 completions "
        f"(with the stage enabled, and with it disabled) -- ESTIMATED, not measured."
    )

    if args.dry_run:
        print("--dry-run: no completions made.")
        return

    if estimate > args.cost_ceiling:
        print(
            f"REFUSED: estimated cost ${estimate:.4f} exceeds --cost-ceiling "
            f"${args.cost_ceiling:.2f}. Re-run with a higher --cost-ceiling if you "
            "actually intend to spend this much."
        )
        return

    if not args.confirm_spend:
        print("REFUSED: real billed calls require --confirm-spend.")
        return

    client = anthropic.Anthropic()
    results = []
    for sample in samples:
        tk_on = Tokenetics(client=client, stage_config=sample.stage_config)
        response_on = client.messages.create(**tk_on.prepare(**sample.kwargs))
        result_on = _grade(response_on, sample)

        tk_off = Tokenetics(client=client, stage_config=sample.stage_config)
        for stage in tk_off.stages:
            if stage.name == sample.stage:
                stage.enabled = False
        response_off = client.messages.create(**tk_off.prepare(**sample.kwargs))
        result_off = _grade(response_off, sample)

        print(f"\n=== {sample.id} ({sample.stage}, {sample.category}) ===")
        print(f"  with {sample.stage}:    passed={result_on.passed}  missing={result_on.missing}")
        print(f"  without {sample.stage}: passed={result_off.passed}  missing={result_off.missing}")
        # Print the actual response text on any failure -- otherwise a
        # missing-element diagnosis requires guessing whether the model
        # got it wrong or just phrased it differently than the rubric
        # expects. Caught needing this the hard way (2026-08-09): a failure
        # with no visibility into the real response is unfalsifiable.
        if not result_on.passed:
            print(f"  with {sample.stage} -- response: {response_text_with_tool_inputs(list(getattr(response_on, 'content', [])))!r}")
            print(f"  with {sample.stage} -- stop_reason: {getattr(response_on, 'stop_reason', None)!r}")
        if not result_off.passed:
            print(f"  without {sample.stage} -- response: {response_text_with_tool_inputs(list(getattr(response_off, 'content', [])))!r}")
            print(f"  without {sample.stage} -- stop_reason: {getattr(response_off, 'stop_reason', None)!r}")
        results.append(
            {
                "id": sample.id,
                "stage": sample.stage,
                "category": sample.category,
                "with_stage_passed": result_on.passed,
                "with_stage_missing": result_on.missing,
                "without_stage_passed": result_off.passed,
                "without_stage_missing": result_off.missing,
            }
        )

    passed_with = sum(1 for r in results if r["with_stage_passed"])
    passed_without = sum(1 for r in results if r["without_stage_passed"])
    print(
        f"\n=== TOTAL: {passed_with}/{len(results)} passed with the stage enabled, "
        f"{passed_without}/{len(results)} passed with it disabled ==="
    )

    if args.save:
        _save_results(
            "quality_check",
            {
                "date": date.today().isoformat(),
                "estimated_cost_usd": estimate,
                "results": results,
                "measured": True,
            },
        )


def _save_results(kind: str, data: dict[str, object]) -> None:
    results_dir = _BENCHMARKS_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = results_dir / f"{kind}_{timestamp}.json"
    path.write_text(json.dumps(data, indent=2))
    print(f"\nSaved results to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    corpus_parser = subparsers.add_parser("corpus", help="Free, prepare()-only corpus run.")
    corpus_parser.add_argument("--save", action="store_true", help="Save results to benchmarks/results/.")
    corpus_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print the per-sample summary line and category ranges, not the per-stage trace.",
    )
    corpus_parser.add_argument(
        "--sample", default=None, metavar="ID", help="Only run this one sample, e.g. tool_heavy_018."
    )
    corpus_parser.add_argument(
        "--category",
        default=None,
        choices=["code", "conversational", "extraction", "tool-heavy", "mixed_workload"],
        help="Only run samples in this category.",
    )
    corpus_parser.set_defaults(func=run_corpus)

    quality_parser = subparsers.add_parser("quality-check", help="Real-completion quality-check run.")
    quality_parser.add_argument(
        "--stage", default=None, help="Only run quality checks for this stage name, e.g. brevity_injector."
    )
    quality_parser.add_argument(
        "--dry-run", action="store_true", help="Print the cost estimate only, make no completions."
    )
    quality_parser.add_argument(
        "--confirm-spend", action="store_true", help="Required to actually make real, billed completions."
    )
    quality_parser.add_argument(
        "--cost-ceiling",
        type=float,
        default=_DEFAULT_COST_CEILING_USD,
        help=f"Refuse to run if the estimated cost exceeds this (USD, default ${_DEFAULT_COST_CEILING_USD:.2f}).",
    )
    quality_parser.add_argument("--model", default=None, help="Override the model used for cost estimation.")
    quality_parser.add_argument("--save", action="store_true", help="Save results to benchmarks/results/.")
    quality_parser.set_defaults(func=run_quality_check)

    usage_parser = subparsers.add_parser(
        "usage", help="Real, full-$-cost benchmark over the corpus (baseline vs. pipeline, input+output+cache)."
    )
    usage_parser.add_argument(
        "--sample", default=None, metavar="ID", help="Only run this one sample, e.g. tool_heavy_018."
    )
    usage_parser.add_argument(
        "--category",
        default=None,
        choices=["code", "conversational", "extraction", "tool-heavy", "mixed_workload"],
        help="Only run samples in this category.",
    )
    usage_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Only run the first N samples per category (cost control on a full 82-sample corpus).",
    )
    usage_parser.add_argument(
        "--dry-run", action="store_true", help="Print the cost estimate only, make no completions."
    )
    usage_parser.add_argument(
        "--confirm-spend", action="store_true", help="Required to actually make real, billed completions."
    )
    usage_parser.add_argument(
        "--cost-ceiling",
        type=float,
        default=_DEFAULT_COST_CEILING_USD,
        help=f"Refuse to run if the estimated cost exceeds this (USD, default ${_DEFAULT_COST_CEILING_USD:.2f}).",
    )
    usage_parser.add_argument("--model", default=None, help="Override the model used for cost estimation.")
    usage_parser.add_argument("--save", action="store_true", help="Save results to benchmarks/results/.")
    usage_parser.set_defaults(func=run_usage_benchmark)

    session_parser = subparsers.add_parser(
        "session",
        help="Real, growing multi-turn session (baseline vs. pipeline) -- the repeated-traffic "
        "cost picture a one-shot corpus can't show.",
    )
    session_parser.add_argument(
        "--dry-run", action="store_true", help="Print the cost estimate only, make no completions."
    )
    session_parser.add_argument(
        "--confirm-spend", action="store_true", help="Required to actually make real, billed completions."
    )
    session_parser.add_argument(
        "--cost-ceiling",
        type=float,
        default=_DEFAULT_COST_CEILING_USD,
        help=f"Refuse to run if the estimated cost exceeds this (USD, default ${_DEFAULT_COST_CEILING_USD:.2f}).",
    )
    session_parser.add_argument("--model", default=None, help="Override the model used for cost estimation.")
    session_parser.add_argument("--save", action="store_true", help="Save results to benchmarks/results/.")
    session_parser.set_defaults(func=run_session_benchmark)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
