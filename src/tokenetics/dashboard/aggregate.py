"""Phase 11 dashboard: pure, offline aggregation over a `FileCostLogger`
JSONL log. No network, no Anthropic client -- this module only ever reads
a file already written by `core.logger.FileCostLogger`, matching the
brief's "read-only consumer of the cost logger's structured log, fully
decoupled from the request path" requirement.

Two record shapes share one file, distinguished by `"type"`:
- `"type": "stage"` (from `log_stage()`) -- per-stage token counts, timing,
  enabled/disabled state, handled by `aggregate()`.
- `"type": "event"` (from `log_event()`, added in the second Phase 11
  pass) -- data that exists OUTSIDE the pipeline: real cache usage from
  the caller's own `messages.create()` call, TALE's round-trip cost,
  compress's achieved ratio, semantic-cache's hit/miss. Handled by
  `aggregate_events()`. This is what fills the gap flagged in the first
  pass ("cache hit rate / Tier 2 cost-benefit aren't shown yet") -- none
  of that data was ever logged anywhere before `log_event()` existed, so
  there was nothing for a dashboard to read; `dev_demo.py` now calls it
  wherever it already computes this data (for `--measure-usage`, `--tale`,
  `--compress-ratio`, `--semantic-cache`).

Records written before `log_event()` existed have no `"type"` key at all
-- treated as `"stage"` for backward compatibility with older log files.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# The only response-side stage that has ever existed in this project --
# used as an exact (not guessed) fallback for classifying "phase" on log
# entries written before the phase tag existed. See aggregate()'s
# docstring for why this matters.
_KNOWN_RESPONSE_SIDE_STAGES = {"post_hoc_trim"}


def load_entries(path: str | Path) -> list[dict[str, Any]]:
    """Reads a `FileCostLogger` JSONL file. Skips (rather than raises on) any
    line that isn't valid JSON -- a single corrupted line (e.g. a partial
    write from a crashed process) must not take down the whole dashboard,
    per its read-only/decoupled design requirement.
    """
    path = Path(path)
    if not path.exists():
        return []
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def _pct_saved(before: int, after: int) -> float:
    if before == 0:
        return 0.0
    return (before - after) / before * 100


@dataclass
class StageStats:
    stage_name: str
    times_seen: int = 0
    times_enabled: int = 0
    times_errored: int = 0
    total_tokens_before: int = 0
    total_tokens_after: int = 0
    total_timing_seconds: float = 0.0
    # One %-saved value per occurrence of this stage (added 2026-09-06, per
    # the user asking to see the real spread, not just one blended average --
    # matches CLAUDE.md's "report savings as ranges, never a single flat
    # percentage" rule, which benchmark_runner.py's corpus/usage reports
    # already followed but the dashboard never had until now).
    per_occurrence_pct_saved: list[float] = field(default_factory=list)

    @property
    def total_saved(self) -> int:
        return self.total_tokens_before - self.total_tokens_after

    @property
    def pct_saved(self) -> float:
        if self.total_tokens_before == 0:
            return 0.0
        return self.total_saved / self.total_tokens_before * 100

    @property
    def avg_timing_seconds(self) -> float:
        return self.total_timing_seconds / self.times_seen if self.times_seen else 0.0

    @property
    def min_pct_saved(self) -> float | None:
        return min(self.per_occurrence_pct_saved) if self.per_occurrence_pct_saved else None

    @property
    def median_pct_saved(self) -> float | None:
        return statistics.median(self.per_occurrence_pct_saved) if self.per_occurrence_pct_saved else None

    @property
    def max_pct_saved(self) -> float | None:
        return max(self.per_occurrence_pct_saved) if self.per_occurrence_pct_saved else None


@dataclass
class DashboardStats:
    total_runs: int = 0
    total_tokens_before: int = 0
    total_tokens_after: int = 0
    stages: dict[str, StageStats] = field(default_factory=dict)
    # One %-saved value per RUN (first request-stage's tokens_before -> last
    # request-stage's tokens_after for that run), same real-spread rationale
    # as StageStats.per_occurrence_pct_saved above, but for the overall
    # per-request number rather than one stage's.
    per_run_pct_saved: list[float] = field(default_factory=list)

    @property
    def total_saved(self) -> int:
        return self.total_tokens_before - self.total_tokens_after

    @property
    def pct_saved(self) -> float:
        if self.total_tokens_before == 0:
            return 0.0
        return self.total_saved / self.total_tokens_before * 100

    @property
    def min_pct_saved(self) -> float | None:
        return min(self.per_run_pct_saved) if self.per_run_pct_saved else None

    @property
    def median_pct_saved(self) -> float | None:
        return statistics.median(self.per_run_pct_saved) if self.per_run_pct_saved else None

    @property
    def max_pct_saved(self) -> float | None:
        return max(self.per_run_pct_saved) if self.per_run_pct_saved else None


def aggregate(entries: list[dict[str, Any]]) -> DashboardStats:
    """Groups entries by `run_id` (one `FileCostLogger` instance -- per this
    project's convention, one fresh `Tokenetics()` per request) to compute
    per-request overall REQUEST-side savings (first request-stage's
    tokens_before -> last request-stage's tokens_after), plus a flat
    per-stage rollup across every run in the file. Only `"type": "stage"`
    records (or records with no `"type"` at all, for older log files) are
    considered -- `"type": "event"` records go through `aggregate_events()`
    instead.

    The overall `total_tokens_before`/`total_tokens_after` are computed
    from request-side stages ONLY (`extra["phase"] == "request"`, tagged by
    the orchestrator's `_run_stage`) -- a real bug caught via a live
    dashboard hand-check (2026-08-12): mixing in `post_hoc_trim` (the one
    response-side stage, which measures the REPLY's token count via
    `count_text_tokens`, an entirely different axis from the request's
    token count) made "overall savings" compare request-size-before
    against reply-length-after for any run that called both `prepare()`
    AND `finalize()` -- since replies are often longer than an optimized
    request, this produced a misleading NEGATIVE "savings" number that had
    nothing to do with real token reduction. Per-stage rows in
    `stats.stages` are unaffected by this fix (each stage's own before/
    after is already self-consistent on its own axis, request or
    response); only the cross-stage "overall" summary was ever mixing them.
    Entries with no `phase` tag at all (log files written before this fix
    existed) fall back to `_KNOWN_RESPONSE_SIDE_STAGES` by stage NAME --
    caught as a second real bug the same day: defaulting an untagged
    entry to `"request"` unconditionally (the first version of this fix)
    still misclassified `post_hoc_trim`'s own untagged entries as
    request-side in any log file predating the `phase` tag, which is
    exactly the file the bug was originally found in, so the fix visibly
    didn't change anything for that data. `post_hoc_trim` is the only
    response-side stage that has ever existed in this project, so this
    fallback is exact, not a guess.
    """
    entries = [e for e in entries if e.get("type", "stage") == "stage"]
    stats = DashboardStats()
    by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        run_id = entry.get("run_id")
        if run_id is None:
            continue
        phase = (entry.get("extra") or {}).get("phase")
        if phase is None:
            phase = "response" if entry.get("stage_name") in _KNOWN_RESPONSE_SIDE_STAGES else "request"
        if phase != "request":
            continue
        by_run[run_id].append(entry)

    stats.total_runs = len({e.get("run_id") for e in entries if e.get("run_id") is not None})
    for run_entries in by_run.values():
        before_values = [e["tokens_before"] for e in run_entries if e.get("tokens_before") is not None]
        after_values = [e["tokens_after"] for e in run_entries if e.get("tokens_after") is not None]
        if before_values:
            stats.total_tokens_before += before_values[0]
        if after_values:
            stats.total_tokens_after += after_values[-1]
        if before_values and after_values:
            stats.per_run_pct_saved.append(_pct_saved(before_values[0], after_values[-1]))

    for entry in entries:
        stage_name = entry.get("stage_name")
        if not stage_name:
            continue
        stage = stats.stages.setdefault(stage_name, StageStats(stage_name=stage_name))
        stage.times_seen += 1
        if entry.get("enabled"):
            stage.times_enabled += 1
        extra = entry.get("extra") or {}
        if extra.get("error"):
            stage.times_errored += 1
        tokens_before = entry.get("tokens_before")
        tokens_after = entry.get("tokens_after")
        if tokens_before is not None:
            stage.total_tokens_before += tokens_before
        if tokens_after is not None:
            stage.total_tokens_after += tokens_after
        if tokens_before is not None and tokens_after is not None:
            stage.per_occurrence_pct_saved.append(_pct_saved(tokens_before, tokens_after))
        timing = extra.get("timing_seconds")
        if timing is not None:
            stage.total_timing_seconds += timing

    return stats


@dataclass
class CacheUsageStats:
    """From `log_event("cache_usage", ...)` -- real `usage.cache_read_input_tokens`/
    `cache_creation_input_tokens` from the caller's own `messages.create()`
    call, which Tier 0 never sees (it doesn't make the API call itself)."""

    sample_count: int = 0
    total_cache_read_tokens: int = 0
    total_cache_creation_tokens: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.total_cache_read_tokens + self.total_cache_creation_tokens
        if total == 0:
            return 0.0
        return self.total_cache_read_tokens / total * 100


@dataclass
class GenerationUsageStats:
    """From `log_event("generation_usage", ...)` -- real baseline-vs-
    optimized `output_tokens` (measured, from a genuine with/without-
    pipeline A/B of real completions -- see dev_demo.py's `_measure_usage`,
    the only place in this project that runs both sides of that comparison
    for real) plus each side's real visible-text token count.

    `output_tokens_saved`/`output_pct_saved` are real, measured savings on
    the output/generation side -- something nothing else in this project's
    logging previously captured (Tier 0's own cost logger only ever sees
    the request side, since `prepare()`/`finalize()` never make the actual
    completion call).

    `estimated_thinking_tokens_saved` is NOT measured, and its name says so
    on purpose: it's derived as (output_tokens - visible_text_tokens) on
    each side, then diffed. Kept only as a fallback for log data written
    before 2026-09-06 -- see `thinking_tokens_consumed` below for the real
    measured number now used for any current data.

    `total_baseline_thinking_tokens`/`total_optimized_thinking_tokens`
    (added 2026-09-06) ARE real, measured: Anthropic's `usage.
    output_tokens_details.thinking_tokens` is a genuine field ("Computed by
    re-tokenizing the raw reasoning text, so it may differ from the model's
    exact generation count by a small number of tokens" -- close to exact,
    not a derived guess), confirmed against the installed anthropic SDK
    (0.117.0) after the earlier "no separate line item exists" assumption
    turned out to be wrong. This is CLAUDE.md's stage 9c line item 1
    ("configured effort level vs. actual thinking tokens consumed at
    generation time") -- kept as its own field pair, not merged into
    `output_tokens_saved`, per that same rule ("must not be conflated"
    with visible-output/max_tokens savings, a different cost dimension).
    `thinking_sample_count` counts only samples where at least one side
    actually reported this field (most scenarios never engage thinking at
    all, so summing across every sample would understate real usage where
    it did fire).

    **Not automatically attributable to `adaptive_budget`'s effort knob**
    -- confirmed 2026-09-06 via a real dashboard run: several scenarios
    with NO `thinking` config anywhere (caller-side or `adaptive_budget`'s
    own opt-in never enabled) still showed real nonzero
    `baseline_thinking_tokens` on their completely untouched baseline call
    (e.g. a plain `code` scenario measured 45 real thinking tokens with no
    `thinking` parameter sent at all). The model can apparently report
    nonzero `output_tokens_details.thinking_tokens` independent of whether
    extended thinking was explicitly requested by anyone. `thinking_
    tokens_consumed` is an honest sum of the real field either way, but
    reading it as "the cost of the effort knob firing" overstates what it
    proves -- isolating the knob's real incremental cost needs a scenario
    where NOTHING else sets `thinking` and only `adaptive_budget`'s own
    opt-in does (see `dev_demo.py`'s `thinking_effort_code` scenario).

    `baseline_truncated_count` matters for reading `output_tokens_saved`
    honestly: dev_demo.py's own scenarios deliberately use tiny caller-set
    `max_tokens` values to demonstrate adaptive_budget's widening firing at
    all -- not to represent a realistic caller's own cap. A real run
    (2026-09-06) showed output_tokens "cost" concentrated exactly in the
    scenarios with the largest widen, and a real, nonzero
    `baseline_truncated_count` confirmed the baseline side was genuinely
    cut off in several of them -- meaning that "cost" often isn't the
    pipeline being wasteful, it's the baseline being an incomplete
    fragment vs. the optimized side's complete answer. Same confound this
    project already found and fixed once before, in `benchmark_runner.py
    usage`'s `_REALISTIC_MAX_TOKENS_FLOOR`.
    """

    sample_count: int = 0
    total_baseline_output_tokens: int = 0
    total_optimized_output_tokens: int = 0
    total_baseline_visible_text_tokens: int = 0
    total_optimized_visible_text_tokens: int = 0
    baseline_truncated_count: int = 0
    optimized_truncated_count: int = 0
    thinking_sample_count: int = 0
    total_baseline_thinking_tokens: int = 0
    total_optimized_thinking_tokens: int = 0
    # One %-saved value per real output-token sample, and one raw token
    # delta per real thinking-token sample (added 2026-09-06, same "report
    # ranges, not one blended average" rationale as StageStats.per_
    # occurrence_pct_saved). Thinking uses a raw delta, not a %, because
    # baseline_thinking_tokens is 0 on most samples (no thinking config
    # sent at all) -- a %-saved figure would be undefined (divide by zero)
    # far too often to be a meaningful summary.
    per_sample_output_pct_saved: list[float] = field(default_factory=list)
    per_sample_thinking_tokens_delta: list[int] = field(default_factory=list)

    @property
    def output_tokens_saved(self) -> int:
        return self.total_baseline_output_tokens - self.total_optimized_output_tokens

    @property
    def output_pct_saved(self) -> float:
        if self.total_baseline_output_tokens == 0:
            return 0.0
        return self.output_tokens_saved / self.total_baseline_output_tokens * 100

    @property
    def estimated_thinking_tokens_saved(self) -> int:
        baseline_overhead = self.total_baseline_output_tokens - self.total_baseline_visible_text_tokens
        optimized_overhead = self.total_optimized_output_tokens - self.total_optimized_visible_text_tokens
        return baseline_overhead - optimized_overhead

    @property
    def thinking_tokens_consumed(self) -> int:
        # The real, measured cost of adaptive_budget's thinking-effort knob
        # firing -- usually a spend (baseline has no thinking config at
        # all), not a saving, so reported as "consumed" rather than
        # "saved". A negative value means the baseline side actually used
        # MORE thinking tokens than the optimized side (e.g. the caller
        # already had their own thinking config on both calls).
        return self.total_optimized_thinking_tokens - self.total_baseline_thinking_tokens

    @property
    def min_output_pct_saved(self) -> float | None:
        return min(self.per_sample_output_pct_saved) if self.per_sample_output_pct_saved else None

    @property
    def median_output_pct_saved(self) -> float | None:
        return (
            statistics.median(self.per_sample_output_pct_saved) if self.per_sample_output_pct_saved else None
        )

    @property
    def max_output_pct_saved(self) -> float | None:
        return max(self.per_sample_output_pct_saved) if self.per_sample_output_pct_saved else None

    @property
    def min_thinking_tokens_delta(self) -> int | None:
        return min(self.per_sample_thinking_tokens_delta) if self.per_sample_thinking_tokens_delta else None

    @property
    def median_thinking_tokens_delta(self) -> float | None:
        return (
            statistics.median(self.per_sample_thinking_tokens_delta)
            if self.per_sample_thinking_tokens_delta
            else None
        )

    @property
    def max_thinking_tokens_delta(self) -> int | None:
        return max(self.per_sample_thinking_tokens_delta) if self.per_sample_thinking_tokens_delta else None


@dataclass
class TaleStats:
    """From `log_event("tale", ...)` -- TALE's real round-trip cost
    (estimation_input_tokens/estimation_output_tokens, both `measured`
    from the side-call's own `usage`) vs. the budget it estimated."""

    sample_count: int = 0
    total_estimation_input_tokens: int = 0
    total_estimation_output_tokens: int = 0
    total_budget_tokens: int = 0

    @property
    def avg_budget_tokens(self) -> float:
        return self.total_budget_tokens / self.sample_count if self.sample_count else 0.0

    @property
    def avg_estimation_cost_tokens(self) -> float:
        total = self.total_estimation_input_tokens + self.total_estimation_output_tokens
        return total / self.sample_count if self.sample_count else 0.0


@dataclass
class CompressStats:
    """From `log_event("compress", ...)` -- compress's real achieved ratio
    and how often the RECOMMENDED_MAX_RATIO clamp fired."""

    sample_count: int = 0
    clamped_count: int = 0
    total_ratio_achieved: float = 0.0

    @property
    def clamp_rate(self) -> float:
        return self.clamped_count / self.sample_count * 100 if self.sample_count else 0.0

    @property
    def avg_ratio_achieved(self) -> float:
        return self.total_ratio_achieved / self.sample_count if self.sample_count else 0.0


@dataclass
class SemanticCacheStats:
    """From `log_event("semantic_cache", ...)` -- real lookup hit/miss
    outcomes (the CALLER's pre-completion lookup, i.e. the realistic
    "would this have skipped the API call" signal -- not a lookup made
    only to demo the store()/lookup() round trip)."""

    sample_count: int = 0
    hit_count: int = 0

    @property
    def hit_rate(self) -> float:
        return self.hit_count / self.sample_count * 100 if self.sample_count else 0.0


@dataclass
class ThinkingReinjectionStats:
    """From `log_event("thinking_reinjection", ...)` -- CLAUDE.md's stage
    9c line item 2, real and measured: a later turn's real `usage.
    input_tokens` when a previous turn's thinking block is echoed back as
    history, compared between `display="omitted"` (adaptive_budget's real
    default) and `display="summarized"` (the counterfactual that WOULD
    get re-billed). A genuinely different cost dimension from
    `GenerationUsageStats.thinking_tokens_consumed` (that one measures
    THIS turn's generation-time thinking spend; this one measures a LATER
    turn's re-injection cost) -- kept as its own dataclass/event rather
    than folded in, per CLAUDE.md's explicit "must not be conflated" rule
    for this exact pair of stage 9c line items. See dev_demo.py's
    `_measure_thinking_reinjection` for how the real 2-turn/2-thread A/B
    that produces this data is built.
    """

    sample_count: int = 0
    total_omitted_turn2_input_tokens: int = 0
    total_summarized_turn2_input_tokens: int = 0

    @property
    def tokens_saved(self) -> int:
        return self.total_summarized_turn2_input_tokens - self.total_omitted_turn2_input_tokens

    @property
    def pct_saved(self) -> float:
        if self.total_summarized_turn2_input_tokens == 0:
            return 0.0
        return self.tokens_saved / self.total_summarized_turn2_input_tokens * 100


@dataclass
class Tier2Stats:
    cache_usage: CacheUsageStats = field(default_factory=CacheUsageStats)
    generation_usage: GenerationUsageStats = field(default_factory=GenerationUsageStats)
    tale: TaleStats = field(default_factory=TaleStats)
    compress: CompressStats = field(default_factory=CompressStats)
    semantic_cache: SemanticCacheStats = field(default_factory=SemanticCacheStats)
    thinking_reinjection: ThinkingReinjectionStats = field(default_factory=ThinkingReinjectionStats)

    @property
    def has_any_data(self) -> bool:
        return bool(
            self.cache_usage.sample_count
            or self.generation_usage.sample_count
            or self.tale.sample_count
            or self.compress.sample_count
            or self.semantic_cache.sample_count
            or self.thinking_reinjection.sample_count
        )


def aggregate_events(entries: list[dict[str, Any]]) -> Tier2Stats:
    """Aggregates `"type": "event"` records (from `log_event()`) into
    per-mechanism Tier 2/cache stats. Unknown `event_type` values are
    silently ignored rather than raising -- a forward-compatible read, in
    the same "must stay decoupled, never crash the dashboard" spirit as
    `load_entries()` skipping malformed lines.
    """
    stats = Tier2Stats()
    for entry in entries:
        if entry.get("type") != "event":
            continue
        event_type = entry.get("event_type")
        fields = entry.get("fields") or {}

        if event_type == "cache_usage":
            stats.cache_usage.sample_count += 1
            stats.cache_usage.total_cache_read_tokens += fields.get("cache_read_input_tokens") or 0
            stats.cache_usage.total_cache_creation_tokens += fields.get("cache_creation_input_tokens") or 0
        elif event_type == "generation_usage":
            stats.generation_usage.sample_count += 1
            baseline_output = fields.get("baseline_output_tokens") or 0
            optimized_output = fields.get("optimized_output_tokens") or 0
            stats.generation_usage.total_baseline_output_tokens += baseline_output
            stats.generation_usage.total_optimized_output_tokens += optimized_output
            if baseline_output:
                stats.generation_usage.per_sample_output_pct_saved.append(
                    (baseline_output - optimized_output) / baseline_output * 100
                )
            stats.generation_usage.total_baseline_visible_text_tokens += (
                fields.get("baseline_visible_text_tokens") or 0
            )
            stats.generation_usage.total_optimized_visible_text_tokens += (
                fields.get("optimized_visible_text_tokens") or 0
            )
            if fields.get("baseline_truncated"):
                stats.generation_usage.baseline_truncated_count += 1
            if fields.get("optimized_truncated"):
                stats.generation_usage.optimized_truncated_count += 1
            baseline_thinking = fields.get("baseline_thinking_tokens")
            optimized_thinking = fields.get("optimized_thinking_tokens")
            if baseline_thinking is not None or optimized_thinking is not None:
                stats.generation_usage.thinking_sample_count += 1
                stats.generation_usage.total_baseline_thinking_tokens += baseline_thinking or 0
                stats.generation_usage.total_optimized_thinking_tokens += optimized_thinking or 0
                stats.generation_usage.per_sample_thinking_tokens_delta.append(
                    (optimized_thinking or 0) - (baseline_thinking or 0)
                )
        elif event_type == "tale":
            stats.tale.sample_count += 1
            stats.tale.total_estimation_input_tokens += fields.get("estimation_input_tokens") or 0
            stats.tale.total_estimation_output_tokens += fields.get("estimation_output_tokens") or 0
            stats.tale.total_budget_tokens += fields.get("budget_tokens") or 0
        elif event_type == "compress":
            stats.compress.sample_count += 1
            if fields.get("clamped"):
                stats.compress.clamped_count += 1
            stats.compress.total_ratio_achieved += fields.get("ratio_achieved") or 0.0
        elif event_type == "semantic_cache":
            stats.semantic_cache.sample_count += 1
            if fields.get("hit"):
                stats.semantic_cache.hit_count += 1
        elif event_type == "thinking_reinjection":
            stats.thinking_reinjection.sample_count += 1
            stats.thinking_reinjection.total_omitted_turn2_input_tokens += (
                fields.get("omitted_turn2_input_tokens") or 0
            )
            stats.thinking_reinjection.total_summarized_turn2_input_tokens += (
                fields.get("summarized_turn2_input_tokens") or 0
            )

    return stats
