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


@dataclass
class StageStats:
    stage_name: str
    times_seen: int = 0
    times_enabled: int = 0
    times_errored: int = 0
    total_tokens_before: int = 0
    total_tokens_after: int = 0
    total_timing_seconds: float = 0.0

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


@dataclass
class DashboardStats:
    total_runs: int = 0
    total_tokens_before: int = 0
    total_tokens_after: int = 0
    stages: dict[str, StageStats] = field(default_factory=dict)

    @property
    def total_saved(self) -> int:
        return self.total_tokens_before - self.total_tokens_after

    @property
    def pct_saved(self) -> float:
        if self.total_tokens_before == 0:
            return 0.0
        return self.total_saved / self.total_tokens_before * 100


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
class Tier2Stats:
    cache_usage: CacheUsageStats = field(default_factory=CacheUsageStats)
    tale: TaleStats = field(default_factory=TaleStats)
    compress: CompressStats = field(default_factory=CompressStats)
    semantic_cache: SemanticCacheStats = field(default_factory=SemanticCacheStats)

    @property
    def has_any_data(self) -> bool:
        return bool(
            self.cache_usage.sample_count
            or self.tale.sample_count
            or self.compress.sample_count
            or self.semantic_cache.sample_count
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

    return stats
