"""Phase 11 dashboard tests. Entirely offline -- no Anthropic client, no
network calls other than a real loopback HTTP request to the dashboard's
own local server (127.0.0.1), which is what the server tests exercise
directly rather than mocking.
"""

from __future__ import annotations

import json
import threading
import urllib.request

from tokenetics.dashboard.aggregate import Tier2Stats, aggregate, aggregate_events, load_entries
from tokenetics.dashboard.render import render_html
from tokenetics.dashboard.server import run_server
from tokenetics.core.logger import FileCostLogger


def _write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


# --- load_entries -----------------------------------------------------


def test_load_entries_returns_empty_list_for_a_missing_file(tmp_path):
    assert load_entries(tmp_path / "does_not_exist.jsonl") == []


def test_load_entries_parses_valid_lines(tmp_path):
    path = tmp_path / "costs.jsonl"
    _write_jsonl(path, [{"run_id": "r1", "stage_name": "dedup"}])
    entries = load_entries(path)
    assert len(entries) == 1
    assert entries[0]["stage_name"] == "dedup"


def test_load_entries_skips_malformed_lines_rather_than_raising(tmp_path):
    path = tmp_path / "costs.jsonl"
    path.write_text('{"run_id": "r1", "stage_name": "dedup"}\nnot valid json\n{"run_id": "r1", "stage_name": "near_dup"}\n')
    entries = load_entries(path)
    assert len(entries) == 2
    assert [e["stage_name"] for e in entries] == ["dedup", "near_dup"]


def test_load_entries_skips_blank_lines(tmp_path):
    path = tmp_path / "costs.jsonl"
    path.write_text('{"run_id": "r1", "stage_name": "dedup"}\n\n\n')
    assert len(load_entries(path)) == 1


# --- aggregate ----------------------------------------------------------


def test_aggregate_computes_per_run_overall_savings():
    entries = [
        {"run_id": "r1", "stage_name": "dedup", "enabled": True, "tokens_before": 100, "tokens_after": 90, "extra": {}},
        {"run_id": "r1", "stage_name": "near_dup", "enabled": True, "tokens_before": 90, "tokens_after": 70, "extra": {}},
        {"run_id": "r2", "stage_name": "dedup", "enabled": True, "tokens_before": 200, "tokens_after": 200, "extra": {}},
    ]
    stats = aggregate(entries)
    assert stats.total_runs == 2
    # r1: first before (100) -> last after (70); r2: 200 -> 200
    assert stats.total_tokens_before == 300
    assert stats.total_tokens_after == 270
    assert stats.total_saved == 30


def test_aggregate_overall_savings_ignores_response_side_stages():
    # Regression test for a real bug caught via a live dashboard hand-check
    # (2026-08-12): post_hoc_trim (response-side, measures REPLY tokens via
    # count_text_tokens) was getting mixed into the same "first before ->
    # last after" calculation as request-side stages (which measure the
    # REQUEST). Since replies are often longer than an optimized request,
    # this produced a misleading negative "overall savings" number that
    # had nothing to do with real token reduction. A run with a small,
    # well-optimized request (100 -> 80) and a long, untrimmed reply
    # (500 -> 500) must report the REQUEST-side numbers only.
    entries = [
        {"run_id": "r1", "stage_name": "dedup", "enabled": True, "tokens_before": 100, "tokens_after": 80, "extra": {"phase": "request"}},
        {"run_id": "r1", "stage_name": "post_hoc_trim", "enabled": True, "tokens_before": 500, "tokens_after": 500, "extra": {"phase": "response"}},
    ]
    stats = aggregate(entries)
    assert stats.total_tokens_before == 100
    assert stats.total_tokens_after == 80
    assert stats.total_saved == 20
    assert stats.total_runs == 1  # still counts the run, even though it has a response-side stage too


def test_aggregate_excludes_post_hoc_trim_even_when_untagged():
    # Regression test for a SECOND real bug, found immediately after the
    # first fix, via the same live user hand-check: the user's actual log
    # file predated the phase-tagging fix entirely, so `post_hoc_trim`'s
    # own entries had no "phase" key at all -- the first version of this
    # fix defaulted ANY untagged entry to "request", which still
    # misclassified post_hoc_trim and left the bug fully unfixed for that
    # exact file. The fallback must identify post_hoc_trim by STAGE NAME
    # when no phase tag is present, not blanket-default to "request".
    entries = [
        {"run_id": "r1", "stage_name": "dedup", "enabled": True, "tokens_before": 100, "tokens_after": 80, "extra": {}},
        {"run_id": "r1", "stage_name": "post_hoc_trim", "enabled": True, "tokens_before": 500, "tokens_after": 500, "extra": {}},
    ]
    stats = aggregate(entries)
    assert stats.total_tokens_before == 100
    assert stats.total_tokens_after == 80
    assert stats.total_saved == 20


def test_aggregate_overall_savings_defaults_missing_phase_to_request():
    # Backward compatibility with log files written before the phase tag
    # existed -- no "phase" key at all should behave as "request", matching
    # prior behavior (every entry logged before post_hoc_trim's phase tag
    # existed WAS a request-side entry in practice at the time).
    entries = [
        {"run_id": "r1", "stage_name": "dedup", "enabled": True, "tokens_before": 100, "tokens_after": 80, "extra": {}},
    ]
    stats = aggregate(entries)
    assert stats.total_tokens_before == 100
    assert stats.total_tokens_after == 80


def test_aggregate_rolls_up_per_stage_across_runs():
    entries = [
        {"run_id": "r1", "stage_name": "dedup", "enabled": True, "tokens_before": 100, "tokens_after": 90, "extra": {"timing_seconds": 0.01}},
        {"run_id": "r2", "stage_name": "dedup", "enabled": False, "tokens_before": 50, "tokens_after": 50, "extra": {"timing_seconds": 0.02}},
    ]
    stats = aggregate(entries)
    dedup = stats.stages["dedup"]
    assert dedup.times_seen == 2
    assert dedup.times_enabled == 1
    assert dedup.total_tokens_before == 150
    assert dedup.total_tokens_after == 140
    assert dedup.total_saved == 10
    assert abs(dedup.avg_timing_seconds - 0.015) < 1e-9


def test_aggregate_tracks_min_median_max_pct_saved_per_stage():
    # Per CLAUDE.md's "report savings as ranges, never a single flat
    # percentage" rule -- a blended total can hide real per-occurrence
    # spread. Added 2026-09-06 per the user asking to see it on the
    # dashboard, not just in benchmark_runner.py's corpus/usage reports.
    entries = [
        {"run_id": "r1", "stage_name": "dedup", "enabled": True, "tokens_before": 100, "tokens_after": 90, "extra": {}},  # 10%
        {"run_id": "r2", "stage_name": "dedup", "enabled": True, "tokens_before": 100, "tokens_after": 50, "extra": {}},  # 50%
        {"run_id": "r3", "stage_name": "dedup", "enabled": True, "tokens_before": 100, "tokens_after": 100, "extra": {}},  # 0%
    ]
    dedup = aggregate(entries).stages["dedup"]
    assert dedup.min_pct_saved == 0.0
    assert dedup.median_pct_saved == 10.0
    assert dedup.max_pct_saved == 50.0


def test_stage_stats_pct_saved_range_is_none_with_no_samples():
    from tokenetics.dashboard.aggregate import StageStats

    stage = StageStats(stage_name="dedup")
    assert stage.min_pct_saved is None
    assert stage.median_pct_saved is None
    assert stage.max_pct_saved is None


def test_aggregate_tracks_min_median_max_pct_saved_per_run():
    entries = [
        {"run_id": "r1", "stage_name": "dedup", "enabled": True, "tokens_before": 100, "tokens_after": 80, "extra": {"phase": "request"}},
        {"run_id": "r2", "stage_name": "dedup", "enabled": True, "tokens_before": 100, "tokens_after": 60, "extra": {"phase": "request"}},
    ]
    stats = aggregate(entries)
    assert stats.min_pct_saved == 20.0
    assert stats.median_pct_saved == 30.0
    assert stats.max_pct_saved == 40.0


def test_aggregate_counts_errors():
    entries = [
        {"run_id": "r1", "stage_name": "dedup", "enabled": True, "tokens_before": 100, "tokens_after": 100, "extra": {"error": True}},
    ]
    stats = aggregate(entries)
    assert stats.stages["dedup"].times_errored == 1


def test_aggregate_of_empty_entries_is_a_zero_stats_object():
    stats = aggregate([])
    assert stats.total_runs == 0
    assert stats.total_tokens_before == 0
    assert stats.pct_saved == 0.0
    assert stats.stages == {}


def test_aggregate_ignores_entries_with_no_run_id():
    stats = aggregate([{"stage_name": "dedup", "enabled": True}])
    assert stats.total_runs == 0


def test_aggregate_ignores_event_records():
    entries = [
        {"run_id": "r1", "type": "stage", "stage_name": "dedup", "enabled": True, "tokens_before": 100, "tokens_after": 90, "extra": {}},
        {"run_id": "r1", "type": "event", "event_type": "tale", "fields": {"budget_tokens": 150}},
    ]
    stats = aggregate(entries)
    assert stats.total_runs == 1
    assert stats.total_tokens_before == 100
    assert "tale" not in stats.stages


# --- aggregate_events -----------------------------------------------------


def test_aggregate_events_computes_cache_hit_rate():
    entries = [
        {"type": "event", "event_type": "cache_usage", "fields": {"cache_read_input_tokens": 1724, "cache_creation_input_tokens": 0}},
        {"type": "event", "event_type": "cache_usage", "fields": {"cache_read_input_tokens": 0, "cache_creation_input_tokens": 500}},
    ]
    tier2 = aggregate_events(entries)
    assert tier2.cache_usage.sample_count == 2
    assert tier2.cache_usage.total_cache_read_tokens == 1724
    assert tier2.cache_usage.total_cache_creation_tokens == 500
    assert abs(tier2.cache_usage.hit_rate - (1724 / 2224 * 100)) < 1e-9


def test_aggregate_events_computes_output_tokens_saved():
    entries = [
        {
            "type": "event",
            "event_type": "generation_usage",
            "fields": {
                "baseline_output_tokens": 900,
                "optimized_output_tokens": 850,
                "baseline_visible_text_tokens": 870,
                "optimized_visible_text_tokens": 848,
            },
        },
        {
            "type": "event",
            "event_type": "generation_usage",
            "fields": {
                "baseline_output_tokens": 500,
                "optimized_output_tokens": 460,
                "baseline_visible_text_tokens": 480,
                "optimized_visible_text_tokens": 458,
            },
        },
    ]
    tier2 = aggregate_events(entries)
    gen = tier2.generation_usage
    assert gen.sample_count == 2
    assert gen.output_tokens_saved == 90  # (900+500) - (850+460)
    assert abs(gen.output_pct_saved - (90 / 1400 * 100)) < 1e-9
    # thinking overhead: baseline (1400 - 1350 = 50) minus optimized (1310 - 1306 = 4)
    assert gen.estimated_thinking_tokens_saved == 46


def test_aggregate_events_computes_real_thinking_tokens_consumed():
    entries = [
        {
            "type": "event",
            "event_type": "generation_usage",
            "fields": {
                "baseline_output_tokens": 900,
                "optimized_output_tokens": 1200,
                "baseline_visible_text_tokens": 870,
                "optimized_visible_text_tokens": 848,
                "baseline_thinking_tokens": None,
                "optimized_thinking_tokens": 300,
            },
        },
        {
            "type": "event",
            "event_type": "generation_usage",
            # A scenario that never engaged thinking on either side --
            # must not be counted in thinking_sample_count nor pollute
            # the totals with a phantom 0.
            "fields": {
                "baseline_output_tokens": 500,
                "optimized_output_tokens": 460,
                "baseline_visible_text_tokens": 480,
                "optimized_visible_text_tokens": 458,
            },
        },
    ]
    tier2 = aggregate_events(entries)
    gen = tier2.generation_usage
    assert gen.thinking_sample_count == 1
    assert gen.total_baseline_thinking_tokens == 0
    assert gen.total_optimized_thinking_tokens == 300
    assert gen.thinking_tokens_consumed == 300


def test_aggregate_events_tracks_per_sample_output_pct_saved_range():
    entries = [
        {
            "type": "event",
            "event_type": "generation_usage",
            "fields": {"baseline_output_tokens": 900, "optimized_output_tokens": 850},
        },
        {
            "type": "event",
            "event_type": "generation_usage",
            "fields": {"baseline_output_tokens": 500, "optimized_output_tokens": 100},
        },
        {
            "type": "event",
            "event_type": "generation_usage",
            "fields": {"baseline_output_tokens": 200, "optimized_output_tokens": 200},
        },
    ]
    tier2 = aggregate_events(entries)
    gen = tier2.generation_usage
    # sample 1: (900-850)/900*100 = 5.555...%, sample 2: 80.0%, sample 3: 0.0%
    assert abs(gen.min_output_pct_saved - 0.0) < 1e-9
    assert abs(gen.max_output_pct_saved - 80.0) < 1e-9
    assert abs(gen.median_output_pct_saved - (900 - 850) / 900 * 100) < 1e-9


def test_aggregate_events_tracks_per_sample_thinking_tokens_delta_range():
    entries = [
        {
            "type": "event",
            "event_type": "generation_usage",
            "fields": {
                "baseline_output_tokens": 100,
                "optimized_output_tokens": 100,
                "baseline_thinking_tokens": 0,
                "optimized_thinking_tokens": 50,
            },
        },
        {
            "type": "event",
            "event_type": "generation_usage",
            "fields": {
                "baseline_output_tokens": 100,
                "optimized_output_tokens": 100,
                "baseline_thinking_tokens": 40,
                "optimized_thinking_tokens": 0,
            },
        },
        {
            "type": "event",
            "event_type": "generation_usage",
            # No thinking data on this sample -- must not add a phantom 0
            # to the delta list.
            "fields": {"baseline_output_tokens": 100, "optimized_output_tokens": 100},
        },
    ]
    tier2 = aggregate_events(entries)
    gen = tier2.generation_usage
    assert gen.per_sample_thinking_tokens_delta == [50, -40]
    assert gen.min_thinking_tokens_delta == -40
    assert gen.max_thinking_tokens_delta == 50
    assert gen.median_thinking_tokens_delta == 5.0


def test_thinking_tokens_consumed_can_be_negative():
    # Baseline used MORE thinking than optimized (e.g. caller already had
    # their own thinking config on both calls) -- a real, measured saving,
    # not a cost.
    entries = [
        {
            "type": "event",
            "event_type": "generation_usage",
            "fields": {
                "baseline_output_tokens": 900,
                "optimized_output_tokens": 850,
                "baseline_visible_text_tokens": 700,
                "optimized_visible_text_tokens": 700,
                "baseline_thinking_tokens": 200,
                "optimized_thinking_tokens": 150,
            },
        },
    ]
    tier2 = aggregate_events(entries)
    assert tier2.generation_usage.thinking_tokens_consumed == -50


def test_aggregate_events_tracks_truncation_counts():
    # Regression test: a real run (2026-09-06) showed output_tokens "cost"
    # concentrated in scenarios where the baseline call's tiny caller-set
    # max_tokens genuinely truncated it -- a negative output_tokens_saved
    # in that case means "the baseline is an incomplete fragment," not
    # real waste. Tracked so the dashboard can show this caveat rather
    # than let the number be misread as pure inefficiency.
    entries = [
        {
            "type": "event",
            "event_type": "generation_usage",
            "fields": {
                "baseline_output_tokens": 60,
                "optimized_output_tokens": 960,
                "baseline_visible_text_tokens": 60,
                "optimized_visible_text_tokens": 900,
                "baseline_truncated": True,
                "optimized_truncated": False,
            },
        },
        {
            "type": "event",
            "event_type": "generation_usage",
            "fields": {
                "baseline_output_tokens": 195,
                "optimized_output_tokens": 158,
                "baseline_visible_text_tokens": 190,
                "optimized_visible_text_tokens": 155,
                "baseline_truncated": False,
                "optimized_truncated": False,
            },
        },
    ]
    tier2 = aggregate_events(entries)
    assert tier2.generation_usage.baseline_truncated_count == 1
    assert tier2.generation_usage.optimized_truncated_count == 0


def test_aggregate_events_computes_tale_averages():
    entries = [
        {"type": "event", "event_type": "tale", "fields": {"budget_tokens": 100, "estimation_input_tokens": 50, "estimation_output_tokens": 5}},
        {"type": "event", "event_type": "tale", "fields": {"budget_tokens": 200, "estimation_input_tokens": 50, "estimation_output_tokens": 5}},
    ]
    tier2 = aggregate_events(entries)
    assert tier2.tale.sample_count == 2
    assert tier2.tale.avg_budget_tokens == 150
    assert tier2.tale.avg_estimation_cost_tokens == 55


def test_aggregate_events_computes_compress_clamp_rate_and_avg_ratio():
    entries = [
        {"type": "event", "event_type": "compress", "fields": {"ratio_achieved": 0.4, "clamped": True}},
        {"type": "event", "event_type": "compress", "fields": {"ratio_achieved": 0.3, "clamped": False}},
    ]
    tier2 = aggregate_events(entries)
    assert tier2.compress.sample_count == 2
    assert tier2.compress.clamped_count == 1
    assert tier2.compress.clamp_rate == 50.0
    assert abs(tier2.compress.avg_ratio_achieved - 0.35) < 1e-9


def test_aggregate_events_computes_semantic_cache_hit_rate():
    entries = [
        {"type": "event", "event_type": "semantic_cache", "fields": {"hit": True, "similarity": 0.99}},
        {"type": "event", "event_type": "semantic_cache", "fields": {"hit": False, "similarity": 0.5}},
        {"type": "event", "event_type": "semantic_cache", "fields": {"hit": False, "similarity": 0.3}},
    ]
    tier2 = aggregate_events(entries)
    assert tier2.semantic_cache.sample_count == 3
    assert tier2.semantic_cache.hit_count == 1
    assert abs(tier2.semantic_cache.hit_rate - (1 / 3 * 100)) < 1e-9


def test_aggregate_events_computes_thinking_reinjection_savings():
    entries = [
        {
            "type": "event",
            "event_type": "thinking_reinjection",
            "fields": {"omitted_turn2_input_tokens": 120, "summarized_turn2_input_tokens": 450},
        },
    ]
    tier2 = aggregate_events(entries)
    reinjection = tier2.thinking_reinjection
    assert reinjection.sample_count == 1
    assert reinjection.tokens_saved == 330
    assert abs(reinjection.pct_saved - (330 / 450 * 100)) < 1e-9


def test_aggregate_events_ignores_unknown_event_types_rather_than_raising():
    entries = [{"type": "event", "event_type": "something_from_a_future_version", "fields": {"x": 1}}]
    tier2 = aggregate_events(entries)
    assert tier2.has_any_data is False


def test_aggregate_events_ignores_stage_records():
    entries = [{"type": "stage", "run_id": "r1", "stage_name": "dedup", "enabled": True}]
    tier2 = aggregate_events(entries)
    assert tier2.has_any_data is False


def test_tier2_stats_has_any_data_reflects_any_populated_mechanism():
    assert Tier2Stats().has_any_data is False
    tier2 = aggregate_events([{"type": "event", "event_type": "tale", "fields": {"budget_tokens": 100}}])
    assert tier2.has_any_data is True


# --- render_html ---------------------------------------------------------


def test_render_html_shows_empty_state_with_no_runs(tmp_path):
    html_out = render_html(aggregate([]), str(tmp_path / "costs.jsonl"))
    assert "No runs logged yet" in html_out
    assert "dev_demo.py" in html_out


def test_render_html_shows_stage_table_when_data_exists():
    entries = [
        {"run_id": "r1", "stage_name": "dedup", "enabled": True, "tokens_before": 100, "tokens_after": 80, "extra": {}},
    ]
    html_out = render_html(aggregate(entries), "costs.jsonl")
    assert "dedup" in html_out
    assert "20.0%" in html_out
    assert "Runs logged" in html_out


def test_render_html_includes_chartjs_cdn_and_canvases_when_data_exists():
    entries = [
        {"run_id": "r1", "stage_name": "dedup", "enabled": True, "tokens_before": 100, "tokens_after": 80, "extra": {}},
    ]
    html_out = render_html(aggregate(entries), "costs.jsonl")
    assert "chart.umd.min.js" in html_out
    assert "tkOverviewRequest" in html_out
    assert "tkSavingsPie" in html_out
    assert "needing internet access" in html_out  # the honest tradeoff note


def test_render_html_stage_savings_pie_is_percent_based_and_excludes_negative_stages():
    # Changed 2026-09-06 at the user's request: one %-based pie chart,
    # replacing the old absolute-tokens pie + separate "% saved by stage"
    # bar chart entirely.
    entries = [
        {"run_id": "r1", "stage_name": "dedup", "enabled": True, "tokens_before": 100, "tokens_after": 80, "extra": {}},
        {"run_id": "r1", "stage_name": "brevity_injector", "enabled": True, "tokens_before": 80, "tokens_after": 108, "extra": {}},
    ]
    html_out = render_html(aggregate(entries), "costs.jsonl")
    pie_segment = html_out.split("tkSavingsPie")[2]  # [0]=canvas id, [1]=getElementById call, [2]=chart config
    assert '"dedup"' in pie_segment.split(");")[0]
    assert "brevity_injector" not in pie_segment.split(");")[0]
    assert "20.0" in pie_segment.split(");")[0]  # % saved, not the absolute 20 tokens


def test_render_html_stage_savings_pie_excludes_no_token_impact_stages():
    entries = [
        {"run_id": "r1", "stage_name": "dedup", "enabled": True, "tokens_before": 100, "tokens_after": 80, "extra": {}},
        {"run_id": "r1", "stage_name": "cache_reorder_guard", "enabled": True, "tokens_before": 80, "tokens_after": 80, "extra": {}},
    ]
    html_out = render_html(aggregate(entries), "costs.jsonl")
    pie_segment = html_out.split("tkSavingsPie")[2].split(");")[0]
    assert "cache_reorder_guard" not in pie_segment


def test_render_html_cache_cards_only_appear_with_real_cache_data():
    # The cache donut was removed 2026-09-06, replaced by a request-tokens
    # donut (tkRequestDonut) for consistency with Output/Thinking -- cache
    # data now shows as cards only.
    entries = [{"run_id": "r1", "stage_name": "dedup", "enabled": True, "tokens_before": 100, "tokens_after": 80, "extra": {}}]
    no_cache_out = render_html(aggregate(entries), "costs.jsonl", aggregate_events([]))
    assert "No cache usage data logged" in no_cache_out
    assert "tkCacheDonut" not in no_cache_out

    cache_events = [{"type": "event", "event_type": "cache_usage", "fields": {"cache_read_input_tokens": 500, "cache_creation_input_tokens": 1724}}]
    with_cache_out = render_html(aggregate(entries), "costs.jsonl", aggregate_events(cache_events))
    assert "Cache read tokens" in with_cache_out
    assert "500" in with_cache_out
    assert "tkCacheDonut" not in with_cache_out
    assert "tkRequestDonut" in with_cache_out


def test_render_html_output_and_thinking_sections_only_show_real_data():
    entries = [{"run_id": "r1", "stage_name": "dedup", "enabled": True, "tokens_before": 100, "tokens_after": 80, "extra": {}}]
    no_data_out = render_html(aggregate(entries), "costs.jsonl", aggregate_events([]))
    assert "No output-token data logged" in no_data_out
    assert "No thinking-token data logged" in no_data_out
    assert "tkOutputDonut" not in no_data_out
    assert "tkThinkingDonut" not in no_data_out

    output_only_events = [
        {
            "type": "event",
            "event_type": "generation_usage",
            "fields": {
                "baseline_output_tokens": 900,
                "optimized_output_tokens": 850,
                "baseline_visible_text_tokens": 870,
                "optimized_visible_text_tokens": 848,
            },
        },
    ]
    output_only_out = render_html(aggregate(entries), "costs.jsonl", aggregate_events(output_only_events))
    assert "tkOutputDonut" in output_only_out
    assert "tkThinkingDonut" not in output_only_out

    with_thinking_events = [
        {
            "type": "event",
            "event_type": "generation_usage",
            "fields": {
                "baseline_output_tokens": 900,
                "optimized_output_tokens": 1200,
                "baseline_visible_text_tokens": 870,
                "optimized_visible_text_tokens": 848,
                "baseline_thinking_tokens": None,
                "optimized_thinking_tokens": 300,
            },
        },
    ]
    with_thinking_out = render_html(aggregate(entries), "costs.jsonl", aggregate_events(with_thinking_events))
    assert "tkOutputDonut" in with_thinking_out
    assert "tkThinkingDonut" in with_thinking_out
    output_script = with_thinking_out.split('id="tkOutputDonut"')[1].split("<script>", 1)[1].split(
        "</script>", 1
    )[0]
    assert "[900, 1200]" in output_script
    thinking_script = with_thinking_out.split('id="tkThinkingDonut"')[1].split("<script>", 1)[1].split(
        "</script>", 1
    )[0]
    assert "[0, 300]" in thinking_script
    # Output tokens section must come before Thinking tokens, matching the
    # user's requested section order.
    assert with_thinking_out.index('<div class="section-title">Output tokens</div>') < with_thinking_out.index(
        '<div class="section-title">Thinking tokens</div>'
    )


def test_render_html_combined_overview_donut_needs_at_least_two_categories():
    # A single-category "combined" chart would just duplicate the
    # individual donut above it, so it's omitted until there's something
    # real to compare.
    entries = [{"run_id": "r1", "stage_name": "dedup", "enabled": True, "tokens_before": 100, "tokens_after": 80, "extra": {"phase": "request"}}]
    request_only_out = render_html(aggregate(entries), "costs.jsonl", aggregate_events([]))
    assert "tkOverviewCombined" not in request_only_out

    output_events = [
        {
            "type": "event",
            "event_type": "generation_usage",
            "fields": {
                "baseline_output_tokens": 900,
                "optimized_output_tokens": 850,
                "baseline_visible_text_tokens": 870,
                "optimized_visible_text_tokens": 848,
            },
        },
    ]
    with_output_out = render_html(aggregate(entries), "costs.jsonl", aggregate_events(output_events))
    assert "tkOverviewCombined" in with_output_out
    script = with_output_out.split("<script>")[1].split("</script>", 1)[0]  # overview script is first
    combined_segment = script.split("tkOverviewCombined")[1].split("});")[0]
    assert '"Request (baseline)"' in combined_segment
    assert '"Output (baseline)"' in combined_segment
    assert '"Thinking (baseline)"' not in combined_segment

    no_stage_data_out = render_html(aggregate([]), "costs.jsonl", aggregate_events([]))
    assert "tkOverviewCombined" not in no_stage_data_out


def test_render_html_shows_min_median_max_columns_and_cards():
    entries = [
        {"run_id": "r1", "stage_name": "dedup", "enabled": True, "tokens_before": 100, "tokens_after": 80, "extra": {"phase": "request"}},
        {"run_id": "r2", "stage_name": "dedup", "enabled": True, "tokens_before": 100, "tokens_after": 60, "extra": {"phase": "request"}},
    ]
    html_out = render_html(aggregate(entries), "costs.jsonl")
    assert "Min run saved %" in html_out
    assert "Median run saved %" in html_out
    assert "Max run saved %" in html_out
    assert "Min / Median / Max %" in html_out
    row = html_out.split(">dedup<")[1].split("</tr>")[0]
    assert "20.0% / 30.0% / 40.0%" in row


def test_render_html_no_token_impact_stage_shows_dash_for_range_column():
    entries = [
        {"run_id": "r1", "stage_name": "cache_reorder_guard", "enabled": True, "tokens_before": 100, "tokens_after": 100, "extra": {}},
    ]
    html_out = render_html(aggregate(entries), "costs.jsonl")
    row = html_out.split(">cache_reorder_guard<")[1].split("</tr>")[0]
    assert row.count("&mdash;") == 2  # Saved cell AND the Min/Median/Max range cell


def test_render_html_shows_a_note_instead_of_a_misleading_zero_percent():
    # Regression test: a stage whose real effect isn't a request-token-
    # count change (e.g. cache_breakpoint_optimizer attaches cache_control
    # -- its benefit is a cheaper price per cached token, not fewer tokens)
    # used to render a bare "0.0%", which a user reasonably read as "did
    # nothing" rather than "working correctly, wrong metric for this
    # stage." Found via the user asking about it directly on a real
    # dashboard render (2026-09-06).
    entries = [
        {
            "run_id": "r1",
            "stage_name": "cache_breakpoint_optimizer",
            "enabled": True,
            "tokens_before": 100,
            "tokens_after": 100,
            "extra": {},
        },
    ]
    html_out = render_html(aggregate(entries), "costs.jsonl")
    row = html_out.split("cache_breakpoint_optimizer")[1].split("</tr>")[0]
    assert "cheaper price per cached token" in row
    assert "0.0%" not in row


def test_render_html_adds_a_contextual_note_alongside_the_real_percentage():
    # brevity_injector/post_hoc_trim keep their real, meaningful bar/percentage
    # (unlike the structurally-zero stages above) but get a short explanatory
    # note alongside it, since a negative or zero value for these two is
    # easy to misread without context (real overhead only on extraction
    # requests; a real zero that depends on what a batch of replies happened
    # to contain, not a broken mechanism).
    entries = [
        {"run_id": "r1", "stage_name": "brevity_injector", "enabled": True, "tokens_before": 72, "tokens_after": 100, "extra": {}},
    ]
    html_out = render_html(aggregate(entries), "costs.jsonl")
    row = html_out.split(">brevity_injector<")[1].split("</tr>")[0]
    assert "-38.9%" in row
    assert "only fires on extraction-shaped requests" in row


def test_render_html_still_shows_a_real_percentage_for_token_impacting_stages():
    entries = [
        {"run_id": "r1", "stage_name": "dedup", "enabled": True, "tokens_before": 100, "tokens_after": 80, "extra": {}},
    ]
    html_out = render_html(aggregate(entries), "costs.jsonl")
    assert "20.0%" in html_out
    assert "stage-note" not in html_out.split("dedup")[1].split("</tr>")[0]


def test_render_html_escapes_the_log_path():
    html_out = render_html(aggregate([]), "<script>alert(1)</script>.jsonl")
    assert "<script>alert(1)</script>" not in html_out
    assert "&lt;script&gt;" in html_out


def test_render_html_shows_no_data_placeholders_when_tier2_is_empty():
    entries = [{"run_id": "r1", "stage_name": "dedup", "enabled": True, "tokens_before": 100, "tokens_after": 80, "extra": {}}]
    html_out = render_html(aggregate(entries), "costs.jsonl", aggregate_events([]))
    assert "No cache usage data logged" in html_out
    assert "No output-token data logged" in html_out
    assert "No thinking-token data logged" in html_out
    assert "No TALE data logged" in html_out
    assert "No compress data logged" in html_out
    assert "No semantic-cache data logged" in html_out
    assert "No thinking re-injection data logged" in html_out


def test_render_html_savings_overview_appears_before_every_other_section():
    stage_entries = [{"run_id": "r1", "stage_name": "dedup", "enabled": True, "tokens_before": 100, "tokens_after": 80, "extra": {}}]
    html_out = render_html(aggregate(stage_entries), "costs.jsonl", aggregate_events([]))
    # Section title, not the CSS comment -- rendered exactly once as a
    # visible <div class="section-title">, so this can't false-pass off a
    # docstring/comment the way an earlier version of this test did.
    section_title = '<div class="section-title overview-title">Savings overview</div>'
    assert section_title in html_out
    assert "tkOverviewRequest" in html_out
    # Savings overview must come before every other section, per the
    # user's explicit ask ("this section should be placed at top before
    # any metric cards").
    overview_index = html_out.index(section_title)
    assert overview_index < html_out.index('<div class="section-title">Request tokens</div>')
    assert overview_index < html_out.index("Pipeline stage breakdown")
    assert overview_index < html_out.index('<div class="section-title">Output tokens</div>')
    assert overview_index < html_out.index('<div class="section-title">Thinking tokens</div>')


def test_render_html_savings_overview_only_shows_request_donut():
    # Output/thinking donuts live in their OWN sections (next to their
    # Tier 2 cards), not in the top "Savings overview" -- per the user's
    # explicit ask that each token type's chart sit in its relevant
    # section, the same way the request-tokens donut sits with the hero.
    stage_entries = [{"run_id": "r1", "stage_name": "dedup", "enabled": True, "tokens_before": 100, "tokens_after": 80, "extra": {}}]
    no_tier2_out = render_html(aggregate(stage_entries), "costs.jsonl", aggregate_events([]))
    assert "Savings overview" in no_tier2_out
    assert "tkOverviewRequest" in no_tier2_out
    assert "tkOutputDonut" not in no_tier2_out
    assert "tkThinkingDonut" not in no_tier2_out

    event_entries = [
        {
            "type": "event",
            "event_type": "generation_usage",
            "fields": {
                "baseline_output_tokens": 900,
                "optimized_output_tokens": 1200,
                "baseline_visible_text_tokens": 870,
                "optimized_visible_text_tokens": 848,
                "baseline_thinking_tokens": 0,
                "optimized_thinking_tokens": 300,
            },
        },
    ]
    with_gen_out = render_html(aggregate(stage_entries), "costs.jsonl", aggregate_events(event_entries))
    # Still only the request donut in the top section...
    overview_script = with_gen_out.split("<script>")[1].split("</script>", 1)[0]
    assert "tkOverviewRequest" in overview_script
    assert "tkOutputDonut" not in overview_script
    # ...but the output/thinking donuts now exist elsewhere on the page,
    # each with the correct real values.
    assert "tkOutputDonut" in with_gen_out
    assert "tkThinkingDonut" in with_gen_out
    output_script = with_gen_out.split("id=\"tkOutputDonut\"")[1].split("<script>", 1)[1].split("</script>", 1)[0]
    assert "[900, 1200]" in output_script
    thinking_script = with_gen_out.split("id=\"tkThinkingDonut\"")[1].split("<script>", 1)[1].split("</script>", 1)[0]
    assert "[0, 300]" in thinking_script
    assert "real cost, no baseline usage" in thinking_script


def test_render_html_shows_thinking_reinjection_savings_when_present():
    stage_entries = [{"run_id": "r1", "stage_name": "dedup", "enabled": True, "tokens_before": 100, "tokens_after": 80, "extra": {}}]
    event_entries = [
        {
            "type": "event",
            "event_type": "thinking_reinjection",
            "fields": {"omitted_turn2_input_tokens": 120, "summarized_turn2_input_tokens": 450},
        },
    ]
    html_out = render_html(aggregate(stage_entries), "costs.jsonl", aggregate_events(event_entries))
    assert "Thinking re-injection tokens saved (real, measured)" in html_out
    assert "330 (73.3%)" in html_out
    assert "RE-INJECTION SAVINGS" in html_out
    assert "No thinking re-injection data logged" not in html_out
    # n=1 is below the noise-floor threshold -- caveat must show.
    assert "independently-sampled real completions" in html_out


def test_render_html_hides_reinjection_noise_caveat_above_the_sample_threshold():
    stage_entries = [{"run_id": "r1", "stage_name": "dedup", "enabled": True, "tokens_before": 100, "tokens_after": 80, "extra": {}}]
    event_entries = [
        {
            "type": "event",
            "event_type": "thinking_reinjection",
            "fields": {"omitted_turn2_input_tokens": 120, "summarized_turn2_input_tokens": 450},
        }
        for _ in range(10)
    ]
    html_out = render_html(aggregate(stage_entries), "costs.jsonl", aggregate_events(event_entries))
    assert "Thinking re-injection tokens saved (real, measured)" in html_out
    assert "independently-sampled real completions" not in html_out


def test_render_html_shows_real_output_tokens_saved_when_present():
    stage_entries = [{"run_id": "r1", "stage_name": "dedup", "enabled": True, "tokens_before": 100, "tokens_after": 80, "extra": {}}]
    event_entries = [
        {
            "type": "event",
            "event_type": "generation_usage",
            "fields": {
                "baseline_output_tokens": 900,
                "optimized_output_tokens": 850,
                "baseline_visible_text_tokens": 870,
                "optimized_visible_text_tokens": 848,
            },
        },
    ]
    html_out = render_html(aggregate(stage_entries), "costs.jsonl", aggregate_events(event_entries))
    assert "Output tokens saved" in html_out
    assert "50 (5.6%)" in html_out
    assert "No output/thinking-token data logged" not in html_out
    # No sample logged real thinking data -- falls back to the estimate,
    # not the real-measured card.
    assert "Est. thinking tokens saved" in html_out
    assert "Thinking tokens consumed (real, measured)" not in html_out


def test_render_html_shows_real_thinking_tokens_consumed_when_present():
    stage_entries = [{"run_id": "r1", "stage_name": "dedup", "enabled": True, "tokens_before": 100, "tokens_after": 80, "extra": {}}]
    event_entries = [
        {
            "type": "event",
            "event_type": "generation_usage",
            "fields": {
                "baseline_output_tokens": 900,
                "optimized_output_tokens": 1200,
                "baseline_visible_text_tokens": 870,
                "optimized_visible_text_tokens": 848,
                "baseline_thinking_tokens": None,
                "optimized_thinking_tokens": 300,
            },
        },
    ]
    html_out = render_html(aggregate(stage_entries), "costs.jsonl", aggregate_events(event_entries))
    assert "Thinking tokens consumed (real, measured)" in html_out
    assert "300" in html_out
    assert "output_tokens_details.thinking_tokens" in html_out
    assert "Est. thinking tokens saved" not in html_out


def test_render_html_shows_output_tokens_min_median_max_cards_when_present():
    stage_entries = [{"run_id": "r1", "stage_name": "dedup", "enabled": True, "tokens_before": 100, "tokens_after": 80, "extra": {}}]
    event_entries = [
        {
            "type": "event",
            "event_type": "generation_usage",
            "fields": {"baseline_output_tokens": 900, "optimized_output_tokens": 850},
        },
        {
            "type": "event",
            "event_type": "generation_usage",
            "fields": {"baseline_output_tokens": 500, "optimized_output_tokens": 100},
        },
    ]
    html_out = render_html(aggregate(stage_entries), "costs.jsonl", aggregate_events(event_entries))
    assert "Min sample saved %" in html_out
    assert "Median sample saved %" in html_out
    assert "Max sample saved %" in html_out
    assert "80.0%" in html_out  # the 500->100 sample's real % saved


def test_render_html_hides_output_min_median_max_cards_when_no_samples():
    stage_entries = [{"run_id": "r1", "stage_name": "dedup", "enabled": True, "tokens_before": 100, "tokens_after": 80, "extra": {}}]
    html_out = render_html(aggregate(stage_entries), "costs.jsonl", aggregate_events([]))
    assert "Min sample saved %" not in html_out


def test_render_html_shows_thinking_tokens_min_median_max_cards_when_present():
    stage_entries = [{"run_id": "r1", "stage_name": "dedup", "enabled": True, "tokens_before": 100, "tokens_after": 80, "extra": {}}]
    event_entries = [
        {
            "type": "event",
            "event_type": "generation_usage",
            "fields": {
                "baseline_output_tokens": 100,
                "optimized_output_tokens": 100,
                "baseline_thinking_tokens": 0,
                "optimized_thinking_tokens": 50,
            },
        },
        {
            "type": "event",
            "event_type": "generation_usage",
            "fields": {
                "baseline_output_tokens": 100,
                "optimized_output_tokens": 100,
                "baseline_thinking_tokens": 40,
                "optimized_thinking_tokens": 0,
            },
        },
    ]
    html_out = render_html(aggregate(stage_entries), "costs.jsonl", aggregate_events(event_entries))
    assert "Min sample delta (tokens)" in html_out
    assert "Median sample delta (tokens)" in html_out
    assert "Max sample delta (tokens)" in html_out
    assert "-40" in html_out
    assert "50" in html_out


def test_render_html_shows_truncation_caveat_when_baseline_was_truncated():
    stage_entries = [{"run_id": "r1", "stage_name": "dedup", "enabled": True, "tokens_before": 100, "tokens_after": 80, "extra": {}}]
    event_entries = [
        {
            "type": "event",
            "event_type": "generation_usage",
            "fields": {
                "baseline_output_tokens": 60,
                "optimized_output_tokens": 960,
                "baseline_visible_text_tokens": 60,
                "optimized_visible_text_tokens": 900,
                "baseline_truncated": True,
                "optimized_truncated": False,
            },
        },
    ]
    html_out = render_html(aggregate(stage_entries), "costs.jsonl", aggregate_events(event_entries))
    assert "baseline replies were truncated" in html_out
    assert "1/1 baseline" in html_out


def test_render_html_no_truncation_caveat_when_nothing_was_truncated():
    stage_entries = [{"run_id": "r1", "stage_name": "dedup", "enabled": True, "tokens_before": 100, "tokens_after": 80, "extra": {}}]
    event_entries = [
        {
            "type": "event",
            "event_type": "generation_usage",
            "fields": {
                "baseline_output_tokens": 900,
                "optimized_output_tokens": 850,
                "baseline_visible_text_tokens": 870,
                "optimized_visible_text_tokens": 848,
                "baseline_truncated": False,
                "optimized_truncated": False,
            },
        },
    ]
    html_out = render_html(aggregate(stage_entries), "costs.jsonl", aggregate_events(event_entries))
    assert "baseline replies were truncated" not in html_out


def test_render_html_shows_real_tier2_numbers_when_present():
    stage_entries = [{"run_id": "r1", "stage_name": "dedup", "enabled": True, "tokens_before": 100, "tokens_after": 80, "extra": {}}]
    event_entries = [
        {"type": "event", "event_type": "cache_usage", "fields": {"cache_read_input_tokens": 1000, "cache_creation_input_tokens": 0}},
    ]
    html_out = render_html(aggregate(stage_entries), "costs.jsonl", aggregate_events(event_entries))
    assert "100.0%" in html_out  # cache hit rate
    assert "No cache usage data logged" not in html_out
    assert "No TALE data logged" in html_out  # still no TALE data


def test_render_html_empty_state_still_triggers_if_only_tier2_data_exists():
    # An edge case: events logged but no stage entries at all -- should
    # NOT show the "no runs logged" empty state, since there IS real data.
    event_entries = [{"type": "event", "event_type": "tale", "fields": {"budget_tokens": 100}}]
    html_out = render_html(aggregate([]), "costs.jsonl", aggregate_events(event_entries))
    assert "No runs logged yet" not in html_out
    assert "No TALE data logged" not in html_out


# --- server (real loopback HTTP, no mocks) --------------------------------


def test_server_serves_a_real_http_response(tmp_path):
    path = tmp_path / "costs.jsonl"
    logger = FileCostLogger(path)
    logger.log_stage("dedup", enabled=True, tokens_before=100, tokens_after=80, measured=True)

    server = run_server(str(path), port=0)  # port=0 -> OS picks a free port
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5) as resp:
            assert resp.status == 200
            body = resp.read().decode("utf-8")
            assert "dedup" in body
            assert "Tokenetics Dashboard" in body
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_server_reflects_the_log_file_fresh_on_each_request(tmp_path):
    # Read-only/decoupled requirement: no caching between requests -- a
    # second write to the log file must show up on the next GET, since the
    # dashboard has to work against a log another process keeps appending to.
    path = tmp_path / "costs.jsonl"
    logger = FileCostLogger(path)
    logger.log_stage("dedup", enabled=True, tokens_before=100, tokens_after=80, measured=True)

    server = run_server(str(path), port=0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5) as resp:
            assert "near_dup" not in resp.read().decode("utf-8")

        logger.log_stage("near_dup", enabled=True, tokens_before=80, tokens_after=60, measured=True)

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5) as resp:
            assert "near_dup" in resp.read().decode("utf-8")
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_server_serves_real_tier2_event_data(tmp_path):
    path = tmp_path / "costs.jsonl"
    logger = FileCostLogger(path)
    logger.log_stage("dedup", enabled=True, tokens_before=100, tokens_after=80, measured=True)
    logger.log_event("tale", budget_tokens=150, estimation_input_tokens=128, estimation_output_tokens=5)

    server = run_server(str(path), port=0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5) as resp:
            body = resp.read().decode("utf-8")
            assert "150 tok" in body
            assert "No TALE data logged" not in body
            assert "No compress data logged" in body
    finally:
        server.shutdown()
        thread.join(timeout=5)
