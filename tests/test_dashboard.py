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


def test_render_html_escapes_the_log_path():
    html_out = render_html(aggregate([]), "<script>alert(1)</script>.jsonl")
    assert "<script>alert(1)</script>" not in html_out
    assert "&lt;script&gt;" in html_out


def test_render_html_shows_no_data_placeholders_when_tier2_is_empty():
    entries = [{"run_id": "r1", "stage_name": "dedup", "enabled": True, "tokens_before": 100, "tokens_after": 80, "extra": {}}]
    html_out = render_html(aggregate(entries), "costs.jsonl", aggregate_events([]))
    assert "No cache usage data logged" in html_out
    assert "No TALE data logged" in html_out
    assert "No compress data logged" in html_out
    assert "No semantic-cache data logged" in html_out


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
