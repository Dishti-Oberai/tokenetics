import json

from tokenetics.core.logger import FileCostLogger, InMemoryCostLogger, NullLogger


def test_in_memory_logger_records_entries():
    logger = InMemoryCostLogger()
    logger.log_stage(
        "dedup",
        enabled=True,
        tokens_before=100,
        tokens_after=80,
        measured=True,
        reason="exact match",
    )

    assert len(logger.entries) == 1
    entry = logger.entries[0]
    assert entry.stage_name == "dedup"
    assert entry.enabled is True
    assert entry.tokens_before == 100
    assert entry.tokens_after == 80
    assert entry.measured is True
    assert entry.extra == {"reason": "exact match"}


def test_in_memory_logger_records_multiple_entries_in_order():
    logger = InMemoryCostLogger()
    logger.log_stage("dedup", enabled=True)
    logger.log_stage("near_dup", enabled=False)

    assert [e.stage_name for e in logger.entries] == ["dedup", "near_dup"]


def test_null_logger_does_not_raise_and_records_nothing():
    logger = NullLogger()
    assert logger.log_stage("dedup", enabled=True, tokens_before=100, tokens_after=80) is None


def test_file_cost_logger_appends_one_json_line_per_call(tmp_path):
    path = tmp_path / "costs.jsonl"
    logger = FileCostLogger(path)
    logger.log_stage("dedup", enabled=True, tokens_before=100, tokens_after=80, measured=True, reason="x")
    logger.log_stage("near_dup", enabled=False)

    lines = path.read_text().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["stage_name"] == "dedup"
    assert first["enabled"] is True
    assert first["tokens_before"] == 100
    assert first["tokens_after"] == 80
    assert first["measured"] is True
    assert first["extra"] == {"reason": "x"}
    assert "timestamp" in first
    assert "run_id" in first


def test_file_cost_logger_shares_one_run_id_across_calls(tmp_path):
    path = tmp_path / "costs.jsonl"
    logger = FileCostLogger(path)
    logger.log_stage("dedup", enabled=True)
    logger.log_stage("near_dup", enabled=True)

    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert lines[0]["run_id"] == lines[1]["run_id"]


def test_file_cost_logger_appends_to_an_existing_file_rather_than_overwriting(tmp_path):
    path = tmp_path / "costs.jsonl"
    FileCostLogger(path).log_stage("dedup", enabled=True)
    FileCostLogger(path).log_stage("near_dup", enabled=True)  # a second "run"

    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(lines) == 2
    assert lines[0]["run_id"] != lines[1]["run_id"]  # different Tokenetics() instances


def test_file_cost_logger_log_stage_is_tagged_type_stage(tmp_path):
    path = tmp_path / "costs.jsonl"
    FileCostLogger(path).log_stage("dedup", enabled=True)
    record = json.loads(path.read_text().splitlines()[0])
    assert record["type"] == "stage"


def test_file_cost_logger_log_event_appends_a_type_event_record(tmp_path):
    path = tmp_path / "costs.jsonl"
    logger = FileCostLogger(path)
    logger.log_event("tale", budget_tokens=150, estimation_input_tokens=128)

    record = json.loads(path.read_text().splitlines()[0])
    assert record["type"] == "event"
    assert record["event_type"] == "tale"
    assert record["fields"] == {"budget_tokens": 150, "estimation_input_tokens": 128}
    assert "run_id" in record
    assert "timestamp" in record


def test_file_cost_logger_log_event_shares_run_id_with_log_stage(tmp_path):
    path = tmp_path / "costs.jsonl"
    logger = FileCostLogger(path)
    logger.log_stage("dedup", enabled=True)
    logger.log_event("tale", budget_tokens=150)

    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert lines[0]["run_id"] == lines[1]["run_id"]


def test_in_memory_logger_records_events():
    logger = InMemoryCostLogger()
    logger.log_event("compress", ratio_achieved=0.4, clamped=False)
    assert len(logger.events) == 1
    assert logger.events[0].event_type == "compress"
    assert logger.events[0].fields == {"ratio_achieved": 0.4, "clamped": False}


def test_null_logger_log_event_does_not_raise_and_records_nothing():
    logger = NullLogger()
    assert logger.log_event("tale", budget_tokens=150) is None
