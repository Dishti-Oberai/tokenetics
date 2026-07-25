from tokenetics.core.logger import InMemoryCostLogger, NullLogger


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
