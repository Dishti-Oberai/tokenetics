from typing import Any

from tokenetics import Tokenetics
from tokenetics.core.logger import CostLogger, InMemoryCostLogger, NullLogger
from tokenetics.core.plugin import Stage, StageConfig
from tokenetics.core.request import TokeneticsRequest, from_api_kwargs

SAMPLE_KWARGS = {
    "model": "claude-sonnet-5",
    "max_tokens": 100,
    "messages": [{"role": "user", "content": "hi"}],
}


class NoOpStage(Stage):
    name = "noop"

    def run(
        self, request: TokeneticsRequest, config: StageConfig, logger: CostLogger
    ) -> TokeneticsRequest:
        return request


def test_noop_stage_preserves_request():
    request = from_api_kwargs(**SAMPLE_KWARGS)
    result = NoOpStage().run(request, {}, NullLogger())
    assert result == request


def test_tokenetics_prepare_with_noop_stage_matches_input(stub_client):
    tk = Tokenetics(stages=[NoOpStage()], client=stub_client)
    prepared = tk.prepare(**SAMPLE_KWARGS)
    assert prepared == SAMPLE_KWARGS


def test_disabled_stage_uses_degraded_fallback_not_run(stub_client):
    class BoomStage(Stage):
        name = "boom"

        def run(
            self, request: TokeneticsRequest, config: StageConfig, logger: CostLogger
        ) -> TokeneticsRequest:
            raise AssertionError("run() must not be called when the stage is disabled")

        def degraded_fallback(
            self, request: TokeneticsRequest, config: StageConfig, logger: CostLogger
        ) -> TokeneticsRequest:
            return request

    tk = Tokenetics(stages=[BoomStage(enabled=False)], client=stub_client)
    prepared = tk.prepare(**SAMPLE_KWARGS)
    assert prepared == SAMPLE_KWARGS


def test_tokenetics_uses_in_memory_logger_by_default():
    tk = Tokenetics()
    assert isinstance(tk._logger, InMemoryCostLogger)


def test_tokenetics_accepts_custom_logger():
    class RecordingLogger:
        def log_stage(self, stage_name: str, **kwargs: Any) -> None:
            pass

    logger = RecordingLogger()
    tk = Tokenetics(stages=[NoOpStage()], logger=logger)
    assert tk._logger is logger


def test_prepare_records_a_cost_logger_entry_per_stage(stub_client):
    tk = Tokenetics(stages=[NoOpStage()], client=stub_client)
    tk.prepare(**SAMPLE_KWARGS)

    entries = tk.logger.entries  # type: ignore[attr-defined]
    assert len(entries) == 1
    entry = entries[0]
    assert entry.stage_name == "noop"
    assert entry.enabled is True
    assert entry.measured is True
    assert entry.tokens_before == entry.tokens_after  # no-op stage changes nothing
    assert "timing_seconds" in entry.extra


def test_default_pipeline_runs_the_real_foundation_stages_in_order():
    tk = Tokenetics()
    assert [s.name for s in tk.stages] == ["dedup", "near_dup", "schema_minification"]
    assert [s.name for s in tk.response_stages] == ["post_hoc_trim"]


def test_stage_notes_merge_into_the_single_orchestrator_log_entry(stub_client):
    # Regression test: a stage calling self.note() (e.g. DedupStage reporting
    # dropped_duplicates) must land as extra metadata on the orchestrator's
    # one log_stage() call per stage, not as a second, separate entry with
    # no token counts or timing.
    from tokenetics.stages.dedup import DedupStage

    tk = Tokenetics(stages=[DedupStage()], client=stub_client)
    tk.prepare(
        model="claude-sonnet-5",
        max_tokens=100,
        messages=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "hi"},
        ],
    )

    entries = tk.logger.entries  # type: ignore[attr-defined]
    assert len(entries) == 1
    entry = entries[0]
    assert entry.extra["dropped_duplicates"] == 1
    assert entry.tokens_before is not None
    assert entry.tokens_after is not None
    assert "timing_seconds" in entry.extra


def test_default_stages_are_not_shared_across_instances():
    # Regression test: default stage instances must be fresh per Tokenetics()
    # call. If they were shared singletons, disabling a stage on one instance
    # would silently disable it on every other instance too.
    tk_a = Tokenetics()
    tk_b = Tokenetics()

    tk_a.stages[0].enabled = False

    assert tk_a.stages[0].enabled is False
    assert tk_b.stages[0].enabled is True
