from typing import Any

from tokenetics import Tokenetics
from tokenetics.core.logger import CostLogger, NullLogger
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


def test_tokenetics_prepare_with_noop_stage_matches_input():
    tk = Tokenetics(stages=[NoOpStage()])
    prepared = tk.prepare(**SAMPLE_KWARGS)
    assert prepared == SAMPLE_KWARGS


def test_disabled_stage_uses_degraded_fallback_not_run():
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

    tk = Tokenetics(stages=[BoomStage(enabled=False)])
    prepared = tk.prepare(**SAMPLE_KWARGS)
    assert prepared == SAMPLE_KWARGS


def test_tokenetics_uses_null_logger_by_default():
    tk = Tokenetics()
    assert isinstance(tk._logger, NullLogger)


def test_tokenetics_accepts_custom_logger():
    class RecordingLogger:
        def log_stage(self, stage_name: str, **kwargs: Any) -> None:
            pass

    logger = RecordingLogger()
    tk = Tokenetics(stages=[NoOpStage()], logger=logger)
    assert tk._logger is logger
