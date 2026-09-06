import logging

from tokenetics import Tokenetics
from tokenetics.core.logger import CostLogger, InMemoryCostLogger
from tokenetics.core.plugin import Stage, StageConfig
from tokenetics.core.request import TokeneticsRequest
from tokenetics.core.response_stage import ResponseStage

SAMPLE_KWARGS = {
    "model": "claude-sonnet-5",
    "max_tokens": 100,
    "messages": [{"role": "user", "content": "hi"}],
}


class BoomStage(Stage):
    name = "boom"

    def run(
        self, request: TokeneticsRequest, config: StageConfig, logger: CostLogger
    ) -> TokeneticsRequest:
        raise RuntimeError("deliberately broken")


class MarkerStage(Stage):
    """A stage that just needs to exist, to prove the pipeline reached it."""

    name = "after-boom"

    def run(
        self, request: TokeneticsRequest, config: StageConfig, logger: CostLogger
    ) -> TokeneticsRequest:
        return request


class BoomResponseStage(ResponseStage):
    name = "boom_response"

    def run(self, text: str, config: StageConfig, logger: CostLogger) -> str:
        raise RuntimeError("deliberately broken")


def test_broken_stage_request_passes_through_unmodified(stub_client):
    tk = Tokenetics(stages=[BoomStage()], client=stub_client)
    prepared = tk.prepare(**SAMPLE_KWARGS)
    assert prepared == SAMPLE_KWARGS


def test_broken_stage_logs_at_error_level(stub_client, caplog):
    with caplog.at_level(logging.ERROR, logger="tokenetics.orchestrator"):
        tk = Tokenetics(stages=[BoomStage()], client=stub_client)
        tk.prepare(**SAMPLE_KWARGS)

    assert any(record.levelno == logging.ERROR for record in caplog.records)


def test_broken_stage_records_error_flag_in_cost_logger(stub_client):
    cost_logger = InMemoryCostLogger()
    tk = Tokenetics(stages=[BoomStage()], logger=cost_logger, client=stub_client)
    tk.prepare(**SAMPLE_KWARGS)

    assert len(cost_logger.entries) == 1
    entry = cost_logger.entries[0]
    assert entry.stage_name == "boom"
    assert entry.extra.get("error") is True
    assert entry.tokens_before == entry.tokens_after  # nothing changed


def test_pipeline_continues_after_broken_stage(stub_client):
    cost_logger = InMemoryCostLogger()
    tk = Tokenetics(stages=[BoomStage(), MarkerStage()], logger=cost_logger, client=stub_client)
    tk.prepare(**SAMPLE_KWARGS)

    assert [e.stage_name for e in cost_logger.entries] == ["boom", "after-boom"]


def test_broken_response_stage_text_passes_through_unmodified(stub_client, make_fake_response):
    tk = Tokenetics(response_stages=[BoomResponseStage()], client=stub_client)
    result = tk.finalize(make_fake_response("hello world"))
    assert result == "hello world"


def test_broken_response_stage_logs_at_error_level(stub_client, make_fake_response, caplog):
    with caplog.at_level(logging.ERROR, logger="tokenetics.orchestrator"):
        tk = Tokenetics(response_stages=[BoomResponseStage()], client=stub_client)
        tk.finalize(make_fake_response("hello world"))

    assert any(record.levelno == logging.ERROR for record in caplog.records)


def test_broken_response_stage_records_error_flag_in_cost_logger(stub_client, make_fake_response):
    cost_logger = InMemoryCostLogger()
    tk = Tokenetics(response_stages=[BoomResponseStage()], logger=cost_logger, client=stub_client)
    tk.finalize(make_fake_response("hello world"))

    assert len(cost_logger.entries) == 1
    entry = cost_logger.entries[0]
    assert entry.stage_name == "boom_response"
    assert entry.extra.get("error") is True
    assert entry.tokens_before == entry.tokens_after


class _EmptiesMessagesStage(Stage):
    """A stage that succeeds (doesn't raise) but produces a result the
    real count_tokens endpoint can't measure -- e.g. an empty message list.
    """

    name = "empties_messages"

    def run(
        self, request: TokeneticsRequest, config: StageConfig, logger: CostLogger
    ) -> TokeneticsRequest:
        from dataclasses import replace

        return replace(request, messages=[])


class _ReturnsEmptyStringStage(ResponseStage):
    """A response stage that succeeds but returns empty text -- the same
    unmeasurable-result shape, on the response side.
    """

    name = "returns_empty"

    def run(self, text: str, config: StageConfig, logger: CostLogger) -> str:
        return ""


class _RejectsUnmeasurableResultClient:
    """Simulates the real Anthropic API rejecting a request/response shape
    that a stage produced but that count_tokens can't be called on --
    e.g. empty messages or empty assistant content. Regression coverage
    for a real gap found via code review (2026-09-06): the orchestrator
    used to call count_tokens/count_text_tokens on a stage's result
    OUTSIDE the fail-open try/except, so a failure measuring an otherwise-
    successful stage's output would have crashed prepare()/finalize()
    entirely instead of falling back like every other stage failure does.
    """

    class _Messages:
        def count_tokens(self, **kwargs: object) -> object:
            from dataclasses import dataclass

            @dataclass
            class _Result:
                input_tokens: int

            messages = kwargs.get("messages") or []
            if not messages or any(m.get("content") == "" for m in messages):  # type: ignore[union-attr]
                raise RuntimeError("simulated: real API rejects this content shape")
            return _Result(input_tokens=10)

    def __init__(self) -> None:
        self.messages = self._Messages()


def test_stage_whose_result_cannot_be_measured_still_fails_open():
    cost_logger = InMemoryCostLogger()
    tk = Tokenetics(
        stages=[_EmptiesMessagesStage()],
        logger=cost_logger,
        client=_RejectsUnmeasurableResultClient(),  # type: ignore[arg-type]
    )
    prepared = tk.prepare(**SAMPLE_KWARGS)
    assert prepared == SAMPLE_KWARGS  # unmodified -- the stage's result was discarded

    assert len(cost_logger.entries) == 1
    entry = cost_logger.entries[0]
    assert entry.extra.get("error") is True
    assert entry.tokens_before == entry.tokens_after


def test_response_stage_whose_result_cannot_be_measured_still_fails_open(make_fake_response):
    cost_logger = InMemoryCostLogger()
    tk = Tokenetics(
        response_stages=[_ReturnsEmptyStringStage()],
        logger=cost_logger,
        client=_RejectsUnmeasurableResultClient(),  # type: ignore[arg-type]
    )
    result = tk.finalize(make_fake_response("hello world"))
    assert result == "hello world"  # unmodified -- the empty result was discarded

    assert len(cost_logger.entries) == 1
    entry = cost_logger.entries[0]
    assert entry.extra.get("error") is True
    assert entry.tokens_before == entry.tokens_after
