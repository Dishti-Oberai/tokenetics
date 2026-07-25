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
