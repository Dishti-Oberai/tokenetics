"""The Tokenetics orchestrator: runs the fixed pipeline of stages.

Pipeline order is not configurable -- see CLAUDE.md's non-negotiable
constraints. Every stage call (request-side or response-side) is wrapped
with the fail-open framework: a raised exception is caught, the stage is
skipped, and the request/text passes through unmodified rather than
breaking the whole call.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from typing import Any

import anthropic

from tokenetics.core.logger import CostLogger, InMemoryCostLogger
from tokenetics.core.plugin import Stage, StageConfig
from tokenetics.core.request import TokeneticsRequest, from_api_kwargs, to_api_kwargs
from tokenetics.core.response_stage import ResponseStage
from tokenetics.core.tokenizer import count_text_tokens, count_tokens
from tokenetics.stages.dedup import DedupStage
from tokenetics.stages.near_dup import NearDupStage
from tokenetics.stages.post_hoc_trim import PostHocTrimStage
from tokenetics.stages.schema_minification import SchemaMinificationStage

_log = logging.getLogger(__name__)


def _default_stages() -> tuple[Stage, ...]:
    # Fresh instances every call -- callers (e.g. dev_demo.py's --disable flag)
    # mutate stage.enabled directly, and sharing singleton instances across
    # Tokenetics() objects would leak that mutation between them.
    return (DedupStage(), NearDupStage(), SchemaMinificationStage())


def _default_response_stages() -> tuple[ResponseStage, ...]:
    return (PostHocTrimStage(),)


class Tokenetics:
    def __init__(
        self,
        stages: Sequence[Stage] | None = None,
        response_stages: Sequence[ResponseStage] | None = None,
        logger: CostLogger | None = None,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        self._stages: tuple[Stage, ...] = tuple(stages) if stages is not None else _default_stages()
        self._response_stages: tuple[ResponseStage, ...] = (
            tuple(response_stages) if response_stages is not None else _default_response_stages()
        )
        self._logger: CostLogger = logger if logger is not None else InMemoryCostLogger()
        self._client: anthropic.Anthropic | None = client

    @property
    def stages(self) -> tuple[Stage, ...]:
        return self._stages

    @property
    def response_stages(self) -> tuple[ResponseStage, ...]:
        return self._response_stages

    @property
    def logger(self) -> CostLogger:
        return self._logger

    def _get_client(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = anthropic.Anthropic()
        return self._client

    def prepare(self, **api_kwargs: Any) -> dict[str, Any]:
        """Run an Anthropic Messages API request through the request-side pipeline."""
        request = from_api_kwargs(**api_kwargs)
        for stage in self._stages:
            request = self._run_stage(stage, request)
        return to_api_kwargs(request)

    def finalize(self, response: Any) -> str:
        """Run a Messages API response through the response-side pipeline
        (stage 11: post-hoc trim), returning the text ready to store/re-inject
        as future context.

        Only text content is extracted for now -- tool_use/other block types
        aren't part of what post-hoc trim operates on.
        """
        text = "".join(
            block.text
            for block in getattr(response, "content", [])
            if getattr(block, "type", None) == "text"
        )
        model = getattr(response, "model", "")
        for stage in self._response_stages:
            text = self._run_response_stage(stage, text, model)
        return text

    def _run_stage(self, stage: Stage, request: TokeneticsRequest) -> TokeneticsRequest:
        config: StageConfig = {}
        client = self._get_client()
        tokens_before = count_tokens(request, client)
        start = time.perf_counter()

        try:
            if stage.enabled:
                result = stage.run(request, config, self._logger)
            else:
                result = stage.degraded_fallback(request, config, self._logger)
        except Exception:
            elapsed = time.perf_counter() - start
            _log.error(
                "stage %r raised; skipping it, request passed through unmodified",
                stage.name,
                exc_info=True,
            )
            self._logger.log_stage(
                stage.name,
                enabled=stage.enabled,
                tokens_before=tokens_before,
                tokens_after=tokens_before,  # unchanged -- the stage never applied
                measured=True,
                timing_seconds=elapsed,
                error=True,
            )
            return request

        elapsed = time.perf_counter() - start
        tokens_after = count_tokens(result, client)
        self._logger.log_stage(
            stage.name,
            enabled=stage.enabled,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            measured=True,
            timing_seconds=elapsed,
        )
        return result

    def _run_response_stage(self, stage: ResponseStage, text: str, model: str) -> str:
        config: StageConfig = {}
        client = self._get_client()
        tokens_before = count_text_tokens(text, model, client)
        start = time.perf_counter()

        try:
            if stage.enabled:
                result = stage.run(text, config, self._logger)
            else:
                result = stage.degraded_fallback(text, config, self._logger)
        except Exception:
            elapsed = time.perf_counter() - start
            _log.error(
                "response stage %r raised; skipping it, text passed through unmodified",
                stage.name,
                exc_info=True,
            )
            self._logger.log_stage(
                stage.name,
                enabled=stage.enabled,
                tokens_before=tokens_before,
                tokens_after=tokens_before,
                measured=True,
                timing_seconds=elapsed,
                error=True,
            )
            return text

        elapsed = time.perf_counter() - start
        tokens_after = count_text_tokens(result, model, client)
        self._logger.log_stage(
            stage.name,
            enabled=stage.enabled,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            measured=True,
            timing_seconds=elapsed,
        )
        return result
