"""The Tokenetics orchestrator: runs the fixed pipeline of stages.

Pipeline order is not configurable -- see CLAUDE.md's non-negotiable
constraints. As of Phase 0 there are no real stages yet; this just proves the
orchestrator/plugin contract round-trips correctly.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from tokenetics.core.logger import CostLogger, NullLogger
from tokenetics.core.plugin import Stage
from tokenetics.core.request import from_api_kwargs, to_api_kwargs


class Tokenetics:
    def __init__(
        self,
        stages: Sequence[Stage] = (),
        logger: CostLogger | None = None,
    ) -> None:
        self._stages: tuple[Stage, ...] = tuple(stages)
        self._logger: CostLogger = logger if logger is not None else NullLogger()

    @property
    def stages(self) -> tuple[Stage, ...]:
        return self._stages

    def prepare(self, **api_kwargs: Any) -> dict[str, Any]:
        """Run an Anthropic Messages API request through the pipeline."""
        request = from_api_kwargs(**api_kwargs)
        for stage in self._stages:
            config: dict[str, Any] = {}
            if stage.enabled:
                request = stage.run(request, config, self._logger)
            else:
                request = stage.degraded_fallback(request, config, self._logger)
        return to_api_kwargs(request)
