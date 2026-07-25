"""The plugin contract every pipeline stage implements.

Per CLAUDE.md: `plugin(request, config, logger) -> request`, against the
internal request representation -- never raw Anthropic API JSON.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from tokenetics.core.logger import CostLogger
from tokenetics.core.request import TokeneticsRequest

StageConfig = dict[str, Any]


class Stage(ABC):
    """Base class every pipeline stage subclasses.

    Concrete stages set `name` and implement `run`. `degraded_fallback` is
    only overridden by stages that define a real degraded mode (e.g. the
    context scheduler's last-N truncation) -- see CLAUDE.md's non-negotiable
    constraint that disabling a stage with a defined fallback must engage it,
    not silently no-op.
    """

    name: str

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    @abstractmethod
    def run(
        self, request: TokeneticsRequest, config: StageConfig, logger: CostLogger
    ) -> TokeneticsRequest: ...

    def degraded_fallback(
        self, request: TokeneticsRequest, config: StageConfig, logger: CostLogger
    ) -> TokeneticsRequest:
        return request
