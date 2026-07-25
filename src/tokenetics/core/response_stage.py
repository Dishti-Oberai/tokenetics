"""The response-side plugin contract -- stage 11 (post-hoc trim) and any
future response-side stages.

Parallel to Stage, but operates on the model's reply text instead of a
TokeneticsRequest: a response and a request are different shapes, and stage
11 runs after the API call (Tokenetics.finalize()), not before it
(Tokenetics.prepare()).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from tokenetics.core.logger import CostLogger
from tokenetics.core.plugin import StageConfig


class ResponseStage(ABC):
    name: str

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.extra: dict[str, Any] = {}

    def note(self, **extra: Any) -> None:
        """See Stage.note() -- same contract, response-side."""
        self.extra.update(extra)

    @abstractmethod
    def run(self, text: str, config: StageConfig, logger: CostLogger) -> str: ...

    def degraded_fallback(self, text: str, config: StageConfig, logger: CostLogger) -> str:
        return text
