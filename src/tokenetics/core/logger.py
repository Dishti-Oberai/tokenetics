"""The cost-logger interface every stage logs through.

This is a stub: Phase 1 builds the real implementation (token counts via the
shared tokenizer, timing, measured/estimated tagging). Defining the interface
now means the plugin contract's signature is stable before any real stage --
or the real logger -- exists.
"""

from __future__ import annotations

from typing import Any, Protocol


class CostLogger(Protocol):
    def log_stage(
        self,
        stage_name: str,
        *,
        enabled: bool,
        tokens_before: int | None = None,
        tokens_after: int | None = None,
        measured: bool = False,
        **extra: Any,
    ) -> None: ...


class NullLogger:
    """No-op CostLogger, used until Phase 1's real implementation exists."""

    def log_stage(
        self,
        stage_name: str,
        *,
        enabled: bool,
        tokens_before: int | None = None,
        tokens_after: int | None = None,
        measured: bool = False,
        **extra: Any,
    ) -> None:
        return None
