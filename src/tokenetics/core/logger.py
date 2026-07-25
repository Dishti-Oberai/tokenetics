"""The cost logger every stage logs through.

`InMemoryCostLogger` is the real, default implementation: it just remembers
every log_stage() call so it can be read back afterward (by scripts/dev_demo.py
today, and by the Phase 12 dashboard eventually). `NullLogger` is kept as an
explicit "log nothing" option.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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


@dataclass
class LogEntry:
    stage_name: str
    enabled: bool
    tokens_before: int | None = None
    tokens_after: int | None = None
    measured: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


class InMemoryCostLogger:
    """The default CostLogger: records every log_stage() call for later reading."""

    def __init__(self) -> None:
        self.entries: list[LogEntry] = []

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
        self.entries.append(
            LogEntry(
                stage_name=stage_name,
                enabled=enabled,
                tokens_before=tokens_before,
                tokens_after=tokens_after,
                measured=measured,
                extra=extra,
            )
        )


class NullLogger:
    """No-op CostLogger, for callers who explicitly want logging off."""

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
