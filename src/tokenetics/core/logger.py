"""The cost logger every stage logs through.

`InMemoryCostLogger` is the real, default implementation: it just remembers
every log_stage() call so it can be read back afterward (by scripts/dev_demo.py).
`NullLogger` is kept as an explicit "log nothing" option. `FileCostLogger`
(added Phase 11) persists entries to a JSONL file -- the durable "structured
log" the Phase 11 dashboard reads; nothing else in the project persists
CostLogger entries anywhere, so this is what makes the dashboard possible.

`log_event()` (added Phase 11, second pass) is a sibling to `log_stage()`
for data that exists OUTSIDE the pipeline entirely -- real `usage.
cache_read_input_tokens` from the caller's own `messages.create()` call,
TALE's round-trip cost, compress's achieved ratio, semantic-cache's hit/
miss. None of that is ever known inside `Tokenetics.prepare()`/
`finalize()` (Tier 0 never makes the actual API call, and the Tier 2
extras live entirely outside the pipeline), so it can't go through
`log_stage()`. `log_event()` gives it the same `run_id`-tagged, JSONL-
persisted home instead, which is what let the dashboard grow a Tier 2/
cache section without touching the orchestrator at all.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
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

    def log_event(self, event_type: str, **fields: Any) -> None: ...


@dataclass
class LogEntry:
    stage_name: str
    enabled: bool
    tokens_before: int | None = None
    tokens_after: int | None = None
    measured: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class LogEvent:
    event_type: str
    fields: dict[str, Any] = field(default_factory=dict)


class InMemoryCostLogger:
    """The default CostLogger: records every log_stage()/log_event() call
    for later reading.
    """

    def __init__(self) -> None:
        self.entries: list[LogEntry] = []
        self.events: list[LogEvent] = []

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

    def log_event(self, event_type: str, **fields: Any) -> None:
        self.events.append(LogEvent(event_type=event_type, fields=fields))


class FileCostLogger:
    """Persists every log_stage()/log_event() call as one JSON line
    appended to `path`.

    Read-only/decoupled by construction (Phase 11's dashboard requirement):
    this logger only ever appends to a file -- it never reads its own
    output back, has no effect on the request path beyond a local disk
    write, and a failure here (disk full, permissions) would surface as a
    normal exception from `log_stage()`/`log_event()` like any other stage
    bug, not a special path.

    Every entry from one `FileCostLogger` instance shares the same
    `run_id` (a UUID generated once, at construction) -- the convention
    throughout this project is one fresh `Tokenetics()` (and therefore one
    fresh logger) per request, so this groups a single request's stage
    entries (and any `log_event()` calls the caller makes against the same
    instance) together for the dashboard without needing any change to the
    orchestrator's frozen `prepare()`/`finalize()` API (`Tokenetics.
    __init__` already accepts a `logger` -- passing a `FileCostLogger`
    needs no new plumbing).

    Records are tagged `"type": "stage"` or `"type": "event"` so the
    dashboard's `aggregate.py` can route them separately without guessing
    from field shape.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.run_id = str(uuid.uuid4())

    def _write(self, record: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

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
        self._write(
            {
                "type": "stage",
                "run_id": self.run_id,
                "timestamp": time.time(),
                "stage_name": stage_name,
                "enabled": enabled,
                "tokens_before": tokens_before,
                "tokens_after": tokens_after,
                "measured": measured,
                "extra": extra,
            }
        )

    def log_event(self, event_type: str, **fields: Any) -> None:
        self._write(
            {
                "type": "event",
                "run_id": self.run_id,
                "timestamp": time.time(),
                "event_type": event_type,
                "fields": fields,
            }
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

    def log_event(self, event_type: str, **fields: Any) -> None:
        return None
