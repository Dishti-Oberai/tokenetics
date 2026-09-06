"""Thin MCP server layer over the Tokenetics core (Phase 12).

Per the project brief: "core PyPI package + MCP server (thin layer
importing the core)". This module owns no pipeline logic of its own --
it exposes a single tool, `optimize_request`, that forwards its input
straight to `Tokenetics().prepare()` and returns the optimized request
plus a token-savings summary computed from that same call's own
CostLogger entries.

Deliberately does NOT call the real Anthropic API itself. The returned
kwargs are meant to be handed to the MCP client's own `messages.create()`
call -- keeping this server free of API keys, billed calls, and any
model-call surface of its own, matching Tier 0's "no network calls of
its own" principle even though this file lives outside Tier 0 proper.

STDIO transport requirement (per the MCP Python SDK docs): nothing in
this module may write to stdout, since that would corrupt the JSON-RPC
stream the server communicates over. Errors are returned as tool results,
never printed.

Optional persistence: if `TOKENETICS_MCP_LOG_FILE` is set, every call's
stage entries are also replayed into a `FileCostLogger` at that path --
the same JSONL log `scripts/dashboard.py` already reads, so MCP-driven
runs show up alongside `dev_demo.py --log-to` runs on the same dashboard.
Off by default (this project's established opt-in-for-persistence
pattern; a stdio MCP server writing to an unannounced file by default
would be a surprising side effect).

Optional dashboard auto-open: if `TOKENETICS_MCP_OPEN_DASHBOARD` is also
set (any truthy value), `main()` starts the same read-only dashboard
server `scripts/dashboard.py` runs (via `tokenetics.dashboard.server.
run_server`, imported directly rather than shelling out to the script) in
a background thread, against `TOKENETICS_MCP_LOG_FILE`, and opens it in
the default browser (`webbrowser.open`). Same opt-in-only reasoning as
persistence itself, doubled: this has a real side effect (binds a port,
launches a browser) on top of the file-write one, and only makes sense
once `TOKENETICS_MCP_LOG_FILE` is already set -- there's nothing to show
otherwise. Port defaults to 8765 (dashboard.py's own default), overridable
via `TOKENETICS_MCP_DASHBOARD_PORT`. Best-effort: any failure (port
already bound, no browser/display in a headless environment) is logged
via the stdlib `logging` module (stderr by default, never stdout -- see
the stdio requirement above) and never prevents the MCP server itself
from starting.
"""

from __future__ import annotations

import logging
import os
import threading
import webbrowser
from typing import Any

from mcp.server import MCPServer

from tokenetics.core.logger import FileCostLogger, InMemoryCostLogger
from tokenetics.dashboard.server import run_server
from tokenetics.orchestrator import Tokenetics

mcp = MCPServer("tokenetics")

_log = logging.getLogger(__name__)

_LOG_FILE_ENV = "TOKENETICS_MCP_LOG_FILE"
_OPEN_DASHBOARD_ENV = "TOKENETICS_MCP_OPEN_DASHBOARD"
_DASHBOARD_PORT_ENV = "TOKENETICS_MCP_DASHBOARD_PORT"
_DEFAULT_DASHBOARD_PORT = 8765


def _stats_from_entries(logger: InMemoryCostLogger) -> dict[str, Any]:
    # Same convention as dashboard/aggregate.py's aggregate(): overall
    # savings is the first request-side stage's tokens_before -> the last
    # request-side stage's tokens_after. Every entry here is already
    # phase="request" (optimize_request only ever calls prepare(), never
    # finalize()), but the phase check is kept for consistency with that
    # same logic rather than assuming it.
    request_entries = [e for e in logger.entries if e.extra.get("phase", "request") == "request"]
    before_values = [e.tokens_before for e in request_entries if e.tokens_before is not None]
    after_values = [e.tokens_after for e in request_entries if e.tokens_after is not None]
    tokens_before = before_values[0] if before_values else 0
    tokens_after = after_values[-1] if after_values else 0
    tokens_saved = tokens_before - tokens_after
    saved_pct = (tokens_saved / tokens_before * 100) if tokens_before else 0.0
    return {
        "tokens_before": tokens_before,
        "tokens_after": tokens_after,
        "tokens_saved": tokens_saved,
        "saved_pct": round(saved_pct, 1),
    }


def _persist_if_configured(logger: InMemoryCostLogger) -> None:
    log_file = os.environ.get(_LOG_FILE_ENV)
    if not log_file:
        return
    file_logger = FileCostLogger(log_file)
    for entry in logger.entries:
        file_logger.log_stage(
            entry.stage_name,
            enabled=entry.enabled,
            tokens_before=entry.tokens_before,
            tokens_after=entry.tokens_after,
            measured=entry.measured,
            **entry.extra,
        )


@mcp.tool()
def optimize_request(
    request: dict[str, Any], stage_config: dict[str, dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Run an Anthropic Messages API request through Tokenetics' Tier 0 pipeline.

    Args:
        request: Anthropic Messages API kwargs as a JSON object -- the same
            fields you'd pass to `client.messages.create()` (model, messages,
            system, tools, max_tokens, etc.).
        stage_config: optional per-stage config, keyed by stage name (e.g.
            {"context_scheduler": {"token_budget": 4000}}). A stage not
            listed here uses its own conservative defaults.

    Returns:
        On success: {"request": <optimized kwargs, ready for
        client.messages.create(**result["request"])>, "stats": {
        "tokens_before", "tokens_after", "tokens_saved", "saved_pct"}}.
        On failure (malformed input): {"error": "<message>"} -- the caller's
        original request is never silently dropped or guessed at.
    """
    logger = InMemoryCostLogger()
    try:
        tk = Tokenetics(stage_config=stage_config or {}, logger=logger)
        optimized = tk.prepare(**request)
    except Exception as exc:
        return {"error": str(exc)}

    _persist_if_configured(logger)
    return {"request": optimized, "stats": _stats_from_entries(logger)}


def _maybe_open_dashboard() -> None:
    """See module docstring's "Optional dashboard auto-open" section.
    Opt-in via TOKENETICS_MCP_OPEN_DASHBOARD, and only does anything if
    TOKENETICS_MCP_LOG_FILE is also set -- no log file, nothing to show.
    Best-effort: never raises, never blocks server startup, never writes
    to stdout (uses `logging`, which defaults to stderr).
    """
    if not os.environ.get(_OPEN_DASHBOARD_ENV):
        return
    log_file = os.environ.get(_LOG_FILE_ENV)
    if not log_file:
        _log.info(
            "%s is set but %s is not -- nothing to show, skipping dashboard.",
            _OPEN_DASHBOARD_ENV,
            _LOG_FILE_ENV,
        )
        return

    port = int(os.environ.get(_DASHBOARD_PORT_ENV, _DEFAULT_DASHBOARD_PORT))
    try:
        server = run_server(log_file, port=port)
    except OSError as exc:
        _log.info("Could not start the dashboard server on port %d: %s", port, exc)
        return

    threading.Thread(target=server.serve_forever, daemon=True).start()

    url = f"http://127.0.0.1:{port}/"
    try:
        webbrowser.open(url)
    except Exception as exc:  # pragma: no cover -- environment-dependent (headless, no browser)
        _log.info("Dashboard server started at %s but could not open a browser: %s", url, exc)


def main() -> None:
    _maybe_open_dashboard()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
