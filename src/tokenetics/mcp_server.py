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
"""

from __future__ import annotations

import os
from typing import Any

from mcp.server import MCPServer

from tokenetics.core.logger import FileCostLogger, InMemoryCostLogger
from tokenetics.orchestrator import Tokenetics

mcp = MCPServer("tokenetics")

_LOG_FILE_ENV = "TOKENETICS_MCP_LOG_FILE"


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


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
