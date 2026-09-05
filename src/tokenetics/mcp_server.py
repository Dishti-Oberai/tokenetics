"""Thin MCP server layer over the Tokenetics core (Phase 12).

Per the project brief: "core PyPI package + MCP server (thin layer
importing the core)". This module owns no pipeline logic of its own --
it exposes a single tool, `optimize_request`, that forwards its input
straight to `Tokenetics().prepare()` and returns the optimized request.

Deliberately does NOT call the real Anthropic API itself. The returned
kwargs are meant to be handed to the MCP client's own `messages.create()`
call -- keeping this server free of API keys, billed calls, and any
model-call surface of its own, matching Tier 0's "no network calls of
its own" principle even though this file lives outside Tier 0 proper.

STDIO transport requirement (per the MCP Python SDK docs): nothing in
this module may write to stdout, since that would corrupt the JSON-RPC
stream the server communicates over. Errors are returned as tool results,
never printed.
"""

from __future__ import annotations

from typing import Any

from mcp.server import MCPServer

from tokenetics.orchestrator import Tokenetics

mcp = MCPServer("tokenetics")


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
        On success: the optimized request kwargs, ready to pass straight
        to `client.messages.create(**result)`.
        On failure (malformed input): {"error": "<message>"} -- the caller's
        original request is never silently dropped or guessed at.
    """
    try:
        tk = Tokenetics(stage_config=stage_config or {})
        return tk.prepare(**request)
    except Exception as exc:
        return {"error": str(exc)}


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
