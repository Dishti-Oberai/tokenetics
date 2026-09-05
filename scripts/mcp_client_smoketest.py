"""Real MCP client round-trip smoke test for the Phase 12 MCP server.

Not part of pytest (spawns a real subprocess over stdio and, with a live
ANTHROPIC_API_KEY set, makes a real `count_tokens` call inside
`Tokenetics().prepare()`) -- run by hand, same convention as dev_demo.py.

Sends two requests so real savings are visible by contrast, not just
asserted: a trivial one where nothing is prunable (the honest 0% case),
and a realistic one -- an accidentally double-sent user message (a common
real frontend-retry bug) plus four tools irrelevant to what's actually
being asked -- where dedup and schema_minification both have real work
to do.
"""

import asyncio
import os
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

_WEATHER_QUESTION = (
    "Hey, quick question -- what's the weather like in Paris today? I'm trying "
    "to decide whether to pack an umbrella for my trip. It's been raining a lot "
    "lately and I want to plan accordingly before I head out this afternoon."
)

_TOOLS = [
    {
        "name": "get_weather",
        "description": "Get the current weather conditions and forecast for a given city.",
        "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
    {
        "name": "send_email",
        "description": "Send an email on the user's behalf to a given recipient with a subject and body.",
        "input_schema": {
            "type": "object",
            "properties": {
                "to": {"type": "string"},
                "subject": {"type": "string"},
                "body": {"type": "string"},
            },
        },
    },
    {
        "name": "search_web",
        "description": "Search the public web for a query and return the top matching result snippets.",
        "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
    },
    {
        "name": "calculate",
        "description": "Evaluate a mathematical expression and return the numeric result.",
        "input_schema": {"type": "object", "properties": {"expression": {"type": "string"}}},
    },
    {
        "name": "get_stock_price",
        "description": "Look up the current trading price for a given stock ticker symbol.",
        "input_schema": {"type": "object", "properties": {"ticker": {"type": "string"}}},
    },
]


async def _optimize(session: ClientSession, label: str, request: dict[str, Any]) -> None:
    result = await session.call_tool("optimize_request", {"request": request})
    content = result.structured_content or {}
    print(f"\n--- {label} ---")
    print("STATS:", content.get("stats"))
    if "error" in content:
        print("ERROR:", content["error"])
    else:
        optimized = content["request"]
        print("tools kept:", [t["name"] for t in optimized.get("tools", [])])
        print("messages kept:", len(optimized.get("messages", [])))


async def main() -> None:
    params = StdioServerParameters(
        command="uv",
        args=["run", "python", "-m", "tokenetics.mcp_server"],
        # mcp.client.stdio only inherits a small env allowlist by default
        # (HOME/LOGNAME/PATH/SHELL/TERM/USER) -- ANTHROPIC_API_KEY needs to
        # be passed explicitly, same as any real MCP host config would.
        env=dict(os.environ),
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print("TOOLS:", [t.name for t in tools.tools])

            await _optimize(
                session,
                "trivial request (nothing prunable -- the honest 0% case)",
                {
                    "model": "claude-sonnet-5",
                    "max_tokens": 100,
                    "system": "You are a helpful assistant. " * 100,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )

            await _optimize(
                session,
                "realistic request (duplicate user turn + 4 irrelevant tools)",
                {
                    "model": "claude-sonnet-5",
                    "max_tokens": 200,
                    "tools": _TOOLS,
                    "messages": [
                        {"role": "user", "content": _WEATHER_QUESTION},
                        {"role": "user", "content": _WEATHER_QUESTION},  # accidental double-send
                    ],
                },
            )


if __name__ == "__main__":
    asyncio.run(main())
