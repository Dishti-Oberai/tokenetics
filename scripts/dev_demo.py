#!/usr/bin/env python3
"""Manual dev-loop script for eyeballing Tokenetics against the real Anthropic API.

NOT part of pytest or CI -- it makes a real, billed API call. Run it by hand:

    uv run python scripts/dev_demo.py
    uv run python scripts/dev_demo.py --disable dedup

Requires ANTHROPIC_API_KEY in the environment. See CLAUDE.md's "Incremental
runnability" section for what this script is for and how it's expected to
grow as pipeline stages land in later phases.
"""

from __future__ import annotations

import argparse
from typing import Any

import anthropic

from tokenetics import Tokenetics
from tokenetics.core.request import from_api_kwargs
from tokenetics.core.tokenizer import count_tokens

# Includes a deliberate exact repeat (turns 0-1 repeated verbatim as turns
# 2-3) so dedup has something real to remove -- otherwise before/after would
# look identical even with the real stages running.
SAMPLE_KWARGS: dict[str, Any] = {
    "model": "claude-sonnet-5",
    "max_tokens": 300,
    "messages": [
        {"role": "user", "content": "What is a hash map?"},
        {
            "role": "assistant",
            "content": "A hash map stores key-value pairs and offers average "
            "O(1) lookup by hashing the key to an index.",
        },
        {"role": "user", "content": "What is a hash map?"},
        {
            "role": "assistant",
            "content": "A hash map stores key-value pairs and offers average "
            "O(1) lookup by hashing the key to an index.",
        },
        {
            "role": "user",
            "content": "In one short paragraph, explain what a hash map is.",
        },
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--disable",
        action="append",
        default=[],
        metavar="STAGE_NAME",
        help="Disable a stage by name for an on/off comparison, e.g. --disable dedup.",
    )
    args = parser.parse_args()

    client = anthropic.Anthropic()
    tk = Tokenetics(client=client)  # default pipeline: dedup, near_dup, schema_minification
    for stage in tk.stages:
        if stage.name in args.disable:
            stage.enabled = False

    before_tokens = count_tokens(from_api_kwargs(**SAMPLE_KWARGS), client)
    prepared = tk.prepare(**SAMPLE_KWARGS)
    after_tokens = count_tokens(from_api_kwargs(**prepared), client)

    fired = [s.name for s in tk.stages if s.enabled]
    print(f"stages fired: {fired or '(none -- all disabled)'}")
    print(f"request tokens before -> after: {before_tokens} -> {after_tokens}")

    response = client.messages.create(**prepared)
    print("\nreply (raw):")
    print(response.content)

    stored = tk.finalize(response)
    print("\nreply after post-hoc trim (what gets stored for future turns):")
    print(stored)

    entries = getattr(tk.logger, "entries", None)
    if entries:
        print("\nper-stage log:")
        for entry in entries:
            flag = " [ERROR]" if entry.extra.get("error") else ""
            timing = entry.extra.get("timing_seconds", 0.0)
            print(
                f"  {entry.stage_name}: {entry.tokens_before} -> {entry.tokens_after} tokens, "
                f"{timing:.4f}s, enabled={entry.enabled}{flag}"
            )


if __name__ == "__main__":
    main()
