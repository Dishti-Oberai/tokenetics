#!/usr/bin/env python3
"""Manual dev-loop script for eyeballing Tokenetics against the real Anthropic API.

NOT part of pytest or CI -- it makes a real, billed API call. Run it by hand:

    uv run python scripts/dev_demo.py
    uv run python scripts/dev_demo.py --disable dedup   # once real stages exist

Requires ANTHROPIC_API_KEY in the environment. See CLAUDE.md's "Incremental
runnability" section for what this script is for and how it's expected to
grow as pipeline stages land in later phases.
"""

from __future__ import annotations

import argparse
from typing import Any

import anthropic

from tokenetics import Tokenetics

SAMPLE_KWARGS: dict[str, Any] = {
    "model": "claude-sonnet-5",
    "max_tokens": 300,
    "messages": [
        {
            "role": "user",
            "content": "In one short paragraph, explain what a hash map is.",
        }
    ],
}


def _rough_size(kwargs: dict[str, Any]) -> int:
    """Character count of the request messages, as a stand-in for a token count.

    Phase 0 has no shared tokenizer yet -- CLAUDE.md requires token counts
    come from exactly one shared utility, which is a Phase 1 deliverable.
    This is a rough proxy only, swapped out once that utility exists.
    """
    return sum(len(str(m["content"])) for m in kwargs["messages"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--disable",
        action="append",
        default=[],
        metavar="STAGE_NAME",
        help="Disable a stage by name for an on/off comparison. No-op until "
        "real stages exist (Phase 2+).",
    )
    args = parser.parse_args()

    tk = Tokenetics()  # no real stages wired in yet -- Phase 0 baseline
    for stage in tk.stages:
        if stage.name in args.disable:
            stage.enabled = False

    before_size = _rough_size(SAMPLE_KWARGS)
    prepared = tk.prepare(**SAMPLE_KWARGS)
    after_size = _rough_size(prepared)

    fired = [s.name for s in tk.stages if s.enabled]
    print(f"stages fired: {fired or '(none yet -- Phase 0 baseline)'}")
    print(f"request size before -> after (chars, not tokens yet): {before_size} -> {after_size}")

    client = anthropic.Anthropic()
    response = client.messages.create(**prepared)
    print("\nreply:")
    print(response.content)


if __name__ == "__main__":
    main()
