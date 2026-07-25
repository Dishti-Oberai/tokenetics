#!/usr/bin/env python3
"""Manual dev-loop script for eyeballing Tokenetics against the real Anthropic API.

NOT part of pytest or CI -- it makes a real, billed API call. Run it by hand:

    uv run python scripts/dev_demo.py
    uv run python scripts/dev_demo.py --disable dedup
    uv run python scripts/dev_demo.py --model claude-haiku-4-5   # cheapest smoke test

Requires ANTHROPIC_API_KEY in the environment. See CLAUDE.md's "Incremental
runnability" section for what this script is for and how it's expected to
grow as pipeline stages land in later phases.

Cost notes: thinking is off by default here (pass --show-thinking to turn it
on) since it's pure overhead for eyeballing pipeline behavior -- on Sonnet 5,
omitting `thinking` entirely runs adaptive thinking automatically, so leaving
it unset would silently bill thinking tokens on every run. The count_tokens
calls (before/after) are free; only the one messages.create() call is billed.
"""

from __future__ import annotations

import argparse
from typing import Any

import anthropic

from tokenetics import Tokenetics
from tokenetics.core.request import from_api_kwargs
from tokenetics.core.tokenizer import count_text_tokens, count_tokens

# Includes a deliberate exact repeat (turns 0-1 repeated verbatim as turns
# 2-3) so dedup has something real to remove -- otherwise before/after would
# look identical even with the real stages running.
SAMPLE_KWARGS: dict[str, Any] = {
    "model": "claude-sonnet-5",
    "max_tokens": 100,
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
    parser.add_argument(
        "--show-thinking",
        action="store_true",
        help=(
            "Turn on adaptive thinking with display='summarized' and estimate "
            "thinking-token spend from response.usage. Costs more than the "
            "default run. Note: no stage in the pipeline reduces thinking "
            "tokens yet (that's stage 9c, Phase 4) -- this is a baseline "
            "measurement, not an optimization."
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        metavar="MODEL_ID",
        help="Override SAMPLE_KWARGS['model'], e.g. claude-haiku-4-5 for the cheapest smoke test.",
    )
    args = parser.parse_args()

    client = anthropic.Anthropic()
    tk = Tokenetics(client=client)  # default pipeline: dedup, near_dup, schema_minification
    for stage in tk.stages:
        if stage.name in args.disable:
            stage.enabled = False

    request_kwargs = dict(SAMPLE_KWARGS)
    if args.model:
        request_kwargs["model"] = args.model

    if args.show_thinking:
        # display="summarized" populates the thinking block's text so we can
        # measure it. The default, "omitted", bills the same tokens but ships
        # an empty text field, so there'd be nothing to count.
        request_kwargs["thinking"] = {"type": "adaptive", "display": "summarized"}
    else:
        # Sonnet 5 runs adaptive thinking automatically when `thinking` is
        # omitted -- disable it explicitly so a default run doesn't silently
        # bill thinking tokens nobody asked to see.
        request_kwargs["thinking"] = {"type": "disabled"}

    before_tokens = count_tokens(from_api_kwargs(**request_kwargs), client)
    prepared = tk.prepare(**request_kwargs)
    after_tokens = count_tokens(from_api_kwargs(**prepared), client)

    fired = [s.name for s in tk.stages if s.enabled]
    print(f"stages fired: {fired or '(none -- all disabled)'}")
    print(f"request tokens before -> after: {before_tokens} -> {after_tokens}")

    response = client.messages.create(**prepared)
    print("\nreply (raw):")
    print(response.content)

    usage = getattr(response, "usage", None)
    if usage is not None:
        print(f"\nusage: input_tokens={usage.input_tokens}, output_tokens={usage.output_tokens}")

    if args.show_thinking and usage is not None:
        visible_text = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
        visible_text_tokens = count_text_tokens(visible_text, request_kwargs["model"], client)
        thinking_tokens_estimate = usage.output_tokens - visible_text_tokens
        print(
            f"thinking tokens (estimated, measured=False): "
            f"output_tokens({usage.output_tokens}) - visible_text_tokens({visible_text_tokens}) "
            f"= {thinking_tokens_estimate}"
        )
        thinking_blocks = [
            block for block in response.content if getattr(block, "type", None) == "thinking"
        ]
        if thinking_blocks:
            print("\nthinking (summarized):")
            for block in thinking_blocks:
                print(block.thinking)

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
