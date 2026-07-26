#!/usr/bin/env python3
"""Manual dev-loop script for eyeballing Tokenetics against the real Anthropic API.

NOT part of pytest or CI. Run it by hand:

    uv run python scripts/dev_demo.py --list-scenarios
    uv run python scripts/dev_demo.py --all-scenarios          # free -- no completions
    uv run python scripts/dev_demo.py --scenario extraction
    uv run python scripts/dev_demo.py --scenario code --disable adaptive_budget
    uv run python scripts/dev_demo.py --scenario dedup --model claude-haiku-4-5

Requires ANTHROPIC_API_KEY in the environment. See CLAUDE.md's "Incremental
runnability" section for what this script is for and how it's expected to
grow as pipeline stages land in later phases.

Scenarios: one fixed sample conversation (`dedup`) doesn't exercise most of
the pipeline -- structured_output/brevity_injector only ever fire on
extraction-shaped requests, near_dup needs genuinely similar (not identical)
text, tool-relevance filtering needs an irrelevant tool actually registered,
and so on. SCENARIOS below is a small library, each one built to make a
specific stage (or corner case) visibly do something. --list-scenarios
prints what's available without making any API calls.

Cost model: most of these stages only touch the *request* -- they never
need a real completion to verify they fired correctly. `tk.prepare()` only
calls the free count_tokens endpoint, never the billed messages.create().
--all-scenarios uses exactly this: it runs every scenario through
`prepare()` only, so you can check all of Phase 0-4's behavior, including
corner cases, for $0. Use --scenario NAME (without --prepare-only) when you
actually want to see a real reply for one scenario -- that's the only path
that spends money, and it's exactly one call.

As of Phase 4, `adaptive_budget` (stage 9c) sets a task-type-derived
thinking effort + max_tokens whenever the classifier confidently labels the
request and the model supports adaptive thinking, so this script no longer
needs to force-disable thinking itself. Pass --show-thinking to force
display="summarized" and inspect the reasoning text yourself (this
pre-empts adaptive_budget's own choice, since it always respects an
already-caller-set thinking config).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

import anthropic

from tokenetics import Tokenetics
from tokenetics.core.request import from_api_kwargs
from tokenetics.core.tokenizer import count_text_tokens, count_tokens

_WEATHER_TOOL = {
    "name": "get_weather",
    "description": "Get the current weather for a city",
    "input_schema": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}
_EMAIL_TOOL = {
    "name": "send_email",
    "description": "Send an email to a recipient with a subject and body",
    "input_schema": {
        "type": "object",
        "properties": {
            "to": {"type": "string"},
            "subject": {"type": "string"},
            "body": {"type": "string"},
        },
        "required": ["to", "subject", "body"],
    },
}
_EXTRACT_TOOL = {
    "name": "extract_order_info",
    "description": "Extract a customer's name, email, and order ID from a support message",
    "input_schema": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "email": {"type": "string"},
            "order_id": {"type": "string"},
        },
        "required": ["name", "email", "order_id"],
    },
}


@dataclass
class Scenario:
    description: str
    kwargs: dict[str, Any]
    # True for scenarios the real API would likely reject outright (e.g.
    # empty message content) -- always prepare()-only regardless of flags.
    prepare_only: bool = False
    # Per-stage config, e.g. {"context_scheduler": {"token_budget": 150}}.
    stage_config: dict[str, dict[str, Any]] | None = None


SCENARIOS: dict[str, Scenario] = {
    "dedup": Scenario(
        description="Exact-repeat conversation -- dedup should drop the repeated turns.",
        kwargs={
            "model": "claude-sonnet-5",
            "max_tokens": 400,
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
                {"role": "user", "content": "In one short paragraph, explain what a hash map is."},
            ],
        },
    ),
    "near_dup": Scenario(
        description=(
            "A near-duplicate Q&A pair (both turns differ from their earlier twin by only "
            "the last word, shingle-similarity ~96-100%) -- near_dup should merge both "
            "pairs, dropping the older user and assistant turn and leaving a valid "
            "alternating conversation behind. Exact dedup wouldn't catch this since the "
            "text isn't byte-identical."
        ),
        kwargs={
            "model": "claude-sonnet-5",
            "max_tokens": 300,
            "messages": [
                {
                    "role": "user",
                    "content": "Can you explain what a binary search tree is and how it "
                    "works in simple terms, including how insertion, deletion, and lookup "
                    "operations behave on average for a reasonably balanced tree structure "
                    "across most common real world usage patterns that developers actually "
                    "run into day to day at work today?",
                },
                {
                    "role": "assistant",
                    "content": "A binary search tree is a data structure where each node "
                    "has at most two children, and every left child in the subtree is "
                    "smaller than its parent while every right child in the subtree is "
                    "larger, which is what makes searching, inserting, and deleting values "
                    "relatively fast on average for a reasonably balanced tree across most "
                    "practical real world workloads that show up in typical software "
                    "engineering interview settings.",
                },
                {
                    "role": "user",
                    "content": "Can you explain what a binary search tree is and how it "
                    "works in simple terms, including how insertion, deletion, and lookup "
                    "operations behave on average for a reasonably balanced tree structure "
                    "across most common real world usage patterns that developers actually "
                    "run into day to day at work now?",
                },
                {
                    "role": "assistant",
                    "content": "A binary search tree is a data structure where each node "
                    "has at most two children, and every left child in the subtree is "
                    "smaller than its parent while every right child in the subtree is "
                    "larger, which is what makes searching, inserting, and deleting values "
                    "relatively fast on average for a reasonably balanced tree across most "
                    "practical real world workloads that show up in typical software "
                    "engineering interview contexts.",
                },
                {"role": "user", "content": "Also, what's a good use case for one?"},
            ],
        },
    ),
    "extraction": Scenario(
        description=(
            "Extraction-shaped question + a matching tool -- structured_output should "
            "force tool_choice, brevity_injector should inject the MODERATE instruction, "
            "adaptive_budget should pick low thinking effort."
        ),
        kwargs={
            "model": "claude-sonnet-5",
            "max_tokens": 100,
            "messages": [
                {
                    "role": "user",
                    "content": "Extract the customer name, email, and order ID from this "
                    "message: Hi, this is Jane Smith, my email is jane@example.com and my "
                    "order ID is A12345.",
                }
            ],
            "tools": [_EXTRACT_TOOL],
        },
    ),
    "code": Scenario(
        description=(
            "Code-heavy question with a small max_tokens cap -- task_classifier should "
            "say 'code', adaptive_budget should widen max_tokens and pick high thinking "
            "effort."
        ),
        kwargs={
            "model": "claude-sonnet-5",
            "max_tokens": 60,
            "messages": [
                {
                    "role": "user",
                    "content": "```python\ndef fib(n):\n    if n <= 1:\n        return n\n"
                    "    return fib(n-1) + fib(n-2)\n```\nHow can I make this more efficient?",
                }
            ],
        },
    ),
    "tool_heavy": Scenario(
        description=(
            "A conversation with a real tool_use/tool_result exchange already in it -- "
            "task_classifier should say 'tool-heavy' via the structural scan, not text."
        ),
        kwargs={
            "model": "claude-sonnet-5",
            "max_tokens": 150,
            "messages": [
                {"role": "user", "content": "What's the weather in Boston?"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "1",
                            "name": "get_weather",
                            "input": {"city": "Boston"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "1", "content": "58F, cloudy"}
                    ],
                },
            ],
            "tools": [_WEATHER_TOOL],
        },
    ),
    "irrelevant_tool": Scenario(
        description=(
            "One relevant tool + one clearly irrelevant tool registered -- "
            "schema_minification's relevance filter should drop only the irrelevant one."
        ),
        kwargs={
            "model": "claude-sonnet-5",
            "max_tokens": 150,
            "messages": [{"role": "user", "content": "What's the weather like today in Paris?"}],
            "tools": [_WEATHER_TOOL, _EMAIL_TOOL],
        },
    ),
    # No "unclassified" scenario here: task_classifier only ever returns
    # task_type=None for empty/whitespace-only text, and the real API (even
    # its free count_tokens endpoint, which Tokenetics.prepare() calls
    # before any stage runs) rejects that content outright -- "text content
    # blocks must contain non-whitespace text". There's no string that can
    # satisfy the API's validation and still trigger the unclassified path,
    # so this corner case can't be demonstrated against the real API at
    # all, free or billed. It's covered directly instead, bypassing
    # count_tokens/the API entirely, by
    # tests/test_task_classifier.py::test_empty_message_is_unclassified_not_guessed.
    "caller_sets_thinking": Scenario(
        description=(
            "Corner case: a code-shaped request where the caller already set their own "
            "`thinking` config -- adaptive_budget should respect it and not override, "
            "even though the task type would normally get an effort level set."
        ),
        kwargs={
            "model": "claude-sonnet-5",
            "max_tokens": 200,
            "messages": [
                {
                    "role": "user",
                    "content": "```python\ndef is_prime(n):\n    return n > 1 and all("
                    "n % i for i in range(2, n))\n```\nExplain why this works.",
                }
            ],
            "thinking": {"type": "adaptive", "display": "summarized"},
        },
    ),
    "long_history": Scenario(
        description=(
            "A long conversation with a mix of turn types under a tight token_budget -- "
            "context_scheduler should pin the decision and the still-unresolved error, "
            "drop low-value small-talk padding to fit the budget, and NOT pin the first "
            "error (it's resolved by an explicit user signal a couple turns later). "
            "Under this particular tight budget the two pinned turns (both role='user') "
            "end up as the sole survivors and get losslessly merged into one message by "
            "the stage's own role-alternation safety net -- see "
            "src/tokenetics/stages/context_scheduler.py's _fix_alternation for the general "
            "fix (merge adjacent same-role survivors; drop a lone leading non-user turn "
            "only if merging still isn't enough)."
        ),
        kwargs={
            "model": "claude-sonnet-5",
            "max_tokens": 150,
            "messages": [
                {"role": "user", "content": "Hey, quick question before we start."},
                {"role": "assistant", "content": "Sure, go ahead."},
                {
                    "role": "user",
                    "content": "Traceback (most recent call last):\n  File \"app.py\", "
                    "line 12, in handle\n    user = payload['user_id']\nKeyError: 'user_id'",
                },
                {
                    "role": "assistant",
                    "content": "```python\nif 'user_id' in payload:\n    handle(payload)\n"
                    "```\nTry this guard clause before accessing the key.",
                },
                {
                    "role": "user",
                    "content": "That fixed it, thanks! Now, let's go with the async queue "
                    "approach for the retry logic.",
                },
                {"role": "assistant", "content": "Got it, I'll set that up now."},
                {
                    "role": "user",
                    "content": "Also seeing this one:\nTraceback (most recent call last):\n"
                    "  File \"worker.py\", line 40, in poll\n    resp = conn.get(url)\n"
                    "TimeoutError: connection timed out",
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "1",
                            "name": "check_status",
                            "input": {"service": "worker"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "1", "content": "degraded"}
                    ],
                },
                {"role": "assistant", "content": "Status looks degraded on our end."},
                {"role": "user", "content": "Can you summarize everything we've discussed so far?"},
            ],
        },
        stage_config={"context_scheduler": {"token_budget": 150}},
    ),
}


def _run_prepare(
    name: str, scenario: Scenario, client: anthropic.Anthropic, disabled: list[str]
) -> tuple[int, int, dict[str, int]]:
    tk = Tokenetics(client=client, stage_config=scenario.stage_config)
    for stage in tk.stages:
        if stage.name in disabled:
            stage.enabled = False

    before_tokens = count_tokens(from_api_kwargs(**scenario.kwargs), client)
    prepared = tk.prepare(**scenario.kwargs)
    after_tokens = count_tokens(from_api_kwargs(**prepared), client)

    print(f"\n=== {name} === {scenario.description}")
    fired = [s.name for s in tk.stages if s.enabled]
    print(f"stages fired: {fired or '(none -- all disabled)'}")
    print(f"request tokens before -> after: {before_tokens} -> {after_tokens}")
    extras = {
        k: v
        for k, v in prepared.items()
        if k in ("tool_choice", "thinking", "output_config", "system", "max_tokens")
    }
    if extras:
        print(f"notable prepared fields: {extras}")
    _print_log(tk)
    deltas = _print_summary(before_tokens, after_tokens, tk)
    return before_tokens, after_tokens, deltas


def _print_log(tk: Tokenetics) -> None:
    entries = getattr(tk.logger, "entries", None)
    if not entries:
        return
    print("per-stage log:")
    for entry in entries:
        flag = " [ERROR]" if entry.extra.get("error") else ""
        timing = entry.extra.get("timing_seconds", 0.0)
        notes = {k: v for k, v in entry.extra.items() if k not in ("error", "timing_seconds")}
        notes_str = f", {notes}" if notes else ""
        print(
            f"  {entry.stage_name}: {entry.tokens_before} -> {entry.tokens_after} tokens, "
            f"{timing:.4f}s, enabled={entry.enabled}{flag}{notes_str}"
        )


# Stages that never change the *request's* token count by design, even when
# they did real, useful work -- their savings (or costs) show up on the
# generation/output side instead, which needs a real completion (and a
# without-pipeline baseline to compare against) to actually measure. Rather
# than print a misleading "0 tokens saved" with no context, the summary
# below explains why each one is zero on the input side.
_ZERO_DELTA_NOTE = {
    "task_classifier": "annotates the request only, no token-count impact by design",
    "structured_output": "forces tool_choice, no token-count impact by design",
    "adaptive_budget": (
        "sets the max_tokens cap / thinking effort -- affects OUTPUT/generation-time "
        "cost, not visible in request token counts (needs a real completion to measure)"
    ),
}


def _print_summary(before_tokens: int, after_tokens: int, tk: Tokenetics) -> dict[str, int]:
    """Prints a request-side (input) token-savings summary, all measured for
    free from count_tokens -- no completion required. Returns per-stage
    deltas so callers (e.g. --all-scenarios) can aggregate across runs.
    """
    print("--- token savings summary (request/input side, measured) ---")
    saved = before_tokens - after_tokens
    pct = (saved / before_tokens * 100) if before_tokens else 0.0
    print(f"overall: {before_tokens} -> {after_tokens} tokens (saved {saved}, {pct:.1f}%)")

    entries = getattr(tk.logger, "entries", None) or []
    deltas: dict[str, int] = {}
    print("per stage:")
    for entry in entries:
        delta = (entry.tokens_before or 0) - (entry.tokens_after or 0)
        deltas[entry.stage_name] = deltas.get(entry.stage_name, 0) + delta
        if delta > 0:
            stage_pct = (delta / before_tokens * 100) if before_tokens else 0.0
            note = f"saved {delta} tokens ({stage_pct:.1f}% of original)"
        elif delta < 0:
            note = f"cost {-delta} tokens (adds content, e.g. a brevity instruction)"
        else:
            reason = _ZERO_DELTA_NOTE.get(entry.stage_name)
            note = f"0 tokens{f' -- {reason}' if reason else ''}"
        print(f"  {entry.stage_name}: {note}")
    return deltas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--scenario",
        choices=sorted(SCENARIOS),
        default="dedup",
        help="Which scenario to run (default: dedup).",
    )
    parser.add_argument(
        "--list-scenarios",
        action="store_true",
        help="Print all available scenarios and exit -- no API calls at all.",
    )
    parser.add_argument(
        "--all-scenarios",
        action="store_true",
        help="Run every scenario through prepare() only (free -- no completions) and print each result.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="For --scenario: skip the billed messages.create() call, just show prepare() output.",
    )
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
            "thinking-token spend from response.usage. Costs more than the default run."
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        metavar="MODEL_ID",
        help="Override the scenario's model, e.g. claude-haiku-4-5 for the cheapest smoke test.",
    )
    args = parser.parse_args()

    if args.list_scenarios:
        for name, scenario in sorted(SCENARIOS.items()):
            print(f"{name}: {scenario.description}")
        return

    client = anthropic.Anthropic()

    if args.all_scenarios:
        total_before = 0
        total_after = 0
        total_deltas: dict[str, int] = {}
        for name, scenario in sorted(SCENARIOS.items()):
            before_tokens, after_tokens, deltas = _run_prepare(name, scenario, client, args.disable)
            total_before += before_tokens
            total_after += after_tokens
            for stage_name, delta in deltas.items():
                total_deltas[stage_name] = total_deltas.get(stage_name, 0) + delta

        print("\n=== TOTAL across all scenarios ===")
        total_saved = total_before - total_after
        total_pct = (total_saved / total_before * 100) if total_before else 0.0
        print(
            f"overall: {total_before} -> {total_after} tokens "
            f"(saved {total_saved}, {total_pct:.1f}%)"
        )
        print("per stage (summed across scenarios):")
        for stage_name, delta in total_deltas.items():
            if delta > 0:
                note = f"saved {delta} tokens total"
            elif delta < 0:
                note = f"cost {-delta} tokens total"
            else:
                reason = _ZERO_DELTA_NOTE.get(stage_name)
                note = f"0 tokens{f' -- {reason}' if reason else ''}"
            print(f"  {stage_name}: {note}")
        return

    scenario = SCENARIOS[args.scenario]
    prepare_only = args.prepare_only or scenario.prepare_only
    if scenario.prepare_only and not args.prepare_only:
        print(
            f"note: scenario {args.scenario!r} is always prepare()-only "
            "(a real completion would likely be rejected by the API) -- skipping messages.create()."
        )

    if prepare_only:
        _run_prepare(args.scenario, scenario, client, args.disable)
        return

    # default pipeline: dedup, near_dup, task_classifier, schema_minification,
    # context_scheduler, structured_output, brevity_injector, adaptive_budget
    tk = Tokenetics(client=client, stage_config=scenario.stage_config)
    for stage in tk.stages:
        if stage.name in args.disable:
            stage.enabled = False

    request_kwargs = dict(scenario.kwargs)
    if args.model:
        request_kwargs["model"] = args.model

    if args.show_thinking and "thinking" not in request_kwargs:
        # display="summarized" populates the thinking block's text so we can
        # measure it. Setting this here means adaptive_budget (stage 9c)
        # will leave it alone -- it only sets thinking config when the
        # caller hasn't already.
        request_kwargs["thinking"] = {"type": "adaptive", "display": "summarized"}

    before_tokens = count_tokens(from_api_kwargs(**request_kwargs), client)
    prepared = tk.prepare(**request_kwargs)
    after_tokens = count_tokens(from_api_kwargs(**prepared), client)

    fired = [s.name for s in tk.stages if s.enabled]
    print(f"scenario: {args.scenario} -- {scenario.description}")
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

    _print_log(tk)
    _print_summary(before_tokens, after_tokens, tk)


if __name__ == "__main__":
    main()
