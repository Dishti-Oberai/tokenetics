#!/usr/bin/env python3
"""Manual dev-loop script for eyeballing Tokenetics against the real Anthropic API.

NOT part of pytest or CI. Run it by hand:

    uv run python scripts/dev_demo.py --list-scenarios
    uv run python scripts/dev_demo.py --all-scenarios          # free -- no completions
    uv run python scripts/dev_demo.py --scenario extraction
    uv run python scripts/dev_demo.py --scenario code --disable adaptive_budget
    uv run python scripts/dev_demo.py --scenario dedup --model claude-haiku-4-5
    uv run python scripts/dev_demo.py --scenario code --tale  # Tier 2c TALE estimate vs. the free heuristic
    uv run python scripts/dev_demo.py --scenario code --log-to costs.jsonl  # feed the Phase 11 dashboard

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
import json
from dataclasses import dataclass
from typing import Any

import anthropic

from tokenetics import Tokenetics
from tokenetics.core.cache_hash import stable_prefix_hash
from tokenetics.core.cache_pricing import (
    READ_MULTIPLIER,
    WRITE_MULTIPLIER_1H,
    WRITE_MULTIPLIER_5M,
    base_input_price_for,
)
from tokenetics.core.errors import CacheSafetyError
from tokenetics.core.logger import FileCostLogger
from tokenetics.core.request import ToolSpec, from_api_kwargs
from tokenetics.core.tokenizer import count_text_tokens, count_tokens
from tokenetics.extras.compress import compress_text
from tokenetics.extras.semantic_cache import SemanticCache
from tokenetics.extras.tale import estimate_token_budget

_WEATHER_TOOL: dict[str, Any] = {
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

_CACHE_HIT_SYSTEM = (
    # >=1024 tokens: claude-sonnet-5's real minimum cacheable prefix length
    # (core.cache_pricing.MIN_CACHEABLE_TOKENS) -- below that, the real API
    # silently skips caching (no error, cache_creation/cache_read stay 0),
    # which CacheBreakpointOptimizerStage now checks for before placing a
    # breakpoint. A short system prompt here would make --measure-usage
    # show a "breakpoint placed" that the real API just ignores.
    "You are a customer support agent for a software company. Be concise and "
    "helpful, and always confirm the customer's account details before making "
    "any changes. "
) * 35
_CACHE_HIT_PREFIX_HASH = stable_prefix_hash(
    _CACHE_HIT_SYSTEM,
    [ToolSpec(name=_WEATHER_TOOL["name"], description=_WEATHER_TOOL["description"], input_schema=_WEATHER_TOOL["input_schema"])],
)

_DELTA_PREVIOUS_STATUS = json.dumps(
    {
        "order_id": "ORD-12345",
        "customer": "Jane Smith",
        "status": "pending",
        "history": [{"ts": i, "event": "checked"} for i in range(30)],
    }
)
_DELTA_NEW_STATUS = _DELTA_PREVIOUS_STATUS.replace('"pending"', '"shipped"')


@dataclass
class Scenario:
    description: str
    kwargs: dict[str, Any]
    # True for scenarios the real API would likely reject outright (e.g.
    # empty message content) -- always prepare()-only regardless of flags.
    prepare_only: bool = False
    # Per-stage config, e.g. {"context_scheduler": {"token_budget": 150}}.
    stage_config: dict[str, dict[str, Any]] | None = None
    # True for scenarios built specifically to trip the cache-safety guard's
    # hard-raise -- handled specially, since CacheSafetyError is the one
    # exception the pipeline deliberately does NOT fail open on.
    expect_cache_safety_error: bool = False


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
            "say 'code', adaptive_budget should widen max_tokens. Thinking effort was opt-in "
            "only from 2026-09-04 (see adaptive_budget.py's docstring for why -- a real "
            "$-cost benchmark showed the old automatic default made 'code'/'conversational' "
            "requests 2-3x more expensive) until 2026-09-06, when a bounded-shape default was "
            "added: this prompt's meta.bounded_shape is True (short, single question), so "
            "thinking effort now fires here WITHOUT opting in, at the downgraded 'medium' "
            "level (task_type 'code' -> 'high', downgraded one level for being bounded) -- "
            "see 'caller_sets_thinking' for the caller-sets-their-own-config corner case."
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
    "delta_hit": Scenario(
        description=(
            "A tool_result whose content is mostly unchanged from a caller-supplied "
            "previous version (only the 'status' field flipped, everything else -- "
            "customer, order id, 30-entry history -- identical) -- delta_compression "
            "should replace it with a much smaller delta-encoded payload instead of "
            "resending the whole JSON blob. This is the mechanism, not a claim about "
            "Claude's reply quality -- see the stage's docstring on the deferred "
            "quality-interpretation question."
        ),
        kwargs={
            "model": "claude-sonnet-5",
            "max_tokens": 150,
            "messages": [
                {"role": "user", "content": "What's the status of order ORD-12345?"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_delta_1",
                            "name": "check_order_status",
                            "input": {"order_id": "ORD-12345"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_delta_1",
                            "content": _DELTA_NEW_STATUS,
                        }
                    ],
                },
            ],
        },
        stage_config={
            "delta_compression": {"previous_payloads": {"toolu_delta_1": _DELTA_PREVIOUS_STATUS}}
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
    "aggressive_brevity": Scenario(
        description=(
            "Opt-in-only AGGRESSIVE brevity (stage_config brevity_injector."
            "aggressive_for_task_types=['code']), added 2026-09-06 after a real, close-to-neutral "
            "output-token benchmark prompted trying it -- never auto-selected by default (see "
            "brevity_injector.py's docstring: the classifier's 'code' category is too broad to "
            "trust generally, same reasoning as adaptive_budget's thinking-effort opt-in). A "
            "genuinely simple, single-fact code question with a real code fence (so "
            "task_classifier actually lands it in 'code', not 'conversational' -- a prose-only "
            "version was tried first and silently never engaged the mechanism, the same "
            "classifier-bucket lesson quality_adaptive_budget_002 already taught this project). "
            "See quality_brevity_aggressive_001 (benchmarks/quality_checks/brevity_injector.json) "
            "for the held-out quality-check counterpart before trusting this for anything broader."
        ),
        kwargs={
            "model": "claude-sonnet-5",
            "max_tokens": 150,
            "messages": [
                {
                    "role": "user",
                    "content": "Given this Python code:\n```python\na = [1, 2]\nb = [1, 2]\n"
                    "print(a == b, a is b)\n```\nWhat's the difference between `==` and `is` here?",
                }
            ],
        },
        stage_config={"brevity_injector": {"aggressive_for_task_types": ["code"]}},
    ),
    "aggressive_brevity_hard_bugfix": Scenario(
        description=(
            "Real output-token A/B for the HARDER end of AGGRESSIVE brevity -- same content as "
            "quality_brevity_aggressive_004 (a real bug-fix with a genuine edge case: "
            "second_largest() on lists with fewer than 2 unique values), which already passed "
            "quality-check twice for real (correctness confirmed), but has never had its actual "
            "output-token effect measured. `aggressive_brevity`'s scenario (the simple ==/is "
            "case) showed 85.5% real output-token savings -- this checks whether that holds, or "
            "shrinks, once the answer genuinely needs more content (a diagnosis AND a working "
            "code fix, not just one sentence)."
        ),
        kwargs={
            "model": "claude-sonnet-5",
            "max_tokens": 200,
            "messages": [
                {
                    "role": "user",
                    "content": "This function is supposed to return the second-largest unique "
                    "value in a list, but it's buggy:\n```python\ndef second_largest(nums):\n"
                    "    unique = list(set(nums))\n    unique.sort()\n    return unique[-2]\n"
                    "```\nWhat's wrong with it and how do you fix it?",
                }
            ],
        },
        stage_config={"brevity_injector": {"aggressive_for_task_types": ["code"]}},
    ),
    "aggressive_brevity_hard_complexity": Scenario(
        description=(
            "Real output-token A/B for the HARDER end of AGGRESSIVE brevity -- same content as "
            "quality_brevity_aggressive_005 (a genuine two-part complexity/design question: why "
            "naive recursive fib() is exponential-time AND how to fix it), which already passed "
            "quality-check twice for real. Same purpose as aggressive_brevity_hard_bugfix: checks "
            "whether the 85.5% real output-token savings seen on the simple ==/is case holds up "
            "on a question that genuinely needs a two-part, more substantial answer."
        ),
        kwargs={
            "model": "claude-sonnet-5",
            "max_tokens": 200,
            "messages": [
                {
                    "role": "user",
                    "content": "```python\ndef fib(n):\n    if n <= 1:\n        return n\n"
                    "    return fib(n-1) + fib(n-2)\n```\nWhy is this slow for large n, "
                    "and how would you make it faster?",
                }
            ],
        },
        stage_config={"brevity_injector": {"aggressive_for_task_types": ["code"]}},
    ),
    "aggressive_brevity_conversational_simple": Scenario(
        description=(
            "Extending AGGRESSIVE brevity validation to a new task type ('conversational'), per "
            "the user asking to test + opt-in more task types rather than default it on for all "
            "of 'code'. Real output-token A/B for a genuinely simple single-fact question -- same "
            "content as quality_brevity_aggressive_006. No code fence needed: 'conversational' is "
            "task_classifier's weak catch-all default, so a plain question classifies correctly."
        ),
        kwargs={
            "model": "claude-sonnet-5",
            "max_tokens": 150,
            "messages": [{"role": "user", "content": "Why is the sky blue?"}],
        },
        stage_config={"brevity_injector": {"aggressive_for_task_types": ["conversational"]}},
    ),
    "aggressive_brevity_conversational_hard": Scenario(
        description=(
            "Harder conversational AGGRESSIVE sample -- real output-token A/B for a genuine "
            "two-sided distinction (weather vs. climate), same content as "
            "quality_brevity_aggressive_007. Tests whether the output-token savings pattern seen "
            "on 'code' (larger absolute savings on harder questions, since baseline verbosity "
            "grows faster than the terse answer does) holds on an entirely different task type."
        ),
        kwargs={
            "model": "claude-sonnet-5",
            "max_tokens": 150,
            "messages": [
                {
                    "role": "user",
                    "content": "What's the actual difference between weather and climate? "
                    "People use the terms interchangeably but I don't think they mean the same thing.",
                }
            ],
        },
        stage_config={"brevity_injector": {"aggressive_for_task_types": ["conversational"]}},
    ),
    "aggressive_brevity_tool_heavy_simple": Scenario(
        description=(
            "Extending AGGRESSIVE brevity validation to a third task type ('tool-heavy'), the "
            "last major classifier category untested. Real output-token A/B for a genuinely "
            "simple follow-up question after a real tool_use/tool_result exchange (a stock-price "
            "lookup) -- same content as quality_brevity_aggressive_008. Clean numbers ($165 -> "
            "$181.50) so the answer ('higher') is unambiguous."
        ),
        kwargs={
            "model": "claude-sonnet-5",
            "max_tokens": 150,
            "messages": [
                {"role": "user", "content": "What's the AAPL stock price?"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_01",
                            "name": "get_stock_price",
                            "input": {"ticker": "AAPL"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_01",
                            "content": "AAPL: $181.50, up 1.2% today",
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": "AAPL is currently trading at $181.50, up 1.2% today.",
                },
                {
                    "role": "user",
                    "content": "Is that higher or lower than it was trading at exactly a "
                    "month ago, which was $165?",
                },
            ],
            "tools": [
                {
                    "name": "get_stock_price",
                    "description": "Get the current stock price for a given ticker symbol.",
                    "input_schema": {"type": "object", "properties": {"ticker": {"type": "string"}}},
                }
            ],
        },
        stage_config={"brevity_injector": {"aggressive_for_task_types": ["tool-heavy"]}},
    ),
    "aggressive_brevity_tool_heavy_hard": Scenario(
        description=(
            "Harder tool-heavy AGGRESSIVE sample -- real output-token A/B for a follow-up "
            "requiring an actual calculation (percentage change) on the tool result, not just a "
            "directional comparison -- same content as quality_brevity_aggressive_009. $165 -> "
            "$181.50 is exactly +10%, chosen to avoid rounding ambiguity."
        ),
        kwargs={
            "model": "claude-sonnet-5",
            "max_tokens": 150,
            "messages": [
                {"role": "user", "content": "What's the AAPL stock price?"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_01",
                            "name": "get_stock_price",
                            "input": {"ticker": "AAPL"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_01",
                            "content": "AAPL: $181.50, up 1.2% today",
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": "AAPL is currently trading at $181.50, up 1.2% today.",
                },
                {
                    "role": "user",
                    "content": "By what percentage has it increased since it was trading at "
                    "$165 a month ago?",
                },
            ],
            "tools": [
                {
                    "name": "get_stock_price",
                    "description": "Get the current stock price for a given ticker symbol.",
                    "input_schema": {"type": "object", "properties": {"ticker": {"type": "string"}}},
                }
            ],
        },
        stage_config={"brevity_injector": {"aggressive_for_task_types": ["tool-heavy"]}},
    ),
    "aggressive_brevity_extraction_simple": Scenario(
        description=(
            "Extending AGGRESSIVE brevity validation to the last untested task type "
            "('extraction' currently only ever gets MODERATE by default). Real output-token A/B "
            "for a simple 2-field extraction -- same content as quality_brevity_aggressive_010. "
            "A registered tool is included (realistic shape) so structured_output also forces "
            "tool_choice -- this measures whether AGGRESSIVE adds anything on top of that "
            "already-compact forced JSON reply, or whether the two are largely redundant here."
        ),
        kwargs={
            "model": "claude-sonnet-5",
            "max_tokens": 150,
            "messages": [
                {
                    "role": "user",
                    "content": "Extract the customer's name and email from this message: Hi, "
                    "this is Jane Smith, my email is jane@example.com.",
                }
            ],
            "tools": [
                {
                    "name": "extract_contact_info",
                    "description": "Extract a customer's name and email from a message",
                    "input_schema": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}, "email": {"type": "string"}},
                        "required": ["name", "email"],
                    },
                }
            ],
        },
        stage_config={"brevity_injector": {"aggressive_for_task_types": ["extraction"]}},
    ),
    "aggressive_brevity_extraction_hard": Scenario(
        description=(
            "Harder extraction AGGRESSIVE sample -- real output-token A/B for a multi-item "
            "extraction (two action items, each with owner and due date) -- same content as "
            "quality_brevity_aggressive_011."
        ),
        kwargs={
            "model": "claude-sonnet-5",
            "max_tokens": 200,
            "messages": [
                {
                    "role": "user",
                    "content": "Extract all action items from this meeting note, including WHO "
                    "is responsible and WHEN it's due: \"Sarah will finalize the budget by "
                    "Friday. Also, someone needs to email the vendor about the contract renewal "
                    "-- Tom said he'd handle that by end of next week.\"",
                }
            ],
            "tools": [
                {
                    "name": "extract_action_items",
                    "description": "Extract action items with owner and due date from meeting notes",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "items": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "task": {"type": "string"},
                                        "owner": {"type": "string"},
                                        "due": {"type": "string"},
                                    },
                                    "required": ["task", "owner", "due"],
                                },
                            }
                        },
                        "required": ["items"],
                    },
                }
            ],
        },
        stage_config={"brevity_injector": {"aggressive_for_task_types": ["extraction"]}},
    ),
    "caller_sets_thinking": Scenario(
        description=(
            "Corner case: a code-shaped request where the caller already set their own "
            "`thinking` config AND opted into adaptive_budget's thinking-effort sub-stage "
            "(`enable_thinking_effort=True`, opt-in only since 2026-09-04 -- see "
            "adaptive_budget.py's docstring) -- adaptive_budget should respect the caller's "
            "own config and not override it, even though it's opted in and the task type "
            "would normally get an effort level set."
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
        stage_config={"adaptive_budget": {"enable_thinking_effort": True}},
    ),
    "thinking_effort_code": Scenario(
        description=(
            "Isolates adaptive_budget's OWN thinking-effort choice for real measurement -- "
            "added 2026-09-06 after a real dashboard run showed 'caller_sets_thinking' never "
            "actually exercises this stage's own effort selection at all (that scenario's "
            "whole point is proving adaptive_budget backs off when the caller already set "
            "`thinking`, so its 91 real thinking tokens come entirely from the CALLER's config, "
            "not the stage). No caller-set `thinking` here, so a task_type of 'code' genuinely "
            "drives adaptive_budget's own `_EFFORT_BY_TASK_TYPE['code'] = 'high'` choice on the "
            "optimized side -- now downgraded to 'medium' (2026-09-06), since this prompt's "
            "meta.bounded_shape is True (short, single question) -- see adaptive_budget.py's "
            "bounded-shape downgrade. Run with `--disable brevity_injector` to keep AGGRESSIVE "
            "brevity's own token effect (now a default for bounded 'code' requests, see "
            "brevity_injector.py) from confounding this measurement -- the goal is ONLY the "
            "effort knob's real cost, isolated from every other stage."
        ),
        kwargs={
            "model": "claude-sonnet-5",
            "max_tokens": 400,
            "messages": [
                {
                    "role": "user",
                    "content": "```python\ndef merge_sorted(a, b):\n    result = []\n"
                    "    while a and b:\n        result.append(a.pop(0) if a[0] <= b[0] else b.pop(0))\n"
                    "    return result + a + b\n```\nWhat's inefficient about this merge, and how "
                    "would you fix it?",
                }
            ],
        },
        stage_config={"adaptive_budget": {"enable_thinking_effort": True}},
    ),
    "thinking_effort_conversational": Scenario(
        description=(
            "Same purpose as thinking_effort_code, but for the OTHER real downgrade transition "
            "(2026-09-06): task_type 'conversational' -> 'medium' effort by default, downgraded "
            "to 'low' since this prompt's meta.bounded_shape is True. The classic bat-and-ball "
            "cognitive-reflection-test question (same content as quality_adaptive_budget_004, "
            "which already validated quality holds at 'low' effort) -- this scenario measures "
            "the real DOLLAR side of that same downgrade. No code fence/tool involved, so no "
            "--disable flag needed beyond the default pipeline."
        ),
        kwargs={
            "model": "claude-sonnet-5",
            "max_tokens": 300,
            "messages": [
                {
                    "role": "user",
                    "content": "A bat and a ball cost $1.10 together. The bat costs $1.00 more "
                    "than the ball. How much does the ball cost?",
                }
            ],
        },
        stage_config={"adaptive_budget": {"enable_thinking_effort": True}},
    ),
    "thinking_effort_tool_heavy": Scenario(
        description=(
            "Real dollar evidence (2026-09-06) for the class where bounded_shape does NOT "
            "change the effort level -- 'tool-heavy' already maps to 'low' by default, so this "
            "measures whether enabling thinking at all (even at its floor) has any hidden real "
            "cost, not the downgrade itself. Same content as quality_adaptive_budget_005 (already "
            "quality-validated). Run with --disable brevity_injector to isolate this from "
            "AGGRESSIVE brevity's own token effect (also a default for bounded 'tool-heavy' now)."
        ),
        kwargs={
            "model": "claude-sonnet-5",
            "max_tokens": 150,
            "messages": [
                {"role": "user", "content": "What's the AAPL stock price?"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "toolu_01", "name": "get_stock_price", "input": {"ticker": "AAPL"}}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "toolu_01", "content": "AAPL: $181.50, up 1.2% today"}
                    ],
                },
                {"role": "assistant", "content": "AAPL is currently trading at $181.50, up 1.2% today."},
                {
                    "role": "user",
                    "content": "Is that higher or lower than it was trading at exactly a month ago, which was $165?",
                },
            ],
            "tools": [
                {
                    "name": "get_stock_price",
                    "description": "Get the current stock price for a given ticker symbol.",
                    "input_schema": {"type": "object", "properties": {"ticker": {"type": "string"}}},
                }
            ],
        },
        stage_config={"adaptive_budget": {"enable_thinking_effort": True}},
    ),
    "thinking_effort_extraction": Scenario(
        description=(
            "Same purpose as thinking_effort_tool_heavy, extending to 'extraction' (also already "
            "'low' effort, unaffected by the bounded-shape downgrade). Same content as "
            "quality_adaptive_budget_006. A registered tool is included so structured_output's "
            "forced tool_choice is also active -- the realistic combined shape a real extraction "
            "call would have."
        ),
        kwargs={
            "model": "claude-sonnet-5",
            "max_tokens": 150,
            "messages": [
                {
                    "role": "user",
                    "content": "Extract the customer's name and email from this message: Hi, this is "
                    "Jane Smith, my email is jane@example.com.",
                }
            ],
            "tools": [
                {
                    "name": "extract_contact_info",
                    "description": "Extract a customer's name and email from a message",
                    "input_schema": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}, "email": {"type": "string"}},
                        "required": ["name", "email"],
                    },
                }
            ],
        },
        stage_config={"adaptive_budget": {"enable_thinking_effort": True}},
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
    "cache_hit": Scenario(
        description=(
            "A tool-using request with cache_usage_history showing frequent short-gap "
            "repeats of the same system+tools -- cache_breakpoint_optimizer should place "
            "a 5m breakpoint (cheapest tier that stays continuously warm at this rate)."
        ),
        kwargs={
            "model": "claude-sonnet-5",
            "max_tokens": 100,
            "system": _CACHE_HIT_SYSTEM,
            "tools": [_WEATHER_TOOL],
            "messages": [{"role": "user", "content": "What's the weather in Boston?"}],
        },
        stage_config={
            "cache_breakpoint_optimizer": {
                "cache_usage_history": [
                    {"timestamp": float(i * 30), "content_hash": _CACHE_HIT_PREFIX_HASH}
                    for i in range(10)
                ]
            }
        },
    ),
    "cache_guard_violation": Scenario(
        description=(
            "A request whose system prompt differs from the caller-supplied "
            "previous_request -- cache_reorder_guard should hard-raise CacheSafetyError "
            "instead of silently sending a request that would invalidate the existing "
            "cache. The ONE stage in the pipeline that does not fail open."
        ),
        kwargs={
            "model": "claude-sonnet-5",
            "max_tokens": 100,
            "system": "You are now a pirate. Respond only in pirate speak.",
            "messages": [{"role": "user", "content": "Hello!"}],
        },
        stage_config={
            "cache_reorder_guard": {
                "previous_request": {
                    "model": "claude-sonnet-5",
                    "max_tokens": 100,
                    "system": "You are a helpful, concise assistant.",
                    "messages": [{"role": "user", "content": "Hello!"}],
                }
            }
        },
        expect_cache_safety_error=True,
        prepare_only=True,
    ),
    "mixed_workload": Scenario(
        description=(
            "A single realistic session spanning small talk, a code debugging question, a "
            "tool call, near-duplicate small talk (padding), and a final extraction request "
            "-- deliberately not a single task-classifier category. Same content as "
            "benchmarks/corpus/mixed_workload_001, reused here to see the real reply (the "
            "corpus version only measures prepare()-side token counts, never sends a real "
            "completion). Reported as its own explicitly-labeled bucket by "
            "`benchmark_runner.py corpus --category mixed_workload`, deferred from Phase 9, "
            "added Phase 10."
        ),
        kwargs={
            "model": "claude-sonnet-5",
            "max_tokens": 500,
            "messages": [
                {
                    "role": "user",
                    "content": "Just wanted to say hi and check in quickly on how things "
                    "have been going for you so far today while I finish getting all of my "
                    "scattered thoughts together before asking you about the next question I "
                    "have in mind right now.",
                },
                {
                    "role": "assistant",
                    "content": "Thanks so much for checking in things have been going well "
                    "on my end so far today and I have kept careful track of everything we "
                    "already covered earlier so I am ready whenever you want to move on to "
                    "asking that next question right now.",
                },
                {
                    "role": "user",
                    "content": "I've got a bug in this Python function:\n```python\ndef "
                    "average(nums):\n    return sum(nums) / len(nums)\n```\nIt throws a "
                    "ZeroDivisionError on an empty list. How should I fix it?",
                },
                {
                    "role": "assistant",
                    "content": "Add a guard for the empty-list case before dividing, e.g. "
                    "`return sum(nums) / len(nums) if nums else 0.0` -- or raise a clearer "
                    "error if an empty list genuinely shouldn't be allowed for your use case.",
                },
                {
                    "role": "user",
                    "content": "Good point. Separately -- what's the weather like in Austin "
                    "right now?",
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "1",
                            "name": "get_weather",
                            "input": {"city": "Austin"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "1", "content": "91F, sunny"}
                    ],
                },
                {
                    "role": "assistant",
                    "content": "It's 91F and sunny in Austin right now.",
                },
                {
                    "role": "user",
                    "content": "Just wanted to say hi and check in quickly on how things "
                    "have been going for you so far today while I finish getting all of my "
                    "scattered thoughts together before asking you about the next question I "
                    "have in mind right away.",
                },
                {
                    "role": "assistant",
                    "content": "Thanks so much for checking in things have been going well "
                    "on my end so far today and I have kept careful track of everything we "
                    "already covered earlier so I am ready whenever you want to move on to "
                    "asking that next question right away.",
                },
                {
                    "role": "user",
                    "content": "Last thing -- extract the customer name, order number, and "
                    'total as JSON from this: "Hi, this is Marcus Bell writing about order '
                    '#58204, total came to $142.75, and I haven\'t received it yet."',
                },
            ],
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Get the current weather for a city",
                    "input_schema": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                }
            ],
        },
    ),
    "mixed_workload_recap": Scenario(
        description=(
            "Same shape as `mixed_workload` (small talk, a genuine conversational question, "
            "a code fix, a tool call, near-duplicate padding, an extraction request) but the "
            "extraction request is now ANSWERED mid-conversation and the final turn is an "
            "open-ended recap request instead -- so the real reply has to actually engage "
            "with everything discussed, not just answer one narrow isolated task like "
            "`mixed_workload`'s ending does. Same content as benchmarks/corpus/"
            "mixed_workload_003."
        ),
        kwargs={
            "model": "claude-sonnet-5",
            "max_tokens": 600,
            "messages": [
                {
                    "role": "user",
                    "content": "Just wanted to say hi and check in quickly on how things "
                    "have been going for you so far today while I finish getting all of my "
                    "scattered thoughts together before asking you about the next question I "
                    "have in mind right now.",
                },
                {
                    "role": "assistant",
                    "content": "Thanks so much for checking in things have been going well "
                    "on my end so far today and I have kept careful track of everything we "
                    "already covered earlier so I am ready whenever you want to move on to "
                    "asking that next question right now.",
                },
                {
                    "role": "user",
                    "content": "It's a beautiful sunny day here -- any recommendations for a "
                    "good weekend activity?",
                },
                {
                    "role": "assistant",
                    "content": "A day like that is great for anything outdoors -- a hike, a "
                    "farmers market, or just reading in a park. If you want something more "
                    "active, biking or a pickup game of basketball works well too.",
                },
                {
                    "role": "user",
                    "content": "I've got a bug in this Python function:\n```python\ndef "
                    "average(nums):\n    return sum(nums) / len(nums)\n```\nIt throws a "
                    "ZeroDivisionError on an empty list. How should I fix it?",
                },
                {
                    "role": "assistant",
                    "content": "Add a guard for the empty-list case before dividing, e.g. "
                    "`return sum(nums) / len(nums) if nums else 0.0` -- or raise a clearer "
                    "error if an empty list genuinely shouldn't be allowed for your use case.",
                },
                {
                    "role": "user",
                    "content": "Good point. Separately -- what's the weather like in Austin "
                    "right now?",
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "1",
                            "name": "get_weather",
                            "input": {"city": "Austin"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "1", "content": "91F, sunny"}
                    ],
                },
                {
                    "role": "assistant",
                    "content": "It's 91F and sunny in Austin right now.",
                },
                {
                    "role": "user",
                    "content": "Just wanted to say hi and check in quickly on how things "
                    "have been going for you so far today while I finish getting all of my "
                    "scattered thoughts together before asking you about the next question I "
                    "have in mind right away.",
                },
                {
                    "role": "assistant",
                    "content": "Thanks so much for checking in things have been going well "
                    "on my end so far today and I have kept careful track of everything we "
                    "already covered earlier so I am ready whenever you want to move on to "
                    "asking that next question right away.",
                },
                {
                    "role": "user",
                    "content": "Last thing -- extract the customer name, order number, and "
                    'total as JSON from this: "Hi, this is Marcus Bell writing about order '
                    '#58204, total came to $142.75, and I haven\'t received it yet."',
                },
                {
                    "role": "assistant",
                    "content": '{"customer_name": "Marcus Bell", "order_number": "58204", '
                    '"total": 142.75}',
                },
                {
                    "role": "user",
                    "content": "Before we wrap up, can you give me a quick recap of "
                    "everything we've covered today -- the weekend suggestion, the bug fix, "
                    "the weather, and the order extraction?",
                },
            ],
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Get the current weather for a city",
                    "input_schema": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                }
            ],
        },
    ),
}


def _run_prepare(
    name: str,
    scenario: Scenario,
    client: anthropic.Anthropic,
    disabled: list[str],
    logger: FileCostLogger | None = None,
) -> tuple[int, int, dict[str, int], float]:
    """`logger`, if given, additionally persists this call's real per-stage
    entries to the dashboard's log file -- found missing 2026-09-06, when
    a user's `--all-scenarios --measure-usage --log-to` run showed real
    `generation_usage` events but an empty input-token stage table, since
    this function always used its own private InMemoryCostLogger (needed
    for the terminal's own "per-stage log" printout, via `_print_log`,
    which reads `.entries` -- a `FileCostLogger` has no such attribute, so
    it can't be swapped in directly without losing that output). Fixed by
    keeping the in-memory logger for printing, same as before, and
    replaying its entries into `logger` afterward -- the exact pattern
    `mcp_server.py`'s `_persist_if_configured` already established for the
    identical need.
    """
    tk = Tokenetics(client=client, stage_config=scenario.stage_config)
    for stage in tk.stages:
        if stage.name in disabled:
            stage.enabled = False

    if scenario.expect_cache_safety_error:
        print(f"\n=== {name} === {scenario.description}")
        try:
            tk.prepare(**scenario.kwargs)
        except CacheSafetyError as exc:
            print(f"CacheSafetyError raised as expected: {exc}")
        else:
            print("WARNING: expected CacheSafetyError but none was raised")
        _persist_stage_entries(tk, logger)
        return 0, 0, {}, 0.0

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
    deltas, cache_benefit = _print_summary(before_tokens, after_tokens, tk)
    _persist_stage_entries(tk, logger)
    return before_tokens, after_tokens, deltas, cache_benefit


def _persist_stage_entries(tk: Tokenetics, logger: FileCostLogger | None) -> None:
    if logger is None:
        return
    entries = getattr(tk.logger, "entries", None) or []
    for entry in entries:
        logger.log_stage(
            entry.stage_name,
            enabled=entry.enabled,
            tokens_before=entry.tokens_before,
            tokens_after=entry.tokens_after,
            measured=entry.measured,
            **entry.extra,
        )


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
    "cache_reorder_guard": "verifies/normalizes only, no token-count impact by design",
    "cache_breakpoint_optimizer": (
        "attaches cache_control -- changes PRICE PER TOKEN via cheaper cache reads, "
        "not token count (needs --measure-usage against real usage.cache_read_input_tokens "
        "to measure)"
    ),
}


def _print_summary(
    before_tokens: int, after_tokens: int, tk: Tokenetics
) -> tuple[dict[str, int], float]:
    """Prints a request-side (input) token-savings summary, all measured for
    free from count_tokens -- no completion required. Returns per-stage
    deltas plus an estimated cache-dollar benefit (see below) so callers
    (e.g. --all-scenarios) can aggregate across runs.
    """
    print("--- token savings summary (request/input side, measured) ---")
    saved = before_tokens - after_tokens
    pct = (saved / before_tokens * 100) if before_tokens else 0.0
    print(f"overall: {before_tokens} -> {after_tokens} tokens (saved {saved}, {pct:.1f}%)")

    entries = getattr(tk.logger, "entries", None) or []
    deltas: dict[str, int] = {}
    cache_benefit_estimate = 0.0
    print("per stage:")
    for entry in entries:
        delta = (entry.tokens_before or 0) - (entry.tokens_after or 0)
        deltas[entry.stage_name] = deltas.get(entry.stage_name, 0) + delta
        if delta > 0:
            stage_pct = (delta / before_tokens * 100) if before_tokens else 0.0
            note = f"saved {delta} tokens ({stage_pct:.1f}% of original)"
        elif delta < 0:
            note = f"cost {-delta} tokens (adds content, e.g. a brevity instruction)"
        elif entry.stage_name == "cache_breakpoint_optimizer" and entry.extra.get(
            "breakpoint_placed"
        ):
            # Token count is unchanged by design (cache_control doesn't add/remove
            # tokens) -- the real effect is PRICE PER TOKEN, which the stage already
            # computes and logs (core/cache_pricing.py-derived, estimated not
            # measured). Multiply out size x net-benefit/token for a concrete
            # per-repeat-cycle $ figure, instead of a bare "0 tokens".
            size = entry.extra["estimated_size_tokens"]
            benefit_per_token = entry.extra["estimated_net_benefit_per_token"]
            benefit = size * benefit_per_token
            cache_benefit_estimate += benefit
            ttl = entry.extra["cache_ttl"]
            note = (
                f"0 tokens (unchanged by design) -- estimated net benefit "
                f"${benefit:.6f} per repeat cycle ({size} tokens cached at the "
                f"'{ttl}' tier, {benefit_per_token:.2e}/token, estimated)"
            )
        else:
            reason = _ZERO_DELTA_NOTE.get(entry.stage_name)
            note = f"0 tokens{f' -- {reason}' if reason else ''}"
        print(f"  {entry.stage_name}: {note}")
    return deltas, cache_benefit_estimate


def _usage_dict(response: Any) -> dict[str, int]:
    usage = response.usage
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", 0) or 0,
        "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0) or 0,
    }


def _extract_cache_ttl(prepared: dict[str, Any]) -> str | None:
    system = prepared.get("system")
    if isinstance(system, list) and system and "cache_control" in system[-1]:
        ttl = system[-1]["cache_control"].get("ttl")
        return str(ttl) if ttl is not None else None
    tools = prepared.get("tools")
    if tools and "cache_control" in tools[-1]:
        ttl = tools[-1]["cache_control"].get("ttl")
        return str(ttl) if ttl is not None else None
    return None


# Mirrors benchmark_runner.py's own `_REALISTIC_MAX_TOKENS_FLOOR` fix
# exactly, for exactly the same reason: SCENARIOS' own max_tokens values
# are deliberately tiny (to demonstrate adaptive_budget's widening
# mechanism firing at all in the non---measure-usage demo path), not to
# represent a realistic caller's own cap. Confirmed live 2026-09-06 --
# ~88% of a real --measure-usage run's total "output tokens cost" traced
# to exactly 4 scenarios where BOTH the baseline AND optimized call hit
# stop_reason="max_tokens": the baseline was cut off almost immediately at
# its tiny original cap, the optimized one cut off later after writing
# more -- not a fair comparison, the same shape of confound already found
# and fixed once before. Scoped to ONLY `_measure_usage()`'s own use, not
# `scenario.kwargs` itself or any other code path -- the plain (non-
# --measure-usage) demo path still needs the real, tiny caller-set value
# to demonstrate widening firing at all.
_REALISTIC_MAX_TOKENS_FLOOR = 2048


def _with_realistic_max_tokens(kwargs: dict[str, Any]) -> dict[str, Any]:
    kwargs = dict(kwargs)
    kwargs["max_tokens"] = max(kwargs.get("max_tokens", 0), _REALISTIC_MAX_TOKENS_FLOOR)
    return kwargs


def _thinking_tokens(response: Any) -> int | None:
    # Real, measured -- NOT the same estimate _measure_usage's docstring
    # used to rely on (output_tokens minus visible-text tokens). Confirmed
    # 2026-09-06 against the installed anthropic SDK (0.117.0):
    # `usage.output_tokens_details.thinking_tokens` is a real field ("Number
    # of output tokens the model generated as internal reasoning... Computed
    # by re-tokenizing the raw reasoning text, so it may differ from the
    # model's exact generation count by a small number of tokens" -- close
    # to exact, not a derived guess). None (not 0) when the side didn't use
    # thinking at all, so callers can tell "used 0 thinking tokens" apart
    # from "thinking wasn't engaged."
    details = getattr(response.usage, "output_tokens_details", None)
    if details is None:
        return None
    return int(details.thinking_tokens)


def _visible_text(response: Any) -> str:
    # .rstrip() matters, not just cosmetic: the real Messages API rejects
    # trailing whitespace in assistant content (the exact bug already found
    # and fixed once in Tokenetics.finalize()) -- count_text_tokens() wraps
    # this text as a synthetic assistant message, so without stripping here
    # too, any real reply ending in whitespace crashes count_tokens with a
    # real 400 (confirmed live 2026-09-06: this exact crash mid-run, after
    # `long_history` completed and before the next scenario's measurement).
    return "".join(
        getattr(block, "text", "") for block in response.content if getattr(block, "type", None) == "text"
    ).rstrip()


def _measure_usage(
    name: str,
    scenario: Scenario,
    client: anthropic.Anthropic,
    disabled: list[str],
    model: str | None,
    logger: FileCostLogger | None = None,
) -> tuple[dict[str, int], float, float]:
    """Runs the SAME scenario as two real completions -- once through the
    pipeline, once raw -- and diffs response.usage. This is the only way to
    measure output-side savings (adaptive_budget's effort/max_tokens choices)
    and the authoritative version of cache-tier savings: request-token counts
    (what _print_summary measures for free) can't see cache_breakpoint_optimizer's
    effect at all, since attaching cache_control doesn't change token count, only
    price-per-token. Costs 2 real completions -- opt-in via --measure-usage.

    If the pipeline-enabled call actually placed a breakpoint AND paid a cache
    write (cache_creation_input_tokens > 0), a third identical pipeline call is
    made immediately after, while still inside the TTL window, to demonstrate
    the cheap cache READ that a write-only comparison can't show -- a single
    with-vs-without diff always lands on the write, never the read, since the
    prefix has never been cached before that first call.

    Returns (usage_diff, never_cache_cost, with_cache_cost) -- the latter two
    are 0.0 unless a real cache read was confirmed on the repeat call, so
    callers can sum them across scenarios for an aggregate $ verdict.

    If `logger` is given, also logs a `generation_usage` event -- real
    baseline/optimized `output_tokens` (measured, from usage) plus each
    side's visible-text token count (also measured, via count_text_tokens
    on the extracted TextBlock text -- same technique --show-thinking
    already uses for one response), PLUS each side's real
    `output_tokens_details.thinking_tokens` (measured directly from usage,
    added 2026-09-06 -- see `_thinking_tokens()`; supersedes the older
    output-tokens-minus-visible-text estimate as the authoritative number
    for CLAUDE.md's stage 9c line item "configured effort level vs. actual
    thinking tokens consumed at generation time"). The older estimate stays
    in aggregate.py as a fallback for any historical log data logged before
    this field existed, but is no longer the primary signal.

    Also logs a `cache_usage` event per real completion made here (the
    pipeline-enabled call, and the repeat call when one fires) -- a real
    gap found 2026-09-06: this function already computes and even makes a
    real cache-read repeat call, but never told the logger about it, so a
    `--measure-usage` run of `cache_hit` left the dashboard's cache section
    completely unchanged (only the plain, non---measure-usage single-
    scenario path logged `cache_usage` at all). Fixed by logging it here
    too, same event shape as that path already uses.

    `max_tokens` is floored to `_REALISTIC_MAX_TOKENS_FLOOR` for BOTH
    calls (see that constant's own comment) -- a real diagnostic run
    confirmed the scenario's own tiny caller-set value made this
    comparison unfair, truncating the baseline almost immediately.
    """
    request_kwargs = _with_realistic_max_tokens(dict(scenario.kwargs))
    if model:
        request_kwargs["model"] = model

    tk = Tokenetics(client=client, stage_config=scenario.stage_config)
    for stage in tk.stages:
        if stage.name in disabled:
            stage.enabled = False
    prepared = tk.prepare(**request_kwargs)

    without_response = client.messages.create(**request_kwargs)
    with_response = client.messages.create(**prepared)

    without_usage = _usage_dict(without_response)
    with_usage = _usage_dict(with_response)
    diff = {k: without_usage[k] - with_usage[k] for k in with_usage}

    print(f"--- measured usage (real completions, {name}: with vs without pipeline) ---")
    for key in with_usage:
        sign = "saved" if diff[key] >= 0 else "cost"
        print(
            f"  {key}: without={without_usage[key]}, with={with_usage[key]}, "
            f"{sign} {abs(diff[key])}"
        )
    # Printed unconditionally (not just when --log-to is set): a scenario's
    # own tiny caller-set max_tokens (chosen deliberately small, to
    # demonstrate adaptive_budget's widening mechanism firing at all) means
    # the BASELINE call is a real truncation risk -- if it got cut off,
    # "output tokens cost more with the pipeline" isn't waste, it's "the
    # baseline is a cut-off fragment, the optimized one is a complete
    # answer." Confirmed necessary 2026-09-06: a real run showed every
    # scenario with a large max_tokens widen also showing a large output-
    # token "cost", and this is the one piece of evidence (stop_reason)
    # that was missing to tell truncation apart from genuine waste.
    print(
        f"  stop_reason: without={without_response.stop_reason!r}, with={with_response.stop_reason!r}"
    )

    baseline_thinking = _thinking_tokens(without_response)
    optimized_thinking = _thinking_tokens(with_response)
    if baseline_thinking is not None or optimized_thinking is not None:
        print(
            f"  thinking tokens (real, measured): without={baseline_thinking}, "
            f"with={optimized_thinking}"
        )

    if logger is not None:
        model_name = request_kwargs["model"]
        without_visible_tokens = count_text_tokens(_visible_text(without_response), model_name, client)
        with_visible_tokens = count_text_tokens(_visible_text(with_response), model_name, client)
        logger.log_event(
            "generation_usage",
            scenario=name,
            baseline_output_tokens=without_usage["output_tokens"],
            optimized_output_tokens=with_usage["output_tokens"],
            baseline_truncated=without_response.stop_reason == "max_tokens",
            optimized_truncated=with_response.stop_reason == "max_tokens",
            baseline_visible_text_tokens=without_visible_tokens,
            optimized_visible_text_tokens=with_visible_tokens,
            baseline_thinking_tokens=baseline_thinking,
            optimized_thinking_tokens=optimized_thinking,
        )
        logger.log_event(
            "cache_usage",
            cache_read_input_tokens=with_usage["cache_read_input_tokens"],
            cache_creation_input_tokens=with_usage["cache_creation_input_tokens"],
            input_tokens=with_usage["input_tokens"],
        )

    if with_usage["cache_creation_input_tokens"] > 0:
        repeat_response = client.messages.create(**prepared)
        repeat_usage = _usage_dict(repeat_response)
        print(
            f"  cache repeat (immediate re-send, still inside TTL): "
            f"cache_read_input_tokens={repeat_usage['cache_read_input_tokens']} "
            f"(expect > 0 -- the write from the previous call being reused), "
            f"cache_creation_input_tokens={repeat_usage['cache_creation_input_tokens']} "
            f"(expect 0 -- should not write again)"
        )
        diff["cache_read_input_tokens_on_repeat"] = repeat_usage["cache_read_input_tokens"]
        if logger is not None:
            logger.log_event(
                "cache_usage",
                cache_read_input_tokens=repeat_usage["cache_read_input_tokens"],
                cache_creation_input_tokens=repeat_usage["cache_creation_input_tokens"],
                input_tokens=repeat_usage["input_tokens"],
            )

        read_tokens = repeat_usage["cache_read_input_tokens"]
        if read_tokens > 0:
            # Cost verdict: measured write + measured read tokens run through
            # core.cache_pricing's real multipliers, vs. the same prefix sent
            # twice at full price with no caching at all. Uses MEASURED token
            # counts, not the request-side estimate -- this is the actual
            # dollar reduction, not a projection. A one-time write premium
            # amortizes further with each additional repeat inside the TTL,
            # so this two-call snapshot is a floor on the real savings, not
            # a ceiling.
            write_tokens = with_usage["cache_creation_input_tokens"]
            ttl = _extract_cache_ttl(prepared)
            write_multiplier = WRITE_MULTIPLIER_1H if ttl == "1h" else WRITE_MULTIPLIER_5M
            base = base_input_price_for(request_kwargs["model"]) / 1_000_000
            never_cache_cost = (write_tokens + read_tokens) * base
            with_cache_cost = write_tokens * base * write_multiplier + read_tokens * base * READ_MULTIPLIER
            reduction_pct = (
                (1 - with_cache_cost / never_cache_cost) * 100 if never_cache_cost else 0.0
            )
            print(
                f"  cost verdict (measured tokens x core.cache_pricing, "
                f"model={request_kwargs['model']}, ttl={ttl}): never-cache "
                f"${never_cache_cost:.6f} vs write+read ${with_cache_cost:.6f} -- "
                f"{reduction_pct:.1f}% cheaper over these two calls (amortizes "
                f"further with each additional repeat inside the TTL)"
            )
            return diff, never_cache_cost, with_cache_cost

    return diff, 0.0, 0.0


# Fixed prompt for --measure-thinking-reinjection -- deliberately meaty
# enough (a real code question, not a one-liner) to reliably produce a
# non-trivial thinking block worth measuring the re-injection cost of.
_THINKING_REINJECTION_FIRST_TURN: dict[str, Any] = {
    "model": "claude-sonnet-5",
    "max_tokens": 1024,
    "messages": [
        {
            "role": "user",
            "content": (
                "Why is this recursive Fibonacci function slow for large n, and how "
                "would you make it faster?\n```python\ndef fib(n):\n"
                "    if n <= 1:\n        return n\n    return fib(n-1) + fib(n-2)\n```"
            ),
        }
    ],
}

# A deliberately harder prompt (added 2026-09-06 after the simple prompt's
# real 16-sample result came back statistically indistinguishable from
# zero -- 13/16 deltas within +-1 token, 95% CI [-4.0, +2.1]): a genuine
# multi-approach algorithm-design question, meant to elicit a much LONGER
# summarized reasoning block than a single "why is this slow" question
# does. The hypothesis this tests: the simple prompt showed no measurable
# re-injection saving because its summarized thinking block was probably
# short to begin with, not because the mechanism doesn't work -- if that's
# right, a much longer thinking block should show a real, clearly-above-
# noise-floor gap between omitted and summarized turn-2 input tokens.
_THINKING_REINJECTION_HARD_FIRST_TURN: dict[str, Any] = {
    "model": "claude-sonnet-5",
    "max_tokens": 2048,
    "messages": [
        {
            "role": "user",
            "content": (
                "Design an efficient algorithm to find the k-th smallest element in the "
                "union of two sorted arrays, without merging them, in better than O(n) time. "
                "Compare at least two approaches (e.g. a naive merge-based one and a "
                "binary-search-based one), explain why the better one achieves its complexity, "
                "and prove its correctness."
            ),
        }
    ],
}

_THINKING_REINJECTION_FOLLOW_UP = "Restate your fix in one short sentence."


def _measure_thinking_reinjection(
    client: anthropic.Anthropic,
    logger: FileCostLogger | None,
    first_turn_kwargs: dict[str, Any] = _THINKING_REINJECTION_FIRST_TURN,
    variant: str = "simple",
) -> None:
    """Measures CLAUDE.md's stage 9c line item 2 -- tokens saved by
    `display: "omitted"` not being re-billed when a thinking block is
    echoed back as history on a later turn. A genuinely different cost
    dimension from _measure_usage()'s output-token savings (generation-
    time) or _thinking_tokens()'s real spend at generation time (both
    about THIS turn's cost) -- this one is about RE-INJECTION cost: what a
    LATER turn's real input tokens cost once a previous turn's thinking
    block is part of history. Not built until the user asked for it
    directly (2026-09-06); flagged as the known remaining gap when the
    first thinking-tokens dashboard work landed.

    Runs two full two-turn threads with the IDENTICAL first_turn_kwargs,
    differing only in `thinking.display`:
      - "omitted" -- adaptive_budget's real default. The block's reasoning
        text isn't returned, so echoing it back on turn 2 should cost
        near-zero extra input tokens.
      - "summarized" -- the block's summarized reasoning text IS returned,
        so echoing it back on turn 2 re-bills those tokens as real input.

    The real `usage.input_tokens` on turn 2 (both threads sent the exact
    same follow-up text and message shape, differing only in the prior
    thinking block's content) gives a genuine, measured delta -- not an
    estimate. `response.content` is passed straight back as the next
    message's `content` (the Anthropic SDK's `MessageParam.content`
    documents `ContentBlock` -- i.e. a real response's own blocks -- as a
    directly accepted type, no re-serialization needed). Costs 4 real
    completions. Standalone -- ignores --scenario.

    **Known noise floor, confirmed 2026-09-06 via a real 16-sample run of
    the "simple" (default) prompt that came back statistically
    indistinguishable from zero** (mean delta -0.94, stdev 6.08, 95% CI
    [-3.98, 2.10]; 13/16 individual deltas within +-1 token): the two
    threads' turn-1 completions are independently sampled -- extended
    thinking doesn't support `temperature=0`, so the model's exact reply
    text (and therefore turn 2's full preceding-conversation token count)
    varies a little between the omitted-thread and summarized-thread runs
    for reasons that have nothing to do with `display`. At small n, or on
    a prompt whose thinking block is naturally short, this reply-length
    variance can swamp the true effect entirely -- see `variant="hard"`
    (the `_THINKING_REINJECTION_HARD_FIRST_TURN` prompt) for an attempt at
    a question whose summarized reasoning is long enough to rise above
    this noise floor. Don't read one run's number as "the" answer; the
    dashboard doesn't separate variants or correct for this noise, so
    treat any single run as more speculative than the other Tier 2 cards.
    """
    omitted_kwargs = dict(first_turn_kwargs)
    omitted_kwargs["thinking"] = {"type": "adaptive", "display": "omitted"}
    summarized_kwargs = dict(first_turn_kwargs)
    summarized_kwargs["thinking"] = {"type": "adaptive", "display": "summarized"}

    print(f"--- thinking re-injection measurement ({variant}, turn 1, 2 real completions) ---")
    omitted_turn1 = client.messages.create(**omitted_kwargs)
    summarized_turn1 = client.messages.create(**summarized_kwargs)
    print(
        f"  turn 1 thinking tokens (real): omitted={_thinking_tokens(omitted_turn1)}, "
        f"summarized={_thinking_tokens(summarized_turn1)}"
    )

    def _next_turn_kwargs(base_kwargs: dict[str, Any], turn1_response: Any) -> dict[str, Any]:
        kwargs = dict(base_kwargs)
        kwargs["messages"] = [
            *base_kwargs["messages"],
            {"role": "assistant", "content": turn1_response.content},
            {"role": "user", "content": _THINKING_REINJECTION_FOLLOW_UP},
        ]
        return kwargs

    print(f"--- thinking re-injection measurement ({variant}, turn 2, 2 real completions) ---")
    omitted_turn2 = client.messages.create(**_next_turn_kwargs(omitted_kwargs, omitted_turn1))
    summarized_turn2 = client.messages.create(
        **_next_turn_kwargs(summarized_kwargs, summarized_turn1)
    )

    omitted_input = omitted_turn2.usage.input_tokens
    summarized_input = summarized_turn2.usage.input_tokens
    saved = summarized_input - omitted_input

    print(f"  turn 2 input tokens: omitted={omitted_input}, summarized={summarized_input}")
    print(f"  re-injection tokens saved by display='omitted' (real, measured): {saved}")

    if logger is not None:
        logger.log_event(
            "thinking_reinjection",
            variant=variant,
            omitted_turn2_input_tokens=omitted_input,
            summarized_turn2_input_tokens=summarized_input,
        )


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
        "--measure-usage",
        action="store_true",
        help=(
            "With --all-scenarios: for each real-completion-eligible scenario, also run a "
            "real completion with and without the pipeline and diff response.usage "
            "(input/output/cache tokens) -- the only way to measure output-side and "
            "cache-tier savings. Costs 2 real completions per eligible scenario."
        ),
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
    parser.add_argument(
        "--tale",
        action="store_true",
        help=(
            "Tier 2c (opt-in, Phase 10): run a real TALE-style budget-estimation side-call "
            "(cheap, on claude-haiku-4-5 by default) before prepare(), and use its estimate "
            "in place of adaptive_budget's free task-type heuristic. Prints both the "
            "heuristic's and TALE's max_tokens for a head-to-head comparison. Costs 1 extra "
            "small completion. Only applies with --scenario (not --all-scenarios)."
        ),
    )
    parser.add_argument(
        "--compress-ratio",
        type=float,
        default=None,
        metavar="RATIO",
        help=(
            "Tier 2a (opt-in, Phase 10): locally compress (free, no network) the latest user "
            "turn's text at this ratio in (0.0, 1.0) via tokenetics.extras.compress before "
            "sending the real completion. Prints the original text, the compressed text, and "
            "the real achieved word-reduction ratio. Only applies with --scenario. Requires "
            "tokenetics[compress] installed."
        ),
    )
    parser.add_argument(
        "--semantic-cache",
        action="store_true",
        help=(
            "Tier 2b (opt-in, Phase 10): demonstrate SemanticCache's store/lookup round trip "
            "against the scenario's latest user turn -- a first lookup (expected miss, cache "
            "starts empty), the real completion, a store() of the real reply, then a second "
            "lookup with the same query (expected hit) to show the mechanism working within "
            "one run, not just that it doesn't crash. No network calls of its own -- entirely "
            "local embedding inference. Requires tokenetics[semantic-cache] installed."
        ),
    )
    parser.add_argument(
        "--measure-thinking-reinjection",
        action="store_true",
        help=(
            "Standalone (ignores --scenario): real 2-turn, 2-thread A/B measuring CLAUDE.md's "
            "stage 9c line item 2 -- tokens saved by adaptive_budget's display='omitted' "
            "thinking config not being re-billed when history including a thinking block is "
            "echoed back on a later turn, vs. display='summarized' which does get re-billed. "
            "Uses a fixed 'simple' built-in prompt -- a real 16-sample run of it came back "
            "statistically indistinguishable from zero (see --measure-thinking-reinjection-hard "
            "for a prompt designed to produce a longer thinking block). Costs 4 real completions."
        ),
    )
    parser.add_argument(
        "--measure-thinking-reinjection-hard",
        action="store_true",
        help=(
            "Same measurement as --measure-thinking-reinjection, but with a deliberately harder "
            "algorithm-design prompt meant to elicit a much longer summarized thinking block -- "
            "testing whether the simple prompt's null result was because its own thinking block "
            "was too short to show a savings above the natural sampling-noise floor. Costs 4 real "
            "completions. Logs to the same thinking_reinjection event, tagged variant='hard'."
        ),
    )
    parser.add_argument(
        "--log-to",
        default=None,
        metavar="PATH",
        help=(
            "Phase 11 dashboard: append this run's CostLogger entries as JSONL to PATH via "
            "FileCostLogger, instead of the default in-memory-only logger. View with "
            "scripts/dashboard.py --log-file PATH. Only applies with --scenario (not "
            "--all-scenarios, which doesn't build a single Tokenetics() instance)."
        ),
    )
    args = parser.parse_args()

    if args.list_scenarios:
        for name, scenario in sorted(SCENARIOS.items()):
            print(f"{name}: {scenario.description}")
        return

    client = anthropic.Anthropic()

    # Constructed here (not later, right before the single-scenario path's
    # Tokenetics()) so --all-scenarios --measure-usage --log-to can also
    # log real generation_usage events (see _measure_usage) -- previously
    # this was constructed after the --all-scenarios branch's own `return`,
    # so --log-to was silently a no-op in that mode entirely.
    logger = FileCostLogger(args.log_to) if args.log_to else None

    if args.measure_thinking_reinjection:
        _measure_thinking_reinjection(client, logger)
        return

    if args.measure_thinking_reinjection_hard:
        _measure_thinking_reinjection(
            client, logger, first_turn_kwargs=_THINKING_REINJECTION_HARD_FIRST_TURN, variant="hard"
        )
        return

    if args.all_scenarios:
        if logger is not None:
            print(f"--log-to: appending this run's CostLogger entries to {args.log_to!r} (run_id={logger.run_id})")
        total_before = 0
        total_after = 0
        total_deltas: dict[str, int] = {}
        total_usage_diff: dict[str, int] = {}
        total_cache_benefit = 0.0
        total_never_cache_cost = 0.0
        total_with_cache_cost = 0.0
        for name, scenario in sorted(SCENARIOS.items()):
            # A fresh FileCostLogger per scenario, not the single outer
            # `logger` -- its run_id is fixed at construction and shared by
            # every log_stage() call made through it, matching this
            # project's "one fresh Tokenetics() (and therefore one fresh
            # logger) per request" convention. Reusing one shared instance
            # across all scenarios here (the original version of this fix)
            # tagged every scenario's stage entries with the SAME run_id,
            # so aggregate() treated a dozen unrelated requests as one
            # "run" -- comparing the first stage's tokens_before from
            # whichever scenario happened to log first against the last
            # stage's tokens_after from a completely different scenario.
            # Confirmed live 2026-09-06: a real dashboard showed "30 -> 503
            # tokens (saved -473, -1576.7%)", nonsense from mixing scenarios
            # this way. Each scenario IS conceptually a separate request,
            # so each needs its own run_id.
            scenario_logger = FileCostLogger(args.log_to) if args.log_to else None
            before_tokens, after_tokens, deltas, cache_benefit = _run_prepare(
                name, scenario, client, args.disable, scenario_logger
            )
            total_before += before_tokens
            total_after += after_tokens
            total_cache_benefit += cache_benefit
            for stage_name, delta in deltas.items():
                total_deltas[stage_name] = total_deltas.get(stage_name, 0) + delta

            if args.measure_usage:
                if scenario.prepare_only or scenario.expect_cache_safety_error:
                    print(f"(skipping --measure-usage for {name}: prepare()-only scenario)")
                    continue
                usage_diff, never_cache_cost, with_cache_cost = _measure_usage(
                    name, scenario, client, args.disable, args.model, scenario_logger
                )
                for key, value in usage_diff.items():
                    total_usage_diff[key] = total_usage_diff.get(key, 0) + value
                total_never_cache_cost += never_cache_cost
                total_with_cache_cost += with_cache_cost

        if args.measure_usage and total_usage_diff:
            print("\n=== TOTAL measured usage across all scenarios (real completions) ===")
            for key, value in total_usage_diff.items():
                sign = "saved" if value >= 0 else "cost"
                print(f"  {key}: {sign} {abs(value)} total")
            if total_never_cache_cost > 0:
                reduction_pct = (1 - total_with_cache_cost / total_never_cache_cost) * 100
                print(
                    f"  cost verdict (measured, summed across scenarios with a confirmed "
                    f"cache read): never-cache ${total_never_cache_cost:.6f} vs write+read "
                    f"${total_with_cache_cost:.6f} -- {reduction_pct:.1f}% cheaper"
                )

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
            elif stage_name == "cache_breakpoint_optimizer" and total_cache_benefit != 0:
                note = (
                    f"0 tokens total (unchanged by design) -- estimated net cache "
                    f"benefit ${total_cache_benefit:.6f} total (per-repeat-cycle, "
                    f"estimated)"
                )
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
        _run_prepare(args.scenario, scenario, client, args.disable, logger)
        return

    if args.measure_usage:
        # `--measure-usage` was previously only ever checked inside the
        # --all-scenarios loop -- combined with a single --scenario, the
        # flag was silently a no-op (accepted by argparse, never read
        # anywhere else). Caught 2026-09-06 when a user ran `--scenario
        # aggressive_brevity --measure-usage` expecting the real with-vs-
        # without diff and got a single plain completion instead, with no
        # way to test one scenario's real output-token effect without
        # paying for the entire --all-scenarios battery. `_measure_usage`
        # already does exactly what's needed; this mirrors how --all-
        # scenarios itself calls it, as an alternative path rather than
        # combined with the demo flags below (--compress-ratio/
        # --semantic-cache/--tale/--show-thinking), the same scope
        # --all-scenarios already has.
        _run_prepare(args.scenario, scenario, client, args.disable, logger)
        _measure_usage(args.scenario, scenario, client, args.disable, args.model, logger)
        return

    request_kwargs = dict(scenario.kwargs)
    if args.model:
        request_kwargs["model"] = args.model

    if args.show_thinking and "thinking" not in request_kwargs:
        # display="summarized" populates the thinking block's text so we can
        # measure it. Setting this here means adaptive_budget (stage 9c)
        # will leave it alone -- it only sets thinking config when the
        # caller hasn't already.
        request_kwargs["thinking"] = {"type": "adaptive", "display": "summarized"}

    if args.compress_ratio is not None:
        # Tier 2a: entirely local, no network call -- compress the latest
        # user turn's text before it's ever sent.
        messages = request_kwargs.get("messages", [])
        last_user_index = next(
            (i for i in range(len(messages) - 1, -1, -1) if messages[i].get("role") == "user"),
            None,
        )
        if last_user_index is None or not isinstance(messages[last_user_index].get("content"), str):
            print("--compress-ratio: no plain-string user turn found to compress; skipping")
        else:
            original_text = messages[last_user_index]["content"]
            compress_result = compress_text(original_text, args.compress_ratio)
            print(f"--compress-ratio: original text: {original_text!r}")
            if compress_result.clamped:
                print(
                    f"--compress-ratio: requested {compress_result.requested_ratio:.2f} exceeds "
                    "the recommended max -- clamped down (see extras/compress.py's "
                    "RECOMMENDED_MAX_RATIO)"
                )
            print(
                f"--compress-ratio: compressed text (success={compress_result.success}, "
                f"achieved={compress_result.ratio_achieved:.2f}): {compress_result.compressed_text!r}"
            )
            if logger is not None:
                logger.log_event(
                    "compress",
                    requested_ratio=compress_result.requested_ratio,
                    ratio_achieved=compress_result.ratio_achieved,
                    clamped=compress_result.clamped,
                    success=compress_result.success,
                )
            if compress_result.success:
                messages = list(messages)
                messages[last_user_index] = {
                    **messages[last_user_index],
                    "content": compress_result.compressed_text,
                }
                request_kwargs["messages"] = messages

    semantic_cache: SemanticCache | None = None
    semantic_cache_query: str | None = None
    if args.semantic_cache:
        # Tier 2b: entirely local embedding inference, no network call of
        # its own -- store()/lookup() round trip demonstrated within one
        # run, since this script has no cross-run persistence to show a
        # hit against a PREVIOUS invocation. The second lookup (after the
        # real response comes back and gets stored) happens further down.
        messages = request_kwargs.get("messages", [])
        last_user_index = next(
            (i for i in range(len(messages) - 1, -1, -1) if messages[i].get("role") == "user"),
            None,
        )
        if last_user_index is None or not isinstance(messages[last_user_index].get("content"), str):
            print("--semantic-cache: no plain-string user turn found to demo against; skipping")
        else:
            semantic_cache_query = messages[last_user_index]["content"]
            semantic_cache = SemanticCache()
            first_lookup = semantic_cache.lookup(semantic_cache_query)
            print(f"--semantic-cache: first lookup (expect miss, cache is empty): hit={first_lookup.hit}")
            if logger is not None:
                # Only THIS lookup counts toward the dashboard's hit-rate
                # stats -- it's the realistic "would this have skipped the
                # API call" check a real caller would do. The second
                # lookup below (after store()) only demonstrates the
                # mechanism works, not a real cache-hit-rate data point.
                logger.log_event("semantic_cache", hit=first_lookup.hit, similarity=first_lookup.similarity)

    stage_config = {k: dict(v) for k, v in (scenario.stage_config or {}).items()}
    if args.tale:
        # Tier 2c: this is the one real network call in the whole script
        # that isn't Tokenetics itself -- a caller-run side-call, per
        # `extras/tale.py`'s docstring on why this can't live inside the
        # (network-free) Tier 0 core.
        tale_result = estimate_token_budget(
            client,
            messages=request_kwargs.get("messages", []),
            model=request_kwargs.get("model", "claude-sonnet-5"),
            system=request_kwargs.get("system"),
        )
        if tale_result is None:
            print("--tale: estimation call failed or was unparseable; falling back to the free heuristic")
        else:
            print(
                f"--tale: TALE estimate = {tale_result.budget_tokens} tokens "
                f"(estimation side-call cost: {tale_result.estimation_input_tokens} in / "
                f"{tale_result.estimation_output_tokens} out, measured)"
            )
            if logger is not None:
                logger.log_event(
                    "tale",
                    budget_tokens=tale_result.budget_tokens,
                    estimation_input_tokens=tale_result.estimation_input_tokens,
                    estimation_output_tokens=tale_result.estimation_output_tokens,
                )
            stage_config.setdefault("adaptive_budget", {})
            stage_config["adaptive_budget"]["external_budget_estimate"] = tale_result.budget_tokens

    # default pipeline: dedup, near_dup, task_classifier, schema_minification,
    # context_scheduler, structured_output, brevity_injector, adaptive_budget
    tk = Tokenetics(client=client, stage_config=stage_config, logger=logger)
    for stage in tk.stages:
        if stage.name in args.disable:
            stage.enabled = False

    before_tokens = count_tokens(from_api_kwargs(**request_kwargs), client)
    prepared = tk.prepare(**request_kwargs)
    after_tokens = count_tokens(from_api_kwargs(**prepared), client)

    fired = [s.name for s in tk.stages if s.enabled]
    if logger is not None:
        print(f"--log-to: appending this run's CostLogger entries to {args.log_to!r} (run_id={logger.run_id})")
    print(f"scenario: {args.scenario} -- {scenario.description}")
    print(f"stages fired: {fired or '(none -- all disabled)'}")
    print(f"request tokens before -> after: {before_tokens} -> {after_tokens}")

    response = client.messages.create(**prepared)
    print("\nreply (raw):")
    print(response.content)

    if semantic_cache is not None and semantic_cache_query is not None:
        reply_text = "".join(
            getattr(block, "text", "") for block in response.content if getattr(block, "type", None) == "text"
        )
        semantic_cache.store(semantic_cache_query, reply_text)
        second_lookup = semantic_cache.lookup(semantic_cache_query)
        print(
            f"--semantic-cache: after store(), second lookup with the SAME query "
            f"(expect hit): hit={second_lookup.hit}, similarity={second_lookup.similarity}"
        )

    usage = getattr(response, "usage", None)
    if usage is not None:
        print(f"\nusage: input_tokens={usage.input_tokens}, output_tokens={usage.output_tokens}")
        if logger is not None:
            logger.log_event(
                "cache_usage",
                cache_read_input_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
                cache_creation_input_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
                input_tokens=usage.input_tokens,
            )

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
