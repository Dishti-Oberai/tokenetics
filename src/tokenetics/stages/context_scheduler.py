"""Stage 5: context scheduler (DP knapsack).

Decides which turns from a conversation history are worth sending, instead
of the two naive extremes (send everything, or blindly keep the last N).
Per the project brief's stage 5 section:

  per-turn classifier -> retention scoring -> pinning pre-allocation ->
  0/1 knapsack (DP, falling back to greedy above a size threshold) ->
  degraded-mode last-N truncation when the stage is disabled entirely.

Two design questions were resolved with the user on 2026-07-26 (see the
project brief's Amendments log for the full reasoning, and the amended
stage 5 section for the mechanism text):

- An `error`-classified turn is pinned by default and only loses pinned
  status when a LATER turn from the *user* (not the assistant, not merely
  "the conversation moved on") contains an explicit positive resolution
  signal. Weak/inferred resolution signals fail in the dangerous direction
  (dropping a still-live problem), so this stays narrow on purpose.
- The token budget is caller-supplied via `config["token_budget"]` when
  given; otherwise it falls back to a context-window-derived default
  (`core/context_window.py`), scaled by the request's `task_type`, and
  logged as `estimated` rather than `measured`.

Per-turn token costs used for the knapsack's internal sizing are a fast,
local, ~4-chars/token estimate -- stages have no client/network access
(Tier 0 is deterministic, no network calls of its own), so this cannot call
the real tokenizer. The orchestrator's own before/after counts (which do
use the real tokenizer) remain the authoritative `measured` numbers for
anything logged; this internal estimate only drives which turns get kept.

Turn-level dropping (both the knapsack path and last-N degraded mode)
doesn't inherently preserve the real API's strict user/assistant
alternation. `_fix_alternation` is the safety net, applied as the last
step in both `run()` and `degraded_fallback()`: adjacent same-role
survivors get merged into one message (nothing lost), and a lone leading
"assistant" survivor gets dropped (the API requires the first message to
be "user" -- the one case where content is actually lost, and it's noted
when it happens). If that fix would empty the result entirely, the stage
fails open and returns the request unmodified rather than send an empty
history.
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import replace
from typing import Any

from tokenetics.core.context_window import default_token_budget
from tokenetics.core.logger import CostLogger
from tokenetics.core.plugin import Stage, StageConfig
from tokenetics.core.request import Message, TokeneticsRequest

_log = logging.getLogger(__name__)

_TOOL_CONTENT_TYPES = {"tool_use", "tool_result"}

_ERROR_TRACE_RE = re.compile(
    r"(Traceback \(most recent call last\)|Exception in thread|"
    r"\b[A-Za-z_.]+(Error|Exception):|at [\w.$]+\([\w./]+:\d+\))"
)
_DECISION_RE = re.compile(
    r"\b(let'?s go with|we'?ll use|decided to|i'?ll go with|let'?s do that|"
    r"agreed[,:]|final answer|we should use|going with)\b",
    re.IGNORECASE,
)
_CODE_FENCE_RE = re.compile(r"```")
_CODE_KEYWORD_RE = re.compile(
    r"\b(def|class|function|import|const|let|var|console\.log|public\s+static|#include)\b",
    re.IGNORECASE,
)
# Deliberately narrow: only fires on an explicit, positive resolution
# statement, never on the conversation simply moving to another topic.
_RESOLUTION_RE = re.compile(
    r"\b(that fixed it|fixed it|works now|that worked|solved it|"
    r"that solved it|resolved now|all good now|working now|got it working)\b",
    re.IGNORECASE,
)

_TYPE_WEIGHT = {
    "decision": 1.0,
    "error": 0.7,
    "tool_result": 0.5,
    "code": 0.4,
    "small_talk": 0.1,
}

_BUCKET_SIZE = 100  # per the brief: budget discretized into 50-100-token buckets
_DEGRADED_LAST_N = 20

# Empirically measured (n items x capacity_buckets, pure-Python DP):
#   n=300,  capacity=300  -> ~4ms
#   n=800,  capacity=800  -> ~38ms
#   n=1000, capacity=1000 -> ~60ms
#   n=2000, capacity=2000 -> ~250ms
# 250,000 (~n=500 turns at a 500-bucket/50k-token budget) keeps DP under
# ~15ms in the common case while still catching genuinely large histories
# before they start costing real latency. Revisit once Phase 9 has real
# request-latency numbers to weigh this against.
_DP_SIZE_THRESHOLD = 250_000


def _message_text(content: str | list[dict[str, Any]]) -> str:
    if isinstance(content, str):
        return content
    parts = [block["text"] for block in content if isinstance(block.get("text"), str)]
    return " ".join(parts)


def _is_tool_content(content: str | list[dict[str, Any]]) -> bool:
    if isinstance(content, str):
        return False
    return any(block.get("type") in _TOOL_CONTENT_TYPES for block in content)


def _classify_turn(message: Message) -> str:
    if _is_tool_content(message.content):
        return "tool_result"
    text = _message_text(message.content)
    if _ERROR_TRACE_RE.search(text):
        return "error"
    if _DECISION_RE.search(text):
        return "decision"
    if _CODE_FENCE_RE.search(text) or _CODE_KEYWORD_RE.search(text):
        return "code"
    return "small_talk"


def _is_resolved(messages: list[Message], error_index: int) -> bool:
    for message in messages[error_index + 1 :]:
        if message.role == "user" and _RESOLUTION_RE.search(_message_text(message.content)):
            return True
    return False


def _retention_score(turn_type: str, index: int, n: int) -> float:
    age = (n - 1) - index  # 0 for the most recent turn
    recency_bonus = 0.5 * (1 - age / n) if n else 0.0
    return _TYPE_WEIGHT[turn_type] + recency_bonus


def _estimate_message_tokens(message: Message) -> int:
    content = message.content
    text = content if isinstance(content, str) else json.dumps(content, sort_keys=True)
    return max(1, len(text) // 4)


def _merge_same_role_content(
    content_a: str | list[dict[str, Any]], content_b: str | list[dict[str, Any]]
) -> str | list[dict[str, Any]]:
    if isinstance(content_a, str) and isinstance(content_b, str):
        return f"{content_a}\n\n{content_b}"
    blocks_a = content_a if isinstance(content_a, list) else [{"type": "text", "text": content_a}]
    blocks_b = content_b if isinstance(content_b, list) else [{"type": "text", "text": content_b}]
    return [*blocks_a, *blocks_b]


def _fix_alternation(messages: list[Message]) -> tuple[list[Message], bool]:
    """Turn-level dropping (pinning + knapsack, or last-N truncation) can't
    guarantee the survivors still alternate user/assistant the way the real
    API requires. This is the safety net: adjacent same-role survivors are
    merged into one message (nothing is dropped, just physically combined
    -- a shorter message list from this alone is not content loss); if the
    result would still start with "assistant" (the API requires the first
    message to be "user"), that lone leading message is dropped -- the one
    case where content is actually lost. Returns (fixed_messages,
    dropped_a_lead_turn) so callers can log the two cases distinctly rather
    than conflating "merged" with "lost content".
    """
    if not messages:
        return messages, False

    merged: list[Message] = [messages[0]]
    for message in messages[1:]:
        if message.role == merged[-1].role:
            merged[-1] = Message(
                role=merged[-1].role,
                content=_merge_same_role_content(merged[-1].content, message.content),
            )
        else:
            merged.append(message)

    dropped_lead = merged[0].role != "user"
    if dropped_lead:
        merged = merged[1:]

    return merged, dropped_lead


def _dp_knapsack(
    indices: list[int], costs: list[int], values: list[float], capacity: int
) -> list[int]:
    n = len(indices)
    dp = [[0.0] * (capacity + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost, value = costs[i - 1], values[i - 1]
        for w in range(capacity + 1):
            dp[i][w] = dp[i - 1][w]
            if cost <= w and dp[i - 1][w - cost] + value > dp[i][w]:
                dp[i][w] = dp[i - 1][w - cost] + value

    chosen: list[int] = []
    w = capacity
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            chosen.append(indices[i - 1])
            w -= costs[i - 1]
    chosen.reverse()
    return chosen


def _greedy_select(
    indices: list[int], costs: list[int], values: list[float], capacity: int
) -> list[int]:
    order = sorted(
        range(len(indices)),
        key=lambda k: (values[k] / costs[k] if costs[k] else float("inf")),
        reverse=True,
    )
    chosen = []
    remaining = capacity
    for k in order:
        if costs[k] <= remaining:
            chosen.append(indices[k])
            remaining -= costs[k]
    return chosen


class ContextSchedulerStage(Stage):
    name = "context_scheduler"

    def run(
        self, request: TokeneticsRequest, config: StageConfig, logger: CostLogger
    ) -> TokeneticsRequest:
        messages = request.messages
        n = len(messages)
        if n == 0:
            return request

        types = [_classify_turn(m) for m in messages]
        pinned = {
            i
            for i, t in enumerate(types)
            if t == "decision" or (t == "error" and not _is_resolved(messages, i))
        }

        token_budget = config.get("token_budget")
        budget_estimated = token_budget is None
        if token_budget is None:
            token_budget = default_token_budget(request.model, request.meta.task_type)

        costs = [_estimate_message_tokens(m) for m in messages]
        pinned_cost = sum(costs[i] for i in pinned)
        remaining_budget = max(0, token_budget - pinned_cost)
        capacity_buckets = remaining_budget // _BUCKET_SIZE

        candidates = [i for i in range(n) if i not in pinned]
        candidate_costs = [max(1, math.ceil(costs[i] / _BUCKET_SIZE)) for i in candidates]
        candidate_values = [_retention_score(types[i], i, n) for i in candidates]

        size_metric = len(candidates) * (capacity_buckets + 1)
        used_greedy = size_metric > _DP_SIZE_THRESHOLD
        if used_greedy:
            chosen = _greedy_select(candidates, candidate_costs, candidate_values, capacity_buckets)
        else:
            chosen = _dp_knapsack(candidates, candidate_costs, candidate_values, capacity_buckets)

        kept_indices = sorted(pinned | set(chosen))
        kept_messages = [messages[i] for i in kept_indices]
        fixed_messages, dropped_lead_turn = _fix_alternation(kept_messages)

        # Budget provenance is always worth logging (cheap, and tells a
        # reader whether the caller supplied it or it's a heuristic
        # fallback); the rest only matters when something actually happened.
        self.note(token_budget=token_budget, token_budget_estimated=budget_estimated)

        if not fixed_messages:
            # The alternation fix emptied everything out (extreme edge
            # case) -- sending an empty history is clearly wrong, so fail
            # open: skip this stage's effect entirely rather than send it.
            self.note(alternation_fix_emptied_result=True)
            return request

        dropped = n - len(fixed_messages)
        if dropped or used_greedy:
            self.note(
                dropped_turns=dropped,
                pinned_turns=len(pinned),
                used_greedy_fallback=used_greedy,
            )
        if dropped_lead_turn:
            self.note(alternation_fix_dropped_lead_turn=True)
        return replace(request, messages=fixed_messages)

    def degraded_fallback(
        self, request: TokeneticsRequest, config: StageConfig, logger: CostLogger
    ) -> TokeneticsRequest:
        messages = request.messages
        if len(messages) <= _DEGRADED_LAST_N:
            self.note(degraded=True, mode="last_n_truncation", kept=len(messages), dropped_turns=0)
            return request

        kept = messages[-_DEGRADED_LAST_N:]
        # A contiguous slice of an alternating conversation can't have
        # adjacent same-role turns, but it can still *start* on the wrong
        # role depending on N's parity -- _fix_alternation's merge loop is
        # a no-op here, only the leading-role check can actually apply.
        fixed, _dropped_lead_turn = _fix_alternation(kept)
        if not fixed:
            self.note(degraded=True, mode="last_n_truncation", alternation_fix_emptied_result=True)
            return request

        self.note(
            degraded=True,
            mode="last_n_truncation",
            kept=len(fixed),
            dropped_turns=len(messages) - len(fixed),
        )
        return replace(request, messages=fixed)
