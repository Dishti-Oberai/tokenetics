"""Stage 4: schema minification + tool-relevance filtering.

(a) Minification -- strips schema fields that don't affect model behavior
(JSON-Schema metadata keys Anthropic's tool-calling never reads, redundant
whitespace in descriptions). Purely structural, works on any schema.

Deliberately does not attempt the brief's "defaults duplicated in
descriptions" heuristic -- too fragile to implement safely as a
deterministic, always-on stage; skipped rather than risk stripping wanted
schema information.

(b) Relevance filtering -- scores each tool by lexical overlap between its
name/description and the latest user message; drops only tools with zero
word overlap ("clearly irrelevant" per the brief), keeps anything with any
connection at all ("keep anything ambiguous"). If there's no user text to
compare against, filtering is skipped entirely and all tools are kept --
we can't judge relevance without something to judge it against, and the
conservative branch is to keep, not drop.

Tools invoked recently -- a `tool_use` block for that tool name within the
last `_TOOL_REUSE_WINDOW_TURNS` genuine user follow-ups -- are exempt from
the drop entirely, regardless of lexical score against the latest turn.
Caught via Phase 9's benchmark corpus (2026-08-02): a real conversation
follow-up ("What about a large?" after already checking medium-size stock)
shares zero words with the tool's name/description, so the un-exempted
filter dropped the tool from `tools` completely, and a quality-check sample
built to test exactly this confirmed it: the model couldn't re-check stock
at all with the tool gone (missing=['check_inventory']), vs. correctly
calling it again with the filter disabled. This exemption is an extension
of the stage's own existing bias, not a new behavior class -- a tool the
model recently reached for is inherently less ambiguous than a never-used
one, so keeping it fits "keep anything ambiguous" rather than overriding it.

Windowed, not permanent (revised 2026-08-09): a first version exempted a
used tool for the rest of the conversation, unconditionally. That's safe
but unbounded -- in a long multi-tool agentic session, every tool ever used
accumulates and never gets pruned again, eventually defeating the point of
this filter. A window bounds that: tools genuinely abandoned partway
through a long conversation still get reclaimed once enough real user
turns have passed without them, while the realistic case that caused the
original bug (an immediate or near-term follow-up) stays protected.
Genuine user turns only -- a `tool_result`-carrying message is mechanically
role="user" but isn't a real follow-up question, so it doesn't consume the
window. Window size (3) reasoned, not guessed: 1 is too tight (fails a
two-step clarification -- "what about a large?" then "in blue too?"); 5+
starts approximating "permanent" for most realistic tool-using threads
(commonly well under 10 turns total), so the marginal safety gained over 3
isn't worth the marginal cost given up. 3 covers a short realistic
clarification thread while still eventually reverting to fresh relevance
scoring once a topic has genuinely moved on.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from tokenetics.core.lexical import latest_user_text, overlap_score
from tokenetics.core.logger import CostLogger
from tokenetics.core.plugin import Stage, StageConfig
from tokenetics.core.request import Message, ToolSpec, TokeneticsRequest

_INERT_SCHEMA_KEYS = {"$schema", "$id", "$comment"}
_IRRELEVANCE_THRESHOLD = 0.0  # drop only tools with literally zero word overlap
_TOOL_REUSE_WINDOW_TURNS = 3  # see module docstring for the sizing rationale


def _is_tool_result_message(message: Message) -> bool:
    if not isinstance(message.content, list) or not message.content:
        return False
    return all(block.get("type") == "tool_result" for block in message.content)


def _tools_recently_used(messages: list[Message], window_turns: int) -> set[str]:
    """Tool names with a `tool_use` block whose "turns since" count is still
    within `window_turns` GENUINE user follow-ups -- not counting mechanical
    tool_result deliveries (role="user" but not a real question) as a turn.
    Forward scan: each tool_use resets that tool's counter to 0 (most-recent-
    use semantics, not first-use); each genuine user turn -- including the
    current query being answered, which does count as "one turn since" for
    a tool used right before it -- increments every tool's counter by 1.
    """
    turns_since_use: dict[str, int] = {}
    for message in messages:
        if message.role == "user" and not _is_tool_result_message(message):
            for tool_name in turns_since_use:
                turns_since_use[tool_name] += 1
            continue
        if not isinstance(message.content, list):
            continue
        for block in message.content:
            if block.get("type") == "tool_use":
                block_name = block.get("name")
                if isinstance(block_name, str):
                    turns_since_use[block_name] = 0
    return {tool_name for tool_name, turns in turns_since_use.items() if turns <= window_turns}


def _collapse_whitespace(text: str) -> str:
    return " ".join(text.split())


def _minify_schema(schema: dict[str, Any]) -> dict[str, Any]:
    minified: dict[str, Any] = {}
    for key, value in schema.items():
        if key in _INERT_SCHEMA_KEYS:
            continue
        if key == "description" and isinstance(value, str):
            minified[key] = _collapse_whitespace(value)
        elif isinstance(value, dict):
            minified[key] = _minify_schema(value)
        elif isinstance(value, list):
            minified[key] = [
                _minify_schema(item) if isinstance(item, dict) else item for item in value
            ]
        else:
            minified[key] = value
    return minified


class SchemaMinificationStage(Stage):
    name = "schema_minification"

    def run(
        self, request: TokeneticsRequest, config: StageConfig, logger: CostLogger
    ) -> TokeneticsRequest:
        if not request.tools:
            return request

        minified_tools: list[ToolSpec] = [
            replace(
                tool,
                description=_collapse_whitespace(tool.description),
                input_schema=_minify_schema(tool.input_schema),
            )
            for tool in request.tools
        ]

        comparison_text = latest_user_text(request.messages)
        if not comparison_text.strip():
            return replace(request, tools=minified_tools)

        recently_used = _tools_recently_used(request.messages, _TOOL_REUSE_WINDOW_TURNS)

        kept_tools: list[ToolSpec] = []
        dropped: list[dict[str, Any]] = []
        for tool in minified_tools:
            if tool.name in recently_used:
                kept_tools.append(tool)
                continue
            score = overlap_score(f"{tool.name} {tool.description}", comparison_text)
            if score <= _IRRELEVANCE_THRESHOLD:
                dropped.append({"tool": tool.name, "score": round(score, 3)})
                continue
            kept_tools.append(tool)

        if dropped:
            self.note(dropped_tools=dropped)

        return replace(request, tools=kept_tools)
