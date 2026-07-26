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
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from tokenetics.core.lexical import latest_user_text, overlap_score
from tokenetics.core.logger import CostLogger
from tokenetics.core.plugin import Stage, StageConfig
from tokenetics.core.request import ToolSpec, TokeneticsRequest

_INERT_SCHEMA_KEYS = {"$schema", "$id", "$comment"}
_IRRELEVANCE_THRESHOLD = 0.0  # drop only tools with literally zero word overlap


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

        kept_tools: list[ToolSpec] = []
        dropped: list[dict[str, Any]] = []
        for tool in minified_tools:
            score = overlap_score(f"{tool.name} {tool.description}", comparison_text)
            if score <= _IRRELEVANCE_THRESHOLD:
                dropped.append({"tool": tool.name, "score": round(score, 3)})
                continue
            kept_tools.append(tool)

        if dropped:
            self.note(dropped_tools=dropped)

        return replace(request, tools=kept_tools)
