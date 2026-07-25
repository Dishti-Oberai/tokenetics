"""Stage 4a: schema minification (structural half of stage 4).

Strips schema fields that don't affect model behavior -- JSON-Schema
metadata keys Anthropic's tool-calling never reads, and redundant
whitespace in descriptions. Purely structural, works on any schema.

Deliberately does not attempt the brief's "defaults duplicated in
descriptions" heuristic -- too fragile to implement safely as a
deterministic, always-on stage; skipped rather than risk stripping wanted
schema information.

Tool-*relevance* filtering (the other half of stage 4 in the brief) needs
the task classifier's signal and is deferred to Phase 4.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from tokenetics.core.logger import CostLogger
from tokenetics.core.plugin import Stage, StageConfig
from tokenetics.core.request import ToolSpec, TokeneticsRequest

_INERT_SCHEMA_KEYS = {"$schema", "$id", "$comment"}


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
        return replace(request, tools=minified_tools)
