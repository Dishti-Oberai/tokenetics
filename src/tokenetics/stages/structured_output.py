"""Stage 9a: structured-output enforcement.

When the task classifier flags `extraction`, force the response through an
already-registered tool's schema (via `tool_choice`) rather than hoping the
model free-forms valid JSON. Tier 0 has no model call of its own and can't
invent a schema, so this only ever reuses a tool the caller already defined
-- if no registered tool plausibly matches the request, this is a no-op.
Confirmed with the user during Phase 4 planning (2026-07-26): this was the
"reuse a matching existing tool's schema" option, chosen over inventing a
generic JSON-object constraint or descoping to a logged no-op.

Conservative by design: skips entirely if the caller already set their own
`tool_choice`, if there are no tools registered at all, or if no tool's
name/description clears the lexical-match bar against the latest user
message.
"""

from __future__ import annotations

from dataclasses import replace

from tokenetics.core.lexical import latest_user_text, overlap_score
from tokenetics.core.logger import CostLogger
from tokenetics.core.plugin import Stage, StageConfig
from tokenetics.core.request import TokeneticsRequest

_MATCH_THRESHOLD = 0.05


class StructuredOutputStage(Stage):
    name = "structured_output"

    def run(
        self, request: TokeneticsRequest, config: StageConfig, logger: CostLogger
    ) -> TokeneticsRequest:
        if request.meta.task_type != "extraction":
            return request
        if not request.tools:
            return request
        if "tool_choice" in request.extra:
            return request

        comparison_text = latest_user_text(request.messages)
        if not comparison_text.strip():
            return request

        best_tool = None
        best_score = 0.0
        for tool in request.tools:
            score = overlap_score(f"{tool.name} {tool.description}", comparison_text)
            if score > best_score:
                best_score = score
                best_tool = tool

        if best_tool is None or best_score < _MATCH_THRESHOLD:
            return request

        self.note(tool_choice_forced=best_tool.name, match_score=round(best_score, 3))
        new_extra = dict(request.extra)
        new_extra["tool_choice"] = {"type": "tool", "name": best_tool.name}
        return replace(request, extra=new_extra)
