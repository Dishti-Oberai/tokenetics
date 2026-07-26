"""Stage 9c: adaptive generation budgets (max_tokens + thinking effort).

max_tokens: task-type-derived estimate + safety margin, widen-only -- never
narrows below what the caller explicitly supplied, per CLAUDE.md's
conservative default ("uncertain output size -> widen max_tokens, not
narrow it"). The brief's "rolling truncation rate" that widens the margin
further is caller-supplied via `config["truncation_stats"]`, not tracked by
this stage -- Tier 0 is stateless, so the stage can't remember truncation
history across calls itself. Resolved 2026-07-26; see the project brief's
Amendments log for the full reasoning.

thinking effort: task-type-derived `output_config.effort` via
`thinking: {type: "adaptive"}`, never the legacy `budget_tokens` API. Skips
entirely (info-level log, per CLAUDE.md's feature-unsupported-by-model
fail-open flavor) if: the caller already set their own `thinking` config
(respect explicit intent over guessing), the target model isn't listed as
adaptive-capable in the model-compatibility table, or the classifier left
task_type unclassified (ambiguous -> leave untouched, never guess an
effort level). When this stage does set thinking config, it defaults
`display` to "omitted" -- avoids re-billing old reasoning traces as input
tokens if this response is echoed back into history on a later turn.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

from tokenetics.core.logger import CostLogger
from tokenetics.core.model_compatibility import thinking_support_for
from tokenetics.core.plugin import Stage, StageConfig
from tokenetics.core.request import TokeneticsRequest

_log = logging.getLogger(__name__)

_BASE_MAX_TOKENS = {
    "conversational": 300,
    "extraction": 250,
    "tool-heavy": 400,
    "code": 800,
}
_SAFETY_MARGIN = 1.2
_TRUNCATION_RATE_THRESHOLD = 0.02
_TRUNCATION_WIDEN_FACTOR = 1.5

_EFFORT_BY_TASK_TYPE = {
    "extraction": "low",
    "tool-heavy": "low",
    "conversational": "medium",
    "code": "high",
}


class AdaptiveBudgetStage(Stage):
    name = "adaptive_budget"

    def run(
        self, request: TokeneticsRequest, config: StageConfig, logger: CostLogger
    ) -> TokeneticsRequest:
        request = self._apply_max_tokens(request, config)
        request = self._apply_thinking_effort(request)
        return request

    def _apply_max_tokens(
        self, request: TokeneticsRequest, config: StageConfig
    ) -> TokeneticsRequest:
        task_type = request.meta.task_type
        if task_type is None or task_type not in _BASE_MAX_TOKENS:
            return request

        estimate = round(_BASE_MAX_TOKENS[task_type] * _SAFETY_MARGIN)

        truncation_stats: dict[str, float] = config.get("truncation_stats", {})
        truncation_rate = truncation_stats.get(task_type)
        widened_for_truncation = False
        if truncation_rate is not None and truncation_rate > _TRUNCATION_RATE_THRESHOLD:
            estimate = round(estimate * _TRUNCATION_WIDEN_FACTOR)
            widened_for_truncation = True

        if estimate <= request.max_tokens:
            return request  # never narrow below the caller's own value

        self.note(
            max_tokens_widened_to=estimate,
            max_tokens_widen_reason=(
                "truncation_rate" if widened_for_truncation else "task_type_estimate"
            ),
        )
        return replace(request, max_tokens=estimate)

    def _apply_thinking_effort(self, request: TokeneticsRequest) -> TokeneticsRequest:
        if "thinking" in request.extra:
            return request  # respect the caller's own explicit thinking config

        effort = _EFFORT_BY_TASK_TYPE.get(request.meta.task_type or "")
        if effort is None:
            return request  # unclassified, or no mapping for this type -- leave untouched

        support = thinking_support_for(request.model)
        if support != "adaptive":
            _log.info(
                "model %r is not adaptive-thinking-capable (compatibility: %r); "
                "skipping thinking-effort sub-stage",
                request.model,
                support,
            )
            self.note(thinking_effort_skipped="model_unsupported", model=request.model)
            return request

        self.note(thinking_effort=effort)
        new_extra: dict[str, Any] = dict(request.extra)
        new_extra["thinking"] = {"type": "adaptive", "display": "omitted"}
        new_extra["output_config"] = {
            **request.extra.get("output_config", {}),
            "effort": effort,
        }
        return replace(request, extra=new_extra)
