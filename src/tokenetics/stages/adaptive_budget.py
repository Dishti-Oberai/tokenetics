"""Stage 9c: adaptive generation budgets (max_tokens + thinking effort).

max_tokens: task-type-derived estimate + safety margin, widen-only -- never
narrows below what the caller explicitly supplied, per CLAUDE.md's
conservative default ("uncertain output size -> widen max_tokens, not
narrow it"). The brief's "rolling truncation rate" that widens the margin
further is caller-supplied via `config["truncation_stats"]`, not tracked by
this stage -- Tier 0 is stateless, so the stage can't remember truncation
history across calls itself. Resolved 2026-07-26; see the project brief's
Amendments log for the full reasoning.

Tier 2c (TALE-style budget estimation, opt-in, added Phase 10): a caller
may supply `config["external_budget_estimate"]` (an int token count from
`tokenetics.extras.tale.estimate_token_budget()`, or anywhere else) to
replace the free task-type heuristic as the estimate source. This keeps
this stage itself network-free -- the estimation LLM call, if any, happens
entirely outside Tier 0, in the opt-in extras layer, same as every other
caller-supplied input this stage already accepts.

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

**Opt-in, not automatic (revised 2026-09-04, after a real 83-sample $-cost
benchmark)**: this sub-stage used to set thinking effort automatically by
task type (`"code": "high"`, `"conversational": "medium"`) with no caller
signal required. A real end-to-end benchmark (`scripts/benchmark_runner.py
usage`, baseline vs. pipeline, real `response.usage`, real per-token
pricing) measured the actual dollar effect of that default across the full
corpus: `code` came out ~59% MORE expensive and `conversational` ~19% MORE
expensive with the pipeline enabled than sending the request raw --
overwhelmingly driven by extra output tokens from thinking (output is
priced ~5x input for Sonnet 5), not offset by anything else in the
pipeline. `tool-heavy`/`extraction` (both defaulted to `"low"` effort)
did not show this problem. The automatic default was silently making some
calls 2-3x more expensive with no caller signal requesting that trade --
a real credibility problem for a tool whose whole premise is cutting cost.

Fixed by requiring an explicit opt-in: `config["enable_thinking_effort"]`
must be `True` for this sub-stage to run at all; the effort-by-task-type
mapping and every other skip condition above are unchanged, they just no
longer fire without that flag. This matches the "safe by default, explicit
opt-in for anything that trades cost for a different benefit" pattern
already used everywhere else Tier 0/Tier 2 makes this kind of tradeoff
(`tale.py`'s estimation call is entirely opt-in; `compress`'s
`RECOMMENDED_MAX_RATIO` clamps by default and needs
`allow_above_recommended_max=True` to go further) -- this sub-stage was
the one place in the pipeline that didn't follow that pattern, and the
real cost data is exactly why it should.
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
        request = self._apply_thinking_effort(request, config)
        return request

    def _apply_max_tokens(
        self, request: TokeneticsRequest, config: StageConfig
    ) -> TokeneticsRequest:
        # Tier 2c (TALE-style budget estimation, opt-in): a caller may run
        # `tokenetics.extras.tale.estimate_token_budget()` themselves before
        # `prepare()` and pass the result in via `config["external_budget_
        # estimate"]`. This stage stays network-free either way -- Tier 0
        # never makes model calls of its own -- the estimate is just another
        # caller-supplied input, same shape as `truncation_stats`/
        # `cache_usage_history`.
        #
        # It does NOT blindly replace the heuristic (fixed 2026-08-10 after
        # a real `--tale` run against the live API produced a reply with NO
        # visible text at all): TALE estimates only the *visible-answer*
        # length, with no way to know this same stage is about to turn on
        # high-effort adaptive thinking for the same classified task type --
        # and thinking tokens draw from the identical max_tokens budget. A
        # code question TALE sized at 150 tokens (-> 180 after margin) left
        # zero room for text once high-effort thinking consumed the cap,
        # while the free heuristic's 960-token code estimate had headroom
        # for both. An external estimate that undercuts the heuristic must
        # never leave the caller worse off than not using it at all -- per
        # CLAUDE.md's conservative default ("uncertain output size -> widen
        # max_tokens, not narrow it"), so when a classified task type also
        # has a heuristic value, the *wider* of the two wins, not whichever
        # source happened to run.
        external_estimate = config.get("external_budget_estimate")
        task_type = request.meta.task_type
        heuristic_available = task_type is not None and task_type in _BASE_MAX_TOKENS

        if external_estimate is not None and heuristic_available:
            external_based = round(external_estimate * _SAFETY_MARGIN)
            heuristic_based = round(_BASE_MAX_TOKENS[task_type] * _SAFETY_MARGIN)  # type: ignore[index]
            estimate = max(external_based, heuristic_based)
            source = "external_estimate" if external_based >= heuristic_based else "task_type_estimate"
        elif external_estimate is not None:
            estimate = round(external_estimate * _SAFETY_MARGIN)
            source = "external_estimate"
        elif heuristic_available:
            estimate = round(_BASE_MAX_TOKENS[task_type] * _SAFETY_MARGIN)  # type: ignore[index]
            source = "task_type_estimate"
        else:
            return request

        truncation_stats: dict[str, float] = config.get("truncation_stats", {})
        truncation_rate = truncation_stats.get(task_type) if task_type is not None else None
        widened_for_truncation = False
        if truncation_rate is not None and truncation_rate > _TRUNCATION_RATE_THRESHOLD:
            estimate = round(estimate * _TRUNCATION_WIDEN_FACTOR)
            widened_for_truncation = True

        if estimate <= request.max_tokens:
            return request  # never narrow below the caller's own value

        reason = "truncation_rate" if widened_for_truncation else source
        self.note(max_tokens_widened_to=estimate, max_tokens_widen_reason=reason)
        return replace(request, max_tokens=estimate)

    def _apply_thinking_effort(
        self, request: TokeneticsRequest, config: StageConfig
    ) -> TokeneticsRequest:
        if not config.get("enable_thinking_effort", False):
            # Opt-in only (see module docstring: a real $-cost benchmark
            # showed the old automatic default made some task types 2-3x
            # more expensive with no caller signal requesting that trade).
            self.note(thinking_effort_skipped="opt_in_not_enabled")
            return request

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
