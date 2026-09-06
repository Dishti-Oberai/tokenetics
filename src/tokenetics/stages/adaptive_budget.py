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

**Bounded-shape effort downgrade (added 2026-09-06)**: this sub-stage's
task-type mapping is still a coarse proxy for "how much reasoning does
this actually need" -- `code` always gets `"high"` effort whether the
question is a one-liner or a genuine design problem, the same
classifier-category-too-broad issue `brevity_injector`'s AGGRESSIVE bucket
ran into (see that module's docstring). `request.meta.bounded_shape`
(the same signal `brevity_injector` uses, computed by `task_classifier`)
downgrades the task-type effort by one level -- `high`->`medium`,
`medium`->`low`, `low` stays `low` -- for a short, single-question,
self-contained request. This is a genuine reduction in real thinking-token
*generation* (`output_config.effort` is the only lever the Anthropic API
exposes for that with `type: "adaptive"` thinking; there's no numeric
budget for this thinking type the way legacy `budget_tokens` had).

**Bounded-shape default, not opt-in-only (revised 2026-09-06, same day)**:
once the downgrade above was real-validated -- 7/7 quality-check samples
clean across all 4 task types (two risk-case samples each in `code`/
`conversational`, specifically chosen because their SHORT shape could
tempt under-resourcing: a bug whose failure is silent rather than a loud
exception, the bat-and-ball and 5-machines cognitive-reflection-test
questions famous for tricking reasoners into a fast wrong answer), AND
real dollar evidence across all 4 task types (`adaptive_budget`'s own
contribution measured positive in every one -- see ROADMAP.md's "real
dollar-cost evidence" section for the numbers) -- `enable_thinking_effort`
is no longer required for a BOUNDED request specifically: `meta.
bounded_shape=True` alone is now sufficient to engage thinking-effort, ALWAYS
at the downgraded level. `config["enable_thinking_effort"]=True` remains
available and still means what it always did (applies task-type effort to
ANY classified request, bounded or not) -- it's still required for an
UNBOUNDED or ambiguous-shape request, which is exactly where the original
2-3x cost regression was measured (that benchmark mixed bounded and
unbounded content at full, undowngraded task-type effort; this default
only ever fires on the narrower, real-evidence-backed slice). `self.note()`
tags `thinking_effort_source` as `"bounded_shape_default"` vs `"opt_in"` so
the cost logger can tell the two paths apart.
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

# Downgrade one level when meta.bounded_shape is True (added 2026-09-06,
# per the user asking for a real technique to reduce thinking-token
# GENERATION, not just track its cost -- see module docstring's new
# section below). "low" has nowhere lower to go.
_EFFORT_DOWNGRADE = {"high": "medium", "medium": "low", "low": "low"}


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
        opted_in = config.get("enable_thinking_effort", False)
        bounded_shape = bool(request.meta.bounded_shape)

        # Bounded-shape default (added 2026-09-06, after real quality-check
        # validation across all 4 task types (7/7 clean, two risk-case
        # samples each in code/conversational) AND real dollar evidence
        # across all 4 (adaptive_budget's own contribution positive in
        # every one -- see ROADMAP.md's "real dollar-cost evidence"
        # section): a bounded/simple request gets thinking-effort even
        # without the explicit opt-in, ALWAYS at the downgraded level.
        # Everything else -- unbounded, or ambiguous shape -- still
        # requires the explicit opt-in below, unchanged. This is narrower
        # than the old automatic-by-task-type default that caused the
        # original 2-3x cost regression: that measured the WHOLE corpus
        # (bounded and unbounded mixed) at full task-type effort; this
        # only ever fires on the specific validated slice, always at the
        # downgraded level.
        if not opted_in and not bounded_shape:
            self.note(thinking_effort_skipped="opt_in_not_enabled")
            return request

        if "thinking" in request.extra:
            return request  # respect the caller's own explicit thinking config

        effort = _EFFORT_BY_TASK_TYPE.get(request.meta.task_type or "")
        if effort is None:
            return request  # unclassified, or no mapping for this type -- leave untouched

        # A bounded/simple question doesn't need task-type's default depth
        # of reasoning regardless of category -- "code" always mapping to
        # "high" effort wastes real thinking tokens on a trivial one-liner
        # the same way it would on a genuine multi-file design question.
        # Downgraded, not skipped entirely: still gets SOME effort, just
        # not more than the shape actually calls for.
        if bounded_shape:
            effort = _EFFORT_DOWNGRADE[effort]

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

        self.note(
            thinking_effort=effort,
            thinking_effort_source="opt_in" if opted_in else "bounded_shape_default",
        )
        new_extra: dict[str, Any] = dict(request.extra)
        new_extra["thinking"] = {"type": "adaptive", "display": "omitted"}
        new_extra["output_config"] = {
            **request.extra.get("output_config", {}),
            "effort": effort,
        }
        return replace(request, extra=new_extra)
