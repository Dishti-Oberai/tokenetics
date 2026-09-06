"""Stage 9b: response-budget injector (brevity instructions).

Injects one of three instruction buckets into the system prompt based on
task type. Conservative default is OFF (per CLAUDE.md: "ambiguous task
shape -> skip the brevity instruction, bucket = OFF") -- brevity is only
ever turned on for a task type where terseness is clearly appropriate.

`extraction` maps to MODERATE by default. `code`, `conversational`, and
`tool-heavy` default to AGGRESSIVE -- but ONLY when `request.meta.
bounded_shape` is True (see task_classifier.py's `_is_bounded_shape`).
task_type alone was never a reliable signal for AGGRESSIVE's "verifiable
short answer with minimal reasoning steps" shape -- `code` covers
everything from a one-line fix to a multi-file design question, and
`conversational` covers everything from "what's the capital of France" to
"explain photosynthesis." Real validation (2026-09-06, benchmarks/
quality_checks/brevity_injector.json's `quality_brevity_aggressive_*`
samples, 11/11 clean, plus real output-token A/Bs) covered short,
single-question, self-contained requests across code/conversational/
tool-heavy -- never the open-ended end of any category. `bounded_shape` is
the classifier's attempt to isolate exactly that validated slice, so the
default only fires where there's real evidence, not for the whole category.
`extraction` stays MODERATE-only even when bounded -- its real economics
came out roughly break-even (20-32 output tokens saved, no consistent net
win), so it was deliberately left out of the default rather than added on
the same "bounded" basis as the other three.

`config["aggressive_for_task_types"]` remains as an explicit opt-in for
task types the default doesn't cover (e.g. `extraction`) -- same "trades
cost for a different risk, needs explicit choice" pattern this project
uses for `adaptive_budget`'s thinking-effort knob. Note that the
`bounded_shape` gate applies uniformly, including to this opt-in list:
an explicitly opted-in task type still only gets AGGRESSIVE on a bounded
request, since the risk `bounded_shape` guards against (terse pressure on
genuinely open-ended content) doesn't go away just because a caller opted
in.

Instruction overhead (~50-100 tokens) is a real cost counted honestly by
the orchestrator's before/after token measurement, not hidden.
"""

from __future__ import annotations

from dataclasses import replace

from tokenetics.core.logger import CostLogger
from tokenetics.core.plugin import Stage, StageConfig
from tokenetics.core.request import TokeneticsRequest

MODERATE_INSTRUCTION = (
    "Skip any preamble, restatement of the question, or closing summary -- answer directly."
)
AGGRESSIVE_INSTRUCTION = (
    "Use minimal reasoning steps (about 5 words each), no filler, no restating the problem "
    "-- Chain-of-Draft style."
)

_TASK_TYPE_BUCKET = {
    "extraction": MODERATE_INSTRUCTION,
}

# The narrow slice AGGRESSIVE has real, both-ways-validated evidence for --
# "extraction" is deliberately excluded (break-even real economics, see
# module docstring). Gated further per-request by meta.bounded_shape.
_DEFAULT_AGGRESSIVE_TASK_TYPES = {"code", "conversational", "tool-heavy"}


class BrevityInjectorStage(Stage):
    name = "brevity_injector"

    def run(
        self, request: TokeneticsRequest, config: StageConfig, logger: CostLogger
    ) -> TokeneticsRequest:
        task_type = request.meta.task_type or ""
        bounded_shape = bool(request.meta.bounded_shape)
        aggressive_task_types = _DEFAULT_AGGRESSIVE_TASK_TYPES | set(
            config.get("aggressive_for_task_types", [])
        )

        instruction: str | None
        if bounded_shape and task_type in aggressive_task_types:
            instruction = AGGRESSIVE_INSTRUCTION
            bucket = "aggressive"
        else:
            instruction = _TASK_TYPE_BUCKET.get(task_type)
            bucket = "moderate"

        if instruction is None:
            return request

        self.note(brevity_bucket=bucket)
        new_system = f"{request.system}\n\n{instruction}" if request.system else instruction
        return replace(request, system=new_system)
