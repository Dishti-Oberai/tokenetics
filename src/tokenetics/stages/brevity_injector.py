"""Stage 9b: response-budget injector (brevity instructions).

Injects one of three instruction buckets into the system prompt based on
task type. Conservative default is OFF (per CLAUDE.md: "ambiguous task
shape -> skip the brevity instruction, bucket = OFF") -- brevity is only
ever turned on for a task type where terseness is clearly appropriate.

Only `extraction` maps to MODERATE today. The classifier's four categories
(code, conversational, extraction, tool-heavy) don't give a reliable signal
for AGGRESSIVE's "verifiable short answer with minimal reasoning steps"
shape -- `code` covers everything from a one-line fix to a multi-file
design question, and `conversational` covers everything from "what's the
capital of France" to "explain photosynthesis." AGGRESSIVE stays defined
below for when finer task-shape detection exists, but nothing auto-selects
it yet; that's a deliberate scope decision, not an oversight.

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


class BrevityInjectorStage(Stage):
    name = "brevity_injector"

    def run(
        self, request: TokeneticsRequest, config: StageConfig, logger: CostLogger
    ) -> TokeneticsRequest:
        instruction = _TASK_TYPE_BUCKET.get(request.meta.task_type or "")
        if instruction is None:
            return request

        self.note(brevity_bucket="moderate")
        new_system = f"{request.system}\n\n{instruction}" if request.system else instruction
        return replace(request, system=new_system)
