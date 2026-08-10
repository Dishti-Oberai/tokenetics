"""Phase 9 benchmark suite: corpus/quality-check loading, cost estimation,
range aggregation, and rubric grading.

Two separate sample sets, per the brief:

- **Corpus** (`benchmarks/corpus/*.json`) -- organized by the task
  classifier's own categories, used to measure real token savings. This is
  entirely FREE to run: `Tokenetics.prepare()` + `count_tokens` never calls
  `messages.create()`, same as `dev_demo.py`'s `--all-scenarios`.
- **Quality-check set** (`benchmarks/quality_checks/*.json`) -- a separate,
  held-out set for risk-bearing stages (brevity, cap tuning, thinking-effort
  tuning, pruning, delta compression). This DOES need real completions (on
  vs. off, to compare actual reply quality), which is why it's the only part
  of this phase gated behind a cost estimate and an explicit confirmation
  flag -- see `estimate_cost_usd`.

Quality grading (resolved with the user 2026-08-02): a hand-written rubric's
`required_elements` (facts/keywords that must appear in the response) is the
authoritative, objective pass/fail gate. An LLM-judge score against
`judge_rubric` is logged alongside as a secondary signal, but never solely
determines a published quality claim -- grading Claude's output by asking
Claude is a reasonable cheap secondary check, but using it as the *sole*
authority in exactly the risk-bearing cases this set exists to catch would
undercut the whole point.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tokenetics.core.cache_pricing import base_input_price_for, output_price_for


@dataclass
class CorpusSample:
    id: str
    category: str
    description: str
    kwargs: dict[str, Any]
    # Some stages (cache_breakpoint_optimizer's cache_usage_history,
    # delta_compression's previous_payloads) can't fire at all without a
    # per-call config a cold one-shot corpus request doesn't otherwise
    # supply. Optional, defaults to none -- most samples don't need it.
    stage_config: dict[str, dict[str, Any]] | None = None


@dataclass
class QualityCheckSample:
    id: str
    stage: str
    category: str
    description: str
    kwargs: dict[str, Any]
    required_elements: list[str | list[str]]
    judge_rubric: str
    # Some stages (context_scheduler's token_budget) can't be exercised
    # meaningfully without a per-call config -- optional, defaults to none.
    stage_config: dict[str, dict[str, Any]] | None = None
    # For samples where the CORRECT response is a tool_use call, not text
    # (e.g. tool-relevance filtering incorrectly removing a tool the model
    # genuinely needs to invoke again) -- `required_elements` alone can't
    # express this, since a correct tool_use-only response has no text to
    # check at all. Optional, defaults to none.
    required_tool_calls: list[str] | None = None


def _load_json_files(directory: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        records.extend(json.loads(path.read_text()))
    return records


def load_corpus(directory: Path) -> list[CorpusSample]:
    return [
        CorpusSample(
            id=r["id"],
            category=r["category"],
            description=r["description"],
            kwargs=r["kwargs"],
            stage_config=r.get("stage_config"),
        )
        for r in _load_json_files(directory)
    ]


def load_quality_checks(directory: Path) -> list[QualityCheckSample]:
    return [
        QualityCheckSample(
            id=r["id"],
            stage=r["stage"],
            category=r["category"],
            description=r["description"],
            kwargs=r["kwargs"],
            required_elements=r["required_elements"],
            judge_rubric=r["judge_rubric"],
            stage_config=r.get("stage_config"),
            required_tool_calls=r.get("required_tool_calls"),
        )
        for r in _load_json_files(directory)
    ]


# Same fast, local, ~4-chars/token estimate used elsewhere internally
# (context scheduler, cache breakpoint optimizer) for sizing before a real
# tokenizer call is available/warranted -- fine for a pre-run cost ESTIMATE
# (explicitly not tagged `measured`), not for anything logged as a real count.
def _estimate_tokens(kwargs: dict[str, Any]) -> int:
    text = json.dumps(kwargs)
    return max(1, len(text) // 4)


def estimate_cost_usd(
    samples: list[QualityCheckSample], model: str, completions_per_sample: int = 2
) -> float:
    """Rough pre-run cost estimate for the quality-check set (the only part
    of this phase that spends real money) -- `completions_per_sample=2` for
    the default on/off comparison; callers doing on/off + LLM-judge grading
    should pass a higher count. Assumes each completion's output is roughly
    half the size of `max_tokens` (a rough, deliberately conservative-ish
    planning assumption, not a promise) -- this is an ESTIMATE for a
    pre-flight cost check, never logged as `measured`.
    """
    input_price = base_input_price_for(model) / 1_000_000
    output_price = output_price_for(model) / 1_000_000
    total = 0.0
    for sample in samples:
        input_tokens = _estimate_tokens(sample.kwargs)
        output_tokens = sample.kwargs.get("max_tokens", 500) * 0.5
        cost_per_completion = input_tokens * input_price + output_tokens * output_price
        total += cost_per_completion * completions_per_sample
    return total


@dataclass
class SavingsRecord:
    category: str
    sample_id: str
    before_tokens: int
    after_tokens: int

    @property
    def pct_saved(self) -> float:
        if self.before_tokens == 0:
            return 0.0
        return (self.before_tokens - self.after_tokens) / self.before_tokens * 100


@dataclass
class CategoryRange:
    category: str
    min_pct: float
    max_pct: float
    sample_count: int


def aggregate_savings_by_category(records: list[SavingsRecord]) -> list[CategoryRange]:
    """Groups per-sample savings by category and reports a MIN-MAX range per
    category, never a single flat percentage -- per CLAUDE.md's honest-
    benchmarking discipline ("report savings as ranges tied to a stated
    workload assumption"). A category with just one sample still reports a
    (degenerate, min==max) range rather than special-casing it away, so the
    caller can see the sample count and judge confidence themselves.
    """
    by_category: dict[str, list[float]] = {}
    for record in records:
        by_category.setdefault(record.category, []).append(record.pct_saved)

    return [
        CategoryRange(
            category=category,
            min_pct=min(pcts),
            max_pct=max(pcts),
            sample_count=len(pcts),
        )
        for category, pcts in sorted(by_category.items())
    ]


@dataclass
class RequiredElementsResult:
    passed: bool
    missing: list[str] = field(default_factory=list)


def check_required_elements(
    response_text: str, required_elements: list[str | list[str]]
) -> RequiredElementsResult:
    """The authoritative, objective pass/fail gate for a quality-check
    sample -- case-insensitive substring presence, nothing fuzzier. An LLM
    judge score is a secondary signal (see module docstring), not this.

    An entry may be a plain string (must appear verbatim) or a list of
    strings (an OR-group -- any one alternate phrasing is enough). Added
    2026-08-10 after a real quality-check run: `quality_adaptive_budget_001`
    failed identically with/without the stage under test because two
    genuinely-correct responses described the same fact two different ways
    ("only one unique value" vs. "only 1 element"), with `judge_rubric`
    confirmed unwired to any actual grading call -- so exact-substring
    matching was, in practice, the sole gate, and too rigid for any
    required element that isn't a fixed technical term. Plain-string
    entries are unaffected, so every pre-existing sample keeps its exact
    prior behavior.
    """
    lowered = response_text.lower()
    missing: list[str] = []
    for el in required_elements:
        if isinstance(el, list):
            if not any(alt.lower() in lowered for alt in el):
                missing.append(" OR ".join(el))
        elif el.lower() not in lowered:
            missing.append(el)
    return RequiredElementsResult(passed=not missing, missing=missing)


def response_text_with_tool_inputs(response_content: list[Any]) -> str:
    """Text used for `check_required_elements` grading -- includes both
    visible text AND any `tool_use` block's `input` (JSON-serialized).

    Caught via a real quality-check run (2026-08-09): every extraction-
    shaped sample failed `check_required_elements` identically with AND
    without the stage under test, which only makes sense if the check was
    looking in the wrong place -- structured_output forces `tool_choice`
    for extraction-shaped requests, so the model's real answer is the
    tool_use block's `input` (structured JSON arguments), not free text.
    A response-text-only check can never find "Jane Smith" sitting inside
    `{"name": "Jane Smith", ...}`. Not a stage regression; a benchmark-
    harness gap in the same family as the one `check_tool_calls` fixed for
    tool-use-only responses -- this one covers the "answer's content lives
    inside a tool call" case rather than the "response IS a tool call" case.
    """
    parts: list[str] = []
    for block in response_content:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            parts.append(getattr(block, "text", ""))
        elif block_type == "tool_use":
            parts.append(json.dumps(getattr(block, "input", {})))
    return "".join(parts)


def check_tool_calls(response_content: list[Any], required_tool_calls: list[str]) -> RequiredElementsResult:
    """Same authoritative-gate role as `check_required_elements`, but for
    samples where the CORRECT response is a `tool_use` call, not text --
    e.g. testing whether tool-relevance filtering incorrectly removed a
    tool the model genuinely needed to invoke again. A response can pass
    this AND fail `check_required_elements` (a correct tool_use-only reply
    has no text to check), so callers should treat the two as independent
    signals, not combine them into one pass/fail.
    """
    called = {
        block.name
        for block in response_content
        if getattr(block, "type", None) == "tool_use" and hasattr(block, "name")
    }
    missing = [name for name in required_tool_calls if name not in called]
    return RequiredElementsResult(passed=not missing, missing=missing)
