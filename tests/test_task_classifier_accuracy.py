"""Phase 3 accuracy gate.

CLAUDE.md's build order requires the task classifier to clear an accuracy
bar on a hand-labeled validation set before anything downstream (stage 4's
tool-relevance filtering, stage 9's structured-output/brevity/adaptive
budgets) is allowed to consume its output. This test is that gate.

_ACCURACY_THRESHOLD = 0.85, confirmed with the user during Phase 3 planning.
"""

from __future__ import annotations

import json
from pathlib import Path

from tokenetics.core.logger import InMemoryCostLogger
from tokenetics.core.request import from_api_kwargs
from tokenetics.stages.task_classifier import TaskClassifierStage

_ACCURACY_THRESHOLD = 0.85
_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "classifier_validation_set.json"


def _load_validation_set() -> list[dict]:
    return json.loads(_FIXTURE_PATH.read_text())


def test_validation_set_has_enough_hand_labeled_examples():
    examples = _load_validation_set()
    assert 30 <= len(examples) <= 50


def test_classifier_clears_the_accuracy_gate():
    examples = _load_validation_set()
    logger = InMemoryCostLogger()
    correct = 0
    mismatches = []

    for example in examples:
        request = from_api_kwargs(
            model="claude-sonnet-5", max_tokens=100, messages=example["messages"]
        )
        result = TaskClassifierStage().run(request, {}, logger)
        if result.meta.task_type == example["expected"]:
            correct += 1
        else:
            mismatches.append((example["expected"], result.meta.task_type))

    accuracy = correct / len(examples)
    assert accuracy >= _ACCURACY_THRESHOLD, (
        f"accuracy {accuracy:.2%} below {_ACCURACY_THRESHOLD:.0%} gate; mismatches: {mismatches}"
    )
