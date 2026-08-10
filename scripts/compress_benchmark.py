#!/usr/bin/env python3
"""Phase 10b benchmark: compress's quality-degradation curve (reduction
ratio vs. accuracy), against a held-out passage+question set
(`benchmarks/compress_quality/passages.json`).

NOT part of pytest or CI -- this makes real, billed API calls, same
convention as `benchmark_runner.py quality-check`. Gated the same way:
an upfront cost estimate always prints first, refuses to run past
--cost-ceiling (default $1.00) even with --confirm-spend, and refuses to
run at all without --confirm-spend regardless of estimate size.

    uv run python scripts/compress_benchmark.py --dry-run
    uv run python scripts/compress_benchmark.py --confirm-spend

For each sample and each compression ratio in `_RATIOS` (0.0 = uncompressed
baseline; the rest cluster tightly between 0.4 and 0.6, 2026-08-10's known
"still fine" and "clearly broken" points from the first real run, to locate
the actual cliff rather than just its two endpoints), compresses the
sample's passage locally (free, `tokenetics.extras.compress`), sends ONE
real completion with the (possibly compressed) passage + question, and
grades the response against `required_elements` (reusing Phase 9's
authoritative substring/OR-group gate from `core/benchmark.py` -- same
grading contract, different corpus). Reports pass rate per ratio -- the
reduction-vs-accuracy curve CLAUDE.md's build order calls for -- plus the
real achieved word-reduction at each ratio, so the curve is measured
against real compression output, not the nominal ratio requested.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import anthropic

from tokenetics.core.benchmark import check_required_elements, response_text_with_tool_inputs
from tokenetics.core.cache_pricing import base_input_price_for, output_price_for
from tokenetics.extras.compress import Compressor

_DATASET_PATH = Path(__file__).resolve().parent.parent / "benchmarks" / "compress_quality" / "passages.json"
_RATIOS = [0.0, 0.4, 0.45, 0.5, 0.55, 0.6]
_DEFAULT_COST_CEILING_USD = 1.00


def _estimate_cost_usd(samples: list[dict[str, Any]], model: str) -> float:
    input_price = base_input_price_for(model) / 1_000_000
    output_price = output_price_for(model) / 1_000_000
    total = 0.0
    for sample in samples:
        input_tokens = len(sample["passage"]) // 4 + len(sample["question"]) // 4
        output_tokens = sample["max_tokens"] * 0.5
        cost_per_completion = input_tokens * input_price + output_tokens * output_price
        total += cost_per_completion * len(_RATIOS)
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Print the cost estimate and exit -- no completions.")
    parser.add_argument("--confirm-spend", action="store_true", help="Required to actually run (real billed calls).")
    parser.add_argument("--cost-ceiling", type=float, default=_DEFAULT_COST_CEILING_USD)
    args = parser.parse_args()

    samples = json.loads(_DATASET_PATH.read_text())
    model = samples[0]["model"]
    estimate = _estimate_cost_usd(samples, model)
    print(
        f"Estimated cost: ${estimate:.4f} for {len(samples)} sample(s) x {len(_RATIOS)} ratios "
        f"({len(samples) * len(_RATIOS)} completions) -- ESTIMATED, not measured."
    )

    if args.dry_run:
        print("--dry-run: no completions made.")
        return
    if estimate > args.cost_ceiling:
        print(f"REFUSED: estimated cost ${estimate:.4f} exceeds --cost-ceiling ${args.cost_ceiling:.2f}.")
        return
    if not args.confirm_spend:
        print("REFUSED: real billed calls require --confirm-spend.")
        return

    client = anthropic.Anthropic()
    compressor = Compressor()
    # {ratio: [(sample_id, passed, ratio_achieved), ...]}
    results: dict[float, list[tuple[str, bool, float]]] = {r: [] for r in _RATIOS}

    for sample in samples:
        print(f"\n=== {sample['id']} ===")
        for ratio in _RATIOS:
            if ratio == 0.0:
                passage_text = sample["passage"]
                ratio_achieved = 0.0
            else:
                # This benchmark exists specifically to test past the
                # default cap and locate the cliff -- must opt out of the
                # clamp, or every ratio above RECOMMENDED_MAX_RATIO would
                # silently collapse to testing 0.4 repeatedly.
                compress_result = compressor.compress(
                    sample["passage"], ratio, allow_above_recommended_max=True
                )
                passage_text = compress_result.compressed_text
                ratio_achieved = compress_result.ratio_achieved
                if not compress_result.success:
                    print(f"  ratio={ratio}: compression failed, falling back to baseline text")

            prompt = f"{passage_text}\n\nQuestion: {sample['question']}"
            response = client.messages.create(
                model=sample["model"],
                max_tokens=sample["max_tokens"],
                messages=[{"role": "user", "content": prompt}],
            )
            text = response_text_with_tool_inputs(list(getattr(response, "content", [])))
            grade = check_required_elements(text, sample["required_elements"])
            results[ratio].append((sample["id"], grade.passed, ratio_achieved))
            status = "PASS" if grade.passed else f"FAIL missing={grade.missing}"
            print(f"  ratio={ratio} (achieved {ratio_achieved:.2f}): {status}")
            if not grade.passed:
                # Same lesson as Phase 9's adaptive_budget_001: a failure
                # with no visible text is unfalsifiable -- can't tell a real
                # compression-caused content loss from a rubric/phrasing
                # mismatch without seeing both the actual input the model
                # saw and what it actually said.
                for element in grade.missing:
                    # `grade.missing` entries are always plain strings --
                    # check_required_elements already joins OR-groups (e.g.
                    # "2-second OR 2 second") before returning them here.
                    alternatives = element.split(" OR ")
                    in_passage = any(alt.lower() in passage_text.lower() for alt in alternatives)
                    print(f"    missing element {element!r} -- present in (possibly compressed) passage sent: {in_passage}")
                print(f"    passage sent: {passage_text!r}")
                print(f"    response text: {text!r}")

    print("\n=== Reduction-vs-accuracy curve (measured) ===")
    for ratio in _RATIOS:
        entries = results[ratio]
        passed = sum(1 for _, p, _ in entries if p)
        avg_achieved = sum(r for _, _, r in entries) / len(entries) if entries else 0.0
        print(
            f"  requested ratio={ratio:.2f} (avg real word-reduction {avg_achieved:.1%}): "
            f"{passed}/{len(entries)} passed ({passed / len(entries):.1%})"
        )


if __name__ == "__main__":
    main()
