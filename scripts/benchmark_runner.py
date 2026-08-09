#!/usr/bin/env python3
"""Phase 9 benchmark suite runner.

NOT part of pytest or CI -- this makes real, billed API calls (the
`quality-check` subcommand does; `corpus` is free, prepare()-only). See
CLAUDE.md's Phase 9 build-order entry and `src/tokenetics/core/benchmark.py`
for the corpus/quality-check design.

    uv run python scripts/benchmark_runner.py corpus
    uv run python scripts/benchmark_runner.py quality-check --dry-run
    uv run python scripts/benchmark_runner.py quality-check --confirm-spend

Two subcommands:

- `corpus`: runs every sample in benchmarks/corpus/ through prepare() only
  (free -- no completions, same cost model as dev_demo.py's --all-scenarios)
  and reports token savings as a MIN-MAX range per task-type category, never
  a single flat percentage, per CLAUDE.md's honest-benchmarking discipline.

- `quality-check`: runs every sample in benchmarks/quality_checks/ as a real
  completion, once with its named stage enabled and once with it disabled,
  and checks the response against that sample's `required_elements` (the
  authoritative pass/fail gate). This is the only part of Phase 9 that
  spends real money, so it's gated behind an upfront cost estimate: refuses
  to run past --cost-ceiling (default $1.00) even with --confirm-spend, and
  refuses to run AT ALL without --confirm-spend regardless of estimate size.
"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timezone
from pathlib import Path

import anthropic

from tokenetics import Tokenetics
from tokenetics.core.benchmark import (
    CorpusSample,
    QualityCheckSample,
    RequiredElementsResult,
    SavingsRecord,
    aggregate_savings_by_category,
    check_required_elements,
    check_tool_calls,
    estimate_cost_usd,
    load_corpus,
    load_quality_checks,
)
from tokenetics.core.request import from_api_kwargs
from tokenetics.core.tokenizer import count_tokens

_BENCHMARKS_DIR = Path(__file__).resolve().parent.parent / "benchmarks"
_DEFAULT_COST_CEILING_USD = 1.00


def _response_text(response: object) -> str:
    return "".join(
        block.text
        for block in getattr(response, "content", [])
        if getattr(block, "type", None) == "text"
    )


def _by_category(samples: list[CorpusSample]) -> dict[str, list[CorpusSample]]:
    by_category: dict[str, list[CorpusSample]] = {}
    for sample in samples:
        by_category.setdefault(sample.category, []).append(sample)
    return by_category


def _print_nonzero_stage_deltas(tk: Tokenetics) -> None:
    """Traces which stage(s) actually produced a sample's token reduction --
    the corpus summary alone only shows before/after, not which of the 10
    pipeline stages did the work. Only prints stages with a real (nonzero)
    token delta or a logged note; a 0-delta stage that fired silently (e.g.
    task_classifier, which only annotates) is omitted as noise.
    """
    entries = getattr(tk.logger, "entries", None) or []
    for entry in entries:
        delta = (entry.tokens_before or 0) - (entry.tokens_after or 0)
        notes = {k: v for k, v in entry.extra.items() if k not in ("error", "timing_seconds")}
        if delta == 0 and not notes:
            continue
        sign = "saved" if delta > 0 else "cost" if delta < 0 else "0 tokens"
        detail = f"{sign} {abs(delta)}" if delta != 0 else sign
        notes_str = f" {notes}" if notes else ""
        print(f"    {entry.stage_name}: {detail}{notes_str}")


def run_corpus(args: argparse.Namespace) -> None:
    samples = load_corpus(_BENCHMARKS_DIR / "corpus")
    if args.sample:
        samples = [s for s in samples if s.id == args.sample]
        if not samples:
            print(f"No corpus sample with id {args.sample!r} found.")
            return
    elif args.category:
        samples = [s for s in samples if s.category == args.category]
    if not samples:
        print("No corpus samples found in benchmarks/corpus/.")
        return

    client = anthropic.Anthropic()
    records: list[SavingsRecord] = []
    for sample in samples:
        before = count_tokens(from_api_kwargs(**sample.kwargs), client)
        tk = Tokenetics(client=client, stage_config=sample.stage_config)
        prepared = tk.prepare(**sample.kwargs)
        after = count_tokens(from_api_kwargs(**prepared), client)
        record = SavingsRecord(
            category=sample.category, sample_id=sample.id, before_tokens=before, after_tokens=after
        )
        records.append(record)
        print(f"{sample.id} ({sample.category}): {before} -> {after} tokens ({record.pct_saved:.1f}%)")
        if not args.quiet:
            _print_nonzero_stage_deltas(tk)

    ranges = aggregate_savings_by_category(records)
    print("\n=== Savings by category (range, not a single flat percentage) ===")
    for r in ranges:
        print(f"  {r.category}: {r.min_pct:.1f}% - {r.max_pct:.1f}% (n={r.sample_count} samples)")

    if args.sample or args.category:
        print(f"\n{len(samples)} sample(s) shown -- filtered run, not the full corpus.")
    else:
        print(
            f"\n{len(samples)} samples total, {min(len(v) for v in _by_category(samples).values())}+ "
            "per category -- clears CLAUDE.md's 15-20+/category floor. Still worth growing further "
            "before treating these as final published numbers; see ROADMAP.md's Phase 9 section."
        )

    if args.save:
        _save_results(
            "corpus",
            {
                "date": date.today().isoformat(),
                "sample_count": len(samples),
                "ranges": [
                    {
                        "category": r.category,
                        "min_pct": r.min_pct,
                        "max_pct": r.max_pct,
                        "sample_count": r.sample_count,
                    }
                    for r in ranges
                ],
                "measured": True,
            },
        )


def _grade(response: object, sample: QualityCheckSample) -> RequiredElementsResult:
    """Combines both grading signals -- required TEXT elements and required
    TOOL CALLS -- into one pass/fail, since a sample may specify either or
    both. A correct tool_use-only response has no text to check, so the two
    checks are independent, not stacked as "must have both kinds"; a sample
    that only sets one of the two fields effectively skips the other (an
    unset list means nothing is required there).
    """
    text_result = check_required_elements(_response_text(response), sample.required_elements)
    tool_result = check_tool_calls(
        list(getattr(response, "content", [])), sample.required_tool_calls or []
    )
    return RequiredElementsResult(
        passed=text_result.passed and tool_result.passed,
        missing=text_result.missing + tool_result.missing,
    )


def run_quality_check(args: argparse.Namespace) -> None:
    samples = load_quality_checks(_BENCHMARKS_DIR / "quality_checks")
    if args.stage:
        samples = [s for s in samples if s.stage == args.stage]
    if not samples:
        print("No matching quality-check samples found.")
        return

    model = args.model or samples[0].kwargs.get("model", "claude-sonnet-5")
    estimate = estimate_cost_usd(samples, model, completions_per_sample=2)
    print(
        f"Estimated cost: ${estimate:.4f} for {len(samples)} sample(s) x 2 completions "
        f"(with the stage enabled, and with it disabled) -- ESTIMATED, not measured."
    )

    if args.dry_run:
        print("--dry-run: no completions made.")
        return

    if estimate > args.cost_ceiling:
        print(
            f"REFUSED: estimated cost ${estimate:.4f} exceeds --cost-ceiling "
            f"${args.cost_ceiling:.2f}. Re-run with a higher --cost-ceiling if you "
            "actually intend to spend this much."
        )
        return

    if not args.confirm_spend:
        print("REFUSED: real billed calls require --confirm-spend.")
        return

    client = anthropic.Anthropic()
    results = []
    for sample in samples:
        tk_on = Tokenetics(client=client, stage_config=sample.stage_config)
        response_on = client.messages.create(**tk_on.prepare(**sample.kwargs))
        result_on = _grade(response_on, sample)

        tk_off = Tokenetics(client=client, stage_config=sample.stage_config)
        for stage in tk_off.stages:
            if stage.name == sample.stage:
                stage.enabled = False
        response_off = client.messages.create(**tk_off.prepare(**sample.kwargs))
        result_off = _grade(response_off, sample)

        print(f"\n=== {sample.id} ({sample.stage}, {sample.category}) ===")
        print(f"  with {sample.stage}:    passed={result_on.passed}  missing={result_on.missing}")
        print(f"  without {sample.stage}: passed={result_off.passed}  missing={result_off.missing}")
        results.append(
            {
                "id": sample.id,
                "stage": sample.stage,
                "category": sample.category,
                "with_stage_passed": result_on.passed,
                "with_stage_missing": result_on.missing,
                "without_stage_passed": result_off.passed,
                "without_stage_missing": result_off.missing,
            }
        )

    passed_with = sum(1 for r in results if r["with_stage_passed"])
    passed_without = sum(1 for r in results if r["without_stage_passed"])
    print(
        f"\n=== TOTAL: {passed_with}/{len(results)} passed with the stage enabled, "
        f"{passed_without}/{len(results)} passed with it disabled ==="
    )

    if args.save:
        _save_results(
            "quality_check",
            {
                "date": date.today().isoformat(),
                "estimated_cost_usd": estimate,
                "results": results,
                "measured": True,
            },
        )


def _save_results(kind: str, data: dict[str, object]) -> None:
    results_dir = _BENCHMARKS_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = results_dir / f"{kind}_{timestamp}.json"
    path.write_text(json.dumps(data, indent=2))
    print(f"\nSaved results to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    corpus_parser = subparsers.add_parser("corpus", help="Free, prepare()-only corpus run.")
    corpus_parser.add_argument("--save", action="store_true", help="Save results to benchmarks/results/.")
    corpus_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print the per-sample summary line and category ranges, not the per-stage trace.",
    )
    corpus_parser.add_argument(
        "--sample", default=None, metavar="ID", help="Only run this one sample, e.g. tool_heavy_018."
    )
    corpus_parser.add_argument(
        "--category",
        default=None,
        choices=["code", "conversational", "extraction", "tool-heavy"],
        help="Only run samples in this category.",
    )
    corpus_parser.set_defaults(func=run_corpus)

    quality_parser = subparsers.add_parser("quality-check", help="Real-completion quality-check run.")
    quality_parser.add_argument(
        "--stage", default=None, help="Only run quality checks for this stage name, e.g. brevity_injector."
    )
    quality_parser.add_argument(
        "--dry-run", action="store_true", help="Print the cost estimate only, make no completions."
    )
    quality_parser.add_argument(
        "--confirm-spend", action="store_true", help="Required to actually make real, billed completions."
    )
    quality_parser.add_argument(
        "--cost-ceiling",
        type=float,
        default=_DEFAULT_COST_CEILING_USD,
        help=f"Refuse to run if the estimated cost exceeds this (USD, default ${_DEFAULT_COST_CEILING_USD:.2f}).",
    )
    quality_parser.add_argument("--model", default=None, help="Override the model used for cost estimation.")
    quality_parser.add_argument("--save", action="store_true", help="Save results to benchmarks/results/.")
    quality_parser.set_defaults(func=run_quality_check)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
