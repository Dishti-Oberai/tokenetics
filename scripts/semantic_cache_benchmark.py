#!/usr/bin/env python3
"""Phase 10c benchmark: semantic_cache's false-positive/recall rate at
scale, against a held-out near-miss set (`benchmarks/semantic_cache/
near_miss_set.json`).

NOT part of pytest or CI, same reasoning as `benchmark_runner.py` -- kept
separate from the automated suite even though (unlike `quality-check`) this
makes NO network calls and costs nothing: it needs `tokenetics[semantic-
cache]` installed (a real embedding model load + inference per query), and
per CLAUDE.md's Phase 9/10 convention, Tier 2 extras' benchmarks are their
own thing, not folded into the core test run.

    uv run python scripts/semantic_cache_benchmark.py

Loads every `stored_entries` item into one shared `SemanticCache` (a
realistic multi-topic FAQ-cache scenario, not one isolated pair at a time --
this also exercises cross-domain confusion, not just single-pair
similarity). For each `test_queries` item:

- `expected_hit_id` set -> a genuine paraphrase; correct outcome is a hit
  AND it must match the RIGHT stored entry (a hit on the wrong entry is a
  worse failure than a plain miss, and reported separately).
- `expected_hit_id` null -> a near-miss or unrelated query; correct outcome
  is a miss. Any hit here is a false positive -- the failure mode this
  whole extra's design exists to avoid.

Reports recall (fraction of genuine paraphrases correctly hit), false-
positive rate (fraction of should-miss queries that incorrectly hit), and
prints every individual miss/false-positive/wrong-match so a threshold
adjustment can be made from real evidence, not guessing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tokenetics.extras.semantic_cache import SemanticCache

_DATASET_PATH = Path(__file__).resolve().parent.parent / "benchmarks" / "semantic_cache" / "near_miss_set.json"


def main() -> None:
    dataset = json.loads(_DATASET_PATH.read_text())
    cache = SemanticCache()

    stored_by_id = {e["id"]: e for e in dataset["stored_entries"]}
    for entry in dataset["stored_entries"]:
        cache.store(entry["query"], entry["response"])
    print(f"Loaded {len(stored_by_id)} stored entries, threshold={cache.similarity_threshold}")

    should_hit = [q for q in dataset["test_queries"] if q["expected_hit_id"] is not None]
    should_miss = [q for q in dataset["test_queries"] if q["expected_hit_id"] is None]

    correct_hits = 0
    wrong_entry_hits: list[dict[str, Any]] = []
    missed_paraphrases: list[dict[str, Any]] = []
    print("\n=== Should-hit queries (genuine paraphrases) ===")
    for item in should_hit:
        result = cache.lookup(item["query"])
        expected_query = stored_by_id[item["expected_hit_id"]]["query"]
        sim = f"{result.similarity:.3f}" if result.similarity is not None else "n/a"
        if result.hit and result.matched_query == expected_query:
            correct_hits += 1
            status = "OK  "
        elif result.hit:
            wrong_entry_hits.append({**item, "matched": result.matched_query, "similarity": result.similarity})
            status = "WRONG_MATCH"
        else:
            missed_paraphrases.append({**item, "similarity": result.similarity})
            status = "MISS"
        print(f"  {status:12s} sim={sim}  {item['query']!r}")

    false_positives: list[dict[str, Any]] = []
    print("\n=== Should-miss queries (near-misses + unrelated) ===")
    for item in should_miss:
        result = cache.lookup(item["query"])
        sim = f"{result.similarity:.3f}" if result.similarity is not None else "n/a"
        if result.hit:
            false_positives.append({**item, "matched": result.matched_query, "similarity": result.similarity})
            status = "FALSE_POSITIVE"
        else:
            status = "OK  "
        print(f"  {status:14s} sim={sim}  {item['query']!r}  ({item['note']})")

    recall = correct_hits / len(should_hit) if should_hit else 0.0
    fp_rate = len(false_positives) / len(should_miss) if should_miss else 0.0

    print("\n=== Summary (measured, real embedding model, no mocks) ===")
    print(f"threshold: {cache.similarity_threshold}")
    print(f"recall (correct hits / genuine paraphrases): {correct_hits}/{len(should_hit)} = {recall:.1%}")
    print(f"false-positive rate (false hits / should-miss queries): {len(false_positives)}/{len(should_miss)} = {fp_rate:.1%}")
    if wrong_entry_hits:
        print(f"wrong-entry hits (worse than a plain miss): {len(wrong_entry_hits)}")
        for item in wrong_entry_hits:
            print(f"  {item['query']!r} matched {item['matched']!r} instead of the intended entry")
    if missed_paraphrases:
        print("\nmissed genuine paraphrases (recall gap, safe direction to be wrong):")
        for item in missed_paraphrases:
            sim = item["similarity"]
            sim_str = f"{sim:.3f}" if sim is not None else "n/a"
            print(f"  {item['query']!r} (similarity={sim_str})")
    if false_positives:
        print("\nfalse positives (unsafe direction -- would return a wrong answer):")
        for item in false_positives:
            print(f"  {item['query']!r} incorrectly matched {item['matched']!r} ({item['note']})")


if __name__ == "__main__":
    main()
