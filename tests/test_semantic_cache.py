"""Tier 2b tests. Only meaningful under `tokenetics[semantic-cache]`
installed -- skips cleanly (not a failure) if `sentence_transformers` isn't
available, per CLAUDE.md's "tested only under their own extras install"
rule for Tier 2 extras.
"""

from __future__ import annotations

import pytest

pytest.importorskip("sentence_transformers")

from tokenetics.extras.semantic_cache import SemanticCache  # noqa: E402


def test_miss_on_an_empty_cache():
    cache = SemanticCache()
    result = cache.lookup("What is the capital of France?")
    assert result.hit is False
    assert result.response is None
    assert result.similarity is None


def test_stores_and_hits_on_a_genuine_paraphrase():
    cache = SemanticCache()
    cache.store("What is the capital of France?", "Paris.")
    result = cache.lookup("What's the capital city of France?")
    assert result.hit is True
    assert result.response == "Paris."
    assert result.similarity is not None
    assert result.similarity >= cache.similarity_threshold


def test_misses_on_a_genuinely_different_related_question():
    # The exact near-miss family the threshold was empirically tuned
    # against (see semantic_cache.py's module docstring): related topic,
    # different actual question -- must NOT hit, or this is the false-
    # positive failure mode the whole module exists to avoid.
    cache = SemanticCache()
    cache.store("How do I reset my password?", "Go to Settings > Security > Reset Password.")
    result = cache.lookup("How do I delete my account?")
    assert result.hit is False
    # Match-quality is still reported on a miss, for false-positive auditing.
    assert result.similarity is not None


def test_misses_on_a_clearly_unrelated_query():
    cache = SemanticCache()
    cache.store("What is the capital of France?", "Paris.")
    result = cache.lookup("Recommend me a good pizza recipe.")
    assert result.hit is False


def test_returns_the_best_match_among_multiple_entries():
    cache = SemanticCache()
    cache.store("What is the capital of France?", "Paris.")
    cache.store("What is the capital of Germany?", "Berlin.")
    result = cache.lookup("What's the capital city of France?")
    assert result.hit is True
    assert result.response == "Paris."
    assert result.matched_query == "What is the capital of France?"


def test_invalidate_removes_matching_entries_and_reports_count():
    cache = SemanticCache()
    cache.store("What is the capital of France?", "Paris.")
    removed = cache.invalidate("What is the capital of France?")
    assert removed == 1
    assert len(cache) == 0
    result = cache.lookup("What's the capital city of France?")
    assert result.hit is False


def test_clear_empties_the_cache():
    cache = SemanticCache()
    cache.store("a", "1")
    cache.store("b", "2")
    cache.clear()
    assert len(cache) == 0


def test_ttl_expires_old_entries():
    cache = SemanticCache(ttl_seconds=0.05)
    cache.store("What is the capital of France?", "Paris.")
    assert len(cache) == 1
    import time

    time.sleep(0.1)
    result = cache.lookup("What's the capital city of France?")
    assert result.hit is False
    assert len(cache) == 0


def test_fails_open_on_embedding_model_load_failure():
    cache = SemanticCache(embedding_model_name="this-model-definitely-does-not-exist-12345")
    cache.store("hi", "there")  # must not raise
    result = cache.lookup("hi")  # must not raise
    assert result.hit is False


def test_len_reflects_stored_entry_count():
    cache = SemanticCache()
    assert len(cache) == 0
    cache.store("a", "1")
    assert len(cache) == 1


def test_the_real_near_miss_benchmark_set_has_valid_referential_integrity():
    # Data-integrity check for benchmarks/semantic_cache/near_miss_set.json
    # (used by scripts/semantic_cache_benchmark.py, Phase 10c's at-scale
    # false-positive/recall validation) -- catches a malformed dataset (a
    # typo'd expected_hit_id, a missing field) at test time rather than
    # mid-benchmark-run.
    import json
    from pathlib import Path

    path = (
        Path(__file__).resolve().parent.parent
        / "benchmarks"
        / "semantic_cache"
        / "near_miss_set.json"
    )
    dataset = json.loads(path.read_text())

    stored_ids = {e["id"] for e in dataset["stored_entries"]}
    assert len(stored_ids) == len(dataset["stored_entries"]), "duplicate stored entry ids"
    assert len(stored_ids) >= 10

    should_hit = [q for q in dataset["test_queries"] if q["expected_hit_id"] is not None]
    should_miss = [q for q in dataset["test_queries"] if q["expected_hit_id"] is None]
    assert len(should_hit) >= 10
    assert len(should_miss) >= 10
    for item in should_hit:
        assert item["expected_hit_id"] in stored_ids, (
            f"test query {item['query']!r} references unknown stored entry "
            f"{item['expected_hit_id']!r}"
        )
