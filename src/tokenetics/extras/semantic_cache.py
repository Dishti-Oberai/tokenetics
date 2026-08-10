"""Tier 2b: semantic response caching (opt-in, `tokenetics[semantic-cache]`).

Per the brief: workloads with repetitive queries (FAQ/support-style) can skip
generation entirely on a near-duplicate repeat, at the cost of the largest
correctness risk any stage in this project carries -- a bad match returns a
*wrong* answer to a *different* question, not just a suboptimal one. Every
design choice below follows from taking that risk seriously.

Deliberately outside Tier 0: this is the one genuinely *stateful* piece in
the whole project -- a request/response cache has to persist between calls
by definition, which is exactly why it's opt-in and lives here rather than
being a contradiction of the core's statelessness (CLAUDE.md: "if a feature
seems to need state, it belongs in a separate opt-in wrapper layered on top
of the stateless core -- not in Tier 0"). `SemanticCache` is a plain object
the caller constructs and holds themselves, same "caller owns persistence"
shape the brief uses everywhere else.

Similarity threshold (0.90, cosine, via `all-MiniLM-L6-v2`): empirically
measured 2026-08-10, not guessed -- same discipline `near_dup`'s 0.95
Jaccard bar got. Real same-intent paraphrase pairs ("What's the capital of
France?" / "What is the capital city of France?") scored 0.92-0.99; the
worst genuinely-different near-miss pair tested ("How do I reset my
password?" / "How do I reset my email password?" -- a different question
about a related topic) scored 0.87. 0.90 sits in that gap with margin on
both sides. This is a small, hand-picked sample, not a rigorous benchmark --
flagged in ROADMAP.md as the next thing to validate against a larger
held-out near-miss set before this threshold is called production-safe.

Dependency choice (resolved with the user 2026-08-10): `sentence-transformers`
for embeddings, plus a minimal in-memory cosine-similarity store -- not a
dedicated vector-DB library (chromadb/faiss). Smaller footprint, simpler
fail-open surface (one dependency to go missing, not two), matches the
brief's "lightweight local vector store" language.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

_log = logging.getLogger(__name__)

DEFAULT_SIMILARITY_THRESHOLD = 0.90
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


@dataclass
class CacheLookupResult:
    """What a `lookup()` call found, hit or miss -- match-quality (the real
    similarity score) is always populated, even on a miss, per the module
    docstring's false-positive-auditing requirement: a near-miss that just
    barely stayed below threshold is exactly the case worth being able to
    review later, not just the hits.
    """

    hit: bool
    response: Any | None
    similarity: float | None
    matched_query: str | None = None


@dataclass
class _CacheEntry:
    query: str
    embedding: Any
    response: Any
    stored_at: float


@dataclass
class SemanticCache:
    """Caller-constructed, caller-held semantic cache over (query, response)
    pairs. Not thread-safe, not persisted to disk -- an in-process cache for
    a single caller's own use, matching Tier 0's "no hidden persistence"
    spirit even though this object itself is the one place that's
    deliberately stateful.

    Fail-open (per CLAUDE.md's "hard error" flavor): if the embedding model
    can't load, or embedding/similarity computation raises for any reason,
    `lookup()` returns a clean miss (never raises) so the caller always
    falls through to a normal API call. `store()` fails open the same way
    (a failed store is a missed future cache hit, not a corrupted one).
    """

    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    ttl_seconds: float | None = None
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL
    _entries: list[_CacheEntry] = field(default_factory=list)
    _model: Any = field(default=None, repr=False)

    def _model_instance(self) -> Any | None:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.embedding_model_name)
        except Exception:
            _log.error("semantic_cache: failed to load embedding model %r", self.embedding_model_name, exc_info=True)
            return None
        return self._model

    def _embed(self, text: str) -> Any | None:
        model = self._model_instance()
        if model is None:
            return None
        try:
            return model.encode([text])[0]
        except Exception:
            _log.error("semantic_cache: embedding call failed", exc_info=True)
            return None

    def _expire(self) -> None:
        if self.ttl_seconds is None:
            return
        cutoff = time.time() - self.ttl_seconds
        self._entries = [e for e in self._entries if e.stored_at >= cutoff]

    def lookup(self, query: str) -> CacheLookupResult:
        """Fail-open: any embedding/model failure returns a clean miss, the
        same as an empty cache -- never raises, never blocks the caller's
        real API call.
        """
        self._expire()
        if not self._entries:
            return CacheLookupResult(hit=False, response=None, similarity=None)

        embedding = self._embed(query)
        if embedding is None:
            return CacheLookupResult(hit=False, response=None, similarity=None)

        try:
            best_entry, best_similarity = self._best_match(embedding)
        except Exception:
            _log.error("semantic_cache: similarity computation failed", exc_info=True)
            return CacheLookupResult(hit=False, response=None, similarity=None)

        hit = best_similarity >= self.similarity_threshold
        _log.info(
            "semantic_cache lookup: hit=%s similarity=%.4f threshold=%.2f query=%r matched=%r",
            hit, best_similarity, self.similarity_threshold, query, best_entry.query,
        )
        return CacheLookupResult(
            hit=hit,
            response=best_entry.response if hit else None,
            similarity=best_similarity,
            matched_query=best_entry.query if hit else None,
        )

    def _best_match(self, embedding: Any) -> tuple[_CacheEntry, float]:
        best_entry = self._entries[0]
        best_similarity = _cosine_similarity(embedding, best_entry.embedding)
        for entry in self._entries[1:]:
            similarity = _cosine_similarity(embedding, entry.embedding)
            if similarity > best_similarity:
                best_entry, best_similarity = entry, similarity
        return best_entry, best_similarity

    def store(self, query: str, response: Any) -> None:
        """Fail-open: if embedding fails, the entry is silently not stored
        (a missed future cache hit, not a corrupted cache) rather than
        raising and interrupting the caller's real request flow.
        """
        embedding = self._embed(query)
        if embedding is None:
            return
        self._entries.append(
            _CacheEntry(query=query, embedding=embedding, response=response, stored_at=time.time())
        )

    def invalidate(self, query: str) -> int:
        """Removes all entries whose stored query exactly matches `query`.
        Returns the number of entries removed."""
        before = len(self._entries)
        self._entries = [e for e in self._entries if e.query != query]
        return before - len(self._entries)

    def clear(self) -> None:
        self._entries = []

    def __len__(self) -> int:
        self._expire()
        return len(self._entries)


def _cosine_similarity(a: Any, b: Any) -> float:
    import numpy as np

    a_arr, b_arr = np.asarray(a), np.asarray(b)
    denom = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    if denom == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / denom)
