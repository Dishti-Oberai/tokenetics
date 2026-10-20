# Tokenetics Roadmap

Checklist of phases, steps and substeps — what was implemented, not the process of getting there. `[x]` done, `[ ]` to do.

---

## Phase 1 — Cross-cutting systems (complete)

**1.1 Cost logger** (`core/logger.py`) — `InMemoryCostLogger`, later `FileCostLogger` for persistence to a JSONL log.

**1.2 Fail-open framework** — wrapped around every stage call in `orchestrator.py`; a stage failure degrades to a no-op for that stage rather than crashing `prepare()`/`finalize()`. The measurement call that reads a stage's result (`count_tokens`) lives inside the same try/except as the stage itself, not after it.

**1.3 Tokenizer** (`core/tokenizer.py`) — wraps Anthropic's `POST /v1/messages/count_tokens`. Forwards `tool_choice`, `thinking`, and `output_config` (all documented, count-affecting parameters) in addition to the base request fields.

---

## Phase 2 — Foundation stages (complete)

**2.1 Dedup** (`stages/dedup.py`) — SHA-256 fingerprint per message; an exact repeat is dropped, the first copy kept.

**2.2 Near-dup** (`stages/near_dup.py`) — MinHash/shingling estimate of Jaccard similarity between chunks; stricter threshold for real conversation turns than tool output; the most recent message is never touched; every merge is logged.

**2.3 Schema minification** (`stages/schema_minification.py`) — drops tool definitions clearly irrelevant to the latest message; anything ambiguous stays.

**2.4 Post-hoc trim** (`stages/post_hoc_trim.py`, `core/response_stage.py`) — strips known boilerplate sign-offs/pleasantries from a reply before it's stored, wired into `Tokenetics.finalize()`. Never trims a reply down to nothing — keeps the original text if trimming would leave it empty.

---

## Phase 3 — Task classifier (complete)

`stages/task_classifier.py` — deterministic regex/keyword scan (no model call), 4 categories (`code`, `conversational`, `tool-heavy`, `extraction`) plus `None` below a confidence threshold. 32-example hand-labeled validation set, gated at 85% accuracy. `task_type` is reused by stages 4, 5, 8, 9 rather than re-derived.

`request.meta.bounded_shape` (`_is_bounded_shape`) — a second, independent signal from the same stage: deterministic, regex/word-count based (strips fenced code first, then checks for open-ended markers, more than one real question, or prose over 40 words). True only for a short/self-contained request; ambiguous resolves to `False`.

---

## Phase 4 — Classifier-dependent stages (complete)

**4.1 Tool-relevance filtering** — folded into `schema_minification`.

**4.2 Structured output** (`stages/structured_output.py`) — forces `tool_choice` directly when the task is confidently data-shaped and a matching tool is registered.

**4.3 Brevity injector** (`stages/brevity_injector.py`) — MODERATE/AGGRESSIVE terse-instruction buckets, applied only where brevity is confidently safe. `aggressive_for_task_types` is a caller opt-in. Default (validated across all 4 task-classifier categories): AGGRESSIVE fires automatically for `code`/`conversational`/`tool-heavy` when `bounded_shape=True`; `extraction` stays out of the default (its forced `tool_choice` reply is already compact, leaving the economics break-even/slightly negative) but remains available via opt-in, still gated on `bounded_shape`.

**4.4 Adaptive budget** (`stages/adaptive_budget.py`) — per-task-type `max_tokens` cap with a safety margin, widening automatically on real truncation. Model's thinking-effort level: tuned via `config["enable_thinking_effort"]` by task type, and downgraded one level (`high→medium`, `medium→low`, `low` floor) whenever `bounded_shape=True` — defaulted on for that case with no opt-in required; unbounded/ambiguous requests still need the explicit opt-in. `thinking_effort_source` tags which path fired (`"bounded_shape_default"` vs `"opt_in"`).

**4.5 Supporting modules** — `core/model_compatibility.py`, `core/lexical.py` (`_words()` excludes single-character tokens, so contraction remnants like `"'s"` don't produce false-negative zero-overlap results).

---

## Phase 5 — Context scheduler (complete)

`stages/context_scheduler.py` — per-turn activity classifier; pinning rules protect an unresolved error or explicit decision; 0/1 knapsack DP picks the best-value combination of turns under a token budget; falls back to a faster greedy approximation past a latency threshold, or to "keep the last N" if disabled outright. `Tokenetics(stage_config=...)` parameter. `_fix_alternation` post-processes role alternation after selection/merging.

---

## Phase 6 — Caching (complete)

`core/cache_pricing.py` — versioned, dated pricing table (Sonnet 5: $2.00/$10.00 per MTok input/output, confirmed permanent as of 2026-09-06, not introductory). `core/errors.py`'s `CacheSafetyError`. `TokeneticsRequest.cache_breakpoint`.

**6.1 Cache reorder + safety guard** (`stages/cache_reorder_guard.py`) — moves stable content (system prompt, tool definitions) before frequently-changing content, then hard-raises if anything before the previous cache point changed since the last call. The one stage in the pipeline that does not fail open.

**6.2 Cache breakpoint optimizer** (`stages/cache_breakpoint_optimizer.py`) — picks the 5-minute or 1-hour tier from real repeat frequency and gap intervals (using the *maximum* observed gap, not the average, for safety), or skips caching if traffic doesn't justify the write. `core/cache_hash.py`.

**6.3 Benchmark** (`scripts/cache_benchmark.py`, synthetic traffic) — ~28-31% cheaper than never caching (estimated).

---

## Phase 7 — Delta compression (complete)

`core/delta.py`'s `compute_delta`/`apply_delta`, `stages/delta_compression.py` — sends only the diff when a payload changed by a small amount, scoped to `tool_result` blocks keyed by `tool_use_id`.

---

## Phase 8 — Tier 0 freeze (complete, 2026-08-02)

Deterministic core (phases 1-7) assembled, integration-tested (`tests/test_tier0_integration.py`), and signature-frozen (`inspect.signature`-based tests) — API-stable; nothing in phases 9-13 starts before this.

---

## Phase 9 — Benchmark suite (complete, 2026-08-10)

`core/benchmark.py`, `scripts/benchmark_runner.py` (`corpus`/`quality-check`/`usage`/`session` subcommands). Corpus: 80 samples, 20 per category across the 4 task-classifier categories. Quality-check set: grew to 13 samples spanning all 4 categories, covering 5 risk-bearing concerns. Real spend gated behind a cost ceiling throughout.

---

## Phase 10 — Tier 2 extras (complete)

Three independent opt-in extras, each with its own `pyproject.toml` extras group, fail-open, own benchmark.

**10a. TALE budget estimation** (`extras/tale.py`) — `estimate_token_budget()`, wired into `adaptive_budget` via `config["external_budget_estimate"]`; widens but never narrows the stage's own estimate.

**10b. LLMLingua-style compression** (`extras/compress.py`) — `Compressor`/`compress_text()`, word-granularity perplexity-based dropping via `distilgpt2`. `RECOMMENDED_MAX_RATIO = 0.4` safety cap (quality holds 100% up to a 40% compression ratio; a real correctness cliff starts at 0.45).

**10c. Semantic caching** (`extras/semantic_cache.py`) — `SemanticCache`, `sentence-transformers`/`all-MiniLM-L6-v2` + in-memory cosine similarity, 0.90 threshold. Measured at scale: 0% false-positive rate, 64.3% recall.

---

## Phase 11 — Dashboard (complete)

`core/logger.py`'s `FileCostLogger`, `dashboard/{aggregate,render,server}.py`, `scripts/dashboard.py`. Local, read-only web view over the cost logger's own log file.

- **Savings overview** (top of page): baseline-vs-optimized donuts for Request/Output/Thinking tokens, each shown only once real data exists, plus a combined 6-slice donut once ≥2 categories have data.
- **Request tokens section** — hero + summary cards, min/median/max %-saved, cache read/write cards (lookups, hit rate, read/write tokens — no donut), pipeline stage breakdown table, a %-saved-by-stage pie chart. Stages with no expressible token-count effect (`task_classifier`, `cache_reorder_guard`, `structured_output`, `adaptive_budget`, `cache_breakpoint_optimizer`) get an inline explanatory note instead of a misleading `0.0%`.
- **Output tokens section** — `GenerationUsageStats` (real, measured `usage.output_tokens` before/after via `scripts/dev_demo.py --measure-usage`), min/median/max %-saved, its own donut. Caveats shown when either side's completion was truncated (`stop_reason="max_tokens"`).
- **Thinking tokens section** — "Thinking tokens consumed" (real, from `usage.output_tokens_details.thinking_tokens`, falling back to an estimate when no real samples exist), its own donut, min/median/max token-delta, and "Thinking re-injection tokens saved" (a separate 2-turn `display:"omitted"` vs `"summarized"` A/B via `--measure-thinking-reinjection`) — kept as two distinct, never-conflated cost dimensions, with a noise-floor caveat below 10 samples on the re-injection card.
- **Other optimizations section** — TALE/compress/semantic-cache cards (none cleanly request/output/thinking-specific).
- Cosmetic pass: Chart.js (CDN) pie/bar/donut charts, hero cards with sign-aware color/badges, inline SVG logo, zebra-striped tables, section titles rewritten for a non-technical audience ("Advanced optimizations", "Pipeline stage breakdown", "Savings overview").

---

## Phase 12 — Distribution (complete)

MIT license. `src/tokenetics/mcp_server.py` — single `optimize_request` MCP tool over stdio; opt-in `TOKENETICS_MCP_LOG_FILE` persistence; opt-in `TOKENETICS_MCP_OPEN_DASHBOARD` (+ `TOKENETICS_MCP_DASHBOARD_PORT`) auto-opens the dashboard in a background thread, logging failures to stderr rather than touching the JSON-RPC stdout stream. `uv build` + clean-venv install verified: zero Tier-2-dependency leakage into the base install. Not yet on PyPI — publish is the one remaining step.

---

## Real-world benchmark results

- **Single, isolated request** — -2.1% (roughly cost-neutral; caching/pruning need repeat calls to pay off, which a lone message doesn't have).
- **Real, repeated 8-turn conversation** (`scripts/benchmark_runner.py session`) — four real runs, all positive: +43.1%, +42.4%, +62.8%, +62.7% cheaper (avg ~53%), broken down as input tokens -67 to -70%, output tokens -62 to -74%, thinking tokens real and measured throughout. Re-confirmed after the brevity/thinking-effort defaults landed (previous pre-default figure: ~11.5%).

---

## Out of scope (do not build, do not stub)

Anything not an opt-in Tier 2 extra behind its own extras group; any new default beyond the validated `bounded_shape`-gated ones above.

---

## How to test

```bash
uv sync
uv run pytest           # full suite, no API key needed
uv run ruff check .     # style
```

Real-API scripts (need `ANTHROPIC_API_KEY`, cost real money): `scripts/benchmark_runner.py corpus|quality-check|usage|session`, `scripts/dev_demo.py --measure-usage|--measure-thinking-reinjection`, `scripts/cache_benchmark.py`, `scripts/dashboard.py --log-file costs.jsonl`.
