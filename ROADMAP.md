# Tokenetics Roadmap

Checklist of phases, steps and substeps — what was implemented, not the process of getting there. `[x]` done, `[ ]` to do.

---

## Phase 1 — Cross-cutting systems (complete)

**1.1 Cost logger** (`core/logger.py`) — `InMemoryCostLogger` for in-process use; `FileCostLogger` for persisting events to a JSONL log, one `run_id` per `Tokenetics()` instance.

**1.2 Fail-open framework** — a wrapping try/except around every stage call in `orchestrator.py`; a stage raising degrades to a no-op for that stage rather than crashing `prepare()`/`finalize()`. The measurement call that reads a stage's result (`count_tokens`) lives inside the same try/except as the stage's own execution, not after it, so an unmeasurable-but-valid result also fails open.

**1.3 Tokenizer** (`core/tokenizer.py`) — wraps Anthropic's `POST /v1/messages/count_tokens`. Forwards `tool_choice`, `thinking`, and `output_config` (all documented, count-affecting request parameters) in addition to the base message/tool fields, so a `structured_output`- or `thinking`-configured request is counted accurately.

---

## Phase 2 — Foundation stages (complete)

**2.1 Dedup** (`stages/dedup.py`) — SHA-256 fingerprint per message; an exact repeat is dropped, the first copy kept.

**2.2 Near-dup** (`stages/near_dup.py`) — MinHash/shingling estimate of Jaccard similarity between chunks; a stricter threshold for real conversation turns than for tool output; the most recent message is never touched; every merge is logged.

**2.3 Schema minification** (`stages/schema_minification.py`) — drops tool definitions clearly irrelevant to the latest message; anything ambiguous stays, conservative by default.

**2.4 Post-hoc trim** (`stages/post_hoc_trim.py`, `core/response_stage.py`) — strips known boilerplate sign-offs/pleasantries from a reply before it's stored, wired into `Tokenetics.finalize()`. Never trims a reply down to nothing — keeps the original text if trimming would leave it empty (an earlier version allowed this, which produced empty-content requests the real API rejects).

---

## Phase 3 — Task classifier (complete)

**3.1 Task type** (`stages/task_classifier.py`) — deterministic regex/keyword scan (no model call), 4 categories (`code`, `conversational`, `tool-heavy`, `extraction`) plus `None` below a confidence threshold, so downstream stages take their conservative branch instead of acting on a weak signal. 32-example hand-labeled validation set, gated at 85% accuracy. `task_type` is computed once and reused by stages 4, 5, 8, and 9 rather than re-derived.

**3.2 Bounded-shape signal** (`_is_bounded_shape`, same module) — a second, independent signal: deterministic, regex/word-count based (strips fenced code blocks first, then checks for open-ended markers like "design"/"architecture"/"walk me through", more than one real question, or prose over 40 words). True only for a short/self-contained request; ambiguous or ambiguous-adjacent input resolves to `False`. Added later (Phase 10) to drive the brevity/thinking-effort defaults below, not part of the original 4-category classification.

---

## Phase 4 — Classifier-dependent stages (complete)

**4.1 Tool-relevance filtering** — folded into `schema_minification` (2.3) rather than a separate stage.

**4.2 Structured output** (`stages/structured_output.py`) — forces `tool_choice` directly when the task is confidently data-shaped and a matching tool is registered, instead of hoping the model stays terse in free text.

**4.3 Brevity injector** (`stages/brevity_injector.py`) — MODERATE/AGGRESSIVE terse-instruction buckets, applied only where brevity is confidently safe for the request's task type. `aggressive_for_task_types` is a caller-supplied opt-in (empty/off by default).

**4.4 Adaptive budget** (`stages/adaptive_budget.py`) — a per-task-type `max_tokens` cap with a safety margin, widening automatically on real truncation. `config["enable_thinking_effort"]` tunes the model's adaptive-thinking effort level by task type (validated against real quality checks, not just a truncation-rate count, since an under-tuned reasoning setting fails silently).

**4.5 Supporting modules** — `core/model_compatibility.py` (per-model capability flags), `core/lexical.py` (`_words()` tokenizer helper, excludes single-character tokens so contraction remnants like `"'s"` don't produce false-negative zero-overlap results).

**4.6 Bounded-shape defaults** (added later, same stages) — validated across all 4 task-classifier categories (quality-check + real output-token A/B per category), then defaulted without requiring the opt-in:
  - `brevity_injector`: AGGRESSIVE fires automatically for `code`/`conversational`/`tool-heavy` when `bounded_shape=True`. `extraction` stays out of the default — its forced `tool_choice` reply is already compact, leaving the combined economics break-even to slightly negative — but remains available via `aggressive_for_task_types` opt-in, still gated on `bounded_shape`.
  - `adaptive_budget`: thinking effort downgrades one level (`high→medium`, `medium→low`, `low` is the floor) whenever `bounded_shape=True`, with no opt-in required. Unbounded or ambiguous-shape requests are unaffected — `enable_thinking_effort=True` is still required there. `thinking_effort_source` tags which path fired (`"bounded_shape_default"` vs `"opt_in"`).
  - Real measured economics behind the default: `code` 72.9–85.5% output-token savings (3 measurements), `conversational` 71.5–81.6% (2), `tool-heavy` 37.6–43.7% (2, thinner margin — its baseline is already terse), `extraction` break-even-to-negative (2). Thinking-effort downgrade: positive on all 4 types, 0 input-token cost on every measurement.

---

## Phase 5 — Context scheduler (complete)

**5.1 Turn scoring** (`stages/context_scheduler.py`) — a per-turn activity classifier; pinning rules protect an unresolved error or an explicit decision from being dropped.

**5.2 Selection** — a 0/1 knapsack DP picks the best-value combination of turns under a token budget; falls back to a faster greedy approximation past a latency threshold on very long conversations.

**5.3 Degraded mode** — falls back to "keep the last N" turns if the stage is disabled outright.

**5.4 Alternation fix** — `_fix_alternation` post-processes the selected/merged turns so role alternation (user/assistant/user/...) is never broken by a selection that keeps two same-role turns adjacent.

**5.5 Config surface** — `Tokenetics(stage_config=...)` parameter, the general mechanism later reused by `cache_reorder_guard`, `cache_breakpoint_optimizer`, and `delta_compression` to accept caller-supplied context.

---

## Phase 6 — Caching (complete)

**6.1 Pricing table** (`core/cache_pricing.py`) — versioned, dated pricing table (Sonnet 5: $2.00/$10.00 per MTok input/output; re-verified live 2026-09-06 as the permanent standard price, not an introductory rate).

**6.2 Cache reorder + safety guard** (`stages/cache_reorder_guard.py`, `core/errors.py`'s `CacheSafetyError`) — moves stable content (system prompt, tool definitions) before frequently-changing content, then hard-raises if anything before the previous cache point changed since the last call. The one stage in the whole pipeline that does not fail open — a silent cache-safety violation is worse than a crash.

**6.3 Cache breakpoint optimizer** (`stages/cache_breakpoint_optimizer.py`, `TokeneticsRequest.cache_breakpoint`) — picks the 5-minute or 1-hour tier from real repeat frequency and gap intervals between calls (using the *maximum* observed gap, not the average, for safety), or skips caching entirely if the traffic pattern doesn't justify the write. `core/cache_hash.py` for breakpoint identity.

**6.4 Benchmark** (`scripts/cache_benchmark.py`, synthetic traffic) — ~28-31% cheaper than never caching (estimated, not a real-API measurement).

---

## Phase 7 — Delta compression (complete)

**7.1 Diff mechanism** (`core/delta.py`) — `compute_delta`/`apply_delta`.

**7.2 Stage wiring** (`stages/delta_compression.py`) — sends only the diff when a payload changed by a meaningfully small amount, scoped to `tool_result` blocks keyed by `tool_use_id` (the join key agent-style tool responses naturally carry).

---

## Phase 8 — Tier 0 freeze (complete, 2026-08-02)

**8.1 Integration** — the deterministic core (Phases 1-7) assembled end to end, integration-tested in `tests/test_tier0_integration.py`.

**8.2 Signature freeze** — `inspect.signature`-based tests pin every public function/class signature so Tier 2 work (Phase 10) can build on a stable API.

---

## Phase 9 — Benchmark suite (complete, 2026-08-10)

**9.1 Benchmark core** (`core/benchmark.py`) — shared scoring/reporting machinery.

**9.2 Corpus command** (`scripts/benchmark_runner.py corpus`) — grown to 80 samples, 20 per category across the 4 task-classifier categories.

**9.3 Quality-check command** (`scripts/benchmark_runner.py quality-check`) — grown to 13 samples spanning all 4 categories, covering 5 distinct risk-bearing concerns (edge cases, ambiguous shapes, etc.), real spend gated behind a cost ceiling.

---

## Phase 10 — Tier 2 extras (complete)

Three independent opt-in extras, each with its own `pyproject.toml` extras group, fail-open, own benchmark.

**10a. TALE budget estimation** (`extras/tale.py`) — `estimate_token_budget()`, wired into `adaptive_budget` via `config["external_budget_estimate"]`; widens but never narrows the stage's own internal estimate.

**10b. LLMLingua-style compression** (`extras/compress.py`) — `Compressor`/`compress_text()`, word-granularity perplexity-based dropping via a local `distilgpt2` model. `RECOMMENDED_MAX_RATIO = 0.4` safety cap, set after a real quality benchmark found correctness holds 100% up to a 40% compression ratio but a real cliff starting at 0.45.

**10c. Semantic caching** (`extras/semantic_cache.py`) — `SemanticCache`, `sentence-transformers`/`all-MiniLM-L6-v2` embeddings + in-memory cosine similarity, 0.90 threshold chosen empirically, validated at scale: 0% false-positive rate, 64.3% recall.

---

## Phase 11 — Dashboard (complete)

**11.1 Data pipeline** — `core/logger.py`'s `FileCostLogger` (append-only JSONL), `dashboard/aggregate.py` (reads the log, computes per-stage and per-mechanism stats), `dashboard/server.py` + `scripts/dashboard.py` (local read-only HTTP server, `--log-file` flag).

**11.2 Savings overview** (top of page) — baseline-vs-optimized donuts for Request/Output/Thinking tokens (Chart.js via CDN), each shown only once real data exists for it, plus a combined 6-slice donut once ≥2 categories have data.

**11.3 Request tokens section** — hero + summary cards, min/median/max %-saved (never a single blended average), cache cards (lookups, hit rate, read/write tokens), a pipeline stage-breakdown table, a %-saved-by-stage pie chart. Stages with no expressible token-count effect (`task_classifier`, `cache_reorder_guard`, `structured_output`, `adaptive_budget`, `cache_breakpoint_optimizer`) get an inline explanatory note instead of a misleading bare `0.0%`.

**11.4 Output tokens section** — `GenerationUsageStats`, populated by real `usage.output_tokens` before/after measurements from `scripts/dev_demo.py --measure-usage`; min/median/max %-saved; its own donut; a caveat shown whenever either side of a measured pair was truncated (`stop_reason="max_tokens"`), so a negative number can't be misread as pure waste.

**11.5 Thinking tokens section** — "Thinking tokens consumed" (real, from `usage.output_tokens_details.thinking_tokens`, falling back to an estimate when no real samples exist yet); its own donut; min/median/max token-delta; "Thinking re-injection tokens saved" (a separate measurement — a 2-turn `display:"omitted"` vs `display:"summarized"` A/B via `--measure-thinking-reinjection`), kept as its own distinct, never-conflated cost dimension, with a noise-floor caveat below 10 samples.

**11.6 Other optimizations section** — TALE / compress / semantic-cache cards, grouped separately since none are cleanly request/output/thinking-specific.

**11.7 Visual pass** — hero cards with sign-aware color/badges, inline SVG logo mark, zebra-striped tables, section titles rewritten for a non-technical audience ("Advanced optimizations", "Pipeline stage breakdown", "Savings overview" — no internal tier/function names as page headings).

---

## Phase 12 — Distribution (complete)

**12.1 License** — MIT, committed.

**12.2 MCP server** (`src/tokenetics/mcp_server.py`) — a single `optimize_request` tool over stdio. Opt-in `TOKENETICS_MCP_LOG_FILE` env var for persistence (no logging by default — a stdio server shouldn't have surprising side effects). Opt-in `TOKENETICS_MCP_OPEN_DASHBOARD` (+ `TOKENETICS_MCP_DASHBOARD_PORT`) starts the dashboard server in a background thread and opens it in the default browser; failures are logged to stderr, never stdout, since stdout is the JSON-RPC stream the server itself depends on.

**12.3 Build verification** — `uv build` + a clean-venv install confirmed zero Tier-2-dependency leakage into the base install (`compress`/`semantic-cache`'s heavier dependencies stay opt-in at install time, not just at runtime).

---

## Real-world benchmark results

- **Single, isolated request** — -2.1% (roughly cost-neutral; caching and pruning need repeat calls or accumulated redundancy to pay off, which a lone message has neither of).
- **Real, repeated 8-turn conversation** (`scripts/benchmark_runner.py session`) — four real runs, all positive: +43.1%, +42.4%, +62.8%, +62.7% cheaper (avg ~53%), broken down as input tokens -67 to -70%, output tokens -62 to -74%, thinking tokens real and measured throughout. Re-confirmed after the Phase 4.6 brevity/thinking-effort defaults landed — the pre-default figure on the same benchmark was ~11.5%.

---

## How to test

```bash
uv sync
uv run pytest           # full suite, no API key needed
uv run ruff check .     # style
```

Real-API scripts (need `ANTHROPIC_API_KEY`, cost real money): `scripts/benchmark_runner.py corpus|quality-check|usage|session`, `scripts/dev_demo.py --measure-usage|--measure-thinking-reinjection`, `scripts/cache_benchmark.py`, `scripts/dashboard.py --log-file costs.jsonl`.
