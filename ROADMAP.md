# Roadmap

This is the dated, actionable version of the build order in [CLAUDE.md](CLAUDE.md). CLAUDE.md is the source of truth for *how* each phase must be built (fail-open discipline, conservative defaults, testing requirements); [tokenetics-project-brief.md](tokenetics-project-brief.md) is the source of truth for *what* gets built and why. This file is only about sequencing, effort, and dates.

## Immediate next steps (this week)

Phase 0 is partially done already — `pyproject.toml`, pytest, and `src/tokenetics/__init__.py` (with a version string and a passing import test) exist. What's left to finish it:

- [x] Set up CI (run `ruff check`, `mypy`, and `pytest` on every push/PR) — `.github/workflows/ci.yml`
- [x] Define the internal request representation (the type every stage reads/writes — not raw Anthropic API JSON) — `src/tokenetics/core/request.py`
- [x] Define the plugin contract as an ABC/protocol: `plugin(request, config, logger) -> request` — `src/tokenetics/core/plugin.py` (`Stage` ABC), `src/tokenetics/core/logger.py` (`CostLogger` interface stub)
- [x] Build the orchestrator skeleton (runs plugins in the fixed pipeline order, nothing real inside yet) — `src/tokenetics/orchestrator.py` (`Tokenetics`)
- [x] Write a no-op plugin round-trip test (request goes in, comes out unchanged, through the full orchestrator) — `tests/test_orchestrator.py`, `tests/test_request.py`
- [x] Add a runnable dev demo script (e.g. `scripts/dev_demo.py`, outside pytest/CI) that hits the real Anthropic API through the current pipeline and prints token counts before/after, which stages fired, and the reply text — per [CLAUDE.md](CLAUDE.md)'s "Incremental runnability" convention. Keep it wired up as each new stage lands in later phases, not just Phase 0. — `scripts/dev_demo.py`

**Phase 0 complete.** Next up: Phase 1 (cost logger, fail-open framework, tokenizer, deliberately-broken-plugin test).

## Phase 1 (cost logger, fail-open framework, tokenizer)

- [x] Shared tokenizer utility, wrapping the real `count_tokens` API (never a local approximation) — `src/tokenetics/core/tokenizer.py`
- [x] Real cost logger (`InMemoryCostLogger`), recording tokens before/after, timing, on/off, and `measured`/`estimated` per stage — `src/tokenetics/core/logger.py`
- [x] Fail-open framework: every stage call wrapped so a raised exception is caught, the stage skipped, the request passed through unmodified, logged at error level — `src/tokenetics/orchestrator.py`
- [x] The deliberately-broken-plugin test — `tests/test_fail_open.py`
- [x] `scripts/dev_demo.py` updated to print real token counts and the per-stage cost-logger summary

**Phase 1 complete.** Next up: Phase 2 (dedup, near-dup, schema minification, post-hoc trim — the first real pipeline stages).

## Phase 2 (dedup, near-dup, schema minification, post-hoc trim)

- [x] `DedupStage` — exact-match, hash-based, keeps first occurrence — `src/tokenetics/stages/dedup.py`
- [x] `NearDupStage` — hand-rolled MinHash/shingling (no new dependency), dual thresholds, most-recent-turn exemption falls out of the algorithm rather than a special case — `src/tokenetics/stages/near_dup.py`
- [x] `SchemaMinificationStage` — structural-only (inert JSON-Schema keys + whitespace collapse); tool-*relevance* filtering deferred to Phase 4 since it needs the classifier — `src/tokenetics/stages/schema_minification.py`
- [x] `ResponseStage` — a parallel interface for response-side stages, since post-hoc trim operates on reply text, not a `TokeneticsRequest` — `src/tokenetics/core/response_stage.py`
- [x] `PostHocTrimStage` — end-anchored boilerplate stripping, applied iteratively for stacked sign-offs — `src/tokenetics/stages/post_hoc_trim.py`
- [x] `Tokenetics.finalize(response)` — new public method mirroring `prepare()`'s fail-open + cost-logger pattern, on response text
- [x] `Tokenetics()`'s default pipeline now runs the real stages automatically, in fixed order — `dedup → near_dup → schema_minification` — with fresh instances per call (not shared singletons, to avoid `--disable`-style mutation leaking across instances)
- [x] `scripts/dev_demo.py` updated with a sample conversation containing a real duplicate, plus a `finalize()` call demonstrating post-hoc trim

**Phase 2 complete.** Next up: Phase 3 (shared task classifier + a 30-50 example hand-labeled validation set — clears the accuracy gate before Phase 4 consumes it).

## Phase 3 (shared task classifier + hand-labeled validation set)

- [x] `TaskClassifierStage` — deterministic regex/keyword scoring over the latest user message + a structural scan for `tool_use`/`tool_result` content; emits `(task_type, confidence)` onto `request.meta` for stages 4 and 9 to read later — `src/tokenetics/stages/task_classifier.py`
- [x] Categories: `code`, `conversational`, `extraction`, `tool-heavy`, plus an explicit `None` ("unclassified") outcome below the confidence threshold, so downstream stages get an honest "don't know" instead of a guessed label
- [x] Slotted into the fixed pipeline order between `near_dup` and `schema_minification` — `src/tokenetics/orchestrator.py`
- [x] 30-50 example hand-labeled validation set (32 examples, covering all 4 categories plus 2 unclassified edge cases) — `tests/fixtures/classifier_validation_set.json`
- [x] Accuracy gate test: **85%** threshold (confirmed with the user) — `tests/test_task_classifier_accuracy.py`
- [x] Per-category unit tests — `tests/test_task_classifier.py`
- [x] `scripts/dev_demo.py` updated (classifier now fires by default; its `task_type`/`confidence` show up in the existing per-stage log via `self.note()`)

**Phase 3 complete.** Next up: Phase 4 (classifier-dependent stages: tool-relevance filtering, structured-output enforcement, brevity injector, adaptive generation budgets).

## Phase 4 (classifier-dependent stages)

- [x] Tool-relevance filtering added to `SchemaMinificationStage` (same pipeline slot as minification, not a new stage): drops only tools with zero lexical-word-overlap against the latest user message; keeps all tools untouched if there's no user text to compare against — `src/tokenetics/stages/schema_minification.py`
- [x] `StructuredOutputStage` (9a) — forces `tool_choice` onto an already-registered tool when the classifier flags `extraction` and a tool's name/description clears a lexical-match bar; never invents a schema, respects a caller-set `tool_choice`. Schema-source approach confirmed with the user (2026-07-26): reuse an existing matching tool, not a generic JSON constraint or a logged-only no-op — `src/tokenetics/stages/structured_output.py`
- [x] `BrevityInjectorStage` (9b) — `extraction` → MODERATE instruction appended to the system prompt; everything else → OFF. AGGRESSIVE is defined but deliberately not auto-selected yet — the classifier's 4 categories don't reliably signal "verifiable short answer, minimal reasoning" — `src/tokenetics/stages/brevity_injector.py`
- [x] `AdaptiveBudgetStage` (9c) — task-type-derived `max_tokens` (widen-only, never narrows below the caller's value) and task-type-derived thinking effort via `thinking: {type: "adaptive"}` + `output_config.effort`, gated by a new model-compatibility table; skips entirely (info-level log) on unsupported models or when the caller already set their own `thinking` config — `src/tokenetics/stages/adaptive_budget.py`
- [x] Resolved the max_tokens/stateless-core conflict flagged during Phase 4 planning: the truncation-rate stat that widens the margin further is caller-supplied per call (`config["truncation_stats"]`), not stage-tracked — see the project brief's Amendments log (2026-07-26)
- [x] New versioned model-compatibility table (`LAST_VERIFIED = "2026-07-26"`), unknown/unconfirmed models conservatively resolve to unsupported rather than guessed — `src/tokenetics/core/model_compatibility.py`
- [x] Shared `core/lexical.py` helper (`overlap_score`, `latest_user_text`, stopword-filtered) — used by both tool-relevance filtering and structured-output's tool matching; `task_classifier.py` refactored to use the same `latest_user_text` instead of its own copy
- [x] Slotted into the fixed pipeline order after `schema_minification`: `structured_output → brevity_injector → adaptive_budget` — `src/tokenetics/orchestrator.py`
- [x] Per-stage unit tests — `tests/test_structured_output.py`, `tests/test_brevity_injector.py`, `tests/test_adaptive_budget.py`, `tests/test_lexical.py`, `tests/test_model_compatibility.py`, plus new relevance-filtering tests in `tests/test_schema_minification.py`
- [x] `scripts/dev_demo.py` updated: the Phase-2-era blanket `thinking: {type: "disabled"}` default removed now that `adaptive_budget` owns that decision for classified requests on supported models; documented which stages the default sample does/doesn't exercise

**Caveat, flagged during Phase 4 planning and still true**: CLAUDE.md's testing bar for stage 9c calls for a real-response `max_tokens` stress test and a thinking-effort validation against the benchmark suite's held-out quality-check set — neither exists yet, since both need Phase 9's corpus. Phase 4's tests are structural (correct routing per task type, conservative fallback, fail-open on unsupported models), not the full quality/truncation-rate study. That's the acknowledged forward dependency, not a skipped requirement — the real validation is still owed at Phase 9.

**Phase 4 complete.** Next up: Phase 5 (context scheduler: per-turn classifier, pinning, DP knapsack, greedy fallback, degraded mode, latency benchmark).

## Phase 5 (context scheduler / DP knapsack)

- [x] Per-turn activity classifier (`code`/`error`/`decision`/`tool_result`/`small_talk`, priority-ordered, `small_talk` the catch-all) — separate from stage 3's request-level classifier — `src/tokenetics/stages/context_scheduler.py`
- [x] Retention-value scoring: type weight + a bounded recency bonus; heuristic defaults, not yet benchmarked (Phase 9's job)
- [x] Pinning: `decision` turns always pinned; `error` turns pinned unless a **later turn from the user** (not the assistant, not just "conversation moved on") contains an explicit positive resolution signal — resolved with the user 2026-07-26, see the project brief's Amendments log
- [x] Token budget: caller-supplied via `config["token_budget"]` when given; otherwise falls back to a context-window-derived default scaled by `task_type`, logged `estimated` — `src/tokenetics/core/context_window.py`
- [x] 0/1 knapsack via DP (budget discretized into 100-token buckets), verified against brute-force on small synthetic cases
- [x] Greedy value-per-token fallback above a threshold empirically justified with real timing data (documented in-code: DP stays under ~15ms through ~n=500/budget=500-bucket histories, crosses into hundreds of ms well beyond that) — informal justification per Phase 5's build-order requirement; a rigorous latency benchmark is still Phase 9 scope
- [x] Real degraded-mode fallback (last-N-turns truncation, N=20) — the first stage whose `degraded_fallback()` does something other than pass through unchanged
- [x] Closed a real gap found while wiring this up: `Tokenetics.prepare()` previously always passed an empty `{}` config to every stage, which meant `config["token_budget"]` (and Phase 4's `adaptive_budget` `truncation_stats`) was unreachable through the real public API, only testable by calling a stage's `run()` directly. Added `Tokenetics(..., stage_config={...})`, keyed by stage name — `src/tokenetics/orchestrator.py`
- [x] Repositioned in the fixed pipeline order: `... → schema_minification → context_scheduler → structured_output → ...` (stages 6-8 still don't exist, so the stage-9 trio still immediately follows)
- [x] Per-turn token costs for the knapsack's internal sizing use a fast local ~4-chars/token estimate, not the real tokenizer — stages have no client/network access (Tier 0 is deterministic). The orchestrator's own before/after counts remain the authoritative `measured` numbers for anything logged
- [x] Tests: turn classification, pinning/resolution (including that assistant-claimed resolution doesn't count), DP-vs-brute-force, greedy-fallback trigger, degraded mode (including through the orchestrator with the stage disabled), budget-source (caller-supplied vs. estimated fallback), `stage_config` threading — `tests/test_context_scheduler.py`, `tests/test_context_window.py`, additions to `tests/test_orchestrator.py`
- [x] `scripts/dev_demo.py` `long_history` scenario added

**Role-alternation limitation, flagged during implementation and resolved same-day (2026-07-26)**: unlike `near_dup`'s pair-aware dropping, `context_scheduler` drops individual turns independently based on score, so nothing inherently guaranteed the surviving subset still alternated user/assistant roles correctly for the real API. Fixed with a `_fix_alternation` post-processing pass, applied as the last step of both `run()` and `degraded_fallback()`: adjacent same-role survivors are merged into one message (nothing lost, just combined — logged distinctly from an actual drop), and a lone leading non-`user` survivor is dropped (the API requires the first message to be `user` — the one case where content is genuinely lost, noted as `alternation_fix_dropped_lead_turn`). If the fix would empty the result entirely, the stage fails open and returns the request unmodified. Caught a related logging bug while verifying this against `long_history`: a lossless merge was initially (incorrectly) logged as a "dropped lead turn" — fixed by having `_fix_alternation` report the two cases distinctly rather than inferring from a count difference.

**Phase 5 complete (structural).** Next up: Phase 6 (caching: safety guard → reorder → pricing config table → breakpoint optimizer + synthetic-traffic benchmark).

## Phase 6 (caching: safety guard → reorder → pricing table → breakpoint optimizer)

- [x] Cache pricing sourced live (not guessed) from Anthropic's prompt-caching docs on 2026-07-27: 5m write = 1.25x, 1h write = 2x, read = 0.1x base input price, universal multipliers stacked on each model's own base price — `src/tokenetics/core/cache_pricing.py`
- [x] `CacheSafetyError` — dedicated exception type (not a bare `Exception`), specifically catchable, with a message identifying exactly what changed — `src/tokenetics/core/errors.py`
- [x] Extended `TokeneticsRequest` with `cache_breakpoint: CacheBreakpoint | None` (anchor: `"system"`/`"tools"`, ttl: `"5m"`/`"1h"`); `from_api_kwargs`/`to_api_kwargs` round-trip real `cache_control` block placement, including parsing an already-cached `previous_request` back into a comparable internal shape — `src/tokenetics/core/request.py`
- [x] `CacheReorderGuardStage` (stage 7) — the guard diffs the current request's system+tools against a caller-supplied `config["previous_request"]` (raw API kwargs) and hard-raises `CacheSafetyError` if either changed; the reorder half is a normalize-and-verify no-op given today's already-stable-first request shape (per the 2026-07-27 scoping decision) — `src/tokenetics/stages/cache_reorder_guard.py`
- [x] **Special-cased `CacheSafetyError` in the orchestrator's fail-open framework** — a necessary fix caught while wiring this up: the generic `except Exception:` catch-all would otherwise have silently swallowed the guard's hard-raise exactly like every other stage's failures, defeating the entire point of having one. Added a regression test asserting it actually propagates out of `Tokenetics.prepare()` — `src/tokenetics/orchestrator.py`
- [x] `CacheBreakpointOptimizerStage` (stage 8) — anchors to `tools` if present else `system` (prefix-based caching means one marker at the end of the last stable block caches everything before it too); needs ≥2 matching `config["cache_usage_history"]` entries (`{"timestamp", "content_hash"}`, hashed via `core/cache_hash.py`) to estimate a repeat interval, conservative no-breakpoint otherwise — `src/tokenetics/stages/cache_breakpoint_optimizer.py`
- [x] **Caught and fixed a real modeling bug** while writing this stage's tests: the first version compared each TTL tier's *total* expected benefit independently over its own window, which meant the 1-hour tier's larger window always won mathematically (more window = more assumed reads), so the 5-minute tier could never actually be selected even at very short repeat gaps. Fixed by recognizing that a cache read refreshes its own TTL — once a tier's window covers the repeat gap, the cache stays continuously warm off one amortized write, so the cheapest viable tier should win. Redesigned to a simple "cheapest tier that stays warm" rule; the pricing multipliers are now used for an informational net-benefit estimate logged alongside the decision, not for the tier choice itself.
- [x] **Caught and fixed a second real gap, post-Phase-6, via `--measure-usage` against the real API (2026-08-02)**: `cache_hit`'s real `usage.cache_creation_input_tokens`/`cache_read_input_tokens` both came back 0 despite the stage logging `breakpoint_placed=True` — first traced to a per-model minimum cacheable prefix length (1024 tokens for `claude-sonnet-5`) that the real API enforces silently, with no error, just an ordinary uncached request. Added `MIN_CACHEABLE_TOKENS` (sourced live, same doc, re-verified 2026-08-02) to `core/cache_pricing.py`, and the stage declined to place a breakpoint below that minimum (`reason="prefix_below_minimum_cacheable_size"`). `cache_hit`'s demo system prompt and `cache_benchmark.py`'s three traffic profiles were resized above the threshold.
- [x] **Caught and fixed a third, more fundamental gap in the same investigation**: even after the resize, the numbers still came back 0/0. Root cause was the anchor-selection logic itself: Anthropic's cache hierarchy is `tools -> system -> messages`, and a breakpoint caches everything up to AND INCLUDING the block it's on, in that order — so a marker on the last `tool` caches **only the tools array**, never the `system` prompt after it. The stage had been anchoring to `"tools"` whenever tools were present, on the mistaken assumption (confirmed live against the docs, 2026-08-02) that this also covered `system` — it was confusing the JSON *key* order `to_api_kwargs` happens to emit (system, tools, messages) with Anthropic's actual cache-*hierarchy* order (tools, system, messages), which is the reverse for system/tools. For `cache_hit`, this meant only the ~30-token tools array was ever being offered to the cache, far below the minimum, regardless of how large the system prompt was. Fixed by anchoring to `"system"` whenever one exists (covers tools+system together) and only falling back to `"tools"` when there's no system to fold in — `src/tokenetics/stages/cache_breakpoint_optimizer.py`. `cache_reorder_guard.py`'s docstring (the likely source of the original confusion) was also corrected. **Confirmed end-to-end against the real API same day**: `--all-scenarios --measure-usage` showed `cache_hit`'s pipeline-enabled call writing a real `cache_creation_input_tokens=1724`, then an immediate repeat call reading those same 1,724 tokens back via `cache_read_input_tokens` — the full write-then-read cycle, measured, not just unit-tested.
- [x] Synthetic-traffic benchmark, per this phase's explicit build-order requirement (distinct from Phase 9's real-corpus benchmark suite) — `scripts/cache_benchmark.py`, no network calls, deterministic. Headline numbers at 1000/5000 simulated calls across three traffic profiles (hot/warm/cold, each sized above the real per-model minimum cacheable prefix — see the `cache_breakpoint_optimizer` minimum-size fix below): **~28-31% cheaper than never caching, ~10-11% cheaper than a naive always-5m-breakpoint habit**, ~80% cache hit rate
- [x] Repositioned in the fixed pipeline order: `... → context_scheduler → cache_reorder_guard → cache_breakpoint_optimizer → structured_output → ...`
- [x] Tests, including the standalone hard-raise test required before anything else in this phase per CLAUDE.md — `tests/test_cache_reorder_guard.py`, `tests/test_cache_breakpoint_optimizer.py`, `tests/test_cache_pricing.py`, `tests/test_cache_hash.py`, plus cache-control round-trip tests added to `tests/test_request.py`
- [x] `scripts/dev_demo.py`: `cache_hit` scenario (breakpoint optimizer placing a 5m breakpoint) and `cache_guard_violation` scenario (demonstrates the hard-raise, caught and printed rather than crashing `--all-scenarios`)

**Known stub-only artifact, not a real bug**: `dev_demo.py`'s offline (`StubAnthropicClient`) token counts show `cache_breakpoint_optimizer` "costing" tokens when it places a breakpoint — this is the stub's `len(str(system))` approximation reacting to `system` becoming a block-list once `cache_control` is attached, not a real cost. The real Anthropic API only counts actual text content, not JSON/cache_control structure; against the real API this stage should show ~0 request-token change.

**Phase 6 complete.** Next up: Phase 7 (structured delta compression + `apply_delta()` + round-trip tests).

## Phase 7 (structured delta compression + `apply_delta()`)

- [x] **Join-key design (resolved 2026-08-02)**: scoped to `tool_result` content blocks only, keyed by `tool_use_id` — `config["previous_payloads"]: dict[str, str]` maps a tool_result's `tool_use_id` (unique within the current request, already known to the caller when constructing that reply) to whatever they consider its previous version. General prose-text repeats stay near_dup's job. See the brief's stage 6 amendment for the full resolution.
- [x] `src/tokenetics/core/delta.py` — `compute_delta`/`apply_delta`, stdlib-only (`difflib` + `json`, no new core dependency). Two content shapes, auto-detected: JSON content (both previous/new parse as an object/array) gets a minimal hand-rolled structural patch (add/remove/replace by escaped JSON-pointer-style path, RFC 6902-flavored subset, not the full spec); everything else gets a line-level patch via `difflib.SequenceMatcher` opcodes, applied by directly replaying those opcodes against `previous` rather than re-parsing unified-diff hunk text (avoids real correctness risk for no benefit, since nothing outside the module ever reads a human-authored diff). `DELTA_SIZE_RATIO_THRESHOLD = 0.6` — a plain documented constant (not pricing/compat data, so no versioned table, same precedent as near_dup's similarity thresholds).
- [x] **"Exact" round-trip means two different things by design**: line-patch reconstruction is byte-exact; JSON-patch reconstruction is semantically exact (`json.loads(apply_delta(...)) == json.loads(new)`, re-serialized via `json.dumps(..., sort_keys=True)`) — tool-result payloads are data, not formatted prose, so byte-exact JSON formatting was never a real invariant worth preserving. Documented explicitly in the module docstring so it reads as a decision, not an oversight.
- [x] `DeltaCompressionStage` (stage 6) — `src/tokenetics/stages/delta_compression.py`. Only `tool_result` blocks whose `content` is a plain string are eligible (list-of-blocks content, e.g. an image result, is left untouched — conservative by default, not guessed at); falls back to full content when the encoded delta isn't meaningfully smaller than the threshold.
- [x] Repositioned in the fixed pipeline order: `... → context_scheduler → delta_compression → cache_reorder_guard → ...`
- [x] Tests: `tests/test_delta.py` (round-trip correctness for both shapes, no-change case, totally-different-content-returns-None, path-escaping, JSON top-level-type-change fallback, marker-rejection) and `tests/test_delta_compression.py` (stage-level matching/skipping behavior, multi-block requests, plus the deliberately-broken-plugin fail-open regression test required before any stage is considered done — a malformed content block crashes the stage internally, and the orchestrator must catch it and pass the request through unmodified). Pipeline-order assertion updated in `tests/test_orchestrator.py`.
- [x] `scripts/dev_demo.py`: `delta_hit` scenario — a tool_result whose JSON content differs from a caller-supplied previous version by one field out of many; offline-verified 1035 → 113 chars (single-op JSON patch), confirmed it survives the full pipeline unmodified through every later stage.
- [x] **Deliberately deferred, not resolved here**: whether Claude actually interprets a delta-encoded tool_result correctly (vs. just seeing the full payload) is a real model-quality question, not a mechanical one — belongs in Phase 9's held-out quality-check set, same treatment as brevity/thinking-effort tuning, not assumed to just work. Flagged in both the brief and the stage's own docstring.

**Phase 7 complete.** Next up: Phase 8 (assemble Tier 0 in fixed order, end-to-end integration test, freeze the Tier 0 API).

## Assumptions

- **Original pace assumption (2026-07-19, superseded below)**: solo, ~4 hours/day, ~5–6 days/week (~20–24 hrs/week) — a human-solo-dev estimate. This turned out to badly understate actual velocity once implementation started (see "Re-forecast" below) and is kept here only for context on how the original per-phase effort weights were derived.
- **Confidence**: estimates for phases 0–8 (Tier 0) are moderately confident. Phases 9–13 (benchmark suite, Tier 2 extras, distribution, dashboard, docs) are rougher and should be re-forecast again once Tier 0 actually ships — actual velocity through 0–8 is better evidence than either the original guess or this first re-forecast.
- **Scope**: full v1 as specified in the brief — nothing cut. This was a deliberate choice: the alternative (a minimal Tier 0 slice, or thin-rigor stages) was offered and rejected in favor of keeping the brief's rigor requirements (measured-not-estimated benchmarks, hand-labeled classifier validation set, fail-open test coverage per stage, quality-degradation benchmarks for lossy compression) intact.

### Re-forecast (2026-07-26, based on actual Phase 0–3 velocity)

Phases 0–3 — originally estimated at ~4.5 weeks (31.5 days) combined — were actually completed in **7 days** (2026-07-19 to 2026-07-26). That's roughly a **4.5x** speedup over the original human-solo-dev pacing model, which makes sense: this project is being implemented in AI-paired sessions, not solo part-time hours.

The remaining phases don't all scale the same way, though, so this re-forecast applies two different multipliers instead of a flat 4.5x everywhere:

- **Phases 4–8** (Tier 0 build-out — tool-relevance filtering, generation config, context scheduler, caching, delta compression, integration): same *kind* of work as phases 0–3 — design + implement + test a deterministic stage. Scaled at the observed **4.5x**.
- **Phases 9–13** (benchmark suite, Tier 2 extras, distribution, dashboard, docs): partly bound by things that don't compress with coding speed — real, paid API calls across a 15–20+ sample corpus, new heavy dependencies (`torch`/`transformers` for `compress`), external approval/publish steps (PyPI, an MCP client test), and honest human-in-the-loop quality review. Scaled at a more conservative **2x**.

Both multipliers are still guesses extrapolated from four phases of one kind of work — **re-check this again once Tier 0 (phase 8) actually freezes**, the same way the original roadmap already called for.

## Timeline

| Phase | Covers | Effort | Target window |
|---|---|---|---|
| 0 | Finish scaffolding: CI, internal request type, plugin ABC, orchestrator skeleton, no-op round-trip test | done | Jul 19 – Jul 26 (actual) |
| 1 | Cost logger, fail-open framework, tokenizer, deliberately-broken-plugin test | done | Jul 19 – Jul 26 (actual) |
| 2 | Foundation stages: dedup, near-dup (MinHash), schema minification, post-hoc trim | done | Jul 19 – Jul 26 (actual) |
| 3 | Task classifier + 30–50 example hand-labeled validation set, accuracy gate | done | Jul 19 – Jul 26 (actual) |
| 4 | Classifier-dependent stages: tool-relevance filtering, structured-output enforcement, brevity injector, adaptive generation budgets — max_tokens (real-response stress tests) + thinking effort level (validated against the quality-check set, not truncation rate; needs the model-compatibility table) | done | Jul 19 – Jul 26 (actual) |
| 5 | Context scheduler: per-turn classifier, pinning, DP knapsack, greedy fallback, degraded mode, latency benchmark | done | Jul 19 – Jul 26 (actual) |
| 6 | Caching: safety guard (hard-raise tested) → reorder → pricing config table → breakpoint optimizer + synthetic-traffic benchmark | done | Jul 19 – Jul 27 (actual) |
| 7 | Delta compression + `apply_delta()` + round-trip tests | ~1 day | Aug 4 |
| 8 | Assemble Tier 0 in fixed order, end-to-end integration test, stage 4/5 double-prune regression test, **freeze Tier 0 API** | ~2 days | Aug 5 – Aug 6 |
| 9 | Benchmark suite: corpus (15–20+ samples × 4 task-type categories) + held-out quality-check sets, real measured numbers | ~1 week | Aug 7 – Aug 13 |
| 10 | Tier 2 extras: `compress`, `semantic-cache`, TALE budget estimation (each independent, each benchmarked) | ~1.5 weeks | Aug 14 – Aug 23 |
| 11 | Distribution: PyPI publish, MCP server + real-client test | ~3 days | Aug 24 – Aug 26 |
| 12 | Dashboard (CLI table or small local web view, read-only) | ~3 days | Aug 27 – Aug 29 |
| 13 | Docs: architecture doc, benchmark write-up, "considered and rejected" section | ~2 days | Aug 30 – Aug 31 |

**Original total: ~19 weeks (~4.5 months)** at the superseded human-solo-dev pace. **Re-forecast total: ~6 weeks** (2026-07-19 – 2026-08-31) at observed AI-paired velocity, with the phase 9–13 half of that number weighted more conservatively per above.

## Milestones

- **Jul 26 (actual)** — Phases 0–3 done: orchestrator, cross-cutting systems (logger, fail-open framework, tokenizer), all four foundation stages, and the task classifier (accuracy-gated at 85%) all exist.
- **Aug 6 — Tier 0 frozen (re-forecast).** The deterministic core is feature-complete, integration-tested, and API-stable. This is the most load-bearing milestone: per CLAUDE.md, nothing in phases 9–13 should start before it. Re-forecast the rest of the timeline against the *actual* freeze date, not this estimate, once it happens.
- **Aug 13 (re-forecast)** — Benchmark suite produces the first real measured numbers.
- **Aug 31 (re-forecast) — v1 shippable.** PyPI package, MCP server, dashboard, and docs all done.

## Dependencies to respect

These are already implied by CLAUDE.md's build order, but easy to miss in a flat table:

- Phase 4 cannot start until phase 3's task classifier clears its accuracy gate.
- Phase 9 (benchmarking) must not start before phase 8 (Tier 0 freeze).
- Phase 10 (Tier 2 extras) must not start before phase 9 has produced real measured numbers.

## Revisiting this roadmap

- **2026-07-26** — First re-forecast, based on phases 0–3 finishing in 7 actual days against a 4.5-week estimate (~4.5x faster than the original human-solo-dev pacing model). Applied 4.5x to phases 4–8 (same kind of work as 0–3) and a more conservative 2x to phases 9–13 (partly bound by real API costs, new dependencies, and external publish steps — see "Re-forecast" under Assumptions above). Tier 0 freeze moved from ~Oct 10 to ~Aug 6; v1 shippable moved from ~Dec 3 to ~Aug 31.
- Phases 9–13 remain the least confident estimates here since they're furthest out and least like the work actually measured so far. Once Tier 0 is frozen (now ~Aug 6), re-forecast the rest of the roadmap again using actual velocity from phases 0–8 rather than trusting this second guess either.
