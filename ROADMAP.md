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
| 5 | Context scheduler: per-turn classifier, pinning, DP knapsack, greedy fallback, degraded mode, latency benchmark | ~2 days | Jul 30 – Jul 31 |
| 6 | Caching: safety guard (hard-raise tested) → reorder → pricing config table → breakpoint optimizer + synthetic-traffic benchmark | ~3 days | Aug 1 – Aug 3 |
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
