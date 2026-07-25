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

## Assumptions

- **Pace**: solo, ~4 hours/day, ~5–6 days/week (~20–24 hrs/week).
- **Confidence**: estimates for phases 0–8 (Tier 0) are moderately confident. Phases 9–13 (benchmark suite, Tier 2 extras, distribution, dashboard, docs) are rougher and should be re-forecast once Tier 0 ships — actual velocity through 0–8 will be better evidence than this initial guess.
- **Dates**: computed from today, 2026-07-19, at the pace above. If the pace or start date changes, these windows need to be recomputed — they aren't self-updating.
- **Scope**: full v1 as specified in the brief — nothing cut. This was a deliberate choice: the alternative (a minimal Tier 0 slice, or thin-rigor stages) was offered and rejected in favor of keeping the brief's rigor requirements (measured-not-estimated benchmarks, hand-labeled classifier validation set, fail-open test coverage per stage, quality-degradation benchmarks for lossy compression) intact.

## Timeline

| Phase | Covers | Effort | Target window |
|---|---|---|---|
| 0 | Finish scaffolding: CI, internal request type, plugin ABC, orchestrator skeleton, no-op round-trip test | ~1 week | Jul 20 – Jul 26 |
| 1 | Cost logger, fail-open framework, tokenizer, deliberately-broken-plugin test | ~1 week | Jul 27 – Aug 2 |
| 2 | Foundation stages: dedup, near-dup (MinHash), schema minification, post-hoc trim | ~1.5 weeks | Aug 3 – Aug 12 |
| 3 | Task classifier + 30–50 example hand-labeled validation set, accuracy gate | ~1 week | Aug 13 – Aug 19 |
| 4 | Classifier-dependent stages: tool-relevance filtering, structured-output enforcement, brevity injector, adaptive generation budgets — max_tokens (real-response stress tests) + thinking effort level (validated against the quality-check set, not truncation rate; needs the model-compatibility table) | ~2 weeks | Aug 20 – Sep 2 |
| 5 | Context scheduler: per-turn classifier, pinning, DP knapsack, greedy fallback, degraded mode, latency benchmark | ~1.5 weeks | Sep 3 – Sep 12 |
| 6 | Caching: safety guard (hard-raise tested) → reorder → pricing config table → breakpoint optimizer + synthetic-traffic benchmark | ~2 weeks | Sep 13 – Sep 26 |
| 7 | Delta compression + `apply_delta()` + round-trip tests | ~0.75 week | Sep 27 – Oct 3 |
| 8 | Assemble Tier 0 in fixed order, end-to-end integration test, stage 4/5 double-prune regression test, **freeze Tier 0 API** | ~1 week | Oct 4 – Oct 10 |
| 9 | Benchmark suite: corpus (15–20+ samples × 4 task-type categories) + held-out quality-check sets, real measured numbers | ~2 weeks | Oct 11 – Oct 24 |
| 10 | Tier 2 extras: `compress`, `semantic-cache`, TALE budget estimation (each independent, each benchmarked) | ~3 weeks | Oct 25 – Nov 14 |
| 11 | Distribution: PyPI publish, MCP server + real-client test | ~0.75 week | Nov 15 – Nov 21 |
| 12 | Dashboard (CLI table or small local web view, read-only) | ~0.75 week | Nov 22 – Nov 28 |
| 13 | Docs: architecture doc, benchmark write-up, "considered and rejected" section | ~0.5 week | Nov 29 – Dec 3 |

**Total: ~19 weeks (~4.5 months) to full v1** at the stated pace.

## Milestones

- **Aug 2** — Phase 0–1 done: orchestrator + cross-cutting systems (logger, fail-open framework, tokenizer) exist.
- **Oct 10 — Tier 0 frozen.** The deterministic core is feature-complete, integration-tested, and API-stable. This is the most load-bearing milestone: per CLAUDE.md, nothing in phases 9–13 should start before it.
- **Oct 24** — Benchmark suite produces the first real measured numbers.
- **Dec 3 — v1 shippable.** PyPI package, MCP server, dashboard, and docs all done.

## Dependencies to respect

These are already implied by CLAUDE.md's build order, but easy to miss in a flat table:

- Phase 4 cannot start until phase 3's task classifier clears its accuracy gate.
- Phase 9 (benchmarking) must not start before phase 8 (Tier 0 freeze).
- Phase 10 (Tier 2 extras) must not start before phase 9 has produced real measured numbers.

## Revisiting this roadmap

Phases 9–13 are the least confident estimates here since they're furthest out. Once Tier 0 is frozen (~Oct 10), re-forecast the rest of the roadmap using actual velocity from phases 0–8 rather than trusting these initial guesses.
