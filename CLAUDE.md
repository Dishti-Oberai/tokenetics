# CLAUDE.md

Guidance for Claude Code when working in this repository. See [tokenetics-project-brief.md](tokenetics-project-brief.md) for the full design spec this project was scaffolded from — the decisions in that file are final; this file governs *how* to implement them, not whether to redesign them.

## What this project is

Tokenetics is a stateless Python SDK wrapper that shapes both sides of an Anthropic API request — deduplicating and cache-optimizing what goes in, constraining brevity and token caps on what comes out — to cut token cost without degrading response quality. Core (Tier 0) is deterministic: no model calls, no network calls of its own. See [README.md](README.md) for the full pipeline and architecture.

## Working style

- **Ask before assuming.** If a request doesn't map cleanly onto something already decided in the brief or this file, ask rather than guess — the pipeline order and stage contracts are load-bearing, so a wrong assumption two stages deep is expensive to unwind.
- **Push back on conflicts.** If an instruction would violate a non-negotiable constraint below (reordering the pipeline, a new hard-raise, Tier 2 deps leaking into core, skipping the fail-open contract, etc.), say so and explain the conflict instead of quietly complying or quietly ignoring it. The brief is final on *what* to build; that doesn't mean every incoming instruction about *how* is automatically compatible with it.
- **Plan before touching code.** Before editing, state which stage/file is affected, which existing test(s) it interacts with, and which constraint or build-order phase it belongs to. If that plan doesn't line up with the build order (e.g. it jumps ahead of a phase whose gate hasn't been cleared), flag that before writing code, not after.
- **Re-evaluate after the change.** Once a change is made, check it back against this file: fail-open path covered and tested, conservative default chosen where ambiguous, measured/estimated tags honest, core still stateless, pipeline order untouched. Don't mark work done on the basis of "it runs" alone.

## Non-negotiable constraints

- **The pipeline order is fixed.** Dedup → near-dup → task classifier → schema minification/tool-relevance → context scheduler → delta compression → cache-aware reorder/guard → cache breakpoint optimizer → generation config → API call → post-hoc trim. Do not reorder stages. Do not make ordering configurable, even as an "advanced" option.
- **The core is stateless.** The caller supplies history on every call; there is no hidden persistence between requests. If a feature seems to need state, it belongs in a separate opt-in wrapper layered on top of the stateless core — not in Tier 0.
- **Tier 2 dependencies never leak into the core install.** `torch`, `transformers`, embedding models, vector stores — these live behind `tokenetics[compress]` / `tokenetics[semantic-cache]` extras only. The base `tokenetics` install stays lightweight.
- **The cache-safety guard (stage 7) is the only permitted hard-raise in the project.** Every other failure mode is fail-open. Do not add a new hard-raise without explicit discussion with the user first.
- **Every stage is independently toggleable**, but toggling off a stage that has a defined degraded-mode fallback (e.g. context scheduler → last-N truncation) must engage that fallback and log it as degraded, not silently no-op.

## Fail-open discipline

Three categories, and every new stage must declare which applies where:

1. **Hard error** → skip the stage, pass the request through unmodified, log at error level.
2. **Low-confidence decision** → take the conservative branch (keep the tool, skip the brevity instruction, widen the cap), log at info level.
3. **Partial batch failure** → skip only the failing items, let the stage continue on the rest.

Feature-unsupported-by-the-target-model is its own flavor worth naming explicitly, since it'll recur (adaptive thinking today, other model-specific params later): skip just that sub-stage, log at info level, and don't let it block the rest of the pipeline — e.g. stage 9c's thinking-effort knob skips silently (info-level log) if the target model doesn't support adaptive thinking (pre-4.6 models still require the legacy `budget_tokens` API or lack thinking support entirely — see the model-compatibility table).

Every fail-open path requires a deliberately-broken test case proving the request still goes out unmodified (or degraded-but-safe) rather than raising or corrupting the request.

## Conservative-by-default

Wherever a stage has a choice between an aggressive and a safe option, default to safe:

- Ambiguous tool relevance → keep the tool.
- Ambiguous task shape → skip the brevity instruction (bucket = OFF).
- Uncertain output size → widen `max_tokens`, not narrow it.
- Ambiguous structured-output fit → leave the API param unset.
- Ambiguous thinking-effort fit → don't set `output_config.effort` at all; leave the caller's own `thinking`/`output_config` untouched — don't guess an effort level just because the classifier leans one way.

## Logging and measurement

- Token counts come from **one shared tokenizer utility** — no stage computes its own count a different way.
- Every logged number is tagged `measured` or `estimated`. Never report an estimated number as measured; this tag is load-bearing for the benchmark suite's credibility.
- The cost logger auto-captures per-stage baseline metrics (tokens before/after, timing, on/off) at the orchestrator level; plugins may add richer metadata (confidence, reason, flags) but don't bypass the baseline capture.
- Stage 9c's thinking-effort knob logs two separate `measured` line items: (1) configured `output_config.effort` level vs. actual thinking tokens consumed at generation time, and (2) tokens saved by `display: "omitted"` not being re-billed when history is echoed back on a later turn — these are different cost dimensions (generation-time vs. re-injection-time) and must not be conflated. Both are kept separate from visible-output (max_tokens) savings and reported separately in the benchmark write-up.
- The dashboard is a read-only consumer of the logger, fully decoupled from the request path — it must be able to crash without affecting request handling.

## Plugin contract

Every stage/plugin implements `plugin(request, config, logger) -> request` against the internal request representation (an ABC/protocol), not raw Anthropic API JSON. Conversion between raw API JSON and the internal representation happens once, at the pipeline boundary.

## Incremental runnability

The project must stay demoable against the real Anthropic API from Phase 0 onward — it should never go dark until the phase-8 integration test is the first time it's actually exercised end-to-end. Token savings and reply quality need to be checkable by hand, stage by stage, as each one lands.

- Starting in Phase 0, maintain a small runnable dev script (e.g. `scripts/dev_demo.py` — not part of the pytest suite or CI, since it makes real, billed API calls) that builds a `Tokenetics()` instance from whatever stages currently exist, sends a sample request through it, and calls the real API. It should print: token count before/after the pipeline, which stages actually fired, and the reply text itself.
- Every time a new stage lands, wire it into this same script instead of leaving it orphaned — it should always reflect the pipeline's current state, not just Tier 0's eventual final shape.
- Support an on/off comparison in the script: the same request run with a given stage enabled vs. disabled, so the token-count delta and both replies are visible side by side in one run. This is how "more optimization, same reply quality" actually gets verified by a human per stage — passing unit tests alone don't show that.
- This is deliberately separate from both the automated pytest suite (shouldn't hit the network or cost money on every CI run) and the phase-9 benchmark suite (which needs full measured-not-estimated rigor across a real corpus). Treat it as a fast, informal sanity check — good for "did this change obviously break something," not a substitute for either.

## Testing

- Python only, type hints everywhere, `pytest` for all tests.
- Every new stage requires its own test file before it's considered done.
- Every fail-open path requires a deliberately-broken-plugin test case (per the cross-cutting fail-open framework — this must pass before any real stage is built, per phase 1 below).
- Stage 9c (adaptive generation budgets) is highest test priority: stress-test the max_tokens knob's truncation-monitoring against real responses, not just synthetic ones; validate the thinking-effort knob against the benchmark suite's held-out quality-check set (quality with vs. without effort tuning, per task type) since it has no truncation signal to monitor; and test that `display: "omitted"` doesn't break multi-turn continuation (an echoed-back thinking block with empty text must still be accepted by the API on the next turn).
- Add a regression test that stages 4 (tool-relevance filtering) and 5 (context scheduler) never double-prune the same content.
- The cache-safety guard needs a standalone hard-raise test before reorder or the breakpoint optimizer are built on top of it.

## Build order

Follow this order — it's already planned, don't reshuffle it:

0. Scaffolding: repo, `pyproject.toml`, pytest, CI, internal request type, plugin ABC, orchestrator skeleton, no-op plugin round-trip test.
1. Cross-cutting first: cost logger, fail-open framework, tokenizer. The deliberately-broken-plugin test must pass before any real stage is written.
2. Foundation stages (independent of each other): dedup, near-dup, schema minification, post-hoc trim.
3. Shared task classifier + a small hand-labeled validation set (30-50 examples) — clear the accuracy gate before anything downstream consumes its output.
4. Classifier-dependent stages: tool-relevance filtering, structured-output enforcement, brevity injector, adaptive generation budgets (max_tokens + thinking effort level).
5. Context scheduler: per-turn classifier → pinning pre-allocation → DP knapsack → greedy fallback → degraded-mode truncation. Needs a latency benchmark to justify the DP-to-greedy threshold.
6. Caching: safety guard first (standalone, hard-raise tested) → reorder → pricing config table → breakpoint optimizer + synthetic-traffic benchmark (the headline number).
7. Delta compression + `apply_delta()` + round-trip tests.
8. Assemble Tier 0 in fixed order; end-to-end integration test; freeze the Tier 0 API.
9. Benchmark suite — real measured numbers, before any extras are built.
10. Tier 2 extras (`compress`, `semantic-cache`, TALE budget estimation) — each independent.
11. Dashboard.
12. Distribution: PyPI publish, MCP server + a real-client test.
13. Docs: architecture doc, benchmark write-up (methodology + honest caveats), "considered and rejected" section.

Don't jump ahead — e.g. don't build the context scheduler before the task classifier has cleared its accuracy gate, and don't start Tier 2 extras before Tier 0 is frozen and benchmarked.

## Pricing and compatibility data

Cache pricing (write/read rates, TTL tier thresholds) lives in a **versioned config table with a "last verified" date**. Never hardcode a price inline in stage logic — if you need a price, read it from that table, and if the table looks stale, flag it rather than silently trusting it.

The same discipline applies to model capability data: stage 9c's mapping of model generations to their accepted thinking API surface (`adaptive` + `output_config.effort` vs. the legacy `budget_tokens` vs. no thinking support at all) lives in its own versioned, dated config table — never as an inline version check hardcoded into stage logic.

## Honest benchmarking

- Report savings as ranges tied to a stated workload assumption (e.g. "12-18% on tool-call-heavy conversational workloads"), never a single flat percentage.
- Document where a stage can go net-negative — e.g. the brevity injector's ~50-100 token instruction overhead can exceed savings on already-short requests. This belongs in the write-up, not buried.
- The benchmark corpus is organized by the classifier's task-type categories, with a separate held-out quality-check set for anything risk-bearing (brevity, cap tuning, thinking-effort tuning, pruning, lossy compression).

## Scope discipline

- Distribution target for v1 is PyPI core package + a thin MCP server layer. A Claude Code skill is explicitly deferred — don't build one unless asked.
- Don't compare against or name competitor projects in docs — this is a repo-wide documentation convention, not just a README rule.
- Don't add a fourth tier, a new hard-raise, or configurable pipeline ordering without raising it with the user first — these are locked design decisions, not open questions.
