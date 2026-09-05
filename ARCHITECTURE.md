# Tokenetics Architecture

This document describes how Tokenetics is built internally — the pipeline, the plugin contract, the fail-open framework, and the supporting systems (logging, dashboard, MCP server, versioned config tables). It's aimed at anyone extending or reviewing the codebase, not at someone just using the library — for that, see [README.md](README.md). The design decisions here are locked (see [tokenetics-project-brief.md](tokenetics-project-brief.md)); this document explains *how* they're implemented, and [CLAUDE.md](CLAUDE.md) is the authoritative source on the constraints that must never be violated. [ROADMAP.md](ROADMAP.md) has the full, dated build history and every real benchmark result this document summarizes.

## Three tiers

- **Tier 0 (core, always on)** — every stage described below. Fully deterministic: no model calls, no network calls of its own except the shared tokenizer's real `/v1/messages/count_tokens` calls (needed to log real, measured token counts — not a generation call, and not billed as one). Stateless: the caller supplies conversation history on every call; nothing persists between requests inside Tokenetics itself.
- **Tier 2 (opt-in extras)** — `tokenetics[compress]`, `tokenetics[semantic-cache]`, and TALE-style budget estimation, each its own pip extra, each independent of the others and of Tier 0. Heavier dependencies (`torch`, `transformers`, `sentence-transformers`) live only behind these extras — the base `tokenetics` install never pulls them in (verified: `pyproject.toml`'s `dependencies` list is just `anthropic`).
- There is no Tier 1 and no Tier 3. Adding a fourth tier is explicitly out of scope without discussing it with the project owner first (see CLAUDE.md's scope-discipline section).

## The internal request representation

Every stage operates on `TokeneticsRequest` (`src/tokenetics/core/request.py`), not raw Anthropic API JSON. Conversion happens exactly twice, at the pipeline's boundary:

- `from_api_kwargs(**api_kwargs) -> TokeneticsRequest` — parses raw `messages.create()`-style kwargs (called once, at the top of `Tokenetics.prepare()`). Also extracts any `cache_control` already present on `system`/`tools` blocks into `TokeneticsRequest.cache_breakpoint`, so a previously-cached request that's echoed back as `previous_request` round-trips correctly.
- `to_api_kwargs(request) -> dict[str, Any]` — converts back to a plain dict ready for `client.messages.create(**result)` (called once, at the bottom of `prepare()`).

No stage ever touches raw API JSON directly. This is what lets every stage be written, tested, and reasoned about against one stable shape regardless of how the Anthropic API's own wire format evolves.

## The fixed 11-stage pipeline

Pipeline order is a non-negotiable constraint — it is never reordered, and ordering is never exposed as a configuration option, even an "advanced" one. Each stage can be independently toggled off, but the sequence itself is fixed because later stages depend on invariants earlier ones establish (e.g. stage 8's cache-tier decision assumes stage 7 has already arranged stable content first and verified nothing before the cache point silently changed).

| # | Stage | Module | What it does |
|---|---|---|---|
| 1 | Dedup | `stages/dedup.py` | Removes exact-duplicate content by fingerprint, keeping the first occurrence. |
| 2 | Near-dup | `stages/near_dup.py` | MinHash/shingling-based similarity merge — stricter threshold for real conversation turns than tool output; never touches the most recent turn; every merge logged with its similarity score. |
| 3 | Task classifier | `stages/task_classifier.py` | Cheap, rule-based (no model call) labeling of the current task type (code / conversational / extraction / tool-heavy / unclassified). Result is reused by stages 4, 5, 8, and 9 rather than re-derived. |
| 4 | Schema minification / tool-relevance | `stages/schema_minification.py` | Strips unnecessary bulk from tool schemas; drops a tool only when it's clearly irrelevant to the current turn — ambiguous cases keep the tool (conservative default). |
| 5 | Context scheduler | `stages/context_scheduler.py` | Per-turn classification (decision/error/tangent/etc.) → pinning → 0/1 knapsack DP solver to pick the best-fitting subset of history under a token budget → greedy fallback above a latency-justified size threshold → last-N truncation as the degraded-mode fallback when the stage is disabled. |
| 6 | Delta compression | `stages/delta_compression.py` | For a `tool_result` block with a supplied previous version, sends a computed diff (JSON-patch or line-patch) instead of the full payload, when the diff is meaningfully smaller. `core/delta.py` owns `apply_delta()` for the reverse operation. |
| 7 | Cache reorder + safety guard | `stages/cache_reorder_guard.py` | Reorders content so stable material (system, tools) precedes frequently-changing material, maximizing what stays cacheable. Then the **safety guard**: if content at or before the last cache breakpoint differs from the caller-supplied `previous_request`, raises `CacheSafetyError` — the **only** hard-raise in the project (see below). |
| 8 | Cache breakpoint optimizer | `stages/cache_breakpoint_optimizer.py` | Decides *whether* to place a `cache_control` marker, *where* (system vs. tools, per Anthropic's tools→system→messages cache hierarchy), and *which* TTL tier (5-minute vs. 1-hour), from real observed repeat-call history (`config["cache_usage_history"]`). Tier choice is based on the **worst** observed repeat gap, not the average (a conservative-by-default fix — see ROADMAP.md's cache_breakpoint_optimizer history for the real bug this fixed). |
| 9a | Structured output | `stages/structured_output.py` | Constrains the API's response format when the task is confidently data-shaped (extraction/classification) and a matching tool is registered; respects any `tool_choice` the caller already set. |
| 9b | Brevity injector | `stages/brevity_injector.py` | Adds a short terseness instruction, off by default, on only for task types where brevity is confidently safe (documented net-negative risk: ~50-100 token overhead can exceed savings on already-short requests). |
| 9c | Adaptive budget | `stages/adaptive_budget.py` | Widens (never narrows) `max_tokens` based on observed truncation rate; separately, an **opt-in only** (`config["enable_thinking_effort"] = True`) thinking-effort knob routes reasoning depth by task type, reading model support from `core/model_compatibility.py` and skipping silently on models that don't support it. |
| — | *(the actual API call)* | *(caller's own code)* | Tokenetics never calls the Anthropic API to generate a completion — `prepare()` returns kwargs, the caller runs `client.messages.create(**result)`. |
| 11 | Post-hoc trim | `stages/post_hoc_trim.py` | Response-side (runs in `finalize()`, not `prepare()`): strips known boilerplate from the reply text before it's stored for reuse as future context. The only response-side stage that currently exists. |

Stage numbering follows the brief's original scheme (there is no "stage 10" — 9a/9b/9c are sub-stages of 9, and the API call itself sits between the request-side stages and stage 11).

## The plugin contract

Every request-side stage subclasses `Stage` (`core/plugin.py`):

```python
class Stage(ABC):
    name: str
    def run(self, request: TokeneticsRequest, config: StageConfig, logger: CostLogger) -> TokeneticsRequest: ...
    def degraded_fallback(self, request, config, logger) -> TokeneticsRequest: return request  # only overridden where a real fallback exists
    def note(self, **extra) -> None: ...  # attaches metadata to this run's single log_stage() call
```

The response-side equivalent, `ResponseStage` (`core/response_stage.py`), has the same shape but operates on `(text: str, config, logger) -> str` instead of a `TokeneticsRequest`, since a reply and a request are different shapes and stage 11 runs after the API call, not before it.

Stages call `self.note(**extra)` to attach rich metadata (confidence, reason, counts) but never call `logger.log_stage()` themselves — the orchestrator owns exactly one `log_stage()` call per stage per run, merging in whatever the stage noted. This is deliberate: it guarantees the baseline capture (tokens before/after, timing, enabled/error) can never be bypassed by a stage, per CLAUDE.md's plugin contract.

## The orchestrator (`orchestrator.py`)

`Tokenetics.prepare(**api_kwargs)` runs the request through all 11 request-side stages in fixed order and returns API-ready kwargs. `Tokenetics.finalize(response)` extracts the reply's text (stripping trailing whitespace immediately — the real Messages API rejects assistant content ending in trailing whitespace) and runs it through the response-side stages.

**Fail-open framework**: `_run_stage`/`_run_response_stage` wrap every stage call. Three categories, all handled the same mechanical way but declared per-stage in that stage's own docstring:

1. **Hard error** (an exception escapes `stage.run()`) → caught, logged at error level, the request/text is passed through **completely unmodified** from before that stage.
2. **Low-confidence decision** → the stage itself takes the conservative branch (this is stage-internal logic, not something the orchestrator enforces) — e.g. schema_minification keeps an ambiguous tool, adaptive_budget widens rather than narrows.
3. **Partial batch failure** → also stage-internal — a stage skips just the failing item and continues on the rest.

**The one exception**: `CacheSafetyError` (`core/errors.py`), raised by stage 7's safety guard, is allowed to propagate all the way out of `Tokenetics.prepare()` instead of being caught like every other exception. This is the sole hard-raise in the entire project, and it exists because silently continuing would mean paying full price on every future call while believing caching is still active — a cost regression with no error to reveal it. Every other failure mode prioritizes "never break the request" over "never miss a token."

Every `log_stage()` call is tagged `phase="request"` or `phase="response"` — this exists specifically so the dashboard's aggregation never mixes request-token counts with reply-token counts (a real bug found and fixed during Phase 11; see ROADMAP.md).

## Versioned config tables

Two tables carry real, externally-sourced data that must never be hardcoded inline in stage logic, each with a `LAST_VERIFIED` date and a source URL:

- **`core/cache_pricing.py`** — cache write/read multipliers, TTL tier windows, per-model base input/output prices, and `MIN_CACHEABLE_TOKENS` per model (below this length, Anthropic's real API silently never caches at all — no error, just no discount). Last verified 2026-08-02 against Anthropic's live pricing and prompt-caching docs.
- **`core/model_compatibility.py`** — which thinking API surface each model generation accepts (`adaptive` + `output_config.effort`, the legacy `budget_tokens`, or no thinking support at all). Unknown/unlisted models conservatively resolve to `unsupported` rather than guessing at a shape that could 400 the request. Last verified 2026-07-26.

If either table looks stale when you're reading it, re-verify against the source URL in the file rather than trusting the date blindly — this is a "flag it" situation per CLAUDE.md, not a "silently keep using it" one.

## Cost logging (`core/logger.py`)

`CostLogger` is a Protocol with `log_stage()` (called once per stage per run, by the orchestrator only) and `log_event()` (a sibling for data that exists *outside* the pipeline entirely — real `usage.cache_read_input_tokens` from the caller's own `messages.create()` call, TALE's round-trip cost, compress's achieved ratio, semantic-cache's hit/miss — none of which `Tokenetics.prepare()`/`finalize()` ever sees directly). Three implementations:

- `InMemoryCostLogger` — the default; keeps entries in a list for the lifetime of one `Tokenetics()` instance.
- `NullLogger` — logs nothing, explicitly.
- `FileCostLogger` — appends every entry as one JSON line to a file, tagged with a UUID `run_id` shared across all of one instance's `log_stage()`/`log_event()` calls. This is the durable log the dashboard reads; nothing else in the project persists cost data anywhere.

Every logged number carries a `measured` or `estimated` tag — this is load-bearing for the benchmark suite's credibility and is never fudged: an estimated number is never reported as measured.

## The dashboard (`src/tokenetics/dashboard/`)

A local, read-only web view — deliberately decoupled from the request path (it can crash without affecting a single API call, since it runs in a separate process and only ever reads the log file, never writes to it).

- `aggregate.py` — `load_entries()` (skips malformed lines rather than raising) and `aggregate()` (per-stage rollup + overall request-side savings, filtered by the `phase` tag above), plus `aggregate_events()` for the Tier 2/cache section (cache hit rate, TALE cost, compress ratio, semantic-cache hit rate).
- `render.py` — one self-contained HTML page (inline CSS, no JS framework, no charting library, no new dependency). Real numbers where data exists, an honest "no data logged (needs --flag)" placeholder where it doesn't — never a fabricated zero passed off as a measurement.
- `server.py` — `http.server.ThreadingHTTPServer` (stdlib only). Re-aggregates the log file fresh on every request rather than caching state, so a second process appending to the same log shows up on the next page load.

Run via `uv run python scripts/dashboard.py --log-file costs.jsonl`.

## The MCP server (`src/tokenetics/mcp_server.py`)

Per the brief's distribution plan — "core PyPI package + MCP server (thin layer importing the core)." A single tool, `optimize_request(request: dict, stage_config: dict | None) -> dict`, forwards straight to `Tokenetics().prepare()` and returns the optimized kwargs. Deliberately does **not** call the real Anthropic API itself — the caller's own MCP client still makes that call — keeping this layer free of API keys and its own billed-call surface, in the same spirit as Tier 0's "no network calls of its own" principle. Built on the official `mcp` SDK (`mcp[cli]`, its own pip extra — `tokenetics[mcp]`, kept out of the base install like the Tier 2 extras). Run via the `tokenetics-mcp` console script, over stdio transport.

## Benchmarking (`scripts/`)

- **`scripts/dev_demo.py`** — the fast, informal, per-stage sanity check. Builds a `Tokenetics()` from whatever stages currently exist, runs a sample scenario against the real API, prints token counts before/after and which stages fired. Not part of pytest/CI (it makes real, billed calls) and not a substitute for the benchmark suite below — it exists so the project stays demoable and hand-checkable at every phase, per CLAUDE.md's incremental-runnability requirement.
- **`scripts/benchmark_runner.py`** — the rigorous suite, cost-ceiling-gated (`--dry-run` by default, `--confirm-spend` required to actually spend money): `corpus` (per-category token-savings pass over `benchmarks/corpus/`), `quality-check` (held-out quality rubric set for anything risk-bearing — brevity, cap tuning, thinking effort, pruning), `usage` (full real-dollar cost, one-shot per sample), `session` (a realistic multi-turn conversation, measuring what one-shot sampling structurally can't see: cache amortization and accumulated-redundancy pruning).
- **`scripts/cache_benchmark.py`**, **`scripts/semantic_cache_benchmark.py`**, **`scripts/compress_benchmark.py`** — free/local synthetic benchmarks for the cache breakpoint optimizer and the two Tier 2 extras that ship their own quality/effectiveness report.

Every number these produce is reported as a range tied to a stated workload assumption, never a single flat percentage — see [README.md](README.md)'s "Real-world impact" section for the current headline results, and ROADMAP.md for the full run-by-run history.

## Testing conventions

Every stage has its own test file, plus a deliberately-broken-plugin fail-open test proving the request still goes out unmodified (or degraded-but-safe) rather than raising or corrupting it. `scripts/*.py` files are validated by `--dry-run`/manual smoke-testing and real-client checks (e.g. the MCP server's real stdio round-trip), not direct pytest coverage — the logic they call already has full unit coverage, and what's left to verify is real-world behavior a synthetic test can't substitute for.
