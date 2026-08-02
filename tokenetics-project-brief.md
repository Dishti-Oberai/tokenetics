# Tokenetics — Project Brief (Context for Claude Code)

> Purpose of this file: complete context handoff from the design phase. Use this to generate the README.md, CLAUDE.md, and initial project scaffolding. All decisions below are FINAL and locked — do not redesign them, implement them.

## One-line description

Tokenetics is a stateless Python SDK wrapper that shapes both sides of an Anthropic API request — deduplicating, cache-optimizing, and context-scheduling what goes in, and constraining brevity, output structure, and token caps on what comes out — to cut token cost without degrading response quality, backed by a measured (not just modeled) benchmark suite and an opt-in tier of heavier compression/caching plugins.

## Positioning (for README)

- **What it is**: a request-level SDK wrapper for applications calling the Anthropic API directly (`client.messages.create()`-style usage). Import it into your code; it shapes each request before it's sent and constrains what the model generates.
- **What it is NOT**: not a session/agent-harness plugin (does not hook into Claude Code / coding-assistant sessions), not a proxy/gateway, not an LLM-calls-LLM compressor by default.
- **Core differentiators**: (1) dual-direction — optimizes both input AND output tokens, (2) cache breakpoint optimizer that models Anthropic's actual cache pricing/TTL tiers, (3) measured-vs-estimated benchmark discipline, (4) strict fail-open safety contract.
- Do NOT name or compare against competitor projects in the README. Focus on what the project consists of.

## Project facts

- **Name**: Tokenetics. Package: `tokenetics` (PyPI).
- **Language**: Python only (v1).
- **Distribution**: core PyPI package + MCP server (thin layer importing the core). Claude Code skill deferred.
- **Design principles**: stateless core (caller supplies history), deterministic Tier 0 (no LLM/model calls in default path), every stage independently toggleable, fixed pipeline order, fail-open everywhere except one deliberate hard-raise.

## Architecture: three tiers

- **Tier 0 (core, always available, on by default)**: all deterministic stages listed below. No model calls, no network calls of its own.
- **Tier 2 extras (opt-in, disabled by default, separate pip extras)**:
  - `tokenetics[compress]` — LLMLingua-style compression via small local model (GPT-2-small class). Ships its own quality-degradation benchmark. Fail-open if model can't load.
  - `tokenetics[semantic-cache]` — semantic response caching: local embedding model + lightweight local vector store, conservative similarity threshold, TTL expiry + manual `invalidate()`/`clear()`, hit/miss + match-quality logging. Fail-open to normal API call.
  - TALE-style budget estimation — opt-in upgrade to the response-budget injector's classifier slot (small LLM call estimates a token budget). Benchmarked head-to-head vs. the free heuristic. Own cost-accounting.

## Fixed pipeline order (sequential, NOT parallel; order is non-configurable)

1. **Dedup (exact-match)** — hash-based, always-on.
2. **Near-dup detection** — MinHash/shingling. Dual thresholds: loose (~85-90%) for tool outputs/re-reads, strict (~95%+) for conversation turns. NEVER touches the most recent turn. Logs every merge with similarity score.
3. **Task classifier** — cheap deterministic regex/keyword classifier, runs ONCE, output (`task_type`, `confidence`) shared by stages 4, 5, 8, 9, 10.
4. **Schema minification + tool-relevance filtering** — generic/schema-agnostic (NOT per-tool handlers). Relevance filtering is conservative: ambiguous → keep the tool. Logs every dropped tool with score.
5. **Context scheduler** — turn-level retention scoring + 0/1 knapsack solved with dynamic programming (budget discretized into 50-100 token buckets). Per-turn activity classifier (code/error/decision/small-talk). Pinned categories (explicit decisions, unresolved errors) are pre-allocated BEFORE the DP solve — never prunable by score. Auto-fallback to greedy approximation above a size threshold (logged). If this stage is disabled entirely, a minimal last-N-turns fallback truncation activates and is logged as degraded mode.
6. **Structured delta compression** — caller supplies previous version; wrapper produces JSON patch/unified diff when diff is meaningfully smaller than full content (size-threshold fallback to full payload). Ships `apply_delta()` companion helper for reconstruction.
7. **Cache-aware reorder + cache-safety guard** — guard diffs incoming messages against caller-supplied previous request; anything before the last cache breakpoint must be byte-identical. Violation = HARD RAISE (the only non-fail-open behavior in the project).
8. **Cache breakpoint optimizer** — headline feature. Derives content-repeat patterns from caller-supplied history, computes optimal `cache_control` breakpoint placement and TTL tier choice (5-min vs 1-hour) using Anthropic's cache write/read pricing. Pricing lives in a versioned config table with a "last verified" date — never hardcoded silently.
9. **Generation config (pre-send)**: (a) structured-output enforcement — auto-set API param when classifier flags extraction/classification-shaped tasks; (b) response-budget injector — three instruction buckets (off/moderate/aggressive terse, Chain-of-Draft style), conservative default = OFF unless clearly terse-appropriate; (c) adaptive generation budgets — max_tokens/stop-sequences (task-type-derived cap + 15-20% safety margin, truncation-rate monitored, auto-widens margin per task type if truncation rate exceeds ~1-2%, logged) **and** thinking effort (task-type-derived `output_config.effort` level via `thinking: {type: "adaptive"}` off the same classifier signal — never the legacy `budget_tokens` API, which 400s on current-generation models; leaves caller's config untouched if classifier is uncertain; `display: "omitted"` avoids re-billing old reasoning traces as input tokens on future turns; monitored via the benchmark suite's quality-check set, not truncation rate; fails open on models without adaptive-thinking support).
10. **API call sent** (single call).
11. **Post-hoc output trim** — deterministic boilerplate stripping only (no summarization) before response is stored/re-injected as future context.

## What each technique exactly does (mechanism-level detail)

> This section eliminates context gaps: for each stage — the problem it solves, the exact transformation it performs, its inputs/outputs, and why it saves tokens.

### 1. Dedup (exact-match)
- **Problem**: the same block of text (repeated instructions, boilerplate, re-pasted content) appears more than once in the messages array, paying full token price each time.
- **Mechanism**: hash each message/content block (e.g. SHA-256 of normalized text). Maintain a set of seen hashes within the current request. On a repeat hash, remove the duplicate block (keep the first occurrence, since later stages and cache safety depend on earlier positions staying stable).
- **In → out**: messages array → same array with literal duplicate blocks removed.
- **Saves tokens by**: not sending identical bytes twice in one request.

### 2. Near-dup detection (MinHash/shingling)
- **Problem**: content that is ~equivalent but not byte-identical (a re-run tool output that changed trivially, two turns restating the same thing) escapes exact-match dedup.
- **Mechanism**: split each candidate block into overlapping n-gram shingles → compute MinHash signatures → estimate Jaccard similarity between block pairs cheaply. If similarity exceeds the threshold for that content class (tool outputs ~85-90%, conversation turns ~95%+), drop/merge the older near-duplicate, keeping the newer one. The most recent turn is exempt entirely.
- **In → out**: messages array → array with older near-duplicate blocks removed; each removal logged with its similarity score.
- **Saves tokens by**: collapsing "same information, slightly different bytes" down to one copy.

### 3. Task classifier (shared signal, not itself an optimizer)
- **Problem**: several downstream stages need to know what *kind* of task this request is, and each guessing independently would be duplicated, inconsistent logic.
- **Mechanism**: deterministic regex/keyword scoring over the latest user message (+ light context): detects code blocks/fences, math notation, error traces, extraction-style phrasing ("list all", "extract", "return JSON"), question simplicity markers. Emits `(task_type ∈ {code, conversational, extraction, tool-heavy, ...}, confidence)`. Runs once per request; result attached to the internal request object for stages 4, 5, and 9 to read.
- **Saves tokens by**: nothing directly — it is the decision signal that lets other stages act safely.

### 4. Schema minification + tool-relevance filtering
- **Problem**: every registered tool's full JSON schema is sent with every request, even tools irrelevant to the current turn; schemas themselves carry verbose descriptions.
- **Mechanism**: (a) minification — strip schema fields that don't affect model behavior (defaults duplicated in descriptions, redundant whitespace, unused metadata), purely structurally, works on any schema; (b) relevance filtering — score each tool by cheap lexical overlap between its name/description and recent conversation content; drop only tools scoring clearly-irrelevant, keep anything ambiguous. Every drop logged with score.
- **In → out**: tools array → smaller tools array with slimmer schemas.
- **Saves tokens by**: not paying for tool definitions the model won't need this turn.

### 5. Context scheduler (DP knapsack)
- **Problem**: naive history handling (keep everything, or keep last N turns) either wastes tokens on low-value turns or blindly cuts high-value ones; long irrelevant context also measurably degrades model accuracy (context rot).
- **Mechanism**: (1) per-turn activity classifier (regex/keyword: code / error trace / explicit decision / tool result / small talk) assigns each turn a type; (2) each turn gets a retention-value score from type weight + recency; (3) pinned types (explicit decisions, unresolved errors) have their token cost reserved off the budget first and are never candidates for removal; (4) remaining turns + remaining budget are solved as a 0/1 knapsack via dynamic programming (budget discretized into 50-100-token buckets): maximize total retention value subject to total tokens ≤ budget; (5) if turn count/budget exceeds a size threshold, automatically switch to a greedy value-per-token approximation (logged); (6) if the whole stage is disabled, a naive last-N-turns truncation runs instead, logged as degraded mode.
- **Error-resolution pinning (resolved 2026-07-26)**: an `error`-classified turn is pinned by default and stays pinned unless a *later* turn contains an explicit, positive resolution signal from the user (e.g. "that fixed it," "works now," "thanks, resolved") — computed fresh from the caller-supplied history each call, never persisted. Conversation simply moving to another topic is **not** treated as resolution; that's weak evidence and the wrong failure direction (misjudging a live problem as resolved is worse than pinning a few already-resolved ones).
- **Token budget (resolved 2026-07-26)**: caller-supplied via `config["token_budget"]` when given (same caller-supplies-it-per-call shape as stage 6's previous payload, stage 7's previous request, and stage 9c's truncation-rate stat). When not supplied, falls back to a default derived from the target model's context window (new versioned lookup table, same "last verified" discipline as the model-compatibility table) **scaled by the request's `task_type`** from stage 3's classifier — e.g. code-heavy conversations retain a larger share of the context window than small-talk-shaped ones. Logged as `estimated`, not `measured`, since it's a heuristic default, not a value the caller actually supplied.
- **In → out**: full conversation history + token budget → subset of turns that fits the budget with maximum retained value.
- **Saves tokens by**: sending only the turns that matter; can also *improve* accuracy by shrinking the irrelevant haystack.

### 6. Structured delta compression
- **Problem**: agentic loops re-send large, mostly-unchanged payloads (a re-read file after a small edit, a polled API response) at full price every time.
- **Mechanism**: caller passes the previous version of the payload alongside the new one. Compute a unified diff / JSON patch. If the diff is meaningfully smaller than the full content (below the size-ratio threshold), send the diff annotated as a delta; otherwise fall back to sending full content. The companion `apply_delta(previous, delta)` utility reconstructs the full payload for local use.
- **In → out**: (previous payload, new payload) → delta or full payload, whichever is smaller under the threshold.
- **Saves tokens by**: paying only for what changed, not the entire payload again.

### 7. Cache-aware reorder + cache-safety guard
- **Problem**: Anthropic's prompt cache is prefix-based — if any byte before a cache breakpoint changes or moves between calls, the cache misses and every subsequent call re-pays full price. Careless "optimization" of already-sent content silently *increases* cost.
- **Mechanism**: (a) reorder — for content not yet sent, place stable blocks (system prompt, tool defs, standing instructions) before volatile blocks so the stable prefix is maximal; (b) guard — diff the incoming messages array against the caller-supplied previous request; if anything at or before the last cache breakpoint differs, HARD RAISE with a descriptive error (sole non-fail-open behavior in the project). Never reorder retroactively.
- **Hard-raise contract (resolved 2026-07-27)**: a dedicated `CacheSafetyError` (new `core/errors.py`), not a bare exception type — specifically catchable by callers, distinct from every other exception in the codebase (which the orchestrator silently fail-opens). The message identifies exactly what changed (e.g. which message index, or "system prompt", or "tools") so it's actionable, not just "cache mismatch."
- **Reorder scope (resolved 2026-07-27)**: today's request shape already emits system → tools → messages in a stable order by construction (`to_api_kwargs`), so there's little to physically reorder yet. Still implemented as a real stage now — a normalize-and-verify step defensively re-asserting the stable-before-volatile invariant (protecting against a caller-supplied `extra` override disrupting it) — rather than skipped or spuriously built out for structure the data model doesn't have yet. Expected to grow real reordering logic once delta compression (stage 6) or multi-block tool/system structures land.
- **In → out**: request + previous request → identically-prefixed request with new content optimally ordered, or an explicit error.
- **Saves tokens by**: keeping the cached prefix valid so repeat content bills at cache-read rates (~90% cheaper) instead of full price.

### 8. Cache breakpoint optimizer (headline feature)
- **Problem**: developers place one `cache_control` breakpoint by habit (end of system prompt) with no cost model; breakpoint position and TTL tier choice (5-min vs 1-hour) are actually an optimization problem over write-cost vs read-discount vs how often/when content repeats.
- **Mechanism**: (1) derive a usage model from caller-supplied history — which content blocks repeat across calls and with what time gaps; (2) evaluate candidate breakpoint positions: a breakpoint is worth placing where the content before it is stable across calls and large enough that read-discounts outweigh the write premium; (3) choose TTL tier by comparing expected repeat interval to tier windows under the pricing table (e.g. gaps >5min make the 1-hour tier's higher write cost worth it); (4) emit the request with `cache_control` markers placed accordingly. All prices come from the versioned pricing config (with "last verified" date), never hardcoded inline.
- **Caller-supplied history shape (resolved 2026-07-27)**: `config["cache_usage_history"]` — a list of small `{"timestamp": float, "content_hash": str}` records, not full past `TokeneticsRequest` objects. Keeps the same "cheap, caller-owned, no persistence" shape every other caller-supplied input already uses (delta compression's previous payload, the safety guard's `previous_request`, `truncation_stats`, context scheduler's `token_budget`) instead of requiring the caller to hand over entire historical request bodies just to estimate a repeat interval.
- **Pricing numbers (resolved 2026-07-27)**: sourced from a live, dated lookup against Anthropic's current cache pricing docs as the first implementation step for the pricing config table — never seeded with a remembered/guessed figure. Getting a cache price wrong doesn't just mis-optimize like a stale model-capability entry would; it actively costs the caller money on every cache write, so this table's "last verified" discipline starts at the sourcing step, not after.
- **In → out**: request + usage history → same request with cost-optimal `cache_control` placement and tier.
- **Saves tokens by**: nothing removed — it converts as many tokens as possible from full-price to cache-read pricing, and avoids unprofitable cache writes.

### 9a. Structured-output enforcement
- **Problem**: extraction/classification tasks answered in free prose carry filler ("Sure! Here's the data you asked for...") that a schema-constrained response cannot contain.
- **Mechanism**: when the task classifier flags extraction/classification shape with sufficient confidence, set the API's structured-output/tool-response parameter with an appropriate schema so the response is format-constrained by the API itself, not by hoping the model complies. Conservative: uncertain → leave unset.
- **Saves tokens by**: making filler structurally impossible for data-shaped tasks.

### 9b. Response-budget injector (Chain-of-Draft style)
- **Problem**: models over-elaborate by default; reasoning prose and pleasantries inflate output tokens (priced ~5x input) without adding accuracy — and research shows forced brevity can *improve* accuracy.
- **Mechanism**: based on classifier output, inject one of three pre-written instruction templates into the system prompt: OFF (default — anything explanatory/teaching-shaped), MODERATE (skip preamble/recap, answer directly), AGGRESSIVE (Chain-of-Draft style: minimal per-step reasoning, ~5-word steps, no filler; for tasks with verifiable short answers). Instruction overhead ~50-100 tokens, counted honestly against savings in the logger.
- **Saves tokens by**: instructing away verbosity on tasks where brevity is safe, at the cost of a small fixed instruction.

### 9c. Adaptive generation budgets (max_tokens / stop-sequences / thinking effort)
- **Problem**: static high caps allow runaway generation; static low caps truncate real answers. Both are wrong for half of all tasks. On reasoning-heavy calls this compounds: thinking tokens bill as output tokens and can dominate total cost if left unconstrained.
- **Mechanism — max_tokens/stop-sequences**: derive an expected output size from the task type (short factual → small; code/multi-step → large; extraction → sized to schema), set `max_tokens` = estimate + 15-20% safety margin. The "rolling window truncation rate" this widens against is **caller-supplied**, not stage-tracked — Tier 0 is stateless (per CLAUDE.md's non-negotiable constraints), so the stage cannot itself remember truncation history across calls. Callers who want auto-widening pass a per-task-type truncation-rate map into the stage config each call (the same pattern as stage 6's caller-supplied previous payload and stage 7's caller-supplied previous request); if a passed-in rate exceeds ~1-2%, that task type's margin widens this call and the adjustment is logged as `measured`. Absent that input, the stage always uses the base margin — conservative default, no guessing, no hidden state. Stop-sequences set only where a natural terminator is known safe (never inside code-heavy tasks).
- **Mechanism — thinking effort**: Tokenetics targets current-generation models, which require `thinking: {type: "adaptive"}` plus `output_config: {effort: <level>}` — the legacy `thinking.budget_tokens` API returns a 400 on Opus 4.7+ and must never be used on those models. Using the same shared classifier signal: trivial/factual/extraction-shaped tasks map to `low` or `medium` effort (adaptive thinking stays on, but the model spends far fewer thinking tokens at these levels); code/multi-step reasoning tasks map to `high`. Conservative default: if the classifier is uncertain or the task is unclassified, don't set `effort` at all — leave the caller's own thinking config untouched entirely, rather than guessing a value. Additionally, set `thinking: {display: "omitted"}` when the caller signals they don't need the reasoning trace surfaced. **This does not reduce the tokens billed for thinking at generation time** — thinking is billed identically regardless of `display` — but an omitted block's `thinking` text field is empty, so if the response is later echoed back into conversation history for a follow-up turn (the standard multi-turn pattern), that reasoning trace isn't re-billed as input tokens on the next call. It's a re-injection-cost optimization, not a generation-cost one, and is logged as such. Fails open: if the target model doesn't support adaptive thinking (pre-4.6 models, which still require `budget_tokens` or have no thinking support at all), skip the thinking sub-stage silently with an info-level log — never break the request over it. A versioned model-compatibility table (same "last verified" discipline as the cache pricing config) maps model generations to their accepted thinking API surface, so this mapping is never hardcoded silently into stage logic.
- **Saves tokens by**: capping the worst-case generation length per task instead of globally over-provisioning; via the effort knob, spending fewer thinking tokens at generation time on tasks that don't need deep reasoning; and via `display: "omitted"`, avoiding re-billing old reasoning traces as input tokens on future turns. Generation-time thinking savings and re-injection savings are logged as separate line items (both tagged `measured`), since they're distinct cost dimensions from each other and from visible-output savings.

### 11. Post-hoc output trim
- **Problem**: responses stored into history carry closing pleasantries/boilerplate that get re-paid as input tokens on every future turn.
- **Mechanism**: deterministic pattern-stripping of known boilerplate (trailing "let me know if...", opening restatements) from the response *after* it's received, before it's stored/re-injected. No summarization, no semantic rewriting — pattern removal only.
- **Saves tokens by**: shrinking what every future request re-includes; zero effect on the current call's cost.

### Tier 2a. LLMLingua-style compression (`tokenetics[compress]`)
- **Problem**: beyond structural redundancy, prose context contains individually low-information tokens a small model can identify.
- **Mechanism**: run a small local model (GPT-2-small class) over candidate context; score tokens by perplexity/importance; drop lowest-importance tokens up to the configured compression ratio. Lossy by design — ships its own quality-degradation benchmark (reduction-vs-accuracy curve at several ratios). Fail-open if model weights unavailable.
- **Saves tokens by**: aggressive content-level compression at a measured, documented quality cost — opt-in only.

### Tier 2b. Semantic caching (`tokenetics[semantic-cache]`)
- **Problem**: workloads with repetitive queries (FAQ/support-style) re-generate near-identical answers at full price.
- **Mechanism**: embed the incoming request with a local embedding model; nearest-neighbor lookup in a local vector store of previous (request, response) pairs; if similarity exceeds the conservative threshold, return the cached response and skip the API call entirely. TTL expiry + manual `invalidate()`/`clear()`. Hit/miss and match-quality logged for auditing false positives. Fail-open to a normal API call on any store/model failure.
- **Saves tokens by**: skipping generation entirely on a hit — the largest possible per-request saving, gated by the strictest correctness risk.

### Tier 2c. TALE-style budget estimation
- **Problem**: 9c's heuristic output-size estimate is coarse; the paper shows a model can estimate its own reasonable budget per task more precisely.
- **Mechanism**: opt-in replacement for 9c's estimate source — a small, cheap LLM call asks for a token-budget estimate for the specific task; that number (plus the same safety margin and truncation monitoring) feeds `max_tokens` and the brevity instruction. Its own round-trip cost (~100-200 tokens) is logged and reported against what it saved; benchmarked head-to-head vs. the free heuristic.
- **Saves tokens by**: tighter, task-specific budgets — only worthwhile when the tighter cap saves more than the estimation call costs, which the logger must prove per workload.

## Cross-cutting systems

- **Plugin contract**: `plugin(request, config, logger) -> request` over an internal request representation (not raw API JSON; converted at pipeline boundary). Abstract base class/protocol.
- **Cost logger**: orchestrator auto-captures baseline per stage (name, tokens before/after via real tokenizer, timing, on/off). Plugins optionally add rich metadata (confidence, reason, flags). Every entry tagged `measured` or `estimated`. Single tokenizer dependency is the sole source of token counts.
- **Fail-open standard (three categories)**: hard error → skip stage, pass request unmodified, error-level log; low-confidence decision → conservative branch, info-level log; partial batch failure → skip failing items, stage continues. Sole exception: cache-safety guard hard-raises.
- **Benchmark suite**: corpus organized by classifier task-type categories (code-heavy, conversational, extraction/structured, tool-call-heavy), 15-20+ samples each; separate held-out quality-check set for risk-bearing stages (brevity, cap tuning, pruning, lossy compression). Publish ranges per workload, never one flat percentage. All published numbers must be measured, not modeled.
- **Dashboard**: read-only consumer of the cost logger's structured log, fully decoupled from the request path (can crash without affecting requests). Scope: per-stage savings, measured/estimated split, truncation rate, cache hit rate, Tier 2 cost/benefit. v1 form: start with the simpler of CLI table (`tokenetics stats`) or small local web view.

## Development phases (already planned — follow this order)

0. Scaffolding: repo, pyproject.toml, pytest, CI, internal request type, plugin ABC, orchestrator skeleton, no-op plugin round-trip test.
1. Cross-cutting first: cost logger, fail-open framework, tokenizer. Deliberately-broken-plugin test must pass before any real stage.
2. Foundation stages (independent): dedup, near-dup, schema minification, post-hoc trim.
3. Shared classifier + small hand-labeled validation set (30-50 examples) — accuracy gate before anything consumes it.
4. Classifier-dependent stages: tool-relevance filtering, structured output, brevity injector, adaptive generation budgets — max_tokens tuning (highest test priority — stress test truncation monitoring on real responses) and thinking effort tuning (validated against the benchmark suite's held-out quality-check set, not truncation rate; requires the model-compatibility table for adaptive-thinking support).
5. Context scheduler: per-turn classifier → pinning pre-allocation → DP knapsack → greedy fallback → degraded-mode truncation. Latency benchmark for the DP threshold.
6. Caching: safety guard first (standalone, hard-raise tested), then reorder, then pricing config table, then breakpoint optimizer + synthetic-traffic benchmark (headline number).
7. Delta compression + `apply_delta()` + round-trip tests.
8. Assemble Tier 0 in fixed order; end-to-end integration test; regression test that stage 4 and stage 5 never double-prune; freeze Tier 0 API.
9. Benchmark suite (real measured numbers before extras).
10. Tier 2 extras (each independent).
11. Distribution: PyPI publish, MCP server + real-client test.
12. Dashboard.
13. Docs: README, architecture doc, benchmark write-up with methodology + honest caveats, "considered and rejected" section (LLM-based compression as default, semantic caching in core — and why not).

## Guidance for CLAUDE.md (project conventions Claude Code should follow)

- Python only. Type hints everywhere. pytest for all tests.
- Every stage defaults conservative when a choice exists (keep the tool, skip the brevity instruction, widen the cap).
- Every new stage requires its own test file before it is "done"; every fail-open path requires a deliberately-broken test case.
- The core stays stateless: no hidden persistence between calls. Caller supplies history. Any convenience state layer must be a separate opt-in wrapper on top of the stateless core.
- Never let Tier 2 dependencies (torch/transformers, embedding models, vector store) leak into the core install — extras only.
- The cache-safety guard is the only permitted hard-raise. Do not add others without explicit discussion.
- Token counts come from the one shared tokenizer utility only.
- Log honestly: `measured` vs `estimated` tags are load-bearing; never report an estimated number as measured.
- Do not reorder the pipeline. Do not make ordering configurable.
- Honest benchmarking: report ranges tied to stated workload assumptions; document where savings can go net-negative (already-lean short requests + injected instruction overhead).

## Tone for the README

Professional, honest, engineering-focused. State what the project does and its design constraints. Include the pipeline diagram (sequential), the three-tier structure, quickstart, and benchmark methodology. Avoid hype claims; every number must trace to the benchmark suite. No competitor comparisons.

## Amendments

- **2026-07-19** — Stage 9c extended from "adaptive max_tokens/stop-sequences" to "adaptive generation budgets," adding extended-thinking budget (`thinking.budget_tokens`) as a first-class optimization knob alongside max_tokens, driven by the same task-classifier signal. Rationale: thinking tokens bill as output tokens and can dominate cost on reasoning-heavy calls. Key differences from max_tokens tuning: conservative default is to leave the caller's thinking config untouched (not guess a value) when the classifier is uncertain; monitored via the benchmark suite's held-out quality-check set rather than truncation rate, since an undersized budget degrades quality silently; fails open (info-level log, skip sub-stage) if the target model doesn't support extended thinking. See the amended stage 9c section above for full mechanism detail.
- **2026-07-25** — Stage 9c's thinking mechanism replaced `thinking.budget_tokens` (deprecated/returns 400 on Opus 4.7+, Sonnet 5, Fable 5) with `thinking: {type: "adaptive"}` + `output_config.effort`, since Tokenetics targets current-generation models. Task types now map to effort levels (trivial/factual/extraction → low/medium, code/multi-step → high, uncertain → leave unset) instead of token budgets. Added `display: "omitted"` as a distinct optimization — it does **not** reduce generation-time thinking cost (billing is display-independent) but avoids re-billing old reasoning traces as input tokens when history is echoed back on a later turn, so it's logged as its own line item separate from the effort-level generation savings. Added a requirement for a versioned model-compatibility table (mirroring the cache-pricing table's "last verified" discipline) mapping model generations to their accepted thinking API surface, since pre-4.6 models still require `budget_tokens` or lack thinking support entirely. See the amended stage 9c section above for full mechanism detail.
- **2026-07-26** — Resolved a stateless-core conflict in stage 9c's `max_tokens` auto-widening, caught during Phase 4 planning: "auto-widen if truncation rate exceeds ~1-2% over a rolling window" implied the stage tracking history across calls, which violates CLAUDE.md's non-negotiable stateless-core constraint. Resolved by making the truncation-rate stat caller-supplied per call (same shape as stage 6's previous-payload and stage 7's previous-request inputs) rather than stage-tracked; absent input → base margin, conservative default, no hidden state. See the amended max_tokens mechanism line above.
- **2026-07-26** — Resolved two open questions in stage 5 (context scheduler) during Phase 5 planning, before implementation started: (1) "unresolved errors" (pinned type) are determined by an explicit positive resolution signal from the user in a later turn, not by the conversation merely moving on to another topic — the latter is weak evidence and fails in the dangerous direction (silently dropping a live problem), consistent with the project's conservative-by-default stance elsewhere. (2) The stage's token budget is caller-supplied per call (`config["token_budget"]`, same shape as stages 6/7/9c's caller-supplied inputs) when given; absent that, it falls back to a context-window-derived default (new versioned lookup table) scaled by the request's `task_type`, logged as `estimated` rather than `measured`. See the amended stage 5 section above for full mechanism detail.
- **2026-07-27** — Resolved four open questions in stage 7/8 (caching) during Phase 6 planning, before implementation started: (1) the cache-safety guard's hard-raise uses a dedicated `CacheSafetyError` type, not a bare exception, with a message identifying exactly what changed. (2) The breakpoint optimizer's caller-supplied usage history is a list of small `{timestamp, content_hash}` records (`config["cache_usage_history"]`), not full past request bodies — keeping the same cheap, caller-owned, no-persistence shape as every other caller-supplied input in the project. (3) Cache pricing numbers will be sourced from a live, dated lookup as the first implementation step, never seeded from a remembered/guessed figure, since a wrong cache price actively costs the caller money rather than just mis-optimizing. (4) The reorder stage is still implemented now (not skipped), scoped as a normalize-and-verify step given today's request shape already has little to physically reorder, with room to grow once delta compression or multi-block structures land. See the amended stage 7/8 sections above for full detail.
