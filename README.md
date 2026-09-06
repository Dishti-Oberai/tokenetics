# Tokenetics

**Less burn, same brain.**

**Up to 63% cheaper on real, repeated conversations — confirmed across 4 separate live runs against the Anthropic API (42-63% every time).**

[![CI](https://github.com/Dishti-Oberai/tokenetics/actions/workflows/ci.yml/badge.svg)](https://github.com/Dishti-Oberai/tokenetics/actions/workflows/ci.yml)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/Tier%200-frozen%20%26%20benchmarked-success)

<img src="docs/screenshots/dashboard-overview.png" alt="Tokenetics dashboard — real, measured token savings across request, output, and thinking tokens" width="820">

<sub>Real data from this repo's own benchmark runs. [Full dashboard →](#dashboard) · [Full results →](#real-world-results)</sub>

---

Tokenetics is an 11-stage optimization pipeline that sits in front of Anthropic API calls, deterministically stripping duplicate content, stale context, and bloated replies before a single token gets billed — and it optimizes both directions, shaping the reply that comes back too, not just what gets sent. Stateless by design: no hidden state between calls, history and cache metadata are caller-supplied every time. No prompt rewrites, no infrastructure, no model call of its own in the core path.

```python
from tokenetics import Tokenetics

tk = Tokenetics()                          # every optimization on by default
request = tk.prepare(messages=messages, tools=tools)
response = client.messages.create(**request)   # a normal Anthropic SDK call
stored = tk.finalize(response)             # cleans the reply before it's stored
```

## What it is — and isn't

A small library, imported wherever `client.messages.create(...)` gets called — the request passes through Tokenetics first. It is **not** a Claude Code plugin, not a proxy/server, and its "always on" core doesn't use another model call — that's an opt-in extra, not the default.

## Real-world results

Every number is **measured** against the live Anthropic API (Claude Sonnet 5, $2/$10 per MTok input/output) unless marked **estimated**. No flat single percentage — each result is tied to the workload it came from. Full methodology: [ROADMAP.md](ROADMAP.md).

| Workload | Result |
|---|---|
| A single, isolated request | **-2.1%** (roughly cost-neutral) — caching and pruning need repeat calls/accumulated redundancy to pay off, which a lone message has neither of |
| A real, repeated 8-turn conversation (4 separate live runs) | **+43.1%, +42.4%, +62.8%, +62.7% cheaper** — four for four, averaging ~53% |
| ↳ same 4 runs, broken down by dimension | input tokens **-67 to -70%** · output tokens **-62 to -74%** · thinking tokens real and measured throughout, never additive to the output figure |
| Caching alone *(estimated, synthetic traffic)* | ~28-31% cheaper than never caching |
| `semantic-cache` extra | 0% false-positive rate, 64.3% recall — deliberately conservative |
| `compress` extra | Quality holds 100% up to 40% compression; clamped there by default |

## Install

```bash
git clone https://github.com/Dishti-Oberai/tokenetics.git
cd tokenetics
uv sync          # install dependencies
uv run pytest    # run the full test suite
```

Not yet on PyPI — publish is the one remaining distribution step. Needs Python 3.10+; an [Anthropic API key](https://console.anthropic.com/) is only needed for real API calls, not for the test suite.

## The 11-stage pipeline

A fixed sequence, always in this order — each stage is individually toggleable, but the *order* never changes, since later stages depend on what earlier ones guarantee (the task classification from stage 3, for instance, is reused by stages 4, 5, 8, and 9 rather than re-derived).

```
 1. Remove exact duplicates
 2. Remove near-duplicates
 3. Classify the task type          ─┐ reused by stages 4, 5, 8, 9
 4. Trim unneeded tool definitions   │
 5. Keep only the messages worth it  │
 6. Send diffs, not whole payloads  ─┘
 7. Arrange for caching + safety check   ← the one hard-raise, see below
 8. Decide where the cache breakpoint goes
 9. Configure the response  (structured output · brevity · length & reasoning cap)
10. ── request sent to the API ──
11. Strip boilerplate from the reply before it's stored
```

<details>
<summary><b>What each stage actually does</b> (click to expand — SHA-256 fingerprinting, MinHash/Jaccard similarity, 0/1 knapsack scheduling, real cache-tier math, and more)</summary>

**1. Dedup.** The exact same block of text sometimes appears twice in one request — a retried frontend call, a re-sent system reminder — and gets billed twice. Each message is hashed with SHA-256; if the same fingerprint shows up again, the repeat is dropped and the first copy is kept.

**2. Near-dup.** Some content is *almost* identical but not byte-for-byte — a tool result that changed by one field, a restated point with different wording — which exact hashing misses entirely. MinHash/shingling estimates Jaccard similarity between chunks, using a stricter threshold for real conversation turns than for tool output, and merges anything past it. The most recent message is never touched, and every merge is logged.

**3. Task classifier.** Later stages each need to know what kind of request this is — code, a quick lookup, a data extraction, something tool-heavy — and shouldn't each re-derive that independently. A deterministic regex/keyword scan (no model call) labels the request once, with a confidence score; below a threshold it's left `None` rather than guessed, so downstream stages take their conservative branch instead of acting on a weak signal.

**4. Tool trimming.** Register ten tools and all ten full definitions get sent even when only two are relevant this turn. A tool is dropped only when it's clearly irrelevant to the latest message; anything ambiguous stays, on purpose — conservative by default.

**5. Context scheduler.** Sending the entire chat history every turn is expensive; blindly truncating old messages risks losing something that still matters. Each turn is scored by how likely it is to matter (an unresolved error or an explicit decision is protected), then a 0/1 knapsack DP picks the best-value combination that fits a token budget — falling back to a faster greedy approximation past a latency threshold on very long conversations, or to "keep the last N" if the stage is disabled outright.

**6. Delta compression.** Agent-style apps often resend an entire file or API response after only a small part of it changed. Given the previous version of a payload alongside the new one, only the computed diff is sent — and only when that diff is meaningfully smaller than the whole thing.

**7. Cache reorder + safety guard.** Anthropic's prompt cache only works if the request is byte-for-byte identical up to the cache point — one silent change anywhere before it, and the discount is gone with no error telling you so. Stable content (system prompt, tool definitions) gets moved before content that changes often, then the stage checks: did anything before the previous cache point actually change since the last call? If so, it hard-raises rather than let a silent cost regression through. This is the *only* stage in the whole pipeline that behaves this way — every other stage fails open.

**8. Cache breakpoint.** Most people cache "out of habit," always after the system prompt, without checking whether the write premium is actually worth it. This stage checks real repeat frequency and gap intervals between calls, then picks the 5-minute or 1-hour tier against a versioned, dated pricing table (or skips caching entirely if the traffic pattern doesn't justify the write).

**9a. Structured output.** A data-extraction ask answered in a friendly paragraph instead of clean JSON wastes tokens on filler. When the task is confidently data-shaped and a matching tool is registered, `tool_choice` is forced directly rather than hoping the model stays terse on its own.

**9b. Brevity.** Models over-explain by default, and output is billed roughly 5x higher than input for Sonnet-class models. A short instruction nudges toward a terse, direct answer — but only for task shapes where brevity is confidently safe; ambiguous shapes stay untouched.

**9c. Length & reasoning cap.** A fixed `max_tokens` is wrong for someone — too low and answers truncate, too high and headroom goes to waste — and on reasoning-heavy questions, Claude's own internal "thinking" tokens can end up costing more than the visible answer if left unconstrained. A sensible cap is estimated per task type with a safety margin, widening automatically if real truncation happens too often. Separately, the model's own adaptive-thinking effort level is tuned down for short, bounded questions and up for genuinely hard ones — validated against real quality checks rather than just a truncation-rate count, since an under-tuned reasoning setting fails silently instead of visibly cutting off a response.

**11. Reply cleanup.** Sign-offs and pleasantries in a reply get re-sent — and re-billed — every time that reply is included in future context. Known boilerplate phrases are stripped before the reply is stored, which pays off on every future call that reuses it, not the current one.

Full internals, the plugin contract, and the versioned pricing tables: [ARCHITECTURE.md](ARCHITECTURE.md). Optional Tier 2 extras (`compress`, `semantic-cache`, smarter length estimation via a cheap side-call) are documented there too.

</details>

## Dashboard

A local, read-only web view over the cost logger's own log file — no new dependency beyond Chart.js from a CDN for the charts. Shows per-stage token savings, cache hit rate, and real output/thinking-token measurements (read directly from `usage.output_tokens_details.thinking_tokens`, not estimated), or an honest "no data logged yet" placeholder where they aren't. Fully decoupled from the request path.

```bash
uv run python scripts/dashboard.py --log-file costs.jsonl
```

<details>
<summary>Full dashboard view</summary>

<img src="docs/screenshots/dashboard-full.png" alt="Full Tokenetics dashboard, all sections" width="820">

</details>

## Advanced config

Most stages need no configuration. A few (cache guard, breakpoint optimizer, delta compression) read caller-supplied context via `stage_config`:

```python
tk = Tokenetics(
    stage_config={
        "cache_reorder_guard": {"previous_request": previous_request},
        "cache_breakpoint_optimizer": {"cache_usage_history": cache_usage_history},
    },
)
```

## Development

```bash
uv run pytest         # tests
uv run ruff check .   # style
```

- [ARCHITECTURE.md](ARCHITECTURE.md) — internals, plugin contract, versioned pricing tables
- [ROADMAP.md](ROADMAP.md) — full dated build history, every benchmark run and bug found
- [TESTING.md](TESTING.md) — testing conventions and coverage map
