# Tokenetics

**Tokenetics makes calls to Claude (Anthropic's AI) cheaper, automatically — without changing the quality of the answers you get back.**

If you're building an app on top of Claude, you already know the basic deal: every word you send costs money (this is called a "token"), and every word Claude sends back costs money too. The longer your conversations get, the more you pay — often for the same repeated instructions, tool definitions, and chat history, over and over again.

Tokenetics sits between your code and the Anthropic API. You hand it your request, it quietly trims the waste — duplicate content, stale context, over-verbose replies — and passes a leaner version through. You don't have to rewrite your prompts or change how you talk to Claude.

> **New to some of these terms?** A **token** is roughly a chunk of a word — it's the unit Anthropic bills by. The **context window** is the running history of a conversation that gets sent with every request. **Prompt caching** is a feature where Anthropic charges less for content it's already seen recently, if you mark it correctly — Tokenetics automates that marking.

> ✅ **Status: Tier 0 core is built, frozen, and benchmarked against the real Anthropic API.** Tier 2 extras (`compress`, `semantic-cache`, smarter length estimation) and the local dashboard are also done. Distribution (PyPI publish + MCP server) is in progress. Everything described below is real, running code — see [Results](#real-world-impact) for the honest, measured numbers, or [ROADMAP.md](ROADMAP.md) for the full build history.

## What it does, in plain terms

- **Trims what you send** — removes repeated content, drops context that isn't relevant to the current task, and arranges your request so Anthropic's caching discount kicks in more often.
- **Trims what comes back** — nudges Claude toward shorter, more direct answers when brevity won't hurt the answer, and prevents it from generating far more than the task needs.

## What this is (and isn't)

**It is:** a small library you import into your own Python code. Wherever you currently call `client.messages.create(...)`, you pass your request through Tokenetics first.

**It is not:**
- A plugin for Claude Code or other coding assistants — it doesn't touch your terminal sessions.
- A proxy or server you have to run — there's no infrastructure to stand up.
- Something that uses AI to shrink your prompts by default — the core, "always on" behavior uses plain, predictable logic (regex, hashing, that kind of thing), not another model call. AI-based compression exists, but it's an optional extra you turn on yourself.

## Why it's different from just "prompt engineering harder"

1. **It optimizes both directions.** Most cost-saving advice focuses on what you send. Tokenetics also shapes what Claude generates back.
2. **It understands Anthropic's actual cache pricing.** Prompt caching has real rules — a 5-minute discount tier and a 1-hour tier, each with different trade-offs. Tokenetics calculates which one is worth it for your traffic instead of you guessing.
3. **It won't tell you it saved money unless it actually measured it.** Every number this project ever publishes is checked against a real benchmark, and clearly labeled as either *measured* or *estimated* — never one dressed up as the other.
4. **It's designed to never break your request.** If any optimization step is unsure or fails, it backs off and sends your request through unchanged rather than risk a bad or broken call. There is exactly one exception, explained below, and it exists to stop a subtle bug (silently paying full price when you thought you were caching) rather than to break anything.

## Setup

Tokenetics isn't published to PyPI yet (that's the one remaining piece of distribution, in progress). For now, install from source:

**You'll need:**
- Python 3.10 or newer
- [uv](https://docs.astral.sh/uv/) (a fast Python package manager) to install dependencies
- An [Anthropic API key](https://console.anthropic.com/) — only if you want to make real Claude calls; not needed just to run the tests

**Get it running:**

```bash
git clone https://github.com/Dishti-Oberai/tokenetics.git
cd tokenetics
uv sync          # installs dependencies
uv run pytest    # runs the full test suite (300+ tests)
```

Once published, this will become a one-line `pip install tokenetics`.

## How it's organized: three tiers

Think of this as three layers you can opt into, each doing more (and asking more of your machine):

- **Tier 0 — the core.** On by default. Everything here is deterministic: no AI models, no extra network calls, just fast, predictable logic. This is what you get from the base install.
- **Tier 2 — optional extras**, each its own separate install:
  - `tokenetics[compress]` — uses a small local AI model to more aggressively shrink prompt text. This one *can* affect quality, so it ships its own before/after quality report and quietly steps aside if it can't load.
  - `tokenetics[semantic-cache]` — for apps that get asked similar questions repeatedly (think FAQ-style traffic). It recognizes a "we've basically answered this before" question and returns the cached answer instead of calling Claude again.
  - **Smarter output-length estimation** — instead of a free, simple guess at how long a response should be, this pays for a tiny extra Claude call to estimate it more precisely, and reports honestly on whether that trade was worth it.

## What happens to your request, step by step

Your request passes through a fixed sequence of steps, always in this order. Each step can be switched off individually, but the *order* never changes — later steps depend on assumptions earlier ones guarantee.

```
 1. Remove exact duplicates
 2. Remove near-duplicates
 3. Figure out what kind of task this is        ─┐ this result is reused
 4. Trim unneeded tool definitions               │ by steps 4, 5, 8, 9
 5. Pick which past messages are worth keeping   │
 6. Send only what changed, not the whole thing ─┘
 7. Arrange content for caching + safety check    ← the one exception, see below
 8. Decide where the cache "savepoint" goes
 9. Configure how Claude should respond
    9a. Force structured output where it fits
    9b. Ask for a shorter answer where it's safe
    9c. Cap the response length — and how much it reasons — sensibly
10. ── Your request is sent to Claude ──
11. Clean up boilerplate from the reply before it's stored
```

### Each step, explained simply

| Step | The problem it solves | How it works |
|---|---|---|
| 1. Remove exact duplicates | The exact same text sometimes appears twice in one request, and you pay for it twice. | Fingerprint each chunk of text; if the same fingerprint shows up again, drop the repeat and keep the first copy. |
| 2. Remove near-duplicates | Some content is *basically* the same but not byte-for-byte identical (a tool result that barely changed, a restated point) — exact matching misses it. | A fast similarity check (technically: MinHash/shingling) flags content that's "close enough" and merges it, using a stricter bar for real conversation turns than for tool output. It never touches your most recent message, and every merge is logged so you can see what happened. |
| 3. Figure out the task type | Later steps need to know what kind of request this is (a coding question? a quick lookup? a data extraction?) — that shouldn't be re-guessed five different ways. | A cheap, rule-based check (no AI call) looks at your latest message for signals like code blocks, error messages, or "extract/list" phrasing, and labels the task once for everyone downstream to reuse. |
| 4. Trim unneeded tools | If you've registered ten tools, all ten full definitions get sent even when only two are relevant this turn. | Strips unnecessary bulk from tool definitions, and only removes a tool from the request when it's clearly irrelevant — if it's unclear, it stays, on purpose. |
| 5. Pick which past messages to keep | Sending your entire chat history every time is expensive; blindly cutting old messages risks losing something important. | Scores each part of the conversation by how likely it is to matter (an unresolved error or an explicit decision is protected from being cut), then picks the best combination that fits your budget — a classic optimization technique (0/1 knapsack). On very long conversations it switches to a faster approximation to stay quick, and if you turn this step off entirely, it falls back to simply keeping the last few messages. |
| 6. Send only what changed | Agent-style apps often resend a whole file or API response after only a small part of it changed. | If you give it the previous version of something alongside the new version, it computes just the difference and sends that instead — but only when the difference is meaningfully smaller than sending the whole thing. |
| 7. Arrange for caching + safety check | Anthropic's cache only works if the beginning of your request is byte-for-byte identical to a previous call. One accidental change anywhere before the cache point, and you silently start paying full price with no error. | Puts stable content (system prompt, tool definitions) before content that changes often, so more of your request stays cacheable. Then it double-checks: if anything before your last cache point *did* change compared to your previous request, it deliberately stops and raises an error rather than let a silent cost regression slip through. This is the one and only step in the whole project that behaves this way — everywhere else, it fails safely instead. |
| 8. Decide where to place the cache savepoint | People usually cache "out of habit" (like, always after the system prompt) without actually checking if it's worth the extra cost of writing to cache. | Looks at which content actually repeats across your calls and how often, then works out whether it's worth caching, and whether the 5-minute or 1-hour cache tier makes more sense for how frequently you're calling. |
| 9a. Force structured output where it fits | If you're asking for a clean data extraction and Claude answers in a friendly paragraph instead, you're paying for filler words you don't need. | When it's confident your task is data-shaped (extraction, classification), it tells the API to constrain the response format directly, instead of hoping the model keeps it short on its own. |
| 9b. Ask for a shorter answer where it's safe | Models tend to over-explain by default, and extra reasoning/pleasantries cost real money on the output side (which is billed higher than input). | Adds a short instruction nudging Claude to skip preamble or reasoning filler — but only for tasks where being terse won't hurt correctness. Off by default; only turns on when it's fairly confident brevity is safe. |
| 9c. Cap the response length — and how much it reasons — sensibly | A fixed length limit is always wrong for someone — too low and answers get cut off, too high and you're paying for wasted headroom. On reasoning-heavy questions, Claude's internal "thinking" tokens can end up costing more than the visible answer if left unconstrained. | Estimates a sensible length limit based on the type of task, adds a safety margin, and watches how often real answers get cut off — automatically widening the margin if that happens too often. Does something similar for reasoning: nudges Claude to spend less effort thinking through simple/factual questions and more on code/math/multi-step tasks, using the model's own reasoning-depth setting — and leaves your own settings alone if it isn't confident which bucket applies. It also avoids re-sending old reasoning traces as input on future turns once they're no longer needed, since paying to re-read your own past thinking is pure waste. Since an under-tuned reasoning setting hurts answer quality silently instead of visibly cutting off a response, this part is checked against real quality comparisons rather than a cutoff-rate count. |
| 11. Clean up the reply before storing it | The pleasantries and sign-offs in Claude's replies get re-sent (and re-billed) every time that reply is included in future context. | Strips known boilerplate phrases from the response after it comes back, before you store it for later. This doesn't shrink the current call — it saves on every future call that reuses this reply. |

### The optional extras, explained simply

| Extra | The problem it solves | How it works |
|---|---|---|
| `tokenetics[compress]` | Even after removing duplicates, ordinary prose still has plenty of low-value words. | Runs a small local AI model over your text, scores each word by how much it actually matters, and drops the least useful ones up to a ratio you choose. This can affect answer quality, so it comes with its own measured report of how much quality is traded for how much savings — and quietly does nothing if the model can't load. |
| `tokenetics[semantic-cache]` | Apps that answer similar questions repeatedly (support bots, FAQs) waste money regenerating near-identical answers. | Compares your new question to previously answered ones using an embedding model (a way of comparing meaning, not just exact words), and if it's confident enough that it's basically the same question, returns the old answer instead of calling Claude again. Cached answers expire automatically, and you can also clear them by hand. |
| Smarter length estimation | The free, rule-based length guess (step 9c) is good but coarse. | Instead of guessing for free, this makes one small, cheap Claude call to ask "how long should this answer realistically be?" — and is honest in its reporting about whether that extra call actually paid for itself. |

## Behind the scenes

A few systems run underneath all of this to keep it honest and extensible:

- **Plugins** — every step above is written against the same simple interface, so if you want to write your own custom step, you can plug it in the same way the built-in ones work.
- **The cost logger** — automatically records, for every step, how many tokens went in and came out, using one single, consistent way of counting tokens across the whole project.
- **Fail-safe by default** — if a step hits an error, it gets skipped and your request goes through unchanged. If a step isn't confident about a judgment call, it takes the cautious option. The cache safety check (step 7) is the sole deliberate exception — everything else prioritizes "don't break the request" over "save every possible token."
- **The benchmark suite** — a set of real test conversations across different task types (coding, chit-chat, data extraction, tool-heavy), used to produce every number this project ever claims.
- **A local dashboard** — a small, self-contained web view (stdlib `http.server`, plus Chart.js from a CDN for the charts) reading the cost logger's own log file. Shows per-stage token savings, cache hit rate, and how the optional extras are paying for themselves — or an honest "no data logged yet" placeholder where they aren't. Fully decoupled from the request path: it can crash without affecting a single API call, since it never runs in that process. Run it with `uv run python scripts/dashboard.py --log-file costs.jsonl`.

  <img src="docs/screenshots/dashboard-overview.png" alt="Tokenetics dashboard: savings overview donuts and request-tokens summary" width="800">

  <details>
  <summary>Full dashboard (click to expand) — pipeline stage breakdown, output/thinking token sections, and Tier 2 extras</summary>

  <img src="docs/screenshots/dashboard-full.png" alt="Full Tokenetics dashboard, all sections" width="800">

  </details>

## Usage

```python
from tokenetics import Tokenetics

tk = Tokenetics()  # the core, with every optimization on by default

request = tk.prepare(
    messages=messages,  # your full conversation so far — Tokenetics doesn't store anything itself
    tools=tools,
)

response = client.messages.create(**request)  # this part is just the normal Anthropic SDK call

stored = tk.finalize(response)  # cleans up the reply before you save it for next time
```

**Enabling the cache-safety guard and other advanced features:** most stages need no configuration at all, but a few (the cache-safety guard, the breakpoint optimizer, delta compression) read caller-supplied context through `stage_config`, passed when you construct `Tokenetics()` — not as a `prepare()` keyword argument:

```python
tk = Tokenetics(
    stage_config={
        "cache_reorder_guard": {"previous_request": previous_request},  # the raw kwargs you passed to prepare() last time
        "cache_breakpoint_optimizer": {"cache_usage_history": cache_usage_history},
    },
)
request = tk.prepare(messages=messages, tools=tools)
```

## Real-world impact

Every number below is **measured** — a real, billed call against the live Anthropic API, not a simulation or a model of expected behavior — except the caching figures explicitly marked **estimated** (a free, offline model of Anthropic's real cache pricing, run without spending money). Per this project's own honesty rule: no single flat percentage, every result tied to the workload it came from, and the bad news is included, not filtered out. Full methodology and every run's raw numbers live in [ROADMAP.md](ROADMAP.md).

**A single, isolated request is roughly cost-neutral.** Across an 83-sample corpus spanning coding, conversational, extraction, tool-heavy, and mixed workloads — one baseline call vs. one Tokenetics-optimized call each, real dollar cost including input, output, and cache pricing — the net result was **-2.1%** (Tokenetics was very slightly more expensive on this batch, well within normal call-to-call variance). This makes sense once you know why: caching only pays off on *repeated* calls, and pruning stages need real accumulated redundancy to find — a single independent message rarely has either.

**A real, repeated conversation is where it counts.** Four separate real runs of the same realistic 8-turn conversation (a coding question, a genuine decision, some padding, a tangent, an extraction ask, and a final recap) — one thread run completely without Tokenetics, one run completely with it, both against the live API — came back **+16.0%, +4.4%, +12.0%, and +13.7% cheaper**, four for four. The pattern in every run: a small one-time cache write, then several turns reading off that cache at a steep discount, occasionally interrupted by a genuine cache-expiry re-write (ordinary variance in how long a cache entry actually stays warm) that's still net-positive across the whole session.

**Caching alone (estimated, synthetic traffic, no live calls):** ~28-31% cheaper than never caching, ~10-11% cheaper than the common habit of "always cache with a 5-minute TTL and hope," at roughly an 80% hit rate — from modeling Anthropic's real published cache pricing (write premiums, read discount, 5-minute vs. 1-hour tiers) against a repeat-traffic pattern, not from a live benchmark run.

**The Tier 2 extras, measured on their own terms:**
- `tokenetics[semantic-cache]` — 0% false-positive rate (0 of 16 genuinely different questions wrongly served a cached answer) at its default similarity threshold, with 64.3% recall (9 of 14 genuinely repeat questions correctly caught) — a deliberately conservative trade: safe to be wrong in the direction of an extra API call, never in the direction of a wrong cached answer.
- `tokenetics[compress]` — quality holds up completely (100% pass on a held-out quality-check set) up to a 40% compression ratio; by 45% quality starts degrading, and by 55-60% only a quarter of samples still pass. This is why the library clamps to 40% by default and requires an explicit opt-in flag to go higher — a real, tested limit, not a guess.

**Where this can go net-negative, documented rather than hidden:** the brevity injector's ~50-100 token instruction overhead can exceed its savings on requests that were already short. And a version of `adaptive_budget` that auto-selected higher reasoning effort by task type was found, via this same real benchmarking, to make coding and conversational requests 2-3x more expensive with no explicit signal the caller wanted that trade — it's since been changed to require an explicit opt-in, and the numbers above already reflect that fix.

See [ARCHITECTURE.md](ARCHITECTURE.md) for how the pipeline works internally, or [ROADMAP.md](ROADMAP.md) for the full, dated history of every benchmark run, bug found, and fix made along the way.

## Development

Once you've done the [Setup](#setup) steps above:

```bash
uv run pytest         # run the tests
uv run ruff check .   # check code style
```
