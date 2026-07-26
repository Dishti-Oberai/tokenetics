# Testing

Tokenetics has two distinct testing layers, and they're deliberately kept separate — see [CLAUDE.md](CLAUDE.md)'s "Incremental runnability" section for why:

1. **Automated tests** (pytest + ruff + mypy) — free, fast, run on every push/PR via CI. Never hits the network.
2. **Manual dev-loop testing** (`scripts/dev_demo.py`) — hits the real Anthropic API, costs real money, run by hand when you want to eyeball actual behavior.

## Prerequisites

```bash
uv sync
```

## 1. Automated tests

```bash
uv run pytest          # run the test suite
uv run pytest -v       # verbose — see each test by name
uv run ruff check .    # lint
uv run mypy src        # type check (strict mode)
```

All three are required to pass CI (`.github/workflows/ci.yml`). Run all three locally before pushing:

```bash
uv run ruff check . && uv run mypy src && uv run pytest
```

### Current coverage (Phases 0-4)

| File | What it checks |
|---|---|
| `tests/test_import.py` | Package imports and `__version__` is set |
| `tests/test_request.py` | `from_api_kwargs()` / `to_api_kwargs()` round-trip: known fields survive unchanged, unmodeled kwargs (e.g. `temperature`, `thinking`) pass through via `extra` instead of being dropped, unset optional fields are omitted rather than sent empty |
| `tests/test_orchestrator.py` | The plugin contract round-trips through `Tokenetics.prepare()` unchanged with a no-op stage; a disabled stage calls `degraded_fallback()` instead of `run()`; a custom logger can be injected; the default pipeline runs the real foundation stages in the correct fixed order; default stage instances are never shared/mutated across separate `Tokenetics()` calls |
| `tests/test_tokenizer.py` | `count_tokens()` returns a real count and forwards only token-relevant fields (never `max_tokens`/`temperature`); `count_text_tokens()` correctly wraps plain text for response-side stages |
| `tests/test_logger.py` | `InMemoryCostLogger` records entries in order with all fields intact; `NullLogger` genuinely does nothing |
| `tests/test_fail_open.py` | **The deliberately-broken-plugin test**, for both the request-side and response-side paths: a stage that raises is caught, the request/text passes through unmodified, an error-level log line is actually emitted, the cost logger records the failure, and the pipeline continues to the next stage afterward |
| `tests/test_dedup.py` | Exact duplicates removed, first occurrence kept, same text under a different role is *not* treated as a duplicate, drops are logged |
| `tests/test_near_dup.py` | Shingling and MinHash-similarity helpers directly; identical messages get merged (older dropped); the most-recent message is never the one dropped; clearly dissimilar messages are left alone; tool-output threshold is confirmed looser than the conversation-turn threshold |
| `tests/test_schema_minification.py` | Inert JSON-Schema keys stripped, description whitespace collapsed, functionally important fields (`enum`, `additionalProperties`, `required`) preserved exactly |
| `tests/test_post_hoc_trim.py` | Trailing boilerplate stripped, stacked sign-offs fully removed via the iterate-until-stable loop, no false positives on legitimate content that merely resembles a sign-off phrase mid-answer |
| `tests/test_finalize.py` | `Tokenetics.finalize()` extracts text from a response and runs it through the default response-side pipeline; an empty `response_stages` list is a genuine no-op; a cost-logger entry gets recorded |
| `tests/test_task_classifier.py` | Per-category detection (code, extraction, tool-heavy via structural `tool_use`/`tool_result` scan, conversational fallback); an empty message is left `None` ("unclassified") rather than guessed; `confidence` is always set; the stage never mutates `messages`; `task_type`/`confidence` are noted for the cost logger |
| `tests/test_task_classifier_accuracy.py` | **The Phase 3 accuracy gate**: the classifier must clear 85% accuracy against `tests/fixtures/classifier_validation_set.json` (32 hand-labeled examples across all 4 categories + 2 unclassified edge cases) before Phase 4 is allowed to consume its output |
| `tests/test_lexical.py` | The shared `overlap_score`/`latest_user_text` helpers directly: identical text scores 1.0, disjoint text scores 0.0, case-insensitive, empty-string sides score 0.0, `latest_user_text` correctly finds the last user turn and ignores non-text content blocks |
| `tests/test_model_compatibility.py` | Current-generation models resolve to `"adaptive"`; an unknown model and an unconfirmed one (Haiku 4.5) both conservatively resolve to `"unsupported"` rather than a guessed value |
| `tests/test_schema_minification.py` (extended) | Tool-relevance filtering: a tool with zero word overlap against the latest user message is dropped, any real overlap keeps it, no user text to compare against means every tool is kept, drops are noted with their score |
| `tests/test_structured_output.py` | `tool_choice` is forced only when task_type is `extraction` *and* a registered tool clears the lexical-match bar; no-ops when there are no tools, no match, or the caller already set their own `tool_choice`; match is noted with its score |
| `tests/test_brevity_injector.py` | `extraction` gets the MODERATE instruction appended to (or set as) the system prompt; `code`/`conversational`/unclassified stay OFF; the selected bucket is noted, nothing is noted when OFF |
| `tests/test_adaptive_budget.py` | `max_tokens` widens for a known task type but never narrows below the caller's own value; unclassified requests are left untouched; a caller-supplied `truncation_stats` entry above threshold widens further; thinking effort varies correctly by task type, is skipped when unclassified, respects a caller-set `thinking` config, and fails open (no exception, no thinking config set) on a model the compatibility table doesn't mark adaptive-capable |

See [ROADMAP.md](ROADMAP.md) for what's coming next.

## 2. Manual dev-loop testing (`scripts/dev_demo.py`)

This script is **not** part of pytest or CI — it makes a real, billed call to the Anthropic API. Use it to sanity-check real behavior by hand, not as a correctness gate.

**Requirements:**
- `ANTHROPIC_API_KEY` set in your own shell environment (never paste a key into a chat or commit it to a file — see the security note below)
- A funded Anthropic account (check [console.anthropic.com/settings/billing](https://console.anthropic.com/settings/billing) if you get a "credit balance too low" error)

**Run it:**

```bash
export ANTHROPIC_API_KEY=sk-ant-...
uv run python scripts/dev_demo.py --list-scenarios   # see what's available, free, no calls
uv run python scripts/dev_demo.py --all-scenarios    # exercise every stage + corner case, free
uv run python scripts/dev_demo.py --scenario extraction   # one real completion, billed once
```

**As of Phase 4, the script is scenario-based**, not a single fixed sample — one conversation can't exercise seven stages meaningfully (`structured_output`/`brevity_injector` only ever fire on `extraction`-shaped requests, `near_dup` needs genuinely similar-not-identical text, tool-relevance filtering needs an actually-irrelevant tool registered, etc.). `SCENARIOS` in the script is a small library, each one built to make a specific stage or corner case visibly do something:

| Scenario | Exercises |
|---|---|
| `dedup` (default) | Exact-repeat turns → `dedup` drops them |
| `near_dup` | A Q&A pair paraphrased (not identical) elsewhere in the conversation → `near_dup` merges both turns via MinHash similarity, `dedup` correctly leaves it alone |
| `extraction` | An extraction-shaped question + a matching tool → `structured_output` forces `tool_choice`, `brevity_injector` injects the MODERATE instruction, `adaptive_budget` picks low thinking effort |
| `code` | A code question with a deliberately small `max_tokens` cap → `task_classifier` says `code`, `adaptive_budget` widens the cap and picks high thinking effort |
| `tool_heavy` | A conversation already containing a real `tool_use`/`tool_result` exchange → `task_classifier` detects `tool-heavy` structurally, not from text |
| `irrelevant_tool` | One relevant tool + one clearly unrelated tool registered → `schema_minification`'s relevance filter drops only the irrelevant one |
| `caller_sets_thinking` (corner case) | A code-shaped request where the caller already set their own `thinking` config → `adaptive_budget` respects it and doesn't override, even though `code` would normally get `effort: "high"` |

There's deliberately no `unclassified` scenario here: `task_classifier` only ever returns `task_type=None` for empty/whitespace-only text, and `Tokenetics.prepare()` calls the real `count_tokens` endpoint before any stage runs — which rejects that content outright ("text content blocks must contain non-whitespace text"), free or not. No string can satisfy the API's validation and still trigger the unclassified path, so this corner case can't be demonstrated by running anything through the real API at all. It's covered directly instead, bypassing `count_tokens`/the API entirely, by `tests/test_task_classifier.py::test_empty_message_is_unclassified_not_guessed`.

Most of these stages only touch the *request* — `tk.prepare()` only calls the free `count_tokens` endpoint, never the billed `messages.create()`. `--all-scenarios` uses exactly this to check all of it for $0; only `--scenario NAME` without `--prepare-only` spends money, and only once per run.

For any single scenario, the per-stage log shows each stage's own before/after tokens, timing, and any noted metadata (`dropped_duplicates`, `merges`, `task_type`/`confidence`, `dropped_tools`, `tool_choice_forced`, `brevity_bucket`, `max_tokens_widened_to`, `thinking_effort`, `thinking_effort_skipped`, ...). A stage showing no token change and no notes is a correct no-op for that particular scenario, not a bug — check the table above for which scenario actually exercises which stage.

**On/off comparison:** `--disable STAGE_NAME` (e.g. `--disable dedup`) now does something real — compare `request tokens before -> after` with and without a given stage to see its individual contribution, per [CLAUDE.md](CLAUDE.md)'s "Incremental runnability" section.

**Cost-conscious flags** (see the script's own docstring for more): as of Phase 4, `adaptive_budget` sets a bounded thinking effort automatically for classified requests on supported models, so the script no longer needs to force-disable thinking itself. Pass `--show-thinking` to force `display="summarized"` and inspect the reasoning text yourself (this pre-empts `adaptive_budget`'s own choice, since it always respects a caller-set `thinking` config), or `--model claude-haiku-4-5` for the cheapest possible smoke test.

### Security note

Never paste a real API key into a chat, issue tracker, or commit — treat any key that's been pasted somewhere as compromised and rotate it in the console immediately. Keep keys in your own shell environment or a local `.env` file (already excluded via `.gitignore`).

## Testing conventions for new work

These apply once pipeline stages start landing (Phase 2 onward) — full detail in [CLAUDE.md](CLAUDE.md):

- **Every new stage needs its own test file** before it's considered done.
- **Every fail-open path needs a deliberately-broken test case** proving the request goes out unmodified (or degraded-but-safe), not raising or corrupting the request.
- **Stage 9c (adaptive generation budgets)** is highest test priority: stress-test the `max_tokens` truncation-monitoring against real responses, and validate the thinking-effort knob against the benchmark suite's held-out quality-check set (it has no truncation signal to monitor).
- **Regression test required:** stages 4 (tool-relevance filtering) and 5 (context scheduler) must never double-prune the same content.
- **The cache-safety guard (stage 7)** needs a standalone hard-raise test before reorder or the breakpoint optimizer are built on top of it — it's the one deliberate non-fail-open behavior in the whole project.
- **Type hints everywhere** — `mypy --strict` must stay clean as new modules are added.

## Testing by phase

Concrete testing approach for each build-order phase — both the automated (pytest, free) and manual (`dev_demo.py`/real API, costs money) side. Phases are cumulative: later phases' automated suites still run everything from earlier phases.

| Phase | What's being verified | Automated approach | Manual (`dev_demo.py`) approach |
|---|---|---|---|
| **0** ✅ done | Request type, plugin contract, orchestrator skeleton | Round-trip equality on `from/to_api_kwargs`; no-op stage preserves request; disabled stage calls `degraded_fallback()` not `run()` | Pass-through baseline: identical before/after size, empty stage list, real reply returned |
| **1** | Cost logger, fail-open framework, shared tokenizer | Logger records correct tokens-before/after, timing, on/off per stage; every logged number tagged `measured`/`estimated` correctly; **the deliberately-broken-plugin test** — a stage that raises must be caught, skipped, request passed through unmodified, logged at error level (this must pass before any real stage is built) | Extend the script to print the logger's per-stage summary after a real call |
| **2** | Dedup, near-dup (MinHash), schema minification, post-hoc trim — each independent | Dedup: exact duplicates removed, first occurrence kept, near-duplicates untouched. Near-dup: dual thresholds (~85-90% tool output, ~95%+ conversation turns) hit correctly, most-recent-turn exemption holds, every merge logs its similarity score. Schema minification: stripped schema still validates real tool calls. Post-hoc trim: known boilerplate stripped, no false positives on legitimate content | Send a prompt with deliberately repeated/near-duplicate content, verify size drops and the repeat is gone from `to_api_kwargs()` output; check a boilerplate-heavy reply gets trimmed before it would be re-stored |
| **3** | Task classifier + accuracy gate | Accuracy against the 30–50 hand-labeled examples must clear the gate threshold before Phase 4 starts; per-category precision/recall (code, conversational, extraction, tool-heavy) | Run varied real prompts through it, print `(task_type, confidence)` and eyeball whether it matches intuition |
| **4** ✅ done (structural) | Tool-relevance filtering, structured-output enforcement, brevity injector, adaptive generation budgets (max_tokens + thinking effort level) | **Done**: tool filtering keeps anything ambiguous, drops only zero-overlap tools, logs drops with score. Structured output only fires on `extraction` + a matching registered tool, respects a caller-set `tool_choice`. Brevity: default OFF except `extraction` → MODERATE, overhead counted honestly via the normal token measurement. max_tokens: widen-only, never narrows below the caller's value, conservative on unclassified. Thinking effort: correct per-task-type routing, conservative default (unclassified/caller-set `thinking` → untouched), fail-open tested against a model the compatibility table marks unsupported. **Deferred to Phase 9** (needs the benchmark suite, doesn't exist yet): the **max_tokens real-response stress test** (truncation rate over many live completions, confirm auto-widen triggers above ~1-2%) and **thinking-effort validation against the held-out quality-check set** (with vs. without effort tuning, per task type) — Phase 4 shipped the structural/routing tests only, not this rigor bar, and that's flagged in ROADMAP.md as owed, not skipped. `display: "omitted"`'s multi-turn-continuation behavior also not yet tested | Run an extraction-shaped prompt with a matching tool registered, confirm `tool_choice`/system-prompt/`max_tokens`/`thinking` all show the expected values in the per-stage log and `prepare()` output; run a multi-turn exchange with the default (now-omitted-by-default) thinking display to confirm the second call's input tokens don't include the first call's reasoning trace |
| **5** | Context scheduler (DP knapsack) | Per-turn activity classifier accuracy; pinned categories (explicit decisions, unresolved errors) never removed regardless of score; DP solver verified against brute-force on small synthetic cases; greedy fallback triggers above the size threshold and logs it; degraded mode (last-N truncation) engages and logs when the stage is disabled; **latency benchmark** to justify the DP→greedy threshold | Feed a long synthetic conversation through, print which turns were kept vs. dropped and sanity-check the important ones survived |
| **6** | Caching: safety guard → reorder → pricing config → breakpoint optimizer | **Safety guard: standalone hard-raise test** — any byte change at/before the last cache breakpoint vs. the previous request must raise; byte-identical prefix must pass silently. Reorder: stable content (system, tools) placed before volatile content, never reordered retroactively. Pricing table: staleness of the "last verified" date is flagged, not silently trusted. Breakpoint optimizer: placement and TTL-tier choice verified against synthetic repeat patterns. **Synthetic-traffic benchmark** — the headline number, measuring real cache-hit-rate improvement | Send the same request pattern twice with `previous_request` wired in; check `cache_control` placement in `to_api_kwargs()`, then check `usage.cache_read_input_tokens` on the real response |
| **7** | Structured delta compression + `apply_delta()` | Diff/patch computed correctly for a modified payload; size-threshold fallback sends full payload when the diff isn't meaningfully smaller; **round-trip test** — `apply_delta(previous, delta)` reconstructs the original exactly | Re-send a slightly modified tool-output payload, confirm a delta (not the full payload) went out and the size drop is reported |
| **8** | Tier 0 assembled in fixed order, frozen API | End-to-end integration test over a realistic multi-turn conversation, asserting the pipeline ran in the fixed order (via the logger's stage trace); **regression test** that stages 4 and 5 never double-prune the same content; a contract/signature test on `Tokenetics.prepare()`/`finalize()` so the frozen API can't silently drift | Full run against a real multi-turn, tool-using conversation; compare total token reduction against the Phase 0 baseline |
| **9** | Benchmark suite — real measured numbers | Build the corpus (15–20+ samples × 4 task-type categories) plus a held-out quality-check set for risk-bearing stages; the benchmark runner itself is unit-tested for correct `measured`/`estimated` tagging and correct range aggregation | This phase *is* mostly manual/real-API work — running the full corpus through the real pipeline repeatedly to produce the numbers the write-up will cite |
| **10** | Tier 2 extras: `compress`, `semantic-cache`, TALE budget estimation | `compress`: quality-degradation benchmark (reduction ratio vs. accuracy at several levels), fail-open test with model weights unavailable. `semantic-cache`: false-positive rate on held-out near-miss pairs, TTL expiry, `invalidate()`/`clear()`, fail-open to a normal call on store/model failure. TALE: head-to-head benchmark vs. the free heuristic, round-trip cost logged honestly. All three tested only under their own extras install — never pulled into the core test run | Exercise each extra against real repeated/similar traffic with the relevant extra installed |
| **11** | Distribution: PyPI + MCP server | Package builds cleanly (`uv build`) and installs in a fresh venv; **MCP server real-client test** — an actual MCP client connects and round-trips | Install from a built wheel (not the repo checkout) in a clean environment, run `dev_demo.py` against that install to catch packaging bugs |
| **12** | Dashboard | Decoupling test — crashing the dashboard process must not affect a live `Tokenetics.prepare()` call; correct parsing/aggregation of the cost logger's structured log | Visual check of the CLI table / local web view against known logger output |
| **13** | Docs | Every published benchmark number traces back to an actual Phase 9 run; cross-links between README/CLAUDE.md/ROADMAP/architecture doc resolve; the "considered and rejected" section reflects real amendment history | Fresh read-through of the quickstart by someone (or a cold Claude session) who hasn't seen the code, to catch doc rot |

See [ROADMAP.md](ROADMAP.md) for the phase-by-phase build order and dates these tests map to.
