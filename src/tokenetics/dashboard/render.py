"""Renders `DashboardStats` as a single HTML page. Kept separate from
`server.py` so the HTML itself is unit-testable without spinning up a real
HTTP server.

Charts (added 2026-09-06, at the user's explicit request after being told
the tradeoff): Chart.js, loaded from a CDN in <head>. This is a deliberate
departure from this project's usual "no new dependency" scoping -- every
other choice in Tokenetics (Tier 2 extras behind pip extras, the dashboard
itself originally being stdlib-only, MCP as its own extra) avoided adding
an external dependency; this is the first thing in the project that needs
network access to fully render, purely for cosmetic charts layered on top
of numbers the page already shows in the table/cards without them. If the
machine loading this page has no internet access, the chart canvases stay
empty but every other part of the page (all real data) still works --
this is stated on the page itself, not left to fail silently.
"""

from __future__ import annotations

import html
import json

from tokenetics.dashboard.aggregate import DashboardStats, GenerationUsageStats, StageStats, Tier2Stats

_CHART_PALETTE = [
    "#4f8cff", "#22c55e", "#f59e0b", "#a855f7", "#14b8a6",
    "#eab308", "#ec4899", "#3b82f6", "#84cc16", "#f97316",
]

# Below this many thinking_reinjection samples, a caveat is shown -- see the
# comment where it's used for why (independent-completion sampling noise).
_NOISE_FLOOR_SAMPLE_THRESHOLD = 10

# Shared color for the Request/Output/Thinking token donuts (2026-09-06, at
# the user's explicit request -- previously Request/Output used a sign-based
# green/red while Thinking always used this purple; now all three match).
_TOKEN_DONUT_COLOR = "#a855f7"


def _json_for_script(value: object) -> str:
    # Defensive against a `</script>` sequence breaking out of the script
    # tag if it ever appeared inside embedded data -- stage names are
    # internal constants today, not user input, but this is the standard,
    # cheap precaution for embedding JSON inside HTML <script> tags.
    return json.dumps(value).replace("</", "<\\/")

# Stages whose real effect is never expressible as a request-token-count
# change -- always structurally 0/0.0% here, not a measurement that
# happened to come out zero. Added 2026-09-06 after a user, looking at the
# dashboard alone (not dev_demo.py's own per-stage terminal explanations,
# nor the source), reasonably read a bare "0.0%" as "did nothing" for a
# stage that was actually working correctly (e.g. cache_breakpoint_
# optimizer placing a real cache_control marker -- its benefit is a
# cheaper price per token on a hit, not fewer tokens). The row itself is
# kept (Enabled/Errors/Avg time are still real, useful signals for these
# stages -- removing the row would lose that), only the Saved/% Saved
# cells are replaced with a short explanation instead of a misleading 0%.
_NO_TOKEN_IMPACT_NOTES = {
    "task_classifier": "annotates the request only",
    "cache_reorder_guard": "verifies/normalizes only",
    "structured_output": "forces tool_choice, not token count",
    "adaptive_budget": "affects output/generation cost, not request size",
    "cache_breakpoint_optimizer": "cheaper price per cached token, not fewer tokens",
}

# Stages whose Saved/% Saved IS a real, meaningful token-count measurement
# -- unlike the structural stages above, these keep their real bar/percentage
# -- but whose number can still be surprising without context: brevity_
# injector's cost is real instruction overhead, only ever added (never
# saved), and only on extraction-shaped requests; post_hoc_trim's saving is
# entirely dependent on whether a given batch of real replies happened to
# contain a recognized boilerplate sign-off at all. Shown as a small note
# alongside the real numbers, not instead of them.
_CONTEXTUAL_NOTES = {
    "brevity_injector": (
        "only fires on extraction-shaped requests; adds ~50-100 tokens of "
        "instruction overhead when it does, never removes any -- a real, "
        "documented net-negative case, not an error"
    ),
    "post_hoc_trim": (
        "only trims boilerplate actually present in a reply -- 0 across "
        "these runs means none of them happened to end in a recognized "
        "sign-off yet, not that trimming is broken"
    ),
}

_LOGO_MARK = (
    '<svg width="30" height="30" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">'
    '<rect width="32" height="32" rx="8" fill="url(#tkLogoGrad)"/>'
    '<path d="M9 20L15 12L19 17L23 11" stroke="white" stroke-width="2.25" '
    'stroke-linecap="round" stroke-linejoin="round"/>'
    '<circle cx="23" cy="11" r="2" fill="white"/>'
    '<defs><linearGradient id="tkLogoGrad" x1="0" y1="0" x2="32" y2="32">'
    '<stop stop-color="#6366f1"/><stop offset="1" stop-color="#22c55e"/></linearGradient></defs>'
    "</svg>"
)

_PAGE_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Tokenetics Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
<style>
  * {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    margin: 0; padding: 2.5rem 3rem 3.5rem;
    background: radial-gradient(1200px 600px at 10% -10%, #14182a 0%, #090a0e 55%), #090a0e;
    color: #e8eaf0;
    -webkit-font-smoothing: antialiased;
  }}
  .header {{ display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.4rem; }}
  h1 {{ font-size: 1.55rem; font-weight: 700; letter-spacing: -0.01em; margin: 0; }}
  .subtitle {{ color: #8b90a0; font-size: 0.83rem; margin-bottom: 0.35rem; line-height: 1.5; }}
  .subtitle code {{ background: #161a24; padding: 0.1rem 0.35rem; border-radius: 4px; }}

  .hero {{
    display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 1.5rem;
    margin: 1.75rem 0 1.5rem; padding: 1.75rem 2rem; border-radius: 16px;
    background: linear-gradient(135deg, #161c2e 0%, #12151f 100%);
    border: 1px solid #262c3d; box-shadow: 0 8px 30px rgba(0,0,0,0.35);
  }}
  .hero .label {{ font-size: 0.78rem; color: #9aa0b0; text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600; }}
  .hero .value {{ font-size: 3rem; font-weight: 800; letter-spacing: -0.02em; margin-top: 0.2rem; line-height: 1; }}
  .hero .value.positive {{ color: #34d399; }}
  .hero .value.negative {{ color: #f87171; }}
  .hero .sub {{ color: #8b90a0; font-size: 0.85rem; margin-top: 0.4rem; }}
  .hero .badge {{
    font-size: 0.78rem; font-weight: 600; padding: 0.35rem 0.85rem; border-radius: 999px;
    background: rgba(52,211,153,0.12); color: #34d399; border: 1px solid rgba(52,211,153,0.3);
  }}
  .hero .badge.negative {{ background: rgba(248,113,113,0.12); color: #f87171; border-color: rgba(248,113,113,0.3); }}

  /* Smaller sibling of .hero -- same visual language (gradient card, color-
     coded value, badge) for a headline number that lives inside a
     sub-section rather than at the top of the page. Added 2026-09-06 so
     the real, measured "output tokens saved" figure gets the same
     prominence as the input-tokens hero above, instead of sitting as a
     plain card indistinguishable from Tier 2 mechanisms with far less
     riding on them. */
  .sub-hero {{
    display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 1.25rem;
    margin-bottom: 1rem; padding: 1.25rem 1.5rem; border-radius: 14px;
    background: linear-gradient(135deg, #161c2e 0%, #12151f 100%);
    border: 1px solid #262c3d; box-shadow: 0 4px 18px rgba(0,0,0,0.3);
  }}
  .sub-hero .label {{ font-size: 0.72rem; color: #9aa0b0; text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600; }}
  .sub-hero .value {{ font-size: 2rem; font-weight: 800; letter-spacing: -0.02em; margin-top: 0.2rem; line-height: 1.15; }}
  .sub-hero .value.positive {{ color: #34d399; }}
  .sub-hero .value.negative {{ color: #f87171; }}
  .sub-hero .sub {{ color: #8b90a0; font-size: 0.8rem; margin-top: 0.4rem; }}
  .sub-hero .badge {{
    font-size: 0.72rem; font-weight: 600; padding: 0.3rem 0.75rem; border-radius: 999px;
    background: rgba(52,211,153,0.12); color: #34d399; border: 1px solid rgba(52,211,153,0.3);
    white-space: nowrap;
  }}
  .sub-hero .badge.negative {{ background: rgba(248,113,113,0.12); color: #f87171; border-color: rgba(248,113,113,0.3); }}

  .cards {{ display: flex; gap: 1rem; margin-bottom: 1.75rem; flex-wrap: wrap; }}
  .card {{
    background: #12151f; border: 1px solid #232838; border-radius: 12px;
    padding: 1.1rem 1.35rem; min-width: 165px; flex: 1 1 165px;
    box-shadow: 0 1px 0 rgba(255,255,255,0.02) inset;
  }}
  .card .label {{ font-size: 0.72rem; color: #8b90a0; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600; }}
  .card .value {{ font-size: 1.55rem; font-weight: 700; margin-top: 0.3rem; letter-spacing: -0.01em; }}

  table {{ border-collapse: collapse; width: 100%; border-radius: 12px; overflow: hidden; }}
  th, td {{ text-align: left; padding: 0.65rem 0.9rem; border-bottom: 1px solid #1d212e; font-size: 0.87rem; }}
  th {{
    color: #8b90a0; font-weight: 600; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em;
    background: #12151f; border-bottom: 1px solid #262c3d;
  }}
  tbody tr {{ background: #0e1017; }}
  tbody tr:nth-child(even) {{ background: #12141d; }}
  tbody tr:hover {{ background: #171b28; }}
  td {{ color: #d3d6de; }}
  td:first-child {{ font-weight: 600; color: #e8eaf0; }}

  .bar-track {{ background: #1d212e; border-radius: 3px; height: 6px; width: 100px; overflow: hidden; }}
  .bar-fill {{ background: #6366f1; height: 100%; }}
  .bar-fill.negative {{ background: #f87171; }}
  .empty {{ color: #8b90a0; padding: 3rem 0; text-align: center; }}
  .empty code {{ background: #161a24; padding: 0.15rem 0.4rem; border-radius: 4px; }}
  .section-title {{
    font-size: 0.95rem; font-weight: 700; margin: 2.5rem 0 0.9rem; color: #e8eaf0;
    padding-left: 0.65rem; border-left: 3px solid #6366f1;
  }}
  .no-data {{ color: #6b7280; font-size: 0.85rem; padding: 1rem 1.1rem; background: #12151f; border: 1px solid #232838; border-radius: 10px; }}
  .stage-note {{ color: #7b8094; font-size: 0.78rem; font-style: italic; }}
  .scope-note {{ margin-top: 3rem; font-size: 0.78rem; color: #6b7280; border-top: 1px solid #1d212e; padding-top: 1.25rem; line-height: 1.6; }}
  .scope-note code {{ background: #161a24; padding: 0.1rem 0.35rem; border-radius: 4px; }}
  .charts-wrap {{ display: flex; gap: 1.25rem; flex-wrap: wrap; margin-bottom: 1.5rem; }}
  .chart-card {{
    background: #12151f; border: 1px solid #232838; border-radius: 12px;
    padding: 1.1rem; width: 340px; max-width: 100%;
  }}
  .chart-card canvas {{ max-height: 260px; }}

  /* Savings overview donut (2026-09-06): bigger and centered, meant to
     be the first visual thing read after the hero number -- distinct from
     the denser, analyst-oriented charts further down the page. Also
     shared by the smaller inline donut cards placed next to the output/
     thinking Tier 2 cards further down (.donut-inline). */
  .overview-wrap {{ justify-content: center; margin: 0.25rem 0 2rem; }}
  .donut-inline {{ justify-content: flex-start; margin-bottom: 1.5rem; }}
  .chart-card.overview {{
    width: 300px; padding: 1.35rem 1.35rem 1.1rem;
    border: 1px solid #2c3348; box-shadow: 0 6px 22px rgba(0,0,0,0.3);
  }}
  .chart-card.overview canvas {{ max-height: 230px; }}
  .section-title.overview-title {{ margin-top: 0.5rem; }}
</style>
</head>
<body>
<div class="header">{logo}<h1>Tokenetics Dashboard</h1></div>
<div class="subtitle">Read-only view over <code>{log_path}</code> &mdash; refresh this page to re-read the log.</div>
{body}
<div class="scope-note">
  Pipeline stage breakdown: per-stage token savings, timing, and enabled/error state from
  <code>log_stage()</code>. Output tokens, Thinking tokens, and Other optimizations: real usage
  data from <code>log_event()</code>, logged only where the caller (e.g. <code>dev_demo.py
  --measure-usage</code>/<code>--tale</code>/<code>--compress-ratio</code>/
  <code>--semantic-cache</code>/<code>--measure-thinking-reinjection</code>) actually computed it --
  a mechanism with no logged events shows "no data logged" rather than a fabricated number.<br>
  Output tokens saved is a real, measured baseline-vs-optimized comparison (two real completions,
  <code>--all-scenarios --measure-usage --log-to</code>). Thinking tokens consumed is also real and
  measured, read directly from <code>usage.output_tokens_details.thinking_tokens</code> -- the real cost
  of <code>adaptive_budget</code>'s thinking-effort knob firing at GENERATION time, shown only once at
  least one logged sample actually engaged thinking. Where no such sample exists yet, the page falls
  back to an older, explicitly-labeled ESTIMATE (output tokens minus visible-reply tokens, each side) --
  treat that fallback number as directional, not exact. Thinking re-injection tokens saved is a
  separate, later-turn cost dimension -- what a FOLLOW-UP turn's real input tokens cost once a thinking
  block from an earlier turn is part of history, comparing <code>display="omitted"</code> against
  <code>display="summarized"</code> -- and is never combined with the generation-time numbers above.<br>
  Charts load Chart.js from a CDN &mdash; the only part of this page needing internet access;
  every number on it, including the charts' own data, is a fully local, offline read of the log file.
  The "Request vs. output vs. thinking tokens" chart spans two different sample populations: Request
  covers every logged run, Output/Thinking cover only the (usually smaller) subset that went through a
  real <code>--measure-usage</code> A/B -- read it as an at-a-glance overview, not a precise ratio.
</div>
</body>
</html>
"""

_EMPTY_BODY = """<div class="empty">
  No runs logged yet at this path.<br><br>
  Generate some with:<br>
  <code>uv run python scripts/dev_demo.py --scenario code --log-to {log_path}</code>
</div>"""


def _card(label: str, value: str) -> str:
    return f'<div class="card"><div class="label">{label}</div><div class="value">{value}</div></div>'


def _render_cache_subsection(tier2: Tier2Stats) -> str:
    # Placed in the Request tokens section (not Output), since cache reads/
    # writes are a REQUEST-side cost -- cache_control affects price-per-
    # token on the input side, not anything about generation. Moved here
    # 2026-09-06 per the user's own "if it's required" hedge when asking
    # for it under Output; request-side is the more honest home for it.
    # No longer has its own donut (removed 2026-09-06, replaced by a
    # request-tokens baseline-vs-optimized donut instead -- see
    # _render_request_tokens_donut -- for consistency with how the Output/
    # Thinking sections each get exactly one donut for their own metric).
    cache = tier2.cache_usage
    if not cache.sample_count:
        return '<div class="no-data">No cache usage data logged (needs --measure-usage).</div>'
    return (
        '<div class="cards">'
        + _card("Cache lookups logged", str(cache.sample_count))
        + _card("Cache hit rate", f"{cache.hit_rate:.1f}%")
        + _card("Cache read tokens", str(cache.total_cache_read_tokens))
        + _card("Cache write tokens", str(cache.total_cache_creation_tokens))
        + "</div>"
    )


def _render_request_tokens_donut(stats: DashboardStats) -> str:
    # Mirrors the Output/Thinking sections, each of which has exactly one
    # baseline-vs-optimized donut for their own metric -- the Request
    # tokens section previously had the cache donut instead, replaced
    # 2026-09-06 per the user's explicit ask for consistency.
    if not (stats.total_tokens_before or stats.total_tokens_after):
        return ""
    return _render_donut_card(
        "tkRequestDonut", "Request tokens", stats.total_tokens_before, stats.total_tokens_after, _TOKEN_DONUT_COLOR
    )


def _render_output_tokens_section(tier2: Tier2Stats | None) -> str:
    parts = ['<div class="section-title">Output tokens</div>']
    generation = tier2.generation_usage if tier2 is not None else None
    if generation is None or not generation.sample_count:
        parts.append(
            '<div class="no-data">No output-token data logged '
            "(needs --all-scenarios --measure-usage --log-to).</div>"
        )
        return "".join(parts)

    gen_sign = "positive" if generation.output_pct_saved >= 0 else "negative"
    truncation_note = ""
    if generation.baseline_truncated_count:
        truncation_note = (
            f" &mdash; {generation.baseline_truncated_count}/{generation.sample_count} baseline "
            'replies were truncated (<code>stop_reason="max_tokens"</code>): a negative number '
            "here may mean &quot;baseline is an incomplete fragment,&quot; not real waste"
        )
    parts.append(
        f"""<div class="sub-hero">
  <div>
    <div class="label">Output tokens saved (real, measured)</div>
    <div class="value {gen_sign}">{generation.output_tokens_saved} ({generation.output_pct_saved:.1f}%)</div>
    <div class="sub">{generation.sample_count} baseline-vs-optimized A/B samples, real completions{truncation_note}</div>
  </div>
  <div class="badge {gen_sign}">{"OUTPUT SAVINGS" if generation.output_pct_saved >= 0 else "OUTPUT COST"}</div>
</div>"""
    )
    if generation.min_output_pct_saved is not None:
        # Per-CLAUDE.md's "report savings as ranges" rule -- same rationale
        # as the request-tokens min/median/max cards above.
        parts.append(f"""<div class="cards">
  <div class="card"><div class="label">Min sample saved %</div><div class="value">{generation.min_output_pct_saved:.1f}%</div></div>
  <div class="card"><div class="label">Median sample saved %</div><div class="value">{generation.median_output_pct_saved:.1f}%</div></div>
  <div class="card"><div class="label">Max sample saved %</div><div class="value">{generation.max_output_pct_saved:.1f}%</div></div>
</div>""")
    parts.append(
        _render_donut_card(
            "tkOutputDonut",
            "Output tokens",
            generation.total_baseline_output_tokens,
            generation.total_optimized_output_tokens,
            _TOKEN_DONUT_COLOR,
        )
    )
    return "".join(parts)


def _render_thinking_tokens_section(tier2: Tier2Stats | None) -> str:
    parts = ['<div class="section-title">Thinking tokens</div>']
    generation = tier2.generation_usage if tier2 is not None else None

    if generation is not None and generation.thinking_sample_count:
        parts.append(
            '<div class="cards">'
            + _card(
                "Thinking tokens consumed (real, measured)",
                f"{generation.thinking_tokens_consumed}"
                '<br><span class="stage-note">NOT an extra loss on top of Output tokens saved above -- '
                "thinking tokens are a SUBSET of output_tokens (Anthropic's own field documents "
                "thinking_tokens as \"Always &le; output_tokens\"), so this cost is already netted "
                "into that real savings figure, not additive to it. "
                f"{generation.thinking_sample_count} samples with thinking engaged, from "
                "<code>usage.output_tokens_details.thinking_tokens</code> -- NOT necessarily from "
                "adaptive_budget's effort knob: confirmed 2026-09-06 that the model can report "
                "nonzero thinking tokens on a scenario's UNTOUCHED baseline call too, with no thinking "
                "config sent at all, so some of this total may be the model's own reasoning "
                "independent of anything this project configured</span>",
            )
            + "</div>"
        )
        if generation.min_thinking_tokens_delta is not None:
            # Raw token deltas, not %, per the field's own docstring --
            # baseline_thinking_tokens is 0 on most samples, so a %-saved
            # figure would be undefined far too often to summarize.
            parts.append(f"""<div class="cards">
  <div class="card"><div class="label">Min sample delta (tokens)</div><div class="value">{generation.min_thinking_tokens_delta}</div></div>
  <div class="card"><div class="label">Median sample delta (tokens)</div><div class="value">{generation.median_thinking_tokens_delta:.1f}</div></div>
  <div class="card"><div class="label">Max sample delta (tokens)</div><div class="value">{generation.max_thinking_tokens_delta}</div></div>
</div>""")
        parts.append(
            _render_donut_card(
                "tkThinkingDonut",
                "Thinking tokens",
                generation.total_baseline_thinking_tokens,
                generation.total_optimized_thinking_tokens,
                _TOKEN_DONUT_COLOR,
            )
        )
    elif generation is not None and generation.sample_count:
        parts.append(
            '<div class="cards">'
            + _card(
                "Est. thinking tokens saved",
                f"{generation.estimated_thinking_tokens_saved}"
                '<br><span class="stage-note">estimated, not measured -- see scope note below. '
                "No samples yet with real thinking data logged.</span>",
            )
            + "</div>"
        )
    else:
        parts.append(
            '<div class="no-data">No thinking-token data logged '
            "(needs --all-scenarios --measure-usage --log-to).</div>"
        )

    reinjection = tier2.thinking_reinjection if tier2 is not None else None
    if reinjection is not None and reinjection.sample_count:
        re_sign = "positive" if reinjection.tokens_saved >= 0 else "negative"
        # Confirmed 2026-09-06 via a real 4-sample run (deltas of 1, 1, 0, 21
        # -- no consistent direction, small total): each sample's two
        # threads make independent completions (extended thinking can't use
        # temperature=0), so natural reply-length variance is comparable to
        # or larger than the real re-injection effect at low n. Caveat shown
        # below this threshold so the number isn't read as more settled
        # than it is; _NOISE_FLOOR_SAMPLE_THRESHOLD is a judgment call, not
        # a statistically derived cutoff.
        noise_note = (
            f'<br><span class="stage-note">only {reinjection.sample_count} sample(s) -- each '
            "sample's two threads are independently-sampled real completions (extended thinking "
            "can't force temperature=0), so natural reply-length variance can be comparable to "
            "the real effect at this n; treat this number as speculative until run at a much "
            "larger sample size</span>"
            if reinjection.sample_count < _NOISE_FLOOR_SAMPLE_THRESHOLD
            else ""
        )
        parts.append(
            f"""<div class="sub-hero">
  <div>
    <div class="label">Thinking re-injection tokens saved (real, measured)</div>
    <div class="value {re_sign}">{reinjection.tokens_saved} ({reinjection.pct_saved:.1f}%)</div>
    <div class="sub">{reinjection.sample_count} sample(s) -- turn-2 input tokens, <code>display="summarized"</code>
    (re-billed) vs <code>display="omitted"</code> (adaptive_budget's real default){noise_note}</div>
  </div>
  <div class="badge {re_sign}">{"RE-INJECTION SAVINGS" if reinjection.tokens_saved >= 0 else "RE-INJECTION COST"}</div>
</div>"""
        )
    else:
        parts.append(
            '<div class="no-data">No thinking re-injection data logged '
            "(needs --measure-thinking-reinjection).</div>"
        )

    return "".join(parts)


def _render_other_optimizations_section(tier2: Tier2Stats) -> str:
    parts = ['<div class="section-title">Other optimizations</div>']

    tale = tier2.tale
    if tale.sample_count:
        parts.append(
            '<div class="cards">'
            + _card("TALE estimates logged", str(tale.sample_count))
            + _card("Avg budget estimate", f"{tale.avg_budget_tokens:.0f} tok")
            + _card("Avg estimation cost", f"{tale.avg_estimation_cost_tokens:.0f} tok")
            + "</div>"
        )
    else:
        parts.append('<div class="no-data">No TALE data logged (needs --tale).</div>')

    compress = tier2.compress
    if compress.sample_count:
        parts.append(
            '<div class="cards">'
            + _card("Compress calls logged", str(compress.sample_count))
            + _card("Avg ratio achieved", f"{compress.avg_ratio_achieved:.2f}")
            + _card("Clamp rate", f"{compress.clamp_rate:.1f}%")
            + "</div>"
        )
    else:
        parts.append('<div class="no-data">No compress data logged (needs --compress-ratio).</div>')

    semantic = tier2.semantic_cache
    if semantic.sample_count:
        parts.append(
            '<div class="cards">'
            + _card("Semantic-cache lookups logged", str(semantic.sample_count))
            + _card("Hit rate", f"{semantic.hit_rate:.1f}%")
            + "</div>"
        )
    else:
        parts.append('<div class="no-data">No semantic-cache data logged (needs --semantic-cache).</div>')

    return "".join(parts)


def _pie_chart_data(stages: list[StageStats]) -> tuple[list[str], list[float], list[str]]:
    # % SAVED per stage (not absolute tokens saved, changed 2026-09-06 at
    # the user's request) -- replaces the old separate "% saved by stage"
    # bar chart entirely, rather than keeping both an absolute-tokens pie
    # and a %-based bar. Pie slices only make sense for non-negative shares
    # -- a stage that COST tokens (brevity_injector) doesn't have a "share
    # saved" to plot; the table above already shows its real (negative)
    # number for anyone who wants it.
    positive = sorted((s for s in stages if s.total_saved > 0), key=lambda s: s.pct_saved, reverse=True)
    labels = [s.stage_name for s in positive]
    values = [round(s.pct_saved, 2) for s in positive]
    colors = [_CHART_PALETTE[i % len(_CHART_PALETTE)] for i in range(len(positive))]
    return labels, values, colors


def _render_stage_savings_pie(stats: DashboardStats) -> str:
    pie_labels, pie_values, pie_colors = _pie_chart_data(list(stats.stages.values()))
    if not pie_labels:
        return ""
    call = (
        "  new Chart(document.getElementById('tkSavingsPie'), {\n"
        f"    type: 'pie',\n"
        f"    data: {{ labels: {_json_for_script(pie_labels)}, datasets: [{{ "
        f"data: {_json_for_script(pie_values)}, backgroundColor: {_json_for_script(pie_colors)} }}] }},\n"
        "    options: { plugins: { title: { display: true, text: '% tokens saved by stage "
        "(positive contributors)' } } }\n"
        "  });"
    )
    return (
        '<div class="charts-wrap donut-inline">'
        '<div class="chart-card overview"><canvas id="tkSavingsPie"></canvas></div>'
        "</div>"
        f"<script>if (window.Chart) {{\n{call}\n}}</script>"
    )


def _combined_tokens_chart_data(
    stats: DashboardStats, generation: GenerationUsageStats | None
) -> tuple[list[str], list[int], list[int]]:
    # Deliberately spans two different sample populations -- "Request" is
    # every logged run (from the always-free stage table), "Output"/
    # "Thinking" are only the subset that went through a real --measure-
    # usage A/B. Not apples-to-apples in sample size, so this is an at-a-
    # glance overview, not a precise ratio -- the chart's own title says
    # so, and each category is simply omitted (not shown as a fabricated
    # zero) when its underlying data doesn't exist yet.
    categories: list[str] = []
    before: list[int] = []
    after: list[int] = []
    if stats.total_tokens_before or stats.total_tokens_after:
        categories.append("Request")
        before.append(stats.total_tokens_before)
        after.append(stats.total_tokens_after)
    if generation is not None and generation.sample_count:
        categories.append("Output")
        before.append(generation.total_baseline_output_tokens)
        after.append(generation.total_optimized_output_tokens)
    if generation is not None and generation.thinking_sample_count:
        categories.append("Thinking")
        before.append(generation.total_baseline_thinking_tokens)
        after.append(generation.total_optimized_thinking_tokens)
    return categories, before, after


def _overview_donut_title(name: str, before: int, after: int) -> str:
    # Honest phrasing for all three real shapes this can take: a genuine
    # before/after with data on both sides, a pure-cost metric starting
    # from a zero baseline (thinking tokens usually have no baseline spend
    # to compare against), and the empty case. Never says "X% saved" when
    # there was no baseline to save against -- that would silently divide
    # by zero into a misleading "0.0%".
    saved = before - after
    if before == 0 and after == 0:
        pct_text = "no data"
    elif before == 0:
        pct_text = "real cost, no baseline usage"
    else:
        pct = saved / before * 100
        pct_text = f"{pct:.1f}% saved" if saved >= 0 else f"{abs(pct):.1f}% more spent"
    return f"{name}: {pct_text} ({before} → {after})"


def _overview_donut_js(canvas_id: str, name: str, before: int, after: int, color: str) -> str:
    title = _overview_donut_title(name, before, after)
    return (
        f"  new Chart(document.getElementById('{canvas_id}'), {{\n"
        "    type: 'doughnut',\n"
        "    data: { labels: ['Baseline', 'Optimized'], datasets: [{ "
        f"data: [{before}, {after}], backgroundColor: ['#3a4152', '{color}'] }}] }},\n"
        "    options: { plugins: { title: { display: true, "
        f"text: {_json_for_script(title)}, font: {{ size: 13 }} }}, legend: {{ position: 'bottom' }} }} }}\n"
        "  });"
    )


_COMBINED_DONUT_COLORS = {
    # (baseline shade, optimized shade) per category -- distinct hue per
    # category so all 6 slices stay visually groupable even without
    # hovering for the tooltip label.
    "Request": ("#1f2b47", "#4f8cff"),
    "Output": ("#14532d", "#34d399"),
    "Thinking": ("#4c1d75", "#a855f7"),
}


def _combined_donut_js(canvas_id: str, categories: list[str], before: list[int], after: list[int]) -> str:
    labels: list[str] = []
    values: list[int] = []
    colors: list[str] = []
    for cat, b, a in zip(categories, before, after):
        base_color, opt_color = _COMBINED_DONUT_COLORS.get(cat, ("#3a4152", "#4f8cff"))
        labels.append(f"{cat} (baseline)")
        values.append(b)
        colors.append(base_color)
        labels.append(f"{cat} (optimized)")
        values.append(a)
        colors.append(opt_color)
    return (
        f"  new Chart(document.getElementById('{canvas_id}'), {{\n"
        "    type: 'doughnut',\n"
        f"    data: {{ labels: {_json_for_script(labels)}, datasets: [{{ "
        f"data: {_json_for_script(values)}, backgroundColor: {_json_for_script(colors)} }}] }},\n"
        "    options: { plugins: { title: { display: true, text: 'Request vs. output vs. thinking "
        "tokens (baseline vs. optimized, different sample populations -- see scope note)', "
        "font: { size: 13 } }, legend: { position: 'bottom' } } }\n"
        "  });"
    )


def _render_savings_overview(stats: DashboardStats, tier2: Tier2Stats | None) -> str:
    # Meant to be the FIRST thing on the page after the header: one glance
    # at all three token dimensions plus how they compare, before any
    # metric cards -- per the user's explicit ask (2026-09-06). Each
    # token type ALSO gets its own donut placed with its own section
    # further down the page (see _render_output_tokens_section /
    # _render_thinking_tokens_section / the request-tokens section) --
    # this top row is the "one glance" summary, not the only place these
    # numbers appear.
    generation = tier2.generation_usage if tier2 is not None else None
    cards = []
    calls = []

    if stats.total_tokens_before or stats.total_tokens_after:
        cards.append('<div class="chart-card overview"><canvas id="tkOverviewRequest"></canvas></div>')
        calls.append(
            _overview_donut_js(
                "tkOverviewRequest",
                "Request tokens",
                stats.total_tokens_before,
                stats.total_tokens_after,
                _TOKEN_DONUT_COLOR,
            )
        )

    if generation is not None and generation.sample_count:
        cards.append('<div class="chart-card overview"><canvas id="tkOverviewOutput"></canvas></div>')
        calls.append(
            _overview_donut_js(
                "tkOverviewOutput",
                "Output tokens",
                generation.total_baseline_output_tokens,
                generation.total_optimized_output_tokens,
                _TOKEN_DONUT_COLOR,
            )
        )

    if generation is not None and generation.thinking_sample_count:
        cards.append('<div class="chart-card overview"><canvas id="tkOverviewThinking"></canvas></div>')
        calls.append(
            _overview_donut_js(
                "tkOverviewThinking",
                "Thinking tokens",
                generation.total_baseline_thinking_tokens,
                generation.total_optimized_thinking_tokens,
                _TOKEN_DONUT_COLOR,
            )
        )

    combined_categories, combined_before, combined_after = _combined_tokens_chart_data(stats, generation)
    if len(combined_categories) >= 2:
        # Only worth showing once there are at least two dimensions to
        # actually compare -- a single-category "combined" chart would
        # just duplicate the individual donut above it.
        cards.append('<div class="chart-card overview"><canvas id="tkOverviewCombined"></canvas></div>')
        calls.append(_combined_donut_js("tkOverviewCombined", combined_categories, combined_before, combined_after))

    if not cards:
        return ""

    script = "if (window.Chart) {\n" + "\n".join(calls) + "\n}"
    return (
        '<div class="section-title overview-title">Savings overview</div>'
        f'<div class="charts-wrap overview-wrap">{"".join(cards)}</div>'
        f"<script>{script}</script>"
    )


def _render_donut_card(canvas_id: str, name: str, before: int, after: int, color: str) -> str:
    # A single canvas + its own script tag, meant to be dropped inline
    # right next to a metric's existing card row (see
    # _render_output_tokens_section/_render_thinking_tokens_section) --
    # self-contained so it doesn't depend on script placement/ordering
    # elsewhere on the page; the canvas always precedes its own
    # registration script.
    call = _overview_donut_js(canvas_id, name, before, after, color)
    script = f"if (window.Chart) {{\n{call}\n}}"
    return (
        '<div class="charts-wrap donut-inline">'
        f'<div class="chart-card overview"><canvas id="{canvas_id}"></canvas></div>'
        "</div>"
        f"<script>{script}</script>"
    )



def render_html(stats: DashboardStats, log_path: str, tier2: Tier2Stats | None = None) -> str:
    escaped_path = html.escape(log_path)
    if stats.total_runs == 0 and (tier2 is None or not tier2.has_any_data):
        body = _EMPTY_BODY.format(log_path=escaped_path)
        return _PAGE_TEMPLATE.format(log_path=escaped_path, body=body, logo=_LOGO_MARK)

    # Savings overview (2026-09-06): the very first thing on the page,
    # before any metric cards -- per the user's explicit ask.
    overview_section = _render_savings_overview(stats, tier2)

    sign_class = "positive" if stats.pct_saved >= 0 else "negative"
    hero = f"""<div class="section-title">Request tokens</div>
<div class="hero">
  <div>
    <div class="label">Request tokens saved</div>
    <div class="value {sign_class}">{stats.total_saved} ({stats.pct_saved:.1f}%)</div>
    <div class="sub">{stats.total_tokens_before} &rarr; {stats.total_tokens_after} tokens, across {stats.total_runs} logged runs</div>
  </div>
  <div class="badge {sign_class}">{"NET SAVINGS" if stats.pct_saved >= 0 else "NET COST"}</div>
</div>"""

    cards = f"""<div class="cards">
  <div class="card"><div class="label">Runs logged</div><div class="value">{stats.total_runs}</div></div>
  <div class="card"><div class="label">Request tokens before</div><div class="value">{stats.total_tokens_before}</div></div>
  <div class="card"><div class="label">Request tokens after</div><div class="value">{stats.total_tokens_after}</div></div>
</div>"""

    # Per-CLAUDE.md's "report savings as ranges, never a single flat
    # percentage" rule -- the blended total above can hide a lot of real
    # per-run spread (one big win, one big loss, averaging out to something
    # unremarkable). Added 2026-09-06, per the user asking to see it.
    if stats.per_run_pct_saved:
        cards += f"""<div class="cards">
  <div class="card"><div class="label">Min run saved %</div><div class="value">{stats.min_pct_saved:.1f}%</div></div>
  <div class="card"><div class="label">Median run saved %</div><div class="value">{stats.median_pct_saved:.1f}%</div></div>
  <div class="card"><div class="label">Max run saved %</div><div class="value">{stats.max_pct_saved:.1f}%</div></div>
</div>"""

    request_donut = _render_request_tokens_donut(stats)

    # Cache is a REQUEST-side cost (cache_control changes price-per-token
    # on the input side, nothing about generation) -- placed here rather
    # than under Output tokens for that reason (2026-09-06).
    cache_section = _render_cache_subsection(tier2) if tier2 is not None else ""

    rows = []
    for stage in sorted(stats.stages.values(), key=lambda s: s.stage_name):
        no_token_note = _NO_TOKEN_IMPACT_NOTES.get(stage.stage_name)
        if no_token_note is not None:
            saved_cell = "&mdash;"
            pct_cell = f'<span class="stage-note">{html.escape(no_token_note)}</span>'
            range_cell = "&mdash;"
        else:
            pct = stage.pct_saved
            bar_class = "bar-fill negative" if pct < 0 else "bar-fill"
            bar_width = min(abs(pct), 100)
            saved_cell = str(stage.total_saved)
            pct_cell = (
                f'<div class="bar-track"><div class="{bar_class}" style="width:{bar_width}%"></div></div>'
                f"{pct:.1f}%"
            )
            contextual_note = _CONTEXTUAL_NOTES.get(stage.stage_name)
            if contextual_note is not None:
                pct_cell += f'<br><span class="stage-note">{html.escape(contextual_note)}</span>'
            if stage.min_pct_saved is None:
                range_cell = "&mdash;"
            else:
                range_cell = (
                    f"{stage.min_pct_saved:.1f}% / {stage.median_pct_saved:.1f}% / {stage.max_pct_saved:.1f}%"
                )
        rows.append(
            f"""<tr>
  <td>{html.escape(stage.stage_name)}</td>
  <td>{stage.times_seen}</td>
  <td>{stage.times_enabled}</td>
  <td>{stage.total_tokens_before}</td>
  <td>{stage.total_tokens_after}</td>
  <td>{saved_cell}</td>
  <td>{pct_cell}</td>
  <td>{range_cell}</td>
  <td>{stage.avg_timing_seconds * 1000:.1f}ms</td>
  <td>{stage.times_errored}</td>
</tr>"""
        )

    table = f"""<div class="section-title">Pipeline stage breakdown</div>
<table>
<thead><tr>
  <th>Stage</th><th>Runs</th><th>Enabled</th><th>Before</th><th>After</th>
  <th>Saved</th><th>% Saved</th><th>Min / Median / Max %</th><th>Avg time</th><th>Errors</th>
</tr></thead>
<tbody>
{"".join(rows)}
</tbody>
</table>"""

    stage_savings_pie = _render_stage_savings_pie(stats)
    output_section = _render_output_tokens_section(tier2)
    thinking_section = _render_thinking_tokens_section(tier2)
    other_section = _render_other_optimizations_section(tier2) if tier2 is not None else ""

    body = (
        overview_section
        + hero
        + cards
        + request_donut
        + cache_section
        + table
        + stage_savings_pie
        + output_section
        + thinking_section
        + other_section
    )
    return _PAGE_TEMPLATE.format(log_path=escaped_path, body=body, logo=_LOGO_MARK)
