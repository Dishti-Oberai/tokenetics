"""Renders `DashboardStats` as a single, self-contained HTML page -- no
external requests, no JS framework, no charting library, matching "small
local web view" (the brief's stated v1 form). Kept separate from
`server.py` so the HTML itself is unit-testable without spinning up a real
HTTP server.
"""

from __future__ import annotations

import html

from tokenetics.dashboard.aggregate import DashboardStats, Tier2Stats

_PAGE_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Tokenetics Dashboard</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; margin: 2rem; background: #0b0d12; color: #e6e6e6; }}
  h1 {{ font-size: 1.4rem; margin-bottom: 0.25rem; }}
  .subtitle {{ color: #9aa0aa; font-size: 0.85rem; margin-bottom: 1.5rem; }}
  .cards {{ display: flex; gap: 1rem; margin-bottom: 2rem; flex-wrap: wrap; }}
  .card {{ background: #161a22; border: 1px solid #262c38; border-radius: 8px; padding: 1rem 1.25rem; min-width: 160px; }}
  .card .label {{ font-size: 0.75rem; color: #9aa0aa; text-transform: uppercase; letter-spacing: 0.03em; }}
  .card .value {{ font-size: 1.6rem; font-weight: 600; margin-top: 0.25rem; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ text-align: left; padding: 0.5rem 0.75rem; border-bottom: 1px solid #262c38; font-size: 0.9rem; }}
  th {{ color: #9aa0aa; font-weight: 500; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.03em; }}
  .bar-track {{ background: #262c38; border-radius: 3px; height: 6px; width: 100px; overflow: hidden; }}
  .bar-fill {{ background: #4f8cff; height: 100%; }}
  .bar-fill.negative {{ background: #ff6b6b; }}
  .empty {{ color: #9aa0aa; padding: 2rem 0; }}
  .empty code {{ background: #161a22; padding: 0.15rem 0.4rem; border-radius: 4px; }}
  .section-title {{ font-size: 1rem; margin: 2rem 0 0.75rem; color: #c7cad1; }}
  .no-data {{ color: #6b7280; font-size: 0.85rem; padding: 1rem; background: #161a22; border-radius: 8px; }}
  .scope-note {{ margin-top: 2rem; font-size: 0.8rem; color: #6b7280; border-top: 1px solid #262c38; padding-top: 1rem; }}
</style>
</head>
<body>
<h1>Tokenetics Dashboard</h1>
<div class="subtitle">Read-only view over {log_path} &mdash; refresh this page to re-read the log.</div>
{body}
<div class="scope-note">
  Stage table: per-stage token savings, timing, and enabled/error state from <code>log_stage()</code>.
  Tier 2/cache section: real usage data from <code>log_event()</code>, logged only where the caller
  (e.g. <code>dev_demo.py --measure-usage</code>/<code>--tale</code>/<code>--compress-ratio</code>/
  <code>--semantic-cache</code>) actually computed it -- a mechanism with no logged events shows
  "no data logged" rather than a fabricated number.
</div>
</body>
</html>
"""

_EMPTY_BODY = """<div class="empty">
  No runs logged yet at this path.<br><br>
  Generate some with:<br>
  <code>uv run python scripts/dev_demo.py --scenario code --log-to {log_path}</code>
</div>"""


def _render_tier2_section(tier2: Tier2Stats) -> str:
    def card(label: str, value: str) -> str:
        return f'<div class="card"><div class="label">{label}</div><div class="value">{value}</div></div>'

    parts = ['<div class="section-title">Tier 2 / cache (from log_event())</div>']

    cache = tier2.cache_usage
    if cache.sample_count:
        parts.append(
            '<div class="cards">'
            + card("Cache lookups logged", str(cache.sample_count))
            + card("Cache hit rate", f"{cache.hit_rate:.1f}%")
            + card("Cache read tokens", str(cache.total_cache_read_tokens))
            + card("Cache write tokens", str(cache.total_cache_creation_tokens))
            + "</div>"
        )
    else:
        parts.append('<div class="no-data">No cache usage data logged (needs --measure-usage).</div>')

    tale = tier2.tale
    if tale.sample_count:
        parts.append(
            '<div class="cards">'
            + card("TALE estimates logged", str(tale.sample_count))
            + card("Avg budget estimate", f"{tale.avg_budget_tokens:.0f} tok")
            + card("Avg estimation cost", f"{tale.avg_estimation_cost_tokens:.0f} tok")
            + "</div>"
        )
    else:
        parts.append('<div class="no-data">No TALE data logged (needs --tale).</div>')

    compress = tier2.compress
    if compress.sample_count:
        parts.append(
            '<div class="cards">'
            + card("Compress calls logged", str(compress.sample_count))
            + card("Avg ratio achieved", f"{compress.avg_ratio_achieved:.2f}")
            + card("Clamp rate", f"{compress.clamp_rate:.1f}%")
            + "</div>"
        )
    else:
        parts.append('<div class="no-data">No compress data logged (needs --compress-ratio).</div>')

    semantic = tier2.semantic_cache
    if semantic.sample_count:
        parts.append(
            '<div class="cards">'
            + card("Semantic-cache lookups logged", str(semantic.sample_count))
            + card("Hit rate", f"{semantic.hit_rate:.1f}%")
            + "</div>"
        )
    else:
        parts.append('<div class="no-data">No semantic-cache data logged (needs --semantic-cache).</div>')

    return "".join(parts)


def render_html(stats: DashboardStats, log_path: str, tier2: Tier2Stats | None = None) -> str:
    escaped_path = html.escape(log_path)
    if stats.total_runs == 0 and (tier2 is None or not tier2.has_any_data):
        body = _EMPTY_BODY.format(log_path=escaped_path)
        return _PAGE_TEMPLATE.format(log_path=escaped_path, body=body)

    cards = f"""<div class="cards">
  <div class="card"><div class="label">Runs logged</div><div class="value">{stats.total_runs}</div></div>
  <div class="card"><div class="label">Tokens before</div><div class="value">{stats.total_tokens_before}</div></div>
  <div class="card"><div class="label">Tokens after</div><div class="value">{stats.total_tokens_after}</div></div>
  <div class="card"><div class="label">Saved</div><div class="value">{stats.total_saved} ({stats.pct_saved:.1f}%)</div></div>
</div>"""

    rows = []
    for stage in sorted(stats.stages.values(), key=lambda s: s.stage_name):
        pct = stage.pct_saved
        bar_class = "bar-fill negative" if pct < 0 else "bar-fill"
        bar_width = min(abs(pct), 100)
        rows.append(
            f"""<tr>
  <td>{html.escape(stage.stage_name)}</td>
  <td>{stage.times_seen}</td>
  <td>{stage.times_enabled}</td>
  <td>{stage.total_tokens_before}</td>
  <td>{stage.total_tokens_after}</td>
  <td>{stage.total_saved}</td>
  <td>
    <div class="bar-track"><div class="{bar_class}" style="width:{bar_width}%"></div></div>
    {pct:.1f}%
  </td>
  <td>{stage.avg_timing_seconds * 1000:.1f}ms</td>
  <td>{stage.times_errored}</td>
</tr>"""
        )

    table = f"""<table>
<thead><tr>
  <th>Stage</th><th>Runs</th><th>Enabled</th><th>Before</th><th>After</th>
  <th>Saved</th><th>% Saved</th><th>Avg time</th><th>Errors</th>
</tr></thead>
<tbody>
{"".join(rows)}
</tbody>
</table>"""

    tier2_section = _render_tier2_section(tier2) if tier2 is not None else ""
    return _PAGE_TEMPLATE.format(log_path=escaped_path, body=cards + table + tier2_section)
