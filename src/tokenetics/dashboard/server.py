"""A small local HTTP server for the Phase 11 dashboard -- stdlib
`http.server` only, no new dependency, matching the brief's "small local
web view" v1 form.

Read-only/decoupled by design: every request re-reads the log file fresh
(`load_entries` + `aggregate`) and serves a rendered snapshot -- there is
no shared mutable state between requests, no write path, and no
connection back to a live Tokenetics request pipeline. If this process
crashes, no real request is affected, since nothing calls into it.
"""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from tokenetics.dashboard.aggregate import aggregate, aggregate_events, load_entries
from tokenetics.dashboard.render import render_html


def make_handler(log_path: str) -> type[BaseHTTPRequestHandler]:
    class DashboardHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 (stdlib method name)
            entries = load_entries(log_path)
            stats = aggregate(entries)
            tier2 = aggregate_events(entries)
            body = render_html(stats, log_path, tier2).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            pass  # quiet by default -- dev_demo.py/benchmark_runner.py's scripts set the convention

    return DashboardHandler


def run_server(log_path: str, port: int = 8765, host: str = "127.0.0.1") -> ThreadingHTTPServer:
    """Returns a started-but-not-yet-serving server; caller runs
    `server.serve_forever()` (or, for tests, drives it in a background
    thread and calls `server.shutdown()`).
    """
    handler = make_handler(log_path)
    return ThreadingHTTPServer((host, port), handler)
