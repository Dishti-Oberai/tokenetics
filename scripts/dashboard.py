#!/usr/bin/env python3
"""Phase 11 dashboard: a small local web view over a `FileCostLogger` log.

Unlike dev_demo.py/benchmark_runner.py, this makes NO network calls of its
own (it's a read-only consumer of an already-written log file) -- still
kept as a script rather than pytest since it's a long-running server, not
a test.

    uv run python scripts/dashboard.py --log-file costs.jsonl
    uv run python scripts/dashboard.py --log-file costs.jsonl --port 9000

Generate some log data first with dev_demo.py's --log-to flag:

    uv run python scripts/dev_demo.py --scenario code --log-to costs.jsonl
    uv run python scripts/dev_demo.py --scenario mixed_workload_recap --log-to costs.jsonl
"""

from __future__ import annotations

import argparse

from tokenetics.dashboard.server import run_server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log-file", required=True, help="Path to a FileCostLogger JSONL log.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    server = run_server(args.log_file, port=args.port, host=args.host)
    url = f"http://{args.host}:{args.port}/"
    print(f"Tokenetics dashboard serving {args.log_file!r} at {url}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
