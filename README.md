# tokenetics

A stateless Python SDK wrapper around the Anthropic API that shapes both sides of a request to cut token cost without degrading response quality:

- **Input side** — deduplicates repeated context, optimizes prompt-cache placement, and schedules what goes into the context window.
- **Output side** — constrains brevity, output structure, and token caps on what comes back.

Backed by a measured (not just modeled) benchmark suite, with an opt-in tier of heavier compression/caching plugins.

## Status

Early scaffolding — no implementation yet.

## Development

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync
uv run pytest
```
