"""Stable-prefix content hashing, shared by the cache-safety guard and the
breakpoint optimizer -- and by callers building `cache_usage_history`
entries, since the optimizer can only match history against the current
request's stable prefix if both sides hash it the same way.
"""

from __future__ import annotations

import hashlib
import json

from tokenetics.core.request import ToolSpec


def stable_prefix_hash(system: str | None, tools: list[ToolSpec]) -> str:
    normalized = {
        "system": system,
        "tools": [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema}
            for t in tools
        ],
    }
    blob = json.dumps(normalized, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
