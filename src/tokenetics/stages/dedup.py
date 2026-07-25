"""Stage 1: exact-match dedup. Always-on, hash-based.

Hashes each message's (role, content) pair; on a repeat hash, drops the
later block and keeps the first occurrence, since later stages and cache
safety depend on earlier positions staying stable.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from typing import Any

from tokenetics.core.logger import CostLogger
from tokenetics.core.plugin import Stage, StageConfig
from tokenetics.core.request import Message, TokeneticsRequest


def _content_hash(role: str, content: str | list[dict[str, Any]]) -> str:
    normalized = content if isinstance(content, str) else json.dumps(content, sort_keys=True)
    return hashlib.sha256(f"{role}:{normalized}".encode("utf-8")).hexdigest()


class DedupStage(Stage):
    name = "dedup"

    def run(
        self, request: TokeneticsRequest, config: StageConfig, logger: CostLogger
    ) -> TokeneticsRequest:
        seen: set[str] = set()
        kept: list[Message] = []
        dropped = 0
        for message in request.messages:
            h = _content_hash(message.role, message.content)
            if h in seen:
                dropped += 1
                continue
            seen.add(h)
            kept.append(message)

        if dropped:
            logger.log_stage(self.name, enabled=True, dropped_duplicates=dropped)
        return replace(request, messages=kept)
