"""Stage 2: near-dup detection (MinHash/shingling).

Hand-rolled with stdlib only -- no new dependency, keeps Tier 0 lightweight.
Splits each message's text into overlapping word shingles, estimates
Jaccard similarity via MinHash signatures, and drops the older of any pair
that clears the content-class threshold (loose for tool output, strict for
conversation turns).

The "most recent turn is exempt" rule isn't a separate check: this stage
only ever drops the *older* message of a matched pair (i < j), so the last
message -- the highest index -- can never be the one removed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from typing import Any

from tokenetics.core.logger import CostLogger
from tokenetics.core.plugin import Stage, StageConfig
from tokenetics.core.request import Message, TokeneticsRequest

_SHINGLE_SIZE = 4
_NUM_HASHES = 32
_TOOL_OUTPUT_THRESHOLD = 0.85
_CONVERSATION_THRESHOLD = 0.95


def _content_text(content: str | list[dict[str, Any]]) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        text = block.get("text")
        parts.append(text if isinstance(text, str) else json.dumps(block, sort_keys=True))
    return " ".join(parts)


def _is_tool_output(message: Message) -> bool:
    if isinstance(message.content, str):
        return False
    return any(block.get("type") == "tool_result" for block in message.content)


def _threshold_for(a: Message, b: Message) -> float:
    if _is_tool_output(a) or _is_tool_output(b):
        return _TOOL_OUTPUT_THRESHOLD
    return _CONVERSATION_THRESHOLD


def _shingles(text: str, k: int = _SHINGLE_SIZE) -> set[str]:
    words = text.split()
    if len(words) < k:
        return {text} if text else set()
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


def _minhash_signature(shingles: set[str], num_hashes: int = _NUM_HASHES) -> tuple[int, ...]:
    if not shingles:
        return tuple(0 for _ in range(num_hashes))
    signature = []
    for seed in range(num_hashes):
        min_hash = min(
            int(hashlib.sha256(f"{seed}:{shingle}".encode("utf-8")).hexdigest(), 16)
            for shingle in shingles
        )
        signature.append(min_hash)
    return tuple(signature)


def _estimate_similarity(sig_a: tuple[int, ...], sig_b: tuple[int, ...]) -> float:
    if not sig_a or not sig_b:
        return 0.0
    matches = sum(1 for a, b in zip(sig_a, sig_b, strict=True) if a == b)
    return matches / len(sig_a)


class NearDupStage(Stage):
    name = "near_dup"

    def run(
        self, request: TokeneticsRequest, config: StageConfig, logger: CostLogger
    ) -> TokeneticsRequest:
        messages = request.messages
        n = len(messages)
        signatures = [_minhash_signature(_shingles(_content_text(m.content))) for m in messages]
        to_drop: set[int] = set()

        for i in range(n):
            if i in to_drop:
                continue
            for j in range(i + 1, n):
                if j in to_drop:
                    continue
                similarity = _estimate_similarity(signatures[i], signatures[j])
                if similarity >= _threshold_for(messages[i], messages[j]):
                    to_drop.add(i)
                    logger.log_stage(
                        self.name,
                        enabled=True,
                        merged_older_index=i,
                        merged_newer_index=j,
                        similarity=round(similarity, 3),
                    )
                    break

        kept = [m for idx, m in enumerate(messages) if idx not in to_drop]
        return replace(request, messages=kept)
