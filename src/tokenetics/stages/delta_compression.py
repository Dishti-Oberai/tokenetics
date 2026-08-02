"""Stage 6: structured delta compression.

Problem (per the project brief): agentic loops re-send large, mostly-
unchanged payloads -- a re-read file after a small edit, a polled API
response -- at full price every time.

Scope (resolved 2026-08-02, see the brief's stage 6 amendment): `tool_result`
content blocks only, matched by `tool_use_id`. Caller supplies
`config["previous_payloads"]: dict[str, str]` mapping a tool_result's
`tool_use_id` -- unique within the current request, already known to the
caller when they construct that reply -- to whatever they consider its
previous version. Plain message text is deliberately out of scope: near_dup
(stage 2) already handles prose-level repeats across turns, and identifying
"the same" text block across turns without an id is a genuinely different,
harder problem this stage doesn't attempt.

Only `tool_result` blocks whose `content` is a plain string are eligible --
list-of-blocks content (e.g. an image result) is left untouched, conservative
by default rather than guessed at.

Whether Claude reliably interprets a delta-encoded tool_result as "this
changed from a previous value" is a real model-quality question, not a
mechanical one -- explicitly deferred to Phase 9's held-out quality-check
set (same treatment as brevity/thinking-effort), not assumed to just work.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from tokenetics.core.delta import compute_delta
from tokenetics.core.logger import CostLogger
from tokenetics.core.plugin import Stage, StageConfig
from tokenetics.core.request import Message, TokeneticsRequest


class DeltaCompressionStage(Stage):
    name = "delta_compression"

    def run(
        self, request: TokeneticsRequest, config: StageConfig, logger: CostLogger
    ) -> TokeneticsRequest:
        previous_payloads: dict[str, str] = config.get("previous_payloads", {})
        if not previous_payloads:
            self.note(compressed_count=0, reason="no_previous_payloads_supplied")
            return request

        compressed = 0
        skipped_not_smaller = 0
        skipped_non_string_content = 0
        new_messages: list[Message] = []

        for message in request.messages:
            if not isinstance(message.content, list):
                new_messages.append(message)
                continue

            changed = False
            new_blocks: list[dict[str, Any]] = []
            for block in message.content:
                tool_use_id = block.get("tool_use_id")
                is_targeted = (
                    block.get("type") == "tool_result"
                    and isinstance(tool_use_id, str)
                    and tool_use_id in previous_payloads
                )
                if not is_targeted:
                    new_blocks.append(block)
                    continue
                assert isinstance(tool_use_id, str)  # narrowed by is_targeted above

                content = block.get("content")
                if not isinstance(content, str):
                    skipped_non_string_content += 1
                    new_blocks.append(block)
                    continue

                delta_text = compute_delta(previous_payloads[tool_use_id], content)
                if delta_text is None:
                    skipped_not_smaller += 1
                    new_blocks.append(block)
                    continue

                new_blocks.append({**block, "content": delta_text})
                compressed += 1
                changed = True

            new_messages.append(replace(message, content=new_blocks) if changed else message)

        self.note(
            compressed_count=compressed,
            skipped_not_meaningfully_smaller=skipped_not_smaller,
            skipped_non_string_content=skipped_non_string_content,
        )
        if compressed == 0:
            return request
        return replace(request, messages=new_messages)
