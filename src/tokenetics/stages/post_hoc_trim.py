"""Stage 11: post-hoc output trim.

Deterministic boilerplate stripping only -- no summarization, no semantic
rewriting. Runs on the response text after it's received, via
Tokenetics.finalize(), before it's stored/re-injected as future context.

Patterns are anchored to the end of the text so only an actual trailing
sign-off gets stripped, never similar-looking text mid-answer. Applied
repeatedly until nothing more matches, since stacked sign-offs ("I hope
this helps! Let me know if you have questions.") only expose the next
pattern's match after the outer one is removed.
"""

from __future__ import annotations

import re

from tokenetics.core.logger import CostLogger
from tokenetics.core.plugin import StageConfig
from tokenetics.core.response_stage import ResponseStage

_BOILERPLATE_PATTERNS = [
    re.compile(
        r"\s*Let me know if (you have any questions|you need anything else"
        r"|you'd like [^.!?]*|there's anything else[^.!?]*)[.!]*\s*$",
        re.IGNORECASE,
    ),
    re.compile(r"\s*Feel free to (ask|reach out)[^.!?]*[.!]*\s*$", re.IGNORECASE),
    re.compile(r"\s*(I hope (this|that) helps)[.!]*\s*$", re.IGNORECASE),
    re.compile(r"\s*Happy to (help|answer)[^.!?]*[.!]*\s*$", re.IGNORECASE),
]


class PostHocTrimStage(ResponseStage):
    name = "post_hoc_trim"

    def run(self, text: str, config: StageConfig, logger: CostLogger) -> str:
        trimmed = text
        removed_count = 0
        changed = True
        while changed:
            changed = False
            for pattern in _BOILERPLATE_PATTERNS:
                new_trimmed = pattern.sub("", trimmed)
                if new_trimmed != trimmed:
                    trimmed = new_trimmed
                    removed_count += 1
                    changed = True

        trimmed = trimmed.rstrip()
        if removed_count:
            logger.log_stage(self.name, enabled=True, boilerplate_removed=removed_count)
        return trimmed
