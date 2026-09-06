"""Stage 3: task classifier (shared signal, not itself an optimizer).

Deterministic regex/keyword scoring over the latest user message, plus a
structural scan of the whole message list for tool_use/tool_result content.
Runs once per request and writes (task_type, confidence) onto request.meta
for stages 4 and 9 to read later -- this stage never itself changes tokens,
it only annotates.

Conservative by design: below _UNCLASSIFIED_THRESHOLD, task_type is left
None rather than guessing, so downstream stages take their conservative
branch (per CLAUDE.md's "ambiguous task shape -> skip the brevity
instruction" etc.) instead of acting on a low-confidence label.

`bounded_shape` (added 2026-09-06): a second, independent scan alongside
task_type, distinguishing a short/self-contained/single-question request
from an open-ended one. Built specifically to gate brevity_injector's
AGGRESSIVE default -- see that module's docstring for why task_type alone
(code/conversational/tool-heavy/extraction) is too coarse a proxy for "is
this the shape AGGRESSIVE was actually validated on." Deterministic,
regex/word-count based, same conservative posture as task_type itself:
anything ambiguous (empty text, an open-ended marker, more than one real
question, or a long prose body) resolves to False/unbounded, never guessed
True.
"""

from __future__ import annotations

import re
from dataclasses import replace

from tokenetics.core.lexical import latest_user_text
from tokenetics.core.logger import CostLogger
from tokenetics.core.plugin import Stage, StageConfig
from tokenetics.core.request import Message, TokeneticsRequest

TASK_TYPES = ("code", "conversational", "extraction", "tool-heavy")

_UNCLASSIFIED_THRESHOLD = 0.3

_CODE_FENCE_RE = re.compile(r"```")
_CODE_KEYWORD_RE = re.compile(
    r"\b(def|class|function|import|const|let|var|return|console\.log|"
    r"public\s+static|#include|SELECT\b.*\bFROM)\b",
    re.IGNORECASE,
)
_ERROR_TRACE_RE = re.compile(
    r"(Traceback \(most recent call last\)|Exception in thread|"
    r"\b[A-Za-z_.]+(Error|Exception):|at [\w.$]+\([\w./]+:\d+\))"
)

_EXTRACTION_RE = re.compile(
    r"\b(list all|extract|return (a |the )?json|return a list|"
    r"summarize as bullet|format (this |it )?as (a )?table|give me a table|"
    r"parse (this|the)|return (only )?the (following|fields))\b",
    re.IGNORECASE,
)

_TOOL_CONTENT_TYPES = {"tool_use", "tool_result"}

_CODE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)
_OPEN_ENDED_RE = re.compile(
    r"\b(design|architect(ure)?|refactor|from scratch|entire (codebase|system|"
    r"application|project)|walk me through|compare and contrast|pros and cons|"
    r"best practices|brainstorm|discuss|elaborate|in detail|multiple files|"
    r"end.to.end|comprehensive|step.by.step guide)\b",
    re.IGNORECASE,
)
_BOUNDED_WORD_LIMIT = 40


def _is_bounded_shape(text: str) -> bool:
    # Strip fenced code blocks before judging length/open-endedness -- a
    # short question about a 15-line snippet shouldn't read as "long prose"
    # just because the snippet itself is long.
    prose = _CODE_BLOCK_RE.sub(" ", text).strip()
    if not prose:
        return False
    if _OPEN_ENDED_RE.search(prose):
        return False
    if prose.count("?") > 1:
        return False
    return len(prose.split()) <= _BOUNDED_WORD_LIMIT


def _has_tool_content(messages: list[Message]) -> bool:
    for message in messages:
        if isinstance(message.content, list):
            for block in message.content:
                if block.get("type") in _TOOL_CONTENT_TYPES:
                    return True
    return False


def _score_code(text: str) -> float:
    score = 0.0
    if _CODE_FENCE_RE.search(text):
        score += 0.6
    if _CODE_KEYWORD_RE.search(text):
        score += 0.45
    if _ERROR_TRACE_RE.search(text):
        score += 0.5
    return min(score, 1.0)


def _score_extraction(text: str) -> float:
    return 0.8 if _EXTRACTION_RE.search(text) else 0.0


def _score_tool_heavy(messages: list[Message], has_tools: bool) -> float:
    if _has_tool_content(messages):
        return 0.9
    if has_tools:
        return 0.3
    return 0.0


def _score_conversational(text: str) -> float:
    # Weak baseline for ordinary questions/chat -- deliberately lower than
    # any single code/extraction/tool-heavy signal so a real marker always
    # wins, while still clearing _UNCLASSIFIED_THRESHOLD on its own.
    return 0.4 if text.strip() else 0.0


class TaskClassifierStage(Stage):
    name = "task_classifier"

    def run(
        self, request: TokeneticsRequest, config: StageConfig, logger: CostLogger
    ) -> TokeneticsRequest:
        text = latest_user_text(request.messages)
        scores = {
            "code": _score_code(text),
            "extraction": _score_extraction(text),
            "tool-heavy": _score_tool_heavy(request.messages, bool(request.tools)),
            "conversational": _score_conversational(text),
        }
        best_type, confidence = max(scores.items(), key=lambda item: item[1])
        task_type: str | None = best_type if confidence >= _UNCLASSIFIED_THRESHOLD else None
        bounded_shape = _is_bounded_shape(text)

        self.note(task_type=task_type, confidence=round(confidence, 3), bounded_shape=bounded_shape)
        meta = replace(
            request.meta, task_type=task_type, confidence=confidence, bounded_shape=bounded_shape
        )
        return replace(request, meta=meta)
