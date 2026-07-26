"""Shared lexical-overlap scoring, used by stage 4's tool-relevance filtering
and stage 9a's structured-output tool matching -- both need the same "how
much does this text have to do with that text" heuristic, so it lives here
once instead of being reimplemented per stage.
"""

from __future__ import annotations

import re

from tokenetics.core.request import Message

_WORD_RE = re.compile(r"[a-z0-9]+")

# Common English function words, excluded so a shared "the"/"a"/"for" doesn't
# register as a real topical match -- matters most for stage 9a, where a
# false-positive match forces tool_choice rather than just keeping a tool
# around (schema minification's relevance filter only drops on exactly zero
# overlap, so stopword noise there just biases further toward keeping, which
# is already the safe direction; 9a's stronger action needs the stricter cut).
_STOPWORDS = {
    "a", "an", "the", "this", "that", "these", "those", "is", "are", "was",
    "were", "be", "been", "being", "to", "of", "in", "on", "at", "for",
    "from", "with", "and", "or", "but", "not", "it", "its", "i", "you",
    "he", "she", "we", "they", "me", "him", "her", "us", "them", "my",
    "your", "his", "their", "our", "as", "by", "if", "so", "do", "does",
    "did", "can", "could", "will", "would", "should", "please", "help",
}


def latest_user_text(messages: list[Message]) -> str:
    for message in reversed(messages):
        if message.role == "user":
            content = message.content
            if isinstance(content, str):
                return content
            parts = [
                block["text"] for block in content if isinstance(block.get("text"), str)
            ]
            return " ".join(parts)
    return ""


def _words(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower())) - _STOPWORDS


def overlap_score(a: str, b: str) -> float:
    """Jaccard-style word overlap between two strings, in [0.0, 1.0].

    0.0 if either side has no words at all (nothing to compare).
    """
    words_a, words_b = _words(a), _words(b)
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)
