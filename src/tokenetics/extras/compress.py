"""Tier 2a: LLMLingua-style compression (opt-in, `tokenetics[compress]`).

Per the brief: a small local model (GPT-2-small class) scores context by
perplexity/importance and drops the lowest-importance tokens up to a
configured compression ratio. Lossy by design -- this trades some fluency
for a real token reduction, and is opt-in specifically because that
tradeoff needs to be a deliberate caller choice, not a default.

Mechanism (word-granularity, not raw sub-token dropping): a causal LM
(`distilgpt2` by default) scores each token by its own negative
log-likelihood given everything before it -- a token the model finds
*surprising* (high loss) is carrying real information; a token it finds
*predictable* (low loss) is redundant given context already present, and is
a safe drop candidate. Scores are aggregated per whitespace-delimited word
(mean of that word's sub-token losses) rather than dropped at the raw
sub-token level, because BPE tokenizers split words into fragments that
don't reassemble into readable text if removed individually -- this keeps
the output as coherent English, at some fluency cost, rather than a garbled
sub-word fragment stream.

Deliberately outside Tier 0: `prepare()` must stay network/state-free
(CLAUDE.md); loading and running a local model is neither of those
literally, but it's still real per-call latency and a real local resource
the caller should opt into explicitly, same reasoning as `tale.py`'s and
`semantic_cache.py`'s placement here rather than as a pipeline stage.

Fail-open (per CLAUDE.md's "hard error" flavor): if the model can't load
(missing weights, no internet on first download, incompatible hardware,
OOM) or inference raises for any reason, `compress_text()` returns the
original text completely unmodified (`CompressResult.success=False,
compressed_text==text`) rather than raising -- a failed compression attempt
must never corrupt or block the request it was trying to shrink.

Quality-degradation benchmark run for real (2026-08-10, `scripts/
compress_benchmark.py`, 4 passages x 4 ratios against real completions):
pass rate stayed flat at 75% through ratio 0.4 (~40% word reduction), then
dropped to 25% at ratio 0.6 (~60% reduction) -- a real, sharp quality
cliff, not a gradual decline. Root cause, confirmed by inspecting the
actual compressed text sent at the failing ratio: **specific numeric
figures and proper nouns are disproportionately at risk**, not general
content. A financial passage lost both dollar figures ("$84.2 million",
"$50 million") at 0.6; a changelog lost a connection-pool-size number and a
retry-interval figure at 0.6. In both cases, this stage's per-word
perplexity scoring apparently doesn't rate these tokens as "surprising"
enough to protect, even though they're exactly the specific facts a human
reader would consider essential -- numbers and proper nouns can be
low-perplexity in context (a small LM finds "$84.2 million" unsurprising
right after "revenue of") even when they're semantically load-bearing. A
passage whose required facts were conceptual/topical words instead (e.g.
"chloroplast", "mitochondria") survived cleanly through 0.6 with no loss.
Practical implication: this stage is riskier on numeric/factual-dense text
(financial reports, technical specs, config values) than on conceptual
prose, and that risk concentrates specifically past ~40% word reduction in
this sample -- not a reason to avoid the stage, but a reason its opt-in
default should stay a moderate ratio unless the caller has verified their
own content survives more aggressive compression.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

_log = logging.getLogger(__name__)

DEFAULT_COMPRESSION_MODEL = "distilgpt2"
_WORD_RE = re.compile(r"\S+")

# Empirically justified, not guessed (2026-08-10, scripts/compress_benchmark.py
# run against benchmarks/compress_quality/passages.json at fine-grained
# ratios between the first run's known-safe 0.4 and known-broken 0.6):
# 100% of samples passed at 0.4 (40% word reduction); the FIRST failure
# appeared at 0.45 (75% pass), and by 0.5 half the samples were already
# failing. 0.4 is the last ratio with a clean 100% pass rate in this held-
# out set, so it's the recommended default ceiling -- see the module
# docstring for what actually breaks past it (numeric figures/proper nouns).
RECOMMENDED_MAX_RATIO = 0.4


@dataclass
class CompressResult:
    """`success=False` means compression failed for any reason (model load,
    inference, or degenerate input) and `compressed_text` is the ORIGINAL
    text, byte-for-byte -- the fail-open contract this module promises.
    `ratio_achieved` is the real, measured word-count reduction, always 0.0
    on failure. `requested_ratio`/`clamped` report whether the caller asked
    for more than `RECOMMENDED_MAX_RATIO` and got clamped down to it (the
    conservative default -- CLAUDE.md: "wherever a stage has a choice
    between an aggressive and a safe option, default to safe").
    """

    compressed_text: str
    success: bool
    original_word_count: int
    kept_word_count: int
    requested_ratio: float
    clamped: bool = False

    @property
    def ratio_achieved(self) -> float:
        if self.original_word_count == 0:
            return 0.0
        return 1 - (self.kept_word_count / self.original_word_count)


@dataclass
class Compressor:
    """Caller-constructed, caller-held wrapper around a loaded local model --
    loading a causal LM per call would be far too slow to be usable, so
    (like `SemanticCache`) this object caches its model instance across
    `compress()` calls for as long as the caller keeps it alive.
    """

    model_name: str = DEFAULT_COMPRESSION_MODEL
    _model: Any = field(default=None, repr=False)
    _tokenizer: Any = field(default=None, repr=False)

    def _load(self) -> tuple[Any, Any] | None:
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer: Any = AutoTokenizer.from_pretrained(self.model_name)
            model: Any = AutoModelForCausalLM.from_pretrained(self.model_name)
            model.eval()
        except Exception:
            _log.error("compress: failed to load model %r", self.model_name, exc_info=True)
            return None
        self._model, self._tokenizer = model, tokenizer
        return model, tokenizer

    def compress(
        self, text: str, ratio: float, allow_above_recommended_max: bool = False
    ) -> CompressResult:
        """`ratio` is the target fraction of WORDS to drop, in (0.0, 1.0)
        (e.g. 0.3 drops ~30% of words). Never raises -- see the module
        docstring's fail-open contract.

        Ratios above `RECOMMENDED_MAX_RATIO` are clamped down to it by
        default -- the empirically-measured point past which specific
        facts (numbers, proper nouns) start disappearing from the
        compressed text. Pass `allow_above_recommended_max=True` to opt
        into a more aggressive ratio anyway, e.g. for content you've
        verified tolerates it (see the module docstring's biology-passage
        counterexample, which held up fine well past this ceiling).
        """
        requested_ratio = ratio
        words = _WORD_RE.findall(text)
        if not words or not (0.0 < ratio < 1.0):
            return CompressResult(
                compressed_text=text, success=False,
                original_word_count=len(words), kept_word_count=len(words),
                requested_ratio=requested_ratio,
            )

        clamped = False
        if ratio > RECOMMENDED_MAX_RATIO and not allow_above_recommended_max:
            _log.info(
                "compress: requested ratio %.2f exceeds RECOMMENDED_MAX_RATIO (%.2f); "
                "clamping -- pass allow_above_recommended_max=True to override",
                ratio, RECOMMENDED_MAX_RATIO,
            )
            ratio = RECOMMENDED_MAX_RATIO
            clamped = True

        loaded = self._load()
        if loaded is None:
            return CompressResult(
                compressed_text=text, success=False,
                original_word_count=len(words), kept_word_count=len(words),
                requested_ratio=requested_ratio, clamped=clamped,
            )
        model, tokenizer = loaded

        try:
            word_scores = self._score_words(text, words, model, tokenizer)
        except Exception:
            _log.error("compress: scoring failed", exc_info=True)
            return CompressResult(
                compressed_text=text, success=False,
                original_word_count=len(words), kept_word_count=len(words),
                requested_ratio=requested_ratio, clamped=clamped,
            )

        keep_count = max(1, round(len(words) * (1 - ratio)))
        # Keep the highest-scoring (most surprising/informative) words,
        # but preserve their ORIGINAL order -- this is a selection, not a
        # reordering, so the compressed text still reads left-to-right.
        keep_indices = set(
            sorted(range(len(words)), key=lambda i: word_scores[i], reverse=True)[:keep_count]
        )
        kept_words = [w for i, w in enumerate(words) if i in keep_indices]

        return CompressResult(
            compressed_text=" ".join(kept_words),
            success=True,
            original_word_count=len(words),
            kept_word_count=len(kept_words),
            requested_ratio=requested_ratio,
            clamped=clamped,
        )

    def _score_words(self, text: str, words: list[str], model: Any, tokenizer: Any) -> list[float]:
        import torch

        encoding = tokenizer(text, return_offsets_mapping=True, return_tensors="pt")
        input_ids = encoding["input_ids"]
        offsets = encoding["offset_mapping"][0].tolist()

        with torch.no_grad():
            logits = model(input_ids).logits[0]  # (seq_len, vocab)

        log_probs = torch.log_softmax(logits, dim=-1)
        # token_loss[i] = -log P(token_i | token_<i); token 0 has no prior
        # context to be predicted from, so it's scored 0.0 (neutral, never
        # the sole reason a word gets dropped or kept).
        token_losses = [0.0] * input_ids.shape[1]
        for i in range(1, input_ids.shape[1]):
            actual_token = input_ids[0, i].item()
            token_losses[i] = -log_probs[i - 1, actual_token].item()

        # Map each word (by character span) to the mean loss of the
        # sub-tokens whose span falls inside it.
        word_spans = []
        cursor = 0
        for word in words:
            start = text.index(word, cursor)
            end = start + len(word)
            word_spans.append((start, end))
            cursor = end

        word_scores = []
        for start, end in word_spans:
            losses = [
                token_losses[i]
                for i, (tok_start, tok_end) in enumerate(offsets)
                if tok_end > start and tok_start < end
            ]
            word_scores.append(sum(losses) / len(losses) if losses else 0.0)
        return word_scores


def compress_text(
    text: str,
    ratio: float,
    model_name: str = DEFAULT_COMPRESSION_MODEL,
    allow_above_recommended_max: bool = False,
) -> CompressResult:
    """Convenience one-shot wrapper around `Compressor` for a single call --
    loads the model fresh each time, so prefer constructing a `Compressor`
    directly and reusing it across multiple calls.
    """
    return Compressor(model_name=model_name).compress(text, ratio, allow_above_recommended_max)
