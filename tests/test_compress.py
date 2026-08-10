"""Tier 2a tests. Only meaningful under `tokenetics[compress]` installed --
skips cleanly (not a failure) if `transformers`/`torch` aren't available,
per CLAUDE.md's "tested only under their own extras install" rule.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from tokenetics.extras.compress import RECOMMENDED_MAX_RATIO, Compressor, compress_text  # noqa: E402

_SAMPLE_TEXT = (
    "The quick brown fox jumps over the lazy dog. This sentence is used as "
    "a classic pangram because it contains every letter of the English "
    "alphabet at least once, which makes it useful for testing typefaces "
    "and keyboards."
)


def test_compresses_below_the_original_word_count():
    result = compress_text(_SAMPLE_TEXT, ratio=0.3)
    assert result.success is True
    assert result.kept_word_count < result.original_word_count
    assert len(result.compressed_text) < len(_SAMPLE_TEXT)


def test_ratio_achieved_is_close_to_requested():
    result = compress_text(_SAMPLE_TEXT, ratio=0.4)
    assert result.success is True
    # Rounding means it won't be exact, but should be in the ballpark.
    assert abs(result.ratio_achieved - 0.4) < 0.15


def test_kept_words_preserve_original_relative_order():
    result = compress_text(_SAMPLE_TEXT, ratio=0.5)
    assert result.success is True
    original_words = _SAMPLE_TEXT.split()
    kept = result.compressed_text.split()
    # Every kept word must appear in the original in the same relative
    # order (a subsequence check) -- proves this is a selection, not a
    # reordering or hallucination.
    it = iter(original_words)
    assert all(word in it for word in kept)


def test_always_keeps_at_least_one_word():
    result = compress_text("Hello", ratio=0.99)
    assert result.success is True
    assert result.kept_word_count >= 1


def test_fails_open_on_empty_text():
    result = compress_text("", ratio=0.3)
    assert result.success is False
    assert result.compressed_text == ""


def test_fails_open_on_invalid_ratio():
    result = compress_text(_SAMPLE_TEXT, ratio=1.0)
    assert result.success is False
    assert result.compressed_text == _SAMPLE_TEXT

    result = compress_text(_SAMPLE_TEXT, ratio=0.0)
    assert result.success is False
    assert result.compressed_text == _SAMPLE_TEXT


def test_fails_open_on_model_load_failure():
    result = compress_text(_SAMPLE_TEXT, ratio=0.3, model_name="this-model-does-not-exist-xyz")
    assert result.success is False
    assert result.compressed_text == _SAMPLE_TEXT
    assert result.ratio_achieved == 0.0


def test_ratio_above_recommended_max_is_clamped_by_default():
    # Regression/behavior test for the 2026-08-10 real-benchmark finding:
    # ratios above RECOMMENDED_MAX_RATIO (0.4) start dropping specific
    # facts. Default behavior clamps down to the safe ceiling.
    result = compress_text(_SAMPLE_TEXT, ratio=0.7)
    assert result.clamped is True
    assert result.requested_ratio == 0.7
    assert result.ratio_achieved <= RECOMMENDED_MAX_RATIO + 0.1  # rounding slack


def test_ratio_at_or_below_recommended_max_is_not_clamped():
    result = compress_text(_SAMPLE_TEXT, ratio=0.3)
    assert result.clamped is False
    assert result.requested_ratio == 0.3


def test_allow_above_recommended_max_opts_out_of_the_clamp():
    result = compress_text(_SAMPLE_TEXT, ratio=0.7, allow_above_recommended_max=True)
    assert result.clamped is False
    assert result.ratio_achieved > RECOMMENDED_MAX_RATIO + 0.1


def test_reuses_the_loaded_model_across_calls():
    compressor = Compressor()
    first = compressor.compress(_SAMPLE_TEXT, ratio=0.3)
    assert compressor._model is not None
    second = compressor.compress("A short different sentence here.", ratio=0.3)
    assert first.success is True
    assert second.success is True
