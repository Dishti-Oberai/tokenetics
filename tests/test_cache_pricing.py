from tokenetics.core.cache_pricing import base_input_price_for, min_cacheable_tokens_for


def test_base_input_price_for_known_model():
    assert base_input_price_for("claude-sonnet-5") == 3.00


def test_base_input_price_for_unknown_model_is_conservative():
    # Highest known base price, never assume a cheaper price than
    # guaranteed -- that would understate the write premium's real cost.
    assert base_input_price_for("some-model-nobody-has-heard-of") == 10.00


def test_min_cacheable_tokens_for_known_model():
    assert min_cacheable_tokens_for("claude-sonnet-5") == 1024


def test_min_cacheable_tokens_for_unknown_model_is_conservative():
    # Highest known minimum, never assume a smaller prefix is cacheable
    # than guaranteed -- that would place breakpoints the API would ignore.
    assert min_cacheable_tokens_for("some-model-nobody-has-heard-of") == 4096
