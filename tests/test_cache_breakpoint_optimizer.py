from tokenetics.core.cache_hash import stable_prefix_hash
from tokenetics.core.logger import InMemoryCostLogger
from tokenetics.core.request import from_api_kwargs
from tokenetics.stages.cache_breakpoint_optimizer import CacheBreakpointOptimizerStage

_TOOL = {
    "name": "get_weather",
    "description": "Get the current weather for a city, including temperature and conditions",
    "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
}

# claude-sonnet-5's minimum cacheable prefix is 1024 tokens (see
# core.cache_pricing.MIN_CACHEABLE_TOKENS) -- _estimate_tokens is a ~4
# chars/token estimate, so this needs >=4096 chars to clear it with margin.
_LONG_SYSTEM = (
    "You are a helpful, thorough, and diligent customer support assistant "
    "who answers questions in detail. "
) * 45


def _request(system=_LONG_SYSTEM, tools=None):
    return from_api_kwargs(
        model="claude-sonnet-5",
        max_tokens=100,
        system=system,
        tools=tools,
        messages=[{"role": "user", "content": "hi"}],
    )


def test_no_system_no_tools_is_a_noop():
    stage = CacheBreakpointOptimizerStage()
    request = _request(system=None, tools=None)
    result = stage.run(request, {}, InMemoryCostLogger())
    assert result.cache_breakpoint is None


def test_insufficient_history_places_no_breakpoint():
    stage = CacheBreakpointOptimizerStage()
    request = _request()
    content_hash = stable_prefix_hash(request.system, request.tools)
    # only one matching entry -- can't estimate a repeat interval from that
    config = {"cache_usage_history": [{"timestamp": 100.0, "content_hash": content_hash}]}
    result = stage.run(request, config, InMemoryCostLogger())
    assert result.cache_breakpoint is None
    assert stage.extra["breakpoint_placed"] is False
    assert stage.extra["reason"] == "insufficient_usage_history"


def test_non_matching_history_is_ignored():
    stage = CacheBreakpointOptimizerStage()
    request = _request()
    config = {
        "cache_usage_history": [
            {"timestamp": 100.0, "content_hash": "unrelated-hash-1"},
            {"timestamp": 200.0, "content_hash": "unrelated-hash-2"},
            {"timestamp": 300.0, "content_hash": "unrelated-hash-3"},
        ]
    }
    result = stage.run(request, config, InMemoryCostLogger())
    assert result.cache_breakpoint is None
    assert stage.extra["reason"] == "insufficient_usage_history"


def test_frequent_repeats_choose_the_5m_tier():
    stage = CacheBreakpointOptimizerStage()
    request = _request(tools=[_TOOL])
    content_hash = stable_prefix_hash(request.system, request.tools)
    # repeats every ~60 seconds -- well within the 5-minute window
    history = [
        {"timestamp": float(i * 60), "content_hash": content_hash} for i in range(10)
    ]
    result = stage.run(request, {"cache_usage_history": history}, InMemoryCostLogger())
    assert result.cache_breakpoint is not None
    assert result.cache_breakpoint.ttl == "5m"
    # system present -> anchor to system, which covers tools+system together
    # per Anthropic's tools -> system -> messages cache hierarchy; anchoring
    # to tools here would cache ONLY the (tiny) tools array.
    assert result.cache_breakpoint.anchor == "system"
    assert stage.extra["breakpoint_placed"] is True


def test_anchors_to_tools_only_when_no_system():
    # No system to fold in -- tools is the correct (and only) anchor here.
    stage = CacheBreakpointOptimizerStage()
    long_tool = {
        "name": "get_weather",
        "description": _LONG_SYSTEM,  # reuse as a long, cacheable tool description
        "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
    }
    request = _request(system=None, tools=[long_tool])
    content_hash = stable_prefix_hash(request.system, request.tools)
    history = [{"timestamp": float(i * 60), "content_hash": content_hash} for i in range(10)]
    result = stage.run(request, {"cache_usage_history": history}, InMemoryCostLogger())
    assert result.cache_breakpoint is not None
    assert result.cache_breakpoint.anchor == "tools"


def test_gap_wider_than_5m_but_within_1h_chooses_1h_tier():
    stage = CacheBreakpointOptimizerStage()
    # A ~10-minute repeat gap would let the 5m cache expire between calls
    # (always a fresh write), but the 1h window stays continuously warm.
    request = _request(tools=[_TOOL])
    content_hash = stable_prefix_hash(request.system, request.tools)
    history = [
        {"timestamp": float(i * 600), "content_hash": content_hash} for i in range(10)
    ]
    result = stage.run(request, {"cache_usage_history": history}, InMemoryCostLogger())
    assert result.cache_breakpoint is not None
    assert result.cache_breakpoint.ttl == "1h"


def test_gap_wider_than_1h_places_no_breakpoint():
    stage = CacheBreakpointOptimizerStage()
    request = _request()
    content_hash = stable_prefix_hash(request.system, request.tools)
    # repeats every ~2 hours -- would expire before it repeats again
    history = [
        {"timestamp": float(i * 7200), "content_hash": content_hash} for i in range(5)
    ]
    result = stage.run(request, {"cache_usage_history": history}, InMemoryCostLogger())
    assert result.cache_breakpoint is None
    assert stage.extra["reason"] == "not_cost_effective"


def test_anchors_to_system_when_no_tools():
    stage = CacheBreakpointOptimizerStage()
    request = _request(system=_LONG_SYSTEM, tools=None)
    content_hash = stable_prefix_hash(request.system, request.tools)
    history = [{"timestamp": float(i * 60), "content_hash": content_hash} for i in range(10)]
    result = stage.run(request, {"cache_usage_history": history}, InMemoryCostLogger())
    assert result.cache_breakpoint is not None
    assert result.cache_breakpoint.anchor == "system"


def test_prefix_below_minimum_cacheable_size_places_no_breakpoint():
    # Same frequent-repeat pattern as test_frequent_repeats_choose_the_5m_tier
    # (which would otherwise place a 5m breakpoint), but with a short system
    # prompt -- claude-sonnet-5 requires >=1024 tokens before the real API
    # will actually cache anything, so this must decline even though the
    # repeat interval alone looks favorable.
    stage = CacheBreakpointOptimizerStage()
    request = _request(system="You are a helpful assistant.", tools=[_TOOL])
    content_hash = stable_prefix_hash(request.system, request.tools)
    history = [{"timestamp": float(i * 60), "content_hash": content_hash} for i in range(10)]
    result = stage.run(request, {"cache_usage_history": history}, InMemoryCostLogger())
    assert result.cache_breakpoint is None
    assert stage.extra["breakpoint_placed"] is False
    assert stage.extra["reason"] == "prefix_below_minimum_cacheable_size"
    assert stage.extra["minimum_required_tokens"] == 1024
