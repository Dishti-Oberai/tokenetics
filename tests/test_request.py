from dataclasses import replace

from tokenetics.core.request import CacheBreakpoint, from_api_kwargs, to_api_kwargs


def test_round_trip_preserves_known_fields():
    kwargs = {
        "model": "claude-sonnet-5",
        "max_tokens": 100,
        "system": "Be terse.",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [
            {
                "name": "lookup",
                "description": "look stuff up",
                "input_schema": {"type": "object"},
            }
        ],
        "stop_sequences": ["STOP"],
    }
    request = from_api_kwargs(**kwargs)
    assert to_api_kwargs(request) == kwargs


def test_round_trip_preserves_unmodeled_kwargs():
    kwargs = {
        "model": "claude-sonnet-5",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.5,
        "thinking": {"type": "enabled", "budget_tokens": 2000},
    }
    request = from_api_kwargs(**kwargs)
    out = to_api_kwargs(request)
    assert out["temperature"] == 0.5
    assert out["thinking"] == {"type": "enabled", "budget_tokens": 2000}


def test_omits_unset_optional_fields():
    kwargs = {
        "model": "claude-sonnet-5",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hi"}],
    }
    request = from_api_kwargs(**kwargs)
    out = to_api_kwargs(request)
    assert "system" not in out
    assert "tools" not in out
    assert "stop_sequences" not in out


def test_cache_breakpoint_on_tools_round_trips_through_api_kwargs():
    tool = {"name": "lookup", "description": "look stuff up", "input_schema": {"type": "object"}}
    request = from_api_kwargs(
        model="claude-sonnet-5",
        max_tokens=100,
        messages=[{"role": "user", "content": "hi"}],
        tools=[tool],
    )
    request = replace(request, cache_breakpoint=CacheBreakpoint(anchor="tools", ttl="1h"))

    out = to_api_kwargs(request)
    assert out["tools"][-1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}

    # round-trip: parsing that back should recover the same breakpoint
    reparsed = from_api_kwargs(**out)
    assert reparsed.cache_breakpoint == CacheBreakpoint(anchor="tools", ttl="1h")
    assert reparsed.tools[0].description == "look stuff up"  # cache_control stripped from the ToolSpec


def test_cache_breakpoint_on_system_round_trips_through_api_kwargs():
    request = from_api_kwargs(
        model="claude-sonnet-5",
        max_tokens=100,
        messages=[{"role": "user", "content": "hi"}],
        system="Be terse.",
    )
    request = replace(request, cache_breakpoint=CacheBreakpoint(anchor="system", ttl="5m"))

    out = to_api_kwargs(request)
    assert out["system"] == [
        {"type": "text", "text": "Be terse.", "cache_control": {"type": "ephemeral", "ttl": "5m"}}
    ]

    reparsed = from_api_kwargs(**out)
    assert reparsed.cache_breakpoint == CacheBreakpoint(anchor="system", ttl="5m")
    assert reparsed.system == "Be terse."


def test_no_cache_breakpoint_by_default():
    request = from_api_kwargs(
        model="claude-sonnet-5", max_tokens=100, messages=[{"role": "user", "content": "hi"}]
    )
    assert request.cache_breakpoint is None
    assert "cache_control" not in str(to_api_kwargs(request))
