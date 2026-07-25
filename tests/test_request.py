from tokenetics.core.request import from_api_kwargs, to_api_kwargs


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
