from tokenetics.core.request import from_api_kwargs
from tokenetics.core.tokenizer import count_text_tokens, count_tokens


def test_count_tokens_returns_a_positive_int(stub_client):
    request = from_api_kwargs(
        model="claude-sonnet-5",
        max_tokens=100,
        messages=[{"role": "user", "content": "hi"}],
    )
    count = count_tokens(request, stub_client)
    assert isinstance(count, int)
    assert count > 0


def test_count_tokens_forwards_only_token_relevant_fields(stub_client):
    request = from_api_kwargs(
        model="claude-sonnet-5",
        max_tokens=100,
        messages=[{"role": "user", "content": "hi"}],
        system="Be terse.",
        temperature=0.5,
    )
    count_tokens(request, stub_client)
    assert len(stub_client.messages.calls) == 1
    forwarded = stub_client.messages.calls[0]
    assert set(forwarded) == {"model", "messages", "system"}
    assert "max_tokens" not in forwarded
    assert "temperature" not in forwarded


def test_count_tokens_forwards_tools_when_present(stub_client):
    request = from_api_kwargs(
        model="claude-sonnet-5",
        max_tokens=100,
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"name": "lookup", "description": "look stuff up", "input_schema": {"type": "object"}}],
    )
    count_tokens(request, stub_client)
    forwarded = stub_client.messages.calls[0]
    assert "tools" in forwarded


def test_count_text_tokens_wraps_plain_text(stub_client):
    count = count_text_tokens("hello world", "claude-sonnet-5", stub_client)
    assert isinstance(count, int)
    assert count > 0
