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


def test_count_tokens_forwards_tool_choice_when_present(stub_client):
    # Regression test for a real gap (2026-09-06): tool_choice used to be
    # excluded on the wrong assumption it was a generation-only param like
    # max_tokens -- Anthropic's real count_tokens endpoint documents
    # tool_choice as an accepted, count-affecting parameter, and a real
    # structured_output-forced call showed a ~128-129 token gap between
    # this project's own (under-)counting and the real measured usage.
    request = from_api_kwargs(
        model="claude-sonnet-5",
        max_tokens=100,
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"name": "lookup", "description": "look stuff up", "input_schema": {"type": "object"}}],
        tool_choice={"type": "tool", "name": "lookup"},
    )
    count_tokens(request, stub_client)
    forwarded = stub_client.messages.calls[0]
    assert forwarded.get("tool_choice") == {"type": "tool", "name": "lookup"}


def test_count_tokens_forwards_thinking_and_output_config_when_present(stub_client):
    # Same class of gap as tool_choice -- both are documented, accepted
    # parameters of the real count_tokens endpoint.
    request = from_api_kwargs(
        model="claude-sonnet-5",
        max_tokens=100,
        messages=[{"role": "user", "content": "hi"}],
        thinking={"type": "adaptive"},
        output_config={"effort": "high"},
    )
    count_tokens(request, stub_client)
    forwarded = stub_client.messages.calls[0]
    assert forwarded.get("thinking") == {"type": "adaptive"}
    assert forwarded.get("output_config") == {"effort": "high"}


def test_count_text_tokens_wraps_plain_text(stub_client):
    count = count_text_tokens("hello world", "claude-sonnet-5", stub_client)
    assert isinstance(count, int)
    assert count > 0
