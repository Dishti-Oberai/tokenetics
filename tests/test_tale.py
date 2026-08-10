from dataclasses import dataclass, field
from typing import Any

from tokenetics.extras.tale import estimate_token_budget


@dataclass
class _FakeBlock:
    type: str
    text: str = ""


@dataclass
class _FakeUsage:
    input_tokens: int
    output_tokens: int


@dataclass
class _FakeResponse:
    content: list[_FakeBlock] = field(default_factory=list)
    usage: _FakeUsage | None = None


class _FakeMessages:
    def __init__(self, reply_text: str, raise_error: bool = False) -> None:
        self.reply_text = reply_text
        self.raise_error = raise_error
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> _FakeResponse:
        self.calls.append(kwargs)
        if self.raise_error:
            raise RuntimeError("simulated API failure")
        return _FakeResponse(
            content=[_FakeBlock(type="text", text=self.reply_text)],
            usage=_FakeUsage(input_tokens=40, output_tokens=3),
        )


class _FakeClient:
    def __init__(self, reply_text: str = "250", raise_error: bool = False) -> None:
        self.messages = _FakeMessages(reply_text, raise_error=raise_error)


def test_parses_a_clean_integer_reply():
    client = _FakeClient(reply_text="250")
    result = estimate_token_budget(client, messages=[{"role": "user", "content": "hi"}])
    assert result is not None
    assert result.budget_tokens == 250
    assert result.estimation_input_tokens == 40
    assert result.estimation_output_tokens == 3


def test_parses_an_integer_embedded_in_extra_text():
    # Fail-open should still extract a usable number even if the model
    # didn't follow the "only a number" instruction exactly.
    client = _FakeClient(reply_text="Approximately 400 tokens.")
    result = estimate_token_budget(client, messages=[{"role": "user", "content": "hi"}])
    assert result is not None
    assert result.budget_tokens == 400


def test_returns_none_on_unparseable_reply():
    client = _FakeClient(reply_text="I'm not sure.")
    result = estimate_token_budget(client, messages=[{"role": "user", "content": "hi"}])
    assert result is None


def test_returns_none_on_api_failure_rather_than_raising():
    client = _FakeClient(raise_error=True)
    result = estimate_token_budget(client, messages=[{"role": "user", "content": "hi"}])
    assert result is None


def test_uses_a_cheaper_estimation_model_by_default():
    client = _FakeClient(reply_text="250")
    estimate_token_budget(client, messages=[{"role": "user", "content": "hi"}], model="claude-sonnet-5")
    call = client.messages.calls[0]
    assert call["model"] == "claude-haiku-4-5"


def test_estimation_prompt_includes_the_real_model_and_conversation():
    client = _FakeClient(reply_text="250")
    estimate_token_budget(
        client,
        messages=[{"role": "user", "content": "explain quantum tunneling"}],
        model="claude-sonnet-5",
    )
    call = client.messages.calls[0]
    prompt = call["messages"][0]["content"]
    assert "claude-sonnet-5" in prompt
    assert "explain quantum tunneling" in prompt


def test_estimation_prompt_includes_system_prompt_when_supplied():
    client = _FakeClient(reply_text="250")
    estimate_token_budget(
        client,
        messages=[{"role": "user", "content": "hi"}],
        system="You are a helpful assistant that writes Python code.",
    )
    call = client.messages.calls[0]
    prompt = call["messages"][0]["content"]
    assert "writes Python code" in prompt
