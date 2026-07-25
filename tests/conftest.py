"""Shared pytest fixtures.

`stub_client` stands in for anthropic.Anthropic() so tests that exercise
Tokenetics.prepare()/finalize() never hit the network or need an API key.
`make_fake_response` builds a minimal stand-in for a Messages API response,
for tests that exercise Tokenetics.finalize().
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pytest


@dataclass
class _CountTokensResponse:
    input_tokens: int


class _StubMessages:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def count_tokens(self, **kwargs: Any) -> _CountTokensResponse:
        self.calls.append(kwargs)
        size = len(str(kwargs.get("messages", "")))
        size += len(str(kwargs.get("system", "") or ""))
        size += len(str(kwargs.get("tools", "") or ""))
        return _CountTokensResponse(input_tokens=size)


class StubAnthropicClient:
    """A minimal stand-in for anthropic.Anthropic() -- no network, no API key."""

    def __init__(self) -> None:
        self.messages = _StubMessages()


@pytest.fixture
def stub_client() -> StubAnthropicClient:
    return StubAnthropicClient()


@dataclass
class _FakeContentBlock:
    type: str
    text: str = ""


@dataclass
class FakeResponse:
    model: str
    content: list[_FakeContentBlock] = field(default_factory=list)


@pytest.fixture
def make_fake_response() -> Callable[..., FakeResponse]:
    def _make(text: str, model: str = "claude-sonnet-5") -> FakeResponse:
        return FakeResponse(model=model, content=[_FakeContentBlock(type="text", text=text)])

    return _make
