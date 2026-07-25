"""The internal request representation every pipeline stage reads and writes.

Stages never touch raw Anthropic API JSON directly -- conversion happens once
at the pipeline boundary, via `from_api_kwargs` / `to_api_kwargs`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["user", "assistant"]


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class Message:
    role: Role
    content: str | list[dict[str, Any]]


@dataclass
class RequestMeta:
    """Pipeline bookkeeping attached to a request as it moves through stages.

    `task_type`/`confidence` are set once by the task classifier (stage 3) and
    read by later stages (4, 5, 9). More fields land here as those stages are
    built -- this is deliberately minimal for Phase 0.
    """

    task_type: str | None = None
    confidence: float | None = None


@dataclass
class TokeneticsRequest:
    messages: list[Message]
    model: str
    max_tokens: int
    system: str | None = None
    tools: list[ToolSpec] = field(default_factory=list)
    stop_sequences: list[str] = field(default_factory=list)
    meta: RequestMeta = field(default_factory=RequestMeta)
    extra: dict[str, Any] = field(default_factory=dict)
    """Passthrough for API kwargs not yet modeled above (e.g. temperature,
    thinking). Preserved untouched through the pipeline unless a stage
    explicitly opts to read/modify it -- nothing is silently dropped."""


def from_api_kwargs(
    *,
    messages: list[dict[str, Any]],
    model: str,
    max_tokens: int,
    system: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    stop_sequences: list[str] | None = None,
    **extra: Any,
) -> TokeneticsRequest:
    """Convert raw Anthropic Messages API kwargs into the internal representation."""
    return TokeneticsRequest(
        messages=[Message(role=m["role"], content=m["content"]) for m in messages],
        model=model,
        max_tokens=max_tokens,
        system=system,
        tools=[
            ToolSpec(
                name=t["name"],
                description=t.get("description", ""),
                input_schema=t["input_schema"],
            )
            for t in (tools or [])
        ],
        stop_sequences=list(stop_sequences or []),
        extra=extra,
    )


def to_api_kwargs(request: TokeneticsRequest) -> dict[str, Any]:
    """Convert the internal representation back into raw Anthropic Messages API kwargs."""
    kwargs: dict[str, Any] = dict(request.extra)
    kwargs["model"] = request.model
    kwargs["max_tokens"] = request.max_tokens
    kwargs["messages"] = [{"role": m.role, "content": m.content} for m in request.messages]
    if request.system is not None:
        kwargs["system"] = request.system
    if request.tools:
        kwargs["tools"] = [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema}
            for t in request.tools
        ]
    if request.stop_sequences:
        kwargs["stop_sequences"] = request.stop_sequences
    return kwargs
