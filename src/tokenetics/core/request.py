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

    `bounded_shape` (added 2026-09-06): a second, independent signal from the
    task classifier -- whether the latest user turn looks like a short,
    single-question/self-contained request rather than an open-ended one.
    `task_type` alone is too coarse to gate AGGRESSIVE brevity (see
    brevity_injector.py's docstring): "code" spans a one-line fix and a
    multi-file design question, "conversational" spans "what's the capital
    of France" and "explain the history of the Cold War". `bounded_shape`
    narrows within a category to the specific shape that was actually
    validated. None/False is the conservative default (ambiguous shape ->
    treat as unbounded, same "skip the aggressive instruction" posture as an
    unclassified task_type).
    """

    task_type: str | None = None
    confidence: float | None = None
    bounded_shape: bool | None = None


CacheAnchor = Literal["system", "tools"]
CacheTTL = Literal["5m", "1h"]


@dataclass
class CacheBreakpoint:
    """Where stage 8 has decided to place a `cache_control` marker.

    Scoped to two anchors for v1 -- "system" (end of the system prompt) and
    "tools" (end of the tool list) -- since those are exactly the two
    blocks stage 7's guard protects as the stable prefix. Anchoring is
    prefix-based: since caching is prefix-based on Anthropic's side too, a
    single marker at the end of the *last* stable block (tools, if any
    exist, otherwise system) is what actually caches everything before it;
    see `to_api_kwargs` for the emission logic.
    """

    anchor: CacheAnchor
    ttl: CacheTTL


@dataclass
class TokeneticsRequest:
    messages: list[Message]
    model: str
    max_tokens: int
    system: str | None = None
    tools: list[ToolSpec] = field(default_factory=list)
    stop_sequences: list[str] = field(default_factory=list)
    meta: RequestMeta = field(default_factory=RequestMeta)
    cache_breakpoint: CacheBreakpoint | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    """Passthrough for API kwargs not yet modeled above (e.g. temperature,
    thinking). Preserved untouched through the pipeline unless a stage
    explicitly opts to read/modify it -- nothing is silently dropped."""


def _extract_system_cache_control(
    system: str | list[dict[str, Any]] | None,
) -> tuple[str | None, CacheTTL | None]:
    """Normalizes `system` back to a plain string, pulling out any
    `cache_control` a previous Tokenetics call (or the caller directly)
    already placed on it -- needed to round-trip a `previous_request` for
    the cache-safety guard without a stale marker looking like new content.
    """
    if system is None:
        return None, None
    if isinstance(system, str):
        return system, None
    text_parts = [block.get("text", "") for block in system]
    ttl = None
    for block in system:
        if "cache_control" in block:
            ttl = block["cache_control"].get("ttl")
    return " ".join(text_parts), ttl


def _extract_tools_cache_control(
    tools: list[dict[str, Any]],
) -> tuple[list[ToolSpec], CacheTTL | None]:
    specs = []
    ttl = None
    for t in tools:
        if "cache_control" in t:
            ttl = t["cache_control"].get("ttl")
        specs.append(
            ToolSpec(name=t["name"], description=t.get("description", ""), input_schema=t["input_schema"])
        )
    return specs, ttl


def from_api_kwargs(
    *,
    messages: list[dict[str, Any]],
    model: str,
    max_tokens: int,
    system: str | list[dict[str, Any]] | None = None,
    tools: list[dict[str, Any]] | None = None,
    stop_sequences: list[str] | None = None,
    **extra: Any,
) -> TokeneticsRequest:
    """Convert raw Anthropic Messages API kwargs into the internal representation.

    `system` accepts either a plain string or the block-list form (so a
    previously-cached request, echoed back as `previous_request`, round-trips
    correctly) -- any `cache_control` found on a system or tool block is
    extracted into `cache_breakpoint` rather than silently dropped.
    """
    system_text, system_ttl = _extract_system_cache_control(system)
    tool_specs, tools_ttl = _extract_tools_cache_control(tools or [])

    cache_breakpoint = None
    if tool_specs and tools_ttl is not None:
        cache_breakpoint = CacheBreakpoint(anchor="tools", ttl=tools_ttl)
    elif system_ttl is not None:
        cache_breakpoint = CacheBreakpoint(anchor="system", ttl=system_ttl)

    return TokeneticsRequest(
        messages=[Message(role=m["role"], content=m["content"]) for m in messages],
        model=model,
        max_tokens=max_tokens,
        system=system_text,
        tools=tool_specs,
        stop_sequences=list(stop_sequences or []),
        cache_breakpoint=cache_breakpoint,
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

    bp = request.cache_breakpoint
    if bp is not None:
        cache_control = {"type": "ephemeral", "ttl": bp.ttl}
        if bp.anchor == "tools" and kwargs.get("tools"):
            kwargs["tools"] = [*kwargs["tools"][:-1], {**kwargs["tools"][-1], "cache_control": cache_control}]
        elif bp.anchor == "system" and kwargs.get("system") is not None:
            kwargs["system"] = [{"type": "text", "text": kwargs["system"], "cache_control": cache_control}]
    return kwargs
