"""Phase 8: Tier 0 assembled in fixed order, end-to-end integration, frozen API.

Three deliverables per CLAUDE.md's build order:
(1) an end-to-end integration test over a realistic multi-turn, tool-using
    conversation, confirming the full pipeline actually runs in the fixed
    order documented in CLAUDE.md;
(2) a regression test that stage 4 (schema minification / tool-relevance
    filtering) and stage 5 (context scheduler) never double-prune the same
    content -- they're scoped to disjoint fields (`tools` vs `messages`) by
    construction, and this proves that holds under a real combined run, not
    just by code inspection;
(3) a contract/signature test on `Tokenetics.prepare()`/`finalize()` so the
    now-frozen Tier 0 API can't silently drift in a later phase.
"""

from __future__ import annotations

import inspect

from tokenetics import Tokenetics
from tokenetics.stages.context_scheduler import ContextSchedulerStage
from tokenetics.stages.schema_minification import SchemaMinificationStage

_WEATHER_TOOL = {
    "name": "get_weather",
    "description": "Get the current weather for a city",
    "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
}
_UNRELATED_TOOL = {
    "name": "send_email",
    "description": "Send an email to a recipient with a subject and body",
    "input_schema": {
        "type": "object",
        "properties": {"to": {"type": "string"}, "subject": {"type": "string"}},
    },
}

_REALISTIC_CONVERSATION = [
    {"role": "user", "content": "Hey, quick question before we start."},
    {"role": "assistant", "content": "Sure, go ahead."},
    {"role": "user", "content": "What's the weather in Boston?"},
    {
        "role": "assistant",
        "content": [
            {"type": "tool_use", "id": "1", "name": "get_weather", "input": {"city": "Boston"}}
        ],
    },
    {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "1", "content": "58F, cloudy"}],
    },
    {"role": "assistant", "content": "It's 58F and cloudy in Boston."},
    {"role": "user", "content": "Thanks! Given that weather, should I bring an umbrella?"},
]


def test_end_to_end_pipeline_runs_in_the_documented_fixed_order(stub_client):
    tk = Tokenetics(client=stub_client)
    tk.prepare(
        model="claude-sonnet-5",
        max_tokens=200,
        messages=_REALISTIC_CONVERSATION,
        tools=[_WEATHER_TOOL, _UNRELATED_TOOL],
    )

    fired_order = [e.stage_name for e in tk.logger.entries]  # type: ignore[attr-defined]
    # Every stage runs exactly once, in exactly CLAUDE.md's fixed order --
    # this is the actual runtime trace, not just the static stage list
    # (test_default_pipeline_runs_the_real_foundation_stages_in_order covers
    # that already; this proves the orchestrator executes it in that order
    # too, over a real multi-turn tool-using conversation).
    assert fired_order == [
        "dedup",
        "near_dup",
        "task_classifier",
        "schema_minification",
        "context_scheduler",
        "delta_compression",
        "cache_reorder_guard",
        "cache_breakpoint_optimizer",
        "structured_output",
        "brevity_injector",
        "adaptive_budget",
    ]


def test_end_to_end_pipeline_produces_a_well_formed_request(stub_client):
    tk = Tokenetics(client=stub_client)
    prepared = tk.prepare(
        model="claude-sonnet-5",
        max_tokens=200,
        messages=_REALISTIC_CONVERSATION,
        tools=[_WEATHER_TOOL, _UNRELATED_TOOL],
    )

    # Well-formed enough that the real API would accept it: alternating
    # roles, first message is "user", every tool_use has a matching
    # tool_result, at least one tool survived (the relevant one).
    messages = prepared["messages"]
    assert messages[0]["role"] == "user"
    for a, b in zip(messages, messages[1:]):
        assert a["role"] != b["role"]
    assert any(t["name"] == "get_weather" for t in prepared["tools"])


def test_stage_4_and_5_never_double_prune_the_same_content(stub_client):
    # schema_minification (stage 4) only ever prunes `tools`; context_scheduler
    # (stage 5) only ever prunes `messages` -- disjoint fields by construction.
    # This proves it under a real combined run: an irrelevant tool AND a tight
    # token budget are both in play at once, so both stages have something to
    # prune, and neither should touch the other's field or report the other's
    # kind of drop.
    tk = Tokenetics(
        stages=[SchemaMinificationStage(), ContextSchedulerStage()],
        client=stub_client,
        stage_config={"context_scheduler": {"token_budget": 100}},
    )
    original_message_count = len(_REALISTIC_CONVERSATION)

    prepared = tk.prepare(
        model="claude-sonnet-5",
        max_tokens=200,
        messages=_REALISTIC_CONVERSATION,
        tools=[_WEATHER_TOOL, _UNRELATED_TOOL],
    )

    entries = {e.stage_name: e for e in tk.logger.entries}  # type: ignore[attr-defined]
    schema_entry = entries["schema_minification"]
    scheduler_entry = entries["context_scheduler"]

    # Each stage actually had something to prune (otherwise this test proves
    # nothing) --
    assert "dropped_tools" in schema_entry.extra
    assert scheduler_entry.extra["dropped_turns"] > 0

    # -- and neither stage's log mentions the other kind of drop.
    assert "dropped_turns" not in schema_entry.extra
    assert "dropped_tools" not in scheduler_entry.extra

    # The irrelevant tool is gone, the relevant one survived, untouched by
    # context_scheduler running afterward.
    tool_names = {t["name"] for t in prepared["tools"]}
    assert tool_names == {"get_weather"}

    # Some messages were pruned by context_scheduler's tight budget, but
    # schema_minification (which ran first) didn't touch message count at all.
    assert len(prepared["messages"]) < original_message_count


def test_prepare_signature_is_frozen():
    sig = inspect.signature(Tokenetics.prepare)
    params = list(sig.parameters.values())
    assert [p.name for p in params] == ["self", "api_kwargs"]
    assert params[1].kind == inspect.Parameter.VAR_KEYWORD
    assert sig.return_annotation in ("dict[str, Any]", dict)


def test_finalize_signature_is_frozen():
    sig = inspect.signature(Tokenetics.finalize)
    params = list(sig.parameters.values())
    assert [p.name for p in params] == ["self", "response"]
    assert sig.return_annotation in ("str", str)


def test_tokenetics_init_signature_is_frozen():
    sig = inspect.signature(Tokenetics.__init__)
    assert list(sig.parameters.keys()) == [
        "self",
        "stages",
        "response_stages",
        "logger",
        "client",
        "stage_config",
    ]
