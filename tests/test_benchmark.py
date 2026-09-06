import json
from pathlib import Path

from tokenetics.core.benchmark import (
    SavingsRecord,
    aggregate_savings_by_category,
    check_required_elements,
    check_tool_calls,
    estimate_cost_usd,
    load_corpus,
    load_quality_checks,
    response_text_with_tool_inputs,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_corpus_file(tmp_path: Path, name: str, records: list[dict]) -> None:
    (tmp_path / name).write_text(json.dumps(records))


def test_load_corpus_reads_all_json_files_in_a_directory(tmp_path):
    _write_corpus_file(
        tmp_path,
        "code.json",
        [
            {
                "id": "code_001",
                "category": "code",
                "description": "d",
                "kwargs": {"model": "claude-sonnet-5", "max_tokens": 100, "messages": []},
            }
        ],
    )
    _write_corpus_file(
        tmp_path,
        "conversational.json",
        [
            {
                "id": "conv_001",
                "category": "conversational",
                "description": "d",
                "kwargs": {"model": "claude-sonnet-5", "max_tokens": 100, "messages": []},
            }
        ],
    )
    samples = load_corpus(tmp_path)
    assert {s.id for s in samples} == {"code_001", "conv_001"}
    assert {s.category for s in samples} == {"code", "conversational"}
    assert samples[0].stage_config is None  # absent in the file -- not silently defaulted to {}


def test_load_corpus_reads_optional_stage_config(tmp_path):
    _write_corpus_file(
        tmp_path,
        "tool_heavy.json",
        [
            {
                "id": "th_001",
                "category": "tool-heavy",
                "description": "d",
                "kwargs": {"model": "claude-sonnet-5", "max_tokens": 100, "messages": []},
                "stage_config": {"delta_compression": {"previous_payloads": {"toolu_1": "x"}}},
            }
        ],
    )
    samples = load_corpus(tmp_path)
    assert samples[0].stage_config == {"delta_compression": {"previous_payloads": {"toolu_1": "x"}}}


def test_the_real_corpus_directory_parses_and_covers_all_four_categories():
    samples = load_corpus(_REPO_ROOT / "benchmarks" / "corpus")
    # 20 per category clears the TOP of CLAUDE.md's stated 15-20+ floor,
    # not just the low end.
    assert len(samples) >= 80
    categories = {s.category for s in samples}
    # `mixed_workload` (Phase 10) is a deliberately separate, non-task-type
    # bucket -- not one of task_classifier's 4 real categories, so it's
    # checked for presence separately below, not folded into this set.
    assert {"code", "conversational", "extraction", "tool-heavy"} <= categories
    ids = [s.id for s in samples]
    assert len(ids) == len(set(ids))  # no duplicate sample ids

    by_category: dict[str, int] = {}
    for sample in samples:
        by_category[sample.category] = by_category.get(sample.category, 0) + 1
    for category, count in by_category.items():
        if category == "mixed_workload":
            continue
        assert count >= 20, f"{category} has only {count} samples, below the 15-20+ floor"


def test_the_real_corpus_directory_has_a_mixed_workload_bucket():
    # Deferred from Phase 9, added Phase 10: a single realistic multi-turn
    # session spanning several task types in one conversation, reported as
    # its own explicitly-labeled bucket by benchmark_runner.py's `corpus`
    # command -- never blended into the 4 categorized ranges above.
    samples = load_corpus(_REPO_ROOT / "benchmarks" / "corpus")
    mixed = [s for s in samples if s.category == "mixed_workload"]
    assert len(mixed) >= 2


def test_real_corpus_stage_config_samples_actually_engage_their_stage(stub_client):
    # tool_heavy_016/017 exist specifically to give the tool-heavy category
    # something other than a flat 0.0% in the corpus report -- verify they
    # actually engage delta_compression/cache_breakpoint_optimizer, not just
    # that they parse.
    from tokenetics import Tokenetics

    samples = {s.id: s for s in load_corpus(_REPO_ROOT / "benchmarks" / "corpus")}

    delta_sample = samples["tool_heavy_016"]
    tk = Tokenetics(client=stub_client, stage_config=delta_sample.stage_config)
    tk.prepare(**delta_sample.kwargs)
    delta_entry = next(e for e in tk.logger.entries if e.stage_name == "delta_compression")
    assert delta_entry.extra["compressed_count"] == 1

    cache_sample = samples["tool_heavy_017"]
    tk = Tokenetics(client=stub_client, stage_config=cache_sample.stage_config)
    tk.prepare(**cache_sample.kwargs)
    cache_entry = next(e for e in tk.logger.entries if e.stage_name == "cache_breakpoint_optimizer")
    assert cache_entry.extra["breakpoint_placed"] is True


def test_mixed_workload_samples_actually_engage_near_dup(stub_client):
    # The mixed_workload samples' near-duplicate small-talk padding was
    # empirically verified (2026-08-10) to clear near_dup's 0.95
    # conversation threshold, not assumed -- confirm that holds through the
    # real pipeline, not just an isolated near_dup._shingles check.
    from tokenetics import Tokenetics

    samples = {s.id: s for s in load_corpus(_REPO_ROOT / "benchmarks" / "corpus")}
    for sample_id in ("mixed_workload_001", "mixed_workload_002", "mixed_workload_003"):
        sample = samples[sample_id]
        tk = Tokenetics(client=stub_client, stage_config=sample.stage_config)
        tk.prepare(**sample.kwargs)
        near_dup_entry = next(e for e in tk.logger.entries if e.stage_name == "near_dup")
        assert len(near_dup_entry.extra.get("merges", [])) >= 2, sample_id


def test_real_corpus_sample_demonstrates_the_tool_reuse_window_expiring(stub_client):
    # tool_heavy_018 exists specifically to show the schema_minification
    # reuse-window revision (2026-08-09) actually expiring in a real corpus
    # sample, not just in test_schema_minification.py's synthetic cases --
    # 5 genuine unrelated follow-ups after the tool's use is past the
    # 3-turn window, so it should be dropped again, reclaiming the tokens.
    from tokenetics import Tokenetics

    samples = {s.id: s for s in load_corpus(_REPO_ROOT / "benchmarks" / "corpus")}
    sample = samples["tool_heavy_018"]
    tk = Tokenetics(client=stub_client)
    prepared = tk.prepare(**sample.kwargs)
    assert prepared.get("tools", []) == []
    sm_entry = next(e for e in tk.logger.entries if e.stage_name == "schema_minification")
    assert sm_entry.extra["dropped_tools"] == [{"tool": "get_weather", "score": 0.0}]


def test_real_quality_check_samples_actually_engage_their_mechanism(stub_client):
    # Each new quality-check sample added 2026-08-09 exists to test a
    # specific mechanism -- verify it actually engages that mechanism
    # offline (mechanically, not the completion quality itself), same
    # discipline as every other stage_config-equipped sample in this suite.
    from tokenetics import Tokenetics

    samples = {s.id: s for s in load_quality_checks(_REPO_ROOT / "benchmarks" / "quality_checks")}

    # delta_compression_002: line_patch (text diff), not json_patch.
    sample = samples["quality_delta_compression_002"]
    tk = Tokenetics(client=stub_client, stage_config=sample.stage_config)
    prepared = tk.prepare(**sample.kwargs)
    delta_text = prepared["messages"][-1]["content"][0]["content"]
    assert '"format": "line_patch"' in delta_text

    # context_scheduler_002: the unresolved error survives a tight budget.
    sample = samples["quality_context_scheduler_002"]
    tk = Tokenetics(client=stub_client, stage_config=sample.stage_config)
    prepared = tk.prepare(**sample.kwargs)
    assert any("IndexError" in str(m["content"]) for m in prepared["messages"])

    # schema_minification_002: the never-used, genuinely irrelevant tool is dropped.
    sample = samples["quality_schema_minification_002"]
    tk = Tokenetics(client=stub_client)
    prepared = tk.prepare(**sample.kwargs)
    assert prepared.get("tools", []) == []

    # schema_minification_003: exactly at the reuse window boundary -- still kept.
    sample = samples["quality_schema_minification_003"]
    tk = Tokenetics(client=stub_client)
    prepared = tk.prepare(**sample.kwargs)
    assert [t["name"] for t in prepared.get("tools", [])] == ["check_inventory"]

    # adaptive_budget_001 (2026-09-04): needs stage_config to opt into
    # thinking-effort, since that sub-stage became opt-in-only after a real
    # $-cost benchmark -- without the flag this sample wouldn't exercise
    # what it's meant to test at all. Effort is "medium", not "high"
    # (changed 2026-09-06): this sample's prompt is short/single-question
    # (meta.bounded_shape=True), so the new bounded-shape downgrade
    # correctly kicks in even though its task_type ("code") would
    # otherwise map to "high" -- this sample is now ALSO the real risk
    # case for that downgrade (a bounded-looking prompt that still needs
    # genuine reasoning depth), see the real quality-check run in
    # ROADMAP.md for whether "medium" still solves it correctly.
    sample = samples["quality_adaptive_budget_001"]
    tk = Tokenetics(client=stub_client, stage_config=sample.stage_config)
    tk.prepare(**sample.kwargs)
    entry = next(e for e in tk.logger.entries if e.stage_name == "adaptive_budget")
    assert entry.extra["thinking_effort"] == "medium"

    # adaptive_budget_003/004 (2026-09-06): two more real risk-case samples
    # broadening the bounded-shape downgrade validation past n=1 -- _003
    # stays in 'code' (high->medium, a different bug shape: silent/logical
    # failure rather than _001's loud IndexError), _004 extends to
    # 'conversational' (medium->low) with the classic bat-and-ball
    # cognitive-reflection-test question, chosen specifically because it's
    # famous for tricking reasoners into the fast wrong answer.
    # adaptive_budget_005/006/007 (2026-09-06): broadening real evidence
    # further -- _005 (tool-heavy) and _006 (extraction) confirm no HIDDEN
    # downside where bounded_shape doesn't change the effort at all (both
    # already map to "low"); _007 is a second conversational risk case.
    for sample_id, expected_effort in (
        ("quality_adaptive_budget_003", "medium"),
        ("quality_adaptive_budget_004", "low"),
        ("quality_adaptive_budget_005", "low"),
        ("quality_adaptive_budget_006", "low"),
        ("quality_adaptive_budget_007", "low"),
    ):
        sample = samples[sample_id]
        tk = Tokenetics(client=stub_client, stage_config=sample.stage_config)
        tk.prepare(**sample.kwargs)
        entry = next(e for e in tk.logger.entries if e.stage_name == "adaptive_budget")
        assert entry.extra["thinking_effort"] == expected_effort, sample_id

    # brevity_aggressive_001 (2026-09-06): needs stage_config to opt into
    # AGGRESSIVE brevity for task types where it isn't already the default
    # (extraction) -- without the flag those samples would silently fall
    # through to MODERATE and not test what they're meant to. code/
    # conversational/tool-heavy samples get AGGRESSIVE from the default
    # (added 2026-09-06, gated on meta.bounded_shape) even without the flag.
    from tokenetics.stages.brevity_injector import AGGRESSIVE_INSTRUCTION, MODERATE_INSTRUCTION

    for sample_id in (
        "quality_brevity_aggressive_001",
        "quality_brevity_aggressive_002",
        "quality_brevity_aggressive_003",
        "quality_brevity_aggressive_004",
        "quality_brevity_aggressive_005",
        "quality_brevity_aggressive_006",
        "quality_brevity_aggressive_007",
        "quality_brevity_aggressive_008",
        "quality_brevity_aggressive_009",
        "quality_brevity_aggressive_010",
    ):
        sample = samples[sample_id]
        tk = Tokenetics(client=stub_client, stage_config=sample.stage_config)
        prepared = tk.prepare(**sample.kwargs)
        assert prepared["system"] == AGGRESSIVE_INSTRUCTION, sample_id

    # _011 is the deliberately harder, multi-item extraction case -- its
    # prose is long enough (>40 words) that meta.bounded_shape comes out
    # False, so even with the explicit opt-in it falls back to MODERATE.
    # That's the gate working as intended, not a broken sample: extraction
    # was never validated past the narrow bounded shape, so it shouldn't
    # get AGGRESSIVE just because a caller opted the task type in.
    sample = samples["quality_brevity_aggressive_011"]
    tk = Tokenetics(client=stub_client, stage_config=sample.stage_config)
    prepared = tk.prepare(**sample.kwargs)
    assert prepared["system"] == MODERATE_INSTRUCTION


def test_the_real_quality_check_directory_parses():
    samples = load_quality_checks(_REPO_ROOT / "benchmarks" / "quality_checks")
    assert len(samples) >= 11
    stages_covered = {s.stage for s in samples}
    # All 5 risk-bearing concerns CLAUDE.md names (brevity, cap tuning,
    # thinking-effort tuning, pruning, delta compression), PLUS tool-relevance
    # filtering -- a real risk discovered via the corpus run's per-stage
    # trace (2026-08-02): schema_minification dropped an already-used tool
    # based only on lexical mismatch with the latest turn.
    assert stages_covered == {
        "brevity_injector",
        "adaptive_budget",
        "context_scheduler",
        "delta_compression",
        "schema_minification",
    }
    by_stage: dict[str, int] = {}
    for sample in samples:
        by_stage[sample.stage] = by_stage.get(sample.stage, 0) + 1
        # A sample needs at least one real grading signal -- either text
        # elements or a required tool call (schema_minification's sample
        # uses only the latter, since a correct response there is a
        # tool_use call with no meaningful text).
        assert sample.required_elements or sample.required_tool_calls
        assert sample.judge_rubric
    for stage, count in by_stage.items():
        assert count >= 2, f"{stage} has only {count} quality-check sample(s), below the 2+ floor"


def test_load_quality_checks_reads_required_elements_and_rubric(tmp_path):
    _write_corpus_file(
        tmp_path,
        "brevity.json",
        [
            {
                "id": "q1",
                "stage": "brevity_injector",
                "category": "extraction",
                "description": "d",
                "kwargs": {"model": "claude-sonnet-5", "max_tokens": 100, "messages": []},
                "required_elements": ["Jane", "A12345"],
                "judge_rubric": "rate 1-5",
            }
        ],
    )
    samples = load_quality_checks(tmp_path)
    assert samples[0].required_elements == ["Jane", "A12345"]
    assert samples[0].judge_rubric == "rate 1-5"
    assert samples[0].stage_config is None  # absent in the file -- not silently defaulted to {}


def test_load_quality_checks_reads_optional_stage_config(tmp_path):
    _write_corpus_file(
        tmp_path,
        "context_scheduler.json",
        [
            {
                "id": "q1",
                "stage": "context_scheduler",
                "category": "conversational",
                "description": "d",
                "kwargs": {"model": "claude-sonnet-5", "max_tokens": 100, "messages": []},
                "required_elements": ["x"],
                "judge_rubric": "rate 1-5",
                "stage_config": {"context_scheduler": {"token_budget": 60}},
            }
        ],
    )
    samples = load_quality_checks(tmp_path)
    assert samples[0].stage_config == {"context_scheduler": {"token_budget": 60}}


def test_estimate_cost_usd_scales_with_completions_per_sample(tmp_path):
    from tokenetics.core.benchmark import QualityCheckSample

    sample = QualityCheckSample(
        id="q1",
        stage="brevity_injector",
        category="extraction",
        description="d",
        kwargs={"model": "claude-sonnet-5", "max_tokens": 200, "messages": []},
        required_elements=["x"],
        judge_rubric="r",
    )
    cost_2x = estimate_cost_usd([sample], "claude-sonnet-5", completions_per_sample=2)
    cost_4x = estimate_cost_usd([sample], "claude-sonnet-5", completions_per_sample=4)
    assert cost_2x > 0
    assert cost_4x == cost_2x * 2


def test_estimate_cost_usd_empty_samples_is_zero():
    assert estimate_cost_usd([], "claude-sonnet-5") == 0.0


def test_aggregate_savings_by_category_reports_min_max_range():
    records = [
        SavingsRecord(category="code", sample_id="c1", before_tokens=100, after_tokens=80),
        SavingsRecord(category="code", sample_id="c2", before_tokens=100, after_tokens=50),
        SavingsRecord(
            category="conversational", sample_id="v1", before_tokens=200, after_tokens=190
        ),
    ]
    ranges = aggregate_savings_by_category(records)
    ranges_by_category = {r.category: r for r in ranges}

    assert ranges_by_category["code"].min_pct == 20.0
    assert ranges_by_category["code"].max_pct == 50.0
    assert ranges_by_category["code"].sample_count == 2

    assert ranges_by_category["conversational"].min_pct == 5.0
    assert ranges_by_category["conversational"].max_pct == 5.0
    assert ranges_by_category["conversational"].sample_count == 1


def test_aggregate_savings_by_category_empty_input_is_empty_output():
    assert aggregate_savings_by_category([]) == []


def test_savings_record_pct_saved_handles_zero_before_tokens():
    record = SavingsRecord(category="code", sample_id="c1", before_tokens=0, after_tokens=0)
    assert record.pct_saved == 0.0


def test_check_required_elements_all_present_passes():
    result = check_required_elements(
        "Name: Jane Smith, Email: jane@example.com, Order: A12345",
        ["Jane Smith", "jane@example.com", "A12345"],
    )
    assert result.passed is True
    assert result.missing == []


def test_check_required_elements_reports_missing_and_fails():
    result = check_required_elements(
        "Name: Jane Smith",
        ["Jane Smith", "jane@example.com", "A12345"],
    )
    assert result.passed is False
    assert result.missing == ["jane@example.com", "A12345"]


def test_check_required_elements_is_case_insensitive():
    result = check_required_elements("the fix raises an INDEXERROR", ["IndexError"])
    assert result.passed is True


def test_check_required_elements_or_group_passes_on_any_alternate_phrasing():
    result = check_required_elements(
        "it fails when unique has only 1 element",
        [["fewer than two", "only 1 element"], "unique"],
    )
    assert result.passed is True
    assert result.missing == []


def test_check_required_elements_or_group_fails_when_no_alternate_matches():
    result = check_required_elements(
        "the list is empty",
        [["fewer than two", "only 1 element"]],
    )
    assert result.passed is False
    assert result.missing == ["fewer than two OR only 1 element"]


class _FakeToolUseBlock:
    def __init__(self, name: str, input: dict | None = None) -> None:
        self.type = "tool_use"
        self.name = name
        self.input = input or {}


class _FakeTextBlock:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


def test_check_tool_calls_passes_when_the_tool_was_called():
    content = [_FakeTextBlock("checking now"), _FakeToolUseBlock("check_inventory")]
    result = check_tool_calls(content, ["check_inventory"])
    assert result.passed is True
    assert result.missing == []


def test_check_tool_calls_fails_and_reports_missing_when_not_called():
    content = [_FakeTextBlock("I don't have that information.")]
    result = check_tool_calls(content, ["check_inventory"])
    assert result.passed is False
    assert result.missing == ["check_inventory"]


def test_check_tool_calls_empty_requirement_always_passes():
    result = check_tool_calls([_FakeTextBlock("anything")], [])
    assert result.passed is True


def test_response_text_with_tool_inputs_includes_text_blocks():
    content = [_FakeTextBlock("the answer is 42")]
    assert response_text_with_tool_inputs(content) == "the answer is 42"


def test_response_text_with_tool_inputs_includes_tool_use_input():
    # The real bug this fixes: an extraction-shaped sample's real answer
    # lives in the tool_use block's input (structured_output forces
    # tool_choice), not in text -- a text-only check can never find it.
    content = [
        _FakeToolUseBlock(
            "extract_order_info",
            input={"name": "Jane Smith", "email": "jane@example.com", "order_id": "A12345"},
        )
    ]
    combined = response_text_with_tool_inputs(content)
    assert "Jane Smith" in combined
    assert "jane@example.com" in combined
    assert "A12345" in combined


def test_check_required_elements_finds_values_via_response_text_with_tool_inputs():
    content = [_FakeToolUseBlock("extract_order_info", input={"name": "Jane Smith"})]
    result = check_required_elements(response_text_with_tool_inputs(content), ["Jane Smith"])
    assert result.passed is True


def test_response_text_with_tool_inputs_combines_text_and_tool_blocks():
    content = [
        _FakeTextBlock("Here's what I found: "),
        _FakeToolUseBlock("extract_order_info", input={"order_id": "A12345"}),
    ]
    combined = response_text_with_tool_inputs(content)
    assert "Here's what I found" in combined
    assert "A12345" in combined


def test_load_quality_checks_reads_optional_required_tool_calls(tmp_path):
    _write_corpus_file(
        tmp_path,
        "schema_minification.json",
        [
            {
                "id": "q1",
                "stage": "schema_minification",
                "category": "tool-heavy",
                "description": "d",
                "kwargs": {"model": "claude-sonnet-5", "max_tokens": 100, "messages": []},
                "required_elements": [],
                "judge_rubric": "rate 1-5",
                "required_tool_calls": ["check_inventory"],
            }
        ],
    )
    samples = load_quality_checks(tmp_path)
    assert samples[0].required_tool_calls == ["check_inventory"]


def test_load_quality_checks_defaults_required_tool_calls_to_none(tmp_path):
    _write_corpus_file(
        tmp_path,
        "brevity.json",
        [
            {
                "id": "q1",
                "stage": "brevity_injector",
                "category": "extraction",
                "description": "d",
                "kwargs": {"model": "claude-sonnet-5", "max_tokens": 100, "messages": []},
                "required_elements": ["x"],
                "judge_rubric": "rate 1-5",
            }
        ],
    )
    samples = load_quality_checks(tmp_path)
    assert samples[0].required_tool_calls is None
