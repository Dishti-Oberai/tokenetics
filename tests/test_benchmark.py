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
    # 15 per category clears CLAUDE.md's stated 15-20+ floor -- at the low
    # end of that range, not padded past it for its own sake.
    assert len(samples) >= 63
    categories = {s.category for s in samples}
    assert categories == {"code", "conversational", "extraction", "tool-heavy"}
    ids = [s.id for s in samples]
    assert len(ids) == len(set(ids))  # no duplicate sample ids

    by_category: dict[str, int] = {}
    for sample in samples:
        by_category[sample.category] = by_category.get(sample.category, 0) + 1
    for category, count in by_category.items():
        assert count >= 15, f"{category} has only {count} samples, below the 15-20+ floor"


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


def test_the_real_quality_check_directory_parses():
    samples = load_quality_checks(_REPO_ROOT / "benchmarks" / "quality_checks")
    assert len(samples) >= 5
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
    for sample in samples:
        # A sample needs at least one real grading signal -- either text
        # elements or a required tool call (schema_minification's sample
        # uses only the latter, since a correct response there is a
        # tool_use call with no meaningful text).
        assert sample.required_elements or sample.required_tool_calls
        assert sample.judge_rubric


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


class _FakeToolUseBlock:
    def __init__(self, name: str) -> None:
        self.type = "tool_use"
        self.name = name


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
