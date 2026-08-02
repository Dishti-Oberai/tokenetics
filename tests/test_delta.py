import json

import pytest

from tokenetics.core.delta import apply_delta, compute_delta, is_delta_wire_text


def test_localized_text_change_produces_a_smaller_delta():
    previous = "line one\nline two\nline three\n" * 50 + "the target line\n" + "line x\n" * 50
    new = previous.replace("the target line", "the CHANGED line")
    delta_text = compute_delta(previous, new)
    assert delta_text is not None
    assert len(delta_text) < len(new)
    assert is_delta_wire_text(delta_text)


def test_line_patch_round_trips_exactly():
    previous = "def add(a, b):\n    return a + b\n" + "# padding line\n" * 40
    new = "def add(a, b):\n    return a + b + 1\n" + "# padding line\n" * 40
    delta_text = compute_delta(previous, new)
    assert delta_text is not None
    assert apply_delta(previous, delta_text) == new


def test_no_change_produces_a_tiny_delta_and_round_trips():
    previous = "line one\nline two\n" * 200
    new = previous
    delta_text = compute_delta(previous, new)
    assert delta_text is not None
    assert len(delta_text) < len(new)
    assert apply_delta(previous, delta_text) == new


def test_json_patch_round_trips_semantically():
    previous = json.dumps({"status": "pending", "id": 42, "tags": ["a", "b"], "note": "x" * 200})
    new = json.dumps({"status": "done", "id": 42, "tags": ["a", "b"], "note": "x" * 200})
    delta_text = compute_delta(previous, new)
    assert delta_text is not None
    reconstructed = apply_delta(previous, delta_text)
    # JSON round-trip is semantically exact, not byte-exact -- formatting
    # (key order, whitespace) was never a real invariant for a data payload.
    assert json.loads(reconstructed) == json.loads(new)


def test_json_patch_handles_added_and_removed_keys():
    previous = json.dumps({"a": 1, "b": 2, "note": "y" * 300})
    new = json.dumps({"a": 1, "c": 3, "note": "y" * 300})
    delta_text = compute_delta(previous, new)
    assert delta_text is not None
    assert json.loads(apply_delta(previous, delta_text)) == json.loads(new)


def test_json_patch_handles_nested_structures():
    previous = json.dumps({"user": {"name": "Jane", "score": 10}, "pad": "z" * 300})
    new = json.dumps({"user": {"name": "Jane", "score": 11}, "pad": "z" * 300})
    delta_text = compute_delta(previous, new)
    assert delta_text is not None
    assert json.loads(apply_delta(previous, delta_text)) == json.loads(new)


def test_json_patch_escapes_path_parts_containing_slash():
    previous = json.dumps({"a/b": 1, "pad": "w" * 300})
    new = json.dumps({"a/b": 2, "pad": "w" * 300})
    delta_text = compute_delta(previous, new)
    assert delta_text is not None
    assert json.loads(apply_delta(previous, delta_text)) == json.loads(new)


def test_totally_different_content_returns_none_not_a_bloated_delta():
    previous = "short"
    new = "a completely different and much longer piece of text that shares nothing"
    assert compute_delta(previous, new) is None


def test_json_top_level_type_change_falls_back_to_replace_and_still_round_trips():
    previous = json.dumps({"data": "y" * 300})
    new = json.dumps(["different", "shape", "y" * 300])
    delta_text = compute_delta(previous, new)
    # Whole-value replace is a valid (if unremarkable) delta -- only assert
    # the round-trip contract, not that a delta was necessarily chosen.
    if delta_text is not None:
        assert json.loads(apply_delta(previous, delta_text)) == json.loads(new)


def test_apply_delta_rejects_text_without_the_marker():
    with pytest.raises(ValueError):
        apply_delta("previous", "not a real delta")
