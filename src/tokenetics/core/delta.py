"""Structured delta compression: `compute_delta` / `apply_delta`.

Stdlib-only by design (`difflib`, `json`) -- no new core dependency, matching
CLAUDE.md's "base install stays lightweight" discipline. Two content shapes,
picked automatically:

- **JSON content** (both `previous` and `new` parse as a JSON object/array):
  a minimal, hand-rolled structural patch (add/remove/replace by JSON-pointer-
  style path) -- not full RFC 6902, just the operations this needs. Lists are
  diffed wholesale (any change inside a list replaces the whole list) rather
  than element-wise -- element-level list diffing has real ordering ambiguity
  that isn't worth the complexity for v1; a wholesale-replaced list just makes
  that particular delta less likely to clear the size threshold, which is a
  safe (if suboptimal) outcome, not a correctness risk.
- **Everything else**: a line-level patch via `difflib.SequenceMatcher`
  opcodes (equal/replace/insert/delete), applied by directly replaying those
  opcodes against `previous` -- not by re-parsing unified-diff hunk text,
  which would add real correctness risk (hunk-header edge cases) for no
  benefit here, since nothing outside this module ever needs to read a
  human-authored diff.

"Exact" round-trip means two different things depending on shape, and that's
deliberate, not an oversight: line-patch reconstruction is byte-exact (source
text, whitespace-sensitive); JSON-patch reconstruction is *semantically*
exact (`json.loads(apply_delta(...)) == json.loads(new)`), re-serialized via
`json.dumps(..., sort_keys=True)` for determinism -- tool-result payloads are
data, not formatted prose, so byte-exact JSON formatting (key order,
whitespace) was never a real invariant worth preserving.
"""

from __future__ import annotations

import difflib
import json
from dataclasses import dataclass
from typing import Any, Literal

# A computed delta must encode to no more than this fraction of `new`'s own
# length to be worth sending instead of the full payload -- below this, the
# encoding/marker overhead isn't paying for itself. A plain design tunable,
# not sourced/pricing data, so it lives here as a documented constant rather
# than in a versioned config table (same precedent as near_dup's similarity
# thresholds).
DELTA_SIZE_RATIO_THRESHOLD = 0.6

_MARKER = "[tokenetics:delta v1]"

DeltaFormat = Literal["line_patch", "json_patch"]


@dataclass
class _Delta:
    format: DeltaFormat
    ops: list[dict[str, Any]]

    def to_json(self) -> str:
        return json.dumps({"format": self.format, "ops": self.ops})

    @staticmethod
    def from_json(text: str) -> "_Delta":
        data = json.loads(text)
        fmt: DeltaFormat = data["format"]
        return _Delta(format=fmt, ops=data["ops"])


def _line_patch_ops(previous: str, new: str) -> list[dict[str, Any]]:
    prev_lines = previous.split("\n")
    new_lines = new.split("\n")
    matcher = difflib.SequenceMatcher(None, prev_lines, new_lines, autojunk=False)
    ops: list[dict[str, Any]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            ops.append({"op": "equal", "old_start": i1, "old_end": i2})
        else:
            ops.append(
                {
                    "op": tag,  # "replace" | "insert" | "delete"
                    "old_start": i1,
                    "old_end": i2,
                    "new_lines": new_lines[j1:j2],
                }
            )
    return ops


def _apply_line_patch(previous: str, ops: list[dict[str, Any]]) -> str:
    prev_lines = previous.split("\n")
    out: list[str] = []
    for op in ops:
        if op["op"] == "equal":
            out.extend(prev_lines[op["old_start"] : op["old_end"]])
        elif op["op"] == "delete":
            continue
        else:  # "replace" or "insert"
            out.extend(op["new_lines"])
    return "\n".join(out)


def _escape_path_part(part: str) -> str:
    return part.replace("~", "~0").replace("/", "~1")


def _unescape_path_part(part: str) -> str:
    return part.replace("~1", "/").replace("~0", "~")


def _json_patch_ops(previous: Any, new: Any) -> list[dict[str, Any]]:
    ops: list[dict[str, Any]] = []

    def walk(path: str, old: Any, new_: Any) -> None:
        if isinstance(old, dict) and isinstance(new_, dict):
            for key in old.keys() - new_.keys():
                ops.append({"op": "remove", "path": f"{path}/{_escape_path_part(str(key))}"})
            for key in new_.keys() - old.keys():
                ops.append(
                    {
                        "op": "add",
                        "path": f"{path}/{_escape_path_part(str(key))}",
                        "value": new_[key],
                    }
                )
            for key in old.keys() & new_.keys():
                walk(f"{path}/{_escape_path_part(str(key))}", old[key], new_[key])
        elif old != new_:
            ops.append({"op": "replace", "path": path, "value": new_})

    walk("", previous, new)
    return ops


def _apply_json_patch(previous: Any, ops: list[dict[str, Any]]) -> Any:
    import copy

    result = copy.deepcopy(previous)
    for op in ops:
        parts = [_unescape_path_part(p) for p in op["path"].split("/") if p]
        if op["op"] == "replace" and not parts:
            result = copy.deepcopy(op["value"])
            continue
        target = result
        for part in parts[:-1]:
            target = target[part]
        last = parts[-1]
        if op["op"] == "remove":
            del target[last]
        else:  # "add" or "replace"
            target[last] = op["value"]
    return result


def _try_parse_json(text: str) -> Any | None:
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    return value if isinstance(value, (dict, list)) else None


def compute_delta(previous: str, new: str) -> str | None:
    """Returns delta-encoded wire text if it's meaningfully smaller than
    `new`, else None -- callers (e.g. the delta_compression stage) should
    fall back to sending `new` in full when None comes back.
    """
    prev_json = _try_parse_json(previous)
    new_json = _try_parse_json(new)

    if prev_json is not None and new_json is not None:
        delta = _Delta(format="json_patch", ops=_json_patch_ops(prev_json, new_json))
    else:
        delta = _Delta(format="line_patch", ops=_line_patch_ops(previous, new))

    wire_text = f"{_MARKER}\n{delta.to_json()}"
    if len(wire_text) >= len(new) * DELTA_SIZE_RATIO_THRESHOLD:
        return None
    return wire_text


def is_delta_wire_text(text: str) -> bool:
    return text.startswith(_MARKER)


def apply_delta(previous: str, delta_text: str) -> str:
    """Reconstructs the payload `compute_delta(previous, new)` was computed
    from. `delta_text` must be exactly what `compute_delta` returned (the
    wire text a caller would see echoed back as a tool_result's content).
    """
    if not is_delta_wire_text(delta_text):
        raise ValueError("not a tokenetics delta-encoded payload (missing marker)")
    json_part = delta_text[len(_MARKER) :].lstrip("\n")
    delta = _Delta.from_json(json_part)

    if delta.format == "line_patch":
        return _apply_line_patch(previous, delta.ops)

    prev_json = json.loads(previous)
    result = _apply_json_patch(prev_json, delta.ops)
    return json.dumps(result, sort_keys=True)
