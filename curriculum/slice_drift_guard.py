"""
Slice drift guardrail utilities.

Provides deterministic snapshot comparisons for curriculum slices and
lightweight provenance events that the governance plane can ingest.

Runtime hook guidance
---------------------
RFLRunner (or any curriculum ratchet) should call
``compute_slice_drift_and_provenance`` immediately after evaluating the
active slice. The helper returns both the drift snapshot and an already
serialized provenance payload so the caller can persist the JSON alongside
gate verdicts without re-deriving the comparison. This guarantees every
activation records drift while staying deterministic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

from substrate.repro.determinism import deterministic_isoformat

Severity = Literal["NONE", "PARAMETRIC", "SEMANTIC"]
Status = Literal["OK", "WARN", "BLOCK"]


@dataclass(frozen=True)
class _DriftRule:
    path: Tuple[str, ...]
    constraint: Literal["increasing", "decreasing", "boolean_true", "any"]


def build_slice_drift_snapshot(
    baseline_slice: Dict[str, Any],
    current_slice: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compare params and gate thresholds for drift.

    Args:
        baseline_slice: Canonical slice specification.
        current_slice: Candidate slice specification.

    Returns:
        Dict with changed_params, severity, and status.
    """

    rules: List[_DriftRule] = [
        _DriftRule(("params", "atoms"), "increasing"),
        _DriftRule(("params", "depth_max"), "increasing"),
        _DriftRule(("params", "breadth_max"), "increasing"),
        _DriftRule(("params", "total_max"), "increasing"),
        _DriftRule(("gates", "coverage", "ci_lower_min"), "increasing"),
        _DriftRule(("gates", "coverage", "sample_min"), "increasing"),
        _DriftRule(("gates", "coverage", "require_attestation"), "boolean_true"),
        _DriftRule(("gates", "abstention", "max_rate_pct"), "decreasing"),
        _DriftRule(("gates", "abstention", "max_mass"), "decreasing"),
        _DriftRule(("gates", "velocity", "min_pph"), "increasing"),
        _DriftRule(("gates", "velocity", "stability_cv_max"), "decreasing"),
        _DriftRule(("gates", "velocity", "window_minutes"), "any"),
        _DriftRule(("gates", "caps", "min_attempt_mass"), "increasing"),
        _DriftRule(("gates", "caps", "min_runtime_minutes"), "increasing"),
        _DriftRule(("gates", "caps", "backlog_max"), "decreasing"),
    ]

    changes: List[Dict[str, Any]] = []
    highest: Severity = "NONE"
    for rule in rules:
        baseline_value = _get_nested_value(baseline_slice, rule.path)
        current_value = _get_nested_value(current_slice, rule.path)
        if _values_equal(baseline_value, current_value):
            continue

        classification: Severity = "PARAMETRIC"
        if _is_semantic_violation(rule, baseline_value, current_value):
            classification = "SEMANTIC"

        entry: Dict[str, Any] = {
            "path": ".".join(rule.path),
            "baseline": baseline_value,
            "current": current_value,
            "classification": classification,
        }
        delta = _safe_numeric_delta(baseline_value, current_value)
        if delta is not None:
            entry["delta"] = delta
        entry["constraint"] = rule.constraint
        changes.append(entry)
        highest = _max_severity(highest, classification)

    changes.sort(key=lambda entry: entry["path"])
    status = _status_for_severity(highest)
    return {
        "changed_params": changes,
        "severity": highest,
        "status": status,
    }


def build_curriculum_provenance_event(
    curriculum_fingerprint: str,
    slice_name: str,
    drift_snapshot: Dict[str, Any],
) -> str:
    """
    Serialize a deterministic curriculum provenance event.

    Args:
        curriculum_fingerprint: Stable fingerprint for the curriculum file.
        slice_name: Active slice name.
        drift_snapshot: Output of build_slice_drift_snapshot.

    Returns:
        JSON encoded event payload.
    """
    status = drift_snapshot.get("status", "OK")
    severity = drift_snapshot.get("severity", "NONE")
    changed = drift_snapshot.get("changed_params", [])
    payload = {
        "curriculum_fingerprint": curriculum_fingerprint,
        "slice_name": slice_name,
        "drift_status": status,
        "drift_severity": severity,
        "changed_params": changed,
        "emitted_at": deterministic_isoformat(
            curriculum_fingerprint,
            slice_name,
            status,
            severity,
        ),
    }
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def compute_slice_drift_and_provenance(
    baseline_slice: Dict[str, Any],
    current_slice: Dict[str, Any],
    curriculum_fingerprint: str,
) -> Tuple[Dict[str, Any], str]:
    """
    Produce drift snapshot + provenance payload for a single activation.

    Intended usage inside RFLRunner:

        snapshot, event = compute_slice_drift_and_provenance(
            baseline_slice=reference_slice,
            current_slice=active_slice,
            curriculum_fingerprint=curriculum_hash,
        )
        audit_log.append(json.loads(event))

    Args:
        baseline_slice: Sealed canonical slice definition.
        current_slice: Runtime slice definition.
        curriculum_fingerprint: Deterministic curriculum hash.

    Returns:
        (drift snapshot dict, serialized provenance JSON string).
    """
    snapshot = build_slice_drift_snapshot(baseline_slice, current_slice)
    event = build_curriculum_provenance_event(
        curriculum_fingerprint=curriculum_fingerprint,
        slice_name=current_slice.get("name", baseline_slice.get("name", "")),
        drift_snapshot=snapshot,
    )
    return snapshot, event


def summarize_slice_drift_for_global_health(
    events: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Aggregate drift events into a global health tile payload.

    Args:
        events: Sequence of provenance events (already decoded to dicts).

    Returns:
        Dict containing overall status, per-slice counters, and stressed slices.
    """
    slice_rollup: Dict[str, Dict[str, Any]] = {}
    warn_total = 0
    block_total = 0
    for event in events:
        slice_name = str(event.get("slice_name") or "unknown")
        status = str(event.get("drift_status", "UNKNOWN")).upper()
        entry = slice_rollup.setdefault(
            slice_name,
            {
                "warns": 0,
                "blocks": 0,
                "events": 0,
                "last_status": None,
                "last_emitted_at": None,
            },
        )
        entry["events"] += 1
        entry["last_status"] = status
        emitted_at = event.get("emitted_at")
        if emitted_at is not None:
            entry["last_emitted_at"] = emitted_at
        if status == "WARN":
            entry["warns"] += 1
            warn_total += 1
        elif status == "BLOCK":
            entry["blocks"] += 1
            block_total += 1

    stressed: List[Dict[str, Any]] = []
    for name, data in slice_rollup.items():
        if data["warns"] or data["blocks"]:
            stressed.append(
                {
                    "slice_name": name,
                    "warns": data["warns"],
                    "blocks": data["blocks"],
                    "last_status": data["last_status"],
                    "last_emitted_at": data["last_emitted_at"],
                }
            )
    stressed.sort(
        key=lambda item: (-item["blocks"], -item["warns"], item["slice_name"])
    )
    stressed = stressed[:3]

    overall: Status = "OK"
    if block_total:
        overall = "BLOCK"
    elif warn_total:
        overall = "WARN"

    return {
        "overall_status": overall,
        "warn_events": warn_total,
        "block_events": block_total,
        "slice_rollup": slice_rollup,
        "stressed_slices": stressed,
        "event_count": len(events),
    }


def _get_nested_value(root: Dict[str, Any], path: Tuple[str, ...]) -> Any:
    node: Any = root
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def _values_equal(a: Any, b: Any) -> bool:
    return a == b


def _is_semantic_violation(rule: _DriftRule, baseline: Any, current: Any) -> bool:
    if rule.constraint == "any":
        return False
    if baseline is None and current is None:
        return False
    if current is None and baseline is not None:
        return True
    if rule.constraint == "boolean_true":
        return bool(baseline) and not bool(current)
    if baseline is None or current is None:
        return False

    if rule.constraint == "increasing":
        return _compare_numbers(current, baseline, "<")
    if rule.constraint == "decreasing":
        return _compare_numbers(current, baseline, ">")
    return False


def _compare_numbers(current: Any, baseline: Any, op: Literal["<", ">"]) -> bool:
    if not (_is_number(current) and _is_number(baseline)):
        return False
    if op == "<":
        return float(current) < float(baseline)
    return float(current) > float(baseline)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _safe_numeric_delta(baseline: Any, current: Any) -> Optional[float]:
    if _is_number(baseline) and _is_number(current):
        return float(current) - float(baseline)
    return None


def _max_severity(left: Severity, right: Severity) -> Severity:
    order = {"NONE": 0, "PARAMETRIC": 1, "SEMANTIC": 2}
    return right if order[right] > order[left] else left


def _status_for_severity(severity: Severity) -> Status:
    if severity == "SEMANTIC":
        return "BLOCK"
    if severity == "PARAMETRIC":
        return "WARN"
    return "OK"
