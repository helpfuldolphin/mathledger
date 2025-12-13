from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Mapping, Optional


def _extract_violation_count(source: Mapping[str, Any]) -> int:
    value = source.get("violation_counts")
    if value is None:
        value = source.get("violation_count")
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    try:
        return int(value) if value is not None else 0
    except (TypeError, ValueError):
        return 0


def _build_summary(ledger_tile: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "status_light": ledger_tile.get("status_light", "UNKNOWN"),
        "violation_counts": _extract_violation_count(ledger_tile),
        "headline": ledger_tile.get("headline", "Ledger guard v2: status unavailable"),
    }


def attach_ledger_guard_to_p3_stability_report(
    stability_report: Mapping[str, Any] | None,
    ledger_tile: Mapping[str, Any],
) -> Dict[str, Any]:
    """Return a copy of the stability report with the ledger summary attached."""

    report_copy: Dict[str, Any] = deepcopy(stability_report) if stability_report is not None else {}
    report_copy["ledger_guard_summary"] = _build_summary(ledger_tile)
    return report_copy


def attach_ledger_guard_to_p4_calibration_report(
    calibration_report: Mapping[str, Any] | None,
    ledger_tile: Mapping[str, Any],
) -> Dict[str, Any]:
    """Return a copy of the calibration report with the ledger summary attached."""

    report_copy: Dict[str, Any] = deepcopy(calibration_report) if calibration_report is not None else {}
    report_copy["ledger_guard_summary"] = _build_summary(ledger_tile)
    return report_copy


def attach_ledger_guard_to_p4_calibration(
    calibration_report: Mapping[str, Any] | None,
    ledger_guard_summary: Mapping[str, Any],
) -> Dict[str, Any]:
    """Attach ledger guard summary under calibration_report["ledger_guard"]."""

    report_copy: Dict[str, Any] = deepcopy(calibration_report) if calibration_report is not None else {}
    report_copy["ledger_guard"] = _build_summary(ledger_guard_summary)
    return report_copy


def summarize_ledger_guard_for_evidence(
    ledger_guard_summary: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build the snapshot stored inside evidence.governance.ledger_guard.first_light_summary."""

    summary = _build_summary(ledger_guard_summary)
    return {
        "status_light": summary["status_light"],
        "violation_counts": summary["violation_counts"],
        "headline": summary["headline"],
    }


def attach_ledger_guard_to_evidence(
    evidence: Mapping[str, Any] | None,
    ledger_tile: Mapping[str, Any],
) -> Dict[str, Any]:
    """Attach the ledger guard summary under evidence["governance"]["ledger_guard"]."""

    evidence_copy: Dict[str, Any] = deepcopy(evidence) if evidence is not None else {}
    governance: Optional[Dict[str, Any]] = evidence_copy.get("governance")
    if governance is None:
        governance = {}
        evidence_copy["governance"] = governance

    existing_bucket = governance.get("ledger_guard") or {}
    ledger_guard_bucket = deepcopy(existing_bucket)
    ledger_guard_bucket["first_light_summary"] = summarize_ledger_guard_for_evidence(ledger_tile)
    governance["ledger_guard"] = ledger_guard_bucket
    return evidence_copy
