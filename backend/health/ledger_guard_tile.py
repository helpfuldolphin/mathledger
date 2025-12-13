from __future__ import annotations

from typing import Any, Dict, Iterable, List

_SCHEMA_VERSION = "1.0.0"
_STATUS_GREEN = "GREEN"
_STATUS_YELLOW = "YELLOW"
_STATUS_RED = "RED"


def _classify_violation(message: str) -> str:
    lower_msg = message.lower()
    if "critical" in lower_msg:
        return "critical"
    if "does not match" in lower_msg and "prev_hash" in lower_msg:
        return "link"
    if "not greater" in lower_msg and "height" in lower_msg:
        return "link"
    if "regression" in lower_msg and "height" in lower_msg:
        return "link"
    if "missing field" in lower_msg or "field '" in lower_msg:
        return "schema"
    return "schema"


def _status_from_classes(classes: Iterable[str]) -> str:
    classes = list(classes)
    if not classes:
        return _STATUS_GREEN
    if any(cls in {"link", "critical"} for cls in classes):
        return _STATUS_RED
    return _STATUS_YELLOW


def _headline(status: str, violation_count: int) -> str:
    if violation_count == 0:
        return "Ledger guard v2: monotone chain confirmed"
    if status == _STATUS_RED:
        return f"Ledger guard v2: {violation_count} monotonicity violation(s) detected"
    return f"Ledger guard v2: {violation_count} schema issue(s) detected"


def build_ledger_guard_tile(check_result: Dict[str, Any]) -> Dict[str, Any]:
    """Build a JSON-safe advisory tile for the ledger monotone guard output."""

    violations: List[str] = list(check_result.get("violations", []))
    classes = [_classify_violation(message) for message in violations]
    status = _status_from_classes(classes)
    violation_count = len(violations)

    return {
        "schema_version": _SCHEMA_VERSION,
        "status_light": status,
        "violation_count": violation_count,
        "headline": _headline(status, violation_count),
    }