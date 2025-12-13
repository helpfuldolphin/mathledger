"""Policy drift tile adapter & First Light wiring."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence

_STATUS_TO_LIGHT = {
    "BLOCK": "RED",
    "WARN": "YELLOW",
    "OK": "GREEN",
    "UNKNOWN": "YELLOW",
}

_POLICY_STATUS_ORDER = {"OK": 0, "WARN": 1, "BLOCK": 2, "UNKNOWN": 1}
_NCI_STATUS_ORDER = {"OK": 0, "WARN": 1, "BREACH": 2, "UNKNOWN": 1}


def _normalize_status(status: Any) -> str:
    normalized = str(status or "UNKNOWN").upper()
    if normalized not in ("OK", "WARN", "BLOCK"):
        return "UNKNOWN"
    return normalized


def _map_status_to_light(status: str) -> str:
    return _STATUS_TO_LIGHT.get(_normalize_status(status), "YELLOW")


def _coerce_rule_count(value: Any) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = 0
    return max(number, 0)


def _count_entries(tile: Mapping[str, Any], keys: Sequence[str]) -> int:
    total = 0
    found = False
    for key in keys:
        value = tile.get(key)
        if isinstance(value, (list, tuple)):
            total += len(value)
            found = True
    return total if found else 0


def _infer_total_rules(tile: Mapping[str, Any]) -> int:
    if "num_rules" in tile:
        return _coerce_rule_count(tile.get("num_rules"))
    return _count_entries(
        tile,
        (
            "soft_changes",
            "soft_rules",
            "soft_drift",
            "soft_events",
            "hard_blocks",
            "breaking_changes",
            "blocking_changes",
        ),
    )


def _infer_blocked_rules(tile: Mapping[str, Any]) -> int:
    for key in ("num_blocking", "blocked_rules", "num_blocked", "blocking_rules"):
        if key in tile:
            return _coerce_rule_count(tile.get(key))
    return _count_entries(tile, ("hard_blocks", "breaking_changes", "blocking_changes"))


def _derive_headline(
    tile: Mapping[str, Any],
    status: str,
    num_rules: int,
    blocked_rules: int,
) -> str:
    headline = tile.get("headline")
    if isinstance(headline, str) and headline.strip():
        return headline.strip()
    if blocked_rules:
        return f"{blocked_rules} blocking policy rule change(s) detected."
    if num_rules:
        return f"{num_rules} policy change(s) under review (status {status})."
    return f"Policy drift status {status}; no tracked rule changes."


def build_policy_drift_summary(policy_tile: Mapping[str, Any]) -> Dict[str, Any]:
    """Build canonical policy drift summary used across reports and evidence."""
    status = _normalize_status(policy_tile.get("status"))
    num_rules = _infer_total_rules(policy_tile)
    blocked_rules = _infer_blocked_rules(policy_tile)
    headline = _derive_headline(policy_tile, status, num_rules, blocked_rules)
    schema_version = str(policy_tile.get("schema_version", "policy-drift-tile/1.0.0"))
    return {
        "schema_version": schema_version,
        "status": status,
        "status_light": _map_status_to_light(status),
        "num_rules": num_rules,
        "blocked_rules": blocked_rules,
        "headline": headline,
    }


def _resolve_summary(
    policy_tile: Mapping[str, Any],
    override: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    summary = build_policy_drift_summary(policy_tile)
    if not override:
        return summary
    result = dict(summary)
    if "status" in override:
        result["status"] = _normalize_status(override["status"])
        result["status_light"] = _map_status_to_light(result["status"])
    if "num_rules" in override:
        result["num_rules"] = _coerce_rule_count(override["num_rules"])
    if "blocked_rules" in override:
        result["blocked_rules"] = _coerce_rule_count(override["blocked_rules"])
    elif "num_blocking" in override:
        result["blocked_rules"] = _coerce_rule_count(override["num_blocking"])
    if "headline" in override and isinstance(override["headline"], str):
        result["headline"] = override["headline"]
    return result


def attach_policy_drift_tile(
    global_health: Mapping[str, Any] | None,
    tile: Mapping[str, Any],
) -> Dict[str, Any]:
    """Return a new global health dict that includes the policy drift tile."""
    summary = build_policy_drift_summary(tile)
    updated = dict(global_health or {})
    updated["policy_drift"] = dict(tile)
    signals = dict(updated.get("signals", {}))
    signals["policy_drift"] = summary
    updated["signals"] = signals
    return updated


def extract_policy_drift_signal_for_first_light(
    tile: Mapping[str, Any],
) -> Dict[str, Any]:
    """Downgrade summary to minimal signal for existing First Light hooks."""
    summary = build_policy_drift_summary(tile)
    return {
        "status": summary["status"],
        "num_rules": summary["num_rules"],
        "blocked_rules": summary["blocked_rules"],
    }


def attach_policy_drift_to_p3_stability_report(
    stability_report: Mapping[str, Any],
    policy_tile: Mapping[str, Any],
) -> Dict[str, Any]:
    """Attach policy drift summary to the P3 stability report (non-mutating)."""
    summary = build_policy_drift_summary(policy_tile)
    updated_report = dict(stability_report or {})
    updated_report["policy_drift_summary"] = summary
    return updated_report


def build_first_light_policy_drift_summary(
    policy_tile: Mapping[str, Any],
) -> Dict[str, Any]:
    """Backward-compatible helper that returns the canonical policy drift summary."""
    return build_policy_drift_summary(policy_tile)


def _build_policy_explanation(
    policy_tile: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> str:
    blocked_rules = summary.get("blocked_rules", 0)
    if blocked_rules:
        return (
            f"Policy drift {summary.get('status', 'UNKNOWN')} with "
            f"{blocked_rules} blocking rule change(s) across "
            f"{summary.get('num_rules', 0) or 'no'} tracked updates."
        )
    return summary.get("headline", _derive_headline(policy_tile, "UNKNOWN", 0, 0))


def attach_policy_drift_to_evidence(
    evidence: Mapping[str, Any],
    policy_tile: Mapping[str, Any],
    first_light_signal: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Attach policy drift governance block to evidence (non-mutating).

    Includes status, rule counts, status_light, and a neutral explanation suitable
    for sharing in First Light evidence packs.
    """
    summary = _resolve_summary(policy_tile, first_light_signal)
    block = {
        **summary,
        "explanation": _build_policy_explanation(policy_tile, summary),
        "first_light_summary": summary,
    }

    updated = dict(evidence or {})
    governance: MutableMapping[str, Any]
    if "governance" in updated:
        governance = dict(updated["governance"])
    else:
        governance = {}
    governance["policy_drift"] = block
    updated["governance"] = governance
    return updated


def summarize_policy_drift_vs_nci_consistency(
    policy_drift_summary: Mapping[str, Any],
    nci_signal: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Compare policy drift summary against NCI signal for observational consistency.
    """
    policy_status = _normalize_status(policy_drift_summary.get("status"))
    nci_status = str(
        nci_signal.get("health_contribution", {}).get("status", "UNKNOWN")
    ).upper()
    policy_score = _POLICY_STATUS_ORDER.get(policy_status, 1)
    nci_score = _NCI_STATUS_ORDER.get(nci_status, 1)
    delta = abs(policy_score - nci_score)
    if delta == 0:
        consistency = "CONSISTENT"
    elif delta == 1:
        consistency = "PARTIAL"
    else:
        consistency = "INCONSISTENT"

    global_nci = nci_signal.get("health_contribution", {}).get("global_nci")
    if isinstance(global_nci, (int, float)):
        nci_note = f"NCI signal status {nci_status} with global NCI {global_nci:.2f}."
    else:
        nci_note = f"NCI signal status {nci_status}."

    notes = [
        f"Policy drift status {policy_status} with "
        f"{policy_drift_summary.get('blocked_rules', 0)} blocking rule(s).",
        nci_note,
    ]
    return {"consistency": consistency, "notes": notes}


def policy_drift_vs_nci_for_alignment_view(
    signal: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """
    Convert policy drift vs NCI consistency signal to GGFL alignment view format.

    SHADOW MODE CONTRACT:
    - Observational only, no gating
    - Deterministic mapping for identical inputs
    - conflict is always False

    Args:
        signal: first_light_status.json signals["policy_drift_vs_nci"] block.

    Returns:
        Dict with:
        - signal_type: "SIG-PD"
        - status: "ok" | "warn"
        - conflict: False
        - drivers: List[str] (reason codes, max 3)
        - summary: str (one sentence)
    """
    if not signal:
        return {
            "signal_type": "SIG-PD",
            "status": "ok",
            "conflict": False,
            "drivers": [],
            "summary": "Policy drift vs NCI alignment signal not available.",
        }

    consistency = str(signal.get("consistency_status", "UNKNOWN") or "UNKNOWN").upper()
    policy_light = str(signal.get("policy_status_light", "YELLOW") or "YELLOW").upper()
    nci_light = str(signal.get("nci_status_light", "YELLOW") or "YELLOW").upper()

    status = "ok" if consistency == "CONSISTENT" else "warn"

    drivers: List[str] = []
    if status == "warn":
        drivers.append("DRIVER_STATUS_INCONSISTENT")
        if policy_light == "RED":
            drivers.append("DRIVER_POLICY_BLOCK")
        if nci_light == "RED":
            drivers.append("DRIVER_NCI_BREACH")

    drivers = drivers[:3]

    summary = (
        f"Policy drift and NCI are {consistency} "
        f"(policy {policy_light}, NCI {nci_light})."
    )
    return {
        "signal_type": "SIG-PD",
        "status": status,
        "conflict": False,
        "drivers": drivers,
        "summary": summary,
    }


__all__ = [
    "attach_policy_drift_tile",
    "attach_policy_drift_to_evidence",
    "attach_policy_drift_to_p3_stability_report",
    "build_policy_drift_summary",
    "build_first_light_policy_drift_summary",
    "extract_policy_drift_signal_for_first_light",
    "policy_drift_vs_nci_for_alignment_view",
    "summarize_policy_drift_vs_nci_consistency",
]
