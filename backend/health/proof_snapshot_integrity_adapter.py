from __future__ import annotations

from typing import Any, Dict, List, Optional

PROOF_SNAPSHOT_INTEGRITY_SIGNAL_TYPE = "SIG-PROOF"

PROOF_SNAPSHOT_INTEGRITY_FAILURE_PRIORITY: Dict[str, int] = {
    "MISSING_FILE": 0,
    "SHA256_MISMATCH": 1,
    "CANONICAL_HASH_MISMATCH": 2,
    "ENTRY_COUNT_MISMATCH": 3,
}


def proof_snapshot_integrity_for_alignment_view(
    integrity: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Convert proof_snapshot_integrity object into a GGFL alignment-view signal.

    Returns None when integrity is missing.

    Output contract (fixed shape):
    - signal_type: "SIG-PROOF"
    - status: "ok" | "warn"
    - conflict: False
    - weight_hint: "LOW"
    - drivers: <= 3 entries, canonical reason codes only (top failure code first)
    """
    if integrity is None or not isinstance(integrity, dict):
        return None

    ok = integrity.get("ok") is True
    failure_codes_raw = integrity.get("failure_codes")
    failure_codes = _sorted_failure_codes(failure_codes_raw)
    drivers = failure_codes[:3]
    status = "ok" if ok and not drivers else "warn"

    return {
        "signal_type": PROOF_SNAPSHOT_INTEGRITY_SIGNAL_TYPE,
        "status": status,
        "conflict": False,
        "weight_hint": "LOW",
        "drivers": drivers,
    }


def _sorted_failure_codes(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []

    canonical: List[str] = []
    for item in value:
        if isinstance(item, str) and item in PROOF_SNAPSHOT_INTEGRITY_FAILURE_PRIORITY:
            canonical.append(item)

    unique = sorted(
        set(canonical),
        key=lambda code: (PROOF_SNAPSHOT_INTEGRITY_FAILURE_PRIORITY[code], code),
    )
    return unique

