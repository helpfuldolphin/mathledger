"""Uplift governance tile builder.

Provides minimalist governance tile for uplift pipeline integration.
"""

from typing import Any, Dict

GOVERNANCE_TILE_SCHEMA_VERSION = "1.0.0"


def build_uplift_governance_tile(
    combined_eval: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build minimalist uplift governance tile from combined evaluation.

    This tile provides a compact summary for governance dashboards and
    uplift pipeline logging.

    Args:
        combined_eval: Combined evaluation from compose_abstention_with_uplift_decision

    Returns:
        {
            "schema_version": "1.0.0",
            "final_status": "OK" | "WARN" | "BLOCK",
            "epistemic_upgrade_applied": bool,
            "blocking_slices": List[str],
            "headline": str
        }
    """
    final_status = combined_eval.get("final_status", "OK")
    epistemic_upgrade = combined_eval.get("epistemic_upgrade_applied", False)
    blocking_slices = combined_eval.get("blocking_slices", [])
    reasons = combined_eval.get("reasons", [])

    # Generate neutral headline
    if final_status == "BLOCK":
        headline = (
            f"Uplift governance: BLOCKED. "
            f"{len(blocking_slices)} slice(s) with blocking conditions."
        )
    elif final_status == "WARN":
        headline = (
            f"Uplift governance: WARN. "
            f"Review recommended for {len(blocking_slices)} slice(s)."
        )
    else:
        headline = "Uplift governance: OK. No blocking conditions detected."

    if epistemic_upgrade:
        headline += " Epistemic gate applied."

    return {
        "schema_version": GOVERNANCE_TILE_SCHEMA_VERSION,
        "final_status": final_status,
        "epistemic_upgrade_applied": epistemic_upgrade,
        "blocking_slices": blocking_slices,
        "headline": headline,
    }

