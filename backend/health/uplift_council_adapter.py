"""
Adapter for attaching uplift council tile to global health.

This module provides the attach function that can be imported by global_surface.py
to avoid circular dependencies.
"""

from typing import Any, Dict, MutableMapping


def attach_uplift_council_tile(
    payload: MutableMapping[str, Any],
    council_tile: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Attach uplift council tile to an existing payload (SHADOW mode only).

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict mutation)
    - The uplift_council tile does NOT influence any other tiles
    - No control flow depends on the uplift_council tile contents
    - Provides multi-dimensional uplift readiness signal (budget + perf + metrics)

    Phase X Context:
    - P3 (synthetic): Budget is typically N/A or stable. Synthetic experiments
      don't enforce real budget constraints, so budget dimension may be omitted.
    - P4 (shadow): Budget can flag shadow experiments that are too costly or
      unstable. Budget health from real-runner telemetry feeds into council decisions.

    Args:
        payload: Existing global health payload
        council_tile: Council tile from summarize_uplift_council_for_global_console()

    Returns:
        Updated payload with uplift_council tile attached
    """
    payload = dict(payload)
    payload["uplift_council"] = dict(council_tile)
    return payload

