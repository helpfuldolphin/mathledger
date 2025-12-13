"""Epistemic evidence adapter for evidence packs.

STATUS: PHASE X — EPISTEMIC GOVERNANCE INTEGRATION

Provides compact epistemic evidence tile for inclusion in First-Light / release evidence packs.

SHADOW MODE CONTRACT:
- All functions are read-only and side-effect free
- The evidence tile is purely observational
- It does NOT influence any other tiles or system health classification
- No control flow depends on this tile
"""

from typing import Any, Dict

EPISTEMIC_EVIDENCE_SCHEMA_VERSION = "1.0.0"


def extract_epistemic_evidence_for_pack(
    alignment_tensor: Dict[str, Any],
    misalignment_forecast: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compact epistemic evidence tile for inclusion in First-Light / release evidence packs.

    This must be read-only and neutral.

    Args:
        alignment_tensor: Output from build_epistemic_alignment_tensor()
        misalignment_forecast: Output from forecast_epistemic_misalignment()

    Returns:
        Compact evidence dictionary with:
        - tensor_norm: from alignment_tensor["alignment_tensor_norm"]
        - predicted_band: from misalignment_forecast["predicted_band"]
        - misalignment_hotspots: from alignment_tensor["misalignment_hotspots"]
        - confidence: from misalignment_forecast["confidence"]
        - schema_version: "1.0.0"

    All outputs are JSON-safe and deterministic.
    """
    # Extract tensor_norm from alignment tensor
    tensor_norm = alignment_tensor.get("alignment_tensor_norm", 0.0)

    # Extract predicted_band from misalignment forecast
    predicted_band = misalignment_forecast.get("predicted_band", "MEDIUM")

    # Extract misalignment_hotspots from alignment tensor
    misalignment_hotspots = alignment_tensor.get("misalignment_hotspots", [])

    # Extract confidence from misalignment forecast
    confidence = misalignment_forecast.get("confidence", 0.5)

    # Build compact evidence tile
    evidence: Dict[str, Any] = {
        "schema_version": EPISTEMIC_EVIDENCE_SCHEMA_VERSION,
        "tensor_norm": tensor_norm,
        "predicted_band": predicted_band,
        "misalignment_hotspots": misalignment_hotspots,
        "confidence": confidence,
    }

    return evidence


__all__ = [
    "EPISTEMIC_EVIDENCE_SCHEMA_VERSION",
    "extract_epistemic_evidence_for_pack",
]

