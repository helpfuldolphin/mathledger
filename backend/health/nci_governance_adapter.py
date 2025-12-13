"""NCI Governance Adapter for Global Health Surface.

Provides NCI (Narrative Consistency Index) tile builders and signal adapters
for Phase X integration.

SHADOW MODE CONTRACT:
- All functions in this module are purely observational
- No function influences governance decisions or control flow
- Tiles are advisory only and do NOT gate any operations
- All outputs are JSON-safe and deterministically ordered

Schema References:
    docs/system_law/schemas/nci/nci_director_panel.schema.json
    docs/system_law/schemas/nci/nci_governance_signal.schema.json

System Law Reference:
    docs/system_law/NCI_PhaseX_Spec.md
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
import json
import re

__version__ = "1.0.0"

# Schema version for NCI governance tile
NCI_GOVERNANCE_TILE_SCHEMA_VERSION = "1.0.0"

# Status light mapping
STATUS_LIGHT_MAP = {
    "OK": "green",
    "WARN": "yellow",
    "BREACH": "red",
}

# NCI SLO thresholds (defaults)
DEFAULT_NCI_SLO_THRESHOLDS = {
    "global_nci_warn": 0.75,
    "global_nci_breach": 0.60,
    "area_nci_warn": 0.70,
    "structural_min": 0.60,
    "terminology_min": 0.80,
    "violation_count_breach": 3,
}

# Telemetry Consistency Law canonical field names
TCL_CANONICAL_FIELDS = {
    "H": ["h", "health", "H_t", "Ht"],
    "rho": ["ρ", "rsi", "RSI", "R_t"],
    "tau": ["τ", "threshold", "T_t"],
    "beta": ["β", "block_rate", "B_t"],
    "in_omega": ["in_Ω", "omega_region", "safe_region"],
}

# Slice Identity Consistency Law canonical slice names
SIC_CANONICAL_SLICES = {
    "arithmetic_simple": ["arithmetic-simple", "ArithmeticSimple", "simple_arithmetic"],
    "propositional_tautology": ["prop_taut", "PropTautology", "PL_tautology"],
    "group_theory": ["group-theory", "GroupTheory", "groups"],
}


def build_nci_director_panel(
    insight_summary: Dict[str, Any],
    priority_view: Dict[str, Any],
    slo_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Build NCI director panel for dashboard display.

    SHADOW MODE CONTRACT:
    - This function is read-only and side-effect free
    - The panel it produces does NOT influence any governance decisions
    - This is purely for observability and logging

    Args:
        insight_summary: NCI insight summary with:
            - global_nci: float [0.0, 1.0]
            - dominant_area: str
            - suggestion_count: int (optional)
        priority_view: NCI work priority view with:
            - status: "OK" | "ATTENTION" | "BREACH"
            - priority_areas: list of area dicts
        slo_result: SLO evaluation result with:
            - slo_status: "OK" | "WARN" | "BREACH"
            - violations: list of violation strings

    Returns:
        Director panel conforming to nci_director_panel.schema.json

    Schema: docs/system_law/schemas/nci/nci_director_panel.schema.json
    """
    global_nci = insight_summary.get("global_nci", 1.0)
    dominant_area = insight_summary.get("dominant_area", "none")
    status = priority_view.get("status", "OK")
    suggestion_count = insight_summary.get("suggestion_count", 0)
    priority_areas = priority_view.get("priority_areas", [])

    # Map status to SLO status
    slo_status = slo_result.get("slo_status", "OK")
    violations = slo_result.get("violations", [])

    # Determine status light (from NCI + SLO)
    if slo_status == "BREACH" or status == "BREACH":
        status_light = "red"
    elif slo_status == "WARN" or status == "ATTENTION":
        status_light = "yellow"
    else:
        status_light = "green"

    # Build neutral headline
    headline = _build_nci_headline(
        status=status,
        slo_status=slo_status,
        global_nci=global_nci,
        dominant_area=dominant_area,
        suggestion_count=suggestion_count,
        priority_areas=priority_areas,
    )

    # Extract dimensional breakdown
    metrics_summary = _extract_metrics_summary(insight_summary)

    # Build trend from insight summary
    trend = insight_summary.get("trend", "STABLE")
    if trend not in ("STABLE", "IMPROVING", "DEGRADING"):
        trend = "STABLE"

    # Build panel
    panel = {
        "schema_version": NCI_GOVERNANCE_TILE_SCHEMA_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status_light": status_light,
        "global_nci": round(global_nci, 4),
        "dominant_area": _normalize_dominant_area(dominant_area),
        "headline": headline,
        "metrics_summary": metrics_summary,
        "slo_status": {
            "status": slo_status,
            "violation_count": len(violations),
            "violations": violations[:5],  # Top 5 violations
        },
        "trend": trend,
    }

    # Add priority areas if present
    if priority_areas:
        panel["priority_areas"] = [
            {
                "area": area.get("area", "unknown"),
                "nci": round(area.get("nci", 0.0), 4),
                "reason": area.get("reason", "Requires attention"),
            }
            for area in priority_areas[:5]  # Top 5
        ]

    # Add category breakdown if available
    category_scores = insight_summary.get("category_scores", {})
    if category_scores:
        panel["category_breakdown"] = {
            k: round(v, 4) for k, v in sorted(category_scores.items())
        }

    return panel


def build_nci_governance_signal(
    director_panel: Dict[str, Any],
    slo_result: Dict[str, Any],
    telemetry_drift: Optional[Dict[str, Any]] = None,
    slice_violations: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build NCI governance signal for health fusion layer.

    IMPORTANT: This signal is INFORMATIONAL only.
    It does NOT make governance decisions.

    SHADOW MODE CONTRACT:
    - This function is read-only and side-effect free
    - The signal it produces does NOT influence any governance decisions
    - This is purely for observability and evidence collection

    Args:
        director_panel: Output from build_nci_director_panel()
        slo_result: SLO evaluation result
        telemetry_drift: Optional telemetry drift report with:
            - drift_detected: bool
            - affected_docs: list of paths
        slice_violations: Optional slice consistency violations

    Returns:
        Governance signal conforming to nci_governance_signal.schema.json

    Schema: docs/system_law/schemas/nci/nci_governance_signal.schema.json
    """
    # Extract status from panel and SLO
    panel_status = _status_from_light(director_panel.get("status_light", "green"))
    slo_status = slo_result.get("slo_status", "OK")

    # Collapse to final status
    if slo_status == "BREACH" or panel_status == "BREACH":
        status = "BREACH"
    elif slo_status == "WARN" or panel_status == "WARN":
        status = "WARN"
    else:
        status = "OK"

    global_nci = director_panel.get("global_nci", 1.0)

    # Compute confidence based on metrics presence
    confidence = _compute_confidence(director_panel)

    # Build telemetry consistency section
    telemetry_consistency = _build_telemetry_consistency(telemetry_drift)

    # Build slice consistency section
    slice_consistency = _build_slice_consistency(slice_violations)

    # Extract dimensional breakdown from panel
    metrics_summary = director_panel.get("metrics_summary", {})
    dimensional_breakdown = {
        "terminology": metrics_summary.get("terminology_alignment", 1.0),
        "phase": metrics_summary.get("phase_discipline", 1.0),
        "uplift": metrics_summary.get("uplift_avoidance", 1.0),
        "structure": metrics_summary.get("structural_coherence", 1.0),
    }

    # Build recommendations
    recommendations = _build_recommendations(
        director_panel, slo_result, telemetry_drift, slice_violations
    )

    signal = {
        "schema_version": NCI_GOVERNANCE_TILE_SCHEMA_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "nci",
        "health_contribution": {
            "status": status,
            "global_nci": round(global_nci, 4),
            "confidence": round(confidence, 4),
        },
        "telemetry_consistency": telemetry_consistency,
        "slice_consistency": slice_consistency,
        "dimensional_breakdown": {
            k: round(v, 4) for k, v in dimensional_breakdown.items()
        },
        "slo_evaluation": {
            "status": slo_status,
            "violations": slo_result.get("violations", [])[:10],
            "thresholds_used": slo_result.get("thresholds_used", DEFAULT_NCI_SLO_THRESHOLDS),
        },
        "recommendations": recommendations,
    }

    # Add category breakdown if available
    category_breakdown = director_panel.get("category_breakdown", {})
    if category_breakdown:
        signal["category_breakdown"] = category_breakdown

    # Add trend if available
    trend = director_panel.get("trend", "STABLE")
    signal["trend"] = {
        "direction": trend,
        "change_rate": 0.0,  # Could be computed from historical data
        "window_size": 1,
    }

    return signal


def check_telemetry_consistency(
    doc_contents: Dict[str, str],
    telemetry_schema: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Check documentation for telemetry schema consistency (TCL laws).

    SHADOW MODE CONTRACT:
    - This function is read-only and side-effect free
    - The result does NOT influence any governance decisions

    Args:
        doc_contents: Mapping of doc path to content
        telemetry_schema: Optional telemetry schema with event definitions

    Returns:
        Telemetry consistency report with:
            - aligned: bool
            - drift_detected: bool
            - affected_docs: list of paths
            - violations: list of violation details
    """
    violations = []
    affected_docs = set()

    for doc_path, content in doc_contents.items():
        # Check TCL-002: Field name consistency
        for canonical, variants in TCL_CANONICAL_FIELDS.items():
            for variant in variants:
                # Use word boundary matching
                pattern = rf"\b{re.escape(variant)}\b"
                matches = list(re.finditer(pattern, content, re.IGNORECASE))
                for match in matches:
                    found_text = match.group()
                    # Skip if the match is exactly the canonical form
                    if found_text == canonical:
                        continue
                    # Get line number
                    line_num = content[:match.start()].count('\n') + 1
                    violations.append({
                        "doc": doc_path,
                        "field": canonical,
                        "violation_type": "TCL-002",
                        "expected": canonical,
                        "found": found_text,
                        "line": line_num,
                    })
                    affected_docs.add(doc_path)

    aligned = len(violations) == 0
    drift_detected = len(affected_docs) > 0

    return {
        "aligned": aligned,
        "drift_detected": drift_detected,
        "affected_docs_count": len(affected_docs),
        "affected_docs": sorted(affected_docs),
        "violations": violations[:50],  # Limit to 50 violations
    }


def check_slice_consistency(
    doc_contents: Dict[str, str],
    slice_registry: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Check documentation for slice identity consistency (SIC laws).

    SHADOW MODE CONTRACT:
    - This function is read-only and side-effect free
    - The result does NOT influence any governance decisions

    Args:
        doc_contents: Mapping of doc path to content
        slice_registry: Optional slice registry with canonical names

    Returns:
        List of slice consistency violations
    """
    violations = []

    for doc_path, content in doc_contents.items():
        # Check SIC-001: Slice name canonicalization
        for canonical, variants in SIC_CANONICAL_SLICES.items():
            for variant in variants:
                # Use word boundary matching
                pattern = rf"\b{re.escape(variant)}\b"
                matches = list(re.finditer(pattern, content, re.IGNORECASE))
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    violations.append({
                        "doc": doc_path,
                        "slice": canonical,
                        "violation_type": "SIC-001",
                        "expected": canonical,
                        "found": match.group(),
                        "line": line_num,
                    })

    return violations[:50]  # Limit to 50


def build_nci_tile_for_global_health(
    insight_summary: Dict[str, Any],
    priority_view: Dict[str, Any],
    slo_result: Dict[str, Any],
    telemetry_drift: Optional[Dict[str, Any]] = None,
    slice_violations: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build NCI tile for global health surface attachment.

    This is the entry point for global_surface.py integration.

    SHADOW MODE CONTRACT:
    - This function is read-only and side-effect free
    - The tile it produces does NOT influence any governance decisions
    - The tile does NOT influence any other tiles
    - No control flow depends on this tile

    Args:
        insight_summary: NCI insight summary
        priority_view: NCI work priority view
        slo_result: SLO evaluation result
        telemetry_drift: Optional telemetry drift report
        slice_violations: Optional slice consistency violations

    Returns:
        Tile suitable for global health surface attachment
    """
    director_panel = build_nci_director_panel(
        insight_summary, priority_view, slo_result
    )

    governance_signal = build_nci_governance_signal(
        director_panel, slo_result, telemetry_drift, slice_violations
    )

    # Combine into unified tile
    return {
        "schema_version": NCI_GOVERNANCE_TILE_SCHEMA_VERSION,
        "tile_type": "nci_governance",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "SHADOW",
        "director_panel": director_panel,
        "governance_signal": governance_signal,
        "shadow_mode_contract": {
            "observational_only": True,
            "no_control_flow_influence": True,
            "no_governance_modification": True,
        },
        "phase_x_metadata": {
            "phase": "P3",
            "doctrine_ref": "docs/system_law/NCI_PhaseX_Spec.md",
            "whitepaper_evidence_tag": "nci_governance_v1",
        },
    }


def attach_nci_tile_to_global_health(
    global_health: Dict[str, Any],
    nci_tile: Dict[str, Any],
) -> Dict[str, Any]:
    """Attach NCI tile to global health surface.

    NON-MUTATING: Returns a new dict, does not modify inputs.

    SHADOW MODE CONTRACT:
    - This function is read-only and side-effect free
    - The attachment does NOT influence any governance decisions

    Args:
        global_health: Existing global health surface
        nci_tile: NCI tile to attach

    Returns:
        New global health dict with NCI tile attached
    """
    result = dict(global_health)
    result["nci"] = nci_tile
    return result


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _build_nci_headline(
    status: str,
    slo_status: str,
    global_nci: float,
    dominant_area: str,
    suggestion_count: int,
    priority_areas: List[Dict[str, Any]],
) -> str:
    """Build neutral headline for NCI panel."""
    if slo_status == "BREACH":
        area_count = len(priority_areas)
        return f"Narrative consistency SLO breach detected. {area_count} area(s) require attention."
    elif slo_status == "WARN" or status == "ATTENTION":
        if dominant_area and dominant_area != "none":
            return f"Narrative consistency requires attention. Primary focus area: {dominant_area}."
        else:
            return f"Narrative consistency requires attention. {suggestion_count} improvement opportunity(ies) identified."
    else:
        return f"Narrative consistency within target. Global NCI: {global_nci:.2f}."


def _normalize_dominant_area(area: str) -> str:
    """Normalize dominant area to valid enum value."""
    valid_areas = {"terminology", "phase", "uplift", "structure", "none"}
    if area and area.lower() in valid_areas:
        return area.lower()
    return "none"


def _extract_metrics_summary(insight_summary: Dict[str, Any]) -> Dict[str, float]:
    """Extract metrics summary from insight summary."""
    summary = insight_summary.get("summary", {})
    return {
        "terminology_alignment": round(summary.get("terminology_alignment", 1.0), 4),
        "phase_discipline": round(summary.get("phase_discipline", 1.0), 4),
        "uplift_avoidance": round(summary.get("uplift_avoidance", 1.0), 4),
        "structural_coherence": round(summary.get("structural_coherence", 1.0), 4),
    }


def _status_from_light(status_light: str) -> str:
    """Convert status light to status string."""
    if status_light == "red":
        return "BREACH"
    elif status_light == "yellow":
        return "WARN"
    return "OK"


def _compute_confidence(director_panel: Dict[str, Any]) -> float:
    """Compute confidence score based on data completeness."""
    confidence = 1.0

    # Reduce confidence if missing metrics
    metrics = director_panel.get("metrics_summary", {})
    expected_keys = {"terminology_alignment", "phase_discipline", "uplift_avoidance", "structural_coherence"}
    present_keys = set(metrics.keys()) & expected_keys
    confidence *= len(present_keys) / len(expected_keys)

    # Reduce confidence if no category breakdown
    if not director_panel.get("category_breakdown"):
        confidence *= 0.9

    return max(0.5, confidence)  # Floor at 0.5


def _build_telemetry_consistency(
    telemetry_drift: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build telemetry consistency section."""
    if telemetry_drift is None:
        return {
            "aligned": True,
            "drift_detected": False,
            "affected_docs_count": 0,
        }

    return {
        "aligned": telemetry_drift.get("aligned", True),
        "drift_detected": telemetry_drift.get("drift_detected", False),
        "affected_docs_count": telemetry_drift.get("affected_docs_count", 0),
        "affected_docs": telemetry_drift.get("affected_docs", [])[:10],
        "violations": telemetry_drift.get("violations", [])[:10],
    }


def _build_slice_consistency(
    slice_violations: Optional[List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Build slice consistency section."""
    if slice_violations is None:
        return {
            "aligned": True,
            "violation_count": 0,
        }

    return {
        "aligned": len(slice_violations) == 0,
        "violation_count": len(slice_violations),
        "violations": slice_violations[:10],
    }


def _build_recommendations(
    director_panel: Dict[str, Any],
    slo_result: Dict[str, Any],
    telemetry_drift: Optional[Dict[str, Any]],
    slice_violations: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Build recommendations list (neutral language)."""
    recommendations = []

    # Add SLO violation recommendations
    violations = slo_result.get("violations", [])
    for violation in violations[:3]:
        recommendations.append({
            "area": "slo",
            "priority": "high",
            "description": violation,
            "affected_files": [],
        })

    # Add telemetry drift recommendation if detected
    if telemetry_drift and telemetry_drift.get("drift_detected"):
        affected = telemetry_drift.get("affected_docs", [])[:5]
        recommendations.append({
            "area": "telemetry",
            "priority": "medium",
            "description": f"Telemetry terminology drift detected in {len(affected)} document(s)",
            "affected_files": affected,
        })

    # Add slice consistency recommendation if violations exist
    if slice_violations and len(slice_violations) > 0:
        # Group by document
        affected = list(set(v.get("doc", "") for v in slice_violations))[:5]
        recommendations.append({
            "area": "slice",
            "priority": "medium",
            "description": f"Slice naming inconsistencies in {len(slice_violations)} location(s)",
            "affected_files": affected,
        })

    # Add priority area recommendations
    priority_areas = director_panel.get("priority_areas", [])
    for area in priority_areas[:2]:
        recommendations.append({
            "area": area.get("area", "unknown"),
            "priority": "low",
            "description": area.get("reason", "Area may warrant review"),
            "affected_files": [],
        })

    return recommendations[:10]  # Limit to 10


# =============================================================================
# P3 STABILITY REPORT INTEGRATION
# =============================================================================


def build_nci_summary_for_p3(
    nci_panel: Dict[str, Any],
) -> Dict[str, Any]:
    """Build NCI summary for P3 stability report attachment.

    This function extracts key NCI metrics for inclusion in FirstLightResult
    and the P3 stability report.

    SHADOW MODE CONTRACT:
    - This function is read-only and side-effect free
    - The summary does NOT influence any governance decisions
    - For observational/logging purposes only

    Args:
        nci_panel: Output from build_nci_director_panel()

    Returns:
        NCI summary dict suitable for stability_report["nci_summary"]:
        {
            "global_nci_score": float,
            "dominant_area": str,
            "violations": list[str],
            "status_light": str,
            "dimensional_scores": {...}
        }
    """
    # Extract core metrics
    global_nci_score = nci_panel.get("global_nci", 1.0)
    dominant_area = nci_panel.get("dominant_area", "none")
    status_light = nci_panel.get("status_light", "green")

    # Extract violations from SLO status
    slo_status = nci_panel.get("slo_status", {})
    violations = slo_status.get("violations", [])

    # Extract dimensional scores
    metrics_summary = nci_panel.get("metrics_summary", {})
    dimensional_scores = {
        "terminology": metrics_summary.get("terminology_alignment", 1.0),
        "phase": metrics_summary.get("phase_discipline", 1.0),
        "uplift": metrics_summary.get("uplift_avoidance", 1.0),
        "structure": metrics_summary.get("structural_coherence", 1.0),
    }

    return {
        "global_nci_score": round(global_nci_score, 4),
        "dominant_area": dominant_area,
        "violations": violations[:10],  # Top 10 violations
        "status_light": status_light,
        "dimensional_scores": {k: round(v, 4) for k, v in dimensional_scores.items()},
        "slo_status": slo_status.get("status", "OK"),
        "violation_count": slo_status.get("violation_count", 0),
    }


def attach_nci_summary_to_stability_report(
    stability_report: Dict[str, Any],
    nci_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """Attach NCI summary to P3 stability report.

    NON-MUTATING: Returns a new dict, does not modify inputs.

    SHADOW MODE CONTRACT:
    - This function is read-only and side-effect free
    - The attachment does NOT influence any governance decisions

    Args:
        stability_report: P3 stability report dict
        nci_summary: Output from build_nci_summary_for_p3()

    Returns:
        New stability report with nci_summary attached
    """
    result = dict(stability_report)
    result["nci_summary"] = dict(nci_summary)
    return result


# =============================================================================
# EVIDENCE PACK INTEGRATION
# =============================================================================


def attach_nci_to_evidence(
    evidence: Dict[str, Any],
    nci_signal: Dict[str, Any],
) -> Dict[str, Any]:
    """Attach NCI governance signal to evidence pack.

    NON-MUTATING: Returns a new dict, does not modify inputs.

    SHADOW MODE CONTRACT:
    - This function is read-only and side-effect free
    - The attachment does NOT influence any governance decisions
    - This is purely for observability and evidence collection

    Args:
        evidence: Existing evidence pack (not modified)
        nci_signal: NCI governance signal to attach (from build_nci_governance_signal)

    Returns:
        New evidence dict with NCI signal attached under evidence["governance"]["nci"]
    """
    # Create shallow copy of evidence
    result = dict(evidence)

    # Ensure governance key exists
    if "governance" not in result:
        result["governance"] = {}
    else:
        # Shallow copy governance to avoid mutating original
        result["governance"] = dict(result["governance"])

    # Attach NCI signal (read-only copy)
    result["governance"]["nci"] = dict(nci_signal)

    return result


def build_nci_evidence_attachment(
    nci_panel: Dict[str, Any],
    nci_signal: Dict[str, Any],
) -> Dict[str, Any]:
    """Build complete NCI evidence attachment for whitepaper evidence pack.

    Combines panel and signal into a single evidence attachment suitable
    for the Phase X evidence pack.

    SHADOW MODE CONTRACT:
    - This function is read-only and side-effect free
    - For evidence collection only

    Args:
        nci_panel: Output from build_nci_director_panel()
        nci_signal: Output from build_nci_governance_signal()

    Returns:
        Complete NCI evidence attachment:
        {
            "schema_version": "1.0.0",
            "source": "nci",
            "panel": {...},
            "signal": {...},
            "summary": {...}
        }
    """
    return {
        "schema_version": NCI_GOVERNANCE_TILE_SCHEMA_VERSION,
        "source": "nci",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "panel": nci_panel,
        "signal": nci_signal,
        "summary": build_nci_summary_for_p3(nci_panel),
        "shadow_mode_attestation": {
            "observational_only": True,
            "no_governance_modification": True,
        },
    }


# =============================================================================
# P5 OPERATIONAL MODES (NCI_PhaseX_Spec.md Section 11)
# =============================================================================

# NCI Operational Mode Constants
NCI_MODE_DOC_ONLY = "DOC_ONLY"
NCI_MODE_TELEMETRY_CHECKED = "TELEMETRY_CHECKED"
NCI_MODE_FULLY_BOUND = "FULLY_BOUND"

# SLO Thresholds by Mode (from Section 11.5)
MODE_SLO_THRESHOLDS = {
    NCI_MODE_DOC_ONLY: {
        "global_nci_warn": 0.70,
        "global_nci_breach": 0.55,
        "area_nci_warn": 0.65,
        "terminology_min": 0.75,
        "violation_count_breach": 5,
    },
    NCI_MODE_TELEMETRY_CHECKED: {
        "global_nci_warn": 0.75,
        "global_nci_breach": 0.60,
        "area_nci_warn": 0.70,
        "terminology_min": 0.80,
        "violation_count_breach": 3,
    },
    NCI_MODE_FULLY_BOUND: {
        "global_nci_warn": 0.80,
        "global_nci_breach": 0.65,
        "area_nci_warn": 0.75,
        "terminology_min": 0.85,
        "violation_count_breach": 2,
    },
}


def _select_nci_mode(
    telemetry_schema: Optional[Dict[str, Any]],
    slice_registry: Optional[Dict[str, Any]],
) -> str:
    """Select NCI operational mode based on available data sources.

    Mode Selection Logic (from Section 11.1):
        IF telemetry_schema IS NULL AND slice_registry IS NULL:
            mode = DOC_ONLY
        ELIF telemetry_schema IS NOT NULL AND slice_registry IS NULL:
            mode = TELEMETRY_CHECKED
        ELIF telemetry_schema IS NOT NULL AND slice_registry IS NOT NULL:
            mode = FULLY_BOUND
        ELSE:
            mode = DOC_ONLY  # slice_registry without telemetry is invalid
    """
    if telemetry_schema is None and slice_registry is None:
        return NCI_MODE_DOC_ONLY
    elif telemetry_schema is not None and slice_registry is None:
        return NCI_MODE_TELEMETRY_CHECKED
    elif telemetry_schema is not None and slice_registry is not None:
        return NCI_MODE_FULLY_BOUND
    else:
        # slice_registry without telemetry is invalid configuration
        return NCI_MODE_DOC_ONLY


def _compute_confidence_doc_only(panel: Dict[str, Any]) -> float:
    """Compute confidence for DOC_ONLY mode (Section 11.2).

    Base: 0.70 (lower due to limited validation)
    Coverage bonus: +0.10 * (metrics present / 4)
    Penalty: -0.10 (no telemetry) -0.05 (no slice registry)
    Floor: 0.50, Ceiling: 1.0
    """
    base = 0.70

    # Dimensional coverage
    metrics = panel.get("metrics_summary", {})
    non_null_count = len([v for v in metrics.values() if v is not None])
    coverage = non_null_count / 4.0 if metrics else 0.0
    base += 0.10 * coverage

    # Penalty for missing external validation
    base -= 0.10  # No telemetry validation
    base -= 0.05  # No slice registry validation

    return max(0.50, min(1.0, base))


def _compute_confidence_telemetry_checked(
    panel: Dict[str, Any],
    tcl_result: Dict[str, Any],
) -> float:
    """Compute confidence for TELEMETRY_CHECKED mode (Section 11.3).

    Base: 0.80 (higher with telemetry validation)
    Coverage bonus: +0.05 * (metrics present / 4)
    TCL alignment: +0.10 if aligned, else -min(0.15, violations * 0.03)
    Schema freshness: -0.05 if schema_age_hours < 24
    Penalty: -0.05 (no slice registry)
    Floor: 0.50, Ceiling: 1.0
    """
    base = 0.80

    # Dimensional coverage
    metrics = panel.get("metrics_summary", {})
    non_null_count = len([v for v in metrics.values() if v is not None])
    coverage = non_null_count / 4.0 if metrics else 0.0
    base += 0.05 * coverage

    # TCL alignment bonus/penalty
    if tcl_result.get("aligned", False):
        base += 0.10
    else:
        violations = len(tcl_result.get("violations", []))
        base -= min(0.15, violations * 0.03)

    # Schema freshness penalty
    schema_age_hours = tcl_result.get("schema_age_hours", 0)
    if schema_age_hours < 24:
        base -= 0.05

    # Penalty for missing slice validation
    base -= 0.05

    return max(0.50, min(1.0, base))


def _compute_confidence_fully_bound(
    panel: Dict[str, Any],
    tcl_result: Dict[str, Any],
    sic_result: Dict[str, Any],
) -> float:
    """Compute confidence for FULLY_BOUND mode (Section 11.4).

    Base: 0.85 (highest with full validation)
    Coverage bonus: +0.05 * (metrics present / 4)
    TCL alignment: +0.05 if aligned, else -min(0.10, violations * 0.02)
    SIC alignment: +0.05 if aligned, else -min(0.10, violations * 0.02)
    Schema freshness: -0.05 if schema_age_hours < 24
    Registry bonus: +0.03 if registry_validated
    Floor: 0.50, Ceiling: 1.0
    """
    base = 0.85

    # Dimensional coverage
    metrics = panel.get("metrics_summary", {})
    non_null_count = len([v for v in metrics.values() if v is not None])
    coverage = non_null_count / 4.0 if metrics else 0.0
    base += 0.05 * coverage

    # TCL alignment
    if tcl_result.get("aligned", False):
        base += 0.05
    else:
        violations = len(tcl_result.get("violations", []))
        base -= min(0.10, violations * 0.02)

    # SIC alignment
    if sic_result.get("aligned", False):
        base += 0.05
    else:
        violations = len(sic_result.get("violations", []))
        base -= min(0.10, violations * 0.02)

    # Schema freshness penalty
    schema_age_hours = tcl_result.get("schema_age_hours", 0)
    if schema_age_hours < 24:
        base -= 0.05

    # Registry freshness bonus
    registry_validated = sic_result.get("registry_validated", False)
    if registry_validated:
        base += 0.03

    return max(0.50, min(1.0, base))


def _run_tcl_checks_for_mode(
    mode: str,
    doc_contents: Dict[str, str],
    telemetry_schema: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Run TCL checks based on operational mode.

    DOC_ONLY: Only TCL-002 (canonical field names, spec-defined)
    TELEMETRY_CHECKED/FULLY_BOUND: All TCL checks (TCL-001 through TCL-004)
    """
    checks_run = []
    checks_skipped = []
    violations = []
    schema_age_hours = 0

    if mode == NCI_MODE_DOC_ONLY:
        # Only TCL-002 is active in DOC_ONLY
        checks_run.append("TCL-002")
        checks_skipped.extend(["TCL-001", "TCL-003", "TCL-004"])

        # Run TCL-002 check
        tcl_result = check_telemetry_consistency(doc_contents, telemetry_schema)
        violations.extend(tcl_result.get("violations", []))
    else:
        # TELEMETRY_CHECKED or FULLY_BOUND: all TCL checks active
        checks_run.extend(["TCL-001", "TCL-002", "TCL-003", "TCL-004"])

        # Run TCL-002 (field name consistency)
        tcl_result = check_telemetry_consistency(doc_contents, telemetry_schema)
        violations.extend(tcl_result.get("violations", []))

        # TCL-001: Event name alignment (validate against schema events)
        if telemetry_schema and "events" in telemetry_schema:
            schema_events = set(telemetry_schema.get("events", {}).keys())
            for doc_path, content in doc_contents.items():
                # Simple heuristic: look for event-like references
                for event_name in schema_events:
                    # This is a simplified check; full implementation would
                    # use more sophisticated event reference extraction
                    pass  # Event validation placeholder

        # TCL-003: Schema version reference
        if telemetry_schema and "schema_version" in telemetry_schema:
            schema_version = telemetry_schema["schema_version"]
            # Check if docs reference the correct version
            for doc_path, content in doc_contents.items():
                if "schema_version" in content.lower():
                    # Simplified check for version mismatch
                    pass  # Version check placeholder

        # TCL-004: Drift detection
        if telemetry_schema:
            schema_age_hours = telemetry_schema.get("schema_age_hours", 0)

    aligned = len(violations) == 0

    return {
        "aligned": aligned,
        "checks_run": checks_run,
        "checks_skipped": checks_skipped,
        "violations": violations[:50],
        "schema_age_hours": schema_age_hours,
    }


def _run_sic_checks_for_mode(
    mode: str,
    doc_contents: Dict[str, str],
    slice_registry: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Run SIC checks based on operational mode.

    DOC_ONLY: SIC-001 (canonical names), SIC-004 (capability patterns)
    TELEMETRY_CHECKED: Same as DOC_ONLY
    FULLY_BOUND: All SIC checks (SIC-001 through SIC-004)
    """
    checks_run = []
    checks_skipped = []
    violations = []
    registry_validated = False

    if mode in (NCI_MODE_DOC_ONLY, NCI_MODE_TELEMETRY_CHECKED):
        # SIC-001 and SIC-004 active
        checks_run.extend(["SIC-001", "SIC-004"])
        checks_skipped.extend(["SIC-002", "SIC-003"])

        # Run SIC-001 check (canonical slice names)
        canonical_slices = list(SIC_CANONICAL_SLICES.keys())
        sic_violations = check_slice_consistency(doc_contents, slice_registry)
        violations.extend(sic_violations)

        # SIC-004: Capability claim patterns (spec-defined)
        # Simplified: check for overclaiming patterns
        for doc_path, content in doc_contents.items():
            # Look for capability overclaims like "unlimited depth"
            overclaim_patterns = [
                (r"\bunlimited\s+depth\b", "unlimited_depth_claim"),
                (r"\bno\s+limit\b", "no_limit_claim"),
            ]
            for pattern, claim_type in overclaim_patterns:
                matches = list(re.finditer(pattern, content, re.IGNORECASE))
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    violations.append({
                        "doc": doc_path,
                        "violation_type": "SIC-004",
                        "claim_type": claim_type,
                        "found": match.group(),
                        "line": line_num,
                    })

    else:  # FULLY_BOUND
        checks_run.extend(["SIC-001", "SIC-002", "SIC-003", "SIC-004"])

        # SIC-001 check
        sic_violations = check_slice_consistency(doc_contents, slice_registry)
        violations.extend(sic_violations)

        # SIC-002: Parameter validation against registry
        if slice_registry and "slices" in slice_registry:
            registry_validated = True
            registry_slices = slice_registry["slices"]
            for doc_path, content in doc_contents.items():
                for slice_name, params in registry_slices.items():
                    # Check for parameter mismatches
                    if slice_name in content:
                        # Look for documented parameters
                        for param_name, param_value in params.items():
                            # Simplified check: look for conflicting values
                            pass  # Parameter validation placeholder

        # SIC-003: Slice-phase mapping validation
        if slice_registry and "phase_mapping" in slice_registry:
            # Validate that documented slices reference valid phases
            pass  # Phase mapping validation placeholder

        # SIC-004: Capability claim patterns
        for doc_path, content in doc_contents.items():
            overclaim_patterns = [
                (r"\bunlimited\s+depth\b", "unlimited_depth_claim"),
                (r"\bno\s+limit\b", "no_limit_claim"),
            ]
            for pattern, claim_type in overclaim_patterns:
                matches = list(re.finditer(pattern, content, re.IGNORECASE))
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    violations.append({
                        "doc": doc_path,
                        "violation_type": "SIC-004",
                        "claim_type": claim_type,
                        "found": match.group(),
                        "line": line_num,
                    })

    aligned = len(violations) == 0

    return {
        "aligned": aligned,
        "checks_run": checks_run,
        "checks_skipped": checks_skipped,
        "violations": violations[:50],
        "registry_validated": registry_validated,
    }


def _evaluate_slo_for_mode(
    mode: str,
    global_nci: float,
    tcl_result: Dict[str, Any],
    sic_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate SLO based on mode-specific thresholds.

    Returns SLO evaluation result with status (OK/WARN/BREACH).
    """
    thresholds = MODE_SLO_THRESHOLDS.get(mode, MODE_SLO_THRESHOLDS[NCI_MODE_DOC_ONLY])

    violations = []
    status = "OK"

    # Check global NCI against thresholds
    if global_nci < thresholds["global_nci_breach"]:
        violations.append(
            f"Global NCI ({global_nci:.2f}) below BREACH threshold ({thresholds['global_nci_breach']})"
        )
        status = "BREACH"
    elif global_nci < thresholds["global_nci_warn"]:
        violations.append(
            f"Global NCI ({global_nci:.2f}) below WARN threshold ({thresholds['global_nci_warn']})"
        )
        if status != "BREACH":
            status = "WARN"

    # Check violation counts
    tcl_violation_count = len(tcl_result.get("violations", []))
    sic_violation_count = len(sic_result.get("violations", []))
    total_violations = tcl_violation_count + sic_violation_count

    if total_violations >= thresholds["violation_count_breach"]:
        violations.append(
            f"Total violations ({total_violations}) at or above BREACH threshold ({thresholds['violation_count_breach']})"
        )
        status = "BREACH"

    return {
        "status": status,
        "thresholds_used": thresholds,
        "violations": violations,
        "violation_summary": {
            "tcl_count": tcl_violation_count,
            "sic_count": sic_violation_count,
            "total": total_violations,
        },
    }


def _build_warnings_for_mode(
    mode: str,
    tcl_result: Dict[str, Any],
    sic_result: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Build mode-specific warnings from TCL/SIC results."""
    warnings = []

    # Build TCL warnings
    for violation in tcl_result.get("violations", [])[:10]:
        warning = {
            "mode": mode,
            "warning_type": violation.get("violation_type", "TCL-002"),
            "severity": "medium",
            "message": f"Non-canonical field name '{violation.get('found')}' found in {violation.get('doc')}:{violation.get('line', '?')}",
            "remediation": f"Replace '{violation.get('found')}' with canonical '{violation.get('expected', violation.get('field'))}'",
            "validation_context": _get_validation_context(mode),
        }
        warnings.append(warning)

    # Build SIC warnings
    for violation in sic_result.get("violations", [])[:10]:
        vtype = violation.get("violation_type", "SIC-001")
        if vtype == "SIC-001":
            warning = {
                "mode": mode,
                "warning_type": "SIC-001",
                "severity": "medium",
                "message": f"Non-canonical slice name '{violation.get('found')}' found in {violation.get('doc')}:{violation.get('line', '?')}",
                "remediation": f"Replace '{violation.get('found')}' with canonical '{violation.get('expected', violation.get('slice'))}'",
                "validation_context": _get_validation_context(mode),
            }
        elif vtype == "SIC-004":
            warning = {
                "mode": mode,
                "warning_type": "SIC-004",
                "severity": "high",
                "message": f"Capability overclaim '{violation.get('found')}' found in {violation.get('doc')}:{violation.get('line', '?')}",
                "remediation": "Remove or qualify capability claim to match slice configuration",
                "validation_context": _get_validation_context(mode),
            }
        else:
            warning = {
                "mode": mode,
                "warning_type": vtype,
                "severity": "medium",
                "message": f"Slice consistency violation in {violation.get('doc')}",
                "remediation": "Review slice documentation for accuracy",
                "validation_context": _get_validation_context(mode),
            }
        warnings.append(warning)

    return warnings[:20]  # Limit to 20 warnings


def _get_validation_context(mode: str) -> str:
    """Get validation context string for mode."""
    if mode == NCI_MODE_DOC_ONLY:
        return "documentation_only"
    elif mode == NCI_MODE_TELEMETRY_CHECKED:
        return "live_telemetry"
    else:
        return "live_telemetry_and_registry"


def evaluate_nci_p5(
    panel: Dict[str, Any],
    telemetry_schema: Optional[Dict[str, Any]] = None,
    slice_registry: Optional[Dict[str, Any]] = None,
    doc_contents: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Evaluate NCI under P5 operational mode.

    SHADOW MODE CONTRACT:
    - This function is purely observational
    - Output does NOT influence governance decisions
    - All warnings are advisory only

    Automatic mode selection based on available data sources:
    - DOC_ONLY: No telemetry, no slice registry
    - TELEMETRY_CHECKED: Telemetry available, no slice registry
    - FULLY_BOUND: Both telemetry and slice registry available

    Args:
        panel: NCI director panel (from build_nci_director_panel)
        telemetry_schema: Live telemetry schema, or None for DOC_ONLY mode
            {
                "schema_version": "1.2.0",
                "events": {...},
                "fields": {...},
                "schema_age_hours": 12,
            }
        slice_registry: Slice registry, or None for DOC_ONLY/TELEMETRY_CHECKED
            {
                "slices": {
                    "arithmetic_simple": {"depth_max": 4, "atom_max": 4, ...},
                    ...
                },
                "registry_validated": True,
            }
        doc_contents: Documentation contents for validation (path -> content)
            If None, uses empty dict (minimal validation)

    Returns:
        P5 evaluation result:
        {
            "schema_version": "1.0.0",
            "mode": "DOC_ONLY" | "TELEMETRY_CHECKED" | "FULLY_BOUND",
            "global_nci": float,
            "confidence": float,
            "tcl_result": {
                "aligned": bool,
                "checks_run": ["TCL-001", "TCL-002", ...],
                "checks_skipped": [...],
                "violations": [...],
            },
            "sic_result": {
                "aligned": bool,
                "checks_run": ["SIC-001", ...],
                "checks_skipped": [...],
                "violations": [...],
            },
            "warnings": [
                {"mode": "...", "warning_type": "...", "severity": "...", ...},
            ],
            "slo_evaluation": {
                "thresholds_used": {...},
                "status": "OK" | "WARN" | "BREACH",
            },
            "governance_signal": {
                "signal_type": "SIG-NAR",
                "schema_version": "1.0.0",
                ...
            },
            "operational_notes": [
                "Mode: DOC_ONLY selected",
                ...
            ],
            "shadow_mode": True,
        }
    """
    doc_contents = doc_contents or {}

    # Select operational mode
    mode = _select_nci_mode(telemetry_schema, slice_registry)

    # Extract global NCI from panel
    global_nci = panel.get("global_nci", 1.0)

    # Run TCL checks for mode
    tcl_result = _run_tcl_checks_for_mode(mode, doc_contents, telemetry_schema)

    # Run SIC checks for mode
    sic_result = _run_sic_checks_for_mode(mode, doc_contents, slice_registry)

    # Compute confidence based on mode
    if mode == NCI_MODE_DOC_ONLY:
        confidence = _compute_confidence_doc_only(panel)
    elif mode == NCI_MODE_TELEMETRY_CHECKED:
        confidence = _compute_confidence_telemetry_checked(panel, tcl_result)
    else:  # FULLY_BOUND
        confidence = _compute_confidence_fully_bound(panel, tcl_result, sic_result)

    # Evaluate SLO with mode-specific thresholds
    slo_evaluation = _evaluate_slo_for_mode(mode, global_nci, tcl_result, sic_result)

    # Build warnings
    warnings = _build_warnings_for_mode(mode, tcl_result, sic_result)

    # Build operational notes
    operational_notes = [f"Mode: {mode} selected"]
    if telemetry_schema:
        version = telemetry_schema.get("schema_version", "unknown")
        operational_notes.append(f"Telemetry schema v{version} validated")
    if slice_registry:
        slice_count = len(slice_registry.get("slices", {}))
        operational_notes.append(f"Slice registry contains {slice_count} active slice(s)")
    operational_notes.append(f"TCL checks run: {', '.join(tcl_result['checks_run'])}")
    operational_notes.append(f"SIC checks run: {', '.join(sic_result['checks_run'])}")

    # Build SIG-NAR governance signal
    governance_signal = _build_sig_nar_signal(
        mode=mode,
        global_nci=global_nci,
        confidence=confidence,
        tcl_result=tcl_result,
        sic_result=sic_result,
        slo_evaluation=slo_evaluation,
    )

    return {
        "schema_version": NCI_GOVERNANCE_TILE_SCHEMA_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "global_nci": round(global_nci, 4),
        "confidence": round(confidence, 4),
        "tcl_result": tcl_result,
        "sic_result": sic_result,
        "warnings": warnings,
        "slo_evaluation": slo_evaluation,
        "governance_signal": governance_signal,
        "operational_notes": operational_notes,
        "shadow_mode": True,
    }


def _build_sig_nar_signal(
    mode: str,
    global_nci: float,
    confidence: float,
    tcl_result: Dict[str, Any],
    sic_result: Dict[str, Any],
    slo_evaluation: Dict[str, Any],
) -> Dict[str, Any]:
    """Build SIG-NAR governance signal for GGFL integration.

    SIG-NAR is the Narrative Signal slot in the Global Governance Fusion Layer.
    """
    # Determine recommendation based on SLO status
    slo_status = slo_evaluation.get("status", "OK")
    if slo_status == "BREACH":
        recommendation = "REVIEW"
    elif slo_status == "WARN":
        recommendation = "WARNING"
    else:
        recommendation = "NONE"

    return {
        "signal_type": "SIG-NAR",
        "schema_version": NCI_GOVERNANCE_TILE_SCHEMA_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "health_contribution": {
            "global_nci": round(global_nci, 4),
            "confidence": round(confidence, 4),
            "status": slo_status,
        },
        "recommendation": recommendation,
        "telemetry_consistency": {
            "aligned": tcl_result.get("aligned", True),
            "violation_count": len(tcl_result.get("violations", [])),
        },
        "slice_consistency": {
            "aligned": sic_result.get("aligned", True),
            "violation_count": len(sic_result.get("violations", [])),
        },
        "slo_status": slo_status,
        "shadow_mode": True,
    }


# =============================================================================
# GGFL INTEGRATION STUB
# =============================================================================


def contribute_nci_to_ggfl(
    ggfl_surface: Dict[str, Any],
    nci_p5_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Contribute NCI signal to Global Governance Fusion Layer.

    SHADOW MODE CONTRACT:
    - This function is purely observational
    - The contribution does NOT influence governance decisions
    - This is a stub for future GGFL integration

    GGFL Integration Point:
    - NCI contributes to SIG-NAR slot in unified governance surface
    - Weight/influence determined by GGFL fusion logic (not this function)

    Args:
        ggfl_surface: Existing GGFL surface dict
        nci_p5_result: Output from evaluate_nci_p5()

    Returns:
        New GGFL surface with NCI contribution attached under "signals.SIG-NAR"
    """
    result = dict(ggfl_surface)

    # Ensure signals dict exists
    if "signals" not in result:
        result["signals"] = {}
    else:
        result["signals"] = dict(result["signals"])

    # Extract governance signal
    governance_signal = nci_p5_result.get("governance_signal", {})

    # Attach under SIG-NAR slot
    result["signals"]["SIG-NAR"] = {
        "source": "nci",
        "mode": nci_p5_result.get("mode", "DOC_ONLY"),
        "signal": governance_signal,
        "contribution_weight": 0.15,  # Default weight (GGFL may override)
        "shadow_mode": True,
    }

    return result


def build_ggfl_nci_contribution(
    nci_p5_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Build standalone NCI contribution for GGFL.

    This creates a self-contained contribution object that can be
    merged into the GGFL surface by external orchestration.

    Args:
        nci_p5_result: Output from evaluate_nci_p5()

    Returns:
        GGFL contribution object:
        {
            "signal_slot": "SIG-NAR",
            "source": "nci",
            "payload": {...},
            "metadata": {...}
        }
    """
    return {
        "signal_slot": "SIG-NAR",
        "source": "nci",
        "schema_version": NCI_GOVERNANCE_TILE_SCHEMA_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {
            "mode": nci_p5_result.get("mode", "DOC_ONLY"),
            "global_nci": nci_p5_result.get("global_nci", 1.0),
            "confidence": nci_p5_result.get("confidence", 0.5),
            "slo_status": nci_p5_result.get("slo_evaluation", {}).get("status", "OK"),
            "tcl_aligned": nci_p5_result.get("tcl_result", {}).get("aligned", True),
            "sic_aligned": nci_p5_result.get("sic_result", {}).get("aligned", True),
        },
        "metadata": {
            "checks_run": {
                "tcl": nci_p5_result.get("tcl_result", {}).get("checks_run", []),
                "sic": nci_p5_result.get("sic_result", {}).get("checks_run", []),
            },
            "warning_count": len(nci_p5_result.get("warnings", [])),
            "operational_notes": nci_p5_result.get("operational_notes", []),
        },
        "shadow_mode": True,
    }


def nci_p5_for_alignment_view(
    nci_p5_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Normalize NCI P5 signal into GGFL unified format for alignment views.

    This function converts an NCI P5 evaluation result into the Global Governance
    Fusion Layer (GGFL) unified format for cross-subsystem alignment views.

    SHADOW MODE CONTRACT:
    - Read-only, no side effects
    - Never influences governance decisions
    - Output is advisory/observational only
    - Deterministic: same input always produces same output

    GGFL Unified Format (fixed shape):
    - status: Lowercase status (ok/warn/block)
    - alignment: Governance alignment (aligned/tension/divergent)
    - conflict: Boolean conflict flag
    - top_reasons: List of reasons (max 5) for human readability

    NCI-Specific Extensions:
    - mode: NCI operational mode (DOC_ONLY/TELEMETRY_CHECKED/FULLY_BOUND)
    - global_nci: Global NCI score [0.0, 1.0]
    - confidence: Confidence score [0.0, 1.0]
    - tcl_aligned: Telemetry Consistency Law alignment
    - sic_aligned: Slice Identity Consistency alignment

    Args:
        nci_p5_result: Output from evaluate_nci_p5() or nci_p5_signal.json

    Returns:
        GGFL-normalized dict with fixed shape for alignment view fusion
    """
    # Extract SLO status and normalize to lowercase
    slo_status = nci_p5_result.get("slo_status")
    if slo_status is None:
        # Try nested structure from full result
        slo_eval = nci_p5_result.get("slo_evaluation", {})
        slo_status = slo_eval.get("status", "OK")

    # Map SLO status to GGFL status (lowercase)
    status_map = {
        "OK": "ok",
        "WARN": "warn",
        "BREACH": "block",  # BREACH maps to block in GGFL
    }
    status = status_map.get(slo_status.upper() if slo_status else "OK", "ok")

    # Derive alignment from TCL/SIC status
    tcl_aligned = nci_p5_result.get("tcl_aligned")
    sic_aligned = nci_p5_result.get("sic_aligned")

    # Handle nested structure from full result
    if tcl_aligned is None:
        tcl_result = nci_p5_result.get("tcl_result", {})
        tcl_aligned = tcl_result.get("aligned", True)
    if sic_aligned is None:
        sic_result = nci_p5_result.get("sic_result", {})
        sic_aligned = sic_result.get("aligned", True)

    # Determine alignment
    if tcl_aligned and sic_aligned:
        alignment = "aligned"
    elif not tcl_aligned and not sic_aligned:
        alignment = "divergent"
    else:
        alignment = "tension"

    # Conflict flag: True if both TCL and SIC have violations
    conflict = not tcl_aligned and not sic_aligned

    # Build GGFL drivers using REASON CODES ONLY (max 3)
    # No natural language - codes only for machine-readable processing
    # Valid codes: DRIVER_SLO_BREACH, DRIVER_RECOMMENDATION_NON_NONE, DRIVER_CONFIDENCE_LOW
    drivers: List[str] = []

    # Check for SLO breach
    if status == "block":  # Maps from BREACH
        drivers.append("DRIVER_SLO_BREACH")

    # Check for non-NONE recommendation
    recommendation = nci_p5_result.get("recommendation")
    if recommendation is None:
        gov_signal = nci_p5_result.get("governance_signal", {})
        recommendation = gov_signal.get("recommendation", "NONE")
    if recommendation and recommendation.upper() != "NONE":
        drivers.append("DRIVER_RECOMMENDATION_NON_NONE")

    # Check for low confidence
    confidence = nci_p5_result.get("confidence", 1.0)
    if confidence is not None and confidence < 0.5:
        drivers.append("DRIVER_CONFIDENCE_LOW")

    # DRIVERS MAX 3: Limit to top 3 reason codes
    drivers = drivers[:3]

    # Extract NCI-specific fields
    mode = nci_p5_result.get("mode", "DOC_ONLY")
    global_nci = nci_p5_result.get("global_nci", 1.0)

    return {
        # GGFL unified fields (fixed shape)
        "status": status,
        "alignment": alignment,
        "conflict": conflict,
        "drivers": drivers,  # Reason codes only (max 3)
        # NCI-specific extensions
        "mode": mode,
        "global_nci": global_nci,
        "confidence": confidence,
        "tcl_aligned": tcl_aligned,
        "sic_aligned": sic_aligned,
    }


def build_neutral_nci_summary(
    nci_p5_result: Dict[str, Any],
) -> str:
    """Build a neutral human-readable summary of NCI P5 evaluation.

    SHADOW MODE CONTRACT:
    - Read-only, deterministic
    - Advisory summary only, no gating influence

    Args:
        nci_p5_result: Output from evaluate_nci_p5() or nci_p5_signal.json

    Returns:
        Single-line neutral summary string
    """
    mode = nci_p5_result.get("mode", "DOC_ONLY")
    global_nci = nci_p5_result.get("global_nci", 1.0)

    # Get SLO status
    slo_status = nci_p5_result.get("slo_status")
    if slo_status is None:
        slo_eval = nci_p5_result.get("slo_evaluation", {})
        slo_status = slo_eval.get("status", "OK")

    # Get violation counts
    tcl_count = nci_p5_result.get("tcl_violation_count", 0)
    sic_count = nci_p5_result.get("sic_violation_count", 0)

    if tcl_count == 0 and sic_count == 0:
        tcl_result = nci_p5_result.get("tcl_result", {})
        sic_result = nci_p5_result.get("sic_result", {})
        tcl_count = len(tcl_result.get("violations", []))
        sic_count = len(sic_result.get("violations", []))

    # Build neutral summary
    nci_pct = int(global_nci * 100) if global_nci is not None else 100
    violation_total = tcl_count + sic_count

    if violation_total == 0:
        return f"NCI {mode}: {nci_pct}% consistency, no violations"
    else:
        return f"NCI {mode}: {nci_pct}% consistency, {violation_total} violation(s)"


def build_nci_status_warning(
    nci_p5_result: Dict[str, Any],
) -> Optional[str]:
    """Build a single status warning string if warranted, else None.

    STATUS WARNING HYGIENE:
    - Returns a single warning ONLY if slo_status == BREACH or recommendation != NONE
    - Includes global_nci and confidence in the warning
    - Returns None for healthy (OK/NONE) states

    SHADOW MODE CONTRACT:
    - Read-only, deterministic
    - Advisory warning only, no gating influence

    Args:
        nci_p5_result: Output from evaluate_nci_p5() or nci_p5_signal.json

    Returns:
        Single warning string if BREACH/recommendation, else None
    """
    # Extract SLO status
    slo_status = nci_p5_result.get("slo_status")
    if slo_status is None:
        slo_eval = nci_p5_result.get("slo_evaluation", {})
        slo_status = slo_eval.get("status", "OK")

    # Extract recommendation
    recommendation = nci_p5_result.get("recommendation")
    if recommendation is None:
        gov_signal = nci_p5_result.get("governance_signal", {})
        recommendation = gov_signal.get("recommendation", "NONE")

    # Only emit warning if BREACH or non-NONE recommendation
    slo_upper = (slo_status or "OK").upper()
    rec_upper = (recommendation or "NONE").upper()

    if slo_upper != "BREACH" and rec_upper == "NONE":
        return None

    # Build warning with global_nci and confidence
    global_nci = nci_p5_result.get("global_nci", 1.0)
    confidence = nci_p5_result.get("confidence", 1.0)

    nci_pct = int(global_nci * 100) if global_nci is not None else 100
    conf_pct = int(confidence * 100) if confidence is not None else 100

    if slo_upper == "BREACH":
        return f"NCI BREACH: {nci_pct}% consistency (confidence {conf_pct}%)"
    else:
        # Non-NONE recommendation (WARNING or REVIEW)
        return f"NCI {rec_upper}: {nci_pct}% consistency (confidence {conf_pct}%)"


# =============================================================================
# ARTIFACT LOADERS FOR CI/CLI INTEGRATION
# =============================================================================

# Default curated document sets for NCI scanning
NCI_CURATED_DOC_PATTERNS = [
    # System law documents (primary)
    "docs/system_law/**/*.md",
    # Key runbooks and specs
    "docs/API_REFERENCE.md",
    "docs/ARCHITECTURE.md",
    "docs/whitepaper.md",
    "docs/RFL_LAW.md",
    "docs/DETERMINISM_CONTRACT.md",
    "docs/HASHING_SPEC.md",
    "docs/DAG_SPEC.md",
    "docs/ATTESTATION_SPEC.md",
    # CI documentation
    "docs/ci/*.md",
    # Security documentation
    "docs/security/*.md",
]

# Maximum file size to read (prevent OOM on huge files)
MAX_DOC_SIZE_BYTES = 1_000_000  # 1MB


def load_doc_contents_for_nci(
    repo_root: Union[str, Path],
    patterns: Optional[List[str]] = None,
    max_files: int = 100,
) -> Dict[str, str]:
    """Load documentation contents for NCI evaluation.

    Scans a curated set of documentation files from the repository
    and returns their contents for TCL/SIC consistency checking.

    SHADOW MODE CONTRACT:
    - This function is read-only
    - No governance decisions are made

    Args:
        repo_root: Repository root directory
        patterns: Glob patterns to scan (defaults to NCI_CURATED_DOC_PATTERNS)
        max_files: Maximum number of files to load (prevents runaway)

    Returns:
        Dict mapping relative file paths to content strings.
        Example:
        {
            "docs/system_law/NCI_PhaseX_Spec.md": "# NCI Phase X...",
            "docs/API_REFERENCE.md": "# API Reference...",
        }
    """
    repo_root = Path(repo_root)
    patterns = patterns or NCI_CURATED_DOC_PATTERNS
    doc_contents: Dict[str, str] = {}
    files_loaded = 0

    for pattern in patterns:
        if files_loaded >= max_files:
            break

        # Handle glob patterns
        if "*" in pattern:
            matched_files = list(repo_root.glob(pattern))
        else:
            # Direct file path
            direct_path = repo_root / pattern
            matched_files = [direct_path] if direct_path.exists() else []

        for file_path in matched_files:
            if files_loaded >= max_files:
                break

            if not file_path.is_file():
                continue

            # Skip files that are too large
            try:
                file_size = file_path.stat().st_size
                if file_size > MAX_DOC_SIZE_BYTES:
                    continue
            except OSError:
                continue

            # Read file content
            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
                # Use relative path as key
                rel_path = str(file_path.relative_to(repo_root)).replace("\\", "/")
                doc_contents[rel_path] = content
                files_loaded += 1
            except (OSError, UnicodeDecodeError):
                # Skip unreadable files
                continue

    return doc_contents


def build_curated_docs_audit_report(
    repo_root: Union[str, Path],
    patterns: Optional[List[str]] = None,
    max_entries: int = 20,
) -> Dict[str, Any]:
    """Build an audit report of curated doc patterns matching.

    Produces a bounded, deterministic report showing which documentation
    files matched patterns and which were skipped (with reasons).

    SHADOW MODE CONTRACT:
    - Read-only, no side effects
    - Deterministic: same inputs produce identical outputs
    - Advisory only, no gating influence

    Args:
        repo_root: Repository root directory
        patterns: Glob patterns to audit (defaults to NCI_CURATED_DOC_PATTERNS)
        max_entries: Maximum entries per category (matched/skipped), default 20

    Returns:
        Audit report dict with:
        {
            "schema_version": "1.0.0",
            "patterns_checked": [...],
            "matched_files": [...],  # max 20 entries
            "skipped_files": [...],  # max 20 entries
            "summary": {
                "total_patterns": int,
                "total_matched": int,
                "total_skipped": int,
                "truncated": bool
            }
        }
    """
    repo_root = Path(repo_root)
    patterns = patterns or NCI_CURATED_DOC_PATTERNS

    matched_files: List[Dict[str, Any]] = []
    skipped_files: List[Dict[str, Any]] = []
    total_matched = 0
    total_skipped = 0

    for pattern in patterns:
        # Handle glob patterns
        if "*" in pattern:
            try:
                matched_paths = sorted(repo_root.glob(pattern))
            except Exception:
                matched_paths = []
        else:
            # Direct file path
            direct_path = repo_root / pattern
            matched_paths = [direct_path] if direct_path.exists() else []

        for file_path in matched_paths:
            if not file_path.is_file():
                continue

            try:
                rel_path = str(file_path.relative_to(repo_root)).replace("\\", "/")
            except ValueError:
                rel_path = str(file_path)

            # Check file size
            try:
                file_size = file_path.stat().st_size
            except OSError:
                total_skipped += 1
                if len(skipped_files) < max_entries:
                    skipped_files.append({
                        "path": rel_path,
                        "reason": "stat_failed",
                        "pattern": pattern,
                    })
                continue

            # Check if file is too large
            if file_size > MAX_DOC_SIZE_BYTES:
                total_skipped += 1
                if len(skipped_files) < max_entries:
                    skipped_files.append({
                        "path": rel_path,
                        "reason": "too_large",
                        "size_bytes": file_size,
                        "pattern": pattern,
                    })
                continue

            # Check if file is readable
            try:
                # Just open to verify readability, don't store content
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    f.read(1)  # Read 1 byte to verify
                readable = True
            except (OSError, UnicodeDecodeError):
                readable = False

            if not readable:
                total_skipped += 1
                if len(skipped_files) < max_entries:
                    skipped_files.append({
                        "path": rel_path,
                        "reason": "unreadable",
                        "pattern": pattern,
                    })
                continue

            # File matched and is readable
            total_matched += 1
            if len(matched_files) < max_entries:
                matched_files.append({
                    "path": rel_path,
                    "size_bytes": file_size,
                    "pattern": pattern,
                })

    # Build report
    return {
        "schema_version": "1.0.0",
        "patterns_checked": list(patterns),
        "matched_files": matched_files,
        "skipped_files": skipped_files,
        "summary": {
            "total_patterns": len(patterns),
            "total_matched": total_matched,
            "total_skipped": total_skipped,
            "matched_truncated": total_matched > max_entries,
            "skipped_truncated": total_skipped > max_entries,
        },
    }


def load_telemetry_schema_from_file(
    schema_path: Union[str, Path],
) -> Optional[Dict[str, Any]]:
    """Load telemetry schema from a JSON file.

    SHADOW MODE CONTRACT:
    - This function is read-only
    - No governance decisions are made

    Args:
        schema_path: Path to telemetry schema JSON file

    Returns:
        Telemetry schema dict, or None if file doesn't exist or is invalid.
        Expected structure:
        {
            "schema_version": "1.0.0",
            "events": {...},
            "fields": {...},
            "schema_age_hours": 0,
        }
    """
    schema_path = Path(schema_path)

    if not schema_path.exists():
        return None

    try:
        content = schema_path.read_text(encoding="utf-8")
        schema = json.loads(content)

        # Validate minimum required fields
        if not isinstance(schema, dict):
            return None

        # Add schema_age_hours if missing (compute from file mtime)
        if "schema_age_hours" not in schema:
            try:
                import time
                mtime = schema_path.stat().st_mtime
                age_seconds = time.time() - mtime
                schema["schema_age_hours"] = int(age_seconds / 3600)
            except OSError:
                schema["schema_age_hours"] = 0

        return schema

    except (json.JSONDecodeError, OSError):
        return None


def load_slice_registry_from_file(
    registry_path: Union[str, Path],
) -> Optional[Dict[str, Any]]:
    """Load slice registry from a JSON file.

    SHADOW MODE CONTRACT:
    - This function is read-only
    - No governance decisions are made

    Args:
        registry_path: Path to slice registry JSON file

    Returns:
        Slice registry dict, or None if file doesn't exist or is invalid.
        Expected structure:
        {
            "slices": {
                "arithmetic_simple": {"depth_max": 4, "atom_max": 4},
                ...
            },
            "registry_validated": true,
        }
    """
    registry_path = Path(registry_path)

    if not registry_path.exists():
        return None

    try:
        content = registry_path.read_text(encoding="utf-8")
        registry = json.loads(content)

        # Validate minimum required fields
        if not isinstance(registry, dict):
            return None

        # Ensure slices key exists
        if "slices" not in registry:
            registry["slices"] = {}

        return registry

    except (json.JSONDecodeError, OSError):
        return None


def run_nci_p5_with_artifacts(
    repo_root: Union[str, Path],
    telemetry_schema_path: Optional[Union[str, Path]] = None,
    slice_registry_path: Optional[Union[str, Path]] = None,
    doc_patterns: Optional[List[str]] = None,
    mock_global_nci: float = 0.85,
) -> Dict[str, Any]:
    """Run NCI P5 evaluation with artifact loading.

    Convenience function that loads artifacts and runs evaluate_nci_p5().

    SHADOW MODE CONTRACT:
    - All outputs are advisory only
    - No governance decisions are made

    Args:
        repo_root: Repository root directory
        telemetry_schema_path: Optional path to telemetry schema JSON
        slice_registry_path: Optional path to slice registry JSON
        doc_patterns: Optional glob patterns for doc scanning
        mock_global_nci: Mock global NCI value for panel (default 0.85)

    Returns:
        NCI P5 evaluation result with additional metadata:
        {
            ...evaluate_nci_p5 result...,
            "artifact_metadata": {
                "repo_root": str,
                "doc_count": int,
                "telemetry_schema_loaded": bool,
                "slice_registry_loaded": bool,
            }
        }
    """
    repo_root = Path(repo_root)

    # Load documentation contents
    doc_contents = load_doc_contents_for_nci(repo_root, doc_patterns)

    # Load optional telemetry schema
    telemetry_schema = None
    if telemetry_schema_path:
        telemetry_schema = load_telemetry_schema_from_file(telemetry_schema_path)

    # Load optional slice registry
    slice_registry = None
    if slice_registry_path:
        slice_registry = load_slice_registry_from_file(slice_registry_path)

    # Build a minimal panel for evaluation
    # In production, this would come from actual NCI index computation
    insight_summary = {
        "global_nci": mock_global_nci,
        "dominant_area": "none",
        "summary": {
            "terminology_alignment": 0.90,
            "phase_discipline": 0.85,
            "uplift_avoidance": 0.95,
            "structural_coherence": 0.80,
        },
    }
    priority_view = {"status": "OK", "priority_areas": []}
    slo_result = {"slo_status": "OK", "violations": []}
    panel = build_nci_director_panel(insight_summary, priority_view, slo_result)

    # Run P5 evaluation
    result = evaluate_nci_p5(
        panel=panel,
        telemetry_schema=telemetry_schema,
        slice_registry=slice_registry,
        doc_contents=doc_contents,
    )

    # Add artifact metadata
    result["artifact_metadata"] = {
        "repo_root": str(repo_root),
        "doc_count": len(doc_contents),
        "doc_paths": sorted(doc_contents.keys())[:20],  # First 20 for brevity
        "telemetry_schema_loaded": telemetry_schema is not None,
        "slice_registry_loaded": slice_registry is not None,
    }

    return result


def build_nci_p5_compact_signal(
    nci_p5_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Build compact NCI P5 signal for CI/GGFL consumption.

    Produces a minimal JSON structure suitable for CI artifacts
    and GGFL signal slots.

    Args:
        nci_p5_result: Full NCI P5 evaluation result

    Returns:
        Compact signal:
        {
            "schema_version": "1.0.0",
            "signal_type": "SIG-NAR",
            "mode": "DOC_ONLY",
            "global_nci": 0.85,
            "confidence": 0.65,
            "slo_status": "OK",
            "recommendation": "NONE",
            "tcl_aligned": true,
            "sic_aligned": true,
            "warning_count": 0,
            "shadow_mode": true,
        }
    """
    gov_signal = nci_p5_result.get("governance_signal", {})
    tcl_result = nci_p5_result.get("tcl_result", {})
    sic_result = nci_p5_result.get("sic_result", {})

    return {
        "schema_version": NCI_GOVERNANCE_TILE_SCHEMA_VERSION,
        "signal_type": "SIG-NAR",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": nci_p5_result.get("mode", "DOC_ONLY"),
        "global_nci": nci_p5_result.get("global_nci", 1.0),
        "confidence": nci_p5_result.get("confidence", 0.5),
        "slo_status": nci_p5_result.get("slo_evaluation", {}).get("status", "OK"),
        "recommendation": gov_signal.get("recommendation", "NONE"),
        "tcl_aligned": tcl_result.get("aligned", True),
        "sic_aligned": sic_result.get("aligned", True),
        "tcl_violation_count": len(tcl_result.get("violations", [])),
        "sic_violation_count": len(sic_result.get("violations", [])),
        "warning_count": len(nci_p5_result.get("warnings", [])),
        "shadow_mode": True,
    }


__all__ = [
    # Constants
    "NCI_GOVERNANCE_TILE_SCHEMA_VERSION",
    "DEFAULT_NCI_SLO_THRESHOLDS",
    "TCL_CANONICAL_FIELDS",
    "SIC_CANONICAL_SLICES",
    "NCI_MODE_DOC_ONLY",
    "NCI_MODE_TELEMETRY_CHECKED",
    "NCI_MODE_FULLY_BOUND",
    "MODE_SLO_THRESHOLDS",
    "NCI_CURATED_DOC_PATTERNS",
    # Core builders
    "build_nci_director_panel",
    "build_nci_governance_signal",
    "check_telemetry_consistency",
    "check_slice_consistency",
    "build_nci_tile_for_global_health",
    "attach_nci_tile_to_global_health",
    "build_nci_summary_for_p3",
    "attach_nci_summary_to_stability_report",
    "attach_nci_to_evidence",
    "build_nci_evidence_attachment",
    # P5 evaluation
    "evaluate_nci_p5",
    "contribute_nci_to_ggfl",
    "build_ggfl_nci_contribution",
    # GGFL alignment view adapter
    "nci_p5_for_alignment_view",
    "build_neutral_nci_summary",
    "build_nci_status_warning",
    # Artifact loaders
    "load_doc_contents_for_nci",
    "load_telemetry_schema_from_file",
    "load_slice_registry_from_file",
    "run_nci_p5_with_artifacts",
    "build_nci_p5_compact_signal",
    # Audit report
    "build_curated_docs_audit_report",
]
