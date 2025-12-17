#!/usr/bin/env python3
"""
PHASE VI — NOT RUN IN PHASE I

Predictive Uplift Safety & Stability Forecaster v6.0

This module unifies epistemic, semantic, drift, metric, atlas, and telemetry
signals into a predictive uplift-stability model for MAAS (Model-as-a-Service)
uplift gate decisions.

ABSOLUTE SAFEGUARDS:
    - This tool is DESCRIPTIVE, not NORMATIVE
    - No modifications to experimental data
    - No inference or claims regarding uplift beyond safety assessment
"""

import json
import logging
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).parent.parent

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data Structure Definitions
# ─────────────────────────────────────────────────────────────────────────────

def normalize_tensor_value(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Normalize a tensor value to [0, 1] range."""
    if max_val == min_val:
        return 0.5  # Default neutral value
    return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))


def compute_tensor_norm(tensor: Dict[str, Any]) -> float:
    """
    Compute L2 norm of a multi-dimensional tensor.
    
    Supports various tensor formats:
    - {"values": [v1, v2, ...]} - 1D vector
    - {"matrix": [[...], [...]]} - 2D matrix
    - {"components": {"x": v1, "y": v2, ...}} - Named components
    """
    if "values" in tensor:
        # 1D vector
        values = tensor["values"]
        return math.sqrt(sum(v * v for v in values if isinstance(v, (int, float))))
    
    elif "matrix" in tensor:
        # 2D matrix
        matrix = tensor["matrix"]
        if not matrix:
            return 0.0
        squared_sum = sum(
            cell * cell
            for row in matrix
            for cell in row
            if isinstance(cell, (int, float))
        )
        return math.sqrt(squared_sum)
    
    elif "components" in tensor:
        # Named components
        components = tensor["components"]
        squared_sum = sum(
            v * v
            for v in components.values()
            if isinstance(v, (int, float))
        )
        return math.sqrt(squared_sum)
    
    elif "norm" in tensor:
        # Pre-computed norm
        return float(tensor["norm"])
    
    else:
        # Fallback: try to extract numeric values
        values = [v for v in tensor.values() if isinstance(v, (int, float))]
        if values:
            return math.sqrt(sum(v * v for v in values))
        return 0.0


def extract_risk_indicators(
    epistemic_tensor: Dict[str, Any],
    drift_tensor: Dict[str, Any],
    atlas_lattice: Dict[str, Any],
    telemetry_safety_panel: Dict[str, Any],
) -> Dict[str, float]:
    """
    Extract risk indicators from each input signal.
    
    Returns a dictionary mapping signal names to normalized risk values [0, 1].
    """
    indicators: Dict[str, float] = {}
    
    # Epistemic risk: uncertainty, confidence degradation
    if "uncertainty" in epistemic_tensor:
        indicators["epistemic_uncertainty"] = normalize_tensor_value(
            float(epistemic_tensor["uncertainty"]), 0.0, 1.0
        )
    elif "confidence" in epistemic_tensor:
        # Lower confidence = higher risk
        indicators["epistemic_uncertainty"] = 1.0 - normalize_tensor_value(
            float(epistemic_tensor["confidence"]), 0.0, 1.0
        )
    else:
        indicators["epistemic_uncertainty"] = 0.5  # Neutral
    
    # Drift risk: coverage degradation, contract misalignment
    if "coverage_trend" in drift_tensor:
        trend = drift_tensor["coverage_trend"]
        if trend == "DEGRADING":
            indicators["drift_risk"] = 0.8
        elif trend == "STABLE":
            indicators["drift_risk"] = 0.4
        else:  # IMPROVING
            indicators["drift_risk"] = 0.2
    elif "coverage_pct" in drift_tensor:
        # Lower coverage = higher risk
        coverage = float(drift_tensor["coverage_pct"])
        indicators["drift_risk"] = 1.0 - normalize_tensor_value(coverage, 0.0, 100.0)
    else:
        indicators["drift_risk"] = 0.5  # Neutral
    
    # Atlas risk: structural instability
    if "stability_score" in atlas_lattice:
        # Lower stability = higher risk
        stability = float(atlas_lattice["stability_score"])
        indicators["atlas_risk"] = 1.0 - normalize_tensor_value(stability, 0.0, 1.0)
    elif "instability_indicators" in atlas_lattice:
        instability_count = len(atlas_lattice.get("instability_indicators", []))
        indicators["atlas_risk"] = normalize_tensor_value(float(instability_count), 0.0, 10.0)
    else:
        indicators["atlas_risk"] = 0.5  # Neutral
    
    # Telemetry risk: operational degradation
    if "safety_status" in telemetry_safety_panel:
        status = telemetry_safety_panel["safety_status"]
        status_map = {"OK": 0.2, "WARN": 0.5, "ATTENTION": 0.7, "CRITICAL": 0.9}
        indicators["telemetry_risk"] = status_map.get(status, 0.5)
    elif "risk_score" in telemetry_safety_panel:
        indicators["telemetry_risk"] = normalize_tensor_value(
            float(telemetry_safety_panel["risk_score"]), 0.0, 1.0
        )
    else:
        indicators["telemetry_risk"] = 0.5  # Neutral
    
    return indicators


def build_global_uplift_safety_tensor(
    epistemic_tensor: Dict[str, Any],
    drift_tensor: Dict[str, Any],
    atlas_lattice: Dict[str, Any],
    telemetry_safety_panel: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build global uplift safety tensor from unified signals.
    
    PHASE VI — NOT RUN IN PHASE I
    No uplift claims are made.
    Deterministic execution guaranteed.
    
    Args:
        epistemic_tensor: Epistemic uncertainty/confidence signals
        drift_tensor: Field Manual drift and coverage signals
        atlas_lattice: Structural stability signals
        telemetry_safety_panel: Operational telemetry safety signals
    
    Returns:
        {
            "schema_version": "1.0.0",
            "tensor_norm": float,
            "uplift_risk_band": "LOW" | "MEDIUM" | "HIGH",
            "hotspot_axes": List[str],
            "risk_indicators": Dict[str, float],
            "neutral_notes": List[str]
        }
    
    Risk Band Logic:
        - HIGH: All signals show moderate degradation (risk > 0.6)
        - MEDIUM: Some signals show degradation (risk > 0.4)
        - LOW: Otherwise
    """
    # Extract risk indicators from each signal
    risk_indicators = extract_risk_indicators(
        epistemic_tensor, drift_tensor, atlas_lattice, telemetry_safety_panel
    )
    
    # Compute tensor norm from risk indicators
    risk_values = list(risk_indicators.values())
    tensor_norm = math.sqrt(sum(v * v for v in risk_values)) if risk_values else 0.0
    
    # Determine risk band
    max_risk = max(risk_values) if risk_values else 0.0
    avg_risk = sum(risk_values) / len(risk_values) if risk_values else 0.0
    
    # High risk if all signals show moderate degradation
    if max_risk > 0.6 and avg_risk > 0.5:
        risk_band = "HIGH"
    elif max_risk > 0.4 or avg_risk > 0.35:
        risk_band = "MEDIUM"
    else:
        risk_band = "LOW"
    
    # Identify hotspot axes (signals with highest risk)
    sorted_risks = sorted(risk_indicators.items(), key=lambda x: x[1], reverse=True)
    hotspot_axes = [
        axis for axis, risk in sorted_risks if risk > 0.5
    ]
    
    # Build neutral notes
    neutral_notes: List[str] = []
    neutral_notes.append(f"Tensor norm: {tensor_norm:.3f}")
    neutral_notes.append(f"Risk band: {risk_band}")
    
    if hotspot_axes:
        neutral_notes.append(f"Hotspot axes: {', '.join(hotspot_axes)}")
    
    for axis, risk in sorted_risks[:3]:  # Top 3 risks
        neutral_notes.append(f"{axis}: {risk:.2f}")
    
    return {
        "schema_version": "1.0.0",
        "tensor_norm": tensor_norm,
        "uplift_risk_band": risk_band,
        "hotspot_axes": hotspot_axes,
        "risk_indicators": risk_indicators,
        "neutral_notes": neutral_notes,
    }


def predict_instability_window(
    safety_tensors: Sequence[Dict[str, Any]],
    current_cycle: int,
    cycle_length_days: float = 1.0,
) -> Dict[str, Any]:
    """
    Predict instability windows based on historical safety tensor trends.
    
    PHASE VI — NOT RUN IN PHASE I
    No uplift claims are made.
    Deterministic execution guaranteed.
    
    Args:
        safety_tensors: Sequence of historical safety tensors
        current_cycle: Current cycle/version number
        cycle_length_days: Average length of a cycle in days
    
    Returns:
        {
            "schema_version": "1.0.0",
            "predicted_instability_cycles": List[int],
            "predicted_instability_days": List[float],
            "predicted_instability_versions": List[int],
            "confidence": float,
            "neutral_notes": List[str]
        }
    """
    if len(safety_tensors) < 2:
        return {
            "schema_version": "1.0.0",
            "predicted_instability_cycles": [],
            "predicted_instability_days": [],
            "predicted_instability_versions": [],
            "confidence": 0.0,
            "neutral_notes": ["Insufficient historical data for prediction"],
        }
    
    # Extract risk bands and norms from historical data
    risk_bands = [t.get("uplift_risk_band", "LOW") for t in safety_tensors]
    norms = [t.get("tensor_norm", 0.0) for t in safety_tensors]
    
    # Map risk bands to numeric values
    risk_map = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}
    risk_values = [risk_map.get(band, 1) for band in risk_bands]
    
    # Simple trend analysis: detect increasing risk
    predicted_cycles: List[int] = []
    predicted_days: List[float] = []
    predicted_versions: List[int] = []
    
    # If recent trend shows increasing risk, predict instability
    if len(risk_values) >= 3:
        recent_trend = risk_values[-3:]
        if recent_trend[-1] > recent_trend[0]:
            # Risk is increasing, predict instability in next 1-3 cycles
            for offset in [1, 2, 3]:
                predicted_cycles.append(current_cycle + offset)
                predicted_days.append((current_cycle + offset) * cycle_length_days)
                predicted_versions.append(current_cycle + offset)
    
    # If norm is increasing, also predict instability
    if len(norms) >= 3:
        recent_norms = norms[-3:]
        if recent_norms[-1] > recent_norms[0] * 1.2:  # 20% increase
            # Add additional predictions if not already present
            for offset in [1, 2]:
                cycle = current_cycle + offset
                if cycle not in predicted_cycles:
                    predicted_cycles.append(cycle)
                    predicted_days.append(cycle * cycle_length_days)
                    predicted_versions.append(cycle)
    
    # Calculate confidence based on data quality and trend strength
    confidence = 0.0
    if len(safety_tensors) >= 5:
        confidence = 0.7  # Good historical data
    elif len(safety_tensors) >= 3:
        confidence = 0.5  # Moderate historical data
    else:
        confidence = 0.3  # Limited historical data
    
    # Adjust confidence based on trend strength
    if predicted_cycles:
        # Strong trend = higher confidence
        if len(predicted_cycles) >= 2:
            confidence = min(0.9, confidence + 0.1)
    
    # Build neutral notes
    neutral_notes: List[str] = []
    neutral_notes.append(f"Analyzed {len(safety_tensors)} historical tensor(s)")
    
    if predicted_cycles:
        neutral_notes.append(f"Predicted instability in {len(predicted_cycles)} cycle(s)")
        neutral_notes.append(f"Confidence: {confidence:.1%}")
    else:
        neutral_notes.append("No instability windows predicted")
        neutral_notes.append("Current trend does not indicate instability")
    
    return {
        "schema_version": "1.0.0",
        "predicted_instability_cycles": sorted(predicted_cycles),
        "predicted_instability_days": sorted(predicted_days),
        "predicted_instability_versions": sorted(predicted_versions),
        "confidence": confidence,
        "neutral_notes": neutral_notes,
    }


def build_uplift_stability_forecaster(
    safety_tensors: Sequence[Dict[str, Any]],
    current_cycle: int,
    cycle_length_days: float = 1.0,
) -> Dict[str, Any]:
    """
    Build uplift stability forecaster from historical safety tensors.
    
    PHASE VI — NOT RUN IN PHASE I
    No uplift claims are made.
    Deterministic execution guaranteed.
    
    This is a wrapper around predict_instability_window with additional
    stability metrics and trend analysis.
    
    Args:
        safety_tensors: Sequence of historical safety tensors
        current_cycle: Current cycle/version number
        cycle_length_days: Average length of a cycle in days
    
    Returns:
        {
            "schema_version": "1.0.0",
            "current_stability": "STABLE" | "UNSTABLE" | "DEGRADING",
            "stability_trend": "IMPROVING" | "STABLE" | "DEGRADING",
            "instability_prediction": Dict[str, Any],
            "neutral_notes": List[str]
        }
    """
    if not safety_tensors:
        return {
            "schema_version": "1.0.0",
            "current_stability": "UNKNOWN",
            "stability_trend": "UNKNOWN",
            "instability_prediction": {},
            "neutral_notes": ["No safety tensor data available"],
        }
    
    # Get current stability from latest tensor
    latest_tensor = safety_tensors[-1]
    current_risk_band = latest_tensor.get("uplift_risk_band", "LOW")
    
    if current_risk_band == "HIGH":
        current_stability = "UNSTABLE"
    elif current_risk_band == "MEDIUM":
        current_stability = "DEGRADING"
    else:
        current_stability = "STABLE"
    
    # Determine stability trend
    if len(safety_tensors) >= 2:
        risk_bands = [t.get("uplift_risk_band", "LOW") for t in safety_tensors]
        risk_map = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}
        risk_values = [risk_map.get(band, 1) for band in risk_bands]
        
        first_risk = risk_values[0]
        last_risk = risk_values[-1]
        
        if last_risk > first_risk:
            stability_trend = "DEGRADING"
        elif last_risk < first_risk:
            stability_trend = "IMPROVING"
        else:
            stability_trend = "STABLE"
    else:
        stability_trend = "UNKNOWN"
    
    # Get instability prediction
    instability_prediction = predict_instability_window(
        safety_tensors, current_cycle, cycle_length_days
    )
    
    # Build neutral notes
    neutral_notes: List[str] = []
    neutral_notes.append(f"Current stability: {current_stability}")
    neutral_notes.append(f"Stability trend: {stability_trend}")
    
    if instability_prediction.get("predicted_instability_cycles"):
        cycles = instability_prediction["predicted_instability_cycles"]
        neutral_notes.append(f"Instability predicted in cycles: {cycles}")
    
    return {
        "schema_version": "1.0.0",
        "current_stability": current_stability,
        "stability_trend": stability_trend,
        "instability_prediction": instability_prediction,
        "neutral_notes": neutral_notes,
    }


def compute_maas_uplift_gate_v3(
    safety_tensor: Dict[str, Any],
    stability_forecaster: Dict[str, Any],
    additional_gates: Optional[Dict[str, Any]] = None,
    epistemic_eval: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    MAAS Uplift Gate v3: Final uplift safety decision.
    
    PHASE VI — NOT RUN IN PHASE I
    No uplift claims are made.
    Deterministic execution guaranteed.
    
    Integrates all pillars (safety tensor, stability forecaster, additional gates)
    to compute a final uplift safety decision.
    
    Phase V Integration:
    - If epistemic_eval is provided, uses compose_abstention_with_uplift_decision
      to combine epistemic gate with uplift evaluation (epistemic gate has veto power).
    - Logs governance tile for observability (shadow mode).
    
    Args:
        safety_tensor: Global uplift safety tensor
        stability_forecaster: Uplift stability forecaster output
        additional_gates: Optional additional gate results
        epistemic_eval: Optional epistemic abstention evaluation
    
    Returns:
        {
            "schema_version": "1.0.0",
            "gate_version": "v3",
            "uplift_safety_decision": "PASS" | "WARN" | "BLOCK",
            "decision_rationale": List[str],
            "risk_band": str,
            "stability_status": str,
            "neutral_notes": List[str],
            "governance_tile": Optional[Dict]  # If epistemic_eval provided
        }
    
    Decision Logic:
        - BLOCK: Risk band HIGH OR stability UNSTABLE OR predicted instability
        - WARN: Risk band MEDIUM OR stability DEGRADING
        - PASS: Otherwise
        - If epistemic_eval provided: Epistemic gate can upgrade/block decision
    """
    risk_band = safety_tensor.get("uplift_risk_band", "LOW")
    current_stability = stability_forecaster.get("current_stability", "STABLE")
    stability_trend = stability_forecaster.get("stability_trend", "STABLE")
    instability_prediction = stability_forecaster.get("instability_prediction", {})
    predicted_cycles = instability_prediction.get("predicted_instability_cycles", [])
    
    # Check additional gates if provided
    additional_block = False
    additional_warn = False
    
    if additional_gates:
        for gate_name, gate_result in additional_gates.items():
            if isinstance(gate_result, dict):
                gate_status = gate_result.get("status", "PASS")
                if gate_status in ["BLOCK", "FAIL", "CRITICAL"]:
                    additional_block = True
                elif gate_status in ["WARN", "WARNING", "ATTENTION"]:
                    additional_warn = True
    
    # Determine final decision
    decision_rationale: List[str] = []
    
    if risk_band == "HIGH" or current_stability == "UNSTABLE" or predicted_cycles or additional_block:
        decision = "BLOCK"
        if risk_band == "HIGH":
            decision_rationale.append(f"Risk band is HIGH")
        if current_stability == "UNSTABLE":
            decision_rationale.append(f"Current stability is UNSTABLE")
        if predicted_cycles:
            decision_rationale.append(f"Instability predicted in cycles: {predicted_cycles}")
        if additional_block:
            decision_rationale.append("Additional gates indicate BLOCK")
    elif risk_band == "MEDIUM" or current_stability == "DEGRADING" or additional_warn:
        decision = "WARN"
        if risk_band == "MEDIUM":
            decision_rationale.append(f"Risk band is MEDIUM")
        if current_stability == "DEGRADING":
            decision_rationale.append(f"Stability trend is DEGRADING")
        if additional_warn:
            decision_rationale.append("Additional gates indicate WARN")
    else:
        decision = "PASS"
        decision_rationale.append("All safety indicators within acceptable ranges")
    
    # Build neutral notes
    neutral_notes: List[str] = []
    neutral_notes.append(f"Uplift safety decision: {decision}")
    neutral_notes.append(f"Risk band: {risk_band}")
    neutral_notes.append(f"Stability: {current_stability} ({stability_trend})")
    
    if decision_rationale:
        neutral_notes.append("Decision rationale:")
        for rationale in decision_rationale:
            neutral_notes.append(f"  - {rationale}")
    
    # Phase V: Compose with epistemic evaluation if provided
    governance_tile = None
    final_decision = decision
    
    if epistemic_eval is not None:
        try:
            from rfl.verification import compose_abstention_with_uplift_decision
            from rfl.verification.governance_tile import build_uplift_governance_tile
            
            # Build uplift_eval dict from current decision
            uplift_eval = {
                "uplift_safety_decision": decision,
                "decision_rationale": decision_rationale,
                "risk_band": risk_band,
                "blocking_slices": [],  # Uplift gate doesn't track slices
            }
            
            # Compose with epistemic evaluation
            combined = compose_abstention_with_uplift_decision(
                epistemic_eval=epistemic_eval,
                uplift_eval=uplift_eval,
            )
            
            # Update final decision based on composition
            final_decision_map = {
                "OK": "PASS",
                "WARN": "WARN",
                "BLOCK": "BLOCK",
            }
            final_decision = final_decision_map.get(combined.get("final_status", "OK"), decision)
            
            # Build governance tile
            governance_tile = build_uplift_governance_tile(combined)
            
            # Log governance tile (shadow mode - observational only)
            logger.info(
                f"Uplift governance tile: final_status={governance_tile['final_status']}, "
                f"epistemic_upgrade={governance_tile['epistemic_upgrade_applied']}, "
                f"blocking_slices={governance_tile['blocking_slices']}"
            )
            
            # Add epistemic upgrade note if applied
            if combined.get("epistemic_upgrade_applied", False):
                neutral_notes.append(
                    f"Epistemic gate applied: upgraded from {decision} to {final_decision}"
                )
                if combined.get("advisory"):
                    neutral_notes.append(f"Advisory: {combined['advisory']}")
        
        except ImportError:
            # Gracefully degrade if abstention module not available
            logger.warning("Abstention verification module not available, skipping epistemic composition")
        except Exception as e:
            # Shadow mode: never fail the gate due to abstention integration issues
            logger.warning(f"Error composing with epistemic evaluation: {e}, using uplift decision only")
    
    return {
        "schema_version": "1.0.0",
        "gate_version": "v3",
        "uplift_safety_decision": final_decision,
        "decision_rationale": decision_rationale,
        "risk_band": risk_band,
        "stability_status": current_stability,
        "stability_trend": stability_trend,
        "neutral_notes": neutral_notes,
        "governance_tile": governance_tile,  # None if epistemic_eval not provided
    }

