"""
Budget-aware modulation for calibration experiments.

Provides functions to compute budget confounding and effective learning rate
adjustments for P5 drift-modulation analysis.

Also provides calibration exclusion recommendations based on cross-signal checks:
- Budget confounded
- AND PRNG not volatile
- AND topology stable
"""

from typing import Dict, Any, List, Optional


def annotate_calibration_windows_with_budget_modulation(
    calibration_windows: List[Dict[str, Any]],
    budget_cross_view: Dict[str, Any],
    slice_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Annotate calibration windows with budget-aware modulation fields.
    
    Adds budget_confounded, effective_lr_adjustment, drift_classification,
    and budget_health_during_window to each calibration window.
    
    This enables P5 drift-modulation analysis to distinguish resource-driven
    drift from model inadequacy.
    
    Args:
        calibration_windows: List of calibration window dictionaries
            Format: [{"start_cycle": int, "end_cycle": int, ...}, ...]
        budget_cross_view: Budget health view (from budget_integration)
            Format: {"slices": [{"slice_name": str, "health_status": "SAFE|TIGHT|STARVED", ...}]}
        slice_name: Name of slice for these calibration windows (optional)
    
    Returns:
        List of annotated calibration windows with budget modulation fields
    """
    from experiments.uplift_council import compute_budget_modulation_for_calibration_window
    
    annotated = []
    for window in calibration_windows:
        # Use provided slice_name or try to extract from window
        window_slice = slice_name or window.get("slice_name")
        
        if window_slice:
            modulation = compute_budget_modulation_for_calibration_window(
                window=window,
                budget_cross_view=budget_cross_view,
                slice_name=window_slice,
            )
            
            # Create annotated window
            annotated_window = dict(window)
            annotated_window.update(modulation)
            annotated.append(annotated_window)
        else:
            # No slice name available, add window without modulation
            annotated.append(window)
    
    return annotated


def compute_calibration_exclusion_recommendation(
    budget_modulation: Dict[str, Any],
    prng_signal: Optional[Dict[str, Any]] = None,
    topology_signal: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Compute calibration exclusion recommendation based on cross-signal checks.
    
    Exclusion is recommended ONLY if ALL of the following are true:
    1. Budget is confounded (budget_confounded = True)
    2. PRNG is NOT volatile (drift_status != "VOLATILE" or missing)
    3. Topology is stable (pressure_band != "HIGH" or missing)
    
    This ensures we only exclude windows when budget is the primary confounding
    factor, not when multiple signals indicate instability.
    
    Args:
        budget_modulation: Output from compute_budget_modulation_for_calibration_window()
            Must include: budget_confounded, drift_classification
        prng_signal: Optional PRNG signal dictionary
            Format: {"drift_status": "STABLE" | "DRIFTING" | "VOLATILE", ...}
        topology_signal: Optional topology signal dictionary
            Format: {"pressure_band": "LOW" | "MEDIUM" | "HIGH", ...}
    
    Returns:
        Exclusion recommendation dictionary:
        {
            "calibration_exclusion_recommended": bool,
            "exclusion_reason": "BUDGET_CONFOUNDED_TRANSIENT" | "BUDGET_CONFOUNDED_PERSISTENT" | None,
            "cross_signal_checks": {
                "budget_confounded": bool,
                "prng_not_volatile": bool,
                "topology_stable": bool,
            },
            "exclusion_trace": {
                "missing_signal_policy": "DEFAULT_TRUE_MISSING",
                "checks": {
                    "budget_confounded": {"value": bool, "source": "budget_modulation", "raw_value": str},
                    "prng_not_volatile": {"value": bool, "source": str, "raw_value": str},
                    "topology_stable": {"value": bool, "source": str, "raw_value": str},
                },
                "decision": bool,
                "reason": str | None,
                "thresholds": {
                    "prng_volatile_threshold": "VOLATILE",
                    "topology_high_pressure_threshold": "HIGH",
                },
            },
        }
    """
    budget_confounded = budget_modulation.get("budget_confounded", False)
    drift_classification = budget_modulation.get("drift_classification", "NONE")
    
    # Check PRNG volatility with trace
    prng_drift_status = prng_signal.get("drift_status", "UNKNOWN") if prng_signal else "UNKNOWN"
    prng_not_volatile = prng_drift_status != "VOLATILE"
    prng_source = "prng_signal" if prng_signal else "DEFAULT_TRUE_MISSING"
    
    # Check topology stability with trace
    topology_pressure = topology_signal.get("pressure_band", "UNKNOWN") if topology_signal else "UNKNOWN"
    topology_stable = topology_pressure != "HIGH"
    topology_source = "topology_signal" if topology_signal else "DEFAULT_TRUE_MISSING"
    
    # Cross-signal check: recommend exclusion only if ALL conditions met
    calibration_exclusion_recommended = (
        budget_confounded
        and prng_not_volatile
        and topology_stable
    )
    
    # Determine exclusion reason
    exclusion_reason = None
    if calibration_exclusion_recommended:
        if drift_classification == "TRANSIENT":
            exclusion_reason = "BUDGET_CONFOUNDED_TRANSIENT"
        elif drift_classification == "PERSISTENT":
            exclusion_reason = "BUDGET_CONFOUNDED_PERSISTENT"
    
    # Build exclusion trace for auditability
    # Use sorted keys for deterministic JSON serialization
    missing_signal_policy = "DEFAULT_TRUE_MISSING"
    
    exclusion_trace = {
        "missing_signal_policy": missing_signal_policy,
        "checks": {
            "budget_confounded": {
                "value": budget_confounded,
                "source": "budget_modulation",
                "raw_value": str(budget_confounded),
            },
            "prng_not_volatile": {
                "value": prng_not_volatile,
                "source": prng_source,
                "raw_value": prng_drift_status,
            },
            "topology_stable": {
                "value": topology_stable,
                "source": topology_source,
                "raw_value": topology_pressure,
            },
        },
        "decision": calibration_exclusion_recommended,
        "reason": exclusion_reason,
        "thresholds": {
            "prng_volatile_threshold": "VOLATILE",
            "topology_high_pressure_threshold": "HIGH",
        },
    }
    
    return {
        "calibration_exclusion_recommended": calibration_exclusion_recommended,
        "exclusion_reason": exclusion_reason,
        "cross_signal_checks": {
            "budget_confounded": budget_confounded,
            "prng_not_volatile": prng_not_volatile,
            "topology_stable": topology_stable,
        },
        "exclusion_trace": exclusion_trace,
    }


def annotate_calibration_windows_with_exclusion_recommendations(
    calibration_windows: List[Dict[str, Any]],
    prng_signal: Optional[Dict[str, Any]] = None,
    topology_signal: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Annotate calibration windows with exclusion recommendations.
    
    Adds calibration_exclusion_recommended and exclusion_reason to windows
    that already have budget modulation fields.
    
    Args:
        calibration_windows: List of calibration windows (should already have
            budget modulation fields from annotate_calibration_windows_with_budget_modulation)
        prng_signal: Optional PRNG signal for cross-check
        topology_signal: Optional topology signal for cross-check
    
    Returns:
        List of windows with exclusion recommendations added
    """
    annotated = []
    for window in calibration_windows:
        # Check if window has budget modulation fields
        if "budget_confounded" in window:
            budget_modulation = {
                "budget_confounded": window.get("budget_confounded", False),
                "drift_classification": window.get("drift_classification", "NONE"),
            }
            
            exclusion = compute_calibration_exclusion_recommendation(
                budget_modulation=budget_modulation,
                prng_signal=prng_signal,
                topology_signal=topology_signal,
            )
            
            # Add exclusion fields to window
            annotated_window = dict(window)
            annotated_window.update(exclusion)
            annotated.append(annotated_window)
        else:
            # No budget modulation, no exclusion recommendation
            # Still provide trace for auditability
            prng_drift_status = prng_signal.get("drift_status", "UNKNOWN") if prng_signal else "UNKNOWN"
            topology_pressure = topology_signal.get("pressure_band", "UNKNOWN") if topology_signal else "UNKNOWN"
            prng_source = "prng_signal" if prng_signal else "DEFAULT_TRUE_MISSING"
            topology_source = "topology_signal" if topology_signal else "DEFAULT_TRUE_MISSING"
            
            annotated_window = dict(window)
            annotated_window["calibration_exclusion_recommended"] = False
            annotated_window["exclusion_reason"] = None
            annotated_window["cross_signal_checks"] = {
                "budget_confounded": False,
                "prng_not_volatile": True,  # Default to True if no signal
                "topology_stable": True,     # Default to True if no signal
            }
            annotated_window["exclusion_trace"] = {
                "missing_signal_policy": "DEFAULT_TRUE_MISSING",
                "checks": {
                    "budget_confounded": {
                        "value": False,
                        "source": "budget_modulation",
                        "raw_value": "false",
                    },
                    "prng_not_volatile": {
                        "value": True,
                        "source": prng_source,
                        "raw_value": prng_drift_status,
                    },
                    "topology_stable": {
                        "value": True,
                        "source": topology_source,
                        "raw_value": topology_pressure,
                    },
                },
                "decision": False,
                "reason": None,
                "thresholds": {
                    "prng_volatile_threshold": "VOLATILE",
                    "topology_high_pressure_threshold": "HIGH",
                },
            }
            annotated.append(annotated_window)
    
    return annotated


def demonstrate_budget_confounding_example() -> Dict[str, Any]:
    """
    Worked example demonstrating how budget confounding can falsely inflate divergence.
    
    Returns:
        Example dictionary showing:
        - Calibration window with high divergence
        - Budget modulation analysis
        - Adjusted divergence interpretation
    """
    # Example: Calibration window with high divergence
    calibration_window = {
        "start_cycle": 0,
        "end_cycle": 50,
        "divergence_rate": 0.12,
        "mean_delta_p": 0.12,
        "delta_bias": 0.08,
        "delta_variance": 0.04,
    }
    
    # Budget status: STARVED with 15% exhaustion
    budget_cross_view = {
        "slices": [
            {
                "slice_name": "slice_uplift_goal",
                "health_status": "STARVED",
                "frequently_starved": False,  # Transient, not persistent
                "budget_exhausted_pct": 15.0,
                "timeout_abstentions_avg": 1.2,
            }
        ]
    }
    
    from experiments.uplift_council import compute_budget_modulation_for_calibration_window
    
    modulation = compute_budget_modulation_for_calibration_window(
        window=calibration_window,
        budget_cross_view=budget_cross_view,
        slice_name="slice_uplift_goal",
    )
    
    # Compute adjusted divergence
    original_divergence = calibration_window["mean_delta_p"]
    # Apply severity multiplier from Budget_PhaseX_Doctrine.md Section 3.3.1
    # VOLATILE budget → 0.4 multiplier
    adjusted_divergence = original_divergence * 0.4
    
    return {
        "original_calibration_window": calibration_window,
        "budget_status": budget_cross_view["slices"][0],
        "budget_modulation": modulation,
        "divergence_interpretation": {
            "original_delta_p": original_divergence,
            "original_severity": "WARN",  # 0.12 is in WARN range (0.05-0.15)
            "adjusted_delta_p": adjusted_divergence,
            "adjusted_severity": "INFO",  # 0.048 is in INFO range (<0.05)
            "interpretation": "Divergence is primarily due to budget constraints, not model inadequacy. "
                             "Twin model calibration should NOT be adjusted based on this window.",
        },
        "calibration_recommendation": {
            "exclude_from_calibration": True,
            "reason": "budget_confounded",
            "note": "Window shows TRANSIENT budget-driven drift. Twin model parameters are correct; "
                   "divergence is due to real runner resource constraints.",
        },
    }



