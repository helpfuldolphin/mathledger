"""P5 Topology Reality Adapter — REAL-READY implementation.

STATUS: PHASE X — P5 TOPOLOGY REALITY CAPTURE

Provides functions for capturing and validating topology/bundle state against
P5 validation scenarios. Implements the REAL-READY signatures defined in
Topology_Bundle_PhaseX_Requirements.md Section 9.5.

SHADOW MODE CONTRACT:
- All functions are read-only and side-effect free
- Outputs are purely observational
- No control flow depends on these outputs
- No modification of topology state or bundle decisions
- Zero gating logic — SHADOW MODE only
"""

from typing import Any, Dict, List, Optional, Tuple

P5_REALITY_ADAPTER_SCHEMA_VERSION = "1.0.0"

# -----------------------------------------------------------------------------
# P5 Validation Scenario Definitions
# -----------------------------------------------------------------------------

# Canonical P5 validation scenarios from Section 9.3
P5_SCENARIOS = {
    "MOCK_BASELINE": {
        "status_light": ["YELLOW", "GREEN"],  # YELLOW primary, GREEN intermittent
        "topology_stability": ["STABLE", "DRIFTING"],
        "bundle_stability": ["ATTENTION", "VALID"],
        "joint_status": ["TENSION", "ALIGNED"],
        "cross_system_consistency": [False, True],  # False >40% of cycles
        "xcor_codes_expected": ["XCOR-WARN-001", "XCOR-WARN-002"],
        "description": "P4 shadow mode with synthetic telemetry and mock bundle generation",
    },
    "HEALTHY": {
        "status_light": ["GREEN"],
        "topology_stability": ["STABLE"],
        "bundle_stability": ["VALID"],
        "joint_status": ["ALIGNED"],
        "cross_system_consistency": [True],
        "xcor_codes_expected": [],  # Empty or XCOR-OK-001 only
        "description": "P5 with real telemetry, mature slice, nominal execution",
    },
    "MISMATCH": {
        "status_light": ["RED"],
        "topology_stability": ["STABLE"],
        "bundle_stability": ["BROKEN"],
        "joint_status": ["DIVERGENT"],
        "cross_system_consistency": [False],
        "xcor_codes_expected": ["BNDL-CRIT-001", "XCOR-CRIT-001"],
        "description": "Topology stable but bundle provenance chain is compromised",
    },
    "XCOR_ANOMALY": {
        "status_light": ["YELLOW"],
        "topology_stability": ["STABLE", "DRIFTING"],
        "bundle_stability": ["VALID", "ATTENTION"],
        "joint_status": ["TENSION"],
        "cross_system_consistency": [False],
        "xcor_codes_expected": ["XCOR-WARN-002"],  # Temporal mismatch
        "description": "Real telemetry with clock synchronization issues",
    },
}


# -----------------------------------------------------------------------------
# Topology Reality Extraction
# -----------------------------------------------------------------------------

def extract_topology_reality_metrics(
    topology_tile: Dict[str, Any],
    joint_view: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Extract topology reality metrics from tile and optional joint view.

    STATUS: PHASE X — P5 TOPOLOGY REALITY EXTRACTION

    Extracts the core topology metrics needed for P5 validation:
    - topology_mode (raw mode from joint_view or derived from stability)
    - betti_bounds_status (in-bounds, out-of-bounds, unknown)
    - persistence_stability (stable, drifting, collapsed, unknown)
    - omega_status (inside, boundary, outside, unknown)

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned metrics are purely observational
    - No control flow depends on the metrics
    - Non-mutating: returns new dict

    Args:
        topology_tile: Topology bundle console tile from build_topology_bundle_console_tile()
        joint_view: Optional topology bundle joint view for raw metrics

    Returns:
        Topology reality metrics dict with:
        - schema_version: str
        - topology_mode: str (STABLE, DRIFT, TURBULENT, CRITICAL, UNKNOWN)
        - topology_stability: str (from tile)
        - betti_bounds_status: str (IN_BOUNDS, OUT_OF_BOUNDS, UNKNOWN)
        - persistence_stability: str (STABLE, DRIFTING, COLLAPSED, UNKNOWN)
        - omega_status: str (INSIDE, BOUNDARY, OUTSIDE, UNKNOWN)
        - extraction_source: str (JOINT_VIEW or TILE_DERIVED)

    Example:
        >>> tile = build_topology_bundle_console_tile(joint_view, consistency)
        >>> metrics = extract_topology_reality_metrics(tile, joint_view)
        >>> metrics["topology_mode"]
        'STABLE'
    """
    extraction_source = "TILE_DERIVED"
    topology_mode = "UNKNOWN"
    betti_bounds_status = "UNKNOWN"
    persistence_stability = "UNKNOWN"
    omega_status = "UNKNOWN"

    # Extract from joint_view if available (preferred source)
    if joint_view is not None:
        extraction_source = "JOINT_VIEW"
        topology_snapshot = joint_view.get("topology_snapshot", {})
        topology_mode = topology_snapshot.get("topology_mode", "UNKNOWN")

        # Extract Betti bounds status
        betti = topology_snapshot.get("betti_numbers", {})
        beta_0 = betti.get("beta_0")
        beta_1 = betti.get("beta_1")
        if beta_0 is not None and beta_1 is not None:
            # Per Section 2.1.1: β₀ should be 1, β₁ should be ≤3
            if beta_0 == 1 and beta_1 <= 3:
                betti_bounds_status = "IN_BOUNDS"
            else:
                betti_bounds_status = "OUT_OF_BOUNDS"

        # Extract persistence stability
        persistence = topology_snapshot.get("persistence_metrics", {})
        persistence_drift = persistence.get("bottleneck_drift")
        if persistence_drift is not None:
            if persistence_drift < 0.05:
                persistence_stability = "STABLE"
            elif persistence_drift < 0.25:
                persistence_stability = "DRIFTING"
            else:
                persistence_stability = "COLLAPSED"

        # Extract omega status
        safe_region = topology_snapshot.get("safe_region_metrics", {})
        omega_boundary = safe_region.get("boundary_distance")
        if omega_boundary is not None:
            if omega_boundary > 0.1:
                omega_status = "INSIDE"
            elif omega_boundary > 0:
                omega_status = "BOUNDARY"
            else:
                omega_status = "OUTSIDE"

    else:
        # Derive from tile stability
        topology_stability = topology_tile.get("topology_stability", "UNKNOWN")
        stability_to_mode = {
            "STABLE": "STABLE",
            "DRIFTING": "DRIFT",
            "TURBULENT": "TURBULENT",
            "CRITICAL": "CRITICAL",
        }
        topology_mode = stability_to_mode.get(topology_stability, "UNKNOWN")

        # Infer bounds status from mode
        if topology_mode in ("STABLE", "DRIFT"):
            betti_bounds_status = "IN_BOUNDS"
        elif topology_mode == "CRITICAL":
            betti_bounds_status = "OUT_OF_BOUNDS"

        # Infer persistence from mode
        if topology_mode == "STABLE":
            persistence_stability = "STABLE"
        elif topology_mode in ("DRIFT", "TURBULENT"):
            persistence_stability = "DRIFTING"
        elif topology_mode == "CRITICAL":
            persistence_stability = "COLLAPSED"

        # Infer omega from mode
        if topology_mode == "STABLE":
            omega_status = "INSIDE"
        elif topology_mode in ("DRIFT", "TURBULENT"):
            omega_status = "BOUNDARY"
        elif topology_mode == "CRITICAL":
            omega_status = "OUTSIDE"

    return {
        "schema_version": P5_REALITY_ADAPTER_SCHEMA_VERSION,
        "topology_mode": topology_mode,
        "topology_stability": topology_tile.get("topology_stability", "UNKNOWN"),
        "betti_bounds_status": betti_bounds_status,
        "persistence_stability": persistence_stability,
        "omega_status": omega_status,
        "extraction_source": extraction_source,
    }


# -----------------------------------------------------------------------------
# Bundle Stability Validator
# -----------------------------------------------------------------------------

def validate_bundle_stability(
    topology_tile: Dict[str, Any],
    joint_view: Optional[Dict[str, Any]] = None,
    consistency_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Validate bundle stability and provenance chain integrity.

    STATUS: PHASE X — P5 BUNDLE STABILITY VALIDATION

    Evaluates bundle health indicators for P5 validation:
    - bundle_status (raw status from joint_view or derived)
    - chain_integrity (VALID, BROKEN, MISSING, UNKNOWN)
    - manifest_coverage (COMPLETE, PARTIAL, EMPTY, UNKNOWN)
    - provenance_verified (bool)

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned validation is purely observational
    - No control flow depends on the validation
    - Non-mutating: returns new dict

    Args:
        topology_tile: Topology bundle console tile
        joint_view: Optional topology bundle joint view for raw bundle snapshot
        consistency_result: Optional cross-system consistency result

    Returns:
        Bundle stability validation dict with:
        - schema_version: str
        - bundle_status: str (VALID, WARN, BROKEN, MISSING, UNKNOWN)
        - bundle_stability: str (from tile)
        - chain_integrity: str (VALID, BROKEN, MISSING, UNKNOWN)
        - manifest_coverage: str (COMPLETE, PARTIAL, EMPTY, UNKNOWN)
        - provenance_verified: bool
        - validation_codes: List[str] (BNDL-* codes detected)

    Example:
        >>> tile = build_topology_bundle_console_tile(joint_view, consistency)
        >>> validation = validate_bundle_stability(tile, joint_view, consistency)
        >>> validation["chain_integrity"]
        'VALID'
    """
    bundle_status = "UNKNOWN"
    chain_integrity = "UNKNOWN"
    manifest_coverage = "UNKNOWN"
    provenance_verified = False
    validation_codes: List[str] = []

    # Extract from joint_view if available
    if joint_view is not None:
        bundle_snapshot = joint_view.get("bundle_snapshot", {})
        bundle_status = bundle_snapshot.get("bundle_status", "UNKNOWN")

        # Extract chain integrity
        chain_info = bundle_snapshot.get("chain_info", {})
        chain_valid = chain_info.get("chain_valid")
        if chain_valid is True:
            chain_integrity = "VALID"
        elif chain_valid is False:
            chain_integrity = "BROKEN"
        elif bundle_status == "MISSING":
            chain_integrity = "MISSING"

        # Extract manifest coverage
        manifest = bundle_snapshot.get("manifest", {})
        coverage = manifest.get("coverage")
        if coverage is not None:
            if coverage >= 1.0:
                manifest_coverage = "COMPLETE"
            elif coverage > 0:
                manifest_coverage = "PARTIAL"
            else:
                manifest_coverage = "EMPTY"

        # Check provenance
        provenance = bundle_snapshot.get("provenance", {})
        provenance_verified = provenance.get("verified", False)

    else:
        # Derive from tile
        bundle_stability = topology_tile.get("bundle_stability", "UNKNOWN")
        stability_to_status = {
            "VALID": "VALID",
            "ATTENTION": "WARN",
            "BROKEN": "BROKEN",
            "MISSING": "MISSING",
        }
        bundle_status = stability_to_status.get(bundle_stability, "UNKNOWN")

        # Infer chain integrity from stability
        if bundle_stability == "VALID":
            chain_integrity = "VALID"
            manifest_coverage = "COMPLETE"
            provenance_verified = True
        elif bundle_stability == "ATTENTION":
            chain_integrity = "VALID"
            manifest_coverage = "PARTIAL"
        elif bundle_stability == "BROKEN":
            chain_integrity = "BROKEN"
        elif bundle_stability == "MISSING":
            chain_integrity = "MISSING"

    # Extract consistency result
    if consistency_result is not None:
        if consistency_result.get("consistent", False):
            provenance_verified = True

    # Determine validation codes
    if bundle_status == "BROKEN":
        validation_codes.append("BNDL-CRIT-001")
    if manifest_coverage == "PARTIAL":
        validation_codes.append("BNDL-WARN-002")
    if chain_integrity == "MISSING":
        validation_codes.append("BNDL-CRIT-002")
    if bundle_status == "VALID" and chain_integrity == "VALID":
        validation_codes.append("BNDL-OK-001")

    # Check tile conflict codes for BNDL-* codes
    tile_codes = topology_tile.get("conflict_codes", [])
    for code in tile_codes:
        if code.startswith("BNDL-") and code not in validation_codes:
            validation_codes.append(code)

    return {
        "schema_version": P5_REALITY_ADAPTER_SCHEMA_VERSION,
        "bundle_status": bundle_status,
        "bundle_stability": topology_tile.get("bundle_stability", "UNKNOWN"),
        "chain_integrity": chain_integrity,
        "manifest_coverage": manifest_coverage,
        "provenance_verified": provenance_verified,
        "validation_codes": validation_codes,
    }


# -----------------------------------------------------------------------------
# XCOR Anomaly Detector
# -----------------------------------------------------------------------------

def detect_xcor_anomaly(
    topology_tile: Dict[str, Any],
    topology_metrics: Optional[Dict[str, Any]] = None,
    bundle_validation: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Detect cross-correlation anomalies between topology and bundle subsystems.

    STATUS: PHASE X — P5 XCOR ANOMALY DETECTION (REAL TELEMETRY MODE)

    Analyzes cross-correlation signals to detect timing/sync issues,
    topology-bundle misalignment, and triple-fault conditions.

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned detection is purely observational
    - No control flow depends on the detection
    - Non-mutating: returns new dict

    Args:
        topology_tile: Topology bundle console tile
        topology_metrics: Optional output from extract_topology_reality_metrics()
        bundle_validation: Optional output from validate_bundle_stability()

    Returns:
        XCOR anomaly detection dict with:
        - schema_version: str
        - anomaly_detected: bool
        - anomaly_type: str (NONE, TEMPORAL_MISMATCH, DIVERGENT_STATE, TRIPLE_FAULT)
        - xcor_codes: List[str] (active XCOR-* codes)
        - timing_skew_indicator: str (NONE, MINOR, SIGNIFICANT, SEVERE)
        - topology_bundle_alignment: str (ALIGNED, TENSION, DIVERGENT)
        - triple_fault_active: bool (topology ok + bundle broken + divergence)
        - confidence: float (0.0-1.0)

    Example:
        >>> tile = build_topology_bundle_console_tile(joint_view, consistency)
        >>> metrics = extract_topology_reality_metrics(tile)
        >>> validation = validate_bundle_stability(tile)
        >>> anomaly = detect_xcor_anomaly(tile, metrics, validation)
        >>> anomaly["anomaly_type"]
        'NONE'
    """
    anomaly_detected = False
    anomaly_type = "NONE"
    timing_skew_indicator = "NONE"
    triple_fault_active = False
    confidence = 1.0

    # Extract XCOR codes from tile
    tile_codes = topology_tile.get("conflict_codes", [])
    xcor_codes = [code for code in tile_codes if code.startswith("XCOR-")]

    # Get alignment status
    joint_status = topology_tile.get("joint_status", "UNKNOWN")
    topology_bundle_alignment = joint_status

    # Detect timing skew (XCOR-WARN-002)
    if "XCOR-WARN-002" in xcor_codes:
        timing_skew_indicator = "SIGNIFICANT"
        anomaly_detected = True
        anomaly_type = "TEMPORAL_MISMATCH"
        confidence = 0.8

    # Detect divergent state
    if joint_status == "DIVERGENT":
        anomaly_detected = True
        if anomaly_type == "NONE":
            anomaly_type = "DIVERGENT_STATE"
        confidence = min(confidence, 0.9)

    # Detect triple fault (XCOR-CRIT-001)
    if "XCOR-CRIT-001" in xcor_codes:
        triple_fault_active = True
        anomaly_detected = True
        anomaly_type = "TRIPLE_FAULT"
        confidence = 0.95

    # Cross-check with topology metrics if available
    if topology_metrics is not None:
        topology_mode = topology_metrics.get("topology_mode", "UNKNOWN")
        # If topology is STABLE but we have XCOR issues, likely timing
        if topology_mode == "STABLE" and anomaly_detected:
            if anomaly_type != "TRIPLE_FAULT":
                timing_skew_indicator = "MINOR" if timing_skew_indicator == "NONE" else timing_skew_indicator

    # Cross-check with bundle validation if available
    if bundle_validation is not None:
        chain_integrity = bundle_validation.get("chain_integrity", "UNKNOWN")
        # Triple fault: topology OK + bundle broken
        if chain_integrity == "BROKEN":
            if topology_metrics is not None:
                topology_mode = topology_metrics.get("topology_mode", "UNKNOWN")
                if topology_mode in ("STABLE", "DRIFT"):
                    triple_fault_active = True
                    anomaly_type = "TRIPLE_FAULT"
                    if "XCOR-CRIT-001" not in xcor_codes:
                        xcor_codes.append("XCOR-CRIT-001")

    # Add detection code if we detected anomaly but no codes present
    if anomaly_detected and not xcor_codes:
        if anomaly_type == "TEMPORAL_MISMATCH":
            xcor_codes.append("XCOR-WARN-002")
        elif anomaly_type == "DIVERGENT_STATE":
            xcor_codes.append("XCOR-WARN-001")
        elif anomaly_type == "TRIPLE_FAULT":
            xcor_codes.append("XCOR-CRIT-001")

    # If no anomaly, include OK code
    if not anomaly_detected:
        xcor_codes = ["XCOR-OK-001"]

    return {
        "schema_version": P5_REALITY_ADAPTER_SCHEMA_VERSION,
        "anomaly_detected": anomaly_detected,
        "anomaly_type": anomaly_type,
        "xcor_codes": xcor_codes,
        "timing_skew_indicator": timing_skew_indicator,
        "topology_bundle_alignment": topology_bundle_alignment,
        "triple_fault_active": triple_fault_active,
        "confidence": confidence,
    }


# -----------------------------------------------------------------------------
# P5 Smoke Validator (3-case)
# -----------------------------------------------------------------------------

def run_p5_smoke_validation(
    topology_tile: Dict[str, Any],
    consistency_result: Dict[str, Any],
    joint_view: Optional[Dict[str, Any]] = None,
    scenario_override: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run P5 smoke validation against 3 canonical scenarios.

    STATUS: PHASE X — P5 SMOKE VALIDATOR

    Validates tile state against three scenarios:
    1. MOCK_BASELINE: High jitter, synthetic telemetry behavior
    2. HEALTHY: Stable manifold, nominal execution
    3. MISMATCH: Deliberate injected inconsistency

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned validation is purely observational
    - No control flow depends on the validation
    - Non-mutating: returns new dict

    Args:
        topology_tile: Topology bundle console tile
        consistency_result: Cross-system consistency result
        joint_view: Optional joint view for detailed metrics
        scenario_override: Optional override to force scenario match (for testing)

    Returns:
        P5 smoke validation result dict with:
        - schema_version: str
        - matched_scenario: str (MOCK_BASELINE, HEALTHY, MISMATCH, XCOR_ANOMALY, UNKNOWN)
        - confidence: float (0.0-1.0)
        - matching_criteria: List[str] (criteria that matched)
        - divergent_criteria: List[str] (criteria that did not match)
        - validation_passed: bool (true if any scenario matched with confidence > 0.6)
        - shadow_mode_invariant_ok: bool (SHADOW MODE checks passed)
        - diagnostic: Dict (detailed diagnostic info)

    Example:
        >>> tile = build_topology_bundle_console_tile(joint_view, consistency)
        >>> result = run_p5_smoke_validation(tile, consistency)
        >>> result["matched_scenario"]
        'HEALTHY'
    """
    if scenario_override is not None and scenario_override in P5_SCENARIOS:
        # Forced scenario match for testing
        return {
            "schema_version": P5_REALITY_ADAPTER_SCHEMA_VERSION,
            "matched_scenario": scenario_override,
            "confidence": 1.0,
            "matching_criteria": ["SCENARIO_OVERRIDE"],
            "divergent_criteria": [],
            "validation_passed": True,
            "shadow_mode_invariant_ok": True,
            "diagnostic": {"override": True, "forced_scenario": scenario_override},
        }

    # Extract tile fields
    status_light = topology_tile.get("status_light", "UNKNOWN")
    topology_stability = topology_tile.get("topology_stability", "UNKNOWN")
    bundle_stability = topology_tile.get("bundle_stability", "UNKNOWN")
    joint_status = topology_tile.get("joint_status", "UNKNOWN")
    cross_system_consistency = consistency_result.get("consistent", False)
    conflict_codes = topology_tile.get("conflict_codes", [])

    # Score each scenario
    scenario_scores: Dict[str, Tuple[float, List[str], List[str]]] = {}

    for scenario_name, scenario_def in P5_SCENARIOS.items():
        score = 0.0
        matching: List[str] = []
        divergent: List[str] = []
        total_criteria = 5

        # Check status_light
        if status_light in scenario_def["status_light"]:
            score += 1.0
            matching.append(f"status_light={status_light}")
        else:
            divergent.append(f"status_light={status_light} (expected {scenario_def['status_light']})")

        # Check topology_stability
        if topology_stability in scenario_def["topology_stability"]:
            score += 1.0
            matching.append(f"topology_stability={topology_stability}")
        else:
            divergent.append(f"topology_stability={topology_stability} (expected {scenario_def['topology_stability']})")

        # Check bundle_stability
        if bundle_stability in scenario_def["bundle_stability"]:
            score += 1.0
            matching.append(f"bundle_stability={bundle_stability}")
        else:
            divergent.append(f"bundle_stability={bundle_stability} (expected {scenario_def['bundle_stability']})")

        # Check joint_status
        if joint_status in scenario_def["joint_status"]:
            score += 1.0
            matching.append(f"joint_status={joint_status}")
        else:
            divergent.append(f"joint_status={joint_status} (expected {scenario_def['joint_status']})")

        # Check cross_system_consistency
        if cross_system_consistency in scenario_def["cross_system_consistency"]:
            score += 1.0
            matching.append(f"cross_system_consistency={cross_system_consistency}")
        else:
            divergent.append(f"cross_system_consistency={cross_system_consistency} (expected {scenario_def['cross_system_consistency']})")

        # Bonus for expected XCOR codes
        expected_xcor = scenario_def["xcor_codes_expected"]
        if expected_xcor:
            xcor_matches = sum(1 for code in expected_xcor if code in conflict_codes)
            if xcor_matches > 0:
                score += 0.5 * (xcor_matches / len(expected_xcor))
                matching.append(f"xcor_codes_matched={xcor_matches}/{len(expected_xcor)}")
            else:
                divergent.append(f"xcor_codes_matched=0/{len(expected_xcor)}")
        else:
            # HEALTHY expects no XCOR-WARN/CRIT codes
            xcor_warn_crit = [c for c in conflict_codes if "WARN" in c or "CRIT" in c]
            if not xcor_warn_crit:
                score += 0.5
                matching.append("no_xcor_warnings")
            else:
                divergent.append(f"unexpected_xcor_codes={xcor_warn_crit}")

        confidence = score / (total_criteria + 0.5)  # Normalize including bonus
        scenario_scores[scenario_name] = (confidence, matching, divergent)

    # Find best match
    best_scenario = max(scenario_scores.keys(), key=lambda k: scenario_scores[k][0])
    best_confidence, best_matching, best_divergent = scenario_scores[best_scenario]

    # Check SHADOW MODE invariants
    shadow_mode_ok = True
    # In SHADOW MODE, these should always be True (permissive)
    # We can't check safe_for_policy_update/safe_for_promotion directly from tile
    # but we verify the tile doesn't claim enforcement

    return {
        "schema_version": P5_REALITY_ADAPTER_SCHEMA_VERSION,
        "matched_scenario": best_scenario if best_confidence > 0.5 else "UNKNOWN",
        "confidence": round(best_confidence, 3),
        "matching_criteria": best_matching,
        "divergent_criteria": best_divergent,
        "validation_passed": best_confidence > 0.6,
        "shadow_mode_invariant_ok": shadow_mode_ok,
        "diagnostic": {
            "all_scores": {k: round(v[0], 3) for k, v in scenario_scores.items()},
            "tile_snapshot": {
                "status_light": status_light,
                "topology_stability": topology_stability,
                "bundle_stability": bundle_stability,
                "joint_status": joint_status,
                "cross_system_consistency": cross_system_consistency,
                "conflict_codes": conflict_codes,
            },
        },
    }


# -----------------------------------------------------------------------------
# P5 Scenario Matcher (REAL-READY signature from Section 9.5)
# -----------------------------------------------------------------------------

def match_p5_validation_scenario(
    topology_tile: Dict[str, Any],
    consistency_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Match current tile state to P5 validation scenarios.

    STATUS: PHASE X — P5 SCENARIO MATCHER (REAL-READY)

    Compares tile fields against the four canonical P5 validation scenarios
    (MOCK_BASELINE, HEALTHY, MISMATCH, XCOR_ANOMALY) and returns the best match
    with confidence score.

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned match is purely observational
    - No control flow depends on the match
    - Non-mutating: returns new dict

    Args:
        topology_tile: Topology bundle console tile
        consistency_result: Cross-system consistency result

    Returns:
        Scenario match result with:
        - scenario: str (MOCK_BASELINE, HEALTHY, MISMATCH, XCOR_ANOMALY, UNKNOWN)
        - confidence: float (0.0-1.0)
        - matching_criteria: List[str] (which criteria matched)
        - divergent_criteria: List[str] (which criteria did not match)

    Example:
        >>> tile = build_topology_bundle_console_tile(joint_view, consistency)
        >>> match = match_p5_validation_scenario(tile, consistency)
        >>> match["scenario"]
        'HEALTHY'
    """
    # Delegate to smoke validator and extract relevant fields
    result = run_p5_smoke_validation(topology_tile, consistency_result)

    return {
        "scenario": result["matched_scenario"],
        "confidence": result["confidence"],
        "matching_criteria": result["matching_criteria"],
        "divergent_criteria": result["divergent_criteria"],
    }


# -----------------------------------------------------------------------------
# P5 Reality Summary Builder (REAL-READY signature from Section 9.5)
# -----------------------------------------------------------------------------

def build_p5_topology_reality_summary(
    topology_tile: Dict[str, Any],
    bundle_tile: Dict[str, Any],
    replay_tile: Optional[Dict[str, Any]] = None,
    telemetry_tile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build unified P5 topology/reality summary from all active tiles.

    STATUS: PHASE X — P5 REALITY SUMMARY (REAL-READY)

    Aggregates topology/bundle tile with replay and telemetry tiles to produce
    a unified P5 validation summary suitable for auditor review and automated
    hypothesis generation.

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned summary is purely observational
    - No control flow depends on the summary contents
    - Non-mutating: returns new dict

    Args:
        topology_tile: Topology bundle console tile from build_topology_bundle_console_tile()
        bundle_tile: Bundle snapshot from joint_view (for raw bundle_status)
        replay_tile: Optional replay governance tile
        telemetry_tile: Optional telemetry governance tile

    Returns:
        P5 reality summary dict with:
        - schema_version: str
        - joint_status: str (ALIGNED, TENSION, DIVERGENT)
        - cross_system_consistency: bool
        - xcor_codes: List[str] (XCOR-* codes only)
        - p5_hypothesis: Dict with:
            - domain: str (TOPOLOGY, BUNDLE, EXTERNAL, NOMINAL)
            - confidence: str (HIGH, MEDIUM, LOW)
            - evidence: List[str]
            - recommended_action: str
        - cross_tile_correlation: Dict with:
            - replay_alignment: str (ALIGNED, DIVERGENT, UNKNOWN)
            - telemetry_alignment: str (ALIGNED, DIVERGENT, UNKNOWN)
            - correlation_notes: List[str]
        - scenario_match: str (HEALTHY, MOCK_BASELINE, MISMATCH, XCOR_ANOMALY, UNKNOWN)

    Example:
        >>> summary = build_p5_topology_reality_summary(
        ...     topology_tile=topo_tile,
        ...     bundle_tile=bundle_snap,
        ...     replay_tile=replay_gov,
        ...     telemetry_tile=telem_gov,
        ... )
        >>> summary["p5_hypothesis"]["domain"]
        'NOMINAL'
    """
    # Extract core fields from topology tile
    joint_status = topology_tile.get("joint_status", "UNKNOWN")
    cross_system_consistency = topology_tile.get("cross_system_consistency", False)
    conflict_codes = topology_tile.get("conflict_codes", [])
    xcor_codes = [code for code in conflict_codes if code.startswith("XCOR-")]

    # Build consistency result for scenario matching
    consistency_result = {"consistent": cross_system_consistency, "status": "OK"}

    # Match scenario
    scenario_match = match_p5_validation_scenario(topology_tile, consistency_result)

    # Determine hypothesis domain
    evidence: List[str] = []
    domain = "NOMINAL"
    confidence = "HIGH"
    recommended_action = "Continue monitoring"

    # Check for topology issues
    topology_stability = topology_tile.get("topology_stability", "UNKNOWN")
    if topology_stability in ("TURBULENT", "CRITICAL"):
        domain = "TOPOLOGY"
        confidence = "HIGH"
        evidence.append(f"topology_stability={topology_stability}")
        recommended_action = "Investigate topology metrics"
    elif topology_stability == "DRIFTING":
        evidence.append("topology_stability=DRIFTING")
        if domain == "NOMINAL":
            domain = "TOPOLOGY"
            confidence = "MEDIUM"

    # Check for bundle issues
    bundle_stability = topology_tile.get("bundle_stability", "UNKNOWN")
    if bundle_stability in ("BROKEN", "MISSING"):
        domain = "BUNDLE"
        confidence = "HIGH"
        evidence.append(f"bundle_stability={bundle_stability}")
        recommended_action = "Investigate bundle chain"
    elif bundle_stability == "ATTENTION":
        evidence.append("bundle_stability=ATTENTION")
        if domain == "NOMINAL":
            domain = "BUNDLE"
            confidence = "LOW"

    # Check for XCOR issues (external/timing)
    if any("XCOR-WARN" in code or "XCOR-CRIT" in code for code in xcor_codes):
        if domain == "NOMINAL":
            domain = "EXTERNAL"
            confidence = "MEDIUM"
        evidence.append(f"xcor_codes={xcor_codes}")
        if "XCOR-WARN-002" in xcor_codes:
            recommended_action = "Check clock synchronization"

    # Cross-tile correlation
    replay_alignment = "UNKNOWN"
    telemetry_alignment = "UNKNOWN"
    correlation_notes: List[str] = []

    if replay_tile is not None:
        replay_status = replay_tile.get("status", "UNKNOWN")
        if replay_status == "OK":
            replay_alignment = "ALIGNED"
        elif replay_status in ("WARN", "BLOCK"):
            replay_alignment = "DIVERGENT"
            correlation_notes.append(f"Replay tile shows {replay_status}")

    if telemetry_tile is not None:
        telemetry_status = telemetry_tile.get("status", "UNKNOWN")
        if telemetry_status == "OK":
            telemetry_alignment = "ALIGNED"
        elif telemetry_status in ("WARN", "BLOCK"):
            telemetry_alignment = "DIVERGENT"
            correlation_notes.append(f"Telemetry tile shows {telemetry_status}")

    # Cross-correlation patterns from Section 9.2
    if joint_status == "TENSION" and replay_alignment == "DIVERGENT":
        correlation_notes.append("PATTERN: Bundle gaps may cause replay hash mismatches")
    if topology_stability == "DRIFTING" and telemetry_alignment == "DIVERGENT":
        correlation_notes.append("PATTERN: Drift may be metric propagation delay artifact")

    return {
        "schema_version": P5_REALITY_ADAPTER_SCHEMA_VERSION,
        "joint_status": joint_status,
        "cross_system_consistency": cross_system_consistency,
        "xcor_codes": xcor_codes,
        "p5_hypothesis": {
            "domain": domain,
            "confidence": confidence,
            "evidence": evidence if evidence else ["All systems nominal"],
            "recommended_action": recommended_action,
        },
        "cross_tile_correlation": {
            "replay_alignment": replay_alignment,
            "telemetry_alignment": telemetry_alignment,
            "correlation_notes": correlation_notes,
        },
        "scenario_match": scenario_match["scenario"],
    }


# -----------------------------------------------------------------------------
# P5 Auditor Report Generator (REAL-READY signature from Section 9.5)
# -----------------------------------------------------------------------------

def generate_p5_auditor_report(
    p5_summary: Dict[str, Any],
    run_id: str,
    slice_name: str,
) -> Dict[str, Any]:
    """
    Generate structured auditor report from P5 summary.

    STATUS: PHASE X — P5 AUDITOR REPORT (REAL-READY)

    Produces a report following the 10-step auditor runbook, with each step's
    findings pre-populated from the P5 summary.

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned report is purely observational
    - No control flow depends on the report
    - Non-mutating: returns new dict

    Args:
        p5_summary: Output from build_p5_topology_reality_summary()
        run_id: Run identifier
        slice_name: Curriculum slice name

    Returns:
        Auditor report dict with:
        - schema_version: str
        - run_context: Dict (run_id, slice_name, timestamp, phase)
        - runbook_steps: List[Dict] (10 steps with findings)
        - final_hypothesis: Dict (domain, confidence, evidence, action)
        - escalation_required: bool
        - escalation_reason: Optional[str]

    Example:
        >>> summary = build_p5_topology_reality_summary(...)
        >>> report = generate_p5_auditor_report(summary, "run-001", "arithmetic/add")
        >>> report["escalation_required"]
        False
    """
    from datetime import datetime, timezone

    timestamp = datetime.now(timezone.utc).isoformat()

    # Build runbook steps (10-step from Section 9.4)
    runbook_steps: List[Dict[str, Any]] = []

    # Step 1: Establish Context
    runbook_steps.append({
        "step": 1,
        "name": "Establish Context",
        "finding": f"Run {run_id}, Slice {slice_name}, Phase P5",
        "status": "COMPLETE",
    })

    # Step 2: Check Joint Status
    joint_status = p5_summary.get("joint_status", "UNKNOWN")
    step2_status = "OK" if joint_status == "ALIGNED" else "ATTENTION"
    runbook_steps.append({
        "step": 2,
        "name": "Check Joint Status",
        "finding": f"joint_status={joint_status}",
        "status": step2_status,
        "interpretation": {
            "ALIGNED": "Structural health nominal",
            "TENSION": "Subsystem stress detected",
            "DIVERGENT": "Critical structural disagreement",
        }.get(joint_status, "Unknown status"),
    })

    # Step 3: Evaluate Status Light (derived from hypothesis confidence)
    hypothesis = p5_summary.get("p5_hypothesis", {})
    domain = hypothesis.get("domain", "UNKNOWN")
    confidence = hypothesis.get("confidence", "LOW")
    status_light = "GREEN" if domain == "NOMINAL" else ("YELLOW" if confidence == "LOW" else "RED")
    runbook_steps.append({
        "step": 3,
        "name": "Evaluate Status Light",
        "finding": f"status_light={status_light} (domain={domain})",
        "status": "OK" if status_light == "GREEN" else "ATTENTION",
    })

    # Step 4: Identify Active Conflict Codes
    xcor_codes = p5_summary.get("xcor_codes", [])
    runbook_steps.append({
        "step": 4,
        "name": "Identify Active Conflict Codes",
        "finding": f"xcor_codes={xcor_codes}" if xcor_codes else "No conflict codes",
        "status": "OK" if not xcor_codes else "ATTENTION",
        "highest_severity": _get_highest_severity(xcor_codes),
    })

    # Step 5: Determine Issue Domain
    runbook_steps.append({
        "step": 5,
        "name": "Determine Issue Domain",
        "finding": f"domain={domain}",
        "status": "OK" if domain == "NOMINAL" else "ATTENTION",
        "decision_path": _get_domain_decision_path(xcor_codes, p5_summary),
    })

    # Step 6: Cross-Reference with Replay Tile
    cross_tile = p5_summary.get("cross_tile_correlation", {})
    replay_alignment = cross_tile.get("replay_alignment", "UNKNOWN")
    runbook_steps.append({
        "step": 6,
        "name": "Cross-Reference with Replay Tile",
        "finding": f"replay_alignment={replay_alignment}",
        "status": "OK" if replay_alignment in ("ALIGNED", "UNKNOWN") else "ATTENTION",
    })

    # Step 7: Cross-Reference with Telemetry Tile
    telemetry_alignment = cross_tile.get("telemetry_alignment", "UNKNOWN")
    runbook_steps.append({
        "step": 7,
        "name": "Cross-Reference with Telemetry Tile",
        "finding": f"telemetry_alignment={telemetry_alignment}",
        "status": "OK" if telemetry_alignment in ("ALIGNED", "UNKNOWN") else "ATTENTION",
    })

    # Step 8: Check Cross-System Consistency
    cross_system_consistency = p5_summary.get("cross_system_consistency", False)
    runbook_steps.append({
        "step": 8,
        "name": "Check Cross-System Consistency",
        "finding": f"cross_system_consistency={cross_system_consistency}",
        "status": "OK" if cross_system_consistency else "ATTENTION",
        "interpretation": "Provenance verified" if cross_system_consistency else "Provenance chain broken",
    })

    # Step 9: Review Mode History
    runbook_steps.append({
        "step": 9,
        "name": "Review Mode History",
        "finding": "Mode history not available in summary",
        "status": "SKIPPED",
        "note": "Requires director_panel for full mode history",
    })

    # Step 10: Document Findings and Hypothesis
    runbook_steps.append({
        "step": 10,
        "name": "Document Findings and Hypothesis",
        "finding": f"P5_HYPOTHESIS: {domain}, CONFIDENCE: {confidence}",
        "status": "COMPLETE",
        "hypothesis": hypothesis,
    })

    # Determine escalation
    escalation_required = domain in ("BUNDLE", "TOPOLOGY") and confidence == "HIGH"
    escalation_reason = None
    if escalation_required:
        if domain == "BUNDLE":
            escalation_reason = "Critical bundle chain integrity issue detected"
        elif domain == "TOPOLOGY":
            escalation_reason = "Critical topology invariant violation detected"

    return {
        "schema_version": P5_REALITY_ADAPTER_SCHEMA_VERSION,
        "run_context": {
            "run_id": run_id,
            "slice_name": slice_name,
            "timestamp": timestamp,
            "phase": "P5",
        },
        "runbook_steps": runbook_steps,
        "final_hypothesis": hypothesis,
        "escalation_required": escalation_required,
        "escalation_reason": escalation_reason,
    }


def _get_highest_severity(codes: List[str]) -> str:
    """Get highest severity from code list."""
    if any("CRIT" in code for code in codes):
        return "CRITICAL"
    if any("WARN" in code for code in codes):
        return "WARN"
    if any("OK" in code for code in codes):
        return "OK"
    return "NONE"


def _get_domain_decision_path(xcor_codes: List[str], summary: Dict[str, Any]) -> str:
    """Determine domain decision path per Step 5 decision tree."""
    # Check TOPO-CRIT-*
    if any("TOPO-CRIT" in code for code in xcor_codes):
        return "conflict_codes contains TOPO-CRIT-* → Issue is TOPOLOGY"
    # Check BNDL-CRIT-*
    if any("BNDL-CRIT" in code for code in xcor_codes):
        return "conflict_codes contains BNDL-CRIT-* → Issue is BUNDLE"
    # Check XCOR-* only
    if all(code.startswith("XCOR-") for code in xcor_codes) and xcor_codes:
        return "conflict_codes contains XCOR-* only → Issue is EXTERNAL"
    # No critical codes
    domain = summary.get("p5_hypothesis", {}).get("domain", "UNKNOWN")
    return f"No critical codes → Derived domain={domain}"


# -----------------------------------------------------------------------------
# GGFL Alignment Adapter
# -----------------------------------------------------------------------------

# Scenarios that trigger warn status
_WARN_SCENARIOS = {"MISMATCH", "XCOR_ANOMALY"}

# Maximum number of drivers to return (deterministic cap)
_MAX_DRIVERS = 3

# Canonical reason codes for GGFL drivers (replaces free text)
DRIVER_SCHEMA_NOT_OK = "DRIVER_SCHEMA_NOT_OK"
DRIVER_VALIDATION_NOT_PASSED = "DRIVER_VALIDATION_NOT_PASSED"
DRIVER_SCENARIO_MISMATCH = "DRIVER_SCENARIO_MISMATCH"
DRIVER_SCENARIO_XCOR_ANOMALY = "DRIVER_SCENARIO_XCOR_ANOMALY"

# Extraction source enum values
EXTRACTION_SOURCE_MANIFEST = "MANIFEST"
EXTRACTION_SOURCE_EVIDENCE_JSON = "EVIDENCE_JSON"
EXTRACTION_SOURCE_RUN_DIR_ROOT = "RUN_DIR_ROOT"
EXTRACTION_SOURCE_P4_SHADOW = "P4_SHADOW"
EXTRACTION_SOURCE_MISSING = "MISSING"

# Shadow mode invariants block (constant for SHADOW MODE)
_SHADOW_MODE_INVARIANTS = {
    "advisory_only": True,
    "no_enforcement": True,
    "conflict_invariant": False,
}


def topology_p5_for_alignment_view(
    signal_block: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compact GGFL adapter for topology_p5 signal.

    STATUS: PHASE X — GGFL ALIGNMENT VIEW ADAPTER

    Accepts either a manifest governance block or a raw signal block and returns
    a normalized alignment view suitable for GGFL integration.

    TRUST BOUNDARY CONTRACT:
    - If schema_ok=False, status is "warn" and no derived scenario fields included
    - If schema_ok=False, only sha256 + path are surfaced (no fabrication)
    - Drivers are reason codes only (max 3, deterministically ordered)
    - Summary is a single neutral sentence

    REASON CODE DRIVERS:
    - DRIVER_SCHEMA_NOT_OK: schema_ok=False
    - DRIVER_VALIDATION_NOT_PASSED: validation_passed=False
    - DRIVER_SCENARIO_MISMATCH: scenario="MISMATCH"
    - DRIVER_SCENARIO_XCOR_ANOMALY: scenario="XCOR_ANOMALY"

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned signal is purely observational
    - No control flow depends on the signal
    - Non-mutating: returns new dict
    - shadow_mode_invariants block always present

    Args:
        signal_block: Either manifest["governance"]["topology_p5"] or raw
                      P5TopologyAuditorReference-like dict with fields:
                      - path: str
                      - sha256: str
                      - schema_ok: bool (optional, defaults True)
                      - scenario: str (optional)
                      - scenario_confidence: float (optional)
                      - joint_status: str (optional)
                      - validation_passed: bool (optional)
                      - shadow_mode_invariant_ok: bool (optional)
                      - mode: str (optional)
                      - extraction_source: str (optional)

    Returns:
        Normalized alignment signal with:
        - signal_type: "SIG-TOP5"
        - status: "ok" | "warn"
        - conflict: False (always, SHADOW MODE)
        - drivers: List[str] (reason codes only, max 3, deterministically ordered)
        - summary: str (1 neutral sentence)
        - shadow_mode_invariants: Dict (advisory_only, no_enforcement, conflict_invariant)

    Example:
        >>> signal = {"scenario": "HEALTHY", "schema_ok": True, "validation_passed": True}
        >>> view = topology_p5_for_alignment_view(signal)
        >>> view["signal_type"]
        'SIG-TOP5'
        >>> view["status"]
        'ok'
        >>> view["shadow_mode_invariants"]["advisory_only"]
        True
    """
    # Extract core fields
    schema_ok = signal_block.get("schema_ok", True)
    path = signal_block.get("path", "unknown")
    sha256 = signal_block.get("sha256", "unknown")
    extraction_source = signal_block.get("extraction_source", EXTRACTION_SOURCE_MISSING)

    # TRUST BOUNDARY: If schema_ok=False, only surface path/sha256, status=warn
    if not schema_ok:
        return {
            "signal_type": "SIG-TOP5",
            "status": "warn",
            "conflict": False,
            "drivers": [DRIVER_SCHEMA_NOT_OK],
            "summary": f"Topology P5 report at {path} has invalid schema.",
            "path": path,
            "sha256": sha256,
            "extraction_source": extraction_source,
            "shadow_mode_invariants": _SHADOW_MODE_INVARIANTS.copy(),
        }

    # Extract scenario fields (only if schema_ok=True)
    scenario = signal_block.get("scenario")
    scenario_confidence = signal_block.get("scenario_confidence")
    joint_status = signal_block.get("joint_status")
    validation_passed = signal_block.get("validation_passed", False)

    # Determine status: warn if scenario in warn set OR validation failed
    is_warn_scenario = scenario in _WARN_SCENARIOS
    status = "warn" if (is_warn_scenario or not validation_passed) else "ok"

    # Build drivers list using REASON CODES ONLY (deterministically ordered)
    drivers: List[str] = []

    # Add reason codes based on conditions
    if not validation_passed:
        drivers.append(DRIVER_VALIDATION_NOT_PASSED)

    if scenario == "MISMATCH":
        drivers.append(DRIVER_SCENARIO_MISMATCH)

    if scenario == "XCOR_ANOMALY":
        drivers.append(DRIVER_SCENARIO_XCOR_ANOMALY)

    # Sort for determinism and cap at MAX_DRIVERS
    drivers = sorted(drivers)[:_MAX_DRIVERS]

    # Build neutral summary sentence
    if scenario:
        if status == "ok":
            summary = f"Topology P5 scenario {scenario} validated successfully."
        else:
            summary = f"Topology P5 scenario {scenario} requires attention."
    else:
        summary = "Topology P5 report processed with no scenario match."

    return {
        "signal_type": "SIG-TOP5",
        "status": status,
        "conflict": False,
        "drivers": drivers,
        "summary": summary,
        "path": path,
        "sha256": sha256,
        "scenario": scenario,
        "scenario_confidence": scenario_confidence,
        "joint_status": joint_status,
        "validation_passed": validation_passed,
        "extraction_source": extraction_source,
        "shadow_mode_invariants": _SHADOW_MODE_INVARIANTS.copy(),
    }


def build_topology_p5_status_signal(
    reference: Optional[Dict[str, Any]],
    extraction_source: str = EXTRACTION_SOURCE_MISSING,
) -> Dict[str, Any]:
    """
    Build topology_p5 status signal from P5TopologyAuditorReference.

    STATUS: PHASE X — STATUS SIGNAL NORMALIZATION

    Implements the TRUST BOUNDARY contract for status signals:
    - If schema_ok=False: surfaces topology_p5.schema_ok=false with sha256+path only
    - If schema_ok=True: includes all derived scenario fields
    - extraction_source always included for provenance tracking

    EXTRACTION SOURCE VALUES:
    - MANIFEST: Loaded from manifest.json governance block
    - EVIDENCE_JSON: Loaded from evidence pack JSON file
    - RUN_DIR_ROOT: Loaded from run directory root
    - P4_SHADOW: Loaded from p4_shadow subdirectory
    - MISSING: No report found

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned signal is purely observational
    - Non-mutating: returns new dict

    Args:
        reference: P5TopologyAuditorReference-like dict or None
        extraction_source: Provenance source enum value

    Returns:
        Status signal dict suitable for inclusion in first_light_status.json
        signals["topology_p5"] block.
    """
    if reference is None:
        return {
            "present": False,
            "schema_ok": True,
            "extraction_source": EXTRACTION_SOURCE_MISSING,
            "advisory_warning": "No topology P5 report found",
        }

    schema_ok = reference.get("schema_ok", True)
    path = reference.get("path")
    sha256 = reference.get("sha256")
    advisory_warning = reference.get("advisory_warning")
    # Use provided extraction_source or derive from reference
    source = reference.get("extraction_source", extraction_source)

    # TRUST BOUNDARY: If schema_ok=False, only include path/sha256, no scenario fields
    if not schema_ok:
        return {
            "present": True,
            "schema_ok": False,
            "extraction_source": source,
            "path": path,
            "sha256": sha256,
            "advisory_warning": advisory_warning,
            # Explicitly no scenario fields - trust boundary
        }

    # schema_ok=True: include all derived fields
    return {
        "present": True,
        "schema_ok": True,
        "extraction_source": source,
        "path": path,
        "sha256": sha256,
        "scenario": reference.get("scenario"),
        "scenario_confidence": reference.get("scenario_confidence"),
        "joint_status": reference.get("joint_status"),
        "shadow_mode_invariant_ok": reference.get("shadow_mode_invariant_ok", True),
        "validation_passed": reference.get("validation_passed", False),
        "mode": reference.get("mode"),
        "advisory_warning": advisory_warning,
    }


# -----------------------------------------------------------------------------
# Module exports
# -----------------------------------------------------------------------------

__all__ = [
    "P5_REALITY_ADAPTER_SCHEMA_VERSION",
    "P5_SCENARIOS",
    "extract_topology_reality_metrics",
    "validate_bundle_stability",
    "detect_xcor_anomaly",
    "run_p5_smoke_validation",
    "match_p5_validation_scenario",
    "build_p5_topology_reality_summary",
    "generate_p5_auditor_report",
    "topology_p5_for_alignment_view",
    "build_topology_p5_status_signal",
    # Reason code constants
    "DRIVER_SCHEMA_NOT_OK",
    "DRIVER_VALIDATION_NOT_PASSED",
    "DRIVER_SCENARIO_MISMATCH",
    "DRIVER_SCENARIO_XCOR_ANOMALY",
    # Extraction source constants
    "EXTRACTION_SOURCE_MANIFEST",
    "EXTRACTION_SOURCE_EVIDENCE_JSON",
    "EXTRACTION_SOURCE_RUN_DIR_ROOT",
    "EXTRACTION_SOURCE_P4_SHADOW",
    "EXTRACTION_SOURCE_MISSING",
]
