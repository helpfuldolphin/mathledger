"""Unit tests for P5 Topology Reality Adapter.

STATUS: PHASE X — P5 VALIDATION TESTS

Tests for:
- Topology reality extraction
- Bundle stability validation
- XCOR anomaly detection
- 3-case smoke validator (MOCK_BASELINE, HEALTHY, MISMATCH)
- P5 scenario matching
- P5 reality summary builder
- P5 auditor report generator
- SHADOW MODE invariants

SHADOW MODE CONTRACT:
- All tested functions must be read-only
- All outputs must be observational only
- No enforcement logic should be present
"""

import json
import pytest
from typing import Any, Dict

from backend.health.p5_topology_reality_adapter import (
    P5_REALITY_ADAPTER_SCHEMA_VERSION,
    P5_SCENARIOS,
    extract_topology_reality_metrics,
    validate_bundle_stability,
    detect_xcor_anomaly,
    run_p5_smoke_validation,
    match_p5_validation_scenario,
    build_p5_topology_reality_summary,
    generate_p5_auditor_report,
    topology_p5_for_alignment_view,
    build_topology_p5_status_signal,
    # Reason code constants
    DRIVER_SCHEMA_NOT_OK,
    DRIVER_VALIDATION_NOT_PASSED,
    DRIVER_SCENARIO_MISMATCH,
    DRIVER_SCENARIO_XCOR_ANOMALY,
    # Extraction source constants
    EXTRACTION_SOURCE_MANIFEST,
    EXTRACTION_SOURCE_EVIDENCE_JSON,
    EXTRACTION_SOURCE_RUN_DIR_ROOT,
    EXTRACTION_SOURCE_P4_SHADOW,
    EXTRACTION_SOURCE_MISSING,
)


# -----------------------------------------------------------------------------
# Test fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def healthy_topology_tile() -> Dict[str, Any]:
    """Create a HEALTHY scenario topology tile."""
    return {
        "schema_version": "1.0.0",
        "status_light": "GREEN",
        "topology_stability": "STABLE",
        "bundle_stability": "VALID",
        "cross_system_consistency": True,
        "joint_status": "ALIGNED",
        "conflict_codes": [],
        "headline": "Topology: STABLE | Bundle: VALID | Alignment: ALIGNED",
    }


@pytest.fixture
def healthy_joint_view() -> Dict[str, Any]:
    """Create a HEALTHY scenario joint view."""
    return {
        "topology_snapshot": {
            "topology_mode": "STABLE",
            "betti_numbers": {"beta_0": 1, "beta_1": 0},
            "persistence_metrics": {"bottleneck_drift": 0.02},
            "safe_region_metrics": {"boundary_distance": 0.5},
        },
        "bundle_snapshot": {
            "bundle_status": "VALID",
            "chain_info": {"chain_valid": True},
            "manifest": {"coverage": 1.0},
            "provenance": {"verified": True},
        },
        "alignment_status": {"overall_status": "ALIGNED"},
    }


@pytest.fixture
def healthy_consistency_result() -> Dict[str, Any]:
    """Create a HEALTHY scenario consistency result."""
    return {"consistent": True, "status": "OK"}


@pytest.fixture
def mock_baseline_topology_tile() -> Dict[str, Any]:
    """Create a MOCK_BASELINE scenario topology tile."""
    return {
        "schema_version": "1.0.0",
        "status_light": "YELLOW",
        "topology_stability": "DRIFTING",
        "bundle_stability": "ATTENTION",
        "cross_system_consistency": False,
        "joint_status": "TENSION",
        "conflict_codes": ["XCOR-WARN-001", "XCOR-WARN-002"],
        "headline": "Topology: DRIFTING | Bundle: ATTENTION | Alignment: TENSION",
    }


@pytest.fixture
def mock_baseline_consistency_result() -> Dict[str, Any]:
    """Create a MOCK_BASELINE scenario consistency result."""
    return {"consistent": False, "status": "WARN"}


@pytest.fixture
def mismatch_topology_tile() -> Dict[str, Any]:
    """Create a MISMATCH scenario topology tile."""
    return {
        "schema_version": "1.0.0",
        "status_light": "RED",
        "topology_stability": "STABLE",
        "bundle_stability": "BROKEN",
        "cross_system_consistency": False,
        "joint_status": "DIVERGENT",
        "conflict_codes": ["BNDL-CRIT-001", "XCOR-CRIT-001"],
        "headline": "Topology: STABLE | Bundle: BROKEN | Alignment: DIVERGENT",
    }


@pytest.fixture
def mismatch_consistency_result() -> Dict[str, Any]:
    """Create a MISMATCH scenario consistency result."""
    return {"consistent": False, "status": "BLOCK"}


@pytest.fixture
def xcor_anomaly_topology_tile() -> Dict[str, Any]:
    """Create an XCOR_ANOMALY scenario topology tile."""
    return {
        "schema_version": "1.0.0",
        "status_light": "YELLOW",
        "topology_stability": "STABLE",
        "bundle_stability": "VALID",
        "cross_system_consistency": False,
        "joint_status": "TENSION",
        "conflict_codes": ["XCOR-WARN-002"],
        "headline": "Topology: STABLE | Bundle: VALID | Alignment: TENSION",
    }


# -----------------------------------------------------------------------------
# Test: extract_topology_reality_metrics
# -----------------------------------------------------------------------------

class TestExtractTopologyRealityMetrics:
    """Tests for extract_topology_reality_metrics function."""

    def test_extracts_from_joint_view(self, healthy_topology_tile, healthy_joint_view):
        """Test extraction from joint_view when available."""
        metrics = extract_topology_reality_metrics(healthy_topology_tile, healthy_joint_view)

        assert metrics["schema_version"] == P5_REALITY_ADAPTER_SCHEMA_VERSION
        assert metrics["topology_mode"] == "STABLE"
        assert metrics["betti_bounds_status"] == "IN_BOUNDS"
        assert metrics["persistence_stability"] == "STABLE"
        assert metrics["omega_status"] == "INSIDE"
        assert metrics["extraction_source"] == "JOINT_VIEW"

    def test_derives_from_tile_when_no_joint_view(self, healthy_topology_tile):
        """Test derivation from tile when joint_view not available."""
        metrics = extract_topology_reality_metrics(healthy_topology_tile, joint_view=None)

        assert metrics["extraction_source"] == "TILE_DERIVED"
        assert metrics["topology_mode"] == "STABLE"
        assert metrics["topology_stability"] == "STABLE"

    def test_detects_out_of_bounds_betti(self):
        """Test detection of out-of-bounds Betti numbers."""
        tile = {"topology_stability": "CRITICAL"}
        joint_view = {
            "topology_snapshot": {
                "topology_mode": "CRITICAL",
                "betti_numbers": {"beta_0": 2, "beta_1": 5},  # Out of bounds
                "persistence_metrics": {"bottleneck_drift": 0.3},
                "safe_region_metrics": {"boundary_distance": -0.1},
            },
            "bundle_snapshot": {"bundle_status": "VALID"},
            "alignment_status": {"overall_status": "DIVERGENT"},
        }

        metrics = extract_topology_reality_metrics(tile, joint_view)

        assert metrics["betti_bounds_status"] == "OUT_OF_BOUNDS"
        assert metrics["persistence_stability"] == "COLLAPSED"
        assert metrics["omega_status"] == "OUTSIDE"

    def test_output_is_json_serializable(self, healthy_topology_tile, healthy_joint_view):
        """Test that output is JSON serializable (SHADOW MODE requirement)."""
        metrics = extract_topology_reality_metrics(healthy_topology_tile, healthy_joint_view)
        serialized = json.dumps(metrics)
        assert serialized is not None
        deserialized = json.loads(serialized)
        assert deserialized == metrics


# -----------------------------------------------------------------------------
# Test: validate_bundle_stability
# -----------------------------------------------------------------------------

class TestValidateBundleStability:
    """Tests for validate_bundle_stability function."""

    def test_validates_healthy_bundle(self, healthy_topology_tile, healthy_joint_view, healthy_consistency_result):
        """Test validation of healthy bundle."""
        validation = validate_bundle_stability(
            healthy_topology_tile, healthy_joint_view, healthy_consistency_result
        )

        assert validation["schema_version"] == P5_REALITY_ADAPTER_SCHEMA_VERSION
        assert validation["bundle_status"] == "VALID"
        assert validation["chain_integrity"] == "VALID"
        assert validation["manifest_coverage"] == "COMPLETE"
        assert validation["provenance_verified"] is True
        assert "BNDL-OK-001" in validation["validation_codes"]

    def test_detects_broken_bundle(self, mismatch_topology_tile):
        """Test detection of broken bundle."""
        joint_view = {
            "topology_snapshot": {"topology_mode": "STABLE"},
            "bundle_snapshot": {
                "bundle_status": "BROKEN",
                "chain_info": {"chain_valid": False},
                "manifest": {"coverage": 0.5},
            },
            "alignment_status": {"overall_status": "DIVERGENT"},
        }

        validation = validate_bundle_stability(mismatch_topology_tile, joint_view)

        assert validation["bundle_status"] == "BROKEN"
        assert validation["chain_integrity"] == "BROKEN"
        assert "BNDL-CRIT-001" in validation["validation_codes"]

    def test_derives_from_tile_stability(self, healthy_topology_tile):
        """Test derivation from tile when no joint_view."""
        validation = validate_bundle_stability(healthy_topology_tile)

        assert validation["bundle_stability"] == "VALID"
        assert validation["chain_integrity"] == "VALID"
        assert validation["provenance_verified"] is True

    def test_output_is_json_serializable(self, healthy_topology_tile):
        """Test that output is JSON serializable."""
        validation = validate_bundle_stability(healthy_topology_tile)
        serialized = json.dumps(validation)
        assert serialized is not None


# -----------------------------------------------------------------------------
# Test: detect_xcor_anomaly
# -----------------------------------------------------------------------------

class TestDetectXcorAnomaly:
    """Tests for detect_xcor_anomaly function."""

    def test_detects_no_anomaly_for_healthy(self, healthy_topology_tile):
        """Test no anomaly detected for healthy tile."""
        metrics = extract_topology_reality_metrics(healthy_topology_tile)
        validation = validate_bundle_stability(healthy_topology_tile)

        anomaly = detect_xcor_anomaly(healthy_topology_tile, metrics, validation)

        assert anomaly["anomaly_detected"] is False
        assert anomaly["anomaly_type"] == "NONE"
        assert "XCOR-OK-001" in anomaly["xcor_codes"]
        assert anomaly["triple_fault_active"] is False

    def test_detects_temporal_mismatch(self, xcor_anomaly_topology_tile):
        """Test detection of temporal mismatch anomaly."""
        anomaly = detect_xcor_anomaly(xcor_anomaly_topology_tile)

        assert anomaly["anomaly_detected"] is True
        assert anomaly["anomaly_type"] == "TEMPORAL_MISMATCH"
        assert "XCOR-WARN-002" in anomaly["xcor_codes"]
        assert anomaly["timing_skew_indicator"] == "SIGNIFICANT"

    def test_detects_triple_fault(self, mismatch_topology_tile):
        """Test detection of triple fault condition."""
        metrics = {"topology_mode": "STABLE"}
        validation = {"chain_integrity": "BROKEN"}

        anomaly = detect_xcor_anomaly(mismatch_topology_tile, metrics, validation)

        assert anomaly["triple_fault_active"] is True
        assert anomaly["anomaly_type"] == "TRIPLE_FAULT"
        assert "XCOR-CRIT-001" in anomaly["xcor_codes"]

    def test_detects_divergent_state(self):
        """Test detection of divergent state."""
        tile = {
            "joint_status": "DIVERGENT",
            "conflict_codes": [],
        }

        anomaly = detect_xcor_anomaly(tile)

        assert anomaly["anomaly_detected"] is True
        assert anomaly["topology_bundle_alignment"] == "DIVERGENT"

    def test_output_is_json_serializable(self, healthy_topology_tile):
        """Test that output is JSON serializable."""
        anomaly = detect_xcor_anomaly(healthy_topology_tile)
        serialized = json.dumps(anomaly)
        assert serialized is not None


# -----------------------------------------------------------------------------
# Test: run_p5_smoke_validation (3-case validator)
# -----------------------------------------------------------------------------

class TestRunP5SmokeValidation:
    """Tests for run_p5_smoke_validation function (3-case validator)."""

    def test_matches_healthy_scenario(self, healthy_topology_tile, healthy_consistency_result):
        """Test matching HEALTHY scenario."""
        result = run_p5_smoke_validation(healthy_topology_tile, healthy_consistency_result)

        assert result["matched_scenario"] == "HEALTHY"
        assert result["confidence"] > 0.8
        assert result["validation_passed"] is True
        assert result["shadow_mode_invariant_ok"] is True

    def test_matches_mock_baseline_scenario(self, mock_baseline_topology_tile, mock_baseline_consistency_result):
        """Test matching MOCK_BASELINE scenario."""
        result = run_p5_smoke_validation(mock_baseline_topology_tile, mock_baseline_consistency_result)

        assert result["matched_scenario"] == "MOCK_BASELINE"
        assert result["confidence"] > 0.6
        assert result["validation_passed"] is True

    def test_matches_mismatch_scenario(self, mismatch_topology_tile, mismatch_consistency_result):
        """Test matching MISMATCH scenario."""
        result = run_p5_smoke_validation(mismatch_topology_tile, mismatch_consistency_result)

        assert result["matched_scenario"] == "MISMATCH"
        assert result["confidence"] > 0.7
        assert result["validation_passed"] is True

    def test_scenario_override_forces_match(self, healthy_topology_tile, healthy_consistency_result):
        """Test scenario override for testing purposes."""
        result = run_p5_smoke_validation(
            healthy_topology_tile,
            healthy_consistency_result,
            scenario_override="MOCK_BASELINE",
        )

        assert result["matched_scenario"] == "MOCK_BASELINE"
        assert result["confidence"] == 1.0
        assert result["diagnostic"]["override"] is True

    def test_returns_unknown_for_ambiguous_tile(self):
        """Test UNKNOWN returned for ambiguous tile state."""
        ambiguous_tile = {
            "status_light": "UNKNOWN",
            "topology_stability": "UNKNOWN",
            "bundle_stability": "UNKNOWN",
            "joint_status": "UNKNOWN",
            "conflict_codes": [],
        }
        consistency = {"consistent": None, "status": "UNKNOWN"}

        result = run_p5_smoke_validation(ambiguous_tile, consistency)

        # Should still return a result, possibly UNKNOWN
        assert result["schema_version"] == P5_REALITY_ADAPTER_SCHEMA_VERSION
        assert "matched_scenario" in result

    def test_includes_diagnostic_info(self, healthy_topology_tile, healthy_consistency_result):
        """Test diagnostic info is included."""
        result = run_p5_smoke_validation(healthy_topology_tile, healthy_consistency_result)

        assert "diagnostic" in result
        assert "all_scores" in result["diagnostic"]
        assert "tile_snapshot" in result["diagnostic"]

    def test_output_is_json_serializable(self, healthy_topology_tile, healthy_consistency_result):
        """Test that output is JSON serializable."""
        result = run_p5_smoke_validation(healthy_topology_tile, healthy_consistency_result)
        serialized = json.dumps(result)
        assert serialized is not None

    def test_shadow_mode_invariant_always_true(self, mismatch_topology_tile, mismatch_consistency_result):
        """Test SHADOW MODE invariant is always OK."""
        result = run_p5_smoke_validation(mismatch_topology_tile, mismatch_consistency_result)

        # Even for critical conditions, SHADOW MODE invariant should be OK
        assert result["shadow_mode_invariant_ok"] is True


# -----------------------------------------------------------------------------
# Test: match_p5_validation_scenario
# -----------------------------------------------------------------------------

class TestMatchP5ValidationScenario:
    """Tests for match_p5_validation_scenario function."""

    def test_returns_expected_fields(self, healthy_topology_tile, healthy_consistency_result):
        """Test that expected fields are returned."""
        match = match_p5_validation_scenario(healthy_topology_tile, healthy_consistency_result)

        assert "scenario" in match
        assert "confidence" in match
        assert "matching_criteria" in match
        assert "divergent_criteria" in match

    def test_matches_healthy(self, healthy_topology_tile, healthy_consistency_result):
        """Test matching HEALTHY scenario."""
        match = match_p5_validation_scenario(healthy_topology_tile, healthy_consistency_result)

        assert match["scenario"] == "HEALTHY"
        assert match["confidence"] > 0.8

    def test_output_is_json_serializable(self, healthy_topology_tile, healthy_consistency_result):
        """Test that output is JSON serializable."""
        match = match_p5_validation_scenario(healthy_topology_tile, healthy_consistency_result)
        serialized = json.dumps(match)
        assert serialized is not None


# -----------------------------------------------------------------------------
# Test: build_p5_topology_reality_summary
# -----------------------------------------------------------------------------

class TestBuildP5TopologyRealitySummary:
    """Tests for build_p5_topology_reality_summary function."""

    def test_builds_nominal_summary(self, healthy_topology_tile):
        """Test building summary for nominal/healthy state."""
        bundle_tile = {"bundle_status": "VALID"}

        summary = build_p5_topology_reality_summary(
            topology_tile=healthy_topology_tile,
            bundle_tile=bundle_tile,
        )

        assert summary["schema_version"] == P5_REALITY_ADAPTER_SCHEMA_VERSION
        assert summary["joint_status"] == "ALIGNED"
        assert summary["cross_system_consistency"] is True
        assert summary["p5_hypothesis"]["domain"] == "NOMINAL"
        assert summary["scenario_match"] == "HEALTHY"

    def test_builds_summary_with_replay_tile(self, healthy_topology_tile):
        """Test building summary with replay tile."""
        bundle_tile = {"bundle_status": "VALID"}
        replay_tile = {"status": "OK"}

        summary = build_p5_topology_reality_summary(
            topology_tile=healthy_topology_tile,
            bundle_tile=bundle_tile,
            replay_tile=replay_tile,
        )

        assert summary["cross_tile_correlation"]["replay_alignment"] == "ALIGNED"

    def test_builds_summary_with_telemetry_tile(self, healthy_topology_tile):
        """Test building summary with telemetry tile."""
        bundle_tile = {"bundle_status": "VALID"}
        telemetry_tile = {"status": "WARN"}

        summary = build_p5_topology_reality_summary(
            topology_tile=healthy_topology_tile,
            bundle_tile=bundle_tile,
            telemetry_tile=telemetry_tile,
        )

        assert summary["cross_tile_correlation"]["telemetry_alignment"] == "DIVERGENT"
        assert len(summary["cross_tile_correlation"]["correlation_notes"]) > 0

    def test_detects_topology_issues(self):
        """Test detection of topology issues."""
        tile = {
            "topology_stability": "CRITICAL",
            "bundle_stability": "VALID",
            "joint_status": "DIVERGENT",
            "cross_system_consistency": False,
            "conflict_codes": ["TOPO-CRIT-001"],
        }
        bundle_tile = {"bundle_status": "VALID"}

        summary = build_p5_topology_reality_summary(tile, bundle_tile)

        assert summary["p5_hypothesis"]["domain"] == "TOPOLOGY"
        assert summary["p5_hypothesis"]["confidence"] == "HIGH"

    def test_detects_bundle_issues(self):
        """Test detection of bundle issues."""
        tile = {
            "topology_stability": "STABLE",
            "bundle_stability": "BROKEN",
            "joint_status": "DIVERGENT",
            "cross_system_consistency": False,
            "conflict_codes": ["BNDL-CRIT-001"],
        }
        bundle_tile = {"bundle_status": "BROKEN"}

        summary = build_p5_topology_reality_summary(tile, bundle_tile)

        assert summary["p5_hypothesis"]["domain"] == "BUNDLE"

    def test_output_is_json_serializable(self, healthy_topology_tile):
        """Test that output is JSON serializable."""
        summary = build_p5_topology_reality_summary(healthy_topology_tile, {})
        serialized = json.dumps(summary)
        assert serialized is not None


# -----------------------------------------------------------------------------
# Test: generate_p5_auditor_report
# -----------------------------------------------------------------------------

class TestGenerateP5AuditorReport:
    """Tests for generate_p5_auditor_report function."""

    def test_generates_report_structure(self, healthy_topology_tile):
        """Test that report has expected structure."""
        summary = build_p5_topology_reality_summary(healthy_topology_tile, {})
        report = generate_p5_auditor_report(summary, "run-001", "arithmetic/add")

        assert report["schema_version"] == P5_REALITY_ADAPTER_SCHEMA_VERSION
        assert "run_context" in report
        assert "runbook_steps" in report
        assert "final_hypothesis" in report
        assert "escalation_required" in report

    def test_has_10_runbook_steps(self, healthy_topology_tile):
        """Test that report has exactly 10 runbook steps."""
        summary = build_p5_topology_reality_summary(healthy_topology_tile, {})
        report = generate_p5_auditor_report(summary, "run-001", "arithmetic/add")

        assert len(report["runbook_steps"]) == 10

    def test_run_context_populated(self, healthy_topology_tile):
        """Test that run context is populated correctly."""
        summary = build_p5_topology_reality_summary(healthy_topology_tile, {})
        report = generate_p5_auditor_report(summary, "run-123", "logic/implies")

        assert report["run_context"]["run_id"] == "run-123"
        assert report["run_context"]["slice_name"] == "logic/implies"
        assert report["run_context"]["phase"] == "P5"
        assert "timestamp" in report["run_context"]

    def test_no_escalation_for_nominal(self, healthy_topology_tile):
        """Test no escalation required for nominal state."""
        summary = build_p5_topology_reality_summary(healthy_topology_tile, {})
        report = generate_p5_auditor_report(summary, "run-001", "test")

        assert report["escalation_required"] is False
        assert report["escalation_reason"] is None

    def test_escalation_for_bundle_critical(self):
        """Test escalation required for critical bundle issues."""
        tile = {
            "topology_stability": "STABLE",
            "bundle_stability": "BROKEN",
            "joint_status": "DIVERGENT",
            "cross_system_consistency": False,
            "conflict_codes": ["BNDL-CRIT-001"],
        }
        summary = build_p5_topology_reality_summary(tile, {})
        report = generate_p5_auditor_report(summary, "run-001", "test")

        assert report["escalation_required"] is True
        assert "bundle" in report["escalation_reason"].lower()

    def test_output_is_json_serializable(self, healthy_topology_tile):
        """Test that output is JSON serializable."""
        summary = build_p5_topology_reality_summary(healthy_topology_tile, {})
        report = generate_p5_auditor_report(summary, "run-001", "test")
        serialized = json.dumps(report)
        assert serialized is not None


# -----------------------------------------------------------------------------
# Test: SHADOW MODE Invariants
# -----------------------------------------------------------------------------

class TestShadowModeInvariants:
    """Tests to verify SHADOW MODE contract is maintained."""

    def test_no_enforcement_fields_in_topology_metrics(self, healthy_topology_tile, healthy_joint_view):
        """Test topology metrics contain no enforcement fields."""
        metrics = extract_topology_reality_metrics(healthy_topology_tile, healthy_joint_view)

        # Should not contain enforcement-related fields
        assert "abort" not in str(metrics).lower()
        assert "enforce" not in str(metrics).lower()
        assert "gate" not in str(metrics).lower()
        assert "block_execution" not in str(metrics).lower()

    def test_no_enforcement_fields_in_bundle_validation(self, healthy_topology_tile):
        """Test bundle validation contains no enforcement fields."""
        validation = validate_bundle_stability(healthy_topology_tile)

        assert "abort" not in str(validation).lower()
        assert "enforce" not in str(validation).lower()
        assert "block_execution" not in str(validation).lower()

    def test_no_enforcement_fields_in_xcor_detection(self, healthy_topology_tile):
        """Test XCOR detection contains no enforcement fields."""
        anomaly = detect_xcor_anomaly(healthy_topology_tile)

        assert "abort" not in str(anomaly).lower()
        assert "enforce" not in str(anomaly).lower()
        assert "block_execution" not in str(anomaly).lower()

    def test_smoke_validation_shadow_invariant(self, mismatch_topology_tile, mismatch_consistency_result):
        """Test smoke validation maintains SHADOW MODE invariant."""
        result = run_p5_smoke_validation(mismatch_topology_tile, mismatch_consistency_result)

        # SHADOW MODE invariant should be OK even for critical conditions
        assert result["shadow_mode_invariant_ok"] is True

    def test_all_outputs_are_observational(self, healthy_topology_tile, healthy_consistency_result):
        """Test all outputs are purely observational."""
        # All functions should return dicts, not trigger side effects
        metrics = extract_topology_reality_metrics(healthy_topology_tile)
        validation = validate_bundle_stability(healthy_topology_tile)
        anomaly = detect_xcor_anomaly(healthy_topology_tile)
        smoke = run_p5_smoke_validation(healthy_topology_tile, healthy_consistency_result)
        match = match_p5_validation_scenario(healthy_topology_tile, healthy_consistency_result)
        summary = build_p5_topology_reality_summary(healthy_topology_tile, {})
        report = generate_p5_auditor_report(summary, "run-001", "test")

        # All should be dicts
        assert isinstance(metrics, dict)
        assert isinstance(validation, dict)
        assert isinstance(anomaly, dict)
        assert isinstance(smoke, dict)
        assert isinstance(match, dict)
        assert isinstance(summary, dict)
        assert isinstance(report, dict)


# -----------------------------------------------------------------------------
# Test: P5_SCENARIOS constant
# -----------------------------------------------------------------------------

class TestP5ScenariosConstant:
    """Tests for P5_SCENARIOS constant definition."""

    def test_has_four_scenarios(self):
        """Test that all four scenarios are defined."""
        assert "MOCK_BASELINE" in P5_SCENARIOS
        assert "HEALTHY" in P5_SCENARIOS
        assert "MISMATCH" in P5_SCENARIOS
        assert "XCOR_ANOMALY" in P5_SCENARIOS

    def test_scenarios_have_required_fields(self):
        """Test that all scenarios have required fields."""
        required_fields = [
            "status_light",
            "topology_stability",
            "bundle_stability",
            "joint_status",
            "cross_system_consistency",
            "xcor_codes_expected",
            "description",
        ]

        for scenario_name, scenario_def in P5_SCENARIOS.items():
            for field in required_fields:
                assert field in scenario_def, f"{scenario_name} missing {field}"

    def test_scenarios_are_json_serializable(self):
        """Test that scenario definitions are JSON serializable."""
        serialized = json.dumps(P5_SCENARIOS)
        assert serialized is not None
        deserialized = json.loads(serialized)
        assert deserialized == P5_SCENARIOS


# -----------------------------------------------------------------------------
# Test: GGFL Alignment Adapter
# -----------------------------------------------------------------------------

class TestTopologyP5ForAlignmentView:
    """Tests for topology_p5_for_alignment_view() GGFL adapter.

    TRUST BOUNDARY CONTRACT:
    - If schema_ok=False, status is "warn" and no derived scenario fields
    - Drivers are REASON CODES ONLY (max 3, deterministically ordered)
    - Summary is a single neutral sentence
    - shadow_mode_invariants block always present

    REASON CODE DRIVERS:
    - DRIVER_SCHEMA_NOT_OK
    - DRIVER_VALIDATION_NOT_PASSED
    - DRIVER_SCENARIO_MISMATCH
    - DRIVER_SCENARIO_XCOR_ANOMALY
    """

    def test_returns_fixed_shape_signal_type(self):
        """Test that signal_type is always SIG-TOP5."""
        signal = {"scenario": "HEALTHY", "validation_passed": True}
        view = topology_p5_for_alignment_view(signal)
        assert view["signal_type"] == "SIG-TOP5"

    def test_conflict_always_false_shadow_mode(self):
        """Test that conflict is always False (SHADOW MODE)."""
        for scenario in ["HEALTHY", "MISMATCH", "XCOR_ANOMALY", "MOCK_BASELINE"]:
            signal = {"scenario": scenario, "validation_passed": True}
            view = topology_p5_for_alignment_view(signal)
            assert view["conflict"] is False

    def test_schema_ok_false_returns_warn_status_with_reason_code(self):
        """Test that schema_ok=False returns warn status with DRIVER_SCHEMA_NOT_OK."""
        signal = {
            "schema_ok": False,
            "path": "test/path.json",
            "sha256": "abc123",
        }
        view = topology_p5_for_alignment_view(signal)

        assert view["status"] == "warn"
        assert view["drivers"] == [DRIVER_SCHEMA_NOT_OK]
        assert "invalid schema" in view["summary"].lower()
        # Should NOT include scenario fields
        assert "scenario" not in view or view.get("scenario") is None

    def test_schema_ok_false_only_includes_path_and_sha256(self):
        """Test that schema_ok=False only surfaces path and sha256."""
        signal = {
            "schema_ok": False,
            "path": "evidence/p5_report.json",
            "sha256": "deadbeef" * 8,
            "scenario": "SHOULD_NOT_APPEAR",  # Trust boundary - don't include
        }
        view = topology_p5_for_alignment_view(signal)

        assert view["path"] == "evidence/p5_report.json"
        assert view["sha256"] == "deadbeef" * 8
        # Scenario should NOT be included when schema_ok=False
        assert "scenario" not in view or view.get("scenario") is None

    def test_healthy_scenario_returns_ok_status(self):
        """Test that HEALTHY scenario returns ok status."""
        signal = {
            "scenario": "HEALTHY",
            "validation_passed": True,
            "joint_status": "ALIGNED",
            "schema_ok": True,
        }
        view = topology_p5_for_alignment_view(signal)

        assert view["status"] == "ok"
        assert "validated successfully" in view["summary"]

    def test_mismatch_scenario_returns_warn_status_with_reason_code(self):
        """Test that MISMATCH scenario returns warn status with DRIVER_SCENARIO_MISMATCH."""
        signal = {
            "scenario": "MISMATCH",
            "validation_passed": True,
            "joint_status": "DIVERGENT",
            "schema_ok": True,
        }
        view = topology_p5_for_alignment_view(signal)

        assert view["status"] == "warn"
        assert DRIVER_SCENARIO_MISMATCH in view["drivers"]
        assert "requires attention" in view["summary"]

    def test_xcor_anomaly_scenario_returns_warn_status_with_reason_code(self):
        """Test that XCOR_ANOMALY scenario returns warn status with DRIVER_SCENARIO_XCOR_ANOMALY."""
        signal = {
            "scenario": "XCOR_ANOMALY",
            "validation_passed": True,
            "joint_status": "TENSION",
            "schema_ok": True,
        }
        view = topology_p5_for_alignment_view(signal)

        assert view["status"] == "warn"
        assert DRIVER_SCENARIO_XCOR_ANOMALY in view["drivers"]

    def test_validation_failed_returns_warn_status_with_reason_code(self):
        """Test that validation_passed=False returns warn status with DRIVER_VALIDATION_NOT_PASSED."""
        signal = {
            "scenario": "HEALTHY",
            "validation_passed": False,
            "joint_status": "ALIGNED",
            "schema_ok": True,
        }
        view = topology_p5_for_alignment_view(signal)

        assert view["status"] == "warn"
        assert DRIVER_VALIDATION_NOT_PASSED in view["drivers"]

    def test_drivers_are_reason_codes_only(self):
        """Test that drivers contain only canonical reason codes."""
        valid_reason_codes = {
            DRIVER_SCHEMA_NOT_OK,
            DRIVER_VALIDATION_NOT_PASSED,
            DRIVER_SCENARIO_MISMATCH,
            DRIVER_SCENARIO_XCOR_ANOMALY,
        }
        signal = {
            "scenario": "MISMATCH",
            "validation_passed": False,
            "joint_status": "DIVERGENT",
            "schema_ok": True,
        }
        view = topology_p5_for_alignment_view(signal)

        for driver in view["drivers"]:
            assert driver in valid_reason_codes, f"Driver {driver} is not a valid reason code"

    def test_drivers_deterministically_ordered(self):
        """Test that reason code drivers are deterministically sorted."""
        signal = {
            "scenario": "MISMATCH",
            "validation_passed": False,
            "joint_status": "DIVERGENT",
            "schema_ok": True,
        }
        view = topology_p5_for_alignment_view(signal)

        # Drivers should be sorted alphabetically
        drivers = view["drivers"]
        assert drivers == sorted(drivers), "Reason code drivers must be sorted for determinism"

    def test_drivers_capped_at_three(self):
        """Test that drivers are capped at max 3."""
        signal = {
            "scenario": "MISMATCH",
            "validation_passed": False,
            "joint_status": "DIVERGENT",
            "scenario_confidence": 0.3,
            "shadow_mode_invariant_ok": False,
            "schema_ok": True,
        }
        view = topology_p5_for_alignment_view(signal)

        # Should have at most 3 drivers
        assert len(view["drivers"]) <= 3

    def test_summary_is_single_sentence(self):
        """Test that summary is a single neutral sentence."""
        signal = {"scenario": "HEALTHY", "validation_passed": True, "schema_ok": True}
        view = topology_p5_for_alignment_view(signal)

        summary = view["summary"]
        # Should end with period and be non-empty
        assert summary.endswith(".")
        # Should be reasonably short (single sentence)
        assert len(summary) < 100

    def test_output_is_json_serializable(self):
        """Test that output is JSON serializable."""
        signal = {
            "scenario": "MOCK_BASELINE",
            "validation_passed": True,
            "joint_status": "TENSION",
            "scenario_confidence": 0.85,
            "schema_ok": True,
            "path": "test.json",
            "sha256": "abc123",
        }
        view = topology_p5_for_alignment_view(signal)

        serialized = json.dumps(view)
        assert serialized is not None


class TestBuildTopologyP5StatusSignal:
    """Tests for build_topology_p5_status_signal() status signal builder.

    TRUST BOUNDARY CONTRACT:
    - If schema_ok=False: surfaces topology_p5.schema_ok=false with sha256+path only
    - If schema_ok=True: includes all derived scenario fields
    """

    def test_none_reference_returns_not_present(self):
        """Test that None reference returns present=False."""
        signal = build_topology_p5_status_signal(None)

        assert signal["present"] is False
        assert signal["schema_ok"] is True
        assert "No topology P5 report found" in signal["advisory_warning"]

    def test_schema_ok_false_excludes_scenario_fields(self):
        """Test that schema_ok=False excludes scenario fields (trust boundary)."""
        reference = {
            "schema_ok": False,
            "path": "test/report.json",
            "sha256": "abc123" * 10,
            "advisory_warning": "JSON parse error",
            # These should NOT appear in output
            "scenario": "SHOULD_NOT_APPEAR",
            "joint_status": "SHOULD_NOT_APPEAR",
        }
        signal = build_topology_p5_status_signal(reference)

        assert signal["present"] is True
        assert signal["schema_ok"] is False
        assert signal["path"] == "test/report.json"
        assert signal["sha256"] == "abc123" * 10
        assert signal["advisory_warning"] == "JSON parse error"
        # Trust boundary: no scenario fields
        assert "scenario" not in signal
        assert "joint_status" not in signal
        assert "validation_passed" not in signal

    def test_schema_ok_true_includes_all_fields(self):
        """Test that schema_ok=True includes all derived fields."""
        reference = {
            "schema_ok": True,
            "path": "evidence/p5_report.json",
            "sha256": "deadbeef" * 8,
            "scenario": "HEALTHY",
            "scenario_confidence": 0.95,
            "joint_status": "ALIGNED",
            "shadow_mode_invariant_ok": True,
            "validation_passed": True,
            "mode": "SHADOW",
        }
        signal = build_topology_p5_status_signal(reference)

        assert signal["present"] is True
        assert signal["schema_ok"] is True
        assert signal["scenario"] == "HEALTHY"
        assert signal["scenario_confidence"] == 0.95
        assert signal["joint_status"] == "ALIGNED"
        assert signal["shadow_mode_invariant_ok"] is True
        assert signal["validation_passed"] is True
        assert signal["mode"] == "SHADOW"

    def test_missing_schema_ok_defaults_to_true(self):
        """Test that missing schema_ok defaults to True."""
        reference = {
            "path": "test.json",
            "sha256": "abc123",
            "scenario": "MOCK_BASELINE",
        }
        signal = build_topology_p5_status_signal(reference)

        assert signal["schema_ok"] is True
        assert signal["scenario"] == "MOCK_BASELINE"

    def test_output_is_json_serializable(self):
        """Test that output is JSON serializable."""
        reference = {
            "schema_ok": True,
            "path": "test.json",
            "sha256": "abc123",
            "scenario": "HEALTHY",
            "validation_passed": True,
        }
        signal = build_topology_p5_status_signal(reference)

        serialized = json.dumps(signal)
        assert serialized is not None


class TestTrustBoundaryContract:
    """Tests for TRUST BOUNDARY contract across all adapters.

    Contract:
    - schema_ok=False must NOT fabricate scenario data
    - Only path + sha256 are trusted when schema_ok=False
    """

    def test_alignment_view_trust_boundary(self):
        """Test GGFL adapter respects trust boundary."""
        malformed = {
            "schema_ok": False,
            "path": "corrupted.json",
            "sha256": "hash123",
            "scenario": "FAKE_SCENARIO",  # Should be ignored
            "joint_status": "FAKE_STATUS",  # Should be ignored
        }
        view = topology_p5_for_alignment_view(malformed)

        # Should not include the fake scenario data
        assert view.get("scenario") is None or "scenario" not in view
        assert view["status"] == "warn"
        assert view["path"] == "corrupted.json"
        assert view["sha256"] == "hash123"

    def test_status_signal_trust_boundary(self):
        """Test status signal builder respects trust boundary."""
        malformed = {
            "schema_ok": False,
            "path": "corrupted.json",
            "sha256": "hash123",
            "scenario": "FAKE_SCENARIO",  # Should be excluded
            "validation_passed": True,  # Should be excluded
        }
        signal = build_topology_p5_status_signal(malformed)

        # Should not include scenario fields
        assert "scenario" not in signal
        assert "validation_passed" not in signal
        assert signal["schema_ok"] is False

    def test_single_warning_cap_in_drivers(self):
        """Test that drivers list has single warning type per category."""
        signal = {
            "scenario": "MISMATCH",
            "validation_passed": False,
            "joint_status": "DIVERGENT",
            "schema_ok": True,
        }
        view = topology_p5_for_alignment_view(signal)

        # Each driver should be unique (reason codes are unique by design)
        assert len(view["drivers"]) == len(set(view["drivers"])), \
            "Each driver reason code should appear at most once"


# -----------------------------------------------------------------------------
# Test: Extraction Source Provenance
# -----------------------------------------------------------------------------

class TestExtractionSourceProvenance:
    """Tests for extraction_source provenance tracking.

    PROVENANCE VALUES:
    - MANIFEST: Loaded from manifest.json governance block
    - EVIDENCE_JSON: Loaded from evidence pack JSON file
    - RUN_DIR_ROOT: Loaded from run directory root
    - P4_SHADOW: Loaded from p4_shadow subdirectory
    - MISSING: No report found
    """

    def test_extraction_source_constants_defined(self):
        """Test that all extraction source constants are defined."""
        assert EXTRACTION_SOURCE_MANIFEST == "MANIFEST"
        assert EXTRACTION_SOURCE_EVIDENCE_JSON == "EVIDENCE_JSON"
        assert EXTRACTION_SOURCE_RUN_DIR_ROOT == "RUN_DIR_ROOT"
        assert EXTRACTION_SOURCE_P4_SHADOW == "P4_SHADOW"
        assert EXTRACTION_SOURCE_MISSING == "MISSING"

    def test_status_signal_includes_extraction_source_manifest(self):
        """Test status signal includes MANIFEST extraction source."""
        reference = {
            "schema_ok": True,
            "path": "manifest.json",
            "sha256": "abc123",
            "scenario": "HEALTHY",
            "extraction_source": EXTRACTION_SOURCE_MANIFEST,
        }
        signal = build_topology_p5_status_signal(reference)

        assert signal["extraction_source"] == EXTRACTION_SOURCE_MANIFEST

    def test_status_signal_includes_extraction_source_evidence_json(self):
        """Test status signal includes EVIDENCE_JSON extraction source."""
        reference = {
            "schema_ok": True,
            "path": "evidence/p5_report.json",
            "sha256": "def456",
            "scenario": "HEALTHY",
            "extraction_source": EXTRACTION_SOURCE_EVIDENCE_JSON,
        }
        signal = build_topology_p5_status_signal(reference)

        assert signal["extraction_source"] == EXTRACTION_SOURCE_EVIDENCE_JSON

    def test_status_signal_includes_extraction_source_run_dir_root(self):
        """Test status signal includes RUN_DIR_ROOT extraction source."""
        reference = {
            "schema_ok": True,
            "path": "run/p5_report.json",
            "sha256": "ghi789",
            "scenario": "MOCK_BASELINE",
            "extraction_source": EXTRACTION_SOURCE_RUN_DIR_ROOT,
        }
        signal = build_topology_p5_status_signal(reference)

        assert signal["extraction_source"] == EXTRACTION_SOURCE_RUN_DIR_ROOT

    def test_status_signal_includes_extraction_source_p4_shadow(self):
        """Test status signal includes P4_SHADOW extraction source."""
        reference = {
            "schema_ok": True,
            "path": "p4_shadow/report.json",
            "sha256": "jkl012",
            "scenario": "XCOR_ANOMALY",
            "extraction_source": EXTRACTION_SOURCE_P4_SHADOW,
        }
        signal = build_topology_p5_status_signal(reference)

        assert signal["extraction_source"] == EXTRACTION_SOURCE_P4_SHADOW

    def test_status_signal_none_reference_returns_missing(self):
        """Test None reference returns MISSING extraction source."""
        signal = build_topology_p5_status_signal(None)

        assert signal["extraction_source"] == EXTRACTION_SOURCE_MISSING
        assert signal["present"] is False

    def test_status_signal_defaults_to_parameter_extraction_source(self):
        """Test status signal uses parameter extraction_source when not in reference."""
        reference = {
            "schema_ok": True,
            "path": "report.json",
            "sha256": "xyz123",
            "scenario": "HEALTHY",
            # No extraction_source in reference
        }
        signal = build_topology_p5_status_signal(
            reference,
            extraction_source=EXTRACTION_SOURCE_EVIDENCE_JSON,
        )

        assert signal["extraction_source"] == EXTRACTION_SOURCE_EVIDENCE_JSON

    def test_alignment_view_includes_extraction_source(self):
        """Test alignment view includes extraction_source."""
        signal = {
            "scenario": "HEALTHY",
            "validation_passed": True,
            "schema_ok": True,
            "path": "test.json",
            "sha256": "abc123",
            "extraction_source": EXTRACTION_SOURCE_MANIFEST,
        }
        view = topology_p5_for_alignment_view(signal)

        assert view["extraction_source"] == EXTRACTION_SOURCE_MANIFEST

    def test_alignment_view_defaults_to_missing(self):
        """Test alignment view defaults extraction_source to MISSING."""
        signal = {
            "scenario": "HEALTHY",
            "validation_passed": True,
            "schema_ok": True,
            "path": "test.json",
            "sha256": "abc123",
            # No extraction_source
        }
        view = topology_p5_for_alignment_view(signal)

        assert view["extraction_source"] == EXTRACTION_SOURCE_MISSING

    def test_alignment_view_schema_not_ok_preserves_extraction_source(self):
        """Test alignment view preserves extraction_source even when schema_ok=False."""
        signal = {
            "schema_ok": False,
            "path": "corrupted.json",
            "sha256": "bad123",
            "extraction_source": EXTRACTION_SOURCE_P4_SHADOW,
        }
        view = topology_p5_for_alignment_view(signal)

        assert view["extraction_source"] == EXTRACTION_SOURCE_P4_SHADOW


# -----------------------------------------------------------------------------
# Test: Shadow Mode Invariants Block
# -----------------------------------------------------------------------------

class TestShadowModeInvariantsBlock:
    """Tests for shadow_mode_invariants block in GGFL output.

    SHADOW MODE INVARIANTS:
    - advisory_only: True (always in shadow mode)
    - no_enforcement: True (always in shadow mode)
    - conflict_invariant: False (shadow mode never triggers conflicts)
    """

    def test_shadow_mode_invariants_always_present(self):
        """Test shadow_mode_invariants block is always present."""
        signal = {
            "scenario": "HEALTHY",
            "validation_passed": True,
            "schema_ok": True,
        }
        view = topology_p5_for_alignment_view(signal)

        assert "shadow_mode_invariants" in view
        assert isinstance(view["shadow_mode_invariants"], dict)

    def test_shadow_mode_invariants_advisory_only(self):
        """Test advisory_only is always True."""
        signal = {"scenario": "HEALTHY", "validation_passed": True, "schema_ok": True}
        view = topology_p5_for_alignment_view(signal)

        assert view["shadow_mode_invariants"]["advisory_only"] is True

    def test_shadow_mode_invariants_no_enforcement(self):
        """Test no_enforcement is always True."""
        signal = {"scenario": "MISMATCH", "validation_passed": True, "schema_ok": True}
        view = topology_p5_for_alignment_view(signal)

        assert view["shadow_mode_invariants"]["no_enforcement"] is True

    def test_shadow_mode_invariants_conflict_invariant_false(self):
        """Test conflict_invariant is always False (shadow mode)."""
        signal = {"scenario": "XCOR_ANOMALY", "validation_passed": False, "schema_ok": True}
        view = topology_p5_for_alignment_view(signal)

        assert view["shadow_mode_invariants"]["conflict_invariant"] is False

    def test_shadow_mode_invariants_present_when_schema_not_ok(self):
        """Test shadow_mode_invariants present even when schema_ok=False."""
        signal = {
            "schema_ok": False,
            "path": "bad.json",
            "sha256": "xyz",
        }
        view = topology_p5_for_alignment_view(signal)

        assert "shadow_mode_invariants" in view
        assert view["shadow_mode_invariants"]["advisory_only"] is True
        assert view["shadow_mode_invariants"]["no_enforcement"] is True
        assert view["shadow_mode_invariants"]["conflict_invariant"] is False

    def test_shadow_mode_invariants_immutable_across_scenarios(self):
        """Test shadow_mode_invariants are identical across all scenarios."""
        scenarios = [
            {"scenario": "HEALTHY", "validation_passed": True, "schema_ok": True},
            {"scenario": "MISMATCH", "validation_passed": True, "schema_ok": True},
            {"scenario": "XCOR_ANOMALY", "validation_passed": False, "schema_ok": True},
            {"scenario": "MOCK_BASELINE", "validation_passed": True, "schema_ok": True},
        ]

        views = [topology_p5_for_alignment_view(s) for s in scenarios]
        invariants_sets = [frozenset(v["shadow_mode_invariants"].items()) for v in views]

        # All should be identical
        assert len(set(invariants_sets)) == 1, "Shadow mode invariants must be identical across scenarios"

    def test_shadow_mode_invariants_json_serializable(self):
        """Test shadow_mode_invariants block is JSON serializable."""
        signal = {"scenario": "HEALTHY", "validation_passed": True, "schema_ok": True}
        view = topology_p5_for_alignment_view(signal)

        serialized = json.dumps(view["shadow_mode_invariants"])
        deserialized = json.loads(serialized)
        assert deserialized == view["shadow_mode_invariants"]


# -----------------------------------------------------------------------------
# Test: Reason Code Deterministic Ordering
# -----------------------------------------------------------------------------

class TestReasonCodeDeterministicOrdering:
    """Tests for deterministic ordering of reason code drivers.

    CONTRACT:
    - Drivers are reason codes ONLY (no free text)
    - Drivers are sorted alphabetically for determinism
    - Maximum 3 drivers
    """

    def test_drivers_sorted_alphabetically(self):
        """Test drivers are sorted alphabetically."""
        signal = {
            "scenario": "MISMATCH",
            "validation_passed": False,
            "schema_ok": True,
        }
        view = topology_p5_for_alignment_view(signal)

        drivers = view["drivers"]
        assert drivers == sorted(drivers), "Drivers must be sorted alphabetically"

    def test_multiple_drivers_deterministic_order(self):
        """Test multiple drivers maintain deterministic order across calls."""
        signal = {
            "scenario": "MISMATCH",
            "validation_passed": False,
            "schema_ok": True,
        }

        # Call multiple times and verify same order
        results = [topology_p5_for_alignment_view(signal)["drivers"] for _ in range(5)]

        assert all(r == results[0] for r in results), "Driver order must be deterministic"

    def test_validation_mismatch_driver_order(self):
        """Test DRIVER_VALIDATION_NOT_PASSED comes before DRIVER_SCENARIO_MISMATCH."""
        signal = {
            "scenario": "MISMATCH",
            "validation_passed": False,
            "schema_ok": True,
        }
        view = topology_p5_for_alignment_view(signal)

        # Both should be present
        assert DRIVER_VALIDATION_NOT_PASSED in view["drivers"]
        assert DRIVER_SCENARIO_MISMATCH in view["drivers"]

        # DRIVER_SCENARIO_MISMATCH < DRIVER_VALIDATION_NOT_PASSED alphabetically
        idx_mismatch = view["drivers"].index(DRIVER_SCENARIO_MISMATCH)
        idx_validation = view["drivers"].index(DRIVER_VALIDATION_NOT_PASSED)
        assert idx_mismatch < idx_validation, "Drivers must be alphabetically sorted"

    def test_xcor_anomaly_with_validation_failed_order(self):
        """Test XCOR_ANOMALY + validation failed driver ordering."""
        signal = {
            "scenario": "XCOR_ANOMALY",
            "validation_passed": False,
            "schema_ok": True,
        }
        view = topology_p5_for_alignment_view(signal)

        # Both should be present
        assert DRIVER_VALIDATION_NOT_PASSED in view["drivers"]
        assert DRIVER_SCENARIO_XCOR_ANOMALY in view["drivers"]

        # DRIVER_SCENARIO_XCOR_ANOMALY < DRIVER_VALIDATION_NOT_PASSED alphabetically
        idx_xcor = view["drivers"].index(DRIVER_SCENARIO_XCOR_ANOMALY)
        idx_validation = view["drivers"].index(DRIVER_VALIDATION_NOT_PASSED)
        assert idx_xcor < idx_validation, "Drivers must be alphabetically sorted"

    def test_schema_not_ok_single_driver(self):
        """Test schema_ok=False produces single DRIVER_SCHEMA_NOT_OK."""
        signal = {
            "schema_ok": False,
            "path": "bad.json",
            "sha256": "xyz",
        }
        view = topology_p5_for_alignment_view(signal)

        assert view["drivers"] == [DRIVER_SCHEMA_NOT_OK]

    def test_healthy_scenario_no_drivers(self):
        """Test HEALTHY scenario with validation passed has empty drivers."""
        signal = {
            "scenario": "HEALTHY",
            "validation_passed": True,
            "schema_ok": True,
        }
        view = topology_p5_for_alignment_view(signal)

        assert view["drivers"] == []

    def test_max_three_drivers(self):
        """Test drivers capped at 3."""
        signal = {
            "scenario": "MISMATCH",
            "validation_passed": False,
            "schema_ok": True,
        }
        view = topology_p5_for_alignment_view(signal)

        assert len(view["drivers"]) <= 3

    def test_no_free_text_drivers(self):
        """Test that all drivers are canonical reason codes, not free text."""
        valid_codes = {
            DRIVER_SCHEMA_NOT_OK,
            DRIVER_VALIDATION_NOT_PASSED,
            DRIVER_SCENARIO_MISMATCH,
            DRIVER_SCENARIO_XCOR_ANOMALY,
        }

        test_signals = [
            {"scenario": "HEALTHY", "validation_passed": True, "schema_ok": True},
            {"scenario": "MISMATCH", "validation_passed": False, "schema_ok": True},
            {"scenario": "XCOR_ANOMALY", "validation_passed": False, "schema_ok": True},
            {"schema_ok": False, "path": "x.json", "sha256": "abc"},
        ]

        for sig in test_signals:
            view = topology_p5_for_alignment_view(sig)
            for driver in view["drivers"]:
                assert driver in valid_codes, f"Driver '{driver}' is not a valid reason code"
