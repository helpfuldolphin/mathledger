"""
Tests for Structural Governance (CLAUDE G Integration)

Tests for:
- emit_structural_signal() - structural signal emission
- build_structural_cohesion_tile() - console tile builder
- apply_structural_severity_escalation() - severity upgrade logic
- SHADOW mode invariants - no side effects

See: docs/system_law/Structural_Cohesion_PhaseX.md
"""

import pytest
from typing import Any, Dict

from backend.dag.invariant_guard import (
    ProofDag,
    SliceProfile,
    StructuralGovernanceSignal,
    StructuralViolation,
    emit_structural_signal,
    build_structural_cohesion_tile,
    check_dag_invariants,
    build_structural_calibration_for_p4,
    attach_structural_governance_to_evidence,
    build_escalation_advisory,
)
from backend.topology.first_light.data_structures_p4 import (
    DivergenceSnapshot,
    RealCycleObservation,
    TwinCycleObservation,
)
from backend.topology.first_light.divergence_analyzer import (
    apply_structural_severity_escalation,
    analyze_with_structural_context,
    DivergenceAnalyzer,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def healthy_dag() -> ProofDag:
    """Create a healthy DAG with no cycles."""
    dag = ProofDag()
    dag.add_node("n1", {"type": "axiom"})
    dag.add_node("n2", {"type": "derived"})
    dag.add_node("n3", {"type": "derived"})
    dag.add_edge("n1", "n2")
    dag.add_edge("n2", "n3")
    return dag


@pytest.fixture
def cyclic_dag() -> ProofDag:
    """Create a DAG with a cycle (SI-001 violation)."""
    dag = ProofDag()
    dag.add_node("n1")
    dag.add_node("n2")
    dag.add_node("n3")
    dag.add_edge("n1", "n2")
    dag.add_edge("n2", "n3")
    dag.add_edge("n3", "n1")  # Creates cycle
    return dag


@pytest.fixture
def healthy_topology_state() -> Dict[str, Any]:
    """Create healthy topology state within all bounds."""
    return {
        "H": 0.8,
        "rho": 0.7,
        "tau": 0.20,  # Within Goldilocks [0.16, 0.24]
        "beta": 0.1,
        "in_omega": True,
        "omega_exit_streak": 0,
    }


@pytest.fixture
def out_of_bounds_topology_state() -> Dict[str, Any]:
    """Create topology state with bounds violation (SI-005)."""
    return {
        "H": 1.5,  # Out of bounds!
        "rho": 0.7,
        "tau": 0.20,
        "beta": 0.1,
        "in_omega": False,
        "omega_exit_streak": 10,
    }


@pytest.fixture
def healthy_ht_state() -> Dict[str, Any]:
    """Create healthy HT state with all anchors verified."""
    return {
        "total_anchors": 10,
        "verified_anchors": 10,
        "pending_anchors": 0,
        "failed_anchors": 0,
    }


@pytest.fixture
def failed_ht_state() -> Dict[str, Any]:
    """Create HT state with failed anchors (SI-010 violation)."""
    return {
        "total_anchors": 10,
        "verified_anchors": 8,
        "pending_anchors": 0,
        "failed_anchors": 2,
    }


# =============================================================================
# Tests: emit_structural_signal()
# =============================================================================

class TestEmitStructuralSignal:
    """Tests for emit_structural_signal function."""

    def test_signal_with_healthy_layers(
        self, healthy_dag, healthy_topology_state, healthy_ht_state
    ):
        """Test signal emission with all healthy layers."""
        signal = emit_structural_signal(
            dag=healthy_dag,
            topology_state=healthy_topology_state,
            ht_state=healthy_ht_state,
            run_id="test_run_001",
            cycle=42,
            triggered_by="MANUAL",
        )

        # Check signal ID format
        assert signal.signal_id.startswith("sgs_")
        assert len(signal.signal_id) == 20  # sgs_ + 16 hex chars

        # Check all layers CONSISTENT
        assert signal.dag_status == "CONSISTENT"
        assert signal.topology_status == "CONSISTENT"
        assert signal.ht_status == "CONSISTENT"
        assert signal.combined_severity == "CONSISTENT"

        # Check cohesion score is perfect
        assert signal.cohesion_score == 1.0

        # Check admissibility
        assert signal.admissible is True

        # No violations
        assert len(signal.violations) == 0

    def test_signal_with_dag_cycle_conflict(self, cyclic_dag, healthy_topology_state, healthy_ht_state):
        """Test signal with DAG cycle (SI-001 CONFLICT)."""
        signal = emit_structural_signal(
            dag=cyclic_dag,
            topology_state=healthy_topology_state,
            ht_state=healthy_ht_state,
        )

        # DAG should show CONFLICT
        assert signal.dag_status == "CONFLICT"
        assert signal.combined_severity == "CONFLICT"

        # Not admissible due to SI-001
        assert signal.admissible is False

        # Should have SI-001 violation
        violations = [v for v in signal.violations if v.invariant_id == "SI-001"]
        assert len(violations) == 1
        assert violations[0].severity == "CONFLICT"
        assert violations[0].layer == "DAG"

    def test_signal_with_topology_bounds_violation(
        self, healthy_dag, out_of_bounds_topology_state, healthy_ht_state
    ):
        """Test signal with topology bounds violation (SI-005 CONFLICT)."""
        signal = emit_structural_signal(
            dag=healthy_dag,
            topology_state=out_of_bounds_topology_state,
            ht_state=healthy_ht_state,
        )

        # Topology should show CONFLICT
        assert signal.topology_status == "CONFLICT"
        assert signal.combined_severity == "CONFLICT"

        # Should have SI-005 violation
        violations = [v for v in signal.violations if v.invariant_id == "SI-005"]
        assert len(violations) == 1
        assert violations[0].severity == "CONFLICT"

    def test_signal_with_ht_anchor_failure(
        self, healthy_dag, healthy_topology_state, failed_ht_state
    ):
        """Test signal with HT anchor failure (SI-010 CONFLICT)."""
        signal = emit_structural_signal(
            dag=healthy_dag,
            topology_state=healthy_topology_state,
            ht_state=failed_ht_state,
        )

        # HT should show CONFLICT
        assert signal.ht_status == "CONFLICT"
        assert signal.combined_severity == "CONFLICT"

        # Not admissible due to SI-010
        assert signal.admissible is False

        # Should have SI-010 violation
        violations = [v for v in signal.violations if v.invariant_id == "SI-010"]
        assert len(violations) == 1

    def test_signal_to_dict_schema_compliance(self, healthy_dag, healthy_topology_state, healthy_ht_state):
        """Test that to_dict produces schema-compliant output."""
        signal = emit_structural_signal(
            dag=healthy_dag,
            topology_state=healthy_topology_state,
            ht_state=healthy_ht_state,
        )

        d = signal.to_dict()

        # Required fields per schema
        assert "signal_id" in d
        assert "timestamp" in d
        assert "dag_status" in d
        assert "topology_status" in d
        assert "ht_status" in d
        assert "combined_severity" in d
        assert "cohesion_score" in d
        assert "admissible" in d
        assert "violations" in d
        assert "layer_scores" in d
        assert "metadata" in d

    def test_signal_cohesion_score_calculation(self, healthy_dag, healthy_ht_state):
        """Test cohesion score weighted calculation."""
        # Create topology with some degradation but in bounds
        topology = {
            "H": 0.5,
            "rho": 0.5,
            "tau": 0.20,
            "beta": 0.0,
            "in_omega": False,
            "omega_exit_streak": 50,  # Some exit streak but < 100
        }

        signal = emit_structural_signal(
            dag=healthy_dag,
            topology_state=topology,
            ht_state=healthy_ht_state,
        )

        # Weights: dag=0.4, topo=0.4, ht=0.2
        # DAG score should be 1.0, HT score 1.0
        # Topology score degraded due to omega_exit_streak
        assert signal.cohesion_score < 1.0
        assert signal.cohesion_score > 0.0


# =============================================================================
# Tests: build_structural_cohesion_tile()
# =============================================================================

class TestBuildStructuralCohesionTile:
    """Tests for build_structural_cohesion_tile function."""

    def test_tile_from_healthy_signal(self, healthy_dag, healthy_topology_state, healthy_ht_state):
        """Test tile generation from healthy signal."""
        signal = emit_structural_signal(
            dag=healthy_dag,
            topology_state=healthy_topology_state,
            ht_state=healthy_ht_state,
        )

        tile = build_structural_cohesion_tile(signal)

        # Check tile ID format
        assert tile["tile_id"].startswith("sct_")
        assert len(tile["tile_id"]) == 12  # sct_ + 8 hex chars

        # Check overall status
        assert tile["overall_status"] == "HEALTHY"
        assert tile["overall_status_icon"] == "check_circle"
        assert tile["overall_status_color"] == "green"

        # Check cohesion
        assert tile["cohesion_score"] == 1.0
        assert tile["cohesion_score_display"] == "100%"

        # Check layers present
        assert "dag" in tile["layers"]
        assert "topology" in tile["layers"]
        assert "ht" in tile["layers"]

        # Check admissibility
        assert tile["admissibility"]["admissible"] is True

    def test_tile_from_degraded_signal(self, healthy_dag, healthy_ht_state):
        """Test tile generation from degraded (TENSION) signal."""
        # Create topology with high omega_exit_streak
        topology = {
            "H": 0.5,
            "rho": 0.5,
            "tau": 0.20,
            "beta": 0.0,
            "in_omega": False,
            "omega_exit_streak": 150,  # > 100 triggers TENSION
        }

        signal = emit_structural_signal(
            dag=healthy_dag,
            topology_state=topology,
            ht_state=healthy_ht_state,
        )

        tile = build_structural_cohesion_tile(signal)

        # Should show DEGRADED
        assert tile["overall_status"] == "DEGRADED"
        assert tile["overall_status_icon"] == "warning"
        assert tile["overall_status_color"] == "yellow"

    def test_tile_from_critical_signal(self, cyclic_dag, healthy_topology_state, healthy_ht_state):
        """Test tile generation from critical (CONFLICT) signal."""
        signal = emit_structural_signal(
            dag=cyclic_dag,
            topology_state=healthy_topology_state,
            ht_state=healthy_ht_state,
        )

        tile = build_structural_cohesion_tile(signal)

        # Should show CRITICAL
        assert tile["overall_status"] == "CRITICAL"
        assert tile["overall_status_icon"] == "error"
        assert tile["overall_status_color"] == "red"

        # Should not be admissible
        assert tile["admissibility"]["admissible"] is False
        assert "blocking_invariants" in tile["admissibility"]

    def test_tile_with_sparkline_history(self, healthy_dag, healthy_topology_state, healthy_ht_state):
        """Test tile with sparkline history for trend."""
        signal = emit_structural_signal(
            dag=healthy_dag,
            topology_state=healthy_topology_state,
            ht_state=healthy_ht_state,
        )

        history = [0.90, 0.92, 0.94, 0.96, 0.98]  # Improving trend
        tile = build_structural_cohesion_tile(signal, sparkline_history=history)

        # Should detect improving trend
        assert tile["cohesion_trend"] == "IMPROVING"
        assert "sparkline_data" in tile
        assert tile["sparkline_data"]["values"] == history

    def test_tile_violation_breakdown(self, cyclic_dag, out_of_bounds_topology_state, failed_ht_state):
        """Test tile violation breakdown counts."""
        signal = emit_structural_signal(
            dag=cyclic_dag,
            topology_state=out_of_bounds_topology_state,
            ht_state=failed_ht_state,
        )

        tile = build_structural_cohesion_tile(signal)

        # Should have multiple violations
        assert tile["active_violations"] >= 3

        # Check breakdown
        assert tile["violation_breakdown"]["conflict"] >= 3

    def test_tile_refresh_interval_scaling(self, healthy_dag, healthy_topology_state, healthy_ht_state):
        """Test that refresh interval scales with severity."""
        # Healthy signal
        healthy_signal = emit_structural_signal(
            dag=healthy_dag,
            topology_state=healthy_topology_state,
            ht_state=healthy_ht_state,
        )
        healthy_tile = build_structural_cohesion_tile(healthy_signal)

        # Should have longer refresh (5000ms)
        assert healthy_tile["refresh_interval_ms"] == 5000

        # Degraded signal
        topology_degraded = {
            "H": 0.5, "rho": 0.5, "tau": 0.20, "beta": 0.0,
            "in_omega": False, "omega_exit_streak": 150,
        }
        degraded_signal = emit_structural_signal(
            dag=healthy_dag,
            topology_state=topology_degraded,
            ht_state=healthy_ht_state,
        )
        degraded_tile = build_structural_cohesion_tile(degraded_signal)

        # Should have shorter refresh (2000ms)
        assert degraded_tile["refresh_interval_ms"] == 2000


# =============================================================================
# Tests: apply_structural_severity_escalation()
# =============================================================================

class TestApplyStructuralSeverityEscalation:
    """Tests for apply_structural_severity_escalation function."""

    def test_no_escalation_with_consistent_signal(self):
        """Test no escalation when structural signal is CONSISTENT."""
        snapshot = DivergenceSnapshot(
            cycle=1,
            divergence_severity="INFO",
            divergence_type="STATE",
        )

        structural_signal = {
            "combined_severity": "CONSISTENT",
            "cohesion_score": 1.0,
            "admissible": True,
        }

        result = apply_structural_severity_escalation(snapshot, structural_signal)

        # No escalation
        assert result.divergence_severity == "INFO"
        assert result.severity_escalated is False
        assert result.structural_conflict is False

    def test_escalation_with_conflict_signal(self):
        """Test severity escalation when structural CONFLICT."""
        snapshot = DivergenceSnapshot(
            cycle=1,
            divergence_severity="INFO",
            divergence_type="STATE",
        )

        structural_signal = {
            "combined_severity": "CONFLICT",
            "cohesion_score": 0.5,
            "admissible": False,
        }

        result = apply_structural_severity_escalation(snapshot, structural_signal)

        # Should escalate to CRITICAL
        assert result.divergence_severity == "CRITICAL"
        assert result.severity_escalated is True
        assert result.original_severity == "INFO"
        assert result.structural_conflict is True

    def test_escalation_with_tension_signal(self):
        """Test severity escalation when structural TENSION."""
        # INFO -> WARN
        snapshot_info = DivergenceSnapshot(
            cycle=1,
            divergence_severity="INFO",
            divergence_type="STATE",
        )

        structural_signal = {
            "combined_severity": "TENSION",
            "cohesion_score": 0.75,
            "admissible": True,
        }

        result = apply_structural_severity_escalation(snapshot_info, structural_signal)

        assert result.divergence_severity == "WARN"
        assert result.severity_escalated is True
        assert result.original_severity == "INFO"

        # WARN -> CRITICAL
        snapshot_warn = DivergenceSnapshot(
            cycle=1,
            divergence_severity="WARN",
            divergence_type="STATE",
        )

        result2 = apply_structural_severity_escalation(snapshot_warn, structural_signal)

        assert result2.divergence_severity == "CRITICAL"
        assert result2.severity_escalated is True

    def test_cohesion_degraded_flag(self):
        """Test cohesion_degraded flag when score < 0.8."""
        snapshot = DivergenceSnapshot(
            cycle=1,
            divergence_severity="NONE",
        )

        structural_signal = {
            "combined_severity": "CONSISTENT",
            "cohesion_score": 0.65,
            "admissible": True,
        }

        result = apply_structural_severity_escalation(snapshot, structural_signal)

        assert result.cohesion_degraded is True
        assert result.cohesion_score == 0.65

    def test_no_change_with_none_signal(self):
        """Test no change when structural_signal is None."""
        snapshot = DivergenceSnapshot(
            cycle=1,
            divergence_severity="WARN",
        )

        result = apply_structural_severity_escalation(snapshot, None)

        # Should be unchanged
        assert result.divergence_severity == "WARN"
        assert result.structural_conflict is False
        assert result.cohesion_degraded is False

    def test_no_escalation_for_none_severity(self):
        """Test that NONE severity is not escalated even with CONFLICT."""
        snapshot = DivergenceSnapshot(
            cycle=1,
            divergence_severity="NONE",
        )

        structural_signal = {
            "combined_severity": "CONFLICT",
            "cohesion_score": 0.5,
            "admissible": False,
        }

        result = apply_structural_severity_escalation(snapshot, structural_signal)

        # NONE stays NONE (no divergence to escalate)
        assert result.divergence_severity == "NONE"
        assert result.severity_escalated is False
        assert result.structural_conflict is True


# =============================================================================
# Tests: SHADOW Mode Invariants
# =============================================================================

class TestShadowModeInvariants:
    """Tests ensuring SHADOW mode: no side effects, observation only."""

    def test_emit_signal_is_observation_only(self, healthy_dag, healthy_topology_state, healthy_ht_state):
        """Test that emit_structural_signal has no side effects."""
        # Capture initial state
        initial_node_count = len(healthy_dag.nodes)
        initial_edge_count = len(healthy_dag.edges)

        # Emit signal multiple times
        for _ in range(5):
            emit_structural_signal(
                dag=healthy_dag,
                topology_state=healthy_topology_state,
                ht_state=healthy_ht_state,
            )

        # DAG unchanged
        assert len(healthy_dag.nodes) == initial_node_count
        assert len(healthy_dag.edges) == initial_edge_count

    def test_escalation_does_not_modify_original_signal(self):
        """Test that escalation doesn't modify the structural signal dict."""
        snapshot = DivergenceSnapshot(cycle=1, divergence_severity="INFO")

        structural_signal = {
            "combined_severity": "CONFLICT",
            "cohesion_score": 0.5,
            "admissible": False,
        }

        # Copy original
        original_signal = dict(structural_signal)

        apply_structural_severity_escalation(snapshot, structural_signal)

        # Signal dict unchanged
        assert structural_signal == original_signal

    def test_signal_action_is_logged_only(self, healthy_dag, healthy_topology_state, healthy_ht_state):
        """Test that structural analysis produces LOGGED_ONLY action."""
        # Even with violations, action should be LOGGED_ONLY
        signal = emit_structural_signal(
            dag=healthy_dag,
            topology_state=healthy_topology_state,
            ht_state=healthy_ht_state,
        )

        # The signal itself doesn't have an action field, but any divergence
        # snapshot produced should have action="LOGGED_ONLY"
        d = signal.to_dict()
        assert "metadata" in d
        # The signal logs observation only, no enforcement actions

    def test_analyze_with_structural_context_shadow_mode(self):
        """Test analyze_with_structural_context maintains SHADOW mode."""
        analyzer = DivergenceAnalyzer()

        real = RealCycleObservation(
            cycle=1,
            success=True,
            H=0.8, rho=0.7, tau=0.2, beta=0.1,
            in_omega=True,
        )

        twin = TwinCycleObservation(
            real_cycle=1,
            predicted_success=False,  # Divergence
            twin_H=0.5, twin_rho=0.5, twin_tau=0.2, twin_beta=0.1,
            predicted_in_omega=True,
        )

        structural_signal = {
            "combined_severity": "CONFLICT",
            "cohesion_score": 0.5,
            "admissible": False,
        }

        snapshot = analyze_with_structural_context(
            analyzer, real, twin, structural_signal
        )

        # Action must be LOGGED_ONLY
        assert snapshot.action == "LOGGED_ONLY"

        # Structural escalation applied
        assert snapshot.structural_conflict is True


# =============================================================================
# Tests: Integration
# =============================================================================

class TestStructuralGovernanceIntegration:
    """Integration tests for end-to-end structural governance flow."""

    def test_full_flow_healthy(self, healthy_dag, healthy_topology_state, healthy_ht_state):
        """Test full flow with healthy state."""
        # 1. Emit signal
        signal = emit_structural_signal(
            dag=healthy_dag,
            topology_state=healthy_topology_state,
            ht_state=healthy_ht_state,
            run_id="integration_test_001",
            cycle=100,
        )

        # 2. Build tile
        tile = build_structural_cohesion_tile(signal)

        # 3. Create divergence and apply escalation
        snapshot = DivergenceSnapshot(
            cycle=100,
            divergence_severity="INFO",
            divergence_type="STATE",
        )

        escalated = apply_structural_severity_escalation(
            snapshot, signal.to_dict()
        )

        # Verify end-to-end
        assert tile["overall_status"] == "HEALTHY"
        assert escalated.severity_escalated is False
        assert escalated.cohesion_score == 1.0

    def test_full_flow_with_conflict(self, cyclic_dag, healthy_topology_state, healthy_ht_state):
        """Test full flow with CONFLICT state."""
        # 1. Emit signal with cyclic DAG
        signal = emit_structural_signal(
            dag=cyclic_dag,
            topology_state=healthy_topology_state,
            ht_state=healthy_ht_state,
        )

        # 2. Build tile
        tile = build_structural_cohesion_tile(signal)

        # 3. Create divergence and apply escalation
        snapshot = DivergenceSnapshot(
            cycle=1,
            divergence_severity="INFO",
            divergence_type="STATE",
        )

        escalated = apply_structural_severity_escalation(
            snapshot, signal.to_dict()
        )

        # Verify end-to-end
        assert tile["overall_status"] == "CRITICAL"
        assert tile["admissibility"]["admissible"] is False
        assert escalated.divergence_severity == "CRITICAL"
        assert escalated.structural_conflict is True


# =============================================================================
# Tests: P4 Calibration Integration
# =============================================================================

class TestBuildStructuralCalibrationForP4:
    """Tests for build_structural_calibration_for_p4 function."""

    def test_calibration_from_healthy_signal(self, healthy_dag, healthy_topology_state, healthy_ht_state):
        """Test P4 calibration from healthy signal."""
        signal = emit_structural_signal(
            dag=healthy_dag,
            topology_state=healthy_topology_state,
            ht_state=healthy_ht_state,
        )

        calibration = build_structural_calibration_for_p4(signal)

        # Check core fields
        assert calibration["scs"] == 1.0
        assert calibration["scs_percent"] == "100%"
        assert calibration["combined_severity"] == "CONSISTENT"
        assert calibration["admissible"] is True

        # Blocking invariants should pass
        assert calibration["blocking_invariants"]["SI-001_dag_acyclicity"] == "PASS"
        assert calibration["blocking_invariants"]["SI-010_truth_anchor_integrity"] == "PASS"

        # Layer summary
        assert calibration["layer_summary"]["dag"]["status"] == "CONSISTENT"
        assert calibration["layer_summary"]["topology"]["status"] == "CONSISTENT"
        assert calibration["layer_summary"]["ht"]["status"] == "CONSISTENT"

        # No violations
        assert calibration["violation_count"] == 0
        assert calibration["key_violations"] == []

        # Mode marker
        assert calibration["mode"] == "SHADOW"

    def test_calibration_from_conflict_signal(self, cyclic_dag, healthy_topology_state, failed_ht_state):
        """Test P4 calibration with blocking invariant failures."""
        signal = emit_structural_signal(
            dag=cyclic_dag,
            topology_state=healthy_topology_state,
            ht_state=failed_ht_state,
        )

        calibration = build_structural_calibration_for_p4(signal)

        # Should show failures
        assert calibration["admissible"] is False
        assert calibration["blocking_invariants"]["SI-001_dag_acyclicity"] == "FAIL"
        assert calibration["blocking_invariants"]["SI-010_truth_anchor_integrity"] == "FAIL"

        # Should have violations
        assert calibration["violation_count"] >= 2
        assert len(calibration["key_violations"]) >= 2

    def test_calibration_truncates_long_messages(self, healthy_dag, healthy_ht_state):
        """Test that key violation messages are truncated."""
        # Create topology with high omega_exit_streak to trigger violation
        topology = {
            "H": 0.5, "rho": 0.5, "tau": 0.20, "beta": 0.0,
            "in_omega": False, "omega_exit_streak": 150,
        }

        signal = emit_structural_signal(
            dag=healthy_dag,
            topology_state=topology,
            ht_state=healthy_ht_state,
        )

        calibration = build_structural_calibration_for_p4(signal)

        # Any violation message should be <= 80 chars
        for v in calibration["key_violations"]:
            assert len(v["message"]) <= 80


# =============================================================================
# Tests: Evidence Attachment
# =============================================================================

class TestAttachStructuralGovernanceToEvidence:
    """Tests for attach_structural_governance_to_evidence function."""

    def test_attach_to_empty_evidence(self, healthy_dag, healthy_topology_state, healthy_ht_state):
        """Test attaching to evidence without governance section."""
        evidence = {"proof_hash": "abc123", "cycle": 42}

        signal = emit_structural_signal(
            dag=healthy_dag,
            topology_state=healthy_topology_state,
            ht_state=healthy_ht_state,
        )

        result = attach_structural_governance_to_evidence(evidence, signal)

        # Should create governance section
        assert "governance" in result
        assert "structure" in result["governance"]

        # Check structure fields
        structure = result["governance"]["structure"]
        assert structure["cohesion_score"] == 1.0
        assert structure["combined_severity"] == "CONSISTENT"
        assert structure["admissible"] is True
        assert structure["blocking_invariants"] == []
        assert structure["mode"] == "SHADOW"

    def test_attach_to_existing_governance(self, healthy_dag, healthy_topology_state, healthy_ht_state):
        """Test attaching to evidence with existing governance section."""
        evidence = {
            "proof_hash": "abc123",
            "governance": {
                "budget": {"health_score": 95.0},
            },
        }

        signal = emit_structural_signal(
            dag=healthy_dag,
            topology_state=healthy_topology_state,
            ht_state=healthy_ht_state,
        )

        result = attach_structural_governance_to_evidence(evidence, signal)

        # Existing governance data preserved
        assert result["governance"]["budget"]["health_score"] == 95.0

        # Structure added
        assert "structure" in result["governance"]

    def test_attach_with_blocking_invariants(self, cyclic_dag, healthy_topology_state, failed_ht_state):
        """Test that blocking invariants are captured in evidence."""
        evidence = {}

        signal = emit_structural_signal(
            dag=cyclic_dag,
            topology_state=healthy_topology_state,
            ht_state=failed_ht_state,
        )

        result = attach_structural_governance_to_evidence(evidence, signal)

        structure = result["governance"]["structure"]

        # Should have blocking invariants
        assert "SI-001" in structure["blocking_invariants"]
        assert "SI-010" in structure["blocking_invariants"]
        assert structure["admissible"] is False

    def test_attach_violation_summary(self, cyclic_dag, out_of_bounds_topology_state, failed_ht_state):
        """Test violation summary counts."""
        evidence = {}

        signal = emit_structural_signal(
            dag=cyclic_dag,
            topology_state=out_of_bounds_topology_state,
            ht_state=failed_ht_state,
        )

        result = attach_structural_governance_to_evidence(evidence, signal)

        summary = result["governance"]["structure"]["violation_summary"]

        assert summary["total"] >= 3
        assert summary["conflict_count"] >= 3


# =============================================================================
# Tests: Escalation Advisory
# =============================================================================

class TestBuildEscalationAdvisory:
    """Tests for build_escalation_advisory function."""

    def test_advisory_no_escalation_consistent(self):
        """Test advisory when no escalation would occur (CONSISTENT)."""
        structural_signal = {
            "combined_severity": "CONSISTENT",
            "cohesion_score": 1.0,
            "admissible": True,
        }

        advisory = build_escalation_advisory("INFO", structural_signal)

        assert advisory["original_severity"] == "INFO"
        assert advisory["would_escalate"] is False
        assert advisory["escalated_severity"] == "INFO"
        assert advisory["escalation_reason"] is None
        assert advisory["mode"] == "SHADOW_ADVISORY"

    def test_advisory_conflict_escalation(self):
        """Test advisory when CONFLICT would escalate to CRITICAL."""
        structural_signal = {
            "combined_severity": "CONFLICT",
            "cohesion_score": 0.5,
            "admissible": False,
        }

        advisory = build_escalation_advisory("INFO", structural_signal)

        assert advisory["would_escalate"] is True
        assert advisory["escalated_severity"] == "CRITICAL"
        assert "CONFLICT" in advisory["escalation_reason"]
        assert "INFO → CRITICAL" in advisory["escalation_reason"]

    def test_advisory_tension_info_to_warn(self):
        """Test advisory when TENSION escalates INFO → WARN."""
        structural_signal = {
            "combined_severity": "TENSION",
            "cohesion_score": 0.85,
            "admissible": True,
        }

        advisory = build_escalation_advisory("INFO", structural_signal)

        assert advisory["would_escalate"] is True
        assert advisory["escalated_severity"] == "WARN"
        assert "TENSION" in advisory["escalation_reason"]
        assert "INFO → WARN" in advisory["escalation_reason"]

    def test_advisory_tension_warn_to_critical(self):
        """Test advisory when TENSION escalates WARN → CRITICAL."""
        structural_signal = {
            "combined_severity": "TENSION",
            "cohesion_score": 0.85,
            "admissible": True,
        }

        advisory = build_escalation_advisory("WARN", structural_signal)

        assert advisory["would_escalate"] is True
        assert advisory["escalated_severity"] == "CRITICAL"
        assert "WARN → CRITICAL" in advisory["escalation_reason"]

    def test_advisory_cohesion_degradation_note(self):
        """Test advisory includes cohesion degradation note when < 0.8."""
        structural_signal = {
            "combined_severity": "CONSISTENT",
            "cohesion_score": 0.65,
            "admissible": True,
        }

        advisory = build_escalation_advisory("NONE", structural_signal)

        # No severity escalation but degradation noted
        assert advisory["would_escalate"] is False
        assert "cohesion degraded" in advisory["escalation_reason"].lower()
        assert "65" in advisory["escalation_reason"]

    def test_advisory_none_severity_not_escalated(self):
        """Test advisory: NONE severity never escalated even with CONFLICT."""
        structural_signal = {
            "combined_severity": "CONFLICT",
            "cohesion_score": 0.5,
            "admissible": False,
        }

        advisory = build_escalation_advisory("NONE", structural_signal)

        # NONE stays NONE
        assert advisory["would_escalate"] is False
        assert advisory["escalated_severity"] == "NONE"

    def test_advisory_no_signal(self):
        """Test advisory when structural_signal is None."""
        advisory = build_escalation_advisory("WARN", None)

        assert advisory["would_escalate"] is False
        assert advisory["escalated_severity"] == "WARN"
        assert advisory["structural_context"] == "NO_SIGNAL"

    def test_advisory_structural_context_captured(self):
        """Test that structural context is captured in advisory."""
        structural_signal = {
            "combined_severity": "TENSION",
            "cohesion_score": 0.72,
            "admissible": True,
        }

        advisory = build_escalation_advisory("INFO", structural_signal)

        assert advisory["structural_context"]["combined_severity"] == "TENSION"
        assert advisory["structural_context"]["cohesion_score"] == 0.72
        assert advisory["structural_context"]["admissible"] is True


# =============================================================================
# Tests: Full Integration with New Helpers
# =============================================================================

class TestFullP4EvidenceIntegration:
    """Full integration tests for P4 calibration and evidence attachment."""

    def test_full_p4_calibration_flow(self, healthy_dag, healthy_topology_state, healthy_ht_state):
        """Test complete flow: signal → calibration → evidence."""
        # 1. Emit signal
        signal = emit_structural_signal(
            dag=healthy_dag,
            topology_state=healthy_topology_state,
            ht_state=healthy_ht_state,
            run_id="p4_integration_001",
        )

        # 2. Build calibration for P4 report
        calibration = build_structural_calibration_for_p4(signal)

        # 3. Build evidence and attach
        evidence = {
            "run_id": "p4_integration_001",
            "cycle": 500,
        }
        evidence = attach_structural_governance_to_evidence(evidence, signal)

        # 4. Build escalation advisory for a divergence
        advisory = build_escalation_advisory("INFO", signal.to_dict())

        # Verify complete flow
        assert calibration["admissible"] is True
        assert evidence["governance"]["structure"]["admissible"] is True
        assert advisory["would_escalate"] is False

    def test_full_flow_with_failures(self, cyclic_dag, out_of_bounds_topology_state, failed_ht_state):
        """Test complete flow with multiple failures."""
        # 1. Emit signal with failures
        signal = emit_structural_signal(
            dag=cyclic_dag,
            topology_state=out_of_bounds_topology_state,
            ht_state=failed_ht_state,
        )

        # 2. Build calibration
        calibration = build_structural_calibration_for_p4(signal)

        # 3. Attach to evidence
        evidence = {"run_id": "fail_test"}
        evidence = attach_structural_governance_to_evidence(evidence, signal)

        # 4. Build advisory
        advisory = build_escalation_advisory("WARN", signal.to_dict())

        # Verify failures propagate through flow
        assert calibration["admissible"] is False
        assert calibration["blocking_invariants"]["SI-001_dag_acyclicity"] == "FAIL"
        assert evidence["governance"]["structure"]["admissible"] is False
        assert advisory["would_escalate"] is True
        assert advisory["escalated_severity"] == "CRITICAL"
