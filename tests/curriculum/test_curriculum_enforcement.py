"""
Tests for curriculum enforcement module.

Tests cover:
- P3 pre-execution verification
- Drift detection and classification
- Governance signal generation
- Determinism guarantees
"""

import json
import pytest
from copy import deepcopy
from typing import Any, Dict

from curriculum.enforcement import (
    DriftSeverity,
    DriftStatus,
    GovernanceSignalType,
    Violation,
    MonotonicityViolation,
    GateEvolutionViolation,
    ChangedParam,
    CurriculumSnapshot,
    P3VerificationResult,
    DriftTimelineEvent,
    GovernanceSignal,
    verify_curriculum_for_p3,
    DriftTimelineGenerator,
    GovernanceSignalBuilder,
    CurriculumRuntimeGuard,
    GATE_EVOLUTION_RULES,
)


# -----------------------------------------------------------------------------
# Test Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def valid_curriculum_config() -> Dict[str, Any]:
    """Return a valid curriculum configuration."""
    return {
        "version": 2,
        "systems": {
            "propositional": {
                "description": "Propositional logic curriculum",
                "active": "slice_1",
                "invariants": {
                    "monotonic_axes": ["atoms", "depth_max"],
                },
                "slices": [
                    {
                        "name": "slice_1",
                        "params": {
                            "atoms": 4,
                            "depth_max": 5,
                            "breadth_max": 1200,
                            "total_max": 6000,
                        },
                        "gates": {
                            "coverage": {
                                "ci_lower_min": 0.90,
                                "sample_min": 20,
                                "require_attestation": True,
                            },
                            "abstention": {
                                "max_rate_pct": 20.0,
                                "max_mass": 1000,
                            },
                            "velocity": {
                                "min_pph": 100.0,
                                "stability_cv_max": 0.15,
                                "window_minutes": 60,
                            },
                            "caps": {
                                "min_attempt_mass": 2000,
                                "min_runtime_minutes": 20.0,
                                "backlog_max": 0.40,
                            },
                        },
                    },
                    {
                        "name": "slice_2",
                        "params": {
                            "atoms": 5,
                            "depth_max": 6,
                            "breadth_max": 1500,
                            "total_max": 8000,
                        },
                        "gates": {
                            "coverage": {
                                "ci_lower_min": 0.92,
                                "sample_min": 25,
                                "require_attestation": True,
                            },
                            "abstention": {
                                "max_rate_pct": 18.0,
                                "max_mass": 900,
                            },
                            "velocity": {
                                "min_pph": 120.0,
                                "stability_cv_max": 0.12,
                                "window_minutes": 60,
                            },
                            "caps": {
                                "min_attempt_mass": 2500,
                                "min_runtime_minutes": 25.0,
                                "backlog_max": 0.35,
                            },
                        },
                    },
                ],
            },
        },
    }


@pytest.fixture
def invalid_curriculum_config() -> Dict[str, Any]:
    """Return an invalid curriculum configuration (version wrong)."""
    return {
        "version": 1,  # Wrong version
        "systems": {},
    }


@pytest.fixture
def monotonicity_violating_config() -> Dict[str, Any]:
    """Return a config with monotonicity violation."""
    return {
        "version": 2,
        "systems": {
            "propositional": {
                "description": "Test system",
                "active": "slice_1",
                "invariants": {
                    "monotonic_axes": ["atoms", "depth_max"],
                },
                "slices": [
                    {
                        "name": "slice_1",
                        "params": {
                            "atoms": 5,  # Higher first
                            "depth_max": 6,
                            "breadth_max": 1200,
                            "total_max": 6000,
                        },
                        "gates": {
                            "coverage": {"ci_lower_min": 0.90, "sample_min": 20, "require_attestation": True},
                            "abstention": {"max_rate_pct": 20.0, "max_mass": 1000},
                            "velocity": {"min_pph": 100.0, "stability_cv_max": 0.15, "window_minutes": 60},
                            "caps": {"min_attempt_mass": 2000, "min_runtime_minutes": 20.0, "backlog_max": 0.40},
                        },
                    },
                    {
                        "name": "slice_2",
                        "params": {
                            "atoms": 4,  # Lower second - VIOLATION
                            "depth_max": 5,  # Also lower - VIOLATION
                            "breadth_max": 1500,
                            "total_max": 8000,
                        },
                        "gates": {
                            "coverage": {"ci_lower_min": 0.92, "sample_min": 25, "require_attestation": True},
                            "abstention": {"max_rate_pct": 18.0, "max_mass": 900},
                            "velocity": {"min_pph": 120.0, "stability_cv_max": 0.12, "window_minutes": 60},
                            "caps": {"min_attempt_mass": 2500, "min_runtime_minutes": 25.0, "backlog_max": 0.35},
                        },
                    },
                ],
            },
        },
    }


# -----------------------------------------------------------------------------
# P3 Pre-Execution Verification Tests
# -----------------------------------------------------------------------------

class TestP3Verification:
    """Tests for P3 pre-execution verification."""

    def test_valid_config_passes(self, valid_curriculum_config: Dict[str, Any]) -> None:
        """Valid config should pass verification."""
        result = verify_curriculum_for_p3(valid_curriculum_config, "propositional")

        assert result["valid"] is True
        assert result["phase"] == "P3"
        assert result["mode"] == "SHADOW"
        assert len(result["violations"]) == 0
        assert result["active_slice"] == "slice_1"
        assert result["curriculum_fingerprint"] != ""

    def test_invalid_version_fails(self, invalid_curriculum_config: Dict[str, Any]) -> None:
        """Invalid version should produce violation."""
        result = verify_curriculum_for_p3(invalid_curriculum_config, "propositional")

        assert result["valid"] is False
        violation_codes = [v["code"] for v in result["violations"]]
        assert "CUR-P3-VERSION" in violation_codes

    def test_missing_system_fails(self, valid_curriculum_config: Dict[str, Any]) -> None:
        """Missing system should produce violation."""
        result = verify_curriculum_for_p3(valid_curriculum_config, "nonexistent")

        assert result["valid"] is False
        violation_codes = [v["code"] for v in result["violations"]]
        assert "CUR-P3-SYSTEM" in violation_codes

    def test_monotonicity_violation_detected(
        self, monotonicity_violating_config: Dict[str, Any]
    ) -> None:
        """Monotonicity violations should be detected."""
        result = verify_curriculum_for_p3(monotonicity_violating_config, "propositional")

        assert result["valid"] is False
        violation_codes = [v["code"] for v in result["violations"]]
        assert "CUR-P3-MONO-VIOLATION" in violation_codes

    def test_missing_slices_fails(self, valid_curriculum_config: Dict[str, Any]) -> None:
        """Config with no slices should fail."""
        config = deepcopy(valid_curriculum_config)
        config["systems"]["propositional"]["slices"] = []

        result = verify_curriculum_for_p3(config, "propositional")

        assert result["valid"] is False
        violation_codes = [v["code"] for v in result["violations"]]
        assert "CUR-P3-SLICES" in violation_codes

    def test_missing_gate_detected(self, valid_curriculum_config: Dict[str, Any]) -> None:
        """Missing gate should be detected."""
        config = deepcopy(valid_curriculum_config)
        del config["systems"]["propositional"]["slices"][0]["gates"]["coverage"]

        result = verify_curriculum_for_p3(config, "propositional")

        assert result["valid"] is False
        violation_codes = [v["code"] for v in result["violations"]]
        assert "CUR-P3-GATE-MISSING" in violation_codes

    def test_fingerprint_is_deterministic(self, valid_curriculum_config: Dict[str, Any]) -> None:
        """Fingerprint should be deterministic across calls."""
        result1 = verify_curriculum_for_p3(valid_curriculum_config, "propositional")
        result2 = verify_curriculum_for_p3(valid_curriculum_config, "propositional")

        assert result1["curriculum_fingerprint"] == result2["curriculum_fingerprint"]


# -----------------------------------------------------------------------------
# Drift Detection Tests
# -----------------------------------------------------------------------------

class TestDriftDetection:
    """Tests for drift detection."""

    def test_no_drift_detected_when_unchanged(
        self, valid_curriculum_config: Dict[str, Any]
    ) -> None:
        """No drift should be detected when config is unchanged."""
        generator = DriftTimelineGenerator(phase="P4")
        generator.capture_baseline(valid_curriculum_config, "propositional")

        event = generator.generate_drift_event(valid_curriculum_config, "propositional")

        assert event.drift_severity == DriftSeverity.NONE
        assert event.drift_status == DriftStatus.OK
        assert len(event.changed_params) == 0
        assert len(event.monotonicity_violations) == 0
        assert len(event.gate_evolution_violations) == 0

    def test_parametric_drift_detected_on_allowed_change(
        self, valid_curriculum_config: Dict[str, Any]
    ) -> None:
        """Parametric drift should be detected for allowed changes."""
        generator = DriftTimelineGenerator(phase="P4")
        generator.capture_baseline(valid_curriculum_config, "propositional")

        # Modify config with allowed change (increase atoms)
        modified = deepcopy(valid_curriculum_config)
        modified["systems"]["propositional"]["slices"][0]["params"]["atoms"] = 5

        event = generator.generate_drift_event(modified, "propositional")

        assert event.drift_severity == DriftSeverity.PARAMETRIC
        assert event.drift_status == DriftStatus.WARN
        assert len(event.changed_params) > 0

        param_paths = [p.path for p in event.changed_params]
        assert "params.atoms" in param_paths

    def test_semantic_drift_detected_on_regression(
        self, valid_curriculum_config: Dict[str, Any]
    ) -> None:
        """Semantic drift should be detected for forbidden changes."""
        generator = DriftTimelineGenerator(phase="P4")
        generator.capture_baseline(valid_curriculum_config, "propositional")

        # Modify config with forbidden change (decrease atoms)
        modified = deepcopy(valid_curriculum_config)
        modified["systems"]["propositional"]["slices"][0]["params"]["atoms"] = 3

        event = generator.generate_drift_event(modified, "propositional")

        assert event.drift_severity == DriftSeverity.SEMANTIC
        assert event.drift_status == DriftStatus.BLOCK
        assert len(event.monotonicity_violations) > 0

    def test_gate_relaxation_detected(
        self, valid_curriculum_config: Dict[str, Any]
    ) -> None:
        """Gate relaxation should be detected as semantic violation."""
        generator = DriftTimelineGenerator(phase="P4")
        generator.capture_baseline(valid_curriculum_config, "propositional")

        # Relax abstention rate (forbidden - max_rate_pct should only decrease)
        modified = deepcopy(valid_curriculum_config)
        modified["systems"]["propositional"]["slices"][0]["gates"]["abstention"]["max_rate_pct"] = 25.0

        event = generator.generate_drift_event(modified, "propositional")

        assert event.drift_severity == DriftSeverity.SEMANTIC
        assert len(event.gate_evolution_violations) > 0
        assert event.gate_evolution_violations[0].gate == "abstention"

    def test_gate_tightening_allowed(
        self, valid_curriculum_config: Dict[str, Any]
    ) -> None:
        """Gate tightening should be allowed (parametric)."""
        generator = DriftTimelineGenerator(phase="P4")
        generator.capture_baseline(valid_curriculum_config, "propositional")

        # Tighten coverage requirement (allowed)
        modified = deepcopy(valid_curriculum_config)
        modified["systems"]["propositional"]["slices"][0]["gates"]["coverage"]["ci_lower_min"] = 0.95

        event = generator.generate_drift_event(modified, "propositional")

        assert event.drift_severity == DriftSeverity.PARAMETRIC
        assert len(event.gate_evolution_violations) == 0

    def test_event_id_is_deterministic(
        self, valid_curriculum_config: Dict[str, Any]
    ) -> None:
        """Event ID should be deterministic."""
        generator1 = DriftTimelineGenerator(phase="P4")
        generator1.capture_baseline(valid_curriculum_config, "propositional")
        event1 = generator1.generate_drift_event(valid_curriculum_config, "propositional", "run1", 1)

        generator2 = DriftTimelineGenerator(phase="P4")
        generator2.capture_baseline(valid_curriculum_config, "propositional")
        event2 = generator2.generate_drift_event(valid_curriculum_config, "propositional", "run1", 1)

        assert event1.event_id == event2.event_id

    def test_baseline_required(self, valid_curriculum_config: Dict[str, Any]) -> None:
        """Generate should fail without baseline."""
        generator = DriftTimelineGenerator(phase="P4")

        with pytest.raises(ValueError, match="No baseline captured"):
            generator.generate_drift_event(valid_curriculum_config, "propositional")


# -----------------------------------------------------------------------------
# Governance Signal Tests
# -----------------------------------------------------------------------------

class TestGovernanceSignals:
    """Tests for governance signal generation."""

    def test_signal_from_drift_event(self, valid_curriculum_config: Dict[str, Any]) -> None:
        """Signal should be generated from drift event."""
        generator = DriftTimelineGenerator(phase="P4")
        generator.capture_baseline(valid_curriculum_config, "propositional")

        # Create drift
        modified = deepcopy(valid_curriculum_config)
        modified["systems"]["propositional"]["slices"][0]["params"]["atoms"] = 3

        event = generator.generate_drift_event(modified, "propositional", "run1", 42)

        builder = GovernanceSignalBuilder(phase="P4")
        signal = builder.from_drift_event(event, "run1", 42)

        assert signal.phase == "P4"
        assert signal.mode == "SHADOW"
        assert signal.signal_type == GovernanceSignalType.INVARIANT_VIOLATION
        assert signal.severity == DriftSeverity.SEMANTIC
        assert signal.status == DriftStatus.BLOCK
        assert signal.governance_action == "LOGGED_ONLY"
        assert signal.context is not None
        assert signal.context["run_id"] == "run1"
        assert signal.context["cycle"] == 42
        assert signal.hypothetical is not None
        assert signal.hypothetical["would_allow_transition"] is False

    def test_signal_from_verification_result(
        self, valid_curriculum_config: Dict[str, Any]
    ) -> None:
        """Signal should be generated from verification result."""
        result = verify_curriculum_for_p3(valid_curriculum_config, "propositional")

        builder = GovernanceSignalBuilder(phase="P3")
        signal = builder.from_verification_result(result)

        assert signal.phase == "P3"
        assert signal.mode == "SHADOW"
        assert signal.signal_type == GovernanceSignalType.SNAPSHOT_CAPTURED
        assert signal.severity == DriftSeverity.NONE
        assert signal.status == DriftStatus.OK
        assert signal.governance_action == "LOGGED_ONLY"

    def test_transition_requested_signal(
        self, valid_curriculum_config: Dict[str, Any]
    ) -> None:
        """Transition requested signal should be created."""
        builder = GovernanceSignalBuilder(phase="P4")
        signal = builder.transition_requested(
            fingerprint="abc123",
            from_slice="slice_1",
            to_slice="slice_2",
            run_id="run1",
        )

        assert signal.signal_type == GovernanceSignalType.TRANSITION_REQUESTED
        assert signal.active_slice == "slice_1"
        assert signal.target_slice == "slice_2"
        assert signal.governance_action == "LOGGED_ONLY"

    def test_signal_to_jsonl(self, valid_curriculum_config: Dict[str, Any]) -> None:
        """Signal should serialize to valid JSONL."""
        result = verify_curriculum_for_p3(valid_curriculum_config, "propositional")
        builder = GovernanceSignalBuilder(phase="P3")
        signal = builder.from_verification_result(result)

        jsonl = signal.to_jsonl()

        # Should be valid JSON
        parsed = json.loads(jsonl)
        assert parsed["schema"] == "curriculum-governance-signal/1.0.0"
        assert parsed["mode"] == "SHADOW"


# -----------------------------------------------------------------------------
# Runtime Guard Tests
# -----------------------------------------------------------------------------

class TestRuntimeGuard:
    """Tests for curriculum runtime guard."""

    def test_snapshot_captured(self, valid_curriculum_config: Dict[str, Any]) -> None:
        """Guard should capture snapshot."""
        guard = CurriculumRuntimeGuard(phase="P3")
        snapshot = guard.capture_snapshot(valid_curriculum_config, "propositional")

        assert snapshot is not None
        assert snapshot.fingerprint != ""
        assert snapshot.active_slice_name == "slice_1"
        assert len(guard.get_signals()) == 1
        assert guard.get_signals()[0].signal_type == GovernanceSignalType.SNAPSHOT_CAPTURED

    def test_verify_unchanged_succeeds(
        self, valid_curriculum_config: Dict[str, Any]
    ) -> None:
        """Verify unchanged should succeed for identical config."""
        guard = CurriculumRuntimeGuard(phase="P3")
        guard.capture_snapshot(valid_curriculum_config, "propositional")

        unchanged, event, signal = guard.verify_unchanged(
            valid_curriculum_config, "propositional", "run1", 1
        )

        assert unchanged is True
        assert event is not None
        assert event.drift_severity == DriftSeverity.NONE
        assert signal is not None
        assert signal.signal_type == GovernanceSignalType.SNAPSHOT_VERIFIED

    def test_verify_unchanged_detects_drift(
        self, valid_curriculum_config: Dict[str, Any]
    ) -> None:
        """Verify unchanged should detect drift."""
        guard = CurriculumRuntimeGuard(phase="P3")
        guard.capture_snapshot(valid_curriculum_config, "propositional")

        # Modify config
        modified = deepcopy(valid_curriculum_config)
        modified["systems"]["propositional"]["slices"][0]["params"]["atoms"] = 3

        unchanged, event, signal = guard.verify_unchanged(modified, "propositional", "run1", 1)

        assert unchanged is False
        assert event is not None
        assert event.drift_severity == DriftSeverity.SEMANTIC
        assert signal is not None
        assert signal.signal_type == GovernanceSignalType.INVARIANT_VIOLATION

    def test_export_jsonl(self, valid_curriculum_config: Dict[str, Any]) -> None:
        """Guard should export events and signals as JSONL."""
        guard = CurriculumRuntimeGuard(phase="P4")
        guard.capture_snapshot(valid_curriculum_config, "propositional")
        guard.verify_unchanged(valid_curriculum_config, "propositional")

        timeline_jsonl = guard.export_timeline_jsonl()
        signals_jsonl = guard.export_signals_jsonl()

        # Should have at least one event and signal
        assert len(timeline_jsonl) > 0
        assert len(signals_jsonl) > 0

        # Each line should be valid JSON
        for line in signals_jsonl.strip().split("\n"):
            if line:
                parsed = json.loads(line)
                assert "schema" in parsed


# -----------------------------------------------------------------------------
# Determinism Tests
# -----------------------------------------------------------------------------

class TestDeterminism:
    """Tests for determinism guarantees."""

    def test_fingerprint_deterministic(self, valid_curriculum_config: Dict[str, Any]) -> None:
        """Fingerprint should be deterministic."""
        snap1 = CurriculumSnapshot.from_config(valid_curriculum_config, "propositional")
        snap2 = CurriculumSnapshot.from_config(valid_curriculum_config, "propositional")

        assert snap1.fingerprint == snap2.fingerprint

    def test_event_timestamp_deterministic(
        self, valid_curriculum_config: Dict[str, Any]
    ) -> None:
        """Event timestamp should be deterministic."""
        generator1 = DriftTimelineGenerator(phase="P4")
        generator1.capture_baseline(valid_curriculum_config, "propositional")
        event1 = generator1.generate_drift_event(valid_curriculum_config, "propositional", "run1", 1)

        generator2 = DriftTimelineGenerator(phase="P4")
        generator2.capture_baseline(valid_curriculum_config, "propositional")
        event2 = generator2.generate_drift_event(valid_curriculum_config, "propositional", "run1", 1)

        assert event1.timestamp == event2.timestamp

    def test_signal_id_deterministic(self, valid_curriculum_config: Dict[str, Any]) -> None:
        """Signal ID should be deterministic."""
        result = verify_curriculum_for_p3(valid_curriculum_config, "propositional")

        builder1 = GovernanceSignalBuilder(phase="P3")
        signal1 = builder1.from_verification_result(result)

        builder2 = GovernanceSignalBuilder(phase="P3")
        signal2 = builder2.from_verification_result(result)

        assert signal1.signal_id == signal2.signal_id

    def test_jsonl_output_deterministic(
        self, valid_curriculum_config: Dict[str, Any]
    ) -> None:
        """JSONL output should be deterministic."""
        guard1 = CurriculumRuntimeGuard(phase="P4")
        guard1.capture_snapshot(valid_curriculum_config, "propositional")
        guard1.verify_unchanged(valid_curriculum_config, "propositional", "run1", 1)
        output1 = guard1.export_timeline_jsonl()

        guard2 = CurriculumRuntimeGuard(phase="P4")
        guard2.capture_snapshot(valid_curriculum_config, "propositional")
        guard2.verify_unchanged(valid_curriculum_config, "propositional", "run1", 1)
        output2 = guard2.export_timeline_jsonl()

        assert output1 == output2

    def test_multiple_runs_produce_same_output(
        self, valid_curriculum_config: Dict[str, Any]
    ) -> None:
        """Multiple runs should produce identical output."""
        outputs = []
        for _ in range(5):
            guard = CurriculumRuntimeGuard(phase="P4")
            guard.capture_snapshot(valid_curriculum_config, "propositional")
            guard.verify_unchanged(valid_curriculum_config, "propositional", "run1", 1)
            outputs.append(guard.export_signals_jsonl())

        assert all(o == outputs[0] for o in outputs)


# -----------------------------------------------------------------------------
# Schema Compliance Tests
# -----------------------------------------------------------------------------

class TestSchemaCompliance:
    """Tests for schema compliance."""

    def test_drift_event_schema_fields(
        self, valid_curriculum_config: Dict[str, Any]
    ) -> None:
        """Drift event should have all required schema fields."""
        generator = DriftTimelineGenerator(phase="P4")
        generator.capture_baseline(valid_curriculum_config, "propositional")
        event = generator.generate_drift_event(valid_curriculum_config, "propositional")

        data = event.to_dict()

        # Required fields per schema
        assert data["schema"] == "curriculum-drift-timeline/1.0.0"
        assert "event_id" in data
        assert "timestamp" in data
        assert "phase" in data
        assert data["mode"] == "SHADOW"
        assert "curriculum_fingerprint" in data
        assert "slice_name" in data
        assert "drift_status" in data
        assert "drift_severity" in data
        assert data["action_taken"] == "LOGGED_ONLY"

    def test_governance_signal_schema_fields(
        self, valid_curriculum_config: Dict[str, Any]
    ) -> None:
        """Governance signal should have all required schema fields."""
        result = verify_curriculum_for_p3(valid_curriculum_config, "propositional")
        builder = GovernanceSignalBuilder(phase="P3")
        signal = builder.from_verification_result(result)

        data = signal.to_dict()

        # Required fields per schema
        assert data["schema"] == "curriculum-governance-signal/1.0.0"
        assert "signal_id" in data
        assert "timestamp" in data
        assert "phase" in data
        assert data["mode"] == "SHADOW"
        assert "signal_type" in data
        assert "curriculum_fingerprint" in data
        assert "active_slice" in data
        assert "severity" in data
        assert "status" in data
        assert data["governance_action"] == "LOGGED_ONLY"


# -----------------------------------------------------------------------------
# Edge Cases
# -----------------------------------------------------------------------------

class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_slices_list(self) -> None:
        """Empty slices list should be handled."""
        config = {
            "version": 2,
            "systems": {
                "test": {
                    "description": "Test",
                    "slices": [],
                },
            },
        }

        result = verify_curriculum_for_p3(config, "test")
        assert result["valid"] is False

    def test_missing_gates_section(self) -> None:
        """Missing gates section should be detected."""
        config = {
            "version": 2,
            "systems": {
                "test": {
                    "description": "Test",
                    "active": "slice_1",
                    "slices": [
                        {
                            "name": "slice_1",
                            "params": {"atoms": 4},
                            # Missing "gates"
                        },
                    ],
                },
            },
        }

        result = verify_curriculum_for_p3(config, "test")
        assert result["valid"] is False
        violation_codes = [v["code"] for v in result["violations"]]
        assert "CUR-P3-SLICE-GATES" in violation_codes

    def test_none_values_in_params(self, valid_curriculum_config: Dict[str, Any]) -> None:
        """None values in params should be handled."""
        config = deepcopy(valid_curriculum_config)
        config["systems"]["propositional"]["slices"][0]["params"]["atoms"] = None

        result = verify_curriculum_for_p3(config, "propositional")
        # Should not crash, may produce warning or violation
        assert "valid" in result

    def test_boolean_gate_disable_detection(
        self, valid_curriculum_config: Dict[str, Any]
    ) -> None:
        """Disabling boolean gate should be detected."""
        generator = DriftTimelineGenerator(phase="P4")
        generator.capture_baseline(valid_curriculum_config, "propositional")

        # Disable attestation requirement (forbidden)
        modified = deepcopy(valid_curriculum_config)
        modified["systems"]["propositional"]["slices"][0]["gates"]["coverage"]["require_attestation"] = False

        event = generator.generate_drift_event(modified, "propositional")

        assert event.drift_severity == DriftSeverity.SEMANTIC
        assert len(event.gate_evolution_violations) > 0
        assert event.gate_evolution_violations[0].violation_type == "DISABLE"
