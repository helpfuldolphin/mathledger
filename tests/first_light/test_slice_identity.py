"""
Phase X: Slice Identity Verification Tests

Tests for the slice identity verification module that implements
pre-execution blocking for P3/P4 shadow experiments.

SHADOW MODE CONTRACT:
- All tests verify ADVISORY behavior, not enforcement
- Tests ensure identity checks are deterministic
- No governance modification is tested

See: docs/system_law/SliceIdentity_PhaseX_Invariants.md

Status: PHASE X PRE-EXECUTION BLOCKER TESTS
"""

from __future__ import annotations

import json
import pytest
from typing import Any, Dict


class TestComputeSliceFingerprint:
    """Tests for compute_slice_fingerprint() - SI-001 implementation."""

    def test_fingerprint_is_64_hex_chars(self) -> None:
        """Verify fingerprint is a 64-character hex string (SHA-256)."""
        from backend.topology.first_light import compute_slice_fingerprint

        slice_config = {"params": {"depth_max": 5}}
        fp = compute_slice_fingerprint(slice_config)

        assert len(fp) == 64
        assert all(c in "0123456789abcdef" for c in fp)

    def test_fingerprint_deterministic(self) -> None:
        """Verify same config always produces same fingerprint."""
        from backend.topology.first_light import compute_slice_fingerprint

        slice_config = {
            "params": {"depth_max": 5, "atoms": 4},
            "gates": {"coverage": {"ci_lower_min": 0.8}},
        }

        fp1 = compute_slice_fingerprint(slice_config)
        fp2 = compute_slice_fingerprint(slice_config)
        fp3 = compute_slice_fingerprint(slice_config)

        assert fp1 == fp2 == fp3

    def test_fingerprint_key_order_invariant(self) -> None:
        """Verify fingerprint is independent of key order."""
        from backend.topology.first_light import compute_slice_fingerprint

        config1 = {"params": {"a": 1, "b": 2}}
        config2 = {"params": {"b": 2, "a": 1}}

        fp1 = compute_slice_fingerprint(config1)
        fp2 = compute_slice_fingerprint(config2)

        assert fp1 == fp2

    def test_fingerprint_different_for_different_configs(self) -> None:
        """Verify different configs produce different fingerprints."""
        from backend.topology.first_light import compute_slice_fingerprint

        config1 = {"params": {"depth_max": 5}}
        config2 = {"params": {"depth_max": 6}}

        fp1 = compute_slice_fingerprint(config1)
        fp2 = compute_slice_fingerprint(config2)

        assert fp1 != fp2

    def test_fingerprint_excludes_metadata(self) -> None:
        """Verify metadata fields are excluded from fingerprint."""
        from backend.topology.first_light import compute_slice_fingerprint

        config1 = {"params": {"depth_max": 5}}
        config2 = {
            "params": {"depth_max": 5},
            "name": "test_slice",
            "version": "1.0.0",
            "description": "A test slice",
            "created_at": "2025-01-01T00:00:00Z",
        }

        fp1 = compute_slice_fingerprint(config1)
        fp2 = compute_slice_fingerprint(config2)

        assert fp1 == fp2

    def test_fingerprint_empty_config(self) -> None:
        """Verify empty config produces valid fingerprint."""
        from backend.topology.first_light import compute_slice_fingerprint

        fp = compute_slice_fingerprint({})
        assert len(fp) == 64


class TestVerifySliceIdentityForP3:
    """Tests for verify_slice_identity_for_p3() - pre-flight verification."""

    def test_verification_returns_result_object(self) -> None:
        """Verify function returns SliceIdentityResult."""
        from backend.topology.first_light import (
            verify_slice_identity_for_p3,
            SliceIdentityResult,
        )

        result = verify_slice_identity_for_p3(
            slice_config={"params": {"depth_max": 5}},
        )

        assert isinstance(result, SliceIdentityResult)

    def test_verification_passes_for_valid_config(self) -> None:
        """Verify verification passes for valid config without baseline."""
        from backend.topology.first_light import verify_slice_identity_for_p3

        result = verify_slice_identity_for_p3(
            slice_config={"params": {"depth_max": 5}},
            slice_name="test_slice",
            curriculum_fingerprint="test_curriculum_fp",
        )

        assert result.identity_verified is True
        assert result.advisory_block is False
        # No violations when curriculum fingerprint provided
        assert len(result.violations) == 0

    def test_verification_passes_with_matching_baseline(self) -> None:
        """Verify verification passes when fingerprints match."""
        from backend.topology.first_light import (
            verify_slice_identity_for_p3,
            compute_slice_fingerprint,
        )

        slice_config = {"params": {"depth_max": 5}}
        baseline_fp = compute_slice_fingerprint(slice_config)

        result = verify_slice_identity_for_p3(
            slice_config=slice_config,
            baseline_fingerprint=baseline_fp,
        )

        assert result.identity_verified is True
        assert result.fingerprint_match is True
        assert result.advisory_block is False

    def test_verification_fails_with_mismatched_baseline(self) -> None:
        """Verify verification fails when fingerprints don't match."""
        from backend.topology.first_light import verify_slice_identity_for_p3

        slice_config = {"params": {"depth_max": 5}}
        wrong_baseline = "a" * 64  # Wrong fingerprint

        result = verify_slice_identity_for_p3(
            slice_config=slice_config,
            baseline_fingerprint=wrong_baseline,
        )

        assert result.identity_verified is False
        assert result.fingerprint_match is False
        assert result.advisory_block is True  # Critical invariant failed
        assert any("SI-005" in v for v in result.violations)

    def test_all_invariants_checked(self) -> None:
        """Verify all SI-001 through SI-006 invariants are checked."""
        from backend.topology.first_light import verify_slice_identity_for_p3

        result = verify_slice_identity_for_p3(
            slice_config={"params": {"depth_max": 5}},
            curriculum_fingerprint="curriculum_fp_123",
        )

        expected_invariants = ["SI-001", "SI-002", "SI-003", "SI-004", "SI-005", "SI-006"]
        for inv in expected_invariants:
            assert inv in result.invariant_status

    def test_result_to_dict_includes_schema(self) -> None:
        """Verify result serializes with proper schema."""
        from backend.topology.first_light import verify_slice_identity_for_p3

        result = verify_slice_identity_for_p3(
            slice_config={"params": {"depth_max": 5}},
        )

        d = result.to_dict()
        assert d["schema"] == "slice-identity-verification/1.0.0"
        assert d["mode"] == "SHADOW"

    def test_result_includes_computed_fingerprint(self) -> None:
        """Verify result includes computed fingerprint."""
        from backend.topology.first_light import verify_slice_identity_for_p3

        result = verify_slice_identity_for_p3(
            slice_config={"params": {"depth_max": 5}},
        )

        assert len(result.computed_fingerprint) == 64

    def test_slice_name_extracted_from_config(self) -> None:
        """Verify slice name is extracted from config if not provided."""
        from backend.topology.first_light import verify_slice_identity_for_p3

        result = verify_slice_identity_for_p3(
            slice_config={"name": "my_slice", "params": {}},
        )

        assert result.slice_name == "my_slice"


class TestSliceIdentityVerifier:
    """Tests for SliceIdentityVerifier - continuous monitoring."""

    def test_verifier_initialization(self) -> None:
        """Verify verifier initializes with baseline config."""
        from backend.topology.first_light import SliceIdentityVerifier

        baseline = {"params": {"depth_max": 5}}
        verifier = SliceIdentityVerifier(baseline)

        assert verifier.baseline_fingerprint is not None
        assert len(verifier.baseline_fingerprint) == 64

    def test_check_identity_stable_returns_true_for_same_config(self) -> None:
        """Verify stability check returns True when config unchanged."""
        from backend.topology.first_light import SliceIdentityVerifier

        baseline = {"params": {"depth_max": 5}}
        verifier = SliceIdentityVerifier(baseline)

        stable = verifier.check_identity_stable(baseline, cycle=1)
        assert stable is True

    def test_check_identity_stable_returns_false_for_changed_config(self) -> None:
        """Verify stability check returns False when config changed."""
        from backend.topology.first_light import SliceIdentityVerifier

        baseline = {"params": {"depth_max": 5}}
        changed = {"params": {"depth_max": 6}}
        verifier = SliceIdentityVerifier(baseline)

        stable = verifier.check_identity_stable(changed, cycle=1)
        assert stable is False

    def test_verifier_records_drift_events(self) -> None:
        """Verify drift events are recorded."""
        from backend.topology.first_light import SliceIdentityVerifier

        baseline = {"params": {"depth_max": 5}}
        changed = {"params": {"depth_max": 6}}
        verifier = SliceIdentityVerifier(baseline)

        verifier.check_identity_stable(changed, cycle=1)
        verifier.check_identity_stable(changed, cycle=2)

        events = verifier.get_drift_events()
        assert len(events) == 2
        assert events[0]["cycle"] == 1
        assert events[1]["cycle"] == 2

    def test_stability_score_perfect_when_no_drift(self) -> None:
        """Verify stability score is 1.0 when no drift occurs."""
        from backend.topology.first_light import SliceIdentityVerifier

        baseline = {"params": {"depth_max": 5}}
        verifier = SliceIdentityVerifier(baseline)

        for i in range(10):
            verifier.check_identity_stable(baseline, cycle=i)

        score = verifier.get_stability_score()
        assert score == 1.0

    def test_stability_score_degraded_with_drift(self) -> None:
        """Verify stability score < 1.0 when drift occurs."""
        from backend.topology.first_light import SliceIdentityVerifier

        baseline = {"params": {"depth_max": 5}}
        changed = {"params": {"depth_max": 6}}
        verifier = SliceIdentityVerifier(baseline)

        # 5 stable, 5 drifted
        for i in range(5):
            verifier.check_identity_stable(baseline, cycle=i)
        for i in range(5, 10):
            verifier.check_identity_stable(changed, cycle=i)

        score = verifier.get_stability_score()
        assert score == 0.5

    def test_consecutive_stable_cycles_tracked(self) -> None:
        """Verify consecutive stable cycles are tracked."""
        from backend.topology.first_light import SliceIdentityVerifier

        baseline = {"params": {"depth_max": 5}}
        changed = {"params": {"depth_max": 6}}
        verifier = SliceIdentityVerifier(baseline)

        # Check baseline 3 times
        for i in range(3):
            verifier.check_identity_stable(baseline, cycle=i)
        assert verifier._consecutive_stable_cycles == 3

        # Drift resets counter
        verifier.check_identity_stable(changed, cycle=3)
        assert verifier._consecutive_stable_cycles == 0

    def test_get_summary_returns_dict(self) -> None:
        """Verify get_summary returns proper dict."""
        from backend.topology.first_light import SliceIdentityVerifier

        baseline = {"params": {"depth_max": 5}}
        verifier = SliceIdentityVerifier(baseline, curriculum_fingerprint="curr_fp")

        summary = verifier.get_summary()
        assert "slice_name" in summary
        assert "baseline_fingerprint" in summary
        assert "curriculum_fingerprint" in summary
        assert "stability_score" in summary


class TestBuildIdentityConsoleTile:
    """Tests for build_identity_console_tile() - dashboard integration."""

    def test_tile_includes_required_fields(self) -> None:
        """Verify tile includes all schema-required fields."""
        from backend.topology.first_light import build_identity_console_tile

        tile = build_identity_console_tile()

        assert tile["schema_version"] == "1.0.0"
        assert tile["tile_type"] == "slice_identity"
        assert "timestamp" in tile
        assert "status" in tile
        assert "headline" in tile
        assert "identity_summary" in tile

    def test_tile_status_ok_by_default(self) -> None:
        """Verify tile status is OK when no issues."""
        from backend.topology.first_light import build_identity_console_tile

        tile = build_identity_console_tile()

        assert tile["status"] == "OK"

    def test_tile_status_error_on_verification_failure(self) -> None:
        """Verify tile status is ERROR when verification fails."""
        from backend.topology.first_light import (
            build_identity_console_tile,
            verify_slice_identity_for_p3,
        )

        # Create failed verification
        result = verify_slice_identity_for_p3(
            slice_config={"params": {"depth_max": 5}},
            baseline_fingerprint="wrong_fp_" + "x" * 55,
        )

        tile = build_identity_console_tile(verification_result=result)

        assert tile["status"] == "ERROR"
        assert len(tile["alerts"]) > 0

    def test_tile_includes_active_run_context(self) -> None:
        """Verify tile includes active run when provided."""
        from backend.topology.first_light import build_identity_console_tile

        tile = build_identity_console_tile(
            active_run_id="fl_test_123",
            active_run_type="P3",
            active_cycle=42,
        )

        assert tile["active_run"] is not None
        assert tile["active_run"]["run_id"] == "fl_test_123"
        assert tile["active_run"]["run_type"] == "P3"
        assert tile["active_run"]["cycle"] == 42

    def test_tile_from_verifier(self) -> None:
        """Verify tile can be built from verifier state."""
        from backend.topology.first_light import (
            build_identity_console_tile,
            SliceIdentityVerifier,
        )

        baseline = {"params": {"depth_max": 5}}
        verifier = SliceIdentityVerifier(baseline)

        # Run some cycles
        for i in range(10):
            verifier.check_identity_stable(baseline, cycle=i)

        tile = build_identity_console_tile(verifier=verifier)

        assert tile["status"] == "OK"
        assert tile["identity_summary"]["consecutive_stable_cycles"] == 10
        assert tile["trend"]["stability_score"] == 1.0

    def test_tile_warn_on_drift_detected(self) -> None:
        """Verify tile status is WARN when drift detected."""
        from backend.topology.first_light import (
            build_identity_console_tile,
            SliceIdentityVerifier,
        )

        baseline = {"params": {"depth_max": 5}}
        changed = {"params": {"depth_max": 6}}
        verifier = SliceIdentityVerifier(baseline)

        # 8 stable, 2 drifted = 80% stability
        for i in range(8):
            verifier.check_identity_stable(baseline, cycle=i)
        for i in range(8, 10):
            verifier.check_identity_stable(changed, cycle=i)

        tile = build_identity_console_tile(verifier=verifier)

        assert tile["status"] == "WARN"


class TestInvariantStatus:
    """Tests for InvariantStatus enum."""

    def test_invariant_status_values(self) -> None:
        """Verify InvariantStatus has expected values."""
        from backend.topology.first_light import InvariantStatus

        assert InvariantStatus.PASS.value == "PASS"
        assert InvariantStatus.FAIL.value == "FAIL"
        assert InvariantStatus.UNCHECKED.value == "UNCHECKED"


class TestDeterminism:
    """Tests for deterministic behavior across all identity functions."""

    def test_fingerprint_determinism_100_runs(self) -> None:
        """Verify fingerprint is deterministic across 100 runs."""
        from backend.topology.first_light import compute_slice_fingerprint

        config = {
            "params": {"depth_max": 5, "atoms": 4, "breadth_max": 100},
            "gates": {
                "coverage": {"ci_lower_min": 0.8},
                "abstention": {"max_rate_pct": 15},
            },
        }

        fingerprints = [compute_slice_fingerprint(config) for _ in range(100)]
        assert len(set(fingerprints)) == 1

    def test_verification_determinism(self) -> None:
        """Verify verification result is deterministic."""
        from backend.topology.first_light import verify_slice_identity_for_p3

        config = {"params": {"depth_max": 5}}
        baseline = "a" * 64

        results = [
            verify_slice_identity_for_p3(config, baseline_fingerprint=baseline)
            for _ in range(10)
        ]

        # All results should have same violations
        violations_list = [tuple(r.violations) for r in results]
        assert len(set(violations_list)) == 1


class TestCycleObservationIdentityField:
    """Tests for identity_stable field in CycleObservation."""

    def test_cycle_observation_has_identity_stable_field(self) -> None:
        """Verify CycleObservation has identity_stable field."""
        from backend.topology.first_light import CycleObservation

        obs = CycleObservation(
            cycle=1,
            timestamp="2025-01-01T00:00:00Z",
            success=True,
            depth=5,
            H=0.8,
            rho=0.9,
            tau=0.2,
            beta=0.1,
            in_omega=True,
            real_blocked=False,
            sim_blocked=False,
            governance_aligned=True,
            hard_ok=True,
            abstained=False,
        )

        assert hasattr(obs, "identity_stable")
        assert obs.identity_stable is True  # Default value

    def test_cycle_observation_to_dict_includes_identity_stable(self) -> None:
        """Verify to_dict() includes identity_stable field."""
        from backend.topology.first_light import CycleObservation

        obs = CycleObservation(
            cycle=1,
            timestamp="2025-01-01T00:00:00Z",
            success=True,
            depth=5,
            H=0.8,
            rho=0.9,
            tau=0.2,
            beta=0.1,
            in_omega=True,
            real_blocked=False,
            sim_blocked=False,
            governance_aligned=True,
            hard_ok=True,
            abstained=False,
            identity_stable=False,
        )

        d = obs.to_dict()
        assert "identity_stable" in d
        assert d["identity_stable"] is False


class TestFirstLightConfigIdentityFields:
    """Tests for identity fields in FirstLightConfig."""

    def test_config_has_identity_fields(self) -> None:
        """Verify FirstLightConfig has identity fields."""
        from backend.topology.first_light import FirstLightConfig

        config = FirstLightConfig()

        assert hasattr(config, "baseline_slice_fingerprint")
        assert hasattr(config, "identity_verified")
        assert hasattr(config, "curriculum_fingerprint")

    def test_config_identity_fields_default_values(self) -> None:
        """Verify identity fields have correct defaults."""
        from backend.topology.first_light import FirstLightConfig

        config = FirstLightConfig()

        assert config.baseline_slice_fingerprint is None
        assert config.identity_verified is False
        assert config.curriculum_fingerprint is None

    def test_config_identity_fields_settable(self) -> None:
        """Verify identity fields can be set."""
        from backend.topology.first_light import FirstLightConfig

        config = FirstLightConfig(
            baseline_slice_fingerprint="a" * 64,
            identity_verified=True,
            curriculum_fingerprint="b" * 64,
        )

        assert config.baseline_slice_fingerprint == "a" * 64
        assert config.identity_verified is True
        assert config.curriculum_fingerprint == "b" * 64


class TestTileSchemaCompliance:
    """Tests for console tile schema compliance."""

    def test_tile_conforms_to_schema_structure(self) -> None:
        """Verify tile output matches slice_identity_console_tile.schema.json."""
        from backend.topology.first_light import (
            build_identity_console_tile,
            verify_slice_identity_for_p3,
            SliceIdentityVerifier,
        )

        # Full tile with all contexts
        config = {"params": {"depth_max": 5}}
        result = verify_slice_identity_for_p3(config)
        verifier = SliceIdentityVerifier(config)
        verifier.check_identity_stable(config, cycle=1)

        tile = build_identity_console_tile(
            verification_result=result,
            verifier=verifier,
            active_run_id="test_run",
            active_run_type="P3",
            active_cycle=1,
        )

        # Check top-level required fields
        assert tile["schema_version"] == "1.0.0"
        assert tile["tile_type"] == "slice_identity"
        assert "timestamp" in tile
        assert tile["status"] in ["OK", "WARN", "ERROR", "UNKNOWN"]
        assert "headline" in tile
        assert "identity_summary" in tile

        # Check identity_summary required fields
        summary = tile["identity_summary"]
        assert "current_slice" in summary
        assert "fingerprint_stable" in summary
        assert "drift_events_24h" in summary

        # Check invariant_status
        inv_status = tile["invariant_status"]
        for inv_id in ["SI-001", "SI-002", "SI-003", "SI-004", "SI-005", "SI-006"]:
            assert inv_id in inv_status
            assert inv_status[inv_id] in ["PASS", "FAIL", "UNCHECKED"]

        # Check active_run
        assert tile["active_run"] is not None
        assert tile["active_run"]["run_id"] == "test_run"
        assert "evidence_admissibility" in tile["active_run"]

        # Check trend
        assert "trend" in tile
        assert "direction" in tile["trend"]
        assert "stability_score" in tile["trend"]


class TestAttachSliceIdentityToP3StabilityReport:
    """Tests for attach_slice_identity_to_p3_stability_report()."""

    def test_attaches_slice_identity_summary(self) -> None:
        """Verify slice_identity_summary is attached to report."""
        from backend.topology.first_light import (
            attach_slice_identity_to_p3_stability_report,
            build_identity_console_tile,
            verify_slice_identity_for_p3,
        )

        report = {"run_id": "test_run", "metrics": {"success_rate": 0.95}}
        config = {"params": {"depth_max": 5}}
        result = verify_slice_identity_for_p3(config)
        tile = build_identity_console_tile(verification_result=result)

        enriched = attach_slice_identity_to_p3_stability_report(report, tile)

        assert "slice_identity_summary" in enriched
        assert enriched["run_id"] == "test_run"  # Original preserved

    def test_non_mutating(self) -> None:
        """Verify original report is not mutated."""
        from backend.topology.first_light import (
            attach_slice_identity_to_p3_stability_report,
            build_identity_console_tile,
        )

        report = {"run_id": "test_run"}
        tile = build_identity_console_tile()

        enriched = attach_slice_identity_to_p3_stability_report(report, tile)

        assert "slice_identity_summary" not in report
        assert "slice_identity_summary" in enriched

    def test_includes_required_fields(self) -> None:
        """Verify slice_identity_summary includes required fields."""
        from backend.topology.first_light import (
            attach_slice_identity_to_p3_stability_report,
            build_identity_console_tile,
            verify_slice_identity_for_p3,
        )

        report = {}
        config = {"params": {"depth_max": 5}}
        result = verify_slice_identity_for_p3(config)
        tile = build_identity_console_tile(verification_result=result)

        enriched = attach_slice_identity_to_p3_stability_report(report, tile)

        summary = enriched["slice_identity_summary"]
        assert "identity_verified" in summary
        assert "fingerprint_match" in summary
        assert "violations" in summary
        assert "stability_score" in summary
        assert "invariant_status" in summary
        assert "slice_name" in summary

    def test_includes_evidence_admissibility_from_active_run(self) -> None:
        """Verify evidence_admissibility is included from active run."""
        from backend.topology.first_light import (
            attach_slice_identity_to_p3_stability_report,
            build_identity_console_tile,
        )

        report = {}
        tile = build_identity_console_tile(
            active_run_id="test",
            active_run_type="P3",
        )

        enriched = attach_slice_identity_to_p3_stability_report(report, tile)

        summary = enriched["slice_identity_summary"]
        assert "evidence_admissibility" in summary


class TestAttachSliceIdentityToEvidence:
    """Tests for attach_slice_identity_to_evidence()."""

    def test_attaches_to_governance_slice_identity(self) -> None:
        """Verify identity is attached under governance.slice_identity."""
        from backend.topology.first_light import (
            attach_slice_identity_to_evidence,
            build_identity_console_tile,
        )

        evidence = {"artifacts": [], "governance": {"cohesion": {}}}
        tile = build_identity_console_tile()

        enriched = attach_slice_identity_to_evidence(evidence, tile)

        assert "slice_identity" in enriched["governance"]

    def test_creates_governance_if_missing(self) -> None:
        """Verify governance key is created if missing."""
        from backend.topology.first_light import (
            attach_slice_identity_to_evidence,
            build_identity_console_tile,
        )

        evidence = {"artifacts": []}
        tile = build_identity_console_tile()

        enriched = attach_slice_identity_to_evidence(evidence, tile)

        assert "governance" in enriched
        assert "slice_identity" in enriched["governance"]

    def test_non_mutating(self) -> None:
        """Verify original evidence is not mutated."""
        from backend.topology.first_light import (
            attach_slice_identity_to_evidence,
            build_identity_console_tile,
        )

        evidence = {"governance": {}}
        tile = build_identity_console_tile()

        enriched = attach_slice_identity_to_evidence(evidence, tile)

        assert "slice_identity" not in evidence["governance"]
        assert "slice_identity" in enriched["governance"]

    def test_includes_invariant_statuses(self) -> None:
        """Verify invariant_statuses is included."""
        from backend.topology.first_light import (
            attach_slice_identity_to_evidence,
            build_identity_console_tile,
            verify_slice_identity_for_p3,
        )

        evidence = {}
        config = {"params": {"depth_max": 5}}
        result = verify_slice_identity_for_p3(config)
        tile = build_identity_console_tile(verification_result=result)

        enriched = attach_slice_identity_to_evidence(evidence, tile)

        slice_id = enriched["governance"]["slice_identity"]
        assert "invariant_statuses" in slice_id
        for inv in ["SI-001", "SI-002", "SI-003", "SI-004", "SI-005", "SI-006"]:
            assert inv in slice_id["invariant_statuses"]

    def test_evidence_admissibility_full_when_ok(self) -> None:
        """Verify evidence_admissibility is FULL when status is OK."""
        from backend.topology.first_light import (
            attach_slice_identity_to_evidence,
            build_identity_console_tile,
        )

        evidence = {}
        tile = build_identity_console_tile()  # Default is OK

        enriched = attach_slice_identity_to_evidence(evidence, tile)

        assert enriched["governance"]["slice_identity"]["evidence_admissibility"] == "FULL"

    def test_evidence_admissibility_inadmissible_on_critical_failure(self) -> None:
        """Verify evidence_admissibility is INADMISSIBLE on critical failure."""
        from backend.topology.first_light import (
            attach_slice_identity_to_evidence,
            build_identity_console_tile,
            verify_slice_identity_for_p3,
        )

        evidence = {}
        config = {"params": {"depth_max": 5}}
        result = verify_slice_identity_for_p3(
            config,
            baseline_fingerprint="wrong_" + "x" * 58,
        )
        tile = build_identity_console_tile(verification_result=result)

        enriched = attach_slice_identity_to_evidence(evidence, tile)

        assert enriched["governance"]["slice_identity"]["evidence_admissibility"] == "INADMISSIBLE"


class TestComputeP4IdentityDriftContext:
    """Tests for compute_p4_identity_drift_context()."""

    def test_returns_p4_identity_drift_context(self) -> None:
        """Verify function returns P4IdentityDriftContext."""
        from backend.topology.first_light import (
            compute_p4_identity_drift_context,
            verify_slice_identity_for_p3,
            DivergenceSnapshot,
            P4IdentityDriftContext,
        )

        config = {"params": {"depth_max": 5}}
        result = verify_slice_identity_for_p3(config)
        div = DivergenceSnapshot(cycle=1)

        ctx = compute_p4_identity_drift_context(div, result)

        assert isinstance(ctx, P4IdentityDriftContext)

    def test_no_divergence_when_identity_verified(self) -> None:
        """Verify no divergence when identity is verified."""
        from backend.topology.first_light import (
            compute_p4_identity_drift_context,
            verify_slice_identity_for_p3,
            DivergenceSnapshot,
        )

        config = {"params": {"depth_max": 5}}
        result = verify_slice_identity_for_p3(config)
        div = DivergenceSnapshot(cycle=1)

        ctx = compute_p4_identity_drift_context(div, result)

        assert ctx.identity_diverged is False
        assert ctx.identity_divergence_type is None

    def test_fingerprint_mismatch_detected(self) -> None:
        """Verify fingerprint mismatch is detected."""
        from backend.topology.first_light import (
            compute_p4_identity_drift_context,
            verify_slice_identity_for_p3,
            DivergenceSnapshot,
        )

        config = {"params": {"depth_max": 5}}
        result = verify_slice_identity_for_p3(
            config,
            baseline_fingerprint="wrong_" + "x" * 58,
        )
        div = DivergenceSnapshot(cycle=1)

        ctx = compute_p4_identity_drift_context(div, result)

        assert ctx.identity_diverged is True
        assert ctx.identity_divergence_type == "FINGERPRINT_MISMATCH"

    def test_to_dict_includes_mode_shadow(self) -> None:
        """Verify to_dict() includes mode=SHADOW."""
        from backend.topology.first_light import (
            compute_p4_identity_drift_context,
            verify_slice_identity_for_p3,
            DivergenceSnapshot,
        )

        config = {"params": {"depth_max": 5}}
        result = verify_slice_identity_for_p3(config)
        div = DivergenceSnapshot(cycle=1)

        ctx = compute_p4_identity_drift_context(div, result)
        d = ctx.to_dict()

        assert d["mode"] == "SHADOW"

    def test_structural_conflict_detected(self) -> None:
        """Verify structural conflict triggers identity divergence."""
        from backend.topology.first_light import (
            compute_p4_identity_drift_context,
            verify_slice_identity_for_p3,
            DivergenceSnapshot,
        )

        config = {"params": {"depth_max": 5}}
        result = verify_slice_identity_for_p3(config)
        div = DivergenceSnapshot(cycle=1, structural_conflict=True)

        ctx = compute_p4_identity_drift_context(div, result)

        assert ctx.identity_diverged is True
        assert ctx.identity_divergence_type == "STRUCTURAL_CONFLICT"


class TestP4IdentityDriftContext:
    """Tests for P4IdentityDriftContext dataclass."""

    def test_default_values(self) -> None:
        """Verify default values."""
        from backend.topology.first_light import P4IdentityDriftContext

        ctx = P4IdentityDriftContext()

        assert ctx.identity_diverged is False
        assert ctx.identity_divergence_type is None
        assert ctx.invariant_violations == []

    def test_to_dict(self) -> None:
        """Verify to_dict() output."""
        from backend.topology.first_light import P4IdentityDriftContext

        ctx = P4IdentityDriftContext(
            identity_diverged=True,
            identity_divergence_type="FINGERPRINT_MISMATCH",
            slice_name="test_slice",
        )

        d = ctx.to_dict()

        assert d["identity_diverged"] is True
        assert d["identity_divergence_type"] == "FINGERPRINT_MISMATCH"
        assert d["slice_name"] == "test_slice"
        assert d["mode"] == "SHADOW"
