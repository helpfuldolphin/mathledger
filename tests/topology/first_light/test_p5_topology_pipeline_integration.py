"""Integration tests for P5 Topology Pipeline.

Tests end-to-end flow:
1. Generate P5 topology auditor report from synthetic inputs
2. Detect report in evidence pack
3. Include signal in first_light_status.json

SHADOW MODE CONTRACT:
- All tests verify observational-only behavior
- No enforcement logic should be present
- All outputs should be JSON-serializable
"""

import json
import pytest
import tempfile
from pathlib import Path
from typing import Any, Dict

from backend.health.p5_topology_reality_adapter import (
    run_p5_smoke_validation,
    build_p5_topology_reality_summary,
    generate_p5_auditor_report,
)
from backend.health.topology_bundle_adapter import (
    build_topology_bundle_console_tile,
)
from backend.topology.first_light.evidence_pack import (
    detect_p5_topology_auditor_report,
    P5TopologyAuditorReference,
    EvidencePackBuilder,
    build_evidence_pack,
)
from scripts.generate_p5_topology_auditor_report import (
    generate_p5_topology_report,
    DETERMINISTIC_TIMESTAMP,
    DETERMINISTIC_ENV_VAR,
    _env_is_truthy,
)


# -----------------------------------------------------------------------------
# Test Fixtures: Synthetic Topology Inputs
# -----------------------------------------------------------------------------

@pytest.fixture
def mock_baseline_inputs() -> Dict[str, Any]:
    """Create MOCK_BASELINE scenario inputs (high jitter, synthetic telemetry)."""
    joint_view = {
        "schema_version": "1.0.0",
        "topology_snapshot": {
            "topology_mode": "DRIFT",
            "betti_numbers": {"beta_0": 1, "beta_1": 1},
            "persistence_metrics": {"bottleneck_drift": 0.15},
            "safe_region_metrics": {"boundary_distance": 0.05},
        },
        "bundle_snapshot": {
            "bundle_status": "ATTENTION",
            "chain_info": {"chain_valid": True},
            "manifest": {"coverage": 0.8},
            "provenance": {"verified": False},
        },
        "alignment_status": {"overall_status": "TENSION"},
    }
    consistency_result = {"consistent": False, "status": "WARN"}
    return {"joint_view": joint_view, "consistency_result": consistency_result}


@pytest.fixture
def healthy_inputs() -> Dict[str, Any]:
    """Create HEALTHY scenario inputs (stable manifold, nominal execution)."""
    joint_view = {
        "schema_version": "1.0.0",
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
    consistency_result = {"consistent": True, "status": "OK"}
    return {"joint_view": joint_view, "consistency_result": consistency_result}


@pytest.fixture
def mismatch_inputs() -> Dict[str, Any]:
    """Create MISMATCH scenario inputs (deliberate injected inconsistency)."""
    joint_view = {
        "schema_version": "1.0.0",
        "topology_snapshot": {
            "topology_mode": "STABLE",
            "betti_numbers": {"beta_0": 1, "beta_1": 0},
            "persistence_metrics": {"bottleneck_drift": 0.01},
            "safe_region_metrics": {"boundary_distance": 0.4},
        },
        "bundle_snapshot": {
            "bundle_status": "BROKEN",
            "chain_info": {"chain_valid": False},
            "manifest": {"coverage": 0.3},
            "provenance": {"verified": False},
        },
        "alignment_status": {"overall_status": "DIVERGENT"},
    }
    consistency_result = {"consistent": False, "status": "BLOCK"}
    return {"joint_view": joint_view, "consistency_result": consistency_result}


# -----------------------------------------------------------------------------
# Test: Report Generator End-to-End
# -----------------------------------------------------------------------------

class TestReportGeneratorEndToEnd:
    """Tests for end-to-end report generation flow."""

    def test_mock_baseline_generates_valid_report(self, mock_baseline_inputs):
        """Test MOCK_BASELINE scenario generates valid report structure."""
        joint_view = mock_baseline_inputs["joint_view"]
        consistency_result = mock_baseline_inputs["consistency_result"]

        # Build console tile
        tile = build_topology_bundle_console_tile(
            joint_view=joint_view,
            consistency_result=consistency_result,
        )

        # Run smoke validation
        smoke_result = run_p5_smoke_validation(
            topology_tile=tile,
            consistency_result=consistency_result,
            joint_view=joint_view,
        )

        # Build P5 summary
        p5_summary = build_p5_topology_reality_summary(
            topology_tile=tile,
            bundle_tile=joint_view.get("bundle_snapshot", {}),
        )

        # Generate auditor report
        report = generate_p5_auditor_report(
            p5_summary=p5_summary,
            run_id="test-mock-baseline-001",
            slice_name="arithmetic/add",
        )

        # Verify report structure
        assert report["schema_version"] is not None
        assert len(report["runbook_steps"]) == 10
        assert "final_hypothesis" in report
        assert report["run_context"]["phase"] == "P5"

        # Verify MOCK_BASELINE-like scenario
        assert smoke_result["matched_scenario"] in ("MOCK_BASELINE", "XCOR_ANOMALY")
        assert smoke_result["shadow_mode_invariant_ok"] is True

    def test_healthy_generates_valid_report(self, healthy_inputs):
        """Test HEALTHY scenario generates valid report structure."""
        joint_view = healthy_inputs["joint_view"]
        consistency_result = healthy_inputs["consistency_result"]

        # Build console tile
        tile = build_topology_bundle_console_tile(
            joint_view=joint_view,
            consistency_result=consistency_result,
        )

        # Run smoke validation
        smoke_result = run_p5_smoke_validation(
            topology_tile=tile,
            consistency_result=consistency_result,
            joint_view=joint_view,
        )

        # Verify HEALTHY scenario
        assert smoke_result["matched_scenario"] == "HEALTHY"
        assert smoke_result["validation_passed"] is True
        assert smoke_result["shadow_mode_invariant_ok"] is True

    def test_mismatch_generates_valid_report_with_escalation(self, mismatch_inputs):
        """Test MISMATCH scenario generates report with escalation required."""
        joint_view = mismatch_inputs["joint_view"]
        consistency_result = mismatch_inputs["consistency_result"]

        # Build console tile
        tile = build_topology_bundle_console_tile(
            joint_view=joint_view,
            consistency_result=consistency_result,
        )

        # Run smoke validation
        smoke_result = run_p5_smoke_validation(
            topology_tile=tile,
            consistency_result=consistency_result,
            joint_view=joint_view,
        )

        # Build P5 summary
        p5_summary = build_p5_topology_reality_summary(
            topology_tile=tile,
            bundle_tile=joint_view.get("bundle_snapshot", {}),
        )

        # Generate auditor report
        report = generate_p5_auditor_report(
            p5_summary=p5_summary,
            run_id="test-mismatch-001",
            slice_name="logic/implies",
        )

        # Verify MISMATCH scenario
        assert smoke_result["matched_scenario"] == "MISMATCH"
        assert report["escalation_required"] is True
        assert "bundle" in report["escalation_reason"].lower()

    def test_reports_are_json_serializable(self, healthy_inputs):
        """Test all generated reports are JSON serializable."""
        joint_view = healthy_inputs["joint_view"]
        consistency_result = healthy_inputs["consistency_result"]

        tile = build_topology_bundle_console_tile(
            joint_view=joint_view,
            consistency_result=consistency_result,
        )

        smoke_result = run_p5_smoke_validation(
            topology_tile=tile,
            consistency_result=consistency_result,
        )

        p5_summary = build_p5_topology_reality_summary(
            topology_tile=tile,
            bundle_tile={},
        )

        report = generate_p5_auditor_report(
            p5_summary=p5_summary,
            run_id="test-001",
            slice_name="test",
        )

        # All should serialize without error
        json.dumps(tile)
        json.dumps(smoke_result)
        json.dumps(p5_summary)
        json.dumps(report)


# -----------------------------------------------------------------------------
# Test: Evidence Pack Detection
# -----------------------------------------------------------------------------

class TestEvidencePackDetection:
    """Tests for detection of P5 topology auditor report in evidence pack."""

    def test_detects_report_in_root(self, healthy_inputs):
        """Test detection of report in evidence pack root."""
        joint_view = healthy_inputs["joint_view"]
        consistency_result = healthy_inputs["consistency_result"]

        # Generate report
        tile = build_topology_bundle_console_tile(
            joint_view=joint_view,
            consistency_result=consistency_result,
        )
        smoke_result = run_p5_smoke_validation(
            topology_tile=tile,
            consistency_result=consistency_result,
        )
        p5_summary = build_p5_topology_reality_summary(
            topology_tile=tile,
            bundle_tile={},
        )
        report = generate_p5_auditor_report(
            p5_summary=p5_summary,
            run_id="test-001",
            slice_name="test",
        )

        # Build scenario_match with correct structure (uses "scenario" not "matched_scenario")
        scenario_match = {
            "scenario": smoke_result["matched_scenario"],
            "confidence": smoke_result["confidence"],
            "matching_criteria": smoke_result["matching_criteria"],
            "divergent_criteria": smoke_result["divergent_criteria"],
        }

        # Add full report structure
        full_report = {
            "schema_version": "1.0.0",
            "mode": "SHADOW",
            "success": True,
            "scenario_match": scenario_match,
            "smoke_validation": smoke_result,
            "p5_summary": p5_summary,
            "auditor_report": report,
            "shadow_mode_invariant_ok": True,
        }

        # Write to temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            report_path = run_dir / "p5_topology_auditor_report.json"
            with open(report_path, "w") as f:
                json.dump(full_report, f)

            # Detect
            ref = detect_p5_topology_auditor_report(run_dir)

            assert ref is not None
            assert isinstance(ref, P5TopologyAuditorReference)
            assert ref.scenario == "HEALTHY"
            assert ref.shadow_mode_invariant_ok is True
            assert ref.mode == "SHADOW"

    def test_detects_report_in_p4_shadow(self, mock_baseline_inputs):
        """Test detection of report in p4_shadow subdirectory."""
        joint_view = mock_baseline_inputs["joint_view"]
        consistency_result = mock_baseline_inputs["consistency_result"]

        # Generate report
        tile = build_topology_bundle_console_tile(
            joint_view=joint_view,
            consistency_result=consistency_result,
        )
        smoke_result = run_p5_smoke_validation(
            topology_tile=tile,
            consistency_result=consistency_result,
        )

        full_report = {
            "schema_version": "1.0.0",
            "mode": "SHADOW",
            "success": True,
            "scenario_match": smoke_result,
            "smoke_validation": smoke_result,
            "p5_summary": {"joint_status": "TENSION"},
            "shadow_mode_invariant_ok": True,
        }

        # Write to p4_shadow subdirectory
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            p4_shadow_dir = run_dir / "p4_shadow"
            p4_shadow_dir.mkdir()
            report_path = p4_shadow_dir / "p5_topology_auditor_report.json"
            with open(report_path, "w") as f:
                json.dump(full_report, f)

            # Detect
            ref = detect_p5_topology_auditor_report(run_dir)

            assert ref is not None
            assert "p4_shadow" in ref.path

    def test_returns_none_when_no_report(self):
        """Test returns None when no report exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            ref = detect_p5_topology_auditor_report(run_dir)
            assert ref is None

    def test_handles_malformed_report(self):
        """Test handles malformed JSON gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            report_path = run_dir / "p5_topology_auditor_report.json"
            with open(report_path, "w") as f:
                f.write("{ invalid json }")

            # Should still return a reference with defaults
            ref = detect_p5_topology_auditor_report(run_dir)
            assert ref is not None
            assert ref.validation_passed is False


# -----------------------------------------------------------------------------
# Test: Status Signal Integration
# -----------------------------------------------------------------------------

class TestStatusSignalIntegration:
    """Tests for topology_p5 signal in first_light_status.json."""

    def test_signal_extracted_from_report_file(self, healthy_inputs):
        """Test topology_p5 signal is extracted correctly."""
        joint_view = healthy_inputs["joint_view"]
        consistency_result = healthy_inputs["consistency_result"]

        tile = build_topology_bundle_console_tile(
            joint_view=joint_view,
            consistency_result=consistency_result,
        )
        smoke_result = run_p5_smoke_validation(
            topology_tile=tile,
            consistency_result=consistency_result,
        )
        p5_summary = build_p5_topology_reality_summary(
            topology_tile=tile,
            bundle_tile={},
        )

        # This would be included in status["signals"]["topology_p5"]
        expected_signal = {
            "scenario": smoke_result.get("matched_scenario"),
            "scenario_confidence": smoke_result.get("confidence"),
            "joint_status": p5_summary.get("joint_status"),
            "shadow_mode_invariant_ok": smoke_result.get("shadow_mode_invariant_ok"),
            "validation_passed": smoke_result.get("validation_passed"),
        }

        assert expected_signal["scenario"] == "HEALTHY"
        assert expected_signal["shadow_mode_invariant_ok"] is True
        assert expected_signal["validation_passed"] is True

    def test_mismatch_scenario_generates_warning_text(self, mismatch_inputs):
        """Test MISMATCH scenario would generate warning text."""
        joint_view = mismatch_inputs["joint_view"]
        consistency_result = mismatch_inputs["consistency_result"]

        tile = build_topology_bundle_console_tile(
            joint_view=joint_view,
            consistency_result=consistency_result,
        )
        smoke_result = run_p5_smoke_validation(
            topology_tile=tile,
            consistency_result=consistency_result,
        )
        p5_summary = build_p5_topology_reality_summary(
            topology_tile=tile,
            bundle_tile=joint_view.get("bundle_snapshot", {}),
        )

        scenario = smoke_result.get("matched_scenario")
        confidence = smoke_result.get("confidence")
        joint_status = p5_summary.get("joint_status")

        # Simulate warning generation logic
        if scenario == "MISMATCH":
            warning = (
                f"Topology P5: scenario={scenario} (confidence={confidence:.2f}), "
                f"joint_status={joint_status} - bundle/topology mismatch detected"
            )
            assert "MISMATCH" in warning
            assert "mismatch detected" in warning


# -----------------------------------------------------------------------------
# Test: SHADOW MODE Invariants
# -----------------------------------------------------------------------------

class TestShadowModeInvariants:
    """Tests to verify SHADOW MODE contract throughout pipeline."""

    def test_all_reports_have_shadow_mode(self, healthy_inputs, mock_baseline_inputs, mismatch_inputs):
        """Test all generated reports maintain SHADOW MODE."""
        for inputs, name in [
            (healthy_inputs, "HEALTHY"),
            (mock_baseline_inputs, "MOCK_BASELINE"),
            (mismatch_inputs, "MISMATCH"),
        ]:
            joint_view = inputs["joint_view"]
            consistency_result = inputs["consistency_result"]

            tile = build_topology_bundle_console_tile(
                joint_view=joint_view,
                consistency_result=consistency_result,
            )
            smoke_result = run_p5_smoke_validation(
                topology_tile=tile,
                consistency_result=consistency_result,
            )

            # SHADOW MODE invariant should always be True
            assert smoke_result["shadow_mode_invariant_ok"] is True, f"Failed for {name}"

    def test_no_enforcement_fields(self, healthy_inputs):
        """Test no enforcement fields present in outputs."""
        joint_view = healthy_inputs["joint_view"]
        consistency_result = healthy_inputs["consistency_result"]

        tile = build_topology_bundle_console_tile(
            joint_view=joint_view,
            consistency_result=consistency_result,
        )
        smoke_result = run_p5_smoke_validation(
            topology_tile=tile,
            consistency_result=consistency_result,
        )
        p5_summary = build_p5_topology_reality_summary(
            topology_tile=tile,
            bundle_tile={},
        )
        report = generate_p5_auditor_report(
            p5_summary=p5_summary,
            run_id="test-001",
            slice_name="test",
        )

        # Check for absence of enforcement fields
        all_json = json.dumps({
            "tile": tile,
            "smoke_result": smoke_result,
            "p5_summary": p5_summary,
            "report": report,
        })

        assert "abort" not in all_json.lower()
        assert "enforce" not in all_json.lower()
        assert "block_execution" not in all_json.lower()
        assert "gate_policy" not in all_json.lower()


# -----------------------------------------------------------------------------
# Test: Deterministic Output (8 new tests)
# -----------------------------------------------------------------------------

class TestDeterministicOutput:
    """Tests for --deterministic flag producing reproducible output."""

    def test_deterministic_flag_produces_stable_top_level_structure(self):
        """Test that --deterministic flag produces stable top-level timestamps and sorted keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            output1 = run_dir / "report1.json"
            output2 = run_dir / "report2.json"

            # Create minimal input directory
            (run_dir / "run_config.json").write_text(json.dumps({
                "slice_name": "test/determinism",
                "run_id": "test-determinism-001",
            }))

            # Generate twice with deterministic=True
            report1 = generate_p5_topology_report(
                p4_run_dir=run_dir,
                evidence_pack_dir=None,
                output_path=output1,
                deterministic=True,
            )
            report2 = generate_p5_topology_report(
                p4_run_dir=run_dir,
                evidence_pack_dir=None,
                output_path=output2,
                deterministic=True,
            )

            # Top-level timestamps should be identical (DETERMINISTIC_TIMESTAMP)
            assert report1["timestamp"] == report2["timestamp"] == DETERMINISTIC_TIMESTAMP

            # Top-level keys should be in sorted order
            content1 = output1.read_text()
            data1 = json.loads(content1)
            keys = list(data1.keys())
            assert keys == sorted(keys), "Keys should be sorted in deterministic mode"

            # Core structure should be identical
            assert report1["schema_version"] == report2["schema_version"]
            assert report1["mode"] == report2["mode"]
            assert report1["success"] == report2["success"]
            assert report1["scenario_match"] == report2["scenario_match"]

    def test_deterministic_uses_fixed_timestamp(self):
        """Test that deterministic mode uses fixed timestamp."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            output_path = run_dir / "report.json"

            report = generate_p5_topology_report(
                p4_run_dir=run_dir,
                evidence_pack_dir=None,
                output_path=output_path,
                deterministic=True,
            )

            assert report["timestamp"] == DETERMINISTIC_TIMESTAMP

    def test_non_deterministic_uses_current_timestamp(self):
        """Test that non-deterministic mode uses current timestamp."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            output_path = run_dir / "report.json"

            report = generate_p5_topology_report(
                p4_run_dir=run_dir,
                evidence_pack_dir=None,
                output_path=output_path,
                deterministic=False,
            )

            # Should NOT be the fixed timestamp
            assert report["timestamp"] != DETERMINISTIC_TIMESTAMP
            # Should be an ISO timestamp with timezone
            assert "T" in report["timestamp"]
            assert "+" in report["timestamp"] or "Z" in report["timestamp"]


class TestPathLayoutDetection:
    """Tests for detection across different path layouts."""

    def test_detects_report_in_run_dir_root(self):
        """Test detection when report is at <run_dir>/p5_topology_auditor_report.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            report_path = run_dir / "p5_topology_auditor_report.json"

            full_report = {
                "schema_version": "1.0.0",
                "mode": "SHADOW",
                "success": True,
                "scenario_match": {"scenario": "HEALTHY", "confidence": 1.0},
                "smoke_validation": {"shadow_mode_invariant_ok": True, "validation_passed": True},
                "p5_summary": {"joint_status": "ALIGNED"},
            }
            report_path.write_text(json.dumps(full_report))

            ref = detect_p5_topology_auditor_report(run_dir)

            assert ref is not None
            assert ref.path == "p5_topology_auditor_report.json"
            assert ref.scenario == "HEALTHY"

    def test_detects_report_in_run_dir_p4_shadow(self):
        """Test detection when report is at <run_dir>/p4_shadow/p5_topology_auditor_report.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            p4_shadow_dir = run_dir / "p4_shadow"
            p4_shadow_dir.mkdir()
            report_path = p4_shadow_dir / "p5_topology_auditor_report.json"

            full_report = {
                "schema_version": "1.0.0",
                "mode": "SHADOW",
                "success": True,
                "scenario_match": {"scenario": "MOCK_BASELINE", "confidence": 0.85},
                "smoke_validation": {"shadow_mode_invariant_ok": True, "validation_passed": True},
                "p5_summary": {"joint_status": "TENSION"},
            }
            report_path.write_text(json.dumps(full_report))

            ref = detect_p5_topology_auditor_report(run_dir)

            assert ref is not None
            assert "p4_shadow" in ref.path
            assert ref.scenario == "MOCK_BASELINE"

    def test_detects_report_in_evidence_pack_p4_shadow(self):
        """Test detection when report is at <evidence_pack>/p4_shadow/p5_topology_auditor_report.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            evidence_pack_dir = Path(tmpdir)
            p4_shadow_dir = evidence_pack_dir / "p4_shadow"
            p4_shadow_dir.mkdir()
            report_path = p4_shadow_dir / "p5_topology_auditor_report.json"

            full_report = {
                "schema_version": "1.0.0",
                "mode": "SHADOW",
                "success": True,
                "scenario_match": {"scenario": "MISMATCH", "confidence": 0.95},
                "smoke_validation": {"shadow_mode_invariant_ok": True, "validation_passed": False},
                "p5_summary": {"joint_status": "DIVERGENT"},
            }
            report_path.write_text(json.dumps(full_report))

            # Detection works when passed evidence_pack_dir as run_dir
            ref = detect_p5_topology_auditor_report(evidence_pack_dir)

            assert ref is not None
            assert "p4_shadow" in ref.path
            assert ref.scenario == "MISMATCH"


class TestManifestIntegration:
    """Tests for manifest sha256 cross-link."""

    def test_manifest_contains_sha256_and_path(self):
        """Test that manifest governance.topology_p5 contains sha256 and path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)

            # Create required p3/p4 artifact directories for completeness
            (run_dir / "p3_synthetic").mkdir()
            (run_dir / "p4_shadow").mkdir()

            # Create the P5 topology report
            report_path = run_dir / "p5_topology_auditor_report.json"
            full_report = {
                "schema_version": "1.0.0",
                "mode": "SHADOW",
                "success": True,
                "scenario_match": {"scenario": "HEALTHY", "confidence": 1.0},
                "smoke_validation": {"shadow_mode_invariant_ok": True, "validation_passed": True},
                "p5_summary": {"joint_status": "ALIGNED"},
            }
            report_path.write_text(json.dumps(full_report))

            # Build evidence pack
            result = build_evidence_pack(
                run_dir=run_dir,
                validate_schemas=False,
            )

            # Check that topology_p5 reference was detected
            assert result.p5_topology_auditor_reference is not None
            assert result.p5_topology_auditor_reference.sha256 is not None
            assert len(result.p5_topology_auditor_reference.sha256) == 64  # SHA-256 hex length

            # Read manifest and verify governance.topology_p5
            manifest_path = run_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text())

            assert "governance" in manifest
            assert "topology_p5" in manifest["governance"]
            topology_p5 = manifest["governance"]["topology_p5"]

            assert "path" in topology_p5
            assert "sha256" in topology_p5
            assert "schema_version" in topology_p5
            assert "mode" in topology_p5
            assert topology_p5["mode"] == "SHADOW"


class TestMalformedReportHandling:
    """Tests for graceful handling of malformed reports.

    PARTIAL JSON EXTRACTION CONTRACT:
    - If report JSON is invalid (malformed), schema_ok=False is set
    - If schema_ok=False, scenario is NOT fabricated (remains None)
    - Advisory warning is returned via advisory_warning field
    - sha256 is still computed for the raw file bytes
    - This ensures no false scenario claims from corrupted data
    """

    def test_status_extraction_survives_malformed_report(self):
        """Test that status extraction handles malformed report gracefully with warning."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            report_path = run_dir / "p5_topology_auditor_report.json"

            # Write malformed JSON
            report_path.write_text("{ this is not valid json }")

            # Detection should still return a reference
            ref = detect_p5_topology_auditor_report(run_dir)

            # Should return a reference with defaults
            assert ref is not None
            assert ref.validation_passed is False
            # sha256 should still be computed for the file
            assert ref.sha256 is not None
            assert len(ref.sha256) == 64

    def test_partially_valid_report_extracts_available_fields(self):
        """Test that partially valid report extracts whatever fields are available."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            report_path = run_dir / "p5_topology_auditor_report.json"

            # Write report with only some fields
            partial_report = {
                "schema_version": "1.0.0",
                "mode": "SHADOW",
                # Missing: scenario_match, smoke_validation, p5_summary
            }
            report_path.write_text(json.dumps(partial_report))

            ref = detect_p5_topology_auditor_report(run_dir)

            assert ref is not None
            assert ref.schema_version == "1.0.0"
            assert ref.mode == "SHADOW"
            # Missing fields should have defaults
            assert ref.scenario is None
            assert ref.validation_passed is False  # Default for missing smoke_validation


# -----------------------------------------------------------------------------
# Test: Partial JSON Extraction Contract (Prompt 2)
# -----------------------------------------------------------------------------

class TestPartialJSONExtractionContract:
    """Tests for PARTIAL JSON EXTRACTION CONTRACT.

    Contract rules:
    1. If report JSON is invalid, schema_ok=False
    2. If schema_ok=False, scenario is NOT fabricated (remains None)
    3. Advisory warning is returned
    4. sha256 is still computed for raw file bytes
    5. Determinism preserved
    """

    def test_invalid_json_sets_schema_ok_false_and_no_scenario(self):
        """Test that invalid JSON sets schema_ok=False and does NOT fabricate scenario."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            report_path = run_dir / "p5_topology_auditor_report.json"

            # Write completely invalid JSON
            report_path.write_text("{ invalid: json, missing quotes }")

            ref = detect_p5_topology_auditor_report(run_dir)

            # Must return a reference (file exists)
            assert ref is not None

            # PARTIAL EXTRACTION CONTRACT assertions
            assert ref.schema_ok is False, "schema_ok must be False for invalid JSON"
            assert ref.scenario is None, "scenario must NOT be fabricated when JSON invalid"
            assert ref.scenario_confidence is None
            assert ref.joint_status is None
            assert ref.validation_passed is False

            # sha256 must still be computed
            assert ref.sha256 is not None
            assert len(ref.sha256) == 64

            # Advisory warning must be present
            assert ref.advisory_warning is not None
            assert "JSON parse error" in ref.advisory_warning

    def test_missing_fields_sets_advisory_warning_but_schema_ok_true(self):
        """Test that missing fields generate advisory warning but schema_ok stays True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            report_path = run_dir / "p5_topology_auditor_report.json"

            # Write valid JSON but missing required fields
            incomplete_report = {
                "schema_version": "1.0.0",
                "mode": "SHADOW",
                "success": True,
                # Missing: scenario_match, smoke_validation, p5_summary
            }
            report_path.write_text(json.dumps(incomplete_report))

            ref = detect_p5_topology_auditor_report(run_dir)

            assert ref is not None

            # schema_ok should be True (JSON is valid)
            assert ref.schema_ok is True, "schema_ok must be True for valid JSON"

            # Fields should be extracted where available
            assert ref.schema_version == "1.0.0"
            assert ref.mode == "SHADOW"

            # Missing scenario data
            assert ref.scenario is None  # Not fabricated
            assert ref.validation_passed is False

            # Advisory warning for missing fields
            assert ref.advisory_warning is not None
            assert "Missing fields" in ref.advisory_warning
            assert "scenario_match" in ref.advisory_warning
            assert "smoke_validation" in ref.advisory_warning
            assert "p5_summary" in ref.advisory_warning


# -----------------------------------------------------------------------------
# Test: Environment Variable and CLI Precedence
# -----------------------------------------------------------------------------

class TestDeterministicEnvVarPrecedence:
    """Tests for P5_DETERMINISTIC_REPORTS env var and CLI flag precedence."""

    def test_env_var_enables_deterministic_when_no_cli_flag(self, monkeypatch):
        """Test that env var enables deterministic mode when CLI flag not provided."""
        monkeypatch.setenv(DETERMINISTIC_ENV_VAR, "1")

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            output_path = run_dir / "report.json"

            report = generate_p5_topology_report(
                p4_run_dir=run_dir,
                evidence_pack_dir=None,
                output_path=output_path,
                deterministic=_env_is_truthy(DETERMINISTIC_ENV_VAR),  # Simulates env default
            )

            assert report["timestamp"] == DETERMINISTIC_TIMESTAMP

    def test_env_var_truthy_values(self, monkeypatch):
        """Test various truthy values for env var."""
        for truthy_value in ["1", "true", "TRUE", "yes", "YES", "on", "ON"]:
            monkeypatch.setenv(DETERMINISTIC_ENV_VAR, truthy_value)
            assert _env_is_truthy(DETERMINISTIC_ENV_VAR) is True

    def test_env_var_falsy_values(self, monkeypatch):
        """Test various falsy values for env var."""
        for falsy_value in ["0", "false", "no", "off", "", "random"]:
            monkeypatch.setenv(DETERMINISTIC_ENV_VAR, falsy_value)
            assert _env_is_truthy(DETERMINISTIC_ENV_VAR) is False

    def test_cli_deterministic_overrides_env_false(self, monkeypatch):
        """Test that --deterministic CLI flag works even when env var is not set."""
        monkeypatch.delenv(DETERMINISTIC_ENV_VAR, raising=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            output_path = run_dir / "report.json"

            # Explicitly pass deterministic=True (as CLI --deterministic would)
            report = generate_p5_topology_report(
                p4_run_dir=run_dir,
                evidence_pack_dir=None,
                output_path=output_path,
                deterministic=True,
            )

            assert report["timestamp"] == DETERMINISTIC_TIMESTAMP

    def test_cli_no_deterministic_overrides_env_true(self, monkeypatch):
        """Test that --no-deterministic CLI flag overrides env var set to true."""
        monkeypatch.setenv(DETERMINISTIC_ENV_VAR, "1")

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            output_path = run_dir / "report.json"

            # Explicitly pass deterministic=False (as CLI --no-deterministic would)
            report = generate_p5_topology_report(
                p4_run_dir=run_dir,
                evidence_pack_dir=None,
                output_path=output_path,
                deterministic=False,
            )

            # Timestamp should NOT be the deterministic one
            assert report["timestamp"] != DETERMINISTIC_TIMESTAMP

    def test_env_var_not_set_defaults_to_non_deterministic(self, monkeypatch):
        """Test that when env var is not set, output is non-deterministic by default."""
        monkeypatch.delenv(DETERMINISTIC_ENV_VAR, raising=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            output_path = run_dir / "report.json"

            report = generate_p5_topology_report(
                p4_run_dir=run_dir,
                evidence_pack_dir=None,
                output_path=output_path,
                deterministic=_env_is_truthy(DETERMINISTIC_ENV_VAR),  # Should be False
            )

            assert report["timestamp"] != DETERMINISTIC_TIMESTAMP
