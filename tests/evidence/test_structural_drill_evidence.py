"""Integration tests for structural drill artifact in evidence pack.

Verifies that:
1. Drill artifacts are detected in evidence pack run directories
2. Evidence pack manifest includes drill reference under governance.structure.drill
3. SHA256 hash is included in manifest

SHADOW MODE: All tests verify observational behavior only.

See: docs/system_law/P5_Structural_Drill_Package.md
"""

import json
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory

from backend.dag.structural_drill_runner import run_structural_drill
from backend.topology.first_light.evidence_pack import (
    EvidencePackBuilder,
    detect_structural_drill_artifact,
    StructuralDrillReference,
    STRUCTURAL_DRILL_ARTIFACT,
    STRUCTURAL_DRILL_DIR,
)


class TestStructuralDrillDetection:
    """Test structural drill artifact detection."""

    def test_detect_drill_artifact_not_present(self, tmp_path: Path):
        """Returns None when no drill artifact exists."""
        result = detect_structural_drill_artifact(tmp_path)
        assert result is None

    def test_detect_drill_artifact_in_root(self, tmp_path: Path):
        """Detects drill artifact in run directory root."""
        # Run drill and generate artifact
        artifact = run_structural_drill(
            scenario_id="TEST-DETECT-001",
            sample_rate=50,
            output_dir=None,
        )

        # Write artifact to root
        artifact_path = tmp_path / STRUCTURAL_DRILL_ARTIFACT
        with open(artifact_path, "w", encoding="utf-8") as f:
            json.dump(artifact.to_dict(), f)

        # Detect
        reference = detect_structural_drill_artifact(tmp_path)

        assert reference is not None
        assert reference.drill_id == artifact.drill_id
        assert reference.scenario_id == "TEST-DETECT-001"
        assert reference.sha256 is not None
        assert len(reference.sha256) == 64  # SHA-256 hex

    def test_detect_drill_artifact_in_subdir(self, tmp_path: Path):
        """Detects drill artifact in p5_structural_drill subdirectory."""
        # Create subdir
        drill_dir = tmp_path / STRUCTURAL_DRILL_DIR
        drill_dir.mkdir(parents=True)

        # Run drill and write to subdir
        artifact = run_structural_drill(
            scenario_id="TEST-DETECT-002",
            sample_rate=50,
            output_dir=None,
        )

        artifact_path = drill_dir / STRUCTURAL_DRILL_ARTIFACT
        with open(artifact_path, "w", encoding="utf-8") as f:
            json.dump(artifact.to_dict(), f)

        # Detect from parent
        reference = detect_structural_drill_artifact(tmp_path)

        assert reference is not None
        assert reference.drill_id == artifact.drill_id
        assert STRUCTURAL_DRILL_DIR in reference.path

    def test_detect_extracts_summary_fields(self, tmp_path: Path):
        """Verifies summary fields are extracted correctly."""
        artifact = run_structural_drill(
            scenario_id="TEST-SUMMARY-001",
            sample_rate=1,  # Sample every cycle for accurate counts
            output_dir=None,
        )

        artifact_path = tmp_path / STRUCTURAL_DRILL_ARTIFACT
        with open(artifact_path, "w", encoding="utf-8") as f:
            json.dump(artifact.to_dict(), f)

        reference = detect_structural_drill_artifact(tmp_path)

        assert reference is not None
        assert reference.drill_success == artifact.summary.get("drill_success")
        assert reference.max_streak == artifact.summary.get("max_streak")
        assert reference.break_events == artifact.summary.get("break_events")
        assert reference.pattern_counts == artifact.summary.get("pattern_counts")
        assert reference.mode == "SHADOW"


class TestEvidencePackDrillIntegration:
    """Integration test: drill artifact included in evidence pack."""

    def test_evidence_pack_includes_drill_reference(self, tmp_path: Path):
        """Evidence pack manifest includes structural drill reference."""
        # Setup: Create minimal run directory structure
        p3_dir = tmp_path / "p3_synthetic"
        p4_dir = tmp_path / "p4_shadow"
        p3_dir.mkdir()
        p4_dir.mkdir()

        # Create required P3/P4 artifacts (minimal stubs)
        (p3_dir / "synthetic_raw.jsonl").write_text('{"cycle":1}\n')
        (p3_dir / "stability_report.json").write_text('{"stable":true}')
        (p3_dir / "red_flag_matrix.json").write_text('{"flags":[]}')
        (p3_dir / "metrics_windows.json").write_text('{"windows":[]}')
        (p4_dir / "divergence_log.jsonl").write_text('{"cycle":1}\n')
        (p4_dir / "twin_trajectory.jsonl").write_text('{"cycle":1}\n')
        (p4_dir / "calibration_report.json").write_text('{"calibrated":true}')

        # Run structural drill into run directory
        drill_dir = tmp_path / STRUCTURAL_DRILL_DIR
        drill_artifact = run_structural_drill(
            scenario_id="TEST-EVIDENCE-001",
            sample_rate=50,
            output_dir=drill_dir,
        )

        # Build evidence pack
        builder = EvidencePackBuilder(validate_schemas=False)
        result = builder.build_evidence_pack(
            run_dir=tmp_path,
            output_dir=tmp_path,
            p3_run_id="test-p3",
            p4_run_id="test-p4",
        )

        assert result.success is True
        assert result.structural_drill_reference is not None
        assert result.structural_drill_reference.drill_id == drill_artifact.drill_id

        # Verify manifest includes drill reference
        manifest_path = tmp_path / "manifest.json"
        assert manifest_path.exists()

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        assert "governance" in manifest
        assert "structure" in manifest["governance"]
        assert "drill" in manifest["governance"]["structure"]

        drill_ref = manifest["governance"]["structure"]["drill"]
        assert drill_ref["drill_id"] == drill_artifact.drill_id
        assert drill_ref["scenario_id"] == "TEST-EVIDENCE-001"
        assert "sha256" in drill_ref
        assert drill_ref["mode"] == "SHADOW"
        assert drill_ref["shadow_mode_contract"]["observational_only"] is True

    def test_evidence_pack_without_drill_has_no_reference(self, tmp_path: Path):
        """Evidence pack without drill artifact has no structural drill reference."""
        # Setup: Create minimal run directory structure WITHOUT drill
        p3_dir = tmp_path / "p3_synthetic"
        p4_dir = tmp_path / "p4_shadow"
        p3_dir.mkdir()
        p4_dir.mkdir()

        # Create required P3/P4 artifacts (minimal stubs)
        (p3_dir / "synthetic_raw.jsonl").write_text('{"cycle":1}\n')
        (p3_dir / "stability_report.json").write_text('{"stable":true}')
        (p3_dir / "red_flag_matrix.json").write_text('{"flags":[]}')
        (p3_dir / "metrics_windows.json").write_text('{"windows":[]}')
        (p4_dir / "divergence_log.jsonl").write_text('{"cycle":1}\n')
        (p4_dir / "twin_trajectory.jsonl").write_text('{"cycle":1}\n')
        (p4_dir / "calibration_report.json").write_text('{"calibrated":true}')

        # Build evidence pack (no drill)
        builder = EvidencePackBuilder(validate_schemas=False)
        result = builder.build_evidence_pack(
            run_dir=tmp_path,
            output_dir=tmp_path,
            p3_run_id="test-p3",
            p4_run_id="test-p4",
        )

        assert result.success is True
        assert result.structural_drill_reference is None

        # Verify manifest does not include drill reference
        manifest_path = tmp_path / "manifest.json"
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        # governance.structure.drill should not exist
        if "governance" in manifest and "structure" in manifest.get("governance", {}):
            assert "drill" not in manifest["governance"]["structure"]

    def test_drill_artifact_hash_in_manifest(self, tmp_path: Path):
        """Drill artifact SHA256 hash is correctly included in manifest."""
        import hashlib

        # Setup run directory
        p3_dir = tmp_path / "p3_synthetic"
        p4_dir = tmp_path / "p4_shadow"
        p3_dir.mkdir()
        p4_dir.mkdir()

        (p3_dir / "synthetic_raw.jsonl").write_text('{"cycle":1}\n')
        (p3_dir / "stability_report.json").write_text('{"stable":true}')
        (p3_dir / "red_flag_matrix.json").write_text('{"flags":[]}')
        (p3_dir / "metrics_windows.json").write_text('{"windows":[]}')
        (p4_dir / "divergence_log.jsonl").write_text('{"cycle":1}\n')
        (p4_dir / "twin_trajectory.jsonl").write_text('{"cycle":1}\n')
        (p4_dir / "calibration_report.json").write_text('{"calibrated":true}')

        # Run drill
        drill_dir = tmp_path / STRUCTURAL_DRILL_DIR
        drill_artifact = run_structural_drill(
            scenario_id="TEST-HASH-001",
            sample_rate=50,
            output_dir=drill_dir,
        )

        # Compute expected hash
        artifact_path = drill_dir / STRUCTURAL_DRILL_ARTIFACT
        with open(artifact_path, "rb") as f:
            expected_hash = hashlib.sha256(f.read()).hexdigest()

        # Build evidence pack
        builder = EvidencePackBuilder(validate_schemas=False)
        result = builder.build_evidence_pack(
            run_dir=tmp_path,
            output_dir=tmp_path,
        )

        # Verify hash matches
        with open(tmp_path / "manifest.json", "r", encoding="utf-8") as f:
            manifest = json.load(f)

        drill_ref = manifest["governance"]["structure"]["drill"]
        assert drill_ref["sha256"] == expected_hash


class TestShadowModeInvariants:
    """Test SHADOW MODE invariants are maintained."""

    def test_detection_is_observational_only(self, tmp_path: Path):
        """Detection does not modify any files."""
        # Create drill artifact
        artifact = run_structural_drill(
            scenario_id="TEST-SHADOW-001",
            sample_rate=50,
        )

        artifact_path = tmp_path / STRUCTURAL_DRILL_ARTIFACT
        with open(artifact_path, "w", encoding="utf-8") as f:
            json.dump(artifact.to_dict(), f)

        # Record file stats before detection
        stat_before = artifact_path.stat()

        # Detect (should not modify)
        detect_structural_drill_artifact(tmp_path)

        # Verify file unchanged
        stat_after = artifact_path.stat()
        assert stat_before.st_mtime == stat_after.st_mtime
        assert stat_before.st_size == stat_after.st_size

    def test_manifest_includes_shadow_mode_contract(self, tmp_path: Path):
        """Manifest drill reference includes SHADOW MODE contract markers."""
        # Setup run directory
        p3_dir = tmp_path / "p3_synthetic"
        p4_dir = tmp_path / "p4_shadow"
        p3_dir.mkdir()
        p4_dir.mkdir()

        (p3_dir / "synthetic_raw.jsonl").write_text('{"cycle":1}\n')
        (p3_dir / "stability_report.json").write_text('{"stable":true}')
        (p3_dir / "red_flag_matrix.json").write_text('{"flags":[]}')
        (p3_dir / "metrics_windows.json").write_text('{"windows":[]}')
        (p4_dir / "divergence_log.jsonl").write_text('{"cycle":1}\n')
        (p4_dir / "twin_trajectory.jsonl").write_text('{"cycle":1}\n')
        (p4_dir / "calibration_report.json").write_text('{"calibrated":true}')

        # Run drill
        drill_dir = tmp_path / STRUCTURAL_DRILL_DIR
        run_structural_drill(
            scenario_id="TEST-CONTRACT-001",
            sample_rate=50,
            output_dir=drill_dir,
        )

        # Build evidence pack
        builder = EvidencePackBuilder(validate_schemas=False)
        builder.build_evidence_pack(run_dir=tmp_path, output_dir=tmp_path)

        # Verify SHADOW MODE contract
        with open(tmp_path / "manifest.json", "r", encoding="utf-8") as f:
            manifest = json.load(f)

        drill_ref = manifest["governance"]["structure"]["drill"]

        assert drill_ref["mode"] == "SHADOW"
        assert "shadow_mode_contract" in drill_ref
        assert drill_ref["shadow_mode_contract"]["observational_only"] is True
        assert drill_ref["shadow_mode_contract"]["no_control_flow_influence"] is True
        assert drill_ref["shadow_mode_contract"]["no_governance_modification"] is True
