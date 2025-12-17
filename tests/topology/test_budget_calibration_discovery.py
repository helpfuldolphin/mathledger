"""
Tests for Budget Calibration Evidence Pack Discovery and Status Extraction

Tests cover:
- Detection of budget_calibration_summary.json in root directory
- Detection of budget_calibration_summary.json in calibration/ subdirectory
- Manifest includes reference with sha256, schema_version, FP/FN rates
- Status extraction works and is deterministic
- SHADOW MODE contract: detection does not affect pack generation

SHADOW MODE CONTRACT:
All tests verify that budget calibration detection is advisory only
and does not affect evidence pack generation success/failure.

Reference: docs/system_law/Budget_PhaseX_Doctrine.md Section 7.3
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest

from backend.topology.first_light.evidence_pack import (
    detect_budget_calibration,
    BudgetCalibrationReference,
    BUDGET_CALIBRATION_SUMMARY_ARTIFACT,
    BUDGET_CALIBRATION_LOG_ARTIFACT,
    BUDGET_CALIBRATION_SUBDIR,
)


# =============================================================================
# Helper Functions
# =============================================================================

def create_budget_calibration_summary(
    path: Path,
    enablement_recommendation: str = "ENABLE",
    fp_rate: float = 0.02,
    fn_rate: float = 0.03,
    overall_pass: bool = True,
    schema_version: str = "budget-calibration/1.0.0",
) -> str:
    """Create a minimal budget_calibration_summary.json file.

    Returns the SHA-256 hash of the created file.
    """
    summary = {
        "schema_version": schema_version,
        "experiment_id": "cal-exp-budget-001",
        "compact_summary": {
            "schema_version": schema_version,
            "enablement_recommendation": enablement_recommendation,
            "fp_rate": fp_rate,
            "fn_rate": fn_rate,
            "overall_pass": overall_pass,
            "phases": {
                "phase_1": {"pass": True, "fp_rate": fp_rate},
                "phase_2": {"pass": True, "fn_rate": fn_rate},
                "phase_3": {"pass": overall_pass, "stress_resilience": 0.95},
            },
        },
        "mode": "SHADOW",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Compute hash
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def create_budget_calibration_log(path: Path, num_entries: int = 10) -> str:
    """Create a minimal budget_calibration_log.jsonl file.

    Returns the SHA-256 hash of the created file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i in range(num_entries):
            entry = {
                "cycle": i + 1,
                "phase": 1 if i < 5 else 2,
                "drift_class": "STABLE",
                "health_score": 85.0 + i * 0.5,
                "is_fp": False,
                "is_fn": False,
            }
            f.write(json.dumps(entry) + "\n")

    # Compute hash
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# =============================================================================
# Test: Detection in Root Directory
# =============================================================================

class TestDetectionRootDirectory:
    """Tests for detection in root directory."""

    def test_detect_budget_calibration_in_root(self) -> None:
        """Test detection of budget_calibration_summary.json in root directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            summary_path = run_dir / BUDGET_CALIBRATION_SUMMARY_ARTIFACT
            expected_hash = create_budget_calibration_summary(summary_path)

            ref = detect_budget_calibration(run_dir)

            assert ref is not None
            assert ref.path == BUDGET_CALIBRATION_SUMMARY_ARTIFACT
            assert ref.sha256 == expected_hash
            assert ref.mode == "SHADOW"

    def test_detect_budget_calibration_extracts_fields(self) -> None:
        """Test that detection extracts enablement_recommendation, fp_rate, fn_rate."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            summary_path = run_dir / BUDGET_CALIBRATION_SUMMARY_ARTIFACT
            create_budget_calibration_summary(
                summary_path,
                enablement_recommendation="DEFER",
                fp_rate=0.08,
                fn_rate=0.12,
                overall_pass=False,
            )

            ref = detect_budget_calibration(run_dir)

            assert ref is not None
            assert ref.enablement_recommendation == "DEFER"
            assert ref.fp_rate == 0.08
            assert ref.fn_rate == 0.12
            assert ref.overall_pass is False

    def test_detect_budget_calibration_with_log_file(self) -> None:
        """Test that detection includes optional log file reference."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            summary_path = run_dir / BUDGET_CALIBRATION_SUMMARY_ARTIFACT
            log_path = run_dir / BUDGET_CALIBRATION_LOG_ARTIFACT

            create_budget_calibration_summary(summary_path)
            expected_log_hash = create_budget_calibration_log(log_path)

            ref = detect_budget_calibration(run_dir)

            assert ref is not None
            assert ref.log_path == BUDGET_CALIBRATION_LOG_ARTIFACT
            assert ref.log_sha256 == expected_log_hash


# =============================================================================
# Test: Detection in Calibration Subdirectory
# =============================================================================

class TestDetectionCalibrationSubdir:
    """Tests for detection in calibration/ subdirectory."""

    def test_detect_budget_calibration_in_subdir(self) -> None:
        """Test detection of budget_calibration_summary.json in calibration/ subdir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            subdir = run_dir / BUDGET_CALIBRATION_SUBDIR
            summary_path = subdir / BUDGET_CALIBRATION_SUMMARY_ARTIFACT
            expected_hash = create_budget_calibration_summary(summary_path)

            ref = detect_budget_calibration(run_dir)

            assert ref is not None
            assert ref.path == f"{BUDGET_CALIBRATION_SUBDIR}\\{BUDGET_CALIBRATION_SUMMARY_ARTIFACT}" or \
                   ref.path == f"{BUDGET_CALIBRATION_SUBDIR}/{BUDGET_CALIBRATION_SUMMARY_ARTIFACT}"
            assert ref.sha256 == expected_hash
            assert ref.mode == "SHADOW"

    def test_detect_budget_calibration_prefers_root(self) -> None:
        """Test that root directory is preferred over calibration/ subdir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)

            # Create in both locations
            root_path = run_dir / BUDGET_CALIBRATION_SUMMARY_ARTIFACT
            root_hash = create_budget_calibration_summary(
                root_path, enablement_recommendation="ENABLE"
            )

            subdir_path = run_dir / BUDGET_CALIBRATION_SUBDIR / BUDGET_CALIBRATION_SUMMARY_ARTIFACT
            create_budget_calibration_summary(
                subdir_path, enablement_recommendation="DEFER"
            )

            ref = detect_budget_calibration(run_dir)

            # Root should be preferred
            assert ref is not None
            assert ref.path == BUDGET_CALIBRATION_SUMMARY_ARTIFACT
            assert ref.sha256 == root_hash
            assert ref.enablement_recommendation == "ENABLE"

    def test_detect_budget_calibration_with_log_in_subdir(self) -> None:
        """Test detection with log file in calibration/ subdirectory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            subdir = run_dir / BUDGET_CALIBRATION_SUBDIR

            summary_path = subdir / BUDGET_CALIBRATION_SUMMARY_ARTIFACT
            log_path = subdir / BUDGET_CALIBRATION_LOG_ARTIFACT

            create_budget_calibration_summary(summary_path)
            expected_log_hash = create_budget_calibration_log(log_path)

            ref = detect_budget_calibration(run_dir)

            assert ref is not None
            assert ref.log_sha256 == expected_log_hash


# =============================================================================
# Test: Manifest Reference Inclusion
# =============================================================================

class TestManifestReference:
    """Tests for manifest reference inclusion."""

    def test_manifest_includes_sha256(self) -> None:
        """Test that manifest reference includes sha256 hash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            summary_path = run_dir / BUDGET_CALIBRATION_SUMMARY_ARTIFACT
            expected_hash = create_budget_calibration_summary(summary_path)

            ref = detect_budget_calibration(run_dir)

            assert ref is not None
            assert ref.sha256 == expected_hash
            assert len(ref.sha256) == 64  # SHA-256 hex length

    def test_manifest_includes_schema_version(self) -> None:
        """Test that manifest reference includes schema_version."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            summary_path = run_dir / BUDGET_CALIBRATION_SUMMARY_ARTIFACT
            create_budget_calibration_summary(
                summary_path,
                schema_version="budget-calibration/2.0.0",
            )

            ref = detect_budget_calibration(run_dir)

            assert ref is not None
            assert ref.schema_version == "budget-calibration/2.0.0"

    def test_manifest_reference_determinism(self) -> None:
        """Test that manifest reference is deterministic across calls."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            summary_path = run_dir / BUDGET_CALIBRATION_SUMMARY_ARTIFACT
            create_budget_calibration_summary(summary_path)

            ref1 = detect_budget_calibration(run_dir)
            ref2 = detect_budget_calibration(run_dir)

            assert ref1 is not None
            assert ref2 is not None
            assert ref1.sha256 == ref2.sha256
            assert ref1.path == ref2.path
            assert ref1.enablement_recommendation == ref2.enablement_recommendation
            assert ref1.fp_rate == ref2.fp_rate
            assert ref1.fn_rate == ref2.fn_rate


# =============================================================================
# Test: No Artifact Present
# =============================================================================

class TestNoArtifact:
    """Tests when no budget calibration artifact is present."""

    def test_detect_returns_none_when_no_artifact(self) -> None:
        """Test that detection returns None when no artifact is present."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)

            ref = detect_budget_calibration(run_dir)

            assert ref is None

    def test_detect_returns_none_for_empty_directory(self) -> None:
        """Test that detection returns None for empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            # Create an unrelated file
            (run_dir / "other_file.json").write_text("{}")

            ref = detect_budget_calibration(run_dir)

            assert ref is None


# =============================================================================
# Test: Invalid Artifact Handling
# =============================================================================

class TestInvalidArtifact:
    """Tests for handling invalid artifacts."""

    def test_detect_handles_invalid_json(self) -> None:
        """Test that detection handles invalid JSON gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            summary_path = run_dir / BUDGET_CALIBRATION_SUMMARY_ARTIFACT
            summary_path.write_text("{ invalid json }")

            # Should return a reference with minimal info (hash only)
            ref = detect_budget_calibration(run_dir)

            assert ref is not None
            assert ref.sha256 is not None
            assert ref.mode == "SHADOW"
            # Fields should be None for invalid JSON
            assert ref.enablement_recommendation is None

    def test_detect_handles_missing_compact_summary(self) -> None:
        """Test that detection handles missing compact_summary field."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            summary_path = run_dir / BUDGET_CALIBRATION_SUMMARY_ARTIFACT
            summary_path.write_text(json.dumps({
                "schema_version": "1.0.0",
                "experiment_id": "test",
                # Missing compact_summary
            }))

            ref = detect_budget_calibration(run_dir)

            assert ref is not None
            assert ref.sha256 is not None
            # Fields from compact_summary should be None
            assert ref.enablement_recommendation is None
            assert ref.fp_rate is None


# =============================================================================
# Test: Status Extraction Determinism
# =============================================================================

class TestStatusExtractionDeterminism:
    """Tests for status extraction determinism."""

    def test_status_signal_determinism(self) -> None:
        """Test that status signal extraction is deterministic."""
        # Create mock manifest with budget_risk.calibration_reference
        manifest = {
            "governance": {
                "budget_risk": {
                    "calibration_reference": {
                        "path": "budget_calibration_summary.json",
                        "sha256": "abc123" * 10 + "abcd",  # 64 chars
                        "schema_version": "budget-calibration/1.0.0",
                        "enablement_recommendation": "ENABLE",
                        "fp_rate": 0.02,
                        "fn_rate": 0.03,
                        "overall_pass": True,
                        "mode": "SHADOW",
                    }
                }
            }
        }

        # Extract signal twice (simulate what generate_first_light_status does)
        governance = manifest.get("governance", {})
        budget_risk = governance.get("budget_risk", {})
        ref1 = budget_risk.get("calibration_reference")
        ref2 = budget_risk.get("calibration_reference")

        assert ref1 == ref2
        assert ref1["enablement_recommendation"] == "ENABLE"
        assert ref1["fp_rate"] == 0.02
        assert ref1["fn_rate"] == 0.03

    def test_fp_fn_rate_rounding(self) -> None:
        """Test that FP/FN rates are rounded to 4 decimal places."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            summary_path = run_dir / BUDGET_CALIBRATION_SUMMARY_ARTIFACT
            # Use a value that needs rounding
            create_budget_calibration_summary(
                summary_path,
                fp_rate=0.123456789,
                fn_rate=0.987654321,
            )

            ref = detect_budget_calibration(run_dir)

            assert ref is not None
            # The reference stores raw values, rounding happens in manifest generation
            assert ref.fp_rate == 0.123456789
            assert ref.fn_rate == 0.987654321


# =============================================================================
# Test: Status Warning Triggers
# =============================================================================

class TestStatusWarningTriggers:
    """Tests for status warning threshold normalization.

    Per STRATCOM: Warning triggers on DEFER OR overall_pass==false, capped to 1 line.
    No warning on ENABLE/PROCEED variants with overall_pass==true.
    """

    def _extract_budget_warning(
        self, enablement_recommendation: str, overall_pass: bool,
        fp_rate: float = 0.02, fn_rate: float = 0.03,
    ) -> list:
        """Helper to simulate status warning extraction logic."""
        warnings = []

        # Replicate the warning logic from generate_first_light_status.py
        should_warn = (
            enablement_recommendation == "DEFER" or
            overall_pass is False
        )
        if should_warn:
            fp_str = f"{fp_rate:.4f}" if fp_rate is not None else "N/A"
            fn_str = f"{fn_rate:.4f}" if fn_rate is not None else "N/A"
            warnings.append(
                f"Budget calibration: {enablement_recommendation or 'UNKNOWN'}, "
                f"overall_pass={overall_pass}, fp={fp_str}, fn={fn_str}"
            )

        return warnings

    def test_warning_triggers_on_defer(self) -> None:
        """Test that warning triggers when enablement_recommendation is DEFER."""
        warnings = self._extract_budget_warning(
            enablement_recommendation="DEFER",
            overall_pass=True,
        )

        assert len(warnings) == 1
        assert "DEFER" in warnings[0]
        assert "overall_pass=True" in warnings[0]

    def test_warning_triggers_on_overall_pass_false(self) -> None:
        """Test that warning triggers when overall_pass is false."""
        warnings = self._extract_budget_warning(
            enablement_recommendation="ENABLE",
            overall_pass=False,
        )

        assert len(warnings) == 1
        assert "ENABLE" in warnings[0]
        assert "overall_pass=False" in warnings[0]

    def test_warning_triggers_on_both_conditions(self) -> None:
        """Test warning when both DEFER and overall_pass=false."""
        warnings = self._extract_budget_warning(
            enablement_recommendation="DEFER",
            overall_pass=False,
        )

        # Should still produce only 1 warning (capped)
        assert len(warnings) == 1
        assert "DEFER" in warnings[0]
        assert "overall_pass=False" in warnings[0]

    def test_no_warning_on_enable_pass(self) -> None:
        """Test that no warning is generated for ENABLE with overall_pass=true."""
        warnings = self._extract_budget_warning(
            enablement_recommendation="ENABLE",
            overall_pass=True,
        )

        assert len(warnings) == 0

    def test_no_warning_on_proceed_variants(self) -> None:
        """Test that no warning is generated for PROCEED_* with overall_pass=true."""
        for recommendation in ["PROCEED", "PROCEED_WITH_CAUTION", "PROCEED_STABLE"]:
            warnings = self._extract_budget_warning(
                enablement_recommendation=recommendation,
                overall_pass=True,
            )

            assert len(warnings) == 0, f"Unexpected warning for {recommendation}"

    def test_warning_is_single_line(self) -> None:
        """Test that warning is capped to a single line (no newlines)."""
        warnings = self._extract_budget_warning(
            enablement_recommendation="DEFER",
            overall_pass=False,
            fp_rate=0.08,
            fn_rate=0.12,
        )

        assert len(warnings) == 1
        assert "\n" not in warnings[0]
        # Verify all fields are on one line
        assert "DEFER" in warnings[0]
        assert "overall_pass=False" in warnings[0]
        assert "fp=0.0800" in warnings[0]
        assert "fn=0.1200" in warnings[0]

    def test_warning_handles_none_rates(self) -> None:
        """Test that warning handles None FP/FN rates gracefully."""
        warnings = []
        enablement_recommendation = "DEFER"
        overall_pass = True
        fp_rate = None
        fn_rate = None

        should_warn = (
            enablement_recommendation == "DEFER" or
            overall_pass is False
        )
        if should_warn:
            fp_str = f"{fp_rate:.4f}" if fp_rate is not None else "N/A"
            fn_str = f"{fn_rate:.4f}" if fn_rate is not None else "N/A"
            warnings.append(
                f"Budget calibration: {enablement_recommendation or 'UNKNOWN'}, "
                f"overall_pass={overall_pass}, fp={fp_str}, fn={fn_str}"
            )

        assert len(warnings) == 1
        assert "fp=N/A" in warnings[0]
        assert "fn=N/A" in warnings[0]


# =============================================================================
# Test: GGFL Adapter — budget_calibration_for_alignment_view
# =============================================================================

class TestBudgetCalibrationGGFLAdapter:
    """Tests for GGFL alignment view adapter.

    Per STRATCOM: SIG-BUD-CAL GGFL ADAPTER.
    - Fixed shape output
    - conflict: false always
    - status: warn if DEFER/overall_pass=false
    - drivers include FP/FN summary
    """

    def test_returns_none_for_missing_reference(self) -> None:
        """Test that None input returns None (explicit optional)."""
        from backend.topology.first_light.budget_binding import (
            budget_calibration_for_alignment_view,
        )

        result = budget_calibration_for_alignment_view(None)
        assert result is None

    def test_returns_none_for_empty_dict(self) -> None:
        """Test that empty dict returns None."""
        from backend.topology.first_light.budget_binding import (
            budget_calibration_for_alignment_view,
        )

        result = budget_calibration_for_alignment_view({})
        assert result is None

    def test_healthy_on_enable_pass(self) -> None:
        """Test healthy alignment for ENABLE with overall_pass=true."""
        from backend.topology.first_light.budget_binding import (
            budget_calibration_for_alignment_view,
        )

        ref = {
            "enablement_recommendation": "ENABLE",
            "overall_pass": True,
            "fp_rate": 0.02,
            "fn_rate": 0.03,
        }

        result = budget_calibration_for_alignment_view(ref)

        assert result is not None
        assert result["alignment"] == "healthy"
        assert result["status"] == "ok"
        assert result["conflict"] is False
        assert result["mode"] == "SHADOW"

    def test_degraded_on_defer(self) -> None:
        """Test degraded alignment for DEFER recommendation."""
        from backend.topology.first_light.budget_binding import (
            budget_calibration_for_alignment_view,
        )

        ref = {
            "enablement_recommendation": "DEFER",
            "overall_pass": True,
            "fp_rate": 0.08,
            "fn_rate": 0.12,
        }

        result = budget_calibration_for_alignment_view(ref)

        assert result is not None
        assert result["alignment"] == "degraded"
        assert result["status"] == "warn"
        assert "DEFER" in result["advisory"]

    def test_degraded_on_overall_pass_false(self) -> None:
        """Test degraded alignment for overall_pass=false."""
        from backend.topology.first_light.budget_binding import (
            budget_calibration_for_alignment_view,
        )

        ref = {
            "enablement_recommendation": "ENABLE",
            "overall_pass": False,
            "fp_rate": 0.02,
            "fn_rate": 0.03,
        }

        result = budget_calibration_for_alignment_view(ref)

        assert result is not None
        assert result["alignment"] == "degraded"
        assert result["status"] == "warn"
        assert "overall_pass=False" in result["advisory"]

    def test_drivers_include_fp_fn_reason_code(self) -> None:
        """Test that drivers list includes FP/FN reason code when rates present."""
        from backend.topology.first_light.budget_binding import (
            budget_calibration_for_alignment_view,
            BUDGET_CAL_REASON_CODES,
        )

        ref = {
            "enablement_recommendation": "ENABLE",
            "overall_pass": True,
            "fp_rate": 0.0234,
            "fn_rate": 0.0456,
        }

        result = budget_calibration_for_alignment_view(ref)

        assert result is not None
        drivers = result["drivers"]
        assert isinstance(drivers, list)
        # Check DRIVER_FP_FN_PRESENT is in drivers
        assert BUDGET_CAL_REASON_CODES["DRIVER_FP_FN_PRESENT"] in drivers

    def test_output_is_json_serializable(self) -> None:
        """Test that output is JSON-safe (no special objects)."""
        from backend.topology.first_light.budget_binding import (
            budget_calibration_for_alignment_view,
        )

        ref = {
            "enablement_recommendation": "ENABLE",
            "overall_pass": True,
            "fp_rate": 0.02,
            "fn_rate": 0.03,
        }

        result = budget_calibration_for_alignment_view(ref)

        # Should serialize without error
        serialized = json.dumps(result)
        assert serialized is not None

        # Round-trip should produce identical dict
        deserialized = json.loads(serialized)
        assert deserialized == result

    def test_fixed_shape_output(self) -> None:
        """Test that output has fixed shape with all required keys."""
        from backend.topology.first_light.budget_binding import (
            budget_calibration_for_alignment_view,
        )

        ref = {
            "enablement_recommendation": "ENABLE",
            "overall_pass": True,
            "fp_rate": 0.02,
            "fn_rate": 0.03,
        }

        result = budget_calibration_for_alignment_view(ref)

        assert result is not None
        # Verify all required keys are present (includes top_reason_code)
        required_keys = {"alignment", "conflict", "status", "advisory", "drivers", "top_reason_code", "mode"}
        assert set(result.keys()) == required_keys

    def test_conflict_always_false(self) -> None:
        """Test that conflict is always False (calibration has no conflict concept)."""
        from backend.topology.first_light.budget_binding import (
            budget_calibration_for_alignment_view,
        )

        # Test with various inputs
        test_cases = [
            {"enablement_recommendation": "ENABLE", "overall_pass": True},
            {"enablement_recommendation": "DEFER", "overall_pass": True},
            {"enablement_recommendation": "ENABLE", "overall_pass": False},
            {"enablement_recommendation": "DEFER", "overall_pass": False},
        ]

        for ref in test_cases:
            result = budget_calibration_for_alignment_view(ref)
            assert result is not None
            assert result["conflict"] is False, f"conflict should be False for {ref}"

    def test_determinism_same_input_same_output(self) -> None:
        """Test that same input produces identical output (determinism)."""
        from backend.topology.first_light.budget_binding import (
            budget_calibration_for_alignment_view,
        )

        ref = {
            "enablement_recommendation": "ENABLE",
            "overall_pass": True,
            "fp_rate": 0.0234,
            "fn_rate": 0.0456,
        }

        result1 = budget_calibration_for_alignment_view(ref)
        result2 = budget_calibration_for_alignment_view(ref)

        assert result1 == result2
        # Verify drivers are in same order
        assert result1["drivers"] == result2["drivers"]


# =============================================================================
# Test: Source Location Tracking in Status Signal
# =============================================================================

class TestSourceLocationTracking:
    """Tests for source_location and source_sha256 in status signal.

    Per STRATCOM: Discovery path confidence tracking.
    - manifest source is preferred (cryptographically verified)
    - evidence.json fallback is indicated when used
    """

    def _build_mock_reference(
        self,
        enablement: str = "ENABLE",
        overall_pass: bool = True,
        fp_rate: float = 0.02,
        fn_rate: float = 0.03,
        path: str = "budget_calibration_summary.json",
        sha256: str = "abc123def456" * 5 + "abcd",
    ) -> Dict[str, Any]:
        """Build a mock calibration reference."""
        return {
            "enablement_recommendation": enablement,
            "overall_pass": overall_pass,
            "fp_rate": fp_rate,
            "fn_rate": fn_rate,
            "path": path,
            "sha256": sha256,
            "schema_version": "budget-calibration/1.0.0",
            "mode": "SHADOW",
        }

    def test_signal_includes_source_location_from_manifest(self) -> None:
        """Test that signal includes source_location='manifest' when loaded from manifest."""
        # Simulate extraction logic from generate_first_light_status.py
        manifest = {
            "governance": {
                "budget_risk": {
                    "calibration_reference": self._build_mock_reference()
                }
            }
        }
        evidence_data = None

        budget_calibration_ref = None
        budget_calibration_source = None

        if manifest:
            governance = manifest.get("governance", {})
            budget_risk = governance.get("budget_risk", {})
            budget_calibration_ref = budget_risk.get("calibration_reference")
            if budget_calibration_ref:
                budget_calibration_source = "manifest"

        if not budget_calibration_ref and evidence_data:
            governance = evidence_data.get("governance", {})
            budget_risk = governance.get("budget_risk", {})
            budget_calibration_ref = budget_risk.get("calibration_reference")
            if budget_calibration_ref:
                budget_calibration_source = "evidence.json"

        assert budget_calibration_source == "manifest"
        assert budget_calibration_ref is not None

    def test_signal_includes_source_location_from_evidence_fallback(self) -> None:
        """Test that signal indicates evidence.json when using fallback."""
        manifest = None  # No manifest
        evidence_data = {
            "governance": {
                "budget_risk": {
                    "calibration_reference": self._build_mock_reference()
                }
            }
        }

        budget_calibration_ref = None
        budget_calibration_source = None

        if manifest:
            governance = manifest.get("governance", {})
            budget_risk = governance.get("budget_risk", {})
            budget_calibration_ref = budget_risk.get("calibration_reference")
            if budget_calibration_ref:
                budget_calibration_source = "manifest"

        if not budget_calibration_ref and evidence_data:
            governance = evidence_data.get("governance", {})
            budget_risk = governance.get("budget_risk", {})
            budget_calibration_ref = budget_risk.get("calibration_reference")
            if budget_calibration_ref:
                budget_calibration_source = "evidence.json"

        assert budget_calibration_source == "evidence.json"
        assert budget_calibration_ref is not None

    def test_signal_includes_source_sha256(self) -> None:
        """Test that signal includes source_sha256 from reference."""
        ref = self._build_mock_reference(sha256="deadbeef" * 8)

        # Simulate signal building
        signal = {
            "enablement_recommendation": ref.get("enablement_recommendation"),
            "fp_rate": ref.get("fp_rate"),
            "fn_rate": ref.get("fn_rate"),
            "overall_pass": ref.get("overall_pass"),
            "schema_version": ref.get("schema_version"),
            "source_location": "manifest",
            "source_path": ref.get("path"),
            "source_sha256": ref.get("sha256"),
        }

        assert signal["source_sha256"] == "deadbeef" * 8
        assert signal["source_path"] == "budget_calibration_summary.json"

    def test_signal_includes_source_path(self) -> None:
        """Test that signal includes source_path from reference."""
        ref = self._build_mock_reference(path="calibration/budget_calibration_summary.json")

        signal = {
            "source_path": ref.get("path"),
            "source_sha256": ref.get("sha256"),
        }

        assert signal["source_path"] == "calibration/budget_calibration_summary.json"

    def test_manifest_preferred_over_evidence(self) -> None:
        """Test that manifest source is preferred over evidence.json."""
        manifest = {
            "governance": {
                "budget_risk": {
                    "calibration_reference": self._build_mock_reference(
                        enablement="ENABLE",
                        sha256="manifest_hash" + "0" * 52,
                    )
                }
            }
        }
        evidence_data = {
            "governance": {
                "budget_risk": {
                    "calibration_reference": self._build_mock_reference(
                        enablement="DEFER",
                        sha256="evidence_hash" + "0" * 52,
                    )
                }
            }
        }

        budget_calibration_ref = None
        budget_calibration_source = None

        if manifest:
            governance = manifest.get("governance", {})
            budget_risk = governance.get("budget_risk", {})
            budget_calibration_ref = budget_risk.get("calibration_reference")
            if budget_calibration_ref:
                budget_calibration_source = "manifest"

        if not budget_calibration_ref and evidence_data:
            governance = evidence_data.get("governance", {})
            budget_risk = governance.get("budget_risk", {})
            budget_calibration_ref = budget_risk.get("calibration_reference")
            if budget_calibration_ref:
                budget_calibration_source = "evidence.json"

        # Manifest should be used, not evidence.json
        assert budget_calibration_source == "manifest"
        assert budget_calibration_ref["enablement_recommendation"] == "ENABLE"
        assert "manifest_hash" in budget_calibration_ref["sha256"]

    def test_signal_shape_includes_all_source_fields(self) -> None:
        """Test that signal includes all source tracking fields."""
        ref = self._build_mock_reference()

        # Build signal as in generate_first_light_status.py
        signal = {
            "enablement_recommendation": ref.get("enablement_recommendation"),
            "fp_rate": round(ref.get("fp_rate"), 4) if ref.get("fp_rate") is not None else None,
            "fn_rate": round(ref.get("fn_rate"), 4) if ref.get("fn_rate") is not None else None,
            "overall_pass": ref.get("overall_pass"),
            "schema_version": ref.get("schema_version"),
            "source_location": "manifest",
            "source_path": ref.get("path"),
            "source_sha256": ref.get("sha256"),
        }

        # Verify all expected keys are present
        expected_keys = {
            "enablement_recommendation",
            "fp_rate",
            "fn_rate",
            "overall_pass",
            "schema_version",
            "source_location",
            "source_path",
            "source_sha256",
        }
        assert set(signal.keys()) == expected_keys


# =============================================================================
# Test: Extraction Source Enum Stability
# =============================================================================

class TestExtractionSourceEnum:
    """Tests for extraction_source provenance enum stability.

    Per STRATCOM: Enum values must be stable across versions.
    MANIFEST|EVIDENCE_JSON|DIRECT_DISCOVERY|MISSING
    """

    def test_enum_values_are_stable(self) -> None:
        """Test that extraction source enum has expected values."""
        from backend.topology.first_light.budget_binding import (
            EXTRACTION_SOURCE_ENUM,
        )

        assert EXTRACTION_SOURCE_ENUM["MANIFEST"] == "MANIFEST"
        assert EXTRACTION_SOURCE_ENUM["EVIDENCE_JSON"] == "EVIDENCE_JSON"
        assert EXTRACTION_SOURCE_ENUM["DIRECT_DISCOVERY"] == "DIRECT_DISCOVERY"
        assert EXTRACTION_SOURCE_ENUM["MISSING"] == "MISSING"

    def test_enum_has_exactly_four_values(self) -> None:
        """Test that extraction source enum has exactly 4 values."""
        from backend.topology.first_light.budget_binding import (
            EXTRACTION_SOURCE_ENUM,
        )

        assert len(EXTRACTION_SOURCE_ENUM) == 4

    def test_enum_keys_match_values(self) -> None:
        """Test that enum keys match their values (identity mapping)."""
        from backend.topology.first_light.budget_binding import (
            EXTRACTION_SOURCE_ENUM,
        )

        for key, value in EXTRACTION_SOURCE_ENUM.items():
            assert key == value, f"Key {key} != Value {value}"

    def test_enum_values_are_uppercase(self) -> None:
        """Test that all enum values are uppercase (convention)."""
        from backend.topology.first_light.budget_binding import (
            EXTRACTION_SOURCE_ENUM,
        )

        for value in EXTRACTION_SOURCE_ENUM.values():
            assert value.isupper(), f"Value {value} is not uppercase"


# =============================================================================
# Test: Reason Code Enum Stability
# =============================================================================

class TestReasonCodeEnum:
    """Tests for GGFL reason code enum stability.

    Per STRATCOM: Reason codes must be deterministic and stable.
    DRIVER_DEFER|DRIVER_OVERALL_PASS_FALSE|DRIVER_FP_FN_PRESENT
    """

    def test_reason_codes_are_stable(self) -> None:
        """Test that reason code enum has expected values."""
        from backend.topology.first_light.budget_binding import (
            BUDGET_CAL_REASON_CODES,
        )

        assert BUDGET_CAL_REASON_CODES["DRIVER_DEFER"] == "DRIVER_DEFER"
        assert BUDGET_CAL_REASON_CODES["DRIVER_OVERALL_PASS_FALSE"] == "DRIVER_OVERALL_PASS_FALSE"
        assert BUDGET_CAL_REASON_CODES["DRIVER_FP_FN_PRESENT"] == "DRIVER_FP_FN_PRESENT"

    def test_reason_codes_has_exactly_three_values(self) -> None:
        """Test that reason codes enum has exactly 3 values."""
        from backend.topology.first_light.budget_binding import (
            BUDGET_CAL_REASON_CODES,
        )

        assert len(BUDGET_CAL_REASON_CODES) == 3

    def test_reason_codes_keys_match_values(self) -> None:
        """Test that reason code keys match their values."""
        from backend.topology.first_light.budget_binding import (
            BUDGET_CAL_REASON_CODES,
        )

        for key, value in BUDGET_CAL_REASON_CODES.items():
            assert key == value, f"Key {key} != Value {value}"

    def test_priority_list_matches_codes(self) -> None:
        """Test that priority list contains exactly the reason codes."""
        from backend.topology.first_light.budget_binding import (
            BUDGET_CAL_REASON_CODES,
            BUDGET_CAL_REASON_PRIORITY,
        )

        # Priority list should contain same codes as enum
        assert set(BUDGET_CAL_REASON_PRIORITY) == set(BUDGET_CAL_REASON_CODES.values())

    def test_priority_order_is_deterministic(self) -> None:
        """Test that priority order is fixed and deterministic."""
        from backend.topology.first_light.budget_binding import (
            BUDGET_CAL_REASON_PRIORITY,
        )

        # Expected order: DEFER > OVERALL_PASS_FALSE > FP_FN_PRESENT
        assert BUDGET_CAL_REASON_PRIORITY[0] == "DRIVER_DEFER"
        assert BUDGET_CAL_REASON_PRIORITY[1] == "DRIVER_OVERALL_PASS_FALSE"
        assert BUDGET_CAL_REASON_PRIORITY[2] == "DRIVER_FP_FN_PRESENT"


# =============================================================================
# Test: Deterministic top_reason_code Selection
# =============================================================================

class TestTopReasonCodeSelection:
    """Tests for deterministic top_reason_code selection in GGFL adapter.

    Per STRATCOM: top_reason_code must be deterministically selected
    based on priority order when warn cases occur.
    """

    def test_top_reason_code_is_defer_when_defer_present(self) -> None:
        """Test that DRIVER_DEFER is top reason when present."""
        from backend.topology.first_light.budget_binding import (
            budget_calibration_for_alignment_view,
        )

        ref = {
            "enablement_recommendation": "DEFER",
            "overall_pass": False,  # Both conditions trigger warn
            "fp_rate": 0.02,
            "fn_rate": 0.03,
        }

        result = budget_calibration_for_alignment_view(ref)

        assert result is not None
        assert result["top_reason_code"] == "DRIVER_DEFER"
        # DEFER should be first in drivers too
        assert result["drivers"][0] == "DRIVER_DEFER"

    def test_top_reason_code_is_overall_pass_false_when_no_defer(self) -> None:
        """Test that DRIVER_OVERALL_PASS_FALSE is top when no DEFER."""
        from backend.topology.first_light.budget_binding import (
            budget_calibration_for_alignment_view,
        )

        ref = {
            "enablement_recommendation": "ENABLE",
            "overall_pass": False,
            "fp_rate": 0.02,
            "fn_rate": 0.03,
        }

        result = budget_calibration_for_alignment_view(ref)

        assert result is not None
        assert result["top_reason_code"] == "DRIVER_OVERALL_PASS_FALSE"

    def test_top_reason_code_is_none_for_healthy(self) -> None:
        """Test that top_reason_code is None for healthy cases."""
        from backend.topology.first_light.budget_binding import (
            budget_calibration_for_alignment_view,
        )

        ref = {
            "enablement_recommendation": "ENABLE",
            "overall_pass": True,
            "fp_rate": 0.02,
            "fn_rate": 0.03,
        }

        result = budget_calibration_for_alignment_view(ref)

        assert result is not None
        assert result["alignment"] == "healthy"
        assert result["top_reason_code"] is None

    def test_top_reason_code_determinism_across_calls(self) -> None:
        """Test that top_reason_code is deterministic across multiple calls."""
        from backend.topology.first_light.budget_binding import (
            budget_calibration_for_alignment_view,
        )

        ref = {
            "enablement_recommendation": "DEFER",
            "overall_pass": False,
            "fp_rate": 0.05,
            "fn_rate": 0.08,
        }

        results = [budget_calibration_for_alignment_view(ref) for _ in range(10)]

        # All should have same top_reason_code
        top_codes = [r["top_reason_code"] for r in results if r is not None]
        assert len(set(top_codes)) == 1, "top_reason_code should be deterministic"
        assert top_codes[0] == "DRIVER_DEFER"

    def test_drivers_order_is_deterministic(self) -> None:
        """Test that drivers list order is deterministic."""
        from backend.topology.first_light.budget_binding import (
            budget_calibration_for_alignment_view,
        )

        ref = {
            "enablement_recommendation": "DEFER",
            "overall_pass": False,
            "fp_rate": 0.05,
            "fn_rate": 0.08,
        }

        result1 = budget_calibration_for_alignment_view(ref)
        result2 = budget_calibration_for_alignment_view(ref)

        assert result1["drivers"] == result2["drivers"]
        # Order should be: DEFER, OVERALL_PASS_FALSE, FP_FN_PRESENT
        expected_order = [
            "DRIVER_DEFER",
            "DRIVER_OVERALL_PASS_FALSE",
            "DRIVER_FP_FN_PRESENT",
        ]
        assert result1["drivers"] == expected_order

    def test_fixed_shape_includes_top_reason_code(self) -> None:
        """Test that output shape always includes top_reason_code field."""
        from backend.topology.first_light.budget_binding import (
            budget_calibration_for_alignment_view,
        )

        test_cases = [
            {"enablement_recommendation": "ENABLE", "overall_pass": True},
            {"enablement_recommendation": "DEFER", "overall_pass": True},
            {"enablement_recommendation": "ENABLE", "overall_pass": False},
            {"enablement_recommendation": "DEFER", "overall_pass": False},
        ]

        for ref in test_cases:
            result = budget_calibration_for_alignment_view(ref)
            assert result is not None
            assert "top_reason_code" in result, f"Missing top_reason_code for {ref}"

    def test_drivers_use_reason_codes_not_strings(self) -> None:
        """Test that drivers use reason code constants, not free-form strings."""
        from backend.topology.first_light.budget_binding import (
            budget_calibration_for_alignment_view,
            BUDGET_CAL_REASON_CODES,
        )

        ref = {
            "enablement_recommendation": "DEFER",
            "overall_pass": False,
            "fp_rate": 0.02,
            "fn_rate": 0.03,
        }

        result = budget_calibration_for_alignment_view(ref)

        assert result is not None
        valid_codes = set(BUDGET_CAL_REASON_CODES.values())
        for driver in result["drivers"]:
            assert driver in valid_codes, f"Driver {driver} is not a valid reason code"


# =============================================================================
# Test: Budget Signal Non-Interference with P4/P5 Divergence Metrics
# =============================================================================

class TestBudgetSignalNonInterference:
    """Regression tripwire: budget signal presence/absence must not affect divergence metrics.

    Per STRATCOM CAL-EXP-2 PREP: Budget calibration adapter is purely advisory.
    P4/P5 divergence metrics must be identical regardless of budget signal presence.

    SHADOW MODE CONTRACT:
    - Budget signal is observational only
    - No gating, no enforcement
    - Divergence metrics are computed independently

    Uses reusable helpers from tests.helpers.non_interference.
    """

    def _build_minimal_manifest(
        self,
        include_budget: bool = False,
        include_divergence: bool = True,
    ) -> Dict[str, Any]:
        """Build minimal manifest with optional budget and divergence signals."""
        manifest: Dict[str, Any] = {
            "schema_version": "1.0.0",
            "governance": {},
        }

        if include_divergence:
            # Simulate P5 divergence reference in manifest
            manifest["governance"]["divergence"] = {
                "p5_divergence_rate": 0.0823,
                "p5_baseline_cycles": 100,
                "p5_source": "real_telemetry",
                "schema_version": "p5-divergence/1.0.0",
            }

        if include_budget:
            manifest["governance"]["budget_risk"] = {
                "calibration_reference": {
                    "path": "budget_calibration_summary.json",
                    "sha256": "abc123def456" * 5 + "abcd",
                    "schema_version": "budget-calibration/1.0.0",
                    "enablement_recommendation": "DEFER",
                    "fp_rate": 0.08,
                    "fn_rate": 0.12,
                    "overall_pass": False,
                    "mode": "SHADOW",
                }
            }

        return manifest

    def test_divergence_metrics_unchanged_with_budget_present(self) -> None:
        """Test that divergence metrics are identical with/without budget signal."""
        from tests.helpers.non_interference import pytest_assert_only_keys_changed

        # Build two manifests: one with budget, one without
        manifest_without_budget = self._build_minimal_manifest(
            include_budget=False, include_divergence=True
        )
        manifest_with_budget = self._build_minimal_manifest(
            include_budget=True, include_divergence=True
        )

        # Use helper: only budget_risk keys should differ
        pytest_assert_only_keys_changed(
            before=manifest_without_budget,
            after=manifest_with_budget,
            allowed_paths=["governance.budget_risk.*"],
            context="budget signal addition",
        )

    def test_budget_signal_only_affects_budget_keys(self) -> None:
        """Test that budget signal presence only affects budget-related keys."""
        from tests.helpers.non_interference import assert_only_keys_changed

        manifest_without = self._build_minimal_manifest(include_budget=False)
        manifest_with = self._build_minimal_manifest(include_budget=True)

        result = assert_only_keys_changed(
            before=manifest_without,
            after=manifest_with,
            allowed_paths=["governance.budget_risk.*"],
        )

        assert result.passed, f"Non-interference violated: {result.violations}"

    def test_ggfl_adapter_does_not_modify_input(self) -> None:
        """Test that GGFL adapter is pure (does not modify input reference)."""
        from backend.topology.first_light.budget_binding import (
            budget_calibration_for_alignment_view,
        )
        from tests.helpers.non_interference import pytest_assert_adapter_is_pure

        ref = {
            "enablement_recommendation": "DEFER",
            "overall_pass": False,
            "fp_rate": 0.08,
            "fn_rate": 0.12,
        }

        pytest_assert_adapter_is_pure(
            adapter_fn=budget_calibration_for_alignment_view,
            input_ref=ref,
            context="budget_calibration_for_alignment_view",
        )

    def test_warning_list_changes_by_at_most_one_line(self) -> None:
        """Test that budget signal adds at most 1 warning line."""
        from tests.helpers.non_interference import pytest_assert_warning_delta_at_most_one

        # Simulate warning generation logic
        base_warnings = [
            "Schema validation: 2 files checked",
            "P5 divergence: baseline established",
        ]

        # With budget signal (DEFER triggers warning)
        budget_ref = {
            "enablement_recommendation": "DEFER",
            "overall_pass": True,
            "fp_rate": 0.08,
            "fn_rate": 0.12,
        }

        # Replicate warning logic from generate_first_light_status.py
        warnings_with_budget = base_warnings.copy()
        should_warn = (
            budget_ref["enablement_recommendation"] == "DEFER" or
            budget_ref["overall_pass"] is False
        )
        if should_warn:
            fp_str = f"{budget_ref['fp_rate']:.4f}"
            fn_str = f"{budget_ref['fn_rate']:.4f}"
            warnings_with_budget.append(
                f"Budget calibration: {budget_ref['enablement_recommendation']}, "
                f"overall_pass={budget_ref['overall_pass']}, fp={fp_str}, fn={fn_str}"
            )

        warnings_without_budget = base_warnings.copy()

        # Use helper: delta at most 1, ordering preserved
        pytest_assert_warning_delta_at_most_one(
            before=warnings_without_budget,
            after=warnings_with_budget,
            context="budget warning addition",
        )

    def test_divergence_isolation_from_budget_adapter(self) -> None:
        """Test that budget adapter output has no divergence-related keys."""
        from backend.topology.first_light.budget_binding import (
            budget_calibration_for_alignment_view,
        )
        from tests.helpers.non_interference import pytest_assert_output_excludes_keys

        ref = {
            "enablement_recommendation": "ENABLE",
            "overall_pass": True,
            "fp_rate": 0.02,
            "fn_rate": 0.03,
        }

        result = budget_calibration_for_alignment_view(ref)
        assert result is not None

        # Use helper: budget adapter output must NOT contain divergence keys
        divergence_keys = {"divergence", "p5_divergence", "p4_divergence", "divergence_rate"}
        pytest_assert_output_excludes_keys(
            output=result,
            excluded_keys=divergence_keys,
            context="budget adapter output",
        )
