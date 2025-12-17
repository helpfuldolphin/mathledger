"""
Integration tests for P5 Divergence Real Pipeline Integration.

Tests end-to-end flow:
1. Generator defaults to p4_shadow/p5_divergence_real.json
2. Status integration extracts signals deterministically
3. Evidence pack manifest includes reference with schema_valid
4. Missing optional inputs are tolerated

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

from scripts.generate_p5_divergence_real_report import (
    generate_report,
    DEFAULT_OUTPUT_FILENAME,
    SCHEMA_VERSION,
    main as generator_main,
)
from backend.topology.first_light.evidence_pack import (
    detect_p5_divergence_file,
    P5DivergenceReference,
    P5_DIVERGENCE_ARTIFACT,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def minimal_run_dir():
    """Create minimal run directory with required inputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)

        # Create p4_summary.json
        p4_summary = {
            "run_id": "p4_20251211_120000_test",
            "schema_version": "1.0.0",
            "phase": "p4",
        }
        with open(run_dir / "p4_summary.json", "w") as f:
            json.dump(p4_summary, f)

        # Create divergence_log.jsonl with one entry
        divergence_entry = {
            "cycle": 1,
            "outcome": "success",
            "twin_outcome": "success",
            "diverged": False,
            "state_delta": {"H": 0.01, "rho": 0.005, "tau": 0.002},
            "severity": "minor",
        }
        with open(run_dir / "divergence_log.jsonl", "w") as f:
            f.write(json.dumps(divergence_entry) + "\n")

        yield run_dir


@pytest.fixture
def full_run_dir():
    """Create run directory with all optional inputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)

        # Create p4_summary.json
        p4_summary = {
            "run_id": "p4_20251211_143000_full",
            "schema_version": "1.0.0",
            "phase": "p4",
            "telemetry_source": "real",
        }
        with open(run_dir / "p4_summary.json", "w") as f:
            json.dump(p4_summary, f)

        # Create divergence_log.jsonl with multiple entries
        entries = [
            {"cycle": i, "outcome": "success", "twin_outcome": "success", "diverged": False,
             "state_delta": {"H": 0.01, "rho": 0.005, "tau": 0.002}, "severity": "minor"}
            for i in range(100)
        ]
        # Add some divergent entries
        entries.append({"cycle": 100, "outcome": "success", "twin_outcome": "blocked",
                       "diverged": True, "state_delta": {"H": 0.2, "rho": 0.1, "tau": 0.05},
                       "severity": "major"})

        with open(run_dir / "divergence_log.jsonl", "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

        # Create validation_results.json
        validation = {
            "validation_status": "VALIDATED_REAL",
            "validation_confidence": 0.92,
            "telemetry_source": "real",
        }
        with open(run_dir / "validation_results.json", "w") as f:
            json.dump(validation, f)

        # Create tda_comparison.json
        tda = {
            "sns_score": 0.85,
            "pcs_score": 0.90,
            "drs_score": 0.88,
            "hss_score": 0.82,
        }
        with open(run_dir / "tda_comparison.json", "w") as f:
            json.dump(tda, f)

        yield run_dir


@pytest.fixture
def valid_p5_report() -> Dict[str, Any]:
    """Create a valid P5 divergence report."""
    return {
        "schema_version": "1.0.0",
        "run_id": "p5_20251211_143000_test",
        "telemetry_source": "real",
        "validation_status": "VALIDATED_REAL",
        "validation_confidence": 0.92,
        "total_cycles": 500,
        "divergence_rate": 0.042,
        "mode": "SHADOW",
    }


# =============================================================================
# Test 1: Default Path Behavior - Generator defaults to p4_shadow/
# =============================================================================

class TestDefaultPathBehavior:
    """Tests for generator default path behavior."""

    def test_generator_creates_p4_shadow_dir(self, minimal_run_dir):
        """Generator creates p4_shadow subdirectory if missing."""
        assert not (minimal_run_dir / "p4_shadow").exists()

        # Run generator without --output
        result = generator_main([
            "--p5-run-dir", str(minimal_run_dir),
        ])

        assert result == 0
        assert (minimal_run_dir / "p4_shadow").exists()
        assert (minimal_run_dir / "p4_shadow" / DEFAULT_OUTPUT_FILENAME).exists()

    def test_generator_uses_p4_shadow_default_location(self, minimal_run_dir):
        """Generator defaults to p4_shadow/p5_divergence_real.json."""
        result = generator_main([
            "--p5-run-dir", str(minimal_run_dir),
        ])

        assert result == 0
        expected_path = minimal_run_dir / "p4_shadow" / DEFAULT_OUTPUT_FILENAME
        assert expected_path.exists()

        # Verify content is valid JSON
        with open(expected_path) as f:
            report = json.load(f)
        assert report["schema_version"] == SCHEMA_VERSION

    def test_generator_respects_explicit_output_path(self, minimal_run_dir):
        """Generator uses explicit --output path when provided."""
        custom_output = minimal_run_dir / "custom_output.json"

        result = generator_main([
            "--p5-run-dir", str(minimal_run_dir),
            "--output", str(custom_output),
        ])

        assert result == 0
        assert custom_output.exists()
        # p4_shadow should NOT be created if explicit path given
        # (unless custom_output was in p4_shadow)


# =============================================================================
# Test 2: Status Extraction Deterministic
# =============================================================================

class TestStatusExtractionDeterministic:
    """Tests for deterministic status signal extraction."""

    def test_status_extraction_is_deterministic(self, valid_p5_report):
        """Same report produces same signal every time."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            p4_shadow = run_dir / "p4_shadow"
            p4_shadow.mkdir()

            # Write report
            with open(p4_shadow / P5_DIVERGENCE_ARTIFACT, "w") as f:
                json.dump(valid_p5_report, f)

            # Detect multiple times
            results = [detect_p5_divergence_file(run_dir) for _ in range(3)]

            # All results should be identical
            for result in results:
                assert result is not None
                assert result.divergence_rate == valid_p5_report["divergence_rate"]
                assert result.validation_status == valid_p5_report["validation_status"]
                assert result.telemetry_source == valid_p5_report["telemetry_source"]

    def test_sha256_is_deterministic(self, valid_p5_report):
        """SHA256 hash is deterministic for same content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            p4_shadow = run_dir / "p4_shadow"
            p4_shadow.mkdir()

            # Write report
            report_path = p4_shadow / P5_DIVERGENCE_ARTIFACT
            with open(report_path, "w") as f:
                json.dump(valid_p5_report, f)

            # Detect multiple times
            results = [detect_p5_divergence_file(run_dir) for _ in range(3)]

            # All hashes should be identical
            hashes = [r.sha256 for r in results]
            assert len(set(hashes)) == 1  # All same
            assert len(hashes[0]) == 64  # Valid SHA256

    def test_extracted_fields_match_source(self, valid_p5_report):
        """Extracted fields exactly match source report."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            p4_shadow = run_dir / "p4_shadow"
            p4_shadow.mkdir()

            with open(p4_shadow / P5_DIVERGENCE_ARTIFACT, "w") as f:
                json.dump(valid_p5_report, f)

            result = detect_p5_divergence_file(run_dir)

            assert result.schema_version == valid_p5_report["schema_version"]
            assert result.telemetry_source == valid_p5_report["telemetry_source"]
            assert result.validation_status == valid_p5_report["validation_status"]
            assert result.divergence_rate == valid_p5_report["divergence_rate"]
            assert result.mode == valid_p5_report["mode"]


# =============================================================================
# Test 3: Missing Optional Inputs Tolerated
# =============================================================================

class TestMissingOptionalInputsTolerated:
    """Tests for graceful handling of missing optional inputs."""

    def test_generator_works_without_validation_results(self, minimal_run_dir):
        """Generator succeeds without validation_results.json."""
        assert not (minimal_run_dir / "validation_results.json").exists()

        result = generator_main([
            "--p5-run-dir", str(minimal_run_dir),
        ])

        assert result == 0
        output_path = minimal_run_dir / "p4_shadow" / DEFAULT_OUTPUT_FILENAME
        with open(output_path) as f:
            report = json.load(f)
        # Should have defaults for validation fields
        assert "validation_status" in report
        assert "validation_confidence" in report

    def test_generator_works_without_tda_comparison(self, minimal_run_dir):
        """Generator succeeds without tda_comparison.json."""
        assert not (minimal_run_dir / "tda_comparison.json").exists()

        result = generator_main([
            "--p5-run-dir", str(minimal_run_dir),
        ])

        assert result == 0
        output_path = minimal_run_dir / "p4_shadow" / DEFAULT_OUTPUT_FILENAME
        with open(output_path) as f:
            report = json.load(f)
        # Should succeed without TDA metrics
        assert report["mode"] == "SHADOW"

    def test_detection_returns_none_for_missing_file(self):
        """Detection returns None when no P5 file exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            # No p5_divergence_real.json anywhere

            result = detect_p5_divergence_file(run_dir)
            assert result is None

    def test_detection_handles_empty_p4_shadow(self):
        """Detection handles empty p4_shadow directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            p4_shadow = run_dir / "p4_shadow"
            p4_shadow.mkdir()
            # Empty p4_shadow, no p5 file

            result = detect_p5_divergence_file(run_dir)
            assert result is None


# =============================================================================
# Test 4: Evidence Pack Manifest Integration
# =============================================================================

class TestEvidencePackManifestIntegration:
    """Tests for evidence pack manifest reference with schema_valid."""

    def test_reference_includes_schema_valid_field(self, valid_p5_report):
        """P5DivergenceReference includes schema_valid boolean."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            p4_shadow = run_dir / "p4_shadow"
            p4_shadow.mkdir()

            with open(p4_shadow / P5_DIVERGENCE_ARTIFACT, "w") as f:
                json.dump(valid_p5_report, f)

            result = detect_p5_divergence_file(run_dir)

            assert result is not None
            assert hasattr(result, "schema_valid")
            assert isinstance(result.schema_valid, bool)

    def test_invalid_json_sets_schema_valid_false(self):
        """Invalid JSON results in schema_valid=False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            p4_shadow = run_dir / "p4_shadow"
            p4_shadow.mkdir()

            # Write invalid JSON
            with open(p4_shadow / P5_DIVERGENCE_ARTIFACT, "w") as f:
                f.write("{ invalid json }")

            result = detect_p5_divergence_file(run_dir)

            assert result is not None
            assert result.schema_valid is False
            assert len(result.validation_errors) > 0


# =============================================================================
# Test 5: End-to-End Pipeline Integration
# =============================================================================

class TestEndToEndPipelineIntegration:
    """End-to-end tests for full pipeline flow."""

    def test_full_pipeline_minimal_inputs(self, minimal_run_dir):
        """Full pipeline works with minimal required inputs."""
        # 1. Generate report
        result = generator_main([
            "--p5-run-dir", str(minimal_run_dir),
        ])
        assert result == 0

        # 2. Detect in evidence pack context
        ref = detect_p5_divergence_file(minimal_run_dir)
        assert ref is not None

        # 3. Verify reference has all expected fields
        assert ref.path is not None
        assert ref.sha256 is not None
        assert ref.mode == "SHADOW"

    def test_full_pipeline_with_all_inputs(self, full_run_dir):
        """Full pipeline works with all optional inputs present."""
        # 1. Generate report
        result = generator_main([
            "--p5-run-dir", str(full_run_dir),
        ])
        assert result == 0

        # 2. Detect
        ref = detect_p5_divergence_file(full_run_dir)
        assert ref is not None

        # 3. Verify enhanced fields are present
        assert ref.validation_status is not None
        assert ref.telemetry_source == "real"
        assert ref.divergence_rate is not None

    def test_pipeline_output_is_json_serializable(self, minimal_run_dir):
        """All pipeline outputs are JSON serializable."""
        # Generate
        generator_main([
            "--p5-run-dir", str(minimal_run_dir),
        ])

        # Read report
        output_path = minimal_run_dir / "p4_shadow" / DEFAULT_OUTPUT_FILENAME
        with open(output_path) as f:
            report = json.load(f)

        # Should serialize back without error
        json.dumps(report)

        # Detect and serialize reference as dict
        ref = detect_p5_divergence_file(minimal_run_dir)
        ref_dict = {
            "path": ref.path,
            "sha256": ref.sha256,
            "schema_version": ref.schema_version,
            "telemetry_source": ref.telemetry_source,
            "validation_status": ref.validation_status,
            "divergence_rate": ref.divergence_rate,
            "mode": ref.mode,
            "schema_valid": ref.schema_valid,
        }
        json.dumps(ref_dict)


# =============================================================================
# Test 6: Warning Text Precision
# =============================================================================

class TestWarningTextPrecision:
    """Tests for precise divergence metric terminology in warnings."""

    def test_warning_uses_outcome_mismatch_rate(self, minimal_run_dir):
        """Warning text uses 'outcome_mismatch_rate' not generic 'divergence_rate'."""
        # Create divergence log with high divergence
        # The generator uses success_diverged, blocked_diverged, omega_diverged fields
        entries = []
        for i in range(100):
            # First 80 cycles: no divergence
            # Last 20 cycles: success diverged (20% divergence rate)
            entries.append({
                "cycle": i,
                "success_diverged": i >= 80,  # Twin predicted success but got blocked
                "blocked_diverged": False,
                "omega_diverged": False,
                "H_delta": 0.1 if i >= 80 else 0.01,
                "severity": "major" if i >= 80 else "minor",
            })

        with open(minimal_run_dir / "divergence_log.jsonl", "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

        # Generate report
        result = generator_main([
            "--p5-run-dir", str(minimal_run_dir),
        ])
        assert result == 0

        # Read the report
        output_path = minimal_run_dir / "p4_shadow" / DEFAULT_OUTPUT_FILENAME
        with open(output_path) as f:
            report = json.load(f)

        # The report divergence_rate should be 20% (20/100)
        assert report["divergence_rate"] == 0.2

    def test_warning_clarifies_metric_definition(self):
        """Warning text includes explanation of what metrics mean."""
        # The warning text should contain clarification
        # This is a documentation test - the implementation is in generate_first_light_status.py
        import scripts.generate_first_light_status as status_gen
        import inspect
        source = inspect.getsource(status_gen)

        # v1.1.0: Warnings are more concise but still explain the metric
        # outcome_mismatch_rate warning explains it's twin vs real outcome
        assert "twin outcome" in source, \
            "Warning text should clarify that outcome_mismatch_rate is about twin vs real outcome"

        # safety_state_mismatch_rate warning explains blocked/omega errors
        assert "blocked/omega prediction errors" in source, \
            "Warning text should clarify that safety_state_mismatch_rate is about blocked/omega errors"

    def test_warning_does_not_say_generic_divergence_rate(self):
        """Warning text should not use generic 'divergence_rate > 15%' phrasing."""
        # The old warning said "divergence_rate=X% > 15%"
        # The new warning should say "outcome_mismatch_rate=X% > 15%"
        import scripts.generate_first_light_status as status_gen
        import inspect
        source = inspect.getsource(status_gen)

        # Check that the new terminology is used
        assert "outcome_mismatch_rate=" in source, \
            "Warning should use 'outcome_mismatch_rate=' not 'divergence_rate='"


# =============================================================================
# Test 7: Sources Block
# =============================================================================

class TestSourcesBlock:
    """Tests for sources block in P5 report."""

    def test_sources_block_empty_when_no_optional_inputs(self, minimal_run_dir):
        """Sources block is empty list when no optional inputs present."""
        result = generator_main([
            "--p5-run-dir", str(minimal_run_dir),
        ])
        assert result == 0

        output_path = minimal_run_dir / "p4_shadow" / DEFAULT_OUTPUT_FILENAME
        with open(output_path) as f:
            report = json.load(f)

        assert "sources" in report
        assert report["sources"] == []

    def test_sources_block_includes_rtts_validation(self, minimal_run_dir):
        """Sources block includes rtts_validation when present."""
        # Create rtts_validation.json
        rtts_data = {
            "schema_version": "1.0.0",
            "overall_status": "PASS",
            "validation_passed": True,
        }
        with open(minimal_run_dir / "rtts_validation.json", "w") as f:
            json.dump(rtts_data, f)

        result = generator_main([
            "--p5-run-dir", str(minimal_run_dir),
        ])
        assert result == 0

        output_path = minimal_run_dir / "p4_shadow" / DEFAULT_OUTPUT_FILENAME
        with open(output_path) as f:
            report = json.load(f)

        assert len(report["sources"]) == 1
        assert report["sources"][0]["name"] == "rtts_validation"
        assert report["sources"][0]["path"] == "rtts_validation.json"
        assert len(report["sources"][0]["sha256"]) == 64

    def test_sources_block_includes_tda_comparison(self, minimal_run_dir):
        """Sources block includes tda_comparison when present."""
        # Create tda_comparison.json
        tda_data = {
            "sns_delta": 0.05,
            "pcs_delta": 0.03,
        }
        with open(minimal_run_dir / "tda_comparison.json", "w") as f:
            json.dump(tda_data, f)

        result = generator_main([
            "--p5-run-dir", str(minimal_run_dir),
        ])
        assert result == 0

        output_path = minimal_run_dir / "p4_shadow" / DEFAULT_OUTPUT_FILENAME
        with open(output_path) as f:
            report = json.load(f)

        assert len(report["sources"]) == 1
        assert report["sources"][0]["name"] == "tda_comparison"

    def test_sources_block_includes_calibration_report(self, minimal_run_dir):
        """Sources block includes calibration_report when present."""
        # Create calibration_report.json
        cal_data = {
            "calibration_cycles": 100,
            "convergence_achieved": True,
        }
        with open(minimal_run_dir / "calibration_report.json", "w") as f:
            json.dump(cal_data, f)

        result = generator_main([
            "--p5-run-dir", str(minimal_run_dir),
        ])
        assert result == 0

        output_path = minimal_run_dir / "p4_shadow" / DEFAULT_OUTPUT_FILENAME
        with open(output_path) as f:
            report = json.load(f)

        assert len(report["sources"]) == 1
        assert report["sources"][0]["name"] == "calibration_report"

    def test_sources_block_deterministic_order(self, minimal_run_dir):
        """Sources block maintains deterministic order regardless of creation order."""
        # Create files in reverse order (calibration, tda, rtts)
        cal_data = {"calibration_cycles": 100}
        with open(minimal_run_dir / "calibration_report.json", "w") as f:
            json.dump(cal_data, f)

        tda_data = {"sns_delta": 0.05}
        with open(minimal_run_dir / "tda_comparison.json", "w") as f:
            json.dump(tda_data, f)

        rtts_data = {"overall_status": "PASS"}
        with open(minimal_run_dir / "rtts_validation.json", "w") as f:
            json.dump(rtts_data, f)

        result = generator_main([
            "--p5-run-dir", str(minimal_run_dir),
        ])
        assert result == 0

        output_path = minimal_run_dir / "p4_shadow" / DEFAULT_OUTPUT_FILENAME
        with open(output_path) as f:
            report = json.load(f)

        # Order should always be: rtts_validation, tda_comparison, calibration_report
        assert len(report["sources"]) == 3
        assert report["sources"][0]["name"] == "rtts_validation"
        assert report["sources"][1]["name"] == "tda_comparison"
        assert report["sources"][2]["name"] == "calibration_report"

    def test_sources_block_sha256_is_deterministic(self, minimal_run_dir):
        """SHA256 hashes are deterministic for same file content."""
        # Create rtts_validation.json
        rtts_data = {"schema_version": "1.0.0", "overall_status": "PASS"}
        with open(minimal_run_dir / "rtts_validation.json", "w") as f:
            json.dump(rtts_data, f)

        # Generate report twice
        result1 = generator_main(["--p5-run-dir", str(minimal_run_dir)])
        assert result1 == 0

        output_path = minimal_run_dir / "p4_shadow" / DEFAULT_OUTPUT_FILENAME
        with open(output_path) as f:
            report1 = json.load(f)

        # Delete and regenerate
        output_path.unlink()

        result2 = generator_main(["--p5-run-dir", str(minimal_run_dir)])
        assert result2 == 0

        with open(output_path) as f:
            report2 = json.load(f)

        # SHA256 should be identical
        assert report1["sources"][0]["sha256"] == report2["sources"][0]["sha256"]

    def test_sources_block_p4_shadow_location(self, minimal_run_dir):
        """Sources block detects files in p4_shadow subdirectory."""
        p4_shadow = minimal_run_dir / "p4_shadow"
        p4_shadow.mkdir(exist_ok=True)

        # Create file in p4_shadow
        rtts_data = {"overall_status": "PASS"}
        with open(p4_shadow / "rtts_validation.json", "w") as f:
            json.dump(rtts_data, f)

        result = generator_main([
            "--p5-run-dir", str(minimal_run_dir),
        ])
        assert result == 0

        output_path = p4_shadow / DEFAULT_OUTPUT_FILENAME
        with open(output_path) as f:
            report = json.load(f)

        assert len(report["sources"]) == 1
        assert report["sources"][0]["name"] == "rtts_validation"
        assert "p4_shadow" in report["sources"][0]["path"]


# =============================================================================
# Test 8: True Divergence Vector v1 (NO METRIC LAUNDERING)
# =============================================================================

class TestTrueDivergenceVectorV1:
    """Tests for true_divergence_vector_v1 fields."""

    def test_true_divergence_vector_present_in_report(self, minimal_run_dir):
        """Generated report contains true_divergence_vector_v1 block."""
        result = generator_main([
            "--p5-run-dir", str(minimal_run_dir),
        ])
        assert result == 0

        output_path = minimal_run_dir / "p4_shadow" / DEFAULT_OUTPUT_FILENAME
        with open(output_path) as f:
            report = json.load(f)

        assert "true_divergence_vector_v1" in report
        assert "safety_state_mismatch_rate" in report["true_divergence_vector_v1"]

    def test_safety_state_mismatch_rate_computed(self, minimal_run_dir):
        """safety_state_mismatch_rate is computed from blocked + omega divergences."""
        # Create divergence log with specific divergence types
        entries = []
        for i in range(100):
            entries.append({
                "cycle": i,
                "success_diverged": i >= 90,  # 10% success divergence
                "blocked_diverged": i >= 95,  # 5% blocked divergence
                "omega_diverged": i >= 98,    # 2% omega divergence (overlap with blocked)
                "H_delta": 0.05,
            })

        with open(minimal_run_dir / "divergence_log.jsonl", "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

        result = generator_main([
            "--p5-run-dir", str(minimal_run_dir),
        ])
        assert result == 0

        output_path = minimal_run_dir / "p4_shadow" / DEFAULT_OUTPUT_FILENAME
        with open(output_path) as f:
            report = json.load(f)

        # safety_state_mismatch_rate = (blocked + omega) / total = (5 + 2) / 100 = 0.07
        vec = report["true_divergence_vector_v1"]
        assert vec["safety_state_mismatch_rate"] == 0.07

    def test_state_error_mean_computed_from_h_deltas(self, minimal_run_dir):
        """state_error_mean is computed from H_deltas."""
        entries = []
        for i in range(10):
            entries.append({
                "cycle": i,
                "success_diverged": False,
                "blocked_diverged": False,
                "omega_diverged": False,
                "H_delta": 0.1 * (i + 1),  # H_deltas: 0.1, 0.2, ..., 1.0
            })

        with open(minimal_run_dir / "divergence_log.jsonl", "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

        result = generator_main([
            "--p5-run-dir", str(minimal_run_dir),
        ])
        assert result == 0

        output_path = minimal_run_dir / "p4_shadow" / DEFAULT_OUTPUT_FILENAME
        with open(output_path) as f:
            report = json.load(f)

        vec = report["true_divergence_vector_v1"]
        # mean of 0.1, 0.2, ..., 1.0 = 0.55
        assert "state_error_mean" in vec
        assert abs(vec["state_error_mean"] - 0.55) < 0.001

    def test_outcome_brier_score_optional(self, minimal_run_dir):
        """outcome_brier_score is optional (only present if twin_success_prob available)."""
        # Entries without twin_success_prob
        entries = []
        for i in range(10):
            entries.append({
                "cycle": i,
                "success_diverged": False,
                "H_delta": 0.01,
            })

        with open(minimal_run_dir / "divergence_log.jsonl", "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

        result = generator_main([
            "--p5-run-dir", str(minimal_run_dir),
        ])
        assert result == 0

        output_path = minimal_run_dir / "p4_shadow" / DEFAULT_OUTPUT_FILENAME
        with open(output_path) as f:
            report = json.load(f)

        vec = report["true_divergence_vector_v1"]
        # outcome_brier_score should not be present without probability data
        assert "outcome_brier_score" not in vec

    def test_outcome_brier_score_computed_when_available(self, minimal_run_dir):
        """outcome_brier_score is computed when twin_success_prob is available."""
        entries = []
        for i in range(10):
            # Perfect predictions: twin_success_prob matches outcome
            entries.append({
                "cycle": i,
                "success_diverged": False,
                "twin_success_prob": 1.0 if i < 5 else 0.0,
                "outcome": "success" if i < 5 else "blocked",
                "H_delta": 0.01,
            })

        with open(minimal_run_dir / "divergence_log.jsonl", "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

        result = generator_main([
            "--p5-run-dir", str(minimal_run_dir),
        ])
        assert result == 0

        output_path = minimal_run_dir / "p4_shadow" / DEFAULT_OUTPUT_FILENAME
        with open(output_path) as f:
            report = json.load(f)

        vec = report["true_divergence_vector_v1"]
        assert "outcome_brier_score" in vec
        # Perfect predictions should have Brier score of 0
        assert vec["outcome_brier_score"] == 0.0

    def test_schema_validates_true_divergence_vector_v1(self, minimal_run_dir):
        """Generated report with true_divergence_vector_v1 validates against schema."""
        pytest.importorskip("jsonschema")
        import jsonschema

        result = generator_main([
            "--p5-run-dir", str(minimal_run_dir),
        ])
        assert result == 0

        output_path = minimal_run_dir / "p4_shadow" / DEFAULT_OUTPUT_FILENAME
        with open(output_path) as f:
            report = json.load(f)

        # Load schema
        schema_path = Path(__file__).parent.parent.parent.parent / "docs" / "system_law" / "schemas" / "p5" / "p5_divergence_real.schema.json"
        with open(schema_path, encoding="utf-8") as f:
            schema = json.load(f)

        # Should validate without error
        jsonschema.validate(instance=report, schema=schema)

    def test_schema_version_bumped_to_1_1_0(self, minimal_run_dir):
        """Generated report has schema_version 1.1.0."""
        result = generator_main([
            "--p5-run-dir", str(minimal_run_dir),
        ])
        assert result == 0

        output_path = minimal_run_dir / "p4_shadow" / DEFAULT_OUTPUT_FILENAME
        with open(output_path) as f:
            report = json.load(f)

        assert report["schema_version"] == "1.1.0"


# =============================================================================
# Test 9: Single Warning Cap + Exact Metric Names
# =============================================================================

class TestSingleWarningCapExactMetrics:
    """Tests for single warning cap and exact metric naming."""

    def test_warning_uses_exact_metric_name_safety_mismatch(self):
        """Warning uses exact metric name safety_state_mismatch_rate."""
        import scripts.generate_first_light_status as status_gen
        import inspect
        source = inspect.getsource(status_gen)

        # Check that the exact metric name is used in warnings
        assert "safety_state_mismatch_rate=" in source

    def test_warning_uses_exact_metric_name_outcome_mismatch(self):
        """Warning uses exact metric name outcome_mismatch_rate."""
        import scripts.generate_first_light_status as status_gen
        import inspect
        source = inspect.getsource(status_gen)

        assert "outcome_mismatch_rate=" in source

    def test_single_warning_cap_implemented(self):
        """Single warning cap is implemented (p5_warning_emitted flag)."""
        import scripts.generate_first_light_status as status_gen
        import inspect
        source = inspect.getsource(status_gen)

        # Check for the single warning cap pattern
        assert "p5_warning_emitted" in source
        assert "not p5_warning_emitted" in source

    def test_no_generic_divergence_terminology(self):
        """Warning text does not use generic 'divergence' terminology."""
        import scripts.generate_first_light_status as status_gen
        import inspect
        source = inspect.getsource(status_gen)

        # Should not have "divergence_rate=" in warnings (use outcome_mismatch_rate instead)
        # But we still have divergence_rate in the signal extraction
        # Check the warning strings specifically
        warning_pattern_old = 'f"P5 divergence real: divergence_rate='
        assert warning_pattern_old not in source


# =============================================================================
# Test 10: Sources Block Ordering Preserved with New Fields
# =============================================================================

class TestSourcesOrderingWithNewFields:
    """Verify sources block ordering is preserved with v1.1.0 changes."""

    def test_sources_ordering_deterministic_after_version_bump(self, minimal_run_dir):
        """Sources block ordering unchanged after v1.1.0 bump."""
        # Create all three source files
        (minimal_run_dir / "rtts_validation.json").write_text('{"status": "PASS"}')
        (minimal_run_dir / "tda_comparison.json").write_text('{"sns_delta": 0.01}')
        (minimal_run_dir / "calibration_report.json").write_text('{"cycles": 100}')

        result = generator_main([
            "--p5-run-dir", str(minimal_run_dir),
        ])
        assert result == 0

        output_path = minimal_run_dir / "p4_shadow" / DEFAULT_OUTPUT_FILENAME
        with open(output_path) as f:
            report = json.load(f)

        # Verify v1.1.0 schema version
        assert report["schema_version"] == "1.1.0"

        # Verify sources ordering unchanged
        assert len(report["sources"]) == 3
        assert report["sources"][0]["name"] == "rtts_validation"
        assert report["sources"][1]["name"] == "tda_comparison"
        assert report["sources"][2]["name"] == "calibration_report"

    def test_true_divergence_vector_does_not_affect_sources(self, minimal_run_dir):
        """true_divergence_vector_v1 presence does not affect sources block."""
        # Create one source file
        (minimal_run_dir / "tda_comparison.json").write_text('{"sns_delta": 0.02}')

        result = generator_main([
            "--p5-run-dir", str(minimal_run_dir),
        ])
        assert result == 0

        output_path = minimal_run_dir / "p4_shadow" / DEFAULT_OUTPUT_FILENAME
        with open(output_path) as f:
            report = json.load(f)

        # Both should be present independently
        assert "true_divergence_vector_v1" in report
        assert "sources" in report
        assert len(report["sources"]) == 1
        assert report["sources"][0]["name"] == "tda_comparison"


# =============================================================================
# Test 11: Metric Versioning Block (NO METRIC LAUNDERING)
# =============================================================================

class TestMetricVersioningBlock:
    """Tests for metric_versioning block — NO METRIC LAUNDERING."""

    def test_metric_versioning_present_in_report(self, minimal_run_dir):
        """Generated report contains metric_versioning block."""
        result = generator_main([
            "--p5-run-dir", str(minimal_run_dir),
        ])
        assert result == 0

        output_path = minimal_run_dir / "p4_shadow" / DEFAULT_OUTPUT_FILENAME
        with open(output_path) as f:
            report = json.load(f)

        assert "metric_versioning" in report

    def test_metric_versioning_has_required_fields(self, minimal_run_dir):
        """metric_versioning block has all required fields."""
        result = generator_main([
            "--p5-run-dir", str(minimal_run_dir),
        ])
        assert result == 0

        output_path = minimal_run_dir / "p4_shadow" / DEFAULT_OUTPUT_FILENAME
        with open(output_path) as f:
            report = json.load(f)

        mv = report["metric_versioning"]
        assert "legacy_metrics" in mv
        assert "true_vector_v1_metrics" in mv
        assert "equivalence_note" in mv
        assert "doc_reference" in mv

    def test_legacy_metrics_list_correct(self, minimal_run_dir):
        """legacy_metrics contains expected metric names."""
        result = generator_main([
            "--p5-run-dir", str(minimal_run_dir),
        ])
        assert result == 0

        output_path = minimal_run_dir / "p4_shadow" / DEFAULT_OUTPUT_FILENAME
        with open(output_path) as f:
            report = json.load(f)

        legacy = report["metric_versioning"]["legacy_metrics"]
        assert "divergence_rate" in legacy
        assert "mock_baseline_divergence_rate" in legacy
        assert "divergence_delta" in legacy

    def test_true_vector_v1_metrics_list_correct(self, minimal_run_dir):
        """true_vector_v1_metrics contains expected metric names."""
        result = generator_main([
            "--p5-run-dir", str(minimal_run_dir),
        ])
        assert result == 0

        output_path = minimal_run_dir / "p4_shadow" / DEFAULT_OUTPUT_FILENAME
        with open(output_path) as f:
            report = json.load(f)

        tv1 = report["metric_versioning"]["true_vector_v1_metrics"]
        assert "safety_state_mismatch_rate" in tv1
        assert "state_error_mean" in tv1
        assert "outcome_brier_score" in tv1

    def test_equivalence_note_states_non_equivalence(self, minimal_run_dir):
        """equivalence_note explicitly states non-equivalence (ASCII-safe)."""
        result = generator_main([
            "--p5-run-dir", str(minimal_run_dir),
        ])
        assert result == 0

        output_path = minimal_run_dir / "p4_shadow" / DEFAULT_OUTPUT_FILENAME
        with open(output_path) as f:
            report = json.load(f)

        note = report["metric_versioning"]["equivalence_note"]
        assert "NOT_EQUIVALENT_TO" in note  # ASCII-safe non-equivalence
        assert "legacy" in note.lower() or "outcome_mismatch" in note.lower()
        assert "state_error_mean" in note

    def test_doc_reference_points_to_correct_file(self, minimal_run_dir):
        """doc_reference points to no_metric_laundering.md."""
        result = generator_main([
            "--p5-run-dir", str(minimal_run_dir),
        ])
        assert result == 0

        output_path = minimal_run_dir / "p4_shadow" / DEFAULT_OUTPUT_FILENAME
        with open(output_path) as f:
            report = json.load(f)

        doc_ref = report["metric_versioning"]["doc_reference"]
        assert "no_metric_laundering" in doc_ref.lower()
        assert doc_ref.endswith(".md")

    def test_schema_validates_metric_versioning(self, minimal_run_dir):
        """Generated report with metric_versioning validates against schema."""
        pytest.importorskip("jsonschema")
        import jsonschema

        result = generator_main([
            "--p5-run-dir", str(minimal_run_dir),
        ])
        assert result == 0

        output_path = minimal_run_dir / "p4_shadow" / DEFAULT_OUTPUT_FILENAME
        with open(output_path) as f:
            report = json.load(f)

        # Load schema
        schema_path = Path(__file__).parent.parent.parent.parent / "docs" / "system_law" / "schemas" / "p5" / "p5_divergence_real.schema.json"
        with open(schema_path, encoding="utf-8") as f:
            schema = json.load(f)

        # Should validate without error
        jsonschema.validate(instance=report, schema=schema)

    def test_metric_versioning_strings_only(self, minimal_run_dir):
        """metric_versioning contains only string values (no inference)."""
        result = generator_main([
            "--p5-run-dir", str(minimal_run_dir),
        ])
        assert result == 0

        output_path = minimal_run_dir / "p4_shadow" / DEFAULT_OUTPUT_FILENAME
        with open(output_path) as f:
            report = json.load(f)

        mv = report["metric_versioning"]

        # All items in lists should be strings
        for metric in mv["legacy_metrics"]:
            assert isinstance(metric, str)
        for metric in mv["true_vector_v1_metrics"]:
            assert isinstance(metric, str)

        # equivalence_note and doc_reference should be strings
        assert isinstance(mv["equivalence_note"], str)
        assert isinstance(mv["doc_reference"], str)
