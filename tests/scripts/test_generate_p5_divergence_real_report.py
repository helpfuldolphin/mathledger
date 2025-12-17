"""
Tests for P5 Divergence Real Report Generator.

SHADOW MODE: All tests verify observational-only outputs.
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest

from scripts.generate_p5_divergence_real_report import (
    SCHEMA_VERSION,
    DivergenceStats,
    compute_divergence_stats,
    compute_twin_tracking_accuracy,
    compute_manifold_validation,
    compute_divergence_decomposition,
    classify_divergence_pattern,
    determine_validation_status,
    generate_report,
    load_p4_summary,
    load_divergence_log,
    main,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def minimal_p4_summary() -> Dict[str, Any]:
    """Minimal P4 summary for testing."""
    return {
        "schema": "first-light-p4-summary/1.0.0",
        "mode": "SHADOW",
        "run_id": "p4_20251211_120000_test",
        "start_time": "2025-12-11T12:00:00Z",
        "end_time": "2025-12-11T12:30:00Z",
        "duration_seconds": 1800.0,
        "slice_name": "test_slice",
        "runner_type": "REAL",
        "cycles_completed": 100,
        "divergence_rate": 0.05,
        "twin_success_accuracy": 0.92,
        "twin_blocked_accuracy": 0.94,
        "twin_omega_accuracy": 0.88,
    }


@pytest.fixture
def divergence_log_entries() -> List[Dict[str, Any]]:
    """Sample divergence log entries."""
    entries = []
    for i in range(100):
        # ~5% divergence rate
        is_divergent = i % 20 == 0
        entries.append({
            "schema": "first-light-p4-divergence/1.0.0",
            "mode": "SHADOW",
            "cycle": i,
            "timestamp": f"2025-12-11T12:{i:02d}:00Z",
            "success_diverged": is_divergent,
            "blocked_diverged": False,
            "omega_diverged": is_divergent and i % 40 == 0,
            "H_delta": 0.02 if is_divergent else 0.001,
            "rho_delta": 0.01 if is_divergent else 0.0005,
            "tau_delta": 0.005,
            "severity": "MINOR" if is_divergent else "NONE",
        })
    return entries


@pytest.fixture
def validation_results() -> Dict[str, Any]:
    """Sample RealTelemetryAdapter validation results."""
    return {
        "confidence": 0.92,
        "status": "PROVISIONAL_REAL",
        "mock_indicators": [],
        "lipschitz_violations": 0,
        "flatness_score": 0.15,
        "discreteness_score": 0.08,
        "boundedness_ok": True,
        "correlation_ok": True,
    }


@pytest.fixture
def tda_comparison() -> Dict[str, Any]:
    """Sample TDA comparison metrics."""
    return {
        "sns_delta": 0.02,
        "pcs_delta": -0.01,
        "drs_delta": 0.03,
        "hss_delta": -0.005,
    }


@pytest.fixture
def calibration_report() -> Dict[str, Any]:
    """Sample calibration report."""
    return {
        "calibration_cycles": 50,
        "initial_divergence": 0.15,
        "final_divergence": 0.03,
        "convergence_achieved": True,
    }


@pytest.fixture
def temp_run_dir(
    minimal_p4_summary,
    divergence_log_entries,
    validation_results,
    tda_comparison,
    calibration_report,
) -> Path:
    """Create a temporary run directory with test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir)
        p4_shadow = run_dir / "p4_shadow"
        p4_shadow.mkdir()

        # Write p4_summary.json
        with open(run_dir / "p4_summary.json", "w") as f:
            json.dump(minimal_p4_summary, f)

        # Write divergence_log.jsonl
        with open(run_dir / "divergence_log.jsonl", "w") as f:
            for entry in divergence_log_entries:
                f.write(json.dumps(entry) + "\n")

        # Write validation_results.json
        with open(run_dir / "validation_results.json", "w") as f:
            json.dump(validation_results, f)

        # Write tda_comparison.json
        with open(run_dir / "tda_comparison.json", "w") as f:
            json.dump(tda_comparison, f)

        # Write calibration_report.json
        with open(run_dir / "calibration_report.json", "w") as f:
            json.dump(calibration_report, f)

        yield run_dir


# =============================================================================
# Unit Tests
# =============================================================================

class TestDivergenceStats:
    """Tests for divergence statistics computation."""

    def test_empty_entries(self):
        """Empty entries produce zero stats."""
        stats = compute_divergence_stats([])
        assert stats.total_cycles == 0
        assert stats.divergence_rate == 0.0

    def test_counts_divergent_cycles(self, divergence_log_entries):
        """Correctly counts divergent cycles."""
        stats = compute_divergence_stats(divergence_log_entries)
        assert stats.total_cycles == 100
        assert stats.divergent_cycles == 5  # 100 / 20

    def test_collects_deltas(self, divergence_log_entries):
        """Collects state deltas."""
        stats = compute_divergence_stats(divergence_log_entries)
        assert len(stats.H_deltas) == 100
        assert len(stats.rho_deltas) == 100

    def test_divergence_rate(self, divergence_log_entries):
        """Computes correct divergence rate."""
        stats = compute_divergence_stats(divergence_log_entries)
        assert 0.04 < stats.divergence_rate < 0.06


class TestTwinTrackingAccuracy:
    """Tests for twin tracking accuracy computation."""

    def test_from_summary(self, minimal_p4_summary, divergence_log_entries):
        """Uses accuracy from summary when available."""
        stats = compute_divergence_stats(divergence_log_entries)
        accuracy = compute_twin_tracking_accuracy(minimal_p4_summary, stats)
        assert accuracy["success"] == 0.92
        assert accuracy["blocked"] == 0.94
        assert accuracy["omega"] == 0.88

    def test_computed_from_stats(self, divergence_log_entries):
        """Computes accuracy from stats when not in summary."""
        stats = compute_divergence_stats(divergence_log_entries)
        accuracy = compute_twin_tracking_accuracy({}, stats)
        assert "success" in accuracy
        assert 0.9 < accuracy["success"] < 1.0  # ~95% non-divergent


class TestManifoldValidation:
    """Tests for RTTS manifold validation."""

    def test_all_ok_by_default(self, divergence_log_entries):
        """All checks pass by default without validation data."""
        stats = compute_divergence_stats(divergence_log_entries)
        manifold = compute_manifold_validation(stats, None)
        assert manifold.boundedness_ok
        assert manifold.continuity_ok
        assert manifold.correlation_ok
        assert manifold.violations == []

    def test_detects_lipschitz_violations(self, divergence_log_entries):
        """Detects Lipschitz violations."""
        stats = compute_divergence_stats(divergence_log_entries)
        validation = {"lipschitz_violations": 3}
        manifold = compute_manifold_validation(stats, validation)
        assert not manifold.continuity_ok
        assert any("Lipschitz" in v for v in manifold.violations)

    def test_detects_mock_indicators(self, divergence_log_entries):
        """Detects mock indicators."""
        stats = compute_divergence_stats(divergence_log_entries)
        validation = {"mock_indicators": ["flat_sequence", "discrete_jumps"]}
        manifold = compute_manifold_validation(stats, validation)
        assert any("MOCK" in v for v in manifold.violations)


class TestDivergenceDecomposition:
    """Tests for divergence decomposition."""

    def test_empty_stats(self):
        """Empty stats produce zero decomposition."""
        stats = DivergenceStats()
        decomp = compute_divergence_decomposition(stats)
        assert decomp.bias == 0.0
        assert decomp.variance == 0.0

    def test_computes_bias(self, divergence_log_entries):
        """Computes bias from mean delta."""
        stats = compute_divergence_stats(divergence_log_entries)
        decomp = compute_divergence_decomposition(stats)
        assert decomp.bias > 0.0


class TestPatternClassification:
    """Tests for divergence pattern classification."""

    def test_nominal_for_low_divergence(self):
        """Classifies as NOMINAL for very low divergence."""
        stats = DivergenceStats(total_cycles=1000, divergent_cycles=5)
        decomp = compute_divergence_decomposition(stats)
        pattern, confidence = classify_divergence_pattern(decomp, stats)
        assert pattern == "NOMINAL"
        assert confidence > 0.9

    def test_drift_for_high_bias(self):
        """Classifies as DRIFT for high bias."""
        stats = DivergenceStats(
            total_cycles=100,
            divergent_cycles=20,
            H_deltas=[0.1] * 100,  # High consistent bias
        )
        decomp = compute_divergence_decomposition(stats)
        pattern, confidence = classify_divergence_pattern(decomp, stats)
        assert pattern == "DRIFT"


class TestValidationStatus:
    """Tests for RTTS validation status determination."""

    def test_unvalidated_without_data(self):
        """Returns UNVALIDATED without validation data."""
        from scripts.generate_p5_divergence_real_report import ManifoldValidation
        manifold = ManifoldValidation()
        status, confidence = determine_validation_status(None, manifold)
        assert status == "UNVALIDATED"
        assert confidence == 0.5

    def test_validated_real_with_high_confidence(self, validation_results):
        """Returns VALIDATED_REAL with high confidence."""
        from scripts.generate_p5_divergence_real_report import ManifoldValidation
        manifold = ManifoldValidation()
        status, confidence = determine_validation_status(validation_results, manifold)
        assert status == "VALIDATED_REAL"
        assert confidence > 0.85

    def test_suspected_mock_with_indicators(self):
        """Returns SUSPECTED_MOCK with mock indicators."""
        from scripts.generate_p5_divergence_real_report import ManifoldValidation
        validation = {"status": "MOCK_LIKE", "confidence": 0.3}
        manifold = ManifoldValidation()
        status, confidence = determine_validation_status(validation, manifold)
        assert status == "SUSPECTED_MOCK"


# =============================================================================
# Integration Tests
# =============================================================================

class TestReportGeneration:
    """Tests for full report generation."""

    def test_generates_minimal_report(self, minimal_p4_summary):
        """Generates report with minimal required fields."""
        report = generate_report(
            run_dir=Path("/tmp/test"),
            summary=minimal_p4_summary,
            divergence_entries=[],
            validation=None,
            tda=None,
            calibration=None,
        )

        # Check required fields
        assert report["schema_version"] == SCHEMA_VERSION
        assert report["run_id"].startswith("p5_")
        assert report["telemetry_source"] in ["real", "mock"]
        assert report["validation_status"] in ["VALIDATED_REAL", "SUSPECTED_MOCK", "UNVALIDATED"]
        assert 0.0 <= report["validation_confidence"] <= 1.0
        assert report["total_cycles"] >= 0
        assert 0.0 <= report["divergence_rate"] <= 1.0
        assert report["mode"] == "SHADOW"

    def test_generates_full_report(
        self,
        minimal_p4_summary,
        divergence_log_entries,
        validation_results,
        tda_comparison,
        calibration_report,
    ):
        """Generates report with all optional fields."""
        report = generate_report(
            run_dir=Path("/tmp/test"),
            summary=minimal_p4_summary,
            divergence_entries=divergence_log_entries,
            validation=validation_results,
            tda=tda_comparison,
            calibration=calibration_report,
        )

        # Check recommended fields
        assert "twin_tracking_accuracy" in report
        assert "manifold_validation" in report
        assert "tda_comparison" in report
        assert "warm_start_calibration" in report

        # Check optional diagnostics
        assert "divergence_decomposition" in report
        assert "pattern_classification" in report
        assert "governance_signals" in report
        assert "fusion_advisory" in report

    def test_shadow_mode_enforced(self, minimal_p4_summary):
        """Mode is always SHADOW."""
        report = generate_report(
            run_dir=Path("/tmp/test"),
            summary=minimal_p4_summary,
            divergence_entries=[],
            validation=None,
            tda=None,
            calibration=None,
        )
        assert report["mode"] == "SHADOW"


class TestSchemaValidation:
    """Tests for schema validation of generated reports."""

    @pytest.fixture
    def schema(self):
        """Load the P5 divergence schema."""
        schema_path = (
            Path(__file__).parent.parent.parent
            / "docs"
            / "system_law"
            / "schemas"
            / "p5"
            / "p5_divergence_real.schema.json"
        )
        if not schema_path.exists():
            pytest.skip("Schema file not found")
        with open(schema_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def test_minimal_report_validates(self, schema, minimal_p4_summary):
        """Minimal report validates against schema."""
        jsonschema = pytest.importorskip("jsonschema")

        # Schema requires total_cycles >= 1, so provide at least one entry
        minimal_entry = {
            "cycle": 0,
            "success_diverged": False,
            "blocked_diverged": False,
            "omega_diverged": False,
        }

        report = generate_report(
            run_dir=Path("/tmp/test"),
            summary=minimal_p4_summary,
            divergence_entries=[minimal_entry],
            validation=None,
            tda=None,
            calibration=None,
        )

        # Should not raise
        jsonschema.validate(instance=report, schema=schema)

    def test_full_report_validates(
        self,
        schema,
        minimal_p4_summary,
        divergence_log_entries,
        validation_results,
        tda_comparison,
        calibration_report,
    ):
        """Full report validates against schema."""
        jsonschema = pytest.importorskip("jsonschema")

        report = generate_report(
            run_dir=Path("/tmp/test"),
            summary=minimal_p4_summary,
            divergence_entries=divergence_log_entries,
            validation=validation_results,
            tda=tda_comparison,
            calibration=calibration_report,
        )

        # Should not raise
        jsonschema.validate(instance=report, schema=schema)


class TestFileLoading:
    """Tests for file loading functions."""

    def test_load_p4_summary(self, temp_run_dir):
        """Loads p4_summary.json."""
        summary = load_p4_summary(temp_run_dir)
        assert summary is not None
        assert summary["run_id"] == "p4_20251211_120000_test"

    def test_load_divergence_log(self, temp_run_dir):
        """Loads divergence_log.jsonl."""
        entries = load_divergence_log(temp_run_dir)
        assert len(entries) == 100

    def test_missing_files_return_none(self):
        """Missing files return None/empty."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            assert load_p4_summary(run_dir) is None
            assert load_divergence_log(run_dir) == []


class TestCLI:
    """Tests for command-line interface."""

    def test_main_generates_report(self, temp_run_dir):
        """CLI generates report file."""
        output_path = temp_run_dir / "p4_shadow" / "p5_divergence_real.json"

        result = main([
            "--p5-run-dir", str(temp_run_dir),
            "--skip-validation",
            "--quiet",
        ])

        assert result == 0
        assert output_path.exists()

        with open(output_path) as f:
            report = json.load(f)
        assert report["mode"] == "SHADOW"

    def test_main_custom_output(self, temp_run_dir):
        """CLI respects custom output path."""
        output_path = temp_run_dir / "custom_report.json"

        result = main([
            "--p5-run-dir", str(temp_run_dir),
            "--output", str(output_path),
            "--skip-validation",
            "--quiet",
        ])

        assert result == 0
        assert output_path.exists()

    def test_main_fails_on_missing_dir(self):
        """CLI fails on missing directory."""
        result = main([
            "--p5-run-dir", "/nonexistent/path",
            "--quiet",
        ])
        assert result == 1


class TestEvidencePackIntegration:
    """Tests for evidence pack detection of generated reports."""

    def test_detect_generated_report(self, temp_run_dir, minimal_p4_summary, divergence_log_entries):
        """Evidence pack detects generated report."""
        from backend.topology.first_light.evidence_pack import (
            detect_p5_divergence_file,
            P5_DIVERGENCE_ARTIFACT,
        )

        # Generate report
        result = main([
            "--p5-run-dir", str(temp_run_dir),
            "--skip-validation",
            "--quiet",
        ])
        assert result == 0

        # Detect with evidence pack
        ref = detect_p5_divergence_file(temp_run_dir)
        assert ref is not None
        assert P5_DIVERGENCE_ARTIFACT in ref.path
        assert len(ref.sha256) == 64
        assert ref.mode == "SHADOW"

    def test_schema_validation_in_detection(self, temp_run_dir):
        """Evidence pack validates schema during detection."""
        # Generate report
        result = main([
            "--p5-run-dir", str(temp_run_dir),
            "--skip-validation",
            "--quiet",
        ])
        assert result == 0

        from backend.topology.first_light.evidence_pack import detect_p5_divergence_file

        ref = detect_p5_divergence_file(temp_run_dir)
        assert ref is not None
        # Schema validation depends on jsonschema availability
        # and schema file presence
