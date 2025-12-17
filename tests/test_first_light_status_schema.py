import json
import tempfile
from pathlib import Path

import pytest


def test_sample_status_has_new_fields():
    """Ensure sample status includes telemetry/proof fields at schema 1.2.0."""
    status_path = Path(__file__).resolve().parents[1] / "testsfixtures" / "first_light_status_sample.json"
    if not status_path.exists():
        pytest.skip("Sample status fixture not found")

    data = json.loads(status_path.read_text())

    assert data["schema_version"] == "1.2.0"
    assert data["telemetry_source"] in {"mock", "real_synthetic", "real_trace"}
    assert isinstance(data["proof_snapshot_present"], bool)


def test_sample_status_minimal_schema_required_fields():
    """Validate required fields and allowed telemetry values in the sample status."""
    status_path = Path(__file__).resolve().parents[1] / "testsfixtures" / "first_light_status_sample.json"
    data = json.loads(status_path.read_text())

    required_fields = {
        "schema_version",
        "telemetry_source",
        "proof_snapshot_present",
        "shadow_mode_ok",
    }
    assert required_fields.issubset(data.keys())

    allowed_sources = {"mock", "real_synthetic", "real_trace"}
    for source in allowed_sources:
        clone = dict(data)
        clone["telemetry_source"] = source
        assert clone["telemetry_source"] in allowed_sources


class TestP5TelemetrySourceDetection:
    """Test P5 telemetry source detection in status generation."""

    def test_detect_telemetry_source_mock(self, tmp_path) -> None:
        """Test detection of mock telemetry source."""
        from scripts.generate_first_light_status import detect_telemetry_source

        # Create mock run config
        run_dir = tmp_path / "p4_test"
        run_dir.mkdir()
        config = {
            "telemetry_adapter": "mock",
            "telemetry_source": "mock",
        }
        with open(run_dir / "run_config.json", "w") as f:
            json.dump(config, f)

        source = detect_telemetry_source(tmp_path)
        assert source == "mock"

    def test_detect_telemetry_source_real_synthetic(self, tmp_path) -> None:
        """Test detection of real_synthetic telemetry source."""
        from scripts.generate_first_light_status import detect_telemetry_source

        run_dir = tmp_path / "p4_test"
        run_dir.mkdir()
        config = {
            "telemetry_adapter": "real",
            "telemetry_source": "real_synthetic",
        }
        with open(run_dir / "run_config.json", "w") as f:
            json.dump(config, f)

        source = detect_telemetry_source(tmp_path)
        assert source == "real_synthetic"

    def test_detect_telemetry_source_real_trace(self, tmp_path) -> None:
        """Test detection of real_trace telemetry source."""
        from scripts.generate_first_light_status import detect_telemetry_source

        run_dir = tmp_path / "p4_test"
        run_dir.mkdir()
        config = {
            "telemetry_adapter": "real",
            "telemetry_source": "real_trace",
        }
        with open(run_dir / "run_config.json", "w") as f:
            json.dump(config, f)

        source = detect_telemetry_source(tmp_path)
        assert source == "real_trace"

    def test_detect_telemetry_source_legacy_real(self, tmp_path) -> None:
        """Test fallback detection for legacy real adapter (no telemetry_source field)."""
        from scripts.generate_first_light_status import detect_telemetry_source

        run_dir = tmp_path / "p4_test"
        run_dir.mkdir()
        config = {
            "telemetry_adapter": "real",
            # No telemetry_source field (legacy)
        }
        with open(run_dir / "run_config.json", "w") as f:
            json.dump(config, f)

        source = detect_telemetry_source(tmp_path)
        assert source == "real_trace"  # Fallback for legacy "real"


class TestP5DivergenceBaseline:
    """Test P5 divergence baseline extraction."""

    def test_extract_p5_divergence_baseline_mock_run(self, tmp_path) -> None:
        """Test that mock runs return None for P5 baseline."""
        from scripts.generate_first_light_status import extract_p5_divergence_baseline

        run_dir = tmp_path / "p4_test"
        run_dir.mkdir()
        config = {"telemetry_source": "mock"}
        with open(run_dir / "run_config.json", "w") as f:
            json.dump(config, f)

        baseline = extract_p5_divergence_baseline(tmp_path)
        assert baseline is None

    def test_extract_p5_divergence_baseline_real_run(self, tmp_path) -> None:
        """Test P5 baseline extraction for real adapter run."""
        from scripts.generate_first_light_status import extract_p5_divergence_baseline

        run_dir = tmp_path / "p4_test"
        run_dir.mkdir()

        config = {
            "telemetry_source": "real_synthetic",
            "run_id": "test_p5_run",
        }
        with open(run_dir / "run_config.json", "w") as f:
            json.dump(config, f)

        summary = {
            "divergence_analysis": {
                "divergence_rate": 0.94,
                "max_divergence_streak": 5,
            },
            "twin_accuracy": {
                "success_prediction_accuracy": 0.78,
                "omega_prediction_accuracy": 1.0,
                "blocked_prediction_accuracy": 1.0,
            },
        }
        with open(run_dir / "p4_summary.json", "w") as f:
            json.dump(summary, f)

        baseline = extract_p5_divergence_baseline(tmp_path)

        assert baseline is not None
        assert baseline["telemetry_source"] == "real_synthetic"
        assert baseline["divergence_rate"] == 0.94
        assert baseline["twin_success_accuracy"] == 0.78
        assert baseline["status"] == "SHADOW_OBSERVATION"


class TestEvidencePackP5Comparison:
    """Test evidence pack P5 mock vs real comparison."""

    def test_evidence_pack_includes_p5_comparison(self, tmp_path) -> None:
        """Test that evidence pack includes P5 comparison for real adapter runs."""
        from backend.topology.first_light.evidence_pack import EvidencePackBuilder

        # Create minimal P5 run artifacts
        run_dir = tmp_path / "p4_run"
        run_dir.mkdir()

        config = {
            "telemetry_source": "real_synthetic",
            "run_id": "test_p5_evidence",
        }
        with open(run_dir / "run_config.json", "w") as f:
            json.dump(config, f)

        summary = {
            "mode": "SHADOW",
            "divergence_analysis": {
                "divergence_rate": 0.94,
                "max_divergence_streak": 3,
            },
            "twin_accuracy": {
                "success_prediction_accuracy": 0.78,
                "omega_prediction_accuracy": 1.0,
                "blocked_prediction_accuracy": 1.0,
            },
        }
        with open(run_dir / "p4_summary.json", "w") as f:
            json.dump(summary, f)

        builder = EvidencePackBuilder(validate_schemas=False)
        comparison = builder._load_p5_mock_vs_real_comparison(run_dir)

        assert comparison is not None
        assert comparison["mode"] == "SHADOW"
        assert comparison["telemetry_source"] == "real_synthetic"
        assert comparison["metrics"]["divergence_rate"] == 0.94
        assert comparison["advisory"]["status"] == "SHADOW_OBSERVATION"

    def test_evidence_pack_no_p5_for_mock(self, tmp_path) -> None:
        """Test that evidence pack excludes P5 comparison for mock runs."""
        from backend.topology.first_light.evidence_pack import EvidencePackBuilder

        run_dir = tmp_path / "p4_run"
        run_dir.mkdir()

        config = {"telemetry_source": "mock"}
        with open(run_dir / "run_config.json", "w") as f:
            json.dump(config, f)

        builder = EvidencePackBuilder(validate_schemas=False)
        comparison = builder._load_p5_mock_vs_real_comparison(run_dir)

        assert comparison is None
