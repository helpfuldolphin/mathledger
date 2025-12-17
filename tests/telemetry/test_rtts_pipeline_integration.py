"""
RTTS Pipeline Integration Tests

Phase X P5.2: Tests for RTTS validation pipeline integration.

Tests the pipeline from P4 harness emission to P5 divergence real generator consumption:
1. P4 harness emits rtts_validation.json when --emit-rtts-validation is set
2. P5 divergence generator loads and integrates rtts_validation.json
3. MOCK-* codes propagate deterministically through the pipeline

SHADOW MODE CONTRACT:
- All tests verify OBSERVATIONAL outputs only
- No gating or enforcement is tested
- mode="SHADOW" and action="LOGGED_ONLY" are always verified

Test Count: 14 tests
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_run_dir():
    """Create a temporary run directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_real_cycles_jsonl(temp_run_dir: Path) -> Path:
    """Create mock real_cycles.jsonl with healthy telemetry data."""
    cycles_path = temp_run_dir / "real_cycles.jsonl"

    # Generate 50 cycles of healthy telemetry
    # Use field names that map to TelemetrySnapshot
    with open(cycles_path, "w", encoding="utf-8") as f:
        for i in range(50):
            entry = {
                "cycle": i,
                "H": 0.5 + 0.1 * (i % 5 - 2) / 10,  # Small variations
                "rho": 0.6 + 0.05 * (i % 7 - 3) / 10,
                "tau": 0.2 + 0.01 * (i % 3 - 1) / 10,
                "beta": 0.5 + 0.08 * (i % 4 - 2) / 10,
                "in_omega": True,  # TelemetrySnapshot uses in_omega
                "success": True,
                "real_blocked": False,  # TelemetrySnapshot uses real_blocked
                "hard_ok": True,
                "timestamp": f"2025-01-01T00:00:{i:02d}Z",
            }
            f.write(json.dumps(entry) + "\n")

    return cycles_path


@pytest.fixture
def mock_p4_summary(temp_run_dir: Path) -> Dict[str, Any]:
    """Create mock p4_summary.json."""
    summary = {
        "run_id": "p4_20250101_000000",
        "cycles_completed": 50,
        "divergence_rate": 0.1,
        "mode": "SHADOW",
    }
    summary_path = temp_run_dir / "p4_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f)
    return summary


@pytest.fixture
def mock_divergence_log(temp_run_dir: Path) -> Path:
    """Create mock divergence_log.jsonl."""
    log_path = temp_run_dir / "divergence_log.jsonl"

    with open(log_path, "w", encoding="utf-8") as f:
        for i in range(50):
            entry = {
                "cycle": i,
                "success_diverged": i % 10 == 5,  # 10% divergence
                "blocked_diverged": False,
                "omega_diverged": False,
                "severity": "NONE" if i % 10 != 5 else "MINOR",
            }
            f.write(json.dumps(entry) + "\n")

    return log_path


# =============================================================================
# P4 Harness Emission Tests (7 tests)
# =============================================================================

class TestP4HarnessEmission:
    """Tests for P4 harness rtts_validation.json emission."""

    def test_save_rtts_validation_creates_file(
        self, temp_run_dir: Path, mock_real_cycles_jsonl: Path
    ):
        """Test that save_rtts_validation creates rtts_validation.json."""
        from scripts.usla_first_light_p4_harness import save_rtts_validation

        result = save_rtts_validation(
            output_dir=temp_run_dir,
            real_cycles_path=mock_real_cycles_jsonl,
            run_id="p4_test_001",
            cycle_count=50,
        )

        assert result is not None
        assert result.exists()
        assert result.name == "rtts_validation.json"

    def test_save_rtts_validation_schema_version(
        self, temp_run_dir: Path, mock_real_cycles_jsonl: Path
    ):
        """Test that emitted file has correct schema_version."""
        from scripts.usla_first_light_p4_harness import save_rtts_validation

        save_rtts_validation(
            output_dir=temp_run_dir,
            real_cycles_path=mock_real_cycles_jsonl,
            run_id="p4_test_002",
            cycle_count=50,
        )

        with open(temp_run_dir / "rtts_validation.json") as f:
            data = json.load(f)

        assert data["schema_version"] == "1.0.0"

    def test_save_rtts_validation_shadow_mode(
        self, temp_run_dir: Path, mock_real_cycles_jsonl: Path
    ):
        """Test that emitted file has SHADOW mode markers."""
        from scripts.usla_first_light_p4_harness import save_rtts_validation

        save_rtts_validation(
            output_dir=temp_run_dir,
            real_cycles_path=mock_real_cycles_jsonl,
            run_id="p4_test_003",
            cycle_count=50,
        )

        with open(temp_run_dir / "rtts_validation.json") as f:
            data = json.load(f)

        assert data["mode"] == "SHADOW"
        assert data["action"] == "LOGGED_ONLY"

    def test_save_rtts_validation_contains_mock_flags(
        self, temp_run_dir: Path, mock_real_cycles_jsonl: Path
    ):
        """Test that emitted file contains mock_detection_flags array."""
        from scripts.usla_first_light_p4_harness import save_rtts_validation

        save_rtts_validation(
            output_dir=temp_run_dir,
            real_cycles_path=mock_real_cycles_jsonl,
            run_id="p4_test_004",
            cycle_count=50,
        )

        with open(temp_run_dir / "rtts_validation.json") as f:
            data = json.load(f)

        assert "mock_detection_flags" in data
        assert isinstance(data["mock_detection_flags"], list)

    def test_save_rtts_validation_contains_window_metadata(
        self, temp_run_dir: Path, mock_real_cycles_jsonl: Path
    ):
        """Test that emitted file contains window metadata."""
        from scripts.usla_first_light_p4_harness import save_rtts_validation

        save_rtts_validation(
            output_dir=temp_run_dir,
            real_cycles_path=mock_real_cycles_jsonl,
            run_id="p4_test_005",
            cycle_count=50,
        )

        with open(temp_run_dir / "rtts_validation.json") as f:
            data = json.load(f)

        assert "window" in data
        assert "size" in data["window"]
        assert "total_cycles" in data["window"]
        assert data["window"]["total_cycles"] == 50

    def test_save_rtts_validation_contains_overall_status(
        self, temp_run_dir: Path, mock_real_cycles_jsonl: Path
    ):
        """Test that emitted file contains overall_status."""
        from scripts.usla_first_light_p4_harness import save_rtts_validation

        save_rtts_validation(
            output_dir=temp_run_dir,
            real_cycles_path=mock_real_cycles_jsonl,
            run_id="p4_test_006",
            cycle_count=50,
        )

        with open(temp_run_dir / "rtts_validation.json") as f:
            data = json.load(f)

        assert "overall_status" in data
        assert data["overall_status"] in ("OK", "ATTENTION", "WARN", "CRITICAL", "UNKNOWN")

    def test_save_rtts_validation_returns_none_for_missing_cycles(
        self, temp_run_dir: Path
    ):
        """Test that save_rtts_validation returns None when real_cycles.jsonl missing."""
        from scripts.usla_first_light_p4_harness import save_rtts_validation

        result = save_rtts_validation(
            output_dir=temp_run_dir,
            real_cycles_path=temp_run_dir / "nonexistent.jsonl",
            run_id="p4_test_007",
            cycle_count=50,
        )

        assert result is None


# =============================================================================
# P5 Generator Consumption Tests (5 tests)
# =============================================================================

class TestP5GeneratorConsumption:
    """Tests for P5 divergence real generator rtts_validation.json consumption."""

    def test_load_rtts_validation_finds_file(self, temp_run_dir: Path):
        """Test that load_rtts_validation finds rtts_validation.json."""
        from scripts.generate_p5_divergence_real_report import load_rtts_validation

        # Create rtts_validation.json
        validation = {
            "schema_version": "1.0.0",
            "mode": "SHADOW",
            "mock_detection_flags": ["MOCK-001", "MOCK-003"],
        }
        with open(temp_run_dir / "rtts_validation.json", "w") as f:
            json.dump(validation, f)

        result = load_rtts_validation(temp_run_dir)

        assert result is not None
        assert result["schema_version"] == "1.0.0"

    def test_load_rtts_validation_returns_none_when_missing(self, temp_run_dir: Path):
        """Test that load_rtts_validation returns None when file missing."""
        from scripts.generate_p5_divergence_real_report import load_rtts_validation

        result = load_rtts_validation(temp_run_dir)

        assert result is None

    def test_extract_mock_flags_prefers_rtts_validation(self, temp_run_dir: Path):
        """Test that extract_mock_detection_flags prefers rtts_validation.json."""
        from scripts.generate_p5_divergence_real_report import (
            extract_mock_detection_flags,
            ManifoldValidation,
        )

        rtts_validation = {
            "mock_detection_flags": ["MOCK-001", "MOCK-003", "MOCK-005"],
        }
        manifold = ManifoldValidation()
        validation = {"mock_indicators": ["indicator1", "indicator2"]}

        flags = extract_mock_detection_flags(validation, manifold, rtts_validation)

        assert flags == ["MOCK-001", "MOCK-003", "MOCK-005"]

    def test_extract_mock_flags_falls_back_to_legacy(self, temp_run_dir: Path):
        """Test fallback to legacy validation when rtts_validation missing."""
        from scripts.generate_p5_divergence_real_report import (
            extract_mock_detection_flags,
            ManifoldValidation,
        )

        manifold = ManifoldValidation(violations=["MOCK-002: test violation"])
        validation = None

        flags = extract_mock_detection_flags(validation, manifold, None)

        assert "MOCK-002" in flags

    def test_generate_report_includes_rtts_validation(
        self, temp_run_dir: Path, mock_p4_summary: Dict[str, Any], mock_divergence_log: Path
    ):
        """Test that generate_report includes rtts_validation when available."""
        from scripts.generate_p5_divergence_real_report import generate_report

        # Create rtts_validation
        rtts_validation = {
            "schema_version": "1.0.0",
            "overall_status": "OK",
            "validation_passed": True,
            "warning_count": 0,
            "window": {"size": 50, "total_cycles": 50},
        }

        # Load divergence entries
        entries = []
        with open(mock_divergence_log) as f:
            for line in f:
                entries.append(json.loads(line))

        report = generate_report(
            run_dir=temp_run_dir,
            summary=mock_p4_summary,
            divergence_entries=entries,
            validation=None,
            tda=None,
            calibration=None,
            rtts_validation=rtts_validation,
        )

        assert "rtts_validation" in report
        assert report["rtts_validation"]["overall_status"] == "OK"
        assert report["rtts_validation"]["source"] == "rtts_validation.json"


# =============================================================================
# MOCK-* Code Propagation Tests (2 tests)
# =============================================================================

class TestMockCodePropagation:
    """Tests for MOCK-* code propagation through the pipeline."""

    def test_mock_codes_propagate_from_harness_to_generator(
        self, temp_run_dir: Path, mock_real_cycles_jsonl: Path, mock_p4_summary: Dict[str, Any]
    ):
        """Test that MOCK-* codes propagate from harness emission to generator."""
        from scripts.usla_first_light_p4_harness import save_rtts_validation
        from scripts.generate_p5_divergence_real_report import (
            load_rtts_validation,
            extract_mock_detection_flags,
            ManifoldValidation,
        )

        # Step 1: Emit from harness
        save_rtts_validation(
            output_dir=temp_run_dir,
            real_cycles_path=mock_real_cycles_jsonl,
            run_id="p4_propagation_test",
            cycle_count=50,
        )

        # Step 2: Load in generator
        rtts_validation = load_rtts_validation(temp_run_dir)
        assert rtts_validation is not None

        # Step 3: Extract flags
        manifold = ManifoldValidation()
        flags = extract_mock_detection_flags(None, manifold, rtts_validation)

        # Verify flags are in MOCK-NNN format
        for flag in flags:
            assert flag.startswith("MOCK-")
            assert len(flag) == 8  # "MOCK-NNN"

    def test_mock_codes_deterministic(
        self, temp_run_dir: Path, mock_real_cycles_jsonl: Path
    ):
        """Test that MOCK-* codes are deterministic for same input."""
        from scripts.usla_first_light_p4_harness import save_rtts_validation

        # Run twice with same input
        result1 = save_rtts_validation(
            output_dir=temp_run_dir,
            real_cycles_path=mock_real_cycles_jsonl,
            run_id="p4_determinism_1",
            cycle_count=50,
        )
        with open(result1) as f:
            data1 = json.load(f)
        flags1 = data1["mock_detection_flags"]

        # Create fresh temp dir for second run
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir2:
            tmpdir2_path = Path(tmpdir2)

            # Copy cycles file
            import shutil
            cycles2 = tmpdir2_path / "real_cycles.jsonl"
            shutil.copy(mock_real_cycles_jsonl, cycles2)

            result2 = save_rtts_validation(
                output_dir=tmpdir2_path,
                real_cycles_path=cycles2,
                run_id="p4_determinism_2",
                cycle_count=50,
            )
            with open(result2) as f:
                data2 = json.load(f)
            flags2 = data2["mock_detection_flags"]

        # Flags should be identical
        assert sorted(flags1) == sorted(flags2)


# =============================================================================
# Run all tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
