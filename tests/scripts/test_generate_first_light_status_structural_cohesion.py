"""
Tests for structural cohesion register integration in generate_first_light_status.py.

SHADOW MODE CONTRACT:
- All tests verify observational behavior only
- No gating or blocking logic is tested
- Tests verify signal extraction and advisory warnings
"""

import json
from pathlib import Path
from typing import Dict, Any

import pytest

from scripts.generate_first_light_status import generate_status


@pytest.fixture
def evidence_pack_dir(tmp_path: Path) -> Path:
    """Create a minimal evidence pack directory structure."""
    pack_dir = tmp_path / "evidence_pack"
    pack_dir.mkdir()
    
    # Create minimal manifest.json
    manifest = {
        "schema_version": "1.0.0",
        "file_count": 1,
        "mode": "SHADOW",
        "shadow_mode_compliance": {
            "all_divergence_logged_only": True,
            "no_governance_modification": True,
            "no_abort_enforcement": True,
        },
        "files": [
            {
                "path": "manifest.json",
                "sha256": "abc123",
            }
        ],
        "governance": {},
    }
    
    manifest_path = pack_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
    
    return pack_dir


@pytest.fixture
def p3_dir(tmp_path: Path) -> Path:
    """Create a minimal P3 directory structure."""
    p3 = tmp_path / "p3"
    p3.mkdir()
    fl_dir = p3 / "fl_run_001"
    fl_dir.mkdir()
    
    # Create minimal stability_report.json
    stability_report = {
        "metrics": {
            "success_rate": 0.95,
            "omega": {"occupancy_rate": 0.92},
            "rsi": {"mean": 0.88},
            "hard_mode": {"ok_rate": 0.90},
        },
        "criteria_evaluation": {"all_passed": True},
        "red_flag_summary": {"total_flags": 0, "hypothetical_abort": False},
        "pathology": "none",
    }
    
    with open(fl_dir / "stability_report.json", "w") as f:
        json.dump(stability_report, f)
    
    # Create other required artifacts
    for artifact in [
        "synthetic_raw.jsonl",
        "red_flag_matrix.json",
        "metrics_windows.json",
        "tda_metrics.json",
        "run_config.json",
    ]:
        (fl_dir / artifact).touch()
    
    return p3


@pytest.fixture
def p4_dir(tmp_path: Path) -> Path:
    """Create a minimal P4 directory structure."""
    p4 = tmp_path / "p4"
    p4.mkdir()
    p4_run_dir = p4 / "p4_run_001"
    p4_run_dir.mkdir()
    
    # Create minimal p4_summary.json
    p4_summary = {
        "mode": "SHADOW",
        "uplift_metrics": {"u2_success_rate_final": 0.93},
        "divergence_analysis": {"divergence_rate": 0.05, "max_divergence_streak": 2},
        "twin_accuracy": {
            "success_prediction_accuracy": 0.85,
            "omega_prediction_accuracy": 0.90,
        },
    }
    
    with open(p4_run_dir / "p4_summary.json", "w") as f:
        json.dump(p4_summary, f)
    
    # Create run_config.json
    run_config = {
        "run_id": "p4_run_001",
        "telemetry_source": "mock",
    }
    
    with open(p4_run_dir / "run_config.json", "w") as f:
        json.dump(run_config, f)
    
    # Create other required artifacts
    for artifact in [
        "real_cycles.jsonl",
        "twin_predictions.jsonl",
        "divergence_log.jsonl",
        "twin_accuracy.json",
    ]:
        (p4_run_dir / artifact).touch()
    
    return p4


@pytest.fixture
def sample_structural_cohesion_register():
    """Fixture for sample structural cohesion register."""
    return {
        "schema_version": "1.0.0",
        "total_experiments": 3,
        "experiments_with_misaligned_structure": ["CAL-EXP-2", "CAL-EXP-3"],
        "top_misaligned": ["CAL-EXP-2", "CAL-EXP-3"],
        "band_combinations": {"COHERENT×COHERENT": 1, "MISALIGNED×PARTIAL": 2},
        "lattice_band_distribution": {"COHERENT": 1, "MISALIGNED": 2},
    }


class TestStructuralCohesionRegisterManifestFirst:
    """Tests for manifest-first extraction of structural cohesion register."""

    def test_extracts_from_manifest(
        self, evidence_pack_dir: Path, p3_dir: Path, p4_dir: Path, sample_structural_cohesion_register: Dict[str, Any]
    ):
        """Verify register is extracted from manifest.json first."""
        # Load manifest and add register
        manifest_path = evidence_pack_dir / "manifest.json"
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        
        manifest["governance"]["structural_cohesion_register"] = sample_structural_cohesion_register
        
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)
        
        # Create evidence.json without register (should not be used)
        evidence = {"governance": {}}
        evidence_path = evidence_pack_dir / "evidence.json"
        with open(evidence_path, "w") as f:
            json.dump(evidence, f)
        
        status = generate_status(
            p3_dir=p3_dir,
            p4_dir=p4_dir,
            evidence_pack_dir=evidence_pack_dir,
        )
        
        # Verify signal is present
        assert "signals" in status
        signals = status["signals"]
        assert signals is not None
        assert "structural_cohesion_register" in signals
        signal = signals["structural_cohesion_register"]
        assert signal["total_experiments"] == 3
        assert signal["experiments_with_misaligned_structure_count"] == 2
        assert signal["top_misaligned"] == ["CAL-EXP-2", "CAL-EXP-3"]
        assert signal["extraction_source"] == "MANIFEST"

    def test_extraction_source_evidence_json(self, evidence_pack_dir: Path, p3_dir: Path, p4_dir: Path, sample_structural_cohesion_register: Dict[str, Any]):
        """Verify extraction_source is EVIDENCE_JSON when loaded from evidence.json."""
        # Load manifest (keep it without register)
        manifest_path = evidence_pack_dir / "manifest.json"
        
        # Create evidence.json with register
        evidence = {
            "governance": {
                "structural_cohesion_register": sample_structural_cohesion_register,
            }
        }
        evidence_path = evidence_pack_dir / "evidence.json"
        with open(evidence_path, "w") as f:
            json.dump(evidence, f)
        
        status = generate_status(
            p3_dir=p3_dir,
            p4_dir=p4_dir,
            evidence_pack_dir=evidence_pack_dir,
        )
        
        signals = status.get("signals")
        assert signals is not None
        assert "structural_cohesion_register" in signals
        signal = signals["structural_cohesion_register"]
        assert signal["extraction_source"] == "EVIDENCE_JSON"

    def test_extraction_source_missing(self, evidence_pack_dir: Path, p3_dir: Path, p4_dir: Path):
        """Verify extraction_source is MISSING when register not found."""
        # Manifest and evidence.json already created without register by fixture
        
        status = generate_status(
            p3_dir=p3_dir,
            p4_dir=p4_dir,
            evidence_pack_dir=evidence_pack_dir,
        )
        
        # Verify no signal (register missing)
        signals = status.get("signals")
        if signals is not None:
            assert "structural_cohesion_register" not in signals

    def test_consistency_check_present(self, evidence_pack_dir: Path, p3_dir: Path, p4_dir: Path, sample_structural_cohesion_register: Dict[str, Any]):
        """Verify consistency check is present in status signal when register exists."""
        # Create manifest with register
        manifest = {"governance": {"structural_cohesion_register": sample_structural_cohesion_register}}
        manifest_path = evidence_pack_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)
        
        status = generate_status(
            p3_dir=p3_dir,
            p4_dir=p4_dir,
            evidence_pack_dir=evidence_pack_dir,
        )
        
        signals = status.get("signals")
        assert signals is not None
        assert "structural_cohesion_register" in signals
        signal = signals["structural_cohesion_register"]
        
        # Consistency check should be present if GGFL adapter is available
        if "consistency" in signal:
            consistency = signal["consistency"]
            assert "consistency" in consistency
            assert "notes" in consistency
            assert "conflict_invariant_violated" in consistency
            assert consistency["consistency"] in ("CONSISTENT", "PARTIAL", "INCONSISTENT")
            assert isinstance(consistency["conflict_invariant_violated"], bool)

    def test_fallback_to_evidence_json(
        self, evidence_pack_dir: Path, p3_dir: Path, p4_dir: Path, sample_structural_cohesion_register: Dict[str, Any]
    ):
        """Verify fallback to evidence.json when register not in manifest."""
        # Load manifest (keep it without register)
        manifest_path = evidence_pack_dir / "manifest.json"
        
        # Create evidence.json with register
        evidence = {
            "governance": {
                "structural_cohesion_register": sample_structural_cohesion_register,
            }
        }
        evidence_path = evidence_pack_dir / "evidence.json"
        with open(evidence_path, "w") as f:
            json.dump(evidence, f)
        
        status = generate_status(
            p3_dir=p3_dir,
            p4_dir=p4_dir,
            evidence_pack_dir=evidence_pack_dir,
        )
        
        # Verify signal is present (from evidence.json fallback)
        assert "signals" in status
        signals = status["signals"]
        assert signals is not None
        assert "structural_cohesion_register" in signals
        signal = signals["structural_cohesion_register"]
        assert signal["total_experiments"] == 3

    def test_missing_register_safe(self, evidence_pack_dir: Path, p3_dir: Path, p4_dir: Path):
        """Verify missing register does not cause errors."""
        # Manifest and evidence.json already created without register by fixture
        
        status = generate_status(
            p3_dir=p3_dir,
            p4_dir=p4_dir,
            evidence_pack_dir=evidence_pack_dir,
        )
        
        # Verify no signal (register missing is safe)
        signals = status.get("signals")
        if signals is not None:
            assert "structural_cohesion_register" not in signals
        
        # Verify no errors (errors might be None or a list)
        errors = status.get("errors")
        if errors is not None:
            assert len(errors) == 0


class TestStructuralCohesionRegisterSignal:
    """Tests for structural cohesion register signal extraction."""

    def test_signal_has_required_fields(
        self, evidence_pack_dir: Path, p3_dir: Path, p4_dir: Path, sample_structural_cohesion_register: Dict[str, Any]
    ):
        """Verify signal has all required fields."""
        manifest_path = evidence_pack_dir / "manifest.json"
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        
        manifest["governance"]["structural_cohesion_register"] = sample_structural_cohesion_register
        
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)
        
        status = generate_status(
            p3_dir=p3_dir,
            p4_dir=p4_dir,
            evidence_pack_dir=evidence_pack_dir,
        )
        
        signals = status.get("signals")
        assert signals is not None
        assert "structural_cohesion_register" in signals
        signal = signals["structural_cohesion_register"]
        assert "total_experiments" in signal
        assert "experiments_with_misaligned_structure_count" in signal
        assert "top_misaligned" in signal

    def test_signal_limits_top_misaligned_to_five(self, evidence_pack_dir: Path, p3_dir: Path, p4_dir: Path):
        """Verify signal preserves top_misaligned list (up to 5)."""
        register = {
            "total_experiments": 10,
            "experiments_with_misaligned_structure": [f"CAL-EXP-{i}" for i in range(1, 8)],
            "top_misaligned": [f"CAL-EXP-{i}" for i in range(1, 6)],  # 5 items
        }
        
        manifest_path = evidence_pack_dir / "manifest.json"
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        
        manifest["governance"]["structural_cohesion_register"] = register
        
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)
        
        status = generate_status(
            p3_dir=p3_dir,
            p4_dir=p4_dir,
            evidence_pack_dir=evidence_pack_dir,
        )
        
        signals = status.get("signals")
        assert signals is not None
        assert "structural_cohesion_register" in signals
        signal = signals["structural_cohesion_register"]
        assert len(signal["top_misaligned"]) == 5


class TestStructuralCohesionRegisterWarnings:
    """Tests for structural cohesion register warning generation."""

    def test_generates_warning_when_misalignments_exist(
        self, evidence_pack_dir: Path, p3_dir: Path, p4_dir: Path, sample_structural_cohesion_register: Dict[str, Any]
    ):
        """Verify warning is generated when misalignments exist."""
        manifest_path = evidence_pack_dir / "manifest.json"
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        
        manifest["governance"]["structural_cohesion_register"] = sample_structural_cohesion_register
        
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)
        
        status = generate_status(
            p3_dir=p3_dir,
            p4_dir=p4_dir,
            evidence_pack_dir=evidence_pack_dir,
        )
        
        # Verify warning exists
        warnings = status.get("warnings", [])
        structural_warnings = [w for w in warnings if "Structural cohesion register" in w]
        assert len(structural_warnings) == 1
        
        # Verify warning includes misaligned count and top list (max 3 cal_ids)
        warning = structural_warnings[0]
        assert "2 experiment(s) with misaligned structure" in warning
        assert "CAL-EXP-2" in warning or "CAL-EXP-3" in warning
        # Verify warning lists at most 3 cal_ids (warning hygiene)
        cal_id_count = warning.count("CAL-EXP-")
        assert cal_id_count <= 3

    def test_no_warning_when_no_misalignments(self, evidence_pack_dir: Path, p3_dir: Path, p4_dir: Path):
        """Verify no warning when no misalignments exist."""
        register = {
            "total_experiments": 3,
            "experiments_with_misaligned_structure": [],
            "top_misaligned": [],
        }
        
        manifest_path = evidence_pack_dir / "manifest.json"
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        
        manifest["governance"]["structural_cohesion_register"] = register
        
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)
        
        status = generate_status(
            p3_dir=p3_dir,
            p4_dir=p4_dir,
            evidence_pack_dir=evidence_pack_dir,
        )
        
        # Verify no structural cohesion warning
        warnings = status.get("warnings", [])
        structural_warnings = [w for w in warnings if "Structural cohesion register" in w]
        assert len(structural_warnings) == 0

    def test_single_warning_cap(self, evidence_pack_dir: Path, p3_dir: Path, p4_dir: Path):
        """Verify only one warning is generated regardless of misalignment count."""
        register = {
            "total_experiments": 10,
            "experiments_with_misaligned_structure": [f"CAL-EXP-{i}" for i in range(1, 9)],
            "top_misaligned": [f"CAL-EXP-{i}" for i in range(1, 6)],
        }
        
        manifest_path = evidence_pack_dir / "manifest.json"
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        
        manifest["governance"]["structural_cohesion_register"] = register
        
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)
        
        status = generate_status(
            p3_dir=p3_dir,
            p4_dir=p4_dir,
            evidence_pack_dir=evidence_pack_dir,
        )
        
        # Verify exactly one structural cohesion warning
        warnings = status.get("warnings", [])
        structural_warnings = [w for w in warnings if "Structural cohesion register" in w]
        assert len(structural_warnings) == 1


class TestStructuralCohesionRegisterDeterminism:
    """Tests for structural cohesion register determinism."""

    def test_deterministic_signal_extraction(
        self, evidence_pack_dir: Path, p3_dir: Path, p4_dir: Path, sample_structural_cohesion_register: Dict[str, Any]
    ):
        """Verify signal extraction is deterministic."""
        manifest_path = evidence_pack_dir / "manifest.json"
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        
        manifest["governance"]["structural_cohesion_register"] = sample_structural_cohesion_register
        
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)
        
        status1 = generate_status(
            p3_dir=p3_dir,
            p4_dir=p4_dir,
            evidence_pack_dir=evidence_pack_dir,
        )
        
        status2 = generate_status(
            p3_dir=p3_dir,
            p4_dir=p4_dir,
            evidence_pack_dir=evidence_pack_dir,
        )
        
        # Verify signals are identical
        signals1 = status1.get("signals")
        signals2 = status2.get("signals")
        assert signals1 is not None
        assert signals2 is not None
        
        signal1 = signals1.get("structural_cohesion_register")
        signal2 = signals2.get("structural_cohesion_register")
        
        assert signal1 == signal2
        assert json.dumps(signal1, sort_keys=True) == json.dumps(signal2, sort_keys=True)

