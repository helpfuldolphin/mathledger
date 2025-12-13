"""Tests for CAL-EXP Structural Cohesion Register.

STATUS: PHASE X — CAL-EXP STRUCTURAL COHESION REGISTER

Tests verify:
- Per-experiment annex emission and persistence
- Structural cohesion register aggregation
- Evidence attachment
- Classification logic (misaligned structure detection)
- JSON safety and determinism
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest

from backend.health.atlas_governance_adapter import (
    attach_structural_cohesion_register_signal_to_evidence,
    build_first_light_structural_cohesion_annex,
    build_structural_cohesion_register,
    emit_cal_exp_structural_cohesion_annex,
    attach_structural_cohesion_register_to_evidence,
    extract_structural_cohesion_register_signal,
)


def sample_structural_annex() -> Dict[str, Any]:
    """Create sample structural cohesion annex for testing."""
    return {
        "schema_version": "1.0.0",
        "lattice_band": "COHERENT",
        "transition_status": "OK",
        "lean_shadow_status": "OK",
        "coherence_band": "COHERENT",
    }


@pytest.fixture
def sample_annex():
    """Fixture for sample structural cohesion annex."""
    return sample_structural_annex()


@pytest.fixture
def temp_calibration_dir():
    """Fixture for temporary calibration directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestEmitCalExpStructuralCohesionAnnex:
    """Tests for emit_cal_exp_structural_cohesion_annex."""

    def test_has_required_keys(self, sample_annex, temp_calibration_dir):
        """Verify emitted annex has all required keys."""
        emitted = emit_cal_exp_structural_cohesion_annex(
            "CAL-EXP-1", sample_annex, output_dir=temp_calibration_dir
        )
        
        assert "schema_version" in emitted
        assert "cal_id" in emitted
        assert "lattice_band" in emitted
        assert "transition_status" in emitted
        assert "lean_shadow_status" in emitted
        assert "coherence_band" in emitted

    def test_persists_to_file(self, sample_annex, temp_calibration_dir):
        """Verify annex is persisted to JSON file."""
        cal_id = "CAL-EXP-1"
        emitted = emit_cal_exp_structural_cohesion_annex(
            cal_id, sample_annex, output_dir=temp_calibration_dir
        )
        
        # Check file exists
        filepath = Path(temp_calibration_dir) / f"structural_cohesion_annex_{cal_id}.json"
        assert filepath.exists()
        
        # Verify file contents
        with open(filepath) as f:
            persisted = json.load(f)
        
        assert persisted["cal_id"] == cal_id
        assert persisted["lattice_band"] == "COHERENT"

    def test_handles_missing_optional_fields(self, temp_calibration_dir):
        """Verify function handles annex with missing optional fields."""
        annex = {
            "lattice_band": "COHERENT",
            "transition_status": "OK",
            # Missing lean_shadow_status and coherence_band
        }
        
        emitted = emit_cal_exp_structural_cohesion_annex(
            "CAL-EXP-1", annex, output_dir=temp_calibration_dir
        )
        
        assert emitted["lean_shadow_status"] is None
        assert emitted["coherence_band"] is None

    def test_creates_output_directory(self, sample_annex):
        """Verify function creates output directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as base_dir:
            output_dir = Path(base_dir) / "new_calibration_dir"
            
            emitted = emit_cal_exp_structural_cohesion_annex(
                "CAL-EXP-1", sample_annex, output_dir=str(output_dir)
            )
            
            assert output_dir.exists()
            filepath = output_dir / "structural_cohesion_annex_CAL-EXP-1.json"
            assert filepath.exists()

    def test_deterministic(self, sample_annex, temp_calibration_dir):
        """Verify function is deterministic."""
        result1 = emit_cal_exp_structural_cohesion_annex(
            "CAL-EXP-1", sample_annex, output_dir=temp_calibration_dir
        )
        result2 = emit_cal_exp_structural_cohesion_annex(
            "CAL-EXP-1", sample_annex, output_dir=temp_calibration_dir
        )
        
        assert json.dumps(result1, sort_keys=True) == json.dumps(result2, sort_keys=True)

    def test_json_safe(self, sample_annex, temp_calibration_dir):
        """Verify emitted annex is JSON serializable."""
        emitted = emit_cal_exp_structural_cohesion_annex(
            "CAL-EXP-1", sample_annex, output_dir=temp_calibration_dir
        )
        
        # Should not raise
        json_str = json.dumps(emitted, sort_keys=True)
        assert len(json_str) > 0
        
        # Should round-trip
        parsed = json.loads(json_str)
        assert parsed["cal_id"] == "CAL-EXP-1"

    def test_non_mutating(self, sample_annex, temp_calibration_dir):
        """Verify function does not mutate input annex."""
        annex_copy = dict(sample_annex)
        
        emit_cal_exp_structural_cohesion_annex(
            "CAL-EXP-1", sample_annex, output_dir=temp_calibration_dir
        )
        
        assert sample_annex == annex_copy


class TestBuildStructuralCohesionRegister:
    """Tests for build_structural_cohesion_register."""

    def test_has_required_keys(self):
        """Verify register has all required keys."""
        annexes = [
            {
                "schema_version": "1.0.0",
                "cal_id": "CAL-EXP-1",
                "lattice_band": "COHERENT",
                "transition_status": "OK",
                "lean_shadow_status": "OK",
                "coherence_band": "COHERENT",
            }
        ]
        
        register = build_structural_cohesion_register(annexes)
        
        assert "schema_version" in register
        assert "total_experiments" in register
        assert "band_combinations" in register
        assert "experiments_with_misaligned_structure" in register
        assert "lattice_band_distribution" in register
        assert "transition_status_distribution" in register
        assert "lean_shadow_status_distribution" in register
        assert "coherence_band_distribution" in register

    def test_counts_band_combinations(self):
        """Verify register counts lattice × coherence band combinations."""
        annexes = [
            {
                "cal_id": "CAL-EXP-1",
                "lattice_band": "COHERENT",
                "transition_status": "OK",
                "coherence_band": "COHERENT",
            },
            {
                "cal_id": "CAL-EXP-2",
                "lattice_band": "COHERENT",
                "transition_status": "OK",
                "coherence_band": "COHERENT",
            },
            {
                "cal_id": "CAL-EXP-3",
                "lattice_band": "PARTIAL",
                "transition_status": "ATTENTION",
                "coherence_band": "PARTIAL",
            },
        ]
        
        register = build_structural_cohesion_register(annexes)
        
        assert register["band_combinations"]["COHERENT×COHERENT"] == 2
        assert register["band_combinations"]["PARTIAL×PARTIAL"] == 1

    def test_identifies_misaligned_lattice(self):
        """Verify register identifies experiments with MISALIGNED lattice."""
        annexes = [
            {
                "cal_id": "CAL-EXP-1",
                "lattice_band": "MISALIGNED",
                "transition_status": "BLOCK",
                "coherence_band": "PARTIAL",
            },
            {
                "cal_id": "CAL-EXP-2",
                "lattice_band": "COHERENT",
                "transition_status": "OK",
                "coherence_band": "COHERENT",
            },
        ]
        
        register = build_structural_cohesion_register(annexes)
        
        assert "CAL-EXP-1" in register["experiments_with_misaligned_structure"]
        assert "CAL-EXP-2" not in register["experiments_with_misaligned_structure"]

    def test_identifies_misaligned_lean_shadow(self):
        """Verify register identifies experiments with BLOCK lean shadow status."""
        annexes = [
            {
                "cal_id": "CAL-EXP-1",
                "lattice_band": "COHERENT",
                "transition_status": "OK",
                "lean_shadow_status": "BLOCK",
                "coherence_band": "COHERENT",
            },
            {
                "cal_id": "CAL-EXP-2",
                "lattice_band": "COHERENT",
                "transition_status": "OK",
                "lean_shadow_status": "OK",
                "coherence_band": "COHERENT",
            },
        ]
        
        register = build_structural_cohesion_register(annexes)
        
        assert "CAL-EXP-1" in register["experiments_with_misaligned_structure"]
        assert "CAL-EXP-2" not in register["experiments_with_misaligned_structure"]

    def test_identifies_misaligned_coherence(self):
        """Verify register identifies experiments with MISALIGNED coherence."""
        annexes = [
            {
                "cal_id": "CAL-EXP-1",
                "lattice_band": "COHERENT",
                "transition_status": "OK",
                "coherence_band": "MISALIGNED",
            },
            {
                "cal_id": "CAL-EXP-2",
                "lattice_band": "COHERENT",
                "transition_status": "OK",
                "coherence_band": "COHERENT",
            },
        ]
        
        register = build_structural_cohesion_register(annexes)
        
        assert "CAL-EXP-1" in register["experiments_with_misaligned_structure"]
        assert "CAL-EXP-2" not in register["experiments_with_misaligned_structure"]

    def test_counts_distributions(self):
        """Verify register counts distributions correctly."""
        annexes = [
            {
                "cal_id": "CAL-EXP-1",
                "lattice_band": "COHERENT",
                "transition_status": "OK",
                "lean_shadow_status": "OK",
                "coherence_band": "COHERENT",
            },
            {
                "cal_id": "CAL-EXP-2",
                "lattice_band": "PARTIAL",
                "transition_status": "ATTENTION",
                "lean_shadow_status": "WARN",
                "coherence_band": "PARTIAL",
            },
            {
                "cal_id": "CAL-EXP-3",
                "lattice_band": "COHERENT",
                "transition_status": "OK",
                "lean_shadow_status": "OK",
                "coherence_band": "COHERENT",
            },
        ]
        
        register = build_structural_cohesion_register(annexes)
        
        assert register["lattice_band_distribution"]["COHERENT"] == 2
        assert register["lattice_band_distribution"]["PARTIAL"] == 1
        assert register["transition_status_distribution"]["OK"] == 2
        assert register["transition_status_distribution"]["ATTENTION"] == 1
        assert register["lean_shadow_status_distribution"]["OK"] == 2
        assert register["lean_shadow_status_distribution"]["WARN"] == 1
        assert register["coherence_band_distribution"]["COHERENT"] == 2
        assert register["coherence_band_distribution"]["PARTIAL"] == 1

    def test_handles_empty_annexes(self):
        """Verify register handles empty annex list."""
        register = build_structural_cohesion_register([])
        
        assert register["total_experiments"] == 0
        assert len(register["experiments_with_misaligned_structure"]) == 0
        assert len(register["band_combinations"]) == 0

    def test_handles_missing_optional_fields(self):
        """Verify register handles annexes with missing optional fields."""
        annexes = [
            {
                "cal_id": "CAL-EXP-1",
                "lattice_band": "COHERENT",
                "transition_status": "OK",
                # Missing lean_shadow_status and coherence_band
            }
        ]
        
        register = build_structural_cohesion_register(annexes)
        
        assert register["total_experiments"] == 1
        assert len(register["experiments_with_misaligned_structure"]) == 0
        assert len(register["band_combinations"]) == 0

    def test_deterministic(self):
        """Verify register is deterministic."""
        annexes = [
            {
                "cal_id": "CAL-EXP-1",
                "lattice_band": "COHERENT",
                "transition_status": "OK",
                "coherence_band": "COHERENT",
            },
            {
                "cal_id": "CAL-EXP-2",
                "lattice_band": "PARTIAL",
                "transition_status": "ATTENTION",
                "coherence_band": "PARTIAL",
            },
        ]
        
        result1 = build_structural_cohesion_register(annexes)
        result2 = build_structural_cohesion_register(annexes)
        
        assert json.dumps(result1, sort_keys=True) == json.dumps(result2, sort_keys=True)

    def test_json_safe(self):
        """Verify register is JSON serializable."""
        annexes = [
            {
                "cal_id": "CAL-EXP-1",
                "lattice_band": "COHERENT",
                "transition_status": "OK",
                "coherence_band": "COHERENT",
            }
        ]
        
        register = build_structural_cohesion_register(annexes)
        
        # Should not raise
        json_str = json.dumps(register, sort_keys=True)
        assert len(json_str) > 0
        
        # Should round-trip
        parsed = json.loads(json_str)
        assert parsed["total_experiments"] == 1

    def test_non_mutating(self):
        """Verify function does not mutate input annexes."""
        annexes = [
            {
                "cal_id": "CAL-EXP-1",
                "lattice_band": "COHERENT",
                "transition_status": "OK",
                "coherence_band": "COHERENT",
            }
        ]
        annexes_copy = [dict(a) for a in annexes]
        
        build_structural_cohesion_register(annexes)
        
        assert annexes == annexes_copy

    def test_sorts_misaligned_experiments(self):
        """Verify misaligned experiments are sorted for determinism."""
        annexes = [
            {
                "cal_id": "CAL-EXP-3",
                "lattice_band": "MISALIGNED",
                "transition_status": "BLOCK",
            },
            {
                "cal_id": "CAL-EXP-1",
                "lattice_band": "MISALIGNED",
                "transition_status": "BLOCK",
            },
            {
                "cal_id": "CAL-EXP-2",
                "lattice_band": "COHERENT",
                "transition_status": "OK",
            },
        ]
        
        register = build_structural_cohesion_register(annexes)
        
        assert register["experiments_with_misaligned_structure"] == ["CAL-EXP-1", "CAL-EXP-3"]

    def test_has_top_misaligned_field(self):
        """Verify register includes top_misaligned field."""
        annexes = [
            {
                "cal_id": "CAL-EXP-1",
                "lattice_band": "COHERENT",
                "transition_status": "OK",
                "coherence_band": "COHERENT",
            }
        ]
        
        register = build_structural_cohesion_register(annexes)
        
        assert "top_misaligned" in register
        assert isinstance(register["top_misaligned"], list)

    def test_top_misaligned_ranks_lattice_misaligned_first(self):
        """Verify lattice MISALIGNED experiments rank highest."""
        annexes = [
            {
                "cal_id": "CAL-EXP-1",
                "lattice_band": "COHERENT",
                "transition_status": "OK",
                "lean_shadow_status": "BLOCK",
            },
            {
                "cal_id": "CAL-EXP-2",
                "lattice_band": "MISALIGNED",
                "transition_status": "BLOCK",
            },
            {
                "cal_id": "CAL-EXP-3",
                "lattice_band": "COHERENT",
                "transition_status": "OK",
                "coherence_band": "MISALIGNED",
            },
        ]
        
        register = build_structural_cohesion_register(annexes)
        
        # CAL-EXP-2 (lattice MISALIGNED) should rank first
        assert register["top_misaligned"][0] == "CAL-EXP-2"

    def test_top_misaligned_ranks_lean_shadow_block_second(self):
        """Verify lean shadow BLOCK experiments rank second (after lattice MISALIGNED)."""
        annexes = [
            {
                "cal_id": "CAL-EXP-1",
                "lattice_band": "COHERENT",
                "transition_status": "OK",
                "lean_shadow_status": "BLOCK",
            },
            {
                "cal_id": "CAL-EXP-2",
                "lattice_band": "COHERENT",
                "transition_status": "OK",
                "coherence_band": "MISALIGNED",
            },
        ]
        
        register = build_structural_cohesion_register(annexes)
        
        # CAL-EXP-1 (lean shadow BLOCK) should rank before CAL-EXP-2 (coherence MISALIGNED)
        assert register["top_misaligned"][0] == "CAL-EXP-1"
        assert register["top_misaligned"][1] == "CAL-EXP-2"

    def test_top_misaligned_ranks_coherence_misaligned_third(self):
        """Verify coherence MISALIGNED experiments rank third."""
        annexes = [
            {
                "cal_id": "CAL-EXP-1",
                "lattice_band": "COHERENT",
                "transition_status": "OK",
                "coherence_band": "MISALIGNED",
            },
            {
                "cal_id": "CAL-EXP-2",
                "lattice_band": "COHERENT",
                "transition_status": "OK",
                "coherence_band": "PARTIAL",
            },
        ]
        
        register = build_structural_cohesion_register(annexes)
        
        # CAL-EXP-1 (coherence MISALIGNED) should be in top_misaligned
        assert "CAL-EXP-1" in register["top_misaligned"]
        assert "CAL-EXP-2" not in register["top_misaligned"]

    def test_top_misaligned_limits_to_five(self):
        """Verify top_misaligned is limited to 5 experiments."""
        annexes = [
            {
                "cal_id": f"CAL-EXP-{i}",
                "lattice_band": "MISALIGNED",
                "transition_status": "BLOCK",
            }
            for i in range(1, 8)  # 7 misaligned experiments
        ]
        
        register = build_structural_cohesion_register(annexes)
        
        assert len(register["top_misaligned"]) == 5

    def test_top_misaligned_sorted_by_cal_id_for_determinism(self):
        """Verify top_misaligned is sorted by cal_id within same rank for determinism."""
        annexes = [
            {
                "cal_id": "CAL-EXP-3",
                "lattice_band": "MISALIGNED",
                "transition_status": "BLOCK",
            },
            {
                "cal_id": "CAL-EXP-1",
                "lattice_band": "MISALIGNED",
                "transition_status": "BLOCK",
            },
            {
                "cal_id": "CAL-EXP-2",
                "lattice_band": "MISALIGNED",
                "transition_status": "BLOCK",
            },
        ]
        
        register = build_structural_cohesion_register(annexes)
        
        # Should be sorted by cal_id
        assert register["top_misaligned"] == ["CAL-EXP-1", "CAL-EXP-2", "CAL-EXP-3"]

    def test_top_misaligned_deterministic(self):
        """Verify top_misaligned ranking is deterministic."""
        annexes = [
            {
                "cal_id": "CAL-EXP-1",
                "lattice_band": "COHERENT",
                "transition_status": "OK",
                "lean_shadow_status": "BLOCK",
            },
            {
                "cal_id": "CAL-EXP-2",
                "lattice_band": "MISALIGNED",
                "transition_status": "BLOCK",
            },
            {
                "cal_id": "CAL-EXP-3",
                "lattice_band": "COHERENT",
                "transition_status": "OK",
                "coherence_band": "MISALIGNED",
            },
        ]
        
        result1 = build_structural_cohesion_register(annexes)
        result2 = build_structural_cohesion_register(annexes)
        
        assert result1["top_misaligned"] == result2["top_misaligned"]

    def test_top_misaligned_empty_when_no_misalignments(self):
        """Verify top_misaligned is empty when no misalignments."""
        annexes = [
            {
                "cal_id": "CAL-EXP-1",
                "lattice_band": "COHERENT",
                "transition_status": "OK",
                "coherence_band": "COHERENT",
            }
        ]
        
        register = build_structural_cohesion_register(annexes)
        
        assert register["top_misaligned"] == []


class TestAttachStructuralCohesionRegisterToEvidence:
    """Tests for attach_structural_cohesion_register_to_evidence."""

    def test_attaches_to_governance_section(self):
        """Verify register is attached under governance.structural_cohesion_register."""
        evidence = {
            "timestamp": "2024-01-01T00:00:00Z",
            "run_id": "test_run_001",
        }
        register = {
            "schema_version": "1.0.0",
            "total_experiments": 2,
            "experiments_with_misaligned_structure": [],
        }
        
        updated = attach_structural_cohesion_register_to_evidence(evidence, register)
        
        assert "governance" in updated
        assert "structural_cohesion_register" in updated["governance"]
        assert updated["governance"]["structural_cohesion_register"]["total_experiments"] == 2

    def test_creates_governance_section_if_missing(self):
        """Verify governance section is created if missing."""
        evidence = {"timestamp": "2024-01-01", "data": {}}
        register = {
            "schema_version": "1.0.0",
            "total_experiments": 1,
            "experiments_with_misaligned_structure": [],
        }
        
        updated = attach_structural_cohesion_register_to_evidence(evidence, register)
        
        assert "governance" in updated
        assert "structural_cohesion_register" in updated["governance"]

    def test_non_mutating(self):
        """Verify function does not mutate input."""
        evidence = {"timestamp": "2024-01-01", "data": {}}
        register = {
            "schema_version": "1.0.0",
            "total_experiments": 1,
            "experiments_with_misaligned_structure": [],
        }
        
        evidence_copy = dict(evidence)
        register_copy = dict(register)
        
        attach_structural_cohesion_register_to_evidence(evidence, register)
        
        assert evidence == evidence_copy
        assert register == register_copy

    def test_deterministic(self):
        """Verify function is deterministic."""
        evidence = {"timestamp": "2024-01-01", "data": {}}
        register = {
            "schema_version": "1.0.0",
            "total_experiments": 1,
            "experiments_with_misaligned_structure": [],
        }
        
        result1 = attach_structural_cohesion_register_to_evidence(evidence, register)
        result2 = attach_structural_cohesion_register_to_evidence(evidence, register)
        
        assert json.dumps(result1, sort_keys=True) == json.dumps(result2, sort_keys=True)

    def test_json_safe(self):
        """Verify output is JSON serializable."""
        evidence = {"timestamp": "2024-01-01", "data": {}}
        register = {
            "schema_version": "1.0.0",
            "total_experiments": 1,
            "experiments_with_misaligned_structure": [],
        }
        
        updated = attach_structural_cohesion_register_to_evidence(evidence, register)
        
        # Should not raise
        json_str = json.dumps(updated)
        assert len(json_str) > 0


class TestStructuralCohesionRegisterIntegration:
    """Integration tests for structural cohesion register workflow."""

    def test_full_workflow(self, temp_calibration_dir):
        """Verify full workflow: annex → emission → register → evidence."""
        # Step 1: Build annexes
        atlas_tile1 = {
            "lattice_coherence_band": "COHERENT",
            "transition_status": "OK",
        }
        structure_tile1 = {"status": "OK"}
        coherence_tile1 = {"coherence_band": "COHERENT"}
        
        annex1 = build_first_light_structural_cohesion_annex(
            atlas_tile1, structure_tile=structure_tile1, coherence_tile=coherence_tile1
        )
        
        atlas_tile2 = {
            "lattice_coherence_band": "MISALIGNED",
            "transition_status": "BLOCK",
        }
        annex2 = build_first_light_structural_cohesion_annex(atlas_tile2)
        
        # Step 2: Emit annexes
        emitted1 = emit_cal_exp_structural_cohesion_annex(
            "CAL-EXP-1", annex1, output_dir=temp_calibration_dir
        )
        emitted2 = emit_cal_exp_structural_cohesion_annex(
            "CAL-EXP-2", annex2, output_dir=temp_calibration_dir
        )
        
        # Step 3: Build register
        register = build_structural_cohesion_register([emitted1, emitted2])
        
        # Step 4: Attach to evidence
        evidence = {"timestamp": "2024-01-01", "data": {}}
        enriched = attach_structural_cohesion_register_to_evidence(evidence, register)
        
        # Verify
        assert register["total_experiments"] == 2
        assert "CAL-EXP-2" in register["experiments_with_misaligned_structure"]
        assert "structural_cohesion_register" in enriched["governance"]
        
        # Verify files exist
        file1 = Path(temp_calibration_dir) / "structural_cohesion_annex_CAL-EXP-1.json"
        file2 = Path(temp_calibration_dir) / "structural_cohesion_annex_CAL-EXP-2.json"
        assert file1.exists()
        assert file2.exists()


class TestExtractStructuralCohesionRegisterSignal:
    """Tests for extract_structural_cohesion_register_signal."""

    def test_extracts_signal_when_register_present(self):
        """Verify signal is extracted when register present in evidence."""
        evidence = {
            "governance": {
                "structural_cohesion_register": {
                    "schema_version": "1.0.0",
                    "total_experiments": 3,
                    "experiments_with_misaligned_structure": ["CAL-EXP-2"],
                    "top_misaligned": ["CAL-EXP-2"],
                }
            }
        }
        
        signal = extract_structural_cohesion_register_signal(evidence)
        
        assert signal is not None
        assert signal["total_experiments"] == 3
        assert signal["experiments_with_misaligned_structure_count"] == 1
        assert signal["top_misaligned"] == ["CAL-EXP-2"]

    def test_returns_none_when_register_missing(self):
        """Verify returns None when register not present."""
        evidence = {"timestamp": "2024-01-01", "data": {}}
        
        signal = extract_structural_cohesion_register_signal(evidence)
        
        assert signal is None

    def test_returns_none_when_governance_missing(self):
        """Verify returns None when governance section missing."""
        evidence = {"timestamp": "2024-01-01"}
        
        signal = extract_structural_cohesion_register_signal(evidence)
        
        assert signal is None

    def test_handles_missing_fields_gracefully(self):
        """Verify handles missing register fields gracefully."""
        evidence = {
            "governance": {
                "structural_cohesion_register": {
                    "schema_version": "1.0.0",
                    # Missing other fields
                }
            }
        }
        
        signal = extract_structural_cohesion_register_signal(evidence)
        
        assert signal is not None
        assert signal["total_experiments"] == 0
        assert signal["experiments_with_misaligned_structure_count"] == 0
        assert signal["top_misaligned"] == []

    def test_json_safe(self):
        """Verify signal is JSON serializable."""
        evidence = {
            "governance": {
                "structural_cohesion_register": {
                    "total_experiments": 2,
                    "experiments_with_misaligned_structure": ["CAL-EXP-1"],
                    "top_misaligned": ["CAL-EXP-1"],
                }
            }
        }
        
        signal = extract_structural_cohesion_register_signal(evidence)
        
        # Should not raise
        json_str = json.dumps(signal)
        assert len(json_str) > 0


class TestAttachStructuralCohesionRegisterSignalToEvidence:
    """Tests for attach_structural_cohesion_register_signal_to_evidence."""

    def test_attaches_signal_when_register_present(self):
        """Verify signal is attached when register present in evidence."""
        evidence = {
            "governance": {
                "structural_cohesion_register": {
                    "total_experiments": 3,
                    "experiments_with_misaligned_structure": ["CAL-EXP-2"],
                    "top_misaligned": ["CAL-EXP-2"],
                }
            }
        }
        
        enriched = attach_structural_cohesion_register_signal_to_evidence(evidence)
        
        assert "signals" in enriched
        assert "structural_cohesion_register" in enriched["signals"]
        signal = enriched["signals"]["structural_cohesion_register"]
        assert signal["total_experiments"] == 3
        assert signal["experiments_with_misaligned_structure_count"] == 1

    def test_creates_signals_section_if_missing(self):
        """Verify signals section is created if missing."""
        evidence = {
            "governance": {
                "structural_cohesion_register": {
                    "total_experiments": 1,
                    "top_misaligned": [],
                }
            }
        }
        
        enriched = attach_structural_cohesion_register_signal_to_evidence(evidence)
        
        assert "signals" in enriched
        assert "structural_cohesion_register" in enriched["signals"]

    def test_no_signal_when_register_missing(self):
        """Verify no signal attached when register not present."""
        evidence = {"timestamp": "2024-01-01", "data": {}}
        
        enriched = attach_structural_cohesion_register_signal_to_evidence(evidence)
        
        # Should not have signals section if register missing
        assert "signals" not in enriched or "structural_cohesion_register" not in enriched.get("signals", {})

    def test_non_mutating(self):
        """Verify function does not mutate input."""
        evidence = {
            "governance": {
                "structural_cohesion_register": {
                    "total_experiments": 1,
                    "top_misaligned": [],
                }
            }
        }
        
        evidence_copy = dict(evidence)
        
        attach_structural_cohesion_register_signal_to_evidence(evidence)
        
        assert evidence == evidence_copy

    def test_deterministic(self):
        """Verify function is deterministic."""
        evidence = {
            "governance": {
                "structural_cohesion_register": {
                    "total_experiments": 2,
                    "experiments_with_misaligned_structure": ["CAL-EXP-1"],
                    "top_misaligned": ["CAL-EXP-1"],
                }
            }
        }
        
        result1 = attach_structural_cohesion_register_signal_to_evidence(evidence)
        result2 = attach_structural_cohesion_register_signal_to_evidence(evidence)
        
        assert json.dumps(result1, sort_keys=True) == json.dumps(result2, sort_keys=True)

    def test_json_safe(self):
        """Verify output is JSON serializable."""
        evidence = {
            "governance": {
                "structural_cohesion_register": {
                    "total_experiments": 1,
                    "top_misaligned": [],
                }
            }
        }
        
        enriched = attach_structural_cohesion_register_signal_to_evidence(evidence)
        
        # Should not raise
        json_str = json.dumps(enriched)
        assert len(json_str) > 0



