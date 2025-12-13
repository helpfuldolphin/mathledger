"""Tests for structural cohesion register signal consistency check.

STATUS: PHASE X — STRUCTURAL COHESION REGISTER CONSISTENCY CHECK

Tests verify:
- Consistency check output shape and fields
- Status mismatch detection
- Conflict invariant violation detection
- Signal type and weight hint validation
- Determinism
- JSON safety
"""

import json
from typing import Any, Dict

import pytest

from backend.health.atlas_governance_adapter import (
    structural_cohesion_register_for_alignment_view,
    summarize_structural_cohesion_register_signal_consistency,
)


@pytest.fixture
def sample_status_signal():
    """Fixture for sample status signal."""
    return {
        "total_experiments": 3,
        "experiments_with_misaligned_structure_count": 2,
        "top_misaligned": ["CAL-EXP-2", "CAL-EXP-3"],
        "extraction_source": "MANIFEST",
    }


@pytest.fixture
def sample_ggfl_signal():
    """Fixture for sample GGFL signal."""
    return {
        "signal_type": "SIG-STR",
        "status": "warn",
        "conflict": False,
        "weight_hint": "LOW",
        "drivers": ["DRIVER_MISALIGNED_EXPERIMENTS_PRESENT"],
        "summary": "Structural cohesion register: 2 out of 3 calibration experiment(s) show misaligned structure.",
        "shadow_mode_invariants": {
            "advisory_only": True,
            "no_enforcement": True,
            "conflict_invariant": True,
        },
    }


class TestSummarizeStructuralCohesionRegisterSignalConsistency:
    """Tests for summarize_structural_cohesion_register_signal_consistency."""

    def test_has_required_fields(
        self, sample_status_signal: Dict[str, Any], sample_ggfl_signal: Dict[str, Any]
    ):
        """Verify consistency check output has all required fields."""
        result = summarize_structural_cohesion_register_signal_consistency(
            sample_status_signal, sample_ggfl_signal
        )
        
        assert "schema_version" in result
        assert "mode" in result
        assert "consistency" in result
        assert "notes" in result
        assert "conflict_invariant_violated" in result
        assert "top_mismatch_type" in result

    def test_consistent_when_signals_match(
        self, sample_status_signal: Dict[str, Any], sample_ggfl_signal: Dict[str, Any]
    ):
        """Verify consistency check returns CONSISTENT when signals match."""
        result = summarize_structural_cohesion_register_signal_consistency(
            sample_status_signal, sample_ggfl_signal
        )
        
        assert result["consistency"] == "CONSISTENT"
        assert result["conflict_invariant_violated"] is False
        assert any("consistent" in note.lower() for note in result["notes"])

    def test_partial_when_status_mismatch(self):
        """Verify consistency check detects status mismatch."""
        status_signal = {
            "total_experiments": 3,
            "experiments_with_misaligned_structure_count": 2,  # Should be "warn"
        }
        ggfl_signal = {
            "signal_type": "SIG-STR",
            "status": "ok",  # Mismatch: status says 2 misaligned but GGFL says ok
            "conflict": False,
            "weight_hint": "LOW",
        }
        
        result = summarize_structural_cohesion_register_signal_consistency(
            status_signal, ggfl_signal
        )
        
        assert result["consistency"] == "PARTIAL"
        assert result["conflict_invariant_violated"] is False
        assert result["top_mismatch_type"] == "status_mismatch"
        assert any("status mismatch" in note.lower() for note in result["notes"])

    def test_inconsistent_when_conflict_violated(self):
        """Verify consistency check detects conflict invariant violation."""
        status_signal = {
            "total_experiments": 3,
            "experiments_with_misaligned_structure_count": 0,
        }
        ggfl_signal = {
            "signal_type": "SIG-STR",
            "status": "ok",
            "conflict": True,  # VIOLATION: conflict must always be False
            "weight_hint": "LOW",
        }
        
        result = summarize_structural_cohesion_register_signal_consistency(
            status_signal, ggfl_signal
        )
        
        assert result["consistency"] == "INCONSISTENT"
        assert result["conflict_invariant_violated"] is True
        assert result["top_mismatch_type"] == "conflict_invariant_violated"
        assert any("conflict invariant violated" in note.lower() for note in result["notes"])

    def test_partial_when_signal_type_mismatch(self):
        """Verify consistency check detects signal type mismatch."""
        status_signal = {
            "total_experiments": 3,
            "experiments_with_misaligned_structure_count": 0,
        }
        ggfl_signal = {
            "signal_type": "SIG-OTHER",  # Mismatch: should be "SIG-STR"
            "status": "ok",
            "conflict": False,
            "weight_hint": "LOW",
        }
        
        result = summarize_structural_cohesion_register_signal_consistency(
            status_signal, ggfl_signal
        )
        
        assert result["consistency"] == "PARTIAL"
        assert result["top_mismatch_type"] == "signal_type_mismatch"
        assert any("signal type mismatch" in note.lower() for note in result["notes"])

    def test_partial_when_weight_hint_mismatch(self):
        """Verify consistency check detects weight hint mismatch."""
        status_signal = {
            "total_experiments": 3,
            "experiments_with_misaligned_structure_count": 0,
        }
        ggfl_signal = {
            "signal_type": "SIG-STR",
            "status": "ok",
            "conflict": False,
            "weight_hint": "HIGH",  # Mismatch: should be "LOW"
        }
        
        result = summarize_structural_cohesion_register_signal_consistency(
            status_signal, ggfl_signal
        )
        
        assert result["consistency"] == "PARTIAL"
        assert result["top_mismatch_type"] == "weight_hint_mismatch"
        assert any("weight hint mismatch" in note.lower() for note in result["notes"])

    def test_consistent_when_no_misalignments(self):
        """Verify consistency check returns CONSISTENT when no misalignments."""
        status_signal = {
            "total_experiments": 3,
            "experiments_with_misaligned_structure_count": 0,
        }
        ggfl_signal = {
            "signal_type": "SIG-STR",
            "status": "ok",
            "conflict": False,
            "weight_hint": "LOW",
        }
        
        result = summarize_structural_cohesion_register_signal_consistency(
            status_signal, ggfl_signal
        )
        
        assert result["consistency"] == "CONSISTENT"
        assert result["conflict_invariant_violated"] is False

    def test_deterministic(
        self, sample_status_signal: Dict[str, Any], sample_ggfl_signal: Dict[str, Any]
    ):
        """Verify consistency check is deterministic."""
        result1 = summarize_structural_cohesion_register_signal_consistency(
            sample_status_signal, sample_ggfl_signal
        )
        result2 = summarize_structural_cohesion_register_signal_consistency(
            sample_status_signal, sample_ggfl_signal
        )
        
        assert json.dumps(result1, sort_keys=True) == json.dumps(result2, sort_keys=True)

    def test_json_safe(
        self, sample_status_signal: Dict[str, Any], sample_ggfl_signal: Dict[str, Any]
    ):
        """Verify consistency check output is JSON serializable."""
        result = summarize_structural_cohesion_register_signal_consistency(
            sample_status_signal, sample_ggfl_signal
        )
        
        # Should not raise
        json_str = json.dumps(result, sort_keys=True)
        assert len(json_str) > 0
        
        # Should round-trip
        parsed = json.loads(json_str)
        assert parsed["consistency"] == "CONSISTENT"

    def test_non_mutating(
        self, sample_status_signal: Dict[str, Any], sample_ggfl_signal: Dict[str, Any]
    ):
        """Verify consistency check does not mutate inputs."""
        status_copy = dict(sample_status_signal)
        ggfl_copy = dict(sample_ggfl_signal)
        
        summarize_structural_cohesion_register_signal_consistency(
            sample_status_signal, sample_ggfl_signal
        )
        
        assert sample_status_signal == status_copy
        assert sample_ggfl_signal == ggfl_copy


class TestStructuralCohesionRegisterConsistencyIntegration:
    """Integration tests for consistency check with GGFL adapter."""

    def test_consistent_with_matching_ggfl_adapter_output(self):
        """Verify consistency check works with actual GGFL adapter output."""
        status_signal = {
            "total_experiments": 3,
            "experiments_with_misaligned_structure_count": 2,
            "top_misaligned": ["CAL-EXP-2"],
        }
        
        # Generate GGFL signal from adapter
        ggfl_signal = structural_cohesion_register_for_alignment_view(status_signal)
        
        # Check consistency
        result = summarize_structural_cohesion_register_signal_consistency(
            status_signal, ggfl_signal
        )
        
        # Should be consistent (adapter derives status from misaligned_count correctly)
        assert result["consistency"] == "CONSISTENT"
        assert result["conflict_invariant_violated"] is False

    def test_mode_is_shadow(self):
        """Verify consistency check always returns mode=SHADOW."""
        status_signal = {
            "total_experiments": 1,
            "experiments_with_misaligned_structure_count": 0,
        }
        ggfl_signal = {
            "signal_type": "SIG-STR",
            "status": "ok",
            "conflict": False,
            "weight_hint": "LOW",
        }
        
        result = summarize_structural_cohesion_register_signal_consistency(
            status_signal, ggfl_signal
        )
        
        assert result["mode"] == "SHADOW"

