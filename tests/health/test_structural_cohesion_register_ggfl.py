"""Tests for structural cohesion register GGFL adapter.

STATUS: PHASE X — STRUCTURAL COHESION REGISTER GGFL ADAPTER

Tests verify:
- GGFL adapter output shape and fields
- Status determination (ok vs warn)
- Determinism
- JSON safety
"""

import json
from typing import Any, Dict

import pytest

from backend.health.atlas_governance_adapter import (
    structural_cohesion_register_for_alignment_view,
)


@pytest.fixture
def sample_signal():
    """Fixture for sample structural cohesion register signal."""
    return {
        "total_experiments": 3,
        "experiments_with_misaligned_structure_count": 2,
        "top_misaligned": ["CAL-EXP-2", "CAL-EXP-3"],
        "extraction_source": "MANIFEST",
    }


@pytest.fixture
def sample_register():
    """Fixture for sample structural cohesion register."""
    return {
        "schema_version": "1.0.0",
        "total_experiments": 3,
        "experiments_with_misaligned_structure": ["CAL-EXP-2", "CAL-EXP-3"],
        "top_misaligned": ["CAL-EXP-2", "CAL-EXP-3"],
        "band_combinations": {},
    }


class TestStructuralCohesionRegisterForAlignmentView:
    """Tests for structural_cohesion_register_for_alignment_view."""

    def test_has_required_fields(self, sample_signal: Dict[str, Any]):
        """Verify GGFL adapter output has all required fields."""
        view = structural_cohesion_register_for_alignment_view(sample_signal)
        
        assert "signal_type" in view
        assert "status" in view
        assert "conflict" in view
        assert "weight_hint" in view
        assert "drivers" in view
        assert "summary" in view

    def test_signal_type_is_sig_str(self, sample_signal: Dict[str, Any]):
        """Verify signal_type is SIG-STR."""
        view = structural_cohesion_register_for_alignment_view(sample_signal)
        
        assert view["signal_type"] == "SIG-STR"

    def test_conflict_is_false(self, sample_signal: Dict[str, Any]):
        """Verify conflict is always False."""
        view = structural_cohesion_register_for_alignment_view(sample_signal)
        
        assert view["conflict"] is False

    def test_weight_hint_is_low(self, sample_signal: Dict[str, Any]):
        """Verify weight_hint is LOW."""
        view = structural_cohesion_register_for_alignment_view(sample_signal)
        
        assert view["weight_hint"] == "LOW"

    def test_status_warn_when_misalignments_exist(self, sample_signal: Dict[str, Any]):
        """Verify status is warn when misalignments exist."""
        view = structural_cohesion_register_for_alignment_view(sample_signal)
        
        assert view["status"] == "warn"

    def test_status_ok_when_no_misalignments(self):
        """Verify status is ok when no misalignments exist."""
        signal = {
            "total_experiments": 3,
            "experiments_with_misaligned_structure_count": 0,
            "top_misaligned": [],
        }
        
        view = structural_cohesion_register_for_alignment_view(signal)
        
        assert view["status"] == "ok"

    def test_drivers_has_driver_when_misalignments_exist(self, sample_signal: Dict[str, Any]):
        """Verify drivers includes DRIVER_MISALIGNED_EXPERIMENTS_PRESENT when misalignments exist."""
        view = structural_cohesion_register_for_alignment_view(sample_signal)
        
        assert "DRIVER_MISALIGNED_EXPERIMENTS_PRESENT" in view["drivers"]
        assert len(view["drivers"]) == 1

    def test_drivers_empty_when_no_misalignments(self):
        """Verify drivers is empty when no misalignments exist."""
        signal = {
            "total_experiments": 3,
            "experiments_with_misaligned_structure_count": 0,
            "top_misaligned": [],
        }
        
        view = structural_cohesion_register_for_alignment_view(signal)
        
        assert len(view["drivers"]) == 0

    def test_works_with_register_format(self, sample_register: Dict[str, Any]):
        """Verify adapter works with full register format."""
        view = structural_cohesion_register_for_alignment_view(sample_register)
        
        assert view["signal_type"] == "SIG-STR"
        assert view["status"] == "warn"
        assert "DRIVER_MISALIGNED_EXPERIMENTS_PRESENT" in view["drivers"]

    def test_summary_includes_misalignment_info(self, sample_signal: Dict[str, Any]):
        """Verify summary includes misalignment information."""
        view = structural_cohesion_register_for_alignment_view(sample_signal)
        
        assert "2 out of 3" in view["summary"]
        assert "misaligned structure" in view["summary"]

    def test_summary_ok_when_no_misalignments(self):
        """Verify summary indicates ok when no misalignments."""
        signal = {
            "total_experiments": 3,
            "experiments_with_misaligned_structure_count": 0,
            "top_misaligned": [],
        }
        
        view = structural_cohesion_register_for_alignment_view(signal)
        
        assert "aligned structure" in view["summary"]
        assert "3 calibration experiment(s)" in view["summary"]

    def test_deterministic(self, sample_signal: Dict[str, Any]):
        """Verify adapter output is deterministic."""
        view1 = structural_cohesion_register_for_alignment_view(sample_signal)
        view2 = structural_cohesion_register_for_alignment_view(sample_signal)
        
        assert json.dumps(view1, sort_keys=True) == json.dumps(view2, sort_keys=True)

    def test_json_safe(self, sample_signal: Dict[str, Any]):
        """Verify adapter output is JSON serializable."""
        view = structural_cohesion_register_for_alignment_view(sample_signal)
        
        # Should not raise
        json_str = json.dumps(view, sort_keys=True)
        assert len(json_str) > 0
        
        # Should round-trip
        parsed = json.loads(json_str)
        assert parsed["signal_type"] == "SIG-STR"

    def test_non_mutating(self, sample_signal: Dict[str, Any]):
        """Verify adapter does not mutate input."""
        signal_copy = dict(sample_signal)
        
        structural_cohesion_register_for_alignment_view(sample_signal)
        
        assert sample_signal == signal_copy

    def test_shadow_mode_invariants_present(self, sample_signal: Dict[str, Any]):
        """Verify shadow_mode_invariants are present in output."""
        view = structural_cohesion_register_for_alignment_view(sample_signal)
        
        assert "shadow_mode_invariants" in view
        invariants = view["shadow_mode_invariants"]
        assert invariants["advisory_only"] is True
        assert invariants["no_enforcement"] is True
        assert invariants["conflict_invariant"] is True

    def test_shadow_mode_invariants_always_true(self):
        """Verify shadow_mode_invariants are always True regardless of input."""
        signal_ok = {
            "total_experiments": 3,
            "experiments_with_misaligned_structure_count": 0,
        }
        signal_warn = {
            "total_experiments": 3,
            "experiments_with_misaligned_structure_count": 2,
        }
        
        view_ok = structural_cohesion_register_for_alignment_view(signal_ok)
        view_warn = structural_cohesion_register_for_alignment_view(signal_warn)
        
        # Both should have invariants set to True
        assert view_ok["shadow_mode_invariants"]["advisory_only"] is True
        assert view_ok["shadow_mode_invariants"]["no_enforcement"] is True
        assert view_ok["shadow_mode_invariants"]["conflict_invariant"] is True
        
        assert view_warn["shadow_mode_invariants"]["advisory_only"] is True
        assert view_warn["shadow_mode_invariants"]["no_enforcement"] is True
        assert view_warn["shadow_mode_invariants"]["conflict_invariant"] is True

