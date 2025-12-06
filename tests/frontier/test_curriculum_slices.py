"""
Tests for Curriculum Slice Configuration

Validates that curriculum slices, especially the Wide Slice (slice_medium),
meet the requirements for RFL uplift experiments.
"""

import pytest

from backend.frontier.curriculum import CurriculumSystem, load


class TestWideSliceConfiguration:
    """Tests for Wide Slice (slice_medium) configuration."""

    def test_wide_slice_exists(self):
        """Wide Slice (slice_medium) must exist in the curriculum."""
        system = load("pl")
        wide_slice = next((s for s in system.slices if s.name == "slice_medium"), None)
        assert wide_slice is not None, "slice_medium (Wide Slice) must exist in curriculum"

    def test_wide_slice_atoms_threshold(self):
        """Wide Slice must have atoms >= 5 for non-trivial complexity."""
        system = load("pl")
        wide_slice = next(s for s in system.slices if s.name == "slice_medium")
        assert wide_slice.params.get("atoms") >= 5, \
            f"Wide Slice atoms={wide_slice.params.get('atoms')} must be >= 5"

    def test_wide_slice_depth_threshold(self):
        """Wide Slice must have depth_max >= 7 for sufficient depth."""
        system = load("pl")
        wide_slice = next(s for s in system.slices if s.name == "slice_medium")
        assert wide_slice.params.get("depth_max") >= 7, \
            f"Wide Slice depth_max={wide_slice.params.get('depth_max')} must be >= 7"

    def test_wide_slice_total_max_threshold(self):
        """Wide Slice must have total_max >= 8000 for wide search space."""
        system = load("pl")
        wide_slice = next(s for s in system.slices if s.name == "slice_medium")
        assert wide_slice.params.get("total_max") >= 8000, \
            f"Wide Slice total_max={wide_slice.params.get('total_max')} must be >= 8000"

    def test_wide_slice_abstention_config(self):
        """Wide Slice abstention gate should allow 5-20% abstention for statistical interest."""
        system = load("pl")
        wide_slice = next(s for s in system.slices if s.name == "slice_medium")
        max_rate = wide_slice.gates.abstention.max_rate_pct
        # Should allow at least 15% abstention (current config allows 15%)
        assert max_rate >= 10.0, \
            f"Wide Slice abstention.max_rate_pct={max_rate} should allow >= 10% for statistical interest"
        assert max_rate <= 20.0, \
            f"Wide Slice abstention.max_rate_pct={max_rate} should not exceed 20% for meaningful coverage"

    def test_wide_slice_caps_min_attempt_mass(self):
        """Wide Slice should have sufficient min_attempt_mass for partial RFL runs / Phase I prototype."""
        system = load("pl")
        wide_slice = next(s for s in system.slices if s.name == "slice_medium")
        min_mass = wide_slice.gates.caps.min_attempt_mass
        # Should have enough mass to produce meaningful statistics over partial RFL runs
        # With 3000 min_attempt_mass, we get at least 3 attempts per cycle on average
        # NOTE: This validates configuration only; no Wide Slice experiments have been run yet.
        # Phase I has partial RFL runs (e.g., fo_rfl.jsonl with ~330 cycles), not full 1000-cycle runs.
        assert min_mass >= 2000, \
            f"Wide Slice caps.min_attempt_mass={min_mass} should be >= 2000 for sufficient events"

    def test_wide_slice_gate_compatibility(self):
        """Wide Slice gates should be compatible with NormalizedMetrics.from_raw expectations."""
        system = load("pl")
        wide_slice = next(s for s in system.slices if s.name == "slice_medium")
        
        # All required gates must be present
        assert wide_slice.gates.coverage is not None
        assert wide_slice.gates.abstention is not None
        assert wide_slice.gates.velocity is not None
        assert wide_slice.gates.caps is not None
        
        # Coverage gate thresholds should be reasonable (0.85 is appropriate for wide slice)
        assert 0.70 <= wide_slice.gates.coverage.ci_lower_min <= 0.95, \
            f"coverage.ci_lower_min={wide_slice.gates.coverage.ci_lower_min} should be in [0.70, 0.95]"
        
        # Velocity gate should have reasonable thresholds
        assert wide_slice.gates.velocity.min_pph > 0, \
            "velocity.min_pph must be positive"
        assert 0 < wide_slice.gates.velocity.stability_cv_max <= 0.20, \
            f"velocity.stability_cv_max={wide_slice.gates.velocity.stability_cv_max} should be in (0, 0.20]"


class TestSliceOrderingAndMonotonicity:
    """Tests for slice ordering and monotonicity constraints."""

    def test_slice_medium_comes_after_easy(self):
        """slice_medium should come after slice_easy_fo in the curriculum."""
        system = load("pl")
        slice_names = [s.name for s in system.slices]
        easy_idx = slice_names.index("slice_easy_fo")
        medium_idx = slice_names.index("slice_medium")
        assert easy_idx < medium_idx, \
            "slice_medium should come after slice_easy_fo in curriculum ordering"

    def test_monotonicity_respected(self):
        """All slices should respect monotonicity constraints on atoms and depth_max."""
        system = load("pl")
        # This is validated by CurriculumSystem._validate_monotonicity()
        # If we can load the system without errors, monotonicity is respected
        assert len(system.slices) > 0, "System should have at least one slice"


class TestSliceAccessibility:
    """Tests for accessing and querying slices."""

    def test_can_resolve_slice_by_name(self):
        """Should be able to resolve slice_medium by name."""
        system = load("pl")
        wide_slice = next((s for s in system.slices if s.name == "slice_medium"), None)
        assert wide_slice is not None
        assert wide_slice.name == "slice_medium"

    def test_wide_slice_is_completable(self):
        """Wide Slice should have all required fields for completion tracking."""
        system = load("pl")
        wide_slice = next(s for s in system.slices if s.name == "slice_medium")
        # Should have name, params, and gates (completed_at is optional)
        assert wide_slice.name is not None
        assert wide_slice.params is not None
        assert wide_slice.gates is not None

