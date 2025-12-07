"""
Tests for Curriculum Stability Envelope

Tests fingerprinting, invariant validation, promotion guards, and CLI integration.
"""

import json
import pytest
import tempfile
from pathlib import Path

from curriculum.gates import (
    CurriculumSystem,
    CurriculumSlice,
    SliceGates,
    CoverageGateSpec,
    AbstentionGateSpec,
    VelocityGateSpec,
    CapsGateSpec,
)
from curriculum.stability_envelope import (
    compute_fingerprint,
    compute_fingerprint_diff,
    validate_curriculum_invariants,
    evaluate_curriculum_stability,
    FingerprintDiff,
    CurriculumInvariantReport,
    PromotionStabilityReport,
)
from curriculum.cli import main as cli_main


# Test Fixtures
# -----------------------------------------------------------------------------

def _make_test_gates(ci_lower: float = 0.9) -> SliceGates:
    """Create test gates with configurable coverage threshold."""
    return SliceGates(
        coverage=CoverageGateSpec(ci_lower_min=ci_lower, sample_min=10, require_attestation=False),
        abstention=AbstentionGateSpec(max_rate_pct=20.0, max_mass=500),
        velocity=VelocityGateSpec(min_pph=100.0, stability_cv_max=0.15, window_minutes=60),
        caps=CapsGateSpec(min_attempt_mass=1000, min_runtime_minutes=10.0, backlog_max=0.4),
    )


def _make_test_slice(name: str, atoms: int = 3, depth: int = 4, ci_lower: float = 0.9) -> CurriculumSlice:
    """Create a test curriculum slice."""
    return CurriculumSlice(
        name=name,
        params={"atoms": atoms, "depth_max": depth, "breadth_max": 100, "total_max": 1000},
        gates=_make_test_gates(ci_lower),
        metadata={"test": True},
    )


def _make_test_system(slices: list = None) -> CurriculumSystem:
    """Create a test curriculum system."""
    if slices is None:
        slices = [
            _make_test_slice("slice_a", atoms=3, depth=4),
            _make_test_slice("slice_b", atoms=4, depth=5),
        ]
    return CurriculumSystem(
        slug="test",
        description="Test curriculum",
        slices=slices,
        active_index=0,
        monotonic_axes=("atoms", "depth_max"),
        version=2,
    )


# Fingerprint Tests
# -----------------------------------------------------------------------------

class TestFingerprinting:
    """Tests for curriculum fingerprinting."""
    
    def test_compute_fingerprint_basic(self):
        """Test basic fingerprint computation."""
        system = _make_test_system()
        fp = compute_fingerprint(system)
        
        assert fp["slug"] == "test"
        assert fp["version"] == 2
        assert fp["description"] == "Test curriculum"
        assert fp["monotonic_axes"] == ["atoms", "depth_max"]
        assert fp["active_index"] == 0
        assert fp["active_name"] == "slice_a"
        assert len(fp["slices"]) == 2
    
    def test_fingerprint_slices_are_sorted(self):
        """Test that slices are sorted by name in fingerprint."""
        slices = [
            _make_test_slice("slice_z"),
            _make_test_slice("slice_a"),
            _make_test_slice("slice_m"),
        ]
        system = CurriculumSystem(
            slug="test",
            description="Test",
            slices=slices,
            active_index=0,
            monotonic_axes=(),
            version=2,
        )
        fp = compute_fingerprint(system)
        
        slice_names = [s["name"] for s in fp["slices"]]
        assert slice_names == ["slice_a", "slice_m", "slice_z"]
    
    def test_fingerprint_params_are_normalized(self):
        """Test that parameters are normalized and sorted."""
        system = _make_test_system()
        fp = compute_fingerprint(system)
        
        slice_fp = fp["slices"][0]
        params = slice_fp["params"]
        
        # Check keys are sorted
        assert list(params.keys()) == sorted(params.keys())
        
        # Check values are normalized
        assert isinstance(params["atoms"], int)
        assert isinstance(params["depth_max"], int)
    
    def test_fingerprint_gates_are_normalized(self):
        """Test that gates are normalized and sorted."""
        system = _make_test_system()
        fp = compute_fingerprint(system)
        
        slice_fp = fp["slices"][0]
        gates = slice_fp["gates"]
        
        # Check gate types are sorted
        assert list(gates.keys()) == ["abstention", "caps", "coverage", "velocity"]
        
        # Check each gate's keys are sorted
        for gate_type, gate_spec in gates.items():
            assert list(gate_spec.keys()) == sorted(gate_spec.keys())


class TestFingerprintDiff:
    """Tests for fingerprint diffing."""
    
    def test_identical_fingerprints_produce_no_diff(self):
        """Test that identical fingerprints produce no differences."""
        system = _make_test_system()
        fp = compute_fingerprint(system)
        
        diff = compute_fingerprint_diff(fp, fp)
        
        assert not diff.has_changes
        assert len(diff.changed_slices) == 0
        assert len(diff.param_diffs) == 0
        assert len(diff.gate_diffs) == 0
        assert len(diff.invariant_diffs) == 0
    
    def test_detects_added_slices(self):
        """Test detection of added slices."""
        system_a = _make_test_system([_make_test_slice("slice_a")])
        system_b = _make_test_system([
            _make_test_slice("slice_a"),
            _make_test_slice("slice_b"),
        ])
        
        fp_a = compute_fingerprint(system_a)
        fp_b = compute_fingerprint(system_b)
        
        diff = compute_fingerprint_diff(fp_a, fp_b)
        
        assert diff.has_changes
        assert diff.added_slices == ["slice_b"]
        assert len(diff.removed_slices) == 0
    
    def test_detects_removed_slices(self):
        """Test detection of removed slices."""
        system_a = _make_test_system([
            _make_test_slice("slice_a"),
            _make_test_slice("slice_b"),
        ])
        system_b = _make_test_system([_make_test_slice("slice_a")])
        
        fp_a = compute_fingerprint(system_a)
        fp_b = compute_fingerprint(system_b)
        
        diff = compute_fingerprint_diff(fp_a, fp_b)
        
        assert diff.has_changes
        assert diff.removed_slices == ["slice_b"]
        assert len(diff.added_slices) == 0
    
    def test_detects_param_changes(self):
        """Test detection of parameter changes."""
        system_a = _make_test_system([_make_test_slice("slice_a", atoms=3)])
        system_b = _make_test_system([_make_test_slice("slice_a", atoms=4)])
        
        fp_a = compute_fingerprint(system_a)
        fp_b = compute_fingerprint(system_b)
        
        diff = compute_fingerprint_diff(fp_a, fp_b)
        
        assert diff.has_changes
        assert "slice_a" in diff.changed_slices
        assert "slice_a" in diff.param_diffs
        assert diff.param_diffs["slice_a"]["atoms"] == (3, 4)
    
    def test_detects_gate_changes(self):
        """Test detection of gate threshold changes."""
        system_a = _make_test_system([_make_test_slice("slice_a", ci_lower=0.9)])
        system_b = _make_test_system([_make_test_slice("slice_a", ci_lower=0.95)])
        
        fp_a = compute_fingerprint(system_a)
        fp_b = compute_fingerprint(system_b)
        
        diff = compute_fingerprint_diff(fp_a, fp_b)
        
        assert diff.has_changes
        assert "slice_a" in diff.changed_slices
        assert "slice_a" in diff.gate_diffs
        assert "coverage.ci_lower_min" in diff.gate_diffs["slice_a"]
        assert diff.gate_diffs["slice_a"]["coverage.ci_lower_min"] == (0.9, 0.95)
    
    def test_detects_invariant_changes(self):
        """Test detection of system-level invariant changes."""
        system_a = _make_test_system()
        system_b = CurriculumSystem(
            slug="test",
            description="Different description",
            slices=system_a.slices,
            active_index=0,
            monotonic_axes=("atoms",),  # Different monotonic axes
            version=2,
        )
        
        fp_a = compute_fingerprint(system_a)
        fp_b = compute_fingerprint(system_b)
        
        diff = compute_fingerprint_diff(fp_a, fp_b)
        
        assert diff.has_changes
        assert "description" in diff.invariant_diffs
        assert "monotonic_axes" in diff.invariant_diffs


# Invariant Validation Tests
# -----------------------------------------------------------------------------

class TestInvariantValidation:
    """Tests for curriculum invariant validation."""
    
    def test_valid_curriculum_passes(self):
        """Test that a valid curriculum passes all checks."""
        system = _make_test_system()
        report = validate_curriculum_invariants(system)
        
        assert report.valid
        assert len(report.errors) == 0
    
    def test_detects_whitespace_in_slice_name(self):
        """Test detection of whitespace in slice names."""
        slice_with_space = _make_test_slice("slice with space")
        system = CurriculumSystem(
            slug="test",
            description="Test",
            slices=[slice_with_space],
            active_index=0,
            monotonic_axes=(),
            version=2,
        )
        
        report = validate_curriculum_invariants(system)
        
        assert not report.valid
        assert any("whitespace" in err.lower() for err in report.errors)
    
    def test_detects_non_slug_safe_name(self):
        """Test detection of non-slug-safe characters."""
        slice_bad_name = _make_test_slice("slice@bad#name!")
        system = CurriculumSystem(
            slug="test",
            description="Test",
            slices=[slice_bad_name],
            active_index=0,
            monotonic_axes=(),
            version=2,
        )
        
        report = validate_curriculum_invariants(system)
        
        assert not report.valid
        assert any("slug-safe" in err.lower() for err in report.errors)
    
    def test_detects_negative_depth(self):
        """Test detection of negative depth_max."""
        bad_slice = CurriculumSlice(
            name="bad_slice",
            params={"atoms": 3, "depth_max": -1, "breadth_max": 100},
            gates=_make_test_gates(),
        )
        system = CurriculumSystem(
            slug="test",
            description="Test",
            slices=[bad_slice],
            active_index=0,
            monotonic_axes=(),
            version=2,
        )
        
        report = validate_curriculum_invariants(system)
        
        assert not report.valid
        assert any("non-positive depth_max" in err for err in report.errors)
    
    def test_detects_invalid_coverage_threshold(self):
        """Test detection of invalid coverage CI threshold."""
        # This should raise during gate creation
        with pytest.raises(ValueError, match="coverage ci_lower_min"):
            SliceGates(
                coverage=CoverageGateSpec(ci_lower_min=1.5, sample_min=10),  # > 1.0
                abstention=AbstentionGateSpec(max_rate_pct=20.0, max_mass=500),
                velocity=VelocityGateSpec(min_pph=100.0, stability_cv_max=0.15, window_minutes=60),
                caps=CapsGateSpec(min_attempt_mass=1000, min_runtime_minutes=10.0, backlog_max=0.4),
            )
    
    def test_detects_negative_sample_min(self):
        """Test detection of negative sample_min."""
        with pytest.raises(ValueError, match="sample_min must be positive"):
            CoverageGateSpec(ci_lower_min=0.9, sample_min=-5)
    
    def test_detects_excessive_abstention_rate(self):
        """Test detection of abstention rate > 100%."""
        with pytest.raises(ValueError, match="abstention max_rate_pct"):
            AbstentionGateSpec(max_rate_pct=150.0, max_mass=500)
    
    def test_detects_negative_velocity(self):
        """Test detection of negative velocity threshold."""
        with pytest.raises(ValueError, match="velocity min_pph"):
            VelocityGateSpec(min_pph=-10.0, stability_cv_max=0.15, window_minutes=60)
    
    def test_coverage_increase_generates_warning(self):
        """Test that increasing coverage CI generates a warning."""
        slices = [
            _make_test_slice("slice_a", ci_lower=0.85),
            _make_test_slice("slice_b", ci_lower=0.95),  # Increases
        ]
        system = CurriculumSystem(
            slug="test",
            description="Test",
            slices=slices,
            active_index=0,
            monotonic_axes=(),
            version=2,
        )
        
        report = validate_curriculum_invariants(system)
        
        # Should still be valid but have warnings
        assert report.valid
        assert len(report.warnings) > 0
        assert any("coverage CI lower" in warn for warn in report.warnings)


# Promotion Stability Tests
# -----------------------------------------------------------------------------

class TestPromotionStability:
    """Tests for promotion stability evaluation."""
    
    def test_allows_promotion_with_no_changes(self):
        """Test that promotion is allowed when there are no changes."""
        system = _make_test_system()
        fp = compute_fingerprint(system)
        invariants = validate_curriculum_invariants(system)
        
        report = evaluate_curriculum_stability(fp, fp, invariants)
        
        assert report.allow_promotion
        assert "no" in report.reason.lower() and "changes" in report.reason.lower()
    
    def test_allows_promotion_with_minor_changes(self):
        """Test that promotion is allowed with changes within limits."""
        system_a = _make_test_system([_make_test_slice("slice_a", atoms=3)])
        system_b = _make_test_system([_make_test_slice("slice_a", atoms=4)])
        
        fp_a = compute_fingerprint(system_a)
        fp_b = compute_fingerprint(system_b)
        invariants = validate_curriculum_invariants(system_b)
        
        report = evaluate_curriculum_stability(fp_a, fp_b, invariants)
        
        assert report.allow_promotion
        assert report.fingerprint_changes == 1
    
    def test_blocks_promotion_on_invariant_violation(self):
        """Test that promotion is blocked when invariants are violated."""
        system = _make_test_system()
        fp = compute_fingerprint(system)
        
        # Create invalid report
        invalid_invariants = CurriculumInvariantReport(
            valid=False,
            errors=["Test error: invalid configuration"],
        )
        
        report = evaluate_curriculum_stability(fp, fp, invalid_invariants)
        
        assert not report.allow_promotion
        assert "invariant violations" in report.reason.lower()
    
    def test_blocks_promotion_on_removed_slice(self):
        """Test that promotion is blocked when a slice is removed."""
        system_a = _make_test_system([
            _make_test_slice("slice_a"),
            _make_test_slice("slice_b"),
        ])
        system_b = _make_test_system([_make_test_slice("slice_a")])
        
        fp_a = compute_fingerprint(system_a)
        fp_b = compute_fingerprint(system_b)
        invariants = validate_curriculum_invariants(system_b)
        
        report = evaluate_curriculum_stability(fp_a, fp_b, invariants)
        
        assert not report.allow_promotion
        assert "removed" in report.reason.lower()
        assert "slice_b" in report.removed_slices
    
    def test_blocks_promotion_on_too_many_changes(self):
        """Test that promotion is blocked when too many slices change."""
        slices_a = [
            _make_test_slice("slice_a", atoms=3),
            _make_test_slice("slice_b", atoms=4),
            _make_test_slice("slice_c", atoms=5),
            _make_test_slice("slice_d", atoms=6),
        ]
        slices_b = [
            _make_test_slice("slice_a", atoms=4),  # Changed
            _make_test_slice("slice_b", atoms=5),  # Changed
            _make_test_slice("slice_c", atoms=6),  # Changed
            _make_test_slice("slice_d", atoms=7),  # Changed
        ]
        
        system_a = CurriculumSystem(
            slug="test", description="Test", slices=slices_a,
            active_index=0, monotonic_axes=(), version=2,
        )
        system_b = CurriculumSystem(
            slug="test", description="Test", slices=slices_b,
            active_index=0, monotonic_axes=(), version=2,
        )
        
        fp_a = compute_fingerprint(system_a)
        fp_b = compute_fingerprint(system_b)
        invariants = validate_curriculum_invariants(system_b)
        
        report = evaluate_curriculum_stability(
            fp_a, fp_b, invariants, max_slice_changes=3
        )
        
        assert not report.allow_promotion
        assert "too many slices changed" in report.reason.lower()
        assert report.fingerprint_changes == 4
    
    def test_blocks_promotion_on_excessive_gate_change(self):
        """Test that promotion is blocked when gate thresholds change too much."""
        system_a = _make_test_system([_make_test_slice("slice_a", ci_lower=0.90)])
        system_b = _make_test_system([_make_test_slice("slice_a", ci_lower=0.99)])  # +10%
        
        fp_a = compute_fingerprint(system_a)
        fp_b = compute_fingerprint(system_b)
        invariants = validate_curriculum_invariants(system_b)
        
        report = evaluate_curriculum_stability(
            fp_a, fp_b, invariants, max_gate_change_pct=5.0
        )
        
        assert not report.allow_promotion
        assert "gate threshold changed excessively" in report.reason.lower()
        assert len(report.gate_threshold_changes) > 0


# CLI Integration Tests
# -----------------------------------------------------------------------------

class TestCLI:
    """Tests for CLI integration."""
    
    def test_validate_invariants_success(self, capsys):
        """Test validate-invariants command with valid curriculum."""
        exit_code = cli_main(["validate-invariants", "--system", "pl"])
        
        captured = capsys.readouterr()
        assert exit_code == 0
        assert "Valid: True" in captured.out
    
    def test_validate_invariants_handles_nonexistent_system(self):
        """Test validate-invariants with non-existent system."""
        exit_code = cli_main(["validate-invariants", "--system", "nonexistent"])
        
        assert exit_code == 2
    
    def test_stability_envelope_without_baseline(self, capsys):
        """Test stability-envelope command without baseline."""
        exit_code = cli_main(["stability-envelope", "--system", "pl"])
        
        captured = capsys.readouterr()
        assert exit_code == 0
        assert "Curriculum Stability Envelope" in captured.out
    
    def test_stability_envelope_with_fingerprint_save(self, capsys):
        """Test stability-envelope with fingerprint saving."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fp_path = Path(tmpdir) / "fingerprint.json"
            
            exit_code = cli_main([
                "stability-envelope",
                "--system", "pl",
                "--save-fingerprint", str(fp_path),
            ])
            
            captured = capsys.readouterr()
            assert exit_code == 0
            assert fp_path.exists()
            
            # Verify fingerprint is valid JSON
            with open(fp_path) as f:
                fp = json.load(f)
                assert "slug" in fp
                assert "slices" in fp
    
    def test_diff_fingerprint_identical(self, capsys):
        """Test diff-fingerprint with identical fingerprints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fp_path_a = Path(tmpdir) / "fp_a.json"
            fp_path_b = Path(tmpdir) / "fp_b.json"
            
            # Create identical fingerprints
            system = _make_test_system()
            fp = compute_fingerprint(system)
            
            with open(fp_path_a, 'w') as f:
                json.dump(fp, f)
            with open(fp_path_b, 'w') as f:
                json.dump(fp, f)
            
            exit_code = cli_main([
                "diff-fingerprint",
                str(fp_path_a),
                str(fp_path_b),
            ])
            
            captured = capsys.readouterr()
            assert exit_code == 0
            assert "No changes detected" in captured.out
    
    def test_diff_fingerprint_with_changes(self, capsys):
        """Test diff-fingerprint with actual changes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fp_path_a = Path(tmpdir) / "fp_a.json"
            fp_path_b = Path(tmpdir) / "fp_b.json"
            
            # Create different fingerprints
            system_a = _make_test_system([_make_test_slice("slice_a", atoms=3)])
            system_b = _make_test_system([_make_test_slice("slice_a", atoms=4)])
            
            with open(fp_path_a, 'w') as f:
                json.dump(compute_fingerprint(system_a), f)
            with open(fp_path_b, 'w') as f:
                json.dump(compute_fingerprint(system_b), f)
            
            exit_code = cli_main([
                "diff-fingerprint",
                str(fp_path_a),
                str(fp_path_b),
            ])
            
            captured = capsys.readouterr()
            assert exit_code == 0
            assert "Changed Slices" in captured.out
            assert "slice_a" in captured.out
            assert "atoms" in captured.out
    
    def test_diff_fingerprint_json_output(self, capsys):
        """Test diff-fingerprint with JSON output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fp_path_a = Path(tmpdir) / "fp_a.json"
            fp_path_b = Path(tmpdir) / "fp_b.json"
            
            system_a = _make_test_system([_make_test_slice("slice_a")])
            system_b = _make_test_system([
                _make_test_slice("slice_a"),
                _make_test_slice("slice_b"),
            ])
            
            with open(fp_path_a, 'w') as f:
                json.dump(compute_fingerprint(system_a), f)
            with open(fp_path_b, 'w') as f:
                json.dump(compute_fingerprint(system_b), f)
            
            exit_code = cli_main([
                "diff-fingerprint",
                str(fp_path_a),
                str(fp_path_b),
                "--json",
            ])
            
            captured = capsys.readouterr()
            assert exit_code == 0
            assert "JSON Output" in captured.out
            
            # Check that valid JSON is present
            assert '"added_slices"' in captured.out
            assert '"slice_b"' in captured.out


# Mixed Scenarios
# -----------------------------------------------------------------------------

class TestMixedScenarios:
    """Tests for mixed drift and invariant violation scenarios."""
    
    def test_drift_with_invariant_violation(self):
        """Test handling of drift combined with invariant violations."""
        # Create a system with a bad slice name and parameter changes
        bad_slice = CurriculumSlice(
            name="bad slice name",  # Whitespace violation
            params={"atoms": 4, "depth_max": -1},  # Negative depth violation
            gates=_make_test_gates(),
        )
        
        system_a = _make_test_system()
        system_b = CurriculumSystem(
            slug="test",
            description="Test",
            slices=[bad_slice],
            active_index=0,
            monotonic_axes=(),
            version=2,
        )
        
        fp_a = compute_fingerprint(system_a)
        fp_b = compute_fingerprint(system_b)
        
        # Invariant check should fail
        invariants = validate_curriculum_invariants(system_b)
        assert not invariants.valid
        
        # Promotion should be blocked due to invariants
        report = evaluate_curriculum_stability(fp_a, fp_b, invariants)
        assert not report.allow_promotion
        assert "invariant" in report.reason.lower()
    
    def test_multiple_simultaneous_violations(self):
        """Test handling of multiple simultaneous violations."""
        slices_a = [
            _make_test_slice("slice_a", atoms=3),
            _make_test_slice("slice_b", atoms=4),
            _make_test_slice("slice_c", atoms=5),
        ]
        slices_b = [
            _make_test_slice("slice_a", atoms=10),  # Param change
            _make_test_slice("slice_b", atoms=11),  # Param change
            _make_test_slice("slice_c", atoms=12),  # Param change
            _make_test_slice("slice_d", atoms=13),  # Added (counts as change)
        ]
        
        system_a = CurriculumSystem(
            slug="test", description="Test", slices=slices_a,
            active_index=0, monotonic_axes=(), version=2,
        )
        system_b = CurriculumSystem(
            slug="test", description="Test", slices=slices_b,
            active_index=0, monotonic_axes=(), version=2,
        )
        
        fp_a = compute_fingerprint(system_a)
        fp_b = compute_fingerprint(system_b)
        invariants = validate_curriculum_invariants(system_b)
        
        # Should block due to too many changes (3 changed slices)
        report = evaluate_curriculum_stability(
            fp_a, fp_b, invariants, max_slice_changes=2
        )
        
        assert not report.allow_promotion
        assert report.fingerprint_changes == 3
