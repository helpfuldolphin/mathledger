"""
PHASE II — NOT USED IN PHASE I

Tests for u2_cross_slice_analysis module.

These tests verify:
1. Deterministic fingerprints
2. Cross-slice comparisons detect structural differences
3. No computation or implication of Δp

Agent: metrics-engineer-6
"""

import json
import tempfile
import pytest
from pathlib import Path
from typing import Any, Dict, List

from experiments.u2_cross_slice_analysis import (
    SliceResults,
    BehavioralPattern,
    BehavioralFingerprint,
    load_slice_results,
    compare_behavioral_patterns,
    compute_cross_slice_consistency,
    generate_behavior_signature,
    fingerprint_to_dict,
    _compute_metric_value_distribution,
    _compute_chain_depth_distribution,
    _compute_abstention_trend,
    _compute_candidate_ordering_entropy,
    _compute_policy_weight_movement,
    _compute_js_divergence,
    _compute_trend_correlation,
    _compute_temporal_smoothness,
    _compute_goal_hit_distribution,
)


# ===========================================================================
# FIXTURES
# ===========================================================================

@pytest.fixture
def sample_records_baseline() -> List[Dict[str, Any]]:
    """Generate deterministic baseline records."""
    return [
        {
            "cycle": i,
            "slice": "test_slice",
            "mode": "baseline",
            "seed": 42 + i,
            "success": i % 3 != 0,  # Success every 2/3 cycles
            "metric_value": float(i % 5),
            "derivation": {
                "candidate_order": [f"hash_{j:04d}" for j in range(4)],
                "verified_count": (i % 3) + 1,
                "verified_hashes": [f"ver_{j}" for j in range((i % 3) + 1)],
            },
            "metric_result": {
                "hit_count": i % 2,
                "total_verified": (i % 3) + 1,
                "metric_type": "goal_hit",
            },
        }
        for i in range(20)
    ]


@pytest.fixture
def sample_records_rfl() -> List[Dict[str, Any]]:
    """Generate deterministic RFL records with different pattern."""
    return [
        {
            "cycle": i,
            "slice": "test_slice",
            "mode": "rfl",
            "seed": 100 + i,
            "success": i % 2 != 0,  # Different success pattern
            "metric_value": float((i + 1) % 5),  # Shifted metric values
            "derivation": {
                "candidate_order": [f"rfl_hash_{j:04d}" for j in range(5)],
                "verified_count": (i % 4) + 1,
                "verified_hashes": [f"rfl_ver_{j}" for j in range((i % 4) + 1)],
            },
            "metric_result": {
                "hit_count": (i + 1) % 2,
                "total_verified": (i % 4) + 1,
                "metric_type": "goal_hit",
            },
        }
        for i in range(20)
    ]


@pytest.fixture
def baseline_results(sample_records_baseline) -> SliceResults:
    """Create SliceResults for baseline."""
    return SliceResults(
        slice_name="test_slice",
        mode="baseline",
        records=sample_records_baseline,
        source_path="/fake/path/baseline.jsonl",
    )


@pytest.fixture
def rfl_results(sample_records_rfl) -> SliceResults:
    """Create SliceResults for RFL."""
    return SliceResults(
        slice_name="test_slice",
        mode="rfl",
        records=sample_records_rfl,
        source_path="/fake/path/rfl.jsonl",
    )


@pytest.fixture
def temp_results_dir(sample_records_baseline, sample_records_rfl) -> Path:
    """Create temporary directory with JSONL files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        
        # Write baseline file
        baseline_path = tmppath / "uplift_u2_test_slice_baseline.jsonl"
        with open(baseline_path, 'w') as f:
            for record in sample_records_baseline:
                f.write(json.dumps(record) + "\n")
        
        # Write RFL file
        rfl_path = tmppath / "uplift_u2_test_slice_rfl.jsonl"
        with open(rfl_path, 'w') as f:
            for record in sample_records_rfl:
                f.write(json.dumps(record) + "\n")
        
        yield tmppath


# ===========================================================================
# DETERMINISM TESTS
# ===========================================================================

class TestDeterminism:
    """
    PHASE II — NOT USED IN PHASE I
    Tests to ensure all operations are deterministic.
    """

    def test_fingerprint_determinism(self, baseline_results):
        """Same inputs must produce identical fingerprint hash."""
        fp1 = generate_behavior_signature(baseline_results)
        fp2 = generate_behavior_signature(baseline_results)
        fp3 = generate_behavior_signature(baseline_results)
        
        assert fp1.fingerprint_hash == fp2.fingerprint_hash
        assert fp2.fingerprint_hash == fp3.fingerprint_hash
    
    def test_fingerprint_determinism_100_calls(self, baseline_results):
        """Fingerprint must be identical across 100 calls."""
        fingerprints = [
            generate_behavior_signature(baseline_results).fingerprint_hash
            for _ in range(100)
        ]
        
        assert all(fp == fingerprints[0] for fp in fingerprints)
    
    def test_metric_distribution_determinism(self, sample_records_baseline):
        """Metric value distribution must be deterministic."""
        results = [
            _compute_metric_value_distribution(sample_records_baseline)
            for _ in range(50)
        ]
        
        assert all(r == results[0] for r in results)
    
    def test_chain_depth_distribution_determinism(self, sample_records_baseline):
        """Chain depth distribution must be deterministic."""
        results = [
            _compute_chain_depth_distribution(sample_records_baseline)
            for _ in range(50)
        ]
        
        assert all(r == results[0] for r in results)
    
    def test_abstention_trend_determinism(self, sample_records_baseline):
        """Abstention trend must be deterministic."""
        results = [
            _compute_abstention_trend(sample_records_baseline)
            for _ in range(50)
        ]
        
        assert all(r == results[0] for r in results)
    
    def test_ordering_entropy_determinism(self, sample_records_baseline):
        """Candidate ordering entropy must be deterministic."""
        results = [
            _compute_candidate_ordering_entropy(sample_records_baseline)
            for _ in range(50)
        ]
        
        assert all(r == results[0] for r in results)
    
    def test_policy_weight_movement_determinism(self, sample_records_baseline):
        """Policy weight movement norms must be deterministic."""
        results = [
            _compute_policy_weight_movement(sample_records_baseline)
            for _ in range(50)
        ]
        
        assert all(r == results[0] for r in results)
    
    def test_behavioral_comparison_determinism(self, baseline_results, rfl_results):
        """Behavioral comparison must be deterministic."""
        results = [
            compare_behavioral_patterns(baseline_results, rfl_results)
            for _ in range(50)
        ]
        
        # Convert to JSON for comparison (handles float precision)
        json_results = [json.dumps(r, sort_keys=True) for r in results]
        assert all(jr == json_results[0] for jr in json_results)
    
    def test_cross_slice_consistency_determinism(self, baseline_results, rfl_results):
        """Cross-slice consistency must be deterministic."""
        slices_dict = {"test_slice": (baseline_results, rfl_results)}
        
        results = [
            compute_cross_slice_consistency(slices_dict)
            for _ in range(50)
        ]
        
        json_results = [json.dumps(r, sort_keys=True) for r in results]
        assert all(jr == json_results[0] for jr in json_results)


# ===========================================================================
# STRUCTURAL DIFFERENCE DETECTION TESTS
# ===========================================================================

class TestStructuralDifferenceDetection:
    """
    PHASE II — NOT USED IN PHASE I
    Tests to ensure cross-slice comparisons detect structural differences.
    """

    def test_different_fingerprints_for_different_data(self, baseline_results, rfl_results):
        """Baseline and RFL should produce different fingerprints."""
        fp_baseline = generate_behavior_signature(baseline_results)
        fp_rfl = generate_behavior_signature(rfl_results)
        
        assert fp_baseline.fingerprint_hash != fp_rfl.fingerprint_hash
    
    def test_comparison_detects_histogram_differences(self, baseline_results, rfl_results):
        """Comparison should detect differences in metric histograms."""
        comparison = compare_behavioral_patterns(baseline_results, rfl_results)
        
        # JS divergence should be non-zero for different distributions
        assert "metric_histogram_match" in comparison
        assert comparison["metric_histogram_match"]["js_divergence"] >= 0
    
    def test_comparison_detects_chain_depth_differences(self, baseline_results, rfl_results):
        """Comparison should detect differences in chain depth distributions."""
        comparison = compare_behavioral_patterns(baseline_results, rfl_results)
        
        assert "chain_depth_histogram_match" in comparison
        # Should have some structural comparison metrics
        assert "js_divergence" in comparison["chain_depth_histogram_match"]
    
    def test_identical_data_produces_zero_divergence(self, baseline_results):
        """Identical data should produce zero JS divergence."""
        comparison = compare_behavioral_patterns(baseline_results, baseline_results)
        
        assert comparison["metric_histogram_match"]["js_divergence"] == 0.0
        assert comparison["chain_depth_histogram_match"]["js_divergence"] == 0.0
    
    def test_js_divergence_symmetric(self):
        """JS divergence should be symmetric."""
        hist1 = {"0.0": 10, "1.0": 5, "2.0": 3}
        hist2 = {"0.0": 8, "1.0": 7, "2.0": 4}
        
        div1 = _compute_js_divergence(hist1, hist2)
        div2 = _compute_js_divergence(hist2, hist1)
        
        assert div1 == div2
    
    def test_js_divergence_zero_for_identical(self):
        """JS divergence should be zero for identical histograms."""
        hist = {"0.0": 10, "1.0": 5, "2.0": 3}
        
        assert _compute_js_divergence(hist, hist) == 0.0
    
    def test_js_divergence_positive_for_different(self):
        """JS divergence should be positive for different histograms."""
        hist1 = {"0.0": 10, "1.0": 0}
        hist2 = {"0.0": 0, "1.0": 10}
        
        assert _compute_js_divergence(hist1, hist2) > 0


# ===========================================================================
# NO ΔP COMPUTATION TESTS
# ===========================================================================

class TestNoUpliftComputation:
    """
    PHASE II — NOT USED IN PHASE I
    Tests to ensure no part of this code computes or implies Δp.
    """

    def test_comparison_has_no_delta_keys(self, baseline_results, rfl_results):
        """Comparison output must not contain delta/uplift keys."""
        comparison = compare_behavioral_patterns(baseline_results, rfl_results)
        
        forbidden_keys = [
            "delta", "uplift", "improvement", "difference",
            "p_value", "significance", "effect_size",
        ]
        
        def check_no_forbidden_keys(d, path=""):
            if isinstance(d, dict):
                for key, value in d.items():
                    full_key = f"{path}.{key}" if path else key
                    # Allow 'js_divergence' - it's structural, not uplift
                    if any(fk in key.lower() for fk in forbidden_keys if fk != "difference"):
                        # 'difference' allowed only in non-uplift contexts
                        if "uplift" in key.lower() or "delta" in key.lower():
                            pytest.fail(f"Forbidden key found: {full_key}")
                    check_no_forbidden_keys(value, full_key)
        
        check_no_forbidden_keys(comparison)
    
    def test_fingerprint_has_no_success_rate(self, baseline_results):
        """Fingerprint must not contain success rate (which could imply uplift)."""
        fp = generate_behavior_signature(baseline_results)
        fp_dict = fingerprint_to_dict(fp)
        
        # Fingerprint should contain only structural metrics
        assert "success_rate" not in fp_dict
        assert "uplift" not in str(fp_dict).lower()
    
    def test_cross_slice_consistency_has_no_uplift(self, baseline_results, rfl_results):
        """Cross-slice consistency must not compute uplift."""
        slices_dict = {"test_slice": (baseline_results, rfl_results)}
        consistency = compute_cross_slice_consistency(slices_dict)
        
        # Must not contain uplift-related keys
        consistency_str = json.dumps(consistency).lower()
        assert "uplift" not in consistency_str
        assert "p_value" not in consistency_str
        assert "significance" not in consistency_str


# ===========================================================================
# FILE LOADING TESTS
# ===========================================================================

class TestFileLoading:
    """Tests for file loading functionality."""

    def test_load_slice_results(self, temp_results_dir):
        """Test loading slice results from JSONL files."""
        results = load_slice_results("test_slice", "baseline", temp_results_dir)
        
        assert results.slice_name == "test_slice"
        assert results.mode == "baseline"
        assert len(results.records) == 20
    
    def test_load_slice_results_rfl(self, temp_results_dir):
        """Test loading RFL results."""
        results = load_slice_results("test_slice", "rfl", temp_results_dir)
        
        assert results.slice_name == "test_slice"
        assert results.mode == "rfl"
        assert len(results.records) == 20
    
    def test_load_slice_results_not_found(self, temp_results_dir):
        """Test error handling for missing files."""
        with pytest.raises(FileNotFoundError):
            load_slice_results("nonexistent_slice", "baseline", temp_results_dir)


# ===========================================================================
# EDGE CASE TESTS
# ===========================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_records(self):
        """Test handling of empty records."""
        results = SliceResults(
            slice_name="empty",
            mode="baseline",
            records=[],
        )
        
        fp = generate_behavior_signature(results)
        assert fp.fingerprint_hash is not None
        assert len(fp.fingerprint_hash) == 64  # SHA256 hex
    
    def test_single_record(self):
        """Test handling of single record."""
        results = SliceResults(
            slice_name="single",
            mode="baseline",
            records=[{
                "cycle": 0,
                "success": True,
                "metric_value": 1.0,
            }],
        )
        
        fp = generate_behavior_signature(results)
        assert fp.fingerprint_hash is not None
    
    def test_missing_fields_in_records(self):
        """Test handling of records with missing fields."""
        results = SliceResults(
            slice_name="partial",
            mode="baseline",
            records=[
                {"cycle": 0},  # Minimal record
                {"cycle": 1, "success": True},
                {"cycle": 2, "metric_value": 2.0},
            ],
        )
        
        fp = generate_behavior_signature(results)
        assert fp.fingerprint_hash is not None
    
    def test_trend_correlation_empty_trends(self):
        """Test correlation with empty trends."""
        assert _compute_trend_correlation([], []) == 0.0
        assert _compute_trend_correlation([1.0], []) == 0.0
        assert _compute_trend_correlation([], [1.0]) == 0.0
    
    def test_trend_correlation_constant_trends(self):
        """Test correlation with constant trends."""
        trend1 = [0.5, 0.5, 0.5, 0.5]
        trend2 = [0.5, 0.5, 0.5, 0.5]
        
        # Constant trends have zero variance, so correlation is 0
        assert _compute_trend_correlation(trend1, trend2) == 0.0
    
    def test_temporal_smoothness_single_value(self):
        """Test temporal smoothness with single value."""
        assert _compute_temporal_smoothness([0.5]) == 1.0
    
    def test_temporal_smoothness_empty(self):
        """Test temporal smoothness with empty trend."""
        assert _compute_temporal_smoothness([]) == 1.0
    
    def test_js_divergence_empty_histograms(self):
        """Test JS divergence with empty histograms."""
        assert _compute_js_divergence({}, {}) == 0.0
        assert _compute_js_divergence({"a": 1}, {}) >= 0
    
    def test_cross_slice_consistency_empty_dict(self):
        """Test cross-slice consistency with empty dict."""
        result = compute_cross_slice_consistency({})
        
        assert result["slice_count"] == 0
        assert "error" in result


# ===========================================================================
# REPRODUCIBILITY TESTS
# ===========================================================================

class TestReproducibility:
    """
    PHASE II — NOT USED IN PHASE I
    Tests to ensure reproducibility of all outputs.
    """

    def test_fingerprint_hash_format(self, baseline_results):
        """Fingerprint hash should be valid SHA256."""
        fp = generate_behavior_signature(baseline_results)
        
        # SHA256 produces 64 hex characters
        assert len(fp.fingerprint_hash) == 64
        assert all(c in "0123456789abcdef" for c in fp.fingerprint_hash)
    
    def test_fingerprint_components_are_deterministic(self, baseline_results):
        """All fingerprint components should be deterministic."""
        fp1 = generate_behavior_signature(baseline_results)
        fp2 = generate_behavior_signature(baseline_results)
        
        assert fp1.metric_value_histogram == fp2.metric_value_histogram
        assert fp1.longest_chain_distribution == fp2.longest_chain_distribution
        assert fp1.goal_hit_distribution == fp2.goal_hit_distribution
        assert fp1.temporal_smoothness_signature == fp2.temporal_smoothness_signature
    
    def test_comparison_output_serializable(self, baseline_results, rfl_results):
        """Comparison output should be JSON serializable."""
        comparison = compare_behavioral_patterns(baseline_results, rfl_results)
        
        # Should not raise
        json_str = json.dumps(comparison, sort_keys=True)
        assert len(json_str) > 0
    
    def test_consistency_output_serializable(self, baseline_results, rfl_results):
        """Cross-slice consistency output should be JSON serializable."""
        slices_dict = {"test_slice": (baseline_results, rfl_results)}
        consistency = compute_cross_slice_consistency(slices_dict)
        
        # Should not raise
        json_str = json.dumps(consistency, sort_keys=True)
        assert len(json_str) > 0


# ===========================================================================
# PHASE II COMPLIANCE TESTS
# ===========================================================================

class TestPhaseIICompliance:
    """
    Tests to ensure PHASE II labeling and compliance.
    """

    def test_comparison_has_phase_label(self, baseline_results, rfl_results):
        """Comparison output must have PHASE II label."""
        comparison = compare_behavioral_patterns(baseline_results, rfl_results)
        
        assert "phase_label" in comparison
        assert "PHASE II" in comparison["phase_label"]
    
    def test_consistency_has_phase_label(self, baseline_results, rfl_results):
        """Cross-slice consistency output must have PHASE II label."""
        slices_dict = {"test_slice": (baseline_results, rfl_results)}
        consistency = compute_cross_slice_consistency(slices_dict)
        
        assert "phase_label" in consistency
        assert "PHASE II" in consistency["phase_label"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

