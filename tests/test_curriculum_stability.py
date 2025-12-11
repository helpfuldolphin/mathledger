"""
Tests for Curriculum Stability Envelope

Verifies HSS computation, variance tracking, suitability scoring,
and integration adapters for P3/P4/Evidence/Council.
"""

import json
import pytest
from curriculum.stability import (
    compute_hss,
    compute_variance_metric,
    compute_suitability_score,
    build_stability_envelope,
    attach_curriculum_stability_to_evidence,
    summarize_curriculum_stability_for_council,
    add_stability_to_first_light,
    add_stability_to_p4_calibration,
    CurriculumStabilityEnvelope,
    SliceHealthMetrics,
)


class TestHSSComputation:
    """Test HSS (Homogeneity-Stability Score) computation."""
    
    def test_hss_basic(self):
        """Test basic HSS computation with minimal metrics."""
        slice_metrics = {
            "slice_name": "test_slice_a",
            "params": {"atoms": 5, "depth_max": 6, "breadth_max": 1500},
            "coverage_rate": 0.85,
            "abstention_rate": 0.10,
        }
        
        hss = compute_hss(slice_metrics)
        
        assert 0.0 <= hss <= 1.0
        assert hss > 0.5  # Should be decent with these parameters
    
    def test_hss_with_history(self):
        """Test HSS computation with historical data."""
        slice_metrics = {
            "slice_name": "test_slice_b",
            "params": {"atoms": 4, "depth_max": 5, "breadth_max": 1000},
            "coverage_rate": 0.90,
            "abstention_rate": 0.05,
        }
        
        historical = [
            {"coverage_rate": 0.88, "abstention_rate": 0.06},
            {"coverage_rate": 0.89, "abstention_rate": 0.05},
            {"coverage_rate": 0.90, "abstention_rate": 0.05},
        ]
        
        hss = compute_hss(slice_metrics, historical)
        
        assert 0.0 <= hss <= 1.0
        # With stable history, HSS should be high
        assert hss > 0.6
    
    def test_hss_unstable_history(self):
        """Test HSS with unstable historical metrics."""
        slice_metrics = {
            "slice_name": "test_slice_c",
            "params": {"atoms": 6, "depth_max": 8, "breadth_max": 2000},
            "coverage_rate": 0.75,
            "abstention_rate": 0.20,
        }
        
        # Highly variable history
        historical = [
            {"coverage_rate": 0.90, "abstention_rate": 0.05},
            {"coverage_rate": 0.60, "abstention_rate": 0.30},
            {"coverage_rate": 0.75, "abstention_rate": 0.20},
        ]
        
        hss = compute_hss(slice_metrics, historical)
        
        assert 0.0 <= hss <= 1.0
        # Unstable history should lower HSS
        assert hss < 0.7


class TestVarianceMetric:
    """Test variance metric computation."""
    
    def test_variance_stable(self):
        """Test variance for stable metrics."""
        slice_metrics = {"slice_name": "stable_slice"}
        
        historical = [
            {"coverage_rate": 0.88},
            {"coverage_rate": 0.89},
            {"coverage_rate": 0.90},
            {"coverage_rate": 0.89},
        ]
        
        variance = compute_variance_metric(slice_metrics, historical)
        
        assert 0.0 <= variance <= 1.0
        # Stable metrics should have low variance
        assert variance < 0.3
    
    def test_variance_unstable(self):
        """Test variance for unstable metrics."""
        slice_metrics = {"slice_name": "unstable_slice"}
        
        historical = [
            {"coverage_rate": 0.95},
            {"coverage_rate": 0.60},
            {"coverage_rate": 0.85},
            {"coverage_rate": 0.50},
        ]
        
        variance = compute_variance_metric(slice_metrics, historical)
        
        assert 0.0 <= variance <= 1.0
        # Unstable metrics should have high variance
        assert variance > 0.3
    
    def test_variance_insufficient_data(self):
        """Test variance with insufficient historical data."""
        slice_metrics = {"slice_name": "new_slice"}
        historical = []
        
        variance = compute_variance_metric(slice_metrics, historical)
        
        # Should return neutral value
        assert variance == 0.5


class TestSuitabilityScore:
    """Test suitability score computation."""
    
    def test_suitability_high(self):
        """Test high suitability with good metrics."""
        slice_metrics = {
            "coverage_rate": 0.90,
            "abstention_rate": 0.05,
        }
        
        hss = 0.85
        variance = 0.10
        
        suitability = compute_suitability_score("test_slice", hss, variance, slice_metrics)
        
        assert 0.0 <= suitability <= 1.0
        assert suitability > 0.7
    
    def test_suitability_low(self):
        """Test low suitability with poor metrics."""
        slice_metrics = {
            "coverage_rate": 0.50,
            "abstention_rate": 0.30,
        }
        
        hss = 0.40
        variance = 0.60
        
        suitability = compute_suitability_score("test_slice", hss, variance, slice_metrics)
        
        assert 0.0 <= suitability <= 1.0
        assert suitability < 0.5


class TestStabilityEnvelope:
    """Test curriculum stability envelope construction."""
    
    def test_envelope_single_slice(self):
        """Test envelope with single slice."""
        slice_metrics_list = [
            {
                "slice_name": "slice_a",
                "params": {"atoms": 5, "depth_max": 6, "breadth_max": 1500},
                "coverage_rate": 0.85,
                "abstention_rate": 0.10,
            }
        ]
        
        envelope = build_stability_envelope(slice_metrics_list)
        
        assert isinstance(envelope, CurriculumStabilityEnvelope)
        assert 0.0 <= envelope.mean_hss <= 1.0
        assert envelope.status_light in ["GREEN", "YELLOW", "RED"]
        assert "slice_a" in envelope.suitability_scores
    
    def test_envelope_multiple_slices(self):
        """Test envelope with multiple slices."""
        slice_metrics_list = [
            {
                "slice_name": "slice_a",
                "params": {"atoms": 4, "depth_max": 5, "breadth_max": 1000},
                "coverage_rate": 0.90,
                "abstention_rate": 0.05,
            },
            {
                "slice_name": "slice_b",
                "params": {"atoms": 5, "depth_max": 6, "breadth_max": 1500},
                "coverage_rate": 0.85,
                "abstention_rate": 0.10,
            },
            {
                "slice_name": "slice_c",
                "params": {"atoms": 6, "depth_max": 8, "breadth_max": 2000},
                "coverage_rate": 0.80,
                "abstention_rate": 0.15,
            },
        ]
        
        envelope = build_stability_envelope(slice_metrics_list)
        
        assert len(envelope.suitability_scores) == 3
        assert envelope.mean_hss > 0
        assert envelope.status_light in ["GREEN", "YELLOW", "RED"]
    
    def test_envelope_flagging(self):
        """Test that low-suitability slices are flagged."""
        slice_metrics_list = [
            {
                "slice_name": "good_slice",
                "params": {"atoms": 5, "depth_max": 6, "breadth_max": 1500},
                "coverage_rate": 0.90,
                "abstention_rate": 0.05,
            },
            {
                "slice_name": "bad_slice",
                "params": {"atoms": 8, "depth_max": 10, "breadth_max": 3500},
                "coverage_rate": 0.40,
                "abstention_rate": 0.50,
            },
        ]
        
        envelope = build_stability_envelope(slice_metrics_list)
        
        # Bad slice should be flagged
        assert "bad_slice" in envelope.slices_flagged or "bad_slice" in envelope.unstable_slices
        assert envelope.status_light in ["YELLOW", "RED"]
    
    def test_envelope_json_serializable(self):
        """Test that envelope can be serialized to JSON."""
        slice_metrics_list = [
            {
                "slice_name": "slice_a",
                "params": {"atoms": 5, "depth_max": 6, "breadth_max": 1500},
                "coverage_rate": 0.85,
                "abstention_rate": 0.10,
            }
        ]
        
        envelope = build_stability_envelope(slice_metrics_list)
        envelope_dict = envelope.to_dict()
        
        # Should be JSON-serializable
        json_str = json.dumps(envelope_dict)
        assert json_str is not None
        
        # Verify structure
        assert "mean_HSS" in envelope_dict
        assert "HSS_variance" in envelope_dict
        assert "status_light" in envelope_dict
        assert "suitability_scores" in envelope_dict


class TestEvidencePackAdapter:
    """Test evidence pack adapter."""
    
    def test_attach_non_mutating(self):
        """Test that attachment doesn't mutate original evidence."""
        original_evidence = {
            "experiment_id": "test_001",
            "results": {"coverage": 0.85},
            "governance": {"version": "1.0"},
        }
        
        envelope = CurriculumStabilityEnvelope(
            mean_hss=0.75,
            hss_variance=0.05,
            low_hss_fraction=0.2,
            slices_flagged=["slice_b"],
            suitability_scores={"slice_a": 0.85, "slice_b": 0.55},
            status_light="YELLOW",
        )
        
        new_evidence = attach_curriculum_stability_to_evidence(original_evidence, envelope)
        
        # Original should be unchanged
        assert "curriculum_stability" not in original_evidence.get("governance", {})
        
        # New evidence should have stability
        assert "curriculum_stability" in new_evidence["governance"]
        assert new_evidence["governance"]["curriculum_stability"]["status_light"] == "YELLOW"
    
    def test_attach_creates_governance(self):
        """Test that attachment creates governance section if missing."""
        original_evidence = {
            "experiment_id": "test_002",
        }
        
        envelope = CurriculumStabilityEnvelope(
            mean_hss=0.80,
            hss_variance=0.03,
            low_hss_fraction=0.0,
            slices_flagged=[],
            suitability_scores={"slice_a": 0.90},
            status_light="GREEN",
        )
        
        new_evidence = attach_curriculum_stability_to_evidence(original_evidence, envelope)
        
        assert "governance" in new_evidence
        assert "curriculum_stability" in new_evidence["governance"]
    
    def test_attach_limited_fields(self):
        """Test that only specified fields are attached."""
        original_evidence = {"experiment_id": "test_003"}
        
        envelope = CurriculumStabilityEnvelope(
            mean_hss=0.75,
            hss_variance=0.10,
            low_hss_fraction=0.1,
            slices_flagged=["slice_x"],
            suitability_scores={"slice_x": 0.50, "slice_y": 0.80},
            status_light="YELLOW",
            stable_slices=["slice_y"],
            unstable_slices=["slice_x"],
        )
        
        new_evidence = attach_curriculum_stability_to_evidence(original_evidence, envelope)
        
        stability = new_evidence["governance"]["curriculum_stability"]
        
        # Should have only these fields
        assert "status_light" in stability
        assert "slices_flagged" in stability
        assert "suitability_scores" in stability
        
        # Should NOT have these
        assert "mean_HSS" not in stability
        assert "stable_slices" not in stability
    
    def test_attach_json_serializable(self):
        """Test that attached evidence is JSON-serializable."""
        original_evidence = {"experiment_id": "test_004"}
        
        envelope = CurriculumStabilityEnvelope(
            mean_hss=0.80,
            hss_variance=0.05,
            low_hss_fraction=0.0,
            slices_flagged=[],
            suitability_scores={"slice_a": 0.90},
            status_light="GREEN",
        )
        
        new_evidence = attach_curriculum_stability_to_evidence(original_evidence, envelope)
        
        # Should serialize without error
        json_str = json.dumps(new_evidence)
        assert json_str is not None


class TestUpliftCouncilAdapter:
    """Test Uplift Council adapter."""
    
    def test_council_ok_status(self):
        """Test OK status for healthy curriculum."""
        envelope = CurriculumStabilityEnvelope(
            mean_hss=0.85,
            hss_variance=0.03,
            low_hss_fraction=0.0,
            slices_flagged=[],
            suitability_scores={"slice_a": 0.90, "slice_b": 0.85},
            status_light="GREEN",
        )
        
        advisory = summarize_curriculum_stability_for_council(envelope)
        
        assert advisory["status"] == "OK"
        assert len(advisory["blocked_slices"]) == 0
        assert len(advisory["marginal_slices"]) == 0
    
    def test_council_warn_status(self):
        """Test WARN status for marginal slices."""
        envelope = CurriculumStabilityEnvelope(
            mean_hss=0.70,
            hss_variance=0.08,
            low_hss_fraction=0.1,
            slices_flagged=["slice_b"],
            suitability_scores={"slice_a": 0.75, "slice_b": 0.55},
            status_light="YELLOW",
        )
        
        advisory = summarize_curriculum_stability_for_council(envelope)
        
        assert advisory["status"] == "WARN"
        assert "slice_b" in advisory["marginal_slices"]
    
    def test_council_block_status(self):
        """Test BLOCK status for critical slices."""
        envelope = CurriculumStabilityEnvelope(
            mean_hss=0.50,
            hss_variance=0.15,
            low_hss_fraction=0.5,
            slices_flagged=["slice_c"],
            suitability_scores={"slice_a": 0.80, "slice_c": 0.25},
            status_light="RED",
        )
        
        advisory = summarize_curriculum_stability_for_council(envelope)
        
        assert advisory["status"] == "BLOCK"
        assert "slice_c" in advisory["blocked_slices"]
    
    def test_council_includes_metrics(self):
        """Test that council advisory includes key metrics."""
        envelope = CurriculumStabilityEnvelope(
            mean_hss=0.75,
            hss_variance=0.06,
            low_hss_fraction=0.1,
            slices_flagged=[],
            suitability_scores={"slice_a": 0.80},
            status_light="GREEN",
        )
        
        advisory = summarize_curriculum_stability_for_council(envelope)
        
        assert "mean_hss" in advisory
        assert "hss_variance" in advisory
        assert "status_light" in advisory
        assert advisory["mean_hss"] == 0.75


class TestFirstLightBinding:
    """Test P3 First Light binding."""
    
    def test_first_light_binding(self):
        """Test adding stability to First Light summary."""
        first_light = {
            "experiment_id": "first_light_001",
            "timestamp": "2025-12-11T00:00:00Z",
            "results": {"coverage": 0.85},
        }
        
        envelope = CurriculumStabilityEnvelope(
            mean_hss=0.80,
            hss_variance=0.05,
            low_hss_fraction=0.1,
            slices_flagged=["slice_b"],
            suitability_scores={"slice_a": 0.85, "slice_b": 0.60},
            status_light="YELLOW",
        )
        
        updated = add_stability_to_first_light(first_light, envelope)
        
        assert "curriculum_stability_envelope" in updated
        assert updated["curriculum_stability_envelope"]["status_light"] == "YELLOW"
        assert "mean_HSS" in updated["curriculum_stability_envelope"]
    
    def test_first_light_deterministic(self):
        """Test that First Light output is deterministic."""
        first_light = {"experiment_id": "test"}
        
        envelope = CurriculumStabilityEnvelope(
            mean_hss=0.75,
            hss_variance=0.08,
            low_hss_fraction=0.2,
            slices_flagged=["slice_x"],
            suitability_scores={"slice_x": 0.50},
            status_light="YELLOW",
        )
        
        result1 = add_stability_to_first_light(dict(first_light), envelope)
        result2 = add_stability_to_first_light(dict(first_light), envelope)
        
        # Should be identical
        assert json.dumps(result1, sort_keys=True) == json.dumps(result2, sort_keys=True)


class TestP4CalibrationBinding:
    """Test P4 calibration binding."""
    
    def test_p4_calibration_binding(self):
        """Test adding stability to P4 calibration report."""
        p4_report = {
            "calibration_id": "p4_001",
            "timestamp": "2025-12-11T00:00:00Z",
        }
        
        envelope = CurriculumStabilityEnvelope(
            mean_hss=0.70,
            hss_variance=0.10,
            low_hss_fraction=0.3,
            slices_flagged=["slice_b", "slice_c"],
            suitability_scores={"slice_a": 0.80, "slice_b": 0.55, "slice_c": 0.50},
            status_light="YELLOW",
            stable_slices=["slice_a"],
            unstable_slices=["slice_b", "slice_c"],
            hss_variance_spikes=["slice_c"],
        )
        
        updated = add_stability_to_p4_calibration(p4_report, envelope)
        
        assert "curriculum_stability" in updated
        stability = updated["curriculum_stability"]
        
        assert "stable_slices" in stability
        assert "unstable_slices" in stability
        assert "HSS_variance_spikes" in stability
        assert "stability_gate_decisions" in stability
    
    def test_p4_gate_decisions_shadow(self):
        """Test that P4 gate decisions are in shadow mode."""
        p4_report = {"calibration_id": "p4_002"}
        
        envelope = CurriculumStabilityEnvelope(
            mean_hss=0.65,
            hss_variance=0.12,
            low_hss_fraction=0.4,
            slices_flagged=["slice_bad"],
            suitability_scores={"slice_good": 0.85, "slice_bad": 0.40},
            status_light="RED",
        )
        
        updated = add_stability_to_p4_calibration(p4_report, envelope)
        
        decisions = updated["curriculum_stability"]["stability_gate_decisions"]
        
        # slice_bad should be marked BLOCK (shadow mode)
        assert decisions["slice_bad"] == "BLOCK"
        assert decisions["slice_good"] == "ALLOW"
