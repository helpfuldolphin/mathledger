"""
Tests for Evidence Quality Phase-Portrait Engine (D3 Phase VI).

This module tests the geometric phase-portrait analysis, envelope forecasting,
and enhanced director panel integration.
"""

import pytest
from typing import Dict, Any, List


class TestEvidencePhasePortrait:
    """Tests for build_evidence_phase_portrait (D3 Phase VI)."""
    
    @pytest.mark.unit
    def test_phase_portrait_improving_trajectory(self):
        """Test 182: Phase portrait detects IMPROVING trajectory."""
        from backend.metrics.u2_analysis import (
            build_evidence_quality_timeline,
            build_evidence_phase_portrait,
        )
        
        snapshots = [
            {"run_id": "run1", "quality_tier": "TIER_1", "analysis_count": 1},
            {"run_id": "run2", "quality_tier": "TIER_2", "analysis_count": 2},
            {"run_id": "run3", "quality_tier": "TIER_3", "analysis_count": 3},
        ]
        
        timeline = build_evidence_quality_timeline(snapshots)
        portrait = build_evidence_phase_portrait(timeline)
        
        assert portrait["trajectory_class"] == "IMPROVING"
        assert len(portrait["phase_points"]) == 3
        assert portrait["phase_points"][0] == [0, 1]  # run_index=0, TIER_1=1
        assert portrait["phase_points"][1] == [1, 2]  # run_index=1, TIER_2=2
        assert portrait["phase_points"][2] == [2, 3]  # run_index=2, TIER_3=3
    
    @pytest.mark.unit
    def test_phase_portrait_degrading_trajectory(self):
        """Test 183: Phase portrait detects DEGRADING trajectory."""
        from backend.metrics.u2_analysis import (
            build_evidence_quality_timeline,
            build_evidence_phase_portrait,
        )
        
        snapshots = [
            {"run_id": "run1", "quality_tier": "TIER_3", "analysis_count": 3},
            {"run_id": "run2", "quality_tier": "TIER_2", "analysis_count": 2},
            {"run_id": "run3", "quality_tier": "TIER_1", "analysis_count": 1},
        ]
        
        timeline = build_evidence_quality_timeline(snapshots)
        portrait = build_evidence_phase_portrait(timeline)
        
        assert portrait["trajectory_class"] == "DEGRADING"
        assert portrait["phase_points"][0] == [0, 3]
        assert portrait["phase_points"][1] == [1, 2]
        assert portrait["phase_points"][2] == [2, 1]
    
    @pytest.mark.unit
    def test_phase_portrait_oscillating_trajectory(self):
        """Test 184: Phase portrait detects OSCILLATING trajectory."""
        from backend.metrics.u2_analysis import (
            build_evidence_quality_timeline,
            build_evidence_phase_portrait,
        )
        
        snapshots = [
            {"run_id": "run1", "quality_tier": "TIER_3", "analysis_count": 3},
            {"run_id": "run2", "quality_tier": "TIER_2", "analysis_count": 2},
            {"run_id": "run3", "quality_tier": "TIER_3", "analysis_count": 3},
            {"run_id": "run4", "quality_tier": "TIER_2", "analysis_count": 2},
        ]
        
        timeline = build_evidence_quality_timeline(snapshots)
        portrait = build_evidence_phase_portrait(timeline)
        
        assert portrait["trajectory_class"] == "OSCILLATING"
        assert "alternating improve/degrade cycles" in " ".join(portrait["neutral_notes"])
    
    @pytest.mark.unit
    def test_phase_portrait_stable_trajectory(self):
        """Test 185: Phase portrait detects STABLE trajectory."""
        from backend.metrics.u2_analysis import (
            build_evidence_quality_timeline,
            build_evidence_phase_portrait,
        )
        
        snapshots = [
            {"run_id": "run1", "quality_tier": "TIER_2", "analysis_count": 2},
            {"run_id": "run2", "quality_tier": "TIER_2", "analysis_count": 2},
            {"run_id": "run3", "quality_tier": "TIER_2", "analysis_count": 2},
        ]
        
        timeline = build_evidence_quality_timeline(snapshots)
        portrait = build_evidence_phase_portrait(timeline)
        
        assert portrait["trajectory_class"] == "STABLE"
        assert all(point[1] == 2 for point in portrait["phase_points"])
    
    @pytest.mark.unit
    def test_phase_portrait_empty_timeline(self):
        """Test 186: Phase portrait handles empty timeline."""
        from backend.metrics.u2_analysis import (
            build_evidence_quality_timeline,
            build_evidence_phase_portrait,
        )
        
        timeline = build_evidence_quality_timeline([])
        portrait = build_evidence_phase_portrait(timeline)
        
        assert portrait["trajectory_class"] == "STABLE"
        assert portrait["phase_points"] == []
        assert "no timeline data available" in portrait["neutral_notes"][0]
    
    @pytest.mark.unit
    def test_phase_portrait_neutral_notes_sorted(self):
        """Test 187: Phase portrait neutral_notes are sorted for determinism."""
        from backend.metrics.u2_analysis import (
            build_evidence_quality_timeline,
            build_evidence_phase_portrait,
        )
        
        snapshots = [
            {"run_id": "run1", "quality_tier": "TIER_1", "analysis_count": 1},
            {"run_id": "run2", "quality_tier": "TIER_2", "analysis_count": 2},
        ]
        
        timeline = build_evidence_quality_timeline(snapshots)
        portrait = build_evidence_phase_portrait(timeline)
        
        assert portrait["neutral_notes"] == sorted(portrait["neutral_notes"])
    
    @pytest.mark.unit
    def test_phase_portrait_no_uplift_semantics(self):
        """Test 188: Phase portrait has no uplift semantics."""
        from backend.metrics.u2_analysis import (
            build_evidence_quality_timeline,
            build_evidence_phase_portrait,
            GOVERNANCE_SUMMARY_FORBIDDEN_WORDS,
        )
        import json
        
        snapshots = [
            {"run_id": "run1", "quality_tier": "TIER_3", "analysis_count": 3},
            {"run_id": "run2", "quality_tier": "TIER_1", "analysis_count": 1},
        ]
        
        timeline = build_evidence_quality_timeline(snapshots)
        portrait = build_evidence_phase_portrait(timeline)
        
        portrait_str = json.dumps(portrait).lower()
        for word in GOVERNANCE_SUMMARY_FORBIDDEN_WORDS:
            assert word.lower() not in portrait_str, \
                f"Forbidden word '{word}' found in phase portrait"


class TestEvidenceEnvelopeForecast:
    """Tests for forecast_evidence_envelope (D3 Phase VI)."""
    
    @pytest.mark.unit
    def test_forecast_high_band_improving(self):
        """Test 189: Forecast predicts HIGH band for improving trajectory."""
        from backend.metrics.u2_analysis import (
            build_evidence_quality_timeline,
            build_evidence_phase_portrait,
            evaluate_evidence_quality_regression,
            forecast_evidence_envelope,
        )
        
        snapshots = [
            {"run_id": "run1", "quality_tier": "TIER_2", "analysis_count": 2},
            {"run_id": "run2", "quality_tier": "TIER_3", "analysis_count": 3},
            {"run_id": "run3", "quality_tier": "TIER_3", "analysis_count": 3},
        ]
        
        timeline = build_evidence_quality_timeline(snapshots)
        portrait = build_evidence_phase_portrait(timeline)
        regression = evaluate_evidence_quality_regression(timeline)
        forecast = forecast_evidence_envelope(portrait, regression)
        
        assert forecast["predicted_band"] == "HIGH"
        assert forecast["confidence"] > 0.0
        assert forecast["cycles_until_risk"] > 0
    
    @pytest.mark.unit
    def test_forecast_low_band_blocking(self):
        """Test 190: Forecast predicts LOW band for blocking regression."""
        from backend.metrics.u2_analysis import (
            build_evidence_quality_timeline,
            build_evidence_phase_portrait,
            evaluate_evidence_quality_regression,
            forecast_evidence_envelope,
        )
        
        snapshots = [
            {"run_id": "run1", "quality_tier": "TIER_3", "analysis_count": 3},
            {"run_id": "run2", "quality_tier": "TIER_1", "analysis_count": 1},
            {"run_id": "run3", "quality_tier": "TIER_3", "analysis_count": 3},
            {"run_id": "run4", "quality_tier": "TIER_2", "analysis_count": 2},
        ]
        
        timeline = build_evidence_quality_timeline(snapshots)
        portrait = build_evidence_phase_portrait(timeline)
        regression = evaluate_evidence_quality_regression(timeline)
        forecast = forecast_evidence_envelope(portrait, regression)
        
        assert forecast["predicted_band"] == "LOW"
        assert forecast["cycles_until_risk"] == 0
        assert "regression watchdog indicates blocking" in " ".join(forecast["neutral_explanation"])
    
    @pytest.mark.unit
    def test_forecast_medium_band_oscillating(self):
        """Test 191: Forecast predicts MEDIUM band for oscillating pattern."""
        from backend.metrics.u2_analysis import (
            build_evidence_quality_timeline,
            build_evidence_phase_portrait,
            evaluate_evidence_quality_regression,
            forecast_evidence_envelope,
        )
        
        snapshots = [
            {"run_id": "run1", "quality_tier": "TIER_3", "analysis_count": 3},
            {"run_id": "run2", "quality_tier": "TIER_2", "analysis_count": 2},
            {"run_id": "run3", "quality_tier": "TIER_3", "analysis_count": 3},
            {"run_id": "run4", "quality_tier": "TIER_2", "analysis_count": 2},
        ]
        
        timeline = build_evidence_quality_timeline(snapshots)
        portrait = build_evidence_phase_portrait(timeline)
        regression = evaluate_evidence_quality_regression(timeline)
        forecast = forecast_evidence_envelope(portrait, regression)
        
        assert forecast["predicted_band"] == "MEDIUM"
        assert "oscillating pattern" in " ".join(forecast["neutral_explanation"])
    
    @pytest.mark.unit
    def test_forecast_confidence_range(self):
        """Test 192: Forecast confidence is in valid range [0.0, 1.0]."""
        from backend.metrics.u2_analysis import (
            build_evidence_quality_timeline,
            build_evidence_phase_portrait,
            evaluate_evidence_quality_regression,
            forecast_evidence_envelope,
        )
        
        snapshots = [
            {"run_id": "run1", "quality_tier": "TIER_2", "analysis_count": 2},
        ]
        
        timeline = build_evidence_quality_timeline(snapshots)
        portrait = build_evidence_phase_portrait(timeline)
        regression = evaluate_evidence_quality_regression(timeline)
        forecast = forecast_evidence_envelope(portrait, regression)
        
        assert 0.0 <= forecast["confidence"] <= 1.0
    
    @pytest.mark.unit
    def test_forecast_empty_portrait(self):
        """Test 193: Forecast handles empty phase portrait."""
        from backend.metrics.u2_analysis import (
            build_evidence_quality_timeline,
            build_evidence_phase_portrait,
            evaluate_evidence_quality_regression,
            forecast_evidence_envelope,
        )
        
        timeline = build_evidence_quality_timeline([])
        portrait = build_evidence_phase_portrait(timeline)
        regression = evaluate_evidence_quality_regression(timeline)
        forecast = forecast_evidence_envelope(portrait, regression)
        
        assert forecast["predicted_band"] == "LOW"
        assert forecast["confidence"] == 0.0
        assert "insufficient data" in forecast["neutral_explanation"][0]
    
    @pytest.mark.unit
    def test_forecast_neutral_explanation_sorted(self):
        """Test 194: Forecast neutral_explanation is sorted for determinism."""
        from backend.metrics.u2_analysis import (
            build_evidence_quality_timeline,
            build_evidence_phase_portrait,
            evaluate_evidence_quality_regression,
            forecast_evidence_envelope,
        )
        
        snapshots = [
            {"run_id": "run1", "quality_tier": "TIER_2", "analysis_count": 2},
            {"run_id": "run2", "quality_tier": "TIER_3", "analysis_count": 3},
        ]
        
        timeline = build_evidence_quality_timeline(snapshots)
        portrait = build_evidence_phase_portrait(timeline)
        regression = evaluate_evidence_quality_regression(timeline)
        forecast = forecast_evidence_envelope(portrait, regression)
        
        assert forecast["neutral_explanation"] == sorted(forecast["neutral_explanation"])
    
    @pytest.mark.unit
    def test_forecast_cycles_until_risk_non_negative(self):
        """Test 195: Forecast cycles_until_risk is non-negative."""
        from backend.metrics.u2_analysis import (
            build_evidence_quality_timeline,
            build_evidence_phase_portrait,
            evaluate_evidence_quality_regression,
            forecast_evidence_envelope,
        )
        
        snapshots = [
            {"run_id": "run1", "quality_tier": "TIER_1", "analysis_count": 1},
        ]
        
        timeline = build_evidence_quality_timeline(snapshots)
        portrait = build_evidence_phase_portrait(timeline)
        regression = evaluate_evidence_quality_regression(timeline)
        forecast = forecast_evidence_envelope(portrait, regression)
        
        assert forecast["cycles_until_risk"] >= 0


class TestEnhancedDirectorPanel:
    """Tests for build_evidence_director_panel_v2 (D3 Phase VI)."""
    
    @pytest.mark.unit
    def test_director_panel_v2_green_light(self):
        """Test 196: Enhanced director panel shows GREEN for OK status."""
        from backend.metrics.u2_analysis import (
            build_evidence_pack,
            build_evidence_quality_snapshot,
            evaluate_evidence_readiness,
            classify_evidence_quality_level,
            evaluate_evidence_for_promotion,
            build_evidence_quality_timeline,
            build_evidence_phase_portrait,
            evaluate_evidence_quality_regression,
            forecast_evidence_envelope,
            build_evidence_director_panel_v2,
            PairedDeltaResult,
        )
        
        result = PairedDeltaResult(
            delta=0.1, ci_lower=0.0, ci_upper=0.2, n_baseline=100, n_rfl=100,
            metric_path="accuracy", seed=42, n_bootstrap=10000, method="BCa",
            analysis_id="test_1",
        )
        
        pack = build_evidence_pack([result])
        snapshot = build_evidence_quality_snapshot(pack)
        readiness = evaluate_evidence_readiness(snapshot)
        tier_info = classify_evidence_quality_level(snapshot)
        adversarial = {"status": "OK"}
        promotion = evaluate_evidence_for_promotion(snapshot, readiness, adversarial)
        
        timeline_snapshots = [
            {"run_id": "run1", "quality_tier": tier_info["quality_tier"], "analysis_count": 1},
        ]
        timeline = build_evidence_quality_timeline(timeline_snapshots)
        portrait = build_evidence_phase_portrait(timeline)
        regression = evaluate_evidence_quality_regression(timeline)
        forecast = forecast_evidence_envelope(portrait, regression)
        
        panel = build_evidence_director_panel_v2(tier_info, promotion, portrait, forecast, regression)
        
        assert panel["status_light"] == "GREEN"
        assert panel["evidence_ok"] == True
        assert "trajectory_class" in panel
        assert "regression_status" in panel
    
    @pytest.mark.unit
    def test_director_panel_v2_red_light_regression(self):
        """Test 197: Enhanced director panel shows RED for regression."""
        from backend.metrics.u2_analysis import (
            build_evidence_quality_timeline,
            build_evidence_phase_portrait,
            evaluate_evidence_quality_regression,
            forecast_evidence_envelope,
            build_evidence_director_panel_v2,
        )
        
        snapshots = [
            {"run_id": "run1", "quality_tier": "TIER_3", "analysis_count": 3},
            {"run_id": "run2", "quality_tier": "TIER_1", "analysis_count": 1},
            {"run_id": "run3", "quality_tier": "TIER_3", "analysis_count": 3},
            {"run_id": "run4", "quality_tier": "TIER_2", "analysis_count": 2},
        ]
        
        timeline = build_evidence_quality_timeline(snapshots)
        portrait = build_evidence_phase_portrait(timeline)
        regression = evaluate_evidence_quality_regression(timeline)
        forecast = forecast_evidence_envelope(portrait, regression)
        
        tier_info = {"quality_tier": "TIER_2"}
        promotion = {"promotion_ok": False, "status": "BLOCK", "blocking_reasons": []}
        
        panel = build_evidence_director_panel_v2(tier_info, promotion, portrait, forecast, regression)
        
        assert panel["status_light"] == "RED"
        assert "regression_detected" in panel["flags"]
    
    @pytest.mark.unit
    def test_director_panel_v2_flags_oscillating(self):
        """Test 198: Enhanced director panel flags oscillating trajectory."""
        from backend.metrics.u2_analysis import (
            build_evidence_quality_timeline,
            build_evidence_phase_portrait,
            evaluate_evidence_quality_regression,
            forecast_evidence_envelope,
            build_evidence_director_panel_v2,
        )
        
        snapshots = [
            {"run_id": "run1", "quality_tier": "TIER_3", "analysis_count": 3},
            {"run_id": "run2", "quality_tier": "TIER_2", "analysis_count": 2},
            {"run_id": "run3", "quality_tier": "TIER_3", "analysis_count": 3},
            {"run_id": "run4", "quality_tier": "TIER_2", "analysis_count": 2},
        ]
        
        timeline = build_evidence_quality_timeline(snapshots)
        portrait = build_evidence_phase_portrait(timeline)
        regression = evaluate_evidence_quality_regression(timeline)
        forecast = forecast_evidence_envelope(portrait, regression)
        
        tier_info = {"quality_tier": "TIER_2"}
        promotion = {"promotion_ok": True, "status": "OK", "blocking_reasons": []}
        
        panel = build_evidence_director_panel_v2(tier_info, promotion, portrait, forecast, regression)
        
        assert "oscillating_trajectory" in panel["flags"]
        assert panel["trajectory_class"] == "OSCILLATING"
    
    @pytest.mark.unit
    def test_director_panel_v2_flags_imminent_risk(self):
        """Test 199: Enhanced director panel flags imminent risk."""
        from backend.metrics.u2_analysis import (
            build_evidence_quality_timeline,
            build_evidence_phase_portrait,
            evaluate_evidence_quality_regression,
            forecast_evidence_envelope,
            build_evidence_director_panel_v2,
        )
        
        snapshots = [
            {"run_id": "run1", "quality_tier": "TIER_3", "analysis_count": 3},
            {"run_id": "run2", "quality_tier": "TIER_1", "analysis_count": 1},
        ]
        
        timeline = build_evidence_quality_timeline(snapshots)
        portrait = build_evidence_phase_portrait(timeline)
        regression = evaluate_evidence_quality_regression(timeline)
        forecast = forecast_evidence_envelope(portrait, regression)
        
        tier_info = {"quality_tier": "TIER_1"}
        promotion = {"promotion_ok": False, "status": "BLOCK", "blocking_reasons": []}
        
        panel = build_evidence_director_panel_v2(tier_info, promotion, portrait, forecast, regression)
        
        # If cycles_until_risk <= 1, should have imminent_risk flag
        if forecast["cycles_until_risk"] <= 1:
            assert "imminent_risk" in panel["flags"]
    
    @pytest.mark.unit
    def test_director_panel_v2_all_fields_present(self):
        """Test 200: Enhanced director panel includes all required fields."""
        from backend.metrics.u2_analysis import (
            build_evidence_quality_timeline,
            build_evidence_phase_portrait,
            evaluate_evidence_quality_regression,
            forecast_evidence_envelope,
            build_evidence_director_panel_v2,
        )
        
        snapshots = [
            {"run_id": "run1", "quality_tier": "TIER_2", "analysis_count": 2},
        ]
        
        timeline = build_evidence_quality_timeline(snapshots)
        portrait = build_evidence_phase_portrait(timeline)
        regression = evaluate_evidence_quality_regression(timeline)
        forecast = forecast_evidence_envelope(portrait, regression)
        
        tier_info = {"quality_tier": "TIER_2"}
        promotion = {"promotion_ok": True, "status": "OK", "blocking_reasons": [], "analysis_count": 1}
        
        panel = build_evidence_director_panel_v2(tier_info, promotion, portrait, forecast, regression)
        
        required_fields = [
            "status_light", "quality_tier", "trajectory_class", "regression_status",
            "analysis_count", "evidence_ok", "headline", "flags"
        ]
        for field in required_fields:
            assert field in panel, f"Missing required field: {field}"
    
    @pytest.mark.unit
    def test_director_panel_v2_flags_sorted(self):
        """Test 201: Enhanced director panel flags are sorted for determinism."""
        from backend.metrics.u2_analysis import (
            build_evidence_quality_timeline,
            build_evidence_phase_portrait,
            evaluate_evidence_quality_regression,
            forecast_evidence_envelope,
            build_evidence_director_panel_v2,
        )
        
        snapshots = [
            {"run_id": "run1", "quality_tier": "TIER_3", "analysis_count": 3},
            {"run_id": "run2", "quality_tier": "TIER_2", "analysis_count": 2},
            {"run_id": "run3", "quality_tier": "TIER_3", "analysis_count": 3},
        ]
        
        timeline = build_evidence_quality_timeline(snapshots)
        portrait = build_evidence_phase_portrait(timeline)
        regression = evaluate_evidence_quality_regression(timeline)
        forecast = forecast_evidence_envelope(portrait, regression)
        
        tier_info = {"quality_tier": "TIER_2"}
        promotion = {"promotion_ok": True, "status": "OK", "blocking_reasons": []}
        
        panel = build_evidence_director_panel_v2(tier_info, promotion, portrait, forecast, regression)
        
        assert panel["flags"] == sorted(panel["flags"])
    
    @pytest.mark.unit
    def test_director_panel_v2_no_uplift_semantics(self):
        """Test 202: Enhanced director panel has no uplift semantics."""
        from backend.metrics.u2_analysis import (
            build_evidence_quality_timeline,
            build_evidence_phase_portrait,
            evaluate_evidence_quality_regression,
            forecast_evidence_envelope,
            build_evidence_director_panel_v2,
            GOVERNANCE_SUMMARY_FORBIDDEN_WORDS,
        )
        import json
        
        snapshots = [
            {"run_id": "run1", "quality_tier": "TIER_3", "analysis_count": 3},
            {"run_id": "run2", "quality_tier": "TIER_1", "analysis_count": 1},
        ]
        
        timeline = build_evidence_quality_timeline(snapshots)
        portrait = build_evidence_phase_portrait(timeline)
        regression = evaluate_evidence_quality_regression(timeline)
        forecast = forecast_evidence_envelope(portrait, regression)
        
        tier_info = {"quality_tier": "TIER_2"}
        promotion = {"promotion_ok": False, "status": "BLOCK", "blocking_reasons": []}
        
        panel = build_evidence_director_panel_v2(tier_info, promotion, portrait, forecast, regression)
        
        panel_str = json.dumps(panel).lower()
        for word in GOVERNANCE_SUMMARY_FORBIDDEN_WORDS:
            assert word.lower() not in panel_str, \
                f"Forbidden word '{word}' found in enhanced director panel"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

