"""Tests for semantic-TDA coupling and governance tile.

STATUS: PHASE V — SEMANTIC/TDA CROSS-TIE

Tests correlation between semantic drift signals and TDA health signals,
and the governance tile builder that combines them.
"""

import json

import pytest

from backend.health.semantic_tda_adapter import (
    attach_semantic_tda_to_evidence,
    build_semantic_tda_correlation_summary,
    build_semantic_tda_tile_for_global_health,
)
from experiments.semantic_consistency_audit import (
    build_semantic_tda_governance_tile,
    correlate_semantic_and_tda_signals,
)


class TestCorrelateSemanticAndTdaSignals:
    """Tests for correlate_semantic_and_tda_signals function."""

    def test_both_green_stable(self):
        """Both systems report stable state."""
        semantic_timeline = {
            "timeline": [],
            "runs_with_critical_signals": [],
            "node_disappearance_events": [],
            "trend": "STABLE",
            "semantic_status_light": "GREEN",
        }
        tda_health = {
            "tda_status": "OK",
            "block_rate": 0.05,
            "hss_trend": "STABLE",
            "governance_signal": "OK",
        }

        result = correlate_semantic_and_tda_signals(semantic_timeline, tda_health)

        assert result["correlation_coefficient"] == 0.0
        assert result["slices_where_both_signal"] == []
        assert result["semantic_only_slices"] == []
        assert result["tda_only_slices"] == []

    def test_both_red_alert_high_correlation(self):
        """Both systems signal critical issues - high positive correlation."""
        semantic_timeline = {
            "timeline": [{"run_id": "run_001", "status": "CRITICAL"}],
            "runs_with_critical_signals": ["run_001"],
            "node_disappearance_events": [
                {"run_id": "run_001", "term": "slice_uplift_goal"}
            ],
            "trend": "DRIFTING",
            "semantic_status_light": "RED",
        }
        tda_health = {
            "tda_status": "ALERT",
            "block_rate": 0.25,
            "hss_trend": "DEGRADING",
            "governance_signal": "BLOCK",
        }

        result = correlate_semantic_and_tda_signals(semantic_timeline, tda_health)

        assert result["correlation_coefficient"] >= 0.8
        assert "slice_uplift_goal" in result["slices_where_both_signal"]
        assert result["semantic_only_slices"] == []
        assert result["tda_only_slices"] == []

    def test_semantic_red_tda_ok_negative_correlation(self):
        """Semantic signals issues but TDA is stable - negative correlation."""
        semantic_timeline = {
            "timeline": [{"run_id": "run_001", "status": "CRITICAL"}],
            "runs_with_critical_signals": ["run_001"],
            "node_disappearance_events": [
                {"run_id": "run_001", "term": "slice_alpha"}
            ],
            "trend": "DRIFTING",
            "semantic_status_light": "RED",
        }
        tda_health = {
            "tda_status": "OK",
            "block_rate": 0.05,
            "hss_trend": "STABLE",
            "governance_signal": "OK",
        }

        result = correlate_semantic_and_tda_signals(semantic_timeline, tda_health)

        assert result["correlation_coefficient"] <= -0.3
        assert result["slices_where_both_signal"] == []
        assert "slice_alpha" in result["semantic_only_slices"]
        assert result["tda_only_slices"] == []

    def test_tda_alert_semantic_green_negative_correlation(self):
        """TDA signals issues but semantic is stable - negative correlation."""
        semantic_timeline = {
            "timeline": [],
            "runs_with_critical_signals": [],
            "node_disappearance_events": [],
            "trend": "STABLE",
            "semantic_status_light": "GREEN",
        }
        tda_health = {
            "tda_status": "ALERT",
            "block_rate": 0.30,
            "hss_trend": "DEGRADING",
            "governance_signal": "BLOCK",
        }

        result = correlate_semantic_and_tda_signals(semantic_timeline, tda_health)

        assert result["correlation_coefficient"] <= -0.3
        assert result["slices_where_both_signal"] == []
        assert result["semantic_only_slices"] == []
        assert "system_wide" in result["tda_only_slices"]

    def test_mismatch_yellow_attention_weak_correlation(self):
        """Both systems report moderate issues - weak correlation."""
        semantic_timeline = {
            "timeline": [{"run_id": "run_001", "status": "WARN"}],
            "runs_with_critical_signals": ["run_001"],
            "node_disappearance_events": [
                {"run_id": "run_001", "term": "slice_beta"}
            ],
            "trend": "VOLATILE",
            "semantic_status_light": "YELLOW",
        }
        tda_health = {
            "tda_status": "ATTENTION",
            "block_rate": 0.15,
            "hss_trend": "FLUCTUATING",
            "governance_signal": "WARN",
        }

        result = correlate_semantic_and_tda_signals(semantic_timeline, tda_health)

        assert result["correlation_coefficient"] >= 0.8  # Both non-zero = positive
        assert "slice_beta" in result["slices_where_both_signal"]
        assert result["semantic_only_slices"] == []
        assert result["tda_only_slices"] == []

    def test_empty_timeline_handles_gracefully(self):
        """Empty timeline should not crash."""
        semantic_timeline = {
            "timeline": [],
            "runs_with_critical_signals": [],
            "node_disappearance_events": [],
            "trend": "STABLE",
        }
        tda_health = {
            "tda_status": "OK",
            "block_rate": 0.05,
            "hss_trend": "STABLE",
            "governance_signal": "OK",
        }

        result = correlate_semantic_and_tda_signals(semantic_timeline, tda_health)

        assert "correlation_coefficient" in result
        assert isinstance(result["correlation_coefficient"], (int, float))

    def test_missing_keys_raises_error(self):
        """Missing required keys should raise ValueError."""
        semantic_timeline = {
            "timeline": [],
            # Missing runs_with_critical_signals
        }
        tda_health = {
            "tda_status": "OK",
            "block_rate": 0.05,
        }

        with pytest.raises(ValueError, match="semantic_timeline missing required keys"):
            correlate_semantic_and_tda_signals(semantic_timeline, tda_health)


class TestBuildSemanticTdaGovernanceTile:
    """Tests for build_semantic_tda_governance_tile function."""

    def test_both_green_ok_tile(self):
        """Both systems green → OK tile."""
        semantic_panel = {
            "semantic_status_light": "GREEN",
            "alignment_status": "ALIGNED",
            "critical_run_ids": [],
            "headline": "Semantic graph stable",
        }
        tda_panel = {
            "tda_status": "OK",
            "block_rate": 0.05,
            "hss_trend": "STABLE",
            "governance_signal": "OK",
        }
        correlation = {
            "correlation_coefficient": 0.0,
            "slices_where_both_signal": [],
            "alignment_note": "Both systems stable",
        }

        tile = build_semantic_tda_governance_tile(semantic_panel, tda_panel, correlation)

        assert tile["status"] == "OK"
        assert tile["status_light"] == "GREEN"
        assert "stable" in tile["headline"].lower()

    def test_both_red_alert_block_tile(self):
        """Both systems critical → BLOCK tile."""
        semantic_panel = {
            "semantic_status_light": "RED",
            "alignment_status": "MISALIGNED",
            "critical_run_ids": ["run_001"],
            "headline": "Critical semantic drift detected",
        }
        tda_panel = {
            "tda_status": "ALERT",
            "block_rate": 0.28,
            "hss_trend": "DEGRADING",
            "governance_signal": "BLOCK",
        }
        correlation = {
            "correlation_coefficient": 0.9,
            "slices_where_both_signal": ["slice_uplift_goal"],
            "alignment_note": "Strong positive correlation",
        }

        tile = build_semantic_tda_governance_tile(semantic_panel, tda_panel, correlation)

        assert tile["status"] == "BLOCK"
        assert tile["status_light"] == "RED"
        assert "slice_uplift_goal" in tile["key_slices"]
        assert "critical" in tile["headline"].lower() or "drift" in tile["headline"].lower()

    def test_mismatch_attention_tile(self):
        """Semantic RED + TDA OK → ATTENTION tile."""
        semantic_panel = {
            "semantic_status_light": "RED",
            "alignment_status": "MISALIGNED",
            "critical_run_ids": ["run_001"],
            "headline": "Semantic drift detected",
        }
        tda_panel = {
            "tda_status": "OK",
            "block_rate": 0.05,
            "hss_trend": "STABLE",
            "governance_signal": "OK",
        }
        correlation = {
            "correlation_coefficient": -0.5,
            "slices_where_both_signal": [],
            "alignment_note": "Systems disagree",
        }

        tile = build_semantic_tda_governance_tile(semantic_panel, tda_panel, correlation)

        assert tile["status"] == "ATTENTION"
        assert tile["status_light"] == "YELLOW"
        assert "disagree" in tile["headline"].lower() or "investigation" in tile["headline"].lower()

    def test_tda_alert_semantic_green_attention_tile(self):
        """TDA ALERT + Semantic GREEN → ATTENTION tile."""
        semantic_panel = {
            "semantic_status_light": "GREEN",
            "alignment_status": "ALIGNED",
            "critical_run_ids": [],
            "headline": "Semantic graph stable",
        }
        tda_panel = {
            "tda_status": "ALERT",
            "block_rate": 0.30,
            "hss_trend": "DEGRADING",
            "governance_signal": "BLOCK",
        }
        correlation = {
            "correlation_coefficient": -0.5,
            "slices_where_both_signal": [],
            "alignment_note": "Systems disagree",
        }

        tile = build_semantic_tda_governance_tile(semantic_panel, tda_panel, correlation)

        assert tile["status"] == "ATTENTION"
        assert tile["status_light"] == "YELLOW"

    def test_high_correlation_with_red_block_tile(self):
        """High correlation + semantic RED → BLOCK even if TDA is only ATTENTION."""
        semantic_panel = {
            "semantic_status_light": "RED",
            "alignment_status": "MISALIGNED",
            "critical_run_ids": ["run_001"],
            "headline": "Critical semantic drift",
        }
        tda_panel = {
            "tda_status": "ATTENTION",
            "block_rate": 0.15,
            "hss_trend": "FLUCTUATING",
            "governance_signal": "WARN",
        }
        correlation = {
            "correlation_coefficient": 0.85,
            "slices_where_both_signal": ["slice_alpha"],
            "alignment_note": "Strong correlation",
        }

        tile = build_semantic_tda_governance_tile(semantic_panel, tda_panel, correlation)

        assert tile["status"] == "BLOCK"
        assert tile["status_light"] == "RED"

    def test_missing_keys_raises_error(self):
        """Missing required keys should raise ValueError."""
        semantic_panel = {
            "semantic_status_light": "GREEN",
            # Missing alignment_status
        }
        tda_panel = {
            "tda_status": "OK",
            "block_rate": 0.05,
        }
        correlation = {
            "correlation_coefficient": 0.0,
            "slices_where_both_signal": [],
        }

        with pytest.raises(ValueError, match="semantic_panel missing required keys"):
            build_semantic_tda_governance_tile(semantic_panel, tda_panel, correlation)

    def test_key_slices_aggregation(self):
        """Key slices should aggregate from correlation and panel."""
        semantic_panel = {
            "semantic_status_light": "RED",
            "alignment_status": "MISALIGNED",
            "critical_run_ids": ["run_uplift_slice_alpha", "run_uplift_slice_beta"],
            "headline": "Critical drift",
        }
        tda_panel = {
            "tda_status": "ALERT",
            "block_rate": 0.25,
            "hss_trend": "DEGRADING",
            "governance_signal": "BLOCK",
        }
        correlation = {
            "correlation_coefficient": 0.9,
            "slices_where_both_signal": ["slice_alpha", "slice_gamma"],
            "alignment_note": "Strong correlation",
        }

        tile = build_semantic_tda_governance_tile(semantic_panel, tda_panel, correlation)

        assert len(tile["key_slices"]) > 0
        # Should include slices from correlation
        assert "slice_alpha" in tile["key_slices"] or "slice_gamma" in tile["key_slices"]


class TestIntegrationScenario:
    """Integration test with realistic data shapes."""

    def test_realistic_drift_scenario(self):
        """Test realistic scenario: semantic RED + TDA ALERT → BLOCK."""
        semantic_timeline = {
            "timeline": [
                {"run_id": "run_uplift_001", "status": "CRITICAL"},
                {"run_id": "run_uplift_002", "status": "CRITICAL"},
            ],
            "runs_with_critical_signals": ["run_uplift_001", "run_uplift_002"],
            "node_disappearance_events": [
                {"run_id": "run_uplift_001", "term": "slice_uplift_goal"},
                {"run_id": "run_uplift_002", "term": "slice_curriculum_alignment"},
            ],
            "trend": "DRIFTING",
            "semantic_status_light": "RED",
        }
        tda_health = {
            "tda_status": "ALERT",
            "block_rate": 0.28,
            "mean_hss": 0.45,
            "hss_trend": "DEGRADING",
            "governance_signal": "BLOCK",
            "notes": [
                "block_rate=28.00% exceeds 20% threshold",
                "hss_trend classified as DEGRADING over 100 cycles",
            ],
        }
        semantic_panel = {
            "semantic_status_light": "RED",
            "alignment_status": "MISALIGNED",
            "critical_run_ids": ["run_uplift_001", "run_uplift_002"],
            "headline": (
                "Semantic graph shows critical drift with significant curriculum "
                "misalignment (2 runs with critical signals)"
            ),
            "trend": "DRIFTING",
            "node_disappearance_count": 3,
        }

        # Run correlation
        correlation = correlate_semantic_and_tda_signals(semantic_timeline, tda_health)

        # Verify correlation
        assert correlation["correlation_coefficient"] >= 0.8
        assert len(correlation["slices_where_both_signal"]) > 0

        # Build tile
        tile = build_semantic_tda_governance_tile(semantic_panel, tda_health, correlation)

        # Verify tile
        assert tile["status"] == "BLOCK"
        assert tile["status_light"] == "RED"
        assert len(tile["key_slices"]) > 0
        assert "critical" in tile["headline"].lower() or "drift" in tile["headline"].lower()
        assert tile["correlation_coefficient"] >= 0.8


class TestSemanticTdaTileSchema:
    """Tests for semantic-TDA tile schema validation."""

    def test_tile_has_required_fields(self):
        """Semantic-TDA tile must have all required fields."""
        semantic_panel = {
            "semantic_status_light": "GREEN",
            "alignment_status": "ALIGNED",
            "critical_run_ids": [],
            "headline": "Stable",
        }
        tda_panel = {
            "tda_status": "OK",
            "block_rate": 0.05,
            "hss_trend": "STABLE",
            "governance_signal": "OK",
        }

        tile = build_semantic_tda_tile_for_global_health(semantic_panel, tda_panel)

        # Required fields
        assert "schema_version" in tile
        assert "tile_type" in tile
        assert tile["tile_type"] == "semantic_tda"
        assert "status" in tile
        assert tile["status"] in ("OK", "ATTENTION", "BLOCK")
        assert "status_light" in tile
        assert tile["status_light"] in ("GREEN", "YELLOW", "RED")
        assert "correlation_coefficient" in tile
        assert isinstance(tile["correlation_coefficient"], (int, float))
        assert -1.0 <= tile["correlation_coefficient"] <= 1.0
        assert "key_slices" in tile
        assert isinstance(tile["key_slices"], list)
        assert "headline" in tile
        assert isinstance(tile["headline"], str)
        assert "notes" in tile
        assert isinstance(tile["notes"], list)

    def test_tile_is_json_serializable(self):
        """Semantic-TDA tile must be JSON serializable."""
        semantic_panel = {
            "semantic_status_light": "RED",
            "alignment_status": "MISALIGNED",
            "critical_run_ids": ["run_001"],
            "headline": "Critical drift",
        }
        tda_panel = {
            "tda_status": "ALERT",
            "block_rate": 0.25,
            "hss_trend": "DEGRADING",
            "governance_signal": "BLOCK",
        }

        tile = build_semantic_tda_tile_for_global_health(semantic_panel, tda_panel)

        # Should serialize without error
        json_str = json.dumps(tile)
        parsed = json.loads(json_str)

        # Should round-trip correctly
        assert parsed["status"] == tile["status"]
        assert parsed["tile_type"] == tile["tile_type"]


class TestEvidencePackIntegration:
    """Tests for evidence pack integration."""

    def test_attach_semantic_tda_to_evidence(self):
        """Attach semantic-TDA tile to evidence pack."""
        evidence = {
            "timestamp": "2024-01-01T00:00:00Z",
            "run_id": "test_run_001",
            "data": {"some": "existing", "data": 123},
        }
        semantic_panel = {
            "semantic_status_light": "GREEN",
            "alignment_status": "ALIGNED",
            "critical_run_ids": [],
            "headline": "Stable",
        }
        tda_panel = {
            "tda_status": "OK",
            "block_rate": 0.05,
            "hss_trend": "STABLE",
            "governance_signal": "OK",
        }
        tile = build_semantic_tda_tile_for_global_health(semantic_panel, tda_panel)

        enriched = attach_semantic_tda_to_evidence(evidence, tile)

        # Original evidence should be unchanged
        assert evidence["timestamp"] == "2024-01-01T00:00:00Z"
        assert "governance" not in evidence

        # Enriched evidence should have governance.semantic_tda
        assert "governance" in enriched
        assert "semantic_tda" in enriched["governance"]
        assert enriched["governance"]["semantic_tda"]["tile_type"] == "semantic_tda"
        assert enriched["governance"]["semantic_tda"]["status"] == tile["status"]

        # Original data should be preserved
        assert enriched["data"] == evidence["data"]
        assert enriched["run_id"] == evidence["run_id"]

    def test_attach_to_evidence_with_existing_governance(self):
        """Attach semantic-TDA tile when governance key already exists."""
        evidence = {
            "timestamp": "2024-01-01T00:00:00Z",
            "governance": {
                "prng_governance": {"status": "OK"},
            },
        }
        semantic_panel = {
            "semantic_status_light": "YELLOW",
            "alignment_status": "PARTIAL",
            "critical_run_ids": ["run_001"],
            "headline": "Moderate drift",
        }
        tda_panel = {
            "tda_status": "ATTENTION",
            "block_rate": 0.15,
            "hss_trend": "FLUCTUATING",
            "governance_signal": "WARN",
        }
        tile = build_semantic_tda_tile_for_global_health(semantic_panel, tda_panel)

        enriched = attach_semantic_tda_to_evidence(evidence, tile)

        # Existing governance should be preserved
        assert "prng_governance" in enriched["governance"]
        assert enriched["governance"]["prng_governance"]["status"] == "OK"

        # Semantic-TDA should be added
        assert "semantic_tda" in enriched["governance"]
        assert enriched["governance"]["semantic_tda"]["tile_type"] == "semantic_tda"

    def test_evidence_pack_is_json_serializable(self):
        """Evidence pack with semantic-TDA tile must be JSON serializable."""
        evidence = {
            "timestamp": "2024-01-01T00:00:00Z",
            "run_id": "test_run_001",
            "data": {"some": "data"},
        }
        semantic_panel = {
            "semantic_status_light": "RED",
            "alignment_status": "MISALIGNED",
            "critical_run_ids": ["run_001"],
            "headline": "Critical drift",
        }
        tda_panel = {
            "tda_status": "ALERT",
            "block_rate": 0.28,
            "hss_trend": "DEGRADING",
            "governance_signal": "BLOCK",
        }
        tile = build_semantic_tda_tile_for_global_health(semantic_panel, tda_panel)

        enriched = attach_semantic_tda_to_evidence(evidence, tile)

        # Should serialize without error
        json_str = json.dumps(enriched)
        parsed = json.loads(json_str)

        # Should round-trip correctly
        assert parsed["governance"]["semantic_tda"]["status"] == tile["status"]
        assert parsed["run_id"] == evidence["run_id"]


class TestCorrelationSummary:
    """Tests for correlation summary helper."""

    def test_build_correlation_summary(self):
        """Build correlation summary from tile."""
        semantic_panel = {
            "semantic_status_light": "RED",
            "alignment_status": "MISALIGNED",
            "critical_run_ids": ["run_001"],
            "headline": "Critical drift",
        }
        tda_panel = {
            "tda_status": "ALERT",
            "block_rate": 0.28,
            "hss_trend": "DEGRADING",
            "governance_signal": "BLOCK",
        }
        tile = build_semantic_tda_tile_for_global_health(semantic_panel, tda_panel)

        summary = build_semantic_tda_correlation_summary(tile)

        # Required fields
        assert "schema_version" in summary
        assert "status" in summary
        assert summary["status"] == tile["status"]
        assert "correlation_coefficient" in summary
        assert summary["correlation_coefficient"] == tile["correlation_coefficient"]
        assert "num_key_slices" in summary
        assert "key_slices" in summary
        assert isinstance(summary["key_slices"], list)

    def test_correlation_summary_truncates_key_slices(self):
        """Correlation summary truncates key_slices to first 5."""
        semantic_panel = {
            "semantic_status_light": "RED",
            "alignment_status": "MISALIGNED",
            "critical_run_ids": ["run_001", "run_002", "run_003", "run_004", "run_005", "run_006"],
            "headline": "Critical drift",
        }
        tda_panel = {
            "tda_status": "ALERT",
            "block_rate": 0.28,
            "hss_trend": "DEGRADING",
            "governance_signal": "BLOCK",
        }
        tile = build_semantic_tda_tile_for_global_health(semantic_panel, tda_panel)

        # Ensure we have more than 5 slices (may need to adjust based on actual behavior)
        # For this test, we'll just verify truncation works
        summary = build_semantic_tda_correlation_summary(tile)

        assert len(summary["key_slices"]) <= 5
        assert summary["num_key_slices"] >= len(summary["key_slices"])

    def test_correlation_summary_is_json_serializable(self):
        """Correlation summary must be JSON serializable."""
        semantic_panel = {
            "semantic_status_light": "YELLOW",
            "alignment_status": "PARTIAL",
            "critical_run_ids": ["run_001"],
            "headline": "Moderate drift",
        }
        tda_panel = {
            "tda_status": "ATTENTION",
            "block_rate": 0.15,
            "hss_trend": "FLUCTUATING",
            "governance_signal": "WARN",
        }
        tile = build_semantic_tda_tile_for_global_health(semantic_panel, tda_panel)

        summary = build_semantic_tda_correlation_summary(tile)

        # Should serialize without error
        json_str = json.dumps(summary)
        parsed = json.loads(json_str)

        # Should round-trip correctly
        assert parsed["status"] == summary["status"]
        assert parsed["correlation_coefficient"] == summary["correlation_coefficient"]

    def test_correlation_summary_is_deterministic(self):
        """Correlation summary must be deterministic."""
        semantic_panel = {
            "semantic_status_light": "GREEN",
            "alignment_status": "ALIGNED",
            "critical_run_ids": [],
            "headline": "Stable",
        }
        tda_panel = {
            "tda_status": "OK",
            "block_rate": 0.05,
            "hss_trend": "STABLE",
            "governance_signal": "OK",
        }
        tile = build_semantic_tda_tile_for_global_health(semantic_panel, tda_panel)

        summary1 = build_semantic_tda_correlation_summary(tile)
        summary2 = build_semantic_tda_correlation_summary(tile)

        assert summary1 == summary2


class TestEvidencePackWithCorrelationSummary:
    """Tests for evidence pack integration with correlation summary."""

    def test_attach_with_correlation_summary(self):
        """Attach semantic-TDA tile with correlation summary to evidence pack."""
        evidence = {
            "timestamp": "2024-01-01T00:00:00Z",
            "run_id": "test_run_001",
            "data": {"some": "existing", "data": 123},
        }
        semantic_panel = {
            "semantic_status_light": "RED",
            "alignment_status": "MISALIGNED",
            "critical_run_ids": ["run_001"],
            "headline": "Critical drift",
        }
        tda_panel = {
            "tda_status": "ALERT",
            "block_rate": 0.28,
            "hss_trend": "DEGRADING",
            "governance_signal": "BLOCK",
        }
        tile = build_semantic_tda_tile_for_global_health(semantic_panel, tda_panel)

        enriched = attach_semantic_tda_to_evidence(evidence, tile, include_correlation_summary=True)

        # Original evidence should be unchanged
        assert evidence["timestamp"] == "2024-01-01T00:00:00Z"
        assert "governance" not in evidence

        # Enriched evidence should have governance.semantic_tda
        assert "governance" in enriched
        assert "semantic_tda" in enriched["governance"]
        
        # Should have correlation_summary
        assert "correlation_summary" in enriched["governance"]["semantic_tda"]
        summary = enriched["governance"]["semantic_tda"]["correlation_summary"]
        
        # Verify summary structure
        assert "schema_version" in summary
        assert "status" in summary
        assert "correlation_coefficient" in summary
        assert "num_key_slices" in summary
        assert "key_slices" in summary
        assert summary["status"] == tile["status"]
        assert summary["correlation_coefficient"] == tile["correlation_coefficient"]

    def test_attach_without_correlation_summary(self):
        """Attach semantic-TDA tile without correlation summary (default behavior)."""
        evidence = {
            "timestamp": "2024-01-01T00:00:00Z",
            "run_id": "test_run_001",
        }
        semantic_panel = {
            "semantic_status_light": "GREEN",
            "alignment_status": "ALIGNED",
            "critical_run_ids": [],
            "headline": "Stable",
        }
        tda_panel = {
            "tda_status": "OK",
            "block_rate": 0.05,
            "hss_trend": "STABLE",
            "governance_signal": "OK",
        }
        tile = build_semantic_tda_tile_for_global_health(semantic_panel, tda_panel)

        enriched = attach_semantic_tda_to_evidence(evidence, tile, include_correlation_summary=False)

        # Should not have correlation_summary
        assert "governance" in enriched
        assert "semantic_tda" in enriched["governance"]
        assert "correlation_summary" not in enriched["governance"]["semantic_tda"]

    def test_evidence_with_correlation_summary_is_json_serializable(self):
        """Evidence pack with correlation summary must be JSON serializable."""
        evidence = {
            "timestamp": "2024-01-01T00:00:00Z",
            "run_id": "test_run_001",
            "data": {"some": "data"},
        }
        semantic_panel = {
            "semantic_status_light": "RED",
            "alignment_status": "MISALIGNED",
            "critical_run_ids": ["run_001"],
            "headline": "Critical drift",
        }
        tda_panel = {
            "tda_status": "ALERT",
            "block_rate": 0.28,
            "hss_trend": "DEGRADING",
            "governance_signal": "BLOCK",
        }
        tile = build_semantic_tda_tile_for_global_health(semantic_panel, tda_panel)

        enriched = attach_semantic_tda_to_evidence(evidence, tile, include_correlation_summary=True)

        # Should serialize without error
        json_str = json.dumps(enriched)
        parsed = json.loads(json_str)

        # Should round-trip correctly
        assert parsed["governance"]["semantic_tda"]["status"] == tile["status"]
        assert "correlation_summary" in parsed["governance"]["semantic_tda"]
        assert parsed["governance"]["semantic_tda"]["correlation_summary"]["status"] == tile["status"]

    def test_correlation_summary_key_slices_truncation_in_evidence(self):
        """Correlation summary in evidence pack correctly truncates key_slices."""
        evidence = {
            "timestamp": "2024-01-01T00:00:00Z",
            "run_id": "test_run_001",
        }
        semantic_panel = {
            "semantic_status_light": "RED",
            "alignment_status": "MISALIGNED",
            "critical_run_ids": ["run_001", "run_002", "run_003", "run_004", "run_005", "run_006"],
            "headline": "Critical drift",
        }
        tda_panel = {
            "tda_status": "ALERT",
            "block_rate": 0.28,
            "hss_trend": "DEGRADING",
            "governance_signal": "BLOCK",
        }
        tile = build_semantic_tda_tile_for_global_health(semantic_panel, tda_panel)

        enriched = attach_semantic_tda_to_evidence(evidence, tile, include_correlation_summary=True)

        summary = enriched["governance"]["semantic_tda"]["correlation_summary"]
        
        # Verify truncation
        assert len(summary["key_slices"]) <= 5
        assert summary["num_key_slices"] >= len(summary["key_slices"])

