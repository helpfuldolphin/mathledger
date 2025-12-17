# tests/backend/verification/test_lean_shadow_health_tile.py
"""
Test suite for Lean shadow health tile integration.

These tests verify that the Lean shadow tile:
1. Correctly maps shadow radar to health tile
2. Determines status correctly (OK/WARN/BLOCK)
3. Produces deterministic headlines
4. Is JSON-serializable
5. Remains purely observational (no control paths)

Markers:
    - unit: Fast, no external dependencies
"""

import json
from pathlib import Path

import pytest

from backend.health.lean_shadow_adapter import (
    build_lean_shadow_tile_for_global_health,
    build_first_light_lean_shadow_summary,
    build_cal_exp_structural_summary,
    compare_lean_vs_structural_signal,
    build_structural_calibration_panel,
    extract_structural_calibration_panel_signal,
    attach_cal_exp_structural_summary_to_report,
    attach_lean_shadow_to_evidence,
    LEAN_SHADOW_TILE_SCHEMA_VERSION,
    CAL_EXP_STRUCTURAL_SUMMARY_SCHEMA_VERSION,
    STRUCTURAL_CALIBRATION_PANEL_SCHEMA_VERSION,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def minimal_radar() -> dict:
    """Minimal valid shadow radar."""
    return {
        "structural_error_rate": 0.0,
        "shadow_resource_band": "LOW",
        "anomaly_signatures": [],
        "total_shadow_requests": 0,
    }


@pytest.fixture
def ok_radar() -> dict:
    """Shadow radar that should produce OK status."""
    return {
        "structural_error_rate": 0.1,  # < 0.2
        "shadow_resource_band": "LOW",
        "anomaly_signatures": ["abc123"],
        "total_shadow_requests": 100,
    }


@pytest.fixture
def warn_radar_medium_error() -> dict:
    """Shadow radar with medium error rate (should produce WARN)."""
    return {
        "structural_error_rate": 0.3,  # > 0.2, < 0.5
        "shadow_resource_band": "LOW",
        "anomaly_signatures": ["def456", "ghi789"],
        "total_shadow_requests": 50,
    }


@pytest.fixture
def warn_radar_medium_band() -> dict:
    """Shadow radar with medium resource band (should produce WARN)."""
    return {
        "structural_error_rate": 0.1,  # < 0.2
        "shadow_resource_band": "MEDIUM",
        "anomaly_signatures": ["jkl012"],
        "total_shadow_requests": 75,
    }


@pytest.fixture
def block_radar_high_error() -> dict:
    """Shadow radar with high error rate (should produce BLOCK)."""
    return {
        "structural_error_rate": 0.6,  # > 0.5
        "shadow_resource_band": "LOW",
        "anomaly_signatures": ["mno345", "pqr678", "stu901"],
        "total_shadow_requests": 25,
    }


@pytest.fixture
def block_radar_high_band() -> dict:
    """Shadow radar with high resource band (should produce BLOCK)."""
    return {
        "structural_error_rate": 0.1,  # < 0.5
        "shadow_resource_band": "HIGH",
        "anomaly_signatures": ["vwx234"],
        "total_shadow_requests": 30,
    }


# =============================================================================
# BASIC FUNCTIONALITY
# =============================================================================

@pytest.mark.unit
class TestLeanShadowTileBasic:
    """Test basic Lean shadow tile functionality."""

    def test_tile_schema_version(self, minimal_radar: dict) -> None:
        """Tile should have correct schema version."""
        tile = build_lean_shadow_tile_for_global_health(minimal_radar)
        assert tile["schema_version"] == LEAN_SHADOW_TILE_SCHEMA_VERSION

    def test_tile_required_fields(self, minimal_radar: dict) -> None:
        """Tile should have all required fields."""
        tile = build_lean_shadow_tile_for_global_health(minimal_radar)
        
        required_fields = [
            "schema_version",
            "status",
            "structural_error_rate",
            "shadow_resource_band",
            "dominant_anomalies",
            "headline",
        ]
        
        for field in required_fields:
            assert field in tile, f"Missing required field: {field}"

    def test_tile_json_serializable(self, minimal_radar: dict) -> None:
        """Tile should be JSON serializable."""
        tile = build_lean_shadow_tile_for_global_health(minimal_radar)
        
        # Should not raise
        json_str = json.dumps(tile)
        assert isinstance(json_str, str)
        
        # Should round-trip
        parsed = json.loads(json_str)
        assert parsed["schema_version"] == tile["schema_version"]

    def test_tile_deterministic(self, ok_radar: dict) -> None:
        """Tile should be deterministic for same inputs."""
        tile1 = build_lean_shadow_tile_for_global_health(ok_radar)
        tile2 = build_lean_shadow_tile_for_global_health(ok_radar)
        
        assert tile1 == tile2

    def test_validation_missing_required_key(self) -> None:
        """Should raise ValueError if required key is missing."""
        invalid_radar = {
            "structural_error_rate": 0.1,
            # Missing shadow_resource_band
            "anomaly_signatures": [],
        }
        
        with pytest.raises(ValueError, match="missing required keys"):
            build_lean_shadow_tile_for_global_health(invalid_radar)

    def test_validation_invalid_resource_band(self) -> None:
        """Should raise ValueError if resource band is invalid."""
        invalid_radar = {
            "structural_error_rate": 0.1,
            "shadow_resource_band": "INVALID",
            "anomaly_signatures": [],
        }
        
        with pytest.raises(ValueError, match="Invalid shadow_resource_band"):
            build_lean_shadow_tile_for_global_health(invalid_radar)


# =============================================================================
# STATUS DETERMINATION
# =============================================================================

@pytest.mark.unit
class TestLeanShadowTileStatus:
    """Test status determination logic."""

    def test_status_ok_low_error_low_band(self, ok_radar: dict) -> None:
        """LOW structural error + LOW resource band → status OK."""
        tile = build_lean_shadow_tile_for_global_health(ok_radar)
        assert tile["status"] == "OK"

    def test_status_warn_medium_error(self, warn_radar_medium_error: dict) -> None:
        """MEDIUM structural error → status WARN."""
        tile = build_lean_shadow_tile_for_global_health(warn_radar_medium_error)
        assert tile["status"] == "WARN"

    def test_status_warn_medium_band(self, warn_radar_medium_band: dict) -> None:
        """MEDIUM resource band → status WARN."""
        tile = build_lean_shadow_tile_for_global_health(warn_radar_medium_band)
        assert tile["status"] == "WARN"

    def test_status_block_high_error(self, block_radar_high_error: dict) -> None:
        """HIGH structural error → status BLOCK."""
        tile = build_lean_shadow_tile_for_global_health(block_radar_high_error)
        assert tile["status"] == "BLOCK"

    def test_status_block_high_band(self, block_radar_high_band: dict) -> None:
        """HIGH resource band → status BLOCK."""
        tile = build_lean_shadow_tile_for_global_health(block_radar_high_band)
        assert tile["status"] == "BLOCK"

    def test_status_block_high_error_and_high_band(self) -> None:
        """HIGH error AND HIGH band → status BLOCK."""
        radar = {
            "structural_error_rate": 0.6,
            "shadow_resource_band": "HIGH",
            "anomaly_signatures": ["critical"],
            "total_shadow_requests": 10,
        }
        tile = build_lean_shadow_tile_for_global_health(radar)
        assert tile["status"] == "BLOCK"

    def test_status_warn_medium_error_and_medium_band(self) -> None:
        """MEDIUM error AND MEDIUM band → status WARN."""
        radar = {
            "structural_error_rate": 0.3,
            "shadow_resource_band": "MEDIUM",
            "anomaly_signatures": ["warning"],
            "total_shadow_requests": 20,
        }
        tile = build_lean_shadow_tile_for_global_health(radar)
        assert tile["status"] == "WARN"

    def test_status_edge_case_0_2_error_rate(self) -> None:
        """Error rate exactly 0.2 should be WARN (not OK)."""
        radar = {
            "structural_error_rate": 0.2,
            "shadow_resource_band": "LOW",
            "anomaly_signatures": [],
            "total_shadow_requests": 100,
        }
        tile = build_lean_shadow_tile_for_global_health(radar)
        assert tile["status"] == "WARN"

    def test_status_edge_case_0_5_error_rate(self) -> None:
        """Error rate exactly 0.5 should be WARN (not BLOCK)."""
        radar = {
            "structural_error_rate": 0.5,
            "shadow_resource_band": "LOW",
            "anomaly_signatures": [],
            "total_shadow_requests": 100,
        }
        tile = build_lean_shadow_tile_for_global_health(radar)
        assert tile["status"] == "WARN"

    def test_status_edge_case_0_5001_error_rate(self) -> None:
        """Error rate > 0.5 should be BLOCK."""
        radar = {
            "structural_error_rate": 0.5001,
            "shadow_resource_band": "LOW",
            "anomaly_signatures": [],
            "total_shadow_requests": 100,
        }
        tile = build_lean_shadow_tile_for_global_health(radar)
        assert tile["status"] == "BLOCK"


# =============================================================================
# TILE CONTENT
# =============================================================================

@pytest.mark.unit
class TestLeanShadowTileContent:
    """Test tile content correctness."""

    def test_structural_error_rate_preserved(self, ok_radar: dict) -> None:
        """Structural error rate should be preserved and rounded."""
        tile = build_lean_shadow_tile_for_global_health(ok_radar)
        assert tile["structural_error_rate"] == 0.1

    def test_resource_band_preserved(self, ok_radar: dict) -> None:
        """Resource band should be preserved."""
        tile = build_lean_shadow_tile_for_global_health(ok_radar)
        assert tile["shadow_resource_band"] == "LOW"

    def test_dominant_anomalies_top_3(self) -> None:
        """Dominant anomalies should be top 3."""
        radar = {
            "structural_error_rate": 0.1,
            "shadow_resource_band": "LOW",
            "anomaly_signatures": ["a", "b", "c", "d", "e"],
            "total_shadow_requests": 100,
        }
        tile = build_lean_shadow_tile_for_global_health(radar)
        assert tile["dominant_anomalies"] == ["a", "b", "c"]

    def test_dominant_anomalies_empty(self, minimal_radar: dict) -> None:
        """Dominant anomalies should be empty list if no anomalies."""
        tile = build_lean_shadow_tile_for_global_health(minimal_radar)
        assert tile["dominant_anomalies"] == []

    def test_dominant_anomalies_less_than_3(self) -> None:
        """Dominant anomalies should handle < 3 anomalies."""
        radar = {
            "structural_error_rate": 0.1,
            "shadow_resource_band": "LOW",
            "anomaly_signatures": ["a", "b"],
            "total_shadow_requests": 100,
        }
        tile = build_lean_shadow_tile_for_global_health(radar)
        assert tile["dominant_anomalies"] == ["a", "b"]

    def test_headline_includes_status(self, ok_radar: dict) -> None:
        """Headline should include status."""
        tile = build_lean_shadow_tile_for_global_health(ok_radar)
        assert "Status: OK" in tile["headline"]

    def test_headline_includes_error_rate(self, ok_radar: dict) -> None:
        """Headline should include error rate."""
        tile = build_lean_shadow_tile_for_global_health(ok_radar)
        assert "10.0%" in tile["headline"]  # 0.1 * 100

    def test_headline_includes_resource_band(self, ok_radar: dict) -> None:
        """Headline should include resource band."""
        tile = build_lean_shadow_tile_for_global_health(ok_radar)
        assert "LOW resource band" in tile["headline"]

    def test_headline_includes_request_count(self, ok_radar: dict) -> None:
        """Headline should include request count."""
        tile = build_lean_shadow_tile_for_global_health(ok_radar)
        assert "100 requests" in tile["headline"]

    def test_headline_no_requests(self, minimal_radar: dict) -> None:
        """Headline should handle zero requests."""
        tile = build_lean_shadow_tile_for_global_health(minimal_radar)
        assert "No Lean shadow mode activity" in tile["headline"]

    def test_headline_deterministic(self, ok_radar: dict) -> None:
        """Headline should be deterministic."""
        tile1 = build_lean_shadow_tile_for_global_health(ok_radar)
        tile2 = build_lean_shadow_tile_for_global_health(ok_radar)
        assert tile1["headline"] == tile2["headline"]

    def test_headline_warn_status(self, warn_radar_medium_error: dict) -> None:
        """Headline should include WARN status."""
        tile = build_lean_shadow_tile_for_global_health(warn_radar_medium_error)
        assert "Status: WARN" in tile["headline"]

    def test_headline_block_status(self, block_radar_high_error: dict) -> None:
        """Headline should include BLOCK status."""
        tile = build_lean_shadow_tile_for_global_health(block_radar_high_error)
        assert "Status: BLOCK" in tile["headline"]


# =============================================================================
# P3/P4 COMPATIBILITY
# =============================================================================

@pytest.mark.unit
class TestP3P4Compatibility:
    """Test P3/P4 compatibility (observational only)."""

    def test_tile_pure_function(self, ok_radar: dict) -> None:
        """Tile builder should be a pure function (no side effects)."""
        # Call multiple times - should produce same result
        tile1 = build_lean_shadow_tile_for_global_health(ok_radar)
        tile2 = build_lean_shadow_tile_for_global_health(ok_radar)
        tile3 = build_lean_shadow_tile_for_global_health(ok_radar)
        
        assert tile1 == tile2 == tile3

    def test_tile_no_control_paths(self, ok_radar: dict) -> None:
        """Tile should not contain control flow indicators."""
        tile = build_lean_shadow_tile_for_global_health(ok_radar)
        
        # Should not contain control-related fields
        control_fields = ["abort", "block", "enable", "disable", "trigger"]
        tile_str = json.dumps(tile).lower()
        
        # These words might appear in headline, but not as control fields
        # We're checking that there are no explicit control fields
        assert "abort" not in tile or "abort" not in str(tile.keys())

    def test_tile_observational_only(self, block_radar_high_error: dict) -> None:
        """Even BLOCK status should be observational only."""
        tile = build_lean_shadow_tile_for_global_health(block_radar_high_error)
        
        # Status is informational, not a control signal
        assert tile["status"] == "BLOCK"
        # But the tile itself is just data - no control logic
        assert isinstance(tile, dict)
        assert "schema_version" in tile


# =============================================================================
# INTEGRATION WITH GLOBAL HEALTH
# =============================================================================

@pytest.mark.unit
class TestGlobalHealthIntegration:
    """Test integration with global health surface."""

    def test_tile_can_be_attached_to_payload(self, ok_radar: dict) -> None:
        """Tile should be attachable to global health payload."""
        from backend.health.global_surface import attach_lean_shadow_tile
        
        payload = {"schema_version": "test/1.0.0"}
        updated = attach_lean_shadow_tile(payload, ok_radar)
        
        assert "lean_shadow" in updated
        assert updated["lean_shadow"]["status"] == "OK"

    def test_tile_in_build_global_health_surface(self, ok_radar: dict) -> None:
        """Tile should be included in build_global_health_surface."""
        from backend.health.global_surface import build_global_health_surface
        
        payload = build_global_health_surface(
            base_payload={"test": "data"},
            shadow_radar=ok_radar,
        )
        
        assert "lean_shadow" in payload
        assert payload["lean_shadow"]["status"] == "OK"

    def test_tile_optional_in_build_global_health_surface(self) -> None:
        """Tile should be optional in build_global_health_surface."""
        from backend.health.global_surface import build_global_health_surface
        
        payload = build_global_health_surface(
            base_payload={"test": "data"},
            # No shadow_radar provided
        )
        
        # Should not have lean_shadow if not provided
        assert "lean_shadow" not in payload

    def test_tile_handles_exception_gracefully(self) -> None:
        """Tile attachment should handle exceptions gracefully."""
        from backend.health.global_surface import attach_lean_shadow_tile
        
        # Invalid radar (missing required fields)
        invalid_radar = {"invalid": "data"}
        
        # Should not raise - should silently continue
        payload = {"schema_version": "test/1.0.0"}
        updated = attach_lean_shadow_tile(payload, invalid_radar)
        
        # Should not have lean_shadow if exception occurred
        assert "lean_shadow" not in updated


# =============================================================================
# EVIDENCE PACK INTEGRATION
# =============================================================================

@pytest.mark.unit
class TestEvidencePackIntegration:
    """Test evidence pack integration."""

    def test_attach_lean_shadow_to_evidence(self, ok_radar: dict) -> None:
        """Should attach lean shadow tile to evidence pack."""
        tile = build_lean_shadow_tile_for_global_health(ok_radar)
        
        evidence = {
            "timestamp": "2024-01-01T00:00:00Z",
            "run_id": "test_run_001",
        }
        
        enriched = attach_lean_shadow_to_evidence(evidence, tile)
        
        assert "governance" in enriched
        assert "lean_shadow" in enriched["governance"]
        assert enriched["governance"]["lean_shadow"]["status"] == "OK"

    def test_attach_lean_shadow_creates_governance_key(self, ok_radar: dict) -> None:
        """Should create governance key if it doesn't exist."""
        tile = build_lean_shadow_tile_for_global_health(ok_radar)
        
        evidence = {"timestamp": "2024-01-01T00:00:00Z"}
        
        enriched = attach_lean_shadow_to_evidence(evidence, tile)
        
        assert "governance" in enriched

    def test_attach_lean_shadow_preserves_existing_governance(self, ok_radar: dict) -> None:
        """Should preserve existing governance fields."""
        tile = build_lean_shadow_tile_for_global_health(ok_radar)
        
        evidence = {
            "timestamp": "2024-01-01T00:00:00Z",
            "governance": {
                "other_tile": {"status": "OK"},
            },
        }
        
        enriched = attach_lean_shadow_to_evidence(evidence, tile)
        
        assert "other_tile" in enriched["governance"]
        assert "lean_shadow" in enriched["governance"]

    def test_attach_lean_shadow_includes_key_fields(self, ok_radar: dict) -> None:
        """Should include key fields in evidence tile."""
        tile = build_lean_shadow_tile_for_global_health(ok_radar)
        
        evidence = {"timestamp": "2024-01-01T00:00:00Z"}
        enriched = attach_lean_shadow_to_evidence(evidence, tile)
        
        shadow_evidence = enriched["governance"]["lean_shadow"]
        
        assert "status" in shadow_evidence
        assert "structural_error_rate" in shadow_evidence
        assert "shadow_resource_band" in shadow_evidence
        assert "dominant_anomalies" in shadow_evidence

    def test_attach_lean_shadow_deterministic(self, ok_radar: dict) -> None:
        """Evidence attachment should be deterministic."""
        tile = build_lean_shadow_tile_for_global_health(ok_radar)
        
        evidence = {"timestamp": "2024-01-01T00:00:00Z"}
        
        enriched1 = attach_lean_shadow_to_evidence(evidence, tile)
        enriched2 = attach_lean_shadow_to_evidence(evidence, tile)
        
        assert enriched1 == enriched2

    def test_attach_lean_shadow_non_mutating(self, ok_radar: dict) -> None:
        """Evidence attachment should not mutate input."""
        tile = build_lean_shadow_tile_for_global_health(ok_radar)
        
        evidence = {"timestamp": "2024-01-01T00:00:00Z"}
        evidence_copy = evidence.copy()
        
        enriched = attach_lean_shadow_to_evidence(evidence, tile)
        
        # Original evidence should be unchanged
        assert evidence == evidence_copy
        # Enriched should be different
        assert enriched != evidence

    def test_attach_lean_shadow_json_serializable(self, ok_radar: dict) -> None:
        """Evidence with lean shadow should be JSON serializable."""
        import json
        
        tile = build_lean_shadow_tile_for_global_health(ok_radar)
        
        evidence = {"timestamp": "2024-01-01T00:00:00Z"}
        enriched = attach_lean_shadow_to_evidence(evidence, tile)
        
        # Should not raise
        json_str = json.dumps(enriched)
        assert isinstance(json_str, str)
        
        # Should round-trip
        parsed = json.loads(json_str)
        assert "governance" in parsed
        assert "lean_shadow" in parsed["governance"]


# =============================================================================
# FIRST LIGHT STRUCTURE ANNEX
# =============================================================================

@pytest.mark.unit
class TestFirstLightLeanShadowSummary:
    """Test First Light Lean shadow summary."""

    def test_build_first_light_summary_basic(self, ok_radar: dict) -> None:
        """Should build First Light summary from shadow tile."""
        tile = build_lean_shadow_tile_for_global_health(ok_radar)
        summary = build_first_light_lean_shadow_summary(tile)
        
        assert summary["schema_version"] == "1.0.0"
        assert summary["status"] == "OK"
        assert "structural_error_rate" in summary
        assert "shadow_resource_band" in summary
        assert "dominant_anomalies" in summary

    def test_build_first_light_summary_required_fields(self, ok_radar: dict) -> None:
        """Summary should have all required fields."""
        tile = build_lean_shadow_tile_for_global_health(ok_radar)
        summary = build_first_light_lean_shadow_summary(tile)
        
        required_fields = [
            "schema_version",
            "status",
            "structural_error_rate",
            "shadow_resource_band",
            "dominant_anomalies",
        ]
        
        for field in required_fields:
            assert field in summary, f"Missing required field: {field}"

    def test_build_first_light_summary_truncates_anomalies(self) -> None:
        """Summary should truncate anomalies to top 5."""
        # Note: The tile itself only stores top 3 anomalies, so we need to test
        # with a tile that has more than 5 anomalies directly
        tile = {
            "schema_version": "1.0.0",
            "status": "OK",
            "structural_error_rate": 0.1,
            "shadow_resource_band": "LOW",
            "dominant_anomalies": ["a", "b", "c", "d", "e", "f", "g", "h"],  # 8 anomalies
            "headline": "Test headline",
        }
        summary = build_first_light_lean_shadow_summary(tile)
        
        # Should have at most 5 anomalies
        assert len(summary["dominant_anomalies"]) <= 5
        # Should have exactly 5 when tile has more than 5
        assert len(summary["dominant_anomalies"]) == 5
        assert summary["dominant_anomalies"] == ["a", "b", "c", "d", "e"]

    def test_build_first_light_summary_preserves_anomalies_if_less_than_5(self) -> None:
        """Summary should preserve all anomalies if less than 5."""
        radar = {
            "structural_error_rate": 0.1,
            "shadow_resource_band": "LOW",
            "anomaly_signatures": ["a", "b", "c"],
            "total_shadow_requests": 100,
        }
        tile = build_lean_shadow_tile_for_global_health(radar)
        summary = build_first_light_lean_shadow_summary(tile)
        
        # Should preserve all 3 anomalies
        assert len(summary["dominant_anomalies"]) == 3
        assert summary["dominant_anomalies"] == ["a", "b", "c"]

    def test_build_first_light_summary_deterministic(self, ok_radar: dict) -> None:
        """Summary should be deterministic for same inputs."""
        tile = build_lean_shadow_tile_for_global_health(ok_radar)
        
        summary1 = build_first_light_lean_shadow_summary(tile)
        summary2 = build_first_light_lean_shadow_summary(tile)
        
        assert summary1 == summary2

    def test_build_first_light_summary_json_serializable(self, ok_radar: dict) -> None:
        """Summary should be JSON serializable."""
        import json
        
        tile = build_lean_shadow_tile_for_global_health(ok_radar)
        summary = build_first_light_lean_shadow_summary(tile)
        
        # Should not raise
        json_str = json.dumps(summary)
        assert isinstance(json_str, str)
        
        # Should round-trip
        parsed = json.loads(json_str)
        assert parsed["schema_version"] == summary["schema_version"]

    def test_build_first_light_summary_extracts_correct_fields(self, ok_radar: dict) -> None:
        """Summary should extract correct fields from tile."""
        tile = build_lean_shadow_tile_for_global_health(ok_radar)
        summary = build_first_light_lean_shadow_summary(tile)
        
        assert summary["status"] == tile["status"]
        assert summary["structural_error_rate"] == tile["structural_error_rate"]
        assert summary["shadow_resource_band"] == tile["shadow_resource_band"]

    def test_attach_lean_shadow_with_first_light_summary(self, ok_radar: dict) -> None:
        """Should attach First Light summary when requested."""
        tile = build_lean_shadow_tile_for_global_health(ok_radar)
        
        evidence = {"timestamp": "2024-01-01T00:00:00Z"}
        enriched = attach_lean_shadow_to_evidence(evidence, tile, include_first_light_summary=True)
        
        assert "first_light_summary" in enriched["governance"]["lean_shadow"]
        summary = enriched["governance"]["lean_shadow"]["first_light_summary"]
        assert summary["schema_version"] == "1.0.0"
        assert summary["status"] == "OK"

    def test_attach_lean_shadow_without_first_light_summary(self, ok_radar: dict) -> None:
        """Should not attach First Light summary when not requested."""
        tile = build_lean_shadow_tile_for_global_health(ok_radar)
        
        evidence = {"timestamp": "2024-01-01T00:00:00Z"}
        enriched = attach_lean_shadow_to_evidence(evidence, tile, include_first_light_summary=False)
        
        assert "first_light_summary" not in enriched["governance"]["lean_shadow"]

    def test_attach_lean_shadow_default_no_first_light_summary(self, ok_radar: dict) -> None:
        """Should not attach First Light summary by default."""
        tile = build_lean_shadow_tile_for_global_health(ok_radar)
        
        evidence = {"timestamp": "2024-01-01T00:00:00Z"}
        enriched = attach_lean_shadow_to_evidence(evidence, tile)
        
        assert "first_light_summary" not in enriched["governance"]["lean_shadow"]

    def test_first_light_summary_in_evidence_json_serializable(self, ok_radar: dict) -> None:
        """Evidence with First Light summary should be JSON serializable."""
        import json
        
        tile = build_lean_shadow_tile_for_global_health(ok_radar)
        evidence = {"timestamp": "2024-01-01T00:00:00Z"}
        enriched = attach_lean_shadow_to_evidence(evidence, tile, include_first_light_summary=True)
        
        # Should not raise
        json_str = json.dumps(enriched)
        assert isinstance(json_str, str)
        
        # Should round-trip
        parsed = json.loads(json_str)
        assert "first_light_summary" in parsed["governance"]["lean_shadow"]

    def test_first_light_summary_deterministic_in_evidence(self, ok_radar: dict) -> None:
        """First Light summary in evidence should be deterministic."""
        tile = build_lean_shadow_tile_for_global_health(ok_radar)
        
        evidence = {"timestamp": "2024-01-01T00:00:00Z"}
        enriched1 = attach_lean_shadow_to_evidence(evidence, tile, include_first_light_summary=True)
        enriched2 = attach_lean_shadow_to_evidence(evidence, tile, include_first_light_summary=True)
        
        assert enriched1 == enriched2
        assert enriched1["governance"]["lean_shadow"]["first_light_summary"] == \
               enriched2["governance"]["lean_shadow"]["first_light_summary"]


# =============================================================================
# STRUCTURAL COHESION LINK
# =============================================================================

@pytest.mark.unit
class TestStructuralCohesionLink:
    """Test structural cohesion link in global health."""

    def test_structure_lean_shadow_status_added(self, ok_radar: dict) -> None:
        """Should add lean_shadow_status to structure tile."""
        from backend.health.global_surface import build_global_health_surface
        
        payload = build_global_health_surface(
            base_payload={"test": "data"},
            shadow_radar=ok_radar,
        )
        
        assert "structure" in payload
        assert "lean_shadow_status" in payload["structure"]
        assert payload["structure"]["lean_shadow_status"] == "OK"

    def test_structure_lean_shadow_status_warn(self, warn_radar_medium_error: dict) -> None:
        """Should set lean_shadow_status to WARN when appropriate."""
        from backend.health.global_surface import build_global_health_surface
        
        payload = build_global_health_surface(
            base_payload={"test": "data"},
            shadow_radar=warn_radar_medium_error,
        )
        
        assert payload["structure"]["lean_shadow_status"] == "WARN"

    def test_structure_lean_shadow_status_block(self, block_radar_high_error: dict) -> None:
        """Should set lean_shadow_status to BLOCK when appropriate."""
        from backend.health.global_surface import build_global_health_surface
        
        payload = build_global_health_surface(
            base_payload={"test": "data"},
            shadow_radar=block_radar_high_error,
        )
        
        assert payload["structure"]["lean_shadow_status"] == "BLOCK"

    def test_structure_created_if_not_exists(self, ok_radar: dict) -> None:
        """Should create structure key if it doesn't exist."""
        from backend.health.global_surface import build_global_health_surface
        
        payload = build_global_health_surface(
            base_payload={},
            shadow_radar=ok_radar,
        )
        
        assert "structure" in payload

    def test_structure_preserves_existing_fields(self, ok_radar: dict) -> None:
        """Should preserve existing structure fields."""
        from backend.health.global_surface import build_global_health_surface
        
        payload = build_global_health_surface(
            base_payload={"structure": {"other_status": "OK"}},
            shadow_radar=ok_radar,
        )
        
        assert payload["structure"]["other_status"] == "OK"
        assert payload["structure"]["lean_shadow_status"] == "OK"


# =============================================================================
# CALIBRATION EXPERIMENT STRUCTURAL SUMMARY
# =============================================================================

@pytest.mark.unit
class TestCalExpStructuralSummary:
    """Test calibration experiment structural summary."""

    def test_build_cal_exp_summary_empty_sequence(self) -> None:
        """Should handle empty tile sequence."""
        summary = build_cal_exp_structural_summary([])
        
        assert summary["schema_version"] == CAL_EXP_STRUCTURAL_SUMMARY_SCHEMA_VERSION
        assert summary["mean_structural_error_rate"] == 0.0
        assert summary["max_structural_error_rate"] == 0.0
        assert summary["anomaly_bursts"] == []
        assert summary["dominant_anomalies"] == []

    def test_build_cal_exp_summary_single_tile(self, ok_radar: dict) -> None:
        """Should handle single tile."""
        tile = build_lean_shadow_tile_for_global_health(ok_radar)
        summary = build_cal_exp_structural_summary([tile])
        
        assert summary["mean_structural_error_rate"] == 0.1
        assert summary["max_structural_error_rate"] == 0.1
        assert len(summary["anomaly_bursts"]) == 0

    def test_build_cal_exp_summary_multiple_tiles(self) -> None:
        """Should compute mean and max across multiple tiles."""
        radar1 = {
            "structural_error_rate": 0.1,
            "shadow_resource_band": "LOW",
            "anomaly_signatures": ["a"],
            "total_shadow_requests": 100,
        }
        radar2 = {
            "structural_error_rate": 0.3,
            "shadow_resource_band": "MEDIUM",
            "anomaly_signatures": ["b"],
            "total_shadow_requests": 50,
        }
        radar3 = {
            "structural_error_rate": 0.2,
            "shadow_resource_band": "LOW",
            "anomaly_signatures": ["c"],
            "total_shadow_requests": 75,
        }
        
        tiles = [
            build_lean_shadow_tile_for_global_health(radar1),
            build_lean_shadow_tile_for_global_health(radar2),
            build_lean_shadow_tile_for_global_health(radar3),
        ]
        
        summary = build_cal_exp_structural_summary(tiles)
        
        assert summary["mean_structural_error_rate"] == 0.2  # (0.1 + 0.3 + 0.2) / 3
        assert summary["max_structural_error_rate"] == 0.3

    def test_build_cal_exp_summary_detects_sustained_burst(self) -> None:
        """Should detect sustained anomaly bursts (3+ consecutive high error rates)."""
        # Create sequence with sustained burst
        tiles = []
        for i in range(10):
            error_rate = 0.4 if 3 <= i <= 6 else 0.1  # Burst from index 3-6 (4 consecutive)
            radar = {
                "structural_error_rate": error_rate,
                "shadow_resource_band": "LOW",
                "anomaly_signatures": [],
                "total_shadow_requests": 100,
            }
            tiles.append(build_lean_shadow_tile_for_global_health(radar))
        
        summary = build_cal_exp_structural_summary(tiles)
        
        assert len(summary["anomaly_bursts"]) == 1
        burst = summary["anomaly_bursts"][0]
        assert burst["start_index"] == 3
        assert burst["end_index"] == 6
        assert burst["length"] == 4

    def test_build_cal_exp_summary_ignores_single_spike(self) -> None:
        """Should ignore single spikes (less than 3 consecutive high error rates)."""
        # Create sequence with single spike
        tiles = []
        for i in range(10):
            error_rate = 0.4 if i == 5 else 0.1  # Single spike at index 5
            radar = {
                "structural_error_rate": error_rate,
                "shadow_resource_band": "LOW",
                "anomaly_signatures": [],
                "total_shadow_requests": 100,
            }
            tiles.append(build_lean_shadow_tile_for_global_health(radar))
        
        summary = build_cal_exp_structural_summary(tiles)
        
        # Should not detect burst (single spike < 3 consecutive)
        assert len(summary["anomaly_bursts"]) == 0

    def test_build_cal_exp_summary_ignores_two_consecutive(self) -> None:
        """Should ignore two consecutive high error rates (not a burst)."""
        # Create sequence with two consecutive high rates
        tiles = []
        for i in range(10):
            error_rate = 0.4 if i in (5, 6) else 0.1  # Two consecutive at index 5-6
            radar = {
                "structural_error_rate": error_rate,
                "shadow_resource_band": "LOW",
                "anomaly_signatures": [],
                "total_shadow_requests": 100,
            }
            tiles.append(build_lean_shadow_tile_for_global_health(radar))
        
        summary = build_cal_exp_structural_summary(tiles)
        
        # Should not detect burst (only 2 consecutive, need 3+)
        assert len(summary["anomaly_bursts"]) == 0

    def test_build_cal_exp_summary_multiple_bursts(self) -> None:
        """Should detect multiple separate bursts."""
        # Create sequence with two separate bursts
        tiles = []
        for i in range(15):
            # First burst: 2-4, second burst: 10-12
            error_rate = 0.4 if (2 <= i <= 4) or (10 <= i <= 12) else 0.1
            radar = {
                "structural_error_rate": error_rate,
                "shadow_resource_band": "LOW",
                "anomaly_signatures": [],
                "total_shadow_requests": 100,
            }
            tiles.append(build_lean_shadow_tile_for_global_health(radar))
        
        summary = build_cal_exp_structural_summary(tiles)
        
        assert len(summary["anomaly_bursts"]) == 2
        assert summary["anomaly_bursts"][0]["start_index"] == 2
        assert summary["anomaly_bursts"][0]["end_index"] == 4
        assert summary["anomaly_bursts"][1]["start_index"] == 10
        assert summary["anomaly_bursts"][1]["end_index"] == 12

    def test_build_cal_exp_summary_aggregates_anomalies(self) -> None:
        """Should aggregate dominant anomalies across all tiles."""
        radar1 = {
            "structural_error_rate": 0.1,
            "shadow_resource_band": "LOW",
            "anomaly_signatures": ["a", "b"],
            "total_shadow_requests": 100,
        }
        radar2 = {
            "structural_error_rate": 0.2,
            "shadow_resource_band": "LOW",
            "anomaly_signatures": ["b", "c"],
            "total_shadow_requests": 100,
        }
        radar3 = {
            "structural_error_rate": 0.1,
            "shadow_resource_band": "LOW",
            "anomaly_signatures": ["a", "c", "d"],
            "total_shadow_requests": 100,
        }
        
        tiles = [
            build_lean_shadow_tile_for_global_health(radar1),
            build_lean_shadow_tile_for_global_health(radar2),
            build_lean_shadow_tile_for_global_health(radar3),
        ]
        
        summary = build_cal_exp_structural_summary(tiles)
        
        # "a" appears 2 times, "b" appears 2 times, "c" appears 2 times, "d" appears 1 time
        # Should be sorted by frequency
        assert "a" in summary["dominant_anomalies"]
        assert "b" in summary["dominant_anomalies"]
        assert "c" in summary["dominant_anomalies"]

    def test_build_cal_exp_summary_deterministic(self) -> None:
        """Summary should be deterministic for same inputs."""
        radar1 = {
            "structural_error_rate": 0.1,
            "shadow_resource_band": "LOW",
            "anomaly_signatures": ["a"],
            "total_shadow_requests": 100,
        }
        radar2 = {
            "structural_error_rate": 0.3,
            "shadow_resource_band": "MEDIUM",
            "anomaly_signatures": ["b"],
            "total_shadow_requests": 50,
        }
        
        tiles = [
            build_lean_shadow_tile_for_global_health(radar1),
            build_lean_shadow_tile_for_global_health(radar2),
        ]
        
        summary1 = build_cal_exp_structural_summary(tiles)
        summary2 = build_cal_exp_structural_summary(tiles)
        
        assert summary1 == summary2

    def test_build_cal_exp_summary_json_serializable(self) -> None:
        """Summary should be JSON serializable."""
        import json
        
        radar1 = {
            "structural_error_rate": 0.1,
            "shadow_resource_band": "LOW",
            "anomaly_signatures": ["a"],
            "total_shadow_requests": 100,
        }
        tiles = [build_lean_shadow_tile_for_global_health(radar1)]
        summary = build_cal_exp_structural_summary(tiles)
        
        # Should not raise
        json_str = json.dumps(summary)
        assert isinstance(json_str, str)
        
        # Should round-trip
        parsed = json.loads(json_str)
        assert parsed["schema_version"] == summary["schema_version"]

    def test_build_cal_exp_summary_burst_at_end(self) -> None:
        """Should detect burst that extends to end of sequence."""
        tiles = []
        for i in range(10):
            error_rate = 0.4 if i >= 7 else 0.1  # Burst from index 7 to end
            radar = {
                "structural_error_rate": error_rate,
                "shadow_resource_band": "LOW",
                "anomaly_signatures": [],
                "total_shadow_requests": 100,
            }
            tiles.append(build_lean_shadow_tile_for_global_health(radar))
        
        summary = build_cal_exp_structural_summary(tiles)
        
        assert len(summary["anomaly_bursts"]) == 1
        burst = summary["anomaly_bursts"][0]
        assert burst["start_index"] == 7
        assert burst["end_index"] == 9
        assert burst["length"] == 3

    def test_attach_cal_exp_summary_to_report(self) -> None:
        """Should attach structural summary to calibration report."""
        report = {
            "schema_version": "1.0.0",
            "summary": {"final_divergence_rate": 0.1},
        }
        
        radar1 = {
            "structural_error_rate": 0.1,
            "shadow_resource_band": "LOW",
            "anomaly_signatures": ["a"],
            "total_shadow_requests": 100,
        }
        tiles = [build_lean_shadow_tile_for_global_health(radar1)]
        
        enriched = attach_cal_exp_structural_summary_to_report(report, tiles)
        
        assert "structural_summary" in enriched
        assert enriched["structural_summary"]["schema_version"] == CAL_EXP_STRUCTURAL_SUMMARY_SCHEMA_VERSION

    def test_attach_cal_exp_summary_non_mutating(self) -> None:
        """Attachment should not mutate input report."""
        report = {
            "schema_version": "1.0.0",
            "summary": {"final_divergence_rate": 0.1},
        }
        report_copy = report.copy()
        
        tiles = []
        enriched = attach_cal_exp_structural_summary_to_report(report, tiles)
        
        # Original report should be unchanged
        assert report == report_copy
        # Enriched should be different
        assert enriched != report


# =============================================================================
# LEAN VS STRUCTURAL CROSS-CHECK
# =============================================================================

@pytest.mark.unit
class TestLeanVsStructuralCrossCheck:
    """Test Lean vs Structural signal cross-check."""

    def test_compare_consistent_both_ok(self) -> None:
        """Should return CONSISTENT when both signals are OK/CONSISTENT."""
        lean_summary = {
            "mean_structural_error_rate": 0.1,
            "max_structural_error_rate": 0.15,
            "anomaly_bursts": [],
        }
        structural_summary = {
            "combined_severity": "CONSISTENT",
            "cohesion_score": 0.95,
        }
        
        comparison = compare_lean_vs_structural_signal(lean_summary, structural_summary)
        
        assert comparison["status"] == "CONSISTENT"
        assert comparison["lean_signal_severity"] == "OK"
        assert comparison["structural_signal_severity"] == "CONSISTENT"

    def test_compare_consistent_both_warn(self) -> None:
        """Should return CONSISTENT when both signals are WARN/TENSION."""
        lean_summary = {
            "mean_structural_error_rate": 0.25,
            "max_structural_error_rate": 0.3,
            "anomaly_bursts": [],
        }
        structural_summary = {
            "combined_severity": "TENSION",
            "cohesion_score": 0.75,
        }
        
        comparison = compare_lean_vs_structural_signal(lean_summary, structural_summary)
        
        assert comparison["status"] == "CONSISTENT"
        assert comparison["lean_signal_severity"] == "WARN"
        assert comparison["structural_signal_severity"] == "TENSION"

    def test_compare_consistent_both_block(self) -> None:
        """Should return CONSISTENT when both signals are BLOCK/CONFLICT."""
        lean_summary = {
            "mean_structural_error_rate": 0.6,
            "max_structural_error_rate": 0.7,
            "anomaly_bursts": [{"start_index": 0, "end_index": 5, "length": 6}],
        }
        structural_summary = {
            "combined_severity": "CONFLICT",
            "cohesion_score": 0.3,
        }
        
        comparison = compare_lean_vs_structural_signal(lean_summary, structural_summary)
        
        assert comparison["status"] == "CONSISTENT"
        assert comparison["lean_signal_severity"] == "BLOCK"
        assert comparison["structural_signal_severity"] == "CONFLICT"

    def test_compare_conflict_ok_vs_conflict(self) -> None:
        """Should return CONFLICT when Lean is OK but Structural is CONFLICT."""
        lean_summary = {
            "mean_structural_error_rate": 0.1,
            "max_structural_error_rate": 0.15,
            "anomaly_bursts": [],
        }
        structural_summary = {
            "combined_severity": "CONFLICT",
            "cohesion_score": 0.3,
        }
        
        comparison = compare_lean_vs_structural_signal(lean_summary, structural_summary)
        
        assert comparison["status"] == "CONFLICT"
        assert "strongly disagree" in comparison["advisory_notes"][0].lower()

    def test_compare_conflict_block_vs_consistent(self) -> None:
        """Should return CONFLICT when Lean is BLOCK but Structural is CONSISTENT."""
        lean_summary = {
            "mean_structural_error_rate": 0.6,
            "max_structural_error_rate": 0.7,
            "anomaly_bursts": [{"start_index": 0, "end_index": 5, "length": 6}],
        }
        structural_summary = {
            "combined_severity": "CONSISTENT",
            "cohesion_score": 0.95,
        }
        
        comparison = compare_lean_vs_structural_signal(lean_summary, structural_summary)
        
        assert comparison["status"] == "CONFLICT"
        assert "strongly disagree" in comparison["advisory_notes"][0].lower()

    def test_compare_tension_warn_vs_consistent(self) -> None:
        """Should return TENSION when Lean is WARN but Structural is CONSISTENT."""
        lean_summary = {
            "mean_structural_error_rate": 0.25,
            "max_structural_error_rate": 0.3,
            "anomaly_bursts": [],
        }
        structural_summary = {
            "combined_severity": "CONSISTENT",
            "cohesion_score": 0.95,
        }
        
        comparison = compare_lean_vs_structural_signal(lean_summary, structural_summary)
        
        assert comparison["status"] == "TENSION"
        assert "partially disagree" in comparison["advisory_notes"][0].lower()

    def test_compare_includes_advisory_notes(self) -> None:
        """Comparison should include advisory notes."""
        lean_summary = {
            "mean_structural_error_rate": 0.35,
            "max_structural_error_rate": 0.4,
            "anomaly_bursts": [],
        }
        structural_summary = {
            "combined_severity": "CONSISTENT",
            "cohesion_score": 0.6,
        }
        
        comparison = compare_lean_vs_structural_signal(lean_summary, structural_summary)
        
        assert len(comparison["advisory_notes"]) > 0
        assert isinstance(comparison["advisory_notes"], list)

    def test_compare_notes_high_error_rate(self) -> None:
        """Should include note about high error rate."""
        lean_summary = {
            "mean_structural_error_rate": 0.35,
            "max_structural_error_rate": 0.4,
            "anomaly_bursts": [],
        }
        structural_summary = {
            "combined_severity": "CONSISTENT",
            "cohesion_score": 0.95,
        }
        
        comparison = compare_lean_vs_structural_signal(lean_summary, structural_summary)
        
        # Should have note about elevated error rate
        notes_text = " ".join(comparison["advisory_notes"]).lower()
        assert "error rate" in notes_text or "elevated" in notes_text

    def test_compare_notes_low_cohesion(self) -> None:
        """Should include note about low cohesion score."""
        lean_summary = {
            "mean_structural_error_rate": 0.1,
            "max_structural_error_rate": 0.15,
            "anomaly_bursts": [],
        }
        structural_summary = {
            "combined_severity": "CONSISTENT",
            "cohesion_score": 0.6,
        }
        
        comparison = compare_lean_vs_structural_signal(lean_summary, structural_summary)
        
        # Should have note about low cohesion
        notes_text = " ".join(comparison["advisory_notes"]).lower()
        assert "cohesion" in notes_text or "low" in notes_text

    def test_compare_notes_anomaly_bursts(self) -> None:
        """Should include note about anomaly bursts."""
        lean_summary = {
            "mean_structural_error_rate": 0.1,
            "max_structural_error_rate": 0.15,
            "anomaly_bursts": [
                {"start_index": 0, "end_index": 3, "length": 4},
                {"start_index": 10, "end_index": 12, "length": 3},
            ],
        }
        structural_summary = {
            "combined_severity": "CONSISTENT",
            "cohesion_score": 0.95,
        }
        
        comparison = compare_lean_vs_structural_signal(lean_summary, structural_summary)
        
        # Should have note about bursts
        notes_text = " ".join(comparison["advisory_notes"]).lower()
        assert "burst" in notes_text

    def test_compare_deterministic(self) -> None:
        """Comparison should be deterministic for same inputs."""
        lean_summary = {
            "mean_structural_error_rate": 0.2,
            "max_structural_error_rate": 0.3,
            "anomaly_bursts": [],
        }
        structural_summary = {
            "combined_severity": "TENSION",
            "cohesion_score": 0.8,
        }
        
        comparison1 = compare_lean_vs_structural_signal(lean_summary, structural_summary)
        comparison2 = compare_lean_vs_structural_signal(lean_summary, structural_summary)
        
        assert comparison1 == comparison2

    def test_compare_json_serializable(self) -> None:
        """Comparison should be JSON serializable."""
        import json
        
        lean_summary = {
            "mean_structural_error_rate": 0.2,
            "max_structural_error_rate": 0.3,
            "anomaly_bursts": [],
        }
        structural_summary = {
            "combined_severity": "TENSION",
            "cohesion_score": 0.8,
        }
        
        comparison = compare_lean_vs_structural_signal(lean_summary, structural_summary)
        
        # Should not raise
        json_str = json.dumps(comparison)
        assert isinstance(json_str, str)
        
        # Should round-trip
        parsed = json.loads(json_str)
        assert parsed["status"] == comparison["status"]

    def test_attach_lean_shadow_with_cross_check(self, ok_radar: dict) -> None:
        """Should attach cross-check to evidence when provided."""
        tile = build_lean_shadow_tile_for_global_health(ok_radar)
        
        lean_summary = build_cal_exp_structural_summary([tile])
        structural_summary = {
            "combined_severity": "CONSISTENT",
            "cohesion_score": 0.95,
        }
        cross_check = compare_lean_vs_structural_signal(lean_summary, structural_summary)
        
        evidence = {"timestamp": "2024-01-01T00:00:00Z"}
        enriched = attach_lean_shadow_to_evidence(evidence, tile, lean_cross_check=cross_check)
        
        assert "governance" in enriched
        assert "structure" in enriched["governance"]
        assert "lean_cross_check" in enriched["governance"]["structure"]
        assert enriched["governance"]["structure"]["lean_cross_check"]["status"] == "CONSISTENT"

    def test_attach_lean_shadow_without_cross_check(self, ok_radar: dict) -> None:
        """Should not attach cross-check when not provided."""
        tile = build_lean_shadow_tile_for_global_health(ok_radar)
        
        evidence = {"timestamp": "2024-01-01T00:00:00Z"}
        enriched = attach_lean_shadow_to_evidence(evidence, tile)
        
        # Should not have cross-check if not provided
        if "governance" in enriched and "structure" in enriched["governance"]:
            assert "lean_cross_check" not in enriched["governance"]["structure"]


# =============================================================================
# STRUCTURAL CALIBRATION PANEL
# =============================================================================

@pytest.mark.unit
class TestStructuralCalibrationPanel:
    """Test structural calibration panel builder."""

    def test_build_panel_empty_reports(self) -> None:
        """Should handle empty report list."""
        panel = build_structural_calibration_panel([])
        
        assert panel["schema_version"] == STRUCTURAL_CALIBRATION_PANEL_SCHEMA_VERSION
        assert panel["experiments"] == []
        assert panel["counts"]["num_experiments"] == 0
        assert panel["counts"]["num_burst"] == 0
        assert panel["counts"]["num_conflict"] == 0

    def test_build_panel_single_report(self) -> None:
        """Should handle single report."""
        report = {
            "cal_id": "CAL-EXP-1",
            "structural_summary": {
                "mean_structural_error_rate": 0.1,
                "max_structural_error_rate": 0.2,
                "anomaly_bursts": [],
            },
        }
        panel = build_structural_calibration_panel([report])
        
        assert len(panel["experiments"]) == 1
        exp = panel["experiments"][0]
        assert exp["cal_id"] == "CAL-EXP-1"
        assert exp["mean_error_rate"] == 0.1
        assert exp["max_error_rate"] == 0.2
        assert exp["burst_detected"] is False
        # Missing cross-check -> UNKNOWN (hardened)
        assert exp["cross_check_status"] == "UNKNOWN"

    def test_build_panel_multiple_reports(self) -> None:
        """Should handle multiple reports."""
        report1 = {
            "cal_id": "CAL-EXP-1",
            "structural_summary": {
                "mean_structural_error_rate": 0.1,
                "max_structural_error_rate": 0.2,
                "anomaly_bursts": [],
            },
        }
        report2 = {
            "cal_id": "CAL-EXP-2",
            "structural_summary": {
                "mean_structural_error_rate": 0.3,
                "max_structural_error_rate": 0.4,
                "anomaly_bursts": [{"start_index": 0, "end_index": 2, "length": 3}],
            },
        }
        panel = build_structural_calibration_panel([report1, report2])
        
        assert panel["counts"]["num_experiments"] == 2
        assert panel["counts"]["num_burst"] == 1
        assert panel["counts"]["num_conflict"] == 0

    def test_build_panel_detects_bursts(self) -> None:
        """Should detect bursts in reports."""
        report = {
            "cal_id": "CAL-EXP-1",
            "structural_summary": {
                "mean_structural_error_rate": 0.1,
                "max_structural_error_rate": 0.2,
                "anomaly_bursts": [
                    {"start_index": 0, "end_index": 2, "length": 3},
                    {"start_index": 10, "end_index": 12, "length": 3},
                ],
            },
        }
        panel = build_structural_calibration_panel([report])
        
        assert panel["counts"]["num_burst"] == 1
        assert panel["experiments"][0]["burst_detected"] is True

    def test_build_panel_detects_conflicts(self) -> None:
        """Should detect conflicts from cross-check."""
        report = {
            "cal_id": "CAL-EXP-1",
            "structural_summary": {
                "mean_structural_error_rate": 0.1,
                "max_structural_error_rate": 0.2,
                "anomaly_bursts": [],
            },
            "governance": {
                "structure": {
                    "lean_cross_check": {
                        "status": "CONFLICT",
                    },
                },
            },
        }
        panel = build_structural_calibration_panel([report])
        
        assert panel["counts"]["num_conflict"] == 1
        assert panel["experiments"][0]["cross_check_status"] == "CONFLICT"

    def test_build_panel_extracts_cal_id_variants(self) -> None:
        """Should extract cal_id from various report formats."""
        # Test cal_id
        report1 = {"cal_id": "CAL-EXP-1", "structural_summary": {}}
        panel1 = build_structural_calibration_panel([report1])
        assert panel1["experiments"][0]["cal_id"] == "CAL-EXP-1"
        
        # Test experiment_id
        report2 = {"experiment_id": "CAL-EXP-2", "structural_summary": {}}
        panel2 = build_structural_calibration_panel([report2])
        assert panel2["experiments"][0]["cal_id"] == "CAL-EXP-2"
        
        # Test run_id fallback
        report3 = {"run_id": "cal_exp3_20250101_120000", "structural_summary": {}}
        panel3 = build_structural_calibration_panel([report3])
        assert panel3["experiments"][0]["cal_id"] == "cal_exp3"

    def test_build_panel_deterministic(self) -> None:
        """Panel should be deterministic for same inputs."""
        report1 = {
            "cal_id": "CAL-EXP-1",
            "structural_summary": {
                "mean_structural_error_rate": 0.1,
                "max_structural_error_rate": 0.2,
                "anomaly_bursts": [],
            },
        }
        report2 = {
            "cal_id": "CAL-EXP-2",
            "structural_summary": {
                "mean_structural_error_rate": 0.3,
                "max_structural_error_rate": 0.4,
                "anomaly_bursts": [{"start_index": 0}],
            },
        }
        
        panel1 = build_structural_calibration_panel([report1, report2])
        panel2 = build_structural_calibration_panel([report1, report2])
        
        assert panel1 == panel2

    def test_build_panel_json_serializable(self) -> None:
        """Panel should be JSON serializable."""
        import json
        
        report = {
            "cal_id": "CAL-EXP-1",
            "structural_summary": {
                "mean_structural_error_rate": 0.1,
                "max_structural_error_rate": 0.2,
                "anomaly_bursts": [],
            },
        }
        panel = build_structural_calibration_panel([report])
        
        # Should not raise
        json_str = json.dumps(panel)
        assert isinstance(json_str, str)
        
        # Should round-trip
        parsed = json.loads(json_str)
        assert parsed["schema_version"] == panel["schema_version"]

    def test_extract_panel_signal(self) -> None:
        """Should extract compact signal from panel."""
        report = {
            "cal_id": "CAL-EXP-1",
            "structural_summary": {
                "mean_structural_error_rate": 0.1,
                "max_structural_error_rate": 0.2,
                "anomaly_bursts": [{"start_index": 0}],
            },
            "governance": {
                "structure": {
                    "lean_cross_check": {
                        "status": "CONFLICT",
                    },
                },
            },
        }
        panel = build_structural_calibration_panel([report])
        signal = extract_structural_calibration_panel_signal(panel)
        
        assert signal["num_conflict"] == 1
        assert signal["num_burst"] == 1
        assert signal["last_cross_check_status"] == "CONFLICT"

    def test_extract_panel_signal_no_cross_check(self) -> None:
        """Should handle missing cross-check status."""
        report = {
            "cal_id": "CAL-EXP-1",
            "structural_summary": {
                "mean_structural_error_rate": 0.1,
                "max_structural_error_rate": 0.2,
                "anomaly_bursts": [],
            },
        }
        panel = build_structural_calibration_panel([report])
        signal = extract_structural_calibration_panel_signal(panel)
        
        assert signal["num_conflict"] == 0
        assert signal["num_burst"] == 0
        # Missing cross-check -> UNKNOWN (hardened)
        assert signal["last_cross_check_status"] == "UNKNOWN"

    def test_extract_panel_signal_last_status(self) -> None:
        """Should extract last cross-check status from multiple experiments."""
        report1 = {
            "cal_id": "CAL-EXP-1",
            "structural_summary": {"anomaly_bursts": []},
            "governance": {
                "structure": {
                    "lean_cross_check": {"status": "CONSISTENT"},
                },
            },
        }
        report2 = {
            "cal_id": "CAL-EXP-2",
            "structural_summary": {"anomaly_bursts": []},
            "governance": {
                "structure": {
                    "lean_cross_check": {"status": "CONFLICT"},
                },
            },
        }
        panel = build_structural_calibration_panel([report1, report2])
        signal = extract_structural_calibration_panel_signal(panel)
        
        # Should get last status (CONFLICT)
        assert signal["last_cross_check_status"] == "CONFLICT"

    def test_attach_lean_shadow_with_calibration_panel(self, ok_radar: dict) -> None:
        """Should attach calibration panel to evidence when provided."""
        tile = build_lean_shadow_tile_for_global_health(ok_radar)
        
        report = {
            "cal_id": "CAL-EXP-1",
            "structural_summary": {
                "mean_structural_error_rate": 0.1,
                "max_structural_error_rate": 0.2,
                "anomaly_bursts": [],
            },
        }
        panel = build_structural_calibration_panel([report])
        
        evidence = {"timestamp": "2024-01-01T00:00:00Z"}
        enriched = attach_lean_shadow_to_evidence(evidence, tile, structural_calibration_panel=panel)
        
        assert "governance" in enriched
        assert "structure" in enriched["governance"]
        assert "calibration_panel" in enriched["governance"]["structure"]
        assert enriched["governance"]["structure"]["calibration_panel"]["schema_version"] == STRUCTURAL_CALIBRATION_PANEL_SCHEMA_VERSION

    def test_attach_lean_shadow_without_calibration_panel(self, ok_radar: dict) -> None:
        """Should not attach calibration panel when not provided."""
        tile = build_lean_shadow_tile_for_global_health(ok_radar)
        
        evidence = {"timestamp": "2024-01-01T00:00:00Z"}
        enriched = attach_lean_shadow_to_evidence(evidence, tile)
        
        # Should not have calibration panel if not provided
        if "governance" in enriched and "structure" in enriched["governance"]:
            assert "calibration_panel" not in enriched["governance"]["structure"]

    def test_attach_calibration_panel_non_mutating(self) -> None:
        """Attachment should not mutate input evidence."""
        evidence = {"timestamp": "2024-01-01T00:00:00Z"}
        evidence_copy = evidence.copy()
        
        report = {
            "cal_id": "CAL-EXP-1",
            "structural_summary": {"anomaly_bursts": []},
        }
        panel = build_structural_calibration_panel([report])
        
        tile = build_lean_shadow_tile_for_global_health({
            "structural_error_rate": 0.1,
            "shadow_resource_band": "LOW",
            "anomaly_signatures": [],
            "total_shadow_requests": 100,
        })
        
        enriched = attach_lean_shadow_to_evidence(evidence, tile, structural_calibration_panel=panel)
        
        # Original evidence should be unchanged
        assert evidence == evidence_copy
        # Enriched should be different
        assert enriched != evidence

    def test_calibration_panel_in_evidence_json_serializable(self) -> None:
        """Evidence with calibration panel should be JSON serializable."""
        import json
        
        report = {
            "cal_id": "CAL-EXP-1",
            "structural_summary": {"anomaly_bursts": []},
        }
        panel = build_structural_calibration_panel([report])
        
        tile = build_lean_shadow_tile_for_global_health({
            "structural_error_rate": 0.1,
            "shadow_resource_band": "LOW",
            "anomaly_signatures": [],
            "total_shadow_requests": 100,
        })
        
        evidence = {"timestamp": "2024-01-01T00:00:00Z"}
        enriched = attach_lean_shadow_to_evidence(evidence, tile, structural_calibration_panel=panel)
        
        # Should not raise
        json_str = json.dumps(enriched)
        assert isinstance(json_str, str)
        
        # Should round-trip
        parsed = json.loads(json_str)
        assert "calibration_panel" in parsed["governance"]["structure"]


# =============================================================================
# MANIFEST BINDING + STATUS EXTRACTION INTEGRATION
# =============================================================================

@pytest.mark.unit
class TestManifestBindingAndStatusExtraction:
    """Test manifest binding and status extraction integration."""

    def test_manifest_includes_calibration_panel(self) -> None:
        """Manifest should include calibration panel from evidence."""
        # Build evidence with calibration panel
        evidence = {"timestamp": "2024-01-01T00:00:00Z"}
        
        report1 = {
            "cal_id": "CAL-EXP-1",
            "structural_summary": {
                "mean_structural_error_rate": 0.1,
                "max_structural_error_rate": 0.2,
                "anomaly_bursts": [],
            },
        }
        report2 = {
            "cal_id": "CAL-EXP-2",
            "structural_summary": {
                "mean_structural_error_rate": 0.3,
                "max_structural_error_rate": 0.4,
                "anomaly_bursts": [{"start_index": 0, "end_index": 2, "length": 3}],
            },
            "governance": {
                "structure": {
                    "lean_cross_check": {
                        "status": "CONFLICT",
                    },
                },
            },
        }
        
        panel = build_structural_calibration_panel([report1, report2])
        
        tile = build_lean_shadow_tile_for_global_health({
            "structural_error_rate": 0.1,
            "shadow_resource_band": "LOW",
            "anomaly_signatures": [],
            "total_shadow_requests": 100,
        })
        
        enriched_evidence = attach_lean_shadow_to_evidence(evidence, tile, structural_calibration_panel=panel)
        
        # Simulate manifest building (mirroring evidence governance blocks)
        manifest = {
            "schema_version": "1.0.0",
            "mode": "SHADOW",
        }
        
        # Mirror calibration panel from evidence to manifest
        governance = enriched_evidence.get("governance", {})
        structure = governance.get("structure", {})
        calibration_panel = structure.get("calibration_panel")
        if calibration_panel:
            manifest["governance"] = manifest.get("governance", {})
            if "structure" not in manifest["governance"]:
                manifest["governance"]["structure"] = {}
            manifest["governance"]["structure"]["calibration_panel"] = calibration_panel
        
        # Verify manifest includes calibration panel
        assert "governance" in manifest
        assert "structure" in manifest["governance"]
        assert "calibration_panel" in manifest["governance"]["structure"]
        assert manifest["governance"]["structure"]["calibration_panel"]["schema_version"] == STRUCTURAL_CALIBRATION_PANEL_SCHEMA_VERSION

    def test_status_extraction_from_manifest(self) -> None:
        """Status extraction should read calibration panel from manifest."""
        # Build panel
        report1 = {
            "cal_id": "CAL-EXP-1",
            "structural_summary": {
                "mean_structural_error_rate": 0.1,
                "max_structural_error_rate": 0.2,
                "anomaly_bursts": [],
            },
        }
        report2 = {
            "cal_id": "CAL-EXP-2",
            "structural_summary": {
                "mean_structural_error_rate": 0.3,
                "max_structural_error_rate": 0.4,
                "anomaly_bursts": [{"start_index": 0}],
            },
            "governance": {
                "structure": {
                    "lean_cross_check": {
                        "status": "CONFLICT",
                    },
                },
            },
        }
        
        panel = build_structural_calibration_panel([report1, report2])
        
        # Simulate manifest with calibration panel
        manifest = {
            "governance": {
                "structure": {
                    "calibration_panel": panel,
                },
            },
        }
        
        # Extract signal from manifest (simulating status extraction)
        governance = manifest.get("governance", {})
        structure = governance.get("structure", {})
        calibration_panel = structure.get("calibration_panel")
        
        assert calibration_panel is not None
        
        # Extract signal
        signal = extract_structural_calibration_panel_signal(calibration_panel)
        
        # Verify signal includes all required fields
        assert "num_conflict" in signal
        assert "num_burst" in signal
        assert "last_cross_check_status" in signal
        assert "most_recent_cal_id" in signal
        
        # Verify values
        assert signal["num_conflict"] == 1
        assert signal["num_burst"] == 1
        assert signal["last_cross_check_status"] == "CONFLICT"
        assert signal["most_recent_cal_id"] == "CAL-EXP-2"

    def test_status_extraction_missing_panel(self) -> None:
        """Status extraction should handle missing calibration panel gracefully."""
        # Manifest without calibration panel
        manifest = {
            "governance": {
                "structure": {},
            },
        }
        
        # Extract (should not raise)
        governance = manifest.get("governance", {})
        structure = governance.get("structure", {})
        calibration_panel = structure.get("calibration_panel")
        
        # Should be None, not an error
        assert calibration_panel is None

    def test_cross_check_status_hardening(self) -> None:
        """Cross-check status should be hardened to valid values."""
        # Test with invalid status
        report = {
            "cal_id": "CAL-EXP-1",
            "structural_summary": {"anomaly_bursts": []},
            "governance": {
                "structure": {
                    "lean_cross_check": {
                        "status": "INVALID_STATUS",  # Invalid value
                    },
                },
            },
        }
        
        panel = build_structural_calibration_panel([report])
        
        # Should normalize to UNKNOWN
        exp = panel["experiments"][0]
        assert exp["cross_check_status"] == "UNKNOWN"

    def test_cross_check_status_missing_becomes_unknown(self) -> None:
        """Missing cross-check should become UNKNOWN, not drop experiment."""
        report = {
            "cal_id": "CAL-EXP-1",
            "structural_summary": {"anomaly_bursts": []},
            # No governance.structure.lean_cross_check
        }
        
        panel = build_structural_calibration_panel([report])
        
        # Should still include experiment with UNKNOWN status
        assert len(panel["experiments"]) == 1
        exp = panel["experiments"][0]
        assert exp["cross_check_status"] == "UNKNOWN"
        assert exp["cal_id"] == "CAL-EXP-1"

    def test_signal_includes_most_recent_cal_id(self) -> None:
        """Signal should include most_recent_cal_id from last experiment."""
        report1 = {
            "cal_id": "CAL-EXP-1",
            "structural_summary": {"anomaly_bursts": []},
        }
        report2 = {
            "cal_id": "CAL-EXP-2",
            "structural_summary": {"anomaly_bursts": []},
        }
        report3 = {
            "cal_id": "CAL-EXP-3",
            "structural_summary": {"anomaly_bursts": []},
        }
        
        panel = build_structural_calibration_panel([report1, report2, report3])
        signal = extract_structural_calibration_panel_signal(panel)
        
        # Should get last experiment's cal_id
        assert signal["most_recent_cal_id"] == "CAL-EXP-3"


@pytest.mark.integration
class TestSmokeHarnessManifestBinding:
    """
    Integration tests for smoke harness manifest binding verification.
    
    These tests verify that:
    1. If evidence_pack.json contains the panel but manifest.json does not, status does not surface it
    2. Once mirroring runs, status surfaces it
    """

    def test_status_does_not_surface_panel_from_evidence_only(
        self, tmp_path: Path
    ) -> None:
        """
        Verify that status does NOT surface panel if it's only in evidence_pack.json
        (not in manifest.json).
        """
        # Create evidence pack directory
        pack_dir = tmp_path / "evidence_pack"
        pack_dir.mkdir()
        
        # Create evidence_pack.json with calibration panel
        evidence = {
            "schema_version": "1.0.0",
            "governance": {
                "structure": {
                    "calibration_panel": build_structural_calibration_panel([
                        {
                            "cal_id": "CAL-EXP-1",
                            "structural_summary": {
                                "mean_structural_error_rate": 0.1,
                                "max_structural_error_rate": 0.2,
                                "anomaly_bursts": [],
                            },
                            "governance": {
                                "structure": {
                                    "lean_cross_check": {"status": "CONSISTENT"},
                                },
                            },
                        },
                    ]),
                },
            },
        }
        (pack_dir / "evidence_pack.json").write_text(json.dumps(evidence, indent=2))
        
        # Create manifest.json WITHOUT calibration panel
        manifest = {
            "schema_version": "1.0.0",
            "pack_type": "first_light_evidence",
            "mode": "SHADOW",
            "files": [],
            "file_count": 0,
        }
        (pack_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        
        # Create minimal P3/P4 dirs
        p3_dir = tmp_path / "p3_synthetic"
        p4_dir = tmp_path / "p4_shadow"
        p3_dir.mkdir()
        p4_dir.mkdir()
        
        # Generate status
        from scripts.generate_first_light_status import generate_status
        status = generate_status(
            p3_dir=p3_dir,
            p4_dir=p4_dir,
            evidence_pack_dir=pack_dir,
            pipeline="smoke_test",
        )
        
        # Status should NOT contain the panel signal (manifest is canonical source)
        signals = status.get("signals")
        if signals:
            assert "structure_calibration_panel" not in signals

    def test_status_surfaces_panel_after_manifest_mirroring(
        self, tmp_path: Path
    ) -> None:
        """
        Verify that status DOES surface panel once it's mirrored to manifest.json.
        """
        # Create evidence pack directory
        pack_dir = tmp_path / "evidence_pack"
        pack_dir.mkdir()
        
        # Create evidence_pack.json with calibration panel
        panel = build_structural_calibration_panel([
            {
                "cal_id": "CAL-EXP-1",
                "structural_summary": {
                    "mean_structural_error_rate": 0.1,
                    "max_structural_error_rate": 0.2,
                    "anomaly_bursts": [],
                },
                "governance": {
                    "structure": {
                        "lean_cross_check": {"status": "CONSISTENT"},
                    },
                },
            },
        ])
        
        evidence = {
            "schema_version": "1.0.0",
            "governance": {
                "structure": {
                    "calibration_panel": panel,
                },
            },
        }
        (pack_dir / "evidence_pack.json").write_text(json.dumps(evidence, indent=2))
        
        # Create manifest.json WITH calibration panel (mirrored)
        manifest = {
            "schema_version": "1.0.0",
            "pack_type": "first_light_evidence",
            "mode": "SHADOW",
            "files": [],
            "file_count": 0,
            "governance": {
                "structure": {
                    "calibration_panel": panel,
                },
            },
        }
        (pack_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        
        # Create minimal P3/P4 dirs
        p3_dir = tmp_path / "p3_synthetic"
        p4_dir = tmp_path / "p4_shadow"
        p3_dir.mkdir()
        p4_dir.mkdir()
        
        # Generate status
        from scripts.generate_first_light_status import generate_status
        status = generate_status(
            p3_dir=p3_dir,
            p4_dir=p4_dir,
            evidence_pack_dir=pack_dir,
            pipeline="smoke_test",
        )
        
        # Status SHOULD contain the panel signal (manifest is canonical source)
        signals = status.get("signals")
        assert signals is not None
        assert "structure_calibration_panel" in signals
        
        panel_signal = signals["structure_calibration_panel"]
        assert panel_signal["num_conflict"] == 0
        assert panel_signal["num_burst"] == 0
        assert panel_signal["last_cross_check_status"] == "CONSISTENT"
        assert panel_signal["most_recent_cal_id"] == "CAL-EXP-1"

    def test_smoke_harness_output_json_shape(self, tmp_path: Path) -> None:
        """
        Verify that smoke harness produces correct JSON output shape.
        """
        import subprocess
        import json
        from pathlib import Path as PathType
        
        # Get repo root (assume we're in tests/backend/verification/)
        repo_root = PathType(__file__).parent.parent.parent.parent
        
        # Run smoke harness with --keep-temp and --output-json
        output_json_path = tmp_path / "smoke_result.json"
        result = subprocess.run(
            [
                "uv", "run", "python", "scripts/smoke_structural_cal_panel.py",
                "--keep-temp",
                "--output-json", str(output_json_path),
            ],
            capture_output=True,
            text=True,
            cwd=str(repo_root),  # Run from repo root
        )
        
        # Should exit with code 0 (non-gating)
        assert result.returncode == 0, f"Smoke harness failed: {result.stderr}"
        
        # Verify output JSON exists and has correct shape
        assert output_json_path.exists(), "Output JSON file was not created"
        
        with open(output_json_path, "r", encoding="utf-8") as f:
            result_data = json.load(f)
        
        # Verify required fields
        assert "schema_version" in result_data
        assert "manifest_mirroring_ok" in result_data
        assert "status_signal_present" in result_data
        assert "warnings_count" in result_data
        
        # Verify types
        assert isinstance(result_data["schema_version"], str)
        assert isinstance(result_data["manifest_mirroring_ok"], bool)
        assert isinstance(result_data["status_signal_present"], bool)
        assert isinstance(result_data["warnings_count"], int)
        
        # Verify expected values (smoke harness should succeed)
        assert result_data["manifest_mirroring_ok"] is True
        assert result_data["status_signal_present"] is True
        assert result_data["warnings_count"] >= 0  # May have warnings for conflicts/bursts
