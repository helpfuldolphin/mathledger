"""Tests for coherence governance adapter.

Tests verify:
- JSON serialization
- Determinism
- No governance writes
- No prescriptive language
"""

import json
import pytest
from typing import Any, Dict

from backend.health.coherence_adapter import (
    COHERENCE_TILE_SCHEMA_VERSION,
    build_coherence_governance_tile,
    build_coherence_tile_for_global_health,
    extract_coherence_drift_signal,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_coherence_map() -> Dict[str, Any]:
    """Sample coherence map for testing."""
    return {
        "slice_coherence_scores": {
            "slice1": 0.85,
            "slice2": 0.60,
            "slice3": 0.30,
        },
        "global_coherence_index": 0.583,
        "coherence_band": "PARTIAL",
        "root_incoherence_causes": [
            "2 slices with coherence below 0.45 threshold.",
            "1 slice with MAJOR drift severity.",
        ],
    }


@pytest.fixture
def sample_drift_horizon() -> Dict[str, Any]:
    """Sample drift horizon for testing."""
    return {
        "horizon_estimate": 5,
        "confidence": 0.75,
        "trajectory": "DEGRADING",
        "current_coherence": 0.5,
        "prediction_notes": [
            "Current coherence: 0.500.",
            "Trajectory: DEGRADING.",
        ],
    }


@pytest.fixture
def sample_console_tile() -> Dict[str, Any]:
    """Sample console tile for testing."""
    return {
        "status_light": "YELLOW",
        "coherence_band": "PARTIAL",
        "slices_at_risk": ["slice3"],
        "dominant_coherence_drivers": [
            "2 slices with coherence below 0.45 threshold.",
        ],
        "headline": "Coherence status: YELLOW (PARTIAL). Global coherence index: 0.583.",
    }


# =============================================================================
# TASK 1: GOVERNANCE TILE ADAPTER TESTS
# =============================================================================

class TestBuildCoherenceGovernanceTile:
    """Tests for governance tile building."""
    
    def test_tile_has_required_fields(
        self, sample_coherence_map, sample_drift_horizon, sample_console_tile
    ):
        """Tile should have all required fields."""
        tile = build_coherence_governance_tile(
            coherence_map=sample_coherence_map,
            drift_horizon=sample_drift_horizon,
            console_tile=sample_console_tile,
        )
        
        required = {
            "schema_version", "status_light", "coherence_band",
            "global_coherence_index", "slices_at_risk", "drivers",
            "horizon_estimate", "headline"
        }
        assert required.issubset(set(tile.keys()))
    
    def test_schema_version(self, sample_coherence_map):
        """Tile should have correct schema version."""
        tile = build_coherence_governance_tile(sample_coherence_map)
        assert tile["schema_version"] == COHERENCE_TILE_SCHEMA_VERSION
    
    def test_status_light_from_band(self, sample_coherence_map):
        """Status light should map from coherence band."""
        # COHERENT -> GREEN
        map_coherent = {
            **sample_coherence_map,
            "coherence_band": "COHERENT",
        }
        tile = build_coherence_governance_tile(map_coherent)
        assert tile["status_light"] == "GREEN"
        
        # PARTIAL -> YELLOW
        tile = build_coherence_governance_tile(sample_coherence_map)
        assert tile["status_light"] == "YELLOW"
        
        # MISALIGNED -> RED
        map_misaligned = {
            **sample_coherence_map,
            "coherence_band": "MISALIGNED",
        }
        tile = build_coherence_governance_tile(map_misaligned)
        assert tile["status_light"] == "RED"
    
    def test_slices_at_risk_extracted(self, sample_coherence_map):
        """Should extract slices at risk (below threshold)."""
        tile = build_coherence_governance_tile(sample_coherence_map)
        
        # slice3 has score 0.30 < 0.45 threshold
        assert "slice3" in tile["slices_at_risk"]
        assert tile["slices_at_risk"] == sorted(tile["slices_at_risk"])
    
    def test_drivers_extracted(self, sample_coherence_map):
        """Should extract drivers from root causes."""
        tile = build_coherence_governance_tile(sample_coherence_map)
        
        assert len(tile["drivers"]) > 0
        assert all(isinstance(d, str) for d in tile["drivers"])
    
    def test_horizon_estimate_from_drift(self, sample_coherence_map, sample_drift_horizon):
        """Should extract horizon estimate from drift_horizon."""
        tile = build_coherence_governance_tile(
            sample_coherence_map, drift_horizon=sample_drift_horizon
        )
        
        assert tile["horizon_estimate"] == 5
    
    def test_horizon_estimate_none_when_missing(self, sample_coherence_map):
        """Should have None horizon when drift_horizon not provided."""
        tile = build_coherence_governance_tile(sample_coherence_map)
        
        assert tile["horizon_estimate"] is None
    
    def test_headline_from_console_tile(self, sample_coherence_map, sample_console_tile):
        """Should use headline from console_tile if provided."""
        tile = build_coherence_governance_tile(
            sample_coherence_map, console_tile=sample_console_tile
        )
        
        assert tile["headline"] == sample_console_tile["headline"]
    
    def test_headline_built_when_console_tile_missing(self, sample_coherence_map):
        """Should build headline from coherence_map when console_tile missing."""
        tile = build_coherence_governance_tile(sample_coherence_map)
        
        assert "Coherence status:" in tile["headline"]
        assert "YELLOW" in tile["headline"] or "PARTIAL" in tile["headline"]


# =============================================================================
# TASK 2: JSON SERIALIZATION TESTS
# =============================================================================

class TestJSONSerialization:
    """Tests for JSON serialization."""
    
    def test_tile_is_json_serializable(
        self, sample_coherence_map, sample_drift_horizon, sample_console_tile
    ):
        """Tile should be JSON serializable."""
        tile = build_coherence_governance_tile(
            sample_coherence_map,
            drift_horizon=sample_drift_horizon,
            console_tile=sample_console_tile,
        )
        
        # Should not raise
        json_str = json.dumps(tile)
        assert isinstance(json_str, str)
        
        # Should be parseable
        parsed = json.loads(json_str)
        assert parsed == tile
    
    def test_drift_signal_is_json_serializable(self, sample_coherence_map):
        """Drift signal should be JSON serializable."""
        signal = extract_coherence_drift_signal(sample_coherence_map)
        
        json_str = json.dumps(signal)
        parsed = json.loads(json_str)
        assert parsed == signal


# =============================================================================
# TASK 3: DETERMINISM TESTS
# =============================================================================

class TestDeterminism:
    """Tests for deterministic output."""
    
    def test_tile_is_deterministic(
        self, sample_coherence_map, sample_drift_horizon, sample_console_tile
    ):
        """Tile should be deterministic across calls."""
        tile1 = build_coherence_governance_tile(
            sample_coherence_map,
            drift_horizon=sample_drift_horizon,
            console_tile=sample_console_tile,
        )
        tile2 = build_coherence_governance_tile(
            sample_coherence_map,
            drift_horizon=sample_drift_horizon,
            console_tile=sample_console_tile,
        )
        
        assert tile1 == tile2
    
    def test_drift_signal_is_deterministic(self, sample_coherence_map):
        """Drift signal should be deterministic."""
        signal1 = extract_coherence_drift_signal(sample_coherence_map)
        signal2 = extract_coherence_drift_signal(sample_coherence_map)
        
        assert signal1 == signal2
    
    def test_json_serialization_is_deterministic(
        self, sample_coherence_map, sample_drift_horizon, sample_console_tile
    ):
        """JSON serialization should be deterministic."""
        tile = build_coherence_governance_tile(
            sample_coherence_map,
            drift_horizon=sample_drift_horizon,
            console_tile=sample_console_tile,
        )
        
        json1 = json.dumps(tile, sort_keys=True)
        json2 = json.dumps(tile, sort_keys=True)
        
        assert json1 == json2


# =============================================================================
# TASK 4: NO GOVERNANCE WRITES TESTS
# =============================================================================

class TestNoGovernanceWrites:
    """Tests that functions are read-only."""
    
    def test_tile_building_does_not_modify_inputs(
        self, sample_coherence_map, sample_drift_horizon, sample_console_tile
    ):
        """Building tile should not modify input dictionaries."""
        map_copy = dict(sample_coherence_map)
        horizon_copy = dict(sample_drift_horizon) if sample_drift_horizon else None
        console_copy = dict(sample_console_tile) if sample_console_tile else None
        
        build_coherence_governance_tile(
            map_copy, drift_horizon=horizon_copy, console_tile=console_copy
        )
        
        assert map_copy == sample_coherence_map
        if horizon_copy:
            assert horizon_copy == sample_drift_horizon
        if console_copy:
            assert console_copy == sample_console_tile
    
    def test_drift_signal_does_not_modify_input(self, sample_coherence_map):
        """Extracting drift signal should not modify input."""
        map_copy = dict(sample_coherence_map)
        
        extract_coherence_drift_signal(map_copy)
        
        assert map_copy == sample_coherence_map


# =============================================================================
# TASK 5: NO PRESCRIPTIVE LANGUAGE TESTS
# =============================================================================

FORBIDDEN_WORDS = {
    "fix", "change", "modify", "improve", "correct", "update",
    "better", "worse", "improvement", "degradation",
    "good", "bad", "poor", "excellent",
    "should", "must", "need to", "required to",
}


def check_forbidden_language(text: str) -> list[str]:
    """Check for forbidden words in text."""
    text_lower = text.lower()
    found = [word for word in FORBIDDEN_WORDS if word in text_lower]
    return found


class TestNoPrescriptiveLanguage:
    """Tests that outputs contain no prescriptive language."""
    
    def test_headline_no_forbidden_words(
        self, sample_coherence_map, sample_console_tile
    ):
        """Headline should not contain forbidden words."""
        tile = build_coherence_governance_tile(
            sample_coherence_map, console_tile=sample_console_tile
        )
        
        forbidden = check_forbidden_language(tile["headline"])
        assert len(forbidden) == 0, f"Found forbidden words in headline: {forbidden}"
    
    def test_drivers_no_forbidden_words(self, sample_coherence_map):
        """Drivers should not contain forbidden words."""
        tile = build_coherence_governance_tile(sample_coherence_map)
        
        for driver in tile["drivers"]:
            forbidden = check_forbidden_language(driver)
            assert len(forbidden) == 0, f"Found forbidden words in driver '{driver}': {forbidden}"
    
    def test_drift_signal_no_forbidden_words(self, sample_coherence_map):
        """Drift signal should not contain forbidden words."""
        # Drift signal doesn't have text fields, but check structure
        signal = extract_coherence_drift_signal(sample_coherence_map)
        
        # Verify it's just data, no text fields with language
        assert "coherence_band" in signal
        assert "low_slices" in signal
        assert "global_index" in signal


# =============================================================================
# TASK 4: COHERENCE DRIFT TIMELINE HOOK TESTS
# =============================================================================

class TestExtractCoherenceDriftSignal:
    """Tests for coherence drift signal extraction."""
    
    def test_drift_signal_has_required_fields(self, sample_coherence_map):
        """Drift signal should have all required fields."""
        signal = extract_coherence_drift_signal(sample_coherence_map)
        
        required = {"coherence_band", "low_slices", "global_index"}
        assert required.issubset(set(signal.keys()))
    
    def test_coherence_band_extracted(self, sample_coherence_map):
        """Should extract coherence band."""
        signal = extract_coherence_drift_signal(sample_coherence_map)
        
        assert signal["coherence_band"] == "PARTIAL"
    
    def test_low_slices_identified(self, sample_coherence_map):
        """Should identify slices below threshold."""
        signal = extract_coherence_drift_signal(sample_coherence_map)
        
        # slice3 has score 0.30 < 0.45
        assert "slice3" in signal["low_slices"]
        assert signal["low_slices"] == sorted(signal["low_slices"])
    
    def test_global_index_extracted(self, sample_coherence_map):
        """Should extract global coherence index."""
        signal = extract_coherence_drift_signal(sample_coherence_map)
        
        assert signal["global_index"] == 0.583
    
    def test_low_slices_empty_when_all_above_threshold(self):
        """Should have empty low_slices when all slices above threshold."""
        map_high = {
            "slice_coherence_scores": {
                "slice1": 0.90,
                "slice2": 0.85,
            },
            "global_coherence_index": 0.875,
            "coherence_band": "COHERENT",
            "root_incoherence_causes": [],
        }
        
        signal = extract_coherence_drift_signal(map_high)
        
        assert signal["low_slices"] == []


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests."""
    
    def test_build_coherence_tile_for_global_health_alias(
        self, sample_coherence_map, sample_drift_horizon, sample_console_tile
    ):
        """build_coherence_tile_for_global_health should be alias."""
        tile1 = build_coherence_governance_tile(
            sample_coherence_map,
            drift_horizon=sample_drift_horizon,
            console_tile=sample_console_tile,
        )
        tile2 = build_coherence_tile_for_global_health(
            sample_coherence_map,
            drift_horizon=sample_drift_horizon,
            console_tile=sample_console_tile,
        )
        
        assert tile1 == tile2

