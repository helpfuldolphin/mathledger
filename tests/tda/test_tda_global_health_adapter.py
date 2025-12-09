"""
TDA Global Health Adapter Tests — Phase VI

Operation CORTEX: Phase VI Auto-Watchdog & Global Health Coupler
=================================================================

Tests for:
1. Adapter classification (OK/ATTENTION/ALERT) for synthetic snapshots
2. JSON-serializable output
3. Deterministic ordering
4. Status rule verification
"""

from __future__ import annotations

import json
from typing import Any, Dict

import pytest

from backend.health.tda_adapter import (
    # Schema version
    TDA_HEALTH_TILE_SCHEMA_VERSION,
    # Status constants
    TDA_STATUS_OK,
    TDA_STATUS_ATTENTION,
    TDA_STATUS_ALERT,
    # HSS trend constants
    HSS_TREND_IMPROVING,
    HSS_TREND_STABLE,
    HSS_TREND_DEGRADING,
    HSS_TREND_UNKNOWN,
    # Functions
    summarize_tda_for_global_health,
    validate_tda_health_tile,
    extend_global_health_with_tda,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def healthy_snapshot() -> Dict[str, Any]:
    """Snapshot with all metrics in healthy range."""
    return {
        "schema_version": "tda-governance-console-1.0.0",
        "mode": "hard",
        "cycle_count": 100,
        "block_rate": 0.0,
        "warn_rate": 0.0,
        "mean_hss": 0.75,
        "hss_trend": "stable",
        "golden_alignment": "ALIGNED",
        "exception_windows_active": 0,
        "recent_exceptions": [],
        "governance_signal": "HEALTHY",
    }


@pytest.fixture
def attention_snapshot() -> Dict[str, Any]:
    """Snapshot with elevated metrics (ATTENTION level)."""
    return {
        "schema_version": "tda-governance-console-1.0.0",
        "mode": "hard",
        "cycle_count": 100,
        "block_rate": 0.08,
        "warn_rate": 0.05,
        "mean_hss": 0.55,
        "hss_trend": "degrading",
        "golden_alignment": "DRIFTING",
        "exception_windows_active": 1,
        "recent_exceptions": [{"active": True}],
        "governance_signal": "DEGRADED",
    }


@pytest.fixture
def alert_snapshot() -> Dict[str, Any]:
    """Snapshot with critical metrics (ALERT level)."""
    return {
        "schema_version": "tda-governance-console-1.0.0",
        "mode": "hard",
        "cycle_count": 100,
        "block_rate": 0.25,
        "warn_rate": 0.10,
        "mean_hss": 0.35,
        "hss_trend": "degrading",
        "golden_alignment": "BROKEN",
        "exception_windows_active": 2,
        "recent_exceptions": [{"active": True}, {"active": True}],
        "governance_signal": "CRITICAL",
    }


# ============================================================================
# Test: Status Classification
# ============================================================================

class TestStatusClassification:
    """Tests for TDA status classification rules."""

    def test_healthy_snapshot_returns_ok_with_zero_block_rate(self):
        """Zero block rate and healthy metrics returns OK."""
        snapshot = {
            "block_rate": 0.0,
            "mean_hss": 0.8,
            "hss_trend": "stable",
            "governance_signal": "HEALTHY",
        }

        tile = summarize_tda_for_global_health(snapshot)

        # Zero block rate still causes ATTENTION (any non-zero consideration)
        # Actually, block_rate == 0.0 should be OK
        assert tile["tda_status"] == TDA_STATUS_OK

    def test_nonzero_block_rate_returns_attention(self):
        """Any non-zero block rate returns ATTENTION."""
        snapshot = {
            "block_rate": 0.01,
            "mean_hss": 0.8,
            "hss_trend": "stable",
            "governance_signal": "HEALTHY",
        }

        tile = summarize_tda_for_global_health(snapshot)

        assert tile["tda_status"] == TDA_STATUS_ATTENTION

    def test_governance_block_returns_alert(self):
        """Governance signal BLOCK returns ALERT."""
        snapshot = {
            "block_rate": 0.05,
            "mean_hss": 0.7,
            "hss_trend": "stable",
            "governance_signal": "CRITICAL",  # Maps to BLOCK
        }

        tile = summarize_tda_for_global_health(snapshot)

        assert tile["tda_status"] == TDA_STATUS_ALERT

    def test_high_block_rate_with_degrading_trend_returns_alert(self):
        """Block rate >= 0.2 AND degrading trend returns ALERT."""
        snapshot = {
            "block_rate": 0.22,
            "mean_hss": 0.5,
            "hss_trend": "degrading",
            "governance_signal": "HEALTHY",
        }

        tile = summarize_tda_for_global_health(snapshot)

        assert tile["tda_status"] == TDA_STATUS_ALERT

    def test_broken_golden_alignment_returns_alert(self):
        """BROKEN golden alignment returns ALERT."""
        snapshot = {
            "block_rate": 0.0,
            "mean_hss": 0.8,
            "hss_trend": "stable",
            "golden_alignment": "BROKEN",
            "governance_signal": "HEALTHY",
        }

        tile = summarize_tda_for_global_health(snapshot)

        assert tile["tda_status"] == TDA_STATUS_ALERT

    def test_drifting_golden_alignment_returns_attention(self):
        """DRIFTING golden alignment returns ATTENTION."""
        snapshot = {
            "block_rate": 0.0,
            "mean_hss": 0.8,
            "hss_trend": "stable",
            "golden_alignment": "DRIFTING",
            "governance_signal": "HEALTHY",
        }

        tile = summarize_tda_for_global_health(snapshot)

        assert tile["tda_status"] == TDA_STATUS_ATTENTION

    def test_degrading_trend_with_low_block_rate_returns_attention(self):
        """Degrading HSS trend with low block rate returns ATTENTION."""
        snapshot = {
            "block_rate": 0.0,
            "mean_hss": 0.7,
            "hss_trend": "degrading",
            "governance_signal": "HEALTHY",
        }

        tile = summarize_tda_for_global_health(snapshot)

        assert tile["tda_status"] == TDA_STATUS_ATTENTION


# ============================================================================
# Test: HSS Trend Normalization
# ============================================================================

class TestHSSTrendNormalization:
    """Tests for HSS trend normalization to uppercase."""

    def test_lowercase_trend_normalized(self):
        """Lowercase trend is normalized to uppercase."""
        snapshot = {"hss_trend": "improving", "block_rate": 0.0}
        tile = summarize_tda_for_global_health(snapshot)
        assert tile["hss_trend"] == HSS_TREND_IMPROVING

    def test_mixed_case_trend_normalized(self):
        """Mixed case trend is normalized to uppercase."""
        snapshot = {"hss_trend": "Degrading", "block_rate": 0.0}
        tile = summarize_tda_for_global_health(snapshot)
        assert tile["hss_trend"] == HSS_TREND_DEGRADING

    def test_unknown_trend_returns_unknown(self):
        """Unknown trend value returns UNKNOWN."""
        snapshot = {"hss_trend": "invalid", "block_rate": 0.0}
        tile = summarize_tda_for_global_health(snapshot)
        assert tile["hss_trend"] == HSS_TREND_UNKNOWN

    def test_none_trend_returns_unknown(self):
        """None trend returns UNKNOWN."""
        snapshot = {"hss_trend": None, "block_rate": 0.0}
        tile = summarize_tda_for_global_health(snapshot)
        assert tile["hss_trend"] == HSS_TREND_UNKNOWN


# ============================================================================
# Test: Governance Signal Normalization
# ============================================================================

class TestGovernanceSignalNormalization:
    """Tests for governance signal normalization."""

    def test_healthy_maps_to_ok(self):
        """HEALTHY governance signal maps to OK."""
        snapshot = {"governance_signal": "HEALTHY", "block_rate": 0.0}
        tile = summarize_tda_for_global_health(snapshot)
        assert tile["governance_signal"] == "OK"

    def test_degraded_maps_to_warn(self):
        """DEGRADED governance signal maps to WARN."""
        snapshot = {"governance_signal": "DEGRADED", "block_rate": 0.0}
        tile = summarize_tda_for_global_health(snapshot)
        assert tile["governance_signal"] == "WARN"

    def test_critical_maps_to_block(self):
        """CRITICAL governance signal maps to BLOCK."""
        snapshot = {"governance_signal": "CRITICAL", "block_rate": 0.0}
        tile = summarize_tda_for_global_health(snapshot)
        assert tile["governance_signal"] == "BLOCK"


# ============================================================================
# Test: Notes Generation
# ============================================================================

class TestNotesGeneration:
    """Tests for neutral, structural notes generation."""

    def test_high_block_rate_generates_note(self):
        """High block rate generates appropriate note."""
        snapshot = {"block_rate": 0.25, "cycle_count": 100}
        tile = summarize_tda_for_global_health(snapshot)
        assert any("25%" in note or "exceeds" in note for note in tile["notes"])

    def test_elevated_block_rate_generates_note(self):
        """Elevated block rate generates appropriate note."""
        snapshot = {"block_rate": 0.12, "cycle_count": 100}
        tile = summarize_tda_for_global_health(snapshot)
        assert any("elevated" in note.lower() or "12%" in note for note in tile["notes"])

    def test_degrading_trend_generates_note(self):
        """Degrading HSS trend generates note."""
        snapshot = {"hss_trend": "degrading", "block_rate": 0.0, "cycle_count": 50}
        tile = summarize_tda_for_global_health(snapshot)
        assert any("DEGRADING" in note for note in tile["notes"])

    def test_exception_window_generates_note(self):
        """Active exception window generates note."""
        snapshot = {"exception_windows_active": 2, "block_rate": 0.0}
        tile = summarize_tda_for_global_health(snapshot)
        assert any("exception" in note.lower() for note in tile["notes"])

    def test_notes_are_neutral(self):
        """Notes do not contain normative language."""
        snapshot = {"block_rate": 0.3, "mean_hss": 0.2, "governance_signal": "CRITICAL"}
        tile = summarize_tda_for_global_health(snapshot)

        notes_text = " ".join(tile["notes"]).lower()
        assert "good" not in notes_text
        assert "bad" not in notes_text
        assert "excellent" not in notes_text
        assert "poor" not in notes_text

    def test_healthy_snapshot_has_normal_note(self):
        """Healthy snapshot generates 'normal operating range' note."""
        snapshot = {
            "block_rate": 0.0,
            "mean_hss": 0.8,
            "hss_trend": "stable",
            "governance_signal": "HEALTHY",
            "cycle_count": 100,
        }
        tile = summarize_tda_for_global_health(snapshot)
        assert any("normal" in note.lower() for note in tile["notes"])


# ============================================================================
# Test: JSON Serialization
# ============================================================================

class TestJSONSerialization:
    """Tests for JSON-serializable output."""

    def test_tile_is_json_serializable(self, healthy_snapshot: Dict[str, Any]):
        """Tile output is JSON serializable."""
        tile = summarize_tda_for_global_health(healthy_snapshot)

        # Should not raise
        json_str = json.dumps(tile)
        assert isinstance(json_str, str)

    def test_tile_roundtrips_through_json(self, attention_snapshot: Dict[str, Any]):
        """Tile roundtrips through JSON correctly."""
        tile = summarize_tda_for_global_health(attention_snapshot)

        json_str = json.dumps(tile)
        parsed = json.loads(json_str)

        assert parsed == tile

    def test_tile_has_schema_version(self, healthy_snapshot: Dict[str, Any]):
        """Tile includes schema version."""
        tile = summarize_tda_for_global_health(healthy_snapshot)
        assert tile["schema_version"] == TDA_HEALTH_TILE_SCHEMA_VERSION


# ============================================================================
# Test: Deterministic Behavior
# ============================================================================

class TestDeterministicBehavior:
    """Tests for deterministic output."""

    def test_same_input_produces_same_output(self, healthy_snapshot: Dict[str, Any]):
        """Same input always produces same output."""
        tile1 = summarize_tda_for_global_health(healthy_snapshot)
        tile2 = summarize_tda_for_global_health(healthy_snapshot)

        assert tile1 == tile2

    def test_notes_ordering_is_deterministic(self):
        """Notes are generated in deterministic order."""
        snapshot = {
            "block_rate": 0.15,
            "hss_trend": "degrading",
            "governance_signal": "DEGRADED",
            "exception_windows_active": 1,
            "cycle_count": 100,
        }

        tiles = [summarize_tda_for_global_health(snapshot) for _ in range(5)]
        notes_lists = [tuple(t["notes"]) for t in tiles]

        # All should be identical
        assert len(set(notes_lists)) == 1


# ============================================================================
# Test: Tile Validation
# ============================================================================

class TestTileValidation:
    """Tests for tile validation function."""

    def test_valid_tile_passes_validation(self, healthy_snapshot: Dict[str, Any]):
        """Valid tile passes validation."""
        tile = summarize_tda_for_global_health(healthy_snapshot)
        validated = validate_tda_health_tile(tile)
        assert validated == tile

    def test_missing_field_raises_error(self):
        """Missing required field raises error."""
        tile = {
            "schema_version": TDA_HEALTH_TILE_SCHEMA_VERSION,
            # Missing tda_status
        }

        with pytest.raises(ValueError, match="Missing required field"):
            validate_tda_health_tile(tile)

    def test_invalid_schema_version_raises_error(self):
        """Invalid schema version raises error."""
        tile = {
            "schema_version": "invalid",
            "tda_status": "OK",
            "block_rate": 0.0,
            "hss_trend": "STABLE",
            "governance_signal": "OK",
            "notes": [],
        }

        with pytest.raises(ValueError, match="Invalid schema_version"):
            validate_tda_health_tile(tile)

    def test_invalid_status_raises_error(self):
        """Invalid TDA status raises error."""
        tile = {
            "schema_version": TDA_HEALTH_TILE_SCHEMA_VERSION,
            "tda_status": "INVALID",
            "block_rate": 0.0,
            "hss_trend": "STABLE",
            "governance_signal": "OK",
            "notes": [],
        }

        with pytest.raises(ValueError, match="Invalid tda_status"):
            validate_tda_health_tile(tile)


# ============================================================================
# Test: Global Health Extension
# ============================================================================

class TestGlobalHealthExtension:
    """Tests for extend_global_health_with_tda function."""

    def test_extends_global_health(self, healthy_snapshot: Dict[str, Any]):
        """Extends global health with TDA section."""
        global_health = {
            "schema_version": "1.0.0",
            "fm_ok": True,
            "status": "OK",
        }

        extended = extend_global_health_with_tda(global_health, healthy_snapshot)

        assert "tda" in extended
        assert extended["tda"]["schema_version"] == TDA_HEALTH_TILE_SCHEMA_VERSION

    def test_does_not_mutate_original(self, healthy_snapshot: Dict[str, Any]):
        """Does not mutate original global health dict."""
        global_health = {
            "schema_version": "1.0.0",
            "fm_ok": True,
            "status": "OK",
        }

        original_keys = set(global_health.keys())
        _ = extend_global_health_with_tda(global_health, healthy_snapshot)

        assert set(global_health.keys()) == original_keys
        assert "tda" not in global_health


# ============================================================================
# Test: Edge Cases
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_snapshot(self):
        """Handles empty snapshot gracefully."""
        tile = summarize_tda_for_global_health({})

        assert tile["tda_status"] == TDA_STATUS_OK
        assert tile["block_rate"] == 0.0
        assert tile["hss_trend"] == HSS_TREND_UNKNOWN

    def test_none_mean_hss(self):
        """Handles None mean_hss."""
        snapshot = {"block_rate": 0.0, "mean_hss": None}
        tile = summarize_tda_for_global_health(snapshot)

        assert tile["mean_hss"] is None

    def test_block_rate_exactly_at_threshold(self):
        """Block rate exactly at threshold triggers ALERT."""
        snapshot = {
            "block_rate": 0.2,
            "hss_trend": "degrading",
        }
        tile = summarize_tda_for_global_health(snapshot)

        assert tile["tda_status"] == TDA_STATUS_ALERT

    def test_float_precision(self):
        """Float values are rounded appropriately."""
        snapshot = {
            "block_rate": 0.123456789,
            "mean_hss": 0.987654321,
        }
        tile = summarize_tda_for_global_health(snapshot)

        assert tile["block_rate"] == 0.1235
        assert tile["mean_hss"] == 0.9877
