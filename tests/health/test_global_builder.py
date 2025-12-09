"""
Global Health Surface Builder Tests — Phase VII NEURAL LINK

Operation CORTEX: Phase VII Global Health Assembly Tests
========================================================

Tests for:
1. Status aggregation (BLOCK > WARN > OK)
2. TDA tile integration
3. Replay tile extraction
4. Learning tile extraction
5. FM health validation
"""

from __future__ import annotations

import json
from typing import Any, Dict

import pytest

from backend.health.global_builder import (
    GlobalHealthSurface,
    build_global_health_surface,
    GLOBAL_BUILDER_SCHEMA_VERSION,
)
from backend.health.global_schema import SchemaValidationError


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def healthy_fm_payload() -> Dict[str, Any]:
    """FM health payload with all OK status."""
    return {
        "schema_version": "1.0.0",
        "fm_ok": True,
        "coverage_pct": 95.5,
        "status": "OK",
        "alignment_status": "WELL_DISTRIBUTED",
        "external_only_labels": 0,
    }


@pytest.fixture
def warn_fm_payload() -> Dict[str, Any]:
    """FM health payload with WARN status."""
    return {
        "schema_version": "1.0.0",
        "fm_ok": False,
        "coverage_pct": 75.0,
        "status": "WARN",
        "alignment_status": "CONCENTRATED",
        "external_only_labels": 0,
    }


@pytest.fixture
def block_fm_payload() -> Dict[str, Any]:
    """FM health payload with BLOCK status."""
    return {
        "schema_version": "1.0.0",
        "fm_ok": False,
        "coverage_pct": 50.0,
        "status": "BLOCK",
        "alignment_status": "SPARSE",
        "external_only_labels": 5,
    }


@pytest.fixture
def tda_ok_snapshot() -> Dict[str, Any]:
    """TDA snapshot with OK status."""
    return {
        "tda_status": "OK",
        "block_rate": 0.0,
        "mean_hss": 0.85,
        "hss_trend": "STABLE",
        "governance_signal": "HEALTHY",
    }


@pytest.fixture
def tda_attention_snapshot() -> Dict[str, Any]:
    """TDA snapshot with ATTENTION status."""
    return {
        "tda_status": "ATTENTION",
        "block_rate": 0.08,
        "mean_hss": 0.55,
        "hss_trend": "DEGRADING",
        "governance_signal": "DEGRADED",
    }


@pytest.fixture
def tda_alert_snapshot() -> Dict[str, Any]:
    """TDA snapshot with ALERT status."""
    return {
        "tda_status": "ALERT",
        "block_rate": 0.25,
        "mean_hss": 0.35,
        "hss_trend": "DEGRADING",
        "governance_signal": "CRITICAL",
    }


# ============================================================================
# Test: Basic Surface Building
# ============================================================================

class TestBasicSurfaceBuilding:
    """Tests for basic surface building."""

    def test_build_with_only_fm_health(self, healthy_fm_payload: Dict[str, Any]):
        """Build surface with only FM health."""
        surface = build_global_health_surface(healthy_fm_payload)

        assert surface.status == "OK"
        assert surface.fm_ok is True
        assert surface.fm_coverage_pct == 95.5
        assert surface.tda is None
        assert surface.replay is None
        assert surface.learning is None

    def test_to_dict_is_json_serializable(self, healthy_fm_payload: Dict[str, Any]):
        """Surface to_dict is JSON serializable."""
        surface = build_global_health_surface(healthy_fm_payload)
        d = surface.to_dict()

        # Should not raise
        json_str = json.dumps(d)
        assert isinstance(json_str, str)

    def test_schema_version_included(self, healthy_fm_payload: Dict[str, Any]):
        """Surface includes schema version."""
        surface = build_global_health_surface(healthy_fm_payload)
        d = surface.to_dict()

        assert "schema_version" in d
        assert d["schema_version"] == "1.0.0"

    def test_generated_at_included(self, healthy_fm_payload: Dict[str, Any]):
        """Surface includes generated_at timestamp."""
        surface = build_global_health_surface(healthy_fm_payload)
        d = surface.to_dict()

        assert "generated_at" in d
        assert isinstance(d["generated_at"], str)


# ============================================================================
# Test: Status Aggregation
# ============================================================================

class TestStatusAggregation:
    """Tests for status aggregation logic."""

    def test_all_ok_returns_ok(
        self,
        healthy_fm_payload: Dict[str, Any],
        tda_ok_snapshot: Dict[str, Any],
    ):
        """All OK components return OK aggregate status."""
        surface = build_global_health_surface(
            fm_health=healthy_fm_payload,
            tda_snapshot=tda_ok_snapshot,
        )

        assert surface.status == "OK"

    def test_tda_alert_returns_block(
        self,
        healthy_fm_payload: Dict[str, Any],
        tda_alert_snapshot: Dict[str, Any],
    ):
        """TDA ALERT causes aggregate status BLOCK."""
        surface = build_global_health_surface(
            fm_health=healthy_fm_payload,
            tda_snapshot=tda_alert_snapshot,
        )

        assert surface.status == "BLOCK"

    def test_tda_attention_fm_ok_returns_warn(
        self,
        healthy_fm_payload: Dict[str, Any],
        tda_attention_snapshot: Dict[str, Any],
    ):
        """TDA ATTENTION with FM OK returns WARN."""
        surface = build_global_health_surface(
            fm_health=healthy_fm_payload,
            tda_snapshot=tda_attention_snapshot,
        )

        assert surface.status == "WARN"

    def test_replay_failure_returns_block(
        self,
        healthy_fm_payload: Dict[str, Any],
        tda_ok_snapshot: Dict[str, Any],
    ):
        """Replay failure causes aggregate status BLOCK."""
        replay_result = {
            "governance_admissible": False,
            "status": "REPLAY_FAILED",
            "confidence_score": 0.0,
        }

        surface = build_global_health_surface(
            fm_health=healthy_fm_payload,
            tda_snapshot=tda_ok_snapshot,
            replay_result=replay_result,
        )

        assert surface.status == "BLOCK"

    def test_fm_block_overrides_tda_ok(
        self,
        block_fm_payload: Dict[str, Any],
        tda_ok_snapshot: Dict[str, Any],
    ):
        """FM BLOCK status overrides TDA OK."""
        surface = build_global_health_surface(
            fm_health=block_fm_payload,
            tda_snapshot=tda_ok_snapshot,
        )

        assert surface.status == "BLOCK"


# ============================================================================
# Test: TDA Tile Integration
# ============================================================================

class TestTDATileIntegration:
    """Tests for TDA tile integration."""

    def test_tda_tile_included_when_provided(
        self,
        healthy_fm_payload: Dict[str, Any],
        tda_ok_snapshot: Dict[str, Any],
    ):
        """TDA tile is included when snapshot provided."""
        surface = build_global_health_surface(
            fm_health=healthy_fm_payload,
            tda_snapshot=tda_ok_snapshot,
        )

        assert surface.tda is not None
        assert "tda_status" in surface.tda

    def test_tda_tile_has_required_fields(
        self,
        healthy_fm_payload: Dict[str, Any],
        tda_attention_snapshot: Dict[str, Any],
    ):
        """TDA tile has required fields."""
        surface = build_global_health_surface(
            fm_health=healthy_fm_payload,
            tda_snapshot=tda_attention_snapshot,
        )

        assert surface.tda is not None
        assert "tda_status" in surface.tda
        assert "block_rate" in surface.tda
        assert "governance_signal" in surface.tda


# ============================================================================
# Test: Replay Tile Extraction
# ============================================================================

class TestReplayTileExtraction:
    """Tests for replay tile extraction."""

    def test_replay_tile_included_when_provided(
        self,
        healthy_fm_payload: Dict[str, Any],
    ):
        """Replay tile is included when result provided."""
        replay_result = {
            "governance_admissible": True,
            "status": "REPLAY_VERIFIED",
            "confidence_score": 0.95,
        }

        surface = build_global_health_surface(
            fm_health=healthy_fm_payload,
            replay_result=replay_result,
        )

        assert surface.replay is not None
        assert surface.replay["replay_safety_ok"] is True
        assert surface.replay["status"] == "REPLAY_VERIFIED"
        assert surface.replay["confidence_score"] == 0.95


# ============================================================================
# Test: Learning Tile Extraction
# ============================================================================

class TestLearningTileExtraction:
    """Tests for learning tile extraction."""

    def test_learning_tile_included_when_provided(
        self,
        healthy_fm_payload: Dict[str, Any],
    ):
        """Learning tile is included when report provided."""
        learning_report = {
            "status": "HEALTHY",
            "metrics": {
                "supports": 5,
                "contradicts": 0,
                "inconclusive": 2,
            },
        }

        surface = build_global_health_surface(
            fm_health=healthy_fm_payload,
            learning_report=learning_report,
        )

        assert surface.learning is not None
        assert surface.learning["status"] == "HEALTHY"
        assert surface.learning["supports"] == 5
        assert surface.learning["contradicts"] == 0
        assert surface.learning["inconclusive"] == 2

    def test_learning_critical_returns_block(
        self,
        healthy_fm_payload: Dict[str, Any],
    ):
        """Learning CRITICAL status causes BLOCK."""
        learning_report = {
            "status": "CRITICAL",
            "metrics": {"supports": 0, "contradicts": 5, "inconclusive": 0},
        }

        surface = build_global_health_surface(
            fm_health=healthy_fm_payload,
            learning_report=learning_report,
        )

        assert surface.status == "BLOCK"


# ============================================================================
# Test: FM Health Validation
# ============================================================================

class TestFMHealthValidation:
    """Tests for FM health validation."""

    def test_invalid_fm_payload_raises_error(self):
        """Invalid FM payload raises SchemaValidationError."""
        invalid_payload = {
            "fm_ok": "not_a_bool",  # Should be bool
            "coverage_pct": 95.0,
            "status": "OK",
            "alignment_status": "WELL_DISTRIBUTED",
            "external_only_labels": 0,
        }

        with pytest.raises(SchemaValidationError):
            build_global_health_surface(invalid_payload)

    def test_missing_required_field_raises_error(self):
        """Missing required field raises error."""
        incomplete_payload = {
            "fm_ok": True,
            # Missing coverage_pct
            "status": "OK",
            "alignment_status": "WELL_DISTRIBUTED",
            "external_only_labels": 0,
        }

        with pytest.raises((SchemaValidationError, KeyError, TypeError)):
            build_global_health_surface(incomplete_payload)
