"""
Phase X: CI Smoke Test for Metrics Governance Tile Serialization

This test verifies that the metrics governance tile can be produced and serialized
without error. It does NOT test governance logic, metrics computation, or
promotion decisions.

SHADOW MODE CONTRACT:
- This test only verifies serialization and structural stability
- No governance decisions are tested or modified
- No metrics computation or real data processing is performed
- The test is purely for observability validation

Test requirements (per Phase X spec):
1. Create mock drift_compass, budget_view, governance_signal
2. Call build_metrics_console_tile()
3. Assert: isinstance(tile, dict)
4. Assert: json.dumps(tile) does not raise
5. Assert: tile has required fields
6. Assert: metrics_for_alignment_view() works
7. Assert: determinism and neutrality

CLAUDE D - Metrics Conformance Layer
"""

from __future__ import annotations

import json
from typing import Any, Dict

import pytest


# ==============================================================================
# Test Fixtures
# ==============================================================================


@pytest.fixture
def sample_drift_compass() -> Dict[str, Any]:
    """Create sample drift compass data."""
    return {
        "schema_version": "1.0.0",
        "signal_type": "metric_drift_compass",
        "timestamp": "2025-12-10T00:00:00Z",
        "mode": "SHADOW",
        "phase": "P4",
        "compass_heading": "STABLE",
        "drift_magnitude": 0.12,
        "drift_velocity": 0.002,
        "axes": [
            {
                "axis_name": "success_rate",
                "drift_value": 0.05,
                "threshold": 0.1,
                "critical_threshold": 0.2,
                "status": "OK",
                "direction": "flat",
                "contribution": 0.42,
            },
            {
                "axis_name": "rsi",
                "drift_value": 0.07,
                "threshold": 0.15,
                "critical_threshold": 0.3,
                "status": "OK",
                "direction": "up",
                "contribution": 0.58,
            },
        ],
        "poly_cause_detected": False,
        "poly_cause_axes": [],
    }


@pytest.fixture
def sample_budget_view() -> Dict[str, Any]:
    """Create sample budget view data."""
    return {
        "schema_version": "1.0.0",
        "signal_type": "metric_budget_joint_view",
        "timestamp": "2025-12-10T00:00:00Z",
        "mode": "SHADOW",
        "phase": "P4",
        "budget_status": "NOMINAL",
        "utilization_pct": 65.5,
        "allocations": [
            {
                "resource_name": "derivation_steps",
                "layer": "budget",
                "allocated": 1000,
                "consumed": 650,
                "remaining": 350,
                "utilization_pct": 65.0,
                "status": "NOMINAL",
                "trend": "stable",
            },
        ],
        "governance_implication": {
            "safe_for_promotion": True,
            "throttle_recommended": False,
            "boost_allowed": True,
            "blocking_resources": [],
        },
    }


@pytest.fixture
def sample_governance_signal() -> Dict[str, Any]:
    """Create sample governance signal data."""
    return {
        "schema_version": "1.0.0",
        "signal_type": "metric_governance_signal",
        "timestamp": "2025-12-10T00:00:00Z",
        "mode": "SHADOW",
        "phase": "P4",
        "governance_status": "OK",
        "governance_alignment": "ALIGNED",
        "sub_signals": {
            "drift_compass": {
                "status": "OK",
                "compass_heading": "STABLE",
                "drift_magnitude": 0.12,
                "poly_cause_detected": False,
            },
            "budget_view": {
                "status": "OK",
                "budget_status": "NOMINAL",
                "utilization_pct": 65.5,
                "blocking_resources": [],
            },
            "fo_vital_signs": {
                "status": "OK",
                "health_status": "ALIVE",
                "success_rate": 95.0,
                "should_throttle": False,
                "should_boost": False,
                "confidence": 1.0,
            },
        },
        "conflict_matrix": {
            "drift_vs_budget": "ALIGNED",
            "drift_vs_fo": "ALIGNED",
            "budget_vs_fo": "ALIGNED",
            "any_conflict": False,
        },
        "reasons": [
            "[DriftCompass] Compass heading STABLE, magnitude 0.12",
            "[BudgetView] Utilization nominal at 65.5%",
            "[FOVitalSigns] Health ALIVE, success rate 95.0%",
        ],
        "safe_for_policy_update": True,
        "safe_for_promotion": True,
        "recommended_action": "CONTINUE",
        "intensity_multiplier": 1.0,
    }


# ==============================================================================
# Test Class: Tile Serialization
# ==============================================================================


class TestMetricsGovernanceTileSerializes:
    """
    CI smoke tests for metrics governance tile serialization.

    SHADOW MODE: These tests verify serialization only.
    No governance logic is tested.
    """

    def test_metrics_governance_tile_serializes_without_error(
        self,
        sample_drift_compass: Dict[str, Any],
        sample_budget_view: Dict[str, Any],
        sample_governance_signal: Dict[str, Any],
    ) -> None:
        """
        Verify metrics governance tile can be produced and serialized.

        This is the primary CI gate test per Phase X spec.
        """
        from backend.health.metrics_governance_adapter import (
            build_metrics_console_tile,
            METRICS_GOVERNANCE_TILE_SCHEMA_VERSION,
        )

        # Call build_metrics_console_tile()
        tile = build_metrics_console_tile(
            drift_compass=sample_drift_compass,
            budget_view=sample_budget_view,
            governance_signal=sample_governance_signal,
        )

        # Assert: isinstance(tile, dict)
        assert tile is not None, "Tile should not be None"
        assert isinstance(tile, dict), f"Tile should be dict, got {type(tile)}"

        # Assert: json.dumps(tile) does not raise
        json_str = json.dumps(tile)
        assert json_str is not None
        assert len(json_str) > 0

        # Verify round-trip
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)

    def test_metrics_governance_tile_has_required_fields(
        self,
        sample_drift_compass: Dict[str, Any],
        sample_budget_view: Dict[str, Any],
        sample_governance_signal: Dict[str, Any],
    ) -> None:
        """Verify tile contains required fields per schema."""
        from backend.health.metrics_governance_adapter import (
            build_metrics_console_tile,
            METRICS_GOVERNANCE_TILE_SCHEMA_VERSION,
        )

        tile = build_metrics_console_tile(
            drift_compass=sample_drift_compass,
            budget_view=sample_budget_view,
            governance_signal=sample_governance_signal,
        )

        # Required fields per Phase X spec
        required_fields = [
            "schema_version",
            "status_light",
            "headline",
            "drift_heading",
            "blocked_metrics",
            "budget_risk",
            "fo_vitality_summary",
            "governance_status",
            "safe_for_promotion",
        ]

        for field in required_fields:
            assert field in tile, f"Missing required field: {field}"

        # Verify schema version
        assert tile["schema_version"] == METRICS_GOVERNANCE_TILE_SCHEMA_VERSION

        # Verify status_light is valid
        assert tile["status_light"] in ("GREEN", "YELLOW", "RED")

        # Verify blocked_metrics is a list
        assert isinstance(tile["blocked_metrics"], list)

        # Verify fo_vitality_summary is a dict
        assert isinstance(tile["fo_vitality_summary"], dict)

    def test_metrics_governance_tile_builds_with_partial_inputs(self) -> None:
        """Verify tile builds even with partial inputs."""
        from backend.health.metrics_governance_adapter import (
            build_metrics_console_tile,
        )

        # Build with only drift_compass
        tile_drift_only = build_metrics_console_tile(
            drift_compass={"compass_heading": "STABLE", "drift_magnitude": 0.1}
        )
        assert isinstance(tile_drift_only, dict)
        assert "schema_version" in tile_drift_only
        assert "status_light" in tile_drift_only

        # Build with only budget_view
        tile_budget_only = build_metrics_console_tile(
            budget_view={"budget_status": "NOMINAL", "utilization_pct": 50.0}
        )
        assert isinstance(tile_budget_only, dict)
        assert "schema_version" in tile_budget_only

        # Build with no inputs (edge case)
        tile_empty = build_metrics_console_tile()
        assert isinstance(tile_empty, dict)
        assert tile_empty["status_light"] == "GREEN"

    def test_metrics_governance_tile_serializes_to_json(
        self,
        sample_drift_compass: Dict[str, Any],
        sample_budget_view: Dict[str, Any],
        sample_governance_signal: Dict[str, Any],
    ) -> None:
        """Test that json.dumps(tile) succeeds with all data types."""
        from backend.health.metrics_governance_adapter import (
            build_metrics_console_tile,
        )

        tile = build_metrics_console_tile(
            drift_compass=sample_drift_compass,
            budget_view=sample_budget_view,
            governance_signal=sample_governance_signal,
        )

        json_str = json.dumps(tile)
        assert isinstance(json_str, str)

        parsed = json.loads(json_str)
        assert parsed["schema_version"] == tile["schema_version"]
        assert parsed["status_light"] == tile["status_light"]


# ==============================================================================
# Test Class: Determinism
# ==============================================================================


class TestMetricsGovernanceTileDeterminism:
    """
    Tests for deterministic output.

    SHADOW MODE: These tests verify output consistency only.
    """

    def test_tile_output_is_deterministic(
        self,
        sample_drift_compass: Dict[str, Any],
        sample_budget_view: Dict[str, Any],
        sample_governance_signal: Dict[str, Any],
    ) -> None:
        """Verify tile output is deterministic."""
        from backend.health.metrics_governance_adapter import (
            build_metrics_console_tile,
        )

        tile1 = build_metrics_console_tile(
            drift_compass=sample_drift_compass,
            budget_view=sample_budget_view,
            governance_signal=sample_governance_signal,
        )

        tile2 = build_metrics_console_tile(
            drift_compass=sample_drift_compass,
            budget_view=sample_budget_view,
            governance_signal=sample_governance_signal,
        )

        assert tile1 == tile2, "Tile output should be deterministic"

        # Verify JSON serialization is also deterministic
        json1 = json.dumps(tile1, sort_keys=True)
        json2 = json.dumps(tile2, sort_keys=True)
        assert json1 == json2

    def test_alignment_view_is_deterministic(
        self,
        sample_governance_signal: Dict[str, Any],
    ) -> None:
        """Verify alignment view output is deterministic."""
        from backend.health.metrics_governance_adapter import (
            metrics_for_alignment_view,
        )

        view1 = metrics_for_alignment_view(sample_governance_signal)
        view2 = metrics_for_alignment_view(sample_governance_signal)

        assert view1 == view2, "Alignment view should be deterministic"


# ==============================================================================
# Test Class: Drift-to-Status Mapping
# ==============================================================================


class TestDriftToStatusMapping:
    """
    Tests for drift-to-status mapping logic.

    SHADOW MODE: These tests verify mapping only.
    """

    def test_stable_drift_maps_to_green(self) -> None:
        """Verify STABLE drift maps to GREEN status."""
        from backend.health.metrics_governance_adapter import (
            build_metrics_console_tile,
        )

        drift_compass = {
            "compass_heading": "STABLE",
            "drift_magnitude": 0.1,
            "axes": [],
        }

        tile = build_metrics_console_tile(drift_compass=drift_compass)
        assert tile["status_light"] == "GREEN"

    def test_drifting_maps_to_yellow(self) -> None:
        """Verify DRIFTING drift maps to YELLOW status."""
        from backend.health.metrics_governance_adapter import (
            build_metrics_console_tile,
        )

        drift_compass = {
            "compass_heading": "DRIFTING",
            "drift_magnitude": 0.4,
            "axes": [],
        }

        tile = build_metrics_console_tile(drift_compass=drift_compass)
        assert tile["status_light"] == "YELLOW"

    def test_diverging_maps_to_yellow(self) -> None:
        """Verify DIVERGING drift maps to YELLOW status."""
        from backend.health.metrics_governance_adapter import (
            build_metrics_console_tile,
        )

        drift_compass = {
            "compass_heading": "DIVERGING",
            "drift_magnitude": 0.5,
            "axes": [],
        }

        tile = build_metrics_console_tile(drift_compass=drift_compass)
        assert tile["status_light"] == "YELLOW"

    def test_critical_drift_maps_to_red(self) -> None:
        """Verify CRITICAL drift maps to RED status."""
        from backend.health.metrics_governance_adapter import (
            build_metrics_console_tile,
        )

        drift_compass = {
            "compass_heading": "CRITICAL",
            "drift_magnitude": 0.85,
            "axes": [],
        }

        tile = build_metrics_console_tile(drift_compass=drift_compass)
        assert tile["status_light"] == "RED"

    def test_high_drift_magnitude_maps_to_red(self) -> None:
        """Verify high drift magnitude (>=0.7) maps to RED status."""
        from backend.health.metrics_governance_adapter import (
            build_metrics_console_tile,
        )

        drift_compass = {
            "compass_heading": "STABLE",  # Even if heading says STABLE
            "drift_magnitude": 0.75,
            "axes": [],
        }

        tile = build_metrics_console_tile(drift_compass=drift_compass)
        assert tile["status_light"] == "RED"

    def test_budget_exceeded_maps_to_red(self) -> None:
        """Verify EXCEEDED budget status maps to RED."""
        from backend.health.metrics_governance_adapter import (
            build_metrics_console_tile,
        )

        budget_view = {
            "budget_status": "EXCEEDED",
            "utilization_pct": 105.0,
        }

        tile = build_metrics_console_tile(budget_view=budget_view)
        assert tile["status_light"] == "RED"

    def test_budget_critical_maps_to_red(self) -> None:
        """Verify CRITICAL budget status maps to RED."""
        from backend.health.metrics_governance_adapter import (
            build_metrics_console_tile,
        )

        budget_view = {
            "budget_status": "CRITICAL",
            "utilization_pct": 95.0,
        }

        tile = build_metrics_console_tile(budget_view=budget_view)
        assert tile["status_light"] == "RED"

    def test_budget_elevated_maps_to_yellow(self) -> None:
        """Verify ELEVATED budget status maps to YELLOW."""
        from backend.health.metrics_governance_adapter import (
            build_metrics_console_tile,
        )

        budget_view = {
            "budget_status": "ELEVATED",
            "utilization_pct": 82.0,
        }

        tile = build_metrics_console_tile(budget_view=budget_view)
        assert tile["status_light"] == "YELLOW"

    def test_governance_block_maps_to_red(self) -> None:
        """Verify BLOCK governance status maps to RED."""
        from backend.health.metrics_governance_adapter import (
            build_metrics_console_tile,
        )

        governance_signal = {
            "governance_status": "BLOCK",
            "sub_signals": {},
        }

        tile = build_metrics_console_tile(governance_signal=governance_signal)
        assert tile["status_light"] == "RED"

    def test_governance_warn_maps_to_yellow(self) -> None:
        """Verify WARN governance status maps to YELLOW."""
        from backend.health.metrics_governance_adapter import (
            build_metrics_console_tile,
        )

        governance_signal = {
            "governance_status": "WARN",
            "sub_signals": {},
        }

        tile = build_metrics_console_tile(governance_signal=governance_signal)
        assert tile["status_light"] == "YELLOW"


# ==============================================================================
# Test Class: Alignment View Adapter
# ==============================================================================


class TestMetricsForAlignmentView:
    """
    Tests for metrics_for_alignment_view fusion adapter.

    SHADOW MODE: These tests verify mapping and summarization only.
    """

    def test_alignment_view_has_required_fields(
        self,
        sample_governance_signal: Dict[str, Any],
    ) -> None:
        """Verify alignment view contains required fields."""
        from backend.health.metrics_governance_adapter import (
            metrics_for_alignment_view,
        )

        view = metrics_for_alignment_view(sample_governance_signal)

        required_fields = [
            "layer",
            "status",
            "alignment",
            "severity",
            "summary",
            "sub_signal_statuses",
            "safe_for_promotion",
        ]

        for field in required_fields:
            assert field in view, f"Missing required field: {field}"

        # Verify layer is always "metrics"
        assert view["layer"] == "metrics"

        # Verify status is lowercase
        assert view["status"] in ("ok", "warn", "block")

    def test_alignment_view_maps_ok_status(self) -> None:
        """Verify OK status maps correctly."""
        from backend.health.metrics_governance_adapter import (
            metrics_for_alignment_view,
        )

        signal = {
            "governance_status": "OK",
            "governance_alignment": "ALIGNED",
            "sub_signals": {},
            "reasons": [],
            "safe_for_promotion": True,
        }

        view = metrics_for_alignment_view(signal)
        assert view["status"] == "ok"
        assert view["severity"] == "info"

    def test_alignment_view_maps_warn_status(self) -> None:
        """Verify WARN status maps correctly."""
        from backend.health.metrics_governance_adapter import (
            metrics_for_alignment_view,
        )

        signal = {
            "governance_status": "WARN",
            "governance_alignment": "TENSION",
            "sub_signals": {},
            "reasons": ["[DriftCompass] Approaching threshold"],
            "safe_for_promotion": True,
        }

        view = metrics_for_alignment_view(signal)
        assert view["status"] == "warn"
        assert view["severity"] == "warning"

    def test_alignment_view_maps_block_status(self) -> None:
        """Verify BLOCK status maps correctly."""
        from backend.health.metrics_governance_adapter import (
            metrics_for_alignment_view,
        )

        signal = {
            "governance_status": "BLOCK",
            "governance_alignment": "DIVERGENT",
            "sub_signals": {},
            "reasons": ["[BudgetView] Budget exceeded"],
            "safe_for_promotion": False,
        }

        view = metrics_for_alignment_view(signal)
        assert view["status"] == "block"
        assert view["severity"] == "critical"
        assert view["safe_for_promotion"] is False

    def test_alignment_view_extracts_sub_signal_statuses(
        self,
        sample_governance_signal: Dict[str, Any],
    ) -> None:
        """Verify sub-signal statuses are extracted."""
        from backend.health.metrics_governance_adapter import (
            metrics_for_alignment_view,
        )

        view = metrics_for_alignment_view(sample_governance_signal)

        assert "sub_signal_statuses" in view
        statuses = view["sub_signal_statuses"]
        assert isinstance(statuses, dict)

        # Should have entries for each sub-signal
        assert "drift_compass" in statuses
        assert "budget_view" in statuses
        assert "fo_vital_signs" in statuses


# ==============================================================================
# Test Class: Global Health Integration
# ==============================================================================


class TestGlobalHealthSurfaceMetricsGovernanceIntegration:
    """
    Tests for metrics governance tile integration with GlobalHealthSurface.

    SHADOW MODE: These tests verify the tile attachment mechanism only.
    """

    def test_build_global_health_surface_with_metrics_governance(
        self,
        sample_drift_compass: Dict[str, Any],
        sample_budget_view: Dict[str, Any],
        sample_governance_signal: Dict[str, Any],
    ) -> None:
        """Verify metrics governance tile attached when inputs provided."""
        from backend.health.global_surface import build_global_health_surface

        payload = build_global_health_surface(
            metrics_drift_compass=sample_drift_compass,
            metrics_budget_view=sample_budget_view,
            metrics_governance_signal=sample_governance_signal,
        )

        assert isinstance(payload, dict)
        assert "schema_version" in payload
        assert "dynamics" in payload
        assert "metrics_governance" in payload, (
            "Metrics governance tile should be present when inputs provided"
        )

        metrics_tile = payload["metrics_governance"]
        assert isinstance(metrics_tile, dict)
        assert "status_light" in metrics_tile
        assert "headline" in metrics_tile

        # Verify serializable
        json_str = json.dumps(payload)
        assert len(json_str) > 0

    def test_build_global_health_surface_without_metrics_governance(self) -> None:
        """Verify build works without metrics governance inputs."""
        from backend.health.global_surface import build_global_health_surface

        payload = build_global_health_surface()

        assert isinstance(payload, dict)
        assert "schema_version" in payload
        assert "dynamics" in payload
        assert "metrics_governance" not in payload, (
            "Metrics governance tile should not be present when inputs missing"
        )

    def test_metrics_governance_tile_does_not_affect_dynamics(
        self,
        sample_drift_compass: Dict[str, Any],
        sample_budget_view: Dict[str, Any],
        sample_governance_signal: Dict[str, Any],
    ) -> None:
        """Verify metrics governance tile presence doesn't change dynamics tile."""
        from backend.health.global_surface import build_global_health_surface

        # Build without metrics governance
        payload_without = build_global_health_surface()

        # Build with metrics governance
        payload_with = build_global_health_surface(
            metrics_drift_compass=sample_drift_compass,
            metrics_budget_view=sample_budget_view,
            metrics_governance_signal=sample_governance_signal,
        )

        # Dynamics should be identical
        assert payload_without["dynamics"] == payload_with["dynamics"]

    def test_attach_metrics_governance_tile_function(
        self,
        sample_drift_compass: Dict[str, Any],
        sample_budget_view: Dict[str, Any],
        sample_governance_signal: Dict[str, Any],
    ) -> None:
        """Verify attach_metrics_governance_tile() works correctly."""
        from backend.health.global_surface import (
            attach_metrics_governance_tile,
        )

        # Start with empty payload
        payload = {"schema_version": "1.0.0"}

        # Attach tile
        updated = attach_metrics_governance_tile(
            payload,
            drift_compass=sample_drift_compass,
            budget_view=sample_budget_view,
            governance_signal=sample_governance_signal,
        )

        assert "metrics_governance" in updated
        assert updated["metrics_governance"]["status_light"] == "GREEN"

    def test_attach_metrics_governance_tile_no_inputs(self) -> None:
        """Verify attach returns unchanged payload when no inputs."""
        from backend.health.global_surface import (
            attach_metrics_governance_tile,
        )

        payload = {"schema_version": "1.0.0", "existing_key": "value"}

        updated = attach_metrics_governance_tile(payload)

        # Should be unchanged (no metrics_governance key added)
        assert "metrics_governance" not in updated
        assert updated["existing_key"] == "value"


# ==============================================================================
# Test Class: Neutral Language
# ==============================================================================


class TestNeutralLanguage:
    """
    Tests for neutral, descriptive language in tile output.

    SHADOW MODE: These tests verify language compliance only.
    """

    def test_headline_uses_neutral_language(
        self,
        sample_drift_compass: Dict[str, Any],
        sample_budget_view: Dict[str, Any],
        sample_governance_signal: Dict[str, Any],
    ) -> None:
        """Verify headline uses neutral, descriptive language only."""
        from backend.health.metrics_governance_adapter import (
            build_metrics_console_tile,
        )

        tile = build_metrics_console_tile(
            drift_compass=sample_drift_compass,
            budget_view=sample_budget_view,
            governance_signal=sample_governance_signal,
        )

        headline = tile["headline"].lower()

        # Forbidden evaluative words
        forbidden_words = [
            "good", "bad", "success", "failure", "better", "worse",
            "improve", "degrade", "excellent", "terrible", "perfect",
        ]

        for word in forbidden_words:
            assert word not in headline, (
                f"Headline should not contain evaluative word: {word}"
            )

        # Headline should be descriptive and end with period
        assert len(tile["headline"]) > 0
        assert tile["headline"].endswith(".")

    def test_council_summary_uses_neutral_language(
        self,
        sample_drift_compass: Dict[str, Any],
        sample_budget_view: Dict[str, Any],
        sample_governance_signal: Dict[str, Any],
    ) -> None:
        """Verify council summary uses neutral language."""
        from backend.health.metrics_governance_adapter import (
            build_metrics_console_tile,
            summarize_metrics_for_council,
        )

        tile = build_metrics_console_tile(
            drift_compass=sample_drift_compass,
            budget_view=sample_budget_view,
            governance_signal=sample_governance_signal,
        )

        summary = summarize_metrics_for_council(tile)

        summary_text = summary["summary"].lower()

        forbidden_words = ["good", "bad", "success", "failure"]

        for word in forbidden_words:
            assert word not in summary_text, (
                f"Summary should not contain evaluative word: {word}"
            )


# ==============================================================================
# Test Class: Alignment View Extended - WARN/BLOCK Paths with Realistic FO Signals
# ==============================================================================


class TestAlignmentViewWarnBlockPaths:
    """
    Extended regression tests for metrics_for_alignment_view covering:
    - WARN paths with realistic FO signals (degraded health, success_rate < 80%)
    - BLOCK paths with realistic FO signals (critical health, success_rate < 50%)
    - JSON round-trip of the fused view

    SHADOW MODE: These tests verify mapping and serialization only.
    """

    # --------------------------------------------------------------------------
    # Realistic FO Signal Fixtures
    # --------------------------------------------------------------------------

    @staticmethod
    def _build_warn_governance_signal_with_degraded_fo() -> Dict[str, Any]:
        """
        Build a realistic WARN governance signal with degraded FO vitals.

        Scenario: FO success_rate at 72%, health_status DEGRADED, throttle recommended.
        Drift compass showing DRIFTING with poly-cause on success_rate axis.
        Budget at ELEVATED utilization (85%).
        """
        return {
            "schema_version": "1.0.0",
            "signal_type": "metric_governance_signal",
            "timestamp": "2025-12-10T14:30:00Z",
            "mode": "SHADOW",
            "phase": "P4",
            "governance_status": "WARN",
            "governance_alignment": "TENSION",
            "sub_signals": {
                "drift_compass": {
                    "status": "WARN",
                    "compass_heading": "DRIFTING",
                    "drift_magnitude": 0.42,
                    "poly_cause_detected": True,
                    "poly_cause_axes": ["success_rate", "abstention"],
                },
                "budget_view": {
                    "status": "WARN",
                    "budget_status": "ELEVATED",
                    "utilization_pct": 85.3,
                    "blocking_resources": [],
                },
                "fo_vital_signs": {
                    "status": "WARN",
                    "health_status": "DEGRADED",
                    "success_rate": 72.5,
                    "should_throttle": True,
                    "should_boost": False,
                    "confidence": 0.85,
                    "abstention_rate": 8.2,
                    "runs_total": 1250,
                    "trend": "down",
                },
            },
            "conflict_matrix": {
                "drift_vs_budget": "TENSION",
                "drift_vs_fo": "TENSION",
                "budget_vs_fo": "ALIGNED",
                "any_conflict": True,
            },
            "reasons": [
                "[DriftCompass] Compass heading DRIFTING, magnitude 0.42",
                "[DriftCompass] Poly-cause detected on axes: success_rate, abstention",
                "[BudgetView] Utilization elevated at 85.3%",
                "[FOVitalSigns] Health DEGRADED, success rate 72.5%",
                "[FOVitalSigns] Throttle recommended due to degraded health",
            ],
            "safe_for_policy_update": True,
            "safe_for_promotion": True,
            "recommended_action": "THROTTLE",
            "intensity_multiplier": 0.75,
        }

    @staticmethod
    def _build_block_governance_signal_with_critical_fo() -> Dict[str, Any]:
        """
        Build a realistic BLOCK governance signal with critical FO vitals.

        Scenario: FO success_rate at 38%, health_status CRITICAL, throttle mandatory.
        Drift compass showing CRITICAL with high magnitude.
        Budget EXCEEDED (102%), derivation_steps blocking.
        """
        return {
            "schema_version": "1.0.0",
            "signal_type": "metric_governance_signal",
            "timestamp": "2025-12-10T15:45:00Z",
            "mode": "SHADOW",
            "phase": "P4",
            "governance_status": "BLOCK",
            "governance_alignment": "DIVERGENT",
            "sub_signals": {
                "drift_compass": {
                    "status": "BLOCK",
                    "compass_heading": "CRITICAL",
                    "drift_magnitude": 0.78,
                    "poly_cause_detected": True,
                    "poly_cause_axes": ["success_rate", "abstention", "rsi"],
                },
                "budget_view": {
                    "status": "BLOCK",
                    "budget_status": "EXCEEDED",
                    "utilization_pct": 102.4,
                    "blocking_resources": ["derivation_steps", "memory_pool"],
                },
                "fo_vital_signs": {
                    "status": "BLOCK",
                    "health_status": "CRITICAL",
                    "success_rate": 38.2,
                    "should_throttle": True,
                    "should_boost": False,
                    "confidence": 0.95,
                    "abstention_rate": 24.5,
                    "runs_total": 890,
                    "trend": "down",
                    "consecutive_failures": 12,
                },
            },
            "conflict_matrix": {
                "drift_vs_budget": "DIVERGENT",
                "drift_vs_fo": "DIVERGENT",
                "budget_vs_fo": "DIVERGENT",
                "any_conflict": True,
            },
            "reasons": [
                "[DriftCompass] Compass heading CRITICAL, magnitude 0.78",
                "[DriftCompass] Poly-cause detected on axes: success_rate, abstention, rsi",
                "[BudgetView] Budget EXCEEDED at 102.4%",
                "[BudgetView] Blocking resources: derivation_steps, memory_pool",
                "[FOVitalSigns] Health CRITICAL, success rate 38.2%",
                "[FOVitalSigns] Abstention rate elevated at 24.5%",
                "[FOVitalSigns] 12 consecutive failures detected",
            ],
            "safe_for_policy_update": False,
            "safe_for_promotion": False,
            "recommended_action": "HALT",
            "intensity_multiplier": 0.0,
        }

    @staticmethod
    def _build_warn_governance_signal_fo_throttle_only() -> Dict[str, Any]:
        """
        Build a WARN signal where FO is the primary concern (other signals OK).

        Scenario: Drift and budget are nominal, but FO showing early degradation.
        success_rate at 78%, health_status ALIVE but throttle recommended.
        """
        return {
            "schema_version": "1.0.0",
            "signal_type": "metric_governance_signal",
            "timestamp": "2025-12-10T16:00:00Z",
            "mode": "SHADOW",
            "phase": "P4",
            "governance_status": "WARN",
            "governance_alignment": "TENSION",
            "sub_signals": {
                "drift_compass": {
                    "status": "OK",
                    "compass_heading": "STABLE",
                    "drift_magnitude": 0.15,
                    "poly_cause_detected": False,
                    "poly_cause_axes": [],
                },
                "budget_view": {
                    "status": "OK",
                    "budget_status": "NOMINAL",
                    "utilization_pct": 62.0,
                    "blocking_resources": [],
                },
                "fo_vital_signs": {
                    "status": "WARN",
                    "health_status": "ALIVE",
                    "success_rate": 78.3,
                    "should_throttle": True,
                    "should_boost": False,
                    "confidence": 0.88,
                    "abstention_rate": 5.1,
                    "runs_total": 2100,
                    "trend": "down",
                },
            },
            "conflict_matrix": {
                "drift_vs_budget": "ALIGNED",
                "drift_vs_fo": "TENSION",
                "budget_vs_fo": "ALIGNED",
                "any_conflict": True,
            },
            "reasons": [
                "[DriftCompass] Compass heading STABLE, magnitude 0.15",
                "[BudgetView] Utilization nominal at 62.0%",
                "[FOVitalSigns] Health ALIVE, success rate 78.3%",
                "[FOVitalSigns] Throttle recommended due to declining trend",
            ],
            "safe_for_policy_update": True,
            "safe_for_promotion": True,
            "recommended_action": "THROTTLE",
            "intensity_multiplier": 0.85,
        }

    @staticmethod
    def _build_block_governance_signal_budget_only() -> Dict[str, Any]:
        """
        Build a BLOCK signal where budget is the primary blocker.

        Scenario: Drift and FO are nominal, but budget critically exceeded.
        """
        return {
            "schema_version": "1.0.0",
            "signal_type": "metric_governance_signal",
            "timestamp": "2025-12-10T17:30:00Z",
            "mode": "SHADOW",
            "phase": "P4",
            "governance_status": "BLOCK",
            "governance_alignment": "DIVERGENT",
            "sub_signals": {
                "drift_compass": {
                    "status": "OK",
                    "compass_heading": "STABLE",
                    "drift_magnitude": 0.08,
                    "poly_cause_detected": False,
                    "poly_cause_axes": [],
                },
                "budget_view": {
                    "status": "BLOCK",
                    "budget_status": "EXCEEDED",
                    "utilization_pct": 115.0,
                    "blocking_resources": ["derivation_steps"],
                },
                "fo_vital_signs": {
                    "status": "OK",
                    "health_status": "ALIVE",
                    "success_rate": 94.2,
                    "should_throttle": False,
                    "should_boost": True,
                    "confidence": 0.98,
                    "abstention_rate": 1.2,
                    "runs_total": 3500,
                    "trend": "flat",
                },
            },
            "conflict_matrix": {
                "drift_vs_budget": "DIVERGENT",
                "drift_vs_fo": "ALIGNED",
                "budget_vs_fo": "DIVERGENT",
                "any_conflict": True,
            },
            "reasons": [
                "[DriftCompass] Compass heading STABLE, magnitude 0.08",
                "[BudgetView] Budget EXCEEDED at 115.0%",
                "[BudgetView] Blocking resources: derivation_steps",
                "[FOVitalSigns] Health ALIVE, success rate 94.2%",
            ],
            "safe_for_policy_update": False,
            "safe_for_promotion": False,
            "recommended_action": "HALT",
            "intensity_multiplier": 0.0,
        }

    # --------------------------------------------------------------------------
    # WARN Path Tests with Realistic FO Signals
    # --------------------------------------------------------------------------

    def test_warn_path_degraded_fo_health(self) -> None:
        """
        Test WARN path with degraded FO health (success_rate 72.5%).

        Verifies alignment view correctly captures:
        - status = "warn"
        - severity = "warning"
        - FO sub-signal status reflected
        - Throttle recommendation visible
        """
        from backend.health.metrics_governance_adapter import (
            metrics_for_alignment_view,
        )

        signal = self._build_warn_governance_signal_with_degraded_fo()
        view = metrics_for_alignment_view(signal)

        # Status should be warn
        assert view["status"] == "warn", f"Expected 'warn', got '{view['status']}'"
        assert view["severity"] == "warning"

        # Sub-signal statuses should reflect FO degradation (lowercase per schema)
        statuses = view["sub_signal_statuses"]
        assert statuses["fo_vital_signs"] == "warn"
        assert statuses["drift_compass"] == "warn"
        assert statuses["budget_view"] == "warn"

        # Should still be safe for promotion (WARN doesn't block)
        assert view["safe_for_promotion"] is True

        # Summary should be present and non-empty
        assert len(view["summary"]) > 0

    def test_warn_path_fo_throttle_only(self) -> None:
        """
        Test WARN path where only FO signals throttle (drift/budget OK).

        Verifies alignment view captures FO-specific concerns even when
        other signals are healthy.
        """
        from backend.health.metrics_governance_adapter import (
            metrics_for_alignment_view,
        )

        signal = self._build_warn_governance_signal_fo_throttle_only()
        view = metrics_for_alignment_view(signal)

        assert view["status"] == "warn"
        assert view["severity"] == "warning"

        # Only FO should show warn (lowercase per schema)
        statuses = view["sub_signal_statuses"]
        assert statuses["fo_vital_signs"] == "warn"
        assert statuses["drift_compass"] == "ok"
        assert statuses["budget_view"] == "ok"

        # Layer should be metrics
        assert view["layer"] == "metrics"

    def test_warn_path_degraded_fo_json_roundtrip(self) -> None:
        """
        Test JSON round-trip of WARN alignment view with degraded FO.

        Verifies the fused view survives serialization/deserialization
        with all fields intact.
        """
        from backend.health.metrics_governance_adapter import (
            metrics_for_alignment_view,
        )

        signal = self._build_warn_governance_signal_with_degraded_fo()
        view = metrics_for_alignment_view(signal)

        # Serialize to JSON
        json_str = json.dumps(view, sort_keys=True)
        assert isinstance(json_str, str)
        assert len(json_str) > 0

        # Deserialize back
        parsed = json.loads(json_str)

        # Verify all fields survived
        assert parsed["status"] == view["status"]
        assert parsed["severity"] == view["severity"]
        assert parsed["alignment"] == view["alignment"]
        assert parsed["safe_for_promotion"] == view["safe_for_promotion"]
        assert parsed["layer"] == view["layer"]
        assert parsed["summary"] == view["summary"]

        # Verify sub_signal_statuses survived
        assert parsed["sub_signal_statuses"] == view["sub_signal_statuses"]

    # --------------------------------------------------------------------------
    # BLOCK Path Tests with Realistic FO Signals
    # --------------------------------------------------------------------------

    def test_block_path_critical_fo_health(self) -> None:
        """
        Test BLOCK path with critical FO health (success_rate 38.2%).

        Verifies alignment view correctly captures:
        - status = "block"
        - severity = "critical"
        - safe_for_promotion = False
        - All sub-signals showing block
        """
        from backend.health.metrics_governance_adapter import (
            metrics_for_alignment_view,
        )

        signal = self._build_block_governance_signal_with_critical_fo()
        view = metrics_for_alignment_view(signal)

        # Status should be block
        assert view["status"] == "block", f"Expected 'block', got '{view['status']}'"
        assert view["severity"] == "critical"

        # Should NOT be safe for promotion
        assert view["safe_for_promotion"] is False

        # All sub-signal statuses should be block (lowercase per schema)
        statuses = view["sub_signal_statuses"]
        assert statuses["fo_vital_signs"] == "block"
        assert statuses["drift_compass"] == "block"
        assert statuses["budget_view"] == "block"

        # Alignment should indicate divergence
        assert view["alignment"] == "DIVERGENT"

    def test_block_path_budget_only(self) -> None:
        """
        Test BLOCK path where only budget blocks (FO and drift OK).

        Verifies BLOCK propagates even when only one sub-signal is blocking.
        """
        from backend.health.metrics_governance_adapter import (
            metrics_for_alignment_view,
        )

        signal = self._build_block_governance_signal_budget_only()
        view = metrics_for_alignment_view(signal)

        assert view["status"] == "block"
        assert view["severity"] == "critical"
        assert view["safe_for_promotion"] is False

        # Only budget should show block (lowercase per schema)
        statuses = view["sub_signal_statuses"]
        assert statuses["budget_view"] == "block"
        assert statuses["fo_vital_signs"] == "ok"
        assert statuses["drift_compass"] == "ok"

    def test_block_path_critical_fo_json_roundtrip(self) -> None:
        """
        Test JSON round-trip of BLOCK alignment view with critical FO.

        Verifies the fused view survives serialization/deserialization
        with all fields intact including nested structures.
        """
        from backend.health.metrics_governance_adapter import (
            metrics_for_alignment_view,
        )

        signal = self._build_block_governance_signal_with_critical_fo()
        view = metrics_for_alignment_view(signal)

        # Serialize to JSON
        json_str = json.dumps(view, sort_keys=True)

        # Deserialize back
        parsed = json.loads(json_str)

        # Verify critical fields
        assert parsed["status"] == "block"
        assert parsed["severity"] == "critical"
        assert parsed["safe_for_promotion"] is False
        assert parsed["alignment"] == "DIVERGENT"

        # Verify structure equivalence
        assert parsed == view, "JSON round-trip should produce identical structure"

    # --------------------------------------------------------------------------
    # JSON Round-Trip Tests for Fused View
    # --------------------------------------------------------------------------

    def test_alignment_view_json_roundtrip_preserves_types(self) -> None:
        """
        Verify JSON round-trip preserves correct Python types.

        Ensures booleans remain booleans, strings remain strings,
        dicts remain dicts after serialization.
        """
        from backend.health.metrics_governance_adapter import (
            metrics_for_alignment_view,
        )

        signal = self._build_warn_governance_signal_with_degraded_fo()
        view = metrics_for_alignment_view(signal)

        json_str = json.dumps(view)
        parsed = json.loads(json_str)

        # Type checks
        assert isinstance(parsed["status"], str)
        assert isinstance(parsed["severity"], str)
        assert isinstance(parsed["safe_for_promotion"], bool)
        assert isinstance(parsed["sub_signal_statuses"], dict)
        assert isinstance(parsed["summary"], str)
        assert isinstance(parsed["layer"], str)
        assert isinstance(parsed["alignment"], str)

    def test_alignment_view_json_roundtrip_determinism(self) -> None:
        """
        Verify JSON round-trip produces deterministic output.

        Multiple serializations should produce identical JSON strings
        when sorted by keys.
        """
        from backend.health.metrics_governance_adapter import (
            metrics_for_alignment_view,
        )

        signal = self._build_block_governance_signal_with_critical_fo()
        view = metrics_for_alignment_view(signal)

        # Serialize multiple times
        json1 = json.dumps(view, sort_keys=True)
        json2 = json.dumps(view, sort_keys=True)
        json3 = json.dumps(view, sort_keys=True)

        assert json1 == json2 == json3, "JSON serialization should be deterministic"

    def test_alignment_view_all_scenarios_json_serializable(self) -> None:
        """
        Verify all test scenarios produce JSON-serializable alignment views.

        Comprehensive test covering all fixture scenarios.
        """
        from backend.health.metrics_governance_adapter import (
            metrics_for_alignment_view,
        )

        scenarios = [
            ("WARN degraded FO", self._build_warn_governance_signal_with_degraded_fo()),
            ("WARN FO throttle only", self._build_warn_governance_signal_fo_throttle_only()),
            ("BLOCK critical FO", self._build_block_governance_signal_with_critical_fo()),
            ("BLOCK budget only", self._build_block_governance_signal_budget_only()),
        ]

        for scenario_name, signal in scenarios:
            view = metrics_for_alignment_view(signal)

            # Must be serializable
            try:
                json_str = json.dumps(view)
                parsed = json.loads(json_str)
                assert parsed == view, f"Round-trip failed for scenario: {scenario_name}"
            except (TypeError, ValueError) as e:
                pytest.fail(f"Scenario '{scenario_name}' not JSON-serializable: {e}")
