"""Tests for Replay Governance Adapter.

Tests the replay governance adapter functions for Phase X integration.

These tests validate:
1. Console tile serialization (JSON-safe output)
2. Governance signal mapping (OK/WARN/BLOCK transitions)
3. Determinism with static input
4. Evidence attachment (non-mutating)
5. Schema-conforming output
6. SHADOW mode invariants respected

System Law Reference:
    docs/system_law/Replay_Governance_PhaseX_Binding.md

Schema References:
    docs/system_law/schemas/replay/replay_governance_radar.schema.json
    docs/system_law/schemas/replay/replay_promotion_eval.schema.json
    docs/system_law/schemas/replay/replay_director_panel.schema.json
    docs/system_law/schemas/replay/replay_global_console_tile.schema.json
"""

import copy
import json
import pytest
from typing import Any, Dict


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def radar_ok() -> Dict[str, Any]:
    """Radar with OK status and aligned governance."""
    return {
        "status": "ok",
        "governance_alignment": "aligned",
        "safe_for_policy_update": True,
        "safe_for_promotion": True,
        "conflict": False,
        "reasons": [],
        "metrics": {
            "determinism_score": 100,
            "hash_match_rate": 1.0,
            "coverage_pct": 95.0,
            "violation_count": 0,
            "warning_count": 0,
        },
        "drift_indicators": {
            "h_t_drift_detected": False,
            "config_drift_detected": False,
            "state_drift_detected": False,
        },
    }


@pytest.fixture
def radar_warn() -> Dict[str, Any]:
    """Radar with WARN status and tension alignment."""
    return {
        "status": "warn",
        "governance_alignment": "tension",
        "safe_for_policy_update": True,
        "safe_for_promotion": False,
        "conflict": False,
        "reasons": ["Minor hash variance detected"],
        "metrics": {
            "determinism_score": 85,
            "hash_match_rate": 0.95,
            "coverage_pct": 80.0,
            "violation_count": 1,
            "warning_count": 3,
        },
        "drift_indicators": {
            "h_t_drift_detected": True,
            "config_drift_detected": False,
            "state_drift_detected": False,
        },
    }


@pytest.fixture
def radar_block() -> Dict[str, Any]:
    """Radar with BLOCK status and divergent alignment."""
    return {
        "status": "block",
        "governance_alignment": "divergent",
        "safe_for_policy_update": False,
        "safe_for_promotion": False,
        "conflict": True,
        "reasons": ["Critical determinism failure", "Hash mismatch"],
        "metrics": {
            "determinism_score": 40,
            "hash_match_rate": 0.5,
            "coverage_pct": 60.0,
            "violation_count": 5,
            "warning_count": 10,
        },
        "drift_indicators": {
            "h_t_drift_detected": True,
            "config_drift_detected": True,
            "state_drift_detected": True,
        },
    }


@pytest.fixture
def promotion_eval_ok() -> Dict[str, Any]:
    """Promotion evaluation with OK status."""
    return {
        "status": "ok",
        "safe": True,
        "safe_for_policy_update": True,
        "safe_for_promotion": True,
        "reasons": [],
        "violations": [],
        "trace_summary": {
            "confidence_score": 1.0,
        },
    }


@pytest.fixture
def promotion_eval_warn() -> Dict[str, Any]:
    """Promotion evaluation with WARN status."""
    return {
        "status": "warn",
        "safe": True,
        "safe_for_policy_update": True,
        "safe_for_promotion": False,
        "reasons": ["Confidence below threshold"],
        "violations": [],
        "trace_summary": {
            "confidence_score": 0.75,
        },
    }


@pytest.fixture
def promotion_eval_block() -> Dict[str, Any]:
    """Promotion evaluation with BLOCK status."""
    return {
        "status": "block",
        "safe": False,
        "safe_for_policy_update": False,
        "safe_for_promotion": False,
        "reasons": ["Determinism failure"],
        "violations": ["Hash mismatch in trace 3"],
        "trace_summary": {
            "confidence_score": 0.3,
        },
    }


@pytest.fixture
def director_panel_ok() -> Dict[str, Any]:
    """Director panel with OK status."""
    return {
        "status": "ok",
        "safe": True,
        "violation_count": 0,
        "warning_count": 0,
        "headline": "All replay checks passed",
        "recommendation": "proceed",
        "conflict_flag": False,
    }


@pytest.fixture
def evidence_base() -> Dict[str, Any]:
    """Base evidence pack for attachment tests."""
    return {
        "run_id": "test_run_001",
        "timestamp": "2025-12-10T00:00:00Z",
        "governance": {
            "topology": {"status": "ok"},
        },
    }


# =============================================================================
# TEST 1: Console Tile Serialization (JSON-Safe Output)
# =============================================================================

class TestConsoleTileSerialization:
    """Test that console tile produces JSON-safe output."""

    def test_console_tile_is_json_serializable(
        self, radar_ok: Dict[str, Any], promotion_eval_ok: Dict[str, Any]
    ) -> None:
        """Console tile must be JSON serializable."""
        from backend.health.replay_governance_adapter import build_replay_console_tile

        tile = build_replay_console_tile(radar_ok, promotion_eval_ok)

        # Must not raise
        json_str = json.dumps(tile)
        assert json_str is not None
        assert len(json_str) > 0

        # Must round-trip correctly
        parsed = json.loads(json_str)
        assert parsed["schema_version"] == tile["schema_version"]
        assert parsed["status"] == tile["status"]

    def test_console_tile_with_director_panel_is_json_serializable(
        self,
        radar_ok: Dict[str, Any],
        promotion_eval_ok: Dict[str, Any],
        director_panel_ok: Dict[str, Any],
    ) -> None:
        """Console tile with director panel must be JSON serializable."""
        from backend.health.replay_governance_adapter import build_replay_console_tile

        tile = build_replay_console_tile(radar_ok, promotion_eval_ok, director_panel_ok)

        json_str = json.dumps(tile)
        assert json_str is not None
        assert "director_panel" in tile

    def test_console_tile_has_required_fields(
        self, radar_ok: Dict[str, Any], promotion_eval_ok: Dict[str, Any]
    ) -> None:
        """Console tile must have all required schema fields."""
        from backend.health.replay_governance_adapter import build_replay_console_tile

        tile = build_replay_console_tile(radar_ok, promotion_eval_ok)

        # Required fields from schema
        assert "schema_version" in tile
        assert "tile_type" in tile
        assert "timestamp" in tile
        assert "mode" in tile
        assert "status" in tile
        assert "safe" in tile

        # Tile type must be correct
        assert tile["tile_type"] == "replay_governance"
        assert tile["mode"] == "SHADOW"


# =============================================================================
# TEST 2: Governance Signal Mapping (OK/WARN/BLOCK Transitions)
# =============================================================================

class TestGovernanceSignalMapping:
    """Test governance signal collapse rules from System Law."""

    def test_ok_ok_produces_ok(
        self, radar_ok: Dict[str, Any], promotion_eval_ok: Dict[str, Any]
    ) -> None:
        """OK radar + OK promotion = OK signal."""
        from backend.health.replay_governance_adapter import replay_to_governance_signal

        signal = replay_to_governance_signal(radar_ok, promotion_eval_ok)

        assert signal["status"] == "ok"
        assert signal["governance_status"] == "ok"
        assert signal["conflict"] is False

    def test_warn_status_produces_warn(
        self, radar_warn: Dict[str, Any], promotion_eval_ok: Dict[str, Any]
    ) -> None:
        """WARN radar + OK promotion = WARN signal."""
        from backend.health.replay_governance_adapter import replay_to_governance_signal

        signal = replay_to_governance_signal(radar_warn, promotion_eval_ok)

        assert signal["status"] == "warn"
        assert signal["governance_alignment"] == "tension"

    def test_block_status_produces_block(
        self, radar_block: Dict[str, Any], promotion_eval_ok: Dict[str, Any]
    ) -> None:
        """BLOCK radar + OK promotion = BLOCK signal."""
        from backend.health.replay_governance_adapter import replay_to_governance_signal

        signal = replay_to_governance_signal(radar_block, promotion_eval_ok)

        assert signal["status"] == "block"
        assert signal["conflict"] is True

    def test_divergent_alignment_forces_block(
        self, radar_ok: Dict[str, Any], promotion_eval_ok: Dict[str, Any]
    ) -> None:
        """DIVERGENT alignment forces BLOCK regardless of status."""
        from backend.health.replay_governance_adapter import replay_to_governance_signal

        # Modify radar to have divergent alignment but OK status
        radar = dict(radar_ok)
        radar["governance_alignment"] = "divergent"

        signal = replay_to_governance_signal(radar, promotion_eval_ok)

        assert signal["status"] == "block"
        # Should include conflict reason
        conflict_reasons = [r for r in signal["reasons"] if "CONFLICT" in r]
        assert len(conflict_reasons) >= 1

    def test_reasons_are_prefixed(
        self, radar_warn: Dict[str, Any], promotion_eval_warn: Dict[str, Any]
    ) -> None:
        """All reasons must be prefixed with [Replay]."""
        from backend.health.replay_governance_adapter import replay_to_governance_signal

        signal = replay_to_governance_signal(radar_warn, promotion_eval_warn)

        for reason in signal["reasons"]:
            assert reason.startswith("[Replay]"), f"Reason not prefixed: {reason}"


# =============================================================================
# TEST 3: Determinism with Static Input
# =============================================================================

class TestDeterminism:
    """Test that functions produce deterministic output for same input."""

    def test_console_tile_deterministic(
        self, radar_ok: Dict[str, Any], promotion_eval_ok: Dict[str, Any]
    ) -> None:
        """Console tile must be deterministic (minus timestamp)."""
        from backend.health.replay_governance_adapter import build_replay_console_tile

        tile1 = build_replay_console_tile(radar_ok, promotion_eval_ok)
        tile2 = build_replay_console_tile(radar_ok, promotion_eval_ok)

        # Remove timestamps for comparison
        tile1_no_ts = {k: v for k, v in tile1.items() if k != "timestamp"}
        tile2_no_ts = {k: v for k, v in tile2.items() if k != "timestamp"}

        assert tile1_no_ts == tile2_no_ts

    def test_governance_signal_deterministic(
        self, radar_warn: Dict[str, Any], promotion_eval_warn: Dict[str, Any]
    ) -> None:
        """Governance signal must be fully deterministic."""
        from backend.health.replay_governance_adapter import replay_to_governance_signal

        signal1 = replay_to_governance_signal(radar_warn, promotion_eval_warn)
        signal2 = replay_to_governance_signal(radar_warn, promotion_eval_warn)

        assert signal1 == signal2

    def test_evidence_attachment_deterministic(
        self, evidence_base: Dict[str, Any], radar_ok: Dict[str, Any], promotion_eval_ok: Dict[str, Any]
    ) -> None:
        """Evidence attachment must be deterministic."""
        from backend.health.replay_governance_adapter import (
            attach_replay_governance_to_evidence,
            replay_to_governance_signal,
        )

        signal = replay_to_governance_signal(radar_ok, promotion_eval_ok)

        result1 = attach_replay_governance_to_evidence(evidence_base, signal)
        result2 = attach_replay_governance_to_evidence(evidence_base, signal)

        assert result1 == result2


# =============================================================================
# TEST 4: Evidence Attachment (Non-Mutating)
# =============================================================================

class TestEvidenceAttachment:
    """Test that evidence attachment is non-mutating."""

    def test_attach_does_not_mutate_original_evidence(
        self, evidence_base: Dict[str, Any], radar_ok: Dict[str, Any], promotion_eval_ok: Dict[str, Any]
    ) -> None:
        """Evidence attachment must not mutate original evidence."""
        from backend.health.replay_governance_adapter import (
            attach_replay_governance_to_evidence,
            replay_to_governance_signal,
        )

        signal = replay_to_governance_signal(radar_ok, promotion_eval_ok)
        original = copy.deepcopy(evidence_base)

        result = attach_replay_governance_to_evidence(evidence_base, signal)

        # Original must be unchanged
        assert evidence_base == original

        # Result must be different object
        assert result is not evidence_base

    def test_attach_does_not_mutate_original_signal(
        self, evidence_base: Dict[str, Any], radar_ok: Dict[str, Any], promotion_eval_ok: Dict[str, Any]
    ) -> None:
        """Evidence attachment must not mutate original signal."""
        from backend.health.replay_governance_adapter import (
            attach_replay_governance_to_evidence,
            replay_to_governance_signal,
        )

        signal = replay_to_governance_signal(radar_ok, promotion_eval_ok)
        original_signal = copy.deepcopy(signal)

        _ = attach_replay_governance_to_evidence(evidence_base, signal)

        # Signal must be unchanged
        assert signal == original_signal

    def test_attach_creates_governance_replay_key(
        self, evidence_base: Dict[str, Any], radar_ok: Dict[str, Any], promotion_eval_ok: Dict[str, Any]
    ) -> None:
        """Evidence attachment must create governance.replay key."""
        from backend.health.replay_governance_adapter import (
            attach_replay_governance_to_evidence,
            replay_to_governance_signal,
        )

        signal = replay_to_governance_signal(radar_ok, promotion_eval_ok)
        result = attach_replay_governance_to_evidence(evidence_base, signal)

        assert "governance" in result
        assert "replay" in result["governance"]
        assert result["governance"]["replay"]["signal_type"] == "replay_safety"


# =============================================================================
# TEST 5: Schema-Conforming Output
# =============================================================================

class TestSchemaConformance:
    """Test that output conforms to defined schemas."""

    def test_console_tile_has_schema_version(
        self, radar_ok: Dict[str, Any], promotion_eval_ok: Dict[str, Any]
    ) -> None:
        """Console tile must have schema_version field."""
        from backend.health.replay_governance_adapter import build_replay_console_tile

        tile = build_replay_console_tile(radar_ok, promotion_eval_ok)

        assert tile["schema_version"] == "1.0.0"

    def test_governance_signal_has_signal_type(
        self, radar_ok: Dict[str, Any], promotion_eval_ok: Dict[str, Any]
    ) -> None:
        """Governance signal must have signal_type field."""
        from backend.health.replay_governance_adapter import replay_to_governance_signal

        signal = replay_to_governance_signal(radar_ok, promotion_eval_ok)

        assert signal["signal_type"] == "replay_safety"

    def test_console_tile_status_is_valid_enum(
        self, radar_ok: Dict[str, Any], promotion_eval_ok: Dict[str, Any]
    ) -> None:
        """Console tile status must be valid enum value."""
        from backend.health.replay_governance_adapter import build_replay_console_tile

        tile = build_replay_console_tile(radar_ok, promotion_eval_ok)

        assert tile["status"] in ("ok", "warn", "block")

    def test_console_tile_mode_is_shadow(
        self, radar_ok: Dict[str, Any], promotion_eval_ok: Dict[str, Any]
    ) -> None:
        """Console tile mode must be SHADOW in Phase X."""
        from backend.health.replay_governance_adapter import build_replay_console_tile

        tile = build_replay_console_tile(radar_ok, promotion_eval_ok)

        assert tile["mode"] == "SHADOW"

    def test_summary_metrics_have_required_fields(
        self, radar_ok: Dict[str, Any], promotion_eval_ok: Dict[str, Any]
    ) -> None:
        """Summary metrics must have required fields."""
        from backend.health.replay_governance_adapter import build_replay_console_tile

        tile = build_replay_console_tile(radar_ok, promotion_eval_ok)

        metrics = tile["summary_metrics"]
        assert "determinism_rate" in metrics
        assert "critical_incident_rate" in metrics
        assert "hot_fingerprints_count" in metrics
        assert "replay_ok_for_promotion" in metrics


# =============================================================================
# TEST 6: SHADOW Mode Invariants Respected
# =============================================================================

class TestShadowModeInvariants:
    """Test that SHADOW mode invariants are respected."""

    def test_console_tile_has_shadow_mode_contract(
        self, radar_ok: Dict[str, Any], promotion_eval_ok: Dict[str, Any]
    ) -> None:
        """Console tile must attest to SHADOW mode contract."""
        from backend.health.replay_governance_adapter import build_replay_console_tile

        tile = build_replay_console_tile(radar_ok, promotion_eval_ok)

        contract = tile["shadow_mode_contract"]
        assert contract["observational_only"] is True
        assert contract["no_control_flow_influence"] is True
        assert contract["no_governance_modification"] is True

    def test_console_tile_does_not_contain_control_flow_hooks(
        self, radar_ok: Dict[str, Any], promotion_eval_ok: Dict[str, Any]
    ) -> None:
        """Console tile must not contain control flow hooks."""
        from backend.health.replay_governance_adapter import build_replay_console_tile

        tile = build_replay_console_tile(radar_ok, promotion_eval_ok)

        # Should not contain any callable or hook-like fields
        json_str = json.dumps(tile)
        assert "callback" not in json_str.lower()
        assert "hook" not in json_str.lower()
        assert "function" not in json_str.lower()

    def test_governance_signal_is_read_only(
        self, radar_ok: Dict[str, Any], promotion_eval_ok: Dict[str, Any]
    ) -> None:
        """Governance signal must be a plain dict (read-only)."""
        from backend.health.replay_governance_adapter import replay_to_governance_signal

        signal = replay_to_governance_signal(radar_ok, promotion_eval_ok)

        # Must be a plain dict, not a dataclass or object with methods
        assert isinstance(signal, dict)
        assert not hasattr(signal, "__dataclass_fields__")

    def test_attach_function_is_pure(
        self, evidence_base: Dict[str, Any], radar_ok: Dict[str, Any], promotion_eval_ok: Dict[str, Any]
    ) -> None:
        """Attach function must be pure (no side effects)."""
        from backend.health.replay_governance_adapter import (
            attach_replay_governance_to_evidence,
            replay_to_governance_signal,
        )

        signal = replay_to_governance_signal(radar_ok, promotion_eval_ok)

        # Call multiple times
        result1 = attach_replay_governance_to_evidence(evidence_base, signal)
        result2 = attach_replay_governance_to_evidence(evidence_base, signal)

        # Results must be equal (function is pure)
        assert result1 == result2

        # Neither result should be the same object as input
        assert result1 is not evidence_base
        assert result2 is not evidence_base

    def test_status_light_maps_correctly(
        self,
        radar_ok: Dict[str, Any],
        radar_warn: Dict[str, Any],
        radar_block: Dict[str, Any],
        promotion_eval_ok: Dict[str, Any],
    ) -> None:
        """Status light must map correctly from status."""
        from backend.health.replay_governance_adapter import build_replay_console_tile

        tile_ok = build_replay_console_tile(radar_ok, promotion_eval_ok)
        tile_warn = build_replay_console_tile(radar_warn, promotion_eval_ok)
        tile_block = build_replay_console_tile(radar_block, promotion_eval_ok)

        assert tile_ok["status_light"] == "GREEN"
        assert tile_warn["status_light"] == "YELLOW"
        assert tile_block["status_light"] == "RED"


# =============================================================================
# TEST 7: P3 Stability Report Binding
# =============================================================================

class TestP3StabilityReportBinding:
    """Test P3 stability report attachment."""

    @pytest.fixture
    def p3_stability_report_base(self) -> Dict[str, Any]:
        """Base P3 stability report for testing."""
        return {
            "run_id": "p3_test_run_001",
            "timestamp": "2025-12-10T00:00:00Z",
            "phase": "P3",
            "synthetic": True,
            "stability_score": 0.95,
        }

    def test_p3_attachment_is_json_serializable(
        self,
        p3_stability_report_base: Dict[str, Any],
        radar_ok: Dict[str, Any],
        promotion_eval_ok: Dict[str, Any],
    ) -> None:
        """P3 attachment must produce JSON-serializable output."""
        from backend.health.replay_governance_adapter import (
            attach_replay_governance_to_p3_stability_report,
            replay_to_governance_signal,
        )

        signal = replay_to_governance_signal(radar_ok, promotion_eval_ok)
        result = attach_replay_governance_to_p3_stability_report(p3_stability_report_base, signal)

        # Must serialize without error
        json_str = json.dumps(result)
        assert json_str is not None
        assert len(json_str) > 0

    def test_p3_attachment_is_non_mutating(
        self,
        p3_stability_report_base: Dict[str, Any],
        radar_ok: Dict[str, Any],
        promotion_eval_ok: Dict[str, Any],
    ) -> None:
        """P3 attachment must not mutate original report."""
        from backend.health.replay_governance_adapter import (
            attach_replay_governance_to_p3_stability_report,
            replay_to_governance_signal,
        )

        signal = replay_to_governance_signal(radar_ok, promotion_eval_ok)
        original = copy.deepcopy(p3_stability_report_base)

        result = attach_replay_governance_to_p3_stability_report(p3_stability_report_base, signal)

        # Original must be unchanged
        assert p3_stability_report_base == original
        # Result must be different object
        assert result is not p3_stability_report_base

    def test_p3_attachment_has_required_fields(
        self,
        p3_stability_report_base: Dict[str, Any],
        radar_ok: Dict[str, Any],
        promotion_eval_ok: Dict[str, Any],
    ) -> None:
        """P3 attachment must have all required P3 summary fields."""
        from backend.health.replay_governance_adapter import (
            attach_replay_governance_to_p3_stability_report,
            replay_to_governance_signal,
        )

        signal = replay_to_governance_signal(radar_ok, promotion_eval_ok)
        result = attach_replay_governance_to_p3_stability_report(p3_stability_report_base, signal)

        assert "replay_governance" in result
        summary = result["replay_governance"]

        # Required P3 fields
        assert "status" in summary
        assert "determinism_rate" in summary
        assert "critical_incident_rate" in summary
        assert "hot_fingerprints_count" in summary
        assert "governance_alignment" in summary

    def test_p3_status_maps_correctly(
        self,
        p3_stability_report_base: Dict[str, Any],
        radar_ok: Dict[str, Any],
        radar_warn: Dict[str, Any],
        radar_block: Dict[str, Any],
        promotion_eval_ok: Dict[str, Any],
    ) -> None:
        """P3 status must map correctly from signal status."""
        from backend.health.replay_governance_adapter import (
            attach_replay_governance_to_p3_stability_report,
            replay_to_governance_signal,
        )

        signal_ok = replay_to_governance_signal(radar_ok, promotion_eval_ok)
        signal_warn = replay_to_governance_signal(radar_warn, promotion_eval_ok)
        signal_block = replay_to_governance_signal(radar_block, promotion_eval_ok)

        result_ok = attach_replay_governance_to_p3_stability_report(p3_stability_report_base, signal_ok)
        result_warn = attach_replay_governance_to_p3_stability_report(p3_stability_report_base, signal_warn)
        result_block = attach_replay_governance_to_p3_stability_report(p3_stability_report_base, signal_block)

        assert result_ok["replay_governance"]["status"] == "ok"
        assert result_warn["replay_governance"]["status"] == "warn"
        assert result_block["replay_governance"]["status"] == "block"


# =============================================================================
# TEST 8: P4 Calibration Report Binding
# =============================================================================

class TestP4CalibrationReportBinding:
    """Test P4 calibration report attachment."""

    @pytest.fixture
    def p4_calibration_report_base(self) -> Dict[str, Any]:
        """Base P4 calibration report for testing."""
        return {
            "run_id": "p4_test_run_001",
            "timestamp": "2025-12-10T00:00:00Z",
            "phase": "P4",
            "shadow_mode": True,
            "calibration_score": 0.92,
        }

    def test_p4_attachment_is_json_serializable(
        self,
        p4_calibration_report_base: Dict[str, Any],
        radar_ok: Dict[str, Any],
        promotion_eval_ok: Dict[str, Any],
    ) -> None:
        """P4 attachment must produce JSON-serializable output."""
        from backend.health.replay_governance_adapter import (
            attach_replay_governance_to_p4_calibration_report,
            replay_to_governance_signal,
        )

        signal = replay_to_governance_signal(radar_ok, promotion_eval_ok)
        result = attach_replay_governance_to_p4_calibration_report(p4_calibration_report_base, signal)

        # Must serialize without error
        json_str = json.dumps(result)
        assert json_str is not None
        assert len(json_str) > 0

    def test_p4_attachment_is_non_mutating(
        self,
        p4_calibration_report_base: Dict[str, Any],
        radar_ok: Dict[str, Any],
        promotion_eval_ok: Dict[str, Any],
    ) -> None:
        """P4 attachment must not mutate original report."""
        from backend.health.replay_governance_adapter import (
            attach_replay_governance_to_p4_calibration_report,
            replay_to_governance_signal,
        )

        signal = replay_to_governance_signal(radar_ok, promotion_eval_ok)
        original = copy.deepcopy(p4_calibration_report_base)

        result = attach_replay_governance_to_p4_calibration_report(p4_calibration_report_base, signal)

        # Original must be unchanged
        assert p4_calibration_report_base == original
        # Result must be different object
        assert result is not p4_calibration_report_base

    def test_p4_attachment_has_required_fields(
        self,
        p4_calibration_report_base: Dict[str, Any],
        radar_ok: Dict[str, Any],
        promotion_eval_ok: Dict[str, Any],
    ) -> None:
        """P4 attachment must have all required P4 calibration fields."""
        from backend.health.replay_governance_adapter import (
            attach_replay_governance_to_p4_calibration_report,
            replay_to_governance_signal,
        )

        signal = replay_to_governance_signal(radar_ok, promotion_eval_ok)
        result = attach_replay_governance_to_p4_calibration_report(p4_calibration_report_base, signal)

        assert "replay_calibration" in result
        calibration = result["replay_calibration"]

        # Required P4 fields
        assert "status" in calibration
        assert "recency_of_replay" in calibration
        assert "safety_status" in calibration
        assert "radar_status" in calibration
        assert "conflict" in calibration

    def test_p4_recency_uses_provided_timestamp(
        self,
        p4_calibration_report_base: Dict[str, Any],
        radar_ok: Dict[str, Any],
        promotion_eval_ok: Dict[str, Any],
    ) -> None:
        """P4 attachment must use provided recency timestamp."""
        from backend.health.replay_governance_adapter import (
            attach_replay_governance_to_p4_calibration_report,
            replay_to_governance_signal,
        )

        signal = replay_to_governance_signal(radar_ok, promotion_eval_ok)
        custom_ts = "2025-12-10T12:00:00Z"

        result = attach_replay_governance_to_p4_calibration_report(
            p4_calibration_report_base, signal, recency_timestamp=custom_ts
        )

        assert result["replay_calibration"]["recency_of_replay"] == custom_ts

    def test_p4_conflict_flag_propagates(
        self,
        p4_calibration_report_base: Dict[str, Any],
        radar_block: Dict[str, Any],
        promotion_eval_ok: Dict[str, Any],
    ) -> None:
        """P4 attachment must propagate conflict flag from signal."""
        from backend.health.replay_governance_adapter import (
            attach_replay_governance_to_p4_calibration_report,
            replay_to_governance_signal,
        )

        signal = replay_to_governance_signal(radar_block, promotion_eval_ok)
        result = attach_replay_governance_to_p4_calibration_report(p4_calibration_report_base, signal)

        # Block radar has conflict=True
        assert result["replay_calibration"]["conflict"] is True


# =============================================================================
# TEST 9: GGFL Fusion Adapter (replay_for_alignment_view)
# =============================================================================

class TestGGFLFusionAdapter:
    """Test GGFL fusion adapter replay_for_alignment_view()."""

    def test_ggfl_produces_correct_shape(
        self, radar_ok: Dict[str, Any], promotion_eval_ok: Dict[str, Any]
    ) -> None:
        """GGFL adapter must produce correct shape."""
        from backend.health.replay_governance_adapter import (
            replay_for_alignment_view,
            replay_to_governance_signal,
        )

        signal = replay_to_governance_signal(radar_ok, promotion_eval_ok)
        result = replay_for_alignment_view(signal)

        # Required GGFL fields
        assert "status" in result
        assert "alignment" in result
        assert "conflict" in result
        assert "top_reasons" in result

    def test_ggfl_status_is_lowercase(
        self, radar_ok: Dict[str, Any], promotion_eval_ok: Dict[str, Any]
    ) -> None:
        """GGFL status must be lowercase."""
        from backend.health.replay_governance_adapter import (
            replay_for_alignment_view,
            replay_to_governance_signal,
        )

        signal = replay_to_governance_signal(radar_ok, promotion_eval_ok)
        result = replay_for_alignment_view(signal)

        assert result["status"] == result["status"].lower()
        assert result["status"] in ("ok", "warn", "block")

    def test_ggfl_strips_replay_prefix_from_reasons(
        self, radar_warn: Dict[str, Any], promotion_eval_warn: Dict[str, Any]
    ) -> None:
        """GGFL adapter must strip [Replay] prefix from reasons."""
        from backend.health.replay_governance_adapter import (
            replay_for_alignment_view,
            replay_to_governance_signal,
        )

        signal = replay_to_governance_signal(radar_warn, promotion_eval_warn)
        result = replay_for_alignment_view(signal)

        # All reasons should have [Replay] stripped
        for reason in result["top_reasons"]:
            assert not reason.startswith("[Replay]"), f"Prefix not stripped: {reason}"

    def test_ggfl_preserves_conflict_prefix(
        self, radar_block: Dict[str, Any], promotion_eval_ok: Dict[str, Any]
    ) -> None:
        """GGFL adapter must preserve [CONFLICT] prefix."""
        from backend.health.replay_governance_adapter import (
            replay_for_alignment_view,
            replay_to_governance_signal,
        )

        signal = replay_to_governance_signal(radar_block, promotion_eval_ok)
        result = replay_for_alignment_view(signal)

        # Should have at least one reason with [CONFLICT]
        conflict_reasons = [r for r in result["top_reasons"] if "[CONFLICT]" in r]
        assert len(conflict_reasons) >= 1

    def test_ggfl_limits_reasons_to_five(
        self, radar_block: Dict[str, Any], promotion_eval_block: Dict[str, Any]
    ) -> None:
        """GGFL adapter must limit top_reasons to 5."""
        from backend.health.replay_governance_adapter import (
            replay_for_alignment_view,
            replay_to_governance_signal,
        )

        signal = replay_to_governance_signal(radar_block, promotion_eval_block)
        result = replay_for_alignment_view(signal)

        assert len(result["top_reasons"]) <= 5

    def test_ggfl_alignment_mapping(
        self,
        radar_ok: Dict[str, Any],
        radar_warn: Dict[str, Any],
        radar_block: Dict[str, Any],
        promotion_eval_ok: Dict[str, Any],
    ) -> None:
        """GGFL alignment must map correctly from signal."""
        from backend.health.replay_governance_adapter import (
            replay_for_alignment_view,
            replay_to_governance_signal,
        )

        signal_ok = replay_to_governance_signal(radar_ok, promotion_eval_ok)
        signal_warn = replay_to_governance_signal(radar_warn, promotion_eval_ok)
        signal_block = replay_to_governance_signal(radar_block, promotion_eval_ok)

        result_ok = replay_for_alignment_view(signal_ok)
        result_warn = replay_for_alignment_view(signal_warn)
        result_block = replay_for_alignment_view(signal_block)

        assert result_ok["alignment"] == "aligned"
        assert result_warn["alignment"] == "tension"
        assert result_block["alignment"] == "divergent"

    def test_ggfl_conflict_flag_mapping(
        self,
        radar_ok: Dict[str, Any],
        radar_block: Dict[str, Any],
        promotion_eval_ok: Dict[str, Any],
    ) -> None:
        """GGFL conflict flag must map correctly from signal."""
        from backend.health.replay_governance_adapter import (
            replay_for_alignment_view,
            replay_to_governance_signal,
        )

        signal_ok = replay_to_governance_signal(radar_ok, promotion_eval_ok)
        signal_block = replay_to_governance_signal(radar_block, promotion_eval_ok)

        result_ok = replay_for_alignment_view(signal_ok)
        result_block = replay_for_alignment_view(signal_block)

        assert result_ok["conflict"] is False
        assert result_block["conflict"] is True

    def test_ggfl_is_read_only(
        self, radar_ok: Dict[str, Any], promotion_eval_ok: Dict[str, Any]
    ) -> None:
        """GGFL adapter must be read-only (not modify input)."""
        from backend.health.replay_governance_adapter import (
            replay_for_alignment_view,
            replay_to_governance_signal,
        )

        signal = replay_to_governance_signal(radar_ok, promotion_eval_ok)
        original_signal = copy.deepcopy(signal)

        _ = replay_for_alignment_view(signal)

        # Signal must be unchanged
        assert signal == original_signal


# =============================================================================
# TEST 10: P5 Replay Safety Extraction (P5-C Unit Tests)
# =============================================================================

class TestP5ReplaySafetyExtraction:
    """Test P5 replay safety signal extraction from logs."""

    @pytest.fixture
    def p5_replay_logs_healthy(self) -> list:
        """Healthy replay logs with all hashes matching."""
        return [
            {
                "cycle_id": "cycle_001",
                "trace_hash": "abc123",
                "timestamp": "2025-12-10T00:00:00Z",
                "latency_ms": 50.0,
            },
            {
                "cycle_id": "cycle_002",
                "trace_hash": "def456",
                "timestamp": "2025-12-10T00:01:00Z",
                "latency_ms": 55.0,
            },
            {
                "cycle_id": "cycle_003",
                "trace_hash": "ghi789",
                "timestamp": "2025-12-10T00:02:00Z",
                "latency_ms": 48.0,
            },
        ]

    @pytest.fixture
    def p5_expected_hashes_match(self) -> Dict[str, str]:
        """Expected hashes that match the healthy logs."""
        return {
            "cycle_001": "abc123",
            "cycle_002": "def456",
            "cycle_003": "ghi789",
        }

    @pytest.fixture
    def p5_expected_hashes_mismatch(self) -> Dict[str, str]:
        """Expected hashes with mismatches."""
        return {
            "cycle_001": "abc123",
            "cycle_002": "WRONG_HASH",  # Mismatch
            "cycle_003": "ghi789",
        }

    @pytest.fixture
    def p5_replay_logs_with_failure(self) -> list:
        """Replay logs with determinism failure."""
        return [
            {
                "cycle_id": "cycle_001",
                "trace_hash": "abc123",
                "timestamp": "2025-12-10T00:00:00Z",
                "latency_ms": 50.0,
            },
            {
                "cycle_id": "cycle_002",
                "trace_hash": "def456",
                "timestamp": "2025-12-10T00:01:00Z",
                "latency_ms": 55.0,
                "determinism_check": {
                    "failed": True,
                    "reason": "State divergence detected",
                },
            },
        ]

    def test_extract_p5_green_band(
        self, p5_replay_logs_healthy: list, p5_expected_hashes_match: Dict[str, str]
    ) -> None:
        """Extract P5 signal with GREEN band (all hashes match)."""
        from backend.health.replay_governance_adapter import extract_p5_replay_safety_from_logs

        signal = extract_p5_replay_safety_from_logs(
            p5_replay_logs_healthy,
            production_run_id="prod_run_001",
            expected_hashes=p5_expected_hashes_match,
            telemetry_source="real",
        )

        assert signal["status"] == "ok"
        assert signal["determinism_band"] == "GREEN"
        assert signal["determinism_rate"] == 1.0
        assert signal["hash_match_count"] == 3
        assert signal["hash_mismatch_count"] == 0
        assert signal["p5_grade"] is True

    def test_extract_p5_yellow_band(
        self, p5_replay_logs_healthy: list, p5_expected_hashes_mismatch: Dict[str, str]
    ) -> None:
        """Extract P5 signal with YELLOW band (some hash mismatches)."""
        from backend.health.replay_governance_adapter import extract_p5_replay_safety_from_logs

        signal = extract_p5_replay_safety_from_logs(
            p5_replay_logs_healthy,
            production_run_id="prod_run_001",
            expected_hashes=p5_expected_hashes_mismatch,
            telemetry_source="real",
        )

        # 2 matches, 1 mismatch = 66.7% < 0.70 threshold -> RED
        # Actually: 2/3 = 0.667 < 0.70 -> RED band
        assert signal["determinism_rate"] == pytest.approx(2/3, rel=0.01)
        assert signal["determinism_band"] == "RED"
        assert signal["status"] == "block"

    def test_extract_p5_includes_critical_incidents(
        self, p5_replay_logs_healthy: list, p5_expected_hashes_mismatch: Dict[str, str]
    ) -> None:
        """Extract P5 signal includes critical incidents for mismatches."""
        from backend.health.replay_governance_adapter import extract_p5_replay_safety_from_logs

        signal = extract_p5_replay_safety_from_logs(
            p5_replay_logs_healthy,
            production_run_id="prod_run_001",
            expected_hashes=p5_expected_hashes_mismatch,
            telemetry_source="real",
        )

        assert len(signal["critical_incidents"]) >= 1
        incident = signal["critical_incidents"][0]
        assert incident["type"] == "hash_mismatch"
        assert incident["cycle_id"] == "cycle_002"

    def test_extract_p5_latency_computed(
        self, p5_replay_logs_healthy: list
    ) -> None:
        """Extract P5 signal computes average latency."""
        from backend.health.replay_governance_adapter import extract_p5_replay_safety_from_logs

        signal = extract_p5_replay_safety_from_logs(
            p5_replay_logs_healthy,
            production_run_id="prod_run_001",
            telemetry_source="real",
        )

        # Average of 50, 55, 48 = 51.0
        assert signal["replay_latency_ms"] == pytest.approx(51.0, rel=0.01)

    def test_extract_p5_grade_requires_real_telemetry(
        self, p5_replay_logs_healthy: list, p5_expected_hashes_match: Dict[str, str]
    ) -> None:
        """P5 grade requires telemetry_source='real'."""
        from backend.health.replay_governance_adapter import extract_p5_replay_safety_from_logs

        signal = extract_p5_replay_safety_from_logs(
            p5_replay_logs_healthy,
            production_run_id="prod_run_001",
            expected_hashes=p5_expected_hashes_match,
            telemetry_source="synthetic",  # Not real
        )

        assert signal["p5_grade"] is False
        assert signal["telemetry_source"] == "synthetic"

    def test_extract_p5_handles_determinism_failure(
        self, p5_replay_logs_with_failure: list
    ) -> None:
        """Extract P5 signal handles determinism_check failures."""
        from backend.health.replay_governance_adapter import extract_p5_replay_safety_from_logs

        signal = extract_p5_replay_safety_from_logs(
            p5_replay_logs_with_failure,
            production_run_id="prod_run_001",
            telemetry_source="real",
        )

        # Should have critical incident for determinism failure
        det_incidents = [i for i in signal["critical_incidents"] if i["type"] == "determinism_failure"]
        assert len(det_incidents) >= 1

    def test_extract_p5_is_json_serializable(
        self, p5_replay_logs_healthy: list, p5_expected_hashes_match: Dict[str, str]
    ) -> None:
        """P5 signal must be JSON serializable."""
        from backend.health.replay_governance_adapter import extract_p5_replay_safety_from_logs

        signal = extract_p5_replay_safety_from_logs(
            p5_replay_logs_healthy,
            production_run_id="prod_run_001",
            expected_hashes=p5_expected_hashes_match,
            telemetry_source="real",
        )

        json_str = json.dumps(signal)
        assert json_str is not None
        assert len(json_str) > 0

    def test_extract_p5_empty_logs_defaults_deterministic(self) -> None:
        """Empty logs default to deterministic (rate=1.0)."""
        from backend.health.replay_governance_adapter import extract_p5_replay_safety_from_logs

        signal = extract_p5_replay_safety_from_logs(
            [],
            production_run_id="prod_run_001",
            telemetry_source="real",
        )

        assert signal["determinism_rate"] == 1.0
        assert signal["determinism_band"] == "GREEN"
        # No checks performed, so not P5 grade
        assert signal["p5_grade"] is False


# =============================================================================
# TEST 11: P5 Replay Governance Tile Builder (P5-C Unit Tests)
# =============================================================================

class TestP5ReplayGovernanceTileBuilder:
    """Test P5 replay governance tile builder."""

    @pytest.fixture
    def p5_signal_ok(self) -> Dict[str, Any]:
        """P5 signal with OK status."""
        return {
            "schema_version": "1.0.0",
            "status": "ok",
            "determinism_rate": 1.0,
            "determinism_band": "GREEN",
            "hash_match_count": 10,
            "hash_mismatch_count": 0,
            "critical_incidents": [],
            "telemetry_source": "real",
            "production_run_id": "prod_run_001",
            "replay_latency_ms": 50.0,
            "p5_grade": True,
            "reasons": [],
        }

    @pytest.fixture
    def p5_signal_warn(self) -> Dict[str, Any]:
        """P5 signal with WARN status."""
        return {
            "schema_version": "1.0.0",
            "status": "warn",
            "determinism_rate": 0.80,
            "determinism_band": "YELLOW",
            "hash_match_count": 8,
            "hash_mismatch_count": 2,
            "critical_incidents": [],
            "telemetry_source": "real",
            "production_run_id": "prod_run_001",
            "replay_latency_ms": 55.0,
            "p5_grade": True,
            "reasons": ["[Replay] Determinism rate 80.00% in YELLOW band"],
        }

    @pytest.fixture
    def p5_signal_block(self) -> Dict[str, Any]:
        """P5 signal with BLOCK status."""
        return {
            "schema_version": "1.0.0",
            "status": "block",
            "determinism_rate": 0.50,
            "determinism_band": "RED",
            "hash_match_count": 5,
            "hash_mismatch_count": 5,
            "critical_incidents": [{"type": "hash_mismatch", "cycle_id": "cycle_001"}],
            "telemetry_source": "real",
            "production_run_id": "prod_run_001",
            "replay_latency_ms": 100.0,
            "p5_grade": True,
            "reasons": ["[Replay] Determinism rate 50.00% in RED band"],
        }

    def test_build_p5_tile_ok_status(
        self,
        p5_signal_ok: Dict[str, Any],
        radar_ok: Dict[str, Any],
        promotion_eval_ok: Dict[str, Any],
    ) -> None:
        """Build P5 tile with OK status."""
        from backend.health.replay_governance_adapter import build_p5_replay_governance_tile

        tile = build_p5_replay_governance_tile(p5_signal_ok, radar_ok, promotion_eval_ok)

        assert tile["phase"] == "P5"
        assert tile["status"] == "ok"
        assert tile["p5_grade"] is True
        assert tile["determinism_band"] == "GREEN"

    def test_build_p5_tile_warn_status(
        self,
        p5_signal_warn: Dict[str, Any],
        radar_ok: Dict[str, Any],
        promotion_eval_ok: Dict[str, Any],
    ) -> None:
        """Build P5 tile with WARN status."""
        from backend.health.replay_governance_adapter import build_p5_replay_governance_tile

        tile = build_p5_replay_governance_tile(p5_signal_warn, radar_ok, promotion_eval_ok)

        assert tile["status"] == "warn"
        assert tile["determinism_band"] == "YELLOW"
        assert tile["status_light"] == "YELLOW"

    def test_build_p5_tile_block_status(
        self,
        p5_signal_block: Dict[str, Any],
        radar_ok: Dict[str, Any],
        promotion_eval_ok: Dict[str, Any],
    ) -> None:
        """Build P5 tile with BLOCK status."""
        from backend.health.replay_governance_adapter import build_p5_replay_governance_tile

        tile = build_p5_replay_governance_tile(p5_signal_block, radar_ok, promotion_eval_ok)

        assert tile["status"] == "block"
        assert tile["determinism_band"] == "RED"
        assert tile["status_light"] == "RED"
        assert tile["safe"] is False

    def test_build_p5_tile_includes_extensions(
        self,
        p5_signal_ok: Dict[str, Any],
        radar_ok: Dict[str, Any],
        promotion_eval_ok: Dict[str, Any],
    ) -> None:
        """P5 tile includes extension fields when enabled."""
        from backend.health.replay_governance_adapter import build_p5_replay_governance_tile

        tile = build_p5_replay_governance_tile(
            p5_signal_ok, radar_ok, promotion_eval_ok, include_p5_extensions=True
        )

        assert "telemetry_source" in tile
        assert "production_run_id" in tile
        assert "replay_latency_ms" in tile
        assert tile["telemetry_source"] == "real"
        assert tile["production_run_id"] == "prod_run_001"

    def test_build_p5_tile_excludes_extensions(
        self,
        p5_signal_ok: Dict[str, Any],
        radar_ok: Dict[str, Any],
        promotion_eval_ok: Dict[str, Any],
    ) -> None:
        """P5 tile excludes extension fields when disabled."""
        from backend.health.replay_governance_adapter import build_p5_replay_governance_tile

        tile = build_p5_replay_governance_tile(
            p5_signal_ok, radar_ok, promotion_eval_ok, include_p5_extensions=False
        )

        assert "telemetry_source" not in tile
        assert "production_run_id" not in tile
        assert "replay_latency_ms" not in tile

    def test_build_p5_tile_is_json_serializable(
        self,
        p5_signal_ok: Dict[str, Any],
        radar_ok: Dict[str, Any],
        promotion_eval_ok: Dict[str, Any],
    ) -> None:
        """P5 tile must be JSON serializable."""
        from backend.health.replay_governance_adapter import build_p5_replay_governance_tile

        tile = build_p5_replay_governance_tile(p5_signal_ok, radar_ok, promotion_eval_ok)

        json_str = json.dumps(tile)
        assert json_str is not None
        assert len(json_str) > 0

    def test_build_p5_tile_has_shadow_mode_contract(
        self,
        p5_signal_ok: Dict[str, Any],
        radar_ok: Dict[str, Any],
        promotion_eval_ok: Dict[str, Any],
    ) -> None:
        """P5 tile must have shadow mode contract."""
        from backend.health.replay_governance_adapter import build_p5_replay_governance_tile

        tile = build_p5_replay_governance_tile(p5_signal_ok, radar_ok, promotion_eval_ok)

        assert "shadow_mode_contract" in tile
        contract = tile["shadow_mode_contract"]
        assert contract["observational_only"] is True
        assert contract["no_control_flow_influence"] is True
        assert contract["no_governance_modification"] is True

    def test_build_p5_tile_uses_more_severe_status(
        self,
        p5_signal_block: Dict[str, Any],
        radar_ok: Dict[str, Any],
        promotion_eval_ok: Dict[str, Any],
    ) -> None:
        """P5 tile uses more severe status between signal and base."""
        from backend.health.replay_governance_adapter import build_p5_replay_governance_tile

        # P5 signal is BLOCK, radar is OK - should use BLOCK
        tile = build_p5_replay_governance_tile(p5_signal_block, radar_ok, promotion_eval_ok)

        assert tile["status"] == "block"


# =============================================================================
# TEST 12: P5 Evidence Attachment (P5-C Unit Tests)
# =============================================================================

class TestP5EvidenceAttachment:
    """Test P5 replay governance evidence attachment."""

    @pytest.fixture
    def p5_signal(self) -> Dict[str, Any]:
        """P5 signal for evidence attachment."""
        return {
            "schema_version": "1.0.0",
            "status": "ok",
            "determinism_rate": 1.0,
            "determinism_band": "GREEN",
            "hash_match_count": 10,
            "hash_mismatch_count": 0,
            "critical_incidents": [],
            "telemetry_source": "real",
            "production_run_id": "prod_run_001",
            "replay_latency_ms": 50.0,
            "p5_grade": True,
            "reasons": [],
        }

    @pytest.fixture
    def p5_tile(self) -> Dict[str, Any]:
        """P5 tile for evidence attachment."""
        return {
            "schema_version": "1.0.0",
            "tile_type": "replay_governance",
            "phase": "P5",
            "status": "ok",
            "safe": True,
            "p5_grade": True,
            "determinism_band": "GREEN",
            "governance_alignment": "aligned",
            "governance_signal": {
                "status": "ok",
                "conflict": False,
            },
            "summary_metrics": {
                "replay_ok_for_promotion": True,
            },
        }

    def test_attach_p5_creates_governance_keys(
        self,
        evidence_base: Dict[str, Any],
        p5_signal: Dict[str, Any],
        p5_tile: Dict[str, Any],
    ) -> None:
        """P5 attachment creates governance.replay and governance.replay_p5."""
        from backend.health.replay_governance_adapter import attach_p5_replay_governance_to_evidence

        result = attach_p5_replay_governance_to_evidence(evidence_base, p5_signal, p5_tile)

        assert "governance" in result
        assert "replay" in result["governance"]
        assert "replay_p5" in result["governance"]

    def test_attach_p5_sets_p5_grade_flag(
        self,
        evidence_base: Dict[str, Any],
        p5_signal: Dict[str, Any],
        p5_tile: Dict[str, Any],
    ) -> None:
        """P5 attachment sets replay_p5_grade flag."""
        from backend.health.replay_governance_adapter import attach_p5_replay_governance_to_evidence

        result = attach_p5_replay_governance_to_evidence(evidence_base, p5_signal, p5_tile)

        assert result["replay_p5_grade"] is True
        assert result["replay_safety_ok"] is True

    def test_attach_p5_is_non_mutating(
        self,
        evidence_base: Dict[str, Any],
        p5_signal: Dict[str, Any],
        p5_tile: Dict[str, Any],
    ) -> None:
        """P5 attachment must not mutate original evidence."""
        from backend.health.replay_governance_adapter import attach_p5_replay_governance_to_evidence

        original = copy.deepcopy(evidence_base)

        result = attach_p5_replay_governance_to_evidence(evidence_base, p5_signal, p5_tile)

        assert evidence_base == original
        assert result is not evidence_base

    def test_attach_p5_validates_grade_by_default(
        self,
        evidence_base: Dict[str, Any],
        p5_tile: Dict[str, Any],
    ) -> None:
        """P5 attachment validates P5 grade by default."""
        from backend.health.replay_governance_adapter import attach_p5_replay_governance_to_evidence

        invalid_signal = {
            "telemetry_source": "synthetic",  # Not "real"
            "production_run_id": "prod_run_001",
        }

        with pytest.raises(ValueError, match="Signal is not P5-grade"):
            attach_p5_replay_governance_to_evidence(evidence_base, invalid_signal, p5_tile)

    def test_attach_p5_skips_validation_when_disabled(
        self,
        evidence_base: Dict[str, Any],
        p5_tile: Dict[str, Any],
    ) -> None:
        """P5 attachment skips validation when disabled."""
        from backend.health.replay_governance_adapter import attach_p5_replay_governance_to_evidence

        invalid_signal = {
            "telemetry_source": "synthetic",
            "production_run_id": "prod_run_001",
            "determinism_band": "GREEN",
            "determinism_rate": 1.0,
            "hash_match_count": 0,
            "hash_mismatch_count": 0,
            "replay_latency_ms": None,
            "reasons": [],
        }

        # Should not raise when validation disabled
        result = attach_p5_replay_governance_to_evidence(
            evidence_base, invalid_signal, p5_tile, validate_p5_grade=False
        )

        assert result["replay_p5_grade"] is False

    def test_attach_p5_includes_extension_fields(
        self,
        evidence_base: Dict[str, Any],
        p5_signal: Dict[str, Any],
        p5_tile: Dict[str, Any],
    ) -> None:
        """P5 attachment includes extension fields in replay_p5."""
        from backend.health.replay_governance_adapter import attach_p5_replay_governance_to_evidence

        result = attach_p5_replay_governance_to_evidence(evidence_base, p5_signal, p5_tile)

        p5_ext = result["governance"]["replay_p5"]
        assert "determinism_band" in p5_ext
        assert "determinism_rate" in p5_ext
        assert "hash_match_count" in p5_ext
        assert "hash_mismatch_count" in p5_ext
        assert "production_run_id" in p5_ext
        assert "telemetry_source" in p5_ext

    def test_attach_p5_is_json_serializable(
        self,
        evidence_base: Dict[str, Any],
        p5_signal: Dict[str, Any],
        p5_tile: Dict[str, Any],
    ) -> None:
        """P5 attachment must produce JSON-serializable output."""
        from backend.health.replay_governance_adapter import attach_p5_replay_governance_to_evidence

        result = attach_p5_replay_governance_to_evidence(evidence_base, p5_signal, p5_tile)

        json_str = json.dumps(result)
        assert json_str is not None
        assert len(json_str) > 0


# =============================================================================
# TEST 13: P5 GGFL Fusion Adapter (P5-C Unit Tests)
# =============================================================================

class TestP5GGFLFusionAdapter:
    """Test P5 GGFL fusion adapter replay_for_alignment_view_p5()."""

    @pytest.fixture
    def p5_signal_ok(self) -> Dict[str, Any]:
        """P5 signal with OK status for GGFL."""
        return {
            "status": "ok",
            "determinism_band": "GREEN",
            "p5_grade": True,
            "telemetry_source": "real",
            "reasons": [],
        }

    @pytest.fixture
    def p5_signal_warn(self) -> Dict[str, Any]:
        """P5 signal with WARN status for GGFL."""
        return {
            "status": "warn",
            "determinism_band": "YELLOW",
            "p5_grade": True,
            "telemetry_source": "real",
            "reasons": ["[Replay] Determinism rate 80.00% in YELLOW band"],
        }

    @pytest.fixture
    def p5_signal_block(self) -> Dict[str, Any]:
        """P5 signal with BLOCK status for GGFL."""
        return {
            "status": "block",
            "determinism_band": "RED",
            "p5_grade": True,
            "telemetry_source": "real",
            "reasons": ["[Replay] Determinism rate 50.00% in RED band", "[Replay] Hash mismatch at cycle_001"],
        }

    def test_p5_ggfl_produces_correct_shape(
        self, p5_signal_ok: Dict[str, Any]
    ) -> None:
        """P5 GGFL adapter produces correct shape."""
        from backend.health.replay_governance_adapter import replay_for_alignment_view_p5

        result = replay_for_alignment_view_p5(p5_signal_ok)

        assert "status" in result
        assert "alignment" in result
        assert "conflict" in result
        assert "top_reasons" in result
        assert "p5_grade" in result
        assert "determinism_band" in result
        assert "telemetry_source" in result

    def test_p5_ggfl_ok_aligned_no_conflict(
        self, p5_signal_ok: Dict[str, Any]
    ) -> None:
        """P5 GGFL OK status maps to aligned, no conflict."""
        from backend.health.replay_governance_adapter import replay_for_alignment_view_p5

        result = replay_for_alignment_view_p5(p5_signal_ok)

        assert result["status"] == "ok"
        assert result["alignment"] == "aligned"
        assert result["conflict"] is False

    def test_p5_ggfl_warn_tension(
        self, p5_signal_warn: Dict[str, Any]
    ) -> None:
        """P5 GGFL WARN status maps to tension."""
        from backend.health.replay_governance_adapter import replay_for_alignment_view_p5

        result = replay_for_alignment_view_p5(p5_signal_warn)

        assert result["status"] == "warn"
        assert result["alignment"] == "tension"
        assert result["conflict"] is False

    def test_p5_ggfl_block_divergent_conflict(
        self, p5_signal_block: Dict[str, Any]
    ) -> None:
        """P5 GGFL BLOCK status maps to divergent with conflict."""
        from backend.health.replay_governance_adapter import replay_for_alignment_view_p5

        result = replay_for_alignment_view_p5(p5_signal_block)

        assert result["status"] == "block"
        assert result["alignment"] == "divergent"
        assert result["conflict"] is True

    def test_p5_ggfl_strips_replay_prefix(
        self, p5_signal_block: Dict[str, Any]
    ) -> None:
        """P5 GGFL adapter strips [Replay] prefix from reasons."""
        from backend.health.replay_governance_adapter import replay_for_alignment_view_p5

        result = replay_for_alignment_view_p5(p5_signal_block)

        for reason in result["top_reasons"]:
            assert not reason.startswith("[Replay]"), f"Prefix not stripped: {reason}"

    def test_p5_ggfl_limits_reasons(self) -> None:
        """P5 GGFL adapter limits reasons to 5."""
        from backend.health.replay_governance_adapter import replay_for_alignment_view_p5

        signal_many_reasons = {
            "status": "block",
            "determinism_band": "RED",
            "p5_grade": True,
            "telemetry_source": "real",
            "reasons": [f"[Replay] Reason {i}" for i in range(10)],
        }

        result = replay_for_alignment_view_p5(signal_many_reasons)

        assert len(result["top_reasons"]) == 5

    def test_p5_ggfl_includes_p5_fields(
        self, p5_signal_ok: Dict[str, Any]
    ) -> None:
        """P5 GGFL adapter includes P5-specific fields."""
        from backend.health.replay_governance_adapter import replay_for_alignment_view_p5

        result = replay_for_alignment_view_p5(p5_signal_ok)

        assert result["p5_grade"] is True
        assert result["determinism_band"] == "GREEN"
        assert result["telemetry_source"] == "real"

    def test_p5_ggfl_is_read_only(
        self, p5_signal_ok: Dict[str, Any]
    ) -> None:
        """P5 GGFL adapter is read-only."""
        from backend.health.replay_governance_adapter import replay_for_alignment_view_p5

        original = copy.deepcopy(p5_signal_ok)

        _ = replay_for_alignment_view_p5(p5_signal_ok)

        assert p5_signal_ok == original
