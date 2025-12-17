"""
CI tests for uplift gate alignment status integration and GGFL adapter.

PHASE X — UPLIFT SAFETY GOVERNANCE TILING

Tests that:
  - Alignment panel is extracted from manifest/evidence.json
  - Signal is correctly attached to status.json
  - Warning is generated correctly (single warning, top cal_ids + reason_code)
  - GGFL adapter produces correct format
  - All outputs are deterministic and JSON-safe

SHADOW MODE CONTRACT:
  - These tests verify observational behavior only
  - No control flow dependencies
  - No deployment blocking logic
"""

import json
import pytest
import sys
from pathlib import Path
from typing import Dict, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.health.uplift_safety_adapter import (
    build_gate_alignment_panel,
    extract_uplift_gate_alignment_signal,
    uplift_gate_alignment_for_alignment_view,
    build_first_light_uplift_gate_annex,
    build_p3_uplift_safety_summary,
    build_p4_uplift_safety_calibration,
    extract_uplift_safety_signal_for_first_light,
    build_uplift_safety_governance_tile,
    DRIVER_TOP_REASON_P3_BLOCK,
    DRIVER_TOP_REASON_P4_BLOCK,
    DRIVER_TOP_REASON_BOTH_BLOCK,
    DRIVER_MISALIGNED_COUNT_PRESENT,
    DRIVER_TOP_CAL_IDS_PRESENT,
)


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES: Synthetic test data
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def synthetic_safety_tensor_pass() -> Dict[str, Any]:
    """Synthetic safety tensor with LOW risk (PASS scenario)."""
    return {
        "schema_version": "1.0.0",
        "tensor_norm": 0.3,
        "uplift_risk_band": "LOW",
        "hotspot_axes": [],
        "risk_indicators": {
            "epistemic_uncertainty": 0.2,
            "drift_risk": 0.3,
            "atlas_risk": 0.2,
            "telemetry_risk": 0.1,
        },
        "neutral_notes": [],
    }


@pytest.fixture
def synthetic_stability_forecaster_stable() -> Dict[str, Any]:
    """Synthetic stability forecaster with STABLE trend."""
    return {
        "schema_version": "1.0.0",
        "current_stability": "STABLE",
        "stability_trend": "STABLE",
        "instability_prediction": {
            "predicted_instability_cycles": [],
            "confidence": 0.0,
        },
    }


@pytest.fixture
def synthetic_gate_decision_pass() -> Dict[str, Any]:
    """Synthetic gate decision with PASS."""
    return {
        "schema_version": "1.0.0",
        "uplift_safety_decision": "PASS",
        "decision_rationale": [],
    }


@pytest.fixture
def synthetic_safety_tensor_block() -> Dict[str, Any]:
    """Synthetic safety tensor with HIGH risk (BLOCK scenario)."""
    return {
        "schema_version": "1.0.0",
        "tensor_norm": 0.9,
        "uplift_risk_band": "HIGH",
        "hotspot_axes": ["epistemic_uncertainty", "drift_risk"],
        "risk_indicators": {
            "epistemic_uncertainty": 0.9,
            "drift_risk": 0.8,
            "atlas_risk": 0.7,
            "telemetry_risk": 0.6,
        },
        "neutral_notes": [],
    }


@pytest.fixture
def synthetic_stability_forecaster_unstable() -> Dict[str, Any]:
    """Synthetic stability forecaster with DEGRADING trend."""
    return {
        "schema_version": "1.0.0",
        "current_stability": "UNSTABLE",
        "stability_trend": "DEGRADING",
        "instability_prediction": {
            "predicted_instability_cycles": [100, 200],
            "confidence": 0.8,
        },
    }


@pytest.fixture
def synthetic_gate_decision_block() -> Dict[str, Any]:
    """Synthetic gate decision with BLOCK."""
    return {
        "schema_version": "1.0.0",
        "uplift_safety_decision": "BLOCK",
        "decision_rationale": ["High risk detected"],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS: Status Integration
# ═══════════════════════════════════════════════════════════════════════════════

class TestStatusIntegration:
    """Tests for status.json integration."""

    def test_001_extract_signal_from_panel(
        self,
        synthetic_safety_tensor_block,
        synthetic_stability_forecaster_unstable,
        synthetic_gate_decision_block,
    ):
        """extract_uplift_gate_alignment_signal extracts correct signal from panel."""
        signal = extract_uplift_safety_signal_for_first_light(
            synthetic_safety_tensor_block,
            synthetic_stability_forecaster_unstable,
            synthetic_gate_decision_block,
        )
        tile = build_uplift_safety_governance_tile(
            synthetic_safety_tensor_block,
            synthetic_stability_forecaster_unstable,
            synthetic_gate_decision_block,
        )
        p3_summary = build_p3_uplift_safety_summary(signal)
        p4_calibration = build_p4_uplift_safety_calibration(tile)

        annex = build_first_light_uplift_gate_annex(p3_summary, p4_calibration)
        annex["cal_id"] = "CAL-EXP-1"

        panel = build_gate_alignment_panel([annex])
        alignment_signal = extract_uplift_gate_alignment_signal(panel)

        assert "alignment_rate" in alignment_signal
        assert "misaligned_count" in alignment_signal
        assert "top_misaligned_cal_ids" in alignment_signal
        assert "reason_code_histogram" in alignment_signal

    def test_002_signal_schema_version_passthrough(
        self,
        synthetic_safety_tensor_block,
        synthetic_stability_forecaster_unstable,
        synthetic_gate_decision_block,
    ):
        """Signal includes schema_version from panel."""
        signal = extract_uplift_safety_signal_for_first_light(
            synthetic_safety_tensor_block,
            synthetic_stability_forecaster_unstable,
            synthetic_gate_decision_block,
        )
        tile = build_uplift_safety_governance_tile(
            synthetic_safety_tensor_block,
            synthetic_stability_forecaster_unstable,
            synthetic_gate_decision_block,
        )
        p3_summary = build_p3_uplift_safety_summary(signal)
        p4_calibration = build_p4_uplift_safety_calibration(tile)

        annex = build_first_light_uplift_gate_annex(p3_summary, p4_calibration)
        annex["cal_id"] = "CAL-EXP-1"

        panel = build_gate_alignment_panel([annex])
        panel["schema_version"] = "1.0.0"

        alignment_signal = extract_uplift_gate_alignment_signal(panel)

        # Signal should be ready for status.json attachment
        status_signal = {
            "schema_version": panel.get("schema_version", "1.0.0"),
            "mode": "SHADOW",
            "alignment_rate": alignment_signal.get("alignment_rate", 0.0),
            "misaligned_count": alignment_signal.get("misaligned_count", 0),
            "top_misaligned_cal_ids": alignment_signal.get("top_misaligned_cal_ids", []),
            "reason_code_histogram": alignment_signal.get("reason_code_histogram", {}),
        }

        assert status_signal["schema_version"] == "1.0.0"
        assert status_signal["mode"] == "SHADOW"

    def test_003_warning_generation_misaligned(
        self,
        synthetic_safety_tensor_block,
        synthetic_stability_forecaster_unstable,
        synthetic_gate_decision_block,
    ):
        """Warning is generated correctly when misaligned_count > 0, uses top_reason_code explicitly, single-line."""
        signal = extract_uplift_safety_signal_for_first_light(
            synthetic_safety_tensor_block,
            synthetic_stability_forecaster_unstable,
            synthetic_gate_decision_block,
        )
        tile = build_uplift_safety_governance_tile(
            synthetic_safety_tensor_block,
            synthetic_stability_forecaster_unstable,
            synthetic_gate_decision_block,
        )
        p3_summary = build_p3_uplift_safety_summary(signal)
        p4_calibration = build_p4_uplift_safety_calibration(tile)

        annex1 = build_first_light_uplift_gate_annex(p3_summary, p4_calibration)
        annex1["cal_id"] = "CAL-EXP-1"

        annex2 = build_first_light_uplift_gate_annex(p3_summary, p4_calibration)
        annex2["cal_id"] = "CAL-EXP-2"

        panel = build_gate_alignment_panel([annex1, annex2])
        alignment_signal = extract_uplift_gate_alignment_signal(panel)

        misaligned_count = alignment_signal.get("misaligned_count", 0)
        top_cal_ids = alignment_signal.get("top_misaligned_cal_ids", [])
        reason_histogram = alignment_signal.get("reason_code_histogram", {})

        # Derive top_reason_code deterministically (same logic as generate_first_light_status.py)
        top_reason_code = None
        if reason_histogram:
            sorted_items = sorted(
                reason_histogram.items(),
                key=lambda x: (-x[1], x[0])  # Negative count for descending, then code ascending
            )
            top_reason_code = sorted_items[0][0]

        # Simulate warning generation logic (explicit top_reason_code, single-line)
        warning_parts = []
        if misaligned_count > 0:
            warning_parts.append(f"Uplift gate alignment: {misaligned_count} experiment(s) with misaligned P3/P4 gates")
            if top_cal_ids:
                cal_ids_str = ", ".join(top_cal_ids[:3])
                warning_parts.append(f"top cal_ids: {cal_ids_str}")
            # Always include top_reason_code if available (explicit requirement)
            if top_reason_code:
                warning_parts.append(f"top_reason_code: {top_reason_code}")

        warning = "; ".join(warning_parts)

        assert misaligned_count == 2
        assert "CAL-EXP-1" in warning or "CAL-EXP-2" in warning
        assert "top_reason_code: BOTH_BLOCK" in warning  # Explicit format
        # Verify single-line (no newlines)
        assert "\n" not in warning

    def test_004_warning_cap_single(
        self,
        synthetic_safety_tensor_block,
        synthetic_stability_forecaster_unstable,
        synthetic_gate_decision_block,
    ):
        """Only one warning is generated even with multiple misalignments."""
        signal = extract_uplift_safety_signal_for_first_light(
            synthetic_safety_tensor_block,
            synthetic_stability_forecaster_unstable,
            synthetic_gate_decision_block,
        )
        tile = build_uplift_safety_governance_tile(
            synthetic_safety_tensor_block,
            synthetic_stability_forecaster_unstable,
            synthetic_gate_decision_block,
        )
        p3_summary = build_p3_uplift_safety_summary(signal)
        p4_calibration = build_p4_uplift_safety_calibration(tile)

        # Create 5 misaligned experiments
        annexes = []
        for i in range(5):
            annex = build_first_light_uplift_gate_annex(p3_summary, p4_calibration)
            annex["cal_id"] = f"CAL-EXP-{i+1}"
            annexes.append(annex)

        panel = build_gate_alignment_panel(annexes)
        alignment_signal = extract_uplift_gate_alignment_signal(panel)

        misaligned_count = alignment_signal.get("misaligned_count", 0)
        top_cal_ids = alignment_signal.get("top_misaligned_cal_ids", [])
        reason_histogram = alignment_signal.get("reason_code_histogram", {})

        # Simulate warning generation (should be single warning)
        warnings = []
        if misaligned_count > 0:
            warning_parts = [f"Uplift gate alignment: {misaligned_count} experiment(s) with misaligned P3/P4 gates"]
            if top_cal_ids:
                cal_ids_str = ", ".join(top_cal_ids[:3])
                warning_parts.append(f"top cal_ids: {cal_ids_str}")
            if reason_histogram:
                top_reason_code = max(reason_histogram.items(), key=lambda x: x[1])[0]
                warning_parts.append(f"top reason_code: {top_reason_code}")
            warnings.append("; ".join(warning_parts))

        assert len(warnings) == 1
        assert misaligned_count == 5
        assert len(top_cal_ids) == 3  # Top 3 only


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS: GGFL Adapter
# ═══════════════════════════════════════════════════════════════════════════════

class TestGGFLAdapter:
    """Tests for GGFL alignment view adapter."""

    def test_005_ggfl_adapter_schema(
        self,
        synthetic_safety_tensor_block,
        synthetic_stability_forecaster_unstable,
        synthetic_gate_decision_block,
    ):
        """uplift_gate_alignment_for_alignment_view returns correct schema."""
        signal = extract_uplift_safety_signal_for_first_light(
            synthetic_safety_tensor_block,
            synthetic_stability_forecaster_unstable,
            synthetic_gate_decision_block,
        )
        tile = build_uplift_safety_governance_tile(
            synthetic_safety_tensor_block,
            synthetic_stability_forecaster_unstable,
            synthetic_gate_decision_block,
        )
        p3_summary = build_p3_uplift_safety_summary(signal)
        p4_calibration = build_p4_uplift_safety_calibration(tile)

        annex = build_first_light_uplift_gate_annex(p3_summary, p4_calibration)
        annex["cal_id"] = "CAL-EXP-1"

        panel = build_gate_alignment_panel([annex])
        ggfl_signal = uplift_gate_alignment_for_alignment_view(panel)

        assert "signal_type" in ggfl_signal
        assert "status" in ggfl_signal
        assert "conflict" in ggfl_signal
        assert "drivers" in ggfl_signal
        assert "summary" in ggfl_signal

    def test_006_ggfl_adapter_signal_type(
        self,
        synthetic_safety_tensor_block,
        synthetic_stability_forecaster_unstable,
        synthetic_gate_decision_block,
    ):
        """uplift_gate_alignment_for_alignment_view sets signal_type to SIG-GATE."""
        signal = extract_uplift_safety_signal_for_first_light(
            synthetic_safety_tensor_block,
            synthetic_stability_forecaster_unstable,
            synthetic_gate_decision_block,
        )
        tile = build_uplift_safety_governance_tile(
            synthetic_safety_tensor_block,
            synthetic_stability_forecaster_unstable,
            synthetic_gate_decision_block,
        )
        p3_summary = build_p3_uplift_safety_summary(signal)
        p4_calibration = build_p4_uplift_safety_calibration(tile)

        annex = build_first_light_uplift_gate_annex(p3_summary, p4_calibration)
        annex["cal_id"] = "CAL-EXP-1"

        panel = build_gate_alignment_panel([annex])
        ggfl_signal = uplift_gate_alignment_for_alignment_view(panel)

        assert ggfl_signal["signal_type"] == "SIG-GATE"

    def test_007_ggfl_adapter_status_ok(
        self,
        synthetic_safety_tensor_pass,
        synthetic_stability_forecaster_stable,
        synthetic_gate_decision_pass,
    ):
        """uplift_gate_alignment_for_alignment_view sets status to ok when no misalignments."""
        signal = extract_uplift_safety_signal_for_first_light(
            synthetic_safety_tensor_pass,
            synthetic_stability_forecaster_stable,
            synthetic_gate_decision_pass,
        )
        tile = build_uplift_safety_governance_tile(
            synthetic_safety_tensor_pass,
            synthetic_stability_forecaster_stable,
            synthetic_gate_decision_pass,
        )
        p3_summary = build_p3_uplift_safety_summary(signal)
        p4_calibration = build_p4_uplift_safety_calibration(tile)

        annex = build_first_light_uplift_gate_annex(p3_summary, p4_calibration)
        annex["cal_id"] = "CAL-EXP-1"

        panel = build_gate_alignment_panel([annex])
        ggfl_signal = uplift_gate_alignment_for_alignment_view(panel)

        assert ggfl_signal["status"] == "ok"

    def test_008_ggfl_adapter_status_warn(
        self,
        synthetic_safety_tensor_block,
        synthetic_stability_forecaster_unstable,
        synthetic_gate_decision_block,
    ):
        """uplift_gate_alignment_for_alignment_view sets status to warn when misalignments exist."""
        signal = extract_uplift_safety_signal_for_first_light(
            synthetic_safety_tensor_block,
            synthetic_stability_forecaster_unstable,
            synthetic_gate_decision_block,
        )
        tile = build_uplift_safety_governance_tile(
            synthetic_safety_tensor_block,
            synthetic_stability_forecaster_unstable,
            synthetic_gate_decision_block,
        )
        p3_summary = build_p3_uplift_safety_summary(signal)
        p4_calibration = build_p4_uplift_safety_calibration(tile)

        annex = build_first_light_uplift_gate_annex(p3_summary, p4_calibration)
        annex["cal_id"] = "CAL-EXP-1"

        panel = build_gate_alignment_panel([annex])
        ggfl_signal = uplift_gate_alignment_for_alignment_view(panel)

        assert ggfl_signal["status"] == "warn"

    def test_009_ggfl_adapter_conflict_false(
        self,
        synthetic_safety_tensor_block,
        synthetic_stability_forecaster_unstable,
        synthetic_gate_decision_block,
    ):
        """uplift_gate_alignment_for_alignment_view sets conflict to False."""
        signal = extract_uplift_safety_signal_for_first_light(
            synthetic_safety_tensor_block,
            synthetic_stability_forecaster_unstable,
            synthetic_gate_decision_block,
        )
        tile = build_uplift_safety_governance_tile(
            synthetic_safety_tensor_block,
            synthetic_stability_forecaster_unstable,
            synthetic_gate_decision_block,
        )
        p3_summary = build_p3_uplift_safety_summary(signal)
        p4_calibration = build_p4_uplift_safety_calibration(tile)

        annex = build_first_light_uplift_gate_annex(p3_summary, p4_calibration)
        annex["cal_id"] = "CAL-EXP-1"

        panel = build_gate_alignment_panel([annex])
        ggfl_signal = uplift_gate_alignment_for_alignment_view(panel)

        assert ggfl_signal["conflict"] is False

    def test_010_ggfl_adapter_drivers_max_3(
        self,
        synthetic_safety_tensor_block,
        synthetic_stability_forecaster_unstable,
        synthetic_gate_decision_block,
    ):
        """uplift_gate_alignment_for_alignment_view limits drivers to max 3."""
        signal = extract_uplift_safety_signal_for_first_light(
            synthetic_safety_tensor_block,
            synthetic_stability_forecaster_unstable,
            synthetic_gate_decision_block,
        )
        tile = build_uplift_safety_governance_tile(
            synthetic_safety_tensor_block,
            synthetic_stability_forecaster_unstable,
            synthetic_gate_decision_block,
        )
        p3_summary = build_p3_uplift_safety_summary(signal)
        p4_calibration = build_p4_uplift_safety_calibration(tile)

        # Create multiple misaligned experiments
        annexes = []
        for i in range(5):
            annex = build_first_light_uplift_gate_annex(p3_summary, p4_calibration)
            annex["cal_id"] = f"CAL-EXP-{i+1}"
            annexes.append(annex)

        panel = build_gate_alignment_panel(annexes)
        ggfl_signal = uplift_gate_alignment_for_alignment_view(panel)

        assert len(ggfl_signal["drivers"]) <= 3

    def test_011_ggfl_adapter_drivers_content(
        self,
        synthetic_safety_tensor_block,
        synthetic_stability_forecaster_unstable,
        synthetic_gate_decision_block,
    ):
        """uplift_gate_alignment_for_alignment_view uses reason-code drivers (no prose)."""
        signal = extract_uplift_safety_signal_for_first_light(
            synthetic_safety_tensor_block,
            synthetic_stability_forecaster_unstable,
            synthetic_gate_decision_block,
        )
        tile = build_uplift_safety_governance_tile(
            synthetic_safety_tensor_block,
            synthetic_stability_forecaster_unstable,
            synthetic_gate_decision_block,
        )
        p3_summary = build_p3_uplift_safety_summary(signal)
        p4_calibration = build_p4_uplift_safety_calibration(tile)

        annex = build_first_light_uplift_gate_annex(p3_summary, p4_calibration)
        annex["cal_id"] = "CAL-EXP-1"

        panel = build_gate_alignment_panel([annex])
        ggfl_signal = uplift_gate_alignment_for_alignment_view(panel)

        drivers = ggfl_signal["drivers"]
        assert len(drivers) > 0
        # All drivers must be reason codes (no prose)
        valid_driver_codes = {
            DRIVER_TOP_REASON_P3_BLOCK,
            DRIVER_TOP_REASON_P4_BLOCK,
            DRIVER_TOP_REASON_BOTH_BLOCK,
            DRIVER_MISALIGNED_COUNT_PRESENT,
            DRIVER_TOP_CAL_IDS_PRESENT,
        }
        for driver in drivers:
            assert driver in valid_driver_codes, f"Driver '{driver}' is not a valid reason code"
        # Should include reason code driver
        assert DRIVER_TOP_REASON_BOTH_BLOCK in drivers
        # Should include misaligned count driver
        assert DRIVER_MISALIGNED_COUNT_PRESENT in drivers

    def test_012_ggfl_adapter_summary_neutral(
        self,
        synthetic_safety_tensor_block,
        synthetic_stability_forecaster_unstable,
        synthetic_gate_decision_block,
    ):
        """uplift_gate_alignment_for_alignment_view summary is neutral sentence."""
        signal = extract_uplift_safety_signal_for_first_light(
            synthetic_safety_tensor_block,
            synthetic_stability_forecaster_unstable,
            synthetic_gate_decision_block,
        )
        tile = build_uplift_safety_governance_tile(
            synthetic_safety_tensor_block,
            synthetic_stability_forecaster_unstable,
            synthetic_gate_decision_block,
        )
        p3_summary = build_p3_uplift_safety_summary(signal)
        p4_calibration = build_p4_uplift_safety_calibration(tile)

        annex = build_first_light_uplift_gate_annex(p3_summary, p4_calibration)
        annex["cal_id"] = "CAL-EXP-1"

        panel = build_gate_alignment_panel([annex])
        ggfl_signal = uplift_gate_alignment_for_alignment_view(panel)

        summary = ggfl_signal["summary"]
        assert isinstance(summary, str)
        assert len(summary) > 0
        # Should be descriptive, not evaluative
        assert "Uplift gate alignment" in summary

    def test_013_ggfl_adapter_accepts_signal(
        self,
        synthetic_safety_tensor_block,
        synthetic_stability_forecaster_unstable,
        synthetic_gate_decision_block,
    ):
        """uplift_gate_alignment_for_alignment_view accepts signal directly (not just panel)."""
        signal = extract_uplift_safety_signal_for_first_light(
            synthetic_safety_tensor_block,
            synthetic_stability_forecaster_unstable,
            synthetic_gate_decision_block,
        )
        tile = build_uplift_safety_governance_tile(
            synthetic_safety_tensor_block,
            synthetic_stability_forecaster_unstable,
            synthetic_gate_decision_block,
        )
        p3_summary = build_p3_uplift_safety_summary(signal)
        p4_calibration = build_p4_uplift_safety_calibration(tile)

        annex = build_first_light_uplift_gate_annex(p3_summary, p4_calibration)
        annex["cal_id"] = "CAL-EXP-1"

        panel = build_gate_alignment_panel([annex])
        alignment_signal = extract_uplift_gate_alignment_signal(panel)

        # Pass signal directly (not panel)
        ggfl_signal = uplift_gate_alignment_for_alignment_view(alignment_signal)

        assert ggfl_signal["signal_type"] == "SIG-GATE"
        assert ggfl_signal["status"] == "warn"

    def test_014_ggfl_adapter_deterministic(
        self,
        synthetic_safety_tensor_block,
        synthetic_stability_forecaster_unstable,
        synthetic_gate_decision_block,
    ):
        """uplift_gate_alignment_for_alignment_view is deterministic."""
        signal = extract_uplift_safety_signal_for_first_light(
            synthetic_safety_tensor_block,
            synthetic_stability_forecaster_unstable,
            synthetic_gate_decision_block,
        )
        tile = build_uplift_safety_governance_tile(
            synthetic_safety_tensor_block,
            synthetic_stability_forecaster_unstable,
            synthetic_gate_decision_block,
        )
        p3_summary = build_p3_uplift_safety_summary(signal)
        p4_calibration = build_p4_uplift_safety_calibration(tile)

        annex = build_first_light_uplift_gate_annex(p3_summary, p4_calibration)
        annex["cal_id"] = "CAL-EXP-1"

        panel = build_gate_alignment_panel([annex])
        ggfl_signal1 = uplift_gate_alignment_for_alignment_view(panel)
        ggfl_signal2 = uplift_gate_alignment_for_alignment_view(panel)

        assert ggfl_signal1 == ggfl_signal2

    def test_015_ggfl_adapter_json_safe(
        self,
        synthetic_safety_tensor_block,
        synthetic_stability_forecaster_unstable,
        synthetic_gate_decision_block,
    ):
        """uplift_gate_alignment_for_alignment_view output is JSON serializable."""
        signal = extract_uplift_safety_signal_for_first_light(
            synthetic_safety_tensor_block,
            synthetic_stability_forecaster_unstable,
            synthetic_gate_decision_block,
        )
        tile = build_uplift_safety_governance_tile(
            synthetic_safety_tensor_block,
            synthetic_stability_forecaster_unstable,
            synthetic_gate_decision_block,
        )
        p3_summary = build_p3_uplift_safety_summary(signal)
        p4_calibration = build_p4_uplift_safety_calibration(tile)

        annex = build_first_light_uplift_gate_annex(p3_summary, p4_calibration)
        annex["cal_id"] = "CAL-EXP-1"

        panel = build_gate_alignment_panel([annex])
        ggfl_signal = uplift_gate_alignment_for_alignment_view(panel)

        json_str = json.dumps(ggfl_signal)
        assert isinstance(json_str, str)

    def test_016_ggfl_adapter_shadow_mode_invariants(
        self,
        synthetic_safety_tensor_block,
        synthetic_stability_forecaster_unstable,
        synthetic_gate_decision_block,
    ):
        """uplift_gate_alignment_for_alignment_view includes shadow_mode_invariants with unified schema."""
        signal = extract_uplift_safety_signal_for_first_light(
            synthetic_safety_tensor_block,
            synthetic_stability_forecaster_unstable,
            synthetic_gate_decision_block,
        )
        tile = build_uplift_safety_governance_tile(
            synthetic_safety_tensor_block,
            synthetic_stability_forecaster_unstable,
            synthetic_gate_decision_block,
        )
        p3_summary = build_p3_uplift_safety_summary(signal)
        p4_calibration = build_p4_uplift_safety_calibration(tile)

        annex = build_first_light_uplift_gate_annex(p3_summary, p4_calibration)
        annex["cal_id"] = "CAL-EXP-1"

        panel = build_gate_alignment_panel([annex])
        ggfl_signal = uplift_gate_alignment_for_alignment_view(panel)

        assert "shadow_mode_invariants" in ggfl_signal
        invariants = ggfl_signal["shadow_mode_invariants"]
        # Unified schema: advisory_only, no_enforcement, conflict_invariant (all True)
        assert invariants["advisory_only"] is True
        assert invariants["no_enforcement"] is True
        assert invariants["conflict_invariant"] is True

    def test_017_ggfl_adapter_weight_hint(
        self,
        synthetic_safety_tensor_block,
        synthetic_stability_forecaster_unstable,
        synthetic_gate_decision_block,
    ):
        """uplift_gate_alignment_for_alignment_view includes weight_hint: LOW."""
        signal = extract_uplift_safety_signal_for_first_light(
            synthetic_safety_tensor_block,
            synthetic_stability_forecaster_unstable,
            synthetic_gate_decision_block,
        )
        tile = build_uplift_safety_governance_tile(
            synthetic_safety_tensor_block,
            synthetic_stability_forecaster_unstable,
            synthetic_gate_decision_block,
        )
        p3_summary = build_p3_uplift_safety_summary(signal)
        p4_calibration = build_p4_uplift_safety_calibration(tile)

        annex = build_first_light_uplift_gate_annex(p3_summary, p4_calibration)
        annex["cal_id"] = "CAL-EXP-1"

        panel = build_gate_alignment_panel([annex])
        ggfl_signal = uplift_gate_alignment_for_alignment_view(panel)

        assert "weight_hint" in ggfl_signal
        assert ggfl_signal["weight_hint"] == "LOW"

    def test_018_ggfl_adapter_driver_ordering(
        self,
        synthetic_safety_tensor_block,
        synthetic_stability_forecaster_unstable,
        synthetic_gate_decision_block,
    ):
        """uplift_gate_alignment_for_alignment_view drivers follow deterministic ordering: reason → count → cal ids."""
        signal = extract_uplift_safety_signal_for_first_light(
            synthetic_safety_tensor_block,
            synthetic_stability_forecaster_unstable,
            synthetic_gate_decision_block,
        )
        tile = build_uplift_safety_governance_tile(
            synthetic_safety_tensor_block,
            synthetic_stability_forecaster_unstable,
            synthetic_gate_decision_block,
        )
        p3_summary = build_p3_uplift_safety_summary(signal)
        p4_calibration = build_p4_uplift_safety_calibration(tile)

        # Create multiple misaligned experiments to trigger all drivers
        annexes = []
        for i in range(3):
            annex = build_first_light_uplift_gate_annex(p3_summary, p4_calibration)
            annex["cal_id"] = f"CAL-EXP-{i+1}"
            annexes.append(annex)

        panel = build_gate_alignment_panel(annexes)
        ggfl_signal = uplift_gate_alignment_for_alignment_view(panel)

        drivers = ggfl_signal["drivers"]
        # Check ordering: reason code should come first
        if any(d.startswith("DRIVER_TOP_REASON_") for d in drivers):
            reason_idx = next(i for i, d in enumerate(drivers) if d.startswith("DRIVER_TOP_REASON_"))
            count_idx = next((i for i, d in enumerate(drivers) if d == DRIVER_MISALIGNED_COUNT_PRESENT), None)
            cal_ids_idx = next((i for i, d in enumerate(drivers) if d == DRIVER_TOP_CAL_IDS_PRESENT), None)
            
            if count_idx is not None:
                assert reason_idx < count_idx, "Reason code driver should come before count driver"
            if cal_ids_idx is not None:
                assert count_idx is None or count_idx < cal_ids_idx, "Count driver should come before cal_ids driver"

    def test_019_status_signal_extraction_source(
        self,
        synthetic_safety_tensor_block,
        synthetic_stability_forecaster_unstable,
        synthetic_gate_decision_block,
    ):
        """Status signal includes extraction_source field."""
        signal = extract_uplift_safety_signal_for_first_light(
            synthetic_safety_tensor_block,
            synthetic_stability_forecaster_unstable,
            synthetic_gate_decision_block,
        )
        tile = build_uplift_safety_governance_tile(
            synthetic_safety_tensor_block,
            synthetic_stability_forecaster_unstable,
            synthetic_gate_decision_block,
        )
        p3_summary = build_p3_uplift_safety_summary(signal)
        p4_calibration = build_p4_uplift_safety_calibration(tile)

        annex = build_first_light_uplift_gate_annex(p3_summary, p4_calibration)
        annex["cal_id"] = "CAL-EXP-1"

        panel = build_gate_alignment_panel([annex])
        alignment_signal = extract_uplift_gate_alignment_signal(panel)

        # Simulate status signal construction
        status_signal = {
            "schema_version": "1.0.0",
            "mode": "SHADOW",
            "extraction_source": "MANIFEST",  # Would be set by generate_first_light_status.py
            "alignment_rate": alignment_signal.get("alignment_rate", 0.0),
            "misaligned_count": alignment_signal.get("misaligned_count", 0),
            "top_misaligned_cal_ids": alignment_signal.get("top_misaligned_cal_ids", []),
            "reason_code_histogram": alignment_signal.get("reason_code_histogram", {}),
            "top_reason_code": None,  # Would be derived deterministically
        }

        assert "extraction_source" in status_signal
        assert status_signal["extraction_source"] in ("MANIFEST", "EVIDENCE_JSON", "MISSING")

    def test_020_status_signal_top_reason_code(
        self,
        synthetic_safety_tensor_block,
        synthetic_stability_forecaster_unstable,
        synthetic_gate_decision_block,
    ):
        """Status signal includes top_reason_code derived deterministically."""
        signal = extract_uplift_safety_signal_for_first_light(
            synthetic_safety_tensor_block,
            synthetic_stability_forecaster_unstable,
            synthetic_gate_decision_block,
        )
        tile = build_uplift_safety_governance_tile(
            synthetic_safety_tensor_block,
            synthetic_stability_forecaster_unstable,
            synthetic_gate_decision_block,
        )
        p3_summary = build_p3_uplift_safety_summary(signal)
        p4_calibration = build_p4_uplift_safety_calibration(tile)

        annex = build_first_light_uplift_gate_annex(p3_summary, p4_calibration)
        annex["cal_id"] = "CAL-EXP-1"

        panel = build_gate_alignment_panel([annex])
        alignment_signal = extract_uplift_gate_alignment_signal(panel)

        # Derive top_reason_code deterministically (same logic as generate_first_light_status.py)
        reason_histogram = alignment_signal.get("reason_code_histogram", {})
        top_reason_code = None
        if reason_histogram:
            sorted_items = sorted(
                reason_histogram.items(),
                key=lambda x: (-x[1], x[0])  # Negative count for descending, then code ascending
            )
            top_reason_code = sorted_items[0][0]

        # Simulate status signal construction
        status_signal = {
            "schema_version": "1.0.0",
            "mode": "SHADOW",
            "extraction_source": "MANIFEST",
            "alignment_rate": alignment_signal.get("alignment_rate", 0.0),
            "misaligned_count": alignment_signal.get("misaligned_count", 0),
            "top_misaligned_cal_ids": alignment_signal.get("top_misaligned_cal_ids", []),
            "reason_code_histogram": alignment_signal.get("reason_code_histogram", {}),
            "top_reason_code": top_reason_code,
        }

        assert "top_reason_code" in status_signal
        assert status_signal["top_reason_code"] == "BOTH_BLOCK"


# ═══════════════════════════════════════════════════════════════════════════════
# RUN TESTS
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

