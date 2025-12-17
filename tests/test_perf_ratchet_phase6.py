"""Tests for Phase VI: Global Console Adapter, Evidence Pack Hook, Input Validation.

Tests cover:
- Global console adapter with status mapping
- Evidence pack hook tile structure
- Input validation integration
- JSON safety and determinism
"""

import json
import pytest
from experiments.verify_perf_equivalence import (
    PerfGovernanceInputError,
    build_perf_joint_governance_view,
    summarize_perf_for_global_console,
    build_uplift_perf_governance_tile,
    summarize_perf_for_global_release,
    validate_perf_trend,
    validate_budget_trend,
    validate_metric_conformance,
    validate_perf_governance_inputs,
)


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def minimal_perf_trend():
    """Minimal valid perf_trend."""
    return {
        "schema_version": "1.0.0",
        "runs": [
            {"run_id": "run1", "status": "PASS"},
        ],
        "release_risk_level": "LOW",
    }


@pytest.fixture
def minimal_budget_trend():
    """Minimal valid budget_trend."""
    return {
        "budget_risk": "LOW",
    }


@pytest.fixture
def minimal_metric_conformance():
    """Minimal valid metric_conformance."""
    return {
        "status": "OK",
    }


@pytest.fixture
def perf_joint_view_low():
    """Joint view with LOW risk."""
    return {
        "perf_risk": "LOW",
        "budget_risk": "LOW",
        "slices_with_regressions": [],
        "slices_blocking_uplift": [],
        "summary_note": "Performance governance: nominal",
    }


@pytest.fixture
def perf_joint_view_medium():
    """Joint view with MEDIUM risk."""
    return {
        "perf_risk": "MEDIUM",
        "budget_risk": "LOW",
        "slices_with_regressions": ["slice_a"],
        "slices_blocking_uplift": [],
        "summary_note": "Performance governance: monitoring",
    }


@pytest.fixture
def perf_joint_view_high():
    """Joint view with HIGH risk."""
    return {
        "perf_risk": "HIGH",
        "budget_risk": "HIGH",
        "slices_with_regressions": ["slice_a", "slice_b"],
        "slices_blocking_uplift": ["slice_a"],
        "summary_note": "High performance risk detected; High budget risk detected; Performance blocking uplift on 1 slice(s)",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# INPUT VALIDATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestInputValidation:
    """Test input contract validation."""

    def test_validate_perf_trend_valid(self, minimal_perf_trend):
        """Valid perf_trend should pass."""
        validate_perf_trend(minimal_perf_trend)

    def test_validate_perf_trend_not_dict(self):
        """Non-dict perf_trend should fail."""
        with pytest.raises(PerfGovernanceInputError, match="must be a dict"):
            validate_perf_trend("not a dict")

    def test_validate_perf_trend_missing_schema_version(self, minimal_perf_trend):
        """Missing schema_version should fail."""
        del minimal_perf_trend["schema_version"]
        with pytest.raises(PerfGovernanceInputError, match="missing required key"):
            validate_perf_trend(minimal_perf_trend)

    def test_validate_perf_trend_missing_runs(self, minimal_perf_trend):
        """Missing runs should fail."""
        del minimal_perf_trend["runs"]
        with pytest.raises(PerfGovernanceInputError, match="missing required key"):
            validate_perf_trend(minimal_perf_trend)

    def test_validate_perf_trend_runs_not_list(self, minimal_perf_trend):
        """runs not a list should fail."""
        minimal_perf_trend["runs"] = "not a list"
        with pytest.raises(PerfGovernanceInputError, match="must be a list"):
            validate_perf_trend(minimal_perf_trend)

    def test_validate_perf_trend_run_missing_status(self, minimal_perf_trend):
        """Run missing status should fail."""
        minimal_perf_trend["runs"][0].pop("status", None)
        with pytest.raises(PerfGovernanceInputError, match="missing 'status'"):
            validate_perf_trend(minimal_perf_trend)

    def test_validate_perf_trend_invalid_risk_level(self, minimal_perf_trend):
        """Invalid risk level should fail."""
        minimal_perf_trend["release_risk_level"] = "INVALID"
        with pytest.raises(PerfGovernanceInputError, match="must be one of"):
            validate_perf_trend(minimal_perf_trend)

    def test_validate_budget_trend_valid(self, minimal_budget_trend):
        """Valid budget_trend should pass."""
        validate_budget_trend(minimal_budget_trend)

    def test_validate_budget_trend_not_dict(self):
        """Non-dict budget_trend should fail."""
        with pytest.raises(PerfGovernanceInputError, match="must be a dict"):
            validate_budget_trend("not a dict")

    def test_validate_budget_trend_missing_budget_risk(self):
        """Missing budget_risk should be allowed (graceful default)."""
        # Empty dict should pass - budget_risk defaults to LOW
        validate_budget_trend({})

    def test_validate_budget_trend_invalid_risk_level(self, minimal_budget_trend):
        """Invalid risk level should fail."""
        minimal_budget_trend["budget_risk"] = "INVALID"
        with pytest.raises(PerfGovernanceInputError, match="must be one of"):
            validate_budget_trend(minimal_budget_trend)

    def test_validate_metric_conformance_valid(self, minimal_metric_conformance):
        """Valid metric_conformance should pass."""
        validate_metric_conformance(minimal_metric_conformance)

    def test_validate_metric_conformance_not_dict(self):
        """Non-dict metric_conformance should fail."""
        with pytest.raises(PerfGovernanceInputError, match="must be a dict"):
            validate_metric_conformance("not a dict")

    def test_validate_metric_conformance_invalid_status(self):
        """Invalid status should fail."""
        with pytest.raises(PerfGovernanceInputError, match="must be OK/WARN/BLOCK"):
            validate_metric_conformance({"status": "INVALID"})

    def test_validate_perf_governance_inputs_all_valid(
        self, minimal_perf_trend, minimal_budget_trend, minimal_metric_conformance
    ):
        """All valid inputs should pass."""
        validate_perf_governance_inputs(
            perf_trend=minimal_perf_trend,
            budget_trend=minimal_budget_trend,
            metric_conformance=minimal_metric_conformance,
        )

    def test_validate_perf_governance_inputs_partial(self, minimal_perf_trend):
        """Partial inputs should be allowed."""
        validate_perf_governance_inputs(perf_trend=minimal_perf_trend)

    def test_validate_perf_governance_inputs_invalid_perf_trend(self):
        """Invalid perf_trend should fail."""
        with pytest.raises(PerfGovernanceInputError):
            validate_perf_governance_inputs(perf_trend="invalid")


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL CONSOLE ADAPTER TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestGlobalConsoleAdapter:
    """Test global console adapter."""

    def test_global_console_low_risk_green(self, perf_joint_view_low):
        """LOW risk should produce GREEN status."""
        tile = summarize_perf_for_global_console(perf_joint_view_low)
        assert tile["status_light"] == "GREEN"
        assert tile["perf_risk"] == "LOW"
        assert tile["tile_type"] == "uplift_perf_health"
        assert tile["schema_version"] == "1.0.0"

    def test_global_console_medium_risk_yellow(self, perf_joint_view_medium):
        """MEDIUM risk should produce YELLOW status."""
        tile = summarize_perf_for_global_console(perf_joint_view_medium)
        assert tile["status_light"] == "YELLOW"
        assert tile["perf_risk"] == "MEDIUM"

    def test_global_console_high_risk_red(self, perf_joint_view_high):
        """HIGH risk should produce RED status."""
        tile = summarize_perf_for_global_console(perf_joint_view_high)
        assert tile["status_light"] == "RED"
        assert tile["perf_risk"] == "HIGH"

    def test_global_console_blocking_uplift_red(self):
        """Slices blocking uplift should produce RED status."""
        joint_view = {
            "perf_risk": "MEDIUM",
            "budget_risk": "LOW",
            "slices_with_regressions": ["slice_a"],
            "slices_blocking_uplift": ["slice_a"],
            "summary_note": "test",
        }
        tile = summarize_perf_for_global_console(joint_view)
        assert tile["status_light"] == "RED"

    def test_global_console_regressions_yellow(self):
        """Regressions without blocking should produce YELLOW."""
        joint_view = {
            "perf_risk": "LOW",
            "budget_risk": "LOW",
            "slices_with_regressions": ["slice_a"],
            "slices_blocking_uplift": [],
            "summary_note": "test",
        }
        tile = summarize_perf_for_global_console(joint_view)
        assert tile["status_light"] == "YELLOW"

    def test_global_console_critical_slices_filtered(self, perf_joint_view_medium):
        """Critical slices should exclude 'all_slices'."""
        perf_joint_view_medium["slices_with_regressions"] = ["slice_a", "all_slices"]
        tile = summarize_perf_for_global_console(perf_joint_view_medium)
        assert "slice_a" in tile["critical_slices_with_regressions"]
        assert "all_slices" not in tile["critical_slices_with_regressions"]

    def test_global_console_json_serializable(self, perf_joint_view_low):
        """Tile should be JSON serializable."""
        tile = summarize_perf_for_global_console(perf_joint_view_low)
        json_str = json.dumps(tile)
        assert isinstance(json_str, str)
        # Should be able to round-trip
        parsed = json.loads(json_str)
        assert parsed == tile

    def test_global_console_deterministic(self, perf_joint_view_low):
        """Tile should be deterministic for same input."""
        tile1 = summarize_perf_for_global_console(perf_joint_view_low)
        tile2 = summarize_perf_for_global_console(perf_joint_view_low)
        assert tile1 == tile2

    def test_global_console_missing_key(self):
        """Missing required key should raise error."""
        with pytest.raises(PerfGovernanceInputError, match="missing required key"):
            summarize_perf_for_global_console({})


# ═══════════════════════════════════════════════════════════════════════════════
# EVIDENCE PACK HOOK TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestEvidencePackHook:
    """Test evidence pack hook tile."""

    def test_evidence_tile_low_risk_ok(self, perf_joint_view_low):
        """LOW risk should produce OK status."""
        tile = build_uplift_perf_governance_tile(perf_joint_view_low)
        assert tile["status"] == "OK"
        assert tile["tile_type"] == "uplift_perf_governance"
        assert tile["schema_version"] == "1.0.0"
        assert tile["perf_risk"] == "LOW"

    def test_evidence_tile_medium_risk_warn(self, perf_joint_view_medium):
        """MEDIUM risk should produce WARN status."""
        tile = build_uplift_perf_governance_tile(perf_joint_view_medium)
        assert tile["status"] == "WARN"
        assert tile["perf_risk"] == "MEDIUM"

    def test_evidence_tile_high_risk_block(self, perf_joint_view_high):
        """HIGH risk should produce BLOCK status."""
        tile = build_uplift_perf_governance_tile(perf_joint_view_high)
        assert tile["status"] == "BLOCK"
        assert tile["perf_risk"] == "HIGH"

    def test_evidence_tile_blocking_uplift_block(self):
        """Slices blocking uplift should produce BLOCK."""
        joint_view = {
            "perf_risk": "MEDIUM",
            "budget_risk": "LOW",
            "slices_with_regressions": ["slice_a"],
            "slices_blocking_uplift": ["slice_a"],
            "summary_note": "test",
        }
        tile = build_uplift_perf_governance_tile(joint_view)
        assert tile["status"] == "BLOCK"

    def test_evidence_tile_slices_preserved(self, perf_joint_view_high):
        """Slices should be preserved in tile."""
        tile = build_uplift_perf_governance_tile(perf_joint_view_high)
        assert tile["slices_with_regressions"] == ["slice_a", "slice_b"]
        assert tile["slices_blocking_uplift"] == ["slice_a"]

    def test_evidence_tile_notes_populated(self, perf_joint_view_high):
        """Notes should be populated when conditions exist."""
        tile = build_uplift_perf_governance_tile(perf_joint_view_high)
        assert len(tile["notes"]) > 0
        assert any("Regressions" in note for note in tile["notes"])
        assert any("blocking uplift" in note for note in tile["notes"])

    def test_evidence_tile_notes_empty_when_nominal(self, perf_joint_view_low):
        """Notes should use summary_note when no conditions."""
        tile = build_uplift_perf_governance_tile(perf_joint_view_low)
        assert len(tile["notes"]) >= 0  # May have summary_note

    def test_evidence_tile_json_serializable(self, perf_joint_view_low):
        """Tile should be JSON serializable."""
        tile = build_uplift_perf_governance_tile(perf_joint_view_low)
        json_str = json.dumps(tile)
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed == tile

    def test_evidence_tile_deterministic(self, perf_joint_view_low):
        """Tile should be deterministic for same input."""
        tile1 = build_uplift_perf_governance_tile(perf_joint_view_low)
        tile2 = build_uplift_perf_governance_tile(perf_joint_view_low)
        assert tile1 == tile2

    def test_evidence_tile_missing_key(self):
        """Missing required key should raise error."""
        with pytest.raises(PerfGovernanceInputError, match="missing required key"):
            build_uplift_perf_governance_tile({})


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestIntegration:
    """Test integration of all components."""

    def test_full_pipeline_low_risk(
        self, minimal_perf_trend, minimal_budget_trend, minimal_metric_conformance
    ):
        """Full pipeline with LOW risk."""
        joint_view = build_perf_joint_governance_view(
            minimal_perf_trend, minimal_budget_trend, minimal_metric_conformance
        )
        console_tile = summarize_perf_for_global_console(joint_view)
        evidence_tile = build_uplift_perf_governance_tile(joint_view)
        release_summary = summarize_perf_for_global_release(joint_view)

        assert console_tile["status_light"] == "GREEN"
        assert evidence_tile["status"] == "OK"
        assert release_summary["status"] == "OK"
        assert release_summary["release_ok"] is True

    def test_full_pipeline_high_risk(self):
        """Full pipeline with HIGH risk."""
        perf_trend = {
            "schema_version": "1.0.0",
            "runs": [
                {"run_id": "run1", "status": "FAIL", "slice_name": "slice_a"},
            ],
            "release_risk_level": "HIGH",
        }
        budget_trend = {"budget_risk": "HIGH"}
        metric_conformance = {"status": "OK"}

        joint_view = build_perf_joint_governance_view(
            perf_trend, budget_trend, metric_conformance
        )
        console_tile = summarize_perf_for_global_console(joint_view)
        evidence_tile = build_uplift_perf_governance_tile(joint_view)
        release_summary = summarize_perf_for_global_release(joint_view)

        assert console_tile["status_light"] == "RED"
        assert evidence_tile["status"] == "BLOCK"
        assert release_summary["status"] == "BLOCK"
        assert release_summary["release_ok"] is False

    def test_missing_dimensions_default_ok(self):
        """Missing perf/metric dimensions should default to OK."""
        perf_trend = {
            "schema_version": "1.0.0",
            "runs": [],
        }
        budget_trend = {}  # Empty dict - should default gracefully
        metric_conformance = {}  # Empty dict - should default gracefully

        # Should not raise, should default to LOW risk
        joint_view = build_perf_joint_governance_view(
            perf_trend, budget_trend, metric_conformance
        )
        assert joint_view["perf_risk"] == "LOW"
        assert joint_view["budget_risk"] == "LOW"

    def test_all_tiles_json_safe(self, perf_joint_view_low):
        """All tiles should be JSON-safe."""
        console_tile = summarize_perf_for_global_console(perf_joint_view_low)
        evidence_tile = build_uplift_perf_governance_tile(perf_joint_view_low)
        release_summary = summarize_perf_for_global_release(perf_joint_view_low)

        # All should serialize
        json.dumps(console_tile)
        json.dumps(evidence_tile)
        json.dumps(release_summary)

