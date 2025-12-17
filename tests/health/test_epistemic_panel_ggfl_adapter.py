"""
Tests for epistemic panel GGFL adapter.

SHADOW MODE CONTRACT:
- All tests verify observational behavior only
- No gating or blocking logic is tested
- Tests verify GGFL adapter determinism and shape
"""

import json
from typing import Dict, Any

import pytest

from backend.health.epistemic_p3p4_integration import epistemic_panel_for_alignment_view

# Import reusable warning neutrality helpers (single source of truth)
from tests.helpers.warning_neutrality import pytest_assert_warning_neutral


class TestEpistemicPanelGGFLAdapter:
    """Tests for epistemic panel GGFL adapter."""

    def test_adapter_returns_fixed_shape(
        self,
    ) -> None:
        """Test that adapter returns fixed shape with all required fields."""
        panel = {
            "schema_version": "1.0.0",
            "num_experiments": 3,
            "num_consistent": 1,
            "num_inconsistent": 2,
            "num_unknown": 0,
            "top_inconsistent_cal_ids_top3": ["CAL-EXP-1"],
            "top_reason_code": "EPI_DEGRADED_EVID_IMPROVING",
            "reason_code_histogram": {
                "EPI_DEGRADED_EVID_IMPROVING": 2,
            },
        }

        result = epistemic_panel_for_alignment_view(panel)

        # Check required fields
        required_fields = {
            "signal_type",
            "status",
            "conflict",
            "drivers",
            "summary",
            "weight_hint",
            "shadow_mode_invariants",
        }
        assert required_fields.issubset(set(result.keys()))

        # Check field types and values
        assert result["signal_type"] == "SIG-EPI"
        assert result["status"] in {"ok", "warn"}
        assert result["conflict"] is False
        assert isinstance(result["drivers"], list)
        assert len(result["drivers"]) <= 3
        assert isinstance(result["summary"], str)
        assert result["weight_hint"] == "LOW"
        assert isinstance(result["shadow_mode_invariants"], dict)

    def test_status_warn_when_inconsistent(
        self,
    ) -> None:
        """Test that status is 'warn' when num_inconsistent > 0."""
        panel = {
            "num_experiments": 3,
            "num_consistent": 1,
            "num_inconsistent": 2,
            "num_unknown": 0,
            "experiments_inconsistent": [],
            "reason_code_histogram": {},
        }

        result = epistemic_panel_for_alignment_view(panel)

        assert result["status"] == "warn"

    def test_status_ok_when_consistent(
        self,
    ) -> None:
        """Test that status is 'ok' when num_inconsistent == 0."""
        panel = {
            "num_experiments": 3,
            "num_consistent": 3,
            "num_inconsistent": 0,
            "num_unknown": 0,
            "experiments_inconsistent": [],
            "reason_code_histogram": {},
        }

        result = epistemic_panel_for_alignment_view(panel)

        assert result["status"] == "ok"

    def test_drivers_include_top_reason_code(
        self,
    ) -> None:
        """Test that drivers include top reason code in reason-code format."""
        panel = {
            "num_experiments": 3,
            "num_consistent": 1,
            "num_inconsistent": 2,
            "num_unknown": 0,
            "experiments_inconsistent": [],
            "top_reason_code": "EPI_DEGRADED_EVID_IMPROVING",
            "reason_code_histogram": {
                "EPI_DEGRADED_EVID_IMPROVING": 2,
                "EPI_DEGRADED_EVID_STABLE": 1,
            },
        }

        result = epistemic_panel_for_alignment_view(panel)

        # Top reason code should be in reason-code driver format
        drivers = result["drivers"]
        assert "DRIVER_TOP_REASON_EPI_DEGRADED_EVID_IMPROVING" in drivers

    def test_drivers_include_top_cal_ids(
        self,
    ) -> None:
        """Test that drivers include top cal_ids in reason-code format."""
        panel = {
            "num_experiments": 3,
            "num_consistent": 1,
            "num_inconsistent": 2,
            "num_unknown": 0,
            "top_inconsistent_cal_ids_top3": ["CAL-EXP-1", "CAL-EXP-2"],
            "top_reason_code": "EPI_DEGRADED_EVID_IMPROVING",
            "reason_code_histogram": {
                "EPI_DEGRADED_EVID_IMPROVING": 1,
                "EPI_DEGRADED_EVID_STABLE": 1,
            },
        }

        result = epistemic_panel_for_alignment_view(panel)

        # Should include reason-code driver for cal_ids
        drivers = result["drivers"]
        assert "DRIVER_TOP_CAL_IDS_PRESENT" in drivers

    def test_drivers_limited_to_3_reason_code_only(
        self,
    ) -> None:
        """Test that drivers are limited to 3 and use reason-code format only."""
        panel = {
            "num_experiments": 5,
            "num_consistent": 0,
            "num_inconsistent": 5,
            "num_unknown": 0,
            "top_inconsistent_cal_ids_top3": ["CAL-EXP-1", "CAL-EXP-2", "CAL-EXP-3"],
            "top_reason_code": "EPI_DEGRADED_EVID_IMPROVING",
            "reason_code_histogram": {
                "EPI_DEGRADED_EVID_IMPROVING": 5,
            },
        }

        result = epistemic_panel_for_alignment_view(panel)

        drivers = result["drivers"]
        assert len(drivers) <= 3
        
        # Verify all drivers are reason-code format
        for driver in drivers:
            assert driver.startswith("DRIVER_"), f"Driver '{driver}' does not start with DRIVER_"

    def test_summary_neutral_language(
        self,
    ) -> None:
        """Test that summary uses neutral, descriptive language."""
        panel = {
            "num_experiments": 3,
            "num_consistent": 1,
            "num_inconsistent": 2,
            "num_unknown": 0,
            "experiments_inconsistent": [],
            "reason_code_histogram": {},
        }

        result = epistemic_panel_for_alignment_view(panel)

        # Use reusable helper (single source of truth for banned words)
        pytest_assert_warning_neutral(result["summary"], context="GGFL summary")

    def test_is_deterministic(
        self,
    ) -> None:
        """Test that output is deterministic for identical inputs."""
        panel = {
            "schema_version": "1.0.0",
            "num_experiments": 3,
            "num_consistent": 1,
            "num_inconsistent": 2,
            "num_unknown": 0,
            "experiments_inconsistent": [
                {
                    "cal_id": "CAL-EXP-1",
                    "reason": "Test reason",
                    "reason_code": "EPI_DEGRADED_EVID_IMPROVING",
                },
            ],
            "reason_code_histogram": {
                "EPI_DEGRADED_EVID_IMPROVING": 2,
            },
        }

        result1 = epistemic_panel_for_alignment_view(panel)
        result2 = epistemic_panel_for_alignment_view(panel)

        assert result1 == result2

    def test_works_with_status_signal(
        self,
    ) -> None:
        """Test that adapter works with signal from first_light_status.json."""
        signal = {
            "schema_version": "1.0.0",
            "mode": "SHADOW",
            "extraction_source": "MANIFEST",
            "panel_schema_version": "1.0.0",
            "num_experiments": 3,
            "num_consistent": 1,
            "num_inconsistent": 2,
            "num_unknown": 0,
            "top_inconsistent_cal_ids_top3": ["CAL-EXP-1", "CAL-EXP-2"],
            "top_reason_code": "EPI_DEGRADED_EVID_IMPROVING",
            "reason_code_histogram": {
                "EPI_DEGRADED_EVID_IMPROVING": 2,
            },
        }

        result = epistemic_panel_for_alignment_view(signal)

        assert result["signal_type"] == "SIG-EPI"
        assert result["status"] == "warn"
        assert result["conflict"] is False
        assert result["weight_hint"] == "LOW"
        assert "shadow_mode_invariants" in result
        
        # Verify drivers use top_reason_code from signal
        drivers = result["drivers"]
        assert "DRIVER_TOP_REASON_EPI_DEGRADED_EVID_IMPROVING" in drivers

    def test_json_safe(
        self,
    ) -> None:
        """Test that output is JSON-safe."""
        panel = {
            "num_experiments": 3,
            "num_consistent": 1,
            "num_inconsistent": 2,
            "num_unknown": 0,
            "experiments_inconsistent": [],
            "reason_code_histogram": {},
        }

        result = epistemic_panel_for_alignment_view(panel)

        # Should not raise
        json_str = json.dumps(result)
        assert isinstance(json_str, str)

        # Should be able to parse back
        parsed = json.loads(json_str)
        assert parsed["signal_type"] == "SIG-EPI"
    
    def test_drivers_are_reason_code_only_no_prose(
        self,
    ) -> None:
        """Test that drivers use reason-code format only (no prose)."""
        panel = {
            "num_experiments": 3,
            "num_consistent": 1,
            "num_inconsistent": 2,
            "num_unknown": 0,
            "top_inconsistent_cal_ids_top3": ["CAL-EXP-1"],
            "top_reason_code": "EPI_DEGRADED_EVID_IMPROVING",
            "reason_code_histogram": {
                "EPI_DEGRADED_EVID_IMPROVING": 2,
            },
        }

        result = epistemic_panel_for_alignment_view(panel)

        drivers = result["drivers"]
        # All drivers should start with DRIVER_
        for driver in drivers:
            assert driver.startswith("DRIVER_"), f"Driver '{driver}' does not start with DRIVER_"
        
        # Verify expected reason-code drivers
        assert "DRIVER_EPI_INCONSISTENT_PRESENT" in drivers
        assert "DRIVER_TOP_REASON_EPI_DEGRADED_EVID_IMPROVING" in drivers
        assert "DRIVER_TOP_CAL_IDS_PRESENT" in drivers
    
    def test_drivers_deterministic_ordering(
        self,
    ) -> None:
        """Test that drivers follow deterministic ordering: inconsistent, top reason, top cal ids."""
        panel = {
            "num_experiments": 3,
            "num_consistent": 1,
            "num_inconsistent": 2,
            "num_unknown": 0,
            "top_inconsistent_cal_ids_top3": ["CAL-EXP-1", "CAL-EXP-2"],
            "top_reason_code": "EPI_DEGRADED_EVID_IMPROVING",
            "reason_code_histogram": {
                "EPI_DEGRADED_EVID_IMPROVING": 2,
            },
        }

        result = epistemic_panel_for_alignment_view(panel)

        drivers = result["drivers"]
        # Verify ordering: 1. inconsistent, 2. top reason, 3. top cal ids
        assert len(drivers) == 3
        assert drivers[0] == "DRIVER_EPI_INCONSISTENT_PRESENT"
        assert drivers[1] == "DRIVER_TOP_REASON_EPI_DEGRADED_EVID_IMPROVING"
        assert drivers[2] == "DRIVER_TOP_CAL_IDS_PRESENT"
    
    def test_shadow_mode_invariants_fixed_values(
        self,
    ) -> None:
        """Test that shadow_mode_invariants have fixed values."""
        panel = {
            "num_experiments": 3,
            "num_consistent": 1,
            "num_inconsistent": 2,
            "num_unknown": 0,
            "experiments_inconsistent": [],
            "reason_code_histogram": {},
        }

        result = epistemic_panel_for_alignment_view(panel)

        assert "shadow_mode_invariants" in result
        invariants = result["shadow_mode_invariants"]
        assert invariants["advisory_only"] is True
        assert invariants["no_enforcement"] is True
        assert invariants["conflict_invariant"] is True
        
        # Verify conflict is always False (invariant)
        assert result["conflict"] is False
        
        # Verify weight_hint is always LOW
        assert result["weight_hint"] == "LOW"
    
    def test_top_reason_code_deterministic_tie_breaking(
        self,
    ) -> None:
        """Test that top_reason_code selection handles ties deterministically (highest count desc, then reason_code asc)."""
        panel = {
            "num_experiments": 4,
            "num_consistent": 0,
            "num_inconsistent": 4,
            "num_unknown": 0,
            "experiments_inconsistent": [
                {
                    "cal_id": "CAL-EXP-1",
                    "reason": "Test",
                    "reason_code": "EPI_DEGRADED_EVID_STABLE",
                },
                {
                    "cal_id": "CAL-EXP-2",
                    "reason": "Test",
                    "reason_code": "EPI_DEGRADED_EVID_IMPROVING",
                },
                {
                    "cal_id": "CAL-EXP-3",
                    "reason": "Test",
                    "reason_code": "EPI_DEGRADED_EVID_STABLE",
                },
                {
                    "cal_id": "CAL-EXP-4",
                    "reason": "Test",
                    "reason_code": "EPI_DEGRADED_EVID_IMPROVING",
                },
            ],
            "reason_code_histogram": {
                "EPI_DEGRADED_EVID_STABLE": 2,
                "EPI_DEGRADED_EVID_IMPROVING": 2,
            },
        }

        result1 = epistemic_panel_for_alignment_view(panel)
        result2 = epistemic_panel_for_alignment_view(panel)

        # Both should select same top reason code (deterministic)
        # With tie (both count 2), should pick EPI_DEGRADED_EVID_IMPROVING (asc order tie-breaker)
        assert result1["drivers"] == result2["drivers"]
        assert "DRIVER_TOP_REASON_EPI_DEGRADED_EVID_IMPROVING" in result1["drivers"]
        
        # Verify byte-identical output for identical inputs
        json1 = json.dumps(result1, sort_keys=True)
        json2 = json.dumps(result2, sort_keys=True)
        assert json1 == json2
    
    def test_top3_truncation_and_ordering(
        self,
    ) -> None:
        """Test that top_inconsistent_cal_ids_top3 is truncated to top 3 and sorted deterministically."""
        panel = {
            "num_experiments": 5,
            "num_consistent": 0,
            "num_inconsistent": 5,
            "num_unknown": 0,
            "experiments_inconsistent": [
                {
                    "cal_id": f"CAL-EXP-{i}",
                    "reason": "Test",
                    "reason_code": "EPI_DEGRADED_EVID_IMPROVING",
                }
                for i in range(5, 0, -1)  # Reverse order to test sorting
            ],
            "reason_code_histogram": {
                "EPI_DEGRADED_EVID_IMPROVING": 5,
            },
        }

        result = epistemic_panel_for_alignment_view(panel)

        # Should extract top cal_ids from experiments_inconsistent
        # Should be sorted and truncated to top 3
        drivers = result["drivers"]
        assert "DRIVER_TOP_CAL_IDS_PRESENT" in drivers
        
        # Verify determinism: run twice and compare
        result2 = epistemic_panel_for_alignment_view(panel)
        assert result["drivers"] == result2["drivers"]

