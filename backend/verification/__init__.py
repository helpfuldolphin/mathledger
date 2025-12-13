"""
Verification module for MathLedger.

This module provides verification capabilities including mock oracle components
for testing and negative control scenarios.
"""

import os
import sys

# Check if we're in a test context
_IN_TEST_CONTEXT = (
    "pytest" in sys.modules
    or "unittest" in sys.modules
    or os.getenv("MATHLEDGER_ALLOW_MOCK_ORACLE") == "1"
)

# Conditionally export mock oracle components
if _IN_TEST_CONTEXT:
    try:
        from .mock_oracle_minimal import (
            mock_verify,
            list_profiles,
            get_profile_info,
            build_mock_oracle_fleet_summary,
            evaluate_mock_oracle_fleet_for_ci,
            build_mock_oracle_drift_tile,
            build_first_light_mock_oracle_summary,
            build_control_arm_calibration_summary,
            build_control_vs_twin_panel,
            control_arm_for_alignment_view,
            summarize_control_arm_signal_consistency,
            attach_mock_oracle_to_evidence,
            MOCK_ORACLE_SCHEMA_VERSION,
            FLEET_SUMMARY_SCHEMA_VERSION,
            DRIFT_TILE_SCHEMA_VERSION,
            FIRST_LIGHT_SUMMARY_SCHEMA_VERSION,
            CONTROL_ARM_CALIBRATION_SCHEMA_VERSION,
            CONTROL_VS_TWIN_PANEL_SCHEMA_VERSION,
            FLEET_STATUS_OK,
            FLEET_STATUS_DRIFTING,
            FLEET_STATUS_BROKEN,
            DRIFT_STATUS_OK,
            DRIFT_STATUS_DRIFTING,
            DRIFT_STATUS_INVALID_HEAVY,
            STATUS_LIGHT_GREEN,
            STATUS_LIGHT_YELLOW,
            STATUS_LIGHT_RED,
            VERDICT_SUCCESS,
            VERDICT_FAILURE,
            VERDICT_ABSTAIN,
        )
        
        __all__ = [
            "mock_verify",
            "list_profiles",
            "get_profile_info",
            "build_mock_oracle_fleet_summary",
            "evaluate_mock_oracle_fleet_for_ci",
            "build_mock_oracle_drift_tile",
            "build_first_light_mock_oracle_summary",
            "build_control_arm_calibration_summary",
            "build_control_vs_twin_panel",
            "control_arm_for_alignment_view",
            "summarize_control_arm_signal_consistency",
            "attach_mock_oracle_to_evidence",
            "MOCK_ORACLE_SCHEMA_VERSION",
            "FLEET_SUMMARY_SCHEMA_VERSION",
            "DRIFT_TILE_SCHEMA_VERSION",
            "FIRST_LIGHT_SUMMARY_SCHEMA_VERSION",
            "CONTROL_ARM_CALIBRATION_SCHEMA_VERSION",
            "CONTROL_VS_TWIN_PANEL_SCHEMA_VERSION",
            "FLEET_STATUS_OK",
            "FLEET_STATUS_DRIFTING",
            "FLEET_STATUS_BROKEN",
            "DRIFT_STATUS_OK",
            "DRIFT_STATUS_DRIFTING",
            "DRIFT_STATUS_INVALID_HEAVY",
            "STATUS_LIGHT_GREEN",
            "STATUS_LIGHT_YELLOW",
            "STATUS_LIGHT_RED",
            "VERDICT_SUCCESS",
            "VERDICT_FAILURE",
            "VERDICT_ABSTAIN",
        ]
    except ImportError:
        # Mock oracle module not available
        __all__ = []
else:
    # Not in test context and mock oracle not explicitly enabled
    __all__ = []
    
    def __getattr__(name: str):
        """Raise helpful error if trying to import mock oracle without enabling it."""
        if name in [
            "mock_verify",
            "list_profiles",
            "get_profile_info",
            "build_mock_oracle_fleet_summary",
            "evaluate_mock_oracle_fleet_for_ci",
            "build_mock_oracle_drift_tile",
            "build_first_light_mock_oracle_summary",
            "build_control_arm_calibration_summary",
            "build_control_vs_twin_panel",
            "control_arm_for_alignment_view",
            "summarize_control_arm_signal_consistency",
            "attach_mock_oracle_to_evidence",
        ]:
            raise RuntimeError(
                f"Mock oracle component '{name}' is not available. "
                "Set MATHLEDGER_ALLOW_MOCK_ORACLE=1 to enable. "
                "This is a test-only feature and must not be used in production."
            )
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

