"""
RFL Runner TDA Integration Tests — Phase VII NEURAL LINK

Operation CORTEX: Phase VII Integration Tests
=============================================

Tests for TDA governance hook integration with RFLRunner.
Tests verify:
1. Hook registration works correctly
2. HARD mode blocks low-HSS cycles and reverts policy updates
3. DRY_RUN mode logs but does not block
4. SHADOW mode records decisions without affecting execution
5. Export includes TDA governance summary
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest


# ============================================================================
# Mock Attestation Context for Testing
# ============================================================================

@dataclass
class MockAttestedRunContext:
    """Mock attestation context for testing without DB dependencies."""
    slice_id: str = "test_slice"
    policy_id: str = "default"
    composite_root: str = "a" * 64
    reasoning_root: str = "b" * 64
    ui_root: str = "c" * 64
    statement_hash: str = "d" * 64
    abstention_rate: float = 0.1
    abstention_mass: float = 10.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.metadata:
            self.metadata = {
                "timestamp": "2025-12-09T12:00:00Z",
                "first_organism_ended_at": "2025-12-09T12:00:01Z",
                "first_organism_duration_seconds": 1.0,
                "first_organism_abstentions": 1,
                "verified_count": 5,
                "coverage_rate": 0.95,
                "novelty_rate": 0.1,
                "throughput": 100.0,
                "success_rate": 0.95,
                "derive_steps": 10,
                "max_breadth": 100,
                "max_total": 500,
                "abstention_breakdown": {},
            }


# ============================================================================
# Test: TDA Hook Registration
# ============================================================================

class TestTDAHookRegistration:
    """Tests for TDA hook registration on RFLRunner."""

    def test_runner_has_tda_hook_attribute(self):
        """RFLRunner has _tda_hook attribute initialized to None."""
        # We can't instantiate real RFLRunner without DB, so we test the integration
        # by checking that our modifications work with mock objects
        from backend.tda.runner_hook import TDAGovernanceHook, create_tda_hook
        from backend.tda.governance import TDAHardGateMode, ExceptionWindowManager

        hook = create_tda_hook(mode=TDAHardGateMode.SHADOW)
        assert hook is not None
        assert hook.mode == TDAHardGateMode.SHADOW

    def test_hook_can_process_rfl_style_telemetry(self):
        """Hook correctly processes RFL-style telemetry dict."""
        from backend.tda.runner_hook import TDAGovernanceHook
        from backend.tda.governance import TDAHardGateMode, ExceptionWindowManager

        hook = TDAGovernanceHook(
            mode=TDAHardGateMode.SHADOW,
            exception_manager=ExceptionWindowManager(),
        )

        # Simulate RFL-style telemetry
        telemetry = {
            "abstention_rate": 0.1,
            "abstention_mass": 10.0,
            "composite_root": "a" * 64,
            "slice_name": "test_slice",
            "policy_update_applied": True,
            "success": True,
        }

        # Simulate RFL-style cycle result (HSS derived from abstention)
        cycle_result = {
            "hss": 0.9,  # 1.0 - 0.1 abstention
            "sns": 0.0,
            "pcs": 0.0,
            "drs": 0.0,
        }

        decision = hook.on_cycle_complete(
            cycle_index=0,
            success=True,
            cycle_result=cycle_result,
            telemetry=telemetry,
        )

        # Shadow mode should not block
        assert decision.should_block is False
        assert len(hook.decisions) == 1


# ============================================================================
# Test: TDA HARD Mode Blocking
# ============================================================================

class TestTDAHardModeBlocking:
    """Tests for HARD mode blocking behavior."""

    def test_hard_mode_blocks_high_abstention(self):
        """HARD mode blocks when abstention is high (HSS low)."""
        from backend.tda.runner_hook import TDAGovernanceHook
        from backend.tda.governance import TDAHardGateMode, ExceptionWindowManager

        hook = TDAGovernanceHook(
            mode=TDAHardGateMode.HARD,
            exception_manager=ExceptionWindowManager(),
        )

        # High abstention = low HSS = should block
        telemetry = {
            "abstention_rate": 0.9,  # 90% abstention
            "success": True,
        }
        cycle_result = {
            "hss": 0.1,  # 1.0 - 0.9 = very low
            "sns": 0.0,
            "pcs": 0.0,
            "drs": 0.0,
        }

        decision = hook.on_cycle_complete(
            cycle_index=0,
            success=True,
            cycle_result=cycle_result,
            telemetry=telemetry,
        )

        assert decision.should_block is True
        assert hook.get_block_count() == 1

    def test_hard_mode_allows_low_abstention(self):
        """HARD mode allows when abstention is low (HSS high)."""
        from backend.tda.runner_hook import TDAGovernanceHook
        from backend.tda.governance import TDAHardGateMode, ExceptionWindowManager

        hook = TDAGovernanceHook(
            mode=TDAHardGateMode.HARD,
            exception_manager=ExceptionWindowManager(),
        )

        # Low abstention = high HSS = should allow
        telemetry = {
            "abstention_rate": 0.05,  # 5% abstention
            "success": True,
        }
        cycle_result = {
            "hss": 0.95,  # High HSS
            "sns": 0.0,
            "pcs": 0.0,
            "drs": 0.0,
        }

        decision = hook.on_cycle_complete(
            cycle_index=0,
            success=True,
            cycle_result=cycle_result,
            telemetry=telemetry,
        )

        assert decision.should_block is False
        assert hook.get_block_count() == 0


# ============================================================================
# Test: TDA DRY_RUN Mode
# ============================================================================

class TestTDADryRunMode:
    """Tests for DRY_RUN mode behavior."""

    def test_dry_run_logs_would_block(self):
        """DRY_RUN mode logs would_block but doesn't actually block."""
        from backend.tda.runner_hook import TDAGovernanceHook
        from backend.tda.governance import TDAHardGateMode, ExceptionWindowManager

        hook = TDAGovernanceHook(
            mode=TDAHardGateMode.DRY_RUN,
            exception_manager=ExceptionWindowManager(),
        )

        # High abstention that would trigger block in HARD mode
        cycle_result = {"hss": 0.1, "sns": 0.0, "pcs": 0.0, "drs": 0.0}
        telemetry = {"abstention_rate": 0.9, "success": True}

        decision = hook.on_cycle_complete(
            cycle_index=0,
            success=True,
            cycle_result=cycle_result,
            telemetry=telemetry,
        )

        assert decision.should_block is False
        assert decision.should_log_as_would_block is True
        assert hook.get_would_block_count() == 1


# ============================================================================
# Test: TDA Export Integration
# ============================================================================

class TestTDAExportIntegration:
    """Tests for TDA governance export in results."""

    def test_export_summary_contains_required_fields(self):
        """Export summary contains all required fields."""
        from backend.tda.runner_hook import TDAGovernanceHook
        from backend.tda.governance import TDAHardGateMode, ExceptionWindowManager

        hook = TDAGovernanceHook(
            mode=TDAHardGateMode.HARD,
            exception_manager=ExceptionWindowManager(),
        )

        # Process a few cycles
        for i in range(3):
            cycle_result = {"hss": 0.8, "sns": 0.0, "pcs": 0.0, "drs": 0.0}
            telemetry = {"abstention_rate": 0.2, "success": True}
            hook.on_cycle_complete(
                cycle_index=i,
                success=True,
                cycle_result=cycle_result,
                telemetry=telemetry,
            )

        summary = hook.get_summary_for_export()

        assert "schema_version" in summary
        assert "mode" in summary
        assert "cycle_count" in summary
        assert "block_count" in summary
        assert "block_rate" in summary
        assert "mean_hss" in summary

        assert summary["mode"] == "hard"
        assert summary["cycle_count"] == 3
        assert summary["block_count"] == 0


# ============================================================================
# Test: Simulated RFL Integration Flow
# ============================================================================

class TestSimulatedRFLFlow:
    """Simulated RFL integration flow tests."""

    def test_full_rfl_cycle_with_tda_block(self):
        """Simulate full RFL cycle where TDA blocks due to high abstention."""
        from backend.tda.runner_hook import TDAGovernanceHook
        from backend.tda.governance import TDAHardGateMode, ExceptionWindowManager

        # Create hook in HARD mode
        hook = TDAGovernanceHook(
            mode=TDAHardGateMode.HARD,
            exception_manager=ExceptionWindowManager(),
        )

        # Simulate attestation with high abstention
        attestation = MockAttestedRunContext(
            abstention_rate=0.85,  # 85% abstention = HSS 0.15 (below threshold)
        )

        # Build cycle result as RFLRunner would
        cycle_result = {
            "hss": max(0.0, 1.0 - attestation.abstention_rate),
            "sns": 0.0,
            "pcs": 0.0,
            "drs": 0.0,
        }

        # Build telemetry as RFLRunner would
        telemetry = {
            "abstention_rate": attestation.abstention_rate,
            "abstention_mass": attestation.abstention_mass,
            "composite_root": attestation.composite_root,
            "slice_name": attestation.slice_id,
            "success": True,
        }

        # Call hook
        decision = hook.on_cycle_complete(
            cycle_index=1,
            success=True,
            cycle_result=cycle_result,
            telemetry=telemetry,
        )

        # Verify blocking behavior
        assert decision.should_block is True
        assert hook.get_block_count() == 1
        assert "HSS=0.150" in decision.reason or "hss" in decision.reason.lower()

    def test_full_rfl_cycle_with_tda_pass(self):
        """Simulate full RFL cycle where TDA allows low abstention."""
        from backend.tda.runner_hook import TDAGovernanceHook
        from backend.tda.governance import TDAHardGateMode, ExceptionWindowManager

        # Create hook in HARD mode
        hook = TDAGovernanceHook(
            mode=TDAHardGateMode.HARD,
            exception_manager=ExceptionWindowManager(),
        )

        # Simulate attestation with low abstention
        attestation = MockAttestedRunContext(
            abstention_rate=0.05,  # 5% abstention = HSS 0.95 (above threshold)
        )

        # Build cycle result as RFLRunner would
        cycle_result = {
            "hss": max(0.0, 1.0 - attestation.abstention_rate),
            "sns": 0.0,
            "pcs": 0.0,
            "drs": 0.0,
        }

        telemetry = {
            "abstention_rate": attestation.abstention_rate,
            "success": True,
        }

        decision = hook.on_cycle_complete(
            cycle_index=1,
            success=True,
            cycle_result=cycle_result,
            telemetry=telemetry,
        )

        # Verify pass behavior
        assert decision.should_block is False
        assert hook.get_block_count() == 0

    def test_mode_transition_shadow_to_hard(self):
        """Test transition from SHADOW to HARD mode across cycles."""
        from backend.tda.runner_hook import TDAGovernanceHook
        from backend.tda.governance import TDAHardGateMode, ExceptionWindowManager

        # Low HSS cycle result
        low_hss_result = {"hss": 0.1, "sns": 0.0, "pcs": 0.0, "drs": 0.0}
        telemetry = {"abstention_rate": 0.9, "success": True}

        # SHADOW mode: should not block
        shadow_hook = TDAGovernanceHook(
            mode=TDAHardGateMode.SHADOW,
            exception_manager=ExceptionWindowManager(),
        )
        shadow_decision = shadow_hook.on_cycle_complete(
            cycle_index=0, success=True, cycle_result=low_hss_result, telemetry=telemetry
        )
        assert shadow_decision.should_block is False
        assert shadow_decision.should_log_as_would_block is False

        # DRY_RUN mode: should log but not block
        dryrun_hook = TDAGovernanceHook(
            mode=TDAHardGateMode.DRY_RUN,
            exception_manager=ExceptionWindowManager(),
        )
        dryrun_decision = dryrun_hook.on_cycle_complete(
            cycle_index=0, success=True, cycle_result=low_hss_result, telemetry=telemetry
        )
        assert dryrun_decision.should_block is False
        assert dryrun_decision.should_log_as_would_block is True

        # HARD mode: should block
        hard_hook = TDAGovernanceHook(
            mode=TDAHardGateMode.HARD,
            exception_manager=ExceptionWindowManager(),
        )
        hard_decision = hard_hook.on_cycle_complete(
            cycle_index=0, success=True, cycle_result=low_hss_result, telemetry=telemetry
        )
        assert hard_decision.should_block is True
        assert hard_decision.should_log_as_would_block is False
