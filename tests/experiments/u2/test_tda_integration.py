"""
U2 Runner TDA Integration Tests — Phase VII NEURAL LINK

Operation CORTEX: Phase VII Integration Tests
=============================================

Tests for TDA governance hook integration with U2Runner.
Tests verify:
1. Hook registration works correctly
2. HARD mode blocks failed cycles (HSS=0.0)
3. DRY_RUN mode logs but does not block
4. SHADOW mode records decisions without affecting execution
5. Cycle result includes TDA block info when blocked
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock

import pytest


# ============================================================================
# Test: TDA Hook Registration
# ============================================================================

class TestU2TDAHookRegistration:
    """Tests for TDA hook registration on U2Runner."""

    def test_hook_can_process_u2_style_telemetry(self):
        """Hook correctly processes U2-style telemetry dict."""
        from backend.tda.runner_hook import TDAGovernanceHook
        from backend.tda.governance import TDAHardGateMode, ExceptionWindowManager

        hook = TDAGovernanceHook(
            mode=TDAHardGateMode.SHADOW,
            exception_manager=ExceptionWindowManager(),
        )

        # Simulate U2-style telemetry
        telemetry = {
            "slice_name": "arithmetic_simple",
            "cycle_index": 0,
            "mode": "baseline",
            "seed": 12345,
            "success": True,
        }

        # Simulate U2-style cycle result (HSS from success)
        cycle_result = {
            "hss": 1.0,  # success = HSS 1.0
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

class TestU2TDAHardModeBlocking:
    """Tests for HARD mode blocking behavior in U2 context."""

    def test_hard_mode_blocks_failed_cycles(self):
        """HARD mode blocks when cycle fails (HSS=0.0)."""
        from backend.tda.runner_hook import TDAGovernanceHook
        from backend.tda.governance import TDAHardGateMode, ExceptionWindowManager

        hook = TDAGovernanceHook(
            mode=TDAHardGateMode.HARD,
            exception_manager=ExceptionWindowManager(),
        )

        # Failed cycle = HSS 0.0 = should block
        cycle_result = {"hss": 0.0, "sns": 0.0, "pcs": 0.0, "drs": 0.0}
        telemetry = {"slice_name": "test", "success": False}

        decision = hook.on_cycle_complete(
            cycle_index=0,
            success=False,
            cycle_result=cycle_result,
            telemetry=telemetry,
        )

        assert decision.should_block is True
        assert hook.get_block_count() == 1

    def test_hard_mode_allows_successful_cycles(self):
        """HARD mode allows when cycle succeeds (HSS=1.0)."""
        from backend.tda.runner_hook import TDAGovernanceHook
        from backend.tda.governance import TDAHardGateMode, ExceptionWindowManager

        hook = TDAGovernanceHook(
            mode=TDAHardGateMode.HARD,
            exception_manager=ExceptionWindowManager(),
        )

        # Successful cycle = HSS 1.0 = should allow
        cycle_result = {"hss": 1.0, "sns": 0.0, "pcs": 0.0, "drs": 0.0}
        telemetry = {"slice_name": "test", "success": True}

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

class TestU2TDADryRunMode:
    """Tests for DRY_RUN mode behavior in U2 context."""

    def test_dry_run_logs_would_block(self):
        """DRY_RUN mode logs would_block but doesn't actually block."""
        from backend.tda.runner_hook import TDAGovernanceHook
        from backend.tda.governance import TDAHardGateMode, ExceptionWindowManager

        hook = TDAGovernanceHook(
            mode=TDAHardGateMode.DRY_RUN,
            exception_manager=ExceptionWindowManager(),
        )

        # Failed cycle that would trigger block in HARD mode
        cycle_result = {"hss": 0.0, "sns": 0.0, "pcs": 0.0, "drs": 0.0}
        telemetry = {"slice_name": "test", "success": False}

        decision = hook.on_cycle_complete(
            cycle_index=0,
            success=False,
            cycle_result=cycle_result,
            telemetry=telemetry,
        )

        assert decision.should_block is False
        assert decision.should_log_as_would_block is True
        assert hook.get_would_block_count() == 1


# ============================================================================
# Test: Simulated U2 Integration Flow
# ============================================================================

class TestSimulatedU2Flow:
    """Simulated U2 integration flow tests."""

    def test_multi_cycle_workflow(self):
        """Test multi-cycle workflow with mixed success/failure."""
        from backend.tda.runner_hook import TDAGovernanceHook
        from backend.tda.governance import TDAHardGateMode, ExceptionWindowManager

        hook = TDAGovernanceHook(
            mode=TDAHardGateMode.HARD,
            exception_manager=ExceptionWindowManager(),
        )

        # Simulate 5 cycles: 3 success, 2 failure
        results = [
            (True, 1.0),   # success
            (True, 1.0),   # success
            (False, 0.0),  # failure -> should block
            (True, 1.0),   # success
            (False, 0.0),  # failure -> should block
        ]

        blocks = 0
        for i, (success, hss) in enumerate(results):
            cycle_result = {"hss": hss, "sns": 0.0, "pcs": 0.0, "drs": 0.0}
            telemetry = {"slice_name": "test", "success": success}

            decision = hook.on_cycle_complete(
                cycle_index=i,
                success=success,
                cycle_result=cycle_result,
                telemetry=telemetry,
            )

            if decision.should_block:
                blocks += 1

        assert blocks == 2
        assert hook.get_block_count() == 2
        assert len(hook.decisions) == 5

    def test_export_summary_after_workflow(self):
        """Export summary contains correct metrics after workflow."""
        from backend.tda.runner_hook import TDAGovernanceHook
        from backend.tda.governance import TDAHardGateMode, ExceptionWindowManager

        hook = TDAGovernanceHook(
            mode=TDAHardGateMode.HARD,
            exception_manager=ExceptionWindowManager(),
        )

        # Process 4 cycles: 3 success, 1 failure
        for i, (success, hss) in enumerate([(True, 1.0), (True, 1.0), (True, 1.0), (False, 0.0)]):
            cycle_result = {"hss": hss, "sns": 0.0, "pcs": 0.0, "drs": 0.0}
            telemetry = {"slice_name": "test", "success": success}
            hook.on_cycle_complete(
                cycle_index=i,
                success=success,
                cycle_result=cycle_result,
                telemetry=telemetry,
            )

        summary = hook.get_summary_for_export()

        assert summary["cycle_count"] == 4
        assert summary["block_count"] == 1
        assert summary["block_rate"] == 0.25
        # Mean HSS = (1.0 + 1.0 + 1.0 + 0.0) / 4 = 0.75
        assert summary["mean_hss"] == 0.75


# ============================================================================
# Test: Exception Window Integration
# ============================================================================

class TestU2ExceptionWindowIntegration:
    """Tests for exception window behavior in U2 context."""

    def test_exception_window_allows_blocks_in_hard_mode(self):
        """Exception window converts blocks to would_block."""
        from backend.tda.runner_hook import TDAGovernanceHook
        from backend.tda.governance import TDAHardGateMode, ExceptionWindowManager

        mgr = ExceptionWindowManager(max_runs=5)
        mgr.activate("test divergence")

        hook = TDAGovernanceHook(
            mode=TDAHardGateMode.HARD,
            exception_manager=mgr,
        )

        # Failed cycle that would normally block
        cycle_result = {"hss": 0.0, "sns": 0.0, "pcs": 0.0, "drs": 0.0}
        telemetry = {"slice_name": "test", "success": False}

        decision = hook.on_cycle_complete(
            cycle_index=0,
            success=False,
            cycle_result=cycle_result,
            telemetry=telemetry,
        )

        # Should NOT block due to exception window
        assert decision.should_block is False
        assert decision.should_log_as_would_block is True
        assert decision.exception_window_active is True
        assert hook.get_block_count() == 0
        assert hook.get_would_block_count() == 1
