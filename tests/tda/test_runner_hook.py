"""
TDA Runner Hook Tests — Phase VII NEURAL LINK

Operation CORTEX: Phase VII Integration Layer Tests
===================================================

Tests for:
1. TDAMonitorResult construction and serialization
2. TDAGovernanceHook behavior across modes (OFF, SHADOW, DRY_RUN, HARD)
3. Exception window interaction
4. Deterministic serialization (to_dict)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from backend.tda.runner_hook import (
    TDAMonitorResult,
    TDAGovernanceHook,
    create_tda_hook,
    TDA_RUNNER_HOOK_SCHEMA_VERSION,
)
from backend.tda.governance import (
    TDAHardGateMode,
    ExceptionWindowManager,
    HardGateDecision,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def mock_exception_manager() -> ExceptionWindowManager:
    """Create a fresh exception manager for testing."""
    return ExceptionWindowManager(max_runs=10)


@pytest.fixture
def hook_hard_mode(mock_exception_manager: ExceptionWindowManager) -> TDAGovernanceHook:
    """Create hook in HARD mode."""
    return TDAGovernanceHook(
        mode=TDAHardGateMode.HARD,
        exception_manager=mock_exception_manager,
    )


@pytest.fixture
def hook_dry_run_mode(mock_exception_manager: ExceptionWindowManager) -> TDAGovernanceHook:
    """Create hook in DRY_RUN mode."""
    return TDAGovernanceHook(
        mode=TDAHardGateMode.DRY_RUN,
        exception_manager=mock_exception_manager,
    )


@pytest.fixture
def hook_shadow_mode(mock_exception_manager: ExceptionWindowManager) -> TDAGovernanceHook:
    """Create hook in SHADOW mode."""
    return TDAGovernanceHook(
        mode=TDAHardGateMode.SHADOW,
        exception_manager=mock_exception_manager,
    )


@pytest.fixture
def high_hss_cycle_result() -> Dict[str, Any]:
    """Cycle result with HSS above block threshold."""
    return {
        "hss": 0.85,
        "sns": 0.70,
        "pcs": 0.65,
        "drs": 0.10,
    }


@pytest.fixture
def low_hss_cycle_result() -> Dict[str, Any]:
    """Cycle result with HSS below block threshold (should block)."""
    return {
        "hss": 0.05,
        "sns": 0.20,
        "pcs": 0.15,
        "drs": 0.50,
    }


@pytest.fixture
def minimal_telemetry() -> Dict[str, Any]:
    """Minimal telemetry dict."""
    return {
        "abstention_rate": 0.02,
        "slice_name": "test_slice",
    }


# ============================================================================
# TDAMonitorResult Tests
# ============================================================================

class TestTDAMonitorResult:
    """Tests for TDAMonitorResult dataclass."""

    def test_create_with_all_fields(self):
        """TDAMonitorResult can be created with all fields."""
        result = TDAMonitorResult(
            cycle_id=42,
            hss=0.75,
            sns=0.60,
            pcs=0.55,
            drs=0.15,
            timestamp="2025-12-09T12:00:00+00:00",
            metadata={"test": True},
        )

        assert result.cycle_id == 42
        assert result.hss == 0.75
        assert result.sns == 0.60
        assert result.pcs == 0.55
        assert result.drs == 0.15
        assert result.timestamp == "2025-12-09T12:00:00+00:00"
        assert result.metadata == {"test": True}

    def test_block_property_below_threshold(self):
        """block property is True when HSS < 0.2."""
        result = TDAMonitorResult(
            cycle_id=0, hss=0.15, sns=0.0, pcs=0.0, drs=0.0,
            timestamp="2025-12-09T12:00:00+00:00",
        )
        assert result.block is True

    def test_block_property_above_threshold(self):
        """block property is False when HSS >= 0.2."""
        result = TDAMonitorResult(
            cycle_id=0, hss=0.25, sns=0.0, pcs=0.0, drs=0.0,
            timestamp="2025-12-09T12:00:00+00:00",
        )
        assert result.block is False

    def test_block_property_at_threshold(self):
        """block property is False when HSS == 0.2 exactly."""
        result = TDAMonitorResult(
            cycle_id=0, hss=0.2, sns=0.0, pcs=0.0, drs=0.0,
            timestamp="2025-12-09T12:00:00+00:00",
        )
        assert result.block is False

    def test_warn_property(self):
        """warn property is True when HSS < 0.4."""
        result_warn = TDAMonitorResult(
            cycle_id=0, hss=0.35, sns=0.0, pcs=0.0, drs=0.0,
            timestamp="2025-12-09T12:00:00+00:00",
        )
        result_ok = TDAMonitorResult(
            cycle_id=0, hss=0.45, sns=0.0, pcs=0.0, drs=0.0,
            timestamp="2025-12-09T12:00:00+00:00",
        )
        assert result_warn.warn is True
        assert result_ok.warn is False

    def test_to_dict_serialization(self):
        """to_dict produces JSON-serializable dict."""
        result = TDAMonitorResult(
            cycle_id=42,
            hss=0.123456789,
            sns=0.234567891,
            pcs=0.345678912,
            drs=0.456789123,
            timestamp="2025-12-09T12:00:00+00:00",
            metadata={"key": "value"},
        )

        d = result.to_dict()

        # Check JSON serializable
        json_str = json.dumps(d)
        assert isinstance(json_str, str)

        # Check rounding
        assert d["hss"] == 0.123457
        assert d["sns"] == 0.234568
        assert d["pcs"] == 0.345679
        assert d["drs"] == 0.456789

    def test_frozen_immutability(self):
        """TDAMonitorResult is immutable (frozen)."""
        result = TDAMonitorResult(
            cycle_id=0, hss=0.5, sns=0.0, pcs=0.0, drs=0.0,
            timestamp="2025-12-09T12:00:00+00:00",
        )

        with pytest.raises(AttributeError):
            result.hss = 0.9


# ============================================================================
# TDAGovernanceHook Tests — HARD Mode
# ============================================================================

class TestTDAGovernanceHookHardMode:
    """Tests for TDAGovernanceHook in HARD mode."""

    def test_hard_mode_blocks_low_hss(
        self,
        hook_hard_mode: TDAGovernanceHook,
        low_hss_cycle_result: Dict[str, Any],
        minimal_telemetry: Dict[str, Any],
    ):
        """HARD mode blocks when HSS below threshold."""
        decision = hook_hard_mode.on_cycle_complete(
            cycle_index=0,
            success=True,
            cycle_result=low_hss_cycle_result,
            telemetry=minimal_telemetry,
        )

        assert decision.should_block is True
        assert decision.should_log_as_would_block is False
        assert decision.mode == TDAHardGateMode.HARD
        assert hook_hard_mode.get_block_count() == 1

    def test_hard_mode_allows_high_hss(
        self,
        hook_hard_mode: TDAGovernanceHook,
        high_hss_cycle_result: Dict[str, Any],
        minimal_telemetry: Dict[str, Any],
    ):
        """HARD mode allows when HSS above threshold."""
        decision = hook_hard_mode.on_cycle_complete(
            cycle_index=0,
            success=True,
            cycle_result=high_hss_cycle_result,
            telemetry=minimal_telemetry,
        )

        assert decision.should_block is False
        assert decision.should_log_as_would_block is False
        assert hook_hard_mode.get_block_count() == 0

    def test_hard_mode_increments_block_count(
        self,
        hook_hard_mode: TDAGovernanceHook,
        low_hss_cycle_result: Dict[str, Any],
        minimal_telemetry: Dict[str, Any],
    ):
        """HARD mode increments block_count on each block."""
        for i in range(3):
            hook_hard_mode.on_cycle_complete(
                cycle_index=i,
                success=True,
                cycle_result=low_hss_cycle_result,
                telemetry=minimal_telemetry,
            )

        assert hook_hard_mode.get_block_count() == 3

    def test_hard_mode_with_exception_window_active(
        self,
        mock_exception_manager: ExceptionWindowManager,
        low_hss_cycle_result: Dict[str, Any],
        minimal_telemetry: Dict[str, Any],
    ):
        """HARD mode with exception window acts as dry-run."""
        # Activate exception window
        mock_exception_manager.activate("test divergence")

        hook = TDAGovernanceHook(
            mode=TDAHardGateMode.HARD,
            exception_manager=mock_exception_manager,
        )

        decision = hook.on_cycle_complete(
            cycle_index=0,
            success=True,
            cycle_result=low_hss_cycle_result,
            telemetry=minimal_telemetry,
        )

        # Should NOT block due to exception window
        assert decision.should_block is False
        assert decision.should_log_as_would_block is True
        assert decision.exception_window_active is True
        assert hook.get_block_count() == 0
        assert hook.get_would_block_count() == 1


# ============================================================================
# TDAGovernanceHook Tests — DRY_RUN Mode
# ============================================================================

class TestTDAGovernanceHookDryRunMode:
    """Tests for TDAGovernanceHook in DRY_RUN mode."""

    def test_dry_run_never_blocks(
        self,
        hook_dry_run_mode: TDAGovernanceHook,
        low_hss_cycle_result: Dict[str, Any],
        minimal_telemetry: Dict[str, Any],
    ):
        """DRY_RUN mode never actually blocks."""
        decision = hook_dry_run_mode.on_cycle_complete(
            cycle_index=0,
            success=True,
            cycle_result=low_hss_cycle_result,
            telemetry=minimal_telemetry,
        )

        assert decision.should_block is False
        assert hook_dry_run_mode.get_block_count() == 0

    def test_dry_run_logs_would_block(
        self,
        hook_dry_run_mode: TDAGovernanceHook,
        low_hss_cycle_result: Dict[str, Any],
        minimal_telemetry: Dict[str, Any],
    ):
        """DRY_RUN mode logs would_block for low HSS."""
        decision = hook_dry_run_mode.on_cycle_complete(
            cycle_index=0,
            success=True,
            cycle_result=low_hss_cycle_result,
            telemetry=minimal_telemetry,
        )

        assert decision.should_log_as_would_block is True
        assert hook_dry_run_mode.get_would_block_count() == 1

    def test_dry_run_no_would_block_for_high_hss(
        self,
        hook_dry_run_mode: TDAGovernanceHook,
        high_hss_cycle_result: Dict[str, Any],
        minimal_telemetry: Dict[str, Any],
    ):
        """DRY_RUN mode does not log would_block for high HSS."""
        decision = hook_dry_run_mode.on_cycle_complete(
            cycle_index=0,
            success=True,
            cycle_result=high_hss_cycle_result,
            telemetry=minimal_telemetry,
        )

        assert decision.should_log_as_would_block is False
        assert hook_dry_run_mode.get_would_block_count() == 0


# ============================================================================
# TDAGovernanceHook Tests — SHADOW Mode
# ============================================================================

class TestTDAGovernanceHookShadowMode:
    """Tests for TDAGovernanceHook in SHADOW mode."""

    def test_shadow_mode_never_blocks(
        self,
        hook_shadow_mode: TDAGovernanceHook,
        low_hss_cycle_result: Dict[str, Any],
        minimal_telemetry: Dict[str, Any],
    ):
        """SHADOW mode never blocks."""
        decision = hook_shadow_mode.on_cycle_complete(
            cycle_index=0,
            success=True,
            cycle_result=low_hss_cycle_result,
            telemetry=minimal_telemetry,
        )

        assert decision.should_block is False
        assert hook_shadow_mode.get_block_count() == 0

    def test_shadow_mode_no_would_block_logging(
        self,
        hook_shadow_mode: TDAGovernanceHook,
        low_hss_cycle_result: Dict[str, Any],
        minimal_telemetry: Dict[str, Any],
    ):
        """SHADOW mode does not log would_block."""
        decision = hook_shadow_mode.on_cycle_complete(
            cycle_index=0,
            success=True,
            cycle_result=low_hss_cycle_result,
            telemetry=minimal_telemetry,
        )

        assert decision.should_log_as_would_block is False
        assert hook_shadow_mode.get_would_block_count() == 0

    def test_shadow_mode_still_records_decisions(
        self,
        hook_shadow_mode: TDAGovernanceHook,
        low_hss_cycle_result: Dict[str, Any],
        high_hss_cycle_result: Dict[str, Any],
        minimal_telemetry: Dict[str, Any],
    ):
        """SHADOW mode still records all decisions."""
        hook_shadow_mode.on_cycle_complete(
            cycle_index=0,
            success=True,
            cycle_result=low_hss_cycle_result,
            telemetry=minimal_telemetry,
        )
        hook_shadow_mode.on_cycle_complete(
            cycle_index=1,
            success=True,
            cycle_result=high_hss_cycle_result,
            telemetry=minimal_telemetry,
        )

        assert len(hook_shadow_mode.decisions) == 2


# ============================================================================
# TDAGovernanceHook Tests — OFF Mode
# ============================================================================

class TestTDAGovernanceHookOffMode:
    """Tests for TDAGovernanceHook in OFF mode."""

    def test_off_mode_never_blocks(
        self,
        mock_exception_manager: ExceptionWindowManager,
        low_hss_cycle_result: Dict[str, Any],
        minimal_telemetry: Dict[str, Any],
    ):
        """OFF mode never blocks."""
        hook = TDAGovernanceHook(
            mode=TDAHardGateMode.OFF,
            exception_manager=mock_exception_manager,
        )

        decision = hook.on_cycle_complete(
            cycle_index=0,
            success=True,
            cycle_result=low_hss_cycle_result,
            telemetry=minimal_telemetry,
        )

        assert decision.should_block is False
        assert decision.should_log_as_would_block is False


# ============================================================================
# TDAGovernanceHook Tests — Serialization
# ============================================================================

class TestTDAGovernanceHookSerialization:
    """Tests for TDAGovernanceHook serialization."""

    def test_to_dict_is_json_serializable(
        self,
        hook_hard_mode: TDAGovernanceHook,
        high_hss_cycle_result: Dict[str, Any],
        low_hss_cycle_result: Dict[str, Any],
        minimal_telemetry: Dict[str, Any],
    ):
        """to_dict produces JSON-serializable output."""
        hook_hard_mode.on_cycle_complete(
            cycle_index=0,
            success=True,
            cycle_result=high_hss_cycle_result,
            telemetry=minimal_telemetry,
        )
        hook_hard_mode.on_cycle_complete(
            cycle_index=1,
            success=True,
            cycle_result=low_hss_cycle_result,
            telemetry=minimal_telemetry,
        )

        d = hook_hard_mode.to_dict()

        # Should not raise
        json_str = json.dumps(d)
        assert isinstance(json_str, str)

    def test_to_dict_is_deterministic(
        self,
        mock_exception_manager: ExceptionWindowManager,
        high_hss_cycle_result: Dict[str, Any],
        minimal_telemetry: Dict[str, Any],
    ):
        """to_dict produces identical output for identical inputs."""
        # Create two hooks with same inputs
        hook1 = TDAGovernanceHook(
            mode=TDAHardGateMode.HARD,
            exception_manager=ExceptionWindowManager(max_runs=10),
        )
        hook2 = TDAGovernanceHook(
            mode=TDAHardGateMode.HARD,
            exception_manager=ExceptionWindowManager(max_runs=10),
        )

        # Process same cycle result (with mocked timestamp)
        with patch('backend.tda.runner_hook.datetime') as mock_dt:
            mock_dt.now.return_value = datetime(2025, 12, 9, 12, 0, 0, tzinfo=timezone.utc)
            mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)

            hook1.on_cycle_complete(
                cycle_index=0,
                success=True,
                cycle_result=high_hss_cycle_result,
                telemetry=minimal_telemetry,
            )
            hook2.on_cycle_complete(
                cycle_index=0,
                success=True,
                cycle_result=high_hss_cycle_result,
                telemetry=minimal_telemetry,
            )

        # Compare serializations (excluding timestamp which varies)
        d1 = hook1.to_dict()
        d2 = hook2.to_dict()

        # Mode, cycle_count, block_count should match
        assert d1["mode"] == d2["mode"]
        assert d1["cycle_count"] == d2["cycle_count"]
        assert d1["block_count"] == d2["block_count"]
        assert d1["would_block_count"] == d2["would_block_count"]

    def test_to_dict_includes_schema_version(
        self,
        hook_hard_mode: TDAGovernanceHook,
    ):
        """to_dict includes schema version."""
        d = hook_hard_mode.to_dict()
        assert d["schema_version"] == TDA_RUNNER_HOOK_SCHEMA_VERSION

    def test_get_summary_for_export(
        self,
        hook_hard_mode: TDAGovernanceHook,
        high_hss_cycle_result: Dict[str, Any],
        low_hss_cycle_result: Dict[str, Any],
        minimal_telemetry: Dict[str, Any],
    ):
        """get_summary_for_export produces compact summary."""
        # Process cycles
        hook_hard_mode.on_cycle_complete(
            cycle_index=0,
            success=True,
            cycle_result=high_hss_cycle_result,
            telemetry=minimal_telemetry,
        )
        hook_hard_mode.on_cycle_complete(
            cycle_index=1,
            success=True,
            cycle_result=low_hss_cycle_result,
            telemetry=minimal_telemetry,
        )

        summary = hook_hard_mode.get_summary_for_export()

        assert summary["cycle_count"] == 2
        assert summary["block_count"] == 1
        assert summary["block_rate"] == 0.5
        assert "decisions" not in summary  # Compact, no per-cycle data
        assert "results" not in summary


# ============================================================================
# TDAGovernanceHook Tests — Missing TDA Data
# ============================================================================

class TestTDAGovernanceHookMissingData:
    """Tests for TDAGovernanceHook handling of missing TDA data."""

    def test_missing_hss_defaults_to_zero(
        self,
        hook_hard_mode: TDAGovernanceHook,
        minimal_telemetry: Dict[str, Any],
    ):
        """Missing HSS defaults to 0.0 and sets tda_missing flag."""
        # Cycle result without HSS
        cycle_result = {"some_other_field": 123}

        decision = hook_hard_mode.on_cycle_complete(
            cycle_index=0,
            success=True,
            cycle_result=cycle_result,
            telemetry=minimal_telemetry,
        )

        # Should block since HSS=0.0 < 0.2
        assert decision.should_block is True

        # Check metadata flag
        assert len(hook_hard_mode._results) == 1
        assert hook_hard_mode._results[0].metadata.get("tda_missing") is True

    def test_missing_optional_metrics_default_to_zero(
        self,
        hook_hard_mode: TDAGovernanceHook,
        minimal_telemetry: Dict[str, Any],
    ):
        """Missing SNS/PCS/DRS default to 0.0."""
        cycle_result = {"hss": 0.85}  # Only HSS

        hook_hard_mode.on_cycle_complete(
            cycle_index=0,
            success=True,
            cycle_result=cycle_result,
            telemetry=minimal_telemetry,
        )

        result = hook_hard_mode._results[0]
        assert result.hss == 0.85
        assert result.sns == 0.0
        assert result.pcs == 0.0
        assert result.drs == 0.0


# ============================================================================
# Factory Function Tests
# ============================================================================

class TestCreateTdaHook:
    """Tests for create_tda_hook factory function."""

    def test_create_with_explicit_mode(self):
        """create_tda_hook respects explicit mode."""
        hook = create_tda_hook(mode=TDAHardGateMode.DRY_RUN)
        assert hook.mode == TDAHardGateMode.DRY_RUN

    def test_create_with_explicit_exception_manager(self):
        """create_tda_hook respects explicit exception manager."""
        mgr = ExceptionWindowManager(max_runs=50)
        hook = create_tda_hook(exception_manager=mgr)
        assert hook.exception_manager is mgr

    def test_create_defaults_to_shadow_or_env(self):
        """create_tda_hook defaults based on environment."""
        # Without env var set, defaults to HARD (per from_env)
        hook = create_tda_hook()
        # Mode comes from TDAHardGateMode.from_env() which defaults to HARD
        assert hook.mode in [TDAHardGateMode.HARD, TDAHardGateMode.SHADOW]


# ============================================================================
# Integration-style Tests
# ============================================================================

class TestTDAGovernanceHookIntegration:
    """Integration-style tests for complete workflows."""

    def test_multi_cycle_workflow(
        self,
        mock_exception_manager: ExceptionWindowManager,
    ):
        """Test complete multi-cycle workflow."""
        hook = TDAGovernanceHook(
            mode=TDAHardGateMode.HARD,
            exception_manager=mock_exception_manager,
        )

        # Simulate 5 cycles with varying HSS
        results = [
            {"hss": 0.90, "sns": 0.8, "pcs": 0.7, "drs": 0.1},  # OK
            {"hss": 0.85, "sns": 0.7, "pcs": 0.6, "drs": 0.1},  # OK
            {"hss": 0.10, "sns": 0.2, "pcs": 0.1, "drs": 0.5},  # BLOCK
            {"hss": 0.75, "sns": 0.6, "pcs": 0.5, "drs": 0.2},  # OK
            {"hss": 0.05, "sns": 0.1, "pcs": 0.1, "drs": 0.6},  # BLOCK
        ]

        telemetry = {"slice_name": "test"}

        for i, cycle_result in enumerate(results):
            hook.on_cycle_complete(
                cycle_index=i,
                success=True,
                cycle_result=cycle_result,
                telemetry=telemetry,
            )

        # Verify counts
        assert len(hook.decisions) == 5
        assert hook.get_block_count() == 2

        # Verify summary
        summary = hook.get_summary_for_export()
        assert summary["cycle_count"] == 5
        assert summary["block_count"] == 2
        assert summary["block_rate"] == 0.4

    def test_mode_transition_simulation(
        self,
        mock_exception_manager: ExceptionWindowManager,
    ):
        """Simulate what happens during mode transitions."""
        low_hss = {"hss": 0.05, "sns": 0.1, "pcs": 0.1, "drs": 0.5}
        telemetry = {"slice_name": "test"}

        # Start in SHADOW mode
        shadow_hook = TDAGovernanceHook(
            mode=TDAHardGateMode.SHADOW,
            exception_manager=ExceptionWindowManager(max_runs=10),
        )
        shadow_decision = shadow_hook.on_cycle_complete(
            cycle_index=0, success=True, cycle_result=low_hss, telemetry=telemetry
        )
        assert shadow_decision.should_block is False
        assert shadow_decision.should_log_as_would_block is False

        # Upgrade to DRY_RUN mode
        dryrun_hook = TDAGovernanceHook(
            mode=TDAHardGateMode.DRY_RUN,
            exception_manager=ExceptionWindowManager(max_runs=10),
        )
        dryrun_decision = dryrun_hook.on_cycle_complete(
            cycle_index=0, success=True, cycle_result=low_hss, telemetry=telemetry
        )
        assert dryrun_decision.should_block is False
        assert dryrun_decision.should_log_as_would_block is True

        # Upgrade to HARD mode
        hard_hook = TDAGovernanceHook(
            mode=TDAHardGateMode.HARD,
            exception_manager=ExceptionWindowManager(max_runs=10),
        )
        hard_decision = hard_hook.on_cycle_complete(
            cycle_index=0, success=True, cycle_result=low_hss, telemetry=telemetry
        )
        assert hard_decision.should_block is True
        assert hard_decision.should_log_as_would_block is False
