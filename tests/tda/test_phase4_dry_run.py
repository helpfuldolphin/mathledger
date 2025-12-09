"""
TDA Phase IV: Dry-Run Mode Tests

Operation CORTEX: Phase IV Guardrails
=====================================

Tests for the dry-run hard gate mode, ensuring operators can preview
what would be blocked without actually blocking.

Test Coverage:
- TDAHardGateMode enum behavior
- Mode resolution from environment
- evaluate_hard_gate_decision() for each mode
- Exception window interaction with HARD mode
- Deterministic behavior guarantees
- Evidence tile generation in dry-run mode
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional
from unittest.mock import patch

import pytest

from backend.tda.governance import (
    TDAHardGateMode,
    HardGateDecision,
    evaluate_hard_gate_decision,
    ExceptionWindowManager,
    ExceptionWindowState,
    summarize_tda_for_global_health_v2,
    build_tda_hard_gate_evidence_tile,
)


# ============================================================================
# Mock TDA Result for Testing
# ============================================================================

@dataclass
class MockTDAResult:
    """Mock TDAMonitorResult for testing hard gate decisions."""
    hss: float
    sns: float = 0.5
    pcs: float = 0.5
    drs: float = 0.1
    block: bool = False
    warn: bool = False

    def __post_init__(self):
        # Auto-compute block/warn from HSS if not explicitly set
        if self.hss < 0.2:
            object.__setattr__(self, 'block', True)
        elif self.hss < 0.4:
            object.__setattr__(self, 'warn', True)


@dataclass
class MockTDAConfig:
    """Mock TDAMonitorConfig for testing."""
    hss_block_threshold: float = 0.2
    hss_warn_threshold: float = 0.4
    mode: TDAHardGateMode = TDAHardGateMode.HARD
    lifetime_threshold: float = 0.01
    deviation_max: float = 0.3
    max_simplex_dim: int = 2
    max_homology_dim: int = 1
    fail_open: bool = False


# ============================================================================
# Test: TDAHardGateMode Enum
# ============================================================================

class TestTDAHardGateModeEnum:
    """Tests for TDAHardGateMode enumeration."""

    def test_mode_values_are_lowercase_strings(self):
        """Mode values are lowercase for environment compatibility."""
        assert TDAHardGateMode.OFF.value == "off"
        assert TDAHardGateMode.SHADOW.value == "shadow"
        assert TDAHardGateMode.DRY_RUN.value == "dry_run"
        assert TDAHardGateMode.HARD.value == "hard"

    def test_all_modes_are_unique(self):
        """All mode values are unique."""
        values = [m.value for m in TDAHardGateMode]
        assert len(values) == len(set(values))

    def test_mode_from_string_lowercase(self):
        """Can construct mode from lowercase string."""
        assert TDAHardGateMode("off") == TDAHardGateMode.OFF
        assert TDAHardGateMode("shadow") == TDAHardGateMode.SHADOW
        assert TDAHardGateMode("dry_run") == TDAHardGateMode.DRY_RUN
        assert TDAHardGateMode("hard") == TDAHardGateMode.HARD


class TestTDAHardGateModeFromEnv:
    """Tests for mode resolution from environment variables."""

    def test_from_env_defaults_to_hard(self):
        """Default mode is HARD when env var not set."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove the env var if present
            os.environ.pop("MATHLEDGER_TDA_HARD_GATE_MODE", None)
            mode = TDAHardGateMode.from_env()
            assert mode == TDAHardGateMode.HARD

    def test_from_env_reads_off(self):
        """Can read OFF mode from environment."""
        with patch.dict(os.environ, {"MATHLEDGER_TDA_HARD_GATE_MODE": "off"}):
            mode = TDAHardGateMode.from_env()
            assert mode == TDAHardGateMode.OFF

    def test_from_env_reads_shadow(self):
        """Can read SHADOW mode from environment."""
        with patch.dict(os.environ, {"MATHLEDGER_TDA_HARD_GATE_MODE": "shadow"}):
            mode = TDAHardGateMode.from_env()
            assert mode == TDAHardGateMode.SHADOW

    def test_from_env_reads_dry_run(self):
        """Can read DRY_RUN mode from environment."""
        with patch.dict(os.environ, {"MATHLEDGER_TDA_HARD_GATE_MODE": "dry_run"}):
            mode = TDAHardGateMode.from_env()
            assert mode == TDAHardGateMode.DRY_RUN

    def test_from_env_reads_hard(self):
        """Can read HARD mode from environment."""
        with patch.dict(os.environ, {"MATHLEDGER_TDA_HARD_GATE_MODE": "hard"}):
            mode = TDAHardGateMode.from_env()
            assert mode == TDAHardGateMode.HARD

    def test_from_env_case_insensitive(self):
        """Mode parsing is case-insensitive."""
        with patch.dict(os.environ, {"MATHLEDGER_TDA_HARD_GATE_MODE": "DRY_RUN"}):
            mode = TDAHardGateMode.from_env()
            assert mode == TDAHardGateMode.DRY_RUN

    def test_from_env_invalid_value_defaults_to_hard(self):
        """Invalid mode value defaults to HARD with warning."""
        with patch.dict(os.environ, {"MATHLEDGER_TDA_HARD_GATE_MODE": "invalid_mode"}):
            mode = TDAHardGateMode.from_env()
            assert mode == TDAHardGateMode.HARD


# ============================================================================
# Test: evaluate_hard_gate_decision - OFF Mode
# ============================================================================

class TestHardGateDecisionOffMode:
    """Tests for OFF mode behavior."""

    def test_off_mode_never_blocks(self):
        """OFF mode never blocks, regardless of HSS."""
        low_hss_result = MockTDAResult(hss=0.05)  # Very low HSS
        decision = evaluate_hard_gate_decision(low_hss_result, TDAHardGateMode.OFF)

        assert decision.should_block is False
        assert decision.should_log_as_would_block is False
        assert decision.mode == TDAHardGateMode.OFF

    def test_off_mode_reason_indicates_disabled(self):
        """OFF mode decision reason indicates TDA is disabled."""
        result = MockTDAResult(hss=0.1)
        decision = evaluate_hard_gate_decision(result, TDAHardGateMode.OFF)

        assert "OFF" in decision.reason or "disabled" in decision.reason.lower()

    def test_off_mode_ignores_exception_window(self):
        """OFF mode ignores exception window state."""
        result = MockTDAResult(hss=0.1)
        manager = ExceptionWindowManager(max_runs=10)
        manager.activate("test")

        decision = evaluate_hard_gate_decision(result, TDAHardGateMode.OFF, manager)

        assert decision.should_block is False
        assert decision.exception_window_active is False


# ============================================================================
# Test: evaluate_hard_gate_decision - SHADOW Mode
# ============================================================================

class TestHardGateDecisionShadowMode:
    """Tests for SHADOW mode behavior."""

    def test_shadow_mode_never_blocks(self):
        """SHADOW mode never blocks, regardless of HSS."""
        low_hss_result = MockTDAResult(hss=0.05)
        decision = evaluate_hard_gate_decision(low_hss_result, TDAHardGateMode.SHADOW)

        assert decision.should_block is False
        assert decision.mode == TDAHardGateMode.SHADOW

    def test_shadow_mode_does_not_log_would_block(self):
        """SHADOW mode logs TDA scores but not 'would block'."""
        low_hss_result = MockTDAResult(hss=0.05)
        decision = evaluate_hard_gate_decision(low_hss_result, TDAHardGateMode.SHADOW)

        # SHADOW differs from DRY_RUN: no "would block" logging
        assert decision.should_log_as_would_block is False

    def test_shadow_mode_reason_indicates_logging_only(self):
        """SHADOW mode reason indicates logging-only behavior."""
        result = MockTDAResult(hss=0.1)
        decision = evaluate_hard_gate_decision(result, TDAHardGateMode.SHADOW)

        assert "shadow" in decision.reason.lower() or "logging" in decision.reason.lower()


# ============================================================================
# Test: evaluate_hard_gate_decision - DRY_RUN Mode
# ============================================================================

class TestHardGateDecisionDryRunMode:
    """Tests for DRY_RUN mode behavior."""

    def test_dry_run_mode_never_blocks(self):
        """DRY_RUN mode never blocks, regardless of HSS."""
        low_hss_result = MockTDAResult(hss=0.05)
        decision = evaluate_hard_gate_decision(low_hss_result, TDAHardGateMode.DRY_RUN)

        assert decision.should_block is False
        assert decision.mode == TDAHardGateMode.DRY_RUN

    def test_dry_run_logs_would_block_for_low_hss(self):
        """DRY_RUN mode logs 'would block' for low HSS."""
        low_hss_result = MockTDAResult(hss=0.05)
        decision = evaluate_hard_gate_decision(low_hss_result, TDAHardGateMode.DRY_RUN)

        assert decision.should_log_as_would_block is True

    def test_dry_run_does_not_log_would_block_for_high_hss(self):
        """DRY_RUN mode does not log 'would block' for high HSS."""
        high_hss_result = MockTDAResult(hss=0.8)
        decision = evaluate_hard_gate_decision(high_hss_result, TDAHardGateMode.DRY_RUN)

        assert decision.should_log_as_would_block is False

    def test_dry_run_reason_includes_hss_value(self):
        """DRY_RUN mode reason includes HSS value for auditability."""
        result = MockTDAResult(hss=0.15)
        decision = evaluate_hard_gate_decision(result, TDAHardGateMode.DRY_RUN)

        assert "0.15" in decision.reason or "HSS" in decision.reason

    def test_dry_run_reason_indicates_would_block_status(self):
        """DRY_RUN mode reason indicates what would have happened."""
        low_result = MockTDAResult(hss=0.05)
        decision = evaluate_hard_gate_decision(low_result, TDAHardGateMode.DRY_RUN)

        assert "would_block=True" in decision.reason or "dry" in decision.reason.lower()


# ============================================================================
# Test: evaluate_hard_gate_decision - HARD Mode
# ============================================================================

class TestHardGateDecisionHardMode:
    """Tests for HARD mode behavior."""

    def test_hard_mode_blocks_low_hss(self):
        """HARD mode blocks when HSS is below threshold."""
        low_hss_result = MockTDAResult(hss=0.05)
        decision = evaluate_hard_gate_decision(low_hss_result, TDAHardGateMode.HARD)

        assert decision.should_block is True
        assert decision.mode == TDAHardGateMode.HARD

    def test_hard_mode_allows_high_hss(self):
        """HARD mode allows when HSS is above threshold."""
        high_hss_result = MockTDAResult(hss=0.8)
        decision = evaluate_hard_gate_decision(high_hss_result, TDAHardGateMode.HARD)

        assert decision.should_block is False

    def test_hard_mode_does_not_log_would_block(self):
        """HARD mode blocks directly, not via 'would block' logging."""
        low_hss_result = MockTDAResult(hss=0.05)
        decision = evaluate_hard_gate_decision(low_hss_result, TDAHardGateMode.HARD)

        # In HARD mode, we block directly, so no "would block" log
        assert decision.should_log_as_would_block is False

    def test_hard_mode_reason_includes_block_status(self):
        """HARD mode reason indicates actual block status."""
        low_result = MockTDAResult(hss=0.05)
        decision = evaluate_hard_gate_decision(low_result, TDAHardGateMode.HARD)

        assert "block=True" in decision.reason or "Hard" in decision.reason


# ============================================================================
# Test: Exception Window with HARD Mode
# ============================================================================

class TestExceptionWindowWithHardMode:
    """Tests for exception window interaction with HARD mode."""

    def test_exception_window_prevents_blocking(self):
        """Active exception window prevents blocking in HARD mode."""
        low_hss_result = MockTDAResult(hss=0.05)
        manager = ExceptionWindowManager(max_runs=10)
        manager.activate("test divergence")

        decision = evaluate_hard_gate_decision(low_hss_result, TDAHardGateMode.HARD, manager)

        assert decision.should_block is False
        assert decision.exception_window_active is True

    def test_exception_window_logs_would_block(self):
        """Exception window logs 'would block' for auditability."""
        low_hss_result = MockTDAResult(hss=0.05)
        manager = ExceptionWindowManager(max_runs=10)
        manager.activate("test divergence")

        decision = evaluate_hard_gate_decision(low_hss_result, TDAHardGateMode.HARD, manager)

        assert decision.should_log_as_would_block is True

    def test_exception_window_consumes_run(self):
        """Each decision consumes one run from exception window."""
        low_hss_result = MockTDAResult(hss=0.05)
        manager = ExceptionWindowManager(max_runs=3)
        manager.activate("test")

        assert manager.runs_remaining == 3

        evaluate_hard_gate_decision(low_hss_result, TDAHardGateMode.HARD, manager)
        assert manager.runs_remaining == 2

        evaluate_hard_gate_decision(low_hss_result, TDAHardGateMode.HARD, manager)
        assert manager.runs_remaining == 1

    def test_exception_window_expires_after_max_runs(self):
        """Exception window expires and blocking resumes."""
        low_hss_result = MockTDAResult(hss=0.05)
        manager = ExceptionWindowManager(max_runs=2)
        manager.activate("test")

        # First two runs: exception window active
        decision1 = evaluate_hard_gate_decision(low_hss_result, TDAHardGateMode.HARD, manager)
        assert decision1.should_block is False

        decision2 = evaluate_hard_gate_decision(low_hss_result, TDAHardGateMode.HARD, manager)
        assert decision2.should_block is False

        # Third run: window exhausted, blocking resumes
        decision3 = evaluate_hard_gate_decision(low_hss_result, TDAHardGateMode.HARD, manager)
        assert decision3.should_block is True
        assert decision3.exception_window_active is False

    def test_exception_window_not_consumed_for_high_hss(self):
        """High HSS runs still consume exception window slots."""
        high_hss_result = MockTDAResult(hss=0.8)
        manager = ExceptionWindowManager(max_runs=3)
        manager.activate("test")

        # Even high-HSS runs consume the window
        evaluate_hard_gate_decision(high_hss_result, TDAHardGateMode.HARD, manager)
        assert manager.runs_remaining == 2

    def test_inactive_exception_window_does_not_prevent_blocking(self):
        """Inactive exception window does not prevent blocking."""
        low_hss_result = MockTDAResult(hss=0.05)
        manager = ExceptionWindowManager(max_runs=10)
        # NOT activated

        decision = evaluate_hard_gate_decision(low_hss_result, TDAHardGateMode.HARD, manager)

        assert decision.should_block is True
        assert decision.exception_window_active is False


# ============================================================================
# Test: ExceptionWindowManager
# ============================================================================

class TestExceptionWindowManager:
    """Tests for ExceptionWindowManager class."""

    def test_manager_starts_inactive(self):
        """Manager starts with inactive window."""
        manager = ExceptionWindowManager(max_runs=10)
        assert manager.active is False
        assert manager.runs_remaining == 0

    def test_activate_enables_window(self):
        """Activation enables the exception window."""
        manager = ExceptionWindowManager(max_runs=5)
        result = manager.activate("test reason")

        assert result is True
        assert manager.active is True
        assert manager.runs_remaining == 5

    def test_activate_fails_with_zero_max_runs(self):
        """Cannot activate with max_runs=0."""
        manager = ExceptionWindowManager(max_runs=0)
        result = manager.activate("test")

        assert result is False
        assert manager.active is False

    def test_double_activation_fails(self):
        """Cannot activate an already active window."""
        manager = ExceptionWindowManager(max_runs=5)
        manager.activate("first")
        result = manager.activate("second")

        assert result is False

    def test_get_state_returns_complete_state(self):
        """get_state returns complete exception window state."""
        manager = ExceptionWindowManager(max_runs=5)
        manager.activate("test reason")

        state = manager.get_state()

        assert isinstance(state, ExceptionWindowState)
        assert state.active is True
        assert state.runs_remaining == 5
        assert state.total_runs == 5
        assert state.activation_reason == "test reason"
        assert state.activated_at is not None

    def test_reset_clears_state(self):
        """Reset clears all exception window state."""
        manager = ExceptionWindowManager(max_runs=5)
        manager.activate("test")
        manager.reset()

        assert manager.active is False
        assert manager.runs_remaining == 0

        state = manager.get_state()
        assert state.activation_reason is None

    def test_consume_run_decrements_remaining(self):
        """consume_run decrements remaining runs."""
        manager = ExceptionWindowManager(max_runs=3)
        manager.activate("test")

        state = manager.consume_run()
        assert state.runs_remaining == 2

        state = manager.consume_run()
        assert state.runs_remaining == 1

    def test_consume_run_deactivates_at_zero(self):
        """Exception window deactivates when runs exhausted."""
        manager = ExceptionWindowManager(max_runs=1)
        manager.activate("test")

        state = manager.consume_run()

        assert state.runs_remaining == 0
        assert manager.active is False

    def test_reads_max_runs_from_env(self):
        """Manager reads max_runs from environment variable."""
        with patch.dict(os.environ, {"MATHLEDGER_TDA_EXCEPTION_WINDOW_RUNS": "42"}):
            manager = ExceptionWindowManager()
            manager.activate("test")

            assert manager.runs_remaining == 42


# ============================================================================
# Test: Deterministic Behavior
# ============================================================================

class TestDeterministicBehavior:
    """Tests ensuring deterministic behavior across invocations."""

    def test_same_input_produces_same_decision(self):
        """Same input always produces same decision."""
        result = MockTDAResult(hss=0.15)

        decision1 = evaluate_hard_gate_decision(result, TDAHardGateMode.DRY_RUN)
        decision2 = evaluate_hard_gate_decision(result, TDAHardGateMode.DRY_RUN)

        assert decision1.should_block == decision2.should_block
        assert decision1.should_log_as_would_block == decision2.should_log_as_would_block
        assert decision1.mode == decision2.mode

    def test_decision_deterministic_across_modes(self):
        """Block determination is consistent (only enforcement varies)."""
        result = MockTDAResult(hss=0.05)

        # All modes should agree on whether it WOULD be blocked
        off_decision = evaluate_hard_gate_decision(result, TDAHardGateMode.OFF)
        shadow_decision = evaluate_hard_gate_decision(result, TDAHardGateMode.SHADOW)
        dry_run_decision = evaluate_hard_gate_decision(result, TDAHardGateMode.DRY_RUN)
        hard_decision = evaluate_hard_gate_decision(result, TDAHardGateMode.HARD)

        # DRY_RUN and HARD should agree on "would block"
        assert dry_run_decision.should_log_as_would_block == hard_decision.should_block

    def test_mode_ordering_for_severity(self):
        """Modes have clear ordering: OFF < SHADOW < DRY_RUN < HARD."""
        # This test documents the expected mode progression
        modes = [TDAHardGateMode.OFF, TDAHardGateMode.SHADOW, TDAHardGateMode.DRY_RUN, TDAHardGateMode.HARD]
        result = MockTDAResult(hss=0.05)

        blocks = []
        logs_would_block = []

        for mode in modes:
            decision = evaluate_hard_gate_decision(result, mode)
            blocks.append(decision.should_block)
            logs_would_block.append(decision.should_log_as_would_block)

        # Only HARD actually blocks
        assert blocks == [False, False, False, True]
        # Only DRY_RUN logs "would block" (HARD actually blocks, doesn't log)
        assert logs_would_block == [False, False, True, False]


# ============================================================================
# Test: Extended Health Summary (Phase IV)
# ============================================================================

class TestExtendedHealthSummary:
    """Tests for Phase IV extended health summary."""

    def test_summary_includes_hard_gate_mode(self):
        """Summary includes hard gate mode."""
        results = [MockTDAResult(hss=0.6)]
        config = MockTDAConfig()

        summary = summarize_tda_for_global_health_v2(
            results, config,
            hard_gate_mode=TDAHardGateMode.DRY_RUN
        )

        assert summary["hard_gate_mode"] == "dry_run"

    def test_summary_includes_exception_window_state(self):
        """Summary includes exception window state."""
        results = [MockTDAResult(hss=0.6)]
        config = MockTDAConfig()
        manager = ExceptionWindowManager(max_runs=10)
        manager.activate("test")

        summary = summarize_tda_for_global_health_v2(
            results, config,
            hard_gate_mode=TDAHardGateMode.HARD,
            exception_manager=manager
        )

        assert summary["hard_gate_exception_window_active"] is True
        assert summary["hard_gate_exception_runs_remaining"] == 10

    def test_summary_includes_hypothetical_blocks(self):
        """Summary includes hypothetical block statistics."""
        results = [MockTDAResult(hss=0.6)]
        config = MockTDAConfig()

        summary = summarize_tda_for_global_health_v2(
            results, config,
            hard_gate_mode=TDAHardGateMode.DRY_RUN,
            hypothetical_blocks=5
        )

        assert summary["hypothetical_block_count"] == 5
        assert summary["hypothetical_block_rate"] == 5.0  # 5 / 1 result

    def test_summary_handles_empty_results(self):
        """Summary handles empty results gracefully."""
        config = MockTDAConfig()

        summary = summarize_tda_for_global_health_v2(
            [], config,
            hard_gate_mode=TDAHardGateMode.HARD
        )

        assert summary["cycle_count"] == 0
        assert summary["hard_gate_mode"] == "hard"
        assert summary["hypothetical_block_rate"] == 0.0


# ============================================================================
# Test: Evidence Tile in Dry-Run Mode
# ============================================================================

class TestEvidenceTileGeneration:
    """Tests for evidence tile generation in various modes."""

    def test_evidence_tile_includes_mode(self):
        """Evidence tile includes operational mode."""
        summary = {
            "hard_gate_mode": "dry_run",
            "block_rate": 0.05,
            "mean_hss": 0.7,
            "hss_trend": 0.001,
            "structural_health": 0.85,
            "cycle_count": 100,
            "block_count": 5,
            "warn_count": 10,
            "ok_count": 85,
            "hypothetical_block_count": 8,
        }

        tile = build_tda_hard_gate_evidence_tile(summary)

        assert tile["mode"] == "dry_run"
        assert tile["hypothetical_block_count"] == 8

    def test_evidence_tile_uses_neutral_language(self):
        """Evidence tile uses neutral, factual language."""
        summary = {
            "hard_gate_mode": "hard",
            "block_rate": 0.5,  # High block rate
            "mean_hss": 0.2,    # Low HSS
            "hss_trend": -0.05,
            "structural_health": 0.1,
            "cycle_count": 10,
            "block_count": 5,
            "warn_count": 2,
            "ok_count": 3,
        }

        tile = build_tda_hard_gate_evidence_tile(summary)

        # Tile should NOT contain normative words
        tile_str = str(tile)
        assert "good" not in tile_str.lower()
        assert "bad" not in tile_str.lower()
        assert "excellent" not in tile_str.lower()
        assert "poor" not in tile_str.lower()

    def test_evidence_tile_is_deterministic(self):
        """Same input produces identical evidence tile."""
        summary = {
            "hard_gate_mode": "hard",
            "block_rate": 0.1,
            "mean_hss": 0.6,
            "hss_trend": 0.002,
            "structural_health": 0.75,
            "cycle_count": 50,
            "block_count": 5,
            "warn_count": 5,
            "ok_count": 40,
        }

        tile1 = build_tda_hard_gate_evidence_tile(summary)
        tile2 = build_tda_hard_gate_evidence_tile(summary)

        assert tile1 == tile2

    def test_evidence_tile_has_schema_version(self):
        """Evidence tile includes schema version for compatibility."""
        summary = {
            "hard_gate_mode": "hard",
            "block_rate": 0.0,
            "mean_hss": 0.9,
        }

        tile = build_tda_hard_gate_evidence_tile(summary)

        assert "schema_version" in tile
        assert tile["schema_version"].startswith("tda-evidence-tile-")

    def test_evidence_tile_rounds_floats(self):
        """Evidence tile rounds floats for determinism."""
        summary = {
            "hard_gate_mode": "hard",
            "block_rate": 0.123456789,
            "mean_hss": 0.987654321,
            "hss_trend": 0.000123456789,
            "structural_health": 0.555555555,
            "cycle_count": 100,
        }

        tile = build_tda_hard_gate_evidence_tile(summary)

        # Verify rounding
        assert tile["block_rate"] == 0.1235
        assert tile["mean_hss"] == 0.9877
        assert tile["hss_trend"] == 0.000123
        assert tile["structural_health"] == 0.5556


# ============================================================================
# Test: HardGateDecision Serialization
# ============================================================================

class TestHardGateDecisionSerialization:
    """Tests for HardGateDecision serialization."""

    def test_decision_to_dict(self):
        """Decision can be serialized to dictionary."""
        decision = HardGateDecision(
            should_block=True,
            should_log_as_would_block=False,
            mode=TDAHardGateMode.HARD,
            exception_window_active=False,
            reason="test reason"
        )

        d = decision.to_dict()

        assert d["should_block"] is True
        assert d["mode"] == TDAHardGateMode.HARD
        assert d["reason"] == "test reason"

    def test_decision_dict_is_json_serializable(self):
        """Decision dict can be JSON serialized."""
        import json

        result = MockTDAResult(hss=0.1)
        decision = evaluate_hard_gate_decision(result, TDAHardGateMode.HARD)

        # Convert enum to string for JSON
        d = decision.to_dict()
        d["mode"] = d["mode"].value

        # Should not raise
        json_str = json.dumps(d)
        assert isinstance(json_str, str)


# ============================================================================
# Test: Edge Cases
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_hss_exactly_at_threshold(self):
        """HSS exactly at threshold is not blocked (threshold is exclusive)."""
        # Create result where HSS equals threshold
        result = MockTDAResult(hss=0.2)
        result.block = False  # At threshold, not below

        decision = evaluate_hard_gate_decision(result, TDAHardGateMode.HARD)

        # At threshold = not blocked (threshold is lower bound for OK)
        assert decision.should_block is False

    def test_hss_just_below_threshold(self):
        """HSS just below threshold is blocked."""
        result = MockTDAResult(hss=0.199)
        result.block = True  # Below threshold

        decision = evaluate_hard_gate_decision(result, TDAHardGateMode.HARD)

        assert decision.should_block is True

    def test_zero_hss(self):
        """Zero HSS is handled correctly."""
        result = MockTDAResult(hss=0.0)
        result.block = True

        decision = evaluate_hard_gate_decision(result, TDAHardGateMode.HARD)

        assert decision.should_block is True

    def test_maximum_hss(self):
        """Maximum HSS (1.0) is handled correctly."""
        result = MockTDAResult(hss=1.0)
        result.block = False

        decision = evaluate_hard_gate_decision(result, TDAHardGateMode.HARD)

        assert decision.should_block is False

    def test_none_exception_manager(self):
        """None exception manager is handled gracefully."""
        result = MockTDAResult(hss=0.1)
        result.block = True

        decision = evaluate_hard_gate_decision(result, TDAHardGateMode.HARD, None)

        assert decision.should_block is True
        assert decision.exception_window_active is False
