"""
Smoke tests for What-If Governance Engine.

Tests:
1. Invariant fail scenario (G2)
2. Omega fail scenario (G3)
3. RSI fail scenario (G4)

All tests verify HYPOTHETICAL verdicts only - no enforcement.
"""

import pytest
from datetime import datetime, timezone

from backend.governance.what_if_engine import (
    WhatIfEngine,
    WhatIfCycleInput,
    WhatIfConfig,
    WhatIfReport,
    build_what_if_report,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def default_config() -> WhatIfConfig:
    """Default configuration."""
    return WhatIfConfig.default()


@pytest.fixture
def engine(default_config) -> WhatIfEngine:
    """Fresh engine instance."""
    return WhatIfEngine(config=default_config, run_id="test-run")


@pytest.fixture
def timestamp() -> str:
    """Current timestamp."""
    return datetime.now(timezone.utc).isoformat()


# =============================================================================
# SMOKE TEST 1: INVARIANT FAIL (G2)
# =============================================================================

class TestInvariantFailScenario:
    """
    Smoke Test: G2 Invariant Gate failure.

    Scenario: Run where INV-001 (Shear Monotonicity) is violated.
    Expected: Hypothetical BLOCK at G2_INVARIANT.
    """

    def test_invariant_violation_causes_block(self, engine, timestamp):
        """Single invariant violation causes hypothetical BLOCK."""
        # Cycle with invariant violation
        input = WhatIfCycleInput(
            cycle=100,
            timestamp=timestamp,
            invariant_violations=["INV-001"],
            in_omega=True,
            omega_exit_streak=0,
            rho=0.8,
            rho_collapse_streak=0,
        )

        result = engine.evaluate_cycle(input)

        assert result.verdict == "BLOCK"
        assert result.blocking_gate == "G2_INVARIANT"
        assert result.g2_status == "FAIL"
        assert "INV-001" in result.blocking_reason

    def test_invariant_scenario_full_run(self, timestamp):
        """Full run with mid-run invariant violation."""
        engine = WhatIfEngine(run_id="invariant-test")

        # 50 healthy cycles
        for i in range(1, 51):
            input = WhatIfCycleInput(
                cycle=i,
                timestamp=timestamp,
                invariant_violations=[],
                in_omega=True,
                rho=0.8,
            )
            result = engine.evaluate_cycle(input)
            assert result.verdict == "ALLOW"

        # 10 cycles with INV-001 violation
        for i in range(51, 61):
            input = WhatIfCycleInput(
                cycle=i,
                timestamp=timestamp,
                invariant_violations=["INV-001"],
                in_omega=True,
                rho=0.8,
            )
            result = engine.evaluate_cycle(input)
            assert result.verdict == "BLOCK"
            assert result.blocking_gate == "G2_INVARIANT"

        # 40 recovery cycles
        for i in range(61, 101):
            input = WhatIfCycleInput(
                cycle=i,
                timestamp=timestamp,
                invariant_violations=[],
                in_omega=True,
                rho=0.8,
            )
            result = engine.evaluate_cycle(input)
            assert result.verdict == "ALLOW"

        # Build report
        report = engine.build_report()

        assert report.total_cycles == 100
        assert report.hypothetical_blocks == 10
        assert report.hypothetical_allows == 90
        assert report.blocking_gate_distribution.get("G2_INVARIANT") == 10
        assert report.first_hypothetical_block_cycle == 51

        # Check G2 analysis
        assert report.g2_analysis is not None
        assert report.g2_analysis.hypothetical_fail_count == 10
        assert report.g2_analysis.hypothetical_block_count == 10
        assert len(report.g2_analysis.threshold_breaches) == 1
        assert report.g2_analysis.threshold_breaches[0].cycle == 51
        assert report.g2_analysis.threshold_breaches[0].duration == 10

    def test_multiple_invariant_violations(self, engine, timestamp):
        """Multiple simultaneous invariant violations."""
        input = WhatIfCycleInput(
            cycle=200,
            timestamp=timestamp,
            invariant_violations=["INV-001", "INV-003", "INV-005"],
            in_omega=True,
            rho=0.75,
        )

        result = engine.evaluate_cycle(input)

        assert result.verdict == "BLOCK"
        assert result.blocking_gate == "G2_INVARIANT"
        assert result.g2_trigger == 3  # 3 violations


# =============================================================================
# SMOKE TEST 2: OMEGA FAIL (G3)
# =============================================================================

class TestOmegaFailScenario:
    """
    Smoke Test: G3 Safe Region Gate failure.

    Scenario: Run where state exits Omega for prolonged period.
    Expected: Hypothetical BLOCK at G3_SAFE_REGION after threshold.
    """

    def test_omega_exit_streak_causes_block(self, timestamp):
        """Prolonged Omega exit causes hypothetical BLOCK."""
        config = WhatIfConfig(omega_exit_threshold=100)
        engine = WhatIfEngine(config=config, run_id="omega-test")

        # Cycle with omega exit streak exceeding threshold
        input = WhatIfCycleInput(
            cycle=500,
            timestamp=timestamp,
            invariant_violations=[],
            in_omega=False,
            omega_exit_streak=101,  # Exceeds 100
            rho=0.7,
            rho_collapse_streak=0,
        )

        result = engine.evaluate_cycle(input)

        assert result.verdict == "BLOCK"
        assert result.blocking_gate == "G3_SAFE_REGION"
        assert result.g3_status == "FAIL"
        assert "101 cycles" in result.blocking_reason

    def test_omega_exit_below_threshold_allows(self, timestamp):
        """Omega exit below threshold still allows."""
        config = WhatIfConfig(omega_exit_threshold=100)
        engine = WhatIfEngine(config=config, run_id="omega-test")

        input = WhatIfCycleInput(
            cycle=300,
            timestamp=timestamp,
            invariant_violations=[],
            in_omega=False,
            omega_exit_streak=99,  # Below threshold
            rho=0.7,
        )

        result = engine.evaluate_cycle(input)

        assert result.verdict == "ALLOW"
        assert result.g3_status == "PASS"

    def test_omega_scenario_full_run(self, timestamp):
        """Full run with prolonged Omega exit."""
        config = WhatIfConfig(omega_exit_threshold=50)
        engine = WhatIfEngine(config=config, run_id="omega-full-test")

        # 30 cycles in Omega
        for i in range(1, 31):
            input = WhatIfCycleInput(
                cycle=i,
                timestamp=timestamp,
                in_omega=True,
                omega_exit_streak=0,
                rho=0.8,
            )
            result = engine.evaluate_cycle(input)
            assert result.verdict == "ALLOW"

        # 70 cycles outside Omega (first 50 pass, next 20 block)
        for i in range(31, 101):
            streak = i - 30  # 1, 2, 3, ..., 70
            input = WhatIfCycleInput(
                cycle=i,
                timestamp=timestamp,
                in_omega=False,
                omega_exit_streak=streak,
                rho=0.75,
            )
            result = engine.evaluate_cycle(input)

            if streak <= 50:
                assert result.verdict == "ALLOW", f"Cycle {i} should ALLOW with streak {streak}"
            else:
                assert result.verdict == "BLOCK", f"Cycle {i} should BLOCK with streak {streak}"
                assert result.blocking_gate == "G3_SAFE_REGION"

        # Build report
        report = engine.build_report()

        assert report.total_cycles == 100
        assert report.hypothetical_blocks == 20  # Cycles 81-100 (streak 51-70)
        assert report.blocking_gate_distribution.get("G3_SAFE_REGION") == 20
        assert report.first_hypothetical_block_cycle == 81

        # Check G3 analysis
        assert report.g3_analysis is not None
        assert report.g3_analysis.hypothetical_block_count == 20
        assert report.g3_analysis.peak_trigger_value == 70  # Final streak


# =============================================================================
# SMOKE TEST 3: RSI FAIL (G4)
# =============================================================================

class TestRSIFailScenario:
    """
    Smoke Test: G4 Soft Gate failure (RSI collapse).

    Scenario: Run where RSI drops below threshold for extended streak.
    Expected: Hypothetical BLOCK at G4_SOFT after streak threshold.
    """

    def test_rsi_collapse_causes_block(self, timestamp):
        """RSI collapse with sufficient streak causes hypothetical BLOCK."""
        config = WhatIfConfig(rho_min=0.4, rho_streak_threshold=10)
        engine = WhatIfEngine(config=config, run_id="rsi-test")

        # Cycle with RSI below threshold and sufficient streak
        input = WhatIfCycleInput(
            cycle=300,
            timestamp=timestamp,
            invariant_violations=[],
            in_omega=True,
            omega_exit_streak=0,
            rho=0.35,  # Below 0.4
            rho_collapse_streak=12,  # Above 10
        )

        result = engine.evaluate_cycle(input)

        assert result.verdict == "BLOCK"
        assert result.blocking_gate == "G4_SOFT"
        assert result.g4_status == "FAIL"
        assert "RSI collapse" in result.blocking_reason
        assert "0.350" in result.blocking_reason

    def test_rsi_low_but_short_streak_allows(self, timestamp):
        """Low RSI with short streak still allows."""
        config = WhatIfConfig(rho_min=0.4, rho_streak_threshold=10)
        engine = WhatIfEngine(config=config, run_id="rsi-test")

        input = WhatIfCycleInput(
            cycle=200,
            timestamp=timestamp,
            invariant_violations=[],
            in_omega=True,
            rho=0.35,  # Below threshold
            rho_collapse_streak=8,  # Below streak threshold
        )

        result = engine.evaluate_cycle(input)

        assert result.verdict == "ALLOW"
        assert result.g4_status == "PASS"

    def test_rsi_above_threshold_allows(self, timestamp):
        """RSI above threshold always allows (regardless of streak)."""
        config = WhatIfConfig(rho_min=0.4, rho_streak_threshold=10)
        engine = WhatIfEngine(config=config, run_id="rsi-test")

        input = WhatIfCycleInput(
            cycle=150,
            timestamp=timestamp,
            invariant_violations=[],
            in_omega=True,
            rho=0.45,  # Above threshold
            rho_collapse_streak=20,  # High streak doesn't matter
        )

        result = engine.evaluate_cycle(input)

        assert result.verdict == "ALLOW"
        assert result.g4_status == "PASS"

    def test_rsi_scenario_full_run(self, timestamp):
        """Full run with RSI collapse and recovery."""
        config = WhatIfConfig(rho_min=0.4, rho_streak_threshold=10)
        engine = WhatIfEngine(config=config, run_id="rsi-full-test")

        # 40 healthy cycles
        for i in range(1, 41):
            input = WhatIfCycleInput(
                cycle=i,
                timestamp=timestamp,
                in_omega=True,
                rho=0.75,
                rho_collapse_streak=0,
            )
            result = engine.evaluate_cycle(input)
            assert result.verdict == "ALLOW"

        # RSI starts collapsing - first 10 cycles below threshold but streak building
        for i in range(41, 51):
            streak = i - 40  # 1, 2, ..., 10
            input = WhatIfCycleInput(
                cycle=i,
                timestamp=timestamp,
                in_omega=True,
                rho=0.32,  # Below threshold
                rho_collapse_streak=streak,
            )
            result = engine.evaluate_cycle(input)
            # Streak 1-9 allows, streak 10+ blocks
            if streak < 10:
                assert result.verdict == "ALLOW"
            else:
                assert result.verdict == "BLOCK"
                assert result.blocking_gate == "G4_SOFT"

        # 15 more cycles in collapse (all should block)
        for i in range(51, 66):
            streak = i - 40  # 11, 12, ..., 25
            input = WhatIfCycleInput(
                cycle=i,
                timestamp=timestamp,
                in_omega=True,
                rho=0.30,
                rho_collapse_streak=streak,
            )
            result = engine.evaluate_cycle(input)
            assert result.verdict == "BLOCK"
            assert result.blocking_gate == "G4_SOFT"

        # Recovery - 35 healthy cycles
        for i in range(66, 101):
            input = WhatIfCycleInput(
                cycle=i,
                timestamp=timestamp,
                in_omega=True,
                rho=0.70,
                rho_collapse_streak=0,
            )
            result = engine.evaluate_cycle(input)
            assert result.verdict == "ALLOW"

        # Build report
        report = engine.build_report()

        assert report.total_cycles == 100
        assert report.hypothetical_blocks == 16  # Cycles 50-65
        assert report.blocking_gate_distribution.get("G4_SOFT") == 16
        assert report.first_hypothetical_block_cycle == 50

        # Check G4 analysis
        assert report.g4_analysis is not None
        assert report.g4_analysis.hypothetical_block_count == 16
        assert report.g4_analysis.peak_trigger_value is not None
        assert report.g4_analysis.peak_trigger_value["rho"] == 0.30


# =============================================================================
# REPORT GENERATION TESTS
# =============================================================================

class TestReportGeneration:
    """Tests for What-If report generation."""

    def test_report_schema_compliance(self, timestamp):
        """Report matches expected schema structure."""
        engine = WhatIfEngine(run_id="schema-test")

        # Run some cycles
        for i in range(1, 11):
            input = WhatIfCycleInput(
                cycle=i,
                timestamp=timestamp,
                in_omega=True,
                rho=0.8,
            )
            engine.evaluate_cycle(input)

        report = engine.build_report()
        report_dict = report.to_dict()

        # Check required fields
        assert report_dict["schema_version"] == "1.0.0"
        assert report_dict["run_id"] == "schema-test"
        assert report_dict["mode"] == "HYPOTHETICAL"
        assert "analysis_timestamp" in report_dict

        # Check summary structure
        summary = report_dict["summary"]
        assert "total_cycles" in summary
        assert "hypothetical_allows" in summary
        assert "hypothetical_blocks" in summary
        assert "hypothetical_block_rate" in summary
        assert "blocking_gate_distribution" in summary
        assert "max_consecutive_blocks" in summary

        # Check gate analysis structure
        gate_analysis = report_dict["gate_analysis"]
        assert "g2_invariant" in gate_analysis
        assert "g3_safe_region" in gate_analysis
        assert "g4_soft" in gate_analysis

        for gate in ["g2_invariant", "g3_safe_region", "g4_soft"]:
            ga = gate_analysis[gate]
            assert "hypothetical_fail_count" in ga
            assert "hypothetical_block_count" in ga
            assert "fail_rate" in ga
            assert "threshold_breaches" in ga

    def test_notable_events_recorded(self, timestamp):
        """First block is recorded as notable event."""
        engine = WhatIfEngine(run_id="events-test")

        # Healthy cycle
        engine.evaluate_cycle(WhatIfCycleInput(
            cycle=1, timestamp=timestamp, in_omega=True, rho=0.8
        ))

        # Block cycle
        engine.evaluate_cycle(WhatIfCycleInput(
            cycle=2, timestamp=timestamp,
            invariant_violations=["INV-001"],
            in_omega=True, rho=0.8
        ))

        report = engine.build_report()

        assert len(report.notable_events) >= 1
        first_event = report.notable_events[0]
        assert first_event.event_type == "FIRST_HYPOTHETICAL_BLOCK"
        assert first_event.cycle == 2
        assert first_event.gate_id == "G2_INVARIANT"

    def test_auditor_notes_generated(self, timestamp):
        """Auditor notes are generated."""
        engine = WhatIfEngine(run_id="notes-test")

        for i in range(1, 51):
            input = WhatIfCycleInput(
                cycle=i, timestamp=timestamp,
                in_omega=True, rho=0.8
            )
            engine.evaluate_cycle(input)

        report = engine.build_report()

        assert report.auditor_notes != ""
        assert "HYPOTHETICAL" in report.auditor_notes
        assert "SHADOW MODE" in report.auditor_notes


# =============================================================================
# CONVENIENCE FUNCTION TESTS
# =============================================================================

class TestBuildWhatIfReport:
    """Tests for build_what_if_report convenience function."""

    def test_build_from_telemetry_list(self, timestamp):
        """Build report from list of telemetry dicts."""
        telemetry = [
            {"cycle": 1, "timestamp": timestamp, "in_omega": True, "rho": 0.8},
            {"cycle": 2, "timestamp": timestamp, "in_omega": True, "rho": 0.75},
            {"cycle": 3, "timestamp": timestamp, "invariant_violations": ["INV-001"], "rho": 0.7},
            {"cycle": 4, "timestamp": timestamp, "in_omega": True, "rho": 0.72},
            {"cycle": 5, "timestamp": timestamp, "in_omega": True, "rho": 0.8},
        ]

        report = build_what_if_report(telemetry, run_id="telemetry-test")

        assert report.total_cycles == 5
        assert report.hypothetical_blocks == 1
        assert report.blocking_gate_distribution.get("G2_INVARIANT") == 1


# =============================================================================
# GATE PRECEDENCE TESTS
# =============================================================================

class TestGatePrecedence:
    """Tests for gate evaluation precedence (G2 > G3 > G4)."""

    def test_g2_blocks_before_g3(self, timestamp):
        """G2 failure takes precedence over G3 failure."""
        engine = WhatIfEngine()

        input = WhatIfCycleInput(
            cycle=100,
            timestamp=timestamp,
            invariant_violations=["INV-001"],  # G2 fail
            in_omega=False,
            omega_exit_streak=200,  # G3 would also fail
            rho=0.8,
        )

        result = engine.evaluate_cycle(input)

        assert result.verdict == "BLOCK"
        assert result.blocking_gate == "G2_INVARIANT"  # G2 wins
        assert result.g2_status == "FAIL"
        assert result.g3_status == "FAIL"  # Both fail, but G2 reported

    def test_g3_blocks_before_g4(self, timestamp):
        """G3 failure takes precedence over G4 failure."""
        config = WhatIfConfig(omega_exit_threshold=50, rho_min=0.4, rho_streak_threshold=5)
        engine = WhatIfEngine(config=config)

        input = WhatIfCycleInput(
            cycle=100,
            timestamp=timestamp,
            invariant_violations=[],  # G2 pass
            in_omega=False,
            omega_exit_streak=60,  # G3 fail
            rho=0.3,
            rho_collapse_streak=10,  # G4 would also fail
        )

        result = engine.evaluate_cycle(input)

        assert result.verdict == "BLOCK"
        assert result.blocking_gate == "G3_SAFE_REGION"  # G3 wins
        assert result.g2_status == "PASS"
        assert result.g3_status == "FAIL"
        assert result.g4_status == "FAIL"  # Both fail, but G3 reported
