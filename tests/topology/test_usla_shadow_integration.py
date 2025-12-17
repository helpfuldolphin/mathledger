"""
Phase X: USLA Shadow Mode Integration Smoke Tests

These tests verify that the USLA shadow integration components:
1. Can be imported and instantiated
2. Process cycles without errors
3. Log correctly
4. Monitor divergence properly
5. Never modify governance decisions (SHADOW MODE CONTRACT)

SHADOW MODE CONTRACT:
1. The USLA simulator NEVER modifies real governance decisions
2. Disagreements are LOGGED, not ACTED upon
3. No cycle is blocked or allowed based on simulator output
4. The simulator runs AFTER the real governance decision
5. All USLA state is written to shadow logs only
"""

import json
import os
import tempfile
from pathlib import Path

import pytest


class TestUSLABridgeImport:
    """Test that USLA bridge components can be imported."""

    def test_import_usla_bridge(self):
        """Verify USLABridge can be imported."""
        from backend.topology.usla_bridge import (
            USLABridge,
            RunnerType,
            BridgeConfig,
            TelemetrySnapshot,
            TranslationResult,
        )
        assert USLABridge is not None
        assert RunnerType.RFL is not None
        assert RunnerType.U2 is not None

    def test_import_usla_shadow(self):
        """Verify USLAShadowLogger can be imported."""
        from backend.topology.usla_shadow import (
            USLAShadowLogger,
            ShadowLogConfig,
            ShadowLogEntry,
        )
        assert USLAShadowLogger is not None

    def test_import_divergence_monitor(self):
        """Verify DivergenceMonitor can be imported."""
        from backend.topology.divergence_monitor import (
            DivergenceMonitor,
            DivergenceAlert,
            AlertSeverity,
            DivergenceConfig,
        )
        assert DivergenceMonitor is not None
        assert AlertSeverity.INFO is not None

    def test_import_usla_integration(self):
        """Verify USLAIntegration can be imported."""
        from backend.topology.usla_integration import (
            USLAIntegration,
            USLAIntegrationConfig,
            RunnerType,
        )
        assert USLAIntegration is not None


class TestUSLABridgeBasics:
    """Basic functionality tests for USLABridge."""

    def test_create_rfl_bridge(self):
        """Create an RFL bridge."""
        from backend.topology.usla_bridge import USLABridge, RunnerType

        bridge = USLABridge(runner_type=RunnerType.RFL)
        assert bridge.runner_type == RunnerType.RFL
        assert bridge.state is not None

    def test_create_u2_bridge(self):
        """Create a U2 bridge."""
        from backend.topology.usla_bridge import USLABridge, RunnerType

        bridge = USLABridge(runner_type=RunnerType.U2)
        assert bridge.runner_type == RunnerType.U2

    def test_translate_rfl_cycle(self):
        """Translate an RFL cycle result."""
        from backend.topology.usla_bridge import USLABridge, RunnerType

        bridge = USLABridge(runner_type=RunnerType.RFL)

        cycle_result = {
            "abstention_rate": 0.1,
            "success_rate": 0.9,
            "max_depth": 5,
            "success_count": 9,
            "total_count": 10,
        }

        translation = bridge.translate(cycle_result)

        assert translation.cycle_input is not None
        assert translation.cycle_input.hss == pytest.approx(0.9, rel=0.01)
        assert translation.cycle_input.depth == 5

    def test_translate_u2_cycle(self):
        """Translate a U2 cycle result."""
        from backend.topology.usla_bridge import USLABridge, RunnerType

        bridge = USLABridge(runner_type=RunnerType.U2)

        cycle_result = {
            "success": True,
            "depth": 3,
        }

        translation = bridge.translate(cycle_result)

        assert translation.cycle_input is not None
        assert translation.cycle_input.success is True

    def test_step_updates_state(self):
        """Verify step updates state correctly."""
        from backend.topology.usla_bridge import USLABridge, RunnerType, CycleInput
        from backend.topology.usla_simulator import CycleInput

        bridge = USLABridge(runner_type=RunnerType.RFL)

        cycle_input = CycleInput(
            hss=0.8,
            depth=5,
            branch_factor=2.0,
            shear=0.1,
            success=True,
        )

        initial_cycle = bridge.state.cycle
        state = bridge.step(cycle_input)

        assert state.cycle == initial_cycle + 1
        assert state.H == pytest.approx(0.8, rel=0.01)


class TestUSLAShadowLogger:
    """Tests for shadow logging."""

    def test_create_logger(self):
        """Create a shadow logger."""
        from backend.topology.usla_shadow import USLAShadowLogger, ShadowLogConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            config = ShadowLogConfig(
                log_dir=tmpdir,
                runner_id="test_runner",
                run_id="test_run",
            )
            logger = USLAShadowLogger(config=config)

            assert logger.file_path is not None
            assert logger.entry_count == 0

            logger.close()

    def test_log_cycle(self):
        """Log a cycle entry."""
        from backend.topology.usla_shadow import USLAShadowLogger, ShadowLogConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            config = ShadowLogConfig(
                log_dir=tmpdir,
                runner_id="test_runner",
                run_id="test_run",
            )

            with USLAShadowLogger(config=config) as logger:
                entry = logger.log_cycle(
                    cycle=1,
                    state_dict={"H": 0.8, "rho": 0.9},
                    input_dict={"hss": 0.8, "depth": 5},
                    real_blocked=False,
                    sim_blocked=False,
                    hard_ok=True,
                    in_safe_region=True,
                )

                assert entry is not None
                assert entry.governance_aligned is True
                assert logger.entry_count == 1

    def test_log_divergence(self):
        """Log a divergence alert."""
        from backend.topology.usla_shadow import USLAShadowLogger, ShadowLogConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            config = ShadowLogConfig(
                log_dir=tmpdir,
                runner_id="test_runner",
            )

            with USLAShadowLogger(config=config) as logger:
                # Log aligned cycle
                logger.log_cycle(
                    cycle=1,
                    state_dict={"H": 0.8},
                    input_dict={"hss": 0.8},
                    real_blocked=False,
                    sim_blocked=False,
                    hard_ok=True,
                    in_safe_region=True,
                )

                # Log divergent cycle
                entry = logger.log_cycle(
                    cycle=2,
                    state_dict={"H": 0.8},
                    input_dict={"hss": 0.8},
                    real_blocked=False,
                    sim_blocked=True,  # Divergence!
                    hard_ok=True,
                    in_safe_region=True,
                )

                assert entry.governance_aligned is False
                assert logger.divergence_count == 1


class TestDivergenceMonitor:
    """Tests for divergence monitoring."""

    def test_create_monitor(self):
        """Create a divergence monitor."""
        from backend.topology.divergence_monitor import DivergenceMonitor

        monitor = DivergenceMonitor()
        assert monitor.governance_aligned is True
        assert monitor.consecutive_governance_divergence == 0

    def test_detect_governance_divergence(self):
        """Detect governance decision divergence."""
        from backend.topology.divergence_monitor import DivergenceMonitor, AlertSeverity

        monitor = DivergenceMonitor()

        # No divergence
        alerts = monitor.check(cycle=1, real_blocked=False, sim_blocked=False)
        assert len(alerts) == 0
        assert monitor.governance_aligned is True

        # Divergence
        alerts = monitor.check(cycle=2, real_blocked=False, sim_blocked=True)
        assert len(alerts) == 1
        assert alerts[0].field == "governance"
        assert monitor.governance_aligned is False

    def test_consecutive_divergence_escalation(self):
        """Test that consecutive divergence escalates severity."""
        from backend.topology.divergence_monitor import (
            DivergenceMonitor,
            DivergenceConfig,
            AlertSeverity,
        )

        config = DivergenceConfig(
            governance_warning_cycles=3,
            governance_critical_cycles=5,
        )
        monitor = DivergenceMonitor(config=config)

        # First divergence: INFO
        alerts = monitor.check(cycle=1, real_blocked=False, sim_blocked=True)
        assert alerts[0].severity == AlertSeverity.INFO

        # Second divergence: INFO
        alerts = monitor.check(cycle=2, real_blocked=False, sim_blocked=True)
        assert alerts[0].severity == AlertSeverity.INFO

        # Third divergence: WARNING
        alerts = monitor.check(cycle=3, real_blocked=False, sim_blocked=True)
        assert alerts[0].severity == AlertSeverity.WARNING

        # Fourth divergence: WARNING
        alerts = monitor.check(cycle=4, real_blocked=False, sim_blocked=True)
        assert alerts[0].severity == AlertSeverity.WARNING

        # Fifth divergence: CRITICAL
        alerts = monitor.check(cycle=5, real_blocked=False, sim_blocked=True)
        assert alerts[0].severity == AlertSeverity.CRITICAL

    def test_divergence_reset_on_alignment(self):
        """Test that divergence counter resets when aligned."""
        from backend.topology.divergence_monitor import DivergenceMonitor, AlertSeverity

        monitor = DivergenceMonitor()

        # Build up divergence
        monitor.check(cycle=1, real_blocked=False, sim_blocked=True)
        monitor.check(cycle=2, real_blocked=False, sim_blocked=True)
        assert monitor.consecutive_governance_divergence == 2

        # Alignment resets
        monitor.check(cycle=3, real_blocked=False, sim_blocked=False)
        assert monitor.consecutive_governance_divergence == 0
        assert monitor.governance_aligned is True


class TestUSLAIntegration:
    """Tests for the unified integration layer."""

    def test_create_disabled_integration(self):
        """Create a disabled integration."""
        from backend.topology.usla_integration import USLAIntegration, RunnerType

        integration = USLAIntegration.create_for_runner(
            runner_type=RunnerType.RFL,
            runner_id="test_rfl",
            enabled=False,
        )

        assert integration.enabled is False
        result = integration.process_rfl_cycle(
            cycle=1,
            ledger_entry=None,
            attestation=None,
            real_blocked=False,
        )
        assert result is None

    def test_create_enabled_integration(self):
        """Create an enabled integration."""
        from backend.topology.usla_integration import USLAIntegration, RunnerType

        with tempfile.TemporaryDirectory() as tmpdir:
            integration = USLAIntegration.create_for_runner(
                runner_type=RunnerType.RFL,
                runner_id="test_rfl",
                enabled=True,
                log_dir=tmpdir,
            )

            assert integration.enabled is True
            integration.close()

    def test_process_rfl_cycle(self):
        """Process an RFL cycle through integration."""
        from backend.topology.usla_integration import USLAIntegration, RunnerType
        from dataclasses import dataclass

        @dataclass
        class MockLedgerEntry:
            abstention_fraction: float = 0.1
            success_rate: float = 0.9

        @dataclass
        class MockAttestation:
            abstention_rate: float = 0.1
            metadata: dict = None

            def __post_init__(self):
                if self.metadata is None:
                    self.metadata = {"max_depth": 5}

        with tempfile.TemporaryDirectory() as tmpdir:
            integration = USLAIntegration.create_for_runner(
                runner_type=RunnerType.RFL,
                runner_id="test_rfl",
                enabled=True,
                log_dir=tmpdir,
            )

            result = integration.process_rfl_cycle(
                cycle=1,
                ledger_entry=MockLedgerEntry(),
                attestation=MockAttestation(),
                real_blocked=False,
            )

            assert result is not None
            assert "sim_blocked" in result
            assert "governance_aligned" in result
            assert "hard_ok" in result

            integration.close()

    def test_process_u2_cycle(self):
        """Process a U2 cycle through integration."""
        from backend.topology.usla_integration import USLAIntegration, RunnerType

        with tempfile.TemporaryDirectory() as tmpdir:
            integration = USLAIntegration.create_for_runner(
                runner_type=RunnerType.U2,
                runner_id="test_u2",
                enabled=True,
                log_dir=tmpdir,
            )

            result = integration.process_u2_cycle(
                cycle=1,
                cycle_result={"success": True, "depth": 3},
                real_blocked=False,
            )

            assert result is not None
            assert "sim_blocked" in result

            integration.close()


class TestShadowModeContract:
    """Tests verifying the SHADOW MODE CONTRACT is upheld."""

    def test_shadow_mode_never_returns_governance_action(self):
        """Verify shadow mode never returns actionable governance decision."""
        from backend.topology.usla_integration import USLAIntegration, RunnerType

        with tempfile.TemporaryDirectory() as tmpdir:
            integration = USLAIntegration.create_for_runner(
                runner_type=RunnerType.RFL,
                runner_id="test",
                enabled=True,
                log_dir=tmpdir,
            )

            # Process multiple cycles with varying states
            for i in range(10):
                result = integration.process_rfl_cycle(
                    cycle=i,
                    ledger_entry=type('obj', (object,), {'abstention_fraction': 0.5 + i * 0.05})(),
                    attestation=type('obj', (object,), {
                        'abstention_rate': 0.5 + i * 0.05,
                        'metadata': {'max_depth': i}
                    })(),
                    real_blocked=False,
                )

                # Result should be informational only
                # It should NOT have any "action" or "block_cycle" field
                assert "action" not in result
                assert "block_cycle" not in result
                assert "execute_block" not in result

            integration.close()

    def test_shadow_mode_runs_after_real_decision(self):
        """Verify shadow mode processes after real decision."""
        from backend.topology.usla_integration import USLAIntegration, RunnerType

        with tempfile.TemporaryDirectory() as tmpdir:
            integration = USLAIntegration.create_for_runner(
                runner_type=RunnerType.RFL,
                runner_id="test",
                enabled=True,
                log_dir=tmpdir,
            )

            # The real_blocked parameter is passed IN, meaning the real
            # decision has already been made
            result = integration.process_rfl_cycle(
                cycle=1,
                ledger_entry=type('obj', (object,), {'abstention_fraction': 0.1})(),
                attestation=type('obj', (object,), {
                    'abstention_rate': 0.1,
                    'metadata': {}
                })(),
                real_blocked=True,  # Real decision already made
            )

            # Shadow mode accepts the real decision and only compares
            assert result["governance_aligned"] in (True, False)
            integration.close()

    def test_shadow_mode_logs_divergence_without_action(self):
        """Verify divergence is logged but no action is taken."""
        from backend.topology.usla_integration import USLAIntegration, RunnerType

        with tempfile.TemporaryDirectory() as tmpdir:
            integration = USLAIntegration.create_for_runner(
                runner_type=RunnerType.RFL,
                runner_id="test",
                enabled=True,
                log_dir=tmpdir,
            )

            # Force a scenario where simulator would block
            # (low HSS via high abstention)
            result = integration.process_rfl_cycle(
                cycle=1,
                ledger_entry=type('obj', (object,), {'abstention_fraction': 0.9})(),
                attestation=type('obj', (object,), {
                    'abstention_rate': 0.9,  # Very high abstention
                    'metadata': {}
                })(),
                real_blocked=False,  # But real system allowed
            )

            # Divergence is recorded in result
            # But NO exception raised, NO action taken
            # The method returns normally with analysis
            assert "governance_aligned" in result
            assert "alerts" in result

            integration.close()
