"""
USLAIntegration — Integration layer for USLA shadow mode in runners.

Phase X: SHADOW MODE ONLY

This module provides a unified integration layer that combines:
- USLABridge: Telemetry translation
- USLAShadowLogger: Structured logging
- DivergenceMonitor: Real-time divergence tracking

SHADOW MODE CONTRACT:
1. The USLA simulator NEVER modifies real governance decisions
2. Disagreements are LOGGED, not ACTED upon
3. No cycle is blocked or allowed based on simulator output
4. The simulator runs AFTER the real governance decision
5. All USLA state is written to shadow logs only

Usage in RFL Runner:
    from backend.topology.usla_integration import USLAIntegration, RunnerType

    # In __init__:
    self.usla = USLAIntegration.create_for_runner(
        runner_type=RunnerType.RFL,
        runner_id=config.experiment_id,
        enabled=config.usla_shadow_enabled,
    )

    # In run_with_attestation, after main processing:
    if self.usla.enabled:
        self.usla.process_rfl_cycle(
            cycle=self.first_organism_runs_total,
            ledger_entry=ledger_entry,
            attestation=attestation,
            real_blocked=False,  # TDA decision
        )

Usage in U2 Runner:
    # In __init__:
    self.usla = USLAIntegration.create_for_runner(
        runner_type=RunnerType.U2,
        runner_id="u2_arithmetic",
        enabled=config.usla_shadow_enabled,
    )

    # In run_cycle, after main processing:
    if self.usla.enabled:
        self.usla.process_u2_cycle(
            cycle=cycle_number,
            cycle_result=result,
            real_blocked=False,  # TDA decision
        )
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.topology.usla_bridge import (
    BridgeConfig,
    RunnerType,
    TelemetrySnapshot,
    TranslationResult,
    USLABridge,
)
from backend.topology.usla_shadow import (
    ShadowLogConfig,
    ShadowLogEntry,
    USLAShadowLogger,
)
from backend.topology.divergence_monitor import (
    AlertSeverity,
    DivergenceAlert,
    DivergenceConfig,
    DivergenceMonitor,
)
from backend.topology.usla_simulator import USLAParams, USLAState
from backend.topology.tda_telemetry_provider import (
    TDATelemetryProvider,
    TDATelemetrySnapshot,
    TDATelemetryConfig,
)

__all__ = [
    "USLAIntegration",
    "USLAIntegrationConfig",
    "RunnerType",
]

logger = logging.getLogger("USLAIntegration")


@dataclass
class USLAIntegrationConfig:
    """Configuration for USLA integration."""
    enabled: bool = False
    log_dir: str = "results/usla_shadow"
    runner_id: str = "unknown"
    run_id: Optional[str] = None

    # Bridge config
    bridge_config: Optional[BridgeConfig] = None

    # Shadow log config
    log_every_n_cycles: int = 1
    include_full_state: bool = True

    # Divergence config
    divergence_config: Optional[DivergenceConfig] = None

    # USLA params (optional override)
    usla_params: Optional[USLAParams] = None

    @classmethod
    def from_env(cls, runner_type: RunnerType, runner_id: str) -> "USLAIntegrationConfig":
        """Create config from environment variables."""
        enabled = os.getenv("USLA_SHADOW_ENABLED", "").lower() in ("1", "true", "yes")
        log_dir = os.getenv("USLA_SHADOW_LOG_DIR", "results/usla_shadow")

        return cls(
            enabled=enabled,
            log_dir=log_dir,
            runner_id=runner_id,
        )


class USLAIntegration:
    """
    Unified integration layer for USLA shadow mode.

    Combines bridge, logger, and monitor into a single interface
    for easy integration with runners.
    """

    def __init__(
        self,
        runner_type: RunnerType,
        config: USLAIntegrationConfig,
    ):
        self.runner_type = runner_type
        self.config = config
        self._enabled = config.enabled

        if not self._enabled:
            self._bridge = None
            self._logger = None
            self._monitor = None
            self._telemetry_provider = None
            return

        # Initialize components
        bridge_config = config.bridge_config or BridgeConfig.default()
        usla_params = config.usla_params or USLAParams()

        self._bridge = USLABridge(
            runner_type=runner_type,
            config=bridge_config,
            params=usla_params,
        )

        shadow_config = ShadowLogConfig(
            log_dir=config.log_dir,
            runner_id=config.runner_id,
            run_id=config.run_id,
            log_every_n_cycles=config.log_every_n_cycles,
            include_full_state=config.include_full_state,
        )
        self._logger = USLAShadowLogger(config=shadow_config)

        divergence_config = config.divergence_config or DivergenceConfig.default()
        self._monitor = DivergenceMonitor(config=divergence_config)

        # Phase X P1: TDA Telemetry Provider for real telemetry capture
        self._telemetry_provider = TDATelemetryProvider()

        logger.info(f"[USLA] Shadow mode enabled for {runner_type.value}")
        logger.info(f"[USLA] Log file: {self._logger.file_path}")

    @classmethod
    def create_for_runner(
        cls,
        runner_type: RunnerType,
        runner_id: str,
        enabled: bool = False,
        log_dir: str = "results/usla_shadow",
        run_id: Optional[str] = None,
        usla_params: Optional[USLAParams] = None,
    ) -> "USLAIntegration":
        """
        Factory method to create integration for a specific runner.

        Args:
            runner_type: Type of runner (RFL or U2)
            runner_id: Unique identifier for this runner instance
            enabled: Whether USLA shadow mode is enabled
            log_dir: Directory for shadow logs
            run_id: Optional run ID for log file naming
            usla_params: Optional USLA parameters override

        Returns:
            Configured USLAIntegration instance
        """
        config = USLAIntegrationConfig(
            enabled=enabled,
            log_dir=log_dir,
            runner_id=runner_id,
            run_id=run_id,
            usla_params=usla_params,
        )
        return cls(runner_type=runner_type, config=config)

    @property
    def enabled(self) -> bool:
        """Check if USLA integration is enabled."""
        return self._enabled

    def process_rfl_cycle(
        self,
        cycle: int,
        ledger_entry: Any,
        attestation: Any,
        real_blocked: bool = False,
        telemetry: Optional[TelemetrySnapshot] = None,
        governance_threshold: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Process an RFL cycle through USLA shadow mode.

        SHADOW MODE: This never modifies governance decisions.

        Args:
            cycle: Cycle number
            ledger_entry: RunLedgerEntry from RFL runner
            attestation: AttestedRunContext from RFL runner
            real_blocked: The real TDA governance decision
            telemetry: Optional TDA telemetry snapshot (overrides auto-capture)
            governance_threshold: Optional real governance threshold τ

        Returns:
            Dictionary with shadow analysis results, or None if disabled
        """
        if not self._enabled:
            return None

        # Phase X P1: Capture real telemetry if not provided
        if telemetry is None and self._telemetry_provider is not None:
            tda_snapshot = self._telemetry_provider.capture_from_rfl(
                ledger_entry=ledger_entry,
                attestation=attestation,
                blocked=real_blocked,
                governance_threshold=governance_threshold,
            )
            telemetry = TelemetrySnapshot.from_tda_snapshot(tda_snapshot)

        # Extract cycle result from RFL data
        cycle_result = self._extract_rfl_cycle_result(ledger_entry, attestation)

        # Translate to CycleInput
        translation = self._bridge.translate(cycle_result, telemetry)

        # Step simulator
        state = self._bridge.step(translation.cycle_input, real_blocked)
        sim_blocked = state.blocked

        # Extract real metrics from telemetry for divergence check
        real_hss = None
        real_threshold = None
        real_rsi = None
        real_beta = None

        if hasattr(ledger_entry, 'abstention_fraction'):
            real_hss = 1.0 - ledger_entry.abstention_fraction

        if telemetry is not None:
            real_threshold = telemetry.threshold
            real_rsi = telemetry.tda_rsi
            real_beta = telemetry.tda_block_rate

        # Check for divergence
        alerts = self._monitor.check(
            cycle=cycle,
            real_blocked=real_blocked,
            sim_blocked=sim_blocked,
            real_hss=real_hss,
            sim_hss=state.H,
            real_threshold=real_threshold,
            sim_threshold=state.tau,
            real_rsi=real_rsi,
            sim_rsi=state.rho,
            real_beta=real_beta,
            sim_beta=state.beta,
        )

        # Log to shadow log
        entry = self._logger.log_cycle(
            cycle=cycle,
            state_dict=state.to_dict(),
            input_dict=translation.to_dict()["input"],
            real_blocked=real_blocked,
            sim_blocked=sim_blocked,
            hard_ok=self._bridge.is_hard_ok(),
            in_safe_region=state.is_in_safe_region(self._bridge.params),
            translation_quality=translation.source_quality,
            fallbacks_used=translation.fallbacks_used,
        )

        # Log any divergence alerts
        for alert in alerts:
            if alert.severity in (AlertSeverity.WARNING, AlertSeverity.CRITICAL):
                self._logger.log_divergence_alert(
                    cycle=cycle,
                    severity=alert.severity.value,
                    field=alert.field,
                    real_value=alert.real_value,
                    sim_value=alert.sim_value,
                    consecutive_cycles=alert.consecutive_cycles,
                )
                logger.warning(f"[USLA] {alert.severity.value}: {alert.description}")

        return {
            "cycle": cycle,
            "sim_blocked": sim_blocked,
            "governance_aligned": real_blocked == sim_blocked,
            "hard_ok": self._bridge.is_hard_ok(),
            "in_safe_region": state.is_in_safe_region(self._bridge.params),
            "alerts": [a.to_dict() for a in alerts],
            "state": state.to_dict(),
        }

    def process_u2_cycle(
        self,
        cycle: int,
        cycle_result: Dict[str, Any],
        real_blocked: bool = False,
        telemetry: Optional[TelemetrySnapshot] = None,
        governance_threshold: Optional[float] = None,
        success_history: Optional[List[bool]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Process a U2 cycle through USLA shadow mode.

        SHADOW MODE: This never modifies governance decisions.

        Args:
            cycle: Cycle number
            cycle_result: CycleResult from U2 runner (as dict or object)
            real_blocked: The real TDA governance decision
            telemetry: Optional TDA telemetry snapshot (overrides auto-capture)
            governance_threshold: Optional real governance threshold τ
            success_history: Optional success history for HSS computation

        Returns:
            Dictionary with shadow analysis results, or None if disabled
        """
        if not self._enabled:
            return None

        # Convert cycle_result to dict if needed
        if hasattr(cycle_result, '__dict__'):
            result_dict = {
                "success": getattr(cycle_result, 'success', False),
                "depth": getattr(cycle_result, 'depth', None),
                "branch_factor": getattr(cycle_result, 'branch_factor', None),
            }
        else:
            result_dict = cycle_result

        # Phase X P1: Capture real telemetry if not provided
        if telemetry is None and self._telemetry_provider is not None:
            tda_snapshot = self._telemetry_provider.capture_from_u2(
                cycle_result=cycle_result,
                blocked=real_blocked,
                governance_threshold=governance_threshold,
                success_history=success_history,
            )
            telemetry = TelemetrySnapshot.from_tda_snapshot(tda_snapshot)

        # Translate to CycleInput
        translation = self._bridge.translate(result_dict, telemetry)

        # Step simulator
        state = self._bridge.step(translation.cycle_input, real_blocked)
        sim_blocked = state.blocked

        # Extract real metrics from telemetry for divergence check
        real_threshold = None
        real_rsi = None
        real_beta = None

        if telemetry is not None:
            real_threshold = telemetry.threshold
            real_rsi = telemetry.tda_rsi
            real_beta = telemetry.tda_block_rate

        # Check for divergence
        alerts = self._monitor.check(
            cycle=cycle,
            real_blocked=real_blocked,
            sim_blocked=sim_blocked,
            sim_hss=state.H,
            real_threshold=real_threshold,
            sim_threshold=state.tau,
            real_rsi=real_rsi,
            sim_rsi=state.rho,
            real_beta=real_beta,
            sim_beta=state.beta,
        )

        # Log to shadow log
        entry = self._logger.log_cycle(
            cycle=cycle,
            state_dict=state.to_dict(),
            input_dict=translation.to_dict()["input"],
            real_blocked=real_blocked,
            sim_blocked=sim_blocked,
            hard_ok=self._bridge.is_hard_ok(),
            in_safe_region=state.is_in_safe_region(self._bridge.params),
            translation_quality=translation.source_quality,
            fallbacks_used=translation.fallbacks_used,
        )

        # Log any divergence alerts
        for alert in alerts:
            if alert.severity in (AlertSeverity.WARNING, AlertSeverity.CRITICAL):
                self._logger.log_divergence_alert(
                    cycle=cycle,
                    severity=alert.severity.value,
                    field=alert.field,
                    real_value=alert.real_value,
                    sim_value=alert.sim_value,
                    consecutive_cycles=alert.consecutive_cycles,
                )
                logger.warning(f"[USLA] {alert.severity.value}: {alert.description}")

        return {
            "cycle": cycle,
            "sim_blocked": sim_blocked,
            "governance_aligned": real_blocked == sim_blocked,
            "hard_ok": self._bridge.is_hard_ok(),
            "in_safe_region": state.is_in_safe_region(self._bridge.params),
            "alerts": [a.to_dict() for a in alerts],
            "state": state.to_dict(),
        }

    def _extract_rfl_cycle_result(
        self,
        ledger_entry: Any,
        attestation: Any,
    ) -> Dict[str, Any]:
        """Extract cycle result data from RFL structures."""
        result = {}

        # From ledger_entry
        if hasattr(ledger_entry, 'abstention_fraction'):
            result['abstention_rate'] = ledger_entry.abstention_fraction

        if hasattr(ledger_entry, 'success_rate'):
            result['success_rate'] = ledger_entry.success_rate

        # From attestation metadata
        if hasattr(attestation, 'metadata') and attestation.metadata:
            meta = attestation.metadata
            if 'max_depth' in meta:
                result['max_depth'] = meta['max_depth']
            if 'branch_factor' in meta:
                result['branch_factor'] = meta['branch_factor']
            if 'success_count' in meta:
                result['success_count'] = meta['success_count']
            if 'total_count' in meta:
                result['total_count'] = meta['total_count']
            if 'proofs' in meta:
                result['proofs'] = meta['proofs']

        # From attestation direct attributes
        if hasattr(attestation, 'abstention_rate'):
            result['abstention_rate'] = attestation.abstention_rate

        return result

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of shadow mode operation."""
        if not self._enabled:
            return {"enabled": False}

        return {
            "enabled": True,
            "runner_type": self.runner_type.value,
            "runner_id": self.config.runner_id,
            "log_file": str(self._logger.file_path) if self._logger else None,
            "logger_summary": self._logger.get_summary() if self._logger else None,
            "monitor_summary": self._monitor.get_summary() if self._monitor else None,
            "final_state": self._bridge.state.to_dict() if self._bridge else None,
        }

    def close(self) -> Dict[str, Any]:
        """
        Close integration and return final summary.

        Should be called at end of run.
        """
        summary = self.get_summary()

        if self._logger:
            self._logger.close()

        return summary

    def should_abort(self) -> bool:
        """Check if divergence monitor recommends abort."""
        if not self._enabled or not self._monitor:
            return False
        return self._monitor.should_abort

    def abort_reason(self) -> Optional[str]:
        """Get abort reason if should_abort is True."""
        if not self._enabled or not self._monitor:
            return None
        return self._monitor.abort_reason
