"""
TDA Governance Hook for Runner Integration — Phase VII NEURAL LINK

Operation CORTEX: Phase VII Integration Layer
=============================================

This module provides the TDAGovernanceHook class for integrating TDA
governance into experiment runners (RFLRunner, U2Runner).

The hook provides:
1. TDAMonitorResult construction from cycle telemetry
2. Hard gate decision evaluation via evaluate_hard_gate_decision()
3. Decision accumulation for snapshot building
4. Deterministic serialization for audit trails

Usage:
    from backend.tda.runner_hook import TDAGovernanceHook, TDAMonitorResult
    from backend.tda.governance import TDAHardGateMode, ExceptionWindowManager

    hook = TDAGovernanceHook(
        mode=TDAHardGateMode.SHADOW,
        exception_manager=ExceptionWindowManager(),
    )

    # In runner cycle loop:
    decision = hook.on_cycle_complete(
        cycle_index=42,
        success=True,
        cycle_result={"hss": 0.85, ...},
        telemetry={"abstention_rate": 0.02, ...},
    )

    if decision.should_block:
        # Abort cycle, return ABANDONED_TDA
        ...

    # At session end:
    snapshot = hook.build_snapshot()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, TYPE_CHECKING

from .governance import (
    TDAHardGateMode,
    ExceptionWindowManager,
    HardGateDecision,
    evaluate_hard_gate_decision,
    summarize_tda_for_global_health_v2,
    build_tda_hard_gate_evidence_tile,
)

if TYPE_CHECKING:
    from .governance_console import TDAGovernanceSnapshot

logger = logging.getLogger(__name__)

# Schema version for runner hook output
TDA_RUNNER_HOOK_SCHEMA_VERSION = "1.0.0"


# ============================================================================
# TDAMonitorResult — Minimal TDA result for governance evaluation
# ============================================================================

@dataclass(frozen=True)
class TDAMonitorResult:
    """
    Minimal TDA result structure for hard gate evaluation.

    This is the input to evaluate_hard_gate_decision(). Fields map to
    the TDA metrics computed during cycle evaluation.

    Attributes:
        cycle_id: Unique cycle identifier (typically the cycle index).
        hss: Hallucination Stability Score [0, 1]. Primary gating metric.
        sns: Structural Novelty Score [0, 1].
        pcs: Proof Coherence Score [0, 1].
        drs: Deviation Risk Score [0, 1]. Higher = more deviation.
        timestamp: ISO8601 UTC timestamp of evaluation.
        metadata: Additional cycle-specific metadata.
    """
    cycle_id: int
    hss: float
    sns: float
    pcs: float
    drs: float
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def block(self) -> bool:
        """
        Whether this result would trigger a block at default threshold.

        Default threshold is 0.2 (HSS < 0.2 triggers block).
        """
        return self.hss < 0.2

    @property
    def warn(self) -> bool:
        """
        Whether this result would trigger a warning.

        Default warning threshold is 0.4 (HSS < 0.4 triggers warn).
        """
        return self.hss < 0.4

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "cycle_id": self.cycle_id,
            "hss": round(self.hss, 6),
            "sns": round(self.sns, 6),
            "pcs": round(self.pcs, 6),
            "drs": round(self.drs, 6),
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


# ============================================================================
# TDAGovernanceHook — Runner integration hook
# ============================================================================

@dataclass
class TDAGovernanceHook:
    """
    Hook for integrating TDA governance into experiment runners.

    This hook is registered with a runner and invoked after each cycle
    completes. It evaluates the hard gate decision and accumulates
    results for session-level reporting.

    Attributes:
        mode: Hard gate operational mode (OFF, SHADOW, DRY_RUN, HARD).
        exception_manager: Manager for exception window state.
        decisions: Accumulated hard gate decisions.
        block_count: Count of cycles that were actually blocked.

    Thread Safety:
        This class is NOT thread-safe. Use one hook per runner instance.
    """
    mode: TDAHardGateMode
    exception_manager: ExceptionWindowManager
    decisions: List[HardGateDecision] = field(default_factory=list)
    block_count: int = 0

    # Internal: accumulated TDA results for snapshot building
    _results: List[TDAMonitorResult] = field(default_factory=list, repr=False)
    _would_block_count: int = field(default=0, repr=False)

    def on_cycle_complete(
        self,
        cycle_index: int,
        success: bool,
        cycle_result: Mapping[str, Any],
        telemetry: Mapping[str, Any],
    ) -> HardGateDecision:
        """
        Process cycle completion and evaluate hard gate.

        This is the BLOCKING CALL in the runner's inner loop. The decision
        returned determines whether the cycle proceeds or is abandoned.

        Args:
            cycle_index: Zero-based cycle index.
            success: Whether the cycle succeeded (before TDA evaluation).
            cycle_result: Dict containing cycle outputs. Expected keys:
                - hss: float (required for gating)
                - sns: float (optional, defaults to 0.0)
                - pcs: float (optional, defaults to 0.0)
                - drs: float (optional, defaults to 0.0)
            telemetry: Dict containing cycle telemetry. Keys vary by runner.

        Returns:
            HardGateDecision indicating:
            - should_block: True if cycle should be abandoned
            - should_log_as_would_block: True if would have blocked (dry-run)
            - mode: The mode that was applied
            - exception_window_active: Whether exception window affected decision
            - reason: Human-readable explanation
        """
        # Build TDAMonitorResult from cycle data
        tda_result = self._build_tda_result(cycle_index, cycle_result, telemetry)
        self._results.append(tda_result)

        # Evaluate hard gate decision
        decision = evaluate_hard_gate_decision(
            tda_result=tda_result,
            mode=self.mode,
            exception_manager=self.exception_manager,
        )

        # Track decision
        self.decisions.append(decision)

        # Update counters
        if decision.should_block:
            self.block_count += 1
            logger.warning(
                "[TDA-GATE] Cycle %d BLOCKED: HSS=%.4f, mode=%s, reason=%s",
                cycle_index,
                tda_result.hss,
                self.mode.value,
                decision.reason,
            )
        elif decision.should_log_as_would_block:
            self._would_block_count += 1
            logger.info(
                "[TDA-GATE] Cycle %d WOULD_BLOCK (dry-run): HSS=%.4f, mode=%s",
                cycle_index,
                tda_result.hss,
                self.mode.value,
            )

        return decision

    def _build_tda_result(
        self,
        cycle_index: int,
        cycle_result: Mapping[str, Any],
        telemetry: Mapping[str, Any],
    ) -> TDAMonitorResult:
        """
        Build TDAMonitorResult from cycle data.

        Extracts HSS/SNS/PCS/DRS from cycle_result with fallbacks.
        Missing metrics default to 0.0 with a metadata flag.
        """
        # Extract TDA metrics with fallbacks
        hss = float(cycle_result.get("hss", 0.0))
        sns = float(cycle_result.get("sns", 0.0))
        pcs = float(cycle_result.get("pcs", 0.0))
        drs = float(cycle_result.get("drs", 0.0))

        # Check if TDA data is missing
        tda_missing = "hss" not in cycle_result

        # Build metadata
        metadata: Dict[str, Any] = {
            "tda_missing": tda_missing,
            "cycle_success": telemetry.get("success", True),
        }

        # Include available telemetry fields
        for key in ["abstention_rate", "abstention_mass", "composite_root",
                    "slice_name", "ht_root", "proof_outcome"]:
            if key in telemetry:
                metadata[key] = telemetry[key]

        # Generate timestamp
        timestamp = datetime.now(timezone.utc).isoformat()

        return TDAMonitorResult(
            cycle_id=cycle_index,
            hss=hss,
            sns=sns,
            pcs=pcs,
            drs=drs,
            timestamp=timestamp,
            metadata=metadata,
        )

    def build_snapshot(self) -> "TDAGovernanceSnapshot":
        """
        Build governance console snapshot from accumulated results.

        Returns:
            TDAGovernanceSnapshot for reporting and global health integration.

        Note:
            This imports governance_console lazily to avoid circular imports.
        """
        from .governance_console import build_governance_console_snapshot

        return build_governance_console_snapshot(
            tda_results=self._results,
            hard_gate_decisions=[d.to_dict() for d in self.decisions],
            golden_state=None,  # Golden state loaded separately
            exception_manager=self.exception_manager,
            mode=self.mode,
        )

    def get_block_count(self) -> int:
        """Return number of cycles that were actually blocked."""
        return self.block_count

    def get_would_block_count(self) -> int:
        """Return number of cycles that would have been blocked (dry-run/shadow)."""
        return self._would_block_count

    def to_dict(self) -> Dict[str, Any]:
        """
        Export hook state for logging and serialization.

        Returns deterministic, JSON-serializable dictionary.
        """
        return {
            "schema_version": TDA_RUNNER_HOOK_SCHEMA_VERSION,
            "mode": self.mode.value,
            "cycle_count": len(self._results),
            "block_count": self.block_count,
            "would_block_count": self._would_block_count,
            "exception_window_active": self.exception_manager.active,
            "exception_runs_remaining": self.exception_manager.runs_remaining,
            "decisions": [d.to_dict() for d in self.decisions],
            "results": [r.to_dict() for r in self._results],
        }

    def get_summary_for_export(self) -> Dict[str, Any]:
        """
        Get compact summary for runner export.

        This is a lighter-weight alternative to to_dict() that omits
        per-cycle details, suitable for inclusion in result manifests.
        """
        # Compute aggregate metrics
        if self._results:
            hss_values = [r.hss for r in self._results]
            mean_hss = sum(hss_values) / len(hss_values)
            block_rate = self.block_count / len(self._results)
        else:
            mean_hss = 0.0
            block_rate = 0.0

        return {
            "schema_version": TDA_RUNNER_HOOK_SCHEMA_VERSION,
            "mode": self.mode.value,
            "cycle_count": len(self._results),
            "block_count": self.block_count,
            "would_block_count": self._would_block_count,
            "block_rate": round(block_rate, 6),
            "mean_hss": round(mean_hss, 6),
            "exception_window_active": self.exception_manager.active,
        }


# ============================================================================
# Factory function for creating hooks
# ============================================================================

def create_tda_hook(
    mode: Optional[TDAHardGateMode] = None,
    exception_manager: Optional[ExceptionWindowManager] = None,
) -> TDAGovernanceHook:
    """
    Factory function for creating TDAGovernanceHook instances.

    Args:
        mode: Hard gate mode. Defaults to reading from environment
              (MATHLEDGER_TDA_HARD_GATE_MODE) or SHADOW.
        exception_manager: Exception window manager. Creates new if None.

    Returns:
        Configured TDAGovernanceHook instance.
    """
    if mode is None:
        mode = TDAHardGateMode.from_env()

    if exception_manager is None:
        exception_manager = ExceptionWindowManager()

    return TDAGovernanceHook(
        mode=mode,
        exception_manager=exception_manager,
    )


__all__ = [
    "TDAMonitorResult",
    "TDAGovernanceHook",
    "create_tda_hook",
    "TDA_RUNNER_HOOK_SCHEMA_VERSION",
]
