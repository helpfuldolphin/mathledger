"""
USLABridge — Adapter bridging runner telemetry to USLA simulator.

Phase X: SHADOW MODE ONLY

This module translates real runner cycle results into CycleInput format
for the USLA simulator. It does NOT modify governance decisions.

SHADOW MODE CONTRACT:
1. The USLA simulator NEVER modifies real governance decisions
2. Disagreements are LOGGED, not ACTED upon
3. No cycle is blocked or allowed based on simulator output
4. The simulator runs AFTER the real governance decision
5. All USLA state is written to shadow logs only

Usage:
    from backend.topology.usla_bridge import USLABridge, RunnerType

    bridge = USLABridge(runner_type=RunnerType.RFL)

    # After each cycle:
    cycle_input = bridge.translate(cycle_result, telemetry)
    state = bridge.step(cycle_input, real_blocked=governance_blocked)
    bridge.log_shadow(state, real_blocked)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from backend.topology.usla_simulator import (
    CycleInput,
    USLAParams,
    USLASimulator,
    USLAState,
)

__all__ = [
    "RunnerType",
    "TelemetrySnapshot",
    "BridgeConfig",
    "USLABridge",
    "TranslationResult",
]


class RunnerType(Enum):
    """Runner type for translation semantics."""
    RFL = "rfl"
    U2 = "u2"


@dataclass
class TelemetrySnapshot:
    """
    Snapshot of telemetry data from TDA governance hook.

    This captures the real governance decision and related metrics
    for comparison with simulated state.

    Phase X P1: Extended to support real TDA telemetry via TDATelemetrySnapshot.
    """
    blocked: bool = False
    threshold: Optional[float] = None
    hss_by_depth: Optional[Dict[int, float]] = None

    # Additional TDA metrics (if available)
    tda_rsi: Optional[float] = None
    tda_block_rate: Optional[float] = None

    # TDA topology metrics (Phase X P1)
    min_cut_capacity: Optional[float] = None
    betti_numbers: Optional[List[int]] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TelemetrySnapshot":
        return cls(
            blocked=d.get("blocked", False),
            threshold=d.get("threshold"),
            hss_by_depth=d.get("hss_by_depth"),
            tda_rsi=d.get("rsi") or d.get("tda_rsi") or d.get("real_rsi"),
            tda_block_rate=d.get("block_rate") or d.get("tda_block_rate") or d.get("real_block_rate"),
            min_cut_capacity=d.get("min_cut_capacity"),
            betti_numbers=d.get("betti_numbers"),
        )

    @classmethod
    def from_tda_snapshot(cls, tda_snapshot: Any) -> "TelemetrySnapshot":
        """
        Create from TDATelemetrySnapshot.

        Args:
            tda_snapshot: TDATelemetrySnapshot instance

        Returns:
            TelemetrySnapshot with data copied from TDA snapshot
        """
        return cls(
            blocked=getattr(tda_snapshot, 'blocked', False),
            threshold=getattr(tda_snapshot, 'threshold', None),
            hss_by_depth=getattr(tda_snapshot, 'hss_by_depth', None),
            tda_rsi=getattr(tda_snapshot, 'real_rsi', None),
            tda_block_rate=getattr(tda_snapshot, 'real_block_rate', None),
            min_cut_capacity=getattr(tda_snapshot, 'min_cut_capacity', None),
            betti_numbers=getattr(tda_snapshot, 'betti_numbers', None),
        )


@dataclass
class BridgeConfig:
    """Configuration for the USLABridge."""
    # Fallback values when data is missing
    default_depth: int = 5
    default_branch_factor: float = 2.0
    default_shear: float = 0.1
    default_hss: float = 1.0

    # History for computing fallbacks
    use_history_fallbacks: bool = True
    history_window: int = 20

    # HSS computation window for U2
    u2_hss_window: int = 5

    # Shear computation settings
    shear_expected_gradient: float = 0.1

    @classmethod
    def default(cls) -> "BridgeConfig":
        return cls()


@dataclass
class TranslationResult:
    """Result of translating runner output to CycleInput."""
    cycle_input: CycleInput
    fallbacks_used: List[str] = field(default_factory=list)
    source_quality: str = "full"  # "full", "partial", "degraded"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input": {
                "hss": self.cycle_input.hss,
                "depth": self.cycle_input.depth,
                "branch_factor": self.cycle_input.branch_factor,
                "shear": self.cycle_input.shear,
                "success": self.cycle_input.success,
            },
            "fallbacks_used": self.fallbacks_used,
            "source_quality": self.source_quality,
        }


class USLABridge:
    """
    Bridge adapter connecting real runners to USLA simulator.

    SHADOW MODE ONLY: This bridge never modifies governance decisions.
    It translates runner telemetry, steps the simulator, and logs
    divergence for offline analysis.
    """

    def __init__(
        self,
        runner_type: RunnerType,
        config: Optional[BridgeConfig] = None,
        params: Optional[USLAParams] = None,
    ):
        self.runner_type = runner_type
        self.config = config or BridgeConfig.default()
        self.params = params or USLAParams()

        # Initialize simulator
        self.simulator = USLASimulator(self.params)
        self.state = USLAState.initial()

        # History tracking for fallbacks
        self._depth_history: List[int] = []
        self._bf_history: List[float] = []
        self._shear_history: List[float] = []
        self._hss_history: List[float] = []

        # U2-specific: success history for windowed HSS
        self._success_history: List[bool] = []

        # Cycle counter
        self._cycle: int = 0

    def translate_rfl(
        self,
        cycle_result: Dict[str, Any],
        telemetry: Optional[TelemetrySnapshot] = None,
    ) -> TranslationResult:
        """
        Translate RFL runner cycle result to CycleInput.

        RFL Mapping:
        - hss: 1 - abstention_rate
        - depth: max(proof.depth for proof in proofs)
        - branch_factor: mean(len(proof.parents))
        - shear: max_gradient(hss_by_depth) or from telemetry
        - success: success_count > 0
        """
        fallbacks = []

        # HSS from abstention rate
        abstention_rate = cycle_result.get("abstention_rate")
        if abstention_rate is not None:
            hss = 1.0 - abstention_rate
        elif "success_count" in cycle_result and "total_count" in cycle_result:
            total = cycle_result["total_count"]
            success = cycle_result["success_count"]
            hss = success / total if total > 0 else self.config.default_hss
        else:
            hss = self._fallback_hss()
            fallbacks.append("hss")

        # Depth from proofs or direct field
        depth = cycle_result.get("max_depth")
        if depth is None:
            proofs = cycle_result.get("proofs", [])
            if proofs:
                depth = max(p.get("depth", 0) for p in proofs)
            else:
                depth = self._fallback_depth()
                fallbacks.append("depth")

        # Branch factor from proofs
        branch_factor = cycle_result.get("branch_factor")
        if branch_factor is None:
            proofs = cycle_result.get("proofs", [])
            if proofs:
                parent_counts = [len(p.get("parents", [])) for p in proofs]
                branch_factor = sum(parent_counts) / len(parent_counts) if parent_counts else self.config.default_branch_factor
            else:
                branch_factor = self._fallback_bf()
                fallbacks.append("branch_factor")

        # Shear from telemetry hss_by_depth or estimate
        shear = None
        if telemetry and telemetry.hss_by_depth:
            shear = self._compute_shear_from_hss_by_depth(telemetry.hss_by_depth)
        if shear is None:
            shear = self._fallback_shear()
            if "shear" not in fallbacks:
                fallbacks.append("shear")

        # Success flag
        success = cycle_result.get("success_count", 0) > 0

        # Update history
        self._update_history(hss, depth, branch_factor, shear)

        cycle_input = CycleInput(
            hss=hss,
            depth=depth,
            branch_factor=branch_factor,
            shear=shear,
            success=success,
        )

        quality = "full" if not fallbacks else ("partial" if len(fallbacks) <= 2 else "degraded")

        return TranslationResult(
            cycle_input=cycle_input,
            fallbacks_used=fallbacks,
            source_quality=quality,
        )

    def translate_u2(
        self,
        cycle_result: Dict[str, Any],
        telemetry: Optional[TelemetrySnapshot] = None,
    ) -> TranslationResult:
        """
        Translate U2 runner cycle result to CycleInput.

        U2 Mapping:
        - hss: windowed_success_rate(window=5)
        - depth: cycle_result.depth
        - branch_factor: cycle_result.branch_factor (optional)
        - shear: same as RFL
        - success: cycle_result.success
        """
        fallbacks = []

        # Success for this cycle
        success = cycle_result.get("success", False)
        self._success_history.append(success)

        # Windowed HSS for U2
        window = self.config.u2_hss_window
        recent_successes = self._success_history[-window:]
        if recent_successes:
            hss = sum(1 for s in recent_successes if s) / len(recent_successes)
        else:
            hss = self.config.default_hss
            fallbacks.append("hss")

        # Depth from cycle result
        depth = cycle_result.get("depth")
        if depth is None:
            depth = self._fallback_depth()
            fallbacks.append("depth")

        # Branch factor (often not available in U2)
        branch_factor = cycle_result.get("branch_factor")
        if branch_factor is None:
            branch_factor = self._fallback_bf()
            fallbacks.append("branch_factor")

        # Shear
        shear = None
        if telemetry and telemetry.hss_by_depth:
            shear = self._compute_shear_from_hss_by_depth(telemetry.hss_by_depth)
        if shear is None:
            shear = self._fallback_shear()
            if "shear" not in fallbacks:
                fallbacks.append("shear")

        # Update history
        self._update_history(hss, depth, branch_factor, shear)

        cycle_input = CycleInput(
            hss=hss,
            depth=depth,
            branch_factor=branch_factor,
            shear=shear,
            success=success,
        )

        quality = "full" if not fallbacks else ("partial" if len(fallbacks) <= 2 else "degraded")

        return TranslationResult(
            cycle_input=cycle_input,
            fallbacks_used=fallbacks,
            source_quality=quality,
        )

    def translate(
        self,
        cycle_result: Dict[str, Any],
        telemetry: Optional[TelemetrySnapshot] = None,
    ) -> TranslationResult:
        """
        Translate cycle result based on runner type.

        Dispatches to translate_rfl or translate_u2.
        """
        if self.runner_type == RunnerType.RFL:
            return self.translate_rfl(cycle_result, telemetry)
        elif self.runner_type == RunnerType.U2:
            return self.translate_u2(cycle_result, telemetry)
        else:
            raise ValueError(f"Unknown runner type: {self.runner_type}")

    def step(
        self,
        cycle_input: CycleInput,
        real_blocked: bool = False,
    ) -> USLAState:
        """
        Step the simulator forward and return new state.

        SHADOW MODE: real_blocked is recorded for divergence tracking
        but never modified.
        """
        self._cycle += 1
        prev_state = self.state

        # Step simulator
        self.state = self.simulator.step(prev_state, cycle_input)
        self.state.cycle = self._cycle

        return self.state

    def get_governance_decision(self) -> bool:
        """
        Get the simulated governance decision (BLOCK or ALLOW).

        SHADOW MODE: This is for comparison only, never for action.
        """
        return self.state.blocked

    def is_hard_ok(self) -> bool:
        """
        Check if system would be in HARD mode.

        HARD_OK ⟺ (x ∈ Ω) ∧ (I(x)) ∧ (D(x) = ∅) ∧ (ρ ≥ ρ_min)
        """
        return self.simulator.is_hard_ok(self.state)

    def get_state_snapshot(self) -> Dict[str, Any]:
        """
        Get current state as dictionary for logging.
        """
        return {
            "cycle": self._cycle,
            "state": self.state.to_dict(),
            "hard_ok": self.is_hard_ok(),
            "in_safe_region": self.state.is_in_safe_region(self.params),
        }

    def reset(self) -> None:
        """
        Reset bridge to initial state.
        """
        self.simulator = USLASimulator(self.params)
        self.state = USLAState.initial()
        self._depth_history.clear()
        self._bf_history.clear()
        self._shear_history.clear()
        self._hss_history.clear()
        self._success_history.clear()
        self._cycle = 0

    # ───────────────────────────────────────────────────────────────────
    # Private helpers
    # ───────────────────────────────────────────────────────────────────

    def _update_history(
        self,
        hss: float,
        depth: int,
        branch_factor: float,
        shear: float,
    ) -> None:
        """Update history buffers for fallback computation."""
        window = self.config.history_window

        self._hss_history.append(hss)
        self._depth_history.append(depth)
        self._bf_history.append(branch_factor)
        self._shear_history.append(shear)

        # Trim to window
        if len(self._hss_history) > window:
            self._hss_history = self._hss_history[-window:]
        if len(self._depth_history) > window:
            self._depth_history = self._depth_history[-window:]
        if len(self._bf_history) > window:
            self._bf_history = self._bf_history[-window:]
        if len(self._shear_history) > window:
            self._shear_history = self._shear_history[-window:]

    def _fallback_hss(self) -> float:
        """Compute fallback HSS from history or default."""
        if self.config.use_history_fallbacks and self._hss_history:
            return sum(self._hss_history) / len(self._hss_history)
        return self.config.default_hss

    def _fallback_depth(self) -> int:
        """Compute fallback depth from history or default."""
        if self.config.use_history_fallbacks and self._depth_history:
            return int(sum(self._depth_history) / len(self._depth_history))
        return self.config.default_depth

    def _fallback_bf(self) -> float:
        """Compute fallback branch factor from history or default."""
        if self.config.use_history_fallbacks and self._bf_history:
            return sum(self._bf_history) / len(self._bf_history)
        return self.config.default_branch_factor

    def _fallback_shear(self) -> float:
        """Compute fallback shear from history variance or default."""
        if self.config.use_history_fallbacks and len(self._hss_history) >= 2:
            # Use HSS variance as shear proxy
            mean_hss = sum(self._hss_history) / len(self._hss_history)
            variance = sum((h - mean_hss) ** 2 for h in self._hss_history) / len(self._hss_history)
            return min(1.0, variance * 10)  # Scale variance to [0, 1]
        return self.config.default_shear

    def _compute_shear_from_hss_by_depth(
        self,
        hss_by_depth: Dict[int, float],
    ) -> Optional[float]:
        """
        Compute shear from depth-stratified HSS.

        S = max(|hss[d+1] - hss[d]|) / expected_gradient
        """
        if not hss_by_depth or len(hss_by_depth) < 2:
            return None

        depths = sorted(hss_by_depth.keys())
        max_gradient = 0.0

        for i in range(len(depths) - 1):
            d1, d2 = depths[i], depths[i + 1]
            gradient = abs(hss_by_depth[d2] - hss_by_depth[d1])
            max_gradient = max(max_gradient, gradient)

        shear = max_gradient / self.config.shear_expected_gradient
        return min(1.0, shear)
