"""
Phase X P3: Metrics Window

This module implements the MetricsWindow helper for windowed metrics computation.
See docs/system_law/Phase_X_P3_Spec.md for full specification.

SHADOW MODE CONTRACT:
- All metrics are observational only
- No governance control or abort logic
- Used for logging and analysis purposes
- This code runs OFFLINE only, never in production governance paths

Status: P3 IMPLEMENTATION (OFFLINE, SHADOW-ONLY)

TDA Integration (PhaseX-TDA-P3):
- MetricsWindow tracks H_trajectory, rho_trajectory, success_trajectory for TDA input
- MetricsAccumulator provides get_all_windows() for TDA computation at finalize
- TDAMonitor.compute() is called by runner at window boundaries
- See: docs/system_law/Phase_X_Prelaunch_Review.md Section 3.3
- See: docs/system_law/TDA_PhaseX_Binding.md for TDA metric definitions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from backend.tda.monitor import TDAMonitor

__all__ = [
    "MetricsWindow",
    "MetricsAccumulator",
]


@dataclass
class MetricsWindow:
    """
    Windowed metrics accumulator for First-Light experiments.

    SHADOW MODE CONTRACT:
    - All metrics are observational only
    - No governance decisions depend on window contents
    - Used for logging and trajectory computation

    See: docs/system_law/Phase_X_P3_Spec.md Section 2.1
    """

    # Window configuration
    window_size: int = 50
    window_index: int = 0
    start_cycle: int = 0
    end_cycle: int = 0

    # Success metrics (U2)
    success_count: int = 0
    total_count: int = 0

    # Abstention metrics (RFL)
    abstention_count: int = 0

    # Safe region metrics
    omega_count: int = 0

    # HARD mode metrics
    hard_ok_count: int = 0

    # RSI accumulator
    rsi_sum: float = 0.0
    rsi_count: int = 0
    rsi_values: List[float] = field(default_factory=list)

    # Block rate tracking
    blocked_count: int = 0

    # TDA input trajectories (for TDAMonitor integration)
    H_trajectory: List[float] = field(default_factory=list)
    rho_trajectory: List[float] = field(default_factory=list)
    success_trajectory: List[bool] = field(default_factory=list)

    # Computed rates (updated on finalize)
    success_rate: Optional[float] = None
    abstention_rate: Optional[float] = None
    omega_occupancy: Optional[float] = None
    hard_ok_rate: Optional[float] = None
    mean_rsi: Optional[float] = None
    min_rsi: Optional[float] = None
    max_rsi: Optional[float] = None
    block_rate: Optional[float] = None

    def add(
        self,
        success: bool,
        abstained: bool,
        in_omega: bool,
        hard_ok: bool,
        rsi: float,
        blocked: bool,
        H: Optional[float] = None,
    ) -> None:
        """
        Add a cycle observation to the window.

        Args:
            success: Whether cycle succeeded (U2)
            abstained: Whether cycle abstained (RFL)
            in_omega: Whether state was in safe region
            hard_ok: Whether HARD mode was OK
            rsi: RSI value for this cycle (also known as rho)
            blocked: Whether governance blocked
            H: Health metric value for TDA computation (optional for backwards compat)
        """
        self.total_count += 1
        self.end_cycle = self.start_cycle + self.total_count - 1

        if success:
            self.success_count += 1

        if abstained:
            self.abstention_count += 1

        if in_omega:
            self.omega_count += 1

        if hard_ok:
            self.hard_ok_count += 1

        if blocked:
            self.blocked_count += 1

        # RSI tracking
        self.rsi_sum += rsi
        self.rsi_count += 1
        self.rsi_values.append(rsi)

        # TDA trajectory tracking
        if H is not None:
            self.H_trajectory.append(H)
        self.rho_trajectory.append(rsi)
        self.success_trajectory.append(success)

    def is_full(self) -> bool:
        """Check if window has reached its size limit."""
        return self.total_count >= self.window_size

    def finalize(self) -> Dict[str, Any]:
        """
        Finalize window and compute rates.

        Returns:
            Dictionary with computed metrics
        """
        if self.total_count > 0:
            self.success_rate = self.success_count / self.total_count
            self.abstention_rate = self.abstention_count / self.total_count
            self.omega_occupancy = self.omega_count / self.total_count
            self.hard_ok_rate = self.hard_ok_count / self.total_count
            self.block_rate = self.blocked_count / self.total_count
        else:
            self.success_rate = 0.0
            self.abstention_rate = 0.0
            self.omega_occupancy = 0.0
            self.hard_ok_rate = 0.0
            self.block_rate = 0.0

        if self.rsi_count > 0:
            self.mean_rsi = self.rsi_sum / self.rsi_count
            self.min_rsi = min(self.rsi_values) if self.rsi_values else 0.0
            self.max_rsi = max(self.rsi_values) if self.rsi_values else 0.0
        else:
            self.mean_rsi = 0.0
            self.min_rsi = 0.0
            self.max_rsi = 0.0

        return {
            "window_index": self.window_index,
            "start_cycle": self.start_cycle,
            "end_cycle": self.end_cycle,
            "total_count": self.total_count,
            "mode": "SHADOW",
            "success_metrics": {
                "success_count": self.success_count,
                "success_rate": round(self.success_rate, 4) if self.success_rate else 0.0,
            },
            "abstention_metrics": {
                "abstention_count": self.abstention_count,
                "abstention_rate": round(self.abstention_rate, 4) if self.abstention_rate else 0.0,
            },
            "safe_region_metrics": {
                "omega_count": self.omega_count,
                "omega_occupancy": round(self.omega_occupancy, 4) if self.omega_occupancy else 0.0,
            },
            "hard_mode_metrics": {
                "hard_ok_count": self.hard_ok_count,
                "hard_ok_rate": round(self.hard_ok_rate, 4) if self.hard_ok_rate else 0.0,
            },
            "stability_metrics": {
                "mean_rsi": round(self.mean_rsi, 4) if self.mean_rsi else 0.0,
                "min_rsi": round(self.min_rsi, 4) if self.min_rsi else 0.0,
                "max_rsi": round(self.max_rsi, 4) if self.max_rsi else 0.0,
            },
            "block_metrics": {
                "blocked_count": self.blocked_count,
                "block_rate": round(self.block_rate, 4) if self.block_rate else 0.0,
            },
            "tda_inputs": {
                "H_trajectory": list(self.H_trajectory),
                "rho_trajectory": list(self.rho_trajectory),
                "success_trajectory": list(self.success_trajectory),
            },
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert current state to dictionary."""
        return self.finalize()

    def finalize_with_tda(
        self,
        tda_monitor: "TDAMonitor",
    ) -> Dict[str, Any]:
        """
        Finalize window with TDA metrics integration.

        This extends finalize() by calling TDAMonitor.finalize_window()
        and including TDA metrics in the output.

        Args:
            tda_monitor: TDAMonitor instance tracking TDA metrics

        Returns:
            Dictionary with standard metrics + TDA metrics
        """
        # Get standard metrics
        result = self.finalize()

        # Get TDA window metrics
        tda_window = tda_monitor.finalize_window(self.window_index)

        # Add TDA metrics to result
        result["tda_metrics"] = tda_window.to_dict()

        return result

    def reset(self, window_index: int, start_cycle: int) -> None:
        """
        Reset window for next iteration.

        Args:
            window_index: New window index
            start_cycle: Starting cycle for new window
        """
        self.window_index = window_index
        self.start_cycle = start_cycle
        self.end_cycle = start_cycle
        self.success_count = 0
        self.total_count = 0
        self.abstention_count = 0
        self.omega_count = 0
        self.hard_ok_count = 0
        self.rsi_sum = 0.0
        self.rsi_count = 0
        self.rsi_values = []
        self.blocked_count = 0
        self.success_rate = None
        self.abstention_rate = None
        self.omega_occupancy = None
        self.hard_ok_rate = None
        self.mean_rsi = None
        self.min_rsi = None
        self.max_rsi = None
        self.block_rate = None
        # Reset TDA trajectories
        self.H_trajectory = []
        self.rho_trajectory = []
        self.success_trajectory = []


class MetricsAccumulator:
    """
    Accumulates metrics across multiple windows.

    SHADOW MODE CONTRACT:
    - All metrics are observational only
    - No governance decisions depend on accumulated values
    - Used for trajectory and Deltap computation
    """

    def __init__(self, window_size: int = 50) -> None:
        """
        Initialize accumulator.

        Args:
            window_size: Size of each metrics window
        """
        self.window_size = window_size
        self._current_window: MetricsWindow = MetricsWindow(window_size=window_size)
        self._completed_windows: List[Dict[str, Any]] = []

        # Cumulative counters
        self._total_success: int = 0
        self._total_abstention: int = 0
        self._total_omega: int = 0
        self._total_hard_ok: int = 0
        self._total_blocked: int = 0
        self._total_cycles: int = 0
        self._total_rsi_sum: float = 0.0

    def add(
        self,
        success: bool,
        abstained: bool,
        in_omega: bool,
        hard_ok: bool,
        rsi: float,
        blocked: bool,
        H: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Add a cycle observation.

        Args:
            success: Whether cycle succeeded
            abstained: Whether cycle abstained
            in_omega: Whether state was in safe region
            hard_ok: Whether HARD mode was OK
            rsi: RSI value (also known as rho)
            blocked: Whether governance blocked
            H: Health metric value for TDA computation (optional)

        Returns:
            Finalized window dict if window completed, None otherwise
        """
        # Update cumulative counters
        self._total_cycles += 1
        if success:
            self._total_success += 1
        if abstained:
            self._total_abstention += 1
        if in_omega:
            self._total_omega += 1
        if hard_ok:
            self._total_hard_ok += 1
        if blocked:
            self._total_blocked += 1
        self._total_rsi_sum += rsi

        # Add to current window (pass H for TDA tracking)
        self._current_window.add(success, abstained, in_omega, hard_ok, rsi, blocked, H=H)

        # Check if window is full
        if self._current_window.is_full():
            return self._finalize_current_window()

        return None

    def _finalize_current_window(self) -> Dict[str, Any]:
        """Finalize current window and start a new one."""
        window_data = self._current_window.finalize()
        self._completed_windows.append(window_data)

        # Start new window
        new_index = len(self._completed_windows)
        new_start = self._current_window.end_cycle + 1
        self._current_window = MetricsWindow(window_size=self.window_size)
        self._current_window.reset(new_index, new_start)

        return window_data

    def finalize_partial(self) -> Optional[Dict[str, Any]]:
        """Finalize any partial window at end of run."""
        if self._current_window.total_count > 0:
            return self._finalize_current_window()
        return None

    def get_trajectories(self) -> Dict[str, List[float]]:
        """Get trajectory lists for all metrics."""
        success_rates: List[float] = []
        abstention_rates: List[float] = []
        omega_occupancies: List[float] = []
        hard_ok_rates: List[float] = []
        mean_rsis: List[float] = []

        for window in self._completed_windows:
            success_rates.append(window["success_metrics"]["success_rate"])
            abstention_rates.append(window["abstention_metrics"]["abstention_rate"])
            omega_occupancies.append(window["safe_region_metrics"]["omega_occupancy"])
            hard_ok_rates.append(window["hard_mode_metrics"]["hard_ok_rate"])
            mean_rsis.append(window["stability_metrics"]["mean_rsi"])

        return {
            "success_rate": success_rates,
            "abstention_rate": abstention_rates,
            "omega_occupancy": omega_occupancies,
            "hard_ok_rate": hard_ok_rates,
            "mean_rsi": mean_rsis,
        }

    def get_cumulative_rates(self) -> Dict[str, float]:
        """Get cumulative rates across all cycles."""
        if self._total_cycles == 0:
            return {
                "success_rate": 0.0,
                "abstention_rate": 0.0,
                "omega_occupancy": 0.0,
                "hard_ok_rate": 0.0,
                "mean_rsi": 0.0,
                "block_rate": 0.0,
            }

        return {
            "success_rate": self._total_success / self._total_cycles,
            "abstention_rate": self._total_abstention / self._total_cycles,
            "omega_occupancy": self._total_omega / self._total_cycles,
            "hard_ok_rate": self._total_hard_ok / self._total_cycles,
            "mean_rsi": self._total_rsi_sum / self._total_cycles,
            "block_rate": self._total_blocked / self._total_cycles,
        }

    def get_completed_windows(self) -> List[Dict[str, Any]]:
        """Get list of all completed window data."""
        return list(self._completed_windows)

    def get_all_windows(self) -> List[Dict[str, Any]]:
        """
        Get all windows including any partial current window.

        This method is useful for TDA computation at finalize time, where
        we want to process all windows that have accumulated data.

        Returns:
            List of all window data dicts (completed + partial current)
        """
        result = list(self._completed_windows)
        # Include current window if it has any data
        if self._current_window.total_count > 0:
            result.append(self._current_window.finalize())
        return result

    @property
    def total_windows(self) -> int:
        """Get total number of completed windows."""
        return len(self._completed_windows)

    @property
    def total_cycles(self) -> int:
        """Get total number of cycles processed."""
        return self._total_cycles

    def reset(self) -> None:
        """Reset all accumulated data."""
        self._current_window = MetricsWindow(window_size=self.window_size)
        self._completed_windows.clear()
        self._total_success = 0
        self._total_abstention = 0
        self._total_omega = 0
        self._total_hard_ok = 0
        self._total_blocked = 0
        self._total_cycles = 0
        self._total_rsi_sum = 0.0
