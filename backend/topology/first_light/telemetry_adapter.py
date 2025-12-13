"""
Phase X P4: Telemetry Adapter for Real Runner Shadow Coupling

This module implements the telemetry provider interface for P4 shadow mode.
See docs/system_law/Phase_X_P4_Spec.md for full specification.

SHADOW MODE CONTRACT:
- TelemetryProvider is READ-ONLY
- get_snapshot() NEVER modifies any state
- No side-effects from any method
- All telemetry is captured AFTER real execution

Status: P4 IMPLEMENTATION
"""

from __future__ import annotations

import hashlib
import json
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional, TYPE_CHECKING

from backend.topology.first_light.data_structures_p4 import TelemetrySnapshot

if TYPE_CHECKING:
    pass  # Future: USLAIntegration type hint

__all__ = [
    "TelemetryProviderInterface",
    "MockTelemetryProvider",
    "USLAIntegrationAdapter",
]


class TelemetryProviderInterface(ABC):
    """
    Abstract interface for telemetry providers.

    SHADOW MODE CONTRACT:
    - All implementations must be READ-ONLY
    - get_snapshot() must not modify any state
    - is_available() must not have side-effects

    See: docs/system_law/Phase_X_P4_Spec.md Section 5.2
    """

    @abstractmethod
    def get_snapshot(self) -> Optional[TelemetrySnapshot]:
        """
        Get current telemetry snapshot.

        SHADOW MODE: This is a read-only operation.
        Must not modify any state in the real runner.

        Returns:
            TelemetrySnapshot if available, None otherwise
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if telemetry is available.

        SHADOW MODE: This is a read-only check.

        Returns:
            True if telemetry can be retrieved
        """
        pass

    @abstractmethod
    def get_current_cycle(self) -> int:
        """
        Get current cycle number.

        Returns:
            Current cycle number (0 if not started)
        """
        pass

    @abstractmethod
    def get_runner_type(self) -> str:
        """
        Get runner type being observed.

        Returns:
            "u2" or "rfl"
        """
        pass


class MockTelemetryProvider(TelemetryProviderInterface):
    """
    Mock telemetry provider for testing.

    SHADOW MODE: Generates synthetic telemetry for P4 testing.
    Does not connect to any real runner.

    See: docs/system_law/Phase_X_P4_Spec.md Section 5.3
    """

    def __init__(
        self,
        runner_type: str = "u2",
        slice_name: str = "arithmetic_simple",
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize mock provider.

        Args:
            runner_type: Type of runner ("u2" or "rfl")
            slice_name: Name of slice
            seed: Random seed for reproducibility
        """
        if runner_type not in ("u2", "rfl"):
            raise ValueError(f"runner_type must be 'u2' or 'rfl', got '{runner_type}'")

        self._runner_type = runner_type
        self._slice_name = slice_name
        self._seed = seed
        self._rng = random.Random(seed)

        # Internal state
        self._cycle = 0
        self._H = 0.5
        self._rho = 0.7
        self._tau = 0.2
        self._beta = 0.1
        self._available = True

        # Historical snapshots
        self._history: List[TelemetrySnapshot] = []

    def get_snapshot(self) -> Optional[TelemetrySnapshot]:
        """
        Get mock telemetry snapshot.

        SHADOW MODE: Generates synthetic data for testing.

        Returns:
            TelemetrySnapshot with mock data
        """
        if not self._available:
            return None

        self._cycle += 1
        self._update_state()

        # Generate snapshot data
        timestamp = datetime.now(timezone.utc).isoformat()
        success = self._rng.random() < (0.6 + self._cycle * 0.001)
        in_omega = self._H > self._tau and self._rho > 0.5
        hard_ok = self._rng.random() > 0.02
        real_blocked = self._beta > 0.7
        abstained = self._rng.random() < 0.1 if self._runner_type == "rfl" else None

        # Build snapshot data for hash
        data = {
            "cycle": self._cycle,
            "timestamp": timestamp,
            "runner_type": self._runner_type,
            "H": self._H,
            "rho": self._rho,
            "tau": self._tau,
            "beta": self._beta,
        }
        snapshot_hash = TelemetrySnapshot.compute_hash(data)

        snapshot = TelemetrySnapshot(
            cycle=self._cycle,
            timestamp=timestamp,
            runner_type=self._runner_type,
            slice_name=self._slice_name,
            success=success,
            depth=self._rng.randint(3, 8) if success else self._rng.randint(1, 3),
            proof_hash=None,
            H=self._H,
            rho=self._rho,
            tau=self._tau,
            beta=self._beta,
            in_omega=in_omega,
            real_blocked=real_blocked,
            governance_aligned=True,
            governance_reason=None,
            hard_ok=hard_ok,
            abstained=abstained,
            abstention_reason="uncertainty" if abstained else None,
            reasoning_graph_hash=None,
            proof_dag_size=0,
            snapshot_hash=snapshot_hash,
        )

        self._history.append(snapshot)
        return snapshot

    def _update_state(self) -> None:
        """Update internal state with realistic dynamics."""
        # Gradual improvement over time
        learning = min(0.3, self._cycle * 0.001)

        # H tends upward with noise
        self._H = max(0.0, min(1.0, self._H + 0.01 + self._rng.gauss(0, 0.02)))

        # rho is stable with small fluctuations
        self._rho = max(0.0, min(1.0, self._rho + self._rng.gauss(0, 0.03)))

        # tau drifts slowly
        self._tau = max(0.0, min(1.0, self._tau + self._rng.gauss(0, 0.005)))

        # beta inversely related to success
        if self._rng.random() < 0.6 + learning:
            self._beta = max(0.0, self._beta - 0.01 + self._rng.gauss(0, 0.01))
        else:
            self._beta = min(1.0, self._beta + 0.02 + self._rng.gauss(0, 0.01))

    def is_available(self) -> bool:
        """Check if mock telemetry is available."""
        return self._available

    def get_current_cycle(self) -> int:
        """Get current cycle number."""
        return self._cycle

    def get_runner_type(self) -> str:
        """Get runner type being observed."""
        return self._runner_type

    def get_historical_snapshots(
        self, start_cycle: int, end_cycle: int
    ) -> Iterator[TelemetrySnapshot]:
        """Get historical snapshots in range (READ-ONLY)."""
        for snapshot in self._history:
            if start_cycle <= snapshot.cycle <= end_cycle:
                yield snapshot

    def set_available(self, available: bool) -> None:
        """Set availability for testing."""
        self._available = available

    def add_snapshot(self, snapshot: TelemetrySnapshot) -> None:
        """Add a snapshot to the mock (for test setup)."""
        self._history.append(snapshot)
        self._cycle = max(self._cycle, snapshot.cycle)

    def reset(self) -> None:
        """Reset mock provider state."""
        self._cycle = 0
        self._H = 0.5
        self._rho = 0.7
        self._tau = 0.2
        self._beta = 0.1
        self._rng = random.Random(self._seed)
        self._history.clear()


class USLAIntegrationAdapter(TelemetryProviderInterface):
    """
    Adapter for real USLA integration telemetry.

    SHADOW MODE: Wraps real USLA integration to provide read-only telemetry.
    This adapter NEVER modifies the underlying integration.

    See: docs/system_law/Phase_X_P4_Spec.md Section 5.4

    NOTE: This requires the actual USLAIntegration to be available.
    In P4, we typically use MockTelemetryProvider for testing.
    """

    def __init__(
        self,
        integration_ref: Any = None,  # Optional[USLAIntegration]
        runner_type: str = "u2",
        slice_name: str = "arithmetic_simple",
    ) -> None:
        """
        Initialize USLA integration adapter.

        Args:
            integration_ref: The real USLAIntegration instance (optional)
            runner_type: Type of runner
            slice_name: Name of slice

        Raises:
            ValueError: If runner_type is invalid
        """
        if runner_type not in ("u2", "rfl"):
            raise ValueError(f"runner_type must be 'u2' or 'rfl', got '{runner_type}'")

        self._integration = integration_ref
        self._runner_type = runner_type
        self._slice_name = slice_name
        self._cycle = 0
        self._history: List[TelemetrySnapshot] = []

    def get_snapshot(self) -> Optional[TelemetrySnapshot]:
        """
        Get telemetry snapshot from real USLA integration.

        SHADOW MODE: READ-ONLY access to USLA state.

        Returns:
            TelemetrySnapshot from real runner, None if unavailable
        """
        if self._integration is None:
            return None

        self._cycle += 1

        try:
            # Get state from real integration (read-only)
            state = self._integration.get_current_state()
            if state is None:
                return None

            timestamp = datetime.now(timezone.utc).isoformat()

            # Build snapshot data for hash
            data = {
                "cycle": self._cycle,
                "timestamp": timestamp,
                "runner_type": self._runner_type,
                "H": state.get("H", 0.0),
                "rho": state.get("rho", 0.0),
                "tau": state.get("tau", 0.0),
                "beta": state.get("beta", 0.0),
            }
            snapshot_hash = TelemetrySnapshot.compute_hash(data)

            snapshot = TelemetrySnapshot(
                cycle=self._cycle,
                timestamp=timestamp,
                runner_type=self._runner_type,
                slice_name=self._slice_name,
                success=state.get("success", False),
                depth=state.get("depth"),
                proof_hash=state.get("proof_hash"),
                H=state.get("H", 0.0),
                rho=state.get("rho", 0.0),
                tau=state.get("tau", 0.0),
                beta=state.get("beta", 0.0),
                in_omega=state.get("in_omega", False),
                real_blocked=state.get("blocked", False),
                governance_aligned=state.get("governance_aligned", True),
                governance_reason=state.get("governance_reason"),
                hard_ok=state.get("hard_ok", True),
                abstained=state.get("abstained"),
                abstention_reason=state.get("abstention_reason"),
                reasoning_graph_hash=state.get("reasoning_graph_hash"),
                proof_dag_size=state.get("proof_dag_size", 0),
                snapshot_hash=snapshot_hash,
            )

            self._history.append(snapshot)
            return snapshot

        except Exception:
            # SHADOW MODE: Never propagate errors that could affect real runner
            return None

    def is_available(self) -> bool:
        """Check if USLA integration is available."""
        if self._integration is None:
            return False

        try:
            # Check if integration is ready (read-only check)
            return hasattr(self._integration, "get_current_state")
        except Exception:
            return False

    def get_current_cycle(self) -> int:
        """Get current cycle number."""
        return self._cycle

    def get_runner_type(self) -> str:
        """Get runner type being observed."""
        return self._runner_type

    def get_historical_snapshots(
        self, start_cycle: int, end_cycle: int
    ) -> Iterator[TelemetrySnapshot]:
        """Get historical snapshots in range (READ-ONLY)."""
        for snapshot in self._history:
            if start_cycle <= snapshot.cycle <= end_cycle:
                yield snapshot

    def reset(self) -> None:
        """Reset adapter state."""
        self._cycle = 0
        self._history.clear()
