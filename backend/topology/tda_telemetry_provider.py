"""
TDATelemetryProvider — Adapter exposing TDA governance telemetry for USLA bridge.

Phase X P1: Real Telemetry Wiring

This module provides a standardized interface for extracting governance-relevant
telemetry from the system. It serves as the data source for USLABridge.

SHADOW MODE CONTRACT:
This module is READ-ONLY with respect to governance state. It observes and
reports but never modifies governance decisions.

Current Implementation Status:
- blocked: Extracted from governance decision metadata
- threshold: Computed from USLA params or extracted from governance layer
- hss_by_depth: Aggregated from proof/cycle metadata if available
- min_cut_capacity: Placeholder (requires TDA filtration integration)
- betti_numbers: Placeholder (requires TDA filtration integration)

Usage:
    from backend.topology.tda_telemetry_provider import TDATelemetryProvider

    provider = TDATelemetryProvider()

    # After governance decision:
    snapshot = provider.capture_from_rfl(ledger_entry, attestation, blocked=False)
    # or
    snapshot = provider.capture_from_u2(cycle_result, blocked=False)

    # Use in bridge:
    bridge.translate(cycle_result, telemetry=snapshot)

# =============================================================================
# TODO: [TDA-TEL-001] TDA Feedback from Telemetry Anomalies
# =============================================================================
# Priority: Post-P4 Implementation
# Contract: docs/system_law/Telemetry_PhaseX_Contract.md Section 10.2
#
# This TODO tracks the implementation of TDA (Topological Data Analysis) feedback
# derived from telemetry anomalies. The feedback loop will:
#
# 1. [ ] Define TelemetryAnomalyWindow dataclass
#        - Window of telemetry records for analysis
#        - Configurable size (default: 100 cycles)
#        - Anomaly classification (emission_gap, schema_violation, rate_anomaly, etc.)
#
# 2. [ ] Implement anomaly clustering
#        - Use DBSCAN or similar for spatial clustering of anomalies
#        - Cluster parameters tuned for telemetry domain
#        - Output: anomaly clusters with centroid and radius
#
# 3. [ ] Compute Betti numbers from anomaly point cloud
#        - β₀: Number of connected components (anomaly clusters)
#        - β₁: Number of 1-dimensional holes (cyclical anomaly patterns)
#        - Use ripser or similar for persistence computation
#
# 4. [ ] Implement persistence diagram computation
#        - Track birth/death of topological features
#        - Identify persistent anomaly patterns
#        - Flag patterns that exceed persistence threshold
#
# 5. [ ] Generate tda_feedback section for telemetry_governance_signal.schema.json
#        - topology_alert_level: NORMAL | ELEVATED | WARNING | CRITICAL
#        - betti_anomaly_detected: boolean
#        - persistence_anomaly_detected: boolean
#        - min_cut_capacity_degraded: boolean (if telemetry flow drops)
#        - recommended_actions: list of observational recommendations
#
# 6. [ ] Integration with telemetry_governance_signal.schema.json
#        - Emit TDA feedback as part of governance signal
#        - All feedback is OBSERVATIONAL ONLY (SHADOW MODE)
#        - No enforcement of recommended actions
#
# Shadow Mode Constraints:
# - TDA feedback is OBSERVATIONAL ONLY
# - recommended_actions are LOGGED, not ENFORCED
# - No modification of upstream telemetry flow
# - No feedback loop to real runner execution
#
# Dependencies:
# - telemetry_governance_signal.schema.json (output schema)
# - Optional: ripser, scikit-tda for TDA computations
# - backend/verification/drift_radar/ (existing anomaly detection patterns)
#
# Implementation Location: backend/topology/tda_telemetry_feedback.py (NEW)
# =============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

__all__ = [
    "TDATelemetryProvider",
    "TDATelemetrySnapshot",
    "TDATelemetryConfig",
]


@dataclass
class TDATelemetrySnapshot:
    """
    Snapshot of TDA governance telemetry.

    This is the authoritative structure for telemetry data passed to USLABridge.
    """
    # Core governance decision
    blocked: bool = False
    threshold: Optional[float] = None

    # Depth-stratified HSS (key: depth, value: HSS at that depth)
    hss_by_depth: Optional[Dict[int, float]] = None

    # TDA topology metrics (optional, for future integration)
    min_cut_capacity: Optional[float] = None
    betti_numbers: Optional[List[int]] = None

    # Stability metrics from real system
    real_rsi: Optional[float] = None
    real_block_rate: Optional[float] = None

    # Metadata
    timestamp: Optional[str] = None
    source: str = "unknown"  # "rfl", "u2", "synthetic"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        result = {
            "blocked": self.blocked,
            "source": self.source,
        }
        if self.threshold is not None:
            result["threshold"] = self.threshold
        if self.hss_by_depth is not None:
            result["hss_by_depth"] = self.hss_by_depth
        if self.min_cut_capacity is not None:
            result["min_cut_capacity"] = self.min_cut_capacity
        if self.betti_numbers is not None:
            result["betti_numbers"] = self.betti_numbers
        if self.real_rsi is not None:
            result["real_rsi"] = self.real_rsi
        if self.real_block_rate is not None:
            result["real_block_rate"] = self.real_block_rate
        if self.timestamp is not None:
            result["timestamp"] = self.timestamp
        return result


@dataclass
class TDATelemetryConfig:
    """Configuration for telemetry provider."""
    # Default threshold (τ₀) when not available from governance
    default_threshold: float = 0.2

    # Window for computing rolling metrics
    history_window: int = 20

    # Enable depth-stratified HSS computation
    compute_hss_by_depth: bool = True

    # Depth bins for HSS stratification
    depth_bins: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    @classmethod
    def default(cls) -> "TDATelemetryConfig":
        return cls()


class TDATelemetryProvider:
    """
    Provider for TDA governance telemetry.

    This class extracts telemetry from runner outputs and system state,
    packaging it into TDATelemetrySnapshot for consumption by USLABridge.

    SHADOW MODE: This provider is purely observational. It never modifies
    governance decisions or system state.
    """

    def __init__(self, config: Optional[TDATelemetryConfig] = None):
        self.config = config or TDATelemetryConfig.default()

        # History buffers for computing derived metrics
        self._hss_history: List[float] = []
        self._block_history: List[bool] = []
        self._depth_hss_accumulator: Dict[int, List[float]] = {}

    def capture_from_rfl(
        self,
        ledger_entry: Any,
        attestation: Any,
        blocked: bool = False,
        governance_threshold: Optional[float] = None,
    ) -> TDATelemetrySnapshot:
        """
        Capture telemetry from RFL runner outputs.

        Args:
            ledger_entry: RunLedgerEntry from RFL runner
            attestation: AttestedRunContext from RFL runner
            blocked: Whether governance blocked this cycle
            governance_threshold: Effective threshold τ if known

        Returns:
            TDATelemetrySnapshot with extracted data
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        # Extract HSS from abstention rate
        hss = None
        if hasattr(ledger_entry, 'abstention_fraction'):
            hss = 1.0 - ledger_entry.abstention_fraction
        elif hasattr(attestation, 'abstention_rate'):
            hss = 1.0 - attestation.abstention_rate

        # Update history
        if hss is not None:
            self._hss_history.append(hss)
            if len(self._hss_history) > self.config.history_window:
                self._hss_history = self._hss_history[-self.config.history_window:]

        self._block_history.append(blocked)
        if len(self._block_history) > self.config.history_window:
            self._block_history = self._block_history[-self.config.history_window:]

        # Compute rolling block rate
        block_rate = None
        if self._block_history:
            block_rate = sum(1 for b in self._block_history if b) / len(self._block_history)

        # Extract depth and accumulate for hss_by_depth
        depth = None
        if hasattr(attestation, 'metadata') and attestation.metadata:
            depth = attestation.metadata.get('max_depth')

        hss_by_depth = None
        if self.config.compute_hss_by_depth and depth is not None and hss is not None:
            # Accumulate HSS by depth
            if depth not in self._depth_hss_accumulator:
                self._depth_hss_accumulator[depth] = []
            self._depth_hss_accumulator[depth].append(hss)

            # Trim to window
            if len(self._depth_hss_accumulator[depth]) > self.config.history_window:
                self._depth_hss_accumulator[depth] = self._depth_hss_accumulator[depth][-self.config.history_window:]

            # Compute mean HSS by depth
            hss_by_depth = {
                d: sum(vals) / len(vals)
                for d, vals in self._depth_hss_accumulator.items()
                if vals
            }

        # Compute RSI proxy from HSS stability
        real_rsi = None
        if len(self._hss_history) >= 3:
            # Simple stability: 1 - normalized variance
            mean_hss = sum(self._hss_history) / len(self._hss_history)
            if mean_hss > 0:
                variance = sum((h - mean_hss) ** 2 for h in self._hss_history) / len(self._hss_history)
                real_rsi = max(0.0, min(1.0, 1.0 - variance * 10))

        return TDATelemetrySnapshot(
            blocked=blocked,
            threshold=governance_threshold or self.config.default_threshold,
            hss_by_depth=hss_by_depth,
            min_cut_capacity=None,  # Not available yet
            betti_numbers=None,  # Not available yet
            real_rsi=real_rsi,
            real_block_rate=block_rate,
            timestamp=timestamp,
            source="rfl",
        )

    def capture_from_u2(
        self,
        cycle_result: Any,
        blocked: bool = False,
        governance_threshold: Optional[float] = None,
        success_history: Optional[List[bool]] = None,
    ) -> TDATelemetrySnapshot:
        """
        Capture telemetry from U2 runner outputs.

        Args:
            cycle_result: CycleResult from U2 runner
            blocked: Whether governance blocked this cycle
            governance_threshold: Effective threshold τ if known
            success_history: Optional success history for HSS computation

        Returns:
            TDATelemetrySnapshot with extracted data
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        # Extract success for HSS computation
        success = False
        if hasattr(cycle_result, 'success'):
            success = cycle_result.success
        elif isinstance(cycle_result, dict):
            success = cycle_result.get('success', False)

        # Compute HSS from success history
        hss = None
        if success_history is not None and success_history:
            hss = sum(1 for s in success_history if s) / len(success_history)

        # Update history
        if hss is not None:
            self._hss_history.append(hss)
            if len(self._hss_history) > self.config.history_window:
                self._hss_history = self._hss_history[-self.config.history_window:]

        self._block_history.append(blocked)
        if len(self._block_history) > self.config.history_window:
            self._block_history = self._block_history[-self.config.history_window:]

        # Compute rolling block rate
        block_rate = None
        if self._block_history:
            block_rate = sum(1 for b in self._block_history if b) / len(self._block_history)

        # Extract depth
        depth = None
        if hasattr(cycle_result, 'metadata'):
            depth = cycle_result.metadata.get('max_depth') if cycle_result.metadata else None
        elif isinstance(cycle_result, dict):
            depth = cycle_result.get('depth') or cycle_result.get('max_depth')

        # Accumulate HSS by depth if available
        hss_by_depth = None
        if self.config.compute_hss_by_depth and depth is not None and hss is not None:
            if depth not in self._depth_hss_accumulator:
                self._depth_hss_accumulator[depth] = []
            self._depth_hss_accumulator[depth].append(hss)

            if len(self._depth_hss_accumulator[depth]) > self.config.history_window:
                self._depth_hss_accumulator[depth] = self._depth_hss_accumulator[depth][-self.config.history_window:]

            hss_by_depth = {
                d: sum(vals) / len(vals)
                for d, vals in self._depth_hss_accumulator.items()
                if vals
            }

        # Compute RSI proxy
        real_rsi = None
        if len(self._hss_history) >= 3:
            mean_hss = sum(self._hss_history) / len(self._hss_history)
            if mean_hss > 0:
                variance = sum((h - mean_hss) ** 2 for h in self._hss_history) / len(self._hss_history)
                real_rsi = max(0.0, min(1.0, 1.0 - variance * 10))

        return TDATelemetrySnapshot(
            blocked=blocked,
            threshold=governance_threshold or self.config.default_threshold,
            hss_by_depth=hss_by_depth,
            min_cut_capacity=None,
            betti_numbers=None,
            real_rsi=real_rsi,
            real_block_rate=block_rate,
            timestamp=timestamp,
            source="u2",
        )

    def capture_synthetic(
        self,
        hss: float = 0.8,
        depth: int = 5,
        blocked: bool = False,
        threshold: float = 0.2,
    ) -> TDATelemetrySnapshot:
        """
        Create synthetic telemetry for testing.

        Args:
            hss: Health signal score
            depth: Current depth
            blocked: Whether blocked
            threshold: Governance threshold

        Returns:
            TDATelemetrySnapshot with synthetic data
        """
        return TDATelemetrySnapshot(
            blocked=blocked,
            threshold=threshold,
            hss_by_depth={depth: hss} if depth else None,
            real_rsi=0.8,
            real_block_rate=0.1 if blocked else 0.0,
            timestamp=datetime.now(timezone.utc).isoformat(),
            source="synthetic",
        )

    def reset(self) -> None:
        """Reset all history buffers."""
        self._hss_history.clear()
        self._block_history.clear()
        self._depth_hss_accumulator.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get provider statistics."""
        return {
            "hss_history_len": len(self._hss_history),
            "block_history_len": len(self._block_history),
            "depth_bins_tracked": len(self._depth_hss_accumulator),
            "depths": list(self._depth_hss_accumulator.keys()),
        }
