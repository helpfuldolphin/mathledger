"""
TDA Telemetry Feedback Provider — Topological Data Analysis from Telemetry Anomalies

Phase X: Telemetry Canonical Interface

This module implements TDA (Topological Data Analysis) feedback derived from
telemetry anomalies. The feedback loop analyzes anomaly patterns to detect
topological signatures of system degradation.

SHADOW MODE CONTRACT:
- All functions are READ-ONLY and side-effect free
- TDA feedback is OBSERVATIONAL ONLY
- Recommended actions are LOGGED, not ENFORCED
- No modification of upstream telemetry flow
- No feedback loop to real runner execution

See: docs/system_law/Telemetry_PhaseX_Contract.md Section 9

Status: P4 IMPLEMENTATION
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

__all__ = [
    "TDAFeedbackProvider",
    "TDAFeedback",
    "TelemetryAnomalyWindow",
    "AnomalyRecord",
    "TopologyMetrics",
]


@dataclass
class AnomalyRecord:
    """
    Single anomaly record from telemetry stream.

    SHADOW MODE: This is an observation record, not an enforcement trigger.
    """
    cycle: int
    timestamp: str
    anomaly_type: str  # emission_gap, schema_violation, rate_anomaly, value_anomaly, sequence_anomaly
    severity: str  # INFO, WARN, CRITICAL
    component: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "cycle": self.cycle,
            "timestamp": self.timestamp,
            "anomaly_type": self.anomaly_type,
            "severity": self.severity,
            "component": self.component,
            "details": self.details,
        }


@dataclass
class TelemetryAnomalyWindow:
    """
    Window of telemetry anomalies for TDA analysis.

    SHADOW MODE: This is a read-only collection for analysis.
    """
    window_size: int = 100
    anomalies: List[AnomalyRecord] = field(default_factory=list)
    start_cycle: int = 0
    end_cycle: int = 0

    def add_anomaly(self, anomaly: AnomalyRecord) -> None:
        """Add anomaly to window, maintaining size limit."""
        self.anomalies.append(anomaly)
        if len(self.anomalies) > self.window_size:
            self.anomalies = self.anomalies[-self.window_size:]

        if self.anomalies:
            self.start_cycle = self.anomalies[0].cycle
            self.end_cycle = self.anomalies[-1].cycle

    def get_by_type(self, anomaly_type: str) -> List[AnomalyRecord]:
        """Get anomalies by type."""
        return [a for a in self.anomalies if a.anomaly_type == anomaly_type]

    def get_by_severity(self, severity: str) -> List[AnomalyRecord]:
        """Get anomalies by severity."""
        return [a for a in self.anomalies if a.severity == severity]

    def count_by_type(self) -> Dict[str, int]:
        """Count anomalies by type."""
        counts: Dict[str, int] = defaultdict(int)
        for a in self.anomalies:
            counts[a.anomaly_type] += 1
        return dict(counts)

    def count_by_severity(self) -> Dict[str, int]:
        """Count anomalies by severity."""
        counts: Dict[str, int] = defaultdict(int)
        for a in self.anomalies:
            counts[a.severity] += 1
        return dict(counts)

    def clear(self) -> None:
        """Clear the window."""
        self.anomalies.clear()
        self.start_cycle = 0
        self.end_cycle = 0


@dataclass
class TopologyMetrics:
    """
    Topological metrics computed from anomaly point cloud.

    SHADOW MODE: These metrics are for observation only.

    Metrics:
    - betti_0: Number of connected components (anomaly clusters)
    - betti_1: Number of 1-dimensional holes (cyclical patterns)
    - persistence_max: Maximum persistence of any feature
    - persistence_mean: Mean persistence across features
    - min_cut_capacity: Estimated min-cut in anomaly flow graph
    """
    betti_0: int = 0  # Connected components
    betti_1: int = 0  # 1-dimensional holes
    persistence_max: float = 0.0
    persistence_mean: float = 0.0
    min_cut_capacity: float = 1.0  # 1.0 = no degradation
    cluster_count: int = 0
    cluster_sizes: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "betti_0": self.betti_0,
            "betti_1": self.betti_1,
            "persistence_max": round(self.persistence_max, 4),
            "persistence_mean": round(self.persistence_mean, 4),
            "min_cut_capacity": round(self.min_cut_capacity, 4),
            "cluster_count": self.cluster_count,
            "cluster_sizes": self.cluster_sizes,
        }


@dataclass
class TDAFeedback:
    """
    TDA feedback signal for governance.

    SHADOW MODE: This feedback is OBSERVATIONAL ONLY.
    Recommended actions are LOGGED, not ENFORCED.

    See: docs/system_law/schemas/telemetry/telemetry_governance_signal.schema.json
    """
    feedback_available: bool = True
    topology_alert_level: str = "NORMAL"  # NORMAL, ELEVATED, WARNING, CRITICAL
    betti_anomaly_detected: bool = False
    persistence_anomaly_detected: bool = False
    min_cut_capacity_degraded: bool = False
    feedback_cycle: int = 0
    recommended_actions: List[str] = field(default_factory=list)

    # Underlying metrics
    topology_metrics: Optional[TopologyMetrics] = None

    # Metadata
    mode: str = "SHADOW"
    enforcement_status: str = "LOGGED_ONLY"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for schema compliance."""
        return {
            "feedback_available": self.feedback_available,
            "topology_alert_level": self.topology_alert_level,
            "betti_anomaly_detected": self.betti_anomaly_detected,
            "persistence_anomaly_detected": self.persistence_anomaly_detected,
            "min_cut_capacity_degraded": self.min_cut_capacity_degraded,
            "feedback_cycle": self.feedback_cycle,
            "recommended_actions": self.recommended_actions,
            "mode": self.mode,
            "enforcement_status": self.enforcement_status,
        }


class TDAFeedbackProvider:
    """
    Provider for TDA feedback from telemetry anomalies.

    SHADOW MODE CONTRACT:
    - All methods are READ-ONLY
    - TDA feedback is OBSERVATIONAL ONLY
    - Recommended actions are LOGGED, not ENFORCED
    - No modification of upstream telemetry flow

    This provider:
    1. Collects anomalies into a sliding window
    2. Computes topological features (Betti numbers, persistence)
    3. Detects anomalous topology patterns
    4. Generates feedback signals with recommended actions

    See: docs/system_law/Telemetry_PhaseX_Contract.md Section 9
    """

    def __init__(
        self,
        window_size: int = 100,
        betti_0_threshold: int = 5,  # Max clusters before anomaly
        betti_1_threshold: int = 2,  # Max holes before anomaly
        persistence_threshold: float = 0.3,  # Persistence ratio for anomaly
        min_cut_threshold: float = 0.5,  # Min-cut below this is degraded
        cluster_distance: float = 3.0,  # Distance threshold for clustering
    ):
        """
        Initialize TDA feedback provider.

        Args:
            window_size: Size of anomaly window for analysis
            betti_0_threshold: Threshold for Betti-0 anomaly
            betti_1_threshold: Threshold for Betti-1 anomaly
            persistence_threshold: Threshold for persistence anomaly
            min_cut_threshold: Threshold for min-cut degradation
            cluster_distance: Distance threshold for clustering
        """
        self.window_size = window_size
        self.betti_0_threshold = betti_0_threshold
        self.betti_1_threshold = betti_1_threshold
        self.persistence_threshold = persistence_threshold
        self.min_cut_threshold = min_cut_threshold
        self.cluster_distance = cluster_distance

        # Anomaly window
        self._window = TelemetryAnomalyWindow(window_size=window_size)
        self._current_cycle = 0

        # History for trend analysis
        self._feedback_history: List[TDAFeedback] = []

    def add_anomaly(
        self,
        cycle: int,
        anomaly_type: str,
        severity: str,
        component: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add an anomaly to the analysis window.

        SHADOW MODE: This is an observation operation only.

        Args:
            cycle: Cycle number
            anomaly_type: Type of anomaly
            severity: Severity level
            component: Source component
            details: Additional details
        """
        anomaly = AnomalyRecord(
            cycle=cycle,
            timestamp=datetime.now(timezone.utc).isoformat(),
            anomaly_type=anomaly_type,
            severity=severity,
            component=component,
            details=details or {},
        )
        self._window.add_anomaly(anomaly)
        self._current_cycle = max(self._current_cycle, cycle)

    def compute_topology_metrics(self) -> TopologyMetrics:
        """
        Compute topological metrics from anomaly window.

        SHADOW MODE: This is a read-only computation.

        Uses simplified DBSCAN-like clustering and estimates Betti numbers.
        For production, consider using ripser or similar for persistence.

        Returns:
            TopologyMetrics with computed values
        """
        anomalies = self._window.anomalies
        if not anomalies:
            return TopologyMetrics()

        # Convert anomalies to point cloud (cycle, severity_numeric)
        points = []
        severity_map = {"INFO": 1.0, "WARN": 2.0, "CRITICAL": 3.0}
        for a in anomalies:
            x = a.cycle
            y = severity_map.get(a.severity, 1.0)
            points.append((x, y))

        # Simple clustering (DBSCAN-like)
        clusters = self._cluster_points(points, self.cluster_distance)
        betti_0 = len(clusters)  # Number of connected components

        # Estimate Betti-1 from cyclic patterns in time series
        betti_1 = self._estimate_betti_1(anomalies)

        # Compute persistence from cluster lifespans
        persistence_values = []
        for cluster in clusters:
            if cluster:
                cycles = [p[0] for p in cluster]
                lifespan = max(cycles) - min(cycles) + 1
                # Normalize by window size
                persistence = lifespan / self.window_size if self.window_size > 0 else 0
                persistence_values.append(persistence)

        persistence_max = max(persistence_values) if persistence_values else 0.0
        persistence_mean = sum(persistence_values) / len(persistence_values) if persistence_values else 0.0

        # Estimate min-cut capacity from anomaly density
        # Lower density = higher capacity (healthy)
        # Higher density = lower capacity (degraded)
        window_span = self._window.end_cycle - self._window.start_cycle + 1
        if window_span > 0:
            density = len(anomalies) / window_span
            # Map density to capacity: 0 density = 1.0 capacity, high density = low capacity
            min_cut_capacity = max(0.0, 1.0 - density * 0.5)
        else:
            min_cut_capacity = 1.0

        cluster_sizes = [len(c) for c in clusters]

        return TopologyMetrics(
            betti_0=betti_0,
            betti_1=betti_1,
            persistence_max=persistence_max,
            persistence_mean=persistence_mean,
            min_cut_capacity=min_cut_capacity,
            cluster_count=len(clusters),
            cluster_sizes=cluster_sizes,
        )

    def _cluster_points(
        self,
        points: List[Tuple[float, float]],
        eps: float,
    ) -> List[List[Tuple[float, float]]]:
        """
        Simple DBSCAN-like clustering.

        Args:
            points: List of (x, y) points
            eps: Distance threshold

        Returns:
            List of clusters, each cluster is a list of points
        """
        if not points:
            return []

        visited = set()
        clusters = []

        def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
            return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

        def expand_cluster(point_idx: int, neighbors: List[int]) -> List[Tuple[float, float]]:
            cluster = [points[point_idx]]
            visited.add(point_idx)

            i = 0
            while i < len(neighbors):
                neighbor_idx = neighbors[i]
                if neighbor_idx not in visited:
                    visited.add(neighbor_idx)
                    cluster.append(points[neighbor_idx])

                    # Find neighbors of neighbor
                    new_neighbors = [
                        j for j, p in enumerate(points)
                        if j not in visited and distance(points[neighbor_idx], p) <= eps
                    ]
                    neighbors.extend(new_neighbors)
                i += 1

            return cluster

        for i, point in enumerate(points):
            if i in visited:
                continue

            neighbors = [
                j for j, p in enumerate(points)
                if j != i and distance(point, p) <= eps
            ]

            if neighbors:  # Has neighbors, form cluster
                cluster = expand_cluster(i, neighbors)
                clusters.append(cluster)
            else:  # Isolated point is its own cluster
                visited.add(i)
                clusters.append([point])

        return clusters

    def _estimate_betti_1(self, anomalies: List[AnomalyRecord]) -> int:
        """
        Estimate Betti-1 (number of 1-dimensional holes) from anomaly patterns.

        Looks for cyclical patterns in anomaly occurrence.

        Args:
            anomalies: List of anomaly records

        Returns:
            Estimated Betti-1 count
        """
        if len(anomalies) < 4:
            return 0

        # Look for periodic patterns in cycle numbers
        cycles = [a.cycle for a in anomalies]

        # Compute differences between consecutive anomalies
        diffs = [cycles[i + 1] - cycles[i] for i in range(len(cycles) - 1)]

        if not diffs:
            return 0

        # Look for repeating patterns (simplified)
        # Count how many similar inter-arrival times we see
        diff_counts: Dict[int, int] = defaultdict(int)
        for d in diffs:
            # Bucket into ranges
            bucket = d // 5 * 5  # 5-cycle buckets
            diff_counts[bucket] += 1

        # Periodic patterns show up as concentrated diff counts
        max_count = max(diff_counts.values()) if diff_counts else 0

        # If more than half of diffs are in same bucket, likely periodic
        if max_count > len(diffs) / 2:
            return 1  # One periodic pattern detected

        return 0

    def generate_feedback(self) -> TDAFeedback:
        """
        Generate TDA feedback from current anomaly window.

        SHADOW MODE: This feedback is OBSERVATIONAL ONLY.
        Recommended actions are LOGGED, not ENFORCED.

        Returns:
            TDAFeedback with topology analysis and recommendations
        """
        metrics = self.compute_topology_metrics()

        # Detect anomalies in topology
        betti_anomaly = (
            metrics.betti_0 > self.betti_0_threshold or
            metrics.betti_1 > self.betti_1_threshold
        )

        persistence_anomaly = metrics.persistence_max > self.persistence_threshold

        min_cut_degraded = metrics.min_cut_capacity < self.min_cut_threshold

        # Determine alert level
        alert_level = "NORMAL"
        critical_count = self._window.count_by_severity().get("CRITICAL", 0)
        warn_count = self._window.count_by_severity().get("WARN", 0)

        if critical_count > 0 or (betti_anomaly and persistence_anomaly):
            alert_level = "CRITICAL"
        elif min_cut_degraded or persistence_anomaly:
            alert_level = "WARNING"
        elif betti_anomaly or warn_count > 5:
            alert_level = "ELEVATED"

        # Generate recommended actions (OBSERVATIONAL ONLY)
        actions = []
        if betti_anomaly:
            actions.append(f"Review anomaly clustering: {metrics.betti_0} clusters detected")
        if persistence_anomaly:
            actions.append(f"Investigate persistent anomaly pattern: persistence={metrics.persistence_max:.2f}")
        if min_cut_degraded:
            actions.append(f"Telemetry flow degraded: min_cut_capacity={metrics.min_cut_capacity:.2f}")

        # Add type-specific recommendations
        type_counts = self._window.count_by_type()
        for anomaly_type, count in type_counts.items():
            if count > 3:
                actions.append(f"High frequency of {anomaly_type}: {count} occurrences")

        feedback = TDAFeedback(
            feedback_available=True,
            topology_alert_level=alert_level,
            betti_anomaly_detected=betti_anomaly,
            persistence_anomaly_detected=persistence_anomaly,
            min_cut_capacity_degraded=min_cut_degraded,
            feedback_cycle=self._current_cycle,
            recommended_actions=actions,
            topology_metrics=metrics,
            mode="SHADOW",
            enforcement_status="LOGGED_ONLY",
        )

        self._feedback_history.append(feedback)
        return feedback

    def get_anomaly_summary(self) -> Dict[str, Any]:
        """
        Get summary of current anomaly window.

        Returns:
            Summary dictionary
        """
        return {
            "window_size": self._window.window_size,
            "anomaly_count": len(self._window.anomalies),
            "start_cycle": self._window.start_cycle,
            "end_cycle": self._window.end_cycle,
            "by_type": self._window.count_by_type(),
            "by_severity": self._window.count_by_severity(),
        }

    def reset(self) -> None:
        """Reset provider state."""
        self._window.clear()
        self._current_cycle = 0
        self._feedback_history.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get provider statistics."""
        return {
            "window_size": self.window_size,
            "current_cycle": self._current_cycle,
            "anomaly_count": len(self._window.anomalies),
            "feedback_history_len": len(self._feedback_history),
            "thresholds": {
                "betti_0": self.betti_0_threshold,
                "betti_1": self.betti_1_threshold,
                "persistence": self.persistence_threshold,
                "min_cut": self.min_cut_threshold,
            },
        }
