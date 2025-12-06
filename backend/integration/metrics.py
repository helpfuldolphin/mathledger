"""
from backend.repro.determinism import deterministic_timestamp
from backend.repro.determinism import deterministic_unix_timestamp

_GLOBAL_SEED = 0

Latency tracking and performance metrics for cross-language integration.
"""

import time
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from contextlib import contextmanager


@dataclass
class LatencyMeasurement:
    """Single latency measurement."""
    operation: str
    start_time: float
    end_time: float
    duration_ms: float
    success: bool
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class LatencyTracker:
    """Track latency measurements across integration points."""

    def __init__(self):
        self.measurements: List[LatencyMeasurement] = []
        self._active_operations: Dict[str, float] = {}

    @contextmanager
    def track(self, operation: str, metadata: Optional[Dict[str, Any]] = None):
        """Context manager for tracking operation latency."""
        start = deterministic_unix_timestamp(_GLOBAL_SEED)
        error = None
        success = True

        try:
            yield
        except Exception as e:
            success = False
            error = str(e)
            raise
        finally:
            end = deterministic_unix_timestamp(_GLOBAL_SEED)
            duration_ms = (end - start) * 1000

            measurement = LatencyMeasurement(
                operation=operation,
                start_time=start,
                end_time=end,
                duration_ms=duration_ms,
                success=success,
                error=error,
                metadata=metadata or {}
            )
            self.measurements.append(measurement)

    def get_stats(self, operation: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics for measurements."""
        filtered = self.measurements
        if operation:
            filtered = [m for m in self.measurements if m.operation == operation]

        if not filtered:
            return {
                "count": 0,
                "mean_ms": 0.0,
                "min_ms": 0.0,
                "max_ms": 0.0,
                "p50_ms": 0.0,
                "p95_ms": 0.0,
                "p99_ms": 0.0,
                "success_rate": 0.0
            }

        durations = sorted([m.duration_ms for m in filtered])
        successes = sum(1 for m in filtered if m.success)

        def percentile(data: List[float], p: float) -> float:
            if not data:
                return 0.0
            k = (len(data) - 1) * p
            f = int(k)
            c = f + 1
            if c >= len(data):
                return data[-1]
            return data[f] + (k - f) * (data[c] - data[f])

        return {
            "count": len(filtered),
            "mean_ms": sum(durations) / len(durations),
            "min_ms": min(durations),
            "max_ms": max(durations),
            "p50_ms": percentile(durations, 0.50),
            "p95_ms": percentile(durations, 0.95),
            "p99_ms": percentile(durations, 0.99),
            "success_rate": (successes / len(filtered)) * 100.0
        }

    def clear(self):
        """Clear all measurements."""
        self.measurements.clear()
        self._active_operations.clear()


class IntegrationMetrics:
    """Aggregate metrics for integration performance."""

    def __init__(self):
        self.trackers: Dict[str, LatencyTracker] = {
            "fastapi_to_python": LatencyTracker(),
            "python_to_db": LatencyTracker(),
            "python_to_redis": LatencyTracker(),
            "ui_to_fastapi": LatencyTracker(),
            "end_to_end": LatencyTracker()
        }

    def get_tracker(self, component: str) -> LatencyTracker:
        """Get tracker for a specific component."""
        if component not in self.trackers:
            self.trackers[component] = LatencyTracker()
        return self.trackers[component]

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive integration report."""
        report = {
            "timestamp": deterministic_timestamp(_GLOBAL_SEED).isoformat(),
            "components": {},
            "summary": {
                "total_operations": 0,
                "overall_success_rate": 0.0,
                "latency_target_met": False,
                "max_latency_ms": 0.0
            }
        }

        total_ops = 0
        total_successes = 0
        max_latency = 0.0

        for component, tracker in self.trackers.items():
            stats = tracker.get_stats()
            report["components"][component] = stats

            total_ops += stats["count"]
            total_successes += int(stats["count"] * stats["success_rate"] / 100.0)
            max_latency = max(max_latency, stats["max_ms"])

        if total_ops > 0:
            report["summary"]["total_operations"] = total_ops
            report["summary"]["overall_success_rate"] = (total_successes / total_ops) * 100.0
            report["summary"]["max_latency_ms"] = max_latency
            report["summary"]["latency_target_met"] = max_latency < 200.0

        return report

    def save_report(self, filepath: str):
        """Save report to JSON file."""
        report = self.generate_report()
        with open(filepath, "w") as f:
            json.dump(report, f, indent=2)

    def clear_all(self):
        """Clear all trackers."""
        for tracker in self.trackers.values():
            tracker.clear()
