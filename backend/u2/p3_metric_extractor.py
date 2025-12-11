# REAL-READY
"""
U2 → P3 Metric Extractor

This module implements the U2 → P3 Integration Contract, extracting
P3-level performance metrics (Ω, Δp, RSI) from U2 JSONL trace logs.

Implements: docs/u2_p3_integration_contract.md

Author: Manus-F
Date: 2025-12-06
Status: REAL-READY
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set


# ============================================================================
# P3 METRICS DATA STRUCTURES
# ============================================================================

@dataclass
class P3Metrics:
    """
    P3-level performance metrics derived from U2 traces.
    """
    omega: Set[str]  # Set of unique proven statement hashes
    delta_p: int  # Count of unique proven statements
    rsi: float  # Reasoning Step Intensity (executions/second)
    
    # Metadata
    total_executions: int
    total_wall_time_seconds: float
    min_timestamp_ms: int
    max_timestamp_ms: int
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "omega": sorted(list(self.omega)),  # Sorted for determinism
            "delta_p": self.delta_p,
            "rsi": self.rsi,
            "metadata": {
                "total_executions": self.total_executions,
                "total_wall_time_seconds": self.total_wall_time_seconds,
                "min_timestamp_ms": self.min_timestamp_ms,
                "max_timestamp_ms": self.max_timestamp_ms,
            }
        }


# ============================================================================
# P3 METRIC EXTRACTOR
# ============================================================================

class P3MetricExtractor:
    """
    Extracts P3 metrics from U2 JSONL trace logs.
    
    Implements the U2 → P3 Integration Contract with guaranteed determinism.
    """
    
    def extract(self, trace_path: Path) -> P3Metrics:
        """
        Extract P3 metrics from U2 trace log.
        
        Args:
            trace_path: Path to U2 trace JSONL file
            
        Returns:
            P3Metrics object
        """
        # Read and parse all execution events
        execution_events = self._read_execution_events(trace_path)
        
        # Sort events for deterministic processing
        sorted_events = self._sort_events_canonically(execution_events)
        
        # Extract Ω (omega): set of unique proven statements
        omega = self._extract_omega(sorted_events)
        
        # Compute Δp (delta-p): count of unique proven statements
        delta_p = len(omega)
        
        # Compute RSI (Reasoning Step Intensity)
        rsi, metadata = self._compute_rsi(sorted_events)
        
        return P3Metrics(
            omega=omega,
            delta_p=delta_p,
            rsi=rsi,
            total_executions=metadata["total_executions"],
            total_wall_time_seconds=metadata["total_wall_time_seconds"],
            min_timestamp_ms=metadata["min_timestamp_ms"],
            max_timestamp_ms=metadata["max_timestamp_ms"],
        )
    
    def _read_execution_events(self, trace_path: Path) -> List[Dict]:
        """
        Read all execution events from trace log.
        
        Args:
            trace_path: Path to trace JSONL file
            
        Returns:
            List of execution event dictionaries
        """
        events = []
        with open(trace_path, 'r') as f:
            for line in f:
                event = json.loads(line)
                if event.get("event_type") == "execution":
                    events.append(event)
        return events
    
    def _sort_events_canonically(self, events: List[Dict]) -> List[Dict]:
        """
        Sort events in canonical order for deterministic processing.
        
        Canonical order: (cycle, worker_id, statement.hash)
        
        Args:
            events: List of execution events
            
        Returns:
            Sorted list of events
        """
        return sorted(events, key=lambda e: (
            e.get("cycle", 0),
            e.get("worker_id", 0),
            e.get("data", {}).get("statement", {}).get("hash", ""),
        ))
    
    def _extract_omega(self, events: List[Dict]) -> Set[str]:
        """
        Extract Ω: set of unique proven statement hashes.
        
        Ω is defined as the set of all unique statements proven to be
        tautologies during the experiment.
        
        Args:
            events: Sorted list of execution events
            
        Returns:
            Set of SHA-256 hashes
        """
        omega = set()
        for event in events:
            data = event.get("data", {})
            if data.get("is_tautology") == True:
                statement_hash = data.get("statement", {}).get("hash")
                if statement_hash:
                    omega.add(statement_hash)
        return omega
    
    def _compute_rsi(self, events: List[Dict]) -> tuple:
        """
        Compute RSI: Reasoning Step Intensity (executions per second).
        
        RSI = total_executions / total_wall_time_seconds
        
        Args:
            events: Sorted list of execution events
            
        Returns:
            Tuple of (rsi, metadata)
        """
        if not events:
            return 0.0, {
                "total_executions": 0,
                "total_wall_time_seconds": 0.0,
                "min_timestamp_ms": 0,
                "max_timestamp_ms": 0,
            }
        
        total_executions = len(events)
        
        timestamps = [e.get("timestamp_ms", 0) for e in events]
        min_timestamp = min(timestamps)
        max_timestamp = max(timestamps)
        
        total_wall_time_seconds = (max_timestamp - min_timestamp) / 1000.0
        
        if total_wall_time_seconds == 0:
            rsi = 0.0
        else:
            rsi = total_executions / total_wall_time_seconds
        
        metadata = {
            "total_executions": total_executions,
            "total_wall_time_seconds": total_wall_time_seconds,
            "min_timestamp_ms": min_timestamp,
            "max_timestamp_ms": max_timestamp,
        }
        
        return rsi, metadata
    
    def save_metrics(self, metrics: P3Metrics, output_path: Path):
        """
        Save P3 metrics to JSON file.
        
        Args:
            metrics: P3Metrics object
            output_path: Path to output JSON file
        """
        with open(output_path, 'w') as f:
            json.dump(metrics.to_dict(), f, indent=2, sort_keys=True)


# ============================================================================
# COMMAND-LINE INTERFACE
# ============================================================================

def main():
    """Command-line interface for P3 metric extraction."""
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python p3_metric_extractor.py <trace_path> <output_path>")
        sys.exit(1)
    
    trace_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    
    if not trace_path.exists():
        print(f"Error: Trace file not found: {trace_path}")
        sys.exit(1)
    
    extractor = P3MetricExtractor()
    metrics = extractor.extract(trace_path)
    extractor.save_metrics(metrics, output_path)
    
    print(f"P3 Metrics extracted:")
    print(f"  Ω (omega): {len(metrics.omega)} unique proven statements")
    print(f"  Δp (delta-p): {metrics.delta_p}")
    print(f"  RSI: {metrics.rsi:.2f} executions/second")
    print(f"  Total executions: {metrics.total_executions}")
    print(f"  Wall time: {metrics.total_wall_time_seconds:.2f}s")
    print(f"\nMetrics saved to: {output_path}")


if __name__ == "__main__":
    main()
