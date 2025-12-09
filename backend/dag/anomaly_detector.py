# backend/dag/anomaly_detector.py
"""
PHASE II - Anomaly Detector for Global DAG Evolution.

This module consumes the evolution metrics produced by the GlobalDagBuilder
and detects structural anomalies based on predefined rules.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Announce compliance on import
print("PHASE II — NOT USED IN PHASE I: Loading Anomaly Detector.", file=__import__("sys").stderr)

from backend.dag.invariant_guard import ProofDag, evaluate_dag_invariants

class AnomalyDetector:
    """Detects structural anomalies in the evolution of a Global DAG."""

    def __init__(
        self,
        metrics: List[Dict[str, Any]],
        config: Dict[str, Any] = None,
        *,
        proof_dag: Optional[ProofDag] = None,
        invariant_rules: Optional[Dict[str, Any]] = None,
    ):
        self.metrics = metrics
        # Default configuration for anomaly thresholds. Can be overridden.
        self.config = {
            "collapse_threshold": -5,
            "stagnation_duration": 10,
            "stagnation_threshold": 1,
            "branching_factor_std_dev_threshold": 3.0,
            "duplication_ratio_threshold": 0.8,
            ** (config or {}),
        }
        self.invariant_rules = invariant_rules
        self.proof_dag = proof_dag or ProofDag(metric_ledger=metrics)
        self._last_invariant_report: Optional[Dict[str, Any]] = None

    def detect_all(self) -> List[Dict[str, Any]]:
        """Runs all anomaly detectors and returns a consolidated report."""
        anomalies = []
        anomalies.extend(self.detect_proof_chain_collapse())
        anomalies.extend(self.detect_depth_stagnation())
        anomalies.extend(self.detect_explosive_branching())
        anomalies.extend(self.detect_duplicate_proof_patterns())
        if self.invariant_rules:
            self._last_invariant_report = evaluate_dag_invariants(
                self.proof_dag, self.invariant_rules
            )
        else:
            self._last_invariant_report = None
        return anomalies

    def invariant_report(self) -> Optional[Dict[str, Any]]:
        """Return the result of the last invariant evaluation, if available."""
        return self._last_invariant_report

    def detect_proof_chain_collapse(self) -> List[Dict[str, Any]]:
        """Detects sudden, significant decreases in the maximum proof depth."""
        anomalies = []
        threshold = self.config["collapse_threshold"]
        for metric in self.metrics:
            if metric.get("ΔMaxDepth(t)", 0) < threshold:
                anomalies.append({
                    "anomaly_type": "ProofChainCollapse",
                    "cycle": metric["cycle"],
                    "current_max_depth": metric["MaxDepth(t)"],
                    "delta_max_depth": metric["ΔMaxDepth(t)"],
                    "threshold_violated": threshold,
                })
        return anomalies

    def detect_depth_stagnation(self) -> List[Dict[str, Any]]:
        """Detects periods where the maximum proof depth fails to grow."""
        anomalies = []
        duration = self.config["stagnation_duration"]
        threshold = self.config["stagnation_threshold"]
        
        for i in range(len(self.metrics) - duration):
            window = self.metrics[i : i + duration]
            stagnant = all(m.get("ΔMaxDepth(t)", 0) <= threshold for m in window)
            if stagnant:
                start_cycle = window[0]["cycle"]
                end_cycle = window[-1]["cycle"]
                anomalies.append({
                    "anomaly_type": "DepthStagnation",
                    "start_cycle": start_cycle,
                    "end_cycle": end_cycle,
                    "duration_cycles": duration,
                })
        return anomalies

    def detect_explosive_branching(self) -> List[Dict[str, Any]]:
        """Detects unusually rapid increases in the global branching factor."""
        anomalies = []
        threshold = self.config["branching_factor_std_dev_threshold"]
        
        # Simple threshold check for now, a real implementation would use a moving average and std dev
        for metric in self.metrics:
            if metric["GlobalBranchingFactor(t)"] > threshold:
                anomalies.append({
                    "anomaly_type": "ExplosiveBranching",
                    "cycle": metric["cycle"],
                    "branching_factor": metric["GlobalBranchingFactor(t)"],
                    "threshold_violated": threshold,
                })
        return anomalies

    def detect_duplicate_proof_patterns(self) -> List[Dict[str, Any]]:
        """Detects cycles with a high ratio of duplicate derivations."""
        anomalies = []
        threshold = self.config["duplication_ratio_threshold"]
        for metric in self.metrics:
            total_derivations = metric.get("total_derivations_in_cycle", 0)
            if total_derivations > 0:
                duplication_ratio = metric.get("duplicate_derivations_in_cycle", 0) / total_derivations
                if duplication_ratio > threshold:
                    anomalies.append({
                        "anomaly_type": "DuplicateProofPattern",
                        "cycle": metric["cycle"],
                        "duplication_ratio": duplication_ratio,
                        "threshold_violated": threshold,
                    })
        return anomalies

    def save_report(self, anomalies: List[Dict[str, Any]], path: Path):
        """Saves the anomaly report to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(anomalies, f, indent=2)
