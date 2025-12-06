"""
RFL Experiment Logging Adapter

Handles structured JSONL logging for RFL experiments, complying with schema_v_rfl_experiment_1.json.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from .config import RFLConfig


class RFLExperimentLogger:
    """
    Logs RFL experiment cycles to a versioned JSONL file.
    
    Path structure:
      artifacts/experiments/rfl/{experiment_id}/experiment_log.jsonl
    """

    SCHEMA_VERSION = "1.0.0"

    def __init__(self, config: RFLConfig):
        self.config = config
        self.log_dir = Path(config.artifacts_dir) / "experiments" / "rfl" / config.experiment_id
        self.log_file = self.log_dir / "experiment_log.jsonl"
        self._ensure_log_dir()

    def _ensure_log_dir(self) -> None:
        """Ensure the log directory exists."""
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def log_cycle(
        self,
        cycle_index: int,
        mode: str,
        attestation: Any,  # AttestedRunContext
        result: Any,       # RflResult
        metrics_cartographer_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Append a structured log entry for a single RFL cycle.
        
        Args:
            cycle_index: Monotonically increasing cycle number.
            mode: 'baseline' or 'rfl'.
            attestation: The input AttestedRunContext.
            result: The output RflResult.
            metrics_cartographer_data: Optional performance metrics (latency, memory).
        """
        
        # 1. Construct the entry according to schema
        entry = {
            "schema_version": self.SCHEMA_VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "run_id": result.step_id,  # Use the deterministic step_id as the unique run identifier for this cycle
            "cycle_index": cycle_index,
            "mode": mode,
            "config": {
                "experiment_id": self.config.experiment_id,
                "slice_id": attestation.slice_id,
                "policy_id": attestation.policy_id or "default",
                "seed": str(self.config.random_seed)
            },
            "attestation": {
                "h_t": attestation.composite_root,
                "r_t": attestation.reasoning_root,
                "u_t": attestation.ui_root
            },
            "metrics": {
                "abstention": {
                    "rate": attestation.abstention_rate,
                    "mass": attestation.abstention_mass,
                    "attempt_mass": float(attestation.metadata.get("attempt_mass", max(attestation.abstention_mass, 1.0))),
                    "tolerance": self.config.abstention_tolerance
                },
                "derivation": {
                    # Mapped from AttestedRunContext metadata or defaults
                    "candidates_total": int(attestation.metadata.get("max_total", 0)), # Approximation if not strictly tracked
                    "verified_count": int(attestation.metadata.get("verified_count", 0)),
                    "abstained_count": int(attestation.metadata.get("abstention_count", 0)), 
                    "depth_max": int(attestation.metadata.get("depth_max", 0))
                },
                "performance": {
                    # Default to 0 if not provided, or extract from metadata if available
                    "latency_ms": float(metrics_cartographer_data.get("latency_ms", 0.0)) if metrics_cartographer_data else float(attestation.metadata.get("latency_ms", 0.0)),
                    "throughput_ops": float(metrics_cartographer_data.get("throughput_ops", 0.0)) if metrics_cartographer_data else float(attestation.metadata.get("throughput", 0.0)),
                    "memory_mb": float(metrics_cartographer_data.get("memory_mb", 0.0)) if metrics_cartographer_data else 0.0
                }
            }
        }

        # Add RFL Law section if in RFL mode or if data is available
        if result.ledger_entry:
            entry["rfl_law"] = {
                "step_id": result.step_id,
                "symbolic_descent": result.ledger_entry.symbolic_descent,
                "policy_reward": result.ledger_entry.policy_reward,
                "mass_delta": result.abstention_mass_delta,
                "update_applied": result.policy_update_applied
            }

        # 2. Write to file (append-only)
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            # Fallback logging to stderr in case of file system failure
            print(f"[ERROR] Failed to write to experiment log: {e}")
