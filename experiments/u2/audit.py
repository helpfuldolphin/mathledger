"""
PHASE-II — NOT USED IN PHASE I

Audit Logging for U2 Uplift Experiments
=======================================

This module provides audit trail logging for U2 uplift experiments.
The audit log captures detailed information about each experiment cycle
for reproducibility verification and post-hoc analysis.

**Determinism Notes:**
    - Audit entries are timestamped with cycle index (not wall clock).
    - All serialization uses sorted keys for stable output.
    - Audit log hash is computed deterministically from entries.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class AuditEntry:
    """A single audit entry for an experiment cycle.

    Attributes:
        cycle: The cycle index (0-based).
        slice_name: The experiment slice name.
        mode: The execution mode ("baseline" or "rfl").
        seed: The seed used for this cycle.
        item: The item selected for evaluation.
        result: The string representation of the result.
        success: Whether the metric evaluated to success.
        policy_score: The policy score for the item (RFL mode only).

    **Determinism Notes:**
        - All fields are immutable after creation.
        - Serialization uses sorted keys for stability.
    """

    cycle: int
    slice_name: str
    mode: str
    seed: int
    item: str
    result: str
    success: bool
    policy_score: Optional[float] = None
    label: str = field(default="PHASE II — NOT USED IN PHASE I")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary representation.

        Returns:
            A dictionary with all audit entry fields.
        """
        return asdict(self)

    def to_json(self) -> str:
        """Convert to a JSON string representation.

        Returns:
            A JSON string with sorted keys.
        """
        return json.dumps(self.to_dict(), sort_keys=True)


class AuditLogger:
    """Audit logger for U2 uplift experiments.

    The audit logger maintains a list of audit entries and provides
    methods for serialization, export, and verification.

    Attributes:
        entries: List of audit entries recorded.
        slice_name: The experiment slice name.
        mode: The execution mode.

    Example:
        >>> logger = AuditLogger("arithmetic_simple", "baseline")
        >>> logger.log_cycle(0, 12345, "1+1", "2", True)
        >>> len(logger.entries)
        1

    **Determinism Notes:**
        - Entries are stored in cycle order.
        - Audit hash is computed deterministically from all entries.
    """

    def __init__(self, slice_name: str, mode: str) -> None:
        """Initialize the audit logger.

        Args:
            slice_name: The experiment slice name.
            mode: The execution mode ("baseline" or "rfl").

        Raises:
            ValueError: If slice_name is empty or mode is invalid.
        """
        if not slice_name:
            raise ValueError("slice_name cannot be empty")
        if mode not in ("baseline", "rfl"):
            raise ValueError(f"Invalid mode: {mode}. Expected 'baseline' or 'rfl'.")

        self.slice_name = slice_name
        self.mode = mode
        self.entries: List[AuditEntry] = []

    def log_cycle(
        self,
        cycle: int,
        seed: int,
        item: str,
        result: str,
        success: bool,
        policy_score: Optional[float] = None,
    ) -> AuditEntry:
        """Log a single experiment cycle.

        Args:
            cycle: The cycle index (0-based).
            seed: The seed used for this cycle.
            item: The item selected for evaluation.
            result: The string representation of the result.
            success: Whether the metric evaluated to success.
            policy_score: The policy score (RFL mode only).

        Returns:
            The created AuditEntry.

        Raises:
            ValueError: If cycle index is invalid or out of order.
        """
        expected_cycle = len(self.entries)
        if cycle != expected_cycle:
            raise ValueError(
                f"Cycle index out of order: expected {expected_cycle}, got {cycle}. "
                f"Audit entries must be logged in sequence."
            )

        entry = AuditEntry(
            cycle=cycle,
            slice_name=self.slice_name,
            mode=self.mode,
            seed=seed,
            item=item,
            result=result,
            success=success,
            policy_score=policy_score,
        )
        self.entries.append(entry)
        return entry

    def compute_audit_hash(self) -> str:
        """Compute a deterministic hash of the audit log.

        Returns:
            The SHA-256 hash of the JSON-serialized entries.

        **Determinism Notes:**
            - Uses sorted keys for stable serialization.
            - Same entries always produce same hash.
        """
        entries_json = json.dumps(
            [e.to_dict() for e in self.entries], sort_keys=True
        )
        return hashlib.sha256(entries_json.encode("utf-8")).hexdigest()

    def export(self, path: Path) -> None:
        """Export the audit log to a JSON file.

        Args:
            path: The output file path.

        Raises:
            IOError: If the file cannot be written.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "label": "PHASE II — NOT USED IN PHASE I",
            "slice_name": self.slice_name,
            "mode": self.mode,
            "entry_count": len(self.entries),
            "audit_hash": self.compute_audit_hash(),
            "entries": [e.to_dict() for e in self.entries],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)

    def export_jsonl(self, path: Path) -> None:
        """Export the audit log as JSONL (one entry per line).

        Args:
            path: The output file path.

        Raises:
            IOError: If the file cannot be written.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for entry in self.entries:
                f.write(entry.to_json() + "\n")

    def get_success_rate(self) -> float:
        """Calculate the success rate across all logged cycles.

        Returns:
            The fraction of cycles that succeeded (0.0 to 1.0).
            Returns 0.0 if no cycles have been logged.
        """
        if not self.entries:
            return 0.0
        successes = sum(1 for e in self.entries if e.success)
        return successes / len(self.entries)

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the audit log.

        Returns:
            A dictionary with summary statistics.
        """
        return {
            "slice_name": self.slice_name,
            "mode": self.mode,
            "total_cycles": len(self.entries),
            "success_count": sum(1 for e in self.entries if e.success),
            "success_rate": self.get_success_rate(),
            "audit_hash": self.compute_audit_hash(),
        }

    @classmethod
    def load(cls, path: Path) -> "AuditLogger":
        """Load an audit log from a JSON file.

        Args:
            path: The path to the audit log file.

        Returns:
            An AuditLogger instance with loaded entries.

        Raises:
            FileNotFoundError: If the file does not exist.
            json.JSONDecodeError: If the file contains invalid JSON.
            ValueError: If the file format is invalid.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "slice_name" not in data or "mode" not in data:
            raise ValueError(
                "Invalid audit log format: missing slice_name or mode. "
                "Ensure the file was created by AuditLogger.export()."
            )

        logger = cls(data["slice_name"], data["mode"])
        for entry_dict in data.get("entries", []):
            # Reconstruct entries (preserving cycle order validation)
            logger.log_cycle(
                cycle=entry_dict["cycle"],
                seed=entry_dict["seed"],
                item=entry_dict["item"],
                result=entry_dict["result"],
                success=entry_dict["success"],
                policy_score=entry_dict.get("policy_score"),
            )
        return logger
