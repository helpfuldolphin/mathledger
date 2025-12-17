"""
USLAShadowLogger — Structured shadow logging for USLA simulator.

Phase X: SHADOW MODE ONLY

This module provides structured logging for USLA shadow mode operations.
All USLA state transitions, divergences, and alerts are logged here
for offline analysis without affecting real governance.

SHADOW MODE CONTRACT:
1. The USLA simulator NEVER modifies real governance decisions
2. Disagreements are LOGGED, not ACTED upon
3. No cycle is blocked or allowed based on simulator output
4. The simulator runs AFTER the real governance decision
5. All USLA state is written to shadow logs only

Usage:
    from backend.topology.usla_shadow import USLAShadowLogger

    logger = USLAShadowLogger(log_dir="results/usla_shadow")
    logger.log_cycle(cycle, state, real_blocked, sim_blocked, ...)
    logger.close()
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO

__all__ = [
    "USLAShadowLogger",
    "ShadowLogEntry",
    "ShadowLogConfig",
]


@dataclass
class ShadowLogConfig:
    """Configuration for shadow logging."""
    log_dir: str = "results/usla_shadow"
    runner_id: str = "unknown"
    run_id: Optional[str] = None
    log_every_n_cycles: int = 1  # Log every N cycles (1 = all)
    include_full_state: bool = True
    include_divergence_detail: bool = True
    flush_every_n: int = 10  # Flush to disk every N entries

    @classmethod
    def default(cls) -> "ShadowLogConfig":
        return cls()


@dataclass
class ShadowLogEntry:
    """
    Single entry in the shadow log.

    Schema matches Phase X Integration Spec Section 2.3.
    """
    cycle: int
    timestamp: str
    runner: str
    input: Dict[str, Any]
    state: Dict[str, Any]
    hard_ok: bool
    in_safe_region: bool
    real_blocked: bool
    sim_blocked: bool
    governance_aligned: bool
    translation_quality: str = "full"
    fallbacks_used: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cycle": self.cycle,
            "timestamp": self.timestamp,
            "runner": self.runner,
            "input": self.input,
            "state": self.state,
            "hard_ok": self.hard_ok,
            "in_safe_region": self.in_safe_region,
            "real_blocked": self.real_blocked,
            "sim_blocked": self.sim_blocked,
            "governance_aligned": self.governance_aligned,
            "translation_quality": self.translation_quality,
            "fallbacks_used": self.fallbacks_used,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"))


class USLAShadowLogger:
    """
    Structured shadow logger for USLA simulator output.

    Writes JSONL format for easy analysis and streaming.
    """

    def __init__(self, config: Optional[ShadowLogConfig] = None):
        self.config = config or ShadowLogConfig.default()
        self._entries: List[ShadowLogEntry] = []
        self._file: Optional[TextIO] = None
        self._file_path: Optional[Path] = None
        self._entry_count: int = 0
        self._divergence_count: int = 0

        # Initialize log file
        self._init_log_file()

    def _init_log_file(self) -> None:
        """Initialize the log file."""
        log_dir = Path(self.config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_id = self.config.run_id or timestamp
        filename = f"usla_shadow_{self.config.runner_id}_{run_id}.jsonl"

        self._file_path = log_dir / filename
        self._file = open(self._file_path, "w", encoding="utf-8")

        # Write header comment (as JSON for parsability)
        header = {
            "_header": True,
            "schema_version": "1.0.0",
            "runner": self.config.runner_id,
            "run_id": run_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "mode": "SHADOW",
        }
        self._file.write(json.dumps(header) + "\n")

    def log_cycle(
        self,
        cycle: int,
        state_dict: Dict[str, Any],
        input_dict: Dict[str, Any],
        real_blocked: bool,
        sim_blocked: bool,
        hard_ok: bool,
        in_safe_region: bool,
        translation_quality: str = "full",
        fallbacks_used: Optional[List[str]] = None,
    ) -> ShadowLogEntry:
        """
        Log a single cycle.

        Returns the log entry created.
        """
        # Check if we should log this cycle
        if self.config.log_every_n_cycles > 1:
            if cycle % self.config.log_every_n_cycles != 0:
                return None

        entry = ShadowLogEntry(
            cycle=cycle,
            timestamp=datetime.now(timezone.utc).isoformat(),
            runner=self.config.runner_id,
            input=input_dict if self.config.include_full_state else {},
            state=state_dict if self.config.include_full_state else self._minimal_state(state_dict),
            hard_ok=hard_ok,
            in_safe_region=in_safe_region,
            real_blocked=real_blocked,
            sim_blocked=sim_blocked,
            governance_aligned=(real_blocked == sim_blocked),
            translation_quality=translation_quality,
            fallbacks_used=fallbacks_used or [],
        )

        self._entries.append(entry)
        self._entry_count += 1

        if not entry.governance_aligned:
            self._divergence_count += 1

        # Write to file
        if self._file:
            self._file.write(entry.to_json() + "\n")

            # Periodic flush
            if self._entry_count % self.config.flush_every_n == 0:
                self._file.flush()

        return entry

    def _minimal_state(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Extract minimal state for compact logging."""
        return {
            "H": state_dict.get("H"),
            "rho": state_dict.get("rho"),
            "tau": state_dict.get("tau"),
            "beta": state_dict.get("beta"),
            "C": state_dict.get("C"),
            "delta": state_dict.get("delta"),
        }

    def log_divergence_alert(
        self,
        cycle: int,
        severity: str,
        field: str,
        real_value: Any,
        sim_value: Any,
        consecutive_cycles: int,
    ) -> None:
        """Log a divergence alert."""
        if not self.config.include_divergence_detail:
            return

        alert = {
            "_alert": True,
            "cycle": cycle,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "severity": severity,
            "field": field,
            "real_value": real_value,
            "sim_value": sim_value,
            "consecutive_cycles": consecutive_cycles,
        }

        if self._file:
            self._file.write(json.dumps(alert) + "\n")
            self._file.flush()

    def log_abort(self, cycle: int, reason: str) -> None:
        """Log an abort event."""
        abort = {
            "_abort": True,
            "cycle": cycle,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "reason": reason,
        }

        if self._file:
            self._file.write(json.dumps(abort) + "\n")
            self._file.flush()

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if not self._entries:
            return {
                "total_cycles": 0,
                "divergences": 0,
                "divergence_rate": 0.0,
            }

        hard_ok_count = sum(1 for e in self._entries if e.hard_ok)
        safe_region_count = sum(1 for e in self._entries if e.in_safe_region)

        return {
            "total_cycles": self._entry_count,
            "divergences": self._divergence_count,
            "divergence_rate": self._divergence_count / self._entry_count if self._entry_count > 0 else 0.0,
            "hard_ok_rate": hard_ok_count / self._entry_count if self._entry_count > 0 else 0.0,
            "safe_region_rate": safe_region_count / self._entry_count if self._entry_count > 0 else 0.0,
            "log_file": str(self._file_path) if self._file_path else None,
        }

    def close(self) -> Dict[str, Any]:
        """
        Close the logger and write final summary.

        Returns the summary statistics.
        """
        summary = self.get_summary()

        # Write footer
        if self._file:
            footer = {
                "_footer": True,
                "ended_at": datetime.now(timezone.utc).isoformat(),
                "summary": summary,
            }
            self._file.write(json.dumps(footer) + "\n")
            self._file.close()
            self._file = None

        return summary

    def __enter__(self) -> "USLAShadowLogger":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    @property
    def file_path(self) -> Optional[Path]:
        """Get the log file path."""
        return self._file_path

    @property
    def entry_count(self) -> int:
        """Get total entries logged."""
        return self._entry_count

    @property
    def divergence_count(self) -> int:
        """Get total divergences detected."""
        return self._divergence_count
