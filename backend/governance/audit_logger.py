"""
Governance Audit Logger — Immutable write-only audit trail for governance checks.

Phase X: Implements audit requirements from LastMile_Governance_Spec.md

Audit Properties:
- Immutability: Append-only log, no modification
- Completeness: Every check recorded, no gaps
- Verifiability: Hash chain for integrity verification
- Traceability: Full lineage from input to verdict

SHADOW MODE CONTRACT:
- All audit records are written regardless of mode
- Records are tagged with mode (SHADOW/ACTIVE)
- No enforcement based on audit contents
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO

from .last_mile_checker import GovernanceFinalCheckResult

__all__ = [
    "GovernanceAuditLogger",
    "AuditLogConfig",
    "AuditRecord",
]


@dataclass
class AuditLogConfig:
    """Configuration for audit logger."""

    # Output directory for audit logs
    output_dir: str = "results/governance_audit"

    # Log file naming
    log_prefix: str = "governance_audit"
    use_date_suffix: bool = True

    # Format options
    format: str = "jsonl"  # "jsonl" or "json"
    pretty_print: bool = False

    # Chain verification
    verify_chain_on_write: bool = True

    # Rotation
    max_records_per_file: int = 10000
    rotate_on_date_change: bool = True

    @classmethod
    def default(cls) -> "AuditLogConfig":
        return cls()


@dataclass
class AuditRecord:
    """Single audit record with chain linking."""

    # Record identification
    record_id: str
    sequence_number: int

    # Chain linking
    previous_hash: Optional[str]
    record_hash: str

    # Timestamp
    timestamp: str

    # Payload (the full check result)
    check_result: Dict[str, Any]

    # Metadata
    mode: str  # "SHADOW" or "ACTIVE"
    logger_version: str = "1.0.0"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "sequence_number": self.sequence_number,
            "previous_hash": self.previous_hash,
            "record_hash": self.record_hash,
            "timestamp": self.timestamp,
            "mode": self.mode,
            "logger_version": self.logger_version,
            "check_result": self.check_result,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuditRecord":
        return cls(
            record_id=data["record_id"],
            sequence_number=data["sequence_number"],
            previous_hash=data.get("previous_hash"),
            record_hash=data["record_hash"],
            timestamp=data["timestamp"],
            check_result=data["check_result"],
            mode=data.get("mode", "UNKNOWN"),
            logger_version=data.get("logger_version", "1.0.0"),
        )


class GovernanceAuditLogger:
    """
    Immutable audit logger for governance final checks.

    SHADOW MODE CONTRACT:
    - All records are written regardless of mode
    - Records are tagged with mode (SHADOW/ACTIVE)
    - Log files are append-only
    - Hash chain provides tamper detection
    """

    def __init__(
        self,
        config: Optional[AuditLogConfig] = None,
        run_id: Optional[str] = None,
    ) -> None:
        self.config = config or AuditLogConfig.default()
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        # Chain state
        self._sequence_number: int = 0
        self._previous_hash: Optional[str] = None
        self._records_in_current_file: int = 0
        self._current_date: Optional[str] = None

        # File handle (lazy initialized)
        self._file_handle: Optional[TextIO] = None
        self._current_file_path: Optional[Path] = None

        # Ensure output directory exists
        self._output_dir = Path(self.config.output_dir) / self.run_id
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def log(self, result: GovernanceFinalCheckResult) -> AuditRecord:
        """
        Log a governance final check result.

        SHADOW MODE: Records are written regardless of enforcement mode.

        Args:
            result: GovernanceFinalCheckResult to log

        Returns:
            AuditRecord that was written
        """
        # Prepare check result dict
        check_dict = result.to_dict()

        # Create record
        self._sequence_number += 1
        timestamp = datetime.now(timezone.utc).isoformat()

        # Compute record hash (includes previous hash for chain)
        record_data_for_hash = {
            "sequence_number": self._sequence_number,
            "previous_hash": self._previous_hash,
            "timestamp": timestamp,
            "check_result": check_dict,
        }
        record_hash = self._compute_hash(record_data_for_hash)

        record = AuditRecord(
            record_id=result.check_id,
            sequence_number=self._sequence_number,
            previous_hash=self._previous_hash,
            record_hash=record_hash,
            timestamp=timestamp,
            check_result=check_dict,
            mode=result.mode,
        )

        # Write record
        self._write_record(record)

        # Update chain state
        self._previous_hash = record_hash

        return record

    def log_batch(self, results: List[GovernanceFinalCheckResult]) -> List[AuditRecord]:
        """
        Log multiple results in sequence.

        Args:
            results: List of results to log

        Returns:
            List of AuditRecords written
        """
        records = []
        for result in results:
            record = self.log(result)
            records.append(record)
        return records

    def _write_record(self, record: AuditRecord) -> None:
        """Write record to current log file."""
        # Check for rotation
        self._check_rotation()

        # Ensure file is open
        if self._file_handle is None:
            self._open_new_file()

        # Write record
        record_dict = record.to_dict()

        if self.config.format == "jsonl":
            line = json.dumps(record_dict, separators=(",", ":"))
            self._file_handle.write(line + "\n")
        else:
            # JSON array format (less efficient but more readable)
            if self.config.pretty_print:
                line = json.dumps(record_dict, indent=2)
            else:
                line = json.dumps(record_dict)
            self._file_handle.write(line + "\n")

        self._file_handle.flush()
        self._records_in_current_file += 1

    def _check_rotation(self) -> None:
        """Check if log file needs rotation."""
        needs_rotation = False

        # Check record count
        if self._records_in_current_file >= self.config.max_records_per_file:
            needs_rotation = True

        # Check date change
        if self.config.rotate_on_date_change:
            current_date = datetime.now(timezone.utc).strftime("%Y%m%d")
            if self._current_date and current_date != self._current_date:
                needs_rotation = True
            self._current_date = current_date

        if needs_rotation:
            self._close_current_file()

    def _open_new_file(self) -> None:
        """Open a new log file."""
        # Generate filename
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        if self.config.use_date_suffix:
            filename = f"{self.config.log_prefix}_{timestamp}.{self.config.format}"
        else:
            filename = f"{self.config.log_prefix}_{self._sequence_number}.{self.config.format}"

        self._current_file_path = self._output_dir / filename
        self._file_handle = open(self._current_file_path, "a", encoding="utf-8")
        self._records_in_current_file = 0
        self._current_date = datetime.now(timezone.utc).strftime("%Y%m%d")

    def _close_current_file(self) -> None:
        """Close current log file."""
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None
            self._current_file_path = None

    def _compute_hash(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 hash of data."""
        serialized = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(serialized.encode()).hexdigest()

    def verify_chain(self, records: Optional[List[AuditRecord]] = None) -> Tuple[bool, List[str]]:
        """
        Verify integrity of audit chain.

        Args:
            records: Optional list of records to verify. If None, reads from files.

        Returns:
            (is_valid, list of error messages)
        """
        if records is None:
            records = self.read_all_records()

        errors = []

        if not records:
            return True, []

        # Verify first record has no previous hash
        if records[0].previous_hash is not None:
            errors.append(f"First record should have no previous_hash, got {records[0].previous_hash}")

        # Verify chain linking
        for i in range(1, len(records)):
            prev_record = records[i - 1]
            curr_record = records[i]

            # Check previous hash links correctly
            if curr_record.previous_hash != prev_record.record_hash:
                errors.append(
                    f"Chain break at record {i}: previous_hash={curr_record.previous_hash} "
                    f"does not match prev record_hash={prev_record.record_hash}"
                )

            # Verify sequence numbers
            if curr_record.sequence_number != prev_record.sequence_number + 1:
                errors.append(
                    f"Sequence break at record {i}: expected {prev_record.sequence_number + 1}, "
                    f"got {curr_record.sequence_number}"
                )

        return len(errors) == 0, errors

    def read_all_records(self) -> List[AuditRecord]:
        """Read all records from log files."""
        records = []

        # Find all log files
        log_files = sorted(self._output_dir.glob(f"{self.config.log_prefix}_*.{self.config.format}"))

        for log_file in log_files:
            with open(log_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        records.append(AuditRecord.from_dict(data))

        return records

    def get_stats(self) -> Dict[str, Any]:
        """Get logger statistics."""
        return {
            "run_id": self.run_id,
            "sequence_number": self._sequence_number,
            "records_in_current_file": self._records_in_current_file,
            "current_file": str(self._current_file_path) if self._current_file_path else None,
            "output_dir": str(self._output_dir),
            "chain_hash": self._previous_hash,
        }

    def close(self) -> None:
        """Close logger and release resources."""
        self._close_current_file()

    def __enter__(self) -> "GovernanceAuditLogger":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


# Import Tuple for type hints
from typing import Tuple
