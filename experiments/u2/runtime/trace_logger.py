"""
PHASE II — NOT USED IN PHASE I

Trace Logger Module
===================

Provides telemetry logging utilities for U2 uplift experiments.
Extracts the telemetry record construction and JSONL writing logic
from experiments/run_uplift_u2.py.

This module provides:
    - TelemetryRecord: Frozen dataclass matching the JSONL schema
    - TraceWriter: JSONL file writer with append semantics
    - build_telemetry_record: Pure function to construct records

The field names and structure exactly match the existing JSONL schema
to ensure backward compatibility.

Example:
    >>> record = build_telemetry_record(
    ...     cycle=0,
    ...     slice_name="test_slice",
    ...     mode="baseline",
    ...     cycle_seed=42,
    ...     chosen_item="1+1",
    ...     result={"outcome": "VERIFIED"},
    ...     success=True
    ... )
    >>> writer = TraceWriter(Path("results.jsonl"))
    >>> writer.write(record)
    >>> writer.flush()
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional


# Phase label constant - must match original implementation exactly
PHASE_II_LABEL = "PHASE II — NOT USED IN PHASE I"


@dataclass(frozen=True)
class TelemetryRecord:
    """
    Immutable telemetry record for a single experiment cycle.

    Field names exactly match the JSONL schema from the original
    experiments/run_uplift_u2.py implementation:

    ```python
    telemetry_record = {
        "cycle": i,
        "slice": slice_name,
        "mode": mode,
        "seed": cycle_seed,
        "item": chosen_item,
        "result": str(mock_result),
        "success": success,
        "label": "PHASE II — NOT USED IN PHASE I",
    }
    ```

    Attributes:
        cycle: Zero-based cycle index.
        slice: Name of the curriculum slice.
        mode: Execution mode ("baseline" or "rfl").
        seed: Cycle-specific seed value.
        item: The chosen item for this cycle.
        result: String representation of the result dictionary.
        success: Whether the cycle succeeded.
        label: Phase label (always PHASE II — NOT USED IN PHASE I).

    Example:
        >>> record = TelemetryRecord(
        ...     cycle=0,
        ...     slice="arithmetic_simple",
        ...     mode="baseline",
        ...     seed=123456,
        ...     item="1+1",
        ...     result="{'outcome': 'VERIFIED'}",
        ...     success=True
        ... )
    """

    cycle: int
    slice: str
    mode: str
    seed: int
    item: str
    result: str
    success: bool
    label: str = PHASE_II_LABEL

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for JSON serialization.

        Returns:
            Dictionary with all fields in schema order.
        """
        return {
            "cycle": self.cycle,
            "slice": self.slice,
            "mode": self.mode,
            "seed": self.seed,
            "item": self.item,
            "result": self.result,
            "success": self.success,
            "label": self.label,
        }

    def to_json(self) -> str:
        """
        Serialize to JSON string.

        Returns:
            JSON string suitable for JSONL output.
        """
        return json.dumps(self.to_dict())


def build_telemetry_record(
    cycle: int,
    slice_name: str,
    mode: str,
    cycle_seed: int,
    chosen_item: Any,
    result: Dict[str, Any],
    success: bool,
) -> TelemetryRecord:
    """
    Build a TelemetryRecord from cycle execution data.

    This pure function constructs a telemetry record matching the
    exact schema of the original implementation. The result dictionary
    is converted to string representation as in the original:
    `"result": str(mock_result)`

    Args:
        cycle: Zero-based cycle index.
        slice_name: Name of the curriculum slice.
        mode: Execution mode ("baseline" or "rfl").
        cycle_seed: The seed used for this cycle.
        chosen_item: The item selected for this cycle.
        result: Raw result dictionary from substrate execution.
        success: Whether the cycle succeeded.

    Returns:
        Immutable TelemetryRecord ready for logging.

    Example:
        >>> record = build_telemetry_record(
        ...     cycle=0,
        ...     slice_name="test",
        ...     mode="baseline",
        ...     cycle_seed=42,
        ...     chosen_item="1+1",
        ...     result={"outcome": "VERIFIED"},
        ...     success=True
        ... )
        >>> record.label
        'PHASE II — NOT USED IN PHASE I'
    """
    return TelemetryRecord(
        cycle=cycle,
        slice=slice_name,
        mode=mode,
        seed=cycle_seed,
        item=str(chosen_item),
        result=str(result),
        success=success,
        label=PHASE_II_LABEL,
    )


class TraceWriter:
    """
    JSONL trace file writer with append semantics.

    This class handles writing telemetry records to a JSONL file,
    matching the output format of the original implementation:

    ```python
    results_f.write(json.dumps(telemetry_record) + "\\n")
    ```

    Attributes:
        path: Path to the output JSONL file.

    Example:
        >>> writer = TraceWriter(Path("results/trace.jsonl"))
        >>> record = build_telemetry_record(...)
        >>> writer.write(record)
        >>> writer.flush()
    """

    def __init__(self, path: Path) -> None:
        """
        Initialize the trace writer.

        Args:
            path: Path to the output JSONL file.
                Parent directories will be created if needed.
        """
        self.path = Path(path)
        self._file = None
        self._records_written = 0

    def _ensure_open(self) -> None:
        """Ensure the file is open for writing."""
        if self._file is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(self.path, "w", encoding="utf-8")

    def write(self, record: TelemetryRecord) -> None:
        """
        Write a telemetry record to the JSONL file.

        Args:
            record: The TelemetryRecord to write.
        """
        self._ensure_open()
        self._file.write(record.to_json() + "\n")
        self._records_written += 1

    def write_dict(self, record_dict: Dict[str, Any]) -> None:
        """
        Write a raw dictionary to the JSONL file.

        This method allows writing records that don't use the
        TelemetryRecord dataclass, for backward compatibility.

        Args:
            record_dict: Dictionary to serialize and write.
        """
        self._ensure_open()
        self._file.write(json.dumps(record_dict) + "\n")
        self._records_written += 1

    def flush(self) -> None:
        """Flush buffered writes to disk."""
        if self._file is not None:
            self._file.flush()

    def close(self) -> None:
        """Close the file handle."""
        if self._file is not None:
            self._file.close()
            self._file = None

    @property
    def records_written(self) -> int:
        """Return the number of records written."""
        return self._records_written

    def __enter__(self) -> "TraceWriter":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - closes the file."""
        self.close()


class TraceReader:
    """
    JSONL trace file reader for verification and analysis.

    This class reads telemetry records from a JSONL file,
    useful for testing and verification.

    Example:
        >>> reader = TraceReader(Path("results/trace.jsonl"))
        >>> for record in reader.read_all():
        ...     print(record.cycle, record.success)
    """

    def __init__(self, path: Path) -> None:
        """
        Initialize the trace reader.

        Args:
            path: Path to the JSONL file to read.
        """
        self.path = Path(path)

    def read_all(self) -> list[TelemetryRecord]:
        """
        Read all records from the JSONL file.

        Returns:
            List of TelemetryRecord objects.

        Raises:
            FileNotFoundError: If the file doesn't exist.
        """
        records = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    records.append(TelemetryRecord(
                        cycle=data["cycle"],
                        slice=data["slice"],
                        mode=data["mode"],
                        seed=data["seed"],
                        item=data["item"],
                        result=data["result"],
                        success=data["success"],
                        label=data.get("label", PHASE_II_LABEL),
                    ))
        return records

    def read_dicts(self) -> list[Dict[str, Any]]:
        """
        Read all records as raw dictionaries.

        Returns:
            List of dictionaries.
        """
        records = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records


__all__ = [
    "PHASE_II_LABEL",
    "TelemetryRecord",
    "TraceWriter",
    "TraceReader",
    "build_telemetry_record",
]

