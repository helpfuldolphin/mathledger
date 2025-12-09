# experiments/telemetry_consistency_auditor.py

import json
import datetime
from decimal import Decimal, getcontext
from typing import Dict, Any, List, Tuple
from abc import ABC, abstractmethod
from pathlib import Path

# Ensure backend modules can be imported
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backend.telemetry.u2_schema import validate_cycle_event, validate_experiment_summary, VALID_SLICES

# Set precision for Decimal calculations
getcontext().prec = 12

class TelemetryConsistencyAuditor:
    """
    Audits a batch of telemetry data against the U2 Telemetry Consistency Contract.
    """
    def __init__(self, cycle_events: List[Dict[str, Any]], summary: Dict[str, Any]):
        self.cycle_events = cycle_events
        self.summary = summary
        self.report = {
            "audit_timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "summary_run_ids": {
                "baseline": self.summary.get("baseline_run_id"),
                "rfl": self.summary.get("rfl_run_id"),
            },
            "total_cycle_events": len(self.cycle_events),
            "checks": []
        }
        self.baseline_events = [
            e for e in self.cycle_events 
            if e.get("run_id") == self.summary.get("baseline_run_id")
        ]
        self.rfl_events = [
            e for e in self.cycle_events
            if e.get("run_id") == self.summary.get("rfl_run_id")
        ]

    def _add_check(self, name: str, passed: bool, details: str):
        self.report["checks"].append({"check_name": name, "status": "PASSED" if passed else "FAILED", "details": details})

    def check_schema_adherence(self):
        """Validates that all records conform to their basic schema."""
        try:
            for event in self.cycle_events:
                validate_cycle_event(event)
            validate_experiment_summary(self.summary)
            self._add_check("schema_adherence", True, "All records passed basic schema validation.")
            return True
        except (ValueError, TypeError) as e:
            self._add_check("schema_adherence", False, f"Schema validation failed: {e}")
            return False

    def check_cardinality(self):
        """Checks if enum fields contain only allowed values."""
        try:
            for event in self.cycle_events:
                if event.get("slice") not in VALID_SLICES:
                    raise ValueError(f"Invalid slice '{event.get('slice')}' in cycle event.")
            if self.summary.get("slice") not in VALID_SLICES:
                raise ValueError(f"Invalid slice '{self.summary.get('slice')}' in summary.")
            self._add_check("cardinality", True, "All enum fields have valid values.")
        except ValueError as e:
            self._add_check("cardinality", False, str(e))

    def check_monotonic_timestamps(self):
        """Checks if timestamps are strictly increasing per cycle within each run."""
        try:
            # Baseline run
            self.baseline_events.sort(key=lambda e: e['cycle'])
            for i in range(len(self.baseline_events) - 1):
                ts1 = self.baseline_events[i]['ts']
                ts2 = self.baseline_events[i+1]['ts']
                if ts1 >= ts2:
                    raise ValueError(f"Baseline timestamp not monotonic at cycle {self.baseline_events[i+1]['cycle']}: {ts1} >= {ts2}")
            
            # RFL run
            self.rfl_events.sort(key=lambda e: e['cycle'])
            for i in range(len(self.rfl_events) - 1):
                ts1 = self.rfl_events[i]['ts']
                ts2 = self.rfl_events[i+1]['ts']
                if ts1 >= ts2:
                    raise ValueError(f"RFL timestamp not monotonic at cycle {self.rfl_events[i+1]['cycle']}: {ts1} >= {ts2}")
            
            self._add_check("monotonic_timestamps", True, "Timestamps are strictly monotonic within each run.")
        except ValueError as e:
            self._add_check("monotonic_timestamps", False, str(e))

    def check_aggregate_consistency(self):
        """Re-calculates summary aggregates from cycle events and compares."""
        try:
            # Check cycle counts
            if len(self.baseline_events) != self.summary["n_cycles"]["baseline"]:
                raise ValueError(f"Baseline cycle count mismatch. Expected {self.summary['n_cycles']['baseline']}, found {len(self.baseline_events)}")
            if len(self.rfl_events) != self.summary["n_cycles"]["rfl"]:
                raise ValueError(f"RFL cycle count mismatch. Expected {self.summary['n_cycles']['rfl']}, found {len(self.rfl_events)}")

            # Check p_base
            base_success_count = sum(1 for e in self.baseline_events if e.get("success"))
            p_base_calc = Decimal(base_success_count) / Decimal(len(self.baseline_events)) if len(self.baseline_events) > 0 else Decimal(0)
            if abs(p_base_calc - Decimal(self.summary["p_base"])) > Decimal("1e-9"):
                 raise ValueError(f"p_base mismatch. Stored: {self.summary['p_base']}, Calculated: {p_base_calc}")

            # Check p_rfl
            rfl_success_count = sum(1 for e in self.rfl_events if e.get("success"))
            p_rfl_calc = Decimal(rfl_success_count) / Decimal(len(self.rfl_events)) if len(self.rfl_events) > 0 else Decimal(0)
            if abs(p_rfl_calc - Decimal(self.summary["p_rfl"])) > Decimal("1e-9"):
                 raise ValueError(f"p_rfl mismatch. Stored: {self.summary['p_rfl']}, Calculated: {p_rfl_calc}")

            self._add_check("aggregate_consistency", True, "p_base, p_rfl, and n_cycles are consistent with cycle events.")
        except (ValueError, ZeroDivisionError) as e:
            self._add_check("aggregate_consistency", False, str(e))

    def check_delta_arithmetic(self):
        """Checks if summary.delta = summary.p_rfl - summary.p_base."""
        try:
            delta_calc = Decimal(self.summary["p_rfl"]) - Decimal(self.summary["p_base"])
            if abs(delta_calc - Decimal(self.summary["delta"])) > Decimal("1e-9"):
                raise ValueError(f"Delta arithmetic incorrect. Stored: {self.summary['delta']}, Calculated: {delta_calc}")
            self._add_check("delta_arithmetic", True, "Summary.delta is consistent with p_rfl - p_base.")
        except ValueError as e:
            self._add_check("delta_arithmetic", False, str(e))

    def check_value_ranges(self):
        """Checks if numeric values are within their logical ranges."""
        try:
            for key in ['p_base', 'p_rfl']:
                val = self.summary.get(key)
                if not (0.0 <= val <= 1.0):
                    raise ValueError(f"{key} must be between 0 and 1, got {val}")
            
            delta = self.summary.get('delta')
            if not (-1.0 <= delta <= 1.0):
                raise ValueError(f"delta must be between -1 and 1, got {delta}")

            self._add_check("value_ranges", True, "Numeric values are within their logical ranges.")
        except (ValueError, TypeError) as e:
            self._add_check("value_ranges", False, str(e))

    def run_audit(self):
        """Runs all consistency checks."""
        # Schema check is fundamental. If it fails, don't bother with the rest.
        if self.check_schema_adherence():
            self.check_cardinality()
            self.check_value_ranges()
            self.check_monotonic_timestamps()
            self.check_aggregate_consistency()
            self.check_delta_arithmetic()
        return self.report

class AbstractDataLoader(ABC):
    """Abstract base class for data loaders."""
    @abstractmethod
    def load_data(self, source: Any) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Load data from a source and return cycle events and summary."""
        pass

class FileSystemDataLoader(AbstractDataLoader):
    """Loads telemetry data from a directory on the local file system."""
    def load_data(self, source_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Loads data from a specified directory path.
        Expects `summary.json` and `cycles.jsonl` to be present.
        """
        path = Path(source_path)
        summary_file = path / "summary.json"
        cycles_file = path / "cycles.jsonl"

        if not summary_file.is_file():
            raise FileNotFoundError(f"Summary file not found at {summary_file}")
        if not cycles_file.is_file():
            raise FileNotFoundError(f"Cycles log not found at {cycles_file}")

        print(f"Loading summary from: {summary_file}")
        with open(summary_file, 'r') as f:
            summary = json.load(f)

        print(f"Loading cycle events from: {cycles_file}")
        cycle_events = []
        with open(cycles_file, 'r') as f:
            for line in f:
                if line.strip():
                    cycle_events.append(json.loads(line))
        
        return cycle_events, summary
