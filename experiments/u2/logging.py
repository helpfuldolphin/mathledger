"""
U2 Planner Trace Logger

Provides structured logging for:
- Determinism verification
- Replay debugging
- RFL evidence generation
- Performance analysis
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Set, TextIO

from .schema import (
    EventType,
    TraceEvent,
    CycleTrace,
    ExperimentTrace,
    CORE_EVENTS,
    ALL_EVENT_TYPES,
)


class U2TraceLogger:
    """
    Structured trace logger for U2 planner.
    
    DESIGN:
    - Append-only JSONL format for streaming
    - Configurable event filtering
    - Automatic cycle trace management
    - Deterministic timestamp generation
    """
    
    def __init__(
        self,
        output_path: Path,
        experiment_id: str,
        slice_name: str,
        mode: str,
        master_seed: str,
        event_filter: Optional[Set[EventType]] = None,
    ):
        """
        Initialize trace logger.
        
        Args:
            output_path: Path to JSONL output file
            experiment_id: Unique experiment identifier
            slice_name: Slice being executed
            mode: "baseline" or "rfl"
            master_seed: Master PRNG seed
            event_filter: Set of event types to log (None = all)
        """
        self.output_path = output_path
        self.experiment_id = experiment_id
        self.slice_name = slice_name
        self.mode = mode
        self.master_seed = master_seed
        self.event_filter = event_filter or ALL_EVENT_TYPES
        
        self.current_cycle: Optional[CycleTrace] = None
        self.file_handle: Optional[TextIO] = None
        self.start_time_ms = int(time.time() * 1000)
        
        # Open output file
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.file_handle = open(self.output_path, 'w', encoding='utf-8')
        
        # Write experiment header
        self._write_header()
    
    def _write_header(self) -> None:
        """Write experiment metadata header."""
        header = {
            "type": "experiment_header",
            "experiment_id": self.experiment_id,
            "slice_name": self.slice_name,
            "mode": self.mode,
            "master_seed": self.master_seed,
            "start_time_ms": self.start_time_ms,
        }
        self._write_line(header)
    
    def _write_line(self, data: Dict[str, Any]) -> None:
        """Write JSONL line."""
        if self.file_handle:
            line = json.dumps(data, sort_keys=True, separators=(',', ':'))
            self.file_handle.write(line + '\n')
            self.file_handle.flush()
    
    def start_cycle(self, cycle: int, seed: str) -> None:
        """
        Start a new cycle trace.
        
        Args:
            cycle: Cycle number
            seed: Cycle-specific seed
        """
        if self.current_cycle is not None:
            # Finalize previous cycle
            self.end_cycle()
        
        self.current_cycle = CycleTrace(cycle=cycle, seed=seed)
        
        # Log cycle start event
        self.log_event(
            EventType.CYCLE_START,
            cycle=cycle,
            data={"seed": seed}
        )
    
    def end_cycle(self, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        End current cycle trace and write to file.
        
        Args:
            metadata: Optional cycle metadata
        """
        if self.current_cycle is None:
            return
        
        # Log cycle end event
        self.log_event(
            EventType.CYCLE_END,
            cycle=self.current_cycle.cycle,
            data=metadata or {}
        )
        
        # Add metadata
        if metadata:
            self.current_cycle.metadata.update(metadata)
        
        # Write cycle trace
        cycle_data = {
            "type": "cycle_trace",
            **self.current_cycle.to_dict()
        }
        self._write_line(cycle_data)
        
        self.current_cycle = None
    
    def log_event(
        self,
        event_type: EventType,
        cycle: int,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log a trace event.
        
        Args:
            event_type: Type of event
            cycle: Cycle number
            data: Event-specific data
        """
        # Filter events
        if event_type not in self.event_filter:
            return
        
        # Create event
        timestamp_ms = int(time.time() * 1000) - self.start_time_ms
        event = TraceEvent(
            timestamp_ms=timestamp_ms,
            event_type=event_type,
            cycle=cycle,
            data=data or {},
        )
        
        # Add to current cycle trace
        if self.current_cycle and self.current_cycle.cycle == cycle:
            self.current_cycle.add_event(event)
    
    def close(self) -> None:
        """Close trace logger and finalize output."""
        # Finalize current cycle if any
        if self.current_cycle is not None:
            self.end_cycle()
        
        # Write footer
        footer = {
            "type": "experiment_footer",
            "experiment_id": self.experiment_id,
            "end_time_ms": int(time.time() * 1000),
        }
        self._write_line(footer)
        
        # Close file
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def load_experiment_trace(trace_path: Path) -> ExperimentTrace:
    """
    Load experiment trace from JSONL file.
    
    Args:
        trace_path: Path to trace JSONL file
        
    Returns:
        ExperimentTrace object
    """
    experiment_trace: Optional[ExperimentTrace] = None
    
    with open(trace_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            
            if data.get("type") == "experiment_header":
                experiment_trace = ExperimentTrace(
                    experiment_id=data["experiment_id"],
                    slice_name=data["slice_name"],
                    mode=data["mode"],
                    master_seed=data["master_seed"],
                )
            
            elif data.get("type") == "cycle_trace" and experiment_trace:
                cycle_trace = CycleTrace.from_dict(data)
                experiment_trace.add_cycle(cycle_trace)
    
    if experiment_trace is None:
        raise ValueError(f"Invalid trace file: {trace_path}")
    
    return experiment_trace


def verify_trace_determinism(trace1_path: Path, trace2_path: Path) -> bool:
    """
    Verify that two trace files are identical (determinism check).
    
    Args:
        trace1_path: First trace file
        trace2_path: Second trace file
        
    Returns:
        True if traces match, False otherwise
    """
    trace1 = load_experiment_trace(trace1_path)
    trace2 = load_experiment_trace(trace2_path)
    
    # Check experiment metadata
    if (trace1.experiment_id != trace2.experiment_id or
        trace1.slice_name != trace2.slice_name or
        trace1.mode != trace2.mode or
        trace1.master_seed != trace2.master_seed):
        return False
    
    # Check cycle count
    if len(trace1.cycles) != len(trace2.cycles):
        return False
    
    # Check each cycle hash
    for c1, c2 in zip(trace1.cycles, trace2.cycles):
        if c1.hash() != c2.hash():
            return False
    
    return True
