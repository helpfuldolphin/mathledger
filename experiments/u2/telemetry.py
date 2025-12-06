"""
U2 Planner Telemetry Export

Exports telemetry for:
- RFL Evidence Packs
- Performance analysis
- Determinism verification
- Audit trails
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .logging import load_experiment_trace
from .schema import EventType, ExperimentTrace


@dataclass
class TelemetryReport:
    """
    Comprehensive telemetry report for U2 planner.
    
    Includes:
    - Experiment metadata
    - Per-cycle statistics
    - Frontier dynamics
    - Policy effectiveness
    - Determinism verification data
    """
    
    experiment_id: str
    slice_name: str
    mode: str
    master_seed: str
    
    # Summary statistics
    total_cycles: int = 0
    total_candidates_processed: int = 0
    total_candidates_generated: int = 0
    total_time_s: float = 0.0
    
    # Per-cycle data
    cycle_stats: List[Dict[str, Any]] = field(default_factory=list)
    
    # Frontier telemetry
    frontier_telemetry: Dict[str, Any] = field(default_factory=dict)
    
    # Policy telemetry
    policy_telemetry: Dict[str, Any] = field(default_factory=dict)
    
    # Determinism verification
    trace_hashes: List[str] = field(default_factory=list)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "experiment_id": self.experiment_id,
            "slice_name": self.slice_name,
            "mode": self.mode,
            "master_seed": self.master_seed,
            "summary": {
                "total_cycles": self.total_cycles,
                "total_candidates_processed": self.total_candidates_processed,
                "total_candidates_generated": self.total_candidates_generated,
                "total_time_s": self.total_time_s,
                "avg_time_per_cycle_s": self.total_time_s / max(self.total_cycles, 1),
                "avg_candidates_per_cycle": self.total_candidates_processed / max(self.total_cycles, 1),
            },
            "cycle_stats": self.cycle_stats,
            "frontier_telemetry": self.frontier_telemetry,
            "policy_telemetry": self.policy_telemetry,
            "determinism": {
                "trace_hashes": self.trace_hashes,
                "hash_count": len(self.trace_hashes),
            },
            "metadata": self.metadata,
        }


def extract_telemetry_from_trace(trace_path: Path) -> TelemetryReport:
    """
    Extract telemetry from experiment trace.
    
    Args:
        trace_path: Path to trace JSONL file
        
    Returns:
        TelemetryReport
    """
    trace = load_experiment_trace(trace_path)
    
    report = TelemetryReport(
        experiment_id=trace.experiment_id,
        slice_name=trace.slice_name,
        mode=trace.mode,
        master_seed=trace.master_seed,
        total_cycles=len(trace.cycles),
    )
    
    # Process each cycle
    frontier_pushes = 0
    frontier_pops = 0
    policy_ranks = 0
    
    for cycle_trace in trace.cycles:
        # Count events
        derive_success = 0
        derive_failure = 0
        cycle_frontier_pushes = 0
        cycle_frontier_pops = 0
        
        for event in cycle_trace.events:
            if event.event_type == EventType.DERIVE_SUCCESS:
                derive_success += 1
            elif event.event_type == EventType.DERIVE_FAILURE:
                derive_failure += 1
            elif event.event_type == EventType.FRONTIER_PUSH:
                cycle_frontier_pushes += 1
                frontier_pushes += 1
            elif event.event_type == EventType.FRONTIER_POP:
                cycle_frontier_pops += 1
                frontier_pops += 1
            elif event.event_type == EventType.POLICY_RANK:
                policy_ranks += 1
        
        # Compute cycle stats
        cycle_stat = {
            "cycle": cycle_trace.cycle,
            "candidates_processed": cycle_frontier_pops,
            "candidates_generated": cycle_frontier_pushes,
            "derive_success": derive_success,
            "derive_failure": derive_failure,
            "success_rate": derive_success / max(derive_success + derive_failure, 1),
            "trace_hash": cycle_trace.hash(),
        }
        
        report.cycle_stats.append(cycle_stat)
        report.total_candidates_processed += cycle_frontier_pops
        report.total_candidates_generated += cycle_frontier_pushes
        report.trace_hashes.append(cycle_trace.hash())
    
    # Frontier telemetry
    report.frontier_telemetry = {
        "total_pushes": frontier_pushes,
        "total_pops": frontier_pops,
        "avg_pushes_per_cycle": frontier_pushes / max(len(trace.cycles), 1),
        "avg_pops_per_cycle": frontier_pops / max(len(trace.cycles), 1),
    }
    
    # Policy telemetry
    report.policy_telemetry = {
        "total_ranks": policy_ranks,
        "mode": trace.mode,
    }
    
    return report


def export_telemetry(
    report: TelemetryReport,
    output_path: Path,
    format: str = "json",
) -> None:
    """
    Export telemetry report to file.
    
    Args:
        report: Telemetry report
        output_path: Output file path
        format: Output format ("json" or "jsonl")
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "json":
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, indent=2, sort_keys=True)
    
    elif format == "jsonl":
        with open(output_path, 'w', encoding='utf-8') as f:
            # Write header
            header = {
                "type": "telemetry_header",
                "experiment_id": report.experiment_id,
                "slice_name": report.slice_name,
                "mode": report.mode,
                "master_seed": report.master_seed,
            }
            f.write(json.dumps(header, sort_keys=True) + '\n')
            
            # Write cycle stats
            for cycle_stat in report.cycle_stats:
                line = {
                    "type": "cycle_stat",
                    **cycle_stat
                }
                f.write(json.dumps(line, sort_keys=True) + '\n')
            
            # Write summary
            summary = {
                "type": "telemetry_summary",
                **report.to_dict()
            }
            f.write(json.dumps(summary, sort_keys=True) + '\n')
    
    else:
        raise ValueError(f"Unknown format: {format}")


def create_evidence_pack(
    trace_path: Path,
    output_dir: Path,
    include_trace: bool = True,
) -> Path:
    """
    Create RFL Evidence Pack from experiment trace.
    
    Evidence Pack contains:
    - Telemetry report (JSON)
    - Trace file (JSONL) [optional]
    - Metadata (JSON)
    
    Args:
        trace_path: Path to trace JSONL file
        output_dir: Output directory for evidence pack
        include_trace: Whether to include full trace
        
    Returns:
        Path to evidence pack directory
    """
    # Extract telemetry
    report = extract_telemetry_from_trace(trace_path)
    
    # Create evidence pack directory
    pack_dir = output_dir / f"evidence_pack_{report.experiment_id}"
    pack_dir.mkdir(parents=True, exist_ok=True)
    
    # Export telemetry
    telemetry_path = pack_dir / "telemetry.json"
    export_telemetry(report, telemetry_path, format="json")
    
    # Copy trace if requested
    if include_trace:
        import shutil
        trace_dest = pack_dir / "trace.jsonl"
        shutil.copy(trace_path, trace_dest)
    
    # Create metadata
    metadata = {
        "evidence_pack_version": "1.0",
        "experiment_id": report.experiment_id,
        "slice_name": report.slice_name,
        "mode": report.mode,
        "master_seed": report.master_seed,
        "files": {
            "telemetry": "telemetry.json",
            "trace": "trace.jsonl" if include_trace else None,
        },
        "determinism_verification": {
            "trace_hash_count": len(report.trace_hashes),
            "master_trace_hash": report.trace_hashes[0] if report.trace_hashes else None,
        }
    }
    
    metadata_path = pack_dir / "metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    
    return pack_dir


def compare_telemetry(
    report1: TelemetryReport,
    report2: TelemetryReport,
) -> Dict[str, Any]:
    """
    Compare two telemetry reports for determinism verification.
    
    Args:
        report1: First telemetry report
        report2: Second telemetry report
        
    Returns:
        Comparison result dictionary
    """
    result = {
        "deterministic": True,
        "differences": [],
    }
    
    # Check metadata
    if (report1.experiment_id != report2.experiment_id or
        report1.slice_name != report2.slice_name or
        report1.mode != report2.mode or
        report1.master_seed != report2.master_seed):
        result["deterministic"] = False
        result["differences"].append("Metadata mismatch")
    
    # Check cycle count
    if report1.total_cycles != report2.total_cycles:
        result["deterministic"] = False
        result["differences"].append(f"Cycle count: {report1.total_cycles} vs {report2.total_cycles}")
    
    # Check trace hashes
    if report1.trace_hashes != report2.trace_hashes:
        result["deterministic"] = False
        result["differences"].append("Trace hashes differ")
        
        # Find first differing cycle
        for i, (h1, h2) in enumerate(zip(report1.trace_hashes, report2.trace_hashes)):
            if h1 != h2:
                result["first_difference_cycle"] = i
                break
    
    return result
