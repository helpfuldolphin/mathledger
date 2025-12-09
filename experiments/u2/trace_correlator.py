# PHASE II — NOT RUN IN PHASE I
"""
U2 Trace Correlator

STATUS: PHASE II — NOT RUN IN PHASE I

Correlates trace events with manifest and budget health data for enriched analysis.

Features:
- Cross-reference trace cycles with manifest expectations
- Detect cycles with errors vs budget exhaustion
- Verify trace coverage matches manifest
- Generate correlation summary reports

INVARIANTS:
- Read-only: Never modifies input files
- Deterministic: Same inputs always produce same output
- No semantic changes: Pure analysis/reporting

Usage:
    from experiments.u2.trace_correlator import TraceCorrelator
    
    correlator = TraceCorrelator(
        trace_path=Path("trace.jsonl"),
        manifest_path=Path("manifest.json"),
        budget_health_path=Path("budget_health.json"),  # optional
    )
    summary = correlator.correlate()
    print(summary.format_human())

CLI:
    uv run python -m experiments.u2.trace_correlator \\
        --trace trace.jsonl \\
        --manifest manifest.json \\
        --budget-health budget_health.json
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .inspector import TraceLogInspector


@dataclass
class CorrelationSummary:
    """Summary of trace/manifest/budget correlation analysis."""
    
    # Trace info
    trace_path: str
    trace_cycles: Set[int] = field(default_factory=set)
    trace_total_records: int = 0
    trace_error_cycles: Set[int] = field(default_factory=set)
    
    # Manifest info
    manifest_path: Optional[str] = None
    manifest_expected_cycles: Optional[int] = None
    manifest_completed_cycles: Optional[int] = None
    manifest_mode: Optional[str] = None
    manifest_slice: Optional[str] = None
    
    # Budget health info
    budget_health_path: Optional[str] = None
    budget_exhausted_cycles: Set[int] = field(default_factory=set)
    budget_warnings: List[str] = field(default_factory=list)
    
    # Correlation results
    coverage_status: str = "UNKNOWN"  # "FULL", "PARTIAL", "MISSING"
    coverage_percentage: float = 0.0
    missing_cycles: List[int] = field(default_factory=list)
    extra_cycles: List[int] = field(default_factory=list)
    
    # Error/budget correlation
    errors_with_budget_exhaustion: Set[int] = field(default_factory=set)
    errors_without_budget_exhaustion: Set[int] = field(default_factory=set)
    budget_exhaustion_without_errors: Set[int] = field(default_factory=set)
    
    # Overall status
    status: str = "OK"  # "OK", "WARN", "ERROR"
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "trace_path": self.trace_path,
            "trace_cycles": sorted(self.trace_cycles),
            "trace_total_records": self.trace_total_records,
            "trace_error_cycles": sorted(self.trace_error_cycles),
            "manifest_path": self.manifest_path,
            "manifest_expected_cycles": self.manifest_expected_cycles,
            "manifest_completed_cycles": self.manifest_completed_cycles,
            "manifest_mode": self.manifest_mode,
            "manifest_slice": self.manifest_slice,
            "budget_health_path": self.budget_health_path,
            "budget_exhausted_cycles": sorted(self.budget_exhausted_cycles),
            "budget_warnings": self.budget_warnings,
            "coverage_status": self.coverage_status,
            "coverage_percentage": self.coverage_percentage,
            "missing_cycles": self.missing_cycles,
            "extra_cycles": self.extra_cycles,
            "errors_with_budget_exhaustion": sorted(self.errors_with_budget_exhaustion),
            "errors_without_budget_exhaustion": sorted(self.errors_without_budget_exhaustion),
            "budget_exhaustion_without_errors": sorted(self.budget_exhaustion_without_errors),
            "status": self.status,
            "warnings": self.warnings,
            "errors": self.errors,
        }
    
    def format_human(self) -> str:
        """Format summary for human-readable output."""
        status_emoji = {"OK": "✅", "WARN": "⚠️", "ERROR": "❌"}.get(self.status, "?")
        
        lines = [
            "=" * 70,
            "TRACE CORRELATION SUMMARY",
            "=" * 70,
            f"Status: {status_emoji} {self.status}",
            "",
            "TRACE INFO:",
            f"  Path: {self.trace_path}",
            f"  Total records: {self.trace_total_records:,}",
            f"  Cycles found: {len(self.trace_cycles)}",
            f"  Error cycles: {len(self.trace_error_cycles)}",
        ]
        
        if self.manifest_path:
            lines.extend([
                "",
                "MANIFEST INFO:",
                f"  Path: {self.manifest_path}",
                f"  Expected cycles: {self.manifest_expected_cycles}",
                f"  Completed cycles: {self.manifest_completed_cycles}",
                f"  Mode: {self.manifest_mode}",
                f"  Slice: {self.manifest_slice}",
            ])
        
        if self.budget_health_path:
            lines.extend([
                "",
                "BUDGET HEALTH:",
                f"  Path: {self.budget_health_path}",
                f"  Exhausted cycles: {len(self.budget_exhausted_cycles)}",
            ])
            if self.budget_warnings:
                for w in self.budget_warnings[:5]:
                    lines.append(f"    ⚠️ {w}")
        
        lines.extend([
            "",
            "COVERAGE:",
            f"  Status: {self.coverage_status}",
            f"  Percentage: {self.coverage_percentage:.1f}%",
        ])
        
        if self.missing_cycles:
            missing_str = ", ".join(str(c) for c in self.missing_cycles[:10])
            if len(self.missing_cycles) > 10:
                missing_str += f", ... ({len(self.missing_cycles) - 10} more)"
            lines.append(f"  Missing: [{missing_str}]")
        
        if self.extra_cycles:
            extra_str = ", ".join(str(c) for c in self.extra_cycles[:10])
            if len(self.extra_cycles) > 10:
                extra_str += f", ... ({len(self.extra_cycles) - 10} more)"
            lines.append(f"  Extra: [{extra_str}]")
        
        if self.trace_error_cycles or self.budget_exhausted_cycles:
            lines.extend([
                "",
                "ERROR/BUDGET CORRELATION:",
                f"  Errors with budget exhaustion: {len(self.errors_with_budget_exhaustion)}",
                f"  Errors without budget exhaustion: {len(self.errors_without_budget_exhaustion)}",
                f"  Budget exhaustion without errors: {len(self.budget_exhaustion_without_errors)}",
            ])
        
        if self.warnings:
            lines.extend(["", "WARNINGS:"])
            for w in self.warnings:
                lines.append(f"  ⚠️ {w}")
        
        if self.errors:
            lines.extend(["", "ERRORS:"])
            for e in self.errors:
                lines.append(f"  ❌ {e}")
        
        lines.append("=" * 70)
        return "\n".join(lines)


class TraceCorrelator:
    """
    Correlates trace events with manifest and budget health data.
    
    INVARIANTS:
    - Read-only: Never modifies input files
    - Deterministic: Same inputs always produce same output
    """
    
    def __init__(
        self,
        trace_path: Path,
        manifest_path: Optional[Path] = None,
        budget_health_path: Optional[Path] = None,
    ):
        """
        Initialize correlator.
        
        Args:
            trace_path: Path to trace JSONL file.
            manifest_path: Optional path to manifest JSON file.
            budget_health_path: Optional path to budget health JSON file.
        """
        self._trace_path = Path(trace_path)
        self._manifest_path = Path(manifest_path) if manifest_path else None
        self._budget_health_path = Path(budget_health_path) if budget_health_path else None
        
        # Validate paths exist
        if not self._trace_path.exists():
            raise FileNotFoundError(f"Trace file not found: {self._trace_path}")
        if self._manifest_path and not self._manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {self._manifest_path}")
        if self._budget_health_path and not self._budget_health_path.exists():
            raise FileNotFoundError(f"Budget health file not found: {self._budget_health_path}")
    
    def _load_manifest(self) -> Optional[Dict[str, Any]]:
        """Load and parse manifest file."""
        if not self._manifest_path:
            return None
        
        with open(self._manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def _load_budget_health(self) -> Optional[Dict[str, Any]]:
        """Load and parse budget health file."""
        if not self._budget_health_path:
            return None
        
        with open(self._budget_health_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def _extract_trace_info(self, summary: CorrelationSummary) -> None:
        """Extract information from trace file."""
        inspector = TraceLogInspector(self._trace_path)
        trace_summary = inspector.summarize()
        
        summary.trace_total_records = trace_summary.total_records
        
        # Collect cycles and detect errors
        error_indicators = {"error", "fail", "exception", "timeout", "abort"}
        
        for record in inspector.filter_events():
            payload = record.get("payload", {})
            cycle = payload.get("cycle")
            
            if cycle is not None:
                summary.trace_cycles.add(cycle)
                
                # Check for errors in various places
                record_str = json.dumps(record).lower()
                if any(ind in record_str for ind in error_indicators):
                    summary.trace_error_cycles.add(cycle)
                
                # Check success field
                raw_record = payload.get("raw_record", {})
                if isinstance(raw_record, dict) and raw_record.get("success") is False:
                    summary.trace_error_cycles.add(cycle)
    
    def _extract_manifest_info(self, summary: CorrelationSummary) -> None:
        """Extract information from manifest file."""
        manifest = self._load_manifest()
        if not manifest:
            return
        
        summary.manifest_path = str(self._manifest_path)
        summary.manifest_expected_cycles = manifest.get("cycles")
        summary.manifest_completed_cycles = manifest.get("ht_series_length")
        summary.manifest_mode = manifest.get("mode")
        summary.manifest_slice = manifest.get("slice")
    
    def _extract_budget_health_info(self, summary: CorrelationSummary) -> None:
        """Extract information from budget health file."""
        budget = self._load_budget_health()
        if not budget:
            return
        
        summary.budget_health_path = str(self._budget_health_path)
        
        # Extract exhausted cycles
        exhausted = budget.get("exhausted_cycles", [])
        if isinstance(exhausted, list):
            summary.budget_exhausted_cycles = set(exhausted)
        
        # Extract budget warnings
        warnings = budget.get("warnings", [])
        if isinstance(warnings, list):
            summary.budget_warnings = warnings[:10]  # Limit to 10
        
        # Also check per-cycle budget status
        per_cycle = budget.get("per_cycle", {})
        if isinstance(per_cycle, dict):
            for cycle_str, status in per_cycle.items():
                try:
                    cycle = int(cycle_str)
                    if isinstance(status, dict) and status.get("exhausted"):
                        summary.budget_exhausted_cycles.add(cycle)
                except (ValueError, TypeError):
                    pass
    
    def _compute_coverage(self, summary: CorrelationSummary) -> None:
        """Compute coverage metrics."""
        if summary.manifest_expected_cycles is None:
            # No manifest, can't compute coverage
            summary.coverage_status = "UNKNOWN"
            return
        
        expected_set = set(range(summary.manifest_expected_cycles))
        
        # Compute coverage
        covered = summary.trace_cycles & expected_set
        missing = expected_set - summary.trace_cycles
        extra = summary.trace_cycles - expected_set
        
        summary.missing_cycles = sorted(missing)
        summary.extra_cycles = sorted(extra)
        
        if len(expected_set) > 0:
            summary.coverage_percentage = (len(covered) / len(expected_set)) * 100
        else:
            summary.coverage_percentage = 100.0
        
        if summary.coverage_percentage >= 100.0 and not missing:
            summary.coverage_status = "FULL"
        elif summary.coverage_percentage > 0:
            summary.coverage_status = "PARTIAL"
        else:
            summary.coverage_status = "MISSING"
    
    def _correlate_errors_and_budget(self, summary: CorrelationSummary) -> None:
        """Correlate error cycles with budget exhaustion."""
        error_cycles = summary.trace_error_cycles
        budget_cycles = summary.budget_exhausted_cycles
        
        summary.errors_with_budget_exhaustion = error_cycles & budget_cycles
        summary.errors_without_budget_exhaustion = error_cycles - budget_cycles
        summary.budget_exhaustion_without_errors = budget_cycles - error_cycles
    
    def _compute_status(self, summary: CorrelationSummary) -> None:
        """Compute overall status and collect warnings/errors."""
        # Check for errors
        if summary.missing_cycles:
            summary.errors.append(
                f"{len(summary.missing_cycles)} cycles missing from trace"
            )
        
        if summary.trace_error_cycles:
            summary.warnings.append(
                f"{len(summary.trace_error_cycles)} cycles have errors"
            )
        
        if summary.budget_exhausted_cycles:
            summary.warnings.append(
                f"{len(summary.budget_exhausted_cycles)} cycles exhausted budget"
            )
        
        if summary.coverage_status == "MISSING":
            summary.errors.append("No trace coverage of expected cycles")
        elif summary.coverage_status == "PARTIAL":
            summary.warnings.append(
                f"Partial coverage: {summary.coverage_percentage:.1f}%"
            )
        
        # Determine overall status
        if summary.errors:
            summary.status = "ERROR"
        elif summary.warnings:
            summary.status = "WARN"
        else:
            summary.status = "OK"
    
    def correlate(self) -> CorrelationSummary:
        """
        Perform correlation analysis.
        
        Returns:
            CorrelationSummary with all correlation results.
        """
        summary = CorrelationSummary(trace_path=str(self._trace_path))
        
        # Extract info from each source
        self._extract_trace_info(summary)
        self._extract_manifest_info(summary)
        self._extract_budget_health_info(summary)
        
        # Compute correlations
        self._compute_coverage(summary)
        self._correlate_errors_and_budget(summary)
        self._compute_status(summary)
        
        return summary


def main():
    """CLI entry point for trace correlator."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="PHASE II — U2 Trace Correlator",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  # Basic correlation with manifest
  python -m experiments.u2.trace_correlator \\
    --trace trace.jsonl \\
    --manifest manifest.json
  
  # Full correlation with budget health
  python -m experiments.u2.trace_correlator \\
    --trace trace.jsonl \\
    --manifest manifest.json \\
    --budget-health budget_health.json
  
  # JSON output
  python -m experiments.u2.trace_correlator \\
    --trace trace.jsonl \\
    --manifest manifest.json \\
    --json
""",
    )
    
    parser.add_argument(
        "--trace",
        type=str,
        required=True,
        help="Path to trace log file (JSONL).",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        help="Path to manifest JSON file.",
    )
    parser.add_argument(
        "--budget-health",
        type=str,
        help="Path to budget health JSON file.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format.",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Write output to file instead of stdout.",
    )
    
    args = parser.parse_args()
    
    # Validate paths
    trace_path = Path(args.trace)
    manifest_path = Path(args.manifest) if args.manifest else None
    budget_path = Path(args.budget_health) if args.budget_health else None
    
    try:
        correlator = TraceCorrelator(
            trace_path=trace_path,
            manifest_path=manifest_path,
            budget_health_path=budget_path,
        )
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    
    summary = correlator.correlate()
    
    # Format output
    if args.json:
        output = json.dumps(summary.to_dict(), indent=2)
    else:
        output = summary.format_human()
    
    # Write output
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output + "\n")
        print(f"Output written to: {args.output}", file=sys.stderr)
    else:
        print(output)
    
    # Exit code based on status
    exit_code = {"OK": 0, "WARN": 0, "ERROR": 1}.get(summary.status, 1)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

