#!/usr/bin/env python3
# PHASE II — NOT USED IN PHASE I
"""
Budget Trend Analyzer
═══════════════════════════════════════════════════════════════════════════════

Analyze budget health trends across multiple experiment runs.
Read-only tool for understanding how budget health evolves over time.

Usage:
    # Analyze trends from multiple health JSON files
    uv run python experiments/budget_trends.py --inputs run1.json run2.json run3.json
    
    # Output as JSON
    uv run python experiments/budget_trends.py --inputs *.json --json
    
    # Markdown report
    uv run python experiments/budget_trends.py --inputs *.json --markdown

Trend Classification:
    IMPROVING:  Health status improved over the analysis window
                (e.g., STARVED → TIGHT → SAFE)
    STABLE:     Health status remained constant
    DEGRADING:  Health status worsened over the analysis window
                (e.g., SAFE → TIGHT → STARVED)

═══════════════════════════════════════════════════════════════════════════════
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# =============================================================================
# TREND CLASSIFICATION
# =============================================================================


class TrendDirection(Enum):
    """
    Trend classification for budget health over multiple runs.
    
    Neutral naming convention (no value judgments):
        IMPROVING  - Health status getting better (toward SAFE)
        STABLE     - Health status unchanged
        DEGRADING  - Health status getting worse (toward STARVED)
        UNKNOWN    - Insufficient data to determine trend
    """
    IMPROVING = "IMPROVING"
    STABLE = "STABLE"
    DEGRADING = "DEGRADING"
    UNKNOWN = "UNKNOWN"


# Health status ordering for trend calculation
# Lower index = healthier
HEALTH_ORDER = ["SAFE", "TIGHT", "STARVED", "INVALID"]


def health_to_score(status: str) -> int:
    """
    Convert health status to numeric score for trend calculation.
    
    Lower score = healthier:
        SAFE    = 0
        TIGHT   = 1
        STARVED = 2
        INVALID = 3 (worst, no data)
    """
    try:
        return HEALTH_ORDER.index(status)
    except ValueError:
        return 3  # Unknown = INVALID


def classify_trend(statuses: List[str]) -> TrendDirection:
    """
    Classify trend direction from a sequence of health statuses.
    
    Uses simple linear regression on health scores:
        - Negative slope = IMPROVING (scores decreasing)
        - Zero slope = STABLE
        - Positive slope = DEGRADING (scores increasing)
    
    Args:
        statuses: List of health status strings in chronological order
        
    Returns:
        TrendDirection classification
    """
    if len(statuses) < 2:
        return TrendDirection.UNKNOWN
    
    scores = [health_to_score(s) for s in statuses]
    
    # Simple trend: compare first half average to second half average
    mid = len(scores) // 2
    if mid == 0:
        mid = 1
    
    first_half_avg = sum(scores[:mid]) / mid
    second_half_avg = sum(scores[mid:]) / (len(scores) - mid)
    
    # Threshold for change detection (0.5 = one status level)
    threshold = 0.3
    
    diff = second_half_avg - first_half_avg
    
    if diff < -threshold:
        return TrendDirection.IMPROVING
    elif diff > threshold:
        return TrendDirection.DEGRADING
    else:
        return TrendDirection.STABLE


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class RunHealth:
    """Health data from a single run."""
    run_id: str
    path: str
    slice_name: str
    status: str
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class SliceTrend:
    """Trend analysis for a single slice across multiple runs."""
    slice_name: str
    runs: List[RunHealth]
    trend: TrendDirection
    status_sequence: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "slice_name": self.slice_name,
            "trend": self.trend.value,
            "status_sequence": self.status_sequence,
            "num_runs": len(self.runs),
            "first_status": self.status_sequence[0] if self.status_sequence else None,
            "last_status": self.status_sequence[-1] if self.status_sequence else None,
            "runs": [
                {
                    "run_id": r.run_id,
                    "status": r.status,
                    "metrics": r.metrics,
                }
                for r in self.runs
            ],
        }


@dataclass
class TrendReport:
    """Complete trend analysis report."""
    inputs: List[str]
    slices: List[SliceTrend]
    summary: Dict[str, int]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "phase": "PHASE II — NOT USED IN PHASE I",
            "inputs": self.inputs,
            "slices": [s.to_dict() for s in self.slices],
            "summary": self.summary,
        }


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================


def load_health_json(path: Path) -> List[RunHealth]:
    """
    Load health data from a JSON file.
    
    Supports two formats:
        1. Health JSON from --health-json (has "health_report" key)
        2. Basic JSON from --json (has "logs" key)
    
    Args:
        path: Path to JSON file
        
    Returns:
        List of RunHealth objects
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    results = []
    run_id = path.stem  # Use filename as run ID
    
    # Format 1: Health JSON
    if "health_report" in data:
        for entry in data["health_report"]:
            results.append(RunHealth(
                run_id=run_id,
                path=str(path),
                slice_name=entry.get("slice", "unknown"),
                status=entry.get("health", {}).get("status", "INVALID"),
                metrics=entry.get("health", {}).get("metrics", {}),
            ))
    
    # Format 2: Basic JSON with logs
    elif "logs" in data:
        # Need to classify health ourselves
        from experiments.summarize_budget_usage import BudgetSummary, classify_budget_health
        
        for log_entry in data["logs"]:
            summary = BudgetSummary(
                path=log_entry.get("path", ""),
                slice_name=log_entry.get("slice", "unknown"),
                mode=log_entry.get("mode", ""),
                total_cycles=log_entry.get("total_cycles", 0),
                budget_exhausted_count=int(log_entry.get("budget", {}).get("exhausted_count", 0)),
                max_candidates_hit_count=int(log_entry.get("budget", {}).get("max_candidates_hit_count", 0)),
                timeout_abstentions_total=int(log_entry.get("budget", {}).get("timeout_abstentions_total", 0)),
                cycles_with_budget_field=log_entry.get("cycles_with_budget_field", 0),
            )
            health = classify_budget_health(summary)
            results.append(RunHealth(
                run_id=run_id,
                path=str(path),
                slice_name=summary.slice_name,
                status=health.status.value,
                metrics=health.metrics,
            ))
    
    return results


def analyze_trends(inputs: List[Path]) -> TrendReport:
    """
    Analyze budget health trends across multiple input files.
    
    Args:
        inputs: List of paths to health JSON files (in chronological order)
        
    Returns:
        TrendReport with per-slice trends and summary
    """
    # Load all health data
    all_runs: List[RunHealth] = []
    for path in inputs:
        try:
            runs = load_health_json(path)
            all_runs.extend(runs)
        except Exception as e:
            print(f"WARNING: Failed to load {path}: {e}", file=sys.stderr)
    
    # Group by slice (preserving order within each slice)
    by_slice: Dict[str, List[RunHealth]] = {}
    for run in all_runs:
        if run.slice_name not in by_slice:
            by_slice[run.slice_name] = []
        by_slice[run.slice_name].append(run)
    
    # Analyze trends for each slice
    slice_trends: List[SliceTrend] = []
    for slice_name in sorted(by_slice.keys()):
        runs = by_slice[slice_name]
        status_sequence = [r.status for r in runs]
        trend = classify_trend(status_sequence)
        
        slice_trends.append(SliceTrend(
            slice_name=slice_name,
            runs=runs,
            trend=trend,
            status_sequence=status_sequence,
        ))
    
    # Summary counts
    summary = {
        "total_slices": len(slice_trends),
        "improving": sum(1 for s in slice_trends if s.trend == TrendDirection.IMPROVING),
        "stable": sum(1 for s in slice_trends if s.trend == TrendDirection.STABLE),
        "degrading": sum(1 for s in slice_trends if s.trend == TrendDirection.DEGRADING),
        "unknown": sum(1 for s in slice_trends if s.trend == TrendDirection.UNKNOWN),
    }
    
    return TrendReport(
        inputs=[str(p) for p in inputs],
        slices=slice_trends,
        summary=summary,
    )


# =============================================================================
# OUTPUT FORMATTERS
# =============================================================================


TREND_EMOJI = {
    TrendDirection.IMPROVING: "📈",
    TrendDirection.STABLE: "➡️",
    TrendDirection.DEGRADING: "📉",
    TrendDirection.UNKNOWN: "❓",
}

STATUS_EMOJI = {
    "SAFE": "✅",
    "TIGHT": "⚠️",
    "STARVED": "🔥",
    "INVALID": "❓",
}


def format_markdown(report: TrendReport) -> str:
    """Format trend report as Markdown."""
    lines = []
    
    lines.append("## 📊 Budget Health Trend Analysis")
    lines.append("")
    lines.append("> PHASE II — NOT USED IN PHASE I")
    lines.append(f"> Analyzed {len(report.inputs)} runs")
    lines.append("")
    
    # Summary
    lines.append("### Summary")
    lines.append("")
    lines.append(f"- **Improving**: {report.summary['improving']} slices 📈")
    lines.append(f"- **Stable**: {report.summary['stable']} slices ➡️")
    lines.append(f"- **Degrading**: {report.summary['degrading']} slices 📉")
    if report.summary['unknown'] > 0:
        lines.append(f"- **Unknown**: {report.summary['unknown']} slices ❓")
    lines.append("")
    
    # Per-slice table
    lines.append("### Per-Slice Trends")
    lines.append("")
    lines.append("| Slice | Trend | First | Last | Sequence |")
    lines.append("|-------|-------|-------|------|----------|")
    
    for st in report.slices:
        trend_emoji = TREND_EMOJI.get(st.trend, "")
        first_emoji = STATUS_EMOJI.get(st.status_sequence[0], "") if st.status_sequence else ""
        last_emoji = STATUS_EMOJI.get(st.status_sequence[-1], "") if st.status_sequence else ""
        
        # Compact sequence with arrows
        seq_str = " → ".join(st.status_sequence[:5])
        if len(st.status_sequence) > 5:
            seq_str += f" ... ({len(st.status_sequence)} total)"
        
        lines.append(
            f"| `{st.slice_name}` | {trend_emoji} {st.trend.value} | "
            f"{first_emoji} {st.status_sequence[0] if st.status_sequence else '?'} | "
            f"{last_emoji} {st.status_sequence[-1] if st.status_sequence else '?'} | "
            f"{seq_str} |"
        )
    
    lines.append("")
    
    # Advisory
    if report.summary['degrading'] > 0:
        lines.append("### ⚠️ Advisory")
        lines.append("")
        lines.append(f"{report.summary['degrading']} slice(s) show degrading budget health.")
        lines.append("Consider reviewing `cycle_budget_s` and `taut_timeout_s` parameters.")
        lines.append("")
    
    return "\n".join(lines)


def format_json(report: TrendReport) -> str:
    """Format trend report as JSON."""
    return json.dumps(report.to_dict(), indent=2)


# =============================================================================
# CLI
# =============================================================================


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze budget health trends across multiple experiment runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        type=str,
        help="Paths to health JSON files (in chronological order)."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON."
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Output as Markdown."
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Write output to file instead of stdout."
    )
    
    args = parser.parse_args()
    
    print("PHASE II — NOT USED IN PHASE I", file=sys.stderr)
    print("", file=sys.stderr)
    
    # Validate inputs
    input_paths = []
    for path_str in args.inputs:
        path = Path(path_str)
        if not path.exists():
            print(f"WARNING: Input not found, skipping: {path}", file=sys.stderr)
            continue
        input_paths.append(path)
    
    if len(input_paths) < 2:
        print("ERROR: Need at least 2 input files for trend analysis.", file=sys.stderr)
        sys.exit(1)
    
    # Analyze
    report = analyze_trends(input_paths)
    
    # Format output
    if args.json:
        output = format_json(report)
    else:
        # Default to Markdown
        output = format_markdown(report)
    
    # Write output
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"Output written to: {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()

