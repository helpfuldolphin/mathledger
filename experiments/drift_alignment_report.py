#!/usr/bin/env python
"""
Drift Alignment Report CLI.

===============================================================================
STATUS: PHASE II — METRICS BUREAU (D5) — CI & HUMAN-READABLE REPORTING
===============================================================================

This CLI exposes the drift alignment engine as:
  - A CI-compatible analyzer (JSON out, clear exit codes)
  - A short, dev-friendly report generator (Markdown/terminal text)
  - A hookable utility for other Phase II tooling
  - A multi-slice drift dashboard

Usage (Single Slice):
    uv run python experiments/drift_alignment_report.py \\
        --slice slice_uplift_goal \\
        --baseline-log path/to/baseline.jsonl \\
        --rfl-log path/to/rfl.jsonl \\
        --out artifacts/phase_ii/drift/slice_uplift_goal.json

Usage (All Slices):
    uv run python experiments/drift_alignment_report.py \\
        --all-slices \\
        --slices-dir path/to/slices/ \\
        --out artifacts/phase_ii/drift/drift_report_all_slices.json

CI Mode:
    uv run python experiments/drift_alignment_report.py \\
        --slice slice_uplift_goal \\
        --baseline-log path/to/baseline.jsonl \\
        --rfl-log path/to/rfl.jsonl \\
        --ci --profile strict

Exit Codes (CI mode):
    0 - OK or WARN status
    1 - BLOCK status (any slice in --all-slices mode)
    2 - Error (missing files, parse errors, etc.)

ABSOLUTE SAFEGUARDS:
  - No governance/promotion logic.
  - No significance testing.
  - No "good/bad/better/worse" language.
  - All output is purely descriptive.
===============================================================================
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from backend.metrics.drift_alignment import (
    DriftAlignmentResult,
    DriftCell,
    StabilityEnvelope,
    compute_drift_alignment,
    generate_multi_metric_envelopes,
    evaluate_drift_for_ci,
    classify_pattern_hint,
    get_drift_thresholds,
    build_drift_cells,
    drift_cells_to_dicts,
    PATTERN_HINTS,
    DRIFT_THRESHOLDS,
    DRIFT_CELL_COLUMNS,
)


# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------

REPORT_VERSION = "1.0"


# ---------------------------------------------------------------------------
# JSONL Log Parsing
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """
    Load a JSONL file and return list of parsed records.

    Args:
        path: Path to JSONL file

    Returns:
        List of parsed JSON objects

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file contains invalid JSON
    """
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise json.JSONDecodeError(
                    f"Line {line_num}: {e.msg}",
                    e.doc,
                    e.pos,
                )
    return records


def extract_metric_trajectory(
    records: List[Dict[str, Any]],
    metric_key: str,
    default: float = 0.0,
) -> List[float]:
    """
    Extract a metric trajectory from JSONL records.

    Args:
        records: List of JSONL records
        metric_key: Key to extract (supports dot notation like "derivation.depth")
        default: Default value if key is missing

    Returns:
        List of metric values in record order
    """
    values = []
    for record in records:
        value = _get_nested(record, metric_key, default)
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            values.append(default)
    return values


def _get_nested(d: Dict[str, Any], key: str, default: Any = None) -> Any:
    """Get nested dictionary value using dot notation."""
    parts = key.split(".")
    current = d
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return default
    return current


def derive_trajectories_from_logs(
    baseline_records: List[Dict[str, Any]],
    rfl_records: List[Dict[str, Any]],
) -> Dict[str, List[float]]:
    """
    Derive metric trajectories from baseline and RFL logs.

    Extracts and combines trajectories for:
        - success: Success rate (from status field)
        - abstention: Abstention rate
        - depth: Chain/derivation depth
        - entropy: Candidate entropy (if available)

    Args:
        baseline_records: Records from baseline log
        rfl_records: Records from RFL log

    Returns:
        Dict mapping metric names to combined trajectories
    """
    trajectories: Dict[str, List[float]] = {}

    # Success rate trajectory
    baseline_success = _extract_success_series(baseline_records)
    rfl_success = _extract_success_series(rfl_records)
    trajectories["success"] = baseline_success + rfl_success

    # Abstention rate trajectory
    baseline_abstention = _extract_abstention_series(baseline_records)
    rfl_abstention = _extract_abstention_series(rfl_records)
    trajectories["abstention"] = baseline_abstention + rfl_abstention

    # Depth trajectory
    baseline_depth = extract_metric_trajectory(baseline_records, "depth", 0.0)
    if all(d == 0.0 for d in baseline_depth):
        baseline_depth = extract_metric_trajectory(baseline_records, "derivation.depth", 0.0)
    rfl_depth = extract_metric_trajectory(rfl_records, "depth", 0.0)
    if all(d == 0.0 for d in rfl_depth):
        rfl_depth = extract_metric_trajectory(rfl_records, "derivation.depth", 0.0)
    if baseline_depth or rfl_depth:
        trajectories["depth"] = baseline_depth + rfl_depth

    # Entropy trajectory (optional)
    baseline_entropy = extract_metric_trajectory(baseline_records, "entropy", -1.0)
    rfl_entropy = extract_metric_trajectory(rfl_records, "entropy", -1.0)
    combined_entropy = baseline_entropy + rfl_entropy
    if any(e >= 0 for e in combined_entropy):
        trajectories["entropy"] = [max(0.0, e) for e in combined_entropy]

    return trajectories


def _extract_success_series(records: List[Dict[str, Any]]) -> List[float]:
    """Extract success rate series from records."""
    values = []
    for record in records:
        if "success" in record:
            val = record["success"]
            if isinstance(val, bool):
                values.append(1.0 if val else 0.0)
            else:
                values.append(float(val) if val else 0.0)
        elif "status" in record:
            status = str(record["status"]).lower()
            if status in ("success", "ok", "pass", "passed"):
                values.append(1.0)
            elif status in ("failure", "fail", "failed", "error"):
                values.append(0.0)
            else:
                values.append(0.5)
        elif "proofs_succeeded" in record and "proofs_attempted" in record:
            attempted = record["proofs_attempted"]
            succeeded = record["proofs_succeeded"]
            if attempted > 0:
                values.append(succeeded / attempted)
            else:
                values.append(0.0)
        else:
            values.append(0.0)
    return values


def _extract_abstention_series(records: List[Dict[str, Any]]) -> List[float]:
    """Extract abstention rate series from records."""
    values = []
    for record in records:
        if "abstention_rate" in record:
            values.append(float(record["abstention_rate"]))
        elif "abstention_count" in record:
            total = record.get("proofs_attempted", record.get("total", 1))
            if total > 0:
                values.append(record["abstention_count"] / total)
            else:
                values.append(0.0)
        elif "abstention" in record:
            val = record["abstention"]
            if isinstance(val, bool):
                values.append(1.0 if val else 0.0)
            else:
                values.append(float(val) if val else 0.0)
        elif "derivation" in record and isinstance(record["derivation"], dict):
            deriv = record["derivation"]
            if "abstained" in deriv:
                total = deriv.get("attempted", deriv.get("total", 1))
                if total > 0:
                    values.append(deriv["abstained"] / total)
                else:
                    values.append(0.0)
            else:
                values.append(0.0)
        else:
            values.append(0.0)
    return values


# ---------------------------------------------------------------------------
# Pattern Hints for CI
# ---------------------------------------------------------------------------

def compute_pattern_hints(trajectories: Dict[str, List[float]]) -> Dict[str, str]:
    """
    Compute pattern hints for all trajectories.

    Pattern hints are purely descriptive notes for visualization.

    Args:
        trajectories: Dict of metric trajectories

    Returns:
        Dict mapping metric names to pattern hints
    """
    hints: Dict[str, str] = {}
    for metric_name, series in trajectories.items():
        hints[metric_name] = classify_pattern_hint(series)
    return hints


# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------

def generate_drift_report(
    slice_id: str,
    trajectories: Dict[str, List[float]],
    window: int = 20,
    threshold: float = 0.1,
    confidence: float = 0.95,
) -> Dict[str, Any]:
    """
    Generate complete drift alignment report.

    Args:
        slice_id: Slice identifier
        trajectories: Dict of metric trajectories
        window: Rolling window size
        threshold: Drift threshold
        confidence: Confidence level for envelopes

    Returns:
        Complete report dict ready for JSON serialization
    """
    # Compute drift alignment
    alignment_result = compute_drift_alignment(
        slice_id,
        trajectories,
        window_size=window,
        drift_threshold=threshold,
    )

    # Generate stability envelopes
    envelopes = generate_multi_metric_envelopes(
        trajectories,
        window=window,
        confidence=confidence,
    )

    # Compute pattern hints for each metric
    pattern_hints = compute_pattern_hints(trajectories)

    # Build report structure
    report: Dict[str, Any] = {
        "slice": slice_id,
        "drift_alignment_score": alignment_result.drift_alignment_score,
        "coherence_score": alignment_result.coherence_score,
        "trajectory_correlation": dict(sorted(alignment_result.trajectory_correlation.items())),
        "metrics": {},
        "pattern_hints": dict(sorted(pattern_hints.items())),
        "stability_envelopes": {},
        "metadata": {
            "window_size": window,
            "drift_threshold": threshold,
            "confidence_level": confidence,
            "metric_count": len(trajectories),
            "total_points": sum(len(t) for t in trajectories.values()),
        },
    }

    # Add per-metric details with pattern hints
    for metric_name in sorted(trajectories.keys()):
        sig = alignment_result.metrics.get(metric_name)
        if sig:
            report["metrics"][metric_name] = {
                "drift_score": sig.drift_score,
                "direction": sig.direction,
                "stability": sig.stability,
                "trend": sig.trend,
                "monotonicity": sig.monotonicity,
                "pattern_hint": pattern_hints.get(metric_name, "flat"),
            }

    # Add envelope summaries
    for metric_name in sorted(envelopes.keys()):
        env = envelopes[metric_name]
        report["stability_envelopes"][metric_name] = {
            "envelope_width": env.envelope_width,
            "containment_ratio": env.containment_ratio,
            "center_line": env.center_line,
            "upper_band": env.upper_band,
            "lower_band": env.lower_band,
        }

    return report


def generate_multi_slice_report(
    slice_reports: Dict[str, Dict[str, Any]],
    ci_results: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Generate combined multi-slice drift report.

    Args:
        slice_reports: Dict mapping slice IDs to their reports
        ci_results: Optional dict mapping slice IDs to CI results

    Returns:
        Combined report for all slices
    """
    report: Dict[str, Any] = {
        "version": REPORT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "slice_count": len(slice_reports),
        "slices": dict(sorted(slice_reports.items())),
    }

    if ci_results:
        report["ci_summary"] = {
            "slice_statuses": {
                slice_id: result.get("status", "UNKNOWN")
                for slice_id, result in sorted(ci_results.items())
            },
            "any_block": any(
                r.get("status") == "BLOCK" for r in ci_results.values()
            ),
            "all_ok": all(
                r.get("status") == "OK" for r in ci_results.values()
            ),
        }

    return report


def generate_markdown_report(
    report: Dict[str, Any],
    ci_result: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Generate human-readable Markdown summary.

    Args:
        report: Report dict from generate_drift_report
        ci_result: Optional CI gate result

    Returns:
        Markdown string
    """
    lines = []
    lines.append(f"# Drift Alignment Report: {report['slice']}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Drift Alignment Score**: {report['drift_alignment_score']:.3f}")
    lines.append(f"- **Coherence Score**: {report['coherence_score']:.3f}")
    lines.append("")

    if ci_result:
        lines.append(f"- **CI Status**: {ci_result['status']}")
        lines.append(f"- **Max Metric Drift Score**: {ci_result['max_metric_drift_score']:.3f}")
        if ci_result.get("offending_metrics"):
            lines.append(f"- **Metrics Exceeding Threshold**: {', '.join(ci_result['offending_metrics'])}")
        # Add pattern hints to CI summary
        if ci_result.get("pattern_hints"):
            lines.append("")
            lines.append("### Pattern Hints")
            for metric, hint in sorted(ci_result["pattern_hints"].items()):
                lines.append(f"- **{metric}**: {hint}")
        lines.append("")

    lines.append("## Per-Metric Analysis")
    lines.append("")
    lines.append("| Metric | Direction | Drift Score | Stability | Pattern |")
    lines.append("|--------|-----------|-------------|-----------|---------|")

    for metric_name, metric_data in sorted(report.get("metrics", {}).items()):
        direction = metric_data.get("direction", "stable")
        drift_score = metric_data.get("drift_score", 0.0)
        stability = metric_data.get("stability", 1.0)
        pattern = metric_data.get("pattern_hint", "flat")
        lines.append(f"| {metric_name} | {direction} | {drift_score:.3f} | {stability:.3f} | {pattern} |")

    lines.append("")
    lines.append("## Correlation Matrix")
    lines.append("")

    correlations = report.get("trajectory_correlation", {})
    if correlations:
        for pair, corr in sorted(correlations.items()):
            lines.append(f"- `{pair}`: {corr:.3f}")
    else:
        lines.append("_No pairwise correlations available._")

    lines.append("")
    lines.append("## Legend")
    lines.append("")
    lines.append("- **Drift Score**: Magnitude of detected change (0 = no change)")
    lines.append("- **Stability**: Consistency index (1 = perfectly stable)")
    lines.append("- **Pattern**: Shape hint (`flat`, `rising`, `falling`, `oscillatory`)")
    lines.append("- **Coherence**: How aligned metrics move together (1 = fully aligned)")
    lines.append("")
    lines.append("---")
    lines.append(f"_Generated with window={report['metadata']['window_size']}, threshold={report['metadata']['drift_threshold']}_")

    return "\n".join(lines)


def generate_multi_slice_markdown(
    multi_report: Dict[str, Any],
) -> str:
    """
    Generate Markdown summary for multi-slice report.

    Args:
        multi_report: Multi-slice report from generate_multi_slice_report

    Returns:
        Markdown string
    """
    lines = []
    lines.append("# Multi-Slice Drift Alignment Dashboard")
    lines.append("")
    lines.append(f"**Generated**: {multi_report.get('generated_at', 'N/A')}")
    lines.append(f"**Slice Count**: {multi_report.get('slice_count', 0)}")
    lines.append("")

    # CI summary if present
    ci_summary = multi_report.get("ci_summary")
    if ci_summary:
        lines.append("## CI Summary")
        lines.append("")
        lines.append("| Slice | Status |")
        lines.append("|-------|--------|")
        for slice_id, status in sorted(ci_summary.get("slice_statuses", {}).items()):
            lines.append(f"| {slice_id} | {status} |")
        lines.append("")

    # Per-slice summaries
    lines.append("## Per-Slice Overview")
    lines.append("")
    lines.append("| Slice | Alignment | Coherence | Metrics |")
    lines.append("|-------|-----------|-----------|---------|")

    for slice_id, slice_report in sorted(multi_report.get("slices", {}).items()):
        alignment = slice_report.get("drift_alignment_score", 0.0)
        coherence = slice_report.get("coherence_score", 0.0)
        metric_count = slice_report.get("metadata", {}).get("metric_count", 0)
        lines.append(f"| {slice_id} | {alignment:.3f} | {coherence:.3f} | {metric_count} |")

    lines.append("")
    lines.append("---")
    lines.append(f"_Report version: {multi_report.get('version', 'N/A')}_")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CSV Export for Drift Grid
# ---------------------------------------------------------------------------

def export_drift_cells_csv(
    report: Dict[str, Any],
    output_path: str,
    *,
    profile: str = "default",
) -> None:
    """
    Export drift cells as CSV for external visualization.

    The CSV has deterministic column ordering matching DRIFT_CELL_COLUMNS.
    This is purely informational and does not affect CI status.

    Args:
        report: Single-slice or multi-slice report dict.
        output_path: Path to write CSV file.
        profile: Threshold profile used for the report.
    """
    cells = build_drift_cells(report, profile=profile)
    cell_dicts = drift_cells_to_dicts(cells)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(DRIFT_CELL_COLUMNS))
        writer.writeheader()
        writer.writerows(cell_dicts)


def drift_cells_to_csv_string(
    report: Dict[str, Any],
    *,
    profile: str = "default",
) -> str:
    """
    Convert drift cells to CSV string (for stdout).

    Args:
        report: Single-slice or multi-slice report dict.
        profile: Threshold profile used.

    Returns:
        CSV-formatted string with header row.
    """
    cells = build_drift_cells(report, profile=profile)
    cell_dicts = drift_cells_to_dicts(cells)

    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=list(DRIFT_CELL_COLUMNS))
    writer.writeheader()
    writer.writerows(cell_dicts)

    return output.getvalue()


# ---------------------------------------------------------------------------
# Slice Discovery
# ---------------------------------------------------------------------------

def discover_slices(slices_dir: str) -> List[Tuple[str, str, str]]:
    """
    Discover slices from a directory.

    Expects files named: {slice_id}_baseline.jsonl and {slice_id}_rfl.jsonl

    Args:
        slices_dir: Path to directory containing slice logs

    Returns:
        List of (slice_id, baseline_path, rfl_path) tuples
    """
    slices = []
    base_path = Path(slices_dir)

    # Find all baseline files
    baseline_files = list(base_path.glob("*_baseline.jsonl"))

    for baseline_path in sorted(baseline_files):
        # Extract slice ID
        slice_id = baseline_path.stem.replace("_baseline", "")

        # Find corresponding RFL file
        rfl_path = base_path / f"{slice_id}_rfl.jsonl"

        if rfl_path.exists():
            slices.append((slice_id, str(baseline_path), str(rfl_path)))

    return slices


# ---------------------------------------------------------------------------
# CI Gate with Pattern Hints
# ---------------------------------------------------------------------------

def evaluate_ci_with_hints(
    slice_id: str,
    trajectories: Dict[str, List[float]],
    window: int,
    threshold: float,
    drift_score_threshold: float,
    coherence_threshold: float,
) -> Dict[str, Any]:
    """
    Evaluate CI gate and include pattern hints.

    Args:
        slice_id: Slice identifier
        trajectories: Metric trajectories
        window: Rolling window size
        threshold: Drift threshold
        drift_score_threshold: CI drift score threshold
        coherence_threshold: CI coherence threshold

    Returns:
        CI result dict with pattern_hints field
    """
    alignment_result = compute_drift_alignment(
        slice_id,
        trajectories,
        window_size=window,
        drift_threshold=threshold,
    )

    ci_gate = evaluate_drift_for_ci(
        alignment_result,
        drift_score_threshold=drift_score_threshold,
        coherence_threshold=coherence_threshold,
    )

    # Add pattern hints to CI result
    pattern_hints = compute_pattern_hints(trajectories)

    result = ci_gate.to_dict()
    result["pattern_hints"] = dict(sorted(pattern_hints.items()))

    return result


# ---------------------------------------------------------------------------
# CLI Main
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    CLI entry point.

    Returns:
        Exit code (0 = success, 1 = CI BLOCK, 2 = error)
    """
    parser = argparse.ArgumentParser(
        description="Drift Alignment Report Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--slice",
        help="Single slice identifier (e.g., slice_uplift_goal)",
    )
    mode_group.add_argument(
        "--all-slices",
        action="store_true",
        help="Process all slices from --slices-dir",
    )

    # Input paths
    parser.add_argument(
        "--baseline-log",
        help="Path to baseline JSONL log (single slice mode)",
    )
    parser.add_argument(
        "--rfl-log",
        help="Path to RFL JSONL log (single slice mode)",
    )
    parser.add_argument(
        "--slices-dir",
        help="Directory containing slice logs (all-slices mode)",
    )

    # Output options
    parser.add_argument(
        "--out",
        help="Output path for JSON report",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON to stdout instead of writing a file",
    )
    parser.add_argument(
        "--markdown-out",
        help="Optional path for Markdown summary output",
    )
    parser.add_argument(
        "--export-csv",
        help="Export drift cells grid as CSV (purely informational)",
    )

    # Analysis parameters
    parser.add_argument(
        "--window",
        type=int,
        default=20,
        help="Rolling window size (default: 20)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Drift threshold (default: 0.1)",
    )

    # CI options
    parser.add_argument(
        "--ci",
        action="store_true",
        help="CI mode: evaluate gate and set exit code",
    )
    parser.add_argument(
        "--profile",
        choices=sorted(DRIFT_THRESHOLDS.keys()),
        default="default",
        help=f"Threshold profile: {', '.join(sorted(DRIFT_THRESHOLDS.keys()))} (default: default)",
    )
    parser.add_argument(
        "--drift-score-threshold",
        type=float,
        help="CI: override drift score threshold from profile",
    )
    parser.add_argument(
        "--coherence-threshold",
        type=float,
        help="CI: override coherence threshold from profile",
    )

    args = parser.parse_args(argv)

    # Get thresholds from profile
    profile_thresholds = get_drift_thresholds(args.profile)
    drift_score_threshold = args.drift_score_threshold or profile_thresholds["drift_score"]
    coherence_threshold = args.coherence_threshold or profile_thresholds["coherence"]

    # Single slice mode
    if args.slice:
        if not args.baseline_log or not args.rfl_log:
            print("Error: --baseline-log and --rfl-log required for single slice mode", file=sys.stderr)
            return 2

        return process_single_slice(
            args.slice,
            args.baseline_log,
            args.rfl_log,
            window=args.window,
            threshold=args.threshold,
            out_path=args.out,
            json_stdout=args.json,
            markdown_out=args.markdown_out,
            export_csv=args.export_csv,
            ci_mode=args.ci,
            drift_score_threshold=drift_score_threshold,
            coherence_threshold=coherence_threshold,
            profile=args.profile,
        )

    # All slices mode
    if args.all_slices:
        if not args.slices_dir:
            print("Error: --slices-dir required for --all-slices mode", file=sys.stderr)
            return 2

        return process_all_slices(
            args.slices_dir,
            window=args.window,
            threshold=args.threshold,
            out_path=args.out,
            json_stdout=args.json,
            markdown_out=args.markdown_out,
            export_csv=args.export_csv,
            ci_mode=args.ci,
            drift_score_threshold=drift_score_threshold,
            coherence_threshold=coherence_threshold,
            profile=args.profile,
        )

    return 0


def process_single_slice(
    slice_id: str,
    baseline_log: str,
    rfl_log: str,
    *,
    window: int,
    threshold: float,
    out_path: Optional[str],
    json_stdout: bool,
    markdown_out: Optional[str],
    export_csv: Optional[str],
    ci_mode: bool,
    drift_score_threshold: float,
    coherence_threshold: float,
    profile: str,
) -> int:
    """Process a single slice."""
    # Load logs
    try:
        baseline_records = load_jsonl(baseline_log)
    except FileNotFoundError:
        print(f"Error: Baseline log not found: {baseline_log}", file=sys.stderr)
        return 2
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in baseline log: {e}", file=sys.stderr)
        return 2

    try:
        rfl_records = load_jsonl(rfl_log)
    except FileNotFoundError:
        print(f"Error: RFL log not found: {rfl_log}", file=sys.stderr)
        return 2
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in RFL log: {e}", file=sys.stderr)
        return 2

    if not baseline_records and not rfl_records:
        print("Error: Both logs are empty", file=sys.stderr)
        return 2

    # Derive trajectories
    trajectories = derive_trajectories_from_logs(baseline_records, rfl_records)

    if not trajectories:
        print("Error: Could not derive any metric trajectories", file=sys.stderr)
        return 2

    # Generate report
    report = generate_drift_report(
        slice_id,
        trajectories,
        window=window,
        threshold=threshold,
    )

    # Add profile info to metadata
    report["metadata"]["ci_profile"] = profile
    report["metadata"]["ci_drift_score_threshold"] = drift_score_threshold
    report["metadata"]["ci_coherence_threshold"] = coherence_threshold

    # CI evaluation
    ci_result = None
    if ci_mode:
        ci_result = evaluate_ci_with_hints(
            slice_id,
            trajectories,
            window,
            threshold,
            drift_score_threshold,
            coherence_threshold,
        )
        report["ci_gate"] = ci_result

    # Output JSON
    json_output = json.dumps(report, indent=2, sort_keys=True)

    if json_stdout:
        print(json_output)
    elif out_path:
        out_p = Path(out_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json_output, encoding="utf-8")
        print(f"Report written to: {out_path}")

    # Output Markdown
    if markdown_out:
        md_content = generate_markdown_report(report, ci_result)
        md_path = Path(markdown_out)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(md_content, encoding="utf-8")
        print(f"Markdown report written to: {markdown_out}")

    # Output CSV (drift cells grid)
    if export_csv:
        export_drift_cells_csv(report, export_csv, profile=profile)
        print(f"Drift cells CSV written to: {export_csv}")

    # CI exit code
    if ci_mode:
        status = ci_result["status"]
        summary = (
            f"status={status} "
            f"max_metric_drift_score={ci_result['max_metric_drift_score']:.3f} "
            f"drift_alignment_score={ci_result['drift_alignment_score']:.3f} "
            f"coherence_score={ci_result['coherence_score']:.3f} "
            f"profile={profile}"
        )
        print(summary)

        if status == "BLOCK":
            return 1

    return 0


def process_all_slices(
    slices_dir: str,
    *,
    window: int,
    threshold: float,
    out_path: Optional[str],
    json_stdout: bool,
    markdown_out: Optional[str],
    export_csv: Optional[str],
    ci_mode: bool,
    drift_score_threshold: float,
    coherence_threshold: float,
    profile: str,
) -> int:
    """Process all slices in a directory."""
    # Discover slices
    slices = discover_slices(slices_dir)

    if not slices:
        print(f"Error: No slices found in {slices_dir}", file=sys.stderr)
        print("Expected files: {slice_id}_baseline.jsonl and {slice_id}_rfl.jsonl", file=sys.stderr)
        return 2

    print(f"Found {len(slices)} slice(s) to process")

    slice_reports: Dict[str, Dict[str, Any]] = {}
    ci_results: Dict[str, Dict[str, Any]] = {}
    any_block = False

    for slice_id, baseline_path, rfl_path in slices:
        print(f"Processing: {slice_id}")

        try:
            baseline_records = load_jsonl(baseline_path)
            rfl_records = load_jsonl(rfl_path)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"  Error loading logs: {e}", file=sys.stderr)
            continue

        trajectories = derive_trajectories_from_logs(baseline_records, rfl_records)

        if not trajectories:
            print(f"  Warning: No trajectories derived for {slice_id}")
            continue

        # Generate report
        report = generate_drift_report(
            slice_id,
            trajectories,
            window=window,
            threshold=threshold,
        )

        # Add profile info
        report["metadata"]["ci_profile"] = profile
        report["metadata"]["ci_drift_score_threshold"] = drift_score_threshold
        report["metadata"]["ci_coherence_threshold"] = coherence_threshold

        slice_reports[slice_id] = report

        # CI evaluation
        if ci_mode:
            ci_result = evaluate_ci_with_hints(
                slice_id,
                trajectories,
                window,
                threshold,
                drift_score_threshold,
                coherence_threshold,
            )
            ci_results[slice_id] = ci_result
            report["ci_gate"] = ci_result

            if ci_result["status"] == "BLOCK":
                any_block = True

    # Generate combined report
    multi_report = generate_multi_slice_report(
        slice_reports,
        ci_results if ci_mode else None,
    )

    # Add metadata
    multi_report["metadata"] = {
        "window_size": window,
        "drift_threshold": threshold,
        "ci_profile": profile,
        "ci_drift_score_threshold": drift_score_threshold,
        "ci_coherence_threshold": coherence_threshold,
    }

    # Output JSON
    json_output = json.dumps(multi_report, indent=2, sort_keys=True)

    if json_stdout:
        print(json_output)
    elif out_path:
        out_p = Path(out_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json_output, encoding="utf-8")
        print(f"Multi-slice report written to: {out_path}")

    # Output Markdown
    if markdown_out:
        md_content = generate_multi_slice_markdown(multi_report)
        md_path = Path(markdown_out)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(md_content, encoding="utf-8")
        print(f"Markdown report written to: {markdown_out}")

    # Output CSV (drift cells grid)
    if export_csv:
        export_drift_cells_csv(multi_report, export_csv, profile=profile)
        print(f"Drift cells CSV written to: {export_csv}")

    # CI exit code
    if ci_mode:
        statuses = [r["status"] for r in ci_results.values()]
        status_summary = ", ".join(f"{s}={statuses.count(s)}" for s in sorted(set(statuses)))
        print(f"CI summary: {status_summary} profile={profile}")

        if any_block:
            print("BLOCK: One or more slices exceeded drift thresholds")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
