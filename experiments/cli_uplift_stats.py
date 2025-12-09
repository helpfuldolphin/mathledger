#!/usr/bin/env python3
"""
CLI for Bootstrap Uplift Statistics

Exposes the paired bootstrap engine as a command-line tool for computing
uplift confidence intervals between baseline and RFL experiment logs.

Usage:
    python -m experiments.cli_uplift_stats \
        --baseline-log path/to/baseline.jsonl \
        --rfl-log path/to/rfl.jsonl \
        --metric-path success \
        --n-bootstrap 10000 \
        --seed 42

Output:
    JSON summary to stdout with delta, ci_lower, ci_upper, n_baseline, n_rfl.
    No decision logic — just numbers.

GOVERNANCE CONSTRAINTS:
    - No promotion decisions
    - No "significant / not significant" labels
    - Deterministic: fixed seeds, sorted data, sorted keys
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from statistical.bootstrap import paired_bootstrap_delta, PairedBootstrapResult


def _compute_analysis_id(
    baseline_path: str,
    rfl_path: str,
    metric_path: str,
    seed: int,
    n_bootstrap: int,
) -> str:
    """
    Compute deterministic analysis ID as SHA-256 hash of input parameters.
    
    This provides a unique, reproducible identifier for each analysis run.
    The same inputs will always produce the same analysis_id.
    """
    # Normalize paths to use forward slashes for cross-platform consistency
    normalized_baseline = str(Path(baseline_path).as_posix())
    normalized_rfl = str(Path(rfl_path).as_posix())
    
    # Create canonical string representation
    canonical = f"{normalized_baseline}|{normalized_rfl}|{metric_path}|{seed}|{n_bootstrap}"
    
    # Compute SHA-256
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """
    Load a JSONL file and return list of records.
    
    Each line is expected to be a valid JSON object.
    Empty lines are skipped.
    """
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
                records.append(record)
            except json.JSONDecodeError as e:
                print(
                    f"Warning: Invalid JSON at {path}:{line_num}: {e}",
                    file=sys.stderr
                )
    return records


def _sort_records_by_cycle(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Sort records by cycle index for deterministic ordering.
    
    Looks for common cycle fields: 'cycle', 'cycle_index', 'cycle_id'.
    Records without a cycle field are placed at the end.
    """
    def get_cycle_key(record: Dict[str, Any]) -> tuple:
        # Try common cycle field names
        for field in ['cycle', 'cycle_index', 'cycle_id']:
            if field in record:
                val = record[field]
                if isinstance(val, (int, float)):
                    return (0, val)  # Has cycle, sort by value
                elif isinstance(val, str):
                    # Try to extract numeric part
                    try:
                        return (0, int(val))
                    except ValueError:
                        return (0, val)
        # No cycle field found, put at end
        return (1, 0)
    
    return sorted(records, key=get_cycle_key)


def _extract_metric(
    records: List[Dict[str, Any]],
    metric_path: str,
) -> np.ndarray:
    """
    Extract metric values from records using dot-notation path.
    
    Parameters
    ----------
    records : list of dict
        Log records loaded from JSONL.
        
    metric_path : str
        Dot-separated path to metric, e.g., "success", "derivation.verified",
        "timing.duration_ms".
    
    Returns
    -------
    np.ndarray
        Array of metric values. Missing values are excluded.
        Boolean values are converted to 0/1.
    
    Examples
    --------
    >>> records = [{"success": True}, {"success": False}]
    >>> _extract_metric(records, "success")
    array([1., 0.])
    
    >>> records = [{"derivation": {"verified": 0.95}}]
    >>> _extract_metric(records, "derivation.verified")
    array([0.95])
    """
    path_parts = metric_path.split('.')
    values = []
    
    for record in records:
        value = record
        try:
            for part in path_parts:
                if isinstance(value, dict):
                    value = value[part]
                else:
                    raise KeyError(part)
            
            # Convert booleans to numeric
            if isinstance(value, bool):
                value = 1.0 if value else 0.0
            elif isinstance(value, (int, float)):
                value = float(value)
            else:
                # Skip non-numeric values
                continue
            
            values.append(value)
        except (KeyError, TypeError):
            # Metric not found in this record - skip
            continue
    
    return np.array(values, dtype=np.float64)


def format_summary_line(
    metric_path: str,
    delta: float,
    ci_lower: float,
    ci_upper: float,
    n_baseline: int,
    n_rfl: int,
    method: str,
) -> str:
    """
    Format a single evidence summary line for CI logs and human grepping.
    
    Format (single line, no newlines):
        BootstrapEvidence: metric=<metric_path> delta=<delta> ci=[<ci_lower>,<ci_upper>] n_base=<n_baseline> n_rfl=<n_rfl> method=<method>
    
    CONTRACT:
        - Exactly one line, no trailing newline
        - Deterministic format with fixed 6 decimal places
        - No interpretive language
    
    Parameters
    ----------
    metric_path : str
        Path to the metric analyzed.
    delta : float
        Point estimate of the difference.
    ci_lower : float
        Lower bound of the confidence interval.
    ci_upper : float
        Upper bound of the confidence interval.
    n_baseline : int
        Number of baseline observations.
    n_rfl : int
        Number of RFL observations.
    method : str
        Bootstrap method used.
    
    Returns
    -------
    str
        Single-line summary string.
    """
    return (
        f"BootstrapEvidence: "
        f"metric={metric_path} "
        f"delta={delta:.6f} "
        f"ci=[{ci_lower:.6f},{ci_upper:.6f}] "
        f"n_base={n_baseline} "
        f"n_rfl={n_rfl} "
        f"method={method}"
    )


def compute_uplift_stats(
    baseline_log: Path,
    rfl_log: Path,
    metric_path: str,
    n_bootstrap: int = 10000,
    seed: int = 42,
    metric_name: Optional[str] = None,
    baseline_label: Optional[str] = None,
    rfl_label: Optional[str] = None,
    sort_by_cycle: bool = False,
) -> Dict[str, Any]:
    """
    Compute paired bootstrap statistics for uplift between baseline and RFL logs.
    
    Parameters
    ----------
    baseline_log : Path
        Path to baseline experiment JSONL log.
        
    rfl_log : Path
        Path to RFL experiment JSONL log.
        
    metric_path : str
        Dot-notation path to the metric field (e.g., "success", "derivation.verified").
        
    n_bootstrap : int, default=10000
        Number of bootstrap resamples.
        
    seed : int, default=42
        Random seed for deterministic results.
        
    metric_name : str, optional
        Human-readable name for the metric (included in output).
        
    baseline_label : str, optional
        Label for the baseline condition (for multi-run contexts).
        
    rfl_label : str, optional
        Label for the RFL condition (for multi-run contexts).
        
    sort_by_cycle : bool, default=False
        If True, sort log records by cycle index before extraction.
    
    Returns
    -------
    dict
        JSON-serializable dictionary with:
        - analysis_id: SHA-256 of input parameters
        - delta: Point estimate of mean difference (RFL - baseline)
        - ci_lower: Lower bound of 95% CI
        - ci_upper: Upper bound of 95% CI
        - n_baseline: Number of baseline observations
        - n_rfl: Number of RFL observations
        - metric_path: The metric analyzed
        - seed: Random seed used
        - n_bootstrap: Number of resamples
    """
    # Load logs
    baseline_records = _load_jsonl(baseline_log)
    rfl_records = _load_jsonl(rfl_log)
    
    # Sort by cycle if requested
    if sort_by_cycle:
        baseline_records = _sort_records_by_cycle(baseline_records)
        rfl_records = _sort_records_by_cycle(rfl_records)
    
    # Extract metric values
    baseline_values = _extract_metric(baseline_records, metric_path)
    rfl_values = _extract_metric(rfl_records, metric_path)
    
    n_baseline = len(baseline_values)
    n_rfl = len(rfl_values)
    
    # Compute analysis_id
    analysis_id = _compute_analysis_id(
        str(baseline_log),
        str(rfl_log),
        metric_path,
        seed,
        n_bootstrap,
    )
    
    # Validate we have data
    if n_baseline < 2:
        return {
            "analysis_id": analysis_id,
            "error": f"Insufficient baseline data: {n_baseline} values (need >= 2)",
            "n_baseline": n_baseline,
            "n_rfl": n_rfl,
            "metric_path": metric_path,
        }
    
    if n_rfl < 2:
        return {
            "analysis_id": analysis_id,
            "error": f"Insufficient RFL data: {n_rfl} values (need >= 2)",
            "n_baseline": n_baseline,
            "n_rfl": n_rfl,
            "metric_path": metric_path,
        }
    
    # Handle mismatched lengths by truncating to minimum
    # This maintains paired structure for comparable samples
    if n_baseline != n_rfl:
        min_n = min(n_baseline, n_rfl)
        # Sort to ensure determinism, then truncate
        baseline_values = np.sort(baseline_values)[:min_n]
        rfl_values = np.sort(rfl_values)[:min_n]
        paired_note = f"Truncated to {min_n} paired observations (baseline={n_baseline}, rfl={n_rfl})"
    else:
        # Sort for determinism even when lengths match
        baseline_values = np.sort(baseline_values)
        rfl_values = np.sort(rfl_values)
        paired_note = None
    
    # Compute bootstrap CI
    result: PairedBootstrapResult = paired_bootstrap_delta(
        baseline_values,
        rfl_values,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    
    # Build core evidence schema (LOCKED - 10 fields)
    # This is the canonical PairedDeltaResult schema
    output: Dict[str, Any] = {
        "analysis_id": analysis_id,
        "ci_lower": float(result.CI_low),
        "ci_upper": float(result.CI_high),
        "delta": float(result.delta_mean),
        "method": result.method,
        "metric_path": metric_path,
        "n_baseline": n_baseline,
        "n_bootstrap": n_bootstrap,
        "n_rfl": n_rfl,
        "seed": seed,
    }
    
    # Build annotations block for optional labels (separate from core schema)
    # These do NOT affect contract compliance - they are metadata
    annotations: Dict[str, Any] = {}
    
    if baseline_label:
        annotations["baseline_label"] = baseline_label
    
    if metric_name:
        annotations["metric_name"] = metric_name
    
    if rfl_label:
        annotations["rfl_label"] = rfl_label
    
    if sort_by_cycle:
        annotations["sorted_by_cycle"] = True
    
    # Only include annotations block if non-empty
    if annotations:
        output["annotations"] = annotations
    
    # Add processing notes if applicable (not part of core schema)
    if paired_note:
        output["_note"] = paired_note
    
    return output


def main(argv: Optional[List[str]] = None) -> int:
    """
    CLI entry point.
    
    Returns
    -------
    int
        Exit code: 0 for success, 1 for error.
    """
    parser = argparse.ArgumentParser(
        description="Compute bootstrap uplift statistics between baseline and RFL logs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compute success rate uplift
  python -m experiments.cli_uplift_stats \\
      --baseline-log baseline.jsonl \\
      --rfl-log rfl.jsonl \\
      --metric-path success

  # Compute nested metric with custom bootstrap settings
  python -m experiments.cli_uplift_stats \\
      --baseline-log baseline.jsonl \\
      --rfl-log rfl.jsonl \\
      --metric-path derivation.verified \\
      --n-bootstrap 20000 \\
      --seed 123

  # With labels for multi-run contexts
  python -m experiments.cli_uplift_stats \\
      --baseline-log baseline.jsonl \\
      --rfl-log rfl.jsonl \\
      --metric-path success \\
      --metric-name "Success Rate" \\
      --baseline-label "v1.0.0" \\
      --rfl-label "v1.1.0-rfl"

  # Sort by cycle before extraction
  python -m experiments.cli_uplift_stats \\
      --baseline-log baseline.jsonl \\
      --rfl-log rfl.jsonl \\
      --metric-path success \\
      --sort-by-cycle

Output is a JSON object with analysis_id, delta, ci_lower, ci_upper, n_baseline, n_rfl.
No decision logic — just numbers.
        """,
    )
    
    parser.add_argument(
        "--baseline-log",
        type=Path,
        required=True,
        help="Path to baseline experiment JSONL log",
    )
    
    parser.add_argument(
        "--rfl-log",
        type=Path,
        required=True,
        help="Path to RFL experiment JSONL log",
    )
    
    parser.add_argument(
        "--metric-path",
        type=str,
        required=True,
        help="Dot-notation path to metric (e.g., 'success', 'derivation.verified')",
    )
    
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=10000,
        help="Number of bootstrap resamples (default: 10000)",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for determinism (default: 42)",
    )
    
    parser.add_argument(
        "--metric-name",
        type=str,
        default=None,
        help="Human-readable metric name to include in output",
    )
    
    parser.add_argument(
        "--baseline-label",
        type=str,
        default=None,
        help="Label for baseline condition (for multi-run contexts)",
    )
    
    parser.add_argument(
        "--rfl-label",
        type=str,
        default=None,
        help="Label for RFL condition (for multi-run contexts)",
    )
    
    parser.add_argument(
        "--sort-by-cycle",
        action="store_true",
        help="Sort log records by cycle index before extraction",
    )
    
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output with indentation",
    )
    
    parser.add_argument(
        "--summary-line",
        action="store_true",
        help="Output single summary line for CI logs instead of JSON",
    )
    
    args = parser.parse_args(argv)
    
    # Validate paths exist
    if not args.baseline_log.exists():
        print(
            json.dumps({"error": f"Baseline log not found: {args.baseline_log}"}, sort_keys=True),
            file=sys.stdout
        )
        return 1
    
    if not args.rfl_log.exists():
        print(
            json.dumps({"error": f"RFL log not found: {args.rfl_log}"}, sort_keys=True),
            file=sys.stdout
        )
        return 1
    
    # Validate n_bootstrap range
    if args.n_bootstrap < 1000:
        print(
            json.dumps({"error": f"n_bootstrap must be >= 1000, got {args.n_bootstrap}"}, sort_keys=True),
            file=sys.stdout
        )
        return 1
    
    if args.n_bootstrap > 100000:
        print(
            json.dumps({"error": f"n_bootstrap must be <= 100000, got {args.n_bootstrap}"}, sort_keys=True),
            file=sys.stdout
        )
        return 1
    
    # Compute statistics
    try:
        result = compute_uplift_stats(
            baseline_log=args.baseline_log,
            rfl_log=args.rfl_log,
            metric_path=args.metric_path,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
            metric_name=args.metric_name,
            baseline_label=args.baseline_label,
            rfl_label=args.rfl_label,
            sort_by_cycle=args.sort_by_cycle,
        )
    except Exception as e:
        print(
            json.dumps({"error": str(e)}, sort_keys=True),
            file=sys.stdout
        )
        return 1
    
    # Check for errors before output
    if "error" in result:
        print(json.dumps(result, sort_keys=True))
        return 1
    
    # Output based on format requested
    if args.summary_line:
        # Summary line format for CI logs
        summary = format_summary_line(
            metric_path=result["metric_path"],
            delta=result["delta"],
            ci_lower=result["ci_lower"],
            ci_upper=result["ci_upper"],
            n_baseline=result["n_baseline"],
            n_rfl=result["n_rfl"],
            method=result["method"],
        )
        print(summary)
    else:
        # JSON output with sorted keys for determinism
        indent = 2 if args.pretty else None
        print(json.dumps(result, indent=indent, sort_keys=True))
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
