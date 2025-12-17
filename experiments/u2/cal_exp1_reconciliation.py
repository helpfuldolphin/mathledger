"""
PHASE II — NOT USED IN PHASE I

CAL-EXP-1 Reconciliation Module
================================

Provides reconciliation analysis for comparing two CAL-EXP-1 runs.
Pure analysis mode: no execution, no changes to harness.

SHADOW MODE CONTRACT:
- All analysis is advisory only
- Deterministic: identical inputs → identical output (timestamps excluded)
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def load_cal_exp1_report(run_dir: Path) -> Optional[Dict[str, Any]]:
    """
    Load CAL-EXP-1 report from run directory.
    
    Tries:
    1. run_dir/cal_exp1_report.json
    2. run_dir/calibration/cal_exp1_report.json
    3. run_dir/cal_exp1_metrics.json
    4. Evidence pack manifest: governance.p5_calibration.cal_exp1
    
    Args:
        run_dir: Run directory path
        
    Returns:
        Report dict or None if not found
    """
    # Try direct report files
    candidates = [
        run_dir / "cal_exp1_report.json",
        run_dir / "calibration" / "cal_exp1_report.json",
        run_dir / "cal_exp1_metrics.json",
    ]
    
    for candidate in candidates:
        if candidate.exists():
            try:
                with open(candidate, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load {candidate}: {e}")
                continue
    
    # Try evidence pack manifest
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            
            # Navigate to governance.p5_calibration.cal_exp1
            cal_exp1_data = manifest.get("governance", {}).get("p5_calibration", {}).get("cal_exp1")
            if cal_exp1_data:
                # If it's a path, try to load it
                if isinstance(cal_exp1_data, str):
                    cal_path = run_dir / cal_exp1_data
                    if cal_path.exists():
                        with open(cal_path, 'r') as f:
                            return json.load(f)
                # If it's already a dict, return it
                elif isinstance(cal_exp1_data, dict):
                    return cal_exp1_data
        except (json.JSONDecodeError, IOError, KeyError) as e:
            logger.warning(f"Failed to load from manifest: {e}")
            pass
    
    return None


def extract_window_metrics(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract per-window metrics from CAL-EXP-1 report.
    
    Args:
        report: CAL-EXP-1 report dict
        
    Returns:
        List of window metrics dicts
    """
    windows = report.get("windows", [])
    if not windows:
        return []
    
    # Extract metrics of truth: mean_delta_p and state_divergence_rate (divergence_rate)
    window_metrics = []
    for window in windows:
        window_metrics.append({
            "start_cycle": window.get("start_cycle", 0),
            "end_cycle": window.get("end_cycle", 0),
            "mean_delta_p": window.get("mean_delta_p", 0.0),
            "state_divergence_rate": window.get("divergence_rate", 0.0),
            "delta_bias": window.get("delta_bias", 0.0),
            "delta_variance": window.get("delta_variance", 0.0),
            "pattern_tag": window.get("pattern_tag", "UNKNOWN"),
        })
    
    return window_metrics


def reconcile_cal_exp1_runs(
    run_a_dir: Path,
    run_b_dir: Path,
) -> Dict[str, Any]:
    """
    Reconcile two CAL-EXP-1 runs using canonical truth metrics.
    
    Pure analysis mode: no execution, no changes to harness.
    Deterministic: identical inputs → identical output (timestamps excluded).
    
    Args:
        run_a_dir: First run directory
        run_b_dir: Second run directory
        
    Returns:
        Reconciliation result dict with:
        - schema_version
        - metric_of_truth: ["mean_delta_p", "state_divergence_rate"]
        - side_by_side_deltas: Per-window comparison
        - reconciliation_verdict: "AGREE" | "MIXED" | "CONTRADICT"
        - explainability: List of neutral explanations
    """
    # Load reports
    report_a = load_cal_exp1_report(run_a_dir)
    report_b = load_cal_exp1_report(run_b_dir)
    
    if report_a is None:
        raise FileNotFoundError(f"CAL-EXP-1 report not found in {run_a_dir}")
    if report_b is None:
        raise FileNotFoundError(f"CAL-EXP-1 report not found in {run_b_dir}")
    
    # Extract window metrics
    windows_a = extract_window_metrics(report_a)
    windows_b = extract_window_metrics(report_b)
    
    if not windows_a or not windows_b:
        raise ValueError("No window metrics found in one or both reports")
    
    # Align windows by cycle range (handle different window counts)
    side_by_side: List[Dict[str, Any]] = []
    max_windows = max(len(windows_a), len(windows_b))
    
    for i in range(max_windows):
        window_a = windows_a[i] if i < len(windows_a) else None
        window_b = windows_b[i] if i < len(windows_b) else None
        
        if window_a is None or window_b is None:
            # Handle mismatched window counts
            side_by_side.append({
                "window_index": i,
                "run_a": window_a,
                "run_b": window_b,
                "aligned": False,
            })
            continue
        
        # Compute deltas for metrics of truth
        delta_mean_delta_p = window_b["mean_delta_p"] - window_a["mean_delta_p"]
        delta_state_divergence_rate = window_b["state_divergence_rate"] - window_a["state_divergence_rate"]
        
        # Determine agreement per metric (within tolerance)
        tolerance_mean_delta_p = 0.01  # 1% tolerance
        tolerance_state_divergence_rate = 0.05  # 5% tolerance
        
        mean_delta_p_agree = abs(delta_mean_delta_p) <= tolerance_mean_delta_p
        state_divergence_rate_agree = abs(delta_state_divergence_rate) <= tolerance_state_divergence_rate
        
        side_by_side.append({
            "window_index": i,
            "start_cycle_a": window_a["start_cycle"],
            "end_cycle_a": window_a["end_cycle"],
            "start_cycle_b": window_b["start_cycle"],
            "end_cycle_b": window_b["end_cycle"],
            "run_a": {
                "mean_delta_p": round(window_a["mean_delta_p"], 6),
                "state_divergence_rate": round(window_a["state_divergence_rate"], 6),
            },
            "run_b": {
                "mean_delta_p": round(window_b["mean_delta_p"], 6),
                "state_divergence_rate": round(window_b["state_divergence_rate"], 6),
            },
            "deltas": {
                "mean_delta_p": round(delta_mean_delta_p, 6),
                "state_divergence_rate": round(delta_state_divergence_rate, 6),
            },
            "agreement": {
                "mean_delta_p": mean_delta_p_agree,
                "state_divergence_rate": state_divergence_rate_agree,
            },
            "aligned": True,
        })
    
    # Compute overall reconciliation verdict
    aligned_windows = [w for w in side_by_side if w.get("aligned", False)]
    if not aligned_windows:
        reconciliation_verdict = "CONTRADICT"
        explainability = [
            "No aligned windows found. Window structure mismatch between runs.",
        ]
    else:
        # Count agreements per metric
        mean_delta_p_agreements = sum(1 for w in aligned_windows if w["agreement"]["mean_delta_p"])
        state_divergence_rate_agreements = sum(1 for w in aligned_windows if w["agreement"]["state_divergence_rate"])
        total_aligned = len(aligned_windows)
        
        mean_delta_p_agreement_rate = mean_delta_p_agreements / total_aligned if total_aligned > 0 else 0.0
        state_divergence_rate_agreement_rate = state_divergence_rate_agreements / total_aligned if total_aligned > 0 else 0.0
        
        # Verdict logic
        both_agree = mean_delta_p_agreement_rate >= 0.8 and state_divergence_rate_agreement_rate >= 0.8
        both_contradict = mean_delta_p_agreement_rate < 0.5 and state_divergence_rate_agreement_rate < 0.5
        
        if both_agree:
            reconciliation_verdict = "AGREE"
        elif both_contradict:
            reconciliation_verdict = "CONTRADICT"
        else:
            reconciliation_verdict = "MIXED"
        
        # Build explainability
        explainability = [
            f"Total aligned windows: {total_aligned}",
            f"mean_delta_p agreement: {mean_delta_p_agreements}/{total_aligned} ({mean_delta_p_agreement_rate:.1%})",
            f"state_divergence_rate agreement: {state_divergence_rate_agreements}/{total_aligned} ({state_divergence_rate_agreement_rate:.1%})",
            f"Tolerance for mean_delta_p: ±{tolerance_mean_delta_p}",
            f"Tolerance for state_divergence_rate: ±{tolerance_state_divergence_rate}",
        ]
        
        if reconciliation_verdict == "AGREE":
            explainability.append("Both metrics of truth show consistent agreement across windows.")
        elif reconciliation_verdict == "CONTRADICT":
            explainability.append("Both metrics of truth show significant disagreement across windows.")
        else:
            explainability.append("Mixed agreement: one metric agrees while the other contradicts.")
    
    # Extract run metadata (excluding timestamps for determinism)
    run_a_metadata = {
        "run_dir": str(run_a_dir),
        "schema_version": report_a.get("schema_version", "unknown"),
        "params": report_a.get("params", {}),
        "window_count": len(windows_a),
    }
    
    run_b_metadata = {
        "run_dir": str(run_b_dir),
        "schema_version": report_b.get("schema_version", "unknown"),
        "params": report_b.get("params", {}),
        "window_count": len(windows_b),
    }
    
    return {
        "schema_version": "1.0.0",
        "metric_of_truth": ["mean_delta_p", "state_divergence_rate"],
        "run_a_metadata": run_a_metadata,
        "run_b_metadata": run_b_metadata,
        "side_by_side_deltas": side_by_side,
        "reconciliation_verdict": reconciliation_verdict,
        "explainability": explainability,
    }

