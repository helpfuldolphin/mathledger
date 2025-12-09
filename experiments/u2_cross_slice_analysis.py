#!/usr/bin/env python3
"""
PHASE II — NOT USED IN PHASE I

Cross-Slice Comparative Analytics Engine
=========================================

This module provides tools for comparing behavioral patterns across slices
without computing or implying Δp (uplift). It focuses on structural and
behavioral comparisons only.

Agent: metrics-engineer-6

Usage:
    uv run python experiments/u2_cross_slice_analysis.py --input-dir results/uplift_u2 --out summary.json

ABSOLUTE SAFEGUARDS:
    - No interpretation of uplift.
    - No modification of slice definitions.
    - No modification of success metrics.
    - This code MUST NOT compute or imply Δp.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to sys.path for imports
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# ===========================================================================
# PHASE II — DATA STRUCTURES
# ===========================================================================

@dataclass
class SliceResults:
    """
    Container for results from a single slice run.
    Holds raw records and derived behavioral metrics.
    """
    slice_name: str
    mode: str  # 'baseline' or 'rfl'
    records: List[Dict[str, Any]] = field(default_factory=list)
    source_path: Optional[str] = None


@dataclass
class BehavioralPattern:
    """
    Captures behavioral patterns from a slice run.
    These are structural observations, NOT outcome interpretations.
    """
    metric_value_distribution: Dict[str, int] = field(default_factory=dict)
    chain_depth_distribution: Dict[int, int] = field(default_factory=dict)
    abstention_trend: List[float] = field(default_factory=list)
    candidate_ordering_entropy: List[float] = field(default_factory=list)
    policy_weight_movement_norms: List[float] = field(default_factory=list)


@dataclass
class BehavioralFingerprint:
    """
    A deterministic fingerprint summarizing slice behavior.
    Used for reproducibility checks and structural comparisons.
    """
    slice_name: str
    mode: str
    metric_value_histogram: Dict[str, int]
    longest_chain_distribution: Dict[int, int]
    goal_hit_distribution: Dict[int, int]
    temporal_smoothness_signature: float
    fingerprint_hash: str


# ===========================================================================
# PHASE II — CORE FUNCTIONS
# ===========================================================================

def load_slice_results(slice_name: str, mode: str, input_dir: Path) -> SliceResults:
    """
    Load results for a specific slice and mode from JSONL files.
    
    Args:
        slice_name: Name of the slice (e.g., 'arithmetic_simple', 'slice_uplift_goal')
        mode: Either 'baseline' or 'rfl'
        input_dir: Directory containing the result files
        
    Returns:
        SliceResults object with loaded records
        
    Raises:
        FileNotFoundError: If the expected JSONL file does not exist
    """
    # Try different naming conventions used in the codebase
    patterns = [
        f"uplift_u2_{slice_name}_{mode}.jsonl",
        f"{slice_name}_{mode}.jsonl",
        f"uplift_u2_{slice_name.replace('_', '-')}_{mode}.jsonl",
    ]
    
    result_path = None
    for pattern in patterns:
        candidate = input_dir / pattern
        if candidate.exists():
            result_path = candidate
            break
    
    if result_path is None:
        raise FileNotFoundError(
            f"No result file found for slice='{slice_name}', mode='{mode}' in {input_dir}. "
            f"Tried patterns: {patterns}"
        )
    
    records = []
    with open(result_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    record = json.loads(line)
                    records.append(record)
                except json.JSONDecodeError as e:
                    print(f"WARNING: Skipping malformed line in {result_path}: {e}", file=sys.stderr)
    
    return SliceResults(
        slice_name=slice_name,
        mode=mode,
        records=records,
        source_path=str(result_path),
    )


def _compute_metric_value_distribution(records: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Compute histogram of metric_value rounded to 1 decimal place.
    Returns a deterministically sorted dictionary.
    """
    values = []
    for r in records:
        mv = r.get("metric_value")
        if mv is not None:
            # Round to 1 decimal for binning
            values.append(f"{float(mv):.1f}")
        elif "success" in r:
            # Fallback to success as binary metric
            values.append("1.0" if r["success"] else "0.0")
    
    counter = Counter(values)
    # Sort keys deterministically (string sort)
    return {k: counter[k] for k in sorted(counter.keys())}


def _compute_chain_depth_distribution(records: List[Dict[str, Any]]) -> Dict[int, int]:
    """
    Compute distribution of chain depths (longest derivation chains).
    Returns a deterministically sorted dictionary.
    """
    depths = []
    for r in records:
        # Check various possible locations for chain depth info
        derivation = r.get("derivation", {})
        if isinstance(derivation, dict):
            # Look for verified_count or candidates_tried as proxy for chain depth
            verified_count = derivation.get("verified_count", 0)
            if verified_count:
                depths.append(verified_count)
        
        # Also check metric_result for chain metrics
        metric_result = r.get("metric_result", {})
        if isinstance(metric_result, dict):
            # Goal hit total_verified as chain indicator
            total_verified = metric_result.get("total_verified", 0)
            if total_verified:
                depths.append(int(total_verified))
    
    if not depths:
        # No chain info available - count as depth 0
        return {0: len(records)}
    
    counter = Counter(depths)
    return {k: counter[k] for k in sorted(counter.keys())}


def _compute_abstention_trend(records: List[Dict[str, Any]], window_size: int = 10) -> List[float]:
    """
    Compute rolling abstention rate over windows.
    Abstention is defined as success=False.
    Returns a list of floats representing abstention rate per window.
    """
    if not records:
        return []
    
    # Extract abstention indicator
    abstentions = []
    for r in records:
        # Check various indicators for abstention
        if r.get("abstention") is True:
            abstentions.append(1.0)
        elif r.get("success") is False:
            abstentions.append(1.0)
        elif r.get("status") == "abstain":
            abstentions.append(1.0)
        else:
            abstentions.append(0.0)
    
    # Compute rolling rate
    trend = []
    effective_window = min(window_size, len(abstentions))
    if effective_window == 0:
        return []
    
    for i in range(0, len(abstentions), effective_window):
        window = abstentions[i:i + effective_window]
        if window:
            rate = sum(window) / len(window)
            # Round to 4 decimal places for determinism
            trend.append(round(rate, 4))
    
    return trend


def _compute_candidate_ordering_entropy(records: List[Dict[str, Any]]) -> List[float]:
    """
    Compute entropy of candidate ordering per cycle.
    Higher entropy = more uniform distribution of candidate positions.
    Returns a list of entropy values per record.
    """
    entropies = []
    
    for r in records:
        derivation = r.get("derivation", {})
        if not isinstance(derivation, dict):
            entropies.append(0.0)
            continue
            
        candidate_order = derivation.get("candidate_order", [])
        if not candidate_order:
            entropies.append(0.0)
            continue
        
        # Compute entropy based on position distribution
        # Use hash prefix as category for entropy calculation
        n = len(candidate_order)
        if n <= 1:
            entropies.append(0.0)
            continue
        
        # Take first 4 chars of each hash as category
        prefixes = [c[:4] if len(c) >= 4 else c for c in candidate_order]
        prefix_counts = Counter(prefixes)
        
        # Shannon entropy
        total = sum(prefix_counts.values())
        entropy = 0.0
        for count in prefix_counts.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        
        # Normalize by max possible entropy
        max_entropy = math.log2(n) if n > 1 else 1.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        entropies.append(round(normalized_entropy, 4))
    
    return entropies


def _compute_policy_weight_movement(records: List[Dict[str, Any]]) -> List[float]:
    """
    Compute norm of policy weight movement between consecutive cycles.
    This measures how much the RFL policy is adapting.
    Returns L2 norms of weight changes (placeholder - actual weights may not be logged).
    """
    # Since actual policy weights are typically not logged in JSONL,
    # we compute a proxy based on success rate changes
    if len(records) < 2:
        return []
    
    norms = []
    prev_success = 1.0 if records[0].get("success") else 0.0
    
    for r in records[1:]:
        curr_success = 1.0 if r.get("success") else 0.0
        # L1 distance as proxy for weight movement
        movement = abs(curr_success - prev_success)
        norms.append(round(movement, 4))
        prev_success = curr_success
    
    return norms


def compare_behavioral_patterns(
    baseline_results: SliceResults,
    rfl_results: SliceResults,
) -> Dict[str, Any]:
    """
    Compare behavioral patterns between baseline and RFL runs.
    
    IMPORTANT: This function compares STRUCTURAL patterns only.
    It does NOT compute or imply Δp (uplift).
    
    Args:
        baseline_results: Results from baseline mode
        rfl_results: Results from RFL mode
        
    Returns:
        Dictionary containing structural comparison metrics
    """
    baseline_pattern = _extract_behavioral_pattern(baseline_results.records)
    rfl_pattern = _extract_behavioral_pattern(rfl_results.records)
    
    comparison = {
        "phase_label": "PHASE II — NOT USED IN PHASE I",
        "slice_name": baseline_results.slice_name,
        "baseline_mode": baseline_results.mode,
        "rfl_mode": rfl_results.mode,
        "baseline_record_count": len(baseline_results.records),
        "rfl_record_count": len(rfl_results.records),
        "metric_histogram_match": _compare_histograms(
            baseline_pattern.metric_value_distribution,
            rfl_pattern.metric_value_distribution,
        ),
        "chain_depth_histogram_match": _compare_histograms(
            baseline_pattern.chain_depth_distribution,
            rfl_pattern.chain_depth_distribution,
        ),
        "abstention_trend_correlation": _compute_trend_correlation(
            baseline_pattern.abstention_trend,
            rfl_pattern.abstention_trend,
        ),
        "ordering_entropy_mean_baseline": _safe_mean(baseline_pattern.candidate_ordering_entropy),
        "ordering_entropy_mean_rfl": _safe_mean(rfl_pattern.candidate_ordering_entropy),
        "weight_movement_norm_baseline": _safe_mean(baseline_pattern.policy_weight_movement_norms),
        "weight_movement_norm_rfl": _safe_mean(rfl_pattern.policy_weight_movement_norms),
    }
    
    return comparison


def _extract_behavioral_pattern(records: List[Dict[str, Any]]) -> BehavioralPattern:
    """Extract behavioral pattern from records."""
    return BehavioralPattern(
        metric_value_distribution=_compute_metric_value_distribution(records),
        chain_depth_distribution=_compute_chain_depth_distribution(records),
        abstention_trend=_compute_abstention_trend(records),
        candidate_ordering_entropy=_compute_candidate_ordering_entropy(records),
        policy_weight_movement_norms=_compute_policy_weight_movement(records),
    )


def _compare_histograms(hist1: Dict, hist2: Dict) -> Dict[str, Any]:
    """
    Compare two histograms and return structural comparison.
    Does NOT interpret differences as uplift.
    """
    all_keys = set(hist1.keys()) | set(hist2.keys())
    
    shared_keys = set(hist1.keys()) & set(hist2.keys())
    unique_to_hist1 = set(hist1.keys()) - set(hist2.keys())
    unique_to_hist2 = set(hist2.keys()) - set(hist1.keys())
    
    # Compute Jensen-Shannon divergence (symmetric)
    js_div = _compute_js_divergence(hist1, hist2)
    
    return {
        "shared_bins": len(shared_keys),
        "unique_to_baseline": len(unique_to_hist1),
        "unique_to_rfl": len(unique_to_hist2),
        "total_bins": len(all_keys),
        "js_divergence": round(js_div, 6),
    }


def _compute_js_divergence(hist1: Dict, hist2: Dict) -> float:
    """
    Compute Jensen-Shannon divergence between two histograms.
    Returns 0 for identical distributions, higher for more different.
    """
    all_keys = sorted(set(hist1.keys()) | set(hist2.keys()))
    
    if not all_keys:
        return 0.0
    
    total1 = sum(hist1.values()) or 1
    total2 = sum(hist2.values()) or 1
    
    p = [hist1.get(k, 0) / total1 for k in all_keys]
    q = [hist2.get(k, 0) / total2 for k in all_keys]
    
    # Compute mixture
    m = [(pi + qi) / 2 for pi, qi in zip(p, q)]
    
    # KL divergence components
    def kl_div(p_dist, m_dist):
        total = 0.0
        for pi, mi in zip(p_dist, m_dist):
            if pi > 0 and mi > 0:
                total += pi * math.log2(pi / mi)
        return total
    
    js = (kl_div(p, m) + kl_div(q, m)) / 2
    return js


def _compute_trend_correlation(trend1: List[float], trend2: List[float]) -> float:
    """
    Compute correlation between two trends.
    Uses Pearson correlation for comparable-length trends.
    """
    if not trend1 or not trend2:
        return 0.0
    
    # Align lengths
    min_len = min(len(trend1), len(trend2))
    t1 = trend1[:min_len]
    t2 = trend2[:min_len]
    
    if min_len < 2:
        return 0.0
    
    mean1 = sum(t1) / len(t1)
    mean2 = sum(t2) / len(t2)
    
    numerator = sum((a - mean1) * (b - mean2) for a, b in zip(t1, t2))
    
    var1 = sum((a - mean1) ** 2 for a in t1)
    var2 = sum((b - mean2) ** 2 for b in t2)
    
    denominator = math.sqrt(var1 * var2)
    
    if denominator == 0:
        return 0.0
    
    return round(numerator / denominator, 6)


def _safe_mean(values: List[float]) -> float:
    """Compute mean of values, returning 0.0 for empty lists."""
    if not values:
        return 0.0
    return round(sum(values) / len(values), 6)


def compute_cross_slice_consistency(
    slices_results_dict: Dict[str, Tuple[SliceResults, SliceResults]],
) -> Dict[str, Any]:
    """
    Compute consistency metrics across multiple slices.
    
    Args:
        slices_results_dict: Dict mapping slice_name -> (baseline_results, rfl_results)
        
    Returns:
        Dictionary containing cross-slice consistency metrics
    """
    if not slices_results_dict:
        return {
            "phase_label": "PHASE II — NOT USED IN PHASE I",
            "slice_count": 0,
            "error": "No slices provided",
        }
    
    slice_names = sorted(slices_results_dict.keys())
    
    # Compute per-slice patterns
    slice_patterns = {}
    for slice_name in slice_names:
        baseline_results, rfl_results = slices_results_dict[slice_name]
        slice_patterns[slice_name] = {
            "baseline": _extract_behavioral_pattern(baseline_results.records),
            "rfl": _extract_behavioral_pattern(rfl_results.records),
        }
    
    # Compute cross-slice metrics
    abstention_correlation_matrix = {}
    js_divergence_matrix = {}
    
    for i, name1 in enumerate(slice_names):
        for name2 in slice_names[i:]:
            key = f"{name1}__vs__{name2}"
            
            # Compare baseline abstention trends across slices
            trend1 = slice_patterns[name1]["baseline"].abstention_trend
            trend2 = slice_patterns[name2]["baseline"].abstention_trend
            
            abstention_correlation_matrix[key] = _compute_trend_correlation(trend1, trend2)
            
            # Compare metric distributions
            hist1 = slice_patterns[name1]["baseline"].metric_value_distribution
            hist2 = slice_patterns[name2]["baseline"].metric_value_distribution
            
            js_divergence_matrix[key] = round(_compute_js_divergence(hist1, hist2), 6)
    
    # Compute overall consistency score
    # Higher correlation = more consistent behavioral patterns
    correlations = list(abstention_correlation_matrix.values())
    mean_correlation = _safe_mean(correlations)
    
    divergences = list(js_divergence_matrix.values())
    mean_divergence = _safe_mean(divergences)
    
    return {
        "phase_label": "PHASE II — NOT USED IN PHASE I",
        "slice_count": len(slice_names),
        "slice_names": slice_names,
        "abstention_trend_correlation_matrix": abstention_correlation_matrix,
        "metric_js_divergence_matrix": js_divergence_matrix,
        "mean_abstention_correlation": mean_correlation,
        "mean_metric_divergence": mean_divergence,
    }


def generate_behavior_signature(slice_results: SliceResults) -> BehavioralFingerprint:
    """
    Generate a deterministic behavioral fingerprint for a slice run.
    
    This fingerprint is used for:
    1. Reproducibility verification
    2. Structural comparison across runs
    3. Detecting drift in behavioral patterns
    
    The fingerprint hash is deterministic: same inputs always produce same hash.
    
    Args:
        slice_results: Results from a single slice run
        
    Returns:
        BehavioralFingerprint with deterministic hash
    """
    records = slice_results.records
    
    # Compute all components
    metric_histogram = _compute_metric_value_distribution(records)
    chain_distribution = _compute_chain_depth_distribution(records)
    
    # Goal hit distribution (how many goals hit per cycle)
    goal_hit_dist = _compute_goal_hit_distribution(records)
    
    # Temporal smoothness: variance of abstention trend
    abstention_trend = _compute_abstention_trend(records)
    temporal_smoothness = _compute_temporal_smoothness(abstention_trend)
    
    # Build deterministic fingerprint data for hashing
    fingerprint_data = {
        "slice_name": slice_results.slice_name,
        "mode": slice_results.mode,
        "record_count": len(records),
        "metric_histogram": metric_histogram,
        "chain_distribution": chain_distribution,
        "goal_hit_distribution": goal_hit_dist,
        "temporal_smoothness": temporal_smoothness,
    }
    
    # Compute deterministic hash
    fingerprint_str = json.dumps(fingerprint_data, sort_keys=True, separators=(',', ':'))
    fingerprint_hash = hashlib.sha256(fingerprint_str.encode('utf-8')).hexdigest()
    
    return BehavioralFingerprint(
        slice_name=slice_results.slice_name,
        mode=slice_results.mode,
        metric_value_histogram=metric_histogram,
        longest_chain_distribution=chain_distribution,
        goal_hit_distribution=goal_hit_dist,
        temporal_smoothness_signature=temporal_smoothness,
        fingerprint_hash=fingerprint_hash,
    )


def _compute_goal_hit_distribution(records: List[Dict[str, Any]]) -> Dict[int, int]:
    """
    Compute distribution of goal hits per cycle.
    """
    hits = []
    for r in records:
        metric_result = r.get("metric_result", {})
        if isinstance(metric_result, dict):
            hit_count = metric_result.get("hit_count", 0)
            hits.append(int(hit_count))
        elif r.get("success"):
            hits.append(1)
        else:
            hits.append(0)
    
    if not hits:
        return {0: 1}
    
    counter = Counter(hits)
    return {k: counter[k] for k in sorted(counter.keys())}


def _compute_temporal_smoothness(trend: List[float]) -> float:
    """
    Compute temporal smoothness as inverse of variance.
    Higher value = smoother trend.
    """
    if len(trend) < 2:
        return 1.0
    
    mean_val = sum(trend) / len(trend)
    variance = sum((v - mean_val) ** 2 for v in trend) / len(trend)
    
    # Return inverse variance (with epsilon to avoid division by zero)
    smoothness = 1.0 / (variance + 1e-6)
    
    # Cap and round for determinism
    return round(min(smoothness, 1e6), 4)


def fingerprint_to_dict(fp: BehavioralFingerprint) -> Dict[str, Any]:
    """Convert a BehavioralFingerprint to a JSON-serializable dict."""
    return {
        "slice_name": fp.slice_name,
        "mode": fp.mode,
        "metric_value_histogram": fp.metric_value_histogram,
        "longest_chain_distribution": {str(k): v for k, v in fp.longest_chain_distribution.items()},
        "goal_hit_distribution": {str(k): v for k, v in fp.goal_hit_distribution.items()},
        "temporal_smoothness_signature": fp.temporal_smoothness_signature,
        "fingerprint_hash": fp.fingerprint_hash,
    }


# ===========================================================================
# PHASE II — CLI
# ===========================================================================

def discover_slices(input_dir: Path) -> List[Tuple[str, str]]:
    """
    Discover available slice/mode combinations in the input directory.
    
    Returns:
        List of (slice_name, mode) tuples
    """
    discovered = []
    
    for path in input_dir.glob("*.jsonl"):
        filename = path.stem
        
        # Try to parse uplift_u2_<slice>_<mode> format
        if filename.startswith("uplift_u2_"):
            parts = filename[len("uplift_u2_"):].rsplit("_", 1)
            if len(parts) == 2 and parts[1] in ("baseline", "rfl"):
                discovered.append((parts[0], parts[1]))
        else:
            # Try <slice>_<mode> format
            parts = filename.rsplit("_", 1)
            if len(parts) == 2 and parts[1] in ("baseline", "rfl"):
                discovered.append((parts[0], parts[1]))
    
    return sorted(set(discovered))


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="PHASE II Cross-Slice Behavioral Analytics Engine",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
ABSOLUTE SAFEGUARDS:
- No interpretation of uplift.
- No modification of slice definitions.
- No modification of success metrics.
- This code MUST NOT compute or imply Δp.

Examples:
    uv run python experiments/u2_cross_slice_analysis.py --input-dir results --out summary.json
    uv run python experiments/u2_cross_slice_analysis.py --input-dir results/uplift_u2 --slices slice1,slice2 --out out.json
        """
    )
    
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing JSONL result files",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output JSON file for the summary",
    )
    parser.add_argument(
        "--slices",
        type=str,
        default=None,
        help="Comma-separated list of slice names to analyze (auto-discovered if not specified)",
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"ERROR: Input directory does not exist: {input_dir}", file=sys.stderr)
        sys.exit(1)
    
    # Discover or parse slice list
    discovered = discover_slices(input_dir)
    
    if args.slices:
        slice_names = [s.strip() for s in args.slices.split(",")]
    else:
        # Extract unique slice names from discovered (both baseline and rfl)
        slice_names = sorted(set(s[0] for s in discovered))
    
    if not slice_names:
        print(f"ERROR: No slices found in {input_dir}", file=sys.stderr)
        sys.exit(1)
    
    print(f"PHASE II — Cross-Slice Behavioral Analytics")
    print(f"=" * 60)
    print(f"Input directory: {input_dir}")
    print(f"Slices to analyze: {slice_names}")
    print()
    
    # Load and analyze each slice
    slices_results_dict = {}
    fingerprints = {}
    comparisons = {}
    errors = []
    
    for slice_name in slice_names:
        print(f"Processing slice: {slice_name}")
        
        try:
            baseline_results = load_slice_results(slice_name, "baseline", input_dir)
            print(f"  Loaded baseline: {len(baseline_results.records)} records")
        except FileNotFoundError as e:
            print(f"  WARNING: {e}", file=sys.stderr)
            errors.append({"slice": slice_name, "mode": "baseline", "error": str(e)})
            continue
        
        try:
            rfl_results = load_slice_results(slice_name, "rfl", input_dir)
            print(f"  Loaded RFL: {len(rfl_results.records)} records")
        except FileNotFoundError as e:
            print(f"  WARNING: {e}", file=sys.stderr)
            errors.append({"slice": slice_name, "mode": "rfl", "error": str(e)})
            continue
        
        slices_results_dict[slice_name] = (baseline_results, rfl_results)
        
        # Generate fingerprints
        baseline_fp = generate_behavior_signature(baseline_results)
        rfl_fp = generate_behavior_signature(rfl_results)
        
        fingerprints[slice_name] = {
            "baseline": fingerprint_to_dict(baseline_fp),
            "rfl": fingerprint_to_dict(rfl_fp),
        }
        
        # Compare patterns (structural only)
        comparison = compare_behavioral_patterns(baseline_results, rfl_results)
        comparisons[slice_name] = comparison
        
        print(f"  Baseline fingerprint: {baseline_fp.fingerprint_hash[:16]}...")
        print(f"  RFL fingerprint: {rfl_fp.fingerprint_hash[:16]}...")
    
    print()
    
    # Compute cross-slice consistency
    if len(slices_results_dict) > 1:
        cross_slice_consistency = compute_cross_slice_consistency(slices_results_dict)
        print(f"Cross-slice consistency computed for {cross_slice_consistency['slice_count']} slices")
    else:
        cross_slice_consistency = {
            "phase_label": "PHASE II — NOT USED IN PHASE I",
            "slice_count": len(slices_results_dict),
            "note": "Cross-slice consistency requires >= 2 slices",
        }
    
    # Build output summary
    summary = {
        "phase_label": "PHASE II — NOT USED IN PHASE I",
        "input_dir": str(input_dir),
        "slice_count": len(slices_results_dict),
        "slices_analyzed": sorted(slices_results_dict.keys()),
        "fingerprints": fingerprints,
        "behavioral_comparisons": comparisons,
        "cross_slice_consistency": cross_slice_consistency,
        "errors": errors if errors else None,
    }
    
    # Write output
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    
    print(f"Summary written to: {out_path}")
    print()
    
    # Exit with error if no slices were successfully processed
    if not slices_results_dict:
        print("ERROR: No slices were successfully processed", file=sys.stderr)
        sys.exit(1)
    
    print("✓ Cross-slice analysis complete")
    sys.exit(0)


if __name__ == "__main__":
    main()

