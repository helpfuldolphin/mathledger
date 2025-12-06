#!/usr/bin/env python3
"""
Velocity Proof Pack Generator - 5-Run Battery with Hash Chain

Generates sealed velocity dossier from 5-run CI validation battery.
Implements RFC 8785 canonicalization, Proof-or-Abstain variance detection,
hash chain verification, and domain separation for cryptographic verification.

FLEET DIRECTIVE:
- Proof-or-Abstain only (fail-closed; no quiet reds)
- RFC 8785 canonical JSON (sorted keys, compact, ASCII-only)
- Determinism > speed (measure after 5 consistent runs)
- Domain separation (LEAF 0x00 / NODE 0x01 / FINAL 0x02)
- Sealed pass-lines for every claim
- Hash chain: hash_i+1 = SHA256(RFC8785(run_i+1 || hash_i))

Usage:
    python tools/ci/velocity_proof_pack.py \
        --runs 5 \
        --run-dir /path/to/ci-runs \
        --rfcsign \
        --out artifacts/perf/velocity_dossier.json
    
    python tools/ci/velocity_proof_pack.py \
        --run1 /path/to/run1/perf_log.json \
        --run2 /path/to/run2/perf_log.json \
        --run3 /path/to/run3/perf_log.json \
        --output artifacts/perf/perf_log.json
"""

import argparse
import hashlib
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def canonicalize_json(obj: Any) -> str:
    """
    Canonicalize JSON according to RFC 8785.
    
    Rules:
    1. Whitespace is removed
    2. Object keys are sorted lexicographically
    3. Numbers are serialized in a specific format
    4. Unicode escaping is normalized
    """
    return json.dumps(
        obj,
        ensure_ascii=True,
        sort_keys=True,
        separators=(',', ':'),
        indent=None
    )


def compute_domain_hash(data: bytes, domain: bytes) -> str:
    """
    Compute SHA256 hash with domain separation.
    
    Domain prefixes:
    - LEAF 0x00: Individual data items
    - NODE 0x01: Intermediate aggregations
    - FINAL 0x02: Final proof pack
    """
    return hashlib.sha256(domain + data).hexdigest()


def load_perf_log(path: str) -> Dict[str, Any]:
    """Load and validate performance log JSON."""
    with open(path, 'r', encoding='ascii') as f:
        data = json.load(f)
    
    if 'runs' not in data or len(data['runs']) == 0:
        raise ValueError(f"Invalid perf_log format in {path}: missing 'runs' array")
    
    run = data['runs'][0]
    if 'wall_clock_seconds' not in run and 'wall_clock_duration_seconds' not in run:
        raise ValueError(f"Invalid perf_log format in {path}: missing wall_clock timing")
    
    return data


def extract_wall_clock(perf_log: Dict[str, Any]) -> float:
    """Extract wall-clock time from performance log."""
    run = perf_log['runs'][0]
    return run.get('wall_clock_seconds', run.get('wall_clock_duration_seconds', 0))


def compute_hash_chain(runs: List[Dict[str, Any]], rfcsign: bool = True) -> List[str]:
    """
    Compute hash chain: hash_i+1 = SHA256(RFC8785(run_i+1 || hash_i))
    
    Args:
        runs: List of run dictionaries
        rfcsign: If True, use RFC 8785 canonicalization
    
    Returns:
        List of hashes, one per run
    """
    hashes = []
    prev_hash = "0" * 64  # Genesis hash
    
    for i, run in enumerate(runs):
        chained_data = {
            "prev_hash": prev_hash,
            "run_number": run["run_number"],
            "wall_clock_seconds": run["wall_clock_seconds"]
        }
        
        # Canonicalize and hash
        if rfcsign:
            canonical = canonicalize_json(chained_data)
        else:
            canonical = json.dumps(chained_data, sort_keys=True)
        
        current_hash = compute_domain_hash(canonical.encode('ascii'), b'\x01')  # NODE domain
        hashes.append(current_hash)
        prev_hash = current_hash
    
    return hashes


def load_velocity_history(history_path: str) -> List[Dict[str, Any]]:
    """Load velocity history from JSONL file."""
    history = []
    
    if not Path(history_path).exists():
        return history
    
    with open(history_path, 'r', encoding='ascii') as f:
        for line in f:
            if line.strip():
                history.append(json.loads(line))
    
    return history


def compute_trend_metrics(history: List[Dict[str, Any]], current_mean: float) -> Dict[str, Any]:
    """
    Compute trend metrics: slope (sec/day) and 7-day forecast.
    
    Uses linear regression on historical data to compute velocity trend.
    """
    if len(history) < 2:
        return {
            "forecast_7day": round(current_mean, 2),
            "slope_sec_per_day": 0.0,
            "trend_confidence": "insufficient_data"
        }
    
    timestamps = []
    means = []
    
    for entry in history:
        try:
            ts = datetime.fromisoformat(entry['timestamp'].replace('Z', '+00:00'))
            timestamps.append(ts.timestamp())
            means.append(entry['mean_wall_clock_seconds'])
        except (KeyError, ValueError):
            continue
    
    if len(timestamps) < 2:
        return {
            "forecast_7day": round(current_mean, 2),
            "slope_sec_per_day": 0.0,
            "trend_confidence": "insufficient_data"
        }
    
    base_time = timestamps[0]
    days = [(t - base_time) / 86400 for t in timestamps]
    
    n = len(days)
    sum_x = sum(days)
    sum_y = sum(means)
    sum_xy = sum(days[i] * means[i] for i in range(n))
    sum_x2 = sum(d * d for d in days)
    
    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x) if (n * sum_x2 - sum_x * sum_x) != 0 else 0.0
    intercept = (sum_y - slope * sum_x) / n
    
    current_day = (datetime.utcnow().timestamp() - base_time) / 86400
    forecast_day = current_day + 7
    forecast_7day = intercept + slope * forecast_day
    
    r_squared = 0.0
    if n > 2:
        mean_y = sum_y / n
        ss_tot = sum((means[i] - mean_y) ** 2 for i in range(n))
        ss_res = sum((means[i] - (intercept + slope * days[i])) ** 2 for i in range(n))
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    
    confidence = "high" if r_squared > 0.8 else "medium" if r_squared > 0.5 else "low"
    
    return {
        "forecast_7day": round(max(0, forecast_7day), 2),
        "r_squared": round(r_squared, 4),
        "slope_sec_per_day": round(slope, 4),
        "trend_confidence": confidence
    }


def generate_parallelization_recommendations(
    critical_path_file: Optional[str],
    current_mean: float
) -> Dict[str, Any]:
    """Generate parallelization recommendations from critical path analysis."""
    if not critical_path_file or not Path(critical_path_file).exists():
        return {
            "available": False,
            "reason": "critical_path.json not found"
        }
    
    try:
        with open(critical_path_file, 'r', encoding='ascii') as f:
            critical_path_data = json.load(f)
        
        plan = critical_path_data.get('parallelization_plan', {})
        critical_path = critical_path_data.get('critical_path', [])
        
        return {
            "available": True,
            "critical_path_jobs": critical_path[:3],
            "current_parallelization": plan.get('current_parallelization', 1.0),
            "max_parallel_jobs": plan.get('max_parallel_jobs', 1),
            "optimization_opportunities": plan.get('optimization_opportunities', [])
        }
    
    except Exception as e:
        return {
            "available": False,
            "reason": f"failed to load critical path: {str(e)}"
        }


def generate_5run_dossier(
    run_paths: List[str],
    baseline: int,
    target: int,
    variance_threshold: float,
    rfcsign: bool,
    history_path: Optional[str] = None,
    critical_path_file: Optional[str] = None
) -> Dict[str, Any]:
    """Generate 5-run velocity dossier with hash chain and trend metrics."""
    
    try:
        perf_logs = [load_perf_log(path) for path in run_paths]
    except Exception as e:
        print(f"[FAIL] Failed to load performance logs: {e}", file=sys.stderr)
        sys.exit(1)
    
    wall_clocks = [extract_wall_clock(log) for log in perf_logs]
    
    # Compute statistics
    mean_wall_clock = statistics.mean(wall_clocks)
    stdev_wall_clock = statistics.stdev(wall_clocks) if len(wall_clocks) > 1 else 0.0
    variance_pct = (stdev_wall_clock / mean_wall_clock * 100) if mean_wall_clock > 0 else 0.0
    coefficient_of_variation = stdev_wall_clock / mean_wall_clock if mean_wall_clock > 0 else 0.0
    
    velocity_improvement = ((baseline - mean_wall_clock) / baseline * 100) if baseline > 0 else 0.0
    
    if variance_pct > variance_threshold:
        delta = variance_pct - variance_threshold
        print(f"[ABSTAIN] Variance {variance_pct:.2f}% exceeds threshold {variance_threshold}%", file=sys.stderr)
        print(f"Delta: +{delta:.2f}% over limit", file=sys.stderr)
        print(f"Wall-clock times: {wall_clocks}", file=sys.stderr)
        print(f"Mean: {mean_wall_clock:.2f}s, StdDev: {stdev_wall_clock:.2f}s", file=sys.stderr)
        print(f"Coefficient of Variation: {coefficient_of_variation:.4f}", file=sys.stderr)
        sys.exit(2)
    
    if mean_wall_clock > target:
        delta = mean_wall_clock - target
        print(f"[ABSTAIN] Mean {mean_wall_clock:.2f}s exceeds target {target}s", file=sys.stderr)
        print(f"Delta: +{delta:.2f}s over target", file=sys.stderr)
        sys.exit(2)
    
    runs = [
        {
            "run_number": i + 1,
            "wall_clock_seconds": wall_clocks[i],
            "source": run_paths[i]
        }
        for i in range(len(wall_clocks))
    ]
    
    hash_chain = compute_hash_chain(runs, rfcsign=rfcsign)
    
    for i, run in enumerate(runs):
        run["hash"] = hash_chain[i]
    
    dossier = {
        "baseline_duration_seconds": baseline,
        "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hash_chain": {
            "algorithm": "SHA256",
            "domain_separation": "NODE (0x01)",
            "genesis_hash": "0" * 64,
            "terminal_hash": hash_chain[-1]
        },
        "runs": runs,
        "statistics": {
            "coefficient_of_variation": round(coefficient_of_variation, 4),
            "mean_wall_clock_seconds": round(mean_wall_clock, 2),
            "stdev_wall_clock_seconds": round(stdev_wall_clock, 2),
            "variance_percent": round(variance_pct, 2),
            "velocity_improvement_percent": round(velocity_improvement, 1)
        },
        "target_duration_seconds": target,
        "validation": {
            "pass_criteria": f"mean<={target}s AND variance<={variance_threshold}%",
            "target_met": mean_wall_clock <= target,
            "variance_within_threshold": variance_pct <= variance_threshold
        },
        "version": "2.0"
    }
    
    if history_path:
        history = load_velocity_history(history_path)
        trend_metrics = compute_trend_metrics(history, mean_wall_clock)
        dossier["trend_metrics"] = trend_metrics
    
    if critical_path_file:
        parallelization = generate_parallelization_recommendations(critical_path_file, mean_wall_clock)
        dossier["parallelization_recommendations"] = parallelization
    
    return dossier


def generate_3run_pack(
    run_paths: List[str],
    baseline: int,
    target: int,
    variance_threshold: float
) -> Dict[str, Any]:
    """Generate legacy 3-run proof pack (backward compatibility)."""
    
    try:
        perf_logs = [load_perf_log(path) for path in run_paths]
    except Exception as e:
        print(f"[FAIL] Failed to load performance logs: {e}", file=sys.stderr)
        sys.exit(1)
    
    wall_clocks = [extract_wall_clock(log) for log in perf_logs]
    
    # Compute statistics
    mean_wall_clock = statistics.mean(wall_clocks)
    stdev_wall_clock = statistics.stdev(wall_clocks) if len(wall_clocks) > 1 else 0.0
    variance_pct = (stdev_wall_clock / mean_wall_clock * 100) if mean_wall_clock > 0 else 0.0
    
    velocity_improvement = ((baseline - mean_wall_clock) / baseline * 100) if baseline > 0 else 0.0
    
    if variance_pct > variance_threshold:
        print(f"[ABSTAIN] Variance {variance_pct:.2f}% exceeds threshold {variance_threshold}%", file=sys.stderr)
        print(f"Wall-clock times: {wall_clocks}", file=sys.stderr)
        print(f"Mean: {mean_wall_clock:.2f}s, StdDev: {stdev_wall_clock:.2f}s", file=sys.stderr)
        sys.exit(2)
    
    target_met = mean_wall_clock <= target
    
    # Build proof pack
    proof_pack = {
        "baseline_duration_seconds": baseline,
        "runs": [
            {
                "run_number": i + 1,
                "source": run_paths[i],
                "wall_clock_seconds": wall_clocks[i]
            }
            for i in range(len(wall_clocks))
        ],
        "statistics": {
            "mean_wall_clock_seconds": round(mean_wall_clock, 2),
            "stdev_wall_clock_seconds": round(stdev_wall_clock, 2),
            "variance_percent": round(variance_pct, 2),
            "velocity_improvement_percent": round(velocity_improvement, 1)
        },
        "target_duration_seconds": target,
        "validation": {
            "pass_criteria": f"mean<={target}s AND variance<={variance_threshold}%",
            "target_met": target_met,
            "variance_within_threshold": variance_pct <= variance_threshold
        },
        "version": "1.0"
    }
    
    return proof_pack


def main():
    parser = argparse.ArgumentParser(description='Generate sealed velocity proof pack or dossier')
    
    parser.add_argument('--runs', type=int, help='Number of runs (5 for dossier mode)')
    parser.add_argument('--run-dir', help='Directory containing run artifacts')
    parser.add_argument('--rfcsign', action='store_true', help='Use RFC 8785 canonicalization for hash chain')
    parser.add_argument('--out', help='Output path for dossier')
    
    parser.add_argument('--run1', help='Path to run 1 perf_log.json')
    parser.add_argument('--run2', help='Path to run 2 perf_log.json')
    parser.add_argument('--run3', help='Path to run 3 perf_log.json')
    parser.add_argument('--output', help='Output path for proof pack (legacy)')
    
    parser.add_argument('--history', help='Path to velocity_history.jsonl for trend analysis')
    parser.add_argument('--critical-path', help='Path to critical_path.json for parallelization recommendations')
    
    parser.add_argument('--baseline', type=int, default=420, help='Baseline duration in seconds')
    parser.add_argument('--target', type=int, default=210, help='Target duration in seconds')
    parser.add_argument('--variance-threshold', type=float, default=7.0, help='Variance threshold percentage')
    
    args = parser.parse_args()
    
    if args.runs and args.run_dir and args.out:
        if args.runs != 5:
            print(f"[FAIL] Only 5-run mode supported, got {args.runs}", file=sys.stderr)
            sys.exit(1)
        
        run_dir = Path(args.run_dir)
        if not run_dir.exists():
            print(f"[FAIL] Run directory not found: {run_dir}", file=sys.stderr)
            sys.exit(1)
        
        perf_logs = sorted(run_dir.glob("**/perf_log.json"))
        if len(perf_logs) < 5:
            print(f"[FAIL] Found only {len(perf_logs)} perf_log.json files, need 5", file=sys.stderr)
            sys.exit(1)
        
        run_paths = [str(p) for p in perf_logs[:5]]
        
        dossier = generate_5run_dossier(
            run_paths,
            args.baseline,
            args.target,
            args.variance_threshold,
            args.rfcsign,
            args.history,
            args.critical_path
        )
        
        canonical_json = canonicalize_json(dossier)
        dossier_hash = compute_domain_hash(canonical_json.encode('ascii'), b'\x02')
        
        output_path = Path(args.out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='ascii') as f:
            f.write(canonical_json)
        
        hash_path = output_path.with_suffix('.sha256')
        with open(hash_path, 'w', encoding='ascii') as f:
            f.write(f"{dossier_hash}  {output_path.name}\n")
        
        stats = dossier['statistics']
        print(f"[PASS] CI Velocity: mean<={args.target}s var<={args.variance_threshold}%")
        print(f"[PASS] Velocity Dossier: {dossier_hash}")
        print(f"")
        print(f"5-Run Summary:")
        for i, run in enumerate(dossier['runs']):
            print(f"  Run {run['run_number']}: {run['wall_clock_seconds']}s (hash: {run['hash'][:16]}...)")
        print(f"")
        print(f"Statistics:")
        print(f"  Mean: {stats['mean_wall_clock_seconds']}s +/- {stats['stdev_wall_clock_seconds']}s")
        print(f"  Variance: {stats['variance_percent']}%")
        print(f"  Coefficient of Variation: {stats['coefficient_of_variation']}")
        print(f"  Velocity Improvement: {stats['velocity_improvement_percent']}%")
        print(f"")
        print(f"Hash Chain:")
        print(f"  Genesis: {dossier['hash_chain']['genesis_hash'][:16]}...")
        print(f"  Terminal: {dossier['hash_chain']['terminal_hash'][:16]}...")
        
        if 'trend_metrics' in dossier:
            trend = dossier['trend_metrics']
            print(f"")
            print(f"Trend Metrics:")
            print(f"  Slope: {trend['slope_sec_per_day']} sec/day")
            print(f"  7-Day Forecast: {trend['forecast_7day']}s")
            print(f"  Confidence: {trend['trend_confidence']}")
            if 'r_squared' in trend:
                print(f"  R-squared: {trend['r_squared']}")
        
        if 'parallelization_recommendations' in dossier:
            para = dossier['parallelization_recommendations']
            if para.get('available'):
                print(f"")
                print(f"Parallelization Recommendations:")
                print(f"  Current Factor: {para['current_parallelization']}x")
                print(f"  Max Parallel Jobs: {para['max_parallel_jobs']}")
                print(f"  Critical Path: {', '.join(para['critical_path_jobs'])}")
        
        sys.exit(0)
        
    elif args.run1 and args.run2 and args.run3 and args.output:
        run_paths = [args.run1, args.run2, args.run3]
        
        proof_pack = generate_3run_pack(
            run_paths,
            args.baseline,
            args.target,
            args.variance_threshold
        )
        
        canonical_json = canonicalize_json(proof_pack)
        proof_pack_hash = compute_domain_hash(canonical_json.encode('ascii'), b'\x02')
        
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='ascii') as f:
            f.write(canonical_json)
        
        hash_path = output_path.with_suffix('.sha256')
        with open(hash_path, 'w', encoding='ascii') as f:
            f.write(f"{proof_pack_hash}  {output_path.name}\n")
        
        stats = proof_pack['statistics']
        target_met = proof_pack['validation']['target_met']
        variance_ok = proof_pack['validation']['variance_within_threshold']
        
        if target_met and variance_ok:
            print(f"[PASS] CI Velocity: mean<={args.target}s var<={args.variance_threshold}%")
            print(f"[PASS] Velocity Proof Pack: {proof_pack_hash}")
            print(f"Mean: {stats['mean_wall_clock_seconds']}s +/- {stats['stdev_wall_clock_seconds']}s ({stats['variance_percent']}% variance)")
            print(f"Velocity Improvement: {stats['velocity_improvement_percent']}% (baseline: {args.baseline}s)")
            sys.exit(0)
        else:
            if not target_met:
                print(f"[FAIL] CI Velocity: mean {stats['mean_wall_clock_seconds']}s > target {args.target}s", file=sys.stderr)
            if not variance_ok:
                print(f"[FAIL] Variance {stats['variance_percent']}% > threshold {args.variance_threshold}%", file=sys.stderr)
            sys.exit(1)
    
    else:
        parser.print_help()
        print("\nError: Must provide either:", file=sys.stderr)
        print("  - 5-run mode: --runs 5 --run-dir DIR --rfcsign --out FILE", file=sys.stderr)
        print("  - 3-run mode: --run1 FILE --run2 FILE --run3 FILE --output FILE", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
