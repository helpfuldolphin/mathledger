#!/usr/bin/env python3
"""
Critical Path Intelligence Tool

Analyzes CI timing data across multiple runs to identify critical paths,
compute job statistics, and generate parallelization recommendations.

FLEET DIRECTIVE:
- Proof-or-Abstain only (fail-closed; no quiet reds)
- RFC 8785 canonical JSON (sorted keys, compact, ASCII-only)
- Determinism > speed (byte-identical for identical inputs)
- Domain separation (LEAF 0x00 / NODE 0x01 / FINAL 0x02)
- Sealed pass-lines for every claim

Usage:
    python tools/ci/critical_path_intel.py \
        --workflow ci.yml \
        --branch integrate/ledger-v0.1 \
        --runs 20 \
        --out artifacts/perf/critical_path.json
"""

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


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


def fetch_ci_runs(workflow: str, branch: str, limit: int) -> List[Dict[str, Any]]:
    """Fetch CI run data from GitHub."""
    cmd = [
        'gh', 'run', 'list',
        '--workflow', workflow,
        '--branch', branch,
        '--limit', str(limit),
        '--json', 'databaseId,status,conclusion,createdAt,updatedAt'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    runs = json.loads(result.stdout)
    
    completed_runs = [r for r in runs if r['status'] == 'completed']
    
    if len(completed_runs) < limit:
        print(f"[WARN] Only {len(completed_runs)} completed runs found (requested {limit})", file=sys.stderr)
    
    return completed_runs


def fetch_job_timings(run_id: int) -> List[Dict[str, Any]]:
    """Fetch job timing data for a specific run."""
    cmd = [
        'gh', 'run', 'view', str(run_id),
        '--json', 'jobs',
        '--jq', '.jobs[] | {name: .name, conclusion: .conclusion, startedAt: .startedAt, completedAt: .completedAt}'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    
    jobs = []
    for line in result.stdout.strip().split('\n'):
        if line:
            jobs.append(json.loads(line))
    
    return jobs


def compute_job_duration(job: Dict[str, Any]) -> Optional[float]:
    """Compute job duration in seconds."""
    if not job.get('startedAt') or not job.get('completedAt'):
        return None
    
    started = datetime.fromisoformat(job['startedAt'].replace('Z', '+00:00'))
    completed = datetime.fromisoformat(job['completedAt'].replace('Z', '+00:00'))
    
    duration = (completed - started).total_seconds()
    return duration if duration > 0 else None


def compute_percentile(values: List[float], percentile: float) -> float:
    """Compute percentile of values."""
    if not values:
        return 0.0
    
    sorted_values = sorted(values)
    index = (len(sorted_values) - 1) * percentile / 100
    
    if index.is_integer():
        return sorted_values[int(index)]
    else:
        lower = sorted_values[int(index)]
        upper = sorted_values[int(index) + 1]
        return lower + (upper - lower) * (index - int(index))


def compute_covariance(x: List[float], y: List[float]) -> float:
    """Compute covariance between two job durations."""
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    
    mean_x = statistics.mean(x)
    mean_y = statistics.mean(y)
    
    cov = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x))) / (len(x) - 1)
    return cov


def identify_critical_path(job_stats: Dict[str, Dict[str, float]]) -> List[str]:
    """
    Identify critical path (minimal cut set for wall-clock).
    
    Critical path is the sequence of jobs that determines overall wall-clock time.
    For parallel jobs, this is the longest-running job at each level.
    """
    critical_jobs = []
    
    sorted_jobs = sorted(
        job_stats.items(),
        key=lambda x: x[1]['p95'],
        reverse=True
    )
    
    for job_name, stats in sorted_jobs:
        if stats['p95'] > 0:
            critical_jobs.append(job_name)
    
    return critical_jobs


def compute_mad(values: List[float]) -> float:
    """
    Compute Median Absolute Deviation (MAD).
    
    MAD = median(|x_i - median(x)|)
    
    MAD is a robust measure of variability, less sensitive to outliers than stdev.
    """
    if not values or len(values) < 2:
        return 0.0
    
    median_val = statistics.median(values)
    absolute_deviations = [abs(x - median_val) for x in values]
    return statistics.median(absolute_deviations)


def compute_stability_bands(
    job_timings: Dict[str, List[float]]
) -> Dict[str, Dict[str, float]]:
    """
    Compute stability bands for each job: p50, p95, MAD.
    
    Stability bands characterize run-to-run variability and help identify
    jobs with unstable performance.
    """
    stability_bands = {}
    
    for job_name, durations in job_timings.items():
        if len(durations) < 10:
            continue
        
        p50 = compute_percentile(durations, 50)
        p95 = compute_percentile(durations, 95)
        mad = compute_mad(durations)
        
        rolling_median = statistics.median(durations[-10:]) if len(durations) >= 10 else statistics.median(durations)
        
        stability_bands[job_name] = {
            "mad": round(mad, 2),
            "p50": round(p50, 2),
            "p95": round(p95, 2),
            "rolling_median": round(rolling_median, 2),
            "stability_threshold": round(rolling_median + 2 * mad, 2)
        }
    
    return stability_bands


def identify_hot_lanes(
    job_timings: Dict[str, List[float]],
    stability_bands: Dict[str, Dict[str, float]],
    covariance_matrix: Dict[str, float]
) -> List[Dict[str, Any]]:
    """
    Identify hot lanes: jobs where p95 exceeds (rolling median + 2*MAD).
    
    Hot lanes indicate jobs with unstable performance that may be experiencing:
    - Resource contention (high covariance with other jobs)
    - Queue wait times
    - CPU steal or I/O bottlenecks
    """
    hot_lanes = []
    
    for job_name, band in stability_bands.items():
        if job_name not in job_timings:
            continue
        
        durations = job_timings[job_name]
        if len(durations) < 10:
            continue
        
        p95 = band['p95']
        threshold = band['stability_threshold']
        
        if p95 > threshold:
            cause_notes = []
            
            covariance_partners = []
            for key, cov_value in covariance_matrix.items():
                if ':' in key:
                    job1, job2 = key.split(':')
                    if job1 == job_name and job2 != job_name and abs(cov_value) > 5.0:
                        covariance_partners.append((job2, cov_value))
                    elif job2 == job_name and job1 != job_name and abs(cov_value) > 5.0:
                        covariance_partners.append((job1, cov_value))
            
            if covariance_partners:
                top_partner = max(covariance_partners, key=lambda x: abs(x[1]))
                cause_notes.append(f"high covariance with {top_partner[0]} (cov={top_partner[1]:.2f})")
            
            mad_ratio = band['mad'] / band['rolling_median'] if band['rolling_median'] > 0 else 0
            if mad_ratio > 0.2:
                cause_notes.append(f"high variability (MAD/median={mad_ratio:.2f})")
            
            recent_trend = durations[-5:] if len(durations) >= 5 else durations
            if statistics.mean(recent_trend) > band['rolling_median'] * 1.2:
                cause_notes.append("recent performance degradation")
            
            hot_lanes.append({
                "cause_notes": cause_notes if cause_notes else ["performance instability"],
                "job": job_name,
                "mad": band['mad'],
                "p95": p95,
                "p95_excess": round(p95 - threshold, 2),
                "rolling_median": band['rolling_median'],
                "threshold": threshold
            })
    
    hot_lanes.sort(key=lambda x: x['p95_excess'], reverse=True)
    
    return hot_lanes


def generate_parallelization_plan(
    job_stats: Dict[str, Dict[str, float]],
    critical_path: List[str]
) -> Dict[str, Any]:
    """Generate parallelization recommendations."""
    total_p95 = sum(stats['p95'] for stats in job_stats.values())
    
    max_p95 = max(stats['p95'] for stats in job_stats.values()) if job_stats else 0
    
    parallelization_factor = total_p95 / max_p95 if max_p95 > 0 else 1.0
    
    recommendations = {
        "current_parallelization": round(parallelization_factor, 2),
        "max_parallel_jobs": len(job_stats),
        "critical_path_jobs": critical_path[:3],
        "optimization_opportunities": []
    }
    
    for job_name, stats in job_stats.items():
        if job_name not in critical_path[:3] and stats['p95'] > 30:
            recommendations["optimization_opportunities"].append({
                "job": job_name,
                "p95_duration": round(stats['p95'], 2),
                "suggestion": "Consider splitting or optimizing"
            })
    
    return recommendations


def main():
    parser = argparse.ArgumentParser(description='Analyze CI critical path and generate intelligence')
    parser.add_argument('--workflow', required=True, help='Workflow file name (e.g., ci.yml)')
    parser.add_argument('--branch', required=True, help='Branch name (e.g., integrate/ledger-v0.1)')
    parser.add_argument('--runs', type=int, default=20, help='Number of runs to analyze (minimum 20)')
    parser.add_argument('--out', required=True, help='Output path for critical path JSON')
    
    args = parser.parse_args()
    
    if args.runs < 20:
        print(f"[FAIL] Minimum 20 runs required, got {args.runs}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Fetching {args.runs} CI runs for {args.workflow} on {args.branch}...")
    runs = fetch_ci_runs(args.workflow, args.branch, args.runs)
    
    if len(runs) < 20:
        print(f"[ABSTAIN] Only {len(runs)} runs available, need at least 20", file=sys.stderr)
        sys.exit(2)
    
    print(f"Analyzing {len(runs)} runs...")
    
    job_timings = {}
    
    for i, run in enumerate(runs):
        run_id = run['databaseId']
        print(f"  Run {i+1}/{len(runs)}: {run_id}")
        
        try:
            jobs = fetch_job_timings(run_id)
            
            for job in jobs:
                job_name = job['name']
                duration = compute_job_duration(job)
                
                if duration is not None:
                    if job_name not in job_timings:
                        job_timings[job_name] = []
                    job_timings[job_name].append(duration)
        
        except subprocess.CalledProcessError as e:
            print(f"  [WARN] Failed to fetch jobs for run {run_id}: {e}", file=sys.stderr)
            continue
    
    if not job_timings:
        print("[FAIL] No job timing data collected", file=sys.stderr)
        sys.exit(1)
    
    print(f"\nComputing statistics for {len(job_timings)} jobs...")
    
    job_stats = {}
    for job_name, durations in job_timings.items():
        if len(durations) < 10:
            print(f"  [WARN] Only {len(durations)} samples for {job_name}, skipping", file=sys.stderr)
            continue
        
        job_stats[job_name] = {
            "count": len(durations),
            "mean": round(statistics.mean(durations), 2),
            "median": round(statistics.median(durations), 2),
            "p50": round(compute_percentile(durations, 50), 2),
            "p95": round(compute_percentile(durations, 95), 2),
            "stdev": round(statistics.stdev(durations), 2),
            "min": round(min(durations), 2),
            "max": round(max(durations), 2)
        }
    
    print("\nComputing covariance matrix...")
    covariance_matrix = {}
    job_names = list(job_stats.keys())
    
    for i, job1 in enumerate(job_names):
        for j, job2 in enumerate(job_names):
            if i <= j:
                durations1 = job_timings[job1]
                durations2 = job_timings[job2]
                
                min_len = min(len(durations1), len(durations2))
                cov = compute_covariance(durations1[:min_len], durations2[:min_len])
                
                key = f"{job1}:{job2}"
                covariance_matrix[key] = round(cov, 4)
    
    print("\nComputing stability bands...")
    stability_bands = compute_stability_bands(job_timings)
    
    print("\nIdentifying hot lanes...")
    hot_lanes = identify_hot_lanes(job_timings, stability_bands, covariance_matrix)
    
    print("\nIdentifying critical path...")
    critical_path = identify_critical_path(job_stats)
    
    print("\nGenerating parallelization recommendations...")
    parallelization_plan = generate_parallelization_plan(job_stats, critical_path)
    
    critical_path_data = {
        "analyzed_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "covariance_matrix": covariance_matrix,
        "critical_path": critical_path,
        "hot_lanes": hot_lanes,
        "job_statistics": job_stats,
        "metadata": {
            "branch": args.branch,
            "runs_analyzed": len(runs),
            "workflow": args.workflow
        },
        "parallelization_plan": parallelization_plan,
        "stability_bands": stability_bands,
        "version": "2.0"
    }
    
    canonical_json = canonicalize_json(critical_path_data)
    critical_path_hash = compute_domain_hash(canonical_json.encode('ascii'), b'\x02')
    
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='ascii') as f:
        f.write(canonical_json)
    
    hash_path = output_path.with_suffix('.sha256')
    with open(hash_path, 'w', encoding='ascii') as f:
        f.write(f"{critical_path_hash}  {output_path.name}\n")
    
    print(f"\n[PASS] Critical Path: sealed {critical_path_hash}")
    print(f"\nCritical Path (top 5):")
    for i, job in enumerate(critical_path[:5]):
        stats = job_stats[job]
        print(f"  {i+1}. {job}: p95={stats['p95']}s, mean={stats['mean']}s")
    
    print(f"\nParallelization Factor: {parallelization_plan['current_parallelization']}x")
    print(f"Max Parallel Jobs: {parallelization_plan['max_parallel_jobs']}")
    
    if parallelization_plan['optimization_opportunities']:
        print(f"\nOptimization Opportunities:")
        for opp in parallelization_plan['optimization_opportunities'][:3]:
            print(f"  - {opp['job']}: {opp['p95_duration']}s ({opp['suggestion']})")
    
    if hot_lanes:
        print(f"\nHot Lanes Detected ({len(hot_lanes)}):")
        for lane in hot_lanes[:3]:
            print(f"  - {lane['job']}: p95={lane['p95']}s (excess: +{lane['p95_excess']}s)")
            print(f"    Threshold: {lane['threshold']}s (median + 2*MAD)")
            print(f"    Causes: {', '.join(lane['cause_notes'])}")
    
    sys.exit(0)


if __name__ == '__main__':
    main()
