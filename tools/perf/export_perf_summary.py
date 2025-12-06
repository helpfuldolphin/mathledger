#!/usr/bin/env python3
"""
Export performance summary JSON for CI integration.

Reads the latest benchmark results and exports a standardized
perf_summary.json file for CI tracking and regression detection.

Usage:
    python tools/perf/export_perf_summary.py
    python tools/perf/export_perf_summary.py --input artifacts/perf/20251031_205713/bench.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any


def find_latest_bench_json() -> Path:
    """Find the most recent bench.json file."""
    perf_dir = Path("artifacts/perf")
    if not perf_dir.exists():
        raise FileNotFoundError("artifacts/perf directory not found")
    
    bench_files = list(perf_dir.glob("*/bench.json"))
    if not bench_files:
        raise FileNotFoundError("No bench.json files found in artifacts/perf")
    
    bench_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return bench_files[0]


def calculate_speedup(baseline_ms: float, current_ms: float) -> float:
    """Calculate speedup factor."""
    if current_ms == 0:
        return 0.0
    return baseline_ms / current_ms


def export_perf_summary(bench_json_path: Path, output_path: Path = None) -> Dict[str, Any]:
    """
    Export performance summary from benchmark results.
    
    Args:
        bench_json_path: Path to bench.json file
        output_path: Optional output path for perf_summary.json
        
    Returns:
        Performance summary dictionary
    """
    if output_path is None:
        output_path = Path("artifacts/perf/perf_summary.json")
    
    with open(bench_json_path) as f:
        bench_data = json.load(f)
    
    benchmarks = bench_data.get("benchmarks", {})
    
    mp_results = benchmarks.get("modus_ponens", {})
    canon_results = benchmarks.get("canonicalization", {})
    cache_results = benchmarks.get("cache", {})
    
    BASELINE_MP_5K = 926.49  # ms
    BASELINE_MP_10K = 1913.90  # ms
    BASELINE_CANON_10K = 168.34  # ms
    
    mp_5k = mp_results.get("mp_5000", {})
    mp_10k = mp_results.get("mp_10000", {})
    canon_10k = canon_results.get("canon_10000", {})
    
    current_mp_5k = mp_5k.get("wall_time_ms", 0)
    current_mp_10k = mp_10k.get("wall_time_ms", 0)
    current_canon_10k = canon_10k.get("wall_time_ms", 0)
    
    speedup_mp_5k = calculate_speedup(BASELINE_MP_5K, current_mp_5k)
    speedup_mp_10k = calculate_speedup(BASELINE_MP_10K, current_mp_10k)
    speedup_canon_10k = calculate_speedup(BASELINE_CANON_10K, current_canon_10k)
    
    baseline_total = BASELINE_MP_5K + BASELINE_CANON_10K
    current_total = current_mp_5k + current_canon_10k
    overall_speedup = calculate_speedup(baseline_total, current_total)
    
    cache_info = cache_results.get("cache_effectiveness", {})
    cache_hit_rate = cache_info.get("hit_rate", 0)
    
    summary = {
        "timestamp": bench_data.get("timestamp"),
        "source_file": str(bench_json_path),
        "overall_speedup": round(overall_speedup, 2),
        "target_speedup": 3.0,
        "speedup_achieved": overall_speedup >= 3.0,
        "metrics": {
            "modus_ponens_5k": {
                "baseline_ms": BASELINE_MP_5K,
                "current_ms": round(current_mp_5k, 2),
                "speedup": round(speedup_mp_5k, 2),
                "improvement_pct": round((speedup_mp_5k - 1) * 100, 1)
            },
            "modus_ponens_10k": {
                "baseline_ms": BASELINE_MP_10K,
                "current_ms": round(current_mp_10k, 2),
                "speedup": round(speedup_mp_10k, 2),
                "improvement_pct": round((speedup_mp_10k - 1) * 100, 1)
            },
            "canonicalization_10k": {
                "baseline_ms": BASELINE_CANON_10K,
                "current_ms": round(current_canon_10k, 2),
                "speedup": round(speedup_canon_10k, 2),
                "improvement_pct": round((speedup_canon_10k - 1) * 100, 1)
            }
        },
        "cache": {
            "hit_rate": round(cache_hit_rate, 4),
            "hit_rate_pct": round(cache_hit_rate * 100, 2)
        },
        "ci_status": "PASS" if overall_speedup >= 3.0 else "FAIL",
        "ci_message": f"[{'PASS' if overall_speedup >= 3.0 else 'FAIL'}] PERF Uplift {overall_speedup:.2f}×"
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canonical = json.dumps(
        summary,
        sort_keys=True,
        separators=(',', ':'),
        ensure_ascii=True,
    )
    with open(output_path, "w", encoding="ascii") as f:
        f.write(canonical)
    
    print(f"Performance summary exported to: {output_path}")
    print(f"\n{summary['ci_message']}")
    print(f"Overall speedup: {overall_speedup:.2f}× (target: 3.0×)")
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Export performance summary JSON for CI integration"
    )
    parser.add_argument(
        "--input",
        help="Path to bench.json file (default: latest in artifacts/perf)"
    )
    parser.add_argument(
        "--output",
        default="artifacts/perf/perf_summary.json",
        help="Output path for perf_summary.json"
    )
    
    args = parser.parse_args()
    
    try:
        if args.input:
            bench_json_path = Path(args.input)
        else:
            bench_json_path = find_latest_bench_json()
            print(f"Using latest benchmark: {bench_json_path}")
        
        if not bench_json_path.exists():
            print(f"ERROR: Benchmark file not found: {bench_json_path}", file=sys.stderr)
            sys.exit(1)
        
        summary = export_perf_summary(bench_json_path, Path(args.output))
        
        if summary["speedup_achieved"]:
            sys.exit(0)  # Success
        else:
            sys.exit(1)  # Failure - target not met
    
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
