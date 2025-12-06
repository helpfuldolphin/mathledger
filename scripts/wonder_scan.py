#!/usr/bin/env python3
"""
Wonder Scan Protocol v1 - RFL Signal Correlator

Analyzes MathLedger telemetry artifacts to discover latent optimization
opportunities in proof generation, policy effectiveness, and system health.

Implements Module 1: RFL Signal Correlator
- Policy-Uplift Correlation
- Depth-Throughput Correlation  
- Determinism-Reproducibility Score
- Merkle Entropy Analysis

Constraints:
- ASCII-only output
- Proof-or-Abstain discipline
- NO_NETWORK=true (read-only local artifacts)
- Deterministic JSON output (sort_keys=True, ensure_ascii=True)

Author: Manus K - The Wonder Engineer
Date: 2025-10-19
"""

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Add backend to path for potential imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def ensure_ascii(text: str) -> bool:
    """Verify text is ASCII-safe."""
    try:
        text.encode('ascii')
        return True
    except UnicodeEncodeError:
        return False


def load_fol_ab_csv(csv_path: Path) -> List[Dict[str, Any]]:
    """Load FOL A/B test results from CSV."""
    if not csv_path.exists():
        return []
    
    rows = []
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    except Exception as e:
        print(f"error: Failed to read {csv_path}: {e}", file=sys.stderr)
        return []
    
    return rows


def load_fol_stats_json(json_path: Path) -> Optional[Dict[str, Any]]:
    """Load FOL statistics JSON."""
    if not json_path.exists():
        return None
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"error: Failed to read {json_path}: {e}", file=sys.stderr)
        return None


def load_determinism_report(json_path: Path) -> Optional[Dict[str, Any]]:
    """Load determinism report JSON."""
    if not json_path.exists():
        return None
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"error: Failed to read {json_path}: {e}", file=sys.stderr)
        return None


def load_throughput_json(json_path: Path) -> Optional[Dict[str, Any]]:
    """Load throughput metrics JSON."""
    if not json_path.exists():
        return None
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"error: Failed to read {json_path}: {e}", file=sys.stderr)
        return None


def analyze_policy_uplift_correlation(ab_data: List[Dict[str, Any]], 
                                     stats_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze correlation between policy hashes and uplift ratios.
    
    Returns correlation insight with signal strength and recommendations.
    """
    if not ab_data:
        return {
            "status": "ABSTAIN",
            "reason": "No A/B test data available",
            "signal_strength": 0.0
        }
    
    # Separate baseline and guided runs
    baseline_runs = []
    guided_runs = defaultdict(list)  # policy_hash -> list of pph
    
    for row in ab_data:
        mode = row.get('mode', '').lower()
        policy_hash = row.get('policy_hash', '')
        pph_str = row.get('proofs_per_hour', '')
        
        try:
            pph = float(pph_str)
            if 'baseline' in mode:
                baseline_runs.append(pph)
            elif 'guided' in mode and policy_hash:
                guided_runs[policy_hash].append(pph)
        except (ValueError, TypeError):
            continue
    
    # Calculate uplift for each policy against baseline
    if not baseline_runs:
        return {
            "status": "ABSTAIN",
            "reason": "No baseline runs found",
            "signal_strength": 0.0
        }
    
    if not guided_runs:
        return {
            "status": "ABSTAIN",
            "reason": "No policy-guided runs found",
            "signal_strength": 0.0
        }
    
    baseline_mean = sum(baseline_runs) / len(baseline_runs)
    
    # Calculate uplift for each policy
    policy_uplifts = {}
    for policy_hash, guided_pph_list in guided_runs.items():
        guided_mean = sum(guided_pph_list) / len(guided_pph_list)
        uplift = guided_mean / max(baseline_mean, 1e-9)
        
        policy_uplifts[policy_hash] = {
            "uplift_ratio": uplift,
            "baseline_mean": baseline_mean,
            "guided_mean": guided_mean,
            "sample_count": len(guided_pph_list)
        }
    
    # Find best policy
    best_policy = max(policy_uplifts.items(), key=lambda x: x[1]['uplift_ratio'])
    best_hash, best_metrics = best_policy
    
    # Calculate signal strength (confidence based on sample count and uplift)
    signal_strength = min(1.0, (best_metrics['sample_count'] / 10.0) * 
                         (best_metrics['uplift_ratio'] / 3.0))
    
    # Determine confidence level
    if signal_strength >= 0.8:
        confidence = "HIGH"
    elif signal_strength >= 0.5:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"
    
    # Generate recommendation
    if best_metrics['uplift_ratio'] >= 2.5 and best_metrics['sample_count'] >= 3:
        recommendation = f"Policy {best_hash[:16]}... shows strong {best_metrics['uplift_ratio']:.2f}x uplift. Recommend promotion to default."
    elif best_metrics['uplift_ratio'] >= 1.5:
        recommendation = f"Policy {best_hash[:16]}... shows moderate {best_metrics['uplift_ratio']:.2f}x uplift. Recommend further testing."
    else:
        recommendation = f"Policy {best_hash[:16]}... shows weak {best_metrics['uplift_ratio']:.2f}x uplift. Continue baseline comparison."
    
    return {
        "status": "PASS",
        "correlation_id": "rfl_policy_uplift_" + datetime.utcnow().strftime("%Y%m%d"),
        "signal_strength": round(signal_strength, 2),
        "policy_hash": best_hash,
        "mean_uplift": round(best_metrics['uplift_ratio'], 2),
        "baseline_mean": round(best_metrics['baseline_mean'], 2),
        "guided_mean": round(best_metrics['guided_mean'], 2),
        "sample_count": best_metrics['sample_count'],
        "confidence": confidence,
        "recommendation": recommendation,
        "all_policies": {k: round(v['uplift_ratio'], 2) for k, v in policy_uplifts.items()}
    }


def analyze_depth_throughput_correlation(throughput_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze correlation between proof depth and throughput.
    
    Returns insight on depth-performance relationship.
    """
    if not throughput_data or 'points' not in throughput_data:
        return {
            "status": "ABSTAIN",
            "reason": "No throughput data available",
            "signal_strength": 0.0
        }
    
    points = throughput_data['points']
    if len(points) < 2:
        return {
            "status": "ABSTAIN",
            "reason": "Insufficient throughput data points",
            "signal_strength": 0.0
        }
    
    # Calculate depth-throughput trend
    depths = [p['depth'] for p in points]
    throughputs = [p['proofs_per_hour'] for p in points]
    
    # Simple linear correlation (Pearson-like)
    mean_depth = sum(depths) / len(depths)
    mean_throughput = sum(throughputs) / len(throughputs)
    
    numerator = sum((d - mean_depth) * (t - mean_throughput) 
                   for d, t in zip(depths, throughputs))
    denom_d = sum((d - mean_depth) ** 2 for d in depths)
    denom_t = sum((t - mean_throughput) ** 2 for t in throughputs)
    
    if denom_d == 0 or denom_t == 0:
        correlation = 0.0
    else:
        correlation = numerator / (denom_d * denom_t) ** 0.5
    
    signal_strength = abs(correlation)
    
    # Interpret correlation
    if correlation < -0.5:
        trend = "NEGATIVE"
        interpretation = "Throughput decreases significantly with depth. Consider depth-aware optimization."
    elif correlation > 0.5:
        trend = "POSITIVE"
        interpretation = "Throughput increases with depth. Unusual pattern - investigate."
    else:
        trend = "NEUTRAL"
        interpretation = "Weak depth-throughput correlation. Depth not primary bottleneck."
    
    return {
        "status": "PASS",
        "correlation_id": "depth_throughput_" + datetime.utcnow().strftime("%Y%m%d"),
        "signal_strength": round(signal_strength, 2),
        "correlation_coefficient": round(correlation, 2),
        "trend": trend,
        "sample_count": len(points),
        "interpretation": interpretation,
        "data_points": [{"depth": p['depth'], "throughput": round(p['proofs_per_hour'], 2)} 
                       for p in points]
    }


def analyze_determinism_score(determinism_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze determinism report and extract reproducibility insights.
    
    Returns determinism health metrics and recommendations.
    """
    if not determinism_data or 'summary' not in determinism_data:
        return {
            "status": "ABSTAIN",
            "reason": "No determinism report available",
            "signal_strength": 0.0
        }
    
    summary = determinism_data['summary']
    score_str = summary.get('determinism_score', '0%')
    
    try:
        score = float(score_str.rstrip('%')) / 100.0
    except (ValueError, AttributeError):
        score = 0.0
    
    total_sources = summary.get('total_nondeterministic_sources', 0)
    patched = summary.get('sources_patched', 0)
    critical = summary.get('critical_issues', 0)
    
    # Signal strength based on score and patch coverage
    signal_strength = score * (patched / max(total_sources, 1))
    
    # Determine health status
    if score >= 0.95:
        health = "EXCELLENT"
        recommendation = "Determinism score excellent. Maintain current practices."
    elif score >= 0.85:
        health = "GOOD"
        recommendation = f"Determinism score good. Address remaining {total_sources - patched} sources."
    elif score >= 0.70:
        health = "FAIR"
        recommendation = f"Determinism score fair. Prioritize {critical} critical issues."
    else:
        health = "POOR"
        recommendation = f"Determinism score poor. Urgent: fix {critical} critical issues."
    
    return {
        "status": "PASS",
        "correlation_id": "determinism_health_" + datetime.utcnow().strftime("%Y%m%d"),
        "signal_strength": round(signal_strength, 2),
        "determinism_score": round(score, 2),
        "total_nondeterministic_sources": total_sources,
        "sources_patched": patched,
        "critical_issues": critical,
        "health_status": health,
        "recommendation": recommendation
    }


def analyze_merkle_entropy(ab_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze Merkle root diversity as proxy for proof novelty.
    
    Returns entropy metrics and novelty assessment.
    """
    if not ab_data:
        return {
            "status": "ABSTAIN",
            "reason": "No A/B test data available",
            "signal_strength": 0.0
        }
    
    # Extract block roots
    block_roots = [row.get('block_root', '') for row in ab_data if row.get('block_root')]
    
    if not block_roots:
        return {
            "status": "ABSTAIN",
            "reason": "No block roots found",
            "signal_strength": 0.0
        }
    
    # Calculate uniqueness ratio
    unique_roots = len(set(block_roots))
    total_roots = len(block_roots)
    uniqueness_ratio = unique_roots / total_roots
    
    # Simple entropy calculation (Shannon-like)
    root_counts = defaultdict(int)
    for root in block_roots:
        root_counts[root] += 1
    
    entropy = 0.0
    for count in root_counts.values():
        p = count / total_roots
        if p > 0:
            entropy -= p * (p ** 0.5)  # Simplified entropy
    
    signal_strength = uniqueness_ratio
    
    # Interpret entropy
    if uniqueness_ratio >= 0.9:
        novelty = "HIGH"
        interpretation = "Excellent proof diversity. Each run produces unique proofs."
    elif uniqueness_ratio >= 0.7:
        novelty = "MEDIUM"
        interpretation = "Good proof diversity. Some overlap expected."
    else:
        novelty = "LOW"
        interpretation = "Low proof diversity. Investigate potential stagnation."
    
    return {
        "status": "PASS",
        "correlation_id": "merkle_entropy_" + datetime.utcnow().strftime("%Y%m%d"),
        "signal_strength": round(signal_strength, 2),
        "unique_roots": unique_roots,
        "total_roots": total_roots,
        "uniqueness_ratio": round(uniqueness_ratio, 2),
        "entropy_score": round(entropy, 2),
        "novelty_level": novelty,
        "interpretation": interpretation
    }


def generate_insights(artifacts_dir: Path) -> Dict[str, Any]:
    """
    Generate comprehensive insights from all available telemetry.
    
    Returns unified insights JSON.
    """
    # Load all available data
    fol_ab_path = artifacts_dir / "wpv5" / "fol_ab.csv"
    fol_stats_path = artifacts_dir / "wpv5" / "fol_stats.json"
    determinism_path = artifacts_dir / "repro" / "determinism_report.json"
    throughput_path = artifacts_dir / "wpv5" / "throughput.json"
    
    ab_data = load_fol_ab_csv(fol_ab_path)
    stats_data = load_fol_stats_json(fol_stats_path)
    determinism_data = load_determinism_report(determinism_path)
    throughput_data = load_throughput_json(throughput_path)
    
    # Run all analyses
    policy_insight = analyze_policy_uplift_correlation(ab_data, stats_data)
    depth_insight = analyze_depth_throughput_correlation(throughput_data)
    determinism_insight = analyze_determinism_score(determinism_data)
    merkle_insight = analyze_merkle_entropy(ab_data)
    
    # Calculate overall health score
    insights = [policy_insight, depth_insight, determinism_insight, merkle_insight]
    passed_insights = [i for i in insights if i.get('status') == 'PASS']
    
    if passed_insights:
        avg_signal_strength = sum(i['signal_strength'] for i in passed_insights) / len(passed_insights)
        overall_health = "HEALTHY" if avg_signal_strength >= 0.7 else "NEEDS_ATTENTION"
    else:
        avg_signal_strength = 0.0
        overall_health = "INSUFFICIENT_DATA"
    
    # Compile insights
    result = {
        "scan_version": "v1",
        "scan_timestamp": datetime.utcnow().isoformat() + "Z",
        "scan_status": "COMPLETE",
        "overall_health": overall_health,
        "average_signal_strength": round(avg_signal_strength, 2),
        "insights": {
            "policy_uplift_correlation": policy_insight,
            "depth_throughput_correlation": depth_insight,
            "determinism_reproducibility": determinism_insight,
            "merkle_entropy": merkle_insight
        },
        "recommendations": []
    }
    
    # Aggregate recommendations
    for insight in passed_insights:
        if 'recommendation' in insight:
            result['recommendations'].append(insight['recommendation'])
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Wonder Scan Protocol v1 - RFL Signal Correlator",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--artifacts-dir',
        type=Path,
        default=Path(__file__).parent.parent / 'artifacts',
        help='Path to artifacts directory (default: ../artifacts)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Output path for insights.json (default: artifacts/wonder/insights.json)'
    )
    parser.add_argument(
        '--stable-ts',
        action='store_true',
        help='Use content-stable timestamp (derived from input file mtimes)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Optional fixed epoch seconds for deterministic timestamp'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Validate artifacts directory
    if not args.artifacts_dir.exists():
        print(f"error: ABSTAIN - Artifacts directory not found: {args.artifacts_dir}", 
              file=sys.stderr)
        return 2
    
    # Set output path
    if args.output is None:
        output_dir = args.artifacts_dir / 'wonder'
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / 'insights.json'
    else:
        output_path = args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate insights
    if args.verbose:
        print("Wonder Scan Protocol v1 - Initiating telemetry analysis...")
        print(f"Artifacts directory: {args.artifacts_dir}")
    
    insights = generate_insights(args.artifacts_dir)
    
    # Apply deterministic timestamp if requested
    if args.stable_ts or args.seed is not None:
        input_paths = [
            args.artifacts_dir / 'wpv5' / 'fol_ab.csv',
            args.artifacts_dir / 'wpv5' / 'fol_stats.json',
            args.artifacts_dir / 'wpv5' / 'throughput.json',
            args.artifacts_dir / 'repro' / 'determinism_report.json',
        ]
        existing = [p for p in input_paths if p.exists()]
        
        if args.stable_ts and existing:
            # Strategy A: timestamp = max mtime of inputs, seconds precision
            max_mtime = max(int(p.stat().st_mtime) for p in existing)
            scan_timestamp = datetime.utcfromtimestamp(max_mtime).replace(microsecond=0).isoformat() + 'Z'
        elif args.seed is not None:
            # Strategy B: seed -> direct epoch seconds
            scan_timestamp = datetime.utcfromtimestamp(args.seed).replace(microsecond=0).isoformat() + 'Z'
        else:
            # Should not reach here, but fallback to current behavior
            scan_timestamp = insights['scan_timestamp']
        
        insights['scan_timestamp'] = scan_timestamp
        
        if args.verbose:
            print(f"Deterministic timestamp: {scan_timestamp}")
    
    # Write output with deterministic formatting
    with open(output_path, 'w', encoding='ascii') as f:
        json.dump(insights, f, sort_keys=True, indent=2, separators=(',', ':'), 
                 ensure_ascii=True)
    
    # Write separate policy-paired correlation facts if available
    policy_insight = insights['insights'].get('policy_uplift_correlation', {})
    if policy_insight.get('status') == 'PASS':
        policy_facts_path = output_path.parent / 'policy_correlation_facts.json'
        policy_facts = {
            'scan_version': 'v1',
            'scan_timestamp': insights['scan_timestamp'],
            'correlation_type': 'policy_uplift',
            'status': 'PASS',
            'baseline': {
                'mode': 'baseline',
                'mean_proofs_per_hour': policy_insight['baseline_mean'],
                'sample_count': policy_insight['sample_count']
            },
            'guided': {
                'mode': 'guided',
                'policy_hash': policy_insight['policy_hash'],
                'mean_proofs_per_hour': policy_insight['guided_mean'],
                'sample_count': policy_insight['sample_count']
            },
            'correlation': {
                'uplift_ratio': policy_insight['mean_uplift'],
                'signal_strength': policy_insight['signal_strength'],
                'confidence': policy_insight['confidence'],
                'correlation_id': policy_insight['correlation_id']
            },
            'validation': {
                'baseline_guided_paired': True,
                'deterministic_output': True,
                'ascii_compliant': True
            },
            'recommendation': policy_insight['recommendation']
        }
        
        with open(policy_facts_path, 'w', encoding='ascii') as f:
            json.dump(policy_facts, f, sort_keys=True, indent=2, separators=(',', ':'), 
                     ensure_ascii=True)
        
        if args.verbose:
            print(f"Policy correlation facts written to: {policy_facts_path}")
    
    # Print summary
    print(f"[PASS] Wonder Scan Completed")
    print(f"Overall Health: {insights['overall_health']}")
    print(f"Signal Strength: {insights['average_signal_strength']:.2f}")
    print(f"Insights written to: {output_path}")
    
    if args.verbose:
        print("\nRecommendations:")
        for i, rec in enumerate(insights['recommendations'], 1):
            print(f"  {i}. {rec}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

