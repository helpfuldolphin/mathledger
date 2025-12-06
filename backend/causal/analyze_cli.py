#!/usr/bin/env python3
"""
Causal analysis CLI for MathLedger RFL - Enhanced with policy memos.

Extracts run history from database, builds causal graph,
estimates coefficients, and generates policy recommendation memos.

Usage:
    python -m backend.causal.analyze_cli --config config/causal/default.json
    python -m backend.causal.analyze_cli --system pl --generate-memo
"""

import argparse
import sys
import os
import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.causal.graph import build_rfl_graph, validate_graph
from backend.causal.variables import RunMetrics, extract_run_deltas, summary_statistics, stratify_by_policy_change
from backend.causal.estimator import estimate_all_edges, EstimationMethod, compute_stability
from backend.causal.report import (
    generate_causal_report,
    print_causal_summary,
    format_pass_summary,
    generate_markdown_summary,
    export_for_graphviz
)
from backend.causal.policy_memo import (
    generate_policy_memo,
    create_recommendation_from_analysis
)
from backend.causal.export import export_estimates_json

_GLOBAL_SEED = 0


def load_config(config_path: str) -> Dict:
    """Load configuration from JSON file."""
    if not Path(config_path).exists():
        print(f"Warning: Config file {config_path} not found, using defaults")
        return {}

    with open(config_path, 'r') as f:
        return json.load(f)


def fetch_runs_from_db(
    db_url: str,
    system: Optional[str] = None,
    min_runs: int = 2
) -> List[RunMetrics]:
    """
    Fetch run history from database.

    Args:
        db_url: PostgreSQL connection URL
        system: Filter by system (e.g., 'pl', 'fol')
        min_runs: Minimum number of runs required

    Returns:
        List of RunMetrics ordered by started_at
    """
    import psycopg2
    from psycopg2.extras import RealDictCursor

    try:
        conn = psycopg2.connect(db_url)
        cur = conn.cursor(cursor_factory=RealDictCursor)

        # Check if runs table has required columns
        cur.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'runs'
        """)
        columns = {row['column_name'] for row in cur.fetchall()}

        required = {'started_at', 'ended_at', 'system', 'slice'}
        optional = {'policy_hash', 'abstain_pct', 'proofs_success',
                   'proofs_per_sec', 'depth_max_reached'}

        if not required.issubset(columns):
            missing = required - columns
            print(f"Warning: runs table missing columns: {missing}")
            return []

        # Build query dynamically based on available columns
        select_cols = ['id', 'started_at', 'ended_at', 'system', 'slice']
        for col in optional:
            if col in columns:
                select_cols.append(col)

        query = f"""
            SELECT {', '.join(select_cols)}
            FROM runs
            WHERE ended_at IS NOT NULL
        """

        params = []
        if system:
            query += " AND system = %s"
            params.append(system)

        query += " ORDER BY started_at ASC"

        cur.execute(query, params)
        rows = cur.fetchall()

        if len(rows) < min_runs:
            print(f"Warning: Only {len(rows)} runs found, minimum {min_runs} required")
            return []

        # Convert to RunMetrics
        runs = []
        for row in rows:
            runs.append(RunMetrics(
                run_id=row['id'],
                started_at=row['started_at'],
                ended_at=row['ended_at'],
                policy_hash=row.get('policy_hash'),
                abstain_pct=row.get('abstain_pct', 0.0),
                proofs_success=row.get('proofs_success', 0),
                proofs_per_sec=row.get('proofs_per_sec', 0.0),
                depth_max_reached=row.get('depth_max_reached', 0),
                system=row['system'],
                slice_name=row['slice']
            ))

        cur.close()
        conn.close()

        return runs

    except Exception as e:
        print(f"Error fetching runs from database: {e}", file=sys.stderr)
        return []


def build_data_dict(run_deltas: List) -> Dict[str, List[float]]:
    """
    Convert run deltas to data dictionary for estimation.

    Args:
        run_deltas: List of RunDelta objects

    Returns:
        Dictionary mapping variable names to value lists
    """
    data = {
        'policy_hash': [],
        'abstain_pct': [],
        'proofs_per_sec': [],
        'verify_ms_p50': [],  # Placeholder
        'depth_max': []
    }

    for delta in run_deltas:
        # Use comparison run values (the "after" state)
        run = delta.comparison_run

        # Policy as numeric hash
        policy_val = hash(run.policy_hash) % 1000 if run.policy_hash else 0
        data['policy_hash'].append(float(policy_val))

        data['abstain_pct'].append(run.abstain_pct)
        data['proofs_per_sec'].append(run.proofs_per_sec)
        data['verify_ms_p50'].append(50.0)  # Placeholder
        data['depth_max'].append(float(run.depth_max_reached))

    return data


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Causal analysis for MathLedger RFL with policy memos'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config/causal/default.json',
        help='Path to configuration JSON file'
    )

    parser.add_argument(
        '--db-url',
        type=str,
        default=None,
        help='PostgreSQL database URL (defaults to DATABASE_URL env var)'
    )

    parser.add_argument(
        '--system',
        type=str,
        default=None,
        help='Filter runs by system (pl, fol, etc.)'
    )

    parser.add_argument(
        '--generate-memo',
        action='store_true',
        help='Generate policy recommendation memo'
    )

    parser.add_argument(
        '--current-policy',
        type=str,
        default='baseline',
        help='Current policy identifier'
    )

    parser.add_argument(
        '--recommended-policy',
        type=str,
        default='candidate',
        help='Recommended policy identifier'
    )

    args = parser.parse_args()

    if args.db_url is None:
        from backend.security.runtime_env import get_required_env

        args.db_url = get_required_env("DATABASE_URL")

    # Load configuration
    config = load_config(args.config)

    # Extract config parameters
    min_runs = config.get('data_requirements', {}).get('min_runs', 30)
    bootstrap = config.get('estimation', {}).get('bootstrap_replicates', 10000)
    method_str = config.get('estimation', {}).get('method', 'ols')

    method_map = {'ols': EstimationMethod.OLS, 'matching': EstimationMethod.MATCHING, 'dml': EstimationMethod.DML}
    method = method_map.get(method_str, EstimationMethod.OLS)

    # Output paths
    output_paths = config.get('output_paths', {})
    causal_summary_path = output_paths.get('causal_summary', 'reports/causal_summary.md')
    estimates_json_path = output_paths.get('estimates_json', 'artifacts/causal/estimates.json')
    graphviz_path = output_paths.get('graphviz_dot', 'artifacts/causal/causal_graph.dot')

    print("=" * 60)
    print("CAUSAL ARCHITECT — RFL Policy Analysis")
    print("=" * 60)
    print()

    # Step 1: Fetch runs from database
    print(f"Fetching runs from database (system={args.system or 'all'})...")
    runs = fetch_runs_from_db(args.db_url, args.system, min_runs=2)  # Fetch all, validate later

    if not runs:
        print("[ABSTAIN] No run data available", file=sys.stderr)
        sys.exit(1)

    print(f"✓ Loaded {len(runs)} runs")

    # Check minimum requirement
    if len(runs) < min_runs:
        print(f"⚠ Warning: Only {len(runs)} runs (minimum {min_runs} recommended)")
    print()

    # Step 2: Extract deltas
    print("Extracting run deltas...")
    run_deltas = extract_run_deltas(runs)

    if len(run_deltas) < 1:
        print("[ABSTAIN] Need at least 2 runs to compute deltas", file=sys.stderr)
        sys.exit(1)

    print(f"✓ Computed {len(run_deltas)} consecutive run deltas")

    # Check policy changes
    policy_changed, _ = stratify_by_policy_change(run_deltas)
    print(f"  - Policy changes: {len(policy_changed)}")
    print(f"  - Baseline comparisons: {len(run_deltas) - len(policy_changed)}")
    print()

    # Step 3: Build causal graph
    print("Building RFL causal graph...")
    causal_graph = build_rfl_graph()

    # Validate structure
    warnings = validate_graph(causal_graph)
    if warnings:
        print("⚠ Graph validation warnings:")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("✓ Graph structure valid")
    print()

    # Step 4: Prepare data for estimation
    print("Preparing data for estimation...")
    data = build_data_dict(run_deltas)

    n_samples = len(next(iter(data.values())))
    print(f"✓ Data prepared: {n_samples} samples")
    print()

    # Step 5: Estimate causal coefficients
    print(f"Estimating causal coefficients (method={method_str}, bootstrap={bootstrap})...")
    try:
        coefficients = estimate_all_edges(
            causal_graph,
            data,
            method=method,
            seed=_GLOBAL_SEED
        )
        print(f"✓ Estimated {len(coefficients)} edge coefficients")
    except Exception as e:
        print(f"[ABSTAIN] Coefficient estimation failed: {e}", file=sys.stderr)
        sys.exit(1)
    print()

    # Step 6: Stability testing
    print(f"Testing causal stability (bootstrap={bootstrap})...")
    try:
        stability = compute_stability(
            causal_graph,
            data,
            n_bootstrap=bootstrap,
            seed=_GLOBAL_SEED
        )
        print(f"✓ Stability tested for {len(stability)} edges")
    except Exception as e:
        print(f"Warning: Stability testing failed: {e}")
        stability = {}
    print()

    # Step 7: Generate reports
    print("Generating causal analysis reports...")

    # Summary markdown
    md = generate_markdown_summary(causal_graph, coefficients, run_deltas)
    Path(causal_summary_path).parent.mkdir(parents=True, exist_ok=True)
    with open(causal_summary_path, 'w') as f:
        f.write(md)
    print(f"✓ Saved summary to {causal_summary_path}")

    # Graphviz export
    export_for_graphviz(causal_graph, coefficients, graphviz_path)
    print(f"✓ Saved graph to {graphviz_path}")

    # Estimates JSON (for Codex M handoff)
    export_estimates_json(coefficients, stability, run_deltas, config, estimates_json_path)
    print(f"✓ Saved estimates to {estimates_json_path}")
    print()

    # Step 8: Policy Recommendation Memo (if requested)
    if args.generate_memo:
        print("Generating policy recommendation memo...")

        recommendation = create_recommendation_from_analysis(
            current_policy=args.current_policy,
            recommended_policy=args.recommended_policy,
            causal_estimates={k: v.to_dict() for k, v in coefficients.items()},
            run_deltas=run_deltas,
            stability_results=stability
        )

        date_str = datetime.now().strftime("%Y-%m-%d")
        memo_path = output_paths.get(
            'policy_memo_template',
            'reports/policy_recommendation_{date}.md'
        ).replace('{date}', date_str)

        memo_text = generate_policy_memo(
            recommendation,
            {k: v.to_dict() for k, v in coefficients.items()},
            output_path=memo_path
        )

        print(f"✓ Saved policy memo to {memo_path}")
        print()
        print("RECOMMENDATION SUMMARY:")
        print(f"  Current: {recommendation.current_policy}")
        print(f"  Recommended: {recommendation.recommended_policy}")
        print(f"  Expected uplift: {recommendation.expected_uplift:.2f}x")
        print(f"  Evidence: {recommendation.evidence_strength.upper()}")
        print()

    # Step 9: Print summary to console
    print_causal_summary(causal_graph, coefficients, run_deltas)
    print()

    # Print pass messages
    print("CAUSAL MODEL STATUS")
    print("=" * 60)
    for msg in format_pass_summary(coefficients):
        print(msg)
    print()

    # Summary statistics
    stats = summary_statistics(run_deltas)
    if stats:
        print("EMPIRICAL SUMMARY")
        print("=" * 60)
        print(f"Abstention: μ={stats['abstain']['mean']:.2f}pp, "
              f"σ={stats['abstain']['std']:.2f}pp")
        print(f"Throughput: μ={stats['throughput']['mean']:.3f} proofs/s, "
              f"σ={stats['throughput']['std']:.3f}")
        print(f"Proofs: μ={stats['proof']['mean']:.1f}, "
              f"σ={stats['proof']['std']:.1f}")
        print(f"N={stats['n']} deltas")
        print()

    # Final seal
    n_significant = sum(1 for c in coefficients.values() if c.is_significant)

    print("=" * 60)
    if n_significant >= 2 and len(run_deltas) >= min_runs:
        print(f"[PASS] Causal Model Stable — p<0.05 paths>={n_significant}")
    elif len(run_deltas) < min_runs:
        print(f"[ABSTAIN] Insufficient runs — n={len(run_deltas)}<{min_runs}")
    else:
        print(f"[ABSTAIN] Insufficient significant paths — {n_significant}<2")
    print("=" * 60)


if __name__ == '__main__':
    main()
