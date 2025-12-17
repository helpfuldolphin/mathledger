#!/usr/bin/env python3
"""
PHASE II — NOT USED IN PHASE I

U2 Snapshot Inspector CLI
==========================

Debug tool for inspecting snapshot planning decisions.

Usage:
    python scripts/u2_snapshot_inspect.py --snapshot-root /path/to/snapshots
    python scripts/u2_snapshot_inspect.py --snapshot-root /path/to/snapshots --json

Example: Sanity-Checking Auto-Resume Before P5 Deployment
-----------------------------------------------------------

Before trusting auto-resume in a P5 (production) setting, a human operator should
verify that snapshot planning produces sensible recommendations:

    # Step 1: Run inspector and save JSON output
    uv run python scripts/u2_snapshot_inspect.py \
        --snapshot-root results/u2_snapshots \
        --json > snapshot_plan.json
    
    # Step 2: Extract and examine runbook summary
    python -c "
    import json
    from experiments.u2.snapshot_history import (
        build_multi_run_snapshot_history,
        plan_future_runs,
        build_snapshot_runbook_summary,
    )
    
    with open('snapshot_plan.json') as f:
        data = json.load(f)
    
    # Rebuild runbook summary
    multi_history = data['multi_history']
    plan = data['plan']
    runbook = build_snapshot_runbook_summary(multi_history, plan)
    
    print('=== Snapshot Runbook Summary ===')
    print(f'Status: {runbook[\"status\"]}')
    print(f'Runs Analyzed: {runbook[\"runs_analyzed\"]}')
    print(f'Mean Coverage: {runbook[\"mean_coverage_pct\"]:.1f}%')
    print(f'Max Gap: {runbook[\"max_gap\"]} cycles')
    print(f'Has Resume Targets: {runbook[\"has_resume_targets\"]}')
    if runbook['preferred_run_id']:
        print(f'Preferred Run: {runbook[\"preferred_run_id\"]}')
        print(f'Preferred Snapshot: {runbook[\"preferred_snapshot_path\"]}')
    print(f'Reason: {runbook[\"reason\"]}')
    "
    
    # Step 3: Verify decision makes sense
    # - If RESUME: Check that preferred_snapshot_path exists and is recent
    # - If NEW_RUN: Verify reason explains why (empty root, no viable targets, etc.)
    # - Check mean_coverage_pct is reasonable for checkpointing policy
    # - Check max_gap is acceptable (not too large relative to total cycles)
    
    # Step 4: Cross-reference with evidence pack manifest
    # - Compare runbook in manifest['operations']['auto_resume'] with local inspection
    # - Verify runs_analyzed matches number of run directories
    # - Verify reason is consistent with observed snapshot state

Key Principle: In SHADOW MODE, snapshot planning is advisory only. The runbook
summary explains what the planner would do, but does not enforce any behavior.
Before trusting auto-resume in P5, operators should verify that the planning
logic produces sensible recommendations.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.u2.snapshot_history import (
    build_multi_run_snapshot_history,
    plan_future_runs,
    summarize_snapshot_plans_for_u2_orchestrator,
    summarize_snapshot_plans_for_global_console,
    build_calibration_experiment_runbook,
    compare_multi_run_snapshots,
    classify_calibration_trend,
)
from experiments.u2.cal_exp1_reconciliation import reconcile_cal_exp1_runs


def inspect_snapshots(
    snapshot_root: Path,
    json_output: bool = False,
    calibration_experiment: Optional[str] = None,
    compare: bool = False,
    previous_history_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Inspect snapshot planning decisions.
    
    Args:
        snapshot_root: Root directory containing run directories
        json_output: If True, return full JSON; if False, return summary dict
        
    Returns:
        Inspection results dict
    """
    if not snapshot_root.exists():
        error_msg = f"ERROR: Snapshot root not found: {snapshot_root}"
        if json_output:
            return {
                "error": error_msg,
                "status": "ERROR",
                "snapshot_root": str(snapshot_root),
            }
        else:
            print(error_msg, file=sys.stderr)
            sys.exit(1)
    
    if not snapshot_root.is_dir():
        error_msg = f"ERROR: Not a directory: {snapshot_root}"
        if json_output:
            return {
                "error": error_msg,
                "status": "ERROR",
                "snapshot_root": str(snapshot_root),
            }
        else:
            print(error_msg, file=sys.stderr)
            sys.exit(1)
    
    # Discover run directories
    run_dirs: list[str] = []
    try:
        for item in snapshot_root.iterdir():
            if item.is_dir():
                run_dirs.append(str(item))
    except (OSError, PermissionError) as e:
        error_msg = f"ERROR: Failed to scan snapshot root: {e}"
        if json_output:
            return {
                "error": error_msg,
                "status": "ERROR",
                "snapshot_root": str(snapshot_root),
            }
        else:
            print(error_msg, file=sys.stderr)
            sys.exit(1)
    
    if not run_dirs:
        if json_output:
            return {
                "status": "NO_DATA",
                "snapshot_root": str(snapshot_root),
                "runs_analyzed": 0,
                "message": "No run directories found",
            }
        else:
            print(f"INFO: No run directories found in {snapshot_root}")
            print("      This is normal for a fresh snapshot root.")
            return {}
    
    # Build multi-run history
    try:
        multi_history = build_multi_run_snapshot_history(run_dirs)
    except Exception as e:
        error_msg = f"ERROR: Failed to build snapshot history: {e}"
        if json_output:
            return {
                "error": error_msg,
                "status": "ERROR",
                "snapshot_root": str(snapshot_root),
                "runs_found": len(run_dirs),
            }
        else:
            print(error_msg, file=sys.stderr)
            print(f"      Found {len(run_dirs)} run directories but failed to analyze them.", file=sys.stderr)
            sys.exit(1)
    
    # Plan future runs
    try:
        plan = plan_future_runs(multi_history, target_coverage=10.0)
    except Exception as e:
        error_msg = f"ERROR: Failed to plan future runs: {e}"
        if json_output:
            return {
                "error": error_msg,
                "status": "ERROR",
                "snapshot_root": str(snapshot_root),
                "multi_history": multi_history,
            }
        else:
            print(error_msg, file=sys.stderr)
            sys.exit(1)
    
    # Get orchestrator summary
    try:
        orchestrator_summary = summarize_snapshot_plans_for_u2_orchestrator(plan)
    except Exception as e:
        error_msg = f"ERROR: Failed to summarize for orchestrator: {e}"
        if json_output:
            return {
                "error": error_msg,
                "status": "ERROR",
                "snapshot_root": str(snapshot_root),
                "plan": plan,
            }
        else:
            print(error_msg, file=sys.stderr)
            sys.exit(1)
    
    # Get console tile
    try:
        console_tile = summarize_snapshot_plans_for_global_console(multi_history, plan)
    except Exception as e:
        error_msg = f"ERROR: Failed to build console tile: {e}"
        if json_output:
            return {
                "error": error_msg,
                "status": "ERROR",
                "snapshot_root": str(snapshot_root),
                "orchestrator_summary": orchestrator_summary,
            }
        else:
            print(error_msg, file=sys.stderr)
            sys.exit(1)
    
    # Build calibration experiment runbook if requested
    calibration_runbook = None
    if calibration_experiment:
        try:
            calibration_runbook = build_calibration_experiment_runbook(
                multi_history, plan, calibration_experiment
            )
        except Exception as e:
            error_msg = f"ERROR: Failed to build calibration runbook: {e}"
            if json_output:
                return {
                    "error": error_msg,
                    "status": "ERROR",
                    "snapshot_root": str(snapshot_root),
                }
            else:
                print(error_msg, file=sys.stderr)
                sys.exit(1)
    
    # Multi-run comparison if requested
    comparison_analysis = None
    trend_verdict = None
    if compare:
        try:
            previous_history = None
            if previous_history_path and previous_history_path.exists():
                with open(previous_history_path, 'r') as f:
                    previous_history = json.load(f)
            
            comparison_analysis = compare_multi_run_snapshots(multi_history, previous_history)
            # Generate trend verdict
            trend_verdict = classify_calibration_trend(comparison_analysis)
        except Exception as e:
            error_msg = f"ERROR: Failed to perform comparison analysis: {e}"
            if json_output:
                return {
                    "error": error_msg,
                    "status": "ERROR",
                    "snapshot_root": str(snapshot_root),
                }
            else:
                print(error_msg, file=sys.stderr)
                sys.exit(1)
    
    if json_output:
        result = {
            "status": "OK",
            "snapshot_root": str(snapshot_root),
            "multi_history": multi_history,
            "plan": plan,
            "orchestrator_summary": orchestrator_summary,
            "console_tile": console_tile,
        }
        if calibration_runbook:
            result["calibration_runbook"] = calibration_runbook
        if comparison_analysis:
            result["comparison_analysis"] = comparison_analysis
        if trend_verdict:
            result["trend_verdict"] = trend_verdict
        return result
    else:
        # Print human-readable summary
        print(f"Snapshot Inspector: {snapshot_root}")
        print(f"=" * 60)
        print()
        
        print(f"Runs Analyzed: {multi_history['run_count']}")
        print(f"Valid Snapshots: {multi_history['summary']['total_valid_snapshots']}")
        print(f"Corrupted Snapshots: {multi_history['summary']['total_corrupted_snapshots']}")
        print(f"Average Coverage: {multi_history['summary']['average_coverage_pct']:.1f}%")
        print(f"Global Max Gap: {multi_history['global_max_gap']} cycles")
        print(f"Overall Status: {multi_history['overall_status']}")
        print()
        
        print("Orchestrator Summary:")
        print(f"  Status: {orchestrator_summary['status']}")
        print(f"  Has Resume Targets: {orchestrator_summary['has_resume_targets']}")
        if orchestrator_summary.get('preferred_run_id'):
            print(f"  Preferred Run ID: {orchestrator_summary['preferred_run_id']}")
        if orchestrator_summary.get('preferred_snapshot_path'):
            print(f"  Preferred Snapshot: {orchestrator_summary['preferred_snapshot_path']}")
        print(f"  Message: {orchestrator_summary.get('details', {}).get('message', 'N/A')}")
        print()
        
        print("Console Tile:")
        print(f"  Status Light: {console_tile['status_light']}")
        print(f"  Headline: {console_tile['headline']}")
        print(f"  Runs Analyzed: {console_tile['runs_analyzed']}")
        print(f"  Mean Coverage: {console_tile['mean_coverage_pct']:.1f}%")
        print(f"  Max Gap: {console_tile['max_gap']} cycles")
        print()
        
        if plan['runs_to_extend']:
            print("Top Priority Runs to Extend:")
            for i, run in enumerate(plan['runs_to_extend'][:5], 1):
                print(f"  {i}. {Path(run['run_dir']).name}")
                print(f"     Coverage: {run['coverage_pct']:.1f}%, Max Gap: {run['max_gap']}, Priority: {run['priority_score']:.2f}")
        
        # Print calibration experiment runbook if requested
        if calibration_runbook:
            print()
            print("=== Calibration Experiment Runbook ===")
            print(f"Experiment Type: {calibration_runbook['experiment_type']}")
            print(f"Experiment Focus: {calibration_runbook['experiment_focus']}")
            if 'stability_indicators' in calibration_runbook:
                print("Stability Indicators:")
                for key, value in calibration_runbook['stability_indicators'].items():
                    print(f"  {key}: {value}")
            if 'convergence_indicators' in calibration_runbook:
                print("Convergence Indicators:")
                for key, value in calibration_runbook['convergence_indicators'].items():
                    print(f"  {key}: {value}")
            if 'resilience_indicators' in calibration_runbook:
                print("Resilience Indicators:")
                for key, value in calibration_runbook['resilience_indicators'].items():
                    print(f"  {key}: {value}")
        
        # Print comparison analysis if requested
        if comparison_analysis:
            print()
            print("=== Multi-Run Comparison Analysis ===")
            print("Stability Deltas:")
            sd = comparison_analysis['stability_deltas']
            print(f"  Coverage Mean: {sd['coverage_mean']:.1f}%")
            print(f"  Coverage Std: {sd['coverage_std']:.1f}%")
            print(f"  Coverage Stability: {sd['coverage_stability']}")
            print(f"  Gap Mean: {sd['gap_mean']:.1f} cycles")
            print(f"  Status Distribution: {sd['status_distribution']}")
            
            print("Max Gap Analysis:")
            mga = comparison_analysis['max_gap_analysis']
            print(f"  Global Max Gap: {mga['global_max_gap']} cycles")
            print(f"  Problematic Gaps: {mga['problematic_count']}")
            if mga['problematic_gaps']:
                for gap in mga['problematic_gaps'][:3]:  # Show top 3
                    print(f"    - {Path(gap['run_dir']).name}: gap {gap['max_gap']} cycles ({gap['gap_ratio']:.1%} of total), risk: {gap['risk_level']}")
            
            print("Coverage Regression:")
            cr = comparison_analysis['coverage_regression']
            if cr['detected']:
                print(f"  DETECTED: {cr['severity']} severity")
                print(f"  Mean Degradation: {cr['mean_degradation']:.1f}%")
                print(f"  Previous Mean: {cr.get('previous_mean', 'N/A')}%")
                print(f"  Current Mean: {cr.get('current_mean', 'N/A')}%")
            else:
                print("  No regression detected")
        
        # Print trend verdict if available
        if trend_verdict:
            print()
            print("=== Calibration Trend Verdict ===")
            print(f"Verdict: {trend_verdict['verdict']}")
            print(f"Confidence: {trend_verdict['confidence']:.1%}")
            print(f"Rationale: {trend_verdict['rationale']}")
            print()
            print("Top 3 Contributing Signals:")
            for i, signal in enumerate(trend_verdict['top_signals'], 1):
                print(f"  {i}. {signal['signal']} ({signal['strength']} strength, impact: {signal['impact']:.2f})")
                print(f"     {signal['message']}")
            print()
            print("Weight Summary:")
            ws = trend_verdict['weight_summary']
            print(f"  Positive Weight: {ws['positive_weight']:.3f}")
            print(f"  Negative Weight: {ws['negative_weight']:.3f}")
            print(f"  Total Weight: {ws['total_weight']:.3f}")
        
        return {
            "status": "OK",
            "orchestrator_summary": orchestrator_summary,
            "console_tile": console_tile,
        }


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="U2 Snapshot Inspector - Debug snapshot planning decisions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Human-readable summary
  python scripts/u2_snapshot_inspect.py --snapshot-root results/u2_snapshots
  
  # JSON output for scripting
  python scripts/u2_snapshot_inspect.py --snapshot-root results/u2_snapshots --json
        """
    )
    parser.add_argument(
        "--snapshot-root",
        type=Path,
        required=False,
        help="Root directory containing run directories with snapshots (required unless --reconcile-cal-exp1)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output full JSON instead of human-readable summary"
    )
    
    args = parser.parse_args()
    
    # Handle reconciliation mode (early exit)
    if args.reconcile_cal_exp1:
        if not args.run_a or not args.run_b:
            print("ERROR: --reconcile-cal-exp1 requires --run-a and --run-b", file=sys.stderr)
            sys.exit(1)
        
        try:
            reconciliation_result = reconcile_cal_exp1_runs(args.run_a, args.run_b)
            
            # Output JSON if requested
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(reconciliation_result, f, indent=2)
                print(f"Reconciliation result written to {args.output}")
            
            # Print text summary
            print("=" * 80)
            print("CAL-EXP-1 RECONCILIATION")
            print("=" * 80)
            print()
            print(f"Run A: {args.run_a}")
            print(f"  Schema: {reconciliation_result['run_a_metadata']['schema_version']}")
            print(f"  Windows: {reconciliation_result['run_a_metadata']['window_count']}")
            print(f"  Seed: {reconciliation_result['run_a_metadata']['params'].get('seed', 'N/A')}")
            print()
            print(f"Run B: {args.run_b}")
            print(f"  Schema: {reconciliation_result['run_b_metadata']['schema_version']}")
            print(f"  Windows: {reconciliation_result['run_b_metadata']['window_count']}")
            print(f"  Seed: {reconciliation_result['run_b_metadata']['params'].get('seed', 'N/A')}")
            print()
            print(f"Reconciliation Verdict: {reconciliation_result['reconciliation_verdict']}")
            print()
            print("Explainability:")
            for explanation in reconciliation_result['explainability']:
                print(f"  - {explanation}")
            print()
            print("Side-by-Side Deltas (Top 5 Windows):")
            aligned = [w for w in reconciliation_result['side_by_side_deltas'] if w.get('aligned', False)]
            for window in aligned[:5]:
                print(f"  Window {window['window_index']}:")
                print(f"    Run A: mean_delta_p={window['run_a']['mean_delta_p']:.6f}, state_divergence_rate={window['run_a']['state_divergence_rate']:.6f}")
                print(f"    Run B: mean_delta_p={window['run_b']['mean_delta_p']:.6f}, state_divergence_rate={window['run_b']['state_divergence_rate']:.6f}")
                print(f"    Deltas: mean_delta_p={window['deltas']['mean_delta_p']:.6f}, state_divergence_rate={window['deltas']['state_divergence_rate']:.6f}")
                print(f"    Agreement: mean_delta_p={window['agreement']['mean_delta_p']}, state_divergence_rate={window['agreement']['state_divergence_rate']}")
                print()
            
            if args.json:
                print(json.dumps(reconciliation_result, indent=2))
            
            sys.exit(0)
        except Exception as e:
            print(f"ERROR: Reconciliation failed: {e}", file=sys.stderr)
            sys.exit(1)
    
    # Require snapshot-root for non-reconcile mode
    if not args.snapshot_root:
        print("ERROR: --snapshot-root is required (unless using --reconcile-cal-exp1)", file=sys.stderr)
        sys.exit(1)
    
    result = inspect_snapshots(
        args.snapshot_root,
        json_output=args.json,
        calibration_experiment=args.calibration_experiment,
        compare=args.compare,
        previous_history_path=args.previous_history,
    )
    
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    
    # Exit with error code if status is ERROR
    if isinstance(result, dict) and result.get("status") == "ERROR":
        sys.exit(1)


if __name__ == "__main__":
    main()

