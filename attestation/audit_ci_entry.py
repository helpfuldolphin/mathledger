"""
CI Entry Point for Attestation Auditing
========================================

Provides a CLI interface for running attestation audits in CI/CD pipelines.
Supports single experiment, multi-experiment, and evidence chain ledger modes.
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from attestation.audit_uplift_u2 import (
    audit_experiment,
    render_audit_json,
    render_audit_markdown,
)
from attestation.audit_uplift_u2_all import (
    audit_all_experiments,
    render_aggregate_json,
    render_aggregate_markdown,
)
from attestation.evidence_chain import (
    build_evidence_chain_ledger,
    evaluate_evidence_chain_for_ci,
    render_evidence_chain_section,
)


def main() -> int:
    """Main entry point for CI auditing."""
    parser = argparse.ArgumentParser(
        description="Attestation auditor for U2 uplift experiments"
    )
    
    parser.add_argument(
        "--mode",
        choices=["single", "multi", "evidence-chain"],
        required=True,
        help="Audit mode: single experiment, multiple experiments, or evidence chain"
    )
    
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        help="Path to single experiment directory (for single mode)"
    )
    
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        help="Path to directory containing multiple experiments (for multi/evidence-chain modes)"
    )
    
    parser.add_argument(
        "--repo-root",
        type=Path,
        help="Repository root path (defaults to parent of experiment-dir)"
    )
    
    parser.add_argument(
        "--pattern",
        default="*",
        help="Glob pattern for experiment directories (for multi mode)"
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file path for report (defaults to stdout)"
    )
    
    parser.add_argument(
        "--format",
        choices=["json", "markdown"],
        default="json",
        help="Output format"
    )
    
    parser.add_argument(
        "--exit-code",
        action="store_true",
        help="Exit with status code based on audit results (evidence-chain mode only)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode == "single" and not args.experiment_dir:
        print("Error: --experiment-dir required for single mode", file=sys.stderr)
        return 2
    
    if args.mode in ["multi", "evidence-chain"] and not args.experiments_dir:
        print("Error: --experiments-dir required for multi/evidence-chain mode", file=sys.stderr)
        return 2
    
    # Run appropriate audit
    try:
        if args.mode == "single":
            return run_single_audit(args)
        elif args.mode == "multi":
            return run_multi_audit(args)
        elif args.mode == "evidence-chain":
            return run_evidence_chain_audit(args)
    except Exception as e:
        print(f"Error during audit: {str(e)}", file=sys.stderr)
        return 2
    
    return 0


def run_single_audit(args: argparse.Namespace) -> int:
    """Run single experiment audit."""
    result = audit_experiment(args.experiment_dir, args.repo_root)
    
    # Render output
    if args.format == "json":
        output = render_audit_json(result)
    else:
        output = render_audit_markdown(result)
    
    # Write output
    if args.output:
        args.output.write_text(output)
    else:
        print(output)
    
    # Return exit code based on status
    return 0 if result.status == "PASS" else 1


def run_multi_audit(args: argparse.Namespace) -> int:
    """Run multi-experiment audit."""
    results = audit_all_experiments(
        args.experiments_dir,
        args.repo_root,
        args.pattern
    )
    
    # Render output
    if args.format == "json":
        output = render_aggregate_json(results)
    else:
        output = render_aggregate_markdown(results)
    
    # Write output
    if args.output:
        args.output.write_text(output)
    else:
        print(output)
    
    # Return exit code based on results
    failed_count = sum(1 for r in results if r.status == "FAIL")
    return 0 if failed_count == 0 else 1


def run_evidence_chain_audit(args: argparse.Namespace) -> int:
    """Run evidence chain audit with ledger generation."""
    # Run multi-experiment audit
    results = audit_all_experiments(
        args.experiments_dir,
        args.repo_root,
        args.pattern
    )
    
    # Convert results to dict format for ledger building
    audit_results = [
        {
            "experiment_id": r.experiment_id,
            "status": r.status,
            "manifest_path": r.manifest_path,
            "manifest_hash": r.manifest_hash,
            "artifacts": [
                {
                    "path": a.path,
                    "hash": a.actual_hash
                }
                for a in r.artifacts_checked
            ]
        }
        for r in results
    ]
    
    # Build evidence chain ledger
    ledger = build_evidence_chain_ledger(audit_results)
    
    # Render output
    if args.format == "json":
        output = render_ledger_json(ledger)
    else:
        output = render_ledger_markdown(ledger)
    
    # Write output
    if args.output:
        args.output.write_text(output)
    else:
        print(output)
    
    # Return exit code if requested
    if args.exit_code:
        return evaluate_evidence_chain_for_ci(ledger)
    
    return 0


def render_ledger_json(ledger: Dict[str, Any]) -> str:
    """Render ledger as JSON."""
    import json
    return json.dumps(ledger, indent=2)


def render_ledger_markdown(ledger: Dict[str, Any]) -> str:
    """Render ledger as Markdown."""
    lines = [
        "# Evidence Chain Ledger",
        "",
        f"**Schema Version:** {ledger['schema_version']}",
        f"**Experiment Count:** {ledger['experiment_count']}",
        f"**Global Status:** `{ledger['global_status']}`",
        f"**Ledger Hash:** `{ledger['ledger_hash']}`",
        "",
    ]
    
    # Add evidence chain section
    chain_section = render_evidence_chain_section(ledger)
    lines.append(chain_section)
    
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
