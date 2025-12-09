#!/usr/bin/env python3
"""
Governance Verifier CLI

==============================================================================
STATUS: PHASE II — IMPLEMENTATION
==============================================================================

Thin CLI wrapper for the governance_verifier module. Verifies summary.json
files against the 43 governance rules.

Usage:
    python scripts/run_governance_verifier.py --summary results/summary.json
    python scripts/run_governance_verifier.py --summary results/summary.json --json
    python scripts/run_governance_verifier.py --summary results/summary.json --manifest results/manifest.json

Exit Codes:
    0: PASS or WARN
    1: FAIL (INVALIDATING violations)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, List, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.analytics.governance_verifier import (
    governance_verify,
    GovernanceVerdict,
    __version__,
    explain_verdict,
    build_governance_posture,
    summarize_for_admissibility,
    RULE_DESCRIPTIONS,
)


def load_json(path: str) -> dict:
    """Load a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_yaml(path: str) -> dict:
    """Load a YAML file."""
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except ImportError:
        # Fallback: try to parse as JSON if yaml not available
        return load_json(path)


def format_verdict_text(verdict: GovernanceVerdict, show_explanation: bool = True) -> str:
    """Format verdict for human-readable output."""
    lines = []

    # Header
    lines.append("=" * 60)
    lines.append(f"GOVERNANCE VERIFIER v{verdict.verifier_version}")
    lines.append("=" * 60)
    lines.append("")

    # Status with visual indicator
    status_icons = {
        "PASS": "[PASS]",
        "WARN": "[WARN]",
        "FAIL": "[FAIL]",
    }
    lines.append(f"Status: {status_icons.get(verdict.status, verdict.status)} {verdict.status}")
    lines.append(f"Timestamp: {verdict.timestamp}")
    lines.append("")

    # v2: Governance Explanation section
    if show_explanation and verdict.short_explanation:
        lines.append("-" * 40)
        lines.append("EXPLANATION:")
        lines.append("-" * 40)
        lines.append(f"  {verdict.short_explanation}")
        lines.append("")

    # Summary stats
    lines.append(f"Rules checked: {verdict.rules_checked}")
    lines.append(f"Passed: {len(verdict.passed_rules)}")
    lines.append(f"Warnings: {len(verdict.warnings)}")
    lines.append(f"INVALIDATING: {len(verdict.invalidating_rules)}")
    lines.append("")

    # Violations
    if verdict.invalidating_rules:
        lines.append("-" * 40)
        lines.append("INVALIDATING VIOLATIONS:")
        lines.append("-" * 40)
        for rule_id in verdict.invalidating_rules:
            result = verdict.details.get(rule_id)
            desc = RULE_DESCRIPTIONS.get(rule_id, "")
            if result:
                lines.append(f"  {rule_id}: {result.message}")
                if desc:
                    lines.append(f"          ({desc})")
        lines.append("")

    # Warnings
    if verdict.warnings:
        lines.append("-" * 40)
        lines.append("WARNINGS:")
        lines.append("-" * 40)
        for rule_id in verdict.warnings:
            result = verdict.details.get(rule_id)
            desc = RULE_DESCRIPTIONS.get(rule_id, "")
            if result:
                lines.append(f"  {rule_id}: {result.message}")
                if desc:
                    lines.append(f"          ({desc})")
        lines.append("")

    # v2: Reason codes
    if verdict.reason_codes:
        lines.append("-" * 40)
        lines.append("REASON CODES:")
        lines.append("-" * 40)
        lines.append(f"  {', '.join(verdict.reason_codes)}")
        lines.append("")

    # Inputs
    lines.append("-" * 40)
    lines.append("INPUTS:")
    lines.append("-" * 40)
    for key, value in verdict.inputs.items():
        lines.append(f"  {key}: {value}")
    lines.append("")

    lines.append("=" * 60)

    return "\n".join(lines)


def format_posture_text(posture: dict) -> str:
    """Format governance posture for human-readable output."""
    lines = []

    lines.append("=" * 60)
    lines.append("GOVERNANCE POSTURE SUMMARY")
    lines.append("=" * 60)
    lines.append("")

    # Aggregate status
    status_icons = {
        "PASS": "[PASS]",
        "WARN": "[WARN]",
        "FAIL": "[FAIL]",
    }
    agg_status = posture.get("aggregate_status", "UNKNOWN")
    lines.append(f"Aggregate Status: {status_icons.get(agg_status, agg_status)} {agg_status}")
    lines.append(f"Governance Blocking: {'YES' if posture.get('is_governance_blocking') else 'NO'}")
    lines.append("")

    # Counts
    lines.append("-" * 40)
    lines.append("VERDICT COUNTS:")
    lines.append("-" * 40)
    lines.append(f"  Total files: {posture.get('total_count', 0)}")
    lines.append(f"  PASS: {posture.get('pass_count', 0)}")
    lines.append(f"  WARN: {posture.get('warn_count', 0)}")
    lines.append(f"  FAIL: {posture.get('fail_count', 0)}")
    lines.append("")

    # Failing files
    failing_files = posture.get("failing_files", [])
    if failing_files:
        lines.append("-" * 40)
        lines.append("FAILING FILES:")
        lines.append("-" * 40)
        for ff in failing_files:
            lines.append(f"  {ff.get('file', 'unknown')}")
            lines.append(f"    Reason codes: {', '.join(ff.get('reason_codes', []))}")
            lines.append(f"    {ff.get('message', '')}")
        lines.append("")

    lines.append("=" * 60)

    return "\n".join(lines)


def discover_summary_files(directories: List[str]) -> List[str]:
    """Discover all summary.json files in the given directories."""
    summaries = []
    for dir_path in directories:
        p = Path(dir_path)
        if p.is_dir():
            summaries.extend(str(f) for f in p.rglob("summary.json"))
        elif p.is_file() and p.name == "summary.json":
            summaries.append(str(p))
    return sorted(set(summaries))


def verify_single_file(
    summary_path: str,
    manifest_path: Optional[str] = None,
    telemetry_path: Optional[str] = None,
    prereg_path: Optional[str] = None,
) -> Tuple[GovernanceVerdict, str]:
    """Verify a single summary file. Returns (verdict, file_path)."""
    summary = load_json(summary_path)
    base_path = str(Path(summary_path).parent)

    manifest = None
    if manifest_path:
        try:
            manifest = load_json(manifest_path)
        except (FileNotFoundError, json.JSONDecodeError):
            pass
    else:
        # Try auto-discovery
        auto_manifest = Path(base_path) / "manifest.json"
        if auto_manifest.exists():
            try:
                manifest = load_json(str(auto_manifest))
            except (FileNotFoundError, json.JSONDecodeError):
                pass

    telemetry = None
    if telemetry_path:
        try:
            telemetry = load_json(telemetry_path)
        except (FileNotFoundError, json.JSONDecodeError):
            pass
    else:
        # Try auto-discovery
        auto_telemetry = Path(base_path) / "telemetry_summary.json"
        if auto_telemetry.exists():
            try:
                telemetry = load_json(str(auto_telemetry))
            except (FileNotFoundError, json.JSONDecodeError):
                pass

    prereg = None
    if prereg_path:
        try:
            prereg = load_yaml(prereg_path)
        except (FileNotFoundError, Exception):
            pass

    verdict = governance_verify(
        summary=summary,
        manifest=manifest,
        telemetry=telemetry,
        prereg=prereg,
        base_path=base_path,
    )

    return verdict, summary_path


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Governance Verifier v2 - Verify uplift analysis summaries with explanations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic verification
    python scripts/run_governance_verifier.py --summary results/summary.json

    # Full verification with all inputs
    python scripts/run_governance_verifier.py \\
        --summary results/summary.json \\
        --manifest results/manifest.json \\
        --telemetry results/telemetry.json \\
        --prereg config/PREREG_UPLIFT_U2.yaml

    # JSON output for CI integration
    python scripts/run_governance_verifier.py --summary results/summary.json --json

    # Multi-file posture mode (v2)
    python scripts/run_governance_verifier.py --discover results/ evidence/

    # Output explanation only
    python scripts/run_governance_verifier.py --summary results/summary.json --explain

    # MAAS admissibility summary
    python scripts/run_governance_verifier.py --summary results/summary.json --admissibility
        """,
    )

    parser.add_argument(
        "--summary",
        help="Path to summary.json (required unless --discover is used)",
    )

    parser.add_argument(
        "--discover",
        nargs="*",
        metavar="DIR",
        help="Discover and verify all summary.json files in directories",
    )

    parser.add_argument(
        "--manifest",
        help="Path to manifest.json (optional)",
    )

    parser.add_argument(
        "--telemetry",
        help="Path to telemetry_summary.json (optional)",
    )

    parser.add_argument(
        "--prereg",
        help="Path to preregistration YAML file (optional)",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output verdict as JSON to stdout",
    )

    parser.add_argument(
        "--output",
        "-o",
        help="Write verdict JSON to file",
    )

    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress non-essential output",
    )

    parser.add_argument(
        "--explain",
        action="store_true",
        help="Output explanation block only (v2)",
    )

    parser.add_argument(
        "--admissibility",
        action="store_true",
        help="Output MAAS admissibility summary (v2)",
    )

    parser.add_argument(
        "--posture",
        action="store_true",
        help="Output governance posture (aggregated status) in multi-file mode (v2)",
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"governance_verifier {__version__}",
    )

    args = parser.parse_args()

    # =========================================================================
    # Multi-file discovery mode (v2)
    # =========================================================================
    if args.discover is not None:
        directories = args.discover if args.discover else ["results", "evidence"]
        summary_files = discover_summary_files(directories)

        if not summary_files:
            print("No summary.json files found to verify", file=sys.stderr)
            return 0

        if not args.quiet:
            print(f"Discovered {len(summary_files)} summary file(s)", file=sys.stderr)

        verdicts: List[GovernanceVerdict] = []
        file_paths: List[str] = []

        for sf in summary_files:
            try:
                verdict, path = verify_single_file(
                    sf,
                    manifest_path=args.manifest,
                    telemetry_path=args.telemetry,
                    prereg_path=args.prereg,
                )
                verdicts.append(verdict)
                file_paths.append(path)

                if not args.quiet and not args.json:
                    status_icon = {"PASS": "[PASS]", "WARN": "[WARN]", "FAIL": "[FAIL]"}
                    print(f"  {status_icon.get(verdict.status, verdict.status)} {path}")

            except Exception as e:
                print(f"Error verifying {sf}: {e}", file=sys.stderr)

        # Build posture snapshot
        posture = build_governance_posture(verdicts, file_paths)

        # Output
        if args.json or args.posture:
            print(json.dumps(posture, indent=2))
        elif not args.quiet:
            print()
            print(format_posture_text(posture))

        # Write posture to file if requested
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(posture, f, indent=2)
            if not args.quiet and not args.json:
                print(f"Posture written to: {args.output}")

        # Exit code based on blocking status
        return 1 if posture.get("is_governance_blocking", False) else 0

    # =========================================================================
    # Single-file mode
    # =========================================================================
    if not args.summary:
        print("Error: --summary is required (or use --discover for multi-file mode)", file=sys.stderr)
        return 1

    summary_path = Path(args.summary)
    if not summary_path.exists():
        print(f"Error: Summary file not found: {args.summary}", file=sys.stderr)
        return 1

    # Load inputs
    try:
        summary = load_json(args.summary)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in summary file: {e}", file=sys.stderr)
        return 1

    manifest = None
    if args.manifest:
        try:
            manifest = load_json(args.manifest)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load manifest: {e}", file=sys.stderr)

    telemetry = None
    if args.telemetry:
        try:
            telemetry = load_json(args.telemetry)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load telemetry: {e}", file=sys.stderr)

    prereg = None
    if args.prereg:
        try:
            prereg = load_yaml(args.prereg)
        except (FileNotFoundError, Exception) as e:
            print(f"Warning: Could not load prereg: {e}", file=sys.stderr)

    # Get base path for file checks
    base_path = str(summary_path.parent)

    # Run verification
    verdict = governance_verify(
        summary=summary,
        manifest=manifest,
        telemetry=telemetry,
        prereg=prereg,
        base_path=base_path,
    )

    # =========================================================================
    # v2: Explanation-only mode
    # =========================================================================
    if args.explain:
        explanation = explain_verdict(verdict)
        if args.json:
            print(json.dumps(explanation, indent=2))
        else:
            print("-" * 40)
            print("GOVERNANCE EXPLANATION")
            print("-" * 40)
            print(f"Status: {explanation['status']}")
            print(f"Explanation: {explanation['short_explanation']}")
            print(f"Reason codes: {', '.join(explanation['reason_codes']) if explanation['reason_codes'] else 'None'}")
            print(f"Pass rate: {explanation['pass_rate']:.1%}")
        return 0

    # =========================================================================
    # v2: Admissibility mode (for MAAS integration)
    # =========================================================================
    if args.admissibility:
        admissibility = summarize_for_admissibility(verdict)
        if args.json:
            print(json.dumps(admissibility, indent=2))
        else:
            print("-" * 40)
            print("MAAS ADMISSIBILITY SUMMARY")
            print("-" * 40)
            print(f"Overall status: {admissibility['overall_status']}")
            print(f"Is admissible: {admissibility['is_admissible']}")
            print(f"Has invalidating violations: {admissibility['has_invalidating_violations']}")
            if admissibility['invalidating_rules']:
                print(f"Invalidating rules: {', '.join(admissibility['invalidating_rules'])}")
            print(f"Reason: {admissibility['reason_summary']}")
        return 1 if not admissibility['is_admissible'] else 0

    # =========================================================================
    # Standard output mode
    # =========================================================================
    if args.json:
        # JSON output to stdout
        print(json.dumps(verdict.to_dict(), indent=2))
    else:
        # Human-readable output
        if not args.quiet:
            print(format_verdict_text(verdict))
        else:
            # Minimal output
            print(f"{verdict.status}: {len(verdict.invalidating_rules)} violations, {len(verdict.warnings)} warnings")

    # Write to file if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(verdict.to_dict(), f, indent=2)
        if not args.quiet and not args.json:
            print(f"Verdict written to: {args.output}")

    # Exit code: 0 for PASS/WARN, 1 for FAIL
    if verdict.status == "FAIL":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
