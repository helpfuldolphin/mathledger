#!/usr/bin/env python3
"""
P5 Identity Flight Check CLI

Compares synthetic vs production configurations before enabling RealTelemetryAdapter.

Exit Codes:
    0 = OK (safe to enable RealTelemetryAdapter)
    1 = INVESTIGATE (review items, proceed with caution)
    2 = BLOCK (do not enable, resolve issues first)

Usage Examples:

    # Basic check with YAML configs
    python scripts/check_p5_identity_alignment.py \\
        --synthetic-config tests/fixtures/slice_config.yaml \\
        --prod-config /tmp/prod_slice_config.yaml

    # Check with P4 evidence pack
    python scripts/check_p5_identity_alignment.py \\
        --synthetic-config tests/fixtures/slice_config.yaml \\
        --prod-config /tmp/prod_slice_config.yaml \\
        --p4-evidence-pack evidence/latest_p4_evidence_pack.json

    # JSON output for CI/CD integration
    python scripts/check_p5_identity_alignment.py \\
        --synthetic-config tests/fixtures/slice_config.yaml \\
        --prod-config /tmp/prod_slice_config.yaml \\
        --output-format json

    # Diagnose divergence details
    python scripts/check_p5_identity_alignment.py \\
        --synthetic-config tests/fixtures/slice_config.yaml \\
        --prod-config /tmp/prod_slice_config.yaml \\
        --diagnose

See: docs/system_law/P5_Identity_Flight_Check_Runbook.md

Status: PHASE X P5 PRE-FLIGHT
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.health.identity_alignment_checker import (
    CheckReport,
    CheckResult,
    check_p5_identity_alignment,
    diagnose_config_divergence,
)


def load_config(path: str) -> Dict[str, Any]:
    """Load config from YAML or JSON file."""
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    content = file_path.read_text(encoding="utf-8")

    if file_path.suffix in (".yaml", ".yml"):
        try:
            import yaml
            return yaml.safe_load(content)
        except ImportError:
            # Fallback: try to parse as JSON-like YAML
            raise ImportError(
                "PyYAML not installed. Install with: pip install pyyaml\n"
                "Or use JSON config files instead."
            )
    elif file_path.suffix == ".json":
        return json.loads(content)
    else:
        # Try JSON first, then YAML
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            try:
                import yaml
                return yaml.safe_load(content)
            except ImportError:
                raise ValueError(
                    f"Cannot determine config format for {path}. "
                    "Use .yaml, .yml, or .json extension."
                )


def main() -> int:
    """
    CLI entry point.

    Returns exit code:
        0 = OK
        1 = INVESTIGATE
        2 = BLOCK
    """
    parser = argparse.ArgumentParser(
        description="P5 Identity Flight Check - Compare synthetic vs production configs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Basic check:
    %(prog)s --synthetic-config syn.yaml --prod-config prod.yaml

  With P4 evidence:
    %(prog)s --synthetic-config syn.yaml --prod-config prod.yaml \\
             --p4-evidence-pack evidence.json

  JSON output:
    %(prog)s --synthetic-config syn.yaml --prod-config prod.yaml \\
             --output-format json

  Diagnose mode:
    %(prog)s --synthetic-config syn.yaml --prod-config prod.yaml --diagnose

Exit Codes:
  0 = OK (safe to enable RealTelemetryAdapter)
  1 = INVESTIGATE (review items, proceed with caution)
  2 = BLOCK (do not enable, resolve issues first)
""",
    )

    parser.add_argument(
        "--synthetic-config",
        required=True,
        help="Path to synthetic config (YAML or JSON)",
    )
    parser.add_argument(
        "--prod-config",
        required=True,
        help="Path to production config (YAML or JSON)",
    )
    parser.add_argument(
        "--p4-evidence-pack",
        help="Path to P4 evidence pack JSON (optional)",
    )
    parser.add_argument(
        "--output-format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Show detailed divergence diagnosis",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Only output status (OK/INVESTIGATE/BLOCK)",
    )

    args = parser.parse_args()

    try:
        # Load configs
        synthetic = load_config(args.synthetic_config)
        production = load_config(args.prod_config)

        evidence: Optional[Dict[str, Any]] = None
        if args.p4_evidence_pack:
            evidence = load_config(args.p4_evidence_pack)

        # Run check
        report = check_p5_identity_alignment(synthetic, production, evidence)

        # Diagnose mode
        if args.diagnose:
            diagnosis = diagnose_config_divergence(synthetic, production)
            if args.output_format == "json":
                print(json.dumps(diagnosis, indent=2))
            else:
                print("\n" + "=" * 60)
                print("DIVERGENCE DIAGNOSIS")
                print("=" * 60)
                print(f"\nMatch: {diagnosis['match']}")
                print(f"Diagnosis: {diagnosis['diagnosis']}")
                print(f"\nSynthetic FP: {diagnosis['synthetic_fingerprint'][:32]}...")
                print(f"Production FP: {diagnosis['production_fingerprint'][:32]}...")

                if not diagnosis["match"]:
                    if diagnosis.get("differing_params"):
                        print("\nDiffering Parameters:")
                        for p in diagnosis["differing_params"]:
                            print(f"  - {p['param']}: syn={p['synthetic']} prod={p['production']}")

                    if diagnosis.get("differing_gates"):
                        print("\nDiffering Gates:")
                        for g in diagnosis["differing_gates"]:
                            print(f"  - {g['gate']}: syn={g['synthetic']} prod={g['production']}")

                    if diagnosis.get("recommended_action"):
                        print(f"\nRecommended Action: {diagnosis['recommended_action']}")
                print()

        # Output
        if args.quiet:
            print(report.overall_status.value)
        elif args.output_format == "json":
            print(json.dumps(report.to_dict(), indent=2))
        else:
            print(report.to_text_report())

        return report.get_exit_code()

    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 3
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in config file: {e}", file=sys.stderr)
        return 3
    except ImportError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 3
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
