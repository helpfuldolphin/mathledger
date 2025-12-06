#!/usr/bin/env python3
"""
MathLedger Ratchet CLI

Command-line tool to evaluate curriculum ratchet decisions and advance to next slices.
"""

import argparse
import json
import os
import sys
from typing import Dict, Any

from .curriculum import GateVerdict, activate_next_slice, load, should_ratchet


def fetch_metrics(metrics_url: str) -> Dict[str, Any]:
    """Fetch metrics from the API endpoint."""
    try:
        import urllib.request
        import urllib.error

        with urllib.request.urlopen(metrics_url, timeout=10) as response:
            return json.loads(response.read().decode('utf-8'))
    except Exception as e:
        print(f"Error fetching metrics from {metrics_url}: {e}", file=sys.stderr)
        sys.exit(1)


def load_metrics_from_file(file_path: str) -> Dict[str, Any]:
    """Load metrics from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading metrics from {file_path}: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="MathLedger curriculum ratchet tool")
    parser.add_argument('--system', required=True, help='System slug (e.g., pl)')
    parser.add_argument('--metrics-url', default='http://localhost:8000/metrics',
                       help='Metrics API URL')
    parser.add_argument('--metrics-path', help='Path to metrics JSON file (offline mode)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would happen without making changes')
    parser.add_argument('--format', choices=['plain', 'json'], default='plain', help='Output format')

    args = parser.parse_args()

    try:
        # Load curriculum configuration
        system_cfg = load(args.system)
        active_slice = system_cfg.active_slice
        active_slice_name = system_cfg.active_name

        # Load metrics
        if args.metrics_path:
            metrics = load_metrics_from_file(args.metrics_path)
        else:
            metrics = fetch_metrics(args.metrics_url)

        # Evaluate ratchet decision
        verdict = should_ratchet(metrics, system_cfg)

        if verdict.advance and not args.dry_run:
            # Advance to the next slice
            try:
                updated_cfg = activate_next_slice(args.system, attestation=verdict.audit)
                new_active = updated_cfg.active_name
                decision = "ratchet"
            except ValueError as e:
                # No next slice available
                new_active = active_slice_name
                decision = "hold"
                verdict = GateVerdict(
                    advance=False,
                    reason=str(e),
                    audit={**verdict.audit, 'summary': str(e)}
                )
        else:
            # Hold current slice
            new_active = active_slice_name
            decision = "hold"
            if verdict.advance and args.dry_run:
                # Would ratchet but in dry-run mode
                next_slice = system_cfg.next_slice()
                if next_slice:
                    new_active = next_slice.name
                    decision = "ratchet"

        params = active_slice.params
        atoms = params.get('atoms', 4)
        depth = params.get('depth_max', 4)

        # Prepare result data
        result = {
            'decide': decision,
            'system': args.system,
            'slice': active_slice_name,
            'next_slice': new_active,
            'atoms': atoms,
            'depth': depth,
            'reason': verdict.reason,
            'audit': verdict.audit,
        }

        # Write JSON to metrics file for integration
        if not args.dry_run:
            metrics_dir = 'metrics'
            os.makedirs(metrics_dir, exist_ok=True)
            with open(f'{metrics_dir}/ratchet_last.txt', 'w') as f:
                json.dump(result, f)

        if args.format == 'json':
            # Output as JSON
            print(json.dumps(result))
        else:
            # Print one-line summary
            print(
                f'decide={decision} system={args.system} slice={active_slice_name} '
                f'atoms={atoms} depth={depth} reason="{verdict.reason}"'
            )

    except Exception as e:
        error_result = {
            'decide': 'error',
            'system': args.system,
            'slice': 'unknown',
            'next_slice': 'unknown',
            'atoms': 0,
            'depth': 0,
            'reason': f'error: {e}',
            'audit': None,
        }
        if args.format == 'json':
            print(json.dumps(error_result))
        else:
            print(f'decide=error system={args.system} slice=unknown atoms=0 depth=0 reason="error: {e}"', file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
