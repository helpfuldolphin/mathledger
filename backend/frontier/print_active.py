#!/usr/bin/env python3
"""
MathLedger Print Active Slice Helper

Prints key=value pairs for the active curriculum slice for PowerShell parsing.
"""

import argparse
import sys
from .curriculum import load


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Print active curriculum slice parameters")
    parser.add_argument('--system', required=True, help='System slug (e.g., pl)')

    args = parser.parse_args()

    try:
        # Load curriculum configuration
        system_cfg = load(args.system)
        active_slice = system_cfg.active_slice
        params = active_slice.params

        # Print key=value pairs for PowerShell parsing
        print(f"DEPTH_MAX={params.get('depth_max', 4)}")
        print(f"BREADTH_MAX={params.get('breadth_max', 500)}")
        print(f"TOTAL_MAX={params.get('total_max', 2000)}")
        print(f"ATOMS={params.get('atoms', 4)}")
        print(f"ACTIVE_SLICE={active_slice.name}")

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
