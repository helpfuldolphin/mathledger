"""
CLI entry point for RFL policy comparison tool.

Usage:
    python -m rfl.policy.compare --a policy_a.json --b policy_b.json
    python -m rfl.policy.compare --a policy_a.json --b policy_b.jsonl --index 5
    python -m rfl.policy.compare --a policy_a.json --b policy_b.json --top-k 20 --json
"""

import argparse
import json
import sys
from pathlib import Path

from .compare import compare_policy_states, load_policy_from_json


def main():
    parser = argparse.ArgumentParser(
        description="Compare two RFL policy states",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two JSON files
  python -m rfl.policy.compare --a policy_a.json --b policy_b.json
  
  # Compare with specific JSONL index
  python -m rfl.policy.compare --a policy_a.json --b policy_b.jsonl --index 10
  
  # Output as JSON with top-20 features
  python -m rfl.policy.compare --a a.json --b b.json --top-k 20 --json
  
  # Handle mismatched feature sets using union
  python -m rfl.policy.compare --a a.json --b b.json --handle-missing union
        """
    )
    
    parser.add_argument(
        '--a', '--policy-a',
        required=True,
        dest='policy_a',
        type=Path,
        help='Path to first policy file (JSON or JSONL)'
    )
    
    parser.add_argument(
        '--b', '--policy-b',
        required=True,
        dest='policy_b',
        type=Path,
        help='Path to second policy file (JSON or JSONL)'
    )
    
    parser.add_argument(
        '--index-a',
        type=int,
        default=None,
        help='JSONL line index for policy A (default: last line)'
    )
    
    parser.add_argument(
        '--index-b',
        type=int,
        default=None,
        help='JSONL line index for policy B (default: last line)'
    )
    
    parser.add_argument(
        '--top-k',
        type=int,
        default=10,
        help='Number of top features to show by delta (default: 10)'
    )
    
    parser.add_argument(
        '--handle-missing',
        choices=['error', 'union', 'intersection'],
        default='error',
        help='How to handle mismatched feature sets (default: error)'
    )
    
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output as JSON instead of human-readable format'
    )
    
    parser.add_argument(
        '--name-a',
        type=str,
        default=None,
        help='Name for policy A (default: filename)'
    )
    
    parser.add_argument(
        '--name-b',
        type=str,
        default=None,
        help='Name for policy B (default: filename)'
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not args.policy_a.exists():
        print(f"Error: Policy file not found: {args.policy_a}", file=sys.stderr)
        sys.exit(1)
    
    if not args.policy_b.exists():
        print(f"Error: Policy file not found: {args.policy_b}", file=sys.stderr)
        sys.exit(1)
    
    # Load policies
    try:
        state_a = load_policy_from_json(args.policy_a, args.index_a)
        state_b = load_policy_from_json(args.policy_b, args.index_b)
    except Exception as e:
        print(f"Error loading policy files: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Determine names
    name_a = args.name_a or args.policy_a.stem
    name_b = args.name_b or args.policy_b.stem
    
    # Compare
    try:
        result = compare_policy_states(
            state_a,
            state_b,
            slice_name_a=name_a,
            slice_name_b=name_b,
            top_k=args.top_k,
            handle_missing=args.handle_missing
        )
    except Exception as e:
        print(f"Error comparing policies: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Output
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(result.format_summary())


if __name__ == '__main__':
    main()
