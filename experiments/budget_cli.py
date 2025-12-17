#!/usr/bin/env python3
"""
Minimal CLI for budget observability.

Usage:
    python -m experiments.budget_cli summarize <log_file.jsonl
    python -m experiments.budget_cli summarize <log_file.jsonl --json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from .budget_observability import summarize_budget_from_logs


def main():
    parser = argparse.ArgumentParser(description="Budget observability CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Summarize command
    summarize_parser = subparsers.add_parser("summarize", help="Summarize budget from JSONL logs")
    summarize_parser.add_argument("log_file", type=Path, help="Path to JSONL log file")
    summarize_parser.add_argument("--json", action="store_true", help="Output JSON format")
    
    args = parser.parse_args()
    
    if args.command == "summarize":
        if not args.log_file.exists():
            print(f"Error: Log file not found: {args.log_file}", file=sys.stderr)
            sys.exit(1)
        
        # Read log lines
        with open(args.log_file, 'r', encoding='utf-8') as f:
            log_lines = f.readlines()
        
        # Summarize
        result = summarize_budget_from_logs(log_lines)
        
        # Output
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"Status: {result['status']}")
            print(f"Total Cycles: {result['total_cycles']}")
            print(f"Budget Exhausted: {result['budget_exhausted_pct']:.2f}%")
            print(f"Avg Timeout Abstentions: {result['timeout_abstentions_avg']:.2f}")
    
    sys.exit(0)


if __name__ == "__main__":
    main()

