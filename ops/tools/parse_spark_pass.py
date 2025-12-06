#!/usr/bin/env python3
"""
SPARK PASS Line Parser

Extracts H_t from SPARK run log file by parsing the canonical PASS line:
  [PASS] FIRST ORGANISM ALIVE H_t=<short-hex>

This tool is used by Cursor J (SPARK Execution Verifier) to verify that
the First Organism has completed successfully and emitted the required
certification line.

Usage:
    python ops/tools/parse_spark_pass.py [log_file_path]
    
    Default log file: ops/logs/SPARK_run_log.txt

Exit codes:
    0: H_t found and printed
    1: H_t not found or log file missing
"""

import re
import sys
from pathlib import Path


def extract_h_t_from_log(log_file: Path) -> str | None:
    """
    Extract H_t from SPARK run log file.
    
    Looks for the canonical PASS line:
        [PASS] FIRST ORGANISM ALIVE H_t=<short-hex>
    
    Args:
        log_file: Path to SPARK run log file
        
    Returns:
        Short H_t hex string (typically 12 chars) if found, None otherwise
    """
    if not log_file.exists():
        return None
    
    # Pattern matches: [PASS] FIRST ORGANISM ALIVE H_t=<hex>
    # The hex can be any length (typically 12 chars, but we're flexible)
    pattern = r'\[PASS\]\s+FIRST\s+ORGANISM\s+ALIVE\s+H_t=([0-9a-fA-F]+)'
    
    try:
        with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                match = re.search(pattern, line)
                if match:
                    return match.group(1)
    except Exception as e:
        print(f"Error reading log file: {e}", file=sys.stderr)
        return None
    
    return None


def main():
    """Main entry point."""
    # Default log file path
    default_log = Path(__file__).parent.parent / "logs" / "SPARK_run_log.txt"
    
    # Allow override via command line
    if len(sys.argv) > 1:
        log_file = Path(sys.argv[1])
    else:
        log_file = default_log
    
    # Extract H_t
    h_t = extract_h_t_from_log(log_file)
    
    if h_t is None:
        print(f"Error: PASS line not found in {log_file}", file=sys.stderr)
        print("Expected format: [PASS] FIRST ORGANISM ALIVE H_t=<short-hex>", file=sys.stderr)
        sys.exit(1)
    
    # Print H_t in the format expected by downstream tools
    print(f"H_t={h_t}")
    sys.exit(0)


if __name__ == "__main__":
    main()

