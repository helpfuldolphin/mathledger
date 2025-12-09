#!/usr/bin/env python3
"""
Abstention Health Check CLI
===========================

Reads a JSONL file of abstention records and prints a health summary with red flag detection.

USAGE:
    python scripts/abstention_health_check.py --input path/to/abstentions.jsonl
    python scripts/abstention_health_check.py --input path/to/abstentions.jsonl --json

EXIT CODES:
    0: No red flags detected
    1: One or more red flags detected
    2: Error reading input file

PHASE II — VERIFICATION BUREAU
Agent B4 (verifier-ops-4)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rfl.verification.abstention_record import AbstentionRecord
from rfl.verification.abstention_semantics import (
    summarize_abstentions,
    detect_abstention_red_flags,
)


def load_abstention_records(input_path: Path) -> List[AbstentionRecord]:
    """
    Load AbstentionRecord objects from a JSONL file.
    
    Args:
        input_path: Path to JSONL file with one AbstentionRecord per line
        
    Returns:
        List of AbstentionRecord objects
        
    Raises:
        FileNotFoundError: If input file doesn't exist
        json.JSONDecodeError: If JSON is malformed
        ValueError: If record data is invalid
    """
    records: List[AbstentionRecord] = []
    
    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue  # Skip empty lines
            
            try:
                data = json.loads(line)
                record = AbstentionRecord.from_dict(data)
                records.append(record)
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                raise ValueError(
                    f"Error parsing line {line_num}: {e}\nLine content: {line[:100]}..."
                ) from e
    
    return records


def format_text_output(
    summary: Dict[str, Any],
    red_flags: List[str],
    input_path: Path,
) -> str:
    """
    Format a human-readable text output.
    
    Args:
        summary: Output from summarize_abstentions()
        red_flags: Output from detect_abstention_red_flags()
        input_path: Source file path for header
        
    Returns:
        Formatted text string
    """
    lines: List[str] = []
    
    # Header
    lines.append("=" * 60)
    lines.append("ABSTENTION HEALTH CHECK REPORT")
    lines.append("=" * 60)
    lines.append(f"Source: {input_path}")
    lines.append(f"Total Records: {summary['total']}")
    lines.append("")
    
    # By Type
    lines.append("-" * 40)
    lines.append("BREAKDOWN BY TYPE")
    lines.append("-" * 40)
    by_type = summary.get("by_type", {})
    if by_type:
        for type_key, count in by_type.items():
            if count > 0:
                pct = (count / summary['total']) * 100 if summary['total'] > 0 else 0
                lines.append(f"  {type_key}: {count} ({pct:.1f}%)")
    else:
        lines.append("  (no records)")
    lines.append("")
    
    # By Category
    lines.append("-" * 40)
    lines.append("BREAKDOWN BY CATEGORY")
    lines.append("-" * 40)
    by_category = summary.get("by_category", {})
    if by_category:
        for cat_key, count in by_category.items():
            if count > 0:
                pct = (count / summary['total']) * 100 if summary['total'] > 0 else 0
                lines.append(f"  {cat_key}: {count} ({pct:.1f}%)")
    else:
        lines.append("  (no records)")
    lines.append("")
    
    # Top Reasons
    lines.append("-" * 40)
    lines.append("TOP REASONS")
    lines.append("-" * 40)
    top_reasons = summary.get("top_reasons", [])
    if top_reasons:
        for i, reason_info in enumerate(top_reasons, start=1):
            lines.append(f"  {i}. [{reason_info['type']}] {reason_info['reason']}")
            lines.append(f"     Count: {reason_info['count']}")
    else:
        lines.append("  (no reasons recorded)")
    lines.append("")
    
    # Red Flags
    lines.append("-" * 40)
    if red_flags:
        lines.append(f"⚠️  RED FLAGS DETECTED ({len(red_flags)})")
    else:
        lines.append("✅ NO RED FLAGS DETECTED")
    lines.append("-" * 40)
    
    if red_flags:
        for flag in red_flags:
            lines.append(f"  • {flag}")
    lines.append("")
    
    lines.append("=" * 60)
    
    return "\n".join(lines)


def format_json_output(
    summary: Dict[str, Any],
    red_flags: List[str],
    input_path: Path,
) -> str:
    """
    Format JSON output.
    
    Args:
        summary: Output from summarize_abstentions()
        red_flags: Output from detect_abstention_red_flags()
        input_path: Source file path
        
    Returns:
        JSON string
    """
    output = {
        "source": str(input_path),
        "summary": summary,
        "red_flags": red_flags,
        "red_flag_count": len(red_flags),
        "health_status": "PASS" if not red_flags else "WARN",
    }
    return json.dumps(output, indent=2, ensure_ascii=False)


def main(args: Optional[List[str]] = None) -> int:
    """
    Main entry point for the CLI.
    
    Args:
        args: Command line arguments (for testing). Uses sys.argv if None.
        
    Returns:
        Exit code: 0 = pass, 1 = red flags, 2 = error
    """
    parser = argparse.ArgumentParser(
        description="Abstention Health Check - Analyze abstention JSONL files for red flags",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exit Codes:
  0  No red flags detected
  1  One or more red flags detected
  2  Error reading input file

Examples:
  python scripts/abstention_health_check.py --input run_123.jsonl
  python scripts/abstention_health_check.py --input run_123.jsonl --json > report.json
        """,
    )
    
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Path to JSONL file containing AbstentionRecord objects",
    )
    
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output in JSON format instead of text",
    )
    
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of top reasons to include (default: 5)",
    )
    
    parser.add_argument(
        "--timeout-threshold",
        type=float,
        default=50.0,
        help="Percentage threshold for timeout red flag (default: 50.0)",
    )
    
    parser.add_argument(
        "--crash-threshold",
        type=float,
        default=30.0,
        help="Percentage threshold for crash red flag (default: 30.0)",
    )
    
    parsed = parser.parse_args(args)
    
    # Validate input file
    input_path: Path = parsed.input
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return 2
    
    # Load records
    try:
        records = load_abstention_records(input_path)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error parsing input file: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"Error reading input file: {e}", file=sys.stderr)
        return 2
    
    # Generate summary
    summary = summarize_abstentions(records, top_n=parsed.top_n)
    
    # Detect red flags
    custom_thresholds = {
        "timeout_threshold_pct": parsed.timeout_threshold,
        "crash_threshold_pct": parsed.crash_threshold,
    }
    red_flags = detect_abstention_red_flags(summary, thresholds=custom_thresholds)
    
    # Output
    if parsed.json:
        output = format_json_output(summary, red_flags, input_path)
    else:
        output = format_text_output(summary, red_flags, input_path)
    
    print(output)
    
    # Exit code
    return 1 if red_flags else 0


if __name__ == "__main__":
    sys.exit(main())

