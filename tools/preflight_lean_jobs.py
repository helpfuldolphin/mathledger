#!/usr/bin/env python3
"""
Preflight Lean Job Syntax Validator
Claude B — Veracity Engineer

Scans auto-generated Lean 4 job files for malformations.
Exits 0 if all valid, 1 if any defects found.

Defect Classes:
1. escaped_latex: Backslash-escaped colons/commas (theory\:\Propositional)
2. unicode_escape: Raw Unicode escape sequences (\u2192, \u2227)
3. incomplete_brace: Malformed set/bracket syntax ({\ or p\})
4. unprovable: Goal unprovable without hypothesis

Usage:
    python tools/preflight_lean_jobs.py [--jobs-dir PATH] [--json OUTPUT]
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple


# Defect patterns (compiled for performance)
PATTERNS = {
    "escaped_latex": re.compile(r"theory\\:|goal_type\\:"),
    "unicode_escape": re.compile(r"\\u[0-9a-fA-F]{4}"),
    "incomplete_brace": re.compile(r"{\s*\\|\\}(?!\s*:=)"),
    "unprovable": re.compile(r"theorem\s+\w+\s+\([^)]+\)\s*:\s*(p|q|r|s|t)\s+:=\s+by"),
}


def classify_job(file_path: Path) -> Tuple[str, Dict]:
    """
    Classify a single Lean job file.

    Returns:
        (status, defect_dict) where status in {"valid", "malformed"}
    """
    content = file_path.read_text(encoding="utf-8")

    # Extract theorem line
    theorem_match = re.search(
        r"theorem\s+(\w+)\s+\(p q r s t : Prop\)\s*:\s*(.+?)\s*:=\s*by",
        content,
        re.MULTILINE
    )

    if not theorem_match:
        return "malformed", {
            "pattern": "parse_error",
            "message": "Could not parse theorem declaration",
            "hex_window": None,
        }

    job_id = theorem_match.group(1)
    goal = theorem_match.group(2)

    # Check each defect pattern
    for pattern_name, pattern_re in PATTERNS.items():
        if pattern_re.search(goal):
            # Extract 16-byte hex window around first match
            match_start = pattern_re.search(goal).start()
            window_start = max(0, match_start - 8)
            window_end = min(len(goal), match_start + 8)
            hex_window = goal[window_start:window_end].encode("utf-8").hex()

            return "malformed", {
                "pattern": pattern_name,
                "goal": goal,
                "hex_window": hex_window,
                "fix_hint": get_fix_hint(pattern_name, goal),
            }

    # Special case: unprovable goals (single variable without hypothesis)
    if re.match(r"^\s*[pqrst]\s*$", goal):
        return "malformed", {
            "pattern": "unprovable",
            "goal": goal,
            "hex_window": goal.encode("utf-8").hex(),
            "fix_hint": "Goal requires hypothesis providing variable",
        }

    return "valid", {}


def get_fix_hint(pattern_name: str, goal: str) -> str:
    """Generate fix suggestion for defect pattern."""
    hints = {
        "escaped_latex": "Remove LaTeX escaping; emit pure Lean syntax",
        "unicode_escape": f"Replace escape with symbol: {goal} → (render Unicode)",
        "incomplete_brace": "Fix brace pairing or remove malformed set syntax",
        "unprovable": "Ensure goal is derivable from hypotheses",
    }
    return hints.get(pattern_name, "Unknown defect pattern")


def scan_jobs_directory(jobs_dir: Path) -> Dict:
    """
    Scan all .lean files in directory and classify.

    Returns deterministic report dict.
    """
    lean_files = sorted(jobs_dir.glob("*.lean"))

    valid_jobs = []
    malformed_jobs = []

    for lean_file in lean_files:
        job_id = lean_file.stem
        status, defect = classify_job(lean_file)

        if status == "valid":
            valid_jobs.append(job_id)
        else:
            malformed_jobs.append({
                "job_id": job_id,
                "file": str(lean_file),
                **defect
            })

    total = len(lean_files)
    malformed_count = len(malformed_jobs)
    valid_count = len(valid_jobs)

    malformation_rate = malformed_count / total if total > 0 else 0.0

    # Compute pattern distribution
    pattern_counts = {}
    for job in malformed_jobs:
        pattern = job["pattern"]
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

    report = {
        "audit_type": "preflight_lean_jobs",
        "jobs_scanned": total,
        "valid": valid_count,
        "malformed": malformed_count,
        "malformation_rate": round(malformation_rate, 4),
        "status": "PASS" if malformed_count == 0 else "FAIL",
        "valid_jobs": valid_jobs,
        "malformed_jobs": malformed_jobs,
        "pattern_distribution": pattern_counts,
    }

    return report


def main():
    parser = argparse.ArgumentParser(description="Preflight Lean job syntax validator")
    parser.add_argument(
        "--jobs-dir",
        type=Path,
        default=Path("backend/lean_proj/ML/Jobs"),
        help="Directory containing Lean job files (default: backend/lean_proj/ML/Jobs)",
    )
    parser.add_argument(
        "--json",
        type=Path,
        help="Write JSON report to file",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress stdout (only exit code)",
    )

    args = parser.parse_args()

    if not args.jobs_dir.exists():
        print(f"[ABSTAIN] Jobs directory not found: {args.jobs_dir}", file=sys.stderr)
        sys.exit(2)

    report = scan_jobs_directory(args.jobs_dir)

    # Write JSON if requested (deterministic)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(report, f, sort_keys=True, separators=(",", ":"), indent=2)

    # Print summary unless quiet
    if not args.quiet:
        print(f"[{report['status']}] Preflight Lean Jobs")
        print(f"  Scanned: {report['jobs_scanned']} files")
        print(f"  Valid: {report['valid']} ({100 * (1 - report['malformation_rate']):.1f}%)")
        print(f"  Malformed: {report['malformed']} ({100 * report['malformation_rate']:.1f}%)")

        if report['malformed'] > 0:
            print(f"\n  Defect Distribution:")
            for pattern, count in sorted(report['pattern_distribution'].items()):
                print(f"    {pattern}: {count}")

            print(f"\n  First 5 Defects:")
            for job in report['malformed_jobs'][:5]:
                print(f"    {job['job_id']}: {job['pattern']}")
                print(f"      Goal: {job.get('goal', 'N/A')[:60]}...")
                print(f"      Hex:  {job.get('hex_window', 'N/A')[:32]}...")
                print(f"      Fix:  {job.get('fix_hint', 'N/A')}")

    # Exit code: 0 = PASS, 1 = FAIL, 2 = ABSTAIN
    sys.exit(0 if report['status'] == 'PASS' else 1)


if __name__ == "__main__":
    main()
