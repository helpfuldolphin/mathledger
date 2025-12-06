#!/usr/bin/env python3
"""
Whitelist Audit - Verify whitelist entries have rationale and expiry dates.

Audits artifacts/repro/drift_whitelist.json to ensure all whitelisted files
have documented rationale and expiry dates. Prints ABSTAIN if any entries
lack both rationale and expiry.

Usage:
    python tools/repro/whitelist_audit.py --check
    python tools/repro/whitelist_audit.py --report

Exit Codes:
    0: Success (all entries have rationale and expiry)
    1: Missing rationale or expiry (ABSTAIN)
    2: Invalid whitelist file or missing
"""

import argparse
import json
import os
import sys
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from backend.repro.determinism import deterministic_timestamp


def get_repo_root() -> Path:
    """Get repository root directory."""
    current = Path(__file__).resolve()
    while current.parent != current:
        if (current / ".git").exists():
            return current
        current = current.parent
    raise RuntimeError("Could not find repository root (no .git directory)")


def load_whitelist(whitelist_path: Path) -> Dict:
    """Load whitelist JSON file."""
    if not whitelist_path.exists():
        raise FileNotFoundError(f"Whitelist not found: {whitelist_path}")
    
    with open(whitelist_path, "r") as f:
        return json.load(f)


def check_rationale(whitelist: Dict) -> List[str]:
    """
    Check if all whitelisted files have rationale.
    
    Returns:
        List of files missing rationale
    """
    missing = []
    
    whitelisted_files = whitelist.get("whitelist", [])
    rationale = whitelist.get("rationale", {})
    
    for file in whitelisted_files:
        if file not in rationale or not rationale[file]:
            missing.append(file)
    
    return missing


def check_expiry(whitelist: Dict) -> List[str]:
    """
    Check if all whitelisted files have expiry dates.
    
    Returns:
        List of files missing expiry
    """
    missing = []
    
    whitelisted_files = whitelist.get("whitelist", [])
    expiry = whitelist.get("expiry", {})
    
    for file in whitelisted_files:
        if file not in expiry or not expiry[file]:
            missing.append(file)
    
    return missing


def check_code_owner(whitelist: Dict) -> List[str]:
    """
    Check if all whitelisted files have code owners.
    
    Returns:
        List of files missing code owner
    """
    missing = []
    
    whitelisted_files = whitelist.get("whitelist", [])
    code_owners = whitelist.get("code_owners", {})
    
    for file in whitelisted_files:
        if file not in code_owners or not code_owners[file]:
            missing.append(file)
    
    return missing


def check_expired(whitelist: Dict, today: date) -> List[Tuple[str, str]]:
    """
    Check if any whitelisted files have expired.
    
    Returns:
        List of (file, expiry_date) tuples for expired entries
    """
    expired = []
    
    whitelisted_files = whitelist.get("whitelist", [])
    expiry = whitelist.get("expiry", {})
    
    for file in whitelisted_files:
        if file in expiry and expiry[file]:
            try:
                expiry_date = datetime.strptime(expiry[file], "%Y-%m-%d").date()
                if expiry_date < today:
                    expired.append((file, expiry[file]))
            except ValueError:
                pass
    
    return expired


def check_expiring_soon(whitelist: Dict, today: date, days: int = 14) -> List[Tuple[str, str, int]]:
    """
    Check if any whitelisted files are expiring within specified days.
    
    Returns:
        List of (file, expiry_date, days_remaining) tuples
    """
    expiring = []
    
    whitelisted_files = whitelist.get("whitelist", [])
    expiry = whitelist.get("expiry", {})
    
    for file in whitelisted_files:
        if file in expiry and expiry[file]:
            try:
                expiry_date = datetime.strptime(expiry[file], "%Y-%m-%d").date()
                days_remaining = (expiry_date - today).days
                if 0 <= days_remaining <= days:
                    expiring.append((file, expiry[file], days_remaining))
            except ValueError:
                pass
    
    return expiring


def resolve_audit_date(as_of: Optional[str]) -> date:
    """Resolve deterministic audit date."""
    candidate = as_of or os.environ.get("ML_WHITELIST_AS_OF")
    if candidate:
        return datetime.strptime(candidate, "%Y-%m-%d").date()
    return deterministic_timestamp().date()


def print_audit_report(
    whitelist: Dict,
    missing_rationale: List[str],
    missing_expiry: List[str],
    missing_code_owner: List[str],
    expired: List[Tuple[str, str]],
    expiring_soon: List[Tuple[str, str, int]]
):
    """Print comprehensive audit report."""
    print("=" * 80)
    print("WHITELIST AUDIT REPORT")
    print("=" * 80)
    
    whitelisted_files = whitelist.get("whitelist", [])
    print(f"\nTotal whitelisted files: {len(whitelisted_files)}")
    
    print("\n" + "-" * 80)
    print("RATIONALE CHECK")
    print("-" * 80)
    
    if missing_rationale:
        print(f"\n[FAIL] {len(missing_rationale)} files missing rationale:")
        for file in missing_rationale:
            print(f"  - {file}")
    else:
        print("\n[PASS] All files have rationale")
    
    print("\n" + "-" * 80)
    print("EXPIRY CHECK")
    print("-" * 80)
    
    if missing_expiry:
        print(f"\n[FAIL] {len(missing_expiry)} files missing expiry:")
        for file in missing_expiry:
            print(f"  - {file}")
    else:
        print("\n[PASS] All files have expiry dates")
    
    print("\n" + "-" * 80)
    print("CODE OWNER CHECK")
    print("-" * 80)
    
    if missing_code_owner:
        print(f"\n[FAIL] {len(missing_code_owner)} files missing code owner:")
        for file in missing_code_owner:
            print(f"  - {file}")
    else:
        print("\n[PASS] All files have code owners")
    
    print("\n" + "-" * 80)
    print("EXPIRATION CHECK")
    print("-" * 80)
    
    if expired:
        print(f"\n[WARN] {len(expired)} files have expired:")
        for file, expiry_date in expired:
            print(f"  - {file} (expired: {expiry_date})")
        print("\nExpired entries should be reviewed and either:")
        print("  1. Removed from whitelist if no longer needed")
        print("  2. Extended with new expiry date if still required")
    else:
        print("\n[PASS] No expired entries")
    
    print("\n" + "-" * 80)
    print("EXPIRING SOON CHECK (< 14 days)")
    print("-" * 80)
    
    if expiring_soon:
        print(f"\n[WARN] {len(expiring_soon)} files expiring within 14 days:")
        for file, expiry_date, days_remaining in sorted(expiring_soon, key=lambda x: x[2]):
            print(f"  - {file} (expires: {expiry_date}, {days_remaining} days remaining)")
        print("\nExpiring entries should be reviewed proactively:")
        print("  1. Verify file still requires nondeterministic operations")
        print("  2. Extend expiry date if still needed")
        print("  3. Remove from whitelist if no longer required")
    else:
        print("\n[PASS] No entries expiring within 14 days")
    
    print("\n" + "=" * 80)
    print("AUDIT SUMMARY")
    print("=" * 80)
    
    total_issues = len(missing_rationale) + len(missing_expiry) + len(missing_code_owner)
    
    if total_issues > 0:
        print(f"\n[ABSTAIN] Whitelist audit failed: {total_issues} issues detected")
        print("\nAll whitelisted files MUST have:")
        print("  1. Documented rationale explaining why nondeterministic operations are needed")
        print("  2. Expiry date for periodic review (format: YYYY-MM-DD)")
        print("  3. Code owner responsible for reviewing and maintaining the whitelist entry")
        print("\nUpdate artifacts/repro/drift_whitelist.json to resolve issues.")
    else:
        print("\n[PASS] Whitelist audit passed: 0 missing rationales/expiries/code_owners")
        
        if expired or expiring_soon:
            total_warnings = len(expired) + len(expiring_soon)
            print(f"\n[WARN] {total_warnings} entries require review (expired or expiring soon)")
    
    print()


def print_missing_entries(missing_rationale: List[str], missing_expiry: List[str]):
    """Print concise list of missing entries."""
    if missing_rationale:
        print("\nMissing rationale:")
        for file in missing_rationale:
            print(f"  - {file}")
    
    if missing_expiry:
        print("\nMissing expiry:")
        for file in missing_expiry:
            print(f"  - {file}")


def main():
    parser = argparse.ArgumentParser(
        description="Whitelist audit - Verify rationale and expiry dates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/repro/whitelist_audit.py --check
  
  python tools/repro/whitelist_audit.py --report
        """
    )
    
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check whitelist and print summary"
    )
    
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate detailed audit report"
    )
    
    parser.add_argument(
        "--as-of-date",
        type=str,
        help="Override audit date (YYYY-MM-DD); defaults to deterministic epoch"
    )
    
    args = parser.parse_args()
    
    if not args.check and not args.report:
        args.check = True
    
    try:
        repo_root = get_repo_root()
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    
    whitelist_path = repo_root / "artifacts" / "repro" / "drift_whitelist.json"
    
    try:
        whitelist = load_whitelist(whitelist_path)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in whitelist: {e}", file=sys.stderr)
        return 2
    
    missing_rationale = check_rationale(whitelist)
    missing_expiry = check_expiry(whitelist)
    missing_code_owner = check_code_owner(whitelist)
    today = resolve_audit_date(args.as_of_date)
    expired = check_expired(whitelist, today)
    expiring_soon = check_expiring_soon(whitelist, today, days=14)
    
    if args.report:
        print_audit_report(whitelist, missing_rationale, missing_expiry, missing_code_owner, expired, expiring_soon)
    else:
        total_issues = len(missing_rationale) + len(missing_expiry) + len(missing_code_owner)
        
        if total_issues > 0:
            print(f"[ABSTAIN] Whitelist audit failed: {total_issues} issues detected")
            print_missing_entries(missing_rationale, missing_expiry)
            if missing_code_owner:
                print("\nMissing code owner:")
                for file in missing_code_owner:
                    print(f"  - {file}")
            print("\nRun with --report for detailed audit report")
            return 1
        else:
            print("[PASS] Whitelist audit passed: 0 missing rationales/expiries/code_owners")
            
            if expired or expiring_soon:
                total_warnings = len(expired) + len(expiring_soon)
                print(f"\n[WARN] {total_warnings} entries require review (expired or expiring soon)")
                print("Run with --report for details")
    
    if len(missing_rationale) + len(missing_expiry) + len(missing_code_owner) > 0:
        return 1
    else:
        return 0


if __name__ == "__main__":
    sys.exit(main())
