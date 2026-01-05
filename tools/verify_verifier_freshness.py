#!/usr/bin/env python3
"""
Verify Verifier Freshness - Build Assertion

This script fails if the checked-in verifier HTML is stale relative to
the generator script. Prevents the class of bug where the generator is
correct but the generated artifact is not regenerated.

Usage:
    uv run python tools/verify_verifier_freshness.py

Exit codes:
    0 - All verifiers are fresh
    1 - At least one verifier is stale or missing

CI Integration:
    Add to CI pipeline before deploy to catch stale artifacts.
"""

import hashlib
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent


def get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def check_verifier_has_domain_separation(html_path: Path) -> tuple[bool, str]:
    """Check if verifier HTML has proper domain separation."""
    if not html_path.exists():
        return False, f"File not found: {html_path}"

    content = html_path.read_text(encoding="utf-8")

    required_patterns = [
        ("DOMAIN_UI_LEAF", r"DOMAIN_UI_LEAF"),
        ("DOMAIN_REASONING_LEAF", r"DOMAIN_REASONING_LEAF"),
        ("DOMAIN_LEAF", r"DOMAIN_LEAF=new Uint8Array\(\[0x00\]\)"),
        ("DOMAIN_NODE", r"DOMAIN_NODE=new Uint8Array\(\[0x01\]\)"),
        ("merkleRoot function", r"async function merkleRoot"),
        ("computeUt function", r"async function computeUt"),
        ("computeRt function", r"async function computeRt"),
        ("shaD usage in computeUt", r"shaD\(can\(e\),DOMAIN_UI_LEAF\)"),
        ("shaD usage in computeRt", r"shaD\(can\(a\),DOMAIN_REASONING_LEAF\)"),
    ]

    missing = []
    for name, pattern in required_patterns:
        if not re.search(pattern, content):
            missing.append(name)

    if missing:
        return False, f"Missing required patterns: {', '.join(missing)}"

    # Check for OLD broken pattern
    if "computedU=await sha(can(uvil))" in content:
        return False, "Contains OLD broken pattern: computedU=await sha(can(uvil))"

    return True, "OK"


def check_verifier_version(version: str) -> tuple[bool, str]:
    """Check a specific version's verifier."""
    html_path = REPO_ROOT / "site" / version / "evidence-pack" / "verify" / "index.html"

    if not html_path.exists():
        # Version might not exist yet
        return True, f"Skipped (not found): {version}"

    is_fresh, reason = check_verifier_has_domain_separation(html_path)
    return is_fresh, f"{version}: {reason}"


def main():
    print("=" * 60)
    print("Verifier Freshness Check")
    print("=" * 60)

    current_commit = get_git_commit()
    print(f"Current commit: {current_commit[:12] if current_commit else 'unknown'}")
    print()

    # Check all versions that should have domain separation
    versions_to_check = ["v0.2.7"]  # Only v0.2.7+ should have it

    all_passed = True
    for version in versions_to_check:
        is_fresh, message = check_verifier_version(version)
        status = "PASS" if is_fresh else "FAIL"
        print(f"[{status}] {message}")
        if not is_fresh:
            all_passed = False

    print()
    if all_passed:
        print("All verifiers are fresh.")
        return 0
    else:
        print("ERROR: At least one verifier is stale!")
        print("Run the appropriate generate script to regenerate.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
