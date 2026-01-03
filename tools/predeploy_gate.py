#!/usr/bin/env python3
"""
Pre-Deploy Gate: Verify all conditions before deployment.

Implements the Deploy-by-Tag Doctrine (docs/DEPLOY_BY_TAG_DOCTRINE.md).

Usage:
    uv run python tools/predeploy_gate.py vX.Y.Z-*
    uv run python tools/predeploy_gate.py --check-health  # Also verify /demo/health

Exit codes:
    0 = GO (all checks pass)
    1 = NO-GO (one or more checks failed)
    2 = Usage error

Checks performed:
    1. Git status is clean (no uncommitted changes)
    2. HEAD exactly matches the target tag
    3. Tag exists on origin (GitHub)
    4. releases.json current_version matches tag
    5. releases.json commit matches tag's commit
    6. (optional) /demo/health matches tag/commit
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


REPO_ROOT = Path(__file__).parent.parent
RELEASES_JSON = REPO_ROOT / "releases" / "releases.json"
DEMO_HEALTH_URL = "https://mathledger.ai/demo/health"


def run_git(args: list[str], capture: bool = True) -> tuple[int, str]:
    """Run a git command and return (exit_code, stdout)."""
    cmd = ["git", "-C", str(REPO_ROOT)] + args
    if capture:
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode, result.stdout.strip()
    else:
        result = subprocess.run(cmd)
        return result.returncode, ""


def check_clean_status() -> tuple[bool, str]:
    """Check that git status is clean."""
    code, output = run_git(["status", "--porcelain"])
    if code != 0:
        return False, f"git status failed with code {code}"
    if output:
        return False, f"Working tree is dirty:\n{output}"
    return True, "Git status is clean"


def check_exact_tag_match(target_tag: str) -> tuple[bool, str]:
    """Check that HEAD exactly matches the target tag."""
    code, output = run_git(["describe", "--tags", "--exact-match", "HEAD"])
    if code != 0:
        return False, "HEAD is not an exact tag match"
    if output != target_tag:
        return False, f"HEAD tag is '{output}', expected '{target_tag}'"
    return True, f"HEAD exactly matches tag '{target_tag}'"


def check_tag_on_origin(target_tag: str) -> tuple[bool, str]:
    """Check that the tag exists on origin."""
    code, output = run_git(["ls-remote", "--tags", "origin", target_tag])
    if code != 0:
        return False, f"git ls-remote failed with code {code}"
    if not output:
        return False, f"Tag '{target_tag}' does not exist on origin"
    return True, f"Tag '{target_tag}' exists on origin"


def get_tag_commit(target_tag: str) -> Optional[str]:
    """Get the commit hash that a tag points to."""
    code, output = run_git(["rev-parse", f"{target_tag}^{{commit}}"])
    if code != 0:
        return None
    return output


def check_releases_json(target_tag: str) -> tuple[bool, str]:
    """Check that releases.json matches the target tag."""
    if not RELEASES_JSON.exists():
        return False, f"releases.json not found at {RELEASES_JSON}"

    try:
        with open(RELEASES_JSON, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return False, f"releases.json is not valid JSON: {e}"

    current_version = data.get("current_version")
    if current_version != target_tag:
        return False, (
            f"releases.json current_version is '{current_version}', "
            f"expected '{target_tag}'"
        )

    versions = data.get("versions", {})
    if target_tag not in versions:
        return False, f"releases.json has no entry for '{target_tag}'"

    version_entry = versions[target_tag]
    declared_tag = version_entry.get("tag")
    declared_commit = version_entry.get("commit")

    if declared_tag != target_tag:
        return False, (
            f"releases.json versions[{target_tag}].tag is '{declared_tag}', "
            f"expected '{target_tag}'"
        )

    # Check commit matches
    actual_commit = get_tag_commit(target_tag)
    if actual_commit is None:
        return False, f"Could not get commit for tag '{target_tag}'"

    if declared_commit != actual_commit:
        return False, (
            f"releases.json commit is '{declared_commit}', "
            f"but tag points to '{actual_commit}'"
        )

    return True, (
        f"releases.json matches: current_version={target_tag}, "
        f"commit={declared_commit[:12]}..."
    )


def check_demo_health(target_tag: str) -> tuple[bool, str]:
    """Check that /demo/health matches the tag (optional, requires requests)."""
    if not HAS_REQUESTS:
        return True, "(Skipped: requests library not available)"

    try:
        resp = requests.get(DEMO_HEALTH_URL, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return False, f"Failed to fetch {DEMO_HEALTH_URL}: {e}"

    health_tag = data.get("tag")
    health_commit = data.get("commit")
    release_pin = data.get("release_pin", {})
    is_stale = release_pin.get("is_stale", True)

    errors = []

    if health_tag != target_tag:
        errors.append(f"tag is '{health_tag}', expected '{target_tag}'")

    expected_commit = get_tag_commit(target_tag)
    if expected_commit and health_commit != expected_commit:
        errors.append(
            f"commit is '{health_commit}', expected '{expected_commit}'"
        )

    if is_stale:
        errors.append("release_pin.is_stale is true")

    if errors:
        return False, f"/demo/health mismatch: {'; '.join(errors)}"

    return True, f"/demo/health matches tag '{target_tag}'"


def main():
    parser = argparse.ArgumentParser(
        description="Pre-deploy gate: verify all conditions before deployment."
    )
    parser.add_argument(
        "tag",
        nargs="?",
        help="Target tag to verify (e.g., v0.2.3-audit-path)"
    )
    parser.add_argument(
        "--check-health",
        action="store_true",
        help="Also verify /demo/health endpoint"
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        help="Machine-readable JSON output"
    )
    args = parser.parse_args()

    if not args.tag:
        # Try to get current tag from HEAD
        code, tag = run_git(["describe", "--tags", "--exact-match", "HEAD"])
        if code != 0:
            print("ERROR: No tag specified and HEAD is not a tag")
            print("Usage: uv run python tools/predeploy_gate.py <tag>")
            sys.exit(2)
        args.tag = tag
        print(f"Using tag from HEAD: {args.tag}")

    print("=" * 60)
    print("PRE-DEPLOY GATE")
    print(f"Target tag: {args.tag}")
    print("=" * 60)
    print()

    checks = [
        ("Git status clean", lambda: check_clean_status()),
        ("HEAD matches tag", lambda: check_exact_tag_match(args.tag)),
        ("Tag exists on origin", lambda: check_tag_on_origin(args.tag)),
        ("releases.json matches", lambda: check_releases_json(args.tag)),
    ]

    if args.check_health:
        checks.append(
            ("/demo/health matches", lambda: check_demo_health(args.tag))
        )

    results = []
    all_passed = True

    for name, check_fn in checks:
        passed, message = check_fn()
        results.append({"name": name, "passed": passed, "message": message})
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {name}")
        print(f"       {message}")
        print()
        if not passed:
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("VERDICT: GO")
        print("All pre-deploy checks passed.")
        print()
        print("Deploy commands:")
        print(f"  git checkout {args.tag}")
        print("  wrangler pages deploy site/")
        print("  fly deploy")
        exit_code = 0
    else:
        print("VERDICT: NO-GO")
        print("One or more checks failed. Do not deploy.")
        exit_code = 1
    print("=" * 60)

    if args.ci:
        ci_output = {
            "tag": args.tag,
            "verdict": "GO" if all_passed else "NO-GO",
            "checks": results
        }
        print()
        print("CI OUTPUT:")
        print(json.dumps(ci_output, indent=2))

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
