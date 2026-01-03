#!/usr/bin/env python3
"""
Deploy Mismatch Detector (CI-Grade Anti-Drift Gate)

Verifies that the hosted demo at mathledger.ai/demo/ matches releases.json.
Run this before any outreach, deploy, or external demo.

Usage:
    uv run python tools/check_hosted_demo_matches_release.py
    uv run python tools/check_hosted_demo_matches_release.py --url https://custom.domain/demo/health
    uv run python tools/check_hosted_demo_matches_release.py --ci  # Machine-readable output

Exit codes:
    0: All checks passed
    1: Mismatch detected (deploy required)
    2: Network or file error
    3: Missing release pin info (container rebuild required)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import httpx
except ImportError:
    print("ERROR: httpx not installed. Run: uv pip install httpx")
    sys.exit(2)


REPO_ROOT = Path(__file__).parent.parent
RELEASES_FILE = REPO_ROOT / "releases" / "releases.json"
DEFAULT_DEMO_URL = "https://mathledger.ai/demo/health"


# ---------------------------------------------------------------------------
# Diagnostic Messages (Non-Developer Friendly)
# ---------------------------------------------------------------------------

DIAGNOSTICS = {
    "version_mismatch": {
        "symptom": "Demo version differs from releases.json",
        "cause": "The Fly.io app is running an older version",
        "action": "Redeploy the Fly.io app",
        "command": "fly deploy -a mathledger-demo-v0-2-0-helpfuldolphin",
    },
    "tag_mismatch": {
        "symptom": "Demo tag differs from releases.json",
        "cause": "The Fly.io app is running with wrong tag",
        "action": "Redeploy the Fly.io app",
        "command": "fly deploy -a mathledger-demo-v0-2-0-helpfuldolphin",
    },
    "commit_mismatch": {
        "symptom": "Demo commit differs from releases.json",
        "cause": "The Fly.io app is running from a different commit",
        "action": "Rebuild and redeploy the Docker image",
        "command": "docker build -t mathledger-demo . && fly deploy -a mathledger-demo-v0-2-0-helpfuldolphin",
    },
    "release_pin_stale": {
        "symptom": "Demo reports release_pin.is_stale = true",
        "cause": "The container's releases.json doesn't match the running code",
        "action": "Rebuild Docker image with updated releases.json",
        "command": "docker build --no-cache -t mathledger-demo . && fly deploy",
    },
    "release_pin_missing": {
        "symptom": "Demo /health response missing release_pin field",
        "cause": "The container doesn't include releases.json or demo/app.py is outdated",
        "action": "Verify Dockerfile includes 'COPY releases/' and rebuild",
        "command": "docker build --no-cache -t mathledger-demo . && fly deploy",
    },
    "network_error": {
        "symptom": "Cannot reach hosted demo",
        "cause": "Demo is down or network issue",
        "action": "Check Fly.io app status",
        "command": "fly status -a mathledger-demo-v0-2-0-helpfuldolphin",
    },
}


def print_diagnostic(key: str, expected: Optional[str] = None, actual: Optional[str] = None):
    """Print a human-readable diagnostic message."""
    diag = DIAGNOSTICS.get(key, {})
    print()
    print("┌" + "─" * 58 + "┐")
    print(f"│ ISSUE: {diag.get('symptom', key):<49} │")
    print("├" + "─" * 58 + "┤")
    if expected and actual:
        print(f"│   Expected: {expected:<45} │")
        print(f"│   Actual:   {actual:<45} │")
        print("├" + "─" * 58 + "┤")
    print(f"│ CAUSE: {diag.get('cause', 'Unknown'):<49} │")
    print(f"│ FIX:   {diag.get('action', 'Unknown'):<49} │")
    print("├" + "─" * 58 + "┤")
    cmd = diag.get("command", "")
    if cmd:
        # Wrap long commands
        if len(cmd) > 54:
            print(f"│ $ {cmd[:54]} │")
            print(f"│   {cmd[54:]:<54} │")
        else:
            print(f"│ $ {cmd:<55} │")
    print("└" + "─" * 58 + "┘")


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_releases() -> Dict[str, Any]:
    """Load releases.json and extract current version info."""
    if not RELEASES_FILE.exists():
        print(f"ERROR: {RELEASES_FILE} not found")
        sys.exit(2)

    with open(RELEASES_FILE, encoding="utf-8") as f:
        data = json.load(f)

    current_version = data.get("current_version")
    if not current_version:
        print("ERROR: releases.json missing 'current_version'")
        sys.exit(2)

    version_data = data.get("versions", {}).get(current_version)
    if not version_data:
        print(f"ERROR: releases.json missing version data for '{current_version}'")
        sys.exit(2)

    return {
        "version": current_version.lstrip("v"),  # Normalize: v0.2.1 -> 0.2.1
        "version_raw": current_version,
        "commit": version_data.get("commit"),
        "tag": version_data.get("tag"),
    }


def fetch_hosted_health(url: str) -> Dict[str, Any]:
    """Fetch /health JSON from hosted demo."""
    try:
        with httpx.Client(timeout=10.0, follow_redirects=True) as client:
            response = client.get(url)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        print(f"ERROR: HTTP {e.response.status_code} from {url}")
        print_diagnostic("network_error")
        sys.exit(2)
    except httpx.RequestError as e:
        print(f"ERROR: Network error fetching {url}: {e}")
        print_diagnostic("network_error")
        sys.exit(2)
    except json.JSONDecodeError:
        print(f"ERROR: Invalid JSON response from {url}")
        sys.exit(2)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def check_all(releases: Dict[str, Any], hosted: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Compare releases.json with hosted demo /health response.

    Returns (all_passed, list_of_issues).
    """
    issues = []

    # 1. Check version
    expected_version = releases.get("version")
    hosted_version = (hosted.get("version") or "").lstrip("v")

    if expected_version != hosted_version:
        issues.append("version_mismatch")
        print(f"✗ VERSION MISMATCH")
        print(f"    Expected: {expected_version}")
        print(f"    Hosted:   {hosted_version or '(missing)'}")
    else:
        print(f"✓ Version: {expected_version}")

    # 2. Check tag
    expected_tag = releases.get("tag")
    hosted_tag = hosted.get("tag") or hosted.get("build_tag")

    if expected_tag != hosted_tag:
        issues.append("tag_mismatch")
        print(f"✗ TAG MISMATCH")
        print(f"    Expected: {expected_tag}")
        print(f"    Hosted:   {hosted_tag or '(missing)'}")
    else:
        print(f"✓ Tag: {expected_tag}")

    # 3. Check commit
    expected_commit = releases.get("commit")
    hosted_commit = hosted.get("commit") or hosted.get("build_commit")

    if expected_commit != hosted_commit:
        issues.append("commit_mismatch")
        print(f"✗ COMMIT MISMATCH")
        print(f"    Expected: {expected_commit[:12] if expected_commit else '(none)'}...")
        print(f"    Hosted:   {hosted_commit[:12] if hosted_commit else '(missing)'}...")
    else:
        print(f"✓ Commit: {expected_commit[:12]}...")

    # 4. Check release_pin (NEW: structural anti-drift)
    release_pin = hosted.get("release_pin")
    if release_pin is None:
        issues.append("release_pin_missing")
        print(f"✗ RELEASE PIN MISSING")
        print(f"    /health response does not include 'release_pin' field")
        print(f"    This means the container may not have releases.json bundled")
    elif release_pin.get("is_stale") is True:
        issues.append("release_pin_stale")
        print(f"✗ RELEASE PIN STALE")
        print(f"    Container reports version mismatch with bundled releases.json")
        print(f"    Mismatch fields: {release_pin.get('mismatch_fields', [])}")
    else:
        print(f"✓ Release pin: valid (is_stale=false)")

    # 5. Check overall status
    status = hosted.get("status", "unknown")
    if status == "FAIL_STALE_DEPLOY":
        if "release_pin_stale" not in issues:
            issues.append("release_pin_stale")
        print(f"✗ STATUS: {status}")
    elif status != "ok":
        print(f"⚠ STATUS: {status} (expected 'ok')")
    else:
        print(f"✓ Status: ok")

    return len(issues) == 0, issues


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Verify hosted demo matches releases.json (CI-grade anti-drift gate)"
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_DEMO_URL,
        help=f"URL to fetch /health from (default: {DEFAULT_DEMO_URL})",
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        help="Machine-readable output (JSON)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print on failure",
    )
    args = parser.parse_args()

    # Load canonical source
    releases = load_releases()

    # Fetch hosted demo
    hosted = fetch_hosted_health(args.url)

    # CI mode: JSON output
    if args.ci:
        all_passed, issues = check_all(releases, hosted)
        result = {
            "passed": all_passed,
            "issues": issues,
            "expected": {
                "version": releases["version"],
                "tag": releases["tag"],
                "commit": releases["commit"],
            },
            "hosted": {
                "version": hosted.get("version"),
                "tag": hosted.get("tag"),
                "commit": hosted.get("commit"),
                "status": hosted.get("status"),
                "release_pin": hosted.get("release_pin"),
            },
        }
        print(json.dumps(result, indent=2))
        sys.exit(0 if all_passed else 1)

    # Interactive mode
    if not args.quiet:
        print()
        print("╔" + "═" * 58 + "╗")
        print("║" + " HOSTED DEMO ANTI-DRIFT CHECK ".center(58) + "║")
        print("╚" + "═" * 58 + "╝")
        print()
        print(f"Source: {RELEASES_FILE.name}")
        print(f"Target: {args.url}")
        print()
        print("─" * 60)

    all_passed, issues = check_all(releases, hosted)

    print("─" * 60)
    print()

    if all_passed:
        print("╔" + "═" * 58 + "╗")
        print("║" + " ✓ PASS - Hosted demo matches releases.json ".center(58) + "║")
        print("╚" + "═" * 58 + "╝")
        print()
        print("The hosted demo is correctly pinned. Safe to proceed.")
        sys.exit(0)
    else:
        print("╔" + "═" * 58 + "╗")
        print("║" + " ✗ FAIL - Mismatch detected! ".center(58) + "║")
        print("╚" + "═" * 58 + "╝")
        print()
        print(f"Issues found: {len(issues)}")

        # Print diagnostics for each issue
        for issue in issues:
            expected = releases.get(issue.replace("_mismatch", ""), "")
            actual = hosted.get(issue.replace("_mismatch", ""), "")
            print_diagnostic(issue, str(expected) if expected else None, str(actual) if actual else None)

        # Exit with appropriate code
        if "release_pin_missing" in issues:
            print()
            print("Exit code 3: Container rebuild required (releases.json not bundled)")
            sys.exit(3)
        else:
            print()
            print("Exit code 1: Deploy required")
            sys.exit(1)


if __name__ == "__main__":
    main()
