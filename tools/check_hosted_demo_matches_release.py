#!/usr/bin/env python3
"""
Deploy Mismatch Detector

Verifies that the hosted demo at mathledger.ai/demo/ matches releases.json.
Run this before any outreach or external demo.

Usage:
    uv run python tools/check_hosted_demo_matches_release.py
    uv run python tools/check_hosted_demo_matches_release.py --url https://custom.domain/demo/

Exit codes:
    0: All checks passed
    1: Mismatch detected
    2: Network or file error
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import httpx
except ImportError:
    print("ERROR: httpx not installed. Run: uv pip install httpx")
    sys.exit(2)


REPO_ROOT = Path(__file__).parent.parent
RELEASES_FILE = REPO_ROOT / "releases" / "releases.json"
DEFAULT_DEMO_URL = "https://mathledger.ai/demo/health"


def load_releases() -> dict:
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
        "version": current_version,
        "commit": version_data.get("commit"),
        "tag": version_data.get("tag"),
    }


def fetch_hosted_health(url: str) -> dict:
    """Fetch /health JSON from hosted demo."""
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(url)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        print(f"ERROR: HTTP {e.response.status_code} from {url}")
        sys.exit(2)
    except httpx.RequestError as e:
        print(f"ERROR: Network error fetching {url}: {e}")
        sys.exit(2)
    except json.JSONDecodeError:
        print(f"ERROR: Invalid JSON response from {url}")
        sys.exit(2)


def check_match(releases: dict, hosted: dict) -> bool:
    """Compare releases.json with hosted demo /health response."""
    all_match = True

    # Check commit
    expected_commit = releases.get("commit")
    hosted_commit = hosted.get("commit") or hosted.get("build_commit")

    if expected_commit != hosted_commit:
        print(f"MISMATCH: commit")
        print(f"  Expected (releases.json): {expected_commit}")
        print(f"  Hosted (/health):         {hosted_commit}")
        all_match = False
    else:
        print(f"OK: commit = {expected_commit[:12]}...")

    # Check tag
    expected_tag = releases.get("tag")
    hosted_tag = hosted.get("tag") or hosted.get("build_tag")

    if expected_tag != hosted_tag:
        print(f"MISMATCH: tag")
        print(f"  Expected (releases.json): {expected_tag}")
        print(f"  Hosted (/health):         {hosted_tag}")
        all_match = False
    else:
        print(f"OK: tag = {expected_tag}")

    # Check version (strip leading 'v' for comparison if needed)
    expected_version = releases.get("version")
    hosted_version = hosted.get("version")

    # Normalize: v0.2.0 == 0.2.0
    expected_norm = expected_version.lstrip("v") if expected_version else ""
    hosted_norm = hosted_version.lstrip("v") if hosted_version else ""

    if expected_norm != hosted_norm:
        print(f"MISMATCH: version")
        print(f"  Expected (releases.json): {expected_version}")
        print(f"  Hosted (/health):         {hosted_version}")
        all_match = False
    else:
        print(f"OK: version = {expected_version}")

    return all_match


def main():
    parser = argparse.ArgumentParser(
        description="Verify hosted demo matches releases.json"
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_DEMO_URL,
        help=f"URL to fetch /health from (default: {DEFAULT_DEMO_URL})",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print on mismatch",
    )
    args = parser.parse_args()

    if not args.quiet:
        print("=" * 60)
        print("Deploy Mismatch Detector")
        print("=" * 60)
        print()

    # Load canonical source
    if not args.quiet:
        print(f"Loading: {RELEASES_FILE}")
    releases = load_releases()

    if not args.quiet:
        print(f"Current version: {releases['version']}")
        print(f"Expected commit: {releases['commit'][:12]}...")
        print(f"Expected tag:    {releases['tag']}")
        print()

    # Fetch hosted demo
    if not args.quiet:
        print(f"Fetching: {args.url}")
    hosted = fetch_hosted_health(args.url)

    if not args.quiet:
        print()
        print("Comparing...")
        print("-" * 40)

    # Compare
    if check_match(releases, hosted):
        if not args.quiet:
            print()
            print("=" * 60)
            print("RESULT: PASS - Hosted demo matches releases.json")
            print("=" * 60)
        sys.exit(0)
    else:
        print()
        print("=" * 60)
        print("RESULT: FAIL - Mismatch detected!")
        print()
        print("The hosted demo does not match releases.json.")
        print("This means:")
        print("  - Fly.io app was not redeployed after code change, OR")
        print("  - Wrong app is deployed, OR")
        print("  - releases.json was updated without redeploying")
        print()
        print("To fix:")
        print("  fly deploy -a mathledger-demo-v0-2-0-helpfuldolphin")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
