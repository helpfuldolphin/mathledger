#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Substrate Governance Check Script

This script validates a local global_health.json file against the substrate
governance schema and checks its status.
"""

import argparse
import json
import sys
from jsonschema import validate, ValidationError

# Exit codes as per specification
EXIT_CODES = {
    "OK": 0,
    "GENERAL_ERROR": 1,
    "RED": 64,
    "BLOCK": 65,
}

SCHEMA_PATH = "docs/governance/substrate/substrate_schema_draft07.json"

def get_parser():
    """Configures and returns the argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "health_json_path",
        metavar="GLOBAL_HEALTH_JSON",
        help="Path to the global_health.json file to validate.",
    )
    parser.add_argument(
        "--shadow-only",
        action="store_true",
        default=True,
        help="Run in shadow-only mode. Always exits 0 but prints advisories. (Default: True)",
    )
    parser.add_argument(
        "--no-shadow-only",
        action="store_false",
        dest="shadow_only",
        help="Disable shadow-only mode to enforce hard failures with non-zero exit codes.",
    )
    parser.add_argument(
        "--schema-path",
        default=SCHEMA_PATH,
        help=f"Path to the substrate schema file. (Default: {SCHEMA_PATH})",
    )
    return parser

def main():
    """Main execution function."""
    parser = get_parser()
    args = parser.parse_args()

    try:
        with open(args.schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)
        with open(args.health_json_path, "r", encoding="utf-8") as f:
            health_data = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}", file=sys.stderr)
        sys.exit(EXIT_CODES["GENERAL_ERROR"])
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON - {e}", file=sys.stderr)
        sys.exit(EXIT_CODES["GENERAL_ERROR"])

    try:
        validate(instance=health_data, schema=schema)
    except ValidationError as e:
        print(f"Schema validation failed: {e.message}", file=sys.stderr)
        sys.exit(EXIT_CODES["GENERAL_ERROR"])

    substrate = health_data.get("substrate", {})
    status = substrate.get("status_light", "GREEN")
    headline = substrate.get("headline", "N/A")
    drift_flags = substrate.get("drift_flags", [])
    details_url = substrate.get("details_url", "N/A")

    exit_code = EXIT_CODES["OK"]
    is_failure_state = False

    if status == "RED":
        exit_code = EXIT_CODES["RED"]
        is_failure_state = True
        msg = f"""
SUBSTRATE GOVERNANCE CHECK: FAILED (RED)
REASON: Significant substrate drift detected.
HEADLINE: {headline}
DRIFT FLAGS: {drift_flags}
Deployment is blocked pending security review. See details at: {details_url}
"""
    elif status == "BLOCK":
        exit_code = EXIT_CODES["BLOCK"]
        is_failure_state = True
        msg = f"""
SUBSTRATE GOVERNANCE CHECK: FAILED (BLOCK) - CRITICAL SECURITY ALERT
REASON: Critical substrate integrity failure.
HEADLINE: {headline}
DRIFT FLAGS: {drift_flags}
CI GATE IS HARD-LOCKED. IMMEDIATE MANUAL INTERVENTION REQUIRED.
See alert details at: {details_url}
"""

    if is_failure_state and args.shadow_only:
        print("--- SHADOW-ONLY MODE ENABLED ---", file=sys.stderr)
        print(msg.strip(), file=sys.stderr)
        print("--- EXITING 0 DUE TO SHADOW-ONLY MODE ---", file=sys.stderr)
        sys.exit(EXIT_CODES["OK"])
    elif is_failure_state:
        print(msg.strip(), file=sys.stderr)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
