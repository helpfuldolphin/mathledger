#!/usr/bin/env python3
"""
Coverage Badge Generator
Creates a coverage badge JSON file for shields.io endpoint consumption.
"""
import json
import subprocess
import sys
import os
from typing import Dict, Any

def get_coverage_percentage() -> float:
    """Get coverage percentage from coverage report."""
    try:
        result = subprocess.run(
            ["coverage", "report", "--format=json"],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            print(f"Coverage report failed: {result.stderr}", file=sys.stderr)
            return 0.0

        coverage_data = json.loads(result.stdout)
        return round(coverage_data.get("totals", {}).get("percent_covered", 0.0), 1)

    except subprocess.TimeoutExpired:
        print("Coverage report timed out", file=sys.stderr)
        return 0.0
    except json.JSONDecodeError as e:
        print(f"Failed to parse coverage JSON: {e}", file=sys.stderr)
        return 0.0
    except Exception as e:
        print(f"Error getting coverage: {e}", file=sys.stderr)
        return 0.0

def generate_coverage_badge(coverage_percent: float) -> Dict[str, Any]:
    """Generate coverage badge data for shields.io."""
    if coverage_percent >= 90:
        color = "brightgreen"
    elif coverage_percent >= 80:
        color = "green"
    elif coverage_percent >= 70:
        color = "yellow"
    elif coverage_percent >= 50:
        color = "orange"
    else:
        color = "red"

    return {
        "schemaVersion": 1,
        "label": "coverage",
        "message": f"{coverage_percent}%",
        "color": color
    }

def main():
    """Main function to generate coverage badge."""
    print("Generating coverage badge...")

    output_dir = "artifacts/badges"
    os.makedirs(output_dir, exist_ok=True)

    coverage_percent = get_coverage_percentage()
    print(f"Coverage: {coverage_percent}%")

    badge_data = generate_coverage_badge(coverage_percent)

    output_path = os.path.join(output_dir, "coverage_badge.json")
    with open(output_path, 'w') as f:
        json.dump(badge_data, f, indent=2)

    print(f"Coverage badge created: {output_path}")
    print(f"Badge data: {badge_data}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
