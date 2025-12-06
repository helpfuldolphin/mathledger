#!/usr/bin/env python3
"""
Branch naming guard for local development.

Validates that the current branch name follows the project's naming convention:
^(feature|perf|ops|qa|devxp|docs)/[a-z0-9][a-z0-9\-]+$

This script is designed to be run locally and integrated into make targets.
"""

import re
import subprocess
import sys
from typing import Optional


def get_current_branch() -> Optional[str]:
    """Get the current git branch name."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error getting current branch: {e}", file=sys.stderr)
        return None
    except FileNotFoundError:
        print("Error: git command not found. Make sure git is installed.", file=sys.stderr)
        return None


def validate_branch_name(branch_name: str) -> tuple[bool, str]:
    """
    Validate branch name against the project convention.

    Args:
        branch_name: The branch name to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not branch_name:
        return False, "Empty branch name"

    # Pattern: ^(feature|perf|ops|qa|devxp|docs)/[a-z0-9][a-z0-9\-]+$
    pattern = r'^(feature|perf|ops|qa|devxp|docs)/[a-z0-9][a-z0-9\-]+$'

    if not re.match(pattern, branch_name):
        return False, (
            f"Branch name '{branch_name}' does not match required pattern.\n"
            f"Expected: ^(feature|perf|ops|qa|devxp|docs)/[a-z0-9][a-z0-9\-]+$\n"
            f"Examples: feature/user-auth, perf/db-optimization, ops/deploy-scripts"
        )

    return True, ""


def main() -> int:
    """Main entry point for the branch guard script."""
    current_branch = get_current_branch()

    if current_branch is None:
        return 1

    # Special case: allow main branch for sync operations
    if current_branch == 'main':
        print("Warning: Working on 'main' branch. Consider using a feature branch.")
        return 0

    is_valid, error_message = validate_branch_name(current_branch)

    if not is_valid:
        print(f"Branch naming validation failed:\n{error_message}", file=sys.stderr)
        return 1

    print(f"Branch '{current_branch}' follows naming convention.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
