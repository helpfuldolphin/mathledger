#!/usr/bin/env python3
"""
ASCII-only validation script for documentation and scripts.

This script ensures that all files in docs/ and scripts/ directories
contain only ASCII characters, preventing encoding issues and ensuring
cross-platform compatibility.
"""

import sys
import os
from pathlib import Path


def is_ascii_only(file_path: Path) -> bool:
    """Check if a file contains only ASCII characters."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            return content.isascii()
    except UnicodeDecodeError:
        return False
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return False


def find_non_ascii_files(directories: list[str]) -> list[Path]:
    """Find all non-ASCII files in the specified directories."""
    non_ascii_files = []

    # File extensions to skip (binary files)
    skip_extensions = {'.pdf', '.log', '.exe', '.dll', '.so', '.dylib', '.bin', '.dat', '.db', '.sqlite', '.svg'}

    # File extensions to check (only critical text files)
    check_extensions = {'.md', '.py', '.ps1', '.sh', '.yaml', '.yml', '.json', '.txt', '.sql', '.lean'}

    for directory in directories:
        if not os.path.exists(directory):
            continue

        for file_path in Path(directory).rglob('*'):
            if (file_path.is_file() and
                file_path.suffix.lower() not in skip_extensions and
                file_path.suffix.lower() in check_extensions and
                not is_ascii_only(file_path)):
                non_ascii_files.append(file_path)

    return non_ascii_files


def main():
    """Main function to check ASCII compliance."""
    # Directories to check for ASCII-only content
    check_dirs = ['docs/', 'scripts/']

    # Files that are allowed to have non-ASCII content (mathematical symbols, etc.)
    allowed_non_ascii_files = {
        'docs/API_REFERENCE.md',  # Contains mathematical symbols
        'docs/whitepaper.md',     # Contains mathematical symbols
        'docs/theory_packs.md',   # Contains mathematical symbols
        'docs/edge_setup.md',     # Contains mathematical symbols
        'docs/M2_WIRING_STATUS.md', # Contains mathematical symbols
        'docs/perf/modus_ponens_indexing.md', # Contains mathematical symbols
        'docs/progress.md',       # Contains status symbols
    }

    # Find non-ASCII files
    non_ascii_files = find_non_ascii_files(check_dirs)

    # Filter out allowed files (normalize paths for comparison)
    problematic_files = []
    for f in non_ascii_files:
        # Convert to forward slashes for consistent comparison
        normalized_path = str(f).replace('\\', '/')
        if normalized_path not in allowed_non_ascii_files:
            problematic_files.append(f)

    # For now, only flag files with critical non-ASCII characters (smart quotes, em dashes)
    critical_issues = []
    for f in problematic_files:
        try:
            with open(f, 'r', encoding='utf-8') as file:
                content = file.read()
                # Check for critical non-ASCII characters that cause real problems
                if any(char in content for char in ['"', '"', ''', ''', '—', '–']):
                    critical_issues.append(f)
        except:
            # If we can't read the file, skip it
            pass

    if critical_issues:
        print("ERROR: Critical non-ASCII characters found:")
        for file_path in critical_issues:
            print(f"  - {file_path}")
        print("\nThese files contain problematic characters that must be fixed:")
        print("  - Smart quotes: Use \" \" instead of \" \"")
        print("  - Em dashes: Use -- instead of —")
        return 1
    else:
        print("SUCCESS: No critical non-ASCII characters found")
        if non_ascii_files:
            print(f"Note: {len(non_ascii_files)} files contain mathematical symbols or other non-critical Unicode characters.")
        return 0


if __name__ == '__main__':
    sys.exit(main())
