#!/usr/bin/env python3
"""
ASCII-only check for documentation and script files.
"""
import sys
import os


def check_ascii_only(filepath):
    """Check if file contains only ASCII characters."""
    try:
        with open(filepath, 'rb') as f:
            content = f.read()
            # Check for non-ASCII bytes
            non_ascii = [i for i, byte in enumerate(content) if byte > 127]
            if non_ascii:
                print(f'Non-ASCII characters found in {filepath} at positions: {non_ascii[:10]}')
                return False
            return True
    except Exception as e:
        print(f'Error reading {filepath}: {e}')
        return False


if __name__ == '__main__':
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        if not check_ascii_only(filepath):
            sys.exit(1)
    else:
        sys.exit(0)
