
"""
Path Bootstrap Utility for Pytest

This module is imported for its side effect: it finds the project root
and adds it to the system path. This ensures that pytest, regardless of
how it's invoked, can find and import the project's packages (like 'analysis'
and 'backend').
"""
import sys
from pathlib import Path

# The project root is defined as the parent directory of the 'tests' directory.
# This makes the logic portable and independent of the current working directory.
project_root = Path(__file__).resolve().parents[1]

if str(project_root) not in sys.path:
    # Use insert(0) to ensure the project root takes precedence over any
    # installed packages of the same name.
    sys.path.insert(0, str(project_root))
