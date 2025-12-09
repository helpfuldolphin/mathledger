
"""
Root conftest.py for the mathledger project.

This file is automatically discovered by pytest and is used for project-wide
test configuration.
"""

import os
import sys
from pathlib import Path

# Load environment variables from .env file BEFORE any other imports
# This must happen at module level to ensure variables are set before
# any modules that require them at import time are loaded
_project_root = Path(__file__).resolve().parent
_env_file = _project_root / ".env"

if _env_file.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env_file)
    except ImportError:
        # Fallback: manually parse .env if python-dotenv not installed
        with open(_env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip().strip("'\"")
                    if key and key not in os.environ:
                        os.environ[key] = value

# Set default test environment variables if not already set
if "DATABASE_URL" not in os.environ:
    os.environ["DATABASE_URL"] = "postgresql://ml:mlpass@localhost:5432/mathledger"
if "REDIS_URL" not in os.environ:
    os.environ["REDIS_URL"] = "redis://localhost:6379/0"


def pytest_configure(config):
    """
    Hook that runs at the start of the pytest session, before any tests
    are collected. This is the canonical place to modify sys.path.
    """
    # Determine the project root (the parent of this conftest.py file)
    project_root = Path(__file__).resolve().parent

    # Add the project root to the Python path
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

