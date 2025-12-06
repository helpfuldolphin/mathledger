"""
First Organism Integration Test Configuration.

This module provides pytest fixtures that enforce secure environment
configuration for First Organism integration tests.

The enforcer will FAIL tests if:
- Default/weak credentials are detected (postgres, mlpass, devkey, etc.)
- Passwords are too short (< 12 chars for DB/Redis, < 16 for API key)
- CORS is set to wildcard (*)
- Required environment variables are missing
- RUNTIME_ENV is not set to 'test_hardened' (logs warning if not)
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Generator

import pytest

from backend.security.first_organism_enforcer import (
    FirstOrganismEnvConfig,
    InsecureCredentialsError,
    enforce_first_organism_env,
)


def pytest_configure(config):
    """Register the first_organism marker."""
    config.addinivalue_line(
        "markers",
        "first_organism: mark test as requiring First Organism secure environment",
    )


@pytest.fixture(scope="session")
def first_organism_secure_env() -> Generator[FirstOrganismEnvConfig, None, None]:
    """
    Session-scoped fixture that enforces First Organism security requirements.

    This fixture will FAIL the entire test session if insecure credentials
    are detected. It runs once per session and caches the validated config.
    
    Also enforces RUNTIME_ENV=test_hardened via assert_runtime_env_hardened().

    Usage:
        @pytest.mark.first_organism
        def test_something(first_organism_secure_env):
            config = first_organism_secure_env
            # config.database_url, config.redis_url, etc. are validated

    Raises:
        pytest.fail: If security checks fail
        pytest.skip: If RUNTIME_ENV=production (safety skip)
    """
    try:
        # Enforce RUNTIME_ENV=test_hardened (minimal integration point)
        assert_runtime_env_hardened()
        # Enforce all other security requirements
        config = enforce_first_organism_env()
        yield config
    except InsecureCredentialsError as e:
        pytest.fail(str(e))


@pytest.fixture(scope="function")
def first_organism_env_checked(first_organism_secure_env) -> FirstOrganismEnvConfig:
    """
    Function-scoped fixture that ensures env is checked before each test.

    Depends on the session-scoped secure env fixture.
    """
    return first_organism_secure_env


@pytest.fixture(scope="session")
def first_organism_skip_if_no_env() -> Generator[FirstOrganismEnvConfig, None, None]:
    """
    Session-scoped fixture that SKIPS (not fails) if env is not configured.

    Use this for tests that should run only when First Organism env is set up,
    but shouldn't fail CI if the env isn't available.

    Usage:
        @pytest.mark.first_organism
        def test_optional_integration(first_organism_skip_if_no_env):
            # Skipped if env not configured, runs if it is
            pass
    """
    try:
        config = enforce_first_organism_env()
        yield config
    except InsecureCredentialsError:
        pytest.skip(
            "First Organism secure environment not configured. "
            "Set up .env.first_organism with secure credentials to run this test."
        )


def assert_runtime_env_hardened() -> None:
    """
    Assert that RUNTIME_ENV is set to 'test_hardened' for First Organism tests.
    
    This ensures FO tests don't accidentally run under 'production' or unknown env.
    If RUNTIME_ENV is not 'test_hardened', logs a warning (does not fail).
    
    Usage:
        def test_something():
            assert_runtime_env_hardened()  # Check at start of test
            # ... rest of test ...
    
    Raises:
        pytest.skip: If RUNTIME_ENV is 'production' (safety skip)
    """
    runtime_env = os.getenv("RUNTIME_ENV", "").strip()
    
    if runtime_env == "production":
        pytest.skip(
            f"RUNTIME_ENV is 'production' - skipping First Organism test for safety. "
            f"Set RUNTIME_ENV=test_hardened to run FO tests."
        )
    
    if runtime_env != "test_hardened":
        # Log warning but don't fail (for backward compatibility)
        warnings.warn(
            f"RUNTIME_ENV is '{runtime_env}', expected 'test_hardened' for First Organism tests. "
            f"Set RUNTIME_ENV=test_hardened in .env.first_organism to ensure proper security posture.",
            UserWarning,
            stacklevel=2
        )


def pytest_collection_modifyitems(config, items):
    """
    Automatically apply first_organism marker handling.

    Tests marked with @pytest.mark.first_organism will be:
    - Skipped if FIRST_ORGANISM_TESTS is not set to 'true'
    - Failed if credentials are insecure when running
    """
    skip_marker = pytest.mark.skip(
        reason="First Organism tests disabled. Set FIRST_ORGANISM_TESTS=true to enable."
    )

    first_organism_env = os.getenv("FIRST_ORGANISM_TESTS", "").lower()
    spark_file_trigger = Path(".spark_run_enable").is_file()
    first_organism_enabled = (
        first_organism_env == "true"
        or os.getenv("SPARK_RUN", "") == "1"
        or spark_file_trigger
    )

    for item in items:
        if "first_organism" in item.keywords:
            if not first_organism_enabled:
                item.add_marker(skip_marker)
