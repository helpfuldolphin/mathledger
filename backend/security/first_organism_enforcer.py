"""
First Organism Environment Security Enforcer.

Ensures that First Organism integration tests run ONLY under explicitly
configured, non-default credentials. This prevents accidental test runs
against production or unsecured development databases.

Usage:
    from backend.security.first_organism_enforcer import enforce_first_organism_env

    # Call at the start of any First Organism test
    enforce_first_organism_env()  # raises InsecureCredentialsError if checks fail
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Optional
from urllib.parse import urlparse, parse_qs

__all__ = [
    "InsecureCredentialsError",
    "FirstOrganismEnvConfig",
    "enforce_first_organism_env",
    "validate_database_url",
    "validate_redis_url",
    "validate_api_key",
    "validate_cors_origins",
]


class InsecureCredentialsError(RuntimeError):
    """Raised when First Organism tests are attempted with insecure credentials."""

    def __init__(self, violations: List[str]):
        self.violations = violations
        msg = (
            "First Organism security check FAILED. Cannot run integration tests "
            "with insecure credentials.\n\nViolations:\n"
            + "\n".join(f"  - {v}" for v in violations)
            + "\n\nFix: Set secure credentials in .env.first_organism and load them "
            "before running tests."
        )
        super().__init__(msg)


# Known insecure/default credentials that MUST be rejected
BANNED_POSTGRES_PASSWORDS = frozenset({
    "postgres",
    "password",
    "mlpass",
    "ml",
    "mathledger",
    "admin",
    "root",
    "test",
    "secret",
    "changeme",
    "",
})

BANNED_REDIS_PASSWORDS = frozenset({
    "",
    "redis",
    "password",
    "secret",
    "changeme",
    "test",
})

BANNED_API_KEYS = frozenset({
    "devkey",
    "dev",
    "test",
    "secret",
    "changeme",
    "api-key",
    "apikey",
    "",
})

# Minimum password length for security
MIN_PASSWORD_LENGTH = 12
MIN_API_KEY_LENGTH = 16


@dataclass
class FirstOrganismEnvConfig:
    """Validated environment configuration for First Organism tests."""
    database_url: str
    redis_url: str
    api_key: str
    cors_origins: List[str]
    postgres_user: str
    postgres_password: str
    redis_password: Optional[str]


def _extract_postgres_credentials(url: str) -> tuple[str, str]:
    """Extract username and password from PostgreSQL URL."""
    parsed = urlparse(url)
    user = parsed.username or ""
    password = parsed.password or ""
    return user, password


def _extract_redis_password(url: str) -> Optional[str]:
    """Extract password from Redis URL."""
    parsed = urlparse(url)
    return parsed.password


def validate_database_url(url: str) -> List[str]:
    """
    Validate DATABASE_URL for First Organism security requirements.

    Returns list of violation messages (empty if valid).
    """
    violations = []

    if not url:
        violations.append("DATABASE_URL is not set")
        return violations

    user, password = _extract_postgres_credentials(url)

    # Check for banned passwords
    if password.lower() in BANNED_POSTGRES_PASSWORDS:
        violations.append(
            f"DATABASE_URL uses banned password '{password}'. "
            "Use a strong, unique password."
        )

    # Check password length
    if len(password) < MIN_PASSWORD_LENGTH:
        violations.append(
            f"DATABASE_URL password is too short ({len(password)} chars). "
            f"Minimum {MIN_PASSWORD_LENGTH} characters required."
        )

    # Check for default usernames with weak passwords
    if user.lower() in {"postgres", "ml", "mathledger"} and len(password) < 16:
        violations.append(
            f"DATABASE_URL uses common username '{user}' with weak password. "
            "Use a strong password (16+ chars) or a custom username."
        )

    # Warn if no SSL mode specified (for non-localhost)
    parsed = urlparse(url)
    if parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        query = parse_qs(parsed.query)
        if "sslmode" not in query or query.get("sslmode", [""])[0] == "disable":
            violations.append(
                "DATABASE_URL connects to remote host without sslmode=require. "
                "Enable SSL for non-local connections."
            )

    return violations


def validate_redis_url(url: str) -> List[str]:
    """
    Validate REDIS_URL for First Organism security requirements.

    Returns list of violation messages (empty if valid).
    """
    violations = []

    if not url:
        violations.append("REDIS_URL is not set")
        return violations

    password = _extract_redis_password(url)

    # Check for missing password
    if password is None or password == "":
        violations.append(
            "REDIS_URL has no password. "
            "Configure Redis with requirepass and set password in URL."
        )
        return violations

    # Check for banned passwords
    if password.lower() in BANNED_REDIS_PASSWORDS:
        violations.append(
            f"REDIS_URL uses banned password '{password}'. "
            "Use a strong, unique password."
        )

    # Check password length
    if len(password) < MIN_PASSWORD_LENGTH:
        violations.append(
            f"REDIS_URL password is too short ({len(password)} chars). "
            f"Minimum {MIN_PASSWORD_LENGTH} characters required."
        )

    return violations


def validate_api_key(key: str) -> List[str]:
    """
    Validate LEDGER_API_KEY for First Organism security requirements.

    Returns list of violation messages (empty if valid).
    """
    violations = []

    if not key:
        violations.append("LEDGER_API_KEY is not set")
        return violations

    # Check for banned keys
    if key.lower() in BANNED_API_KEYS:
        violations.append(
            f"LEDGER_API_KEY uses banned value '{key}'. "
            "Use a strong, unique API key."
        )

    # Check key length
    if len(key) < MIN_API_KEY_LENGTH:
        violations.append(
            f"LEDGER_API_KEY is too short ({len(key)} chars). "
            f"Minimum {MIN_API_KEY_LENGTH} characters required."
        )

    # Check for high entropy (simple check: not all same char, has mixed case or numbers)
    if len(set(key)) < 6:
        violations.append(
            "LEDGER_API_KEY has low entropy. "
            "Use a randomly generated key with mixed characters."
        )

    return violations


def validate_cors_origins(origins_str: str) -> List[str]:
    """
    Validate CORS_ALLOWED_ORIGINS for First Organism security requirements.

    Returns list of violation messages (empty if valid).
    """
    violations = []

    if not origins_str:
        violations.append("CORS_ALLOWED_ORIGINS is not set")
        return violations

    # Check for wildcard (allows any origin)
    if origins_str.strip() == "*":
        violations.append(
            "CORS_ALLOWED_ORIGINS is set to '*' (wildcard). "
            "Specify explicit allowed origins."
        )

    origins = [o.strip() for o in origins_str.split(",") if o.strip()]

    if not origins:
        violations.append("CORS_ALLOWED_ORIGINS is empty after parsing")

    return violations


def enforce_first_organism_env() -> FirstOrganismEnvConfig:
    """
    Enforce security requirements for First Organism integration tests.

    Validates all required environment variables and raises InsecureCredentialsError
    if any security checks fail.

    Returns:
        FirstOrganismEnvConfig with validated configuration values.

    Raises:
        InsecureCredentialsError: If any security checks fail.
    """
    all_violations: List[str] = []

    # Gather all environment values
    database_url = os.getenv("DATABASE_URL", "")
    redis_url = os.getenv("REDIS_URL", "")
    api_key = os.getenv("LEDGER_API_KEY", "")
    cors_origins = os.getenv("CORS_ALLOWED_ORIGINS", "")

    # Validate each
    all_violations.extend(validate_database_url(database_url))
    all_violations.extend(validate_redis_url(redis_url))
    all_violations.extend(validate_api_key(api_key))
    all_violations.extend(validate_cors_origins(cors_origins))

    # Check for explicit First Organism mode marker
    # RUNTIME_ENV=test_hardened is required for proper security posture
    runtime_env = os.getenv("RUNTIME_ENV", "").strip()
    if not runtime_env:
        all_violations.append(
            "RUNTIME_ENV is not set. Set RUNTIME_ENV=test_hardened in .env.first_organism "
            "to ensure proper security posture for First Organism tests."
        )
    elif runtime_env == "production":
        all_violations.append(
            "RUNTIME_ENV is 'production' - First Organism tests must not run in production. "
            "Set RUNTIME_ENV=test_hardened in .env.first_organism."
        )
    elif runtime_env != "test_hardened":
        # Allow backward compatibility but warn
        all_violations.append(
            f"RUNTIME_ENV is '{runtime_env}', expected 'test_hardened' for First Organism tests. "
            "Set RUNTIME_ENV=test_hardened in .env.first_organism. "
            "(Legacy values 'first_organism' and 'integration' are deprecated.)"
        )

    if all_violations:
        raise InsecureCredentialsError(all_violations)

    # Extract credentials for return
    postgres_user, postgres_password = _extract_postgres_credentials(database_url)
    redis_password = _extract_redis_password(redis_url)
    cors_list = [o.strip() for o in cors_origins.split(",") if o.strip()]

    return FirstOrganismEnvConfig(
        database_url=database_url,
        redis_url=redis_url,
        api_key=api_key,
        cors_origins=cors_list,
        postgres_user=postgres_user,
        postgres_password=postgres_password,
        redis_password=redis_password,
    )


def require_first_organism_env():
    """
    Pytest fixture-compatible function that enforces First Organism environment.

    Use as a pytest fixture or call directly at test setup.
    """
    return enforce_first_organism_env()
