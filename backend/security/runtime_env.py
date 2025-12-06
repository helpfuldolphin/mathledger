"""
Runtime environment validation helpers.

All security–critical components should go through this module when
loading connection strings or shared secrets so we can enforce
deterministic behaviour and fail fast when configuration is missing.

Strict Mode
-----------
Set ``FIRST_ORGANISM_STRICT=1`` to disable any fallback behavior across
the codebase. When strict mode is enabled, any component that would
otherwise fall back to insecure defaults will instead raise an error.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Optional

__all__ = [
    "MissingEnvironmentVariable",
    "WeakCredentialError",
    "is_strict_mode",
    "get_required_env",
    "get_database_url",
    "get_redis_url",
    "get_allowed_origins",
    "validate_credential_strength",
]

# Blocklist of weak/default passwords that must never be used
_WEAK_PASSWORDS = frozenset({
    "mlpass",
    "postgres",
    "password",
    "test",
    "admin",
    "root",
    "secret",
    "changeme",
    "12345",
    "123456",
    "qwerty",
    "devkey",
    "default",
})


class MissingEnvironmentVariable(RuntimeError):
    """Raised when a required environment variable has not been provided."""


class WeakCredentialError(RuntimeError):
    """Raised when a credential fails strength validation."""


def is_strict_mode() -> bool:
    """
    Return True if FIRST_ORGANISM_STRICT=1 is set.

    When strict mode is enabled, all fallback behavior is disabled and
    any security violation raises immediately.
    """
    return os.getenv("FIRST_ORGANISM_STRICT", "").strip() == "1"


def validate_credential_strength(
    value: str,
    *,
    min_length: int = 12,
    context: str = "credential",
) -> None:
    """
    Validate that a credential meets minimum strength requirements.

    Args:
        value: The credential to validate
        min_length: Minimum required length (default 12)
        context: Description for error messages

    Raises:
        WeakCredentialError: If the credential is too weak
    """
    if not value:
        raise WeakCredentialError(f"{context} cannot be empty")

    if len(value) < min_length:
        raise WeakCredentialError(
            f"{context} must be at least {min_length} characters (got {len(value)})"
        )

    # Check against blocklist
    lowered = value.lower()
    for weak in _WEAK_PASSWORDS:
        if weak in lowered:
            raise WeakCredentialError(
                f"{context} contains weak/default password pattern: {weak!r}"
            )


def get_required_env(name: str, *, validate_strength: bool = False) -> str:
    """
    Return the value for ``name`` or raise :class:`MissingEnvironmentVariable`.

    We deliberately do not allow implicit defaults for security‑sensitive
    configuration values – callers must set them explicitly.

    Args:
        name: Environment variable name
        validate_strength: If True, validate credential strength

    Raises:
        MissingEnvironmentVariable: If variable is not set
        WeakCredentialError: If validate_strength=True and credential is weak
    """
    value = os.getenv(name)
    if value is None or value.strip() == "":
        raise MissingEnvironmentVariable(
            f"Environment variable {name!r} must be set and non-empty."
        )

    if validate_strength:
        validate_credential_strength(value, context=name)

    return value


def _extract_password_from_url(url: str) -> Optional[str]:
    """Extract password from a database or Redis URL."""
    # Pattern: protocol://[user[:password]@]host...
    match = re.search(r"://[^:]+:([^@]+)@", url)
    if match:
        return match.group(1)
    return None


@lru_cache()
def get_database_url() -> str:
    """
    Retrieve the database connection URL.

    The connection string must include credentials and preferred
    connection parameters (for example ``sslmode=require``).

    In strict mode, the password is validated for strength.
    """
    url = get_required_env("DATABASE_URL")

    if is_strict_mode():
        password = _extract_password_from_url(url)
        if password:
            validate_credential_strength(password, context="DATABASE_URL password")

    return url


@lru_cache()
def get_redis_url() -> str:
    """
    Retrieve the Redis connection URL.

    If authentication is required it must be encoded in the URL, e.g.
    ``rediss://:<password>@host:6379/0``.

    In strict mode, the password is validated for strength.
    """
    url = get_required_env("REDIS_URL")

    if is_strict_mode():
        password = _extract_password_from_url(url)
        if password:
            validate_credential_strength(password, context="REDIS_URL password")

    return url


@lru_cache()
def get_allowed_origins() -> list[str]:
    """
    Parse the comma-separated ``CORS_ALLOWED_ORIGINS`` value.

    Returns the origin list to be used with FastAPI's CORSMiddleware.

    In strict mode, wildcards (*) are rejected.
    """
    origins = get_required_env("CORS_ALLOWED_ORIGINS")
    origin_list = [origin.strip() for origin in origins.split(",") if origin.strip()]

    if is_strict_mode():
        if "*" in origin_list:
            raise WeakCredentialError(
                "CORS_ALLOWED_ORIGINS cannot contain wildcard (*) in strict mode"
            )

    return origin_list


