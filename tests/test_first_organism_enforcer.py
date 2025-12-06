"""
Unit tests for the First Organism environment security enforcer.

These tests verify that the enforcer correctly rejects insecure credentials
and accepts properly configured secure environments.
"""

import os
import pytest
from unittest.mock import patch

from backend.security.first_organism_enforcer import (
    InsecureCredentialsError,
    FirstOrganismEnvConfig,
    enforce_first_organism_env,
    validate_database_url,
    validate_redis_url,
    validate_api_key,
    validate_cors_origins,
    BANNED_POSTGRES_PASSWORDS,
    BANNED_REDIS_PASSWORDS,
    BANNED_API_KEYS,
    MIN_PASSWORD_LENGTH,
    MIN_API_KEY_LENGTH,
)


class TestValidateDatabaseUrl:
    """Tests for DATABASE_URL validation."""

    def test_missing_url_returns_violation(self):
        violations = validate_database_url("")
        assert len(violations) == 1
        assert "not set" in violations[0]

    def test_banned_password_rejected(self):
        for banned in ["mlpass", "postgres", "password", "test"]:
            url = f"postgresql://user:{banned}@localhost:5432/db"
            violations = validate_database_url(url)
            assert any("banned password" in v.lower() for v in violations), f"Should reject '{banned}'"

    def test_short_password_rejected(self):
        url = "postgresql://user:short@localhost:5432/db"
        violations = validate_database_url(url)
        assert any("too short" in v.lower() for v in violations)

    def test_weak_password_with_common_user_rejected(self):
        url = "postgresql://postgres:somepassword@localhost:5432/db"
        violations = validate_database_url(url)
        # Common username + weak password triggers warning
        assert any("common username" in v.lower() for v in violations)

    def test_remote_without_ssl_rejected(self):
        url = "postgresql://user:strongpassword123456@remote.example.com:5432/db"
        violations = validate_database_url(url)
        assert any("sslmode" in v.lower() for v in violations)

    def test_remote_with_ssl_accepted(self):
        url = "postgresql://user:strongpassword123456@remote.example.com:5432/db?sslmode=require"
        violations = validate_database_url(url)
        # Should not have SSL warning
        assert not any("sslmode" in v.lower() for v in violations)

    def test_localhost_without_ssl_accepted(self):
        url = "postgresql://user:strongpassword123456@localhost:5432/db"
        violations = validate_database_url(url)
        # Localhost doesn't require SSL
        assert not any("sslmode" in v.lower() for v in violations)

    def test_secure_url_passes(self):
        url = "postgresql://admin:v3ry_s3cur3_p4ssw0rd_123!@localhost:5432/testdb"
        violations = validate_database_url(url)
        assert len(violations) == 0


class TestValidateRedisUrl:
    """Tests for REDIS_URL validation."""

    def test_missing_url_returns_violation(self):
        violations = validate_redis_url("")
        assert len(violations) == 1
        assert "not set" in violations[0]

    def test_no_password_rejected(self):
        url = "redis://localhost:6379/0"
        violations = validate_redis_url(url)
        assert any("no password" in v.lower() for v in violations)

    def test_banned_password_rejected(self):
        for banned in ["redis", "password", "test", "secret"]:
            url = f"redis://:{banned}@localhost:6379/0"
            violations = validate_redis_url(url)
            assert any("banned password" in v.lower() for v in violations), f"Should reject '{banned}'"

    def test_short_password_rejected(self):
        url = "redis://:shortpwd@localhost:6379/0"
        violations = validate_redis_url(url)
        assert any("too short" in v.lower() for v in violations)

    def test_secure_url_passes(self):
        url = "redis://:s3cur3_r3d1s_p4ssw0rd!@localhost:6379/0"
        violations = validate_redis_url(url)
        assert len(violations) == 0


class TestValidateApiKey:
    """Tests for LEDGER_API_KEY validation."""

    def test_missing_key_returns_violation(self):
        violations = validate_api_key("")
        assert len(violations) == 1
        assert "not set" in violations[0]

    def test_banned_key_rejected(self):
        for banned in ["devkey", "test", "secret", "apikey"]:
            violations = validate_api_key(banned)
            assert any("banned value" in v.lower() for v in violations), f"Should reject '{banned}'"

    def test_short_key_rejected(self):
        violations = validate_api_key("shortkey")
        assert any("too short" in v.lower() for v in violations)

    def test_low_entropy_key_rejected(self):
        violations = validate_api_key("aaaaaaaaaaaaaaaa")  # 16 chars but all same
        assert any("low entropy" in v.lower() for v in violations)

    def test_secure_key_passes(self):
        violations = validate_api_key("sk_test_4c7b8d9e0f1a2b3c4d5e6f7g")
        assert len(violations) == 0


class TestValidateCorsOrigins:
    """Tests for CORS_ALLOWED_ORIGINS validation."""

    def test_missing_origins_returns_violation(self):
        violations = validate_cors_origins("")
        assert len(violations) == 1
        assert "not set" in violations[0]

    def test_wildcard_rejected(self):
        violations = validate_cors_origins("*")
        assert any("wildcard" in v.lower() for v in violations)

    def test_specific_origins_accepted(self):
        violations = validate_cors_origins("http://localhost:3000,http://localhost:8000")
        assert len(violations) == 0


class TestEnforceFirstOrganismEnv:
    """Integration tests for the full enforce function."""

    def test_missing_all_env_raises_error(self):
        """Test that completely missing env raises InsecureCredentialsError."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(InsecureCredentialsError) as exc_info:
                enforce_first_organism_env()
            # Should have violations for all required vars
            assert len(exc_info.value.violations) >= 4

    def test_default_credentials_raise_error(self):
        """Test that default/weak credentials are rejected."""
        env = {
            "DATABASE_URL": "postgresql://ml:mlpass@localhost:5432/mathledger",
            "REDIS_URL": "redis://localhost:6379/0",
            "LEDGER_API_KEY": "devkey",
            "CORS_ALLOWED_ORIGINS": "*",
        }
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(InsecureCredentialsError) as exc_info:
                enforce_first_organism_env()
            violations = exc_info.value.violations
            # Should catch multiple issues
            assert len(violations) >= 3

    def test_secure_env_returns_config(self):
        """Test that properly secured env passes and returns config."""
        env = {
            "DATABASE_URL": "postgresql://admin:v3ry_s3cur3_p4ssw0rd!@localhost:5432/testdb",
            "REDIS_URL": "redis://:r3d1s_s3cur3_p4ssw0rd!@localhost:6379/0",
            "LEDGER_API_KEY": "sk_production_key_abc123xyz",
            "CORS_ALLOWED_ORIGINS": "http://localhost:8000",
            "RUNTIME_ENV": "test_hardened",
        }
        with patch.dict(os.environ, env, clear=True):
            config = enforce_first_organism_env()
            assert isinstance(config, FirstOrganismEnvConfig)
            assert config.database_url == env["DATABASE_URL"]
            assert config.redis_url == env["REDIS_URL"]
            assert config.api_key == env["LEDGER_API_KEY"]
            assert config.cors_origins == ["http://localhost:8000"]

    def test_wrong_runtime_env_adds_violation(self):
        """Test that wrong RUNTIME_ENV adds a violation."""
        env = {
            "DATABASE_URL": "postgresql://admin:v3ry_s3cur3_p4ssw0rd!@localhost:5432/testdb",
            "REDIS_URL": "redis://:r3d1s_s3cur3_p4ssw0rd!@localhost:6379/0",
            "LEDGER_API_KEY": "sk_production_key_abc123xyz",
            "CORS_ALLOWED_ORIGINS": "http://localhost:8000",
            "RUNTIME_ENV": "production",  # Wrong env
        }
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(InsecureCredentialsError) as exc_info:
                enforce_first_organism_env()
            assert any("RUNTIME_ENV" in v for v in exc_info.value.violations)


class TestBannedCredentialLists:
    """Tests to verify banned credential lists are comprehensive."""

    def test_postgres_banned_list_coverage(self):
        """Verify common weak passwords are in banned list."""
        common_weak = {"postgres", "password", "mlpass", "admin", "root"}
        assert common_weak.issubset(BANNED_POSTGRES_PASSWORDS)

    def test_redis_banned_list_coverage(self):
        """Verify common weak passwords are in banned list."""
        common_weak = {"redis", "password", "secret"}
        assert common_weak.issubset(BANNED_REDIS_PASSWORDS)

    def test_api_key_banned_list_coverage(self):
        """Verify common weak API keys are in banned list."""
        common_weak = {"devkey", "test", "secret", "apikey"}
        assert common_weak.issubset(BANNED_API_KEYS)

    def test_minimum_lengths(self):
        """Verify minimum lengths are reasonable."""
        assert MIN_PASSWORD_LENGTH >= 12
        assert MIN_API_KEY_LENGTH >= 16
