"""
Shared fixtures for First Organism integration tests.

Provides:
- Database connection fixtures with skip-on-unavailable behavior
- MDAP-compliant deterministic helpers
- First Organism test context builders
- Canonical attestation artifact generation
- Complete [PASS] FIRST ORGANISM ALIVE log pipeline

Mode Detection:
- OFFLINE/MOCK: When DATABASE_URL is unavailable or ML_USE_LOCAL_DB=1 with mock://
  → Tests skip with explicit [SKIP] reasons
- ABSTAIN: When infrastructure is partially available but chain cannot complete
  → Tests emit [ABSTAIN] with reason and synthetic attestation
- REAL: When Postgres/Redis are reachable
  → Full chain execution with deterministic seeding

MDAP Traceability:
- All fixtures emit structured metadata with deterministic IDs and timestamps
- Attestation artifacts include full provenance chain
- Every skip/abstain/pass is logged with MDAP-traceable context
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import psycopg
import pytest

# Canonical imports aligned to the new module structure
from derivation.bounds import SliceBounds
from derivation.pipeline import DerivationPipeline
from derivation.verification import StatementVerifier
from curriculum.gates import (
    AbstentionGateSpec,
    CapsGateSpec,
    CoverageGateSpec,
    CurriculumSlice,
    GateEvaluator,
    NormalizedMetrics,
    SliceGates,
    VelocityGateSpec,
)

# Import from canonical ledger modules
from ledger.ingest import LedgerIngestor
from ledger.ui_events import capture_ui_event, ui_event_store

# Import from canonical attestation module
from attestation.dual_root import compute_composite_root

from substrate.repro.determinism import (
    deterministic_hash,
    deterministic_run_id,
    deterministic_seed_from_content,
    deterministic_timestamp,
    deterministic_unix_timestamp,
    deterministic_isoformat,
    DETERMINISTIC_EPOCH,
)

# Import app + dependency symbol for test client override
from interface.api.app import app, get_db_connection

from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# MDAP Deterministic Epoch & Namespace
# ---------------------------------------------------------------------------
MDAP_EPOCH_SEED = 0x4D444150  # "MDAP" as hex seed
FIRST_ORGANISM_NAMESPACE = "first-organism-integration"
FIRST_ORGANISM_VERSION = "1.0.0"

# Probe timeouts
DB_PROBE_TIMEOUT_SECONDS = 3
REDIS_PROBE_TIMEOUT_SECONDS = 2
DB_PROBE_RETRIES = 2
REDIS_PROBE_RETRIES = 1


# ---------------------------------------------------------------------------
# Environment Status Enum
# ---------------------------------------------------------------------------
class InfraStatus(str, Enum):
    """Infrastructure component status."""
    ONLINE = "online"
    OFFLINE = "offline"
    TIMEOUT = "timeout"
    AUTH_FAILED = "auth_failed"
    UNKNOWN = "unknown"


class ChainStatus(str, Enum):
    """Overall chain execution status."""
    READY = "ready"          # Full chain can execute
    ABSTAIN = "abstain"      # Partial availability, synthetic attestation
    SKIP = "skip"            # Cannot proceed at all
    MOCK = "mock"            # Mock mode for offline testing


# ---------------------------------------------------------------------------
# MDAP Deterministic Helpers
# ---------------------------------------------------------------------------
def mdap_deterministic_id(context: str, *parts: Any) -> str:
    """
    Generate a deterministic ID suitable for MDAP reproducibility.

    Args:
        context: Namespace context (e.g., "ui-event", "derivation")
        *parts: Additional content to hash

    Returns:
        Deterministic identifier string with MDAP namespace
    """
    return deterministic_run_id(context, FIRST_ORGANISM_NAMESPACE, *parts)


def mdap_deterministic_timestamp(content: str) -> int:
    """
    Generate a deterministic Unix timestamp from content.

    Args:
        content: Content to derive timestamp from

    Returns:
        Deterministic Unix timestamp
    """
    seed = deterministic_seed_from_content(content, FIRST_ORGANISM_NAMESPACE)
    return deterministic_unix_timestamp(seed)


def mdap_deterministic_isoformat(content: str) -> str:
    """
    Generate a deterministic ISO-8601 timestamp from content.

    Args:
        content: Content to derive timestamp from

    Returns:
        ISO-8601 formatted timestamp string
    """
    return deterministic_isoformat(content, FIRST_ORGANISM_NAMESPACE)


def mdap_deterministic_seed() -> int:
    """
    Return the canonical MDAP epoch seed.

    Returns:
        Fixed integer seed for reproducibility
    """
    return MDAP_EPOCH_SEED


def mdap_synthetic_roots(statement_hash: str) -> Tuple[str, str, str]:
    """
    Generate synthetic R_t, U_t, H_t for ABSTAIN mode.

    When infrastructure is unavailable, we still need deterministic
    attestation roots for the artifact. These are clearly marked as synthetic.

    Args:
        statement_hash: Hash of the derived statement

    Returns:
        Tuple of (R_t, U_t, H_t) all derived deterministically
    """
    # Synthetic R_t: hash of "SYNTHETIC:R_t:" + statement_hash
    r_t = hashlib.sha256(f"SYNTHETIC:R_t:{statement_hash}".encode()).hexdigest()
    # Synthetic U_t: hash of "SYNTHETIC:U_t:" + statement_hash
    u_t = hashlib.sha256(f"SYNTHETIC:U_t:{statement_hash}".encode()).hexdigest()
    # H_t = SHA256(R_t || U_t) - same formula as real attestation
    h_t = compute_composite_root(r_t, u_t)
    return r_t, u_t, h_t


# ---------------------------------------------------------------------------
# Infrastructure Probing
# ---------------------------------------------------------------------------
def probe_postgres(url: str, timeout: float = DB_PROBE_TIMEOUT_SECONDS, retries: int = DB_PROBE_RETRIES) -> Tuple[InfraStatus, Optional[str]]:
    """
    Probe PostgreSQL connectivity with timeout and retry logic.

    Args:
        url: PostgreSQL connection URL
        timeout: Connection timeout in seconds
        retries: Number of retry attempts

    Returns:
        Tuple of (status, error_message)
    """
    if not url or url.startswith("mock://"):
        return InfraStatus.OFFLINE, "Mock URL or empty"

    last_error = None
    for attempt in range(retries + 1):
        try:
            # Add timeout to URL if not present
            if "connect_timeout" not in url:
                separator = "&" if "?" in url else "?"
                url_with_timeout = f"{url}{separator}connect_timeout={int(timeout)}"
            else:
                url_with_timeout = url

            conn = psycopg.connect(url_with_timeout, connect_timeout=timeout)
            # Quick sanity check
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
            conn.close()
            return InfraStatus.ONLINE, None
        except psycopg.OperationalError as e:
            error_str = str(e).lower()
            # Check for SSL negotiation errors
            if any(term in error_str for term in ["ssl", "ssl negotiation", "could not send", "tls"]):
                last_error = (
                    f"SSL negotiation failed: {e}\n"
                    f"  Hint: Check sslmode in DATABASE_URL (see FIRST_ORGANISM_ENV.md).\n"
                    f"  For local Docker: use ?sslmode=disable\n"
                    f"  For remote DB: use ?sslmode=require"
                )
                return InfraStatus.OFFLINE, last_error
            elif "timeout" in error_str or "timed out" in error_str:
                last_error = f"Connection timeout after {timeout}s"
                return InfraStatus.TIMEOUT, last_error
            elif "password" in error_str or "authentication" in error_str:
                last_error = f"Authentication failed: {e}"
                return InfraStatus.AUTH_FAILED, last_error
            else:
                last_error = str(e)
        except Exception as e:
            error_str = str(e).lower()
            # Check for SSL errors in generic exceptions too
            if any(term in error_str for term in ["ssl", "ssl negotiation", "could not send", "tls"]):
                last_error = (
                    f"SSL negotiation failed: {e}\n"
                    f"  Hint: Check sslmode in DATABASE_URL (see FIRST_ORGANISM_ENV.md).\n"
                    f"  For local Docker: use ?sslmode=disable\n"
                    f"  For remote DB: use ?sslmode=require"
                )
            else:
                last_error = str(e)

        if attempt < retries:
            time.sleep(0.5)  # Brief pause before retry

    return InfraStatus.OFFLINE, last_error


def probe_redis(url: str, timeout: float = REDIS_PROBE_TIMEOUT_SECONDS, retries: int = REDIS_PROBE_RETRIES) -> Tuple[InfraStatus, Optional[str]]:
    """
    Probe Redis connectivity with timeout and retry logic.

    Args:
        url: Redis connection URL
        timeout: Connection timeout in seconds
        retries: Number of retry attempts

    Returns:
        Tuple of (status, error_message)
    """
    if not url:
        return InfraStatus.OFFLINE, "Empty URL"

    try:
        import redis
    except ImportError:
        return InfraStatus.OFFLINE, "redis package not installed"

    last_error = None
    for attempt in range(retries + 1):
        try:
            r = redis.from_url(url, socket_connect_timeout=timeout, socket_timeout=timeout)
            r.ping()
            r.close()
            return InfraStatus.ONLINE, None
        except redis.AuthenticationError as e:
            last_error = f"Authentication failed: {e}"
            return InfraStatus.AUTH_FAILED, last_error
        except redis.TimeoutError:
            last_error = f"Connection timeout after {timeout}s"
            return InfraStatus.TIMEOUT, last_error
        except Exception as e:
            last_error = str(e)

        if attempt < retries:
            time.sleep(0.3)

    return InfraStatus.OFFLINE, last_error


# ---------------------------------------------------------------------------
# Environment Mode Detection
# ---------------------------------------------------------------------------
@dataclass
class EnvironmentMode:
    """
    Describes the current test environment mode with full MDAP traceability.

    This is the single source of truth for infrastructure status and
    chain execution capability.
    """
    # Infrastructure status
    db_status: InfraStatus
    redis_status: InfraStatus
    db_error: Optional[str]
    redis_error: Optional[str]

    # URLs
    db_url: str
    redis_url: str

    # Computed chain status
    chain_status: ChainStatus
    skip_reason: Optional[str]
    abstain_reason: Optional[str]

    # MDAP metadata
    probe_timestamp: str
    probe_run_id: str
    mdap_seed: int = MDAP_EPOCH_SEED

    @property
    def is_offline(self) -> bool:
        return self.chain_status == ChainStatus.SKIP

    @property
    def is_mock(self) -> bool:
        return self.chain_status == ChainStatus.MOCK

    @property
    def is_abstain(self) -> bool:
        return self.chain_status == ChainStatus.ABSTAIN

    @property
    def is_ready(self) -> bool:
        return self.chain_status == ChainStatus.READY

    @property
    def is_real_db(self) -> bool:
        return self.db_status == InfraStatus.ONLINE

    @property
    def is_redis_available(self) -> bool:
        return self.redis_status == InfraStatus.ONLINE

    @property
    def is_fully_online(self) -> bool:
        return self.is_real_db and self.is_redis_available

    def to_dict(self) -> Dict[str, Any]:
        return {
            "db_status": self.db_status.value,
            "redis_status": self.redis_status.value,
            "db_error": self.db_error,
            "redis_error": self.redis_error,
            "db_url_masked": _mask_password(self.db_url),
            "redis_url_masked": _mask_password(self.redis_url),
            "chain_status": self.chain_status.value,
            "skip_reason": self.skip_reason,
            "abstain_reason": self.abstain_reason,
            "probe_timestamp": self.probe_timestamp,
            "probe_run_id": self.probe_run_id,
            "mdap_seed": self.mdap_seed,
        }

    def to_mdap_metadata(self) -> Dict[str, Any]:
        """Return MDAP-traceable metadata for attestation artifacts."""
        return {
            "environment": {
                "chain_status": self.chain_status.value,
                "db_status": self.db_status.value,
                "redis_status": self.redis_status.value,
            },
            "probe": {
                "timestamp": self.probe_timestamp,
                "run_id": self.probe_run_id,
            },
            "mdap_seed": self.mdap_seed,
            "version": FIRST_ORGANISM_VERSION,
        }


def _mask_password(url: str) -> str:
    """Mask password in URL for safe logging."""
    if not url:
        return ""
    import re
    return re.sub(r"://([^:]+):([^@]+)@", r"://\1:****@", url)


def _trim_url_for_display(url: str, max_length: int = 60) -> str:
    """Trim URL for display in skip messages."""
    if not url:
        return ""
    masked = _mask_password(url)
    if len(masked) <= max_length:
        return masked
    return masked[:max_length - 3] + "..."


def assert_first_organism_ready(env_mode: EnvironmentMode, db_url: str) -> None:
    """
    Assert that the First Organism happy path is ready to run.
    
    This is the canonical gating function for all First Organism tests.
    Raises pytest.skip() with a standardized [SKIP][FO] message if not ready.
    
    Args:
        env_mode: The detected environment mode
        db_url: The database URL being used
        
    Raises:
        pytest.skip: If the environment is not ready, with a detailed reason
    """
    # Check if FIRST_ORGANISM_TESTS is enabled
    first_organism_env = os.getenv("FIRST_ORGANISM_TESTS", "").lower()
    spark_file_trigger = Path(".spark_run_enable").is_file()
    first_organism_enabled = (
        first_organism_env == "true"
        or os.getenv("SPARK_RUN", "") == "1"
        or spark_file_trigger
    )
    
    if not first_organism_enabled:
        pytest.skip(
            "[SKIP][FO] FIRST_ORGANISM_TESTS not set to true/SPARK_RUN; refusing to run by default. "
            "(Set FIRST_ORGANISM_TESTS=true, SPARK_RUN=1, or create .spark_run_enable to enable)"
        )
    
    # Check environment mode
    if env_mode.is_mock:
        db_url_trimmed = _trim_url_for_display(db_url)
        pytest.skip(
            f"[SKIP][FO] EnvironmentMode=MOCK (mock:// URL detected; mode=<mock>, db_url=<{db_url_trimmed}>)"
        )
    
    if env_mode.chain_status == ChainStatus.SKIP:
        db_url_trimmed = _trim_url_for_display(db_url)
        error_detail = env_mode.db_error or "Postgres unreachable"
        host_port = "unknown"
        # Try to extract host:port from URL
        import re
        match = re.search(r"@([^:/]+)(?::(\d+))?", db_url)
        if match:
            host = match.group(1)
            port = match.group(2) or "5432"
            host_port = f"{host}:{port}"
        
        pytest.skip(
            f"[SKIP][FO] EnvironmentMode=SKIP (Postgres unreachable at {host_port}; "
            f"error={error_detail}; mode=<skip>, db_url=<{db_url_trimmed}>) "
            f"(see scripts/start_first_organism_infra.ps1 or SPARK_INFRA_CHECKLIST.md)"
        )
    
    # If we get here, environment is ready
    return


def log_resolved_db_url(db_url: str) -> None:
    """
    Log the resolved database URL for debugging.
    
    This function logs the database URL that will be used for tests,
    with password masked for security. Use this at session start to
    help diagnose connection issues.
    
    Args:
        db_url: The resolved database URL (may contain password)
    """
    masked_url = _mask_password(db_url)
    print(f"[FO] Resolved DATABASE_URL={masked_url}")


def detect_environment_mode() -> EnvironmentMode:
    """
    Detect the current environment mode for First Organism tests.

    This function:
    1. Probes PostgreSQL and Redis with timeout/retry logic
    2. Determines chain execution capability
    3. Returns MDAP-traceable environment metadata

    Returns:
        EnvironmentMode with full infrastructure status
    """
    # Generate MDAP-traceable probe metadata
    probe_timestamp = datetime.now(timezone.utc).isoformat()
    probe_run_id = mdap_deterministic_id("probe", probe_timestamp)

    # Resolve URLs
    local_mode = os.environ.get("ML_USE_LOCAL_DB", "") == "1"
    db_url = os.getenv(
        "DATABASE_URL_TEST",
        os.getenv(
            "DATABASE_URL",
            "postgresql://ml:mlpass@127.0.0.1:5432/mathledger?connect_timeout=5",
        ),
    )
    redis_url = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")

    # Check for mock mode
    is_mock_url = db_url.startswith("mock://")
    if local_mode and is_mock_url:
        return EnvironmentMode(
            db_status=InfraStatus.OFFLINE,
            redis_status=InfraStatus.OFFLINE,
            db_error="Mock mode",
            redis_error="Mock mode",
            db_url=db_url,
            redis_url=redis_url,
            chain_status=ChainStatus.MOCK,
            skip_reason=None,
            abstain_reason="[MOCK] Offline/mock mode: ML_USE_LOCAL_DB=1 with mock:// URL",
            probe_timestamp=probe_timestamp,
            probe_run_id=probe_run_id,
        )

    # Probe infrastructure
    db_status, db_error = probe_postgres(db_url)
    redis_status, redis_error = probe_redis(redis_url)

    # Determine chain status
    chain_status: ChainStatus
    skip_reason: Optional[str] = None
    abstain_reason: Optional[str] = None

    if db_status == InfraStatus.ONLINE:
        if redis_status == InfraStatus.ONLINE:
            chain_status = ChainStatus.READY
        else:
            # DB available but Redis not - can still run with ABSTAIN
            chain_status = ChainStatus.ABSTAIN
            abstain_reason = f"[ABSTAIN] Redis unavailable: {redis_error}"
    else:
        # No DB - must skip
        chain_status = ChainStatus.SKIP
        skip_reason = f"[SKIP] Database unavailable: {db_error}"

    return EnvironmentMode(
        db_status=db_status,
        redis_status=redis_status,
        db_error=db_error,
        redis_error=redis_error,
        db_url=db_url,
        redis_url=redis_url,
        chain_status=chain_status,
        skip_reason=skip_reason,
        abstain_reason=abstain_reason,
        probe_timestamp=probe_timestamp,
        probe_run_id=probe_run_id,
    )


# ---------------------------------------------------------------------------
# Database URL Selection
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def test_db_url() -> str:
    """
    Resolve the test database URL.

    Priority:
    1. DATABASE_URL_TEST environment variable
    2. DATABASE_URL environment variable
    3. Default local PostgreSQL
    """
    if "DATABASE_URL_TEST" in os.environ:
        return os.environ["DATABASE_URL_TEST"]
    return os.getenv(
        "DATABASE_URL",
        "postgresql://ml:mlpass@127.0.0.1:5432/mathledger?connect_timeout=5",
    )


@pytest.fixture(scope="session")
def environment_mode() -> EnvironmentMode:
    """
    Fixture exposing the detected environment mode.

    This is the single source of truth for infrastructure status.
    
    Logs the resolved database URL at session start for debugging.
    """
    mode = detect_environment_mode()
    # Log resolved DB URL once at session start for debugging
    log_resolved_db_url(mode.db_url)
    return mode


@pytest.fixture(scope="session")
def test_db_connection(
    test_db_url: str,
    environment_mode: EnvironmentMode,
) -> Generator[psycopg.Connection, None, None]:
    """
    Provide a database connection or skip tests cleanly if DB is unreachable.

    Skips with explicit [SKIP] reason when:
    - Offline/mock mode is detected
    - Database is not reachable
    """
    # Check if this is a SPARK run
    is_spark_run = (
        os.getenv("SPARK_RUN", "") == "1"
        or Path(".spark_run_enable").exists()
        or os.getenv("FIRST_ORGANISM_TESTS", "").lower() == "true"
    )
    spark_indicator = " (SPARK_RUN detected)" if is_spark_run else ""
    
    if environment_mode.is_mock:
        yield _make_mock_connection()
        return

    if environment_mode.chain_status == ChainStatus.SKIP:
        db_url_trimmed = _trim_url_for_display(test_db_url)
        error_detail = environment_mode.db_error or "Postgres unreachable"
        import re
        match = re.search(r"@([^:/]+)(?::(\d+))?", test_db_url)
        host_port = match.group(1) + ":" + (match.group(2) or "5432") if match else "unknown"
        pytest.skip(
            f"[SKIP][FO] EnvironmentMode=SKIP (Postgres unreachable at {host_port}; "
            f"error={error_detail}; mode=<skip>, db_url=<{db_url_trimmed}>) "
            f"(see scripts/start_first_organism_infra.ps1 or SPARK_INFRA_CHECKLIST.md)"
        )
        return

    try:
        conn = psycopg.connect(test_db_url)
    except Exception as e:
        db_url_masked = _mask_password(test_db_url)
        error_str = str(e).lower()
        # Check for SSL negotiation errors
        if any(term in error_str for term in ["ssl", "ssl negotiation", "could not send", "tls"]):
            pytest.skip(
                f"[SKIP][FO] SSL negotiation failed; check sslmode in DATABASE_URL (see FIRST_ORGANISM_ENV.md).{spark_indicator}\n"
                f"  Error: {e}\n"
                f"  Attempted URL: {db_url_masked}\n"
                f"  For local Docker: use ?sslmode=disable\n"
                f"  For remote DB: use ?sslmode=require"
            )
        else:
            db_url_trimmed = _trim_url_for_display(test_db_url)
            import re
            match = re.search(r"@([^:/]+)(?::(\d+))?", test_db_url)
            host_port = match.group(1) + ":" + (match.group(2) or "5432") if match else "unknown"
            pytest.skip(
                f"[SKIP][FO] Database connection failed (Postgres unreachable at {host_port}; "
                f"error={e}; mode=<connection_error>, db_url=<{db_url_trimmed}>) "
                f"(see scripts/start_first_organism_infra.ps1 or SPARK_INFRA_CHECKLIST.md)"
            )
        return

    try:
        yield conn
    finally:
        conn.close()


def _make_mock_connection():
    """Create a mock connection for offline/local testing."""

    class MockCursor:
        def execute(self, *args, **kwargs):
            pass

        def fetchone(self):
            return (0,)

        def fetchall(self):
            return []

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class MockConn:
        autocommit = False

        def cursor(self):
            return MockCursor()

        def commit(self):
            pass

        def rollback(self):
            pass

        def close(self):
            pass

    return MockConn()


# ---------------------------------------------------------------------------
# First Organism Database Fixture
# ---------------------------------------------------------------------------
@pytest.fixture(scope="function")
def first_organism_db(
    test_db_connection: psycopg.Connection,
    environment_mode: EnvironmentMode,
    test_db_url: str,
) -> Generator[psycopg.Connection, None, None]:
    """
    Prepare database for First Organism tests.

    - Assumes migrations are already applied by tests/conftest.py (via scripts/run-migrations.py)
    - Truncates relevant tables
    - Rolls back after test

    Skips with explicit [SKIP][FO] reason if:
    - FIRST_ORGANISM_TESTS not enabled
    - Mock mode is detected
    - Database is unavailable (Postgres/Redis down)
    
    This fixture ensures DB-dependent tests skip gracefully when infrastructure is unavailable.
    """
    # Use canonical gating function - this is the single source of truth for FO readiness
    assert_first_organism_ready(environment_mode, test_db_url)

    # 016_monotone_ledger.sql is now applied once by scripts/run-migrations.py
    # in tests/conftest.py. Re-running it here causes DuplicateObject errors
    # when constraints already exist, so we skip direct execution.
    # migration_path = Path("migrations/016_monotone_ledger.sql")
    # if not migration_path.exists():
    #     db_url_trimmed = _trim_url_for_display(test_db_url)
    #     pytest.skip(
    #         f"[SKIP][FO] Migration 016_monotone_ledger.sql not found "
    #         f"(mode=<migration_missing>, db_url=<{db_url_trimmed}>)"
    #     )

    original_autocommit = test_db_connection.autocommit
    test_db_connection.autocommit = True

    try:
        with test_db_connection.cursor() as cur:
            # Migration execution removed - handled by root tests/conftest.py
            # cur.execute(migration_path.read_text(encoding="utf-8"))
            cur.execute(
                """
                TRUNCATE block_proofs, block_statements, blocks, ledger_sequences,
                         proofs, dependencies, statements, runs, theories
                RESTART IDENTITY CASCADE
                """
            )
    except Exception as e:
        test_db_connection.autocommit = original_autocommit
        db_url_trimmed = _trim_url_for_display(test_db_url)
        pytest.skip(
            f"[SKIP][FO] Migration failed: {e} "
            f"(mode=<migration_error>, db_url=<{db_url_trimmed}>)"
        )

    test_db_connection.autocommit = original_autocommit

    yield test_db_connection

    # Cleanup
    test_db_connection.rollback()
    try:
        with test_db_connection.cursor() as cur:
            cur.execute(
                """
                TRUNCATE block_proofs, block_statements, blocks, ledger_sequences,
                         proofs, dependencies, statements, runs, theories
                RESTART IDENTITY CASCADE
                """
            )
    except Exception:
        pass  # Best effort cleanup


# ---------------------------------------------------------------------------
# Environment Fixture
# ---------------------------------------------------------------------------
@pytest.fixture()
def first_organism_env(
    test_db_url: str,
    environment_mode: EnvironmentMode,
    monkeypatch,
) -> Generator[Dict[str, Any], None, None]:
    """
    Configure environment for First Organism tests.

    Sets:
    - DATABASE_URL
    - REDIS_URL (with graceful skip if unavailable)
    - LEDGER_API_KEY
    - MDAP_SEED (deterministic seed)

    Yields:
        Dictionary with environment status and MDAP metadata
    """
    monkeypatch.setenv("DATABASE_URL", test_db_url)
    monkeypatch.setenv("REDIS_URL", environment_mode.redis_url)
    monkeypatch.setenv("LEDGER_API_KEY", os.getenv("LEDGER_API_KEY", "first-organism"))
    monkeypatch.setenv("MDAP_SEED", str(MDAP_EPOCH_SEED))

    env_status = {
        "mode": environment_mode.to_dict(),
        "mdap_metadata": environment_mode.to_mdap_metadata(),
        "db_url": test_db_url,
        "redis_url": environment_mode.redis_url,
        "redis_available": environment_mode.is_redis_available,
        "mdap_seed": MDAP_EPOCH_SEED,
        "chain_status": environment_mode.chain_status.value,
    }

    yield env_status


# ---------------------------------------------------------------------------
# Curriculum Slice Builders
# ---------------------------------------------------------------------------
def build_first_organism_slice() -> CurriculumSlice:
    """
    Construct the canonical First Organism curriculum slice.

    This slice is permissive enough to allow derivation runs while
    still requiring measurable coverage, abstention, and velocity.
    """
    gates = SliceGates(
        coverage=CoverageGateSpec(
            ci_lower_min=0.50,
            sample_min=1,
            require_attestation=False,
        ),
        abstention=AbstentionGateSpec(
            max_rate_pct=95.0,
            max_mass=1000,
        ),
        velocity=VelocityGateSpec(
            min_pph=0.1,
            stability_cv_max=0.9,
            window_minutes=5,
        ),
        caps=CapsGateSpec(
            min_attempt_mass=1,
            min_runtime_minutes=0.01,
            backlog_max=0.99,
        ),
    )
    params = {
        "atoms": 2,
        "depth_max": 2,
        "breadth_max": 8,
        "total_max": 8,
    }
    return CurriculumSlice(
        name="first-organism-test",
        params=params,
        gates=gates,
    )


def build_first_organism_metrics(
    coverage_ci: float = 0.90,
    sample_size: int = 2,
    abstention_rate: float = 10.0,
    attempt_mass: int = 1,
    proof_velocity_pph: float = 0.5,
    velocity_cv: float = 0.5,
    runtime_minutes: float = 0.1,
    backlog_fraction: float = 0.1,
    attestation_hash: Optional[str] = None,
) -> NormalizedMetrics:
    """
    Build normalized metrics for First Organism gate evaluation.
    """
    return NormalizedMetrics(
        coverage_ci_lower=coverage_ci,
        coverage_sample_size=sample_size,
        abstention_rate_pct=abstention_rate,
        attempt_mass=attempt_mass,
        slice_runtime_minutes=runtime_minutes,
        proof_velocity_pph=proof_velocity_pph,
        velocity_cv=velocity_cv,
        backlog_fraction=backlog_fraction,
        attestation_hash=attestation_hash,
    )


# ---------------------------------------------------------------------------
# Attestation Artifact
# ---------------------------------------------------------------------------
@dataclass
class FirstOrganismAttestation:
    """
    Canonical First Organism attestation artifact.

    This is the single source of truth for attestation data that
    Cursor P and other consumers rely on.
    """
    # Core attestation roots
    statement_hash: str
    reasoning_root: str  # R_t
    ui_root: str  # U_t
    composite_root: str  # H_t = SHA256(R_t || U_t)

    # Determinism metadata
    mdap_seed: int
    run_id: str
    run_timestamp_iso: str
    run_timestamp_unix: int

    # Block/proof pointers (if DB is live)
    block_id: Optional[int]
    proof_id: Optional[int]
    statement_id: Optional[int]

    # Environment context
    version: str
    environment_mode: str
    chain_status: str
    slice_name: str

    # Synthetic flag (True if roots are synthetic due to ABSTAIN)
    is_synthetic: bool = False

    # MDAP metadata
    mdap_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "statement_hash": self.statement_hash,
            "R_t": self.reasoning_root,
            "U_t": self.ui_root,
            "H_t": self.composite_root,
            "mdap_seed": self.mdap_seed,
            "run_id": self.run_id,
            "run_timestamp_iso": self.run_timestamp_iso,
            "run_timestamp_unix": self.run_timestamp_unix,
            "block_id": self.block_id,
            "proof_id": self.proof_id,
            "statement_id": self.statement_id,
            "version": self.version,
            "environment_mode": self.environment_mode,
            "chain_status": self.chain_status,
            "slice_name": self.slice_name,
            "is_synthetic": self.is_synthetic,
            "mdap_metadata": self.mdap_metadata,
        }

    def write_artifact(self, path: Optional[Path] = None) -> Path:
        """
        Write attestation artifact to disk.

        Args:
            path: Optional path override (defaults to artifacts/first_organism/attestation.json)

        Returns:
            Path to written artifact
        """
        if path is None:
            path = Path("artifacts/first_organism/attestation.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    def short_h_t(self, length: int = 12) -> str:
        """Return truncated H_t for logging."""
        return self.composite_root[:length]


def build_first_organism_attestation(
    statement_hash: str,
    reasoning_root: str,
    ui_root: str,
    composite_root: str,
    *,
    block_id: Optional[int] = None,
    proof_id: Optional[int] = None,
    statement_id: Optional[int] = None,
    environment_mode: str = "real",
    chain_status: str = "ready",
    slice_name: str = "first-organism-test",
    is_synthetic: bool = False,
    mdap_metadata: Optional[Dict[str, Any]] = None,
) -> FirstOrganismAttestation:
    """
    Build a canonical First Organism attestation artifact.

    All timestamps and IDs are derived deterministically from MDAP helpers.

    Args:
        statement_hash: Hash of the derived statement
        reasoning_root: R_t Merkle root
        ui_root: U_t Merkle root
        composite_root: H_t = SHA256(R_t || U_t)
        block_id: Optional block ID (if DB is live)
        proof_id: Optional proof ID (if DB is live)
        statement_id: Optional statement ID (if DB is live)
        environment_mode: "real", "mock", or "offline"
        chain_status: "ready", "abstain", "skip", or "mock"
        slice_name: Curriculum slice name
        is_synthetic: Whether roots are synthetic (ABSTAIN mode)
        mdap_metadata: Additional MDAP-traceable metadata

    Returns:
        FirstOrganismAttestation instance
    """
    run_id = mdap_deterministic_id("run", statement_hash, composite_root)
    seed = deterministic_seed_from_content(composite_root, FIRST_ORGANISM_NAMESPACE)
    run_timestamp_unix = deterministic_unix_timestamp(seed)
    run_timestamp_iso = deterministic_timestamp(seed).isoformat()

    return FirstOrganismAttestation(
        statement_hash=statement_hash,
        reasoning_root=reasoning_root,
        ui_root=ui_root,
        composite_root=composite_root,
        mdap_seed=MDAP_EPOCH_SEED,
        run_id=run_id,
        run_timestamp_iso=run_timestamp_iso,
        run_timestamp_unix=run_timestamp_unix,
        block_id=block_id,
        proof_id=proof_id,
        statement_id=statement_id,
        version=FIRST_ORGANISM_VERSION,
        environment_mode=environment_mode,
        chain_status=chain_status,
        slice_name=slice_name,
        is_synthetic=is_synthetic,
        mdap_metadata=mdap_metadata or {},
    )


# ---------------------------------------------------------------------------
# First Organism Attestation Context Fixture
# ---------------------------------------------------------------------------
@pytest.fixture()
def first_organism_attestation_context(
    first_organism_db: psycopg.Connection,
    environment_mode: EnvironmentMode,
) -> Generator[Dict[str, Any], None, None]:
    """
    Execute the First Organism derivation pipeline and produce attestation context.

    This fixture is DETERMINISTIC regardless of infrastructure availability:
    - READY mode: Full chain with real DB writes
    - ABSTAIN mode: Derivation + synthetic attestation
    - MOCK mode: Fully synthetic context

    Returns a dictionary containing:
    - candidate_hash: Hash of the derived statement
    - block: Sealed block record (or None in ABSTAIN/MOCK)
    - statement: Derived statement record
    - proof: Ingested proof record (or None in ABSTAIN/MOCK)
    - gate_statuses: Curriculum gate evaluation results
    - ui_event_id: UI event identifier
    - curriculum_slice_name: Active slice name
    - attestation: FirstOrganismAttestation instance
    - environment_mode: Current environment mode
    - chain_status: "ready", "abstain", or "mock"
    """
    ui_event_store.clear()

    # Bounded derivation pipeline - always runs
    bounds = SliceBounds(
        max_atoms=2,
        max_formula_depth=2,
        max_mp_depth=2,
        max_breadth=8,
        max_total=8,
    )
    verifier = StatementVerifier(bounds)
    pipeline = DerivationPipeline(bounds, verifier)
    outcome = pipeline.run_step(existing=[])

    if not outcome.statements:
        pytest.skip(
            "[SKIP][FO] Derivation pipeline produced no statements "
            "(mode=<derivation_empty>, db_url=<N/A>)"
        )

    candidate = outcome.statements[0]

    # Deterministic UI event - always captured
    ui_event_id = mdap_deterministic_id("ui-event", candidate.hash)
    ui_event_timestamp = mdap_deterministic_timestamp(candidate.hash)
    ui_event_payload = {
        "event_id": ui_event_id,
        "actor": "first-organism-test",
        "kind": "select_statement",
        "target_hash": candidate.hash,
        "meta": {"origin": "first-organism-integration"},
        "timestamp": ui_event_timestamp,
    }
    ui_record = capture_ui_event(ui_event_payload)
    ui_artifact = ui_record.to_artifact()

    # Slice config and metrics - always computed
    slice_cfg = build_first_organism_slice()

    # Branch based on chain status
    if environment_mode.is_ready:
        # Full chain execution
        ingestor = LedgerIngestor()
        with first_organism_db.cursor() as cur:
            ingest_outcome = ingestor.ingest(
                cur,
                theory_name="Propositional",
                ascii_statement=candidate.pretty or candidate.normalized,
                proof_text="lean abstained",
                prover="lean4",
                status="failure",
                module_name=f"ML.Jobs.first_organism.{candidate.hash[:8]}",
                stdout="",
                stderr="lean: synthetic abstention",
                derivation_rule=candidate.rule,
                derivation_depth=candidate.mp_depth,
                method="integration-test",
                duration_ms=1,
                truth_domain="classical",
                ui_events=[ui_artifact],
                sealed_by="first-organism-test",
            )
        first_organism_db.commit()

        block = ingest_outcome.block
        proof = ingest_outcome.proof
        r_t = block.reasoning_root
        u_t = block.ui_merkle_root
        h_t = block.composite_root
        block_id = block.id
        proof_id = getattr(proof, "id", None)
        statement_id = getattr(ingest_outcome.statement, "id", None)
        is_synthetic = False
        chain_status_str = "ready"
    else:
        # ABSTAIN or MOCK mode - synthetic attestation
        r_t, u_t, h_t = mdap_synthetic_roots(candidate.hash)
        block = None
        proof = None
        block_id = None
        proof_id = None
        statement_id = None
        is_synthetic = True
        chain_status_str = environment_mode.chain_status.value

        log_first_organism_abstain(
            environment_mode.abstain_reason or f"[ABSTAIN] Chain status: {chain_status_str}",
            candidate.hash,
            h_t,
        )

    # Gate evaluation - always computed
    normalized_metrics = build_first_organism_metrics(attestation_hash=h_t)
    gate_statuses = GateEvaluator(normalized_metrics, slice_cfg).evaluate()

    # Build canonical attestation artifact
    attestation = build_first_organism_attestation(
        statement_hash=candidate.hash,
        reasoning_root=r_t,
        ui_root=u_t,
        composite_root=h_t,
        block_id=block_id,
        proof_id=proof_id,
        statement_id=statement_id,
        environment_mode="real" if environment_mode.is_real_db else "mock",
        chain_status=chain_status_str,
        slice_name=slice_cfg.name,
        is_synthetic=is_synthetic,
        mdap_metadata=environment_mode.to_mdap_metadata(),
    )

    context = {
        "candidate_hash": candidate.hash,
        "block": block,
        "statement": candidate,
        "proof": proof,
        "gate_statuses": gate_statuses,
        "ui_event_id": ui_record.event_id,
        "curriculum_slice_name": slice_cfg.name,
        "derivation_outcome": outcome,
        "slice_cfg": slice_cfg,
        "normalized_metrics": normalized_metrics,
        "attestation": attestation,
        "environment_mode": environment_mode,
        "chain_status": chain_status_str,
        "is_synthetic": is_synthetic,
    }

    try:
        yield context
    finally:
        ui_event_store.clear()


# ---------------------------------------------------------------------------
# Test Client Fixture
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def migrated_test_db(test_db_connection: psycopg.Connection):
    """
    Run migrations on test database (best-effort).
    """
    migration_files = [
        "migrations/001_init.sql",
        "migrations/002_blocks_lemmas.sql",
        "migrations/003_add_system_id.sql",
        "migrations/004_finalize_core_schema.sql",
    ]
    for mf in migration_files:
        p = Path(mf)
        if not p.exists():
            continue
        sql = p.read_text(encoding="utf-8").lstrip("\ufeff")  # strip BOM
        try:
            with test_db_connection.cursor() as cur:
                cur.execute(sql)
        except Exception as e:
            print(f"[migration skip] {p.name}: {e}")
            test_db_connection.rollback()


@pytest.fixture()
def api_headers() -> Dict[str, str]:
    """Standard API headers for authenticated requests."""
    return {"X-API-Key": os.getenv("LEDGER_API_KEY", "devkey")}


@pytest.fixture()
def test_client(
    migrated_test_db, test_db_url: str, test_db_connection: psycopg.Connection
) -> Generator[TestClient, None, None]:
    """
    FastAPI test client with database dependency override.
    """

    def _get_test_db_connection() -> Generator[psycopg.Connection, None, None]:
        if test_db_url.startswith("mock://"):
            yield _make_mock_connection()
            return
        # Reuse session-scoped connection for consistency
        yield test_db_connection

    app.dependency_overrides[get_db_connection] = _get_test_db_connection

    with TestClient(app) as client:
        yield client

    app.dependency_overrides.pop(get_db_connection, None)


# ---------------------------------------------------------------------------
# Certification Logging Helpers - Complete Pipeline
# ---------------------------------------------------------------------------
def log_first_organism_pass(h_t: str, short_length: int = 12) -> None:
    """
    Log the canonical First Organism PASS status in the investor-grade tone.

    This is the single, unmistakable line that Cursor P looks for to certify
    Wave-1 readiness.

    Args:
        h_t: Composite attestation root (full 64-char hex)
        short_length: Length of H_t to include in log (default 12)
    """
    short_h_t = h_t[:short_length]
    # Exact format required by SPARK mission: [PASS] FIRST ORGANISM ALIVE H_t=<short-hex>
    import sys
    GREEN = "\033[92m"
    RESET = "\033[0m"
    USE_COLOR = sys.stdout.isatty() or os.environ.get("PYTEST_COLOR") == "yes"
    color = GREEN if USE_COLOR else ""
    reset = RESET if USE_COLOR else ""
    sys.stdout.write(f"{color}[PASS] FIRST ORGANISM ALIVE H_t={short_h_t}{reset}\n")
    sys.stdout.flush()


def log_first_organism_skip(reason: str) -> None:
    """
    Log a First Organism SKIP message with an explicit reason.

    Args:
        reason: Reason for skipping
    """
    log_first_organism_phase("SKIP", f"First Organism skipped: {reason}", status="WARN")


def log_first_organism_abstain(reason: str, statement_hash: str, h_t: str) -> None:
    """
    Log a First Organism ABSTAIN message with synthetic attestation.

    ABSTAIN means infrastructure was partially available, so we generated
    synthetic attestation roots for traceability.

    Args:
        reason: Reason for abstaining
        statement_hash: Hash of the derived statement
        h_t: Synthetic composite root
    """
    log_first_organism_phase("ABSTAIN", f"First Organism abstain: {reason}", status="ABSTAIN")
    log_first_organism_phase(
        "ABSTAIN",
        f"Synthetic H_t={h_t[:12]} for statement={statement_hash[:12]}",
        status="ABSTAIN",
    )


def log_first_organism_phase(phase: str, message: str, status: str = "INFO") -> None:
    """
    Emit a phase-specific status indicator in the investor-grade tone.

    Args:
        phase: Phase name (e.g., "GATE", "DERIVE", "ATTEST", "RFL")
        message: Message to log
        status: "PASS", "FAIL", "WARN", "ABSTAIN", or "INFO"
    """
    import sys

    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    RESET = "\033[0m"
    USE_COLOR = sys.stdout.isatty() or os.environ.get("PYTEST_COLOR") == "yes"

    color = ""
    if USE_COLOR:
        if status == "PASS":
            color = GREEN
        elif status == "FAIL":
            color = RED
        elif status in {"WARN", "ABSTAIN"}:
            color = YELLOW
        elif status == "INFO":
            color = CYAN

    reset = RESET if USE_COLOR else ""
    sys.stdout.write(f"{color}[{phase}] {message}{reset}\n")
    sys.stdout.flush()


def write_first_organism_artifact(attestation: FirstOrganismAttestation, extra: Optional[Dict[str, Any]] = None) -> Path:
    """
    Write the canonical First Organism attestation artifact.

    This is the single JSON file that Cursor P relies on to certify Wave-1.

    Args:
        attestation: FirstOrganismAttestation instance
        extra: Optional additional metadata to include

    Returns:
        Path to written artifact
    """
    artifact_dir = Path("artifacts/first_organism")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / "attestation.json"

    payload = attestation.to_dict()
    if extra:
        payload["extra"] = extra

    # Add component versions for auditing
    payload["components"] = {
        "derivation": "axiom_engine",
        "ledger": "LedgerIngestor",
        "attestation": "attestation.dual_root",
        "rfl": "RFLRunner",
    }

    artifact_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log_first_organism_phase("ARTIFACT", f"Written to {artifact_path}", "INFO")
    return artifact_path


# ---------------------------------------------------------------------------
# Pytest Marker Configuration
# ---------------------------------------------------------------------------
def pytest_configure(config):
    """Register First Organism and Wide Slice markers."""
    config.addinivalue_line(
        "markers",
        "first_organism: mark test as First Organism integration test",
    )
    config.addinivalue_line(
        "markers",
        "first_organism_smoke: mark test as DB-optional smoke test for First Organism (no DB required)",
    )
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration test",
    )
    config.addinivalue_line(
        "markers",
        "determinism: mark test as determinism verification test",
    )
    config.addinivalue_line(
        "markers",
        "wide_slice: mark test as Wide Slice Dyno Chart log validation test",
    )


# ---------------------------------------------------------------------------
# First Organism API Client
# ---------------------------------------------------------------------------
@dataclass
class UIEventResponse:
    """Response from posting a UI event."""
    event_id: str
    timestamp: int
    leaf_hash: str


@dataclass
class UIEventEntry:
    """Entry in the UI events list."""
    event_id: str
    timestamp: int
    leaf_hash: str


@dataclass
class UIEventsListResponse:
    """Response from listing UI events."""
    events: List[UIEventEntry]


@dataclass
class DerivationSimulationResponse:
    """Response from derivation simulation."""
    triggered: bool
    job_id: str
    status: str


class FirstOrganismApiClient:
    """
    Typed API client for First Organism integration tests.

    Wraps TestClient with typed request/response methods for the
    First Organism API surface.
    """

    def __init__(self, client: TestClient, api_key: str = "devkey"):
        """
        Initialize the API client.

        Args:
            client: FastAPI TestClient
            api_key: API key for authenticated endpoints
        """
        self._client = client
        self._headers = {"X-API-Key": api_key}

    def post_ui_event(self, payload: Dict[str, Any]) -> UIEventResponse:
        """
        Post a UI event and return typed response.

        Args:
            payload: Event payload dictionary

        Returns:
            UIEventResponse with event_id, timestamp, and leaf_hash
        """
        response = self._client.post(
            "/ui-events",
            json=payload,
            headers=self._headers,
        )
        response.raise_for_status()
        data = response.json()
        return UIEventResponse(
            event_id=data.get("event_id", ""),
            timestamp=data.get("timestamp", 0),
            leaf_hash=data.get("leaf_hash", ""),
        )

    def list_ui_events(self) -> UIEventsListResponse:
        """
        List all UI events.

        Returns:
            UIEventsListResponse with list of events
        """
        response = self._client.get("/ui-events", headers=self._headers)
        response.raise_for_status()
        data = response.json()
        events = [
            UIEventEntry(
                event_id=e.get("event_id", ""),
                timestamp=e.get("timestamp", 0),
                leaf_hash=e.get("leaf_hash", ""),
            )
            for e in data.get("events", [])
        ]
        return UIEventsListResponse(events=events)

    def simulate_derivation(self) -> DerivationSimulationResponse:
        """
        Trigger a derivation simulation.

        Returns:
            DerivationSimulationResponse with trigger status
        """
        response = self._client.post(
            "/simulate-derivation",
            headers=self._headers,
        )
        response.raise_for_status()
        data = response.json()
        return DerivationSimulationResponse(
            triggered=data.get("triggered", False),
            job_id=data.get("job_id", ""),
            status=data.get("status", ""),
        )


@pytest.fixture()
def api_client(test_client: TestClient) -> FirstOrganismApiClient:
    """
    Provide a typed First Organism API client.

    Args:
        test_client: FastAPI TestClient fixture

    Returns:
        FirstOrganismApiClient instance
    """
    return FirstOrganismApiClient(test_client)


def pytest_collection_modifyitems(config, items):
    """
    Automatically apply first_organism and wide_slice marker handling.

    SPARK tests (First Organism):
    - Tests marked with @pytest.mark.first_organism will be:
      - Skipped if FIRST_ORGANISM_TESTS is not set to 'true' (or SPARK_RUN=1 or .spark_run_enable exists)
    - Hermetic tests (standalone, determinism) run when FIRST_ORGANISM_TESTS=true even without DB
    - DB-dependent tests (happy_path) are skipped with clear [SKIP] message when Postgres/Redis are down
    
    WIDE_SLICE tests (Dyno Chart log validation):
    - Tests marked with @pytest.mark.wide_slice will be:
      - Skipped if WIDE_SLICE_TESTS is not set to 'true' (unless explicitly selected with -m wide_slice)
    - These tests do NOT require DB/Redis - they only validate JSONL file structure
    """
    # SPARK/First Organism gating
    first_organism_skip_marker = pytest.mark.skip(
        reason="[SKIP][FO] FIRST_ORGANISM_TESTS not set to true/SPARK_RUN; refusing to run by default. "
               "(Set FIRST_ORGANISM_TESTS=true, SPARK_RUN=1, or create .spark_run_enable to enable)"
    )

    first_organism_env = os.getenv("FIRST_ORGANISM_TESTS", "").lower()
    spark_file_trigger = Path(".spark_run_enable").is_file()
    first_organism_enabled = (
        first_organism_env == "true"
        or os.getenv("SPARK_RUN", "") == "1"
        or spark_file_trigger
    )

    # WIDE_SLICE gating
    wide_slice_skip_marker = pytest.mark.skip(
        reason="Wide Slice tests disabled. Set WIDE_SLICE_TESTS=true or use -m wide_slice to enable."
    )
    
    wide_slice_env = os.getenv("WIDE_SLICE_TESTS", "").lower()
    # Check if wide_slice marker is explicitly requested via -m
    # If -m is used with wide_slice, pytest will filter to those tests automatically,
    # so we only need to check the env var for when running all tests
    marker_expr = config.getoption("-m", default="")
    wide_slice_marker_requested = marker_expr and ("wide_slice" in marker_expr or marker_expr == "wide_slice")
    wide_slice_enabled = (
        wide_slice_env == "true"
        or wide_slice_marker_requested
    )

    for item in items:
        # Handle First Organism/SPARK tests
        if "first_organism" in item.keywords:
            if not first_organism_enabled:
                item.add_marker(first_organism_skip_marker)
        
        # Handle Wide Slice tests
        if "wide_slice" in item.keywords:
            if not wide_slice_enabled:
                item.add_marker(wide_slice_skip_marker)
