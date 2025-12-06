# tests/conftest.py
import os
import subprocess
import time
import pytest

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker, Session

# Register First Organism telemetry plugin for automatic metrics capture
pytest_plugins = [
    "tests.plugins.first_organism_telemetry_hook",
]

# ---- 1) Resolve the real Postgres URL (no SQLite fallback)
# URL Resolution Precedence:
# 1. DATABASE_URL_TEST (for test-specific overrides)
# 2. DATABASE_URL (standard environment variable)
# 3. Empty string (no default - tests must set env var or will fail fast)
# 
# Note: For integration tests, see tests/integration/conftest.py which provides
# a default fallback aligned with First Organism docker-compose.yml (port 5432).
DB_URL = os.environ.get("DATABASE_URL_TEST") or os.environ.get("DATABASE_URL") or ""

# ---- 2) Run migrations ONCE before any tests
_MIGRATIONS_DONE = False

def _run_migrations_once():
    global _MIGRATIONS_DONE
    if _MIGRATIONS_DONE:
        return
    # Prefer your Python migration runner (handles dollar-quoted SQL)
    # This should be idempotent.
    subprocess.run(
        ["uv", "run", "python", "scripts/run-migrations.py"],
        check=True,
        capture_output=False,
    )
    _MIGRATIONS_DONE = True

@pytest.fixture(scope="session")
def db_engine():
    _run_migrations_once()
    eng = create_engine(DB_URL, future=True, pool_pre_ping=True)
    # quick connectivity check
    try:
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as e:
        # TEMP: surface the real error instead of skipping
        # This helps debug Windows/Docker socket issues that were being masked as SSL errors
        raise
    return eng

@pytest.fixture(scope="function")
def db_session(db_engine) -> Session:
    """
    Function-scoped Session with a top-level transaction rolled back after each test.
    This uses a SAVEPOINT so ORM inspection & flushes behave like production.
    """
    connection = db_engine.connect()
    # Begin a non-ORM transaction
    trans = connection.begin()
    # Bind a Session to this connection
    TestingSessionLocal = sessionmaker(bind=connection, autoflush=True, autocommit=False, future=True)
    session: Session = TestingSessionLocal()

    # Each test runs in a SAVEPOINT; if test uses session.begin_nested() this ensures nested behavior
    nested = connection.begin_nested()

    @event.listens_for(session, "after_transaction_end")
    def restart_savepoint(sess, transaction):
        # Restart SAVEPOINT when the nested transaction ends
        nonlocal nested
        if not nested.is_active:
            nested = connection.begin_nested()

    try:
        yield session
    finally:
        session.close()
        # Roll back the outer transaction => DB is clean for the next test
        trans.rollback()
        connection.close()
