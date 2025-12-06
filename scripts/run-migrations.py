#!/usr/bin/env python3
"""Run all database migrations in order with version tracking."""

import os
import psycopg
import re
import hashlib
import time
from pathlib import Path
from typing import Optional, Tuple, List

def _require_database_url() -> str:
    """Return DATABASE_URL or raise if not set (zero-trust policy)."""
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "[FATAL] DATABASE_URL environment variable is not set. "
            "Set it explicitly before running migrations."
        )
    return url


DATABASE_URL = _require_database_url()

def compute_checksum(content: str) -> str:
    """Compute SHA-256 checksum of migration content."""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()

def ensure_schema_migrations_table(conn):
    """Create schema_migrations table if it doesn't exist (pre-flight check)."""
    with conn.cursor() as cur:
        try:
            # Try to query the table to see if it exists
            cur.execute("SELECT 1 FROM schema_migrations LIMIT 1")
        except psycopg.errors.UndefinedTable:
            # Table doesn't exist - create it
            print("  Creating schema_migrations table (fresh DB detected)...")
            cur.execute("""
                CREATE TABLE schema_migrations (
                    version TEXT PRIMARY KEY,
                    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    checksum TEXT,
                    duration_ms INTEGER,
                    status TEXT DEFAULT 'success' CHECK (status IN ('success', 'failed', 'skipped'))
                )
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_schema_migrations_applied_at 
                ON schema_migrations(applied_at DESC)
            """)
            conn.commit()
            print("  ✅ schema_migrations table created successfully")

def migration_already_applied(cur, version: str) -> bool:
    """Check if migration has already been applied."""
    try:
        cur.execute(
            "SELECT 1 FROM schema_migrations WHERE version = %s",
            (version,)
        )
        return cur.fetchone() is not None
    except psycopg.errors.UndefinedTable:
        # schema_migrations table doesn't exist yet (000 migration not run)
        return False

def record_migration(cur, version: str, checksum: str, duration_ms: int, status: str):
    """Record migration in schema_migrations table."""
    try:
        cur.execute(
            """
            INSERT INTO schema_migrations (version, checksum, duration_ms, status)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (version) DO UPDATE
            SET applied_at = NOW(), checksum = EXCLUDED.checksum,
                duration_ms = EXCLUDED.duration_ms, status = EXCLUDED.status
            """,
            (version, checksum, duration_ms, status)
        )
    except psycopg.errors.UndefinedTable:
        # schema_migrations doesn't exist yet - skip recording
        pass

def check_all_migrations_applied(conn: psycopg.Connection) -> Tuple[bool, Optional[int]]:
    """
    Check if all migrations have been applied.
    
    Args:
        conn: Database connection
        
    Returns:
        Tuple of (all_applied: bool, count: Optional[int])
        count is None if schema_migrations table doesn't exist
    """
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM schema_migrations WHERE status = 'success'")
            count = cur.fetchone()[0]
            # Get total migration files
            migrations_dir = Path("migrations")
            if not migrations_dir.exists():
                return False, 0
            migration_files = list(migrations_dir.glob("*.sql"))
            # Exclude baseline files
            migration_files = [f for f in migration_files if not f.name.startswith("baseline")]
            total = len(migration_files)
            return count >= total, count
    except psycopg.errors.UndefinedTable:
        return False, None

def run_migration_file(filename, db_url: Optional[str] = None, quiet: bool = False):
    """Run a migration file as one transaction with version tracking."""
    version = Path(filename).stem  # e.g., "001_init"
    if not quiet:
        print(f"Running migration: {filename}")

    db_url_to_use = db_url or DATABASE_URL
    try:
        with psycopg.connect(db_url_to_use, connect_timeout=10) as conn:
            with conn.cursor() as cur:
                # Check if already applied
                if migration_already_applied(cur, version):
                    if not quiet:
                        print(f"  Migration {version} already applied, skipping...")
                    return True

                with open(filename, "r") as f:
                    migration_sql = f.read()

                checksum = compute_checksum(migration_sql)

                # Remove only standalone comment lines (not inline comments)
                lines = migration_sql.split('\n')
                cleaned_lines = []
                in_dollar_quote = False
                dollar_delimiter = None

                for line in lines:
                    stripped = line.strip()

                    # Check for dollar-quoted string boundaries
                    if '$$' in line:
                        parts = line.split('$$')
                        if len(parts) >= 2:
                            # Look for the delimiter pattern
                            for i in range(0, len(parts), 2):
                                if i + 1 < len(parts):
                                    delimiter = f"$${parts[i+1]}$$" if parts[i+1] else "$$"
                                    if delimiter in line:
                                        if not in_dollar_quote:
                                            in_dollar_quote = True
                                            dollar_delimiter = delimiter
                                        elif delimiter == dollar_delimiter:
                                            in_dollar_quote = False
                                            dollar_delimiter = None
                                        break

                    # Skip standalone comment lines only when not in dollar quotes
                    if not in_dollar_quote and stripped.startswith('--') and not stripped.startswith('-- '):
                        continue

                    cleaned_lines.append(line)

                # Join back the cleaned SQL
                cleaned_sql = '\n'.join(cleaned_lines)

                # Execute the entire migration as one transaction
                start_time = time.time()
                try:
                    print(f"  Executing migration file...")
                    cur.execute(cleaned_sql)
                    duration_ms = int((time.time() - start_time) * 1000)

                    # Record successful migration
                    record_migration(cur, version, checksum, duration_ms, 'success')

                    conn.commit()
                    print(f"  ✅ Migration {version} completed successfully! ({duration_ms}ms)")
                    return True
                except Exception as e:
                    duration_ms = int((time.time() - start_time) * 1000)
                    print(f"  ❌ Error in migration {filename}: {e}")
                    print(f"  Error type: {type(e).__name__}")

                    # Rollback the failed transaction immediately
                    conn.rollback()
                    
                    # Try to record the failed migration (may fail if schema_migrations doesn't exist)
                    try:
                        record_migration(cur, version, checksum, duration_ms, 'failed')
                        conn.commit()
                    except Exception as record_err:
                        # If recording fails, that's okay - we've already rolled back
                        print(f"  ⚠️  Could not record failed migration: {record_err}")
                    
                    return False
    except Exception as e:
        print(f"  Database connection failed for {filename}: {e}")
        return False

def main():
    """Run all migrations in order."""
    migrations_dir = Path("migrations")
    if not migrations_dir.exists():
        print("No migrations directory found")
        return

    # Pre-flight: Ensure schema_migrations table exists before processing any migrations
    print("Pre-flight check: Ensuring schema_migrations table exists...")
    try:
        with psycopg.connect(DATABASE_URL, connect_timeout=10) as conn:
            ensure_schema_migrations_table(conn)
    except Exception as e:
        print(f"  ❌ Failed to ensure schema_migrations table: {e}")
        print("  Cannot proceed with migrations without tracking table.")
        exit(1)

    # Get all .sql files and sort them
    migration_files = list(migrations_dir.glob("*.sql"))
    migration_files.sort(key=lambda x: int(re.findall(r'\d+', x.name)[0]) if re.findall(r'\d+', x.name) else 999)

    print(f"\nFound {len(migration_files)} migration files:")
    for file in migration_files:
        print(f"  - {file.name}")

    failed_migrations = []
    successful_migrations = []

    for migration_file in migration_files:
        try:
            success = run_migration_file(migration_file)
            if success:
                successful_migrations.append(migration_file.name)
            else:
                failed_migrations.append(migration_file.name)
                # Fail fast: stop processing remaining migrations on first failure
                print(f"\n  🛑 Migration {migration_file.name} failed. Stopping migration run.")
                print(f"  Remaining migrations will not be executed to prevent cascading errors.")
                break
        except Exception as e:
            print(f"  ❌ Fatal error running {migration_file.name}: {e}")
            print(f"  Error type: {type(e).__name__}")
            failed_migrations.append(migration_file.name)
            # Fail fast on fatal errors
            break

    print("\n" + "="*60)
    print(f"Migration summary: {len(successful_migrations)} successful, {len(failed_migrations)} failed")
    if failed_migrations:
        print(f"Failed migrations: {', '.join(failed_migrations)}")
        exit(1)

    # Show migration summary from schema_migrations table
    try:
        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT version, status, applied_at, duration_ms
                    FROM schema_migrations
                    ORDER BY applied_at DESC
                    LIMIT 10
                    """
                )
                rows = cur.fetchall()
                if rows:
                    print("\nRecent migrations:")
                    for row in rows:
                        version, status, applied_at, duration_ms = row
                        status_icon = "✅" if status == "success" else "❌"
                        print(f"  {status_icon} {version} ({status}) - {duration_ms}ms at {applied_at}")
    except Exception:
        # schema_migrations table doesn't exist yet
        pass

if __name__ == "__main__":
    main()
