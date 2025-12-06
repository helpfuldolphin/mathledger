#!/usr/bin/env python3
"""
Run all database migrations in order with baseline support.

This migration runner supports both:
1. Baseline migration approach (fresh installs)
2. Legacy migration sequence (existing databases)

When baseline_20251019.sql is detected, all previous migrations (001-014) are skipped.
"""

import os
import psycopg
import re
from pathlib import Path

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

def check_baseline_applied(conn):
    """Check if baseline migration has been applied."""
    try:
        with conn.cursor() as cur:
            # Check if schema_migrations table exists
            cur.execute("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables 
                    WHERE table_name = 'schema_migrations'
                )
            """)
            if not cur.fetchone()[0]:
                return False
            
            # Check if baseline_20251019 is recorded
            cur.execute("""
                SELECT EXISTS (
                    SELECT 1 FROM schema_migrations 
                    WHERE version = 'baseline_20251019'
                )
            """)
            return cur.fetchone()[0]
    except Exception as e:
        print(f"  Warning: Could not check baseline status: {e}")
        return False

def run_migration_file(filename):
    """Run a migration file as one transaction."""
    print(f"Running migration: {filename}")
    try:
        with psycopg.connect(DATABASE_URL, connect_timeout=10) as conn:
            with conn.cursor() as cur:
                with open(filename, "r") as f:
                    migration_sql = f.read()

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
                try:
                    print(f"  Executing migration file...")
                    cur.execute(cleaned_sql)
                    conn.commit()
                    print(f"  Migration {filename} completed successfully!")
                    return True
                except Exception as e:
                    print(f"  Error in migration {filename}: {e}")
                    conn.rollback()
                    return False
    except Exception as e:
        print(f"  Database connection failed for {filename}: {e}")
        return False

def main():
    """Run all migrations in order with baseline support."""
    migrations_dir = Path("migrations")
    if not migrations_dir.exists():
        print("No migrations directory found")
        return

    # Get all .sql files and sort them
    migration_files = list(migrations_dir.glob("*.sql"))
    migration_files.sort(key=lambda x: int(re.findall(r'\d+', x.name)[0]) if re.findall(r'\d+', x.name) else 999)

    print(f"Found {len(migration_files)} migration files")
    
    # Check if baseline is applied
    try:
        with psycopg.connect(DATABASE_URL, connect_timeout=10) as conn:
            baseline_applied = check_baseline_applied(conn)
    except Exception as e:
        print(f"Warning: Could not connect to database: {e}")
        baseline_applied = False

    if baseline_applied:
        print("\n=== BASELINE MIGRATION DETECTED ===")
        print("Skipping legacy migrations (001-014)")
        print("Only running migrations after baseline_20251019.sql")
        
        # Filter out legacy migrations
        legacy_migrations = [
            '001_init.sql',
            '002_add_axioms.sql',
            '002_blocks_lemmas.sql',
            '003_add_system_id.sql',
            '003_fix_progress_compatibility.sql',
            '004_finalize_core_schema.sql',
            '005_add_search_indexes.sql',
            '006_add_pg_trgm_extension.sql',
            '006_add_policy_settings.sql',
            '007_fix_proofs_schema.sql',
            '008_fix_statements_hash.sql',
            '009_normalize_statements.sql',
            '010_idempotent_normalize.sql',
            '011_schema_parity.sql',
            '012_blocks_parity.sql',
            '013_runs_logging.sql',
            '014_ensure_slug_column.sql'
        ]
        
        migration_files = [f for f in migration_files if f.name not in legacy_migrations]
        print(f"Remaining migrations to run: {len(migration_files)}")
    else:
        print("\n=== NO BASELINE DETECTED ===")
        print("Running all migrations in sequence")

    if not migration_files:
        print("No migrations to run")
        return

    print("\nMigrations to execute:")
    for file in migration_files:
        print(f"  - {file.name}")
    print()

    failed_migrations = []
    successful_migrations = []
    
    for migration_file in migration_files:
        try:
            success = run_migration_file(migration_file)
            if success:
                successful_migrations.append(migration_file.name)
            else:
                failed_migrations.append(migration_file.name)
                # Stop on first failure
                break
        except Exception as e:
            print(f"Error running {migration_file}: {e}")
            failed_migrations.append(migration_file.name)
            break

    print(f"\n=== MIGRATION SUMMARY ===")
    print(f"Successful: {len(successful_migrations)}")
    print(f"Failed: {len(failed_migrations)}")
    
    if successful_migrations:
        print("\nSuccessful migrations:")
        for m in successful_migrations:
            print(f"  ✓ {m}")
    
    if failed_migrations:
        print("\nFailed migrations:")
        for m in failed_migrations:
            print(f"  ✗ {m}")
        exit(1)
    else:
        print("\n✓ All migrations completed successfully!")

if __name__ == "__main__":
    main()

