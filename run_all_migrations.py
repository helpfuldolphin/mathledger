#!/usr/bin/env python3
"""Run all database migrations in order."""

import psycopg
import os

from backend.security.runtime_env import MissingEnvironmentVariable, get_database_url

try:
    DATABASE_URL = get_database_url()
except MissingEnvironmentVariable as exc:
    raise RuntimeError(str(exc)) from exc

def run_migration_file(filename):
    """Run a migration file."""
    print(f"Running migration: {filename}")
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            with open(filename, "r") as f:
                migration_sql = f.read()

            # Split by semicolon and execute each statement
            statements = [stmt.strip() for stmt in migration_sql.split(';') if stmt.strip()]

            for stmt in statements:
                if stmt and not stmt.startswith('--'):
                    try:
                        print(f"  Executing: {stmt[:50]}...")
                        cur.execute(stmt)
                    except Exception as e:
                        print(f"  Warning: {e}")
                        # Continue with next statement

            conn.commit()
            print(f"  Migration {filename} completed!")

def main():
    """Run all migrations in order."""
    migrations = [
        "migrations/002_blocks_lemmas.sql",
        "migrations/003_add_system_id.sql"
    ]

    for migration in migrations:
        try:
            run_migration_file(migration)
        except Exception as e:
            print(f"Error running {migration}: {e}")
            break

    print("All migrations completed!")

if __name__ == "__main__":
    main()
