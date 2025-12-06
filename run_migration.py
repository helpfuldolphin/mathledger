#!/usr/bin/env python3
"""Run database migration for system_id support."""

import psycopg
import os

from backend.security.runtime_env import MissingEnvironmentVariable, get_database_url

try:
    DATABASE_URL = get_database_url()
except MissingEnvironmentVariable as exc:
    raise RuntimeError(str(exc)) from exc

def run_migration():
    """Run the system_id migration."""
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            # Read and execute the migration file
            with open("migrations/003_add_system_id.sql", "r") as f:
                migration_sql = f.read()

            # Split by semicolon and execute each statement
            statements = [stmt.strip() for stmt in migration_sql.split(';') if stmt.strip()]

            for stmt in statements:
                if stmt:
                    print(f"Executing: {stmt[:50]}...")
                    cur.execute(stmt)

            conn.commit()
            print("Migration completed successfully!")

if __name__ == "__main__":
    run_migration()
