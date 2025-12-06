#!/usr/bin/env python3
"""Run database migration for system_id support - simple approach."""

import psycopg
import os

from backend.security.runtime_env import MissingEnvironmentVariable, get_database_url

try:
    DATABASE_URL = get_database_url()
except MissingEnvironmentVariable as exc:
    raise RuntimeError(str(exc)) from exc

def main():
    """Run the system_id migration step by step."""
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            # Step 1: Add slug column to theories
            print("Adding slug column to theories...")
            cur.execute("ALTER TABLE theories ADD COLUMN IF NOT EXISTS slug text")
            conn.commit()

            # Step 2: Create unique index on slug
            print("Creating slug index...")
            cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS theories_slug_idx ON theories(slug)")
            conn.commit()

            # Step 3: Update existing Propositional theory
            print("Updating Propositional theory...")
            cur.execute("UPDATE theories SET slug = 'pl' WHERE name = 'Propositional' AND slug IS NULL")
            conn.commit()

            # Step 4: Add system_id to runs (create table first if needed)
            print("Creating runs table...")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    id          bigserial primary key,
                    name        text,
                    status      text not null check (status in ('running', 'completed', 'failed')) default 'running',
                    started_at  timestamptz not null default now(),
                    completed_at timestamptz,
                    created_at  timestamptz not null default now()
                )
            """)
            conn.commit()

            # Step 5: Add system_id to runs
            print("Adding system_id to runs...")
            cur.execute("ALTER TABLE runs ADD COLUMN IF NOT EXISTS system_id uuid REFERENCES theories(id)")
            conn.commit()

            # Step 6: Add system_id to statements
            print("Adding system_id to statements...")
            cur.execute("ALTER TABLE statements ADD COLUMN IF NOT EXISTS system_id uuid REFERENCES theories(id)")
            conn.commit()

            # Step 7: Add system_id to proofs
            print("Adding system_id to proofs...")
            cur.execute("ALTER TABLE proofs ADD COLUMN IF NOT EXISTS system_id uuid REFERENCES theories(id)")
            conn.commit()

            # Step 8: Create blocks table
            print("Creating blocks table...")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS blocks (
                    id          bigserial primary key,
                    run_id      bigint not null references runs(id) on delete cascade,
                    root_hash   text not null,
                    counts      jsonb not null,
                    created_at  timestamptz not null default now()
                )
            """)
            conn.commit()

            # Step 9: Add system_id to blocks
            print("Adding system_id to blocks...")
            cur.execute("ALTER TABLE blocks ADD COLUMN IF NOT EXISTS system_id uuid REFERENCES theories(id)")
            conn.commit()

            # Step 10: Backfill system_id for existing data
            print("Backfilling system_id for existing data...")
            cur.execute("UPDATE statements SET system_id = theory_id WHERE system_id IS NULL")
            conn.commit()

            # Step 11: Create indexes
            print("Creating indexes...")
            cur.execute("CREATE INDEX IF NOT EXISTS runs_system_id_idx ON runs(system_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS statements_system_id_idx ON statements(system_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS proofs_system_id_idx ON proofs(system_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS blocks_system_id_idx ON blocks(system_id)")
            conn.commit()

            print("Migration completed successfully!")

if __name__ == "__main__":
    main()
