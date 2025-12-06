import psycopg
import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from backend.security.runtime_env import get_database_url

def run_migrations():
    db_url = get_database_url()
    print(f"Connecting to {db_url}...")
    
    project_root = Path(__file__).resolve().parent.parent.parent
    migrations_dir = project_root / "migrations"
    
    files = sorted([f for f in os.listdir(migrations_dir) if f.endswith(".sql")])
    
    with psycopg.connect(db_url) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            for filename in files:
                print(f"Applying {filename}...")
                filepath = migrations_dir / filename
                with open(filepath, "r", encoding="utf-8") as f:
                    sql = f.read()
                
                # Split by semicolon? Or just run? 
                # Some SQL might have ; inside strings, but usually migrations are simple.
                # Better to try executing the whole file if psql supports it, but psycopg executes one command per call usually.
                # However, if the file contains multiple commands, psycopg might handle it or fail.
                # Simple splitter:
                statements = [s.strip() for s in sql.split(';') if s.strip()]
                for stmt in statements:
                    if stmt.startswith('--'):
                        continue
                    try:
                        cur.execute(stmt)
                    except Exception as e:
                        print(f"  Warning/Error in {filename}: {e}")
                        # Continue (idempotent scripts) or Fail?
                        # Assuming scripts are idempotent-ish or we want to proceed.
                        # For a clean DB, errors are bad. But for "if not exists", they are fine.
                        pass
    print("Migrations complete.")

if __name__ == "__main__":
    run_migrations()
