#!/usr/bin/env python3
"""
get_schema.py - Get actual database schema for safe deployment
"""

import psycopg
import sys

def main():
    from backend.security.runtime_env import MissingEnvironmentVariable, get_database_url

    try:
        db_url = get_database_url()
    except MissingEnvironmentVariable as exc:
        print(f"❌ {exc}")
        sys.exit(2)

    try:
        print("Connecting to database...")
        conn = psycopg.connect(db_url)
        cur = conn.cursor()

        print("Getting table list...")
        cur.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            ORDER BY table_name
        """)
        tables = cur.fetchall()
        print(f"Tables found: {[t[0] for t in tables]}")

        print("\n" + "="*50)

        for table in tables:
            table_name = table[0]
            print(f"\nSchema for {table_name}:")
            print("-" * 30)

            cur.execute(f"""
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns
                WHERE table_name = '{table_name}'
                ORDER BY ordinal_position
            """)
            columns = cur.fetchall()

            for col in columns:
                col_name, data_type, nullable, default = col
                nullable_str = "NULL" if nullable == "YES" else "NOT NULL"
                default_str = f" DEFAULT {default}" if default else ""
                print(f"  - {col_name}: {data_type} {nullable_str}{default_str}")

        # Check for specific tables we need
        print("\n" + "="*50)
        print("CRITICAL TABLES CHECK:")

        required_tables = ['blocks', 'statements', 'proofs']
        for table in required_tables:
            if any(t[0] == table for t in tables):
                print(f"✓ {table} table exists")
            else:
                print(f"✗ {table} table MISSING")

        conn.close()
        print("\nSchema check completed successfully!")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
