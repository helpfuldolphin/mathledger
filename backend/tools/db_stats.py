#!/usr/bin/env python3
"""
Database statistics tool for MathLedger.

Shows counts by status, theory information, and other useful stats.
"""

import os
import psycopg
from typing import Dict, Any

from backend.security.runtime_env import get_required_env


def get_database_url() -> str:
    """Get database URL from environment."""
    return get_required_env("DATABASE_URL")


def run_query(conn, query: str, params: tuple = ()) -> list:
    """Run a query and return results."""
    with conn.cursor() as cur:
        cur.execute(query, params)
        return cur.fetchall()


def print_stats(conn):
    """Print various database statistics."""
    print("=== MathLedger Database Statistics ===\n")

    # Theory counts
    print("📚 Theories:")
    theories = run_query(conn, "SELECT name, version, logic, created_at FROM theories ORDER BY created_at")
    for name, version, logic, created_at in theories:
        print(f"  {name} (v{version}, {logic}) - {created_at.strftime('%Y-%m-%d %H:%M')}")

    # Statement counts by theory
    print(f"\n📝 Statements by Theory:")
    stmt_by_theory = run_query(conn, """
        SELECT t.name, COUNT(s.id) as count
        FROM theories t
        LEFT JOIN statements s ON t.id = s.theory_id
        GROUP BY t.id, t.name
        ORDER BY count DESC
    """)
    for theory_name, count in stmt_by_theory:
        print(f"  {theory_name}: {count}")

    # Proof counts by status
    print(f"\n🔍 Proofs by Status:")
    proof_by_status = run_query(conn, """
        SELECT status, COUNT(*) as count
        FROM proofs
        GROUP BY status
        ORDER BY count DESC
    """)
    for status, count in proof_by_status:
        print(f"  {status}: {count}")

    # Recent proofs
    print(f"\n⏰ Recent Proofs (last 10):")
    recent_proofs = run_query(conn, """
        SELECT p.status, p.created_at, s.content_norm
        FROM proofs p
        JOIN statements s ON p.statement_id = s.id
        ORDER BY p.created_at DESC
        LIMIT 10
    """)
    for status, created_at, content in recent_proofs:
        content_short = content[:50] + "..." if len(content) > 50 else content
        print(f"  [{status}] {created_at.strftime('%H:%M:%S')} - {content_short}")

    # Success rate
    print(f"\n📊 Success Rate:")
    success_stats = run_query(conn, """
        SELECT
            COUNT(*) as total,
            COUNT(CASE WHEN status = 'success' THEN 1 END) as successful,
            ROUND(100.0 * COUNT(CASE WHEN status = 'success' THEN 1 END) / COUNT(*), 2) as success_rate
        FROM proofs
    """)
    if success_stats:
        total, successful, rate = success_stats[0]
        print(f"  Total proofs: {total}")
        print(f"  Successful: {successful}")
        print(f"  Success rate: {rate}%")

    # Prover breakdown
    print(f"\n🤖 Proofs by Prover:")
    prover_stats = run_query(conn, """
        SELECT prover, COUNT(*) as count
        FROM proofs
        GROUP BY prover
        ORDER BY count DESC
    """)
    for prover, count in prover_stats:
        print(f"  {prover}: {count}")


def main():
    """Main function."""
    try:
        db_url = get_database_url()
        print(f"Connecting to database...")

        with psycopg.connect(db_url) as conn:
            print_stats(conn)

    except psycopg.OperationalError as e:
        print(f"Database connection failed: {e}")
        print("Make sure PostgreSQL is running and accessible.")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
