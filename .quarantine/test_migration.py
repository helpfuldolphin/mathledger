#!/usr/bin/env python3
"""
Test script to verify the migration works and new tables are created.
"""

import os
import psycopg

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://ml:mlpass@localhost:5432/mathledger")

def test_migration():
    """Test that the migration creates the new tables."""
    try:
        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                # Check if runs table exists
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_name = 'runs'
                    );
                """)
                runs_exists = cur.fetchone()[0]
                assert runs_exists, "runs table not found"
                print("✓ runs table exists")

                # Check if blocks table exists
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_name = 'blocks'
                    );
                """)
                blocks_exists = cur.fetchone()[0]
                assert blocks_exists, "blocks table not found"
                print("✓ blocks table exists")

                # Check if lemma_cache table exists
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_name = 'lemma_cache'
                    );
                """)
                lemma_cache_exists = cur.fetchone()[0]
                assert lemma_cache_exists, "lemma_cache table not found"
                print("✓ lemma_cache table exists")

                # Check table structures
                cur.execute("""
                    SELECT column_name, data_type
                    FROM information_schema.columns
                    WHERE table_name = 'blocks'
                    ORDER BY ordinal_position
                """)
                blocks_columns = cur.fetchall()
                expected_blocks_columns = ['id', 'run_id', 'root_hash', 'counts', 'created_at']
                actual_columns = [col[0] for col in blocks_columns]
                for col in expected_blocks_columns:
                    assert col in actual_columns, f"blocks table missing column: {col}"
                print("✓ blocks table has correct structure")

                cur.execute("""
                    SELECT column_name, data_type
                    FROM information_schema.columns
                    WHERE table_name = 'lemma_cache'
                    ORDER BY ordinal_position
                """)
                lemma_cache_columns = cur.fetchall()
                expected_lemma_cache_columns = ['id', 'statement_id', 'usage_count', 'created_at']
                actual_columns = [col[0] for col in lemma_cache_columns]
                for col in expected_lemma_cache_columns:
                    assert col in actual_columns, f"lemma_cache table missing column: {col}"
                print("✓ lemma_cache table has correct structure")

                # Check if default run was created
                cur.execute("SELECT COUNT(*) FROM runs")
                run_count = cur.fetchone()[0]
                assert run_count > 0, "No default run created"
                print("✓ default run created")

                print()
                print("🎉 Migration test passed! All tables created correctly.")

    except psycopg.Error as e:
        print(f"❌ Database error: {e}")
        raise
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise

if __name__ == "__main__":
    print("Testing MathLedger migration...")
    print(f"Database URL: {DATABASE_URL}")
    print()

    test_migration()
