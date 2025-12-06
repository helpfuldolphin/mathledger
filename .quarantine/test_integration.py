#!/usr/bin/env python3
"""
Integration test suite for MathLedger API using FastAPI TestClient.
Uses isolated test database with proper setup/teardown.
"""

import pytest
import psycopg
import os
import tempfile
from fastapi.testclient import TestClient
from backend.orchestrator.app import app

# Test database configuration
TEST_DB_NAME = "mathledger_test"
TEST_DB_URL = f"postgresql://ml:mlpass@localhost:5432/{TEST_DB_NAME}"

@pytest.fixture(scope="session")
def test_db():
    """Create and setup isolated test database."""
    # Create test database
    admin_conn = psycopg.connect("postgresql://ml:mlpass@localhost:5432/postgres")
    admin_conn.autocommit = True

    try:
        # Drop test database if it exists
        admin_conn.execute(f"DROP DATABASE IF EXISTS {TEST_DB_NAME}")
        # Create test database
        admin_conn.execute(f"CREATE DATABASE {TEST_DB_NAME}")
    finally:
        admin_conn.close()

    # Connect to test database and run migrations
    test_conn = psycopg.connect(TEST_DB_URL)
    test_conn.autocommit = True

    try:
        # Create tables (simplified schema for testing)
        test_conn.execute("""
            CREATE TABLE IF NOT EXISTS blocks (
                id SERIAL PRIMARY KEY,
                block_number INTEGER NOT NULL,
                merkle_root TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                header JSONB
            )
        """)

        test_conn.execute("""
            CREATE TABLE IF NOT EXISTS statements (
                id SERIAL PRIMARY KEY,
                text TEXT,
                content_norm TEXT,
                system_id INTEGER,
                is_axiom BOOLEAN DEFAULT FALSE,
                derivation_depth INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        test_conn.execute("""
            CREATE TABLE IF NOT EXISTS proofs (
                id SERIAL PRIMARY KEY,
                statement_id INTEGER,
                status TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Insert test data
        test_conn.execute("""
            INSERT INTO blocks (block_number, merkle_root, header) VALUES
            (1, 'test_merkle_root_1', '{"test": "data1"}'),
            (2, 'test_merkle_root_2', '{"test": "data2"}')
        """)

        test_conn.execute("""
            INSERT INTO statements (text, content_norm, system_id, is_axiom, derivation_depth) VALUES
            ('test statement 1', 'test_statement_1', 1, FALSE, 0),
            ('test statement 2', 'test_statement_2', 1, FALSE, 1)
        """)

        test_conn.execute("""
            INSERT INTO proofs (statement_id, status) VALUES
            (1, 'success'),
            (2, 'failure')
        """)

    finally:
        test_conn.close()

    yield TEST_DB_URL

    # Cleanup: drop test database
    admin_conn = psycopg.connect("postgresql://ml:mlpass@localhost:5432/postgres")
    admin_conn.autocommit = True
    try:
        admin_conn.execute(f"DROP DATABASE IF EXISTS {TEST_DB_NAME}")
    finally:
        admin_conn.close()


@pytest.fixture
def client(test_db):
    """Create TestClient with test database."""
    # Set environment variable for test database
    os.environ["DATABASE_URL"] = test_db
    return TestClient(app)


class TestAPIEndpoints:
    """Test all API endpoints with real database."""

    def test_health_endpoint(self, client):
        """Test health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert data["status"] == "healthy"

    def test_metrics_endpoint(self, client):
        """Test metrics endpoint."""
        response = client.get("/metrics")
        assert response.status_code == 200

        data = response.json()
        required_fields = ["proof_counts", "statement_counts", "success_rate", "queue_length", "block_height"]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"

        assert data["statement_counts"] == 2  # We inserted 2 statements
        assert data["proof_counts"] == 2  # We inserted 2 proofs
        assert data["success_rate"] == 0.5  # 1 success out of 2 proofs

    def test_blocks_latest_endpoint(self, client):
        """Test blocks/latest endpoint."""
        response = client.get("/blocks/latest")
        assert response.status_code == 200

        data = response.json()
        required_fields = ["block_number", "merkle_root", "created_at", "header"]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"

        assert data["block_number"] == 2  # Latest block should be block 2
        assert data["merkle_root"] == "test_merkle_root_2"

    def test_statements_endpoint(self, client):
        """Test statements endpoint."""
        response = client.get("/statements?hash=test_statement_1")
        assert response.status_code == 200

        data = response.json()
        required_fields = ["id", "text", "content_norm", "system_id", "is_axiom", "derivation_depth", "created_at"]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"

        assert data["content_norm"] == "test_statement_1"
        assert data["text"] == "test statement 1"
        assert data["derivation_depth"] == 0

    def test_statements_endpoint_not_found(self, client):
        """Test statements endpoint with non-existent hash."""
        response = client.get("/statements?hash=nonexistent")
        assert response.status_code == 404

        data = response.json()
        assert "detail" in data
        assert "not found" in data["detail"].lower()

    def test_database_connection_isolation(self, client):
        """Test that database connections are properly isolated."""
        # This test ensures that the dependency injection is working
        # and connections are properly managed
        response1 = client.get("/metrics")
        response2 = client.get("/metrics")

        assert response1.status_code == 200
        assert response2.status_code == 200

        # Both responses should be identical (same data)
        assert response1.json() == response2.json()

    def test_error_handling(self, client):
        """Test error handling for database issues."""
        # Temporarily break the database connection
        original_url = os.environ.get("DATABASE_URL")
        os.environ["DATABASE_URL"] = "postgresql://invalid:invalid@localhost:5432/invalid"

        try:
            response = client.get("/metrics")
            assert response.status_code == 500
            assert "error" in response.json()["detail"].lower()
        finally:
            # Restore original database URL
            if original_url:
                os.environ["DATABASE_URL"] = original_url


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
