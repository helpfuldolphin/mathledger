#!/usr/bin/env python3
"""
Unit test suite for MathLedger API using FastAPI TestClient.
Tests dependency injection and basic functionality without database.
"""

import pytest
import os
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from backend.orchestrator.app import app

@pytest.fixture
def client():
    """Create TestClient with mocked database."""
    return TestClient(app)

class TestDependencyInjection:
    """Test that dependency injection is working correctly."""

    def test_app_imports_without_db_connection(self):
        """Test that app can be imported without database connection."""
        # This test passes if we can import the app without errors
        from backend.orchestrator.app import app
        assert app is not None

    def test_health_endpoint_no_db_required(self, client):
        """Test health endpoint doesn't require database."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert data["status"] == "healthy"

    @patch('backend.orchestrator.app.psycopg.connect')
    def test_metrics_endpoint_uses_dependency_injection(self, mock_connect, client):
        """Test that metrics endpoint uses dependency injection."""
        # Mock database connection and cursor
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur

        # Mock database responses
        mock_cur.fetchone.side_effect = [
            [5],  # block_height
            [10], # statement_counts
            [8],  # proofs_total
            [6]   # proofs_success
        ]

        response = client.get("/metrics")
        assert response.status_code == 200

        data = response.json()
        assert "proof_counts" in data
        assert "statement_counts" in data
        assert "success_rate" in data
        assert "queue_length" in data
        assert "block_height" in data

        # Verify that psycopg.connect was called (dependency injection working)
        mock_connect.assert_called_once()

    @patch('backend.orchestrator.app.psycopg.connect')
    def test_blocks_latest_endpoint_uses_dependency_injection(self, mock_connect, client):
        """Test that blocks/latest endpoint uses dependency injection."""
        # Mock database connection and cursor
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur

        # Mock database response
        mock_cur.fetchone.return_value = [1, "test_merkle_root", "2023-01-01T00:00:00", {"test": "data"}]

        response = client.get("/blocks/latest")
        assert response.status_code == 200

        data = response.json()
        assert "block_number" in data
        assert "merkle_root" in data
        assert "created_at" in data
        assert "header" in data

        # Verify that psycopg.connect was called (dependency injection working)
        mock_connect.assert_called_once()

    @patch('backend.orchestrator.app.psycopg.connect')
    def test_statements_endpoint_uses_dependency_injection(self, mock_connect, client):
        """Test that statements endpoint uses dependency injection."""
        # Mock database connection and cursor
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur

        # Mock database response
        mock_cur.fetchone.return_value = [1, "test statement", "test_statement", 1, False, 0, "2023-01-01T00:00:00"]

        response = client.get("/statements?hash=test_statement")
        assert response.status_code == 200

        data = response.json()
        assert "id" in data
        assert "text" in data
        assert "content_norm" in data
        assert "system_id" in data
        assert "is_axiom" in data
        assert "derivation_depth" in data
        assert "created_at" in data

        # Verify that psycopg.connect was called (dependency injection working)
        mock_connect.assert_called_once()

    @patch('backend.orchestrator.app.psycopg.connect')
    def test_database_connection_error_handling(self, mock_connect, client):
        """Test that database connection errors are handled properly."""
        # Mock database connection failure
        mock_connect.side_effect = Exception("Database connection failed")

        response = client.get("/metrics")
        assert response.status_code == 500

        data = response.json()
        assert "detail" in data
        assert "error" in data["detail"].lower()

    def test_dependency_injection_isolation(self, client):
        """Test that dependency injection provides isolated connections."""
        with patch('backend.orchestrator.app.psycopg.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_cur = MagicMock()
            mock_connect.return_value = mock_conn
            mock_conn.cursor.return_value.__enter__.return_value = mock_cur
            mock_cur.fetchone.return_value = [5]

            # Make multiple requests
            response1 = client.get("/metrics")
            response2 = client.get("/metrics")

            assert response1.status_code == 200
            assert response2.status_code == 200

            # Each request should create a new connection (dependency injection)
            assert mock_connect.call_count == 2


class TestAPIStructure:
    """Test basic API structure and responses."""

    def test_all_endpoints_exist(self, client):
        """Test that all expected endpoints exist."""
        endpoints = [
            ("/health", 200),
            ("/metrics", 500),  # Will fail without DB, but endpoint exists
            ("/blocks/latest", 500),  # Will fail without DB, but endpoint exists
        ]

        for endpoint, expected_status in endpoints:
            response = client.get(endpoint)
            # Endpoint should exist (not 404)
            assert response.status_code != 404, f"Endpoint {endpoint} not found"

    def test_statements_endpoint_requires_hash(self, client):
        """Test that statements endpoint requires hash parameter."""
        response = client.get("/statements")
        assert response.status_code == 422  # Validation error for missing parameter


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
