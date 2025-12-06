#!/usr/bin/env python3
"""
Simple test suite for MathLedger v0.5 UI implementation.
Tests core functionality without database dependencies.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import json

# Mock the database connection before importing the app
with patch('psycopg.connect') as mock_connect:
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_connect.return_value.__enter__.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur

    # Mock database responses
    mock_cur.fetchall.return_value = []
    mock_cur.fetchone.return_value = [0]

    from backend.orchestrator.app import app

# Create test client
client = TestClient(app)

class TestV05Core:
    """Test core v0.5 functionality with mocked database."""

    def test_search_endpoint_structure(self):
        """Test search endpoint returns correct structure."""
        response = client.get("/search?q=test")
        assert response.status_code == 200

        data = response.json()
        required_fields = ["results", "total", "limit", "offset", "has_more", "query"]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"

        assert data["query"]["q"] == "test"
        assert isinstance(data["results"], list)
        assert data["limit"] == 50

    def test_search_endpoint_filters(self):
        """Test search endpoint with filters."""
        response = client.get("/search?q=test&system=pl&status=proven&depth_gt=0")
        assert response.status_code == 200

        data = response.json()
        assert data["query"]["system"] == "pl"
        assert data["query"]["status"] == "proven"
        assert data["query"]["depth_gt"] == 0

    def test_search_endpoint_empty_query(self):
        """Test search endpoint with empty query."""
        response = client.get("/search")
        assert response.status_code == 200

        data = response.json()
        assert data["query"]["q"] is None
        assert isinstance(data["results"], list)

    def test_search_partial_endpoint(self):
        """Test search partial endpoint for HTMX."""
        response = client.get("/ui/dashboard/search?q=test")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Search Results" in response.text

    def test_dashboard_search_ui(self):
        """Test dashboard search UI elements."""
        response = client.get("/ui")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

        # Check for search form elements
        assert "Search Statements" in response.text
        assert "search-form" in response.text
        assert "search-results-container" in response.text
        assert "loadSearchResults" in response.text

    def test_block_explorer_ui(self):
        """Test block explorer UI."""
        response = client.get("/ui/blocks")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Block Explorer" in response.text

    def test_block_detail_ui(self):
        """Test block detail UI with verification features."""
        response = client.get("/ui/blocks/1")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

        # Check for verification features
        assert "Verify Block Integrity" in response.text
        assert "verifyBlockIntegrity" in response.text
        assert "merkle-root" in response.text
        assert "verification-result" in response.text

    def test_statement_detail_copy_lean(self):
        """Test statement detail page with Copy for Lean feature."""
        test_hash = "a" * 64
        response = client.get(f"/ui/s/{test_hash}")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

        # Check for Copy for Lean features
        assert "Copy for Lean" in response.text
        assert "copyLeanProof" in response.text
        assert "MathLedger Statement:" in response.text

    def test_worker_status_endpoint(self):
        """Test worker status endpoint."""
        response = client.get("/workers/status")
        assert response.status_code == 200

        data = response.json()
        required_fields = ["queue_length", "active_jobs", "total_workers", "last_updated"]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"

    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test invalid block ID
        response = client.get("/ui/blocks/invalid")
        assert response.status_code == 422  # Validation error

        # Test invalid statement hash
        response = client.get("/ui/s/invalid")
        assert response.status_code == 422  # Validation error

    def test_cors_headers(self):
        """Test CORS headers are present."""
        response = client.options("/search")
        assert "access-control-allow-origin" in response.headers

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
