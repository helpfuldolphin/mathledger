#!/usr/bin/env python3
"""
Final test suite for MathLedger v0.5 UI implementation.
Tests all critical functionality with proper mocking.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import os

# Mock environment variables
os.environ["DATABASE_URL"] = "postgresql://test:test@localhost:5432/test"
os.environ["REDIS_URL"] = "redis://localhost:6379/0"

# Mock all external dependencies
with patch('psycopg.connect') as mock_connect, \
     patch('redis.Redis') as mock_redis, \
     patch('backend.ledger.blockchain.merkle_root') as mock_merkle:

    # Setup database mocks
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_connect.return_value.__enter__.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur
    mock_cur.fetchall.return_value = []
    mock_cur.fetchone.return_value = [0]

    # Setup Redis mocks
    mock_redis_instance = MagicMock()
    mock_redis.return_value = mock_redis_instance
    mock_redis_instance.llen.return_value = 0
    mock_redis_instance.lrange.return_value = []

    # Setup Merkle root mock
    mock_merkle.return_value = "mock_merkle_root"

    from backend.orchestrator.app import app

# Create test client
client = TestClient(app)

class TestV05Final:
    """Final test suite for v0.5 functionality."""

    def test_search_endpoint_basic(self):
        """Test basic search endpoint functionality."""
        response = client.get("/search?q=test")
        assert response.status_code == 200

        data = response.json()
        required_fields = ["results", "total", "limit", "offset", "has_more", "query"]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"

        assert data["query"]["q"] == "test"
        assert isinstance(data["results"], list)
        assert data["limit"] == 50

    def test_search_endpoint_validation(self):
        """Test search endpoint input validation."""
        # Test invalid limit
        response = client.get("/search?limit=0")
        assert response.status_code == 400

        response = client.get("/search?limit=1001")
        assert response.status_code == 400

        # Test invalid offset
        response = client.get("/search?offset=-1")
        assert response.status_code == 400

        # Test invalid status
        response = client.get("/search?status=invalid")
        assert response.status_code == 400

        # Test invalid depth range
        response = client.get("/search?depth_gt=10&depth_lt=5")
        assert response.status_code == 400

    def test_search_endpoint_special_characters(self):
        """Test search endpoint with special characters."""
        response = client.get("/search?q=test%20%26%20%3C%3E%20%22%27")
        assert response.status_code == 200

        data = response.json()
        assert data["query"]["q"] == "test & <> \"'"

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
        assert "showLeanProofFallback" in response.text
        assert "MathLedger Statement:" in response.text

    def test_block_verification_endpoint(self):
        """Test server-side block verification endpoint."""
        response = client.get("/blocks/1/verify")
        assert response.status_code == 200

        data = response.json()
        required_fields = ["valid", "block_id", "block_number", "stored_root", "calculated_root", "statement_count"]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"

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

    def test_dashboard_partials(self):
        """Test dashboard partial endpoints."""
        # Test metrics partial
        response = client.get("/ui/dashboard/metrics")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

        # Test recent proofs partial
        response = client.get("/ui/dashboard/recent-proofs")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

        # Test worker status partial
        response = client.get("/ui/dashboard/worker-status")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_performance_requirements(self):
        """Test that search queries are performant."""
        import time

        start_time = time.time()
        response = client.get("/search?q=test")
        end_time = time.time()

        assert response.status_code == 200
        # Search should complete within 1 second
        assert (end_time - start_time) < 1.0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
