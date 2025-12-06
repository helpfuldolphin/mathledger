#!/usr/bin/env python3
"""
Robust test suite for MathLedger v0.5 UI implementation using FastAPI TestClient.
Tests search functionality, block explorer, and verification features without requiring a live server.
"""

import pytest
from fastapi.testclient import TestClient
from backend.orchestrator.app import app
import json
from unittest.mock import patch, MagicMock
from datetime import datetime

# Create test client
client = TestClient(app)

class TestV05Implementation:
    """Test suite for v0.5 features using FastAPI TestClient."""

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
        assert isinstance(data["total"], int)
        assert data["limit"] == 50  # default limit

    def test_search_endpoint_filters(self):
        """Test search endpoint with filters."""
        response = client.get("/search?q=test&system=pl&status=proven&depth_gt=0")
        assert response.status_code == 200

        data = response.json()
        assert data["query"]["system"] == "pl"
        assert data["query"]["status"] == "proven"
        assert data["query"]["depth_gt"] == 0

    def test_search_endpoint_pagination(self):
        """Test search endpoint pagination."""
        response = client.get("/search?limit=10&offset=20")
        assert response.status_code == 200

        data = response.json()
        assert data["limit"] == 10
        assert data["offset"] == 20

    def test_search_endpoint_empty_query(self):
        """Test search endpoint with empty query."""
        response = client.get("/search")
        assert response.status_code == 200

        data = response.json()
        assert data["query"]["q"] is None
        assert isinstance(data["results"], list)

    def test_search_endpoint_special_characters(self):
        """Test search endpoint with special characters."""
        response = client.get("/search?q=test%20%26%20%3C%3E%20%22%27")
        assert response.status_code == 200

        data = response.json()
        assert data["query"]["q"] == "test & <> \"'"

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

    def test_metrics_endpoint(self):
        """Test metrics endpoint."""
        response = client.get("/metrics")
        assert response.status_code == 200

        data = response.json()
        required_fields = ["statement_count", "proof_count", "success_rate", "latest_block"]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"

    def test_blocks_endpoint(self):
        """Test blocks endpoint."""
        response = client.get("/blocks/latest")
        assert response.status_code == 200

        data = response.json()
        required_fields = ["block_number", "merkle_root", "created_at", "header"]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"

    def test_statements_endpoint(self):
        """Test statements endpoint."""
        test_hash = "a" * 64
        response = client.get(f"/statements?hash={test_hash}")
        assert response.status_code == 200

        data = response.json()
        required_fields = ["statement", "proofs", "parents", "children"]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"

    def test_dag_nodes_endpoint(self):
        """Test DAG nodes endpoint."""
        response = client.get("/ui/dag/nodes/1")
        assert response.status_code == 200

        data = response.json()
        required_fields = ["nodes", "total", "level_min", "level_max", "has_more"]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"

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

    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test invalid block ID
        response = client.get("/ui/blocks/invalid")
        assert response.status_code == 422  # Validation error

        # Test invalid statement hash
        response = client.get("/ui/s/invalid")
        assert response.status_code == 422  # Validation error

        # Test invalid DAG node ID
        response = client.get("/ui/dag/nodes/invalid")
        assert response.status_code == 422  # Validation error

    def test_cors_headers(self):
        """Test CORS headers are present."""
        response = client.options("/search")
        assert "access-control-allow-origin" in response.headers

    def test_api_key_authentication(self):
        """Test API key authentication for protected endpoints."""
        # Test without API key
        response = client.get("/metrics")
        assert response.status_code == 200  # Should work without auth for now

        # Test with invalid API key
        response = client.get("/metrics", headers={"X-API-Key": "invalid"})
        assert response.status_code == 200  # Should work without auth for now

def test_database_connection():
    """Test database connection handling."""
    # This test would require mocking the database connection
    # For now, we'll just ensure the endpoint doesn't crash
    response = client.get("/search")
    assert response.status_code == 200

def test_performance_requirements():
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
