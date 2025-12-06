#!/usr/bin/env python3
"""
Minimal test suite for MathLedger v0.5 UI implementation.
Tests core functionality without complex mocking.
"""

import pytest
from fastapi.testclient import TestClient
import os

# Set environment variables
os.environ["DATABASE_URL"] = "postgresql://test:test@localhost:5432/test"
os.environ["REDIS_URL"] = "redis://localhost:6379/0"

# Import the app
from backend.orchestrator.app import app

# Create test client
client = TestClient(app)

def test_search_endpoint_basic():
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

def test_search_endpoint_validation():
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

def test_search_endpoint_empty_query():
    """Test search endpoint with empty query."""
    response = client.get("/search")
    assert response.status_code == 200

    data = response.json()
    assert data["query"]["q"] is None
    assert isinstance(data["results"], list)

def test_dashboard_search_ui():
    """Test dashboard search UI elements."""
    response = client.get("/ui")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

    # Check for search form elements
    assert "Search Statements" in response.text
    assert "search-form" in response.text
    assert "search-results-container" in response.text

def test_block_explorer_ui():
    """Test block explorer UI."""
    response = client.get("/ui/blocks")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Block Explorer" in response.text

def test_block_detail_ui():
    """Test block detail UI with verification features."""
    response = client.get("/ui/blocks/1")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

    # Check for verification features
    assert "Verify Block Integrity" in response.text
    assert "verifyBlockIntegrity" in response.text

def test_statement_detail_copy_lean():
    """Test statement detail page with Copy for Lean feature."""
    test_hash = "a" * 64
    response = client.get(f"/ui/s/{test_hash}")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

    # Check for Copy for Lean features
    assert "Copy for Lean" in response.text
    assert "copyLeanProof" in response.text

def test_block_verification_endpoint():
    """Test server-side block verification endpoint."""
    response = client.get("/blocks/1/verify")
    assert response.status_code == 200

    data = response.json()
    required_fields = ["valid", "block_id", "block_number", "stored_root", "calculated_root", "statement_count"]
    for field in required_fields:
        assert field in data, f"Missing required field: {field}"

def test_worker_status_endpoint():
    """Test worker status endpoint."""
    response = client.get("/workers/status")
    assert response.status_code == 200

    data = response.json()
    required_fields = ["queue_length", "active_jobs", "total_workers", "last_updated"]
    for field in required_fields:
        assert field in data, f"Missing required field: {field}"

def test_error_handling():
    """Test error handling for invalid inputs."""
    # Test invalid block ID
    response = client.get("/ui/blocks/invalid")
    assert response.status_code == 422  # Validation error

    # Test invalid statement hash
    response = client.get("/ui/s/invalid")
    assert response.status_code == 422  # Validation error

def test_cors_headers():
    """Test CORS headers are present."""
    response = client.options("/search")
    assert "access-control-allow-origin" in response.headers

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
