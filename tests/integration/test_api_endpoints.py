"""
Integration tests for MathLedger API endpoints.
Tests actual database interactions with seeded data.
"""

import pytest
import json
from fastapi.testclient import TestClient


class TestMetricsEndpoint:
    """Test the /metrics endpoint with real data."""

    def test_metrics_basic(self, test_client: TestClient, api_headers: dict):
        """Test basic metrics endpoint functionality."""
        response = test_client.get("/metrics", headers=api_headers)

        assert response.status_code == 200
        data = response.json()

        # Check response structure
        assert "proofs" in data
        assert "block_count" in data
        assert "max_depth" in data
        assert "queue_length" in data

        # Check proofs structure
        assert "success" in data["proofs"]
        assert "failure" in data["proofs"]

        # With seeded data, we should have some proofs
        assert data["proofs"]["success"] >= 0
        assert data["proofs"]["failure"] >= 0
        assert data["block_count"] >= 0
        assert data["max_depth"] >= 0

    def test_metrics_with_seeded_data(self, test_client: TestClient, api_headers: dict):
        """Test metrics with known seeded data."""
        response = test_client.get("/metrics", headers=api_headers)
        data = response.json()

        # We seeded 4 successful proofs and 1 failed proof
        assert data["proofs"]["success"] == 4
        assert data["proofs"]["failure"] == 1
        assert data["block_count"] == 1
        assert data["max_depth"] == 2  # Max depth in seeded data

    def test_metrics_unauthorized(self, test_client: TestClient):
        """Test metrics endpoint without API key."""
        response = test_client.get("/metrics")
        assert response.status_code == 401


class TestBlocksLatestEndpoint:
    """Test the /blocks/latest endpoint."""

    def test_blocks_latest_basic(self, test_client: TestClient, api_headers: dict):
        """Test basic blocks/latest endpoint functionality."""
        response = test_client.get("/blocks/latest", headers=api_headers)

        assert response.status_code == 200
        data = response.json()

        # Check response structure
        assert "block_number" in data
        assert "merkle_root" in data
        assert "created_at" in data
        assert "header" in data

        # Check data types
        assert isinstance(data["block_number"], int)
        assert isinstance(data["merkle_root"], str)
        assert isinstance(data["created_at"], str)
        assert isinstance(data["header"], dict)

    def test_blocks_latest_with_seeded_data(self, test_client: TestClient, api_headers: dict):
        """Test blocks/latest with known seeded data."""
        response = test_client.get("/blocks/latest", headers=api_headers)
        data = response.json()

        # We seeded block number 1
        assert data["block_number"] == 1
        assert len(data["merkle_root"]) == 64  # SHA-256 hex string
        assert "run_name" in data["header"]
        assert "statements" in data["header"]
        assert "metadata" in data["header"]

    def test_blocks_latest_unauthorized(self, test_client: TestClient):
        """Test blocks/latest endpoint without API key."""
        response = test_client.get("/blocks/latest")
        assert response.status_code == 401


class TestStatementsEndpoint:
    """Test the /statements endpoint."""

    def test_statements_by_hash(self, test_client: TestClient, api_headers: dict):
        """Test statements endpoint with valid hash."""
        # Get a known statement hash from seeded data
        import hashlib
        test_content = "(and p q)"
        test_hash = hashlib.sha256(test_content.encode('utf-8')).hexdigest()

        response = test_client.get(f"/statements?hash={test_hash}", headers=api_headers)

        assert response.status_code == 200
        data = response.json()

        # Check response structure
        assert "id" in data
        assert "hash" in data
        assert "text" in data
        assert "proofs" in data
        assert "parents" in data

        # Check data
        assert data["hash"] == test_hash
        assert data["text"] == test_content
        assert isinstance(data["proofs"], list)
        assert isinstance(data["parents"], list)

    def test_statements_by_hash_with_proofs(self, test_client: TestClient, api_headers: dict):
        """Test statements endpoint returns proofs correctly."""
        import hashlib
        test_content = "(and p q)"
        test_hash = hashlib.sha256(test_content.encode('utf-8')).hexdigest()

        response = test_client.get(f"/statements?hash={test_hash}", headers=api_headers)
        data = response.json()

        # Should have one proof
        assert len(data["proofs"]) == 1
        proof = data["proofs"][0]

        assert "id" in proof
        assert "statement_id" in proof
        assert "system_id" in proof
        assert "prover" in proof
        assert "status" in proof
        assert "duration_ms" in proof
        assert "created_at" in proof

        assert proof["prover"] == "lean4"
        assert proof["status"] == "success"
        assert proof["duration_ms"] == 150

    def test_statements_by_hash_with_parents(self, test_client: TestClient, api_headers: dict):
        """Test statements endpoint returns parent statements correctly."""
        import hashlib
        # Test a statement that has dependencies
        test_content = "(or p q)"
        test_hash = hashlib.sha256(test_content.encode('utf-8')).hexdigest()

        response = test_client.get(f"/statements?hash={test_hash}", headers=api_headers)
        data = response.json()

        # Should have one parent (and p q)
        assert len(data["parents"]) == 1
        parent_hash = data["parents"][0]

        # Verify parent hash
        expected_parent_hash = hashlib.sha256("(and p q)".encode('utf-8')).hexdigest()
        assert parent_hash == expected_parent_hash

    def test_statements_invalid_hash(self, test_client: TestClient, api_headers: dict):
        """Test statements endpoint with invalid hash format."""
        response = test_client.get("/statements?hash=invalid_hash", headers=api_headers)
        assert response.status_code == 400

    def test_statements_nonexistent_hash(self, test_client: TestClient, api_headers: dict):
        """Test statements endpoint with non-existent hash."""
        import hashlib
        nonexistent_hash = hashlib.sha256("nonexistent".encode('utf-8')).hexdigest()

        response = test_client.get(f"/statements?hash={nonexistent_hash}", headers=api_headers)
        assert response.status_code == 404

    def test_statements_unauthorized(self, test_client: TestClient):
        """Test statements endpoint without API key."""
        response = test_client.get("/statements?hash=test")
        assert response.status_code == 401


class TestUIEndpoints:
    """Test UI endpoints (no authentication required)."""

    def test_ui_dashboard(self, test_client: TestClient):
        """Test UI dashboard endpoint."""
        response = test_client.get("/ui")

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "MathLedger Dashboard" in response.text
        assert "Statements" in response.text
        assert "Proofs" in response.text

    def test_ui_statement_detail(self, test_client: TestClient):
        """Test UI statement detail endpoint."""
        import hashlib
        test_content = "(and p q)"
        test_hash = hashlib.sha256(test_content.encode('utf-8')).hexdigest()

        response = test_client.get(f"/ui/s/{test_hash}")

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Statement Details" in response.text
        assert test_content in response.text

    def test_ui_statement_detail_invalid_hash(self, test_client: TestClient):
        """Test UI statement detail with invalid hash."""
        response = test_client.get("/ui/s/invalid_hash")
        assert response.status_code == 400


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_check(self, test_client: TestClient):
        """Test health check endpoint."""
        response = test_client.get("/health")

        assert response.status_code == 200
        data = response.json()

        assert "ok" in data
        assert "status" in data
        assert "timestamp" in data
        assert data["ok"] is True
        assert data["status"] == "healthy"


class TestCORS:
    """Test CORS functionality."""

    def test_cors_headers(self, test_client: TestClient):
        """Test that CORS headers are present."""
        response = test_client.options("/metrics", headers={"Origin": "http://localhost:3000"})

        # CORS preflight should be handled
        assert response.status_code in [200, 204]

        # Check for CORS headers
        headers = response.headers
        assert "access-control-allow-origin" in headers or "Access-Control-Allow-Origin" in headers
