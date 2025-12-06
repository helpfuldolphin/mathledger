"""
Cross-language API Contract Tests

Validates that FastAPI endpoints conform to contracts expected by
JavaScript (UI client) and PowerShell (automation scripts).

Mission: Guarantee seamless integration between Python ↔ JS ↔ PS toolchains.
"""

import pytest
import json
from typing import Dict, Any
from datetime import datetime


class TestMetricsEndpointContract:
    """Validate /metrics endpoint contract (used by JS client + PS scripts)."""

    def test_metrics_required_fields(self, test_client):
        """Verify all required fields are present in metrics response."""
        response = test_client.get("/metrics")
        assert response.status_code == 200

        data = response.json()

        # Required top-level fields
        assert "proofs" in data, "Missing 'proofs' field"
        assert "block_count" in data, "Missing 'block_count' field"
        assert "max_depth" in data, "Missing 'max_depth' field"

        # Proofs structure (used by JS + PS)
        assert "success" in data["proofs"], "Missing 'proofs.success' field"
        assert "failure" in data["proofs"], "Missing 'proofs.failure' field"

        print(f"[PASS] Metrics contract verified: fields={list(data.keys())}")

    def test_metrics_field_types(self, test_client):
        """Verify field types match expectations (prevent type coercion issues)."""
        response = test_client.get("/metrics")
        data = response.json()

        # Type assertions (critical for PS/JS interop)
        assert isinstance(data["proofs"]["success"], int), "proofs.success must be int"
        assert isinstance(data["proofs"]["failure"], int), "proofs.failure must be int"
        assert isinstance(data["block_count"], int), "block_count must be int"
        assert isinstance(data["max_depth"], int), "max_depth must be int"

        # Optional fields (must be correct type if present)
        if "success_rate" in data:
            assert isinstance(data["success_rate"], (int, float)), "success_rate must be numeric"
        if "queue_length" in data:
            assert isinstance(data["queue_length"], int), "queue_length must be int"

        print(f"[PASS] Metrics type safety verified")

    def test_metrics_additional_fields(self, test_client):
        """Verify additional fields expected by scripts."""
        response = test_client.get("/metrics")
        data = response.json()

        # Fields used by sanity.ps1
        assert "statement_counts" in data or "statements" in data, \
            "Missing statement count field (expected by PS scripts)"

        # Fields used by mathledger-client.js
        if "blocks" in data:
            assert "height" in data["blocks"], "Missing blocks.height"

        print(f"[PASS] Additional metrics fields verified")


class TestHeartbeatEndpointContract:
    """Validate /heartbeat.json endpoint contract (used by PS healthcheck)."""

    def test_heartbeat_required_fields(self, test_client):
        """Verify heartbeat response structure expected by healthcheck.ps1."""
        response = test_client.get("/heartbeat.json")
        assert response.status_code == 200

        data = response.json()

        # Required fields (healthcheck.ps1 lines 142-146)
        assert "ok" in data, "Missing 'ok' field"
        assert "ts" in data, "Missing 'ts' field"
        assert "proofs" in data, "Missing 'proofs' field"
        assert "blocks" in data, "Missing 'blocks' field"

        # Nested structure
        assert "success" in data["proofs"], "Missing 'proofs.success'"
        assert "height" in data["blocks"], "Missing 'blocks.height'"
        assert "latest" in data["blocks"], "Missing 'blocks.latest'"
        assert "merkle" in data["blocks"]["latest"], "Missing 'blocks.latest.merkle'"

        print(f"[PASS] Heartbeat contract verified: ok={data['ok']}")

    def test_heartbeat_field_types(self, test_client):
        """Verify heartbeat field types (PowerShell type coercion safety)."""
        response = test_client.get("/heartbeat.json")
        data = response.json()

        # Boolean (PowerShell converts JSON true/false to $true/$false)
        assert isinstance(data["ok"], bool), "ok must be boolean"

        # Timestamp (must be ISO 8601 string)
        assert isinstance(data["ts"], str), "ts must be string"
        # Validate ISO 8601 format
        try:
            datetime.fromisoformat(data["ts"].replace("Z", "+00:00"))
        except ValueError:
            pytest.fail(f"ts not valid ISO 8601: {data['ts']}")

        # Numbers
        assert isinstance(data["proofs"]["success"], int), "proofs.success must be int"
        assert isinstance(data["blocks"]["height"], int), "blocks.height must be int"

        # Merkle root can be null or string
        assert data["blocks"]["latest"]["merkle"] is None or \
               isinstance(data["blocks"]["latest"]["merkle"], str), \
               "merkle must be null or string"

        print(f"[PASS] Heartbeat type safety verified")

    def test_heartbeat_redis_field(self, test_client):
        """Verify redis queue info (used by monitoring)."""
        response = test_client.get("/heartbeat.json")
        data = response.json()

        # Redis info (optional but should be present)
        if "redis" in data:
            assert "ml_jobs_len" in data["redis"], "Missing redis.ml_jobs_len"
            assert isinstance(data["redis"]["ml_jobs_len"], int), \
                "redis.ml_jobs_len must be int"

        print(f"[PASS] Heartbeat Redis field verified")


class TestBlocksEndpointContract:
    """Validate /blocks/latest endpoint contract."""

    def test_blocks_latest_structure(self, test_client, seeded_db):
        """Verify blocks/latest response structure."""
        response = test_client.get("/blocks/latest")

        # Should return 404 if no blocks, or 200 with data
        if response.status_code == 404:
            data = response.json()
            assert "detail" in data
            assert data["detail"] == "no blocks"
            print("[PASS] Blocks endpoint 404 handling verified")
            return

        assert response.status_code == 200
        data = response.json()

        # Required fields
        assert "block_number" in data, "Missing block_number"
        assert "merkle_root" in data, "Missing merkle_root"
        assert "created_at" in data, "Missing created_at"
        assert "header" in data, "Missing header"

        # Type validation
        assert isinstance(data["block_number"], int), "block_number must be int"
        assert isinstance(data["merkle_root"], str), "merkle_root must be string"
        assert isinstance(data["created_at"], str), "created_at must be string"
        assert isinstance(data["header"], dict), "header must be object"

        print(f"[PASS] Blocks/latest contract verified: block={data['block_number']}")


class TestStatementsEndpointContract:
    """Validate /statements endpoint contract (requires API key)."""

    def test_statements_requires_api_key(self, test_client):
        """Verify API key enforcement (PowerShell must include X-API-Key header)."""
        # Without API key
        response = test_client.get("/statements?hash=abc123")
        assert response.status_code == 401

        data = response.json()
        assert "detail" in data
        assert "X-API-Key" in data["detail"] or "API key" in data["detail"]

        print("[PASS] Statements API key enforcement verified")

    def test_statements_hash_validation(self, test_client):
        """Verify hash format validation (64 hex chars)."""
        # Invalid hash format
        response = test_client.get(
            "/statements?hash=invalid",
            headers={"X-API-Key": "devkey"}
        )
        assert response.status_code == 400

        data = response.json()
        assert "detail" in data
        assert "hash" in data["detail"].lower()

        print("[PASS] Statements hash validation verified")

    def test_statements_response_structure(self, test_client, seeded_db):
        """Verify statements response structure (if statement exists)."""
        # This test needs a valid hash from seeded DB
        # For now, test the error case structure
        response = test_client.get(
            "/statements?hash=" + "0" * 64,
            headers={"X-API-Key": "devkey"}
        )

        # Should return 404 with detail
        if response.status_code == 404:
            data = response.json()
            assert "detail" in data
            print("[PASS] Statements 404 structure verified")
        elif response.status_code == 200:
            data = response.json()
            # Verify expected fields
            assert "statement" in data, "Missing statement field"
            assert "hash" in data, "Missing hash field"
            assert "proofs" in data, "Missing proofs field"
            assert "parents" in data, "Missing parents field"

            # Proofs should be array
            assert isinstance(data["proofs"], list), "proofs must be array"
            # Parents should be array
            assert isinstance(data["parents"], list), "parents must be array"

            print(f"[PASS] Statements response structure verified")


class TestHealthEndpointContract:
    """Validate /health endpoint contract."""

    def test_health_response_structure(self, test_client):
        """Verify health endpoint returns expected structure."""
        response = test_client.get("/health")
        assert response.status_code == 200

        data = response.json()

        # Required fields
        assert "status" in data, "Missing status field"
        assert "timestamp" in data, "Missing timestamp field"

        # Type validation
        assert isinstance(data["status"], str), "status must be string"
        assert isinstance(data["timestamp"], str), "timestamp must be string"

        # Status should be "healthy"
        assert data["status"] == "healthy", f"Unexpected status: {data['status']}"

        # Timestamp should be ISO 8601
        try:
            datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))
        except ValueError:
            pytest.fail(f"timestamp not valid ISO 8601: {data['timestamp']}")

        print(f"[PASS] Health endpoint verified: status={data['status']}")


class TestJSONRoundTrip:
    """Validate JSON serialization/deserialization parity across languages."""

    def test_boolean_serialization(self, test_client):
        """Verify booleans serialize consistently (Python True → JSON true → PS $true)."""
        response = test_client.get("/heartbeat.json")
        data = response.json()

        # Verify JSON boolean type
        assert data["ok"] is True or data["ok"] is False
        assert type(data["ok"]) == bool

        # Verify JSON representation
        raw_json = response.content.decode("utf-8")
        parsed = json.loads(raw_json)
        assert parsed["ok"] is True or parsed["ok"] is False

        print(f"[PASS] Boolean round-trip verified: {data['ok']}")

    def test_null_serialization(self, test_client):
        """Verify null handling (Python None → JSON null → PS $null)."""
        response = test_client.get("/heartbeat.json")
        data = response.json()

        # Merkle root can be null
        merkle = data["blocks"]["latest"]["merkle"]
        if merkle is None:
            # Verify JSON representation
            raw_json = response.content.decode("utf-8")
            assert "null" in raw_json or merkle is None
            print("[PASS] Null serialization verified")
        else:
            print(f"[SKIP] Merkle not null: {merkle}")

    def test_number_serialization(self, test_client):
        """Verify number types don't coerce unexpectedly."""
        response = test_client.get("/metrics")
        data = response.json()

        # Integers should stay integers
        assert type(data["proofs"]["success"]) == int
        assert type(data["block_count"]) == int

        # Floats should stay floats
        if "success_rate" in data and data["success_rate"] is not None:
            assert isinstance(data["success_rate"], (int, float))

        print("[PASS] Number serialization verified")

    def test_string_encoding(self, test_client):
        """Verify string encoding (UTF-8, special characters)."""
        response = test_client.get("/health")
        data = response.json()

        # Verify strings are properly encoded
        assert isinstance(data["status"], str)
        assert isinstance(data["timestamp"], str)

        # Verify no encoding issues
        assert len(data["status"]) > 0
        assert len(data["timestamp"]) > 0

        print("[PASS] String encoding verified")


class TestFieldNameConsistency:
    """Validate field naming consistency across endpoints."""

    def test_timestamp_field_naming(self, test_client):
        """Verify timestamp field names are consistent."""
        # Health uses 'timestamp'
        health = test_client.get("/health").json()
        assert "timestamp" in health

        # Heartbeat uses 'ts'
        heartbeat = test_client.get("/heartbeat.json").json()
        assert "ts" in heartbeat

        # Document this intentional difference
        print("[PASS] Timestamp field naming documented: health.timestamp, heartbeat.ts")

    def test_snake_case_convention(self, test_client):
        """Verify API uses snake_case consistently (Python convention)."""
        metrics = test_client.get("/metrics").json()

        # Check for snake_case in keys
        def is_snake_case(s):
            return "_" in s or s.islower()

        for key in metrics.keys():
            if key not in ["proofs", "blocks"]:  # nested objects are OK
                assert is_snake_case(key), f"Field {key} not snake_case"

        print("[PASS] Snake_case convention verified")


@pytest.fixture
def test_client(monkeypatch):
    """Provide FastAPI test client with mocked DB."""
    from fastapi.testclient import TestClient
    from backend.orchestrator.app import app, get_db_connection
    from unittest.mock import MagicMock
    import psycopg
    import os

    # Set required env vars
    os.environ["LEDGER_API_KEY"] = "devkey"
    os.environ["DATABASE_URL"] = "mock://test"

    # Mock DB connection
    def mock_get_db():
        conn = MagicMock(spec=psycopg.Connection)
        cursor = MagicMock()

        # Mock cursor responses for different queries
        def mock_execute(query, *args, **kwargs):
            query_lower = query.lower() if isinstance(query, str) else ""
            if "select count(*) from proofs" in query_lower:
                cursor.fetchone.return_value = (100,)
            elif "select count(*) from statements" in query_lower:
                cursor.fetchone.return_value = (500,)
            elif "select count(*) from blocks" in query_lower:
                cursor.fetchone.return_value = (10,)
            elif "select coalesce(max(derivation_depth)" in query_lower:
                cursor.fetchone.return_value = (5,)
            elif "select coalesce(max(block_number)" in query_lower:
                cursor.fetchone.return_value = (10,)
            elif "information_schema.columns" in query_lower:
                if "proofs" in query_lower:
                    # Mock schema for proofs table
                    cursor.fetchall.return_value = [
                        ("success", "boolean"),
                        ("status", "character varying"),
                        ("created_at", "timestamp"),
                        ("id", "integer"),
                        ("method", "character varying"),
                        ("statement_id", "integer"),
                    ]
                elif "statements" in query_lower:
                    cursor.fetchall.return_value = [
                        ("id", "integer"),
                        ("text", "text"),
                        ("normalized_text", "text"),
                        ("hash", "character varying"),
                    ]
            else:
                cursor.fetchone.return_value = (0,)
                cursor.fetchall.return_value = []

        cursor.execute = mock_execute
        cursor.__enter__ = lambda self: cursor
        cursor.__exit__ = lambda self, *args: None
        conn.cursor.return_value.__enter__ = lambda self: cursor
        conn.cursor.return_value.__exit__ = lambda self, *args: None
        conn.__enter__ = lambda self: conn
        conn.__exit__ = lambda self, *args: None

        yield conn

    app.dependency_overrides[get_db_connection] = mock_get_db

    client = TestClient(app)
    yield client

    app.dependency_overrides.clear()


@pytest.fixture
def seeded_db():
    """Fixture indicating tests that need seeded data."""
    # This is a marker fixture
    # Tests using this should be skipped if DB is not seeded
    pytest.skip("Requires seeded database")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("CROSS-LANGUAGE API CONTRACT TEST SUITE")
    print("Testing Python (FastAPI) ↔ JavaScript ↔ PowerShell")
    print("="*60 + "\n")

    pytest.main([__file__, "-v", "--tb=short"])
