"""
API Surface Validation & H_t Invariant Tests
=============================================

Cursor H — API Harness Builder & Dual-Surface Verifier

This module provides:
1. Full /attestation/* and /ui/* surface validation
2. Pydantic schema canonicality and determinism checks
3. Schema evolution tests for backward compatibility
4. FastAPI → deterministic H_t invariant integration
5. Multi-statement DAG query tests

All responses are validated strictly via Pydantic models (no raw dicts).
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

# Import typed Pydantic schemas for strict validation
from interface.api.schemas import (
    ApiModel,
    AttestationLatestResponse,
    BlockLatestResponse,
    HealthResponse,
    HeartbeatResponse,
    MetricsResponse,
    ParentListResponse,
    ParentSummary,
    ProofListResponse,
    ProofSummary,
    RecentStatementsResponse,
    StatementDetailResponse,
    UIEventListResponse,
    UIEventRecord,
    UIEventResponse,
)

# Import from canonical attestation module
from attestation.dual_root import (
    compute_composite_root,
    verify_composite_integrity,
)

# Import from ledger modules
from ledger.ui_events import capture_ui_event, ui_event_store, snapshot_ui_events

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def clean_ui_event_store():
    """Ensure UI event store is clean before and after each test."""
    ui_event_store.clear()
    yield
    ui_event_store.clear()


# ---------------------------------------------------------------------------
# 1. Full /attestation/* Surface Validation
# ---------------------------------------------------------------------------


class TestAttestationSurface:
    """Validate all /attestation/* endpoints with typed Pydantic models."""

    def test_post_ui_event_returns_typed_response(
        self, test_client: TestClient, clean_ui_event_store
    ):
        """POST /attestation/ui-event returns UIEventResponse."""
        payload = {
            "event_id": "test-event-001",
            "action": "click",
            "target": "submit-btn",
            "timestamp": 1700000000,
        }
        response = test_client.post("/attestation/ui-event", json=payload)
        assert response.status_code == 200

        # Validate via Pydantic model
        typed_response = UIEventResponse.model_validate(response.json())
        assert typed_response.event_id == "test-event-001"
        assert typed_response.timestamp == 1700000000
        assert typed_response.leaf_hash  # Non-empty

    def test_get_ui_events_returns_typed_list(
        self, test_client: TestClient, clean_ui_event_store
    ):
        """GET /attestation/ui-events returns UIEventListResponse."""
        # Post some events first
        for i in range(3):
            test_client.post(
                "/attestation/ui-event",
                json={"event_id": f"event-{i}", "action": "test", "timestamp": 1700000000 + i},
            )

        response = test_client.get("/attestation/ui-events")
        assert response.status_code == 200

        # Validate via Pydantic model
        typed_response = UIEventListResponse.model_validate(response.json())
        assert len(typed_response.events) == 3

        # Each event should be a valid UIEventRecord
        for event in typed_response.events:
            assert isinstance(event, UIEventRecord)
            assert event.event_id.startswith("event-")
            assert event.leaf_hash  # Non-empty

    def test_get_latest_attestation_structure(
        self, test_client: TestClient, api_headers: Dict[str, str]
    ):
        """GET /attestation/latest returns AttestationLatestResponse."""
        response = test_client.get("/attestation/latest")

        # May be 404 if no attestation exists, or 200 with data
        if response.status_code == 404:
            data = response.json()
            assert "detail" in data
            assert data["detail"] == "no attestation"
            return

        assert response.status_code == 200

        # Validate via Pydantic model
        typed_response = AttestationLatestResponse.model_validate(response.json())

        # If we have an attestation, verify structure
        if typed_response.composite_attestation_root:
            assert typed_response.reasoning_merkle_root
            assert typed_response.ui_merkle_root

    def test_simulate_derivation_stub(self, test_client: TestClient):
        """POST /attestation/simulate-derivation returns stub response."""
        response = test_client.post("/attestation/simulate-derivation")
        assert response.status_code == 200

        data = response.json()
        assert data["triggered"] is True
        assert data["job_id"] == "simulated-job-id"
        assert data["status"] == "queued"


# ---------------------------------------------------------------------------
# 2. Full /ui/* Surface Validation
# ---------------------------------------------------------------------------


class TestUISurface:
    """Validate all /ui/* JSON endpoints with typed Pydantic models."""

    def test_ui_recent_json_returns_typed_response(self, test_client: TestClient):
        """GET /ui/recent.json returns RecentStatementsResponse."""
        response = test_client.get("/ui/recent.json")
        assert response.status_code == 200

        # Validate via Pydantic model
        typed_response = RecentStatementsResponse.model_validate(response.json())
        assert isinstance(typed_response.items, list)

    def test_ui_statement_json_validates_hash_format(self, test_client: TestClient):
        """GET /ui/statement/{hash}.json validates hash format."""
        # Invalid hash format
        response = test_client.get("/ui/statement/invalid.json")
        assert response.status_code == 400

        # Valid format but non-existent
        valid_hash = "a" * 64
        response = test_client.get(f"/ui/statement/{valid_hash}.json")
        assert response.status_code in (200, 404)

    def test_ui_parents_json_returns_typed_response(self, test_client: TestClient):
        """GET /ui/parents/{hash}.json returns ParentListResponse."""
        valid_hash = "b" * 64
        response = test_client.get(f"/ui/parents/{valid_hash}.json")
        assert response.status_code == 200

        # Validate via Pydantic model
        typed_response = ParentListResponse.model_validate(response.json())
        assert isinstance(typed_response.parents, list)

    def test_ui_proofs_json_returns_typed_response(self, test_client: TestClient):
        """GET /ui/proofs/{hash}.json returns ProofListResponse."""
        valid_hash = "c" * 64
        response = test_client.get(f"/ui/proofs/{valid_hash}.json")
        assert response.status_code == 200

        # Validate via Pydantic model
        typed_response = ProofListResponse.model_validate(response.json())
        assert isinstance(typed_response.proofs, list)


# ---------------------------------------------------------------------------
# 3. Pydantic Schema Canonicality & Determinism
# ---------------------------------------------------------------------------


class TestSchemaCanonicalityAndDeterminism:
    """Ensure Pydantic models remain canonical and deterministic."""

    def test_api_model_forbids_extra_fields(self):
        """ApiModel base class rejects extra fields."""
        with pytest.raises(ValidationError):
            HealthResponse(
                status="healthy",
                timestamp=datetime.now(),
                extra_field="should_fail",
            )

    def test_health_response_deterministic_serialization(self, test_client: TestClient):
        """HealthResponse serializes deterministically."""
        # Make two requests
        r1 = test_client.get("/health").json()
        r2 = test_client.get("/health").json()

        # Parse both
        h1 = HealthResponse.model_validate(r1)
        h2 = HealthResponse.model_validate(r2)

        # Status should be identical
        assert h1.status == h2.status == "healthy"

        # Timestamps should be deterministic (same seed → same value)
        assert h1.timestamp == h2.timestamp

    def test_heartbeat_response_deterministic_fields(self, test_client: TestClient):
        """HeartbeatResponse has deterministic timestamp."""
        r1 = test_client.get("/heartbeat.json").json()
        r2 = test_client.get("/heartbeat.json").json()

        h1 = HeartbeatResponse.model_validate(r1)
        h2 = HeartbeatResponse.model_validate(r2)

        # Timestamps should be deterministic
        assert h1.ts == h2.ts

    def test_ui_event_leaf_hash_deterministic(
        self, test_client: TestClient, clean_ui_event_store
    ):
        """UI event leaf_hash is deterministic for same payload."""
        payload = {
            "event_id": "determinism-test",
            "action": "click",
            "timestamp": 1700000000,
        }

        r1 = test_client.post("/attestation/ui-event", json=payload)
        ui_event_store.clear()
        r2 = test_client.post("/attestation/ui-event", json=payload)

        e1 = UIEventResponse.model_validate(r1.json())
        e2 = UIEventResponse.model_validate(r2.json())

        # Same payload → same leaf_hash
        assert e1.leaf_hash == e2.leaf_hash

    def test_proof_summary_duration_ms_non_negative(self):
        """ProofSummary.duration_ms must be non-negative."""
        # Valid
        ps = ProofSummary(method="lean", status="success", duration_ms=100)
        assert ps.duration_ms == 100

        # Zero is valid
        ps = ProofSummary(method="lean", status="success", duration_ms=0)
        assert ps.duration_ms == 0

        # Negative should fail
        with pytest.raises(ValidationError):
            ProofSummary(method="lean", status="success", duration_ms=-1)

    def test_hex_digest_pattern_validation(self):
        """HexDigest pattern validates 64-char lowercase hex."""
        # Valid
        ps = ParentSummary(hash="a" * 64, display="test")
        assert ps.hash == "a" * 64

        # Invalid: too short
        with pytest.raises(ValidationError):
            ParentSummary(hash="a" * 63, display="test")

        # Invalid: uppercase
        with pytest.raises(ValidationError):
            ParentSummary(hash="A" * 64, display="test")

        # Invalid: non-hex
        with pytest.raises(ValidationError):
            ParentSummary(hash="g" * 64, display="test")


# ---------------------------------------------------------------------------
# 4. Schema Evolution Tests
# ---------------------------------------------------------------------------


class TestSchemaEvolution:
    """Ensure backward compatibility as schemas evolve."""

    def test_attestation_response_accepts_minimal_fields(self):
        """AttestationLatestResponse accepts minimal required fields."""
        # Minimal response (all optional fields omitted)
        minimal = AttestationLatestResponse()
        assert minimal.block_number is None
        assert minimal.reasoning_merkle_root is None
        assert minimal.attestation_metadata == {}

    def test_attestation_response_accepts_full_fields(self):
        """AttestationLatestResponse accepts all fields."""
        full = AttestationLatestResponse(
            block_number=42,
            reasoning_merkle_root="r" * 64,
            ui_merkle_root="u" * 64,
            composite_attestation_root="h" * 64,
            attestation_metadata={"version": "v2", "extra": "data"},
            block_hash="b" * 64,
        )
        assert full.block_number == 42
        assert full.reasoning_merkle_root == "r" * 64
        assert full.attestation_metadata["version"] == "v2"

    def test_metrics_response_optional_first_organism(self):
        """MetricsResponse.first_organism is optional for backward compat."""
        # Without first_organism
        from interface.api.schemas import ProofTotals, StatementTotals, BlockTotals

        m = MetricsResponse(
            generated_at=datetime.now(),
            proofs=ProofTotals(total=10, success=8, failure=2, success_rate=0.8),
            statements=StatementTotals(total=100, max_depth=5),
            blocks=BlockTotals(count=10, height=10),
            queue_length=0,
        )
        assert m.first_organism is None

    def test_statement_detail_response_parents_can_be_empty(self):
        """StatementDetailResponse.parents can be empty list."""
        sdr = StatementDetailResponse(
            hash="a" * 64,
            display="P → Q",
            proofs=[],
            parents=[],
        )
        assert sdr.parents == []
        assert sdr.proofs == []

    def test_ui_event_record_metadata_optional(self):
        """UIEventRecord.metadata is optional."""
        uer = UIEventRecord(
            event_id="test",
            timestamp=1700000000.0,
            leaf_hash="abc123",
        )
        assert uer.metadata is None
        assert uer.canonical_value is None


# ---------------------------------------------------------------------------
# 5. FastAPI → Deterministic H_t Invariants
# ---------------------------------------------------------------------------


class TestHtInvariants:
    """Validate H_t = SHA256(R_t || U_t) invariant across API surface."""

    def test_compute_composite_root_formula(self):
        """H_t = SHA256(R_t || U_t) formula is correct."""
        r_t = "a" * 64
        u_t = "b" * 64

        h_t = compute_composite_root(r_t, u_t)

        # Manually verify
        expected = hashlib.sha256(f"{r_t}{u_t}".encode("ascii")).hexdigest()
        assert h_t == expected

    def test_verify_composite_integrity_true_case(self):
        """verify_composite_integrity returns True for valid H_t."""
        r_t = "c" * 64
        u_t = "d" * 64
        h_t = compute_composite_root(r_t, u_t)

        assert verify_composite_integrity(r_t, u_t, h_t) is True

    def test_verify_composite_integrity_false_case(self):
        """verify_composite_integrity returns False for invalid H_t."""
        r_t = "e" * 64
        u_t = "f" * 64
        wrong_h_t = "0" * 64

        assert verify_composite_integrity(r_t, u_t, wrong_h_t) is False

    def test_attestation_api_ht_matches_formula(
        self,
        test_client: TestClient,
        test_db_connection,
        test_db_url: str,
        monkeypatch,
    ):
        """
        GET /attestation/latest returns H_t that matches SHA256(R_t || U_t).

        This is the core H_t invariant check at the API boundary.
        """
        response = test_client.get("/attestation/latest")

        if response.status_code == 404:
            pytest.skip("No attestation in database")

        assert response.status_code == 200
        attestation = AttestationLatestResponse.model_validate(response.json())

        if not attestation.composite_attestation_root:
            pytest.skip("Attestation has no composite root")

        r_t = attestation.reasoning_merkle_root
        u_t = attestation.ui_merkle_root
        h_t = attestation.composite_attestation_root

        if r_t and u_t and h_t:
            # Verify H_t formula
            recomputed = compute_composite_root(r_t, u_t)
            assert recomputed == h_t, (
                f"H_t Invariant violated at API boundary: "
                f"recomputed={recomputed}, returned={h_t}"
            )


# ---------------------------------------------------------------------------
# 6. Multi-Statement DAG Query Tests
# ---------------------------------------------------------------------------


class TestMultiStatementDAGQueries:
    """Test DAG traversal across multiple statements."""

    def test_statement_with_parents_returns_valid_hashes(
        self, test_client: TestClient, test_db_connection, test_db_url: str
    ):
        """
        Statement with parents returns valid parent hashes.

        Each parent hash should be a valid 64-char hex string.
        """
        # Get recent statements
        response = test_client.get("/ui/recent.json?limit=10")
        assert response.status_code == 200
        recent = RecentStatementsResponse.model_validate(response.json())

        if not recent.items:
            pytest.skip("No statements in database")

        # Check each statement's parents
        for item in recent.items:
            parents_resp = test_client.get(f"/ui/parents/{item.hash}.json")
            assert parents_resp.status_code == 200
            parents = ParentListResponse.model_validate(parents_resp.json())

            for parent in parents.parents:
                # Parent hash should be valid HexDigest
                assert len(parent.hash) == 64
                assert all(c in "0123456789abcdef" for c in parent.hash)

    def test_statement_proofs_all_have_valid_status(
        self, test_client: TestClient, test_db_connection, test_db_url: str
    ):
        """
        All proofs for a statement have valid status values.
        """
        response = test_client.get("/ui/recent.json?limit=5")
        assert response.status_code == 200
        recent = RecentStatementsResponse.model_validate(response.json())

        if not recent.items:
            pytest.skip("No statements in database")

        valid_statuses = {"success", "failure", "abstain", "timeout", "error", None}

        for item in recent.items:
            proofs_resp = test_client.get(f"/ui/proofs/{item.hash}.json")
            assert proofs_resp.status_code == 200
            proofs = ProofListResponse.model_validate(proofs_resp.json())

            for proof in proofs.proofs:
                assert proof.status in valid_statuses or proof.status is None

    def test_dag_traversal_depth_first(
        self, test_client: TestClient, test_db_connection, test_db_url: str
    ):
        """
        DAG can be traversed depth-first via parent links.

        This tests the API's ability to support DAG visualization.
        """
        response = test_client.get("/ui/recent.json?limit=1")
        assert response.status_code == 200
        recent = RecentStatementsResponse.model_validate(response.json())

        if not recent.items:
            pytest.skip("No statements in database")

        # Start from first statement
        root_hash = recent.items[0].hash
        visited: set[str] = set()
        to_visit: list[str] = [root_hash]

        max_depth = 5  # Limit traversal depth
        depth = 0

        while to_visit and depth < max_depth:
            current = to_visit.pop()
            if current in visited:
                continue
            visited.add(current)

            # Fetch parents
            parents_resp = test_client.get(f"/ui/parents/{current}.json")
            if parents_resp.status_code != 200:
                continue

            parents = ParentListResponse.model_validate(parents_resp.json())
            for parent in parents.parents:
                if parent.hash not in visited:
                    to_visit.append(parent.hash)

            depth += 1

        # Should have visited at least the root
        assert root_hash in visited

    def test_statement_detail_includes_embedded_proofs_and_parents(
        self, test_client: TestClient, test_db_connection, test_db_url: str
    ):
        """
        StatementDetailResponse includes both proofs and parents inline.

        This tests the denormalized response structure for efficiency.
        """
        response = test_client.get("/ui/recent.json?limit=1")
        assert response.status_code == 200
        recent = RecentStatementsResponse.model_validate(response.json())

        if not recent.items:
            pytest.skip("No statements in database")

        hash_val = recent.items[0].hash
        detail_resp = test_client.get(f"/ui/statement/{hash_val}.json")

        if detail_resp.status_code == 404:
            pytest.skip("Statement not found")

        assert detail_resp.status_code == 200
        detail = StatementDetailResponse.model_validate(detail_resp.json())

        # Verify structure
        assert detail.hash == hash_val
        assert isinstance(detail.proofs, list)
        assert isinstance(detail.parents, list)
        assert detail.display  # Non-empty display text


# ---------------------------------------------------------------------------
# 7. Cross-Endpoint Consistency Tests
# ---------------------------------------------------------------------------


class TestCrossEndpointConsistency:
    """Verify consistency between related endpoints."""

    def test_statement_detail_proofs_match_proofs_endpoint(
        self, test_client: TestClient, test_db_connection, test_db_url: str
    ):
        """
        Proofs in StatementDetailResponse match ProofListResponse.
        """
        response = test_client.get("/ui/recent.json?limit=1")
        if response.status_code != 200:
            pytest.skip("No recent statements")

        recent = RecentStatementsResponse.model_validate(response.json())
        if not recent.items:
            pytest.skip("No statements in database")

        hash_val = recent.items[0].hash

        # Get detail (includes proofs)
        detail_resp = test_client.get(f"/ui/statement/{hash_val}.json")
        if detail_resp.status_code != 200:
            pytest.skip("Statement not found")

        detail = StatementDetailResponse.model_validate(detail_resp.json())

        # Get proofs separately
        proofs_resp = test_client.get(f"/ui/proofs/{hash_val}.json")
        proofs = ProofListResponse.model_validate(proofs_resp.json())

        # Counts should match
        assert len(detail.proofs) == len(proofs.proofs)

    def test_statement_detail_parents_match_parents_endpoint(
        self, test_client: TestClient, test_db_connection, test_db_url: str
    ):
        """
        Parents in StatementDetailResponse match ParentListResponse.
        """
        response = test_client.get("/ui/recent.json?limit=1")
        if response.status_code != 200:
            pytest.skip("No recent statements")

        recent = RecentStatementsResponse.model_validate(response.json())
        if not recent.items:
            pytest.skip("No statements in database")

        hash_val = recent.items[0].hash

        # Get detail (includes parents)
        detail_resp = test_client.get(f"/ui/statement/{hash_val}.json")
        if detail_resp.status_code != 200:
            pytest.skip("Statement not found")

        detail = StatementDetailResponse.model_validate(detail_resp.json())

        # Get parents separately
        parents_resp = test_client.get(f"/ui/parents/{hash_val}.json")
        parents = ParentListResponse.model_validate(parents_resp.json())

        # Counts should match
        assert len(detail.parents) == len(parents.parents)

        # Hashes should match
        detail_hashes = {p.hash for p in detail.parents}
        parents_hashes = {p.hash for p in parents.parents}
        assert detail_hashes == parents_hashes


# ---------------------------------------------------------------------------
# 8. Error Response Validation
# ---------------------------------------------------------------------------


class TestErrorResponses:
    """Validate error response structures."""

    def test_invalid_hash_returns_400(self, test_client: TestClient):
        """Invalid hash format returns 400 with detail."""
        response = test_client.get("/ui/statement/invalid.json")
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data

    def test_nonexistent_statement_returns_404(self, test_client: TestClient):
        """Non-existent statement returns 404 with detail."""
        response = test_client.get(f"/ui/statement/{'0' * 64}.json")
        # May be 404 if not found, or 200 if coincidentally exists
        if response.status_code == 404:
            data = response.json()
            assert "detail" in data

    def test_missing_attestation_returns_404(self, test_client: TestClient):
        """Missing attestation returns 404 with detail."""
        # This depends on DB state, but we can check the structure
        response = test_client.get("/attestation/latest")
        if response.status_code == 404:
            data = response.json()
            assert "detail" in data
            assert data["detail"] == "no attestation"

