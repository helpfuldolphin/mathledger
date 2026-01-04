"""
Tests for v0.2.10 stateless commit (demo reliability).

These tests verify that:
1. commit_uvil works using draft_payload without server-side cache
2. Legacy commit path still requires proposal_id in cache
3. Boundary demo sequence never produces 404/422 from proposal lookup

The stateless commit path eliminates demo reliability issues caused by:
- Server restarts losing in-memory proposal cache
- Multi-instance hosting (Fly.io) routing requests to different machines
"""

import pytest
from fastapi.testclient import TestClient

from demo.app import app
from backend.api import uvil as uvil_module


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def reset_uvil_state():
    """Reset UVIL module state before each test."""
    uvil_module._draft_proposals.clear()
    uvil_module._committed_snapshots.clear()
    uvil_module._uvil_events.clear()
    uvil_module._reasoning_artifacts.clear()
    uvil_module._committed_proposal_ids.clear()
    uvil_module._partition_reasoning_artifacts.clear()
    uvil_module._epoch_counter = 0
    yield
    # Cleanup after test
    uvil_module._draft_proposals.clear()
    uvil_module._committed_snapshots.clear()
    uvil_module._uvil_events.clear()
    uvil_module._reasoning_artifacts.clear()
    uvil_module._committed_proposal_ids.clear()
    uvil_module._partition_reasoning_artifacts.clear()
    uvil_module._epoch_counter = 0


class TestStatelessCommit:
    """Tests for stateless commit using draft_payload."""

    def test_propose_partition_returns_draft_payload(self, client):
        """Verify propose_partition response includes draft_payload."""
        response = client.post(
            "/uvil/propose_partition",
            json={"problem_statement": "Test problem"}
        )
        assert response.status_code == 200
        data = response.json()

        # Must have draft_payload
        assert "draft_payload" in data, "Response must include draft_payload"

        # draft_payload must have required fields
        draft_payload = data["draft_payload"]
        assert "problem_statement" in draft_payload
        assert "claims" in draft_payload
        assert "proposal_id" in draft_payload

        # proposal_id must match top-level
        assert draft_payload["proposal_id"] == data["proposal_id"]

    def test_commit_stateless_does_not_require_server_cache(self, client):
        """
        CRITICAL: Commit using draft_payload succeeds even after cache is cleared.

        This is the core fix for demo reliability.
        """
        # Step 1: Propose
        propose_response = client.post(
            "/uvil/propose_partition",
            json={"problem_statement": "Stateless commit test"}
        )
        assert propose_response.status_code == 200
        propose_data = propose_response.json()
        draft_payload = propose_data["draft_payload"]

        # Step 2: CLEAR THE CACHE (simulates server restart / different instance)
        uvil_module._draft_proposals.clear()

        # Step 3: Commit using draft_payload (should succeed!)
        commit_response = client.post(
            "/uvil/commit_uvil",
            json={
                "draft_payload": draft_payload,
                "edited_claims": [
                    {"claim_text": "2 + 2 = 4", "trust_class": "MV", "rationale": "test"}
                ],
                "user_fingerprint": "test_user"
            }
        )

        # MUST succeed despite cache being empty
        assert commit_response.status_code == 200, (
            f"Stateless commit failed: {commit_response.json()}"
        )
        commit_data = commit_response.json()
        assert "committed_partition_id" in commit_data
        assert "u_t" in commit_data
        assert "r_t" in commit_data
        assert "h_t" in commit_data

    def test_commit_legacy_requires_cache(self, client):
        """Legacy commit path (proposal_id only) requires cache and fails with new message."""
        # Step 1: Propose
        propose_response = client.post(
            "/uvil/propose_partition",
            json={"problem_statement": "Legacy commit test"}
        )
        assert propose_response.status_code == 200
        propose_data = propose_response.json()
        proposal_id = propose_data["proposal_id"]

        # Step 2: Clear cache
        uvil_module._draft_proposals.clear()

        # Step 3: Attempt legacy commit (should fail)
        commit_response = client.post(
            "/uvil/commit_uvil",
            json={
                "proposal_id": proposal_id,
                "edited_claims": [
                    {"claim_text": "2 + 2 = 4", "trust_class": "MV", "rationale": "test"}
                ],
                "user_fingerprint": "test_user"
            }
        )

        # Should fail with 404 and non-blaming message
        assert commit_response.status_code == 404
        detail = commit_response.json()["detail"]
        assert detail["error_code"] == "PROPOSAL_STATE_LOST"
        # Message must explicitly say it's NOT a user error
        assert "not a user error" in detail["message"].lower() or "demo reliability" in detail["message"].lower()
        # Should provide recovery hint
        assert "refresh" in detail["message"].lower() or "retry" in detail["message"].lower()

    def test_commit_without_any_proposal_data_fails_400(self, client):
        """Commit without proposal_id or draft_payload returns 400."""
        response = client.post(
            "/uvil/commit_uvil",
            json={
                "edited_claims": [
                    {"claim_text": "2 + 2 = 4", "trust_class": "MV", "rationale": "test"}
                ],
                "user_fingerprint": "test_user"
            }
        )
        assert response.status_code == 400
        detail = response.json()["detail"]
        assert detail["error_code"] == "MISSING_PROPOSAL_DATA"


class TestBoundaryDemoReliability:
    """Tests for boundary demo sequence reliability."""

    def test_boundary_demo_sequence_does_not_error(self, client):
        """
        Full boundary demo sequence must not produce 404/422 from proposal lookup.

        This simulates what the frontend does during boundary demo.
        """
        steps = [
            {"claim": "2 + 2 = 4", "trust_class": "ADV", "expected_outcome": "ABSTAINED"},
            {"claim": "2 + 2 = 4", "trust_class": "PA", "expected_outcome": "ABSTAINED"},
            {"claim": "2 + 2 = 4", "trust_class": "MV", "expected_outcome": "VERIFIED"},
            {"claim": "3 * 3 = 8", "trust_class": "MV", "expected_outcome": "REFUTED"},
        ]

        for i, step in enumerate(steps):
            # Propose
            propose_response = client.post(
                "/uvil/propose_partition",
                json={"problem_statement": f"Boundary demo step {i+1}"}
            )
            assert propose_response.status_code == 200, f"Propose failed on step {i+1}"
            draft_payload = propose_response.json()["draft_payload"]

            # Commit using draft_payload (stateless)
            commit_response = client.post(
                "/uvil/commit_uvil",
                json={
                    "draft_payload": draft_payload,
                    "edited_claims": [{
                        "claim_text": step["claim"],
                        "trust_class": step["trust_class"],
                        "rationale": "boundary demo"
                    }],
                    "user_fingerprint": "boundary_demo"
                }
            )
            assert commit_response.status_code == 200, (
                f"Commit failed on step {i+1}: {commit_response.json()}"
            )
            committed_id = commit_response.json()["committed_partition_id"]

            # Verify
            verify_response = client.post(
                "/uvil/run_verification",
                json={"committed_partition_id": committed_id}
            )
            assert verify_response.status_code == 200, (
                f"Verify failed on step {i+1}: {verify_response.json()}"
            )
            outcome = verify_response.json()["outcome"]
            assert outcome == step["expected_outcome"], (
                f"Step {i+1}: expected {step['expected_outcome']}, got {outcome}"
            )

    def test_boundary_demo_works_after_cache_clear(self, client):
        """
        Boundary demo works even if cache is cleared between steps.

        This simulates multi-instance routing where each request
        might hit a different server.
        """
        # Step 1: Propose on "Instance A"
        propose_response = client.post(
            "/uvil/propose_partition",
            json={"problem_statement": "Multi-instance test"}
        )
        assert propose_response.status_code == 200
        draft_payload = propose_response.json()["draft_payload"]

        # Simulate request going to "Instance B" (empty cache)
        uvil_module._draft_proposals.clear()

        # Step 2: Commit on "Instance B" using draft_payload
        commit_response = client.post(
            "/uvil/commit_uvil",
            json={
                "draft_payload": draft_payload,
                "edited_claims": [{
                    "claim_text": "2 + 2 = 4",
                    "trust_class": "MV",
                    "rationale": "test"
                }],
                "user_fingerprint": "test"
            }
        )
        assert commit_response.status_code == 200


class TestDoubleCommitSemantics:
    """Tests for double-commit behavior with stateless path."""

    def test_double_commit_with_same_draft_payload_returns_409(self, client):
        """Double commit with same proposal_id (from draft_payload) returns 409."""
        # Propose
        propose_response = client.post(
            "/uvil/propose_partition",
            json={"problem_statement": "Double commit test"}
        )
        draft_payload = propose_response.json()["draft_payload"]

        # First commit
        first_commit = client.post(
            "/uvil/commit_uvil",
            json={
                "draft_payload": draft_payload,
                "edited_claims": [{"claim_text": "1+1=2", "trust_class": "MV", "rationale": ""}],
                "user_fingerprint": "test"
            }
        )
        assert first_commit.status_code == 200

        # Second commit with SAME draft_payload
        second_commit = client.post(
            "/uvil/commit_uvil",
            json={
                "draft_payload": draft_payload,
                "edited_claims": [{"claim_text": "2+2=4", "trust_class": "MV", "rationale": ""}],
                "user_fingerprint": "test"
            }
        )
        assert second_commit.status_code == 409
        detail = second_commit.json()["detail"]
        assert detail["error_code"] == "DOUBLE_COMMIT"

    def test_different_draft_payloads_both_succeed(self, client):
        """Different proposal_ids can both be committed."""
        # First proposal
        first_propose = client.post(
            "/uvil/propose_partition",
            json={"problem_statement": "First proposal"}
        )
        first_draft = first_propose.json()["draft_payload"]

        # Second proposal
        second_propose = client.post(
            "/uvil/propose_partition",
            json={"problem_statement": "Second proposal"}
        )
        second_draft = second_propose.json()["draft_payload"]

        # Both commits should succeed
        first_commit = client.post(
            "/uvil/commit_uvil",
            json={
                "draft_payload": first_draft,
                "edited_claims": [{"claim_text": "1+1=2", "trust_class": "MV", "rationale": ""}],
                "user_fingerprint": "test"
            }
        )
        assert first_commit.status_code == 200

        second_commit = client.post(
            "/uvil/commit_uvil",
            json={
                "draft_payload": second_draft,
                "edited_claims": [{"claim_text": "2+2=4", "trust_class": "MV", "rationale": ""}],
                "user_fingerprint": "test"
            }
        )
        assert second_commit.status_code == 200
