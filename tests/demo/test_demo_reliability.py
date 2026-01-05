"""
Demo Reliability Regression Tests - v0.2.10

Tests proving the fix for the demo reliability issue where commit_uvil
could fail due to proposal_id cache miss after server restart.

Key invariants tested:
1. commit_uvil with draft_payload works even when proposal cache is cleared
2. Boundary demo sequence completes end-to-end on a single server instance
3. Error messages are system-responsible (not user-blaming)

Run: uv run pytest tests/demo/test_demo_reliability.py -v
"""

import pytest
from fastapi.testclient import TestClient


# Import the demo app
from demo.app import app
from backend.api import uvil


@pytest.fixture
def client():
    """Create a test client for the demo app."""
    return TestClient(app)


@pytest.fixture
def fresh_uvil_state():
    """
    Clear UVIL in-memory state to simulate server restart.

    This fixture simulates the condition that caused the original bug:
    the proposal_id lookup failing because _draft_proposals is empty.
    """
    # Store original state
    original_proposals = uvil._draft_proposals.copy()
    original_snapshots = uvil._committed_snapshots.copy()
    original_events = uvil._uvil_events.copy()
    original_artifacts = uvil._reasoning_artifacts.copy()
    original_committed_ids = uvil._committed_proposal_ids.copy()
    original_epoch = uvil._epoch_counter

    # Clear state
    uvil._draft_proposals.clear()
    uvil._committed_snapshots.clear()
    uvil._uvil_events.clear()
    uvil._reasoning_artifacts.clear()
    uvil._committed_proposal_ids.clear()
    uvil._epoch_counter = 0

    yield

    # Restore original state
    uvil._draft_proposals.update(original_proposals)
    uvil._committed_snapshots.update(original_snapshots)
    uvil._uvil_events.extend(original_events)
    uvil._reasoning_artifacts.extend(original_artifacts)
    uvil._committed_proposal_ids.update(original_committed_ids)
    uvil._epoch_counter = original_epoch


class TestSelfSufficientCommit:
    """Tests for self-sufficient commit using draft_payload."""

    def test_commit_with_draft_payload_succeeds(self, client):
        """
        Commit with draft_payload should succeed even without proposal_id lookup.

        This is the core fix for v0.2.10: draft_payload makes commit self-sufficient.
        """
        # Step 1: Propose
        propose_response = client.post(
            "/uvil/propose_partition",
            json={"problem_statement": "Self-sufficient commit test"}
        )
        assert propose_response.status_code == 200
        propose_data = propose_response.json()

        # Verify draft_payload is included
        assert "draft_payload" in propose_data
        draft_payload = propose_data["draft_payload"]
        assert "proposal_id" in draft_payload
        assert "problem_statement" in draft_payload
        assert "claims" in draft_payload

        # Step 2: Commit using draft_payload (NOT proposal_id)
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
        assert commit_response.status_code == 200
        commit_data = commit_response.json()
        assert "committed_partition_id" in commit_data
        assert "u_t" in commit_data
        assert "r_t" in commit_data
        assert "h_t" in commit_data

    def test_commit_works_after_cache_clear(self, client, fresh_uvil_state):
        """
        Commit with draft_payload succeeds even when proposal cache is cleared.

        This simulates server restart: _draft_proposals is empty, but commit
        still works because draft_payload contains all needed data.
        """
        # Step 1: Propose (this adds to cache, but we'll clear it)
        propose_response = client.post(
            "/uvil/propose_partition",
            json={"problem_statement": "Cache cleared test"}
        )
        assert propose_response.status_code == 200
        draft_payload = propose_response.json()["draft_payload"]
        proposal_id = draft_payload["proposal_id"]

        # Verify proposal is in cache
        assert proposal_id in uvil._draft_proposals

        # Step 2: Clear cache (simulate server restart)
        uvil._draft_proposals.clear()
        assert proposal_id not in uvil._draft_proposals

        # Step 3: Commit with draft_payload should STILL work
        commit_response = client.post(
            "/uvil/commit_uvil",
            json={
                "draft_payload": draft_payload,
                "edited_claims": [
                    {"claim_text": "3 + 3 = 6", "trust_class": "MV", "rationale": "test"}
                ],
                "user_fingerprint": "test_user"
            }
        )
        # This should succeed, not fail with 404
        assert commit_response.status_code == 200
        commit_data = commit_response.json()
        assert "committed_partition_id" in commit_data

    def test_commit_without_draft_payload_fails_after_cache_clear(self, client, fresh_uvil_state):
        """
        Commit with only proposal_id fails after cache clear (legacy behavior).

        This test documents the original bug behavior and verifies the error
        message is now system-responsible.
        """
        # Step 1: Generate a random proposal_id that won't be in cache
        fake_proposal_id = "00000000-0000-0000-0000-000000000000"

        # Step 2: Try to commit with only proposal_id
        commit_response = client.post(
            "/uvil/commit_uvil",
            json={
                "proposal_id": fake_proposal_id,
                "edited_claims": [
                    {"claim_text": "1 + 1 = 2", "trust_class": "MV", "rationale": "test"}
                ],
                "user_fingerprint": "test_user"
            }
        )

        # Should fail with 404
        assert commit_response.status_code == 404

        # Error should be system-responsible, not user-blaming
        error_data = commit_response.json()["detail"]
        assert error_data["error_code"] == "PROPOSAL_STATE_LOST"
        assert "demo reliability issue" in error_data["message"].lower()
        assert "not a user error" in error_data["message"].lower()
        # Should NOT contain old user-blaming message
        assert "Must call /propose_partition first" not in str(error_data)


class TestBoundaryDemoSequence:
    """Tests for the boundary demo sequence."""

    def test_boundary_demo_full_sequence(self, client):
        """
        Boundary demo sequence completes end-to-end on a single server instance.

        This tests the exact sequence used by the boundary demo:
        1. ADV claim -> ABSTAINED (excluded from R_t)
        2. PA claim -> ABSTAINED (no validator)
        3. MV claim (true) -> VERIFIED
        4. MV claim (false) -> REFUTED
        """
        steps = [
            ("2 + 2 = 4", "ADV", "ABSTAINED"),
            ("2 + 2 = 4", "PA", "ABSTAINED"),
            ("2 + 2 = 4", "MV", "VERIFIED"),
            ("3 * 3 = 8", "MV", "REFUTED"),
        ]

        for claim_text, trust_class, expected_outcome in steps:
            # Propose
            propose_response = client.post(
                "/uvil/propose_partition",
                json={"problem_statement": f"Boundary demo: {trust_class}"}
            )
            assert propose_response.status_code == 200
            draft_payload = propose_response.json()["draft_payload"]

            # Commit using draft_payload
            commit_response = client.post(
                "/uvil/commit_uvil",
                json={
                    "draft_payload": draft_payload,
                    "edited_claims": [
                        {"claim_text": claim_text, "trust_class": trust_class, "rationale": "test"}
                    ],
                    "user_fingerprint": "boundary_demo_test"
                }
            )
            assert commit_response.status_code == 200, f"Commit failed for {trust_class}"
            committed_id = commit_response.json()["committed_partition_id"]

            # Verify
            verify_response = client.post(
                "/uvil/run_verification",
                json={"committed_partition_id": committed_id}
            )
            assert verify_response.status_code == 200, f"Verify failed for {trust_class}"

            # Check outcome
            outcome = verify_response.json()["outcome"]
            assert outcome == expected_outcome, \
                f"Expected {expected_outcome} for {trust_class} '{claim_text}', got {outcome}"


class TestErrorMessages:
    """Tests for system-responsible error messages."""

    def test_proposal_state_lost_error_is_system_responsible(self, client, fresh_uvil_state):
        """Error message for cache miss is system-responsible."""
        commit_response = client.post(
            "/uvil/commit_uvil",
            json={
                "proposal_id": "nonexistent-id",
                "edited_claims": [{"claim_text": "x", "trust_class": "MV", "rationale": ""}],
            }
        )

        assert commit_response.status_code == 404
        error = commit_response.json()["detail"]

        # Check for system-responsible language
        assert error["error_code"] == "PROPOSAL_STATE_LOST"
        assert "demo reliability issue" in error["message"].lower()
        assert "not a user error" in error["message"].lower()
        assert "recovery_hint" in error

    def test_missing_proposal_data_error(self, client):
        """Error when neither proposal_id nor draft_payload provided."""
        commit_response = client.post(
            "/uvil/commit_uvil",
            json={
                "edited_claims": [{"claim_text": "x", "trust_class": "MV", "rationale": ""}],
            }
        )

        assert commit_response.status_code == 400
        error = commit_response.json()["detail"]
        assert error["error_code"] == "MISSING_PROPOSAL_DATA"


class TestDoubleCommitProtection:
    """Tests for double-commit protection with draft_payload."""

    def test_double_commit_rejected_with_draft_payload(self, client):
        """
        Double commit is rejected even when using draft_payload.

        The proposal_id in draft_payload is used as the idempotency key.
        """
        # Propose
        propose_response = client.post(
            "/uvil/propose_partition",
            json={"problem_statement": "Double commit test"}
        )
        draft_payload = propose_response.json()["draft_payload"]

        # First commit should succeed
        first_commit = client.post(
            "/uvil/commit_uvil",
            json={
                "draft_payload": draft_payload,
                "edited_claims": [{"claim_text": "1 + 1 = 2", "trust_class": "MV", "rationale": ""}],
            }
        )
        assert first_commit.status_code == 200

        # Second commit with same draft_payload should fail
        second_commit = client.post(
            "/uvil/commit_uvil",
            json={
                "draft_payload": draft_payload,
                "edited_claims": [{"claim_text": "2 + 2 = 4", "trust_class": "MV", "rationale": ""}],
            }
        )
        assert second_commit.status_code == 409
        error = second_commit.json()["detail"]
        assert error["error_code"] == "DOUBLE_COMMIT"
