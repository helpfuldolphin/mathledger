import pytest
from typing import Dict, Any
from fastapi.testclient import TestClient
from backend.orchestrator.app import app, get_db_connection
from tests.integration.conftest import FirstOrganismApiClient

class TestAttestationApi:
    """
    Integration tests for the Attestation API surface.
    Ensures endpoints expected by the First Organism exist and behave correctly.
    """

    def test_ui_event_lifecycle(self, api_client: FirstOrganismApiClient):
        """
        Test posting a UI event and retrieving it using the typed client.
        """
        # 1. Post a UI event
        payload = {
            "event_type": "click",
            "element_id": "submit-btn",
            "value": "confirm",
            "timestamp": 1234567890  # Explicit timestamp for control
        }
        
        response = api_client.post_ui_event(payload)
        
        assert response.timestamp == 1234567890
        assert response.event_id
        assert response.leaf_hash

        # 2. List UI events to verify it's there
        list_response = api_client.list_ui_events()
        
        found = False
        for evt in list_response.events:
            if evt.event_id == response.event_id:
                found = True
                # Verify determinism/persistence
                assert evt.timestamp == 1234567890
                assert evt.leaf_hash == response.leaf_hash
                break
        
        assert found, "Posted event not found in ui-events list"

    def test_simulate_derivation(self, api_client: FirstOrganismApiClient):
        """
        Test the derivation simulation stub.
        """
        response = api_client.simulate_derivation()
        assert response.triggered is True
        assert response.job_id == "simulated-job-id"
        assert response.status == "queued"

    def test_latest_attestation_structure(self, test_client: TestClient, api_headers: Dict[str, str]):
        """
        Test fetching the latest attestation. 
        We mock the DB to return a valid row structure because valid data might not exist in the empty test DB.
        
        Note: We manually construct the client here to inject the DB mock, 
        as the standard api_client fixture uses the standard test_client.
        """
        class MockConn:
            def cursor(self): return self
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def execute(self, *args, **kwargs): pass
            def fetchone(self): 
                # Return: block_number, reasoning_root, ui_root, composite_root, metadata
                return (10, "reasoning_root", "ui_root", "composite_root", {"block_hash": "abc"})
            def fetchall(self): return []
            def close(self): pass

        # Override dependency for this test
        app.dependency_overrides[get_db_connection] = lambda: MockConn()
        
        # Re-create client with override active
        client = TestClient(app)
        fo_client = FirstOrganismApiClient(client, api_key=api_headers.get("X-API-Key"))

        response = fo_client.get_latest_attestation()
        
        assert response.block_number == 10
        assert response.reasoning_merkle_root == "reasoning_root"
        assert response.ui_merkle_root == "ui_root"
        assert response.composite_attestation_root == "composite_root"
        assert response.attestation_metadata == {"block_hash": "abc"}
        
        # cleanup
        app.dependency_overrides.pop(get_db_connection, None)
