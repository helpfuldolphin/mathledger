# backend/dashboard/test_smoke.py
import pytest
from fastapi.testclient import TestClient
import os

from backend.dashboard import api, storage, smoke_populate_db

client = TestClient(api.app)

@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown_db():
    """
    Fixture to ensure the database file is handled cleanly for this test module.
    It removes any pre-existing DB file before tests run and cleans up after.
    """
    db_path = storage.DB_PATH
    if os.path.exists(db_path):
        os.remove(db_path)
    
    yield # Run tests
    
    if os.path.exists(db_path):
        os.remove(db_path)

def test_smoke_flow_empty_then_populated():
    """
    Tests the full smoke-test flow automatically:
    1. Checks that a fresh, empty database returns an empty list.
    2. Populates the database with 10 sample records.
    3. Verifies that the API correctly returns the 10 records.
    """
    # --- Part 1: Test Empty State ---
    storage.initialize_db() # Create empty db
    
    response_empty = client.get("/api/v1/governance/history")
    assert response_empty.status_code == 200
    result_empty = response_empty.json()
    assert result_empty["data"] == []

    # --- Part 2: Populate and Test Populated State ---
    # Use the functions from the smoke_populate_db script
    sample_events = smoke_populate_db.generate_sample_events(10)
    for event in sample_events:
        storage.insert_event(event)
    
    # Query for more than the number of items to check `has_more`
    response_populated = client.get("/api/v1/governance/history?limit=20")
    assert response_populated.status_code == 200
    result_populated = response_populated.json()

    assert len(result_populated["data"]) == 10
    # Verify order is newest first
    assert result_populated["data"][0]["run_id"] == "smoke-run-0"
    assert result_populated["data"][9]["run_id"] == "smoke-run-9"
    # We requested 20 but only got 10, so there are no more pages
    assert result_populated["pagination"]["has_more"] is False
