# backend/dashboard/test_api.py
import pytest
from fastapi.testclient import TestClient
import os

# Adjust the path to import from the parent directory
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from . import api, storage

client = TestClient(api.app)

# Sample data matching the test vectors
SAMPLE_DATA = [
    {
      "timestamp": "2025-12-10T10:05:00Z",
      "run_id": "run-a4b1f2",
      "layer": "P4",
      "governance_status": "WARNING",
      "metrics": { "delta_p": 0.15, "rsi": 78.9, "omega": 0.96, "divergence": 0.08, "quarantine_ratio": 0.05, "budget_invalid_percent": 0.8 }
    },
    {
      "timestamp": "2025-12-10T10:00:00Z",
      "run_id": "run-a4b1c8",
      "layer": "P4",
      "governance_status": "STABLE",
      "metrics": { "delta_p": 0.05, "rsi": 65.2, "omega": 0.98, "divergence": 0.01, "quarantine_ratio": 0.0, "budget_invalid_percent": 0.1 }
    },
    {
      "timestamp": "2025-12-10T09:55:00Z",
      "run_id": "run-a4b1a0",
      "layer": "P3",
      "governance_status": "STABLE",
      "metrics": { "delta_p": 0.02, "rsi": 55.0, "omega": 0.99, "divergence": 0.005, "quarantine_ratio": 0.0, "budget_invalid_percent": 0.05 }
    }
]

@pytest.fixture(scope="module", autouse=True)
def setup_database():
    """Fixture to set up and tear down the database for the test module."""
    storage.initialize_db()
    storage.clear_db_for_testing()
    for event in SAMPLE_DATA:
        storage.insert_event(event)

    yield # This is where the tests run

    # Teardown: remove the test database file
    os.remove(storage.DB_PATH)


def test_vector_1_get_last_run_for_layer_p4():
    """
    Tests fetching the single most recent event for a specific layer.
    Corresponds to Test Vector 1.
    """
    response = client.get("/api/v1/governance/history?limit=1&layers=P4")
    assert response.status_code == 200
    result = response.json()

    assert len(result["data"]) == 1
    assert result["data"][0]["timestamp"] == "2025-12-10T10:05:00Z"
    assert result["data"][0]["layer"] == "P4"
    assert result["pagination"]["limit"] == 1
    assert result["pagination"]["has_more"] is True


def test_vector_2_get_records_in_time_window():
    """
    Tests fetching all records within a specific time window.
    Corresponds to Test Vector 2.
    """
    response = client.get("/api/v1/governance/history?start_time=2025-12-10T09:58:00Z&end_time=2025-12-10T10:06:00Z")
    assert response.status_code == 200
    result = response.json()

    assert len(result["data"]) == 2
    # Results are ordered newest to oldest
    assert result["data"][0]["timestamp"] == "2025-12-10T10:05:00Z"
    assert result["data"][1]["timestamp"] == "2025-12-10T10:00:00Z"
    assert result["pagination"]["has_more"] is False


def test_get_all_layers():
    """Tests fetching events from all layers without a layer filter."""
    response = client.get("/api/v1/governance/history?limit=3")
    assert response.status_code == 200
    result = response.json()
    assert len(result["data"]) == 3
    layers_returned = {event["layer"] for event in result["data"]}
    assert layers_returned == {"P3", "P4"}

def test_limit_parameter():
    """Tests that the limit parameter is respected."""
    response = client.get("/api/v1/governance/history?limit=2")
    assert response.status_code == 200
    result = response.json()
    assert len(result["data"]) == 2

def test_invalid_parameter_handling():
    """
    FastAPI handles basic type validation, e.g., for non-integer limits.
    This test is for documentation.
    """
    response = client.get("/api/v1/governance/history?limit=not-an-integer")
    assert response.status_code == 422 # Unprocessable Entity

def test_database_indexes_are_created():
    """
    Verifies that the database indexes specified in the performance plan are created.
    """
    # The setup_database fixture already calls initialize_db()
    conn = storage._get_db_connection()
    cursor = conn.cursor()
    cursor.execute("PRAGMA index_list('governance_events')")
    indexes = [row['name'] for row in cursor.fetchall()]
    conn.close()

    assert 'idx_governance_events_layer_timestamp' in indexes
    assert 'idx_governance_events_timestamp' in indexes
