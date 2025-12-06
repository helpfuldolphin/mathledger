#!/usr/bin/env python3
"""
Simple test script to verify the new API endpoints work.
This is a temporary test file to validate the new functionality.
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_health():
    """Test the health endpoint."""
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    assert response.json()["ok"] == True
    print("✓ Health endpoint working")

def test_metrics():
    """Test the metrics endpoint includes new blocks and lemmas data."""
    response = requests.get(f"{BASE_URL}/metrics")
    assert response.status_code == 200
    data = response.json()

    # Check that new fields exist
    assert "blocks" in data
    assert "lemmas" in data
    assert "total" in data["blocks"]
    assert "total" in data["lemmas"]
    print("✓ Metrics endpoint includes blocks and lemmas data")

def test_blocks_latest():
    """Test the blocks/latest endpoint."""
    response = requests.get(f"{BASE_URL}/blocks/latest")
    # This might return 404 if no blocks exist, which is expected
    if response.status_code == 404:
        print("✓ Blocks/latest endpoint working (no blocks found - expected)")
    else:
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert "run_id" in data
        assert "counts" in data
        assert "created_at" in data
        print("✓ Blocks/latest endpoint working")

def test_lemmas_top():
    """Test the lemmas/top endpoint."""
    response = requests.get(f"{BASE_URL}/lemmas/top")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    print("✓ Lemmas/top endpoint working")

def test_ui_endpoints():
    """Test the UI endpoints return HTML."""
    # Test blocks UI
    response = requests.get(f"{BASE_URL}/ui/blocks")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    print("✓ UI/blocks endpoint working")

    # Test lemmas UI
    response = requests.get(f"{BASE_URL}/ui/lemmas")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    print("✓ UI/lemmas endpoint working")

if __name__ == "__main__":
    print("Testing new MathLedger API endpoints...")
    print("Make sure the API server is running on localhost:8000")
    print()

    try:
        test_health()
        test_metrics()
        test_blocks_latest()
        test_lemmas_top()
        test_ui_endpoints()
        print()
        print("🎉 All tests passed! New endpoints are working correctly.")
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API server. Make sure it's running on localhost:8000")
        print("Run: make api")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise
