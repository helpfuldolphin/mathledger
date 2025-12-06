#!/usr/bin/env python3
"""
Enhanced API test script for MathLedger.
Tests the improved endpoints with comprehensive validation.
"""

import requests
import json
import sys
import time
from typing import Dict, Any

BASE_URL = "http://localhost:8010"
API_KEY = "devkey"

def test_endpoint(endpoint: str, method: str = "GET", data: Dict[str, Any] = None,
                 headers: Dict[str, str] = None, expected_status: int = 200) -> Dict[str, Any]:
    """Test an API endpoint and validate response."""
    url = f"{BASE_URL}{endpoint}"
    if headers is None:
        headers = {"X-API-Key": API_KEY}

    try:
        if method == "GET":
            response = requests.get(url, headers=headers, params=data)
        elif method == "POST":
            response = requests.post(url, headers=headers, json=data)

        print(f"\n{method} {endpoint}")
        print(f"Status: {response.status_code}")

        if response.status_code == expected_status:
            print("✅ Status code matches expected")
        else:
            print(f"❌ Expected {expected_status}, got {response.status_code}")

        if response.status_code == 200:
            try:
                data = response.json()
                print(f"Response: {json.dumps(data, indent=2)}")
                return data
            except:
                print(f"Response: {response.text}")
                return {"text": response.text}
        else:
            print(f"Error: {response.text}")
            return {"error": response.text}

    except requests.exceptions.ConnectionError:
        print(f"\n{method} {endpoint}")
        print("❌ Could not connect to server. Make sure it's running on port 8010")
        return {"error": "connection_failed"}
    except Exception as e:
        print(f"\n{method} {endpoint}")
        print(f"❌ Error: {e}")
        return {"error": str(e)}

def test_metrics_endpoint():
    """Test the enhanced metrics endpoint."""
    print("\n" + "="*50)
    print("TESTING ENHANCED METRICS ENDPOINT")
    print("="*50)

    data = test_endpoint("/metrics")

    if "error" not in data:
        # Validate enhanced metrics structure
        required_fields = [
            "proofs", "proofs_by_prover", "proofs_by_method", "block_count",
            "max_depth", "queue_length", "statements_by_status",
            "derivation_rules", "recent_activity"
        ]

        for field in required_fields:
            if field in data:
                print(f"✅ {field} field present")
            else:
                print(f"❌ {field} field missing")

        # Validate proofs structure
        if "proofs" in data and isinstance(data["proofs"], dict):
            if "success" in data["proofs"] and "failure" in data["proofs"]:
                print("✅ Proofs structure valid")
            else:
                print("❌ Proofs structure invalid")

        # Validate recent activity structure
        if "recent_activity" in data and isinstance(data["recent_activity"], dict):
            if "proofs_last_hour" in data["recent_activity"] and "proofs_last_day" in data["recent_activity"]:
                print("✅ Recent activity structure valid")
            else:
                print("❌ Recent activity structure invalid")

def test_statements_endpoint():
    """Test the enhanced statements endpoint."""
    print("\n" + "="*50)
    print("TESTING ENHANCED STATEMENTS ENDPOINT")
    print("="*50)

    # Test with hash parameter
    print("\n--- Testing with hash parameter ---")
    test_hash = "test123"  # This will likely fail with 404, which is expected
    test_endpoint(f"/statements?hash={test_hash}", expected_status=404)

    # Test with text parameter
    print("\n--- Testing with text parameter ---")
    test_endpoint("/statements?text=(and p q)", expected_status=404)

    # Test with both parameters (should fail)
    print("\n--- Testing with both parameters (should fail) ---")
    test_endpoint("/statements?hash=test&text=test", expected_status=400)

    # Test with no parameters (should fail)
    print("\n--- Testing with no parameters (should fail) ---")
    test_endpoint("/statements", expected_status=400)

    # Test invalid hash format
    print("\n--- Testing invalid hash format ---")
    test_endpoint("/statements?hash=invalid_hash", expected_status=400)

def test_blocks_endpoint():
    """Test the blocks endpoint."""
    print("\n" + "="*50)
    print("TESTING BLOCKS ENDPOINT")
    print("="*50)

    data = test_endpoint("/blocks/latest")

    if "error" not in data:
        # Validate block structure
        required_fields = ["block_number", "merkle_root", "created_at", "header"]

        for field in required_fields:
            if field in data:
                print(f"✅ {field} field present")
            else:
                print(f"❌ {field} field missing")

        # Validate header structure
        if "header" in data and isinstance(data["header"], dict):
            if "run_name" in data["header"] and "statements" in data["header"]:
                print("✅ Header structure valid")
            else:
                print("❌ Header structure invalid")

def test_ui_endpoints():
    """Test UI endpoints."""
    print("\n" + "="*50)
    print("TESTING UI ENDPOINTS")
    print("="*50)

    # Test dashboard
    print("\n--- Testing dashboard ---")
    response = test_endpoint("/ui", headers={})

    if "error" not in response and "text" in response:
        if "MathLedger Dashboard" in response["text"]:
            print("✅ Dashboard content valid")
        else:
            print("❌ Dashboard content invalid")

    # Test statement detail (will likely fail with 400 due to invalid hash)
    print("\n--- Testing statement detail ---")
    test_endpoint("/ui/s/invalid_hash", headers={}, expected_status=400)

def test_authentication():
    """Test authentication requirements."""
    print("\n" + "="*50)
    print("TESTING AUTHENTICATION")
    print("="*50)

    # Test without API key
    print("\n--- Testing without API key ---")
    test_endpoint("/metrics", headers={}, expected_status=401)

    # Test with invalid API key
    print("\n--- Testing with invalid API key ---")
    test_endpoint("/metrics", headers={"X-API-Key": "invalid"}, expected_status=401)

    # Test with valid API key
    print("\n--- Testing with valid API key ---")
    test_endpoint("/metrics", headers={"X-API-Key": API_KEY}, expected_status=200)

def test_health_endpoint():
    """Test health endpoint."""
    print("\n" + "="*50)
    print("TESTING HEALTH ENDPOINT")
    print("="*50)

    data = test_endpoint("/health", headers={})

    if "error" not in data:
        if data.get("ok") is True and data.get("status") == "healthy":
            print("✅ Health check valid")
        else:
            print("❌ Health check invalid")

def main():
    """Run all enhanced API tests."""
    print("Testing Enhanced MathLedger API")
    print(f"Base URL: {BASE_URL}")
    print(f"API Key: {API_KEY}")

    # Wait for server to be ready
    print("\nWaiting for server to be ready...")
    time.sleep(2)

    # Run all tests
    test_health_endpoint()
    test_authentication()
    test_metrics_endpoint()
    test_blocks_endpoint()
    test_statements_endpoint()
    test_ui_endpoints()

    print("\n" + "="*50)
    print("ENHANCED API TESTS COMPLETED")
    print("="*50)
    print("\nNote: Some tests may show 404/400 errors which are expected")
    print("when the database is empty or contains no matching data.")
    print("The important thing is that the API structure and validation work correctly.")

if __name__ == "__main__":
    main()
