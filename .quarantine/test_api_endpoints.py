#!/usr/bin/env python3
"""
Test script for MathLedger API endpoints.
Run this after starting the server with: uv run uvicorn backend.orchestrator.app:app --port 8010
"""

import requests
import json
import sys

BASE_URL = "http://localhost:8010"
API_KEY = "devkey"  # Default API key from environment

def test_endpoint(endpoint, method="GET", data=None, headers=None):
    """Test an API endpoint and print results."""
    url = f"{BASE_URL}{endpoint}"
    if headers is None:
        headers = {"X-API-Key": API_KEY}

    try:
        if method == "GET":
            response = requests.get(url, headers=headers)
        elif method == "POST":
            response = requests.post(url, headers=headers, json=data)

        print(f"\n{method} {endpoint}")
        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            try:
                data = response.json()
                print(f"Response: {json.dumps(data, indent=2)}")
            except:
                print(f"Response: {response.text}")
        else:
            print(f"Error: {response.text}")

    except requests.exceptions.ConnectionError:
        print(f"\n{method} {endpoint}")
        print("Error: Could not connect to server. Make sure it's running on port 8010")
    except Exception as e:
        print(f"\n{method} {endpoint}")
        print(f"Error: {e}")

def main():
    print("Testing MathLedger API endpoints...")
    print(f"Base URL: {BASE_URL}")
    print(f"API Key: {API_KEY}")

    # Test API endpoints (require authentication)
    test_endpoint("/metrics")
    test_endpoint("/blocks/latest")
    test_endpoint("/statements?hash=test123")  # This will likely fail with 404

    # Test UI endpoints (no authentication required)
    print(f"\nTesting UI endpoints (no auth required)...")
    test_endpoint("/ui", headers={})
    test_endpoint("/ui/s/test123", headers={})  # This will likely fail with 404

    # Test health endpoint
    test_endpoint("/health", headers={})

    print("\nTest completed!")

if __name__ == "__main__":
    main()
