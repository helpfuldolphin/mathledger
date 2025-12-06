#!/usr/bin/env python3
"""
Test script for MathLedger UI implementation.
Tests the enhanced UI components including statement details, dashboard charts, and block explorer.
"""

import requests
import json
import sys
import time
from typing import Dict, Any

BASE_URL = "http://localhost:8010"

def test_ui_endpoint(endpoint: str, expected_content: list = None) -> bool:
    """Test a UI endpoint and validate content."""
    url = f"{BASE_URL}{endpoint}"

    try:
        response = requests.get(url)

        print(f"\nGET {endpoint}")
        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            print("✅ Status code correct")

            # Check content type
            if "text/html" in response.headers.get("content-type", ""):
                print("✅ Content type is HTML")
            else:
                print("❌ Content type is not HTML")
                return False

            # Check for expected content
            if expected_content:
                content = response.text
                for expected in expected_content:
                    if expected in content:
                        print(f"✅ Found expected content: '{expected}'")
                    else:
                        print(f"❌ Missing expected content: '{expected}'")
                        return False

            return True
        else:
            print(f"❌ Unexpected status code: {response.status_code}")
            return False

    except requests.exceptions.ConnectionError:
        print(f"\nGET {endpoint}")
        print("❌ Could not connect to server. Make sure it's running on port 8010")
        return False
    except Exception as e:
        print(f"\nGET {endpoint}")
        print(f"❌ Error: {e}")
        return False

def test_dashboard():
    """Test the enhanced dashboard."""
    print("\n" + "="*50)
    print("TESTING ENHANCED DASHBOARD")
    print("="*50)

    expected_content = [
        "MathLedger Dashboard",
        "Proof Status",
        "Derivation Depth",
        "Proofs by Prover",
        "Statements by Status",
        "Recent Statements",
        "Top Derivation Rules"
    ]

    return test_ui_endpoint("/ui", expected_content)

def test_statement_detail():
    """Test the statement detail page."""
    print("\n" + "="*50)
    print("TESTING STATEMENT DETAIL PAGE")
    print("="*50)

    # Test with a valid hash format (this will likely return 404, which is expected)
    test_hash = "a" * 64  # 64 character hex string
    expected_content = [
        "Statement Details",
        "Statement",
        "Proof Attempts",
        "Premises",
        "Conclusions",
        "Quick Actions"
    ]

    return test_ui_endpoint(f"/ui/s/{test_hash}", expected_content)

def test_block_explorer():
    """Test the block explorer."""
    print("\n" + "="*50)
    print("TESTING BLOCK EXPLORER")
    print("="*50)

    expected_content = [
        "Block Explorer",
        "Total Blocks",
        "Latest Block",
        "Recent Blocks",
        "Block #",
        "Theory",
        "Run",
        "Merkle Root"
    ]

    return test_ui_endpoint("/ui/blocks", expected_content)

def test_block_detail():
    """Test the block detail page."""
    print("\n" + "="*50)
    print("TESTING BLOCK DETAIL PAGE")
    print("="*50)

    # Test with block ID 1 (this will likely return 404, which is expected)
    expected_content = [
        "Block Details",
        "Block Information",
        "Block Header",
        "Statements in Block",
        "Actions",
        "Block Verification",
        "Placeholder Implementation"
    ]

    return test_ui_endpoint("/ui/blocks/1", expected_content)

def test_api_endpoints():
    """Test that API endpoints are still working."""
    print("\n" + "="*50)
    print("TESTING API ENDPOINTS")
    print("="*50)

    # Test health endpoint
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            if data.get("ok") is True:
                print("✅ Health endpoint working")
            else:
                print("❌ Health endpoint not healthy")
                return False
        else:
            print("❌ Health endpoint failed")
            return False
    except Exception as e:
        print(f"❌ Health endpoint error: {e}")
        return False

    # Test metrics endpoint (should require auth)
    try:
        response = requests.get(f"{BASE_URL}/metrics")
        if response.status_code == 401:
            print("✅ Metrics endpoint properly requires authentication")
        else:
            print("❌ Metrics endpoint should require authentication")
            return False
    except Exception as e:
        print(f"❌ Metrics endpoint error: {e}")
        return False

    return True

def test_navigation():
    """Test navigation between pages."""
    print("\n" + "="*50)
    print("TESTING NAVIGATION")
    print("="*50)

    # Test that all main navigation links exist
    try:
        response = requests.get(f"{BASE_URL}/ui")
        content = response.text

        # Check for navigation links
        nav_links = [
            'href="/ui/blocks"',
            'href="/ui/s/',
            'href="/statements?hash=',
            'href="/blocks/latest"'
        ]

        for link in nav_links:
            if link in content:
                print(f"✅ Found navigation link: {link}")
            else:
                print(f"❌ Missing navigation link: {link}")
                return False

        return True

    except Exception as e:
        print(f"❌ Navigation test error: {e}")
        return False

def main():
    """Run all UI tests."""
    print("Testing MathLedger UI Implementation")
    print(f"Base URL: {BASE_URL}")

    # Wait for server to be ready
    print("\nWaiting for server to be ready...")
    time.sleep(2)

    # Run all tests
    tests = [
        ("API Endpoints", test_api_endpoints),
        ("Dashboard", test_dashboard),
        ("Statement Detail", test_statement_detail),
        ("Block Explorer", test_block_explorer),
        ("Block Detail", test_block_detail),
        ("Navigation", test_navigation)
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "="*50)
    print("TEST RESULTS SUMMARY")
    print("="*50)

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All UI tests passed! The implementation is working correctly.")
        print("\nNote: Some tests may show 404 errors for statement/block details,")
        print("which is expected when the database is empty or contains no matching data.")
        print("The important thing is that the UI structure and navigation work correctly.")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please check the implementation.")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
