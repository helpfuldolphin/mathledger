#!/usr/bin/env python3
"""
Test script for MathLedger v0.5 UI implementation.
Tests the search functionality, block explorer, and verification features.
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

def test_search_endpoint():
    """Test the comprehensive search endpoint."""
    print("\n" + "="*50)
    print("TESTING SEARCH ENDPOINT")
    print("="*50)

    # Test basic search
    try:
        response = requests.get(f"{BASE_URL}/search?q=test")
        if response.status_code == 200:
            data = response.json()
            required_fields = ["results", "total", "limit", "offset", "has_more", "query"]
            for field in required_fields:
                if field in data:
                    print(f"✅ Search API has {field}")
                else:
                    print(f"❌ Search API missing {field}")
                    return False
        else:
            print(f"❌ Search API failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Search API error: {e}")
        return False

    # Test search with filters
    try:
        response = requests.get(f"{BASE_URL}/search?q=test&system=pl&status=proven&depth_gt=0")
        if response.status_code == 200:
            data = response.json()
            if "query" in data and data["query"]["system"] == "pl":
                print("✅ Search filters working correctly")
            else:
                print("❌ Search filters not working")
                return False
        else:
            print(f"❌ Filtered search failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Filtered search error: {e}")
        return False

    return True

def test_search_ui():
    """Test the search UI functionality."""
    print("\n" + "="*50)
    print("TESTING SEARCH UI")
    print("="*50)

    # Test main dashboard for search bar
    expected_content = [
        "Search Statements",
        "Text Search",
        "Theory",
        "Min Depth",
        "Status",
        "Search",
        "search-form",
        "search-results-container"
    ]

    return test_ui_endpoint("/ui", expected_content)

def test_search_partial():
    """Test the search partial endpoint."""
    print("\n" + "="*50)
    print("TESTING SEARCH PARTIAL")
    print("="*50)

    try:
        response = requests.get(f"{BASE_URL}/ui/dashboard/search?q=test")
        if response.status_code == 200:
            content = response.text
            if "Search Results" in content:
                print("✅ Search partial renders correctly")
            else:
                print("❌ Search partial missing expected content")
                return False
        else:
            print(f"❌ Search partial failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Search partial error: {e}")
        return False

    return True

def test_block_explorer():
    """Test the block explorer functionality."""
    print("\n" + "="*50)
    print("TESTING BLOCK EXPLORER")
    print("="*50)

    # Test block listing
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

    if not test_ui_endpoint("/ui/blocks", expected_content):
        return False

    # Test block detail (this will likely return 404, which is expected)
    expected_content = [
        "Block Details",
        "Block Statistics",
        "Statements in Block",
        "Proofs in Block",
        "Block Information",
        "Verify Block Integrity",
        "merkle-root",
        "verification-result"
    ]

    return test_ui_endpoint("/ui/blocks/1", expected_content)

def test_copy_lean_feature():
    """Test the Copy for Lean functionality."""
    print("\n" + "="*50)
    print("TESTING COPY FOR LEAN FEATURE")
    print("="*50)

    # Test statement detail page for Copy for Lean button
    test_hash = "a" * 64
    expected_content = [
        "Copy for Lean",
        "copyLeanProof",
        "MathLedger Statement:",
        "Generated:",
        "TODO: Implement proof",
        "theorem statement_",
        "Verification:"
    ]

    return test_ui_endpoint(f"/ui/s/{test_hash}", expected_content)

def test_verification_features():
    """Test the verification features."""
    print("\n" + "="*50)
    print("TESTING VERIFICATION FEATURES")
    print("="*50)

    # Test block detail page for verification features
    expected_content = [
        "Verify Block Integrity",
        "verifyBlockIntegrity",
        "sha256",
        "calculateMerkleRoot",
        "data-statement-hash",
        "verification-result"
    ]

    return test_ui_endpoint("/ui/blocks/1", expected_content)

def test_javascript_integration():
    """Test JavaScript integration for new features."""
    print("\n" + "="*50)
    print("TESTING JAVASCRIPT INTEGRATION")
    print("="*50)

    # Test search functionality
    try:
        response = requests.get(f"{BASE_URL}/ui")
        if response.status_code == 200:
            content = response.text
            if "loadSearchResults" in content:
                print("✅ Search JavaScript functionality present")
            else:
                print("❌ Search JavaScript functionality missing")
                return False

            if "search-form" in content:
                print("✅ Search form integration present")
            else:
                print("❌ Search form integration missing")
                return False
        else:
            print(f"❌ Dashboard failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ JavaScript integration test error: {e}")
        return False

    return True

def test_api_endpoints():
    """Test new API endpoints."""
    print("\n" + "="*50)
    print("TESTING NEW API ENDPOINTS")
    print("="*50)

    # Test search endpoint
    try:
        response = requests.get(f"{BASE_URL}/search")
        if response.status_code == 200:
            print("✅ Search endpoint working")
        else:
            print(f"❌ Search endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Search endpoint error: {e}")
        return False

    # Test search partial endpoint
    try:
        response = requests.get(f"{BASE_URL}/ui/dashboard/search")
        if response.status_code == 200:
            print("✅ Search partial endpoint working")
        else:
            print(f"❌ Search partial endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Search partial endpoint error: {e}")
        return False

    return True

def main():
    """Run all v0.5 UI tests."""
    print("Testing MathLedger v0.5 UI Implementation")
    print("Features: Search, Block Explorer, Copy for Lean, Verification")
    print(f"Base URL: {BASE_URL}")

    # Wait for server to be ready
    print("\nWaiting for server to be ready...")
    time.sleep(2)

    # Run all tests
    tests = [
        ("Search Endpoint", test_search_endpoint),
        ("Search UI", test_search_ui),
        ("Search Partial", test_search_partial),
        ("Block Explorer", test_block_explorer),
        ("Copy for Lean Feature", test_copy_lean_feature),
        ("Verification Features", test_verification_features),
        ("JavaScript Integration", test_javascript_integration),
        ("New API Endpoints", test_api_endpoints)
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
    print("V0.5 TEST RESULTS SUMMARY")
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
        print("\n🎉 All v0.5 UI tests passed! The search and verification features are working correctly.")
        print("\nKey Features Implemented:")
        print("• Comprehensive search endpoint with filtering")
        print("• Interactive search UI with real-time results")
        print("• Complete block explorer with verification")
        print("• Copy for Lean functionality")
        print("• Block integrity verification")
        print("• Enhanced navigation and trust features")
        print("\nNote: Some tests may show 404 errors for statement/block details,")
        print("which is expected when the database is empty or contains no matching data.")
        print("The important thing is that the UI structure and features work correctly.")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please check the implementation.")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
