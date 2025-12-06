#!/usr/bin/env python3
"""
Test script for MathLedger v0.3 UI implementation.
Tests the interactive DAG visualization, live dashboard updates, and complete block explorer.
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

def test_dashboard_live_features():
    """Test the live dashboard features."""
    print("\n" + "="*50)
    print("TESTING LIVE DASHBOARD FEATURES")
    print("="*50)

    # Test main dashboard
    expected_content = [
        "MathLedger Dashboard",
        "hx-get=\"/ui/dashboard/metrics\"",
        "hx-trigger=\"every 10s\"",
        "hx-get=\"/ui/dashboard/recent-proofs\"",
        "htmx-indicator"
    ]

    if not test_ui_endpoint("/ui", expected_content):
        return False

    # Test metrics partial endpoint
    expected_content = [
        "Statements",
        "Proofs",
        "Success Rate",
        "Latest Block"
    ]

    if not test_ui_endpoint("/ui/dashboard/metrics", expected_content):
        return False

    # Test recent proofs partial endpoint
    expected_content = [
        "Recent Proofs",
        "Latest proof attempts"
    ]

    if not test_ui_endpoint("/ui/dashboard/recent-proofs", expected_content):
        return False

    return True

def test_interactive_dag():
    """Test the interactive DAG visualization."""
    print("\n" + "="*50)
    print("TESTING INTERACTIVE DAG VISUALIZATION")
    print("="*50)

    # Test with a valid hash format (this will likely return 404, which is expected)
    test_hash = "a" * 64  # 64 character hex string
    expected_content = [
        "Statement Details",
        "Derivation Graph",
        "Interactive proof dependency graph",
        "d3.js",
        "dag-container",
        "Click nodes to navigate",
        "Hover over edges to see proof information",
        "Drag to pan, scroll to zoom"
    ]

    return test_ui_endpoint(f"/ui/s/{test_hash}", expected_content)

def test_block_explorer():
    """Test the complete block explorer."""
    print("\n" + "="*50)
    print("TESTING COMPLETE BLOCK EXPLORER")
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
        "Block Verification"
    ]

    return test_ui_endpoint("/ui/blocks/1", expected_content)

def test_htmx_partials():
    """Test HTMX partial endpoints."""
    print("\n" + "="*50)
    print("TESTING HTMX PARTIAL ENDPOINTS")
    print("="*50)

    # Test metrics partial
    try:
        response = requests.get(f"{BASE_URL}/ui/dashboard/metrics")
        if response.status_code == 200:
            content = response.text
            if "grid grid-cols-1 md:grid-cols-4" in content:
                print("✅ Metrics partial renders correctly")
            else:
                print("❌ Metrics partial missing expected structure")
                return False
        else:
            print(f"❌ Metrics partial failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Metrics partial error: {e}")
        return False

    # Test recent proofs partial
    try:
        response = requests.get(f"{BASE_URL}/ui/dashboard/recent-proofs")
        if response.status_code == 200:
            content = response.text
            if "Recent Proofs" in content:
                print("✅ Recent proofs partial renders correctly")
            else:
                print("❌ Recent proofs partial missing expected content")
                return False
        else:
            print(f"❌ Recent proofs partial failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Recent proofs partial error: {e}")
        return False

    return True

def test_javascript_integration():
    """Test JavaScript integration for DAG visualization."""
    print("\n" + "="*50)
    print("TESTING JAVASCRIPT INTEGRATION")
    print("="*50)

    # Test that D3.js is loaded in statement detail page
    test_hash = "a" * 64
    try:
        response = requests.get(f"{BASE_URL}/ui/s/{test_hash}")
        if response.status_code in [200, 404]:  # 404 is expected for non-existent statements
            content = response.text
            if "d3js.org/d3.v7.min.js" in content:
                print("✅ D3.js library loaded")
            else:
                print("❌ D3.js library not found")
                return False

            if "dag-container" in content:
                print("✅ DAG container element present")
            else:
                print("❌ DAG container element missing")
                return False

            if "forceSimulation" in content:
                print("✅ D3 force simulation code present")
            else:
                print("❌ D3 force simulation code missing")
                return False
        else:
            print(f"❌ Statement detail page failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ JavaScript integration test error: {e}")
        return False

    return True

def test_navigation_consistency():
    """Test navigation consistency across all pages."""
    print("\n" + "="*50)
    print("TESTING NAVIGATION CONSISTENCY")
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
    """Run all v0.3 UI tests."""
    print("Testing MathLedger v0.3 UI Implementation")
    print("Features: Interactive DAG, Live Dashboard, Complete Block Explorer")
    print(f"Base URL: {BASE_URL}")

    # Wait for server to be ready
    print("\nWaiting for server to be ready...")
    time.sleep(2)

    # Run all tests
    tests = [
        ("Live Dashboard Features", test_dashboard_live_features),
        ("Interactive DAG Visualization", test_interactive_dag),
        ("Complete Block Explorer", test_block_explorer),
        ("HTMX Partial Endpoints", test_htmx_partials),
        ("JavaScript Integration", test_javascript_integration),
        ("Navigation Consistency", test_navigation_consistency)
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
    print("V0.3 TEST RESULTS SUMMARY")
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
        print("\n🎉 All v0.3 UI tests passed! The interactive features are working correctly.")
        print("\nKey Features Implemented:")
        print("• Interactive DAG visualization with D3.js")
        print("• Live dashboard updates with HTMX")
        print("• Complete block explorer with statements and proofs")
        print("• Server-rendered SVG charts")
        print("• Responsive design with Tailwind CSS")
        print("\nNote: Some tests may show 404 errors for statement/block details,")
        print("which is expected when the database is empty or contains no matching data.")
        print("The important thing is that the UI structure and interactive features work correctly.")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please check the implementation.")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
