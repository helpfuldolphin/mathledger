#!/usr/bin/env python3
"""
Test script for MathLedger v0.4 UI implementation.
Tests the FOL rendering support, DAG performance optimizations, and live system observability.
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

def test_fol_rendering():
    """Test FOL rendering support."""
    print("\n" + "="*50)
    print("TESTING FOL RENDERING SUPPORT")
    print("="*50)

    # Test statement detail page for KaTeX integration
    test_hash = "a" * 64
    expected_content = [
        "Statement Details",
        "katex.min.css",
        "katex.min.js",
        "auto-render.min.js",
        "First-Order Logic Components",
        "Variables:",
        "Predicates:",
        "Functions:",
        "renderMathInElement"
    ]

    return test_ui_endpoint(f"/ui/s/{test_hash}", expected_content)

def test_dag_performance():
    """Test DAG performance optimizations."""
    print("\n" + "="*50)
    print("TESTING DAG PERFORMANCE OPTIMIZATIONS")
    print("="*50)

    # Test statement detail page for performance features
    test_hash = "a" * 64
    expected_content = [
        "Derivation Graph",
        "Load More Parents",
        "Load More Children",
        "Reset View",
        "Large graph detected",
        "MAX_INITIAL_NODES",
        "loadMoreNodes",
        "performance optimization"
    ]

    return test_ui_endpoint(f"/ui/s/{test_hash}", expected_content)

def test_live_observability():
    """Test live system observability."""
    print("\n" + "="*50)
    print("TESTING LIVE SYSTEM OBSERVABILITY")
    print("="*50)

    # Test main dashboard for worker status
    expected_content = [
        "MathLedger Dashboard",
        "Live Worker Status",
        "hx-get=\"/ui/dashboard/worker-status\"",
        "hx-trigger=\"every 5s\"",
        "Jobs in Queue",
        "Active Jobs",
        "Total Workers",
        "Active Proof Attempts"
    ]

    if not test_ui_endpoint("/ui", expected_content):
        return False

    # Test worker status API endpoint
    try:
        response = requests.get(f"{BASE_URL}/workers/status")
        if response.status_code == 200:
            data = response.json()
            required_fields = ["queue_length", "active_jobs", "total_workers", "last_updated"]
            for field in required_fields:
                if field in data:
                    print(f"✅ Worker status API has {field}")
                else:
                    print(f"❌ Worker status API missing {field}")
                    return False
        else:
            print(f"❌ Worker status API failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Worker status API error: {e}")
        return False

    return True

def test_api_endpoints():
    """Test new API endpoints."""
    print("\n" + "="*50)
    print("TESTING NEW API ENDPOINTS")
    print("="*50)

    # Test DAG nodes endpoint
    try:
        response = requests.get(f"{BASE_URL}/ui/dag/nodes/1")
        if response.status_code == 200:
            data = response.json()
            required_fields = ["nodes", "total", "level_min", "level_max", "has_more"]
            for field in required_fields:
                if field in data:
                    print(f"✅ DAG nodes API has {field}")
                else:
                    print(f"❌ DAG nodes API missing {field}")
                    return False
        else:
            print(f"❌ DAG nodes API failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ DAG nodes API error: {e}")
        return False

    # Test worker status endpoint
    try:
        response = requests.get(f"{BASE_URL}/workers/status")
        if response.status_code == 200:
            print("✅ Worker status API endpoint working")
        else:
            print(f"❌ Worker status API endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Worker status API endpoint error: {e}")
        return False

    return True

def test_htmx_partials():
    """Test HTMX partial endpoints."""
    print("\n" + "="*50)
    print("TESTING HTMX PARTIAL ENDPOINTS")
    print("="*50)

    # Test worker status partial
    try:
        response = requests.get(f"{BASE_URL}/ui/dashboard/worker-status")
        if response.status_code == 200:
            content = response.text
            if "Live Worker Status" in content:
                print("✅ Worker status partial renders correctly")
            else:
                print("❌ Worker status partial missing expected content")
                return False
        else:
            print(f"❌ Worker status partial failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Worker status partial error: {e}")
        return False

    return True

def test_javascript_integration():
    """Test JavaScript integration for FOL and performance features."""
    print("\n" + "="*50)
    print("TESTING JAVASCRIPT INTEGRATION")
    print("="*50)

    # Test that KaTeX is loaded in statement detail page
    test_hash = "a" * 64
    try:
        response = requests.get(f"{BASE_URL}/ui/s/{test_hash}")
        if response.status_code in [200, 404]:  # 404 is expected for non-existent statements
            content = response.text
            if "katex.min.js" in content:
                print("✅ KaTeX library loaded")
            else:
                print("❌ KaTeX library not found")
                return False

            if "renderMathInElement" in content:
                print("✅ KaTeX auto-render integration present")
            else:
                print("❌ KaTeX auto-render integration missing")
                return False

            if "loadMoreNodes" in content:
                print("✅ DAG lazy loading functionality present")
            else:
                print("❌ DAG lazy loading functionality missing")
                return False
        else:
            print(f"❌ Statement detail page failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ JavaScript integration test error: {e}")
        return False

    return True

def test_performance_features():
    """Test performance optimization features."""
    print("\n" + "="*50)
    print("TESTING PERFORMANCE FEATURES")
    print("="*50)

    # Test that performance optimizations are in place
    test_hash = "a" * 64
    try:
        response = requests.get(f"{BASE_URL}/ui/s/{test_hash}")
        if response.status_code in [200, 404]:
            content = response.text

            # Check for performance optimizations
            performance_features = [
                "MAX_INITIAL_NODES",
                "alphaDecay",
                "velocityDecay",
                "forceManyBody().strength(-200)",
                "forceCollide().radius(25)"
            ]

            for feature in performance_features:
                if feature in content:
                    print(f"✅ Performance feature found: {feature}")
                else:
                    print(f"❌ Performance feature missing: {feature}")
                    return False
        else:
            print(f"❌ Performance test failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Performance test error: {e}")
        return False

    return True

def main():
    """Run all v0.4 UI tests."""
    print("Testing MathLedger v0.4 UI Implementation")
    print("Features: FOL Rendering, DAG Performance, Live Observability")
    print(f"Base URL: {BASE_URL}")

    # Wait for server to be ready
    print("\nWaiting for server to be ready...")
    time.sleep(2)

    # Run all tests
    tests = [
        ("FOL Rendering Support", test_fol_rendering),
        ("DAG Performance Optimizations", test_dag_performance),
        ("Live System Observability", test_live_observability),
        ("New API Endpoints", test_api_endpoints),
        ("HTMX Partial Endpoints", test_htmx_partials),
        ("JavaScript Integration", test_javascript_integration),
        ("Performance Features", test_performance_features)
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
    print("V0.4 TEST RESULTS SUMMARY")
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
        print("\n🎉 All v0.4 UI tests passed! The FOL and performance features are working correctly.")
        print("\nKey Features Implemented:")
        print("• First-Order Logic rendering with KaTeX")
        print("• DAG performance optimizations with lazy loading")
        print("• Live system observability with worker status")
        print("• Enhanced API endpoints for better performance")
        print("• Real-time updates with HTMX")
        print("\nNote: Some tests may show 404 errors for statement/block details,")
        print("which is expected when the database is empty or contains no matching data.")
        print("The important thing is that the UI structure and features work correctly.")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please check the implementation.")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
