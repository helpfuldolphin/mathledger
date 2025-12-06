#!/usr/bin/env python3
"""
Hermetic Build Validation Script

Validates hermetic build properties and generates replay manifest.
Outputs [PASS] NO_NETWORK HERMETIC: TRUE on success.

Usage:
    python scripts/hermetic-validate.py
    python scripts/hermetic-validate.py --full
    python scripts/hermetic-validate.py --generate-manifest
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from backend.testing.hermetic import (
    HermeticPackageManager,
    ExternalAPIMockRegistry,
    ReplayLogComparator,
    HermeticBuildValidator,
    validate_hermetic_build,
    generate_replay_manifest,
)
from backend.testing.no_network import is_no_network_mode


# ============================================================================
# ============================================================================

def log(message: str, level: str = 'INFO'):
    """Log message with timestamp."""
    timestamp = datetime.utcnow().isoformat()
    print(f"[{timestamp}] [{level}] {message}")


# ============================================================================
# ============================================================================

def validate_environment():
    """Validate hermetic environment setup."""
    log("Validating hermetic environment...", "INFO")
    
    issues = []
    
    if not is_no_network_mode():
        issues.append("NO_NETWORK mode not enabled")
    else:
        log("NO_NETWORK mode: enabled", "OK")
    
    if sys.version_info < (3, 8):
        issues.append(f"Python 3.8+ required, found {sys.version}")
    else:
        log(f"Python version: {sys.version.split()[0]}", "OK")
    
    artifacts_dir = REPO_ROOT / 'artifacts' / 'no_network'
    if not artifacts_dir.exists():
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        log(f"Created artifacts directory: {artifacts_dir}", "INFO")
    else:
        log(f"Artifacts directory: {artifacts_dir}", "OK")
    
    if issues:
        log("Environment validation FAILED", "ERROR")
        for issue in issues:
            log(f"  - {issue}", "ERROR")
        return False
    else:
        log("Environment validation PASSED", "OK")
        return True


def setup_hermetic_components():
    """Setup hermetic build components."""
    log("Setting up hermetic components...", "INFO")
    
    pkg_manager = HermeticPackageManager()
    
    core_packages = [
        ('uvicorn', '0.30.0'),
        ('fastapi', '0.115.0'),
        ('redis', '5.0.0'),
        ('psycopg', '3.2.0'),
        ('pydantic', '2.7.0'),
        ('pytest', '8.4.2'),
    ]
    
    for name, version in core_packages:
        pkg_manager.add_package(name, version)
    
    log(f"Added {len(core_packages)} packages to hermetic manifest", "OK")
    
    api_registry = ExternalAPIMockRegistry()
    api_registry.create_pypi_mock()
    api_registry.create_github_mock()
    
    log("Created PyPI and GitHub API mocks", "OK")
    
    return pkg_manager, api_registry


def run_hermetic_validation(full: bool = False):
    """Run hermetic build validation."""
    log("Running hermetic build validation...", "INFO")
    
    pkg_manager, api_registry = setup_hermetic_components()
    
    validator = HermeticBuildValidator()
    results = validator.run_full_validation()
    
    log("=" * 80, "INFO")
    log("HERMETIC BUILD VALIDATION RESULTS", "INFO")
    log("=" * 80, "INFO")
    
    for check_name, check_result in results['checks'].items():
        if isinstance(check_result, bool):
            status = "PASS" if check_result else "FAIL"
            level = "OK" if check_result else "ERROR"
            log(f"{check_name}: {status}", level)
        else:
            log(f"{check_name}: {check_result}", "INFO")
    
    log("=" * 80, "INFO")
    
    is_hermetic = results['hermetic']
    status = results['status']
    
    if is_hermetic:
        log("[PASS] NO_NETWORK HERMETIC: TRUE", "OK")
    else:
        log("[FAIL] NO_NETWORK HERMETIC: FALSE", "ERROR")
    
    return is_hermetic, results


def generate_manifest():
    """Generate replay manifest."""
    log("Generating replay manifest...", "INFO")
    
    manifest_path = REPO_ROOT / 'artifacts' / 'no_network' / 'replay_manifest.json'
    manifest = generate_replay_manifest(str(manifest_path))
    
    log(f"Replay manifest written to: {manifest_path}", "OK")
    log(f"Hermetic status: {manifest['hermetic']}", "INFO")
    
    log("Hermetic components:", "INFO")
    for component, config in manifest['components'].items():
        enabled = config.get('enabled', False)
        status = "enabled" if enabled else "disabled"
        log(f"  - {component}: {status}", "INFO")
    
    return manifest


def run_replay_comparison_tests():
    """Run replay log comparison tests."""
    log("Running replay log comparison tests...", "TEST")
    
    comparator = ReplayLogComparator()
    
    log("Test 1: Identical operations", "TEST")
    comparator.record_operation('test_op', {'x': 1}, {'y': 2}, 10.0)
    comparator.save_log('test1_run1')
    
    comparator.current_log = []
    comparator.record_operation('test_op', {'x': 1}, {'y': 2}, 10.0)
    comparator.save_log('test1_run2')
    
    is_identical, diffs = comparator.compare_logs('test1_run1', 'test1_run2')
    if is_identical:
        log("Test 1: PASS - Operations are deterministic", "OK")
    else:
        log(f"Test 1: FAIL - {len(diffs)} differences found", "ERROR")
        for diff in diffs:
            log(f"  - {diff}", "ERROR")
    
    log("Test 2: Non-deterministic detection", "TEST")
    comparator.current_log = []
    comparator.record_operation('test_op', {'x': 1}, {'y': 2}, 10.0)
    comparator.save_log('test2_run1')
    
    comparator.current_log = []
    comparator.record_operation('test_op', {'x': 1}, {'y': 3}, 10.0)  # Different output
    comparator.save_log('test2_run2')
    
    is_identical, diffs = comparator.compare_logs('test2_run1', 'test2_run2')
    if not is_identical:
        log("Test 2: PASS - Non-determinism detected correctly", "OK")
    else:
        log("Test 2: FAIL - Should have detected differences", "ERROR")
    
    log("Replay comparison tests complete", "INFO")


# ============================================================================
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Hermetic Build Validation - Validate deterministic builds',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/hermetic-validate.py
  
  python scripts/hermetic-validate.py --full
  
  python scripts/hermetic-validate.py --generate-manifest
        """
    )
    
    parser.add_argument(
        '--full',
        action='store_true',
        help='Run full validation including replay comparison tests'
    )
    
    parser.add_argument(
        '--generate-manifest',
        action='store_true',
        help='Generate replay manifest'
    )
    
    args = parser.parse_args()
    
    log("=" * 80, "INFO")
    log("MathLedger Hermetic Build Validation", "INFO")
    log("=" * 80, "INFO")
    
    if not validate_environment():
        sys.exit(1)
    
    if args.generate_manifest:
        generate_manifest()
        sys.exit(0)
    
    is_hermetic, results = run_hermetic_validation(full=args.full)
    
    if args.full:
        run_replay_comparison_tests()
    
    if is_hermetic:
        generate_manifest()
    
    sys.exit(0 if is_hermetic else 1)


if __name__ == '__main__':
    main()
