#!/usr/bin/env python3
"""
Hermetic Verifier v2 Validation Script

Validates full hermetic reproducibility across all lanes with:
- RFC 8785 canonical JSON serialization
- Byte-identical replay log comparison
- Multi-lane hermetic validation
- Fleet state archival on [PASS] ALL BLUE

Outputs: [PASS] NO_NETWORK HERMETIC v2 TRUE

Usage:
    python scripts/hermetic-validate-v2.py
    python scripts/hermetic-validate-v2.py --full
    python scripts/hermetic-validate-v2.py --archive
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from backend.testing.hermetic_v2 import (
    RFC8785Canonicalizer,
    ByteIdenticalComparator,
    MultiLaneValidator,
    FleetStateArchiver,
    HermeticVerifierV2,
    validate_hermetic_v2,
    generate_replay_manifest_v2,
)
from backend.testing.no_network import is_no_network_mode


# ============================================================================
# ============================================================================

def log(message: str, level: str = 'INFO'):
    """Log message with timestamp (ASCII-only)."""
    timestamp = datetime.utcnow().isoformat()
    message_ascii = message.encode('ascii', errors='replace').decode('ascii')
    print(f"[{timestamp}] [{level}] {message_ascii}")


# ============================================================================
# ============================================================================

def validate_environment():
    """Validate hermetic v2 environment setup."""
    log("Validating hermetic v2 environment...", "INFO")
    
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
    
    allblue_dir = REPO_ROOT / 'artifacts' / 'allblue'
    if not allblue_dir.exists():
        allblue_dir.mkdir(parents=True, exist_ok=True)
        log(f"Created allblue directory: {allblue_dir}", "INFO")
    else:
        log(f"Allblue directory: {allblue_dir}", "OK")
    
    if issues:
        log("Environment validation FAILED", "ERROR")
        for issue in issues:
            log(f"  - {issue}", "ERROR")
        return False
    else:
        log("Environment validation PASSED", "OK")
        return True


def test_rfc8785_canonicalization():
    """Test RFC 8785 canonical JSON serialization."""
    log("Testing RFC 8785 canonicalization...", "TEST")
    
    data1 = {'b': 2, 'a': 1}
    canonical1 = RFC8785Canonicalizer.canonicalize_str(data1)
    expected1 = '{"a":1,"b":2}'
    
    if canonical1 == expected1:
        log("Test 1: PASS - Key ordering", "OK")
    else:
        log(f"Test 1: FAIL - Expected {expected1}, got {canonical1}", "ERROR")
        return False
    
    data2 = {'z': {'y': 3, 'x': 2}, 'a': 1}
    canonical2 = RFC8785Canonicalizer.canonicalize_str(data2)
    expected2 = '{"a":1,"z":{"x":2,"y":3}}'
    
    if canonical2 == expected2:
        log("Test 2: PASS - Nested key ordering", "OK")
    else:
        log(f"Test 2: FAIL - Expected {expected2}, got {canonical2}", "ERROR")
        return False
    
    hash1 = RFC8785Canonicalizer.hash_canonical(data1)
    hash2 = RFC8785Canonicalizer.hash_canonical({'a': 1, 'b': 2})
    
    if hash1 == hash2:
        log("Test 3: PASS - Hash consistency", "OK")
    else:
        log(f"Test 3: FAIL - Hashes differ: {hash1} vs {hash2}", "ERROR")
        return False
    
    log("RFC 8785 canonicalization: PASS", "OK")
    return True


def test_byte_identical_comparison():
    """Test byte-identical replay log comparison."""
    log("Testing byte-identical comparison...", "TEST")
    
    comparator = ByteIdenticalComparator()
    
    log("Test 1: Identical operations", "TEST")
    comparator.record_operation('test_op', {'x': 1}, {'y': 2}, 10.0)
    comparator.save_log('v2_test1_run1')
    
    comparator.current_log = []
    comparator.record_operation('test_op', {'x': 1}, {'y': 2}, 10.0)
    comparator.save_log('v2_test1_run2')
    
    is_identical, diffs = comparator.compare_byte_identical('v2_test1_run1', 'v2_test1_run2')
    if is_identical:
        log("Test 1: PASS - Byte-identical operations", "OK")
    else:
        log(f"Test 1: FAIL - {len(diffs)} differences found", "ERROR")
        for diff in diffs:
            log(f"  - {diff}", "ERROR")
        return False
    
    log("Test 2: Non-identical detection", "TEST")
    comparator.current_log = []
    comparator.record_operation('test_op', {'x': 1}, {'y': 2}, 10.0)
    comparator.save_log('v2_test2_run1')
    
    comparator.current_log = []
    comparator.record_operation('test_op', {'x': 1}, {'y': 3}, 10.0)  # Different output
    comparator.save_log('v2_test2_run2')
    
    is_identical, diffs = comparator.compare_byte_identical('v2_test2_run1', 'v2_test2_run2')
    if not is_identical:
        log("Test 2: PASS - Non-identical detected correctly", "OK")
    else:
        log("Test 2: FAIL - Should have detected differences", "ERROR")
        return False
    
    log("Byte-identical comparison: PASS", "OK")
    return True


def test_multi_lane_validation():
    """Test multi-lane hermetic validation."""
    log("Testing multi-lane validation...", "TEST")
    
    validator = MultiLaneValidator()
    
    lane_results = validator.validate_all_lanes()
    
    if len(lane_results) >= 6:
        log(f"Test 1: PASS - Validated {len(lane_results)} lanes", "OK")
    else:
        log(f"Test 1: FAIL - Expected 6+ lanes, got {len(lane_results)}", "ERROR")
        return False
    
    if validator.all_lanes_hermetic():
        log("Test 2: PASS - All lanes hermetic", "OK")
    else:
        log("Test 2: FAIL - Not all lanes hermetic", "ERROR")
        return False
    
    report = validator.generate_lane_report()
    if 'overall_canonical_hash' in report:
        log("Test 3: PASS - Report generated with canonical hash", "OK")
    else:
        log("Test 3: FAIL - Missing canonical hash in report", "ERROR")
        return False
    
    log("Multi-lane validation: PASS", "OK")
    return True


def test_fleet_state_archival():
    """Test fleet state archival."""
    log("Testing fleet state archival...", "TEST")
    
    archiver = FleetStateArchiver()
    validator = MultiLaneValidator()
    
    lane_results = validator.validate_all_lanes()
    
    all_blue = archiver.detect_all_blue(lane_results)
    if all_blue:
        log("Test 1: PASS - ALL BLUE detected", "OK")
    else:
        log("Test 1: FAIL - ALL BLUE not detected", "ERROR")
        return False
    
    fleet_state = archiver.freeze_fleet_state(
        lane_results,
        {'test_log': 'abc123'},
        {'test': True}
    )
    
    if 'state_hash' in fleet_state:
        log("Test 2: PASS - Fleet state frozen with hash", "OK")
    else:
        log("Test 2: FAIL - Missing state hash", "ERROR")
        return False
    
    archive_path = archiver.sign_and_archive(fleet_state)
    if Path(archive_path).exists():
        log(f"Test 3: PASS - State archived at {archive_path}", "OK")
    else:
        log("Test 3: FAIL - Archive file not created", "ERROR")
        return False
    
    is_valid, message = archiver.verify_archived_state(archive_path)
    if is_valid:
        log(f"Test 4: PASS - {message}", "OK")
    else:
        log(f"Test 4: FAIL - {message}", "ERROR")
        return False
    
    log("Fleet state archival: PASS", "OK")
    return True


def run_hermetic_v2_validation(full: bool = False):
    """Run hermetic v2 validation."""
    log("Running hermetic v2 validation...", "INFO")
    
    verifier = HermeticVerifierV2()
    results = verifier.run_full_verification()
    
    log("=" * 80, "INFO")
    log("HERMETIC VERIFIER V2 VALIDATION RESULTS", "INFO")
    log("=" * 80, "INFO")
    
    for check_name, check_result in results['checks'].items():
        if isinstance(check_result, bool):
            status = "PASS" if check_result else "FAIL"
            level = "OK" if check_result else "ERROR"
            log(f"{check_name}: {status}", level)
        else:
            log(f"{check_name}: {check_result}", "INFO")
    
    log("=" * 80, "INFO")
    
    is_hermetic_v2 = results['hermetic_v2']
    status = results['status']
    
    if is_hermetic_v2:
        log("[PASS] NO_NETWORK HERMETIC v2 TRUE", "OK")
    else:
        log("[FAIL] NO_NETWORK HERMETIC v2 FALSE", "ERROR")
    
    if results['checks'].get('all_blue'):
        log("ALL BLUE detected - archiving fleet state...", "INFO")
        archive_path = verifier.archive_on_all_blue(
            replay_logs={'validation': results['lane_report']['overall_canonical_hash']},
            metadata={'validation_timestamp': results['timestamp']}
        )
        if archive_path:
            log(f"Fleet state archived: {archive_path}", "OK")
            log("Verifiable cognition chain advanced", "INFO")
    
    return is_hermetic_v2, results


def generate_manifest():
    """Generate replay manifest v2."""
    log("Generating replay manifest v2...", "INFO")
    
    manifest_path = REPO_ROOT / 'artifacts' / 'no_network' / 'replay_manifest_v2.json'
    manifest = generate_replay_manifest_v2(str(manifest_path))
    
    log(f"Replay manifest v2 written to: {manifest_path}", "OK")
    log(f"Hermetic v2 status: {manifest['hermetic_v2']}", "INFO")
    log(f"RFC 8785 canonical: {manifest['rfc8785_canonical']}", "INFO")
    
    log("Hermetic v2 components:", "INFO")
    for component, config in manifest['components'].items():
        enabled = config.get('enabled', False)
        status = "enabled" if enabled else "disabled"
        log(f"  - {component}: {status}", "INFO")
    
    return manifest


# ============================================================================
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Hermetic Verifier v2 - Full Reproducibility Validation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/hermetic-validate-v2.py
  
  python scripts/hermetic-validate-v2.py --full
  
  python scripts/hermetic-validate-v2.py --generate-manifest
  
  python scripts/hermetic-validate-v2.py --archive
        """
    )
    
    parser.add_argument(
        '--full',
        action='store_true',
        help='Run full validation including all component tests'
    )
    
    parser.add_argument(
        '--generate-manifest',
        action='store_true',
        help='Generate replay manifest v2'
    )
    
    parser.add_argument(
        '--archive',
        action='store_true',
        help='Archive fleet state on ALL BLUE'
    )
    
    args = parser.parse_args()
    
    log("=" * 80, "INFO")
    log("MathLedger Hermetic Verifier v2", "INFO")
    log("=" * 80, "INFO")
    
    if not validate_environment():
        sys.exit(1)
    
    if args.generate_manifest:
        generate_manifest()
        sys.exit(0)
    
    if args.full:
        log("Running full component tests...", "INFO")
        
        tests = [
            ("RFC 8785 Canonicalization", test_rfc8785_canonicalization),
            ("Byte-Identical Comparison", test_byte_identical_comparison),
            ("Multi-Lane Validation", test_multi_lane_validation),
            ("Fleet State Archival", test_fleet_state_archival),
        ]
        
        all_passed = True
        for test_name, test_func in tests:
            log(f"Running {test_name}...", "TEST")
            if not test_func():
                all_passed = False
                log(f"{test_name}: FAILED", "ERROR")
            else:
                log(f"{test_name}: PASSED", "OK")
        
        if not all_passed:
            log("Some component tests failed", "ERROR")
            sys.exit(1)
    
    is_hermetic_v2, results = run_hermetic_v2_validation(full=args.full)
    
    if is_hermetic_v2:
        generate_manifest()
    
    sys.exit(0 if is_hermetic_v2 else 1)


if __name__ == '__main__':
    main()
