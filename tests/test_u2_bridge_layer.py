# REAL-READY
"""
U2 Bridge Layer Test Suite

This test suite validates:
1. P3 Metric Extractor (Ω, Δp, RSI)
2. Provenance Bundle v2 Generator
3. LeanExecutor stub
4. Canonical sort order for deterministic processing

6 tests total.

Author: Manus-F
Date: 2025-12-06
Status: REAL-READY
"""

import hashlib
import json
import tempfile
from pathlib import Path

from backend.u2.p3_metric_extractor import P3MetricExtractor
from backend.u2.provenance_bundle_v2 import (
    ProvenanceBundleV2Generator,
    SliceMetadata,
)
from backend.u2.lean_executor import LeanExecutor, create_executor
from backend.u2.fosubstrate_executor_skeleton import StatementRecord


# ============================================================================
# TEST 1: CANONICAL SORT ORDER
# ============================================================================

def test_canonical_sort_order():
    """
    Test that canonical sort order is deterministic.
    
    Canonical order: (cycle, worker_id, statement.hash)
    """
    print("\n[TEST 1] Canonical Sort Order")
    
    # Create test events in random order
    events = [
        {"cycle": 2, "worker_id": 1, "data": {"statement": {"hash": "ccc"}}},
        {"cycle": 1, "worker_id": 2, "data": {"statement": {"hash": "bbb"}}},
        {"cycle": 1, "worker_id": 1, "data": {"statement": {"hash": "aaa"}}},
        {"cycle": 2, "worker_id": 1, "data": {"statement": {"hash": "aaa"}}},
        {"cycle": 1, "worker_id": 1, "data": {"statement": {"hash": "bbb"}}},
    ]
    
    # Sort using canonical order
    extractor = P3MetricExtractor()
    sorted_events = extractor._sort_events_canonically(events)
    
    # Expected order
    expected_order = [
        (1, 1, "aaa"),
        (1, 1, "bbb"),
        (1, 2, "bbb"),
        (2, 1, "aaa"),
        (2, 1, "ccc"),
    ]
    
    actual_order = [
        (e["cycle"], e["worker_id"], e["data"]["statement"]["hash"])
        for e in sorted_events
    ]
    
    assert actual_order == expected_order, f"Expected {expected_order}, got {actual_order}"
    print("  ✓ Canonical sort order is deterministic")


# ============================================================================
# TEST 2: Ω (OMEGA) EXTRACTION
# ============================================================================

def test_omega_extraction():
    """
    Test Ω extraction from execution events.
    
    Ω = set of unique proven statement hashes
    """
    print("\n[TEST 2] Ω (Omega) Extraction")
    
    # Create test events
    events = [
        {"data": {"is_tautology": True, "statement": {"hash": "aaa"}}},
        {"data": {"is_tautology": True, "statement": {"hash": "bbb"}}},
        {"data": {"is_tautology": False, "statement": {"hash": "ccc"}}},
        {"data": {"is_tautology": True, "statement": {"hash": "aaa"}}},  # Duplicate
    ]
    
    extractor = P3MetricExtractor()
    omega = extractor._extract_omega(events)
    
    expected_omega = {"aaa", "bbb"}
    assert omega == expected_omega, f"Expected {expected_omega}, got {omega}"
    print(f"  ✓ Ω extracted: {omega}")


# ============================================================================
# TEST 3: Δp (DELTA-P) COMPUTATION
# ============================================================================

def test_delta_p_computation():
    """
    Test Δp computation.
    
    Δp = |Ω| (cardinality of Ω)
    """
    print("\n[TEST 3] Δp (Delta-p) Computation")
    
    # Create test trace file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        f.write(json.dumps({"event_type": "execution", "cycle": 0, "worker_id": 0, "timestamp_ms": 1000, "data": {"is_tautology": True, "statement": {"hash": "aaa"}}}) + "\n")
        f.write(json.dumps({"event_type": "execution", "cycle": 0, "worker_id": 0, "timestamp_ms": 2000, "data": {"is_tautology": True, "statement": {"hash": "bbb"}}}) + "\n")
        f.write(json.dumps({"event_type": "execution", "cycle": 0, "worker_id": 0, "timestamp_ms": 3000, "data": {"is_tautology": False, "statement": {"hash": "ccc"}}}) + "\n")
        trace_path = Path(f.name)
    
    try:
        extractor = P3MetricExtractor()
        metrics = extractor.extract(trace_path)
        
        assert metrics.delta_p == 2, f"Expected Δp=2, got {metrics.delta_p}"
        print(f"  ✓ Δp computed: {metrics.delta_p}")
    finally:
        trace_path.unlink()


# ============================================================================
# TEST 4: RSI (REASONING STEP INTENSITY) COMPUTATION
# ============================================================================

def test_rsi_computation():
    """
    Test RSI computation.
    
    RSI = total_executions / total_wall_time_seconds
    """
    print("\n[TEST 4] RSI (Reasoning Step Intensity) Computation")
    
    # Create test trace file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        # 3 executions over 2 seconds (1000ms to 3000ms)
        f.write(json.dumps({"event_type": "execution", "cycle": 0, "worker_id": 0, "timestamp_ms": 1000, "data": {"is_tautology": True, "statement": {"hash": "aaa"}}}) + "\n")
        f.write(json.dumps({"event_type": "execution", "cycle": 0, "worker_id": 0, "timestamp_ms": 2000, "data": {"is_tautology": True, "statement": {"hash": "bbb"}}}) + "\n")
        f.write(json.dumps({"event_type": "execution", "cycle": 0, "worker_id": 0, "timestamp_ms": 3000, "data": {"is_tautology": False, "statement": {"hash": "ccc"}}}) + "\n")
        trace_path = Path(f.name)
    
    try:
        extractor = P3MetricExtractor()
        metrics = extractor.extract(trace_path)
        
        expected_rsi = 3 / 2.0  # 1.5 executions/second
        assert abs(metrics.rsi - expected_rsi) < 0.01, f"Expected RSI={expected_rsi}, got {metrics.rsi}"
        print(f"  ✓ RSI computed: {metrics.rsi:.2f} executions/second")
    finally:
        trace_path.unlink()


# ============================================================================
# TEST 5: PROVENANCE BUNDLE V2 DUAL-HASH COMMITMENT
# ============================================================================

def test_provenance_bundle_v2_dual_hash():
    """
    Test Provenance Bundle v2 dual-hash commitment.
    
    Dual-hash: content_merkle_root + metadata_hash
    """
    print("\n[TEST 5] Provenance Bundle v2 Dual-Hash Commitment")
    
    # Create temporary artifacts directory
    with tempfile.TemporaryDirectory() as tmpdir:
        artifacts_dir = Path(tmpdir)
        
        # Create dummy trace file
        trace_file = artifacts_dir / "trace.jsonl"
        with open(trace_file, 'w') as f:
            f.write(json.dumps({"event_type": "execution", "cycle": 0, "worker_id": 0, "timestamp_ms": 1000, "data": {"is_tautology": True, "statement": {"hash": "aaa"}}}) + "\n")
        
        # Create slice metadata
        slice_metadata = SliceMetadata(
            slice_name="test_slice",
            master_seed="0xtest",
            total_cycles=10,
            policy_config={"name": "baseline"},
            feature_set_version="v1.0.0",
            executor_config={"name": "propositional"},
            budget_config={"max_time_s": 100},
        )
        
        # Generate bundle
        generator = ProvenanceBundleV2Generator()
        output_path = artifacts_dir / "bundle.json"
        bundle = generator.generate(
            experiment_id="test_exp",
            slice_metadata=slice_metadata,
            artifacts_dir=artifacts_dir,
            output_path=output_path,
        )
        
        # Verify dual-hash commitment
        assert bundle.bundle_header.content_merkle_root is not None
        assert len(bundle.bundle_header.content_merkle_root) == 64  # SHA-256 hex
        assert bundle.bundle_header.metadata_hash is not None
        assert len(bundle.bundle_header.metadata_hash) == 64  # SHA-256 hex
        
        print(f"  ✓ Content Merkle Root: {bundle.bundle_header.content_merkle_root[:16]}...")
        print(f"  ✓ Metadata Hash: {bundle.bundle_header.metadata_hash[:16]}...")


# ============================================================================
# TEST 6: LEAN EXECUTOR STUB
# ============================================================================

def test_lean_executor_stub():
    """
    Test LeanExecutor stub integration.
    """
    print("\n[TEST 6] LeanExecutor Stub")
    
    # Create test statement
    statement = StatementRecord(
        normalized="p→p",
        hash=hashlib.sha256(b"p->p").hexdigest(),
        pretty="p → p",
        rule="test",
        is_axiom=False,
        mp_depth=0,
        parents=(),
        verification_method="lean",
    )
    
    # Create LeanExecutor
    executor = create_executor("lean")
    
    # Verify statement (stub always returns True)
    is_tautology, method = executor.verify(statement)
    
    assert is_tautology == True, f"Expected True, got {is_tautology}"
    assert method == "lean-stub", f"Expected 'lean-stub', got {method}"
    
    print(f"  ✓ LeanExecutor stub verified: {statement.normalized}")
    print(f"  ✓ Verification method: {method}")


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def main():
    """Run all tests."""
    print("=" * 60)
    print("U2 BRIDGE LAYER TEST SUITE")
    print("=" * 60)
    
    tests = [
        test_canonical_sort_order,
        test_omega_extraction,
        test_delta_p_computation,
        test_rsi_computation,
        test_provenance_bundle_v2_dual_hash,
        test_lean_executor_stub,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed}/{len(tests)} tests passed")
    if failed > 0:
        print(f"         {failed}/{len(tests)} tests failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
