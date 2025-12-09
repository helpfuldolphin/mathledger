"""
Component Benchmarks

Measures performance of Merkle tree construction, block sealing, and verification.

Author: Manus-H
"""

import time
from dataclasses import dataclass
from typing import List
import statistics

from basis.crypto.hash_versioned import merkle_root_versioned, compute_dual_commitment
from basis.ledger.block_pq import BlockHeaderPQ, seal_block_dual
from backend.consensus_pq.validation import validate_block_full


@dataclass
class ComponentBenchmarkResult:
    """
    Result of a component benchmark.
    
    Attributes:
        component_name: Name of component tested
        operation: Operation performed (e.g., "merkle_tree", "block_seal")
        algorithm_name: Hash algorithm used
        statement_count: Number of statements processed
        iterations: Number of iterations performed
        mean_time_ms: Mean time per operation in milliseconds
        median_time_ms: Median time per operation in milliseconds
        p90_time_ms: 90th percentile time in milliseconds
        p95_time_ms: 95th percentile time in milliseconds
        overhead_percent: Overhead compared to baseline (if applicable)
    """
    
    component_name: str
    operation: str
    algorithm_name: str
    statement_count: int
    iterations: int
    mean_time_ms: float
    median_time_ms: float
    p90_time_ms: float
    p95_time_ms: float
    overhead_percent: float = 0.0


def benchmark_merkle_tree(
    statements: List[str],
    algorithm_id: int,
    algorithm_name: str,
    iterations: int = 100,
) -> ComponentBenchmarkResult:
    """
    Benchmark Merkle tree construction.
    
    Args:
        statements: List of statements to include in tree
        algorithm_id: Hash algorithm ID
        algorithm_name: Hash algorithm name
        iterations: Number of iterations
        
    Returns:
        ComponentBenchmarkResult
    """
    times = []
    
    for _ in range(iterations):
        start = time.perf_counter()
        _ = merkle_root_versioned(statements, algorithm_id=algorithm_id)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
    
    mean_ms = statistics.mean(times)
    median_ms = statistics.median(times)
    sorted_times = sorted(times)
    p90_ms = sorted_times[int(len(sorted_times) * 0.90)]
    p95_ms = sorted_times[int(len(sorted_times) * 0.95)]
    
    return ComponentBenchmarkResult(
        component_name="Merkle Tree",
        operation="merkle_root_construction",
        algorithm_name=algorithm_name,
        statement_count=len(statements),
        iterations=iterations,
        mean_time_ms=mean_ms,
        median_time_ms=median_ms,
        p90_time_ms=p90_ms,
        p95_time_ms=p95_ms,
    )


def benchmark_dual_merkle_tree(
    statements: List[str],
    iterations: int = 100,
) -> ComponentBenchmarkResult:
    """
    Benchmark dual Merkle tree construction (legacy + PQ).
    
    Args:
        statements: List of statements to include in tree
        iterations: Number of iterations
        
    Returns:
        ComponentBenchmarkResult
    """
    times = []
    
    for _ in range(iterations):
        start = time.perf_counter()
        
        # Compute both Merkle roots
        _ = merkle_root_versioned(statements, algorithm_id=0x00)  # SHA-256
        _ = merkle_root_versioned(statements, algorithm_id=0x01)  # SHA3-256
        
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
    
    mean_ms = statistics.mean(times)
    median_ms = statistics.median(times)
    sorted_times = sorted(times)
    p90_ms = sorted_times[int(len(sorted_times) * 0.90)]
    p95_ms = sorted_times[int(len(sorted_times) * 0.95)]
    
    return ComponentBenchmarkResult(
        component_name="Dual Merkle Tree",
        operation="dual_merkle_root_construction",
        algorithm_name="SHA-256 + SHA3-256",
        statement_count=len(statements),
        iterations=iterations,
        mean_time_ms=mean_ms,
        median_time_ms=median_ms,
        p90_time_ms=p90_ms,
        p95_time_ms=p95_ms,
    )


def benchmark_dual_commitment(
    statements: List[str],
    iterations: int = 100,
) -> ComponentBenchmarkResult:
    """
    Benchmark dual commitment computation.
    
    Args:
        statements: List of statements
        iterations: Number of iterations
        
    Returns:
        ComponentBenchmarkResult
    """
    # Pre-compute Merkle roots
    legacy_root = merkle_root_versioned(statements, algorithm_id=0x00)
    pq_root = merkle_root_versioned(statements, algorithm_id=0x01)
    
    times = []
    
    for _ in range(iterations):
        start = time.perf_counter()
        _ = compute_dual_commitment(
            legacy_hash=legacy_root,
            pq_hash=pq_root,
            pq_algorithm_id=0x01,
        )
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
    
    mean_ms = statistics.mean(times)
    median_ms = statistics.median(times)
    sorted_times = sorted(times)
    p90_ms = sorted_times[int(len(sorted_times) * 0.90)]
    p95_ms = sorted_times[int(len(sorted_times) * 0.95)]
    
    return ComponentBenchmarkResult(
        component_name="Dual Commitment",
        operation="dual_commitment_computation",
        algorithm_name="SHA-256",
        statement_count=len(statements),
        iterations=iterations,
        mean_time_ms=mean_ms,
        median_time_ms=median_ms,
        p90_time_ms=p90_ms,
        p95_time_ms=p95_ms,
    )


def benchmark_block_seal_legacy(
    statements: List[str],
    prev_hash: str,
    block_number: int,
    iterations: int = 100,
) -> ComponentBenchmarkResult:
    """
    Benchmark legacy block sealing (SHA-256 only).
    
    Args:
        statements: List of statements
        prev_hash: Previous block hash
        block_number: Block number
        iterations: Number of iterations
        
    Returns:
        ComponentBenchmarkResult
    """
    times = []
    
    for _ in range(iterations):
        start = time.perf_counter()
        
        # Compute Merkle root
        merkle_root = merkle_root_versioned(statements, algorithm_id=0x00)
        
        # Create block header
        _ = BlockHeaderPQ(
            block_number=block_number,
            prev_hash=prev_hash,
            merkle_root=merkle_root,
            timestamp=time.time(),
            statements=statements,
        )
        
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
    
    mean_ms = statistics.mean(times)
    median_ms = statistics.median(times)
    sorted_times = sorted(times)
    p90_ms = sorted_times[int(len(sorted_times) * 0.90)]
    p95_ms = sorted_times[int(len(sorted_times) * 0.95)]
    
    return ComponentBenchmarkResult(
        component_name="Block Seal",
        operation="legacy_block_seal",
        algorithm_name="SHA-256",
        statement_count=len(statements),
        iterations=iterations,
        mean_time_ms=mean_ms,
        median_time_ms=median_ms,
        p90_time_ms=p90_ms,
        p95_time_ms=p95_ms,
    )


def benchmark_block_seal_dual(
    statements: List[str],
    prev_hash: str,
    pq_prev_hash: str,
    block_number: int,
    iterations: int = 100,
) -> ComponentBenchmarkResult:
    """
    Benchmark dual block sealing (SHA-256 + SHA3-256).
    
    Args:
        statements: List of statements
        prev_hash: Previous block hash (legacy)
        pq_prev_hash: Previous block hash (PQ)
        block_number: Block number
        iterations: Number of iterations
        
    Returns:
        ComponentBenchmarkResult
    """
    times = []
    
    for _ in range(iterations):
        start = time.perf_counter()
        
        _ = seal_block_dual(
            statements=statements,
            prev_hash=prev_hash,
            pq_prev_hash=pq_prev_hash,
            block_number=block_number,
            timestamp=time.time(),
            pq_algorithm_id=0x01,
        )
        
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
    
    mean_ms = statistics.mean(times)
    median_ms = statistics.median(times)
    sorted_times = sorted(times)
    p90_ms = sorted_times[int(len(sorted_times) * 0.90)]
    p95_ms = sorted_times[int(len(sorted_times) * 0.95)]
    
    return ComponentBenchmarkResult(
        component_name="Block Seal",
        operation="dual_block_seal",
        algorithm_name="SHA-256 + SHA3-256",
        statement_count=len(statements),
        iterations=iterations,
        mean_time_ms=mean_ms,
        median_time_ms=median_ms,
        p90_time_ms=p90_ms,
        p95_time_ms=p95_ms,
    )


def benchmark_block_validation(
    block: BlockHeaderPQ,
    prev_block: BlockHeaderPQ,
    iterations: int = 100,
) -> ComponentBenchmarkResult:
    """
    Benchmark full block validation.
    
    Args:
        block: Block to validate
        prev_block: Previous block
        iterations: Number of iterations
        
    Returns:
        ComponentBenchmarkResult
    """
    times = []
    
    for _ in range(iterations):
        start = time.perf_counter()
        _ = validate_block_full(block, prev_block)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
    
    mean_ms = statistics.mean(times)
    median_ms = statistics.median(times)
    sorted_times = sorted(times)
    p90_ms = sorted_times[int(len(sorted_times) * 0.90)]
    p95_ms = sorted_times[int(len(sorted_times) * 0.95)]
    
    algorithm_name = "Dual" if block.has_dual_commitment() else "Legacy"
    
    return ComponentBenchmarkResult(
        component_name="Block Validation",
        operation="full_block_validation",
        algorithm_name=algorithm_name,
        statement_count=len(block.statements),
        iterations=iterations,
        mean_time_ms=mean_ms,
        median_time_ms=median_ms,
        p90_time_ms=p90_ms,
        p95_time_ms=p95_ms,
    )


def run_component_benchmarks() -> List[ComponentBenchmarkResult]:
    """
    Run comprehensive component benchmarks.
    
    Returns:
        List of ComponentBenchmarkResult objects
    """
    results = []
    
    print("Running component benchmarks...")
    print("=" * 80)
    
    # Test with different statement counts
    statement_counts = [10, 50, 100, 500, 1000]
    
    for count in statement_counts:
        print(f"\nStatement count: {count}")
        
        # Generate test statements
        statements = [f"Statement {i}: Test data" for i in range(count)]
        
        # Benchmark Merkle tree (SHA-256)
        merkle_sha256 = benchmark_merkle_tree(statements, 0x00, "SHA-256")
        results.append(merkle_sha256)
        print(f"  Merkle (SHA-256): {merkle_sha256.mean_time_ms:.2f} ms")
        
        # Benchmark Merkle tree (SHA3-256)
        merkle_sha3 = benchmark_merkle_tree(statements, 0x01, "SHA3-256")
        results.append(merkle_sha3)
        print(f"  Merkle (SHA3-256): {merkle_sha3.mean_time_ms:.2f} ms")
        
        # Benchmark dual Merkle tree
        merkle_dual = benchmark_dual_merkle_tree(statements)
        results.append(merkle_dual)
        overhead = (merkle_dual.mean_time_ms / merkle_sha256.mean_time_ms - 1) * 100
        merkle_dual.overhead_percent = overhead
        print(f"  Merkle (Dual): {merkle_dual.mean_time_ms:.2f} ms ({overhead:.1f}% overhead)")
        
        # Benchmark dual commitment
        commitment = benchmark_dual_commitment(statements)
        results.append(commitment)
        print(f"  Dual Commitment: {commitment.mean_time_ms:.2f} ms")
        
        # Benchmark block sealing
        prev_hash = "0x" + "00" * 32
        pq_prev_hash = "0x" + "00" * 32
        
        seal_legacy = benchmark_block_seal_legacy(statements, prev_hash, 1)
        results.append(seal_legacy)
        print(f"  Block Seal (Legacy): {seal_legacy.mean_time_ms:.2f} ms")
        
        seal_dual = benchmark_block_seal_dual(statements, prev_hash, pq_prev_hash, 1)
        results.append(seal_dual)
        seal_overhead = (seal_dual.mean_time_ms / seal_legacy.mean_time_ms - 1) * 100
        seal_dual.overhead_percent = seal_overhead
        print(f"  Block Seal (Dual): {seal_dual.mean_time_ms:.2f} ms ({seal_overhead:.1f}% overhead)")
    
    print("\n" + "=" * 80)
    print("Component benchmarks complete")
    
    return results


def export_results_csv(results: List[ComponentBenchmarkResult], filename: str) -> None:
    """Export results to CSV."""
    import csv
    
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Component",
            "Operation",
            "Algorithm",
            "Statement Count",
            "Iterations",
            "Mean (ms)",
            "Median (ms)",
            "P90 (ms)",
            "P95 (ms)",
            "Overhead (%)",
        ])
        
        for result in results:
            writer.writerow([
                result.component_name,
                result.operation,
                result.algorithm_name,
                result.statement_count,
                result.iterations,
                result.mean_time_ms,
                result.median_time_ms,
                result.p90_time_ms,
                result.p95_time_ms,
                result.overhead_percent,
            ])
    
    print(f"\nResults exported to {filename}")


if __name__ == "__main__":
    results = run_component_benchmarks()
    export_results_csv(results, "component_benchmark_results.csv")
