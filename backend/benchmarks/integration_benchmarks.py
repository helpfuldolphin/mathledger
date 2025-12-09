"""
Integration Benchmarks

Measures end-to-end block validation and chain synchronization performance.

Author: Manus-H
"""

import time
from dataclasses import dataclass
from typing import List
import statistics

from basis.ledger.block_pq import BlockHeaderPQ, seal_block_dual
from backend.consensus_pq.validation import validate_block_full, validate_block_batch
from backend.consensus_pq.epoch import register_epoch, HashEpoch, initialize_genesis_epoch
from backend.consensus_pq.rules import ConsensusRuleVersion


@dataclass
class IntegrationBenchmarkResult:
    """
    Result of an integration benchmark.
    
    Attributes:
        scenario_name: Name of the test scenario
        block_count: Number of blocks processed
        statement_count_per_block: Number of statements per block
        total_validation_time_ms: Total time for all validations
        mean_block_time_ms: Mean time per block
        median_block_time_ms: Median time per block
        p90_block_time_ms: 90th percentile time
        p95_block_time_ms: 95th percentile time
        throughput_blocks_per_second: Blocks validated per second
    """
    
    scenario_name: str
    block_count: int
    statement_count_per_block: int
    total_validation_time_ms: float
    mean_block_time_ms: float
    median_block_time_ms: float
    p90_block_time_ms: float
    p95_block_time_ms: float
    throughput_blocks_per_second: float


def create_test_chain(
    block_count: int,
    statements_per_block: int,
    use_dual_commitment: bool = False,
) -> List[BlockHeaderPQ]:
    """
    Create a test blockchain for benchmarking.
    
    Args:
        block_count: Number of blocks to create
        statements_per_block: Number of statements per block
        use_dual_commitment: Whether to use dual-commitment blocks
        
    Returns:
        List of BlockHeaderPQ objects
    """
    blocks = []
    prev_hash = "0x" + "00" * 32
    pq_prev_hash = "0x" + "00" * 32
    
    for i in range(block_count):
        statements = [f"stmt_{i}_{j}" for j in range(statements_per_block)]
        
        if use_dual_commitment:
            block = seal_block_dual(
                statements=statements,
                prev_hash=prev_hash,
                pq_prev_hash=pq_prev_hash,
                block_number=i,
                timestamp=time.time() + i,
                pq_algorithm_id=0x01,
            )
            prev_hash = block.merkle_root
            pq_prev_hash = block.pq_merkle_root
        else:
            block = BlockHeaderPQ(
                block_number=i,
                prev_hash=prev_hash,
                merkle_root=f"0x{i:064x}",
                timestamp=time.time() + i,
                statements=statements,
            )
            prev_hash = block.merkle_root
        
        blocks.append(block)
    
    return blocks


def benchmark_full_block_validation(
    block_count: int,
    statements_per_block: int,
    use_dual_commitment: bool,
    iterations: int = 10,
) -> IntegrationBenchmarkResult:
    """
    Benchmark full block validation.
    
    Args:
        block_count: Number of blocks to validate
        statements_per_block: Number of statements per block
        use_dual_commitment: Whether blocks use dual commitment
        iterations: Number of iterations to run
        
    Returns:
        IntegrationBenchmarkResult
    """
    # Create test chain
    blocks = create_test_chain(block_count, statements_per_block, use_dual_commitment)
    
    # Benchmark validation
    times = []
    
    for _ in range(iterations):
        start = time.perf_counter()
        
        # Validate each block
        for i, block in enumerate(blocks):
            prev_block = blocks[i - 1] if i > 0 else None
            _ = validate_block_full(block, prev_block)
        
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
    
    # Compute statistics
    total_time_ms = statistics.mean(times)
    mean_block_ms = total_time_ms / block_count
    
    # Compute per-block times for percentiles
    per_block_times = []
    for _ in range(iterations):
        for i, block in enumerate(blocks):
            prev_block = blocks[i - 1] if i > 0 else None
            start = time.perf_counter()
            _ = validate_block_full(block, prev_block)
            end = time.perf_counter()
            per_block_times.append((end - start) * 1000)
    
    median_block_ms = statistics.median(per_block_times)
    sorted_times = sorted(per_block_times)
    p90_block_ms = sorted_times[int(len(sorted_times) * 0.90)]
    p95_block_ms = sorted_times[int(len(sorted_times) * 0.95)]
    
    throughput = (block_count * iterations) / (sum(times) / 1000)
    
    scenario_name = f"Full Validation ({'Dual' if use_dual_commitment else 'Legacy'})"
    
    return IntegrationBenchmarkResult(
        scenario_name=scenario_name,
        block_count=block_count,
        statement_count_per_block=statements_per_block,
        total_validation_time_ms=total_time_ms,
        mean_block_time_ms=mean_block_ms,
        median_block_time_ms=median_block_ms,
        p90_block_time_ms=p90_block_ms,
        p95_block_time_ms=p95_block_ms,
        throughput_blocks_per_second=throughput,
    )


def benchmark_chain_validation(
    chain_length: int,
    statements_per_block: int,
    use_dual_commitment: bool,
    iterations: int = 5,
) -> IntegrationBenchmarkResult:
    """
    Benchmark batch chain validation.
    
    Args:
        chain_length: Length of chain to validate
        statements_per_block: Number of statements per block
        use_dual_commitment: Whether blocks use dual commitment
        iterations: Number of iterations to run
        
    Returns:
        IntegrationBenchmarkResult
    """
    # Create test chain
    blocks = create_test_chain(chain_length, statements_per_block, use_dual_commitment)
    
    # Benchmark batch validation
    times = []
    
    for _ in range(iterations):
        start = time.perf_counter()
        _ = validate_block_batch(blocks)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
    
    # Compute statistics
    total_time_ms = statistics.mean(times)
    mean_block_ms = total_time_ms / chain_length
    median_time_ms = statistics.median(times)
    
    sorted_times = sorted(times)
    p90_time_ms = sorted_times[int(len(sorted_times) * 0.90)]
    p95_time_ms = sorted_times[int(len(sorted_times) * 0.95)]
    
    throughput = (chain_length * iterations) / (sum(times) / 1000)
    
    scenario_name = f"Chain Validation ({'Dual' if use_dual_commitment else 'Legacy'})"
    
    return IntegrationBenchmarkResult(
        scenario_name=scenario_name,
        block_count=chain_length,
        statement_count_per_block=statements_per_block,
        total_validation_time_ms=total_time_ms,
        mean_block_time_ms=mean_block_ms,
        median_block_time_ms=median_time_ms / chain_length,
        p90_block_time_ms=p90_time_ms / chain_length,
        p95_block_time_ms=p95_time_ms / chain_length,
        throughput_blocks_per_second=throughput,
    )


def benchmark_historical_verification(
    epoch_count: int,
    blocks_per_epoch: int,
    statements_per_block: int,
    iterations: int = 3,
) -> IntegrationBenchmarkResult:
    """
    Benchmark historical verification across multiple epochs.
    
    Args:
        epoch_count: Number of epochs to create
        blocks_per_epoch: Number of blocks per epoch
        statements_per_block: Number of statements per block
        iterations: Number of iterations to run
        
    Returns:
        IntegrationBenchmarkResult
    """
    # Initialize epochs
    initialize_genesis_epoch()
    
    # Register multiple epochs
    for i in range(1, epoch_count):
        epoch = HashEpoch(
            start_block=i * blocks_per_epoch,
            end_block=(i + 1) * blocks_per_epoch - 1 if i < epoch_count - 1 else None,
            algorithm_id=0x01 if i % 2 == 1 else 0x00,
            algorithm_name=f"Algorithm-{i}",
            rule_version=ConsensusRuleVersion.V2_DUAL_REQUIRED.value,
            activation_timestamp=time.time(),
            governance_hash=f"0x{i:064x}",
        )
        register_epoch(epoch)
    
    # Create blocks across all epochs
    total_blocks = epoch_count * blocks_per_epoch
    blocks = create_test_chain(total_blocks, statements_per_block, use_dual_commitment=True)
    
    # Benchmark historical verification
    times = []
    
    for _ in range(iterations):
        start = time.perf_counter()
        
        # Verify all blocks
        for i, block in enumerate(blocks):
            prev_block = blocks[i - 1] if i > 0 else None
            _ = validate_block_full(block, prev_block)
        
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
    
    # Compute statistics
    total_time_ms = statistics.mean(times)
    mean_block_ms = total_time_ms / total_blocks
    median_time_ms = statistics.median(times)
    
    sorted_times = sorted(times)
    p90_time_ms = sorted_times[int(len(sorted_times) * 0.90)]
    p95_time_ms = sorted_times[int(len(sorted_times) * 0.95)]
    
    throughput = (total_blocks * iterations) / (sum(times) / 1000)
    
    return IntegrationBenchmarkResult(
        scenario_name=f"Historical Verification ({epoch_count} epochs)",
        block_count=total_blocks,
        statement_count_per_block=statements_per_block,
        total_validation_time_ms=total_time_ms,
        mean_block_time_ms=mean_block_ms,
        median_block_time_ms=median_time_ms / total_blocks,
        p90_block_time_ms=p90_time_ms / total_blocks,
        p95_block_time_ms=p95_time_ms / total_blocks,
        throughput_blocks_per_second=throughput,
    )


def benchmark_epoch_transition(
    blocks_before: int,
    blocks_after: int,
    statements_per_block: int,
    iterations: int = 10,
) -> IntegrationBenchmarkResult:
    """
    Benchmark validation across epoch transition.
    
    Args:
        blocks_before: Number of blocks before transition
        blocks_after: Number of blocks after transition
        statements_per_block: Number of statements per block
        iterations: Number of iterations to run
        
    Returns:
        IntegrationBenchmarkResult
    """
    # Initialize with genesis epoch
    initialize_genesis_epoch()
    
    # Register new epoch at transition point
    transition_block = blocks_before
    new_epoch = HashEpoch(
        start_block=transition_block,
        end_block=None,
        algorithm_id=0x01,  # SHA3-256
        algorithm_name="SHA3-256",
        rule_version=ConsensusRuleVersion.V2_DUAL_REQUIRED.value,
        activation_timestamp=time.time(),
        governance_hash="0x" + "AA" * 32,
    )
    register_epoch(new_epoch)
    
    # Create blocks before and after transition
    total_blocks = blocks_before + blocks_after
    blocks = create_test_chain(total_blocks, statements_per_block, use_dual_commitment=True)
    
    # Benchmark transition validation
    times = []
    
    for _ in range(iterations):
        start = time.perf_counter()
        
        # Validate all blocks including transition
        for i, block in enumerate(blocks):
            prev_block = blocks[i - 1] if i > 0 else None
            _ = validate_block_full(block, prev_block)
        
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
    
    # Compute statistics
    total_time_ms = statistics.mean(times)
    mean_block_ms = total_time_ms / total_blocks
    median_time_ms = statistics.median(times)
    
    sorted_times = sorted(times)
    p90_time_ms = sorted_times[int(len(sorted_times) * 0.90)]
    p95_time_ms = sorted_times[int(len(sorted_times) * 0.95)]
    
    throughput = (total_blocks * iterations) / (sum(times) / 1000)
    
    return IntegrationBenchmarkResult(
        scenario_name="Epoch Transition",
        block_count=total_blocks,
        statement_count_per_block=statements_per_block,
        total_validation_time_ms=total_time_ms,
        mean_block_time_ms=mean_block_ms,
        median_block_time_ms=median_time_ms / total_blocks,
        p90_block_time_ms=p90_time_ms / total_blocks,
        p95_block_time_ms=p95_time_ms / total_blocks,
        throughput_blocks_per_second=throughput,
    )


def run_integration_benchmarks() -> List[IntegrationBenchmarkResult]:
    """
    Run comprehensive integration benchmarks.
    
    Returns:
        List of IntegrationBenchmarkResult objects
    """
    results = []
    
    print("Running integration benchmarks...")
    print("=" * 80)
    
    # Test configurations
    configs = [
        (10, 10, False),    # 10 blocks, 10 statements, legacy
        (10, 10, True),     # 10 blocks, 10 statements, dual
        (100, 50, False),   # 100 blocks, 50 statements, legacy
        (100, 50, True),    # 100 blocks, 50 statements, dual
        (1000, 100, True),  # 1000 blocks, 100 statements, dual
    ]
    
    for block_count, stmt_count, use_dual in configs:
        print(f"\nFull Block Validation: {block_count} blocks, {stmt_count} statements, {'dual' if use_dual else 'legacy'}")
        result = benchmark_full_block_validation(block_count, stmt_count, use_dual)
        results.append(result)
        print(f"  Mean: {result.mean_block_time_ms:.2f} ms/block")
        print(f"  Throughput: {result.throughput_blocks_per_second:.2f} blocks/s")
    
    # Chain validation benchmarks
    chain_configs = [
        (100, 50, False),
        (100, 50, True),
        (1000, 100, True),
    ]
    
    for chain_length, stmt_count, use_dual in chain_configs:
        print(f"\nChain Validation: {chain_length} blocks, {stmt_count} statements, {'dual' if use_dual else 'legacy'}")
        result = benchmark_chain_validation(chain_length, stmt_count, use_dual)
        results.append(result)
        print(f"  Mean: {result.mean_block_time_ms:.2f} ms/block")
        print(f"  Throughput: {result.throughput_blocks_per_second:.2f} blocks/s")
    
    # Historical verification
    print(f"\nHistorical Verification: 5 epochs, 100 blocks/epoch")
    result = benchmark_historical_verification(5, 100, 50)
    results.append(result)
    print(f"  Mean: {result.mean_block_time_ms:.2f} ms/block")
    print(f"  Throughput: {result.throughput_blocks_per_second:.2f} blocks/s")
    
    # Epoch transition
    print(f"\nEpoch Transition: 50 blocks before, 50 blocks after")
    result = benchmark_epoch_transition(50, 50, 50)
    results.append(result)
    print(f"  Mean: {result.mean_block_time_ms:.2f} ms/block")
    print(f"  Throughput: {result.throughput_blocks_per_second:.2f} blocks/s")
    
    print("\n" + "=" * 80)
    print("Integration benchmarks complete")
    
    return results


def export_results_csv(results: List[IntegrationBenchmarkResult], filename: str) -> None:
    """Export results to CSV."""
    import csv
    
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Scenario",
            "Block Count",
            "Statements/Block",
            "Total Time (ms)",
            "Mean (ms/block)",
            "Median (ms/block)",
            "P90 (ms/block)",
            "P95 (ms/block)",
            "Throughput (blocks/s)",
        ])
        
        for result in results:
            writer.writerow([
                result.scenario_name,
                result.block_count,
                result.statement_count_per_block,
                result.total_validation_time_ms,
                result.mean_block_time_ms,
                result.median_block_time_ms,
                result.p90_block_time_ms,
                result.p95_block_time_ms,
                result.throughput_blocks_per_second,
            ])
    
    print(f"\nResults exported to {filename}")


if __name__ == "__main__":
    results = run_integration_benchmarks()
    export_results_csv(results, "integration_benchmark_results.csv")
