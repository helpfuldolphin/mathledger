"""
Network Benchmarks

Simulates network propagation and multi-node consensus scenarios.

Note: These are simulation-based benchmarks. For real network testing,
deploy to actual testnet with multiple nodes.

Author: Manus-H
"""

import time
import random
from dataclasses import dataclass
from typing import List
import statistics

from basis.ledger.block_pq import BlockHeaderPQ


@dataclass
class NetworkBenchmarkResult:
    """
    Result of a network benchmark.
    
    Attributes:
        scenario_name: Name of the test scenario
        node_count: Number of nodes in simulation
        block_size_bytes: Average block size in bytes
        propagation_time_ms: Time for block to reach 90% of nodes
        consensus_latency_ms: Time for network to reach consensus
        bandwidth_usage_mbps: Estimated bandwidth usage
        orphan_rate: Percentage of orphaned blocks
    """
    
    scenario_name: str
    node_count: int
    block_size_bytes: int
    propagation_time_ms: float
    consensus_latency_ms: float
    bandwidth_usage_mbps: float
    orphan_rate: float


class NetworkNode:
    """Simulates a network node."""
    
    def __init__(self, node_id: int, latency_ms: float):
        self.node_id = node_id
        self.latency_ms = latency_ms
        self.received_blocks = []
        self.peers = []
    
    def add_peer(self, peer: 'NetworkNode'):
        """Add a peer connection."""
        if peer not in self.peers:
            self.peers.append(peer)
    
    def receive_block(self, block: BlockHeaderPQ, timestamp: float):
        """Receive a block from the network."""
        self.received_blocks.append((block, timestamp))
    
    def propagate_block(self, block: BlockHeaderPQ, timestamp: float):
        """Propagate a block to all peers."""
        for peer in self.peers:
            # Simulate network latency
            arrival_time = timestamp + self.latency_ms / 1000
            peer.receive_block(block, arrival_time)


class NetworkSimulator:
    """Simulates a blockchain network."""
    
    def __init__(self, node_count: int, avg_latency_ms: float):
        self.nodes = []
        self.avg_latency_ms = avg_latency_ms
        
        # Create nodes with varying latencies
        for i in range(node_count):
            latency = random.gauss(avg_latency_ms, avg_latency_ms * 0.2)
            latency = max(1.0, latency)  # Minimum 1ms latency
            node = NetworkNode(i, latency)
            self.nodes.append(node)
        
        # Connect nodes (mesh topology)
        self._create_mesh_topology()
    
    def _create_mesh_topology(self):
        """Create a mesh network topology."""
        for node in self.nodes:
            # Each node connects to ~sqrt(n) peers
            peer_count = max(3, int(len(self.nodes) ** 0.5))
            peers = random.sample(self.nodes, min(peer_count, len(self.nodes) - 1))
            for peer in peers:
                if peer != node:
                    node.add_peer(peer)
    
    def propagate_block(self, block: BlockHeaderPQ) -> float:
        """
        Propagate a block through the network.
        
        Returns:
            Time (in ms) for block to reach 90% of nodes
        """
        start_time = time.time()
        
        # Originating node broadcasts block
        origin_node = self.nodes[0]
        origin_node.receive_block(block, start_time)
        origin_node.propagate_block(block, start_time)
        
        # Simulate propagation waves
        max_waves = 10
        for wave in range(max_waves):
            # Each node that has the block propagates to peers
            for node in self.nodes:
                if len(node.received_blocks) > 0:
                    last_block, recv_time = node.received_blocks[-1]
                    if last_block == block:
                        node.propagate_block(block, recv_time)
            
            # Check if 90% of nodes have received the block
            nodes_with_block = sum(1 for n in self.nodes if any(b == block for b, _ in n.received_blocks))
            if nodes_with_block >= 0.9 * len(self.nodes):
                break
        
        end_time = time.time()
        propagation_time_ms = (end_time - start_time) * 1000
        
        return propagation_time_ms
    
    def simulate_consensus(self, block_count: int) -> tuple[float, float]:
        """
        Simulate consensus for multiple blocks.
        
        Returns:
            Tuple of (avg_consensus_latency_ms, orphan_rate)
        """
        consensus_times = []
        orphaned_blocks = 0
        
        for i in range(block_count):
            # Create block
            block = BlockHeaderPQ(
                block_number=i,
                prev_hash=f"0x{i-1:064x}",
                merkle_root=f"0x{i:064x}",
                timestamp=time.time(),
                statements=[f"stmt_{i}"],
            )
            
            # Propagate block
            prop_time = self.propagate_block(block)
            consensus_times.append(prop_time)
            
            # Simulate orphaning (5% chance)
            if random.random() < 0.05:
                orphaned_blocks += 1
        
        avg_consensus_latency = statistics.mean(consensus_times)
        orphan_rate = orphaned_blocks / block_count
        
        return avg_consensus_latency, orphan_rate


def benchmark_block_propagation(
    node_count: int,
    block_size_bytes: int,
    avg_latency_ms: float,
    iterations: int = 10,
) -> NetworkBenchmarkResult:
    """
    Benchmark block propagation time.
    
    Args:
        node_count: Number of nodes in network
        block_size_bytes: Size of block in bytes
        avg_latency_ms: Average network latency between nodes
        iterations: Number of iterations to run
        
    Returns:
        NetworkBenchmarkResult
    """
    propagation_times = []
    
    for _ in range(iterations):
        # Create network simulator
        simulator = NetworkSimulator(node_count, avg_latency_ms)
        
        # Create test block
        statements = [f"stmt_{i}" for i in range(block_size_bytes // 100)]
        block = BlockHeaderPQ(
            block_number=1,
            prev_hash="0x" + "00" * 32,
            merkle_root="0x" + "11" * 32,
            timestamp=time.time(),
            statements=statements,
        )
        
        # Measure propagation time
        prop_time = simulator.propagate_block(block)
        propagation_times.append(prop_time)
    
    avg_propagation_time = statistics.mean(propagation_times)
    
    # Estimate bandwidth usage (simplified)
    # Bandwidth = block_size * node_count / propagation_time
    bandwidth_mbps = (block_size_bytes * node_count * 8) / (avg_propagation_time / 1000) / (1024 * 1024)
    
    return NetworkBenchmarkResult(
        scenario_name=f"Block Propagation ({node_count} nodes)",
        node_count=node_count,
        block_size_bytes=block_size_bytes,
        propagation_time_ms=avg_propagation_time,
        consensus_latency_ms=avg_propagation_time,  # Same for single block
        bandwidth_usage_mbps=bandwidth_mbps,
        orphan_rate=0.0,  # No orphans in single block test
    )


def benchmark_consensus_latency(
    node_count: int,
    block_count: int,
    avg_latency_ms: float,
    iterations: int = 5,
) -> NetworkBenchmarkResult:
    """
    Benchmark consensus latency for multiple blocks.
    
    Args:
        node_count: Number of nodes in network
        block_count: Number of blocks to process
        avg_latency_ms: Average network latency between nodes
        iterations: Number of iterations to run
        
    Returns:
        NetworkBenchmarkResult
    """
    consensus_latencies = []
    orphan_rates = []
    
    for _ in range(iterations):
        # Create network simulator
        simulator = NetworkSimulator(node_count, avg_latency_ms)
        
        # Simulate consensus
        consensus_latency, orphan_rate = simulator.simulate_consensus(block_count)
        consensus_latencies.append(consensus_latency)
        orphan_rates.append(orphan_rate)
    
    avg_consensus_latency = statistics.mean(consensus_latencies)
    avg_orphan_rate = statistics.mean(orphan_rates)
    
    # Estimate bandwidth (simplified)
    block_size = 2048  # Assume 2KB blocks
    bandwidth_mbps = (block_size * node_count * 8) / (avg_consensus_latency / 1000) / (1024 * 1024)
    
    return NetworkBenchmarkResult(
        scenario_name=f"Consensus Latency ({node_count} nodes, {block_count} blocks)",
        node_count=node_count,
        block_size_bytes=block_size,
        propagation_time_ms=avg_consensus_latency,
        consensus_latency_ms=avg_consensus_latency,
        bandwidth_usage_mbps=bandwidth_mbps,
        orphan_rate=avg_orphan_rate,
    )


def benchmark_bandwidth_usage(
    node_count: int,
    block_size_bytes: int,
    blocks_per_second: float,
    duration_seconds: int = 60,
) -> NetworkBenchmarkResult:
    """
    Benchmark network bandwidth usage.
    
    Args:
        node_count: Number of nodes in network
        block_size_bytes: Size of each block
        blocks_per_second: Block production rate
        duration_seconds: Duration of simulation
        
    Returns:
        NetworkBenchmarkResult
    """
    total_blocks = int(blocks_per_second * duration_seconds)
    total_data_bytes = total_blocks * block_size_bytes * node_count
    
    # Each block is transmitted to all nodes
    bandwidth_mbps = (total_data_bytes * 8) / duration_seconds / (1024 * 1024)
    
    # Estimate propagation time (inverse of block rate)
    propagation_time_ms = 1000 / blocks_per_second
    
    return NetworkBenchmarkResult(
        scenario_name=f"Bandwidth Usage ({node_count} nodes, {blocks_per_second} blocks/s)",
        node_count=node_count,
        block_size_bytes=block_size_bytes,
        propagation_time_ms=propagation_time_ms,
        consensus_latency_ms=propagation_time_ms,
        bandwidth_usage_mbps=bandwidth_mbps,
        orphan_rate=0.01,  # Assume 1% orphan rate
    )


def run_network_benchmarks() -> List[NetworkBenchmarkResult]:
    """
    Run comprehensive network benchmarks.
    
    Returns:
        List of NetworkBenchmarkResult objects
    """
    results = []
    
    print("Running network benchmarks (simulation-based)...")
    print("=" * 80)
    print("Note: These are simulated results. Deploy to actual testnet for real metrics.")
    print("=" * 80)
    
    # Block propagation benchmarks
    propagation_configs = [
        (10, 2048, 50),    # 10 nodes, 2KB blocks, 50ms latency
        (50, 2048, 100),   # 50 nodes, 2KB blocks, 100ms latency
        (100, 4096, 150),  # 100 nodes, 4KB blocks, 150ms latency
    ]
    
    for node_count, block_size, latency in propagation_configs:
        print(f"\nBlock Propagation: {node_count} nodes, {block_size}B blocks, {latency}ms latency")
        result = benchmark_block_propagation(node_count, block_size, latency)
        results.append(result)
        print(f"  Propagation time: {result.propagation_time_ms:.2f} ms")
        print(f"  Bandwidth: {result.bandwidth_usage_mbps:.2f} Mbps")
    
    # Consensus latency benchmarks
    consensus_configs = [
        (10, 100, 50),
        (50, 100, 100),
        (100, 100, 150),
    ]
    
    for node_count, block_count, latency in consensus_configs:
        print(f"\nConsensus Latency: {node_count} nodes, {block_count} blocks, {latency}ms latency")
        result = benchmark_consensus_latency(node_count, block_count, latency)
        results.append(result)
        print(f"  Consensus latency: {result.consensus_latency_ms:.2f} ms")
        print(f"  Orphan rate: {result.orphan_rate:.2%}")
    
    # Bandwidth usage benchmarks
    bandwidth_configs = [
        (10, 2048, 1.0),   # 10 nodes, 2KB blocks, 1 block/s
        (50, 2048, 2.0),   # 50 nodes, 2KB blocks, 2 blocks/s
        (100, 4096, 5.0),  # 100 nodes, 4KB blocks, 5 blocks/s
    ]
    
    for node_count, block_size, block_rate in bandwidth_configs:
        print(f"\nBandwidth Usage: {node_count} nodes, {block_size}B blocks, {block_rate} blocks/s")
        result = benchmark_bandwidth_usage(node_count, block_size, block_rate)
        results.append(result)
        print(f"  Bandwidth: {result.bandwidth_usage_mbps:.2f} Mbps")
    
    print("\n" + "=" * 80)
    print("Network benchmarks complete")
    print("=" * 80)
    
    return results


def export_results_csv(results: List[NetworkBenchmarkResult], filename: str) -> None:
    """Export results to CSV."""
    import csv
    
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Scenario",
            "Node Count",
            "Block Size (bytes)",
            "Propagation Time (ms)",
            "Consensus Latency (ms)",
            "Bandwidth (Mbps)",
            "Orphan Rate (%)",
        ])
        
        for result in results:
            writer.writerow([
                result.scenario_name,
                result.node_count,
                result.block_size_bytes,
                result.propagation_time_ms,
                result.consensus_latency_ms,
                result.bandwidth_usage_mbps,
                result.orphan_rate * 100,
            ])
    
    print(f"\nResults exported to {filename}")


if __name__ == "__main__":
    results = run_network_benchmarks()
    export_results_csv(results, "network_benchmark_results.csv")
