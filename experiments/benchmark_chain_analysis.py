# PHASE II — NOT USED IN PHASE I
#
# Performance benchmark script for the ChainAnalyzer class.
#
# This script generates synthetic derivation data with varying DAG structures
# and measures the performance of ChainAnalyzer instantiation (DAG construction
# and depth computation) and analysis methods.
#
# PRNG Contract (Agent A2 — runtime-ops-2):
#   All randomness uses DeterministicPRNG for reproducible benchmarks.
#   See rfl/prng/deterministic_prng.py for implementation.

import time
import sys
from typing import List, Dict

# Add project root to sys.path for local imports
from pathlib import Path
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from experiments.derivation_chain_analysis import ChainAnalyzer, Derivation
from rfl.prng import DeterministicPRNG, int_to_hex_seed

# Fixed seed for reproducible benchmarks
BENCHMARK_SEED = 20251206
_benchmark_prng = DeterministicPRNG(int_to_hex_seed(BENCHMARK_SEED))

def generate_derivations(num_nodes: int, structure: str, fan_in: int = 2) -> List[Derivation]:
    """
    Generates synthetic derivation data.

    PRNG Contract (Agent A2):
        Uses hierarchical PRNG for deterministic, reproducible benchmark data.
        Path: "benchmark" → structure → f"nodes_{num_nodes}" → component
    """
    derivations: List[Derivation] = []
    nodes = [f"h_{i}" for i in range(num_nodes)]

    # Get RNG scoped to this generation call
    gen_rng = _benchmark_prng.for_path("benchmark", structure, f"nodes_{num_nodes}", "generation")

    if structure == "chain":
        # A -> B -> C -> ...
        for i in range(1, num_nodes):
            derivations.append({"hash": nodes[i], "premises": [nodes[i-1]]})
        # Add the axiom
        derivations.append({"hash": nodes[0], "premises": []})

    elif structure == "tree":
        # Wide and shallow tree, each node has `fan_in` new premises
        next_premise_idx = 1
        for i in range(num_nodes):
            premises = []
            if next_premise_idx < num_nodes:
                for j in range(fan_in):
                    if next_premise_idx < num_nodes:
                        premises.append(nodes[next_premise_idx])
                        next_premise_idx += 1
            derivations.append({"hash": nodes[i], "premises": premises})

    elif structure == "random":
        # Each node connects to `fan_in` random previous nodes
        for i in range(num_nodes):
            premises = []
            if i > 0:
                num_premises = min(i, fan_in)
                premises = gen_rng.sample(nodes[:i], num_premises)
            derivations.append({"hash": nodes[i], "premises": premises})

    # Shuffle to simulate unordered logs using scoped RNG
    shuffle_rng = _benchmark_prng.for_path("benchmark", structure, f"nodes_{num_nodes}", "shuffle")
    shuffle_rng.shuffle(derivations)
    return derivations

def run_benchmark(description: str, derivations: List[Derivation]):
    """Runs a single benchmark and prints the results."""
    print("-" * 60)
    print(description)
    print(f"  Number of derivations: {len(derivations)}")
    
    # --- Benchmark Instantiation ---
    start_time = time.perf_counter()
    analyzer = ChainAnalyzer(derivations)
    end_time = time.perf_counter()
    init_duration = end_time - start_time
    print(f"  -> Instantiation (build + depths): {init_duration:.6f} seconds")

    footprint = analyzer.get_dag_footprint()
    print(f"  -> Footprint: {footprint['node_count']} nodes, {footprint['edge_count']} edges")

    # --- Benchmark Analysis ---
    if not derivations:
        print("-" * 60)
        return

    # Select some nodes to analyze using hierarchical PRNG for reproducibility
    num_samples = min(len(analyzer.dag), 100)
    analysis_rng = _benchmark_prng.for_path("benchmark", "analysis", description[:20])
    sample_nodes = analysis_rng.sample(list(analyzer.dag.keys()), num_samples)

    start_time = time.perf_counter()
    for h in sample_nodes:
        analyzer.get_depth(h)
    end_time = time.perf_counter()
    avg_depth_time = (end_time - start_time) / num_samples if num_samples > 0 else 0
    print(f"  -> Avg `get_depth` (cached): {avg_depth_time * 1e6:.3f} µs")

    start_time = time.perf_counter()
    for h in sample_nodes:
        analyzer.get_longest_chain(h)
    end_time = time.perf_counter()
    avg_chain_time = (end_time - start_time) / num_samples if num_samples > 0 else 0
    print(f"  -> Avg `get_longest_chain`: {avg_chain_time * 1e6:.3f} µs")
    print("-" * 60)


def main():
    """Main function to run all benchmarks."""
    print("=" * 60)
    print("PERFORMANCE BENCHMARKS FOR ChainAnalyzer")
    print("=" * 60)

    # --- Small Scale ---
    run_benchmark(
        "Small Linear Chain (1,000 nodes)",
        generate_derivations(1000, "chain")
    )
    run_benchmark(
        "Small Random DAG (1,000 nodes, fan-in=2)",
        generate_derivations(1000, "random", fan_in=2)
    )

    # --- Medium Scale ---
    run_benchmark(
        "Medium Linear Chain (10,000 nodes)",
        generate_derivations(10000, "chain")
    )
    run_benchmark(
        "Medium Random DAG (10,000 nodes, fan-in=3)",
        generate_derivations(10000, "random", fan_in=3)
    )
    run_benchmark(
        "Medium Wide Tree (10,000 nodes, fan-in=5)",
        generate_derivations(10000, "tree", fan_in=5)
    )

    # --- Large Scale ---
    # Python's recursion limit might be an issue for deep chains here.
    # The iterative computation in ChainAnalyzer should handle it.
    try:
        run_benchmark(
            "Large Linear Chain (100,000 nodes - stress test)",
            generate_derivations(100000, "chain")
        )
    except RecursionError:
        print("\n[ERROR] Hit Python's recursion limit on the large linear chain.")
        print("ChainAnalyzer's internal recursion for depth calculation might need to be converted to an iterative approach for extreme depths.")


    run_benchmark(
        "Large Random DAG (50,000 nodes, fan-in=2)",
        generate_derivations(50000, "random", fan_in=2)
    )


if __name__ == "__main__":
    main()
