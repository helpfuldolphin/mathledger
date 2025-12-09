#!/usr/bin/env python3
# PHASE II — NOT USED IN PHASE I
"""
Seed Lineage Tree Visualizer — Human-Readable PRNG Hierarchy Display.

This tool renders a text-based tree view of PRNG seed namespaces and their
derived seeds, making it easy to understand and debug seed hierarchies.

Example Output:
    ┌─────────────────────────────────────────────────────────────────┐
    │ Master Seed: a1b2c3d4e5f6...                                    │
    ├─────────────────────────────────────────────────────────────────┤
    │ Derivation Scheme: PRNGKey(root, path) -> SHA256 -> seed % 2^32 │
    └─────────────────────────────────────────────────────────────────┘

    slice_uplift_sparse
    ├── baseline
    │   ├── cycle_0000
    │   │   └── ordering → seed: 1234567890
    │   ├── cycle_0001
    │   │   └── ordering → seed: 2345678901
    │   └── cycle_0002
    │       └── ordering → seed: 3456789012
    └── rfl
        ├── cycle_0000
        │   └── ordering → seed: 4567890123
        └── cycle_0001
            └── ordering → seed: 5678901234

Exit Codes:
    0 - Success
    1 - Error loading manifest or generating tree

Usage:
    python scripts/seed_lineage_tree.py --manifest artifacts/manifest.json
    python scripts/seed_lineage_tree.py --manifest manifest.json --max-depth 3
    python scripts/seed_lineage_tree.py --manifest manifest.json --filter-prefix slice_a
    python scripts/seed_lineage_tree.py --seed 42 --cycles 10 --slice test --mode baseline

Author: Agent A2 (runtime-ops-2)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class TreeNode:
    """A node in the seed lineage tree."""
    name: str
    seed: Optional[int] = None
    children: Dict[str, "TreeNode"] = field(default_factory=dict)
    is_leaf: bool = False

    def add_path(self, path: Tuple[str, ...], seed: int) -> None:
        """Add a path with its derived seed to the tree."""
        if not path:
            self.seed = seed
            self.is_leaf = True
            return

        first, *rest = path
        if first not in self.children:
            self.children[first] = TreeNode(name=first)
        self.children[first].add_path(tuple(rest), seed)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {"name": self.name}
        if self.seed is not None:
            result["seed"] = self.seed
        if self.children:
            result["children"] = {
                k: v.to_dict() for k, v in sorted(self.children.items())
            }
        return result


def render_tree(
    node: TreeNode,
    prefix: str = "",
    is_last: bool = True,
    max_depth: Optional[int] = None,
    current_depth: int = 0,
    filter_prefix: Optional[str] = None,
) -> List[str]:
    """
    Render a tree node and its children as text.

    Args:
        node: The tree node to render.
        prefix: Current line prefix for tree structure.
        is_last: Whether this is the last sibling.
        max_depth: Maximum depth to render (None for unlimited).
        current_depth: Current depth in the tree.
        filter_prefix: Only show nodes matching this prefix.

    Returns:
        List of rendered lines.
    """
    lines = []

    # Check depth limit
    if max_depth is not None and current_depth > max_depth:
        return lines

    # Determine connectors
    connector = "└── " if is_last else "├── "
    child_prefix = prefix + ("    " if is_last else "│   ")

    # Format this node
    if node.name == "<root>":
        # Skip root node in output
        pass
    elif node.is_leaf and node.seed is not None:
        lines.append(f"{prefix}{connector}{node.name} → seed: {node.seed}")
    else:
        lines.append(f"{prefix}{connector}{node.name}")

    # Sort children alphabetically for deterministic output
    sorted_children = sorted(node.children.items())

    # Apply filter if specified
    if filter_prefix and current_depth == 0:
        sorted_children = [
            (k, v) for k, v in sorted_children
            if k.startswith(filter_prefix)
        ]

    # Render children
    for i, (child_name, child_node) in enumerate(sorted_children):
        is_last_child = (i == len(sorted_children) - 1)
        child_lines = render_tree(
            child_node,
            prefix=child_prefix if node.name != "<root>" else "",
            is_last=is_last_child,
            max_depth=max_depth,
            current_depth=current_depth + 1,
            filter_prefix=None,  # Only filter at root level
        )
        lines.extend(child_lines)

    return lines


def build_tree_from_paths(
    paths_and_seeds: List[Tuple[Tuple[str, ...], int]],
) -> TreeNode:
    """
    Build a tree from a list of paths and their derived seeds.

    Args:
        paths_and_seeds: List of (path_tuple, seed) pairs.

    Returns:
        Root TreeNode.
    """
    root = TreeNode(name="<root>")

    for path, seed in sorted(paths_and_seeds):
        root.add_path(path, seed)

    return root


def load_lineage_from_manifest(manifest_path: Path) -> Tuple[str, str, List[Tuple[Tuple[str, ...], int]]]:
    """
    Load seed lineage data from a manifest.

    Returns:
        Tuple of (master_seed_hex, derivation_scheme, paths_and_seeds)
    """
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    attestation = manifest.get('prng_attestation') or manifest.get('prng', {})
    master_seed = attestation.get('master_seed_hex', '<unknown>')
    scheme = attestation.get('derivation_scheme', '<unknown>')

    # Try to reconstruct lineage from config
    config = manifest.get('configuration', {}).get('snapshot', {})
    slice_name = config.get('slice_name', 'default')
    mode = config.get('mode', 'baseline')
    num_cycles = attestation.get('lineage_entry_count') or config.get('num_cycles', 10)

    # We need to derive the seeds ourselves
    try:
        from rfl.prng import DeterministicPRNG

        prng = DeterministicPRNG(master_seed)
        paths_and_seeds = []

        for i in range(num_cycles):
            path = (slice_name, mode, f"cycle_{i:04d}", "ordering")
            seed = prng.seed_for_path(*path)
            paths_and_seeds.append((path, seed))

        return master_seed, scheme, paths_and_seeds

    except Exception:
        # Can't derive, return empty
        return master_seed, scheme, []


def generate_lineage(
    master_seed: int,
    num_cycles: int,
    slice_name: str,
    mode: str,
) -> Tuple[str, str, List[Tuple[Tuple[str, ...], int]]]:
    """
    Generate seed lineage from parameters.

    Returns:
        Tuple of (master_seed_hex, derivation_scheme, paths_and_seeds)
    """
    from rfl.prng import DeterministicPRNG, int_to_hex_seed

    master_seed_hex = int_to_hex_seed(master_seed)
    prng = DeterministicPRNG(master_seed_hex)
    scheme = "PRNGKey(root, path) -> SHA256 -> seed % 2^32"

    paths_and_seeds = []
    for i in range(num_cycles):
        path = (slice_name, mode, f"cycle_{i:04d}", "ordering")
        seed = prng.seed_for_path(*path)
        paths_and_seeds.append((path, seed))

    return master_seed_hex, scheme, paths_and_seeds


def format_header(master_seed: str, scheme: str) -> str:
    """Format the header box."""
    lines = []
    width = 70

    lines.append("┌" + "─" * (width - 2) + "┐")
    lines.append(f"│ Master Seed: {master_seed[:40]}...".ljust(width - 1) + "│")
    lines.append("├" + "─" * (width - 2) + "┤")
    scheme_line = f"│ Derivation: {scheme}"
    if len(scheme_line) > width - 1:
        scheme_line = scheme_line[:width - 4] + "..."
    lines.append(scheme_line.ljust(width - 1) + "│")
    lines.append("└" + "─" * (width - 2) + "┘")

    return "\n".join(lines)


def visualize_lineage(
    master_seed_hex: str,
    derivation_scheme: str,
    paths_and_seeds: List[Tuple[Tuple[str, ...], int]],
    max_depth: Optional[int] = None,
    filter_prefix: Optional[str] = None,
) -> str:
    """
    Visualize seed lineage as a tree.

    Args:
        master_seed_hex: The master seed in hex format.
        derivation_scheme: The derivation scheme description.
        paths_and_seeds: List of (path, seed) pairs.
        max_depth: Maximum tree depth to display.
        filter_prefix: Filter to paths starting with this prefix.

    Returns:
        Formatted tree visualization string.
    """
    lines = []

    # Header
    lines.append(format_header(master_seed_hex, derivation_scheme))
    lines.append("")

    if not paths_and_seeds:
        lines.append("(No seed lineage data available)")
        return "\n".join(lines)

    # Build and render tree
    tree = build_tree_from_paths(paths_and_seeds)

    tree_lines = render_tree(
        tree,
        max_depth=max_depth,
        filter_prefix=filter_prefix,
    )

    # Remove leading empty lines from tree
    while tree_lines and not tree_lines[0].strip():
        tree_lines.pop(0)

    lines.extend(tree_lines)

    # Summary
    lines.append("")
    lines.append(f"Total paths: {len(paths_and_seeds)}")

    return "\n".join(lines)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Seed Lineage Tree Visualizer — Human-Readable PRNG Hierarchy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Visualize from manifest
    python scripts/seed_lineage_tree.py --manifest artifacts/manifest.json

    # Limit tree depth
    python scripts/seed_lineage_tree.py --manifest manifest.json --max-depth 3

    # Filter to specific slice
    python scripts/seed_lineage_tree.py --manifest manifest.json --filter-prefix slice_a

    # Generate tree from parameters
    python scripts/seed_lineage_tree.py --seed 42 --cycles 10 --slice test --mode baseline

    # Output as JSON
    python scripts/seed_lineage_tree.py --manifest manifest.json --json

Example Output:
    slice_uplift_sparse
    ├── baseline
    │   ├── cycle_0000
    │   │   └── ordering → seed: 1234567890
    │   └── cycle_0001
    │       └── ordering → seed: 2345678901
    └── rfl
        └── cycle_0000
            └── ordering → seed: 3456789012
        """,
    )

    # Input source (either manifest or parameters)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--manifest", "-m",
        type=Path,
        help="Path to manifest.json file",
    )
    input_group.add_argument(
        "--seed", "-s",
        type=int,
        help="Master seed integer (requires --cycles, --slice, --mode)",
    )

    # Generation parameters
    parser.add_argument(
        "--cycles", "-c",
        type=int,
        default=10,
        help="Number of cycles to generate (default: 10)",
    )
    parser.add_argument(
        "--slice",
        type=str,
        default="default",
        help="Slice name (default: 'default')",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="baseline",
        help="Mode name (default: 'baseline')",
    )

    # Display options
    parser.add_argument(
        "--max-depth", "-d",
        type=int,
        help="Maximum tree depth to display",
    )
    parser.add_argument(
        "--filter-prefix", "-f",
        type=str,
        help="Filter to paths starting with this prefix",
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output as JSON instead of tree",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Write output to file",
    )

    args = parser.parse_args()

    try:
        # Load or generate lineage data
        if args.manifest:
            master_seed, scheme, paths_and_seeds = load_lineage_from_manifest(args.manifest)
        else:
            master_seed, scheme, paths_and_seeds = generate_lineage(
                args.seed,
                args.cycles,
                args.slice,
                args.mode,
            )

        if args.json:
            tree = build_tree_from_paths(paths_and_seeds)
            output_data = {
                "master_seed_hex": master_seed,
                "derivation_scheme": scheme,
                "total_paths": len(paths_and_seeds),
                "tree": tree.to_dict(),
            }
            output = json.dumps(output_data, indent=2)
        else:
            output = visualize_lineage(
                master_seed,
                scheme,
                paths_and_seeds,
                max_depth=args.max_depth,
                filter_prefix=args.filter_prefix,
            )

        if args.output:
            args.output.write_text(output)
            print(f"Output written to: {args.output}")
        else:
            print(output)

        return 0

    except FileNotFoundError as e:
        print(f"Error: File not found: {e.filename}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

