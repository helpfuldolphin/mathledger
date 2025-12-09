# PHASE II — NOT USED IN PHASE I
"""
Tests for Seed Lineage Tree Visualizer.

Verifies:
- Output is stable across runs
- Filtering and max-depth behavior
- JSON output format
"""

import json
import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.seed_lineage_tree import (
    TreeNode,
    build_tree_from_paths,
    render_tree,
    visualize_lineage,
    format_header,
    generate_lineage,
)


class TestTreeNode:
    """Tests for TreeNode class."""

    def test_add_single_path(self):
        """Adding a single path creates correct tree structure."""
        root = TreeNode(name="<root>")
        root.add_path(("a", "b", "c"), seed=123)

        assert "a" in root.children
        assert "b" in root.children["a"].children
        assert "c" in root.children["a"].children["b"].children
        assert root.children["a"].children["b"].children["c"].seed == 123

    def test_add_multiple_paths_shared_prefix(self):
        """Paths with shared prefix share tree nodes."""
        root = TreeNode(name="<root>")
        root.add_path(("a", "b", "c"), seed=123)
        root.add_path(("a", "b", "d"), seed=456)

        assert "a" in root.children
        assert "b" in root.children["a"].children
        assert "c" in root.children["a"].children["b"].children
        assert "d" in root.children["a"].children["b"].children

    def test_to_dict(self):
        """TreeNode converts to dict correctly."""
        root = TreeNode(name="<root>")
        root.add_path(("a", "b"), seed=123)

        d = root.to_dict()

        assert d["name"] == "<root>"
        assert "a" in d["children"]
        assert "b" in d["children"]["a"]["children"]
        assert d["children"]["a"]["children"]["b"]["seed"] == 123


class TestBuildTree:
    """Tests for tree building."""

    def test_build_from_paths(self):
        """Builds correct tree from path list."""
        paths_and_seeds = [
            (("slice_a", "baseline", "cycle_0000"), 100),
            (("slice_a", "baseline", "cycle_0001"), 200),
            (("slice_a", "rfl", "cycle_0000"), 300),
        ]

        tree = build_tree_from_paths(paths_and_seeds)

        assert "slice_a" in tree.children
        assert "baseline" in tree.children["slice_a"].children
        assert "rfl" in tree.children["slice_a"].children

    def test_build_empty(self):
        """Empty path list builds empty tree."""
        tree = build_tree_from_paths([])
        assert len(tree.children) == 0


class TestRenderTree:
    """Tests for tree rendering."""

    def test_render_simple_tree(self):
        """Renders a simple tree correctly."""
        paths_and_seeds = [
            (("a", "b"), 123),
            (("a", "c"), 456),
        ]
        tree = build_tree_from_paths(paths_and_seeds)

        lines = render_tree(tree)

        # Check structure exists
        assert any("a" in line for line in lines)
        assert any("b" in line and "123" in line for line in lines)
        assert any("c" in line and "456" in line for line in lines)

    def test_render_deterministic_ordering(self):
        """Tree rendering is alphabetically ordered."""
        paths_and_seeds = [
            (("z",), 1),
            (("a",), 2),
            (("m",), 3),
        ]
        tree = build_tree_from_paths(paths_and_seeds)

        lines = render_tree(tree)
        content_lines = [l for l in lines if l.strip()]

        # a should come before m, m before z
        a_idx = next(i for i, l in enumerate(content_lines) if "a" in l)
        m_idx = next(i for i, l in enumerate(content_lines) if "m" in l)
        z_idx = next(i for i, l in enumerate(content_lines) if "z" in l)

        assert a_idx < m_idx < z_idx

    def test_render_max_depth(self):
        """Max depth limits tree rendering."""
        paths_and_seeds = [
            (("level1", "level2", "level3", "level4"), 123),
        ]
        tree = build_tree_from_paths(paths_and_seeds)

        lines = render_tree(tree, max_depth=2)

        # level1 and level2 should appear, level3/level4 should not
        assert any("level1" in line for line in lines)
        assert any("level2" in line for line in lines)
        assert not any("level3" in line for line in lines)
        assert not any("level4" in line for line in lines)

    def test_render_filter_prefix(self):
        """Filter prefix limits visible nodes."""
        paths_and_seeds = [
            (("slice_a", "data"), 100),
            (("slice_b", "data"), 200),
        ]
        tree = build_tree_from_paths(paths_and_seeds)

        lines = render_tree(tree, filter_prefix="slice_a")

        assert any("slice_a" in line for line in lines)
        assert not any("slice_b" in line for line in lines)

    def test_render_stable_across_runs(self):
        """Multiple renders produce identical output."""
        paths_and_seeds = [
            (("a", "b", "c"), 123),
            (("a", "d", "e"), 456),
        ]
        tree = build_tree_from_paths(paths_and_seeds)

        lines1 = render_tree(tree)
        lines2 = render_tree(tree)
        lines3 = render_tree(tree)

        assert lines1 == lines2 == lines3


class TestVisualizeLineage:
    """Tests for full visualization."""

    def test_visualize_includes_header(self):
        """Visualization includes header with seed and scheme."""
        output = visualize_lineage(
            master_seed_hex="a" * 64,
            derivation_scheme="test_scheme",
            paths_and_seeds=[],
        )

        assert "a" * 16 in output  # Truncated seed
        assert "test_scheme" in output or "Derivation" in output

    def test_visualize_includes_tree(self):
        """Visualization includes tree content."""
        paths = [
            (("slice", "mode", "cycle"), 123),
        ]
        output = visualize_lineage(
            master_seed_hex="a" * 64,
            derivation_scheme="scheme",
            paths_and_seeds=paths,
        )

        assert "slice" in output
        assert "mode" in output
        assert "123" in output

    def test_visualize_includes_summary(self):
        """Visualization includes path count summary."""
        paths = [(("a",), 1), (("b",), 2)]
        output = visualize_lineage(
            master_seed_hex="a" * 64,
            derivation_scheme="scheme",
            paths_and_seeds=paths,
        )

        assert "Total paths: 2" in output


class TestFormatHeader:
    """Tests for header formatting."""

    def test_header_box_structure(self):
        """Header has correct box structure."""
        header = format_header("a" * 64, "scheme")

        assert "┌" in header
        assert "└" in header
        assert "│" in header
        assert "─" in header

    def test_header_contains_seed(self):
        """Header contains master seed."""
        header = format_header("abcd" * 16, "scheme")
        assert "abcd" in header

    def test_header_contains_scheme(self):
        """Header contains derivation scheme."""
        header = format_header("a" * 64, "test_scheme_name")
        assert "test_scheme" in header or "Derivation" in header


class TestGenerateLineage:
    """Tests for lineage generation."""

    def test_generate_creates_correct_count(self):
        """Generates correct number of paths."""
        master, scheme, paths = generate_lineage(
            master_seed=42,
            num_cycles=5,
            slice_name="test",
            mode="baseline",
        )

        assert len(paths) == 5

    def test_generate_deterministic(self):
        """Generation is deterministic."""
        result1 = generate_lineage(42, 3, "test", "mode")
        result2 = generate_lineage(42, 3, "test", "mode")

        assert result1 == result2

    def test_generate_different_seeds_different_results(self):
        """Different seeds produce different results."""
        _, _, paths1 = generate_lineage(42, 3, "test", "mode")
        _, _, paths2 = generate_lineage(43, 3, "test", "mode")

        seeds1 = [s for _, s in paths1]
        seeds2 = [s for _, s in paths2]

        assert seeds1 != seeds2

    def test_generate_returns_hex_seed(self):
        """Returns 64-char hex master seed."""
        master, _, _ = generate_lineage(42, 1, "test", "mode")

        assert len(master) == 64
        int(master, 16)  # Should not raise


class TestJsonOutput:
    """Tests for JSON output format."""

    def test_tree_to_json(self):
        """Tree serializes to valid JSON."""
        paths = [
            (("a", "b"), 123),
        ]
        tree = build_tree_from_paths(paths)

        d = tree.to_dict()
        json_str = json.dumps(d)

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed["name"] == "<root>"


class TestStability:
    """Tests for output stability."""

    def test_output_stable_across_multiple_calls(self):
        """Output is identical across multiple invocations."""
        paths = [
            (("slice_a", "baseline", "cycle_0000"), 100),
            (("slice_a", "baseline", "cycle_0001"), 200),
            (("slice_a", "rfl", "cycle_0000"), 300),
        ]

        outputs = [
            visualize_lineage("a" * 64, "scheme", paths)
            for _ in range(5)
        ]

        assert all(o == outputs[0] for o in outputs)

    def test_json_output_stable(self):
        """JSON output is deterministically sorted."""
        paths = [
            (("z",), 1),
            (("a",), 2),
            (("m",), 3),
        ]
        tree = build_tree_from_paths(paths)

        outputs = [json.dumps(tree.to_dict(), sort_keys=True) for _ in range(5)]

        assert all(o == outputs[0] for o in outputs)

