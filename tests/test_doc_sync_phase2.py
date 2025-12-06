"""
PHASE II — NOT USED IN PHASE I
Unit tests for scripts/doc_sync_phase2.py

Tests cover:
- Source file parsing (YAML, Python CLI, Python types)
- Content generation (Markdown tables, TeX tables)
- Autogen block updating
- Idempotency and determinism guarantees
"""

import tempfile
import textwrap
import unittest
from pathlib import Path

from scripts.doc_sync_phase2 import (
    _escape_latex,
    _extract_arg_info,
    generate_cli_examples_md,
    generate_metrics_table_md,
    generate_prereg_table_md,
    generate_slice_table_md,
    generate_slice_table_tex,
    generate_status_table_md,
    parse_curriculum,
    parse_metrics_types,
    parse_prereg,
    parse_runner_cli,
    update_autogen_block,
)


class TestYamlParsing(unittest.TestCase):
    """Tests for YAML file parsing."""

    def test_parse_curriculum_valid(self):
        """Parse a valid curriculum YAML file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(textwrap.dedent("""
                version: 2.0
                slices:
                  test_slice:
                    description: "A test slice"
                    items:
                      - "item1"
                      - "item2"
                    prereg_hash: "abc123"
            """))
            f.flush()

            result = parse_curriculum(Path(f.name))
            self.assertEqual(result["version"], 2.0)
            self.assertIn("test_slice", result["slices"])
            self.assertEqual(len(result["slices"]["test_slice"]["items"]), 2)

    def test_parse_curriculum_missing_file(self):
        """Handle missing curriculum file gracefully."""
        result = parse_curriculum(Path("/nonexistent/file.yaml"))
        self.assertEqual(result["version"], "unknown")
        self.assertEqual(result["slices"], {})

    def test_parse_prereg_valid_list(self):
        """Parse a valid prereg YAML list."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(textwrap.dedent("""
                - experiment_id: EXP_001
                  description: "Test experiment"
                  slice_config: "config.json"
                  success_metrics:
                    - metric_a
                    - metric_b
            """))
            f.flush()

            result = parse_prereg(Path(f.name))
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["experiment_id"], "EXP_001")

    def test_parse_prereg_missing_file(self):
        """Handle missing prereg file gracefully."""
        result = parse_prereg(Path("/nonexistent/file.yaml"))
        self.assertEqual(result, [])


class TestPythonParsing(unittest.TestCase):
    """Tests for Python file parsing."""

    def test_parse_runner_cli_valid(self):
        """Parse CLI arguments from a Python script."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(textwrap.dedent('''
                import argparse

                parser = argparse.ArgumentParser(description="Test CLI")
                parser.add_argument("--name", required=True, type=str, help="The name")
                parser.add_argument("--count", required=False, type=int, default=10, help="Count")
            '''))
            f.flush()

            result = parse_runner_cli(Path(f.name))
            self.assertEqual(len(result["args"]), 2)
            
            # Find the --name arg
            name_arg = next(a for a in result["args"] if "--name" in a.get("names", []))
            self.assertTrue(name_arg["required"])
            self.assertEqual(name_arg["help"], "The name")

    def test_parse_runner_cli_missing_file(self):
        """Handle missing CLI file gracefully."""
        result = parse_runner_cli(Path("/nonexistent/file.py"))
        self.assertEqual(result["args"], [])

    def test_parse_metrics_types_valid(self):
        """Parse metric function definitions from a Python script."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(textwrap.dedent('''
                from typing import Tuple

                def compute_test_metric(value: int, threshold: int) -> Tuple[bool, float]:
                    """Compute a test metric."""
                    return value >= threshold, float(value)
            '''))
            f.flush()

            result = parse_metrics_types(Path(f.name))
            self.assertIn("compute_test_metric", result)
            self.assertEqual(result["compute_test_metric"]["params"][0]["name"], "value")

    def test_parse_metrics_types_missing_file(self):
        """Handle missing metrics file gracefully."""
        result = parse_metrics_types(Path("/nonexistent/file.py"))
        self.assertEqual(result, {})


class TestContentGeneration(unittest.TestCase):
    """Tests for content generation functions."""

    def test_generate_slice_table_md(self):
        """Generate a Markdown slice table."""
        curriculum = {
            "slices": {
                "slice_a": {
                    "description": "First slice",
                    "items": ["a", "b"],
                    "prereg_hash": "hash123456789012345",
                },
                "slice_b": {
                    "description": "Second slice",
                    "items": ["c"],
                    "prereg_hash": "hash999",
                },
            }
        }

        result = generate_slice_table_md(curriculum)
        
        # Verify header
        self.assertIn("| Slice Name | Description | Items Count | Prereg Hash |", result)
        
        # Verify slice_a row
        self.assertIn("`slice_a`", result)
        self.assertIn("First slice", result)
        self.assertIn("| 2 |", result)
        
        # Verify slice_b row
        self.assertIn("`slice_b`", result)
        self.assertIn("| 1 |", result)

    def test_generate_slice_table_md_empty(self):
        """Handle empty curriculum gracefully."""
        result = generate_slice_table_md({"slices": {}})
        # Should still have header
        self.assertIn("| Slice Name | Description |", result)

    def test_generate_prereg_table_md(self):
        """Generate a Markdown prereg table."""
        prereg = [
            {
                "experiment_id": "EXP_001",
                "description": "Test experiment",
                "slice_config": "config.json",
                "success_metrics": ["metric_a", "metric_b"],
            }
        ]

        result = generate_prereg_table_md(prereg)
        self.assertIn("`EXP_001`", result)
        self.assertIn("metric_a, metric_b", result)

    def test_generate_cli_examples_md(self):
        """Generate CLI usage examples."""
        cli_info = {
            "args": [
                {"names": ["--name"], "required": True, "type": "str", "help": "Name"},
                {"names": ["--count"], "required": False, "default": 10, "type": "int", "help": "Count"},
            ],
            "description": "Test CLI",
        }

        result = generate_cli_examples_md(cli_info)
        self.assertIn("```bash", result)
        self.assertIn("| `--name` | str | Yes |", result)
        self.assertIn("| `--count` | int | No | `10` |", result)

    def test_generate_metrics_table_md(self):
        """Generate a Markdown metrics table."""
        metrics = {
            "compute_test": {
                "docstring": "Compute a test metric.\n\nMore details.",
                "params": [{"name": "value"}, {"name": "threshold"}],
                "return_type": "Tuple[bool, float]",
            }
        }

        result = generate_metrics_table_md(metrics)
        self.assertIn("`compute_test`", result)
        self.assertIn("`value, threshold`", result)
        self.assertIn("Compute a test metric.", result)

    def test_generate_status_table_md(self):
        """Generate a status table."""
        curriculum = {"slices": {"a": {}, "b": {}}}
        prereg = [{"id": 1}]

        result = generate_status_table_md(curriculum, prereg)
        self.assertIn("✅ 2 slices defined", result)
        self.assertIn("✅ 1 experiments", result)


class TestLatexGeneration(unittest.TestCase):
    """Tests for LaTeX content generation."""

    def test_escape_latex(self):
        """Escape LaTeX special characters."""
        self.assertEqual(_escape_latex("test_name"), "test\\_name")
        self.assertEqual(_escape_latex("50%"), "50\\%")
        self.assertEqual(_escape_latex("a & b"), "a \\& b")

    def test_generate_slice_table_tex(self):
        """Generate a LaTeX slice table."""
        curriculum = {
            "slices": {
                "test_slice": {
                    "description": "Test slice",
                    "items": ["a", "b"],
                    "prereg_hash": "hash123",
                }
            }
        }

        result = generate_slice_table_tex(curriculum)
        self.assertIn("\\begin{table}[h]", result)
        self.assertIn("\\texttt{test_slice}", result)
        self.assertIn("\\end{table}", result)


class TestAutogenBlocks(unittest.TestCase):
    """Tests for autogen block updating."""

    def test_update_autogen_block_md(self):
        """Update a Markdown autogen block."""
        content = textwrap.dedent("""
            # Header

            Some text.

            <!-- BEGIN:AUTOGEN:TEST -->
            old content
            <!-- END:AUTOGEN:TEST -->

            More text.
        """)

        new_content = "new content here"
        result, changed = update_autogen_block(content, "TEST", new_content, is_tex=False)

        self.assertTrue(changed)
        self.assertIn("new content here", result)
        self.assertNotIn("old content", result)
        self.assertIn("# Header", result)
        self.assertIn("More text.", result)

    def test_update_autogen_block_tex(self):
        """Update a TeX autogen block."""
        content = textwrap.dedent("""
            \\section{Test}

            % BEGIN:AUTOGEN:TEST
            old content
            % END:AUTOGEN:TEST

            \\section{Next}
        """)

        new_content = "new content here"
        result, changed = update_autogen_block(content, "TEST", new_content, is_tex=True)

        self.assertTrue(changed)
        self.assertIn("new content here", result)
        self.assertNotIn("old content", result)

    def test_update_autogen_block_no_change(self):
        """Don't report change if content is identical."""
        content = textwrap.dedent("""
            <!-- BEGIN:AUTOGEN:TEST -->
            existing content
            <!-- END:AUTOGEN:TEST -->
        """)

        result, changed = update_autogen_block(content, "TEST", "existing content", is_tex=False)
        self.assertFalse(changed)

    def test_update_autogen_block_missing_markers(self):
        """Handle missing markers gracefully."""
        content = "No markers here"

        result, changed = update_autogen_block(content, "TEST", "new content", is_tex=False)

        self.assertFalse(changed)
        self.assertEqual(result, content)


class TestDeterminism(unittest.TestCase):
    """Tests for deterministic output."""

    def test_slice_table_is_sorted(self):
        """Verify slice table output is sorted alphabetically."""
        curriculum = {
            "slices": {
                "zz_slice": {"description": "Z", "items": [], "prereg_hash": "z"},
                "aa_slice": {"description": "A", "items": [], "prereg_hash": "a"},
                "mm_slice": {"description": "M", "items": [], "prereg_hash": "m"},
            }
        }

        result = generate_slice_table_md(curriculum)
        lines = result.split("\n")
        
        # Find the data lines (after header)
        data_lines = [l for l in lines if l.startswith("| `")]
        
        # Verify order
        self.assertIn("aa_slice", data_lines[0])
        self.assertIn("mm_slice", data_lines[1])
        self.assertIn("zz_slice", data_lines[2])

    def test_multiple_runs_produce_same_output(self):
        """Running generation multiple times produces identical output."""
        curriculum = {"slices": {"slice_a": {"description": "A", "items": ["x"], "prereg_hash": "h"}}}
        
        results = [generate_slice_table_md(curriculum) for _ in range(10)]
        
        self.assertTrue(all(r == results[0] for r in results))


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
