# PHASE II — NOT USED IN PHASE I
"""
Tests for Seed Namespace Linter.

Verifies:
- Duplicate namespace detection across files
- Hard-coded seed detection
- Dynamic path detection
- Suppression via `# prng: namespace-ok` comment
"""

import ast
import tempfile
import pytest
from pathlib import Path
from typing import List

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.seed_namespace_linter import (
    scan_file,
    find_python_files,
    analyze_namespaces,
    lint_namespaces,
    NamespaceUsage,
    HardCodedSeed,
    LintResult,
)


class TestScanFile:
    """Tests for single file scanning."""

    def test_detect_for_path_call(self, tmp_path: Path):
        """Detects prng.for_path() calls."""
        code = '''
from rfl.prng import DeterministicPRNG
prng = DeterministicPRNG("a" * 64)
rng = prng.for_path("slice_a", "baseline", "cycle_0001")
'''
        file_path = tmp_path / "test.py"
        file_path.write_text(code)

        usages, seeds = scan_file(file_path)

        assert len(usages) == 1
        assert usages[0].namespace_path == "slice_a::baseline::cycle_0001"
        assert usages[0].call_type == "for_path"
        assert usages[0].is_constant is True

    def test_detect_seed_for_path_call(self, tmp_path: Path):
        """Detects prng.seed_for_path() calls."""
        code = '''
seed = prng.seed_for_path("test", "mode", "cycle")
'''
        file_path = tmp_path / "test.py"
        file_path.write_text(code)

        usages, seeds = scan_file(file_path)

        assert len(usages) == 1
        assert usages[0].namespace_path == "test::mode::cycle"
        assert usages[0].call_type == "seed_for_path"

    def test_detect_dynamic_path(self, tmp_path: Path):
        """Detects non-constant path arguments."""
        code = '''
cycle_name = f"cycle_{i:04d}"
rng = prng.for_path("slice", mode, cycle_name)
'''
        file_path = tmp_path / "test.py"
        file_path.write_text(code)

        usages, seeds = scan_file(file_path)

        assert len(usages) == 1
        assert usages[0].is_constant is False
        assert "<var:mode>" in usages[0].namespace_path

    def test_detect_hard_coded_random_seed(self, tmp_path: Path):
        """Detects hard-coded random.seed() calls."""
        code = '''
import random
random.seed(12345)
'''
        file_path = tmp_path / "test.py"
        file_path.write_text(code)

        usages, seeds = scan_file(file_path)

        assert len(seeds) == 1
        assert seeds[0].seed_value == "12345"
        assert seeds[0].context == "random.seed"

    def test_detect_hard_coded_numpy_seed(self, tmp_path: Path):
        """Detects hard-coded np.random.seed() calls."""
        code = '''
import numpy as np
np.random.seed(42)
'''
        file_path = tmp_path / "test.py"
        file_path.write_text(code)

        usages, seeds = scan_file(file_path)

        assert len(seeds) == 1
        assert seeds[0].seed_value == "42"
        assert seeds[0].context == "np.random.seed"

    def test_suppression_via_comment(self, tmp_path: Path):
        """Suppresses findings with # prng: namespace-ok comment."""
        code = '''
# This is intentional reuse: # prng: namespace-ok
rng = prng.for_path("shared", "namespace")
'''
        file_path = tmp_path / "test.py"
        file_path.write_text(code)

        usages, seeds = scan_file(file_path)

        assert len(usages) == 1
        assert usages[0].suppressed is True

    def test_multiple_calls_same_file(self, tmp_path: Path):
        """Finds multiple PRNG calls in same file."""
        code = '''
rng1 = prng.for_path("slice_a", "baseline")
rng2 = prng.for_path("slice_a", "rfl")
rng3 = prng.for_numpy("slice_b", "baseline")
'''
        file_path = tmp_path / "test.py"
        file_path.write_text(code)

        usages, seeds = scan_file(file_path)

        assert len(usages) == 3
        namespaces = {u.namespace_path for u in usages}
        assert "slice_a::baseline" in namespaces
        assert "slice_a::rfl" in namespaces
        assert "slice_b::baseline" in namespaces


class TestAnalyzeNamespaces:
    """Tests for namespace analysis."""

    def test_find_duplicates_different_files(self):
        """Detects same namespace in different files."""
        usages = [
            NamespaceUsage(
                file_path="file1.py",
                line_number=10,
                namespace_path="shared::namespace",
                call_type="for_path",
                is_constant=True,
            ),
            NamespaceUsage(
                file_path="file2.py",
                line_number=20,
                namespace_path="shared::namespace",
                call_type="for_path",
                is_constant=True,
            ),
        ]

        duplicates, dynamic = analyze_namespaces(usages)

        assert len(duplicates) == 1
        assert duplicates[0][0] == "shared::namespace"
        assert len(duplicates[0][1]) == 2

    def test_no_duplicates_same_file(self):
        """Same namespace in same file is not flagged by default."""
        usages = [
            NamespaceUsage(
                file_path="file1.py",
                line_number=10,
                namespace_path="namespace",
                call_type="for_path",
                is_constant=True,
            ),
            NamespaceUsage(
                file_path="file1.py",
                line_number=20,
                namespace_path="namespace",
                call_type="for_path",
                is_constant=True,
            ),
        ]

        duplicates, dynamic = analyze_namespaces(usages)

        assert len(duplicates) == 0

    def test_strict_mode_flags_same_file_duplicates(self):
        """Strict mode flags duplicates even in same file."""
        usages = [
            NamespaceUsage(
                file_path="file1.py",
                line_number=10,
                namespace_path="namespace",
                call_type="for_path",
                is_constant=True,
            ),
            NamespaceUsage(
                file_path="file1.py",
                line_number=20,
                namespace_path="namespace",
                call_type="for_path",
                is_constant=True,
            ),
        ]

        duplicates, dynamic = analyze_namespaces(usages, strict=True)

        assert len(duplicates) == 1

    def test_dynamic_paths_collected(self):
        """Dynamic paths are collected separately."""
        usages = [
            NamespaceUsage(
                file_path="file1.py",
                line_number=10,
                namespace_path="static::path",
                call_type="for_path",
                is_constant=True,
            ),
            NamespaceUsage(
                file_path="file1.py",
                line_number=20,
                namespace_path="dynamic::<var:x>",
                call_type="for_path",
                is_constant=False,
            ),
        ]

        duplicates, dynamic = analyze_namespaces(usages)

        assert len(dynamic) == 1
        assert dynamic[0].namespace_path == "dynamic::<var:x>"

    def test_suppressed_not_flagged(self):
        """Suppressed usages are not flagged as duplicates."""
        usages = [
            NamespaceUsage(
                file_path="file1.py",
                line_number=10,
                namespace_path="shared::namespace",
                call_type="for_path",
                is_constant=True,
                suppressed=True,  # Suppressed
            ),
            NamespaceUsage(
                file_path="file2.py",
                line_number=20,
                namespace_path="shared::namespace",
                call_type="for_path",
                is_constant=True,
            ),
        ]

        duplicates, dynamic = analyze_namespaces(usages)

        # Only one non-suppressed usage, so no duplicate
        assert len(duplicates) == 0


class TestLintNamespaces:
    """Integration tests for full linting."""

    def test_lint_synthetic_tree(self, tmp_path: Path):
        """Lint a synthetic directory tree."""
        # Create test files
        (tmp_path / "module_a.py").write_text('''
prng.for_path("shared", "namespace")
''')
        (tmp_path / "module_b.py").write_text('''
prng.for_path("shared", "namespace")  # Duplicate!
''')
        (tmp_path / "module_c.py").write_text('''
random.seed(42)  # Hard-coded seed
''')

        result = lint_namespaces([tmp_path])

        assert result.files_scanned == 3
        assert result.issues_count >= 2  # Duplicate + hard-coded seed
        assert len(result.duplicates) == 1
        assert len(result.hard_coded_seeds) == 1

    def test_lint_no_issues(self, tmp_path: Path):
        """Clean code has no issues."""
        (tmp_path / "clean.py").write_text('''
prng.for_path("unique", "namespace", "path")
''')

        result = lint_namespaces([tmp_path])

        assert result.issues_count == 0

    def test_lint_suppressed_duplicates(self, tmp_path: Path):
        """Suppressed duplicates don't count as issues."""
        (tmp_path / "module_a.py").write_text('''
prng.for_path("shared", "namespace")  # prng: namespace-ok
''')
        (tmp_path / "module_b.py").write_text('''
prng.for_path("shared", "namespace")  # prng: namespace-ok
''')

        result = lint_namespaces([tmp_path])

        assert result.issues_count == 0
        assert result.suppressed_count == 2


class TestJsonOutput:
    """Tests for JSON output format."""

    def test_result_to_dict(self):
        """LintResult serializes to dict correctly."""
        result = LintResult(
            files_scanned=10,
            namespaces_found=5,
            duplicates=[],
            hard_coded_seeds=[],
            dynamic_paths=[],
            issues_count=0,
            suppressed_count=2,
        )

        d = result.to_dict()

        assert d["files_scanned"] == 10
        assert d["namespaces_found"] == 5
        assert d["issues_count"] == 0
        assert d["suppressed_count"] == 2


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_file(self, tmp_path: Path):
        """Empty file doesn't crash."""
        (tmp_path / "empty.py").write_text("")

        usages, seeds = scan_file(tmp_path / "empty.py")

        assert usages == []
        assert seeds == []

    def test_syntax_error_file(self, tmp_path: Path):
        """Syntax error files are skipped gracefully."""
        (tmp_path / "broken.py").write_text("def broken(")

        usages, seeds = scan_file(tmp_path / "broken.py")

        assert usages == []
        assert seeds == []

    def test_non_python_files_ignored(self, tmp_path: Path):
        """Non-Python files are ignored."""
        (tmp_path / "readme.md").write_text("# README")
        (tmp_path / "data.json").write_text("{}")
        (tmp_path / "script.py").write_text("x = 1")

        files = find_python_files([tmp_path])

        assert len(files) == 1
        assert files[0].name == "script.py"

