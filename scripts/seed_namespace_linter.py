#!/usr/bin/env python3
# PHASE II — NOT USED IN PHASE I
"""
Seed Namespace Linter — Static Analysis for PRNG Namespace Collisions.

This tool performs source-level analysis to detect potential PRNG namespace
issues including:

1. Duplicate namespace paths across unrelated modules (potential collision)
2. Hard-coded seeds (should use hierarchical derivation instead)
3. Suspicious patterns (global seeds, non-constant paths)

The linter does NOT import any project modules; it uses pure AST analysis.

Exit Codes:
    0 - No issues found
    1 - Suspicious or conflicting namespaces found
    2 - Error during analysis

Usage:
    python scripts/seed_namespace_linter.py
    python scripts/seed_namespace_linter.py --json
    python scripts/seed_namespace_linter.py rfl/ experiments/
    python scripts/seed_namespace_linter.py --strict

Suppression:
    Add `# prng: namespace-ok` comment to suppress a specific usage.

Author: Agent A2 (runtime-ops-2)
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Project root for default scanning
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class NamespaceUsage:
    """A single usage of a PRNG namespace."""
    file_path: str
    line_number: int
    namespace_path: str
    call_type: str  # 'for_path', 'seed_for_path', 'alloc_seed', etc.
    is_constant: bool  # Whether the path is a constant string
    suppressed: bool = False
    raw_code: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file": self.file_path,
            "line": self.line_number,
            "namespace": self.namespace_path,
            "call_type": self.call_type,
            "is_constant": self.is_constant,
            "suppressed": self.suppressed,
        }


@dataclass
class HardCodedSeed:
    """A hard-coded seed value."""
    file_path: str
    line_number: int
    seed_value: str
    context: str  # 'random.seed', 'np.random.seed', 'DeterministicPRNG', etc.
    suppressed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file": self.file_path,
            "line": self.line_number,
            "seed_value": self.seed_value,
            "context": self.context,
            "suppressed": self.suppressed,
        }


@dataclass
class LintResult:
    """Result of namespace linting."""
    files_scanned: int
    namespaces_found: int
    duplicates: List[Tuple[str, List[NamespaceUsage]]]  # namespace -> usages
    hard_coded_seeds: List[HardCodedSeed]
    dynamic_paths: List[NamespaceUsage]  # Non-constant namespace paths
    issues_count: int
    suppressed_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "files_scanned": self.files_scanned,
            "namespaces_found": self.namespaces_found,
            "issues_count": self.issues_count,
            "suppressed_count": self.suppressed_count,
            "duplicates": [
                {"namespace": ns, "usages": [u.to_dict() for u in usages]}
                for ns, usages in self.duplicates
            ],
            "hard_coded_seeds": [s.to_dict() for s in self.hard_coded_seeds],
            "dynamic_paths": [p.to_dict() for p in self.dynamic_paths],
        }


class PRNGNamespaceVisitor(ast.NodeVisitor):
    """AST visitor to find PRNG namespace usages."""

    # PRNG API method names to look for
    PRNG_METHODS = {
        'for_path', 'for_numpy', 'for_numpy_legacy',
        'seed_for_path', 'alloc_seed', 'record',
        'generate_seed_schedule',
    }

    # Hard-coded seed patterns
    SEED_PATTERNS = {
        'seed': {'random.seed', 'np.random.seed', 'numpy.random.seed'},
        'init': {'DeterministicPRNG', 'Random', 'RandomState'},
    }

    def __init__(self, source_lines: List[str], file_path: str):
        self.source_lines = source_lines
        self.file_path = file_path
        self.namespace_usages: List[NamespaceUsage] = []
        self.hard_coded_seeds: List[HardCodedSeed] = []
        self.suppressed_lines: Set[int] = set()

        # Find suppressed lines (comment on same line or line before)
        for i, line in enumerate(source_lines, start=1):
            if '# prng: namespace-ok' in line.lower():
                self.suppressed_lines.add(i)
                # Also suppress the next line (common pattern: comment above code)
                self.suppressed_lines.add(i + 1)

    def _is_suppressed(self, lineno: int) -> bool:
        """Check if a line is suppressed."""
        return lineno in self.suppressed_lines

    def _extract_string_value(self, node: ast.expr) -> Tuple[Optional[str], bool]:
        """
        Extract string value from an AST node.

        Returns:
            Tuple of (value, is_constant)
            - value: The string value or None if not extractable
            - is_constant: Whether it's a literal constant
        """
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value, True
        elif isinstance(node, ast.Str):  # Python 3.7 compat
            return node.s, True
        elif isinstance(node, ast.JoinedStr):  # f-string
            # Try to extract parts we can
            parts = []
            has_dynamic = False
            for value in node.values:
                if isinstance(value, ast.Constant):
                    parts.append(str(value.value))
                elif isinstance(value, ast.Str):
                    parts.append(value.s)
                else:
                    parts.append("<dynamic>")
                    has_dynamic = True
            return "".join(parts), not has_dynamic
        elif isinstance(node, ast.Name):
            return f"<var:{node.id}>", False
        elif isinstance(node, ast.Attribute):
            return f"<attr:{ast.unparse(node) if hasattr(ast, 'unparse') else '?'}>", False
        return None, False

    def _extract_path_args(self, call: ast.Call) -> Tuple[List[str], bool]:
        """
        Extract path arguments from a PRNG call.

        Returns:
            Tuple of (path_parts, is_all_constant)
        """
        parts = []
        all_constant = True

        for arg in call.args:
            value, is_const = self._extract_string_value(arg)
            if value:
                parts.append(value)
                if not is_const:
                    all_constant = False
            else:
                parts.append("<unknown>")
                all_constant = False

        return parts, all_constant

    def _get_raw_code(self, lineno: int) -> str:
        """Get the raw source line."""
        if 1 <= lineno <= len(self.source_lines):
            return self.source_lines[lineno - 1].strip()
        return ""

    def visit_Call(self, node: ast.Call) -> None:
        """Visit function calls to find PRNG API usage."""
        # Check for method calls like prng.for_path(...)
        if isinstance(node.func, ast.Attribute):
            method_name = node.func.attr

            if method_name in self.PRNG_METHODS:
                path_parts, is_constant = self._extract_path_args(node)
                namespace = "::".join(path_parts) if path_parts else "<empty>"

                usage = NamespaceUsage(
                    file_path=self.file_path,
                    line_number=node.lineno,
                    namespace_path=namespace,
                    call_type=method_name,
                    is_constant=is_constant,
                    suppressed=self._is_suppressed(node.lineno),
                    raw_code=self._get_raw_code(node.lineno),
                )
                self.namespace_usages.append(usage)

            # Check for hard-coded seeds: random.seed(...)
            if isinstance(node.func.value, ast.Name):
                caller = node.func.value.id
                if caller == 'random' and method_name == 'seed' and node.args:
                    seed_arg = node.args[0]
                    if isinstance(seed_arg, ast.Constant):
                        self.hard_coded_seeds.append(HardCodedSeed(
                            file_path=self.file_path,
                            line_number=node.lineno,
                            seed_value=str(seed_arg.value),
                            context=f"{caller}.{method_name}",
                            suppressed=self._is_suppressed(node.lineno),
                        ))

            # Check for np.random.seed(...) or numpy.random.seed(...)
            if isinstance(node.func.value, ast.Attribute):
                if (isinstance(node.func.value.value, ast.Name) and
                    node.func.value.value.id in ('np', 'numpy') and
                    node.func.value.attr == 'random' and
                    method_name == 'seed' and node.args):
                    seed_arg = node.args[0]
                    if isinstance(seed_arg, ast.Constant):
                        self.hard_coded_seeds.append(HardCodedSeed(
                            file_path=self.file_path,
                            line_number=node.lineno,
                            seed_value=str(seed_arg.value),
                            context=f"{node.func.value.value.id}.random.{method_name}",
                            suppressed=self._is_suppressed(node.lineno),
                        ))

        # Check for DeterministicPRNG initialization with hard-coded seed
        elif isinstance(node.func, ast.Name):
            if node.func.id == 'DeterministicPRNG' and node.args:
                seed_arg = node.args[0]
                if isinstance(seed_arg, ast.Constant) and isinstance(seed_arg.value, str):
                    # Check if it's a literal hex string (64 chars)
                    if len(seed_arg.value) == 64:
                        self.hard_coded_seeds.append(HardCodedSeed(
                            file_path=self.file_path,
                            line_number=node.lineno,
                            seed_value=seed_arg.value[:16] + "...",
                            context="DeterministicPRNG",
                            suppressed=self._is_suppressed(node.lineno),
                        ))

        self.generic_visit(node)


def scan_file(file_path: Path) -> Tuple[List[NamespaceUsage], List[HardCodedSeed]]:
    """
    Scan a single Python file for PRNG namespace usage.

    Returns:
        Tuple of (namespace_usages, hard_coded_seeds)
    """
    try:
        # Try UTF-8 first, then fall back to latin-1
        try:
            source = file_path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            source = file_path.read_text(encoding='latin-1')

        source_lines = source.splitlines()
        tree = ast.parse(source)

        visitor = PRNGNamespaceVisitor(source_lines, str(file_path))
        visitor.visit(tree)

        return visitor.namespace_usages, visitor.hard_coded_seeds

    except SyntaxError:
        # Can't parse, skip
        return [], []
    except Exception:
        return [], []


def find_python_files(paths: List[Path]) -> List[Path]:
    """Find all Python files in the given paths."""
    files = []

    for path in paths:
        if path.is_file() and path.suffix == '.py':
            files.append(path)
        elif path.is_dir():
            files.extend(path.rglob('*.py'))

    # Filter out __pycache__
    files = [f for f in files if '__pycache__' not in str(f)]

    return sorted(files)


def analyze_namespaces(
    all_usages: List[NamespaceUsage],
    strict: bool = False,
) -> Tuple[List[Tuple[str, List[NamespaceUsage]]], List[NamespaceUsage]]:
    """
    Analyze namespace usages for duplicates and dynamic paths.

    Args:
        all_usages: All namespace usages found.
        strict: If True, treat all duplicates as issues.

    Returns:
        Tuple of (duplicates, dynamic_paths)
    """
    # Group by namespace
    by_namespace: Dict[str, List[NamespaceUsage]] = defaultdict(list)
    for usage in all_usages:
        if not usage.suppressed:
            by_namespace[usage.namespace_path].append(usage)

    # Find duplicates (same namespace in different files)
    duplicates = []
    for namespace, usages in by_namespace.items():
        if len(usages) > 1:
            # Check if they're in different files
            files = set(u.file_path for u in usages)
            if len(files) > 1 or strict:
                duplicates.append((namespace, usages))

    # Find dynamic paths (non-constant)
    dynamic_paths = [u for u in all_usages if not u.is_constant and not u.suppressed]

    return duplicates, dynamic_paths


def lint_namespaces(
    paths: Optional[List[Path]] = None,
    strict: bool = False,
    verbose: bool = False,
) -> LintResult:
    """
    Lint PRNG namespaces in the given paths.

    Args:
        paths: Paths to scan. Defaults to rfl/ and experiments/.
        strict: If True, treat all duplicates as issues.
        verbose: If True, print progress.

    Returns:
        LintResult with analysis.
    """
    if paths is None:
        paths = [
            PROJECT_ROOT / 'rfl',
            PROJECT_ROOT / 'experiments',
        ]

    files = find_python_files(paths)

    if verbose:
        print(f"Scanning {len(files)} Python files...")

    all_usages: List[NamespaceUsage] = []
    all_hard_coded: List[HardCodedSeed] = []

    for file_path in files:
        usages, hard_coded = scan_file(file_path)
        all_usages.extend(usages)
        all_hard_coded.extend(hard_coded)

    duplicates, dynamic_paths = analyze_namespaces(all_usages, strict)

    # Count non-suppressed issues
    hard_coded_issues = [s for s in all_hard_coded if not s.suppressed]
    suppressed_count = (
        sum(1 for u in all_usages if u.suppressed) +
        sum(1 for s in all_hard_coded if s.suppressed)
    )

    issues_count = len(duplicates) + len(hard_coded_issues) + len(dynamic_paths)

    return LintResult(
        files_scanned=len(files),
        namespaces_found=len(all_usages),
        duplicates=duplicates,
        hard_coded_seeds=hard_coded_issues,
        dynamic_paths=dynamic_paths,
        issues_count=issues_count,
        suppressed_count=suppressed_count,
    )


def format_human_output(result: LintResult) -> str:
    """Format result for human-readable output."""
    lines = []
    lines.append("=" * 70)
    lines.append("Seed Namespace Linter — Agent A2 (runtime-ops-2)")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Files scanned: {result.files_scanned}")
    lines.append(f"Namespaces found: {result.namespaces_found}")
    lines.append(f"Issues found: {result.issues_count}")
    lines.append(f"Suppressed: {result.suppressed_count}")
    lines.append("")

    if result.duplicates:
        lines.append("-" * 70)
        lines.append("⚠ DUPLICATE NAMESPACES (potential collisions)")
        lines.append("-" * 70)
        for namespace, usages in result.duplicates:
            lines.append(f"\n  Namespace: {namespace}")
            for usage in usages:
                lines.append(f"    - {usage.file_path}:{usage.line_number}")
                lines.append(f"      {usage.raw_code}")
        lines.append("")

    if result.hard_coded_seeds:
        lines.append("-" * 70)
        lines.append("⚠ HARD-CODED SEEDS (should use hierarchical derivation)")
        lines.append("-" * 70)
        for seed in result.hard_coded_seeds:
            lines.append(f"  {seed.file_path}:{seed.line_number}")
            lines.append(f"    Context: {seed.context}")
            lines.append(f"    Value: {seed.seed_value}")
        lines.append("")

    if result.dynamic_paths:
        lines.append("-" * 70)
        lines.append("ℹ DYNAMIC NAMESPACE PATHS (may be intentional)")
        lines.append("-" * 70)
        for usage in result.dynamic_paths[:10]:  # Limit output
            lines.append(f"  {usage.file_path}:{usage.line_number}")
            lines.append(f"    Path: {usage.namespace_path}")
        if len(result.dynamic_paths) > 10:
            lines.append(f"  ... and {len(result.dynamic_paths) - 10} more")
        lines.append("")

    lines.append("=" * 70)
    if result.issues_count == 0:
        lines.append("✅ No namespace issues found")
    else:
        lines.append(f"❌ Found {result.issues_count} issue(s)")
    lines.append("=" * 70)

    return "\n".join(lines)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Seed Namespace Linter — Static Analysis for PRNG Collisions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exit Codes:
    0 - No issues found
    1 - Suspicious or conflicting namespaces found
    2 - Error during analysis

Suppression:
    Add `# prng: namespace-ok` to suppress a specific line.

Examples:
    python scripts/seed_namespace_linter.py
    python scripts/seed_namespace_linter.py --json
    python scripts/seed_namespace_linter.py rfl/ experiments/
    python scripts/seed_namespace_linter.py --strict
        """,
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Paths to scan (default: rfl/, experiments/)",
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output JSON format",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat all duplicates as issues (even within same file)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Write output to file",
    )

    args = parser.parse_args()

    try:
        paths = args.paths if args.paths else None
        result = lint_namespaces(paths, args.strict, args.verbose)

        if args.json:
            output = json.dumps(result.to_dict(), indent=2)
        else:
            output = format_human_output(result)

        if args.output:
            args.output.write_text(output)
            print(f"Output written to: {args.output}")
        else:
            print(output)

        return 0 if result.issues_count == 0 else 1

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())

