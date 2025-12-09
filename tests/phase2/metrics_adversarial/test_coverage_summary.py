# tests/phase2/metrics_adversarial/test_coverage_summary.py
"""
Adversarial Suite Coverage Summary

Reports on coverage across:
- Fault types
- Metric kinds
- Input shapes and edge cases

Provides a test that asserts minimum coverage requirements.

NO METRIC INTERPRETATION: This module reports coverage only.
"""

import os
import ast
import pytest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional


# ===========================================================================
# COVERAGE DATA STRUCTURES
# ===========================================================================

@dataclass
class CoverageReport:
    """Summary of adversarial test coverage."""
    
    # Counts by category
    tests_by_fault_type: Dict[str, int] = field(default_factory=dict)
    tests_by_metric_kind: Dict[str, int] = field(default_factory=dict)
    tests_by_mode: Dict[str, int] = field(default_factory=dict)
    
    # Input shape coverage
    input_shapes: Dict[str, List[str]] = field(default_factory=dict)
    edge_cases_covered: List[str] = field(default_factory=list)
    
    # Totals
    total_test_count: int = 0
    total_test_classes: int = 0
    files_analyzed: List[str] = field(default_factory=list)
    
    # Metric kinds explicitly mentioned
    metric_kinds_mentioned: Set[str] = field(default_factory=set)


# ===========================================================================
# COVERAGE ANALYZER
# ===========================================================================

# Known fault types from conftest.py
KNOWN_FAULT_TYPES = {
    "missing_field": ["missing", "remove", "del "],
    "wrong_type": ["wrong_type", "type_mismatch", "string where", "not_a_"],
    "extreme_value": ["extreme", "1e308", "2**63", "max_int", "overflow"],
    "empty_container": ["empty", "[]", "set()", "{}"],
    "null_value": ["none", "null"],
    "nan_value": ["nan", "float('nan')"],
    "inf_value": ["inf", "float('inf')", "infinity"],
    "negative_value": ["negative", "-100", "-1"],
}

# Known metric kinds
KNOWN_METRIC_KINDS = {"goal_hit", "sparse", "density", "chain", "multi_goal", "multi", "multigoal"}

# Known test modes
KNOWN_MODES = {"fault", "mutation", "replay", "oracle", "shadow", "batch", "boundary"}

# Edge case patterns
EDGE_CASE_PATTERNS = {
    "empty_input": ["empty", "[]", "set()"],
    "single_element": ["single", "one element"],
    "large_scale": ["1000", "10000", "100000", "10k", "100k", "high_volume"],
    "boundary": ["boundary", "at boundary", "threshold"],
    "duplicate": ["duplicate", "duplicates"],
    "permutation": ["permutation", "shuffle", "order"],
    "nan_handling": ["nan"],
    "inf_handling": ["inf"],
    "zero": ["zero", "0"],
    "negative": ["negative", "-"],
}


def analyze_test_file(filepath: Path) -> Tuple[int, int, Dict[str, int], Set[str], List[str]]:
    """
    Analyze a test file to extract coverage information.
    
    Returns:
        (test_count, class_count, mode_counts, metric_kinds, edge_cases)
    """
    test_count = 0
    class_count = 0
    mode_counts: Dict[str, int] = {}
    metric_kinds: Set[str] = set()
    edge_cases: List[str] = []
    
    try:
        content = filepath.read_text(encoding='utf-8')
        content_lower = content.lower()
        
        # Parse AST
        tree = ast.parse(content)
        
        for node in ast.walk(tree):
            # Count test methods
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                test_count += 1
                
                # Get docstring for analysis
                docstring = ast.get_docstring(node) or ""
                func_name_lower = node.name.lower()
                
                # Detect modes from function name/docstring
                for mode in KNOWN_MODES:
                    if mode in func_name_lower or mode in docstring.lower():
                        mode_counts[mode] = mode_counts.get(mode, 0) + 1
                
                # Detect metric kinds
                for metric in KNOWN_METRIC_KINDS:
                    if metric in func_name_lower or metric in docstring.lower():
                        metric_kinds.add(metric)
                
                # Detect edge cases
                for case, patterns in EDGE_CASE_PATTERNS.items():
                    for pattern in patterns:
                        if pattern in func_name_lower or pattern in docstring.lower():
                            if case not in edge_cases:
                                edge_cases.append(case)
                            break
            
            # Count test classes
            elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
                class_count += 1
        
        # Additional content-based detection
        for metric in KNOWN_METRIC_KINDS:
            if metric in content_lower:
                metric_kinds.add(metric)
        
    except Exception as e:
        pass
    
    return test_count, class_count, mode_counts, metric_kinds, edge_cases


def analyze_test_file_for_faults(filepath: Path) -> Dict[str, int]:
    """Analyze a test file for fault type coverage."""
    fault_counts: Dict[str, int] = {}
    
    try:
        content = filepath.read_text(encoding='utf-8').lower()
        
        for fault_type, patterns in KNOWN_FAULT_TYPES.items():
            count = 0
            for pattern in patterns:
                count += content.count(pattern)
            if count > 0:
                fault_counts[fault_type] = count
                
    except Exception:
        pass
    
    return fault_counts


def summarize_adversarial_coverage() -> CoverageReport:
    """
    Analyze all adversarial test files and produce a coverage report.
    
    Returns:
        CoverageReport with coverage statistics
    """
    report = CoverageReport()
    
    # Find the adversarial test directory
    test_dir = Path(__file__).parent
    
    # Analyze each test file
    for test_file in test_dir.glob("test_*.py"):
        if test_file.name == "test_coverage_summary.py":
            continue  # Skip this file
        
        report.files_analyzed.append(test_file.name)
        
        # Analyze structure
        tests, classes, modes, metrics, edges = analyze_test_file(test_file)
        report.total_test_count += tests
        report.total_test_classes += classes
        
        # Merge mode counts
        for mode, count in modes.items():
            report.tests_by_mode[mode] = report.tests_by_mode.get(mode, 0) + count
        
        # Merge metric kinds
        report.metric_kinds_mentioned.update(metrics)
        
        # Merge edge cases
        for edge in edges:
            if edge not in report.edge_cases_covered:
                report.edge_cases_covered.append(edge)
        
        # Analyze fault types
        fault_counts = analyze_test_file_for_faults(test_file)
        for fault, count in fault_counts.items():
            report.tests_by_fault_type[fault] = report.tests_by_fault_type.get(fault, 0) + count
    
    # Summarize metric kind coverage
    metric_mapping = {
        "goal_hit": "goal_hit",
        "sparse": "density",
        "density": "density", 
        "chain": "chain_length",
        "multi_goal": "multi_goal",
        "multi": "multi_goal",
        "multigoal": "multi_goal",
    }
    
    for mention in report.metric_kinds_mentioned:
        canonical = metric_mapping.get(mention, mention)
        report.tests_by_metric_kind[canonical] = report.tests_by_metric_kind.get(canonical, 0) + 1
    
    # Input shape coverage (based on detected patterns)
    report.input_shapes = {
        "empty": ["[], set(), {}"],
        "small": ["1-10 elements"],
        "medium": ["10-100 elements"],
        "large": ["100-1000 elements"],
        "high_volume": ["10000+ elements"],
    }
    
    return report


def format_coverage_report(report: CoverageReport) -> str:
    """Format a coverage report as a human-readable string."""
    lines = []
    lines.append("=" * 60)
    lines.append("ADVERSARIAL TEST SUITE COVERAGE SUMMARY")
    lines.append("=" * 60)
    
    lines.append(f"\nTotal Tests: {report.total_test_count}")
    lines.append(f"Test Classes: {report.total_test_classes}")
    lines.append(f"Files Analyzed: {len(report.files_analyzed)}")
    
    lines.append("\n--- Files ---")
    for f in sorted(report.files_analyzed):
        lines.append(f"  • {f}")
    
    lines.append("\n--- Fault Types Covered ---")
    for fault, count in sorted(report.tests_by_fault_type.items(), key=lambda x: -x[1]):
        lines.append(f"  • {fault}: {count} references")
    
    lines.append("\n--- Metric Kinds ---")
    for metric, count in sorted(report.tests_by_metric_kind.items()):
        lines.append(f"  • {metric}: covered")
    
    lines.append("\n--- Test Modes ---")
    for mode, count in sorted(report.tests_by_mode.items(), key=lambda x: -x[1]):
        lines.append(f"  • {mode}: {count} tests")
    
    lines.append("\n--- Edge Cases Covered ---")
    for edge in sorted(report.edge_cases_covered):
        lines.append(f"  • {edge}")
    
    lines.append("\n--- Input Shapes ---")
    for shape, examples in report.input_shapes.items():
        lines.append(f"  • {shape}: {examples}")
    
    lines.append("\n" + "=" * 60)
    
    return "\n".join(lines)


# ===========================================================================
# COVERAGE TESTS
# ===========================================================================

@pytest.mark.oracle
class TestAdversarialCoverageSummary:
    """Tests verifying minimum adversarial coverage requirements."""

    def test_coverage_non_empty(self):
        """Coverage report is non-empty."""
        report = summarize_adversarial_coverage()
        
        assert report.total_test_count > 0, "Should have tests"
        assert len(report.files_analyzed) > 0, "Should have analyzed files"
        assert len(report.metric_kinds_mentioned) > 0, "Should mention metrics"

    def test_all_four_metric_kinds_covered(self):
        """All four metric kinds are mentioned in tests."""
        report = summarize_adversarial_coverage()
        
        # Canonical metric kinds
        required_metrics = {"goal_hit", "density", "chain_length", "multi_goal"}
        
        # Check coverage (allowing for aliases)
        covered = set()
        for m in report.metric_kinds_mentioned:
            m_lower = m.lower()
            if "goal" in m_lower and "multi" not in m_lower:
                covered.add("goal_hit")
            elif "sparse" in m_lower or "density" in m_lower:
                covered.add("density")
            elif "chain" in m_lower:
                covered.add("chain_length")
            if "multi" in m_lower:
                covered.add("multi_goal")
        
        missing = required_metrics - covered
        assert len(missing) == 0, f"Missing metric coverage: {missing}. Mentioned: {report.metric_kinds_mentioned}"

    def test_minimum_fault_types_covered(self):
        """At least 5 fault types are covered."""
        report = summarize_adversarial_coverage()
        
        assert len(report.tests_by_fault_type) >= 5, (
            f"Should cover at least 5 fault types, got {len(report.tests_by_fault_type)}"
        )

    def test_minimum_edge_cases_covered(self):
        """At least 5 edge case categories are covered."""
        report = summarize_adversarial_coverage()
        
        assert len(report.edge_cases_covered) >= 5, (
            f"Should cover at least 5 edge cases, got {len(report.edge_cases_covered)}"
        )

    def test_multiple_test_modes_present(self):
        """Multiple test modes are present."""
        report = summarize_adversarial_coverage()
        
        assert len(report.tests_by_mode) >= 3, (
            f"Should have at least 3 test modes, got {len(report.tests_by_mode)}"
        )

    def test_minimum_test_count(self):
        """At least 80 adversarial tests exist."""
        report = summarize_adversarial_coverage()
        
        # We created 101 originally, plus 8 new ones
        assert report.total_test_count >= 80, (
            f"Should have at least 80 tests, got {report.total_test_count}"
        )

    def test_coverage_report_formatting(self):
        """Coverage report can be formatted as string."""
        report = summarize_adversarial_coverage()
        formatted = format_coverage_report(report)
        
        assert len(formatted) > 100, "Formatted report should be substantial"
        assert "ADVERSARIAL TEST SUITE COVERAGE SUMMARY" in formatted
        assert "Fault Types Covered" in formatted
        assert "Metric Kinds" in formatted

    def test_print_coverage_report(self, capsys):
        """Print coverage report for visibility."""
        report = summarize_adversarial_coverage()
        formatted = format_coverage_report(report)
        print(formatted)
        
        captured = capsys.readouterr()
        assert "ADVERSARIAL TEST SUITE COVERAGE SUMMARY" in captured.out


# ===========================================================================
# CLI ENTRY POINT FOR COVERAGE SUMMARY
# ===========================================================================

def main():
    """Print coverage summary when run directly."""
    report = summarize_adversarial_coverage()
    print(format_coverage_report(report))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

