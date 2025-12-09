"""
PHASE II — NOT USED IN PHASE I

Unit tests for experiments/drift_detection_audit.py

Tests cover:
- Semantic equivalence
- Doc → YAML consistency
- YAML → code consistency  
- Prereg → metrics consistency

20+ new tests for cross-file drift detection.
"""

import unittest
import tempfile
from pathlib import Path
from typing import Dict, Any

import yaml

from experiments.drift_detection_audit import (
    # Data structures
    DriftSeverity,
    DriftFinding,
    DocumentationFinding,
    DriftReport,
    # Documentation linter
    EXPECTED_PLAN_HEADERS,
    EXPECTED_THEORY_HEADERS,
    extract_headers_from_markdown,
    extract_slice_references,
    extract_metric_kind_references,
    extract_hash_references,
    extract_formula_references,
    lint_documentation,
    # Drift detection
    detect_metric_kind_drift,
    detect_parameter_drift,
    detect_function_drift,
    # Formula validation
    validate_formula_references,
    validate_slice_pool_completeness,
    # Reporting
    format_drift_markdown_report,
    format_drift_json_report,
    # Main
    run_drift_detection,
)

from experiments.semantic_consistency_audit import (
    MetricKind,
    DOC_SLICE_METRIC_MAP,
    METRIC_FUNCTION_MAP,
)


class TestDriftSeverityEnum(unittest.TestCase):
    """Tests for DriftSeverity enumeration."""
    
    def test_severity_values(self):
        """All severity levels should be defined."""
        self.assertEqual(DriftSeverity.OK.value, "OK")
        self.assertEqual(DriftSeverity.WARNING.value, "WARNING")
        self.assertEqual(DriftSeverity.FAIL.value, "FAIL")
    
    def test_severity_ordering(self):
        """Severity can be compared via string value."""
        self.assertNotEqual(DriftSeverity.OK, DriftSeverity.FAIL)
        self.assertNotEqual(DriftSeverity.WARNING, DriftSeverity.FAIL)


class TestDriftReportStatus(unittest.TestCase):
    """Tests for DriftReport status aggregation."""
    
    def test_empty_report_is_ok(self):
        """Empty report should have OK status."""
        report = DriftReport()
        self.assertEqual(report.status, DriftSeverity.OK)
    
    def test_warning_only_report(self):
        """Report with only warnings should have WARNING status."""
        report = DriftReport()
        report.drift_findings.append(DriftFinding(
            severity=DriftSeverity.WARNING,
            category="test",
            source_file="a.yaml",
            target_file="b.yaml",
            message="Test warning",
        ))
        self.assertEqual(report.status, DriftSeverity.WARNING)
    
    def test_fail_overrides_warning(self):
        """FAIL should override WARNING."""
        report = DriftReport()
        report.drift_findings.append(DriftFinding(
            severity=DriftSeverity.WARNING,
            category="test",
            source_file="a.yaml",
            target_file="b.yaml",
            message="Test warning",
        ))
        report.doc_findings.append(DocumentationFinding(
            severity=DriftSeverity.FAIL,
            category="test",
            file="doc.md",
            line=1,
            message="Test failure",
        ))
        self.assertEqual(report.status, DriftSeverity.FAIL)


# =============================================================================
# DOCUMENTATION LINTER TESTS
# =============================================================================

class TestMarkdownHeaderExtraction(unittest.TestCase):
    """Tests for extracting headers from markdown."""
    
    def test_extract_level_1_headers(self):
        """Should extract level 1 headers."""
        content = "# Header One\n\nSome text\n\n# Header Two"
        headers = extract_headers_from_markdown(content)
        self.assertEqual(len(headers), 2)
        self.assertEqual(headers[0], (1, "Header One", 1))
        self.assertEqual(headers[1], (1, "Header Two", 5))
    
    def test_extract_multi_level_headers(self):
        """Should extract headers of different levels."""
        content = "# H1\n## H2\n### H3\n#### H4"
        headers = extract_headers_from_markdown(content)
        self.assertEqual(len(headers), 4)
        self.assertEqual(headers[0][0], 1)
        self.assertEqual(headers[1][0], 2)
        self.assertEqual(headers[2][0], 3)
        self.assertEqual(headers[3][0], 4)
    
    def test_empty_content(self):
        """Empty content should return empty list."""
        headers = extract_headers_from_markdown("")
        self.assertEqual(headers, [])


class TestSliceReferenceExtraction(unittest.TestCase):
    """Tests for extracting slice references from markdown."""
    
    def test_extract_slice_names(self):
        """Should extract slice names in backticks."""
        content = "The `slice_uplift_goal` slice and `slice_uplift_sparse` are..."
        slices = extract_slice_references(content)
        self.assertIn("slice_uplift_goal", slices)
        self.assertIn("slice_uplift_sparse", slices)
    
    def test_no_false_positives(self):
        """Should not match non-slice patterns."""
        content = "The `some_other_thing` is not a slice."
        slices = extract_slice_references(content)
        self.assertNotIn("some_other_thing", slices)


class TestMetricKindExtraction(unittest.TestCase):
    """Tests for extracting metric kind references."""
    
    def test_extract_metric_kinds(self):
        """Should extract all metric kinds."""
        content = "We use goal_hit for the first slice, density for sparse..."
        kinds = extract_metric_kind_references(content)
        self.assertIn("goal_hit", kinds)
        self.assertIn("density", kinds)
    
    def test_case_insensitive(self):
        """Should match case-insensitively."""
        content = "GOAL_HIT and Chain_Length"
        kinds = extract_metric_kind_references(content)
        self.assertIn("goal_hit", kinds)
        self.assertIn("chain_length", kinds)


class TestHashExtraction(unittest.TestCase):
    """Tests for extracting SHA256 hashes."""
    
    def test_extract_valid_hash(self):
        """Should extract valid 64-char hex hashes."""
        hash_str = "248e2c30377c23e7a10d20d203eef09b9a136c30729ece89910908a0f36c89b1"
        content = f"The hash is {hash_str}"
        hashes = extract_hash_references(content)
        self.assertIn(hash_str, hashes)
    
    def test_no_partial_hash(self):
        """Should not extract partial hashes."""
        content = "Short hash: abc123"
        hashes = extract_hash_references(content)
        self.assertEqual(len(hashes), 0)


class TestFormulaExtraction(unittest.TestCase):
    """Tests for extracting formula references."""
    
    def test_extract_implication(self):
        """Should extract implication formulas."""
        content = "The formula `p -> q` represents..."
        formulas = extract_formula_references(content)
        self.assertIn("p -> q", formulas)
    
    def test_extract_conjunction(self):
        """Should extract conjunction formulas."""
        content = "The formula `p /\\ q` means..."
        formulas = extract_formula_references(content)
        self.assertIn("p /\\ q", formulas)
    
    def test_extract_negation(self):
        """Should extract negation formulas."""
        content = "Double negation: `~~p -> p`"
        formulas = extract_formula_references(content)
        self.assertIn("~~p -> p", formulas)


# =============================================================================
# CROSS-FILE DRIFT DETECTION TESTS
# =============================================================================

class TestMetricKindDriftDetection(unittest.TestCase):
    """Tests for metric kind drift detection."""
    
    def setUp(self):
        """Create temporary files for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.plan_path = Path(self.temp_dir) / "plan.md"
        self.curriculum_path = Path(self.temp_dir) / "curriculum.yaml"
        self.prereg_path = Path(self.temp_dir) / "prereg.yaml"
    
    def test_no_drift_when_consistent(self):
        """No drift should be detected when files are consistent."""
        # Create minimal consistent files
        self.plan_path.write_text("# Plan\n`slice_uplift_goal` uses goal_hit")
        
        curriculum = {
            "slices": {
                "slice_uplift_goal": {
                    "success_metric": {"kind": "goal_hit"}
                }
            }
        }
        with open(self.curriculum_path, "w") as f:
            yaml.dump(curriculum, f)
        
        prereg = {
            "slice_uplift_goal": {
                "success_metric": {"kind": "goal_hit"}
            }
        }
        with open(self.prereg_path, "w") as f:
            yaml.dump(prereg, f)
        
        findings = detect_metric_kind_drift(
            self.plan_path, self.curriculum_path, self.prereg_path
        )
        # Should find no FAIL-level drift for consistent slices
        fail_findings = [f for f in findings if f.severity == DriftSeverity.FAIL]
        goal_fails = [f for f in fail_findings if "slice_uplift_goal" in f.message]
        self.assertEqual(len(goal_fails), 0)
    
    def test_detect_yaml_prereg_mismatch(self):
        """Should detect when YAML and prereg disagree."""
        curriculum = {
            "slices": {
                "slice_uplift_goal": {
                    "success_metric": {"kind": "goal_hit"}
                }
            }
        }
        with open(self.curriculum_path, "w") as f:
            yaml.dump(curriculum, f)
        
        prereg = {
            "slice_uplift_goal": {
                "success_metric": {"kind": "density"}  # Mismatch!
            }
        }
        with open(self.prereg_path, "w") as f:
            yaml.dump(prereg, f)
        
        self.plan_path.write_text("# Plan")
        
        findings = detect_metric_kind_drift(
            self.plan_path, self.curriculum_path, self.prereg_path
        )
        fail_findings = [f for f in findings if f.severity == DriftSeverity.FAIL]
        self.assertGreater(len(fail_findings), 0)


class TestParameterDriftDetection(unittest.TestCase):
    """Tests for parameter drift detection."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.curriculum_path = Path(self.temp_dir) / "curriculum.yaml"
        self.prereg_path = Path(self.temp_dir) / "prereg.yaml"
    
    def test_detect_param_only_in_yaml(self):
        """Should detect params present in YAML but not prereg."""
        curriculum = {
            "slices": {
                "slice_uplift_goal": {
                    "success_metric": {
                        "kind": "goal_hit",
                        "parameters": {
                            "target_hashes": ["h1"],
                            "min_total_verified": 3,
                            "extra_param": "only_in_yaml",
                        }
                    }
                }
            }
        }
        with open(self.curriculum_path, "w") as f:
            yaml.dump(curriculum, f)
        
        prereg = {
            "slice_uplift_goal": {
                "success_metric": {
                    "kind": "goal_hit",
                    "parameters": {
                        "target_hashes": ["h1"],
                        "min_total_verified": 3,
                    }
                }
            }
        }
        with open(self.prereg_path, "w") as f:
            yaml.dump(prereg, f)
        
        findings = detect_parameter_drift(self.curriculum_path, self.prereg_path)
        param_drift = [f for f in findings if "extra_param" in str(f.details)]
        self.assertGreater(len(param_drift), 0)


class TestFunctionDriftDetection(unittest.TestCase):
    """Tests for function drift detection."""
    
    def test_all_expected_functions_exist(self):
        """All expected metric functions should be importable."""
        findings = detect_function_drift()
        missing_func_findings = [
            f for f in findings if f.category == "missing_function"
        ]
        self.assertEqual(len(missing_func_findings), 0)
    
    def test_function_map_complete(self):
        """METRIC_FUNCTION_MAP should cover all MetricKinds."""
        for kind in MetricKind:
            self.assertIn(kind, METRIC_FUNCTION_MAP)


# =============================================================================
# FORMULA VALIDATION TESTS
# =============================================================================

class TestFormulaReferenceValidation(unittest.TestCase):
    """Tests for formula reference validation."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.plan_path = Path(self.temp_dir) / "plan.md"
        self.curriculum_path = Path(self.temp_dir) / "curriculum.yaml"
    
    def test_detect_hash_in_doc_not_yaml(self):
        """Should detect hashes in docs that aren't in YAML."""
        fake_hash = "a" * 64
        self.plan_path.write_text(f"# Plan\nHash: {fake_hash}")
        
        curriculum = {
            "slices": {
                "slice_uplift_goal": {
                    "formula_pool_entries": [
                        {"hash": "b" * 64}  # Different hash
                    ]
                }
            }
        }
        with open(self.curriculum_path, "w") as f:
            yaml.dump(curriculum, f)
        
        findings = validate_formula_references(self.plan_path, self.curriculum_path)
        self.assertGreater(len(findings), 0)


class TestSlicePoolCompleteness(unittest.TestCase):
    """Tests for slice pool completeness validation."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.curriculum_path = Path(self.temp_dir) / "curriculum.yaml"
    
    def test_detect_empty_formula_pool(self):
        """Should detect slices with no formula_pool_entries."""
        curriculum = {
            "slices": {
                "slice_uplift_goal": {
                    "description": "Test slice",
                    # No formula_pool_entries
                }
            }
        }
        with open(self.curriculum_path, "w") as f:
            yaml.dump(curriculum, f)
        
        findings = validate_slice_pool_completeness(self.curriculum_path)
        empty_pool = [f for f in findings if f.category == "empty_formula_pool"]
        self.assertGreater(len(empty_pool), 0)
    
    def test_detect_no_target_formulas(self):
        """Should detect slices with no target formulas."""
        curriculum = {
            "slices": {
                "slice_uplift_goal": {
                    "formula_pool_entries": [
                        {"name": "bridge1", "role": "bridge", "hash": "a" * 64}
                    ]
                }
            }
        }
        with open(self.curriculum_path, "w") as f:
            yaml.dump(curriculum, f)
        
        findings = validate_slice_pool_completeness(self.curriculum_path)
        no_target = [f for f in findings if f.category == "no_target_formulas"]
        self.assertGreater(len(no_target), 0)


# =============================================================================
# SEMANTIC EQUIVALENCE TESTS
# =============================================================================

class TestSemanticEquivalence(unittest.TestCase):
    """Tests for semantic equivalence between artifacts."""
    
    def test_doc_slice_metric_map_complete(self):
        """DOC_SLICE_METRIC_MAP should have all documented slices."""
        expected_slices = {
            "slice_uplift_goal",
            "slice_uplift_sparse",
            "slice_uplift_tree",
            "slice_uplift_dependency",
        }
        self.assertEqual(set(DOC_SLICE_METRIC_MAP.keys()), expected_slices)
    
    def test_metric_kind_to_function_bijection(self):
        """Each metric kind should map to exactly one function."""
        kinds_with_functions = set(METRIC_FUNCTION_MAP.keys())
        all_kinds = set(MetricKind)
        self.assertEqual(kinds_with_functions, all_kinds)
    
    def test_function_names_follow_convention(self):
        """All function names should start with 'compute_'."""
        for func_name in METRIC_FUNCTION_MAP.values():
            self.assertTrue(func_name.startswith("compute_"))


class TestDocYamlConsistency(unittest.TestCase):
    """Tests for doc → YAML consistency."""
    
    def test_documented_slices_have_expected_metrics(self):
        """Each documented slice should map to expected metric."""
        expected = {
            "slice_uplift_goal": "goal_hit",
            "slice_uplift_sparse": "density",
            "slice_uplift_tree": "chain_length",
            "slice_uplift_dependency": "multi_goal",
        }
        for slice_name, metric_kind in expected.items():
            self.assertEqual(
                DOC_SLICE_METRIC_MAP[slice_name].value,
                metric_kind,
                f"Slice {slice_name} should use {metric_kind}"
            )


class TestYamlCodeConsistency(unittest.TestCase):
    """Tests for YAML → code consistency."""
    
    def test_metric_functions_importable(self):
        """All metric functions should be importable."""
        from experiments.slice_success_metrics import (
            compute_goal_hit,
            compute_sparse_success,
            compute_chain_success,
            compute_multi_goal_success,
        )
        self.assertIsNotNone(compute_goal_hit)
        self.assertIsNotNone(compute_sparse_success)
        self.assertIsNotNone(compute_chain_success)
        self.assertIsNotNone(compute_multi_goal_success)
    
    def test_function_signatures_match_params(self):
        """Function signatures should accept expected parameters."""
        import inspect
        from experiments.slice_success_metrics import compute_goal_hit
        
        sig = inspect.signature(compute_goal_hit)
        params = set(sig.parameters.keys())
        # Should have verified_statements, target_hashes, min_total_verified
        self.assertIn("verified_statements", params)
        self.assertIn("target_hashes", params)
        self.assertIn("min_total_verified", params)


class TestPreregMetricsConsistency(unittest.TestCase):
    """Tests for prereg → metrics consistency."""
    
    def test_prereg_metric_kinds_valid(self):
        """All prereg metric kinds should be valid MetricKind values."""
        valid_kinds = {k.value for k in MetricKind}
        prereg_kinds = {"goal_hit", "density", "chain_length", "multi_goal"}
        self.assertEqual(prereg_kinds, valid_kinds)


# =============================================================================
# REPORTING TESTS
# =============================================================================

class TestMarkdownReportFormat(unittest.TestCase):
    """Tests for Markdown report formatting."""
    
    def test_report_contains_status(self):
        """Report should contain overall status."""
        report = DriftReport()
        markdown = format_drift_markdown_report(report)
        self.assertIn("Overall Status: OK", markdown)
    
    def test_report_contains_phase_label(self):
        """Report should contain PHASE II label."""
        report = DriftReport()
        markdown = format_drift_markdown_report(report)
        self.assertIn("PHASE II", markdown)
    
    def test_report_includes_drift_findings(self):
        """Report should include drift findings."""
        report = DriftReport()
        report.drift_findings.append(DriftFinding(
            severity=DriftSeverity.WARNING,
            category="test_drift",
            source_file="source.yaml",
            target_file="target.yaml",
            message="Test drift message",
        ))
        markdown = format_drift_markdown_report(report)
        self.assertIn("test_drift", markdown)
        self.assertIn("Test drift message", markdown)


class TestJsonReportFormat(unittest.TestCase):
    """Tests for JSON report formatting."""
    
    def test_json_structure(self):
        """JSON report should have required keys."""
        report = DriftReport()
        json_report = format_drift_json_report(report)
        
        self.assertIn("overall_status", json_report)
        self.assertIn("drift_findings", json_report)
        self.assertIn("doc_findings", json_report)
        self.assertIn("formula_findings", json_report)
        self.assertIn("summary", json_report)
    
    def test_json_serializable(self):
        """JSON report should be serializable."""
        import json as json_module
        
        report = DriftReport()
        report.drift_findings.append(DriftFinding(
            severity=DriftSeverity.FAIL,
            category="test",
            source_file="a",
            target_file="b",
            message="Test",
            details={"key": "value"},
        ))
        
        json_report = format_drift_json_report(report)
        # Should not raise
        serialized = json_module.dumps(json_report)
        self.assertIsInstance(serialized, str)


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration(unittest.TestCase):
    """Integration tests for drift detection."""
    
    def test_run_drift_detection_returns_report(self):
        """run_drift_detection should return a DriftReport."""
        report = run_drift_detection(include_base_audit=False)
        self.assertIsInstance(report, DriftReport)
    
    def test_drift_detection_with_base_audit(self):
        """Should include base audit when requested."""
        report = run_drift_detection(include_base_audit=True)
        self.assertIsNotNone(report.base_audit)
    
    def test_drift_detection_without_base_audit(self):
        """Should exclude base audit when requested."""
        report = run_drift_detection(include_base_audit=False)
        self.assertIsNone(report.base_audit)


class TestDeterminism(unittest.TestCase):
    """Tests for determinism guarantees."""
    
    def test_drift_detection_deterministic(self):
        """Multiple runs should produce identical results."""
        report1 = run_drift_detection(include_base_audit=False)
        report2 = run_drift_detection(include_base_audit=False)
        
        self.assertEqual(report1.status, report2.status)
        self.assertEqual(len(report1.drift_findings), len(report2.drift_findings))
        self.assertEqual(len(report1.doc_findings), len(report2.doc_findings))
    
    def test_header_extraction_deterministic(self):
        """Header extraction should be deterministic."""
        content = "# H1\n## H2\n### H3"
        headers1 = extract_headers_from_markdown(content)
        headers2 = extract_headers_from_markdown(content)
        self.assertEqual(headers1, headers2)


if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

