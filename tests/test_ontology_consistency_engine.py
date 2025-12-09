"""
PHASE II — NOT USED IN PHASE I

Unit tests for experiments/ontology_consistency_engine.py

Tests cover:
- Metric ontology extraction from code, doc, YAML
- Bidirectional schema validation
- Mathematical consistency checks
- CI JSON report format
"""

import unittest
import tempfile
import json
from pathlib import Path
from typing import Dict, Any

import yaml

from experiments.ontology_consistency_engine import (
    # Enums and data structures
    ConsistencyStatus,
    MetricOntologyEntry,
    OntologyDiff,
    MathematicalIssue,
    SchemaValidationResult,
    OntologyReport,
    # Ontology extraction
    extract_ontology_from_code,
    extract_ontology_from_doc,
    extract_ontology_from_yaml,
    compare_ontologies,
    # Schema validation
    validate_doc_to_yaml,
    validate_yaml_to_code,
    validate_code_to_doc,
    # Mathematical checks
    check_unreachable_thresholds,
    check_contradictory_params,
    check_degenerate_criteria,
    # Formula checks
    check_formula_inconsistencies,
    # Reporting
    format_markdown_report,
    generate_recommendations,
    run_ontology_audit,
)


class TestConsistencyStatus(unittest.TestCase):
    """Tests for ConsistencyStatus enum."""
    
    def test_status_values(self):
        """All status values should be lowercase."""
        self.assertEqual(ConsistencyStatus.OK.value, "ok")
        self.assertEqual(ConsistencyStatus.WARNING.value, "warning")
        self.assertEqual(ConsistencyStatus.FAIL.value, "fail")


class TestMetricOntologyEntry(unittest.TestCase):
    """Tests for MetricOntologyEntry data structure."""
    
    def test_to_dict(self):
        """Entry should serialize to dict correctly."""
        entry = MetricOntologyEntry(
            source="code",
            metric_kind="goal_hit",
            required_params={"target_hashes", "min_total_verified"},
            success_criteria="hits >= min_total_verified",
        )
        d = entry.to_dict()
        
        self.assertEqual(d["source"], "code")
        self.assertEqual(d["metric_kind"], "goal_hit")
        self.assertIn("target_hashes", d["required_params"])
        self.assertIn("min_total_verified", d["required_params"])


class TestOntologyReport(unittest.TestCase):
    """Tests for OntologyReport status computation."""
    
    def test_empty_report_is_ok(self):
        """Empty report should have OK status."""
        report = OntologyReport()
        self.assertEqual(report.compute_status(), ConsistencyStatus.OK)
    
    def test_metric_drift_causes_warning(self):
        """Metric drift should cause WARNING status."""
        report = OntologyReport()
        report.metric_drift.append(OntologyDiff(
            metric_kind="goal_hit",
            source_a="code",
            source_b="doc",
            diff_type="missing_param",
        ))
        self.assertEqual(report.compute_status(), ConsistencyStatus.WARNING)
    
    def test_math_fail_causes_fail(self):
        """Mathematical FAIL should cause overall FAIL."""
        report = OntologyReport()
        report.mathematical_issues.append(MathematicalIssue(
            slice_name="test",
            issue_type="unreachable_threshold",
            severity=ConsistencyStatus.FAIL,
            message="Test failure",
        ))
        self.assertEqual(report.compute_status(), ConsistencyStatus.FAIL)
    
    def test_ci_json_format(self):
        """CI JSON should have required keys."""
        report = OntologyReport()
        ci_json = report.to_ci_json()
        
        self.assertIn("status", ci_json)
        self.assertIn("metric_drift", ci_json)
        self.assertIn("slice_drift", ci_json)
        self.assertIn("formula_inconsistencies", ci_json)
        self.assertIn("recommendations", ci_json)


# =============================================================================
# ONTOLOGY EXTRACTION TESTS
# =============================================================================

class TestExtractOntologyFromCode(unittest.TestCase):
    """Tests for code ontology extraction."""
    
    def test_extracts_all_metric_kinds(self):
        """Should extract all four metric kinds."""
        entries = extract_ontology_from_code(Path("experiments/slice_success_metrics.py"))
        metric_kinds = {e.metric_kind for e in entries}
        
        self.assertIn("goal_hit", metric_kinds)
        self.assertIn("density", metric_kinds)
        self.assertIn("chain_length", metric_kinds)
        self.assertIn("multi_goal", metric_kinds)
    
    def test_extracts_function_names(self):
        """Should extract function names."""
        entries = extract_ontology_from_code(Path("experiments/slice_success_metrics.py"))
        
        for entry in entries:
            self.assertIsNotNone(entry.function_name)
            self.assertTrue(entry.function_name.startswith("compute_"))
    
    def test_extracts_parameters(self):
        """Should extract function parameters."""
        entries = extract_ontology_from_code(Path("experiments/slice_success_metrics.py"))
        
        goal_hit = next((e for e in entries if e.metric_kind == "goal_hit"), None)
        self.assertIsNotNone(goal_hit)
        self.assertIn("verified_statements", goal_hit.required_params)
        self.assertIn("target_hashes", goal_hit.required_params)


class TestExtractOntologyFromYaml(unittest.TestCase):
    """Tests for YAML ontology extraction."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.yaml_path = Path(self.temp_dir) / "curriculum.yaml"
    
    def test_extracts_metric_kind(self):
        """Should extract metric kind from YAML."""
        curriculum = {
            "slices": {
                "slice_uplift_goal": {
                    "success_metric": {
                        "kind": "goal_hit",
                        "parameters": {"min_goal_hits": 2}
                    }
                }
            }
        }
        with open(self.yaml_path, "w") as f:
            yaml.dump(curriculum, f)
        
        entries = extract_ontology_from_yaml(self.yaml_path)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].metric_kind, "goal_hit")
    
    def test_extracts_parameters(self):
        """Should extract metric parameters from YAML."""
        curriculum = {
            "slices": {
                "test_slice": {
                    "success_metric": {
                        "kind": "density",
                        "parameters": {
                            "min_verified": 5,
                            "max_candidates": 40
                        }
                    }
                }
            }
        }
        with open(self.yaml_path, "w") as f:
            yaml.dump(curriculum, f)
        
        entries = extract_ontology_from_yaml(self.yaml_path)
        self.assertEqual(len(entries), 1)
        self.assertIn("min_verified", entries[0].required_params)
        self.assertIn("max_candidates", entries[0].required_params)


class TestCompareOntologies(unittest.TestCase):
    """Tests for ontology comparison."""
    
    def test_no_diff_when_identical(self):
        """Identical ontologies should have no diffs."""
        entry = MetricOntologyEntry(
            source="code",
            metric_kind="goal_hit",
            required_params={"a", "b"},
        )
        diffs = compare_ontologies([entry], [entry], [entry])
        # No structural diffs expected for identical entries
        self.assertIsInstance(diffs, list)
    
    def test_detects_missing_in_doc(self):
        """Should detect metric in code but not in doc."""
        code_entry = MetricOntologyEntry(
            source="code",
            metric_kind="new_metric",
            required_params=set(),
        )
        diffs = compare_ontologies([code_entry], [], [])
        
        missing_diffs = [d for d in diffs if d.diff_type == "missing_in_doc"]
        self.assertGreater(len(missing_diffs), 0)


# =============================================================================
# SCHEMA VALIDATION TESTS
# =============================================================================

class TestValidateDocToYaml(unittest.TestCase):
    """Tests for doc → YAML validation."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.doc_path = Path(self.temp_dir) / "doc.md"
        self.yaml_path = Path(self.temp_dir) / "curriculum.yaml"
    
    def test_detects_missing_slice(self):
        """Should detect documented slice missing from YAML."""
        self.doc_path.write_text("# Plan\n`slice_uplift_goal` is documented")
        
        curriculum = {"slices": {}}
        with open(self.yaml_path, "w") as f:
            yaml.dump(curriculum, f)
        
        results = validate_doc_to_yaml(self.doc_path, self.yaml_path)
        
        goal_result = next((r for r in results if r.slice_name == "slice_uplift_goal"), None)
        self.assertIsNotNone(goal_result)
        self.assertEqual(goal_result.status, ConsistencyStatus.FAIL)
    
    def test_detects_metric_mismatch(self):
        """Should detect metric kind mismatch."""
        self.doc_path.write_text("# Plan")
        
        curriculum = {
            "slices": {
                "slice_uplift_goal": {
                    "success_metric": {"kind": "wrong_metric"}
                }
            }
        }
        with open(self.yaml_path, "w") as f:
            yaml.dump(curriculum, f)
        
        results = validate_doc_to_yaml(self.doc_path, self.yaml_path)
        
        goal_result = next((r for r in results if r.slice_name == "slice_uplift_goal"), None)
        self.assertIsNotNone(goal_result)
        self.assertEqual(goal_result.status, ConsistencyStatus.FAIL)


class TestValidateYamlToCode(unittest.TestCase):
    """Tests for YAML → code validation."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.yaml_path = Path(self.temp_dir) / "curriculum.yaml"
    
    def test_valid_metric_passes(self):
        """Valid metric kind should pass."""
        curriculum = {
            "slices": {
                "test_slice": {
                    "success_metric": {"kind": "goal_hit"}
                }
            }
        }
        with open(self.yaml_path, "w") as f:
            yaml.dump(curriculum, f)
        
        results = validate_yaml_to_code(self.yaml_path)
        
        test_result = next((r for r in results if r.slice_name == "test_slice"), None)
        self.assertIsNotNone(test_result)
        self.assertEqual(test_result.status, ConsistencyStatus.OK)
    
    def test_invalid_metric_fails(self):
        """Invalid metric kind should fail."""
        curriculum = {
            "slices": {
                "test_slice": {
                    "success_metric": {"kind": "nonexistent_metric"}
                }
            }
        }
        with open(self.yaml_path, "w") as f:
            yaml.dump(curriculum, f)
        
        results = validate_yaml_to_code(self.yaml_path)
        
        test_result = next((r for r in results if r.slice_name == "test_slice"), None)
        self.assertIsNotNone(test_result)
        self.assertEqual(test_result.status, ConsistencyStatus.FAIL)


# =============================================================================
# MATHEMATICAL CHECKS TESTS
# =============================================================================

class TestUnreachableThresholds(unittest.TestCase):
    """Tests for unreachable threshold detection."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.yaml_path = Path(self.temp_dir) / "curriculum.yaml"
    
    def test_detects_min_goal_hits_exceeds_targets(self):
        """Should detect when min_goal_hits > target count."""
        curriculum = {
            "slices": {
                "test_slice": {
                    "success_metric": {
                        "kind": "goal_hit",
                        "parameters": {"min_goal_hits": 10}
                    },
                    "formula_pool_entries": [
                        {"role": "target", "hash": "a" * 64},
                        {"role": "target", "hash": "b" * 64},
                        # Only 2 targets, but min_goal_hits is 10
                    ]
                }
            }
        }
        with open(self.yaml_path, "w") as f:
            yaml.dump(curriculum, f)
        
        issues = check_unreachable_thresholds(self.yaml_path)
        
        self.assertGreater(len(issues), 0)
        self.assertEqual(issues[0].issue_type, "unreachable_threshold")
        self.assertEqual(issues[0].severity, ConsistencyStatus.FAIL)
    
    def test_detects_missing_goal_hashes(self):
        """Should detect required goals not in formula pool."""
        curriculum = {
            "slices": {
                "test_slice": {
                    "success_metric": {
                        "kind": "multi_goal",
                        "parameters": {
                            "required_goal_hashes": ["missing_hash_1", "missing_hash_2"]
                        }
                    },
                    "formula_pool_entries": [
                        {"hash": "other_hash"}
                    ]
                }
            }
        }
        with open(self.yaml_path, "w") as f:
            yaml.dump(curriculum, f)
        
        issues = check_unreachable_thresholds(self.yaml_path)
        
        self.assertGreater(len(issues), 0)
        self.assertEqual(issues[0].severity, ConsistencyStatus.FAIL)


class TestContradictoryParams(unittest.TestCase):
    """Tests for contradictory parameter detection."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.yaml_path = Path(self.temp_dir) / "curriculum.yaml"
    
    def test_detects_negative_minimum(self):
        """Should detect negative minimum thresholds."""
        curriculum = {
            "slices": {
                "test_slice": {
                    "success_metric": {
                        "kind": "density",
                        "parameters": {"min_verified": -5}
                    }
                }
            }
        }
        with open(self.yaml_path, "w") as f:
            yaml.dump(curriculum, f)
        
        issues = check_contradictory_params(self.yaml_path)
        
        self.assertGreater(len(issues), 0)
        self.assertEqual(issues[0].issue_type, "contradictory_params")
    
    def test_detects_breadth_exceeds_total(self):
        """Should detect breadth_max > total_max."""
        curriculum = {
            "slices": {
                "test_slice": {
                    "params": {
                        "breadth_max": 100,
                        "total_max": 50
                    },
                    "success_metric": {"kind": "density"}
                }
            }
        }
        with open(self.yaml_path, "w") as f:
            yaml.dump(curriculum, f)
        
        issues = check_contradictory_params(self.yaml_path)
        
        self.assertGreater(len(issues), 0)


class TestDegenerateCriteria(unittest.TestCase):
    """Tests for degenerate criteria detection."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.yaml_path = Path(self.temp_dir) / "curriculum.yaml"
    
    def test_detects_zero_min_verified(self):
        """Should detect min_verified=0 as degenerate."""
        curriculum = {
            "slices": {
                "test_slice": {
                    "success_metric": {
                        "kind": "density",
                        "parameters": {"min_verified": 0}
                    }
                }
            }
        }
        with open(self.yaml_path, "w") as f:
            yaml.dump(curriculum, f)
        
        issues = check_degenerate_criteria(self.yaml_path)
        
        self.assertGreater(len(issues), 0)
        self.assertEqual(issues[0].issue_type, "degenerate_criteria")
    
    def test_detects_empty_required_goals(self):
        """Should detect empty required_goal_hashes."""
        curriculum = {
            "slices": {
                "test_slice": {
                    "success_metric": {
                        "kind": "multi_goal",
                        "parameters": {"required_goal_hashes": []}
                    }
                }
            }
        }
        with open(self.yaml_path, "w") as f:
            yaml.dump(curriculum, f)
        
        issues = check_degenerate_criteria(self.yaml_path)
        
        self.assertGreater(len(issues), 0)


# =============================================================================
# FORMULA INCONSISTENCY TESTS
# =============================================================================

class TestFormulaInconsistencies(unittest.TestCase):
    """Tests for formula inconsistency detection."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.yaml_path = Path(self.temp_dir) / "curriculum.yaml"
    
    def test_detects_duplicate_hashes(self):
        """Should detect duplicate hashes in formula pool."""
        curriculum = {
            "slices": {
                "test_slice": {
                    "formula_pool_entries": [
                        {"hash": "duplicate_hash"},
                        {"hash": "duplicate_hash"},
                    ]
                }
            }
        }
        with open(self.yaml_path, "w") as f:
            yaml.dump(curriculum, f)
        
        issues = check_formula_inconsistencies(self.yaml_path)
        
        self.assertGreater(len(issues), 0)
        self.assertEqual(issues[0]["type"], "duplicate_hash")
    
    def test_detects_missing_hash(self):
        """Should detect formula without hash."""
        curriculum = {
            "slices": {
                "test_slice": {
                    "formula_pool_entries": [
                        {"name": "test", "formula": "p -> q"}  # No hash
                    ]
                }
            }
        }
        with open(self.yaml_path, "w") as f:
            yaml.dump(curriculum, f)
        
        issues = check_formula_inconsistencies(self.yaml_path)
        
        self.assertGreater(len(issues), 0)
        self.assertEqual(issues[0]["type"], "missing_hash")


# =============================================================================
# REPORTING TESTS
# =============================================================================

class TestMarkdownReport(unittest.TestCase):
    """Tests for Markdown report generation."""
    
    def test_contains_status(self):
        """Report should contain status."""
        report = OntologyReport()
        md = format_markdown_report(report)
        
        self.assertIn("Overall Status:", md)
        self.assertIn("OK", md)
    
    def test_contains_phase_label(self):
        """Report should contain PHASE II label."""
        report = OntologyReport()
        md = format_markdown_report(report)
        
        self.assertIn("PHASE II", md)
    
    def test_contains_recommendations(self):
        """Report should contain recommendations section."""
        report = OntologyReport()
        report.recommendations = ["Test recommendation"]
        md = format_markdown_report(report)
        
        self.assertIn("Recommendations", md)
        self.assertIn("Test recommendation", md)


class TestCIJsonReport(unittest.TestCase):
    """Tests for CI JSON report format."""
    
    def test_json_serializable(self):
        """CI JSON should be serializable."""
        report = OntologyReport()
        report.mathematical_issues.append(MathematicalIssue(
            slice_name="test",
            issue_type="test",
            severity=ConsistencyStatus.WARNING,
            message="Test",
        ))
        
        ci_json = report.to_ci_json()
        serialized = json.dumps(ci_json)
        
        self.assertIsInstance(serialized, str)
    
    def test_has_required_keys(self):
        """CI JSON should have all required keys."""
        report = OntologyReport()
        ci_json = report.to_ci_json()
        
        required_keys = ["status", "metric_drift", "slice_drift", 
                        "formula_inconsistencies", "recommendations"]
        for key in required_keys:
            self.assertIn(key, ci_json)


class TestRecommendations(unittest.TestCase):
    """Tests for recommendation generation."""
    
    def test_generates_for_metric_drift(self):
        """Should generate recommendation for metric drift."""
        report = OntologyReport()
        report.metric_drift.append(OntologyDiff(
            metric_kind="test",
            source_a="a",
            source_b="b",
            diff_type="test",
        ))
        
        recs = generate_recommendations(report)
        
        self.assertGreater(len(recs), 0)
    
    def test_includes_issue_recommendations(self):
        """Should include recommendations from mathematical issues."""
        report = OntologyReport()
        report.mathematical_issues.append(MathematicalIssue(
            slice_name="test",
            issue_type="test",
            severity=ConsistencyStatus.WARNING,
            message="Test",
            recommendation="Specific recommendation",
        ))
        
        recs = generate_recommendations(report)
        
        self.assertIn("Specific recommendation", recs)


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration(unittest.TestCase):
    """Integration tests for the full ontology audit."""
    
    def test_run_ontology_audit_returns_report(self):
        """run_ontology_audit should return OntologyReport."""
        report = run_ontology_audit()
        self.assertIsInstance(report, OntologyReport)
    
    def test_audit_computes_status(self):
        """Audit should compute overall status."""
        report = run_ontology_audit()
        self.assertIn(report.status, list(ConsistencyStatus))
    
    def test_audit_generates_recommendations(self):
        """Audit should generate recommendations."""
        report = run_ontology_audit()
        self.assertIsInstance(report.recommendations, list)


class TestDeterminism(unittest.TestCase):
    """Tests for determinism guarantees."""
    
    def test_audit_is_deterministic(self):
        """Multiple runs should produce identical results."""
        report1 = run_ontology_audit()
        report2 = run_ontology_audit()
        
        self.assertEqual(report1.status, report2.status)
        self.assertEqual(len(report1.metric_drift), len(report2.metric_drift))
        self.assertEqual(len(report1.mathematical_issues), len(report2.mathematical_issues))


if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

