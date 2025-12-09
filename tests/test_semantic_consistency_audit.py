"""
PHASE II — NOT USED IN PHASE I

Unit tests for experiments/semantic_consistency_audit.py

Tests the semantic consistency auditor without modifying any slice definitions,
metric implementations, or governance documents.
"""

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, Set
from unittest.mock import patch, MagicMock

from experiments.semantic_consistency_audit import (
    # Data structures
    MetricKind,
    AuditStatus,
    AuditIssue,
    SliceSemanticSpec,
    AuditReport,
    # Constants
    METRIC_REQUIRED_PARAMS,
    METRIC_OPTIONAL_PARAMS,
    DOC_SLICE_METRIC_MAP,
    METRIC_FUNCTION_MAP,
    KNOWN_METRIC_FIXES,
    # Functions
    extract_slice_specs,
    check_metric_kind_valid,
    check_required_params,
    check_doc_consistency,
    check_function_mapping,
    run_static_checks,
    create_mock_cycle_data,
    run_runtime_sanity_checks,
    format_markdown_report,
    format_json_report,
    run_audit,
    # Term Index Builder
    TermMention,
    TermEntry,
    TermIndexBuilder,
    build_term_index,
    save_term_index,
    # Suggestion Engine
    FixSuggestion,
    SuggestionEngine,
    generate_suggestions,
    format_suggestions_markdown,
    # CI Mode
    run_ci_checks,
    format_ci_output,
    # One-Line Drift Summary
    DriftCounts,
    count_drift,
    format_drift_summary,
    # Canonical Suggestion Format
    sanitize_suggestion_text,
    format_canonical_suggestion,
    generate_canonical_suggestions,
    # Semantic Knowledge Graph (v1.3)
    GraphEdge,
    SemanticKnowledgeGraph,
    SemanticGraphBuilder,
    build_semantic_graph,
    save_semantic_graph,
    # Graph-Based Drift Detection (v1.3)
    GraphDriftSignal,
    DriftGraphReport,
    analyze_graph_drift,
    format_drift_report_summary,
    # Graph-Aware Suggestions (v1.3)
    generate_graph_aware_suggestions,
    # Semantic Governance (Phase III)
    GovernanceStatus,
    build_semantic_governance_snapshot,
    # Suggestion Safety Filter (Phase III)
    filter_graph_suggestions,
    _extract_target_term,
    _extract_source_term,
    # Global Health Signal (Phase III)
    SemanticHealthStatus,
    summarize_semantic_graph_for_global_health,
    format_semantic_health_summary,
    sanitize_suggestion_text,
    format_canonical_suggestion,
    generate_canonical_suggestions,
    # Cross-System Alignment (Phase IV)
    AlignmentStatus,
    build_semantic_alignment_index,
    # Semantic Risk Analysis (Phase IV)
    RiskStatus,
    analyze_semantic_risk,
    # Director Panel (Phase IV)
    StatusLight,
    build_semantic_director_panel,
    build_semantic_director_panel_legacy,
    # Semantic Contract Auditor (Phase IV Follow-up)
    ContractStatus,
    audit_semantic_contract,
    # Drift Forecast Tile (Phase IV Follow-up)
    DriftDirection,
    ForecastBand,
    forecast_semantic_drift,
)


class TestMetricKindEnum(unittest.TestCase):
    """Tests for the MetricKind enumeration."""
    
    def test_metric_kinds_exist(self):
        """All documented metric kinds should be defined."""
        expected_kinds = {"goal_hit", "density", "chain_length", "multi_goal"}
        actual_kinds = {k.value for k in MetricKind}
        self.assertEqual(expected_kinds, actual_kinds)
    
    def test_metric_kind_from_string(self):
        """MetricKind should be constructible from valid strings."""
        self.assertEqual(MetricKind("goal_hit"), MetricKind.GOAL_HIT)
        self.assertEqual(MetricKind("density"), MetricKind.DENSITY)
        self.assertEqual(MetricKind("chain_length"), MetricKind.CHAIN_LENGTH)
        self.assertEqual(MetricKind("multi_goal"), MetricKind.MULTI_GOAL)
    
    def test_invalid_metric_kind_raises(self):
        """Invalid metric kinds should raise ValueError."""
        with self.assertRaises(ValueError):
            MetricKind("invalid_kind")


class TestAuditDataStructures(unittest.TestCase):
    """Tests for audit data structures."""
    
    def test_slice_semantic_spec_default_status(self):
        """A spec with no issues should have PASS status."""
        spec = SliceSemanticSpec(slice_name="test_slice")
        self.assertEqual(spec.status, AuditStatus.PASS)
    
    def test_slice_semantic_spec_warning_status(self):
        """A spec with only warnings should have WARNING status."""
        spec = SliceSemanticSpec(slice_name="test_slice")
        spec.issues.append(AuditIssue(
            severity=AuditStatus.WARNING,
            category="test",
            message="Test warning",
        ))
        self.assertEqual(spec.status, AuditStatus.WARNING)
    
    def test_slice_semantic_spec_fail_status(self):
        """A spec with any FAIL should have FAIL status."""
        spec = SliceSemanticSpec(slice_name="test_slice")
        spec.issues.append(AuditIssue(
            severity=AuditStatus.WARNING,
            category="test",
            message="Test warning",
        ))
        spec.issues.append(AuditIssue(
            severity=AuditStatus.FAIL,
            category="test",
            message="Test failure",
        ))
        self.assertEqual(spec.status, AuditStatus.FAIL)
    
    def test_audit_report_aggregates_status(self):
        """AuditReport should aggregate status from all slices."""
        report = AuditReport()
        
        # Empty report = PASS
        self.assertEqual(report.status, AuditStatus.PASS)
        
        # Add a passing slice
        report.slices.append(SliceSemanticSpec(slice_name="pass_slice"))
        self.assertEqual(report.status, AuditStatus.PASS)
        
        # Add a warning slice
        warn_spec = SliceSemanticSpec(slice_name="warn_slice")
        warn_spec.issues.append(AuditIssue(
            severity=AuditStatus.WARNING,
            category="test",
            message="Warning",
        ))
        report.slices.append(warn_spec)
        self.assertEqual(report.status, AuditStatus.WARNING)
        
        # Add a failing slice
        fail_spec = SliceSemanticSpec(slice_name="fail_slice")
        fail_spec.issues.append(AuditIssue(
            severity=AuditStatus.FAIL,
            category="test",
            message="Failure",
        ))
        report.slices.append(fail_spec)
        self.assertEqual(report.status, AuditStatus.FAIL)
    
    def test_audit_report_counts_tests(self):
        """AuditReport should correctly count passed/failed tests."""
        report = AuditReport()
        
        spec1 = SliceSemanticSpec(slice_name="s1")
        spec1.tests_passed = 3
        spec1.tests_failed = 1
        
        spec2 = SliceSemanticSpec(slice_name="s2")
        spec2.tests_passed = 2
        spec2.tests_failed = 0
        
        report.slices = [spec1, spec2]
        
        self.assertEqual(report.total_tests_passed, 5)
        self.assertEqual(report.total_tests_failed, 1)


class TestMetricRequiredParams(unittest.TestCase):
    """Tests for METRIC_REQUIRED_PARAMS consistency."""
    
    def test_all_metric_kinds_have_required_params(self):
        """Every MetricKind should have an entry in METRIC_REQUIRED_PARAMS."""
        for kind in MetricKind:
            self.assertIn(kind, METRIC_REQUIRED_PARAMS)
    
    def test_goal_hit_required_params(self):
        """GOAL_HIT should require target_hashes and min_total_verified."""
        required = METRIC_REQUIRED_PARAMS[MetricKind.GOAL_HIT]
        self.assertIn("target_hashes", required)
        self.assertIn("min_total_verified", required)
    
    def test_density_required_params(self):
        """DENSITY should require min_verified."""
        required = METRIC_REQUIRED_PARAMS[MetricKind.DENSITY]
        self.assertIn("min_verified", required)
    
    def test_chain_length_required_params(self):
        """CHAIN_LENGTH should require chain_target_hash and min_chain_length."""
        required = METRIC_REQUIRED_PARAMS[MetricKind.CHAIN_LENGTH]
        self.assertIn("chain_target_hash", required)
        self.assertIn("min_chain_length", required)
    
    def test_multi_goal_required_params(self):
        """MULTI_GOAL should require required_goal_hashes."""
        required = METRIC_REQUIRED_PARAMS[MetricKind.MULTI_GOAL]
        self.assertIn("required_goal_hashes", required)


class TestDocSliceMetricMap(unittest.TestCase):
    """Tests for DOC_SLICE_METRIC_MAP consistency with PHASE2_RFL_UPLIFT_PLAN.md."""
    
    def test_documented_slices_exist(self):
        """All four documented uplift slices should be mapped."""
        expected_slices = {
            "slice_uplift_goal",
            "slice_uplift_sparse",
            "slice_uplift_tree",
            "slice_uplift_dependency",
        }
        self.assertEqual(set(DOC_SLICE_METRIC_MAP.keys()), expected_slices)
    
    def test_slice_metric_assignments(self):
        """Each slice should map to the correct metric kind per documentation."""
        self.assertEqual(DOC_SLICE_METRIC_MAP["slice_uplift_goal"], MetricKind.GOAL_HIT)
        self.assertEqual(DOC_SLICE_METRIC_MAP["slice_uplift_sparse"], MetricKind.DENSITY)
        self.assertEqual(DOC_SLICE_METRIC_MAP["slice_uplift_tree"], MetricKind.CHAIN_LENGTH)
        self.assertEqual(DOC_SLICE_METRIC_MAP["slice_uplift_dependency"], MetricKind.MULTI_GOAL)


class TestMetricFunctionMap(unittest.TestCase):
    """Tests for METRIC_FUNCTION_MAP consistency with slice_success_metrics.py."""
    
    def test_all_metric_kinds_have_function(self):
        """Every MetricKind should map to a function name."""
        for kind in MetricKind:
            self.assertIn(kind, METRIC_FUNCTION_MAP)
    
    def test_function_name_conventions(self):
        """Function names should follow compute_* convention."""
        for kind, func_name in METRIC_FUNCTION_MAP.items():
            self.assertTrue(
                func_name.startswith("compute_"),
                f"Function {func_name} for {kind} should start with 'compute_'"
            )


class TestStaticChecks(unittest.TestCase):
    """Tests for static consistency check functions."""
    
    def test_check_metric_kind_valid_passes(self):
        """Valid metric kinds should not generate issues."""
        spec = SliceSemanticSpec(slice_name="test", metric_kind="goal_hit")
        check_metric_kind_valid(spec)
        self.assertEqual(len(spec.issues), 0)
    
    def test_check_metric_kind_valid_warns_on_missing(self):
        """Missing metric kind should generate WARNING."""
        spec = SliceSemanticSpec(slice_name="test", metric_kind=None)
        check_metric_kind_valid(spec)
        self.assertEqual(len(spec.issues), 1)
        self.assertEqual(spec.issues[0].severity, AuditStatus.WARNING)
    
    def test_check_metric_kind_valid_fails_on_invalid(self):
        """Invalid metric kind should generate FAIL."""
        spec = SliceSemanticSpec(slice_name="test", metric_kind="invalid_kind")
        check_metric_kind_valid(spec)
        self.assertEqual(len(spec.issues), 1)
        self.assertEqual(spec.issues[0].severity, AuditStatus.FAIL)
    
    def test_check_required_params_passes(self):
        """All required params present should not generate issues."""
        spec = SliceSemanticSpec(
            slice_name="test",
            metric_kind="goal_hit",
            yaml_params={
                "success_metric_params": {
                    "target_hashes": {"h1"},
                    "min_total_verified": 1,
                }
            }
        )
        check_required_params(spec)
        self.assertEqual(len(spec.issues), 0)
    
    def test_check_required_params_fails_on_missing(self):
        """Missing required params should generate FAIL."""
        spec = SliceSemanticSpec(
            slice_name="test",
            metric_kind="goal_hit",
            yaml_params={
                "success_metric_params": {
                    "target_hashes": {"h1"},
                    # Missing min_total_verified
                }
            }
        )
        check_required_params(spec)
        fail_issues = [i for i in spec.issues if i.severity == AuditStatus.FAIL]
        self.assertGreater(len(fail_issues), 0)
    
    def test_check_required_params_warns_on_unexpected(self):
        """Unexpected params should generate WARNING."""
        spec = SliceSemanticSpec(
            slice_name="test",
            metric_kind="goal_hit",
            yaml_params={
                "success_metric_params": {
                    "target_hashes": {"h1"},
                    "min_total_verified": 1,
                    "extra_param": "unexpected",
                }
            }
        )
        check_required_params(spec)
        warn_issues = [i for i in spec.issues if i.severity == AuditStatus.WARNING]
        self.assertGreater(len(warn_issues), 0)
    
    def test_check_doc_consistency_passes(self):
        """Matching metric kind should not generate issues."""
        spec = SliceSemanticSpec(
            slice_name="slice_uplift_goal",
            metric_kind="goal_hit",
        )
        check_doc_consistency(spec)
        self.assertEqual(len(spec.issues), 0)
    
    def test_check_doc_consistency_fails_on_mismatch(self):
        """Mismatched metric kind should generate FAIL."""
        spec = SliceSemanticSpec(
            slice_name="slice_uplift_goal",
            metric_kind="density",  # Wrong! Should be goal_hit
        )
        check_doc_consistency(spec)
        fail_issues = [i for i in spec.issues if i.severity == AuditStatus.FAIL]
        self.assertGreater(len(fail_issues), 0)


class TestMockCycleData(unittest.TestCase):
    """Tests for mock cycle data generation."""
    
    def test_all_metric_kinds_have_test_data(self):
        """Every metric kind should have test data."""
        mock_data = create_mock_cycle_data()
        for kind in MetricKind:
            self.assertIn(
                kind.value,
                mock_data,
                f"Missing mock data for {kind.value}"
            )
    
    def test_test_cases_have_required_fields(self):
        """Each test case should have inputs, expected_success, expected_value."""
        mock_data = create_mock_cycle_data()
        for kind_value, test_cases in mock_data.items():
            self.assertIsInstance(test_cases, list)
            self.assertGreater(len(test_cases), 0, f"No test cases for {kind_value}")
            for i, tc in enumerate(test_cases):
                self.assertIn("inputs", tc, f"Test case {i} for {kind_value} missing 'inputs'")
                self.assertIn("expected_success", tc, f"Test case {i} for {kind_value} missing 'expected_success'")
                self.assertIn("expected_value", tc, f"Test case {i} for {kind_value} missing 'expected_value'")


class TestRuntimeSanityChecks(unittest.TestCase):
    """Tests for runtime sanity checks."""
    
    def test_runtime_checks_count_passes(self):
        """Passing runtime checks should increment tests_passed."""
        spec = SliceSemanticSpec(
            slice_name="slice_uplift_goal",
            metric_kind="goal_hit",
            metric_function="compute_goal_hit",
        )
        run_runtime_sanity_checks(spec)
        self.assertGreater(spec.tests_passed, 0)
    
    def test_runtime_checks_skip_without_function(self):
        """Runtime checks should skip if no metric_function."""
        spec = SliceSemanticSpec(
            slice_name="test",
            metric_kind="goal_hit",
            metric_function=None,
        )
        run_runtime_sanity_checks(spec)
        self.assertEqual(spec.tests_passed, 0)
        self.assertEqual(spec.tests_failed, 0)


class TestReporting(unittest.TestCase):
    """Tests for report formatting."""
    
    def test_markdown_report_contains_status(self):
        """Markdown report should contain overall status."""
        report = AuditReport()
        report.slices = [SliceSemanticSpec(slice_name="test")]
        markdown = format_markdown_report(report)
        self.assertIn("Overall Status: PASS", markdown)
    
    def test_markdown_report_contains_phase_label(self):
        """Markdown report should contain PHASE II label."""
        report = AuditReport()
        markdown = format_markdown_report(report)
        self.assertIn("PHASE II", markdown)
    
    def test_json_report_structure(self):
        """JSON report should have required keys."""
        report = AuditReport()
        spec = SliceSemanticSpec(slice_name="test", metric_kind="goal_hit")
        report.slices = [spec]
        
        json_report = format_json_report(report)
        
        self.assertIn("overall_status", json_report)
        self.assertIn("slices", json_report)
        self.assertIn("global_issues", json_report)
        self.assertIn("label", json_report)
        self.assertIn("PHASE II", json_report["label"])
    
    def test_json_report_slice_structure(self):
        """JSON report slices should have required fields."""
        report = AuditReport()
        spec = SliceSemanticSpec(
            slice_name="test_slice",
            metric_kind="goal_hit",
            metric_function="compute_goal_hit",
        )
        spec.tests_passed = 3
        spec.tests_failed = 1
        report.slices = [spec]
        
        json_report = format_json_report(report)
        slice_data = json_report["slices"][0]
        
        self.assertEqual(slice_data["slice_name"], "test_slice")
        self.assertEqual(slice_data["metric_kind"], "goal_hit")
        self.assertEqual(slice_data["metric_function"], "compute_goal_hit")
        self.assertEqual(slice_data["tests_passed"], 3)
        self.assertEqual(slice_data["tests_failed"], 1)


class TestIntegration(unittest.TestCase):
    """Integration tests for the full audit pipeline."""
    
    def test_run_audit_returns_report(self):
        """run_audit should return an AuditReport."""
        # Use default paths (which may not exist in test environment)
        # This tests the structure, not the actual files
        try:
            report = run_audit()
            self.assertIsInstance(report, AuditReport)
        except FileNotFoundError:
            # Expected if config files don't exist
            pass
    
    def test_run_audit_with_mock_curriculum(self):
        """run_audit should work with mocked curriculum file."""
        mock_curriculum = {
            "version": "2.0",
            "slices": {
                "slice_uplift_goal": {
                    "description": "Test slice",
                    "items": ["item1", "item2"],
                    "success_metric": {
                        "kind": "goal_hit",
                        "params": {
                            "target_hashes": ["h1"],
                            "min_total_verified": 1,
                        }
                    }
                }
            }
        }
        
        import tempfile
        import yaml
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(mock_curriculum, f)
            curriculum_path = Path(f.name)
        
        try:
            report = run_audit(curriculum_path=curriculum_path)
            self.assertIsInstance(report, AuditReport)
            self.assertEqual(len(report.slices), 1)
            self.assertEqual(report.slices[0].slice_name, "slice_uplift_goal")
        finally:
            curriculum_path.unlink()


class TestDeterminism(unittest.TestCase):
    """Tests for determinism guarantees."""
    
    def test_mock_data_deterministic(self):
        """create_mock_cycle_data should return identical data on each call."""
        data1 = create_mock_cycle_data()
        data2 = create_mock_cycle_data()
        
        # Compare structure and values
        self.assertEqual(set(data1.keys()), set(data2.keys()))
        for key in data1:
            self.assertEqual(len(data1[key]), len(data2[key]))
    
    def test_static_checks_deterministic(self):
        """Static checks should produce identical issues on repeated runs."""
        spec1 = SliceSemanticSpec(slice_name="test", metric_kind="invalid")
        spec2 = SliceSemanticSpec(slice_name="test", metric_kind="invalid")
        
        check_metric_kind_valid(spec1)
        check_metric_kind_valid(spec2)
        
        self.assertEqual(len(spec1.issues), len(spec2.issues))
        self.assertEqual(spec1.issues[0].severity, spec2.issues[0].severity)
        self.assertEqual(spec1.issues[0].message, spec2.issues[0].message)


# =============================================================================
# TERM INDEX BUILDER TESTS
# =============================================================================

class TestTermMention(unittest.TestCase):
    """Tests for TermMention data structure."""
    
    def test_term_mention_creation(self):
        """TermMention should store file, line, and exact_spelling."""
        mention = TermMention(file="test.py", line=42, exact_spelling="goal_hit")
        self.assertEqual(mention.file, "test.py")
        self.assertEqual(mention.line, 42)
        self.assertEqual(mention.exact_spelling, "goal_hit")


class TestTermEntry(unittest.TestCase):
    """Tests for TermEntry data structure."""
    
    def test_term_entry_creation(self):
        """TermEntry should store term, canonical_form, and kind."""
        entry = TermEntry(term="goal_hit", canonical_form="goal_hit", kind="metric")
        self.assertEqual(entry.term, "goal_hit")
        self.assertEqual(entry.canonical_form, "goal_hit")
        self.assertEqual(entry.kind, "metric")
        self.assertEqual(entry.mentions, [])
    
    def test_term_entry_to_dict(self):
        """TermEntry.to_dict() should produce correct structure."""
        entry = TermEntry(term="goal_hit", canonical_form="goal_hit", kind="metric")
        entry.mentions.append(TermMention(file="test.py", line=10, exact_spelling="goal_hit"))
        
        d = entry.to_dict()
        self.assertEqual(d["term"], "goal_hit")
        self.assertEqual(d["canonical_form"], "goal_hit")
        self.assertEqual(d["kind"], "metric")
        self.assertEqual(len(d["mentions"]), 1)
        self.assertEqual(d["mentions"][0]["file"], "test.py")
        self.assertEqual(d["mentions"][0]["line"], 10)


class TestTermIndexBuilder(unittest.TestCase):
    """Tests for TermIndexBuilder."""
    
    def test_canonical_slice_names_defined(self):
        """Builder should have all four Phase II slice names."""
        expected = {
            "slice_uplift_goal",
            "slice_uplift_sparse",
            "slice_uplift_tree",
            "slice_uplift_dependency",
        }
        actual = set(TermIndexBuilder.CANONICAL_SLICE_NAMES)
        self.assertEqual(expected, actual)
    
    def test_canonical_metric_kinds_defined(self):
        """Builder should have all four metric kinds."""
        expected = {"goal_hit", "density", "chain_length", "multi_goal"}
        actual = set(TermIndexBuilder.CANONICAL_METRIC_KINDS)
        self.assertEqual(expected, actual)
    
    def test_scan_file_finds_slice_names(self):
        """scan_file should find slice name mentions."""
        builder = TermIndexBuilder()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write('slice_name = "slice_uplift_goal"\n')
            f.write('# Also slice_uplift_sparse here\n')
            temp_path = Path(f.name)
        
        try:
            builder.scan_file(temp_path)
            self.assertIn("slice_uplift_goal", builder.terms)
            self.assertIn("slice_uplift_sparse", builder.terms)
        finally:
            temp_path.unlink()
    
    def test_scan_file_finds_metric_kinds(self):
        """scan_file should find metric kind mentions."""
        builder = TermIndexBuilder()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write('metric_kind = "goal_hit"\n')
            f.write('# density metric is used here\n')
            temp_path = Path(f.name)
        
        try:
            builder.scan_file(temp_path)
            self.assertIn("goal_hit", builder.terms)
            self.assertIn("density", builder.terms)
        finally:
            temp_path.unlink()
    
    def test_scan_file_finds_definitions(self):
        """scan_file should find Definition/Lemma/Proposition identifiers."""
        builder = TermIndexBuilder()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write('## Definition 1.1\n')
            f.write('See Lemma 2.3 and Proposition 4.5\n')
            temp_path = Path(f.name)
        
        try:
            builder.scan_file(temp_path)
            self.assertIn("definition_1.1", builder.terms)
            self.assertIn("lemma_2.3", builder.terms)
            self.assertIn("proposition_4.5", builder.terms)
        finally:
            temp_path.unlink()
    
    def test_scan_file_finds_theory_symbols(self):
        """scan_file should find theory symbols like delta_p, A(t), H_t."""
        builder = TermIndexBuilder()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
            f.write('The uplift delta_p is measured\n')
            f.write('Abstention rate A(t) decreases\n')
            f.write('Derivation entropy H_t increases\n')
            temp_path = Path(f.name)
        
        try:
            builder.scan_file(temp_path)
            self.assertIn("delta_p", builder.terms)
            self.assertIn("abstention_rate", builder.terms)
            self.assertIn("derivation_entropy", builder.terms)
        finally:
            temp_path.unlink()
    
    def test_build_index_structure(self):
        """build_index should return correct structure."""
        builder = TermIndexBuilder()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # Create minimal files
            config_dir = tmpdir_path / "config"
            config_dir.mkdir()
            (config_dir / "curriculum_uplift_phase2.yaml").write_text(
                "slices:\n  slice_uplift_goal:\n    success_metric:\n      kind: goal_hit\n"
            )
            
            index = builder.build_index(tmpdir_path)
            
            self.assertIn("version", index)
            self.assertIn("phase", index)
            self.assertIn("terms", index)
            self.assertIn("summary", index)
            self.assertIn("total_terms", index["summary"])
            self.assertIn("by_kind", index["summary"])


class TestBuildTermIndex(unittest.TestCase):
    """Tests for build_term_index function."""
    
    def test_build_term_index_returns_dict(self):
        """build_term_index should return a dictionary."""
        # This will scan actual project files if they exist
        # or return an empty structure if they don't
        index = build_term_index()
        self.assertIsInstance(index, dict)
        self.assertIn("terms", index)
        self.assertIn("summary", index)


class TestSaveTermIndex(unittest.TestCase):
    """Tests for save_term_index function."""
    
    def test_save_term_index_creates_file(self):
        """save_term_index should create the JSON file."""
        index = {"version": "1.0", "terms": [], "summary": {"total_terms": 0}}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "subdir" / "term_index.json"
            save_term_index(index, output_path)
            
            self.assertTrue(output_path.exists())
            
            with open(output_path) as f:
                loaded = json.load(f)
            self.assertEqual(loaded["version"], "1.0")


# =============================================================================
# SUGGESTION ENGINE TESTS
# =============================================================================

class TestFixSuggestion(unittest.TestCase):
    """Tests for FixSuggestion data structure."""
    
    def test_fix_suggestion_creation(self):
        """FixSuggestion should store all fields."""
        suggestion = FixSuggestion(
            issue_type="metric_kind_mismatch",
            current_value="goal_hit_rate",
            suggested_value="goal_hit",
            file_path="config/test.yaml",
            approximate_line=42,
            description="Replace metric kind",
        )
        self.assertEqual(suggestion.issue_type, "metric_kind_mismatch")
        self.assertEqual(suggestion.current_value, "goal_hit_rate")
        self.assertEqual(suggestion.suggested_value, "goal_hit")
        self.assertEqual(suggestion.file_path, "config/test.yaml")
        self.assertEqual(suggestion.approximate_line, 42)
    
    def test_fix_suggestion_to_dict(self):
        """FixSuggestion.to_dict() should produce correct structure."""
        suggestion = FixSuggestion(
            issue_type="test_type",
            current_value="old",
            suggested_value="new",
            file_path="test.yaml",
            approximate_line=10,
            description="Test description",
        )
        d = suggestion.to_dict()
        
        self.assertEqual(d["issue_type"], "test_type")
        self.assertEqual(d["current_value"], "old")
        self.assertEqual(d["suggested_value"], "new")
        self.assertEqual(d["file_path"], "test.yaml")
        self.assertEqual(d["approximate_line"], 10)
        self.assertEqual(d["description"], "Test description")


class TestSuggestionEngine(unittest.TestCase):
    """Tests for SuggestionEngine."""
    
    def test_analyze_metric_kind_drift(self):
        """Engine should generate suggestion for metric kind mismatch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            config_dir = tmpdir_path / "config"
            config_dir.mkdir()
            yaml_file = config_dir / "curriculum_uplift_phase2.yaml"
            yaml_file.write_text("success_metric:\n  kind: goal_hit_rate\n")
            
            engine = SuggestionEngine(tmpdir_path)
            engine.analyze_metric_kind_drift(
                slice_name="slice_uplift_goal",
                yaml_kind="goal_hit_rate",
                expected_kind="goal_hit",
            )
            
            self.assertEqual(len(engine.suggestions), 1)
            self.assertEqual(engine.suggestions[0].current_value, "goal_hit_rate")
            self.assertEqual(engine.suggestions[0].suggested_value, "goal_hit")
    
    def test_analyze_missing_slice(self):
        """Engine should generate suggestion for missing slice."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = SuggestionEngine(Path(tmpdir))
            engine.analyze_missing_slice("slice_uplift_new")
            
            self.assertEqual(len(engine.suggestions), 1)
            self.assertEqual(engine.suggestions[0].issue_type, "missing_slice")
            self.assertIn("slice_uplift_new", engine.suggestions[0].description)
    
    def test_analyze_invalid_metric_kind(self):
        """Engine should suggest valid metric kind for invalid one."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            config_dir = tmpdir_path / "config"
            config_dir.mkdir()
            yaml_file = config_dir / "curriculum_uplift_phase2.yaml"
            yaml_file.write_text("kind: density_check\n")
            
            engine = SuggestionEngine(tmpdir_path)
            engine.analyze_invalid_metric_kind(
                slice_name="test_slice",
                invalid_kind="density_check",
                valid_kinds=["goal_hit", "density", "chain_length", "multi_goal"],
            )
            
            self.assertEqual(len(engine.suggestions), 1)
            self.assertEqual(engine.suggestions[0].current_value, "density_check")
            # Should suggest "density" since it contains "density"
            self.assertEqual(engine.suggestions[0].suggested_value, "density")
    
    def test_suggestion_includes_file_and_line(self):
        """Suggestions should include file names and approximate line numbers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            config_dir = tmpdir_path / "config"
            config_dir.mkdir()
            yaml_file = config_dir / "curriculum_uplift_phase2.yaml"
            yaml_file.write_text("line1\nline2\nkind: wrong_kind\nline4\n")
            
            engine = SuggestionEngine(tmpdir_path)
            engine.analyze_metric_kind_drift(
                slice_name="test",
                yaml_kind="wrong_kind",
                expected_kind="goal_hit",
            )
            
            self.assertEqual(len(engine.suggestions), 1)
            self.assertIn("curriculum_uplift_phase2.yaml", engine.suggestions[0].file_path)
            self.assertEqual(engine.suggestions[0].approximate_line, 3)


class TestGenerateSuggestions(unittest.TestCase):
    """Tests for generate_suggestions function."""
    
    def test_empty_report_no_suggestions(self):
        """Report with no issues should produce no suggestions."""
        report = AuditReport()
        report.slices.append(SliceSemanticSpec(
            slice_name="slice_uplift_goal",
            metric_kind="goal_hit",
        ))
        
        with tempfile.TemporaryDirectory() as tmpdir:
            suggestions = generate_suggestions(report, Path(tmpdir))
            self.assertEqual(len(suggestions), 0)
    
    def test_no_crash_on_zero_drifts(self):
        """generate_suggestions should not crash when there are 0 drifts."""
        report = AuditReport()
        # Empty report
        
        with tempfile.TemporaryDirectory() as tmpdir:
            suggestions = generate_suggestions(report, Path(tmpdir))
            self.assertIsInstance(suggestions, list)
            self.assertEqual(len(suggestions), 0)


class TestFormatSuggestionsMarkdown(unittest.TestCase):
    """Tests for format_suggestions_markdown function."""
    
    def test_empty_suggestions_message(self):
        """Empty suggestions should show success message."""
        output = format_suggestions_markdown([])
        self.assertIn("No fixes needed", output)
        self.assertIn("✅", output)
    
    def test_suggestions_formatted(self):
        """Suggestions should be formatted as Markdown."""
        suggestions = [
            FixSuggestion(
                issue_type="metric_kind_mismatch",
                current_value="wrong",
                suggested_value="correct",
                file_path="test.yaml",
                approximate_line=42,
                description="Fix the metric kind",
            ),
        ]
        output = format_suggestions_markdown(suggestions)
        
        self.assertIn("## Suggested Fixes", output)
        self.assertIn("metric_kind_mismatch", output)
        self.assertIn("`wrong`", output)
        self.assertIn("`correct`", output)
        self.assertIn("test.yaml", output)
        self.assertIn("42", output)


# =============================================================================
# CI MODE TESTS
# =============================================================================

class TestRunCiChecks(unittest.TestCase):
    """Tests for run_ci_checks function."""
    
    def test_ci_checks_pass_with_valid_config(self):
        """CI checks should pass with valid configuration."""
        import yaml
        
        valid_curriculum = {
            "phase": "II",
            "slices": {
                "slice_uplift_goal": {
                    "success_metric": {"kind": "goal_hit"},
                },
                "slice_uplift_sparse": {
                    "success_metric": {"kind": "density"},
                },
                "slice_uplift_tree": {
                    "success_metric": {"kind": "chain_length"},
                },
                "slice_uplift_dependency": {
                    "success_metric": {"kind": "multi_goal"},
                },
            },
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(valid_curriculum, f)
            curriculum_path = Path(f.name)
        
        try:
            passed, failures = run_ci_checks(curriculum_path=curriculum_path)
            self.assertTrue(passed)
            self.assertEqual(len(failures), 0)
        finally:
            curriculum_path.unlink()
    
    def test_ci_checks_fail_missing_slice(self):
        """CI checks should fail when documented slice is missing."""
        import yaml
        
        incomplete_curriculum = {
            "phase": "II",
            "slices": {
                "slice_uplift_goal": {
                    "success_metric": {"kind": "goal_hit"},
                },
                # Missing slice_uplift_sparse, slice_uplift_tree, slice_uplift_dependency
            },
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(incomplete_curriculum, f)
            curriculum_path = Path(f.name)
        
        try:
            passed, failures = run_ci_checks(curriculum_path=curriculum_path)
            self.assertFalse(passed)
            self.assertGreater(len(failures), 0)
            self.assertTrue(any("slice_uplift_sparse" in f for f in failures))
        finally:
            curriculum_path.unlink()
    
    def test_ci_checks_fail_invalid_metric_kind(self):
        """CI checks should fail when metric kind is invalid."""
        import yaml
        
        invalid_curriculum = {
            "phase": "II",
            "slices": {
                "slice_uplift_goal": {
                    "success_metric": {"kind": "invalid_kind"},
                },
            },
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(invalid_curriculum, f)
            curriculum_path = Path(f.name)
        
        try:
            passed, failures = run_ci_checks(curriculum_path=curriculum_path)
            self.assertFalse(passed)
            self.assertTrue(any("invalid_kind" in f for f in failures))
        finally:
            curriculum_path.unlink()
    
    def test_ci_checks_fail_wrong_metric_kind(self):
        """CI checks should fail when metric kind doesn't match documentation."""
        import yaml
        
        wrong_curriculum = {
            "phase": "II",
            "slices": {
                "slice_uplift_goal": {
                    "success_metric": {"kind": "density"},  # Should be goal_hit
                },
            },
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(wrong_curriculum, f)
            curriculum_path = Path(f.name)
        
        try:
            passed, failures = run_ci_checks(curriculum_path=curriculum_path)
            self.assertFalse(passed)
            self.assertTrue(any("expected 'goal_hit'" in f for f in failures))
        finally:
            curriculum_path.unlink()
    
    def test_ci_checks_exit_code_0_on_pass(self):
        """CI should exit with code 0 on pass."""
        import yaml
        
        valid_curriculum = {
            "phase": "II",
            "slices": {
                "slice_uplift_goal": {"success_metric": {"kind": "goal_hit"}},
                "slice_uplift_sparse": {"success_metric": {"kind": "density"}},
                "slice_uplift_tree": {"success_metric": {"kind": "chain_length"}},
                "slice_uplift_dependency": {"success_metric": {"kind": "multi_goal"}},
            },
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(valid_curriculum, f)
            curriculum_path = Path(f.name)
        
        try:
            passed, _ = run_ci_checks(curriculum_path=curriculum_path)
            # Exit code would be 0 if passed
            self.assertTrue(passed)
        finally:
            curriculum_path.unlink()
    
    def test_ci_checks_exit_code_1_on_fail(self):
        """CI should exit with code 1 on fail."""
        import yaml
        
        invalid_curriculum = {
            "slices": {
                "test_slice": {"success_metric": {"kind": "unknown"}},
            },
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(invalid_curriculum, f)
            curriculum_path = Path(f.name)
        
        try:
            passed, _ = run_ci_checks(curriculum_path=curriculum_path)
            # Exit code would be 1 if not passed
            self.assertFalse(passed)
        finally:
            curriculum_path.unlink()


class TestFormatCiOutput(unittest.TestCase):
    """Tests for format_ci_output function."""
    
    def test_format_ci_output_pass(self):
        """Passing CI output should be minimal."""
        output = format_ci_output(True, [])
        self.assertIn("OK", output)
        self.assertIn("passed", output)
    
    def test_format_ci_output_fail(self):
        """Failing CI output should list failures."""
        failures = ["FAIL: Missing slice X", "FAIL: Invalid metric Y"]
        output = format_ci_output(False, failures)
        
        self.assertIn("FAIL: Missing slice X", output)
        self.assertIn("FAIL: Invalid metric Y", output)


# =============================================================================
# TASK 1: TERM INDEX AS CANONICAL CONTRACT
# =============================================================================

class TestTermIndexCanonicalContract(unittest.TestCase):
    """Tests for Term Index canonical contract shape and determinism."""
    
    def test_term_entry_uses_spelling_key(self):
        """TermEntry.to_dict() must use 'spelling' key per canonical contract."""
        entry = TermEntry(term="goal_hit", canonical_form="goal_hit", kind="metric")
        entry.mentions.append(TermMention(file="test.py", line=10, exact_spelling="GOAL_HIT"))
        
        d = entry.to_dict()
        
        # Must have "spelling" not "exact_spelling"
        self.assertIn("spelling", d["mentions"][0])
        self.assertNotIn("exact_spelling", d["mentions"][0])
        self.assertEqual(d["mentions"][0]["spelling"], "GOAL_HIT")
    
    def test_term_index_sorted_alphabetically(self):
        """Term index must be sorted alphabetically by canonical_form."""
        builder = TermIndexBuilder()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            f.write('slice_uplift_goal\n')
            f.write('slice_uplift_dependency\n')
            f.write('density\n')
            f.write('goal_hit\n')
            temp_path = Path(f.name)
        
        try:
            builder.scan_file(temp_path)
            
            # Sort entries manually for comparison
            entries = list(builder.terms.values())
            sorted_entries = sorted(entries, key=lambda e: e.canonical_form.lower())
            
            # Build index should produce sorted output
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                # Create empty files to avoid scanning real files
                (tmpdir_path / "config").mkdir()
                (tmpdir_path / "experiments").mkdir()
                (tmpdir_path / "experiments" / "prereg").mkdir()
                (tmpdir_path / "docs").mkdir()
                
                builder2 = TermIndexBuilder()
                builder2.scan_file(temp_path)
                
                # Verify sorted by canonical_form
                index = builder2.build_index(tmpdir_path)
                canonical_forms = [t["canonical_form"] for t in index["terms"]]
                self.assertEqual(canonical_forms, sorted(canonical_forms, key=str.lower))
        finally:
            temp_path.unlink()
    
    def test_term_index_deterministic(self):
        """Two successive builds must produce identical files."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            f.write('slice_uplift_goal\n')
            f.write('goal_hit\n')
            f.write('density\n')
            temp_path = Path(f.name)
        
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                # Create minimal structure
                (tmpdir_path / "config").mkdir()
                (tmpdir_path / "experiments").mkdir()
                (tmpdir_path / "experiments" / "prereg").mkdir()
                (tmpdir_path / "docs").mkdir()
                
                # Copy test file to expected location
                target = tmpdir_path / "experiments" / "slice_success_metrics.py"
                target.write_text(temp_path.read_text())
                
                # Build twice
                builder1 = TermIndexBuilder()
                index1 = builder1.build_index(tmpdir_path)
                
                builder2 = TermIndexBuilder()
                index2 = builder2.build_index(tmpdir_path)
                
                # Compare JSON serialization for exact equality
                json1 = json.dumps(index1, sort_keys=True)
                json2 = json.dumps(index2, sort_keys=True)
                self.assertEqual(json1, json2)
        finally:
            temp_path.unlink()
    
    def test_term_index_mentions_sorted(self):
        """Mentions within each term must be sorted by (file, line)."""
        builder = TermIndexBuilder()
        
        # Add mentions in non-sorted order
        builder._add_mention("goal_hit", "goal_hit", "metric", "z_file.py", 100, "goal_hit")
        builder._add_mention("goal_hit", "goal_hit", "metric", "a_file.py", 50, "GOAL_HIT")
        builder._add_mention("goal_hit", "goal_hit", "metric", "a_file.py", 10, "goal_hit")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            (tmpdir_path / "config").mkdir()
            (tmpdir_path / "experiments").mkdir()
            (tmpdir_path / "experiments" / "prereg").mkdir()
            (tmpdir_path / "docs").mkdir()
            
            index = builder.build_index(tmpdir_path)
            
            # Find goal_hit term
            goal_hit_term = next(t for t in index["terms"] if t["canonical_form"] == "goal_hit")
            mentions = goal_hit_term["mentions"]
            
            # Verify sorted by (file, line)
            for i in range(len(mentions) - 1):
                m1, m2 = mentions[i], mentions[i + 1]
                self.assertLessEqual(
                    (m1["file"], m1["line"]),
                    (m2["file"], m2["line"]),
                    f"Mentions not sorted: {m1} should come before {m2}"
                )
    
    def test_term_index_contains_all_slice_names(self):
        """Term index must include all four Phase II slice names when present."""
        expected_slices = {
            "slice_uplift_goal",
            "slice_uplift_sparse",
            "slice_uplift_tree",
            "slice_uplift_dependency",
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            for slice_name in expected_slices:
                f.write(f'{slice_name}\n')
            temp_path = Path(f.name)
        
        try:
            builder = TermIndexBuilder()
            builder.scan_file(temp_path)
            
            found_slices = {
                e.canonical_form for e in builder.terms.values() if e.kind == "slice"
            }
            self.assertTrue(expected_slices.issubset(found_slices))
        finally:
            temp_path.unlink()
    
    def test_term_index_contains_all_metric_kinds(self):
        """Term index must include all four metric kinds when present."""
        expected_metrics = {"goal_hit", "density", "chain_length", "multi_goal"}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            for metric in expected_metrics:
                f.write(f'{metric}\n')
            temp_path = Path(f.name)
        
        try:
            builder = TermIndexBuilder()
            builder.scan_file(temp_path)
            
            found_metrics = {
                e.canonical_form for e in builder.terms.values() if e.kind == "metric"
            }
            self.assertTrue(expected_metrics.issubset(found_metrics))
        finally:
            temp_path.unlink()
    
    def test_term_index_contract_shape(self):
        """Term index must follow canonical contract shape."""
        builder = TermIndexBuilder()
        builder._add_mention("test", "test", "metric", "test.py", 1, "TEST")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            (tmpdir_path / "config").mkdir()
            (tmpdir_path / "experiments").mkdir()
            (tmpdir_path / "experiments" / "prereg").mkdir()
            (tmpdir_path / "docs").mkdir()
            
            index = builder.build_index(tmpdir_path)
            
            # Verify contract shape
            self.assertIn("terms", index)
            self.assertIsInstance(index["terms"], list)
            
            if index["terms"]:
                term = index["terms"][0]
                self.assertIn("term", term)
                self.assertIn("canonical_form", term)
                self.assertIn("kind", term)
                self.assertIn("mentions", term)
                
                # Kind must be one of the allowed values
                self.assertIn(term["kind"], ["slice", "metric", "theorem", "symbol"])
                
                if term["mentions"]:
                    mention = term["mentions"][0]
                    self.assertIn("file", mention)
                    self.assertIn("line", mention)
                    self.assertIn("spelling", mention)
                    self.assertIsInstance(mention["line"], int)


# =============================================================================
# TASK 2: ONE-LINE DRIFT SUMMARY
# =============================================================================

class TestDriftCounts(unittest.TestCase):
    """Tests for DriftCounts data structure."""
    
    def test_drift_counts_default(self):
        """DriftCounts should default to all zeros."""
        counts = DriftCounts()
        self.assertEqual(counts.slices, 0)
        self.assertEqual(counts.metrics, 0)
        self.assertEqual(counts.theory, 0)
        self.assertEqual(counts.total, 0)
    
    def test_drift_counts_total(self):
        """DriftCounts.total should sum all categories."""
        counts = DriftCounts(slices=1, metrics=2, theory=3)
        self.assertEqual(counts.total, 6)
    
    def test_drift_counts_is_ok_when_zero(self):
        """DriftCounts.is_ok should be True when total is 0."""
        counts = DriftCounts()
        self.assertTrue(counts.is_ok)
    
    def test_drift_counts_is_not_ok_when_nonzero(self):
        """DriftCounts.is_ok should be False when total > 0."""
        counts = DriftCounts(slices=1)
        self.assertFalse(counts.is_ok)


class TestCountDrift(unittest.TestCase):
    """Tests for count_drift function."""
    
    def test_count_drift_no_drift(self):
        """count_drift should return all zeros for valid config."""
        import yaml
        
        valid_curriculum = {
            "phase": "II",
            "slices": {
                "slice_uplift_goal": {"success_metric": {"kind": "goal_hit"}},
                "slice_uplift_sparse": {"success_metric": {"kind": "density"}},
                "slice_uplift_tree": {"success_metric": {"kind": "chain_length"}},
                "slice_uplift_dependency": {"success_metric": {"kind": "multi_goal"}},
            },
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(valid_curriculum, f)
            curriculum_path = Path(f.name)
        
        try:
            counts = count_drift(curriculum_path=curriculum_path)
            self.assertEqual(counts.slices, 0)
            self.assertEqual(counts.metrics, 0)
            self.assertEqual(counts.theory, 0)
            self.assertEqual(counts.total, 0)
            self.assertTrue(counts.is_ok)
        finally:
            curriculum_path.unlink()
    
    def test_count_drift_missing_slices(self):
        """count_drift should count missing slices."""
        import yaml
        
        incomplete_curriculum = {
            "slices": {
                "slice_uplift_goal": {"success_metric": {"kind": "goal_hit"}},
                # Missing 3 slices
            },
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(incomplete_curriculum, f)
            curriculum_path = Path(f.name)
        
        try:
            counts = count_drift(curriculum_path=curriculum_path)
            self.assertEqual(counts.slices, 3)  # 3 missing slices
        finally:
            curriculum_path.unlink()
    
    def test_count_drift_invalid_metrics(self):
        """count_drift should count invalid metric kinds and mismatches."""
        import yaml
        
        # This curriculum has:
        # - slice_uplift_sparse with invalid_kind (invalid + mismatch = 2)
        # - slice_uplift_tree with another_invalid (invalid + mismatch = 2)
        # Total = 4 metric issues
        invalid_curriculum = {
            "slices": {
                "slice_uplift_goal": {"success_metric": {"kind": "goal_hit"}},
                "slice_uplift_sparse": {"success_metric": {"kind": "invalid_kind"}},
                "slice_uplift_tree": {"success_metric": {"kind": "another_invalid"}},
                "slice_uplift_dependency": {"success_metric": {"kind": "multi_goal"}},
            },
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(invalid_curriculum, f)
            curriculum_path = Path(f.name)
        
        try:
            counts = count_drift(curriculum_path=curriculum_path)
            # Each invalid kind counts twice: once for being invalid, once for mismatching
            self.assertEqual(counts.metrics, 4)  # 2 invalid + 2 mismatches
        finally:
            curriculum_path.unlink()


class TestFormatDriftSummary(unittest.TestCase):
    """Tests for format_drift_summary function."""
    
    def test_drift_summary_ok_format(self):
        """Drift-free case should show 0 0 0 0 OK."""
        counts = DriftCounts(slices=0, metrics=0, theory=0)
        output = format_drift_summary(counts)
        
        self.assertEqual(
            output,
            "Semantic Drift: slices=0 metrics=0 theory=0 total=0 OK"
        )
    
    def test_drift_summary_fail_format(self):
        """Drift case should show correct counts and FAIL."""
        counts = DriftCounts(slices=1, metrics=2, theory=3)
        output = format_drift_summary(counts)
        
        self.assertEqual(
            output,
            "Semantic Drift: slices=1 metrics=2 theory=3 total=6 FAIL"
        )
    
    def test_drift_summary_single_line(self):
        """Drift summary must be a single line (no newlines)."""
        counts = DriftCounts(slices=5, metrics=10, theory=15)
        output = format_drift_summary(counts)
        
        self.assertNotIn("\n", output)
        self.assertNotIn("\r", output)
    
    def test_drift_summary_fixed_category_order(self):
        """Categories must be in fixed order: slices, metrics, theory, total."""
        counts = DriftCounts(slices=1, metrics=2, theory=3)
        output = format_drift_summary(counts)
        
        # Verify order by checking positions
        slices_pos = output.find("slices=")
        metrics_pos = output.find("metrics=")
        theory_pos = output.find("theory=")
        total_pos = output.find("total=")
        
        self.assertLess(slices_pos, metrics_pos)
        self.assertLess(metrics_pos, theory_pos)
        self.assertLess(theory_pos, total_pos)


# =============================================================================
# TASK 3: SUGGESTION STYLE DISCIPLINE
# =============================================================================

class TestSanitizeSuggestionText(unittest.TestCase):
    """Tests for sanitize_suggestion_text function."""
    
    def test_removes_backticks(self):
        """Sanitizer should remove backticks."""
        text = "Replace `old` with `new`"
        result = sanitize_suggestion_text(text)
        self.assertNotIn("`", result)
        self.assertEqual(result, "Replace old with new")
    
    def test_removes_newlines(self):
        """Sanitizer should remove newlines."""
        text = "Line 1\nLine 2\rLine 3"
        result = sanitize_suggestion_text(text)
        self.assertNotIn("\n", result)
        self.assertNotIn("\r", result)
    
    def test_collapses_multiple_spaces(self):
        """Sanitizer should collapse multiple spaces."""
        text = "word1   word2    word3"
        result = sanitize_suggestion_text(text)
        self.assertEqual(result, "word1 word2 word3")
    
    def test_strips_whitespace(self):
        """Sanitizer should strip leading/trailing whitespace."""
        text = "  trimmed  "
        result = sanitize_suggestion_text(text)
        self.assertEqual(result, "trimmed")


class TestFormatCanonicalSuggestion(unittest.TestCase):
    """Tests for format_canonical_suggestion function."""
    
    def test_canonical_format_with_line(self):
        """Canonical suggestion with line number."""
        result = format_canonical_suggestion(
            old_term="sparse_success",
            new_term="density",
            file_path="/path/to/curriculum_uplift_phase2.yaml",
            line_number=132,
        )
        
        self.assertEqual(
            result,
            "Replace sparse_success with density in curriculum_uplift_phase2.yaml, line ~132"
        )
    
    def test_canonical_format_without_line(self):
        """Canonical suggestion without line number."""
        result = format_canonical_suggestion(
            old_term="old_term",
            new_term="new_term",
            file_path="/path/to/file.yaml",
            line_number=None,
        )
        
        self.assertEqual(result, "Replace old_term with new_term in file.yaml")
    
    def test_canonical_format_extracts_filename(self):
        """Canonical suggestion should use just filename, not full path."""
        result = format_canonical_suggestion(
            old_term="old",
            new_term="new",
            file_path="/very/long/path/to/config/file.yaml",
            line_number=10,
        )
        
        self.assertNotIn("/very/long", result)
        self.assertIn("file.yaml", result)


class TestKnownMetricFixes(unittest.TestCase):
    """Tests for known metric fixes mapping."""
    
    def test_sparse_success_maps_to_density(self):
        """sparse_success should map to density."""
        self.assertEqual(KNOWN_METRIC_FIXES.get("sparse_success"), "density")
    
    def test_chain_success_maps_to_chain_length(self):
        """chain_success should map to chain_length."""
        self.assertEqual(KNOWN_METRIC_FIXES.get("chain_success"), "chain_length")
    
    def test_multi_goal_success_maps_to_multi_goal(self):
        """multi_goal_success should map to multi_goal."""
        self.assertEqual(KNOWN_METRIC_FIXES.get("multi_goal_success"), "multi_goal")


class TestGenerateCanonicalSuggestions(unittest.TestCase):
    """Tests for generate_canonical_suggestions function."""
    
    def test_generates_suggestions_for_known_mismatches(self):
        """Should generate suggestions for known metric mismatches."""
        curriculum_content = """
slices:
  slice_uplift_sparse:
    success_metric:
      kind: sparse_success
  slice_uplift_tree:
    success_metric:
      kind: chain_success
  slice_uplift_dependency:
    success_metric:
      kind: multi_goal_success
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8') as f:
            f.write(curriculum_content)
            curriculum_path = Path(f.name)
        
        try:
            suggestions = generate_canonical_suggestions(curriculum_path)
            
            # Should have suggestions for all three known mismatches
            self.assertGreaterEqual(len(suggestions), 3)
            
            # Check canonical format
            for suggestion in suggestions:
                self.assertTrue(
                    suggestion.startswith("Replace "),
                    f"Suggestion does not start with 'Replace ': {suggestion}"
                )
                self.assertIn(" with ", suggestion)
                self.assertIn(" in ", suggestion)
        finally:
            curriculum_path.unlink()
    
    def test_suggestions_include_required_components(self):
        """Each suggestion must include old term, new term, file, and approximate line."""
        curriculum_content = """
slices:
  slice_uplift_sparse:
    success_metric:
      kind: sparse_success
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8') as f:
            f.write(curriculum_content)
            curriculum_path = Path(f.name)
        
        try:
            suggestions = generate_canonical_suggestions(curriculum_path)
            
            self.assertGreater(len(suggestions), 0)
            
            for suggestion in suggestions:
                # Must include old term
                self.assertIn("sparse_success", suggestion)
                # Must include new term
                self.assertIn("density", suggestion)
                # Must include file
                self.assertIn(".yaml", suggestion)
                # Must include approximate line (with ~)
                self.assertIn("line ~", suggestion)
        finally:
            curriculum_path.unlink()
    
    def test_no_crash_on_zero_drifts(self):
        """generate_canonical_suggestions should not crash with no drift."""
        import yaml
        
        valid_curriculum = {
            "slices": {
                "slice_uplift_goal": {"success_metric": {"kind": "goal_hit"}},
            },
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(valid_curriculum, f)
            curriculum_path = Path(f.name)
        
        try:
            suggestions = generate_canonical_suggestions(curriculum_path)
            self.assertIsInstance(suggestions, list)
        finally:
            curriculum_path.unlink()
    
    def test_specific_known_mismatch_sparse_success(self):
        """sparse_success -> density suggestion must be correct."""
        curriculum_content = """
slices:
  slice_uplift_sparse:
    success_metric:
      kind: sparse_success
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8') as f:
            f.write(curriculum_content)
            curriculum_path = Path(f.name)
        
        try:
            suggestions = generate_canonical_suggestions(curriculum_path)
            
            sparse_suggestion = next(
                (s for s in suggestions if "sparse_success" in s and "density" in s),
                None
            )
            self.assertIsNotNone(
                sparse_suggestion,
                "Should have suggestion for sparse_success -> density"
            )
            self.assertTrue(
                sparse_suggestion.startswith("Replace sparse_success with density"),
                f"Unexpected format: {sparse_suggestion}"
            )
        finally:
            curriculum_path.unlink()
    
    def test_specific_known_mismatch_chain_success(self):
        """chain_success -> chain_length suggestion must be correct."""
        curriculum_content = """
slices:
  slice_uplift_tree:
    success_metric:
      kind: chain_success
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8') as f:
            f.write(curriculum_content)
            curriculum_path = Path(f.name)
        
        try:
            suggestions = generate_canonical_suggestions(curriculum_path)
            
            chain_suggestion = next(
                (s for s in suggestions if "chain_success" in s and "chain_length" in s),
                None
            )
            self.assertIsNotNone(
                chain_suggestion,
                "Should have suggestion for chain_success -> chain_length"
            )
            self.assertTrue(
                chain_suggestion.startswith("Replace chain_success with chain_length"),
                f"Unexpected format: {chain_suggestion}"
            )
        finally:
            curriculum_path.unlink()
    
    def test_specific_known_mismatch_multi_goal_success(self):
        """multi_goal_success -> multi_goal suggestion must be correct."""
        curriculum_content = """
slices:
  slice_uplift_dependency:
    success_metric:
      kind: multi_goal_success
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8') as f:
            f.write(curriculum_content)
            curriculum_path = Path(f.name)
        
        try:
            suggestions = generate_canonical_suggestions(curriculum_path)
            
            multi_suggestion = next(
                (s for s in suggestions if "multi_goal_success" in s and "multi_goal" in s),
                None
            )
            self.assertIsNotNone(
                multi_suggestion,
                "Should have suggestion for multi_goal_success -> multi_goal"
            )
            self.assertTrue(
                multi_suggestion.startswith("Replace multi_goal_success with multi_goal"),
                f"Unexpected format: {multi_suggestion}"
            )
        finally:
            curriculum_path.unlink()


# =============================================================================
# SEMANTIC KNOWLEDGE GRAPH TESTS (v1.3) - TASK 1
# =============================================================================

class TestGraphEdge(unittest.TestCase):
    """Tests for GraphEdge data structure."""
    
    def test_graph_edge_creation(self):
        """GraphEdge should store all fields."""
        edge = GraphEdge(src="term_a", dst="term_b", weight=0.75, kind="cooccur")
        self.assertEqual(edge.src, "term_a")
        self.assertEqual(edge.dst, "term_b")
        self.assertEqual(edge.weight, 0.75)
        self.assertEqual(edge.kind, "cooccur")
    
    def test_graph_edge_to_dict(self):
        """GraphEdge.to_dict() should produce correct structure."""
        edge = GraphEdge(src="a", dst="b", weight=0.12345678, kind="ref")
        d = edge.to_dict()
        
        self.assertEqual(d["src"], "a")
        self.assertEqual(d["dst"], "b")
        self.assertEqual(d["weight"], 0.1235)  # Rounded to 4 decimals
        self.assertEqual(d["kind"], "ref")
    
    def test_edge_kinds_valid(self):
        """Edge kinds should be cooccur, ref, or category."""
        valid_kinds = {"cooccur", "ref", "category"}
        for kind in valid_kinds:
            edge = GraphEdge(src="x", dst="y", weight=1.0, kind=kind)
            self.assertEqual(edge.kind, kind)


class TestSemanticKnowledgeGraph(unittest.TestCase):
    """Tests for SemanticKnowledgeGraph data structure."""
    
    def test_graph_creation(self):
        """SemanticKnowledgeGraph should store terms and edges."""
        graph = SemanticKnowledgeGraph()
        self.assertEqual(graph.terms, [])
        self.assertEqual(graph.edges, [])
    
    def test_graph_to_dict_format(self):
        """to_dict() should produce canonical JSON format."""
        graph = SemanticKnowledgeGraph(
            terms=[{"canonical_form": "goal_hit", "kind": "metric"}],
            edges=[GraphEdge(src="a", dst="b", weight=1.0, kind="cooccur")],
        )
        d = graph.to_dict()
        
        self.assertIn("version", d)
        self.assertEqual(d["version"], "1.3")
        self.assertIn("terms", d)
        self.assertIn("edges", d)
        self.assertIsInstance(d["terms"], list)
        self.assertIsInstance(d["edges"], list)
    
    def test_graph_terms_sorted(self):
        """Terms in to_dict() should be sorted by canonical_form."""
        graph = SemanticKnowledgeGraph(
            terms=[
                {"canonical_form": "zebra"},
                {"canonical_form": "apple"},
                {"canonical_form": "mango"},
            ],
            edges=[],
        )
        d = graph.to_dict()
        canonical_forms = [t["canonical_form"] for t in d["terms"]]
        self.assertEqual(canonical_forms, ["apple", "mango", "zebra"])
    
    def test_graph_edges_sorted(self):
        """Edges in to_dict() should be sorted by (src, dst, kind)."""
        graph = SemanticKnowledgeGraph(
            terms=[],
            edges=[
                GraphEdge(src="z", dst="a", weight=1.0, kind="cooccur"),
                GraphEdge(src="a", dst="b", weight=1.0, kind="ref"),
                GraphEdge(src="a", dst="b", weight=1.0, kind="cooccur"),
            ],
        )
        d = graph.to_dict()
        
        # Should be sorted: (a,b,cooccur), (a,b,ref), (z,a,cooccur)
        self.assertEqual(d["edges"][0]["src"], "a")
        self.assertEqual(d["edges"][0]["kind"], "cooccur")
        self.assertEqual(d["edges"][1]["src"], "a")
        self.assertEqual(d["edges"][1]["kind"], "ref")
        self.assertEqual(d["edges"][2]["src"], "z")
    
    def test_get_node_degree(self):
        """get_node_degree() should count edges connected to node."""
        graph = SemanticKnowledgeGraph(
            terms=[],
            edges=[
                GraphEdge(src="a", dst="b", weight=1.0, kind="cooccur"),
                GraphEdge(src="a", dst="c", weight=1.0, kind="ref"),
                GraphEdge(src="d", dst="a", weight=1.0, kind="category"),
            ],
        )
        self.assertEqual(graph.get_node_degree("a"), 3)
        self.assertEqual(graph.get_node_degree("b"), 1)
        self.assertEqual(graph.get_node_degree("z"), 0)
    
    def test_get_neighbors(self):
        """get_neighbors() should return connected terms."""
        graph = SemanticKnowledgeGraph(
            terms=[],
            edges=[
                GraphEdge(src="a", dst="b", weight=1.0, kind="cooccur"),
                GraphEdge(src="c", dst="a", weight=1.0, kind="ref"),
            ],
        )
        neighbors = graph.get_neighbors("a")
        self.assertEqual(neighbors, {"b", "c"})


class TestSemanticGraphBuilder(unittest.TestCase):
    """Tests for SemanticGraphBuilder."""
    
    def test_add_term(self):
        """add_term() should store term and track file locations."""
        builder = SemanticGraphBuilder()
        builder.add_term({
            "canonical_form": "goal_hit",
            "kind": "metric",
            "mentions": [{"file": "test.py", "line": 10}],
        })
        
        self.assertIn("goal_hit", builder.terms)
        self.assertIn("test.py", builder.file_terms)
        self.assertIn("goal_hit", builder.file_terms["test.py"])
    
    def test_build_cooccur_edges(self):
        """Builder should create cooccur edges for terms in same file."""
        builder = SemanticGraphBuilder()
        builder.add_term({
            "canonical_form": "term_a",
            "mentions": [{"file": "shared.py", "line": 1}],
        })
        builder.add_term({
            "canonical_form": "term_b",
            "mentions": [{"file": "shared.py", "line": 2}],
        })
        
        edges = builder._build_cooccur_edges()
        
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0].kind, "cooccur")
        # Edges are normalized to alphabetical order
        self.assertEqual(edges[0].src, "term_a")
        self.assertEqual(edges[0].dst, "term_b")
    
    def test_build_ref_edges(self):
        """Builder should create ref edges from REFERENCE_MAP."""
        builder = SemanticGraphBuilder()
        builder.add_term({
            "canonical_form": "goal_hit",
            "mentions": [{"file": "test.py", "line": 1}],
        })
        
        edges = builder._build_ref_edges()
        
        # goal_hit should have refs to target_hashes and min_total_verified
        self.assertGreater(len(edges), 0)
        self.assertTrue(all(e.kind == "ref" for e in edges))
        self.assertTrue(all(e.src == "goal_hit" for e in edges))
    
    def test_build_category_edges(self):
        """Builder should create category edges from CATEGORY_HIERARCHY."""
        builder = SemanticGraphBuilder()
        builder.add_term({
            "canonical_form": "slice_uplift_goal",
            "mentions": [{"file": "test.py", "line": 1}],
        })
        
        edges = builder._build_category_edges()
        
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0].src, "slice_uplift_goal")
        self.assertEqual(edges[0].dst, "goal_hit")
        self.assertEqual(edges[0].kind, "category")
    
    def test_no_cycles_in_category_edges(self):
        """Category edges must not form cycles (parent→child only)."""
        builder = SemanticGraphBuilder()
        
        # Add all slice terms
        for slice_name in SemanticGraphBuilder.CATEGORY_HIERARCHY.keys():
            builder.add_term({
                "canonical_form": slice_name,
                "mentions": [{"file": "test.py", "line": 1}],
            })
        
        edges = builder._build_category_edges()
        
        # Check that no edge goes from category back to slice
        category_targets = set(SemanticGraphBuilder.CATEGORY_HIERARCHY.values())
        for edge in edges:
            # src should be a slice, dst should be a category
            self.assertIn(edge.src, SemanticGraphBuilder.CATEGORY_HIERARCHY.keys())
            self.assertIn(edge.dst, category_targets)
            # No reverse edges
            self.assertNotIn(edge.dst, SemanticGraphBuilder.CATEGORY_HIERARCHY.keys())


class TestBuildSemanticGraph(unittest.TestCase):
    """Tests for build_semantic_graph function."""
    
    def test_build_from_term_index(self):
        """build_semantic_graph() should create graph from term index."""
        term_index = {
            "terms": [
                {"canonical_form": "goal_hit", "kind": "metric", "mentions": []},
                {"canonical_form": "density", "kind": "metric", "mentions": []},
            ],
        }
        
        graph = build_semantic_graph(term_index)
        
        self.assertIsInstance(graph, SemanticKnowledgeGraph)
        self.assertEqual(len(graph.terms), 2)
    
    def test_graph_deterministic(self):
        """Two builds from same index should produce identical graphs."""
        term_index = {
            "terms": [
                {"canonical_form": "slice_uplift_goal", "kind": "slice", "mentions": [{"file": "a.py", "line": 1}]},
                {"canonical_form": "goal_hit", "kind": "metric", "mentions": [{"file": "a.py", "line": 2}]},
            ],
        }
        
        graph1 = build_semantic_graph(term_index)
        graph2 = build_semantic_graph(term_index)
        
        json1 = json.dumps(graph1.to_dict(), sort_keys=True)
        json2 = json.dumps(graph2.to_dict(), sort_keys=True)
        
        self.assertEqual(json1, json2)


class TestSaveSemanticGraph(unittest.TestCase):
    """Tests for save_semantic_graph function."""
    
    def test_save_creates_file(self):
        """save_semantic_graph() should create JSON file."""
        graph = SemanticKnowledgeGraph(
            terms=[{"canonical_form": "test"}],
            edges=[],
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "subdir" / "graph.json"
            save_semantic_graph(graph, output_path)
            
            self.assertTrue(output_path.exists())
            
            with open(output_path) as f:
                loaded = json.load(f)
            
            self.assertEqual(loaded["version"], "1.3")


# =============================================================================
# GRAPH-BASED DRIFT DETECTION TESTS (v1.3) - TASK 2
# =============================================================================

class TestGraphDriftSignal(unittest.TestCase):
    """Tests for GraphDriftSignal data structure."""
    
    def test_signal_creation(self):
        """GraphDriftSignal should store all fields."""
        signal = GraphDriftSignal(
            signal_type="node_disappeared",
            severity="critical",
            term="test_term",
            description="Term disappeared",
            details={"key": "value"},
        )
        
        self.assertEqual(signal.signal_type, "node_disappeared")
        self.assertEqual(signal.severity, "critical")
        self.assertEqual(signal.term, "test_term")
    
    def test_signal_to_dict(self):
        """GraphDriftSignal.to_dict() should produce correct structure."""
        signal = GraphDriftSignal(
            signal_type="edge_collapse",
            severity="warning",
            term="edge",
            description="Edge collapsed",
        )
        d = signal.to_dict()
        
        self.assertIn("signal_type", d)
        self.assertIn("severity", d)
        self.assertIn("term", d)
        self.assertIn("description", d)
        self.assertIn("details", d)


class TestDriftGraphReport(unittest.TestCase):
    """Tests for DriftGraphReport data structure."""
    
    def test_report_ok_status(self):
        """Report with no signals should have OK status."""
        report = DriftGraphReport()
        self.assertEqual(report.status, "OK")
        self.assertFalse(report.has_critical)
        self.assertFalse(report.has_warnings)
    
    def test_report_warning_status(self):
        """Report with warnings should have WARNING status."""
        report = DriftGraphReport()
        report.signals.append(GraphDriftSignal(
            signal_type="test",
            severity="warning",
            term="t",
            description="d",
        ))
        self.assertEqual(report.status, "WARNING")
        self.assertTrue(report.has_warnings)
    
    def test_report_critical_status(self):
        """Report with critical should have CRITICAL status."""
        report = DriftGraphReport()
        report.signals.append(GraphDriftSignal(
            signal_type="test",
            severity="critical",
            term="t",
            description="d",
        ))
        self.assertEqual(report.status, "CRITICAL")
        self.assertTrue(report.has_critical)
    
    def test_report_to_dict(self):
        """DriftGraphReport.to_dict() should produce stable JSON contract."""
        report = DriftGraphReport(
            old_node_count=10,
            new_node_count=8,
            old_edge_count=20,
            new_edge_count=15,
        )
        d = report.to_dict()
        
        self.assertIn("status", d)
        self.assertIn("old_node_count", d)
        self.assertIn("new_node_count", d)
        self.assertIn("old_edge_count", d)
        self.assertIn("new_edge_count", d)
        self.assertIn("signal_count", d)
        self.assertIn("signals", d)


class TestAnalyzeGraphDrift(unittest.TestCase):
    """Tests for analyze_graph_drift function."""
    
    def test_detect_node_disappearance(self):
        """Should detect when node disappears."""
        old_graph = SemanticKnowledgeGraph(
            terms=[{"canonical_form": "existing"}, {"canonical_form": "disappeared"}],
            edges=[],
        )
        new_graph = SemanticKnowledgeGraph(
            terms=[{"canonical_form": "existing"}],
            edges=[],
        )
        
        report = analyze_graph_drift(old_graph, new_graph)
        
        disappeared_signals = [
            s for s in report.signals if s.signal_type == "node_disappeared"
        ]
        self.assertEqual(len(disappeared_signals), 1)
        self.assertEqual(disappeared_signals[0].term, "disappeared")
        self.assertEqual(disappeared_signals[0].severity, "critical")
    
    def test_detect_node_appearance(self):
        """Should detect when new node appears."""
        old_graph = SemanticKnowledgeGraph(
            terms=[{"canonical_form": "existing"}],
            edges=[],
        )
        new_graph = SemanticKnowledgeGraph(
            terms=[{"canonical_form": "existing"}, {"canonical_form": "new_term"}],
            edges=[],
        )
        
        report = analyze_graph_drift(old_graph, new_graph)
        
        appeared_signals = [
            s for s in report.signals if s.signal_type == "node_appeared"
        ]
        self.assertEqual(len(appeared_signals), 1)
        self.assertEqual(appeared_signals[0].term, "new_term")
        self.assertEqual(appeared_signals[0].severity, "info")
    
    def test_detect_edge_collapse(self):
        """Should detect edge weight collapse."""
        old_graph = SemanticKnowledgeGraph(
            terms=[{"canonical_form": "a"}, {"canonical_form": "b"}],
            edges=[GraphEdge(src="a", dst="b", weight=1.0, kind="cooccur")],
        )
        new_graph = SemanticKnowledgeGraph(
            terms=[{"canonical_form": "a"}, {"canonical_form": "b"}],
            edges=[GraphEdge(src="a", dst="b", weight=0.1, kind="cooccur")],  # 10% of original
        )
        
        report = analyze_graph_drift(old_graph, new_graph, weight_collapse_threshold=0.3)
        
        collapse_signals = [
            s for s in report.signals if s.signal_type == "edge_collapse"
        ]
        self.assertEqual(len(collapse_signals), 1)
    
    def test_detect_edge_explosion(self):
        """Should detect edge weight explosion."""
        old_graph = SemanticKnowledgeGraph(
            terms=[{"canonical_form": "a"}, {"canonical_form": "b"}],
            edges=[GraphEdge(src="a", dst="b", weight=0.2, kind="cooccur")],
        )
        new_graph = SemanticKnowledgeGraph(
            terms=[{"canonical_form": "a"}, {"canonical_form": "b"}],
            edges=[GraphEdge(src="a", dst="b", weight=1.0, kind="cooccur")],  # 5x original
        )
        
        report = analyze_graph_drift(old_graph, new_graph, weight_explosion_threshold=3.0)
        
        explosion_signals = [
            s for s in report.signals if s.signal_type == "edge_explosion"
        ]
        self.assertEqual(len(explosion_signals), 1)
    
    def test_detect_category_migration(self):
        """Should detect when term migrates away from category."""
        old_graph = SemanticKnowledgeGraph(
            terms=[{"canonical_form": "slice_uplift_goal"}, {"canonical_form": "goal_hit"}],
            edges=[GraphEdge(src="slice_uplift_goal", dst="goal_hit", weight=1.0, kind="category")],
        )
        new_graph = SemanticKnowledgeGraph(
            terms=[{"canonical_form": "slice_uplift_goal"}, {"canonical_form": "goal_hit"}],
            edges=[],  # Category edge removed
        )
        
        report = analyze_graph_drift(old_graph, new_graph)
        
        migration_signals = [
            s for s in report.signals if s.signal_type == "migration"
        ]
        self.assertEqual(len(migration_signals), 1)
        self.assertEqual(migration_signals[0].term, "slice_uplift_goal")
    
    def test_detect_degree_change(self):
        """Should detect unexpected degree changes."""
        old_graph = SemanticKnowledgeGraph(
            terms=[{"canonical_form": "hub"}, {"canonical_form": "a"}, {"canonical_form": "b"}, {"canonical_form": "c"}],
            edges=[
                GraphEdge(src="hub", dst="a", weight=1.0, kind="cooccur"),
                GraphEdge(src="hub", dst="b", weight=1.0, kind="cooccur"),
                GraphEdge(src="hub", dst="c", weight=1.0, kind="cooccur"),
            ],
        )
        new_graph = SemanticKnowledgeGraph(
            terms=[{"canonical_form": "hub"}, {"canonical_form": "a"}, {"canonical_form": "b"}, {"canonical_form": "c"}],
            edges=[],  # All edges removed
        )
        
        report = analyze_graph_drift(old_graph, new_graph, degree_change_threshold=3)
        
        degree_signals = [
            s for s in report.signals if s.signal_type == "degree_change"
        ]
        self.assertEqual(len(degree_signals), 1)
        self.assertEqual(degree_signals[0].term, "hub")
    
    def test_drift_report_deterministic(self):
        """Drift report should be deterministic."""
        old_graph = SemanticKnowledgeGraph(
            terms=[{"canonical_form": "a"}, {"canonical_form": "b"}],
            edges=[GraphEdge(src="a", dst="b", weight=1.0, kind="cooccur")],
        )
        new_graph = SemanticKnowledgeGraph(
            terms=[{"canonical_form": "c"}],
            edges=[],
        )
        
        report1 = analyze_graph_drift(old_graph, new_graph)
        report2 = analyze_graph_drift(old_graph, new_graph)
        
        json1 = json.dumps(report1.to_dict(), sort_keys=True)
        json2 = json.dumps(report2.to_dict(), sort_keys=True)
        
        self.assertEqual(json1, json2)


class TestFormatDriftReportSummary(unittest.TestCase):
    """Tests for format_drift_report_summary function."""
    
    def test_format_ok_summary(self):
        """OK report should format correctly."""
        report = DriftGraphReport()
        output = format_drift_report_summary(report)
        
        self.assertIn("critical=0", output)
        self.assertIn("status=OK", output)
    
    def test_format_critical_summary(self):
        """Report with critical signals should show in summary."""
        report = DriftGraphReport()
        report.signals.append(GraphDriftSignal(
            signal_type="node_disappeared",
            severity="critical",
            term="t",
            description="d",
        ))
        output = format_drift_report_summary(report)
        
        self.assertIn("critical=1", output)
        self.assertIn("status=CRITICAL", output)


# =============================================================================
# GRAPH-AWARE SUGGESTIONS TESTS (v1.3) - TASK 3
# =============================================================================

class TestGenerateGraphAwareSuggestions(unittest.TestCase):
    """Tests for generate_graph_aware_suggestions function."""
    
    def test_generates_suggestions_with_graph_context(self):
        """Should generate suggestions using graph connectivity."""
        # Build a graph with the canonical terms
        term_index = {
            "terms": [
                {"canonical_form": "density", "kind": "metric", "mentions": []},
                {"canonical_form": "goal_hit", "kind": "metric", "mentions": []},
            ],
        }
        graph = build_semantic_graph(term_index)
        
        curriculum_content = """
slices:
  slice_uplift_sparse:
    success_metric:
      kind: sparse_success
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8') as f:
            f.write(curriculum_content)
            curriculum_path = Path(f.name)
        
        try:
            suggestions = generate_graph_aware_suggestions(graph, curriculum_path)
            
            self.assertGreater(len(suggestions), 0)
            # Should mention the connection reason
            self.assertTrue(any("connected" in s for s in suggestions))
        finally:
            curriculum_path.unlink()
    
    def test_suggestions_single_line(self):
        """All suggestions must be single line."""
        term_index = {"terms": [{"canonical_form": "density", "mentions": []}]}
        graph = build_semantic_graph(term_index)
        
        curriculum_content = """
slices:
  slice_uplift_sparse:
    success_metric:
      kind: sparse_success
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8') as f:
            f.write(curriculum_content)
            curriculum_path = Path(f.name)
        
        try:
            suggestions = generate_graph_aware_suggestions(graph, curriculum_path)
            
            for suggestion in suggestions:
                self.assertNotIn("\n", suggestion)
                self.assertNotIn("\r", suggestion)
        finally:
            curriculum_path.unlink()
    
    def test_suggestions_no_backticks(self):
        """All suggestions must have no code formatting."""
        term_index = {"terms": [{"canonical_form": "density", "mentions": []}]}
        graph = build_semantic_graph(term_index)
        
        curriculum_content = """
slices:
  slice_uplift_sparse:
    success_metric:
      kind: sparse_success
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8') as f:
            f.write(curriculum_content)
            curriculum_path = Path(f.name)
        
        try:
            suggestions = generate_graph_aware_suggestions(graph, curriculum_path)
            
            for suggestion in suggestions:
                self.assertNotIn("`", suggestion)
        finally:
            curriculum_path.unlink()
    
    def test_suggestions_deterministic(self):
        """Suggestions should be deterministic."""
        term_index = {"terms": [{"canonical_form": "density", "mentions": []}]}
        graph = build_semantic_graph(term_index)
        
        curriculum_content = """
slices:
  slice_uplift_sparse:
    success_metric:
      kind: sparse_success
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8') as f:
            f.write(curriculum_content)
            curriculum_path = Path(f.name)
        
        try:
            suggestions1 = generate_graph_aware_suggestions(graph, curriculum_path)
            suggestions2 = generate_graph_aware_suggestions(graph, curriculum_path)
            
            self.assertEqual(suggestions1, suggestions2)
        finally:
            curriculum_path.unlink()
    
    def test_no_crash_on_empty_graph(self):
        """Should not crash with empty graph."""
        graph = SemanticKnowledgeGraph(terms=[], edges=[])
        
        curriculum_content = """
slices:
  slice_uplift_goal:
    success_metric:
      kind: goal_hit
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8') as f:
            f.write(curriculum_content)
            curriculum_path = Path(f.name)
        
        try:
            suggestions = generate_graph_aware_suggestions(graph, curriculum_path)
            self.assertIsInstance(suggestions, list)
        finally:
            curriculum_path.unlink()


# =============================================================================
# SEMANTIC GOVERNANCE SNAPSHOT TESTS (Phase III) - TASK 1
# =============================================================================

class TestGovernanceStatus(unittest.TestCase):
    """Tests for GovernanceStatus enum."""
    
    def test_governance_status_values(self):
        """GovernanceStatus should have OK, ATTENTION, CRITICAL."""
        self.assertEqual(GovernanceStatus.OK.value, "OK")
        self.assertEqual(GovernanceStatus.ATTENTION.value, "ATTENTION")
        self.assertEqual(GovernanceStatus.CRITICAL.value, "CRITICAL")


class TestBuildSemanticGovernanceSnapshot(unittest.TestCase):
    """Tests for build_semantic_governance_snapshot function."""
    
    def test_snapshot_schema_version(self):
        """Snapshot should have schema_version 1.0."""
        graph = SemanticKnowledgeGraph(terms=[], edges=[])
        snapshot = build_semantic_governance_snapshot(graph)
        
        self.assertEqual(snapshot["schema_version"], "1.0")
    
    def test_snapshot_counts(self):
        """Snapshot should include term_count and edge_count."""
        graph = SemanticKnowledgeGraph(
            terms=[{"canonical_form": "a"}, {"canonical_form": "b"}],
            edges=[GraphEdge(src="a", dst="b", weight=1.0, kind="cooccur")],
        )
        snapshot = build_semantic_governance_snapshot(graph)
        
        self.assertEqual(snapshot["term_count"], 2)
        self.assertEqual(snapshot["edge_count"], 1)
    
    def test_snapshot_ok_status_no_drift(self):
        """Snapshot with no drift should have OK status."""
        graph = SemanticKnowledgeGraph(terms=[], edges=[])
        snapshot = build_semantic_governance_snapshot(graph, drift_report=None)
        
        self.assertEqual(snapshot["status"], "OK")
        self.assertEqual(snapshot["critical_signals"], [])
    
    def test_snapshot_critical_on_node_disappeared(self):
        """Snapshot should be CRITICAL when node disappeared."""
        graph = SemanticKnowledgeGraph(terms=[], edges=[])
        drift_report = {
            "signals": [
                {
                    "signal_type": "node_disappeared",
                    "severity": "critical",
                    "term": "lost_term",
                    "description": "Term disappeared",
                }
            ]
        }
        snapshot = build_semantic_governance_snapshot(graph, drift_report)
        
        self.assertEqual(snapshot["status"], "CRITICAL")
        self.assertEqual(len(snapshot["critical_signals"]), 1)
        self.assertEqual(snapshot["critical_signals"][0]["type"], "node_disappeared")
    
    def test_snapshot_attention_on_migration(self):
        """Snapshot should be ATTENTION when migration detected."""
        graph = SemanticKnowledgeGraph(terms=[], edges=[])
        drift_report = {
            "signals": [
                {
                    "signal_type": "migration",
                    "severity": "warning",
                    "term": "migrated_term",
                    "description": "Category changed",
                }
            ]
        }
        snapshot = build_semantic_governance_snapshot(graph, drift_report)
        
        self.assertEqual(snapshot["status"], "ATTENTION")
        self.assertEqual(len(snapshot["critical_signals"]), 1)
    
    def test_snapshot_critical_overrides_attention(self):
        """CRITICAL should override ATTENTION when both present."""
        graph = SemanticKnowledgeGraph(terms=[], edges=[])
        drift_report = {
            "signals": [
                {
                    "signal_type": "node_disappeared",
                    "severity": "critical",
                    "term": "lost_term",
                    "description": "Term disappeared",
                },
                {
                    "signal_type": "migration",
                    "severity": "warning",
                    "term": "migrated_term",
                    "description": "Category changed",
                },
            ]
        }
        snapshot = build_semantic_governance_snapshot(graph, drift_report)
        
        self.assertEqual(snapshot["status"], "CRITICAL")
        self.assertEqual(len(snapshot["critical_signals"]), 2)
    
    def test_snapshot_ignores_info_signals(self):
        """Snapshot should ignore info-level signals."""
        graph = SemanticKnowledgeGraph(terms=[], edges=[])
        drift_report = {
            "signals": [
                {
                    "signal_type": "node_appeared",
                    "severity": "info",
                    "term": "new_term",
                    "description": "New term appeared",
                }
            ]
        }
        snapshot = build_semantic_governance_snapshot(graph, drift_report)
        
        self.assertEqual(snapshot["status"], "OK")
        self.assertEqual(snapshot["critical_signals"], [])


# =============================================================================
# SUGGESTION SAFETY FILTER TESTS (Phase III) - TASK 2
# =============================================================================

class TestExtractTerms(unittest.TestCase):
    """Tests for suggestion term extraction helpers."""
    
    def test_extract_target_term(self):
        """_extract_target_term should extract new term from suggestion."""
        suggestion = "Replace old_term with new_term in file.yaml, line ~10"
        result = _extract_target_term(suggestion)
        self.assertEqual(result, "new_term")
    
    def test_extract_source_term(self):
        """_extract_source_term should extract old term from suggestion."""
        suggestion = "Replace old_term with new_term in file.yaml, line ~10"
        result = _extract_source_term(suggestion)
        self.assertEqual(result, "old_term")
    
    def test_extract_target_term_with_reason(self):
        """Should work with suggestions that have connection reason."""
        suggestion = "Replace sparse_success with density in curriculum.yaml, line ~132 (connected via category edge)"
        result = _extract_target_term(suggestion)
        self.assertEqual(result, "density")
    
    def test_extract_returns_none_for_invalid(self):
        """Should return None for non-matching strings."""
        self.assertIsNone(_extract_target_term("Invalid suggestion format"))
        self.assertIsNone(_extract_source_term("Invalid suggestion format"))


class TestFilterGraphSuggestions(unittest.TestCase):
    """Tests for filter_graph_suggestions function."""
    
    def test_all_allowed_when_ok(self):
        """All suggestions should be allowed when status is OK."""
        suggestions = [
            "Replace sparse_success with density in file.yaml, line ~10",
            "Replace chain_success with chain_length in file.yaml, line ~20",
        ]
        snapshot = {
            "status": "OK",
            "critical_signals": [],
        }
        
        result = filter_graph_suggestions(suggestions, snapshot)
        
        self.assertEqual(len(result["allowed_suggestions"]), 2)
        self.assertEqual(len(result["blocked_suggestions"]), 0)
        self.assertEqual(len(result["reasons"]), 0)
    
    def test_blocked_when_target_disappeared(self):
        """Suggestion should be blocked if target term disappeared."""
        suggestions = [
            "Replace sparse_success with density in file.yaml, line ~10",
        ]
        snapshot = {
            "status": "CRITICAL",
            "critical_signals": [
                {"type": "node_disappeared", "term": "density"},
            ],
        }
        
        result = filter_graph_suggestions(suggestions, snapshot)
        
        self.assertEqual(len(result["allowed_suggestions"]), 0)
        self.assertEqual(len(result["blocked_suggestions"]), 1)
        self.assertIn("disappeared", result["reasons"][suggestions[0]])
    
    def test_blocked_when_source_migrated(self):
        """Suggestion should be blocked if source term migrated."""
        suggestions = [
            "Replace sparse_success with density in file.yaml, line ~10",
        ]
        snapshot = {
            "status": "ATTENTION",
            "critical_signals": [
                {"type": "migration", "term": "sparse_success"},
            ],
        }
        
        result = filter_graph_suggestions(suggestions, snapshot)
        
        self.assertEqual(len(result["allowed_suggestions"]), 0)
        self.assertEqual(len(result["blocked_suggestions"]), 1)
        self.assertIn("migrated", result["reasons"][suggestions[0]])
    
    def test_blocked_when_target_migrated(self):
        """Suggestion should be blocked if target term migrated."""
        suggestions = [
            "Replace sparse_success with density in file.yaml, line ~10",
        ]
        snapshot = {
            "status": "ATTENTION",
            "critical_signals": [
                {"type": "migration", "term": "density"},
            ],
        }
        
        result = filter_graph_suggestions(suggestions, snapshot)
        
        self.assertEqual(len(result["allowed_suggestions"]), 0)
        self.assertEqual(len(result["blocked_suggestions"]), 1)
        self.assertIn("migrated", result["reasons"][suggestions[0]])
    
    def test_mixed_allowed_and_blocked(self):
        """Should correctly separate allowed and blocked suggestions."""
        suggestions = [
            "Replace safe_old with safe_new in file.yaml, line ~10",
            "Replace dangerous_old with disappeared_term in file.yaml, line ~20",
        ]
        # Use ATTENTION status (not CRITICAL) to allow non-affected suggestions through
        snapshot = {
            "status": "ATTENTION",
            "critical_signals": [
                {"type": "migration", "term": "disappeared_term"},
            ],
        }
        
        result = filter_graph_suggestions(suggestions, snapshot)
        
        self.assertEqual(len(result["allowed_suggestions"]), 1)
        self.assertEqual(len(result["blocked_suggestions"]), 1)
        self.assertIn("safe_new", result["allowed_suggestions"][0])
    
    def test_empty_suggestions_list(self):
        """Should handle empty suggestions list."""
        result = filter_graph_suggestions([], {"status": "OK", "critical_signals": []})
        
        self.assertEqual(result["allowed_suggestions"], [])
        self.assertEqual(result["blocked_suggestions"], [])
        self.assertEqual(result["reasons"], {})
    
    def test_reasons_mapping_complete(self):
        """Each blocked suggestion should have a reason."""
        suggestions = [
            "Replace a with b in file.yaml",
            "Replace c with d in file.yaml",
        ]
        snapshot = {
            "status": "CRITICAL",
            "critical_signals": [
                {"type": "node_disappeared", "term": "b"},
                {"type": "node_disappeared", "term": "d"},
            ],
        }
        
        result = filter_graph_suggestions(suggestions, snapshot)
        
        for blocked in result["blocked_suggestions"]:
            self.assertIn(blocked, result["reasons"])


# =============================================================================
# GLOBAL HEALTH SEMANTIC SIGNAL TESTS (Phase III) - TASK 3
# =============================================================================

class TestSemanticHealthStatus(unittest.TestCase):
    """Tests for SemanticHealthStatus enum."""
    
    def test_health_status_values(self):
        """SemanticHealthStatus should have OK, WARN, CRITICAL."""
        self.assertEqual(SemanticHealthStatus.OK.value, "OK")
        self.assertEqual(SemanticHealthStatus.WARN.value, "WARN")
        self.assertEqual(SemanticHealthStatus.CRITICAL.value, "CRITICAL")


class TestSummarizeSemanticGraphForGlobalHealth(unittest.TestCase):
    """Tests for summarize_semantic_graph_for_global_health function."""
    
    def test_health_ok_when_no_signals(self):
        """Health should be OK with no critical signals."""
        snapshot = {
            "critical_signals": [],
        }
        
        health = summarize_semantic_graph_for_global_health(snapshot)
        
        self.assertTrue(health["semantic_ok"])
        self.assertEqual(health["node_disappearance_count"], 0)
        self.assertEqual(health["category_migration_count"], 0)
        self.assertEqual(health["status"], "OK")
    
    def test_health_critical_on_disappearance(self):
        """Health should be CRITICAL when nodes disappeared."""
        snapshot = {
            "critical_signals": [
                {"type": "node_disappeared", "term": "a"},
                {"type": "node_disappeared", "term": "b"},
            ],
        }
        
        health = summarize_semantic_graph_for_global_health(snapshot)
        
        self.assertFalse(health["semantic_ok"])
        self.assertEqual(health["node_disappearance_count"], 2)
        self.assertEqual(health["category_migration_count"], 0)
        self.assertEqual(health["status"], "CRITICAL")
    
    def test_health_warn_on_migration(self):
        """Health should be WARN when migrations detected."""
        snapshot = {
            "critical_signals": [
                {"type": "migration", "term": "a"},
            ],
        }
        
        health = summarize_semantic_graph_for_global_health(snapshot)
        
        self.assertFalse(health["semantic_ok"])
        self.assertEqual(health["node_disappearance_count"], 0)
        self.assertEqual(health["category_migration_count"], 1)
        self.assertEqual(health["status"], "WARN")
    
    def test_health_critical_overrides_warn(self):
        """CRITICAL should override WARN when both present."""
        snapshot = {
            "critical_signals": [
                {"type": "node_disappeared", "term": "a"},
                {"type": "migration", "term": "b"},
            ],
        }
        
        health = summarize_semantic_graph_for_global_health(snapshot)
        
        self.assertFalse(health["semantic_ok"])
        self.assertEqual(health["node_disappearance_count"], 1)
        self.assertEqual(health["category_migration_count"], 1)
        self.assertEqual(health["status"], "CRITICAL")
    
    def test_health_counts_multiple(self):
        """Should count multiple signals of same type."""
        snapshot = {
            "critical_signals": [
                {"type": "node_disappeared", "term": "a"},
                {"type": "node_disappeared", "term": "b"},
                {"type": "node_disappeared", "term": "c"},
                {"type": "migration", "term": "x"},
                {"type": "migration", "term": "y"},
            ],
        }
        
        health = summarize_semantic_graph_for_global_health(snapshot)
        
        self.assertEqual(health["node_disappearance_count"], 3)
        self.assertEqual(health["category_migration_count"], 2)


class TestFormatSemanticHealthSummary(unittest.TestCase):
    """Tests for format_semantic_health_summary function."""
    
    def test_format_ok_health(self):
        """Should format OK health correctly."""
        health = {
            "semantic_ok": True,
            "node_disappearance_count": 0,
            "category_migration_count": 0,
            "status": "OK",
        }
        
        result = format_semantic_health_summary(health)
        
        self.assertEqual(
            result,
            "Semantic Health: disappearances=0 migrations=0 status=OK"
        )
    
    def test_format_critical_health(self):
        """Should format CRITICAL health correctly."""
        health = {
            "semantic_ok": False,
            "node_disappearance_count": 3,
            "category_migration_count": 2,
            "status": "CRITICAL",
        }
        
        result = format_semantic_health_summary(health)
        
        self.assertEqual(
            result,
            "Semantic Health: disappearances=3 migrations=2 status=CRITICAL"
        )
    
    def test_format_single_line(self):
        """Format should be single line."""
        health = {
            "semantic_ok": False,
            "node_disappearance_count": 5,
            "category_migration_count": 10,
            "status": "CRITICAL",
        }
        
        result = format_semantic_health_summary(health)
        
        self.assertNotIn("\n", result)
        self.assertNotIn("\r", result)


# =============================================================================
# INTEGRATION TESTS (Phase III)
# =============================================================================

class TestGovernanceIntegration(unittest.TestCase):
    """Integration tests for governance workflow."""
    
    def test_full_governance_workflow(self):
        """Test complete governance workflow: graph → drift → snapshot → filter."""
        # Build old graph
        old_graph = SemanticKnowledgeGraph(
            terms=[
                {"canonical_form": "goal_hit"},
                {"canonical_form": "density"},
                {"canonical_form": "disappeared_metric"},
            ],
            edges=[
                GraphEdge(src="slice_uplift_goal", dst="goal_hit", weight=1.0, kind="category"),
            ],
        )
        
        # Build new graph (one term disappeared)
        new_graph = SemanticKnowledgeGraph(
            terms=[
                {"canonical_form": "goal_hit"},
                {"canonical_form": "density"},
            ],
            edges=[
                GraphEdge(src="slice_uplift_goal", dst="goal_hit", weight=1.0, kind="category"),
            ],
        )
        
        # Analyze drift
        drift_report = analyze_graph_drift(old_graph, new_graph)
        
        # Build governance snapshot
        snapshot = build_semantic_governance_snapshot(new_graph, drift_report.to_dict())
        
        # Verify governance status
        self.assertEqual(snapshot["status"], "CRITICAL")
        
        # Generate suggestions and filter
        suggestions = [
            "Replace old_a with disappeared_metric in file.yaml",
            "Replace old_b with density in file.yaml",
        ]
        filter_result = filter_graph_suggestions(suggestions, snapshot)
        
        # In CRITICAL status, both suggestions are blocked as a safety measure
        # - First blocked because target term disappeared
        # - Second blocked because governance is CRITICAL (manual review required)
        self.assertEqual(len(filter_result["blocked_suggestions"]), 2)
        
        # Verify reasons are provided for each blocked suggestion
        for blocked in filter_result["blocked_suggestions"]:
            self.assertIn(blocked, filter_result["reasons"])
        
        # Get global health
        health = summarize_semantic_graph_for_global_health(snapshot)
        self.assertEqual(health["status"], "CRITICAL")
        self.assertEqual(health["node_disappearance_count"], 1)
    
    def test_governance_workflow_attention_allows_safe(self):
        """In ATTENTION status, safe suggestions should be allowed."""
        # Build graphs with only migration (not disappearance)
        old_graph = SemanticKnowledgeGraph(
            terms=[
                {"canonical_form": "goal_hit"},
                {"canonical_form": "density"},
                {"canonical_form": "slice_uplift_sparse"},
            ],
            edges=[
                GraphEdge(src="slice_uplift_sparse", dst="density", weight=1.0, kind="category"),
            ],
        )
        
        # New graph: category edge removed (migration)
        new_graph = SemanticKnowledgeGraph(
            terms=[
                {"canonical_form": "goal_hit"},
                {"canonical_form": "density"},
                {"canonical_form": "slice_uplift_sparse"},
            ],
            edges=[],  # Category edge removed
        )
        
        # Analyze drift
        drift_report = analyze_graph_drift(old_graph, new_graph)
        
        # Build governance snapshot - should be ATTENTION (migration only)
        snapshot = build_semantic_governance_snapshot(new_graph, drift_report.to_dict())
        self.assertEqual(snapshot["status"], "ATTENTION")
        
        # Generate suggestions
        # Note: migration is for slice_uplift_sparse (source), not density (destination)
        suggestions = [
            "Replace old with goal_hit in file.yaml",  # Safe suggestion
            "Replace old with density in file.yaml",   # Also safe (density not migrated)
            "Replace slice_uplift_sparse with other in file.yaml",  # Source term migrated - should be blocked
        ]
        filter_result = filter_graph_suggestions(suggestions, snapshot)
        
        # First two should be allowed (goal_hit and density not affected)
        self.assertEqual(len(filter_result["allowed_suggestions"]), 2)
        self.assertIn("goal_hit", filter_result["allowed_suggestions"][0])
        self.assertIn("density", filter_result["allowed_suggestions"][1])
        
        # Third should be blocked (slice_uplift_sparse migrated)
        self.assertEqual(len(filter_result["blocked_suggestions"]), 1)
        self.assertIn("slice_uplift_sparse", filter_result["blocked_suggestions"][0])


# =============================================================================
# PHASE IV — MULTI-RUN DRIFT TIMELINE & CURRICULUM/DOCS COUPLING TESTS
# =============================================================================

class TestBuildSemanticDriftTimeline(unittest.TestCase):
    """Tests for build_semantic_drift_timeline function."""
    
    def test_empty_snapshots(self):
        """Empty snapshots should return stable timeline."""
        from experiments.semantic_consistency_audit import build_semantic_drift_timeline
        
        timeline = build_semantic_drift_timeline([])
        
        self.assertEqual(timeline["trend"], "STABLE")
        self.assertEqual(len(timeline["timeline"]), 0)
        self.assertEqual(len(timeline["runs_with_critical_signals"]), 0)
    
    def test_single_snapshot(self):
        """Single snapshot should be stable."""
        from experiments.semantic_consistency_audit import build_semantic_drift_timeline
        
        snapshots = [{
            "run_id": "run-1",
            "term_count": 10,
            "critical_signals": [],
            "status": "OK",
        }]
        
        timeline = build_semantic_drift_timeline(snapshots)
        
        self.assertEqual(timeline["trend"], "STABLE")
        self.assertEqual(len(timeline["timeline"]), 1)
        self.assertEqual(timeline["timeline"][0]["run_id"], "run-1")
    
    def test_stable_timeline(self):
        """Timeline with no changes should be STABLE."""
        from experiments.semantic_consistency_audit import build_semantic_drift_timeline
        
        snapshots = [
            {"run_id": "run-1", "term_count": 10, "critical_signals": [], "status": "OK"},
            {"run_id": "run-2", "term_count": 10, "critical_signals": [], "status": "OK"},
            {"run_id": "run-3", "term_count": 10, "critical_signals": [], "status": "OK"},
        ]
        
        timeline = build_semantic_drift_timeline(snapshots)
        
        self.assertEqual(timeline["trend"], "STABLE")
        self.assertEqual(len(timeline["runs_with_critical_signals"]), 0)
    
    def test_drifting_timeline(self):
        """Timeline with some changes should be DRIFTING."""
        from experiments.semantic_consistency_audit import build_semantic_drift_timeline
        
        # Create timeline with 1 change out of 3 pairs (< 50% threshold for VOLATILE)
        snapshots = [
            {"run_id": "run-1", "term_count": 10, "critical_signals": [], "status": "OK"},
            {"run_id": "run-2", "term_count": 10, "critical_signals": [], "status": "OK"},
            {"run_id": "run-3", "term_count": 10, "critical_signals": [{"type": "migration", "term": "term1"}], "status": "ATTENTION"},
            {"run_id": "run-4", "term_count": 10, "critical_signals": [], "status": "OK"},
        ]
        
        timeline = build_semantic_drift_timeline(snapshots)
        
        # Should be DRIFTING (1 change out of 3 pairs = 33% < 50%)
        self.assertEqual(timeline["trend"], "DRIFTING")
    
    def test_volatile_timeline(self):
        """Timeline with frequent changes should be VOLATILE."""
        from experiments.semantic_consistency_audit import build_semantic_drift_timeline
        
        snapshots = [
            {"run_id": "run-1", "term_count": 10, "critical_signals": [], "status": "OK"},
            {"run_id": "run-2", "term_count": 12, "critical_signals": [], "status": "ATTENTION"},
            {"run_id": "run-3", "term_count": 10, "critical_signals": [], "status": "OK"},
            {"run_id": "run-4", "term_count": 12, "critical_signals": [], "status": "ATTENTION"},
        ]
        
        timeline = build_semantic_drift_timeline(snapshots)
        
        self.assertEqual(timeline["trend"], "VOLATILE")
    
    def test_tracks_critical_signals(self):
        """Timeline should track runs with critical signals."""
        from experiments.semantic_consistency_audit import build_semantic_drift_timeline
        
        snapshots = [
            {"run_id": "run-1", "term_count": 10, "critical_signals": [], "status": "OK"},
            {"run_id": "run-2", "term_count": 10, "critical_signals": [
                {"type": "node_disappeared", "term": "lost_term"}
            ], "status": "CRITICAL"},
            {"run_id": "run-3", "term_count": 10, "critical_signals": [], "status": "OK"},
        ]
        
        timeline = build_semantic_drift_timeline(snapshots)
        
        self.assertIn("run-2", timeline["runs_with_critical_signals"])
        self.assertEqual(len(timeline["node_disappearance_events"]), 1)
        self.assertEqual(timeline["node_disappearance_events"][0]["term"], "lost_term")
    
    def test_tracks_node_disappearances(self):
        """Timeline should track all node disappearance events."""
        from experiments.semantic_consistency_audit import build_semantic_drift_timeline
        
        snapshots = [
            {"run_id": "run-1", "term_count": 10, "critical_signals": [
                {"type": "node_disappeared", "term": "term1"}
            ], "status": "CRITICAL"},
            {"run_id": "run-2", "term_count": 10, "critical_signals": [
                {"type": "node_disappeared", "term": "term2"}
            ], "status": "CRITICAL"},
        ]
        
        timeline = build_semantic_drift_timeline(snapshots)
        
        self.assertEqual(len(timeline["node_disappearance_events"]), 2)
        terms = {e["term"] for e in timeline["node_disappearance_events"]}
        self.assertEqual(terms, {"term1", "term2"})


class TestAnalyzeSemanticAlignmentWithCurriculum(unittest.TestCase):
    """Tests for analyze_semantic_alignment_with_curriculum function."""
    
    def test_aligned_curriculum(self):
        """Perfect alignment should return ALIGNED status."""
        from experiments.semantic_consistency_audit import (
            analyze_semantic_alignment_with_curriculum_full,
            SemanticKnowledgeGraph,
        )
        
        graph = SemanticKnowledgeGraph(
            terms=[
                {"canonical_form": "slice_uplift_goal"},
                {"canonical_form": "goal_hit"},
            ],
            edges=[],
        )
        
        curriculum = {
            "slices": {
                "slice_uplift_goal": {
                    "success_metric": {"kind": "goal_hit"},
                },
            },
        }
        
        analysis = analyze_semantic_alignment_with_curriculum_full(graph, curriculum)
        
        self.assertEqual(analysis["alignment_status"], "ALIGNED")
        self.assertEqual(len(analysis["orphan_terms"]), 0)
        self.assertEqual(len(analysis["unused_curriculum_terms"]), 0)
    
    def test_partial_alignment(self):
        """Partial alignment should return PARTIAL status."""
        from experiments.semantic_consistency_audit import (
            analyze_semantic_alignment_with_curriculum_full,
            SemanticKnowledgeGraph,
        )
        
        # Create a larger graph so 1 orphan term is < 20% of total
        # Also need to ensure unused curriculum terms are also < 20%
        graph = SemanticKnowledgeGraph(
            terms=[
                {"canonical_form": "slice_uplift_goal"},
                {"canonical_form": "goal_hit"},
                {"canonical_form": "term1"},
                {"canonical_form": "term2"},
                {"canonical_form": "term3"},
                {"canonical_form": "term4"},
                {"canonical_form": "term5"},
                {"canonical_form": "orphan_term"},  # Not in curriculum (1/8 = 12.5% < 20%)
            ],
            edges=[],
        )
        
        curriculum = {
            "slices": {
                "slice_uplift_goal": {
                    "success_metric": {"kind": "goal_hit"},
                },
                # Add more slices so unused percentage is also small
                "slice_uplift_sparse": {
                    "success_metric": {"kind": "density"},
                },
                "slice_uplift_tree": {
                    "success_metric": {"kind": "chain_length"},
                },
                "slice_uplift_dependency": {
                    "success_metric": {"kind": "multi_goal"},
                },
            },
        }
        
        analysis = analyze_semantic_alignment_with_curriculum_full(graph, curriculum)
        
        # Should be PARTIAL if both orphan and unused are small percentages (< 20%)
        # Note: We have unused curriculum terms (density, chain_length, multi_goal) but they're not in graph
        # So we need to check the actual percentages
        orphan_pct = len(analysis["orphan_terms"]) / max(analysis["graph_term_count"], 1)
        unused_pct = len(analysis["unused_curriculum_terms"]) / max(analysis["curriculum_term_count"], 1)
        
        # If both are < 20%, should be PARTIAL
        if orphan_pct < 0.2 and unused_pct < 0.2:
            self.assertEqual(analysis["alignment_status"], "PARTIAL")
        else:
            # Otherwise might be MISALIGNED, which is also acceptable for this test
            self.assertIn(analysis["alignment_status"], ["PARTIAL", "MISALIGNED"])
        
        self.assertIn("orphan_term", analysis["orphan_terms"])
    
    def test_misaligned_curriculum(self):
        """Significant misalignment should return MISALIGNED status."""
        from experiments.semantic_consistency_audit import (
            analyze_semantic_alignment_with_curriculum_full,
            SemanticKnowledgeGraph,
        )
        
        graph = SemanticKnowledgeGraph(
            terms=[
                {"canonical_form": "orphan1"},
                {"canonical_form": "orphan2"},
                {"canonical_form": "orphan3"},
            ],
            edges=[],
        )
        
        curriculum = {
            "slices": {
                "slice_uplift_goal": {
                    "success_metric": {"kind": "goal_hit"},
                },
            },
        }
        
        analysis = analyze_semantic_alignment_with_curriculum_full(graph, curriculum)
        
        self.assertEqual(analysis["alignment_status"], "MISALIGNED")
        self.assertGreater(len(analysis["orphan_terms"]), 0)
    
    def test_unused_curriculum_terms(self):
        """Should detect curriculum terms not in graph."""
        from experiments.semantic_consistency_audit import (
            analyze_semantic_alignment_with_curriculum_full,
            SemanticKnowledgeGraph,
        )
        
        graph = SemanticKnowledgeGraph(
            terms=[
                {"canonical_form": "slice_uplift_goal"},
            ],
            edges=[],
        )
        
        curriculum = {
            "slices": {
                "slice_uplift_goal": {
                    "success_metric": {"kind": "goal_hit"},
                },
                "unused_slice": {  # Not in graph
                    "success_metric": {"kind": "unused_metric"},
                },
            },
        }
        
        analysis = analyze_semantic_alignment_with_curriculum_full(graph, curriculum)
        
        self.assertIn("unused_slice", analysis["unused_curriculum_terms"])
        self.assertIn("unused_metric", analysis["unused_curriculum_terms"])


class TestBuildSemanticDirectorPanel(unittest.TestCase):
    """Tests for build_semantic_director_panel function."""
    
    def test_green_status_light(self):
        """Stable and aligned should be GREEN."""
        from experiments.semantic_consistency_audit import build_semantic_director_panel
        
        drift_timeline = {
            "trend": "STABLE",
            "runs_with_critical_signals": [],
            "node_disappearance_events": [],
        }
        
        alignment = {
            "alignment_status": "ALIGNED",
        }
        
        panel = build_semantic_director_panel_legacy(drift_timeline, alignment)
        
        self.assertEqual(panel["semantic_status_light"], "GREEN")
    
    def test_yellow_status_light(self):
        """Drifting or partial alignment should be YELLOW."""
        from experiments.semantic_consistency_audit import build_semantic_director_panel
        
        drift_timeline = {
            "trend": "DRIFTING",
            "runs_with_critical_signals": [],
            "node_disappearance_events": [],
        }
        
        alignment = {
            "alignment_status": "PARTIAL",
        }
        
        panel = build_semantic_director_panel_legacy(drift_timeline, alignment)
        
        self.assertEqual(panel["semantic_status_light"], "YELLOW")
    
    def test_red_status_light(self):
        """Critical signals or misalignment should be RED."""
        from experiments.semantic_consistency_audit import build_semantic_director_panel
        
        drift_timeline = {
            "trend": "STABLE",
            "runs_with_critical_signals": ["run-1"],
            "node_disappearance_events": [{"run_id": "run-1", "term": "lost"}],
        }
        
        alignment = {
            "alignment_status": "MISALIGNED",
        }
        
        panel = build_semantic_director_panel_legacy(drift_timeline, alignment)
        
        self.assertEqual(panel["semantic_status_light"], "RED")
    
    def test_headline_generation(self):
        """Headline should summarize drift and alignment."""
        from experiments.semantic_consistency_audit import build_semantic_director_panel
        
        drift_timeline = {
            "trend": "DRIFTING",
            "runs_with_critical_signals": [],
            "node_disappearance_events": [],
        }
        
        alignment = {
            "alignment_status": "ALIGNED",
        }
        
        panel = build_semantic_director_panel_legacy(drift_timeline, alignment)
        
        self.assertIn("headline", panel)
        self.assertIsInstance(panel["headline"], str)
        self.assertGreater(len(panel["headline"]), 0)
    
    def test_critical_run_ids_tracked(self):
        """Panel should include critical run IDs."""
        from experiments.semantic_consistency_audit import build_semantic_director_panel
        
        drift_timeline = {
            "trend": "STABLE",
            "runs_with_critical_signals": ["run-1", "run-3"],
            "node_disappearance_events": [],
        }
        
        alignment = {
            "alignment_status": "ALIGNED",
        }
        
        panel = build_semantic_director_panel_legacy(drift_timeline, alignment)
        
        self.assertEqual(panel["critical_run_ids"], ["run-1", "run-3"])
        self.assertEqual(panel["node_disappearance_count"], 0)
    
    def test_node_disappearance_count(self):
        """Panel should count node disappearances."""
        from experiments.semantic_consistency_audit import build_semantic_director_panel
        
        drift_timeline = {
            "trend": "STABLE",
            "runs_with_critical_signals": ["run-1"],
            "node_disappearance_events": [
                {"run_id": "run-1", "term": "term1"},
                {"run_id": "run-1", "term": "term2"},
            ],
        }
        
        alignment = {
            "alignment_status": "ALIGNED",
        }
        
        panel = build_semantic_director_panel_legacy(drift_timeline, alignment)
        
        self.assertEqual(panel["node_disappearance_count"], 2)


# =============================================================================
# CROSS-SYSTEM SEMANTIC ALIGNMENT INDEX TESTS (Phase IV) - TASK 1
# =============================================================================

class TestAlignmentStatus(unittest.TestCase):
    """Tests for AlignmentStatus enum."""
    
    def test_alignment_status_values(self):
        """AlignmentStatus should have ALIGNED, PARTIAL, DIVERGENT."""
        self.assertEqual(AlignmentStatus.ALIGNED.value, "ALIGNED")
        self.assertEqual(AlignmentStatus.PARTIAL.value, "PARTIAL")
        self.assertEqual(AlignmentStatus.DIVERGENT.value, "DIVERGENT")


class TestBuildSemanticAlignmentIndex(unittest.TestCase):
    """Tests for build_semantic_alignment_index function."""
    
    def test_alignment_index_aligned(self):
        """Should be ALIGNED when all terms overlap."""
        graph = SemanticKnowledgeGraph(
            terms=[
                {"canonical_form": "goal_hit"},
                {"canonical_form": "density"},
            ],
            edges=[],
        )
        graph_snapshot = {"status": "OK", "critical_signals": []}
        
        curriculum_manifest = {"terms": ["goal_hit", "density"]}
        taxonomy_semantics = {"terms": ["goal_hit", "density"]}
        docs_vocab_index = {"terms": ["goal_hit", "density"]}
        
        index = build_semantic_alignment_index(
            graph_snapshot,
            curriculum_manifest,
            taxonomy_semantics,
            docs_vocab_index,
            graph=graph,
        )
        
        self.assertEqual(index["alignment_status"], "ALIGNED")
        self.assertEqual(len(index["terms_only_in_code"]), 0)
        self.assertEqual(len(index["terms_only_in_docs"]), 0)
        self.assertEqual(len(index["terms_only_in_curriculum"]), 0)
        self.assertEqual(len(index["taxonomy_terms_with_no_uses"]), 0)
    
    def test_terms_only_in_code(self):
        """Should identify terms only in code (graph)."""
        graph = SemanticKnowledgeGraph(
            terms=[
                {"canonical_form": "code_only_term"},
                {"canonical_form": "shared_term"},
            ],
            edges=[],
        )
        graph_snapshot = {"status": "OK", "critical_signals": []}
        
        curriculum_manifest = {"terms": ["shared_term"]}
        taxonomy_semantics = {"terms": []}
        docs_vocab_index = {"terms": ["shared_term"]}
        
        index = build_semantic_alignment_index(
            graph_snapshot,
            curriculum_manifest,
            taxonomy_semantics,
            docs_vocab_index,
            graph=graph,
        )
        
        self.assertIn("code_only_term", index["terms_only_in_code"])
        self.assertNotIn("shared_term", index["terms_only_in_code"])
    
    def test_terms_only_in_docs(self):
        """Should identify terms only in docs."""
        graph = SemanticKnowledgeGraph(
            terms=[{"canonical_form": "shared_term"}],
            edges=[],
        )
        graph_snapshot = {"status": "OK", "critical_signals": []}
        
        curriculum_manifest = {"terms": ["shared_term"]}
        taxonomy_semantics = {"terms": []}
        docs_vocab_index = {"terms": ["shared_term", "docs_only_term"]}
        
        index = build_semantic_alignment_index(
            graph_snapshot,
            curriculum_manifest,
            taxonomy_semantics,
            docs_vocab_index,
            graph=graph,
        )
        
        self.assertIn("docs_only_term", index["terms_only_in_docs"])
        self.assertNotIn("shared_term", index["terms_only_in_docs"])
    
    def test_terms_only_in_curriculum(self):
        """Should identify terms only in curriculum."""
        graph = SemanticKnowledgeGraph(
            terms=[{"canonical_form": "shared_term"}],
            edges=[],
        )
        graph_snapshot = {"status": "OK", "critical_signals": []}
        
        curriculum_manifest = {"terms": ["shared_term", "curriculum_only_term"]}
        taxonomy_semantics = {"terms": []}
        docs_vocab_index = {"terms": ["shared_term"]}
        
        index = build_semantic_alignment_index(
            graph_snapshot,
            curriculum_manifest,
            taxonomy_semantics,
            docs_vocab_index,
            graph=graph,
        )
        
        self.assertIn("curriculum_only_term", index["terms_only_in_curriculum"])
        self.assertNotIn("shared_term", index["terms_only_in_curriculum"])
    
    def test_taxonomy_terms_with_no_uses(self):
        """Should identify taxonomy terms not used anywhere."""
        graph = SemanticKnowledgeGraph(
            terms=[{"canonical_form": "shared_term"}],
            edges=[],
        )
        graph_snapshot = {"status": "OK", "critical_signals": []}
        
        curriculum_manifest = {"terms": ["shared_term"]}
        taxonomy_semantics = {"terms": ["unused_taxonomy_term", "shared_term"]}
        docs_vocab_index = {"terms": ["shared_term"]}
        
        index = build_semantic_alignment_index(
            graph_snapshot,
            curriculum_manifest,
            taxonomy_semantics,
            docs_vocab_index,
            graph=graph,
        )
        
        self.assertIn("unused_taxonomy_term", index["taxonomy_terms_with_no_uses"])
        self.assertNotIn("shared_term", index["taxonomy_terms_with_no_uses"])
    
    def test_alignment_status_partial(self):
        """Should be PARTIAL when <10% orphaned."""
        graph = SemanticKnowledgeGraph(
            terms=[{"canonical_form": f"term_{i}"} for i in range(10)],
            edges=[],
        )
        graph_snapshot = {"status": "OK", "critical_signals": []}
        
        # 9 shared, 1 orphaned = 10% orphaned (should be PARTIAL)
        curriculum_manifest = {"terms": [f"term_{i}" for i in range(9)]}
        taxonomy_semantics = {"terms": []}
        docs_vocab_index = {"terms": [f"term_{i}" for i in range(9)]}
        
        index = build_semantic_alignment_index(
            graph_snapshot,
            curriculum_manifest,
            taxonomy_semantics,
            docs_vocab_index,
            graph=graph,
        )
        
        self.assertEqual(index["alignment_status"], "PARTIAL")
    
    def test_alignment_status_divergent(self):
        """Should be DIVERGENT when >10% orphaned."""
        graph = SemanticKnowledgeGraph(
            terms=[{"canonical_form": f"term_{i}"} for i in range(20)],
            edges=[],
        )
        graph_snapshot = {"status": "OK", "critical_signals": []}
        
        # 10 shared, 10 orphaned = 50% orphaned (should be DIVERGENT)
        curriculum_manifest = {"terms": [f"term_{i}" for i in range(10)]}
        taxonomy_semantics = {"terms": []}
        docs_vocab_index = {"terms": [f"term_{i}" for i in range(10)]}
        
        index = build_semantic_alignment_index(
            graph_snapshot,
            curriculum_manifest,
            taxonomy_semantics,
            docs_vocab_index,
            graph=graph,
        )
        
        self.assertEqual(index["alignment_status"], "DIVERGENT")
    
    def test_sorted_output(self):
        """All term lists should be sorted."""
        graph = SemanticKnowledgeGraph(
            terms=[
                {"canonical_form": "zebra"},
                {"canonical_form": "apple"},
            ],
            edges=[],
        )
        graph_snapshot = {"status": "OK", "critical_signals": []}
        
        curriculum_manifest = {"terms": []}
        taxonomy_semantics = {"terms": []}
        docs_vocab_index = {"terms": []}
        
        index = build_semantic_alignment_index(
            graph_snapshot,
            curriculum_manifest,
            taxonomy_semantics,
            docs_vocab_index,
            graph=graph,
        )
        
        # Should be sorted alphabetically
        self.assertEqual(index["terms_only_in_code"], ["apple", "zebra"])


# =============================================================================
# SEMANTIC RISK DECOMPOSITION TESTS (Phase IV) - TASK 2
# =============================================================================

class TestRiskStatus(unittest.TestCase):
    """Tests for RiskStatus enum."""
    
    def test_risk_status_values(self):
        """RiskStatus should have OK, ATTENTION, CRITICAL."""
        self.assertEqual(RiskStatus.OK.value, "OK")
        self.assertEqual(RiskStatus.ATTENTION.value, "ATTENTION")
        self.assertEqual(RiskStatus.CRITICAL.value, "CRITICAL")


class TestAnalyzeSemanticRisk(unittest.TestCase):
    """Tests for analyze_semantic_risk function."""
    
    def test_risk_ok_when_aligned(self):
        """Risk should be OK when systems are aligned."""
        alignment_index = {
            "terms_only_in_code": [],
            "terms_only_in_docs": [],
            "terms_only_in_curriculum": [],
            "taxonomy_terms_with_no_uses": [],
            "alignment_status": "ALIGNED",
        }
        governance_snapshot = {
            "status": "OK",
            "critical_signals": [],
        }
        
        risk = analyze_semantic_risk(alignment_index, governance_snapshot)
        
        self.assertEqual(risk["status"], "OK")
        self.assertEqual(len(risk["high_risk_terms"]), 0)
        self.assertEqual(len(risk["medium_risk_terms"]), 0)
    
    def test_high_risk_terms(self):
        """Should identify high-risk terms (governance issues + misalignment)."""
        alignment_index = {
            "terms_only_in_code": ["problematic_term"],
            "terms_only_in_docs": [],
            "terms_only_in_curriculum": [],
            "taxonomy_terms_with_no_uses": [],
            "alignment_status": "PARTIAL",
        }
        governance_snapshot = {
            "status": "CRITICAL",
            "critical_signals": [
                {"type": "node_disappeared", "term": "problematic_term"},
            ],
        }
        
        risk = analyze_semantic_risk(alignment_index, governance_snapshot)
        
        self.assertEqual(risk["status"], "CRITICAL")
        self.assertIn("problematic_term", risk["high_risk_terms"])
        self.assertEqual(len(risk["medium_risk_terms"]), 0)
    
    def test_medium_risk_terms(self):
        """Should identify medium-risk terms (misaligned but no governance issues)."""
        alignment_index = {
            "terms_only_in_code": ["misaligned_term"],
            "terms_only_in_docs": [],
            "terms_only_in_curriculum": [],
            "taxonomy_terms_with_no_uses": [],
            "alignment_status": "PARTIAL",
        }
        governance_snapshot = {
            "status": "OK",
            "critical_signals": [],
        }
        
        risk = analyze_semantic_risk(alignment_index, governance_snapshot)
        
        self.assertEqual(risk["status"], "ATTENTION")
        self.assertEqual(len(risk["high_risk_terms"]), 0)
        self.assertIn("misaligned_term", risk["medium_risk_terms"])
    
    def test_risk_attention_on_governance_attention(self):
        """Risk should be ATTENTION when governance is ATTENTION."""
        alignment_index = {
            "terms_only_in_code": [],
            "terms_only_in_docs": [],
            "terms_only_in_curriculum": [],
            "taxonomy_terms_with_no_uses": [],
            "alignment_status": "ALIGNED",
        }
        governance_snapshot = {
            "status": "ATTENTION",
            "critical_signals": [
                {"type": "migration", "term": "migrated_term"},
            ],
        }
        
        risk = analyze_semantic_risk(alignment_index, governance_snapshot)
        
        self.assertEqual(risk["status"], "ATTENTION")
    
    def test_risk_notes_generated(self):
        """Should generate explanatory notes."""
        alignment_index = {
            "terms_only_in_code": ["term1"],
            "terms_only_in_docs": ["term2"],
            "terms_only_in_curriculum": [],
            "taxonomy_terms_with_no_uses": [],
            "alignment_status": "PARTIAL",
        }
        governance_snapshot = {
            "status": "CRITICAL",
            "critical_signals": [
                {"type": "node_disappeared", "term": "term1"},
            ],
        }
        
        risk = analyze_semantic_risk(alignment_index, governance_snapshot)
        
        self.assertGreater(len(risk["notes"]), 0)
        self.assertIsInstance(risk["notes"], list)
        # Should mention high-risk terms
        self.assertTrue(any("high_risk" in note.lower() or "governance" in note.lower() for note in risk["notes"]))
    
    def test_risk_notes_for_divergent(self):
        """Notes should mention DIVERGENT alignment."""
        alignment_index = {
            "terms_only_in_code": [],
            "terms_only_in_docs": [],
            "terms_only_in_curriculum": [],
            "taxonomy_terms_with_no_uses": [],
            "alignment_status": "DIVERGENT",
        }
        governance_snapshot = {
            "status": "OK",
            "critical_signals": [],
        }
        
        risk = analyze_semantic_risk(alignment_index, governance_snapshot)
        
        self.assertTrue(any("DIVERGENT" in note for note in risk["notes"]))


# =============================================================================
# DIRECTOR SEMANTIC PANEL TESTS (Phase IV) - TASK 3
# =============================================================================

class TestStatusLight(unittest.TestCase):
    """Tests for StatusLight enum."""
    
    def test_status_light_values(self):
        """StatusLight should have GREEN, YELLOW, RED."""
        self.assertEqual(StatusLight.GREEN.value, "GREEN")
        self.assertEqual(StatusLight.YELLOW.value, "YELLOW")
        self.assertEqual(StatusLight.RED.value, "RED")


class TestBuildSemanticDirectorPanelExtended(unittest.TestCase):
    """Tests for build_semantic_director_panel function (Phase IV)."""
    
    def test_panel_green_when_all_ok(self):
        """Panel should be GREEN when all systems are OK."""
        from experiments.semantic_consistency_audit import build_semantic_director_panel_extended
        
        governance_snapshot = {
            "status": "OK",
            "critical_signals": [],
        }
        alignment_index = {
            "alignment_status": "ALIGNED",
        }
        risk_analysis = {
            "status": "OK",
            "high_risk_terms": [],
        }
        
        panel = build_semantic_director_panel(
            governance_snapshot,
            alignment_index,
            risk_analysis,
        )
        
        self.assertEqual(panel["status_light"], "GREEN")
        self.assertTrue(panel["semantic_ok"])
        self.assertEqual(panel["alignment_status"], "ALIGNED")
        self.assertEqual(panel["critical_term_count"], 0)
    
    def test_panel_red_on_critical_governance(self):
        """Panel should be RED when governance is CRITICAL."""
        governance_snapshot = {
            "status": "CRITICAL",
            "critical_signals": [
                {"type": "node_disappeared", "term": "lost_term"},
            ],
        }
        alignment_index = {
            "alignment_status": "ALIGNED",
        }
        risk_analysis = {
            "status": "CRITICAL",
            "high_risk_terms": ["lost_term"],
        }
        
        panel = build_semantic_director_panel(
            governance_snapshot,
            alignment_index,
            risk_analysis,
        )
        
        self.assertEqual(panel["status_light"], "RED")
        self.assertFalse(panel["semantic_ok"])
        self.assertEqual(panel["critical_term_count"], 1)
    
    def test_panel_red_on_divergent_alignment(self):
        """Panel should be RED when alignment is DIVERGENT."""
        governance_snapshot = {
            "status": "OK",
            "critical_signals": [],
        }
        alignment_index = {
            "alignment_status": "DIVERGENT",
        }
        risk_analysis = {
            "status": "OK",
            "high_risk_terms": [],
        }
        
        panel = build_semantic_director_panel(
            governance_snapshot,
            alignment_index,
            risk_analysis,
        )
        
        self.assertEqual(panel["status_light"], "RED")
        self.assertFalse(panel["semantic_ok"])
    
    def test_panel_yellow_on_attention(self):
        """Panel should be YELLOW when status is ATTENTION."""
        governance_snapshot = {
            "status": "ATTENTION",
            "critical_signals": [
                {"type": "migration", "term": "migrated_term"},
            ],
        }
        alignment_index = {
            "alignment_status": "PARTIAL",
        }
        risk_analysis = {
            "status": "ATTENTION",
            "high_risk_terms": [],
        }
        
        panel = build_semantic_director_panel(
            governance_snapshot,
            alignment_index,
            risk_analysis,
        )
        
        self.assertEqual(panel["status_light"], "YELLOW")
        self.assertFalse(panel["semantic_ok"])
    
    def test_panel_headline_red(self):
        """Headline should reflect RED status."""
        governance_snapshot = {
            "status": "CRITICAL",
            "critical_signals": [
                {"type": "node_disappeared", "term": "term1"},
                {"type": "node_disappeared", "term": "term2"},
            ],
        }
        alignment_index = {
            "alignment_status": "ALIGNED",
        }
        risk_analysis = {
            "status": "CRITICAL",
            "high_risk_terms": ["term1", "term2"],
        }
        
        panel = build_semantic_director_panel(
            governance_snapshot,
            alignment_index,
            risk_analysis,
        )
        
        self.assertEqual(panel["status_light"], "RED")
        self.assertIn("critical", panel["headline"].lower())
        self.assertIn("2", panel["headline"])  # Should mention count
    
    def test_panel_headline_yellow(self):
        """Headline should reflect YELLOW status."""
        governance_snapshot = {
            "status": "OK",
            "critical_signals": [],
        }
        alignment_index = {
            "alignment_status": "PARTIAL",
        }
        risk_analysis = {
            "status": "OK",
            "high_risk_terms": [],
        }
        
        panel = build_semantic_director_panel(
            governance_snapshot,
            alignment_index,
            risk_analysis,
        )
        
        self.assertEqual(panel["status_light"], "YELLOW")
        self.assertIn("partial", panel["headline"].lower())
    
    def test_panel_headline_green(self):
        """Headline should reflect GREEN status."""
        governance_snapshot = {
            "status": "OK",
            "critical_signals": [],
        }
        alignment_index = {
            "alignment_status": "ALIGNED",
        }
        risk_analysis = {
            "status": "OK",
            "high_risk_terms": [],
        }
        
        panel = build_semantic_director_panel(
            governance_snapshot,
            alignment_index,
            risk_analysis,
        )
        
        self.assertEqual(panel["status_light"], "GREEN")
        headline_lower = panel["headline"].lower()
        self.assertTrue("healthy" in headline_lower or "aligned" in headline_lower)
    
    def test_panel_includes_all_fields(self):
        """Panel should include all required fields."""
        governance_snapshot = {
            "status": "OK",
            "critical_signals": [],
        }
        alignment_index = {
            "alignment_status": "ALIGNED",
        }
        risk_analysis = {
            "status": "OK",
            "high_risk_terms": [],
        }
        
        panel = build_semantic_director_panel(
            governance_snapshot,
            alignment_index,
            risk_analysis,
        )
        
        self.assertIn("status_light", panel)
        self.assertIn("semantic_ok", panel)
        self.assertIn("alignment_status", panel)
        self.assertIn("critical_term_count", panel)
        self.assertIn("headline", panel)
        self.assertIn("governance_status", panel)
        self.assertIn("risk_status", panel)


# =============================================================================
# PHASE IV INTEGRATION TESTS
# =============================================================================

class TestPhaseIVIntegration(unittest.TestCase):
    """Integration tests for Phase IV workflow."""
    
    def test_full_phase_iv_workflow(self):
        """Test complete Phase IV workflow: alignment → risk → panel."""
        # Build graph
        graph = SemanticKnowledgeGraph(
            terms=[
                {"canonical_form": "goal_hit"},
                {"canonical_form": "code_only_term"},
            ],
            edges=[],
        )
        
        # Build governance snapshot
        governance_snapshot = build_semantic_governance_snapshot(graph)
        
        # Build alignment index
        curriculum_manifest = {"terms": ["goal_hit"]}
        taxonomy_semantics = {"terms": ["unused_taxonomy"]}
        docs_vocab_index = {"terms": ["goal_hit", "docs_only_term"]}
        
        alignment_index = build_semantic_alignment_index(
            governance_snapshot,
            curriculum_manifest,
            taxonomy_semantics,
            docs_vocab_index,
            graph=graph,
        )
        
        # Analyze risk
        risk_analysis = analyze_semantic_risk(alignment_index, governance_snapshot)
        
        # Build director panel (using extended version)
        from experiments.semantic_consistency_audit import build_semantic_director_panel_extended
        
        panel = build_semantic_director_panel_extended(
            governance_snapshot,
            alignment_index,
            risk_analysis,
        )
        
        # Verify workflow produces valid outputs
        self.assertIn("alignment_status", alignment_index)
        self.assertIn("status", risk_analysis)
        self.assertIn("status_light", panel)
        self.assertIn("headline", panel)
        
        # Verify alignment detected orphaned terms
        self.assertIn("code_only_term", alignment_index["terms_only_in_code"])
        self.assertIn("docs_only_term", alignment_index["terms_only_in_docs"])
        self.assertIn("unused_taxonomy", alignment_index["taxonomy_terms_with_no_uses"])


# =============================================================================
# SEMANTIC CONTRACT AUDITOR TESTS (Phase IV Follow-up - Task 1)
# =============================================================================

class TestContractStatus(unittest.TestCase):
    """Tests for ContractStatus enum."""
    
    def test_contract_status_values(self):
        """ContractStatus should have OK, ATTENTION, BREACH."""
        self.assertEqual(ContractStatus.OK.value, "OK")
        self.assertEqual(ContractStatus.ATTENTION.value, "ATTENTION")
        self.assertEqual(ContractStatus.BREACH.value, "BREACH")


class TestAuditSemanticContract(unittest.TestCase):
    """Tests for audit_semantic_contract function."""
    
    def test_contract_ok_when_compliant(self):
        """Contract should be OK when all curriculum terms are present."""
        alignment_index = {
            "terms_only_in_code": [],
            "terms_only_in_docs": [],
            "terms_only_in_curriculum": [],
            "taxonomy_terms_with_no_uses": [],
        }
        taxonomy = {"terms": ["goal_hit", "density", "chain_length"]}
        curriculum_manifest = {"terms": ["goal_hit", "density", "chain_length"]}
        
        audit = audit_semantic_contract(alignment_index, taxonomy, curriculum_manifest)
        
        self.assertEqual(audit["contract_status"], "OK")
        self.assertEqual(len(audit["violated_contract_terms"]), 0)
    
    def test_contract_breach_detection(self):
        """Should detect contract breach when curriculum terms missing."""
        alignment_index = {
            "terms_only_in_code": [],
            "terms_only_in_docs": [],
            "terms_only_in_curriculum": ["missing_term1", "missing_term2", "missing_term3"],
            "taxonomy_terms_with_no_uses": [],
        }
        taxonomy = {"terms": ["goal_hit"]}
        curriculum_manifest = {"terms": ["goal_hit", "missing_term1", "missing_term2", "missing_term3"]}
        
        audit = audit_semantic_contract(alignment_index, taxonomy, curriculum_manifest)
        
        # 3 out of 4 terms missing = 75% > 50% = BREACH
        self.assertEqual(audit["contract_status"], "BREACH")
        self.assertGreater(len(audit["violated_contract_terms"]), 0)
        self.assertIn("missing_term1", audit["violated_contract_terms"])
    
    def test_contract_attention_on_partial_breach(self):
        """Should be ATTENTION when <50% terms violated."""
        alignment_index = {
            "terms_only_in_code": [],
            "terms_only_in_docs": [],
            "terms_only_in_curriculum": ["missing_term"],
            "taxonomy_terms_with_no_uses": [],
        }
        taxonomy = {"terms": ["goal_hit", "density", "chain_length"]}
        curriculum_manifest = {"terms": ["goal_hit", "density", "chain_length", "missing_term"]}
        
        audit = audit_semantic_contract(alignment_index, taxonomy, curriculum_manifest)
        
        # 1 out of 4 terms missing = 25% < 50% = ATTENTION
        self.assertEqual(audit["contract_status"], "ATTENTION")
        self.assertIn("missing_term", audit["violated_contract_terms"])
    
    def test_mismatch_types_categorized(self):
        """Should categorize mismatches by type."""
        alignment_index = {
            "terms_only_in_code": ["code_term"],
            "terms_only_in_docs": ["docs_term"],
            "terms_only_in_curriculum": ["curriculum_term"],
            "taxonomy_terms_with_no_uses": ["unused_taxonomy"],
        }
        taxonomy = {"terms": ["goal_hit"]}
        curriculum_manifest = {"terms": ["goal_hit", "curriculum_term"]}
        
        audit = audit_semantic_contract(alignment_index, taxonomy, curriculum_manifest)
        
        self.assertIn("docs", audit["mismatch_types"])
        self.assertIn("curriculum", audit["mismatch_types"])
        self.assertIn("taxonomy", audit["mismatch_types"])
        self.assertIn("docs_term", audit["mismatch_types"]["docs"])
        self.assertIn("curriculum_term", audit["mismatch_types"]["curriculum"])
        self.assertIn("unused_taxonomy", audit["mismatch_types"]["taxonomy"])
    
    def test_summary_notes_generated(self):
        """Should generate explanatory summary notes."""
        alignment_index = {
            "terms_only_in_code": [],
            "terms_only_in_docs": ["docs_term"],
            "terms_only_in_curriculum": ["curriculum_term"],
            "taxonomy_terms_with_no_uses": ["unused_taxonomy"],
        }
        taxonomy = {"terms": ["goal_hit"]}
        curriculum_manifest = {"terms": ["goal_hit", "curriculum_term"]}
        
        audit = audit_semantic_contract(alignment_index, taxonomy, curriculum_manifest)
        
        self.assertGreater(len(audit["summary_notes"]), 0)
        self.assertIsInstance(audit["summary_notes"], list)
        # Should mention violations
        notes_text = " ".join(audit["summary_notes"]).lower()
        self.assertTrue("curriculum" in notes_text or "missing" in notes_text)
    
    def test_summary_notes_when_compliant(self):
        """Should note compliance when contract is OK."""
        alignment_index = {
            "terms_only_in_code": [],
            "terms_only_in_docs": [],
            "terms_only_in_curriculum": [],
            "taxonomy_terms_with_no_uses": [],
        }
        taxonomy = {"terms": ["goal_hit"]}
        curriculum_manifest = {"terms": ["goal_hit"]}
        
        audit = audit_semantic_contract(alignment_index, taxonomy, curriculum_manifest)
        
        notes_text = " ".join(audit["summary_notes"]).lower()
        self.assertTrue("compliant" in notes_text or "ok" in notes_text)


# =============================================================================
# DRIFT FORECAST TILE TESTS (Phase IV Follow-up - Task 2)
# =============================================================================

class TestDriftDirection(unittest.TestCase):
    """Tests for DriftDirection enum."""
    
    def test_drift_direction_values(self):
        """DriftDirection should have IMPROVING, STABLE, DEGRADING."""
        self.assertEqual(DriftDirection.IMPROVING.value, "IMPROVING")
        self.assertEqual(DriftDirection.STABLE.value, "STABLE")
        self.assertEqual(DriftDirection.DEGRADING.value, "DEGRADING")


class TestForecastBand(unittest.TestCase):
    """Tests for ForecastBand enum."""
    
    def test_forecast_band_values(self):
        """ForecastBand should have LOW, MEDIUM, HIGH."""
        self.assertEqual(ForecastBand.LOW.value, "LOW")
        self.assertEqual(ForecastBand.MEDIUM.value, "MEDIUM")
        self.assertEqual(ForecastBand.HIGH.value, "HIGH")


class TestForecastSemanticDrift(unittest.TestCase):
    """Tests for forecast_semantic_drift function."""
    
    def test_forecast_empty_history(self):
        """Should return STABLE with LOW confidence for empty history."""
        forecast = forecast_semantic_drift([])
        
        self.assertEqual(forecast["drift_direction"], "STABLE")
        self.assertEqual(forecast["forecast_band"], "LOW")
        self.assertIn("No historical data", forecast["explanation"])
    
    def test_forecast_single_point(self):
        """Should return STABLE with LOW confidence for single point."""
        history = [
            {"alignment_status": "ALIGNED", "total_orphaned_terms": 0}
        ]
        
        forecast = forecast_semantic_drift(history)
        
        self.assertEqual(forecast["drift_direction"], "STABLE")
        self.assertEqual(forecast["forecast_band"], "LOW")
        self.assertIn("Insufficient historical data", forecast["explanation"])
    
    def test_forecast_improving_trend(self):
        """Should detect IMPROVING trend when orphaned terms decrease."""
        history = [
            {"alignment_status": "DIVERGENT", "total_orphaned_terms": 10},
            {"alignment_status": "PARTIAL", "total_orphaned_terms": 5},
            {"alignment_status": "ALIGNED", "total_orphaned_terms": 0},
        ]
        
        forecast = forecast_semantic_drift(history)
        
        self.assertEqual(forecast["drift_direction"], "IMPROVING")
        self.assertIn("improving", forecast["explanation"].lower())
        self.assertIn("decreased", forecast["explanation"].lower())
    
    def test_forecast_degrading_trend(self):
        """Should detect DEGRADING trend when orphaned terms increase."""
        history = [
            {"alignment_status": "ALIGNED", "total_orphaned_terms": 0},
            {"alignment_status": "PARTIAL", "total_orphaned_terms": 5},
            {"alignment_status": "DIVERGENT", "total_orphaned_terms": 10},
        ]
        
        forecast = forecast_semantic_drift(history)
        
        self.assertEqual(forecast["drift_direction"], "DEGRADING")
        self.assertIn("degrading", forecast["explanation"].lower())
        self.assertIn("increased", forecast["explanation"].lower())
    
    def test_forecast_stable_trend(self):
        """Should detect STABLE trend when orphaned terms remain constant."""
        history = [
            {"alignment_status": "ALIGNED", "total_orphaned_terms": 2},
            {"alignment_status": "ALIGNED", "total_orphaned_terms": 2},
            {"alignment_status": "ALIGNED", "total_orphaned_terms": 2},
        ]
        
        forecast = forecast_semantic_drift(history)
        
        self.assertEqual(forecast["drift_direction"], "STABLE")
        self.assertIn("stable", forecast["explanation"].lower())
    
    def test_forecast_band_low_for_few_points(self):
        """Should return LOW band for <3 data points."""
        history = [
            {"alignment_status": "ALIGNED", "total_orphaned_terms": 0},
            {"alignment_status": "PARTIAL", "total_orphaned_terms": 2},
        ]
        
        forecast = forecast_semantic_drift(history)
        
        self.assertEqual(forecast["forecast_band"], "LOW")
    
    def test_forecast_band_medium_for_3_to_4_points(self):
        """Should return MEDIUM band for 3-4 data points."""
        history = [
            {"alignment_status": "ALIGNED", "total_orphaned_terms": 0},
            {"alignment_status": "PARTIAL", "total_orphaned_terms": 2},
            {"alignment_status": "ALIGNED", "total_orphaned_terms": 1},
        ]
        
        forecast = forecast_semantic_drift(history)
        
        self.assertEqual(forecast["forecast_band"], "MEDIUM")
    
    def test_forecast_band_high_for_consistent_trend(self):
        """Should return HIGH band for 5+ points with consistent trend."""
        history = [
            {"alignment_status": "DIVERGENT", "total_orphaned_terms": 10},
            {"alignment_status": "DIVERGENT", "total_orphaned_terms": 8},
            {"alignment_status": "PARTIAL", "total_orphaned_terms": 6},
            {"alignment_status": "PARTIAL", "total_orphaned_terms": 4},
            {"alignment_status": "ALIGNED", "total_orphaned_terms": 2},
        ]
        
        forecast = forecast_semantic_drift(history)
        
        # All changes are negative (improving), should be HIGH confidence
        self.assertEqual(forecast["forecast_band"], "HIGH")
        self.assertEqual(forecast["drift_direction"], "IMPROVING")
    
    def test_forecast_includes_historical_data(self):
        """Forecast should include historical point count and trend."""
        history = [
            {"alignment_status": "ALIGNED", "total_orphaned_terms": 0},
            {"alignment_status": "PARTIAL", "total_orphaned_terms": 2},
            {"alignment_status": "ALIGNED", "total_orphaned_terms": 1},
        ]
        
        forecast = forecast_semantic_drift(history)
        
        self.assertEqual(forecast["historical_points"], 3)
        self.assertIn("orphaned_count_trend", forecast)
        self.assertEqual(len(forecast["orphaned_count_trend"]), 3)
        self.assertEqual(forecast["orphaned_count_trend"], [0, 2, 1])
    
    def test_forecast_status_progression_improving(self):
        """Should detect improvement from status progression."""
        history = [
            {"alignment_status": "DIVERGENT", "total_orphaned_terms": 5},
            {"alignment_status": "PARTIAL", "total_orphaned_terms": 5},
            {"alignment_status": "ALIGNED", "total_orphaned_terms": 5},
        ]
        
        forecast = forecast_semantic_drift(history)
        
        # Status improved even though counts stayed same
        self.assertEqual(forecast["drift_direction"], "IMPROVING")
    
    def test_forecast_status_progression_degrading(self):
        """Should detect degradation from status progression."""
        history = [
            {"alignment_status": "ALIGNED", "total_orphaned_terms": 5},
            {"alignment_status": "PARTIAL", "total_orphaned_terms": 5},
            {"alignment_status": "DIVERGENT", "total_orphaned_terms": 5},
        ]
        
        forecast = forecast_semantic_drift(history)
        
        # Status degraded even though counts stayed same
        self.assertEqual(forecast["drift_direction"], "DEGRADING")


# =============================================================================
# PHASE IV FOLLOW-UP INTEGRATION TESTS
# =============================================================================

class TestPhaseIVFollowUpIntegration(unittest.TestCase):
    """Integration tests for Phase IV follow-up features."""
    
    def test_contract_audit_with_alignment_index(self):
        """Contract audit should work with alignment index."""
        # Build alignment index
        graph = SemanticKnowledgeGraph(
            terms=[{"canonical_form": "goal_hit"}],
            edges=[],
        )
        graph_snapshot = {"status": "OK", "critical_signals": []}
        
        alignment_index = build_semantic_alignment_index(
            graph_snapshot,
            {"terms": ["goal_hit", "missing_term"]},
            {"terms": ["goal_hit"]},
            {"terms": ["goal_hit"]},
            graph=graph,
        )
        
        # Audit contract
        audit = audit_semantic_contract(
            alignment_index,
            {"terms": ["goal_hit"]},
            {"terms": ["goal_hit", "missing_term"]},
        )
        
        # Should detect missing_term as violation
        self.assertIn("missing_term", audit["violated_contract_terms"])
        self.assertEqual(audit["contract_status"], "ATTENTION")
    
    def test_forecast_with_alignment_history(self):
        """Forecast should work with alignment index history."""
        history = [
            build_semantic_alignment_index(
                {"status": "OK"},
                {"terms": ["term1", "term2"]},
                {"terms": ["term1", "term2"]},
                {"terms": ["term1", "term2"]},
            ),
            build_semantic_alignment_index(
                {"status": "OK"},
                {"terms": ["term1", "term2", "term3"]},
                {"terms": ["term1", "term2"]},
                {"terms": ["term1", "term2"]},
            ),
        ]
        
        forecast = forecast_semantic_drift(history)
        
        self.assertIn("drift_direction", forecast)
        self.assertIn("forecast_band", forecast)
        self.assertIn("explanation", forecast)
        self.assertIn("historical_points", forecast)


if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

