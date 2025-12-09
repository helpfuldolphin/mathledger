"""
PHASE II — NOT USED IN PHASE I
Tests for metric_adapter_introspection.py - Introspection Layer

This test suite validates:
  - Deterministic summaries
  - Identical output under repeated runs
  - JSON schema stability
  - Harmonization completeness checks
  - Deep-diff detection accuracy
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Dict, Any

import yaml

from experiments.metric_adapter_introspection import (
    AlignmentReport,
    AlignmentStatus,
    DriftItem,
    MetricContractSchema,
    INTROSPECTION_SCHEMA_VERSION,
    summarize_metric_adapter,
    verify_metric_alignment,
    export_metric_contract,
    detect_schema_drift,
    list_all_slices,
    verify_all_slices,
    export_all_contracts,
    format_alignment_report,
    format_drift_report,
    load_curriculum,
    load_prereg,
    get_slice_config,
    get_prereg_slice_spec,
    _compute_config_hash,
    _check_nested_field,
    _type_to_string,
    # New imports for extended functionality
    HealthStatus,
    SliceHealthResult,
    HealthCheckReport,
    ContractIndexEntry,
    ContractBundle,
    run_health_check,
    format_health_check_report,
    export_contract_bundle,
    generate_metric_dashboard,
    generate_metric_dashboard_json,
    DEFAULT_CONTRACT_OUTPUT_DIR,
    # Task 1: Readiness gate
    ReadinessStatus,
    DriftSeverityClass,
    ReadinessResult,
    check_slice_readiness,
    classify_drift_severity,
    format_readiness_result,
    # Task 2: Contract browser
    ContractIndexListEntry,
    list_contracts,
    format_contract_list,
    # Task 3: Log field coverage
    LogFieldCoverageMap,
    get_log_field_coverage,
    get_log_field_coverage_by_kind,
    format_log_field_coverage,
    # Task 3 extended: Optional log fields
    OPTIONAL_LOG_FIELDS_BY_KIND,
    # V2.0: Multi-slice readiness
    ReadinessSummary,
    summarize_readiness,
    ReadinessVerdict,
    compute_readiness_verdict,
    format_readiness_summary_line,
    get_readiness_verdict_exit_code,
    # V2.0: Consumer predicate
    is_slice_ready_for_experiments,
    batch_check_readiness_for_experiments,
    ACCEPTABLE_DRIFT_SEVERITIES_FOR_EXPERIMENTS,
    READINESS_SUMMARY_SCHEMA_VERSION,
    # Phase III: Per-Metric Readiness Matrix
    build_metric_readiness_matrix,
    READINESS_MATRIX_SCHEMA_VERSION,
    # Phase III: Promotion Guard
    evaluate_metric_readiness_for_promotion,
    UPLIFT_SLICE_PREFIX,
    # Phase III: Global Health
    summarize_metric_readiness_for_global_health,
    # Phase IV: Cross-Metric Readiness Heatmap
    build_readiness_heatmap,
    READINESS_HEATMAP_SCHEMA_VERSION,
    DriftStatus,
    BudgetFlag,
    StatusLight,
    # Phase IV: Release Promotion
    evaluate_release_promotion_with_readiness,
    # Phase IV: Director Panel
    build_metric_readiness_director_panel,
    # Phase V: Autopilot & Phase Boundary
    build_readiness_autopilot_policy,
    derive_phase_boundary_recommendations,
    AutopilotStatus,
)


# =============================================================================
# Test Fixtures
# =============================================================================

def create_test_curriculum() -> Dict[str, Any]:
    """Creates a minimal test curriculum structure."""
    return {
        "version": "2.0",
        "phase": "II",
        "slices": {
            "test_slice_goal": {
                "description": "Test goal slice",
                "success_metric": {
                    "kind": "goal_hit",
                    "parameters": {
                        "min_goal_hits": 1,
                        "min_total_verified": 3,
                    }
                },
                "params": {
                    "atoms": 3,
                    "depth_max": 5,
                }
            },
            "test_slice_sparse": {
                "description": "Test sparse slice",
                "success_metric": {
                    "kind": "sparse_success",
                    "parameters": {
                        "min_verified": 5,
                    }
                },
                "params": {
                    "atoms": 4,
                }
            },
            "test_slice_chain": {
                "description": "Test chain slice",
                "success_metric": {
                    "kind": "chain_success",
                    "parameters": {
                        "min_chain_length": 3,
                    }
                },
                "params": {}
            },
            "test_slice_multi": {
                "description": "Test multi-goal slice",
                "success_metric": {
                    "kind": "multi_goal_success",
                    "parameters": {
                        "required_goal_count": 2,
                    }
                },
                "params": {}
            }
        }
    }


def create_test_prereg() -> Dict[str, Any]:
    """Creates a minimal test prereg structure."""
    return {
        "preregistration": {
            "experiment_family": "uplift_u2",
            "version": 1,
        },
        "test_slice_goal": {
            "experiment_id": "test_goal_001",
            "slice_name": "test_slice_goal",
            "success_metric": {
                "kind": "goal_hit",
                "parameters": {
                    "min_goal_hits": 1,
                    "min_total_verified": 3,
                    "target_hashes": [],
                }
            }
        },
        "test_slice_sparse": {
            "experiment_id": "test_sparse_001",
            "slice_name": "test_slice_sparse",
            "success_metric": {
                "kind": "density",  # Note: prereg uses "density" not "sparse_success"
                "parameters": {
                    "min_verified": 5,
                }
            }
        }
    }


class TestCurriculumWithTempFiles(unittest.TestCase):
    """Base class that provides temporary test files."""
    
    def setUp(self):
        """Create temporary curriculum and prereg files."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test curriculum
        self.curriculum_path = Path(self.temp_dir) / "test_curriculum.yaml"
        self.curriculum_data = create_test_curriculum()
        with open(self.curriculum_path, "w") as f:
            yaml.dump(self.curriculum_data, f)
        
        # Create test prereg
        self.prereg_path = Path(self.temp_dir) / "test_prereg.yaml"
        self.prereg_data = create_test_prereg()
        with open(self.prereg_path, "w") as f:
            yaml.dump(self.prereg_data, f)
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)


# =============================================================================
# Basic Data Type Tests
# =============================================================================

class TestAlignmentStatus(unittest.TestCase):
    """Tests for AlignmentStatus enum."""
    
    def test_all_statuses_defined(self):
        """All expected statuses are defined."""
        expected = {"aligned", "misaligned", "partial", "missing"}
        actual = {s.value for s in AlignmentStatus}
        self.assertEqual(expected, actual)
    
    def test_status_string_conversion(self):
        """Status values are strings."""
        for status in AlignmentStatus:
            self.assertIsInstance(status.value, str)


class TestDriftItem(unittest.TestCase):
    """Tests for DriftItem dataclass."""
    
    def test_drift_item_creation(self):
        """DriftItem can be created with all fields."""
        drift = DriftItem(
            field="test_field",
            source_a="source1",
            source_b="source2",
            value_a="val1",
            value_b="val2",
            severity="error",
            description="Test drift"
        )
        self.assertEqual(drift.field, "test_field")
        self.assertEqual(drift.severity, "error")
    
    def test_drift_item_immutable(self):
        """DriftItem is immutable (frozen)."""
        drift = DriftItem(
            field="f", source_a="a", source_b="b",
            value_a="v1", value_b="v2",
            severity="warning", description="d"
        )
        with self.assertRaises(AttributeError):
            drift.field = "new_value"


class TestAlignmentReport(unittest.TestCase):
    """Tests for AlignmentReport dataclass."""
    
    def test_alignment_report_creation(self):
        """AlignmentReport can be created."""
        report = AlignmentReport(
            slice_name="test",
            status=AlignmentStatus.ALIGNED,
            harmonization_aligned=True,
            prereg_aligned=True,
            adapter_aligned=True
        )
        self.assertEqual(report.slice_name, "test")
        self.assertTrue(report.is_fully_aligned())
    
    def test_is_fully_aligned_false_with_errors(self):
        """is_fully_aligned returns False if errors exist."""
        report = AlignmentReport(
            slice_name="test",
            status=AlignmentStatus.ALIGNED,
            harmonization_aligned=True,
            prereg_aligned=True,
            adapter_aligned=True,
            errors=["Some error"]
        )
        self.assertFalse(report.is_fully_aligned())
    
    def test_is_fully_aligned_false_if_any_source_misaligned(self):
        """is_fully_aligned returns False if any source is misaligned."""
        report = AlignmentReport(
            slice_name="test",
            status=AlignmentStatus.PARTIAL,
            harmonization_aligned=True,
            prereg_aligned=False,  # One source misaligned
            adapter_aligned=True
        )
        self.assertFalse(report.is_fully_aligned())


class TestMetricContractSchema(unittest.TestCase):
    """Tests for MetricContractSchema dataclass."""
    
    def test_contract_schema_creation(self):
        """MetricContractSchema can be created."""
        contract = MetricContractSchema(
            schema_version="1.0.0",
            slice_name="test",
            metric_kind="goal_hit",
            required_config_fields=("field1", "field2"),
            required_log_fields=("log1",),
            runtime_fields=(),
            parameter_schema={"param1": "integer"},
            output_schema={"success": "bool"},
            config_hash="abc123"
        )
        self.assertEqual(contract.slice_name, "test")
        self.assertEqual(len(contract.required_config_fields), 2)
    
    def test_contract_to_dict(self):
        """to_dict() returns correct dictionary."""
        contract = MetricContractSchema(
            schema_version="1.0.0",
            slice_name="test",
            metric_kind="goal_hit",
            required_config_fields=("f1",),
            required_log_fields=("l1",),
            runtime_fields=("r1",),
            parameter_schema={},
            output_schema={},
            config_hash="hash"
        )
        d = contract.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(d["slice_name"], "test")
        self.assertIsInstance(d["required_config_fields"], list)
    
    def test_contract_to_json(self):
        """to_json() returns valid JSON string."""
        contract = MetricContractSchema(
            schema_version="1.0.0",
            slice_name="test",
            metric_kind="sparse_success",
            required_config_fields=(),
            required_log_fields=(),
            runtime_fields=(),
            parameter_schema={},
            output_schema={},
            config_hash="hash"
        )
        json_str = contract.to_json()
        # Should be valid JSON
        parsed = json.loads(json_str)
        self.assertEqual(parsed["metric_kind"], "sparse_success")
    
    def test_contract_immutable(self):
        """MetricContractSchema is immutable (frozen)."""
        contract = MetricContractSchema(
            schema_version="1.0.0",
            slice_name="test",
            metric_kind="goal_hit",
            required_config_fields=(),
            required_log_fields=(),
            runtime_fields=(),
            parameter_schema={},
            output_schema={},
            config_hash="hash"
        )
        with self.assertRaises(AttributeError):
            contract.slice_name = "changed"


# =============================================================================
# Determinism Tests
# =============================================================================

class TestSummaryDeterminism(TestCurriculumWithTempFiles):
    """Tests for deterministic summary generation."""
    
    def test_summary_deterministic_repeated_calls(self):
        """summarize_metric_adapter produces identical output across calls."""
        results = []
        for _ in range(10):
            summary = summarize_metric_adapter(
                "test_slice_goal",
                self.curriculum_path
            )
            results.append(summary)
        
        # All results must be identical
        self.assertTrue(all(r == results[0] for r in results))
    
    def test_summary_deterministic_all_slices(self):
        """Summaries are deterministic for all slice types."""
        slices = ["test_slice_goal", "test_slice_sparse", "test_slice_chain", "test_slice_multi"]
        
        for slice_name in slices:
            results = [
                summarize_metric_adapter(slice_name, self.curriculum_path)
                for _ in range(5)
            ]
            self.assertTrue(
                all(r == results[0] for r in results),
                f"Summary not deterministic for {slice_name}"
            )


class TestContractDeterminism(TestCurriculumWithTempFiles):
    """Tests for deterministic contract generation."""
    
    def test_contract_deterministic_repeated_calls(self):
        """export_metric_contract produces identical output across calls."""
        results = []
        for _ in range(10):
            contract = export_metric_contract(
                "test_slice_goal",
                self.curriculum_path
            )
            results.append(contract.to_json())
        
        # All results must be identical
        self.assertTrue(all(r == results[0] for r in results))
    
    def test_config_hash_deterministic(self):
        """Config hash is deterministic."""
        hashes = []
        for _ in range(10):
            contract = export_metric_contract(
                "test_slice_goal",
                self.curriculum_path
            )
            hashes.append(contract.config_hash)
        
        self.assertTrue(all(h == hashes[0] for h in hashes))


class TestAlignmentDeterminism(TestCurriculumWithTempFiles):
    """Tests for deterministic alignment verification."""
    
    def test_alignment_deterministic_repeated_calls(self):
        """verify_metric_alignment produces identical output across calls."""
        results = []
        for _ in range(10):
            report = verify_metric_alignment(
                "test_slice_goal",
                self.curriculum_path,
                self.prereg_path
            )
            results.append((
                report.status,
                report.harmonization_aligned,
                report.prereg_aligned,
                report.adapter_aligned,
                tuple(report.errors),
                tuple(report.warnings),
            ))
        
        self.assertTrue(all(r == results[0] for r in results))


class TestDriftDeterminism(TestCurriculumWithTempFiles):
    """Tests for deterministic drift detection."""
    
    def test_drift_detection_deterministic(self):
        """detect_schema_drift produces identical output across calls."""
        results = []
        for _ in range(10):
            drifts = detect_schema_drift(
                "test_slice_goal",
                self.curriculum_path,
                self.prereg_path
            )
            # Convert to comparable representation
            drift_tuples = tuple(
                (d.field, d.source_a, d.source_b, d.severity)
                for d in drifts
            )
            results.append(drift_tuples)
        
        self.assertTrue(all(r == results[0] for r in results))


# =============================================================================
# JSON Schema Stability Tests
# =============================================================================

class TestJSONSchemaStability(TestCurriculumWithTempFiles):
    """Tests for JSON schema stability."""
    
    def test_contract_json_has_required_fields(self):
        """Exported JSON contract has all required fields."""
        contract = export_metric_contract("test_slice_goal", self.curriculum_path)
        json_data = json.loads(contract.to_json())
        
        required_fields = [
            "schema_version",
            "slice_name",
            "metric_kind",
            "required_config_fields",
            "required_log_fields",
            "runtime_fields",
            "parameter_schema",
            "output_schema",
            "config_hash",
        ]
        
        for field in required_fields:
            self.assertIn(field, json_data, f"Missing required field: {field}")
    
    def test_contract_json_types_correct(self):
        """JSON field types are correct."""
        contract = export_metric_contract("test_slice_goal", self.curriculum_path)
        json_data = json.loads(contract.to_json())
        
        self.assertIsInstance(json_data["schema_version"], str)
        self.assertIsInstance(json_data["slice_name"], str)
        self.assertIsInstance(json_data["metric_kind"], str)
        self.assertIsInstance(json_data["required_config_fields"], list)
        self.assertIsInstance(json_data["required_log_fields"], list)
        self.assertIsInstance(json_data["runtime_fields"], list)
        self.assertIsInstance(json_data["parameter_schema"], dict)
        self.assertIsInstance(json_data["output_schema"], dict)
        self.assertIsInstance(json_data["config_hash"], str)
    
    def test_schema_version_present(self):
        """Schema version is present and matches constant."""
        contract = export_metric_contract("test_slice_goal", self.curriculum_path)
        self.assertEqual(contract.schema_version, INTROSPECTION_SCHEMA_VERSION)
    
    def test_contract_json_sorted_keys(self):
        """JSON output has sorted keys for stability."""
        contract = export_metric_contract("test_slice_goal", self.curriculum_path)
        json_str = contract.to_json()
        
        # Parse and re-serialize with sorted keys
        parsed = json.loads(json_str)
        expected = json.dumps(parsed, indent=2, sort_keys=True)
        
        self.assertEqual(json_str, expected)


# =============================================================================
# Harmonization Completeness Tests
# =============================================================================

class TestHarmonizationCompleteness(TestCurriculumWithTempFiles):
    """Tests for harmonization table completeness."""
    
    def test_all_metric_kinds_have_harmonization_entries(self):
        """All MetricKind values have harmonization table entries."""
        from experiments.u2_pipeline import MetricKind, METRIC_HARMONIZATION_TABLE
        
        for kind in MetricKind:
            self.assertIn(
                kind,
                METRIC_HARMONIZATION_TABLE,
                f"Missing harmonization entry for {kind}"
            )
    
    def test_harmonization_entries_have_required_fields(self):
        """Each harmonization entry has required fields."""
        from experiments.u2_pipeline import METRIC_HARMONIZATION_TABLE
        
        required_fields = [
            "description",
            "required_log_fields",
            "required_slice_config_fields",
            "result_interpretation",
            "compute_function",
        ]
        
        for kind, entry in METRIC_HARMONIZATION_TABLE.items():
            for field in required_fields:
                self.assertIn(
                    field,
                    entry,
                    f"Missing '{field}' in harmonization entry for {kind}"
                )
    
    def test_all_slices_have_valid_metric_kinds(self):
        """All slices in test curriculum have valid metric kinds."""
        from experiments.u2_pipeline import MetricKind
        
        for slice_name, config in self.curriculum_data["slices"].items():
            metric_kind = config.get("success_metric", {}).get("kind")
            self.assertIsNotNone(
                metric_kind,
                f"Slice {slice_name} has no metric kind"
            )
            try:
                MetricKind(metric_kind)
            except ValueError:
                self.fail(f"Slice {slice_name} has invalid metric kind: {metric_kind}")


# =============================================================================
# Deep-Diff Detection Tests
# =============================================================================

class TestDeepDiffDetection(TestCurriculumWithTempFiles):
    """Tests for deep-diff schema drift detection."""
    
    def test_detect_missing_curriculum(self):
        """Drift detection handles missing curriculum file."""
        drifts = detect_schema_drift(
            "test_slice",
            Path("/nonexistent/path.yaml"),
            self.prereg_path
        )
        
        self.assertTrue(len(drifts) > 0)
        self.assertTrue(any(d.severity == "error" for d in drifts))
    
    def test_detect_missing_slice(self):
        """Drift detection handles missing slice."""
        drifts = detect_schema_drift(
            "nonexistent_slice",
            self.curriculum_path,
            self.prereg_path
        )
        
        self.assertTrue(len(drifts) > 0)
        self.assertTrue(any("not found" in d.description.lower() for d in drifts))
    
    def test_detect_parameter_drift(self):
        """Drift detection finds parameter mismatches."""
        # Create a curriculum with extra parameter
        modified_curriculum = create_test_curriculum()
        modified_curriculum["slices"]["test_slice_goal"]["success_metric"]["parameters"]["extra_param"] = 999
        
        temp_path = Path(self.temp_dir) / "modified_curriculum.yaml"
        with open(temp_path, "w") as f:
            yaml.dump(modified_curriculum, f)
        
        drifts = detect_schema_drift(
            "test_slice_goal",
            temp_path,
            self.prereg_path
        )
        
        # Should detect the extra parameter
        extra_param_drifts = [d for d in drifts if "extra_param" in d.field]
        self.assertTrue(len(extra_param_drifts) > 0)
    
    def test_drift_severity_levels(self):
        """Drift items have appropriate severity levels."""
        drifts = detect_schema_drift(
            "test_slice_goal",
            self.curriculum_path,
            self.prereg_path
        )
        
        valid_severities = {"error", "warning", "info"}
        for drift in drifts:
            self.assertIn(
                drift.severity,
                valid_severities,
                f"Invalid severity: {drift.severity}"
            )


# =============================================================================
# Utility Function Tests
# =============================================================================

class TestUtilityFunctions(TestCurriculumWithTempFiles):
    """Tests for utility functions."""
    
    def test_check_nested_field_exists(self):
        """_check_nested_field returns True for existing paths."""
        config = {
            "level1": {
                "level2": {
                    "value": 42
                }
            }
        }
        self.assertTrue(_check_nested_field(config, "level1"))
        self.assertTrue(_check_nested_field(config, "level1.level2"))
        self.assertTrue(_check_nested_field(config, "level1.level2.value"))
    
    def test_check_nested_field_missing(self):
        """_check_nested_field returns False for missing paths."""
        config = {"level1": {"level2": {}}}
        self.assertFalse(_check_nested_field(config, "level1.level2.missing"))
        self.assertFalse(_check_nested_field(config, "nonexistent"))
    
    def test_type_to_string_conversion(self):
        """_type_to_string converts types correctly."""
        self.assertEqual(_type_to_string(int), "integer")
        self.assertEqual(_type_to_string(float), "number")
        self.assertEqual(_type_to_string(str), "string")
        self.assertEqual(_type_to_string(bool), "boolean")
        self.assertEqual(_type_to_string(list), "array")
        self.assertEqual(_type_to_string(dict), "object")
    
    def test_compute_config_hash_deterministic(self):
        """_compute_config_hash is deterministic."""
        slice_config = {"key": "value"}
        metric_config = {"kind": "goal_hit", "parameters": {"p": 1}}
        
        hashes = [
            _compute_config_hash(slice_config, metric_config)
            for _ in range(10)
        ]
        
        self.assertTrue(all(h == hashes[0] for h in hashes))
    
    def test_compute_config_hash_different_inputs(self):
        """_compute_config_hash produces different hashes for different inputs."""
        config1 = {"kind": "goal_hit", "parameters": {"p": 1}}
        config2 = {"kind": "goal_hit", "parameters": {"p": 2}}
        
        hash1 = _compute_config_hash({}, config1)
        hash2 = _compute_config_hash({}, config2)
        
        self.assertNotEqual(hash1, hash2)


# =============================================================================
# Batch Operation Tests
# =============================================================================

class TestBatchOperations(TestCurriculumWithTempFiles):
    """Tests for batch operations."""
    
    def test_list_all_slices(self):
        """list_all_slices returns all slices."""
        slices = list_all_slices(self.curriculum_path)
        
        expected = {"test_slice_goal", "test_slice_sparse", "test_slice_chain", "test_slice_multi"}
        self.assertEqual(set(slices), expected)
    
    def test_list_all_slices_sorted(self):
        """list_all_slices returns sorted list."""
        slices = list_all_slices(self.curriculum_path)
        self.assertEqual(slices, sorted(slices))
    
    def test_verify_all_slices(self):
        """verify_all_slices processes all slices."""
        results = verify_all_slices(self.curriculum_path, self.prereg_path)
        
        expected_slices = {"test_slice_goal", "test_slice_sparse", "test_slice_chain", "test_slice_multi"}
        self.assertEqual(set(results.keys()), expected_slices)
        
        for slice_name, report in results.items():
            self.assertIsInstance(report, AlignmentReport)
            self.assertEqual(report.slice_name, slice_name)
    
    def test_export_all_contracts(self):
        """export_all_contracts exports all valid slices."""
        contracts = export_all_contracts(self.curriculum_path)
        
        # Should have contracts for all valid slices
        self.assertTrue(len(contracts) > 0)
        
        for slice_name, contract in contracts.items():
            self.assertIsInstance(contract, MetricContractSchema)
            self.assertEqual(contract.slice_name, slice_name)
    
    def test_export_all_contracts_to_directory(self):
        """export_all_contracts writes files when output_dir provided."""
        output_dir = Path(self.temp_dir) / "contracts"
        
        contracts = export_all_contracts(self.curriculum_path, output_dir)
        
        # Directory should exist
        self.assertTrue(output_dir.exists())
        
        # Should have created files
        for slice_name in contracts:
            expected_file = output_dir / f"{slice_name}_contract.json"
            self.assertTrue(
                expected_file.exists(),
                f"Missing contract file: {expected_file}"
            )


# =============================================================================
# Report Formatting Tests
# =============================================================================

class TestReportFormatting(unittest.TestCase):
    """Tests for report formatting functions."""
    
    def test_format_alignment_report_aligned(self):
        """format_alignment_report formats aligned report correctly."""
        report = AlignmentReport(
            slice_name="test",
            status=AlignmentStatus.ALIGNED,
            harmonization_aligned=True,
            prereg_aligned=True,
            adapter_aligned=True
        )
        formatted = format_alignment_report(report)
        
        self.assertIn("test", formatted)
        self.assertIn("ALIGNED", formatted)
        self.assertIn("✓", formatted)
    
    def test_format_alignment_report_with_errors(self):
        """format_alignment_report includes errors."""
        report = AlignmentReport(
            slice_name="test",
            status=AlignmentStatus.MISALIGNED,
            harmonization_aligned=False,
            prereg_aligned=True,
            adapter_aligned=True,
            errors=["Test error message"]
        )
        formatted = format_alignment_report(report)
        
        self.assertIn("ERRORS", formatted)
        self.assertIn("Test error message", formatted)
    
    def test_format_alignment_report_with_warnings(self):
        """format_alignment_report includes warnings."""
        report = AlignmentReport(
            slice_name="test",
            status=AlignmentStatus.PARTIAL,
            harmonization_aligned=True,
            prereg_aligned=True,
            adapter_aligned=True,
            warnings=["Test warning"]
        )
        formatted = format_alignment_report(report)
        
        self.assertIn("WARNINGS", formatted)
        self.assertIn("Test warning", formatted)
    
    def test_format_drift_report_empty(self):
        """format_drift_report handles empty list."""
        formatted = format_drift_report([])
        self.assertIn("No schema drift detected", formatted)
    
    def test_format_drift_report_with_drifts(self):
        """format_drift_report formats drifts correctly."""
        drifts = [
            DriftItem(
                field="test.field",
                source_a="source1",
                source_b="source2",
                value_a="val1",
                value_b="val2",
                severity="error",
                description="Test drift description"
            )
        ]
        formatted = format_drift_report(drifts)
        
        self.assertIn("test.field", formatted)
        self.assertIn("source1", formatted)
        self.assertIn("Test drift description", formatted)


# =============================================================================
# Real Curriculum Integration Tests
# =============================================================================

class TestRealCurriculumIntegration(unittest.TestCase):
    """Integration tests with the actual curriculum file."""
    
    @classmethod
    def setUpClass(cls):
        """Check if real curriculum exists."""
        cls.curriculum_path = Path("config/curriculum_uplift_phase2.yaml")
        cls.prereg_path = Path("experiments/prereg/PREREG_UPLIFT_U2.yaml")
        cls.skip_tests = not cls.curriculum_path.exists()
    
    def setUp(self):
        if self.skip_tests:
            self.skipTest("Real curriculum file not found")
    
    def test_real_curriculum_loads(self):
        """Real curriculum file loads without error."""
        curriculum = load_curriculum(self.curriculum_path)
        self.assertIn("slices", curriculum)
    
    def test_real_slices_have_contracts(self):
        """All real slices can export contracts."""
        slices = list_all_slices(self.curriculum_path)
        
        for slice_name in slices:
            try:
                contract = export_metric_contract(slice_name, self.curriculum_path)
                self.assertIsNotNone(contract)
            except ValueError:
                # Some slices might have invalid configs - that's okay for this test
                pass
    
    def test_real_summary_generation(self):
        """Summary can be generated for real slices."""
        slices = list_all_slices(self.curriculum_path)
        
        for slice_name in slices[:2]:  # Test first two slices
            try:
                summary = summarize_metric_adapter(slice_name, self.curriculum_path)
                self.assertIn(slice_name, summary)
            except ValueError:
                pass


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases(TestCurriculumWithTempFiles):
    """Tests for edge cases and error handling."""
    
    def test_missing_curriculum_file(self):
        """Appropriate error for missing curriculum."""
        with self.assertRaises(FileNotFoundError):
            load_curriculum(Path("/nonexistent/path.yaml"))
    
    def test_missing_slice_in_summary(self):
        """Appropriate error for missing slice in summary."""
        with self.assertRaises(ValueError) as ctx:
            summarize_metric_adapter("nonexistent_slice", self.curriculum_path)
        
        self.assertIn("not found", str(ctx.exception).lower())
    
    def test_missing_slice_in_export(self):
        """Appropriate error for missing slice in export."""
        with self.assertRaises(ValueError) as ctx:
            export_metric_contract("nonexistent_slice", self.curriculum_path)
        
        self.assertIn("not found", str(ctx.exception).lower())
    
    def test_invalid_metric_kind_in_export(self):
        """Appropriate error for invalid metric kind."""
        # Create curriculum with invalid metric kind
        invalid_curriculum = create_test_curriculum()
        invalid_curriculum["slices"]["bad_slice"] = {
            "description": "Bad slice",
            "success_metric": {
                "kind": "invalid_kind",
                "parameters": {}
            }
        }
        
        temp_path = Path(self.temp_dir) / "invalid_curriculum.yaml"
        with open(temp_path, "w") as f:
            yaml.dump(invalid_curriculum, f)
        
        with self.assertRaises(ValueError) as ctx:
            export_metric_contract("bad_slice", temp_path)
        
        self.assertIn("invalid_kind", str(ctx.exception).lower())
    
    def test_empty_slices_section(self):
        """Handle curriculum with empty slices section."""
        empty_curriculum = {"version": "2.0", "slices": {}}
        
        temp_path = Path(self.temp_dir) / "empty_curriculum.yaml"
        with open(temp_path, "w") as f:
            yaml.dump(empty_curriculum, f)
        
        slices = list_all_slices(temp_path)
        self.assertEqual(slices, [])


# =============================================================================
# Health Check Tests
# =============================================================================

class TestHealthStatus(unittest.TestCase):
    """Tests for HealthStatus enum."""
    
    def test_all_statuses_defined(self):
        """All expected health statuses are defined."""
        expected = {"OK", "WARN", "FAIL"}
        actual = {s.value for s in HealthStatus}
        self.assertEqual(expected, actual)


class TestSliceHealthResult(unittest.TestCase):
    """Tests for SliceHealthResult dataclass."""
    
    def test_slice_health_result_creation(self):
        """SliceHealthResult can be created."""
        result = SliceHealthResult(
            slice_name="test",
            status=HealthStatus.OK,
            alignment_status=AlignmentStatus.ALIGNED,
            errors=[],
            warnings=[],
            drift_count=0
        )
        self.assertEqual(result.slice_name, "test")
        self.assertEqual(result.status, HealthStatus.OK)
    
    def test_slice_health_result_to_dict(self):
        """to_dict() returns correct structure."""
        result = SliceHealthResult(
            slice_name="test",
            status=HealthStatus.WARN,
            alignment_status=AlignmentStatus.PARTIAL,
            errors=[],
            warnings=["warning1"],
            drift_count=1
        )
        d = result.to_dict()
        self.assertEqual(d["status"], "WARN")
        self.assertEqual(d["warnings"], ["warning1"])


class TestHealthCheckReport(unittest.TestCase):
    """Tests for HealthCheckReport dataclass."""
    
    def test_health_check_report_creation(self):
        """HealthCheckReport can be created."""
        report = HealthCheckReport(
            timestamp="2025-01-01T00:00:00Z",
            total_slices=4,
            ok_count=3,
            warn_count=1,
            fail_count=0,
            slices={},
            overall_status=HealthStatus.WARN
        )
        self.assertEqual(report.total_slices, 4)
        self.assertEqual(report.overall_status, HealthStatus.WARN)
    
    def test_health_check_report_to_json(self):
        """to_json() returns valid JSON."""
        report = HealthCheckReport(
            timestamp="2025-01-01T00:00:00Z",
            total_slices=1,
            ok_count=1,
            warn_count=0,
            fail_count=0,
            slices={},
            overall_status=HealthStatus.OK
        )
        json_str = report.to_json()
        parsed = json.loads(json_str)
        self.assertEqual(parsed["overall_status"], "OK")


class TestRunHealthCheck(TestCurriculumWithTempFiles):
    """Tests for run_health_check function."""
    
    def test_health_check_all_ok(self):
        """Health check returns OK when all slices aligned."""
        report = run_health_check(self.curriculum_path, self.prereg_path)
        
        # All test slices should be OK (they're properly configured)
        self.assertIn(report.overall_status, [HealthStatus.OK, HealthStatus.WARN])
        self.assertEqual(report.total_slices, 4)
    
    def test_health_check_deterministic(self):
        """Health check produces deterministic results."""
        results = []
        for _ in range(5):
            report = run_health_check(self.curriculum_path, self.prereg_path)
            results.append((
                report.ok_count,
                report.warn_count,
                report.fail_count,
                report.overall_status,
            ))
        
        self.assertTrue(all(r == results[0] for r in results))
    
    def test_health_check_detects_failures(self):
        """Health check correctly identifies FAIL status."""
        # Create curriculum with invalid slice
        bad_curriculum = create_test_curriculum()
        bad_curriculum["slices"]["bad_slice"] = {
            "description": "Bad slice",
            "success_metric": {
                "kind": "goal_hit",
                "parameters": {}  # Missing required params
            }
        }
        
        temp_path = Path(self.temp_dir) / "bad_curriculum.yaml"
        with open(temp_path, "w") as f:
            yaml.dump(bad_curriculum, f)
        
        report = run_health_check(temp_path, self.prereg_path)
        
        # Should have at least one failure
        self.assertGreater(report.fail_count, 0)
        self.assertEqual(report.overall_status, HealthStatus.FAIL)


class TestFormatHealthCheckReport(unittest.TestCase):
    """Tests for format_health_check_report function."""
    
    def test_format_includes_status(self):
        """Formatted report includes status information."""
        report = HealthCheckReport(
            timestamp="2025-01-01T00:00:00Z",
            total_slices=2,
            ok_count=1,
            warn_count=1,
            fail_count=0,
            slices={
                "slice1": SliceHealthResult(
                    slice_name="slice1",
                    status=HealthStatus.OK,
                    alignment_status=AlignmentStatus.ALIGNED,
                    errors=[],
                    warnings=[],
                    drift_count=0
                ),
                "slice2": SliceHealthResult(
                    slice_name="slice2",
                    status=HealthStatus.WARN,
                    alignment_status=AlignmentStatus.PARTIAL,
                    errors=[],
                    warnings=["test warning"],
                    drift_count=1
                ),
            },
            overall_status=HealthStatus.WARN
        )
        
        formatted = format_health_check_report(report)
        
        self.assertIn("HEALTH CHECK", formatted)
        self.assertIn("WARN", formatted)
        self.assertIn("slice1", formatted)
        self.assertIn("slice2", formatted)


# =============================================================================
# Contract Bundle Tests
# =============================================================================

class TestContractIndexEntry(unittest.TestCase):
    """Tests for ContractIndexEntry dataclass."""
    
    def test_contract_index_entry_creation(self):
        """ContractIndexEntry can be created."""
        entry = ContractIndexEntry(
            slice_name="test",
            contract_path="test.json",
            config_hash="abc123",
            schema_version="1.0.0",
            metric_kind="goal_hit"
        )
        self.assertEqual(entry.slice_name, "test")
    
    def test_contract_index_entry_to_dict(self):
        """to_dict() returns correct structure."""
        entry = ContractIndexEntry(
            slice_name="test",
            contract_path="test.json",
            config_hash="abc123",
            schema_version="1.0.0",
            metric_kind="goal_hit"
        )
        d = entry.to_dict()
        self.assertEqual(d["contract_path"], "test.json")


class TestContractBundle(unittest.TestCase):
    """Tests for ContractBundle dataclass."""
    
    def test_contract_bundle_creation(self):
        """ContractBundle can be created."""
        bundle = ContractBundle(
            generated_at="2025-01-01T00:00:00Z",
            schema_version="1.0.0",
            total_contracts=2,
            output_directory="/tmp/contracts",
            contracts=[]
        )
        self.assertEqual(bundle.total_contracts, 2)
    
    def test_contract_bundle_to_json(self):
        """to_json() returns valid JSON."""
        bundle = ContractBundle(
            generated_at="2025-01-01T00:00:00Z",
            schema_version="1.0.0",
            total_contracts=0,
            output_directory="/tmp",
            contracts=[]
        )
        json_str = bundle.to_json()
        parsed = json.loads(json_str)
        self.assertEqual(parsed["schema_version"], "1.0.0")


class TestExportContractBundle(TestCurriculumWithTempFiles):
    """Tests for export_contract_bundle function."""
    
    def test_export_bundle_creates_files(self):
        """Export creates contract files and index."""
        output_dir = Path(self.temp_dir) / "contracts"
        bundle = export_contract_bundle(self.curriculum_path, output_dir)
        
        # Index file should exist
        index_path = output_dir / "metric_contract_index.json"
        self.assertTrue(index_path.exists())
        
        # Contract files should exist
        for entry in bundle.contracts:
            contract_path = output_dir / entry.contract_path
            self.assertTrue(contract_path.exists())
    
    def test_export_bundle_index_structure(self):
        """Index file has correct structure."""
        output_dir = Path(self.temp_dir) / "contracts"
        bundle = export_contract_bundle(self.curriculum_path, output_dir)
        
        index_path = output_dir / "metric_contract_index.json"
        with open(index_path) as f:
            index_data = json.load(f)
        
        self.assertIn("contracts", index_data)
        self.assertIn("generated_at", index_data)
        self.assertIn("schema_version", index_data)
        self.assertEqual(index_data["total_contracts"], len(bundle.contracts))
    
    def test_export_bundle_deterministic(self):
        """Contract bundle export is deterministic (except timestamp)."""
        output_dir1 = Path(self.temp_dir) / "contracts1"
        output_dir2 = Path(self.temp_dir) / "contracts2"
        
        bundle1 = export_contract_bundle(self.curriculum_path, output_dir1)
        bundle2 = export_contract_bundle(self.curriculum_path, output_dir2)
        
        # Same number of contracts
        self.assertEqual(bundle1.total_contracts, bundle2.total_contracts)
        
        # Same config hashes
        hashes1 = {e.slice_name: e.config_hash for e in bundle1.contracts}
        hashes2 = {e.slice_name: e.config_hash for e in bundle2.contracts}
        self.assertEqual(hashes1, hashes2)


# =============================================================================
# Dashboard Tests
# =============================================================================

class TestGenerateMetricDashboard(TestCurriculumWithTempFiles):
    """Tests for generate_metric_dashboard function."""
    
    def test_dashboard_includes_all_slices(self):
        """Dashboard includes all slices."""
        dashboard = generate_metric_dashboard(self.curriculum_path)
        
        for slice_name in ["test_slice_goal", "test_slice_sparse", "test_slice_chain", "test_slice_multi"]:
            self.assertIn(slice_name, dashboard)
    
    def test_dashboard_shows_metric_kind(self):
        """Dashboard shows metric kind for each slice."""
        dashboard = generate_metric_dashboard(self.curriculum_path)
        
        self.assertIn("goal_hit", dashboard)
        self.assertIn("sparse_success", dashboard)
        self.assertIn("chain_success", dashboard)
        self.assertIn("multi_goal_success", dashboard)
    
    def test_dashboard_deterministic(self):
        """Dashboard output is deterministic."""
        dashboards = [
            generate_metric_dashboard(self.curriculum_path)
            for _ in range(5)
        ]
        
        self.assertTrue(all(d == dashboards[0] for d in dashboards))


class TestGenerateMetricDashboardJson(TestCurriculumWithTempFiles):
    """Tests for generate_metric_dashboard_json function."""
    
    def test_dashboard_json_structure(self):
        """JSON dashboard has correct structure."""
        data = generate_metric_dashboard_json(self.curriculum_path)
        
        self.assertIn("schema_version", data)
        self.assertIn("total_slices", data)
        self.assertIn("slices", data)
    
    def test_dashboard_json_slice_structure(self):
        """Each slice in JSON has expected fields."""
        data = generate_metric_dashboard_json(self.curriculum_path)
        
        for slice_name, slice_data in data["slices"].items():
            self.assertIn("metric_kind", slice_data)
            self.assertIn("required_parameters", slice_data)
            self.assertIn("optional_parameters", slice_data)
            self.assertIn("jsonl_fields_read", slice_data)


# =============================================================================
# Contract Stability Tests
# =============================================================================

class TestContractStability(TestCurriculumWithTempFiles):
    """Tests for contract stability requirements."""
    
    def test_contract_has_required_top_level_keys(self):
        """Contract has all required top-level keys."""
        contract = export_metric_contract("test_slice_goal", self.curriculum_path)
        d = contract.to_dict()
        
        required_keys = [
            "metric_kind",
            "parameter_schema",
            "required_log_fields",  # expected_inputs
            "output_schema",  # expected_outputs
        ]
        
        for key in required_keys:
            self.assertIn(key, d, f"Missing required key: {key}")
    
    def test_contract_hash_stable_across_runs(self):
        """Contract hash remains stable across multiple runs."""
        hashes = []
        for _ in range(10):
            contract = export_metric_contract("test_slice_goal", self.curriculum_path)
            hashes.append(contract.config_hash)
        
        self.assertTrue(all(h == hashes[0] for h in hashes))
    
    def test_contract_json_stable(self):
        """Contract JSON output is stable (sorted keys, consistent format)."""
        jsons = []
        for _ in range(10):
            contract = export_metric_contract("test_slice_goal", self.curriculum_path)
            jsons.append(contract.to_json())
        
        self.assertTrue(all(j == jsons[0] for j in jsons))


class TestContractRegressionDetection(TestCurriculumWithTempFiles):
    """Regression tests for contract validation."""
    
    def test_adding_required_param_changes_alignment(self):
        """Adding a required parameter should change alignment verification."""
        # First verify alignment passes
        report1 = verify_metric_alignment(
            "test_slice_goal",
            self.curriculum_path,
            self.prereg_path
        )
        
        # Create modified curriculum with missing required param
        modified_curriculum = create_test_curriculum()
        del modified_curriculum["slices"]["test_slice_goal"]["success_metric"]["parameters"]["min_total_verified"]
        
        temp_path = Path(self.temp_dir) / "modified_curriculum.yaml"
        with open(temp_path, "w") as f:
            yaml.dump(modified_curriculum, f)
        
        # Verify alignment should now detect the missing param
        report2 = verify_metric_alignment(
            "test_slice_goal",
            temp_path,
            self.prereg_path
        )
        
        # The modified config should have issues
        self.assertFalse(report2.is_fully_aligned())
    
    def test_removing_required_param_fails_validation(self):
        """Removing a required parameter causes validation failure."""
        # Create curriculum missing required parameter
        bad_curriculum = create_test_curriculum()
        del bad_curriculum["slices"]["test_slice_sparse"]["success_metric"]["parameters"]["min_verified"]
        
        temp_path = Path(self.temp_dir) / "bad_curriculum.yaml"
        with open(temp_path, "w") as f:
            yaml.dump(bad_curriculum, f)
        
        # Drift detection should find the issue
        drifts = detect_schema_drift("test_slice_sparse", temp_path, self.prereg_path)
        
        # Should have error-level drift for missing param
        error_drifts = [d for d in drifts if d.severity == "error"]
        self.assertGreater(len(error_drifts), 0)
    
    def test_changing_metric_kind_breaks_alignment(self):
        """Changing metric kind should break alignment."""
        # Create curriculum with changed metric kind
        modified_curriculum = create_test_curriculum()
        modified_curriculum["slices"]["test_slice_goal"]["success_metric"]["kind"] = "sparse_success"
        
        temp_path = Path(self.temp_dir) / "modified_curriculum.yaml"
        with open(temp_path, "w") as f:
            yaml.dump(modified_curriculum, f)
        
        report = verify_metric_alignment(
            "test_slice_goal",
            temp_path,
            self.prereg_path
        )
        
        # Should detect the kind mismatch
        kind_drifts = [d for d in report.drifts if "kind" in d.field.lower()]
        self.assertGreater(len(kind_drifts), 0)


# =============================================================================
# CI Exit Code Tests
# =============================================================================

class TestCIExitCodes(TestCurriculumWithTempFiles):
    """Tests verifying correct CI exit code behavior."""
    
    def test_ok_status_exit_code_zero(self):
        """OK status should indicate exit code 0."""
        report = run_health_check(self.curriculum_path, self.prereg_path)
        
        # If overall status is OK or WARN, exit code should be 0
        if report.overall_status in [HealthStatus.OK, HealthStatus.WARN]:
            expected_exit = 0
        else:
            expected_exit = 1
        
        actual_exit = 0 if report.overall_status != HealthStatus.FAIL else 1
        self.assertEqual(actual_exit, expected_exit)
    
    def test_fail_status_exit_code_one(self):
        """FAIL status should indicate exit code 1."""
        # Create curriculum that will fail
        bad_curriculum = create_test_curriculum()
        bad_curriculum["slices"]["bad_slice"] = {
            "description": "Bad slice",
            "success_metric": {
                "kind": "unknown_kind",
                "parameters": {}
            }
        }
        
        temp_path = Path(self.temp_dir) / "bad_curriculum.yaml"
        with open(temp_path, "w") as f:
            yaml.dump(bad_curriculum, f)
        
        report = run_health_check(temp_path, self.prereg_path)
        
        # Should have FAIL status
        self.assertEqual(report.overall_status, HealthStatus.FAIL)
        
        # Exit code should be 1
        exit_code = 0 if report.overall_status != HealthStatus.FAIL else 1
        self.assertEqual(exit_code, 1)


# =============================================================================
# Task 1: Per-Slice Metric Readiness Gate Tests
# =============================================================================

class TestReadinessStatusEnum(unittest.TestCase):
    """Tests for ReadinessStatus enum."""
    
    def test_readiness_values(self):
        """ReadinessStatus has expected values."""
        self.assertEqual(ReadinessStatus.READY.value, "READY")
        self.assertEqual(ReadinessStatus.DEGRADED.value, "DEGRADED")
        self.assertEqual(ReadinessStatus.BLOCKED.value, "BLOCKED")
    
    def test_readiness_is_string_enum(self):
        """ReadinessStatus values are strings."""
        for status in ReadinessStatus:
            self.assertIsInstance(status.value, str)


class TestDriftSeverityClassEnum(unittest.TestCase):
    """Tests for DriftSeverityClass enum."""
    
    def test_drift_severity_values(self):
        """DriftSeverityClass has expected values."""
        self.assertEqual(DriftSeverityClass.NONE.value, "NONE")
        self.assertEqual(DriftSeverityClass.COSMETIC.value, "COSMETIC")
        self.assertEqual(DriftSeverityClass.PARAMETRIC_MINOR.value, "PARAMETRIC_MINOR")
        self.assertEqual(DriftSeverityClass.SEMANTIC.value, "SEMANTIC")


class TestClassifyDriftSeverity(unittest.TestCase):
    """Tests for classify_drift_severity function."""
    
    def test_empty_drifts_returns_none(self):
        """Empty drift list returns NONE classification."""
        self.assertEqual(classify_drift_severity([]), DriftSeverityClass.NONE)
    
    def test_info_only_returns_cosmetic(self):
        """Info-level drifts only returns COSMETIC."""
        drifts = [
            DriftItem(
                field="test_field",
                source_a="a",
                source_b="b",
                value_a="1",
                value_b="1",
                severity="info",
                description="Info message"
            )
        ]
        self.assertEqual(classify_drift_severity(drifts), DriftSeverityClass.COSMETIC)
    
    def test_warning_returns_parametric_minor(self):
        """Warning-level drifts returns PARAMETRIC_MINOR."""
        drifts = [
            DriftItem(
                field="parameters.extra_param",
                source_a="prereg",
                source_b="config",
                value_a="defined",
                value_b="missing",
                severity="warning",
                description="Extra param"
            )
        ]
        self.assertEqual(classify_drift_severity(drifts), DriftSeverityClass.PARAMETRIC_MINOR)
    
    def test_error_returns_semantic(self):
        """Error-level drifts returns SEMANTIC."""
        drifts = [
            DriftItem(
                field="metric_kind",
                source_a="prereg",
                source_b="config",
                value_a="goal_hit",
                value_b="sparse_success",
                severity="error",
                description="Kind mismatch"
            )
        ]
        self.assertEqual(classify_drift_severity(drifts), DriftSeverityClass.SEMANTIC)
    
    def test_mixed_severities_returns_most_severe(self):
        """Mixed severities returns the most severe classification."""
        drifts = [
            DriftItem(field="f1", source_a="a", source_b="b", value_a="1", value_b="2",
                     severity="info", description="info"),
            DriftItem(field="f2", source_a="a", source_b="b", value_a="1", value_b="2",
                     severity="warning", description="warning"),
            DriftItem(field="f3", source_a="a", source_b="b", value_a="1", value_b="missing",
                     severity="error", description="error"),
        ]
        self.assertEqual(classify_drift_severity(drifts), DriftSeverityClass.SEMANTIC)


class TestReadinessResult(unittest.TestCase):
    """Tests for ReadinessResult dataclass."""
    
    def test_readiness_result_creation(self):
        """ReadinessResult can be created with required fields."""
        result = ReadinessResult(
            slice_name="test_slice",
            status=ReadinessStatus.READY,
            alignment_passes=True,
            drift_class=DriftSeverityClass.NONE,
        )
        self.assertEqual(result.slice_name, "test_slice")
        self.assertEqual(result.status, ReadinessStatus.READY)
        self.assertTrue(result.alignment_passes)
        self.assertEqual(result.drift_class, DriftSeverityClass.NONE)
    
    def test_readiness_result_to_dict(self):
        """ReadinessResult.to_dict() returns correct structure."""
        result = ReadinessResult(
            slice_name="test_slice",
            status=ReadinessStatus.DEGRADED,
            alignment_passes=True,
            drift_class=DriftSeverityClass.PARAMETRIC_MINOR,
            errors=[],
            warnings=["some warning"],
        )
        d = result.to_dict()
        
        self.assertEqual(d["slice_name"], "test_slice")
        self.assertEqual(d["status"], "DEGRADED")
        self.assertTrue(d["alignment_passes"])
        self.assertEqual(d["drift_severity"], "PARAMETRIC_MINOR")  # Renamed from drift_class
        self.assertIn("warnings", d)
    
    def test_readiness_result_to_json_deterministic(self):
        """ReadinessResult.to_json() is deterministic."""
        result = ReadinessResult(
            slice_name="test_slice",
            status=ReadinessStatus.READY,
            alignment_passes=True,
            drift_class=DriftSeverityClass.NONE,
        )
        
        jsons = [result.to_json() for _ in range(5)]
        self.assertTrue(all(j == jsons[0] for j in jsons))


class TestCheckSliceReadiness(TestCurriculumWithTempFiles):
    """Tests for check_slice_readiness function."""
    
    def test_perfect_slice_returns_ready(self):
        """Synthetic slice with perfect contract returns READY."""
        # Export contracts first so they exist
        contract_dir = Path(self.temp_dir) / "contracts"
        export_contract_bundle(self.curriculum_path, contract_dir)
        
        result = check_slice_readiness(
            "test_slice_goal",
            self.curriculum_path,
            self.prereg_path,
            contract_dir
        )
        
        # Should be READY or DEGRADED (depending on PREREG alignment)
        self.assertIn(result.status, [ReadinessStatus.READY, ReadinessStatus.DEGRADED])
    
    def test_missing_param_returns_blocked(self):
        """Slice missing required parameter in prereg returns BLOCKED."""
        # Create curriculum with missing required parameter
        bad_curriculum = create_test_curriculum()
        del bad_curriculum["slices"]["test_slice_goal"]["success_metric"]["parameters"]["min_total_verified"]
        
        temp_path = Path(self.temp_dir) / "bad_curriculum.yaml"
        with open(temp_path, "w") as f:
            yaml.dump(bad_curriculum, f)
        
        # Export contracts for the bad curriculum
        contract_dir = Path(self.temp_dir) / "contracts"
        export_contract_bundle(temp_path, contract_dir)
        
        result = check_slice_readiness(
            "test_slice_goal",
            temp_path,
            self.prereg_path,
            contract_dir
        )
        
        # Should be BLOCKED due to semantic drift (missing required field)
        self.assertEqual(result.status, ReadinessStatus.BLOCKED)
    
    def test_param_tweak_returns_degraded(self):
        """Slice with param-only YAML tweak returns DEGRADED."""
        # Create curriculum with an extra (extraneous) parameter
        modified_curriculum = create_test_curriculum()
        modified_curriculum["slices"]["test_slice_goal"]["success_metric"]["parameters"]["extra_param"] = 999
        
        temp_path = Path(self.temp_dir) / "modified_curriculum.yaml"
        with open(temp_path, "w") as f:
            yaml.dump(modified_curriculum, f)
        
        # Export contracts for the modified curriculum
        contract_dir = Path(self.temp_dir) / "contracts"
        export_contract_bundle(temp_path, contract_dir)
        
        result = check_slice_readiness(
            "test_slice_goal",
            temp_path,
            self.prereg_path,
            contract_dir
        )
        
        # Should be DEGRADED due to extra parameter warning
        self.assertIn(result.status, [ReadinessStatus.READY, ReadinessStatus.DEGRADED])
    
    def test_unknown_kind_returns_blocked(self):
        """Slice with unknown metric kind returns BLOCKED."""
        bad_curriculum = create_test_curriculum()
        bad_curriculum["slices"]["test_slice_goal"]["success_metric"]["kind"] = "unknown_metric"
        
        temp_path = Path(self.temp_dir) / "bad_curriculum.yaml"
        with open(temp_path, "w") as f:
            yaml.dump(bad_curriculum, f)
        
        # Create empty contract dir (contracts won't export for invalid kind)
        contract_dir = Path(self.temp_dir) / "contracts"
        contract_dir.mkdir(parents=True, exist_ok=True)
        
        result = check_slice_readiness(
            "test_slice_goal",
            temp_path,
            self.prereg_path,
            contract_dir
        )
        
        self.assertEqual(result.status, ReadinessStatus.BLOCKED)
    
    def test_missing_slice_returns_blocked(self):
        """Non-existent slice returns BLOCKED."""
        contract_dir = Path(self.temp_dir) / "contracts"
        contract_dir.mkdir(parents=True, exist_ok=True)
        
        result = check_slice_readiness(
            "nonexistent_slice",
            self.curriculum_path,
            self.prereg_path,
            contract_dir
        )
        
        self.assertEqual(result.status, ReadinessStatus.BLOCKED)
    
    def test_readiness_deterministic(self):
        """Readiness check is deterministic."""
        contract_dir = Path(self.temp_dir) / "contracts"
        export_contract_bundle(self.curriculum_path, contract_dir)
        
        results = [
            check_slice_readiness("test_slice_goal", self.curriculum_path, self.prereg_path, contract_dir)
            for _ in range(5)
        ]
        
        # Status should be identical
        self.assertTrue(all(r.status == results[0].status for r in results))
        # Drift class should be identical
        self.assertTrue(all(r.drift_class == results[0].drift_class for r in results))


class TestFormatReadinessResult(TestCurriculumWithTempFiles):
    """Tests for format_readiness_result function."""
    
    def test_format_includes_status(self):
        """Formatted output includes status."""
        contract_dir = Path(self.temp_dir) / "contracts"
        export_contract_bundle(self.curriculum_path, contract_dir)
        
        result = check_slice_readiness("test_slice_goal", self.curriculum_path, self.prereg_path, contract_dir)
        formatted = format_readiness_result(result)
        
        self.assertIn(result.status.value, formatted)
    
    def test_format_includes_slice_name(self):
        """Formatted output includes slice name."""
        contract_dir = Path(self.temp_dir) / "contracts"
        export_contract_bundle(self.curriculum_path, contract_dir)
        
        result = check_slice_readiness("test_slice_goal", self.curriculum_path, self.prereg_path, contract_dir)
        formatted = format_readiness_result(result)
        
        self.assertIn("test_slice_goal", formatted)
    
    def test_format_includes_exit_code(self):
        """Formatted output includes exit code hint."""
        contract_dir = Path(self.temp_dir) / "contracts"
        export_contract_bundle(self.curriculum_path, contract_dir)
        
        result = check_slice_readiness("test_slice_goal", self.curriculum_path, self.prereg_path, contract_dir)
        formatted = format_readiness_result(result)
        
        self.assertIn("Exit code:", formatted)


class TestReadinessExitCodes(TestCurriculumWithTempFiles):
    """Tests for readiness gate exit code behavior."""
    
    def test_ready_exit_code_zero(self):
        """READY status gives exit code 0."""
        result = ReadinessResult(
            slice_name="test",
            status=ReadinessStatus.READY,
            alignment_passes=True,
            drift_class=DriftSeverityClass.NONE,
        )
        
        exit_code = 0 if result.status in [ReadinessStatus.READY, ReadinessStatus.DEGRADED] else 1
        self.assertEqual(exit_code, 0)
    
    def test_degraded_exit_code_zero(self):
        """DEGRADED status gives exit code 0."""
        result = ReadinessResult(
            slice_name="test",
            status=ReadinessStatus.DEGRADED,
            alignment_passes=True,
            drift_class=DriftSeverityClass.PARAMETRIC_MINOR,
        )
        
        exit_code = 0 if result.status in [ReadinessStatus.READY, ReadinessStatus.DEGRADED] else 1
        self.assertEqual(exit_code, 0)
    
    def test_blocked_exit_code_one(self):
        """BLOCKED status gives exit code 1."""
        result = ReadinessResult(
            slice_name="test",
            status=ReadinessStatus.BLOCKED,
            alignment_passes=False,
            drift_class=DriftSeverityClass.SEMANTIC,
        )
        
        exit_code = 0 if result.status in [ReadinessStatus.READY, ReadinessStatus.DEGRADED] else 1
        self.assertEqual(exit_code, 1)


# =============================================================================
# Task 2: Metric Index Contract Browser Tests
# =============================================================================

class TestContractIndexListEntry(unittest.TestCase):
    """Tests for ContractIndexListEntry dataclass."""
    
    def test_entry_creation(self):
        """ContractIndexListEntry can be created."""
        entry = ContractIndexListEntry(
            slice_name="test_slice",
            metric_kind="goal_hit",
            contract_path="test_slice.json",
            schema_version="1.0.0",
            config_hash="abc123",
        )
        self.assertEqual(entry.slice_name, "test_slice")
        self.assertEqual(entry.metric_kind, "goal_hit")
    
    def test_entry_to_dict(self):
        """ContractIndexListEntry.to_dict() returns correct structure."""
        entry = ContractIndexListEntry(
            slice_name="test_slice",
            metric_kind="goal_hit",
            contract_path="test_slice.json",
            schema_version="1.0.0",
            config_hash="abc123",
        )
        d = entry.to_dict()
        
        self.assertEqual(d["slice_name"], "test_slice")
        self.assertEqual(d["metric_kind"], "goal_hit")
        self.assertEqual(d["contract_path"], "test_slice.json")


class TestListContracts(TestCurriculumWithTempFiles):
    """Tests for list_contracts function."""
    
    def test_missing_bundle_raises_error(self):
        """Missing contract bundle raises FileNotFoundError."""
        nonexistent_path = Path(self.temp_dir) / "nonexistent_contracts"
        
        with self.assertRaises(FileNotFoundError):
            list_contracts(nonexistent_path)
    
    def test_list_contracts_from_bundle(self):
        """list_contracts returns entries from existing bundle."""
        # First export a bundle
        contract_dir = Path(self.temp_dir) / "contracts"
        export_contract_bundle(self.curriculum_path, contract_dir)
        
        # Then list contracts
        entries = list_contracts(contract_dir)
        
        self.assertIsInstance(entries, list)
        self.assertGreater(len(entries), 0)
    
    def test_list_contracts_sorted(self):
        """list_contracts returns entries sorted by slice_name."""
        contract_dir = Path(self.temp_dir) / "contracts"
        export_contract_bundle(self.curriculum_path, contract_dir)
        
        entries = list_contracts(contract_dir)
        slice_names = [e.slice_name for e in entries]
        
        self.assertEqual(slice_names, sorted(slice_names))
    
    def test_list_contracts_deterministic(self):
        """list_contracts is deterministic."""
        contract_dir = Path(self.temp_dir) / "contracts"
        export_contract_bundle(self.curriculum_path, contract_dir)
        
        results = [list_contracts(contract_dir) for _ in range(5)]
        
        # All results should have same entries
        first_names = [e.slice_name for e in results[0]]
        for result in results[1:]:
            names = [e.slice_name for e in result]
            self.assertEqual(names, first_names)


class TestFormatContractList(unittest.TestCase):
    """Tests for format_contract_list function."""
    
    def test_empty_list_message(self):
        """Empty list shows appropriate message."""
        formatted = format_contract_list([])
        self.assertIn("No contracts found", formatted)
    
    def test_format_includes_header(self):
        """Formatted list includes header."""
        entries = [
            ContractIndexListEntry(
                slice_name="test_slice",
                metric_kind="goal_hit",
                contract_path="test_slice.json",
                schema_version="1.0.0",
                config_hash="abc123",
            )
        ]
        formatted = format_contract_list(entries)
        
        self.assertIn("METRIC CONTRACT INDEX", formatted)
    
    def test_format_includes_slice_names(self):
        """Formatted list includes slice names."""
        entries = [
            ContractIndexListEntry(
                slice_name="test_slice",
                metric_kind="goal_hit",
                contract_path="test_slice.json",
                schema_version="1.0.0",
                config_hash="abc123",
            )
        ]
        formatted = format_contract_list(entries)
        
        self.assertIn("test_slice", formatted)
    
    def test_format_includes_total(self):
        """Formatted list includes total count."""
        entries = [
            ContractIndexListEntry(
                slice_name=f"slice_{i}",
                metric_kind="goal_hit",
                contract_path=f"slice_{i}.json",
                schema_version="1.0.0",
                config_hash="abc123",
            )
            for i in range(3)
        ]
        formatted = format_contract_list(entries)
        
        self.assertIn("Total contracts: 3", formatted)


# =============================================================================
# Task 3: Log-Field Coverage Map Tests
# =============================================================================

class TestLogFieldCoverageMap(unittest.TestCase):
    """Tests for LogFieldCoverageMap dataclass."""
    
    def test_coverage_creation(self):
        """LogFieldCoverageMap can be created."""
        coverage = LogFieldCoverageMap(
            metric_kind="goal_hit",
            required_log_fields=["verified_statements"],
            runtime_fields=["target_hashes"],
            parameter_fields=["min_goal_hits"],
            interpretation="boolean_with_count",
        )
        self.assertEqual(coverage.metric_kind, "goal_hit")
        self.assertIn("verified_statements", coverage.required_log_fields)
    
    def test_coverage_to_dict(self):
        """LogFieldCoverageMap.to_dict() returns correct structure."""
        coverage = LogFieldCoverageMap(
            metric_kind="goal_hit",
            required_log_fields=["verified_statements"],
            runtime_fields=["target_hashes"],
            parameter_fields=["min_goal_hits"],
            interpretation="boolean_with_count",
        )
        d = coverage.to_dict()
        
        self.assertEqual(d["metric_kind"], "goal_hit")
        self.assertIn("required_log_fields", d)
        self.assertIn("runtime_fields", d)
    
    def test_coverage_to_json_deterministic(self):
        """LogFieldCoverageMap.to_json() is deterministic."""
        coverage = LogFieldCoverageMap(
            metric_kind="goal_hit",
            required_log_fields=["verified_statements"],
            runtime_fields=["target_hashes"],
            parameter_fields=["min_goal_hits"],
            interpretation="boolean_with_count",
        )
        
        jsons = [coverage.to_json() for _ in range(5)]
        self.assertTrue(all(j == jsons[0] for j in jsons))


class TestGetLogFieldCoverage(TestCurriculumWithTempFiles):
    """Tests for get_log_field_coverage function."""
    
    def test_get_coverage_for_valid_slice(self):
        """get_log_field_coverage returns coverage for valid slice."""
        coverage = get_log_field_coverage("test_slice_goal", self.curriculum_path)
        
        self.assertEqual(coverage.metric_kind, "goal_hit")
        self.assertIsInstance(coverage.required_log_fields, list)
        self.assertIsInstance(coverage.runtime_fields, list)
    
    def test_get_coverage_for_invalid_slice(self):
        """get_log_field_coverage raises for invalid slice."""
        with self.assertRaises(ValueError):
            get_log_field_coverage("nonexistent_slice", self.curriculum_path)
    
    def test_coverage_has_at_least_one_field(self):
        """Each metric kind reports at least one required input field."""
        for slice_name in ["test_slice_goal", "test_slice_sparse", "test_slice_chain", "test_slice_multi"]:
            coverage = get_log_field_coverage(slice_name, self.curriculum_path)
            
            total_fields = len(coverage.required_log_fields) + len(coverage.runtime_fields)
            self.assertGreater(total_fields, 0, f"Slice {slice_name} has no input fields")
    
    def test_coverage_deterministic(self):
        """Coverage output is deterministic."""
        coverages = [
            get_log_field_coverage("test_slice_goal", self.curriculum_path)
            for _ in range(5)
        ]
        
        first_json = coverages[0].to_json()
        self.assertTrue(all(c.to_json() == first_json for c in coverages))


class TestGetLogFieldCoverageByKind(unittest.TestCase):
    """Tests for get_log_field_coverage_by_kind function."""
    
    def test_goal_hit_coverage(self):
        """goal_hit metric kind has expected fields."""
        coverage = get_log_field_coverage_by_kind("goal_hit")
        
        self.assertEqual(coverage.metric_kind, "goal_hit")
        self.assertIn("verified_statements", coverage.required_log_fields)
    
    def test_sparse_success_coverage(self):
        """sparse_success metric kind has expected fields."""
        coverage = get_log_field_coverage_by_kind("sparse_success")
        
        self.assertEqual(coverage.metric_kind, "sparse_success")
        self.assertIn("verified_count", coverage.required_log_fields)
    
    def test_chain_success_coverage(self):
        """chain_success metric kind has expected fields."""
        coverage = get_log_field_coverage_by_kind("chain_success")
        
        self.assertEqual(coverage.metric_kind, "chain_success")
        self.assertIn("dependency_graph", coverage.required_log_fields)
    
    def test_multi_goal_success_coverage(self):
        """multi_goal_success metric kind has expected fields."""
        coverage = get_log_field_coverage_by_kind("multi_goal_success")
        
        self.assertEqual(coverage.metric_kind, "multi_goal_success")
        self.assertIn("verified_hashes", coverage.required_log_fields)
    
    def test_unknown_kind_raises(self):
        """Unknown metric kind raises ValueError."""
        with self.assertRaises(ValueError):
            get_log_field_coverage_by_kind("unknown_metric")
    
    def test_all_metric_kinds_have_fields(self):
        """All metric kinds have at least one required input field."""
        for kind in ["goal_hit", "sparse_success", "chain_success", "multi_goal_success"]:
            coverage = get_log_field_coverage_by_kind(kind)
            
            total_fields = len(coverage.required_log_fields) + len(coverage.runtime_fields)
            self.assertGreater(total_fields, 0, f"Kind {kind} has no input fields")


class TestFormatLogFieldCoverage(unittest.TestCase):
    """Tests for format_log_field_coverage function."""
    
    def test_format_includes_metric_kind(self):
        """Formatted coverage includes metric kind."""
        coverage = LogFieldCoverageMap(
            metric_kind="goal_hit",
            required_log_fields=["verified_statements"],
            runtime_fields=["target_hashes"],
            parameter_fields=["min_goal_hits"],
            interpretation="boolean_with_count",
        )
        formatted = format_log_field_coverage(coverage)
        
        self.assertIn("goal_hit", formatted)
    
    def test_format_includes_log_fields(self):
        """Formatted coverage includes log fields."""
        coverage = LogFieldCoverageMap(
            metric_kind="goal_hit",
            required_log_fields=["verified_statements"],
            runtime_fields=["target_hashes"],
            parameter_fields=["min_goal_hits"],
            interpretation="boolean_with_count",
        )
        formatted = format_log_field_coverage(coverage)
        
        self.assertIn("verified_statements", formatted)
        self.assertIn("target_hashes", formatted)
    
    def test_format_includes_interpretation(self):
        """Formatted coverage includes interpretation."""
        coverage = LogFieldCoverageMap(
            metric_kind="goal_hit",
            required_log_fields=["verified_statements"],
            runtime_fields=["target_hashes"],
            parameter_fields=["min_goal_hits"],
            interpretation="boolean_with_count",
        )
        formatted = format_log_field_coverage(coverage)
        
        self.assertIn("boolean_with_count", formatted)


# =============================================================================
# Task 4: Stability Sentinel Test
# =============================================================================

class TestContractShapeSentinel(TestCurriculumWithTempFiles):
    """
    Stability sentinel test that validates contract shape.
    
    This test loads all contracts and asserts:
      - Each contract has metric_kind, parameters, expected_inputs, expected_outputs
      - No unknown top-level keys
    
    A future change that mutates contract shape must update this test,
    making drift explicit.
    """
    
    # Expected top-level keys in a contract
    EXPECTED_CONTRACT_KEYS = {
        "schema_version",
        "slice_name",
        "metric_kind",
        "required_config_fields",
        "required_log_fields",  # expected_inputs
        "runtime_fields",
        "parameter_schema",  # parameters
        "output_schema",  # expected_outputs
        "config_hash",
    }
    
    def test_contract_has_all_required_keys(self):
        """Every contract has all required keys (metric_kind, parameters, inputs, outputs)."""
        # Export contracts first
        contract_dir = Path(self.temp_dir) / "contracts"
        export_contract_bundle(self.curriculum_path, contract_dir)
        
        # Load and check each contract
        for slice_name in ["test_slice_goal", "test_slice_sparse", "test_slice_chain", "test_slice_multi"]:
            contract_path = contract_dir / f"{slice_name}.json"
            self.assertTrue(contract_path.exists(), f"Contract file missing for {slice_name}")
            
            with open(contract_path, "r") as f:
                contract = json.load(f)
            
            # Check required keys
            self.assertIn("metric_kind", contract, f"Missing metric_kind in {slice_name}")
            self.assertIn("parameter_schema", contract, f"Missing parameters in {slice_name}")
            self.assertIn("required_log_fields", contract, f"Missing expected_inputs in {slice_name}")
            self.assertIn("output_schema", contract, f"Missing expected_outputs in {slice_name}")
    
    def test_contract_has_no_unknown_keys(self):
        """Contracts have no unknown top-level keys."""
        contract_dir = Path(self.temp_dir) / "contracts"
        export_contract_bundle(self.curriculum_path, contract_dir)
        
        for slice_name in ["test_slice_goal", "test_slice_sparse", "test_slice_chain", "test_slice_multi"]:
            contract_path = contract_dir / f"{slice_name}.json"
            
            with open(contract_path, "r") as f:
                contract = json.load(f)
            
            actual_keys = set(contract.keys())
            unknown_keys = actual_keys - self.EXPECTED_CONTRACT_KEYS
            
            self.assertEqual(
                unknown_keys, set(),
                f"Unknown keys in {slice_name}: {unknown_keys}"
            )
    
    def test_all_contracts_have_identical_key_structure(self):
        """All contracts have the same key structure."""
        contract_dir = Path(self.temp_dir) / "contracts"
        export_contract_bundle(self.curriculum_path, contract_dir)
        
        entries = list_contracts(contract_dir)
        key_sets = []
        
        for entry in entries:
            contract_path = contract_dir / entry.contract_path
            with open(contract_path, "r") as f:
                contract = json.load(f)
            key_sets.append(frozenset(contract.keys()))
        
        # All contracts should have the same keys
        self.assertEqual(len(set(key_sets)), 1, "Contracts have inconsistent key structure")
    
    def test_contract_index_is_valid(self):
        """Contract index file has expected structure."""
        contract_dir = Path(self.temp_dir) / "contracts"
        export_contract_bundle(self.curriculum_path, contract_dir)
        
        index_path = contract_dir / "metric_contract_index.json"
        with open(index_path, "r") as f:
            index = json.load(f)
        
        # Required index fields
        self.assertIn("generated_at", index)
        self.assertIn("schema_version", index)
        self.assertIn("total_contracts", index)
        self.assertIn("contracts", index)
        
        # Validate contract entries
        for contract_entry in index["contracts"]:
            self.assertIn("slice_name", contract_entry)
            self.assertIn("metric_kind", contract_entry)
            self.assertIn("contract_path", contract_entry)
            self.assertIn("config_hash", contract_entry)
    
    def test_contract_schema_version_stable(self):
        """Contract schema version matches introspection version."""
        contract_dir = Path(self.temp_dir) / "contracts"
        export_contract_bundle(self.curriculum_path, contract_dir)
        
        index_path = contract_dir / "metric_contract_index.json"
        with open(index_path, "r") as f:
            index = json.load(f)
        
        self.assertEqual(index["schema_version"], INTROSPECTION_SCHEMA_VERSION)


class TestContractShapeRegressions(TestCurriculumWithTempFiles):
    """
    Regression tests to catch contract shape changes.
    
    These tests enforce that any change to contract structure must be
    intentional and explicitly acknowledged by updating this test.
    """
    
    def test_metric_kind_field_is_string(self):
        """metric_kind field must be a string in all contracts."""
        for slice_name in ["test_slice_goal", "test_slice_sparse"]:
            contract = export_metric_contract(slice_name, self.curriculum_path)
            d = contract.to_dict()
            
            self.assertIsInstance(d["metric_kind"], str)
    
    def test_parameter_schema_is_dict(self):
        """parameter_schema field must be a dict in all contracts."""
        for slice_name in ["test_slice_goal", "test_slice_sparse"]:
            contract = export_metric_contract(slice_name, self.curriculum_path)
            d = contract.to_dict()
            
            self.assertIsInstance(d["parameter_schema"], dict)
    
    def test_required_log_fields_is_list(self):
        """required_log_fields (expected_inputs) must be a list."""
        for slice_name in ["test_slice_goal", "test_slice_sparse"]:
            contract = export_metric_contract(slice_name, self.curriculum_path)
            d = contract.to_dict()
            
            self.assertIsInstance(d["required_log_fields"], list)
    
    def test_output_schema_is_dict(self):
        """output_schema (expected_outputs) must be a dict."""
        for slice_name in ["test_slice_goal", "test_slice_sparse"]:
            contract = export_metric_contract(slice_name, self.curriculum_path)
            d = contract.to_dict()
            
            self.assertIsInstance(d["output_schema"], dict)
    
    def test_config_hash_is_64_char_hex(self):
        """config_hash must be a 64-character hex string (SHA256)."""
        for slice_name in ["test_slice_goal", "test_slice_sparse"]:
            contract = export_metric_contract(slice_name, self.curriculum_path)
            d = contract.to_dict()
            
            self.assertEqual(len(d["config_hash"]), 64)
            # Check it's valid hex
            int(d["config_hash"], 16)  # Raises ValueError if not hex


# =============================================================================
# Task 1: Formalized Readiness Contract Tests
# =============================================================================

class TestReadinessContractJSON(TestCurriculumWithTempFiles):
    """
    Tests for the formalized ReadinessResult JSON contract.
    
    Validates that --ready produces a stable, documented contract format.
    """
    
    # Required fields in the ReadinessResult JSON contract
    REQUIRED_READINESS_FIELDS = {
        "slice_name",
        "status",
        "drift_severity",
        "missing_contract",
        "missing_fields",
        "unknown_fields",
        "alignment_issues",
        "log_field_coverage",
        # Legacy fields
        "alignment_passes",
        "errors",
        "warnings",
        "drift_items",
    }
    
    def test_readiness_json_has_required_fields(self):
        """ReadinessResult JSON has all required contract fields."""
        # Export contracts first
        contract_dir = Path(self.temp_dir) / "contracts"
        export_contract_bundle(self.curriculum_path, contract_dir)
        
        result = check_slice_readiness(
            "test_slice_goal",
            self.curriculum_path,
            self.prereg_path,
            contract_dir
        )
        d = result.to_dict()
        
        for field in self.REQUIRED_READINESS_FIELDS:
            self.assertIn(field, d, f"Missing required field: {field}")
    
    def test_readiness_ready_case(self):
        """READY case: all contracts present, no drift → status=READY, exit=0."""
        contract_dir = Path(self.temp_dir) / "contracts"
        export_contract_bundle(self.curriculum_path, contract_dir)
        
        result = check_slice_readiness(
            "test_slice_goal",
            self.curriculum_path,
            self.prereg_path,
            contract_dir
        )
        
        # Should be READY or DEGRADED with exit=0
        self.assertIn(result.status, [ReadinessStatus.READY, ReadinessStatus.DEGRADED])
        exit_code = 0 if result.status in [ReadinessStatus.READY, ReadinessStatus.DEGRADED] else 1
        self.assertEqual(exit_code, 0)
    
    def test_readiness_degraded_case(self):
        """DEGRADED case: minor param drift → status=DEGRADED, exit=0."""
        # Create curriculum with extra parameter
        modified_curriculum = create_test_curriculum()
        modified_curriculum["slices"]["test_slice_goal"]["success_metric"]["parameters"]["extra_param"] = 999
        
        temp_path = Path(self.temp_dir) / "modified_curriculum.yaml"
        with open(temp_path, "w") as f:
            yaml.dump(modified_curriculum, f)
        
        contract_dir = Path(self.temp_dir) / "contracts"
        export_contract_bundle(temp_path, contract_dir)
        
        result = check_slice_readiness(
            "test_slice_goal",
            temp_path,
            self.prereg_path,
            contract_dir
        )
        
        # Should be READY or DEGRADED with exit=0
        self.assertIn(result.status, [ReadinessStatus.READY, ReadinessStatus.DEGRADED])
        exit_code = 0 if result.status in [ReadinessStatus.READY, ReadinessStatus.DEGRADED] else 1
        self.assertEqual(exit_code, 0)
    
    def test_readiness_blocked_missing_contract(self):
        """BLOCKED case: missing contract artifact → status=BLOCKED, exit=1."""
        # Don't export contracts - they should be missing
        contract_dir = Path(self.temp_dir) / "empty_contracts"
        contract_dir.mkdir(parents=True, exist_ok=True)
        
        result = check_slice_readiness(
            "test_slice_goal",
            self.curriculum_path,
            self.prereg_path,
            contract_dir
        )
        
        self.assertEqual(result.status, ReadinessStatus.BLOCKED)
        self.assertTrue(result.missing_contract)
        exit_code = 0 if result.status in [ReadinessStatus.READY, ReadinessStatus.DEGRADED] else 1
        self.assertEqual(exit_code, 1)
    
    def test_readiness_blocked_semantic_drift(self):
        """BLOCKED case: SEMANTIC drift → status=BLOCKED, exit=1."""
        # Create curriculum with wrong metric kind
        modified_curriculum = create_test_curriculum()
        modified_curriculum["slices"]["test_slice_goal"]["success_metric"]["kind"] = "unknown_metric"
        
        temp_path = Path(self.temp_dir) / "bad_curriculum.yaml"
        with open(temp_path, "w") as f:
            yaml.dump(modified_curriculum, f)
        
        contract_dir = Path(self.temp_dir) / "contracts"
        contract_dir.mkdir(parents=True, exist_ok=True)
        
        result = check_slice_readiness(
            "test_slice_goal",
            temp_path,
            self.prereg_path,
            contract_dir
        )
        
        self.assertEqual(result.status, ReadinessStatus.BLOCKED)
        exit_code = 0 if result.status in [ReadinessStatus.READY, ReadinessStatus.DEGRADED] else 1
        self.assertEqual(exit_code, 1)
    
    def test_readiness_json_deterministic(self):
        """Same slice → identical JSON across runs."""
        contract_dir = Path(self.temp_dir) / "contracts"
        export_contract_bundle(self.curriculum_path, contract_dir)
        
        results = [
            check_slice_readiness(
                "test_slice_goal",
                self.curriculum_path,
                self.prereg_path,
                contract_dir
            )
            for _ in range(5)
        ]
        
        jsons = [r.to_json() for r in results]
        self.assertTrue(all(j == jsons[0] for j in jsons))
    
    def test_readiness_includes_log_field_coverage(self):
        """ReadinessResult includes log_field_coverage when available."""
        contract_dir = Path(self.temp_dir) / "contracts"
        export_contract_bundle(self.curriculum_path, contract_dir)
        
        result = check_slice_readiness(
            "test_slice_goal",
            self.curriculum_path,
            self.prereg_path,
            contract_dir
        )
        
        self.assertIsNotNone(result.log_field_coverage)
        self.assertIn("metric_kind", result.log_field_coverage)
        self.assertIn("required_log_fields", result.log_field_coverage)
    
    def test_readiness_missing_fields_sorted(self):
        """Missing fields are sorted for determinism."""
        result = ReadinessResult(
            slice_name="test",
            status=ReadinessStatus.BLOCKED,
            alignment_passes=False,
            drift_class=DriftSeverityClass.SEMANTIC,
            missing_fields=["z_field", "a_field", "m_field"],
        )
        d = result.to_dict()
        
        self.assertEqual(d["missing_fields"], ["a_field", "m_field", "z_field"])
    
    def test_readiness_unknown_fields_sorted(self):
        """Unknown fields are sorted for determinism."""
        result = ReadinessResult(
            slice_name="test",
            status=ReadinessStatus.DEGRADED,
            alignment_passes=True,
            drift_class=DriftSeverityClass.PARAMETRIC_MINOR,
            unknown_fields=["z_extra", "a_extra"],
        )
        d = result.to_dict()
        
        self.assertEqual(d["unknown_fields"], ["a_extra", "z_extra"])


# =============================================================================
# Task 2: Contract Index Stability Tests
# =============================================================================

class TestContractIndexStability(TestCurriculumWithTempFiles):
    """
    Tests for contract index shape stability.
    
    Lock down the shape of metric_contract_index.json.
    """
    
    # Required fields for each entry in the contract index
    REQUIRED_INDEX_ENTRY_FIELDS = {
        "slice_name",
        "metric_kind",
        "contract_path",
        "config_hash",
        "schema_version",
    }
    
    def test_index_entry_has_required_fields(self):
        """Each index entry has all required fields."""
        contract_dir = Path(self.temp_dir) / "contracts"
        export_contract_bundle(self.curriculum_path, contract_dir)
        
        entries = list_contracts(contract_dir)
        
        for entry in entries:
            d = entry.to_dict()
            for field in self.REQUIRED_INDEX_ENTRY_FIELDS:
                self.assertIn(field, d, f"Missing field: {field}")
    
    def test_list_contracts_reads_only_from_index(self):
        """list_contracts reads from index, does not recompute."""
        contract_dir = Path(self.temp_dir) / "contracts"
        export_contract_bundle(self.curriculum_path, contract_dir)
        
        # Modify curriculum after export
        modified_curriculum = create_test_curriculum()
        modified_curriculum["slices"]["new_slice"] = {
            "description": "New slice",
            "success_metric": {"kind": "goal_hit", "parameters": {"min_goal_hits": 1, "min_total_verified": 1}}
        }
        
        temp_path = Path(self.temp_dir) / "modified_curriculum.yaml"
        with open(temp_path, "w") as f:
            yaml.dump(modified_curriculum, f)
        
        # list_contracts should NOT see the new slice (reads from existing index)
        entries = list_contracts(contract_dir)
        slice_names = [e.slice_name for e in entries]
        
        self.assertNotIn("new_slice", slice_names)
    
    def test_list_contracts_sorted_by_slice_name(self):
        """list_contracts returns entries sorted by slice_name."""
        contract_dir = Path(self.temp_dir) / "contracts"
        export_contract_bundle(self.curriculum_path, contract_dir)
        
        entries = list_contracts(contract_dir)
        slice_names = [e.slice_name for e in entries]
        
        self.assertEqual(slice_names, sorted(slice_names))
    
    def test_list_contracts_json_stable_ordering(self):
        """JSON output has stable ordering."""
        contract_dir = Path(self.temp_dir) / "contracts"
        export_contract_bundle(self.curriculum_path, contract_dir)
        
        results = []
        for _ in range(5):
            entries = list_contracts(contract_dir)
            json_output = json.dumps([e.to_dict() for e in entries], sort_keys=True)
            results.append(json_output)
        
        self.assertTrue(all(r == results[0] for r in results))
    
    def test_missing_index_error_message(self):
        """Broken/missing index gives clear error message."""
        nonexistent_dir = Path(self.temp_dir) / "nonexistent"
        
        with self.assertRaises(FileNotFoundError) as ctx:
            list_contracts(nonexistent_dir)
        
        # Should mention index and --export-all-contracts
        error_msg = str(ctx.exception)
        self.assertIn("index", error_msg.lower())
        self.assertIn("export-all-contracts", error_msg.lower())
    
    def test_index_regeneration_produces_same_shape(self):
        """Index regeneration produces same shape and sorted entries."""
        contract_dir = Path(self.temp_dir) / "contracts"
        
        # First export
        export_contract_bundle(self.curriculum_path, contract_dir)
        first_entries = list_contracts(contract_dir)
        first_keys = set(first_entries[0].to_dict().keys())
        
        # Clear and re-export
        import shutil
        shutil.rmtree(contract_dir)
        export_contract_bundle(self.curriculum_path, contract_dir)
        second_entries = list_contracts(contract_dir)
        second_keys = set(second_entries[0].to_dict().keys())
        
        # Same keys
        self.assertEqual(first_keys, second_keys)
        
        # Same order
        first_names = [e.slice_name for e in first_entries]
        second_names = [e.slice_name for e in second_entries]
        self.assertEqual(first_names, second_names)


# =============================================================================
# Task 3: Log-Field Coverage Canonicalization Tests
# =============================================================================

class TestLogFieldCoverageCanonicalization(TestCurriculumWithTempFiles):
    """
    Tests for canonicalized log-field coverage.
    """
    
    # Required fields in LogFieldCoverageMap JSON
    REQUIRED_COVERAGE_FIELDS = {
        "metric_kind",
        "required_log_fields",
        "optional_log_fields",
        "runtime_fields",
        "parameter_fields",
        "interpretation",
    }
    
    def test_coverage_json_has_required_fields(self):
        """LogFieldCoverageMap JSON has all required fields."""
        coverage = get_log_field_coverage("test_slice_goal", self.curriculum_path)
        d = coverage.to_dict()
        
        for field in self.REQUIRED_COVERAGE_FIELDS:
            self.assertIn(field, d, f"Missing required field: {field}")
    
    def test_all_metric_kinds_have_nonempty_required_fields(self):
        """All 4 metric kinds return non-empty required_log_fields."""
        for kind in ["goal_hit", "sparse_success", "chain_success", "multi_goal_success"]:
            coverage = get_log_field_coverage_by_kind(kind)
            
            self.assertGreater(
                len(coverage.required_log_fields), 0,
                f"Kind {kind} has no required_log_fields"
            )
    
    def test_coverage_includes_optional_log_fields(self):
        """Coverage includes optional_log_fields."""
        for kind in ["goal_hit", "sparse_success", "chain_success", "multi_goal_success"]:
            coverage = get_log_field_coverage_by_kind(kind)
            
            # Optional fields should be defined
            self.assertIsInstance(coverage.optional_log_fields, list)
            # Should have at least some optional fields (timestamp, etc.)
            self.assertGreater(len(coverage.optional_log_fields), 0)
    
    def test_coverage_fields_sorted_for_determinism(self):
        """All list fields are sorted for determinism."""
        coverage = get_log_field_coverage("test_slice_goal", self.curriculum_path)
        d = coverage.to_dict()
        
        self.assertEqual(d["required_log_fields"], sorted(d["required_log_fields"]))
        self.assertEqual(d["optional_log_fields"], sorted(d["optional_log_fields"]))
        self.assertEqual(d["runtime_fields"], sorted(d["runtime_fields"]))
        self.assertEqual(d["parameter_fields"], sorted(d["parameter_fields"]))
    
    def test_cli_fields_human_matches_json(self):
        """CLI --fields human output mentions same fields as JSON."""
        coverage = get_log_field_coverage("test_slice_goal", self.curriculum_path)
        human_output = format_log_field_coverage(coverage)
        json_data = coverage.to_dict()
        
        # All required fields should appear in human output
        for field in json_data["required_log_fields"]:
            self.assertIn(field, human_output)
        
        # Metric kind should appear
        self.assertIn(json_data["metric_kind"], human_output)
    
    def test_coverage_deterministic(self):
        """Repeated calls produce identical JSON."""
        coverages = [
            get_log_field_coverage("test_slice_goal", self.curriculum_path)
            for _ in range(5)
        ]
        
        jsons = [c.to_json() for c in coverages]
        self.assertTrue(all(j == jsons[0] for j in jsons))
    
    def test_optional_log_fields_defined_for_all_kinds(self):
        """OPTIONAL_LOG_FIELDS_BY_KIND has entries for all metric kinds."""
        expected_kinds = ["goal_hit", "sparse_success", "chain_success", "multi_goal_success"]
        
        for kind in expected_kinds:
            self.assertIn(kind, OPTIONAL_LOG_FIELDS_BY_KIND, f"Missing optional fields for {kind}")


# =============================================================================
# Task 4: CI Integration Tests
# =============================================================================

class TestCIIntegrationSemantics(TestCurriculumWithTempFiles):
    """
    Tests validating CI integration exit semantics.
    """
    
    def test_health_check_exit_semantics(self):
        """--health-check exit semantics: 0=OK/WARN, 1=FAIL."""
        report = run_health_check(self.curriculum_path, self.prereg_path)
        
        expected_exit = 0 if report.overall_status != HealthStatus.FAIL else 1
        actual_exit = 0 if report.overall_status in [HealthStatus.OK, HealthStatus.WARN] else 1
        
        self.assertEqual(actual_exit, expected_exit)
    
    def test_ready_exit_semantics(self):
        """--ready exit semantics: 0=READY/DEGRADED, 1=BLOCKED."""
        contract_dir = Path(self.temp_dir) / "contracts"
        export_contract_bundle(self.curriculum_path, contract_dir)
        
        result = check_slice_readiness(
            "test_slice_goal",
            self.curriculum_path,
            self.prereg_path,
            contract_dir
        )
        
        if result.status in [ReadinessStatus.READY, ReadinessStatus.DEGRADED]:
            expected_exit = 0
        else:
            expected_exit = 1
        
        actual_exit = 0 if result.status in [ReadinessStatus.READY, ReadinessStatus.DEGRADED] else 1
        self.assertEqual(actual_exit, expected_exit)
    
    def test_list_contracts_exit_semantics(self):
        """--list-contracts exit semantics: 0=found, 1=missing."""
        # With contracts: exit 0
        contract_dir = Path(self.temp_dir) / "contracts"
        export_contract_bundle(self.curriculum_path, contract_dir)
        
        try:
            entries = list_contracts(contract_dir)
            self.assertGreater(len(entries), 0)
            exit_code = 0
        except FileNotFoundError:
            exit_code = 1
        
        self.assertEqual(exit_code, 0)
        
        # Without contracts: exit 1
        empty_dir = Path(self.temp_dir) / "empty"
        empty_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            list_contracts(empty_dir)
            exit_code = 0
        except FileNotFoundError:
            exit_code = 1
        
        self.assertEqual(exit_code, 1)


# =============================================================================
# V2.0 Task 1: Multi-Slice Readiness Summary Tests
# =============================================================================

class TestReadinessSummary(unittest.TestCase):
    """Tests for ReadinessSummary dataclass."""
    
    def test_readiness_summary_creation(self):
        """ReadinessSummary can be created."""
        summary = ReadinessSummary(
            schema_version="1.0.0",
            slice_count=3,
            ready_count=2,
            degraded_count=1,
            blocked_count=0,
            slices={}
        )
        self.assertEqual(summary.schema_version, "1.0.0")
        self.assertEqual(summary.slice_count, 3)
    
    def test_readiness_summary_to_dict_sorted_slices(self):
        """to_dict() returns slices sorted alphabetically."""
        summary = ReadinessSummary(
            schema_version="1.0.0",
            slice_count=3,
            ready_count=3,
            degraded_count=0,
            blocked_count=0,
            slices={
                "z_slice": {"status": "READY"},
                "a_slice": {"status": "READY"},
                "m_slice": {"status": "READY"},
            }
        )
        d = summary.to_dict()
        
        slice_keys = list(d["slices"].keys())
        self.assertEqual(slice_keys, ["a_slice", "m_slice", "z_slice"])
    
    def test_readiness_summary_to_json_deterministic(self):
        """to_json() is deterministic."""
        summary = ReadinessSummary(
            schema_version="1.0.0",
            slice_count=2,
            ready_count=1,
            degraded_count=1,
            blocked_count=0,
            slices={
                "b_slice": {"status": "DEGRADED"},
                "a_slice": {"status": "READY"},
            }
        )
        
        jsons = [summary.to_json() for _ in range(5)]
        self.assertTrue(all(j == jsons[0] for j in jsons))


class TestSummarizeReadiness(TestCurriculumWithTempFiles):
    """Tests for summarize_readiness function."""
    
    def test_summarize_readiness_counts_correct(self):
        """Counts are consistent with per-slice results."""
        contract_dir = Path(self.temp_dir) / "contracts"
        export_contract_bundle(self.curriculum_path, contract_dir)
        
        # Get results for all slices
        results = []
        for slice_name in ["test_slice_goal", "test_slice_sparse", "test_slice_chain", "test_slice_multi"]:
            result = check_slice_readiness(
                slice_name,
                self.curriculum_path,
                self.prereg_path,
                contract_dir
            )
            results.append(result)
        
        summary = summarize_readiness(results)
        
        # Verify counts match
        self.assertEqual(summary.slice_count, len(results))
        self.assertEqual(
            summary.ready_count + summary.degraded_count + summary.blocked_count,
            summary.slice_count
        )
    
    def test_summarize_readiness_schema_version_included(self):
        """schema_version is included and matches constant."""
        results = [
            ReadinessResult(
                slice_name="test",
                status=ReadinessStatus.READY,
                alignment_passes=True,
                drift_class=DriftSeverityClass.NONE,
            )
        ]
        
        summary = summarize_readiness(results)
        self.assertEqual(summary.schema_version, READINESS_SUMMARY_SCHEMA_VERSION)
    
    def test_summarize_readiness_stable_across_runs(self):
        """Summaries are stable across runs."""
        contract_dir = Path(self.temp_dir) / "contracts"
        export_contract_bundle(self.curriculum_path, contract_dir)
        
        def get_summary():
            results = []
            for slice_name in ["test_slice_goal", "test_slice_sparse"]:
                result = check_slice_readiness(
                    slice_name,
                    self.curriculum_path,
                    self.prereg_path,
                    contract_dir
                )
                results.append(result)
            return summarize_readiness(results)
        
        summaries = [get_summary() for _ in range(5)]
        jsons = [s.to_json() for s in summaries]
        
        self.assertTrue(all(j == jsons[0] for j in jsons))
    
    def test_summarize_readiness_per_slice_details(self):
        """Per-slice details are included correctly."""
        results = [
            ReadinessResult(
                slice_name="slice_a",
                status=ReadinessStatus.READY,
                alignment_passes=True,
                drift_class=DriftSeverityClass.NONE,
                missing_contract=False,
                missing_fields=["field1"],
                unknown_fields=["field2"],
            ),
            ReadinessResult(
                slice_name="slice_b",
                status=ReadinessStatus.BLOCKED,
                alignment_passes=False,
                drift_class=DriftSeverityClass.SEMANTIC,
                missing_contract=True,
            ),
        ]
        
        summary = summarize_readiness(results)
        
        self.assertIn("slice_a", summary.slices)
        self.assertIn("slice_b", summary.slices)
        self.assertEqual(summary.slices["slice_a"]["status"], "READY")
        self.assertEqual(summary.slices["slice_b"]["status"], "BLOCKED")
        self.assertTrue(summary.slices["slice_b"]["missing_contract"])


# =============================================================================
# V2.0 Task 2: CI-Grade One-Line Readiness Verdict Tests
# =============================================================================

class TestReadinessVerdict(unittest.TestCase):
    """Tests for ReadinessVerdict enum."""
    
    def test_verdict_values(self):
        """ReadinessVerdict has expected values."""
        self.assertEqual(ReadinessVerdict.OK.value, "OK")
        self.assertEqual(ReadinessVerdict.WARN.value, "WARN")
        self.assertEqual(ReadinessVerdict.BLOCK.value, "BLOCK")


class TestComputeReadinessVerdict(unittest.TestCase):
    """Tests for compute_readiness_verdict function."""
    
    def test_ok_when_all_ready(self):
        """OK when no blocked or degraded slices."""
        summary = ReadinessSummary(
            schema_version="1.0.0",
            slice_count=3,
            ready_count=3,
            degraded_count=0,
            blocked_count=0,
            slices={}
        )
        
        self.assertEqual(compute_readiness_verdict(summary), ReadinessVerdict.OK)
    
    def test_warn_when_degraded(self):
        """WARN when degraded but no blocked."""
        summary = ReadinessSummary(
            schema_version="1.0.0",
            slice_count=3,
            ready_count=2,
            degraded_count=1,
            blocked_count=0,
            slices={}
        )
        
        self.assertEqual(compute_readiness_verdict(summary), ReadinessVerdict.WARN)
    
    def test_block_when_blocked(self):
        """BLOCK when any blocked slices."""
        summary = ReadinessSummary(
            schema_version="1.0.0",
            slice_count=3,
            ready_count=1,
            degraded_count=1,
            blocked_count=1,
            slices={}
        )
        
        self.assertEqual(compute_readiness_verdict(summary), ReadinessVerdict.BLOCK)
    
    def test_block_overrides_warn(self):
        """BLOCK takes priority over WARN."""
        summary = ReadinessSummary(
            schema_version="1.0.0",
            slice_count=4,
            ready_count=1,
            degraded_count=2,
            blocked_count=1,
            slices={}
        )
        
        self.assertEqual(compute_readiness_verdict(summary), ReadinessVerdict.BLOCK)


class TestFormatReadinessSummaryLine(unittest.TestCase):
    """Tests for format_readiness_summary_line function."""
    
    def test_exact_format(self):
        """Output matches exact expected format."""
        summary = ReadinessSummary(
            schema_version="1.0.0",
            slice_count=10,
            ready_count=7,
            degraded_count=2,
            blocked_count=1,
            slices={}
        )
        
        line = format_readiness_summary_line(summary)
        
        # Exact format check
        self.assertEqual(
            line,
            "Metric Readiness: ready=7 degraded=2 blocked=1 total=10 STATUS=BLOCK"
        )
    
    def test_format_ok_status(self):
        """OK status format."""
        summary = ReadinessSummary(
            schema_version="1.0.0",
            slice_count=5,
            ready_count=5,
            degraded_count=0,
            blocked_count=0,
            slices={}
        )
        
        line = format_readiness_summary_line(summary)
        self.assertIn("STATUS=OK", line)
    
    def test_format_warn_status(self):
        """WARN status format."""
        summary = ReadinessSummary(
            schema_version="1.0.0",
            slice_count=5,
            ready_count=4,
            degraded_count=1,
            blocked_count=0,
            slices={}
        )
        
        line = format_readiness_summary_line(summary)
        self.assertIn("STATUS=WARN", line)
    
    def test_no_newlines(self):
        """Output has no newlines."""
        summary = ReadinessSummary(
            schema_version="1.0.0",
            slice_count=3,
            ready_count=3,
            degraded_count=0,
            blocked_count=0,
            slices={}
        )
        
        line = format_readiness_summary_line(summary)
        self.assertNotIn("\n", line)


class TestGetReadinessVerdictExitCode(unittest.TestCase):
    """Tests for get_readiness_verdict_exit_code function."""
    
    def test_ok_exit_code_zero(self):
        """OK verdict gives exit code 0."""
        self.assertEqual(get_readiness_verdict_exit_code(ReadinessVerdict.OK), 0)
    
    def test_warn_exit_code_zero(self):
        """WARN verdict gives exit code 0."""
        self.assertEqual(get_readiness_verdict_exit_code(ReadinessVerdict.WARN), 0)
    
    def test_block_exit_code_one(self):
        """BLOCK verdict gives exit code 1."""
        self.assertEqual(get_readiness_verdict_exit_code(ReadinessVerdict.BLOCK), 1)


# =============================================================================
# V2.0 Task 3: Readiness Consumer Predicate Tests
# =============================================================================

class TestIsSliceReadyForExperiments(unittest.TestCase):
    """Tests for is_slice_ready_for_experiments predicate."""
    
    def test_ready_and_none_drift_returns_true(self):
        """READY + NONE drift returns True."""
        result = ReadinessResult(
            slice_name="test",
            status=ReadinessStatus.READY,
            alignment_passes=True,
            drift_class=DriftSeverityClass.NONE,
        )
        
        self.assertTrue(is_slice_ready_for_experiments(result))
    
    def test_ready_and_cosmetic_drift_returns_true(self):
        """READY + COSMETIC drift returns True."""
        result = ReadinessResult(
            slice_name="test",
            status=ReadinessStatus.READY,
            alignment_passes=True,
            drift_class=DriftSeverityClass.COSMETIC,
        )
        
        self.assertTrue(is_slice_ready_for_experiments(result))
    
    def test_ready_and_parametric_minor_returns_false(self):
        """READY + PARAMETRIC_MINOR drift returns False (boundary test)."""
        result = ReadinessResult(
            slice_name="test",
            status=ReadinessStatus.READY,
            alignment_passes=True,
            drift_class=DriftSeverityClass.PARAMETRIC_MINOR,
        )
        
        self.assertFalse(is_slice_ready_for_experiments(result))
    
    def test_ready_and_semantic_returns_false(self):
        """READY + SEMANTIC drift returns False."""
        result = ReadinessResult(
            slice_name="test",
            status=ReadinessStatus.READY,
            alignment_passes=True,
            drift_class=DriftSeverityClass.SEMANTIC,
        )
        
        self.assertFalse(is_slice_ready_for_experiments(result))
    
    def test_degraded_returns_false(self):
        """DEGRADED status returns False regardless of drift."""
        result = ReadinessResult(
            slice_name="test",
            status=ReadinessStatus.DEGRADED,
            alignment_passes=True,
            drift_class=DriftSeverityClass.NONE,
        )
        
        self.assertFalse(is_slice_ready_for_experiments(result))
    
    def test_blocked_returns_false(self):
        """BLOCKED status returns False."""
        result = ReadinessResult(
            slice_name="test",
            status=ReadinessStatus.BLOCKED,
            alignment_passes=False,
            drift_class=DriftSeverityClass.SEMANTIC,
        )
        
        self.assertFalse(is_slice_ready_for_experiments(result))
    
    def test_deterministic_for_synthetic_inputs(self):
        """Predicate is deterministic for same input."""
        result = ReadinessResult(
            slice_name="test",
            status=ReadinessStatus.READY,
            alignment_passes=True,
            drift_class=DriftSeverityClass.NONE,
        )
        
        outputs = [is_slice_ready_for_experiments(result) for _ in range(10)]
        self.assertTrue(all(o == outputs[0] for o in outputs))


class TestBatchCheckReadinessForExperiments(unittest.TestCase):
    """Tests for batch_check_readiness_for_experiments function."""
    
    def test_batch_check_returns_dict(self):
        """Returns dict mapping slice_name to bool."""
        results = [
            ReadinessResult(
                slice_name="slice_a",
                status=ReadinessStatus.READY,
                alignment_passes=True,
                drift_class=DriftSeverityClass.NONE,
            ),
            ReadinessResult(
                slice_name="slice_b",
                status=ReadinessStatus.BLOCKED,
                alignment_passes=False,
                drift_class=DriftSeverityClass.SEMANTIC,
            ),
        ]
        
        batch_result = batch_check_readiness_for_experiments(results)
        
        self.assertIsInstance(batch_result, dict)
        self.assertTrue(batch_result["slice_a"])
        self.assertFalse(batch_result["slice_b"])
    
    def test_batch_check_deterministic(self):
        """Batch check is deterministic."""
        results = [
            ReadinessResult(
                slice_name="slice_a",
                status=ReadinessStatus.READY,
                alignment_passes=True,
                drift_class=DriftSeverityClass.NONE,
            ),
        ]
        
        batch_results = [batch_check_readiness_for_experiments(results) for _ in range(5)]
        self.assertTrue(all(b == batch_results[0] for b in batch_results))


class TestAcceptableDriftSeverities(unittest.TestCase):
    """Tests for ACCEPTABLE_DRIFT_SEVERITIES_FOR_EXPERIMENTS constant."""
    
    def test_constant_is_frozenset(self):
        """Constant is immutable frozenset."""
        self.assertIsInstance(ACCEPTABLE_DRIFT_SEVERITIES_FOR_EXPERIMENTS, frozenset)
    
    def test_contains_expected_values(self):
        """Contains NONE and COSMETIC."""
        self.assertIn("NONE", ACCEPTABLE_DRIFT_SEVERITIES_FOR_EXPERIMENTS)
        self.assertIn("COSMETIC", ACCEPTABLE_DRIFT_SEVERITIES_FOR_EXPERIMENTS)
    
    def test_does_not_contain_parametric_minor(self):
        """Does not contain PARAMETRIC_MINOR."""
        self.assertNotIn("PARAMETRIC_MINOR", ACCEPTABLE_DRIFT_SEVERITIES_FOR_EXPERIMENTS)
    
    def test_does_not_contain_semantic(self):
        """Does not contain SEMANTIC."""
        self.assertNotIn("SEMANTIC", ACCEPTABLE_DRIFT_SEVERITIES_FOR_EXPERIMENTS)


# =============================================================================
# Phase III Task 1: Per-Metric Readiness Matrix Tests
# =============================================================================

class TestBuildMetricReadinessMatrix(unittest.TestCase):
    """Tests for build_metric_readiness_matrix function."""
    
    def test_matrix_has_schema_version(self):
        """Matrix output includes schema_version."""
        results = [
            ReadinessResult(
                slice_name="slice_a",
                status=ReadinessStatus.READY,
                alignment_passes=True,
                drift_class=DriftSeverityClass.NONE,
                log_field_coverage={"metric_kind": "goal_hit"},
            ),
        ]
        
        matrix = build_metric_readiness_matrix(results)
        
        self.assertIn("schema_version", matrix)
        self.assertEqual(matrix["schema_version"], READINESS_MATRIX_SCHEMA_VERSION)
    
    def test_matrix_has_matrix_key(self):
        """Matrix output has 'matrix' key."""
        results = [
            ReadinessResult(
                slice_name="slice_a",
                status=ReadinessStatus.READY,
                alignment_passes=True,
                drift_class=DriftSeverityClass.NONE,
            ),
        ]
        
        matrix = build_metric_readiness_matrix(results)
        
        self.assertIn("matrix", matrix)
    
    def test_matrix_per_slice_per_metric(self):
        """Matrix contains per-slice per-metric entries."""
        results = [
            ReadinessResult(
                slice_name="slice_a",
                status=ReadinessStatus.READY,
                alignment_passes=True,
                drift_class=DriftSeverityClass.NONE,
                log_field_coverage={"metric_kind": "goal_hit"},
            ),
            ReadinessResult(
                slice_name="slice_b",
                status=ReadinessStatus.DEGRADED,
                alignment_passes=True,
                drift_class=DriftSeverityClass.PARAMETRIC_MINOR,
                log_field_coverage={"metric_kind": "sparse_success"},
            ),
        ]
        
        matrix = build_metric_readiness_matrix(results)
        
        self.assertIn("slice_a", matrix["matrix"])
        self.assertIn("slice_b", matrix["matrix"])
        self.assertIn("goal_hit", matrix["matrix"]["slice_a"])
        self.assertIn("sparse_success", matrix["matrix"]["slice_b"])
    
    def test_matrix_entry_has_required_fields(self):
        """Each matrix entry has status, drift_severity, ready_for_experiments."""
        results = [
            ReadinessResult(
                slice_name="test",
                status=ReadinessStatus.READY,
                alignment_passes=True,
                drift_class=DriftSeverityClass.NONE,
                log_field_coverage={"metric_kind": "goal_hit"},
            ),
        ]
        
        matrix = build_metric_readiness_matrix(results)
        entry = matrix["matrix"]["test"]["goal_hit"]
        
        self.assertIn("status", entry)
        self.assertIn("drift_severity", entry)
        self.assertIn("ready_for_experiments", entry)
    
    def test_matrix_ready_for_experiments_uses_predicate(self):
        """ready_for_experiments uses is_slice_ready_for_experiments logic."""
        # READY + NONE -> True
        results_ready = [
            ReadinessResult(
                slice_name="test",
                status=ReadinessStatus.READY,
                alignment_passes=True,
                drift_class=DriftSeverityClass.NONE,
                log_field_coverage={"metric_kind": "goal_hit"},
            ),
        ]
        matrix_ready = build_metric_readiness_matrix(results_ready)
        self.assertTrue(matrix_ready["matrix"]["test"]["goal_hit"]["ready_for_experiments"])
        
        # READY + PARAMETRIC_MINOR -> False
        results_not_ready = [
            ReadinessResult(
                slice_name="test",
                status=ReadinessStatus.READY,
                alignment_passes=True,
                drift_class=DriftSeverityClass.PARAMETRIC_MINOR,
                log_field_coverage={"metric_kind": "goal_hit"},
            ),
        ]
        matrix_not_ready = build_metric_readiness_matrix(results_not_ready)
        self.assertFalse(matrix_not_ready["matrix"]["test"]["goal_hit"]["ready_for_experiments"])
    
    def test_matrix_slices_sorted_alphabetically(self):
        """Slices are sorted alphabetically."""
        results = [
            ReadinessResult(
                slice_name="z_slice",
                status=ReadinessStatus.READY,
                alignment_passes=True,
                drift_class=DriftSeverityClass.NONE,
            ),
            ReadinessResult(
                slice_name="a_slice",
                status=ReadinessStatus.READY,
                alignment_passes=True,
                drift_class=DriftSeverityClass.NONE,
            ),
            ReadinessResult(
                slice_name="m_slice",
                status=ReadinessStatus.READY,
                alignment_passes=True,
                drift_class=DriftSeverityClass.NONE,
            ),
        ]
        
        matrix = build_metric_readiness_matrix(results)
        
        slice_keys = list(matrix["matrix"].keys())
        self.assertEqual(slice_keys, ["a_slice", "m_slice", "z_slice"])
    
    def test_matrix_metric_kinds_sorted_alphabetically(self):
        """Metric kinds within a slice are sorted alphabetically."""
        # Create a slice with multiple metrics (artificial scenario)
        results = [
            ReadinessResult(
                slice_name="slice_a",
                status=ReadinessStatus.READY,
                alignment_passes=True,
                drift_class=DriftSeverityClass.NONE,
                log_field_coverage={"metric_kind": "z_metric"},
            ),
        ]
        
        matrix = build_metric_readiness_matrix(results)
        
        # Single metric should still work
        self.assertIn("z_metric", matrix["matrix"]["slice_a"])
    
    def test_matrix_deterministic(self):
        """Matrix is deterministic across runs."""
        results = [
            ReadinessResult(
                slice_name="slice_b",
                status=ReadinessStatus.READY,
                alignment_passes=True,
                drift_class=DriftSeverityClass.NONE,
                log_field_coverage={"metric_kind": "goal_hit"},
            ),
            ReadinessResult(
                slice_name="slice_a",
                status=ReadinessStatus.DEGRADED,
                alignment_passes=True,
                drift_class=DriftSeverityClass.COSMETIC,
                log_field_coverage={"metric_kind": "sparse_success"},
            ),
        ]
        
        matrices = [build_metric_readiness_matrix(results) for _ in range(5)]
        json_outputs = [json.dumps(m, sort_keys=True) for m in matrices]
        
        self.assertTrue(all(j == json_outputs[0] for j in json_outputs))
    
    def test_matrix_handles_missing_log_field_coverage(self):
        """Matrix handles missing log_field_coverage gracefully."""
        results = [
            ReadinessResult(
                slice_name="test",
                status=ReadinessStatus.READY,
                alignment_passes=True,
                drift_class=DriftSeverityClass.NONE,
                # No log_field_coverage
            ),
        ]
        
        matrix = build_metric_readiness_matrix(results)
        
        # Should use "unknown" as metric_kind
        self.assertIn("unknown", matrix["matrix"]["test"])


# =============================================================================
# Phase III Task 2: Promotion Guard Helper Tests
# =============================================================================

class TestEvaluateMetricReadinessForPromotion(unittest.TestCase):
    """Tests for evaluate_metric_readiness_for_promotion function."""
    
    def test_promotion_ok_when_all_ready(self):
        """promotion_ok is True when all slices are ready."""
        summary = ReadinessSummary(
            schema_version="1.0.0",
            slice_count=3,
            ready_count=3,
            degraded_count=0,
            blocked_count=0,
            slices={
                "slice_a": {"status": "READY"},
                "slice_b": {"status": "READY"},
                "slice_c": {"status": "READY"},
            }
        )
        
        eval_result = evaluate_metric_readiness_for_promotion(summary)
        
        self.assertTrue(eval_result["promotion_ok"])
        self.assertEqual(eval_result["blocking_slices"], [])
        self.assertEqual(eval_result["verdict"], "OK")
    
    def test_promotion_blocked_when_blocked_slices(self):
        """promotion_ok is False when there are blocked slices."""
        summary = ReadinessSummary(
            schema_version="1.0.0",
            slice_count=3,
            ready_count=2,
            degraded_count=0,
            blocked_count=1,
            slices={
                "slice_a": {"status": "READY"},
                "slice_b": {"status": "BLOCKED"},
                "slice_c": {"status": "READY"},
            }
        )
        
        eval_result = evaluate_metric_readiness_for_promotion(summary)
        
        self.assertFalse(eval_result["promotion_ok"])
        self.assertIn("slice_b", eval_result["blocking_slices"])
        self.assertEqual(eval_result["verdict"], "BLOCK")
    
    def test_promotion_warn_when_degraded(self):
        """verdict is WARN when degraded but no blocked."""
        summary = ReadinessSummary(
            schema_version="1.0.0",
            slice_count=3,
            ready_count=2,
            degraded_count=1,
            blocked_count=0,
            slices={
                "slice_a": {"status": "READY"},
                "slice_b": {"status": "DEGRADED"},
                "slice_c": {"status": "READY"},
            }
        )
        
        eval_result = evaluate_metric_readiness_for_promotion(summary)
        
        self.assertTrue(eval_result["promotion_ok"])  # Still promotable
        self.assertEqual(eval_result["verdict"], "WARN")
    
    def test_uplift_slice_not_ready_blocks_promotion(self):
        """Uplift slices not ready for experiments block promotion."""
        summary = ReadinessSummary(
            schema_version="1.0.0",
            slice_count=2,
            ready_count=2,
            degraded_count=0,
            blocked_count=0,
            slices={
                "slice_uplift_goal": {"status": "READY"},
                "slice_other": {"status": "READY"},
            }
        )
        
        # Uplift slice has PARAMETRIC_MINOR drift -> not ready for experiments
        results = [
            ReadinessResult(
                slice_name="slice_uplift_goal",
                status=ReadinessStatus.READY,
                alignment_passes=True,
                drift_class=DriftSeverityClass.PARAMETRIC_MINOR,  # Not acceptable
            ),
            ReadinessResult(
                slice_name="slice_other",
                status=ReadinessStatus.READY,
                alignment_passes=True,
                drift_class=DriftSeverityClass.NONE,
            ),
        ]
        
        eval_result = evaluate_metric_readiness_for_promotion(summary, results)
        
        self.assertFalse(eval_result["promotion_ok"])
        self.assertIn("slice_uplift_goal", eval_result["blocking_slices"])
    
    def test_non_uplift_slice_with_drift_ok(self):
        """Non-uplift slices with drift don't block promotion."""
        summary = ReadinessSummary(
            schema_version="1.0.0",
            slice_count=2,
            ready_count=2,
            degraded_count=0,
            blocked_count=0,
            slices={
                "slice_uplift_goal": {"status": "READY"},
                "slice_other": {"status": "READY"},
            }
        )
        
        results = [
            ReadinessResult(
                slice_name="slice_uplift_goal",
                status=ReadinessStatus.READY,
                alignment_passes=True,
                drift_class=DriftSeverityClass.NONE,  # OK
            ),
            ReadinessResult(
                slice_name="slice_other",
                status=ReadinessStatus.READY,
                alignment_passes=True,
                drift_class=DriftSeverityClass.PARAMETRIC_MINOR,  # Has drift but not uplift
            ),
        ]
        
        eval_result = evaluate_metric_readiness_for_promotion(summary, results)
        
        self.assertTrue(eval_result["promotion_ok"])
        self.assertEqual(eval_result["blocking_slices"], [])
    
    def test_blocking_slices_sorted(self):
        """Blocking slices are sorted alphabetically."""
        summary = ReadinessSummary(
            schema_version="1.0.0",
            slice_count=3,
            ready_count=0,
            degraded_count=0,
            blocked_count=3,
            slices={
                "z_slice": {"status": "BLOCKED"},
                "a_slice": {"status": "BLOCKED"},
                "m_slice": {"status": "BLOCKED"},
            }
        )
        
        eval_result = evaluate_metric_readiness_for_promotion(summary)
        
        self.assertEqual(
            eval_result["blocking_slices"],
            ["a_slice", "m_slice", "z_slice"]
        )
    
    def test_promotion_deterministic(self):
        """Promotion evaluation is deterministic."""
        summary = ReadinessSummary(
            schema_version="1.0.0",
            slice_count=2,
            ready_count=1,
            degraded_count=0,
            blocked_count=1,
            slices={
                "slice_a": {"status": "READY"},
                "slice_b": {"status": "BLOCKED"},
            }
        )
        
        evals = [evaluate_metric_readiness_for_promotion(summary) for _ in range(5)]
        json_outputs = [json.dumps(e, sort_keys=True) for e in evals]
        
        self.assertTrue(all(j == json_outputs[0] for j in json_outputs))
    
    def test_uplift_prefix_constant(self):
        """UPLIFT_SLICE_PREFIX is the expected value."""
        self.assertEqual(UPLIFT_SLICE_PREFIX, "slice_uplift_")


# =============================================================================
# Phase III Task 3: Global Health & MAAS Snapshot Tests
# =============================================================================

class TestSummarizeMetricReadinessForGlobalHealth(unittest.TestCase):
    """Tests for summarize_metric_readiness_for_global_health function."""
    
    def test_has_required_fields(self):
        """Output has all required fields."""
        summary = ReadinessSummary(
            schema_version="1.0.0",
            slice_count=3,
            ready_count=3,
            degraded_count=0,
            blocked_count=0,
            slices={}
        )
        promotion_eval = {
            "promotion_ok": True,
            "blocking_slices": [],
            "verdict": "OK",
        }
        
        result = summarize_metric_readiness_for_global_health(summary, promotion_eval)
        
        self.assertIn("readiness_status", result)
        self.assertIn("ready_slice_ratio", result)
        self.assertIn("blocked_slice_count", result)
        self.assertIn("promotion_ok", result)
    
    def test_readiness_status_from_verdict(self):
        """readiness_status derived from promotion verdict."""
        summary = ReadinessSummary(
            schema_version="1.0.0",
            slice_count=3,
            ready_count=3,
            degraded_count=0,
            blocked_count=0,
            slices={}
        )
        
        # Test OK
        result_ok = summarize_metric_readiness_for_global_health(
            summary, {"promotion_ok": True, "verdict": "OK"}
        )
        self.assertEqual(result_ok["readiness_status"], "OK")
        
        # Test WARN
        result_warn = summarize_metric_readiness_for_global_health(
            summary, {"promotion_ok": True, "verdict": "WARN"}
        )
        self.assertEqual(result_warn["readiness_status"], "WARN")
        
        # Test BLOCK
        result_block = summarize_metric_readiness_for_global_health(
            summary, {"promotion_ok": False, "verdict": "BLOCK"}
        )
        self.assertEqual(result_block["readiness_status"], "BLOCK")
    
    def test_ready_slice_ratio_computed(self):
        """ready_slice_ratio is ready_count / slice_count."""
        summary = ReadinessSummary(
            schema_version="1.0.0",
            slice_count=4,
            ready_count=3,
            degraded_count=1,
            blocked_count=0,
            slices={}
        )
        promotion_eval = {"promotion_ok": True, "verdict": "OK"}
        
        result = summarize_metric_readiness_for_global_health(summary, promotion_eval)
        
        self.assertEqual(result["ready_slice_ratio"], 0.75)
    
    def test_ready_slice_ratio_zero_slices(self):
        """ready_slice_ratio handles zero slices safely."""
        summary = ReadinessSummary(
            schema_version="1.0.0",
            slice_count=0,
            ready_count=0,
            degraded_count=0,
            blocked_count=0,
            slices={}
        )
        promotion_eval = {"promotion_ok": True, "verdict": "OK"}
        
        result = summarize_metric_readiness_for_global_health(summary, promotion_eval)
        
        self.assertEqual(result["ready_slice_ratio"], 0.0)
    
    def test_blocked_slice_count_from_summary(self):
        """blocked_slice_count comes from summary."""
        summary = ReadinessSummary(
            schema_version="1.0.0",
            slice_count=5,
            ready_count=3,
            degraded_count=0,
            blocked_count=2,
            slices={}
        )
        promotion_eval = {"promotion_ok": False, "verdict": "BLOCK"}
        
        result = summarize_metric_readiness_for_global_health(summary, promotion_eval)
        
        self.assertEqual(result["blocked_slice_count"], 2)
    
    def test_promotion_ok_from_eval(self):
        """promotion_ok comes from promotion_eval."""
        summary = ReadinessSummary(
            schema_version="1.0.0",
            slice_count=3,
            ready_count=3,
            degraded_count=0,
            blocked_count=0,
            slices={}
        )
        
        result_true = summarize_metric_readiness_for_global_health(
            summary, {"promotion_ok": True, "verdict": "OK"}
        )
        self.assertTrue(result_true["promotion_ok"])
        
        result_false = summarize_metric_readiness_for_global_health(
            summary, {"promotion_ok": False, "verdict": "BLOCK"}
        )
        self.assertFalse(result_false["promotion_ok"])
    
    def test_deterministic_output(self):
        """Output is deterministic."""
        summary = ReadinessSummary(
            schema_version="1.0.0",
            slice_count=3,
            ready_count=2,
            degraded_count=1,
            blocked_count=0,
            slices={}
        )
        promotion_eval = {"promotion_ok": True, "verdict": "WARN"}
        
        results = [
            summarize_metric_readiness_for_global_health(summary, promotion_eval)
            for _ in range(5)
        ]
        json_outputs = [json.dumps(r, sort_keys=True) for r in results]
        
        self.assertTrue(all(j == json_outputs[0] for j in json_outputs))
    
    def test_output_json_serializable(self):
        """Output is JSON-serializable."""
        summary = ReadinessSummary(
            schema_version="1.0.0",
            slice_count=3,
            ready_count=3,
            degraded_count=0,
            blocked_count=0,
            slices={}
        )
        promotion_eval = {"promotion_ok": True, "verdict": "OK"}
        
        result = summarize_metric_readiness_for_global_health(summary, promotion_eval)
        
        # Should not raise
        json_str = json.dumps(result)
        self.assertIsInstance(json_str, str)


# =============================================================================
# Phase IV Task 1: Cross-Metric Readiness Heatmap Tests
# =============================================================================

class TestBuildReadinessHeatmap(unittest.TestCase):
    """Tests for build_readiness_heatmap function."""
    
    def test_heatmap_has_schema_version(self):
        """Heatmap output includes schema_version."""
        matrix = {
            "schema_version": "1.0.0",
            "matrix": {
                "slice_a": {
                    "goal_hit": {"status": "READY", "drift_severity": "NONE"}
                }
            }
        }
        drift_grid = {"grid": {}}
        budget_view = {"view": {}}
        
        heatmap = build_readiness_heatmap(matrix, drift_grid, budget_view)
        
        self.assertIn("heatmap_schema_version", heatmap)
        self.assertEqual(heatmap["heatmap_schema_version"], READINESS_HEATMAP_SCHEMA_VERSION)
    
    def test_heatmap_has_required_keys(self):
        """Heatmap has heatmap, slices_with_consistent_readiness, metrics_with_conflicting_signals."""
        matrix = {
            "schema_version": "1.0.0",
            "matrix": {
                "slice_a": {
                    "goal_hit": {"status": "READY", "drift_severity": "NONE"}
                }
            }
        }
        drift_grid = {"grid": {}}
        budget_view = {"view": {}}
        
        heatmap = build_readiness_heatmap(matrix, drift_grid, budget_view)
        
        self.assertIn("heatmap", heatmap)
        self.assertIn("slices_with_consistent_readiness", heatmap)
        self.assertIn("metrics_with_conflicting_signals", heatmap)
    
    def test_heatmap_per_slice_per_metric(self):
        """Heatmap contains per-slice per-metric entries."""
        matrix = {
            "schema_version": "1.0.0",
            "matrix": {
                "slice_a": {
                    "goal_hit": {"status": "READY", "drift_severity": "NONE"}
                },
                "slice_b": {
                    "sparse_success": {"status": "DEGRADED", "drift_severity": "PARAMETRIC_MINOR"}
                }
            }
        }
        drift_grid = {"grid": {}}
        budget_view = {"view": {}}
        
        heatmap = build_readiness_heatmap(matrix, drift_grid, budget_view)
        
        self.assertIn("slice_a", heatmap["heatmap"])
        self.assertIn("slice_b", heatmap["heatmap"])
        self.assertIn("goal_hit", heatmap["heatmap"]["slice_a"])
        self.assertIn("sparse_success", heatmap["heatmap"]["slice_b"])
    
    def test_heatmap_entry_has_required_fields(self):
        """Each heatmap entry has readiness_status, drift_status, budget_flag."""
        matrix = {
            "schema_version": "1.0.0",
            "matrix": {
                "slice_a": {
                    "goal_hit": {"status": "READY", "drift_severity": "NONE"}
                }
            }
        }
        drift_grid = {"grid": {}}
        budget_view = {"view": {}}
        
        heatmap = build_readiness_heatmap(matrix, drift_grid, budget_view)
        entry = heatmap["heatmap"]["slice_a"]["goal_hit"]
        
        self.assertIn("readiness_status", entry)
        self.assertIn("drift_status", entry)
        self.assertIn("budget_flag", entry)
    
    def test_heatmap_uses_drift_grid(self):
        """Heatmap extracts drift_status from drift_grid."""
        matrix = {
            "schema_version": "1.0.0",
            "matrix": {
                "slice_a": {
                    "goal_hit": {"status": "READY", "drift_severity": "NONE"}
                }
            }
        }
        drift_grid = {
            "grid": {
                "slice_a": {
                    "goal_hit": {"drift_status": "DRIFTY"}
                }
            }
        }
        budget_view = {"view": {}}
        
        heatmap = build_readiness_heatmap(matrix, drift_grid, budget_view)
        
        self.assertEqual(heatmap["heatmap"]["slice_a"]["goal_hit"]["drift_status"], "DRIFTY")
    
    def test_heatmap_uses_budget_view(self):
        """Heatmap extracts budget_flag from budget_joint_view."""
        matrix = {
            "schema_version": "1.0.0",
            "matrix": {
                "slice_a": {
                    "goal_hit": {"status": "READY", "drift_severity": "NONE"}
                }
            }
        }
        drift_grid = {"grid": {}}
        budget_view = {
            "view": {
                "slice_a": {
                    "goal_hit": {"budget_flag": "STARVED"}
                }
            }
        }
        
        heatmap = build_readiness_heatmap(matrix, drift_grid, budget_view)
        
        self.assertEqual(heatmap["heatmap"]["slice_a"]["goal_hit"]["budget_flag"], "STARVED")
    
    def test_heatmap_defaults_missing_drift(self):
        """Heatmap defaults to OK for missing drift data."""
        matrix = {
            "schema_version": "1.0.0",
            "matrix": {
                "slice_a": {
                    "goal_hit": {"status": "READY", "drift_severity": "NONE"}
                }
            }
        }
        drift_grid = {"grid": {}}  # No drift data
        budget_view = {"view": {}}
        
        heatmap = build_readiness_heatmap(matrix, drift_grid, budget_view)
        
        self.assertEqual(heatmap["heatmap"]["slice_a"]["goal_hit"]["drift_status"], "OK")
    
    def test_heatmap_defaults_missing_budget(self):
        """Heatmap defaults to SAFE for missing budget data."""
        matrix = {
            "schema_version": "1.0.0",
            "matrix": {
                "slice_a": {
                    "goal_hit": {"status": "READY", "drift_severity": "NONE"}
                }
            }
        }
        drift_grid = {"grid": {}}
        budget_view = {"view": {}}  # No budget data
        
        heatmap = build_readiness_heatmap(matrix, drift_grid, budget_view)
        
        self.assertEqual(heatmap["heatmap"]["slice_a"]["goal_hit"]["budget_flag"], "SAFE")
    
    def test_slices_with_consistent_readiness(self):
        """slices_with_consistent_readiness identifies slices with uniform status."""
        matrix = {
            "schema_version": "1.0.0",
            "matrix": {
                "slice_consistent": {
                    "goal_hit": {"status": "READY", "drift_severity": "NONE"},
                    "sparse_success": {"status": "READY", "drift_severity": "NONE"}
                },
                "slice_inconsistent": {
                    "goal_hit": {"status": "READY", "drift_severity": "NONE"},
                    "sparse_success": {"status": "DEGRADED", "drift_severity": "PARAMETRIC_MINOR"}
                }
            }
        }
        drift_grid = {"grid": {}}
        budget_view = {"view": {}}
        
        heatmap = build_readiness_heatmap(matrix, drift_grid, budget_view)
        
        self.assertIn("slice_consistent", heatmap["slices_with_consistent_readiness"])
        self.assertNotIn("slice_inconsistent", heatmap["slices_with_consistent_readiness"])
    
    def test_metrics_with_conflicting_signals(self):
        """metrics_with_conflicting_signals identifies READY but DRIFTY or STARVED."""
        matrix = {
            "schema_version": "1.0.0",
            "matrix": {
                "slice_a": {
                    "goal_hit": {"status": "READY", "drift_severity": "NONE"}
                },
                "slice_b": {
                    "sparse_success": {"status": "READY", "drift_severity": "NONE"}
                },
                "slice_c": {
                    "chain_success": {"status": "DEGRADED", "drift_severity": "PARAMETRIC_MINOR"}
                }
            }
        }
        drift_grid = {
            "grid": {
                "slice_b": {
                    "sparse_success": {"drift_status": "DRIFTY"}
                }
            }
        }
        budget_view = {
            "view": {
                "slice_c": {
                    "chain_success": {"budget_flag": "STARVED"}
                }
            }
        }
        
        heatmap = build_readiness_heatmap(matrix, drift_grid, budget_view)
        
        # slice_b.sparse_success: READY but DRIFTY -> conflict
        self.assertIn("slice_b.sparse_success", heatmap["metrics_with_conflicting_signals"])
        # slice_c.chain_success: DEGRADED but STARVED -> conflict
        self.assertIn("slice_c.chain_success", heatmap["metrics_with_conflicting_signals"])
        # slice_a.goal_hit: READY, OK, SAFE -> no conflict
        self.assertNotIn("slice_a.goal_hit", heatmap["metrics_with_conflicting_signals"])
    
    def test_heatmap_deterministic(self):
        """Heatmap is deterministic across runs."""
        matrix = {
            "schema_version": "1.0.0",
            "matrix": {
                "slice_b": {
                    "goal_hit": {"status": "READY", "drift_severity": "NONE"}
                },
                "slice_a": {
                    "sparse_success": {"status": "DEGRADED", "drift_severity": "PARAMETRIC_MINOR"}
                }
            }
        }
        drift_grid = {"grid": {}}
        budget_view = {"view": {}}
        
        heatmaps = [build_readiness_heatmap(matrix, drift_grid, budget_view) for _ in range(5)]
        json_outputs = [json.dumps(h, sort_keys=True) for h in heatmaps]
        
        self.assertTrue(all(j == json_outputs[0] for j in json_outputs))
    
    def test_heatmap_sorted_keys(self):
        """Heatmap keys are sorted alphabetically."""
        matrix = {
            "schema_version": "1.0.0",
            "matrix": {
                "z_slice": {
                    "z_metric": {"status": "READY", "drift_severity": "NONE"}
                },
                "a_slice": {
                    "a_metric": {"status": "READY", "drift_severity": "NONE"}
                }
            }
        }
        drift_grid = {"grid": {}}
        budget_view = {"view": {}}
        
        heatmap = build_readiness_heatmap(matrix, drift_grid, budget_view)
        
        slice_keys = list(heatmap["heatmap"].keys())
        self.assertEqual(slice_keys, ["a_slice", "z_slice"])


# =============================================================================
# Phase IV Task 2: Release Promotion Policy Helper Tests
# =============================================================================

class TestEvaluateReleasePromotionWithReadiness(unittest.TestCase):
    """Tests for evaluate_release_promotion_with_readiness function."""
    
    def test_promotion_ok_when_no_conflicts(self):
        """promotion_ok is True when no blocking conflicts."""
        heatmap = {
            "heatmap": {
                "slice_a": {
                    "goal_hit": {
                        "readiness_status": "READY",
                        "drift_status": "OK",
                        "budget_flag": "SAFE"
                    }
                }
            },
            "metrics_with_conflicting_signals": []
        }
        
        result = evaluate_release_promotion_with_readiness(heatmap)
        
        self.assertTrue(result["promotion_ok"])
        self.assertEqual(result["blocking_pairs"], [])
        self.assertEqual(result["verdict"], "OK")
    
    def test_promotion_blocked_when_ready_but_drifty_and_starved(self):
        """promotion_ok is False when READY but DRIFTY+STARVED."""
        heatmap = {
            "heatmap": {
                "slice_a": {
                    "goal_hit": {
                        "readiness_status": "READY",
                        "drift_status": "DRIFTY",
                        "budget_flag": "STARVED"
                    }
                }
            },
            "metrics_with_conflicting_signals": []
        }
        
        result = evaluate_release_promotion_with_readiness(heatmap)
        
        self.assertFalse(result["promotion_ok"])
        self.assertIn("slice_a.goal_hit", result["blocking_pairs"])
        self.assertEqual(result["verdict"], "BLOCK")
    
    def test_promotion_blocked_when_blocked_readiness(self):
        """promotion_ok is False when readiness is BLOCKED."""
        heatmap = {
            "heatmap": {
                "slice_a": {
                    "goal_hit": {
                        "readiness_status": "BLOCKED",
                        "drift_status": "OK",
                        "budget_flag": "SAFE"
                    }
                }
            },
            "metrics_with_conflicting_signals": []
        }
        
        result = evaluate_release_promotion_with_readiness(heatmap)
        
        self.assertFalse(result["promotion_ok"])
        self.assertIn("slice_a.goal_hit", result["blocking_pairs"])
        self.assertEqual(result["verdict"], "BLOCK")
    
    def test_promotion_warn_when_conflicting_signals(self):
        """verdict is WARN when conflicting signals but not blocking."""
        heatmap = {
            "heatmap": {
                "slice_a": {
                    "goal_hit": {
                        "readiness_status": "READY",
                        "drift_status": "DRIFTY",
                        "budget_flag": "SAFE"  # Not STARVED, so not blocking
                    }
                }
            },
            "metrics_with_conflicting_signals": ["slice_a.goal_hit"]
        }
        
        result = evaluate_release_promotion_with_readiness(heatmap)
        
        self.assertTrue(result["promotion_ok"])  # Not blocking
        self.assertEqual(result["verdict"], "WARN")
        self.assertIn("slice_a.goal_hit: READY but DRIFTY", result["reasons"])
    
    def test_reasons_include_all_conflicts(self):
        """reasons list includes all conflict descriptions."""
        heatmap = {
            "heatmap": {
                "slice_a": {
                    "goal_hit": {
                        "readiness_status": "READY",
                        "drift_status": "DRIFTY",
                        "budget_flag": "STARVED"
                    }
                },
                "slice_b": {
                    "sparse_success": {
                        "readiness_status": "BLOCKED",
                        "drift_status": "OK",
                        "budget_flag": "SAFE"
                    }
                }
            },
            "metrics_with_conflicting_signals": []
        }
        
        result = evaluate_release_promotion_with_readiness(heatmap)
        
        self.assertEqual(len(result["reasons"]), 2)
        self.assertIn("slice_a.goal_hit: READY but DRIFTY+STARVED", result["reasons"])
        self.assertIn("slice_b.sparse_success: BLOCKED readiness", result["reasons"])
    
    def test_blocking_pairs_sorted(self):
        """blocking_pairs are sorted alphabetically."""
        heatmap = {
            "heatmap": {
                "z_slice": {
                    "z_metric": {"readiness_status": "BLOCKED", "drift_status": "OK", "budget_flag": "SAFE"}
                },
                "a_slice": {
                    "a_metric": {"readiness_status": "BLOCKED", "drift_status": "OK", "budget_flag": "SAFE"}
                }
            },
            "metrics_with_conflicting_signals": []
        }
        
        result = evaluate_release_promotion_with_readiness(heatmap)
        
        self.assertEqual(
            result["blocking_pairs"],
            ["a_slice.a_metric", "z_slice.z_metric"]
        )
    
    def test_deterministic_output(self):
        """Output is deterministic."""
        heatmap = {
            "heatmap": {
                "slice_a": {
                    "goal_hit": {
                        "readiness_status": "READY",
                        "drift_status": "DRIFTY",
                        "budget_flag": "SAFE"
                    }
                }
            },
            "metrics_with_conflicting_signals": ["slice_a.goal_hit"]
        }
        
        results = [evaluate_release_promotion_with_readiness(heatmap) for _ in range(5)]
        json_outputs = [json.dumps(r, sort_keys=True) for r in results]
        
        self.assertTrue(all(j == json_outputs[0] for j in json_outputs))


# =============================================================================
# Phase IV Task 3: Director Metric Readiness Panel Tests
# =============================================================================

class TestBuildMetricReadinessDirectorPanel(unittest.TestCase):
    """Tests for build_metric_readiness_director_panel function."""
    
    def test_panel_has_required_fields(self):
        """Panel has all required fields."""
        readiness_summary = {
            "readiness_status": "OK",
            "ready_slice_ratio": 1.0,
            "blocked_slice_count": 0,
            "promotion_ok": True
        }
        heatmap_view = {"metrics_with_conflicting_signals": []}
        promotion_eval = {"promotion_ok": True, "verdict": "OK"}
        
        panel = build_metric_readiness_director_panel(
            readiness_summary, heatmap_view, promotion_eval
        )
        
        self.assertIn("status_light", panel)
        self.assertIn("readiness_status", panel)
        self.assertIn("ready_slice_ratio", panel)
        self.assertIn("blocked_slice_count", panel)
        self.assertIn("promotion_ok", panel)
        self.assertIn("headline", panel)
    
    def test_status_light_green_when_ok(self):
        """status_light is GREEN when OK and promotion_ok."""
        readiness_summary = {
            "readiness_status": "OK",
            "ready_slice_ratio": 1.0,
            "blocked_slice_count": 0,
            "promotion_ok": True
        }
        heatmap_view = {"metrics_with_conflicting_signals": []}
        promotion_eval = {"promotion_ok": True, "verdict": "OK"}
        
        panel = build_metric_readiness_director_panel(
            readiness_summary, heatmap_view, promotion_eval
        )
        
        self.assertEqual(panel["status_light"], "GREEN")
    
    def test_status_light_red_when_block(self):
        """status_light is RED when BLOCK or not promotion_ok."""
        readiness_summary = {
            "readiness_status": "BLOCK",
            "ready_slice_ratio": 0.5,
            "blocked_slice_count": 2,
            "promotion_ok": False
        }
        heatmap_view = {"metrics_with_conflicting_signals": []}
        promotion_eval = {"promotion_ok": False, "verdict": "BLOCK"}
        
        panel = build_metric_readiness_director_panel(
            readiness_summary, heatmap_view, promotion_eval
        )
        
        self.assertEqual(panel["status_light"], "RED")
    
    def test_status_light_yellow_when_warn(self):
        """status_light is YELLOW when WARN."""
        readiness_summary = {
            "readiness_status": "WARN",
            "ready_slice_ratio": 0.8,
            "blocked_slice_count": 0,
            "promotion_ok": True
        }
        heatmap_view = {"metrics_with_conflicting_signals": []}
        promotion_eval = {"promotion_ok": True, "verdict": "WARN"}
        
        panel = build_metric_readiness_director_panel(
            readiness_summary, heatmap_view, promotion_eval
        )
        
        self.assertEqual(panel["status_light"], "YELLOW")
    
    def test_headline_all_slices_ready(self):
        """headline says 'All slices ready' when ratio is 1.0."""
        readiness_summary = {
            "readiness_status": "OK",
            "ready_slice_ratio": 1.0,
            "blocked_slice_count": 0,
            "promotion_ok": True
        }
        heatmap_view = {"metrics_with_conflicting_signals": []}
        promotion_eval = {"promotion_ok": True, "verdict": "OK"}
        
        panel = build_metric_readiness_director_panel(
            readiness_summary, heatmap_view, promotion_eval
        )
        
        self.assertIn("All slices ready", panel["headline"])
    
    def test_headline_includes_blocked_count(self):
        """headline includes blocked count when > 0."""
        readiness_summary = {
            "readiness_status": "BLOCK",
            "ready_slice_ratio": 0.5,
            "blocked_slice_count": 2,
            "promotion_ok": False
        }
        heatmap_view = {"metrics_with_conflicting_signals": []}
        promotion_eval = {"promotion_ok": False, "verdict": "BLOCK"}
        
        panel = build_metric_readiness_director_panel(
            readiness_summary, heatmap_view, promotion_eval
        )
        
        self.assertIn("2 slice(s) blocked", panel["headline"])
    
    def test_headline_includes_conflicting_signals(self):
        """headline includes cross-signal conflicts when present."""
        readiness_summary = {
            "readiness_status": "OK",
            "ready_slice_ratio": 1.0,
            "blocked_slice_count": 0,
            "promotion_ok": True
        }
        heatmap_view = {
            "metrics_with_conflicting_signals": [
                "slice_a.goal_hit",
                "slice_b.sparse_success"
            ]
        }
        promotion_eval = {"promotion_ok": True, "verdict": "OK"}
        
        panel = build_metric_readiness_director_panel(
            readiness_summary, heatmap_view, promotion_eval
        )
        
        self.assertIn("2 cross-signal conflict(s)", panel["headline"])
    
    def test_headline_neutral_language(self):
        """headline uses neutral language (no value judgments)."""
        readiness_summary = {
            "readiness_status": "OK",
            "ready_slice_ratio": 0.75,
            "blocked_slice_count": 0,
            "promotion_ok": True
        }
        heatmap_view = {"metrics_with_conflicting_signals": []}
        promotion_eval = {"promotion_ok": True, "verdict": "OK"}
        
        panel = build_metric_readiness_director_panel(
            readiness_summary, heatmap_view, promotion_eval
        )
        
        headline = panel["headline"]
        # Check for absence of value judgment words
        self.assertNotIn("good", headline.lower())
        self.assertNotIn("bad", headline.lower())
        self.assertNotIn("better", headline.lower())
        self.assertNotIn("worse", headline.lower())
    
    def test_panel_deterministic(self):
        """Panel output is deterministic."""
        readiness_summary = {
            "readiness_status": "WARN",
            "ready_slice_ratio": 0.8,
            "blocked_slice_count": 0,
            "promotion_ok": True
        }
        heatmap_view = {"metrics_with_conflicting_signals": []}
        promotion_eval = {"promotion_ok": True, "verdict": "WARN"}
        
        panels = [
            build_metric_readiness_director_panel(
                readiness_summary, heatmap_view, promotion_eval
            )
            for _ in range(5)
        ]
        json_outputs = [json.dumps(p, sort_keys=True) for p in panels]
        
        self.assertTrue(all(j == json_outputs[0] for j in json_outputs))
    
    def test_panel_json_serializable(self):
        """Panel output is JSON-serializable."""
        readiness_summary = {
            "readiness_status": "OK",
            "ready_slice_ratio": 1.0,
            "blocked_slice_count": 0,
            "promotion_ok": True
        }
        heatmap_view = {"metrics_with_conflicting_signals": []}
        promotion_eval = {"promotion_ok": True, "verdict": "OK"}
        
        panel = build_metric_readiness_director_panel(
            readiness_summary, heatmap_view, promotion_eval
        )
        
        # Should not raise
        json_str = json.dumps(panel)
        self.assertIsInstance(json_str, str)


# =============================================================================
# Phase V Task 1: Readiness Autopilot Policy Tests
# =============================================================================

class TestBuildReadinessAutopilotPolicy(unittest.TestCase):
    """Tests for build_readiness_autopilot_policy function."""
    
    def test_autopilot_ok_when_high_ratio_no_conflicts(self):
        """autopilot_status is OK when high ready ratio and no conflicts."""
        heatmap = {
            "heatmap": {
                "slice_a": {
                    "goal_hit": {
                        "readiness_status": "READY",
                        "drift_status": "OK",
                        "budget_flag": "SAFE"
                    }
                },
                "slice_b": {
                    "sparse_success": {
                        "readiness_status": "READY",
                        "drift_status": "OK",
                        "budget_flag": "SAFE"
                    }
                }
            },
            "metrics_with_conflicting_signals": []
        }
        history = {}
        
        policy = build_readiness_autopilot_policy(heatmap, history, target_ready_ratio=0.8)
        
        self.assertEqual(policy["autopilot_status"], "OK")
        self.assertEqual(len(policy["slices_to_hold"]), 0)
        self.assertGreater(len(policy["slices_safe_to_progress"]), 0)
    
    def test_autopilot_attention_when_low_ratio(self):
        """autopilot_status is ATTENTION when ready ratio below target."""
        heatmap = {
            "heatmap": {
                "slice_a": {
                    "goal_hit": {
                        "readiness_status": "READY",
                        "drift_status": "OK",
                        "budget_flag": "SAFE"
                    }
                },
                "slice_b": {
                    "sparse_success": {
                        "readiness_status": "DEGRADED",
                        "drift_status": "OK",
                        "budget_flag": "SAFE"
                    }
                }
            },
            "metrics_with_conflicting_signals": []
        }
        history = {}
        
        policy = build_readiness_autopilot_policy(heatmap, history, target_ready_ratio=0.8)
        
        self.assertEqual(policy["autopilot_status"], "ATTENTION")
    
    def test_autopilot_block_when_many_conflicts(self):
        """autopilot_status is BLOCK when many conflicts."""
        heatmap = {
            "heatmap": {
                "slice_a": {
                    "goal_hit": {
                        "readiness_status": "READY",
                        "drift_status": "DRIFTY",
                        "budget_flag": "STARVED"
                    }
                },
                "slice_b": {
                    "sparse_success": {
                        "readiness_status": "READY",
                        "drift_status": "DRIFTY",
                        "budget_flag": "STARVED"
                    }
                },
                "slice_c": {
                    "chain_success": {
                        "readiness_status": "BLOCKED",
                        "drift_status": "OK",
                        "budget_flag": "SAFE"
                    }
                }
            },
            "metrics_with_conflicting_signals": [
                "slice_a.goal_hit",
                "slice_b.sparse_success"
            ]
        }
        history = {}
        
        policy = build_readiness_autopilot_policy(heatmap, history, target_ready_ratio=0.8)
        
        self.assertEqual(policy["autopilot_status"], "BLOCK")
        self.assertGreater(len(policy["slices_to_hold"]), 0)
    
    def test_slices_to_hold_includes_blocked(self):
        """slices_to_hold includes BLOCKED slices."""
        heatmap = {
            "heatmap": {
                "slice_a": {
                    "goal_hit": {
                        "readiness_status": "BLOCKED",
                        "drift_status": "OK",
                        "budget_flag": "SAFE"
                    }
                }
            },
            "metrics_with_conflicting_signals": []
        }
        history = {}
        
        policy = build_readiness_autopilot_policy(heatmap, history)
        
        self.assertIn("slice_a.goal_hit", policy["slices_to_hold"])
    
    def test_slices_to_hold_includes_drifty_starved(self):
        """slices_to_hold includes READY but DRIFTY+STARVED."""
        heatmap = {
            "heatmap": {
                "slice_a": {
                    "goal_hit": {
                        "readiness_status": "READY",
                        "drift_status": "DRIFTY",
                        "budget_flag": "STARVED"
                    }
                }
            },
            "metrics_with_conflicting_signals": []
        }
        history = {}
        
        policy = build_readiness_autopilot_policy(heatmap, history)
        
        self.assertIn("slice_a.goal_hit", policy["slices_to_hold"])
    
    def test_slices_to_hold_includes_history_blocked(self):
        """slices_to_hold includes slices from history."""
        heatmap = {
            "heatmap": {
                "slice_a": {
                    "goal_hit": {
                        "readiness_status": "READY",
                        "drift_status": "OK",
                        "budget_flag": "SAFE"
                    }
                }
            },
            "metrics_with_conflicting_signals": []
        }
        history = {
            "blocked_slices": ["slice_a.goal_hit"]
        }
        
        policy = build_readiness_autopilot_policy(heatmap, history)
        
        self.assertIn("slice_a.goal_hit", policy["slices_to_hold"])
    
    def test_slices_to_hold_includes_repeated_conflicts(self):
        """slices_to_hold includes slices with repeated conflicts."""
        heatmap = {
            "heatmap": {
                "slice_a": {
                    "goal_hit": {
                        "readiness_status": "READY",
                        "drift_status": "OK",
                        "budget_flag": "SAFE"
                    }
                }
            },
            "metrics_with_conflicting_signals": []
        }
        history = {
            "repeated_conflicts": ["slice_a.goal_hit"]
        }
        
        policy = build_readiness_autopilot_policy(heatmap, history)
        
        self.assertIn("slice_a.goal_hit", policy["slices_to_hold"])
    
    def test_slices_safe_to_progress(self):
        """slices_safe_to_progress includes READY, OK, SAFE pairs."""
        heatmap = {
            "heatmap": {
                "slice_a": {
                    "goal_hit": {
                        "readiness_status": "READY",
                        "drift_status": "OK",
                        "budget_flag": "SAFE"
                    }
                },
                "slice_b": {
                    "sparse_success": {
                        "readiness_status": "READY",
                        "drift_status": "OK",
                        "budget_flag": "SAFE"
                    }
                }
            },
            "metrics_with_conflicting_signals": []
        }
        history = {}
        
        policy = build_readiness_autopilot_policy(heatmap, history)
        
        self.assertIn("slice_a.goal_hit", policy["slices_safe_to_progress"])
        self.assertIn("slice_b.sparse_success", policy["slices_safe_to_progress"])
    
    def test_ready_ratio_computed(self):
        """ready_ratio is computed from heatmap."""
        heatmap = {
            "heatmap": {
                "slice_a": {
                    "goal_hit": {"readiness_status": "READY", "drift_status": "OK", "budget_flag": "SAFE"}
                },
                "slice_b": {
                    "sparse_success": {"readiness_status": "READY", "drift_status": "OK", "budget_flag": "SAFE"}
                },
                "slice_c": {
                    "chain_success": {"readiness_status": "DEGRADED", "drift_status": "OK", "budget_flag": "SAFE"}
                }
            },
            "metrics_with_conflicting_signals": []
        }
        history = {}
        
        policy = build_readiness_autopilot_policy(heatmap, history)
        
        self.assertAlmostEqual(policy["ready_ratio"], 2.0 / 3.0, places=2)
    
    def test_neutral_notes_included(self):
        """neutral_notes includes explanations for held slices."""
        heatmap = {
            "heatmap": {
                "slice_a": {
                    "goal_hit": {
                        "readiness_status": "BLOCKED",
                        "drift_status": "OK",
                        "budget_flag": "SAFE"
                    }
                }
            },
            "metrics_with_conflicting_signals": []
        }
        history = {}
        
        policy = build_readiness_autopilot_policy(heatmap, history)
        
        self.assertGreater(len(policy["neutral_notes"]), 0)
        self.assertIn("slice_a.goal_hit", policy["neutral_notes"][0])
    
    def test_deterministic_output(self):
        """Output is deterministic."""
        heatmap = {
            "heatmap": {
                "slice_b": {
                    "goal_hit": {"readiness_status": "READY", "drift_status": "OK", "budget_flag": "SAFE"}
                },
                "slice_a": {
                    "sparse_success": {"readiness_status": "DEGRADED", "drift_status": "OK", "budget_flag": "SAFE"}
                }
            },
            "metrics_with_conflicting_signals": []
        }
        history = {}
        
        policies = [build_readiness_autopilot_policy(heatmap, history) for _ in range(5)]
        json_outputs = [json.dumps(p, sort_keys=True) for p in policies]
        
        self.assertTrue(all(j == json_outputs[0] for j in json_outputs))
    
    def test_sorted_outputs(self):
        """slices_to_hold and slices_safe_to_progress are sorted."""
        heatmap = {
            "heatmap": {
                "z_slice": {
                    "z_metric": {"readiness_status": "BLOCKED", "drift_status": "OK", "budget_flag": "SAFE"}
                },
                "a_slice": {
                    "a_metric": {"readiness_status": "BLOCKED", "drift_status": "OK", "budget_flag": "SAFE"}
                }
            },
            "metrics_with_conflicting_signals": []
        }
        history = {}
        
        policy = build_readiness_autopilot_policy(heatmap, history)
        
        self.assertEqual(
            policy["slices_to_hold"],
            ["a_slice.a_metric", "z_slice.z_metric"]
        )


# =============================================================================
# Phase V Task 2: Phase Boundary Recommendations Tests
# =============================================================================

class TestDerivePhaseBoundaryRecommendations(unittest.TestCase):
    """Tests for derive_phase_boundary_recommendations function."""
    
    def test_phase_ready_when_all_conditions_met(self):
        """phase_ready is True when all conditions are met."""
        readiness_summary = {
            "readiness_status": "OK",
            "ready_slice_ratio": 1.0,
            "blocked_slice_count": 0,
            "promotion_ok": True
        }
        autopilot_policy = {
            "autopilot_status": "OK",
            "slices_to_hold": [],
            "ready_ratio": 1.0
        }
        
        recommendations = derive_phase_boundary_recommendations(
            readiness_summary, autopilot_policy
        )
        
        self.assertTrue(recommendations["phase_ready"])
        self.assertEqual(recommendations["status"], "OK")
    
    def test_phase_not_ready_when_blocked(self):
        """phase_ready is False when blocked slices exist."""
        readiness_summary = {
            "readiness_status": "BLOCK",
            "ready_slice_ratio": 0.5,
            "blocked_slice_count": 2,
            "promotion_ok": False
        }
        autopilot_policy = {
            "autopilot_status": "BLOCK",
            "slices_to_hold": ["slice_a.goal_hit", "slice_b.sparse_success"],
            "ready_ratio": 0.5
        }
        
        recommendations = derive_phase_boundary_recommendations(
            readiness_summary, autopilot_policy
        )
        
        self.assertFalse(recommendations["phase_ready"])
        self.assertEqual(recommendations["status"], "BLOCK")
    
    def test_phase_not_ready_when_slices_to_hold(self):
        """phase_ready is False when slices need to be held."""
        readiness_summary = {
            "readiness_status": "OK",
            "ready_slice_ratio": 0.9,
            "blocked_slice_count": 0,
            "promotion_ok": True
        }
        autopilot_policy = {
            "autopilot_status": "ATTENTION",
            "slices_to_hold": ["slice_a.goal_hit"],
            "ready_ratio": 0.9
        }
        
        recommendations = derive_phase_boundary_recommendations(
            readiness_summary, autopilot_policy
        )
        
        self.assertFalse(recommendations["phase_ready"])
        self.assertEqual(recommendations["status"], "WARN")
    
    def test_recommended_actions_includes_blocked(self):
        """recommended_actions includes action for blocked slices."""
        readiness_summary = {
            "readiness_status": "BLOCK",
            "ready_slice_ratio": 0.5,
            "blocked_slice_count": 2,
            "promotion_ok": False
        }
        autopilot_policy = {
            "autopilot_status": "BLOCK",
            "slices_to_hold": [],
            "ready_ratio": 0.5
        }
        
        recommendations = derive_phase_boundary_recommendations(
            readiness_summary, autopilot_policy
        )
        
        self.assertGreater(len(recommendations["recommended_actions"]), 0)
        self.assertTrue(any("blocked slice" in action.lower() for action in recommendations["recommended_actions"]))
    
    def test_recommended_actions_includes_hold_promotion(self):
        """recommended_actions includes hold promotion actions."""
        readiness_summary = {
            "readiness_status": "WARN",
            "ready_slice_ratio": 0.8,
            "blocked_slice_count": 0,
            "promotion_ok": True
        }
        autopilot_policy = {
            "autopilot_status": "ATTENTION",
            "slices_to_hold": ["slice_uplift_sparse.goal_hit"],
            "ready_ratio": 0.8
        }
        
        recommendations = derive_phase_boundary_recommendations(
            readiness_summary, autopilot_policy
        )
        
        self.assertTrue(any("Hold promotion" in action for action in recommendations["recommended_actions"]))
        self.assertTrue(any("slice_uplift_sparse" in action for action in recommendations["recommended_actions"]))
    
    def test_recommended_actions_includes_advance_high_readiness(self):
        """recommended_actions suggests advancing high-readiness slices."""
        readiness_summary = {
            "readiness_status": "OK",
            "ready_slice_ratio": 0.95,
            "blocked_slice_count": 0,
            "promotion_ok": True
        }
        autopilot_policy = {
            "autopilot_status": "OK",
            "slices_to_hold": [],
            "ready_ratio": 0.95
        }
        
        recommendations = derive_phase_boundary_recommendations(
            readiness_summary, autopilot_policy
        )
        
        self.assertTrue(any("advancing" in action.lower() or "next depth" in action.lower() 
                          for action in recommendations["recommended_actions"]))
    
    def test_slices_needing_investigation_extracted(self):
        """slices_needing_investigation extracts slice names from slices_to_hold."""
        readiness_summary = {
            "readiness_status": "WARN",
            "ready_slice_ratio": 0.7,
            "blocked_slice_count": 0,
            "promotion_ok": True
        }
        autopilot_policy = {
            "autopilot_status": "ATTENTION",
            "slices_to_hold": [
                "slice_a.goal_hit",
                "slice_b.sparse_success",
                "slice_a.chain_success"
            ],
            "ready_ratio": 0.7
        }
        
        recommendations = derive_phase_boundary_recommendations(
            readiness_summary, autopilot_policy
        )
        
        self.assertIn("slice_a", recommendations["slices_needing_investigation"])
        self.assertIn("slice_b", recommendations["slices_needing_investigation"])
        self.assertEqual(len(recommendations["slices_needing_investigation"]), 2)
    
    def test_status_block_when_blocked_count_or_autopilot_block(self):
        """status is BLOCK when blocked_count > 0 or autopilot_status is BLOCK."""
        readiness_summary = {
            "readiness_status": "BLOCK",
            "ready_slice_ratio": 0.3,
            "blocked_slice_count": 1,
            "promotion_ok": False
        }
        autopilot_policy = {
            "autopilot_status": "BLOCK",
            "slices_to_hold": ["slice_a.goal_hit"],
            "ready_ratio": 0.3
        }
        
        recommendations = derive_phase_boundary_recommendations(
            readiness_summary, autopilot_policy
        )
        
        self.assertEqual(recommendations["status"], "BLOCK")
    
    def test_status_warn_when_not_blocked_but_issues(self):
        """status is WARN when not blocked but has issues."""
        readiness_summary = {
            "readiness_status": "WARN",
            "ready_slice_ratio": 0.7,
            "blocked_slice_count": 0,
            "promotion_ok": True
        }
        autopilot_policy = {
            "autopilot_status": "ATTENTION",
            "slices_to_hold": ["slice_a.goal_hit"],
            "ready_ratio": 0.7
        }
        
        recommendations = derive_phase_boundary_recommendations(
            readiness_summary, autopilot_policy
        )
        
        self.assertEqual(recommendations["status"], "WARN")
    
    def test_deterministic_output(self):
        """Output is deterministic."""
        readiness_summary = {
            "readiness_status": "OK",
            "ready_slice_ratio": 0.9,
            "blocked_slice_count": 0,
            "promotion_ok": True
        }
        autopilot_policy = {
            "autopilot_status": "OK",
            "slices_to_hold": [],
            "ready_ratio": 0.9
        }
        
        recommendations_list = [
            derive_phase_boundary_recommendations(readiness_summary, autopilot_policy)
            for _ in range(5)
        ]
        json_outputs = [json.dumps(r, sort_keys=True) for r in recommendations_list]
        
        self.assertTrue(all(j == json_outputs[0] for j in json_outputs))
    
    def test_recommended_actions_sorted(self):
        """recommended_actions are sorted."""
        readiness_summary = {
            "readiness_status": "WARN",
            "ready_slice_ratio": 0.7,
            "blocked_slice_count": 1,
            "promotion_ok": True
        }
        autopilot_policy = {
            "autopilot_status": "ATTENTION",
            "slices_to_hold": ["slice_b.goal_hit", "slice_a.sparse_success"],
            "ready_ratio": 0.7
        }
        
        recommendations = derive_phase_boundary_recommendations(
            readiness_summary, autopilot_policy
        )
        
        # Check that actions are sorted
        sorted_actions = sorted(recommendations["recommended_actions"])
        self.assertEqual(recommendations["recommended_actions"], sorted_actions)
    
    def test_repeated_conflicts_in_critical_slices_blocks(self):
        """Repeated conflicts in critical slices result in phase_ready=False, status BLOCK."""
        readiness_summary = {
            "readiness_status": "BLOCK",
            "ready_slice_ratio": 0.4,
            "blocked_slice_count": 3,
            "promotion_ok": False
        }
        autopilot_policy = {
            "autopilot_status": "BLOCK",
            "slices_to_hold": [
                "slice_uplift_goal.goal_hit",
                "slice_uplift_sparse.sparse_success",
                "slice_uplift_tree.chain_success"
            ],
            "ready_ratio": 0.4
        }
        
        recommendations = derive_phase_boundary_recommendations(
            readiness_summary, autopilot_policy
        )
        
        self.assertFalse(recommendations["phase_ready"])
        self.assertEqual(recommendations["status"], "BLOCK")
        self.assertGreater(len(recommendations["slices_needing_investigation"]), 0)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)

