"""
PHASE II — NOT USED IN PHASE I

Tests for the Pre-Flight DAG Health and Drift Eligibility Checks.

Tests cover:
- CHECK-001..008: Structural health validations
- DRIFT-001..005: Run eligibility comparisons
- PreflightAuditor integration
- CLI functionality

Author: CLAUDE G — DAG Pre-Flight Auditor Engineer
"""
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, Set, Tuple

import sys
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.dag.preflight_check import (
    CheckResult,
    CheckStatus,
    CrossLayerStatus,
    DagSnapshot,
    DriftCheckResult,
    DriftDirection,
    DriftEligibilityResult,
    DriftLedger,
    DriftLedgerEntry,
    EligibilityOracleResult,
    GatingLevel,
    GlobalHealthSummary,
    GovernanceStatus,
    LEDGER_SCHEMA_VERSION,
    MaasStatus,
    POSTURE_SCHEMA_VERSION,
    PreflightAuditor,
    PreflightConfig,
    PreflightReport,
    ReleaseStatus,
    Severity,
    StatusLight,
    build_dag_director_panel,
    build_dag_director_panel_extended,
    build_dag_drift_ledger,
    build_dag_ht_alignment_view,
    build_dag_posture_from_snapshot,
    build_dag_posture_snapshot,
    build_global_health_structure_entry,
    build_structure_console_pane,
    check_001_acyclicity,
    check_002_no_self_loops,
    check_003_hash_integrity,
    check_004_parent_resolution,
    check_005_axiom_set_validity,
    check_006_log_integrity,
    check_007_depth_bounds,
    check_008_temporal_consistency,
    check_dag_topology_consistency,
    check_multilayer_consistency,
    compare_dag_postures,
    compare_posture_files,
    drift_001_axiom_alignment,
    drift_002_vertex_divergence,
    drift_003_edge_divergence,
    drift_004_depth_distribution,
    drift_005_cycle_count,
    evaluate_dag_eligibility,
    evaluate_dag_for_release,
    get_exit_code,
    load_dag_from_jsonl,
    merge_dags,
    summarize_dag_for_maas,
    summarize_dag_posture_for_global_health,
    to_governance_signal_for_structure,
)


class TestDagSnapshot(unittest.TestCase):
    """Tests for DagSnapshot creation and manipulation."""

    def test_empty_dag(self):
        """Test empty DAG snapshot."""
        dag = DagSnapshot(
            vertices=set(),
            edges=set(),
            hash_to_formula={},
            vertex_timestamps={},
            depths={},
            axioms=set(),
            cycle_count=0,
        )
        self.assertEqual(len(dag.vertices), 0)
        self.assertEqual(len(dag.edges), 0)

    def test_simple_dag(self):
        """Test simple DAG with axioms and derived statements."""
        dag = DagSnapshot(
            vertices={"A", "B", "C"},
            edges={("B", "A"), ("C", "B")},  # C <- B <- A
            hash_to_formula={"A": "p", "B": "p -> q", "C": "q"},
            vertex_timestamps={},
            depths={"A": 1, "B": 2, "C": 3},
            axioms={"A"},
            cycle_count=10,
        )
        self.assertEqual(len(dag.vertices), 3)
        self.assertEqual(len(dag.edges), 2)
        self.assertEqual(dag.axioms, {"A"})
        self.assertEqual(dag.cycle_count, 10)


class TestLoadDagFromJsonl(unittest.TestCase):
    """Tests for JSONL log loading."""

    def test_load_empty_file(self):
        """Test loading empty JSONL file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write("")
            path = Path(f.name)

        try:
            dag, errors = load_dag_from_jsonl(path)
            self.assertEqual(len(dag.vertices), 0)
            self.assertEqual(len(errors), 0)
        finally:
            path.unlink()

    def test_load_simple_derivations(self):
        """Test loading simple derivation records."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write(json.dumps({"cycle": 0, "derivations": [
                {"hash": "A", "premises": [], "normalized": "p"},
                {"hash": "B", "premises": ["A"], "normalized": "p -> q"},
            ]}) + "\n")
            f.write(json.dumps({"cycle": 1, "derivations": [
                {"hash": "C", "premises": ["A", "B"], "normalized": "q"},
            ]}) + "\n")
            path = Path(f.name)

        try:
            dag, errors = load_dag_from_jsonl(path)
            self.assertEqual(len(errors), 0)
            self.assertIn("A", dag.vertices)
            self.assertIn("B", dag.vertices)
            self.assertIn("C", dag.vertices)
            self.assertIn(("B", "A"), dag.edges)
            self.assertIn(("C", "A"), dag.edges)
            self.assertIn(("C", "B"), dag.edges)
            self.assertEqual(dag.cycle_count, 2)
        finally:
            path.unlink()

    def test_load_file_not_found(self):
        """Test loading non-existent file."""
        dag, errors = load_dag_from_jsonl(Path("/nonexistent/path.jsonl"))
        self.assertEqual(len(dag.vertices), 0)
        self.assertEqual(len(errors), 1)
        self.assertIn("not found", errors[0].lower())

    def test_load_with_parse_errors(self):
        """Test loading file with JSON parse errors."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write(json.dumps({"cycle": 0, "derivations": []}) + "\n")
            f.write("invalid json line\n")
            f.write(json.dumps({"cycle": 1, "derivations": []}) + "\n")
            path = Path(f.name)

        try:
            dag, errors = load_dag_from_jsonl(path)
            self.assertEqual(len(errors), 1)
            self.assertIn("parse error", errors[0].lower())
        finally:
            path.unlink()


class TestMergeDags(unittest.TestCase):
    """Tests for DAG merging."""

    def test_merge_disjoint_dags(self):
        """Test merging disjoint DAGs."""
        dag1 = DagSnapshot(
            vertices={"A", "B"},
            edges={("B", "A")},
            hash_to_formula={"A": "p", "B": "q"},
            vertex_timestamps={},
            depths={"A": 1, "B": 2},
            axioms={"A"},
            cycle_count=5,
        )
        dag2 = DagSnapshot(
            vertices={"C", "D"},
            edges={("D", "C")},
            hash_to_formula={"C": "r", "D": "s"},
            vertex_timestamps={},
            depths={"C": 1, "D": 2},
            axioms={"C"},
            cycle_count=3,
        )

        merged = merge_dags(dag1, dag2)
        self.assertEqual(len(merged.vertices), 4)
        self.assertEqual(len(merged.edges), 2)
        self.assertEqual(merged.cycle_count, 8)

    def test_merge_overlapping_dags(self):
        """Test merging overlapping DAGs."""
        dag1 = DagSnapshot(
            vertices={"A", "B"},
            edges={("B", "A")},
            hash_to_formula={"A": "p", "B": "q"},
            vertex_timestamps={},
            depths={"A": 1, "B": 2},
            axioms={"A"},
            cycle_count=5,
        )
        dag2 = DagSnapshot(
            vertices={"B", "C"},
            edges={("C", "B")},
            hash_to_formula={"B": "q", "C": "r"},
            vertex_timestamps={},
            depths={"B": 1, "C": 2},
            axioms={"B"},
            cycle_count=3,
        )

        merged = merge_dags(dag1, dag2)
        self.assertEqual(len(merged.vertices), 3)
        self.assertEqual(len(merged.edges), 2)


class TestCheck001Acyclicity(unittest.TestCase):
    """Tests for CHECK-001: Acyclicity."""

    def test_acyclic_dag(self):
        """Test acyclic DAG passes."""
        dag = DagSnapshot(
            vertices={"A", "B", "C"},
            edges={("B", "A"), ("C", "B")},
            hash_to_formula={},
            vertex_timestamps={},
            depths={"A": 1, "B": 2, "C": 3},
            axioms={"A"},
            cycle_count=0,
        )
        result = check_001_acyclicity(dag)
        self.assertEqual(result.id, "CHECK-001")
        self.assertEqual(result.status, CheckStatus.PASS)
        self.assertEqual(result.severity, Severity.CRITICAL)
        self.assertEqual(result.details["cycles_found"], 0)

    def test_cyclic_dag(self):
        """Test cyclic DAG fails."""
        dag = DagSnapshot(
            vertices={"A", "B", "C"},
            edges={("B", "A"), ("C", "B"), ("A", "C")},  # A <- C <- B <- A (cycle)
            hash_to_formula={},
            vertex_timestamps={},
            depths={},
            axioms=set(),
            cycle_count=0,
        )
        result = check_001_acyclicity(dag)
        self.assertEqual(result.status, CheckStatus.FAIL)
        self.assertEqual(result.details["cycles_found"], 1)
        self.assertTrue(len(result.details["cycle_nodes"]) > 0)

    def test_empty_dag(self):
        """Test empty DAG passes."""
        dag = DagSnapshot(
            vertices=set(),
            edges=set(),
            hash_to_formula={},
            vertex_timestamps={},
            depths={},
            axioms=set(),
            cycle_count=0,
        )
        result = check_001_acyclicity(dag)
        self.assertEqual(result.status, CheckStatus.PASS)


class TestCheck002NoSelfLoops(unittest.TestCase):
    """Tests for CHECK-002: No Self-Loops."""

    def test_no_self_loops(self):
        """Test DAG without self-loops passes."""
        dag = DagSnapshot(
            vertices={"A", "B"},
            edges={("B", "A")},
            hash_to_formula={},
            vertex_timestamps={},
            depths={},
            axioms={"A"},
            cycle_count=0,
        )
        result = check_002_no_self_loops(dag)
        self.assertEqual(result.status, CheckStatus.PASS)
        self.assertEqual(result.details["self_loops_found"], 0)

    def test_with_self_loop(self):
        """Test DAG with self-loop fails."""
        dag = DagSnapshot(
            vertices={"A", "B"},
            edges={("A", "A"), ("B", "A")},  # A references itself
            hash_to_formula={},
            vertex_timestamps={},
            depths={},
            axioms=set(),
            cycle_count=0,
        )
        result = check_002_no_self_loops(dag)
        self.assertEqual(result.status, CheckStatus.FAIL)
        self.assertEqual(result.details["self_loops_found"], 1)
        self.assertIn("A", result.details["self_loop_vertices"])


class TestCheck003HashIntegrity(unittest.TestCase):
    """Tests for CHECK-003: Hash Integrity."""

    def test_unique_hashes(self):
        """Test unique hash-formula mappings pass."""
        dag = DagSnapshot(
            vertices={"A", "B"},
            edges={("B", "A")},
            hash_to_formula={"A": "p", "B": "q"},
            vertex_timestamps={},
            depths={},
            axioms={"A"},
            cycle_count=0,
        )
        result = check_003_hash_integrity(dag)
        self.assertEqual(result.status, CheckStatus.PASS)
        self.assertEqual(result.details["collisions_found"], 0)

    def test_no_formulas(self):
        """Test DAG with no formula mappings passes."""
        dag = DagSnapshot(
            vertices={"A", "B"},
            edges={("B", "A")},
            hash_to_formula={},
            vertex_timestamps={},
            depths={},
            axioms={"A"},
            cycle_count=0,
        )
        result = check_003_hash_integrity(dag)
        self.assertEqual(result.status, CheckStatus.PASS)


class TestCheck004ParentResolution(unittest.TestCase):
    """Tests for CHECK-004: Parent Resolution."""

    def test_all_parents_resolve(self):
        """Test all parents resolve to known vertices."""
        dag = DagSnapshot(
            vertices={"A", "B", "C"},
            edges={("B", "A"), ("C", "B")},
            hash_to_formula={},
            vertex_timestamps={},
            depths={},
            axioms={"A"},
            cycle_count=0,
        )
        result = check_004_parent_resolution(dag)
        self.assertEqual(result.status, CheckStatus.PASS)
        self.assertEqual(result.details["dangling_found"], 0)

    def test_dangling_parent(self):
        """Test dangling parent reference fails."""
        dag = DagSnapshot(
            vertices={"A", "B"},
            edges={("B", "A"), ("B", "X")},  # X is not in vertices
            hash_to_formula={},
            vertex_timestamps={},
            depths={},
            axioms={"A"},
            cycle_count=0,
        )
        result = check_004_parent_resolution(dag)
        self.assertEqual(result.status, CheckStatus.FAIL)
        self.assertEqual(result.details["dangling_found"], 1)

    def test_dangling_with_tolerance(self):
        """Test dangling parent within tolerance warns."""
        dag = DagSnapshot(
            vertices={"A", "B"},
            edges={("B", "A"), ("B", "X")},
            hash_to_formula={},
            vertex_timestamps={},
            depths={},
            axioms={"A"},
            cycle_count=0,
        )
        result = check_004_parent_resolution(dag, dangling_tolerance=1)
        self.assertEqual(result.status, CheckStatus.WARN)

    def test_dangling_resolved_by_manifest(self):
        """Test dangling parent resolved by axiom manifest."""
        dag = DagSnapshot(
            vertices={"A", "B"},
            edges={("B", "A"), ("B", "X")},
            hash_to_formula={},
            vertex_timestamps={},
            depths={},
            axioms={"A"},
            cycle_count=0,
        )
        result = check_004_parent_resolution(dag, axiom_manifest={"X"})
        self.assertEqual(result.status, CheckStatus.PASS)
        self.assertEqual(result.details["resolved_to_axioms"], 1)


class TestCheck005AxiomSetValidity(unittest.TestCase):
    """Tests for CHECK-005: Axiom Set Validity."""

    def test_valid_json_manifest(self):
        """Test valid JSON axiom manifest."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"axioms": [
                "a" * 64,  # Valid SHA256 hash
                "b" * 64,
            ]}, f)
            path = Path(f.name)

        try:
            result = check_005_axiom_set_validity(path)
            self.assertEqual(result.status, CheckStatus.PASS)
            self.assertEqual(result.details["axiom_count"], 2)
        finally:
            path.unlink()

    def test_missing_manifest(self):
        """Test missing axiom manifest fails."""
        result = check_005_axiom_set_validity(Path("/nonexistent/manifest.json"))
        self.assertEqual(result.status, CheckStatus.FAIL)

    def test_no_manifest_provided(self):
        """Test no manifest path warns."""
        result = check_005_axiom_set_validity(None)
        self.assertEqual(result.status, CheckStatus.WARN)

    def test_invalid_hash_format(self):
        """Test invalid hash format warns."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"axioms": [
                "a" * 64,  # Valid
                "invalid",  # Invalid
            ]}, f)
            path = Path(f.name)

        try:
            result = check_005_axiom_set_validity(path)
            self.assertEqual(result.status, CheckStatus.WARN)
            self.assertEqual(result.details["axiom_count"], 1)
            self.assertEqual(result.details["invalid_entries"], 1)
        finally:
            path.unlink()


class TestCheck006LogIntegrity(unittest.TestCase):
    """Tests for CHECK-006: Log File Integrity."""

    def test_valid_logs(self):
        """Test valid log files pass."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write(json.dumps({"cycle": 0, "derivations": []}) + "\n")
            f.write(json.dumps({"cycle": 1, "derivations": []}) + "\n")
            baseline_path = Path(f.name)

        try:
            result = check_006_log_integrity(baseline_path)
            self.assertEqual(result.status, CheckStatus.PASS)
            self.assertTrue(result.details["baseline"]["exists"])
            self.assertEqual(result.details["baseline"]["parse_errors"], 0)
        finally:
            baseline_path.unlink()

    def test_missing_log(self):
        """Test missing log file fails."""
        result = check_006_log_integrity(Path("/nonexistent/log.jsonl"))
        self.assertEqual(result.status, CheckStatus.FAIL)

    def test_no_cycle_records(self):
        """Test log with no cycle records fails."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write(json.dumps({"data": "no cycle field"}) + "\n")
            path = Path(f.name)

        try:
            result = check_006_log_integrity(path)
            self.assertEqual(result.status, CheckStatus.FAIL)
            self.assertEqual(result.details["baseline"]["cycle_records"], 0)
        finally:
            path.unlink()


class TestCheck007DepthBounds(unittest.TestCase):
    """Tests for CHECK-007: Depth Bound Compliance."""

    def test_within_bounds(self):
        """Test depths within bounds pass."""
        dag = DagSnapshot(
            vertices={"A", "B", "C"},
            edges={("B", "A"), ("C", "B")},
            hash_to_formula={},
            vertex_timestamps={},
            depths={"A": 1, "B": 2, "C": 3},
            axioms={"A"},
            cycle_count=0,
        )
        result = check_007_depth_bounds(dag, max_configured_depth=5)
        self.assertEqual(result.status, CheckStatus.PASS)

    def test_exceeded_bounds_warn(self):
        """Test exceeded bounds warns."""
        dag = DagSnapshot(
            vertices={"A", "B", "C"},
            edges={},
            hash_to_formula={},
            vertex_timestamps={},
            depths={"A": 1, "B": 2, "C": 6},  # 6 > 5
            axioms={"A"},
            cycle_count=0,
        )
        result = check_007_depth_bounds(dag, max_configured_depth=5, depth_tolerance=2)
        self.assertEqual(result.status, CheckStatus.WARN)

    def test_severely_exceeded_fails(self):
        """Test severely exceeded bounds fails."""
        dag = DagSnapshot(
            vertices={"A", "B", "C"},
            edges={},
            hash_to_formula={},
            vertex_timestamps={},
            depths={"A": 1, "B": 2, "C": 10},  # 10 > 5 + 2
            axioms={"A"},
            cycle_count=0,
        )
        result = check_007_depth_bounds(dag, max_configured_depth=5, depth_tolerance=2)
        self.assertEqual(result.status, CheckStatus.FAIL)

    def test_no_configured_depth_skips(self):
        """Test no configured depth skips check."""
        dag = DagSnapshot(
            vertices={"A"},
            edges=set(),
            hash_to_formula={},
            vertex_timestamps={},
            depths={"A": 100},
            axioms={"A"},
            cycle_count=0,
        )
        result = check_007_depth_bounds(dag, max_configured_depth=None)
        self.assertEqual(result.status, CheckStatus.PASS)
        self.assertTrue(result.details.get("skipped"))


class TestCheck008TemporalConsistency(unittest.TestCase):
    """Tests for CHECK-008: Temporal Consistency."""

    def test_consistent_timestamps(self):
        """Test consistent timestamps pass."""
        dag = DagSnapshot(
            vertices={"A", "B"},
            edges={("B", "A")},
            hash_to_formula={},
            vertex_timestamps={"A": 1.0, "B": 2.0},  # B after A
            depths={},
            axioms={"A"},
            cycle_count=0,
        )
        result = check_008_temporal_consistency(dag)
        self.assertEqual(result.status, CheckStatus.PASS)
        self.assertEqual(result.details["violations_found"], 0)

    def test_inconsistent_timestamps(self):
        """Test inconsistent timestamps warn."""
        dag = DagSnapshot(
            vertices={"A", "B"},
            edges={("B", "A")},
            hash_to_formula={},
            vertex_timestamps={"A": 2.0, "B": 1.0},  # B before A (violation)
            depths={},
            axioms={"A"},
            cycle_count=0,
        )
        result = check_008_temporal_consistency(dag)
        self.assertEqual(result.status, CheckStatus.WARN)
        self.assertEqual(result.details["violations_found"], 1)

    def test_no_timestamps_skips(self):
        """Test no timestamps skips check."""
        dag = DagSnapshot(
            vertices={"A", "B"},
            edges={("B", "A")},
            hash_to_formula={},
            vertex_timestamps={},
            depths={},
            axioms={"A"},
            cycle_count=0,
        )
        result = check_008_temporal_consistency(dag)
        self.assertEqual(result.status, CheckStatus.PASS)
        self.assertTrue(result.details.get("skipped"))


class TestDrift001AxiomAlignment(unittest.TestCase):
    """Tests for DRIFT-001: Axiom Set Alignment."""

    def test_identical_axioms(self):
        """Test identical axiom sets pass."""
        baseline = DagSnapshot(
            vertices={"A", "B"},
            edges={("B", "A")},
            hash_to_formula={},
            vertex_timestamps={},
            depths={},
            axioms={"A"},
            cycle_count=0,
        )
        rfl = DagSnapshot(
            vertices={"A", "B"},
            edges={("B", "A")},
            hash_to_formula={},
            vertex_timestamps={},
            depths={},
            axioms={"A"},
            cycle_count=0,
        )
        result = drift_001_axiom_alignment(baseline, rfl)
        self.assertEqual(result.status, CheckStatus.PASS)

    def test_different_axiom_usage(self):
        """Test different axiom usage warns."""
        baseline = DagSnapshot(
            vertices={"A", "B"},
            edges={("B", "A")},
            hash_to_formula={},
            vertex_timestamps={},
            depths={},
            axioms={"A"},
            cycle_count=0,
        )
        rfl = DagSnapshot(
            vertices={"A", "B", "C"},
            edges={("B", "A"), ("C", "B")},
            hash_to_formula={},
            vertex_timestamps={},
            depths={},
            axioms={"A", "X"},  # Different axiom
            cycle_count=0,
        )
        result = drift_001_axiom_alignment(baseline, rfl)
        self.assertEqual(result.status, CheckStatus.WARN)


class TestDrift002VertexDivergence(unittest.TestCase):
    """Tests for DRIFT-002: Vertex Set Divergence."""

    def test_identical_vertices(self):
        """Test identical vertex sets pass."""
        baseline = DagSnapshot(
            vertices={"A", "B", "C"},
            edges=set(),
            hash_to_formula={},
            vertex_timestamps={},
            depths={},
            axioms=set(),
            cycle_count=0,
        )
        rfl = DagSnapshot(
            vertices={"A", "B", "C"},
            edges=set(),
            hash_to_formula={},
            vertex_timestamps={},
            depths={},
            axioms=set(),
            cycle_count=0,
        )
        result = drift_002_vertex_divergence(baseline, rfl)
        self.assertEqual(result.status, CheckStatus.PASS)
        self.assertEqual(result.metric_value, 0.0)

    def test_high_divergence_fails(self):
        """Test high vertex divergence fails."""
        baseline = DagSnapshot(
            vertices={"A", "B"},
            edges=set(),
            hash_to_formula={},
            vertex_timestamps={},
            depths={},
            axioms=set(),
            cycle_count=0,
        )
        rfl = DagSnapshot(
            vertices={"X", "Y", "Z"},
            edges=set(),
            hash_to_formula={},
            vertex_timestamps={},
            depths={},
            axioms=set(),
            cycle_count=0,
        )
        result = drift_002_vertex_divergence(baseline, rfl, max_divergence=0.5)
        # Divergence = 5/3 ≈ 1.67 > 2*0.5
        self.assertEqual(result.status, CheckStatus.FAIL)


class TestDrift003EdgeDivergence(unittest.TestCase):
    """Tests for DRIFT-003: Edge Set Divergence."""

    def test_identical_edges(self):
        """Test identical edge sets pass."""
        baseline = DagSnapshot(
            vertices={"A", "B"},
            edges={("B", "A")},
            hash_to_formula={},
            vertex_timestamps={},
            depths={},
            axioms=set(),
            cycle_count=0,
        )
        rfl = DagSnapshot(
            vertices={"A", "B"},
            edges={("B", "A")},
            hash_to_formula={},
            vertex_timestamps={},
            depths={},
            axioms=set(),
            cycle_count=0,
        )
        result = drift_003_edge_divergence(baseline, rfl)
        self.assertEqual(result.status, CheckStatus.PASS)


class TestDrift004DepthDistribution(unittest.TestCase):
    """Tests for DRIFT-004: Depth Distribution Alignment."""

    def test_similar_depths(self):
        """Test similar max depths pass."""
        baseline = DagSnapshot(
            vertices={"A", "B"},
            edges=set(),
            hash_to_formula={},
            vertex_timestamps={},
            depths={"A": 1, "B": 5},
            axioms=set(),
            cycle_count=0,
        )
        rfl = DagSnapshot(
            vertices={"A", "B"},
            edges=set(),
            hash_to_formula={},
            vertex_timestamps={},
            depths={"A": 1, "B": 6},
            axioms=set(),
            cycle_count=0,
        )
        result = drift_004_depth_distribution(baseline, rfl, max_difference=3)
        self.assertEqual(result.status, CheckStatus.PASS)

    def test_large_depth_difference_fails(self):
        """Test large depth difference fails."""
        baseline = DagSnapshot(
            vertices={"A"},
            edges=set(),
            hash_to_formula={},
            vertex_timestamps={},
            depths={"A": 5},
            axioms=set(),
            cycle_count=0,
        )
        rfl = DagSnapshot(
            vertices={"A"},
            edges=set(),
            hash_to_formula={},
            vertex_timestamps={},
            depths={"A": 15},
            axioms=set(),
            cycle_count=0,
        )
        result = drift_004_depth_distribution(baseline, rfl, max_difference=3)
        # Difference = 10 > 2*3 = 6
        self.assertEqual(result.status, CheckStatus.FAIL)


class TestDrift005CycleCount(unittest.TestCase):
    """Tests for DRIFT-005: Cycle Count Alignment."""

    def test_identical_cycle_counts(self):
        """Test identical cycle counts pass."""
        baseline = DagSnapshot(
            vertices=set(),
            edges=set(),
            hash_to_formula={},
            vertex_timestamps={},
            depths={},
            axioms=set(),
            cycle_count=100,
        )
        rfl = DagSnapshot(
            vertices=set(),
            edges=set(),
            hash_to_formula={},
            vertex_timestamps={},
            depths={},
            axioms=set(),
            cycle_count=100,
        )
        result = drift_005_cycle_count(baseline, rfl)
        self.assertEqual(result.status, CheckStatus.PASS)

    def test_within_tolerance_warns(self):
        """Test cycle count within tolerance warns."""
        baseline = DagSnapshot(
            vertices=set(),
            edges=set(),
            hash_to_formula={},
            vertex_timestamps={},
            depths={},
            axioms=set(),
            cycle_count=100,
        )
        rfl = DagSnapshot(
            vertices=set(),
            edges=set(),
            hash_to_formula={},
            vertex_timestamps={},
            depths={},
            axioms=set(),
            cycle_count=105,
        )
        result = drift_005_cycle_count(baseline, rfl, tolerance=10)
        self.assertEqual(result.status, CheckStatus.WARN)

    def test_exceeds_tolerance_fails(self):
        """Test cycle count exceeds tolerance fails."""
        baseline = DagSnapshot(
            vertices=set(),
            edges=set(),
            hash_to_formula={},
            vertex_timestamps={},
            depths={},
            axioms=set(),
            cycle_count=100,
        )
        rfl = DagSnapshot(
            vertices=set(),
            edges=set(),
            hash_to_formula={},
            vertex_timestamps={},
            depths={},
            axioms=set(),
            cycle_count=150,
        )
        result = drift_005_cycle_count(baseline, rfl, tolerance=10)
        self.assertEqual(result.status, CheckStatus.FAIL)


class TestPreflightAuditor(unittest.TestCase):
    """Tests for PreflightAuditor integration."""

    def test_run_health_checks(self):
        """Test running all health checks."""
        dag = DagSnapshot(
            vertices={"A", "B", "C"},
            edges={("B", "A"), ("C", "B")},
            hash_to_formula={"A": "p", "B": "q", "C": "r"},
            vertex_timestamps={},
            depths={"A": 1, "B": 2, "C": 3},
            axioms={"A"},
            cycle_count=0,
        )
        auditor = PreflightAuditor()
        checks = auditor.run_health_checks(dag)

        self.assertGreater(len(checks), 0)
        check_ids = [c.id for c in checks]
        self.assertIn("CHECK-001", check_ids)
        self.assertIn("CHECK-002", check_ids)
        self.assertIn("CHECK-003", check_ids)
        self.assertIn("CHECK-004", check_ids)

    def test_run_drift_checks(self):
        """Test running all drift checks."""
        baseline = DagSnapshot(
            vertices={"A", "B"},
            edges={("B", "A")},
            hash_to_formula={},
            vertex_timestamps={},
            depths={"A": 1, "B": 2},
            axioms={"A"},
            cycle_count=10,
        )
        rfl = DagSnapshot(
            vertices={"A", "B", "C"},
            edges={("B", "A"), ("C", "B")},
            hash_to_formula={},
            vertex_timestamps={},
            depths={"A": 1, "B": 2, "C": 3},
            axioms={"A"},
            cycle_count=12,
        )
        auditor = PreflightAuditor()
        result = auditor.run_drift_checks(baseline, rfl)

        self.assertIsInstance(result, DriftEligibilityResult)
        self.assertEqual(len(result.drift_checks), 5)
        drift_ids = [c.id for c in result.drift_checks]
        self.assertIn("DRIFT-001", drift_ids)
        self.assertIn("DRIFT-002", drift_ids)
        self.assertIn("DRIFT-003", drift_ids)
        self.assertIn("DRIFT-004", drift_ids)
        self.assertIn("DRIFT-005", drift_ids)

    def test_full_preflight_with_valid_logs(self):
        """Test full pre-flight with valid logs."""
        # Create test log files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write(json.dumps({"cycle": 0, "derivations": [
                {"hash": "A", "premises": [], "normalized": "p"},
                {"hash": "B", "premises": ["A"], "normalized": "q"},
            ]}) + "\n")
            baseline_path = Path(f.name)

        try:
            auditor = PreflightAuditor()
            report = auditor.run_full_preflight(baseline_path)

            self.assertIsInstance(report, PreflightReport)
            self.assertEqual(report.preflight_version, "1.0.0")
            self.assertIn("baseline_log", report.inputs)
            self.assertIn("overall_status", report.summary)
        finally:
            baseline_path.unlink()


class TestPreflightConfig(unittest.TestCase):
    """Tests for PreflightConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PreflightConfig()
        self.assertEqual(config.scope, "FULL")
        self.assertEqual(config.dangling_tolerance, 0)
        self.assertEqual(config.max_vertex_divergence, 0.5)

    def test_from_dict(self):
        """Test loading config from dictionary."""
        data = {
            "preflight": {
                "scope": "BOUNDED",
                "dangling_tolerance": 5,
            },
            "drift": {
                "max_vertex_divergence": 0.3,
                "cycle_tolerance": 20,
            },
        }
        config = PreflightConfig.from_dict(data)
        self.assertEqual(config.scope, "BOUNDED")
        self.assertEqual(config.dangling_tolerance, 5)
        self.assertEqual(config.max_vertex_divergence, 0.3)
        self.assertEqual(config.cycle_tolerance, 20)


class TestGetExitCode(unittest.TestCase):
    """Tests for exit code determination."""

    def test_all_pass_returns_zero(self):
        """Test all pass returns exit code 0."""
        report = PreflightReport(
            preflight_version="1.0.0",
            timestamp="2025-01-01T00:00:00Z",
            label="test",
            inputs={},
            checks=[],
            drift_eligibility=None,
            summary={
                "overall_status": "PASS",
                "critical_failures": 0,
                "errors": 0,
                "warnings": 0,
                "audit_eligible": True,
            },
        )
        self.assertEqual(get_exit_code(report), 0)

    def test_warnings_returns_one(self):
        """Test warnings returns exit code 1."""
        report = PreflightReport(
            preflight_version="1.0.0",
            timestamp="2025-01-01T00:00:00Z",
            label="test",
            inputs={},
            checks=[],
            drift_eligibility=None,
            summary={
                "overall_status": "WARN",
                "critical_failures": 0,
                "errors": 0,
                "warnings": 2,
                "audit_eligible": True,
            },
        )
        self.assertEqual(get_exit_code(report), 1)

    def test_critical_failures_returns_two(self):
        """Test critical failures returns exit code 2."""
        report = PreflightReport(
            preflight_version="1.0.0",
            timestamp="2025-01-01T00:00:00Z",
            label="test",
            inputs={},
            checks=[],
            drift_eligibility=None,
            summary={
                "overall_status": "FAIL",
                "critical_failures": 1,
                "errors": 0,
                "warnings": 0,
                "audit_eligible": False,
            },
        )
        self.assertEqual(get_exit_code(report), 2)

    def test_errors_returns_two(self):
        """Test errors returns exit code 2."""
        report = PreflightReport(
            preflight_version="1.0.0",
            timestamp="2025-01-01T00:00:00Z",
            label="test",
            inputs={},
            checks=[],
            drift_eligibility=None,
            summary={
                "overall_status": "FAIL",
                "critical_failures": 0,
                "errors": 1,
                "warnings": 0,
                "audit_eligible": False,
            },
        )
        self.assertEqual(get_exit_code(report), 2)


class TestCheckResultSerialization(unittest.TestCase):
    """Tests for CheckResult serialization."""

    def test_to_dict(self):
        """Test CheckResult serialization."""
        result = CheckResult(
            id="CHECK-001",
            name="Test Check",
            status=CheckStatus.PASS,
            severity=Severity.CRITICAL,
            details={"key": "value"},
        )
        d = result.to_dict()
        self.assertEqual(d["id"], "CHECK-001")
        self.assertEqual(d["status"], "PASS")
        self.assertEqual(d["severity"], "CRITICAL")
        self.assertEqual(d["details"]["key"], "value")


class TestDriftCheckResultSerialization(unittest.TestCase):
    """Tests for DriftCheckResult serialization."""

    def test_to_dict(self):
        """Test DriftCheckResult serialization."""
        result = DriftCheckResult(
            id="DRIFT-001",
            name="Test Drift",
            status=CheckStatus.WARN,
            metric_value=0.35,
            threshold=0.5,
            details={"info": "test"},
        )
        d = result.to_dict()
        self.assertEqual(d["id"], "DRIFT-001")
        self.assertEqual(d["status"], "WARN")
        self.assertEqual(d["metric_value"], 0.35)
        self.assertEqual(d["threshold"], 0.5)


class TestPreflightReportSerialization(unittest.TestCase):
    """Tests for PreflightReport serialization."""

    def test_to_dict(self):
        """Test PreflightReport serialization."""
        check = CheckResult(
            id="CHECK-001",
            name="Test",
            status=CheckStatus.PASS,
            severity=Severity.CRITICAL,
        )
        drift = DriftEligibilityResult(
            eligible=True,
            reasons=["All pass"],
            drift_checks=[],
        )
        report = PreflightReport(
            preflight_version="1.0.0",
            timestamp="2025-01-01T00:00:00Z",
            label="test",
            inputs={"baseline": "test.jsonl"},
            checks=[check],
            drift_eligibility=drift,
            summary={"overall_status": "PASS"},
        )
        d = report.to_dict()
        self.assertEqual(d["preflight_version"], "1.0.0")
        self.assertIn("CHECK-001", d["checks"])
        self.assertTrue(d["drift_eligibility"]["eligible"])


# =============================================================================
# TASK 1: DAG Posture Snapshot Tests
# =============================================================================

class TestBuildDagPostureSnapshot(unittest.TestCase):
    """Tests for build_dag_posture_snapshot function."""

    def _make_report(
        self,
        has_cycles: bool = False,
        max_depth: int = 5,
        vertex_count: int = 10,
        edge_count: int = 8,
        drift_eligible: bool = True,
        drift_reason: str = None,
    ) -> PreflightReport:
        """Create a test PreflightReport."""
        checks = [
            CheckResult(
                id="CHECK-001",
                name="Acyclicity",
                status=CheckStatus.FAIL if has_cycles else CheckStatus.PASS,
                severity=Severity.CRITICAL,
                details={
                    "vertices_checked": vertex_count,
                    "cycles_found": 1 if has_cycles else 0,
                    "cycle_nodes": [],
                },
            ),
            CheckResult(
                id="CHECK-002",
                name="No Self-Loops",
                status=CheckStatus.PASS,
                severity=Severity.CRITICAL,
                details={
                    "edges_checked": edge_count,
                    "self_loops_found": 0,
                },
            ),
            CheckResult(
                id="CHECK-007",
                name="Depth Bounds",
                status=CheckStatus.PASS,
                severity=Severity.WARNING,
                details={
                    "actual_max_depth": max_depth,
                    "skipped": False,
                },
            ),
        ]

        drift_eligibility = None
        if drift_reason or not drift_eligible:
            reasons = [drift_reason] if drift_reason else ["FAIL: Some check failed"]
            drift_eligibility = DriftEligibilityResult(
                eligible=drift_eligible,
                reasons=reasons,
                drift_checks=[],
            )

        return PreflightReport(
            preflight_version="1.0.0",
            timestamp="2025-01-01T00:00:00Z",
            label="test",
            inputs={},
            checks=checks,
            drift_eligibility=drift_eligibility,
            summary={
                "overall_status": "PASS" if not has_cycles else "FAIL",
                "critical_failures": 1 if has_cycles else 0,
                "errors": 0,
                "warnings": 0,
                "audit_eligible": not has_cycles,
            },
        )

    def test_basic_posture_extraction(self):
        """Test basic posture snapshot extraction from report."""
        report = self._make_report(
            has_cycles=False,
            max_depth=5,
            vertex_count=10,
            edge_count=8,
        )
        posture = build_dag_posture_snapshot(report)

        self.assertEqual(posture["schema_version"], POSTURE_SCHEMA_VERSION)
        self.assertFalse(posture["has_cycles"])
        self.assertEqual(posture["max_depth"], 5)
        self.assertEqual(posture["vertex_count"], 10)
        self.assertEqual(posture["edge_count"], 8)
        self.assertTrue(posture["drift_eligible"])

    def test_posture_with_cycles(self):
        """Test posture snapshot when cycles detected."""
        report = self._make_report(has_cycles=True)
        posture = build_dag_posture_snapshot(report)

        self.assertTrue(posture["has_cycles"])

    def test_posture_with_drift_ineligible(self):
        """Test posture snapshot when drift ineligible."""
        report = self._make_report(
            drift_eligible=False,
            drift_reason="FAIL: DRIFT-002 - Vertex Divergence",
        )
        posture = build_dag_posture_snapshot(report)

        self.assertFalse(posture["drift_eligible"])
        self.assertIn("drift_ineligibility_reason", posture)
        self.assertIn("DRIFT-002", posture["drift_ineligibility_reason"])

    def test_posture_schema_version_present(self):
        """Test that schema version is always present."""
        report = self._make_report()
        posture = build_dag_posture_snapshot(report)

        self.assertIn("schema_version", posture)
        self.assertEqual(posture["schema_version"], "1.0.0")

    def test_posture_json_serializable(self):
        """Test that posture snapshot is JSON serializable."""
        report = self._make_report()
        posture = build_dag_posture_snapshot(report)

        # Should not raise
        json_str = json.dumps(posture, sort_keys=True)
        self.assertIsInstance(json_str, str)

        # Should round-trip
        parsed = json.loads(json_str)
        self.assertEqual(parsed, posture)

    def test_posture_key_ordering_stable(self):
        """Test that posture snapshot has stable key ordering."""
        report = self._make_report()
        posture1 = build_dag_posture_snapshot(report)
        posture2 = build_dag_posture_snapshot(report)

        # Keys should be in same order
        self.assertEqual(list(posture1.keys()), list(posture2.keys()))


class TestBuildDagPostureFromSnapshot(unittest.TestCase):
    """Tests for build_dag_posture_from_snapshot function."""

    def test_basic_dag_posture(self):
        """Test posture from simple DAG snapshot."""
        dag = DagSnapshot(
            vertices={"A", "B", "C"},
            edges={("B", "A"), ("C", "B")},
            hash_to_formula={},
            vertex_timestamps={},
            depths={"A": 1, "B": 2, "C": 3},
            axioms={"A"},
            cycle_count=0,
        )
        posture = build_dag_posture_from_snapshot(dag)

        self.assertEqual(posture["schema_version"], POSTURE_SCHEMA_VERSION)
        self.assertFalse(posture["has_cycles"])
        self.assertEqual(posture["max_depth"], 3)
        self.assertEqual(posture["vertex_count"], 3)
        self.assertEqual(posture["edge_count"], 2)
        self.assertTrue(posture["drift_eligible"])

    def test_dag_with_cycle(self):
        """Test posture from DAG with cycle."""
        dag = DagSnapshot(
            vertices={"A", "B", "C"},
            edges={("B", "A"), ("C", "B"), ("A", "C")},  # Cycle
            hash_to_formula={},
            vertex_timestamps={},
            depths={},
            axioms=set(),
            cycle_count=0,
        )
        posture = build_dag_posture_from_snapshot(dag)

        self.assertTrue(posture["has_cycles"])


# =============================================================================
# TASK 2: Cross-Run DAG Drift Radar Tests
# =============================================================================

class TestCompareDagPostures(unittest.TestCase):
    """Tests for compare_dag_postures function."""

    def test_identical_postures(self):
        """Test comparison of identical postures."""
        posture = {
            "schema_version": "1.0.0",
            "has_cycles": False,
            "max_depth": 5,
            "vertex_count": 10,
            "edge_count": 8,
            "drift_eligible": True,
        }
        comparison = compare_dag_postures(posture, posture)

        self.assertEqual(comparison["depth_delta"], 0)
        self.assertEqual(comparison["vertex_count_delta"], 0)
        self.assertEqual(comparison["edge_count_delta"], 0)
        self.assertEqual(comparison["drift_eligibility_change"], "STABLE_ELIGIBLE")
        self.assertFalse(comparison["cycle_status_changed"])
        self.assertTrue(comparison["schema_compatible"])

    def test_posture_growth(self):
        """Test comparison when DAG grows."""
        old = {
            "schema_version": "1.0.0",
            "has_cycles": False,
            "max_depth": 3,
            "vertex_count": 10,
            "edge_count": 8,
            "drift_eligible": True,
        }
        new = {
            "schema_version": "1.0.0",
            "has_cycles": False,
            "max_depth": 5,
            "vertex_count": 20,
            "edge_count": 18,
            "drift_eligible": True,
        }
        comparison = compare_dag_postures(old, new)

        self.assertEqual(comparison["depth_delta"], 2)
        self.assertEqual(comparison["vertex_count_delta"], 10)
        self.assertEqual(comparison["edge_count_delta"], 10)

    def test_posture_shrink(self):
        """Test comparison when DAG shrinks (regression)."""
        old = {
            "schema_version": "1.0.0",
            "has_cycles": False,
            "max_depth": 10,
            "vertex_count": 100,
            "edge_count": 80,
            "drift_eligible": True,
        }
        new = {
            "schema_version": "1.0.0",
            "has_cycles": False,
            "max_depth": 5,
            "vertex_count": 50,
            "edge_count": 40,
            "drift_eligible": True,
        }
        comparison = compare_dag_postures(old, new)

        self.assertEqual(comparison["depth_delta"], -5)
        self.assertEqual(comparison["vertex_count_delta"], -50)
        self.assertEqual(comparison["edge_count_delta"], -40)

    def test_cycle_status_change_acyclic_to_cyclic(self):
        """Test detection of cycle status change (acyclic -> cyclic)."""
        old = {
            "schema_version": "1.0.0",
            "has_cycles": False,
            "max_depth": 5,
            "vertex_count": 10,
            "edge_count": 8,
            "drift_eligible": True,
        }
        new = {
            "schema_version": "1.0.0",
            "has_cycles": True,
            "max_depth": 5,
            "vertex_count": 10,
            "edge_count": 9,
            "drift_eligible": True,
        }
        comparison = compare_dag_postures(old, new)

        self.assertTrue(comparison["cycle_status_changed"])
        self.assertEqual(comparison["cycle_transition"], "ACYCLIC -> CYCLIC")

    def test_cycle_status_change_cyclic_to_acyclic(self):
        """Test detection of cycle status change (cyclic -> acyclic)."""
        old = {
            "schema_version": "1.0.0",
            "has_cycles": True,
            "max_depth": 5,
            "vertex_count": 10,
            "edge_count": 9,
            "drift_eligible": False,
        }
        new = {
            "schema_version": "1.0.0",
            "has_cycles": False,
            "max_depth": 5,
            "vertex_count": 10,
            "edge_count": 8,
            "drift_eligible": True,
        }
        comparison = compare_dag_postures(old, new)

        self.assertTrue(comparison["cycle_status_changed"])
        self.assertEqual(comparison["cycle_transition"], "CYCLIC -> ACYCLIC")

    def test_eligibility_change_eligible_to_ineligible(self):
        """Test detection of eligibility change (eligible -> ineligible)."""
        old = {
            "schema_version": "1.0.0",
            "has_cycles": False,
            "max_depth": 5,
            "vertex_count": 10,
            "edge_count": 8,
            "drift_eligible": True,
        }
        new = {
            "schema_version": "1.0.0",
            "has_cycles": False,
            "max_depth": 5,
            "vertex_count": 10,
            "edge_count": 8,
            "drift_eligible": False,
            "drift_ineligibility_reason": "FAIL: DRIFT-002",
        }
        comparison = compare_dag_postures(old, new)

        self.assertEqual(comparison["drift_eligibility_change"], "ELIGIBLE_TO_INELIGIBLE")
        self.assertIn("eligibility_reasons", comparison)
        self.assertIsNone(comparison["eligibility_reasons"]["old"])
        self.assertEqual(comparison["eligibility_reasons"]["new"], "FAIL: DRIFT-002")

    def test_eligibility_change_ineligible_to_eligible(self):
        """Test detection of eligibility change (ineligible -> eligible)."""
        old = {
            "schema_version": "1.0.0",
            "has_cycles": False,
            "max_depth": 5,
            "vertex_count": 10,
            "edge_count": 8,
            "drift_eligible": False,
            "drift_ineligibility_reason": "FAIL: Some issue",
        }
        new = {
            "schema_version": "1.0.0",
            "has_cycles": False,
            "max_depth": 5,
            "vertex_count": 10,
            "edge_count": 8,
            "drift_eligible": True,
        }
        comparison = compare_dag_postures(old, new)

        self.assertEqual(comparison["drift_eligibility_change"], "INELIGIBLE_TO_ELIGIBLE")

    def test_stable_ineligible(self):
        """Test stable ineligible status detection."""
        old = {
            "schema_version": "1.0.0",
            "has_cycles": False,
            "max_depth": 5,
            "vertex_count": 10,
            "edge_count": 8,
            "drift_eligible": False,
        }
        new = {
            "schema_version": "1.0.0",
            "has_cycles": False,
            "max_depth": 6,
            "vertex_count": 12,
            "edge_count": 10,
            "drift_eligible": False,
        }
        comparison = compare_dag_postures(old, new)

        self.assertEqual(comparison["drift_eligibility_change"], "STABLE_INELIGIBLE")

    def test_schema_incompatibility(self):
        """Test schema incompatibility detection."""
        old = {
            "schema_version": "1.0.0",
            "has_cycles": False,
            "max_depth": 5,
            "vertex_count": 10,
            "edge_count": 8,
            "drift_eligible": True,
        }
        new = {
            "schema_version": "2.0.0",  # Major version change
            "has_cycles": False,
            "max_depth": 5,
            "vertex_count": 10,
            "edge_count": 8,
            "drift_eligible": True,
        }
        comparison = compare_dag_postures(old, new)

        self.assertFalse(comparison["schema_compatible"])

    def test_schema_minor_version_compatible(self):
        """Test that minor version changes are compatible."""
        old = {
            "schema_version": "1.0.0",
            "has_cycles": False,
            "max_depth": 5,
            "vertex_count": 10,
            "edge_count": 8,
            "drift_eligible": True,
        }
        new = {
            "schema_version": "1.1.0",  # Minor version change
            "has_cycles": False,
            "max_depth": 5,
            "vertex_count": 10,
            "edge_count": 8,
            "drift_eligible": True,
        }
        comparison = compare_dag_postures(old, new)

        self.assertTrue(comparison["schema_compatible"])

    def test_comparison_key_ordering_stable(self):
        """Test that comparison result has stable key ordering."""
        old = {
            "schema_version": "1.0.0",
            "has_cycles": False,
            "max_depth": 5,
            "vertex_count": 10,
            "edge_count": 8,
            "drift_eligible": True,
        }
        new = {
            "schema_version": "1.0.0",
            "has_cycles": False,
            "max_depth": 6,
            "vertex_count": 15,
            "edge_count": 12,
            "drift_eligible": True,
        }
        comparison1 = compare_dag_postures(old, new)
        comparison2 = compare_dag_postures(old, new)

        # Core keys should be present and in consistent order
        core_keys = ["depth_delta", "vertex_count_delta", "edge_count_delta",
                     "drift_eligibility_change", "cycle_status_changed", "schema_compatible"]
        for key in core_keys:
            self.assertIn(key, comparison1)
            self.assertIn(key, comparison2)

    def test_comparison_json_serializable(self):
        """Test that comparison result is JSON serializable."""
        old = {
            "schema_version": "1.0.0",
            "has_cycles": False,
            "max_depth": 5,
            "vertex_count": 10,
            "edge_count": 8,
            "drift_eligible": True,
        }
        new = {
            "schema_version": "1.0.0",
            "has_cycles": True,
            "max_depth": 6,
            "vertex_count": 15,
            "edge_count": 12,
            "drift_eligible": False,
            "drift_ineligibility_reason": "FAIL: Test",
        }
        comparison = compare_dag_postures(old, new)

        # Should not raise
        json_str = json.dumps(comparison, sort_keys=True)
        self.assertIsInstance(json_str, str)

        # Should round-trip
        parsed = json.loads(json_str)
        self.assertEqual(parsed["depth_delta"], comparison["depth_delta"])

    def test_missing_fields_handled(self):
        """Test that missing fields are handled gracefully."""
        old = {"schema_version": "1.0.0"}  # Minimal posture
        new = {
            "schema_version": "1.0.0",
            "has_cycles": False,
            "max_depth": 5,
            "vertex_count": 10,
            "edge_count": 8,
            "drift_eligible": True,
        }
        comparison = compare_dag_postures(old, new)

        # Should not raise, and should use defaults
        self.assertEqual(comparison["depth_delta"], 5)  # 5 - 0
        self.assertEqual(comparison["vertex_count_delta"], 10)  # 10 - 0


class TestComparePostureFiles(unittest.TestCase):
    """Tests for compare_posture_files function."""

    def test_compare_valid_files(self):
        """Test comparing two valid posture files."""
        old_posture = {
            "schema_version": "1.0.0",
            "has_cycles": False,
            "max_depth": 5,
            "vertex_count": 10,
            "edge_count": 8,
            "drift_eligible": True,
        }
        new_posture = {
            "schema_version": "1.0.0",
            "has_cycles": False,
            "max_depth": 7,
            "vertex_count": 15,
            "edge_count": 13,
            "drift_eligible": True,
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(old_posture, f)
            old_path = Path(f.name)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(new_posture, f)
            new_path = Path(f.name)

        try:
            comparison = compare_posture_files(old_path, new_path)
            self.assertEqual(comparison["depth_delta"], 2)
            self.assertEqual(comparison["vertex_count_delta"], 5)
        finally:
            old_path.unlink()
            new_path.unlink()

    def test_compare_nonexistent_file(self):
        """Test that comparing with nonexistent file raises."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"schema_version": "1.0.0"}, f)
            existing_path = Path(f.name)

        try:
            with self.assertRaises(FileNotFoundError):
                compare_posture_files(Path("/nonexistent.json"), existing_path)
        finally:
            existing_path.unlink()


class TestDriftDirection(unittest.TestCase):
    """Tests for DriftDirection enum."""

    def test_enum_values(self):
        """Test that all expected enum values exist."""
        self.assertEqual(DriftDirection.STABLE_ELIGIBLE.value, "STABLE_ELIGIBLE")
        self.assertEqual(DriftDirection.STABLE_INELIGIBLE.value, "STABLE_INELIGIBLE")
        self.assertEqual(DriftDirection.ELIGIBLE_TO_INELIGIBLE.value, "ELIGIBLE_TO_INELIGIBLE")
        self.assertEqual(DriftDirection.INELIGIBLE_TO_ELIGIBLE.value, "INELIGIBLE_TO_ELIGIBLE")


# =============================================================================
# PHASE III: DAG Drift Ledger & Eligibility Oracle Tests
# =============================================================================

class TestGatingLevel(unittest.TestCase):
    """Tests for GatingLevel enum."""

    def test_enum_values(self):
        """Test that all expected gating levels exist."""
        self.assertEqual(GatingLevel.OK.value, "OK")
        self.assertEqual(GatingLevel.WARN.value, "WARN")
        self.assertEqual(GatingLevel.BLOCK.value, "BLOCK")


class TestDriftLedgerEntry(unittest.TestCase):
    """Tests for DriftLedgerEntry dataclass."""

    def test_basic_entry(self):
        """Test basic entry creation."""
        posture = {"schema_version": "1.0.0", "max_depth": 5}
        entry = DriftLedgerEntry(
            index=0,
            posture=posture,
            comparison=None,
            cumulative_drift_score=0.0,
            timestamp="2025-01-01T00:00:00Z",
            label="baseline",
        )
        self.assertEqual(entry.index, 0)
        self.assertEqual(entry.posture, posture)
        self.assertIsNone(entry.comparison)
        self.assertEqual(entry.cumulative_drift_score, 0.0)

    def test_entry_to_dict(self):
        """Test entry serialization."""
        entry = DriftLedgerEntry(
            index=1,
            posture={"max_depth": 5},
            comparison={"depth_delta": 2},
            cumulative_drift_score=0.5,
            timestamp="2025-01-01T01:00:00Z",
            label="run-1",
        )
        d = entry.to_dict()
        self.assertEqual(d["index"], 1)
        self.assertEqual(d["posture"]["max_depth"], 5)
        self.assertEqual(d["comparison"]["depth_delta"], 2)
        self.assertEqual(d["cumulative_drift_score"], 0.5)


class TestBuildDagDriftLedger(unittest.TestCase):
    """Tests for build_dag_drift_ledger function."""

    def _make_posture(
        self,
        max_depth: int = 5,
        vertex_count: int = 10,
        edge_count: int = 8,
        drift_eligible: bool = True,
        has_cycles: bool = False,
    ) -> Dict[str, Any]:
        """Helper to create posture snapshots."""
        return {
            "schema_version": "1.0.0",
            "has_cycles": has_cycles,
            "max_depth": max_depth,
            "vertex_count": vertex_count,
            "edge_count": edge_count,
            "drift_eligible": drift_eligible,
        }

    def test_empty_snapshots(self):
        """Test ledger from empty snapshot list."""
        ledger = build_dag_drift_ledger([])
        self.assertEqual(ledger.schema_version, LEDGER_SCHEMA_VERSION)
        self.assertEqual(len(ledger.entries), 0)
        self.assertEqual(ledger.trends["depth"], [])
        self.assertEqual(len(ledger.eligibility_transitions), 0)

    def test_single_snapshot(self):
        """Test ledger with single snapshot."""
        posture = self._make_posture(max_depth=5, vertex_count=10, edge_count=8)
        ledger = build_dag_drift_ledger([posture])

        self.assertEqual(len(ledger.entries), 1)
        self.assertEqual(ledger.entries[0].index, 0)
        self.assertIsNone(ledger.entries[0].comparison)
        self.assertEqual(ledger.entries[0].cumulative_drift_score, 0.0)
        self.assertEqual(ledger.trends["depth"], [5])
        self.assertEqual(ledger.trends["vertex_count"], [10])

    def test_multiple_snapshots_growth(self):
        """Test ledger with growing DAG."""
        postures = [
            self._make_posture(max_depth=3, vertex_count=10, edge_count=8),
            self._make_posture(max_depth=4, vertex_count=15, edge_count=12),
            self._make_posture(max_depth=5, vertex_count=20, edge_count=18),
        ]
        ledger = build_dag_drift_ledger(postures)

        self.assertEqual(len(ledger.entries), 3)
        self.assertEqual(ledger.trends["depth"], [3, 4, 5])
        self.assertEqual(ledger.trends["vertex_count"], [10, 15, 20])
        self.assertEqual(ledger.trends["depth_delta_total"], 2)
        self.assertEqual(ledger.trends["vertex_delta_total"], 10)

        # Check cumulative drift grows
        self.assertGreater(ledger.entries[2].cumulative_drift_score, 0)

    def test_eligibility_transition_tracking(self):
        """Test that eligibility transitions are tracked."""
        postures = [
            self._make_posture(drift_eligible=True),
            self._make_posture(drift_eligible=False),  # Transition
            self._make_posture(drift_eligible=True),   # Transition back
        ]
        ledger = build_dag_drift_ledger(
            postures,
            labels=["baseline", "run-1", "run-2"],
        )

        self.assertEqual(len(ledger.eligibility_transitions), 2)
        self.assertEqual(ledger.eligibility_transitions[0]["direction"], "ELIGIBLE_TO_INELIGIBLE")
        self.assertEqual(ledger.eligibility_transitions[0]["from_index"], 0)
        self.assertEqual(ledger.eligibility_transitions[0]["to_index"], 1)
        self.assertEqual(ledger.eligibility_transitions[1]["direction"], "INELIGIBLE_TO_ELIGIBLE")

    def test_timestamps_and_labels(self):
        """Test that timestamps and labels are recorded."""
        postures = [self._make_posture(), self._make_posture()]
        timestamps = ["2025-01-01T00:00:00Z", "2025-01-01T01:00:00Z"]
        labels = ["baseline", "run-1"]

        ledger = build_dag_drift_ledger(postures, timestamps=timestamps, labels=labels)

        self.assertEqual(ledger.entries[0].timestamp, timestamps[0])
        self.assertEqual(ledger.entries[0].label, labels[0])
        self.assertEqual(ledger.entries[1].timestamp, timestamps[1])
        self.assertEqual(ledger.entries[1].label, labels[1])

    def test_cumulative_drift_score(self):
        """Test cumulative drift score calculation."""
        postures = [
            self._make_posture(max_depth=5, vertex_count=10, edge_count=8),
            self._make_posture(max_depth=15, vertex_count=110, edge_count=108),  # Big jump
            self._make_posture(max_depth=16, vertex_count=115, edge_count=112),  # Small change
        ]
        ledger = build_dag_drift_ledger(postures)

        # First entry has 0 drift score
        self.assertEqual(ledger.entries[0].cumulative_drift_score, 0.0)
        # Second entry has larger drift score (big jump)
        self.assertGreater(ledger.entries[1].cumulative_drift_score, 0)
        # Third entry accumulates more
        self.assertGreater(ledger.entries[2].cumulative_drift_score, ledger.entries[1].cumulative_drift_score)

    def test_trend_statistics(self):
        """Test trend statistics in ledger."""
        postures = [
            self._make_posture(max_depth=3, vertex_count=10, edge_count=8),
            self._make_posture(max_depth=5, vertex_count=20, edge_count=18),
            self._make_posture(max_depth=4, vertex_count=15, edge_count=12),
        ]
        ledger = build_dag_drift_ledger(postures)

        self.assertEqual(ledger.trends["depth_min"], 3)
        self.assertEqual(ledger.trends["depth_max"], 5)
        self.assertEqual(ledger.trends["vertex_min"], 10)
        self.assertEqual(ledger.trends["vertex_max"], 20)
        self.assertGreater(ledger.trends["cumulative_drift"], 0)
        self.assertGreater(ledger.trends["average_drift"], 0)

    def test_ledger_json_serializable(self):
        """Test that ledger is JSON serializable."""
        postures = [
            self._make_posture(max_depth=3),
            self._make_posture(max_depth=5),
        ]
        ledger = build_dag_drift_ledger(postures, labels=["a", "b"])

        # Should not raise
        d = ledger.to_dict()
        json_str = json.dumps(d)
        self.assertIsInstance(json_str, str)

        # Should round-trip
        parsed = json.loads(json_str)
        self.assertEqual(parsed["schema_version"], LEDGER_SCHEMA_VERSION)
        self.assertEqual(len(parsed["entries"]), 2)


class TestEvaluateDagEligibility(unittest.TestCase):
    """Tests for evaluate_dag_eligibility function."""

    def _make_posture(
        self,
        max_depth: int = 5,
        vertex_count: int = 100,
        edge_count: int = 80,
        drift_eligible: bool = True,
        has_cycles: bool = False,
    ) -> Dict[str, Any]:
        """Helper to create posture snapshots."""
        return {
            "schema_version": "1.0.0",
            "has_cycles": has_cycles,
            "max_depth": max_depth,
            "vertex_count": vertex_count,
            "edge_count": edge_count,
            "drift_eligible": drift_eligible,
        }

    def test_healthy_transition_ok(self):
        """Test healthy transition returns OK."""
        old = self._make_posture(max_depth=5, vertex_count=100, edge_count=80)
        new = self._make_posture(max_depth=6, vertex_count=105, edge_count=84)

        result = evaluate_dag_eligibility(old, new)

        self.assertEqual(result.eligibility_status, "ELIGIBLE")
        self.assertEqual(result.gating_level, GatingLevel.OK)
        self.assertIn("OK: DAG transition is healthy", result.reasons)

    def test_cycle_introduction_blocks(self):
        """Test cycle introduction returns BLOCK."""
        old = self._make_posture(has_cycles=False)
        new = self._make_posture(has_cycles=True)

        result = evaluate_dag_eligibility(old, new)

        self.assertEqual(result.eligibility_status, "INELIGIBLE")
        self.assertEqual(result.gating_level, GatingLevel.BLOCK)
        self.assertTrue(any("Cycle introduced" in r for r in result.reasons))

    def test_cycle_introduction_warn_mode(self):
        """Test cycle introduction with block_on_cycle_introduction=False."""
        old = self._make_posture(has_cycles=False)
        new = self._make_posture(has_cycles=True)

        result = evaluate_dag_eligibility(old, new, block_on_cycle_introduction=False)

        self.assertEqual(result.gating_level, GatingLevel.WARN)

    def test_cycle_resolution_good(self):
        """Test cycle resolution is marked as GOOD."""
        old = self._make_posture(has_cycles=True)
        new = self._make_posture(has_cycles=False)

        result = evaluate_dag_eligibility(old, new)

        self.assertTrue(any("Cycle resolved" in r for r in result.reasons))

    def test_depth_regression_warns(self):
        """Test significant depth regression returns WARN."""
        old = self._make_posture(max_depth=15)
        new = self._make_posture(max_depth=5)  # -10 depth

        result = evaluate_dag_eligibility(old, new, max_depth_regression=5)

        self.assertEqual(result.gating_level, GatingLevel.WARN)
        self.assertTrue(any("depth regression" in r.lower() for r in result.reasons))

    def test_vertex_loss_warns(self):
        """Test significant vertex loss returns WARN."""
        old = self._make_posture(vertex_count=100)
        new = self._make_posture(vertex_count=70)  # -30%

        result = evaluate_dag_eligibility(old, new, max_vertex_loss_pct=0.2)

        self.assertEqual(result.gating_level, GatingLevel.WARN)
        self.assertTrue(any("vertex loss" in r.lower() for r in result.reasons))

    def test_edge_loss_warns(self):
        """Test significant edge loss returns WARN."""
        old = self._make_posture(edge_count=100)
        new = self._make_posture(edge_count=50)  # -50%

        result = evaluate_dag_eligibility(old, new, max_edge_loss_pct=0.3)

        self.assertEqual(result.gating_level, GatingLevel.WARN)
        self.assertTrue(any("edge loss" in r.lower() for r in result.reasons))

    def test_high_drift_score_blocks(self):
        """Test high drift score returns BLOCK."""
        old = self._make_posture(max_depth=5, vertex_count=10, edge_count=8, has_cycles=False)
        new = self._make_posture(max_depth=50, vertex_count=500, edge_count=400, has_cycles=True)

        result = evaluate_dag_eligibility(old, new, max_drift_score=0.5)

        self.assertEqual(result.eligibility_status, "INELIGIBLE")
        self.assertEqual(result.gating_level, GatingLevel.BLOCK)

    def test_eligibility_loss_warns(self):
        """Test drift eligibility loss returns WARN."""
        old = self._make_posture(drift_eligible=True)
        new = self._make_posture(drift_eligible=False)

        result = evaluate_dag_eligibility(old, new)

        self.assertTrue(any("eligibility lost" in r.lower() for r in result.reasons))

    def test_eligibility_restored_good(self):
        """Test drift eligibility restoration is marked GOOD."""
        old = self._make_posture(drift_eligible=False)
        new = self._make_posture(drift_eligible=True)

        result = evaluate_dag_eligibility(old, new)

        self.assertTrue(any("eligibility restored" in r.lower() for r in result.reasons))

    def test_schema_incompatibility_warns(self):
        """Test schema incompatibility returns WARN."""
        old = {"schema_version": "1.0.0", "max_depth": 5, "vertex_count": 10, "edge_count": 8, "drift_eligible": True, "has_cycles": False}
        new = {"schema_version": "2.0.0", "max_depth": 5, "vertex_count": 10, "edge_count": 8, "drift_eligible": True, "has_cycles": False}

        result = evaluate_dag_eligibility(old, new)

        self.assertEqual(result.gating_level, GatingLevel.WARN)
        self.assertTrue(any("Schema version" in r for r in result.reasons))

    def test_result_to_dict(self):
        """Test EligibilityOracleResult serialization."""
        old = self._make_posture()
        new = self._make_posture()
        result = evaluate_dag_eligibility(old, new)

        d = result.to_dict()
        self.assertEqual(d["eligibility_status"], "ELIGIBLE")
        self.assertEqual(d["gating_level"], "OK")
        self.assertIsInstance(d["reasons"], list)
        self.assertIsInstance(d["metrics"], dict)


class TestSummarizeDagPostureForGlobalHealth(unittest.TestCase):
    """Tests for summarize_dag_posture_for_global_health function."""

    def _make_posture(
        self,
        max_depth: int = 5,
        vertex_count: int = 10,
        edge_count: int = 8,
        drift_eligible: bool = True,
        has_cycles: bool = False,
    ) -> Dict[str, Any]:
        """Helper to create posture snapshots."""
        return {
            "schema_version": "1.0.0",
            "has_cycles": has_cycles,
            "max_depth": max_depth,
            "vertex_count": vertex_count,
            "edge_count": edge_count,
            "drift_eligible": drift_eligible,
        }

    def test_empty_ledger_healthy(self):
        """Test empty ledger is healthy."""
        ledger = build_dag_drift_ledger([])
        summary = summarize_dag_posture_for_global_health(ledger)

        self.assertTrue(summary.dag_ok)
        self.assertEqual(len(summary.sustained_regressions), 0)
        self.assertEqual(len(summary.drift_hotspots), 0)
        self.assertEqual(summary.health_score, 1.0)

    def test_single_entry_healthy(self):
        """Test single entry is healthy."""
        ledger = build_dag_drift_ledger([self._make_posture()])
        summary = summarize_dag_posture_for_global_health(ledger)

        self.assertTrue(summary.dag_ok)
        self.assertEqual(summary.summary_metrics["total_entries"], 1)

    def test_growing_dag_healthy(self):
        """Test growing DAG is healthy."""
        postures = [
            self._make_posture(max_depth=3, vertex_count=10, edge_count=8),
            self._make_posture(max_depth=4, vertex_count=15, edge_count=12),
            self._make_posture(max_depth=5, vertex_count=20, edge_count=18),
        ]
        ledger = build_dag_drift_ledger(postures)
        summary = summarize_dag_posture_for_global_health(ledger)

        self.assertTrue(summary.dag_ok)
        self.assertEqual(len(summary.sustained_regressions), 0)

    def test_depth_regression_detected(self):
        """Test sustained depth regression is detected."""
        postures = [
            self._make_posture(max_depth=10),
            self._make_posture(max_depth=8),
            self._make_posture(max_depth=6),
        ]
        ledger = build_dag_drift_ledger(postures)
        summary = summarize_dag_posture_for_global_health(ledger, regression_window=3)

        depth_regressions = [r for r in summary.sustained_regressions if r["metric"] == "depth"]
        self.assertGreater(len(depth_regressions), 0)
        self.assertEqual(depth_regressions[0]["values"], [10, 8, 6])

    def test_vertex_regression_detected(self):
        """Test sustained vertex regression is detected."""
        postures = [
            self._make_posture(vertex_count=100),
            self._make_posture(vertex_count=80),
            self._make_posture(vertex_count=60),
        ]
        ledger = build_dag_drift_ledger(postures)
        summary = summarize_dag_posture_for_global_health(ledger, regression_window=3)

        vertex_regressions = [r for r in summary.sustained_regressions if r["metric"] == "vertex_count"]
        self.assertGreater(len(vertex_regressions), 0)

    def test_drift_hotspots_detected(self):
        """Test drift hotspots are detected."""
        postures = [
            self._make_posture(max_depth=5, vertex_count=10, edge_count=8, has_cycles=False),
            self._make_posture(max_depth=50, vertex_count=200, edge_count=180, has_cycles=True),  # Big jump + cycle
        ]
        ledger = build_dag_drift_ledger(postures, labels=["baseline", "problematic"])
        summary = summarize_dag_posture_for_global_health(ledger, drift_hotspot_threshold=0.5)

        self.assertGreater(len(summary.drift_hotspots), 0)
        self.assertEqual(summary.drift_hotspots[0]["label"], "problematic")

    def test_cycle_introduction_penalizes_health(self):
        """Test cycle introduction reduces health score."""
        postures = [
            self._make_posture(has_cycles=False),
            self._make_posture(has_cycles=True),
        ]
        ledger = build_dag_drift_ledger(postures)
        summary = summarize_dag_posture_for_global_health(ledger)

        self.assertLess(summary.health_score, 1.0)

    def test_recent_cycles_unhealthy(self):
        """Test recent cycles make DAG unhealthy."""
        postures = [
            self._make_posture(has_cycles=False),
            self._make_posture(has_cycles=False),
            self._make_posture(has_cycles=True),  # Recent cycle
        ]
        ledger = build_dag_drift_ledger(postures)
        summary = summarize_dag_posture_for_global_health(ledger)

        self.assertFalse(summary.dag_ok)

    def test_many_regressions_unhealthy(self):
        """Test many sustained regressions make DAG unhealthy."""
        postures = [
            self._make_posture(max_depth=20, vertex_count=100, edge_count=80),
            self._make_posture(max_depth=15, vertex_count=80, edge_count=60),
            self._make_posture(max_depth=10, vertex_count=60, edge_count=40),
            self._make_posture(max_depth=5, vertex_count=40, edge_count=20),
        ]
        ledger = build_dag_drift_ledger(postures)
        summary = summarize_dag_posture_for_global_health(ledger, regression_window=3, max_sustained_regressions=1)

        # With regression_window=3, we should have multiple overlapping regressions
        # which exceeds max_sustained_regressions=1
        self.assertFalse(summary.dag_ok)

    def test_summary_metrics_complete(self):
        """Test summary metrics contain expected fields."""
        postures = [
            self._make_posture(max_depth=3, vertex_count=10, edge_count=8),
            self._make_posture(max_depth=5, vertex_count=15, edge_count=12),
        ]
        ledger = build_dag_drift_ledger(postures)
        summary = summarize_dag_posture_for_global_health(ledger)

        metrics = summary.summary_metrics
        self.assertIn("total_entries", metrics)
        self.assertIn("total_transitions", metrics)
        self.assertIn("eligibility_transitions", metrics)
        self.assertIn("cumulative_drift", metrics)
        self.assertIn("average_drift", metrics)
        self.assertIn("sustained_regression_count", metrics)
        self.assertIn("drift_hotspot_count", metrics)
        self.assertIn("health_penalties", metrics)

        self.assertEqual(metrics["total_entries"], 2)
        self.assertEqual(metrics["total_transitions"], 1)

    def test_global_health_summary_to_dict(self):
        """Test GlobalHealthSummary serialization."""
        ledger = build_dag_drift_ledger([self._make_posture()])
        summary = summarize_dag_posture_for_global_health(ledger)

        d = summary.to_dict()
        self.assertIn("dag_ok", d)
        self.assertIn("sustained_regressions", d)
        self.assertIn("drift_hotspots", d)
        self.assertIn("health_score", d)
        self.assertIn("summary_metrics", d)

    def test_global_health_json_serializable(self):
        """Test GlobalHealthSummary is JSON serializable."""
        postures = [
            self._make_posture(max_depth=5),
            self._make_posture(max_depth=10),
        ]
        ledger = build_dag_drift_ledger(postures)
        summary = summarize_dag_posture_for_global_health(ledger)

        # Should not raise
        json_str = json.dumps(summary.to_dict())
        self.assertIsInstance(json_str, str)

    def test_health_score_bounds(self):
        """Test health score is bounded between 0 and 1."""
        # Very unhealthy scenario
        postures = [
            self._make_posture(max_depth=100, vertex_count=1000, edge_count=800, has_cycles=False),
            self._make_posture(max_depth=50, vertex_count=500, edge_count=400, has_cycles=True),
            self._make_posture(max_depth=25, vertex_count=250, edge_count=200, has_cycles=True),
        ]
        ledger = build_dag_drift_ledger(postures)
        summary = summarize_dag_posture_for_global_health(ledger)

        self.assertGreaterEqual(summary.health_score, 0.0)
        self.assertLessEqual(summary.health_score, 1.0)


class TestDriftLedgerSerialization(unittest.TestCase):
    """Tests for DriftLedger serialization."""

    def test_ledger_to_dict(self):
        """Test DriftLedger.to_dict()."""
        postures = [
            {"schema_version": "1.0.0", "max_depth": 5, "vertex_count": 10, "edge_count": 8, "drift_eligible": True, "has_cycles": False},
            {"schema_version": "1.0.0", "max_depth": 6, "vertex_count": 12, "edge_count": 10, "drift_eligible": True, "has_cycles": False},
        ]
        ledger = build_dag_drift_ledger(postures, labels=["a", "b"])

        d = ledger.to_dict()
        self.assertEqual(d["schema_version"], LEDGER_SCHEMA_VERSION)
        self.assertEqual(len(d["entries"]), 2)
        self.assertIn("trends", d)
        self.assertIn("eligibility_transitions", d)


# =============================================================================
# PHASE IV: Structural Eligibility Gate & Director DAG Panel Tests
# =============================================================================

class TestReleaseStatus(unittest.TestCase):
    """Tests for ReleaseStatus enum."""

    def test_enum_values(self):
        """Test that all expected release statuses exist."""
        self.assertEqual(ReleaseStatus.OK.value, "OK")
        self.assertEqual(ReleaseStatus.WARN.value, "WARN")
        self.assertEqual(ReleaseStatus.BLOCK.value, "BLOCK")


class TestMaasStatus(unittest.TestCase):
    """Tests for MaasStatus enum."""

    def test_enum_values(self):
        """Test that all expected MAAS statuses exist."""
        self.assertEqual(MaasStatus.OK.value, "OK")
        self.assertEqual(MaasStatus.ATTENTION.value, "ATTENTION")
        self.assertEqual(MaasStatus.BLOCK.value, "BLOCK")


class TestStatusLight(unittest.TestCase):
    """Tests for StatusLight enum."""

    def test_enum_values(self):
        """Test that all expected status lights exist."""
        self.assertEqual(StatusLight.GREEN.value, "GREEN")
        self.assertEqual(StatusLight.YELLOW.value, "YELLOW")
        self.assertEqual(StatusLight.RED.value, "RED")


class TestEvaluateDagForRelease(unittest.TestCase):
    """Tests for evaluate_dag_for_release function."""

    def _make_healthy_ledger(self) -> Dict[str, Any]:
        """Create a healthy drift ledger."""
        return {
            "schema_version": "1.0.0",
            "entries": [
                {"index": 0, "posture": {"has_cycles": False, "max_depth": 5}, "comparison": None, "cumulative_drift_score": 0.0},
                {"index": 1, "posture": {"has_cycles": False, "max_depth": 6}, "comparison": {"depth_delta": 1}, "cumulative_drift_score": 0.1},
            ],
            "trends": {"cumulative_drift": 0.1, "depth": [5, 6]},
            "eligibility_transitions": [],
        }

    def _make_healthy_oracle(self) -> Dict[str, Any]:
        """Create a healthy oracle result."""
        return {
            "eligibility_status": "ELIGIBLE",
            "gating_level": "OK",
            "reasons": ["OK: DAG transition is healthy"],
            "metrics": {},
        }

    def test_healthy_release_ok(self):
        """Test healthy DAG returns release OK."""
        ledger = self._make_healthy_ledger()
        oracle = self._make_healthy_oracle()

        result = evaluate_dag_for_release(ledger, oracle)

        self.assertTrue(result["release_ok"])
        self.assertEqual(result["status"], "OK")
        self.assertEqual(len(result["blocking_reasons"]), 0)

    def test_oracle_block_blocks_release(self):
        """Test oracle BLOCK prevents release."""
        ledger = self._make_healthy_ledger()
        oracle = {
            "eligibility_status": "INELIGIBLE",
            "gating_level": "BLOCK",
            "reasons": ["CRITICAL: Cycle introduced in DAG", "CRITICAL: Drift score exceeds threshold"],
            "metrics": {},
        }

        result = evaluate_dag_for_release(ledger, oracle)

        self.assertFalse(result["release_ok"])
        self.assertEqual(result["status"], "BLOCK")
        self.assertGreater(len(result["blocking_reasons"]), 0)
        self.assertTrue(any("Cycle" in r for r in result["blocking_reasons"]))

    def test_oracle_warn_allows_release(self):
        """Test oracle WARN allows release with warnings."""
        ledger = self._make_healthy_ledger()
        oracle = {
            "eligibility_status": "ELIGIBLE",
            "gating_level": "WARN",
            "reasons": ["WARNING: Significant depth regression (-6)"],
            "metrics": {},
        }

        result = evaluate_dag_for_release(ledger, oracle)

        self.assertTrue(result["release_ok"])
        self.assertEqual(result["status"], "WARN")
        self.assertTrue(any("[WARN]" in r for r in result["blocking_reasons"]))

    def test_recent_cycles_block_release(self):
        """Test recent cycles in ledger block release."""
        ledger = {
            "schema_version": "1.0.0",
            "entries": [
                {"index": 0, "posture": {"has_cycles": False}, "comparison": None, "cumulative_drift_score": 0.0},
                {"index": 1, "posture": {"has_cycles": True}, "comparison": {}, "cumulative_drift_score": 1.0},  # Cycle!
            ],
            "trends": {"cumulative_drift": 1.0},
            "eligibility_transitions": [],
        }
        oracle = self._make_healthy_oracle()

        result = evaluate_dag_for_release(ledger, oracle)

        self.assertFalse(result["release_ok"])
        self.assertEqual(result["status"], "BLOCK")
        self.assertTrue(any("cycle" in r.lower() for r in result["blocking_reasons"]))

    def test_high_cumulative_drift_blocks(self):
        """Test high cumulative drift blocks release."""
        ledger = {
            "schema_version": "1.0.0",
            "entries": [{"index": 0, "posture": {"has_cycles": False}, "comparison": None, "cumulative_drift_score": 0.0}],
            "trends": {"cumulative_drift": 6.0},  # > 5.0 threshold
            "eligibility_transitions": [],
        }
        oracle = self._make_healthy_oracle()

        result = evaluate_dag_for_release(ledger, oracle)

        self.assertFalse(result["release_ok"])
        self.assertEqual(result["status"], "BLOCK")
        self.assertTrue(any("drift" in r.lower() for r in result["blocking_reasons"]))

    def test_frequent_eligibility_transitions_blocks(self):
        """Test frequent eligibility transitions block release."""
        ledger = {
            "schema_version": "1.0.0",
            "entries": [{"index": 0, "posture": {"has_cycles": False}, "comparison": None, "cumulative_drift_score": 0.0}],
            "trends": {"cumulative_drift": 0.5},
            "eligibility_transitions": [
                {"from_index": 0, "to_index": 1, "direction": "ELIGIBLE_TO_INELIGIBLE"},
                {"from_index": 1, "to_index": 2, "direction": "INELIGIBLE_TO_ELIGIBLE"},
                {"from_index": 2, "to_index": 3, "direction": "ELIGIBLE_TO_INELIGIBLE"},
            ],
        }
        oracle = self._make_healthy_oracle()

        result = evaluate_dag_for_release(ledger, oracle)

        self.assertFalse(result["release_ok"])
        self.assertEqual(result["status"], "BLOCK")
        self.assertTrue(any("transition" in r.lower() for r in result["blocking_reasons"]))

    def test_result_json_serializable(self):
        """Test release evaluation result is JSON serializable."""
        ledger = self._make_healthy_ledger()
        oracle = self._make_healthy_oracle()

        result = evaluate_dag_for_release(ledger, oracle)

        # Should not raise
        json_str = json.dumps(result)
        self.assertIsInstance(json_str, str)


class TestSummarizeDagForMaas(unittest.TestCase):
    """Tests for summarize_dag_for_maas function."""

    def _make_healthy_oracle(self) -> Dict[str, Any]:
        """Create a healthy oracle result."""
        return {
            "eligibility_status": "ELIGIBLE",
            "gating_level": "OK",
            "reasons": [],
            "metrics": {},
        }

    def _make_healthy_global_health(self) -> Dict[str, Any]:
        """Create a healthy global health summary."""
        return {
            "dag_ok": True,
            "sustained_regressions": [],
            "drift_hotspots": [],
            "health_score": 0.95,
            "summary_metrics": {},
        }

    def test_healthy_dag_maas_ok(self):
        """Test healthy DAG returns MAAS OK."""
        oracle = self._make_healthy_oracle()
        health = self._make_healthy_global_health()

        result = summarize_dag_for_maas(oracle, health)

        self.assertTrue(result["dag_structurally_ok"])
        self.assertFalse(result["has_sustained_regressions"])
        self.assertEqual(result["status"], "OK")

    def test_oracle_block_maas_block(self):
        """Test oracle BLOCK returns MAAS BLOCK."""
        oracle = {
            "eligibility_status": "INELIGIBLE",
            "gating_level": "BLOCK",
            "reasons": [],
            "metrics": {},
        }
        health = self._make_healthy_global_health()

        result = summarize_dag_for_maas(oracle, health)

        self.assertEqual(result["status"], "BLOCK")

    def test_oracle_warn_maas_attention(self):
        """Test oracle WARN returns MAAS ATTENTION."""
        oracle = {
            "eligibility_status": "ELIGIBLE",
            "gating_level": "WARN",
            "reasons": [],
            "metrics": {},
        }
        health = self._make_healthy_global_health()

        result = summarize_dag_for_maas(oracle, health)

        self.assertEqual(result["status"], "ATTENTION")

    def test_sustained_regressions_attention(self):
        """Test sustained regressions return MAAS ATTENTION."""
        oracle = self._make_healthy_oracle()
        health = {
            "dag_ok": True,
            "sustained_regressions": [{"metric": "depth", "values": [10, 8, 6]}],
            "drift_hotspots": [],
            "health_score": 0.8,
            "summary_metrics": {},
        }

        result = summarize_dag_for_maas(oracle, health)

        self.assertTrue(result["has_sustained_regressions"])
        self.assertEqual(result["status"], "ATTENTION")

    def test_drift_hotspots_attention(self):
        """Test drift hotspots return MAAS ATTENTION."""
        oracle = self._make_healthy_oracle()
        health = {
            "dag_ok": True,
            "sustained_regressions": [],
            "drift_hotspots": [{"index": 1, "drift_score": 0.8}],
            "health_score": 0.8,
            "summary_metrics": {},
        }

        result = summarize_dag_for_maas(oracle, health)

        self.assertEqual(result["status"], "ATTENTION")

    def test_low_health_score_attention(self):
        """Test low health score returns MAAS ATTENTION."""
        oracle = self._make_healthy_oracle()
        health = {
            "dag_ok": True,
            "sustained_regressions": [],
            "drift_hotspots": [],
            "health_score": 0.6,  # < 0.7
            "summary_metrics": {},
        }

        result = summarize_dag_for_maas(oracle, health)

        self.assertEqual(result["status"], "ATTENTION")

    def test_dag_not_ok_structurally_not_ok(self):
        """Test dag_ok=False means structurally not OK."""
        oracle = self._make_healthy_oracle()
        health = {
            "dag_ok": False,
            "sustained_regressions": [],
            "drift_hotspots": [],
            "health_score": 0.8,
            "summary_metrics": {},
        }

        result = summarize_dag_for_maas(oracle, health)

        self.assertFalse(result["dag_structurally_ok"])

    def test_ineligible_structurally_not_ok(self):
        """Test INELIGIBLE oracle means structurally not OK."""
        oracle = {
            "eligibility_status": "INELIGIBLE",
            "gating_level": "BLOCK",
            "reasons": [],
            "metrics": {},
        }
        health = self._make_healthy_global_health()

        result = summarize_dag_for_maas(oracle, health)

        self.assertFalse(result["dag_structurally_ok"])

    def test_result_json_serializable(self):
        """Test MAAS summary is JSON serializable."""
        oracle = self._make_healthy_oracle()
        health = self._make_healthy_global_health()

        result = summarize_dag_for_maas(oracle, health)

        # Should not raise
        json_str = json.dumps(result)
        self.assertIsInstance(json_str, str)


class TestBuildDagDirectorPanel(unittest.TestCase):
    """Tests for build_dag_director_panel function."""

    def _make_healthy_drift_summary(self) -> Dict[str, Any]:
        """Create a healthy drift summary."""
        return {
            "dag_ok": True,
            "sustained_regressions": [],
            "drift_hotspots": [],
            "health_score": 0.95,
            "summary_metrics": {},
        }

    def _make_healthy_release_eval(self) -> Dict[str, Any]:
        """Create a healthy release evaluation."""
        return {
            "release_ok": True,
            "status": "OK",
            "blocking_reasons": [],
        }

    def test_healthy_dag_green_light(self):
        """Test healthy DAG shows green status light."""
        drift_summary = self._make_healthy_drift_summary()
        release_eval = self._make_healthy_release_eval()

        panel = build_dag_director_panel(drift_summary, release_eval)

        self.assertEqual(panel["status_light"], "GREEN")
        self.assertEqual(panel["health_score"], 0.95)
        self.assertEqual(len(panel["drift_hotspots"]), 0)
        # 0.95 health score triggers "optimal" headline (>= 0.95)
        self.assertIn("optimal", panel["headline"].lower())

    def test_optimal_headline(self):
        """Test optimal health score generates optimal headline."""
        drift_summary = {
            "dag_ok": True,
            "sustained_regressions": [],
            "drift_hotspots": [],
            "health_score": 0.98,
            "summary_metrics": {},
        }
        release_eval = self._make_healthy_release_eval()

        panel = build_dag_director_panel(drift_summary, release_eval)

        self.assertIn("optimal", panel["headline"].lower())

    def test_blocked_release_red_light(self):
        """Test blocked release shows red status light."""
        drift_summary = self._make_healthy_drift_summary()
        release_eval = {
            "release_ok": False,
            "status": "BLOCK",
            "blocking_reasons": ["Cycle introduced in DAG"],
        }

        panel = build_dag_director_panel(drift_summary, release_eval)

        self.assertEqual(panel["status_light"], "RED")
        self.assertIn("cycle", panel["headline"].lower())

    def test_warn_release_yellow_light(self):
        """Test warn release shows yellow status light."""
        drift_summary = self._make_healthy_drift_summary()
        release_eval = {
            "release_ok": True,
            "status": "WARN",
            "blocking_reasons": ["[WARN] Minor issue"],
        }

        panel = build_dag_director_panel(drift_summary, release_eval)

        self.assertEqual(panel["status_light"], "YELLOW")

    def test_low_health_score_yellow_light(self):
        """Test low health score shows yellow status light."""
        drift_summary = {
            "dag_ok": True,
            "sustained_regressions": [],
            "drift_hotspots": [],
            "health_score": 0.6,  # < 0.7
            "summary_metrics": {},
        }
        release_eval = self._make_healthy_release_eval()

        panel = build_dag_director_panel(drift_summary, release_eval)

        self.assertEqual(panel["status_light"], "YELLOW")
        self.assertIn("below optimal", panel["headline"].lower())

    def test_dag_not_ok_red_light(self):
        """Test dag_ok=False shows red status light."""
        drift_summary = {
            "dag_ok": False,
            "sustained_regressions": [],
            "drift_hotspots": [],
            "health_score": 0.4,
            "summary_metrics": {},
        }
        release_eval = self._make_healthy_release_eval()

        panel = build_dag_director_panel(drift_summary, release_eval)

        self.assertEqual(panel["status_light"], "RED")

    def test_drift_hotspots_included(self):
        """Test drift hotspots are included in panel."""
        drift_summary = {
            "dag_ok": True,
            "sustained_regressions": [],
            "drift_hotspots": [
                {"index": 1, "drift_score": 0.8, "label": "run-1", "timestamp": "2025-01-01"},
                {"index": 2, "drift_score": 0.6, "label": "run-2", "timestamp": "2025-01-02"},
            ],
            "health_score": 0.7,
            "summary_metrics": {},
        }
        release_eval = self._make_healthy_release_eval()

        panel = build_dag_director_panel(drift_summary, release_eval)

        self.assertEqual(len(panel["drift_hotspots"]), 2)
        self.assertEqual(panel["drift_hotspots"][0]["label"], "run-1")
        self.assertEqual(panel["drift_hotspots"][1]["drift_score"], 0.6)

    def test_regressions_headline(self):
        """Test sustained regressions generate appropriate headline."""
        drift_summary = {
            "dag_ok": True,
            "sustained_regressions": [{"metric": "depth", "values": [10, 8, 6]}],
            "drift_hotspots": [],
            "health_score": 0.75,
            "summary_metrics": {},
        }
        release_eval = {
            "release_ok": True,
            "status": "WARN",
            "blocking_reasons": [],
        }

        panel = build_dag_director_panel(drift_summary, release_eval)

        self.assertEqual(panel["status_light"], "YELLOW")
        self.assertIn("regression", panel["headline"].lower())

    def test_hotspot_headline_red(self):
        """Test hotspots with red status generate appropriate headline."""
        drift_summary = {
            "dag_ok": False,
            "sustained_regressions": [],
            "drift_hotspots": [{"index": 1, "drift_score": 1.5, "label": "bad-run"}],
            "health_score": 0.3,
            "summary_metrics": {},
        }
        release_eval = {
            "release_ok": False,
            "status": "BLOCK",
            "blocking_reasons": ["High drift"],
        }

        panel = build_dag_director_panel(drift_summary, release_eval)

        self.assertEqual(panel["status_light"], "RED")
        self.assertIn("hotspot", panel["headline"].lower())

    def test_acceptable_headline(self):
        """Test borderline health score generates acceptable headline."""
        drift_summary = {
            "dag_ok": True,
            "sustained_regressions": [],
            "drift_hotspots": [],
            "health_score": 0.75,  # Between 0.7 and 0.8
            "summary_metrics": {},
        }
        release_eval = self._make_healthy_release_eval()

        panel = build_dag_director_panel(drift_summary, release_eval)

        self.assertEqual(panel["status_light"], "GREEN")
        self.assertIn("acceptable", panel["headline"].lower())

    def test_panel_json_serializable(self):
        """Test director panel is JSON serializable."""
        drift_summary = self._make_healthy_drift_summary()
        release_eval = self._make_healthy_release_eval()

        panel = build_dag_director_panel(drift_summary, release_eval)

        # Should not raise
        json_str = json.dumps(panel)
        self.assertIsInstance(json_str, str)

    def test_panel_has_required_fields(self):
        """Test director panel contains all required fields."""
        drift_summary = self._make_healthy_drift_summary()
        release_eval = self._make_healthy_release_eval()

        panel = build_dag_director_panel(drift_summary, release_eval)

        self.assertIn("status_light", panel)
        self.assertIn("health_score", panel)
        self.assertIn("drift_hotspots", panel)
        self.assertIn("headline", panel)


class TestPhaseIVIntegration(unittest.TestCase):
    """Integration tests for Phase IV functionality."""

    def _make_posture(
        self,
        max_depth: int = 5,
        vertex_count: int = 10,
        edge_count: int = 8,
        drift_eligible: bool = True,
        has_cycles: bool = False,
    ) -> Dict[str, Any]:
        """Helper to create posture snapshots."""
        return {
            "schema_version": "1.0.0",
            "has_cycles": has_cycles,
            "max_depth": max_depth,
            "vertex_count": vertex_count,
            "edge_count": edge_count,
            "drift_eligible": drift_eligible,
        }

    def test_full_pipeline_healthy(self):
        """Test full Phase IV pipeline with healthy DAG."""
        # Build ledger
        postures = [
            self._make_posture(max_depth=5, vertex_count=10, edge_count=8),
            self._make_posture(max_depth=6, vertex_count=15, edge_count=12),
        ]
        ledger = build_dag_drift_ledger(postures)

        # Evaluate eligibility
        oracle_result = evaluate_dag_eligibility(postures[0], postures[1])

        # Get global health
        global_health = summarize_dag_posture_for_global_health(ledger)

        # Release evaluation
        release_eval = evaluate_dag_for_release(ledger.to_dict(), oracle_result.to_dict())

        # MAAS summary
        maas_summary = summarize_dag_for_maas(oracle_result.to_dict(), global_health.to_dict())

        # Director panel
        panel = build_dag_director_panel(global_health.to_dict(), release_eval)

        # Assertions
        self.assertTrue(release_eval["release_ok"])
        self.assertEqual(release_eval["status"], "OK")
        self.assertTrue(maas_summary["dag_structurally_ok"])
        self.assertEqual(maas_summary["status"], "OK")
        self.assertEqual(panel["status_light"], "GREEN")

    def test_full_pipeline_with_cycle(self):
        """Test full Phase IV pipeline with cycle introduction."""
        # Build ledger with cycle
        postures = [
            self._make_posture(has_cycles=False),
            self._make_posture(has_cycles=True),  # Cycle introduced
        ]
        ledger = build_dag_drift_ledger(postures)

        # Evaluate eligibility
        oracle_result = evaluate_dag_eligibility(postures[0], postures[1])

        # Get global health
        global_health = summarize_dag_posture_for_global_health(ledger)

        # Release evaluation
        release_eval = evaluate_dag_for_release(ledger.to_dict(), oracle_result.to_dict())

        # MAAS summary
        maas_summary = summarize_dag_for_maas(oracle_result.to_dict(), global_health.to_dict())

        # Director panel
        panel = build_dag_director_panel(global_health.to_dict(), release_eval)

        # Assertions - cycle should block everything
        self.assertFalse(release_eval["release_ok"])
        self.assertEqual(release_eval["status"], "BLOCK")
        self.assertFalse(maas_summary["dag_structurally_ok"])
        self.assertEqual(maas_summary["status"], "BLOCK")
        self.assertEqual(panel["status_light"], "RED")
        self.assertIn("cycle", panel["headline"].lower())


# =============================================================================
# PHASE V: DAG × Topology × HT Consistency Tests
# =============================================================================

class TestCrossLayerStatus(unittest.TestCase):
    """Tests for CrossLayerStatus enum."""

    def test_enum_values(self):
        """Test that all expected cross-layer statuses exist."""
        self.assertEqual(CrossLayerStatus.CONSISTENT.value, "CONSISTENT")
        self.assertEqual(CrossLayerStatus.TENSION.value, "TENSION")
        self.assertEqual(CrossLayerStatus.CONFLICT.value, "CONFLICT")


class TestCheckDagTopologyConsistency(unittest.TestCase):
    """Tests for check_dag_topology_consistency function."""

    def test_both_ok_consistent(self):
        """Test DAG OK + Topology OK → CONSISTENT."""
        dag_health = {"status": "OK", "health_score": 0.9}
        topology_signal = {"status": "OK", "score": 0.85}

        result = check_dag_topology_consistency(dag_health, topology_signal)

        self.assertTrue(result["consistent"])
        self.assertEqual(result["status"], "CONSISTENT")
        self.assertTrue(any("both healthy" in r.lower() for r in result["reasons"]))

    def test_dag_block_topo_ok_conflict(self):
        """Test DAG BLOCK + Topology OK → CONFLICT."""
        dag_health = {"status": "BLOCK", "health_score": 0.3}
        topology_signal = {"status": "OK", "score": 0.9}

        result = check_dag_topology_consistency(dag_health, topology_signal)

        self.assertFalse(result["consistent"])
        self.assertEqual(result["status"], "CONFLICT")
        self.assertTrue(any("BLOCK" in r and "OK" in r for r in result["reasons"]))

    def test_dag_ok_topo_block_conflict(self):
        """Test DAG OK + Topology BLOCK → CONFLICT."""
        dag_health = {"status": "OK", "health_score": 0.9}
        topology_signal = {"status": "BLOCK", "score": 0.2}

        result = check_dag_topology_consistency(dag_health, topology_signal)

        self.assertFalse(result["consistent"])
        self.assertEqual(result["status"], "CONFLICT")

    def test_both_block_consistent(self):
        """Test DAG BLOCK + Topology BLOCK → CONSISTENT (both agree)."""
        dag_health = {"status": "BLOCK"}
        topology_signal = {"status": "BLOCK"}

        result = check_dag_topology_consistency(dag_health, topology_signal)

        self.assertTrue(result["consistent"])
        self.assertEqual(result["status"], "CONSISTENT")
        self.assertTrue(any("both report critical" in r.lower() for r in result["reasons"]))

    def test_dag_warn_topo_ok_tension(self):
        """Test DAG WARN + Topology OK → TENSION."""
        dag_health = {"status": "WARN"}
        topology_signal = {"status": "OK"}

        result = check_dag_topology_consistency(dag_health, topology_signal)

        self.assertTrue(result["consistent"])
        self.assertEqual(result["status"], "TENSION")

    def test_both_warn_tension(self):
        """Test DAG WARN + Topology WARN → TENSION."""
        dag_health = {"status": "WARN"}
        topology_signal = {"status": "WARN"}

        result = check_dag_topology_consistency(dag_health, topology_signal)

        self.assertTrue(result["consistent"])
        self.assertEqual(result["status"], "TENSION")
        self.assertTrue(any("both" in r.lower() and "warn" in r.lower() for r in result["reasons"]))

    def test_score_divergence_creates_tension(self):
        """Test large score divergence creates TENSION."""
        dag_health = {"status": "OK", "health_score": 0.9}
        topology_signal = {"status": "OK", "score": 0.5}  # >0.3 divergence

        result = check_dag_topology_consistency(dag_health, topology_signal)

        self.assertEqual(result["status"], "TENSION")
        self.assertTrue(any("divergence" in r.lower() for r in result["reasons"]))

    def test_status_variant_normalization(self):
        """Test that status variants are normalized."""
        dag_health = {"status": "PASS"}  # Should normalize to OK
        topology_signal = {"status": "HEALTHY"}  # Should normalize to OK

        result = check_dag_topology_consistency(dag_health, topology_signal)

        self.assertTrue(result["consistent"])
        self.assertEqual(result["status"], "CONSISTENT")

    def test_result_json_serializable(self):
        """Test result is JSON serializable."""
        dag_health = {"status": "OK"}
        topology_signal = {"status": "OK"}

        result = check_dag_topology_consistency(dag_health, topology_signal)

        json_str = json.dumps(result)
        self.assertIsInstance(json_str, str)


class TestBuildDagHtAlignmentView(unittest.TestCase):
    """Tests for build_dag_ht_alignment_view function."""

    def test_both_ok_aligned(self):
        """Test DAG OK + HT OK → aligned."""
        dag_health = {"status": "OK", "release_ok": True}
        ht_health = {"status": "OK", "invariants_ok": True}

        result = build_dag_ht_alignment_view(dag_health, ht_health)

        self.assertTrue(result["aligned"])
        self.assertEqual(result["status"], "CONSISTENT")
        self.assertEqual(result["dag_status"], "OK")
        self.assertEqual(result["ht_status"], "OK")
        self.assertIsNone(result["misalignment_type"])

    def test_dag_block_ht_ok_conflict(self):
        """Test DAG BLOCK + HT OK → CONFLICT with misalignment type."""
        dag_health = {"status": "BLOCK", "blocking_reasons": ["Cycle detected"]}
        ht_health = {"status": "OK"}

        result = build_dag_ht_alignment_view(dag_health, ht_health)

        self.assertFalse(result["aligned"])
        self.assertEqual(result["status"], "CONFLICT")
        self.assertEqual(result["misalignment_type"], "DAG_BLOCK_HT_OK")
        self.assertTrue(any("blocking reasons" in r.lower() for r in result["reasons"]))

    def test_dag_ok_ht_block_conflict(self):
        """Test DAG OK + HT BLOCK → CONFLICT with misalignment type."""
        dag_health = {"status": "OK"}
        ht_health = {"status": "BLOCK", "violations": ["invariant_1", "invariant_2"]}

        result = build_dag_ht_alignment_view(dag_health, ht_health)

        self.assertFalse(result["aligned"])
        self.assertEqual(result["status"], "CONFLICT")
        self.assertEqual(result["misalignment_type"], "DAG_OK_HT_BLOCK")
        self.assertTrue(any("violations" in r.lower() for r in result["reasons"]))

    def test_both_warn_tension(self):
        """Test DAG WARN + HT WARN → TENSION with misalignment type."""
        dag_health = {"status": "WARN"}
        ht_health = {"status": "WARN"}

        result = build_dag_ht_alignment_view(dag_health, ht_health)

        self.assertTrue(result["aligned"])
        self.assertEqual(result["status"], "TENSION")
        self.assertEqual(result["misalignment_type"], "BOTH_WARN")

    def test_both_block_aligned(self):
        """Test DAG BLOCK + HT BLOCK → aligned (both agree)."""
        dag_health = {"status": "BLOCK"}
        ht_health = {"status": "BLOCK"}

        result = build_dag_ht_alignment_view(dag_health, ht_health)

        self.assertTrue(result["aligned"])
        self.assertEqual(result["status"], "CONSISTENT")

    def test_dag_warn_ht_ok_tension(self):
        """Test DAG WARN + HT OK → TENSION."""
        dag_health = {"status": "WARN"}
        ht_health = {"status": "OK"}

        result = build_dag_ht_alignment_view(dag_health, ht_health)

        self.assertTrue(result["aligned"])
        self.assertEqual(result["status"], "TENSION")

    def test_result_has_required_fields(self):
        """Test result contains all required fields."""
        dag_health = {"status": "OK"}
        ht_health = {"status": "OK"}

        result = build_dag_ht_alignment_view(dag_health, ht_health)

        self.assertIn("aligned", result)
        self.assertIn("status", result)
        self.assertIn("dag_status", result)
        self.assertIn("ht_status", result)
        self.assertIn("misalignment_type", result)
        self.assertIn("reasons", result)

    def test_result_json_serializable(self):
        """Test result is JSON serializable."""
        dag_health = {"status": "BLOCK", "blocking_reasons": ["test"]}
        ht_health = {"status": "OK"}

        result = build_dag_ht_alignment_view(dag_health, ht_health)

        json_str = json.dumps(result)
        self.assertIsInstance(json_str, str)


class TestBuildDagDirectorPanelExtended(unittest.TestCase):
    """Tests for build_dag_director_panel_extended function."""

    def _make_healthy_drift_summary(self) -> Dict[str, Any]:
        """Create a healthy drift summary."""
        return {
            "dag_ok": True,
            "sustained_regressions": [],
            "drift_hotspots": [],
            "health_score": 0.95,
            "summary_metrics": {},
        }

    def _make_healthy_release_eval(self) -> Dict[str, Any]:
        """Create a healthy release evaluation."""
        return {
            "release_ok": True,
            "status": "OK",
            "blocking_reasons": [],
        }

    def test_no_cross_layer_returns_base_panel(self):
        """Test without cross-layer signals returns base panel."""
        drift_summary = self._make_healthy_drift_summary()
        release_eval = self._make_healthy_release_eval()

        panel = build_dag_director_panel_extended(drift_summary, release_eval)

        # Should have base fields
        self.assertIn("status_light", panel)
        self.assertIn("health_score", panel)
        # Should NOT have cross-layer fields
        self.assertNotIn("cross_layer_status", panel)
        self.assertNotIn("cross_layer_reasons", panel)

    def test_all_ok_consistent(self):
        """Test all layers OK → CONSISTENT cross-layer status."""
        drift_summary = self._make_healthy_drift_summary()
        release_eval = self._make_healthy_release_eval()
        topology = {"status": "OK"}
        ht = {"status": "OK"}

        panel = build_dag_director_panel_extended(
            drift_summary, release_eval,
            topology_signal=topology,
            ht_replay_health=ht,
        )

        self.assertEqual(panel["status_light"], "GREEN")
        self.assertEqual(panel["cross_layer_status"], "CONSISTENT")
        self.assertIn("cross_layer_reasons", panel)

    def test_topology_conflict_upgrades_light(self):
        """Test topology conflict upgrades status light from GREEN to YELLOW."""
        drift_summary = self._make_healthy_drift_summary()
        release_eval = self._make_healthy_release_eval()
        topology = {"status": "BLOCK"}  # Conflict with OK release

        panel = build_dag_director_panel_extended(
            drift_summary, release_eval,
            topology_signal=topology,
        )

        self.assertEqual(panel["status_light"], "YELLOW")  # Upgraded from GREEN
        self.assertEqual(panel["cross_layer_status"], "CONFLICT")

    def test_ht_conflict_detected(self):
        """Test HT conflict is detected."""
        drift_summary = self._make_healthy_drift_summary()
        release_eval = self._make_healthy_release_eval()
        ht = {"status": "BLOCK"}  # Conflict with OK release

        panel = build_dag_director_panel_extended(
            drift_summary, release_eval,
            ht_replay_health=ht,
        )

        self.assertEqual(panel["cross_layer_status"], "CONFLICT")
        self.assertTrue(any("[HT]" in r for r in panel["cross_layer_reasons"]))

    def test_tension_does_not_upgrade_green(self):
        """Test tension doesn't upgrade GREEN status."""
        drift_summary = self._make_healthy_drift_summary()
        release_eval = self._make_healthy_release_eval()
        topology = {"status": "WARN"}

        panel = build_dag_director_panel_extended(
            drift_summary, release_eval,
            topology_signal=topology,
        )

        self.assertEqual(panel["status_light"], "GREEN")  # Not upgraded
        self.assertEqual(panel["cross_layer_status"], "TENSION")

    def test_multiple_conflicts_aggregated(self):
        """Test multiple conflicts are aggregated."""
        drift_summary = self._make_healthy_drift_summary()
        release_eval = self._make_healthy_release_eval()
        topology = {"status": "BLOCK"}
        ht = {"status": "BLOCK"}

        panel = build_dag_director_panel_extended(
            drift_summary, release_eval,
            topology_signal=topology,
            ht_replay_health=ht,
        )

        self.assertEqual(panel["cross_layer_status"], "CONFLICT")
        # Should have reasons from both layers
        topo_reasons = [r for r in panel["cross_layer_reasons"] if "[Topology]" in r]
        ht_reasons = [r for r in panel["cross_layer_reasons"] if "[HT]" in r]
        self.assertGreater(len(topo_reasons), 0)
        self.assertGreater(len(ht_reasons), 0)

    def test_panel_json_serializable(self):
        """Test extended panel is JSON serializable."""
        drift_summary = self._make_healthy_drift_summary()
        release_eval = self._make_healthy_release_eval()

        panel = build_dag_director_panel_extended(
            drift_summary, release_eval,
            topology_signal={"status": "OK"},
            ht_replay_health={"status": "WARN"},
        )

        json_str = json.dumps(panel)
        self.assertIsInstance(json_str, str)


class TestCheckMultilayerConsistency(unittest.TestCase):
    """Tests for check_multilayer_consistency function."""

    def test_dag_only(self):
        """Test with DAG only returns basic result."""
        dag_health = {"status": "OK"}

        result = check_multilayer_consistency(dag_health)

        self.assertTrue(result["overall_consistent"])
        self.assertEqual(result["overall_status"], "CONSISTENT")
        self.assertEqual(result["layer_statuses"]["dag"], "OK")
        self.assertEqual(result["conflict_count"], 0)
        self.assertEqual(result["tension_count"], 0)

    def test_all_ok_consistent(self):
        """Test all layers OK → CONSISTENT."""
        dag_health = {"status": "OK"}
        topology = {"status": "OK"}
        ht = {"status": "OK"}

        result = check_multilayer_consistency(dag_health, topology, ht)

        self.assertTrue(result["overall_consistent"])
        self.assertEqual(result["overall_status"], "CONSISTENT")
        self.assertEqual(result["conflict_count"], 0)

    def test_one_conflict_detected(self):
        """Test single conflict is detected."""
        dag_health = {"status": "BLOCK"}
        topology = {"status": "OK"}  # Conflict

        result = check_multilayer_consistency(dag_health, topology)

        self.assertFalse(result["overall_consistent"])
        self.assertEqual(result["overall_status"], "CONFLICT")
        self.assertEqual(result["conflict_count"], 1)

    def test_multiple_tensions(self):
        """Test multiple tensions are counted."""
        dag_health = {"status": "OK"}
        topology = {"status": "WARN"}
        ht = {"status": "WARN"}

        result = check_multilayer_consistency(dag_health, topology, ht)

        self.assertTrue(result["overall_consistent"])
        self.assertEqual(result["overall_status"], "TENSION")
        self.assertEqual(result["tension_count"], 2)

    def test_conflict_overrides_tension(self):
        """Test conflict status overrides tension."""
        dag_health = {"status": "BLOCK"}
        topology = {"status": "WARN"}  # Tension
        ht = {"status": "OK"}  # Conflict

        result = check_multilayer_consistency(dag_health, topology, ht)

        self.assertFalse(result["overall_consistent"])
        self.assertEqual(result["overall_status"], "CONFLICT")

    def test_result_has_all_reasons(self):
        """Test all reasons are aggregated."""
        dag_health = {"status": "WARN"}
        topology = {"status": "WARN"}
        ht = {"status": "WARN"}

        result = check_multilayer_consistency(dag_health, topology, ht)

        self.assertGreater(len(result["all_reasons"]), 0)

    def test_result_json_serializable(self):
        """Test result is JSON serializable."""
        dag_health = {"status": "OK"}
        topology = {"status": "BLOCK"}

        result = check_multilayer_consistency(dag_health, topology)

        json_str = json.dumps(result)
        self.assertIsInstance(json_str, str)


class TestPhaseVIntegration(unittest.TestCase):
    """Integration tests for Phase V multi-layer consistency."""

    def test_all_layers_healthy_consistent(self):
        """Test DAG OK + Topology OK + HT OK → CONSISTENT."""
        dag_health = {"status": "OK", "release_ok": True}
        topology = {"status": "OK", "healthy": True}
        ht = {"status": "OK", "invariants_ok": True}

        # Check topology consistency
        topo_result = check_dag_topology_consistency(dag_health, topology)
        self.assertTrue(topo_result["consistent"])
        self.assertEqual(topo_result["status"], "CONSISTENT")

        # Check HT alignment
        ht_result = build_dag_ht_alignment_view(dag_health, ht)
        self.assertTrue(ht_result["aligned"])
        self.assertEqual(ht_result["status"], "CONSISTENT")

        # Check multilayer
        multi_result = check_multilayer_consistency(dag_health, topology, ht)
        self.assertTrue(multi_result["overall_consistent"])
        self.assertEqual(multi_result["overall_status"], "CONSISTENT")

    def test_dag_block_ht_ok_conflict(self):
        """Test DAG BLOCK + HT OK → CONFLICT."""
        dag_health = {"status": "BLOCK", "blocking_reasons": ["Cycle detected"]}
        ht = {"status": "OK", "invariants_ok": True}

        ht_result = build_dag_ht_alignment_view(dag_health, ht)

        self.assertFalse(ht_result["aligned"])
        self.assertEqual(ht_result["status"], "CONFLICT")
        self.assertEqual(ht_result["misalignment_type"], "DAG_BLOCK_HT_OK")

    def test_dag_warn_topology_warn_tension(self):
        """Test DAG WARN + Topology WARN → TENSION."""
        dag_health = {"status": "WARN", "health_score": 0.6}
        topology = {"status": "WARN", "score": 0.65}

        result = check_dag_topology_consistency(dag_health, topology)

        self.assertTrue(result["consistent"])
        self.assertEqual(result["status"], "TENSION")

    def test_full_pipeline_with_conflict(self):
        """Test full Phase V pipeline with conflict detection."""
        # Create healthy base
        drift_summary = {
            "dag_ok": True,
            "sustained_regressions": [],
            "drift_hotspots": [],
            "health_score": 0.9,
            "summary_metrics": {},
        }
        release_eval = {
            "release_ok": True,
            "status": "OK",
            "blocking_reasons": [],
        }

        # But HT reports block
        ht = {"status": "BLOCK", "violations": ["test_violation"]}

        # Extended panel should detect conflict
        panel = build_dag_director_panel_extended(
            drift_summary, release_eval,
            ht_replay_health=ht,
        )

        self.assertEqual(panel["cross_layer_status"], "CONFLICT")
        self.assertEqual(panel["status_light"], "YELLOW")  # Upgraded due to conflict

    def test_full_pipeline_all_healthy(self):
        """Test full Phase V pipeline when all layers are healthy."""
        drift_summary = {
            "dag_ok": True,
            "sustained_regressions": [],
            "drift_hotspots": [],
            "health_score": 0.95,
            "summary_metrics": {},
        }
        release_eval = {
            "release_ok": True,
            "status": "OK",
            "blocking_reasons": [],
        }
        topology = {"status": "OK", "score": 0.9}
        ht = {"status": "OK", "invariants_ok": True}

        panel = build_dag_director_panel_extended(
            drift_summary, release_eval,
            topology_signal=topology,
            ht_replay_health=ht,
        )

        self.assertEqual(panel["status_light"], "GREEN")
        self.assertEqual(panel["cross_layer_status"], "CONSISTENT")


# =============================================================================
# PHASE VI: Cross-Layer GovernanceSignal Tests
# =============================================================================

class TestGovernanceStatus(unittest.TestCase):
    """Tests for GovernanceStatus enum."""

    def test_enum_values(self):
        """Test that all expected governance statuses exist."""
        from backend.dag.preflight_check import GovernanceStatus
        self.assertEqual(GovernanceStatus.OK.value, "OK")
        self.assertEqual(GovernanceStatus.WARN.value, "WARN")
        self.assertEqual(GovernanceStatus.BLOCK.value, "BLOCK")


class TestToGovernanceSignalForStructure(unittest.TestCase):
    """Tests for to_governance_signal_for_structure function."""

    def test_consistent_returns_ok(self):
        """Test CONSISTENT multilayer result → OK governance."""
        from backend.dag.preflight_check import to_governance_signal_for_structure
        consistency = {
            "overall_status": "CONSISTENT",
            "overall_consistent": True,
            "layer_statuses": {"dag": "OK", "topology": "OK"},
            "conflict_count": 0,
            "tension_count": 0,
            "all_reasons": [],
        }

        result = to_governance_signal_for_structure(consistency)

        self.assertEqual(result["status"], "OK")
        self.assertTrue(result["structural_cohesion"])
        self.assertEqual(result["blocking_rules"], [])
        self.assertEqual(result["warning_rules"], [])

    def test_tension_returns_warn(self):
        """Test TENSION multilayer result → WARN governance."""
        from backend.dag.preflight_check import to_governance_signal_for_structure
        consistency = {
            "overall_status": "TENSION",
            "overall_consistent": True,
            "layer_statuses": {"dag": "OK", "topology": "TENSION"},
            "conflict_count": 0,
            "tension_count": 1,
            "all_reasons": ["Score divergence detected"],
        }

        result = to_governance_signal_for_structure(consistency)

        self.assertEqual(result["status"], "WARN")
        self.assertTrue(result["structural_cohesion"])
        self.assertGreater(len(result["warning_rules"]), 0)

    def test_conflict_returns_block(self):
        """Test CONFLICT multilayer result → BLOCK governance."""
        from backend.dag.preflight_check import to_governance_signal_for_structure
        consistency = {
            "overall_status": "CONFLICT",
            "overall_consistent": False,
            "layer_statuses": {"dag": "BLOCK", "topology": "CONFLICT"},
            "conflict_count": 1,
            "tension_count": 0,
            "all_reasons": ["DAG BLOCK vs Topology OK"],
        }

        result = to_governance_signal_for_structure(consistency)

        self.assertEqual(result["status"], "BLOCK")
        self.assertFalse(result["structural_cohesion"])
        self.assertGreater(len(result["blocking_rules"]), 0)

    def test_ht_alignment_conflict_adds_blocking_rule(self):
        """Test HT alignment CONFLICT adds to blocking rules."""
        from backend.dag.preflight_check import to_governance_signal_for_structure
        consistency = {
            "overall_status": "CONSISTENT",
            "overall_consistent": True,
            "layer_statuses": {"dag": "OK"},
            "conflict_count": 0,
            "tension_count": 0,
            "all_reasons": [],
        }
        ht_alignment = {
            "status": "CONFLICT",
            "aligned": False,
            "misalignment_type": "DAG_BLOCK_HT_OK",
            "reasons": ["DAG blocked but HT passes"],
        }

        result = to_governance_signal_for_structure(consistency, ht_alignment)

        self.assertEqual(result["status"], "BLOCK")
        self.assertIn("DAG_BLOCK_HT_OK", result["blocking_rules"])

    def test_ht_alignment_tension_adds_warning_rule(self):
        """Test HT alignment TENSION adds to warning rules."""
        from backend.dag.preflight_check import to_governance_signal_for_structure
        consistency = {
            "overall_status": "CONSISTENT",
            "overall_consistent": True,
            "layer_statuses": {"dag": "OK"},
            "conflict_count": 0,
            "tension_count": 0,
            "all_reasons": [],
        }
        ht_alignment = {
            "status": "TENSION",
            "aligned": True,
            "misalignment_type": "BOTH_WARN",
            "reasons": ["Both layers show warnings"],
        }

        result = to_governance_signal_for_structure(consistency, ht_alignment)

        self.assertEqual(result["status"], "WARN")
        self.assertIn("BOTH_WARN", result["warning_rules"])

    def test_simple_two_layer_result_handled(self):
        """Test simple two-layer result (from check_dag_topology_consistency)."""
        from backend.dag.preflight_check import to_governance_signal_for_structure
        # Simple result format (consistent, status, reasons)
        simple_result = {
            "consistent": True,
            "status": "CONSISTENT",
            "reasons": ["Both layers healthy"],
        }

        result = to_governance_signal_for_structure(simple_result)

        self.assertEqual(result["status"], "OK")
        self.assertTrue(result["structural_cohesion"])

    def test_layer_summary_populated(self):
        """Test layer summary is populated correctly."""
        from backend.dag.preflight_check import to_governance_signal_for_structure
        consistency = {
            "overall_status": "CONSISTENT",
            "overall_consistent": True,
            "layer_statuses": {"dag": "OK", "topology": "OK", "ht": "OK"},
            "conflict_count": 0,
            "tension_count": 0,
            "all_reasons": [],
        }

        result = to_governance_signal_for_structure(consistency)

        self.assertIn("dag", result["layer_summary"])
        self.assertIn("topology", result["layer_summary"])
        self.assertIn("ht", result["layer_summary"])

    def test_result_json_serializable(self):
        """Test result is JSON serializable."""
        from backend.dag.preflight_check import to_governance_signal_for_structure
        consistency = {
            "overall_status": "CONFLICT",
            "overall_consistent": False,
            "layer_statuses": {"dag": "BLOCK"},
            "conflict_count": 1,
            "tension_count": 0,
            "all_reasons": ["Test reason"],
        }

        result = to_governance_signal_for_structure(consistency)

        json_str = json.dumps(result)
        self.assertIsInstance(json_str, str)


class TestBuildStructureConsolePane(unittest.TestCase):
    """Tests for build_structure_console_pane function."""

    def test_ok_governance_green_light(self):
        """Test OK governance → GREEN status light."""
        from backend.dag.preflight_check import build_structure_console_pane
        governance = {
            "status": "OK",
            "structural_cohesion": True,
            "blocking_rules": [],
            "warning_rules": [],
            "layer_summary": {"dag": "OK"},
        }

        result = build_structure_console_pane(governance)

        self.assertEqual(result["status_light"], "GREEN")
        self.assertEqual(result["cross_layer_status"], "CONSISTENT")
        self.assertTrue(result["cohesion_ok"])

    def test_warn_governance_yellow_light(self):
        """Test WARN governance → YELLOW status light."""
        from backend.dag.preflight_check import build_structure_console_pane
        governance = {
            "status": "WARN",
            "structural_cohesion": True,
            "blocking_rules": [],
            "warning_rules": ["DAG_TOPOLOGY_TENSION"],
            "layer_summary": {"dag": "OK", "topology": "WARN"},
        }

        result = build_structure_console_pane(governance)

        self.assertEqual(result["status_light"], "YELLOW")
        self.assertEqual(result["cross_layer_status"], "TENSION")

    def test_block_governance_red_light(self):
        """Test BLOCK governance → RED status light."""
        from backend.dag.preflight_check import build_structure_console_pane
        governance = {
            "status": "BLOCK",
            "structural_cohesion": False,
            "blocking_rules": ["DAG_BLOCK_HT_OK"],
            "warning_rules": [],
            "layer_summary": {"dag": "BLOCK", "ht": "OK"},
        }

        result = build_structure_console_pane(governance)

        self.assertEqual(result["status_light"], "RED")
        self.assertEqual(result["cross_layer_status"], "CONFLICT")
        self.assertFalse(result["cohesion_ok"])
        self.assertIn("blocking_rules", result)

    def test_director_panel_upgrades_light(self):
        """Test director panel can upgrade status light."""
        from backend.dag.preflight_check import build_structure_console_pane
        governance = {
            "status": "OK",
            "structural_cohesion": True,
            "blocking_rules": [],
            "warning_rules": [],
            "layer_summary": {"dag": "OK"},
        }
        director_panel = {
            "status_light": "RED",  # More severe
            "health_score": 0.3,
        }

        result = build_structure_console_pane(governance, director_panel)

        self.assertEqual(result["status_light"], "RED")  # Upgraded

    def test_headline_for_dag_block_ht_ok(self):
        """Test headline for DAG_BLOCK_HT_OK conflict."""
        from backend.dag.preflight_check import build_structure_console_pane
        governance = {
            "status": "BLOCK",
            "structural_cohesion": False,
            "blocking_rules": ["DAG_BLOCK_HT_OK"],
            "warning_rules": [],
            "layer_summary": {"dag": "BLOCK", "ht": "OK"},
        }

        result = build_structure_console_pane(governance)

        self.assertIn("DAG", result["headline"])
        self.assertIn("HT", result["headline"])

    def test_headline_for_tension(self):
        """Test headline for tension state."""
        from backend.dag.preflight_check import build_structure_console_pane
        governance = {
            "status": "WARN",
            "structural_cohesion": True,
            "blocking_rules": [],
            "warning_rules": ["BOTH_WARN"],
            "layer_summary": {"dag": "WARN", "ht": "WARN"},
        }

        result = build_structure_console_pane(governance)

        self.assertIn("warn", result["headline"].lower())

    def test_headline_for_healthy_multilayer(self):
        """Test headline for healthy multi-layer state."""
        from backend.dag.preflight_check import build_structure_console_pane
        governance = {
            "status": "OK",
            "structural_cohesion": True,
            "blocking_rules": [],
            "warning_rules": [],
            "layer_summary": {"dag": "OK", "topology": "OK", "ht": "OK"},
        }

        result = build_structure_console_pane(governance)

        self.assertIn("healthy", result["headline"].lower())
        self.assertIn("3", result["headline"])  # 3 layers

    def test_result_json_serializable(self):
        """Test result is JSON serializable."""
        from backend.dag.preflight_check import build_structure_console_pane
        governance = {
            "status": "BLOCK",
            "structural_cohesion": False,
            "blocking_rules": ["TEST_RULE"],
            "warning_rules": [],
            "layer_summary": {"dag": "BLOCK"},
        }

        result = build_structure_console_pane(governance)

        json_str = json.dumps(result)
        self.assertIsInstance(json_str, str)


class TestBuildGlobalHealthStructureEntry(unittest.TestCase):
    """Tests for build_global_health_structure_entry function."""

    def test_dag_only_minimal(self):
        """Test with DAG health only."""
        from backend.dag.preflight_check import build_global_health_structure_entry
        dag_health = {"status": "OK"}

        result = build_global_health_structure_entry(dag_health)

        self.assertEqual(result["status_light"], "GREEN")
        self.assertEqual(result["governance_status"], "OK")
        self.assertTrue(result["cohesion_ok"])

    def test_all_layers_ok(self):
        """Test with all layers healthy."""
        from backend.dag.preflight_check import build_global_health_structure_entry
        dag_health = {"status": "OK", "release_ok": True}
        topology = {"status": "OK", "score": 0.9}
        ht = {"status": "OK", "invariants_ok": True}

        result = build_global_health_structure_entry(
            dag_health, topology, ht
        )

        self.assertEqual(result["status_light"], "GREEN")
        self.assertEqual(result["cross_layer_status"], "CONSISTENT")
        self.assertEqual(result["governance_status"], "OK")

    def test_conflict_detected(self):
        """Test conflict is detected across layers."""
        from backend.dag.preflight_check import build_global_health_structure_entry
        dag_health = {"status": "BLOCK", "blocking_reasons": ["Cycle"]}
        ht = {"status": "OK"}

        result = build_global_health_structure_entry(dag_health, ht_replay_health=ht)

        self.assertEqual(result["status_light"], "RED")
        self.assertEqual(result["cross_layer_status"], "CONFLICT")
        self.assertEqual(result["governance_status"], "BLOCK")
        self.assertIn("blocking_rules", result)

    def test_with_drift_summary_and_release_eval(self):
        """Test with full director panel integration."""
        from backend.dag.preflight_check import build_global_health_structure_entry
        dag_health = {"status": "OK"}
        topology = {"status": "OK"}
        drift_summary = {
            "dag_ok": True,
            "sustained_regressions": [],
            "drift_hotspots": [],
            "health_score": 0.9,
            "summary_metrics": {},
        }
        release_eval = {
            "release_ok": True,
            "status": "OK",
            "blocking_reasons": [],
        }

        result = build_global_health_structure_entry(
            dag_health,
            topology_signal=topology,
            drift_summary=drift_summary,
            release_eval=release_eval,
        )

        self.assertEqual(result["status_light"], "GREEN")
        self.assertIn("layer_summary", result)

    def test_reasons_limited_to_three(self):
        """Test reasons are limited to top 3."""
        from backend.dag.preflight_check import build_global_health_structure_entry
        dag_health = {"status": "WARN"}
        topology = {"status": "WARN"}
        ht = {"status": "WARN"}

        result = build_global_health_structure_entry(dag_health, topology, ht)

        self.assertLessEqual(len(result["reasons"]), 3)

    def test_result_json_serializable(self):
        """Test result is JSON serializable."""
        from backend.dag.preflight_check import build_global_health_structure_entry
        dag_health = {"status": "OK"}
        topology = {"status": "BLOCK"}

        result = build_global_health_structure_entry(dag_health, topology)

        json_str = json.dumps(result)
        self.assertIsInstance(json_str, str)


class TestPhaseVIIntegration(unittest.TestCase):
    """Integration tests for Phase VI cross-layer governance."""

    def test_full_pipeline_ok(self):
        """Test full pipeline with all layers OK."""
        from backend.dag.preflight_check import (
            build_global_health_structure_entry,
            check_multilayer_consistency,
            to_governance_signal_for_structure,
        )

        dag_health = {"status": "OK", "release_ok": True}
        topology = {"status": "OK", "score": 0.9}
        ht = {"status": "OK", "invariants_ok": True}

        # Step 1: Check consistency
        consistency = check_multilayer_consistency(dag_health, topology, ht)
        self.assertEqual(consistency["overall_status"], "CONSISTENT")

        # Step 2: Get governance signal
        governance = to_governance_signal_for_structure(consistency)
        self.assertEqual(governance["status"], "OK")
        self.assertTrue(governance["structural_cohesion"])

        # Step 3: Full integration
        entry = build_global_health_structure_entry(dag_health, topology, ht)
        self.assertEqual(entry["status_light"], "GREEN")
        self.assertEqual(entry["governance_status"], "OK")

    def test_full_pipeline_conflict(self):
        """Test full pipeline with DAG BLOCK + HT OK conflict."""
        from backend.dag.preflight_check import (
            build_dag_ht_alignment_view,
            build_global_health_structure_entry,
            check_multilayer_consistency,
            to_governance_signal_for_structure,
        )

        dag_health = {"status": "BLOCK", "blocking_reasons": ["Cycle detected"]}
        ht = {"status": "OK", "invariants_ok": True}

        # Step 1: Check HT alignment
        ht_alignment = build_dag_ht_alignment_view(dag_health, ht)
        self.assertEqual(ht_alignment["status"], "CONFLICT")
        self.assertEqual(ht_alignment["misalignment_type"], "DAG_BLOCK_HT_OK")

        # Step 2: Check consistency
        consistency = check_multilayer_consistency(dag_health, ht_replay_health=ht)
        self.assertEqual(consistency["overall_status"], "CONFLICT")

        # Step 3: Get governance signal
        governance = to_governance_signal_for_structure(consistency, ht_alignment)
        self.assertEqual(governance["status"], "BLOCK")
        self.assertFalse(governance["structural_cohesion"])
        self.assertIn("DAG_BLOCK_HT_OK", governance["blocking_rules"])

        # Step 4: Full integration
        entry = build_global_health_structure_entry(dag_health, ht_replay_health=ht)
        self.assertEqual(entry["status_light"], "RED")
        self.assertEqual(entry["governance_status"], "BLOCK")

    def test_full_pipeline_tension(self):
        """Test full pipeline with tension across layers."""
        from backend.dag.preflight_check import (
            build_global_health_structure_entry,
            check_multilayer_consistency,
            to_governance_signal_for_structure,
        )

        dag_health = {"status": "OK", "health_score": 0.9}
        topology = {"status": "WARN", "score": 0.6}

        # Step 1: Check consistency
        consistency = check_multilayer_consistency(dag_health, topology)
        self.assertEqual(consistency["overall_status"], "TENSION")

        # Step 2: Get governance signal
        governance = to_governance_signal_for_structure(consistency)
        self.assertEqual(governance["status"], "WARN")
        self.assertTrue(governance["structural_cohesion"])

        # Step 3: Full integration
        entry = build_global_health_structure_entry(dag_health, topology)
        self.assertEqual(entry["status_light"], "YELLOW")
        self.assertEqual(entry["governance_status"], "WARN")

    def test_global_health_structure_format(self):
        """Test the output format matches global_health['structure'] requirements."""
        from backend.dag.preflight_check import build_global_health_structure_entry

        dag_health = {"status": "OK"}
        entry = build_global_health_structure_entry(dag_health)

        # Verify required fields for global_health["structure"]
        self.assertIn("status_light", entry)
        self.assertIn("headline", entry)
        self.assertIn("cross_layer_status", entry)
        self.assertIn("cohesion_ok", entry)
        self.assertIn("governance_status", entry)
        self.assertIn("layer_summary", entry)

        # Verify types
        self.assertIn(entry["status_light"], ["GREEN", "YELLOW", "RED"])
        self.assertIsInstance(entry["headline"], str)
        self.assertIn(entry["cross_layer_status"], ["CONSISTENT", "TENSION", "CONFLICT"])
        self.assertIsInstance(entry["cohesion_ok"], bool)


if __name__ == "__main__":
    unittest.main()
