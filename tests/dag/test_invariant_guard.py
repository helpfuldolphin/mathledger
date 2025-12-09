# tests/dag/test_invariant_guard.py
"""
Unit tests for the DAG Invariant Guard.
"""
import json
import tempfile
import unittest
from pathlib import Path

# Add project root
import sys
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.dag.invariant_guard import (
    ProofDag,
    SliceProfile,
    evaluate_dag_invariants,
    load_invariant_rules,
    summarize_dag_invariants_for_global_health,
)


class TestInvariantGuard(unittest.TestCase):
    def setUp(self):
        self.rules = {
            "max_depth_per_slice": {"*": 5},
            "max_branching_factor": 3.0,
            "allowed_node_kinds": {"*": {"AXIOM", "LEMMA", "GOAL"}},
        }

    def test_dag_below_thresholds_ok(self):
        dag = ProofDag(
            slices={
                "slice-alpha": SliceProfile(
                    slice_id="slice-alpha",
                    max_depth=3,
                    max_branching_factor=1.5,
                    node_kind_counts={"LEMMA": 4, "GOAL": 1},
                )
            },
            metric_ledger=[{"MaxDepth(t)": 3, "GlobalBranchingFactor(t)": 1.2}],
        )

        result = evaluate_dag_invariants(dag, self.rules)
        self.assertEqual(result["status"], "OK")
        self.assertEqual(result["violated_invariants"], [])

    def test_single_threshold_violation_warn(self):
        dag = ProofDag(
            slices={
                "slice-alpha": SliceProfile(
                    slice_id="slice-alpha",
                    max_depth=7,
                    max_branching_factor=1.5,
                    node_kind_counts={"LEMMA": 2},
                )
            },
            metric_ledger=[{"MaxDepth(t)": 7, "GlobalBranchingFactor(t)": 1.0}],
        )

        result = evaluate_dag_invariants(dag, self.rules)
        self.assertEqual(result["status"], "WARN")
        self.assertEqual(len(result["violated_invariants"]), 1)

    def test_multiple_critical_violations_block(self):
        dag = ProofDag(
            slices={
                "slice-alpha": SliceProfile(
                    slice_id="slice-alpha",
                    max_depth=8,
                    max_branching_factor=4.5,
                    node_kind_counts={"UNKNOWN": 1},
                ),
                "slice-beta": SliceProfile(
                    slice_id="slice-beta",
                    max_depth=2,
                    max_branching_factor=1.0,
                    node_kind_counts={"AXIOM": 1},
                ),
            },
            metric_ledger=[
                {"MaxDepth(t)": 8, "GlobalBranchingFactor(t)": 3.5, "cycle": 0}
            ],
        )

        result = evaluate_dag_invariants(dag, self.rules)
        self.assertEqual(result["status"], "BLOCK")
        self.assertGreaterEqual(len(result["violated_invariants"]), 2)

    def test_load_invariant_rules_from_json(self):
        config = {
            "globals": {"max_branching_factor": 2.5},
            "slices": {
                "slice-alpha": {
                    "max_depth": 4,
                    "max_branching_factor": 1.5,
                    "allowed_node_kinds": ["LEMMA", "AXIOM"],
                }
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rules.json"
            path.write_text(json.dumps(config))
            rules = load_invariant_rules(path)

        self.assertEqual(rules["max_depth_per_slice"]["slice-alpha"], 4)
        self.assertEqual(rules["max_branching_factor"], 2.5)
        self.assertEqual(
            rules["max_branching_factor_per_slice"]["slice-alpha"], 1.5
        )
        self.assertEqual(
            rules["allowed_node_kinds"]["slice-alpha"], {"LEMMA", "AXIOM"}
        )

    def test_load_invariant_rules_from_yaml_defaults(self):
        yaml_content = """
max_depth_per_slice:
  "*": 6
allowed_node_kinds:
  slice-beta:
    - GOAL
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rules.yaml"
            path.write_text(yaml_content)
            rules = load_invariant_rules(path)

        self.assertEqual(rules["max_depth_per_slice"]["*"], 6)
        self.assertEqual(
            rules["allowed_node_kinds"]["slice-beta"], {"GOAL"}
        )

    def test_summarize_dag_invariants_for_global_health(self):
        report = {
            "status": "WARN",
            "violated_invariants": ["slice-alpha.max_depth>5"],
        }
        summary = summarize_dag_invariants_for_global_health(report)
        self.assertEqual(summary["status"], "WARN")
        self.assertEqual(summary["violation_count"], 1)
        self.assertIn("Single invariant", summary["headline"])


if __name__ == "__main__":
    unittest.main()
