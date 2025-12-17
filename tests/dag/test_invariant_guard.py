import json
import tempfile
from pathlib import Path

from backend.dag.invariant_guard import (
    ProofDag,
    SliceProfile,
    evaluate_dag_invariants,
    load_invariant_rules,
    summarize_dag_invariants_for_global_health,
)


def _make_proof_dag(depth: int = 3, branching: float = 1.0) -> ProofDag:
    profile = SliceProfile(
        slice_id="alpha",
        max_depth=depth,
        max_branching_factor=branching,
        node_kind_counts={"LEMMA": 2},
    )
    ledger_entry = {
        "cycle": 0,
        "MaxDepth(t)": depth,
        "GlobalBranchingFactor(t)": branching,
    }
    return ProofDag(slices={"alpha": profile}, metric_ledger=[ledger_entry])


def test_evaluate_dag_invariants_ok():
    dag = _make_proof_dag()
    rules = {
        "max_depth_per_slice": {"alpha": 5},
        "allowed_node_kinds": {"alpha": {"LEMMA"}},
    }
    report = evaluate_dag_invariants(dag, rules)
    assert report["status"] == "OK"
    assert report["violated_invariants"] == []


def test_load_invariant_rules_json_and_yaml_roundtrip():
    config = {
        "globals": {"max_branching_factor": 2.0},
        "slices": {
            "alpha": {
                "max_depth": 4,
                "max_branching_factor": 1.5,
                "allowed_node_kinds": ["LEMMA"],
            }
        },
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "rules.json"
        json_path.write_text(json.dumps(config), encoding="utf-8")
        loaded = load_invariant_rules(json_path)
        assert loaded["max_depth_per_slice"]["alpha"] == 4
        assert loaded["max_branching_factor"] == 2.0
        assert loaded["allowed_node_kinds"]["alpha"] == {"LEMMA"}

        yaml_path = Path(tmpdir) / "rules.yaml"
        yaml_path.write_text(
            "max_depth_per_slice:\n  '*': 3\nallowed_node_kinds:\n  beta:\n    - GOAL\n",
            encoding="utf-8",
        )
        loaded_yaml = load_invariant_rules(yaml_path)
        assert loaded_yaml["max_depth_per_slice"]["*"] == 3
        assert loaded_yaml["allowed_node_kinds"]["beta"] == {"GOAL"}


def test_integration_block_summary_from_config(tmp_path):
    rules = {
        "slices": {
            "alpha": {
                "max_depth": 1,
                "allowed_node_kinds": ["LEMMA"],
            }
        }
    }
    config_path = tmp_path / "rules.json"
    config_path.write_text(json.dumps(rules), encoding="utf-8")
    loaded_rules = load_invariant_rules(config_path)

    dag = _make_proof_dag(depth=4)
    report = evaluate_dag_invariants(dag, loaded_rules)
    summary = summarize_dag_invariants_for_global_health(report)
    assert report["status"] == "BLOCK"
    assert summary["status"] == "BLOCK"
    assert "breached" in summary["headline"]
    assert summary["violation_count"] == 1
