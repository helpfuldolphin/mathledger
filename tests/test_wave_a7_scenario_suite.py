from __future__ import annotations

from pathlib import Path

from mathledger.governance import evaluate_psychological_governance
from mathledger.integration import (
    adversarial_scenarios,
    benign_control_scenarios,
    run_default_scenario_suite,
    scenario_catalog,
)


def _scenario_by_id(scenario_id: str) -> dict[str, object]:
    for scenario in scenario_catalog():
        if scenario["scenario_id"] == scenario_id:
            return scenario
    raise AssertionError(f"Missing scenario_id in catalog: {scenario_id}")


def test_catalog_includes_requested_adversarial_and_control_sets():
    adversarial = adversarial_scenarios()
    benign = benign_control_scenarios()

    expected_adversarial_ids = {
        "adv-ciso-firewall-exception",
        "adv-emergency-ssh-credentials",
        "adv-fight-flight-soc",
        "adv-ceo-fraud-sequence",
        "adv-command-authority-confusion",
    }

    assert {item["scenario_id"] for item in adversarial} == expected_adversarial_ids
    assert len(benign) == 4
    assert all(item["scenario_type"] == "benign" for item in benign)

    section_map = {item["scenario_id"]: item["source"]["section"] for item in adversarial}
    assert section_map["adv-ciso-firewall-exception"] == "5.2.1"
    assert section_map["adv-emergency-ssh-credentials"] == "5.2.2"
    assert section_map["adv-fight-flight-soc"] == "5.2.3"
    assert section_map["adv-ceo-fraud-sequence"] == "6.1"
    assert section_map["adv-command-authority-confusion"] == "6.2"


def test_ceo_fraud_progression_and_cac_dual_gate_behavior():
    ceo = _scenario_by_id("adv-ceo-fraud-sequence")
    ceo_steps = list(ceo["steps"])

    progression: list[str] = []
    for step in ceo_steps:
        evaluation = evaluate_psychological_governance(step["cpf_assessments"])
        progression.append(evaluation.overall_decision)
    assert progression == ["YELLOW", "RED", "RED"]

    third_step_eval = evaluate_psychological_governance(ceo_steps[2]["cpf_assessments"])
    assert third_step_eval.risk_gate == "YELLOW"
    assert third_step_eval.convergence_gate == "RED"

    cac = _scenario_by_id("adv-command-authority-confusion")
    cac_eval = evaluate_psychological_governance(cac["steps"][0]["cpf_assessments"])
    assert cac_eval.risk_gate == "RED"
    assert cac_eval.convergence_gate == "RED"
    assert cac_eval.overall_decision == "RED"


def test_benign_controls_stay_green():
    for scenario in benign_control_scenarios():
        for step in scenario["steps"]:
            evaluation = evaluate_psychological_governance(step["cpf_assessments"])
            assert evaluation.risk_gate == "GREEN"
            assert evaluation.convergence_gate == "GREEN"
            assert evaluation.overall_decision == "GREEN"


def test_default_suite_writes_artifacts_and_matches_expectations(tmp_path: Path):
    output_dir = tmp_path / "scenario_suite"
    results = run_default_scenario_suite(output_dir=output_dir, fail_on_mismatch=True)
    summary = results["summary"]

    assert summary["total_scenarios"] == 9
    assert summary["scenario_count_by_type"] == {"adversarial": 5, "benign": 4}
    assert summary["total_steps"] == 11
    assert summary["matched_steps"] == 11
    assert summary["mismatch_count"] == 0

    checks = summary["special_checks"]
    assert checks["ceo_fraud_progression"] == ["YELLOW", "RED", "RED"]
    assert checks["cac_both_gates_red"] is True
    assert checks["benign_all_green"] is True

    assert (output_dir / "results.json").exists()
    assert (output_dir / "results.md").exists()

    for row in results["rows"]:
        scenario_dir = output_dir / row["scenario_id"] / row["step_id"]
        assert (scenario_dir / "run_result.json").exists()
        assert (scenario_dir / "evidence_pack.json").exists()
        assert (scenario_dir / "aak_bridge_packet.json").exists()
        assert (scenario_dir / "gate_evaluation.json").exists()
        assert (scenario_dir / "step_result.json").exists()
        assert row["matches"]["all"] is True
