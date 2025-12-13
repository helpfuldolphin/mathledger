import json

import pytest

from backend.health.global_surface import build_global_health_surface
from backend.health.policy_drift_tile import (
    attach_policy_drift_to_evidence,
    attach_policy_drift_to_p3_stability_report,
    build_first_light_policy_drift_summary,
    build_policy_drift_summary,
    extract_policy_drift_signal_for_first_light,
    summarize_policy_drift_vs_nci_consistency,
)
from scripts.policy_drift_lint import summarize_policy_drift_for_global_health


@pytest.fixture
def tile_with_counts() -> dict:
    return {
        "schema_version": "policy-drift/1.0",
        "status": "WARN",
        "num_rules": 5,
        "blocked_rules": 2,
        "headline": "Manual override headline.",
    }


@pytest.fixture
def tile_with_lists() -> dict:
    return {
        "schema_version": "policy-drift/1.0",
        "status": "BLOCK",
        "soft_changes": [{"path": "trainer.lr"}, {"path": "trainer.beta"}],
        "hard_blocks": [{"path": "promotion.threshold"}],
    }


@pytest.fixture
def tile_mixed() -> dict:
    return {
        "schema_version": "policy-drift/1.0",
        "status": "OK",
        "num_rules": 4,
        "soft_changes": [{"path": "trainer.lr"}],
        "hard_blocks": [{"path": "promotion.threshold"}],
    }


@pytest.fixture
def make_nci_signal():
    def _make(status: str, global_nci: float = 0.97) -> dict:
        return {"health_contribution": {"status": status, "global_nci": global_nci}}

    return _make


def test_build_policy_drift_summary_prefers_explicit_counts(tile_with_counts: dict) -> None:
    summary = build_policy_drift_summary(tile_with_counts)

    assert summary["num_rules"] == 5
    assert summary["blocked_rules"] == 2
    assert summary["headline"] == "Manual override headline."
    assert summary["status_light"] == "YELLOW"


def test_build_policy_drift_summary_counts_from_lists(tile_with_lists: dict) -> None:
    summary = build_policy_drift_summary(tile_with_lists)

    assert summary["num_rules"] == 3
    assert summary["blocked_rules"] == 1
    assert summary["status_light"] == "RED"


def test_build_policy_drift_summary_mixed_prefers_explicit(tile_mixed: dict) -> None:
    summary = build_policy_drift_summary(tile_mixed)

    assert summary["num_rules"] == 4  # explicit count wins
    assert summary["blocked_rules"] == 1  # inferred from hard blocks
    assert summary["status_light"] == "GREEN"


def test_policy_drift_tile_attachment_preserves_payload_and_signals() -> None:
    base_payload = {"status": "ok"}
    report = {
        "status": "BLOCK",
        "breaking_changes": [{"category": "promotion_thresholds"}],
        "soft_changes": [],
    }
    tile = summarize_policy_drift_for_global_health(report)
    tile["hard_blocks"] = report["breaking_changes"]

    payload = build_global_health_surface(
        base_payload=base_payload,
        dynamics_snapshots=[],
        policy_drift_tile=tile,
    )

    assert "policy_drift" in payload
    assert payload["policy_drift"] == tile
    assert payload["signals"]["policy_drift"] == build_policy_drift_summary(tile)
    assert "policy_drift" not in base_payload


def test_extract_policy_drift_signal_for_first_light_counts_rules() -> None:
    tile = {
        "status": "WARN",
        "soft_changes": [{"path": "trainer.lr"}],
        "hard_blocks": [{"path": "promotion.threshold"}],
    }

    signal = extract_policy_drift_signal_for_first_light(tile)

    assert signal == {"status": "WARN", "num_rules": 2, "blocked_rules": 1}


def test_attach_policy_drift_to_p3_stability_report_non_mutating() -> None:
    stability_report = {"status": "ok"}
    report = {
        "status": "BLOCK",
        "breaking_changes": [{"category": "promotion_thresholds"}],
        "soft_changes": [{"category": "learning_rates"}],
    }
    tile = summarize_policy_drift_for_global_health(report)
    tile["hard_blocks"] = report["breaking_changes"]
    tile["soft_changes"] = report["soft_changes"]

    updated = attach_policy_drift_to_p3_stability_report(stability_report, tile)

    summary = build_policy_drift_summary(tile)
    assert updated["policy_drift_summary"] == summary
    assert "policy_drift_summary" not in stability_report


def test_attach_policy_drift_to_evidence_shapes_block_and_serializes() -> None:
    evidence = {"governance": {"existing": {"status": "OK"}}}
    report = {
        "status": "WARN",
        "breaking_changes": [],
        "soft_changes": [
            {
                "category": "learning_rates",
                "path": "trainer.lr",
                "change": "changed",
                "old": 0.1,
                "new": 0.2,
            }
        ],
    }
    tile = summarize_policy_drift_for_global_health(report)
    tile["soft_changes"] = report["soft_changes"]
    summary = build_first_light_policy_drift_summary(tile)

    updated = attach_policy_drift_to_evidence(evidence, tile, summary)

    assert evidence["governance"].get("policy_drift") is None
    block = updated["governance"]["policy_drift"]
    assert block["first_light_summary"] == summary
    assert block["status"] == summary["status"]
    assert block["blocked_rules"] == summary["blocked_rules"]
    assert isinstance(block["explanation"], str) and block["explanation"]
    assert json.dumps(updated)


def test_build_first_light_policy_drift_summary_structure() -> None:
    tile = {
        "schema_version": "policy-drift/1.0",
        "status": "BLOCK",
        "soft_changes": [{"path": "trainer.lr"}],
        "hard_blocks": [{"path": "promotion.threshold"}],
    }
    summary = build_first_light_policy_drift_summary(tile)

    assert summary["schema_version"] == "policy-drift/1.0"
    assert summary["status"] == "BLOCK"
    assert summary["status_light"] == "RED"
    assert summary["num_rules"] == 2
    assert summary["blocked_rules"] == 1
    assert isinstance(summary["headline"], str) and summary["headline"]
    assert json.dumps(summary)


@pytest.mark.parametrize(
    ("status", "expected_light"),
    [("OK", "GREEN"), ("WARN", "YELLOW"), ("BLOCK", "RED")],
)
def test_attach_policy_drift_to_p3_stability_report_status_light(status, expected_light):
    tile = {
        "status": status,
        "soft_changes": [{"id": 1}],
        "hard_blocks": [{"id": 2}] if status == "BLOCK" else [],
    }
    report = attach_policy_drift_to_p3_stability_report({}, tile)

    assert report["policy_drift_summary"]["status_light"] == expected_light


def test_summarize_policy_drift_vs_nci_consistency_consistent(make_nci_signal) -> None:
    tile = {
        "status": "WARN",
        "soft_changes": [{"path": "trainer.lr"}],
        "hard_blocks": [],
    }
    summary = build_policy_drift_summary(tile)
    result = summarize_policy_drift_vs_nci_consistency(
        summary, make_nci_signal("WARN")
    )

    assert result["consistency"] == "CONSISTENT"
    assert "Policy drift status WARN" in result["notes"][0]
    assert "NCI signal status WARN" in result["notes"][1]


def test_summarize_policy_drift_vs_nci_consistency_partial(make_nci_signal) -> None:
    tile = {"status": "WARN", "soft_changes": [], "hard_blocks": [{"path": "cutoff"}]}
    summary = build_policy_drift_summary(tile)
    result = summarize_policy_drift_vs_nci_consistency(
        summary, make_nci_signal("OK")
    )

    assert result["consistency"] == "PARTIAL"


def test_summarize_policy_drift_vs_nci_consistency_inconsistent(make_nci_signal) -> None:
    tile = {"status": "BLOCK", "soft_changes": [], "hard_blocks": [{"path": "cutoff"}]}
    summary = build_policy_drift_summary(tile)
    result = summarize_policy_drift_vs_nci_consistency(
        summary, make_nci_signal("OK")
    )

    assert result["consistency"] == "INCONSISTENT"
    assert "NCI signal status OK" in result["notes"][1]

