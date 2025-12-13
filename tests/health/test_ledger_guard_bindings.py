import json

from backend.health.ledger_guard_bindings import (
    attach_ledger_guard_to_evidence,
    attach_ledger_guard_to_p3_stability_report,
    attach_ledger_guard_to_p4_calibration,
    attach_ledger_guard_to_p4_calibration_report,
    summarize_ledger_guard_for_evidence,
)

_LEDGER_TILE = {
    "status_light": "YELLOW",
    "violation_count": 2,
    "headline": "Ledger guard v2: 2 schema issue(s) detected",
}


def test_p3_binding_is_pure_and_shapes_summary():
    source = {"phase": "P3"}

    result = attach_ledger_guard_to_p3_stability_report(source, _LEDGER_TILE)

    assert source == {"phase": "P3"}
    assert result is not source
    summary = result["ledger_guard_summary"]
    assert summary == {
        "status_light": _LEDGER_TILE["status_light"],
        "violation_counts": _LEDGER_TILE["violation_count"],
        "headline": _LEDGER_TILE["headline"],
    }


def test_p4_binding_is_deterministic():
    report = {"phase": "P4", "existing": {"value": 1}}

    first = attach_ledger_guard_to_p4_calibration_report(report, _LEDGER_TILE)
    second = attach_ledger_guard_to_p4_calibration_report(report, _LEDGER_TILE)

    assert first == second
    assert report == {"phase": "P4", "existing": {"value": 1}}


def test_evidence_helper_preserves_governance_payload():
    evidence = {"governance": {"existing": {"status": "ok"}}}

    updated = attach_ledger_guard_to_evidence(evidence, _LEDGER_TILE)

    assert updated["governance"]["existing"] == {"status": "ok"}
    summary = updated["governance"]["ledger_guard"]["first_light_summary"]
    assert summary["status_light"] == _LEDGER_TILE["status_light"]
    assert summary["violation_counts"] == _LEDGER_TILE["violation_count"]
    assert summary["headline"] == _LEDGER_TILE["headline"]


def test_p4_calibration_attachment_is_pure_and_json_safe():
    calibration = {"mode": "SHADOW"}

    enriched = attach_ledger_guard_to_p4_calibration(calibration, _LEDGER_TILE)

    assert calibration == {"mode": "SHADOW"}
    assert enriched is not calibration
    assert json.loads(json.dumps(enriched["ledger_guard"]))


def test_summarize_ledger_guard_for_evidence_is_deterministic():
    summary_first = summarize_ledger_guard_for_evidence(_LEDGER_TILE)
    summary_second = summarize_ledger_guard_for_evidence(_LEDGER_TILE)

    assert summary_first == summary_second
    assert summary_first is not summary_second


def test_end_to_end_summary_flow():
    base_summary = {
        "status_light": "GREEN",
        "violation_count": 0,
        "headline": "Ledger guard v2: monotone chain confirmed",
    }

    calibration = attach_ledger_guard_to_p4_calibration({}, base_summary)
    evidence = attach_ledger_guard_to_evidence({}, calibration["ledger_guard"])

    first_light_summary = evidence["governance"]["ledger_guard"]["first_light_summary"]

    # Simulate status signal creation
    status_signals = {
        "ledger_guard": {
            "status_light": first_light_summary["status_light"],
            "violation_counts": first_light_summary["violation_counts"],
        }
    }

    assert status_signals["ledger_guard"]["status_light"] == "GREEN"
    assert status_signals["ledger_guard"]["violation_counts"] == 0
