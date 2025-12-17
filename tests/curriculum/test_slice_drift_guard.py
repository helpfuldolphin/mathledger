import json
from copy import deepcopy

from curriculum.slice_drift_guard import (
    build_curriculum_provenance_event,
    build_slice_drift_snapshot,
    compute_slice_drift_and_provenance,
    summarize_slice_drift_for_global_health,
)


def _baseline_slice() -> dict:
    return {
        "name": "alpha",
        "params": {
            "atoms": 4,
            "depth_max": 5,
            "breadth_max": 320,
            "total_max": 2400,
        },
        "gates": {
            "coverage": {
                "ci_lower_min": 0.91,
                "sample_min": 16,
                "require_attestation": True,
            },
            "abstention": {
                "max_rate_pct": 18.0,
                "max_mass": 640,
            },
            "velocity": {
                "min_pph": 150.0,
                "stability_cv_max": 0.12,
                "window_minutes": 45,
            },
            "caps": {
                "min_attempt_mass": 2000,
                "min_runtime_minutes": 20.0,
                "backlog_max": 0.40,
            },
        },
    }


def test_build_slice_drift_snapshot_no_change() -> None:
    baseline = _baseline_slice()
    current = deepcopy(baseline)

    snapshot = build_slice_drift_snapshot(baseline, current)

    assert snapshot["severity"] == "NONE"
    assert snapshot["status"] == "OK"
    assert snapshot["changed_params"] == []

    event = build_curriculum_provenance_event("abc123", baseline["name"], snapshot)
    payload = json.loads(event)
    assert payload["curriculum_fingerprint"] == "abc123"
    assert payload["slice_name"] == "alpha"
    assert payload["drift_status"] == "OK"
    assert payload["changed_params"] == []


def test_build_slice_drift_snapshot_warn_on_parametric_shift() -> None:
    baseline = _baseline_slice()
    current = deepcopy(baseline)
    current["params"]["atoms"] = baseline["params"]["atoms"] + 1

    snapshot = build_slice_drift_snapshot(baseline, current)

    assert snapshot["severity"] == "PARAMETRIC"
    assert snapshot["status"] == "WARN"
    assert len(snapshot["changed_params"]) == 1
    entry = snapshot["changed_params"][0]
    assert entry["path"] == "params.atoms"
    assert entry["classification"] == "PARAMETRIC"
    assert entry["delta"] == 1.0


def test_build_slice_drift_snapshot_block_on_semantic_regression() -> None:
    baseline = _baseline_slice()
    current = deepcopy(baseline)
    current["params"]["depth_max"] = baseline["params"]["depth_max"] - 2

    snapshot = build_slice_drift_snapshot(baseline, current)

    assert snapshot["severity"] == "SEMANTIC"
    assert snapshot["status"] == "BLOCK"
    assert snapshot["changed_params"][0]["path"] == "params.depth_max"
    assert snapshot["changed_params"][0]["classification"] == "SEMANTIC"


def test_compute_slice_drift_and_provenance_helper_matches_primitives() -> None:
    baseline = _baseline_slice()
    current = deepcopy(baseline)
    fingerprint = "fingerprint-123"

    snapshot, event = compute_slice_drift_and_provenance(
        baseline_slice=baseline,
        current_slice=current,
        curriculum_fingerprint=fingerprint,
    )

    assert snapshot == build_slice_drift_snapshot(baseline, current)
    parsed = json.loads(event)
    expected = json.loads(
        build_curriculum_provenance_event(
            curriculum_fingerprint=fingerprint,
            slice_name=baseline["name"],
            drift_snapshot=snapshot,
        )
    )
    assert parsed == expected
    assert parsed["drift_status"] == "OK"


def test_summarize_slice_drift_for_global_health_rollup() -> None:
    events = [
        {
            "slice_name": "alpha",
            "drift_status": "OK",
            "emitted_at": "2025-01-01T00:00:00+00:00",
        },
        {
            "slice_name": "beta",
            "drift_status": "WARN",
            "emitted_at": "2025-01-01T00:00:01+00:00",
        },
        {
            "slice_name": "beta",
            "drift_status": "BLOCK",
            "emitted_at": "2025-01-01T00:00:02+00:00",
        },
        {
            "slice_name": "gamma",
            "drift_status": "WARN",
            "emitted_at": "2025-01-01T00:00:03+00:00",
        },
    ]

    summary = summarize_slice_drift_for_global_health(events)

    assert summary["overall_status"] == "BLOCK"
    assert summary["warn_events"] == 2
    assert summary["block_events"] == 1
    assert summary["event_count"] == 4

    stressed = summary["stressed_slices"]
    assert len(stressed) == 2
    assert stressed[0]["slice_name"] == "beta"
    assert stressed[0]["blocks"] == 1
    assert stressed[1]["slice_name"] == "gamma"
    assert stressed[1]["warns"] == 1
