import json

from analysis import u2_dynamics as dynamics
from analysis.conjecture_engine import generate_mock_data, _load_thresholds


def test_debug_snapshot_pattern_labels():
    thresholds = dynamics.DynamicsThresholds.from_mapping(_load_thresholds())
    baseline, logistic_rfl = generate_mock_data("positive_logistic")
    _, oscillatory_rfl = generate_mock_data("instability")

    logistic_snapshot = dynamics.build_dynamics_debug_snapshot(
        {
            "run_id": "logistic",
            "slice_name": "synthetic",
            "mode": "rfl",
            "records": logistic_rfl,
            "comparison": {"baseline_records": baseline, "rfl_records": logistic_rfl},
        },
        thresholds,
    )
    oscillatory_snapshot = dynamics.build_dynamics_debug_snapshot(
        {
            "run_id": "oscillation",
            "slice_name": "synthetic",
            "mode": "rfl",
            "records": oscillatory_rfl,
            "comparison": {"baseline_records": baseline, "rfl_records": oscillatory_rfl},
        },
        thresholds,
    )

    assert logistic_snapshot.pattern_label == "logistic"
    assert oscillatory_snapshot.pattern_label == "oscillatory"
    assert oscillatory_snapshot.oscillation_index > logistic_snapshot.oscillation_index


def test_negative_control_checker_ok():
    thresholds = dynamics.DynamicsThresholds.from_mapping(_load_thresholds())
    baseline, null_rfl = generate_mock_data("null")
    snapshot = dynamics.build_dynamics_debug_snapshot(
        {
            "run_id": "nc-baseline",
            "slice_name": "synthetic",
            "mode": "baseline",
            "records": baseline,
            "comparison": {"baseline_records": baseline, "rfl_records": null_rfl},
        },
        thresholds,
    )

    report = dynamics.check_negative_control_dynamics([snapshot], max_allowed_oscillation_index=0.3)
    assert report["status"] == "OK"
    assert report["notes"] == []


def test_debug_snapshot_determinism():
    thresholds = dynamics.DynamicsThresholds.from_mapping(_load_thresholds())
    baseline, rfl = generate_mock_data("positive_logistic")
    snapshot = dynamics.build_dynamics_debug_snapshot(
        {
            "run_id": "deterministic",
            "slice_name": "synthetic",
            "mode": "rfl",
            "records": rfl,
            "comparison": {"baseline_records": baseline, "rfl_records": rfl},
        },
        thresholds,
    )

    left = json.dumps(snapshot.__dict__, sort_keys=True)
    right = json.dumps(snapshot.__dict__, sort_keys=True)
    assert left == right


def test_console_summary_with_nc_ok():
    thresholds = dynamics.DynamicsThresholds.from_mapping(_load_thresholds())
    baseline, null_rfl = generate_mock_data("null")
    _, oscillatory = generate_mock_data("instability")

    nc_snapshot = dynamics.build_dynamics_debug_snapshot(
        {
            "run_id": "nc-run",
            "slice_name": "synthetic",
            "mode": "baseline",
            "records": baseline,
            "comparison": {"baseline_records": baseline, "rfl_records": null_rfl},
        },
        thresholds,
    )
    uplift_snapshot = dynamics.build_dynamics_debug_snapshot(
        {
            "run_id": "uplift-run",
            "slice_name": "synthetic",
            "mode": "rfl",
            "records": oscillatory,
            "comparison": {"baseline_records": baseline, "rfl_records": oscillatory},
        },
        thresholds,
    )

    summary = dynamics.summarize_dynamics_for_global_console([nc_snapshot, uplift_snapshot])
    assert summary["run_count"] == 2
    assert summary["nc_status"] == "OK"
    expected_headline = (
        f"{summary['run_count']} dynamics runs; mean oscillation "
        f"{summary['mean_oscillation_index']:.3f}, "
        f"max {summary['max_oscillation_index']:.3f} ({summary['nc_status']})."
    )
    assert summary["headline"] == expected_headline
