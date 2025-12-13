from analysis.u2_dynamics import DynamicsDebugSnapshot, summarize_dynamics_for_global_console
from backend.health.global_surface import build_global_health_surface


def _make_snapshot(run_id: str, mode: str, oscillation: float, plateau: float) -> DynamicsDebugSnapshot:
    return DynamicsDebugSnapshot(
        run_id=run_id,
        slice_name="sigma",
        mode=mode,
        oscillation_index=oscillation,
        plateau_score=plateau,
        uplift_delta=0.0,
        pattern_label="plateau",
    )


def test_global_console_summary_remains_ok_for_nc_runs():
    nc_snapshots = [
        _make_snapshot("baseline-1", "baseline", 0.02, 0.9),
        _make_snapshot("nc-2", "nc", 0.03, 0.88),
    ]
    uplift_snapshot = DynamicsDebugSnapshot(
        run_id="uplift-main",
        slice_name="sigma",
        mode="rfl",
        oscillation_index=0.4,
        plateau_score=0.5,
        uplift_delta=0.12,
        pattern_label="logistic",
    )

    summary = summarize_dynamics_for_global_console(nc_snapshots + [uplift_snapshot])
    assert summary["run_count"] == 3
    assert summary["nc_status"] == "OK"
    assert summary["nc_notes"] == []
    assert summary["headline"].endswith("(OK).")


def test_global_health_surface_attaches_dynamics_tile():
    snapshots = [
        _make_snapshot("baseline-1", "baseline", 0.02, 0.9),
        DynamicsDebugSnapshot(
            run_id="uplift-main",
            slice_name="sigma",
            mode="rfl",
            oscillation_index=0.4,
            plateau_score=0.5,
            uplift_delta=0.12,
            pattern_label="logistic",
        ),
    ]

    surface = build_global_health_surface({"deterministic_inputs": []}, snapshots)
    assert surface["schema_version"] == "global-health-surface/1.0.0"
    tile = surface["dynamics"]
    assert tile["run_count"] == 2
    assert tile["nc_status"] == "OK"
    assert "headline" in tile
