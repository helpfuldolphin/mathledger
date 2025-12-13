import json

from backend.health.u2_dynamics_tile import (
    attach_u2_dynamics_tile,
    attach_u2_dynamics_to_evidence,
    attach_u2_dynamics_to_first_light_status,
    attach_u2_dynamics_to_p4_summary,
    build_u2_dynamics_tile,
    build_u2_dynamics_decomposition_summary,
    build_u2_dynamics_first_light_status_signal,
    build_u2_dynamics_window_metrics,
    DEFAULT_U2_DYNAMICS_WINDOW_SIZE,
    U2_DYNAMICS_EXTRACTION_SOURCES,
    summarize_u2_dynamics_vs_telemetry,
    u2_dynamics_for_alignment_view,
)


def test_status_light_green_when_success_high_and_depth_modest():
    tile = build_u2_dynamics_tile(
        {"mean_depth": 42.0, "max_depth": 90.0, "success_rate": 0.95, "runs": 25}
    )
    assert tile["status_light"] == "GREEN"


def test_status_light_yellow_for_midrange_metrics():
    tile = build_u2_dynamics_tile(
        {"mean_depth": 80.0, "max_depth": 150.0, "success_rate": 0.78, "runs": 10}
    )
    assert tile["status_light"] == "YELLOW"


def test_status_light_red_for_low_success_or_no_runs():
    tile_low_success = build_u2_dynamics_tile(
        {"mean_depth": 20.0, "max_depth": 50.0, "success_rate": 0.4, "runs": 8}
    )
    tile_no_runs = build_u2_dynamics_tile(
        {"mean_depth": 20.0, "max_depth": 50.0, "success_rate": 0.9, "runs": 0}
    )

    assert tile_low_success["status_light"] == "RED"
    assert tile_no_runs["status_light"] == "RED"


def test_headline_is_present_and_neutral():
    tile = build_u2_dynamics_tile(
        {"mean_depth": 30.0, "max_depth": 120.0, "success_rate": 0.85, "runs": 12}
    )
    headline = tile["headline"]
    assert isinstance(headline, str)
    assert headline.endswith(".")
    assert "good" not in headline.lower()
    assert "bad" not in headline.lower()


def test_tile_is_json_serializable_and_deterministic():
    summary = {"mean_depth": 15.0, "max_depth": 70.0, "success_rate": 0.92, "runs": 5}
    first = build_u2_dynamics_tile(summary)
    second = build_u2_dynamics_tile(summary)

    assert first == second
    # Should not raise
    assert json.loads(json.dumps(first)) == first


def test_attach_u2_dynamics_tile_returns_new_mapping():
    base_health = {"schema_version": "global-health-surface/1.0.0"}
    tile = build_u2_dynamics_tile(
        {"mean_depth": 10.0, "max_depth": 55.0, "success_rate": 0.93, "runs": 3}
    )

    updated = attach_u2_dynamics_tile(base_health, tile)
    assert "u2_dynamics" in updated
    assert updated["u2_dynamics"] == tile
    assert updated["schema_version"] == base_health["schema_version"]
    assert "u2_dynamics" not in base_health
    assert json.loads(json.dumps(updated)) == updated


def test_attach_u2_dynamics_to_first_light_status_is_non_mutating_and_json_safe():
    base_status = {"status": "OK", "signals": {"other": {"foo": "bar"}}}
    signal = {
        "success_rate": 0.82,
        "max_depth": 77,
        "status_light": "YELLOW",
    }
    updated = attach_u2_dynamics_to_first_light_status(base_status, signal)

    assert "u2_dynamics" in updated["signals"]
    assert updated["signals"]["u2_dynamics"]["success_rate"] == 0.82
    # Original should remain untouched
    assert "u2_dynamics" not in base_status.get("signals", {})
    assert json.loads(json.dumps(updated)) == updated


def test_attach_u2_dynamics_to_evidence_is_shape_stable_and_deterministic():
    base_evidence = {"governance": {"other": "x"}, "meta": {"id": "ev-1"}}
    signal = {
        "success_rate": 0.91,
        "max_depth": 120,
        "status_light": "GREEN",
    }

    first = attach_u2_dynamics_to_evidence(base_evidence, signal)
    second = attach_u2_dynamics_to_evidence(base_evidence, signal)

    assert first == second
    assert "u2_dynamics" in first["governance"]
    dyn = first["governance"]["u2_dynamics"]
    assert set(dyn.keys()) == {"success_rate", "max_depth", "status_light", "headline"}
    assert "u2_dynamics" not in base_evidence.get("governance", {})
    assert json.loads(json.dumps(first)) == first


def test_attach_u2_dynamics_to_p4_summary_is_non_mutating_and_json_safe():
    p4_summary = {"phase": "P4", "status": "observational"}
    tile = build_u2_dynamics_tile(
        {"mean_depth": 20.0, "max_depth": 50.0, "success_rate": 0.88, "runs": 4}
    )
    updated = attach_u2_dynamics_to_p4_summary(p4_summary, tile)

    assert "u2_dynamics" in updated
    summary = updated["u2_dynamics"]
    assert set(summary.keys()) == {"success_rate", "max_depth", "status_light", "headline"}
    assert "u2_dynamics" not in p4_summary
    assert json.loads(json.dumps(updated)) == updated


def test_summarize_u2_dynamics_vs_telemetry_reports_consistency_and_inconsistency():
    dynamics_green = {"status_light": "GREEN"}
    telemetry_green = {"status_light": "GREEN"}
    res_consistent = summarize_u2_dynamics_vs_telemetry(dynamics_green, telemetry_green)
    assert res_consistent["consistency_status"] == "CONSISTENT"
    assert json.loads(json.dumps(res_consistent)) == res_consistent

    dynamics_warn = {"status_light": "YELLOW"}
    telemetry_ok = {"status_light": "GREEN"}
    res_inconsistent = summarize_u2_dynamics_vs_telemetry(
        dynamics_warn, telemetry_ok, window_context={"start_cycle": 1, "end_cycle": 50}
    )
    assert res_inconsistent["consistency_status"] == "INCONSISTENT"
    assert any("Window cycles 1-50" in note for note in res_inconsistent["advisory_notes"])
    assert json.loads(json.dumps(res_inconsistent)) == res_inconsistent


def test_attach_u2_dynamics_to_evidence_with_telemetry_adds_consistency_block():
    evidence = {"governance": {}}
    dynamics_signal = {"success_rate": 0.9, "max_depth": 80, "status_light": "YELLOW"}
    telemetry_tile = {"status_light": "GREEN"}

    updated = attach_u2_dynamics_to_evidence(evidence, dynamics_signal, telemetry_tile)
    dyn_block = updated["governance"]["u2_dynamics"]
    assert "u2_telemetry_consistency" in dyn_block
    assert dyn_block["u2_telemetry_consistency"]["consistency_status"] == "INCONSISTENT"
    assert json.loads(json.dumps(updated)) == updated


def test_build_u2_dynamics_window_metrics_produces_windows_and_overall(tmp_path):
    path = tmp_path / "real_cycles.jsonl"
    # 60 cycles -> two windows: one full 50, one partial 10
    records = []
    for i in range(1, 61):
        records.append(
            {
                "cycle": i,
                "success": i % 2 == 0,
                "depth": i,
            }
        )
    with path.open("w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(rec) + "\n")

    tile = build_u2_dynamics_window_metrics(path)
    assert tile["window_size"] == 50
    assert len(tile["windows"]) == 2
    assert tile["metrics"]["runs"] == 60
    assert json.loads(json.dumps(tile)) == tile

    attached = attach_u2_dynamics_to_p4_summary({}, tile)
    assert "u2_dynamics" in attached
    assert "windows" in attached["u2_dynamics"]
    assert json.loads(json.dumps(attached)) == attached


def test_build_u2_dynamics_window_metrics_emits_window_bounds(tmp_path):
    path = tmp_path / "real_cycles.jsonl"
    records = [{"cycle": i, "success": True, "depth": i} for i in range(1, 76)]
    with path.open("w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(rec) + "\n")

    tile = build_u2_dynamics_window_metrics(path)
    windows = tile["windows"]
    assert windows[0]["start_cycle"] == 1
    assert windows[0]["end_cycle"] == 50
    assert windows[1]["start_cycle"] == 51
    assert windows[1]["end_cycle"] == 75


def test_summarize_u2_dynamics_vs_telemetry_produces_per_window_list():
    dynamics_tile = {
        "status_light": "YELLOW",
        "windows": [
            {
                "window_index": 0,
                "start_cycle": 1,
                "end_cycle": 50,
                "tile": {"status_light": "GREEN"},
            },
            {
                "window_index": 1,
                "start_cycle": 51,
                "end_cycle": 100,
                "tile": {"status_light": "RED"},
            },
        ],
    }
    telemetry_tile = {"status_light": "GREEN"}

    summary = summarize_u2_dynamics_vs_telemetry(dynamics_tile, telemetry_tile)
    assert summary["consistency_status"] == "INCONSISTENT"
    assert len(summary["window_consistency"]) == 2
    assert summary["window_consistency"][0]["consistency_status"] == "CONSISTENT"
    assert summary["window_consistency"][1]["consistency_status"] == "INCONSISTENT"


def test_summarize_u2_dynamics_vs_telemetry_matches_telemetry_windows_by_index():
    dynamics_tile = {
        "windows": [
            {
                "window_index": 0,
                "start_cycle": 1,
                "end_cycle": 50,
                "tile": {"status_light": "GREEN"},
            },
            {
                "window_index": 2,
                "start_cycle": 101,
                "end_cycle": 150,
                "tile": {"status_light": "YELLOW"},
            },
        ],
    }
    telemetry_tile = {"status_light": "GREEN"}
    telemetry_windows = [
        {"window_index": 2, "tile": {"status_light": "YELLOW"}},
        {"window_index": 0, "tile": {"status_light": "GREEN"}},
    ]

    summary = summarize_u2_dynamics_vs_telemetry(
        dynamics_tile, telemetry_tile, telemetry_windows=telemetry_windows
    )
    windows = summary["window_consistency"]
    assert windows[0]["window_index"] == 0
    assert windows[0]["telemetry_status"] == "GREEN"
    assert windows[1]["window_index"] == 2
    assert windows[1]["telemetry_status"] == "YELLOW"


def test_build_u2_dynamics_first_light_status_signal_is_compact_and_json_safe(tmp_path):
    path = tmp_path / "real_cycles.jsonl"
    records = [{"cycle": i, "success": True, "depth": i} for i in range(1, 11)]
    with path.open("w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(rec) + "\n")

    tile = build_u2_dynamics_window_metrics(path)
    signal = build_u2_dynamics_first_light_status_signal(tile)
    assert signal["mode"] == "SHADOW"
    assert signal["action"] == "LOGGED_ONLY"
    assert signal["extraction_source"] == "P4_SUMMARY"
    assert signal["extraction_source"] in U2_DYNAMICS_EXTRACTION_SOURCES
    assert signal["window_size"] == DEFAULT_U2_DYNAMICS_WINDOW_SIZE
    assert "windows" in signal and isinstance(signal["windows"], list)
    assert "decomposition_summary" in signal
    assert signal["decomposition_summary"]["state_components"]["window_count"] == len(signal["windows"])
    assert "warning" not in signal
    assert json.loads(json.dumps(signal)) == signal


def test_build_u2_dynamics_decomposition_summary_separates_state_outcome_and_safety(tmp_path):
    path = tmp_path / "real_cycles.jsonl"
    records = [{"cycle": i, "success": i % 2 == 0, "depth": i} for i in range(1, 61)]
    with path.open("w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(rec) + "\n")

    tile = build_u2_dynamics_window_metrics(path)
    summary = build_u2_dynamics_decomposition_summary(tile["windows"])

    assert set(summary.keys()) == {"state_components", "outcome_components", "safety_components"}

    state = summary["state_components"]
    outcome = summary["outcome_components"]
    safety = summary["safety_components"]

    assert state["window_count"] == 2
    assert state["runs_total"] == 60
    assert state["runs_mean"] == 30.0
    assert state["mean_depth_mean"] == 40.5
    assert state["max_depth_mean"] == 55.0

    assert outcome["success_rate_mean"] == 0.5
    assert outcome["success_rate_weighted_mean"] == 0.5

    assert safety["status_light_counts"]["RED"] == 2
    assert safety["status_risk_mean"] == 2.0
    assert json.loads(json.dumps(summary)) == summary


def test_build_u2_dynamics_first_light_status_signal_warning_hygiene():
    tile = {
        "status_light": "YELLOW",
        "success_rate": 0.8,
        "max_depth": 120,
        "window_size": 50,
        "windows": [
            {
                "window_index": 0,
                "start_cycle": 1,
                "end_cycle": 50,
                "tile": {
                    "status_light": "YELLOW",
                    "metrics": {"success_rate": 0.8, "max_depth": 120, "mean_depth": 50.0, "runs": 50},
                },
            }
        ],
    }

    signal = build_u2_dynamics_first_light_status_signal(tile)
    assert "warning" in signal
    warning = signal["warning"]
    assert "\n" not in warning
    assert warning == "window_count=1 driver=DRIVER_U2_SUCCESS_RATE_LOW"


def test_u2_dynamics_for_alignment_view_fixed_shape_and_reason_codes():
    tile = {
        "status_light": "YELLOW",
        "success_rate": 0.8,
        "max_depth": 120,
        "window_size": 50,
        "windows": [
            {
                "window_index": 0,
                "start_cycle": 1,
                "end_cycle": 50,
                "tile": {
                    "status_light": "YELLOW",
                    "metrics": {
                        "success_rate": 0.8,
                        "max_depth": 120,
                        "mean_depth": 50.0,
                        "runs": 50,
                    },
                },
            }
        ],
    }
    signal = build_u2_dynamics_first_light_status_signal(tile, extraction_source="P4_SUMMARY")

    ggfl = u2_dynamics_for_alignment_view(signal)
    assert set(ggfl.keys()) == {
        "signal_type",
        "status",
        "conflict",
        "weight_hint",
        "drivers",
        "summary",
        "shadow_mode_invariants",
    }
    assert ggfl["signal_type"] == "SIG-U2"
    assert ggfl["status"] == "warn"
    assert ggfl["conflict"] is False
    assert ggfl["weight_hint"] == "LOW"
    assert ggfl["drivers"] == ["DRIVER_U2_SUCCESS_RATE_LOW", "DRIVER_U2_MAX_DEPTH_HIGH"]

    drivers = ggfl["drivers"]
    assert len(drivers) <= 3
    assert all(isinstance(d, str) and d.startswith("DRIVER_U2_") and " " not in d for d in drivers)

    summary = ggfl["summary"]
    assert "\n" not in summary
    assert summary.endswith(".")

    # Deterministic output
    assert ggfl == u2_dynamics_for_alignment_view(signal)


def test_u2_dynamics_for_alignment_view_ok_when_green():
    tile = {"status_light": "GREEN", "success_rate": 0.95, "max_depth": 90, "windows": []}
    signal = build_u2_dynamics_first_light_status_signal(tile, extraction_source="P4_SUMMARY")
    ggfl = u2_dynamics_for_alignment_view(signal)
    assert ggfl["status"] == "ok"
    assert ggfl["drivers"] == []
