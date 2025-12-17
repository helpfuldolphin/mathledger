from backend.health.curriculum_drift_tile import (
    build_curriculum_drift_tile_for_global_health,
)


def test_curriculum_drift_tile_global_health_summary_is_deterministic() -> None:
    events = [
        {"slice_name": "alpha", "drift_status": "OK", "emitted_at": "2025-01-01T00:00:00Z"},
        {"slice_name": "beta", "drift_status": "WARN", "emitted_at": "2025-01-01T00:00:01Z"},
        {"slice_name": "gamma", "drift_status": "WARN", "emitted_at": "2025-01-01T00:00:02Z"},
        {"slice_name": "beta", "drift_status": "BLOCK", "emitted_at": "2025-01-01T00:00:03Z"},
        {"slice_name": "delta", "drift_status": "WARN", "emitted_at": "2025-01-01T00:00:04Z"},
    ]

    tile = build_curriculum_drift_tile_for_global_health(events)
    duplicate_tile = build_curriculum_drift_tile_for_global_health(list(events))

    assert tile == duplicate_tile
    assert tile["status"] == "BLOCK"
    assert tile["status_light"] == "RED"
    assert tile["stressed_slices"] == ["beta", "delta", "gamma"]
    assert tile["violation_count"] == 4
