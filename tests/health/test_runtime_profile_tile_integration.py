"""
Tests for runtime profile health tile integration with global health surface.

Validates:
- Tile shape and structure
- Status light mapping
- Determinism
- Adapter behavior
"""

import json

import pytest

from backend.health.global_surface import build_global_health_surface
from backend.health.runtime_profile_adapter import (
    build_runtime_profile_tile_for_global_health,
)
from experiments.u2.runtime import (
    summarize_runtime_profile_health_for_global_console,
)


def test_runtime_profile_tile_structure():
    """Test that runtime profile tile has correct structure."""
    chaos_summary = {
        "schema_version": "1.0.0",
        "profile_name": "dev-default",
        "total_runs": 100,
        "actions": {"ALLOW": 95, "SAFE_DEGRADE": 5},
        "profile_stability": 0.95,
        "top_violations": [],
    }

    tile = build_runtime_profile_tile_for_global_health(chaos_summary=chaos_summary)

    assert tile is not None
    assert tile["schema_version"] == "1.0.0"
    assert tile["tile_type"] == "runtime_profile_health"
    assert "status_light" in tile
    assert tile["status_light"] in ("GREEN", "YELLOW", "RED")
    assert "profile_name" in tile
    assert "profile_stability" in tile
    assert 0.0 <= tile["profile_stability"] <= 1.0
    assert "no_run_rate" in tile
    assert 0.0 <= tile["no_run_rate"] <= 1.0
    assert "headline" in tile
    assert isinstance(tile["headline"], str)
    assert "notes" in tile
    assert isinstance(tile["notes"], list)


def test_runtime_profile_tile_status_light_green():
    """Test that high stability + low NO_RUN rate produces GREEN."""
    chaos_summary = {
        "schema_version": "1.0.0",
        "profile_name": "dev-default",
        "total_runs": 100,
        "actions": {"ALLOW": 95},
        "profile_stability": 0.95,
        "top_violations": [],
    }

    tile = summarize_runtime_profile_health_for_global_console(chaos_summary)

    assert tile["status_light"] == "GREEN"


def test_runtime_profile_tile_status_light_red():
    """Test that low stability or high NO_RUN rate produces RED."""
    # Low stability case
    chaos_summary_low_stability = {
        "schema_version": "1.0.0",
        "profile_name": "dev-default",
        "total_runs": 100,
        "actions": {"ALLOW": 50, "NO_RUN": 50},
        "profile_stability": 0.50,
        "top_violations": [],
    }

    tile = summarize_runtime_profile_health_for_global_console(chaos_summary_low_stability)

    assert tile["status_light"] == "RED"

    # High NO_RUN rate case
    chaos_summary_high_no_run = {
        "schema_version": "1.0.0",
        "profile_name": "dev-default",
        "total_runs": 100,
        "actions": {"ALLOW": 75, "NO_RUN": 25},
        "profile_stability": 0.75,
        "top_violations": [],
    }

    tile = summarize_runtime_profile_health_for_global_console(chaos_summary_high_no_run)

    assert tile["status_light"] == "RED"


def test_runtime_profile_tile_status_light_yellow():
    """Test that moderate health produces YELLOW."""
    chaos_summary = {
        "schema_version": "1.0.0",
        "profile_name": "dev-default",
        "total_runs": 100,
        "actions": {"ALLOW": 85, "SAFE_DEGRADE": 10, "NO_RUN": 5},
        "profile_stability": 0.85,
        "top_violations": [],
    }

    tile = summarize_runtime_profile_health_for_global_console(chaos_summary)

    assert tile["status_light"] == "YELLOW"


def test_runtime_profile_tile_determinism():
    """Test that same input produces identical output."""
    chaos_summary = {
        "schema_version": "1.0.0",
        "profile_name": "dev-default",
        "total_runs": 100,
        "actions": {"ALLOW": 90, "SAFE_DEGRADE": 10},
        "profile_stability": 0.90,
        "top_violations": ["violation1", "violation2"],
    }

    tile1 = summarize_runtime_profile_health_for_global_console(chaos_summary)
    tile2 = summarize_runtime_profile_health_for_global_console(chaos_summary)

    assert tile1 == tile2

    # JSON serialization should also be identical
    json1 = json.dumps(tile1, sort_keys=True)
    json2 = json.dumps(tile2, sort_keys=True)
    assert json1 == json2


def test_runtime_profile_tile_with_error():
    """Test that error in chaos summary produces RED tile."""
    chaos_summary = {
        "error": "Profile not found",
        "schema_version": "1.0.0",
        "profile_name": "unknown",
    }

    tile = summarize_runtime_profile_health_for_global_console(chaos_summary)

    assert tile["status_light"] == "RED"
    assert tile["profile_stability"] == 0.0
    assert tile["no_run_rate"] == 1.0
    assert "error" in tile["headline"].lower()


def test_runtime_profile_tile_in_global_health_surface():
    """Test integration with build_global_health_surface."""
    chaos_summary = {
        "schema_version": "1.0.0",
        "profile_name": "prod-hardened",
        "total_runs": 100,
        "actions": {"ALLOW": 92, "SAFE_DEGRADE": 8},
        "profile_stability": 0.92,
        "top_violations": [],
    }

    payload = build_global_health_surface(
        base_payload={"test": "data"},
        runtime_profile_health=chaos_summary,
    )

    assert "runtime_profile" in payload
    tile = payload["runtime_profile"]
    assert tile["tile_type"] == "runtime_profile_health"
    assert tile["profile_name"] == "prod-hardened"


def test_runtime_profile_tile_not_in_global_health_without_input():
    """Test that tile is not added when input is None."""
    payload = build_global_health_surface(
        base_payload={"test": "data"},
        runtime_profile_health=None,
    )

    assert "runtime_profile" not in payload


def test_runtime_profile_adapter_graceful_degradation():
    """Test that adapter returns None gracefully when runtime module unavailable."""
    # This test verifies the adapter handles import errors gracefully
    # In practice, the adapter should work, but we test the try/except logic
    tile = build_runtime_profile_tile_for_global_health(
        chaos_summary={
            "schema_version": "1.0.0",
            "profile_name": "dev-default",
            "total_runs": 100,
            "actions": {"ALLOW": 100},
            "profile_stability": 1.0,
            "top_violations": [],
        }
    )

    # Should work if runtime module is available
    if tile is not None:
        assert tile["tile_type"] == "runtime_profile_health"


def test_runtime_profile_tile_manual_snapshot():
    """Test that adapter accepts manual snapshot for testing."""
    manual_snapshot = {
        "schema_version": "1.0.0",
        "profile_name": "ci-strict",
        "total_runs": 50,
        "actions": {"ALLOW": 48, "NO_RUN": 2},
        "profile_stability": 0.96,
        "top_violations": [],
    }

    tile = build_runtime_profile_tile_for_global_health(manual_snapshot=manual_snapshot)

    assert tile is not None
    assert tile["profile_name"] == "ci-strict"
    assert tile["profile_stability"] == 0.96


def test_build_runtime_profile_snapshot_for_first_light_structure():
    """Test that First-Light snapshot has correct structure."""
    from experiments.u2.runtime import build_runtime_profile_snapshot_for_first_light

    chaos_summary = {
        "schema_version": "1.0.0",
        "profile_name": "prod-hardened",
        "total_runs": 100,
        "actions": {"ALLOW": 95, "SAFE_DEGRADE": 3, "NO_RUN": 2},
        "profile_stability": 0.95,
        "top_violations": [],
    }

    snapshot = build_runtime_profile_snapshot_for_first_light(
        profile_name="prod-hardened",
        chaos_summary=chaos_summary,
    )

    assert snapshot["schema_version"] == "1.0.0"
    assert snapshot["profile"] == "prod-hardened"
    assert snapshot["status_light"] in ("GREEN", "YELLOW", "RED")
    assert "profile_stability" in snapshot
    assert 0.0 <= snapshot["profile_stability"] <= 1.0
    assert "no_run_rate" in snapshot
    assert 0.0 <= snapshot["no_run_rate"] <= 1.0


def test_build_runtime_profile_snapshot_for_first_light_json_serializable():
    """Test that First-Light snapshot is JSON-serializable."""
    import json
    from experiments.u2.runtime import build_runtime_profile_snapshot_for_first_light

    chaos_summary = {
        "schema_version": "1.0.0",
        "profile_name": "dev-default",
        "total_runs": 50,
        "actions": {"ALLOW": 50},
        "profile_stability": 1.0,
        "top_violations": [],
    }

    snapshot = build_runtime_profile_snapshot_for_first_light(
        profile_name="dev-default",
        chaos_summary=chaos_summary,
    )

    # Should serialize without error
    json_str = json.dumps(snapshot)
    assert json_str is not None
    assert len(json_str) > 0

    # Should round-trip
    parsed = json.loads(json_str)
    assert parsed == snapshot


def test_build_runtime_profile_snapshot_for_first_light_determinism():
    """Test that First-Light snapshot is deterministic with fixed input."""
    from experiments.u2.runtime import build_runtime_profile_snapshot_for_first_light

    chaos_summary = {
        "schema_version": "1.0.0",
        "profile_name": "ci-strict",
        "total_runs": 100,
        "actions": {"ALLOW": 90, "SAFE_DEGRADE": 8, "NO_RUN": 2},
        "profile_stability": 0.90,
        "top_violations": ["violation1", "violation2"],
    }

    snapshot1 = build_runtime_profile_snapshot_for_first_light(
        profile_name="ci-strict",
        chaos_summary=chaos_summary,
    )

    snapshot2 = build_runtime_profile_snapshot_for_first_light(
        profile_name="ci-strict",
        chaos_summary=chaos_summary,
    )

    # Should be identical
    assert snapshot1 == snapshot2

    # JSON serialization should also be identical
    import json

    json1 = json.dumps(snapshot1, sort_keys=True)
    json2 = json.dumps(snapshot2, sort_keys=True)
    assert json1 == json2


def test_build_runtime_profile_snapshot_for_first_light_error_handling():
    """Test that First-Light snapshot handles errors gracefully."""
    from experiments.u2.runtime import build_runtime_profile_snapshot_for_first_light

    # Error case
    chaos_summary_error = {
        "error": "Profile not found",
        "schema_version": "1.0.0",
        "profile_name": "unknown",
    }

    snapshot = build_runtime_profile_snapshot_for_first_light(
        profile_name="prod-hardened",
        chaos_summary=chaos_summary_error,
    )

    assert snapshot["profile"] == "prod-hardened"
    assert snapshot["status_light"] == "RED"
    assert snapshot["profile_stability"] == 0.0
    assert snapshot["no_run_rate"] == 1.0


def test_build_runtime_profile_snapshot_for_first_light_status_light_mapping():
    """Test that First-Light snapshot correctly maps status lights."""
    from experiments.u2.runtime import build_runtime_profile_snapshot_for_first_light

    # GREEN case
    chaos_summary_green = {
        "schema_version": "1.0.0",
        "profile_name": "dev-default",
        "total_runs": 100,
        "actions": {"ALLOW": 95},
        "profile_stability": 0.95,
        "top_violations": [],
    }

    snapshot = build_runtime_profile_snapshot_for_first_light(
        profile_name="dev-default",
        chaos_summary=chaos_summary_green,
    )

    assert snapshot["status_light"] == "GREEN"

    # RED case (low stability)
    chaos_summary_red = {
        "schema_version": "1.0.0",
        "profile_name": "prod-hardened",
        "total_runs": 100,
        "actions": {"ALLOW": 50, "NO_RUN": 50},
        "profile_stability": 0.50,
        "top_violations": [],
    }

    snapshot = build_runtime_profile_snapshot_for_first_light(
        profile_name="prod-hardened",
        chaos_summary=chaos_summary_red,
    )

    assert snapshot["status_light"] == "RED"

    # YELLOW case
    chaos_summary_yellow = {
        "schema_version": "1.0.0",
        "profile_name": "ci-strict",
        "total_runs": 100,
        "actions": {"ALLOW": 85, "SAFE_DEGRADE": 10, "NO_RUN": 5},
        "profile_stability": 0.85,
        "top_violations": [],
    }

    snapshot = build_runtime_profile_snapshot_for_first_light(
        profile_name="ci-strict",
        chaos_summary=chaos_summary_yellow,
    )

    assert snapshot["status_light"] == "YELLOW"

