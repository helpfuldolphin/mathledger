"""
Tests for runtime profile First-Light snapshot helper.

Validates:
- Snapshot structure and fields
- JSON serializability
- Determinism with fixed input
- Error handling
"""

import json

import pytest


def test_build_runtime_profile_snapshot_for_first_light_structure():
    """Test that First-Light snapshot has correct structure."""
    # Import directly to avoid dependency on full u2 module
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


def test_build_runtime_profile_snapshot_for_first_light_rounding():
    """Test that First-Light snapshot rounds floating point values."""
    from experiments.u2.runtime import build_runtime_profile_snapshot_for_first_light

    chaos_summary = {
        "schema_version": "1.0.0",
        "profile_name": "dev-default",
        "total_runs": 100,
        "actions": {"ALLOW": 95, "NO_RUN": 5},
        "profile_stability": 0.950123456789,  # Should round to 0.9501
        "top_violations": [],
    }

    snapshot = build_runtime_profile_snapshot_for_first_light(
        profile_name="dev-default",
        chaos_summary=chaos_summary,
    )

    # Should be rounded to 4 decimal places
    assert snapshot["profile_stability"] == 0.9501
    assert snapshot["no_run_rate"] == 0.05

