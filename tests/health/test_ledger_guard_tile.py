import json

from backend.health.ledger_guard_tile import build_ledger_guard_tile


def _result(violations):
    return {
        "schema_version": "1.0.0",
        "is_monotone": not violations,
        "violations": violations,
    }


def test_tile_green_when_no_violations():
    tile = build_ledger_guard_tile(_result([]))

    assert tile["status_light"] == "GREEN"
    assert tile["violation_count"] == 0
    assert tile["headline"].startswith("Ledger guard v2")
    assert "!" not in tile["headline"]


def test_tile_yellow_for_schema_issue():
    violations = ["header[0]: Field 'height' must be >= 0"]

    tile = build_ledger_guard_tile(_result(violations))

    assert tile["status_light"] == "YELLOW"
    assert tile["violation_count"] == len(violations)


def test_tile_red_for_parent_violation():
    violations = [
        "header[1]: prev_hash 00 does not match previous root_hash 11",
        "header[1]: height 4 is not greater than previous height 5",
    ]

    tile = build_ledger_guard_tile(_result(violations))

    assert tile["status_light"] == "RED"
    assert tile["violation_count"] == len(violations)


def test_tile_is_deterministic_and_json_serializable():
    violations = ["header[0]: Missing field 'prev_hash'"]

    tile_one = build_ledger_guard_tile(_result(violations))
    tile_two = build_ledger_guard_tile(_result(violations))

    assert tile_one == tile_two
    # Must be JSON safe for UI payload
    assert json.loads(json.dumps(tile_one)) == tile_one