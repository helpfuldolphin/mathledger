import pytest

from verify_dual_root import TILE_SCHEMA_VERSION, summarize_dual_root_health


def test_dual_root_tile_ok_status():
    tile = summarize_dual_root_health([])
    assert tile["status"] == "OK"
    assert tile["mismatch_count"] == 0
    assert tile["schema_version"] == TILE_SCHEMA_VERSION
    assert tile["headline"].startswith("[PASS]")


def test_dual_root_tile_fail_status():
    mismatches = [{
        "epoch": "42",
        "block_id": 1,
        "status": "mismatch",
        "reasoning_merkle_root": "a" * 64,
        "ui_merkle_root": "b" * 64,
        "stored_h_t": "c" * 64,
        "computed_h_t": "d" * 64,
    }]
    tile = summarize_dual_root_health(mismatches)
    assert tile["status"] == "FAIL"
    assert tile["mismatch_count"] == 1
    assert tile["headline"].startswith("[FAIL]")
