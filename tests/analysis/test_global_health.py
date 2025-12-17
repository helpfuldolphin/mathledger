from analysis.global_health import attach_pq_policy_tile, build_pq_policy_tile


def test_build_pq_policy_tile_clean():
    verdicts = [
        {"ok": True, "epoch": "legacy", "block_number": 5, "code": None},
        {"ok": True, "epoch": "legacy", "block_number": 6, "code": None},
    ]

    tile = build_pq_policy_tile(verdicts)

    assert tile["status"] == "pass"
    assert tile["violation_count"] == 0
    assert tile["latest_block"] == 6
    assert tile["latest_epoch"] == "legacy"
    assert tile["violation_codes"] == []
    assert tile["headline"] == "PQ policy clean"


def test_attach_pq_policy_tile_with_alert():
    verdicts = [
        {"ok": True, "epoch": "legacy", "block_number": 5, "code": None},
        {"ok": False, "epoch": "transition", "block_number": 12, "code": "DUAL_COMMITMENT_REQUIRED"},
    ]

    global_health = attach_pq_policy_tile({}, verdicts)

    assert "pq_policy" in global_health
    tile = global_health["pq_policy"]
    assert tile["status"] == "alert"
    assert tile["violation_count"] == 1
    assert tile["latest_epoch"] == "transition"
    assert tile["latest_block"] == 12
    assert tile["violation_codes"] == ["DUAL_COMMITMENT_REQUIRED"]
    assert "violations detected" in tile["headline"]
