from backend.ledger.monotone_guard_v2 import check_monotone_ledger


def _make_hash(seed: int) -> str:
    return f"{seed:064x}"


def _header(height: int, prev_hash: str, root_hash: str, timestamp: str) -> dict:
    return {
        "height": height,
        "prev_hash": prev_hash,
        "root_hash": root_hash,
        "timestamp": timestamp,
    }


def test_empty_ledger_is_monotone():
    result = check_monotone_ledger([])
    assert result["is_monotone"] is True
    assert result["violations"] == []


def test_valid_increasing_chain():
    first_root = _make_hash(1)
    second_root = _make_hash(2)
    headers = [
        _header(0, _make_hash(0), first_root, "2024-01-01T00:00:00Z"),
        _header(1, first_root, second_root, "2024-01-01T00:01:00Z"),
    ]

    result = check_monotone_ledger(headers)

    assert result["is_monotone"] is True
    assert result["violations"] == []


def test_height_regression_detected():
    first_root = _make_hash(1)
    headers = [
        _header(1, _make_hash(0), first_root, "2024-01-01T00:00:00Z"),
        _header(1, first_root, _make_hash(2), "2024-01-01T00:01:00Z"),
    ]

    result = check_monotone_ledger(headers)

    assert result["is_monotone"] is False
    assert any("height" in violation for violation in result["violations"])


def test_prev_hash_mismatch_detected():
    first_root = _make_hash(1)
    headers = [
        _header(0, _make_hash(0), first_root, "2024-01-01T00:00:00Z"),
        _header(1, _make_hash(999), _make_hash(2), "2024-01-01T00:01:00Z"),
    ]

    result = check_monotone_ledger(headers)

    assert result["is_monotone"] is False
    assert any("prev_hash" in violation for violation in result["violations"])


def test_invalid_header_schema_is_reported():
    invalid_header = {
        "height": -1,
        "root_hash": _make_hash(2),
        "timestamp": "2024-01-01T00:00:00Z",
    }

    result = check_monotone_ledger([invalid_header])

    assert result["is_monotone"] is False
    assert any("Missing field 'prev_hash'" in violation for violation in result["violations"])