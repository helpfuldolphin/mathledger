import json

from scripts.verify_first_light_determinism import compare_json_files


def test_compare_json_files_ignores_only_timestamp_keys(tmp_path):
    file1 = tmp_path / "run1.json"
    file2 = tmp_path / "run2.json"

    run1 = {"timestamp": "2025-01-01T00:00:00Z", "value": 1, "payload": {"k": "v"}}
    run2 = {"timestamp": "2025-01-02T00:00:00Z", "value": 1, "payload": {"k": "v"}}

    file1.write_text(json.dumps(run1), encoding="utf-8")
    file2.write_text(json.dumps(run2), encoding="utf-8")

    identical, diff = compare_json_files(file1, file2, ignore_keys=["timestamp"])
    assert identical is True
    assert diff == ""

    # A non-timestamp change must still be detected under determinism comparison.
    run2_changed = {"timestamp": "2025-01-02T00:00:00Z", "value": 2, "payload": {"k": "v"}}
    file2.write_text(json.dumps(run2_changed), encoding="utf-8")

    identical, diff = compare_json_files(file1, file2, ignore_keys=["timestamp"])
    assert identical is False
    assert "value" in diff
