import argparse
import hashlib
import json
from pathlib import Path

from scripts.golden_run_lib import (
    build_artifact_snapshot,
    compute_ht_series_hash,
    diff_artifacts,
    summarize_golden_runs_for_global_health,
)
from scripts.validate_golden_run import validate_replay


def _write_jsonl(path: Path, records):
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")


def test_compute_ht_series_hash_deterministic(tmp_path):
    log_path = tmp_path / "ht.jsonl"
    records = [
        {"cycle": 0, "roots": {"h_t": "a" * 64}},
        {"cycle": 1, "roots": {"h_t": "b" * 64}},
    ]
    _write_jsonl(log_path, records)

    expected_digest = hashlib.sha256()
    for record in records:
        expected_digest.update(record["roots"]["h_t"].encode("utf-8"))
        expected_digest.update(b"\n")

    digest, metadata = compute_ht_series_hash(log_path)
    assert digest == expected_digest.hexdigest()
    assert metadata["cycles_included"] == 2
    assert metadata["cycle_range"] == [0, 1]


def test_diff_artifacts_reports_trace_mismatch(tmp_path):
    ht_path = tmp_path / "ht.jsonl"
    trace_path = tmp_path / "trace.jsonl"
    metrics_path = tmp_path / "metrics.json"

    _write_jsonl(ht_path, [{"cycle": 0, "roots": {"h_t": "c" * 64}}])
    trace_path.write_text('{"event_type": "session_start"}\n', encoding="utf-8")
    metrics_path.write_text(json.dumps({"throughput": 1.0}), encoding="utf-8")

    snapshot = build_artifact_snapshot(ht_path, trace_path, metrics_path)
    observed = {
        "ht_series_hash": snapshot.ht_series_hash,
        "ht_series_metadata": snapshot.ht_series_metadata,
        "trace_hash": snapshot.trace_hash,
        "metrics_snapshot_hash": snapshot.metrics_snapshot_hash,
    }
    golden = observed.copy()
    golden["trace_hash"] = "deadbeef"

    diffs = diff_artifacts(golden, observed)
    assert len(diffs) == 1
    assert "trace_hash mismatch" in diffs[0]


def test_validate_replay_detects_mismatch(tmp_path):
    ht_path = tmp_path / "ht.jsonl"
    trace_path = tmp_path / "trace.jsonl"
    metrics_path = tmp_path / "metrics.json"

    _write_jsonl(
        ht_path,
        [
            {"cycle": 0, "roots": {"h_t": "1" * 64}},
            {"cycle": 1, "roots": {"h_t": "2" * 64}},
        ],
    )
    trace_path.write_text('{"event_type":"session_start"}\n', encoding="utf-8")
    metrics_path.write_text(json.dumps({"success_rate": 0.5}), encoding="utf-8")

    snapshot = build_artifact_snapshot(ht_path, trace_path, metrics_path)
    record = {
        "schema_version": "1.0.0",
        "run_id": "test",
        "command": "python -c \"print('noop')\"",
        "command_args": ["python", "-c", "print('noop')"],
        "env": {"variables": {}},
        "ht_series_hash": snapshot.ht_series_hash,
        "ht_series_metadata": snapshot.ht_series_metadata,
        "trace_hash": snapshot.trace_hash,
        "metrics_snapshot": snapshot.metrics_snapshot,
        "metrics_snapshot_hash": snapshot.metrics_snapshot_hash,
        "artifacts": {
            "ht_log": {"path": str(ht_path)},
            "trace_log": {"path": str(trace_path)},
            "metrics": {"path": str(metrics_path)},
        },
    }

    golden_path = tmp_path / "golden.json"
    golden_path.write_text(json.dumps(record), encoding="utf-8")

    # Mutate the trace log after the record was created to force a mismatch.
    trace_path.write_text('{"event_type":"session_start","cycle":1}\n', encoding="utf-8")

    args = argparse.Namespace(
        golden=str(golden_path),
        ht_log=str(ht_path),
        trace_log=str(trace_path),
        metrics=str(metrics_path),
        set_env=[],
        skip_command=True,
        command=[],
        policy=None,
        output=None,
        advisory=False,
    )

    exit_code = validate_replay(args)
    assert exit_code == 1


def test_summarize_golden_runs_for_global_health_json_safe():
    results = [
        {"name": "golden#1", "status": "OK"},
        {"name": "golden#2", "status": "MISMATCH"},
        {"name": "golden#3", "status": "ERROR"},
    ]
    summary = summarize_golden_runs_for_global_health(results)
    assert summary["schema_version"] == "1.0.0"
    assert summary["status"] == "DEGRADED"
    assert summary["mismatch_count"] == 2
    assert summary["headline"] == "1/3 golden runs matched"
    encoded = json.dumps(summary, sort_keys=True)
    assert encoded == json.dumps(summary, sort_keys=True)
