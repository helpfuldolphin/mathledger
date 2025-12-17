import io
import json
import hashlib
from pathlib import Path

from scripts.proof_hash_tools_v2 import (
    backfill_hashes,
    compute_canonical_diff,
    compute_proof_hash,
)
from scripts.first_light_proof_hash_snapshot import generate_snapshot


def _jsonl_string(records):
    return "\n".join(json.dumps(obj) for obj in records) + "\n"


def test_compute_proof_hash_deterministic():
    proof = {
        "id": "proof-123",
        "steps": [{"type": "axiom", "args": [1, 2]}],
        "metadata": {"author": "phase-x"},
    }
    hash_one = compute_proof_hash(proof)

    reordered = {
        "metadata": {"author": "phase-x"},
        "steps": [{"args": [1, 2], "type": "axiom"}],
        "id": "proof-123",
    }
    reordered["proof_hash"] = "stale"
    hash_two = compute_proof_hash(reordered)

    assert hash_one == hash_two
    assert len(hash_one) == 64


def test_backfill_hashes_uses_in_memory_streams():
    raw_records = [
        {"id": "p1", "steps": [1, 2, 3]},
        {"id": "p2", "steps": [{"goal": "foo"}]},
    ]
    input_buffer = io.StringIO(_jsonl_string(raw_records))
    output_buffer = io.StringIO()

    backfill_hashes(input_buffer, output_buffer)

    output_buffer.seek(0)
    lines = [json.loads(line) for line in output_buffer.read().strip().splitlines()]
    assert len(lines) == len(raw_records)

    for obj, original in zip(lines, raw_records):
        assert "proof_hash" in obj
        expected_hash = compute_proof_hash(original)
        assert obj["proof_hash"] == expected_hash


def test_compute_canonical_diff_in_memory_sources():
    base_records = [
        {"id": "old-only", "payload": "legacy"},
        {"id": "shared", "payload": "common"},
    ]
    new_records = [
        {"id": "shared", "payload": "common"},
        {"id": "new-only", "payload": "fresh"},
    ]

    old_buffer = io.StringIO(_jsonl_string(base_records))
    new_buffer = io.StringIO(_jsonl_string(new_records))

    diff = compute_canonical_diff(old_buffer, new_buffer)

    shared_hash = compute_proof_hash(base_records[1])
    removed_hash = compute_proof_hash(base_records[0])
    added_hash = compute_proof_hash(new_records[1])

    assert diff["unchanged"] == [shared_hash]
    assert diff["removed"] == [removed_hash]
    assert diff["added"] == [added_hash]


def test_first_light_snapshot_generation(tmp_path):
    records = [
        {"id": "alpha", "payload": {"value": 1}},
        {"id": "beta", "payload": {"value": 2}},
    ]
    proof_log = tmp_path / "proofs.jsonl"
    proof_log.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")

    snapshot_path = tmp_path / "snapshot.json"
    snapshot = generate_snapshot(str(proof_log), str(snapshot_path))

    assert snapshot_path.exists()
    data = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert data == snapshot
    assert data["schema_version"] == "1.0.0"
    assert data["entry_count"] == len(records)
    assert data["canonical_hash_algorithm"] == "sha256"
    assert data["canonicalization_version"] == "proof-log-v1"

    expected_hashes = sorted(compute_proof_hash(obj) for obj in records)
    canonical_payload = "\n".join(expected_hashes).encode("utf-8")
    expected_canonical = hashlib.sha256(canonical_payload).hexdigest()
    assert data["canonical_hash"] == expected_canonical
    assert data["proof_hashes"] == expected_hashes
    assert Path(data["source"]).name == proof_log.name
