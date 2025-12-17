import hashlib
import json
from pathlib import Path

from scripts.verify_proof_snapshot_integrity import (
    verify_proof_snapshot_integrity,
)


def _write_snapshot(path: Path, proof_hashes):
    canonical_hash = hashlib.sha256("\n".join(proof_hashes).encode("utf-8")).hexdigest()
    snapshot_payload = {
        "schema_version": "1.0.0",
        "canonical_hash_algorithm": "sha256",
        "canonicalization_version": "proof-log-v1",
        "source": "/tmp/proofs.jsonl",
        "proof_hashes": proof_hashes,
        "canonical_hash": canonical_hash,
        "entry_count": len(proof_hashes),
    }
    path.write_text(json.dumps(snapshot_payload), encoding="utf-8")
    return snapshot_payload


def test_verify_snapshot_handles_windows_paths(tmp_path):
    pack_dir = tmp_path / "pack"
    pack_dir.mkdir()
    compliance_dir = pack_dir / "compliance"
    compliance_dir.mkdir()
    snapshot_path = compliance_dir / "proof_log_snapshot.json"

    proof_hashes = ["a" * 64, "b" * 64]
    snapshot_payload = _write_snapshot(snapshot_path, proof_hashes)
    snapshot_hash = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()

    manifest = {
        "proof_log_snapshot": {
            "path": "compliance\\proof_log_snapshot.json",
            "sha256": snapshot_hash,
            "schema_version": snapshot_payload["schema_version"],
            "canonical_hash_algorithm": snapshot_payload["canonical_hash_algorithm"],
            "canonicalization_version": snapshot_payload["canonicalization_version"],
            "canonical_hash": snapshot_payload["canonical_hash"],
            "entry_count": snapshot_payload["entry_count"],
            "source": snapshot_payload["source"],
        },
    }
    manifest_path = pack_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = verify_proof_snapshot_integrity(pack_dir=pack_dir)

    assert result["all_checks_passed"]
    assert result["file_hash_match"]
    assert result["canonical_hash_match"]
    assert result["entry_count_match"]


def test_verify_snapshot_legacy_manifest_posix_and_canonical_mismatch(tmp_path):
    """Legacy manifest (minimal fields) + POSIX path normalization + canonical recompute mismatch."""
    pack_dir = tmp_path / "pack"
    pack_dir.mkdir()
    compliance_dir = pack_dir / "compliance"
    compliance_dir.mkdir()
    snapshot_path = compliance_dir / "proof_log_snapshot.json"

    proof_hashes = ["a" * 64, "b" * 64, "c" * 64]
    snapshot_payload = _write_snapshot(snapshot_path, proof_hashes)
    snapshot_payload["canonical_hash"] = "0" * 64  # force mismatch
    snapshot_path.write_text(json.dumps(snapshot_payload), encoding="utf-8")
    snapshot_hash = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()

    manifest = {
        "proof_log_snapshot": {
            "path": "compliance/proof_log_snapshot.json",  # POSIX style
            "sha256": snapshot_hash,  # legacy manifests omit richer metadata
        },
    }
    manifest_path = pack_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = verify_proof_snapshot_integrity(pack_dir=pack_dir)

    assert result["file_hash_match"]
    assert result["entry_count_match"]
    assert not result["canonical_hash_match"]
    assert not result["all_checks_passed"]
