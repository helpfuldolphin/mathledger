import hashlib
import json
from pathlib import Path


def _write_snapshot(path: Path, proof_hashes: list[str], *, canonical_hash: str | None = None) -> None:
    canonical = canonical_hash or hashlib.sha256(
        "\n".join(proof_hashes).encode("utf-8")
    ).hexdigest()
    payload = {
        "schema_version": "1.0.0",
        "canonical_hash_algorithm": "sha256",
        "canonicalization_version": "proof-log-v1",
        "source": "/tmp/proofs.jsonl",
        "proof_hashes": proof_hashes,
        "canonical_hash": canonical,
        "entry_count": len(proof_hashes),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_status_proof_snapshot_integrity_canonical_mismatch_is_advisory(tmp_path):
    from scripts.generate_first_light_status import generate_status

    p3_dir = tmp_path / "p3"
    p4_dir = tmp_path / "p4"
    pack_dir = tmp_path / "pack"
    p3_dir.mkdir()
    p4_dir.mkdir()
    pack_dir.mkdir()
    compliance_dir = pack_dir / "compliance"
    compliance_dir.mkdir()

    snapshot_path = compliance_dir / "proof_log_snapshot.json"
    proof_hashes = ["a" * 64, "b" * 64]

    # Baseline snapshot (valid canonical hash)
    _write_snapshot(snapshot_path, proof_hashes)
    snapshot_sha = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": "1.0.0",
        "mode": "SHADOW",
        "file_count": 1,
        "shadow_mode_compliance": {
            "all_divergence_logged_only": True,
            "no_governance_modification": True,
            "no_abort_enforcement": True,
        },
        "files": [
            {"path": "compliance/proof_log_snapshot.json", "sha256": snapshot_sha},
        ],
        "proof_log_snapshot": {
            "path": "compliance/proof_log_snapshot.json",
            "sha256": snapshot_sha,
        },
    }
    manifest_path = pack_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    status_ok = generate_status(p3_dir, p4_dir, pack_dir)
    assert status_ok["evidence_pack_ok"] is True

    # Corrupt canonical hash but keep file hash aligned with manifest.
    _write_snapshot(snapshot_path, proof_hashes, canonical_hash="0" * 64)
    corrupted_sha = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
    manifest["proof_log_snapshot"]["sha256"] = corrupted_sha
    manifest["files"][0]["sha256"] = corrupted_sha
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    status_bad = generate_status(p3_dir, p4_dir, pack_dir)

    assert status_bad["evidence_pack_ok"] is True
    assert status_bad["evidence_pack_ok"] == status_ok["evidence_pack_ok"]

    integrity = status_bad["proof_snapshot_integrity"]
    assert integrity is not None
    assert integrity["ok"] is False
    assert integrity["extraction_source"] == "STATUS_CHECK"

    canonical_codes = {
        "MISSING_FILE",
        "SHA256_MISMATCH",
        "CANONICAL_HASH_MISMATCH",
        "ENTRY_COUNT_MISMATCH",
    }
    assert integrity["failure_codes"] == ["CANONICAL_HASH_MISMATCH"]
    assert set(integrity["failure_codes"]).issubset(canonical_codes)

    integrity_warnings = [
        w for w in status_bad["warnings"] if "Proof snapshot integrity advisory failed" in w
    ]
    assert integrity_warnings
    assert len(integrity_warnings) == 1
    assert "CANONICAL_HASH_MISMATCH" in integrity_warnings[0]
