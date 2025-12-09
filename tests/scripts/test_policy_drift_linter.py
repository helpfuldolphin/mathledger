import json
import subprocess
import sys
from pathlib import Path


def _write_manifest(path: Path, policy_hash: str) -> None:
    manifest = {
        "schema_version": "policy_manifest@v2",
        "policy": {
            "hash": policy_hash,
            "version": "2025.12.06-r4",
            "model_type": "reranker",
            "serialization": "pickle@3.11",
            "byte_size": 0,
            "created_at": "2025-12-06T13:45:00Z",
        },
    }
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _write_delta_log(path: Path, before_hash: str, after_hash: str, learning_rate: float = 0.1) -> None:
    entry = {
        "cycle": 1,
        "policy_hash_before": before_hash,
        "policy_hash_after": after_hash,
        "learning_rate": learning_rate,
        "timestamp": "2025-12-06T13:45:07Z",
    }
    path.write_text(json.dumps(entry) + "\n", encoding="utf-8")


def _write_surface(path: Path) -> None:
    path.write_text(
        "surface_version: policy_surface@v1\nfeature_space:\n  - name: atoms\n    transform: identity\n",
        encoding="utf-8",
    )


def _write_weights(path: Path, payload: bytes) -> str:
    path.write_bytes(payload)
    import hashlib

    return hashlib.sha256(payload).hexdigest()


def test_policy_linter_detects_hash_mismatch(tmp_path):
    policy_dir = tmp_path / "policy"
    policy_dir.mkdir()
    weights_hash = _write_weights(policy_dir / "policy.weights.bin", b"abc123")
    _write_manifest(policy_dir / "policy.manifest.json", policy_hash="deadbeef")
    _write_delta_log(policy_dir / "delta_log.jsonl", before_hash="deadbeef", after_hash="deadbeef")
    _write_surface(policy_dir / "policy.surface.yaml")

    cmd = [
        sys.executable,
        "scripts/policy_drift_linter.py",
        "--policy-dir",
        str(policy_dir),
        "--lint",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 2
    assert "weights hash" in result.stdout
    assert weights_hash != "deadbeef"


def test_policy_drift_guard_blocks_hash_change(tmp_path):
    policy_dir = tmp_path / "policy"
    policy_dir.mkdir()
    weights_path = policy_dir / "policy.weights.bin"
    baseline_hash = _write_weights(weights_path, b"baseline")
    _write_manifest(policy_dir / "policy.manifest.json", policy_hash=baseline_hash)
    _write_delta_log(policy_dir / "delta_log.jsonl", before_hash=baseline_hash, after_hash=baseline_hash)
    _write_surface(policy_dir / "policy.surface.yaml")

    ledger_path = tmp_path / "ledger.jsonl"
    cmd_snapshot = [
        sys.executable,
        "scripts/policy_drift_linter.py",
        "--policy-dir",
        str(policy_dir),
        "--ledger",
        str(ledger_path),
        "--lint",
        "--snapshot",
    ]
    subprocess.run(cmd_snapshot, check=True)

    # Mutate weights and manifest to introduce drift
    new_hash = _write_weights(weights_path, b"next-policy-state")
    _write_manifest(policy_dir / "policy.manifest.json", policy_hash=new_hash)
    _write_delta_log(policy_dir / "delta_log.jsonl", before_hash=baseline_hash, after_hash=new_hash)

    cmd_drift = [
        sys.executable,
        "scripts/policy_drift_linter.py",
        "--policy-dir",
        str(policy_dir),
        "--ledger",
        str(ledger_path),
        "--snapshot",
        "--drift-check",
        "--quiet",
    ]
    drift_result = subprocess.run(cmd_drift, capture_output=True, text=True)
    assert drift_result.returncode == 1
