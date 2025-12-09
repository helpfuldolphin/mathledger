import json
import pickle
import subprocess
import sys
from pathlib import Path


def _create_mock_policy(path: Path) -> None:
    from backend.axiom_engine.policy import MockPolicy

    with path.open("wb") as handle:
        pickle.dump(MockPolicy(), handle)


def test_policy_migrate_cli_generates_manifest(tmp_path):
    source_dir = tmp_path / "legacy"
    source_dir.mkdir()
    weights_path = source_dir / "policy.bin"
    _create_mock_policy(weights_path)
    (source_dir / "policy.json").write_text(
        json.dumps({"hash": "legacy", "version": "legacy"}),
        encoding="utf-8",
    )

    output_root = tmp_path / "out"
    cmd = [
        sys.executable,
        "scripts/policy_migrate.py",
        "--source",
        str(weights_path),
        "--output-root",
        str(output_root),
    ]
    subprocess.run(cmd, check=True)
    migrated_dirs = list(output_root.iterdir())
    assert migrated_dirs, "expected at least one migrated policy directory"
    manifest_path = migrated_dirs[0] / "policy.manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["policy"]["hash"]
    weights_out = migrated_dirs[0] / "policy.weights.bin"
    assert weights_out.exists()
