"""Advisory validator for proof_log_snapshot.json artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Optional, Tuple


def _compute_file_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _normalize_manifest_path(value: str) -> str:
    return PurePosixPath(value.replace("\\", "/")).as_posix()


def _compute_canonical_hash_from_list(hashes: list[str]) -> str:
    payload = "\n".join(hashes).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def verify_proof_snapshot_integrity(
    *,
    pack_dir: Optional[Path],
    manifest_path: Optional[Path] = None,
    snapshot_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Verify proof snapshot integrity against the manifest.

    Returns a dictionary containing pass/fail booleans for each check.
    """
    if manifest_path is None:
        if pack_dir is None:
            raise ValueError("pack_dir or manifest_path must be provided")
        manifest_path = pack_dir / "manifest.json"

    manifest_path = manifest_path.resolve()
    pack_root = pack_dir.resolve() if pack_dir else manifest_path.parent

    result: Dict[str, Any] = {
        "manifest_path": str(manifest_path),
        "snapshot_path": None,
        "file_hash_match": False,
        "canonical_hash_match": False,
        "entry_count_match": False,
        "all_checks_passed": False,
        "errors": [],
    }

    if not manifest_path.exists():
        result["errors"].append(f"Manifest not found: {manifest_path}")
        return result

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        result["errors"].append(f"Failed to parse manifest: {exc}")
        return result

    snapshot_meta = manifest.get("proof_log_snapshot")
    if not snapshot_meta:
        result["errors"].append("Manifest missing proof_log_snapshot metadata")
        return result

    manifest_snapshot_path = snapshot_meta.get("path")
    if snapshot_path is None:
        if not manifest_snapshot_path:
            result["errors"].append("Snapshot path missing from manifest metadata")
            return result
        normalized = _normalize_manifest_path(manifest_snapshot_path)
        snapshot_path = (pack_root / Path(normalized)).resolve()
    else:
        snapshot_path = Path(snapshot_path).resolve()
        normalized = _normalize_manifest_path(str(snapshot_path.relative_to(pack_root)))

    result["snapshot_path"] = str(snapshot_path)

    if not snapshot_path.exists():
        result["errors"].append(f"Snapshot file not found: {snapshot_path}")
        return result

    file_hash = _compute_file_hash(snapshot_path)
    result["file_hash_match"] = file_hash == snapshot_meta.get("sha256")

    try:
        snapshot_payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        result["errors"].append(f"Failed to parse snapshot JSON: {exc}")
        return result

    proof_hashes = snapshot_payload.get("proof_hashes")
    if not isinstance(proof_hashes, list) or not all(isinstance(h, str) for h in proof_hashes):
        result["errors"].append("Snapshot missing proof_hashes array")
        proof_hashes = []

    recomputed_canonical = _compute_canonical_hash_from_list(proof_hashes) if proof_hashes else None
    if recomputed_canonical:
        result["canonical_hash_match"] = (
            recomputed_canonical == snapshot_payload.get("canonical_hash")
        )

    snapshot_count = snapshot_payload.get("entry_count")
    result["entry_count_match"] = isinstance(snapshot_count, int) and snapshot_count == len(
        proof_hashes
    )

    checks = [
        result["file_hash_match"],
        result["canonical_hash_match"],
        result["entry_count_match"],
    ]
    result["all_checks_passed"] = all(checks)
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Advisory proof snapshot verifier (non-gating)."
    )
    parser.add_argument(
        "--pack-dir",
        type=str,
        help="Evidence pack directory containing manifest.json and snapshot.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        help="Path to manifest.json (defaults to <pack-dir>/manifest.json).",
    )
    parser.add_argument(
        "--snapshot",
        type=str,
        help="Explicit path to proof_log_snapshot.json (optional).",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    pack_dir = Path(args.pack_dir) if args.pack_dir else None
    manifest_path = Path(args.manifest) if args.manifest else None
    snapshot_path = Path(args.snapshot) if args.snapshot else None

    result = verify_proof_snapshot_integrity(
        pack_dir=pack_dir,
        manifest_path=manifest_path,
        snapshot_path=snapshot_path,
    )

    if result["errors"]:
        print("Verification errors:")
        for err in result["errors"]:
            print(f"  - {err}")
        return 1

    print(f"Snapshot path: {result['snapshot_path']}")
    print(f"Manifest path: {result['manifest_path']}")
    print(f"File hash matches manifest: {result['file_hash_match']}")
    print(f"Canonical hash matches snapshot payload: {result['canonical_hash_match']}")
    print(f"Entry count matches proof_hashes length: {result['entry_count_match']}")

    return 0 if result["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
