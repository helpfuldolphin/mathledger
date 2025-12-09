#!/usr/bin/env python3
"""
Archive a canonical policy directory with verification logs.

Creates rollback-safe snapshots under archive/policy/<timestamp>_<hash>.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
from pathlib import Path

ARCHIVE_ROOT = Path("archive/policy")
POLICY_MANIFEST_FILE = "policy.manifest.json"
POLICY_WEIGHTS_FILE = "policy.weights.bin"
VERIFICATION_FILE = "verification.log"
ROLLBACK_PLAN_FILE = "rollback_plan.md"


def main() -> None:
    parser = argparse.ArgumentParser(description="Archive policy artifacts with verification logs.")
    parser.add_argument("--policy-dir", required=True, help="Path to canonical policy directory.")
    parser.add_argument("--archive-root", default=str(ARCHIVE_ROOT), help="Root directory for archives.")
    parser.add_argument("--ledger-log", help="Optional path to ledger excerpt JSONL to store with archive.")
    parser.add_argument("--force", action="store_true", help="Overwrite archive directory if it exists.")
    args = parser.parse_args()

    policy_dir = Path(args.policy_dir).resolve()
    if not policy_dir.exists():
        raise FileNotFoundError(f"Policy directory not found: {policy_dir}")

    manifest = _load_manifest(policy_dir / POLICY_MANIFEST_FILE)
    policy_hash = manifest.get("policy", {}).get("hash")
    if not policy_hash:
        raise ValueError("Manifest missing policy.hash")

    timestamp = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    archive_root = Path(args.archive_root).resolve()
    destination = archive_root / f"{timestamp}_{policy_hash[:12]}"

    if destination.exists():
        if not args.force:
            raise FileExistsError(f"Archive already exists: {destination}")
        shutil.rmtree(destination)

    shutil.copytree(policy_dir, destination)

    if args.ledger_log:
        ledger_path = Path(args.ledger_log)
        if ledger_path.exists():
            shutil.copy2(ledger_path, destination / "ledger_excerpt.jsonl")

    verification_entries = _build_verification_entries(destination)
    (destination / VERIFICATION_FILE).write_text("\n".join(verification_entries) + "\n", encoding="utf-8")
    (destination / ROLLBACK_PLAN_FILE).write_text(_rollback_template(policy_hash), encoding="utf-8")

    print(f"[policy-archive] archived {policy_dir} -> {destination}")


def _load_manifest(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _build_verification_entries(policy_dir: Path) -> list[str]:
    manifest_path = policy_dir / POLICY_MANIFEST_FILE
    weights_path = policy_dir / POLICY_WEIGHTS_FILE
    entries = [
        f"archived_at={dt.datetime.utcnow().isoformat()}Z",
        f"manifest={manifest_path}",
        f"manifest_sha256={_sha256(manifest_path)}",
    ]
    if weights_path.exists():
        entries.append(f"weights={weights_path}")
        entries.append(f"weights_sha256={_sha256(weights_path)}")
    else:
        entries.append("weights=missing")
    return entries


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rollback_template(policy_hash: str) -> str:
    return (
        "# Policy Rollback Plan\n"
        f"- Archived policy hash: {policy_hash}\n"
        "- To restore:\n"
        "  1. Copy this directory back into artifacts/policy/<hash>/.\n"
        "  2. Update artifacts/policy/policy.json or policy.manifest pointer to reference the restored hash.\n"
        "  3. Re-run verification.log commands to confirm checksums.\n"
    )


if __name__ == "__main__":
    main()
