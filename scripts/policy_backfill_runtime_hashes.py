#!/usr/bin/env python3
"""
Backfill policy runtime metadata using canonical manifests.

Steps:
1. Enumerate artifacts/policy/<hash>/policy.manifest.json.
2. Select the head hash (explicit via --head-hash or most recent).
3. In --dry-run mode, log intended future writes.
4. In --apply mode, rewrite artifacts/policy/policy.json and optionally update policy_settings.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

try:  # Optional dependency for DB updates
    import psycopg  # type: ignore
except Exception:  # pragma: no cover
    psycopg = None

from backend.axiom_engine.policy import load_policy_manifest

HEAD_POINTER = "policy.json"
POLICY_MANIFEST_FILE = "policy.manifest.json"


@dataclass
class PolicyMetadata:
    hash: str
    version: Optional[str]
    model_type: Optional[str]
    created_at: Optional[str]
    path: Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync policy manifests into runtime pointers and DB metadata.")
    parser.add_argument("--policy-root", default="artifacts/policy", help="Root directory containing policy hashes.")
    parser.add_argument("--head-hash", help="Optional policy hash to set as active head pointer.")
    parser.add_argument("--db-url", help="Optional DATABASE_URL for updating policy_settings (requires --apply).")
    parser.add_argument("--dry-run", action="store_true", help="Log-only mode (default).")
    parser.add_argument("--apply", action="store_true", help="Write policy.json and update policy_settings.")
    args = parser.parse_args()

    if args.dry_run and args.apply:
        raise SystemExit("--dry-run and --apply are mutually exclusive")

    dry_run = args.dry_run or not args.apply

    root = Path(args.policy_root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Policy root not found: {root}")

    manifests = _collect_manifests(root)
    if not manifests:
        print(f"[policy-backfill] no manifests found in {root}")
        return

    head_hash = args.head_hash or _determine_head_hash(root, manifests)
    if not head_hash:
        raise RuntimeError("Unable to determine head hash. Provide --head-hash.")
    if head_hash not in manifests:
        raise RuntimeError(f"Head hash {head_hash} not found under {root}.")

    head_manifest = manifests[head_hash]
    print(f"[policy-backfill] head={head_hash} version={head_manifest.version} path={head_manifest.path}")

    if dry_run:
        print("[policy-backfill] DRY-RUN: no files or database rows will be modified")
    else:
        _write_head_pointer(root, head_manifest)
        if args.db_url:
            _update_policy_settings(args.db_url, head_manifest.hash, head_manifest.version)
        elif head_manifest.version:
            print("[policy-backfill] tip: pass --db-url to sync policy_settings with manifest version")

    for meta in manifests.values():
        print(
            f"[policy-backfill] discovered hash={meta.hash} "
            f"version={meta.version} created_at={meta.created_at}"
        )


def _collect_manifests(root: Path) -> Dict[str, PolicyMetadata]:
    manifests: Dict[str, PolicyMetadata] = {}
    for child in root.iterdir():
        if not child.is_dir():
            continue
        manifest_path = child / POLICY_MANIFEST_FILE
        if not manifest_path.exists():
            continue
        manifest = load_policy_manifest(str(manifest_path))
        if not manifest:
            continue
        policy_section = manifest.get("policy", {})
        policy_hash = policy_section.get("hash")
        if not policy_hash:
            continue
        manifests[policy_hash] = PolicyMetadata(
            hash=policy_hash,
            version=policy_section.get("version"),
            model_type=policy_section.get("model_type"),
            created_at=policy_section.get("created_at"),
            path=child,
        )
    return manifests


def _determine_head_hash(root: Path, manifests: Dict[str, PolicyMetadata]) -> Optional[str]:
    pointer_path = root / HEAD_POINTER
    if pointer_path.exists():
        try:
            payload = json.loads(pointer_path.read_text(encoding="utf-8"))
            return payload.get("hash")
        except Exception:
            pass

    sorted_entries = sorted(
        manifests.values(),
        key=lambda meta: meta.created_at or "",
        reverse=True,
    )
    if sorted_entries:
        return sorted_entries[0].hash
    return None


def _write_head_pointer(root: Path, meta: PolicyMetadata) -> None:
    pointer_path = root / HEAD_POINTER
    payload = {
        "hash": meta.hash,
        "version": meta.version,
        "model_type": meta.model_type,
        "created_at": meta.created_at,
    }
    pointer_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[policy-backfill] wrote head pointer {pointer_path}")


def _update_policy_settings(db_url: str, policy_hash: str, policy_version: Optional[str]) -> None:
    if not psycopg:
        raise RuntimeError("psycopg is required for --db-url operations")

    with psycopg.connect(db_url, connect_timeout=5) as conn, conn.cursor() as cur:  # pragma: no cover
        cols = _get_table_columns(cur, "policy_settings")
        _upsert_policy_setting(cur, cols, "active_policy_hash", policy_hash)
        if policy_version:
            _upsert_policy_setting(cur, cols, "active_policy_version", policy_version)
        conn.commit()
        print("[policy-backfill] updated policy_settings table")


def _get_table_columns(cur, table: str) -> list[str]:
    cur.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = %s
        """,
        (table,),
    )
    return [row[0] for row in cur.fetchall()]


def _upsert_policy_setting(cur, cols: list[str], key: str, value: str) -> None:
    if "key" in cols and "value" in cols:
        cur.execute(
            """
            INSERT INTO policy_settings (key, value)
            VALUES (%s, %s)
            ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()
            """,
            (key, value),
        )
        return
    if key == "active_policy_hash" and "policy_hash" in cols:
        cur.execute(
            "INSERT INTO policy_settings (policy_hash) VALUES (%s)",
            (value,),
        )


if __name__ == "__main__":
    main()
