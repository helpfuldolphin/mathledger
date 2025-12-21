#!/usr/bin/env python3
"""
Dual-Root Auditor

Reads R_t, U_t, and H_t from every block (epoch) and verifies the binding
H_t = SHA256("EPOCH:" || R_t || U_t). Any mismatches are emitted as JSON.

CI Example::

    uv run python verify_dual_root.py \
        --blocks artifacts/blocks.jsonl \
        --tile-output artifacts/tiles/dual_root_integrity.json

- Exit codes: 0 (pass), 1 (integrity failure / mismatches), 2 (infrastructure/env error).
- The tile JSON (written when ``--tile-output`` is set) must be merged into the
  global health feed (e.g., append to ``artifacts/global_health.json``) so the
  dashboard reflects dual-root status and CI gating honors the exit code.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import psycopg

from attestation.dual_root import compute_composite_root
from backend.security.runtime_env import MissingEnvironmentVariable, get_database_url

TILE_SCHEMA_VERSION = "1.0.0"


def fetch_block_records(conn: psycopg.Connection) -> Tuple[List[Dict[str, Any]], bool]:
    """Return all block rows along with schema availability."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'blocks'
            ORDER BY ordinal_position
            """
        )
        columns = {row[0] for row in cur.fetchall()}

        required = {"reasoning_merkle_root", "ui_merkle_root", "composite_attestation_root"}
        if not required.issubset(columns):
            return [], False

        cur.execute(
            """
            SELECT
                id,
                block_number,
                reasoning_merkle_root,
                ui_merkle_root,
                composite_attestation_root
            FROM blocks
            ORDER BY block_number
            """
        )
        rows = cur.fetchall()

    results = []
    for row in rows:
        block_id, block_number, r_t, u_t, h_t = row
        results.append(
            {
                "id": block_id,
                "block_number": block_number,
                "reasoning_merkle_root": r_t,
                "ui_merkle_root": u_t,
                "composite_attestation_root": h_t,
            }
        )
    return results, True


def load_blocks_from_file(path: str) -> Tuple[List[Dict[str, Any]], bool]:
    """Load synthetic block rows from JSON/JSONL."""
    data_path = Path(path)
    text = data_path.read_text(encoding="utf-8").strip()
    if not text:
        return [], False

    if data_path.suffix == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        payload = json.loads(text)
        if isinstance(payload, list):
            rows = payload
        else:
            rows = payload.get("blocks", [])

    required = {"id", "block_number", "reasoning_merkle_root", "ui_merkle_root", "composite_attestation_root"}
    ready = all(required.issubset(row.keys()) for row in rows)
    blocks = [
        {
            "id": row["id"],
            "block_number": row["block_number"],
            "reasoning_merkle_root": row["reasoning_merkle_root"],
            "ui_merkle_root": row["ui_merkle_root"],
            "composite_attestation_root": row["composite_attestation_root"],
        }
        for row in rows
    ]
    return blocks, ready


def recompute_epoch_binding(r_t: str, u_t: str) -> str:
    """Compute H_t' = SHA256("EPOCH:" || R_t || U_t)."""
    return compute_composite_root(r_t, u_t)


def verify_blocks(blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
    mismatches: List[Dict[str, Any]] = []
    verified = 0

    for block in blocks:
        epoch_id = str(block["block_number"])
        r_t = block["reasoning_merkle_root"]
        u_t = block["ui_merkle_root"]
        stored_h_t = block["composite_attestation_root"]

        if not (r_t and u_t and stored_h_t):
            mismatch = {
                "epoch": epoch_id,
                "block_id": block["id"],
                "status": "missing_fields",
                "reason": "one or more dual-root columns are NULL",
            }
            mismatches.append(mismatch)
            print(json.dumps(mismatch, sort_keys=True))
            continue

        computed_h_t = recompute_epoch_binding(r_t, u_t)
        if computed_h_t == stored_h_t:
            verified += 1
            continue

        mismatch = {
            "epoch": epoch_id,
            "block_id": block["id"],
            "status": "mismatch",
            "reasoning_merkle_root": r_t,
            "ui_merkle_root": u_t,
            "stored_h_t": stored_h_t,
            "computed_h_t": computed_h_t,
        }
        mismatches.append(mismatch)
        print(json.dumps(mismatch, sort_keys=True))

    return {
        "total_blocks": len(blocks),
        "verified_blocks": verified,
        "mismatches": mismatches,
    }


def summarize_dual_root_health(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Build the dashboard tile for dual-root integrity."""
    mismatch_count = len(results)
    status = "OK" if mismatch_count == 0 else "FAIL"
    headline = (
        "[PASS] Dual roots anchored to epochs"
        if mismatch_count == 0
        else f"[FAIL] Dual-root mismatches detected ({mismatch_count})"
    )
    return {
        "schema_version": TILE_SCHEMA_VERSION,
        "tile_id": "dual_root_integrity",
        "status": status,
        "mismatch_count": mismatch_count,
        "headline": headline,
    }


def dual_root_ci_usage() -> str:
    """
    CI wiring instructions.

    - Run ``uv run python verify_dual_root.py --blocks artifacts/blocks.jsonl --tile-output artifacts/tiles/dual_root_integrity.json``.
    - Exit codes: 0 pass, 1 mismatch failure, 2 infrastructure/env failure.
    - Append ``artifacts/tiles/dual_root_integrity.json`` to your global health feed for dashboard visibility.
    """
    return (
        "Run `uv run python verify_dual_root.py --blocks artifacts/blocks.jsonl --tile-output "
        "artifacts/tiles/dual_root_integrity.json` and propagate the tile JSON to global health."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify dual-root attestation integrity.")
    parser.add_argument(
        "--blocks",
        help="Optional path to JSON/JSONL containing block records (bypass database).")
    parser.add_argument(
        "--tile-output",
        help="Optional path to write summarize_dual_root_health() JSON tile.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.blocks:
        blocks, ready = load_blocks_from_file(args.blocks)
    else:
        try:
            db_url = get_database_url()
        except MissingEnvironmentVariable as exc:
            print(json.dumps({"status": "error", "reason": str(exc)}))
            sys.exit(2)

        try:
            conn = psycopg.connect(db_url)
        except Exception as exc:  # pragma: no cover
            print(json.dumps({"status": "error", "reason": f"database connection failed: {exc}"}))
            sys.exit(2)

        try:
            blocks, ready = fetch_block_records(conn)
        finally:
            conn.close()

    if not ready:
        print(json.dumps({"status": "error", "reason": "blocks table missing dual-root columns"}))
        sys.exit(1)

    if not blocks:
        print(json.dumps({"status": "abstain", "reason": "no blocks available"}))
        sys.exit(1)

    report = verify_blocks(blocks)
    report["status"] = "pass" if report["mismatches"] == [] else "fail"
    report["coverage_percent"] = (
        (report["verified_blocks"] / report["total_blocks"]) * 100 if report["total_blocks"] else 0.0
    )
    report["tile"] = summarize_dual_root_health(report["mismatches"])
    report["ci_usage"] = dual_root_ci_usage()

    if args.tile_output:
        tile_path = Path(args.tile_output)
        tile_path.parent.mkdir(parents=True, exist_ok=True)
        tile_path.write_text(json.dumps(report["tile"], sort_keys=True, indent=2), encoding="utf-8")

    print(json.dumps(report, sort_keys=True, indent=2))

    sys.exit(0 if report["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
