#!/usr/bin/env python3
"""
Dual-Root Auditor

Reads R_t, U_t, and H_t from every block (epoch) and verifies the binding
H_t = SHA256("EPOCH:" || R_t || U_t). Any mismatches are emitted as JSON.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, List, Sequence, Tuple

import psycopg

from attestation.dual_root import compute_composite_root
from backend.security.runtime_env import MissingEnvironmentVariable, get_database_url


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
    """
    Build the health tile emitted to the global dashboard / CI.

    Args:
        results: Sequence of verify_dual_root mismatch dictionaries.

    Returns:
        Dict[str, Any]: tile payload with mismatch counts and headline.
    """
    mismatch_count = len(results)
    status = "pass" if mismatch_count == 0 else "fail"
    headline = (
        "[PASS] Dual roots anchored to epochs"
        if status == "pass"
        else f"[FAIL] Dual-root mismatches detected ({mismatch_count})"
    )
    return {
        "tile_id": "dual_root_integrity",
        "status": status,
        "mismatch_count": mismatch_count,
        "headlines": [headline],
    }


def dual_root_ci_usage() -> str:
    """
    CI wiring instructions for verify_dual_root.py.

    - Invocation: `uv run python verify_dual_root.py > artifacts/tiles/dual_root_integrity.json`
    - Exit codes: 0 => pass, 1 => block/mismatch failure, 2 => infra/env failure.
    - Tile path: write `summarize_dual_root_health(report["mismatches"])` to
      `artifacts/tiles/dual_root_integrity.json` for downstream dashboards.
    """
    return (
        "Run verify_dual_root.py under uv/python, capture stdout to artifacts/tiles/dual_root_integrity.json. "
        "Honor exit codes (0=pass, 1=integrity failure, 2=infra error). "
        "Upload the tile JSON to the CI artifact store for the global health dashboard."
    )


def main() -> None:
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
    print(json.dumps(report, sort_keys=True, indent=2))

    sys.exit(0 if report["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
