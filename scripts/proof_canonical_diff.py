#!/usr/bin/env python3
"""
Emit unified diffs between stored Lean proof text and its canonical form.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import sys
from typing import Iterable, Mapping, Sequence

import psycopg
from psycopg.rows import dict_row

from normalization.proof import canonicalize_module_name, canonicalize_proof_text


def _require_database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "[FATAL] DATABASE_URL environment variable is not set. "
            "Export it before running proof canonical diffs."
        )
    return url


def build_diff(
    original: str,
    canonical: str,
    module_name: str,
    canonical_module: str,
) -> str:
    proof_diff = list(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            canonical.splitlines(keepends=True),
            fromfile="proof_text",
            tofile="proof_text (canonical)",
            lineterm="",
        )
    )
    module_diff: Sequence[str] = ()
    if module_name != canonical_module:
        module_diff = list(
            difflib.unified_diff(
                [f"{module_name}\n"],
                [f"{canonical_module}\n"],
                fromfile="module_name",
                tofile="module_name (canonical)",
                lineterm="",
            )
        )
    if module_diff:
        proof_diff.append("")  # visual separator
        proof_diff.extend(module_diff)
    return "\n".join(proof_diff)


def fetch_proof(conn: psycopg.Connection, proof_id: int) -> Mapping[str, object] | None:
    sql = """
        SELECT id, proof_text, module_name
        FROM proofs
        WHERE id = %s
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, (proof_id,))
        return cur.fetchone()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show canonicalization diffs for Lean proofs.")
    parser.add_argument("--proof-id", type=int, help="Proof ID to diff.")
    parser.add_argument(
        "--from-jsonl",
        type=str,
        help="Path to JSONL report (from proof_hash_backfill) to diff all changed proofs.",
    )
    return parser.parse_args()


def render_diff(conn: psycopg.Connection, proof_id: int) -> str:
    row = fetch_proof(conn, proof_id)
    if not row:
        raise ValueError(f"proof_id {proof_id} not found")
    original_text = row.get("proof_text") or ""
    canonical_text = canonicalize_proof_text(original_text)
    module_name = row.get("module_name") or ""
    canonical_module = canonicalize_module_name(module_name)
    return build_diff(original_text, canonical_text, module_name, canonical_module)


def iter_changed_ids(path: str) -> Iterable[int]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            data = json.loads(line)
            if data.get("changed"):
                yield int(data["proof_id"])


def main() -> None:
    args = parse_args()
    if not args.proof_id and not args.from_jsonl:
        print("Must supply --proof-id or --from-jsonl.", file=sys.stderr)
        sys.exit(2)
    if args.proof_id and args.from_jsonl:
        print("Choose either --proof-id or --from-jsonl, not both.", file=sys.stderr)
        sys.exit(2)

    db_url = _require_database_url()
    with psycopg.connect(db_url) as conn:
        if args.from_jsonl:
            ids = list(iter_changed_ids(args.from_jsonl))
            if not ids:
                print("[INFO] No changed proofs detected in JSONL.")
                return
            for proof_id in ids:
                try:
                    diff_text = render_diff(conn, proof_id)
                except ValueError as exc:
                    print(f"[WARN] {exc}", file=sys.stderr)
                    continue
                if diff_text:
                    print(f"# proof_id {proof_id}")
                    print(diff_text)
                    print()
                else:
                    print(f"[INFO] proof_id {proof_id} already canonical.")
        else:
            try:
                diff_text = render_diff(conn, args.proof_id)
            except ValueError as exc:
                print(f"[ERROR] {exc}", file=sys.stderr)
                sys.exit(1)
            if not diff_text:
                print(f"[INFO] No differences detected for proof_id {args.proof_id}.")
            else:
                print(diff_text)


if __name__ == "__main__":
    main()
