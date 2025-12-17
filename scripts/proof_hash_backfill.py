#!/usr/bin/env python3
"""
Canonical Proof Hash Backfill (read-only audit + SQL generator).

Staging migration recipe:
    # 1. Dry-run + governance summary on a filtered subset
    python scripts/proof_hash_backfill.py \
        --mode report \
        --where "proofs.id < 5000" \
        --batch-size 2000 \
        --output proofs_backfill.jsonl \
        --governance-report proofs_backfill_report.json

    # 2. Generate SQL updates for changed proofs (still read-only)
    python scripts/proof_hash_backfill.py \
        --mode sql \
        --where "proofs.id < 5000" \
        --batch-size 2000 \
        --sql-output proofs_hash_update.sql \
        --governance-report proofs_backfill_report.json

    # 3. Produce human diffs for every changed proof
    python scripts/proof_canonical_diff.py \
        --from-jsonl proofs_backfill.jsonl \
        > proofs_backfill.diff

None of these steps mutate the database; SQL output is a deterministic file for
governance review / staged application.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterator, List, Mapping, Optional, TextIO

import psycopg
from psycopg.rows import dict_row

from ledger.ingest import _canonical_json
from normalization.proof import canonicalize_module_name, canonicalize_proof_text
from substrate.crypto.hashing import DOMAIN_ROOT, sha256_hex


def _require_database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "[FATAL] DATABASE_URL environment variable is not set. "
            "Export it explicitly before running proof hash backfill."
        )
    return url


def compute_canonical_hash(row: Mapping[str, object]) -> str:
    canonical_proof = canonicalize_proof_text(row.get("proof_text"))
    canonical_module = canonicalize_module_name(row.get("module_name"))
    payload = {
        "statement_hash": row.get("statement_hash", ""),
        "prover": row.get("prover", "") or "",
        "status": row.get("status", "") or "",
        "proof_text": canonical_proof,
        "module": canonical_module,
        "derivation_rule": row.get("derivation_rule", "") or "",
    }
    payload_json = _canonical_json(payload)
    return sha256_hex(payload_json.encode("utf-8"), domain=DOMAIN_ROOT)


def summarize_row(row: Mapping[str, object]) -> Mapping[str, object]:
    new_hash = compute_canonical_hash(row)
    old_hash = row.get("proof_hash")
    proof_id = int(row.get("id"))
    return {
        "proof_id": proof_id,
        "old_hash": old_hash,
        "new_hash": new_hash,
        "changed": bool(old_hash != new_hash),
    }


class GovernanceReportBuilder:
    """Aggregates backfill results for downstream dashboards."""

    def __init__(self, sample_limit: int = 10) -> None:
        self._schema_version = "1.0.0"
        self._total = 0
        self._changed = 0
        self._sample_limit = sample_limit
        self._sample: List[Mapping[str, object]] = []

    def update(self, summary: Mapping[str, object]) -> None:
        self._total += 1
        if summary.get("changed"):
            self._changed += 1
            if len(self._sample) < self._sample_limit:
                self._sample.append(
                    {
                        "proof_id": summary.get("proof_id"),
                        "old_hash": summary.get("old_hash"),
                        "new_hash": summary.get("new_hash"),
                    }
                )

    def build(self) -> Mapping[str, object]:
        return {
            "schema_version": self._schema_version,
            "total_proofs": self._total,
            "changed_count": self._changed,
            "unchanged_count": self._total - self._changed,
            "sample_changed": self._sample,
        }


def stream_proofs(
    conn: psycopg.Connection,
    *,
    batch_size: int,
    where_clause: Optional[str],
) -> Iterator[Mapping[str, object]]:
    sql = """
        SELECT
            p.id,
            p.proof_hash,
            p.proof_text,
            p.module_name,
            p.prover,
            p.status,
            p.derivation_rule,
            s.hash AS statement_hash
        FROM proofs p
        JOIN statements s ON s.id = p.statement_id
    """
    if where_clause:
        sql += f" WHERE {where_clause} "
    sql += " ORDER BY p.id"

    with conn.cursor(name="proof_hash_backfill", row_factory=dict_row) as cur:
        cur.itersize = batch_size
        cur.execute(sql)
        for row in cur:
            yield row


def emit_report(
    rows: Iterator[Mapping[str, object]],
    *,
    output: TextIO,
    governance: GovernanceReportBuilder,
) -> None:
    for row in rows:
        summary = summarize_row(row)
        governance.update(summary)
        output.write(json.dumps(summary) + "\n")
        output.flush()


def emit_sql(changed_rows: List[Mapping[str, object]], path: Path) -> None:
    lines = [
        "-- Canonical proof hash updates (generated by proof_hash_backfill.py)",
        "BEGIN;",
    ]
    for entry in sorted(changed_rows, key=lambda r: r["proof_id"]):
        proof_id = entry["proof_id"]
        old_hash = entry["old_hash"] or ""
        new_hash = entry["new_hash"]
        lines.append(
            f"UPDATE proofs "
            f"SET proof_hash = '{new_hash}' "
            f"WHERE id = {proof_id} AND proof_hash = '{old_hash}';"
        )
    lines.append("COMMIT;")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Canonical proof hash audit + SQL generator.")
    parser.add_argument(
        "--mode",
        choices=("report", "sql"),
        default="report",
        help="report=JSONL delta stream, sql=write SQL update script (no DB mutation).",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path for JSONL report (defaults to stdout in report mode).",
    )
    parser.add_argument(
        "--sql-output",
        type=str,
        help="Path for SQL output (required in --mode sql).",
    )
    parser.add_argument(
        "--governance-report",
        type=str,
        help="Optional path for governance summary JSON.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=10,
        help="Sample size for governance report (default: 10).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Number of proofs fetched per DB round-trip (default: 500).",
    )
    parser.add_argument(
        "--where",
        type=str,
        help="Optional SQL WHERE clause to scope proofs (e.g. \"proofs.id < 5000\").",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    db_url = _require_database_url()
    governance = GovernanceReportBuilder(sample_limit=args.sample_limit)
    with psycopg.connect(db_url) as conn:
        rows = stream_proofs(conn, batch_size=args.batch_size, where_clause=args.where)
        if args.mode == "report":
            if args.output:
                with open(args.output, "w", encoding="utf-8") as handle:
                    emit_report(rows, output=handle, governance=governance)
            else:
                emit_report(rows, output=sys.stdout, governance=governance)
        else:
            if not args.sql_output:
                raise RuntimeError("--sql-output is required when --mode sql")
            changed_rows: List[Mapping[str, object]] = []
            for row in rows:
                summary = summarize_row(row)
                governance.update(summary)
                if summary["changed"]:
                    changed_rows.append(summary)
            emit_sql(changed_rows, Path(args.sql_output))

    if args.governance_report:
        Path(args.governance_report).write_text(
            json.dumps(governance.build(), indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
