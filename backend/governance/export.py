#!/usr/bin/env python3
"""
Governance Export: Generate governance_chain.json and declared_roots.json
==========================================================================

Extracts provenance data from:
1. Attestation history → governance_chain.json
2. Blocks table → declared_roots.json
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# Allow running from command line
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def export_governance_chain(
    attestation_dir: Path,
    output_path: Path
) -> int:
    """
    Export governance chain from attestation history.

    Scans attestation_history/*.json and builds a chain of signatures.

    Returns:
        Number of entries exported
    """
    if not attestation_dir.exists():
        print(f"⚠️  Attestation directory not found: {attestation_dir}")
        return 0

    attestation_files = sorted(attestation_dir.glob("attestation_*.json"))

    if not attestation_files:
        print(f"⚠️  No attestation files found in {attestation_dir}")
        return 0

    entries = []

    for attest_file in attestation_files:
        with open(attest_file) as f:
            data = json.load(f)

        entry = {
            "signature": data.get("signature", ""),
            "prev_signature": data.get("prev_signature", ""),
            "timestamp": data.get("timestamp", ""),
            "status": data.get("status", "UNKNOWN"),
            "determinism_score": data.get("determinism_score", 0),
            "version": data.get("version", "1.0.0"),
            "replay_success": data.get("replay_success", False),
        }

        entries.append(entry)

    # Sort by timestamp
    entries.sort(key=lambda e: e["timestamp"])

    governance_chain = {
        "version": "1.0.0",
        "exported_at": datetime.utcnow().isoformat() + "Z",
        "entry_count": len(entries),
        "entries": entries
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(governance_chain, f, indent=2, sort_keys=True)

    print(f"✅ Exported governance chain: {len(entries)} entries → {output_path}")

    return len(entries)


def export_declared_roots_from_db(
    db_url: str,
    output_path: Path
) -> int:
    """
    Export declared roots from blocks table.

    Queries PostgreSQL for all blocks and exports:
    - block_number
    - root_hash (Merkle root)
    - prev_hash
    - statement_count
    - sealed_at

    Returns:
        Number of roots exported
    """
    try:
        import psycopg
    except ImportError:
        print("❌ psycopg not installed. Install with: pip install psycopg[binary]")
        return 0

    try:
        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT
                        block_number,
                        root_hash,
                        prev_hash,
                        jsonb_array_length(statements) as statement_count,
                        created_at
                    FROM blocks
                    ORDER BY block_number ASC
                """)

                rows = cur.fetchall()

                if not rows:
                    print("⚠️  No blocks found in database")
                    return 0

                roots = []

                for row in rows:
                    block_number, root_hash, prev_hash, stmt_count, created_at = row

                    roots.append({
                        "block_number": block_number,
                        "root_hash": root_hash,
                        "prev_hash": prev_hash or "",
                        "statement_count": stmt_count or 0,
                        "sealed_at": created_at.isoformat() if created_at else ""
                    })

                declared_roots = {
                    "version": "1.0.0",
                    "exported_at": datetime.utcnow().isoformat() + "Z",
                    "block_count": len(roots),
                    "roots": roots
                }

                output_path.parent.mkdir(parents=True, exist_ok=True)

                with open(output_path, "w") as f:
                    json.dump(declared_roots, f, indent=2, sort_keys=True)

                print(f"✅ Exported declared roots: {len(roots)} blocks → {output_path}")

                return len(roots)

    except Exception as e:
        print(f"❌ Database export failed: {e}")
        return 0


def export_declared_roots_from_json(
    blocks_json: Path,
    output_path: Path
) -> int:
    """
    Export declared roots from a JSON export of blocks.

    Alternative to database export when DB is not available.
    """
    if not blocks_json.exists():
        print(f"⚠️  Blocks JSON not found: {blocks_json}")
        return 0

    with open(blocks_json) as f:
        data = json.load(f)

    blocks = data.get("blocks", [])

    if not blocks:
        print("⚠️  No blocks found in JSON")
        return 0

    roots = []

    for block in blocks:
        header = block.get("header", {})
        roots.append({
            "block_number": header.get("block_number", 0),
            "root_hash": header.get("merkle_root", ""),
            "prev_hash": header.get("prev_hash", ""),
            "statement_count": header.get("proof_count", 0),
            "sealed_at": header.get("timestamp", "")
        })

    declared_roots = {
        "version": "1.0.0",
        "exported_at": datetime.utcnow().isoformat() + "Z",
        "block_count": len(roots),
        "roots": roots
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(declared_roots, f, indent=2, sort_keys=True)

    print(f"✅ Exported declared roots: {len(roots)} blocks → {output_path}")

    return len(roots)


def main():
    """CLI entry point for governance export."""
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description="Export governance chain and declared roots"
    )
    parser.add_argument(
        "--attestation-dir",
        type=Path,
        default=Path("artifacts/repro/attestation_history"),
        help="Directory containing attestation files"
    )
    parser.add_argument(
        "--governance-output",
        type=Path,
        default=Path("artifacts/governance/governance_chain.json"),
        help="Output path for governance chain"
    )
    parser.add_argument(
        "--roots-output",
        type=Path,
        default=Path("artifacts/governance/declared_roots.json"),
        help="Output path for declared roots"
    )
    parser.add_argument(
        "--db-url",
        type=str,
        default=os.getenv("DATABASE_URL", ""),
        help="Database URL for blocks export"
    )
    parser.add_argument(
        "--blocks-json",
        type=Path,
        help="Alternative: Path to blocks JSON export"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("GOVERNANCE EXPORT — Extracting Provenance Data")
    print("=" * 60)

    # Export governance chain
    gov_count = export_governance_chain(
        args.attestation_dir,
        args.governance_output
    )

    # Export declared roots
    if args.blocks_json and args.blocks_json.exists():
        roots_count = export_declared_roots_from_json(
            args.blocks_json,
            args.roots_output
        )
    elif args.db_url:
        roots_count = export_declared_roots_from_db(
            args.db_url,
            args.roots_output
        )
    else:
        print("⚠️  No database URL or blocks JSON provided, skipping roots export")
        roots_count = 0

    print("=" * 60)
    print(f"EXPORT COMPLETE: {gov_count} governance entries, {roots_count} roots")
    print("=" * 60)


if __name__ == "__main__":
    main()
