"""Shadow-mode helper to snapshot proof log hashes for First Light evidence.

Example:
    python -m scripts.first_light_proof_hash_snapshot --proof-log proofs.jsonl --output snapshot.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Dict

from scripts.proof_hash_tools_v2 import backfill_hashes, compute_proof_hash


SNAPSHOT_SCHEMA_VERSION = "1.0.0"
CANONICAL_HASH_ALGORITHM = "sha256"
CANONICALIZATION_VERSION = "proof-log-v1"


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)


def _sorted_proof_hashes(jsonl_path: Path) -> list[str]:
    hashes: list[str] = []
    for obj in _read_jsonl(jsonl_path):
        hashes.append(compute_proof_hash(obj))
    hashes.sort()
    return hashes


def build_snapshot(proof_log: str) -> Dict[str, object]:
    proof_log_path = Path(proof_log)
    if not proof_log_path.exists():
        raise FileNotFoundError(f"Proof log not found: {proof_log}")

    with tempfile.NamedTemporaryFile("w+", suffix=".jsonl", delete=False) as tmp:
        hashed_path = Path(tmp.name)

    try:
        backfill_hashes(str(proof_log_path), str(hashed_path))
        hashes = _sorted_proof_hashes(hashed_path)
        canonical_payload = "\n".join(hashes).encode("utf-8")
        canonical_hash = hashlib.sha256(canonical_payload).hexdigest()

        snapshot = {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "canonical_hash_algorithm": CANONICAL_HASH_ALGORITHM,
            "canonicalization_version": CANONICALIZATION_VERSION,
            "source": str(proof_log_path),
            "proof_hashes": hashes,
            "canonical_hash": canonical_hash,
            "entry_count": len(hashes),
        }
        return snapshot
    finally:
        if hashed_path.exists():
            os.remove(hashed_path)


def generate_snapshot(proof_log: str, output_path: str) -> Dict[str, object]:
    snapshot = build_snapshot(proof_log)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return snapshot


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute a proof log hash snapshot for First Light evidence."
    )
    parser.add_argument(
        "--proof-log",
        required=True,
        help="Path to the proof JSONL log.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Where to write the snapshot JSON.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    generate_snapshot(args.proof_log, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
