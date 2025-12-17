#!/usr/bin/env python3
"""Ledger guard summary builder (SHADOW mode only)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from backend.ledger.monotone_guard_v2 import check_monotone_ledger
from backend.health.ledger_guard_tile import build_ledger_guard_tile


def _stub_chain() -> List[Dict[str, Any]]:
    return [
        {
            "height": 0,
            "prev_hash": "0" * 64,
            "root_hash": "1" * 64,
            "timestamp": "2024-01-01T00:00:00Z",
        },
        {
            "height": 1,
            "prev_hash": "1" * 64,
            "root_hash": "2" * 64,
            "timestamp": "2024-01-01T00:01:00Z",
        },
    ]


def load_chain(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Ledger chain JSON must be a list of block headers")
    return payload


def build_summary_from_chain(headers: List[Dict[str, Any]]) -> Dict[str, Any]:
    check_result = check_monotone_ledger(headers)
    summary = build_ledger_guard_tile(check_result)
    summary["violation_counts"] = summary.get("violation_count", 0)
    summary.setdefault("schema_version", "1.0.0")
    summary["generated_at"] = datetime.now(timezone.utc).isoformat()
    return summary


def write_summary(summary: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run ledger monotone guard and emit summary JSON (SHADOW mode)"
    )
    parser.add_argument(
        "--ledger-chain",
        type=str,
        help="Path to JSON file containing a list of block headers.",
    )
    parser.add_argument(
        "--use-stub",
        action="store_true",
        help="Use a synthetic stub chain instead of loading from --ledger-chain.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ledger_guard_summary.json",
        help="Output JSON path for the summary (default: ledger_guard_summary.json)",
    )

    args = parser.parse_args()

    if not args.use_stub and not args.ledger_chain:
        parser.error("Provide --ledger-chain or pass --use-stub for synthetic data")

    if args.use_stub:
        headers = _stub_chain()
    else:
        chain_path = Path(args.ledger_chain)
        if not chain_path.exists():
            print(f"ERROR: Ledger chain file does not exist: {chain_path}")
            return 1
        headers = load_chain(chain_path)

    summary = build_summary_from_chain(headers)
    write_summary(summary, Path(args.output))
    print(f"Ledger guard summary written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())