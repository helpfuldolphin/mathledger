#!/usr/bin/env python3
"""
Generate ledger monotone health tile.

Usage:
    uv run python scripts/generate_ledger_monotone_tile.py \
        --blocks artifacts/ledger/blocks.jsonl \
        --tile-output artifacts/tiles/ledger_monotone.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from backend.ledger.monotone_guard import summarize_and_write_tile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Emit ledger monotone global-health tile and governance signal."
    )
    parser.add_argument(
        "--blocks",
        required=True,
        type=Path,
        help="Path to blocks JSON/JSONL artifact.",
    )
    parser.add_argument(
        "--tile-output",
        type=Path,
        default=Path("artifacts/tiles/ledger_monotone.json"),
        help="Destination for ledger monotone tile JSON.",
    )
    parser.add_argument(
        "--signal-output",
        type=Path,
        default=Path("artifacts/tiles/ledger_monotone_signal.json"),
        help="Optional destination for governance signal JSON.",
    )
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Print the tile JSON to stdout in addition to writing the file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = summarize_and_write_tile(
        args.blocks, tile_output=args.tile_output, signal_output=args.signal_output
    )
    if args.print_summary:
        print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
