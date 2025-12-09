#!/usr/bin/env python3
"""
Noise Guard global-health tile generator.

Consumes the latest verifier-noise snapshot and emits a deterministically
serialized JSON tile that plugs into the `global_health/*.json` payload next
to budget/telemetry/PRNG summaries. Status semantics:
    - OK: epsilon below attention band, no guard alarms.
    - ATTENTION: elevated epsilon or bucket instability – monitor but not block.
    - BLOCK: guard is actively suppressing updates (timeout CUSUM, epsilon cap).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from derivation.noise_guard import summarize_noise_guard_for_global_health

SCHEMA_VERSION = "noise-guard-tile-1.0"


def load_snapshot(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Noise guard metrics not found: {path}")
    return json.loads(path.read_text())


def build_tile(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    summary = summarize_noise_guard_for_global_health(snapshot)
    return {
        "tile_id": "noise_guard",
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "window": {
            "window_id": snapshot.get("window_id"),
            "window_size": snapshot.get("window_size"),
            "channels": snapshot.get("channels"),
            "timeout_cusum": snapshot.get("timeout_cusum"),
        },
        "raw_snapshot_path": snapshot.get("__source_path"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Emit a Noise Guard health tile.")
    parser.add_argument(
        "--metrics",
        default="metrics/verifier_noise_window.json",
        help="Path to noise guard metrics JSON (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default="artifacts/global_health/noise_guard.json",
        help="Where to write the health tile JSON (default: %(default)s)",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        help="Print the tile to stdout instead of writing to disk.",
    )
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    snapshot = load_snapshot(metrics_path)
    snapshot["__source_path"] = str(metrics_path)
    tile = build_tile(snapshot)

    if args.print:
        json.dump(tile, fp=sys.stdout, indent=2, sort_keys=True)
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(tile, indent=2, sort_keys=True))
    print(f"Wrote noise guard tile to {output_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - CLI guard
        print(f"[noise_guard_health] ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
