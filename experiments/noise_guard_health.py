#!/usr/bin/env python3
"""
Noise Guard compact tile generator.

Context:
    The verifier noise guard emits metrics snapshots (JSON) that quantify
    timeout noise, tier instability, and epsilon_total. This helper script
    converts that snapshot into a deterministic health tile suitable for the
    global-health console (stored under ``artifacts/tiles/noise_guard.json``).

    Canonical CI usage::

        uv run python experiments/noise_guard_health.py \
            --metrics metrics/verifier_noise_window.json \
            --output artifacts/tiles/noise_guard.json

        # Later in the pipeline:
        import json
        payload = json.loads(Path("artifacts/tiles/noise_guard.json").read_text())
        global_health["noise_guard"] = payload["noise_guard"]

    Status semantics (neutral language):
        - ``OK``: epsilon_total below attention bands and guard idle.
        - ``ATTENTION``: epsilon_total >= 0.10 or bucket instability present.
        - ``BLOCK``: timeout CUSUM alarm or epsilon_total >= 0.25.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

TILE_SCHEMA_VERSION = "noise-guard-tile-1.0"


def summarize_snapshot(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    epsilon = float(snapshot.get("epsilon_total") or 0.0)
    timeout_noisy = bool(snapshot.get("timeout_noisy"))
    unstable_buckets = snapshot.get("unstable_buckets") or []

    status = "OK"
    notes: List[str] = []

    if timeout_noisy or epsilon >= 0.25:
        status = "BLOCK"
        if timeout_noisy:
            notes.append("timeout-noisy")
        if epsilon >= 0.25:
            notes.append("epsilon>=0.25")
    elif epsilon >= 0.10 or unstable_buckets:
        status = "ATTENTION"
        if epsilon >= 0.10:
            notes.append("epsilon>=0.10")
        if unstable_buckets:
            notes.append("bucket-instability")

    if not notes:
        notes.append("stable")

    return {
        "status": status,
        "epsilon_total": round(epsilon, 6),
        "notes": notes,
        "timeout_noisy": timeout_noisy,
        "unstable_bucket_count": len(unstable_buckets),
        "delta_h_bound": snapshot.get("delta_h_bound"),
        "window_id": snapshot.get("window_id"),
    }


def build_tile(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    summary = summarize_snapshot(snapshot)
    return {
        "tile_id": "noise_guard",
        "schema_version": TILE_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "channels": snapshot.get("channels", {}),
        "timeout_cusum": snapshot.get("timeout_cusum"),
        "window_size": snapshot.get("window_size"),
        "raw_snapshot_path": snapshot.get("__source_path"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Emit Noise Guard global-health tile.")
    parser.add_argument(
        "--metrics",
        default="metrics/verifier_noise_window.json",
        help="Path to verifier noise metrics snapshot (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default="artifacts/tiles/noise_guard.json",
        help="Destination for the global health payload (default: %(default)s)",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        help="Print the payload to stdout instead of writing a file.",
    )
    return parser.parse_args()


def load_snapshot(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Noise guard metrics not found: {path}")
    snapshot = json.loads(path.read_text())
    snapshot["__source_path"] = str(path)
    return snapshot


def main() -> None:
    args = parse_args()
    snapshot_path = Path(args.metrics)
    snapshot = load_snapshot(snapshot_path)
    tile = build_tile(snapshot)
    payload = {"noise_guard": tile}

    if args.print:
        json.dump(payload, fp=sys.stdout, indent=2, sort_keys=True)
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"Wrote noise guard tile to {output_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - CLI guard
        print(f"[noise_guard_health] ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
