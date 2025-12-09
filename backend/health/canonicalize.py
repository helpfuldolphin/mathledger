"""
Canonicalization tooling for ``global_health.json`` surfaces.

Run as a module::

    python -m backend.health.canonicalize --input global_health.json --check

The command validates the payload against the schema and enforces RFC-8785
canonical JSON output (sorted keys, ASCII, compact separators).

CI Integration
--------------
Run the canonicalizer as part of your pipeline:

    python -m backend.health.canonicalize --input global_health.json --check

Exit codes:
    0 -> payload already canonical and schema-valid (gate passes)
    1 -> non-canonical bytes or schema violation (gate fails)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .global_schema import SchemaValidationError, validate_global_health


def canonicalize_global_health(payload: Mapping[str, Any]) -> str:
    """
    Return the canonical RFC-8785 JSON string for ``global_health`` payloads.
    """
    normalized = validate_global_health(payload)
    return json.dumps(
        normalized,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    )


def _load_payload(path: Path) -> Mapping[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Failed to read {path}: {exc}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON ({path}): {exc}") from exc
    if not isinstance(data, Mapping):
        raise RuntimeError(f"JSON root must be an object, got {type(data)!r}")
    return data


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Canonicalize global_health.json payloads"
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to the JSON payload to validate/canonicalize",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional destination path. Defaults to --input when omitted.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only verify the file is canonical (do not write output).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    input_path: Path = args.input
    output_path: Path = args.output or input_path

    try:
        payload = _load_payload(input_path)
        canonical = canonicalize_global_health(payload)
    except (RuntimeError, SchemaValidationError) as exc:
        print(f"[global-health] {exc}", file=sys.stderr)
        return 1

    if args.check:
        existing = input_path.read_text(encoding="utf-8")
        if existing.rstrip("\r\n") != canonical:
            print(
                "[global-health] Input is not canonical; run without --check to rewrite.",
                file=sys.stderr,
            )
            return 1
        return 0

    output_path.write_text(canonical, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
