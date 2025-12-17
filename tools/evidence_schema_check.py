#!/usr/bin/env python3
"""
Lightweight helper for validating evidence payloads against JSON Schemas.

Example (First Light synthetic raw JSONL):
    uv run python tools/evidence_schema_check.py `
        results/first_light/evidence_pack_first_light/p3_synthetic/synthetic_raw.jsonl `
        schemas/evidence/first_light_synthetic_raw.schema.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from jsonschema import Draft7Validator
from jsonschema.exceptions import ValidationError

logger = logging.getLogger(__name__)


def _load_json_lines(path: Path) -> list[Any]:
    records: list[Any] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as err:
                raise json.JSONDecodeError(
                    f"Invalid JSON on line {line_number}: {err.msg}",
                    err.doc,
                    err.pos,
                ) from err
    return records


def _load_json(path: Path) -> Any:
    if path.suffix.lower() == ".jsonl":
        return _load_json_lines(path)

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_file_against_schema(file_path: str | Path, schema_path: str | Path) -> bool:
    """
    Validate a JSON payload file against a schema.

    Returns True when the payload conforms to the schema, False otherwise.
    """
    payload_path = Path(file_path)
    schema_path = Path(schema_path)

    payload = _load_json(payload_path)
    schema = _load_json(schema_path)

    validator = Draft7Validator(schema)
    errors = sorted(validator.iter_errors(payload), key=lambda err: err.absolute_path)

    if errors:
        for error in errors:
            location = ".".join(str(part) for part in error.absolute_path) or "<root>"
            logger.error("%s %s -> %s", payload_path, location, error.message)
        return False

    return True


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate an evidence payload against a schema.")
    parser.add_argument("payload", type=Path, help="Path to the JSON payload to validate.")
    parser.add_argument("schema", type=Path, help="Path to the JSON Schema file.")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only emit validation errors.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.ERROR if args.quiet else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    try:
        is_valid = validate_file_against_schema(args.payload, args.schema)
    except FileNotFoundError as err:
        logger.error("File not found: %s", err)
        return 2
    except json.JSONDecodeError as err:
        logger.error("Invalid JSON: %s", err)
        return 2
    except ValidationError as err:
        logger.error("Schema validation error: %s", err)
        return 2

    if is_valid:
        logger.info("Validation succeeded for %s", args.payload)
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
