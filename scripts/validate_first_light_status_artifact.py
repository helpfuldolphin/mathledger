#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tools.ci.first_light_status_artifact_contract import (
    build_first_light_status_artifact_contract_report,
    validate_first_light_status_artifact_file,
)


def _default_fixture_path() -> Path:
    return Path(__file__).resolve().parents[1] / "testsfixtures" / "first_light_status_sample.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a First Light status JSON artifact against the fixture keyset "
            "and allowed telemetry taxonomy."
        )
    )
    parser.add_argument(
        "--status-json",
        type=Path,
        required=True,
        help="Path to first_light_status.json (produced artifact).",
    )
    parser.add_argument(
        "--fixture-json",
        type=Path,
        default=_default_fixture_path(),
        help="Path to the fixture JSON (default: testsfixtures/first_light_status_sample.json).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Max number of reason codes to emit at top-level as reason_codes_topN.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = validate_first_light_status_artifact_file(
        status_path=args.status_json,
        fixture_path=args.fixture_json,
    )
    report = build_first_light_status_artifact_contract_report(result, top_n=args.top_n)
    print(json.dumps(report, indent=2, sort_keys=False))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
