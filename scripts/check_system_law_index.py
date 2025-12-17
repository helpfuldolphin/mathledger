from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

from tools import generate_system_law_index as indexer


SCHEMA_VERSION = 1
MODE = "SHADOW"

UP_TO_DATE_MESSAGE = "system-law-index: up to date (no action needed)"
OUT_OF_DATE_MESSAGE = (
    "system-law-index: out of date; run `python tools/generate_system_law_index.py`"
)


def _build_payload(*, up_to_date: bool, reason_code: str) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": MODE,
        "up_to_date": up_to_date,
        "reason_code": reason_code,
        "remediation": "no action needed"
        if up_to_date
        else "run `python tools/generate_system_law_index.py`",
    }


def _serialize_payload(payload: dict[str, object]) -> str:
    return json.dumps(payload, sort_keys=True)


def run_check(*, json_mode: bool = False, output_path: str | None = None) -> int:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
        io.StringIO()
    ):
        result = indexer.main(["--check"])
    up_to_date = result == 0
    payload = _build_payload(
        up_to_date=up_to_date, reason_code="UP_TO_DATE" if up_to_date else "OUT_OF_DATE"
    )
    json_payload = _serialize_payload(payload)

    if output_path:
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(json_payload, encoding="utf-8")
        except OSError as exc:
            payload["reason_code"] = "WRITE_FAILED"
            json_payload = _serialize_payload(payload)
            print(
                f"system-law-index: failed to write JSON output: {exc}",
                file=sys.stderr,
            )

    if json_mode:
        print(json_payload, end="")
        return 0

    if up_to_date:
        print(UP_TO_DATE_MESSAGE)
    else:
        print(OUT_OF_DATE_MESSAGE)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check that system law index is current.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output machine-consumable JSON status to stdout.",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Write machine-consumable JSON status to a file.",
    )
    args = parser.parse_args(argv)
    return run_check(json_mode=args.json, output_path=args.output)


if __name__ == "__main__":
    raise SystemExit(main())
