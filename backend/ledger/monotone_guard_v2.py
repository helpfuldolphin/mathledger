from __future__ import annotations

from typing import Any, Dict, List

from .block_schema_v2 import BlockHeader, validate_block_header

_SCHEMA_VERSION = "1.0.0"


def _materialize(header: Dict[str, Any]) -> BlockHeader:
    return BlockHeader(
        height=header["height"],
        prev_hash=header["prev_hash"],
        root_hash=header["root_hash"],
        timestamp=header["timestamp"],
    )


def check_monotone_ledger(headers: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(headers, list):
        raise TypeError("headers must be a list of block header dicts")

    violations: List[str] = []
    previous_height = None
    previous_root_hash = None

    for idx, entry in enumerate(headers):
        is_valid, issues = validate_block_header(entry)
        if not is_valid:
            for issue in issues:
                violations.append(f"header[{idx}]: {issue}")
            previous_height = None
            previous_root_hash = None
            continue

        header = _materialize(entry)

        if previous_height is not None and header.height <= previous_height:
            violations.append(
                f"header[{idx}]: height {header.height} is not greater than "
                f"previous height {previous_height}"
            )

        if (
            previous_root_hash is not None
            and header.prev_hash != previous_root_hash
        ):
            violations.append(
                f"header[{idx}]: prev_hash {header.prev_hash} does not match "
                f"previous root_hash {previous_root_hash}"
            )

        previous_height = header.height
        previous_root_hash = header.root_hash

    return {
        "schema_version": _SCHEMA_VERSION,
        "is_monotone": len(violations) == 0,
        "violations": violations,
    }