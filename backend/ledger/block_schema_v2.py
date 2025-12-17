from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Tuple

_HEX_LENGTH = 64


@dataclass(frozen=True)
class BlockHeader:
    """Minimal representation of a ledger block header."""

    height: int
    prev_hash: str
    root_hash: str
    timestamp: str


def _is_hex_hash(value: str) -> bool:
    if len(value) != _HEX_LENGTH:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _is_iso8601(value: str) -> bool:
    if not value:
        return False
    probe = value
    if probe.endswith("Z"):
        probe = probe[:-1] + "+00:00"
    try:
        datetime.fromisoformat(probe)
    except ValueError:
        return False
    return True


def _describe_type(value: Any) -> str:
    return type(value).__name__


def validate_block_header(header: Dict[str, Any]) -> Tuple[bool, List[str]]:
    issues: List[str] = []
    if not isinstance(header, dict):
        return False, ["Header must be a dict"]

    def require_field(name: str) -> Any:
        if name not in header:
            issues.append(f"Missing field '{name}'")
            return None
        return header[name]

    height = require_field("height")
    if height is not None:
        if isinstance(height, bool) or not isinstance(height, int):
            issues.append(
                f"Field 'height' must be an int, got {_describe_type(height)}"
            )
        elif height < 0:
            issues.append("Field 'height' must be >= 0")

    prev_hash = require_field("prev_hash")
    if prev_hash is not None:
        if not isinstance(prev_hash, str):
            issues.append(
                f"Field 'prev_hash' must be a str, got {_describe_type(prev_hash)}"
            )
        elif not _is_hex_hash(prev_hash):
            issues.append("Field 'prev_hash' must be 64 hex characters")

    root_hash = require_field("root_hash")
    if root_hash is not None:
        if not isinstance(root_hash, str):
            issues.append(
                f"Field 'root_hash' must be a str, got {_describe_type(root_hash)}"
            )
        elif not _is_hex_hash(root_hash):
            issues.append("Field 'root_hash' must be 64 hex characters")

    timestamp = require_field("timestamp")
    if timestamp is not None:
        if not isinstance(timestamp, str):
            issues.append(
                f"Field 'timestamp' must be a str, got {_describe_type(timestamp)}"
            )
        elif not _is_iso8601(timestamp):
            issues.append("Field 'timestamp' must be ISO 8601")

    return len(issues) == 0, issues