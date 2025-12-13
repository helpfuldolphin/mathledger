"""Monotone Ledger Guard module.

Provides monotonicity checking for ledger operations, CI-friendly summaries,
and governance signal adapters.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

LEDGER_MONOTONE_SCHEMA_VERSION = "1.0.0"


@dataclass
class MonotoneViolation:
    """Record of a monotonicity violation."""

    violation_type: str
    block_id: str
    expected: Any
    actual: Any
    message: str


@dataclass
class MonotoneCheckResult:
    """Result of monotonicity check."""

    valid: bool = True
    violations: List[MonotoneViolation] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    blocks_checked: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: str = "OK"
    schema_version: str = LEDGER_MONOTONE_SCHEMA_VERSION


def _result_status(violations: Iterable[Any], warnings: Iterable[Any]) -> str:
    violation_count = len(list(violations))
    warning_count = len(list(warnings))
    if violation_count > 0:
        return "BLOCK"
    if warning_count > 0:
        return "WARN"
    return "OK"


def check_monotone_ledger(
    blocks: Sequence[Dict[str, Any]],
    strict: bool = True,
) -> MonotoneCheckResult:
    """Check that ledger blocks maintain monotonicity.

    Verifies:
    - Block heights are strictly increasing
    - Each block's prev_hash matches the previous block's hash
    - Timestamps are non-decreasing
    """
    violations: List[MonotoneViolation] = []
    warnings: List[str] = []

    if len(blocks) < 2:
        return MonotoneCheckResult(
            valid=True,
            violations=[],
            warnings=[],
            blocks_checked=len(blocks),
            metadata={"strict": strict},
            status="OK",
        )

    for i in range(1, len(blocks)):
        prev_block = blocks[i - 1]
        curr_block = blocks[i]

        prev_height = prev_block.get("height", prev_block.get("block_id", i - 1))
        curr_height = curr_block.get("height", curr_block.get("block_id", i))

        if isinstance(prev_height, int) and isinstance(curr_height, int):
            if curr_height <= prev_height:
                violations.append(
                    MonotoneViolation(
                        violation_type="height",
                        block_id=str(curr_block.get("block_id", i)),
                        expected=f"> {prev_height}",
                        actual=curr_height,
                        message=(
                            f"Block height {curr_height} not greater than previous {prev_height}"
                        ),
                    )
                )

        prev_hash = prev_block.get("hash") or prev_block.get("merkle_root")
        curr_prev_hash = curr_block.get("prev_hash")

        if prev_hash and curr_prev_hash:
            if prev_hash != curr_prev_hash:
                violations.append(
                    MonotoneViolation(
                        violation_type="hash_chain",
                        block_id=str(curr_block.get("block_id", i)),
                        expected=prev_hash,
                        actual=curr_prev_hash,
                        message="Previous hash does not match",
                    )
                )
        else:
            warnings.append(
                f"Insufficient hash data for block {curr_block.get('block_id', i)}"
            )

        prev_ts = prev_block.get("timestamp")
        curr_ts = curr_block.get("timestamp")

        if prev_ts is not None and curr_ts is not None:
            if curr_ts < prev_ts:
                violations.append(
                    MonotoneViolation(
                        violation_type="timestamp",
                        block_id=str(curr_block.get("block_id", i)),
                        expected=f">= {prev_ts}",
                        actual=curr_ts,
                        message="Timestamp decreased",
                    )
                )
        elif strict and (prev_ts is None or curr_ts is None):
            warnings.append(
                f"Missing timestamp for block {curr_block.get('block_id', i)}"
            )

    status = _result_status(violations, warnings)

    return MonotoneCheckResult(
        valid=status == "OK",
        violations=violations,
        warnings=warnings,
        blocks_checked=len(blocks),
        metadata={"strict": strict},
        status=status,
    )


def verify_chain_integrity(
    blocks: Sequence[Dict[str, Any]],
) -> Tuple[bool, List[str]]:
    """Verify overall chain integrity."""

    result = check_monotone_ledger(blocks)
    errors = [v.message for v in result.violations]
    return result.valid, errors


def summarize_monotone_ledger_for_global_health(
    result: Dict[str, Any],
) -> Dict[str, Any]:
    """Summarize monotone guard status for global health dashboards."""

    violations = result.get("violations", [])
    warnings = result.get("warnings", [])
    violation_count = len(violations)
    warning_count = len(warnings)

    if violation_count > 0:
        status = "BLOCK"
        ledger_monotone = False
        headline = f"Ledger monotonicity violations detected ({violation_count})"
    elif warning_count > 0:
        status = "WARN"
        ledger_monotone = True
        headline = "Ledger monotone with schema warnings"
    else:
        status = "OK"
        ledger_monotone = True
        headline = "Ledger monotone"

    return {
        "schema_version": result.get(
            "schema_version", LEDGER_MONOTONE_SCHEMA_VERSION
        ),
        "ledger_monotone": ledger_monotone,
        "violation_count": violation_count,
        "status": status,
        "headline": headline,
    }


def to_governance_signal_for_ledger(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Map ledger summary to CLAUDE-I GovernanceSignal format."""

    status = (summary.get("status") or "WARN").upper()
    severity = {
        "OK": "non_blocking",
        "WARN": "advisory",
        "BLOCK": "blocking",
    }.get(status, "advisory")
    code = {
        "OK": "LEDGER-MONO-OK",
        "WARN": "LEDGER-MONO-WARN",
        "BLOCK": "LEDGER-MONO-BLOCK",
    }.get(status, "LEDGER-MONO-WARN")

    return {
        "schema_version": summary.get(
            "schema_version", LEDGER_MONOTONE_SCHEMA_VERSION
        ),
        "signal": "ledger_monotone",
        "status": status,
        "severity": severity,
        "code": code,
        "headline": summary.get(
            "headline", "Ledger monotone tile unavailable"
        ),
        "metadata": {
            "ledger_monotone": summary.get("ledger_monotone"),
            "violation_count": summary.get("violation_count", 0),
        },
    }


def load_blocks_from_path(path: Path) -> List[Dict[str, Any]]:
    """Load block headers from a JSONL/JSON artifact."""

    if not path.exists():
        raise FileNotFoundError(f"Blocks file not found: {path}")

    text = path.read_text(encoding="utf-8").strip()
    entries: List[Any] = []

    if not text:
        return []

    if text.startswith("["):
        payload = json.loads(text)
        if isinstance(payload, list):
            entries = payload
        else:
            entries = [payload]
    else:
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            entries.append(json.loads(stripped))

    blocks: List[Dict[str, Any]] = []
    for entry in entries:
        if isinstance(entry, Mapping) and isinstance(entry.get("header"), Mapping):
            blocks.append(dict(entry["header"]))
        elif isinstance(entry, Mapping):
            blocks.append(dict(entry))
        else:
            raise ValueError("Each entry must be a JSON object or contain a header")
    return blocks


def summarize_blocks_file(path: Path) -> Dict[str, Any]:
    """Convenience helper for CLI/script usage."""

    blocks = load_blocks_from_path(path)
    result = asdict(check_monotone_ledger(blocks))
    return result


def summarize_and_write_tile(
    blocks_path: Path,
    tile_output: Optional[Path] = None,
    signal_output: Optional[Path] = None,
) -> Dict[str, Any]:
    """Generate summary and optionally persist tile + governance signal."""

    result = summarize_blocks_file(blocks_path)
    summary = summarize_monotone_ledger_for_global_health(result)

    if tile_output:
        tile_output.parent.mkdir(parents=True, exist_ok=True)
        tile_output.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    if signal_output:
        signal_output.parent.mkdir(parents=True, exist_ok=True)
        signal_payload = to_governance_signal_for_ledger(summary)
        signal_output.write_text(
            json.dumps(signal_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate ledger monotonicity and emit CI-friendly summaries."
    )
    parser.add_argument(
        "--blocks",
        required=True,
        type=Path,
        help="Path to blocks.jsonl (headers or block entries with 'header').",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON summary only (no human-readable output).",
    )
    parser.add_argument(
        "--tile-output",
        type=Path,
        default=None,
        help="Optional destination for artifacts/tiles/ledger_monotone.json",
    )
    parser.add_argument(
        "--signal-output",
        type=Path,
        default=None,
        help="Optional destination for governance signal JSON.",
    )
    args = parser.parse_args(argv)

    summary = summarize_and_write_tile(
        args.blocks,
        tile_output=args.tile_output,
        signal_output=args.signal_output,
    )

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"[{summary['status']}] {summary['headline']}")
        print(f"  ledger_monotone: {summary['ledger_monotone']}")
        print(f"  violation_count: {summary['violation_count']}")
        if args.tile_output:
            print(f"  tile written to: {args.tile_output}")
        if args.signal_output:
            print(f"  signal written to: {args.signal_output}")

    return 0 if summary["status"] == "OK" else 1


__all__ = [
    "MonotoneViolation",
    "MonotoneCheckResult",
    "check_monotone_ledger",
    "verify_chain_integrity",
    "summarize_monotone_ledger_for_global_health",
    "to_governance_signal_for_ledger",
    "load_blocks_from_path",
    "summarize_blocks_file",
    "summarize_and_write_tile",
    "main",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
