"""
Monotone guard utilities for ledger blocks.

Provides:
- BlockSchemaEnvelope invariant checks (R_t/U_t linkage, ordering, etc.)
- Ledger-level linear pass for block_number ordering and prev_hash linkage
- CLI entry for validating JSONL dumps of block headers
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from attestation.dual_root import compute_composite_root
from substrate.repro.determinism import deterministic_timestamp_from_content

from ledger.block_schema import (
    BLOCK_SCHEMA_CONTRACT,
    BlockSchemaContract,
    BlockSchemaEnvelope,
    validate_block_schema,
)


@dataclass(frozen=True)
class InvariantViolation:
    """Represents a monotone invariant failure."""

    invariant_id: str
    block_number: int
    status: str
    message: str


def check_monotone_invariants(
    blocks: Sequence[BlockSchemaEnvelope],
    *,
    contract: BlockSchemaContract = BLOCK_SCHEMA_CONTRACT,
) -> List[InvariantViolation]:
    """
    Evaluate fine-grained monotone invariants over a block sequence.

    Verified invariants:
    - INV-MONO-SCHEMA: Block header matches schema contract.
    - INV-MONO-001: Block numbers strictly increase by 1.
    - INV-MONO-002: prev_hash equals prior composite_root (or None for genesis).
    - INV-MONO-003: Statement list lexicographically sorted with no duplicates.
    - INV-MONO-004: Proof list sorted lexicographically by hash+statement.
    - INV-MONO-005: composite_root == SHA256(R_t || U_t).
    - INV-MONO-006: sealed_at derives from deterministic H_t seed.
    """
    violations: List[InvariantViolation] = []
    prev_header: Optional[Mapping[str, Any]] = None

    for envelope in blocks:
        header = envelope.header
        block_number = _resolve_block_number(header)

        validation = validate_block_schema(header, contract)
        if validation["status"] != "OK":
            message = _build_schema_violation_message(validation)
            violations.append(
                InvariantViolation(
                    invariant_id="INV-MONO-SCHEMA",
                    block_number=block_number if block_number is not None else -1,
                    status=validation["status"],
                    message=message,
                )
            )
            if validation["status"] == "BLOCK":
                continue

        composite_root = header["composite_root"]
        reasoning_root = header["reasoning_root"]
        ui_root = header["ui_root"]
        sealed_at_iso = header["sealed_at"]

        if prev_header is not None:
            expected_number = prev_header["block_number"] + 1
            if header["block_number"] != expected_number:
                violations.append(
                    InvariantViolation(
                        invariant_id="INV-MONO-001",
                        block_number=header["block_number"],
                        status="BLOCK",
                        message=f"Block height {header['block_number']} does not follow {expected_number - 1}",
                    )
                )

            expected_prev_hash = prev_header.get("composite_root")
            if header.get("prev_hash") != expected_prev_hash:
                violations.append(
                    InvariantViolation(
                        invariant_id="INV-MONO-002",
                        block_number=header["block_number"],
                        status="BLOCK",
                        message=f"prev_hash {header.get('prev_hash')} != prior composite_root {expected_prev_hash}",
                    )
                )
        elif header.get("prev_hash") not in (None, "", 0):
            violations.append(
                InvariantViolation(
                    invariant_id="INV-MONO-002",
                    block_number=header.get("block_number", -1),
                    status="BLOCK",
                    message="Genesis block must not reference prev_hash",
                )
            )

        if composite_root != compute_composite_root(reasoning_root, ui_root):
            violations.append(
                InvariantViolation(
                    invariant_id="INV-MONO-005",
                    block_number=header.get("block_number", -1),
                    status="BLOCK",
                    message="composite_root mismatch with R_t/U_t inputs",
                )
            )

        if not _statements_are_canonical(envelope.statements):
            violations.append(
                InvariantViolation(
                    invariant_id="INV-MONO-003",
                    block_number=header.get("block_number", -1),
                    status="BLOCK",
                    message="Statement hashes must be lexicographically sorted and unique",
                )
            )

        if not _proofs_are_canonical(envelope.canonical_proofs):
            violations.append(
                InvariantViolation(
                    invariant_id="INV-MONO-004",
                    block_number=header.get("block_number", -1),
                    status="BLOCK",
                    message="Proof ordering is not lexicographically canonical",
                )
            )

        expected_iso = _isoformat(deterministic_timestamp_from_content(composite_root))
        if sealed_at_iso != expected_iso:
            violations.append(
                InvariantViolation(
                    invariant_id="INV-MONO-006",
                    block_number=header.get("block_number", -1),
                    status="BLOCK",
                    message="sealed_at must be derived deterministically from composite_root",
                )
            )

        prev_header = header

    return violations


def check_monotone_ledger(
    blocks: Sequence[Mapping[str, Any]],
    *,
    contract: BlockSchemaContract = BLOCK_SCHEMA_CONTRACT,
) -> Dict[str, Any]:
    """
    Coarse ledger-level pass enforcing ordering and prev_hash linkage.

    Guarantees:
    - strictly increasing block_number
    - contiguous prev_hash linkage
    - no duplicate block_number
    """
    violations: List[str] = []
    warnings: List[str] = []
    seen_numbers: set[int] = set()
    prev_number: Optional[int] = None
    prev_composite_root: Optional[str] = None

    for idx, block in enumerate(blocks):
        block_number = _resolve_block_number(block)
        label = (
            f"block {block_number}"
            if block_number is not None
            else f"block@{idx}"
        )

        schema_report = validate_block_schema(block, contract)
        if schema_report["status"] == "BLOCK":
            missing = ", ".join(schema_report["missing_fields"]) or "unknown"
            violations.append(f"{label}: missing required fields ({missing})")
            # Without schema guarantees we cannot enforce further invariants.
            prev_number = block_number if block_number is not None else prev_number
            prev_composite_root = block.get("composite_root", prev_composite_root)
            continue
        if schema_report["status"] == "WARN":
            unexpected = ", ".join(schema_report["unexpected_fields"]) or "unknown"
            warnings.append(f"{label}: unexpected fields ({unexpected})")

        if block_number is None:
            violations.append(f"{label}: invalid block_number")
            continue

        if block_number in seen_numbers:
            violations.append(f"{label}: duplicate block_number")
        else:
            seen_numbers.add(block_number)

        if prev_number is not None:
            expected_number = prev_number + 1
            if block_number != expected_number:
                violations.append(
                    f"{label}: expected block_number {expected_number} after {prev_number}"
                )
            expected_prev_hash = prev_composite_root
            actual_prev = block.get("prev_hash")
            if actual_prev != expected_prev_hash:
                violations.append(
                    f"{label}: prev_hash {actual_prev} != previous composite_root {expected_prev_hash}"
                )
        else:
            if block.get("prev_hash") not in (None, "", 0):
                violations.append(f"{label}: genesis prev_hash must be empty")

        prev_number = block_number
        prev_composite_root = block.get("composite_root")

    if violations:
        status = "BLOCK"
    elif warnings:
        status = "WARN"
    else:
        status = "OK"
    return {
        "schema_version": contract.schema_version,
        "total_blocks": len(blocks),
        "warnings": warnings,
        "violations": violations,
        "status": status,
    }


def summarize_monotone_ledger_for_global_health(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build CI/governance-friendly view of monotone ledger validation.
    """
    violations = result.get("violations", [])
    warnings = result.get("warnings", [])
    violation_count = len(violations)

    if violation_count > 0:
        status = "BLOCK"
        ledger_monotone = False
        headline = f"Ledger monotonicity violations detected ({violation_count})"
    elif warnings:
        status = "WARN"
        ledger_monotone = True
        headline = "Ledger monotone with schema warnings"
    else:
        status = "OK"
        ledger_monotone = True
        headline = "Ledger monotone"

    return {
        "schema_version": result.get("schema_version", BLOCK_SCHEMA_CONTRACT.schema_version),
        "ledger_monotone": ledger_monotone,
        "violation_count": violation_count,
        "status": status,
        "headline": headline,
    }


def to_governance_signal_for_ledger(summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Produce a GovernanceSignal-style payload from ledger summary.
    """
    return {
        "component": "ledger_monotone",
        "status": summary["status"],
        "headline": summary["headline"],
        "metadata": {
            "schema_version": summary["schema_version"],
            "ledger_monotone": summary["ledger_monotone"],
            "violation_count": summary["violation_count"],
        },
    }


def _statements_are_canonical(statements: Sequence[str]) -> bool:
    return list(statements) == sorted(set(statements))


def _proofs_are_canonical(proofs: Sequence[Mapping[str, Any]]) -> bool:
    observed = [(p["hash"], p["statement_hash"]) for p in proofs]
    return observed == sorted(observed)


def _resolve_block_number(header: Mapping[str, Any]) -> Optional[int]:
    value = header.get("block_number")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _isoformat(dt: datetime) -> str:
    if dt.tzinfo is None:
        raise ValueError("sealed_at timestamp must be timezone-aware")
    return dt.isoformat(timespec="microseconds")


def _build_schema_violation_message(validation: Mapping[str, Any]) -> str:
    parts: List[str] = []
    missing = validation.get("missing_fields") or ()
    unexpected = validation.get("unexpected_fields") or ()

    if missing:
        parts.append(f"missing={list(missing)}")
    if unexpected:
        parts.append(f"unexpected={list(unexpected)}")

    return "; ".join(parts) or "Schema deviation detected"


def _load_blocks_from_file(path: Path) -> List[Mapping[str, Any]]:
    blocks: List[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            if isinstance(record, Mapping) and "header" in record and isinstance(
                record["header"], Mapping
            ):
                blocks.append(record["header"])
            elif isinstance(record, Mapping):
                blocks.append(record)
            else:
                raise ValueError("Each JSONL entry must be an object or contain 'header'")
    return blocks


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate monotone ledger properties for block headers"
    )
    parser.add_argument(
        "--blocks",
        required=True,
        type=Path,
        help="Path to blocks.jsonl (headers or block records with 'header')",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit global-health JSON summary instead of human-readable output",
    )
    args = parser.parse_args(argv)

    blocks = _load_blocks_from_file(args.blocks)
    monotone_result = check_monotone_ledger(blocks)
    summary = summarize_monotone_ledger_for_global_health(monotone_result)

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"[{monotone_result['status']}] Ledger monotone check")
        print(f"Total blocks: {monotone_result['total_blocks']}")
        print(f"Violations: {len(monotone_result['violations'])}")
        if monotone_result["violations"]:
            for violation in monotone_result["violations"]:
                print(f"  - {violation}")
        if monotone_result.get("warnings"):
            print("Warnings:")
            for warning in monotone_result["warnings"]:
                print(f"  - {warning}")
        print("Global health summary:")
        print(json.dumps(summary, indent=2))

    return 0 if monotone_result["status"] == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
