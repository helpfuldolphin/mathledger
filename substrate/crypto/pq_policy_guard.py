"""
Post-Quantum Hash Governance Guard.

Evaluates block headers against epoch-based PQ policy windows. Each epoch
declares the permitted algorithm IDs plus whether dual commitments and legacy
hashes are mandatory.  This guard allows the ledger to roll through PQ upgrades
while keeping deterministic enforcement at block-seal time.

Integration guidance (seal_block_with_dual_roots):
1. After computing the per-block hash commitments but before returning the block
   dict, construct a `BlockHeaderPQ` with the block number, selected algorithm
   (typically "sha256" during transition, eventually "sha3-256"), and booleans
   indicating whether the `hash_commitments` payload contains dual commitments
   and legacy hashes.
2. Call `validate_block_pq_policy(header, policy)` using the policy loaded from
   `config/pq_hash_policy.yaml`. If the verdict fails, propagate the result via:
     * block["attestation_metadata"]["pq_policy_verdict"] so downstream systems
       can see which rule tripped,
     * logging / alerting (Hermetic Matrix lane) to fan-out proactive warnings,
     * optional raising/SPAN gating if policy must hard-block sealing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml

# --------------------------------------------------------------------------- #
# Data types
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BlockHeaderPQ:
    """Minimal header view for PQ validation."""

    block_number: int
    algorithm_id: str
    has_dual_commitment: bool
    has_legacy_hash: bool


@dataclass(frozen=True)
class PQPolicyEpoch:
    """Single epoch window."""

    name: str
    start_block: int
    end_block: Optional[int]
    allowed_algorithms: tuple[str, ...]
    require_dual_commitment: bool
    require_legacy_hash: bool

    def contains(self, block_number: int) -> bool:
        if block_number < self.start_block:
            return False
        if self.end_block is not None and block_number > self.end_block:
            return False
        return True


@dataclass(frozen=True)
class PQPolicyConfig:
    """Container for ordered policy epochs."""

    epochs: tuple[PQPolicyEpoch, ...]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PQPolicyConfig":
        raw_epochs = data.get("epochs") or data.get("epoch_windows")
        if not raw_epochs:
            raise ValueError("PQ policy config missing 'epochs'")

        epochs: List[PQPolicyEpoch] = []
        for entry in raw_epochs:
            epochs.append(
                PQPolicyEpoch(
                    name=entry["name"],
                    start_block=int(entry["start_block"]),
                    end_block=(
                        None if entry.get("end_block") in (None, "null") else int(entry["end_block"])
                    ),
                    allowed_algorithms=tuple(entry["allowed_algorithms"]),
                    require_dual_commitment=bool(entry.get("require_dual_commitment", False)),
                    require_legacy_hash=bool(entry.get("require_legacy_hash", True)),
                )
            )

        epochs.sort(key=lambda e: e.start_block)
        cls._validate_ranges(epochs)
        return cls(epochs=tuple(epochs))

    @classmethod
    def from_file(cls, path: str | Path) -> "PQPolicyConfig":
        path = Path(path)
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        return cls.from_dict(data)

    @staticmethod
    def _validate_ranges(epochs: Iterable[PQPolicyEpoch]) -> None:
        prev_end: Optional[int] = None
        for epoch in epochs:
            if prev_end is not None and epoch.start_block <= prev_end:
                raise ValueError("PQ policy epochs overlap or are unsorted")
            if epoch.end_block is not None and epoch.end_block < epoch.start_block:
                raise ValueError(f"Epoch {epoch.name} end_block precedes start_block")
            prev_end = epoch.end_block if epoch.end_block is not None else prev_end

    def epoch_for_block(self, block_number: int) -> Optional[PQPolicyEpoch]:
        for epoch in self.epochs:
            if epoch.contains(block_number):
                return epoch
        return None


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def validate_block_pq_policy(block_header: BlockHeaderPQ, policy: PQPolicyConfig) -> Dict[str, Any]:
    """
    Enforce PQ hash policy for a given block header.

    Returns:
        Dictionary with verdict and context. On failure "ok" is False and
        "code" indicates the violation.
    """
    epoch = policy.epoch_for_block(block_header.block_number)
    base_result = {
        "ok": True,
        "epoch": epoch.name if epoch else None,
        "block_number": block_header.block_number,
        "algorithm_id": block_header.algorithm_id,
    }

    if epoch is None:
        return {
            **base_result,
            "ok": False,
            "code": "UNKNOWN_EPOCH",
            "reason": "no PQ epoch covers this block number",
        }

    if block_header.algorithm_id not in epoch.allowed_algorithms:
        return {
            **base_result,
            "ok": False,
            "code": "ALGORITHM_NOT_ALLOWED",
            "reason": f"algorithm_id '{block_header.algorithm_id}' not permitted in epoch {epoch.name}",
        }

    if epoch.require_dual_commitment and not block_header.has_dual_commitment:
        return {
            **base_result,
            "ok": False,
            "code": "DUAL_COMMITMENT_REQUIRED",
            "reason": f"epoch {epoch.name} requires dual commitments",
        }

    if epoch.require_legacy_hash and not block_header.has_legacy_hash:
        return {
            **base_result,
            "ok": False,
            "code": "LEGACY_HASH_REQUIRED",
            "reason": f"epoch {epoch.name} still mandates legacy hash outputs",
        }

    return base_result


__all__ = [
    "BlockHeaderPQ",
    "PQPolicyEpoch",
    "PQPolicyConfig",
    "validate_block_pq_policy",
]


def summarize_pq_policy_for_global_health(results: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate PQ policy verdicts for display on the global health surface.

    Args:
        results: Sequence of dictionaries returned from validate_block_pq_policy.

    Returns:
        Summary dictionary with status, violation counts, and latest epoch info.
    """
    results = list(results)
    if not results:
        return {
            "status": "unknown",
            "violations": 0,
            "total_checks": 0,
            "current_epoch": None,
            "latest_block": None,
            "violation_codes": [],
        }

    latest = max(results, key=lambda r: r.get("block_number", -1))
    violations = [r for r in results if not r.get("ok", False)]
    violation_codes = sorted({r.get("code") for r in violations if r.get("code")})

    status = "pass" if not violations else "alert"

    return {
        "status": status,
        "violations": len(violations),
        "total_checks": len(results),
        "current_epoch": latest.get("epoch"),
        "latest_block": latest.get("block_number"),
        "violation_codes": violation_codes,
    }


__all__.append("summarize_pq_policy_for_global_health")
