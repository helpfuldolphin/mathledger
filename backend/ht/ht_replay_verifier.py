"""
HT-Series Replay Invariant Verifier for MathLedger Phase II

This module implements the verification logic for the INV-REPLAY-HT-* invariants
defined in H_T_SERIES_GOVERNANCE_CHARTER.md v1.1.0 Section 10.

Contract Version: 1.0.0
    - Defines the stable report format for HT replay triangle verification
    - Includes invariant status map, series hashes, and MDAP binding status
    - Provides governance summary helper for MAAS/global health integration

STATUS: PHASE II - NOT RUN IN PHASE I
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# ==============================================================================
# Contract Version
# ==============================================================================

TRIANGLE_CONTRACT_VERSION = "1.0.0"
"""
Versioned contract for HT replay triangle verification reports.

Version History:
    1.0.0 - Initial stable contract
        - Invariant status map (INV-REPLAY-HT-1..5)
        - Primary/replay series hashes
        - MDAP binding status
        - Governance summary helper
"""

# ==============================================================================
# Domain Prefixes (from Charter §10.6)
# ==============================================================================

DOMAIN_PRIMARY_REPLAY_BINDING = b"MathLedger:PrimaryReplayBinding:v2:"
DOMAIN_REPLAY_RECEIPT = b"MathLedger:ReplayReceipt:v2:"
DOMAIN_HT_MDAP_BIND = b"MathLedger:HtMdapBinding:v2:"
DOMAIN_MDAP_ATTEST = b"MathLedger:MDAPAttestation:v2:"


# ==============================================================================
# Result Types
# ==============================================================================

class InvariantStatus(Enum):
    """Status of an invariant check."""
    PASS = "PASS"
    FAIL = "FAIL"


class FailureSeverity(Enum):
    """Severity classification from Charter §10.5.1."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"


@dataclass
class InvariantResult:
    """Result of a single invariant check."""
    invariant_id: str
    status: InvariantStatus
    message: str
    severity: Optional[FailureSeverity] = None
    expected: Optional[str] = None
    actual: Optional[str] = None
    cycle: Optional[int] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "invariant_id": self.invariant_id,
            "status": self.status.value,
            "message": self.message,
        }
        if self.severity:
            result["severity"] = self.severity.value
        if self.expected:
            result["expected"] = self.expected
        if self.actual:
            result["actual"] = self.actual
        if self.cycle is not None:
            result["cycle"] = self.cycle
        if self.details:
            result["details"] = self.details
        return result


@dataclass
class Divergence:
    """Information about where two Ht series diverge."""
    cycle: int
    primary_ht: Optional[str] = None
    replay_ht: Optional[str] = None
    primary_rt: Optional[str] = None
    replay_rt: Optional[str] = None
    reason: Optional[str] = None


@dataclass
class TriangleVerificationResult:
    """Result of MDAP-Ht-Replay triangle verification."""
    triangle_valid: bool
    invariants_checked: List[str]
    results: List[InvariantResult]
    mdap_vertex_valid: bool = False
    primary_ht_vertex_valid: bool = False
    replay_ht_vertex_valid: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "triangle_valid": self.triangle_valid,
            "invariants_checked": self.invariants_checked,
            "results": [r.to_dict() for r in self.results],
            "mdap_vertex_valid": self.mdap_vertex_valid,
            "primary_ht_vertex_valid": self.primary_ht_vertex_valid,
            "replay_ht_vertex_valid": self.replay_ht_vertex_valid,
        }


# ==============================================================================
# Helper Functions
# ==============================================================================

def canonical_json(obj: Dict[str, Any]) -> bytes:
    """Computes the canonical JSON representation of an object."""
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(',', ':'),
        ensure_ascii=False
    ).encode('utf-8')


def sha256_hex(data: bytes) -> str:
    """Compute SHA256 hash and return as lowercase hex string."""
    return hashlib.sha256(data).hexdigest().lower()


def sha256_file(file_path: Path) -> str:
    """Compute SHA256 hash of a file's contents."""
    with open(file_path, 'rb') as f:
        return sha256_hex(f.read())


def decode_hex(hex_string: str) -> bytes:
    """Decode a hex string to bytes."""
    return bytes.fromhex(hex_string)


# ==============================================================================
# INV-REPLAY-HT-1: Ht Series Identity
# Charter §10.3.1
# ==============================================================================

def check_INV_REPLAY_HT_1(
    primary_ht_series: Union[Dict, Path, str],
    replay_ht_series: Union[Dict, Path, str]
) -> InvariantResult:
    """
    Verify INV-REPLAY-HT-1: Primary and replay Ht series MUST be byte-identical.

    Args:
        primary_ht_series: Primary ht_series.json as dict, Path, or JSON string
        replay_ht_series: Replay ht_series.json as dict, Path, or JSON string

    Returns:
        InvariantResult with PASS if series are identical, FAIL otherwise

    Severity: CRITICAL
    Impact: Run INVALID if failed
    """
    invariant_id = "INV-REPLAY-HT-1"

    # Get raw bytes for hashing
    if isinstance(primary_ht_series, Path):
        with open(primary_ht_series, 'rb') as f:
            primary_bytes = f.read()
    elif isinstance(primary_ht_series, str):
        primary_bytes = primary_ht_series.encode('utf-8')
    else:
        primary_bytes = canonical_json(primary_ht_series)

    if isinstance(replay_ht_series, Path):
        with open(replay_ht_series, 'rb') as f:
            replay_bytes = f.read()
    elif isinstance(replay_ht_series, str):
        replay_bytes = replay_ht_series.encode('utf-8')
    else:
        replay_bytes = canonical_json(replay_ht_series)

    primary_hash = sha256_hex(primary_bytes)
    replay_hash = sha256_hex(replay_bytes)

    if primary_hash == replay_hash:
        return InvariantResult(
            invariant_id=invariant_id,
            status=InvariantStatus.PASS,
            message="Primary and replay Ht series are byte-identical",
            details={"hash": primary_hash}
        )
    else:
        return InvariantResult(
            invariant_id=invariant_id,
            status=InvariantStatus.FAIL,
            message="Primary and replay Ht series differ",
            severity=FailureSeverity.CRITICAL,
            expected=primary_hash,
            actual=replay_hash,
            details={
                "impact": "Run INVALID",
                "cause_candidates": [
                    "Non-deterministic behavior in proof generation",
                    "External state dependency (time, random, network)",
                    "Different candidate sets between runs",
                    "Implementation bug in Rt computation"
                ]
            }
        )


# ==============================================================================
# INV-REPLAY-HT-2: Chain Final Equivalence
# Charter §10.3.2
# ==============================================================================

def find_first_divergence(
    primary_entries: List[Dict],
    replay_entries: List[Dict]
) -> Optional[Divergence]:
    """
    Find the first cycle where primary and replay Ht series diverge.

    Returns None if no divergence found.
    """
    min_len = min(len(primary_entries), len(replay_entries))

    for i in range(min_len):
        if primary_entries[i].get("H_t") != replay_entries[i].get("H_t"):
            return Divergence(
                cycle=i,
                primary_ht=primary_entries[i].get("H_t"),
                replay_ht=replay_entries[i].get("H_t"),
                primary_rt=primary_entries[i].get("R_t"),
                replay_rt=replay_entries[i].get("R_t")
            )

    if len(primary_entries) != len(replay_entries):
        return Divergence(
            cycle=min_len,
            reason="LENGTH_MISMATCH"
        )

    return None


def check_INV_REPLAY_HT_2(
    primary_ht_series: Dict,
    replay_ht_series: Dict
) -> InvariantResult:
    """
    Verify INV-REPLAY-HT-2: Primary and replay chain_final MUST be identical.

    Args:
        primary_ht_series: Parsed primary ht_series.json
        replay_ht_series: Parsed replay ht_series.json

    Returns:
        InvariantResult with PASS if chain finals match, FAIL otherwise

    Severity: CRITICAL
    Impact: Run INVALID if failed
    """
    invariant_id = "INV-REPLAY-HT-2"

    primary_chain = primary_ht_series.get("summary", {}).get("chain_final")
    replay_chain = replay_ht_series.get("summary", {}).get("chain_final")

    if primary_chain is None:
        return InvariantResult(
            invariant_id=invariant_id,
            status=InvariantStatus.FAIL,
            message="Primary series missing chain_final",
            severity=FailureSeverity.CRITICAL,
            details={"error": "MISSING_PRIMARY_CHAIN_FINAL"}
        )

    if replay_chain is None:
        return InvariantResult(
            invariant_id=invariant_id,
            status=InvariantStatus.FAIL,
            message="Replay series missing chain_final",
            severity=FailureSeverity.CRITICAL,
            details={"error": "MISSING_REPLAY_CHAIN_FINAL"}
        )

    if primary_chain == replay_chain:
        return InvariantResult(
            invariant_id=invariant_id,
            status=InvariantStatus.PASS,
            message="Chain finals are identical",
            details={"chain_final": primary_chain}
        )
    else:
        # Find divergence point for diagnostics
        divergence = find_first_divergence(
            primary_ht_series.get("series", []),
            replay_ht_series.get("series", [])
        )

        details = {
            "impact": "Run INVALID",
            "cause": "Cumulative attestation chains diverged"
        }

        if divergence:
            details["first_divergence_cycle"] = divergence.cycle
            if divergence.reason:
                details["divergence_reason"] = divergence.reason
            if divergence.primary_rt and divergence.replay_rt:
                details["rt_match_at_divergence"] = (
                    divergence.primary_rt == divergence.replay_rt
                )

        return InvariantResult(
            invariant_id=invariant_id,
            status=InvariantStatus.FAIL,
            message="Chain finals differ",
            severity=FailureSeverity.CRITICAL,
            expected=primary_chain,
            actual=replay_chain,
            cycle=divergence.cycle if divergence else None,
            details=details
        )


# ==============================================================================
# INV-REPLAY-HT-3: MDAP Binding Preservation
# Charter §10.3.3
# ==============================================================================

def check_INV_REPLAY_HT_3(
    primary_ht_series: Dict,
    replay_ht_series: Dict,
    manifest: Optional[Dict] = None
) -> InvariantResult:
    """
    Verify INV-REPLAY-HT-3: Primary and replay ht_mdap_binding MUST be identical.

    Args:
        primary_ht_series: Parsed primary ht_series.json
        replay_ht_series: Parsed replay ht_series.json
        manifest: Optional parsed manifest for binding verification

    Returns:
        InvariantResult with PASS if bindings match, FAIL otherwise

    Severity: CRITICAL
    Impact: Run INVALID if failed
    """
    invariant_id = "INV-REPLAY-HT-3"

    primary_binding = primary_ht_series.get("summary", {}).get("ht_mdap_binding")
    replay_binding = replay_ht_series.get("summary", {}).get("ht_mdap_binding")

    if primary_binding is None:
        return InvariantResult(
            invariant_id=invariant_id,
            status=InvariantStatus.FAIL,
            message="Primary series missing ht_mdap_binding",
            severity=FailureSeverity.CRITICAL,
            details={"error": "MISSING_PRIMARY_BINDING"}
        )

    if replay_binding is None:
        return InvariantResult(
            invariant_id=invariant_id,
            status=InvariantStatus.FAIL,
            message="Replay series missing ht_mdap_binding",
            severity=FailureSeverity.CRITICAL,
            details={"error": "MISSING_REPLAY_BINDING"}
        )

    if primary_binding != replay_binding:
        return InvariantResult(
            invariant_id=invariant_id,
            status=InvariantStatus.FAIL,
            message="MDAP bindings differ",
            severity=FailureSeverity.CRITICAL,
            expected=primary_binding,
            actual=replay_binding,
            details={
                "impact": "Run INVALID",
                "reason": "BINDING_MISMATCH",
                "cause_candidates": [
                    "Different MDAP attestation used between runs",
                    "Chain final mismatch",
                    "Bug in binding computation"
                ]
            }
        )

    # If manifest provided, verify binding is correctly computed
    if manifest:
        mdap_attestation_hash = manifest.get("mdap_attestation", {}).get("attestation_hash")
        chain_final = primary_ht_series.get("summary", {}).get("chain_final")

        if mdap_attestation_hash and chain_final:
            expected_binding = hashlib.sha256(
                DOMAIN_HT_MDAP_BIND +
                decode_hex(chain_final) +
                decode_hex(mdap_attestation_hash)
            ).hexdigest().lower()

            if expected_binding != primary_binding.lower():
                return InvariantResult(
                    invariant_id=invariant_id,
                    status=InvariantStatus.FAIL,
                    message="MDAP binding computation error",
                    severity=FailureSeverity.CRITICAL,
                    expected=expected_binding,
                    actual=primary_binding,
                    details={
                        "impact": "Run INVALID",
                        "reason": "BINDING_COMPUTATION_ERROR"
                    }
                )

    return InvariantResult(
        invariant_id=invariant_id,
        status=InvariantStatus.PASS,
        message="MDAP bindings are identical",
        details={"ht_mdap_binding": primary_binding}
    )


# ==============================================================================
# INV-REPLAY-HT-4: Replay Receipt Integrity
# Charter §10.3.4
# ==============================================================================

def check_INV_REPLAY_HT_4(
    replay_receipt: Dict,
    primary_series_path: Optional[Path] = None,
    replay_series_path: Optional[Path] = None,
    primary_ht_series: Optional[Dict] = None,
    replay_ht_series: Optional[Dict] = None
) -> InvariantResult:
    """
    Verify INV-REPLAY-HT-4: Replay Receipt MUST correctly bind primary and replay artifacts.

    Args:
        replay_receipt: Parsed replay_receipt.json
        primary_series_path: Path to primary ht_series.json (for hash verification)
        replay_series_path: Path to replay ht_series.json (for hash verification)
        primary_ht_series: Parsed primary ht_series (for chain final verification)
        replay_ht_series: Parsed replay ht_series (for chain final verification)

    Returns:
        InvariantResult with PASS if receipt is valid, FAIL otherwise

    Severity: HIGH
    Impact: Requires manual investigation if failed
    """
    invariant_id = "INV-REPLAY-HT-4"
    checks_passed = []
    checks_failed = []

    # Check 1: Verify primary series hash if path provided
    if primary_series_path:
        actual_primary_hash = sha256_file(primary_series_path)
        claimed_primary_hash = replay_receipt.get("primary_run", {}).get("ht_series_hash")

        if claimed_primary_hash and actual_primary_hash != claimed_primary_hash.lower():
            checks_failed.append({
                "check": "PRIMARY_SERIES_HASH",
                "expected": claimed_primary_hash,
                "actual": actual_primary_hash
            })
        else:
            checks_passed.append("PRIMARY_SERIES_HASH")

    # Check 2: Verify replay series hash if path provided
    if replay_series_path:
        actual_replay_hash = sha256_file(replay_series_path)
        claimed_replay_hash = replay_receipt.get("replay_run", {}).get("ht_series_hash")

        if claimed_replay_hash and actual_replay_hash != claimed_replay_hash.lower():
            checks_failed.append({
                "check": "REPLAY_SERIES_HASH",
                "expected": claimed_replay_hash,
                "actual": actual_replay_hash
            })
        else:
            checks_passed.append("REPLAY_SERIES_HASH")

    # Check 3: Verify primary chain final matches series
    if primary_ht_series:
        series_chain = primary_ht_series.get("summary", {}).get("chain_final")
        receipt_chain = replay_receipt.get("primary_run", {}).get("chain_final")

        if series_chain and receipt_chain and series_chain != receipt_chain:
            checks_failed.append({
                "check": "PRIMARY_CHAIN_FINAL",
                "expected": series_chain,
                "actual": receipt_chain
            })
        else:
            checks_passed.append("PRIMARY_CHAIN_FINAL")

    # Check 4: Verify replay chain final matches series
    if replay_ht_series:
        series_chain = replay_ht_series.get("summary", {}).get("chain_final")
        receipt_chain = replay_receipt.get("replay_run", {}).get("chain_final")

        if series_chain and receipt_chain and series_chain != receipt_chain:
            checks_failed.append({
                "check": "REPLAY_CHAIN_FINAL",
                "expected": series_chain,
                "actual": receipt_chain
            })
        else:
            checks_passed.append("REPLAY_CHAIN_FINAL")

    # Check 5: Verify receipt self-hash
    binding = replay_receipt.get("binding", {})
    claimed_receipt_hash = binding.get("receipt_hash")

    if claimed_receipt_hash:
        # Create copy without receipt_hash for verification
        receipt_for_hash = json.loads(json.dumps(replay_receipt))
        if "binding" in receipt_for_hash and "receipt_hash" in receipt_for_hash["binding"]:
            del receipt_for_hash["binding"]["receipt_hash"]

        expected_receipt_hash = sha256_hex(canonical_json(receipt_for_hash))

        if expected_receipt_hash != claimed_receipt_hash.lower():
            checks_failed.append({
                "check": "RECEIPT_SELF_HASH",
                "expected": expected_receipt_hash,
                "actual": claimed_receipt_hash
            })
        else:
            checks_passed.append("RECEIPT_SELF_HASH")

    if checks_failed:
        return InvariantResult(
            invariant_id=invariant_id,
            status=InvariantStatus.FAIL,
            message=f"Receipt integrity check failed: {len(checks_failed)} issue(s)",
            severity=FailureSeverity.HIGH,
            details={
                "impact": "Requires manual investigation",
                "checks_passed": checks_passed,
                "checks_failed": checks_failed,
                "cause_candidates": [
                    "Receipt generated from wrong artifacts",
                    "Receipt modified after generation",
                    "Hash computation bug",
                    "File corruption"
                ]
            }
        )

    return InvariantResult(
        invariant_id=invariant_id,
        status=InvariantStatus.PASS,
        message="Replay receipt integrity verified",
        details={"checks_passed": checks_passed}
    )


# ==============================================================================
# INV-REPLAY-HT-5: Primary-Replay Binding Hash
# Charter §10.3.5
# ==============================================================================

def check_INV_REPLAY_HT_5(
    replay_receipt: Dict
) -> InvariantResult:
    """
    Verify INV-REPLAY-HT-5: The primary-replay binding MUST cryptographically
    link both chain finals.

    Args:
        replay_receipt: Parsed replay_receipt.json

    Returns:
        InvariantResult with PASS if binding is correct, FAIL otherwise

    Severity: HIGH
    Impact: Requires manual investigation if failed
    """
    invariant_id = "INV-REPLAY-HT-5"

    primary_chain = replay_receipt.get("primary_run", {}).get("chain_final")
    replay_chain = replay_receipt.get("replay_run", {}).get("chain_final")
    claimed_binding = replay_receipt.get("binding", {}).get("primary_replay_binding")

    if not primary_chain:
        return InvariantResult(
            invariant_id=invariant_id,
            status=InvariantStatus.FAIL,
            message="Receipt missing primary chain_final",
            severity=FailureSeverity.HIGH,
            details={"error": "MISSING_PRIMARY_CHAIN"}
        )

    if not replay_chain:
        return InvariantResult(
            invariant_id=invariant_id,
            status=InvariantStatus.FAIL,
            message="Receipt missing replay chain_final",
            severity=FailureSeverity.HIGH,
            details={"error": "MISSING_REPLAY_CHAIN"}
        )

    if not claimed_binding:
        return InvariantResult(
            invariant_id=invariant_id,
            status=InvariantStatus.FAIL,
            message="Receipt missing primary_replay_binding",
            severity=FailureSeverity.HIGH,
            details={"error": "MISSING_BINDING"}
        )

    # Compute expected binding
    expected_binding = hashlib.sha256(
        DOMAIN_PRIMARY_REPLAY_BINDING +
        decode_hex(primary_chain) +
        decode_hex(replay_chain)
    ).hexdigest().lower()

    if expected_binding != claimed_binding.lower():
        return InvariantResult(
            invariant_id=invariant_id,
            status=InvariantStatus.FAIL,
            message="Primary-replay binding hash mismatch",
            severity=FailureSeverity.HIGH,
            expected=expected_binding,
            actual=claimed_binding,
            details={
                "impact": "Requires manual investigation",
                "cause_candidates": [
                    "Bug in binding hash computation",
                    "Wrong domain prefix used",
                    "Receipt corruption"
                ]
            }
        )

    return InvariantResult(
        invariant_id=invariant_id,
        status=InvariantStatus.PASS,
        message="Primary-replay binding hash verified",
        details={"primary_replay_binding": claimed_binding}
    )


# ==============================================================================
# MDAP-Ht-Replay Triangle Verification
# Charter §10.4
# ==============================================================================

def verify_mdap_ht_replay_triangle(
    manifest: Dict,
    primary_ht_series: Dict,
    replay_ht_series: Dict,
    replay_receipt: Dict,
    primary_series_path: Optional[Path] = None,
    replay_series_path: Optional[Path] = None
) -> TriangleVerificationResult:
    """
    Verify the complete MDAP-Ht-Replay triangle as per Charter §10.4.

    The triangle establishes cryptographic relationships between:
    1. MDAP Attestation (seed schedule)
    2. Primary Ht Series
    3. Replay Ht Series + Receipt

    Args:
        manifest: Parsed experiment manifest
        primary_ht_series: Parsed primary ht_series.json
        replay_ht_series: Parsed replay ht_series.json
        replay_receipt: Parsed replay_receipt.json
        primary_series_path: Optional path for file hash verification
        replay_series_path: Optional path for file hash verification

    Returns:
        TriangleVerificationResult with all check results
    """
    results: List[InvariantResult] = []
    invariants_checked: List[str] = []

    # === MDAP VERTEX ===
    mdap_valid = True
    mdap_attestation = manifest.get("mdap_attestation", {})
    if not mdap_attestation.get("attestation_hash"):
        mdap_valid = False
        results.append(InvariantResult(
            invariant_id="MDAP_VERTEX",
            status=InvariantStatus.FAIL,
            message="MDAP attestation missing or invalid",
            details={"vertex": "MDAP", "reason": "INVALID_ATTESTATION"}
        ))

    # === PRIMARY Ht VERTEX ===
    primary_valid = True
    primary_chain = primary_ht_series.get("summary", {}).get("chain_final")
    primary_binding = primary_ht_series.get("summary", {}).get("ht_mdap_binding")

    if not primary_chain or not primary_binding:
        primary_valid = False
        results.append(InvariantResult(
            invariant_id="PRIMARY_HT_VERTEX",
            status=InvariantStatus.FAIL,
            message="Primary Ht series incomplete",
            details={
                "vertex": "PRIMARY_HT",
                "has_chain_final": bool(primary_chain),
                "has_mdap_binding": bool(primary_binding)
            }
        ))

    # === REPLAY Ht VERTEX ===
    replay_valid = True
    replay_chain = replay_ht_series.get("summary", {}).get("chain_final")
    replay_binding = replay_ht_series.get("summary", {}).get("ht_mdap_binding")

    if not replay_chain or not replay_binding:
        replay_valid = False
        results.append(InvariantResult(
            invariant_id="REPLAY_HT_VERTEX",
            status=InvariantStatus.FAIL,
            message="Replay Ht series incomplete",
            details={
                "vertex": "REPLAY_HT",
                "has_chain_final": bool(replay_chain),
                "has_mdap_binding": bool(replay_binding)
            }
        ))

    # === EDGE: PRIMARY <-> REPLAY (INV-REPLAY-HT-1) ===
    result_1 = check_INV_REPLAY_HT_1(primary_ht_series, replay_ht_series)
    results.append(result_1)
    invariants_checked.append("INV-REPLAY-HT-1")

    # === EDGE: PRIMARY <-> REPLAY CHAIN (INV-REPLAY-HT-2) ===
    result_2 = check_INV_REPLAY_HT_2(primary_ht_series, replay_ht_series)
    results.append(result_2)
    invariants_checked.append("INV-REPLAY-HT-2")

    # === EDGE: MDAP BINDING (INV-REPLAY-HT-3) ===
    result_3 = check_INV_REPLAY_HT_3(primary_ht_series, replay_ht_series, manifest)
    results.append(result_3)
    invariants_checked.append("INV-REPLAY-HT-3")

    # === RECEIPT VERTEX (INV-REPLAY-HT-4) ===
    result_4 = check_INV_REPLAY_HT_4(
        replay_receipt,
        primary_series_path,
        replay_series_path,
        primary_ht_series,
        replay_ht_series
    )
    results.append(result_4)
    invariants_checked.append("INV-REPLAY-HT-4")

    # === RECEIPT BINDING (INV-REPLAY-HT-5) ===
    result_5 = check_INV_REPLAY_HT_5(replay_receipt)
    results.append(result_5)
    invariants_checked.append("INV-REPLAY-HT-5")

    # Determine overall validity
    all_passed = all(r.status == InvariantStatus.PASS for r in results)

    return TriangleVerificationResult(
        triangle_valid=all_passed,
        invariants_checked=invariants_checked,
        results=results,
        mdap_vertex_valid=mdap_valid,
        primary_ht_vertex_valid=primary_valid,
        replay_ht_vertex_valid=replay_valid
    )


# ==============================================================================
# Composite Verification
# ==============================================================================

def verify_all_replay_invariants(
    primary_ht_series: Dict,
    replay_ht_series: Dict,
    replay_receipt: Dict,
    manifest: Optional[Dict] = None,
    primary_series_path: Optional[Path] = None,
    replay_series_path: Optional[Path] = None
) -> List[InvariantResult]:
    """
    Run all INV-REPLAY-HT-* checks and return results.

    Args:
        primary_ht_series: Parsed primary ht_series.json
        replay_ht_series: Parsed replay ht_series.json
        replay_receipt: Parsed replay_receipt.json
        manifest: Optional parsed manifest for binding verification
        primary_series_path: Optional path for file hash verification
        replay_series_path: Optional path for file hash verification

    Returns:
        List of InvariantResult for each check
    """
    results = []

    # INV-REPLAY-HT-1
    results.append(check_INV_REPLAY_HT_1(primary_ht_series, replay_ht_series))

    # INV-REPLAY-HT-2
    results.append(check_INV_REPLAY_HT_2(primary_ht_series, replay_ht_series))

    # INV-REPLAY-HT-3
    results.append(check_INV_REPLAY_HT_3(primary_ht_series, replay_ht_series, manifest))

    # INV-REPLAY-HT-4
    results.append(check_INV_REPLAY_HT_4(
        replay_receipt,
        primary_series_path,
        replay_series_path,
        primary_ht_series,
        replay_ht_series
    ))

    # INV-REPLAY-HT-5
    results.append(check_INV_REPLAY_HT_5(replay_receipt))

    return results


def generate_verification_report(
    results: List[InvariantResult],
    experiment_id: str,
    replay_id: str,
    primary_ht_series: Optional[Dict] = None,
    replay_ht_series: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Generate a JSON verification report from invariant results.

    Conforms to Charter §10.5.4 failure logging requirements.
    Implements Triangle Contract v1.0.0 with:
        - contract_version field
        - primary_ht_series_hash, replay_ht_series_hash
        - invariant_status_map (INV-REPLAY-HT-1..5 → PASS/FAIL)
        - mdap_binding_status

    Args:
        results: List of InvariantResult from verification
        experiment_id: Experiment identifier
        replay_id: Replay run identifier
        primary_ht_series: Optional parsed primary series for hash computation
        replay_ht_series: Optional parsed replay series for hash computation

    Returns:
        Dict conforming to Triangle Contract v1.0.0
    """
    passed = [r for r in results if r.status == InvariantStatus.PASS]
    failed = [r for r in results if r.status == InvariantStatus.FAIL]

    # Build invariant status map
    invariant_status_map = {}
    for r in results:
        if r.invariant_id.startswith("INV-REPLAY-HT-"):
            invariant_status_map[r.invariant_id] = r.status.value

    # Determine MDAP binding status from INV-REPLAY-HT-3
    mdap_binding_status = "UNKNOWN"
    for r in results:
        if r.invariant_id == "INV-REPLAY-HT-3":
            mdap_binding_status = r.status.value
            break

    # Compute series hashes if data provided
    primary_ht_series_hash = None
    replay_ht_series_hash = None

    if primary_ht_series:
        primary_ht_series_hash = sha256_hex(canonical_json(primary_ht_series))
    if replay_ht_series:
        replay_ht_series_hash = sha256_hex(canonical_json(replay_ht_series))

    report = {
        "meta": {
            "report_version": "2.0.0",
            "contract_version": TRIANGLE_CONTRACT_VERSION,
            "type": "replay_verification_report",
            "generated_utc": datetime.utcnow().isoformat() + "Z"
        },
        "experiment": {
            "experiment_id": experiment_id,
            "replay_id": replay_id
        },
        "contract": {
            "version": TRIANGLE_CONTRACT_VERSION,
            "primary_ht_series_hash": primary_ht_series_hash,
            "replay_ht_series_hash": replay_ht_series_hash,
            "invariant_status_map": invariant_status_map,
            "mdap_binding_status": mdap_binding_status
        },
        "summary": {
            "all_passed": len(failed) == 0,
            "total_checks": len(results),
            "passed": len(passed),
            "failed": len(failed)
        },
        "results": [r.to_dict() for r in results]
    }

    if failed:
        # Add failure log section per Charter §10.5.4
        critical_failures = [f for f in failed if f.severity == FailureSeverity.CRITICAL]
        high_failures = [f for f in failed if f.severity == FailureSeverity.HIGH]

        report["failure_summary"] = {
            "critical_count": len(critical_failures),
            "high_count": len(high_failures),
            "impact": "INVALID" if critical_failures else "UNDER_REVIEW",
            "failed_invariants": [f.invariant_id for f in failed]
        }

    return report


# ==============================================================================
# Governance Summary
# ==============================================================================

class HtReplayStatus(Enum):
    """
    HT Replay status classification for governance/MAAS consumption.

    Status Mapping Rules:
        OK   - All invariants pass (INV-REPLAY-HT-1..5 all PASS)
        WARN - Critical invariants pass but high severity invariants fail
               (INV-REPLAY-HT-1..3 PASS, INV-REPLAY-HT-4 or 5 FAIL)
        FAIL - Any critical invariant fails (INV-REPLAY-HT-1, 2, or 3 FAIL)
    """
    OK = "OK"
    WARN = "WARN"
    FAIL = "FAIL"


# Critical invariants (INV-REPLAY-HT-1, 2, 3) - Run INVALID if any fail
CRITICAL_INVARIANTS = {"INV-REPLAY-HT-1", "INV-REPLAY-HT-2", "INV-REPLAY-HT-3"}

# High severity invariants (INV-REPLAY-HT-4, 5) - Manual investigation if fail
HIGH_INVARIANTS = {"INV-REPLAY-HT-4", "INV-REPLAY-HT-5"}


def summarize_ht_replay_for_governance(report: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a minimal governance summary from a verification report.

    This summary is designed for consumption by:
        - Global health monitoring (MAAS)
        - Dashboard status indicators
        - Automated alerting systems

    Status Classification Rules:
        OK   - All 5 invariants pass
        WARN - Critical invariants (1-3) pass but high severity (4-5) fail
        FAIL - Any critical invariant fails

    Args:
        report: Full verification report from generate_verification_report()

    Returns:
        Dict containing:
            - all_critical_invariants_pass (bool): True if INV-REPLAY-HT-1..3 all pass
            - high_invariants_pass (bool): True if INV-REPLAY-HT-4..5 all pass
            - ht_replay_status (str): "OK", "WARN", or "FAIL"
            - contract_version (str): Contract version for compatibility checking
            - experiment_id (str): Experiment identifier
            - replay_id (str): Replay identifier
    """
    # Extract invariant status map from contract section
    contract = report.get("contract", {})
    invariant_status_map = contract.get("invariant_status_map", {})

    # If no contract section, fall back to results array
    if not invariant_status_map:
        for result in report.get("results", []):
            inv_id = result.get("invariant_id", "")
            if inv_id.startswith("INV-REPLAY-HT-"):
                invariant_status_map[inv_id] = result.get("status", "UNKNOWN")

    # Check critical invariants (INV-REPLAY-HT-1, 2, 3)
    critical_results = []
    for inv_id in CRITICAL_INVARIANTS:
        status = invariant_status_map.get(inv_id, "UNKNOWN")
        critical_results.append(status == "PASS")

    all_critical_pass = all(critical_results) and len(critical_results) == len(CRITICAL_INVARIANTS)

    # Check high severity invariants (INV-REPLAY-HT-4, 5)
    high_results = []
    for inv_id in HIGH_INVARIANTS:
        status = invariant_status_map.get(inv_id, "UNKNOWN")
        high_results.append(status == "PASS")

    all_high_pass = all(high_results) and len(high_results) == len(HIGH_INVARIANTS)

    # Determine overall status
    if all_critical_pass and all_high_pass:
        ht_replay_status = HtReplayStatus.OK
    elif all_critical_pass and not all_high_pass:
        ht_replay_status = HtReplayStatus.WARN
    else:
        ht_replay_status = HtReplayStatus.FAIL

    return {
        "all_critical_invariants_pass": all_critical_pass,
        "high_invariants_pass": all_high_pass,
        "ht_replay_status": ht_replay_status.value,
        "contract_version": contract.get("version", TRIANGLE_CONTRACT_VERSION),
        "experiment_id": report.get("experiment", {}).get("experiment_id", "unknown"),
        "replay_id": report.get("experiment", {}).get("replay_id", "unknown")
    }


# ==============================================================================
# Phase III: HT Replay History Ledger
# ==============================================================================

class DriftStatus(Enum):
    """
    Drift classification for comparing HT replay histories.

    Status Classification:
        STABLE    - No change in overall health between old and new
        IMPROVED  - Health improved (e.g., FAIL→WARN, WARN→OK, FAIL→OK)
        REGRESSED - Health degraded (e.g., OK→WARN, WARN→FAIL, OK→FAIL)
    """
    STABLE = "STABLE"
    IMPROVED = "IMPROVED"
    REGRESSED = "REGRESSED"


@dataclass
class HistoryEntry:
    """A single entry in the HT replay history ledger."""
    experiment_id: str
    replay_id: str
    timestamp_utc: str
    ht_replay_status: str
    all_critical_pass: bool
    high_pass: bool
    failed_invariants: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "experiment_id": self.experiment_id,
            "replay_id": self.replay_id,
            "timestamp_utc": self.timestamp_utc,
            "ht_replay_status": self.ht_replay_status,
            "all_critical_pass": self.all_critical_pass,
            "high_pass": self.high_pass,
            "failed_invariants": self.failed_invariants
        }


def build_ht_replay_history(
    reports: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Build an HT replay history ledger from a list of verification reports.

    This function aggregates multiple verification reports into a history
    structure that tracks:
    - Individual run entries with status
    - Aggregate counts (ok, warn, fail)
    - Recurrent failures (invariants that fail across multiple runs)

    Args:
        reports: List of verification reports from generate_verification_report()

    Returns:
        Dict containing:
            - run_entries: List of HistoryEntry dicts ordered by timestamp
            - ok_count: Number of runs with OK status
            - warn_count: Number of runs with WARN status
            - fail_count: Number of runs with FAIL status
            - recurrent_failures: Dict mapping invariant_id → count of failures
            - total_runs: Total number of runs in history
    """
    run_entries: List[HistoryEntry] = []
    ok_count = 0
    warn_count = 0
    fail_count = 0
    failure_counts: Dict[str, int] = {}

    for report in reports:
        # Extract governance summary
        summary = summarize_ht_replay_for_governance(report)

        # Get timestamp from report meta
        timestamp = report.get("meta", {}).get("generated_utc", "unknown")

        # Get failed invariants from report
        failed_invariants = []
        failure_summary = report.get("failure_summary", {})
        if failure_summary:
            failed_invariants = failure_summary.get("failed_invariants", [])

        # Track failure recurrence
        for inv_id in failed_invariants:
            failure_counts[inv_id] = failure_counts.get(inv_id, 0) + 1

        # Create history entry
        entry = HistoryEntry(
            experiment_id=summary["experiment_id"],
            replay_id=summary["replay_id"],
            timestamp_utc=timestamp,
            ht_replay_status=summary["ht_replay_status"],
            all_critical_pass=summary["all_critical_invariants_pass"],
            high_pass=summary["high_invariants_pass"],
            failed_invariants=failed_invariants
        )
        run_entries.append(entry)

        # Update counts
        status = summary["ht_replay_status"]
        if status == HtReplayStatus.OK.value:
            ok_count += 1
        elif status == HtReplayStatus.WARN.value:
            warn_count += 1
        elif status == HtReplayStatus.FAIL.value:
            fail_count += 1

    # Sort entries by timestamp
    run_entries.sort(key=lambda e: e.timestamp_utc)

    # Filter recurrent failures (appearing in more than one run)
    recurrent_failures = {
        inv_id: count
        for inv_id, count in failure_counts.items()
        if count > 1
    }

    return {
        "run_entries": [e.to_dict() for e in run_entries],
        "ok_count": ok_count,
        "warn_count": warn_count,
        "fail_count": fail_count,
        "recurrent_failures": recurrent_failures,
        "total_runs": len(run_entries)
    }


# ==============================================================================
# Phase III: Drift Classifier
# ==============================================================================

# Status ranking for drift comparison (higher = healthier)
_STATUS_RANK = {
    HtReplayStatus.FAIL.value: 0,
    HtReplayStatus.WARN.value: 1,
    HtReplayStatus.OK.value: 2,
    "FAIL": 0,
    "WARN": 1,
    "OK": 2
}


def compare_ht_history(
    old_history: Dict[str, Any],
    new_history: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compare two HT replay histories to detect drift.

    Compares the overall health trajectory between two history snapshots
    and classifies the change as STABLE, IMPROVED, or REGRESSED.

    Classification Rules:
        STABLE    - Same overall health metrics (ok/warn/fail ratios similar)
        IMPROVED  - Health improved (higher OK ratio, lower FAIL ratio)
        REGRESSED - Health degraded (lower OK ratio, higher FAIL ratio)

    The comparison uses:
    1. Latest run status comparison (primary signal)
    2. Aggregate health score (ok_count - fail_count) as tiebreaker

    Args:
        old_history: Previous history from build_ht_replay_history()
        new_history: Current history from build_ht_replay_history()

    Returns:
        Dict containing:
            - status: DriftStatus value (STABLE/IMPROVED/REGRESSED)
            - old_latest_status: Last status in old history
            - new_latest_status: Last status in new history
            - old_health_score: Aggregate health score for old
            - new_health_score: Aggregate health score for new
            - details: Additional comparison details
    """
    # Get latest status from each history
    old_entries = old_history.get("run_entries", [])
    new_entries = new_history.get("run_entries", [])

    old_latest_status = old_entries[-1]["ht_replay_status"] if old_entries else "UNKNOWN"
    new_latest_status = new_entries[-1]["ht_replay_status"] if new_entries else "UNKNOWN"

    # Compute health scores: OK=+1, WARN=0, FAIL=-1
    old_health_score = (
        old_history.get("ok_count", 0) -
        old_history.get("fail_count", 0)
    )
    new_health_score = (
        new_history.get("ok_count", 0) -
        new_history.get("fail_count", 0)
    )

    # Compare latest status ranks
    old_rank = _STATUS_RANK.get(old_latest_status, -1)
    new_rank = _STATUS_RANK.get(new_latest_status, -1)

    # Determine drift status
    if new_rank > old_rank:
        drift_status = DriftStatus.IMPROVED
    elif new_rank < old_rank:
        drift_status = DriftStatus.REGRESSED
    else:
        # Same latest status - check aggregate health score
        if new_health_score > old_health_score:
            drift_status = DriftStatus.IMPROVED
        elif new_health_score < old_health_score:
            drift_status = DriftStatus.REGRESSED
        else:
            drift_status = DriftStatus.STABLE

    # Build comparison details
    details = {
        "latest_status_changed": old_latest_status != new_latest_status,
        "old_total_runs": old_history.get("total_runs", 0),
        "new_total_runs": new_history.get("total_runs", 0),
        "old_fail_ratio": (
            old_history.get("fail_count", 0) / old_history.get("total_runs", 1)
            if old_history.get("total_runs", 0) > 0 else 0
        ),
        "new_fail_ratio": (
            new_history.get("fail_count", 0) / new_history.get("total_runs", 1)
            if new_history.get("total_runs", 0) > 0 else 0
        ),
        "new_recurrent_failures": list(new_history.get("recurrent_failures", {}).keys())
    }

    return {
        "status": drift_status.value,
        "old_latest_status": old_latest_status,
        "new_latest_status": new_latest_status,
        "old_health_score": old_health_score,
        "new_health_score": new_health_score,
        "details": details
    }


# ==============================================================================
# Phase III: Global Health Summary
# ==============================================================================

def summarize_ht_for_global_health(
    history: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate a minimal global health summary from HT replay history.

    This summary is designed for integration with global system health
    monitoring, providing a single-glance view of HT replay status.

    Args:
        history: History ledger from build_ht_replay_history()

    Returns:
        Dict containing:
            - ht_ok: bool - True if system is healthy (last run OK, no recurrent critical failures)
            - last_status: str - Status of most recent run (OK/WARN/FAIL/UNKNOWN)
            - critical_fail_runs: List of replay_ids where critical invariants failed
            - health_score: int - Aggregate health score (ok_count - fail_count)
            - recurrent_critical_failures: List of critical invariants failing repeatedly
    """
    run_entries = history.get("run_entries", [])
    recurrent_failures = history.get("recurrent_failures", {})

    # Get last status
    last_status = "UNKNOWN"
    if run_entries:
        last_status = run_entries[-1].get("ht_replay_status", "UNKNOWN")

    # Identify critical fail runs (where all_critical_pass is False)
    critical_fail_runs = [
        entry.get("replay_id", "unknown")
        for entry in run_entries
        if not entry.get("all_critical_pass", True)
    ]

    # Identify recurrent critical failures
    recurrent_critical_failures = [
        inv_id for inv_id in recurrent_failures.keys()
        if inv_id in CRITICAL_INVARIANTS
    ]

    # Compute health score
    health_score = (
        history.get("ok_count", 0) -
        history.get("fail_count", 0)
    )

    # Determine if system is healthy:
    # - Last run must be OK
    # - No recurrent critical failures
    ht_ok = (
        last_status == HtReplayStatus.OK.value and
        len(recurrent_critical_failures) == 0
    )

    return {
        "ht_ok": ht_ok,
        "last_status": last_status,
        "critical_fail_runs": critical_fail_runs,
        "health_score": health_score,
        "recurrent_critical_failures": recurrent_critical_failures,
        "total_runs": history.get("total_runs", 0),
        "ok_count": history.get("ok_count", 0),
        "warn_count": history.get("warn_count", 0),
        "fail_count": history.get("fail_count", 0)
    }


# ==============================================================================
# Phase IV: Hₜ Replay Release Evaluator
# ==============================================================================

class ReleaseStatus(Enum):
    """
    Release gate status for Hₜ replay evaluation.

    Status Classification:
        OK    - All checks pass, safe to release
        WARN  - Minor issues detected, release with caution
        BLOCK - Critical issues detected, block release
    """
    OK = "OK"
    WARN = "WARN"
    BLOCK = "BLOCK"


def evaluate_ht_replay_for_release(
    history: Dict[str, Any],
    drift: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluate Hₜ replay status for release gate decision.

    This function acts as a guardrail for determinism & attestations,
    determining whether a release should proceed based on Hₜ replay health.

    Blocking Conditions (status=BLOCK):
        - Last run status is FAIL
        - Drift status is REGRESSED
        - Recurrent critical failures exist
        - Fail count exceeds threshold (>50% of runs)

    Warning Conditions (status=WARN):
        - Last run status is WARN
        - Any warn_count > 0
        - Drift status changed (but not regressed)

    Args:
        history: History ledger from build_ht_replay_history()
        drift: Drift comparison from compare_ht_history()

    Returns:
        Dict containing:
            - release_ok: bool - True if release can proceed
            - status: "OK" | "WARN" | "BLOCK"
            - blocking_reasons: list[str] - Reasons for blocking (if any)
            - warning_reasons: list[str] - Reasons for warning (if any)
    """
    blocking_reasons: List[str] = []
    warning_reasons: List[str] = []

    # Get history metrics
    run_entries = history.get("run_entries", [])
    total_runs = history.get("total_runs", 0)
    fail_count = history.get("fail_count", 0)
    warn_count = history.get("warn_count", 0)
    recurrent_failures = history.get("recurrent_failures", {})

    # Get last status
    last_status = run_entries[-1].get("ht_replay_status", "UNKNOWN") if run_entries else "UNKNOWN"

    # Get drift metrics
    drift_status = drift.get("status", "UNKNOWN")

    # === BLOCKING CONDITIONS ===

    # Block 1: Last run is FAIL
    if last_status == HtReplayStatus.FAIL.value:
        blocking_reasons.append(f"Last run status is FAIL")

    # Block 2: Drift is REGRESSED
    if drift_status == DriftStatus.REGRESSED.value:
        blocking_reasons.append(f"Drift status is REGRESSED (health degraded)")

    # Block 3: Recurrent critical failures
    recurrent_critical = [
        inv_id for inv_id in recurrent_failures.keys()
        if inv_id in CRITICAL_INVARIANTS
    ]
    if recurrent_critical:
        blocking_reasons.append(
            f"Recurrent critical failures: {', '.join(recurrent_critical)}"
        )

    # Block 4: High fail rate (>50%)
    if total_runs > 0:
        fail_rate = fail_count / total_runs
        if fail_rate > 0.5:
            blocking_reasons.append(
                f"High fail rate: {fail_rate:.1%} ({fail_count}/{total_runs} runs failed)"
            )

    # === WARNING CONDITIONS ===

    # Warn 1: Last run is WARN
    if last_status == HtReplayStatus.WARN.value:
        warning_reasons.append("Last run status is WARN")

    # Warn 2: Any warnings in history
    if warn_count > 0 and not blocking_reasons:
        warning_reasons.append(f"{warn_count} warning(s) in history")

    # Warn 3: Drift status changed (but not regressed)
    if drift.get("details", {}).get("latest_status_changed", False):
        if drift_status != DriftStatus.REGRESSED.value:
            warning_reasons.append("Status changed since last evaluation")

    # Warn 4: Any recurrent high-severity failures
    recurrent_high = [
        inv_id for inv_id in recurrent_failures.keys()
        if inv_id in HIGH_INVARIANTS
    ]
    if recurrent_high and not blocking_reasons:
        warning_reasons.append(
            f"Recurrent high-severity failures: {', '.join(recurrent_high)}"
        )

    # === DETERMINE STATUS ===
    if blocking_reasons:
        status = ReleaseStatus.BLOCK
        release_ok = False
    elif warning_reasons:
        status = ReleaseStatus.WARN
        release_ok = True  # Warnings don't block release
    else:
        status = ReleaseStatus.OK
        release_ok = True

    return {
        "release_ok": release_ok,
        "status": status.value,
        "blocking_reasons": blocking_reasons,
        "warning_reasons": warning_reasons,
        "last_status": last_status,
        "drift_status": drift_status,
        "total_runs": total_runs,
        "fail_count": fail_count
    }


# ==============================================================================
# Phase IV: MAAS Hₜ Summary
# ==============================================================================

class MaasStatus(Enum):
    """
    MAAS integration status for Hₜ replay.

    Status Classification:
        OK        - Hₜ replay is healthy, no action needed
        ATTENTION - Minor issues, monitoring recommended
        BLOCK     - Critical issues, intervention required
    """
    OK = "OK"
    ATTENTION = "ATTENTION"
    BLOCK = "BLOCK"


def summarize_ht_replay_for_maas(
    history: Dict[str, Any],
    drift: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate Hₜ replay summary for MAAS (Monitoring & Alerting System).

    This summary provides a minimal signal for automated monitoring systems,
    with clear status indicators and actionable information.

    Status Classification:
        OK        - All healthy: last run OK, no drift regression, no recurrent critical
        ATTENTION - Needs monitoring: WARN status, minor drift, or high-severity recurrence
        BLOCK     - Requires action: FAIL status, REGRESSED drift, or critical recurrence

    Args:
        history: History ledger from build_ht_replay_history()
        drift: Drift comparison from compare_ht_history()

    Returns:
        Dict containing:
            - ht_replay_admissible: bool - True if replay is acceptable
            - recurrent_critical_failures: list[str] - Critical invariants failing repeatedly
            - status: "OK" | "ATTENTION" | "BLOCK"
            - drift_status: str - Current drift classification
            - last_status: str - Last run status
            - health_score: int - Aggregate health score
    """
    # Get history metrics
    run_entries = history.get("run_entries", [])
    recurrent_failures = history.get("recurrent_failures", {})

    # Get last status
    last_status = run_entries[-1].get("ht_replay_status", "UNKNOWN") if run_entries else "UNKNOWN"

    # Get drift status
    drift_status = drift.get("status", "UNKNOWN")

    # Identify recurrent critical failures
    recurrent_critical = [
        inv_id for inv_id in recurrent_failures.keys()
        if inv_id in CRITICAL_INVARIANTS
    ]

    # Compute health score
    health_score = (
        history.get("ok_count", 0) -
        history.get("fail_count", 0)
    )

    # === DETERMINE STATUS ===

    # BLOCK conditions
    if (last_status == HtReplayStatus.FAIL.value or
        drift_status == DriftStatus.REGRESSED.value or
        len(recurrent_critical) > 0):
        status = MaasStatus.BLOCK
        ht_replay_admissible = False

    # ATTENTION conditions
    elif (last_status == HtReplayStatus.WARN.value or
          history.get("warn_count", 0) > 0 or
          any(inv_id in HIGH_INVARIANTS for inv_id in recurrent_failures.keys())):
        status = MaasStatus.ATTENTION
        ht_replay_admissible = True  # Still admissible but needs attention

    # OK condition
    else:
        status = MaasStatus.OK
        ht_replay_admissible = True

    return {
        "ht_replay_admissible": ht_replay_admissible,
        "recurrent_critical_failures": recurrent_critical,
        "status": status.value,
        "drift_status": drift_status,
        "last_status": last_status,
        "health_score": health_score,
        "total_runs": history.get("total_runs", 0),
        "ok_count": history.get("ok_count", 0),
        "fail_count": history.get("fail_count", 0)
    }


# ==============================================================================
# Phase IV: Director Hₜ Panel
# ==============================================================================

class StatusLight(Enum):
    """
    Director dashboard status light colors.

    Color Classification:
        GREEN  - All healthy, no issues
        YELLOW - Warning state, attention needed
        RED    - Critical state, action required
        GRAY   - Unknown or no data
    """
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"
    GRAY = "GRAY"


def build_ht_replay_director_panel(
    global_health_ht: Dict[str, Any],
    drift: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Build Hₜ replay panel for Director dashboard.

    This function generates a summary tile for the Director dashboard,
    providing at-a-glance status information for Hₜ replay health.

    Status Light Colors:
        GREEN  - ht_ok is True and drift is STABLE or IMPROVED
        YELLOW - ht_ok is True but drift status changed, or WARN state
        RED    - ht_ok is False or drift is REGRESSED
        GRAY   - No data available

    Args:
        global_health_ht: Global health summary from summarize_ht_for_global_health()
        drift: Drift comparison from compare_ht_history()

    Returns:
        Dict containing:
            - status_light: "GREEN" | "YELLOW" | "RED" | "GRAY"
            - last_status: str - Last run status
            - critical_fail_runs_count: int - Number of runs with critical failures
            - headline: str - Neutral sentence summarizing Hₜ replay posture
            - health_score: int - Aggregate health score
            - drift_status: str - Current drift classification
    """
    # Extract metrics
    ht_ok = global_health_ht.get("ht_ok", False)
    last_status = global_health_ht.get("last_status", "UNKNOWN")
    critical_fail_runs = global_health_ht.get("critical_fail_runs", [])
    critical_fail_runs_count = len(critical_fail_runs)
    health_score = global_health_ht.get("health_score", 0)
    total_runs = global_health_ht.get("total_runs", 0)
    recurrent_critical = global_health_ht.get("recurrent_critical_failures", [])

    drift_status = drift.get("status", "UNKNOWN")

    # === DETERMINE STATUS LIGHT ===

    if total_runs == 0:
        status_light = StatusLight.GRAY
    elif not ht_ok or drift_status == DriftStatus.REGRESSED.value:
        status_light = StatusLight.RED
    elif (last_status == HtReplayStatus.WARN.value or
          drift.get("details", {}).get("latest_status_changed", False)):
        status_light = StatusLight.YELLOW
    else:
        status_light = StatusLight.GREEN

    # === GENERATE HEADLINE ===

    if total_runs == 0:
        headline = "No Hₜ replay data available."
    elif status_light == StatusLight.GREEN:
        headline = f"Hₜ replay healthy: {total_runs} runs, all determinism checks passing."
    elif status_light == StatusLight.YELLOW:
        if last_status == HtReplayStatus.WARN.value:
            headline = f"Hₜ replay attention needed: last run has warnings."
        else:
            headline = f"Hₜ replay status changed: monitoring recommended."
    elif status_light == StatusLight.RED:
        if len(recurrent_critical) > 0:
            headline = f"Hₜ replay critical: {len(recurrent_critical)} invariant(s) failing repeatedly."
        elif drift_status == DriftStatus.REGRESSED.value:
            headline = f"Hₜ replay regressed: health degraded from previous evaluation."
        elif critical_fail_runs_count > 0:
            headline = f"Hₜ replay failing: {critical_fail_runs_count} run(s) with critical failures."
        else:
            headline = f"Hₜ replay unhealthy: last run status is {last_status}."
    else:
        headline = "Hₜ replay status unknown."

    return {
        "status_light": status_light.value,
        "last_status": last_status,
        "critical_fail_runs_count": critical_fail_runs_count,
        "headline": headline,
        "health_score": health_score,
        "drift_status": drift_status,
        "total_runs": total_runs,
        "ok_count": global_health_ht.get("ok_count", 0),
        "fail_count": global_health_ht.get("fail_count", 0)
    }


# ==============================================================================
# Phase V: Hₜ as Canonical Replay Invariant
# ==============================================================================

class AlignmentStatus(Enum):
    """
    Alignment status between Hₜ replay and replay radar.

    Status Classification:
        ALIGNED   - Hₜ and radar agree on replay health
        TENSION   - Minor disagreement (one WARN, other OK)
        DIVERGENT - Critical disagreement (one BLOCK/FAIL, other OK)
    """
    ALIGNED = "ALIGNED"
    TENSION = "TENSION"
    DIVERGENT = "DIVERGENT"


def build_ht_replay_governance_view(
    history: Dict[str, Any],
    drift: Dict[str, Any],
    replay_radar: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Build Hₜ replay governance view bridging HT status with replay radar.

    This function creates a unified governance view that compares Hₜ replay
    status with replay radar status, identifying alignment or divergence
    between the two systems.

    Alignment Classification:
        ALIGNED   - Both systems agree (both OK, both WARN, or both BLOCK/FAIL)
        TENSION   - Minor disagreement (e.g., HT WARN vs radar OK)
        DIVERGENT - Critical disagreement (e.g., HT BLOCK vs radar OK)

    Args:
        history: History ledger from build_ht_replay_history()
        drift: Drift comparison from compare_ht_history()
        replay_radar: Replay radar status dict with 'status' field
                     Expected values: "OK", "WARN", "BLOCK", "FAIL", "ATTENTION"

    Returns:
        Dict containing:
            - ht_ok: bool - Hₜ replay is healthy
            - replay_radar_status: str - Radar status
            - alignment_status: "ALIGNED" | "TENSION" | "DIVERGENT"
            - blocking_fingerprints: list[str] - Reasons for divergence
            - ht_status: str - Current Hₜ status
            - drift_status: str - Current drift status
    """
    # Get Hₜ status from MAAS summary
    maas_summary = summarize_ht_replay_for_maas(history, drift)
    ht_status = maas_summary["status"]
    ht_ok = maas_summary["ht_replay_admissible"]

    # Get radar status (normalize to comparable values)
    radar_status = replay_radar.get("status", "UNKNOWN")

    # Normalize radar status for comparison
    # Map ATTENTION -> WARN, FAIL -> BLOCK for alignment comparison
    radar_normalized = radar_status
    if radar_status == "ATTENTION":
        radar_normalized = "WARN"
    elif radar_status == "FAIL":
        radar_normalized = "BLOCK"

    # Map HT status for comparison
    # ATTENTION -> WARN for HT
    ht_normalized = ht_status
    if ht_status == "ATTENTION":
        ht_normalized = "WARN"

    # Determine alignment status
    blocking_fingerprints: List[str] = []

    # Check for DIVERGENT cases (critical disagreement)
    if ht_normalized == "BLOCK" and radar_normalized == "OK":
        alignment_status = AlignmentStatus.DIVERGENT
        blocking_fingerprints.append(
            f"HT reports BLOCK while radar reports OK"
        )
        # Add specific HT blocking reasons
        recurrent_critical = maas_summary.get("recurrent_critical_failures", [])
        if recurrent_critical:
            blocking_fingerprints.append(
                f"HT critical failures: {', '.join(recurrent_critical)}"
            )
        if drift.get("status") == DriftStatus.REGRESSED.value:
            blocking_fingerprints.append("HT drift status is REGRESSED")

    elif ht_normalized == "OK" and radar_normalized == "BLOCK":
        alignment_status = AlignmentStatus.DIVERGENT
        blocking_fingerprints.append(
            f"Radar reports BLOCK while HT reports OK"
        )

    # Check for TENSION cases (minor disagreement)
    elif ht_normalized == "WARN" and radar_normalized == "OK":
        alignment_status = AlignmentStatus.TENSION
        blocking_fingerprints.append("HT reports WARN while radar reports OK")

    elif ht_normalized == "OK" and radar_normalized == "WARN":
        alignment_status = AlignmentStatus.TENSION
        blocking_fingerprints.append("Radar reports WARN while HT reports OK")

    elif ht_normalized == "BLOCK" and radar_normalized == "WARN":
        alignment_status = AlignmentStatus.TENSION
        blocking_fingerprints.append("HT reports BLOCK while radar reports WARN")

    elif ht_normalized == "WARN" and radar_normalized == "BLOCK":
        alignment_status = AlignmentStatus.TENSION
        blocking_fingerprints.append("Radar reports BLOCK while HT reports WARN")

    # ALIGNED cases
    else:
        alignment_status = AlignmentStatus.ALIGNED

    return {
        "ht_ok": ht_ok,
        "replay_radar_status": radar_status,
        "alignment_status": alignment_status.value,
        "blocking_fingerprints": blocking_fingerprints,
        "ht_status": ht_status,
        "drift_status": drift.get("status", "UNKNOWN"),
        "health_score": maas_summary.get("health_score", 0),
        "recurrent_critical_failures": maas_summary.get("recurrent_critical_failures", [])
    }


# ==============================================================================
# Phase V: Global Console Adapter
# ==============================================================================

def summarize_ht_for_global_console(
    global_health_ht: Dict[str, Any],
    drift: Dict[str, Any],
    governance_view: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate Hₜ summary for global console display.

    This function creates a unified summary suitable for the global console,
    combining director panel information with governance alignment status.

    Args:
        global_health_ht: Global health summary from summarize_ht_for_global_health()
        drift: Drift comparison from compare_ht_history()
        governance_view: Governance view from build_ht_replay_governance_view()

    Returns:
        Dict containing:
            - status_light: str - GREEN/YELLOW/RED/GRAY
            - ht_ok: bool - Hₜ replay is healthy
            - drift_status: str - Current drift status
            - headline: str - Summary headline
            - alignment_status: str - Alignment with radar
            - replay_radar_status: str - Radar status
            - blocking_fingerprints: list[str] - Divergence reasons
    """
    # Get director panel for status light and headline
    director_panel = build_ht_replay_director_panel(global_health_ht, drift)

    # Override headline if there's alignment tension or divergence
    headline = director_panel["headline"]
    alignment_status = governance_view.get("alignment_status", "ALIGNED")

    if alignment_status == AlignmentStatus.DIVERGENT.value:
        # Upgrade headline to reflect divergence
        fingerprints = governance_view.get("blocking_fingerprints", [])
        if fingerprints:
            headline = f"Hₜ-radar divergence detected: {fingerprints[0]}"
    elif alignment_status == AlignmentStatus.TENSION.value:
        # Add tension note to headline
        headline = f"{headline} (radar tension detected)"

    # Override status light if divergent
    status_light = director_panel["status_light"]
    if alignment_status == AlignmentStatus.DIVERGENT.value:
        status_light = StatusLight.RED.value

    return {
        "status_light": status_light,
        "ht_ok": governance_view.get("ht_ok", False),
        "drift_status": drift.get("status", "UNKNOWN"),
        "headline": headline,
        "alignment_status": alignment_status,
        "replay_radar_status": governance_view.get("replay_radar_status", "UNKNOWN"),
        "blocking_fingerprints": governance_view.get("blocking_fingerprints", []),
        "health_score": global_health_ht.get("health_score", 0),
        "total_runs": global_health_ht.get("total_runs", 0),
        "critical_fail_runs_count": len(global_health_ht.get("critical_fail_runs", []))
    }


# ==============================================================================
# Phase V: Evidence Pack HT Tile
# ==============================================================================

def build_ht_evidence_pack_tile(
    history: Dict[str, Any],
    drift: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Build a JSON-safe Hₜ replay summary for inclusion in evidence packs.

    This function creates a minimal, JSON-serializable tile suitable for
    embedding in evidence packs, attestations, and audit logs.

    Args:
        history: History ledger from build_ht_replay_history()
        drift: Optional drift comparison from compare_ht_history()

    Returns:
        Dict containing (all JSON-safe):
            - ht_replay_ok: bool - Hₜ replay is healthy
            - critical_fail_runs: list[str] - Run IDs with critical failures
            - recurrent_critical_invariants: list[str] - Invariants failing repeatedly
            - last_status: str - Most recent run status
            - health_score: int - Aggregate health score
            - total_runs: int - Total number of runs
            - drift_status: str - Drift classification (if provided)
            - generated_utc: str - Timestamp of tile generation
    """
    # Get global health summary
    global_health = summarize_ht_for_global_health(history)

    # Extract critical fail runs from history entries
    run_entries = history.get("run_entries", [])
    critical_fail_runs = [
        entry.get("replay_id", "unknown")
        for entry in run_entries
        if not entry.get("all_critical_pass", True)
    ]

    # Get recurrent critical failures
    recurrent_failures = history.get("recurrent_failures", {})
    recurrent_critical_invariants = [
        inv_id for inv_id in recurrent_failures.keys()
        if inv_id in CRITICAL_INVARIANTS
    ]

    # Determine if HT replay is OK
    ht_replay_ok = (
        global_health.get("ht_ok", False) and
        len(recurrent_critical_invariants) == 0
    )

    # Get drift status if provided
    drift_status = "NOT_EVALUATED"
    if drift:
        drift_status = drift.get("status", "UNKNOWN")

    return {
        "ht_replay_ok": ht_replay_ok,
        "critical_fail_runs": critical_fail_runs,
        "recurrent_critical_invariants": recurrent_critical_invariants,
        "last_status": global_health.get("last_status", "UNKNOWN"),
        "health_score": global_health.get("health_score", 0),
        "total_runs": history.get("total_runs", 0),
        "ok_count": history.get("ok_count", 0),
        "fail_count": history.get("fail_count", 0),
        "drift_status": drift_status,
        "generated_utc": datetime.utcnow().isoformat() + "Z"
    }


# ==============================================================================
# Phase V: Governance Signal Layer for Global Alignment
# ==============================================================================

class GovernanceSignal(Enum):
    """
    Governance signal for global alignment view integration.

    These signals map directly to CLAUDE I's build_global_alignment_view
    expected input format for subsystem health signals.

    Signal Semantics:
        OK    - Subsystem is healthy, no action required
        WARN  - Subsystem has minor issues, requires attention
        BLOCK - Subsystem has critical issues, blocks release/promotion
    """
    OK = "OK"
    WARN = "WARN"
    BLOCK = "BLOCK"


def to_governance_signal_for_ht(
    governance_view: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Convert Hₜ governance view to a governance signal for global alignment.

    This function transforms the Hₜ replay governance view into a standardized
    governance signal that can be directly consumed by CLAUDE I's
    build_global_alignment_view function.

    Alignment to Signal Mapping:
        DIVERGENT → BLOCK  (Critical disagreement between HT and radar)
        TENSION   → WARN   (Minor disagreement, requires attention)
        ALIGNED   → OK     (Systems agree, healthy state)

    Args:
        governance_view: Governance view from build_ht_replay_governance_view()

    Returns:
        Dict containing:
            - signal: str - "OK" | "WARN" | "BLOCK"
            - source: str - "ht_replay" (identifies signal source)
            - alignment_status: str - Original alignment status
            - blocking_fingerprints: list[str] - Reasons for non-OK signal
            - ht_ok: bool - Whether HT replay is healthy
            - details: dict - Additional context for downstream consumers
    """
    alignment_status = governance_view.get("alignment_status", "ALIGNED")

    # Map alignment status to governance signal
    if alignment_status == AlignmentStatus.DIVERGENT.value:
        signal = GovernanceSignal.BLOCK
    elif alignment_status == AlignmentStatus.TENSION.value:
        signal = GovernanceSignal.WARN
    else:
        # ALIGNED or unknown defaults to OK
        signal = GovernanceSignal.OK

    # Extract blocking fingerprints for non-OK signals
    blocking_fingerprints = governance_view.get("blocking_fingerprints", [])

    # Build details for downstream consumers
    details = {
        "ht_status": governance_view.get("ht_status", "UNKNOWN"),
        "radar_status": governance_view.get("replay_radar_status", "UNKNOWN"),
        "drift_status": governance_view.get("drift_status", "UNKNOWN"),
        "health_score": governance_view.get("health_score", 0),
        "recurrent_critical_failures": governance_view.get(
            "recurrent_critical_failures", []
        )
    }

    return {
        "signal": signal.value,
        "source": "ht_replay",
        "alignment_status": alignment_status,
        "blocking_fingerprints": blocking_fingerprints,
        "ht_ok": governance_view.get("ht_ok", False),
        "details": details
    }


def build_ht_governance_signal_from_history(
    history: Dict[str, Any],
    drift: Dict[str, Any],
    replay_radar: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Build Hₜ governance signal directly from history and radar data.

    This is a convenience function that combines build_ht_replay_governance_view
    and to_governance_signal_for_ht into a single call for simpler integration.

    Args:
        history: History ledger from build_ht_replay_history()
        drift: Drift comparison from compare_ht_history()
        replay_radar: Replay radar status dict

    Returns:
        Governance signal dict (same as to_governance_signal_for_ht output)
    """
    governance_view = build_ht_replay_governance_view(history, drift, replay_radar)
    return to_governance_signal_for_ht(governance_view)
