"""
PHASE II/III/IV/V/VI — NOT USED FOR FO/PHASE I RUNS

Slice Hash Ledger: Binding Construction, Lineage Verification, and Drift Intelligence.

This module implements the slice_hash_binding mechanics defined in
docs/SLICE_HASH_EXECUTION_BINDING.md. It provides:

Phase II:
1. build_slice_hash_binding() - Construct binding objects for prereg/manifest
2. compute_formula_pool_hash() - Compute digest of formula pool entries
3. compute_slice_config_hash() - Compute canonical slice configuration hash
4. Reconciliation error codes (SHD-*, MHM-*)

Phase III:
5. build_slice_identity_ledger() - Track slice lifecycle with first/last appearance
6. compute_slice_drift_signature() - Short hash summarizing drift between bindings
7. summarize_slice_identity_for_global_health() - Governance health summary

Phase IV:
8. build_slice_identity_curriculum_view() - Curriculum-slice identity alignment
9. evaluate_slice_identity_for_evidence() - Evidence pack identity guard
10. build_slice_identity_director_panel() - Director-level dashboard summary

Phase V:
11. build_slice_identity_drift_view() - Curriculum drift × identity drift coupling
12. summarize_slice_identity_for_global_console() - Global health adapter for console
13. get_slice_identity_summary_for_evidence() - Compact JSON-safe identity per slice

Phase VI:
14. build_slice_identity_console_tile() - Console tile for global dashboard
15. to_governance_signal_for_slice_identity() - Governance signal for CLAUDE I

All hash computations use the canonical pipeline:
    hash(s) = SHA256(DOMAIN_STMT || canonical_bytes(normalize(s)))

Author: Claude E — Slice Hash Identity & Lineage Sentinel (Phase II/III/IV/V/VI)
Date: 2025-12-06
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from normalization.canon import normalize
from substrate.crypto.hashing import hash_statement, DOMAIN_STMT, sha256_hex


# =============================================================================
# Error Code Enumerations
# =============================================================================

class SliceHashDriftError(Enum):
    """
    SHD-* Error Codes: Slice Hash Drift (ledger-level).

    These errors indicate the ledger itself has drifted due to
    normalization changes, encoding changes, or domain tag changes.
    """
    SHD_001_LEDGER_FORMULA_DRIFT = "SHD-001"
    SHD_002_LEDGER_POOL_DRIFT = "SHD-002"
    SHD_003_LEDGER_ENTRY_MISSING = "SHD-003"
    SHD_004_NORMALIZATION_CHANGE = "SHD-004"


class ManifestHashMismatchError(Enum):
    """
    MHM-* Error Codes: Manifest Hash Mismatch (binding-level).

    These errors indicate the manifest disagrees with preregistration or ledger.
    """
    MHM_001_CONFIG_HASH_MISMATCH = "MHM-001"
    MHM_002_POOL_HASH_MISMATCH = "MHM-002"
    MHM_003_LEDGER_ID_MISMATCH = "MHM-003"
    MHM_004_FROZEN_AT_VIOLATION = "MHM-004"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SliceHashBinding:
    """
    The canonical slice_hash_binding object for embedding in prereg/manifest.

    See docs/SLICE_HASH_EXECUTION_BINDING.md §2.1 for schema definition.
    """
    # Identity fields (required)
    slice_name: str
    slice_config_hash: str
    ledger_entry_id: str

    # Temporal fields (required)
    frozen_at: str
    frozen_by: str

    # Provenance fields (required)
    config_source: str
    config_version: str

    # Formula pool digest (required)
    formula_pool_hash: str
    formula_count: int
    target_count: int

    # Verification metadata (optional)
    verification: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON/YAML serialization."""
        result = {
            "slice_name": self.slice_name,
            "slice_config_hash": self.slice_config_hash,
            "ledger_entry_id": self.ledger_entry_id,
            "frozen_at": self.frozen_at,
            "frozen_by": self.frozen_by,
            "config_source": self.config_source,
            "config_version": self.config_version,
            "formula_pool_hash": self.formula_pool_hash,
            "formula_count": self.formula_count,
            "target_count": self.target_count,
        }
        if self.verification is not None:
            result["verification"] = self.verification
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SliceHashBinding":
        """Reconstruct from dictionary."""
        return cls(
            slice_name=data["slice_name"],
            slice_config_hash=data["slice_config_hash"],
            ledger_entry_id=data["ledger_entry_id"],
            frozen_at=data["frozen_at"],
            frozen_by=data["frozen_by"],
            config_source=data["config_source"],
            config_version=data["config_version"],
            formula_pool_hash=data["formula_pool_hash"],
            formula_count=data["formula_count"],
            target_count=data["target_count"],
            verification=data.get("verification"),
        )


@dataclass
class ReconciliationResult:
    """Result of a hash reconciliation check."""
    passed: bool
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    expected: Optional[str] = None
    actual: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = {"passed": self.passed}
        if self.error_code:
            result["error_code"] = self.error_code
        if self.error_message:
            result["error_message"] = self.error_message
        if self.expected:
            result["expected"] = self.expected
        if self.actual:
            result["actual"] = self.actual
        if self.context:
            result["context"] = self.context
        return result


# =============================================================================
# Core Hash Computation Functions
# =============================================================================

def compute_formula_pool_hash(formula_pool_entries: List[Any]) -> str:
    """
    Compute a digest of the formula pool for quick integrity checks.

    Uses the concatenation of all (normalized_formula, hash) pairs,
    sorted by hash for determinism.

    Handles both formats:
    - Dict entries: {"formula": "p->q", "hash": "abc123..."}
    - String entries: "p->q" (hash computed on the fly)

    Args:
        formula_pool_entries: List of entries (dicts or strings)

    Returns:
        64-character hex SHA256 digest
    """
    pairs = []
    for entry in formula_pool_entries:
        if isinstance(entry, dict):
            # Dict entry with formula and hash
            formula = entry.get("formula", "")
            h = entry.get("hash", "")
            if formula and h:
                normalized = normalize(formula)
                pairs.append(f"{normalized}:{h}")
            elif formula:
                # Dict with formula but no hash - compute it
                normalized = normalize(formula)
                h = hash_statement(formula)
                pairs.append(f"{normalized}:{h}")
        elif isinstance(entry, str):
            # Plain string entry - compute hash
            formula = entry
            normalized = normalize(formula)
            h = hash_statement(formula)
            pairs.append(f"{normalized}:{h}")

    # Sort by hash for determinism
    pairs.sort(key=lambda x: x.split(':')[-1])

    payload = "\n".join(pairs).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def compute_slice_config_hash(slice_config: Dict[str, Any]) -> str:
    """
    Compute the canonical hash of a slice configuration.

    The slice config is serialized using RFC 8785-inspired rules:
    - Keys sorted lexicographically
    - No insignificant whitespace
    - Consistent separators

    Args:
        slice_config: Slice configuration dictionary

    Returns:
        64-character hex SHA256 digest
    """
    # Extract only the hashable fields (exclude runtime metadata)
    hashable_fields = {
        "name": slice_config.get("name"),
        "params": slice_config.get("params", slice_config.get("parameters", {})),
        "success_metric": slice_config.get("success_metric", {}),
        "formula_pool_entries": slice_config.get("formula_pool_entries", []),
        "gates": slice_config.get("gates", {}),
        "budget": slice_config.get("budget", {}),
    }

    # Remove None values for cleaner hash
    hashable_fields = {k: v for k, v in hashable_fields.items() if v is not None}

    # Canonical JSON serialization
    canonical = json.dumps(hashable_fields, sort_keys=True, separators=(',', ':'))

    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def count_target_formulas(slice_config: Dict[str, Any]) -> int:
    """
    Count the number of target formulas in a slice configuration.

    Counts entries with role containing 'target' or referenced in
    success_metric target_hashes/chain_target_hash/required_goal_hashes.
    """
    count = 0

    # Count entries with target role
    for entry in slice_config.get("formula_pool_entries", []):
        role = entry.get("role", "")
        if "target" in role.lower():
            count += 1

    # Also count referenced target hashes
    success_metric = slice_config.get("success_metric", {})
    target_hashes = set()

    for h in success_metric.get("target_hashes", []):
        target_hashes.add(h)

    chain_target = success_metric.get("chain_target_hash")
    if chain_target:
        target_hashes.add(chain_target)

    for h in success_metric.get("required_goal_hashes", []):
        target_hashes.add(h)

    # Return max of role-based count or hash-based count
    return max(count, len(target_hashes))


def generate_ledger_entry_id(slice_name: str, config_version: str) -> str:
    """
    Generate a canonical ledger entry ID for a slice.

    Format: SLICE-<slice_name>-v<version>-<date>
    """
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    version_clean = config_version.replace(".", "-")
    return f"SLICE-{slice_name}-v{version_clean}-{date_str}"


# =============================================================================
# Binding Construction
# =============================================================================

def build_slice_hash_binding(
    slice_name: str,
    curriculum_config: Dict[str, Any],
    *,
    config_source: str = "config/curriculum_uplift_phase2.yaml",
    frozen_by: str = "Claude E",
    ledger_entry_id: Optional[str] = None,
    verification_result: Optional[Dict[str, Any]] = None,
) -> SliceHashBinding:
    """
    Construct a slice_hash_binding object matching the schema in §2 of
    docs/SLICE_HASH_EXECUTION_BINDING.md.

    Args:
        slice_name: Name of the slice to bind
        curriculum_config: Full curriculum configuration (with 'slices' or 'systems' key)
        config_source: Path to the configuration file
        frozen_by: Agent identifier creating the binding
        ledger_entry_id: Optional explicit ledger entry ID (auto-generated if None)
        verification_result: Optional verification metadata to include

    Returns:
        SliceHashBinding object ready for embedding in prereg/manifest

    Raises:
        ValueError: If slice not found in configuration
    """
    # Find the slice configuration
    slice_config = None

    # Handle both flat 'slices' dict and nested 'systems' structure
    if "slices" in curriculum_config:
        slices = curriculum_config["slices"]
        if isinstance(slices, dict):
            slice_config = slices.get(slice_name)
            if slice_config:
                slice_config["name"] = slice_name
        elif isinstance(slices, list):
            for s in slices:
                if s.get("name") == slice_name:
                    slice_config = s
                    break
    elif "systems" in curriculum_config:
        for system in curriculum_config.get("systems", []):
            for s in system.get("slices", []):
                if s.get("name") == slice_name:
                    slice_config = s
                    break
            if slice_config:
                break

    if slice_config is None:
        raise ValueError(f"Slice '{slice_name}' not found in curriculum configuration")

    # Get config version
    config_version = str(curriculum_config.get("version", "unknown"))

    # Compute hashes
    slice_config_hash = compute_slice_config_hash(slice_config)

    formula_pool_entries = slice_config.get("formula_pool_entries", [])
    formula_pool_hash = compute_formula_pool_hash(formula_pool_entries)
    formula_count = len(formula_pool_entries)
    target_count = count_target_formulas(slice_config)

    # Generate ledger entry ID if not provided
    if ledger_entry_id is None:
        ledger_entry_id = generate_ledger_entry_id(slice_name, config_version)

    # Frozen timestamp
    frozen_at = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')

    return SliceHashBinding(
        slice_name=slice_name,
        slice_config_hash=slice_config_hash,
        ledger_entry_id=ledger_entry_id,
        frozen_at=frozen_at,
        frozen_by=frozen_by,
        config_source=config_source,
        config_version=config_version,
        formula_pool_hash=formula_pool_hash,
        formula_count=formula_count,
        target_count=target_count,
        verification=verification_result,
    )


# =============================================================================
# Lineage Reconciliation
# =============================================================================

def reconcile_formula_hash(
    formula: str,
    expected_hash: str,
) -> ReconciliationResult:
    """
    Verify that a formula's computed hash matches the expected ledger hash.

    Args:
        formula: The formula text
        expected_hash: The expected hash from the ledger

    Returns:
        ReconciliationResult with pass/fail status and error code if failed
    """
    actual_hash = hash_statement(formula)

    if actual_hash == expected_hash:
        return ReconciliationResult(passed=True)

    return ReconciliationResult(
        passed=False,
        error_code=SliceHashDriftError.SHD_001_LEDGER_FORMULA_DRIFT.value,
        error_message=f"Formula hash drift detected for '{formula[:50]}...'",
        expected=expected_hash,
        actual=actual_hash,
        context={"formula": formula, "normalized": normalize(formula)},
    )


def reconcile_pool_hash(
    formula_pool_entries: List[Dict[str, Any]],
    expected_pool_hash: str,
) -> ReconciliationResult:
    """
    Verify that the formula pool hash matches the expected value.

    Args:
        formula_pool_entries: List of formula pool entries
        expected_pool_hash: The expected pool hash

    Returns:
        ReconciliationResult with pass/fail status and error code if failed
    """
    actual_pool_hash = compute_formula_pool_hash(formula_pool_entries)

    if actual_pool_hash == expected_pool_hash:
        return ReconciliationResult(passed=True)

    return ReconciliationResult(
        passed=False,
        error_code=SliceHashDriftError.SHD_002_LEDGER_POOL_DRIFT.value,
        error_message="Formula pool hash drift detected",
        expected=expected_pool_hash,
        actual=actual_pool_hash,
        context={"entry_count": len(formula_pool_entries)},
    )


def reconcile_config_hash(
    slice_config: Dict[str, Any],
    expected_config_hash: str,
) -> ReconciliationResult:
    """
    Verify that the slice config hash matches the expected value.

    Args:
        slice_config: The slice configuration dictionary
        expected_config_hash: The expected config hash from prereg

    Returns:
        ReconciliationResult with pass/fail status and error code if failed
    """
    actual_config_hash = compute_slice_config_hash(slice_config)

    if actual_config_hash == expected_config_hash:
        return ReconciliationResult(passed=True)

    return ReconciliationResult(
        passed=False,
        error_code=ManifestHashMismatchError.MHM_001_CONFIG_HASH_MISMATCH.value,
        error_message="Slice config hash mismatch (config modified after prereg)",
        expected=expected_config_hash,
        actual=actual_config_hash,
        context={"slice_name": slice_config.get("name")},
    )


def reconcile_binding_against_prereg(
    manifest_binding: SliceHashBinding,
    prereg_binding: SliceHashBinding,
) -> List[ReconciliationResult]:
    """
    Reconcile a manifest's slice_hash_binding against preregistration.

    Checks:
    1. slice_config_hash matches
    2. formula_pool_hash matches
    3. ledger_entry_id matches
    4. frozen_at is not after manifest creation

    Args:
        manifest_binding: The binding from the manifest
        prereg_binding: The binding from preregistration

    Returns:
        List of ReconciliationResults for each check
    """
    results = []

    # Check config hash
    if manifest_binding.slice_config_hash != prereg_binding.slice_config_hash:
        results.append(ReconciliationResult(
            passed=False,
            error_code=ManifestHashMismatchError.MHM_001_CONFIG_HASH_MISMATCH.value,
            error_message="Manifest config hash differs from preregistration",
            expected=prereg_binding.slice_config_hash,
            actual=manifest_binding.slice_config_hash,
        ))
    else:
        results.append(ReconciliationResult(passed=True))

    # Check pool hash
    if manifest_binding.formula_pool_hash != prereg_binding.formula_pool_hash:
        results.append(ReconciliationResult(
            passed=False,
            error_code=ManifestHashMismatchError.MHM_002_POOL_HASH_MISMATCH.value,
            error_message="Manifest pool hash differs from preregistration",
            expected=prereg_binding.formula_pool_hash,
            actual=manifest_binding.formula_pool_hash,
        ))
    else:
        results.append(ReconciliationResult(passed=True))

    # Check ledger ID
    if manifest_binding.ledger_entry_id != prereg_binding.ledger_entry_id:
        results.append(ReconciliationResult(
            passed=False,
            error_code=ManifestHashMismatchError.MHM_003_LEDGER_ID_MISMATCH.value,
            error_message="Manifest ledger entry ID differs from preregistration",
            expected=prereg_binding.ledger_entry_id,
            actual=manifest_binding.ledger_entry_id,
        ))
    else:
        results.append(ReconciliationResult(passed=True))

    return results


def reconcile_slice_integrity(
    slice_config: Dict[str, Any],
) -> List[ReconciliationResult]:
    """
    Verify all formula hashes in a slice configuration.

    Args:
        slice_config: The slice configuration dictionary

    Returns:
        List of ReconciliationResults, one per formula entry
    """
    results = []

    for entry in slice_config.get("formula_pool_entries", []):
        formula = entry.get("formula")
        expected_hash = entry.get("hash")

        if not formula or not expected_hash:
            continue

        result = reconcile_formula_hash(formula, expected_hash)
        result.context["role"] = entry.get("role", "unknown")
        results.append(result)

    return results


# =============================================================================
# Full Reconciliation Pipeline
# =============================================================================

def full_reconciliation(
    curriculum_config: Dict[str, Any],
    prereg_bindings: Dict[str, Dict[str, Any]],
    manifest_bindings: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Perform full reconciliation across ledger → curriculum → prereg → manifest.

    Args:
        curriculum_config: The curriculum configuration
        prereg_bindings: Dict mapping slice_name to prereg slice_hash_binding
        manifest_bindings: Dict mapping slice_name to manifest slice_hash_binding

    Returns:
        Full reconciliation report with all check results
    """
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        "phase": "II",
        "status": "PHASE_II_DESIGNED",
        "checks": [],
        "summary": {
            "total": 0,
            "passed": 0,
            "failed": 0,
        },
    }

    # Find all slices
    slices = {}
    if "slices" in curriculum_config:
        slices_data = curriculum_config["slices"]
        if isinstance(slices_data, dict):
            for name, config in slices_data.items():
                config["name"] = name
                slices[name] = config
        elif isinstance(slices_data, list):
            for config in slices_data:
                slices[config.get("name")] = config
    elif "systems" in curriculum_config:
        for system in curriculum_config.get("systems", []):
            for config in system.get("slices", []):
                slices[config.get("name")] = config

    # Check each slice
    for slice_name, slice_config in slices.items():
        # 1. Verify formula hashes (ledger integrity)
        formula_results = reconcile_slice_integrity(slice_config)
        for i, result in enumerate(formula_results):
            check = {
                "check_id": f"LEDGER_{slice_name}_formula_{i}",
                "check_type": "LEDGER_INTEGRITY",
                "slice_name": slice_name,
                **result.to_dict(),
            }
            report["checks"].append(check)

        # 2. Check prereg binding if exists
        if slice_name in prereg_bindings:
            prereg_binding = SliceHashBinding.from_dict(prereg_bindings[slice_name])

            # Verify config hash against current config
            config_result = reconcile_config_hash(slice_config, prereg_binding.slice_config_hash)
            report["checks"].append({
                "check_id": f"PREREG_{slice_name}_config_hash",
                "check_type": "PREREG_INTEGRITY",
                "slice_name": slice_name,
                **config_result.to_dict(),
            })

            # Verify pool hash
            pool_entries = slice_config.get("formula_pool_entries", [])
            pool_result = reconcile_pool_hash(pool_entries, prereg_binding.formula_pool_hash)
            report["checks"].append({
                "check_id": f"PREREG_{slice_name}_pool_hash",
                "check_type": "PREREG_INTEGRITY",
                "slice_name": slice_name,
                **pool_result.to_dict(),
            })

        # 3. Check manifest binding against prereg if both exist
        if slice_name in manifest_bindings and slice_name in prereg_bindings:
            manifest_binding = SliceHashBinding.from_dict(manifest_bindings[slice_name])
            prereg_binding = SliceHashBinding.from_dict(prereg_bindings[slice_name])

            binding_results = reconcile_binding_against_prereg(manifest_binding, prereg_binding)
            check_names = ["config_hash", "pool_hash", "ledger_id"]
            for i, result in enumerate(binding_results):
                report["checks"].append({
                    "check_id": f"MANIFEST_{slice_name}_{check_names[i]}",
                    "check_type": "MANIFEST_INTEGRITY",
                    "slice_name": slice_name,
                    **result.to_dict(),
                })

    # Compute summary
    report["summary"]["total"] = len(report["checks"])
    report["summary"]["passed"] = sum(1 for c in report["checks"] if c.get("passed", False))
    report["summary"]["failed"] = report["summary"]["total"] - report["summary"]["passed"]
    report["overall_status"] = "CONSISTENT" if report["summary"]["failed"] == 0 else "INCONSISTENT"

    return report


# =============================================================================
# Task 1: Slice Identity Card
# =============================================================================

@dataclass
class SliceIdentityCard:
    """
    A "slice card" representing the lifecycle of a slice across configs and manifests.

    This provides a consolidated view of a slice's identity history, including
    all prereg and manifest bindings, the latest binding, and any drift flags.
    """
    slice_name: str
    prereg_hashes: List[Dict[str, Any]]
    manifest_hashes: List[Dict[str, Any]]
    latest_binding: Optional[Dict[str, Any]]
    drift_flags: List[str]
    created_at: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slice_name": self.slice_name,
            "prereg_hashes": self.prereg_hashes,
            "manifest_hashes": self.manifest_hashes,
            "latest_binding": self.latest_binding,
            "drift_flags": self.drift_flags,
            "created_at": self.created_at,
        }


def build_slice_identity_card(
    slice_name: str,
    prereg_bindings: Dict[str, Dict[str, Any]],
    manifest_bindings: Dict[str, Dict[str, Any]],
) -> SliceIdentityCard:
    """
    Build a "slice card" showing the lifecycle of a slice across prereg and manifests.

    This provides a consolidated view useful for governance audits and
    answering questions like "what versions of this slice have been used?"

    Args:
        slice_name: Name of the slice to build card for
        prereg_bindings: Dict mapping slice_name to prereg slice_hash_binding
        manifest_bindings: Dict mapping slice_name to manifest slice_hash_binding

    Returns:
        SliceIdentityCard with lifecycle information
    """
    prereg_hashes: List[Dict[str, Any]] = []
    manifest_hashes: List[Dict[str, Any]] = []
    drift_flags: List[str] = []
    latest_binding: Optional[Dict[str, Any]] = None

    # Extract prereg binding info
    if slice_name in prereg_bindings:
        binding = prereg_bindings[slice_name]
        prereg_hashes.append({
            "config_hash": binding.get("slice_config_hash"),
            "pool_hash": binding.get("formula_pool_hash"),
            "frozen_at": binding.get("frozen_at"),
            "ledger_entry_id": binding.get("ledger_entry_id"),
        })
        latest_binding = binding

    # Extract manifest binding info
    if slice_name in manifest_bindings:
        binding = manifest_bindings[slice_name]
        manifest_hashes.append({
            "config_hash": binding.get("slice_config_hash"),
            "pool_hash": binding.get("formula_pool_hash"),
            "frozen_at": binding.get("frozen_at"),
            "ledger_entry_id": binding.get("ledger_entry_id"),
        })
        # Manifest binding is more recent
        latest_binding = binding

    # Check for drift between prereg and manifest
    if prereg_hashes and manifest_hashes:
        prereg = prereg_hashes[0]
        manifest = manifest_hashes[0]

        if prereg.get("config_hash") != manifest.get("config_hash"):
            drift_flags.append(ManifestHashMismatchError.MHM_001_CONFIG_HASH_MISMATCH.value)

        if prereg.get("pool_hash") != manifest.get("pool_hash"):
            drift_flags.append(ManifestHashMismatchError.MHM_002_POOL_HASH_MISMATCH.value)

        if prereg.get("ledger_entry_id") != manifest.get("ledger_entry_id"):
            drift_flags.append(ManifestHashMismatchError.MHM_003_LEDGER_ID_MISMATCH.value)

    return SliceIdentityCard(
        slice_name=slice_name,
        prereg_hashes=prereg_hashes,
        manifest_hashes=manifest_hashes,
        latest_binding=latest_binding,
        drift_flags=drift_flags,
        created_at=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
    )


def build_all_slice_identity_cards(
    prereg_bindings: Dict[str, Dict[str, Any]],
    manifest_bindings: Dict[str, Dict[str, Any]],
) -> Dict[str, SliceIdentityCard]:
    """
    Build identity cards for all slices found in prereg and manifest bindings.

    Args:
        prereg_bindings: Dict mapping slice_name to prereg slice_hash_binding
        manifest_bindings: Dict mapping slice_name to manifest slice_hash_binding

    Returns:
        Dict mapping slice_name to SliceIdentityCard
    """
    all_slice_names = set(prereg_bindings.keys()) | set(manifest_bindings.keys())

    cards = {}
    for slice_name in sorted(all_slice_names):
        cards[slice_name] = build_slice_identity_card(
            slice_name, prereg_bindings, manifest_bindings
        )

    return cards


# =============================================================================
# Task 2: Cross-Run Slice Hash Comparison
# =============================================================================

@dataclass
class SliceBindingComparison:
    """
    Result of comparing two slice bindings (e.g., from run A vs run B).
    """
    slice_name: str
    same_formula_pool: bool
    same_config_hash: bool
    same_target_count: bool
    same_ledger_entry_id: bool
    drift_codes: List[str]
    binding_a_summary: Dict[str, Any]
    binding_b_summary: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slice_name": self.slice_name,
            "same_formula_pool": self.same_formula_pool,
            "same_config_hash": self.same_config_hash,
            "same_target_count": self.same_target_count,
            "same_ledger_entry_id": self.same_ledger_entry_id,
            "drift_codes": self.drift_codes,
            "binding_a_summary": self.binding_a_summary,
            "binding_b_summary": self.binding_b_summary,
            "identical": self.is_identical,
        }

    @property
    def is_identical(self) -> bool:
        """True if both bindings are identical in all respects."""
        return (
            self.same_formula_pool
            and self.same_config_hash
            and self.same_target_count
            and self.same_ledger_entry_id
        )


def compare_slice_bindings(
    binding_a: Dict[str, Any],
    binding_b: Dict[str, Any],
) -> SliceBindingComparison:
    """
    Compare two slice bindings to determine if a slice's identity changed between runs.

    This answers: "Did this slice's identity change between run A and run B?"

    Args:
        binding_a: First slice_hash_binding (e.g., from run A)
        binding_b: Second slice_hash_binding (e.g., from run B)

    Returns:
        SliceBindingComparison with detailed comparison results
    """
    drift_codes: List[str] = []

    # Extract values with defaults
    slice_name_a = binding_a.get("slice_name", "unknown")
    slice_name_b = binding_b.get("slice_name", "unknown")
    slice_name = slice_name_a if slice_name_a == slice_name_b else f"{slice_name_a}/{slice_name_b}"

    pool_hash_a = binding_a.get("formula_pool_hash", "")
    pool_hash_b = binding_b.get("formula_pool_hash", "")
    same_formula_pool = pool_hash_a == pool_hash_b

    config_hash_a = binding_a.get("slice_config_hash", "")
    config_hash_b = binding_b.get("slice_config_hash", "")
    same_config_hash = config_hash_a == config_hash_b

    target_count_a = binding_a.get("target_count", 0)
    target_count_b = binding_b.get("target_count", 0)
    same_target_count = target_count_a == target_count_b

    ledger_id_a = binding_a.get("ledger_entry_id", "")
    ledger_id_b = binding_b.get("ledger_entry_id", "")
    same_ledger_entry_id = ledger_id_a == ledger_id_b

    # Assign drift codes for mismatches
    if not same_config_hash:
        drift_codes.append(ManifestHashMismatchError.MHM_001_CONFIG_HASH_MISMATCH.value)

    if not same_formula_pool:
        drift_codes.append(ManifestHashMismatchError.MHM_002_POOL_HASH_MISMATCH.value)

    if not same_ledger_entry_id:
        drift_codes.append(ManifestHashMismatchError.MHM_003_LEDGER_ID_MISMATCH.value)

    # Build summaries
    binding_a_summary = {
        "slice_name": slice_name_a,
        "config_hash": config_hash_a[:16] + "..." if len(config_hash_a) > 16 else config_hash_a,
        "pool_hash": pool_hash_a[:16] + "..." if len(pool_hash_a) > 16 else pool_hash_a,
        "target_count": target_count_a,
        "ledger_entry_id": ledger_id_a,
        "frozen_at": binding_a.get("frozen_at"),
    }

    binding_b_summary = {
        "slice_name": slice_name_b,
        "config_hash": config_hash_b[:16] + "..." if len(config_hash_b) > 16 else config_hash_b,
        "pool_hash": pool_hash_b[:16] + "..." if len(pool_hash_b) > 16 else pool_hash_b,
        "target_count": target_count_b,
        "ledger_entry_id": ledger_id_b,
        "frozen_at": binding_b.get("frozen_at"),
    }

    return SliceBindingComparison(
        slice_name=slice_name,
        same_formula_pool=same_formula_pool,
        same_config_hash=same_config_hash,
        same_target_count=same_target_count,
        same_ledger_entry_id=same_ledger_entry_id,
        drift_codes=drift_codes,
        binding_a_summary=binding_a_summary,
        binding_b_summary=binding_b_summary,
    )


# =============================================================================
# Task 3: Slice Hash Integrity Summary for MAAS
# =============================================================================

@dataclass
class SliceHashIntegritySummary:
    """
    A stable, small object for MAAS or governance to use as "slice integrity verdict".
    """
    timestamp: str
    all_slices_accounted_for: bool
    slices_in_config: List[str]
    slices_in_prereg: List[str]
    slices_in_manifest: List[str]
    slices_missing_prereg: List[str]
    slices_missing_manifest: List[str]
    slices_with_drift: List[str]
    error_codes_by_slice: Dict[str, List[str]]
    overall_verdict: str  # "PASS", "FAIL", "WARN"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "all_slices_accounted_for": self.all_slices_accounted_for,
            "slices_in_config": self.slices_in_config,
            "slices_in_prereg": self.slices_in_prereg,
            "slices_in_manifest": self.slices_in_manifest,
            "slices_missing_prereg": self.slices_missing_prereg,
            "slices_missing_manifest": self.slices_missing_manifest,
            "slices_with_drift": self.slices_with_drift,
            "error_codes_by_slice": self.error_codes_by_slice,
            "overall_verdict": self.overall_verdict,
        }


def summarize_slice_hash_integrity(
    curriculum_config: Dict[str, Any],
    prereg_bindings: Dict[str, Dict[str, Any]],
    manifest_bindings: Dict[str, Dict[str, Any]],
) -> SliceHashIntegritySummary:
    """
    Generate a small, stable summary for MAAS or governance as "slice integrity verdict".

    This integrates cleanly with full_reconciliation() but provides a more
    compact output suitable for automated governance checks.

    Args:
        curriculum_config: The curriculum configuration
        prereg_bindings: Dict mapping slice_name to prereg slice_hash_binding
        manifest_bindings: Dict mapping slice_name to manifest slice_hash_binding

    Returns:
        SliceHashIntegritySummary with verdict information
    """
    # Extract slices from config
    slices_in_config: List[str] = []
    if "slices" in curriculum_config:
        slices_data = curriculum_config["slices"]
        if isinstance(slices_data, dict):
            slices_in_config = list(slices_data.keys())
        elif isinstance(slices_data, list):
            slices_in_config = [s.get("name") for s in slices_data if s.get("name")]
    elif "systems" in curriculum_config:
        for system in curriculum_config.get("systems", []):
            for s in system.get("slices", []):
                if s.get("name"):
                    slices_in_config.append(s.get("name"))

    slices_in_prereg = list(prereg_bindings.keys())
    slices_in_manifest = list(manifest_bindings.keys())

    # Determine missing slices
    config_set = set(slices_in_config)
    prereg_set = set(slices_in_prereg)
    manifest_set = set(slices_in_manifest)

    slices_missing_prereg = sorted(config_set - prereg_set)
    slices_missing_manifest = sorted(prereg_set - manifest_set)

    # Run full reconciliation to get drift info
    recon_report = full_reconciliation(curriculum_config, prereg_bindings, manifest_bindings)

    # Extract drift information
    slices_with_drift: List[str] = []
    error_codes_by_slice: Dict[str, List[str]] = {}

    for check in recon_report.get("checks", []):
        if not check.get("passed", True):
            slice_name = check.get("slice_name", "unknown")
            error_code = check.get("error_code")

            if slice_name not in slices_with_drift:
                slices_with_drift.append(slice_name)

            if slice_name not in error_codes_by_slice:
                error_codes_by_slice[slice_name] = []

            if error_code and error_code not in error_codes_by_slice[slice_name]:
                error_codes_by_slice[slice_name].append(error_code)

    # Determine overall verdict
    all_slices_accounted_for = (
        not slices_missing_prereg
        and not slices_missing_manifest
        and config_set == prereg_set
    )

    if slices_with_drift:
        overall_verdict = "FAIL"
    elif slices_missing_prereg or slices_missing_manifest:
        overall_verdict = "WARN"
    elif all_slices_accounted_for:
        overall_verdict = "PASS"
    else:
        overall_verdict = "WARN"

    return SliceHashIntegritySummary(
        timestamp=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        all_slices_accounted_for=all_slices_accounted_for,
        slices_in_config=sorted(slices_in_config),
        slices_in_prereg=sorted(slices_in_prereg),
        slices_in_manifest=sorted(slices_in_manifest),
        slices_missing_prereg=slices_missing_prereg,
        slices_missing_manifest=slices_missing_manifest,
        slices_with_drift=sorted(slices_with_drift),
        error_codes_by_slice=error_codes_by_slice,
        overall_verdict=overall_verdict,
    )


# =============================================================================
# PHASE III — Slice Identity Ledger & Drift Intelligence
# =============================================================================

# -----------------------------------------------------------------------------
# Task 1: Slice Identity Ledger
# -----------------------------------------------------------------------------

@dataclass
class DriftEvent:
    """
    A recorded drift event between two bindings.

    Captures when a slice's identity changed, what changed, and severity.
    """
    timestamp: str
    from_binding_id: str
    to_binding_id: str
    drift_codes: List[str]
    severity: str  # "blocking", "warning", "info"
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "from_binding_id": self.from_binding_id,
            "to_binding_id": self.to_binding_id,
            "drift_codes": self.drift_codes,
            "severity": self.severity,
            "details": self.details,
        }


@dataclass
class SliceIdentityLedgerEntry:
    """
    A ledger entry tracking a single slice's identity over its lifecycle.

    Tracks first/last appearances, all observed bindings, drift events,
    and computes a lineage stability index.
    """
    slice_name: str
    first_appearance: str  # ISO 8601 timestamp of first binding
    last_appearance: str   # ISO 8601 timestamp of most recent binding
    binding_count: int     # Total number of bindings observed
    bindings: List[Dict[str, Any]]  # All observed bindings (chronological)
    drift_events: List[DriftEvent]  # All detected drift events
    lineage_stability_index: float  # 0.0 (unstable) to 1.0 (perfectly stable)
    current_config_hash: Optional[str]  # Most recent config hash
    current_pool_hash: Optional[str]    # Most recent pool hash

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slice_name": self.slice_name,
            "first_appearance": self.first_appearance,
            "last_appearance": self.last_appearance,
            "binding_count": self.binding_count,
            "bindings": self.bindings,
            "drift_events": [e.to_dict() for e in self.drift_events],
            "lineage_stability_index": self.lineage_stability_index,
            "current_config_hash": self.current_config_hash,
            "current_pool_hash": self.current_pool_hash,
        }

    @property
    def is_stable(self) -> bool:
        """True if slice has never drifted (stability index = 1.0)."""
        return self.lineage_stability_index >= 1.0

    @property
    def has_blocking_drift(self) -> bool:
        """True if any drift event is blocking severity."""
        return any(e.severity == "blocking" for e in self.drift_events)


def _compute_lineage_stability_index(
    binding_count: int,
    drift_events: List[DriftEvent],
) -> float:
    """
    Compute the lineage stability index for a slice.

    Formula: 1.0 - (weighted_drift_score / max_possible_score)

    Where:
    - blocking drift = 1.0 penalty
    - warning drift = 0.5 penalty
    - info drift = 0.1 penalty
    - max_possible_score = binding_count - 1 (one potential drift per transition)

    Returns:
        Float in [0.0, 1.0] where 1.0 = perfectly stable (no drift)
    """
    if binding_count <= 1:
        return 1.0  # Single binding = perfectly stable

    max_transitions = binding_count - 1
    if max_transitions == 0:
        return 1.0

    severity_weights = {
        "blocking": 1.0,
        "warning": 0.5,
        "info": 0.1,
    }

    total_penalty = sum(
        severity_weights.get(e.severity, 0.5)
        for e in drift_events
    )

    # Normalize to [0, 1] and invert (higher = more stable)
    raw_stability = 1.0 - (total_penalty / max_transitions)
    return max(0.0, min(1.0, raw_stability))


def _classify_drift_severity(drift_codes: List[str]) -> str:
    """
    Classify drift severity based on error codes.

    - blocking: Config hash or pool hash mismatch (affects experiment validity)
    - warning: Ledger ID mismatch (affects traceability)
    - info: Minor metadata changes
    """
    blocking_codes = {
        ManifestHashMismatchError.MHM_001_CONFIG_HASH_MISMATCH.value,
        ManifestHashMismatchError.MHM_002_POOL_HASH_MISMATCH.value,
        SliceHashDriftError.SHD_001_LEDGER_FORMULA_DRIFT.value,
        SliceHashDriftError.SHD_002_LEDGER_POOL_DRIFT.value,
    }

    warning_codes = {
        ManifestHashMismatchError.MHM_003_LEDGER_ID_MISMATCH.value,
        SliceHashDriftError.SHD_003_LEDGER_ENTRY_MISSING.value,
    }

    for code in drift_codes:
        if code in blocking_codes:
            return "blocking"

    for code in drift_codes:
        if code in warning_codes:
            return "warning"

    return "info"


def build_slice_identity_ledger(
    bindings: List[Dict[str, Any]],
) -> Dict[str, SliceIdentityLedgerEntry]:
    """
    Build a slice identity ledger tracking lifecycle of all slices across bindings.

    This function takes a chronologically-ordered list of all observed bindings
    (from preregistrations, manifests, or other sources) and builds a ledger
    that tracks:
    - First and last appearance timestamps
    - All drift events between consecutive bindings
    - Lineage stability index per slice

    Args:
        bindings: List of slice_hash_binding dicts, ordered chronologically.
                  Each must have at minimum: slice_name, frozen_at,
                  slice_config_hash, formula_pool_hash, ledger_entry_id

    Returns:
        Dict mapping slice_name to SliceIdentityLedgerEntry
    """
    # Group bindings by slice
    bindings_by_slice: Dict[str, List[Dict[str, Any]]] = {}
    for binding in bindings:
        slice_name = binding.get("slice_name", "unknown")
        if slice_name not in bindings_by_slice:
            bindings_by_slice[slice_name] = []
        bindings_by_slice[slice_name].append(binding)

    ledger: Dict[str, SliceIdentityLedgerEntry] = {}

    for slice_name, slice_bindings in sorted(bindings_by_slice.items()):
        # Sort by frozen_at timestamp
        sorted_bindings = sorted(
            slice_bindings,
            key=lambda b: b.get("frozen_at", "")
        )

        # Determine first/last appearance
        first_appearance = sorted_bindings[0].get("frozen_at", "")
        last_appearance = sorted_bindings[-1].get("frozen_at", "")

        # Detect drift events between consecutive bindings
        drift_events: List[DriftEvent] = []
        for i in range(1, len(sorted_bindings)):
            prev_binding = sorted_bindings[i - 1]
            curr_binding = sorted_bindings[i]

            # Compare bindings
            comparison = compare_slice_bindings(prev_binding, curr_binding)

            if not comparison.is_identical:
                severity = _classify_drift_severity(comparison.drift_codes)
                drift_event = DriftEvent(
                    timestamp=curr_binding.get("frozen_at", ""),
                    from_binding_id=prev_binding.get("ledger_entry_id", ""),
                    to_binding_id=curr_binding.get("ledger_entry_id", ""),
                    drift_codes=comparison.drift_codes,
                    severity=severity,
                    details={
                        "config_hash_changed": not comparison.same_config_hash,
                        "pool_hash_changed": not comparison.same_formula_pool,
                        "ledger_id_changed": not comparison.same_ledger_entry_id,
                        "target_count_changed": not comparison.same_target_count,
                    }
                )
                drift_events.append(drift_event)

        # Compute stability index
        stability_index = _compute_lineage_stability_index(
            len(sorted_bindings),
            drift_events
        )

        # Extract current hashes from latest binding
        latest = sorted_bindings[-1]

        ledger[slice_name] = SliceIdentityLedgerEntry(
            slice_name=slice_name,
            first_appearance=first_appearance,
            last_appearance=last_appearance,
            binding_count=len(sorted_bindings),
            bindings=sorted_bindings,
            drift_events=drift_events,
            lineage_stability_index=stability_index,
            current_config_hash=latest.get("slice_config_hash"),
            current_pool_hash=latest.get("formula_pool_hash"),
        )

    return ledger


# -----------------------------------------------------------------------------
# Task 2: Drift Signature Generator
# -----------------------------------------------------------------------------

@dataclass
class DriftSignature:
    """
    A compact drift signature summarizing changes between two bindings.

    The signature is a short hash (12 chars) that uniquely identifies
    the specific drift pattern, useful for deduplication and tracking.
    """
    signature: str           # 12-character hex hash
    full_hash: str           # Full 64-character SHA256 hash
    drift_codes: List[str]   # Contributing error codes
    changed_fields: List[str]  # Which fields changed
    severity: str            # Overall severity classification

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signature": self.signature,
            "full_hash": self.full_hash,
            "drift_codes": self.drift_codes,
            "changed_fields": self.changed_fields,
            "severity": self.severity,
        }

    def __str__(self) -> str:
        return f"DRIFT-{self.signature}"


def compute_slice_drift_signature(
    binding_a: Dict[str, Any],
    binding_b: Dict[str, Any],
) -> DriftSignature:
    """
    Compute a short hash signature summarizing drift between two bindings.

    The drift signature captures:
    - Which fields changed (config_hash, pool_hash, ledger_id, target_count)
    - The actual changed values (for uniqueness)
    - Error codes triggered

    This produces a deterministic, compact identifier for the drift pattern
    that can be used for:
    - Deduplication of drift reports
    - Tracking recurring drift patterns
    - Quick comparison of drift types

    Args:
        binding_a: First binding (earlier/baseline)
        binding_b: Second binding (later/current)

    Returns:
        DriftSignature with short hash and metadata
    """
    # Compare bindings
    comparison = compare_slice_bindings(binding_a, binding_b)

    # Build canonical representation of the drift
    changed_fields: List[str] = []
    drift_details: Dict[str, Any] = {
        "slice_name": comparison.slice_name,
    }

    if not comparison.same_config_hash:
        changed_fields.append("config_hash")
        drift_details["config_hash_a"] = binding_a.get("slice_config_hash", "")[:16]
        drift_details["config_hash_b"] = binding_b.get("slice_config_hash", "")[:16]

    if not comparison.same_formula_pool:
        changed_fields.append("pool_hash")
        drift_details["pool_hash_a"] = binding_a.get("formula_pool_hash", "")[:16]
        drift_details["pool_hash_b"] = binding_b.get("formula_pool_hash", "")[:16]

    if not comparison.same_ledger_entry_id:
        changed_fields.append("ledger_id")
        drift_details["ledger_id_a"] = binding_a.get("ledger_entry_id", "")
        drift_details["ledger_id_b"] = binding_b.get("ledger_entry_id", "")

    if not comparison.same_target_count:
        changed_fields.append("target_count")
        drift_details["target_count_a"] = binding_a.get("target_count", 0)
        drift_details["target_count_b"] = binding_b.get("target_count", 0)

    # Add sorted drift codes for determinism
    drift_details["drift_codes"] = sorted(comparison.drift_codes)
    drift_details["changed_fields"] = sorted(changed_fields)

    # Compute canonical hash
    canonical = json.dumps(drift_details, sort_keys=True, separators=(',', ':'))
    full_hash = hashlib.sha256(canonical.encode('utf-8')).hexdigest()

    # Short signature is first 12 characters
    signature = full_hash[:12]

    # Classify severity
    severity = _classify_drift_severity(comparison.drift_codes)

    return DriftSignature(
        signature=signature,
        full_hash=full_hash,
        drift_codes=sorted(comparison.drift_codes),
        changed_fields=sorted(changed_fields),
        severity=severity,
    )


def compute_drift_signature_from_comparison(
    comparison: SliceBindingComparison,
    binding_a: Dict[str, Any],
    binding_b: Dict[str, Any],
) -> DriftSignature:
    """
    Compute drift signature from an existing SliceBindingComparison.

    Use this when you already have a comparison result and want to
    generate the signature without recomparing.

    Args:
        comparison: Existing SliceBindingComparison
        binding_a: First binding
        binding_b: Second binding

    Returns:
        DriftSignature with short hash and metadata
    """
    return compute_slice_drift_signature(binding_a, binding_b)


# -----------------------------------------------------------------------------
# Task 3: Global Health Summary for Governance
# -----------------------------------------------------------------------------

@dataclass
class SliceIdentityGlobalHealth:
    """
    Global health summary for governance systems.

    Provides a high-level view of slice identity health across the entire
    curriculum, suitable for MAAS integration and automated governance checks.
    """
    timestamp: str
    identity_stable: bool               # True if all slices are stable
    total_slices: int                   # Total slices in ledger
    stable_slices: int                  # Slices with stability index = 1.0
    unstable_slices: int                # Slices with any drift
    slices_with_drift: List[str]        # Names of slices that have drifted
    blocking_drift: List[str]           # Slices with blocking-severity drift
    warning_drift: List[str]            # Slices with warning-severity drift
    average_stability_index: float      # Mean stability index across all slices
    min_stability_index: float          # Lowest stability index
    min_stability_slice: Optional[str]  # Slice with lowest stability
    drift_signature_summary: Dict[str, int]  # Count of each unique drift signature
    overall_health: str                 # "HEALTHY", "DEGRADED", "CRITICAL"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "identity_stable": self.identity_stable,
            "total_slices": self.total_slices,
            "stable_slices": self.stable_slices,
            "unstable_slices": self.unstable_slices,
            "slices_with_drift": self.slices_with_drift,
            "blocking_drift": self.blocking_drift,
            "warning_drift": self.warning_drift,
            "average_stability_index": self.average_stability_index,
            "min_stability_index": self.min_stability_index,
            "min_stability_slice": self.min_stability_slice,
            "drift_signature_summary": self.drift_signature_summary,
            "overall_health": self.overall_health,
        }


def summarize_slice_identity_for_global_health(
    ledger: Dict[str, SliceIdentityLedgerEntry],
) -> SliceIdentityGlobalHealth:
    """
    Generate a global health summary from a slice identity ledger.

    This provides the governance-level view of slice identity health:
    - identity_stable: True only if ALL slices have never drifted
    - blocking_drift: Slices with drift that invalidates experiments
    - average/min stability indices for trend monitoring
    - Drift signature summary for pattern detection

    Args:
        ledger: Dict mapping slice_name to SliceIdentityLedgerEntry
                (output from build_slice_identity_ledger)

    Returns:
        SliceIdentityGlobalHealth summary for governance
    """
    if not ledger:
        return SliceIdentityGlobalHealth(
            timestamp=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            identity_stable=True,
            total_slices=0,
            stable_slices=0,
            unstable_slices=0,
            slices_with_drift=[],
            blocking_drift=[],
            warning_drift=[],
            average_stability_index=1.0,
            min_stability_index=1.0,
            min_stability_slice=None,
            drift_signature_summary={},
            overall_health="HEALTHY",
        )

    # Compute statistics
    total_slices = len(ledger)
    stable_slices = sum(1 for e in ledger.values() if e.is_stable)
    unstable_slices = total_slices - stable_slices

    # Identify slices with drift
    slices_with_drift: List[str] = []
    blocking_drift: List[str] = []
    warning_drift: List[str] = []

    for name, entry in ledger.items():
        if entry.drift_events:
            slices_with_drift.append(name)

            if entry.has_blocking_drift:
                blocking_drift.append(name)
            elif any(e.severity == "warning" for e in entry.drift_events):
                warning_drift.append(name)

    # Compute stability indices
    stability_indices = [e.lineage_stability_index for e in ledger.values()]
    average_stability = sum(stability_indices) / len(stability_indices)
    min_stability = min(stability_indices)

    # Find slice with minimum stability
    min_stability_slice = None
    for name, entry in ledger.items():
        if entry.lineage_stability_index == min_stability:
            min_stability_slice = name
            break

    # Compute drift signature summary
    drift_signature_summary: Dict[str, int] = {}
    for entry in ledger.values():
        if entry.binding_count > 1 and entry.drift_events:
            # Compute signatures for each consecutive binding pair
            for i in range(1, len(entry.bindings)):
                sig = compute_slice_drift_signature(
                    entry.bindings[i - 1],
                    entry.bindings[i]
                )
                if not compare_slice_bindings(entry.bindings[i-1], entry.bindings[i]).is_identical:
                    sig_str = str(sig)
                    drift_signature_summary[sig_str] = drift_signature_summary.get(sig_str, 0) + 1

    # Determine overall health
    identity_stable = (unstable_slices == 0)

    if blocking_drift:
        overall_health = "CRITICAL"
    elif warning_drift or unstable_slices > 0:
        overall_health = "DEGRADED"
    else:
        overall_health = "HEALTHY"

    return SliceIdentityGlobalHealth(
        timestamp=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        identity_stable=identity_stable,
        total_slices=total_slices,
        stable_slices=stable_slices,
        unstable_slices=unstable_slices,
        slices_with_drift=sorted(slices_with_drift),
        blocking_drift=sorted(blocking_drift),
        warning_drift=sorted(warning_drift),
        average_stability_index=round(average_stability, 4),
        min_stability_index=round(min_stability, 4),
        min_stability_slice=min_stability_slice,
        drift_signature_summary=drift_signature_summary,
        overall_health=overall_health,
    )


def get_blocking_drift_report(
    ledger: Dict[str, SliceIdentityLedgerEntry],
) -> Dict[str, Any]:
    """
    Generate a focused report on blocking drift events.

    This is a convenience function for governance systems that need
    to quickly identify experiment-blocking issues.

    Args:
        ledger: Slice identity ledger

    Returns:
        Report dict with blocking drift details
    """
    blocking_entries: List[Dict[str, Any]] = []

    for name, entry in sorted(ledger.items()):
        blocking_events = [
            e for e in entry.drift_events
            if e.severity == "blocking"
        ]

        if blocking_events:
            blocking_entries.append({
                "slice_name": name,
                "blocking_event_count": len(blocking_events),
                "events": [e.to_dict() for e in blocking_events],
                "current_config_hash": entry.current_config_hash,
                "current_pool_hash": entry.current_pool_hash,
            })

    return {
        "timestamp": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        "has_blocking_drift": len(blocking_entries) > 0,
        "blocking_slice_count": len(blocking_entries),
        "blocking_entries": blocking_entries,
    }


# =============================================================================
# PHASE IV — Slice Identity Guardrail & Curriculum/Evidence Coupler
# =============================================================================

# -----------------------------------------------------------------------------
# Task 1: Curriculum–Slice Identity Coupler
# -----------------------------------------------------------------------------

@dataclass
class SliceIdentityCurriculumView:
    """
    View of slice identity alignment with curriculum manifest.

    This couples the identity ledger to the curriculum, showing which slices
    are properly bound, which are missing bindings, and which have blocking drift.
    """
    timestamp: str
    slices_in_curriculum: List[str]
    slices_missing_bindings: List[str]
    slices_with_blocking_drift: List[str]
    slices_with_warning_drift: List[str]
    slices_stable_and_present: List[str]
    view_status: str  # "ALIGNED" | "PARTIAL" | "BROKEN"
    alignment_ratio: float  # 0.0 to 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "slices_in_curriculum": self.slices_in_curriculum,
            "slices_missing_bindings": self.slices_missing_bindings,
            "slices_with_blocking_drift": self.slices_with_blocking_drift,
            "slices_with_warning_drift": self.slices_with_warning_drift,
            "slices_stable_and_present": self.slices_stable_and_present,
            "view_status": self.view_status,
            "alignment_ratio": self.alignment_ratio,
        }


def _extract_slices_from_curriculum(curriculum_manifest: Dict[str, Any]) -> List[str]:
    """
    Extract slice names from a curriculum manifest.

    Handles both flat 'slices' dict/list and nested 'systems' structure.
    """
    slices: List[str] = []

    if "slices" in curriculum_manifest:
        slices_data = curriculum_manifest["slices"]
        if isinstance(slices_data, dict):
            slices = list(slices_data.keys())
        elif isinstance(slices_data, list):
            slices = [s.get("name") for s in slices_data if s.get("name")]
    elif "systems" in curriculum_manifest:
        for system in curriculum_manifest.get("systems", []):
            for s in system.get("slices", []):
                if s.get("name"):
                    slices.append(s.get("name"))

    return sorted(slices)


def build_slice_identity_curriculum_view(
    identity_ledger: Dict[str, SliceIdentityLedgerEntry],
    curriculum_manifest: Dict[str, Any],
) -> SliceIdentityCurriculumView:
    """
    Build a view showing alignment between slice identity ledger and curriculum.

    This function couples the identity ledger to the curriculum manifest,
    identifying:
    - Slices in curriculum but missing from identity ledger
    - Slices with blocking drift (invalidating experiments)
    - Slices that are stable and properly present

    Args:
        identity_ledger: Dict mapping slice_name to SliceIdentityLedgerEntry
        curriculum_manifest: The curriculum configuration dict

    Returns:
        SliceIdentityCurriculumView with alignment status
    """
    # Extract slices from curriculum
    slices_in_curriculum = _extract_slices_from_curriculum(curriculum_manifest)
    curriculum_set = set(slices_in_curriculum)
    ledger_set = set(identity_ledger.keys())

    # Find slices missing from ledger
    slices_missing_bindings = sorted(curriculum_set - ledger_set)

    # Categorize slices present in ledger
    slices_with_blocking_drift: List[str] = []
    slices_with_warning_drift: List[str] = []
    slices_stable_and_present: List[str] = []

    for slice_name in sorted(curriculum_set & ledger_set):
        entry = identity_ledger[slice_name]

        if entry.has_blocking_drift:
            slices_with_blocking_drift.append(slice_name)
        elif any(e.severity == "warning" for e in entry.drift_events):
            slices_with_warning_drift.append(slice_name)
        elif entry.is_stable:
            slices_stable_and_present.append(slice_name)
        else:
            # Has info-level drift but no blocking/warning
            slices_stable_and_present.append(slice_name)

    # Determine view status
    if slices_with_blocking_drift:
        view_status = "BROKEN"
    elif slices_missing_bindings or slices_with_warning_drift:
        view_status = "PARTIAL"
    elif len(slices_stable_and_present) == len(slices_in_curriculum) and slices_in_curriculum:
        view_status = "ALIGNED"
    elif not slices_in_curriculum:
        view_status = "ALIGNED"  # Empty curriculum is trivially aligned
    else:
        view_status = "PARTIAL"

    # Compute alignment ratio
    if slices_in_curriculum:
        alignment_ratio = len(slices_stable_and_present) / len(slices_in_curriculum)
    else:
        alignment_ratio = 1.0

    return SliceIdentityCurriculumView(
        timestamp=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        slices_in_curriculum=slices_in_curriculum,
        slices_missing_bindings=slices_missing_bindings,
        slices_with_blocking_drift=slices_with_blocking_drift,
        slices_with_warning_drift=slices_with_warning_drift,
        slices_stable_and_present=slices_stable_and_present,
        view_status=view_status,
        alignment_ratio=round(alignment_ratio, 4),
    )


# -----------------------------------------------------------------------------
# Task 2: Evidence Pack Identity Guard
# -----------------------------------------------------------------------------

@dataclass
class SliceIdentityEvidenceEvaluation:
    """
    Evaluation of whether slice identity state allows evidence pack creation.

    This guards evidence pack generation, ensuring that only experiments
    with stable slice identities can produce valid evidence.
    """
    timestamp: str
    identity_ok_for_evidence: bool
    slices_blocking_evidence: List[str]
    slices_with_warnings: List[str]
    slices_checked: int
    slices_passed: int
    status: str  # "OK" | "WARN" | "BLOCK"
    reasons: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "identity_ok_for_evidence": self.identity_ok_for_evidence,
            "slices_blocking_evidence": self.slices_blocking_evidence,
            "slices_with_warnings": self.slices_with_warnings,
            "slices_checked": self.slices_checked,
            "slices_passed": self.slices_passed,
            "status": self.status,
            "reasons": self.reasons,
        }


def _extract_slices_from_evidence_pack(evidence_pack: Dict[str, Any]) -> List[str]:
    """
    Extract slice names referenced in an evidence pack.

    Looks for slices in:
    - evidence_pack["slices"] (list of slice names or dicts)
    - evidence_pack["slice_bindings"] (dict of slice_name -> binding)
    - evidence_pack["experiment"]["slice_name"] (single slice)
    """
    slices: List[str] = []

    # Direct slices list
    if "slices" in evidence_pack:
        slices_data = evidence_pack["slices"]
        if isinstance(slices_data, list):
            for s in slices_data:
                if isinstance(s, str):
                    slices.append(s)
                elif isinstance(s, dict) and s.get("name"):
                    slices.append(s.get("name"))

    # Slice bindings dict
    if "slice_bindings" in evidence_pack:
        slices.extend(evidence_pack["slice_bindings"].keys())

    # Single experiment slice
    if "experiment" in evidence_pack:
        exp = evidence_pack["experiment"]
        if isinstance(exp, dict) and exp.get("slice_name"):
            slices.append(exp.get("slice_name"))

    # Slice hash binding (singular)
    if "slice_hash_binding" in evidence_pack:
        binding = evidence_pack["slice_hash_binding"]
        if isinstance(binding, dict) and binding.get("slice_name"):
            slices.append(binding.get("slice_name"))

    # Slice hash bindings (plural)
    if "slice_hash_bindings" in evidence_pack:
        for binding in evidence_pack.get("slice_hash_bindings", []):
            if isinstance(binding, dict) and binding.get("slice_name"):
                slices.append(binding.get("slice_name"))

    return sorted(set(slices))


def evaluate_slice_identity_for_evidence(
    identity_ledger: Dict[str, SliceIdentityLedgerEntry],
    evidence_pack: Dict[str, Any],
) -> SliceIdentityEvidenceEvaluation:
    """
    Evaluate whether slice identity state permits evidence pack creation.

    This is the guardrail that prevents evidence packs from being created
    when slice identities have blocking drift. It answers: "Can this
    evidence pack be trusted given the current slice identity state?"

    Args:
        identity_ledger: Dict mapping slice_name to SliceIdentityLedgerEntry
        evidence_pack: The evidence pack being evaluated

    Returns:
        SliceIdentityEvidenceEvaluation with status and reasons
    """
    # Extract slices from evidence pack
    slices_in_evidence = _extract_slices_from_evidence_pack(evidence_pack)

    slices_blocking_evidence: List[str] = []
    slices_with_warnings: List[str] = []
    slices_passed: List[str] = []
    reasons: List[str] = []

    for slice_name in slices_in_evidence:
        if slice_name not in identity_ledger:
            # Slice not in ledger - this is a blocking issue
            slices_blocking_evidence.append(slice_name)
            reasons.append(f"Slice '{slice_name}' has no identity binding in ledger")
        else:
            entry = identity_ledger[slice_name]

            if entry.has_blocking_drift:
                slices_blocking_evidence.append(slice_name)
                blocking_codes = [
                    e.drift_codes for e in entry.drift_events
                    if e.severity == "blocking"
                ]
                flat_codes = [c for codes in blocking_codes for c in codes]
                reasons.append(
                    f"Slice '{slice_name}' has blocking drift: {', '.join(set(flat_codes))}"
                )
            elif any(e.severity == "warning" for e in entry.drift_events):
                slices_with_warnings.append(slice_name)
                reasons.append(
                    f"Slice '{slice_name}' has warning-level drift (ledger ID changes)"
                )
                slices_passed.append(slice_name)  # Warnings don't block
            else:
                slices_passed.append(slice_name)

    # Determine status
    if slices_blocking_evidence:
        status = "BLOCK"
        identity_ok = False
    elif slices_with_warnings:
        status = "WARN"
        identity_ok = True  # Warnings don't block evidence
    else:
        status = "OK"
        identity_ok = True

    # Add summary reason
    if status == "BLOCK":
        reasons.insert(0, f"{len(slices_blocking_evidence)} slice(s) have identity issues blocking evidence")
    elif status == "WARN":
        reasons.insert(0, f"{len(slices_with_warnings)} slice(s) have identity warnings")
    elif slices_in_evidence:
        reasons.insert(0, f"All {len(slices_in_evidence)} slice(s) have stable identities")
    else:
        reasons.insert(0, "No slices referenced in evidence pack")

    return SliceIdentityEvidenceEvaluation(
        timestamp=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        identity_ok_for_evidence=identity_ok,
        slices_blocking_evidence=slices_blocking_evidence,
        slices_with_warnings=slices_with_warnings,
        slices_checked=len(slices_in_evidence),
        slices_passed=len(slices_passed),
        status=status,
        reasons=reasons,
    )


# -----------------------------------------------------------------------------
# Task 3: Director Slice Identity Panel
# -----------------------------------------------------------------------------

@dataclass
class SliceIdentityDirectorPanel:
    """
    High-level panel for director/governance view of slice identity health.

    Combines global health, curriculum alignment, and evidence evaluation
    into a single dashboard-ready summary.
    """
    timestamp: str
    status_light: str  # "GREEN" | "YELLOW" | "RED"
    stable_slice_count: int
    total_slice_count: int
    stable_slice_ratio: float
    slices_with_blocking_drift: List[str]
    slices_with_warning_drift: List[str]
    curriculum_alignment: str  # "ALIGNED" | "PARTIAL" | "BROKEN"
    evidence_status: str  # "OK" | "WARN" | "BLOCK"
    headline: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "status_light": self.status_light,
            "stable_slice_count": self.stable_slice_count,
            "total_slice_count": self.total_slice_count,
            "stable_slice_ratio": self.stable_slice_ratio,
            "slices_with_blocking_drift": self.slices_with_blocking_drift,
            "slices_with_warning_drift": self.slices_with_warning_drift,
            "curriculum_alignment": self.curriculum_alignment,
            "evidence_status": self.evidence_status,
            "headline": self.headline,
        }


def _generate_identity_headline(
    status_light: str,
    stable_count: int,
    total_count: int,
    blocking_count: int,
    warning_count: int,
) -> str:
    """
    Generate a neutral headline describing slice identity status.
    """
    if total_count == 0:
        return "No slices configured in curriculum."

    if status_light == "GREEN":
        if stable_count == total_count:
            return f"All {total_count} slice(s) have stable identities."
        else:
            return f"{stable_count} of {total_count} slice(s) have stable identities."

    elif status_light == "YELLOW":
        if warning_count > 0:
            return f"{warning_count} slice(s) have identity warnings; {stable_count} of {total_count} stable."
        else:
            return f"{stable_count} of {total_count} slice(s) have stable identities."

    else:  # RED
        if blocking_count > 0:
            return f"{blocking_count} slice(s) have blocking identity drift; experiments may be invalid."
        else:
            return f"Slice identity issues detected; {stable_count} of {total_count} stable."


def build_slice_identity_director_panel(
    global_health: Dict[str, Any],
    curriculum_view: Dict[str, Any],
    evidence_eval: Dict[str, Any],
) -> SliceIdentityDirectorPanel:
    """
    Build a director-level panel summarizing slice identity health.

    This combines global health, curriculum alignment, and evidence evaluation
    into a single view suitable for governance dashboards and quick status checks.

    Args:
        global_health: Output from summarize_slice_identity_for_global_health().to_dict()
        curriculum_view: Output from build_slice_identity_curriculum_view().to_dict()
        evidence_eval: Output from evaluate_slice_identity_for_evidence().to_dict()

    Returns:
        SliceIdentityDirectorPanel with status light and headline
    """
    # Extract counts from global health
    total_slices = global_health.get("total_slices", 0)
    stable_slices = global_health.get("stable_slices", 0)
    blocking_drift = global_health.get("blocking_drift", [])
    warning_drift = global_health.get("warning_drift", [])
    overall_health = global_health.get("overall_health", "HEALTHY")

    # Get curriculum alignment status
    curriculum_alignment = curriculum_view.get("view_status", "ALIGNED")

    # Get evidence status
    evidence_status = evidence_eval.get("status", "OK")

    # Determine status light
    # Priority: blocking drift > evidence block > curriculum broken > warnings
    if blocking_drift or evidence_status == "BLOCK" or curriculum_alignment == "BROKEN":
        status_light = "RED"
    elif warning_drift or evidence_status == "WARN" or curriculum_alignment == "PARTIAL":
        status_light = "YELLOW"
    else:
        status_light = "GREEN"

    # Compute stable ratio
    if total_slices > 0:
        stable_ratio = stable_slices / total_slices
    else:
        stable_ratio = 1.0

    # Generate headline
    headline = _generate_identity_headline(
        status_light=status_light,
        stable_count=stable_slices,
        total_count=total_slices,
        blocking_count=len(blocking_drift),
        warning_count=len(warning_drift),
    )

    return SliceIdentityDirectorPanel(
        timestamp=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        status_light=status_light,
        stable_slice_count=stable_slices,
        total_slice_count=total_slices,
        stable_slice_ratio=round(stable_ratio, 4),
        slices_with_blocking_drift=sorted(blocking_drift),
        slices_with_warning_drift=sorted(warning_drift),
        curriculum_alignment=curriculum_alignment,
        evidence_status=evidence_status,
        headline=headline,
    )


def quick_identity_status(
    identity_ledger: Dict[str, SliceIdentityLedgerEntry],
    curriculum_manifest: Optional[Dict[str, Any]] = None,
    evidence_pack: Optional[Dict[str, Any]] = None,
) -> SliceIdentityDirectorPanel:
    """
    Convenience function to get a quick director panel from raw inputs.

    This combines all three Phase IV components into a single call.

    Args:
        identity_ledger: The slice identity ledger
        curriculum_manifest: Optional curriculum config (uses empty if None)
        evidence_pack: Optional evidence pack (uses empty if None)

    Returns:
        SliceIdentityDirectorPanel with full status
    """
    # Build global health
    global_health = summarize_slice_identity_for_global_health(identity_ledger).to_dict()

    # Build curriculum view
    curriculum_view = build_slice_identity_curriculum_view(
        identity_ledger,
        curriculum_manifest or {}
    ).to_dict()

    # Build evidence evaluation
    evidence_eval = evaluate_slice_identity_for_evidence(
        identity_ledger,
        evidence_pack or {}
    ).to_dict()

    return build_slice_identity_director_panel(
        global_health,
        curriculum_view,
        evidence_eval,
    )


# =============================================================================
# PHASE V — Slice Identity × Curriculum Drift × Evidence Coupling
# =============================================================================

# -----------------------------------------------------------------------------
# Task 1: Curriculum Drift Coupling
# -----------------------------------------------------------------------------

@dataclass
class SliceIdentityDriftView:
    """
    View coupling slice identity drift with curriculum drift history.

    This provides a unified view of identity drift (from hash binding changes)
    and curriculum drift (from curriculum version/config changes), enabling
    governance to detect when slices should not be promoted or used as evidence.
    """
    timestamp: str
    slices_with_identity_drift: List[str]
    slices_with_curriculum_drift: List[str]
    slices_with_both_drift: List[str]
    slices_clean: List[str]
    alignment_status: str  # "ALIGNED" | "PARTIAL" | "BROKEN"
    reasons: List[str]
    drift_signatures: Dict[str, str]  # slice_name -> DRIFT-<signature>

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "slices_with_identity_drift": self.slices_with_identity_drift,
            "slices_with_curriculum_drift": self.slices_with_curriculum_drift,
            "slices_with_both_drift": self.slices_with_both_drift,
            "slices_clean": self.slices_clean,
            "alignment_status": self.alignment_status,
            "reasons": self.reasons,
            "drift_signatures": self.drift_signatures,
        }


def _extract_curriculum_drift_slices(curriculum_history: Dict[str, Any]) -> List[str]:
    """
    Extract slices that have curriculum drift from a curriculum history object.

    The curriculum_history structure can have:
    - "drift_slices": list of slice names with drift
    - "slices_with_changes": list of slice names with changes
    - "version_changes": dict mapping slice_name to version change info
    - "slice_drift": dict mapping slice_name to drift info
    """
    drift_slices: List[str] = []

    # Direct drift list
    if "drift_slices" in curriculum_history:
        drift_slices.extend(curriculum_history["drift_slices"])

    # Slices with changes
    if "slices_with_changes" in curriculum_history:
        drift_slices.extend(curriculum_history["slices_with_changes"])

    # Version changes dict
    if "version_changes" in curriculum_history:
        drift_slices.extend(curriculum_history["version_changes"].keys())

    # Slice drift dict
    if "slice_drift" in curriculum_history:
        drift_slices.extend(curriculum_history["slice_drift"].keys())

    # Changed slices (alternative name)
    if "changed_slices" in curriculum_history:
        drift_slices.extend(curriculum_history["changed_slices"])

    return sorted(set(drift_slices))


def build_slice_identity_drift_view(
    identity_ledger: Dict[str, SliceIdentityLedgerEntry],
    curriculum_history: Dict[str, Any],
) -> SliceIdentityDriftView:
    """
    Build a view coupling identity drift with curriculum drift.

    This function computes:
    - slices_with_identity_drift: slices where DRIFT signatures exist
    - slices_with_curriculum_drift: slices from curriculum drift history
    - alignment_status: ALIGNED (no drift), PARTIAL (some drift), BROKEN (both drift)

    Args:
        identity_ledger: Dict mapping slice_name to SliceIdentityLedgerEntry
        curriculum_history: Dict with curriculum drift information

    Returns:
        SliceIdentityDriftView with alignment status and reasons
    """
    # Extract identity drift (slices with any drift events)
    slices_with_identity_drift: List[str] = []
    drift_signatures: Dict[str, str] = {}

    for slice_name, entry in identity_ledger.items():
        if entry.drift_events:
            slices_with_identity_drift.append(slice_name)
            # Compute drift signature from most recent drift
            if entry.binding_count >= 2 and len(entry.bindings) >= 2:
                # Find the bindings involved in the last drift
                for i in range(len(entry.bindings) - 1, 0, -1):
                    sig = compute_slice_drift_signature(
                        entry.bindings[i - 1],
                        entry.bindings[i]
                    )
                    if sig.changed_fields:  # Only if there was actual drift
                        drift_signatures[slice_name] = str(sig)
                        break

    # Extract curriculum drift
    slices_with_curriculum_drift = _extract_curriculum_drift_slices(curriculum_history)

    # Compute intersection and difference
    identity_set = set(slices_with_identity_drift)
    curriculum_set = set(slices_with_curriculum_drift)
    all_slices = set(identity_ledger.keys())

    slices_with_both_drift = sorted(identity_set & curriculum_set)
    slices_clean = sorted(all_slices - identity_set - curriculum_set)

    # Build reasons
    reasons: List[str] = []

    if slices_with_both_drift:
        reasons.append(
            f"{len(slices_with_both_drift)} slice(s) have both identity and curriculum drift."
        )

    if slices_with_identity_drift and not slices_with_curriculum_drift:
        reasons.append(
            f"{len(slices_with_identity_drift)} slice(s) have identity drift only."
        )

    if slices_with_curriculum_drift and not slices_with_identity_drift:
        reasons.append(
            f"{len(slices_with_curriculum_drift)} slice(s) have curriculum drift only."
        )

    if slices_clean:
        reasons.append(
            f"{len(slices_clean)} slice(s) are clean with no drift."
        )

    if not reasons:
        if all_slices:
            reasons.append("All slices are clean with no drift detected.")
        else:
            reasons.append("No slices in identity ledger.")

    # Determine alignment status
    if slices_with_both_drift:
        alignment_status = "BROKEN"
    elif slices_with_identity_drift or slices_with_curriculum_drift:
        alignment_status = "PARTIAL"
    else:
        alignment_status = "ALIGNED"

    return SliceIdentityDriftView(
        timestamp=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        slices_with_identity_drift=sorted(slices_with_identity_drift),
        slices_with_curriculum_drift=sorted(slices_with_curriculum_drift),
        slices_with_both_drift=slices_with_both_drift,
        slices_clean=slices_clean,
        alignment_status=alignment_status,
        reasons=reasons,
        drift_signatures=drift_signatures,
    )


# -----------------------------------------------------------------------------
# Task 2: Global Health Adapter for Console
# -----------------------------------------------------------------------------

@dataclass
class SliceIdentityGlobalConsole:
    """
    Adapter for global console display of slice identity health.

    Combines identity global health with drift view for a single
    console-friendly summary suitable for automated monitoring.
    """
    timestamp: str
    identity_ok: bool
    status: str  # "OK" | "WARN" | "BLOCK"
    blocking_slices: List[str]
    warning_slices: List[str]
    headline: str
    detail_lines: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "identity_ok": self.identity_ok,
            "status": self.status,
            "blocking_slices": self.blocking_slices,
            "warning_slices": self.warning_slices,
            "headline": self.headline,
            "detail_lines": self.detail_lines,
        }


def summarize_slice_identity_for_global_console(
    identity_global_health: Dict[str, Any],
    drift_view: Dict[str, Any],
) -> SliceIdentityGlobalConsole:
    """
    Generate a global console summary combining identity health and drift view.

    This adapter formats the combined health and drift information for
    console display, providing:
    - identity_ok: False if any blocking issues exist
    - status: OK/WARN/BLOCK based on severity
    - blocking_slices: combined list of blocking issues
    - headline: single-line summary

    Args:
        identity_global_health: Output from summarize_slice_identity_for_global_health().to_dict()
        drift_view: Output from build_slice_identity_drift_view().to_dict()

    Returns:
        SliceIdentityGlobalConsole with combined status
    """
    # Extract blocking slices from both sources
    blocking_from_health = identity_global_health.get("blocking_drift", [])
    blocking_from_drift = drift_view.get("slices_with_both_drift", [])

    # Combine blocking slices (deduplicated)
    blocking_slices = sorted(set(blocking_from_health) | set(blocking_from_drift))

    # Extract warning slices
    warning_from_health = identity_global_health.get("warning_drift", [])
    identity_drift_only = [
        s for s in drift_view.get("slices_with_identity_drift", [])
        if s not in blocking_slices
    ]
    curriculum_drift_only = [
        s for s in drift_view.get("slices_with_curriculum_drift", [])
        if s not in blocking_slices
    ]
    warning_slices = sorted(
        set(warning_from_health) | set(identity_drift_only) | set(curriculum_drift_only)
    )

    # Determine overall status
    if blocking_slices:
        status = "BLOCK"
        identity_ok = False
    elif warning_slices:
        status = "WARN"
        identity_ok = True  # Warnings don't block
    else:
        status = "OK"
        identity_ok = True

    # Get overall health and alignment
    overall_health = identity_global_health.get("overall_health", "HEALTHY")
    alignment_status = drift_view.get("alignment_status", "ALIGNED")

    # Build headline
    total_slices = identity_global_health.get("total_slices", 0)
    stable_slices = identity_global_health.get("stable_slices", 0)

    if status == "OK":
        if total_slices > 0:
            headline = f"Identity OK: {stable_slices}/{total_slices} slices stable, alignment {alignment_status}."
        else:
            headline = "Identity OK: No slices configured."
    elif status == "WARN":
        headline = f"Identity WARN: {len(warning_slices)} slice(s) with drift, {stable_slices}/{total_slices} stable."
    else:  # BLOCK
        headline = f"Identity BLOCK: {len(blocking_slices)} slice(s) blocking, alignment {alignment_status}."

    # Build detail lines
    detail_lines: List[str] = []

    if blocking_slices:
        detail_lines.append(f"Blocking: {', '.join(blocking_slices)}")

    if warning_slices:
        detail_lines.append(f"Warning: {', '.join(warning_slices)}")

    # Add reasons from drift view
    reasons = drift_view.get("reasons", [])
    for reason in reasons[:3]:  # Limit to first 3 reasons
        detail_lines.append(f"  - {reason}")

    # Add stability info
    avg_stability = identity_global_health.get("average_stability_index", 1.0)
    detail_lines.append(f"Average stability: {avg_stability:.2%}")

    return SliceIdentityGlobalConsole(
        timestamp=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        identity_ok=identity_ok,
        status=status,
        blocking_slices=blocking_slices,
        warning_slices=warning_slices,
        headline=headline,
        detail_lines=detail_lines,
    )


# -----------------------------------------------------------------------------
# Task 3: Evidence Pack Hook - Extended Identity Summary
# -----------------------------------------------------------------------------

@dataclass
class SliceIdentitySummary:
    """
    Compact, JSON-safe identity summary for a single slice.

    This is designed for inclusion in evidence packs, providing all
    necessary identity information in a minimal footprint.
    """
    slice_name: str
    is_stable: bool
    has_blocking_drift: bool
    stability_index: float
    drift_signature: Optional[str]  # DRIFT-<12-hex> or None
    binding_count: int
    first_seen: str
    last_seen: str
    current_config_hash_prefix: str  # First 12 chars
    current_pool_hash_prefix: str    # First 12 chars

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slice_name": self.slice_name,
            "is_stable": self.is_stable,
            "has_blocking_drift": self.has_blocking_drift,
            "stability_index": self.stability_index,
            "drift_signature": self.drift_signature,
            "binding_count": self.binding_count,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "current_config_hash_prefix": self.current_config_hash_prefix,
            "current_pool_hash_prefix": self.current_pool_hash_prefix,
        }


def get_slice_identity_summary(
    entry: SliceIdentityLedgerEntry,
) -> SliceIdentitySummary:
    """
    Generate a compact identity summary for a single slice.

    Args:
        entry: SliceIdentityLedgerEntry for the slice

    Returns:
        SliceIdentitySummary with compact identity info
    """
    # Compute drift signature if there was drift
    drift_signature: Optional[str] = None
    if entry.binding_count >= 2 and entry.drift_events:
        # Get signature from most recent drift
        for i in range(len(entry.bindings) - 1, 0, -1):
            sig = compute_slice_drift_signature(
                entry.bindings[i - 1],
                entry.bindings[i]
            )
            if sig.changed_fields:
                drift_signature = str(sig)
                break

    # Extract hash prefixes (first 12 chars)
    config_prefix = (entry.current_config_hash or "")[:12]
    pool_prefix = (entry.current_pool_hash or "")[:12]

    return SliceIdentitySummary(
        slice_name=entry.slice_name,
        is_stable=entry.is_stable,
        has_blocking_drift=entry.has_blocking_drift,
        stability_index=round(entry.lineage_stability_index, 4),
        drift_signature=drift_signature,
        binding_count=entry.binding_count,
        first_seen=entry.first_appearance,
        last_seen=entry.last_appearance,
        current_config_hash_prefix=config_prefix,
        current_pool_hash_prefix=pool_prefix,
    )


def get_slice_identity_summary_for_evidence(
    identity_ledger: Dict[str, SliceIdentityLedgerEntry],
    slice_names: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Generate compact identity summaries for inclusion in evidence packs.

    This helper produces a JSON-safe dict suitable for embedding in
    evidence pack metadata, providing traceability without bulk.

    Args:
        identity_ledger: The slice identity ledger
        slice_names: Optional list of slice names to include (all if None)

    Returns:
        Dict mapping slice_name to compact summary dict
    """
    if slice_names is None:
        slice_names = list(identity_ledger.keys())

    summaries: Dict[str, Dict[str, Any]] = {}

    for name in sorted(slice_names):
        if name in identity_ledger:
            entry = identity_ledger[name]
            summaries[name] = get_slice_identity_summary(entry).to_dict()

    return summaries


@dataclass
class SliceIdentityEvidenceEvaluationExtended:
    """
    Extended evidence evaluation including drift signatures and alignment.

    This extends SliceIdentityEvidenceEvaluation with Phase V drift coupling.
    """
    timestamp: str
    identity_ok_for_evidence: bool
    slices_blocking_evidence: List[str]
    slices_with_warnings: List[str]
    slices_checked: int
    slices_passed: int
    status: str  # "OK" | "WARN" | "BLOCK"
    reasons: List[str]
    drift_signatures: Dict[str, str]
    alignment_status: str  # "ALIGNED" | "PARTIAL" | "BROKEN"
    identity_summaries: Dict[str, Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "identity_ok_for_evidence": self.identity_ok_for_evidence,
            "slices_blocking_evidence": self.slices_blocking_evidence,
            "slices_with_warnings": self.slices_with_warnings,
            "slices_checked": self.slices_checked,
            "slices_passed": self.slices_passed,
            "status": self.status,
            "reasons": self.reasons,
            "drift_signatures": self.drift_signatures,
            "alignment_status": self.alignment_status,
            "identity_summaries": self.identity_summaries,
        }


def evaluate_slice_identity_for_evidence_extended(
    identity_ledger: Dict[str, SliceIdentityLedgerEntry],
    evidence_pack: Dict[str, Any],
    curriculum_history: Optional[Dict[str, Any]] = None,
) -> SliceIdentityEvidenceEvaluationExtended:
    """
    Extended evaluation including drift signatures and alignment status.

    This combines the basic evidence evaluation with:
    - Drift signatures for each slice with drift
    - Alignment status from curriculum drift coupling
    - Compact identity summaries for evidence inclusion

    Args:
        identity_ledger: The slice identity ledger
        evidence_pack: The evidence pack being evaluated
        curriculum_history: Optional curriculum drift history

    Returns:
        SliceIdentityEvidenceEvaluationExtended with full drift info
    """
    # Get basic evaluation
    basic_eval = evaluate_slice_identity_for_evidence(identity_ledger, evidence_pack)

    # Build drift view
    drift_view = build_slice_identity_drift_view(
        identity_ledger,
        curriculum_history or {}
    )

    # Extract slices from evidence pack
    slices_in_evidence = _extract_slices_from_evidence_pack(evidence_pack)

    # Get drift signatures for relevant slices
    drift_signatures: Dict[str, str] = {}
    for slice_name in slices_in_evidence:
        if slice_name in drift_view.drift_signatures:
            drift_signatures[slice_name] = drift_view.drift_signatures[slice_name]

    # Get identity summaries for evidence slices
    identity_summaries = get_slice_identity_summary_for_evidence(
        identity_ledger,
        slices_in_evidence
    )

    # Update blocking status based on alignment
    blocking_slices = list(basic_eval.slices_blocking_evidence)
    reasons = list(basic_eval.reasons)

    # Add slices with both drift to blocking if not already there
    # Also add reason for slices that are already blocking due to identity drift
    for slice_name in drift_view.slices_with_both_drift:
        if slice_name in slices_in_evidence:
            if slice_name not in blocking_slices:
                blocking_slices.append(slice_name)
            # Always add the "both drift" reason for slices in evidence
            reasons.append(
                f"Slice '{slice_name}' has both identity and curriculum drift."
            )

    # Determine final status - only consider slices that are actually in the evidence pack
    identity_drift_in_evidence = [
        s for s in drift_view.slices_with_identity_drift
        if s in slices_in_evidence
    ]

    if blocking_slices:
        status = "BLOCK"
        identity_ok = False
    elif basic_eval.slices_with_warnings or identity_drift_in_evidence:
        status = "WARN"
        identity_ok = True
    else:
        status = "OK"
        identity_ok = True

    return SliceIdentityEvidenceEvaluationExtended(
        timestamp=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        identity_ok_for_evidence=identity_ok,
        slices_blocking_evidence=sorted(blocking_slices),
        slices_with_warnings=basic_eval.slices_with_warnings,
        slices_checked=basic_eval.slices_checked,
        slices_passed=basic_eval.slices_passed,
        status=status,
        reasons=reasons,
        drift_signatures=drift_signatures,
        alignment_status=drift_view.alignment_status,
        identity_summaries=identity_summaries,
    )


# =============================================================================
# PHASE VI — Global Console Tile + Governance Signal
# =============================================================================

# -----------------------------------------------------------------------------
# Task 1: Global Console Tile Adapter
# -----------------------------------------------------------------------------

@dataclass
class SliceIdentityConsoleTile:
    """
    Console tile for slice identity status display.

    This provides a compact, dashboard-ready representation of slice identity
    health suitable for global console displays and monitoring systems.

    The tile combines:
    - Identity health from the global console adapter
    - Drift view alignment status
    - Counts of slices in various states
    """
    timestamp: str
    identity_ok: bool
    status_light: str  # "GREEN" | "YELLOW" | "RED"
    headline: str
    slices_with_both_drift_count: int
    slices_with_identity_drift_count: int
    slices_with_curriculum_drift_count: int
    slices_clean_count: int
    total_slices: int
    alignment_status: str  # "ALIGNED" | "PARTIAL" | "BROKEN"
    blocking_slices: List[str]
    average_stability: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "identity_ok": self.identity_ok,
            "status_light": self.status_light,
            "headline": self.headline,
            "slices_with_both_drift_count": self.slices_with_both_drift_count,
            "slices_with_identity_drift_count": self.slices_with_identity_drift_count,
            "slices_with_curriculum_drift_count": self.slices_with_curriculum_drift_count,
            "slices_clean_count": self.slices_clean_count,
            "total_slices": self.total_slices,
            "alignment_status": self.alignment_status,
            "blocking_slices": self.blocking_slices,
            "average_stability": self.average_stability,
        }


def build_slice_identity_console_tile(
    identity_console: Dict[str, Any],
    drift_view: Dict[str, Any],
) -> SliceIdentityConsoleTile:
    """
    Build a console tile for slice identity status.

    This adapter combines the identity console summary with drift view
    to create a compact tile suitable for global dashboard display.

    Args:
        identity_console: Output from summarize_slice_identity_for_global_console().to_dict()
        drift_view: Output from build_slice_identity_drift_view().to_dict()

    Returns:
        SliceIdentityConsoleTile with dashboard-ready status
    """
    # Extract counts from drift view
    slices_with_both_drift = drift_view.get("slices_with_both_drift", [])
    slices_with_identity_drift = drift_view.get("slices_with_identity_drift", [])
    slices_with_curriculum_drift = drift_view.get("slices_with_curriculum_drift", [])
    slices_clean = drift_view.get("slices_clean", [])
    alignment_status = drift_view.get("alignment_status", "ALIGNED")

    # Extract status from identity console
    identity_ok = identity_console.get("identity_ok", True)
    status = identity_console.get("status", "OK")
    blocking_slices = identity_console.get("blocking_slices", [])
    headline = identity_console.get("headline", "")

    # Map status to status_light
    if status == "BLOCK":
        status_light = "RED"
    elif status == "WARN":
        status_light = "YELLOW"
    else:
        status_light = "GREEN"

    # Calculate total slices (unique across all categories)
    all_slices = set(slices_with_identity_drift) | set(slices_with_curriculum_drift) | set(slices_clean)
    total_slices = len(all_slices)

    # Extract average stability from detail lines if available
    average_stability = 1.0
    detail_lines = identity_console.get("detail_lines", [])
    for line in detail_lines:
        if "stability:" in line.lower():
            # Extract percentage from line like "Average stability: 85.00%"
            try:
                parts = line.split(":")
                if len(parts) >= 2:
                    pct_str = parts[1].strip().rstrip("%")
                    average_stability = float(pct_str) / 100.0
            except (ValueError, IndexError):
                pass

    return SliceIdentityConsoleTile(
        timestamp=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        identity_ok=identity_ok,
        status_light=status_light,
        headline=headline,
        slices_with_both_drift_count=len(slices_with_both_drift),
        slices_with_identity_drift_count=len(slices_with_identity_drift),
        slices_with_curriculum_drift_count=len(slices_with_curriculum_drift),
        slices_clean_count=len(slices_clean),
        total_slices=total_slices,
        alignment_status=alignment_status,
        blocking_slices=blocking_slices,
        average_stability=average_stability,
    )


# -----------------------------------------------------------------------------
# Task 2: Governance Signal Adapter
# -----------------------------------------------------------------------------

@dataclass
class SliceIdentityGovernanceSignal:
    """
    Governance signal for slice identity status.

    This provides a standardized governance signal that can be consumed
    by CLAUDE I and other governance systems. The signal maps:
    - BROKEN alignment → BLOCK signal
    - PARTIAL alignment → WARN signal
    - ALIGNED alignment → OK signal

    The signal includes:
    - signal: The governance action (OK/WARN/BLOCK)
    - source: The source system identifier
    - severity: Numeric severity (0=OK, 1=WARN, 2=BLOCK)
    - blocking: Whether this signal blocks promotion/evidence
    - details: Additional context for governance decisions
    """
    timestamp: str
    signal: str  # "OK" | "WARN" | "BLOCK"
    source: str  # "slice_identity"
    severity: int  # 0=OK, 1=WARN, 2=BLOCK
    blocking: bool
    alignment_status: str
    identity_ok: bool
    blocking_slices: List[str]
    total_slices: int
    average_stability: float
    headline: str
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "signal": self.signal,
            "source": self.source,
            "severity": self.severity,
            "blocking": self.blocking,
            "alignment_status": self.alignment_status,
            "identity_ok": self.identity_ok,
            "blocking_slices": self.blocking_slices,
            "total_slices": self.total_slices,
            "average_stability": self.average_stability,
            "headline": self.headline,
            "details": self.details,
        }


def to_governance_signal_for_slice_identity(
    console_tile: Dict[str, Any],
) -> SliceIdentityGovernanceSignal:
    """
    Convert a console tile to a governance signal.

    This adapter transforms the console tile into a standardized governance
    signal suitable for consumption by CLAUDE I and other governance systems.

    Signal mapping:
    - BROKEN alignment or RED status_light → BLOCK (severity 2)
    - PARTIAL alignment or YELLOW status_light → WARN (severity 1)
    - ALIGNED alignment and GREEN status_light → OK (severity 0)

    Args:
        console_tile: Output from build_slice_identity_console_tile().to_dict()

    Returns:
        SliceIdentityGovernanceSignal for governance consumption
    """
    alignment_status = console_tile.get("alignment_status", "ALIGNED")
    status_light = console_tile.get("status_light", "GREEN")
    identity_ok = console_tile.get("identity_ok", True)
    blocking_slices = console_tile.get("blocking_slices", [])
    total_slices = console_tile.get("total_slices", 0)
    average_stability = console_tile.get("average_stability", 1.0)
    headline = console_tile.get("headline", "")

    # Determine signal based on alignment and status
    if alignment_status == "BROKEN" or status_light == "RED":
        signal = "BLOCK"
        severity = 2
        blocking = True
    elif alignment_status == "PARTIAL" or status_light == "YELLOW":
        signal = "WARN"
        severity = 1
        blocking = False
    else:
        signal = "OK"
        severity = 0
        blocking = False

    # Build details dict for governance context
    details: Dict[str, Any] = {
        "slices_with_both_drift_count": console_tile.get("slices_with_both_drift_count", 0),
        "slices_with_identity_drift_count": console_tile.get("slices_with_identity_drift_count", 0),
        "slices_with_curriculum_drift_count": console_tile.get("slices_with_curriculum_drift_count", 0),
        "slices_clean_count": console_tile.get("slices_clean_count", 0),
    }

    return SliceIdentityGovernanceSignal(
        timestamp=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        signal=signal,
        source="slice_identity",
        severity=severity,
        blocking=blocking,
        alignment_status=alignment_status,
        identity_ok=identity_ok,
        blocking_slices=blocking_slices,
        total_slices=total_slices,
        average_stability=average_stability,
        headline=headline,
        details=details,
    )


def build_full_slice_identity_governance_pipeline(
    identity_ledger: Dict[str, SliceIdentityLedgerEntry],
    curriculum_history: Optional[Dict[str, Any]] = None,
) -> SliceIdentityGovernanceSignal:
    """
    Full pipeline from identity ledger to governance signal.

    This convenience function runs the complete pipeline:
    1. Compute global health
    2. Build drift view
    3. Generate console summary
    4. Build console tile
    5. Convert to governance signal

    Args:
        identity_ledger: Dict mapping slice_name to SliceIdentityLedgerEntry
        curriculum_history: Optional curriculum drift history

    Returns:
        SliceIdentityGovernanceSignal ready for CLAUDE I
    """
    # Step 1: Compute global health
    global_health = summarize_slice_identity_for_global_health(identity_ledger)

    # Step 2: Build drift view
    drift_view = build_slice_identity_drift_view(
        identity_ledger,
        curriculum_history or {}
    )

    # Step 3: Generate console summary
    console_summary = summarize_slice_identity_for_global_console(
        global_health.to_dict(),
        drift_view.to_dict()
    )

    # Step 4: Build console tile
    console_tile = build_slice_identity_console_tile(
        console_summary.to_dict(),
        drift_view.to_dict()
    )

    # Step 5: Convert to governance signal
    return to_governance_signal_for_slice_identity(console_tile.to_dict())
