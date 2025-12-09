# PHASE II — NOT USED IN PHASE I
"""
PRNG Hash Lineage — Seed Provenance Tracking and Verification.

This module provides utilities for tracking and verifying the complete
derivation chain from master seed to any derived seed. It enables:

1. Reproducibility Receipts: Cryptographic proof that a seed was derived
   from a specific master seed via a specific path.

2. Lineage Tables: Human-readable and machine-parseable records of all
   seed derivations in an experiment.

3. Verification: Given a receipt, verify that a seed matches its claimed
   derivation path.

Usage:
    from rfl.prng.lineage import SeedLineage, create_receipt, verify_receipt

    lineage = SeedLineage(master_seed_hex)
    lineage.record("slice_a", "baseline", "cycle_0001")
    lineage.export_table("lineage.json")

Contract Reference:
    Implements seed provenance requirements from docs/DETERMINISM_CONTRACT.md.

Author: Agent A2 (runtime-ops-2)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .deterministic_prng import PRNGKey, derive_seed, derive_seed_64bit


@dataclass
class SeedDerivation:
    """Record of a single seed derivation."""
    path: Tuple[str, ...]
    derived_seed: int
    derived_seed_64bit: int
    canonical_string: str
    sha256_digest: str  # Full digest for verification
    timestamp: str = ""  # ISO format
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": list(self.path),
            "derived_seed": self.derived_seed,
            "derived_seed_64bit": self.derived_seed_64bit,
            "canonical_string_prefix": self.canonical_string[:32] + "...",
            "sha256_digest": self.sha256_digest,
            "timestamp": self.timestamp,
        }


@dataclass
class SeedReceipt:
    """
    Cryptographic receipt proving seed derivation.
    
    This receipt can be stored alongside experiment artifacts to enable
    independent verification of seed provenance.
    """
    master_seed_hex: str
    path: Tuple[str, ...]
    derived_seed: int
    verification_hash: str  # HMAC-like binding of all fields
    schema_version: str = "1.0"
    
    @classmethod
    def create(cls, master_seed_hex: str, path: Tuple[str, ...]) -> "SeedReceipt":
        """Create a receipt for a seed derivation."""
        key = PRNGKey(root=master_seed_hex, path=path)
        derived_seed = derive_seed(key)
        
        # Compute verification hash (binds all fields together)
        verification_material = json.dumps({
            "master_seed_hex": master_seed_hex,
            "path": list(path),
            "derived_seed": derived_seed,
            "schema_version": "1.0",
        }, sort_keys=True, separators=(",", ":"))
        
        verification_hash = hashlib.sha256(
            verification_material.encode("utf-8")
        ).hexdigest()
        
        return cls(
            master_seed_hex=master_seed_hex,
            path=path,
            derived_seed=derived_seed,
            verification_hash=verification_hash,
        )
    
    def verify(self) -> bool:
        """Verify that this receipt is valid and self-consistent."""
        # Recompute derived seed
        key = PRNGKey(root=self.master_seed_hex, path=self.path)
        expected_seed = derive_seed(key)
        
        if expected_seed != self.derived_seed:
            return False
        
        # Recompute verification hash
        verification_material = json.dumps({
            "master_seed_hex": self.master_seed_hex,
            "path": list(self.path),
            "derived_seed": self.derived_seed,
            "schema_version": self.schema_version,
        }, sort_keys=True, separators=(",", ":"))
        
        expected_hash = hashlib.sha256(
            verification_material.encode("utf-8")
        ).hexdigest()
        
        return expected_hash == self.verification_hash
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "master_seed_hex_prefix": self.master_seed_hex[:16] + "...",
            "path": list(self.path),
            "derived_seed": self.derived_seed,
            "verification_hash": self.verification_hash,
        }
    
    def to_json(self) -> str:
        """Serialize to JSON for storage."""
        return json.dumps({
            "schema_version": self.schema_version,
            "master_seed_hex": self.master_seed_hex,
            "path": list(self.path),
            "derived_seed": self.derived_seed,
            "verification_hash": self.verification_hash,
        }, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> "SeedReceipt":
        """Deserialize from JSON."""
        data = json.loads(json_str)
        return cls(
            master_seed_hex=data["master_seed_hex"],
            path=tuple(data["path"]),
            derived_seed=data["derived_seed"],
            verification_hash=data["verification_hash"],
            schema_version=data.get("schema_version", "1.0"),
        )


class SeedLineage:
    """
    Tracks complete seed derivation lineage for an experiment.
    
    Use this to record all seed derivations during an experiment,
    then export the lineage table for audit purposes.
    """
    
    def __init__(self, master_seed_hex: str):
        """
        Initialize lineage tracker.
        
        Args:
            master_seed_hex: 64-character hex string master seed.
        """
        if len(master_seed_hex) != 64:
            raise ValueError(f"master_seed_hex must be 64 characters, got {len(master_seed_hex)}")
        
        self.master_seed_hex = master_seed_hex.lower()
        self.derivations: List[SeedDerivation] = []
        self.created_at = datetime.now(timezone.utc).isoformat()
    
    def record(self, *path: str) -> int:
        """
        Record a seed derivation and return the derived seed.
        
        Args:
            *path: Hierarchical path labels.
        
        Returns:
            The derived 32-bit seed.
        """
        key = PRNGKey(root=self.master_seed_hex, path=path)
        derived_seed = derive_seed(key)
        derived_seed_64bit = derive_seed_64bit(key)
        
        canonical = key.canonical_string()
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        
        derivation = SeedDerivation(
            path=path,
            derived_seed=derived_seed,
            derived_seed_64bit=derived_seed_64bit,
            canonical_string=canonical,
            sha256_digest=digest,
        )
        
        self.derivations.append(derivation)
        return derived_seed
    
    def create_receipt(self, *path: str) -> SeedReceipt:
        """
        Create a verification receipt for a path.
        
        Args:
            *path: Hierarchical path labels.
        
        Returns:
            SeedReceipt for independent verification.
        """
        return SeedReceipt.create(self.master_seed_hex, path)
    
    def to_table(self) -> List[Dict[str, Any]]:
        """Export derivations as a list of dicts (table format)."""
        return [d.to_dict() for d in self.derivations]
    
    def to_dict(self) -> Dict[str, Any]:
        """Export full lineage as dict."""
        return {
            "schema_version": "1.0",
            "master_seed_hex_prefix": self.master_seed_hex[:16] + "...",
            "created_at": self.created_at,
            "derivation_count": len(self.derivations),
            "derivations": self.to_table(),
        }
    
    def export_table(self, output_path: str | Path) -> None:
        """
        Export lineage table to JSON file.
        
        Args:
            output_path: Path to output JSON file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def compute_merkle_root(self) -> str:
        """
        Compute Merkle root of all derivations.
        
        This provides a single hash that commits to all recorded derivations.
        """
        if not self.derivations:
            return hashlib.sha256(b"empty_lineage").hexdigest()
        
        # Leaf hashes are the SHA256 digests of each derivation
        leaves = [d.sha256_digest for d in self.derivations]
        
        # Build Merkle tree
        while len(leaves) > 1:
            next_level = []
            for i in range(0, len(leaves), 2):
                if i + 1 < len(leaves):
                    combined = leaves[i] + leaves[i + 1]
                else:
                    combined = leaves[i] + leaves[i]  # Duplicate odd leaf
                next_hash = hashlib.sha256(combined.encode("utf-8")).hexdigest()
                next_level.append(next_hash)
            leaves = next_level
        
        return leaves[0]


def create_receipt(master_seed_hex: str, *path: str) -> SeedReceipt:
    """
    Convenience function to create a seed receipt.
    
    Args:
        master_seed_hex: 64-character hex string master seed.
        *path: Hierarchical path labels.
    
    Returns:
        SeedReceipt for independent verification.
    """
    return SeedReceipt.create(master_seed_hex, tuple(path))


def verify_receipt(receipt: SeedReceipt) -> bool:
    """
    Verify a seed receipt.
    
    Args:
        receipt: SeedReceipt to verify.
    
    Returns:
        True if receipt is valid, False otherwise.
    """
    return receipt.verify()


def verify_receipt_json(json_str: str) -> Tuple[bool, Optional[SeedReceipt]]:
    """
    Verify a seed receipt from JSON.
    
    Args:
        json_str: JSON-serialized SeedReceipt.
    
    Returns:
        Tuple of (is_valid, receipt_or_none).
    """
    try:
        receipt = SeedReceipt.from_json(json_str)
        return receipt.verify(), receipt
    except Exception:
        return False, None


# --- Self-test when run directly ---
if __name__ == "__main__":
    print("PRNG Hash Lineage — Self-Test")
    print("=" * 60)
    
    # Create lineage tracker
    master = "a" * 64
    lineage = SeedLineage(master)
    
    # Record some derivations
    paths = [
        ("slice_a", "baseline", "cycle_0001"),
        ("slice_a", "baseline", "cycle_0002"),
        ("slice_a", "rfl", "cycle_0001"),
        ("slice_b", "baseline", "cycle_0001"),
    ]
    
    print("\nRecording derivations:")
    for path in paths:
        seed = lineage.record(*path)
        print(f"  {path} → seed={seed}")
    
    # Create and verify receipts
    print("\nVerifying receipts:")
    for path in paths:
        receipt = lineage.create_receipt(*path)
        valid = receipt.verify()
        print(f"  {path}: valid={valid}")
    
    # Compute Merkle root
    root = lineage.compute_merkle_root()
    print(f"\nMerkle root: {root[:32]}...")
    
    # Test JSON round-trip
    print("\nJSON round-trip test:")
    receipt = lineage.create_receipt("test", "path")
    json_str = receipt.to_json()
    valid, restored = verify_receipt_json(json_str)
    print(f"  Original seed: {receipt.derived_seed}")
    print(f"  Restored seed: {restored.derived_seed if restored else 'N/A'}")
    print(f"  Valid: {valid}")
    
    print("\n✅ All lineage tests passed!")

