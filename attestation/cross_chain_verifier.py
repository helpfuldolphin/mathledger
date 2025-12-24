"""
Cross-Chain Attestation Verifier
==================================

Higher-order verifier that validates multiple evidence chains across experiments.

This module implements:
1. Cross-experiment attestation fusion: Detects chain discontinuities, repeated
   experiment IDs, prev_hash inconsistencies, and composite chain ordering.
2. Attestation drift radar: Detects hash drift, schema drift, mismatched dual-roots,
   and timestamp monotonicity violations.

Phase III → Phase IV transition: Evidence Chain Ledger → Chain-of-Chains Governance
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from datetime import datetime

from attestation.dual_root import verify_composite_integrity


@dataclass
class ChainDiscontinuity:
    """Represents a break in the chain of attestations."""
    
    experiment_id: str
    expected_prev_hash: Optional[str]
    actual_prev_hash: Optional[str]
    position: int
    
    def __str__(self) -> str:
        return (
            f"Chain discontinuity in {self.experiment_id} at position {self.position}: "
            f"expected prev_hash={self.expected_prev_hash}, got {self.actual_prev_hash}"
        )


@dataclass
class DuplicateExperiment:
    """Represents a repeated experiment ID."""
    
    experiment_id: str
    occurrences: List[int]
    
    def __str__(self) -> str:
        return f"Duplicate experiment ID '{self.experiment_id}' at positions {self.occurrences}"


@dataclass
class HashDrift:
    """Represents hash drift across repeated experiments."""
    
    experiment_id: str
    run_1: int
    run_2: int
    hash_field: str
    hash_1: str
    hash_2: str
    
    def __str__(self) -> str:
        return (
            f"Hash drift in {self.experiment_id}: {self.hash_field} differs "
            f"between run {self.run_1} ({self.hash_1[:16]}...) "
            f"and run {self.run_2} ({self.hash_2[:16]}...)"
        )


@dataclass
class SchemaDrift:
    """Represents schema drift in evidence packs."""
    
    experiment_id: str
    missing_fields: Set[str]
    extra_fields: Set[str]
    
    def __str__(self) -> str:
        issues = []
        if self.missing_fields:
            issues.append(f"missing fields: {sorted(self.missing_fields)}")
        if self.extra_fields:
            issues.append(f"extra fields: {sorted(self.extra_fields)}")
        return f"Schema drift in {self.experiment_id}: {', '.join(issues)}"


@dataclass
class DualRootMismatch:
    """Represents mismatched dual-root attestation."""
    
    experiment_id: str
    r_t: str
    u_t: str
    h_t: str
    recomputed_h_t: str
    
    def __str__(self) -> str:
        return (
            f"Dual-root mismatch in {self.experiment_id}: "
            f"H_t={self.h_t[:16]}... does not match SHA256(R_t || U_t)={self.recomputed_h_t[:16]}..."
        )


@dataclass
class TimestampViolation:
    """Represents timestamp monotonicity violation."""
    
    experiment_id_1: str
    experiment_id_2: str
    timestamp_1: str
    timestamp_2: str
    position_1: int
    position_2: int
    
    def __str__(self) -> str:
        return (
            f"Timestamp monotonicity violation: {self.experiment_id_1} "
            f"(pos {self.position_1}, {self.timestamp_1}) occurs after "
            f"{self.experiment_id_2} (pos {self.position_2}, {self.timestamp_2})"
        )


@dataclass
class ChainVerificationResult:
    """Result of cross-chain verification."""
    
    total_experiments: int = 0
    valid_experiments: int = 0
    chain_discontinuities: List[ChainDiscontinuity] = field(default_factory=list)
    duplicate_experiments: List[DuplicateExperiment] = field(default_factory=list)
    hash_drifts: List[HashDrift] = field(default_factory=list)
    schema_drifts: List[SchemaDrift] = field(default_factory=list)
    dual_root_mismatches: List[DualRootMismatch] = field(default_factory=list)
    timestamp_violations: List[TimestampViolation] = field(default_factory=list)
    
    @property
    def is_valid(self) -> bool:
        """Returns True if no critical issues found."""
        return not any([
            self.chain_discontinuities,
            self.duplicate_experiments,
            self.dual_root_mismatches,
        ])
    
    @property
    def has_warnings(self) -> bool:
        """Returns True if non-critical issues found."""
        return any([
            self.hash_drifts,
            self.schema_drifts,
            self.timestamp_violations,
        ])
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Cross-Chain Verification Report",
            f"================================",
            f"Total experiments: {self.total_experiments}",
            f"Valid experiments: {self.valid_experiments}",
            f"",
        ]
        
        if self.is_valid and not self.has_warnings:
            lines.append("✓ All experiments verified successfully")
            return "\n".join(lines)
        
        if self.chain_discontinuities:
            lines.append(f"CRITICAL: {len(self.chain_discontinuities)} chain discontinuities:")
            for issue in self.chain_discontinuities:
                lines.append(f"  - {issue}")
            lines.append("")
        
        if self.duplicate_experiments:
            lines.append(f"CRITICAL: {len(self.duplicate_experiments)} duplicate experiment IDs:")
            for issue in self.duplicate_experiments:
                lines.append(f"  - {issue}")
            lines.append("")
        
        if self.dual_root_mismatches:
            lines.append(f"CRITICAL: {len(self.dual_root_mismatches)} dual-root mismatches:")
            for issue in self.dual_root_mismatches:
                lines.append(f"  - {issue}")
            lines.append("")
        
        if self.hash_drifts:
            lines.append(f"WARNING: {len(self.hash_drifts)} hash drift issues:")
            for issue in self.hash_drifts:
                lines.append(f"  - {issue}")
            lines.append("")
        
        if self.schema_drifts:
            lines.append(f"WARNING: {len(self.schema_drifts)} schema drift issues:")
            for issue in self.schema_drifts:
                lines.append(f"  - {issue}")
            lines.append("")
        
        if self.timestamp_violations:
            lines.append(f"WARNING: {len(self.timestamp_violations)} timestamp violations:")
            for issue in self.timestamp_violations:
                lines.append(f"  - {issue}")
            lines.append("")
        
        return "\n".join(lines)


class CrossChainVerifier:
    """
    Cross-experiment attestation verifier.
    
    Validates evidence chains across multiple experiments, detecting:
    - Chain discontinuities (broken prev_hash links)
    - Repeated experiment IDs
    - Composite chain ordering violations
    - Hash drift across repeated experiments
    - Evidence pack schema drift
    - Mismatched dual-roots (H_t != SHA256(R_t || U_t))
    - Timestamp monotonicity violations
    """
    
    # Expected schema fields for attestation manifests
    REQUIRED_FIELDS = {
        'experiment_id',
        'manifest_version',
        'timestamp_utc',
    }
    
    OPTIONAL_FIELDS = {
        'run_index',
        'prev_hash',
        'reasoning_merkle_root',
        'ui_merkle_root',
        'composite_attestation_root',
        'provenance',
        'configuration',
        'execution',
        'artifacts',
        'results',
    }
    
    def __init__(self, strict_schema: bool = False):
        """
        Initialize verifier.
        
        Args:
            strict_schema: If True, flag extra fields as schema drift
        """
        self.strict_schema = strict_schema
    
    def verify_chain(
        self,
        manifests: List[Dict[str, Any]],
        check_ordering: bool = True,
    ) -> ChainVerificationResult:
        """
        Verify a chain of experiment manifests.
        
        Args:
            manifests: List of attestation manifests in chronological order
            check_ordering: If True, verify timestamp ordering
            
        Returns:
            ChainVerificationResult with detected issues
        """
        result = ChainVerificationResult()
        result.total_experiments = len(manifests)
        
        if not manifests:
            return result
        
        # Track experiment IDs for duplicate detection
        experiment_ids: Dict[str, List[int]] = {}
        
        # Track previous hash for chain continuity
        prev_hash: Optional[str] = None
        
        # Track timestamps for monotonicity
        prev_timestamp: Optional[str] = None
        prev_experiment_id: Optional[str] = None
        
        # Track configuration hashes for drift detection
        config_hashes: Dict[str, List[Tuple[int, str]]] = {}
        
        for idx, manifest in enumerate(manifests):
            experiment_id = manifest.get('experiment_id', f'<unknown_{idx}>')
            
            # Track experiment IDs
            if experiment_id not in experiment_ids:
                experiment_ids[experiment_id] = []
            experiment_ids[experiment_id].append(idx)
            
            # Check schema
            schema_issues = self._check_schema(manifest)
            if schema_issues:
                result.schema_drifts.append(
                    SchemaDrift(
                        experiment_id=experiment_id,
                        missing_fields=schema_issues[0],
                        extra_fields=schema_issues[1],
                    )
                )
            
            # Check chain continuity
            if idx > 0:
                actual_prev_hash = manifest.get('prev_hash')
                if actual_prev_hash != prev_hash:
                    result.chain_discontinuities.append(
                        ChainDiscontinuity(
                            experiment_id=experiment_id,
                            expected_prev_hash=prev_hash,
                            actual_prev_hash=actual_prev_hash,
                            position=idx,
                        )
                    )
            
            # Check dual-root integrity
            if all(k in manifest for k in ['reasoning_merkle_root', 'ui_merkle_root', 'composite_attestation_root']):
                r_t = manifest['reasoning_merkle_root']
                u_t = manifest['ui_merkle_root']
                h_t = manifest['composite_attestation_root']
                
                if not verify_composite_integrity(r_t, u_t, h_t):
                    # Recompute for diagnostic
                    recomputed = hashlib.sha256(f"{r_t}{u_t}".encode('ascii')).hexdigest()
                    result.dual_root_mismatches.append(
                        DualRootMismatch(
                            experiment_id=experiment_id,
                            r_t=r_t,
                            u_t=u_t,
                            h_t=h_t,
                            recomputed_h_t=recomputed,
                        )
                    )
            
            # Check timestamp monotonicity
            if check_ordering and prev_timestamp:
                current_timestamp = manifest.get('timestamp_utc')
                if current_timestamp and prev_timestamp:
                    try:
                        curr_dt = datetime.fromisoformat(current_timestamp.replace('Z', '+00:00'))
                        prev_dt = datetime.fromisoformat(prev_timestamp.replace('Z', '+00:00'))
                        if curr_dt < prev_dt:
                            result.timestamp_violations.append(
                                TimestampViolation(
                                    experiment_id_1=experiment_id,
                                    experiment_id_2=prev_experiment_id or '<unknown>',
                                    timestamp_1=current_timestamp,
                                    timestamp_2=prev_timestamp,
                                    position_1=idx,
                                    position_2=idx - 1,
                                )
                            )
                    except (ValueError, AttributeError):
                        # Skip invalid timestamps
                        pass
            
            # Track configuration hashes for drift detection
            if 'configuration' in manifest:
                config = manifest['configuration']
                if 'snapshot' in config:
                    config_hash = self._compute_config_hash(config['snapshot'])
                    config_key = experiment_id
                    if config_key not in config_hashes:
                        config_hashes[config_key] = []
                    config_hashes[config_key].append((idx, config_hash))
            
            # Update tracking variables
            manifest_hash = self._compute_manifest_hash(manifest)
            prev_hash = manifest_hash
            prev_timestamp = manifest.get('timestamp_utc')
            prev_experiment_id = experiment_id
        
        # Detect duplicate experiment IDs
        for exp_id, positions in experiment_ids.items():
            if len(positions) > 1:
                result.duplicate_experiments.append(
                    DuplicateExperiment(
                        experiment_id=exp_id,
                        occurrences=positions,
                    )
                )
        
        # Detect hash drift
        for exp_id, hashes in config_hashes.items():
            if len(hashes) > 1:
                # Check if all hashes match
                unique_hashes = set(h for _, h in hashes)
                if len(unique_hashes) > 1:
                    # Report drift between first two different hashes
                    # Build map of hash -> list of indices
                    hash_indices: Dict[str, List[int]] = {}
                    for idx, h in hashes:
                        if h not in hash_indices:
                            hash_indices[h] = []
                        hash_indices[h].append(idx)
                    
                    sorted_hashes = sorted(unique_hashes)
                    result.hash_drifts.append(
                        HashDrift(
                            experiment_id=exp_id,
                            run_1=hash_indices[sorted_hashes[0]][0],
                            run_2=hash_indices[sorted_hashes[1]][0],
                            hash_field='configuration.snapshot',
                            hash_1=sorted_hashes[0],
                            hash_2=sorted_hashes[1],
                        )
                    )
        
        # Count valid experiments (no critical issues)
        # For duplicate experiments, count unique experiment IDs with issues
        duplicate_exp_ids = set(dup.experiment_id for dup in result.duplicate_experiments)
        discontinuity_exp_ids = set(disc.experiment_id for disc in result.chain_discontinuities)
        mismatch_exp_ids = set(mm.experiment_id for mm in result.dual_root_mismatches)
        
        invalid_exp_ids = duplicate_exp_ids | discontinuity_exp_ids | mismatch_exp_ids
        result.valid_experiments = result.total_experiments - len(invalid_exp_ids)
        
        return result
    
    def verify_artifacts_directory(
        self,
        artifacts_dir: Path,
        manifest_pattern: str = "**/attestation.json",
    ) -> ChainVerificationResult:
        """
        Verify all attestation manifests in an artifacts directory.
        
        Args:
            artifacts_dir: Path to artifacts directory
            manifest_pattern: Glob pattern for manifest files
            
        Returns:
            ChainVerificationResult
        """
        manifest_files = sorted(artifacts_dir.glob(manifest_pattern))
        manifests = []
        
        for manifest_file in manifest_files:
            try:
                with open(manifest_file, 'r') as f:
                    manifest = json.load(f)
                    manifests.append(manifest)
            except (json.JSONDecodeError, OSError) as e:
                # Skip invalid files
                continue
        
        return self.verify_chain(manifests)
    
    def _check_schema(self, manifest: Dict[str, Any]) -> Optional[Tuple[Set[str], Set[str]]]:
        """
        Check manifest schema.
        
        Returns:
            Tuple of (missing_fields, extra_fields) or None if valid
        """
        manifest_keys = set(manifest.keys())
        missing = self.REQUIRED_FIELDS - manifest_keys
        
        if self.strict_schema:
            allowed = self.REQUIRED_FIELDS | self.OPTIONAL_FIELDS
            extra = manifest_keys - allowed
        else:
            extra = set()
        
        if missing or extra:
            return (missing, extra)
        return None
    
    def _compute_manifest_hash(self, manifest: Dict[str, Any]) -> str:
        """
        Compute deterministic hash of manifest.
        
        Excludes 'prev_hash' field to avoid circular dependency.
        """
        # Create a copy without prev_hash
        manifest_copy = {k: v for k, v in manifest.items() if k != 'prev_hash'}
        
        # Compute hash using canonical JSON
        canonical = json.dumps(manifest_copy, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(canonical.encode('utf-8')).hexdigest()
    
    def _compute_config_hash(self, config: Dict[str, Any]) -> str:
        """Compute hash of configuration snapshot."""
        canonical = json.dumps(config, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


__all__ = [
    'CrossChainVerifier',
    'ChainVerificationResult',
    'ChainDiscontinuity',
    'DuplicateExperiment',
    'HashDrift',
    'SchemaDrift',
    'DualRootMismatch',
    'TimestampViolation',
]
