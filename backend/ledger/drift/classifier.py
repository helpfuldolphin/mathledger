"""
Ledger Drift Radar - Classifier Module

Author: Manus-B (Ledger Replay Architect & Attestation Runtime Engineer)
Phase: III - Drift Radar MVP Implementation
Date: 2025-12-06

Purpose:
    Classify drift signals into actionable categories.
    
    Categories:
    - Benign Schema Evolution (LOW, auto-remediation)
    - Breaking Schema Change (MEDIUM, partial remediation)
    - Hash Algorithm Upgrade (HIGH, auto-remediation)
    - Unintentional Hash Change (HIGH, manual fix)
    - Data Corruption (CRITICAL, restore from backup)
    - Malicious Tampering (CRITICAL, forensic investigation)

Design Principles:
    1. Deterministic: Same signal → same classification
    2. Contextual: Use block history and metadata
    3. Actionable: Provide remediation guidance
    4. Auditable: Log all classification decisions
"""

from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from .scanner import DriftSignal, DriftSignalType, DriftSeverity


# ============================================================================
# DRIFT CATEGORIES
# ============================================================================

class DriftCategory(Enum):
    """Drift categories for classification."""
    BENIGN_SCHEMA_EVOLUTION = "benign_schema_evolution"
    BREAKING_SCHEMA_CHANGE = "breaking_schema_change"
    HASH_ALGORITHM_UPGRADE = "hash_algorithm_upgrade"
    UNINTENTIONAL_HASH_CHANGE = "unintentional_hash_change"
    DATA_CORRUPTION = "data_corruption"
    MALICIOUS_TAMPERING = "malicious_tampering"


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class DriftClassification:
    """
    Represents a classified drift signal.
    
    Attributes:
        signal: Original drift signal
        category: Drift category
        confidence: Classification confidence (0.0-1.0)
        remediation: Remediation guidance
        auto_remediable: Whether drift can be auto-remediated
        metadata: Additional classification metadata
        classified_at: Timestamp when classification was made
    """
    signal: DriftSignal
    category: DriftCategory
    confidence: float
    remediation: str
    auto_remediable: bool
    metadata: Dict[str, Any]
    classified_at: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "signal": self.signal.to_dict(),
            "category": self.category.value,
            "confidence": self.confidence,
            "remediation": self.remediation,
            "auto_remediable": self.auto_remediable,
            "metadata": self.metadata,
            "classified_at": self.classified_at,
        }


# ============================================================================
# CLASSIFICATION FUNCTIONS
# ============================================================================

def classify_schema_drift(
    signal: DriftSignal,
    migration_history: Optional[List[Dict[str, Any]]] = None,
) -> DriftClassification:
    """
    Classify schema drift signal.
    
    Args:
        signal: Schema drift signal
        migration_history: Optional migration history for context
    
    Returns:
        DriftClassification
    
    Classification Logic:
        1. Check if schema change is documented in migration history
        2. Check if change is backward compatible (added fields only)
        3. Check if change is breaking (removed fields, type changes)
        4. Assign category and remediation guidance
    """
    schema_diff = signal.context.get("diff", {})
    
    # Check if documented migration
    is_documented = is_documented_migration(signal, migration_history)
    
    # Check backward compatibility
    is_backward_compatible = (
        not schema_diff.get("removed_fields") and
        not schema_diff.get("type_changes")
    )
    
    if is_documented:
        # Documented migration → benign
        return DriftClassification(
            signal=signal,
            category=DriftCategory.BENIGN_SCHEMA_EVOLUTION,
            confidence=0.9,
            remediation="Documented schema migration. No action required.",
            auto_remediable=True,
            metadata={"is_documented": True, "is_backward_compatible": is_backward_compatible},
            classified_at=datetime.utcnow().isoformat() + "Z",
        )
    elif is_backward_compatible:
        # Backward compatible → benign
        return DriftClassification(
            signal=signal,
            category=DriftCategory.BENIGN_SCHEMA_EVOLUTION,
            confidence=0.8,
            remediation="Backward compatible schema change. Add migration documentation.",
            auto_remediable=True,
            metadata={"is_documented": False, "is_backward_compatible": True},
            classified_at=datetime.utcnow().isoformat() + "Z",
        )
    else:
        # Breaking change
        return DriftClassification(
            signal=signal,
            category=DriftCategory.BREAKING_SCHEMA_CHANGE,
            confidence=0.9,
            remediation="Breaking schema change detected. Add backward compatibility layer or migration script.",
            auto_remediable=False,
            metadata={"is_documented": False, "is_backward_compatible": False, "diff": schema_diff},
            classified_at=datetime.utcnow().isoformat() + "Z",
        )


def classify_hash_delta_drift(
    signal: DriftSignal,
    code_history: Optional[List[Dict[str, Any]]] = None,
) -> DriftClassification:
    """
    Classify hash-delta drift signal.
    
    Args:
        signal: Hash-delta drift signal
        code_history: Optional code change history for context
    
    Returns:
        DriftClassification
    
    Classification Logic:
        1. Check if hash change is intentional (hash algorithm upgrade)
        2. Check if hash change is unintentional (bug in hash computation)
        3. Check if hash change indicates data corruption
        4. Assign category and remediation guidance
    """
    hash_version = signal.context.get("hash_version", "sha256-v1")
    
    # Check if intentional hash upgrade
    is_intentional_upgrade = is_intentional_hash_upgrade(signal, code_history)
    
    # Check for systematic pattern (multiple blocks affected)
    is_systematic = signal.context.get("affected_block_count", 1) > 1
    
    if is_intentional_upgrade:
        # Intentional upgrade → hash algorithm upgrade
        return DriftClassification(
            signal=signal,
            category=DriftCategory.HASH_ALGORITHM_UPGRADE,
            confidence=0.9,
            remediation="Intentional hash algorithm upgrade. Verify migration completed successfully.",
            auto_remediable=True,
            metadata={"is_intentional": True, "hash_version": hash_version},
            classified_at=datetime.utcnow().isoformat() + "Z",
        )
    elif is_systematic:
        # Systematic pattern → unintentional hash change (likely code bug)
        return DriftClassification(
            signal=signal,
            category=DriftCategory.UNINTENTIONAL_HASH_CHANGE,
            confidence=0.8,
            remediation="Unintentional hash change detected. Investigate code changes affecting hash computation.",
            auto_remediable=False,
            metadata={"is_intentional": False, "is_systematic": True},
            classified_at=datetime.utcnow().isoformat() + "Z",
        )
    else:
        # Isolated case → data corruption
        return DriftClassification(
            signal=signal,
            category=DriftCategory.DATA_CORRUPTION,
            confidence=0.7,
            remediation="Isolated hash mismatch. Investigate database integrity and restore from backup if needed.",
            auto_remediable=False,
            metadata={"is_intentional": False, "is_systematic": False},
            classified_at=datetime.utcnow().isoformat() + "Z",
        )


def classify_metadata_drift(signal: DriftSignal) -> DriftClassification:
    """
    Classify metadata drift signal.
    
    Args:
        signal: Metadata drift signal
    
    Returns:
        DriftClassification
    
    Classification Logic:
        Metadata changes are usually benign (LOW severity).
        Only breaking changes (removed fields) require action.
    """
    schema_diff = signal.context.get("diff", {})
    
    is_backward_compatible = (
        not schema_diff.get("removed_fields") and
        not schema_diff.get("type_changes")
    )
    
    if is_backward_compatible:
        return DriftClassification(
            signal=signal,
            category=DriftCategory.BENIGN_SCHEMA_EVOLUTION,
            confidence=0.9,
            remediation="Backward compatible metadata change. No action required.",
            auto_remediable=True,
            metadata={"is_backward_compatible": True},
            classified_at=datetime.utcnow().isoformat() + "Z",
        )
    else:
        return DriftClassification(
            signal=signal,
            category=DriftCategory.BREAKING_SCHEMA_CHANGE,
            confidence=0.8,
            remediation="Breaking metadata change. Add backward compatibility layer.",
            auto_remediable=False,
            metadata={"is_backward_compatible": False, "diff": schema_diff},
            classified_at=datetime.utcnow().isoformat() + "Z",
        )


def classify_statement_drift(signal: DriftSignal) -> DriftClassification:
    """
    Classify statement drift signal.
    
    Args:
        signal: Statement drift signal
    
    Returns:
        DriftClassification
    
    Classification Logic:
        Statement format changes are HIGH severity (affect proof verification).
    """
    return DriftClassification(
        signal=signal,
        category=DriftCategory.UNINTENTIONAL_HASH_CHANGE,
        confidence=0.8,
        remediation="Statement format change detected. Investigate canonicalization logic.",
        auto_remediable=False,
        metadata={},
        classified_at=datetime.utcnow().isoformat() + "Z",
    )


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def is_documented_migration(
    signal: DriftSignal,
    migration_history: Optional[List[Dict[str, Any]]],
) -> bool:
    """
    Check if drift signal corresponds to documented migration.
    
    Args:
        signal: Drift signal
        migration_history: Migration history
    
    Returns:
        True if documented, False otherwise
    """
    if not migration_history:
        return False
    
    # Check if block_number matches any migration
    for migration in migration_history:
        migration_block = migration.get("block_number")
        if migration_block == signal.block_number:
            return True
    
    return False


def is_intentional_hash_upgrade(
    signal: DriftSignal,
    code_history: Optional[List[Dict[str, Any]]],
) -> bool:
    """
    Check if hash change is intentional upgrade.
    
    Args:
        signal: Drift signal
        code_history: Code change history
    
    Returns:
        True if intentional, False otherwise
    """
    if not code_history:
        return False
    
    # Check if hash_version changed
    hash_version = signal.context.get("hash_version", "sha256-v1")
    
    # Check if code history contains hash algorithm upgrade
    for change in code_history:
        if "hash_version" in change.get("description", ""):
            return True
    
    return False


# ============================================================================
# DRIFT CLASSIFIER ORCHESTRATOR
# ============================================================================

class DriftClassifier:
    """
    Orchestrates drift signal classification.
    
    Usage:
        classifier = DriftClassifier()
        classification = classifier.classify(signal)
        report = classifier.generate_report()
    """
    
    def __init__(
        self,
        migration_history: Optional[List[Dict[str, Any]]] = None,
        code_history: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Initialize drift classifier.
        
        Args:
            migration_history: Optional migration history for context
            code_history: Optional code change history for context
        """
        self.migration_history = migration_history or []
        self.code_history = code_history or []
        self.classifications: List[DriftClassification] = []
    
    def classify(self, signal: DriftSignal) -> DriftClassification:
        """
        Classify drift signal.
        
        Args:
            signal: Drift signal to classify
        
        Returns:
            DriftClassification
        """
        # Classify based on signal type
        if signal.signal_type == DriftSignalType.SCHEMA_DRIFT:
            classification = classify_schema_drift(signal, self.migration_history)
        elif signal.signal_type == DriftSignalType.HASH_DELTA_DRIFT:
            classification = classify_hash_delta_drift(signal, self.code_history)
        elif signal.signal_type == DriftSignalType.METADATA_DRIFT:
            classification = classify_metadata_drift(signal)
        elif signal.signal_type == DriftSignalType.STATEMENT_DRIFT:
            classification = classify_statement_drift(signal)
        else:
            # Unknown signal type
            classification = DriftClassification(
                signal=signal,
                category=DriftCategory.DATA_CORRUPTION,
                confidence=0.5,
                remediation="Unknown drift signal type. Manual investigation required.",
                auto_remediable=False,
                metadata={},
                classified_at=datetime.utcnow().isoformat() + "Z",
            )
        
        # Store classification
        self.classifications.append(classification)
        
        return classification
    
    def classify_batch(self, signals: List[DriftSignal]) -> List[DriftClassification]:
        """
        Classify multiple drift signals.
        
        Args:
            signals: List of drift signals
        
        Returns:
            List of DriftClassification objects
        """
        return [self.classify(signal) for signal in signals]
    
    def get_classifications_by_category(self, category: DriftCategory) -> List[DriftClassification]:
        """Get classifications by category."""
        return [c for c in self.classifications if c.category == category]
    
    def get_auto_remediable_classifications(self) -> List[DriftClassification]:
        """Get auto-remediable classifications."""
        return [c for c in self.classifications if c.auto_remediable]
    
    def get_manual_classifications(self) -> List[DriftClassification]:
        """Get classifications requiring manual remediation."""
        return [c for c in self.classifications if not c.auto_remediable]
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Generate classification report.
        
        Returns:
            Report dictionary
        """
        # Count by category
        by_category = {}
        for category in DriftCategory:
            count = len(self.get_classifications_by_category(category))
            if count > 0:
                by_category[category.value] = count
        
        return {
            "total_classifications": len(self.classifications),
            "by_category": by_category,
            "auto_remediable_count": len(self.get_auto_remediable_classifications()),
            "manual_count": len(self.get_manual_classifications()),
            "average_confidence": sum(c.confidence for c in self.classifications) / len(self.classifications) if self.classifications else 0.0,
        }
