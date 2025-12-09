"""
Ledger Drift Radar - Forensics Module

Author: Manus-B (Ledger Replay Architect & Attestation Runtime Engineer)
Phase: III - Drift Radar MVP Implementation
Date: 2025-12-06

Purpose:
    Collect forensic artifacts for drift investigation.
    
    Artifacts collected:
    - Block snapshot (full block data)
    - Replay trace (recomputed roots, intermediate hashes)
    - Code context (git SHA, recent commits)
    - Environment context (DB schema, Python version)

Design Principles:
    1. Comprehensive: Capture all relevant context
    2. Non-invasive: Read-only operations
    3. Structured: Consistent artifact format
    4. Portable: Artifacts can be shared/archived
"""

from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import os
import sys

from .scanner import DriftSignal
from .classifier import DriftClassification


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ForensicArtifact:
    """
    Represents a forensic artifact bundle.
    
    Attributes:
        artifact_id: Unique artifact ID
        signal: Drift signal being investigated
        classification: Drift classification (if available)
        block_snapshot: Full block data
        replay_trace: Replay verification trace
        code_context: Git context
        environment_context: Environment metadata
        collected_at: Timestamp when artifact was collected
    """
    artifact_id: str
    signal: DriftSignal
    classification: Optional[DriftClassification]
    block_snapshot: Dict[str, Any]
    replay_trace: Optional[Dict[str, Any]]
    code_context: Dict[str, Any]
    environment_context: Dict[str, Any]
    collected_at: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "artifact_id": self.artifact_id,
            "signal": self.signal.to_dict(),
            "classification": self.classification.to_dict() if self.classification else None,
            "block_snapshot": self.block_snapshot,
            "replay_trace": self.replay_trace,
            "code_context": self.code_context,
            "environment_context": self.environment_context,
            "collected_at": self.collected_at,
        }


# ============================================================================
# ARTIFACT COLLECTION FUNCTIONS
# ============================================================================

def capture_block_snapshot(
    block_id: int,
    db_connection: Any = None,
) -> Dict[str, Any]:
    """
    Capture full block snapshot.
    
    Args:
        block_id: Block ID to capture
        db_connection: Database connection (placeholder)
    
    Returns:
        Block snapshot dictionary
    
    Snapshot Contents:
        - All block fields
        - Stored attestation roots
        - Canonical payloads
        - Metadata
    """
    # Placeholder: In production, fetch from database
    snapshot = {
        "block_id": block_id,
        "captured_at": datetime.utcnow().isoformat() + "Z",
        "fields": {},  # All block fields
        "attestation_roots": {
            "reasoning_attestation_root": "",
            "ui_attestation_root": "",
            "composite_attestation_root": "",
        },
        "canonical_payloads": {
            "canonical_proofs": {},
            "canonical_statements": [],
        },
        "metadata": {},
    }
    
    return snapshot


def capture_replay_trace(
    block_id: int,
    replay_engine: Any = None,
) -> Optional[Dict[str, Any]]:
    """
    Capture replay verification trace.
    
    Args:
        block_id: Block ID to replay
        replay_engine: Replay engine instance (placeholder)
    
    Returns:
        Replay trace dictionary or None
    
    Trace Contents:
        - Recomputed attestation roots
        - Intermediate hashes
        - Merkle tree structure
        - Comparison with stored roots
    """
    # Placeholder: In production, run replay verification
    trace = {
        "block_id": block_id,
        "replayed_at": datetime.utcnow().isoformat() + "Z",
        "recomputed_roots": {
            "r_t": "",
            "u_t": "",
            "h_t": "",
        },
        "stored_roots": {
            "r_t": "",
            "u_t": "",
            "h_t": "",
        },
        "matches": {
            "r_t": False,
            "u_t": False,
            "h_t": False,
        },
        "intermediate_hashes": [],
        "merkle_tree": {},
    }
    
    return trace


def capture_code_context() -> Dict[str, Any]:
    """
    Capture code context (git SHA, recent commits).
    
    Returns:
        Code context dictionary
    
    Context Contents:
        - Current git SHA
        - Recent commits affecting hash code
        - Branch name
        - Uncommitted changes
    """
    context = {
        "captured_at": datetime.utcnow().isoformat() + "Z",
        "git_sha": get_git_sha(),
        "git_branch": get_git_branch(),
        "recent_commits": get_recent_commits(count=10),
        "uncommitted_changes": has_uncommitted_changes(),
    }
    
    return context


def capture_environment_context() -> Dict[str, Any]:
    """
    Capture environment context (DB schema, Python version).
    
    Returns:
        Environment context dictionary
    
    Context Contents:
        - Python version
        - Database schema version
        - Installed packages
        - System info
    """
    context = {
        "captured_at": datetime.utcnow().isoformat() + "Z",
        "python_version": sys.version,
        "python_executable": sys.executable,
        "db_schema_version": get_db_schema_version(),
        "installed_packages": get_installed_packages(),
        "system_info": {
            "platform": sys.platform,
            "cwd": os.getcwd(),
        },
    }
    
    return context


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_git_sha() -> str:
    """Get current git commit SHA."""
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def get_git_branch() -> str:
    """Get current git branch."""
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def get_recent_commits(count: int = 10) -> List[Dict[str, str]]:
    """Get recent git commits."""
    try:
        import subprocess
        result = subprocess.run(
            ["git", "log", f"-{count}", "--pretty=format:%H|%an|%ad|%s"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return []
        
        commits = []
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            parts = line.split("|", 3)
            if len(parts) == 4:
                commits.append({
                    "sha": parts[0],
                    "author": parts[1],
                    "date": parts[2],
                    "message": parts[3],
                })
        return commits
    except Exception:
        return []


def has_uncommitted_changes() -> bool:
    """Check if there are uncommitted changes."""
    try:
        import subprocess
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return bool(result.stdout.strip()) if result.returncode == 0 else False
    except Exception:
        return False


def get_db_schema_version() -> str:
    """Get database schema version."""
    # Placeholder: In production, query database
    return "unknown"


def get_installed_packages() -> List[str]:
    """Get list of installed Python packages."""
    try:
        import subprocess
        result = subprocess.run(
            ["pip3", "list", "--format=freeze"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip().split("\n")
        else:
            return []
    except Exception:
        return []


# ============================================================================
# FORENSIC COLLECTOR ORCHESTRATOR
# ============================================================================

class ForensicCollector:
    """
    Orchestrates forensic artifact collection.
    
    Usage:
        collector = ForensicCollector()
        artifact = collector.collect(signal, classification)
        collector.export(artifact, "artifacts/drift_001.json")
    """
    
    def __init__(
        self,
        db_connection: Any = None,
        replay_engine: Any = None,
    ):
        """
        Initialize forensic collector.
        
        Args:
            db_connection: Database connection (placeholder)
            replay_engine: Replay engine instance (placeholder)
        """
        self.db_connection = db_connection
        self.replay_engine = replay_engine
        self.artifacts: List[ForensicArtifact] = []
    
    def collect(
        self,
        signal: DriftSignal,
        classification: Optional[DriftClassification] = None,
    ) -> ForensicArtifact:
        """
        Collect forensic artifact for drift signal.
        
        Args:
            signal: Drift signal
            classification: Optional drift classification
        
        Returns:
            ForensicArtifact
        """
        # Generate artifact ID
        artifact_id = generate_artifact_id(signal)
        
        # Capture block snapshot
        block_snapshot = capture_block_snapshot(signal.block_id, self.db_connection)
        
        # Capture replay trace (if available)
        replay_trace = None
        if self.replay_engine:
            replay_trace = capture_replay_trace(signal.block_id, self.replay_engine)
        
        # Capture code context
        code_context = capture_code_context()
        
        # Capture environment context
        environment_context = capture_environment_context()
        
        # Create artifact
        artifact = ForensicArtifact(
            artifact_id=artifact_id,
            signal=signal,
            classification=classification,
            block_snapshot=block_snapshot,
            replay_trace=replay_trace,
            code_context=code_context,
            environment_context=environment_context,
            collected_at=datetime.utcnow().isoformat() + "Z",
        )
        
        # Store artifact
        self.artifacts.append(artifact)
        
        return artifact
    
    def collect_batch(
        self,
        signals: List[DriftSignal],
        classifications: Optional[List[DriftClassification]] = None,
    ) -> List[ForensicArtifact]:
        """
        Collect forensic artifacts for multiple signals.
        
        Args:
            signals: List of drift signals
            classifications: Optional list of classifications
        
        Returns:
            List of ForensicArtifact objects
        """
        artifacts = []
        for i, signal in enumerate(signals):
            classification = classifications[i] if classifications and i < len(classifications) else None
            artifact = self.collect(signal, classification)
            artifacts.append(artifact)
        return artifacts
    
    def export(self, artifact: ForensicArtifact, filepath: str):
        """
        Export artifact to JSON file.
        
        Args:
            artifact: ForensicArtifact to export
            filepath: Output file path
        """
        with open(filepath, 'w') as f:
            json.dump(artifact.to_dict(), f, indent=2)
    
    def export_batch(self, artifacts: List[ForensicArtifact], directory: str):
        """
        Export multiple artifacts to directory.
        
        Args:
            artifacts: List of ForensicArtifact objects
            directory: Output directory
        """
        os.makedirs(directory, exist_ok=True)
        for artifact in artifacts:
            filepath = os.path.join(directory, f"{artifact.artifact_id}.json")
            self.export(artifact, filepath)
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Generate forensic collection report.
        
        Returns:
            Report dictionary
        """
        return {
            "total_artifacts": len(self.artifacts),
            "artifacts_with_replay_trace": sum(1 for a in self.artifacts if a.replay_trace),
            "artifacts_with_classification": sum(1 for a in self.artifacts if a.classification),
        }


def generate_artifact_id(signal: DriftSignal) -> str:
    """
    Generate unique artifact ID.
    
    Args:
        signal: Drift signal
    
    Returns:
        Artifact ID (e.g., "drift_20251206_001_schema_123")
    """
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    signal_type = signal.signal_type.value.split("_")[0]  # "schema", "hash", etc.
    block_number = signal.block_number
    return f"drift_{timestamp}_{signal_type}_{block_number}"
