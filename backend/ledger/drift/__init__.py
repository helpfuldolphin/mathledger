"""
Ledger Drift Radar

Author: Manus-B (Ledger Replay Architect & Attestation Runtime Engineer)
Phase: III - Drift Radar MVP Implementation
Date: 2025-12-06

Purpose:
    Detect, classify, and investigate ledger drift.

Modules:
    - scanner: Detect drift signals
    - classifier: Classify drift signals
    - forensics: Collect forensic artifacts
    - dashboard: Visualize drift data (stub)
"""

from .scanner import (
    DriftSignal,
    DriftSignalType,
    DriftSeverity,
    DriftScanner,
    detect_schema_drift,
    detect_hash_delta_drift,
    detect_metadata_drift,
    detect_statement_drift,
)

from .classifier import (
    DriftClassification,
    DriftCategory,
    DriftClassifier,
    classify_schema_drift,
    classify_hash_delta_drift,
    classify_metadata_drift,
    classify_statement_drift,
)

from .forensics import (
    ForensicArtifact,
    ForensicCollector,
    capture_block_snapshot,
    capture_replay_trace,
    capture_code_context,
    capture_environment_context,
)

from .dashboard import (
    DashboardState,
    DriftDashboard,
    export_prometheus_metrics,
)

__all__ = [
    # Scanner
    "DriftSignal",
    "DriftSignalType",
    "DriftSeverity",
    "DriftScanner",
    "detect_schema_drift",
    "detect_hash_delta_drift",
    "detect_metadata_drift",
    "detect_statement_drift",
    # Classifier
    "DriftClassification",
    "DriftCategory",
    "DriftClassifier",
    "classify_schema_drift",
    "classify_hash_delta_drift",
    "classify_metadata_drift",
    "classify_statement_drift",
    # Forensics
    "ForensicArtifact",
    "ForensicCollector",
    "capture_block_snapshot",
    "capture_replay_trace",
    "capture_code_context",
    "capture_environment_context",
    # Dashboard
    "DashboardState",
    "DriftDashboard",
    "export_prometheus_metrics",
]
