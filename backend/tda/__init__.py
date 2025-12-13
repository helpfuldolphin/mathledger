"""
TDA (Topological Data Analysis) Mind Scanner Module

This module provides the four core TDA metrics for cognitive system stability:
- SNS: Structural Novelty Score
- PCS: Proof Coherence Score
- DRS: Drift Rate Score
- HSS: Homological Stability Score

See: docs/system_law/TDA_PhaseX_Binding.md for full specification.

SHADOW MODE CONTRACT:
- All metrics are observational only
- No governance modification based on TDA values
- No abort enforcement triggered by TDA thresholds
- Metrics logged for analysis and future policy development

Status: P3/P4 IMPLEMENTATION (SHADOW-ONLY)
"""

from backend.tda.metrics import (
    SNSComputer,
    PCSComputer,
    DRSComputer,
    HSSComputer,
    TDAMetrics,
    TDAWindowMetrics,
    compute_sns,
    compute_pcs,
    compute_drs,
    compute_hss,
)
from backend.tda.monitor import TDAMonitor
from backend.tda.evidence import (
    attach_tda_to_evidence,
    format_tda_evidence_summary,
    compute_topology_match_score,
    TDAEvidenceBlock,
)
from backend.tda.pattern_classifier import (
    RTTSPattern,
    PatternClassification,
    TDAPatternClassifier,
    attach_tda_patterns_to_evidence,
)
from backend.tda.patterns_from_windows import (
    classify_windows,
    aggregate_pattern_summary,
    attach_windowed_patterns_to_evidence,
    extract_windowed_patterns_status,
    get_top_events_digest,
    attach_signals_tda_windowed_patterns,
    WindowPatternResult,
    PatternAggregateSummary,
    WindowedPatternsStatus,
    TopEventDigest,
)

__all__ = [
    # Computers
    "SNSComputer",
    "PCSComputer",
    "DRSComputer",
    "HSSComputer",
    # Data structures
    "TDAMetrics",
    "TDAWindowMetrics",
    # Functions
    "compute_sns",
    "compute_pcs",
    "compute_drs",
    "compute_hss",
    # Monitor
    "TDAMonitor",
    # Evidence Pack integration
    "attach_tda_to_evidence",
    "format_tda_evidence_summary",
    "compute_topology_match_score",
    "TDAEvidenceBlock",
    # Pattern Classifier
    "RTTSPattern",
    "PatternClassification",
    "TDAPatternClassifier",
    "attach_tda_patterns_to_evidence",
    # Windowed Pattern Classification
    "classify_windows",
    "aggregate_pattern_summary",
    "attach_windowed_patterns_to_evidence",
    "extract_windowed_patterns_status",
    "get_top_events_digest",
    "attach_signals_tda_windowed_patterns",
    "WindowPatternResult",
    "PatternAggregateSummary",
    "WindowedPatternsStatus",
    "TopEventDigest",
]
