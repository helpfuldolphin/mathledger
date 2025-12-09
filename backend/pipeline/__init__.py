"""
Pipeline health and degradation management.

PHASE II — NOT RUN IN PHASE I
No uplift claims are made.
"""

from backend.pipeline.topology_health import (
    PipelineMode,
    GovernanceLabel,
    NodeStatus,
    NodeHealth,
    HealthSignals,
    SliceResult,
    ValidationStatus,
    IntegrityCheck,
    PipelineHealth,
    DegradationDecision,
    FailurePattern,
    TopologyHealthEvaluator,
    DegradationPolicyEngine,
)

__all__ = [
    "PipelineMode",
    "GovernanceLabel",
    "NodeStatus",
    "NodeHealth",
    "HealthSignals",
    "SliceResult",
    "ValidationStatus",
    "IntegrityCheck",
    "PipelineHealth",
    "DegradationDecision",
    "FailurePattern",
    "TopologyHealthEvaluator",
    "DegradationPolicyEngine",
]
