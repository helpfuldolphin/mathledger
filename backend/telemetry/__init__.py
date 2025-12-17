"""
Telemetry Module  Phase X Canonical I/O Substrate

This module provides telemetry infrastructure for Phase X operations including:
- Governance signal emission
- Conformance checking
- TDA feedback integration

SHADOW MODE CONTRACT:
- All telemetry is OBSERVATIONAL ONLY
- No modification of real runner execution
- All signals have enforcement_status="LOGGED_ONLY"

See: docs/system_law/Telemetry_PhaseX_Contract.md
"""

from backend.telemetry.governance_signal import (
    TelemetryGovernanceSignal,
    TelemetryGovernanceSignalEmitter,
    EmitterHealth,
    AnomalySummary,
    GovernanceRecommendation,
)
from backend.telemetry.p4_integration import (
    build_telemetry_summary_for_p4,
    attach_telemetry_governance_to_evidence,
    telemetry_signal_to_ggfl_telemetry,
    TelemetryP4Summary,
    TelemetryHealthSummary,
    TDAFeedbackSummary,
)

__all__ = [
    # Governance signal
    "TelemetryGovernanceSignal",
    "TelemetryGovernanceSignalEmitter",
    "EmitterHealth",
    "AnomalySummary",
    "GovernanceRecommendation",
    # P4 integration
    "build_telemetry_summary_for_p4",
    "attach_telemetry_governance_to_evidence",
    "telemetry_signal_to_ggfl_telemetry",
    "TelemetryP4Summary",
    "TelemetryHealthSummary",
    "TDAFeedbackSummary",
]
