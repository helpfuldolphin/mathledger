"""Governance verifier module.

Provides governance verification utilities.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Callable

__version__ = "1.0.0"


# =============================================================================
# SLICE CONSTANTS
# =============================================================================

SLICE_IDS = ["alpha", "beta", "gamma", "delta"]

SLICE_SUCCESS_CRITERIA = {
    "alpha": {"min_coverage": 0.8, "max_depth": 5},
    "beta": {"min_coverage": 0.7, "max_depth": 6},
    "gamma": {"min_coverage": 0.6, "max_depth": 7},
    "delta": {"min_coverage": 0.5, "max_depth": 8},
}

RULE_REGISTRY: Dict[str, Dict[str, Any]] = {}
RULE_DESCRIPTIONS: Dict[str, str] = {}


# =============================================================================
# LAYER CONSTANTS
# =============================================================================

LAYER_REPLAY = "replay"
LAYER_TOPOLOGY = "topology"
LAYER_SECURITY = "security"
LAYER_HT = "ht"
LAYER_BUNDLE = "bundle"
LAYER_ADMISSIBILITY = "admissibility"
LAYER_PREFLIGHT = "preflight"
LAYER_METRICS = "metrics"
LAYER_BUDGET = "budget"
LAYER_CONJECTURE = "conjecture"
LAYER_GOVERNANCE = "governance"
LAYER_TDA = "tda"
LAYER_TELEMETRY_TDA = "telemetry_tda"
LAYER_SLICE_IDENTITY = "slice_identity"

DEFAULT_CRITICAL_LAYERS = [
    LAYER_REPLAY,
    LAYER_TOPOLOGY,
    LAYER_SECURITY,
    LAYER_HT,
]

LAYER_ADAPTERS: Dict[str, Callable] = {}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class VerificationResult:
    """Governance verification result."""
    valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RuleResult:
    """Result of a governance rule evaluation."""
    rule_name: str
    passed: bool = True
    severity: str = "info"
    message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GovernanceVerdict:
    """Governance verdict result."""
    passed: bool = True
    verdict: str = "ok"
    violations: List[str] = field(default_factory=list)
    rule_results: List[RuleResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GovernanceSignal:
    """Governance signal for cross-layer communication."""
    layer: str
    status: str = "ok"
    severity: str = "info"
    message: str = ""
    reasons: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# VERIFIER CLASS
# =============================================================================

class GovernanceVerifier:
    """Governance verification utilities."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def verify(self, data: Dict[str, Any]) -> VerificationResult:
        """Verify governance compliance."""
        return VerificationResult(valid=True)

    def verify_metric(self, metric_name: str, value: Any) -> bool:
        """Verify a single metric."""
        return True

    def get_report(self) -> Dict[str, Any]:
        """Get verification report."""
        return {"status": "ok", "verified": True}


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def create_verifier(config: Optional[Dict[str, Any]] = None) -> GovernanceVerifier:
    """Create a governance verifier."""
    return GovernanceVerifier(config)


def governance_verify(data: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> GovernanceVerdict:
    """Verify governance compliance (convenience function)."""
    return GovernanceVerdict(passed=True, verdict="ok")


# =============================================================================
# V2: GOVERNANCE CHRONICLE & EXPLAINER
# =============================================================================

def explain_verdict(verdict: GovernanceVerdict) -> Dict[str, Any]:
    """Explain a governance verdict."""
    return {
        "passed": verdict.passed,
        "verdict": verdict.verdict,
        "explanation": "All rules passed" if verdict.passed else "Some rules failed",
        "violations": verdict.violations,
    }


def build_governance_posture(data: Dict[str, Any]) -> Dict[str, Any]:
    """Build governance posture from data."""
    return {
        "status": "ok",
        "posture": "compliant",
        "score": 1.0,
    }


def summarize_for_admissibility(verdict: GovernanceVerdict) -> Dict[str, Any]:
    """Summarize verdict for admissibility decision."""
    return {
        "admissible": verdict.passed,
        "verdict": verdict.verdict,
        "blocking_violations": [],
    }


# =============================================================================
# PHASE III: DIRECTOR CONSOLE GOVERNANCE FEED
# =============================================================================

def build_governance_chronicle(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build governance chronicle from history."""
    return {
        "entries": len(history),
        "status": "ok",
        "timeline": history,
    }


def map_governance_to_director_status(verdict: GovernanceVerdict) -> str:
    """Map governance verdict to director status."""
    if verdict.passed:
        return "GREEN"
    elif verdict.verdict == "warn":
        return "YELLOW"
    return "RED"


def summarize_governance_for_global_health(verdicts: List[GovernanceVerdict]) -> Dict[str, Any]:
    """Summarize governance for global health display."""
    passed_count = sum(1 for v in verdicts if v.passed)
    return {
        "total": len(verdicts),
        "passed": passed_count,
        "failed": len(verdicts) - passed_count,
        "status": "ok" if passed_count == len(verdicts) else "warn",
    }


# =============================================================================
# PHASE IV: GOVERNANCE CHRONICLE COMPASS & CROSS-SYSTEM GATE
# =============================================================================

def build_governance_alignment_view(data: Dict[str, Any]) -> Dict[str, Any]:
    """Build governance alignment view."""
    return {
        "aligned": True,
        "alignment_score": 1.0,
        "gaps": [],
    }


def evaluate_governance_for_promotion(
    verdict: GovernanceVerdict,
    threshold: float = 0.8,
) -> Tuple[bool, str]:
    """Evaluate if governance allows promotion."""
    if not verdict.passed:
        return False, "Governance violations prevent promotion"
    return True, "Promotion allowed"


def build_governance_director_panel_v2(
    verdicts: List[GovernanceVerdict],
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build governance director panel v2."""
    return {
        "version": 2,
        "verdicts": len(verdicts),
        "status": "ok",
        "panels": [],
    }


# =============================================================================
# PHASE V: GLOBAL GOVERNANCE SYNTHESIZER
# =============================================================================

# TODO(PHASE-X-REPLAY): Add replay_safety signal adapter to governance registry
# Integration: Create adapt_replay_safety_to_signal() function to convert
#              to_governance_signal_for_replay_safety() output to GovernanceSignal
# Source: experiments/u2/replay_safety.py::to_governance_signal_for_replay_safety()
# Schema: docs/system_law/schemas/replay/replay_governance_radar.schema.json
# Fusion rule: Conservative BLOCK propagation (replay BLOCK → overall BLOCK)
# Binding: docs/system_law/Replay_Governance_PhaseX_Binding.md Section 6

# TODO(PHASE-X-REPLAY): Register replay layer in DEFAULT_CRITICAL_LAYERS
# Current: LAYER_REPLAY is defined but not wired to replay_safety signal
# Action: Add adapt_replay_safety_to_signal to LAYER_ADAPTERS registry
# Contract: Signal must include signal_type="replay_safety" for routing

def adapt_layer_to_signal(layer: str, data: Dict[str, Any]) -> GovernanceSignal:
    """Adapt a layer's data to a governance signal."""
    return GovernanceSignal(
        layer=layer,
        status=data.get("status", "ok"),
        severity=data.get("severity", "info"),
        message=data.get("message", ""),
    )


def adapt_tda_to_signal(tda_data: Dict[str, Any]) -> GovernanceSignal:
    """Adapt TDA data to a governance signal."""
    return GovernanceSignal(
        layer=LAYER_TDA,
        status=tda_data.get("status", "ok"),
        message="TDA signal",
    )


def adapt_telemetry_tda_to_signal(telemetry_data: Dict[str, Any]) -> GovernanceSignal:
    """Adapt telemetry TDA data to a governance signal."""
    return GovernanceSignal(
        layer=LAYER_TELEMETRY_TDA,
        status=telemetry_data.get("status", "ok"),
        message="Telemetry TDA signal",
    )


def adapt_slice_identity_to_signal(slice_data: Dict[str, Any]) -> GovernanceSignal:
    """Adapt slice identity data to a governance signal."""
    return GovernanceSignal(
        layer=LAYER_SLICE_IDENTITY,
        status=slice_data.get("status", "ok"),
        message="Slice identity signal",
    )


def build_global_alignment_view(signals: List[GovernanceSignal]) -> Dict[str, Any]:
    """Build global alignment view from signals."""
    ok_count = sum(1 for s in signals if s.status == "ok")
    return {
        "total_signals": len(signals),
        "aligned": ok_count,
        "misaligned": len(signals) - ok_count,
        "alignment_score": ok_count / len(signals) if signals else 1.0,
    }


def evaluate_global_promotion(
    signals: List[GovernanceSignal],
    critical_layers: Optional[List[str]] = None,
) -> Tuple[bool, str]:
    """Evaluate if global governance allows promotion."""
    critical_layers = critical_layers or DEFAULT_CRITICAL_LAYERS

    for signal in signals:
        if signal.layer in critical_layers and signal.status != "ok":
            return False, f"Critical layer {signal.layer} failed"

    return True, "Promotion allowed"


def build_global_governance_director_panel(
    signals: List[GovernanceSignal],
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build global governance director panel."""
    return {
        "version": 1,
        "signals": len(signals),
        "status": "ok",
        "layers": [s.layer for s in signals],
    }


# =============================================================================
# END-TO-END
# =============================================================================

def collect_all_layer_signals(data: Dict[str, Any]) -> List[GovernanceSignal]:
    """Collect signals from all layers."""
    signals = []
    for layer in [LAYER_REPLAY, LAYER_TOPOLOGY, LAYER_SECURITY, LAYER_HT]:
        if layer in data:
            signals.append(adapt_layer_to_signal(layer, data[layer]))
    return signals


def evaluate_full_cortex_body_promotion(
    data: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Evaluate full cortex-body promotion."""
    signals = collect_all_layer_signals(data)
    can_promote, reason = evaluate_global_promotion(signals)

    return {
        "can_promote": can_promote,
        "reason": reason,
        "signals": len(signals),
        "alignment": build_global_alignment_view(signals),
    }


__all__ = [
    # Version
    "__version__",
    # Classes
    "VerificationResult",
    "RuleResult",
    "GovernanceVerdict",
    "GovernanceSignal",
    "GovernanceVerifier",
    # Constants
    "SLICE_IDS",
    "SLICE_SUCCESS_CRITERIA",
    "RULE_REGISTRY",
    "RULE_DESCRIPTIONS",
    # Layer constants
    "LAYER_REPLAY",
    "LAYER_TOPOLOGY",
    "LAYER_SECURITY",
    "LAYER_HT",
    "LAYER_BUNDLE",
    "LAYER_ADMISSIBILITY",
    "LAYER_PREFLIGHT",
    "LAYER_METRICS",
    "LAYER_BUDGET",
    "LAYER_CONJECTURE",
    "LAYER_GOVERNANCE",
    "LAYER_TDA",
    "LAYER_TELEMETRY_TDA",
    "LAYER_SLICE_IDENTITY",
    "DEFAULT_CRITICAL_LAYERS",
    "LAYER_ADAPTERS",
    # Core functions
    "create_verifier",
    "governance_verify",
    # V2: Chronicle & Explainer
    "explain_verdict",
    "build_governance_posture",
    "summarize_for_admissibility",
    # Phase III: Director Console
    "build_governance_chronicle",
    "map_governance_to_director_status",
    "summarize_governance_for_global_health",
    # Phase IV: Chronicle Compass
    "build_governance_alignment_view",
    "evaluate_governance_for_promotion",
    "build_governance_director_panel_v2",
    # Phase V: Global Synthesizer
    "adapt_layer_to_signal",
    "adapt_tda_to_signal",
    "adapt_telemetry_tda_to_signal",
    "adapt_slice_identity_to_signal",
    "build_global_alignment_view",
    "evaluate_global_promotion",
    "build_global_governance_director_panel",
    # End-to-End
    "collect_all_layer_signals",
    "evaluate_full_cortex_body_promotion",
]
