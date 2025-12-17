"""
Adapter to transform Risk Tile outputs into GGFL contributions.
"""

from typing import Dict, Any

# Mapping from the Risk Tile's risk_band to GGFL severity semantics.
# This is a critical junction for making risk actionable within the governance framework.
RISK_BAND_TO_GGFL_SEVERITY = {
    "LOW": "INFO",
    "MEDIUM": "WARNING",
    "HIGH": "ERROR",
    "CRITICAL": "CRITICAL",
}

def create_ggfl_contribution(
    risk_band: str,
    divergence_score: float,
    budget_remaining: float
) -> Dict[str, Any]:
    """
    Creates a GGFL contribution by fusing risk, divergence, and budget data.

    This function represents the core of the integration, turning a risk assessment
    into a structured event for the a wider governance framework.

    Args:
        risk_band: The overall risk band from the risk tile system.
        divergence_score: The mission-specific divergence metric.
        budget_remaining: The remaining computational or operational budget (as a percentage).

    Returns:
        A dictionary representing a GGFL contribution.
    """
    # Start with the baseline severity from the risk band
    severity = RISK_BAND_TO_GGFL_SEVERITY.get(risk_band, "INFO")
    
    # Fusion Logic: Escalate severity based on interactions
    # If risk is already high, a high divergence is a critical signal.
    if risk_band in ["HIGH", "CRITICAL"] and divergence_score > 0.015:
        severity = "CRITICAL"
        reason = "High risk combined with significant mission divergence."
    # If the budget is nearly exhausted, even a medium risk is more concerning.
    elif risk_band == "MEDIUM" and budget_remaining < 10.0:
        severity = "ERROR"
        reason = "Medium risk poses a greater threat with low remaining budget."
    else:
        reason = f"Baseline assessment based on overall risk band: {risk_band}."

    return {
        "source": "RiskTileAdapter",
        "severity": severity,
        "event_type": "RiskAssessment",
        "details": {
            "initial_risk_band": risk_band,
            "divergence": divergence_score,
            "budget_remaining_pct": budget_remaining,
            "justification": reason
        },
        "timestamp": "2025-12-12T12:00:00Z" # In a real system, this would be dynamic
    }
