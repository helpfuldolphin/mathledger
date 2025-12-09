"""
TDA Analyzer

Evaluates topological properties and makes hard gate decisions.
"""

from typing import Any, Dict, Optional
from .modes import TDAMode, TDADecision

# TDA Decision Thresholds (configurable)
ABSTENTION_THRESHOLD = 0.5  # Block if abstention rate exceeds 50%
COVERAGE_THRESHOLD = 0.5  # Block if coverage below 50% after min cycles
MIN_CYCLES_FOR_COVERAGE = 10  # Minimum cycles before coverage matters
MIN_CYCLES_FOR_PROOFS = 5  # Minimum cycles before requiring proofs

# Confidence scores for different blocking conditions
CONFIDENCE_ABSTENTION = 0.95
CONFIDENCE_COVERAGE = 0.85
CONFIDENCE_NO_PROOFS = 0.90


def evaluate_tda_decision(
    context: Dict[str, Any],
    mode: Optional[TDAMode] = None,
) -> TDADecision:
    """
    Evaluate TDA decision for a given context.
    
    Args:
        context: Execution context containing metrics and state
        mode: TDA mode override (defaults to BLOCK)
        
    Returns:
        TDADecision with should_block flag and reasoning
    """
    # Default to BLOCK mode if not specified
    if mode is None:
        mode = TDAMode.BLOCK
    
    # Extract relevant metrics from context
    abstention_rate = context.get("abstention_rate", 0.0)
    coverage_rate = context.get("coverage_rate", 0.0)
    verified_count = context.get("verified_count", 0)
    cycle_index = context.get("cycle_index", 0)
    
    # Topological analysis: check for structural degradation
    should_block = False
    reason = "TDA: System healthy"
    confidence = 1.0
    
    if abstention_rate > ABSTENTION_THRESHOLD:
        should_block = True
        reason = f"TDA: Critical abstention rate {abstention_rate:.2%} > {ABSTENTION_THRESHOLD:.0%}"
        confidence = CONFIDENCE_ABSTENTION
    elif coverage_rate < COVERAGE_THRESHOLD and cycle_index > MIN_CYCLES_FOR_COVERAGE:
        should_block = True
        reason = f"TDA: Poor coverage {coverage_rate:.2%} after {cycle_index} cycles"
        confidence = CONFIDENCE_COVERAGE
    elif verified_count == 0 and cycle_index > MIN_CYCLES_FOR_PROOFS:
        should_block = True
        reason = f"TDA: No verified proofs after {cycle_index} cycles"
        confidence = CONFIDENCE_NO_PROOFS
    
    # Collect metadata for audit trail
    metadata = {
        "abstention_rate": abstention_rate,
        "coverage_rate": coverage_rate,
        "verified_count": verified_count,
        "cycle_index": cycle_index,
        "analysis_version": "v1.0.0",
    }
    
    return TDADecision(
        should_block=should_block,
        mode=mode,
        reason=reason,
        confidence=confidence,
        metadata=metadata,
    )
