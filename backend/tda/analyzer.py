"""
TDA Analyzer

Evaluates topological properties and makes hard gate decisions.
"""

from typing import Any, Dict, Optional
from .modes import TDAMode, TDADecision


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
    # Block if abstention rate exceeds 50% (system is failing)
    should_block = False
    reason = "TDA: System healthy"
    confidence = 1.0
    
    if abstention_rate > 0.5:
        should_block = True
        reason = f"TDA: Critical abstention rate {abstention_rate:.2%} > 50%"
        confidence = 0.95
    elif coverage_rate < 0.5 and cycle_index > 10:
        should_block = True
        reason = f"TDA: Poor coverage {coverage_rate:.2%} after {cycle_index} cycles"
        confidence = 0.85
    elif verified_count == 0 and cycle_index > 5:
        should_block = True
        reason = f"TDA: No verified proofs after {cycle_index} cycles"
        confidence = 0.90
    
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
