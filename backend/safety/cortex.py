"""
Cortex: TDA Hard Gate Decision Engine

The Cortex is the central decision point that integrates TDA (Topological
Decision Analysis) with Safety SLO outcomes. This is a BLOCKING call that
determines whether U2Runner and RFLRunner can proceed.

STRATCOM: FIRST LIGHT PRIORITY. This is the Brain.
"""

from typing import Any, Dict, Optional
from datetime import datetime

from backend.tda import TDAMode, evaluate_tda_decision
from .envelope import SafetyEnvelope, SafetySLO, SLOStatus


def evaluate_hard_gate_decision(
    context: Dict[str, Any],
    tda_mode: Optional[TDAMode] = None,
    timestamp: Optional[datetime] = None,
) -> SafetyEnvelope:
    """
    Evaluate hard gate decision (the "Cortex").
    
    Integrates TDA hard gate with Safety SLO to produce a unified decision
    that controls whether execution can proceed.
    
    BLOCKING BEHAVIOR:
    - When TDA should_block=True AND mode=BLOCK: SafetyEnvelope → BLOCK
    - When TDA mode=DRY_RUN: SafetyEnvelope → ADVISORY (doesn't alter status)
    - When TDA mode=SHADOW: SafetyEnvelope records hypothetical status
    
    Args:
        context: Execution context with metrics and state
        tda_mode: TDA operational mode (BLOCK, DRY_RUN, SHADOW)
        timestamp: Timestamp for audit trail (deterministic if provided)
        
    Returns:
        SafetyEnvelope with final decision and audit trail
        
    Raises:
        ValueError: If context is missing required fields
    """
    # Default to BLOCK mode if not specified
    if tda_mode is None:
        tda_mode = TDAMode.BLOCK
    
    # Use deterministic timestamp if not provided
    if timestamp is None:
        from substrate.repro.determinism import deterministic_timestamp
        timestamp = deterministic_timestamp(0)
    
    timestamp_str = timestamp.isoformat() + "Z"
    
    # Validate context has minimum required fields
    if not isinstance(context, dict):
        raise ValueError("Context must be a dictionary")
    
    # Evaluate TDA decision
    tda_decision = evaluate_tda_decision(context, mode=tda_mode)
    
    # Determine Safety SLO based on TDA decision and mode
    slo_status: SLOStatus
    slo_message: str
    final_decision: str
    
    if tda_mode == TDAMode.BLOCK:
        # BLOCK mode: TDA decision directly affects safety outcome
        if tda_decision.should_block:
            slo_status = SLOStatus.BLOCK
            slo_message = f"BLOCKED: {tda_decision.reason}"
            final_decision = "block"
        else:
            slo_status = SLOStatus.PASS
            slo_message = f"PASS: {tda_decision.reason}"
            final_decision = "proceed"
            
    elif tda_mode == TDAMode.DRY_RUN:
        # DRY_RUN mode: Advisory only, doesn't alter execution status
        slo_status = SLOStatus.ADVISORY
        if tda_decision.should_block:
            slo_message = f"DRY_RUN: Would block - {tda_decision.reason}"
        else:
            slo_message = f"DRY_RUN: Would pass - {tda_decision.reason}"
        final_decision = "proceed"  # Always proceed in DRY_RUN
        
    elif tda_mode == TDAMode.SHADOW:
        # SHADOW mode: Record hypothetical status without affecting execution
        slo_status = SLOStatus.ADVISORY
        if tda_decision.should_block:
            slo_message = f"SHADOW: Hypothetical block - {tda_decision.reason}"
        else:
            slo_message = f"SHADOW: Hypothetical pass - {tda_decision.reason}"
        final_decision = "proceed"  # Always proceed in SHADOW
    else:
        raise ValueError(f"Unknown TDA mode: {tda_mode}")
    
    # Create Safety SLO
    slo = SafetySLO(
        status=slo_status,
        message=slo_message,
        metadata={
            "tda_confidence": tda_decision.confidence,
            "tda_metadata": tda_decision.metadata,
        },
        timestamp=timestamp_str,
    )
    
    # Build audit trail
    audit_trail = {
        "timestamp": timestamp_str,
        "tda_decision": tda_decision.to_dict(),
        "context_keys": list(context.keys()),
        "decision_path": f"{tda_mode.value} -> {slo_status.value} -> {final_decision}",
    }
    
    # Create and return SafetyEnvelope
    envelope = SafetyEnvelope(
        slo=slo,
        tda_should_block=tda_decision.should_block,
        tda_mode=tda_mode.value,
        decision=final_decision,
        audit_trail=audit_trail,
    )
    
    return envelope


def check_gate_decision(envelope: SafetyEnvelope) -> None:
    """
    Check gate decision and raise exception if blocked.
    
    This is the enforcement point - call this to actually block execution
    when the Cortex says to block.
    
    Args:
        envelope: SafetyEnvelope from evaluate_hard_gate_decision
        
    Raises:
        RuntimeError: If gate decision is to block execution
    """
    if envelope.is_blocking():
        raise RuntimeError(
            f"Hard gate BLOCKED: {envelope.slo.message}\n"
            f"TDA Mode: {envelope.tda_mode}\n"
            f"Decision: {envelope.decision}\n"
            f"Audit: {envelope.audit_trail}"
        )
