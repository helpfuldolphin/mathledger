"""
Experiment Integration Helpers for TDA Pipeline Attestation
=============================================================

Provides helper functions to generate attestation blocks with TDA pipeline
hashes from RFL and U2 experiment runs.

Usage in RFL Runner:
    from attestation.experiment_integration import create_rfl_attestation_block
    
    block = create_rfl_attestation_block(
        run_id="run_001",
        experiment_id="U2_EXP_001",
        reasoning_events=proof_events,
        ui_events=ui_events,
        rfl_config=config,
        gate_decisions={"G1": "PASS", "G2": "ABANDONED_TDA"},
        prev_block_hash=prev_hash,
        block_number=0,
    )

Usage in U2 Runner:
    from attestation.experiment_integration import create_u2_attestation_block
    
    block = create_u2_attestation_block(
        run_id="run_001",
        experiment_id="U2_EXP_001",
        reasoning_events=cycle_results,
        ui_events=user_events,
        u2_config=config,
        gate_decisions=decisions,
        prev_block_hash=prev_hash,
        block_number=0,
    )
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .dual_root import (
    compute_reasoning_root,
    compute_ui_root,
    compute_composite_root,
)
from .tda_pipeline import compute_tda_pipeline_hash
from .chain_verifier import ExperimentBlock


def _hash_file(file_path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    if not file_path.exists():
        return "missing"
    
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def extract_tda_config_from_rfl(rfl_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract TDA pipeline configuration from RFL config.
    
    Args:
        rfl_config: RFL experiment configuration
        
    Returns:
        TDA configuration dictionary
    """
    # Extract bounds
    bounds = rfl_config.get("bounds", {})
    
    # Extract slice info
    slice_config = rfl_config.get("curriculum_slice", {})
    slice_id = slice_config.get("slice_name", "unknown")
    
    # Compute slice config hash if config file specified
    slice_config_path = rfl_config.get("slice_config_path")
    if slice_config_path:
        slice_config_hash = _hash_file(Path(slice_config_path))
    else:
        # Hash the slice config dict itself using RFC 8785 canonicalization
        from substrate.crypto.core import rfc8785_canonicalize
        slice_config_hash = hashlib.sha256(
            rfc8785_canonicalize(slice_config).encode("utf-8")
        ).hexdigest()
    
    # Extract verifier settings
    verifier_config = rfl_config.get("verifier", {})
    
    # Extract abstention strategy
    abstention_strategy = rfl_config.get("abstention_strategy", "conservative")
    
    return {
        "max_breadth": bounds.get("max_breadth", 0),
        "max_depth": bounds.get("max_depth", 0),
        "max_total": bounds.get("max_total", 0),
        "verifier_tier": verifier_config.get("tier", "tier1"),
        "verifier_timeout": verifier_config.get("timeout", 10.0),
        "verifier_budget": verifier_config.get("budget"),
        "slice_id": slice_id,
        "slice_config_hash": slice_config_hash,
        "abstention_strategy": abstention_strategy,
        "gates": rfl_config.get("gates"),
    }


def extract_tda_config_from_u2(u2_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract TDA pipeline configuration from U2 config.
    
    Args:
        u2_config: U2 experiment configuration
        
    Returns:
        TDA configuration dictionary
    """
    # U2 uses similar structure but may have different field names
    slice_config = u2_config.get("slice_config", {})
    slice_id = u2_config.get("slice_name", "unknown")
    
    # Compute slice config hash
    slice_config_path = u2_config.get("slice_config_path")
    if slice_config_path:
        slice_config_hash = _hash_file(Path(slice_config_path))
    else:
        # Hash the slice config dict itself using RFC 8785 canonicalization
        from substrate.crypto.core import rfc8785_canonicalize
        slice_config_hash = hashlib.sha256(
            rfc8785_canonicalize(slice_config).encode("utf-8")
        ).hexdigest()
    
    # U2 may have different bound names
    return {
        "max_breadth": u2_config.get("max_breadth", 0),
        "max_depth": u2_config.get("max_depth", 0),
        "max_total": u2_config.get("total_cycles", 0),
        "verifier_tier": u2_config.get("verifier_tier", "tier1"),
        "verifier_timeout": u2_config.get("verifier_timeout", 10.0),
        "verifier_budget": u2_config.get("verifier_budget"),
        "slice_id": slice_id,
        "slice_config_hash": slice_config_hash,
        "abstention_strategy": u2_config.get("abstention_strategy", "conservative"),
        "gates": u2_config.get("gates"),
    }


def create_rfl_attestation_block(
    run_id: str,
    experiment_id: str,
    reasoning_events: Sequence[Any],
    ui_events: Sequence[Any],
    rfl_config: Dict[str, Any],
    gate_decisions: Optional[Dict[str, str]] = None,
    prev_block_hash: Optional[str] = None,
    block_number: int = 0,
) -> ExperimentBlock:
    """
    Create an attestation block from an RFL experiment run.
    
    Args:
        run_id: Unique identifier for this run
        experiment_id: Experiment series identifier
        reasoning_events: List of proof/reasoning artifacts
        ui_events: List of UI/human interaction events
        rfl_config: RFL configuration dictionary
        gate_decisions: Hard gate decisions (e.g., {"G1": "PASS", "G2": "ABANDONED_TDA"})
        prev_block_hash: Hash of previous block in chain (None for genesis)
        block_number: Sequential block number
        
    Returns:
        ExperimentBlock ready for chain verification
    """
    # Compute dual-root attestation
    r_t = compute_reasoning_root(reasoning_events)
    u_t = compute_ui_root(ui_events)
    h_t = compute_composite_root(r_t, u_t)
    
    # Extract TDA configuration
    tda_config = extract_tda_config_from_rfl(rfl_config)
    tda_hash = compute_tda_pipeline_hash(tda_config)
    
    return ExperimentBlock(
        run_id=run_id,
        experiment_id=experiment_id,
        reasoning_root=r_t,
        ui_root=u_t,
        composite_root=h_t,
        tda_pipeline_hash=tda_hash,
        tda_config=tda_config,
        gate_decisions=gate_decisions,
        prev_block_hash=prev_block_hash,
        block_number=block_number,
    )


def create_u2_attestation_block(
    run_id: str,
    experiment_id: str,
    reasoning_events: Sequence[Any],
    ui_events: Sequence[Any],
    u2_config: Dict[str, Any],
    gate_decisions: Optional[Dict[str, str]] = None,
    prev_block_hash: Optional[str] = None,
    block_number: int = 0,
) -> ExperimentBlock:
    """
    Create an attestation block from a U2 experiment run.
    
    Args:
        run_id: Unique identifier for this run
        experiment_id: Experiment series identifier
        reasoning_events: List of cycle results/reasoning artifacts
        ui_events: List of UI/human interaction events
        u2_config: U2 configuration dictionary
        gate_decisions: Hard gate decisions
        prev_block_hash: Hash of previous block in chain (None for genesis)
        block_number: Sequential block number
        
    Returns:
        ExperimentBlock ready for chain verification
    """
    # Compute dual-root attestation
    r_t = compute_reasoning_root(reasoning_events)
    u_t = compute_ui_root(ui_events)
    h_t = compute_composite_root(r_t, u_t)
    
    # Extract TDA configuration
    tda_config = extract_tda_config_from_u2(u2_config)
    tda_hash = compute_tda_pipeline_hash(tda_config)
    
    return ExperimentBlock(
        run_id=run_id,
        experiment_id=experiment_id,
        reasoning_root=r_t,
        ui_root=u_t,
        composite_root=h_t,
        tda_pipeline_hash=tda_hash,
        tda_config=tda_config,
        gate_decisions=gate_decisions,
        prev_block_hash=prev_block_hash,
        block_number=block_number,
    )


def save_attestation_block(
    block: ExperimentBlock,
    output_path: Path,
) -> None:
    """
    Save an attestation block to a JSON file.
    
    Args:
        block: ExperimentBlock to save
        output_path: Path to output attestation.json file
    """
    import json
    
    data = {
        "run_id": block.run_id,
        "experiment_id": block.experiment_id,
        "R_t": block.reasoning_root,
        "U_t": block.ui_root,
        "H_t": block.composite_root,
        "tda_pipeline_hash": block.tda_pipeline_hash,
        "tda_config": block.tda_config,
        "gate_decisions": block.gate_decisions,
        "prev_block_hash": block.prev_block_hash,
        "block_number": block.block_number,
        "block_hash": block.compute_block_hash(),
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


__all__ = [
    "create_rfl_attestation_block",
    "create_u2_attestation_block",
    "save_attestation_block",
    "extract_tda_config_from_rfl",
    "extract_tda_config_from_u2",
]
