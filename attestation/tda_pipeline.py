"""
TDA Pipeline Hash Computation and Verification
================================================

This module provides cryptographic binding for Testing/Derivation/Analysis (TDA)
pipeline configuration, ensuring that experiment runs are verifiable and that
configuration drift is detectable.

TDA Pipeline Hash Components:
- Derivation configuration (bounds, gates, axioms)
- Verification settings (verifier tier, timeout, budget)
- Curriculum slice specification
- Abstention handling rules

This enables:
1. Detection of configuration drift across experiment runs
2. Cryptographic binding of Hard Gate decisions (e.g., ABANDONED_TDA)
3. Attestation chain verification including pipeline integrity
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

from substrate.crypto.core import rfc8785_canonicalize


@dataclass(frozen=True)
class TDAPipelineConfig:
    """
    Immutable TDA pipeline configuration for hashing.
    
    Captures all elements that affect derivation and verification behavior.
    """
    
    # Derivation bounds
    max_breadth: int
    max_depth: int
    max_total: int
    
    # Verification settings
    verifier_tier: str
    verifier_timeout: float
    verifier_budget: Optional[Dict[str, int]]
    
    # Curriculum specification
    slice_id: str
    slice_config_hash: str  # Hash of the slice config file
    
    # Abstention rules
    abstention_strategy: str
    
    # Gate specifications (if applicable)
    gates: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "max_breadth": self.max_breadth,
            "max_depth": self.max_depth,
            "max_total": self.max_total,
            "verifier_tier": self.verifier_tier,
            "verifier_timeout": self.verifier_timeout,
            "verifier_budget": self.verifier_budget,
            "slice_id": self.slice_id,
            "slice_config_hash": self.slice_config_hash,
            "abstention_strategy": self.abstention_strategy,
            "gates": self.gates,
        }
    
    def compute_hash(self) -> str:
        """
        Compute canonical SHA-256 hash of TDA pipeline configuration.
        
        Returns:
            64-character hex digest
        """
        canonical_json = rfc8785_canonicalize(self.to_dict())
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def compute_tda_pipeline_hash(config_dict: Dict[str, Any]) -> str:
    """
    Compute TDA pipeline hash from a configuration dictionary.
    
    Args:
        config_dict: Configuration dictionary with required fields:
            - max_breadth (int): Maximum derivation breadth (e.g., 100)
            - max_depth (int): Maximum derivation depth (e.g., 50)
            - max_total (int): Maximum total statements (e.g., 1000)
            - verifier_tier (str): Verifier tier identifier (e.g., "tier1")
            - verifier_timeout (float): Verification timeout in seconds (e.g., 10.0)
            - slice_id (str): Curriculum slice identifier (e.g., "slice_a")
            - slice_config_hash (str): Hash of slice configuration file
            - abstention_strategy (str): Abstention handling strategy (e.g., "conservative")
            
            Optional fields:
            - verifier_budget (dict or None): Budget constraints
            - gates (dict or None): Gate specifications
        
    Returns:
        64-character hex digest
        
    Raises:
        ValueError: If required fields are missing
    """
    required_fields = [
        "max_breadth", "max_depth", "max_total",
        "verifier_tier", "verifier_timeout",
        "slice_id", "slice_config_hash", "abstention_strategy"
    ]
    
    for field in required_fields:
        if field not in config_dict:
            raise ValueError(f"Missing required field: {field}")
    
    config = TDAPipelineConfig(
        max_breadth=config_dict["max_breadth"],
        max_depth=config_dict["max_depth"],
        max_total=config_dict["max_total"],
        verifier_tier=config_dict["verifier_tier"],
        verifier_timeout=config_dict["verifier_timeout"],
        verifier_budget=config_dict.get("verifier_budget"),
        slice_id=config_dict["slice_id"],
        slice_config_hash=config_dict["slice_config_hash"],
        abstention_strategy=config_dict["abstention_strategy"],
        gates=config_dict.get("gates"),
    )
    
    return config.compute_hash()


@dataclass
class TDADivergence:
    """
    Record of TDA configuration drift between experiment runs.
    """
    
    run_id_1: str
    run_id_2: str
    tda_hash_1: str
    tda_hash_2: str
    divergent_fields: Dict[str, tuple]  # field_name -> (value1, value2)
    
    def __str__(self) -> str:
        """Format divergence for logging."""
        lines = [
            f"TDA Configuration Divergence Detected:",
            f"  Run 1: {self.run_id_1} (hash: {self.tda_hash_1[:16]}...)",
            f"  Run 2: {self.run_id_2} (hash: {self.tda_hash_2[:16]}...)",
            f"  Divergent fields:",
        ]
        for field, (val1, val2) in self.divergent_fields.items():
            lines.append(f"    {field}: {val1} → {val2}")
        return "\n".join(lines)


def detect_tda_divergence(
    run_id_1: str,
    config_1: Dict[str, Any],
    run_id_2: str,
    config_2: Dict[str, Any],
) -> Optional[TDADivergence]:
    """
    Detect configuration drift between two TDA pipeline configs.
    
    Args:
        run_id_1: First run identifier
        config_1: First TDA configuration
        run_id_2: Second run identifier
        config_2: Second TDA configuration
        
    Returns:
        TDADivergence if drift detected, None otherwise
    """
    hash_1 = compute_tda_pipeline_hash(config_1)
    hash_2 = compute_tda_pipeline_hash(config_2)
    
    if hash_1 == hash_2:
        return None
    
    # Find divergent fields
    divergent = {}
    all_keys = set(config_1.keys()) | set(config_2.keys())
    
    for key in all_keys:
        val1 = config_1.get(key)
        val2 = config_2.get(key)
        
        # Normalize None values for comparison
        if val1 != val2:
            divergent[key] = (val1, val2)
    
    return TDADivergence(
        run_id_1=run_id_1,
        run_id_2=run_id_2,
        tda_hash_1=hash_1,
        tda_hash_2=hash_2,
        divergent_fields=divergent,
    )


__all__ = [
    "TDAPipelineConfig",
    "TDADivergence",
    "compute_tda_pipeline_hash",
    "detect_tda_divergence",
]
