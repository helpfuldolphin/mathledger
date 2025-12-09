#!/usr/bin/env python3
"""
==============================================================================
PHASE II — SYNTHETIC TEST DATA ONLY
==============================================================================

Contract Schema
----------------

This module defines the schema for synthetic scenario contracts.
Contracts are machine-readable specifications that describe scenario
characteristics without making any claims about uplift.

Contract Required Fields:
    - probability_ranges: min/max probabilities
    - drift_characteristics: temporal drift settings
    - correlation_settings: intra-class correlation
    - variance_settings: per-cycle and per-item variance
    - rare_event_definitions: list of rare event channels

Must NOT:
    - Produce claims about real uplift
    - Use terms like "positive uplift" or "negative uplift"
    - Mix synthetic and real data

==============================================================================
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from experiments.synthetic_uplift.noise_models import SAFETY_LABEL


# ==============================================================================
# CONTRACT SCHEMA CONSTANTS
# ==============================================================================

CONTRACT_SCHEMA_VERSION = "contract_v2"

REQUIRED_CONTRACT_FIELDS = [
    "label",
    "schema_version",
    "registry_version",
    "scenarios",
]

REQUIRED_SCENARIO_CONTRACT_FIELDS = [
    "name",
    "version",
    "category",
    "description",
    "probability_ranges",
    "drift_characteristics",
    "correlation_settings",
    "variance_settings",
    "rare_event_definitions",
]

REQUIRED_PROBABILITY_FIELDS = ["min", "max"]

REQUIRED_DRIFT_FIELDS = ["mode", "has_temporal_drift"]

REQUIRED_CORRELATION_FIELDS = ["rho", "has_correlation"]

REQUIRED_VARIANCE_FIELDS = ["per_cycle_sigma", "per_item_sigma", "has_variance"]

REQUIRED_RARE_EVENT_FIELDS = ["type"]


# ==============================================================================
# VALIDATION RESULT
# ==============================================================================

@dataclass
class ContractValidationResult:
    """Result of contract validation."""
    valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Counts
    scenario_count: int = 0
    
    # Metadata
    schema_version: str = ""
    registry_version: str = ""
    
    def add_error(self, msg: str):
        """Add an error and mark as invalid."""
        self.errors.append(msg)
        self.valid = False
    
    def add_warning(self, msg: str):
        """Add a warning (does not invalidate)."""
        self.warnings.append(msg)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "label": SAFETY_LABEL,
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "scenario_count": self.scenario_count,
            "schema_version": self.schema_version,
            "registry_version": self.registry_version,
        }


# ==============================================================================
# VALIDATION FUNCTIONS
# ==============================================================================

def validate_contract_structure(contract: Dict[str, Any]) -> ContractValidationResult:
    """
    Validate a contract file against the schema.
    
    Args:
        contract: Parsed contract dictionary
    
    Returns:
        Validation result with any errors
    """
    result = ContractValidationResult()
    
    # Extract metadata
    result.schema_version = contract.get("schema_version", "unknown")
    result.registry_version = contract.get("registry_version", "unknown")
    
    # Check safety label
    if contract.get("label") != SAFETY_LABEL:
        result.add_error(f"Invalid safety label (must be '{SAFETY_LABEL}')")
    
    # Check required top-level fields
    for field in REQUIRED_CONTRACT_FIELDS:
        if field not in contract:
            result.add_error(f"Missing required field: {field}")
    
    # Check schema version
    if result.schema_version != CONTRACT_SCHEMA_VERSION:
        result.add_warning(
            f"Schema version mismatch: expected {CONTRACT_SCHEMA_VERSION}, "
            f"got {result.schema_version}"
        )
    
    # Validate scenarios
    scenarios = contract.get("scenarios", {})
    result.scenario_count = len(scenarios)
    
    for name, scenario_contract in scenarios.items():
        # Check name prefix
        if not name.startswith("synthetic_"):
            result.add_error(f"Contract '{name}' must start with 'synthetic_'")
        
        # Check required fields
        for f in REQUIRED_SCENARIO_CONTRACT_FIELDS:
            if f not in scenario_contract:
                result.add_error(f"Contract '{name}' missing field: {f}")
        
        # Validate probability_ranges
        prob_ranges = scenario_contract.get("probability_ranges", {})
        for f in REQUIRED_PROBABILITY_FIELDS:
            if f not in prob_ranges:
                result.add_error(f"Contract '{name}' probability_ranges missing: {f}")
        
        prob_min = prob_ranges.get("min", 0)
        prob_max = prob_ranges.get("max", 1)
        if prob_min > prob_max:
            result.add_error(f"Contract '{name}' has invalid probability range")
        if not 0.0 <= prob_min <= 1.0 or not 0.0 <= prob_max <= 1.0:
            result.add_error(f"Contract '{name}' probabilities out of [0,1] range")
        
        # Validate drift_characteristics
        drift = scenario_contract.get("drift_characteristics", {})
        for f in REQUIRED_DRIFT_FIELDS:
            if f not in drift:
                result.add_error(f"Contract '{name}' drift_characteristics missing: {f}")
        
        drift_mode = drift.get("mode", "none")
        if drift_mode not in ["none", "monotonic", "cyclical", "shock"]:
            result.add_error(f"Contract '{name}' has invalid drift mode: {drift_mode}")
        
        # Validate correlation_settings
        corr = scenario_contract.get("correlation_settings", {})
        for f in REQUIRED_CORRELATION_FIELDS:
            if f not in corr:
                result.add_error(f"Contract '{name}' correlation_settings missing: {f}")
        
        rho = corr.get("rho", 0)
        if not 0.0 <= rho <= 1.0:
            result.add_error(f"Contract '{name}' correlation rho out of range")
        
        # Validate variance_settings
        variance = scenario_contract.get("variance_settings", {})
        for f in REQUIRED_VARIANCE_FIELDS:
            if f not in variance:
                result.add_error(f"Contract '{name}' variance_settings missing: {f}")
        
        # Validate rare_event_definitions
        rare_events = scenario_contract.get("rare_event_definitions", [])
        if not isinstance(rare_events, list):
            result.add_error(f"Contract '{name}' rare_event_definitions must be list")
        else:
            for i, event in enumerate(rare_events):
                for f in REQUIRED_RARE_EVENT_FIELDS:
                    if f not in event:
                        result.add_error(
                            f"Contract '{name}' rare_event[{i}] missing: {f}"
                        )
    
    return result


def validate_contract_file(path: Path) -> ContractValidationResult:
    """
    Load and validate a contract file.
    
    Args:
        path: Path to contract JSON file
    
    Returns:
        Validation result
    """
    result = ContractValidationResult()
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            contract = json.load(f)
    except FileNotFoundError:
        result.add_error(f"Contract file not found: {path}")
        return result
    except json.JSONDecodeError as e:
        result.add_error(f"Contract JSON malformed: {e}")
        return result
    
    return validate_contract_structure(contract)


def contracts_are_identical(
    contract1: Dict[str, Any],
    contract2: Dict[str, Any],
    ignore_timestamp: bool = True,
) -> Tuple[bool, List[str]]:
    """
    Check if two contracts are semantically identical.
    
    Args:
        contract1: First contract
        contract2: Second contract
        ignore_timestamp: Whether to ignore exported_at field
    
    Returns:
        (identical, differences) tuple
    """
    differences = []
    
    # Fields to check (excluding timestamp if requested)
    exclude_fields = {"exported_at"} if ignore_timestamp else set()
    
    # Compare top-level fields
    all_keys = set(contract1.keys()) | set(contract2.keys())
    for key in all_keys - exclude_fields:
        if key == "scenarios":
            continue  # Handle separately
        
        v1 = contract1.get(key)
        v2 = contract2.get(key)
        if v1 != v2:
            differences.append(f"Field '{key}' differs: {v1} vs {v2}")
    
    # Compare scenarios
    scenarios1 = contract1.get("scenarios", {})
    scenarios2 = contract2.get("scenarios", {})
    
    all_scenario_names = set(scenarios1.keys()) | set(scenarios2.keys())
    for name in all_scenario_names:
        if name not in scenarios1:
            differences.append(f"Scenario '{name}' missing in first contract")
        elif name not in scenarios2:
            differences.append(f"Scenario '{name}' missing in second contract")
        elif scenarios1[name] != scenarios2[name]:
            differences.append(f"Scenario '{name}' differs")
    
    return len(differences) == 0, differences


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def get_contract_fingerprint(contract: Dict[str, Any]) -> str:
    """
    Compute a stable fingerprint for a contract (excluding timestamp).
    
    This can be used to verify contract stability across exports.
    """
    import hashlib
    
    # Create copy without timestamp
    fingerprint_data = {
        k: v for k, v in contract.items()
        if k != "exported_at"
    }
    
    # Serialize deterministically
    serialized = json.dumps(fingerprint_data, sort_keys=True)
    
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


def format_validation_report(result: ContractValidationResult) -> str:
    """Format validation result as human-readable report."""
    lines = [
        f"\n{SAFETY_LABEL}\n",
        "=" * 60,
        "CONTRACT VALIDATION REPORT",
        "=" * 60,
        f"Schema Version:   {result.schema_version}",
        f"Registry Version: {result.registry_version}",
        f"Scenarios:        {result.scenario_count}",
        "",
    ]
    
    if result.valid:
        lines.append("[PASS] Contract is valid")
    else:
        lines.append("[FAIL] Contract validation failed")
    
    if result.errors:
        lines.append("\nERRORS:")
        for e in result.errors:
            lines.append(f"  - {e}")
    
    if result.warnings:
        lines.append("\nWARNINGS:")
        for w in result.warnings:
            lines.append(f"  - {w}")
    
    lines.append("")
    return "\n".join(lines)


# ==============================================================================
# MAIN (for testing)
# ==============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <contract.json>")
        sys.exit(1)
    
    path = Path(sys.argv[1])
    result = validate_contract_file(path)
    print(format_validation_report(result))
    sys.exit(0 if result.valid else 1)

