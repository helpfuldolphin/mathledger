#!/usr/bin/env python3
"""
==============================================================================
PHASE II — SYNTHETIC TEST DATA ONLY
==============================================================================

Registry Enforcement Module
----------------------------

This module enforces the canonical scenario registry as the single source of
truth for synthetic universe definitions.

Enforcement guarantees:
    1. Every scenario in code is present in the registry
    2. No orphan scenario functions exist
    3. Registry structure is valid
    4. Categories and scenarios are properly mapped

Must NOT:
    - Produce claims about real uplift
    - Mix synthetic and real data
    - Modify substrate or metric logic

==============================================================================
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from experiments.synthetic_uplift.noise_models import SAFETY_LABEL


# ==============================================================================
# REGISTRY SCHEMA CONSTANTS
# ==============================================================================

REQUIRED_REGISTRY_FIELDS = [
    "label",
    "registry_version",
    "schema_version",
    "description",
    "categories",
    "scenarios",
    "ci_sweep_scenarios",
    "expected_scenario_count",
]

REQUIRED_CATEGORY_FIELDS = ["description", "purpose"]

REQUIRED_SCENARIO_FIELDS = [
    "version",
    "description",
    "category",
    "ci_sweep_included",
    "parameters",
]

REQUIRED_PARAMETER_FIELDS = [
    "seed",
    "num_cycles",
    "probabilities",
    "drift",
    "correlation",
    "rare_events",
]

VALID_CATEGORIES = ["uplift", "drift", "correlation", "rare_event", "mixed"]


# ==============================================================================
# VALIDATION RESULT
# ==============================================================================

@dataclass
class RegistryValidationResult:
    """Result of registry validation."""
    valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Counts
    scenario_count: int = 0
    category_count: int = 0
    ci_sweep_count: int = 0
    
    # Metadata
    registry_version: str = ""
    schema_version: str = ""
    
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
            "category_count": self.category_count,
            "ci_sweep_count": self.ci_sweep_count,
            "registry_version": self.registry_version,
            "schema_version": self.schema_version,
        }


# ==============================================================================
# REGISTRY LOADER
# ==============================================================================

def get_registry_path() -> Path:
    """Get the canonical registry path."""
    return Path(__file__).parent / "scenario_registry.json"


def load_registry(registry_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load the canonical scenario registry.
    
    Args:
        registry_path: Optional path override (default: scenario_registry.json)
    
    Returns:
        Parsed registry dictionary
    
    Raises:
        FileNotFoundError: If registry doesn't exist
        json.JSONDecodeError: If registry is malformed
    """
    if registry_path is None:
        registry_path = get_registry_path()
    
    with open(registry_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_registry_safe(registry_path: Optional[Path] = None) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Load registry with error handling.
    
    Returns:
        (registry_dict, error_message) - error_message is None on success
    """
    try:
        registry = load_registry(registry_path)
        return registry, None
    except FileNotFoundError:
        return None, f"Registry not found at {registry_path or get_registry_path()}"
    except json.JSONDecodeError as e:
        return None, f"Registry JSON malformed: {e}"
    except Exception as e:
        return None, f"Failed to load registry: {e}"


# ==============================================================================
# VALIDATION FUNCTIONS
# ==============================================================================

def validate_registry_structure(registry: Dict[str, Any]) -> RegistryValidationResult:
    """
    Validate the registry structure against the schema.
    
    Checks:
        - Required top-level fields present
        - Categories are valid
        - All scenarios have required fields
        - CI sweep scenarios exist in registry
        - Expected count matches actual count
    """
    result = RegistryValidationResult()
    
    # Extract metadata
    result.registry_version = registry.get("registry_version", "unknown")
    result.schema_version = registry.get("schema_version", "unknown")
    
    # Check required top-level fields
    for field in REQUIRED_REGISTRY_FIELDS:
        if field not in registry:
            result.add_error(f"Missing required field: {field}")
    
    # Check safety label
    if registry.get("label") != SAFETY_LABEL:
        result.add_error(f"Invalid safety label (must be '{SAFETY_LABEL}')")
    
    # Validate categories
    categories = registry.get("categories", {})
    result.category_count = len(categories)
    
    for cat_name in VALID_CATEGORIES:
        if cat_name not in categories:
            result.add_warning(f"Expected category not present: {cat_name}")
    
    # Check categories are sorted
    cat_keys = list(categories.keys())
    if cat_keys != sorted(cat_keys):
        result.add_error(f"Categories must be in sorted order. Got: {cat_keys}")
    
    for cat_name, cat_data in categories.items():
        for field in REQUIRED_CATEGORY_FIELDS:
            if field not in cat_data:
                result.add_error(f"Category '{cat_name}' missing field: {field}")
    
    # Validate scenarios
    scenarios = registry.get("scenarios", {})
    result.scenario_count = len(scenarios)
    
    # Check expected count
    expected_count = registry.get("expected_scenario_count")
    if expected_count is not None and len(scenarios) != expected_count:
        result.add_error(
            f"Scenario count mismatch: expected {expected_count}, got {len(scenarios)}"
        )
    
    # Check each scenario
    for name, scenario in scenarios.items():
        # Validate name prefix
        if not name.startswith("synthetic_"):
            result.add_error(f"Scenario '{name}' must start with 'synthetic_'")
        
        # Check required fields
        for field in REQUIRED_SCENARIO_FIELDS:
            if field not in scenario:
                result.add_error(f"Scenario '{name}' missing field: {field}")
        
        # Check category mapping is valid
        category = scenario.get("category")
        if category and category not in categories:
            result.add_error(f"Scenario '{name}' has invalid category: {category}")
        
        # Validate parameters
        params = scenario.get("parameters", {})
        for field in REQUIRED_PARAMETER_FIELDS:
            if field not in params:
                result.add_error(f"Scenario '{name}' parameters missing: {field}")
        
        # Validate probability distributions
        probs = params.get("probabilities", {})
        for mode, class_probs in probs.items():
            if isinstance(class_probs, dict):
                for cls, prob in class_probs.items():
                    if not isinstance(prob, (int, float)):
                        result.add_error(
                            f"Scenario '{name}' has invalid probability for {mode}/{cls}"
                        )
                    elif not 0.0 <= prob <= 1.0:
                        result.add_error(
                            f"Scenario '{name}' probability out of range: {mode}/{cls}={prob}"
                        )
    
    # Validate CI sweep scenarios
    ci_sweep = registry.get("ci_sweep_scenarios", [])
    result.ci_sweep_count = len(ci_sweep)
    
    for name in ci_sweep:
        if name not in scenarios:
            result.add_error(f"CI sweep scenario not in registry: {name}")
        else:
            # Check ci_sweep_included flag is consistent
            if not scenarios[name].get("ci_sweep_included"):
                result.add_warning(
                    f"Scenario '{name}' is in ci_sweep_scenarios but ci_sweep_included=False"
                )
    
    # Check total mapping (all categories have at least one scenario)
    category_coverage = {cat: [] for cat in categories}
    for name, scenario in scenarios.items():
        cat = scenario.get("category")
        if cat in category_coverage:
            category_coverage[cat].append(name)
    
    for cat, scenario_list in category_coverage.items():
        if not scenario_list:
            result.add_warning(f"Category '{cat}' has no scenarios")
    
    return result


def validate_code_registry_sync(
    code_scenarios: Set[str],
    registry: Dict[str, Any],
) -> RegistryValidationResult:
    """
    Validate that code scenarios match registry scenarios.
    
    Args:
        code_scenarios: Set of scenario names defined in code
        registry: Loaded registry dictionary
    
    Returns:
        Validation result with any mismatches
    """
    result = RegistryValidationResult()
    
    registry_scenarios = set(registry.get("scenarios", {}).keys())
    
    # Check for orphan scenarios (in code but not registry)
    orphans = code_scenarios - registry_scenarios
    for name in sorted(orphans):
        result.add_error(f"Orphan scenario (in code, not registry): {name}")
    
    # Check for missing implementations (in registry but not code)
    missing = registry_scenarios - code_scenarios
    for name in sorted(missing):
        result.add_error(f"Missing implementation (in registry, not code): {name}")
    
    result.scenario_count = len(registry_scenarios)
    
    return result


# ==============================================================================
# ENFORCEMENT FUNCTIONS
# ==============================================================================

def enforce_registry() -> Tuple[bool, RegistryValidationResult]:
    """
    Load and enforce the canonical registry.
    
    Returns:
        (valid, result) tuple
    
    This is the main entry point for CI validation.
    """
    # Load registry
    registry, error = load_registry_safe()
    if error:
        result = RegistryValidationResult()
        result.add_error(error)
        return False, result
    
    # Validate structure
    result = validate_registry_structure(registry)
    
    # Get code scenarios from universe_browser
    try:
        from experiments.synthetic_uplift.universe_browser import BUILT_IN_UNIVERSES
        code_scenarios = set(BUILT_IN_UNIVERSES.keys())
        
        # Validate sync
        sync_result = validate_code_registry_sync(code_scenarios, registry)
        
        # Merge results
        result.errors.extend(sync_result.errors)
        result.warnings.extend(sync_result.warnings)
        if sync_result.errors:
            result.valid = False
            
    except ImportError as e:
        result.add_warning(f"Could not validate code sync: {e}")
    
    return result.valid, result


def assert_registry_valid():
    """
    Assert that the registry is valid.
    
    Raises:
        AssertionError with details if invalid
    """
    valid, result = enforce_registry()
    
    if not valid:
        error_msg = "\n".join(f"  - {e}" for e in result.errors)
        raise AssertionError(
            f"Registry validation failed:\n{error_msg}\n\n"
            f"Registry version: {result.registry_version}\n"
            f"Scenarios: {result.scenario_count}"
        )


# ==============================================================================
# SCHEMA VALIDATION FOR SCENARIOS
# ==============================================================================

def validate_scenario_schema(scenario_name: str, scenario_data: Dict[str, Any]) -> List[str]:
    """
    Validate a single scenario against the schema.
    
    Returns list of error messages (empty if valid).
    """
    errors = []
    
    # Check name
    if not scenario_name.startswith("synthetic_"):
        errors.append(f"Name must start with 'synthetic_': {scenario_name}")
    
    # Check required fields
    for field in REQUIRED_SCENARIO_FIELDS:
        if field not in scenario_data:
            errors.append(f"Missing required field: {field}")
    
    # Validate parameters
    params = scenario_data.get("parameters", {})
    
    # Validate probabilities
    probs = params.get("probabilities", {})
    if not probs:
        errors.append("Probabilities cannot be empty")
    
    for mode, class_probs in probs.items():
        if not isinstance(class_probs, dict):
            errors.append(f"Probabilities for '{mode}' must be a dict")
            continue
        
        for cls, prob in class_probs.items():
            if not isinstance(prob, (int, float)):
                errors.append(f"Probability {mode}/{cls} must be numeric")
            elif not 0.0 <= prob <= 1.0:
                errors.append(f"Probability {mode}/{cls}={prob} out of range [0,1]")
    
    # Validate drift
    drift = params.get("drift", {})
    drift_mode = drift.get("mode", "none")
    valid_drift_modes = ["none", "monotonic", "cyclical", "shock"]
    if drift_mode not in valid_drift_modes:
        errors.append(f"Invalid drift mode: {drift_mode}")
    
    if drift_mode == "cyclical":
        period = drift.get("period", 0)
        if period <= 0:
            errors.append("Cyclical drift requires period > 0")
    
    # Validate correlation
    corr = params.get("correlation", {})
    rho = corr.get("rho", 0.0)
    if not 0.0 <= rho <= 1.0:
        errors.append(f"Correlation rho={rho} out of range [0,1]")
    
    # Validate rare events
    rare_events = params.get("rare_events", [])
    if not isinstance(rare_events, list):
        errors.append("rare_events must be a list")
    
    return errors


# ==============================================================================
# CLI SUPPORT
# ==============================================================================

def format_validation_report(result: RegistryValidationResult) -> str:
    """Format validation result as human-readable report."""
    lines = [
        f"\n{SAFETY_LABEL}\n",
        "=" * 60,
        "REGISTRY VALIDATION REPORT",
        "=" * 60,
        f"Registry Version: {result.registry_version}",
        f"Schema Version:   {result.schema_version}",
        f"Scenarios:        {result.scenario_count}",
        f"Categories:       {result.category_count}",
        f"CI Sweep:         {result.ci_sweep_count}",
        "",
    ]
    
    if result.valid:
        lines.append("[PASS] Registry is valid")
    else:
        lines.append("[FAIL] Registry validation failed")
    
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
# MAIN
# ==============================================================================

if __name__ == "__main__":
    valid, result = enforce_registry()
    print(format_validation_report(result))
    exit(0 if valid else 1)

