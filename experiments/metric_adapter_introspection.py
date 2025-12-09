# PHASE II — NOT USED IN PHASE I
# File: experiments/metric_adapter_introspection.py
#
# Metric Adapter Introspection Layer
# ==================================
#
# This module provides introspection, validation, and contract export utilities
# for the Slice Metric Adapter Layer. It enables:
#   - Human-readable specification summaries
#   - Cross-source alignment verification (harmonization table, prereg, slice config, adapter)
#   - JSON schema contract export for adapter–slice pairs
#   - Deep-diff detection for schema drift
#   - Per-slice readiness gating for CI/CD
#   - Log-field coverage documentation
#
# ABSOLUTE SAFEGUARDS:
#   - Do NOT modify success metric definitions.
#   - Do NOT compute or imply uplift.
#   - Do NOT alter slice configs or governance language.
#   - No execution impact on U2 Runner.
#
# All functions are pure and deterministic.
#
# =============================================================================
# CI INTEGRATION BLUEPRINT
# =============================================================================
#
# This module provides a diagnostic/contractive layer — NOT promotional.
# It does not compute uplift, p-values, or significance. Numbers are opaque.
#
# RECOMMENDED CI JOBS:
# --------------------
#
# 1. HEALTH CHECK (all slices)
#    Command:
#      uv run python experiments/metric_adapter_introspection.py --health-check --json
#    Exit Semantics:
#      - Exit 0: All slices OK or WARN (safe to proceed)
#      - Exit 1: At least one slice FAIL (blocking issue detected)
#    Output:
#      JSON with overall_status, per-slice status, errors, warnings
#    Use Case:
#      Run on every PR to catch config drift early
#
# 2. READINESS GATE (per-slice pre-flight)
#    Command:
#      uv run python experiments/metric_adapter_introspection.py --ready <slice> --json
#    Exit Semantics:
#      - Exit 0: READY or DEGRADED (slice is callable)
#      - Exit 1: BLOCKED (slice cannot be executed safely)
#      - Exit 2: Internal error (introspection failure)
#    Output:
#      JSON with status, drift_severity, missing_contract, missing_fields,
#      unknown_fields, alignment_issues, log_field_coverage
#    Use Case:
#      Gate slice execution in pipelines; block promotion if BLOCKED
#
# 3. CONTRACT INDEX (read-only contract browser)
#    Command:
#      uv run python experiments/metric_adapter_introspection.py --list-contracts --json
#    Exit Semantics:
#      - Exit 0: Index found and readable
#      - Exit 1: Index missing or corrupt (run --export-all-contracts first)
#    Output:
#      JSON array of {slice_name, metric_kind, contract_path, schema_version, config_hash}
#    Use Case:
#      Allow other agents (D2, D5, governance verifier) to do contract lookups cheaply
#
# CONSUMER GUIDANCE:
# ------------------
# - D2 (Drift Alignment Agent): Use --list-contracts to discover contracts,
#   then read individual contract files for schema comparison.
# - D5 (Governance Verifier): Use --health-check to ensure all slices are
#   lawfully configured before governance sign-off.
# - CI Pipeline: Use --ready <slice> as a gate before any slice execution.
# - B1 (Runner Agent): Use is_slice_ready_for_experiments() predicate.
#
# 4. READINESS SUMMARY (multi-slice aggregation)
#    Command:
#      uv run python experiments/metric_adapter_introspection.py --readiness-summary --json
#    Output:
#      JSON with schema_version, counts, and per-slice status
#    Use Case:
#      Dashboard aggregation, governance sign-off, multi-slice gates
#
# 5. SUMMARY LINE (CI-grade one-liner)
#    Command:
#      uv run python experiments/metric_adapter_introspection.py --summary-line
#    Exit Semantics:
#      - Exit 0: OK or WARN (no BLOCKED slices)
#      - Exit 1: BLOCK (at least one BLOCKED slice)
#    Output:
#      Single line: Metric Readiness: ready=X degraded=Y blocked=Z total=N STATUS=OK|WARN|BLOCK
#    Use Case:
#      CI log parsing, quick status check
#
# READINESS CONSUMER PREDICATE:
# -----------------------------
# is_slice_ready_for_experiments(result: ReadinessResult) -> bool
#   - Returns True iff status == READY and drift_severity in {"NONE", "COSMETIC"}
#   - This predicate is a STABLE CONTRACT; changes require version bump
#   - Intended consumers: B1 runner, D5 drift gate, governance
#
# STABILITY GUARANTEES:
# ---------------------
# - All JSON outputs are sorted and deterministic: same config → same output
# - Contract schema version is tracked in INTROSPECTION_SCHEMA_VERSION
# - ReadinessResult JSON schema is stable; changes are versioned
# - LogFieldCoverageMap JSON schema is stable; changes are versioned
# - ReadinessSummary schema is stable at version READINESS_SUMMARY_SCHEMA_VERSION
# - is_slice_ready_for_experiments() is a stable predicate contract
#

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import yaml

from experiments.u2_pipeline import (
    METRIC_HARMONIZATION_TABLE,
    MetricKind,
    MetricSchemaValidator,
    SliceMetricAdapter,
    get_harmonization_table,
)


# =============================================================================
# Constants
# =============================================================================

DEFAULT_CURRICULUM_PATH = Path("config/curriculum_uplift_phase2.yaml")
DEFAULT_PREREG_PATH = Path("experiments/prereg/PREREG_UPLIFT_U2.yaml")

# Version for contract schema stability tracking
INTROSPECTION_SCHEMA_VERSION = "1.0.0"

# Default output directory for contract bundles
DEFAULT_CONTRACT_OUTPUT_DIR = Path("artifacts/phase_ii/metric_contracts")

# Readiness summary schema version - separate from contract schema
READINESS_SUMMARY_SCHEMA_VERSION = "1.0.0"

# Drift severities considered acceptable for experiment readiness
# This is a contract: changes require documentation update
ACCEPTABLE_DRIFT_SEVERITIES_FOR_EXPERIMENTS = frozenset({"NONE", "COSMETIC"})

# Health check status levels
class HealthStatus(str, Enum):
    """Health check status for CI integration."""
    OK = "OK"
    WARN = "WARN"
    FAIL = "FAIL"


# Readiness gate status levels
class ReadinessStatus(str, Enum):
    """
    Readiness gate status for per-slice pre-flight checks.
    
    READY: Alignment passes, drift is NONE or COSMETIC only
    DEGRADED: Param drift but still callable (PARAMETRIC_MINOR)
    BLOCKED: SEMANTIC drift or missing contract artifacts
    """
    READY = "READY"
    DEGRADED = "DEGRADED"
    BLOCKED = "BLOCKED"


# Drift severity classification for readiness gating
class DriftSeverityClass(str, Enum):
    """
    Classification of drift severity for readiness determination.
    
    Maps raw drift items to categories:
      NONE: No drift at all
      COSMETIC: Non-breaking formatting or naming differences
      PARAMETRIC_MINOR: Parameter value drift but callable
      SEMANTIC: Structural/kind-level drift that breaks execution
    """
    NONE = "NONE"
    COSMETIC = "COSMETIC"
    PARAMETRIC_MINOR = "PARAMETRIC_MINOR"
    SEMANTIC = "SEMANTIC"


# =============================================================================
# Data Types
# =============================================================================

class AlignmentStatus(str, Enum):
    """Status of alignment check between sources."""
    ALIGNED = "aligned"
    MISALIGNED = "misaligned"
    PARTIAL = "partial"
    MISSING = "missing"


@dataclass(frozen=True)
class DriftItem:
    """Represents a single schema drift detection."""
    field: str
    source_a: str
    source_b: str
    value_a: Any
    value_b: Any
    severity: str  # "error", "warning", "info"
    description: str


@dataclass
class AlignmentReport:
    """Complete alignment verification report."""
    slice_name: str
    status: AlignmentStatus
    harmonization_aligned: bool
    prereg_aligned: bool
    adapter_aligned: bool
    drifts: List[DriftItem] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def is_fully_aligned(self) -> bool:
        """Returns True if all sources are aligned with no errors."""
        return (
            self.status == AlignmentStatus.ALIGNED
            and self.harmonization_aligned
            and self.prereg_aligned
            and self.adapter_aligned
            and len(self.errors) == 0
        )


@dataclass(frozen=True)
class MetricContractSchema:
    """JSON schema contract for an adapter–slice pair."""
    schema_version: str
    slice_name: str
    metric_kind: str
    required_config_fields: Tuple[str, ...]
    required_log_fields: Tuple[str, ...]
    runtime_fields: Tuple[str, ...]
    parameter_schema: Dict[str, str]
    output_schema: Dict[str, Any]
    config_hash: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "schema_version": self.schema_version,
            "slice_name": self.slice_name,
            "metric_kind": self.metric_kind,
            "required_config_fields": list(self.required_config_fields),
            "required_log_fields": list(self.required_log_fields),
            "runtime_fields": list(self.runtime_fields),
            "parameter_schema": self.parameter_schema,
            "output_schema": self.output_schema,
            "config_hash": self.config_hash,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


@dataclass
class ReadinessResult:
    """
    Result of a per-slice readiness check.
    
    Used by --ready <slice> CLI mode.
    
    This is a stable, documented contract for the readiness gate.
    The JSON output includes:
      - slice_name: The slice being checked
      - status: READY | DEGRADED | BLOCKED
      - drift_severity: NONE | COSMETIC | PARAMETRIC_MINOR | SEMANTIC
      - missing_contract: bool - True if contract artifact is missing
      - missing_fields: List of required fields that are missing
      - unknown_fields: List of unrecognized fields found
      - alignment_issues: List of alignment problem descriptions
      - log_field_coverage: Coverage map for this slice's metric
    """
    slice_name: str
    status: ReadinessStatus
    alignment_passes: bool
    drift_class: DriftSeverityClass
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    drift_items: List[DriftItem] = field(default_factory=list)
    # Extended fields for formalized contract
    missing_contract: bool = False
    missing_fields: List[str] = field(default_factory=list)
    unknown_fields: List[str] = field(default_factory=list)
    alignment_issues: List[str] = field(default_factory=list)
    log_field_coverage: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for JSON serialization.
        
        This is a stable contract format. Changes to this structure
        should be versioned and documented.
        """
        return {
            "slice_name": self.slice_name,
            "status": self.status.value,
            "drift_severity": self.drift_class.value,  # Renamed for clarity
            "missing_contract": self.missing_contract,
            "missing_fields": sorted(self.missing_fields),  # Sorted for determinism
            "unknown_fields": sorted(self.unknown_fields),  # Sorted for determinism
            "alignment_issues": self.alignment_issues,
            "log_field_coverage": self.log_field_coverage,
            # Legacy fields preserved for backwards compatibility
            "alignment_passes": self.alignment_passes,
            "errors": self.errors,
            "warnings": self.warnings,
            "drift_items": [
                {
                    "field": d.field,
                    "source_a": d.source_a,
                    "source_b": d.source_b,
                    "value_a": str(d.value_a),
                    "value_b": str(d.value_b),
                    "severity": d.severity,
                    "description": d.description,
                }
                for d in self.drift_items
            ],
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


@dataclass
class LogFieldCoverageMap:
    """
    Log field coverage information for a metric kind.
    
    Documents which JSONL fields are actually read by the adapter.
    This is a mini contract for log-field expectations.
    
    Stable contract format:
      - metric_kind: The metric type (goal_hit, sparse_success, etc.)
      - required_log_fields: Fields that MUST be present in JSONL logs
      - optional_log_fields: Fields that MAY be present (for extended analysis)
      - runtime_fields: Fields provided at execution time (not from logs)
      - parameter_fields: Config parameters (required + optional)
      - interpretation: How the metric result should be interpreted
    """
    metric_kind: str
    required_log_fields: List[str]
    runtime_fields: List[str]
    parameter_fields: List[str]
    interpretation: str
    optional_log_fields: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for JSON serialization.
        
        This is a stable contract format for log-field coverage.
        """
        return {
            "metric_kind": self.metric_kind,
            "required_log_fields": sorted(self.required_log_fields),  # Sorted for determinism
            "optional_log_fields": sorted(self.optional_log_fields),  # Sorted for determinism
            "runtime_fields": sorted(self.runtime_fields),  # Sorted for determinism
            "parameter_fields": sorted(self.parameter_fields),  # Sorted for determinism
            "interpretation": self.interpretation,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


# =============================================================================
# Curriculum & PREREG Loaders
# =============================================================================

def load_curriculum(path: Path = DEFAULT_CURRICULUM_PATH) -> Dict[str, Any]:
    """
    Loads the curriculum YAML file.
    
    Returns the full curriculum dict or raises FileNotFoundError.
    """
    if not path.exists():
        raise FileNotFoundError(f"Curriculum file not found: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_prereg(path: Path = DEFAULT_PREREG_PATH) -> Dict[str, Any]:
    """
    Loads the PREREG_UPLIFT_U2.yaml file.
    
    Returns the full prereg dict or raises FileNotFoundError.
    """
    if not path.exists():
        raise FileNotFoundError(f"PREREG file not found: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Handle multi-document YAML (separated by ---)
    docs = list(yaml.safe_load_all(content))
    
    # Merge all documents into a single dict
    result: Dict[str, Any] = {}
    for doc in docs:
        if doc is not None:
            result.update(doc)
    
    return result


def get_slice_config(
    curriculum: Dict[str, Any],
    slice_name: str
) -> Optional[Dict[str, Any]]:
    """
    Extracts a slice configuration from the curriculum.
    
    Returns None if slice not found.
    """
    slices = curriculum.get("slices", {})
    return slices.get(slice_name)


def get_prereg_slice_spec(
    prereg: Dict[str, Any],
    slice_name: str
) -> Optional[Dict[str, Any]]:
    """
    Extracts the preregistration spec for a slice.
    
    Handles both direct slice keys and nested preregistration structure.
    """
    # Try direct slice key first
    if slice_name in prereg:
        return prereg[slice_name]
    
    # Try looking in preregistration block
    preregistration = prereg.get("preregistration", {})
    if preregistration.get("slice_name") == slice_name:
        return preregistration
    
    return None


# =============================================================================
# Summarize Metric Adapter
# =============================================================================

def summarize_metric_adapter(
    slice_name: str,
    curriculum_path: Path = DEFAULT_CURRICULUM_PATH
) -> str:
    """
    Generates a human-readable specification summary for a slice's metric adapter.
    
    Args:
        slice_name: Name of the slice to summarize.
        curriculum_path: Path to the curriculum YAML file.
        
    Returns:
        Multi-line string with human-readable spec.
        
    Raises:
        ValueError: If slice not found or invalid configuration.
    """
    curriculum = load_curriculum(curriculum_path)
    slice_config = get_slice_config(curriculum, slice_name)
    
    if slice_config is None:
        raise ValueError(f"Slice '{slice_name}' not found in curriculum.")
    
    # Extract metric configuration
    metric_config = slice_config.get("success_metric", {})
    metric_kind_str = metric_config.get("kind", "UNKNOWN")
    parameters = metric_config.get("parameters", {})
    
    # Get harmonization table entry
    try:
        metric_kind = MetricKind(metric_kind_str)
        harm_entry = METRIC_HARMONIZATION_TABLE.get(metric_kind, {})
    except ValueError:
        metric_kind = None
        harm_entry = {}
    
    # Build summary
    lines = [
        "=" * 72,
        f"METRIC ADAPTER SUMMARY: {slice_name}",
        "=" * 72,
        "",
        "SLICE CONFIGURATION",
        "-" * 36,
        f"  Slice Name:        {slice_name}",
        f"  Description:       {slice_config.get('description', 'N/A')[:60]}...",
        "",
        "METRIC SPECIFICATION",
        "-" * 36,
        f"  Metric Kind:       {metric_kind_str}",
        f"  Compute Function:  {harm_entry.get('compute_function', 'N/A')}",
        f"  Interpretation:    {harm_entry.get('result_interpretation', 'N/A')}",
        "",
        "PARAMETERS",
        "-" * 36,
    ]
    
    for param, value in parameters.items():
        lines.append(f"  {param}: {value}")
    
    if not parameters:
        lines.append("  (no parameters defined)")
    
    lines.extend([
        "",
        "REQUIRED LOG FIELDS",
        "-" * 36,
    ])
    
    for field_name in harm_entry.get("required_log_fields", []):
        lines.append(f"  - {field_name}")
    
    if not harm_entry.get("required_log_fields"):
        lines.append("  (none specified)")
    
    lines.extend([
        "",
        "RUNTIME FIELDS (must be provided at execution)",
        "-" * 36,
    ])
    
    for field_name in harm_entry.get("runtime_fields", []):
        lines.append(f"  - {field_name}")
    
    if not harm_entry.get("runtime_fields"):
        lines.append("  (none required)")
    
    lines.extend([
        "",
        "OUTPUT SCHEMA",
        "-" * 36,
    ])
    
    output_schema = harm_entry.get("output_schema", {})
    for key, type_desc in output_schema.items():
        if isinstance(type_desc, dict):
            lines.append(f"  {key}:")
            for sub_key, sub_type in type_desc.items():
                lines.append(f"    {sub_key}: {sub_type}")
        else:
            lines.append(f"  {key}: {type_desc}")
    
    # Compute config hash for stability tracking
    config_hash = _compute_config_hash(slice_config, metric_config)
    
    lines.extend([
        "",
        "STABILITY TRACKING",
        "-" * 36,
        f"  Config Hash:       {config_hash[:16]}...",
        f"  Schema Version:    {INTROSPECTION_SCHEMA_VERSION}",
        "",
        "=" * 72,
    ])
    
    return "\n".join(lines)


# =============================================================================
# Verify Metric Alignment
# =============================================================================

def verify_metric_alignment(
    slice_name: str,
    curriculum_path: Path = DEFAULT_CURRICULUM_PATH,
    prereg_path: Path = DEFAULT_PREREG_PATH
) -> AlignmentReport:
    """
    Verifies that harmonization table, prereg, slice config, and adapter all agree.
    
    Args:
        slice_name: Name of the slice to verify.
        curriculum_path: Path to the curriculum YAML file.
        prereg_path: Path to the PREREG YAML file.
        
    Returns:
        AlignmentReport with detailed verification results.
    """
    errors: List[str] = []
    warnings: List[str] = []
    drifts: List[DriftItem] = []
    
    # Load sources
    try:
        curriculum = load_curriculum(curriculum_path)
    except FileNotFoundError as e:
        return AlignmentReport(
            slice_name=slice_name,
            status=AlignmentStatus.MISSING,
            harmonization_aligned=False,
            prereg_aligned=False,
            adapter_aligned=False,
            errors=[str(e)]
        )
    
    try:
        prereg = load_prereg(prereg_path)
    except FileNotFoundError:
        prereg = {}
        warnings.append(f"PREREG file not found at {prereg_path}")
    
    # Get slice config
    slice_config = get_slice_config(curriculum, slice_name)
    if slice_config is None:
        return AlignmentReport(
            slice_name=slice_name,
            status=AlignmentStatus.MISSING,
            harmonization_aligned=False,
            prereg_aligned=False,
            adapter_aligned=False,
            errors=[f"Slice '{slice_name}' not found in curriculum."]
        )
    
    metric_config = slice_config.get("success_metric", {})
    metric_kind_str = metric_config.get("kind", "")
    
    # Check 1: Harmonization table alignment
    harmonization_aligned = True
    try:
        metric_kind = MetricKind(metric_kind_str)
        harm_entry = METRIC_HARMONIZATION_TABLE.get(metric_kind)
        
        if harm_entry is None:
            harmonization_aligned = False
            errors.append(f"Metric kind '{metric_kind_str}' has no harmonization table entry.")
        else:
            # Verify required config fields exist
            for field_path in harm_entry.get("required_slice_config_fields", []):
                if not _check_nested_field(slice_config, field_path):
                    harmonization_aligned = False
                    drifts.append(DriftItem(
                        field=field_path,
                        source_a="harmonization_table",
                        source_b="slice_config",
                        value_a="required",
                        value_b="missing",
                        severity="error",
                        description=f"Required field '{field_path}' missing from slice config."
                    ))
    except ValueError:
        harmonization_aligned = False
        errors.append(f"Unknown metric kind '{metric_kind_str}' - not in MetricKind enum.")
    
    # Check 2: PREREG alignment
    prereg_aligned = True
    prereg_spec = get_prereg_slice_spec(prereg, slice_name)
    
    if prereg_spec is None:
        prereg_aligned = False
        warnings.append(f"No PREREG spec found for slice '{slice_name}'.")
    else:
        prereg_metric = prereg_spec.get("success_metric", {})
        prereg_kind = prereg_metric.get("kind", "")
        
        # Check kind alignment with curriculum metric kind mapping
        # Note: PREREG uses different kind names (density vs sparse_success, etc.)
        kind_mapping = {
            "goal_hit": "goal_hit",
            "density": "sparse_success",  # PREREG calls it "density"
            "chain_length": "chain_success",  # PREREG calls it "chain_length"
            "multi_goal": "multi_goal_success",  # PREREG calls it "multi_goal"
            "sparse_success": "sparse_success",
            "chain_success": "chain_success",
            "multi_goal_success": "multi_goal_success",
        }
        
        expected_kind = kind_mapping.get(prereg_kind, prereg_kind)
        if expected_kind != metric_kind_str:
            prereg_aligned = False
            drifts.append(DriftItem(
                field="success_metric.kind",
                source_a="prereg",
                source_b="curriculum",
                value_a=prereg_kind,
                value_b=metric_kind_str,
                severity="error",
                description=f"Metric kind mismatch: PREREG='{prereg_kind}', curriculum='{metric_kind_str}'."
            ))
        
        # Check parameter alignment
        prereg_params = prereg_metric.get("parameters", {})
        config_params = metric_config.get("parameters", {})
        
        # Check for parameters in prereg but not in config
        for param in prereg_params:
            if param not in config_params and param not in ["target_hashes", "chain_target_hash", "required_goal_hashes"]:
                # Runtime fields are allowed to be missing from config
                drifts.append(DriftItem(
                    field=f"parameters.{param}",
                    source_a="prereg",
                    source_b="curriculum",
                    value_a=prereg_params[param],
                    value_b="missing",
                    severity="warning",
                    description=f"Parameter '{param}' in PREREG but not in curriculum config."
                ))
    
    # Check 3: Adapter alignment (can we construct a valid adapter?)
    adapter_aligned = True
    try:
        adapter = SliceMetricAdapter(slice_config, slice_name)
        
        # Verify adapter metric kind matches
        if adapter.metric_kind.value != metric_kind_str:
            adapter_aligned = False
            drifts.append(DriftItem(
                field="metric_kind",
                source_a="adapter",
                source_b="curriculum",
                value_a=adapter.metric_kind.value,
                value_b=metric_kind_str,
                severity="error",
                description="Adapter metric kind doesn't match curriculum."
            ))
    except ValueError as e:
        adapter_aligned = False
        errors.append(f"Failed to construct adapter: {e}")
    
    # Determine overall status
    if harmonization_aligned and prereg_aligned and adapter_aligned and len(errors) == 0:
        status = AlignmentStatus.ALIGNED
    elif len(errors) > 0:
        status = AlignmentStatus.MISALIGNED
    else:
        status = AlignmentStatus.PARTIAL
    
    return AlignmentReport(
        slice_name=slice_name,
        status=status,
        harmonization_aligned=harmonization_aligned,
        prereg_aligned=prereg_aligned,
        adapter_aligned=adapter_aligned,
        drifts=drifts,
        errors=errors,
        warnings=warnings
    )


def _check_nested_field(config: Dict[str, Any], field_path: str) -> bool:
    """
    Checks if a nested field path exists in a config dict.
    
    Field path uses dot notation, e.g., "success_metric.parameters.min_verified".
    """
    parts = field_path.split(".")
    current = config
    
    for part in parts:
        if not isinstance(current, dict) or part not in current:
            return False
        current = current[part]
    
    return True


# =============================================================================
# Export Metric Contract
# =============================================================================

def export_metric_contract(
    slice_name: str,
    curriculum_path: Path = DEFAULT_CURRICULUM_PATH
) -> MetricContractSchema:
    """
    Exports a JSON schema contract for an adapter–slice pair.
    
    Args:
        slice_name: Name of the slice.
        curriculum_path: Path to the curriculum YAML file.
        
    Returns:
        MetricContractSchema representing the contract.
        
    Raises:
        ValueError: If slice not found or invalid configuration.
    """
    curriculum = load_curriculum(curriculum_path)
    slice_config = get_slice_config(curriculum, slice_name)
    
    if slice_config is None:
        raise ValueError(f"Slice '{slice_name}' not found in curriculum.")
    
    metric_config = slice_config.get("success_metric", {})
    metric_kind_str = metric_config.get("kind", "")
    parameters = metric_config.get("parameters", {})
    
    try:
        metric_kind = MetricKind(metric_kind_str)
    except ValueError:
        raise ValueError(f"Unknown metric kind '{metric_kind_str}'.")
    
    harm_entry = METRIC_HARMONIZATION_TABLE.get(metric_kind, {})
    
    # Build parameter schema from validator
    validator = MetricSchemaValidator()
    prereg_schema = validator.PREREG_SCHEMA.get(metric_kind, {})
    param_types = prereg_schema.get("parameter_types", {})
    
    parameter_schema = {
        param: _type_to_string(ptype)
        for param, ptype in param_types.items()
    }
    
    # Compute config hash
    config_hash = _compute_config_hash(slice_config, metric_config)
    
    return MetricContractSchema(
        schema_version=INTROSPECTION_SCHEMA_VERSION,
        slice_name=slice_name,
        metric_kind=metric_kind_str,
        required_config_fields=tuple(harm_entry.get("required_slice_config_fields", [])),
        required_log_fields=tuple(harm_entry.get("required_log_fields", [])),
        runtime_fields=tuple(harm_entry.get("runtime_fields", [])),
        parameter_schema=parameter_schema,
        output_schema=harm_entry.get("output_schema", {}),
        config_hash=config_hash
    )


def _type_to_string(ptype: type) -> str:
    """Converts a Python type to a string representation."""
    if ptype is int:
        return "integer"
    elif ptype is float:
        return "number"
    elif ptype is str:
        return "string"
    elif ptype is bool:
        return "boolean"
    elif ptype is list:
        return "array"
    elif ptype is dict:
        return "object"
    else:
        return str(ptype.__name__)


def _compute_config_hash(
    slice_config: Dict[str, Any],
    metric_config: Dict[str, Any]
) -> str:
    """
    Computes a deterministic hash of the configuration.
    
    Used for stability tracking and drift detection.
    """
    # Create a canonical representation
    canonical_data = {
        "metric_kind": metric_config.get("kind", ""),
        "parameters": metric_config.get("parameters", {}),
    }
    canonical_json = json.dumps(canonical_data, sort_keys=True)
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


# =============================================================================
# Deep-Diff Detection
# =============================================================================

def detect_schema_drift(
    slice_name: str,
    curriculum_path: Path = DEFAULT_CURRICULUM_PATH,
    prereg_path: Path = DEFAULT_PREREG_PATH
) -> List[DriftItem]:
    """
    Detects silent schema drift between prereg, adapter expectations, and harmonization table.
    
    Args:
        slice_name: Name of the slice to check.
        curriculum_path: Path to the curriculum YAML file.
        prereg_path: Path to the PREREG YAML file.
        
    Returns:
        List of DriftItem objects describing detected drift.
    """
    drifts: List[DriftItem] = []
    
    # Load all sources
    try:
        curriculum = load_curriculum(curriculum_path)
    except FileNotFoundError:
        drifts.append(DriftItem(
            field="curriculum_file",
            source_a="filesystem",
            source_b="expected",
            value_a="missing",
            value_b=str(curriculum_path),
            severity="error",
            description="Curriculum file not found."
        ))
        return drifts
    
    try:
        prereg = load_prereg(prereg_path)
    except FileNotFoundError:
        prereg = {}
    
    slice_config = get_slice_config(curriculum, slice_name)
    if slice_config is None:
        drifts.append(DriftItem(
            field="slice",
            source_a="curriculum",
            source_b="expected",
            value_a="missing",
            value_b=slice_name,
            severity="error",
            description=f"Slice '{slice_name}' not found in curriculum."
        ))
        return drifts
    
    metric_config = slice_config.get("success_metric", {})
    metric_kind_str = metric_config.get("kind", "")
    config_params = metric_config.get("parameters", {})
    
    # Get harmonization table expectations
    try:
        metric_kind = MetricKind(metric_kind_str)
        harm_entry = METRIC_HARMONIZATION_TABLE.get(metric_kind, {})
    except ValueError:
        harm_entry = {}
    
    # Get PREREG expectations
    prereg_spec = get_prereg_slice_spec(prereg, slice_name)
    prereg_params = {}
    if prereg_spec:
        prereg_metric = prereg_spec.get("success_metric", {})
        prereg_params = prereg_metric.get("parameters", {})
    
    # Get adapter (validator) expectations
    validator = MetricSchemaValidator()
    try:
        adapter_schema = validator.PREREG_SCHEMA.get(metric_kind, {})
    except:
        adapter_schema = {}
    
    required_by_adapter = set(adapter_schema.get("required_parameters", []))
    optional_by_adapter = set(adapter_schema.get("optional_parameters", []))
    all_adapter_params = required_by_adapter | optional_by_adapter
    
    # Drift detection: PREREG vs Adapter
    for param in prereg_params:
        if param not in all_adapter_params:
            drifts.append(DriftItem(
                field=f"parameters.{param}",
                source_a="prereg",
                source_b="adapter_schema",
                value_a="defined",
                value_b="not_recognized",
                severity="warning",
                description=f"PREREG defines parameter '{param}' not recognized by adapter schema."
            ))
    
    # Drift detection: Adapter vs Config
    for param in required_by_adapter:
        if param not in config_params:
            drifts.append(DriftItem(
                field=f"parameters.{param}",
                source_a="adapter_schema",
                source_b="curriculum_config",
                value_a="required",
                value_b="missing",
                severity="error",
                description=f"Adapter requires parameter '{param}' missing from curriculum config."
            ))
    
    # Drift detection: Config vs Adapter (extraneous params)
    for param in config_params:
        if param not in all_adapter_params:
            drifts.append(DriftItem(
                field=f"parameters.{param}",
                source_a="curriculum_config",
                source_b="adapter_schema",
                value_a="defined",
                value_b="not_recognized",
                severity="warning",
                description=f"Curriculum config has parameter '{param}' not in adapter schema."
            ))
    
    # Drift detection: Harmonization required fields vs Config
    for field_path in harm_entry.get("required_slice_config_fields", []):
        if not _check_nested_field(slice_config, field_path):
            drifts.append(DriftItem(
                field=field_path,
                source_a="harmonization_table",
                source_b="curriculum_config",
                value_a="required",
                value_b="missing",
                severity="error",
                description=f"Harmonization table requires '{field_path}' but it's missing from config."
            ))
    
    return drifts


# =============================================================================
# Batch Operations
# =============================================================================

def list_all_slices(curriculum_path: Path = DEFAULT_CURRICULUM_PATH) -> List[str]:
    """
    Lists all slice names in the curriculum.
    """
    curriculum = load_curriculum(curriculum_path)
    slices = curriculum.get("slices", {})
    return sorted(slices.keys())


def verify_all_slices(
    curriculum_path: Path = DEFAULT_CURRICULUM_PATH,
    prereg_path: Path = DEFAULT_PREREG_PATH
) -> Dict[str, AlignmentReport]:
    """
    Verifies metric alignment for all slices in the curriculum.
    
    Returns a dict mapping slice_name -> AlignmentReport.
    """
    results = {}
    for slice_name in list_all_slices(curriculum_path):
        results[slice_name] = verify_metric_alignment(
            slice_name, curriculum_path, prereg_path
        )
    return results


def export_all_contracts(
    curriculum_path: Path = DEFAULT_CURRICULUM_PATH,
    output_dir: Optional[Path] = None
) -> Dict[str, MetricContractSchema]:
    """
    Exports metric contracts for all slices.
    
    If output_dir is provided, writes each contract to a JSON file.
    
    Returns a dict mapping slice_name -> MetricContractSchema.
    """
    results = {}
    
    for slice_name in list_all_slices(curriculum_path):
        try:
            contract = export_metric_contract(slice_name, curriculum_path)
            results[slice_name] = contract
            
            if output_dir is not None:
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"{slice_name}_contract.json"
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(contract.to_json())
        except ValueError as e:
            # Skip slices that can't be processed
            pass
    
    return results


# =============================================================================
# Health Check for CI
# =============================================================================

@dataclass
class SliceHealthResult:
    """Health check result for a single slice."""
    slice_name: str
    status: HealthStatus
    alignment_status: AlignmentStatus
    errors: List[str]
    warnings: List[str]
    drift_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "slice_name": self.slice_name,
            "status": self.status.value,
            "alignment_status": self.alignment_status.value,
            "errors": self.errors,
            "warnings": self.warnings,
            "drift_count": self.drift_count,
        }


@dataclass
class HealthCheckReport:
    """Complete health check report for all slices."""
    timestamp: str
    total_slices: int
    ok_count: int
    warn_count: int
    fail_count: int
    slices: Dict[str, SliceHealthResult]
    overall_status: HealthStatus
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "total_slices": self.total_slices,
            "ok_count": self.ok_count,
            "warn_count": self.warn_count,
            "fail_count": self.fail_count,
            "overall_status": self.overall_status.value,
            "slices": {
                name: result.to_dict()
                for name, result in self.slices.items()
            },
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


def run_health_check(
    curriculum_path: Path = DEFAULT_CURRICULUM_PATH,
    prereg_path: Path = DEFAULT_PREREG_PATH
) -> HealthCheckReport:
    """
    Runs comprehensive health check on all slices for CI integration.
    
    For each slice:
      - Runs verify_metric_alignment()
      - Runs detect_schema_drift()
      - Determines OK / WARN / FAIL status
    
    Returns HealthCheckReport with detailed results.
    
    Status logic:
      - FAIL: Any error-level issues (missing required params, unknown metric kind, etc.)
      - WARN: Only warning-level issues (missing prereg, extraneous params, etc.)
      - OK: No issues at all
    """
    from datetime import datetime, timezone
    
    timestamp = datetime.now(timezone.utc).isoformat()
    slices = list_all_slices(curriculum_path)
    slice_results: Dict[str, SliceHealthResult] = {}
    
    ok_count = 0
    warn_count = 0
    fail_count = 0
    
    for slice_name in slices:
        # Run alignment verification
        alignment_report = verify_metric_alignment(
            slice_name, curriculum_path, prereg_path
        )
        
        # Run drift detection
        drifts = detect_schema_drift(slice_name, curriculum_path, prereg_path)
        error_drifts = [d for d in drifts if d.severity == "error"]
        warning_drifts = [d for d in drifts if d.severity == "warning"]
        
        # Combine errors and warnings
        errors = list(alignment_report.errors)
        for drift in error_drifts:
            errors.append(drift.description)
        
        warnings = list(alignment_report.warnings)
        for drift in warning_drifts:
            warnings.append(drift.description)
        
        # Determine status
        if errors or alignment_report.status == AlignmentStatus.MISALIGNED:
            status = HealthStatus.FAIL
            fail_count += 1
        elif warnings or alignment_report.status == AlignmentStatus.PARTIAL:
            status = HealthStatus.WARN
            warn_count += 1
        else:
            status = HealthStatus.OK
            ok_count += 1
        
        slice_results[slice_name] = SliceHealthResult(
            slice_name=slice_name,
            status=status,
            alignment_status=alignment_report.status,
            errors=errors,
            warnings=warnings,
            drift_count=len(drifts),
        )
    
    # Determine overall status
    if fail_count > 0:
        overall_status = HealthStatus.FAIL
    elif warn_count > 0:
        overall_status = HealthStatus.WARN
    else:
        overall_status = HealthStatus.OK
    
    return HealthCheckReport(
        timestamp=timestamp,
        total_slices=len(slices),
        ok_count=ok_count,
        warn_count=warn_count,
        fail_count=fail_count,
        slices=slice_results,
        overall_status=overall_status,
    )


def format_health_check_report(report: HealthCheckReport) -> str:
    """
    Formats a HealthCheckReport as a human-readable string for CLI output.
    """
    status_icons = {
        HealthStatus.OK: "✓",
        HealthStatus.WARN: "!",
        HealthStatus.FAIL: "✗",
    }
    
    lines = [
        "=" * 72,
        "METRIC ADAPTER HEALTH CHECK",
        "=" * 72,
        "",
        f"Timestamp: {report.timestamp}",
        f"Overall Status: {status_icons[report.overall_status]} {report.overall_status.value}",
        "",
        f"Summary: {report.ok_count} OK / {report.warn_count} WARN / {report.fail_count} FAIL",
        "",
        "-" * 72,
        "PER-SLICE RESULTS",
        "-" * 72,
    ]
    
    for slice_name in sorted(report.slices.keys()):
        result = report.slices[slice_name]
        icon = status_icons[result.status]
        lines.append(f"  {icon} {slice_name}: {result.status.value}")
        
        if result.errors:
            for error in result.errors[:3]:  # Show first 3 errors
                lines.append(f"      ERROR: {error[:60]}...")
            if len(result.errors) > 3:
                lines.append(f"      ... and {len(result.errors) - 3} more errors")
        
        if result.warnings:
            for warning in result.warnings[:2]:  # Show first 2 warnings
                lines.append(f"      WARN: {warning[:60]}...")
            if len(result.warnings) > 2:
                lines.append(f"      ... and {len(result.warnings) - 2} more warnings")
    
    lines.extend([
        "",
        "-" * 72,
        f"Exit code: {0 if report.overall_status != HealthStatus.FAIL else 1}",
        "=" * 72,
    ])
    
    return "\n".join(lines)


# =============================================================================
# Contract Bundle Export
# =============================================================================

@dataclass
class ContractIndexEntry:
    """Entry in the contract index."""
    slice_name: str
    contract_path: str
    config_hash: str
    schema_version: str
    metric_kind: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "slice_name": self.slice_name,
            "contract_path": self.contract_path,
            "config_hash": self.config_hash,
            "schema_version": self.schema_version,
            "metric_kind": self.metric_kind,
        }


@dataclass
class ContractBundle:
    """Complete contract bundle with index."""
    generated_at: str
    schema_version: str
    total_contracts: int
    output_directory: str
    contracts: List[ContractIndexEntry]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "schema_version": self.schema_version,
            "total_contracts": self.total_contracts,
            "output_directory": self.output_directory,
            "contracts": [c.to_dict() for c in self.contracts],
        }
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


def export_contract_bundle(
    curriculum_path: Path = DEFAULT_CURRICULUM_PATH,
    output_dir: Path = DEFAULT_CONTRACT_OUTPUT_DIR
) -> ContractBundle:
    """
    Exports a complete contract bundle with index file.
    
    Creates:
      - artifacts/phase_ii/metric_contracts/<slice>.json for each slice
      - artifacts/phase_ii/metric_contracts/metric_contract_index.json
    
    Returns ContractBundle with all metadata.
    """
    from datetime import datetime, timezone
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now(timezone.utc).isoformat()
    contracts: List[ContractIndexEntry] = []
    
    for slice_name in list_all_slices(curriculum_path):
        try:
            contract = export_metric_contract(slice_name, curriculum_path)
            
            # Write individual contract file
            contract_filename = f"{slice_name}.json"
            contract_path = output_dir / contract_filename
            with open(contract_path, "w", encoding="utf-8") as f:
                f.write(contract.to_json())
            
            # Add to index
            contracts.append(ContractIndexEntry(
                slice_name=slice_name,
                contract_path=contract_filename,
                config_hash=contract.config_hash,
                schema_version=contract.schema_version,
                metric_kind=contract.metric_kind,
            ))
        except ValueError:
            # Skip invalid slices
            pass
    
    # Create bundle
    bundle = ContractBundle(
        generated_at=timestamp,
        schema_version=INTROSPECTION_SCHEMA_VERSION,
        total_contracts=len(contracts),
        output_directory=str(output_dir),
        contracts=contracts,
    )
    
    # Write index file
    index_path = output_dir / "metric_contract_index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(bundle.to_json())
    
    return bundle


# =============================================================================
# Metric Dashboard
# =============================================================================

def generate_metric_dashboard(
    curriculum_path: Path = DEFAULT_CURRICULUM_PATH
) -> str:
    """
    Generates a developer-friendly dashboard showing metric configuration for all slices.
    
    For each slice shows:
      - Metric kind
      - Required parameters
      - Optional parameters
      - Expected JSONL fields the adapter reads
    
    Focus: Developer UX — zero semantic changes.
    """
    lines = [
        "=" * 72,
        "METRIC ADAPTER DASHBOARD",
        "=" * 72,
        "",
    ]
    
    slices = list_all_slices(curriculum_path)
    curriculum = load_curriculum(curriculum_path)
    validator = MetricSchemaValidator()
    
    for slice_name in slices:
        slice_config = get_slice_config(curriculum, slice_name)
        if slice_config is None:
            continue
        
        metric_config = slice_config.get("success_metric", {})
        metric_kind_str = metric_config.get("kind", "UNKNOWN")
        config_params = metric_config.get("parameters", {})
        
        try:
            metric_kind = MetricKind(metric_kind_str)
            harm_entry = METRIC_HARMONIZATION_TABLE.get(metric_kind, {})
            prereg_schema = validator.PREREG_SCHEMA.get(metric_kind, {})
        except ValueError:
            harm_entry = {}
            prereg_schema = {}
        
        required_params = prereg_schema.get("required_parameters", [])
        optional_params = prereg_schema.get("optional_parameters", [])
        log_fields = harm_entry.get("required_log_fields", [])
        runtime_fields = harm_entry.get("runtime_fields", [])
        
        lines.extend([
            f"┌─ {slice_name}",
            f"│  Metric Kind: {metric_kind_str}",
            f"│",
            f"│  Required Parameters:",
        ])
        
        for param in required_params:
            value = config_params.get(param, "(not set)")
            lines.append(f"│    • {param}: {value}")
        
        if not required_params:
            lines.append(f"│    (none)")
        
        lines.append(f"│")
        lines.append(f"│  Optional Parameters:")
        
        for param in optional_params:
            if param in config_params:
                lines.append(f"│    • {param}: {config_params[param]}")
            else:
                lines.append(f"│    • {param}: (not set)")
        
        if not optional_params:
            lines.append(f"│    (none)")
        
        lines.append(f"│")
        lines.append(f"│  JSONL Fields Read by Adapter:")
        
        for field in log_fields:
            lines.append(f"│    • {field}")
        
        if not log_fields:
            lines.append(f"│    (none)")
        
        if runtime_fields:
            lines.append(f"│")
            lines.append(f"│  Runtime Fields (provided at execution):")
            for field in runtime_fields:
                lines.append(f"│    • {field}")
        
        lines.append(f"└{'─' * 71}")
        lines.append("")
    
    lines.extend([
        "=" * 72,
        f"Total slices: {len(slices)}",
        "=" * 72,
    ])
    
    return "\n".join(lines)


def generate_metric_dashboard_json(
    curriculum_path: Path = DEFAULT_CURRICULUM_PATH
) -> Dict[str, Any]:
    """
    Generates dashboard data in JSON format.
    """
    slices_data = {}
    
    curriculum = load_curriculum(curriculum_path)
    validator = MetricSchemaValidator()
    
    for slice_name in list_all_slices(curriculum_path):
        slice_config = get_slice_config(curriculum, slice_name)
        if slice_config is None:
            continue
        
        metric_config = slice_config.get("success_metric", {})
        metric_kind_str = metric_config.get("kind", "UNKNOWN")
        config_params = metric_config.get("parameters", {})
        
        try:
            metric_kind = MetricKind(metric_kind_str)
            harm_entry = METRIC_HARMONIZATION_TABLE.get(metric_kind, {})
            prereg_schema = validator.PREREG_SCHEMA.get(metric_kind, {})
        except ValueError:
            harm_entry = {}
            prereg_schema = {}
        
        slices_data[slice_name] = {
            "metric_kind": metric_kind_str,
            "required_parameters": {
                param: config_params.get(param)
                for param in prereg_schema.get("required_parameters", [])
            },
            "optional_parameters": {
                param: config_params.get(param)
                for param in prereg_schema.get("optional_parameters", [])
            },
            "jsonl_fields_read": harm_entry.get("required_log_fields", []),
            "runtime_fields": harm_entry.get("runtime_fields", []),
        }
    
    return {
        "schema_version": INTROSPECTION_SCHEMA_VERSION,
        "total_slices": len(slices_data),
        "slices": slices_data,
    }


# =============================================================================
# Report Formatting
# =============================================================================

def format_alignment_report(report: AlignmentReport) -> str:
    """
    Formats an AlignmentReport as a human-readable string.
    """
    status_icon = {
        AlignmentStatus.ALIGNED: "✓",
        AlignmentStatus.MISALIGNED: "✗",
        AlignmentStatus.PARTIAL: "~",
        AlignmentStatus.MISSING: "?",
    }
    
    lines = [
        "=" * 72,
        f"ALIGNMENT REPORT: {report.slice_name}",
        "=" * 72,
        "",
        f"Overall Status: {status_icon.get(report.status, '?')} {report.status.value.upper()}",
        "",
        "SOURCE ALIGNMENT",
        "-" * 36,
        f"  Harmonization Table: {'✓' if report.harmonization_aligned else '✗'}",
        f"  PREREG Spec:         {'✓' if report.prereg_aligned else '✗'}",
        f"  Adapter:             {'✓' if report.adapter_aligned else '✗'}",
    ]
    
    if report.errors:
        lines.extend([
            "",
            "ERRORS",
            "-" * 36,
        ])
        for error in report.errors:
            lines.append(f"  ✗ {error}")
    
    if report.warnings:
        lines.extend([
            "",
            "WARNINGS",
            "-" * 36,
        ])
        for warning in report.warnings:
            lines.append(f"  ! {warning}")
    
    if report.drifts:
        lines.extend([
            "",
            "SCHEMA DRIFTS",
            "-" * 36,
        ])
        for drift in report.drifts:
            icon = "✗" if drift.severity == "error" else "!" if drift.severity == "warning" else "i"
            lines.append(f"  {icon} [{drift.source_a} vs {drift.source_b}] {drift.field}")
            lines.append(f"      {drift.description}")
    
    lines.extend(["", "=" * 72])
    
    return "\n".join(lines)


def format_drift_report(drifts: List[DriftItem]) -> str:
    """
    Formats a list of DriftItems as a human-readable report.
    """
    if not drifts:
        return "No schema drift detected."
    
    lines = [
        "=" * 72,
        "SCHEMA DRIFT REPORT",
        "=" * 72,
        "",
    ]
    
    # Group by severity
    errors = [d for d in drifts if d.severity == "error"]
    warnings = [d for d in drifts if d.severity == "warning"]
    infos = [d for d in drifts if d.severity == "info"]
    
    if errors:
        lines.extend([
            "ERRORS (must fix)",
            "-" * 36,
        ])
        for drift in errors:
            lines.append(f"  ✗ {drift.field}")
            lines.append(f"    Source: {drift.source_a} → {drift.source_b}")
            lines.append(f"    Values: '{drift.value_a}' vs '{drift.value_b}'")
            lines.append(f"    {drift.description}")
            lines.append("")
    
    if warnings:
        lines.extend([
            "WARNINGS (should review)",
            "-" * 36,
        ])
        for drift in warnings:
            lines.append(f"  ! {drift.field}")
            lines.append(f"    Source: {drift.source_a} → {drift.source_b}")
            lines.append(f"    {drift.description}")
            lines.append("")
    
    if infos:
        lines.extend([
            "INFO",
            "-" * 36,
        ])
        for drift in infos:
            lines.append(f"  i {drift.field}: {drift.description}")
    
    lines.extend(["=" * 72])
    
    return "\n".join(lines)


# =============================================================================
# Per-Slice Metric Readiness Gate
# =============================================================================

def classify_drift_severity(drifts: List[DriftItem]) -> DriftSeverityClass:
    """
    Classifies the overall drift severity from a list of drift items.
    
    Classification logic:
      - NONE: No drifts at all
      - COSMETIC: Only info-level drifts (formatting, naming)
      - PARAMETRIC_MINOR: Warning-level param drifts but still callable
      - SEMANTIC: Any error-level drift (kind mismatch, missing required fields)
    
    Returns the most severe classification found.
    """
    if not drifts:
        return DriftSeverityClass.NONE
    
    # Check for semantic (error-level) drifts
    error_drifts = [d for d in drifts if d.severity == "error"]
    if error_drifts:
        # Check if it's a kind mismatch or missing required field
        for d in error_drifts:
            if "kind" in d.field.lower() or "missing" in d.value_b.lower():
                return DriftSeverityClass.SEMANTIC
        return DriftSeverityClass.SEMANTIC
    
    # Check for parametric (warning-level) drifts
    warning_drifts = [d for d in drifts if d.severity == "warning"]
    if warning_drifts:
        return DriftSeverityClass.PARAMETRIC_MINOR
    
    # Only info-level drifts = cosmetic
    info_drifts = [d for d in drifts if d.severity == "info"]
    if info_drifts:
        return DriftSeverityClass.COSMETIC
    
    return DriftSeverityClass.NONE


def check_slice_readiness(
    slice_name: str,
    curriculum_path: Path = DEFAULT_CURRICULUM_PATH,
    prereg_path: Path = DEFAULT_PREREG_PATH,
    contract_dir: Path = DEFAULT_CONTRACT_OUTPUT_DIR
) -> ReadinessResult:
    """
    Per-slice pre-flight readiness check for CI gating.
    
    This is a formalized readiness contract that returns a stable JSON structure.
    
    Runs:
      - verify_metric_alignment(slice)
      - detect_schema_drift(slice)
      - Checks for contract artifacts
      - Collects missing/unknown fields
      - Includes log field coverage
    
    Returns ReadinessResult with status:
      - READY: Alignment passes, drift is NONE or COSMETIC only, contract exists
      - DEGRADED: Param drift but still callable (PARAMETRIC_MINOR), or partial coverage
      - BLOCKED: SEMANTIC drift, missing contract artifacts, or structural failures
    
    Exit codes for CLI:
      - 0 → READY or DEGRADED
      - 1 → BLOCKED
      - 2 → internal error
    
    Args:
        slice_name: Name of the slice to check.
        curriculum_path: Path to the curriculum YAML file.
        prereg_path: Path to the PREREG YAML file.
        contract_dir: Path to the contract bundle directory.
        
    Returns:
        ReadinessResult with detailed status including extended contract fields.
    """
    errors: List[str] = []
    warnings: List[str] = []
    missing_fields: List[str] = []
    unknown_fields: List[str] = []
    alignment_issues: List[str] = []
    missing_contract = False
    log_field_coverage_dict: Optional[Dict[str, Any]] = None
    
    # Check if contract exists
    contract_path = contract_dir / f"{slice_name}.json"
    if not contract_path.exists():
        missing_contract = True
        errors.append(f"Contract file not found: {contract_path}")
    
    # Run alignment verification
    alignment_report = verify_metric_alignment(slice_name, curriculum_path, prereg_path)
    
    # Run drift detection
    drifts = detect_schema_drift(slice_name, curriculum_path, prereg_path)
    
    # Classify drift severity
    drift_class = classify_drift_severity(drifts)
    
    # Determine if alignment passes
    alignment_passes = alignment_report.is_fully_aligned()
    
    # Collect errors from alignment report
    errors.extend(alignment_report.errors)
    
    # Collect warnings from alignment report
    warnings.extend(alignment_report.warnings)
    
    # Collect errors/warnings from drifts and categorize them
    for drift in drifts:
        if drift.severity == "error":
            errors.append(drift.description)
            # Categorize as missing or unknown field
            if "missing" in drift.value_b.lower() or "missing" in drift.description.lower():
                missing_fields.append(drift.field)
            if "not_recognized" in drift.value_b.lower() or "unknown" in drift.description.lower():
                unknown_fields.append(drift.field)
        elif drift.severity == "warning":
            warnings.append(drift.description)
            if "not_recognized" in drift.value_b.lower():
                unknown_fields.append(drift.field)
    
    # Collect alignment issues from alignment report drifts
    for drift in alignment_report.drifts:
        alignment_issues.append(drift.description)
    
    # Get log field coverage for this slice
    try:
        coverage = get_log_field_coverage(slice_name, curriculum_path)
        log_field_coverage_dict = coverage.to_dict()
    except (ValueError, FileNotFoundError):
        # Slice may not exist or have invalid config
        log_field_coverage_dict = None
    
    # Determine readiness status
    if missing_contract:
        status = ReadinessStatus.BLOCKED
    elif drift_class == DriftSeverityClass.SEMANTIC:
        status = ReadinessStatus.BLOCKED
    elif alignment_report.status == AlignmentStatus.MISSING:
        status = ReadinessStatus.BLOCKED
    elif alignment_report.status == AlignmentStatus.MISALIGNED:
        status = ReadinessStatus.BLOCKED
    elif drift_class == DriftSeverityClass.PARAMETRIC_MINOR:
        status = ReadinessStatus.DEGRADED
    elif alignment_passes and drift_class in [DriftSeverityClass.NONE, DriftSeverityClass.COSMETIC]:
        status = ReadinessStatus.READY
    else:
        # Partial alignment with no semantic issues = degraded
        status = ReadinessStatus.DEGRADED
    
    return ReadinessResult(
        slice_name=slice_name,
        status=status,
        alignment_passes=alignment_passes,
        drift_class=drift_class,
        errors=errors,
        warnings=warnings,
        drift_items=drifts,
        # Extended contract fields
        missing_contract=missing_contract,
        missing_fields=missing_fields,
        unknown_fields=unknown_fields,
        alignment_issues=alignment_issues,
        log_field_coverage=log_field_coverage_dict,
    )


def format_readiness_result(result: ReadinessResult) -> str:
    """
    Formats a ReadinessResult as a human-readable string for CLI output.
    """
    status_icons = {
        ReadinessStatus.READY: "✓",
        ReadinessStatus.DEGRADED: "~",
        ReadinessStatus.BLOCKED: "✗",
    }
    
    lines = [
        "=" * 72,
        f"METRIC READINESS CHECK: {result.slice_name}",
        "=" * 72,
        "",
        f"Status: {status_icons[result.status]} {result.status.value}",
        f"Drift Severity: {result.drift_class.value}",
        f"Alignment: {'PASS' if result.alignment_passes else 'FAIL'}",
        f"Contract: {'MISSING' if result.missing_contract else 'PRESENT'}",
        "",
    ]
    
    # Show missing fields
    if result.missing_fields:
        lines.extend([
            "MISSING FIELDS",
            "-" * 36,
        ])
        for field in result.missing_fields[:5]:
            lines.append(f"  ✗ {field}")
        if len(result.missing_fields) > 5:
            lines.append(f"  ... and {len(result.missing_fields) - 5} more")
        lines.append("")
    
    # Show unknown fields
    if result.unknown_fields:
        lines.extend([
            "UNKNOWN FIELDS",
            "-" * 36,
        ])
        for field in result.unknown_fields[:5]:
            lines.append(f"  ? {field}")
        if len(result.unknown_fields) > 5:
            lines.append(f"  ... and {len(result.unknown_fields) - 5} more")
        lines.append("")
    
    if result.errors:
        lines.extend([
            "ERRORS",
            "-" * 36,
        ])
        for error in result.errors[:5]:
            lines.append(f"  ✗ {error[:70]}...")
        if len(result.errors) > 5:
            lines.append(f"  ... and {len(result.errors) - 5} more errors")
        lines.append("")
    
    if result.warnings:
        lines.extend([
            "WARNINGS",
            "-" * 36,
        ])
        for warning in result.warnings[:3]:
            lines.append(f"  ! {warning[:70]}...")
        if len(result.warnings) > 3:
            lines.append(f"  ... and {len(result.warnings) - 3} more warnings")
        lines.append("")
    
    # Show log field coverage if available
    if result.log_field_coverage:
        lines.extend([
            "LOG FIELD COVERAGE",
            "-" * 36,
            f"  Metric Kind: {result.log_field_coverage.get('metric_kind', 'N/A')}",
            f"  Required Fields: {', '.join(result.log_field_coverage.get('required_log_fields', []))}",
            f"  Runtime Fields: {', '.join(result.log_field_coverage.get('runtime_fields', [])) or '(none)'}",
            "",
        ])
    
    # Exit code hint
    exit_code = 0 if result.status in [ReadinessStatus.READY, ReadinessStatus.DEGRADED] else 1
    lines.extend([
        "-" * 72,
        f"Exit code: {exit_code}",
        "=" * 72,
    ])
    
    return "\n".join(lines)


# =============================================================================
# Multi-Slice Readiness Summary (Task 1)
# =============================================================================

@dataclass
class ReadinessSummary:
    """
    Aggregated readiness summary for all slices.
    
    This is the canonical multi-slice readiness contract for dashboards
    and governance. The schema is frozen at version 1.0.0.
    
    Contract shape:
      {
        "schema_version": "1.0.0",
        "slice_count": N,
        "ready_count": M,
        "degraded_count": K,
        "blocked_count": L,
        "slices": {
          "<slice_name>": {
            "status": "READY|DEGRADED|BLOCKED",
            "drift_severity": "NONE|COSMETIC|PARAMETRIC_MINOR|SEMANTIC",
            "missing_contract": bool,
            "missing_fields": [...],
            "unknown_fields": [...],
            "alignment_issues": [...]
          }
        }
      }
    """
    schema_version: str
    slice_count: int
    ready_count: int
    degraded_count: int
    blocked_count: int
    slices: Dict[str, Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with sorted keys for determinism."""
        # Sort slices alphabetically
        sorted_slices = {k: self.slices[k] for k in sorted(self.slices.keys())}
        
        return {
            "schema_version": self.schema_version,
            "slice_count": self.slice_count,
            "ready_count": self.ready_count,
            "degraded_count": self.degraded_count,
            "blocked_count": self.blocked_count,
            "slices": sorted_slices,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string with sorted keys."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


def summarize_readiness(results: Sequence[ReadinessResult]) -> ReadinessSummary:
    """
    Summarize readiness across multiple slices into a single aggregated view.
    
    This is the canonical multi-slice readiness summary for dashboards,
    governance, and downstream agents.
    
    Args:
        results: Sequence of ReadinessResult objects from check_slice_readiness().
        
    Returns:
        ReadinessSummary with aggregated counts and per-slice details.
        
    Guarantees:
        - Deterministic: same results → same summary
        - Slices sorted alphabetically in output
        - Lists (missing_fields, unknown_fields, alignment_issues) sorted
    """
    ready_count = 0
    degraded_count = 0
    blocked_count = 0
    slices: Dict[str, Dict[str, Any]] = {}
    
    for result in results:
        # Count by status
        if result.status == ReadinessStatus.READY:
            ready_count += 1
        elif result.status == ReadinessStatus.DEGRADED:
            degraded_count += 1
        elif result.status == ReadinessStatus.BLOCKED:
            blocked_count += 1
        
        # Build per-slice entry with sorted lists for determinism
        slices[result.slice_name] = {
            "status": result.status.value,
            "drift_severity": result.drift_class.value,
            "missing_contract": result.missing_contract,
            "missing_fields": sorted(result.missing_fields),
            "unknown_fields": sorted(result.unknown_fields),
            "alignment_issues": result.alignment_issues,  # Keep order as received
        }
    
    return ReadinessSummary(
        schema_version=READINESS_SUMMARY_SCHEMA_VERSION,
        slice_count=len(results),
        ready_count=ready_count,
        degraded_count=degraded_count,
        blocked_count=blocked_count,
        slices=slices,
    )


# =============================================================================
# CI-Grade One-Line Readiness Verdict (Task 2)
# =============================================================================

class ReadinessVerdict(str, Enum):
    """
    Overall readiness verdict for CI summary line.
    
    OK:    No BLOCKED slices
    WARN:  At least one DEGRADED, no BLOCKED
    BLOCK: At least one BLOCKED
    """
    OK = "OK"
    WARN = "WARN"
    BLOCK = "BLOCK"


def compute_readiness_verdict(summary: ReadinessSummary) -> ReadinessVerdict:
    """
    Compute the overall readiness verdict from a summary.
    
    Rules:
      - BLOCK if blocked_count > 0
      - WARN if degraded_count > 0 and blocked_count == 0
      - OK otherwise
    
    Args:
        summary: ReadinessSummary from summarize_readiness().
        
    Returns:
        ReadinessVerdict enum value.
    """
    if summary.blocked_count > 0:
        return ReadinessVerdict.BLOCK
    elif summary.degraded_count > 0:
        return ReadinessVerdict.WARN
    else:
        return ReadinessVerdict.OK


def format_readiness_summary_line(summary: ReadinessSummary) -> str:
    """
    Produce a single-line summary for CI logs.
    
    Format (exactly):
        Metric Readiness: ready=X degraded=Y blocked=Z total=N STATUS=OK|WARN|BLOCK
    
    This format is stable and human-greppable.
    
    Args:
        summary: ReadinessSummary from summarize_readiness().
        
    Returns:
        Single line string with no newlines.
    """
    verdict = compute_readiness_verdict(summary)
    
    return (
        f"Metric Readiness: "
        f"ready={summary.ready_count} "
        f"degraded={summary.degraded_count} "
        f"blocked={summary.blocked_count} "
        f"total={summary.slice_count} "
        f"STATUS={verdict.value}"
    )


def get_readiness_verdict_exit_code(verdict: ReadinessVerdict) -> int:
    """
    Get the CI exit code for a readiness verdict.
    
    Exit codes:
      - 0 for OK and WARN
      - 1 for BLOCK
    
    Args:
        verdict: ReadinessVerdict from compute_readiness_verdict().
        
    Returns:
        0 or 1.
    """
    if verdict == ReadinessVerdict.BLOCK:
        return 1
    return 0


# =============================================================================
# Readiness Consumer Handoff Surface (Task 3)
# =============================================================================
#
# STABILITY CONTRACT:
# -------------------
# The predicate is_slice_ready_for_experiments() is considered a contract.
# Changes to its semantics require documentation update and version bump.
#
# INTENDED CONSUMERS:
# -------------------
# - B1 (Runner Agent): Check before executing slice experiments
# - D5 (Drift Gate Agent): Verify slice stability before promotion
# - Governance: Automated checks before sign-off
#
# SEMANTICS:
# ----------
# Returns True iff:
#   - status == "READY"
#   - drift_severity in ACCEPTABLE_DRIFT_SEVERITIES_FOR_EXPERIMENTS
#     (currently: {"NONE", "COSMETIC"})
# Returns False otherwise.
#

def is_slice_ready_for_experiments(result: ReadinessResult) -> bool:
    """
    Predicate to check if a slice is ready for experiment execution.
    
    This is a stable contract for downstream consumers (B1, D5, governance).
    
    Rules:
      - Returns True iff:
          status == READY AND
          drift_severity in {"NONE", "COSMETIC"}
      - Returns False for DEGRADED or BLOCKED, or non-acceptable drift
    
    The acceptable drift severities are defined by
    ACCEPTABLE_DRIFT_SEVERITIES_FOR_EXPERIMENTS constant.
    
    Args:
        result: ReadinessResult from check_slice_readiness().
        
    Returns:
        True if slice is ready for experiments, False otherwise.
        
    Stability:
        This predicate is considered a contract. Changes require version update.
    """
    if result.status != ReadinessStatus.READY:
        return False
    
    if result.drift_class.value not in ACCEPTABLE_DRIFT_SEVERITIES_FOR_EXPERIMENTS:
        return False
    
    return True


def batch_check_readiness_for_experiments(
    results: Sequence[ReadinessResult]
) -> Dict[str, bool]:
    """
    Check readiness for experiments across multiple slices.
    
    Convenience function for checking multiple slices at once.
    
    Args:
        results: Sequence of ReadinessResult objects.
        
    Returns:
        Dict mapping slice_name → ready_for_experiments (bool).
    """
    return {
        result.slice_name: is_slice_ready_for_experiments(result)
        for result in results
    }


# =============================================================================
# Phase III: Per-Metric Readiness Matrix (Task 1)
# =============================================================================

# Schema version for readiness matrix
READINESS_MATRIX_SCHEMA_VERSION = "1.0.0"


def build_metric_readiness_matrix(
    results: Sequence[ReadinessResult]
) -> Dict[str, Any]:
    """
    Build a per-metric readiness matrix from slice readiness results.
    
    This provides a detailed view of readiness per slice per metric kind,
    suitable for dashboards and fine-grained promotion gates.
    
    Layout:
      {
        "schema_version": "1.0.0",
        "matrix": {
          "<slice_name>": {
            "<metric_kind>": {
              "status": "READY|DEGRADED|BLOCKED",
              "drift_severity": "NONE|COSMETIC|PARAMETRIC_MINOR|SEMANTIC",
              "ready_for_experiments": bool
            }
          }
        }
      }
    
    Args:
        results: Sequence of ReadinessResult objects.
        
    Returns:
        Dict with schema_version and matrix mapping slice → metric → status.
        
    Guarantees:
        - Deterministic: slices and metric kinds sorted alphabetically
        - Ready_for_experiments uses same logic as is_slice_ready_for_experiments()
    """
    matrix: Dict[str, Dict[str, Dict[str, Any]]] = {}
    
    for result in results:
        slice_name = result.slice_name
        
        # Extract metric_kind from log_field_coverage if available
        metric_kind = "unknown"
        if result.log_field_coverage and isinstance(result.log_field_coverage, dict):
            metric_kind = result.log_field_coverage.get("metric_kind", "unknown")
        
        # Compute ready_for_experiments using existing predicate logic
        ready_for_experiments = is_slice_ready_for_experiments(result)
        
        # Build entry for this metric
        metric_entry = {
            "status": result.status.value,
            "drift_severity": result.drift_class.value,
            "ready_for_experiments": ready_for_experiments,
        }
        
        # Initialize slice entry if needed
        if slice_name not in matrix:
            matrix[slice_name] = {}
        
        matrix[slice_name][metric_kind] = metric_entry
    
    # Sort slices alphabetically, and within each slice sort metric kinds
    sorted_matrix: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for slice_name in sorted(matrix.keys()):
        sorted_matrix[slice_name] = {
            mk: matrix[slice_name][mk]
            for mk in sorted(matrix[slice_name].keys())
        }
    
    return {
        "schema_version": READINESS_MATRIX_SCHEMA_VERSION,
        "matrix": sorted_matrix,
    }


# =============================================================================
# Phase III: Promotion Guard Helper (Task 2)
# =============================================================================

# Prefix for slices that matter for uplift promotion
UPLIFT_SLICE_PREFIX = "slice_uplift_"


def evaluate_metric_readiness_for_promotion(
    summary: ReadinessSummary,
    results: Optional[Sequence[ReadinessResult]] = None
) -> Dict[str, Any]:
    """
    Evaluate metric readiness for promotion decisions.
    
    This is a promotion guard helper that determines if the metric layer
    is ready for experiment promotion/governance sign-off.
    
    Rules for promotion_ok:
      - blocked_count == 0 (no blocked slices)
      - All uplift slices (slice_uplift_*) satisfy is_slice_ready_for_experiments()
    
    Args:
        summary: ReadinessSummary from summarize_readiness().
        results: Optional sequence of ReadinessResult for detailed uplift slice checks.
                 If not provided, only blocked_count is checked.
        
    Returns:
        Dict with:
          - promotion_ok: bool
          - blocking_slices: list of slice names blocking promotion
          - verdict: "OK" | "WARN" | "BLOCK"
          
    Guarantees:
        - Strictly derived from existing summary + predicate
        - No new semantics introduced
        - Deterministic output
    """
    blocking_slices: List[str] = []
    
    # Check 1: Any blocked slices?
    if summary.blocked_count > 0:
        # Find blocked slices from summary
        for slice_name, slice_data in summary.slices.items():
            if slice_data.get("status") == "BLOCKED":
                blocking_slices.append(slice_name)
    
    # Check 2: Uplift slices ready for experiments?
    if results is not None:
        for result in results:
            if result.slice_name.startswith(UPLIFT_SLICE_PREFIX):
                if not is_slice_ready_for_experiments(result):
                    if result.slice_name not in blocking_slices:
                        blocking_slices.append(result.slice_name)
    
    # Sort for determinism
    blocking_slices = sorted(blocking_slices)
    
    # Determine promotion_ok and verdict
    promotion_ok = len(blocking_slices) == 0
    
    # Verdict follows ReadinessVerdict logic
    if summary.blocked_count > 0:
        verdict = "BLOCK"
    elif summary.degraded_count > 0 or len(blocking_slices) > 0:
        verdict = "WARN"
    else:
        verdict = "OK"
    
    # If there are blocking slices, verdict is at least WARN
    if len(blocking_slices) > 0 and verdict == "OK":
        verdict = "WARN"
    
    return {
        "promotion_ok": promotion_ok,
        "blocking_slices": blocking_slices,
        "verdict": verdict,
    }


# =============================================================================
# Phase III: Global Health & MAAS Snapshot (Task 3)
# =============================================================================

def summarize_metric_readiness_for_global_health(
    summary: ReadinessSummary,
    promotion_eval: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Summarize metric readiness for global health and MAAS consumption.
    
    This is a compact adapter designed to be embedded directly in
    global_health.json and consumed by MAAS (Metrics-as-a-Service).
    
    Args:
        summary: ReadinessSummary from summarize_readiness().
        promotion_eval: Result from evaluate_metric_readiness_for_promotion().
        
    Returns:
        Dict with:
          - readiness_status: "OK" | "WARN" | "BLOCK"
          - ready_slice_ratio: float (ready_count / slice_count)
          - blocked_slice_count: int
          - promotion_ok: bool
          
    Guarantees:
        - Deterministic output
        - All fields derived from existing summary + promotion_eval
        - Safe division (handles 0 slices)
    """
    # Compute ready slice ratio (safe division)
    if summary.slice_count > 0:
        ready_slice_ratio = summary.ready_count / summary.slice_count
    else:
        ready_slice_ratio = 0.0
    
    # Derive readiness_status from verdict
    readiness_status = promotion_eval.get("verdict", "BLOCK")
    
    return {
        "readiness_status": readiness_status,
        "ready_slice_ratio": ready_slice_ratio,
        "blocked_slice_count": summary.blocked_count,
        "promotion_ok": promotion_eval.get("promotion_ok", False),
    }


# =============================================================================
# Phase IV: Cross-Metric Readiness Heatmap & Release Governance
# =============================================================================

# Schema version for readiness heatmap
READINESS_HEATMAP_SCHEMA_VERSION = "1.0.0"

# Drift status values (from D5 Drift Sentinel Grid)
class DriftStatus(str, Enum):
    """Drift status classification from D5."""
    OK = "OK"
    WARN = "WARN"
    DRIFTY = "DRIFTY"


# Budget flag values (from A5 Budget Joint View)
class BudgetFlag(str, Enum):
    """Budget flag classification from A5."""
    SAFE = "SAFE"
    TIGHT = "TIGHT"
    STARVED = "STARVED"


# Status light values for Director panel
class StatusLight(str, Enum):
    """Status light for Director-facing panel."""
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"


def build_readiness_heatmap(
    matrix: Dict[str, Any],
    drift_grid: Dict[str, Any],
    budget_joint_view: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Build a cross-metric readiness heatmap combining readiness, drift, and budget signals.
    
    This provides a unified view across readiness status, drift detection (D5),
    and budget constraints (A5) for each (slice, metric_kind) pair.
    
    Expected input formats:
      - matrix: From build_metric_readiness_matrix() with "matrix" key
      - drift_grid: From D5 with structure:
          {
            "grid": {
              "<slice_name>": {
                "<metric_kind>": {
                  "drift_status": "OK|WARN|DRIFTY"
                }
              }
            }
          }
      - budget_joint_view: From A5 with structure:
          {
            "view": {
              "<slice_name>": {
                "<metric_kind>": {
                  "budget_flag": "SAFE|TIGHT|STARVED"
                }
              }
            }
          }
    
    Args:
        matrix: Readiness matrix from build_metric_readiness_matrix().
        drift_grid: Drift grid from D5 Drift Sentinel Grid.
        budget_joint_view: Budget joint view from A5.
        
    Returns:
        Dict with:
          - heatmap_schema_version: "1.0.0"
          - heatmap: {
              "<slice_name>": {
                "<metric_kind>": {
                  "readiness_status": "READY|DEGRADED|BLOCKED",
                  "drift_status": "OK|WARN|DRIFTY",
                  "budget_flag": "SAFE|TIGHT|STARVED"
                }
              }
            }
          - slices_with_consistent_readiness: List of slice names where all metrics agree
          - metrics_with_conflicting_signals: List of "slice.metric" pairs with conflicts
            
    Guarantees:
        - Deterministic: sorted keys
        - JSON-safe
        - Handles missing drift/budget data gracefully (defaults to OK/SAFE)
    """
    heatmap: Dict[str, Dict[str, Dict[str, Any]]] = {}
    slices_with_consistent_readiness: List[str] = []
    metrics_with_conflicting_signals: List[str] = []
    
    # Extract grids with safe defaults
    readiness_matrix = matrix.get("matrix", {})
    drift_grid_data = drift_grid.get("grid", {})
    budget_view_data = budget_joint_view.get("view", {})
    
    # Track readiness statuses per slice for consistency check
    slice_readiness_map: Dict[str, Set[str]] = {}
    
    # Build heatmap for each (slice, metric) pair
    for slice_name in sorted(readiness_matrix.keys()):
        slice_readiness_map[slice_name] = set()
        
        for metric_kind in sorted(readiness_matrix[slice_name].keys()):
            readiness_entry = readiness_matrix[slice_name][metric_kind]
            readiness_status = readiness_entry.get("status", "UNKNOWN")
            slice_readiness_map[slice_name].add(readiness_status)
            
            # Get drift status (default to OK if missing)
            drift_status = "OK"
            if slice_name in drift_grid_data:
                if metric_kind in drift_grid_data[slice_name]:
                    drift_status = drift_grid_data[slice_name][metric_kind].get(
                        "drift_status", "OK"
                    )
            
            # Get budget flag (default to SAFE if missing)
            budget_flag = "SAFE"
            if slice_name in budget_view_data:
                if metric_kind in budget_view_data[slice_name]:
                    budget_flag = budget_view_data[slice_name][metric_kind].get(
                        "budget_flag", "SAFE"
                    )
            
            # Initialize slice entry if needed
            if slice_name not in heatmap:
                heatmap[slice_name] = {}
            
            heatmap[slice_name][metric_kind] = {
                "readiness_status": readiness_status,
                "drift_status": drift_status,
                "budget_flag": budget_flag,
            }
            
            # Check for conflicting signals
            # Conflict: READY but (DRIFTY or STARVED)
            if readiness_status == "READY":
                if drift_status == "DRIFTY" or budget_flag == "STARVED":
                    metrics_with_conflicting_signals.append(f"{slice_name}.{metric_kind}")
            # Conflict: DEGRADED but STARVED
            elif readiness_status == "DEGRADED" and budget_flag == "STARVED":
                metrics_with_conflicting_signals.append(f"{slice_name}.{metric_kind}")
    
    # Find slices with consistent readiness (all metrics have same status)
    for slice_name, statuses in slice_readiness_map.items():
        if len(statuses) == 1:
            slices_with_consistent_readiness.append(slice_name)
    
    # Sort for determinism
    slices_with_consistent_readiness = sorted(slices_with_consistent_readiness)
    metrics_with_conflicting_signals = sorted(metrics_with_conflicting_signals)
    
    return {
        "heatmap_schema_version": READINESS_HEATMAP_SCHEMA_VERSION,
        "heatmap": heatmap,
        "slices_with_consistent_readiness": slices_with_consistent_readiness,
        "metrics_with_conflicting_signals": metrics_with_conflicting_signals,
    }


def evaluate_release_promotion_with_readiness(
    readiness_heatmap: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluate release promotion using cross-signal consistency from readiness heatmap.
    
    This is an advisory-only helper that checks for cross-signal consistency
    issues (e.g., READY but DRIFTY+STARVED) that might indicate hidden risks.
    
    Args:
        readiness_heatmap: Result from build_readiness_heatmap().
        
    Returns:
        Dict with:
          - promotion_ok: bool (True if no blocking conflicts)
          - blocking_pairs: List of "slice.metric" identifiers with conflicts
          - verdict: "OK" | "WARN" | "BLOCK"
          - reasons: List of short, neutral reason strings
            
    Guarantees:
        - Deterministic output
        - Advisory-only (does not override existing promotion gates)
        - Structured for CI/MAAS consumption
    """
    heatmap = readiness_heatmap.get("heatmap", {})
    conflicting_signals = readiness_heatmap.get("metrics_with_conflicting_signals", [])
    
    blocking_pairs: List[str] = []
    reasons: List[str] = []
    
    # Check each (slice, metric) pair for blocking conflicts
    for slice_name in sorted(heatmap.keys()):
        for metric_kind in sorted(heatmap[slice_name].keys()):
            entry = heatmap[slice_name][metric_kind]
            readiness = entry.get("readiness_status", "UNKNOWN")
            drift = entry.get("drift_status", "OK")
            budget = entry.get("budget_flag", "SAFE")
            
            pair_id = f"{slice_name}.{metric_kind}"
            
            # Blocking condition: READY but both DRIFTY and STARVED
            if readiness == "READY" and drift == "DRIFTY" and budget == "STARVED":
                blocking_pairs.append(pair_id)
                reasons.append(f"{pair_id}: READY but DRIFTY+STARVED")
            
            # Blocking condition: BLOCKED readiness
            elif readiness == "BLOCKED":
                blocking_pairs.append(pair_id)
                reasons.append(f"{pair_id}: BLOCKED readiness")
            
            # Warning condition: READY but DRIFTY (without STARVED)
            elif readiness == "READY" and drift == "DRIFTY":
                if pair_id not in blocking_pairs:
                    reasons.append(f"{pair_id}: READY but DRIFTY")
            
            # Warning condition: READY but STARVED (without DRIFTY)
            elif readiness == "READY" and budget == "STARVED":
                if pair_id not in blocking_pairs:
                    reasons.append(f"{pair_id}: READY but STARVED")
    
    # Sort for determinism
    blocking_pairs = sorted(blocking_pairs)
    reasons = sorted(reasons)
    
    # Determine verdict
    if len(blocking_pairs) > 0:
        verdict = "BLOCK"
    elif len(conflicting_signals) > 0:
        verdict = "WARN"
    else:
        verdict = "OK"
    
    promotion_ok = len(blocking_pairs) == 0
    
    return {
        "promotion_ok": promotion_ok,
        "blocking_pairs": blocking_pairs,
        "verdict": verdict,
        "reasons": reasons,
    }


def build_metric_readiness_director_panel(
    readiness_summary: Dict[str, Any],
    heatmap_view: Dict[str, Any],
    promotion_eval: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Build a Director-facing metric readiness panel for release decisions.
    
    This is the metric-readiness tile in global_health.json, providing a
    single, high-level view for release governance decisions.
    
    Args:
        readiness_summary: From summarize_metric_readiness_for_global_health().
        heatmap_view: From build_readiness_heatmap().
        promotion_eval: From evaluate_release_promotion_with_readiness().
        
    Returns:
        Dict with:
          - status_light: "GREEN" | "YELLOW" | "RED"
          - readiness_status: "OK" | "WARN" | "BLOCK"
          - ready_slice_ratio: float
          - blocked_slice_count: int
          - promotion_ok: bool
          - headline: Short neutral sentence (no value judgments)
            
    Guarantees:
        - Deterministic output
        - JSON-safe
        - Neutral language (no "good/bad/better/worse")
    """
    readiness_status = readiness_summary.get("readiness_status", "BLOCK")
    ready_slice_ratio = readiness_summary.get("ready_slice_ratio", 0.0)
    blocked_slice_count = readiness_summary.get("blocked_slice_count", 0)
    promotion_ok = promotion_eval.get("promotion_ok", False)
    
    # Determine status light
    if readiness_status == "OK" and promotion_ok:
        status_light = StatusLight.GREEN.value
    elif readiness_status == "BLOCK" or not promotion_ok:
        status_light = StatusLight.RED.value
    else:
        status_light = StatusLight.YELLOW.value
    
    # Generate neutral headline
    if blocked_slice_count > 0:
        headline = f"{blocked_slice_count} slice(s) blocked"
    elif ready_slice_ratio >= 1.0:
        headline = "All slices ready"
    elif ready_slice_ratio >= 0.75:
        headline = f"{int(ready_slice_ratio * 100)}% slices ready"
    else:
        headline = f"{int(ready_slice_ratio * 100)}% slices ready, {blocked_slice_count} blocked"
    
    # Add cross-signal context if available
    conflicting_count = len(heatmap_view.get("metrics_with_conflicting_signals", []))
    if conflicting_count > 0:
        headline += f", {conflicting_count} cross-signal conflict(s)"
    
    return {
        "status_light": status_light,
        "readiness_status": readiness_status,
        "ready_slice_ratio": ready_slice_ratio,
        "blocked_slice_count": blocked_slice_count,
        "promotion_ok": promotion_ok,
        "headline": headline,
    }


# =============================================================================
# Phase V: Readiness Autopilot & Phase-Boundary Playbook
# =============================================================================

# Autopilot status values
class AutopilotStatus(str, Enum):
    """Autopilot status for readiness policy."""
    OK = "OK"
    ATTENTION = "ATTENTION"
    BLOCK = "BLOCK"


def build_readiness_autopilot_policy(
    readiness_heatmap: Dict[str, Any],
    history: Dict[str, Any],
    *,
    target_ready_ratio: float = 0.8
) -> Dict[str, Any]:
    """
    Build a readiness autopilot policy based on heatmap, history, and target ratio.
    
    This is a higher-level policy helper that determines autopilot status and
    identifies slices to hold vs. safe to progress. It's advisory-only with no
    side effects, designed as a policy surface for orchestrators.
    
    Args:
        readiness_heatmap: Result from build_readiness_heatmap().
        history: Optional history dict with structure:
            {
                "blocked_slices": ["slice.metric", ...],  # Previously blocked
                "repeated_conflicts": ["slice.metric", ...],  # Repeated issues
            }
        target_ready_ratio: Target ratio for ready slices (default 0.8).
        
    Returns:
        Dict with:
          - autopilot_status: "OK" | "ATTENTION" | "BLOCK"
          - slices_to_hold: List of "slice.metric" identifiers to hold
          - slices_safe_to_progress: List of "slice.metric" identifiers safe to progress
          - ready_ratio: float (computed from heatmap)
          - neutral_notes: List of neutral note strings
            
    Guarantees:
        - Deterministic output
        - Advisory-only (no side effects)
        - Neutral language
    """
    heatmap = readiness_heatmap.get("heatmap", {})
    conflicting_signals = readiness_heatmap.get("metrics_with_conflicting_signals", [])
    
    # Extract history with safe defaults
    history_blocked = history.get("blocked_slices", [])
    history_repeated = history.get("repeated_conflicts", [])
    
    slices_to_hold: List[str] = []
    slices_safe_to_progress: List[str] = []
    neutral_notes: List[str] = []
    
    # Compute ready ratio from heatmap
    total_pairs = 0
    ready_pairs = 0
    
    for slice_name in sorted(heatmap.keys()):
        for metric_kind in sorted(heatmap[slice_name].keys()):
            total_pairs += 1
            entry = heatmap[slice_name][metric_kind]
            readiness = entry.get("readiness_status", "UNKNOWN")
            drift = entry.get("drift_status", "OK")
            budget = entry.get("budget_flag", "SAFE")
            
            pair_id = f"{slice_name}.{metric_kind}"
            
            # Count ready pairs
            if readiness == "READY":
                ready_pairs += 1
            
            # Determine if pair should be held or safe to progress
            should_hold = False
            
            # Hold if BLOCKED
            if readiness == "BLOCKED":
                should_hold = True
                neutral_notes.append(f"{pair_id}: BLOCKED readiness")
            
            # Hold if READY but DRIFTY+STARVED
            elif readiness == "READY" and drift == "DRIFTY" and budget == "STARVED":
                should_hold = True
                neutral_notes.append(f"{pair_id}: READY but DRIFTY+STARVED")
            
            # Hold if in history of blocked slices
            elif pair_id in history_blocked:
                should_hold = True
                neutral_notes.append(f"{pair_id}: Previously blocked")
            
            # Hold if repeated conflicts
            elif pair_id in history_repeated:
                should_hold = True
                neutral_notes.append(f"{pair_id}: Repeated conflicts")
            
            # Hold if READY but DRIFTY (even without STARVED)
            elif readiness == "READY" and drift == "DRIFTY":
                should_hold = True
                neutral_notes.append(f"{pair_id}: READY but DRIFTY")
            
            # Hold if READY but STARVED (even without DRIFTY)
            elif readiness == "READY" and budget == "STARVED":
                should_hold = True
                neutral_notes.append(f"{pair_id}: READY but STARVED")
            
            # Safe to progress if READY, OK drift, SAFE budget, and not in history
            elif (readiness == "READY" and 
                  drift == "OK" and 
                  budget == "SAFE" and
                  pair_id not in history_blocked and
                  pair_id not in history_repeated):
                slices_safe_to_progress.append(pair_id)
            
            if should_hold:
                slices_to_hold.append(pair_id)
    
    # Compute ready ratio
    if total_pairs > 0:
        ready_ratio = ready_pairs / total_pairs
    else:
        ready_ratio = 0.0
    
    # Determine autopilot status
    if ready_ratio >= target_ready_ratio and len(slices_to_hold) == 0:
        autopilot_status = AutopilotStatus.OK.value
    elif ready_ratio < target_ready_ratio * 0.5 or len(conflicting_signals) > total_pairs * 0.3:
        autopilot_status = AutopilotStatus.BLOCK.value
    else:
        autopilot_status = AutopilotStatus.ATTENTION.value
    
    # Sort for determinism
    slices_to_hold = sorted(slices_to_hold)
    slices_safe_to_progress = sorted(slices_safe_to_progress)
    neutral_notes = sorted(neutral_notes)
    
    return {
        "autopilot_status": autopilot_status,
        "slices_to_hold": slices_to_hold,
        "slices_safe_to_progress": slices_safe_to_progress,
        "ready_ratio": ready_ratio,
        "neutral_notes": neutral_notes,
    }


def derive_phase_boundary_recommendations(
    readiness_summary: Dict[str, Any],
    autopilot_policy: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Derive phase boundary recommendations based on readiness and autopilot policy.
    
    This is a small planner for phase transitions that uses readiness summary
    and autopilot policy to determine if a phase is ready to advance and what
    actions are recommended.
    
    Args:
        readiness_summary: From summarize_metric_readiness_for_global_health().
        autopilot_policy: From build_readiness_autopilot_policy().
        
    Returns:
        Dict with:
          - phase_ready: bool
          - status: "OK" | "WARN" | "BLOCK"
          - recommended_actions: List of neutral action strings
          - slices_needing_investigation: List of slice names
          
    Guarantees:
        - Deterministic output
        - Neutral language
        - Advisory-only
    """
    readiness_status = readiness_summary.get("readiness_status", "BLOCK")
    ready_slice_ratio = readiness_summary.get("ready_slice_ratio", 0.0)
    blocked_slice_count = readiness_summary.get("blocked_slice_count", 0)
    promotion_ok = readiness_summary.get("promotion_ok", False)
    
    autopilot_status = autopilot_policy.get("autopilot_status", "BLOCK")
    slices_to_hold = autopilot_policy.get("slices_to_hold", [])
    ready_ratio = autopilot_policy.get("ready_ratio", 0.0)
    
    recommended_actions: List[str] = []
    slices_needing_investigation: Set[str] = set()
    
    # Extract slice names from slices_to_hold
    for pair_id in slices_to_hold:
        if "." in pair_id:
            slice_name = pair_id.split(".", 1)[0]
            slices_needing_investigation.add(slice_name)
    
    slices_needing_investigation = sorted(list(slices_needing_investigation))
    
    # Determine phase_ready and status
    phase_ready = (
        readiness_status == "OK" and
        promotion_ok and
        autopilot_status == "OK" and
        blocked_slice_count == 0 and
        len(slices_to_hold) == 0
    )
    
    if not phase_ready:
        if blocked_slice_count > 0 or autopilot_status == "BLOCK":
            status = "BLOCK"
        else:
            status = "WARN"
    else:
        status = "OK"
    
    # Generate recommended actions
    if blocked_slice_count > 0:
        recommended_actions.append(
            f"Resolve {blocked_slice_count} blocked slice(s) before phase transition"
        )
    
    for pair_id in slices_to_hold[:5]:  # Limit to first 5 for brevity
        if "." in pair_id:
            slice_name, metric_kind = pair_id.split(".", 1)
            recommended_actions.append(
                f"Hold promotion for {slice_name} until drift clears"
            )
    
    if ready_slice_ratio >= 0.9 and len(slices_to_hold) == 0:
        # Suggest advancing high-performing slices
        recommended_actions.append(
            "Consider advancing high-readiness slices to next depth band"
        )
    
    if len(slices_needing_investigation) > 0:
        recommended_actions.append(
            f"Investigate {len(slices_needing_investigation)} slice(s) with repeated issues"
        )
    
    # Sort for determinism
    recommended_actions = sorted(recommended_actions)
    
    return {
        "phase_ready": phase_ready,
        "status": status,
        "recommended_actions": recommended_actions,
        "slices_needing_investigation": slices_needing_investigation,
    }


# =============================================================================
# Phase VI: Readiness Tensor Engine v1
# =============================================================================

# Schema version for readiness tensor
READINESS_TENSOR_SCHEMA_VERSION = "1.0.0"

# Transition band values
class TransitionBand(str, Enum):
    """Phase transition safety band."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


def build_metric_readiness_tensor(
    readiness_matrix: Dict[str, Any],
    drift_grid: Dict[str, Any],
    budget_joint_view: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Build a metric readiness tensor with multi-axis geometry.
    
    This provides a unified tensor view enabling MAAS/TDA/Cortex to evaluate
    slices through a single tensor rather than per-metric fragments.
    
    Each slice is represented as a vector in readiness-drift-budget space,
    with normalized components and computed vector norms.
    
    Args:
        readiness_matrix: From build_metric_readiness_matrix().
        drift_grid: From D5 Drift Sentinel Grid.
        budget_joint_view: From A5 Budget Joint View.
        
    Returns:
        Dict with:
          - slice_vectors: Per-slice vector components and norms
          - global_norm: Overall system norm
          - ranked_slices: Slices ranked by vector norm (descending)
          - schema_version: "1.0.0"
          
    Guarantees:
        - All components normalized to [0, 1]
        - Deterministic ordering
        - JSON-safe
    """
    import math
    
    matrix = readiness_matrix.get("matrix", {})
    drift_data = drift_grid.get("grid", {})
    budget_data = budget_joint_view.get("view", {})
    
    slice_vectors: Dict[str, Dict[str, float]] = {}
    
    # Normalize readiness status to [0, 1]
    def normalize_readiness(status: str) -> float:
        if status == "READY":
            return 1.0
        elif status == "DEGRADED":
            return 0.5
        elif status == "BLOCKED":
            return 0.0
        return 0.0
    
    # Normalize drift status to [0, 1] (OK=1, WARN=0.5, DRIFTY=0)
    def normalize_drift(drift: str) -> float:
        if drift == "OK":
            return 1.0
        elif drift == "WARN":
            return 0.5
        elif drift == "DRIFTY":
            return 0.0
        return 1.0  # Default to OK if missing
    
    # Normalize budget flag to [0, 1] (SAFE=1, TIGHT=0.5, STARVED=0)
    def normalize_budget(budget: str) -> float:
        if budget == "SAFE":
            return 1.0
        elif budget == "TIGHT":
            return 0.5
        elif budget == "STARVED":
            return 0.0
        return 1.0  # Default to SAFE if missing
    
    # Process each slice
    for slice_name in sorted(matrix.keys()):
        readiness_scores = []
        drift_scores = []
        budget_scores = []
        consistency_scores = []
        
        # Aggregate across all metrics for this slice
        for metric_kind in sorted(matrix[slice_name].keys()):
            entry = matrix[slice_name][metric_kind]
            readiness_status = entry.get("status", "UNKNOWN")
            
            # Get drift and budget for this metric
            drift_status = "OK"
            if slice_name in drift_data and metric_kind in drift_data[slice_name]:
                drift_status = drift_data[slice_name][metric_kind].get("drift_status", "OK")
            
            budget_flag = "SAFE"
            if slice_name in budget_data and metric_kind in budget_data[slice_name]:
                budget_flag = budget_data[slice_name][metric_kind].get("budget_flag", "SAFE")
            
            # Normalize components
            r_score = normalize_readiness(readiness_status)
            d_score = normalize_drift(drift_status)
            b_score = normalize_budget(budget_flag)
            
            readiness_scores.append(r_score)
            drift_scores.append(d_score)
            budget_scores.append(b_score)
            
            # Consistency: how similar are readiness scores across metrics?
            # For now, use variance (lower = more consistent)
            consistency_scores.append(r_score)
        
        # Aggregate slice-level components (weighted average)
        if len(readiness_scores) > 0:
            readiness_component = sum(readiness_scores) / len(readiness_scores)
            drift_component = sum(drift_scores) / len(drift_scores)
            budget_component = sum(budget_scores) / len(budget_scores)
            
            # Consistency component: 1 - variance (higher = more consistent)
            if len(consistency_scores) > 1:
                mean_consistency = sum(consistency_scores) / len(consistency_scores)
                variance = sum((x - mean_consistency) ** 2 for x in consistency_scores) / len(consistency_scores)
                consistency_component = 1.0 - min(variance, 1.0)  # Cap at 1.0
            else:
                consistency_component = 1.0  # Single metric = perfectly consistent
            
            # Compute weighted score (0.5 readiness, 0.25 drift, 0.25 budget)
            readiness_score = (
                0.5 * readiness_component +
                0.25 * drift_component +
                0.25 * budget_component
            )
            
            # Compute vector norm (L2 norm in 4D space)
            vector_norm = math.sqrt(
                readiness_component ** 2 +
                drift_component ** 2 +
                budget_component ** 2 +
                consistency_component ** 2
            )
            
            slice_vectors[slice_name] = {
                "readiness_score": readiness_score,
                "drift_component": drift_component,
                "budget_component": budget_component,
                "metric_consistency_component": consistency_component,
                "vector_norm": vector_norm,
            }
    
    # Compute global norm (average of slice norms)
    if len(slice_vectors) > 0:
        global_norm = sum(v["vector_norm"] for v in slice_vectors.values()) / len(slice_vectors)
    else:
        global_norm = 0.0
    
    # Rank slices by vector norm (descending)
    ranked_slices = sorted(
        slice_vectors.keys(),
        key=lambda s: slice_vectors[s]["vector_norm"],
        reverse=True
    )
    
    return {
        "slice_vectors": slice_vectors,
        "global_norm": global_norm,
        "ranked_slices": ranked_slices,
        "schema_version": READINESS_TENSOR_SCHEMA_VERSION,
    }


def build_metric_drift_polygraph(
    tensor: Dict[str, Any],
    readiness_summary_history: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Build a cross-system drift polygraph detecting drift momentum and entanglement.
    
    This detects:
      - Drift momentum across time windows
      - Metric-slice entanglement (drift propagating diagonally)
      - Poly-fail conditions (drift + budget + readiness all degrading)
    
    Args:
        tensor: From build_metric_readiness_tensor().
        readiness_summary_history: History dict with structure:
            {
                "windows": [
                    {
                        "timestamp": "...",
                        "readiness_summary": {...},
                        "tensor": {...}
                    },
                    ...
                ]
            }
        
    Returns:
        Dict with:
          - drift_momentum: float (rate of change in drift components)
          - entangled_pairs: List of [slice, metric] pairs with correlated drift
          - poly_fail_detected: bool
          - neutral_notes: List of neutral note strings
          
    Guarantees:
        - Deterministic output
        - Neutral language
    """
    slice_vectors = tensor.get("slice_vectors", {})
    history_windows = readiness_summary_history.get("windows", [])
    
    drift_momentum = 0.0
    entangled_pairs: List[List[str]] = []
    poly_fail_detected = False
    neutral_notes: List[str] = []
    
    # Compute drift momentum from history
    if len(history_windows) >= 2:
        # Compare most recent two windows
        recent = history_windows[-1]
        previous = history_windows[-2]
        
        recent_tensor = recent.get("tensor", {})
        previous_tensor = previous.get("tensor", {})
        
        recent_vectors = recent_tensor.get("slice_vectors", {})
        previous_vectors = previous_tensor.get("slice_vectors", {})
        
        # Compute average change in drift components
        drift_changes = []
        for slice_name in set(list(recent_vectors.keys()) + list(previous_vectors.keys())):
            if slice_name in recent_vectors and slice_name in previous_vectors:
                recent_drift = recent_vectors[slice_name].get("drift_component", 1.0)
                previous_drift = previous_vectors[slice_name].get("drift_component", 1.0)
                drift_changes.append(recent_drift - previous_drift)  # Negative = degrading
        
        if len(drift_changes) > 0:
            drift_momentum = sum(drift_changes) / len(drift_changes)
    
    # Detect poly-fail: all components degrading simultaneously
    degrading_slices = []
    for slice_name, vector in slice_vectors.items():
        readiness_score = vector.get("readiness_score", 1.0)
        drift_component = vector.get("drift_component", 1.0)
        budget_component = vector.get("budget_component", 1.0)
        
        # Poly-fail: all three below threshold
        if readiness_score < 0.4 and drift_component < 0.4 and budget_component < 0.4:
            degrading_slices.append(slice_name)
            poly_fail_detected = True
            neutral_notes.append(f"{slice_name}: All components below threshold")
    
    # Detect entanglement: slices with similar drift patterns
    # For now, identify slices with very similar drift components
    slice_names = sorted(slice_vectors.keys())
    for i, slice_a in enumerate(slice_names):
        for slice_b in slice_names[i+1:]:
            drift_a = slice_vectors[slice_a].get("drift_component", 1.0)
            drift_b = slice_vectors[slice_b].get("drift_component", 1.0)
            
            # Entangled if drift components are very similar (within 0.1)
            if abs(drift_a - drift_b) < 0.1 and drift_a < 0.5:
                entangled_pairs.append([slice_a, slice_b])
                neutral_notes.append(f"{slice_a} and {slice_b}: Similar drift patterns")
    
    # Sort for determinism
    entangled_pairs = sorted(entangled_pairs)
    neutral_notes = sorted(neutral_notes)
    
    return {
        "drift_momentum": drift_momentum,
        "entangled_pairs": entangled_pairs,
        "poly_fail_detected": poly_fail_detected,
        "neutral_notes": neutral_notes,
    }


def build_metric_readiness_autopilot_director_panel(
    tensor: Dict[str, Any],
    polygraph: Dict[str, Any],
    autopilot_policy: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Build Readiness Autopilot Director Tile v2 with tensor and polygraph integration.
    
    This is the enhanced director panel that incorporates tensor geometry
    and drift polygraphy for more sophisticated readiness assessment.
    
    Args:
        tensor: From build_metric_readiness_tensor().
        polygraph: From build_metric_drift_polygraph().
        autopilot_policy: From build_readiness_autopilot_policy().
        
    Returns:
        Dict with:
          - status_light: "GREEN" | "YELLOW" | "RED"
          - autopilot_status: From autopilot_policy
          - global_norm: From tensor
          - poly_fail_detected: From polygraph
          - slices_to_hold: From autopilot_policy
          - slices_safe_to_progress: From autopilot_policy
          - headline: Neutral sentence
          
    Guarantees:
        - Deterministic output
        - Neutral language
    """
    global_norm = tensor.get("global_norm", 0.0)
    poly_fail_detected = polygraph.get("poly_fail_detected", False)
    autopilot_status = autopilot_policy.get("autopilot_status", "BLOCK")
    slices_to_hold = autopilot_policy.get("slices_to_hold", [])
    slices_safe_to_progress = autopilot_policy.get("slices_safe_to_progress", [])
    
    # Determine status light
    # RED if poly-fail or global_norm < 0.35
    if poly_fail_detected or global_norm < 0.35:
        status_light = StatusLight.RED.value
    # YELLOW if autopilot is ATTENTION or drift momentum > threshold
    elif autopilot_status == "ATTENTION" or polygraph.get("drift_momentum", 0.0) < -0.1:
        status_light = StatusLight.YELLOW.value
    # GREEN if all stable
    else:
        status_light = StatusLight.GREEN.value
    
    # Generate neutral headline
    if poly_fail_detected:
        headline = "Poly-fail condition detected"
    elif global_norm < 0.35:
        headline = f"Global norm below threshold ({global_norm:.2f})"
    elif len(slices_to_hold) > 0:
        headline = f"{len(slices_to_hold)} slice(s) held, {len(slices_safe_to_progress)} safe to progress"
    else:
        headline = f"Global norm: {global_norm:.2f}, {len(slices_safe_to_progress)} slice(s) safe to progress"
    
    return {
        "status_light": status_light,
        "autopilot_status": autopilot_status,
        "global_norm": global_norm,
        "poly_fail_detected": poly_fail_detected,
        "slices_to_hold": sorted(slices_to_hold),
        "slices_safe_to_progress": sorted(slices_safe_to_progress),
        "headline": headline,
    }


def evaluate_phase_transition_safety_v2(
    tensor: Dict[str, Any],
    polygraph: Dict[str, Any],
    promotions_ready: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluate phase transition safety using tensor and polygraph analysis.
    
    This is the enhanced phase transition sentinel that incorporates
    tensor geometry and drift polygraphy for more sophisticated safety checks.
    
    Args:
        tensor: From build_metric_readiness_tensor().
        polygraph: From build_metric_drift_polygraph().
        promotions_ready: From evaluate_metric_readiness_for_promotion().
        
    Returns:
        Dict with:
          - transition_safe: bool
          - blocking_conditions: List of blocking condition strings
          - transition_band: "LOW" | "MEDIUM" | "HIGH"
          - recommendations: List of neutral recommendation strings
          
    Guarantees:
        - Deterministic output
        - Neutral language
    """
    global_norm = tensor.get("global_norm", 0.0)
    poly_fail_detected = polygraph.get("poly_fail_detected", False)
    drift_momentum = polygraph.get("drift_momentum", 0.0)
    promotion_ok = promotions_ready.get("promotion_ok", False)
    
    blocking_conditions: List[str] = []
    recommendations: List[str] = []
    
    # Determine transition safety
    transition_safe = True
    
    # Blocking conditions
    if poly_fail_detected:
        transition_safe = False
        blocking_conditions.append("Poly-fail condition detected")
    
    if global_norm < 0.35:
        transition_safe = False
        blocking_conditions.append(f"Global norm below threshold ({global_norm:.2f})")
    
    if not promotion_ok:
        transition_safe = False
        blocking_conditions.append("Promotion readiness check failed")
    
    if drift_momentum < -0.2:  # Strong negative momentum
        transition_safe = False
        blocking_conditions.append(f"Strong drift momentum detected ({drift_momentum:.2f})")
    
    # Determine transition band
    if global_norm >= 0.7 and not poly_fail_detected and promotion_ok:
        transition_band = TransitionBand.HIGH.value
    elif global_norm >= 0.5 and not poly_fail_detected:
        transition_band = TransitionBand.MEDIUM.value
    else:
        transition_band = TransitionBand.LOW.value
    
    # Generate recommendations
    if not transition_safe:
        if poly_fail_detected:
            recommendations.append("Resolve poly-fail conditions before transition")
        if global_norm < 0.35:
            recommendations.append(f"Improve global norm (currently {global_norm:.2f})")
        if drift_momentum < -0.1:
            recommendations.append("Address drift momentum before transition")
    else:
        if transition_band == "HIGH":
            recommendations.append("Transition band: HIGH - safe to proceed")
        elif transition_band == "MEDIUM":
            recommendations.append("Transition band: MEDIUM - proceed with caution")
        else:
            recommendations.append("Transition band: LOW - review before proceeding")
    
    # Sort for determinism
    blocking_conditions = sorted(blocking_conditions)
    recommendations = sorted(recommendations)
    
    return {
        "transition_safe": transition_safe,
        "blocking_conditions": blocking_conditions,
        "transition_band": transition_band,
        "recommendations": recommendations,
    }


# =============================================================================
# Metric Index Contract Browser
# =============================================================================

@dataclass
class ContractIndexListEntry:
    """Entry for contract listing display."""
    slice_name: str
    metric_kind: str
    contract_path: str
    schema_version: str
    config_hash: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "slice_name": self.slice_name,
            "metric_kind": self.metric_kind,
            "contract_path": self.contract_path,
            "schema_version": self.schema_version,
            "config_hash": self.config_hash,
        }


def list_contracts(
    contract_dir: Path = DEFAULT_CONTRACT_OUTPUT_DIR
) -> List[ContractIndexListEntry]:
    """
    Lists all contracts from the existing bundle under artifacts/phase_ii/metric_contracts/.
    
    Reads from the contract set as it exists — does NOT recompute anything.
    
    Args:
        contract_dir: Path to the contract bundle directory.
        
    Returns:
        Sorted list of ContractIndexListEntry objects.
        
    Raises:
        FileNotFoundError: If the contract bundle or index does not exist.
    """
    index_path = contract_dir / "metric_contract_index.json"
    
    if not index_path.exists():
        raise FileNotFoundError(
            f"Contract index not found at {index_path}. "
            f"Run --export-all-contracts first to generate the bundle."
        )
    
    with open(index_path, "r", encoding="utf-8") as f:
        index_data = json.load(f)
    
    entries: List[ContractIndexListEntry] = []
    
    for contract_entry in index_data.get("contracts", []):
        entries.append(ContractIndexListEntry(
            slice_name=contract_entry.get("slice_name", ""),
            metric_kind=contract_entry.get("metric_kind", ""),
            contract_path=contract_entry.get("contract_path", ""),
            schema_version=contract_entry.get("schema_version", ""),
            config_hash=contract_entry.get("config_hash", ""),
        ))
    
    # Sort by slice_name for deterministic output
    entries.sort(key=lambda e: e.slice_name)
    
    return entries


def format_contract_list(entries: List[ContractIndexListEntry]) -> str:
    """
    Formats a list of ContractIndexListEntry objects as a human-readable table.
    """
    if not entries:
        return "No contracts found in the bundle."
    
    lines = [
        "=" * 90,
        "METRIC CONTRACT INDEX",
        "=" * 90,
        "",
        f"{'SLICE NAME':<30} {'METRIC KIND':<20} {'VERSION':<10} {'CONFIG HASH':<20}",
        "-" * 90,
    ]
    
    for entry in entries:
        hash_short = entry.config_hash[:16] + "..." if len(entry.config_hash) > 16 else entry.config_hash
        lines.append(
            f"{entry.slice_name:<30} {entry.metric_kind:<20} {entry.schema_version:<10} {hash_short:<20}"
        )
    
    lines.extend([
        "-" * 90,
        f"Total contracts: {len(entries)}",
        "=" * 90,
    ])
    
    return "\n".join(lines)


# =============================================================================
# Log-Field Coverage Map
# =============================================================================

# Optional log fields by metric kind - fields that MAY be present for extended analysis
# These are not required for metric computation but can be used for diagnostics/reporting
OPTIONAL_LOG_FIELDS_BY_KIND: Dict[str, List[str]] = {
    "goal_hit": ["timestamp", "attempt_id", "proof_trace", "cycle_index"],
    "sparse_success": ["timestamp", "attempt_id", "candidates_evaluated", "cycle_index"],
    "chain_success": ["timestamp", "attempt_id", "full_dependency_graph", "cycle_index"],
    "multi_goal_success": ["timestamp", "attempt_id", "per_goal_status", "cycle_index"],
}


def get_log_field_coverage(
    slice_name: str,
    curriculum_path: Path = DEFAULT_CURRICULUM_PATH
) -> LogFieldCoverageMap:
    """
    Returns the log field coverage map for a slice's metric kind.
    
    Documents which JSONL fields are actually read by the adapter, derived
    from the contract/adapter configuration (NOT from AST inspection).
    
    This is a mini contract for log-field expectations.
    
    Args:
        slice_name: Name of the slice.
        curriculum_path: Path to the curriculum YAML file.
        
    Returns:
        LogFieldCoverageMap with field documentation.
        
    Raises:
        ValueError: If slice not found or invalid configuration.
    """
    curriculum = load_curriculum(curriculum_path)
    slice_config = get_slice_config(curriculum, slice_name)
    
    if slice_config is None:
        raise ValueError(f"Slice '{slice_name}' not found in curriculum.")
    
    metric_config = slice_config.get("success_metric", {})
    metric_kind_str = metric_config.get("kind", "")
    parameters = metric_config.get("parameters", {})
    
    try:
        metric_kind = MetricKind(metric_kind_str)
        harm_entry = METRIC_HARMONIZATION_TABLE.get(metric_kind, {})
    except ValueError:
        raise ValueError(f"Unknown metric kind '{metric_kind_str}'.")
    
    return LogFieldCoverageMap(
        metric_kind=metric_kind_str,
        required_log_fields=list(harm_entry.get("required_log_fields", [])),
        optional_log_fields=OPTIONAL_LOG_FIELDS_BY_KIND.get(metric_kind_str, []),
        runtime_fields=list(harm_entry.get("runtime_fields", [])),
        parameter_fields=list(parameters.keys()),
        interpretation=harm_entry.get("result_interpretation", "unknown"),
    )


def get_log_field_coverage_by_kind(metric_kind_str: str) -> LogFieldCoverageMap:
    """
    Returns the log field coverage map for a metric kind directly.
    
    Used for per-metric-kind documentation independent of slice.
    This is a helper for examining coverage by metric type rather than slice.
    
    Args:
        metric_kind_str: String value of the MetricKind.
        
    Returns:
        LogFieldCoverageMap with field documentation.
        
    Raises:
        ValueError: If unknown metric kind.
    """
    try:
        metric_kind = MetricKind(metric_kind_str)
        harm_entry = METRIC_HARMONIZATION_TABLE.get(metric_kind, {})
    except ValueError:
        raise ValueError(f"Unknown metric kind '{metric_kind_str}'.")
    
    # Get parameter fields from validator schema
    validator = MetricSchemaValidator()
    prereg_schema = validator.PREREG_SCHEMA.get(metric_kind, {})
    required_params = prereg_schema.get("required_parameters", [])
    optional_params = prereg_schema.get("optional_parameters", [])
    
    return LogFieldCoverageMap(
        metric_kind=metric_kind_str,
        required_log_fields=list(harm_entry.get("required_log_fields", [])),
        optional_log_fields=OPTIONAL_LOG_FIELDS_BY_KIND.get(metric_kind_str, []),
        runtime_fields=list(harm_entry.get("runtime_fields", [])),
        parameter_fields=required_params + optional_params,
        interpretation=harm_entry.get("result_interpretation", "unknown"),
    )


def format_log_field_coverage(coverage: LogFieldCoverageMap) -> str:
    """
    Formats a LogFieldCoverageMap as a human-readable string.
    
    Ensures human output matches JSON contract structure.
    """
    lines = [
        "=" * 72,
        f"LOG FIELD COVERAGE: {coverage.metric_kind}",
        "=" * 72,
        "",
        f"Interpretation: {coverage.interpretation}",
        "",
        "REQUIRED LOG FIELDS (must be present in JSONL)",
        "-" * 36,
    ]
    
    for field_name in sorted(coverage.required_log_fields):
        lines.append(f"  • {field_name}")
    
    if not coverage.required_log_fields:
        lines.append("  (none)")
    
    lines.extend([
        "",
        "OPTIONAL LOG FIELDS (may be present for diagnostics)",
        "-" * 36,
    ])
    
    for field_name in sorted(coverage.optional_log_fields):
        lines.append(f"  • {field_name}")
    
    if not coverage.optional_log_fields:
        lines.append("  (none)")
    
    lines.extend([
        "",
        "RUNTIME FIELDS (provided at execution)",
        "-" * 36,
    ])
    
    for field_name in sorted(coverage.runtime_fields):
        lines.append(f"  • {field_name}")
    
    if not coverage.runtime_fields:
        lines.append("  (none)")
    
    lines.extend([
        "",
        "PARAMETER FIELDS (from config)",
        "-" * 36,
    ])
    
    for field_name in sorted(coverage.parameter_fields):
        lines.append(f"  • {field_name}")
    
    if not coverage.parameter_fields:
        lines.append("  (none)")
    
    lines.extend(["", "=" * 72])
    
    return "\n".join(lines)


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    """
    CLI entry point for metric adapter introspection.
    """
    parser = argparse.ArgumentParser(
        description="PHASE II — Metric Adapter Introspection Layer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Get summary for a slice
  uv run python experiments/metric_adapter_introspection.py --slice slice_uplift_goal --summary
  
  # Verify alignment for a slice
  uv run python experiments/metric_adapter_introspection.py --slice slice_uplift_goal --verify
  
  # Export contract for a slice
  uv run python experiments/metric_adapter_introspection.py --slice slice_uplift_goal --export
  
  # Detect schema drift for a slice
  uv run python experiments/metric_adapter_introspection.py --slice slice_uplift_goal --drift
  
  # List all slices
  uv run python experiments/metric_adapter_introspection.py --list
  
  # Verify all slices
  uv run python experiments/metric_adapter_introspection.py --verify-all
  
  # CI Health Check (exit 0 if OK/WARN, exit 1 if FAIL)
  uv run python experiments/metric_adapter_introspection.py --health-check
  uv run python experiments/metric_adapter_introspection.py --health-check --json
  
  # Export contract bundle with index
  uv run python experiments/metric_adapter_introspection.py --export-all-contracts
  
  # Developer dashboard
  uv run python experiments/metric_adapter_introspection.py --dashboard
  
  # Per-slice readiness check (exit 0 if READY/DEGRADED, 1 if BLOCKED, 2 on error)
  uv run python experiments/metric_adapter_introspection.py --ready slice_uplift_goal
  uv run python experiments/metric_adapter_introspection.py --ready slice_uplift_goal --json
  
  # List existing contracts from bundle (read-only)
  uv run python experiments/metric_adapter_introspection.py --list-contracts
  uv run python experiments/metric_adapter_introspection.py --list-contracts --json
  
  # Log field coverage map for a slice
  uv run python experiments/metric_adapter_introspection.py --fields slice_uplift_goal
  uv run python experiments/metric_adapter_introspection.py --fields slice_uplift_goal --json
  
  # Multi-slice readiness summary
  uv run python experiments/metric_adapter_introspection.py --readiness-summary
  uv run python experiments/metric_adapter_introspection.py --readiness-summary --json
  
  # CI-grade one-line readiness verdict
  uv run python experiments/metric_adapter_introspection.py --summary-line
        """
    )
    
    parser.add_argument(
        "--slice",
        type=str,
        help="Name of the slice to introspect."
    )
    
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Generate human-readable summary."
    )
    
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify metric alignment across all sources."
    )
    
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export JSON schema contract."
    )
    
    parser.add_argument(
        "--drift",
        action="store_true",
        help="Detect schema drift."
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available slices."
    )
    
    parser.add_argument(
        "--verify-all",
        action="store_true",
        help="Verify alignment for all slices."
    )
    
    parser.add_argument(
        "--export-all",
        action="store_true",
        help="Export contracts for all slices."
    )
    
    parser.add_argument(
        "--health-check",
        action="store_true",
        help="Run CI health check on all slices. Exit 0 if OK/WARN, 1 if FAIL."
    )
    
    parser.add_argument(
        "--export-all-contracts",
        action="store_true",
        help="Export contract bundle with index to artifacts/phase_ii/metric_contracts/."
    )
    
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Show developer-friendly metric dashboard for all slices."
    )
    
    parser.add_argument(
        "--ready",
        type=str,
        metavar="SLICE",
        help="Per-slice readiness gate. Exit 0 if READY/DEGRADED, 1 if BLOCKED, 2 on error."
    )
    
    parser.add_argument(
        "--list-contracts",
        action="store_true",
        help="List contracts from existing bundle (read-only, does not recompute)."
    )
    
    parser.add_argument(
        "--fields",
        type=str,
        metavar="SLICE",
        help="Show log field coverage map for a slice."
    )
    
    parser.add_argument(
        "--readiness-summary",
        action="store_true",
        help="Show multi-slice readiness summary for all slices."
    )
    
    parser.add_argument(
        "--summary-line",
        action="store_true",
        help="Output single-line CI-grade readiness verdict. Exit 0 for OK/WARN, 1 for BLOCK."
    )
    
    parser.add_argument(
        "--curriculum",
        type=Path,
        default=DEFAULT_CURRICULUM_PATH,
        help=f"Path to curriculum YAML (default: {DEFAULT_CURRICULUM_PATH})"
    )
    
    parser.add_argument(
        "--prereg",
        type=Path,
        default=DEFAULT_PREREG_PATH,
        help=f"Path to PREREG YAML (default: {DEFAULT_PREREG_PATH})"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for exported contracts."
    )
    
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format (where applicable)."
    )
    
    args = parser.parse_args()
    
    # List all slices
    if args.list:
        try:
            slices = list_all_slices(args.curriculum)
            if args.json:
                print(json.dumps(slices, indent=2))
            else:
                print("Available slices:")
                for name in slices:
                    print(f"  - {name}")
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        return 0
    
    # Verify all slices
    if args.verify_all:
        try:
            results = verify_all_slices(args.curriculum, args.prereg)
            
            if args.json:
                json_results = {
                    name: {
                        "status": report.status.value,
                        "harmonization_aligned": report.harmonization_aligned,
                        "prereg_aligned": report.prereg_aligned,
                        "adapter_aligned": report.adapter_aligned,
                        "errors": report.errors,
                        "warnings": report.warnings,
                    }
                    for name, report in results.items()
                }
                print(json.dumps(json_results, indent=2))
            else:
                aligned_count = sum(1 for r in results.values() if r.is_fully_aligned())
                total_count = len(results)
                
                print(f"Alignment Summary: {aligned_count}/{total_count} slices fully aligned\n")
                
                for name, report in results.items():
                    status_icon = "✓" if report.is_fully_aligned() else "✗"
                    print(f"  {status_icon} {name}: {report.status.value}")
                    if report.errors:
                        for error in report.errors[:2]:
                            print(f"      Error: {error}")
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        return 0
    
    # Export all contracts
    if args.export_all:
        try:
            contracts = export_all_contracts(args.curriculum, args.output_dir)
            
            if args.json:
                json_contracts = {
                    name: contract.to_dict()
                    for name, contract in contracts.items()
                }
                print(json.dumps(json_contracts, indent=2))
            else:
                print(f"Exported {len(contracts)} contracts")
                for name in contracts:
                    print(f"  - {name}")
                if args.output_dir:
                    print(f"\nOutput directory: {args.output_dir}")
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        return 0
    
    # CI Health Check
    if args.health_check:
        try:
            report = run_health_check(args.curriculum, args.prereg)
            
            if args.json:
                print(report.to_json())
            else:
                print(format_health_check_report(report))
            
            # Exit code: 0 if OK or WARN, 1 if FAIL
            return 0 if report.overall_status != HealthStatus.FAIL else 1
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
    
    # Export contract bundle with index
    if args.export_all_contracts:
        try:
            output_dir = args.output_dir if args.output_dir else DEFAULT_CONTRACT_OUTPUT_DIR
            bundle = export_contract_bundle(args.curriculum, output_dir)
            
            if args.json:
                print(bundle.to_json())
            else:
                print(f"Contract bundle exported successfully")
                print(f"  Output directory: {bundle.output_directory}")
                print(f"  Total contracts: {bundle.total_contracts}")
                print(f"  Index file: {output_dir / 'metric_contract_index.json'}")
                print(f"\nContracts:")
                for entry in bundle.contracts:
                    print(f"  - {entry.slice_name}: {entry.contract_path}")
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        return 0
    
    # Developer dashboard
    if args.dashboard:
        try:
            if args.json:
                dashboard_data = generate_metric_dashboard_json(args.curriculum)
                print(json.dumps(dashboard_data, indent=2))
            else:
                dashboard = generate_metric_dashboard(args.curriculum)
                print(dashboard)
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        return 0
    
    # Per-slice readiness check
    if args.ready:
        try:
            result = check_slice_readiness(args.ready, args.curriculum, args.prereg)
            
            if args.json:
                print(result.to_json())
            else:
                print(format_readiness_result(result))
            
            # Exit codes: 0 = READY/DEGRADED, 1 = BLOCKED, 2 = internal error
            if result.status == ReadinessStatus.BLOCKED:
                return 1
            return 0
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 2
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 2
        except Exception as e:
            print(f"Internal error: {e}", file=sys.stderr)
            return 2
    
    # List contracts from existing bundle
    if args.list_contracts:
        try:
            contract_dir = args.output_dir if args.output_dir else DEFAULT_CONTRACT_OUTPUT_DIR
            entries = list_contracts(contract_dir)
            
            if args.json:
                json_entries = [e.to_dict() for e in entries]
                print(json.dumps(json_entries, indent=2, sort_keys=True))
            else:
                print(format_contract_list(entries))
            
            return 0
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
    
    # Log field coverage map
    if args.fields:
        try:
            coverage = get_log_field_coverage(args.fields, args.curriculum)
            
            if args.json:
                print(coverage.to_json())
            else:
                print(format_log_field_coverage(coverage))
            
            return 0
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
    
    # Multi-slice readiness summary
    if args.readiness_summary:
        try:
            contract_dir = args.output_dir if args.output_dir else DEFAULT_CONTRACT_OUTPUT_DIR
            slices = list_all_slices(args.curriculum)
            
            # Check readiness for all slices
            results = []
            for slice_name in slices:
                result = check_slice_readiness(
                    slice_name,
                    args.curriculum,
                    args.prereg,
                    contract_dir
                )
                results.append(result)
            
            summary = summarize_readiness(results)
            
            if args.json:
                print(summary.to_json())
            else:
                # Human-readable format
                verdict = compute_readiness_verdict(summary)
                print("=" * 72)
                print("MULTI-SLICE READINESS SUMMARY")
                print("=" * 72)
                print(f"Schema Version: {summary.schema_version}")
                print(f"Total Slices: {summary.slice_count}")
                print(f"Ready: {summary.ready_count}")
                print(f"Degraded: {summary.degraded_count}")
                print(f"Blocked: {summary.blocked_count}")
                print(f"Verdict: {verdict.value}")
                print("")
                print("-" * 72)
                print("PER-SLICE STATUS")
                print("-" * 72)
                for slice_name in sorted(summary.slices.keys()):
                    s = summary.slices[slice_name]
                    status_icon = {"READY": "✓", "DEGRADED": "~", "BLOCKED": "✗"}.get(s["status"], "?")
                    print(f"  {status_icon} {slice_name}: {s['status']} (drift={s['drift_severity']})")
                print("=" * 72)
            
            return 0
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
    
    # CI-grade one-line readiness verdict
    if args.summary_line:
        try:
            contract_dir = args.output_dir if args.output_dir else DEFAULT_CONTRACT_OUTPUT_DIR
            slices = list_all_slices(args.curriculum)
            
            # Check readiness for all slices
            results = []
            for slice_name in slices:
                result = check_slice_readiness(
                    slice_name,
                    args.curriculum,
                    args.prereg,
                    contract_dir
                )
                results.append(result)
            
            summary = summarize_readiness(results)
            verdict = compute_readiness_verdict(summary)
            
            # Output single line (no newline at end for CI compatibility)
            print(format_readiness_summary_line(summary))
            
            return get_readiness_verdict_exit_code(verdict)
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
    
    # Single slice operations require --slice
    if not args.slice:
        parser.print_help()
        return 1
    
    slice_name = args.slice
    
    try:
        # Summary
        if args.summary:
            summary = summarize_metric_adapter(slice_name, args.curriculum)
            print(summary)
        
        # Verify
        elif args.verify:
            report = verify_metric_alignment(slice_name, args.curriculum, args.prereg)
            if args.json:
                json_report = {
                    "slice_name": report.slice_name,
                    "status": report.status.value,
                    "harmonization_aligned": report.harmonization_aligned,
                    "prereg_aligned": report.prereg_aligned,
                    "adapter_aligned": report.adapter_aligned,
                    "errors": report.errors,
                    "warnings": report.warnings,
                    "drifts": [
                        {
                            "field": d.field,
                            "source_a": d.source_a,
                            "source_b": d.source_b,
                            "value_a": str(d.value_a),
                            "value_b": str(d.value_b),
                            "severity": d.severity,
                            "description": d.description,
                        }
                        for d in report.drifts
                    ]
                }
                print(json.dumps(json_report, indent=2))
            else:
                print(format_alignment_report(report))
        
        # Export
        elif args.export:
            contract = export_metric_contract(slice_name, args.curriculum)
            print(contract.to_json())
        
        # Drift
        elif args.drift:
            drifts = detect_schema_drift(slice_name, args.curriculum, args.prereg)
            if args.json:
                json_drifts = [
                    {
                        "field": d.field,
                        "source_a": d.source_a,
                        "source_b": d.source_b,
                        "value_a": str(d.value_a),
                        "value_b": str(d.value_b),
                        "severity": d.severity,
                        "description": d.description,
                    }
                    for d in drifts
                ]
                print(json.dumps(json_drifts, indent=2))
            else:
                print(format_drift_report(drifts))
        
        else:
            parser.print_help()
            return 1
    
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

