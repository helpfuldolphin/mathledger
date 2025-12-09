# PHASE II — NOT USED IN PHASE I
# File: experiments/u2_pipeline.py
#
# Slice Metric Adapter Layer for U2 Uplift Experiments
# =====================================================
#
# This module provides a deterministic, schema-validated adapter layer that
# maps slice configurations to their corresponding success metrics.
#
# ABSOLUTE SAFEGUARDS:
#   - No claims of uplift.
#   - No modifications to governance, success_metric definitions, or theory.
#   - No Phase I artifacts touched.
#   - Determinism above all else.
#
# RESPONSIBILITIES:
#   1. Parse slice config from CurriculumLoaderV2
#   2. Resolve metric kind → compute_metric(kind, **kwargs)
#   3. Extract metric kwargs from cycle logs deterministically
#   4. Validate metric schemas against PREREG_UPLIFT_U2.yaml before execution

from typing import Dict, Any, Tuple, List, Set, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import hashlib


# -----------------------------------------------------------------------------
# Metric Kind Enumeration (exhaustive list of supported metric types)
# -----------------------------------------------------------------------------

class MetricKind(str, Enum):
    """Enumeration of all supported metric kinds for U2 experiments."""
    GOAL_HIT = "goal_hit"
    SPARSE_SUCCESS = "sparse_success"
    CHAIN_SUCCESS = "chain_success"
    MULTI_GOAL_SUCCESS = "multi_goal_success"


# -----------------------------------------------------------------------------
# Result Types
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class MetricResult:
    """
    Immutable result of a metric computation.
    
    Attributes:
        success: Boolean indicating whether the metric threshold was met.
        value: Numeric value of the metric (for reporting/analysis).
        details: Dictionary containing computation details for audit.
    """
    success: bool
    value: float
    details: Dict[str, Any]
    
    def to_tuple(self) -> Tuple[bool, float, Dict[str, Any]]:
        """Return as (success, value, details_dict) tuple."""
        return (self.success, self.value, self.details)


@dataclass
class SchemaValidationResult:
    """Result of schema validation."""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# -----------------------------------------------------------------------------
# Metric Harmonization Table
# -----------------------------------------------------------------------------

# This table defines the mapping between metric kinds and their requirements.
# It serves as the single source of truth for:
#   - Required log fields for each metric
#   - Required slice config fields for each metric
#   - Result interpretation modes (boolean vs. numeric)

METRIC_HARMONIZATION_TABLE: Dict[str, Dict[str, Any]] = {
    MetricKind.GOAL_HIT: {
        "description": "Success based on hitting specific target goal hashes.",
        "required_log_fields": ["verified_statements"],
        "required_slice_config_fields": ["success_metric.parameters.min_goal_hits", "success_metric.parameters.min_total_verified"],
        "runtime_fields": ["target_hashes"],  # Must be provided at runtime or in experiment config
        "result_interpretation": "boolean_with_count",
        "compute_function": "compute_goal_hit",
        "output_schema": {
            "success": "bool",
            "value": "float (count of target hits)",
            "details": {
                "target_hashes_count": "int",
                "hits_count": "int",
                "min_required": "int",
                "verified_count": "int",
            }
        }
    },
    MetricKind.SPARSE_SUCCESS: {
        "description": "Success based on minimum count of verified statements.",
        "required_log_fields": ["verified_count", "attempted_count"],
        "required_slice_config_fields": ["success_metric.parameters.min_verified"],
        "runtime_fields": [],
        "result_interpretation": "boolean_with_count",
        "compute_function": "compute_sparse_success",
        "output_schema": {
            "success": "bool",
            "value": "float (verified count)",
            "details": {
                "verified_count": "int",
                "attempted_count": "int",
                "min_required": "int",
            }
        }
    },
    MetricKind.CHAIN_SUCCESS: {
        "description": "Success based on verified dependency chain length.",
        "required_log_fields": ["verified_statements", "dependency_graph"],
        "required_slice_config_fields": ["success_metric.parameters.min_chain_length"],
        "runtime_fields": ["chain_target_hash"],  # Must be provided at runtime or in experiment config
        "result_interpretation": "boolean_with_length",
        "compute_function": "compute_chain_success",
        "output_schema": {
            "success": "bool",
            "value": "float (chain length)",
            "details": {
                "chain_target_hash": "str",
                "chain_length": "int",
                "min_required": "int",
                "verified_count": "int",
            }
        }
    },
    MetricKind.MULTI_GOAL_SUCCESS: {
        "description": "Success based on verifying all required goal hashes.",
        "required_log_fields": ["verified_hashes"],
        "required_slice_config_fields": ["success_metric.parameters.required_goal_count"],
        "runtime_fields": ["required_goal_hashes"],  # Must be provided at runtime or in experiment config
        "result_interpretation": "boolean_with_count",
        "compute_function": "compute_multi_goal_success",
        "output_schema": {
            "success": "bool",
            "value": "float (goals met count)",
            "details": {
                "required_goals_count": "int",
                "met_goals_count": "int",
                "all_met": "bool",
            }
        }
    },
}


def get_harmonization_table() -> Dict[str, Dict[str, Any]]:
    """
    Returns the complete Metric Harmonization Table.
    
    This table is used by audit_uplift_u2.py and other analysis tools
    to understand metric requirements and result interpretation.
    """
    return METRIC_HARMONIZATION_TABLE.copy()


# -----------------------------------------------------------------------------
# Schema Validation
# -----------------------------------------------------------------------------

class MetricSchemaValidator:
    """
    Validates metric configurations against PREREG_UPLIFT_U2.yaml schemas.
    
    This validator ensures:
        - Field completeness (no missing inputs)
        - No extraneous fields
        - Type correctness
        - Schema alignment with preregistration
    """
    
    # PREREG schema expectations per metric kind
    PREREG_SCHEMA: Dict[str, Dict[str, Any]] = {
        MetricKind.GOAL_HIT: {
            "required_parameters": ["min_goal_hits", "min_total_verified"],
            "optional_parameters": ["target_hashes"],
            "parameter_types": {
                "min_goal_hits": int,
                "min_total_verified": int,
                "target_hashes": list,
            }
        },
        MetricKind.SPARSE_SUCCESS: {
            "required_parameters": ["min_verified"],
            "optional_parameters": ["max_candidates"],
            "parameter_types": {
                "min_verified": int,
                "max_candidates": int,
            }
        },
        MetricKind.CHAIN_SUCCESS: {
            "required_parameters": ["min_chain_length"],
            "optional_parameters": ["chain_target_hash"],
            "parameter_types": {
                "min_chain_length": int,
                "chain_target_hash": str,
            }
        },
        MetricKind.MULTI_GOAL_SUCCESS: {
            "required_parameters": ["required_goal_count"],
            "optional_parameters": ["required_goal_hashes", "min_each_goal"],
            "parameter_types": {
                "required_goal_count": int,
                "required_goal_hashes": list,
                "min_each_goal": int,
            }
        },
    }
    
    def validate_slice_metric_config(
        self, 
        slice_name: str, 
        metric_config: Dict[str, Any]
    ) -> SchemaValidationResult:
        """
        Validates a slice's success_metric configuration against PREREG schema.
        
        Args:
            slice_name: Name of the slice being validated.
            metric_config: The success_metric block from slice config.
            
        Returns:
            SchemaValidationResult with validation status and any errors/warnings.
        """
        errors: List[str] = []
        warnings: List[str] = []
        
        # Check metric kind exists
        if "kind" not in metric_config:
            errors.append(f"Slice '{slice_name}': Missing 'kind' in success_metric.")
            return SchemaValidationResult(valid=False, errors=errors)
        
        kind_str = metric_config["kind"]
        
        # Validate metric kind is recognized
        try:
            metric_kind = MetricKind(kind_str)
        except ValueError:
            errors.append(
                f"Slice '{slice_name}': Unknown metric kind '{kind_str}'. "
                f"Valid kinds: {[k.value for k in MetricKind]}"
            )
            return SchemaValidationResult(valid=False, errors=errors)
        
        # Get PREREG schema for this metric kind
        prereg_schema = self.PREREG_SCHEMA.get(metric_kind)
        if prereg_schema is None:
            errors.append(f"Slice '{slice_name}': No PREREG schema defined for '{kind_str}'.")
            return SchemaValidationResult(valid=False, errors=errors)
        
        # Get parameters from metric config
        parameters = metric_config.get("parameters", {})
        
        # Check required parameters
        required_params = prereg_schema["required_parameters"]
        for param in required_params:
            if param not in parameters:
                errors.append(
                    f"Slice '{slice_name}': Missing required parameter '{param}' "
                    f"for metric kind '{kind_str}'."
                )
        
        # Check for extraneous parameters
        all_valid_params = set(required_params) | set(prereg_schema.get("optional_parameters", []))
        for param in parameters:
            if param not in all_valid_params:
                warnings.append(
                    f"Slice '{slice_name}': Extraneous parameter '{param}' "
                    f"for metric kind '{kind_str}'."
                )
        
        # Type validation
        param_types = prereg_schema.get("parameter_types", {})
        for param, value in parameters.items():
            expected_type = param_types.get(param)
            if expected_type is not None and not isinstance(value, expected_type):
                errors.append(
                    f"Slice '{slice_name}': Parameter '{param}' has type "
                    f"'{type(value).__name__}', expected '{expected_type.__name__}'."
                )
        
        return SchemaValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def validate_log_fields(
        self,
        metric_kind: MetricKind,
        log_record: Dict[str, Any]
    ) -> SchemaValidationResult:
        """
        Validates that a cycle log record contains all required fields for a metric.
        
        Args:
            metric_kind: The metric kind to validate against.
            log_record: A single cycle log record.
            
        Returns:
            SchemaValidationResult with validation status and any errors.
        """
        errors: List[str] = []
        
        harmonization_entry = METRIC_HARMONIZATION_TABLE.get(metric_kind)
        if harmonization_entry is None:
            errors.append(f"No harmonization entry for metric kind '{metric_kind}'.")
            return SchemaValidationResult(valid=False, errors=errors)
        
        required_fields = harmonization_entry["required_log_fields"]
        for field in required_fields:
            if field not in log_record:
                errors.append(f"Missing required log field '{field}' for metric '{metric_kind}'.")
        
        return SchemaValidationResult(valid=len(errors) == 0, errors=errors)


# -----------------------------------------------------------------------------
# Log Field Extractors
# -----------------------------------------------------------------------------

class LogFieldExtractor:
    """
    Deterministic extraction of metric inputs from cycle logs.
    
    All extraction methods are pure functions with no side effects,
    ensuring reproducibility under replay.
    """
    
    @staticmethod
    def extract_verified_statements(log_record: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extracts verified statements from a cycle log record.
        
        Handles both new and legacy schema formats:
          - New: log_record["verified_statements"]
          - Legacy: log_record["derivation"]["verified_statements"]
        
        Returns an empty list if not found (deterministic behavior).
        """
        # Try new schema first
        if "verified_statements" in log_record:
            return log_record["verified_statements"]
        
        # Try legacy schema
        derivation = log_record.get("derivation", {})
        if "verified_statements" in derivation:
            return derivation["verified_statements"]
        
        # Fallback: construct from verified hashes if available
        if "verified_hashes" in log_record:
            return [{"hash": h} for h in log_record["verified_hashes"]]
        
        return []
    
    @staticmethod
    def extract_verified_count(log_record: Dict[str, Any]) -> int:
        """
        Extracts the count of verified statements from a cycle log record.
        
        Handles multiple schema formats:
          - Direct: log_record["verified_count"]
          - Derivation: log_record["derivation"]["verified"]
          - From statements: len(verified_statements)
        """
        if "verified_count" in log_record:
            return int(log_record["verified_count"])
        
        derivation = log_record.get("derivation", {})
        if "verified" in derivation:
            return int(derivation["verified"])
        
        # Count from verified_statements
        statements = LogFieldExtractor.extract_verified_statements(log_record)
        return len(statements)
    
    @staticmethod
    def extract_attempted_count(log_record: Dict[str, Any]) -> int:
        """
        Extracts the count of attempted candidates from a cycle log record.
        """
        if "attempted_count" in log_record:
            return int(log_record["attempted_count"])
        
        derivation = log_record.get("derivation", {})
        if "candidates" in derivation:
            return int(derivation["candidates"])
        
        return 0
    
    @staticmethod
    def extract_verified_hashes(log_record: Dict[str, Any]) -> Set[str]:
        """
        Extracts the set of verified statement hashes from a cycle log record.
        """
        if "verified_hashes" in log_record:
            return set(log_record["verified_hashes"])
        
        statements = LogFieldExtractor.extract_verified_statements(log_record)
        return {s.get("hash", "") for s in statements if s.get("hash")}
    
    @staticmethod
    def extract_dependency_graph(log_record: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Extracts the dependency graph from a cycle log record.
        """
        if "dependency_graph" in log_record:
            return log_record["dependency_graph"]
        
        derivation = log_record.get("derivation", {})
        if "dependency_graph" in derivation:
            return derivation["dependency_graph"]
        
        return {}


# -----------------------------------------------------------------------------
# Slice Metric Adapter
# -----------------------------------------------------------------------------

class SliceMetricAdapter:
    """
    Adapter layer that maps slice configurations to metric computations.
    
    This class is the primary entry point for computing success metrics
    for U2 uplift experiments. It:
    
    1. Parses slice config from CurriculumLoaderV2
    2. Resolves metric kind → compute_metric(kind, **kwargs)
    3. Extracts metric kwargs from cycle logs deterministically
    4. Validates metric schemas against PREREG_UPLIFT_U2.yaml before execution
    
    All operations are deterministic and produce identical results under replay.
    """
    
    def __init__(self, slice_config: Dict[str, Any], slice_name: str):
        """
        Initialize the adapter with a slice configuration.
        
        Args:
            slice_config: The full slice configuration from CurriculumLoaderV2.
            slice_name: The name of the slice for error reporting.
        """
        self.slice_config = slice_config
        self.slice_name = slice_name
        self._metric_config = slice_config.get("success_metric", {})
        self._metric_kind = self._resolve_metric_kind()
        self._validator = MetricSchemaValidator()
        self._extractor = LogFieldExtractor()
        
        # Validate on construction
        validation = self._validator.validate_slice_metric_config(
            slice_name, self._metric_config
        )
        if not validation.valid:
            raise ValueError(
                f"Invalid metric configuration for slice '{slice_name}': "
                f"{'; '.join(validation.errors)}"
            )
    
    def _resolve_metric_kind(self) -> MetricKind:
        """Resolves the metric kind from the slice config."""
        kind_str = self._metric_config.get("kind", "")
        try:
            return MetricKind(kind_str)
        except ValueError:
            raise ValueError(
                f"Unknown metric kind '{kind_str}' for slice '{self.slice_name}'. "
                f"Valid kinds: {[k.value for k in MetricKind]}"
            )
    
    @property
    def metric_kind(self) -> MetricKind:
        """Returns the resolved metric kind."""
        return self._metric_kind
    
    @property
    def metric_parameters(self) -> Dict[str, Any]:
        """Returns the metric parameters from slice config."""
        return self._metric_config.get("parameters", {})
    
    def get_required_log_fields(self) -> List[str]:
        """Returns the list of required log fields for this slice's metric."""
        entry = METRIC_HARMONIZATION_TABLE.get(self._metric_kind, {})
        return entry.get("required_log_fields", [])
    
    def get_harmonization_entry(self) -> Dict[str, Any]:
        """Returns the harmonization table entry for this slice's metric."""
        return METRIC_HARMONIZATION_TABLE.get(self._metric_kind, {}).copy()
    
    def validate_log_record(self, log_record: Dict[str, Any]) -> SchemaValidationResult:
        """
        Validates that a log record has all required fields for metric computation.
        """
        return self._validator.validate_log_fields(self._metric_kind, log_record)
    
    def extract_metric_kwargs(
        self, 
        log_record: Dict[str, Any],
        runtime_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Extracts metric computation kwargs from a cycle log record.
        
        This method deterministically extracts all required inputs for the
        metric computation function from the log record and runtime config.
        
        Args:
            log_record: A single cycle log record.
            runtime_config: Optional runtime configuration (e.g., target_hashes).
            
        Returns:
            Dictionary of kwargs suitable for passing to compute_metric().
        """
        runtime_config = runtime_config or {}
        params = self.metric_parameters
        kwargs: Dict[str, Any] = {}
        
        if self._metric_kind == MetricKind.GOAL_HIT:
            kwargs = {
                "verified_statements": self._extractor.extract_verified_statements(log_record),
                "target_hashes": set(runtime_config.get("target_hashes", params.get("target_hashes", []))),
                "min_total_verified": params.get("min_total_verified", params.get("min_goal_hits", 1)),
            }
        
        elif self._metric_kind == MetricKind.SPARSE_SUCCESS:
            kwargs = {
                "verified_count": self._extractor.extract_verified_count(log_record),
                "attempted_count": self._extractor.extract_attempted_count(log_record),
                "min_verified": params.get("min_verified", 1),
            }
        
        elif self._metric_kind == MetricKind.CHAIN_SUCCESS:
            kwargs = {
                "verified_statements": self._extractor.extract_verified_statements(log_record),
                "dependency_graph": self._extractor.extract_dependency_graph(log_record),
                "chain_target_hash": runtime_config.get("chain_target_hash", params.get("chain_target_hash", "")),
                "min_chain_length": params.get("min_chain_length", 1),
            }
        
        elif self._metric_kind == MetricKind.MULTI_GOAL_SUCCESS:
            kwargs = {
                "verified_hashes": self._extractor.extract_verified_hashes(log_record),
                "required_goal_hashes": set(runtime_config.get("required_goal_hashes", params.get("required_goal_hashes", []))),
            }
        
        return kwargs
    
    def compute_metric(
        self,
        log_record: Dict[str, Any],
        runtime_config: Optional[Dict[str, Any]] = None
    ) -> MetricResult:
        """
        Computes the success metric for a cycle log record.
        
        This is the main entry point for metric computation. It:
        1. Validates the log record has required fields
        2. Extracts metric kwargs deterministically
        3. Calls the appropriate compute function
        4. Wraps result in MetricResult with details
        
        Args:
            log_record: A single cycle log record.
            runtime_config: Optional runtime configuration (e.g., target_hashes).
            
        Returns:
            MetricResult containing (success, value, details_dict).
        """
        # Import here to avoid circular imports
        from experiments.slice_success_metrics import (
            compute_goal_hit,
            compute_sparse_success,
            compute_chain_success,
            compute_multi_goal_success,
        )
        
        # Extract kwargs
        kwargs = self.extract_metric_kwargs(log_record, runtime_config)
        
        # Compute metric based on kind
        if self._metric_kind == MetricKind.GOAL_HIT:
            success, value = compute_goal_hit(
                kwargs["verified_statements"],
                kwargs["target_hashes"],
                kwargs["min_total_verified"],
            )
            details = {
                "target_hashes_count": len(kwargs["target_hashes"]),
                "hits_count": int(value),
                "min_required": kwargs["min_total_verified"],
                "verified_count": len(kwargs["verified_statements"]),
            }
        
        elif self._metric_kind == MetricKind.SPARSE_SUCCESS:
            success, value = compute_sparse_success(
                kwargs["verified_count"],
                kwargs["attempted_count"],
                kwargs["min_verified"],
            )
            details = {
                "verified_count": kwargs["verified_count"],
                "attempted_count": kwargs["attempted_count"],
                "min_required": kwargs["min_verified"],
            }
        
        elif self._metric_kind == MetricKind.CHAIN_SUCCESS:
            success, value = compute_chain_success(
                kwargs["verified_statements"],
                kwargs["dependency_graph"],
                kwargs["chain_target_hash"],
                kwargs["min_chain_length"],
            )
            details = {
                "chain_target_hash": kwargs["chain_target_hash"],
                "chain_length": int(value),
                "min_required": kwargs["min_chain_length"],
                "verified_count": len(kwargs["verified_statements"]),
            }
        
        elif self._metric_kind == MetricKind.MULTI_GOAL_SUCCESS:
            success, value = compute_multi_goal_success(
                kwargs["verified_hashes"],
                kwargs["required_goal_hashes"],
            )
            details = {
                "required_goals_count": len(kwargs["required_goal_hashes"]),
                "met_goals_count": int(value),
                "all_met": success,
            }
        
        else:
            raise ValueError(f"Unsupported metric kind: {self._metric_kind}")
        
        return MetricResult(success=success, value=value, details=details)


# -----------------------------------------------------------------------------
# Adapter Factory
# -----------------------------------------------------------------------------

def create_adapter_from_loader(loader, slice_name: str) -> SliceMetricAdapter:
    """
    Creates a SliceMetricAdapter from a CurriculumLoaderV2 instance.
    
    Args:
        loader: CurriculumLoaderV2 instance.
        slice_name: Name of the slice to create adapter for.
        
    Returns:
        SliceMetricAdapter configured for the specified slice.
    """
    slice_config = loader.get_slice_config(slice_name)
    return SliceMetricAdapter(slice_config, slice_name)


def compute_metric_for_slice(
    slice_config: Dict[str, Any],
    slice_name: str,
    log_record: Dict[str, Any],
    runtime_config: Optional[Dict[str, Any]] = None
) -> Tuple[bool, float, Dict[str, Any]]:
    """
    Convenience function to compute a metric for a single log record.
    
    This function creates an adapter, computes the metric, and returns
    the result as a tuple (success, value, details_dict).
    
    Args:
        slice_config: The full slice configuration.
        slice_name: Name of the slice.
        log_record: A single cycle log record.
        runtime_config: Optional runtime configuration.
        
    Returns:
        Tuple of (success, value, details_dict).
    """
    adapter = SliceMetricAdapter(slice_config, slice_name)
    result = adapter.compute_metric(log_record, runtime_config)
    return result.to_tuple()


# -----------------------------------------------------------------------------
# Batch Processing
# -----------------------------------------------------------------------------

def compute_metrics_for_log_batch(
    adapter: SliceMetricAdapter,
    log_records: List[Dict[str, Any]],
    runtime_config: Optional[Dict[str, Any]] = None
) -> List[MetricResult]:
    """
    Computes metrics for a batch of log records.
    
    Args:
        adapter: SliceMetricAdapter instance.
        log_records: List of cycle log records.
        runtime_config: Optional runtime configuration.
        
    Returns:
        List of MetricResult objects, one per log record.
    """
    return [
        adapter.compute_metric(record, runtime_config)
        for record in log_records
    ]


def compute_success_rate(results: List[MetricResult]) -> float:
    """
    Computes the success rate from a list of MetricResults.
    
    Args:
        results: List of MetricResult objects.
        
    Returns:
        Success rate as a float in [0.0, 1.0].
    """
    if not results:
        return 0.0
    successes = sum(1 for r in results if r.success)
    return successes / len(results)


# -----------------------------------------------------------------------------
# Configuration Hash (for determinism verification)
# -----------------------------------------------------------------------------

def hash_adapter_config(adapter: SliceMetricAdapter) -> str:
    """
    Computes a deterministic hash of an adapter's configuration.
    
    This can be used to verify that the same configuration is being
    used across replays.
    """
    config_data = {
        "slice_name": adapter.slice_name,
        "metric_kind": adapter.metric_kind.value,
        "metric_parameters": adapter.metric_parameters,
    }
    canonical_json = json.dumps(config_data, sort_keys=True)
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

