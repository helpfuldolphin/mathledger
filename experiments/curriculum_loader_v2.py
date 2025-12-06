"""
PHASE II — Curriculum Loader V2 with Drift Guard

This module provides structured loading and validation for Phase II curriculum
configurations, including:
- Schema version tracking and validation
- Structural validation with clear error messages
- Success metric specification and cross-validation
- Curriculum introspection CLI
- Fingerprinting for drift detection

Owner: curriculum-architect agent

See:
- config/curriculum_uplift_phase2.yaml — primary curriculum config
- experiments/slice_success_metrics.py — success metric implementations
- experiments/prereg/PREREG_UPLIFT_U2.yaml — preregistration cross-reference
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore


# =============================================================================
# Dataclasses for Curriculum Structure
# =============================================================================


@dataclass(frozen=True)
class SuccessMetricSpec:
    """
    Specification for a slice success metric.
    
    Attributes:
        kind: Metric type name (must match function in slice_success_metrics.py)
        parameters: Metric-specific parameters (thresholds, etc.)
        target_hashes: Optional set of formula hashes for goal-based metrics
    """
    kind: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    target_hashes: Optional[Set[str]] = None
    
    def __post_init__(self):
        # Validate kind is non-empty
        if not self.kind or not isinstance(self.kind, str):
            raise ValueError(f"SuccessMetricSpec.kind must be non-empty string, got: {self.kind}")
        
        # Validate parameters is a dict
        if not isinstance(self.parameters, dict):
            raise ValueError(f"SuccessMetricSpec.parameters must be dict, got: {type(self.parameters)}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            'kind': self.kind,
            'parameters': self.parameters,
        }
        if self.target_hashes is not None:
            result['target_hashes'] = sorted(self.target_hashes)
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SuccessMetricSpec":
        """Create from dictionary."""
        target_hashes = data.get('target_hashes')
        if target_hashes is not None:
            target_hashes = set(target_hashes)
        
        return cls(
            kind=data['kind'],
            parameters=data.get('parameters', {}),
            target_hashes=target_hashes,
        )


@dataclass(frozen=True)
class UpliftSlice:
    """
    A single curriculum slice configuration for Phase II uplift experiments.
    
    Attributes:
        name: Unique slice identifier
        description: Human-readable description
        parameters: Slice parameters (atoms, depth, budget, etc.)
        success_metric: Success metric specification
        uplift: Uplift metadata (phase, experiment family, etc.)
        budget: Budget constraints (max cycles, candidates, etc.)
        formula_pool_entries: Initial formula pool for this slice
    """
    name: str
    description: str
    parameters: Dict[str, Any]
    success_metric: SuccessMetricSpec
    uplift: Dict[str, Any] = field(default_factory=dict)
    budget: Dict[str, Any] = field(default_factory=dict)
    formula_pool_entries: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        # Validate required fields
        if not self.name or not isinstance(self.name, str):
            raise ValueError(f"UpliftSlice.name must be non-empty string, got: {self.name}")
        
        if not isinstance(self.parameters, dict):
            raise ValueError(f"UpliftSlice.parameters must be dict, got: {type(self.parameters)}")
        
        if not isinstance(self.success_metric, SuccessMetricSpec):
            raise ValueError(f"UpliftSlice.success_metric must be SuccessMetricSpec, got: {type(self.success_metric)}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'description': self.description,
            'parameters': self.parameters,
            'success_metric': self.success_metric.to_dict(),
            'uplift': self.uplift,
            'budget': self.budget,
            'formula_pool_entries': self.formula_pool_entries,
        }
    
    @classmethod
    def from_dict(cls, slice_name: str, data: Dict[str, Any]) -> "UpliftSlice":
        """Create from dictionary with validation."""
        # Extract success_metric
        metric_data = data.get('success_metric')
        if not metric_data or not isinstance(metric_data, dict):
            raise ValueError(f"Slice '{slice_name}': missing or invalid 'success_metric'")
        
        success_metric = SuccessMetricSpec.from_dict(metric_data)
        
        return cls(
            name=slice_name,
            description=data.get('description', ''),
            parameters=data.get('parameters', {}),
            success_metric=success_metric,
            uplift=data.get('uplift', {}),
            budget=data.get('budget', {}),
            formula_pool_entries=data.get('formula_pool_entries', []),
        )


# =============================================================================
# Schema Version and Structural Validation
# =============================================================================


# Allowed schema versions
ALLOWED_SCHEMA_VERSIONS = {"phase2-v1"}


class CurriculumValidationError(Exception):
    """Raised when curriculum validation fails."""
    
    def __init__(self, errors: List[str]):
        self.errors = errors
        message = f"Curriculum validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        super().__init__(message)


def validate_curriculum_structure(raw_config: Dict[str, Any]) -> None:
    """
    Validate curriculum structure and raise CurriculumValidationError if invalid.
    
    Checks:
    - Schema version is present and allowed
    - Required top-level keys exist
    - Each slice has required fields
    - Success metric specifications are valid
    - No obviously malformed fields
    
    Args:
        raw_config: Raw configuration dictionary loaded from YAML
        
    Raises:
        CurriculumValidationError: If validation fails with detailed error messages
    """
    errors: List[str] = []
    
    # 1. Check schema version
    schema_version = raw_config.get('schema_version')
    if schema_version is None:
        errors.append("Missing required field 'schema_version' at top level")
    elif schema_version not in ALLOWED_SCHEMA_VERSIONS:
        errors.append(
            f"Unsupported schema_version '{schema_version}'. "
            f"Allowed versions: {sorted(ALLOWED_SCHEMA_VERSIONS)}"
        )
    
    # 2. Check version field (numeric version)
    if 'version' not in raw_config:
        errors.append("Missing required field 'version' at top level")
    
    # 3. Check slices dict exists
    slices = raw_config.get('slices')
    if not isinstance(slices, dict):
        errors.append("Missing or invalid 'slices' field (must be dict)")
        # Can't continue without slices
        if errors:
            raise CurriculumValidationError(errors)
        return
    
    # 4. Validate each slice
    if not slices:
        errors.append("Curriculum has no slices defined")
    
    for slice_name, slice_data in slices.items():
        slice_errors = _validate_slice_structure(slice_name, slice_data)
        errors.extend(slice_errors)
    
    if errors:
        raise CurriculumValidationError(errors)


def _validate_slice_structure(slice_name: str, slice_data: Dict[str, Any]) -> List[str]:
    """Validate a single slice structure."""
    errors: List[str] = []
    prefix = f"Slice '{slice_name}'"
    
    # Check required fields
    required_fields = ['description', 'parameters', 'success_metric']
    for field in required_fields:
        if field not in slice_data:
            errors.append(f"{prefix}: missing required field '{field}'")
    
    # Validate parameters
    params = slice_data.get('parameters')
    if params is not None and not isinstance(params, dict):
        errors.append(f"{prefix}: 'parameters' must be dict, got {type(params).__name__}")
    
    # Validate success_metric
    metric = slice_data.get('success_metric')
    if metric is not None:
        if not isinstance(metric, dict):
            errors.append(f"{prefix}: 'success_metric' must be dict, got {type(metric).__name__}")
        else:
            if 'kind' not in metric:
                errors.append(f"{prefix}: 'success_metric' missing required field 'kind'")
            else:
                kind = metric.get('kind')
                if not isinstance(kind, str) or not kind:
                    errors.append(
                        f"{prefix}: 'success_metric.kind' must be non-empty string, got: {kind}"
                    )
            
            # Validate parameters if present
            if 'parameters' in metric and not isinstance(metric['parameters'], dict):
                errors.append(
                    f"{prefix}: 'success_metric.parameters' must be dict, "
                    f"got {type(metric['parameters']).__name__}"
                )
    
    # Validate uplift metadata if present
    uplift = slice_data.get('uplift')
    if uplift is not None and not isinstance(uplift, dict):
        errors.append(f"{prefix}: 'uplift' must be dict, got {type(uplift).__name__}")
    
    # Validate budget if present
    budget = slice_data.get('budget')
    if budget is not None and not isinstance(budget, dict):
        errors.append(f"{prefix}: 'budget' must be dict, got {type(budget).__name__}")
    
    # Validate formula_pool_entries if present
    pool = slice_data.get('formula_pool_entries')
    if pool is not None and not isinstance(pool, list):
        errors.append(
            f"{prefix}: 'formula_pool_entries' must be list, got {type(pool).__name__}"
        )
    
    return errors


# =============================================================================
# Curriculum Loader V2
# =============================================================================


class CurriculumLoaderV2:
    """
    Phase II curriculum loader with schema validation and introspection.
    
    Provides:
    - Structured loading from YAML with validation
    - Schema version checking
    - Access to slices and metrics
    - Introspection methods for CLI
    """
    
    def __init__(
        self,
        slices: Dict[str, UpliftSlice],
        schema_version: str,
        raw_config: Dict[str, Any],
    ):
        """
        Initialize curriculum loader.
        
        Args:
            slices: Dictionary mapping slice names to UpliftSlice instances
            schema_version: Schema version string (e.g., 'phase2-v1')
            raw_config: Raw configuration dictionary from YAML
        """
        self.slices = slices
        self.schema_version = schema_version
        self.raw_config = raw_config
    
    @classmethod
    def from_yaml(cls, config_path: Path) -> "CurriculumLoaderV2":
        """
        Load curriculum from YAML file with validation.
        
        Args:
            config_path: Path to curriculum YAML file
            
        Returns:
            CurriculumLoaderV2 instance
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            CurriculumValidationError: If validation fails
            yaml.YAMLError: If YAML parsing fails
        """
        if yaml is None:
            raise ImportError(
                "PyYAML is required for curriculum loading. "
                "Install with: pip install pyyaml"
            )
        
        if not config_path.exists():
            raise FileNotFoundError(f"Curriculum config not found: {config_path}")
        
        # Load YAML
        with open(config_path, 'r', encoding='utf-8') as f:
            raw_config = yaml.safe_load(f)
        
        if not isinstance(raw_config, dict):
            raise CurriculumValidationError([f"Config must be dict, got {type(raw_config).__name__}"])
        
        # Validate structure
        validate_curriculum_structure(raw_config)
        
        # Extract schema version
        schema_version = raw_config['schema_version']
        
        # Parse slices
        slices_dict = raw_config['slices']
        slices = {}
        
        for slice_name, slice_data in slices_dict.items():
            try:
                slices[slice_name] = UpliftSlice.from_dict(slice_name, slice_data)
            except Exception as e:
                raise CurriculumValidationError([
                    f"Failed to parse slice '{slice_name}': {e}"
                ])
        
        return cls(slices=slices, schema_version=schema_version, raw_config=raw_config)
    
    @classmethod
    def from_default_phase2_config(cls) -> "CurriculumLoaderV2":
        """Load from default Phase II curriculum config path."""
        # Find config relative to this file
        module_dir = Path(__file__).parent.parent
        config_path = module_dir / "config" / "curriculum_uplift_phase2.yaml"
        return cls.from_yaml(config_path)
    
    def get_slice(self, slice_name: str) -> Optional[UpliftSlice]:
        """Get a slice by name."""
        return self.slices.get(slice_name)
    
    def list_slice_names(self) -> List[str]:
        """Get list of all slice names."""
        return sorted(self.slices.keys())
    
    def get_metric_kinds(self) -> Set[str]:
        """Get set of all success metric kinds used."""
        return {slice_obj.success_metric.kind for slice_obj in self.slices.values()}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'schema_version': self.schema_version,
            'slice_count': len(self.slices),
            'slices': {name: slice_obj.to_dict() for name, slice_obj in self.slices.items()},
        }


# =============================================================================
# Curriculum Fingerprinting for Drift Detection
# =============================================================================


@dataclass
class CurriculumFingerprint:
    """
    Stable fingerprint of curriculum configuration for drift detection.
    
    Attributes:
        schema_version: Schema version string
        slice_count: Number of slices in curriculum
        metric_kinds: Sorted list of unique metric kinds used
        hash: SHA-256 hash of canonical JSON representation
    """
    schema_version: str
    slice_count: int
    metric_kinds: List[str]
    hash: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'schema_version': self.schema_version,
            'slice_count': self.slice_count,
            'metric_kinds': self.metric_kinds,
            'hash': self.hash,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CurriculumFingerprint":
        """Create from dictionary."""
        return cls(
            schema_version=data['schema_version'],
            slice_count=data['slice_count'],
            metric_kinds=data['metric_kinds'],
            hash=data['hash'],
        )


def compute_curriculum_fingerprint(loader: CurriculumLoaderV2) -> CurriculumFingerprint:
    """
    Compute stable fingerprint of curriculum for drift detection.
    
    The fingerprint includes:
    - Schema version
    - Slice count
    - Set of metric kinds
    - SHA-256 hash of canonical JSON (sorted keys, stable order)
    
    Same curriculum configuration will produce same fingerprint across runs.
    Any change to slices, metrics, or parameters will change the hash.
    
    Args:
        loader: CurriculumLoaderV2 instance
        
    Returns:
        CurriculumFingerprint with stable hash
    """
    # Get sorted metric kinds
    metric_kinds = sorted(loader.get_metric_kinds())
    
    # Build canonical representation
    # Sort slices by name and use sorted dict serialization
    canonical_slices = {}
    for slice_name in sorted(loader.slices.keys()):
        slice_obj = loader.slices[slice_name]
        canonical_slices[slice_name] = slice_obj.to_dict()
    
    canonical_repr = {
        'schema_version': loader.schema_version,
        'slices': canonical_slices,
    }
    
    # Compute hash using stable JSON serialization
    canonical_json = json.dumps(canonical_repr, sort_keys=True, separators=(',', ':'))
    hash_digest = hashlib.sha256(canonical_json.encode('utf-8')).hexdigest()
    
    return CurriculumFingerprint(
        schema_version=loader.schema_version,
        slice_count=len(loader.slices),
        metric_kinds=metric_kinds,
        hash=hash_digest,
    )


# =============================================================================
# Drift Checking
# =============================================================================


@dataclass
class DriftReport:
    """Report of differences between two curriculum fingerprints."""
    matches: bool
    differences: List[str]
    
    def __str__(self) -> str:
        if self.matches:
            return "✓ Fingerprints match — no drift detected"
        else:
            return "✗ Fingerprints differ:\n" + "\n".join(f"  - {d}" for d in self.differences)


def check_drift(
    current: CurriculumFingerprint,
    expected: CurriculumFingerprint,
) -> DriftReport:
    """
    Compare two curriculum fingerprints and report differences.
    
    Args:
        current: Current curriculum fingerprint
        expected: Expected curriculum fingerprint
        
    Returns:
        DriftReport with match status and detailed differences
    """
    differences: List[str] = []
    
    # Check schema version
    if current.schema_version != expected.schema_version:
        differences.append(
            f"schema_version: expected '{expected.schema_version}', "
            f"got '{current.schema_version}'"
        )
    
    # Check slice count
    if current.slice_count != expected.slice_count:
        differences.append(
            f"slice_count: expected {expected.slice_count}, got {current.slice_count}"
        )
    
    # Check metric kinds
    current_kinds_set = set(current.metric_kinds)
    expected_kinds_set = set(expected.metric_kinds)
    
    if current_kinds_set != expected_kinds_set:
        added = current_kinds_set - expected_kinds_set
        removed = expected_kinds_set - current_kinds_set
        
        if added:
            differences.append(f"metric_kinds added: {sorted(added)}")
        if removed:
            differences.append(f"metric_kinds removed: {sorted(removed)}")
    
    # Check hash
    if current.hash != expected.hash:
        if not differences:
            # Schema, counts, and kinds match but hash differs
            # This means details/thresholds changed
            differences.append(
                "hash mismatch: structure same, but thresholds or details changed"
            )
        else:
            # Already have other differences, hash mismatch is expected
            differences.append(f"hash: expected {expected.hash[:16]}..., got {current.hash[:16]}...")
    
    return DriftReport(matches=len(differences) == 0, differences=differences)


# =============================================================================
# CLI Introspection Functions
# =============================================================================


def cli_list_slices(loader: CurriculumLoaderV2) -> str:
    """Generate list of slices for CLI output."""
    lines = [
        f"Schema Version: {loader.schema_version}",
        f"Slice Count: {len(loader.slices)}",
        "",
        "Slices:",
    ]
    
    for slice_name in loader.list_slice_names():
        slice_obj = loader.slices[slice_name]
        metric_kind = slice_obj.success_metric.kind
        lines.append(f"  - {slice_name} (metric: {metric_kind})")
    
    return "\n".join(lines)


def cli_show_slice(loader: CurriculumLoaderV2, slice_name: str) -> str:
    """Generate detailed slice information for CLI output."""
    slice_obj = loader.get_slice(slice_name)
    if slice_obj is None:
        return f"Error: Slice '{slice_name}' not found"
    
    lines = [
        f"Slice: {slice_obj.name}",
        f"Description: {slice_obj.description}",
        "",
        "Parameters:",
    ]
    
    for key, value in sorted(slice_obj.parameters.items()):
        lines.append(f"  {key}: {value}")
    
    lines.extend([
        "",
        "Success Metric:",
        f"  kind: {slice_obj.success_metric.kind}",
        "  parameters:",
    ])
    
    for key, value in sorted(slice_obj.success_metric.parameters.items()):
        lines.append(f"    {key}: {value}")
    
    if slice_obj.success_metric.target_hashes:
        lines.extend([
            "  target_hashes:",
            f"    count: {len(slice_obj.success_metric.target_hashes)}",
        ])
    
    if slice_obj.budget:
        lines.append("")
        lines.append("Budget:")
        for key, value in sorted(slice_obj.budget.items()):
            lines.append(f"  {key}: {value}")
    
    if slice_obj.uplift:
        lines.append("")
        lines.append("Uplift Metadata:")
        for key, value in sorted(slice_obj.uplift.items()):
            lines.append(f"  {key}: {value}")
    
    return "\n".join(lines)


def cli_show_metrics(loader: CurriculumLoaderV2) -> str:
    """Generate summary of all success metrics for CLI output."""
    lines = [
        f"Schema Version: {loader.schema_version}",
        f"Unique Metric Kinds: {len(loader.get_metric_kinds())}",
        "",
        "Metric Kinds:",
    ]
    
    for kind in sorted(loader.get_metric_kinds()):
        # Find slices using this metric
        slices_using = [
            name for name, slice_obj in loader.slices.items()
            if slice_obj.success_metric.kind == kind
        ]
        lines.append(f"  - {kind} (used by: {', '.join(sorted(slices_using))})")
    
    return "\n".join(lines)


# =============================================================================
# CLI Entry Point
# =============================================================================


def main():
    """CLI entry point for curriculum introspection and fingerprinting."""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(
        description="Phase II Curriculum Introspection and Drift Guard CLI",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    
    parser.add_argument(
        '--config',
        type=Path,
        default=None,
        help='Path to curriculum YAML file (default: config/curriculum_uplift_phase2.yaml)',
    )
    
    # Introspection commands
    parser.add_argument(
        '--list-slices',
        action='store_true',
        help='List all slices in curriculum',
    )
    
    parser.add_argument(
        '--show-slice',
        type=str,
        metavar='NAME',
        help='Show detailed information for a specific slice',
    )
    
    parser.add_argument(
        '--show-metrics',
        action='store_true',
        help='Show summary of all success metrics',
    )
    
    # Fingerprinting commands
    parser.add_argument(
        '--fingerprint',
        action='store_true',
        help='Compute and display curriculum fingerprint',
    )
    
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output fingerprint as JSON (requires --fingerprint)',
    )
    
    # Drift checking commands
    parser.add_argument(
        '--check-against',
        type=Path,
        metavar='PATH',
        help='Check current curriculum against expected fingerprint JSON file',
    )
    
    args = parser.parse_args()
    
    # Load curriculum
    try:
        if args.config:
            loader = CurriculumLoaderV2.from_yaml(args.config)
        else:
            loader = CurriculumLoaderV2.from_default_phase2_config()
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except CurriculumValidationError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading curriculum: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Execute command
    if args.list_slices:
        print(cli_list_slices(loader))
    
    elif args.show_slice:
        print(cli_show_slice(loader, args.show_slice))
    
    elif args.show_metrics:
        print(cli_show_metrics(loader))
    
    elif args.fingerprint:
        fingerprint = compute_curriculum_fingerprint(loader)
        if args.json:
            print(json.dumps(fingerprint.to_dict(), indent=2))
        else:
            print("Curriculum Fingerprint:")
            print(f"  Schema Version: {fingerprint.schema_version}")
            print(f"  Slice Count: {fingerprint.slice_count}")
            print(f"  Metric Kinds: {', '.join(fingerprint.metric_kinds)}")
            print(f"  Hash: {fingerprint.hash}")
    
    elif args.check_against:
        # Load expected fingerprint
        try:
            with open(args.check_against, 'r') as f:
                expected_data = json.load(f)
            expected = CurriculumFingerprint.from_dict(expected_data)
        except Exception as e:
            print(f"Error loading expected fingerprint: {e}", file=sys.stderr)
            sys.exit(1)
        
        # Compute current fingerprint
        current = compute_curriculum_fingerprint(loader)
        
        # Check for drift
        report = check_drift(current, expected)
        print(report)
        
        # Exit with appropriate code
        sys.exit(0 if report.matches else 1)
    
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == '__main__':
    main()
