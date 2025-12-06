# PHASE II — NOT USED IN PHASE I
#
# Canonical curriculum loader for Phase II uplift experiments.
# This module provides:
#   - UpliftSlice: dataclass for slice descriptors
#   - SuccessMetricSpec: dataclass centralizing metric kind, thresholds, target hashes
#   - CurriculumLoaderV2: loader with from_default_phase2_config() and list_slices()
#
# Cross-validates success_metric.kind against slice_success_metrics.py at load time.
# Preserves hash determinism for reproducibility.

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple

import yaml

# Valid metric kinds from slice_success_metrics.py
# Maps kind name to the corresponding function name
VALID_METRIC_KINDS: FrozenSet[str] = frozenset({
    "goal_hit",     # compute_goal_hit
    "sparse",       # compute_sparse_success (also called "density" in prereg)
    "chain_length", # compute_chain_success
    "multi_goal",   # compute_multi_goal_success
    "density",      # alias for sparse
})

# Map aliases to canonical kind names
METRIC_KIND_ALIASES: Dict[str, str] = {
    "density": "sparse",
}


def _canonical_metric_kind(kind: str) -> str:
    """Normalize metric kind to canonical name."""
    return METRIC_KIND_ALIASES.get(kind, kind)


@dataclass(frozen=True)
class SuccessMetricSpec:
    """
    PHASE II — NOT USED IN PHASE I

    Centralized specification for slice success metrics.

    Attributes:
        kind: The metric kind (goal_hit, sparse, chain_length, multi_goal).
        thresholds: Dict of threshold parameters (e.g., min_verified, min_chain_length).
        target_hashes: Frozenset of target formula hashes for goal-based metrics.
        parameters: Additional metric-specific parameters.
    """
    kind: str
    thresholds: Dict[str, Any] = field(default_factory=dict)
    target_hashes: FrozenSet[str] = field(default_factory=frozenset)
    parameters: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate metric kind at construction time."""
        canonical = _canonical_metric_kind(self.kind)
        if canonical not in VALID_METRIC_KINDS:
            raise ValueError(
                f"Invalid success_metric.kind '{self.kind}'. "
                f"Must be one of: {sorted(VALID_METRIC_KINDS)}"
            )
        # Ensure kind is stored as canonical form
        if self.kind != canonical:
            object.__setattr__(self, "kind", canonical)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], slice_name: str) -> "SuccessMetricSpec":
        """
        Create a SuccessMetricSpec from a dictionary.

        Args:
            data: Dict with 'kind' and optionally 'parameters', 'thresholds', 'target_hashes'.
            slice_name: Name of the slice (for error messages).

        Raises:
            ValueError: If 'kind' is missing or invalid.
        """
        kind = data.get("kind")
        if not kind:
            raise ValueError(f"Slice '{slice_name}' missing success_metric.kind")

        parameters = dict(data.get("parameters", {}))
        thresholds = dict(data.get("thresholds", parameters))  # fallback to parameters
        target_hashes_raw = data.get("target_hashes", parameters.get("target_hashes", []))

        if isinstance(target_hashes_raw, (list, tuple, set, frozenset)):
            target_hashes = frozenset(str(h) for h in target_hashes_raw)
        else:
            target_hashes = frozenset()

        return cls(
            kind=kind,
            thresholds=thresholds,
            target_hashes=target_hashes,
            parameters=parameters,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a dictionary for JSON output."""
        return {
            "kind": self.kind,
            "thresholds": self.thresholds,
            "target_hashes": sorted(self.target_hashes),
            "parameters": self.parameters,
        }


def _compute_deterministic_hash(data: Dict[str, Any]) -> str:
    """
    Compute a deterministic SHA-256 hash of a dictionary.

    Uses JSON canonical serialization (sorted keys) for reproducibility.
    """
    canonical_str = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class UpliftSlice:
    """
    PHASE II — NOT USED IN PHASE I

    Descriptor for an uplift experiment slice.

    Attributes:
        name: Unique slice identifier.
        description: Human-readable description.
        items: Tuple of items/formulas in this slice.
        prereg_hash: Preregistration hash for audit trail.
        success_metric: SuccessMetricSpec for this slice.
        config_hash: Deterministic hash of the slice configuration.
    """
    name: str
    description: str
    items: Tuple[str, ...]
    prereg_hash: str
    success_metric: SuccessMetricSpec
    config_hash: str

    @classmethod
    def from_dict(cls, name: str, data: Dict[str, Any]) -> "UpliftSlice":
        """
        Create an UpliftSlice from a dictionary.

        Args:
            name: Slice name (key in YAML).
            data: Dict with slice configuration.

        Raises:
            ValueError: If required fields are missing or invalid.
        """
        description = data.get("description", "")
        items_raw = data.get("items", [])
        if not isinstance(items_raw, (list, tuple)):
            raise ValueError(f"Slice '{name}' items must be a list")
        items = tuple(str(item) for item in items_raw)

        prereg_hash = data.get("prereg_hash", "")

        # Parse success_metric
        success_metric_data = data.get("success_metric", {})
        if not success_metric_data:
            # Default based on slice name for backward compatibility
            success_metric = _infer_success_metric(name)
        else:
            success_metric = SuccessMetricSpec.from_dict(success_metric_data, name)

        # Compute deterministic config hash
        hashable_data = {
            "name": name,
            "description": description,
            "items": list(items),
            "prereg_hash": prereg_hash,
            "success_metric": success_metric.to_dict(),
        }
        config_hash = _compute_deterministic_hash(hashable_data)

        return cls(
            name=name,
            description=description,
            items=items,
            prereg_hash=prereg_hash,
            success_metric=success_metric,
            config_hash=config_hash,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a dictionary for JSON output."""
        return {
            "name": self.name,
            "description": self.description,
            "items": list(self.items),
            "prereg_hash": self.prereg_hash,
            "success_metric": self.success_metric.to_dict(),
            "config_hash": self.config_hash,
        }


def _infer_success_metric(slice_name: str) -> SuccessMetricSpec:
    """
    Infer a default SuccessMetricSpec for backward compatibility.

    For slices without explicit success_metric config, infer based on name.
    This is a fallback for existing slice definitions.
    """
    # Default to sparse for simple arithmetic-style slices
    return SuccessMetricSpec(
        kind="sparse",
        thresholds={"min_verified": 1},
        target_hashes=frozenset(),
        parameters={"min_verified": 1},
    )


class CurriculumConfigError(Exception):
    """Raised when curriculum configuration is invalid."""

    def __init__(self, errors: List[str]):
        self.errors = errors
        super().__init__(f"Invalid curriculum configuration: {'; '.join(errors)}")


class CurriculumLoaderV2:
    """
    PHASE II — NOT USED IN PHASE I

    Canonical loader for Phase II uplift curriculum configurations.

    Provides:
        - Strict validation at load time
        - Cross-validation with slice_success_metrics.py
        - Deterministic slice hashing
        - list_slices() returning sorted UpliftSlice descriptors
    """

    def __init__(self, config: Dict[str, Any], config_path: Optional[Path] = None):
        """
        Initialize the loader with a parsed configuration.

        Args:
            config: Parsed YAML configuration dict.
            config_path: Optional path to the config file (for error messages).

        Raises:
            CurriculumConfigError: If validation fails.
        """
        self._config = config
        self._config_path = config_path
        self._slices: Dict[str, UpliftSlice] = {}
        self._load_and_validate()

    def _load_and_validate(self) -> None:
        """Load slices from config and validate."""
        errors: List[str] = []

        # Check version
        version = self._config.get("version")
        if version is None:
            errors.append("Missing 'version' field")
        elif version != 2.0 and version != 2:
            errors.append(f"Unsupported version {version}, expected 2.0")

        slices_data = self._config.get("slices", {})
        if not isinstance(slices_data, dict):
            errors.append("'slices' must be a dictionary")
            raise CurriculumConfigError(errors)

        if not slices_data:
            errors.append("No slices defined in configuration")
            raise CurriculumConfigError(errors)

        for slice_name, slice_config in slices_data.items():
            try:
                uplift_slice = UpliftSlice.from_dict(slice_name, slice_config)
                self._slices[slice_name] = uplift_slice
            except ValueError as e:
                errors.append(str(e))

        if errors:
            raise CurriculumConfigError(errors)

        # Cross-validate metric kinds
        self._cross_validate_metrics()

    def _cross_validate_metrics(self) -> None:
        """
        Cross-validate that all success_metric.kind values map to valid metrics.

        Ensures fail-fast at load time if there's a mismatch.
        """
        # Import slice_success_metrics to verify functions exist
        from experiments import slice_success_metrics

        metric_function_map = {
            "goal_hit": "compute_goal_hit",
            "sparse": "compute_sparse_success",
            "chain_length": "compute_chain_success",
            "multi_goal": "compute_multi_goal_success",
        }

        errors: List[str] = []
        for slice_obj in self._slices.values():
            kind = slice_obj.success_metric.kind
            func_name = metric_function_map.get(kind)
            if func_name is None:
                errors.append(
                    f"Slice '{slice_obj.name}': unknown metric kind '{kind}'"
                )
                continue

            if not hasattr(slice_success_metrics, func_name):
                errors.append(
                    f"Slice '{slice_obj.name}': metric function '{func_name}' "
                    f"not found in slice_success_metrics"
                )

        if errors:
            raise CurriculumConfigError(errors)

    def list_slices(self) -> List[UpliftSlice]:
        """
        Return sorted list of UpliftSlice descriptors.

        Slices are sorted by name for deterministic ordering.
        """
        return sorted(self._slices.values(), key=lambda s: s.name)

    def get_slice(self, name: str) -> UpliftSlice:
        """
        Get a specific slice by name.

        Args:
            name: Slice name.

        Raises:
            KeyError: If slice not found.
        """
        if name not in self._slices:
            raise KeyError(f"Slice '{name}' not found in curriculum")
        return self._slices[name]

    def get_success_metric_spec(self, slice_name: str) -> SuccessMetricSpec:
        """
        Get the SuccessMetricSpec for a specific slice.

        Args:
            slice_name: Slice name.

        Returns:
            SuccessMetricSpec for the slice.

        Raises:
            KeyError: If slice not found.
        """
        return self.get_slice(slice_name).success_metric

    @classmethod
    def from_yaml_path(cls, path: Path) -> "CurriculumLoaderV2":
        """
        Load curriculum from a YAML file path.

        Args:
            path: Path to the YAML file.

        Returns:
            CurriculumLoaderV2 instance.

        Raises:
            FileNotFoundError: If file doesn't exist.
            CurriculumConfigError: If validation fails.
        """
        if not path.exists():
            raise FileNotFoundError(f"Curriculum config not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        return cls(config, config_path=path)

    @classmethod
    def from_default_phase2_config(cls) -> "CurriculumLoaderV2":
        """
        Load the default Phase II curriculum configuration.

        Returns:
            CurriculumLoaderV2 instance for Phase II.

        Raises:
            FileNotFoundError: If config file doesn't exist.
            CurriculumConfigError: If validation fails.
        """
        config_path = Path(__file__).parent.parent / "config" / "curriculum_uplift_phase2.yaml"
        return cls.from_yaml_path(config_path)

    @property
    def version(self) -> float:
        """Return the configuration version."""
        return float(self._config.get("version", 2.0))

    @property
    def slice_names(self) -> List[str]:
        """Return sorted list of slice names."""
        return sorted(self._slices.keys())


def get_metric_function(kind: str) -> Callable:
    """
    Get the metric computation function for a given kind.

    Args:
        kind: Metric kind (goal_hit, sparse, chain_length, multi_goal).

    Returns:
        The corresponding function from slice_success_metrics.

    Raises:
        ValueError: If kind is unknown.
    """
    from experiments import slice_success_metrics

    canonical = _canonical_metric_kind(kind)

    function_map = {
        "goal_hit": slice_success_metrics.compute_goal_hit,
        "sparse": slice_success_metrics.compute_sparse_success,
        "chain_length": slice_success_metrics.compute_chain_success,
        "multi_goal": slice_success_metrics.compute_multi_goal_success,
    }

    func = function_map.get(canonical)
    if func is None:
        raise ValueError(f"Unknown metric kind: {kind}")

    return func
