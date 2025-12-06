"""
PHASE II — NOT USED IN PHASE I

CurriculumLoaderV2: A curriculum loader for Phase II uplift experiments.

This module provides dataclasses and loading utilities for Phase II curriculum
configuration. It centralizes metric configuration (kind + thresholds + hash sets)
in SuccessMetricSpec and provides cross-validation with slice_success_metrics.py.

Absolute Safeguards:
- Do NOT reinterpret Phase I logs as uplift evidence.
- All Phase II artifacts must be clearly labeled "PHASE II — NOT USED IN PHASE I".
- All code must remain deterministic.
- RFL uses verifiable feedback only.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

from experiments.slice_success_metrics import METRIC_KINDS, is_valid_metric_kind


class CurriculumLoadError(Exception):
    """
    PHASE II — NOT USED IN PHASE I

    Raised when curriculum configuration cannot be loaded or is invalid.
    """
    pass


class InvalidMetricKindError(ValueError):
    """
    PHASE II — NOT USED IN PHASE I

    Raised when a metric kind is not registered in METRIC_KINDS.
    """
    def __init__(self, kind: str):
        self.kind = kind
        available = ", ".join(sorted(METRIC_KINDS.keys()))
        super().__init__(
            f"Unknown metric kind '{kind}'. Valid kinds: {available}"
        )


class UnknownMetricKindError(CurriculumLoadError):
    """
    PHASE II — NOT USED IN PHASE I

    Raised when a slice references an unknown metric kind.
    """
    def __init__(self, slice_name: str, metric_kind: str):
        self.slice_name = slice_name
        self.metric_kind = metric_kind
        available = ", ".join(sorted(METRIC_KINDS.keys()))
        super().__init__(
            f"Slice '{slice_name}' references unknown metric kind '{metric_kind}'. "
            f"Available kinds: {available}"
        )


@dataclass(frozen=True)
class SuccessMetricSpec:
    """
    PHASE II — NOT USED IN PHASE I

    Specification for a slice-specific success metric.

    Attributes:
        kind: The type of metric (must be registered in METRIC_KINDS).
        thresholds: A dict of threshold parameters for the metric.
        target_hashes: Optional set of target statement hashes for goal-based metrics.
    """
    kind: str
    thresholds: Dict[str, Any] = field(default_factory=dict)
    target_hashes: Optional[Set[str]] = None

    def __post_init__(self) -> None:
        # Validate that kind is known
        if not is_valid_metric_kind(self.kind):
            raise InvalidMetricKindError(self.kind)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SuccessMetricSpec":
        """
        PHASE II — NOT USED IN PHASE I

        Create a SuccessMetricSpec from a dictionary.

        Args:
            data: Dict with 'kind', optional 'thresholds', and optional 'target_hashes'.

        Returns:
            A new SuccessMetricSpec instance.
        """
        kind = data.get("kind")
        if not kind:
            raise ValueError("SuccessMetricSpec requires 'kind' field")

        thresholds = dict(data.get("thresholds", {}))
        target_hashes_list = data.get("target_hashes")
        target_hashes = set(target_hashes_list) if target_hashes_list else None

        return cls(kind=kind, thresholds=thresholds, target_hashes=target_hashes)

    def to_dict(self) -> Dict[str, Any]:
        """
        PHASE II — NOT USED IN PHASE I

        Convert to a dictionary representation.
        """
        result: Dict[str, Any] = {"kind": self.kind}
        if self.thresholds:
            result["thresholds"] = dict(self.thresholds)
        if self.target_hashes:
            result["target_hashes"] = sorted(self.target_hashes)
        return result

    def get_required_params(self) -> Tuple[str, ...]:
        """
        PHASE II — NOT USED IN PHASE I

        Get the required parameters for this metric kind.
        """
        if self.kind in METRIC_KINDS:
            return METRIC_KINDS[self.kind][0]
        return ()

    def get_description(self) -> str:
        """
        PHASE II — NOT USED IN PHASE I

        Get the description for this metric kind.
        """
        if self.kind in METRIC_KINDS:
            return METRIC_KINDS[self.kind][2]
        return "Unknown metric kind"


@dataclass
class UpliftSlice:
    """
    PHASE II — NOT USED IN PHASE I

    Represents a curriculum slice for Phase II uplift experiments.

    Attributes:
        name: Unique identifier for the slice.
        description: Human-readable description.
        items: List of items/formulas in this slice.
        atoms: Number of distinct atoms/variables (for complexity estimation).
        depth_min: Minimum formula depth.
        depth_max: Maximum formula depth.
        total_max: Maximum total formulas to consider.
        formula_pool: Size of the formula pool for sampling.
        success_metric: The success metric specification for this slice.
        prereg_hash: Hash for preregistration cross-reference.
        verifier: The verifier to use (e.g., "truth_table", "lean").
        metadata: Additional slice-specific metadata.
    """
    name: str
    description: str
    items: List[str] = field(default_factory=list)
    atoms: int = 0
    depth_min: int = 0
    depth_max: int = 0
    total_max: int = 0
    formula_pool: int = 0
    success_metric: Optional[SuccessMetricSpec] = None
    prereg_hash: Optional[str] = None
    verifier: str = "truth_table"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, name: str, data: Dict[str, Any]) -> "UpliftSlice":
        """
        PHASE II — NOT USED IN PHASE I

        Create an UpliftSlice from a dictionary.

        Args:
            name: The slice name.
            data: Dict with slice configuration.

        Returns:
            A new UpliftSlice instance.
        """
        description = data.get("description", "")
        items = list(data.get("items", []))
        atoms = int(data.get("atoms", len(items)))
        depth_min = int(data.get("depth_min", 0))
        depth_max = int(data.get("depth_max", 0))
        total_max = int(data.get("total_max", len(items)))
        formula_pool = int(data.get("formula_pool", len(items)))
        prereg_hash = data.get("prereg_hash")
        verifier = data.get("verifier", "truth_table")

        # Parse success metric if present
        success_metric = None
        if "success_metric" in data:
            success_metric = SuccessMetricSpec.from_dict(data["success_metric"])

        # Collect remaining fields as metadata
        known_keys = {
            "description", "items", "atoms", "depth_min", "depth_max",
            "total_max", "formula_pool", "success_metric", "prereg_hash", "verifier"
        }
        metadata = {k: v for k, v in data.items() if k not in known_keys}

        return cls(
            name=name,
            description=description,
            items=items,
            atoms=atoms,
            depth_min=depth_min,
            depth_max=depth_max,
            total_max=total_max,
            formula_pool=formula_pool,
            success_metric=success_metric,
            prereg_hash=prereg_hash,
            verifier=verifier,
            metadata=metadata,
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        PHASE II — NOT USED IN PHASE I

        Convert to a dictionary representation.
        """
        result: Dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "atoms": self.atoms,
            "depth_min": self.depth_min,
            "depth_max": self.depth_max,
            "total_max": self.total_max,
            "formula_pool": self.formula_pool,
            "verifier": self.verifier,
        }
        if self.items:
            result["items"] = self.items
        if self.success_metric:
            result["success_metric"] = self.success_metric.to_dict()
        if self.prereg_hash:
            result["prereg_hash"] = self.prereg_hash
        if self.metadata:
            result.update(self.metadata)
        return result


@dataclass
class DegenerateCheckWarning:
    """
    PHASE II — NOT USED IN PHASE I

    A warning from the non-degenerate defaults check.
    """
    slice_name: str
    check: str
    message: str
    severity: str = "warning"  # "warning" or "error"

    def __str__(self) -> str:
        return f"[{self.severity.upper()}] {self.slice_name}: {self.check} - {self.message}"


class CurriculumLoaderV2:
    """
    PHASE II — NOT USED IN PHASE I

    Loader for Phase II curriculum configuration.

    This loader reads curriculum YAML and provides structured access to slices
    and their success metrics. It performs validation to ensure metric kinds
    are registered in slice_success_metrics.py.

    Attributes:
        config: The raw curriculum configuration dict.
        slices: Dict mapping slice names to UpliftSlice objects.
        version: The curriculum config version.
    """

    DEFAULT_CONFIG_PATH = "config/curriculum_uplift_phase2.yaml"

    def __init__(self, config: Dict[str, Any], validate_metrics: bool = True):
        """
        PHASE II — NOT USED IN PHASE I

        Initialize the loader with a configuration dict.

        Args:
            config: The curriculum configuration dictionary.
            validate_metrics: If True, validate that all metric kinds are known.

        Raises:
            UnknownMetricKindError: If a slice references an unknown metric kind.
        """
        self.config = config
        self.version = config.get("version", "1.0")
        self.slices: Dict[str, UpliftSlice] = {}

        # Parse slices
        slices_data = config.get("slices", {})
        for name, data in slices_data.items():
            try:
                self.slices[name] = UpliftSlice.from_dict(name, data)
            except InvalidMetricKindError as e:
                # Convert to UnknownMetricKindError with slice context
                raise UnknownMetricKindError(name, e.kind) from e

        # Validate metric kinds if requested
        if validate_metrics:
            self._validate_metric_kinds()

    def _validate_metric_kinds(self) -> None:
        """
        PHASE II — NOT USED IN PHASE I

        Validate that all slice success metrics reference known metric kinds.

        Raises:
            UnknownMetricKindError: If a slice references an unknown metric kind.
        """
        for name, slice_obj in self.slices.items():
            if slice_obj.success_metric is not None:
                kind = slice_obj.success_metric.kind
                if not is_valid_metric_kind(kind):
                    raise UnknownMetricKindError(name, kind)

    @classmethod
    def from_yaml_file(cls, path: str, validate_metrics: bool = True) -> "CurriculumLoaderV2":
        """
        PHASE II — NOT USED IN PHASE I

        Load curriculum from a YAML file.

        Args:
            path: Path to the YAML configuration file.
            validate_metrics: If True, validate that all metric kinds are known.

        Returns:
            A new CurriculumLoaderV2 instance.

        Raises:
            CurriculumLoadError: If the file cannot be read or parsed.
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            raise CurriculumLoadError(f"Curriculum config file not found: {path}")
        except yaml.YAMLError as e:
            raise CurriculumLoadError(f"Failed to parse YAML: {e}")

        if config is None:
            raise CurriculumLoadError(f"Empty curriculum config file: {path}")

        return cls(config, validate_metrics=validate_metrics)

    @classmethod
    def from_default_phase2_config(cls, validate_metrics: bool = True) -> "CurriculumLoaderV2":
        """
        PHASE II — NOT USED IN PHASE I

        Load the default Phase II curriculum configuration.

        This method searches for the config file in standard locations:
        1. config/curriculum_uplift_phase2.yaml (relative to cwd)
        2. Relative to this module's location

        Args:
            validate_metrics: If True, validate that all metric kinds are known.

        Returns:
            A new CurriculumLoaderV2 instance.

        Raises:
            CurriculumLoadError: If the config file cannot be found.
        """
        # Try relative to cwd
        cwd_path = Path(cls.DEFAULT_CONFIG_PATH)
        if cwd_path.exists():
            return cls.from_yaml_file(str(cwd_path), validate_metrics=validate_metrics)

        # Try relative to module location
        module_dir = Path(__file__).parent.parent
        module_path = module_dir / cls.DEFAULT_CONFIG_PATH
        if module_path.exists():
            return cls.from_yaml_file(str(module_path), validate_metrics=validate_metrics)

        # Try one more level up
        repo_path = module_dir.parent / cls.DEFAULT_CONFIG_PATH
        if repo_path.exists():
            return cls.from_yaml_file(str(repo_path), validate_metrics=validate_metrics)

        raise CurriculumLoadError(
            f"Default Phase II config not found. Searched: {cwd_path}, {module_path}, {repo_path}"
        )

    def get_slice(self, name: str) -> UpliftSlice:
        """
        PHASE II — NOT USED IN PHASE I

        Get a slice by name.

        Args:
            name: The slice name.

        Returns:
            The UpliftSlice object.

        Raises:
            KeyError: If the slice is not found.
        """
        if name not in self.slices:
            available = ", ".join(sorted(self.slices.keys()))
            raise KeyError(f"Slice '{name}' not found. Available slices: {available}")
        return self.slices[name]

    def get_slice_names(self) -> List[str]:
        """
        PHASE II — NOT USED IN PHASE I

        Get a list of all slice names in definition order.
        """
        return list(self.slices.keys())

    def get_slices_by_metric_kind(self, kind: str) -> List[UpliftSlice]:
        """
        PHASE II — NOT USED IN PHASE I

        Get all slices that use a specific metric kind.

        Args:
            kind: The metric kind to filter by.

        Returns:
            List of UpliftSlice objects using the specified metric kind.
        """
        result = []
        for slice_obj in self.slices.values():
            if slice_obj.success_metric and slice_obj.success_metric.kind == kind:
                result.append(slice_obj)
        return result

    def get_metric_kinds_in_use(self) -> Dict[str, List[str]]:
        """
        PHASE II — NOT USED IN PHASE I

        Get a mapping of metric kinds to the slices that use them.

        Returns:
            Dict mapping metric kind to list of slice names.
        """
        result: Dict[str, List[str]] = {}
        for name, slice_obj in self.slices.items():
            if slice_obj.success_metric:
                kind = slice_obj.success_metric.kind
                if kind not in result:
                    result[kind] = []
                result[kind].append(name)
        return result

    def verify_non_degenerate_defaults(self) -> List[DegenerateCheckWarning]:
        """
        PHASE II — NOT USED IN PHASE I

        Verify that slice defaults are non-degenerate (plausible given constraints).

        This performs simple static checks:
        - min_verified <= total_max (for sparse_success metrics)
        - min_goal_hits <= formula_pool (for goal_hit metrics)
        - depth_min <= depth_max
        - atoms > 0 if items exist
        - total_max > 0 if items exist

        Returns:
            List of DegenerateCheckWarning objects for any issues found.
        """
        warnings_list: List[DegenerateCheckWarning] = []

        for name, slice_obj in self.slices.items():
            # Check depth ordering
            if slice_obj.depth_min > slice_obj.depth_max:
                warnings_list.append(DegenerateCheckWarning(
                    slice_name=name,
                    check="depth_ordering",
                    message=f"depth_min ({slice_obj.depth_min}) > depth_max ({slice_obj.depth_max})",
                    severity="error",
                ))

            # Check atoms > 0 if items exist
            if slice_obj.items and slice_obj.atoms <= 0:
                warnings_list.append(DegenerateCheckWarning(
                    slice_name=name,
                    check="atoms_nonzero",
                    message=f"atoms={slice_obj.atoms} but slice has {len(slice_obj.items)} items",
                    severity="warning",
                ))

            # Check total_max > 0 if items exist
            if slice_obj.items and slice_obj.total_max <= 0:
                warnings_list.append(DegenerateCheckWarning(
                    slice_name=name,
                    check="total_max_nonzero",
                    message=f"total_max={slice_obj.total_max} but slice has {len(slice_obj.items)} items",
                    severity="warning",
                ))

            # Check metric-specific thresholds
            if slice_obj.success_metric:
                metric = slice_obj.success_metric

                if metric.kind == "sparse_success":
                    min_verified = metric.thresholds.get("min_verified", 0)
                    if min_verified > slice_obj.total_max > 0:
                        warnings_list.append(DegenerateCheckWarning(
                            slice_name=name,
                            check="sparse_threshold",
                            message=f"min_verified ({min_verified}) > total_max ({slice_obj.total_max})",
                            severity="warning",
                        ))

                elif metric.kind == "goal_hit":
                    min_hits = metric.thresholds.get("min_total_verified", 0)
                    # Use explicit None check to handle formula_pool=0 correctly
                    pool = slice_obj.formula_pool if slice_obj.formula_pool > 0 else len(slice_obj.items)
                    if min_hits > pool > 0:
                        warnings_list.append(DegenerateCheckWarning(
                            slice_name=name,
                            check="goal_threshold",
                            message=f"min_total_verified ({min_hits}) > formula_pool ({pool})",
                            severity="warning",
                        ))

                    # Also check against target_hashes size if specified
                    if metric.target_hashes:
                        if min_hits > len(metric.target_hashes):
                            warnings_list.append(DegenerateCheckWarning(
                                slice_name=name,
                                check="goal_target_count",
                                message=f"min_total_verified ({min_hits}) > target_hashes count ({len(metric.target_hashes)})",
                                severity="warning",
                            ))

                elif metric.kind == "chain_success":
                    min_chain = metric.thresholds.get("min_chain_length", 0)
                    if min_chain > slice_obj.depth_max > 0:
                        warnings_list.append(DegenerateCheckWarning(
                            slice_name=name,
                            check="chain_threshold",
                            message=f"min_chain_length ({min_chain}) > depth_max ({slice_obj.depth_max})",
                            severity="warning",
                        ))

        return warnings_list

    def to_summary_dict(self) -> Dict[str, Any]:
        """
        PHASE II — NOT USED IN PHASE I

        Get a summary dictionary of the curriculum.
        """
        return {
            "version": self.version,
            "slice_count": len(self.slices),
            "slices": {name: slice_obj.to_dict() for name, slice_obj in self.slices.items()},
            "metric_kinds_in_use": self.get_metric_kinds_in_use(),
        }
