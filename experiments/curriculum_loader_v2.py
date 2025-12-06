# PHASE II — NOT USED IN PHASE I
"""
Runtime curriculum loader for asymmetric slices.

This module provides deterministic loading and validation of Phase II
curriculum configurations. It is designed to fail-fast on malformed YAML
and produce identical hashes for identical configurations.

Absolute Safeguards:
- Do NOT load or modify Phase I curriculum files.
- All loaded slices must be clearly labeled "PHASE II — NOT USED IN PHASE I".
- Deterministic hashing: identical config produces identical hash.
"""

import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# Phase II curriculum file path (relative to repo root)
PHASE2_CURRICULUM_PATH = "config/curriculum_uplift_phase2.yaml"

# Phase I curriculum files that MUST NOT be loaded
PHASE1_FORBIDDEN_PATHS = frozenset([
    "config/curriculum.yaml",
    "configs/curriculum.yaml",
])

# Valid success metric kinds as defined in prereg
VALID_SUCCESS_METRIC_KINDS = frozenset([
    "goal_hit",
    "density",
    "chain_length",
    "multi_goal",
])


class CurriculumLoadError(Exception):
    """Raised when curriculum loading fails."""
    pass


class ValidationError(CurriculumLoadError):
    """Raised when slice validation fails."""
    pass


@dataclass
class UpliftSlice:
    """
    Normalized runtime object for a Phase II uplift slice.

    Attributes:
        name: Unique identifier for the slice.
        params: Slice parameters (atoms, depth bounds, max_candidates, etc.).
        success_metric: Success metric configuration (kind, parameters).
        uplift_spec: Additional uplift experiment specification.
    """
    name: str
    params: Dict[str, Any]
    success_metric: Dict[str, Any]
    uplift_spec: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary for hashing/serialization."""
        return {
            "name": self.name,
            "params": self.params,
            "success_metric": self.success_metric,
            "uplift_spec": self.uplift_spec,
        }


def _compute_deterministic_hash(data: Dict[str, Any]) -> str:
    """
    Compute a deterministic SHA-256 hash of a dictionary.

    Uses JSON with sorted keys to ensure identical hashes for identical data.
    """
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _is_phase1_path(path: Path) -> bool:
    """Check if a path refers to a Phase I curriculum file."""
    path_str = str(path)
    for forbidden in PHASE1_FORBIDDEN_PATHS:
        if path_str.endswith(forbidden) or forbidden in path_str:
            return True
    return False


def _validate_params(params: Dict[str, Any], slice_name: str) -> None:
    """
    Validate slice parameters.

    Raises ValidationError if validation fails.
    """
    # Check for atoms (optional but validated if present)
    if "atoms" in params:
        atoms = params["atoms"]
        if not isinstance(atoms, int) or atoms < 1:
            raise ValidationError(
                f"Slice '{slice_name}': 'atoms' must be a positive integer, got {atoms}"
            )

    # Check for depth bounds (optional but validated if present)
    if "depth_max" in params:
        depth_max = params["depth_max"]
        if not isinstance(depth_max, int) or depth_max < 1:
            raise ValidationError(
                f"Slice '{slice_name}': 'depth_max' must be a positive integer, got {depth_max}"
            )

    if "depth_min" in params:
        depth_min = params["depth_min"]
        if not isinstance(depth_min, int) or depth_min < 0:
            raise ValidationError(
                f"Slice '{slice_name}': 'depth_min' must be a non-negative integer, got {depth_min}"
            )
        # Check depth_min <= depth_max if both present
        if "depth_max" in params and depth_min > params["depth_max"]:
            raise ValidationError(
                f"Slice '{slice_name}': 'depth_min' ({depth_min}) must be <= 'depth_max' ({params['depth_max']})"
            )

    # Check for max_candidates (optional but validated if present)
    if "max_candidates" in params:
        max_candidates = params["max_candidates"]
        if not isinstance(max_candidates, int) or max_candidates < 1:
            raise ValidationError(
                f"Slice '{slice_name}': 'max_candidates' must be a positive integer, got {max_candidates}"
            )


def _validate_success_metric(success_metric: Dict[str, Any], slice_name: str) -> None:
    """
    Validate success metric configuration.

    Raises ValidationError if validation fails.
    """
    if not success_metric:
        # success_metric can be empty/None for some slices
        return

    if "kind" in success_metric:
        kind = success_metric["kind"]
        if kind not in VALID_SUCCESS_METRIC_KINDS:
            raise ValidationError(
                f"Slice '{slice_name}': success_metric.kind must be one of "
                f"{sorted(VALID_SUCCESS_METRIC_KINDS)}, got '{kind}'"
            )


def load_phase2_curriculum(
    config_path: Optional[Path] = None,
    *,
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Load the Phase II curriculum configuration from YAML.

    Args:
        config_path: Path to the curriculum YAML file. Defaults to
                     PHASE2_CURRICULUM_PATH relative to current directory.
        strict: If True, fail-fast on any validation error.

    Returns:
        The raw curriculum configuration dictionary.

    Raises:
        CurriculumLoadError: If the file cannot be loaded or parsed.
        ValidationError: If strict=True and validation fails.
    """
    if config_path is None:
        config_path = Path(PHASE2_CURRICULUM_PATH)

    # Safeguard: refuse to load Phase I curriculum
    if _is_phase1_path(config_path):
        raise CurriculumLoadError(
            f"Refusing to load Phase I curriculum file: {config_path}. "
            "This loader is for Phase II only."
        )

    if not config_path.exists():
        raise CurriculumLoadError(f"Curriculum file not found: {config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise CurriculumLoadError(f"Failed to parse YAML: {e}") from e

    if config is None:
        raise CurriculumLoadError(f"Empty or invalid YAML file: {config_path}")

    if not isinstance(config, dict):
        raise CurriculumLoadError(
            f"Curriculum must be a YAML mapping, got {type(config).__name__}"
        )

    return config


def load_slice(
    slice_name: str,
    config: Optional[Dict[str, Any]] = None,
    config_path: Optional[Path] = None,
    *,
    strict: bool = True,
) -> UpliftSlice:
    """
    Load and validate a single slice from the curriculum.

    Args:
        slice_name: Name of the slice to load.
        config: Pre-loaded curriculum config. If None, loads from config_path.
        config_path: Path to curriculum YAML (used if config is None).
        strict: If True, fail-fast on validation errors.

    Returns:
        A validated UpliftSlice instance.

    Raises:
        CurriculumLoadError: If loading fails.
        ValidationError: If validation fails.
    """
    if config is None:
        config = load_phase2_curriculum(config_path, strict=strict)

    slices = config.get("slices", {})
    if slice_name not in slices:
        available = sorted(slices.keys()) if slices else []
        raise CurriculumLoadError(
            f"Slice '{slice_name}' not found in curriculum. "
            f"Available slices: {available}"
        )

    slice_data = slices[slice_name]
    if not isinstance(slice_data, dict):
        raise ValidationError(
            f"Slice '{slice_name}' must be a mapping, got {type(slice_data).__name__}"
        )

    # Extract and validate params
    params = {}
    if "items" in slice_data:
        # Legacy format: items-based slice
        params["items"] = slice_data["items"]
    if "params" in slice_data:
        # Structured params format
        params.update(slice_data["params"])

    # Extract common fields that may be at top level
    for key in ["atoms", "depth_max", "depth_min", "max_candidates", "breadth_max", "total_max"]:
        if key in slice_data and key not in params:
            params[key] = slice_data[key]

    if strict:
        _validate_params(params, slice_name)

    # Extract success_metric
    success_metric = slice_data.get("success_metric", {})
    if isinstance(success_metric, str):
        # Simple string form: just the kind
        success_metric = {"kind": success_metric}

    if strict:
        _validate_success_metric(success_metric, slice_name)

    # Extract uplift_spec
    uplift_spec = slice_data.get("uplift_spec", {})
    if "description" in slice_data:
        uplift_spec["description"] = slice_data["description"]
    if "prereg_hash" in slice_data:
        uplift_spec["prereg_hash"] = slice_data["prereg_hash"]

    return UpliftSlice(
        name=slice_name,
        params=params,
        success_metric=success_metric,
        uplift_spec=uplift_spec,
    )


def load_all_slices(
    config: Optional[Dict[str, Any]] = None,
    config_path: Optional[Path] = None,
    *,
    strict: bool = True,
) -> List[UpliftSlice]:
    """
    Load all slices from the curriculum.

    Args:
        config: Pre-loaded curriculum config. If None, loads from config_path.
        config_path: Path to curriculum YAML (used if config is None).
        strict: If True, fail-fast on validation errors.

    Returns:
        A list of validated UpliftSlice instances, sorted by name for determinism.
    """
    if config is None:
        config = load_phase2_curriculum(config_path, strict=strict)

    slices = config.get("slices", {})
    if not slices:
        return []

    result = []
    for slice_name in sorted(slices.keys()):
        slice_obj = load_slice(slice_name, config, strict=strict)
        result.append(slice_obj)

    return result


def hash_slice_config(slice_obj: UpliftSlice) -> str:
    """
    Compute a deterministic hash of a slice configuration.

    This hash is stable across runs given identical configuration.

    Args:
        slice_obj: The UpliftSlice to hash.

    Returns:
        A hexadecimal SHA-256 hash string.
    """
    return _compute_deterministic_hash(slice_obj.to_dict())


def hash_curriculum_config(config: Dict[str, Any]) -> str:
    """
    Compute a deterministic hash of the entire curriculum configuration.

    Args:
        config: The curriculum configuration dictionary.

    Returns:
        A hexadecimal SHA-256 hash string.
    """
    return _compute_deterministic_hash(config)


class CurriculumLoader:
    """
    Runtime curriculum loader for Phase II uplift experiments.

    This class provides a convenient interface for loading and validating
    curriculum configurations with caching and deterministic hashing.
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        *,
        strict: bool = True,
    ):
        """
        Initialize the curriculum loader.

        Args:
            config_path: Path to the curriculum YAML file.
            strict: If True, fail-fast on validation errors.
        """
        self._config_path = config_path or Path(PHASE2_CURRICULUM_PATH)
        self._strict = strict
        self._config: Optional[Dict[str, Any]] = None
        self._slices: Dict[str, UpliftSlice] = {}
        self._config_hash: Optional[str] = None

    def _ensure_loaded(self) -> None:
        """Ensure the configuration is loaded."""
        if self._config is None:
            self._config = load_phase2_curriculum(
                self._config_path, strict=self._strict
            )
            self._config_hash = hash_curriculum_config(self._config)

    @property
    def config(self) -> Dict[str, Any]:
        """Get the raw curriculum configuration."""
        self._ensure_loaded()
        return self._config  # type: ignore

    @property
    def config_hash(self) -> str:
        """Get the deterministic hash of the curriculum configuration."""
        self._ensure_loaded()
        return self._config_hash  # type: ignore

    def get_slice(self, slice_name: str) -> UpliftSlice:
        """
        Get a slice by name, loading and validating if necessary.

        Args:
            slice_name: Name of the slice to get.

        Returns:
            The validated UpliftSlice instance.

        Raises:
            CurriculumLoadError: If loading fails.
            ValidationError: If validation fails.
        """
        if slice_name not in self._slices:
            self._ensure_loaded()
            self._slices[slice_name] = load_slice(
                slice_name, self._config, strict=self._strict
            )
        return self._slices[slice_name]

    def get_slice_hash(self, slice_name: str) -> str:
        """
        Get the deterministic hash of a slice configuration.

        Args:
            slice_name: Name of the slice.

        Returns:
            A hexadecimal SHA-256 hash string.
        """
        return hash_slice_config(self.get_slice(slice_name))

    def get_all_slices(self) -> List[UpliftSlice]:
        """
        Get all slices from the curriculum.

        Returns:
            A list of validated UpliftSlice instances, sorted by name.
        """
        self._ensure_loaded()
        slices = load_all_slices(self._config, strict=self._strict)
        for s in slices:
            self._slices[s.name] = s
        return slices

    def list_slice_names(self) -> List[str]:
        """
        List all available slice names.

        Returns:
            A sorted list of slice names.
        """
        self._ensure_loaded()
        return sorted(self._config.get("slices", {}).keys())  # type: ignore


def main() -> int:
    """CLI entry point for testing the curriculum loader."""
    import argparse

    parser = argparse.ArgumentParser(
        description="PHASE II Curriculum Loader v2",
        epilog="Loads and validates Phase II curriculum configurations.",
    )
    parser.add_argument(
        "--config",
        default=PHASE2_CURRICULUM_PATH,
        help="Path to the curriculum YAML file.",
    )
    parser.add_argument(
        "--slice",
        help="Load and display a specific slice.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available slice names.",
    )
    parser.add_argument(
        "--hash",
        action="store_true",
        help="Display config hash.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=True,
        help="Enable strict validation (default: True).",
    )

    args = parser.parse_args()

    try:
        loader = CurriculumLoader(Path(args.config), strict=args.strict)

        if args.list:
            print("PHASE II — NOT USED IN PHASE I")
            print("Available slices:")
            for name in loader.list_slice_names():
                print(f"  - {name}")
            return 0

        if args.slice:
            slice_obj = loader.get_slice(args.slice)
            slice_hash = loader.get_slice_hash(args.slice)
            print("PHASE II — NOT USED IN PHASE I")
            print(f"Slice: {slice_obj.name}")
            print(f"Hash: {slice_hash}")
            print(f"Params: {json.dumps(slice_obj.params, indent=2)}")
            print(f"Success Metric: {json.dumps(slice_obj.success_metric, indent=2)}")
            print(f"Uplift Spec: {json.dumps(slice_obj.uplift_spec, indent=2)}")
            return 0

        if args.hash:
            print("PHASE II — NOT USED IN PHASE I")
            print(f"Config Hash: {loader.config_hash}")
            return 0

        # Default: show summary
        print("PHASE II — NOT USED IN PHASE I")
        print(f"Config: {args.config}")
        print(f"Config Hash: {loader.config_hash}")
        print(f"Slices: {len(loader.list_slice_names())}")
        for name in loader.list_slice_names():
            slice_hash = loader.get_slice_hash(name)
            print(f"  - {name}: {slice_hash[:16]}...")

        return 0

    except CurriculumLoadError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
