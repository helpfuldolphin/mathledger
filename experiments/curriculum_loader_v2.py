# PHASE II — NOT USED IN PHASE I
# File: experiments/curriculum_loader_v2.py
"""
CurriculumLoaderV2: Phase II Uplift Slice Loader

This module provides loading and validation for Phase II uplift slices defined
in `config/curriculum_uplift_phase2.yaml`. It is NOT used for Phase I curriculum
(which lives in `config/curriculum.yaml`).

Phase II Slices:
----------------
The four asymmetric uplift slices are designed for environments where policy-based
candidate ordering should produce measurable improvements over random baseline:

1. `slice_uplift_goal`       - Goal-conditioned target (specific formula hit)
2. `slice_uplift_sparse`     - Sparse reward (few provable in large space)
3. `slice_uplift_tree`       - Chain depth (multi-step derivation required)
4. `slice_uplift_dependency` - Multiple subgoals (coordination required)

Core Field Definitions:
-----------------------
- `atoms`: Number of propositional atoms {p, q, r, ...} in the formula space.
           Higher values = larger search space.

- `depth_min` / `depth_max`: Bounds on syntax tree depth for generated formulas.
           Controls formula complexity; deeper = harder to explore exhaustively.

- `breadth_max`: Maximum candidates considered per Modus Ponens round.
           Tight budgets make ordering matter more.

- `total_max`: Total candidate budget per cycle across all MP rounds.

- `formula_pool`: Size of the initial formula pool from which candidates derive.

- `axiom_instances`: Number of axiom instantiations generated per cycle.
           More instances = more candidate paths, but same budget constraint.

Success Metrics:
----------------
Each slice specifies a `success_metric.kind` that maps to a pure function in
`experiments/slice_success_metrics.py`:

- `goal_hit`          -> compute_goal_hit()
- `sparse_success`    -> compute_sparse_success()
- `chain_success`     -> compute_chain_success()
- `multi_goal_success`-> compute_multi_goal_success()

The loader validates that all `success_metric.kind` values match known metrics.

Usage:
------
    from experiments.curriculum_loader_v2 import CurriculumLoaderV2

    loader = CurriculumLoaderV2()
    config = loader.get_slice_config('slice_uplift_goal')
    metric = loader.get_success_metric_config('slice_uplift_goal')
    config_hash = loader.hash_slice_config('slice_uplift_goal')

Reference Documents:
--------------------
- docs/PHASE2_RFL_UPLIFT_PLAN.md (slice definitions and expected uplift)
- RFL_UPLIFT_THEORY.md (theoretical framework)
- experiments/slice_success_metrics.py (success metric implementations)
- docs/VSD_PHASE_2.md (governance and evidence gates)
"""

import yaml
import json
import hashlib
from typing import Dict, Any, List, Set, Optional, Tuple
from dataclasses import dataclass

# Known success metric kinds from slice_success_metrics.py
VALID_SUCCESS_METRIC_KINDS: Set[str] = {
    'goal_hit',
    'sparse_success',
    'chain_success',
    'multi_goal_success',
}

# Required top-level fields for each slice
REQUIRED_SLICE_FIELDS: List[str] = [
    'description',
    'uplift',
    'parameters',
    'success_metric',
    'budget',
    'formula_pool_entries',
]

# Required fields within the 'parameters' block
REQUIRED_PARAMETER_FIELDS: List[str] = [
    'atoms',
    'depth_min',
    'depth_max',
    'breadth_max',
    'total_max',
    'formula_pool',
    'axiom_instances',
]

# Optional but recognized fields (prevents unknown field errors)
OPTIONAL_SLICE_FIELDS: Set[str] = set()  # Currently none

OPTIONAL_PARAMETER_FIELDS: Set[str] = {
    'timeout_s',
    'lean_timeout_s',
}

# Success metric parameter requirements
# Maps metric kind -> (required_yaml_params, optional_yaml_params)
SUCCESS_METRIC_PARAM_SCHEMA: Dict[str, Tuple[Set[str], Set[str]]] = {
    'goal_hit': (
        {'min_goal_hits', 'min_total_verified'},  # required
        set(),  # optional
    ),
    'sparse_success': (
        {'min_verified'},
        set(),
    ),
    'chain_success': (
        {'min_chain_length'},
        set(),
    ),
    'multi_goal_success': (
        {'required_goal_count'},
        set(),
    ),
}


@dataclass
class FormulaPoolIntegrityResult:
    """
    PHASE II — NOT USED IN PHASE I

    Result of formula pool integrity validation.

    Attributes:
        valid: True if all checks pass.
        duplicate_formulas: List of formulas appearing more than once.
        normalization_errors: List of (formula, error_message) tuples for formulas
                              that fail normalization.
        hash_collisions: List of (formula1, formula2, hash) tuples where different
                         normalized formulas produce the same hash (should never happen
                         with SHA256 but checked for correctness).
        normalized_hashes: Dict mapping original formula -> (normalized_form, hash).
    """
    valid: bool
    duplicate_formulas: List[str]
    normalization_errors: List[Tuple[str, str]]
    hash_collisions: List[Tuple[str, str, str]]
    normalized_hashes: Dict[str, Tuple[str, str]]


@dataclass
class SuccessMetricValidationResult:
    """
    PHASE II — NOT USED IN PHASE I

    Result of success metric parameter validation.

    Attributes:
        valid: True if all required parameters are present and no unknown params exist.
        metric_kind: The success_metric.kind value.
        missing_params: Set of required parameters not found in YAML.
        unknown_params: Set of parameters in YAML not recognized for this metric.
        param_values: Dict of parameter name -> value from YAML.
    """
    valid: bool
    metric_kind: str
    missing_params: Set[str]
    unknown_params: Set[str]
    param_values: Dict[str, Any]


class CurriculumLoaderV2:
    """
    PHASE II — NOT USED IN PHASE I

    Loads and validates the Phase II curriculum from a YAML file.

    This loader is specific to Phase II uplift slices and enforces:
    - Required fields are present
    - No unknown fields exist
    - success_metric.kind values match slice_success_metrics.py
    - Budget fields are valid
    - Parameter monotonicity can be checked (atoms, depth ordering)

    Attributes:
        filepath: Path to the curriculum YAML file.
        _data: Parsed YAML content.

    Example:
        >>> loader = CurriculumLoaderV2()
        >>> loader.list_slices()
        ['slice_uplift_goal', 'slice_uplift_sparse', 'slice_uplift_tree', 'slice_uplift_dependency']
        >>> config = loader.get_slice_config('slice_uplift_goal')
        >>> config['parameters']['atoms']
        4
    """

    def __init__(self, filepath: str = "config/curriculum_uplift_phase2.yaml"):
        """
        Initialize the loader by loading and validating the curriculum file.

        Args:
            filepath: Path to the Phase II curriculum YAML file.

        Raises:
            FileNotFoundError: If the curriculum file does not exist.
            ValueError: If validation fails (missing fields, invalid values).
        """
        self.filepath = filepath
        with open(filepath, 'r', encoding='utf-8') as f:
            self._data: Dict[str, Any] = yaml.safe_load(f)
        self._validate_curriculum()

    def _validate_curriculum(self) -> None:
        """
        Perform validation of the entire curriculum file.

        Validates:
        - Version is 2.x (Phase II)
        - 'slices' block exists
        - Each slice has required fields
        - No unknown fields in slices
        - success_metric.kind values are valid
        - Budget envelope is present

        Raises:
            ValueError: If any validation check fails.
        """
        # Validate version
        if 'version' not in self._data:
            raise ValueError("Curriculum file must contain a 'version' field.")
        version_str = str(self._data['version'])
        if not version_str.startswith('2.'):
            raise ValueError(
                f"Invalid curriculum version '{version_str}'. "
                "Phase II curriculum requires version 2.x."
            )

        # Validate slices block exists
        if 'slices' not in self._data:
            raise ValueError("Curriculum file must contain a 'slices' block.")

        if not isinstance(self._data['slices'], dict):
            raise ValueError("'slices' must be a dictionary mapping slice names to configs.")

        # Validate each slice
        for name, slice_config in self._data['slices'].items():
            self._validate_slice(name, slice_config)

    def _validate_slice(self, name: str, slice_config: Dict[str, Any]) -> None:
        """
        Validate a single slice configuration.

        Args:
            name: The slice name (e.g., 'slice_uplift_goal').
            slice_config: The slice configuration dictionary.

        Raises:
            ValueError: If the slice is missing required fields or has invalid values.
        """
        if not isinstance(slice_config, dict):
            raise ValueError(f"Slice '{name}' must be a dictionary.")

        # Check required top-level fields
        for key in REQUIRED_SLICE_FIELDS:
            if key not in slice_config:
                raise ValueError(f"Slice '{name}' is missing required key: '{key}'")

        # Check for unknown top-level fields
        known_fields = set(REQUIRED_SLICE_FIELDS) | OPTIONAL_SLICE_FIELDS
        for key in slice_config.keys():
            if key not in known_fields:
                raise ValueError(
                    f"Slice '{name}' has unknown field: '{key}'. "
                    f"Known fields: {sorted(known_fields)}"
                )

        # Validate parameters block
        params = slice_config['parameters']
        if not isinstance(params, dict):
            raise ValueError(f"Slice '{name}' parameters must be a dictionary.")

        for key in REQUIRED_PARAMETER_FIELDS:
            if key not in params:
                raise ValueError(
                    f"Slice '{name}' parameters missing required key: '{key}'"
                )

        # Check for unknown parameter fields
        known_params = set(REQUIRED_PARAMETER_FIELDS) | OPTIONAL_PARAMETER_FIELDS
        for key in params.keys():
            if key not in known_params:
                raise ValueError(
                    f"Slice '{name}' parameters has unknown field: '{key}'. "
                    f"Known fields: {sorted(known_params)}"
                )

        # Validate success_metric block
        success_metric = slice_config['success_metric']
        if not isinstance(success_metric, dict):
            raise ValueError(f"Slice '{name}' success_metric must be a dictionary.")

        if 'kind' not in success_metric:
            raise ValueError(f"Slice '{name}' success_metric is missing 'kind'.")

        metric_kind = success_metric['kind']
        if metric_kind not in VALID_SUCCESS_METRIC_KINDS:
            raise ValueError(
                f"Slice '{name}' has invalid success_metric.kind: '{metric_kind}'. "
                f"Valid kinds: {sorted(VALID_SUCCESS_METRIC_KINDS)}"
            )

        # Validate budget envelope
        budget = slice_config['budget']
        if not isinstance(budget, dict):
            raise ValueError(f"Slice '{name}' budget must be a dictionary.")

        if 'max_candidates_per_cycle' not in budget:
            raise ValueError(f"Slice '{name}' budget is missing 'max_candidates_per_cycle'.")

        # Validate formula_pool_entries
        pool = slice_config['formula_pool_entries']
        if not isinstance(pool, list):
            raise ValueError(f"Slice '{name}' formula_pool_entries must be a list.")

        if len(pool) == 0:
            raise ValueError(f"Slice '{name}' formula_pool_entries cannot be empty.")

    def list_slices(self) -> List[str]:
        """
        Return a list of all slice names in the curriculum.

        Returns:
            List of slice names in definition order.
        """
        return list(self._data['slices'].keys())

    def get_slice_config(self, slice_name: str) -> Dict[str, Any]:
        """
        Return the full configuration dictionary for a given slice.

        Args:
            slice_name: The name of the slice to retrieve.

        Returns:
            The complete slice configuration dictionary.

        Raises:
            KeyError: If the slice does not exist.
        """
        if slice_name not in self._data['slices']:
            raise KeyError(f"Slice '{slice_name}' not found in curriculum.")
        return self._data['slices'][slice_name]

    def get_parameters(self, slice_name: str) -> Dict[str, Any]:
        """
        Return the parameters block for a given slice.

        Args:
            slice_name: The name of the slice.

        Returns:
            Dictionary containing atoms, depth_min, depth_max, etc.
        """
        return self.get_slice_config(slice_name)['parameters']

    def get_success_metric_config(self, slice_name: str) -> Dict[str, Any]:
        """
        Return the success_metric configuration for a given slice.

        Args:
            slice_name: The name of the slice.

        Returns:
            Dictionary containing 'kind' and 'parameters' for the metric.
        """
        slice_config = self.get_slice_config(slice_name)
        return slice_config['success_metric']

    def get_budget(self, slice_name: str) -> Dict[str, Any]:
        """
        Return the budget envelope for a given slice.

        Args:
            slice_name: The name of the slice.

        Returns:
            Dictionary containing max_candidates_per_cycle, etc.
        """
        return self.get_slice_config(slice_name)['budget']

    def get_formula_pool(self, slice_name: str) -> List[str]:
        """
        Return the formula pool entries for a given slice.

        Args:
            slice_name: The name of the slice.

        Returns:
            List of formula strings.
        """
        return self.get_slice_config(slice_name)['formula_pool_entries']

    def hash_slice_config(self, slice_name: str) -> str:
        """
        Return a SHA256 hash of the canonical JSON representation of the slice config.

        This hash is deterministic and stable for preregistration purposes.
        Changes to the slice configuration will produce a different hash.

        Args:
            slice_name: The name of the slice to hash.

        Returns:
            Hex-encoded SHA256 hash string.
        """
        slice_config = self.get_slice_config(slice_name)
        # Use sort_keys=True for a canonical representation
        canonical_json = json.dumps(slice_config, sort_keys=True)
        return hashlib.sha256(canonical_json.encode('utf-8')).hexdigest()

    def validate_monotonicity(self) -> List[str]:
        """
        Check that slices respect monotonic complexity ordering.

        For slices within the same "theory rung," complexity should increase.
        This is measured by (atoms, depth_max) pairs.

        Returns:
            List of warning messages if any monotonicity violations are found.
            Empty list if all slices are properly ordered.

        Note:
            This is a soft check; violations are warnings, not errors.
            Different slice families (goal vs sparse vs tree vs dependency)
            may have different complexity orderings.
        """
        warnings: List[str] = []
        slices = self.list_slices()

        for i in range(len(slices) - 1):
            curr_name = slices[i]
            next_name = slices[i + 1]

            curr_params = self.get_parameters(curr_name)
            next_params = self.get_parameters(next_name)

            curr_complexity = (curr_params['atoms'], curr_params['depth_max'])
            next_complexity = (next_params['atoms'], next_params['depth_max'])

            # Check if complexity decreases (violation)
            if next_complexity < curr_complexity:
                warnings.append(
                    f"Monotonicity warning: '{next_name}' ({next_complexity}) "
                    f"has lower complexity than preceding '{curr_name}' ({curr_complexity})"
                )

        return warnings

    def get_version(self) -> str:
        """Return the curriculum version string."""
        return str(self._data['version'])

    @property
    def data(self) -> Dict[str, Any]:
        """Return the raw parsed YAML data (read-only access)."""
        return self._data

    # =========================================================================
    # INTROSPECTION UTILITIES (Phase II)
    # =========================================================================

    def describe_slice(self, slice_name: str) -> str:
        """
        PHASE II — NOT USED IN PHASE I

        Generate a human-readable textual summary of a slice configuration.

        Includes parameters, success metric, budget, and formula pool statistics.

        Args:
            slice_name: The name of the slice to describe.

        Returns:
            Multi-line string summarizing the slice configuration.

        Raises:
            KeyError: If the slice does not exist.

        Example:
            >>> loader = CurriculumLoaderV2()
            >>> print(loader.describe_slice('slice_uplift_goal'))
            Slice: slice_uplift_goal
            Description: Goal-conditioned uplift slice...
            ...
        """
        config = self.get_slice_config(slice_name)
        params = config['parameters']
        metric = config['success_metric']
        budget = config['budget']
        pool = config['formula_pool_entries']

        lines = [
            f"Slice: {slice_name}",
            f"{'=' * (len(slice_name) + 7)}",
            "",
            f"Description:",
            f"  {config['description'][:200]}{'...' if len(config['description']) > 200 else ''}",
            "",
            "Parameters:",
            f"  atoms:           {params['atoms']}",
            f"  depth_min:       {params['depth_min']}",
            f"  depth_max:       {params['depth_max']}",
            f"  breadth_max:     {params['breadth_max']}",
            f"  total_max:       {params['total_max']}",
            f"  formula_pool:    {params['formula_pool']}",
            f"  axiom_instances: {params['axiom_instances']}",
        ]

        if 'timeout_s' in params:
            lines.append(f"  timeout_s:       {params['timeout_s']}")
        if 'lean_timeout_s' in params:
            lines.append(f"  lean_timeout_s:  {params['lean_timeout_s']}")

        lines.extend([
            "",
            "Success Metric:",
            f"  kind: {metric['kind']}",
        ])
        if 'parameters' in metric:
            for k, v in sorted(metric['parameters'].items()):
                lines.append(f"  {k}: {v}")

        lines.extend([
            "",
            "Budget:",
            f"  max_candidates_per_cycle: {budget['max_candidates_per_cycle']}",
        ])
        if 'max_cycles_per_run' in budget:
            lines.append(f"  max_cycles_per_run:       {budget['max_cycles_per_run']}")

        lines.extend([
            "",
            f"Formula Pool: {len(pool)} entries",
            f"  First 5: {pool[:5]}",
        ])

        config_hash = self.hash_slice_config(slice_name)
        lines.extend([
            "",
            f"Config Hash: {config_hash}",
        ])

        return "\n".join(lines)

    def validate_success_metric(self, slice_name: str) -> SuccessMetricValidationResult:
        """
        PHASE II — NOT USED IN PHASE I

        Validate that success_metric.parameters satisfies the function signature
        requirements for the corresponding compute_* function in slice_success_metrics.py.

        Each metric kind has required and optional parameters:
        - goal_hit:          requires {min_goal_hits, min_total_verified}
        - sparse_success:    requires {min_verified}
        - chain_success:     requires {min_chain_length}
        - multi_goal_success: requires {required_goal_count}

        Args:
            slice_name: The name of the slice to validate.

        Returns:
            SuccessMetricValidationResult with validation details.

        Raises:
            KeyError: If the slice does not exist.
        """
        metric_config = self.get_success_metric_config(slice_name)
        kind = metric_config['kind']

        yaml_params = metric_config.get('parameters', {})
        if yaml_params is None:
            yaml_params = {}

        # Get the schema for this metric kind
        required_params, optional_params = SUCCESS_METRIC_PARAM_SCHEMA.get(
            kind, (set(), set())
        )

        yaml_param_keys = set(yaml_params.keys())

        # Check for missing required params
        missing = required_params - yaml_param_keys

        # Check for unknown params
        known_params = required_params | optional_params
        unknown = yaml_param_keys - known_params

        valid = len(missing) == 0 and len(unknown) == 0

        return SuccessMetricValidationResult(
            valid=valid,
            metric_kind=kind,
            missing_params=missing,
            unknown_params=unknown,
            param_values=dict(yaml_params),
        )

    def validate_formula_pool_integrity(
        self,
        slice_name: str,
    ) -> FormulaPoolIntegrityResult:
        """
        PHASE II — NOT USED IN PHASE I

        Validate formula pool integrity:
        - No duplicate formulas (exact string match)
        - No hash collisions after normalization
        - All formulas normalize cleanly (no exceptions)

        Uses normalization/canon.py:normalize() for canonical form and
        hashlib.sha256 for hash computation.

        Args:
            slice_name: The name of the slice to validate.

        Returns:
            FormulaPoolIntegrityResult with validation details.

        Raises:
            KeyError: If the slice does not exist.
        """
        # Import normalize locally to avoid circular imports
        try:
            from normalization.canon import normalize
        except ImportError:
            # Fallback if normalization module not available
            def normalize(s: str) -> str:
                return s.strip()

        pool = self.get_formula_pool(slice_name)

        # Check for duplicates (exact string match)
        seen_formulas: Set[str] = set()
        duplicates: List[str] = []
        for formula in pool:
            if formula in seen_formulas:
                duplicates.append(formula)
            seen_formulas.add(formula)

        # Normalize and hash each formula
        normalization_errors: List[Tuple[str, str]] = []
        normalized_hashes: Dict[str, Tuple[str, str]] = {}
        hash_to_formulas: Dict[str, List[str]] = {}

        for formula in pool:
            try:
                normalized = normalize(formula)
                formula_hash = hashlib.sha256(normalized.encode('utf-8')).hexdigest()
                normalized_hashes[formula] = (normalized, formula_hash)

                # Track hash -> formulas for collision detection
                if formula_hash not in hash_to_formulas:
                    hash_to_formulas[formula_hash] = []
                hash_to_formulas[formula_hash].append(formula)

            except Exception as e:
                normalization_errors.append((formula, str(e)))

        # Check for hash collisions (different normalized forms, same hash)
        # This should never happen with SHA256 but we check anyway
        hash_collisions: List[Tuple[str, str, str]] = []
        for formula_hash, formulas in hash_to_formulas.items():
            if len(formulas) > 1:
                # Check if they have different normalized forms
                normalized_forms = set()
                for f in formulas:
                    if f in normalized_hashes:
                        normalized_forms.add(normalized_hashes[f][0])
                if len(normalized_forms) > 1:
                    # True collision: different normalized forms, same hash
                    hash_collisions.append((formulas[0], formulas[1], formula_hash))

        valid = (
            len(duplicates) == 0
            and len(normalization_errors) == 0
            and len(hash_collisions) == 0
        )

        return FormulaPoolIntegrityResult(
            valid=valid,
            duplicate_formulas=duplicates,
            normalization_errors=normalization_errors,
            hash_collisions=hash_collisions,
            normalized_hashes=normalized_hashes,
        )

    def validate_all(self) -> Dict[str, Any]:
        """
        PHASE II — NOT USED IN PHASE I

        Run all validation checks on all slices and return a comprehensive report.

        Returns:
            Dictionary with structure:
            {
                'valid': bool,  # True if all checks pass
                'version': str,
                'slice_count': int,
                'slices': {
                    'slice_name': {
                        'success_metric_valid': bool,
                        'formula_pool_valid': bool,
                        'config_hash': str,
                        'issues': [...],
                    },
                    ...
                },
                'monotonicity_warnings': [...],
            }
        """
        result: Dict[str, Any] = {
            'valid': True,
            'version': self.get_version(),
            'slice_count': len(self.list_slices()),
            'slices': {},
            'monotonicity_warnings': self.validate_monotonicity(),
        }

        for slice_name in self.list_slices():
            slice_result: Dict[str, Any] = {
                'success_metric_valid': True,
                'formula_pool_valid': True,
                'config_hash': self.hash_slice_config(slice_name),
                'issues': [],
            }

            # Validate success metric
            metric_result = self.validate_success_metric(slice_name)
            slice_result['success_metric_valid'] = metric_result.valid
            if not metric_result.valid:
                result['valid'] = False
                if metric_result.missing_params:
                    slice_result['issues'].append(
                        f"Missing metric params: {sorted(metric_result.missing_params)}"
                    )
                if metric_result.unknown_params:
                    slice_result['issues'].append(
                        f"Unknown metric params: {sorted(metric_result.unknown_params)}"
                    )

            # Validate formula pool
            pool_result = self.validate_formula_pool_integrity(slice_name)
            slice_result['formula_pool_valid'] = pool_result.valid
            if not pool_result.valid:
                result['valid'] = False
                if pool_result.duplicate_formulas:
                    slice_result['issues'].append(
                        f"Duplicate formulas: {pool_result.duplicate_formulas}"
                    )
                if pool_result.normalization_errors:
                    slice_result['issues'].append(
                        f"Normalization errors: {len(pool_result.normalization_errors)}"
                    )
                if pool_result.hash_collisions:
                    slice_result['issues'].append(
                        f"Hash collisions: {len(pool_result.hash_collisions)}"
                    )

            result['slices'][slice_name] = slice_result

        return result

    def to_json(self, indent: int = 2) -> str:
        """
        PHASE II — NOT USED IN PHASE I

        Export the curriculum configuration as deterministic JSON.

        Uses sort_keys=True for canonical ordering.

        Args:
            indent: JSON indentation level (default 2).

        Returns:
            JSON string representation of the curriculum.
        """
        return json.dumps(self._data, sort_keys=True, indent=indent)
