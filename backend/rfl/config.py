"""Deprecated RFL config shim."""

import warnings

from rfl.config import *  # noqa: F401,F403

warnings.warn(
    "backend.rfl.config is deprecated; import rfl.config instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "RFLConfig",
    "CurriculumSlice",
    "load_config",
]
"""
RFL Experiment Configuration

Defines parameters for 40-run reflexive metabolism experiments.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
import json
import os

from backend.security.runtime_env import (
    MissingEnvironmentVariable,
    get_database_url,
    get_redis_url,
)


@dataclass
class CurriculumSlice:
    """
    Deterministic curriculum slice for the Reflexive Formal Learning loop.

    Each slice defines a contiguous run interval with fixed derivation policy.
    """

    name: str
    start_run: int
    end_run: int
    derive_steps: int
    max_breadth: int
    max_total: int
    depth_max: int

    def contains(self, run_index: int) -> bool:
        """Return True if the slice owns the provided 1-indexed run index."""
        return self.start_run <= run_index <= self.end_run

    def validate(self) -> None:
        """Validate structural invariants for the slice."""
        if self.start_run < 1:
            raise ValueError(f"curriculum slice '{self.name}' start_run must be ≥1, got {self.start_run}")
        if self.end_run < self.start_run:
            raise ValueError(
                f"curriculum slice '{self.name}' end_run must be ≥ start_run "
                f"({self.start_run}), got {self.end_run}"
            )
        for attr_name in ("derive_steps", "max_breadth", "max_total", "depth_max"):
            value = getattr(self, attr_name)
            if value <= 0:
                raise ValueError(
                    f"curriculum slice '{self.name}' {attr_name} must be >0, got {value}"
                )


@dataclass
class RFLConfig:
    """Configuration for RFL experiment suite."""

    # Experiment metadata
    experiment_id: str = "rfl_001"
    num_runs: int = 40
    random_seed: int = 42

    # Derivation parameters (per run)
    system_id: int = 1  # 1 = Propositional Logic
    derive_steps: int = 50
    max_breadth: int = 200
    max_total: int = 1000
    depth_max: int = 4

    # Statistical parameters
    bootstrap_replicates: int = 10000
    confidence_level: float = 0.95

    # Acceptance criteria
    coverage_threshold: float = 0.92
    uplift_threshold: float = 1.0

    # Database
    database_url: str = field(
        default_factory=lambda: get_database_url()
    )

    # Redis
    redis_url: str = field(
        default_factory=lambda: get_redis_url()
    )

    # Lean project
    lean_project_dir: str = field(
        default_factory=lambda: os.getenv(
            "LEAN_PROJECT_DIR",
            "C:\\dev\\mathledger\\backend\\lean_proj"
        )
    )

    # Output paths
    artifacts_dir: str = "artifacts/rfl"
    results_file: str = "rfl_results.json"
    coverage_file: str = "rfl_coverage.json"
    curves_file: str = "rfl_curves.png"

    # Curriculum ladder (auto-generated if empty)
    curriculum: List[CurriculumSlice] = field(default_factory=list)

    # Dual attestation - require two independent statistical checks to agree
    dual_attestation: bool = True
    dual_attestation_tolerance: float = 5e-3

    # Abstention discipline
    abstention_tolerance: float = 0.25

    # Baseline comparison
    baseline_run_id: Optional[str] = None  # If None, compare against pre-RFL corpus

    # Parallel execution
    parallel_runs: bool = False
    max_workers: int = 4

    def __post_init__(self) -> None:
        """Ensure curriculum ladder exists and matches configured run count."""
        self._synchronize_curriculum()

    # ---------------------------------------------------------------------#
    # Serialization helpers
    # ---------------------------------------------------------------------#

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def to_json(self, filepath: str) -> None:
        """Save configuration to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=True)

    @classmethod
    def from_json(cls, filepath: str) -> 'RFLConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if "curriculum" in data and data["curriculum"] is not None:
            data["curriculum"] = [
                CurriculumSlice(**slice_data) for slice_data in data["curriculum"]
            ]
        return cls(**data)

    @classmethod
    def from_env(cls) -> 'RFLConfig':
        """
        Load configuration from environment variables.

        Environment variables:
            RFL_EXPERIMENT_ID
            RFL_NUM_RUNS (default 40)
            RFL_RANDOM_SEED (default 42)
            DERIVE_STEPS
            DERIVE_MAX_BREADTH
            DERIVE_MAX_TOTAL
            DERIVE_DEPTH_MAX
            RFL_BOOTSTRAP_REPLICATES (default 10000)
            RFL_COVERAGE_THRESHOLD (default 0.92)
            RFL_UPLIFT_THRESHOLD (default 1.0)
        """
        config = cls(
            experiment_id=os.getenv("RFL_EXPERIMENT_ID", "rfl_001"),
            num_runs=int(os.getenv("RFL_NUM_RUNS", "40")),
            random_seed=int(os.getenv("RFL_RANDOM_SEED", "42")),
            system_id=int(os.getenv("DERIVE_SYSTEM_ID", "1")),
            derive_steps=int(os.getenv("DERIVE_STEPS", "50")),
            max_breadth=int(os.getenv("DERIVE_MAX_BREADTH", "200")),
            max_total=int(os.getenv("DERIVE_MAX_TOTAL", "1000")),
            depth_max=int(os.getenv("DERIVE_DEPTH_MAX", "4")),
            bootstrap_replicates=int(os.getenv("RFL_BOOTSTRAP_REPLICATES", "10000")),
            confidence_level=float(os.getenv("RFL_CONFIDENCE_LEVEL", "0.95")),
            coverage_threshold=float(os.getenv("RFL_COVERAGE_THRESHOLD", "0.92")),
            uplift_threshold=float(os.getenv("RFL_UPLIFT_THRESHOLD", "1.0")),
            artifacts_dir=os.getenv("RFL_ARTIFACTS_DIR", "artifacts/rfl"),
            parallel_runs=os.getenv("RFL_PARALLEL", "false").lower() == "true",
            max_workers=int(os.getenv("RFL_MAX_WORKERS", "4")),
            dual_attestation=os.getenv("RFL_DUAL_ATTESTATION", "true").lower() == "true",
            dual_attestation_tolerance=float(os.getenv("RFL_DUAL_ATTESTATION_TOLERANCE", "0.005")),
            abstention_tolerance=float(os.getenv("RFL_ABSTENTION_TOLERANCE", "0.25"))
        )
        return config

    def validate(self) -> None:
        """Validate configuration parameters."""
        errors = []

        if self.num_runs < 2:
            errors.append(f"num_runs must be ≥2 for statistical inference, got {self.num_runs}")

        if not (0 < self.coverage_threshold <= 1.0):
            errors.append(f"coverage_threshold must be in (0, 1], got {self.coverage_threshold}")

        if self.uplift_threshold < 0:
            errors.append(f"uplift_threshold must be ≥0, got {self.uplift_threshold}")

        if self.bootstrap_replicates < 1000:
            errors.append(f"bootstrap_replicates should be ≥1000 for accuracy, got {self.bootstrap_replicates}")

        if not (0 < self.confidence_level < 1.0):
            errors.append(f"confidence_level must be in (0, 1), got {self.confidence_level}")

        if self.derive_steps <= 0:
            errors.append(f"derive_steps must be >0, got {self.derive_steps}")

        if self.dual_attestation and self.dual_attestation_tolerance <= 0:
            errors.append(
                f"dual_attestation_tolerance must be >0 when dual_attestation is enabled, "
                f"got {self.dual_attestation_tolerance}"
            )

        if not (0 <= self.abstention_tolerance <= 1):
            errors.append(
                f"abstention_tolerance must be in [0, 1], got {self.abstention_tolerance}"
            )

        curriculum_errors = self._validate_curriculum_structure()
        errors.extend(curriculum_errors)

        if errors:
            raise ValueError("Invalid RFL configuration:\n" + "\n".join(f"  - {e}" for e in errors))

    # ------------------------------------------------------------------#
    # Curriculum utilities
    # ------------------------------------------------------------------#

    def resolve_slice(self, run_index: int) -> CurriculumSlice:
        """
        Resolve the curriculum slice for a 1-indexed run index.
        """
        if run_index < 1:
            raise ValueError(f"run_index must be ≥1, got {run_index}")
        for slice_cfg in self.curriculum:
            if slice_cfg.contains(run_index):
                return slice_cfg
        # Fallback to last slice if no match found (should not happen after validation)
        return self.curriculum[-1]

    def _synchronize_curriculum(self) -> None:
        """Ensure curriculum is present and normalized against num_runs."""
        if not self.curriculum:
            self.curriculum = self._build_default_curriculum()

        # Sort slices to ensure deterministic order
        self.curriculum = sorted(self.curriculum, key=lambda s: s.start_run)

        normalized: List[CurriculumSlice] = []
        cursor = 1

        for slice_cfg in self.curriculum:
            # Clamp spans to remaining runs
            span = slice_cfg.end_run - slice_cfg.start_run + 1
            if span <= 0:
                continue
            start = cursor
            end = min(self.num_runs, cursor + span - 1)
            normalized.append(
                CurriculumSlice(
                    name=slice_cfg.name,
                    start_run=start,
                    end_run=end,
                    derive_steps=slice_cfg.derive_steps,
                    max_breadth=slice_cfg.max_breadth,
                    max_total=slice_cfg.max_total,
                    depth_max=slice_cfg.depth_max
                )
            )
            cursor = end + 1
            if cursor > self.num_runs:
                break

        if cursor <= self.num_runs:
            normalized.append(
                CurriculumSlice(
                    name="tail",
                    start_run=cursor,
                    end_run=self.num_runs,
                    derive_steps=self.derive_steps,
                    max_breadth=self.max_breadth,
                    max_total=self.max_total,
                    depth_max=self.depth_max
                )
            )

        self.curriculum = normalized

    def _build_default_curriculum(self) -> List[CurriculumSlice]:
        """Construct deterministic warmup/core/refinement curriculum segments."""
        total_runs = self.num_runs
        if total_runs <= 2:
            return [
                CurriculumSlice(
                    name="core",
                    start_run=1,
                    end_run=total_runs,
                    derive_steps=self.derive_steps,
                    max_breadth=self.max_breadth,
                    max_total=self.max_total,
                    depth_max=self.depth_max
                )
            ]

        warmup_span = max(1, total_runs // 5)
        refinement_span = max(1, total_runs // 5)
        core_span = max(1, total_runs - warmup_span - refinement_span)

        # Ensure spans sum to total_runs
        deficit = total_runs - (warmup_span + core_span + refinement_span)
        core_span += deficit

        slices: List[CurriculumSlice] = []
        start = 1

        def append_slice(name: str, span: int, step_factor: float, breadth_factor: float, depth_shift: int) -> None:
            nonlocal start
            if span <= 0:
                return
            end = start + span - 1
            slices.append(
                CurriculumSlice(
                    name=name,
                    start_run=start,
                    end_run=end,
                    derive_steps=max(1, int(round(self.derive_steps * step_factor))),
                    max_breadth=max(1, int(round(self.max_breadth * breadth_factor))),
                    max_total=max(1, int(round(self.max_total * breadth_factor))),
                    depth_max=max(1, self.depth_max + depth_shift)
                )
            )
            start = end + 1

        append_slice("warmup", warmup_span, 0.5, 0.5, 0)
        append_slice("core", core_span, 1.0, 1.0, 0)
        append_slice("refinement", refinement_span, 1.1, 1.0, 1)

        return slices

    def _validate_curriculum_structure(self) -> List[str]:
        """Validate curriculum spans cover the experiment deterministically."""
        errors: List[str] = []
        if not self.curriculum:
            errors.append("curriculum must define at least one slice")
            return errors

        for slice_cfg in self.curriculum:
            try:
                slice_cfg.validate()
            except ValueError as exc:  # pragma: no cover - defensive
                errors.append(str(exc))

        expected_start = 1
        for slice_cfg in self.curriculum:
            if slice_cfg.start_run != expected_start:
                errors.append(
                    f"curriculum gap detected: expected slice starting at {expected_start}, "
                    f"found '{slice_cfg.name}' starting at {slice_cfg.start_run}"
                )
                break
                # break early to avoid cascading errors
            expected_start = slice_cfg.end_run + 1

        if expected_start - 1 != self.num_runs:
            errors.append(
                f"curriculum does not terminate at num_runs={self.num_runs}; "
                f"last covered run={expected_start - 1}"
            )

        return errors


# Default configuration for quick testing
RFL_QUICK_CONFIG = RFLConfig(
    experiment_id="rfl_quick",
    num_runs=5,
    derive_steps=10,
    max_breadth=50,
    max_total=200,
    bootstrap_replicates=1000
)

# Production configuration (full 40 runs)
RFL_PRODUCTION_CONFIG = RFLConfig(
    experiment_id="rfl_prod",
    num_runs=40,
    derive_steps=100,
    max_breadth=500,
    max_total=2000,
    bootstrap_replicates=10000
)
