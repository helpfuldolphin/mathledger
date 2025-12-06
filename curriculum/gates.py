"""
MathLedger Curriculum Control Module

Implements Reflexive Formal Learning (RFL) advancement gates for curriculum slices.
The gates enforce deterministic progression using coverage, abstention mass, proof
velocity, slice caps, and monotonicity invariants.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from substrate.repro.determinism import deterministic_timestamp, deterministic_timestamp_from_content

# Try to import yaml, fallback to simple parser if not available
try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    # Simple YAML parser fallback for basic structures
    def yaml_safe_load(content: str) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        lines = content.split('\n')
        stack: List[Any] = [result]
        indent_stack = [-1]

        for raw in lines:
            line = raw.rstrip()
            if not line or line.lstrip().startswith('#'):
                continue

            indent = len(raw) - len(raw.lstrip())
            while indent_stack and indent <= indent_stack[-1]:
                stack.pop()
                indent_stack.pop()

            if line.lstrip().startswith('- '):
                item = line.lstrip()[2:]
                current = stack[-1]
                if not isinstance(current, list):
                    new_list: List[Any] = []
                    if isinstance(current, dict):
                        # The parent dict key was already created
                        key = list(current.keys())[-1]
                        current[key] = new_list
                    stack[-1] = new_list
                    current = new_list
                if ':' in item:
                    key, value = item.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    value = _coerce_scalar(value)
                    current.append({key: value})
                    stack.append(current[-1])
                    indent_stack.append(indent)
                else:
                    current.append(_coerce_scalar(item.strip()))
            else:
                if ':' not in line:
                    continue
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                current = stack[-1]
                if not value:
                    new_map: Dict[str, Any] = {}
                    if isinstance(current, list):
                        current.append({key: new_map})
                        stack.append(new_map)
                    else:
                        current[key] = new_map
                        stack.append(new_map)
                    indent_stack.append(indent)
                else:
                    if isinstance(current, list):
                        current.append({key: _coerce_scalar(value)})
                    else:
                        current[key] = _coerce_scalar(value)
        return result

    def yaml_dump(data: Dict[str, Any], f, **kwargs) -> None:
        def render(obj: Any, indent: int = 0) -> List[str]:
            prefix = '  ' * indent
            lines: List[str] = []
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if isinstance(value, (dict, list)):
                        lines.append(f"{prefix}{key}:")
                        lines.extend(render(value, indent + 1))
                    else:
                        scalar = _format_scalar(value)
                        lines.append(f"{prefix}{key}: {scalar}")
            elif isinstance(obj, list):
                for value in obj:
                    if isinstance(value, (dict, list)):
                        lines.append(f"{prefix}-")
                        lines.extend(render(value, indent + 1))
                    else:
                        lines.append(f"{prefix}- {_format_scalar(value)}")
            else:
                lines.append(f"{prefix}{_format_scalar(obj)}")
            return lines

        f.write('\n'.join(render(data)))

    def _coerce_scalar(text: str) -> Any:
        lowered = text.lower()
        if lowered in {'true', 'false'}:
            return lowered == 'true'
        try:
            if '.' in text or 'e' in lowered:
                return float(text)
            return int(text)
        except ValueError:
            if text.startswith('"') and text.endswith('"'):
                return text[1:-1]
            if text.startswith("'") and text.endswith("'"):
                return text[1:-1]
            return text

    def _format_scalar(value: Any) -> str:
        if isinstance(value, bool):
            return 'true' if value else 'false'
        if isinstance(value, (int, float)):
            return str(value)
        return f'"{value}"'

    class MockYaml:
        safe_load = staticmethod(yaml_safe_load)
        dump = staticmethod(yaml_dump)

    yaml = MockYaml()  # type: ignore


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _first_available(root: Dict[str, Any], paths: Iterable[Sequence[str]]) -> Any:
    for path in paths:
        node: Any = root
        missing = False
        for part in path:
            if not isinstance(node, dict) or part not in node:
                missing = True
                break
            node = node[part]
        if not missing:
            return node
    return None


@dataclass(frozen=True)
class CoverageGateSpec:
    ci_lower_min: float
    sample_min: int
    require_attestation: bool = True

    def __post_init__(self) -> None:
        if not (0.0 < self.ci_lower_min <= 1.0):
            raise ValueError(f"coverage ci_lower_min must be in (0, 1], got {self.ci_lower_min}")
        if self.sample_min <= 0:
            raise ValueError("coverage sample_min must be positive")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AbstentionGateSpec:
    max_rate_pct: float
    max_mass: int

    def __post_init__(self) -> None:
        if not (0.0 <= self.max_rate_pct <= 100.0):
            raise ValueError(f"abstention max_rate_pct must be within [0, 100], got {self.max_rate_pct}")
        if self.max_mass <= 0:
            raise ValueError("abstention max_mass must be positive")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VelocityGateSpec:
    min_pph: float
    stability_cv_max: float
    window_minutes: int

    def __post_init__(self) -> None:
        if self.min_pph <= 0:
            raise ValueError("velocity min_pph must be positive")
        if not (0.0 <= self.stability_cv_max <= 1.0):
            raise ValueError("velocity stability_cv_max must be in [0, 1]")
        if self.window_minutes <= 0:
            raise ValueError("velocity window_minutes must be positive")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CapsGateSpec:
    min_attempt_mass: int
    min_runtime_minutes: float
    backlog_max: float

    def __post_init__(self) -> None:
        if self.min_attempt_mass <= 0:
            raise ValueError("caps min_attempt_mass must be positive")
        if self.min_runtime_minutes <= 0:
            raise ValueError("caps min_runtime_minutes must be positive")
        if not (0.0 <= self.backlog_max <= 1.0):
            raise ValueError("caps backlog_max must be within [0, 1]")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SliceGates:
    coverage: CoverageGateSpec
    abstention: AbstentionGateSpec
    velocity: VelocityGateSpec
    caps: CapsGateSpec

    @classmethod
    def from_dict(cls, data: Dict[str, Any], slice_name: str) -> "SliceGates":
        required_keys = ['coverage', 'abstention', 'velocity', 'caps']
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Slice '{slice_name}' missing gate specification '{key}'")
        return cls(
            coverage=CoverageGateSpec(**data['coverage']),
            abstention=AbstentionGateSpec(**data['abstention']),
            velocity=VelocityGateSpec(**data['velocity']),
            caps=CapsGateSpec(**data['caps']),
        )


@dataclass
class CurriculumSlice:
    name: str
    params: Dict[str, Any]
    gates: SliceGates
    completed_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CurriculumSlice":
        for field_name in ['name', 'params', 'gates']:
            if field_name not in data:
                raise ValueError(f"Curriculum slice missing required field '{field_name}'")
        gates = SliceGates.from_dict(data['gates'], data['name'])
        params = dict(data['params'])
        metadata = {k: v for k, v in data.items() if k not in {'name', 'params', 'gates', 'completed_at'}}
        completed_at = data.get('completed_at')
        return cls(
            name=data['name'],
            params=params,
            gates=gates,
            completed_at=completed_at,
            metadata=metadata,
        )


@dataclass
class CurriculumSystem:
    slug: str
    description: str
    slices: List[CurriculumSlice]
    active_index: int
    monotonic_axes: Tuple[str, ...] = ()
    version: int = 2

    @classmethod
    def from_config(cls, slug: str, config: Dict[str, Any]) -> "CurriculumSystem":
        version = config.get('version')
        if version != 2:
            raise ValueError(f"Unsupported curriculum version: {version}")

        systems = config.get('systems', {})
        if slug not in systems:
            raise KeyError(f"System '{slug}' not found in curriculum config")

        system_cfg = systems[slug]
        description = system_cfg.get('description')
        if not description:
            raise ValueError(f"System '{slug}' missing description")

        invariants = system_cfg.get('invariants', {})
        monotonic_axes = tuple(invariants.get('monotonic_axes', []))

        slices_data = system_cfg.get('slices', [])
        if not slices_data:
            raise ValueError(f"System '{slug}' has no slices defined")
        slices = [CurriculumSlice.from_dict(item) for item in slices_data]

        active_name = system_cfg.get('active')
        active_index = cls._resolve_active_index(active_name, slices)

        system = cls(
            slug=slug,
            description=description,
            slices=slices,
            active_index=active_index,
            monotonic_axes=monotonic_axes,
            version=version,
        )
        system._validate_monotonicity()
        return system

    @staticmethod
    def _resolve_active_index(active_name: Optional[str], slices: List[CurriculumSlice]) -> int:
        if active_name:
            for idx, slice_obj in enumerate(slices):
                if slice_obj.name == active_name:
                    return idx
            raise ValueError(f"Active slice '{active_name}' not present in slices list")
        for idx, slice_obj in enumerate(slices):
            if slice_obj.completed_at is None:
                return idx
        raise ValueError("No incomplete slices available to mark as active")

    def _validate_monotonicity(self) -> None:
        if not self.monotonic_axes:
            return
        prior_values: Optional[Tuple[int, ...]] = None
        for slice_obj in self.slices:
            current_values: List[int] = []
            for axis in self.monotonic_axes:
                value = slice_obj.params.get(axis)
                if value is None:
                    raise ValueError(f"Slice '{slice_obj.name}' missing monotonic axis '{axis}'")
                current_values.append(int(value))
            current_tuple = tuple(current_values)
            if prior_values is not None:
                for prev, curr, axis in zip(prior_values, current_tuple, self.monotonic_axes):
                    if curr < prev:
                        raise ValueError(
                            f"Slice '{slice_obj.name}' violates monotonicity on axis '{axis}': "
                            f"{prev} -> {curr}"
                        )
            prior_values = current_tuple

    @property
    def active_slice(self) -> CurriculumSlice:
        return self.slices[self.active_index]

    @property
    def active_name(self) -> str:
        return self.active_slice.name

    def next_slice(self) -> Optional[CurriculumSlice]:
        for idx in range(self.active_index + 1, len(self.slices)):
            if self.slices[idx].completed_at is None:
                return self.slices[idx]
        return None

    def to_public_dict(self) -> Dict[str, Any]:
        return {
            'system_slug': self.slug,
            'description': self.description,
            'active': self.active_name,
            'slices': [self._slice_to_dict(s) for s in self.slices],
            'active_slice': self._slice_to_dict(self.active_slice),
        }

    @staticmethod
    def _slice_to_dict(slice_obj: CurriculumSlice) -> Dict[str, Any]:
        base = {
            'name': slice_obj.name,
            'params': slice_obj.params,
            'gates': {
                'coverage': slice_obj.gates.coverage.to_dict(),
                'abstention': slice_obj.gates.abstention.to_dict(),
                'velocity': slice_obj.gates.velocity.to_dict(),
                'caps': slice_obj.gates.caps.to_dict(),
            },
        }
        if slice_obj.completed_at is not None:
            base['completed_at'] = slice_obj.completed_at
        if slice_obj.metadata:
            base.update(slice_obj.metadata)
        return base


def make_first_organism_slice() -> CurriculumSlice:
    """
    Construct a permissive-but-observable curriculum slice for the First Organism test.

    The thresholds are tuned so a derivation run is allowed, but ratcheting will
    require measurable coverage, abstention, velocity, and backlog evidence.
    """
    gates = SliceGates(
        coverage=CoverageGateSpec(
            ci_lower_min=0.915,
            sample_min=16,
            require_attestation=True,
        ),
        abstention=AbstentionGateSpec(
            max_rate_pct=18.0,
            max_mass=640,
        ),
        velocity=VelocityGateSpec(
            min_pph=160.0,
            stability_cv_max=0.10,
            window_minutes=45,
        ),
        caps=CapsGateSpec(
            min_attempt_mass=2400,
            min_runtime_minutes=20.0,
            backlog_max=0.36,
        ),
    )
    params = {
        'atoms': 4,
        'depth_max': 5,
        'breadth_max': 1200,
        'total_max': 6000,
    }
    return CurriculumSlice(
        name="first-organism-pl",
        params=params,
        gates=gates,
    )


def build_first_organism_metrics(
    coverage_ci: float = 0.90,
    sample_size: int = 22,
    abstention_rate: float = 13.5,
    attempt_mass: int = 3200,
    proof_velocity_pph: float = 190.0,
    velocity_cv: float = 0.06,
    runtime_minutes: float = 28.0,
    backlog_fraction: float = 0.31,
    attestation_hash: str = "deadbeef" * 8,
) -> Dict[str, Any]:
    """
    Build a metrics payload that mirrors the First Organism trial.

    Coverage gate is deliberately below the slice threshold, but all other
    gates satisfy their limits so the gate evaluator can demonstrate a single
    coverage failure while still permitting a run.
    """
    return {
        "metrics": {
            "rfl": {
                "coverage": {
                    "ci_lower": coverage_ci,
                    "sample_size": sample_size,
                }
            },
            "success_rates": {
                "abstention_rate": abstention_rate,
            },
            "curriculum": {
                "active_slice": {
                    "attempt_mass": attempt_mass,
                    "wallclock_minutes": runtime_minutes,
                    "proof_velocity_cv": velocity_cv,
                }
            },
            "throughput": {
                "proofs_per_hour": proof_velocity_pph,
                "coefficient_of_variation": velocity_cv,
                "window_minutes": 45,
            },
            "frontier": {
                "queue_backlog": backlog_fraction,
            },
        },
        "provenance": {
            "merkle_hash": attestation_hash,
        },
    }


def make_first_organism_pl2_hard_slice() -> CurriculumSlice:
    """
    Construct a harder curriculum slice for the First Organism test.

    This slice is designed to be challenging enough that the Lean interface
    might abstain or fail, but the gates are configured to allow the run
    to continue (ratchet stays put).
    """
    gates = SliceGates(
        coverage=CoverageGateSpec(
            ci_lower_min=0.90,
            sample_min=20,
            require_attestation=True,
        ),
        abstention=AbstentionGateSpec(
            max_rate_pct=20.0,
            max_mass=1000,
        ),
        velocity=VelocityGateSpec(
            min_pph=120.0,
            stability_cv_max=0.15,
            window_minutes=60,
        ),
        caps=CapsGateSpec(
            min_attempt_mass=2000,
            min_runtime_minutes=20.0,
            backlog_max=0.40,
        ),
    )
    params = {
        'atoms': 6,
        'depth_max': 8,
        'breadth_max': 2000,
        'total_max': 10000,
    }
    return CurriculumSlice(
        name="first_organism_pl2_hard",
        params=params,
        gates=gates,
    )


@dataclass
class NormalizedMetrics:
    coverage_ci_lower: Optional[float]
    coverage_sample_size: Optional[int]
    abstention_rate_pct: Optional[float]
    attempt_mass: Optional[int]
    slice_runtime_minutes: Optional[float]
    proof_velocity_pph: Optional[float]
    velocity_cv: Optional[float]
    backlog_fraction: Optional[float]
    attestation_hash: Optional[str]

    @classmethod
    def from_raw(cls, metrics: Dict[str, Any]) -> "NormalizedMetrics":
        root = metrics.get('metrics', metrics)

        coverage_ci_lower = _to_float(_first_available(root, [
            ('rfl', 'coverage', 'ci_lower'),
            ('coverage', 'ci_lower'),
            ('coverage_ci_lower',),
        ]))
        coverage_sample_size = _to_int(_first_available(root, [
            ('rfl', 'coverage', 'sample_size'),
            ('coverage', 'sample_size'),
            ('coverage_sample_size',),
        ]))
        abstention_rate = _to_float(_first_available(root, [
            ('success_rates', 'abstention_rate'),
            ('proofs', 'abstention_rate'),
            ('abstention_rate',),
        ]))
        attempt_mass = _to_int(_first_available(root, [
            ('curriculum', 'active_slice', 'attempt_mass'),
            ('curriculum', 'active_slice', 'attempts'),
            ('proofs', 'attempt_mass'),
            ('proofs', 'recent_hour'),
        ]))
        slice_runtime_minutes = _to_float(_first_available(root, [
            ('curriculum', 'active_slice', 'wallclock_minutes'),
            ('throughput', 'window_minutes'),
        ]))
        proof_velocity = _to_float(_first_available(root, [
            ('throughput', 'proofs_per_hour'),
            ('proof_velocity', 'per_hour'),
        ]))
        if proof_velocity is None:
            recent = _to_int(_first_available(root, [('proofs', 'recent_hour'), ('proofs', 'recent_window'), ]))
            if recent is not None:
                proof_velocity = float(recent)
        velocity_cv = _to_float(_first_available(root, [
            ('throughput', 'coefficient_of_variation'),
            ('variance', 'coefficient_of_variation'),
            ('curriculum', 'active_slice', 'proof_velocity_cv'),
        ]))
        backlog_fraction = _to_float(_first_available(root, [
            ('frontier', 'queue_backlog'),
            ('queue', 'backlog_fraction'),
        ]))
        attestation_hash = _first_available(metrics, [
            ('provenance', 'merkle_hash'),
            ('provenance', 'attestation_hash'),
        ])
        return cls(
            coverage_ci_lower=coverage_ci_lower,
            coverage_sample_size=coverage_sample_size,
            abstention_rate_pct=abstention_rate,
            attempt_mass=attempt_mass,
            slice_runtime_minutes=slice_runtime_minutes,
            proof_velocity_pph=proof_velocity,
            velocity_cv=velocity_cv,
            backlog_fraction=backlog_fraction,
            attestation_hash=attestation_hash if isinstance(attestation_hash, str) else None,
        )

    @property
    def abstention_mass(self) -> Optional[float]:
        if self.abstention_rate_pct is None or self.attempt_mass is None:
            return None
        return (self.abstention_rate_pct / 100.0) * self.attempt_mass


@dataclass
class GateStatus:
    gate: str
    passed: bool
    observed: Dict[str, Any]
    thresholds: Dict[str, Any]
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            'gate': self.gate,
            'passed': self.passed,
            'observed': self.observed,
            'thresholds': self.thresholds,
            'message': self.message,
        }


@dataclass
class GateVerdict:
    advance: bool
    reason: str
    audit: Dict[str, Any]


class GateEvaluator:
    def __init__(self, metrics: NormalizedMetrics, slice_cfg: CurriculumSlice):
        self.metrics = metrics
        self.slice = slice_cfg

    def evaluate(self) -> List[GateStatus]:
        return [
            self._coverage_gate(),
            self._abstention_gate(),
            self._velocity_gate(),
            self._caps_gate(),
        ]

    def _coverage_gate(self) -> GateStatus:
        spec = self.slice.gates.coverage
        observed = {
            'ci_lower': self.metrics.coverage_ci_lower,
            'sample_size': self.metrics.coverage_sample_size,
            'attestation_hash': self.metrics.attestation_hash,
        }
        if self.metrics.coverage_ci_lower is None:
            return GateStatus(
                gate='coverage',
                passed=False,
                observed=observed,
                thresholds=spec.to_dict(),
                message="coverage_ci_lower missing",
            )
        ci_lower = self.metrics.coverage_ci_lower
        if ci_lower < spec.ci_lower_min:
            return GateStatus(
                gate='coverage',
                passed=False,
                observed=observed,
                thresholds=spec.to_dict(),
                message=f"ci_lower {ci_lower:.3f} < {spec.ci_lower_min:.3f}",
            )
        sample_size = self.metrics.coverage_sample_size
        if sample_size is None or sample_size < spec.sample_min:
            return GateStatus(
                gate='coverage',
                passed=False,
                observed=observed,
                thresholds=spec.to_dict(),
                message=f"coverage sample {sample_size} < {spec.sample_min}",
            )
        if spec.require_attestation and not self.metrics.attestation_hash:
            return GateStatus(
                gate='coverage',
                passed=False,
                observed=observed,
                thresholds=spec.to_dict(),
                message="attestation hash missing",
            )
        return GateStatus(
            gate='coverage',
            passed=True,
            observed=observed,
            thresholds=spec.to_dict(),
            message=f"ci_lower {ci_lower:.3f} ≥ {spec.ci_lower_min:.3f}, sample {sample_size} ≥ {spec.sample_min}",
        )

    def _abstention_gate(self) -> GateStatus:
        spec = self.slice.gates.abstention
        mass = self.metrics.abstention_mass
        rate = self.metrics.abstention_rate_pct
        observed = {
            'abstention_rate_pct': rate,
            'abstention_mass': mass,
        }
        if rate is None:
            return GateStatus(
                gate='abstention',
                passed=False,
                observed=observed,
                thresholds=spec.to_dict(),
                message="abstention rate missing",
            )
        if rate > spec.max_rate_pct:
            return GateStatus(
                gate='abstention',
                passed=False,
                observed=observed,
                thresholds=spec.to_dict(),
                message=f"abstention rate {rate:.2f}% > {spec.max_rate_pct:.2f}%",
            )
        if mass is None:
            return GateStatus(
                gate='abstention',
                passed=False,
                observed=observed,
                thresholds=spec.to_dict(),
                message="abstention mass indeterminate",
            )
        if mass > spec.max_mass:
            return GateStatus(
                gate='abstention',
                passed=False,
                observed=observed,
                thresholds=spec.to_dict(),
                message=f"abstention mass {mass:.1f} > {spec.max_mass}",
            )
        return GateStatus(
            gate='abstention',
            passed=True,
            observed=observed,
            thresholds=spec.to_dict(),
            message=f"rate {rate:.2f}% ≤ {spec.max_rate_pct:.2f}% and mass {mass:.1f} ≤ {spec.max_mass}",
        )

    def _velocity_gate(self) -> GateStatus:
        spec = self.slice.gates.velocity
        velocity = self.metrics.proof_velocity_pph
        cv = self.metrics.velocity_cv
        observed = {
            'proof_velocity_pph': velocity,
            'velocity_cv': cv,
            'window_minutes': spec.window_minutes,
        }
        if velocity is None:
            return GateStatus(
                gate='velocity',
                passed=False,
                observed=observed,
                thresholds=spec.to_dict(),
                message="proof velocity unavailable",
            )
        if velocity < spec.min_pph:
            return GateStatus(
                gate='velocity',
                passed=False,
                observed=observed,
                thresholds=spec.to_dict(),
                message=f"proof velocity {velocity:.1f} < {spec.min_pph:.1f}",
            )
        if cv is None:
            return GateStatus(
                gate='velocity',
                passed=False,
                observed=observed,
                thresholds=spec.to_dict(),
                message="velocity coefficient of variation missing",
            )
        if cv > spec.stability_cv_max:
            return GateStatus(
                gate='velocity',
                passed=False,
                observed=observed,
                thresholds=spec.to_dict(),
                message=f"velocity CV {cv:.3f} > {spec.stability_cv_max:.3f}",
            )
        return GateStatus(
            gate='velocity',
            passed=True,
            observed=observed,
            thresholds=spec.to_dict(),
            message=f"velocity {velocity:.1f} ≥ {spec.min_pph:.1f} with CV {cv:.3f} ≤ {spec.stability_cv_max:.3f}",
        )

    def _caps_gate(self) -> GateStatus:
        spec = self.slice.gates.caps
        attempt_mass = self.metrics.attempt_mass
        runtime = self.metrics.slice_runtime_minutes
        backlog = self.metrics.backlog_fraction
        total_cap = self.slice.params.get('total_max')
        observed = {
            'attempt_mass': attempt_mass,
            'runtime_minutes': runtime,
            'backlog_fraction': backlog,
            'total_cap': total_cap,
        }
        if attempt_mass is None:
            return GateStatus(
                gate='caps',
                passed=False,
                observed=observed,
                thresholds=spec.to_dict(),
                message="attempt mass missing",
            )
        if attempt_mass < spec.min_attempt_mass:
            return GateStatus(
                gate='caps',
                passed=False,
                observed=observed,
                thresholds=spec.to_dict(),
                message=f"attempt mass {attempt_mass} < {spec.min_attempt_mass}",
            )
        if total_cap is not None and attempt_mass > total_cap:
            return GateStatus(
                gate='caps',
                passed=False,
                observed=observed,
                thresholds=spec.to_dict(),
                message=f"attempt mass {attempt_mass} > total cap {total_cap}",
            )
        if runtime is None:
            return GateStatus(
                gate='caps',
                passed=False,
                observed=observed,
                thresholds=spec.to_dict(),
                message="runtime minutes missing",
            )
        if runtime < spec.min_runtime_minutes:
            return GateStatus(
                gate='caps',
                passed=False,
                observed=observed,
                thresholds=spec.to_dict(),
                message=f"runtime {runtime:.1f}m < {spec.min_runtime_minutes:.1f}m",
            )
        if backlog is None:
            return GateStatus(
                gate='caps',
                passed=False,
                observed=observed,
                thresholds=spec.to_dict(),
                message="backlog fraction missing",
            )
        if backlog > spec.backlog_max:
            return GateStatus(
                gate='caps',
                passed=False,
                observed=observed,
                thresholds=spec.to_dict(),
                message=f"backlog {backlog:.3f} > {spec.backlog_max:.3f}",
            )
        return GateStatus(
            gate='caps',
            passed=True,
            observed=observed,
            thresholds=spec.to_dict(),
            message=(
                f"attempt mass {attempt_mass} ≥ {spec.min_attempt_mass}, "
                f"runtime {runtime:.1f}m ≥ {spec.min_runtime_minutes:.1f}m, "
                f"backlog {backlog:.3f} ≤ {spec.backlog_max:.3f}"
            ),
        )


def load(system_slug: str) -> CurriculumSystem:
    """
    Load curriculum configuration and return a CurriculumSystem dataclass.
    """
    # Go up 2 levels from curriculum/gates.py to project root, then into config/
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "config",
        "curriculum.yaml",
    )

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Curriculum config not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as handle:
        config = yaml.safe_load(handle.read())

    return CurriculumSystem.from_config(system_slug, config)


def should_ratchet(
    metrics: Dict[str, Any],
    system_cfg: CurriculumSystem,
    now: Optional[datetime] = None
) -> GateVerdict:
    """
    Evaluate RFL gates for the active slice and determine whether to advance.
    """
    # Default to a fixed deterministic time if not provided, to ensure reproducible audit logs
    now = now or deterministic_timestamp(0)
    normalized = NormalizedMetrics.from_raw(metrics)
    evaluator = GateEvaluator(normalized, system_cfg.active_slice)
    statuses = evaluator.evaluate()

    audit = {
        'version': system_cfg.version,
        'system': system_cfg.slug,
        'active_slice': system_cfg.active_name,
        'timestamp': now.isoformat(),
        'attestation_hash': normalized.attestation_hash,
        'gates': [status.to_dict() for status in statuses],
    }

    for status in statuses:
        if not status.passed:
            audit['summary'] = status.message
            return GateVerdict(
                advance=False,
                reason=f"{status.gate} gate: {status.message}",
                audit=audit,
            )

    summary = "; ".join(f"{status.gate}: {status.message}" for status in statuses)
    audit['summary'] = summary
    return GateVerdict(
        advance=True,
        reason=summary,
        audit=audit,
    )


def activate_next_slice(system_slug: str, attestation: Optional[Dict[str, Any]] = None) -> CurriculumSystem:
    """
    Advance to the next curriculum slice and update curriculum.yaml.
    """
    # Go up 2 levels from curriculum/gates.py to project root, then into config/
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "config",
        "curriculum.yaml",
    )
    backup_path = f"{config_path}.bak"

    shutil.copy2(config_path, backup_path)

    with open(config_path, 'r', encoding='utf-8') as handle:
        config = yaml.safe_load(handle.read())

    system_cfg = config['systems'][system_slug]
    current_active = system_cfg.get('active')

    # Use deterministic timestamp for reproducibility
    now_iso = deterministic_timestamp_from_content(system_slug, current_active or "").isoformat()
    current_slice: Optional[Dict[str, Any]] = None
    for slice_cfg in system_cfg['slices']:
        if slice_cfg.get('name') == current_active:
            current_slice = slice_cfg
            break

    if current_slice is not None:
        current_slice['completed_at'] = now_iso
        if attestation:
            attestation_record = {
                'sealed_at': now_iso,
                'audit': attestation,
            }
            current_slice.setdefault('attestations', []).append(attestation_record)

    next_slice: Optional[Dict[str, Any]] = None
    for slice_cfg in system_cfg['slices']:
        if slice_cfg.get('completed_at') is None:
            next_slice = slice_cfg
            break

    if next_slice is None:
        raise ValueError(f"No next slice available for system '{system_slug}'")

    system_cfg['active'] = next_slice['name']

    with open(config_path, 'w', encoding='utf-8') as handle:
        yaml.dump(config, handle, default_flow_style=False, sort_keys=False)

    return load(system_slug)
