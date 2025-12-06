# tests/fixtures/first_organism.py
"""
Canonical First Organism attestation fixture.

This module provides a single source of truth for the First Organism
attestation payload used across:
- Unit tests (tests/rfl/test_runner_first_organism.py)
- Integration tests (tests/integration/test_first_organism.py)
- Metrics collectors
- Basis tests in the new repo

The fixture is deterministically generated from MDAP epoch seed and
can be regenerated from integration test artifacts.

Usage:
    from tests.fixtures.first_organism import (
        load_first_organism_attestation,
        make_attested_run_context,
    )

    # Get the canonical fixture
    fixture = load_first_organism_attestation()

    # Convert to AttestedRunContext for RFL runner
    ctx = make_attested_run_context(fixture)
    result = runner.run_with_attestation(ctx)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional

from substrate.bridge.context import AttestedRunContext


# ---------------------------------------------------------------------------
# MDAP Deterministic Constants (aligned with integration conftest)
# ---------------------------------------------------------------------------
MDAP_EPOCH_SEED = 0x4D444150  # "MDAP" as hex seed
FIRST_ORGANISM_NAMESPACE = "first-organism-integration"
FIRST_ORGANISM_VERSION = "1.0.0"


def _deterministic_hash(*parts: Any) -> str:
    """Generate a deterministic 64-char hex hash from parts."""
    content = "|".join(str(p) for p in parts)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Canonical First Organism Attestation Fixture
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FirstOrganismFixture:
    """
    Canonical First Organism attestation fixture.

    This is the single source of truth for First Organism attestation
    data used across all test layers.
    """

    # Core attestation roots (64-char hex strings)
    composite_root: str  # H_t = SHA256(R_t || U_t)
    reasoning_root: str  # R_t
    ui_root: str  # U_t

    # Statement context
    statement_hash: str
    proof_status: str
    block_id: int

    # Slice/policy context
    slice_id: str
    policy_id: str

    # Abstention metrics (the "pain" signal for RFL)
    abstention_rate: float  # fraction [0, 1]
    abstention_mass: float  # absolute count
    attempt_mass: float  # total attempts

    # Abstention breakdown by category
    abstention_breakdown: Dict[str, int] = field(default_factory=dict)

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Determinism metadata
    mdap_seed: int = MDAP_EPOCH_SEED
    version: str = FIRST_ORGANISM_VERSION

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "composite_root": self.composite_root,
            "reasoning_root": self.reasoning_root,
            "ui_root": self.ui_root,
            "statement_hash": self.statement_hash,
            "proof_status": self.proof_status,
            "block_id": self.block_id,
            "slice_id": self.slice_id,
            "policy_id": self.policy_id,
            "abstention_rate": self.abstention_rate,
            "abstention_mass": self.abstention_mass,
            "attempt_mass": self.attempt_mass,
            "abstention_breakdown": dict(self.abstention_breakdown),
            "metadata": dict(self.metadata),
            "mdap_seed": self.mdap_seed,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FirstOrganismFixture":
        """Load from JSON dictionary."""
        block_id = data.get("block_id")
        if block_id is None:
            raise ValueError(
                "block_id is required in fixture data. "
                "All attestation fixtures must have a valid block_id."
            )
        if not isinstance(block_id, int) or block_id < 0:
            raise ValueError(
                f"block_id must be a non-negative integer, got {block_id!r}"
            )
        
        return cls(
            composite_root=data["composite_root"],
            reasoning_root=data["reasoning_root"],
            ui_root=data["ui_root"],
            statement_hash=data["statement_hash"],
            proof_status=data["proof_status"],
            block_id=block_id,
            slice_id=data["slice_id"],
            policy_id=data["policy_id"],
            abstention_rate=data["abstention_rate"],
            abstention_mass=data["abstention_mass"],
            attempt_mass=data["attempt_mass"],
            abstention_breakdown=data.get("abstention_breakdown", {}),
            metadata=data.get("metadata", {}),
            mdap_seed=data.get("mdap_seed", MDAP_EPOCH_SEED),
            version=data.get("version", FIRST_ORGANISM_VERSION),
        )

    def to_json(self, path: Optional[Path] = None) -> str:
        """Serialize to JSON string, optionally writing to file."""
        json_str = json.dumps(self.to_dict(), indent=2, sort_keys=True)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json_str, encoding="utf-8")
        return json_str

    @classmethod
    def from_json(cls, path_or_str: str | Path) -> "FirstOrganismFixture":
        """Load from JSON file or string."""
        if isinstance(path_or_str, Path) or (
            isinstance(path_or_str, str) and Path(path_or_str).exists()
        ):
            data = json.loads(Path(path_or_str).read_text(encoding="utf-8"))
        else:
            data = json.loads(path_or_str)
        return cls.from_dict(data)


# ---------------------------------------------------------------------------
# Canonical Fixture Instance
# ---------------------------------------------------------------------------
# These roots are deterministically derived from MDAP_EPOCH_SEED.
# They are stable across runs and can be verified by:
#   hashlib.sha256(f"{MDAP_EPOCH_SEED}|reasoning".encode()).hexdigest()

_CANONICAL_REASONING_ROOT = _deterministic_hash(MDAP_EPOCH_SEED, "reasoning", "first-organism")
_CANONICAL_UI_ROOT = _deterministic_hash(MDAP_EPOCH_SEED, "ui", "first-organism")
_CANONICAL_COMPOSITE_ROOT = hashlib.sha256(
    (_CANONICAL_REASONING_ROOT + _CANONICAL_UI_ROOT).encode("utf-8")
).hexdigest()
_CANONICAL_STATEMENT_HASH = _deterministic_hash(MDAP_EPOCH_SEED, "statement", "first-organism")

CANONICAL_FIRST_ORGANISM_ATTESTATION = FirstOrganismFixture(
    composite_root=_CANONICAL_COMPOSITE_ROOT,
    reasoning_root=_CANONICAL_REASONING_ROOT,
    ui_root=_CANONICAL_UI_ROOT,
    statement_hash=_CANONICAL_STATEMENT_HASH,
    proof_status="failure",
    block_id=1,
    slice_id="first-organism-test",
    policy_id="policy::first-organism",
    abstention_rate=0.35,
    abstention_mass=7.0,
    attempt_mass=20.0,
    abstention_breakdown={
        "pending_validation": 5,
        "lean_failure": 2,
    },
    metadata={
        "coverage_rate": 0.0,
        "novelty_rate": 0.0,
        "throughput": 0.0,
        "success_rate": 0.0,
        "derive_steps": 1,
        "max_breadth": 8,
        "max_total": 8,
        "origin": "canonical-fixture",
    },
    mdap_seed=MDAP_EPOCH_SEED,
    version=FIRST_ORGANISM_VERSION,
)


# ---------------------------------------------------------------------------
# Fixture Loaders
# ---------------------------------------------------------------------------
def load_first_organism_attestation(
    path: Optional[Path] = None,
    *,
    use_canonical: bool = True,
) -> FirstOrganismFixture:
    """
    Load the First Organism attestation fixture.

    Args:
        path: Optional path to JSON file (overrides canonical)
        use_canonical: If True and no path given, return canonical fixture

    Returns:
        FirstOrganismFixture instance

    Priority:
        1. Explicit path argument
        2. artifacts/first_organism/attestation.json (if exists)
        3. Canonical fixture (if use_canonical=True)
    """
    # Priority 1: explicit path
    if path and Path(path).exists():
        return FirstOrganismFixture.from_json(path)

    # Priority 2: integration test artifact
    artifact_path = Path("artifacts/first_organism/attestation.json")
    if artifact_path.exists():
        try:
            data = json.loads(artifact_path.read_text(encoding="utf-8"))
            # Convert integration artifact format to fixture format
            block_id = data.get("block_id")
            if block_id is None:
                # Default to 1 if missing, but log a warning
                import warnings
                warnings.warn(
                    "Integration artifact missing block_id, defaulting to 1. "
                    "This may indicate an incomplete attestation.",
                    UserWarning
                )
                block_id = 1
            if not isinstance(block_id, int) or block_id < 0:
                raise ValueError(
                    f"block_id in artifact must be a non-negative integer, got {block_id!r}"
                )
            
            return FirstOrganismFixture(
                composite_root=data.get("H_t") or data.get("composite_root", ""),
                reasoning_root=data.get("R_t") or data.get("reasoning_root", ""),
                ui_root=data.get("U_t") or data.get("ui_root", ""),
                statement_hash=data.get("statement_hash", ""),
                proof_status=data.get("proof_status", "failure"),
                block_id=block_id,
                slice_id=data.get("slice_name") or data.get("slice_id", "first-organism-test"),
                policy_id=data.get("policy_id", "policy::first-organism"),
                abstention_rate=data.get("abstention_rate", 0.35),
                abstention_mass=data.get("abstention_mass", 7.0),
                attempt_mass=data.get("attempt_mass", 20.0),
                abstention_breakdown=data.get("abstention_breakdown", {}),
                metadata=data.get("metadata", {}),
                mdap_seed=data.get("mdap_seed", MDAP_EPOCH_SEED),
                version=data.get("version", FIRST_ORGANISM_VERSION),
            )
        except (json.JSONDecodeError, KeyError):
            pass  # Fall through to canonical

    # Priority 3: canonical fixture
    if use_canonical:
        return CANONICAL_FIRST_ORGANISM_ATTESTATION

    raise FileNotFoundError(
        "No First Organism attestation fixture found. "
        "Run integration tests to generate artifacts/first_organism/attestation.json"
    )


def make_attested_run_context(
    fixture: Optional[FirstOrganismFixture] = None,
    *,
    override_metadata: Optional[Dict[str, Any]] = None,
) -> AttestedRunContext:
    """
    Convert a FirstOrganismFixture to AttestedRunContext for RFL runner.

    Args:
        fixture: Fixture to convert (defaults to canonical)
        override_metadata: Optional metadata overrides

    Returns:
        AttestedRunContext ready for run_with_attestation()

    Raises:
        ValueError: If fixture has invalid block_id (None or negative)
    """
    if fixture is None:
        fixture = load_first_organism_attestation()

    # Validate block_id early with clear error
    if fixture.block_id is None:
        raise ValueError(
            f"Fixture block_id cannot be None. "
            f"Fixture must have a valid block_id (non-negative integer). "
            f"Got: {fixture.block_id}"
        )
    if fixture.block_id < 0:
        raise ValueError(
            f"Fixture block_id must be non-negative. "
            f"Got: {fixture.block_id}"
        )

    metadata = dict(fixture.metadata)
    metadata["attempt_mass"] = fixture.attempt_mass
    metadata["abstention_breakdown"] = dict(fixture.abstention_breakdown)

    if override_metadata:
        metadata.update(override_metadata)

    return AttestedRunContext(
        slice_id=fixture.slice_id,
        statement_hash=fixture.statement_hash,
        proof_status=fixture.proof_status,
        block_id=fixture.block_id,
        composite_root=fixture.composite_root,
        reasoning_root=fixture.reasoning_root,
        ui_root=fixture.ui_root,
        abstention_metrics={
            "rate": fixture.abstention_rate,
            "mass": fixture.abstention_mass,
        },
        policy_id=fixture.policy_id,
        metadata=metadata,
    )


def compute_expected_step_id(
    fixture: Optional[FirstOrganismFixture] = None,
    experiment_id: str = "rfl_experiment",
    policy_id: Optional[str] = None,
    resolved_slice_name: Optional[str] = None,
) -> str:
    """
    Compute the expected step_id for a given fixture.

    This allows tests to assert deterministic step_id generation
    without depending on runner internals.

    Args:
        fixture: Fixture to compute step_id for (defaults to canonical)
        experiment_id: RFL experiment ID
        policy_id: Optional policy ID override
        resolved_slice_name: The slice name as resolved by the runner's curriculum.
            If None, uses fixture.slice_id (which may differ from resolved name).

    Returns:
        Expected step_id (64-char hex hash)
    """
    if fixture is None:
        fixture = load_first_organism_attestation()

    policy = policy_id or fixture.policy_id or "default"
    slice_name = resolved_slice_name or fixture.slice_id
    step_material = f"{experiment_id}|{slice_name}|{policy}|{fixture.composite_root}"
    return hashlib.sha256(step_material.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Fixture Regeneration Helper
# ---------------------------------------------------------------------------
def regenerate_canonical_fixture_from_artifact(
    artifact_path: Path = Path("artifacts/first_organism/attestation.json"),
    output_path: Optional[Path] = None,
) -> FirstOrganismFixture:
    """
    Regenerate the canonical fixture from integration test artifacts.

    This should be called after a successful integration test run to
    update the canonical fixture with real attestation data.

    Args:
        artifact_path: Path to integration test artifact
        output_path: Optional path to write regenerated fixture

    Returns:
        Regenerated FirstOrganismFixture
    """
    if not artifact_path.exists():
        raise FileNotFoundError(f"Artifact not found: {artifact_path}")

    fixture = load_first_organism_attestation(artifact_path, use_canonical=False)

    if output_path:
        fixture.to_json(output_path)

    return fixture

