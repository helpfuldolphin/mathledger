"""
Slice-aware derivation pipeline implementing Algorithm 1 from the MathLedger whitepaper.

Responsibilities:
    * Seed bounded axiom instances (K, S) via substitution.
    * Apply Modus Ponens with breadth/depth caps.
    * Deduplicate using canonical hashes.
    * Verify candidates (tautology recogniser → truth table → optional Lean).
    * Track abstained candidates explicitly for RFL metabolism.
    * Emit structured telemetry for observability.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

from derivation.derive_utils import sha256_statement
from substrate.repro.determinism import deterministic_timestamp_from_content
from curriculum.gates import (
    AbstentionGateSpec,
    CapsGateSpec,
    CoverageGateSpec,
    CurriculumSlice,
    SliceGates,
    VelocityGateSpec,
)
from normalization.canon import normalize, normalize_pretty, canonical_bytes
from derivation.axioms import AxiomInstance, instantiate_axioms
from derivation.bounds import SliceBounds
from derivation.structure import (
    atom_frozenset,
    formula_depth,
    implication_parts,
    is_implication,
)
from .verification import StatementVerifier


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Version for telemetry schema evolution
TELEMETRY_VERSION = "1.1.0"

# Verification methods that indicate abstention (not verified as tautology)
ABSTENTION_METHODS: FrozenSet[str] = frozenset({
    "lean-disabled",
    "lean-timeout",
    "lean-error",
    "truth-table-error",
    "truth-table-non-tautology",
})


# ---------------------------------------------------------------------------
# Canonical Helpers
# ---------------------------------------------------------------------------


def _canonical_parents(parents: Tuple[str, ...]) -> Tuple[str, ...]:
    """Return parent hashes in sorted order for deterministic serialization."""
    return tuple(sorted(parents))


def _canonical_pretty(normalized: str) -> str:
    """
    Generate a canonical pretty-printed form from normalized representation.
    
    This ensures the pretty form is always derivable from the normalized form,
    eliminating any non-determinism from the original input.
    """
    return normalize_pretty(normalized)


def _statement_fingerprint(normalized: str, parents: Tuple[str, ...]) -> str:
    """
    Compute a deterministic fingerprint for a statement including its provenance.
    
    This is used for deduplication and integrity checks.
    """
    canonical_parents = _canonical_parents(parents)
    payload = f"{normalized}|{','.join(canonical_parents)}"
    return hashlib.sha256(payload.encode("ascii")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class StatementRecord:
    """
    Canonical representation of a statement tracked by the pipeline.
    
    All fields are normalized for deterministic serialization:
        - normalized: Canonical ASCII form from normalization.canon
        - hash: SHA256 of normalized form
        - pretty: Canonical pretty form derived from normalized
        - parents: Sorted tuple of parent hashes
    """

    normalized: str
    hash: str
    pretty: str
    rule: str
    is_axiom: bool
    mp_depth: int
    parents: Tuple[str, ...] = field(default_factory=tuple)
    verification_method: str = "unknown"

    def __post_init__(self) -> None:
        # Ensure parents are always sorted for canonical ordering
        if self.parents and self.parents != tuple(sorted(self.parents)):
            object.__setattr__(self, "parents", _canonical_parents(self.parents))

    @property
    def formula_depth(self) -> int:
        return formula_depth(self.normalized)

    @property
    def atoms(self) -> frozenset[str]:
        return atom_frozenset(self.normalized)

    @property
    def fingerprint(self) -> str:
        """Deterministic fingerprint including provenance."""
        return _statement_fingerprint(self.normalized, self.parents)

    def to_canonical_dict(self) -> Dict[str, Any]:
        """Return a canonical dictionary representation for JSON serialization."""
        return {
            "hash": self.hash,
            "normalized": self.normalized,
            "pretty": self.pretty,
            "rule": self.rule,
            "is_axiom": self.is_axiom,
            "mp_depth": self.mp_depth,
            "parents": list(self.parents),  # Already sorted
            "verification_method": self.verification_method,
            "formula_depth": self.formula_depth,
            "atoms": sorted(self.atoms),
            "fingerprint": self.fingerprint,
        }


@dataclass(slots=True)
class PipelineStats:
    """Aggregate statistics for a derivation run."""
    axioms_seeded: int = 0
    mp_rounds: int = 0
    candidates_considered: int = 0
    verified: int = 0
    rejected: int = 0
    
    # Extended telemetry
    axioms_rejected: int = 0
    mp_candidates_rejected: int = 0
    depth_filtered: int = 0
    atom_filtered: int = 0
    duplicate_filtered: int = 0


@dataclass(slots=True)
class DerivationOutcome:
    """Result of a single pipeline step."""
    statements: List[StatementRecord]
    stats: PipelineStats
    abstained_candidates: List[StatementRecord] = field(default_factory=list)

    @property
    def abstention_metrics(self) -> Dict[str, Any]:
        total = self.stats.candidates_considered
        rejected = self.stats.rejected
        rate = (rejected / total) if total > 0 else 0.0
        return {
            "rate": rate,
            "mass": rejected,
            "counts": {
                "verified": self.stats.verified,
                "rejected": rejected,
                "considered": total,
            },
        }


# ---------------------------------------------------------------------------
# Pipeline Core
# ---------------------------------------------------------------------------


class DerivationPipeline:
    """
    Deterministic derivation pipeline orchestrating axiom seeding and MP closure.
    """

    def __init__(
        self, 
        bounds: SliceBounds, 
        verifier: StatementVerifier,
        policy_weights: Optional[Dict[str, float]] = None,
        success_count: Optional[Dict[str, int]] = None,
        stop_after_verified: Optional[int] = None,
        max_candidates: Optional[int] = None,
        mode: str = "baseline",
        cycle_seed: int = 0,
    ) -> None:
        self._bounds = bounds
        self._verifier = verifier
        self._policy_weights = policy_weights or {"len": 0.0, "depth": 0.0, "success": 0.0}
        self._success_count = success_count or {}  # candidate_hash -> success count
        # If set, stop derivation after finding this many verified proofs
        self._stop_after_verified = stop_after_verified
        # Max candidates to consider (makes ordering matter for uplift)
        self._max_candidates = max_candidates
        # Mode: "baseline" uses random ordering, "rfl" uses policy-based ordering
        self._mode = mode
        # Seed for deterministic randomization in baseline mode
        self._cycle_seed = cycle_seed

    def run_step(
        self,
        existing: Iterable[StatementRecord],
        *,
        lean_project_root: Path | None = None,
    ) -> DerivationOutcome:
        """
        Execute a single derivation step against the provided statement basis.
        """
        stats = PipelineStats()
        known_by_norm: Dict[str, StatementRecord] = {}
        known_hashes: Dict[str, StatementRecord] = {}
        abstained_candidates: List[StatementRecord] = []

        for statement in existing:
            known_by_norm[statement.normalized] = statement
            known_hashes[statement.hash] = statement

        new_statements: List[StatementRecord] = []

        # Seed axioms (idempotent thanks to canonical hash tracking).
        for axiom in instantiate_axioms(self._bounds):
            record = self._record_for_axiom(axiom)
            if record.hash in known_hashes:
                stats.duplicate_filtered += 1
                continue
            if record.formula_depth > self._bounds.max_formula_depth:
                stats.depth_filtered += 1
                continue
            if len(record.atoms) > self._bounds.max_atoms:
                stats.atom_filtered += 1
                continue
            outcome = self._verifier.verify(record.normalized)
            if not outcome.verified:
                stats.rejected += 1
                stats.axioms_rejected += 1
                abstained_candidates.append(
                    StatementRecord(
                        normalized=record.normalized,
                        hash=record.hash,
                        pretty=_canonical_pretty(record.normalized),
                        rule=f"axiom:{axiom.name}",
                        is_axiom=True,
                        mp_depth=0,
                        parents=(),
                        verification_method=outcome.method,
                    )
                )
                continue
            seeded = StatementRecord(
                normalized=record.normalized,
                hash=record.hash,
                pretty=_canonical_pretty(record.normalized),
                rule=f"axiom:{axiom.name}",
                is_axiom=True,
                mp_depth=0,
                parents=(),
                verification_method=outcome.method,
            )
            new_statements.append(seeded)
            known_by_norm[seeded.normalized] = seeded
            known_hashes[seeded.hash] = seeded
            stats.axioms_seeded += 1
            stats.verified += 1
            if len(new_statements) >= self._bounds.max_total or len(new_statements) >= self._bounds.max_breadth:
                break

        if len(new_statements) >= self._bounds.max_total or len(new_statements) >= self._bounds.max_breadth:
            new_statements.sort(key=lambda s: s.normalized)
            return DerivationOutcome(new_statements, stats, abstained_candidates)

        total_cap = self._bounds.max_total
        breadth_cap = self._bounds.max_breadth

        mp_round = 0
        while mp_round < self._bounds.max_mp_depth:
            additions_this_round = 0
            implications: List[Tuple[StatementRecord, str, str]] = []

            for record in sorted(known_by_norm.values(), key=lambda s: s.normalized):
                if not is_implication(record.normalized):
                    continue
                antecedent, consequent = implication_parts(record.normalized)
                if antecedent is None or consequent is None:
                    continue
                implications.append((record, antecedent, consequent))

            # Apply candidate ordering based on mode:
            # - baseline: random shuffle (seeded for determinism)
            # - rfl: score-based ordering using policy weights
            # Then apply max_candidates limit to make ordering matter
            import os
            import random
            debug_enabled = os.getenv("DEBUG_CANDIDATE_ORDERING", "").lower() in ("1", "true", "yes")
            
            SUCCESS_FEATURE_SCALE = 5.0  # Scale success feature for more influence
            
            def candidate_score(consequent_norm: str) -> float:
                """Score a candidate based on policy weights and success history."""
                f_len = float(len(consequent_norm))
                f_depth = float(formula_depth(consequent_norm))
                cand_hash = sha256_statement(consequent_norm)
                f_success = float(self._success_count.get(cand_hash, 0)) * SUCCESS_FEATURE_SCALE
                return (
                    self._policy_weights.get("len", 0.0) * f_len +
                    self._policy_weights.get("depth", 0.0) * f_depth +
                    self._policy_weights.get("success", 0.0) * f_success
                )
            
            # Convert to list for manipulation
            implications = list(implications)
            scored = None
            has_policy = False
            
            # Apply mode-based ordering
            if self._mode == "rfl":
                # RFL mode: always sort by policy score (even if weights are 0)
                # This ensures deterministic, score-based ordering vs baseline's random shuffle
                scored = []
                for cand in implications:
                    consequent_norm = cand[2]
                    cand_hash = sha256_statement(consequent_norm)
                    score_val = candidate_score(consequent_norm)
                    len_val = len(consequent_norm)
                    depth_val = formula_depth(consequent_norm)
                    scored.append((cand, score_val, len_val, depth_val, cand_hash))
                scored.sort(key=lambda x: x[1], reverse=True)
                implications = [cand for cand, _, _, _, _ in scored]
                has_policy = True
            else:
                # Baseline mode: deterministic random shuffle
                rng = random.Random(self._cycle_seed + mp_round * 1000)
                rng.shuffle(implications)
            
            # Apply max_candidates limit AFTER ordering - this makes ordering matter!
            if self._max_candidates and len(implications) > self._max_candidates:
                implications = implications[:self._max_candidates]
            
            # Debug logging
            if debug_enabled:
                if has_policy and scored:
                    top_candidates_debug = [
                        {
                            "rank": idx + 1,
                            "formula": cand[2][:60],
                            "len": len_val,
                            "depth": depth_val,
                            "hash": cand_hash[:16],
                            "success_feat": self._success_count.get(cand_hash, 0),
                            "score": score_val,
                            "policy_weights": self._policy_weights.copy(),
                        }
                        for idx, (cand, score_val, len_val, depth_val, cand_hash) in enumerate(scored[:10])
                    ]
                else:
                    top_candidates_debug = [
                        {
                            "rank": idx + 1,
                            "formula": cand[2][:60],
                            "len": len(cand[2]),
                            "depth": formula_depth(cand[2]),
                            "hash": sha256_statement(cand[2])[:16],
                            "success_feat": self._success_count.get(sha256_statement(cand[2]), 0),
                            "score": None,
                        }
                        for idx, cand in enumerate(implications[:10])
                    ]
                import json
                import sys
                debug_log = {
                    "type": "CANDIDATE_ORDERING_DEBUG",
                    "mp_round": mp_round,
                    "mode": self._mode,
                    "max_candidates": self._max_candidates,
                    "has_policy": has_policy,
                    "policy_weights": self._policy_weights.copy() if self._policy_weights else None,
                    "top_candidates": top_candidates_debug,
                }
                print(f"CANDIDATE_ORDERING_DEBUG={json.dumps(debug_log)}", file=sys.stderr, flush=True)

            for imp_record, antecedent_norm, consequent_norm in implications:
                stats.candidates_considered += 1
                if consequent_norm in known_by_norm:
                    stats.duplicate_filtered += 1
                    continue
                antecedent_record = known_by_norm.get(antecedent_norm)
                if antecedent_record is None:
                    continue

                candidate_hash = sha256_statement(consequent_norm)
                if candidate_hash in known_hashes:
                    stats.duplicate_filtered += 1
                    continue

                if formula_depth(consequent_norm) > self._bounds.max_formula_depth:
                    stats.depth_filtered += 1
                    continue
                if len(atom_frozenset(consequent_norm)) > self._bounds.max_atoms:
                    stats.atom_filtered += 1
                    continue

                mp_depth = max(antecedent_record.mp_depth, imp_record.mp_depth) + 1

                # Canonical parent ordering
                canonical_parent_hashes = _canonical_parents((antecedent_record.hash, imp_record.hash))

                outcome = self._verifier.verify(consequent_norm)
                if not outcome.verified:
                    stats.rejected += 1
                    stats.mp_candidates_rejected += 1
                    abstained_candidates.append(
                        StatementRecord(
                            normalized=consequent_norm,
                            hash=candidate_hash,
                            pretty=_canonical_pretty(consequent_norm),
                            rule="mp",
                            is_axiom=False,
                            mp_depth=mp_depth,
                            parents=canonical_parent_hashes,
                            verification_method=outcome.method,
                        )
                    )
                    continue

                derived = StatementRecord(
                    normalized=consequent_norm,
                    hash=candidate_hash,
                    pretty=_canonical_pretty(consequent_norm),
                    rule="mp",
                    is_axiom=False,
                    mp_depth=mp_depth,
                    parents=canonical_parent_hashes,
                    verification_method=outcome.method,
                )

                new_statements.append(derived)
                known_by_norm[derived.normalized] = derived
                known_hashes[derived.hash] = derived
                stats.verified += 1
                additions_this_round += 1

                # Early stopping: if we've found enough verified proofs, stop
                # This makes candidate ordering matter for uplift experiments
                if self._stop_after_verified is not None and stats.verified >= self._stop_after_verified:
                    break

                if len(new_statements) >= total_cap or len(new_statements) >= breadth_cap:
                    break

            if additions_this_round == 0:
                break

            mp_round += 1
            stats.mp_rounds = mp_round

            # Early stopping check at end of round
            if self._stop_after_verified is not None and stats.verified >= self._stop_after_verified:
                break

            if len(new_statements) >= total_cap or len(new_statements) >= breadth_cap:
                break

        new_statements.sort(key=lambda s: s.normalized)
        return DerivationOutcome(new_statements, stats, abstained_candidates)

    @staticmethod
    def _record_for_axiom(instance: AxiomInstance) -> StatementRecord:
        normalized = instance.normalized
        return StatementRecord(
            normalized=normalized,
            hash=sha256_statement(normalized),
            pretty=_canonical_pretty(normalized),
            rule=f"axiom:{instance.name}",
            is_axiom=True,
            mp_depth=0,
            parents=(),
            verification_method="unverified",
        )


# ---------------------------------------------------------------------------
# Derivation Result & Summary (Extended Telemetry)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AbstainedStatement:
    """
    Compact, canonical representation of an abstained statement for telemetry.
    
    All fields are normalized for deterministic JSON serialization.
    """
    hash: str
    normalized: str
    pretty: str
    verification_method: str
    rule: str
    mp_depth: int
    parents: Tuple[str, ...]  # Sorted parent hashes
    fingerprint: str  # Deterministic fingerprint

    @classmethod
    def from_record(cls, record: StatementRecord) -> "AbstainedStatement":
        """Create from a StatementRecord with canonical fields."""
        return cls(
            hash=record.hash,
            normalized=record.normalized,
            pretty=record.pretty,
            verification_method=record.verification_method,
            rule=record.rule,
            mp_depth=record.mp_depth,
            parents=_canonical_parents(record.parents),
            fingerprint=record.fingerprint,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return canonical dictionary for JSON serialization."""
        return {
            "hash": self.hash,
            "normalized": self.normalized,
            "pretty": self.pretty,
            "method": self.verification_method,
            "rule": self.rule,
            "mp_depth": self.mp_depth,
            "parents": list(self.parents),
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True)
class DerivationSummary:
    """
    Structured summary of a derivation run, suitable for logging and assertions.

    This is the machine-readable record that the integration test asserts on.
    Extended with richer telemetry for observability.
    """
    # Core metrics
    slice_name: str
    n_candidates: int
    n_verified: int
    n_abstain: int
    abstained_statements: Tuple[AbstainedStatement, ...]
    
    # Extended telemetry
    telemetry_version: str = TELEMETRY_VERSION
    timestamp_iso: str = ""
    duration_ms: float = 0.0
    bounds_fingerprint: str = ""
    
    # Filtering breakdown
    axioms_seeded: int = 0
    axioms_rejected: int = 0
    mp_candidates_rejected: int = 0
    depth_filtered: int = 0
    atom_filtered: int = 0
    duplicate_filtered: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Return canonical dictionary with all fields."""
        return {
            "telemetry_version": self.telemetry_version,
            "timestamp": self.timestamp_iso,
            "slice": self.slice_name,
            "duration_ms": self.duration_ms,
            "bounds_fingerprint": self.bounds_fingerprint,
            "metrics": {
                "n_candidates": self.n_candidates,
                "n_verified": self.n_verified,
                "n_abstain": self.n_abstain,
                "abstention_rate": (self.n_abstain / self.n_candidates) if self.n_candidates > 0 else 0.0,
            },
            "filtering": {
                "axioms_seeded": self.axioms_seeded,
                "axioms_rejected": self.axioms_rejected,
                "mp_candidates_rejected": self.mp_candidates_rejected,
                "depth_filtered": self.depth_filtered,
                "atom_filtered": self.atom_filtered,
                "duplicate_filtered": self.duplicate_filtered,
            },
            "abstained": [s.to_dict() for s in self.abstained_statements],
        }

    def to_json(self) -> str:
        """Return canonical JSON string with sorted keys."""
        return json.dumps(self.to_dict(), sort_keys=True)

    def to_log_line(self) -> str:
        """Return compact single-line log format."""
        return (
            f"DERIVATION_SUMMARY slice={self.slice_name} "
            f"candidates={self.n_candidates} verified={self.n_verified} abstain={self.n_abstain} "
            f"duration_ms={self.duration_ms:.2f} "
            f"abstain_rate={self.n_abstain / self.n_candidates if self.n_candidates > 0 else 0.0:.4f}"
        )


@dataclass(frozen=True)
class DerivationResult:
    """
    Complete result of a derivation run, including abstention tracking.

    Attributes:
        status: "success" if any verified, "abstain" if only rejections, "failure" otherwise.
        n_candidates: Total candidates considered (MP applications attempted).
        n_verified: Candidates that passed verification.
        n_abstained: Candidates that failed verification (abstentions).
        max_depth: Maximum MP depth reached.
        statements: Verified statements produced.
        abstained_candidates: Statements that failed verification.
        bounds: SliceBounds used for this run.
        stats: Aggregate pipeline statistics.
        summary: Structured summary for telemetry.
    """
    status: str
    n_candidates: int
    n_verified: int
    n_abstained: int
    max_depth: int
    statements: Tuple[StatementRecord, ...]
    abstained_candidates: Tuple[StatementRecord, ...]
    bounds: SliceBounds
    stats: PipelineStats
    summary: DerivationSummary

    @property
    def abstention_metrics(self) -> Dict[str, Any]:
        total = self.stats.candidates_considered
        rejected = self.stats.rejected
        rate = (rejected / total) if total > 0 else 0.0
        return {
            "rate": rate,
            "mass": rejected,
            "counts": {
                "verified": self.stats.verified,
                "rejected": rejected,
                "considered": total,
            },
        }

    @property
    def has_abstention(self) -> bool:
        """True if at least one candidate was abstained."""
        return self.n_abstained > 0

    @property
    def abstention_guaranteed(self) -> bool:
        """True if abstention occurred and all abstained have expected methods."""
        if not self.has_abstention:
            return False
        return all(
            s.verification_method in ABSTENTION_METHODS
            for s in self.abstained_candidates
        )


# ---------------------------------------------------------------------------
# Bounds Extraction & Fingerprinting
# ---------------------------------------------------------------------------


_DEFAULT_BOUNDS = SliceBounds()


def _extract_int(params: Dict[str, Any], keys: Sequence[str], default: int) -> int:
    for key in keys:
        if key in params:
            value = params.get(key)
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    return default


def _extract_float(params: Dict[str, Any], keys: Sequence[str], default: float) -> float:
    for key in keys:
        if key in params:
            value = params.get(key)
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return default


def _bounds_from_slice(slice_cfg: CurriculumSlice) -> SliceBounds:
    # Metadata may also contain overrides; params take precedence.
    params: Dict[str, Any] = {**slice_cfg.metadata, **slice_cfg.params}
    base = _DEFAULT_BOUNDS
    return SliceBounds(
        max_atoms=_extract_int(params, ("atoms", "max_atoms", "atom_cap"), base.max_atoms),
        max_formula_depth=_extract_int(
            params,
            ("depth_max", "formula_depth", "max_formula_depth", "proof_depth"),
            base.max_formula_depth,
        ),
        max_mp_depth=_extract_int(
            params,
            ("mp_depth", "max_mp_depth", "depth_mp", "modus_ponens_depth"),
            base.max_mp_depth,
        ),
        max_breadth=_extract_int(
            params,
            ("breadth_cap", "breadth_max", "max_breadth", "proofs_per_step"),
            base.max_breadth,
        ),
        max_total=_extract_int(
            params,
            ("total_cap", "total_max", "attempt_cap", "proof_cap"),
            base.max_total,
        ),
        max_axiom_instances=_extract_int(
            params,
            ("axiom_inst_cap", "axiom_instances", "axiom_cap"),
            base.max_axiom_instances,
        ),
        max_formula_pool=_extract_int(
            params,
            ("formula_pool_cap", "formula_pool", "candidate_pool"),
            base.max_formula_pool,
        ),
        lean_timeout_s=_extract_float(
            params,
            ("lean_timeout_s", "lean_timeout", "lean_time_limit"),
            base.lean_timeout_s,
        ),
    )


def _bounds_fingerprint(bounds: SliceBounds) -> str:
    """Compute a deterministic fingerprint for SliceBounds."""
    payload = (
        f"atoms={bounds.max_atoms}|depth={bounds.max_formula_depth}|"
        f"mp={bounds.max_mp_depth}|breadth={bounds.max_breadth}|"
        f"total={bounds.max_total}|axioms={bounds.max_axiom_instances}|"
        f"pool={bounds.max_formula_pool}|lean={bounds.lean_timeout_s}"
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# First Organism Abstention Generator (Hardened)
# ---------------------------------------------------------------------------


# Non-tautology formulas guaranteed to cause abstention
# These are simple propositional atoms or implications that are NOT tautologies
_GUARANTEED_NON_TAUTOLOGIES: Tuple[str, ...] = (
    "q",           # Bare atom - never a tautology
    "p",           # Bare atom - never a tautology  
    "r",           # Bare atom - never a tautology
    "(p->q)",      # Contingent implication - not a tautology
    "(q->p)",      # Contingent implication - not a tautology
    "(p->r)",      # Contingent implication - not a tautology
)


def _is_guaranteed_non_tautology(normalized: str) -> bool:
    """
    Check if a normalized formula is guaranteed to NOT be a tautology.
    
    This is used to validate that our abstention generator will always
    produce abstentions.
    """
    # Single atoms are never tautologies
    if len(normalized) == 1 and normalized.isalpha():
        return True
    
    # Simple implications p->q where p != q are contingent (not tautologies)
    # A tautology must be true under ALL valuations
    # p->q is false when p=T, q=F
    if normalized in _GUARANTEED_NON_TAUTOLOGIES:
        return True
    
    return False


def make_first_organism_derivation_slice() -> CurriculumSlice:
    """
    Construct a First Organism slice that reliably produces abstentions.

    This slice is designed as a controlled experimental apparatus for the
    First Organism test - predictable, reproducible, and abstention-positive.

    Design rationale:
        - Minimal search space: 2 atoms (p, q), depth ≤ 2, MP depth = 1
        - Tight caps: breadth_max = 4, total_max = 4
        - Lean disabled: timeout so short (0.001s) it always fails
        - axiom_instances = 0: No axiom seeding, use seeds only
        - Expected behavior: Seeds p and (p → q), MP derives q, which is NOT
          a tautology. Truth-table rejects it, Lean is disabled → abstention.

    The slice maps directly to SliceBounds via _bounds_from_slice() and produces
    at least one abstained statement when run with the seed statements from
    make_first_organism_seed_statements().

    Returns:
        CurriculumSlice configured for First Organism abstention experiment
    """
    gates = SliceGates(
        coverage=CoverageGateSpec(ci_lower_min=0.01, sample_min=1, require_attestation=False),
        abstention=AbstentionGateSpec(max_rate_pct=100.0, max_mass=1000),
        velocity=VelocityGateSpec(min_pph=0.01, stability_cv_max=1.0, window_minutes=1),
        caps=CapsGateSpec(min_attempt_mass=1, min_runtime_minutes=0.001, backlog_max=1.0),
    )
    params = {
        # SliceBounds-compatible parameters
        "atoms": 2,                    # max_atoms: only p, q
        "depth_max": 2,                # max_formula_depth: shallow
        "mp_depth": 1,                 # max_mp_depth: single MP round
        "breadth_max": 20,             # max_breadth: increased for diversity
        "total_max": 50,               # max_total: increased for diversity (was 4)
        "axiom_instances": 0,          # max_axiom_instances: no axioms (use seeds only)
        "formula_pool": 20,            # max_formula_pool: increased for diversity
        "lean_timeout_s": 0.001,       # lean_timeout_s: effectively disabled
    }
    return CurriculumSlice(
        name="first-organism-slice",
        params=params,
        gates=gates,
        metadata={
            "description": "Controlled abstention slice for First Organism test",
            "expected_outcome": "At least one non-tautology derived via MP, abstained by verifier",
            "deterministic": True,
            "search_space": "minimal",
            "abstention_guaranteed": True,
        },
    )


def make_first_organism_seed_statements() -> Tuple[StatementRecord, ...]:
    """
    Create seed statements that trigger MP derivation of non-tautologies.

    Seeds:
        - p: A simple propositional atom (verified as axiom-like)
        - (p → q): An implication

    When MP fires with these seeds:
        - p + (p → q) ⊢ q
        - q is NOT a tautology (truth-table: ¬q is satisfiable)
        - Verifier abstains → recorded in abstained_candidates

    Returns:
        Tuple of StatementRecord for seeding the derivation pipeline
    """
    seeds: List[StatementRecord] = []
    for expr, rule in [("p", "seed:atom"), ("(p->(q))", "seed:implication")]:
        normalized = normalize(expr)
        if not normalized:
            continue
        seeds.append(
            StatementRecord(
                normalized=normalized,
                hash=sha256_statement(normalized),
                pretty=_canonical_pretty(normalized),
                rule=rule,
                is_axiom=False,
                mp_depth=0,
                parents=(),
                verification_method="seed",
            )
        )
    return tuple(seeds)


@dataclass(frozen=True)
class FirstOrganismDerivationConfig:
    """
    Configuration bundle for the First Organism abstention experiment.

    Attributes:
        slice_cfg: CurriculumSlice with parameters tuned for abstention.
        seed_statements: Pre-seeded statements to trigger MP derivation.
        expected_abstention_reason: Why we expect Lean/truth-table to abstain.
        guaranteed_non_tautology: The specific formula that will be abstained.
        abstention_method: Expected verification method for abstention.
    """
    slice_cfg: CurriculumSlice
    seed_statements: Tuple[StatementRecord, ...]
    expected_abstention_reason: str
    guaranteed_non_tautology: str
    abstention_method: str

    def validate(self) -> bool:
        """
        Validate that this config will produce abstention.
        
        Returns True if the config is valid and abstention is guaranteed.
        """
        if not self.seed_statements:
            return False
        if not _is_guaranteed_non_tautology(self.guaranteed_non_tautology):
            return False
        if self.abstention_method not in ABSTENTION_METHODS:
            return False
        return True


def make_first_organism_derivation_config() -> FirstOrganismDerivationConfig:
    """
    Construct a controlled slice configuration for the First Organism abstention experiment.

    GUARANTEE: This configuration ALWAYS produces at least one abstention.

    Design rationale:
        - Minimal atoms (2): keeps search space tiny and deterministic.
        - Depth 2: allows `p -> q` but not deep nesting.
        - Short Lean timeout (0.001s): ensures Lean fallback always times out.
        - Seeds `p` and `p -> q`: MP fires to derive `q`, which is NOT a tautology.
        - `q` fails truth-table check (it's a bare atom, not a tautology).
        - Lean fallback is disabled → returns `lean-disabled`.

    The result is a predictable, reproducible abstention that the integration test
    can assert on.
    
    Abstention is GUARANTEED because:
        1. `q` is a bare propositional atom
        2. Bare atoms are NEVER tautologies (they can be false)
        3. Truth-table evaluation will reject `q`
        4. Lean is disabled, so it returns `lean-disabled`
    """
    gates = SliceGates(
        coverage=CoverageGateSpec(ci_lower_min=0.01, sample_min=1, require_attestation=False),
        abstention=AbstentionGateSpec(max_rate_pct=100.0, max_mass=1000),
        velocity=VelocityGateSpec(min_pph=0.01, stability_cv_max=1.0, window_minutes=1),
        caps=CapsGateSpec(min_attempt_mass=1, min_runtime_minutes=0.001, backlog_max=1.0),
    )
    params = {
        "atoms": 2,
        "depth_max": 2,
        "mp_depth": 1,
        "breadth_max": 4,
        "total_max": 4,
        "axiom_instances": 0,  # CRITICAL: No axiom seeding, use seeds only
        "formula_pool": 8,
        "lean_timeout_s": 0.001,  # Effectively disables Lean (too short)
    }
    slice_cfg = CurriculumSlice(
        name="first-organism-abstention-slice",
        params=params,
        gates=gates,
        metadata={
            "description": "Abstention generator for First Organism integration test",
            "expected_outcome": "At least one non-tautology derived via MP, abstained by verifier",
            "abstention_guaranteed": True,
            "guaranteed_non_tautology": "q",
        },
    )

    # Seed statements: p and (p -> q)
    # When MP fires: p + (p -> q) => q
    # q is NOT a tautology, so it will be rejected by truth-table and Lean fallback.
    seeds: List[StatementRecord] = []
    for expr, rule in [("p", "seed:atom"), ("(p->(q))", "seed:implication")]:
        normalized = normalize(expr)
        if not normalized:
            continue
        seeds.append(
            StatementRecord(
                normalized=normalized,
                hash=sha256_statement(normalized),
                pretty=_canonical_pretty(normalized),
                rule=rule,
                is_axiom=False,
                mp_depth=0,
                parents=(),
                verification_method="seed",
            )
        )

    # The guaranteed non-tautology that MP will derive
    guaranteed_non_tautology = "q"

    return FirstOrganismDerivationConfig(
        slice_cfg=slice_cfg,
        seed_statements=tuple(seeds),
        expected_abstention_reason=(
            "MP derives `q` from `p` and `p -> q`. "
            "`q` is a bare propositional atom, which is NEVER a tautology "
            "(it can be assigned False). Truth-table rejects it, "
            "and Lean fallback is disabled, so verifier abstains with method='lean-disabled'."
        ),
        guaranteed_non_tautology=guaranteed_non_tautology,
        abstention_method="lean-disabled",
    )


# ---------------------------------------------------------------------------
# Test Harness Entry Point
# ---------------------------------------------------------------------------


def run_slice_for_test(
    slice_cfg: CurriculumSlice,
    *,
    limit: int = 1,
    existing: Optional[Sequence[StatementRecord]] = None,
    lean_project_root: Optional[Path] = None,
    emit_log: bool = True,
    policy_weights: Optional[Dict[str, float]] = None,
    success_count: Optional[Dict[str, int]] = None,
    stop_after_verified: Optional[int] = None,
    max_candidates: Optional[int] = None,
    mode: str = "baseline",
    cycle_seed: int = 0,
) -> DerivationResult:
    """
    Execute deterministic derivation rounds for a curriculum slice and return summary metrics.

    This is the primary entry point for integration tests. It:
        1. Extracts SliceBounds from the slice configuration.
        2. Runs the derivation pipeline for `limit` iterations.
        3. Collects all abstained candidates with full canonical metadata.
        4. Emits a structured DERIVATION_SUMMARY log.
        5. Returns a DerivationResult with all telemetry.

    Args:
        slice_cfg: Curriculum slice definition providing parameter bounds.
        limit: Maximum number of pipeline iterations to run (default: 1).
        existing: Optional iterable of pre-existing statements to seed the pipeline.
        lean_project_root: Optional override for the Lean project root.
        emit_log: Whether to emit the DERIVATION_SUMMARY log (default: True).

    Returns:
        DerivationResult capturing aggregate statistics, derived statements,
        abstained candidates, and a structured summary.
    """
    start_time = time.perf_counter()
    # Use deterministic timestamp derived from slice config for hermetic runs
    # This ensures FO cycles and Wide Slice experiments are reproducible
    slice_seed = f"{slice_cfg.name}:{limit}:{len(existing) if existing else 0}"
    timestamp_dt = deterministic_timestamp_from_content(slice_seed)
    timestamp_iso = timestamp_dt.isoformat()
    
    bounds = _bounds_from_slice(slice_cfg)
    bounds_fp = _bounds_fingerprint(bounds)
    verifier = StatementVerifier(bounds, lean_project_root)
    pipeline = DerivationPipeline(
        bounds, verifier, 
        policy_weights=policy_weights, 
        success_count=success_count,
        stop_after_verified=stop_after_verified,
        max_candidates=max_candidates,
        mode=mode,
        cycle_seed=cycle_seed,
    )

    working_set: List[StatementRecord] = list(existing or [])
    collected: List[StatementRecord] = []
    abstained: List[StatementRecord] = []

    aggregate = PipelineStats()
    max_depth = 0

    iterations = max(1, limit)
    for _ in range(iterations):
        outcome = pipeline.run_step(working_set)

        if outcome.statements:
            step_max = max(stmt.mp_depth for stmt in outcome.statements)
            max_depth = max(max_depth, step_max)
            working_set.extend(outcome.statements)
            collected.extend(outcome.statements)

        # Accumulate abstained candidates from this step
        abstained.extend(outcome.abstained_candidates)

        aggregate.axioms_seeded += outcome.stats.axioms_seeded
        aggregate.mp_rounds = max(aggregate.mp_rounds, outcome.stats.mp_rounds)
        aggregate.candidates_considered += outcome.stats.candidates_considered
        aggregate.verified += outcome.stats.verified
        aggregate.rejected += outcome.stats.rejected
        aggregate.axioms_rejected += outcome.stats.axioms_rejected
        aggregate.mp_candidates_rejected += outcome.stats.mp_candidates_rejected
        aggregate.depth_filtered += outcome.stats.depth_filtered
        aggregate.atom_filtered += outcome.stats.atom_filtered
        aggregate.duplicate_filtered += outcome.stats.duplicate_filtered

        if (
            not outcome.statements
            and outcome.stats.candidates_considered == 0
            and outcome.stats.rejected == 0
        ):
            # No further progress possible under current bounds.
            break

        if len(collected) >= bounds.max_total:
            break

    duration_ms = (time.perf_counter() - start_time) * 1000

    # Determine status
    if aggregate.verified > 0:
        status = "success"
    elif aggregate.rejected > 0:
        status = "abstain"
    else:
        status = "failure"

    # Build structured summary with canonical abstained statements
    abstained_statements = tuple(
        AbstainedStatement.from_record(s) for s in abstained
    )
    summary = DerivationSummary(
        slice_name=slice_cfg.name,
        n_candidates=aggregate.candidates_considered,
        n_verified=aggregate.verified,
        n_abstain=aggregate.rejected,
        abstained_statements=abstained_statements,
        telemetry_version=TELEMETRY_VERSION,
        timestamp_iso=timestamp_iso,
        duration_ms=duration_ms,
        bounds_fingerprint=bounds_fp,
        axioms_seeded=aggregate.axioms_seeded,
        axioms_rejected=aggregate.axioms_rejected,
        mp_candidates_rejected=aggregate.mp_candidates_rejected,
        depth_filtered=aggregate.depth_filtered,
        atom_filtered=aggregate.atom_filtered,
        duplicate_filtered=aggregate.duplicate_filtered,
    )

    # Emit structured log for telemetry/debugging
    if emit_log:
        print(f"DERIVATION_SUMMARY={summary.to_json()}", file=sys.stderr, flush=True)
        print(summary.to_log_line(), file=sys.stderr, flush=True)

    return DerivationResult(
        status=status,
        n_candidates=aggregate.candidates_considered,
        n_verified=aggregate.verified,
        n_abstained=aggregate.rejected,
        max_depth=max_depth,
        statements=tuple(collected),
        abstained_candidates=tuple(abstained),
        bounds=bounds,
        stats=aggregate,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# AST Normalization Hooks (Future FOL Support)
# ---------------------------------------------------------------------------


class NormalizationStrategy(Enum):
    """
    Normalization strategy for different logical systems.
    
    This enum allows the pipeline to be extended to support First-Order Logic
    and other logical systems in the future.
    """
    PROPOSITIONAL = "pl"      # Current: propositional logic with ASCII normalization
    FIRST_ORDER = "fol"       # Future: first-order logic with quantifier handling
    HIGHER_ORDER = "hol"      # Future: higher-order logic


@dataclass(frozen=True)
class ASTNormalizationConfig:
    """
    Configuration for AST-based normalization.
    
    This is a placeholder for future FOL slice integration where we need
    to handle quantifiers, variable binding, and more complex normalization.
    """
    strategy: NormalizationStrategy = NormalizationStrategy.PROPOSITIONAL
    
    # Propositional options
    flatten_associative: bool = True
    sort_commutative: bool = True
    deduplicate_idempotent: bool = True
    
    # FOL options (future)
    alpha_normalize: bool = False  # Rename bound variables canonically
    skolemize: bool = False        # Convert to Skolem normal form
    prenex_normal_form: bool = False  # Move quantifiers to front

    def normalize(self, formula: str) -> str:
        """
        Normalize a formula according to this configuration.
        
        Currently delegates to the propositional normalizer.
        Future: dispatch based on strategy.
        """
        if self.strategy == NormalizationStrategy.PROPOSITIONAL:
            return normalize(formula)
        # Future: handle FOL/HOL
        raise NotImplementedError(f"Normalization strategy {self.strategy} not implemented")


# Default configuration for propositional logic
DEFAULT_AST_CONFIG = ASTNormalizationConfig()


# ---------------------------------------------------------------------------
# Budget Summary Helper
# ---------------------------------------------------------------------------


def summarize_budget(stats: PipelineStats) -> Dict[str, Any]:
    """
    Summarize budget-related statistics from a pipeline run.

    Args:
        stats: PipelineStats from a derivation run.

    Returns:
        Dictionary with budget fields:
        - timeout_abstentions: Number of timeout-related abstentions
        - candidates_considered: Total candidates considered
        - verified: Number successfully verified
        - rejected: Number rejected
    """
    return {
        "timeout_abstentions": getattr(stats, "timeout_abstentions", 0),
        "candidates_considered": stats.candidates_considered,
        "verified": stats.verified,
        "rejected": stats.rejected,
        "axioms_seeded": stats.axioms_seeded,
        "mp_rounds": stats.mp_rounds,
        "depth_filtered": stats.depth_filtered,
        "atom_filtered": stats.atom_filtered,
        "duplicate_filtered": stats.duplicate_filtered,
    }


# ---------------------------------------------------------------------------
# Module Exports
# ---------------------------------------------------------------------------


__all__ = [
    # Core types
    "StatementRecord",
    "PipelineStats",
    "DerivationOutcome",
    "DerivationPipeline",
    "DerivationResult",
    "DerivationSummary",
    "AbstainedStatement",
    
    # First Organism
    "FirstOrganismDerivationConfig",
    "make_first_organism_derivation_config",
    "make_first_organism_derivation_slice",
    "make_first_organism_seed_statements",
    
    # Entry point
    "run_slice_for_test",
    
    # Constants
    "TELEMETRY_VERSION",
    "ABSTENTION_METHODS",
    
    # AST normalization (future FOL)
    "NormalizationStrategy",
    "ASTNormalizationConfig",
    "DEFAULT_AST_CONFIG",

    # Budget summary
    "summarize_budget",
]
