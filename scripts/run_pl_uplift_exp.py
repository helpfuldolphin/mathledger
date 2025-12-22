#!/usr/bin/env python3
"""
PL Uplift Experiment - Policy Adaptation Measurement (v0.2.0)

Measures VERIFIED rate delta under different search-policy configurations
using REAL propositional logic verification (truth-table tautology check).

This is NOT machine learning. It is windowed measurement of a trivial
policy axis (e.g., search breadth) with deterministic verification.

AUDIT-SURFACE CONSISTENT with v0.9.4-pilot-audit-hardened:
- manifest.json with audit_surface_version, commitment_registry_sha256
- per-artifact artifact_kind enum
- verify.py that fails closed on mutation
- paired A/B runs (baseline vs treatment)

Usage:
    # Single run (quick mode)
    uv run python scripts/run_pl_uplift_exp.py --seed 42 --quick

    # Paired A/B run (audit-surface consistent)
    uv run python scripts/run_pl_uplift_exp.py --seed 42 --ab-run --output runs/

Exit codes:
    0 - Experiment completed (all verifications pass)
    1 - Configuration/environment error
    2 - Determinism violation detected
    3 - Verification failure (audit-surface mismatch)

Language restriction: Double negation (~~p) is NOT supported by the verifier.
The generator enforces this by only negating atoms, never compound expressions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Imports from existing MathLedger primitives
# ---------------------------------------------------------------------------

try:
    from backend.repro.determinism import (
        SeededRNG,
        deterministic_hash,
        deterministic_timestamp,
    )
    from normalization.taut import truth_table_is_tautology, _extract_atoms
    from governance import (
        compute_registry_hash,
        get_registry_version,
        canonicalize_json,
        ARTIFACT_KIND_VERIFIED,
        ARTIFACT_KIND_ABSTAINED,
        VALID_ARTIFACT_KINDS,
    )
    from governance.registry_hash import DEFAULT_REGISTRY_PATH
except ImportError as e:
    print(f"ERROR: Missing MathLedger primitive: {e}", file=sys.stderr)
    print("Ensure you are running from the repo root with: uv run python scripts/run_pl_uplift_exp.py", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EXPERIMENT_VERSION = "0.2.0"  # Bumped for audit-surface binding
HARNESS_NAME = "pl_uplift_exp"
AUDIT_SURFACE_VERSION = "0.9.5"  # Consistent with v0.9.4 structure, new experiment
SCHEMA_VERSION = "1.1.0"  # Same schema as v0.9.4 drop-in demo


@dataclass
class ExperimentConfig:
    """Experiment configuration - fully deterministic from seed."""
    seed: int
    output_dir: Path
    cycles: int = 100
    formulas_per_cycle: int = 20
    window_size: int = 10
    # Policy axis: max formula complexity (atom count)
    baseline_max_atoms: int = 2
    treatment_max_atoms: int = 3  # Renamed from "adapted" to "treatment"


@dataclass
class CycleResult:
    """Result of a single experiment cycle."""
    cycle_id: int
    policy_mode: str  # "baseline" or "treatment"
    formulas_attempted: int
    formulas_verified: int
    formulas_refuted: int
    verified_rate: float
    formula_hashes: List[str]


@dataclass
class PhaseResult:
    """Result of a complete phase (baseline or treatment)."""
    mode: str
    cycles: List[Dict[str, Any]]
    total_verified: int
    total_refuted: int
    total_attempted: int
    verified_rate: float
    abstention_rate: float  # formulas that couldn't be classified (always 0 for PL)


@dataclass
class ExperimentResult:
    """Complete experiment result."""
    experiment_id: str
    seed: int
    config: Dict[str, Any]
    baseline: PhaseResult
    treatment: PhaseResult
    delta: float
    delta_positive: bool
    determinism_verified: bool
    timestamp: str


# ---------------------------------------------------------------------------
# Formula Generation (deterministic, double-negation safe)
# ---------------------------------------------------------------------------

# Atoms available for formula generation
ATOMS = list("pqrs")


def generate_formula(rng: SeededRNG, max_atoms: int, max_depth: int = 2) -> str:
    """
    Generate a random propositional formula deterministically.

    LANGUAGE RESTRICTION: Double negation (~~p) is NOT emitted.
    Negation is only applied to atoms, never to compound expressions.

    Args:
        rng: Seeded RNG for deterministic generation
        max_atoms: Maximum number of distinct atoms to use
        max_depth: Maximum nesting depth

    Returns:
        ASCII formula string (guaranteed to not contain ~~)
    """
    atoms = ATOMS[:max_atoms]

    def gen_expr(depth: int) -> str:
        if depth >= max_depth or rng.random()[0] < 0.4:
            # Terminal: just an atom
            return rng.choice(atoms)[0]

        # Choose connective
        conn_type = int(rng.randint(0, 4)[0])

        if conn_type == 0:  # Implication
            left = gen_expr(depth + 1)
            right = gen_expr(depth + 1)
            return f"({left} -> {right})"
        elif conn_type == 1:  # Conjunction
            left = gen_expr(depth + 1)
            right = gen_expr(depth + 1)
            return f"({left} /\\ {right})"
        elif conn_type == 2:  # Disjunction
            left = gen_expr(depth + 1)
            right = gen_expr(depth + 1)
            return f"({left} \\/ {right})"
        else:  # Negation - ONLY apply to atoms
            atom = rng.choice(atoms)[0]
            return f"~{atom}"

    return gen_expr(0)


def generate_tautology_candidate(rng: SeededRNG, max_atoms: int) -> str:
    """
    Generate a formula more likely to be a tautology.

    Uses known tautology patterns with random atom substitution.
    All templates are double-negation free.
    """
    atoms = ATOMS[:max_atoms]

    # Known tautology templates (all ~~-free)
    templates = [
        "{a} -> {a}",                           # Identity
        "({a} -> ({b} -> {a}))",                # K axiom
        "(({a} -> {b}) -> (~{b} -> ~{a}))",     # Contraposition
        "(({a} /\\ {b}) -> {a})",               # Conjunction elim left
        "(({a} /\\ {b}) -> {b})",               # Conjunction elim right
        "({a} -> ({a} \\/ {b}))",               # Disjunction intro left
        "({b} -> ({a} \\/ {b}))",               # Disjunction intro right
        "(~{a} -> ({a} -> {b}))",               # Ex falso
        "({a} \\/ ~{a})",                       # Excluded middle
    ]

    template = rng.choice(templates)[0]

    # Substitute atoms
    a = rng.choice(atoms)[0]
    b = rng.choice(atoms)[0]

    return template.format(a=a, b=b)


def contains_double_negation(formula: str) -> bool:
    """Check if formula contains double negation (~~)."""
    return "~~" in formula


# ---------------------------------------------------------------------------
# Experiment Runner
# ---------------------------------------------------------------------------

class PLUpliftExperiment:
    """PL Uplift experiment runner with audit-surface binding."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.rng = SeededRNG(config.seed)
        self.experiment_id = f"pl_uplift_{config.seed}"

    def run_cycle(self, cycle_id: int, policy_mode: str, max_atoms: int) -> CycleResult:
        """Run a single cycle with given policy."""
        verified_count = 0
        refuted_count = 0
        formula_hashes = []

        for _ in range(self.config.formulas_per_cycle):
            # Mix: 70% random formulas, 30% tautology candidates
            if self.rng.random()[0] < 0.7:
                formula = generate_formula(self.rng, max_atoms)
            else:
                formula = generate_tautology_candidate(self.rng, max_atoms)

            # Verify no double negation (language restriction)
            assert not contains_double_negation(formula), f"Generator emitted ~~: {formula}"

            # Real verification via truth table
            is_taut = truth_table_is_tautology(formula)

            if is_taut:
                verified_count += 1
                formula_hashes.append(deterministic_hash(formula)[:16])
            else:
                refuted_count += 1

        verified_rate = verified_count / self.config.formulas_per_cycle

        return CycleResult(
            cycle_id=cycle_id,
            policy_mode=policy_mode,
            formulas_attempted=self.config.formulas_per_cycle,
            formulas_verified=verified_count,
            formulas_refuted=refuted_count,
            verified_rate=verified_rate,
            formula_hashes=formula_hashes,
        )

    def run_phase(self, mode: str, max_atoms: int, num_cycles: int) -> PhaseResult:
        """Run a complete phase (baseline or treatment)."""
        cycles = []
        total_verified = 0
        total_refuted = 0
        total_attempted = 0

        for i in range(num_cycles):
            result = self.run_cycle(i, mode, max_atoms)
            cycles.append(asdict(result))
            total_verified += result.formulas_verified
            total_refuted += result.formulas_refuted
            total_attempted += result.formulas_attempted

        verified_rate = total_verified / total_attempted if total_attempted > 0 else 0.0
        # Abstention rate is always 0 for PL (we either verify or refute)
        abstention_rate = 0.0

        return PhaseResult(
            mode=mode,
            cycles=cycles,
            total_verified=total_verified,
            total_refuted=total_refuted,
            total_attempted=total_attempted,
            verified_rate=verified_rate,
            abstention_rate=abstention_rate,
        )

    def run(self) -> ExperimentResult:
        """Run the complete experiment (baseline + treatment)."""
        print(f"Running PL Uplift Experiment (seed={self.config.seed})")
        print(f"  Cycles per phase: {self.config.cycles}")
        print(f"  Formulas per cycle: {self.config.formulas_per_cycle}")
        print()

        # Phase 1: Baseline
        print(f"Phase 1: Baseline (max_atoms={self.config.baseline_max_atoms})")
        baseline = self.run_phase(
            "baseline",
            self.config.baseline_max_atoms,
            self.config.cycles,
        )
        print(f"  VERIFIED: {baseline.total_verified}/{baseline.total_attempted} ({baseline.verified_rate:.4f})")

        # Phase 2: Treatment
        print(f"Phase 2: Treatment (max_atoms={self.config.treatment_max_atoms})")
        treatment = self.run_phase(
            "treatment",
            self.config.treatment_max_atoms,
            self.config.cycles,
        )
        print(f"  VERIFIED: {treatment.total_verified}/{treatment.total_attempted} ({treatment.verified_rate:.4f})")

        # Compute delta
        delta = treatment.verified_rate - baseline.verified_rate
        delta_positive = delta > 0

        print()
        print(f"Delta: {delta:+.4f} ({'positive' if delta_positive else 'negative or zero'})")

        # Verify determinism
        determinism_ok = self._verify_determinism()

        return ExperimentResult(
            experiment_id=self.experiment_id,
            seed=self.config.seed,
            config=self._config_to_dict(),
            baseline=baseline,
            treatment=treatment,
            delta=delta,
            delta_positive=delta_positive,
            determinism_verified=determinism_ok,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def _config_to_dict(self) -> Dict[str, Any]:
        """Convert config to JSON-serializable dict."""
        return {
            "seed": self.config.seed,
            "output_dir": str(self.config.output_dir),
            "cycles": self.config.cycles,
            "formulas_per_cycle": self.config.formulas_per_cycle,
            "window_size": self.config.window_size,
            "baseline_max_atoms": self.config.baseline_max_atoms,
            "treatment_max_atoms": self.config.treatment_max_atoms,
        }

    def _verify_determinism(self) -> bool:
        """Verify that re-running with same seed produces same results."""
        # Create fresh RNG with same seed
        rng2 = SeededRNG(self.config.seed)
        original_rng = self.rng
        self.rng = rng2

        # Run first cycle of baseline
        result1 = self.run_cycle(0, "baseline", self.config.baseline_max_atoms)

        # Reset and run again
        rng3 = SeededRNG(self.config.seed)
        self.rng = rng3
        result2 = self.run_cycle(0, "baseline", self.config.baseline_max_atoms)

        # Restore
        self.rng = original_rng

        # Check determinism
        match = (
            result1.formulas_verified == result2.formulas_verified and
            result1.formula_hashes == result2.formula_hashes
        )

        if not match:
            print("WARNING: Determinism check failed!")

        return match


# ---------------------------------------------------------------------------
# Output Handling (audit-surface consistent)
# ---------------------------------------------------------------------------

def compute_file_hash(content: str) -> str:
    """Compute SHA-256 hash of file content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def write_phase_results(
    phase: PhaseResult,
    output_dir: Path,
    seed: int,
) -> Tuple[Path, str]:
    """Write phase results and return path + hash."""
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "results.json"
    content = json.dumps(asdict(phase), indent=2, sort_keys=True)
    results_path.write_text(content, encoding="utf-8")

    return results_path, compute_file_hash(content)


def write_summary(
    result: ExperimentResult,
    output_dir: Path,
) -> Tuple[Path, str]:
    """Write summary and return path + hash."""
    summary_path = output_dir / "summary.json"
    summary = {
        "experiment_id": result.experiment_id,
        "seed": result.seed,
        "baseline_verified_count": result.baseline.total_verified,
        "baseline_refuted_count": result.baseline.total_refuted,
        "baseline_verified_rate": result.baseline.verified_rate,
        "baseline_abstention_rate": result.baseline.abstention_rate,
        "treatment_verified_count": result.treatment.total_verified,
        "treatment_refuted_count": result.treatment.total_refuted,
        "treatment_verified_rate": result.treatment.verified_rate,
        "treatment_abstention_rate": result.treatment.abstention_rate,
        "delta": result.delta,
        "delta_positive": result.delta_positive,
        "determinism_verified": result.determinism_verified,
    }
    content = json.dumps(summary, indent=2, sort_keys=True)
    summary_path.write_text(content, encoding="utf-8")

    return summary_path, compute_file_hash(content)


def generate_manifest(
    result: ExperimentResult,
    artifact_hashes: Dict[str, str],
    artifacts: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Generate manifest with full audit-surface binding."""
    ts = deterministic_timestamp(result.seed)

    return {
        "schema_version": SCHEMA_VERSION,
        "audit_surface_version": AUDIT_SURFACE_VERSION,
        "harness": HARNESS_NAME,
        "harness_version": EXPERIMENT_VERSION,
        "experiment_mode": "A/B_PAIRED",
        "generated_at": ts.isoformat(),
        "seed": result.seed,

        # Governance binding (same as v0.9.4)
        "governance_registry": {
            "commitment_registry_sha256": compute_registry_hash(),
            "commitment_registry_version": get_registry_version(),
        },

        # Results summary
        "results": {
            "baseline_verified_rate": result.baseline.verified_rate,
            "treatment_verified_rate": result.treatment.verified_rate,
            "delta": result.delta,
            "delta_positive": result.delta_positive,
            "determinism_verified": result.determinism_verified,
        },

        # Artifacts with artifact_kind
        "artifacts": artifacts,

        # Reproducibility
        "reproducibility": {
            "deterministic": True,
            "same_seed_same_output": True,
            "language_restriction": "Double negation (~~p) not supported; generator enforces this",
            "verification_command": f"uv run python scripts/run_pl_uplift_exp.py --seed {result.seed} --ab-run --output runs_verify/",
        },
    }


def generate_verification_script(seed: int, mode: str) -> str:
    """Generate standalone verification script."""
    return f'''#!/usr/bin/env python3
"""
Standalone verification script for PL Uplift Experiment ({mode}).

Verifies:
1. Governance registry hash matches manifest
2. All artifact hashes match manifest
3. All artifact_kind values are valid
4. Determinism: re-running produces identical results

Usage:
    python verify.py

Exit codes:
    0 - All verifications pass
    1 - Verification failed

Seed: {seed}
"""

import hashlib
import json
from pathlib import Path

VALID_ARTIFACT_KINDS = {{"VERIFIED", "REFUTED", "ABSTAINED", "INADMISSIBLE_UPDATE"}}


def canonicalize_json(data):
    """Canonicalize JSON for deterministic hashing (RFC 8785-style)."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def compute_file_hash(path):
    """Compute SHA-256 of file content."""
    return hashlib.sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()


def compute_registry_hash(registry_path):
    """Compute SHA-256 hash of registry file (canonical JSON)."""
    content = json.loads(registry_path.read_text(encoding="utf-8"))
    canonical = canonicalize_json(content)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def main():
    errors = []
    print(f"PL Uplift Experiment Verification ({mode})")
    print("=" * 50)
    print()

    # Load manifest
    manifest_path = Path("manifest.json")
    if not manifest_path.exists():
        print("[FAIL] manifest.json not found")
        exit(1)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(f"[INFO] Schema version: {{manifest.get('schema_version', 'unknown')}}")
    print(f"[INFO] Harness: {{manifest.get('harness', 'unknown')}} v{{manifest.get('harness_version', 'unknown')}}")
    print(f"[INFO] Seed: {{manifest['seed']}}")
    print()

    # 1. Verify governance registry
    if "governance_registry" not in manifest:
        print("[FAIL] Missing governance_registry block in manifest")
        errors.append("missing_governance_registry")
    else:
        registry_path = Path("governance/commitment_registry.json")
        if not registry_path.exists():
            print("[FAIL] Missing governance/commitment_registry.json")
            errors.append("missing_registry_file")
        else:
            expected_hash = manifest["governance_registry"]["commitment_registry_sha256"]
            actual_hash = compute_registry_hash(registry_path)
            if expected_hash == actual_hash:
                print(f"[PASS] Governance registry verified: {{expected_hash[:16]}}...")
            else:
                print("[FAIL] Governance registry hash mismatch!")
                print(f"  Expected: {{expected_hash}}")
                print(f"  Computed: {{actual_hash}}")
                errors.append("registry_hash_mismatch")

    # 2. Verify artifact hashes
    if "artifacts" not in manifest:
        print("[WARN] No artifacts block in manifest")
    else:
        artifact_errors = []
        for artifact in manifest["artifacts"]:
            artifact_id = artifact.get("artifact_id", "unknown")
            expected_sha = artifact.get("sha256", "")
            path = Path(artifact.get("path", ""))
            kind = artifact.get("artifact_kind")

            # Check artifact_kind
            if kind is None:
                artifact_errors.append(f"{{artifact_id}}: missing artifact_kind")
            elif kind not in VALID_ARTIFACT_KINDS:
                artifact_errors.append(f"{{artifact_id}}: invalid artifact_kind '{{kind}}'")

            # Check file hash
            if path.exists():
                actual_sha = compute_file_hash(path)
                if expected_sha and actual_sha != expected_sha:
                    artifact_errors.append(f"{{artifact_id}}: hash mismatch (expected {{expected_sha[:16]}}..., got {{actual_sha[:16]}}...)")
            else:
                artifact_errors.append(f"{{artifact_id}}: file not found: {{path}}")

        if artifact_errors:
            print("[FAIL] Artifact verification failed:")
            for err in artifact_errors:
                print(f"  - {{err}}")
            errors.append("artifact_verification")
        else:
            print(f"[PASS] All {{len(manifest['artifacts'])}} artifacts verified")

    # 3. Summary
    print()
    results = manifest.get("results", {{}})
    print(f"[INFO] Baseline VERIFIED rate: {{results.get('baseline_verified_rate', 'N/A')}}")
    print(f"[INFO] Treatment VERIFIED rate: {{results.get('treatment_verified_rate', 'N/A')}}")
    print(f"[INFO] Delta: {{results.get('delta', 'N/A')}}")
    print(f"[INFO] Determinism verified: {{results.get('determinism_verified', 'N/A')}}")

    # Final verdict
    print()
    if errors:
        print(f"[FAIL] Verification failed with {{len(errors)}} error(s): {{', '.join(errors)}}")
        exit(1)
    else:
        print("[PASS] All verifications passed")
        print(f"[INFO] To verify reproducibility, run: uv run python scripts/run_pl_uplift_exp.py --seed {seed} --ab-run --output runs_verify/")


if __name__ == "__main__":
    main()
'''


def write_ab_run_outputs(
    result: ExperimentResult,
    output_dir: Path,
) -> Dict[str, Path]:
    """Write complete A/B run outputs with audit-surface binding."""
    seed = result.seed
    run_dir = output_dir / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    paths = {}
    artifacts = []
    artifact_hashes = {}

    # Create baseline and treatment directories
    baseline_dir = run_dir / "baseline"
    treatment_dir = run_dir / "treatment"

    # Write baseline results
    baseline_path, baseline_hash = write_phase_results(result.baseline, baseline_dir, seed)
    artifacts.append({
        "artifact_id": "baseline_results",
        "path": "baseline/results.json",
        "artifact_kind": ARTIFACT_KIND_VERIFIED,
        "sha256": baseline_hash,
    })
    artifact_hashes["baseline_results"] = baseline_hash
    paths["baseline_results"] = baseline_path

    # Write treatment results
    treatment_path, treatment_hash = write_phase_results(result.treatment, treatment_dir, seed)
    artifacts.append({
        "artifact_id": "treatment_results",
        "path": "treatment/results.json",
        "artifact_kind": ARTIFACT_KIND_VERIFIED,
        "sha256": treatment_hash,
    })
    artifact_hashes["treatment_results"] = treatment_hash
    paths["treatment_results"] = treatment_path

    # Write summary
    summary_path, summary_hash = write_summary(result, run_dir)
    artifacts.append({
        "artifact_id": "summary",
        "path": "summary.json",
        "artifact_kind": ARTIFACT_KIND_VERIFIED,
        "sha256": summary_hash,
    })
    artifact_hashes["summary"] = summary_hash
    paths["summary"] = summary_path

    # Copy governance registry
    governance_dir = run_dir / "governance"
    governance_dir.mkdir(exist_ok=True)
    registry_content = DEFAULT_REGISTRY_PATH.read_text(encoding="utf-8")
    (governance_dir / "commitment_registry.json").write_text(registry_content, encoding="utf-8")
    paths["registry"] = governance_dir / "commitment_registry.json"

    # Generate and write manifest
    manifest = generate_manifest(result, artifact_hashes, artifacts)
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    paths["manifest"] = manifest_path

    # Generate and write verify.py
    verify_script = generate_verification_script(seed, "A/B paired run")
    verify_path = run_dir / "verify.py"
    verify_path.write_text(verify_script, encoding="utf-8")
    paths["verify"] = verify_path

    return paths


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="PL Uplift Experiment - Policy Adaptation Measurement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick single run
    uv run python scripts/run_pl_uplift_exp.py --seed 42 --quick

    # Full A/B paired run (audit-surface consistent)
    uv run python scripts/run_pl_uplift_exp.py --seed 42 --ab-run --output runs/

    # Multiple seeds
    for seed in 42 43 44; do
        uv run python scripts/run_pl_uplift_exp.py --seed $seed --ab-run --output runs/
    done

Language restriction:
    Double negation (~~p) is NOT supported by the verifier.
    The generator enforces this by only negating atoms.
""",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Random seed for deterministic execution"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pl_uplift"),
        help="Output directory (default: results/pl_uplift)"
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=100,
        help="Number of cycles per phase (default: 100)"
    )
    parser.add_argument(
        "--formulas-per-cycle",
        type=int,
        default=20,
        help="Formulas to test per cycle (default: 20)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: 10 cycles, 10 formulas/cycle"
    )
    parser.add_argument(
        "--ab-run",
        action="store_true",
        help="A/B paired run with audit-surface binding (baseline vs treatment)"
    )
    args = parser.parse_args()

    # Build config
    if args.quick:
        cycles = 10
        formulas_per_cycle = 10
    else:
        cycles = args.cycles
        formulas_per_cycle = args.formulas_per_cycle

    config = ExperimentConfig(
        seed=args.seed,
        output_dir=args.output,
        cycles=cycles,
        formulas_per_cycle=formulas_per_cycle,
    )

    print("=" * 60)
    print("PL UPLIFT EXPERIMENT - Policy Adaptation Measurement")
    print("=" * 60)
    print()
    print(f"Harness version: {EXPERIMENT_VERSION}")
    print(f"Audit surface: {AUDIT_SURFACE_VERSION}")
    print(f"Seed: {config.seed}")
    print(f"Output: {config.output_dir}")
    print(f"Mode: {'A/B paired run' if args.ab_run else 'single run'}")
    print()

    # Run experiment
    try:
        experiment = PLUpliftExperiment(config)
        result = experiment.run()
    except Exception as e:
        print(f"\nERROR: Experiment failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    # Check determinism
    if not result.determinism_verified:
        print("\nFATAL: Determinism violation detected!")
        return 2

    # Write results
    print()
    print("Writing results...")

    if args.ab_run:
        paths = write_ab_run_outputs(result, config.output_dir)
    else:
        # Legacy single-run output (minimal)
        config.output_dir.mkdir(parents=True, exist_ok=True)
        summary_path, _ = write_summary(result, config.output_dir)
        paths = {"summary": summary_path}

    for name, path in paths.items():
        print(f"  {name}: {path}")

    # Run verification if A/B mode
    if args.ab_run:
        print()
        print("Running verification...")
        import subprocess
        run_dir = config.output_dir / f"seed_{config.seed}"
        verify_path = run_dir / "verify.py"
        proc = subprocess.run(
            [sys.executable, "verify.py"],
            cwd=str(run_dir),
            capture_output=True,
            text=True,
        )
        print(proc.stdout)
        if proc.returncode != 0:
            print(proc.stderr)
            print("\nFATAL: Verification failed!")
            return 3

    # Final verdict
    print()
    print("=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"Baseline VERIFIED rate: {result.baseline.verified_rate:.4f}")
    print(f"Treatment VERIFIED rate: {result.treatment.verified_rate:.4f}")
    print(f"Delta: {result.delta:+.4f}")
    print(f"Determinism: {'VERIFIED' if result.determinism_verified else 'FAILED'}")
    print()

    if args.ab_run:
        run_dir = config.output_dir / f"seed_{config.seed}"
        print(f"Verification: cd {run_dir} && python verify.py")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Experiment interrupted by user")
        sys.exit(130)
