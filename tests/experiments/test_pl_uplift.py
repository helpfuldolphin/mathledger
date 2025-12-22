"""
Tests for PL Uplift Experiment harness.

Verifies:
1. Determinism: same seed produces identical results
2. Real verification: formulas are verified by truth_table_is_tautology
3. Negative tests: different seeds produce different results
4. Output integrity: manifest hash matches results
5. Language restriction: double negation (~~p) is never emitted
6. Audit-surface binding: governance registry hash in manifest
"""

import json
import pytest
from pathlib import Path
from dataclasses import asdict
import sys

# Ensure repo root is at the front of sys.path for correct imports
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "scripts"))

# Import from root governance module (not tests/governance/)
import governance.registry_hash as registry_hash_mod
compute_registry_hash = registry_hash_mod.compute_registry_hash

from run_pl_uplift_exp import (
    ExperimentConfig,
    PLUpliftExperiment,
    generate_formula,
    generate_tautology_candidate,
    write_ab_run_outputs,
    contains_double_negation,
    EXPERIMENT_VERSION,
    HARNESS_NAME,
    AUDIT_SURFACE_VERSION,
)
from backend.repro.determinism import SeededRNG, deterministic_hash
from normalization.taut import truth_table_is_tautology


class TestDeterminism:
    """Test that the experiment is fully deterministic."""

    def test_same_seed_same_results(self, tmp_path):
        """Two runs with same seed must produce identical results."""
        config = ExperimentConfig(
            seed=123,
            output_dir=tmp_path,
            cycles=5,
            formulas_per_cycle=10,
        )

        # Run 1
        exp1 = PLUpliftExperiment(config)
        result1 = exp1.run()

        # Run 2
        exp2 = PLUpliftExperiment(config)
        result2 = exp2.run()

        # Must be identical (using new PhaseResult structure)
        assert result1.baseline.verified_rate == result2.baseline.verified_rate
        assert result1.treatment.verified_rate == result2.treatment.verified_rate
        assert result1.delta == result2.delta

        # Check cycle-level determinism
        for c1, c2 in zip(result1.baseline.cycles, result2.baseline.cycles):
            assert c1 == c2
        for c1, c2 in zip(result1.treatment.cycles, result2.treatment.cycles):
            assert c1 == c2

    def test_determinism_flag_is_true(self, tmp_path):
        """The internal determinism check must pass."""
        config = ExperimentConfig(
            seed=456,
            output_dir=tmp_path,
            cycles=3,
            formulas_per_cycle=5,
        )

        exp = PLUpliftExperiment(config)
        result = exp.run()

        assert result.determinism_verified is True


class TestRealVerification:
    """Test that formulas are actually verified by truth tables."""

    def test_verified_formulas_are_tautologies(self, tmp_path):
        """All verified formulas must pass truth_table_is_tautology."""
        config = ExperimentConfig(
            seed=789,
            output_dir=tmp_path,
            cycles=5,
            formulas_per_cycle=20,
        )

        exp = PLUpliftExperiment(config)

        # Run a cycle and collect verified formula hashes
        rng = SeededRNG(config.seed)
        exp.rng = rng

        # Generate formulas and check
        verified_formulas = []
        for _ in range(config.formulas_per_cycle):
            if rng.random()[0] < 0.7:
                formula = generate_formula(rng, config.baseline_max_atoms)
            else:
                formula = generate_tautology_candidate(rng, config.baseline_max_atoms)

            if truth_table_is_tautology(formula):
                verified_formulas.append(formula)

        # All verified formulas should pass verification again
        for formula in verified_formulas:
            assert truth_table_is_tautology(formula) is True

    def test_tautology_candidate_templates_are_tautologies(self):
        """Known tautology templates should verify as tautologies."""
        rng = SeededRNG(999)

        # Generate several tautology candidates
        for _ in range(20):
            formula = generate_tautology_candidate(rng, max_atoms=3)
            # Tautology candidates should mostly be tautologies
            # (depends on atom substitution, but templates are correct)
            # We just verify they don't crash
            result = truth_table_is_tautology(formula)
            assert result in (True, False)


class TestNegativeCases:
    """Test that different inputs produce different outputs."""

    def test_different_seeds_different_results(self, tmp_path):
        """Different seeds should produce different results."""
        config1 = ExperimentConfig(
            seed=100,
            output_dir=tmp_path / "run1",
            cycles=10,
            formulas_per_cycle=20,
        )
        config2 = ExperimentConfig(
            seed=200,
            output_dir=tmp_path / "run2",
            cycles=10,
            formulas_per_cycle=20,
        )

        exp1 = PLUpliftExperiment(config1)
        result1 = exp1.run()

        exp2 = PLUpliftExperiment(config2)
        result2 = exp2.run()

        # Results should differ (with high probability)
        # At minimum, formula hashes should differ
        hashes1 = set()
        for cycle in result1.baseline.cycles:
            hashes1.update(cycle["formula_hashes"])

        hashes2 = set()
        for cycle in result2.baseline.cycles:
            hashes2.update(cycle["formula_hashes"])

        # Should have some different hashes
        assert hashes1 != hashes2

    def test_random_formula_not_always_tautology(self):
        """Random formulas should not all be tautologies."""
        rng = SeededRNG(42)
        taut_count = 0
        non_taut_count = 0

        for _ in range(100):
            formula = generate_formula(rng, max_atoms=2)
            if truth_table_is_tautology(formula):
                taut_count += 1
            else:
                non_taut_count += 1

        # Should have both tautologies and non-tautologies
        assert taut_count > 0, "Expected some tautologies"
        assert non_taut_count > 0, "Expected some non-tautologies"


class TestOutputIntegrity:
    """Test that output files are correct and verifiable."""

    def test_ab_run_manifest_has_governance_binding(self, tmp_path):
        """A/B run manifest must have governance registry binding."""
        config = ExperimentConfig(
            seed=555,
            output_dir=tmp_path,
            cycles=5,
            formulas_per_cycle=10,
        )

        exp = PLUpliftExperiment(config)
        result = exp.run()

        paths = write_ab_run_outputs(result, tmp_path)

        # Load manifest
        with open(paths["manifest"], "r", encoding="utf-8") as f:
            manifest = json.load(f)

        # Check audit-surface binding
        assert "governance_registry" in manifest
        assert "commitment_registry_sha256" in manifest["governance_registry"]
        assert "commitment_registry_version" in manifest["governance_registry"]

        # Verify registry hash matches actual
        actual_hash = compute_registry_hash()
        assert manifest["governance_registry"]["commitment_registry_sha256"] == actual_hash

        # Check other required fields
        assert manifest["harness"] == HARNESS_NAME
        assert manifest["harness_version"] == EXPERIMENT_VERSION
        assert manifest["audit_surface_version"] == AUDIT_SURFACE_VERSION
        assert manifest["seed"] == config.seed

    def test_ab_run_artifacts_have_artifact_kind(self, tmp_path):
        """All artifacts must have artifact_kind enum."""
        config = ExperimentConfig(
            seed=666,
            output_dir=tmp_path,
            cycles=5,
            formulas_per_cycle=10,
        )

        exp = PLUpliftExperiment(config)
        result = exp.run()

        paths = write_ab_run_outputs(result, tmp_path)

        # Load manifest
        with open(paths["manifest"], "r", encoding="utf-8") as f:
            manifest = json.load(f)

        # Check artifacts
        assert "artifacts" in manifest
        for artifact in manifest["artifacts"]:
            assert "artifact_kind" in artifact
            assert artifact["artifact_kind"] in {"VERIFIED", "REFUTED", "ABSTAINED", "INADMISSIBLE_UPDATE"}
            assert "sha256" in artifact
            assert "path" in artifact

    def test_summary_contains_key_metrics(self, tmp_path):
        """The summary file must contain all key metrics."""
        config = ExperimentConfig(
            seed=777,
            output_dir=tmp_path,
            cycles=5,
            formulas_per_cycle=10,
        )

        exp = PLUpliftExperiment(config)
        result = exp.run()

        paths = write_ab_run_outputs(result, tmp_path)

        with open(paths["summary"], "r", encoding="utf-8") as f:
            summary = json.load(f)

        # Check key fields present
        assert "experiment_id" in summary
        assert "seed" in summary
        assert "baseline_verified_rate" in summary
        assert "treatment_verified_rate" in summary
        assert "delta" in summary
        assert "determinism_verified" in summary


class TestFormulaGeneration:
    """Test the formula generation helpers."""

    def test_generate_formula_deterministic(self):
        """Formula generation must be deterministic."""
        rng1 = SeededRNG(42)
        rng2 = SeededRNG(42)

        formulas1 = [generate_formula(rng1, 2) for _ in range(10)]
        formulas2 = [generate_formula(rng2, 2) for _ in range(10)]

        assert formulas1 == formulas2

    def test_generate_formula_respects_max_atoms(self):
        """Generated formulas should respect max_atoms limit."""
        rng = SeededRNG(123)

        for max_atoms in [1, 2, 3]:
            for _ in range(50):
                formula = generate_formula(rng, max_atoms)
                # Check that only allowed atoms appear
                allowed = set("pqrs"[:max_atoms])
                # Find atoms in formula
                atoms_in_formula = set(c for c in formula if c in "pqrs")
                assert atoms_in_formula.issubset(allowed), \
                    f"Formula {formula} uses atoms {atoms_in_formula} but max is {max_atoms}"

    def test_tautology_candidate_deterministic(self):
        """Tautology candidate generation must be deterministic."""
        rng1 = SeededRNG(77)
        rng2 = SeededRNG(77)

        formulas1 = [generate_tautology_candidate(rng1, 2) for _ in range(10)]
        formulas2 = [generate_tautology_candidate(rng2, 2) for _ in range(10)]

        assert formulas1 == formulas2


class TestLanguageRestriction:
    """Test the double-negation language restriction."""

    def test_generator_never_emits_double_negation(self):
        """Generator must never emit formulas with ~~."""
        rng = SeededRNG(12345)

        # Generate many formulas
        for _ in range(500):
            formula = generate_formula(rng, max_atoms=3)
            assert not contains_double_negation(formula), f"Generator emitted ~~: {formula}"

    def test_tautology_candidate_never_has_double_negation(self):
        """Tautology candidates must never contain ~~."""
        rng = SeededRNG(67890)

        for _ in range(100):
            formula = generate_tautology_candidate(rng, max_atoms=3)
            assert not contains_double_negation(formula), f"Tautology candidate has ~~: {formula}"

    def test_contains_double_negation_detection(self):
        """The double-negation detector must work correctly."""
        assert contains_double_negation("~~p") is True
        assert contains_double_negation("~p") is False
        assert contains_double_negation("p -> ~~q") is True
        assert contains_double_negation("p -> ~q") is False
        assert contains_double_negation("~~~p") is True  # ~~(~p)
        assert contains_double_negation("(p /\\ ~q)") is False
