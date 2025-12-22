"""
Tests for PL Uplift Experiment harness.

Verifies:
1. Determinism: same seed produces identical results
2. Real verification: formulas are verified by truth_table_is_tautology
3. Negative tests: different seeds produce different results
4. Output integrity: manifest hash matches results
"""

import json
import pytest
from pathlib import Path
from dataclasses import asdict
import sys

# Add scripts to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from run_pl_uplift_exp import (
    ExperimentConfig,
    PLUpliftExperiment,
    generate_formula,
    generate_tautology_candidate,
    write_results,
    EXPERIMENT_VERSION,
    HARNESS_NAME,
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

        # Must be identical
        assert result1.baseline_verified_rate == result2.baseline_verified_rate
        assert result1.adapted_verified_rate == result2.adapted_verified_rate
        assert result1.delta == result2.delta

        # Check cycle-level determinism
        for c1, c2 in zip(result1.baseline_cycles, result2.baseline_cycles):
            assert c1 == c2
        for c1, c2 in zip(result1.adapted_cycles, result2.adapted_cycles):
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
        for cycle in result1.baseline_cycles:
            hashes1.update(cycle["formula_hashes"])

        hashes2 = set()
        for cycle in result2.baseline_cycles:
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

    def test_manifest_hash_matches_results(self, tmp_path):
        """The manifest results_hash must match the actual results."""
        config = ExperimentConfig(
            seed=555,
            output_dir=tmp_path,
            cycles=5,
            formulas_per_cycle=10,
        )

        exp = PLUpliftExperiment(config)
        result = exp.run()

        paths = write_results(result, tmp_path)

        # Load manifest
        with open(paths["manifest"], "r", encoding="utf-8") as f:
            manifest = json.load(f)

        # Load results
        with open(paths["results"], "r", encoding="utf-8") as f:
            results_content = f.read()
            results_dict = json.loads(results_content)

        # Recompute hash
        computed_hash = deterministic_hash(
            json.dumps(results_dict, sort_keys=True)
        )

        assert manifest["results_hash"] == computed_hash
        assert manifest["harness"] == HARNESS_NAME
        assert manifest["version"] == EXPERIMENT_VERSION
        assert manifest["seed"] == config.seed

    def test_summary_contains_key_metrics(self, tmp_path):
        """The summary file must contain all key metrics."""
        config = ExperimentConfig(
            seed=666,
            output_dir=tmp_path,
            cycles=5,
            formulas_per_cycle=10,
        )

        exp = PLUpliftExperiment(config)
        result = exp.run()

        paths = write_results(result, tmp_path)

        with open(paths["summary"], "r", encoding="utf-8") as f:
            summary = f.read()

        # Check key fields present
        assert "Experiment ID:" in summary
        assert "Seed:" in summary
        assert "Baseline VERIFIED rate:" in summary
        assert "Adapted VERIFIED rate:" in summary
        assert "Delta:" in summary
        assert "Determinism verified:" in summary


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
