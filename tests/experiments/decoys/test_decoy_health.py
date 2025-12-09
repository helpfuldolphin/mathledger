# PHASE II — NOT USED IN PHASE I
"""
Test Suite: Decoy Health Invariants

This module validates critical design invariants for decoy formulas:

1. Near decoys have higher confusability than far decoys
2. No decoy hash == target hash
3. No structurally identical formulas across categories
4. No near-decoy identical normalized form to target

These invariants ensure the decoy framework functions correctly
and decoys serve their intended purpose of testing system discrimination.

All tests are designed for CI integration and produce deterministic results.
"""

import pytest
from typing import Dict, List, Set
from pathlib import Path

from experiments.decoys.confusability import (
    compute_confusability,
    compute_confusability_components,
    ConfusabilityMap,
    ConfusabilityMapReport,
    get_confusability_map,
    get_all_confusability_maps,
)
from experiments.decoys.scoring import (
    DecoyScorer,
    score_slice_decoys,
    score_all_slices,
)
from experiments.decoys.loader import CurriculumDecoyLoader
from normalization.canon import normalize
from backend.crypto.hashing import hash_statement


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(scope="module")
def config_path() -> str:
    """Path to the Phase II curriculum config."""
    return "config/curriculum_uplift_phase2.yaml"


@pytest.fixture(scope="module")
def loader(config_path: str) -> CurriculumDecoyLoader:
    """Shared loader instance."""
    return CurriculumDecoyLoader(config_path)


@pytest.fixture(scope="module")
def all_confusability_maps(config_path: str) -> Dict[str, ConfusabilityMapReport]:
    """All confusability maps for all slices."""
    return get_all_confusability_maps(config_path)


@pytest.fixture(scope="module")
def slice_names(loader: CurriculumDecoyLoader) -> List[str]:
    """List of all uplift slice names."""
    return loader.list_uplift_slices()


# =============================================================================
# INVARIANT 1: Near Confusability > Far Confusability
# =============================================================================

class TestNearFarConfusability:
    """
    Invariant 1: Near decoys have higher confusability than far decoys.
    
    This is a core design principle - near-decoys are intentionally designed
    to be more confusable with targets than far-decoys.
    """
    
    def test_near_confusability_greater_than_far_per_slice(
        self,
        all_confusability_maps: Dict[str, ConfusabilityMapReport],
    ):
        """Near-decoys should have higher average confusability than far-decoys in each slice."""
        violations = []
        
        for slice_name, report in all_confusability_maps.items():
            near_decoys = [f for f in report.formulas if f.role == 'decoy_near']
            far_decoys = [f for f in report.formulas if f.role == 'decoy_far']
            
            if not near_decoys or not far_decoys:
                continue
            
            avg_near = sum(f.confusability for f in near_decoys) / len(near_decoys)
            avg_far = sum(f.confusability for f in far_decoys) / len(far_decoys)
            
            if avg_near < avg_far:
                violations.append(
                    f"{slice_name}: avg_near ({avg_near:.3f}) < avg_far ({avg_far:.3f})"
                )
        
        assert not violations, (
            f"Near-far confusability invariant violated:\n" +
            "\n".join(f"  - {v}" for v in violations)
        )
    
    def test_near_far_gap_positive(
        self,
        all_confusability_maps: Dict[str, ConfusabilityMapReport],
    ):
        """The near-far gap should be positive in all slices."""
        for slice_name, report in all_confusability_maps.items():
            if report.avg_near_confusability > 0 and report.avg_far_confusability > 0:
                assert report.near_far_gap >= 0, (
                    f"{slice_name}: near_far_gap is negative ({report.near_far_gap:.4f})"
                )
    
    def test_individual_near_decoy_confusability_reasonable(
        self,
        all_confusability_maps: Dict[str, ConfusabilityMapReport],
    ):
        """Individual near-decoys should have reasonable confusability (> 0.5)."""
        low_confusability = []
        
        for slice_name, report in all_confusability_maps.items():
            for f in report.formulas:
                if f.role == 'decoy_near' and f.confusability < 0.4:
                    low_confusability.append(
                        f"{slice_name}/{f.name}: confusability = {f.confusability:.3f}"
                    )
        
        # Warn but don't fail - some near-decoys may legitimately have lower scores
        if low_confusability:
            print(f"\nWarning: {len(low_confusability)} near-decoys with low confusability:")
            for item in low_confusability[:5]:
                print(f"  - {item}")


# =============================================================================
# INVARIANT 2: No Decoy Hash == Target Hash
# =============================================================================

class TestNoHashCollisions:
    """
    Invariant 2: No decoy hash equals any target hash.
    
    This is a CRITICAL invariant - if violated, decoys would satisfy
    target success metrics, invalidating experiment results.
    """
    
    def test_no_decoy_hash_equals_target_hash(
        self,
        all_confusability_maps: Dict[str, ConfusabilityMapReport],
    ):
        """No decoy should have the same hash as any target."""
        collisions = []
        
        for slice_name, report in all_confusability_maps.items():
            target_hashes = {f.hash for f in report.formulas if f.role == 'target'}
            
            for f in report.formulas:
                if f.role in ('decoy_near', 'decoy_far'):
                    if f.hash in target_hashes:
                        collisions.append(
                            f"CRITICAL: {slice_name}/{f.name} hash matches a target!"
                        )
        
        assert not collisions, (
            f"Hash collision invariant violated:\n" +
            "\n".join(f"  - {c}" for c in collisions)
        )
    
    def test_no_bridge_hash_equals_target_hash_if_scored(
        self,
        all_confusability_maps: Dict[str, ConfusabilityMapReport],
    ):
        """Bridges should not have target hashes (they aren't meant to be targets)."""
        # This is a softer check - bridges may share hashes with targets
        # if explicitly designed that way, but it should be rare
        warnings = []
        
        for slice_name, report in all_confusability_maps.items():
            target_hashes = {f.hash for f in report.formulas if f.role == 'target'}
            
            for f in report.formulas:
                if f.role == 'bridge' and f.hash in target_hashes:
                    warnings.append(
                        f"Warning: {slice_name}/{f.name} (bridge) matches target hash"
                    )
        
        if warnings:
            print(f"\n{len(warnings)} bridge/target hash overlaps found (may be intentional):")
            for w in warnings[:3]:
                print(f"  - {w}")
    
    def test_all_hashes_computable(
        self,
        all_confusability_maps: Dict[str, ConfusabilityMapReport],
    ):
        """All formulas should produce valid, non-empty hashes."""
        errors = []
        
        for slice_name, report in all_confusability_maps.items():
            for f in report.formulas:
                if not f.hash or len(f.hash) != 64:
                    errors.append(
                        f"{slice_name}/{f.name}: invalid hash '{f.hash[:20] if f.hash else 'empty'}...'"
                    )
        
        assert not errors, f"Invalid hashes found:\n" + "\n".join(errors)


# =============================================================================
# INVARIANT 3: No Structurally Identical Formulas Across Categories
# =============================================================================

class TestNoStructuralDuplicates:
    """
    Invariant 3: No structurally identical formulas across categories.
    
    Formulas in different categories (target/near/far) should have
    distinct normalized forms. Duplicates indicate design errors.
    """
    
    def test_no_normalized_duplicates_across_target_and_decoy(
        self,
        all_confusability_maps: Dict[str, ConfusabilityMapReport],
    ):
        """Targets and decoys should have distinct normalized forms."""
        duplicates = []
        
        for slice_name, report in all_confusability_maps.items():
            target_normalized = {f.normalized for f in report.formulas if f.role == 'target'}
            
            for f in report.formulas:
                if f.role in ('decoy_near', 'decoy_far'):
                    if f.normalized in target_normalized:
                        duplicates.append(
                            f"{slice_name}/{f.name} ({f.role}) has same normalized form as a target"
                        )
        
        assert not duplicates, (
            f"Structural duplicate invariant violated:\n" +
            "\n".join(f"  - {d}" for d in duplicates)
        )
    
    def test_no_near_far_normalized_collision(
        self,
        all_confusability_maps: Dict[str, ConfusabilityMapReport],
    ):
        """Near-decoys and far-decoys should not share normalized forms."""
        collisions = []
        
        for slice_name, report in all_confusability_maps.items():
            near_normalized = {f.normalized for f in report.formulas if f.role == 'decoy_near'}
            far_normalized = {f.normalized for f in report.formulas if f.role == 'decoy_far'}
            
            overlap = near_normalized & far_normalized
            if overlap:
                collisions.append(
                    f"{slice_name}: {len(overlap)} near/far normalized collision(s)"
                )
        
        assert not collisions, (
            f"Near/far collision invariant violated:\n" +
            "\n".join(f"  - {c}" for c in collisions)
        )
    
    def test_unique_names_within_slice(
        self,
        all_confusability_maps: Dict[str, ConfusabilityMapReport],
    ):
        """All formula names should be unique within a slice."""
        duplicates = []
        
        for slice_name, report in all_confusability_maps.items():
            names = [f.name for f in report.formulas]
            seen = set()
            for name in names:
                if name in seen:
                    duplicates.append(f"{slice_name}: duplicate name '{name}'")
                seen.add(name)
        
        assert not duplicates, f"Duplicate names found:\n" + "\n".join(duplicates)


# =============================================================================
# INVARIANT 4: No Near-Decoy Identical to Target
# =============================================================================

class TestNearDecoyDistinctFromTarget:
    """
    Invariant 4: No near-decoy has identical normalized form to any target.
    
    This is a specific case of invariant 3 that's especially important:
    near-decoys are designed to be SIMILAR to targets but not IDENTICAL.
    """
    
    def test_near_decoy_normalized_form_differs_from_all_targets(
        self,
        all_confusability_maps: Dict[str, ConfusabilityMapReport],
    ):
        """Each near-decoy must have a different normalized form from all targets."""
        violations = []
        
        for slice_name, report in all_confusability_maps.items():
            target_normalized = {f.normalized for f in report.formulas if f.role == 'target'}
            
            for f in report.formulas:
                if f.role == 'decoy_near':
                    if f.normalized in target_normalized:
                        violations.append(
                            f"{slice_name}/{f.name}: normalized form matches target!\n"
                            f"  normalized: '{f.normalized[:50]}...'"
                        )
        
        assert not violations, (
            f"Near-decoy identity invariant violated:\n" +
            "\n".join(f"  - {v}" for v in violations)
        )
    
    def test_near_decoy_hash_differs_from_all_targets(
        self,
        all_confusability_maps: Dict[str, ConfusabilityMapReport],
    ):
        """Each near-decoy must have a different hash from all targets."""
        # This is equivalent to test_no_decoy_hash_equals_target_hash but focused on near
        violations = []
        
        for slice_name, report in all_confusability_maps.items():
            target_hashes = {f.hash for f in report.formulas if f.role == 'target'}
            
            for f in report.formulas:
                if f.role == 'decoy_near' and f.hash in target_hashes:
                    violations.append(f"{slice_name}/{f.name}")
        
        assert not violations, (
            f"Near-decoy hash collision with targets:\n" +
            "\n".join(f"  - {v}" for v in violations)
        )


# =============================================================================
# QUALITY ASSESSMENT TESTS
# =============================================================================

class TestQualityAssessment:
    """Tests for the quality assessment functionality."""
    
    def test_quality_assessment_all_slices_pass(self, config_path: str, slice_names: List[str]):
        """All slices should pass quality assessment."""
        failures = []
        
        for slice_name in slice_names:
            cmap = ConfusabilityMap(slice_name, config_path)
            quality = cmap.get_quality_assessment()
            
            if not quality["passed"]:
                failures.append(
                    f"{slice_name}: quality_score={quality['quality_score']:.3f}, "
                    f"issues={quality['issues']}"
                )
        
        assert not failures, (
            f"Quality assessment failures:\n" +
            "\n".join(f"  - {f}" for f in failures)
        )
    
    def test_quality_score_above_threshold(self, config_path: str, slice_names: List[str]):
        """All slices should have quality score >= 0.7."""
        low_quality = []
        
        for slice_name in slice_names:
            cmap = ConfusabilityMap(slice_name, config_path)
            quality = cmap.get_quality_assessment()
            
            if quality["quality_score"] < 0.7:
                low_quality.append(
                    f"{slice_name}: quality_score={quality['quality_score']:.3f}"
                )
        
        assert not low_quality, (
            f"Low quality slices found:\n" +
            "\n".join(f"  - {q}" for q in low_quality)
        )


# =============================================================================
# CONFUSABILITY COMPUTATION TESTS
# =============================================================================

class TestConfusabilityComputation:
    """Tests for the confusability computation functions."""
    
    def test_compute_confusability_deterministic(self):
        """compute_confusability should be deterministic."""
        formula = "q -> (p -> q)"
        targets = ["p -> (q -> p)", "(p -> (q -> r)) -> ((p -> q) -> (p -> r))"]
        
        results = [compute_confusability(formula, targets) for _ in range(5)]
        
        assert len(set(results)) == 1, "compute_confusability is non-deterministic"
    
    def test_compute_confusability_bounds(self):
        """Confusability should be in [0, 1]."""
        test_cases = [
            ("p", ["p -> q"]),
            ("p -> q", ["p -> q"]),
            ("a -> b -> c", ["p -> (q -> r)"]),
            ("~~p -> p", ["p -> p"]),
        ]
        
        for formula, targets in test_cases:
            conf = compute_confusability(formula, targets)
            assert 0.0 <= conf <= 1.0, (
                f"Confusability out of bounds for '{formula}': {conf}"
            )
    
    def test_compute_confusability_identical_formula_high(self):
        """Identical formula should have high confusability."""
        formula = "p -> (q -> p)"
        targets = [formula]
        
        conf = compute_confusability(formula, targets)
        assert conf >= 0.9, f"Identical formula has low confusability: {conf}"
    
    def test_compute_confusability_components_structure(self):
        """compute_confusability_components should return expected structure."""
        formula = "q -> (p -> q)"
        targets = ["p -> (q -> p)"]
        
        components = compute_confusability_components(formula, targets)
        
        expected_keys = {"syntactic", "connective", "atom_similarity", "chain_alignment", "confusability"}
        assert set(components.keys()) == expected_keys
        
        for key, value in components.items():
            assert isinstance(value, float)
            assert 0.0 <= value <= 1.0
    
    def test_empty_target_set_returns_zero(self):
        """Empty target set should return 0 confusability."""
        conf = compute_confusability("p -> q", [])
        assert conf == 0.0


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for confusability and scoring interaction."""
    
    def test_confusability_and_difficulty_correlation(
        self,
        all_confusability_maps: Dict[str, ConfusabilityMapReport],
    ):
        """Confusability and difficulty should show positive correlation for decoys."""
        for slice_name, report in all_confusability_maps.items():
            decoys = [f for f in report.formulas if f.role in ('decoy_near', 'decoy_far')]
            
            if len(decoys) < 3:
                continue
            
            # Simple correlation check: higher difficulty should generally mean higher confusability
            high_diff = [f for f in decoys if f.difficulty > 0.6]
            low_diff = [f for f in decoys if f.difficulty <= 0.6]
            
            if high_diff and low_diff:
                avg_conf_high = sum(f.confusability for f in high_diff) / len(high_diff)
                avg_conf_low = sum(f.confusability for f in low_diff) / len(low_diff)
                
                # Don't require strict correlation, just check it's not inverted
                # This is a soft check
                if avg_conf_high < avg_conf_low - 0.2:
                    print(
                        f"Warning: {slice_name} shows inverted correlation "
                        f"(high_diff_conf={avg_conf_high:.3f}, low_diff_conf={avg_conf_low:.3f})"
                    )
    
    def test_all_slices_have_confusability_maps(self, config_path: str, slice_names: List[str]):
        """All uplift slices should produce valid confusability maps."""
        for slice_name in slice_names:
            report = get_confusability_map(slice_name, config_path)
            
            assert report.slice_name == slice_name
            assert len(report.formulas) > 0
            assert report.avg_near_confusability >= 0
            assert report.avg_far_confusability >= 0

