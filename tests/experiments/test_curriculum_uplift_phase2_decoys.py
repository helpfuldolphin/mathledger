"""
PHASE II — NOT USED IN PHASE I
================================================================================
Test Harness: Decoy Formula Validation for Phase II Uplift Experiments
================================================================================

This test suite validates the decoy formulas defined in curriculum_uplift_phase2.yaml:

1. SYNTACTIC VALIDITY: All formulas (targets, decoys, bridges) are parseable
2. HASH CORRECTNESS: Stored hashes match canonical recomputation
3. NON-TARGET VERIFICATION: Decoys do NOT satisfy slice success metrics
4. STRUCTURAL SIMILARITY: Decoys are non-trivially distinguishable from targets

GOVERNANCE:
- Phase II only - these tests and artifacts cannot be used in Phase I experiments
- No decoys are used as training labels in Phase I
- All hashes are deterministically reproducible via the canonical toolchain

CANONICAL PIPELINE:
  hash(formula) = SHA256(DOMAIN_STMT + normalize(formula).encode('ascii'))
================================================================================
"""

import pytest
from pathlib import Path
from typing import Any, Dict, List, Set

import yaml

from normalization.canon import normalize
from backend.crypto.hashing import hash_statement
from derivation.structure import formula_depth, atom_frozenset
from experiments.slice_success_metrics import (
    compute_goal_hit,
    compute_sparse_success,
    compute_chain_success,
    compute_multi_goal_success,
)


# ================================================================================
# FIXTURES
# ================================================================================

@pytest.fixture(scope="module")
def curriculum_config() -> Dict[str, Any]:
    """Load the Phase II curriculum uplift configuration."""
    config_path = Path(__file__).parent.parent.parent / "config" / "curriculum_uplift_phase2.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def uplift_slices(curriculum_config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the uplift slice definitions with formula pools."""
    slices = curriculum_config.get("slices", {})
    # Filter to slices that have formula_pool_entries (the new decoy-enabled slices)
    return {
        name: data for name, data in slices.items()
        if isinstance(data, dict) and "formula_pool_entries" in data
    }


# ================================================================================
# TEST: SYNTACTIC VALIDITY
# ================================================================================

class TestSyntacticValidity:
    """Verify all formulas are syntactically valid and parseable."""

    def test_all_formulas_normalize_without_error(self, uplift_slices: Dict[str, Any]):
        """All formulas in all slices should normalize without raising exceptions."""
        errors = []
        for slice_name, slice_data in uplift_slices.items():
            for entry in slice_data.get("formula_pool_entries", []):
                formula = entry.get("formula", "")
                name = entry.get("name", "unknown")
                try:
                    normalized = normalize(formula)
                    # Should produce non-empty result
                    if not normalized:
                        errors.append(f"{slice_name}/{name}: empty normalization for '{formula}'")
                except Exception as e:
                    errors.append(f"{slice_name}/{name}: normalization error: {e}")
        
        assert not errors, f"Syntactic errors found:\n" + "\n".join(errors)

    def test_all_formulas_have_valid_depth(self, uplift_slices: Dict[str, Any]):
        """All formulas should have computable, non-negative depth."""
        errors = []
        for slice_name, slice_data in uplift_slices.items():
            max_depth = slice_data.get("params", {}).get("depth_max", 10)
            for entry in slice_data.get("formula_pool_entries", []):
                formula = entry.get("formula", "")
                name = entry.get("name", "unknown")
                try:
                    normalized = normalize(formula)
                    depth = formula_depth(normalized)
                    if depth < 0:
                        errors.append(f"{slice_name}/{name}: negative depth {depth}")
                    if depth > max_depth + 2:  # Allow some slack for complex decoys
                        errors.append(f"{slice_name}/{name}: depth {depth} exceeds slice max {max_depth}")
                except Exception as e:
                    errors.append(f"{slice_name}/{name}: depth computation error: {e}")
        
        assert not errors, f"Depth errors found:\n" + "\n".join(errors)

    def test_all_formulas_have_extractable_atoms(self, uplift_slices: Dict[str, Any]):
        """All formulas should have extractable atomic propositions."""
        errors = []
        for slice_name, slice_data in uplift_slices.items():
            max_atoms = slice_data.get("params", {}).get("atoms", 10)
            for entry in slice_data.get("formula_pool_entries", []):
                formula = entry.get("formula", "")
                name = entry.get("name", "unknown")
                try:
                    normalized = normalize(formula)
                    atoms = atom_frozenset(normalized)
                    if not atoms and not formula.startswith("~"):
                        errors.append(f"{slice_name}/{name}: no atoms found in '{formula}'")
                    if len(atoms) > max_atoms + 2:  # Allow some slack
                        errors.append(f"{slice_name}/{name}: {len(atoms)} atoms exceeds slice max {max_atoms}")
                except Exception as e:
                    errors.append(f"{slice_name}/{name}: atom extraction error: {e}")
        
        assert not errors, f"Atom extraction errors found:\n" + "\n".join(errors)


# ================================================================================
# TEST: HASH CORRECTNESS
# ================================================================================

class TestHashCorrectness:
    """Verify stored hashes match canonical recomputation."""

    def test_all_hashes_are_correct(self, uplift_slices: Dict[str, Any]):
        """
        Stored hashes must match recomputation via canonical pipeline.
        
        The canonical pipeline is:
          hash(s) = SHA256(DOMAIN_STMT + normalize(s).encode('ascii'))
        """
        errors = []
        for slice_name, slice_data in uplift_slices.items():
            for entry in slice_data.get("formula_pool_entries", []):
                formula = entry.get("formula", "")
                stored_hash = entry.get("hash", "")
                name = entry.get("name", "unknown")
                
                if not stored_hash:
                    errors.append(f"{slice_name}/{name}: missing hash")
                    continue
                
                computed_hash = hash_statement(formula)
                if computed_hash != stored_hash:
                    errors.append(
                        f"{slice_name}/{name}: hash mismatch\n"
                        f"  formula: {formula}\n"
                        f"  stored:   {stored_hash}\n"
                        f"  computed: {computed_hash}"
                    )
        
        assert not errors, f"Hash mismatches found:\n" + "\n".join(errors)

    def test_normalized_forms_match_stored(self, uplift_slices: Dict[str, Any]):
        """Stored normalized forms must match recomputation."""
        errors = []
        for slice_name, slice_data in uplift_slices.items():
            for entry in slice_data.get("formula_pool_entries", []):
                formula = entry.get("formula", "")
                stored_normalized = entry.get("normalized", "")
                name = entry.get("name", "unknown")
                
                if not stored_normalized:
                    continue  # Optional field
                
                computed_normalized = normalize(formula)
                if computed_normalized != stored_normalized:
                    errors.append(
                        f"{slice_name}/{name}: normalized mismatch\n"
                        f"  formula:  {formula}\n"
                        f"  stored:   {stored_normalized}\n"
                        f"  computed: {computed_normalized}"
                    )
        
        assert not errors, f"Normalized form mismatches found:\n" + "\n".join(errors)

    def test_hashes_are_deterministic(self, uplift_slices: Dict[str, Any]):
        """Hash computation is deterministic across multiple calls."""
        for slice_name, slice_data in uplift_slices.items():
            for entry in slice_data.get("formula_pool_entries", []):
                formula = entry.get("formula", "")
                # Compute hash multiple times
                hashes = [hash_statement(formula) for _ in range(5)]
                assert len(set(hashes)) == 1, f"Non-deterministic hash for {formula}"


# ================================================================================
# TEST: NON-TARGET VERIFICATION (Decoys don't satisfy metrics)
# ================================================================================

class TestDecoyNonTargetVerification:
    """Verify that decoy formulas do NOT satisfy slice success metrics."""

    def test_goal_slice_decoys_not_in_targets(self, uplift_slices: Dict[str, Any]):
        """
        In slice_uplift_goal, decoy hashes must NOT be in the target hash set.
        """
        slice_data = uplift_slices.get("slice_uplift_goal")
        if not slice_data:
            pytest.skip("slice_uplift_goal not defined")
        
        entries = slice_data.get("formula_pool_entries", [])
        target_hashes = {
            e["hash"] for e in entries if e.get("role") == "target"
        }
        
        errors = []
        for entry in entries:
            if entry.get("role") in ("decoy_near", "decoy_far"):
                decoy_hash = entry.get("hash")
                if decoy_hash in target_hashes:
                    errors.append(
                        f"Decoy {entry.get('name')} has hash in target set!\n"
                        f"  hash: {decoy_hash}"
                    )
        
        assert not errors, f"Decoys incorrectly in target set:\n" + "\n".join(errors)

    def test_tree_slice_decoys_not_chain_targets(self, uplift_slices: Dict[str, Any]):
        """
        In slice_uplift_tree, decoys with chain_role='broken' should not be the chain target.
        """
        slice_data = uplift_slices.get("slice_uplift_tree")
        if not slice_data:
            pytest.skip("slice_uplift_tree not defined")
        
        entries = slice_data.get("formula_pool_entries", [])
        
        # Find the chain target (endpoint)
        chain_targets = [
            e for e in entries 
            if e.get("role") == "target" and e.get("chain_role") == "endpoint"
        ]
        target_hashes = {e["hash"] for e in chain_targets}
        
        errors = []
        for entry in entries:
            if entry.get("chain_role") == "broken":
                decoy_hash = entry.get("hash")
                if decoy_hash in target_hashes:
                    errors.append(
                        f"Broken chain decoy {entry.get('name')} matches chain target!\n"
                        f"  hash: {decoy_hash}"
                    )
        
        assert not errors, f"Broken chain decoys incorrectly match targets:\n" + "\n".join(errors)

    def test_dependency_slice_decoys_not_in_required_goals(self, uplift_slices: Dict[str, Any]):
        """
        In slice_uplift_dependency, decoys with goal_set='none' must not be in required_goal_hashes.
        """
        slice_data = uplift_slices.get("slice_uplift_dependency")
        if not slice_data:
            pytest.skip("slice_uplift_dependency not defined")
        
        metric_params = slice_data.get("success_metric", {}).get("parameters", {})
        required_goals = set(metric_params.get("required_goal_hashes", []))
        
        entries = slice_data.get("formula_pool_entries", [])
        
        errors = []
        for entry in entries:
            if entry.get("goal_set") == "none":
                decoy_hash = entry.get("hash")
                if decoy_hash in required_goals:
                    errors.append(
                        f"Non-goal decoy {entry.get('name')} is in required_goal_hashes!\n"
                        f"  hash: {decoy_hash}"
                    )
        
        assert not errors, f"Non-goal decoys incorrectly in required set:\n" + "\n".join(errors)

    def test_compute_goal_hit_rejects_decoys(self, uplift_slices: Dict[str, Any]):
        """
        Using compute_goal_hit with ONLY decoy hashes should NOT achieve success.
        """
        slice_data = uplift_slices.get("slice_uplift_goal")
        if not slice_data:
            pytest.skip("slice_uplift_goal not defined")
        
        entries = slice_data.get("formula_pool_entries", [])
        target_hashes = {
            e["hash"] for e in entries if e.get("role") == "target"
        }
        decoy_hashes = {
            e["hash"] for e in entries if e.get("role") in ("decoy_near", "decoy_far")
        }
        
        # Simulate "verified" only decoys
        verified_statements = [{"hash": h} for h in decoy_hashes]
        
        metric_params = slice_data.get("success_metric", {}).get("parameters", {})
        min_goal_hits = metric_params.get("min_goal_hits", 1)
        min_total = metric_params.get("min_total_verified", 1)
        
        success, hits = compute_goal_hit(
            verified_statements,
            target_hashes,
            min_total,
        )
        
        # Decoy-only verification should yield 0 hits
        assert hits == 0, f"Decoys hit {hits} targets (expected 0)"
        assert not success, "Decoy-only verification should not succeed"

    def test_compute_multi_goal_rejects_non_goals(self, uplift_slices: Dict[str, Any]):
        """
        Using compute_multi_goal_success with non-goal decoys should NOT achieve success.
        """
        slice_data = uplift_slices.get("slice_uplift_dependency")
        if not slice_data:
            pytest.skip("slice_uplift_dependency not defined")
        
        metric_params = slice_data.get("success_metric", {}).get("parameters", {})
        required_goals = set(metric_params.get("required_goal_hashes", []))
        
        entries = slice_data.get("formula_pool_entries", [])
        non_goal_hashes = {
            e["hash"] for e in entries if e.get("goal_set") == "none"
        }
        
        success, met_count = compute_multi_goal_success(
            non_goal_hashes,
            required_goals,
        )
        
        # Non-goals should not satisfy required goals
        assert met_count < len(required_goals), (
            f"Non-goal decoys met {met_count}/{len(required_goals)} goals"
        )
        assert not success, "Non-goal decoys should not achieve multi_goal success"


# ================================================================================
# TEST: STRUCTURAL SIMILARITY (Decoys are non-trivially distinguishable)
# ================================================================================

class TestStructuralSimilarity:
    """
    Verify that decoys are distinguishable from targets only by semantic checks,
    not trivial syntax differences.
    """

    def test_near_decoys_share_atoms_with_targets(self, uplift_slices: Dict[str, Any]):
        """
        Near-decoys should share at least some atoms with their corresponding targets.
        """
        for slice_name, slice_data in uplift_slices.items():
            entries = slice_data.get("formula_pool_entries", [])
            
            target_atoms = set()
            for e in entries:
                if e.get("role") == "target":
                    normalized = normalize(e.get("formula", ""))
                    target_atoms.update(atom_frozenset(normalized))
            
            if not target_atoms:
                continue  # Skip slices without targets
            
            for entry in entries:
                if entry.get("role") == "decoy_near":
                    formula = entry.get("formula", "")
                    normalized = normalize(formula)
                    decoy_atoms = atom_frozenset(normalized)
                    
                    # Near-decoys should share at least one atom
                    shared = decoy_atoms & target_atoms
                    assert shared, (
                        f"{slice_name}/{entry.get('name')}: near-decoy shares no atoms with targets\n"
                        f"  decoy atoms: {decoy_atoms}\n"
                        f"  target atoms: {target_atoms}"
                    )

    def test_near_decoys_similar_depth_to_targets(self, uplift_slices: Dict[str, Any]):
        """
        Near-decoys should have similar depth (within 2) to targets.
        """
        for slice_name, slice_data in uplift_slices.items():
            entries = slice_data.get("formula_pool_entries", [])
            
            target_depths = []
            for e in entries:
                if e.get("role") == "target":
                    normalized = normalize(e.get("formula", ""))
                    target_depths.append(formula_depth(normalized))
            
            if not target_depths:
                continue
            
            min_target_depth = min(target_depths)
            max_target_depth = max(target_depths)
            
            for entry in entries:
                if entry.get("role") == "decoy_near":
                    formula = entry.get("formula", "")
                    normalized = normalize(formula)
                    decoy_depth = formula_depth(normalized)
                    
                    # Near-decoys should be within 2 levels of target depth range
                    assert min_target_depth - 2 <= decoy_depth <= max_target_depth + 2, (
                        f"{slice_name}/{entry.get('name')}: depth {decoy_depth} "
                        f"too far from target range [{min_target_depth}, {max_target_depth}]"
                    )

    def test_decoys_not_trivial_tautologies(self, uplift_slices: Dict[str, Any]):
        """
        Decoys should not be trivial tautologies like 'p -> p' or 'p \\/ ~p'.
        (These are too easy to distinguish from meaningful targets.)
        """
        trivial_patterns = {
            "p->p",
            "q->q",
            "a->a",
            "p\\/~p",
            "~p\\/p",
            "~~p->p",  # This one IS allowed as far_decoy
        }
        
        errors = []
        for slice_name, slice_data in uplift_slices.items():
            entries = slice_data.get("formula_pool_entries", [])
            
            for entry in entries:
                if entry.get("role") in ("decoy_near", "decoy_far"):
                    formula = entry.get("formula", "")
                    normalized = normalize(formula)
                    
                    # Check for trivial patterns (excluding allowed far_decoys)
                    if normalized in trivial_patterns and entry.get("role") == "decoy_near":
                        errors.append(
                            f"{slice_name}/{entry.get('name')}: near-decoy is trivial tautology '{normalized}'"
                        )
        
        assert not errors, f"Trivial decoys found:\n" + "\n".join(errors)


# ================================================================================
# TEST: ROLE COVERAGE
# ================================================================================

class TestRoleCoverage:
    """Verify each slice has the required mix of formula roles."""

    def test_all_slices_have_targets(self, uplift_slices: Dict[str, Any]):
        """Every uplift slice must have at least one target formula."""
        for slice_name, slice_data in uplift_slices.items():
            entries = slice_data.get("formula_pool_entries", [])
            targets = [e for e in entries if e.get("role") == "target"]
            assert targets, f"{slice_name}: no target formulas defined"

    def test_all_slices_have_decoys(self, uplift_slices: Dict[str, Any]):
        """Every uplift slice must have at least 2 decoys (near or far)."""
        for slice_name, slice_data in uplift_slices.items():
            entries = slice_data.get("formula_pool_entries", [])
            decoys = [e for e in entries if e.get("role") in ("decoy_near", "decoy_far")]
            assert len(decoys) >= 2, (
                f"{slice_name}: only {len(decoys)} decoys defined (need at least 2)"
            )

    def test_minimum_decoys_per_target(self, uplift_slices: Dict[str, Any]):
        """Each slice should have at least 2 decoys per target (on average)."""
        for slice_name, slice_data in uplift_slices.items():
            entries = slice_data.get("formula_pool_entries", [])
            targets = [e for e in entries if e.get("role") == "target"]
            decoys = [e for e in entries if e.get("role") in ("decoy_near", "decoy_far")]
            
            if not targets:
                continue
            
            ratio = len(decoys) / len(targets)
            assert ratio >= 1.5, (
                f"{slice_name}: decoy/target ratio {ratio:.1f} below minimum 1.5\n"
                f"  targets: {len(targets)}, decoys: {len(decoys)}"
            )

    def test_role_field_valid_values(self, uplift_slices: Dict[str, Any]):
        """All role fields must use valid role values."""
        valid_roles = {"target", "decoy_near", "decoy_far", "bridge"}
        
        errors = []
        for slice_name, slice_data in uplift_slices.items():
            entries = slice_data.get("formula_pool_entries", [])
            for entry in entries:
                role = entry.get("role", "")
                if role not in valid_roles:
                    errors.append(
                        f"{slice_name}/{entry.get('name')}: invalid role '{role}'"
                    )
        
        assert not errors, f"Invalid roles found:\n" + "\n".join(errors)


# ================================================================================
# TEST: GOVERNANCE COMPLIANCE
# ================================================================================

class TestGovernanceCompliance:
    """Verify Phase II governance rules are followed."""

    def test_config_declares_phase_ii(self, curriculum_config: Dict[str, Any]):
        """Configuration must declare Phase II."""
        phase = curriculum_config.get("phase")
        assert phase == "II", f"Config phase is '{phase}', expected 'II'"

    def test_version_is_2_0(self, curriculum_config: Dict[str, Any]):
        """Configuration version must be 2.0 or higher."""
        version = curriculum_config.get("version", "0.0")
        major = int(str(version).split(".")[0])
        assert major >= 2, f"Config version {version} is too old for Phase II"

    def test_governance_block_exists(self, curriculum_config: Dict[str, Any]):
        """Configuration must have governance block."""
        governance = curriculum_config.get("governance")
        assert governance is not None, "Missing governance block"
        assert "blocked_artifacts" in governance, "Missing blocked_artifacts in governance"
        assert "no_modify" in governance, "Missing no_modify in governance"

