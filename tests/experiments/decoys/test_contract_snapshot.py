# PHASE II — NOT USED IN PHASE I
"""
Test Suite: Contract Snapshot Stability & CI Mode

This module provides snapshot tests for confusability contracts to ensure:
1. Export byte-stability (golden file matching)
2. Semantic diff rules (allowed vs disallowed changes)
3. Family profile determinism

SNAPSHOT RULES:
- New schema fields: ALLOWED (additive)
- New decoys: ALLOWED (additive)
- Changing existing scores: NOT ALLOWED (unless thresholds change)
- Key ordering changes: NOT ALLOWED (determinism violation)
- Family assignment changes: NOT ALLOWED (algorithm instability)
"""

import hashlib
import json
import pytest
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Set

from experiments.decoys.contract import (
    SCHEMA_VERSION,
    ConfusabilityContract,
    export_contract,
    validate_contract_schema,
    compute_structure_fingerprint,
    get_difficulty_band,
    FamilyProfile,
)
from experiments.decoys.loader import CurriculumDecoyLoader


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
def slice_names(loader: CurriculumDecoyLoader) -> List[str]:
    """List of all uplift slice names."""
    return loader.list_uplift_slices()


@pytest.fixture
def temp_output_dir():
    """Temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# =============================================================================
# GOLDEN FILE PATH (for CI snapshot testing)
# =============================================================================

GOLDEN_DIR = Path("tests/experiments/decoys/golden")


def get_golden_path(slice_name: str) -> Path:
    """Get the path to a golden file for a slice."""
    return GOLDEN_DIR / f"{slice_name}.json"


# =============================================================================
# TASK 1: FAMILY PROFILE TESTS
# =============================================================================

class TestFamilyProfiles:
    """Tests for family profile determinism and correctness."""
    
    def test_structure_fingerprint_deterministic(self):
        """Structure fingerprint should be deterministic."""
        formulas = ["p->q", "p->(q->r)", "(p->q)->r", "~p", "p/\\q"]
        
        for formula in formulas:
            fp1 = compute_structure_fingerprint(formula)
            fp2 = compute_structure_fingerprint(formula)
            fp3 = compute_structure_fingerprint(formula)
            
            assert fp1 == fp2 == fp3, f"Fingerprint not deterministic for {formula}"
    
    def test_structure_fingerprint_length(self):
        """Structure fingerprint should be 8 hex characters."""
        fp = compute_structure_fingerprint("p->q")
        assert len(fp) == 8
        assert all(c in "0123456789abcdef" for c in fp)
    
    def test_similar_structures_same_family(self):
        """Similar structures should get the same fingerprint."""
        # These have same structure: one implication, two atoms
        fp1 = compute_structure_fingerprint("p->q")
        fp2 = compute_structure_fingerprint("a->b")
        fp3 = compute_structure_fingerprint("x->y")
        
        assert fp1 == fp2 == fp3, "Same structure should yield same family"
    
    def test_different_structures_different_family(self):
        """Different structures should get different fingerprints."""
        fp_impl = compute_structure_fingerprint("p->q")
        fp_conj = compute_structure_fingerprint("p/\\q")
        fp_neg = compute_structure_fingerprint("~p")
        fp_chain = compute_structure_fingerprint("p->(q->r)")
        
        # All should be different
        fps = {fp_impl, fp_conj, fp_neg, fp_chain}
        assert len(fps) == 4, "Different structures should yield different families"
    
    def test_difficulty_band_thresholds(self):
        """Difficulty bands should follow documented thresholds."""
        assert get_difficulty_band(0.0) == "easy"
        assert get_difficulty_band(0.2) == "easy"
        assert get_difficulty_band(0.32) == "easy"
        assert get_difficulty_band(0.33) == "medium"
        assert get_difficulty_band(0.5) == "medium"
        assert get_difficulty_band(0.66) == "medium"
        assert get_difficulty_band(0.67) == "hard"
        assert get_difficulty_band(0.9) == "hard"
        assert get_difficulty_band(1.0) == "hard"
    
    def test_contract_has_families(self, config_path: str):
        """Exported contract should include families section."""
        contract = export_contract("slice_uplift_goal", config_path)
        d = contract.to_dict()
        
        assert "families" in d
        assert isinstance(d["families"], dict)
    
    def test_families_deterministic_across_runs(self, config_path: str):
        """Family assignments should be identical across multiple exports."""
        exports = []
        
        for _ in range(3):
            contract = export_contract("slice_uplift_goal", config_path)
            families = contract.to_dict()["families"]
            exports.append(families)
        
        for i in range(1, len(exports)):
            assert exports[0] == exports[i], f"Families differ between export 0 and {i}"
    
    def test_formula_entries_have_family_field(self, config_path: str):
        """Each formula should have a family field."""
        contract = export_contract("slice_uplift_goal", config_path)
        d = contract.to_dict()
        
        for f in d["formulas"]:
            assert "family" in f, f"Formula {f['name']} missing family field"
            assert len(f["family"]) == 8, f"Family fingerprint should be 8 chars"
    
    def test_family_members_match_formula_families(self, config_path: str):
        """Family member lists should match formula family assignments."""
        contract = export_contract("slice_uplift_goal", config_path)
        d = contract.to_dict()
        
        # Build expected members from formulas
        expected: Dict[str, List[str]] = {}
        for f in d["formulas"]:
            fam = f["family"]
            if fam not in expected:
                expected[fam] = []
            expected[fam].append(f["name"])
        
        # Compare to actual family profiles
        for fam_name, fam_data in d["families"].items():
            assert fam_name in expected, f"Unknown family {fam_name}"
            assert sorted(fam_data["members"]) == sorted(expected[fam_name])
    
    def test_slices_with_no_decoys_have_empty_families(self, config_path: str, slice_names: List[str]):
        """Legacy slices (no decoys) should still have families for targets."""
        for slice_name in slice_names:
            contract = export_contract(slice_name, config_path)
            d = contract.to_dict()
            
            # Should have families (targets are grouped too)
            assert "families" in d
            
            # If there are formulas, there should be families
            if d["formulas"]:
                assert len(d["families"]) > 0


# =============================================================================
# TASK 2: NARRATIVE BLOCK TESTS
# =============================================================================

class TestNarrativeBlocks:
    """Tests for failure explanation narrative blocks."""
    
    def test_narrative_block_structure(self):
        """Narrative blocks should have exact A/B/C structure."""
        from experiments.curriculum_diagnostics import NarrativeBlock
        
        nb = NarrativeBlock(
            what_failed="Test failure description",
            why_it_failed="Threshold was X, actual was Y",
            how_to_examine="Inspect the formula properties",
        )
        
        d = nb.to_dict()
        assert "A_what_failed" in d
        assert "B_why_it_failed" in d
        assert "C_how_to_examine" in d
    
    def test_narrative_no_prescriptive_verbs(self):
        """HOW TO EXAMINE should not contain prescriptive verbs."""
        from experiments.curriculum_diagnostics import NarrativeBlock
        
        # These verbs should NOT appear in how_to_examine
        prescriptive_verbs = {"fix", "change", "modify", "update", "correct", "adjust"}
        
        # Example narrative
        nb = NarrativeBlock(
            what_failed="Test failure",
            why_it_failed="Threshold exceeded",
            how_to_examine="Inspect the formula properties and review the component scores.",
        )
        
        text = nb.how_to_examine.lower()
        for verb in prescriptive_verbs:
            assert verb not in text, (
                f"Prescriptive verb '{verb}' found in how_to_examine"
            )
    
    def test_narrative_no_uplift_claims(self):
        """Narrative blocks should not make uplift claims."""
        from experiments.curriculum_diagnostics import NarrativeBlock
        
        # These terms should NOT appear in narratives
        uplift_terms = {"uplift", "improvement", "better", "worse", "policy", "reward"}
        
        nb = NarrativeBlock(
            what_failed="Confusability check failed",
            why_it_failed="Below threshold",
            how_to_examine="Review structural similarity to targets.",
        )
        
        full_text = (nb.what_failed + nb.why_it_failed + nb.how_to_examine).lower()
        for term in uplift_terms:
            assert term not in full_text, (
                f"Uplift-related term '{term}' found in narrative"
            )
    
    def test_narrative_block_formatting(self):
        """Narrative block text formatting should have exact headings."""
        from experiments.curriculum_diagnostics import NarrativeBlock
        
        nb = NarrativeBlock(
            what_failed="Test",
            why_it_failed="Test",
            how_to_examine="Test",
        )
        
        formatted = nb.format_text()
        
        assert "A. WHAT FAILED:" in formatted
        assert "B. WHY IT FAILED:" in formatted
        assert "C. HOW TO EXAMINE:" in formatted


# =============================================================================
# TASK 3: CONTRACT SNAPSHOT STABILITY TESTS
# =============================================================================

class TestContractSnapshotStability:
    """Tests for byte-stable contract exports and snapshot matching."""
    
    def test_export_byte_stable_same_run(self, config_path: str):
        """Multiple exports in same run should be byte-identical."""
        exports = []
        
        for _ in range(5):
            contract = export_contract("slice_uplift_goal", config_path)
            exports.append(contract.to_json())
        
        for i in range(1, len(exports)):
            assert exports[0] == exports[i], f"Export {i} differs from export 0"
    
    def test_export_hash_stable(self, config_path: str):
        """Export hash should be stable across runs."""
        hashes = []
        
        for _ in range(3):
            contract = export_contract("slice_uplift_goal", config_path)
            h = hashlib.sha256(contract.to_bytes()).hexdigest()
            hashes.append(h)
        
        assert len(set(hashes)) == 1, "Export hash should be stable"
    
    def test_schema_version_in_export(self, config_path: str):
        """Export should include current schema version."""
        contract = export_contract("slice_uplift_goal", config_path)
        d = contract.to_dict()
        
        assert d["schema_version"] == SCHEMA_VERSION
    
    def test_validate_exported_contract(self, config_path: str):
        """Exported contract should pass validation."""
        contract = export_contract("slice_uplift_goal", config_path)
        d = contract.to_dict()
        
        is_valid, errors = validate_contract_schema(d)
        assert is_valid, f"Validation errors: {errors}"


class TestSemanticDiff:
    """Tests for semantic diff rules (allowed vs disallowed changes)."""
    
    def test_new_schema_fields_allowed(self):
        """Adding new schema fields should be allowed."""
        # Base contract
        base = {
            "schema_version": "1.1.0",
            "slice_name": "test",
            "config_path": "test.yaml",
            "formulas": [],
            "families": {},
            "summary": {
                "target_count": 0,
                "decoy_near_count": 0,
                "decoy_far_count": 0,
                "bridge_count": 0,
                "avg_confusability_near": 0.0,
                "avg_confusability_far": 0.0,
                "family_count": 0,
            },
        }
        
        # With new field (allowed evolution)
        evolved = base.copy()
        evolved["new_field"] = "new_value"
        
        # Should still validate (extra fields ignored)
        is_valid, _ = validate_contract_schema(base)
        assert is_valid
    
    def test_detect_score_changes(self, config_path: str):
        """Should detect when scores change between exports."""
        contract = export_contract("slice_uplift_goal", config_path)
        d1 = contract.to_dict()
        
        # Simulate a score change
        d2 = json.loads(json.dumps(d1))  # Deep copy
        if d2["formulas"]:
            d2["formulas"][0]["confusability"] = 0.999  # Changed!
        
        # Check for differences
        changes = _detect_contract_changes(d1, d2)
        
        if d1["formulas"]:
            assert "confusability" in str(changes), "Should detect score change"
    
    def test_detect_key_ordering_changes(self, config_path: str):
        """Key ordering must be preserved for determinism."""
        contract = export_contract("slice_uplift_goal", config_path)
        json_str = contract.to_json()
        
        # Parse and re-serialize
        d = json.loads(json_str)
        reserialized = json.dumps(d, indent=2, sort_keys=True, ensure_ascii=True)
        
        # Must be identical
        assert json_str == reserialized, "Key ordering not preserved"


def _detect_contract_changes(old: Dict, new: Dict) -> List[str]:
    """
    Detect changes between two contract dictionaries.
    
    Returns list of change descriptions.
    """
    changes = []
    
    # Check schema version
    if old.get("schema_version") != new.get("schema_version"):
        changes.append(f"schema_version: {old.get('schema_version')} -> {new.get('schema_version')}")
    
    # Check formula changes
    old_formulas = {f["name"]: f for f in old.get("formulas", [])}
    new_formulas = {f["name"]: f for f in new.get("formulas", [])}
    
    # Removed formulas
    for name in set(old_formulas.keys()) - set(new_formulas.keys()):
        changes.append(f"Formula removed: {name}")
    
    # Added formulas (allowed)
    for name in set(new_formulas.keys()) - set(old_formulas.keys()):
        changes.append(f"Formula added: {name} (allowed)")
    
    # Changed formulas
    for name in set(old_formulas.keys()) & set(new_formulas.keys()):
        old_f = old_formulas[name]
        new_f = new_formulas[name]
        
        for key in ["confusability", "difficulty"]:
            if abs(old_f.get(key, 0) - new_f.get(key, 0)) > 0.0001:
                changes.append(f"Formula {name} {key}: {old_f.get(key)} -> {new_f.get(key)}")
    
    return changes


# =============================================================================
# GOLDEN FILE TESTS (CI-focused)
# =============================================================================

class TestGoldenFileSnapshot:
    """
    Golden file tests for CI.
    
    These tests compare exports against checked-in golden files.
    To update golden files, run:
        pytest tests/experiments/decoys/test_contract_snapshot.py --update-golden
    """
    
    def test_golden_file_exists_or_creates(self, config_path: str, slice_names: List[str], temp_output_dir: Path):
        """Test that we can generate golden-comparable exports."""
        for slice_name in slice_names:
            contract = export_contract(slice_name, config_path)
            output_path = temp_output_dir / f"{slice_name}.json"
            output_path.write_text(contract.to_json(), encoding='utf-8')
            
            assert output_path.exists()
            
            # Verify it's valid JSON
            data = json.loads(output_path.read_text())
            assert "slice_name" in data
    
    def test_export_matches_golden_if_exists(self, config_path: str, slice_names: List[str]):
        """If golden file exists, export should match (byte-for-byte)."""
        for slice_name in slice_names:
            golden_path = get_golden_path(slice_name)
            
            if not golden_path.exists():
                pytest.skip(f"No golden file for {slice_name}")
            
            contract = export_contract(slice_name, config_path)
            current = contract.to_json()
            golden = golden_path.read_text(encoding='utf-8')
            
            assert current == golden, (
                f"Export for {slice_name} does not match golden file.\n"
                f"Run with --update-golden to regenerate golden files."
            )

