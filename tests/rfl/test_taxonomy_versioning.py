"""
Tests for Taxonomy Versioning and Diff Tools

Tests the versioning, diff, and CI guard mechanisms for the abstention taxonomy.

PHASE II — VERIFICATION ZONE
Agent B6 (abstention-ops-6)
"""

import pytest
import json
import tempfile
from pathlib import Path
from typing import Dict, Any

from rfl.verification.abstention_taxonomy import AbstentionType
from rfl.verification.abstention_semantics import (
    ABSTENTION_TAXONOMY_VERSION,
    get_taxonomy_version,
    export_semantics,
    SemanticCategory,
)


# ---------------------------------------------------------------------------
# Task 1: Taxonomy Version Tests
# ---------------------------------------------------------------------------

class TestTaxonomyVersion:
    """Tests for taxonomy versioning."""
    
    def test_version_is_defined(self) -> None:
        """ABSTENTION_TAXONOMY_VERSION must be defined."""
        assert ABSTENTION_TAXONOMY_VERSION is not None
        assert isinstance(ABSTENTION_TAXONOMY_VERSION, str)
    
    def test_version_is_semantic(self) -> None:
        """Version must follow semantic versioning (X.Y.Z)."""
        parts = ABSTENTION_TAXONOMY_VERSION.split(".")
        assert len(parts) == 3, f"Version should be X.Y.Z, got {ABSTENTION_TAXONOMY_VERSION}"
        assert all(part.isdigit() for part in parts), "Version parts must be numeric"
    
    def test_get_taxonomy_version_returns_constant(self) -> None:
        """get_taxonomy_version() must return the version constant."""
        assert get_taxonomy_version() == ABSTENTION_TAXONOMY_VERSION
    
    def test_version_appears_in_export(self) -> None:
        """taxonomy_version must appear in exported JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "semantics.json"
            export_semantics(out_path)
            
            with open(out_path) as f:
                data = json.load(f)
            
            assert "taxonomy_version" in data
            assert data["taxonomy_version"] == ABSTENTION_TAXONOMY_VERSION
    
    def test_version_matches_exported_version(self) -> None:
        """Exported version must match get_taxonomy_version()."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "semantics.json"
            export_semantics(out_path)
            
            with open(out_path) as f:
                data = json.load(f)
            
            assert data["taxonomy_version"] == get_taxonomy_version()


# ---------------------------------------------------------------------------
# Task 2: Taxonomy Diff Tests
# ---------------------------------------------------------------------------

class TestTaxonomyDiff:
    """Tests for diff_abstention_taxonomy.py functionality."""
    
    @pytest.fixture
    def base_semantics(self) -> Dict[str, Any]:
        """Base semantics fixture."""
        return {
            "taxonomy_version": "1.0.0",
            "abstention_types": {
                "abstain_timeout": {"category": "timeout_related", "legacy_keys": ["timeout"]},
                "abstain_crash": {"category": "crash_related", "legacy_keys": ["crash"]},
            },
            "categories": {
                "timeout_related": ["abstain_timeout"],
                "crash_related": ["abstain_crash"],
            },
            "legacy_mappings": {
                "timeout": "abstain_timeout",
                "crash": "abstain_crash",
            },
            "verification_methods": ["lean-disabled"],
        }
    
    @pytest.fixture
    def diff_module(self):
        """Import diff module."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        from diff_abstention_taxonomy import compute_diff, TaxonomyDiff
        return compute_diff, TaxonomyDiff
    
    def test_no_changes_detected(self, base_semantics, diff_module) -> None:
        """Identical semantics should have no changes."""
        compute_diff, _ = diff_module
        diff = compute_diff(base_semantics, base_semantics)
        
        assert not diff.has_changes
        assert not diff.has_breaking_changes
        assert diff.added_types == []
        assert diff.removed_types == []
    
    def test_added_type_detected(self, base_semantics, diff_module) -> None:
        """Adding a type should be detected."""
        compute_diff, _ = diff_module
        
        new_semantics = json.loads(json.dumps(base_semantics))
        new_semantics["taxonomy_version"] = "1.1.0"
        new_semantics["abstention_types"]["abstain_budget"] = {
            "category": "resource_related",
            "legacy_keys": ["budget_exceeded"],
        }
        
        diff = compute_diff(base_semantics, new_semantics)
        
        assert diff.has_changes
        assert "abstain_budget" in diff.added_types
        assert diff.removed_types == []
    
    def test_removed_type_detected(self, base_semantics, diff_module) -> None:
        """Removing a type should be detected as breaking."""
        compute_diff, _ = diff_module
        
        new_semantics = json.loads(json.dumps(base_semantics))
        new_semantics["taxonomy_version"] = "2.0.0"
        del new_semantics["abstention_types"]["abstain_crash"]
        
        diff = compute_diff(base_semantics, new_semantics)
        
        assert diff.has_changes
        assert diff.has_breaking_changes
        assert "abstain_crash" in diff.removed_types
    
    def test_category_change_detected(self, base_semantics, diff_module) -> None:
        """Moving a type to a different category should be detected."""
        compute_diff, _ = diff_module
        
        new_semantics = json.loads(json.dumps(base_semantics))
        new_semantics["taxonomy_version"] = "2.0.0"
        new_semantics["abstention_types"]["abstain_timeout"]["category"] = "invalid_related"
        
        diff = compute_diff(base_semantics, new_semantics)
        
        assert diff.has_changes
        assert diff.has_breaking_changes
        assert len(diff.category_changes) == 1
        assert diff.category_changes[0]["type"] == "abstain_timeout"
        assert diff.category_changes[0]["old_category"] == "timeout_related"
        assert diff.category_changes[0]["new_category"] == "invalid_related"
    
    def test_added_legacy_mapping_detected(self, base_semantics, diff_module) -> None:
        """Adding a legacy mapping should be detected."""
        compute_diff, _ = diff_module
        
        new_semantics = json.loads(json.dumps(base_semantics))
        new_semantics["taxonomy_version"] = "1.1.0"
        new_semantics["legacy_mappings"]["new_key"] = "abstain_timeout"
        
        diff = compute_diff(base_semantics, new_semantics)
        
        assert diff.has_changes
        assert "new_key" in diff.added_legacy_mappings
    
    def test_changed_legacy_mapping_detected(self, base_semantics, diff_module) -> None:
        """Changing a legacy mapping target should be detected as breaking."""
        compute_diff, _ = diff_module
        
        new_semantics = json.loads(json.dumps(base_semantics))
        new_semantics["taxonomy_version"] = "2.0.0"
        new_semantics["legacy_mappings"]["timeout"] = "abstain_crash"  # Changed target
        
        diff = compute_diff(base_semantics, new_semantics)
        
        assert diff.has_changes
        assert diff.has_breaking_changes
        assert len(diff.changed_legacy_mappings) == 1
        assert diff.changed_legacy_mappings[0]["key"] == "timeout"
    
    def test_diff_is_deterministic(self, base_semantics, diff_module) -> None:
        """Diff should be deterministic."""
        compute_diff, _ = diff_module
        
        new_semantics = json.loads(json.dumps(base_semantics))
        new_semantics["taxonomy_version"] = "1.1.0"
        new_semantics["abstention_types"]["abstain_budget"] = {
            "category": "resource_related",
            "legacy_keys": [],
        }
        
        diff1 = compute_diff(base_semantics, new_semantics)
        diff2 = compute_diff(base_semantics, new_semantics)
        
        assert diff1.to_dict() == diff2.to_dict()


# ---------------------------------------------------------------------------
# Task 3: Version Guard Tests
# ---------------------------------------------------------------------------

class TestVersionGuard:
    """Tests for check_taxonomy_version.py functionality."""
    
    @pytest.fixture
    def guard_module(self):
        """Import guard module."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        from check_taxonomy_version import (
            generate_current_semantics,
            compare_semantics,
        )
        return generate_current_semantics, compare_semantics
    
    def test_generate_current_semantics_has_version(self, guard_module) -> None:
        """Generated semantics must include taxonomy_version."""
        generate_current_semantics, _ = guard_module
        current = generate_current_semantics()
        
        assert "taxonomy_version" in current
        assert current["taxonomy_version"] == ABSTENTION_TAXONOMY_VERSION
    
    def test_generate_current_semantics_has_all_types(self, guard_module) -> None:
        """Generated semantics must include all AbstentionType values."""
        generate_current_semantics, _ = guard_module
        current = generate_current_semantics()
        
        for abst_type in AbstentionType:
            assert abst_type.value in current["abstention_types"]
    
    def test_generate_current_semantics_has_all_categories(self, guard_module) -> None:
        """Generated semantics must include all SemanticCategory values."""
        generate_current_semantics, _ = guard_module
        current = generate_current_semantics()
        
        for category in SemanticCategory:
            assert category.value in current["categories"]
    
    def test_unchanged_taxonomy_is_in_sync(self, guard_module) -> None:
        """Identical semantics should be in sync."""
        generate_current_semantics, compare_semantics = guard_module
        
        current = generate_current_semantics()
        
        is_in_sync, differences = compare_semantics(current, current)
        
        assert is_in_sync
        assert differences == []
    
    def test_changed_taxonomy_same_version_fails(self, guard_module) -> None:
        """Changed taxonomy with same version should fail."""
        generate_current_semantics, compare_semantics = guard_module
        
        exported = generate_current_semantics()
        current = json.loads(json.dumps(exported))
        
        # Add a fake type without bumping version
        current["abstention_types"]["abstain_fake"] = {
            "category": "timeout_related",
            "legacy_keys": [],
        }
        
        is_in_sync, differences = compare_semantics(exported, current)
        
        assert not is_in_sync
        assert len(differences) > 0
        assert any("abstain_fake" in d for d in differences)
    
    def test_changed_taxonomy_bumped_version_passes(self, guard_module) -> None:
        """Changed taxonomy with bumped version should pass (version change detected)."""
        generate_current_semantics, compare_semantics = guard_module
        
        exported = generate_current_semantics()
        current = json.loads(json.dumps(exported))
        
        # Add a fake type AND bump version
        current["taxonomy_version"] = "99.0.0"
        current["abstention_types"]["abstain_fake"] = {
            "category": "timeout_related",
            "legacy_keys": [],
        }
        
        # The compare function checks if content changed without version change
        # Since version changed, it should pass
        is_in_sync, differences = compare_semantics(exported, current)
        
        # Version changed, so even with differences, it's considered "in sync"
        # (the developer intentionally made changes and bumped version)
        assert is_in_sync  # Version bump makes it OK


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------

class TestVersioningIntegration:
    """Integration tests for the complete versioning workflow."""
    
    def test_export_and_diff_roundtrip(self) -> None:
        """Export, modify, and diff should work end-to-end."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        from diff_abstention_taxonomy import compute_diff, load_taxonomy
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Export current semantics
            path1 = Path(tmpdir) / "v1.json"
            export_semantics(path1)
            
            # Load and modify
            with open(path1) as f:
                data = json.load(f)
            
            data["taxonomy_version"] = "99.99.99"
            
            path2 = Path(tmpdir) / "v2.json"
            with open(path2, "w") as f:
                json.dump(data, f)
            
            # Diff
            old_data = load_taxonomy(path1)
            new_data = load_taxonomy(path2)
            diff = compute_diff(old_data, new_data)
            
            # Version changed but content same
            assert diff.old_version == ABSTENTION_TAXONOMY_VERSION
            assert diff.new_version == "99.99.99"
            assert not diff.has_changes  # No content changes
    
    def test_guard_detects_silent_type_addition(self) -> None:
        """Guard should detect silent type addition."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        from check_taxonomy_version import generate_current_semantics, compare_semantics
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Export current semantics as baseline
            path = Path(tmpdir) / "baseline.json"
            export_semantics(path)
            
            with open(path) as f:
                exported = json.load(f)
            
            # Simulate current code having an extra type without version bump
            current = generate_current_semantics()
            current["abstention_types"]["abstain_test_new"] = {
                "category": "timeout_related",
                "legacy_keys": [],
            }
            # NOTE: Not bumping version
            
            is_in_sync, differences = compare_semantics(exported, current)
            
            assert not is_in_sync
            assert any("abstain_test_new" in d for d in differences)


# ---------------------------------------------------------------------------
# Determinism Tests
# ---------------------------------------------------------------------------

class TestVersioningDeterminism:
    """Tests for deterministic behavior in versioning tools."""
    
    def test_export_is_deterministic(self) -> None:
        """Multiple exports should produce identical content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path1 = Path(tmpdir) / "export1.json"
            path2 = Path(tmpdir) / "export2.json"
            
            export_semantics(path1)
            export_semantics(path2)
            
            with open(path1) as f:
                data1 = json.load(f)
            with open(path2) as f:
                data2 = json.load(f)
            
            # Remove any timestamp fields for comparison
            data1.pop("generated_at", None)
            data2.pop("generated_at", None)
            
            assert data1 == data2
    
    def test_version_constant_is_stable(self) -> None:
        """get_taxonomy_version() should return same value across calls."""
        results = [get_taxonomy_version() for _ in range(5)]
        assert all(r == results[0] for r in results)

