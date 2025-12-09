"""
Tests for Abstention Vocabulary Drift Detection

These tests validate the "no drift" invariant (INV-TAX-1) by ensuring
that the vocabulary checker correctly identifies unknown abstention keys.

INVARIANTS TESTED:
    INV-TAX-1: No drift across layers
    INV-TAX-2: All abstention vocabularies collapse to canonical system
    INV-TAX-3: Serializers stable and forward-compatible

PHASE II — VERIFICATION ZONE
Agent B6 (abstention-ops-6)
"""

import pytest
import sys
import json
import tempfile
from pathlib import Path
from typing import Dict, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rfl.verification.abstention_taxonomy import (
    AbstentionType,
    classify_verification_method,
    classify_breakdown_key,
    COMPLETE_MAPPING,
)
from rfl.verification.abstention_semantics import (
    ABSTENTION_TREE,
    SemanticCategory,
    verify_tree_completeness,
    export_semantics,
    categorize,
)


# ---------------------------------------------------------------------------
# Semantic Tree Completeness Tests
# ---------------------------------------------------------------------------

class TestSemanticTreeCompleteness:
    """Tests for semantic tree coverage."""
    
    def test_all_types_covered(self) -> None:
        """Every AbstentionType must be in the semantic tree."""
        uncovered = verify_tree_completeness()
        assert uncovered == [], f"Uncovered types: {uncovered}"
    
    def test_no_orphan_entries(self) -> None:
        """Semantic tree must not contain invalid types."""
        valid_types = set(AbstentionType)
        for abst_type in ABSTENTION_TREE.keys():
            assert abst_type in valid_types, f"Invalid type in tree: {abst_type}"
    
    def test_all_categories_have_types(self) -> None:
        """Every semantic category must have at least one type."""
        categories_with_types = set(ABSTENTION_TREE.values())
        for category in SemanticCategory:
            assert category in categories_with_types, (
                f"Category {category} has no types mapped to it"
            )
    
    def test_categorize_all_types(self) -> None:
        """categorize() must return a category for every AbstentionType."""
        for abst_type in AbstentionType:
            category = categorize(abst_type)
            assert isinstance(category, SemanticCategory), (
                f"categorize({abst_type}) returned {type(category)}, expected SemanticCategory"
            )


# ---------------------------------------------------------------------------
# Complete Mapping Tests
# ---------------------------------------------------------------------------

class TestCompleteMapping:
    """Tests for the COMPLETE_MAPPING dictionary."""
    
    def test_mapping_covers_known_legacy_keys(self) -> None:
        """Critical legacy keys must be in COMPLETE_MAPPING."""
        critical_keys = [
            "timeout",
            "engine_failure",
            "empty_run",
            "budget_exceeded",
            "lean-disabled",
            "lean-timeout",
            "lean-error",
        ]
        for key in critical_keys:
            assert key in COMPLETE_MAPPING, f"Critical key {key!r} not in COMPLETE_MAPPING"
    
    def test_mapping_values_are_valid(self) -> None:
        """All values in COMPLETE_MAPPING must be valid AbstentionType."""
        for key, value in COMPLETE_MAPPING.items():
            assert isinstance(value, AbstentionType), (
                f"COMPLETE_MAPPING[{key!r}] = {value!r} is not an AbstentionType"
            )
    
    def test_classify_breakdown_key_uses_mapping(self) -> None:
        """classify_breakdown_key must use COMPLETE_MAPPING for known keys."""
        for key, expected_type in COMPLETE_MAPPING.items():
            # Skip method keys (they use classify_verification_method)
            if "-" in key:
                continue
            result = classify_breakdown_key(key)
            if result is not None:
                assert result == expected_type, (
                    f"classify_breakdown_key({key!r}) = {result}, expected {expected_type}"
                )


# ---------------------------------------------------------------------------
# Drift Detection Tests with Fake Keys
# ---------------------------------------------------------------------------

class TestDriftDetection:
    """Tests that verify drift detection catches unknown keys."""
    
    def test_unknown_abstention_type_rejected(self) -> None:
        """Unknown abstention type value must not be valid."""
        with pytest.raises(ValueError):
            AbstentionType("abstain_fake_unknown")
    
    def test_unknown_method_returns_none(self) -> None:
        """Unknown verification method must return None."""
        result = classify_verification_method("fake-unknown-method")
        assert result is None, f"Expected None, got {result}"
    
    def test_unknown_breakdown_key_returns_none(self) -> None:
        """Unknown breakdown key must return None."""
        result = classify_breakdown_key("fake_unknown_key")
        assert result is None, f"Expected None, got {result}"
    
    def test_fake_key_not_in_mapping(self) -> None:
        """Fake keys must not appear in COMPLETE_MAPPING."""
        fake_keys = [
            "fake_abstain",
            "unknown_abstention_xyz",
            "test_fake_key",
        ]
        for key in fake_keys:
            assert key not in COMPLETE_MAPPING, (
                f"Fake key {key!r} should not be in COMPLETE_MAPPING"
            )


# ---------------------------------------------------------------------------
# Governance Export Tests
# ---------------------------------------------------------------------------

class TestGovernanceExport:
    """Tests for the export_semantics function."""
    
    def test_export_creates_valid_json(self) -> None:
        """export_semantics must create valid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "semantics.json"
            result_path = export_semantics(out_path)
            
            assert result_path.exists(), f"File not created at {result_path}"
            
            with open(result_path) as f:
                data = json.load(f)
            
            assert "version" in data
            assert "abstention_types" in data
            assert "categories" in data
    
    def test_export_contains_all_types(self) -> None:
        """Export must contain all AbstentionType values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "semantics.json"
            export_semantics(out_path)
            
            with open(out_path) as f:
                data = json.load(f)
            
            exported_types = set(data["abstention_types"].keys())
            expected_types = {t.value for t in AbstentionType}
            
            assert exported_types == expected_types, (
                f"Missing types: {expected_types - exported_types}"
            )
    
    def test_export_contains_all_categories(self) -> None:
        """Export must contain all semantic categories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "semantics.json"
            export_semantics(out_path)
            
            with open(out_path) as f:
                data = json.load(f)
            
            exported_categories = set(data["categories"].keys())
            expected_categories = {c.value for c in SemanticCategory}
            
            assert exported_categories == expected_categories, (
                f"Missing categories: {expected_categories - exported_categories}"
            )
    
    def test_export_contains_legacy_mappings(self) -> None:
        """Export must contain legacy key mappings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "semantics.json"
            export_semantics(out_path)
            
            with open(out_path) as f:
                data = json.load(f)
            
            assert "legacy_mappings" in data
            assert len(data["legacy_mappings"]) > 0, "No legacy mappings exported"
            
            # Check specific mappings
            assert "lean-disabled" in data["legacy_mappings"]
            assert data["legacy_mappings"]["lean-disabled"] == "abstain_oracle_unavailable"
    
    def test_export_invariants_documented(self) -> None:
        """Export must include invariant documentation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "semantics.json"
            export_semantics(out_path)
            
            with open(out_path) as f:
                data = json.load(f)
            
            assert "invariants" in data
            assert "INV-TAX-1" in data["invariants"]
            assert "INV-TAX-2" in data["invariants"]
            assert "INV-TAX-3" in data["invariants"]


# ---------------------------------------------------------------------------
# Cross-Layer Consistency Tests
# ---------------------------------------------------------------------------

class TestCrossLayerConsistency:
    """Tests that verify consistency across layers."""
    
    def test_method_and_breakdown_dont_conflict(self) -> None:
        """Method and breakdown classifiers must not produce conflicting results."""
        # Keys that might exist in both domains
        shared_keys = ["timeout"]
        
        for key in shared_keys:
            method_result = classify_verification_method(key)
            breakdown_result = classify_breakdown_key(key)
            
            # If both return results, they should be compatible
            if method_result and breakdown_result:
                # Same type or related types (e.g., both timeout)
                assert method_result.value.startswith("abstain_") and breakdown_result.value.startswith("abstain_"), (
                    f"Conflicting classifications for {key!r}: method={method_result}, breakdown={breakdown_result}"
                )
    
    def test_lean_specific_types_consistent(self) -> None:
        """Lean-specific types must have consistent naming."""
        lean_types = [t for t in AbstentionType if "lean" in t.value.lower()]
        
        for abst_type in lean_types:
            assert abst_type.value.startswith("abstain_lean_"), (
                f"Lean type {abst_type} does not follow naming convention"
            )
    
    def test_serialization_roundtrip(self) -> None:
        """All types must survive serialization roundtrip."""
        from rfl.verification.abstention_taxonomy import serialize_abstention, deserialize_abstention
        
        for abst_type in AbstentionType:
            serialized = serialize_abstention(abst_type)
            deserialized = deserialize_abstention(serialized)
            assert deserialized == abst_type, (
                f"Roundtrip failed for {abst_type}: {serialized} → {deserialized}"
            )


# ---------------------------------------------------------------------------
# Invariant Tests
# ---------------------------------------------------------------------------

class TestInvariants:
    """Tests for the three main taxonomy invariants."""
    
    def test_inv_tax_1_no_drift(self) -> None:
        """INV-TAX-1: No drift across layers."""
        # All classification functions must use the same AbstentionType enum
        for method in ["lean-disabled", "lean-timeout", "lean-error"]:
            result = classify_verification_method(method)
            assert isinstance(result, AbstentionType), f"Drift detected for {method}"
        
        for key in ["timeout", "engine_failure", "empty_run"]:
            result = classify_breakdown_key(key)
            assert isinstance(result, AbstentionType), f"Drift detected for {key}"
    
    def test_inv_tax_2_collapse_to_canonical(self) -> None:
        """INV-TAX-2: All vocabularies collapse to canonical system."""
        # Multiple legacy forms must collapse to same canonical type
        timeout_forms = ["timeout", "derivation_timeout", "lean-timeout", "lean_timeout"]
        timeout_results = set()
        
        for form in timeout_forms:
            result = classify_breakdown_key(form) or classify_verification_method(form)
            if result:
                # Should be one of the timeout types
                assert "timeout" in result.value.lower(), (
                    f"Form {form!r} did not collapse to timeout type"
                )
                timeout_results.add(result.value)
        
        # All should map to known timeout types
        assert timeout_results.issubset({"abstain_timeout", "abstain_lean_timeout"}), (
            f"Unexpected timeout types: {timeout_results}"
        )
    
    def test_inv_tax_3_serializers_stable(self) -> None:
        """INV-TAX-3: Serializers stable and forward-compatible."""
        from rfl.verification.abstention_taxonomy import serialize_abstention
        
        # Check that serialization produces stable strings
        expected_serializations = {
            AbstentionType.ABSTAIN_TIMEOUT: "abstain_timeout",
            AbstentionType.ABSTAIN_BUDGET: "abstain_budget",
            AbstentionType.ABSTAIN_CRASH: "abstain_crash",
            AbstentionType.ABSTAIN_INVALID: "abstain_invalid",
            AbstentionType.ABSTAIN_ORACLE_UNAVAILABLE: "abstain_oracle_unavailable",
            AbstentionType.ABSTAIN_LEAN_TIMEOUT: "abstain_lean_timeout",
            AbstentionType.ABSTAIN_LEAN_ERROR: "abstain_lean_error",
        }
        
        for abst_type, expected in expected_serializations.items():
            result = serialize_abstention(abst_type)
            assert result == expected, (
                f"Serialization changed for {abst_type}: {result!r} != {expected!r}"
            )

