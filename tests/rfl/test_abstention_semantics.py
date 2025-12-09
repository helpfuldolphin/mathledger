"""
Tests for Abstention Semantics Module

Tests the semantic tree, categorization, and JSON schema validation
for the abstention taxonomy.

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
    ABSTENTION_TREE,
    SemanticCategory,
    categorize,
    get_category,
    get_types_for_category,
    get_all_categories,
    is_timeout_related,
    is_crash_related,
    is_resource_related,
    is_oracle_related,
    is_invalid_related,
    ABSTENTION_RECORD_SCHEMA,
    get_schema,
    get_schema_path,
    AbstentionValidationError,
    validate_abstention_data,
    validate_abstention_record,
    validate_abstention_json,
    aggregate_by_category,
    aggregate_histogram_by_category,
    verify_tree_completeness,
    export_semantics,
)


# ---------------------------------------------------------------------------
# Semantic Tree Structure Tests
# ---------------------------------------------------------------------------

class TestSemanticTreeStructure:
    """Tests for the ABSTENTION_TREE structure."""
    
    def test_tree_is_dict(self) -> None:
        """ABSTENTION_TREE must be a dictionary."""
        assert isinstance(ABSTENTION_TREE, dict)
    
    def test_tree_keys_are_abstention_types(self) -> None:
        """All keys in ABSTENTION_TREE must be AbstentionType."""
        for key in ABSTENTION_TREE.keys():
            assert isinstance(key, AbstentionType), f"Key {key} is not AbstentionType"
    
    def test_tree_values_are_semantic_categories(self) -> None:
        """All values in ABSTENTION_TREE must be SemanticCategory."""
        for value in ABSTENTION_TREE.values():
            assert isinstance(value, SemanticCategory), f"Value {value} is not SemanticCategory"
    
    def test_tree_covers_all_abstention_types(self) -> None:
        """ABSTENTION_TREE must cover all AbstentionType values."""
        for abst_type in AbstentionType:
            assert abst_type in ABSTENTION_TREE, f"{abst_type} not in tree"
    
    def test_verify_tree_completeness_returns_empty(self) -> None:
        """verify_tree_completeness must return empty list if complete."""
        uncovered = verify_tree_completeness()
        assert uncovered == [], f"Uncovered types: {uncovered}"


# ---------------------------------------------------------------------------
# Semantic Category Tests
# ---------------------------------------------------------------------------

class TestSemanticCategory:
    """Tests for SemanticCategory enum."""
    
    def test_has_all_expected_categories(self) -> None:
        """SemanticCategory must have all expected values."""
        expected = {
            "timeout_related",
            "resource_related",
            "crash_related",
            "oracle_related",
            "invalid_related",
        }
        actual = {c.value for c in SemanticCategory}
        assert actual == expected
    
    def test_str_returns_value(self) -> None:
        """str(SemanticCategory) must return the value."""
        assert str(SemanticCategory.TIMEOUT_RELATED) == "timeout_related"
        assert str(SemanticCategory.CRASH_RELATED) == "crash_related"


# ---------------------------------------------------------------------------
# Categorize Function Tests
# ---------------------------------------------------------------------------

class TestCategorize:
    """Tests for the categorize() function."""
    
    @pytest.mark.parametrize(
        "abst_type,expected_category",
        [
            (AbstentionType.ABSTAIN_TIMEOUT, SemanticCategory.TIMEOUT_RELATED),
            (AbstentionType.ABSTAIN_LEAN_TIMEOUT, SemanticCategory.TIMEOUT_RELATED),
            (AbstentionType.ABSTAIN_BUDGET, SemanticCategory.RESOURCE_RELATED),
            (AbstentionType.ABSTAIN_CRASH, SemanticCategory.CRASH_RELATED),
            (AbstentionType.ABSTAIN_LEAN_ERROR, SemanticCategory.CRASH_RELATED),
            (AbstentionType.ABSTAIN_ORACLE_UNAVAILABLE, SemanticCategory.ORACLE_RELATED),
            (AbstentionType.ABSTAIN_INVALID, SemanticCategory.INVALID_RELATED),
        ],
    )
    def test_categorize_returns_correct_category(
        self, abst_type: AbstentionType, expected_category: SemanticCategory
    ) -> None:
        """categorize() must return correct SemanticCategory."""
        result = categorize(abst_type)
        assert result == expected_category
    
    def test_categorize_returns_semantic_category(self) -> None:
        """categorize() must return SemanticCategory type."""
        result = categorize(AbstentionType.ABSTAIN_TIMEOUT)
        assert isinstance(result, SemanticCategory)
    
    def test_categorize_all_types(self) -> None:
        """categorize() must work for all AbstentionType values."""
        for abst_type in AbstentionType:
            result = categorize(abst_type)
            assert isinstance(result, SemanticCategory)


# ---------------------------------------------------------------------------
# Category Helper Tests
# ---------------------------------------------------------------------------

class TestCategoryHelpers:
    """Tests for category helper functions."""
    
    def test_get_category_returns_category(self) -> None:
        """get_category() must return SemanticCategory or None."""
        result = get_category(AbstentionType.ABSTAIN_TIMEOUT)
        assert result == SemanticCategory.TIMEOUT_RELATED
    
    def test_get_types_for_category(self) -> None:
        """get_types_for_category() must return list of AbstentionTypes."""
        timeout_types = get_types_for_category(SemanticCategory.TIMEOUT_RELATED)
        assert AbstentionType.ABSTAIN_TIMEOUT in timeout_types
        assert AbstentionType.ABSTAIN_LEAN_TIMEOUT in timeout_types
    
    def test_get_all_categories(self) -> None:
        """get_all_categories() must return all SemanticCategory values."""
        categories = get_all_categories()
        assert set(categories) == set(SemanticCategory)
    
    def test_is_timeout_related(self) -> None:
        """is_timeout_related() must correctly identify timeout types."""
        assert is_timeout_related(AbstentionType.ABSTAIN_TIMEOUT)
        assert is_timeout_related(AbstentionType.ABSTAIN_LEAN_TIMEOUT)
        assert not is_timeout_related(AbstentionType.ABSTAIN_CRASH)
    
    def test_is_crash_related(self) -> None:
        """is_crash_related() must correctly identify crash types."""
        assert is_crash_related(AbstentionType.ABSTAIN_CRASH)
        assert is_crash_related(AbstentionType.ABSTAIN_LEAN_ERROR)
        assert not is_crash_related(AbstentionType.ABSTAIN_TIMEOUT)
    
    def test_is_resource_related(self) -> None:
        """is_resource_related() must correctly identify resource types."""
        assert is_resource_related(AbstentionType.ABSTAIN_BUDGET)
        assert not is_resource_related(AbstentionType.ABSTAIN_CRASH)
    
    def test_is_oracle_related(self) -> None:
        """is_oracle_related() must correctly identify oracle types."""
        assert is_oracle_related(AbstentionType.ABSTAIN_ORACLE_UNAVAILABLE)
        assert not is_oracle_related(AbstentionType.ABSTAIN_TIMEOUT)
    
    def test_is_invalid_related(self) -> None:
        """is_invalid_related() must correctly identify invalid types."""
        assert is_invalid_related(AbstentionType.ABSTAIN_INVALID)
        assert not is_invalid_related(AbstentionType.ABSTAIN_CRASH)


# ---------------------------------------------------------------------------
# Schema Tests
# ---------------------------------------------------------------------------

class TestSchema:
    """Tests for JSON schema functionality."""
    
    def test_schema_is_dict(self) -> None:
        """ABSTENTION_RECORD_SCHEMA must be a dictionary."""
        assert isinstance(ABSTENTION_RECORD_SCHEMA, dict)
    
    def test_get_schema_returns_same(self) -> None:
        """get_schema() must return the schema dictionary."""
        assert get_schema() == ABSTENTION_RECORD_SCHEMA
    
    def test_schema_has_required_field(self) -> None:
        """Schema must have 'required' field."""
        assert "required" in ABSTENTION_RECORD_SCHEMA
        assert "abstention_type" in ABSTENTION_RECORD_SCHEMA["required"]
    
    def test_schema_abstention_type_enum(self) -> None:
        """Schema abstention_type must enumerate all types."""
        props = ABSTENTION_RECORD_SCHEMA.get("properties", {})
        abst_type_prop = props.get("abstention_type", {})
        enum_values = abst_type_prop.get("enum", [])
        
        expected = {t.value for t in AbstentionType}
        actual = set(enum_values)
        assert actual == expected


# ---------------------------------------------------------------------------
# Validation Tests
# ---------------------------------------------------------------------------

class TestValidation:
    """Tests for validation functions."""
    
    def test_validate_abstention_data_valid(self) -> None:
        """validate_abstention_data() must return empty list for valid data."""
        data = {"abstention_type": "abstain_timeout"}
        errors = validate_abstention_data(data)
        assert errors == []
    
    def test_validate_abstention_data_missing_type(self) -> None:
        """validate_abstention_data() must catch missing abstention_type."""
        data = {"method": "lean-disabled"}
        errors = validate_abstention_data(data)
        assert len(errors) > 0
        assert any("abstention_type" in e for e in errors)
    
    def test_validate_abstention_data_invalid_type(self) -> None:
        """validate_abstention_data() must catch invalid abstention_type."""
        data = {"abstention_type": "invalid_type_xyz"}
        errors = validate_abstention_data(data)
        assert len(errors) > 0
    
    def test_validate_abstention_data_wrong_field_type(self) -> None:
        """validate_abstention_data() must catch wrong field types."""
        data = {"abstention_type": "abstain_timeout", "method": 123}
        errors = validate_abstention_data(data)
        assert len(errors) > 0
        assert any("method" in e for e in errors)
    
    def test_validate_abstention_data_context_must_be_dict(self) -> None:
        """validate_abstention_data() must require context to be dict."""
        data = {"abstention_type": "abstain_timeout", "context": "not a dict"}
        errors = validate_abstention_data(data)
        assert len(errors) > 0
        assert any("context" in e for e in errors)


# ---------------------------------------------------------------------------
# Aggregation Tests
# ---------------------------------------------------------------------------

class TestAggregation:
    """Tests for aggregation functions."""
    
    def test_aggregate_by_category_basic(self) -> None:
        """aggregate_by_category() must group types by category."""
        types = [
            AbstentionType.ABSTAIN_TIMEOUT,
            AbstentionType.ABSTAIN_LEAN_TIMEOUT,
            AbstentionType.ABSTAIN_CRASH,
        ]
        result = aggregate_by_category(types)
        
        # Should have timeout_related and crash_related
        assert SemanticCategory.TIMEOUT_RELATED.value in result or "timeout_related" in result
        assert SemanticCategory.CRASH_RELATED.value in result or "crash_related" in result
    
    def test_aggregate_by_category_empty(self) -> None:
        """aggregate_by_category() with empty list returns empty dict."""
        result = aggregate_by_category([])
        assert result == {}
    
    def test_aggregate_histogram_by_category(self) -> None:
        """aggregate_histogram_by_category() must aggregate histogram."""
        histogram = {
            "abstain_timeout": 5,
            "abstain_lean_timeout": 3,
            "abstain_crash": 2,
        }
        result = aggregate_histogram_by_category(histogram)
        
        # Check that results have expected categories (by value)
        timeout_key = SemanticCategory.TIMEOUT_RELATED.value
        crash_key = SemanticCategory.CRASH_RELATED.value
        assert result.get(timeout_key, 0) == 8
        assert result.get(crash_key, 0) == 2


# ---------------------------------------------------------------------------
# Export Semantics Tests
# ---------------------------------------------------------------------------

class TestExportSemantics:
    """Tests for export_semantics() function."""
    
    def test_export_creates_file(self) -> None:
        """export_semantics() must create a JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "semantics.json"
            result = export_semantics(out_path)
            assert result.exists()
    
    def test_export_creates_valid_json(self) -> None:
        """export_semantics() must create valid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "semantics.json"
            export_semantics(out_path)
            
            with open(out_path) as f:
                data = json.load(f)
            
            assert isinstance(data, dict)
    
    def test_export_contains_version(self) -> None:
        """Export must contain version field."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "semantics.json"
            export_semantics(out_path)
            
            with open(out_path) as f:
                data = json.load(f)
            
            assert "version" in data
    
    def test_export_contains_abstention_types(self) -> None:
        """Export must contain all abstention types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "semantics.json"
            export_semantics(out_path)
            
            with open(out_path) as f:
                data = json.load(f)
            
            assert "abstention_types" in data
            for abst_type in AbstentionType:
                assert abst_type.value in data["abstention_types"]
    
    def test_export_contains_categories(self) -> None:
        """Export must contain all categories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "semantics.json"
            export_semantics(out_path)
            
            with open(out_path) as f:
                data = json.load(f)
            
            assert "categories" in data
            for category in SemanticCategory:
                assert category.value in data["categories"]
    
    def test_export_contains_invariants(self) -> None:
        """Export must contain invariant documentation."""
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
# Determinism Tests
# ---------------------------------------------------------------------------

class TestDeterminism:
    """Tests for deterministic behavior."""
    
    def test_categorize_deterministic(self) -> None:
        """categorize() must return same result on repeated calls."""
        for abst_type in AbstentionType:
            results = [categorize(abst_type) for _ in range(3)]
            assert all(r == results[0] for r in results)
    
    def test_aggregate_deterministic(self) -> None:
        """aggregate_by_category() must return same result on repeated calls."""
        types = list(AbstentionType)
        results = [aggregate_by_category(types) for _ in range(3)]
        assert all(r == results[0] for r in results)


# ---------------------------------------------------------------------------
# Validation Error Tests
# ---------------------------------------------------------------------------

class TestAbstentionValidationError:
    """Tests for AbstentionValidationError."""
    
    def test_error_has_errors_list(self) -> None:
        """AbstentionValidationError must have errors list."""
        error = AbstentionValidationError(["error1", "error2"])
        assert hasattr(error, "errors")
        assert error.errors == ["error1", "error2"]
    
    def test_error_message_includes_errors(self) -> None:
        """Error message must mention the errors."""
        error = AbstentionValidationError(["field X is invalid"])
        assert "field X is invalid" in str(error)
