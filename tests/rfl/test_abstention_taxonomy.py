"""
Unit Tests for Abstention Taxonomy Module

Tests the canonical abstention type enumeration and classification utilities
defined in rfl/verification/abstention_taxonomy.py.

Test Categories:
    1. Verification method classification (classify_verification_method)
    2. Breakdown key classification (classify_breakdown_key)
    3. Serialization/deserialization roundtrip
    4. Unknown value handling
    5. Completeness and determinism properties

PHASE II — VERIFICATION ZONE
Agent B6 (verifier-ops-6)
"""

import pytest
from typing import Optional

from rfl.verification.abstention_taxonomy import (
    AbstentionType,
    classify_verification_method,
    classify_breakdown_key,
    serialize_abstention,
    deserialize_abstention,
    is_abstention_method,
    ABSTENTION_METHOD_STRINGS,
    COMPLETE_MAPPING,
    get_all_abstention_types,
    get_core_abstention_types,
    get_lean_abstention_types,
    format_abstention_for_log,
    get_mapping_table,
)


# ---------------------------------------------------------------------------
# Test: classify_verification_method known values
# ---------------------------------------------------------------------------

class TestClassifyVerificationMethodKnownValues:
    """Test that each known verification method string maps to the correct AbstentionType."""

    @pytest.mark.parametrize("method,expected", [
        # Lean fallback states
        ("lean-disabled", AbstentionType.ABSTAIN_ORACLE_UNAVAILABLE),
        ("lean-timeout", AbstentionType.ABSTAIN_LEAN_TIMEOUT),
        ("lean-error", AbstentionType.ABSTAIN_LEAN_ERROR),
        # Truth-table states
        ("truth-table-error", AbstentionType.ABSTAIN_INVALID),
        ("truth-table-non-tautology", AbstentionType.ABSTAIN_INVALID),
        # Alternative spellings (underscore variants)
        ("lean_disabled", AbstentionType.ABSTAIN_ORACLE_UNAVAILABLE),
        ("lean_timeout", AbstentionType.ABSTAIN_LEAN_TIMEOUT),
        ("lean_error", AbstentionType.ABSTAIN_LEAN_ERROR),
        ("truth_table_error", AbstentionType.ABSTAIN_INVALID),
        # Bootstrap abstention
        ("ABSTAIN", AbstentionType.ABSTAIN_INVALID),
    ])
    def test_known_verification_methods(self, method: str, expected: AbstentionType) -> None:
        """Each known verification method maps to the expected AbstentionType."""
        result = classify_verification_method(method)
        assert result == expected, f"Expected {method!r} → {expected}, got {result}"

    @pytest.mark.parametrize("method", [
        # Success methods should return None (not abstentions)
        "pattern",
        "truth-table",
        "lean",
        # Unknown methods should return None
        "unknown_method",
        "foobar",
        "",
    ])
    def test_non_abstention_methods_return_none(self, method: str) -> None:
        """Non-abstention methods return None."""
        result = classify_verification_method(method)
        assert result is None, f"Expected {method!r} → None, got {result}"


# ---------------------------------------------------------------------------
# Test: classify_breakdown_key known values
# ---------------------------------------------------------------------------

class TestClassifyBreakdownKeyKnownValues:
    """Test that each known breakdown key maps to the correct AbstentionType."""

    @pytest.mark.parametrize("key,expected", [
        # Crash/error states
        ("engine_failure", AbstentionType.ABSTAIN_CRASH),
        ("unexpected_error", AbstentionType.ABSTAIN_CRASH),
        ("crash", AbstentionType.ABSTAIN_CRASH),
        # Timeout states
        ("timeout", AbstentionType.ABSTAIN_TIMEOUT),
        ("timeout_abstain", AbstentionType.ABSTAIN_TIMEOUT),
        ("derivation_timeout", AbstentionType.ABSTAIN_TIMEOUT),
        # Invalid/empty states
        ("empty_run", AbstentionType.ABSTAIN_INVALID),
        ("no_successful_proofs", AbstentionType.ABSTAIN_INVALID),
        ("zero_throughput", AbstentionType.ABSTAIN_INVALID),
        ("invalid_input", AbstentionType.ABSTAIN_INVALID),
        # Pending/unavailable states
        ("pending_validation", AbstentionType.ABSTAIN_ORACLE_UNAVAILABLE),
        ("oracle_unavailable", AbstentionType.ABSTAIN_ORACLE_UNAVAILABLE),
        # Budget states
        ("budget_exceeded", AbstentionType.ABSTAIN_BUDGET),
        ("candidate_limit", AbstentionType.ABSTAIN_BUDGET),
        ("resource_exhausted", AbstentionType.ABSTAIN_BUDGET),
        ("memory_limit", AbstentionType.ABSTAIN_BUDGET),
        # Lean-specific (histogram keys)
        ("lean_timeout", AbstentionType.ABSTAIN_LEAN_TIMEOUT),
        ("lean_error", AbstentionType.ABSTAIN_LEAN_ERROR),
        # Derivation abstention aggregate
        ("derivation_abstain", AbstentionType.ABSTAIN_ORACLE_UNAVAILABLE),
        # Attestation tracking
        ("attestation_mass", AbstentionType.ABSTAIN_ORACLE_UNAVAILABLE),
        ("attestation_events", AbstentionType.ABSTAIN_ORACLE_UNAVAILABLE),
    ])
    def test_known_breakdown_keys(self, key: str, expected: AbstentionType) -> None:
        """Each known breakdown key maps to the expected AbstentionType."""
        result = classify_breakdown_key(key)
        assert result == expected, f"Expected {key!r} → {expected}, got {result}"

    @pytest.mark.parametrize("key", [
        "unknown_key",
        "foobar",
        "",
        "success",  # Not an abstention
    ])
    def test_unknown_keys_return_none(self, key: str) -> None:
        """Unknown breakdown keys return None."""
        result = classify_breakdown_key(key)
        assert result is None, f"Expected {key!r} → None, got {result}"


# ---------------------------------------------------------------------------
# Test: Serialization/Deserialization Roundtrip
# ---------------------------------------------------------------------------

class TestSerializeDeserializeRoundtrip:
    """Test that serialize and deserialize are inverse operations."""

    @pytest.mark.parametrize("abstention_type", list(AbstentionType))
    def test_roundtrip_for_all_types(self, abstention_type: AbstentionType) -> None:
        """deserialize(serialize(t)) == t for all AbstentionType members."""
        serialized = serialize_abstention(abstention_type)
        deserialized = deserialize_abstention(serialized)
        assert deserialized == abstention_type, (
            f"Roundtrip failed: {abstention_type} → {serialized!r} → {deserialized}"
        )

    def test_serialize_produces_string_value(self) -> None:
        """serialize_abstention returns the enum's string value."""
        for abstention_type in AbstentionType:
            serialized = serialize_abstention(abstention_type)
            assert serialized == abstention_type.value
            assert isinstance(serialized, str)

    def test_deserialize_invalid_value_raises(self) -> None:
        """deserialize_abstention raises ValueError for invalid values."""
        with pytest.raises(ValueError) as exc_info:
            deserialize_abstention("invalid_value")
        assert "Unknown abstention type" in str(exc_info.value)
        assert "invalid_value" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test: Unknown Values Behavior
# ---------------------------------------------------------------------------

class TestUnknownValuesBehavior:
    """Test explicit behavior for unknown/unrecognized values."""

    def test_classify_verification_method_unknown_returns_none(self) -> None:
        """classify_verification_method returns None for unknown methods."""
        assert classify_verification_method("xyz_unknown") is None
        assert classify_verification_method("") is None
        assert classify_verification_method("success") is None  # Not an abstention method

    def test_classify_breakdown_key_unknown_returns_none(self) -> None:
        """classify_breakdown_key returns None for unknown keys."""
        assert classify_breakdown_key("xyz_unknown") is None
        assert classify_breakdown_key("") is None

    def test_deserialize_unknown_raises_value_error(self) -> None:
        """deserialize_abstention raises ValueError for unknown values."""
        invalid_values = ["unknown", "ABSTAIN_UNKNOWN", "abstain", ""]
        for value in invalid_values:
            with pytest.raises(ValueError):
                deserialize_abstention(value)


# ---------------------------------------------------------------------------
# Test: Completeness
# ---------------------------------------------------------------------------

class TestCompleteness:
    """Test that the taxonomy is complete and covers all known abstention scenarios."""

    def test_all_abstention_method_strings_are_classified(self) -> None:
        """Every string in ABSTENTION_METHOD_STRINGS should classify to an AbstentionType."""
        for method in ABSTENTION_METHOD_STRINGS:
            result = classify_verification_method(method)
            assert result is not None, f"ABSTENTION_METHOD_STRINGS contains unclassified method: {method!r}"

    def test_enum_has_expected_members(self) -> None:
        """AbstentionType enum has all expected members."""
        expected_members = {
            "ABSTAIN_TIMEOUT",
            "ABSTAIN_BUDGET",
            "ABSTAIN_CRASH",
            "ABSTAIN_INVALID",
            "ABSTAIN_ORACLE_UNAVAILABLE",
            "ABSTAIN_LEAN_TIMEOUT",
            "ABSTAIN_LEAN_ERROR",
        }
        actual_members = {m.name for m in AbstentionType}
        assert actual_members == expected_members, f"Missing or extra members: {actual_members ^ expected_members}"

    def test_core_types_subset(self) -> None:
        """Core types are a proper subset of all types."""
        core = set(get_core_abstention_types())
        all_types = set(get_all_abstention_types())
        assert core < all_types  # strict subset

    def test_lean_types_subset(self) -> None:
        """Lean types are a proper subset of all types."""
        lean = set(get_lean_abstention_types())
        all_types = set(get_all_abstention_types())
        assert lean < all_types  # strict subset

    def test_core_plus_lean_equals_all(self) -> None:
        """Core types + Lean types = all types."""
        core = set(get_core_abstention_types())
        lean = set(get_lean_abstention_types())
        all_types = set(get_all_abstention_types())
        assert core | lean == all_types


# ---------------------------------------------------------------------------
# Test: Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    """Test that classification is deterministic (same input → same output)."""

    def test_classify_verification_method_deterministic(self) -> None:
        """Same verification method string always maps to same AbstentionType."""
        test_methods = ["lean-disabled", "lean-timeout", "truth-table-error", "unknown"]
        for method in test_methods:
            results = [classify_verification_method(method) for _ in range(100)]
            assert all(r == results[0] for r in results), f"Non-deterministic for {method!r}"

    def test_classify_breakdown_key_deterministic(self) -> None:
        """Same breakdown key always maps to same AbstentionType."""
        test_keys = ["engine_failure", "timeout", "budget_exceeded", "unknown"]
        for key in test_keys:
            results = [classify_breakdown_key(key) for _ in range(100)]
            assert all(r == results[0] for r in results), f"Non-deterministic for {key!r}"

    def test_serialize_deterministic(self) -> None:
        """Serialization is deterministic."""
        for abstention_type in AbstentionType:
            results = [serialize_abstention(abstention_type) for _ in range(100)]
            assert all(r == results[0] for r in results)


# ---------------------------------------------------------------------------
# Test: is_abstention_method helper
# ---------------------------------------------------------------------------

class TestIsAbstentionMethod:
    """Test the is_abstention_method convenience function."""

    @pytest.mark.parametrize("method,expected", [
        # Abstention methods
        ("lean-disabled", True),
        ("lean-timeout", True),
        ("lean-error", True),
        ("truth-table-error", True),
        ("truth-table-non-tautology", True),
        # Alternative spellings
        ("lean_disabled", True),
        ("ABSTAIN", True),
        # Non-abstention methods
        ("pattern", False),
        ("truth-table", False),
        ("lean", False),
        ("unknown", False),
        ("", False),
    ])
    def test_is_abstention_method(self, method: str, expected: bool) -> None:
        """is_abstention_method correctly identifies abstention methods."""
        result = is_abstention_method(method)
        assert result == expected, f"Expected is_abstention_method({method!r}) = {expected}, got {result}"


# ---------------------------------------------------------------------------
# Test: AbstentionType properties
# ---------------------------------------------------------------------------

class TestAbstentionTypeProperties:
    """Test AbstentionType enum properties."""

    def test_is_lean_specific(self) -> None:
        """is_lean_specific property correctly identifies Lean-specific types."""
        lean_specific = [AbstentionType.ABSTAIN_LEAN_TIMEOUT, AbstentionType.ABSTAIN_LEAN_ERROR]
        non_lean = [t for t in AbstentionType if t not in lean_specific]
        
        for t in lean_specific:
            assert t.is_lean_specific, f"{t} should be Lean-specific"
        for t in non_lean:
            assert not t.is_lean_specific, f"{t} should not be Lean-specific"

    def test_general_category(self) -> None:
        """general_category property maps Lean types to core categories."""
        assert AbstentionType.ABSTAIN_LEAN_TIMEOUT.general_category == AbstentionType.ABSTAIN_TIMEOUT
        assert AbstentionType.ABSTAIN_LEAN_ERROR.general_category == AbstentionType.ABSTAIN_CRASH
        
        # Core types map to themselves
        for t in get_core_abstention_types():
            assert t.general_category == t, f"{t}.general_category should be {t}"

    def test_str_returns_value(self) -> None:
        """str(AbstentionType) returns the string value."""
        for t in AbstentionType:
            assert str(t) == t.value


# ---------------------------------------------------------------------------
# Test: format_abstention_for_log
# ---------------------------------------------------------------------------

class TestFormatAbstentionForLog:
    """Test the logging format helper."""

    def test_basic_format(self) -> None:
        """Basic format produces expected output."""
        result = format_abstention_for_log(AbstentionType.ABSTAIN_TIMEOUT)
        assert result == "[ABSTAIN:abstain_timeout]"

    def test_format_with_context(self) -> None:
        """Format with context includes key=value pairs."""
        result = format_abstention_for_log(
            AbstentionType.ABSTAIN_CRASH,
            {"error": "segfault", "code": 11}
        )
        assert "[ABSTAIN:abstain_crash]" in result
        assert "error=segfault" in result
        assert "code=11" in result


# ---------------------------------------------------------------------------
# Test: get_mapping_table
# ---------------------------------------------------------------------------

class TestGetMappingTable:
    """Test the mapping table documentation function."""

    def test_returns_dict(self) -> None:
        """get_mapping_table returns a dict."""
        table = get_mapping_table()
        assert isinstance(table, dict)

    def test_all_values_are_strings(self) -> None:
        """All keys and values are strings."""
        table = get_mapping_table()
        for k, v in table.items():
            assert isinstance(k, str), f"Key {k!r} is not a string"
            assert isinstance(v, str), f"Value {v!r} for key {k!r} is not a string"

    def test_values_are_valid_abstention_types(self) -> None:
        """All values are valid AbstentionType values."""
        table = get_mapping_table()
        valid_values = {t.value for t in AbstentionType}
        for k, v in table.items():
            assert v in valid_values, f"Value {v!r} for key {k!r} is not a valid AbstentionType value"


# ---------------------------------------------------------------------------
# Test: COMPLETE_MAPPING
# ---------------------------------------------------------------------------

class TestCompleteMapping:
    """Test the COMPLETE_MAPPING constant."""

    def test_complete_mapping_includes_verification_methods(self) -> None:
        """COMPLETE_MAPPING includes all ABSTENTION_METHOD_STRINGS."""
        for method in ABSTENTION_METHOD_STRINGS:
            assert method in COMPLETE_MAPPING, f"{method!r} not in COMPLETE_MAPPING"

    def test_complete_mapping_values_are_abstention_types(self) -> None:
        """All COMPLETE_MAPPING values are AbstentionType instances."""
        for k, v in COMPLETE_MAPPING.items():
            assert isinstance(v, AbstentionType), f"COMPLETE_MAPPING[{k!r}] = {v!r} is not an AbstentionType"


# ---------------------------------------------------------------------------
# Test: Backward Compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    """Test that the taxonomy maintains backward compatibility."""

    def test_abstention_method_strings_unchanged(self) -> None:
        """ABSTENTION_METHOD_STRINGS contains the expected legacy values."""
        expected = {
            "lean-disabled",
            "lean-timeout",
            "lean-error",
            "truth-table-error",
            "truth-table-non-tautology",
        }
        assert ABSTENTION_METHOD_STRINGS == expected

    def test_serialization_produces_lowercase_underscored(self) -> None:
        """Serialization produces lowercase underscored strings (JSONL compatible)."""
        for t in AbstentionType:
            serialized = serialize_abstention(t)
            assert serialized.islower() or "_" in serialized, f"Serialized {t} = {serialized!r} unexpected format"
            # All values should start with "abstain_"
            assert serialized.startswith("abstain_"), f"Serialized {t} = {serialized!r} should start with 'abstain_'"

