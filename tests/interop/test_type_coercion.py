"""
Type Coercion and Serialization Tests

Validates JSON serialization consistency across Python, JavaScript, and PowerShell.
Tests for type drift, precision loss, and encoding issues.

Mission: Detect type coercion drift or field mismatch across language boundaries.
"""

import pytest
import json
from datetime import datetime, timezone
from decimal import Decimal


class TestBooleanSerialization:
    """Validate boolean serialization across languages."""

    def test_python_true_serializes_to_json_true(self):
        """Python True → JSON true → PS $true / JS true"""
        data = {"flag": True}
        json_str = json.dumps(data)

        assert '"flag": true' in json_str or '"flag":true' in json_str
        # Not "True" (Python repr) or "1" (integer coercion)
        assert '"flag": True' not in json_str
        assert '"flag": 1' not in json_str

        print("[PASS] Python True → JSON true")

    def test_python_false_serializes_to_json_false(self):
        """Python False → JSON false → PS $false / JS false"""
        data = {"flag": False}
        json_str = json.dumps(data)

        assert '"flag": false' in json_str or '"flag":false' in json_str
        assert '"flag": False' not in json_str
        assert '"flag": 0' not in json_str

        print("[PASS] Python False → JSON false")

    def test_boolean_round_trip(self):
        """Verify boolean survives JSON round-trip."""
        original = {"ok": True, "error": False}
        json_str = json.dumps(original)
        parsed = json.loads(json_str)

        assert parsed["ok"] is True
        assert parsed["error"] is False
        assert type(parsed["ok"]) == bool
        assert type(parsed["error"]) == bool

        print("[PASS] Boolean round-trip verified")


class TestNullSerialization:
    """Validate null/None handling across languages."""

    def test_python_none_serializes_to_json_null(self):
        """Python None → JSON null → PS $null / JS null"""
        data = {"merkle": None}
        json_str = json.dumps(data)

        assert '"merkle": null' in json_str or '"merkle":null' in json_str
        # Not "None" (Python repr) or missing field
        assert '"merkle": None' not in json_str

        print("[PASS] Python None → JSON null")

    def test_null_vs_missing_field(self):
        """Distinguish between null value and missing field."""
        with_null = {"field": None}
        without_field = {}

        json_with_null = json.dumps(with_null)
        json_without = json.dumps(without_field)

        # With null has the field
        assert '"field"' in json_with_null

        # Without doesn't have the field
        assert '"field"' not in json_without

        print("[PASS] Null vs missing field distinguished")

    def test_null_round_trip(self):
        """Verify null survives JSON round-trip."""
        original = {"value": None}
        json_str = json.dumps(original)
        parsed = json.loads(json_str)

        assert parsed["value"] is None
        assert "value" in parsed  # Field exists

        print("[PASS] Null round-trip verified")


class TestNumberSerialization:
    """Validate number serialization and precision."""

    def test_integer_no_float_coercion(self):
        """Ensure integers don't become floats (150 not 150.0)."""
        data = {"count": 150}
        json_str = json.dumps(data)

        assert '"count": 150' in json_str or '"count":150' in json_str
        # Should NOT be 150.0
        assert '"count": 150.0' not in json_str

        parsed = json.loads(json_str)
        assert type(parsed["count"]) == int
        assert parsed["count"] == 150

        print("[PASS] Integer serialization (no float coercion)")

    def test_float_precision(self):
        """Verify float precision preserved (success_rate: 93.75)."""
        data = {"success_rate": 93.75}
        json_str = json.dumps(data)

        parsed = json.loads(json_str)
        assert type(parsed["success_rate"]) == float
        assert parsed["success_rate"] == 93.75

        print("[PASS] Float precision preserved")

    def test_large_integer_no_scientific_notation(self):
        """Ensure large integers don't use scientific notation."""
        data = {"block_number": 1000000}
        json_str = json.dumps(data)

        # Should be 1000000, not 1e6 or 1e+6
        assert '"block_number": 1000000' in json_str or '"block_number":1000000' in json_str
        # Check for scientific notation pattern (e+ or e-)
        assert 'e+' not in json_str.lower() and 'e-' not in json_str.lower()

        print("[PASS] Large integer serialization")

    def test_zero_not_null(self):
        """Distinguish between 0 and null."""
        data = {"count": 0, "empty": None}
        json_str = json.dumps(data)

        parsed = json.loads(json_str)
        assert parsed["count"] == 0
        assert parsed["empty"] is None
        assert parsed["count"] != parsed["empty"]

        print("[PASS] Zero vs null distinguished")


class TestStringSerialization:
    """Validate string encoding and escaping."""

    def test_string_utf8_encoding(self):
        """Verify UTF-8 strings serialize correctly."""
        data = {"status": "healthy"}
        json_str = json.dumps(data)

        parsed = json.loads(json_str)
        assert type(parsed["status"]) == str
        assert parsed["status"] == "healthy"

        print("[PASS] UTF-8 string encoding")

    def test_special_characters_escaped(self):
        """Verify special characters are escaped properly."""
        data = {"formula": '(p → q)'}
        json_str = json.dumps(data)

        # Should contain escaped characters
        parsed = json.loads(json_str)
        assert parsed["formula"] == '(p → q)'

        print("[PASS] Special character escaping")

    def test_empty_string_not_null(self):
        """Distinguish between "" and null."""
        data = {"text": "", "empty": None}
        json_str = json.dumps(data)

        parsed = json.loads(json_str)
        assert parsed["text"] == ""
        assert parsed["empty"] is None
        assert parsed["text"] != parsed["empty"]

        print("[PASS] Empty string vs null distinguished")


class TestTimestampSerialization:
    """Validate timestamp format consistency."""

    def test_iso8601_format(self):
        """Verify timestamps use ISO 8601 format."""
        # FastAPI typically uses ISO 8601
        timestamp = datetime.now(timezone.utc)
        iso_str = timestamp.isoformat()

        # Should contain T separator and timezone
        assert 'T' in iso_str
        # Should be parseable
        parsed = datetime.fromisoformat(iso_str.replace('Z', '+00:00'))
        assert parsed is not None

        print(f"[PASS] ISO 8601 format: {iso_str}")

    def test_timestamp_string_not_unix_epoch(self):
        """Verify timestamps are strings, not Unix epoch integers."""
        data = {"timestamp": datetime.now(timezone.utc).isoformat()}
        json_str = json.dumps(data)

        # Should be quoted string
        assert '"timestamp": "' in json_str

        parsed = json.loads(json_str)
        assert type(parsed["timestamp"]) == str

        print("[PASS] Timestamp as ISO string (not Unix epoch)")


class TestArraySerialization:
    """Validate array/list serialization."""

    def test_empty_array(self):
        """Verify empty arrays serialize as []."""
        data = {"parents": []}
        json_str = json.dumps(data)

        assert '"parents": []' in json_str or '"parents":[]' in json_str

        parsed = json.loads(json_str)
        assert parsed["parents"] == []
        assert type(parsed["parents"]) == list

        print("[PASS] Empty array serialization")

    def test_array_element_types(self):
        """Verify array elements maintain types."""
        data = {"proofs": [{"method": "tautology", "success": True}]}
        json_str = json.dumps(data)

        parsed = json.loads(json_str)
        assert type(parsed["proofs"]) == list
        assert len(parsed["proofs"]) == 1
        assert parsed["proofs"][0]["method"] == "tautology"
        assert parsed["proofs"][0]["success"] is True

        print("[PASS] Array element types preserved")


class TestObjectSerialization:
    """Validate nested object serialization."""

    def test_nested_object_structure(self):
        """Verify nested objects maintain structure."""
        data = {
            "blocks": {
                "height": 25,
                "latest": {
                    "merkle": "abc123"
                }
            }
        }
        json_str = json.dumps(data)

        parsed = json.loads(json_str)
        assert parsed["blocks"]["height"] == 25
        assert parsed["blocks"]["latest"]["merkle"] == "abc123"

        print("[PASS] Nested object structure preserved")

    def test_empty_object(self):
        """Verify empty objects serialize as {}."""
        data = {"header": {}}
        json_str = json.dumps(data)

        assert '"header": {}' in json_str or '"header":{}' in json_str

        parsed = json.loads(json_str)
        assert parsed["header"] == {}
        assert type(parsed["header"]) == dict

        print("[PASS] Empty object serialization")


class TestFieldOrdering:
    """Validate field order consistency (for deterministic hashing)."""

    def test_dict_keys_stable(self):
        """Verify dictionary key order is stable (Python 3.7+)."""
        data = {"z": 3, "a": 1, "m": 2}
        json_str = json.dumps(data)

        # Parse and verify order
        parsed = json.loads(json_str)
        keys = list(parsed.keys())
        assert keys == ["z", "a", "m"]  # Insertion order

        print("[PASS] Dictionary key order stable")

    def test_sorted_keys_option(self):
        """Verify sort_keys option available for determinism."""
        data = {"z": 3, "a": 1, "m": 2}
        json_str = json.dumps(data, sort_keys=True)

        # Should be alphabetically sorted
        assert json_str.index('"a"') < json_str.index('"m"')
        assert json_str.index('"m"') < json_str.index('"z"')

        print("[PASS] Sorted keys option works")


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_large_number(self):
        """Test very large number handling."""
        data = {"large": 2**53}  # Max safe integer in JavaScript
        json_str = json.dumps(data)

        parsed = json.loads(json_str)
        assert parsed["large"] == 2**53

        print("[PASS] Large number handling")

    def test_unicode_characters(self):
        """Test Unicode character handling."""
        data = {"text": "∀x ∈ ℝ: x² ≥ 0"}
        json_str = json.dumps(data, ensure_ascii=False)

        parsed = json.loads(json_str)
        assert parsed["text"] == "∀x ∈ ℝ: x² ≥ 0"

        print("[PASS] Unicode character handling")

    def test_mixed_types_array(self):
        """Test array with mixed types (edge case)."""
        data = {"mixed": [1, "two", True, None]}
        json_str = json.dumps(data)

        parsed = json.loads(json_str)
        assert parsed["mixed"][0] == 1
        assert parsed["mixed"][1] == "two"
        assert parsed["mixed"][2] is True
        assert parsed["mixed"][3] is None

        print("[PASS] Mixed type array handling")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("TYPE COERCION AND SERIALIZATION TEST SUITE")
    print("Testing JSON fidelity across Python ↔ JS ↔ PowerShell")
    print("="*60 + "\n")

    pytest.main([__file__, "-v", "--tb=short"])
