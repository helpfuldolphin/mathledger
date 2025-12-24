"""
Tests for normalization.domain_spec module (FOL_FIN_EQ_v1 domain specification parsing).

Phase 1 RED: These tests define the expected interface for parse_domain_spec().
All tests will fail until normalization/domain_spec.py is implemented.

Design decision: parse_domain_spec() accepts ONLY pathlike objects, not dicts.
This enforces file-based governance and audit trail for domain specifications.
"""

import json
from pathlib import Path

import pytest

# Test fixtures directory
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "fol"


class TestParseDomainSpec:
    """Tests for parse_domain_spec() function."""

    def test_parse_z2_domain(self):
        """Parse a valid Z2 domain specification."""
        from normalization.domain_spec import DomainSpec, parse_domain_spec

        spec = parse_domain_spec(FIXTURES_DIR / "z2_domain.json")

        assert isinstance(spec, DomainSpec)
        assert spec.domain_id == "Z2"
        assert spec.elements == ("0", "1")
        assert spec.identity == "0"
        assert spec.mul_table["0"]["1"] == "1"
        assert spec.mul_table["1"]["0"] == "1"

    def test_parse_z3_domain(self):
        """Parse a valid Z3 domain specification."""
        from normalization.domain_spec import DomainSpec, parse_domain_spec

        spec = parse_domain_spec(FIXTURES_DIR / "z3_domain.json")

        assert isinstance(spec, DomainSpec)
        assert spec.domain_id == "Z3"
        assert len(spec.elements) == 3
        assert spec.identity == "0"
        # Z3 addition: 1 + 2 = 0
        assert spec.mul_table["1"]["2"] == "0"

    def test_elements_order_preserved(self):
        """Elements list order must be preserved for deterministic enumeration."""
        from normalization.domain_spec import parse_domain_spec

        spec = parse_domain_spec(FIXTURES_DIR / "z2_domain.json")
        assert spec.elements == ("0", "1"), "Elements order must match fixture exactly"

    def test_identity_must_be_in_elements(self):
        """Identity constant must reference an element in the domain."""
        from normalization.domain_spec import parse_domain_spec

        spec = parse_domain_spec(FIXTURES_DIR / "z2_domain.json")
        assert spec.identity in spec.elements

    def test_missing_elements_fails(self, tmp_path):
        """Domain spec without 'elements' field must raise ValueError."""
        from normalization.domain_spec import parse_domain_spec

        bad_spec = {"domain_id": "Bad", "constants": {"identity": "e"}, "functions": {"mul": {}}}
        bad_file = tmp_path / "bad_spec.json"
        bad_file.write_text(json.dumps(bad_spec))

        with pytest.raises(ValueError, match="elements"):
            parse_domain_spec(bad_file)

    def test_missing_identity_fails(self, tmp_path):
        """Domain spec without identity constant must raise ValueError."""
        from normalization.domain_spec import parse_domain_spec

        bad_spec = {"domain_id": "Bad", "elements": ["a", "b"], "constants": {}, "functions": {"mul": {}}}
        bad_file = tmp_path / "bad_spec.json"
        bad_file.write_text(json.dumps(bad_spec))

        with pytest.raises(ValueError, match="identity"):
            parse_domain_spec(bad_file)

    def test_identity_not_in_elements_fails(self, tmp_path):
        """Identity constant referencing non-existent element must fail."""
        from normalization.domain_spec import parse_domain_spec

        bad_spec = {
            "domain_id": "Bad",
            "elements": ["a", "b"],
            "constants": {"identity": "c"},  # c not in elements
            "functions": {"mul": {}},
        }
        bad_file = tmp_path / "bad_spec.json"
        bad_file.write_text(json.dumps(bad_spec))

        with pytest.raises(ValueError, match="identity.*not in elements"):
            parse_domain_spec(bad_file)

    def test_incomplete_mul_table_fails(self, tmp_path):
        """Cayley table with missing entries must fail validation."""
        from normalization.domain_spec import parse_domain_spec

        bad_spec = {
            "domain_id": "Bad",
            "elements": ["a", "b"],
            "constants": {"identity": "a"},
            "functions": {
                "mul": {
                    "a": {"a": "a", "b": "b"},
                    # "b" row missing
                }
            },
        }
        bad_file = tmp_path / "bad_spec.json"
        bad_file.write_text(json.dumps(bad_spec))

        with pytest.raises(ValueError, match="mul.*incomplete|missing"):
            parse_domain_spec(bad_file)

    def test_mul_table_invalid_element_fails(self, tmp_path):
        """Cayley table with invalid element reference must fail."""
        from normalization.domain_spec import parse_domain_spec

        bad_spec = {
            "domain_id": "Bad",
            "elements": ["a", "b"],
            "constants": {"identity": "a"},
            "functions": {
                "mul": {
                    "a": {"a": "a", "b": "c"},  # c not in elements
                    "b": {"a": "b", "b": "a"},
                }
            },
        }
        bad_file = tmp_path / "bad_spec.json"
        bad_file.write_text(json.dumps(bad_spec))

        with pytest.raises(ValueError, match="not in elements|invalid element"):
            parse_domain_spec(bad_file)

    def test_domain_size_reported(self):
        """DomainSpec must report domain size correctly."""
        from normalization.domain_spec import parse_domain_spec

        spec = parse_domain_spec(FIXTURES_DIR / "z3_domain.json")
        assert len(spec) == 3
        # Alternative: assert spec.size == 3

    def test_pathlike_required(self):
        """parse_domain_spec() must accept only pathlike objects, not dicts."""
        from normalization.domain_spec import parse_domain_spec

        # Passing a dict should raise TypeError
        with pytest.raises(TypeError):
            parse_domain_spec({"domain_id": "Test", "elements": ["a"]})


class TestConstantResolution:
    """Tests for constant resolution via domain_spec.constants."""

    def test_resolve_identity_constant(self):
        """Constant 'identity' must resolve to element via domain_spec.constants."""
        from normalization.domain_spec import parse_domain_spec

        spec = parse_domain_spec(FIXTURES_DIR / "z2_domain.json")

        # resolve_constant() returns the element value
        assert spec.resolve_constant("identity") == "0"

    def test_resolve_named_constant(self):
        """Named constants (e.g., 'two') must resolve via domain_spec.constants."""
        from normalization.domain_spec import parse_domain_spec

        spec = parse_domain_spec(FIXTURES_DIR / "z3_domain.json")

        # Z3 has constants: {"identity": "0", "two": "2"}
        assert spec.resolve_constant("two") == "2"

    def test_unknown_constant_fails(self):
        """Unknown constant key must raise ValueError (fail-closed).

        NORMATIVE: Const ALWAYS resolves via domain_spec.constants.
        Direct element literals are NOT supported.
        """
        from normalization.domain_spec import parse_domain_spec

        spec = parse_domain_spec(FIXTURES_DIR / "z2_domain.json")

        # "1" is an element, but NOT a constant key
        with pytest.raises(ValueError, match="unknown constant|not in constants"):
            spec.resolve_constant("1")

    def test_all_constants_resolve_to_elements(self):
        """All constant values must be valid elements (enforced at parse time)."""
        from normalization.domain_spec import parse_domain_spec

        spec = parse_domain_spec(FIXTURES_DIR / "z3_domain.json")

        for const_name in spec.constants:
            value = spec.resolve_constant(const_name)
            assert value in spec.elements, f"Constant '{const_name}' resolved to '{value}' not in elements"

    def test_resolve_constant_value_must_be_element(self):
        """resolve_constant() MUST fail if resolved value is not in elements.

        Belt+suspenders: even if parse-time validation should prevent this,
        resolve_constant() must fail-closed at resolution time too.
        This prevents bugs from manually-constructed DomainSpec objects.
        """
        from normalization.domain_spec import DomainSpec

        # Manually construct a DomainSpec with invalid constant value
        # (bypassing parse_domain_spec validation)
        # Note: _constants is the internal field name (constants is a property)
        spec = DomainSpec(
            domain_id="Bad",
            elements=("0", "1"),
            _constants={"identity": "Z"},  # "Z" is NOT in elements
            mul_table={"0": {"0": "0", "1": "1"}, "1": {"0": "1", "1": "0"}},
        )

        # resolve_constant() must fail-closed even though constant key exists
        with pytest.raises(ValueError, match="not in elements|invalid.*element"):
            spec.resolve_constant("identity")

    def test_constants_dict_is_immutable(self):
        """DomainSpec.constants must not be mutated after construction.

        Prevents accidental mutation that could break determinism guarantees.
        """
        from normalization.domain_spec import parse_domain_spec

        spec = parse_domain_spec(FIXTURES_DIR / "z2_domain.json")

        # Attempting to mutate constants should fail
        # (either TypeError from frozen dict, or AttributeError from property)
        with pytest.raises((TypeError, AttributeError)):
            spec.constants["evil"] = "1"


class TestDomainSpecLimits:
    """Tests for domain size limit enforcement."""

    def test_large_domain_triggers_abstention(self):
        """Domain with >50 elements should be flagged for abstention."""
        from normalization.domain_spec import parse_domain_spec

        spec = parse_domain_spec(FIXTURES_DIR / "large_100.json")

        assert spec.exceeds_enumeration_limit()
        # Or: assert spec.size > 50
