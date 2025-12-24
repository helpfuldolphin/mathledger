"""
Tests for FOL certificate schema validation (fail-closed).

Phase 2 RED: These tests will fail until governance/fol_schema_validator.py is implemented.

Tests schema validation including:
- Valid certificate passes
- Missing required fields fail
- Invalid status values fail
- REFUTED without counterexample fails
- ABSTAINED without resource_limits fails
"""

import pytest

# Import module that doesn't exist yet (will fail with ModuleNotFoundError)
from governance.fol_schema_validator import validate_fol_certificate, ValidationResult


class TestSchemaValidation:
    """Schema validation tests."""

    @pytest.fixture
    def valid_cert(self):
        return {
            "schema_version": "v1.0.0",
            "logic_fragment": "FOL_FIN_EQ_v1",
            "domain_spec": {"domain_id": "Z2", "elements": ["0", "1"]},
            "checked_formula": "∀x. e*x=x",
            "checked_formula_ast_hash": "a" * 64,
            "status": "VERIFIED",
            "quantifier_report": {"forall_vars": ["x"], "exists_vars": [], "quantifier_depth": 1},
            "verification_strategy": "exhaustive_enumeration"
        }

    def test_valid_certificate_passes(self, valid_cert):
        """Complete valid certificate passes validation."""
        result = validate_fol_certificate(valid_cert)
        assert result.valid is True
        assert result.errors == ()

    def test_missing_domain_spec_fails(self, valid_cert):
        """Certificate without domain_spec fails."""
        del valid_cert["domain_spec"]
        result = validate_fol_certificate(valid_cert)
        assert result.valid is False
        assert any("domain_spec" in e for e in result.errors)

    def test_missing_quantifier_report_fails(self, valid_cert):
        """Certificate without quantifier_report fails."""
        del valid_cert["quantifier_report"]
        result = validate_fol_certificate(valid_cert)
        assert result.valid is False
        assert any("quantifier_report" in e for e in result.errors)

    def test_invalid_status_fails(self, valid_cert):
        """Certificate with invalid status fails."""
        valid_cert["status"] = "INVALID_STATUS"
        result = validate_fol_certificate(valid_cert)
        assert result.valid is False
        assert any("status" in e for e in result.errors)

    def test_missing_counterexample_for_refuted_fails(self, valid_cert):
        """REFUTED certificate without counterexample fails."""
        valid_cert["status"] = "REFUTED"
        result = validate_fol_certificate(valid_cert)
        assert result.valid is False
        assert any("counterexample" in e for e in result.errors)

    def test_wrong_verification_strategy_fails(self, valid_cert):
        """Non-exhaustive strategy for FOL_FIN_EQ_v1 fails."""
        valid_cert["verification_strategy"] = "smt_z3"
        result = validate_fol_certificate(valid_cert)
        assert result.valid is False
        assert any("verification_strategy" in e for e in result.errors)

    def test_abstained_without_resource_limits_fails(self):
        """ABSTAINED certificate without resource_limits fails validation."""
        cert = {
            "schema_version": "v1.0.0",
            "logic_fragment": "FOL_FIN_EQ_v1",
            "domain_spec": {"domain_id": "D100", "elements": ["e0"]},
            "checked_formula": "∀x. P(x)",
            "checked_formula_ast_hash": "a" * 64,
            "status": "ABSTAINED",
            "resource_limit_reason": "DOMAIN_SIZE_EXCEEDS_LIMIT",
            # Missing resource_limits!
            "quantifier_report": {"forall_vars": ["x"], "exists_vars": [], "quantifier_depth": 1},
            "verification_strategy": "exhaustive_enumeration"
        }
        result = validate_fol_certificate(cert)
        assert result.valid is False
        assert any("resource_limits" in e for e in result.errors)

    def test_abstained_with_resource_limits_passes(self):
        """ABSTAINED certificate with resource_limits passes validation."""
        cert = {
            "schema_version": "v1.0.0",
            "logic_fragment": "FOL_FIN_EQ_v1",
            "domain_spec": {"domain_id": "D100", "elements": ["e0"]},
            "checked_formula": "∀x. P(x)",
            "checked_formula_ast_hash": "a" * 64,
            "status": "ABSTAINED",
            "resource_limit_reason": "DOMAIN_SIZE_EXCEEDS_LIMIT",
            "resource_limits": {
                "max_domain_size": 50,
                "max_assignments": 125000,
                "max_quantifier_depth": 5,
                "computed_estimate": 0,
                "actual_domain_size": 100,
                "actual_quantifier_depth": 1
            },
            "quantifier_report": {"forall_vars": ["x"], "exists_vars": [], "quantifier_depth": 1},
            "verification_strategy": "exhaustive_enumeration"
        }
        result = validate_fol_certificate(cert)
        assert result.valid is True
