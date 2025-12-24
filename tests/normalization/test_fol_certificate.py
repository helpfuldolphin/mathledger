"""
Tests for FOL certificate generation.

Phase 2 RED: These tests will fail until normalization/fol_certificate.py is implemented.

Tests certificate generation including:
- Required fields presence
- Hash determinism
- Witness/counterexample inclusion
- Resource limits for ABSTAINED
- Domain separation via DOMAIN_FOL_CERT
"""

import pytest
from pathlib import Path

# Import modules that exist (Phase 1)
from normalization.domain_spec import parse_domain_spec
from normalization.fol_ast import parse_fol_formula

# Import modules that don't exist yet (will fail with ModuleNotFoundError)
from normalization.fol_fin_eq import verify_fol_fin_eq
from normalization.fol_certificate import generate_certificate, compute_certificate_hash

FIXTURES = Path(__file__).parent.parent / "fixtures" / "fol"


class TestCertificateGeneration:
    """Certificate generation tests."""

    def test_certificate_has_required_fields(self):
        """Generated certificate has all required fields."""
        domain = parse_domain_spec(FIXTURES / "z2_domain.json")
        formula = parse_fol_formula((FIXTURES / "identity_formula.json"))
        result = verify_fol_fin_eq(domain, formula)
        cert = generate_certificate(domain, formula, result)

        required = ["schema_version", "logic_fragment", "domain_spec",
                    "checked_formula", "checked_formula_ast_hash", "status",
                    "quantifier_report", "verification_strategy"]
        for field in required:
            assert field in cert, f"Missing field: {field}"

    def test_certificate_hash_determinism(self):
        """Same certificate produces identical hash."""
        domain = parse_domain_spec(FIXTURES / "z2_domain.json")
        formula = parse_fol_formula((FIXTURES / "identity_formula.json"))
        result = verify_fol_fin_eq(domain, formula)
        cert = generate_certificate(domain, formula, result)
        hash1 = compute_certificate_hash(cert)
        hash2 = compute_certificate_hash(cert)
        assert hash1 == hash2

    def test_quantifier_report_preserved(self):
        """Quantifier report in certificate matches formula structure."""
        domain = parse_domain_spec(FIXTURES / "z2_domain.json")
        formula = parse_fol_formula((FIXTURES / "inverse_formula.json"))
        result = verify_fol_fin_eq(domain, formula)
        cert = generate_certificate(domain, formula, result)
        assert cert["quantifier_report"]["forall_vars"] == ["x"]
        assert cert["quantifier_report"]["exists_vars"] == ["y"]

    def test_witnesses_for_exists(self):
        """VERIFIED certificate with ∃ has witnesses."""
        domain = parse_domain_spec(FIXTURES / "z2_domain.json")
        formula = parse_fol_formula((FIXTURES / "inverse_formula.json"))
        result = verify_fol_fin_eq(domain, formula)
        cert = generate_certificate(domain, formula, result)
        assert cert["status"] == "VERIFIED"
        assert "witnesses" in cert
        assert cert["witnesses"] is not None

    def test_counterexample_for_refuted(self):
        """REFUTED certificate has counterexample."""
        domain = parse_domain_spec(FIXTURES / "z2_broken.json")
        formula = parse_fol_formula((FIXTURES / "associativity_formula.json"))
        result = verify_fol_fin_eq(domain, formula)
        cert = generate_certificate(domain, formula, result)
        assert cert["status"] == "REFUTED"
        assert "counterexample" in cert
        assert cert["counterexample"]["assignment"] is not None

    def test_abstained_has_resource_limits(self):
        """ABSTAINED certificate has resource_limits for audit transparency."""
        domain = parse_domain_spec(FIXTURES / "d20_domain.json")
        formula = parse_fol_formula((FIXTURES / "four_var_formula.json"))
        result = verify_fol_fin_eq(domain, formula)
        cert = generate_certificate(domain, formula, result)
        assert cert["status"] == "ABSTAINED"
        assert "resource_limits" in cert
        rl = cert["resource_limits"]
        assert rl["max_domain_size"] == 50
        assert rl["max_assignments"] == 125000
        assert rl["max_quantifier_depth"] == 5
        assert rl["computed_estimate"] == 160000
        assert rl["actual_domain_size"] == 20
        assert rl["actual_quantifier_depth"] == 4

    def test_certificate_hash_uses_repo_canonicalize_json(self):
        """Certificate hash MUST use governance.registry_hash.canonicalize_json().

        This test verifies that compute_certificate_hash() produces the same
        result as manually hashing with the repo's sha256_hex() helper.
        """
        from governance.registry_hash import canonicalize_json
        from substrate.crypto.hashing import sha256_hex, DOMAIN_FOL_CERT

        domain = parse_domain_spec(FIXTURES / "z2_domain.json")
        formula = parse_fol_formula((FIXTURES / "identity_formula.json"))
        result = verify_fol_fin_eq(domain, formula)
        cert = generate_certificate(domain, formula, result)

        # Compute via the function
        hash_from_func = compute_certificate_hash(cert)
        # Compute manually using repo's sha256_hex helper
        canonical_bytes = canonicalize_json(cert).encode("utf-8")
        hash_manual = sha256_hex(canonical_bytes, domain=DOMAIN_FOL_CERT)

        assert hash_from_func == hash_manual, \
            "compute_certificate_hash() must use sha256_hex() with DOMAIN_FOL_CERT"

    def test_cert_hash_uses_canonical_domain_hash_helper(self):
        """Certificate hash MUST use sha256_hex() from substrate.crypto.hashing.

        Verifies that the implementation uses the canonical helper, not raw hashlib.
        """
        import hashlib
        from governance.registry_hash import canonicalize_json
        from substrate.crypto.hashing import sha256_hex, DOMAIN_FOL_CERT

        domain = parse_domain_spec(FIXTURES / "z2_domain.json")
        formula = parse_fol_formula((FIXTURES / "identity_formula.json"))
        result = verify_fol_fin_eq(domain, formula)
        cert = generate_certificate(domain, formula, result)

        hash_from_func = compute_certificate_hash(cert)
        canonical_bytes = canonicalize_json(cert).encode("utf-8")

        # Verify it matches sha256_hex (the canonical helper)
        hash_via_helper = sha256_hex(canonical_bytes, domain=DOMAIN_FOL_CERT)
        assert hash_from_func == hash_via_helper

        # Verify it differs from naive hashlib usage
        hash_naive = hashlib.sha256(canonical_bytes).hexdigest()
        assert hash_from_func != hash_naive, \
            "compute_certificate_hash() must use domain prefix via sha256_hex()"

    def test_cert_hash_has_domain_prefix(self):
        """Certificate hash uses DOMAIN_FOL_CERT prefix for domain separation.

        Verifies the hash differs from a naive hash without domain prefix.
        """
        import hashlib
        from governance.registry_hash import canonicalize_json

        domain = parse_domain_spec(FIXTURES / "z2_domain.json")
        formula = parse_fol_formula((FIXTURES / "identity_formula.json"))
        result = verify_fol_fin_eq(domain, formula)
        cert = generate_certificate(domain, formula, result)

        hash_with_domain = compute_certificate_hash(cert)

        # Compute without domain prefix
        canonical_bytes = canonicalize_json(cert).encode("utf-8")
        hash_without_domain = hashlib.sha256(canonical_bytes).hexdigest()

        assert hash_with_domain != hash_without_domain, \
            "Cert hash must differ from naive hash (domain separation required)"

    def test_same_content_different_type_different_hash(self):
        """Same JSON content hashed as AST vs certificate produces different hashes.

        This is the core domain separation invariant: prevents type confusion.
        """
        from governance.registry_hash import canonicalize_json
        from substrate.crypto.hashing import sha256_hex, DOMAIN_FOL_AST, DOMAIN_FOL_CERT

        # Create identical JSON content
        content = {"type": "test", "value": "identical"}
        canonical_bytes = canonicalize_json(content).encode("utf-8")

        # Hash with different domain prefixes using canonical helper
        hash_as_ast = sha256_hex(canonical_bytes, domain=DOMAIN_FOL_AST)
        hash_as_cert = sha256_hex(canonical_bytes, domain=DOMAIN_FOL_CERT)

        assert hash_as_ast != hash_as_cert, \
            "Same content with different domain prefixes must produce different hashes"
