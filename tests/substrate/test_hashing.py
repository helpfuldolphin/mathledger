"""
Tests for substrate.crypto.hashing domain tag registration.

Phase 0.5a: RED - These tests MUST fail initially (tags not yet registered).
Phase 0.5b: GREEN - Tests pass after adding DOMAIN_FOL_AST and DOMAIN_FOL_CERT.
"""

import pytest


class TestDomainTagRegistration:
    """Verify FOL domain separation tags are properly registered."""

    def test_domain_tags_registered(self):
        """DOMAIN_FOL_AST (0x08) and DOMAIN_FOL_CERT (0x09) must exist."""
        from substrate.crypto.hashing import DOMAIN_FOL_AST, DOMAIN_FOL_CERT

        assert DOMAIN_FOL_AST == b'\x08', "DOMAIN_FOL_AST must be 0x08"
        assert DOMAIN_FOL_CERT == b'\x09', "DOMAIN_FOL_CERT must be 0x09"

    def test_domain_tags_unique_across_module(self):
        """All DOMAIN_* constants must be unique single-byte values."""
        import substrate.crypto.hashing as h

        # Collect all DOMAIN_* constants
        domain_tags = {}
        for name in dir(h):
            if name.startswith("DOMAIN_"):
                value = getattr(h, name)
                if isinstance(value, bytes):
                    # Enforce single-byte
                    assert len(value) == 1, f"{name} must be single byte, got {len(value)}"
                    domain_tags[value] = name

        # Check for collisions
        values = list(domain_tags.keys())
        if len(values) != len(set(values)):
            # Find the collision
            seen = {}
            for v, name in domain_tags.items():
                if v in seen:
                    pytest.fail(
                        f"Domain tag collision: {seen[v]} and {name} both use 0x{v.hex()}"
                    )
                seen[v] = name

        # Verify baseline tags are present (set difference check)
        baseline_tags = {
            'DOMAIN_LEAF', 'DOMAIN_NODE', 'DOMAIN_STMT', 'DOMAIN_BLCK',
            'DOMAIN_FED', 'DOMAIN_NODE_ATTEST', 'DOMAIN_DOSSIER', 'DOMAIN_ROOT',
            'DOMAIN_FOL_AST', 'DOMAIN_FOL_CERT',
        }
        found_names = set(domain_tags.values())
        missing = baseline_tags - found_names
        assert not missing, f"Missing baseline domain tags: {sorted(missing)}"

    def test_fol_tags_are_distinct_and_single_byte(self):
        """FOL tags must be distinct single-byte values."""
        from substrate.crypto.hashing import DOMAIN_FOL_AST, DOMAIN_FOL_CERT

        assert len(DOMAIN_FOL_AST) == 1, "DOMAIN_FOL_AST must be single byte"
        assert len(DOMAIN_FOL_CERT) == 1, "DOMAIN_FOL_CERT must be single byte"
        assert DOMAIN_FOL_AST != DOMAIN_FOL_CERT, "FOL_AST and FOL_CERT must be distinct"
