"""
FOL_FIN_EQ_v1 verification certificate generation.

This module generates structured certificates for FOL formula verification results.
Certificates provide auditable records of verification including:
- Full formula and domain specification
- Verification status (VERIFIED/REFUTED/ABSTAINED)
- Witnesses for existential quantifiers
- Counterexamples for refutations
- Resource limits for ABSTAINED cases

NORMATIVE INVARIANTS:
- Certificate hashing uses DOMAIN_FOL_CERT domain separation tag
- Certificates use canonical JSON (governance.registry_hash.canonicalize_json)
- checked_formula_ast_hash uses compute_ast_hash() from fol_ast
- Schema version is "v1.0.0"
- Logic fragment is "FOL_FIN_EQ_v1"
- Verification strategy is "exhaustive_enumeration"
"""

from __future__ import annotations

from typing import Any

from governance.registry_hash import canonicalize_json
from normalization.domain_spec import DomainSpec
from normalization.fol_ast import FolAst, compute_ast_hash, _ast_to_dict
from normalization.fol_fin_eq import VerificationResult
from substrate.crypto.hashing import DOMAIN_FOL_CERT, sha256_hex


# =============================================================================
# Constants
# =============================================================================

SCHEMA_VERSION = "v1.0.0"
LOGIC_FRAGMENT = "FOL_FIN_EQ_v1"
VERIFICATION_STRATEGY = "exhaustive_enumeration"


# =============================================================================
# Public API
# =============================================================================


def generate_certificate(
    domain: DomainSpec,
    formula: FolAst,
    result: VerificationResult
) -> dict[str, Any]:
    """Generate a verification certificate from verification result.

    Creates a structured certificate containing all information needed
    for audit and verification replay.

    Args:
        domain: The domain specification used for verification
        formula: The formula that was verified
        result: The verification result from verify_fol_fin_eq()

    Returns:
        Certificate dict with all required fields
    """
    # Required fields present in all certificates
    cert: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "logic_fragment": LOGIC_FRAGMENT,
        "verification_strategy": VERIFICATION_STRATEGY,
        "domain_spec": _serialize_domain_spec(domain),
        "checked_formula": _ast_to_dict(formula),
        "checked_formula_ast_hash": compute_ast_hash(formula),
        "status": result.status,
        "quantifier_report": result.quantifier_report,
    }

    # Status-conditional fields
    if result.status == "VERIFIED":
        # Include witnesses for existential quantifiers if present
        if result.witnesses is not None:
            cert["witnesses"] = result.witnesses

    elif result.status == "REFUTED":
        # Include counterexample
        if result.counterexample is not None:
            cert["counterexample"] = result.counterexample

    elif result.status == "ABSTAINED":
        # Include resource limits for audit transparency
        if result.resource_limits is not None:
            cert["resource_limits"] = {
                "max_domain_size": result.resource_limits.max_domain_size,
                "max_assignments": result.resource_limits.max_assignments,
                "max_quantifier_depth": result.resource_limits.max_quantifier_depth,
                "computed_estimate": result.resource_limits.computed_estimate,
                "actual_domain_size": result.resource_limits.actual_domain_size,
                "actual_quantifier_depth": result.resource_limits.actual_quantifier_depth,
            }
        if result.resource_limit_reason is not None:
            cert["resource_limit_reason"] = result.resource_limit_reason

    return cert


def compute_certificate_hash(cert: dict[str, Any]) -> str:
    """Compute canonical hash of a verification certificate.

    Uses:
    - canonicalize_json from governance.registry_hash
    - sha256_hex with DOMAIN_FOL_CERT domain separation

    NORMATIVE: This hash differs from naive hashlib.sha256() due to
    domain separation prefix. This prevents type confusion attacks.

    Args:
        cert: Certificate dict to hash

    Returns:
        64-character hex hash string
    """
    canonical_bytes = canonicalize_json(cert).encode("utf-8")
    return sha256_hex(canonical_bytes, domain=DOMAIN_FOL_CERT)


# =============================================================================
# Internal Helpers
# =============================================================================


def _serialize_domain_spec(domain: DomainSpec) -> dict[str, Any]:
    """Serialize DomainSpec to dict for certificate inclusion.

    Includes only the fields needed for certificate purposes.

    Args:
        domain: Domain specification

    Returns:
        Dict with domain_id and elements
    """
    return {
        "domain_id": domain.domain_id,
        "elements": list(domain.elements),
    }
