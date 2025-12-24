"""
FOL certificate schema validation (fail-closed).

This module provides schema validation for FOL_FIN_EQ_v1 verification certificates.
All validation is fail-closed: certificates must have all required fields and
meet all constraints to pass validation.

NORMATIVE INVARIANTS:
- VERIFIED certificates may optionally have witnesses
- REFUTED certificates MUST have counterexample
- ABSTAINED certificates MUST have resource_limits and resource_limit_reason
- verification_strategy MUST be "exhaustive_enumeration" for FOL_FIN_EQ_v1
- schema_version MUST be "v1.0.0"
- logic_fragment MUST be "FOL_FIN_EQ_v1"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# =============================================================================
# Constants
# =============================================================================

VALID_STATUSES = frozenset({"VERIFIED", "REFUTED", "ABSTAINED"})
REQUIRED_SCHEMA_VERSION = "v1.0.0"
REQUIRED_LOGIC_FRAGMENT = "FOL_FIN_EQ_v1"
REQUIRED_VERIFICATION_STRATEGY = "exhaustive_enumeration"

# Required fields present in ALL certificates
REQUIRED_FIELDS = (
    "schema_version",
    "logic_fragment",
    "domain_spec",
    "checked_formula",
    "checked_formula_ast_hash",
    "status",
    "quantifier_report",
    "verification_strategy",
)


# =============================================================================
# Result Type
# =============================================================================


@dataclass(frozen=True)
class ValidationResult:
    """Result of certificate schema validation.

    Attributes:
        valid: True if certificate passes all validation checks
        errors: Tuple of error messages (empty if valid)
    """
    valid: bool
    errors: tuple[str, ...]


# =============================================================================
# Public API
# =============================================================================


def validate_fol_certificate(cert: dict[str, Any]) -> ValidationResult:
    """Validate a FOL_FIN_EQ_v1 verification certificate.

    Performs fail-closed validation:
    1. All required fields must be present
    2. Field values must meet constraints
    3. Status-conditional fields must be present

    Args:
        cert: Certificate dict to validate

    Returns:
        ValidationResult with valid flag and any error messages
    """
    errors: list[str] = []

    # Phase 1: Check required fields
    for field in REQUIRED_FIELDS:
        if field not in cert:
            errors.append(f"Missing required field: {field}")

    # If missing required fields, return early (can't validate further)
    if errors:
        return ValidationResult(valid=False, errors=tuple(errors))

    # Phase 2: Validate field values
    _validate_schema_version(cert, errors)
    _validate_logic_fragment(cert, errors)
    _validate_status(cert, errors)
    _validate_verification_strategy(cert, errors)
    _validate_domain_spec(cert, errors)
    _validate_quantifier_report(cert, errors)
    _validate_ast_hash(cert, errors)

    # Phase 3: Status-conditional validation
    status = cert.get("status")
    if status == "REFUTED":
        _validate_refuted_fields(cert, errors)
    elif status == "ABSTAINED":
        _validate_abstained_fields(cert, errors)
    # VERIFIED: witnesses are optional, no additional validation needed

    if errors:
        return ValidationResult(valid=False, errors=tuple(errors))

    return ValidationResult(valid=True, errors=())


# =============================================================================
# Field Validators
# =============================================================================


def _validate_schema_version(cert: dict[str, Any], errors: list[str]) -> None:
    """Validate schema_version field."""
    version = cert.get("schema_version")
    if version != REQUIRED_SCHEMA_VERSION:
        errors.append(
            f"Invalid schema_version: expected '{REQUIRED_SCHEMA_VERSION}', got '{version}'"
        )


def _validate_logic_fragment(cert: dict[str, Any], errors: list[str]) -> None:
    """Validate logic_fragment field."""
    fragment = cert.get("logic_fragment")
    if fragment != REQUIRED_LOGIC_FRAGMENT:
        errors.append(
            f"Invalid logic_fragment: expected '{REQUIRED_LOGIC_FRAGMENT}', got '{fragment}'"
        )


def _validate_status(cert: dict[str, Any], errors: list[str]) -> None:
    """Validate status field."""
    status = cert.get("status")
    if status not in VALID_STATUSES:
        errors.append(
            f"Invalid status: expected one of {sorted(VALID_STATUSES)}, got '{status}'"
        )


def _validate_verification_strategy(cert: dict[str, Any], errors: list[str]) -> None:
    """Validate verification_strategy field."""
    strategy = cert.get("verification_strategy")
    if strategy != REQUIRED_VERIFICATION_STRATEGY:
        errors.append(
            f"Invalid verification_strategy: expected '{REQUIRED_VERIFICATION_STRATEGY}', "
            f"got '{strategy}'"
        )


def _validate_domain_spec(cert: dict[str, Any], errors: list[str]) -> None:
    """Validate domain_spec structure."""
    domain_spec = cert.get("domain_spec")
    if not isinstance(domain_spec, dict):
        errors.append("domain_spec must be a dict")
        return

    if "domain_id" not in domain_spec:
        errors.append("domain_spec missing 'domain_id' field")

    if "elements" not in domain_spec:
        errors.append("domain_spec missing 'elements' field")
    elif not isinstance(domain_spec.get("elements"), list):
        errors.append("domain_spec.elements must be a list")


def _validate_quantifier_report(cert: dict[str, Any], errors: list[str]) -> None:
    """Validate quantifier_report structure."""
    qreport = cert.get("quantifier_report")
    if not isinstance(qreport, dict):
        errors.append("quantifier_report must be a dict")
        return

    # Required fields in quantifier_report
    if "forall_vars" not in qreport:
        errors.append("quantifier_report missing 'forall_vars' field")
    if "exists_vars" not in qreport:
        errors.append("quantifier_report missing 'exists_vars' field")


def _validate_ast_hash(cert: dict[str, Any], errors: list[str]) -> None:
    """Validate checked_formula_ast_hash format."""
    ast_hash = cert.get("checked_formula_ast_hash")
    if not isinstance(ast_hash, str):
        errors.append("checked_formula_ast_hash must be a string")
        return

    if len(ast_hash) != 64:
        errors.append(f"checked_formula_ast_hash must be 64 chars, got {len(ast_hash)}")
        return

    if not all(c in "0123456789abcdef" for c in ast_hash.lower()):
        errors.append("checked_formula_ast_hash must be valid hex")


# =============================================================================
# Status-Conditional Validators
# =============================================================================


def _validate_refuted_fields(cert: dict[str, Any], errors: list[str]) -> None:
    """Validate fields required for REFUTED status."""
    if "counterexample" not in cert:
        errors.append("REFUTED certificate must have 'counterexample' field")
        return

    cex = cert.get("counterexample")
    if not isinstance(cex, dict):
        errors.append("counterexample must be a dict")
        return

    if "assignment" not in cex:
        errors.append("counterexample must have 'assignment' field")


def _validate_abstained_fields(cert: dict[str, Any], errors: list[str]) -> None:
    """Validate fields required for ABSTAINED status."""
    if "resource_limits" not in cert:
        errors.append("ABSTAINED certificate must have 'resource_limits' field")
    else:
        _validate_resource_limits(cert.get("resource_limits"), errors)

    if "resource_limit_reason" not in cert:
        errors.append("ABSTAINED certificate must have 'resource_limit_reason' field")


def _validate_resource_limits(rl: Any, errors: list[str]) -> None:
    """Validate resource_limits structure."""
    if not isinstance(rl, dict):
        errors.append("resource_limits must be a dict")
        return

    required_rl_fields = (
        "max_domain_size",
        "max_assignments",
        "max_quantifier_depth",
        "computed_estimate",
        "actual_domain_size",
        "actual_quantifier_depth",
    )

    for field in required_rl_fields:
        if field not in rl:
            errors.append(f"resource_limits missing '{field}' field")
