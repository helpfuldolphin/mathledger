#!/usr/bin/env python3
"""
FOL_FIN_EQ_v1 Verification Demo Runner.

Produces self-contained evidence packs for FOL formula verification.
Each pack includes:
- manifest.json: Metadata and artifact hashes
- certificates/: Verification certificates
- verify.py: Standalone verification script

Usage:
    python scripts/run_fol_fin_eq_demo.py --domain z2 --output demo_output/
    python scripts/run_fol_fin_eq_demo.py --domain z2_broken --output demo_broken/
    python scripts/run_fol_fin_eq_demo.py --domain large_100 --output demo_large/

Exit codes:
    0: PASS - All operations successful
    1: FAIL - Verification or validation failure
    3: ERROR - Infrastructure error
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Add repo root to path for imports
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from governance.fol_schema_validator import validate_fol_certificate
from governance.registry_hash import canonicalize_json
from normalization.domain_spec import parse_domain_spec
from normalization.fol_ast import parse_fol_formula
from normalization.fol_certificate import compute_certificate_hash, generate_certificate
from normalization.fol_fin_eq import verify_fol_fin_eq

# =============================================================================
# Constants
# =============================================================================

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_ERROR = 3

FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures" / "fol"

# Domain name to fixture file mapping
DOMAIN_FIXTURES = {
    "z2": "z2_domain.json",
    "z3": "z3_domain.json",
    "z2_broken": "z2_broken.json",
    "z2_nonassoc": "z2_nonassoc.json",
    "large_100": "large_100.json",
    "d20": "d20_domain.json",
}

# Group axiom formulas to verify
GROUP_AXIOM_FORMULAS = [
    "identity_formula.json",
    "inverse_formula.json",
    "associativity_formula.json",
]


# =============================================================================
# Public API
# =============================================================================


def run_demo(
    domain_name: str,
    output_dir: Path,
    formulas: list[str] | None = None
) -> int:
    """Run the FOL verification demo and produce evidence pack.

    Args:
        domain_name: Name of domain (z2, z2_broken, large_100, etc.)
        output_dir: Directory to write output files
        formulas: Optional list of formula fixture names. Defaults to group axioms.

    Returns:
        Exit code (0=PASS, 1=FAIL, 3=ERROR)
    """
    if formulas is None:
        formulas = GROUP_AXIOM_FORMULAS

    # Resolve domain fixture
    if domain_name not in DOMAIN_FIXTURES:
        print(f"ERROR: Unknown domain '{domain_name}'", file=sys.stderr)
        print(f"Available: {sorted(DOMAIN_FIXTURES.keys())}", file=sys.stderr)
        return EXIT_ERROR

    domain_path = FIXTURES_DIR / DOMAIN_FIXTURES[domain_name]
    if not domain_path.exists():
        print(f"ERROR: Domain fixture not found: {domain_path}", file=sys.stderr)
        return EXIT_ERROR

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    certs_dir = output_dir / "certificates"
    certs_dir.mkdir(exist_ok=True)

    # Load domain
    try:
        domain = parse_domain_spec(domain_path)
    except Exception as e:
        print(f"ERROR: Failed to load domain: {e}", file=sys.stderr)
        return EXIT_ERROR

    # Process each formula
    artifacts: list[dict[str, Any]] = []
    all_verified = True

    for formula_file in formulas:
        formula_path = FIXTURES_DIR / formula_file
        if not formula_path.exists():
            print(f"WARNING: Formula not found: {formula_path}", file=sys.stderr)
            continue

        # Load formula
        with open(formula_path, encoding="utf-8") as f:
            formula_data = json.load(f)

        formula = parse_fol_formula(formula_data["formula"])
        description = formula_data.get("description", formula_file)

        # Verify
        try:
            result = verify_fol_fin_eq(domain, formula)
        except Exception as e:
            print(f"ERROR: Verification failed for {formula_file}: {e}", file=sys.stderr)
            return EXIT_ERROR

        # Generate certificate
        cert = generate_certificate(domain, formula, result)

        # Validate certificate schema
        validation = validate_fol_certificate(cert)
        if not validation.valid:
            print(f"ERROR: Certificate validation failed: {validation.errors}", file=sys.stderr)
            return EXIT_FAIL

        # Compute certificate hash
        cert_hash = compute_certificate_hash(cert)

        # Create artifact ID (deterministic)
        artifact_id = f"{domain_name}_{Path(formula_file).stem}"

        # Write certificate
        cert_filename = f"{artifact_id}.json"
        cert_path = certs_dir / cert_filename
        with open(cert_path, "w", encoding="utf-8") as f:
            json.dump(cert, f, indent=2, sort_keys=True)

        # Record artifact
        artifacts.append({
            "artifact_id": artifact_id,
            "formula": formula_file,
            "description": description,
            "status": result.status,
            "certificate_file": f"certificates/{cert_filename}",
            "certificate_sha256": cert_hash,
        })

        status_symbol = {"VERIFIED": "[OK]", "REFUTED": "[FAIL]", "ABSTAINED": "[SKIP]"}.get(result.status, "[?]")
        print(f"  {status_symbol} {artifact_id}: {result.status}")

        if result.status == "REFUTED":
            all_verified = False

    # Sort artifacts by artifact_id for determinism
    artifacts.sort(key=lambda a: a["artifact_id"])

    # Generate manifest
    manifest = generate_manifest(domain_name, domain, artifacts)

    # Write manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    # Generate verify.py script
    verify_script = generate_verify_script()
    verify_path = output_dir / "verify.py"
    with open(verify_path, "w", encoding="utf-8") as f:
        f.write(verify_script)

    print(f"\nEvidence pack written to: {output_dir}")
    print(f"  manifest.json: {manifest_path}")
    print(f"  certificates/: {len(artifacts)} files")
    print(f"  verify.py: standalone verifier")

    return EXIT_PASS


def generate_manifest(
    domain_name: str,
    domain: Any,
    artifacts: list[dict[str, Any]]
) -> dict[str, Any]:
    """Generate the manifest.json content.

    Args:
        domain_name: Name of the domain used
        domain: DomainSpec instance
        artifacts: List of artifact metadata dicts

    Returns:
        Manifest dict
    """
    # Note: created_at is intentionally omitted for determinism
    # (byte-identical manifests for identical inputs)
    return {
        "schema_version": "v1.0.0",
        "logic_fragment": "FOL_FIN_EQ_v1",
        "domain": {
            "name": domain_name,
            "domain_id": domain.domain_id,
            "element_count": len(domain),
        },
        "artifacts": artifacts,
        "verification_strategy": "exhaustive_enumeration",
    }


def generate_verify_script() -> str:
    """Generate the standalone verify.py script content.

    Returns:
        Python script content as string
    """
    return '''#!/usr/bin/env python3
"""
Standalone verification script for FOL_FIN_EQ_v1 evidence pack.

Verifies:
1. Manifest schema validity
2. Certificate hash integrity
3. Certificate schema validity

Exit codes:
    0: PASS - All checks passed
    1: FAIL - Integrity or validation failure
"""

import hashlib
import json
import sys
from pathlib import Path


def canonicalize_json(data):
    """Canonical JSON serialization."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def compute_cert_hash(cert):
    """Compute certificate hash with domain separation."""
    DOMAIN_FOL_CERT = b"\\x09"
    canonical = canonicalize_json(cert).encode("utf-8")
    return hashlib.sha256(DOMAIN_FOL_CERT + canonical).hexdigest()


def validate_manifest(manifest):
    """Validate manifest schema."""
    errors = []
    required = ["schema_version", "logic_fragment", "domain", "artifacts", "verification_strategy"]
    for field in required:
        if field not in manifest:
            errors.append(f"Missing required field: {field}")

    if manifest.get("logic_fragment") != "FOL_FIN_EQ_v1":
        errors.append(f"Invalid logic_fragment: expected FOL_FIN_EQ_v1")

    if manifest.get("verification_strategy") != "exhaustive_enumeration":
        errors.append(f"Invalid verification_strategy: expected exhaustive_enumeration")

    return errors


def validate_certificate(cert):
    """Validate certificate schema."""
    errors = []
    required = ["schema_version", "logic_fragment", "domain_spec", "checked_formula",
                "checked_formula_ast_hash", "status", "quantifier_report", "verification_strategy"]
    for field in required:
        if field not in cert:
            errors.append(f"Missing required field: {field}")

    if cert.get("logic_fragment") != "FOL_FIN_EQ_v1":
        errors.append("Invalid logic_fragment")

    if cert.get("verification_strategy") != "exhaustive_enumeration":
        errors.append("Invalid verification_strategy")

    status = cert.get("status")
    if status not in ("VERIFIED", "REFUTED", "ABSTAINED"):
        errors.append(f"Invalid status: {status}")

    if status == "REFUTED" and "counterexample" not in cert:
        errors.append("REFUTED certificate missing counterexample")

    if status == "ABSTAINED" and "resource_limits" not in cert:
        errors.append("ABSTAINED certificate missing resource_limits")

    return errors


def main():
    script_dir = Path(__file__).parent
    manifest_path = script_dir / "manifest.json"

    if not manifest_path.exists():
        print("FAIL: manifest.json not found", file=sys.stderr)
        return 1

    manifest = json.loads(manifest_path.read_text())

    # Validate manifest
    errors = validate_manifest(manifest)
    if errors:
        for e in errors:
            print(f"FAIL: Manifest validation: {e}", file=sys.stderr)
        return 1

    # Validate each certificate
    for artifact in manifest.get("artifacts", []):
        cert_file = artifact.get("certificate_file")
        expected_hash = artifact.get("certificate_sha256")

        cert_path = script_dir / cert_file
        if not cert_path.exists():
            print(f"FAIL: Certificate not found: {cert_file}", file=sys.stderr)
            return 1

        cert = json.loads(cert_path.read_text())

        # Validate certificate schema
        cert_errors = validate_certificate(cert)
        if cert_errors:
            for e in cert_errors:
                print(f"FAIL: Certificate {cert_file}: {e}", file=sys.stderr)
            return 1

        # Verify hash
        actual_hash = compute_cert_hash(cert)
        if actual_hash != expected_hash:
            print(f"FAIL: Hash mismatch for {cert_file}", file=sys.stderr)
            print(f"  Expected: {expected_hash}", file=sys.stderr)
            print(f"  Actual:   {actual_hash}", file=sys.stderr)
            return 1

    print("PASS: All certificates verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


# =============================================================================
# CLI
# =============================================================================


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="FOL_FIN_EQ_v1 Verification Demo Runner"
    )
    parser.add_argument(
        "--domain",
        required=True,
        help=f"Domain name ({', '.join(DOMAIN_FIXTURES.keys())})"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for evidence pack"
    )
    parser.add_argument(
        "--formulas",
        nargs="+",
        help="Formula fixture names (default: group axioms)"
    )

    args = parser.parse_args()

    print(f"FOL_FIN_EQ_v1 Demo: {args.domain}")
    print(f"Output: {args.output}")
    print()

    exit_code = run_demo(args.domain, Path(args.output), args.formulas)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
