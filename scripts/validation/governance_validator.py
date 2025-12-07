#!/usr/bin/env python3
"""
Governance Validator

Validates governance artifacts against JSON schemas and RFC 8785 canonical form.
Enforces version consistency and detects drift.

Exit Codes:
  0 - PASS: All validations passed
  1 - FAIL: Validation error (schema violation, non-canonical form, version mismatch)
  3 - ERROR: Infrastructure failure (missing files, invalid JSON)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import hashlib

try:
    import jsonschema
except ImportError:
    print("❌ ERROR: jsonschema library not installed", file=sys.stderr)
    print("   Install with: pip3 install jsonschema", file=sys.stderr)
    sys.exit(3)

# Exit codes
EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_ERROR = 3


class GovernanceValidator:
    """Governance artifact validation engine."""

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.schemas_dir = repo_root / "schemas"
        self.errors = []

    def rfc8785_canonicalize(self, obj: Any) -> str:
        """
        Serialize an object to RFC 8785 canonical JSON.
        
        This is a simplified implementation. For production, consider using
        a library like `canonicaljson` or `jcs`.
        """
        return json.dumps(obj, ensure_ascii=True, sort_keys=True, separators=(',', ':'))

    def load_json(self, path: Path) -> Optional[Dict[str, Any]]:
        """Load and parse a JSON file."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            self.errors.append(f"File not found: {path}")
            return None
        except json.JSONDecodeError as e:
            self.errors.append(f"Invalid JSON in {path}: {e}")
            return None

    def load_schema(self, schema_name: str) -> Optional[Dict[str, Any]]:
        """Load a JSON schema by name."""
        schema_path = self.schemas_dir / f"{schema_name}.schema.json"
        return self.load_json(schema_path)

    def validate_schema(self, artifact: Dict[str, Any], schema: Dict[str, Any], artifact_path: Path) -> bool:
        """Validate an artifact against a JSON schema."""
        try:
            jsonschema.validate(instance=artifact, schema=schema)
            print(f"✅ Schema validation passed: {artifact_path.name}")
            return True
        except jsonschema.ValidationError as e:
            self.errors.append(f"Schema validation failed for {artifact_path}: {e.message}")
            print(f"❌ Schema validation failed: {artifact_path.name}")
            print(f"   Error: {e.message}")
            if e.path:
                print(f"   Path: {' -> '.join(str(p) for p in e.path)}")
            return False
        except jsonschema.SchemaError as e:
            self.errors.append(f"Invalid schema: {e.message}")
            print(f"❌ Invalid schema: {e.message}")
            return False

    def validate_canonical_form(self, artifact_path: Path) -> bool:
        """Verify that an artifact is in RFC 8785 canonical form."""
        # Read the raw file content
        try:
            with open(artifact_path, 'r', encoding='utf-8') as f:
                raw_content = f.read()
        except Exception as e:
            self.errors.append(f"Failed to read {artifact_path}: {e}")
            return False

        # Parse and re-canonicalize
        artifact = self.load_json(artifact_path)
        if artifact is None:
            return False

        canonical_content = self.rfc8785_canonicalize(artifact)

        # Compare byte-for-byte
        if raw_content.strip() == canonical_content:
            print(f"✅ Canonical form verified: {artifact_path.name}")
            return True
        else:
            self.errors.append(f"Artifact {artifact_path} is not in RFC 8785 canonical form")
            print(f"❌ Canonical form check failed: {artifact_path.name}")
            print(f"   Artifact is not in RFC 8785 canonical form.")
            print(f"   Expected: {canonical_content[:100]}...")
            print(f"   Got:      {raw_content.strip()[:100]}...")
            return False

    def validate_version_consistency(self, artifact: Dict[str, Any], schema: Dict[str, Any], artifact_path: Path) -> bool:
        """Check that the artifact version matches the schema version."""
        artifact_version = artifact.get("version")
        schema_version = schema.get("x-version")

        if not artifact_version:
            self.errors.append(f"Artifact {artifact_path} is missing 'version' field")
            print(f"❌ Version check failed: {artifact_path.name}")
            print(f"   Artifact is missing 'version' field")
            return False

        if not schema_version:
            # Schema doesn't declare a version, skip this check
            print(f"⚠️  Schema version not declared, skipping version check: {artifact_path.name}")
            return True

        if artifact_version != schema_version:
            self.errors.append(
                f"Version mismatch for {artifact_path}: "
                f"artifact={artifact_version}, schema={schema_version}"
            )
            print(f"❌ Version check failed: {artifact_path.name}")
            print(f"   Artifact version: {artifact_version}")
            print(f"   Schema version:   {schema_version}")
            return False

        print(f"✅ Version consistency verified: {artifact_path.name} (v{artifact_version})")
        return True

    def validate_artifact(self, artifact_path: Path, schema_name: str) -> bool:
        """Validate a single artifact."""
        print(f"\n🔍 Validating: {artifact_path}")
        print(f"   Schema: {schema_name}")

        # Load artifact
        artifact = self.load_json(artifact_path)
        if artifact is None:
            return False

        # Load schema
        schema = self.load_schema(schema_name)
        if schema is None:
            return False

        # Run validations
        schema_valid = self.validate_schema(artifact, schema, artifact_path)
        canonical_valid = self.validate_canonical_form(artifact_path)
        version_valid = self.validate_version_consistency(artifact, schema, artifact_path)

        return schema_valid and canonical_valid and version_valid

    def validate_all(self, artifacts_dir: Path) -> bool:
        """Validate all artifacts in a directory."""
        # Mapping of artifact filenames to schema names
        artifact_schema_map = {
            "curriculum_snapshot.json": "curriculum_snapshot",
            "telemetry_schema_snapshot.json": "telemetry_schema_snapshot",
            "ledger_snapshot.json": "ledger_snapshot",
            "attestation_snapshot.json": "attestation_snapshot",
        }

        all_valid = True
        for artifact_name, schema_name in artifact_schema_map.items():
            artifact_path = artifacts_dir / artifact_name
            if artifact_path.exists():
                valid = self.validate_artifact(artifact_path, schema_name)
                all_valid = all_valid and valid
            else:
                print(f"⏭️  Skipping (not found): {artifact_name}")

        return all_valid

    def diff(self, baseline_path: Path, current_path: Path) -> bool:
        """Compare two versions of an artifact and report drift."""
        print(f"\n🔍 Comparing artifacts:")
        print(f"   Baseline: {baseline_path}")
        print(f"   Current:  {current_path}")

        baseline = self.load_json(baseline_path)
        current = self.load_json(current_path)

        if baseline is None or current is None:
            return False

        # Simple deep comparison
        if baseline == current:
            print("✅ No drift detected (artifacts are identical)")
            return True
        else:
            print("⚠️  Drift detected (artifacts differ)")
            # For more detailed diff, you could use a library like `deepdiff`
            return True  # Not a failure, just informational


def main():
    parser = argparse.ArgumentParser(description="Governance Validator")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd(), help="Repository root directory")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # validate command
    validate_parser = subparsers.add_parser("validate", help="Validate a single artifact")
    validate_parser.add_argument("--artifact-path", type=Path, required=True, help="Path to artifact")
    validate_parser.add_argument("--schema-name", type=str, required=True, help="Schema name (without .schema.json)")

    # validate-all command
    validate_all_parser = subparsers.add_parser("validate-all", help="Validate all artifacts in a directory")
    validate_all_parser.add_argument("--artifacts-dir", type=Path, required=True, help="Directory containing artifacts")

    # diff command
    diff_parser = subparsers.add_parser("diff", help="Compare two versions of an artifact")
    diff_parser.add_argument("--baseline-path", type=Path, required=True, help="Baseline artifact path")
    diff_parser.add_argument("--current-path", type=Path, required=True, help="Current artifact path")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(EXIT_ERROR)

    validator = GovernanceValidator(args.repo_root)

    if args.command == "validate":
        success = validator.validate_artifact(args.artifact_path, args.schema_name)
    elif args.command == "validate-all":
        success = validator.validate_all(args.artifacts_dir)
    elif args.command == "diff":
        success = validator.diff(args.baseline_path, args.current_path)
    else:
        parser.print_help()
        sys.exit(EXIT_ERROR)

    print()
    if success:
        print("✅ All validations passed")
        sys.exit(EXIT_PASS)
    else:
        print(f"❌ Validation failed with {len(validator.errors)} error(s)")
        for error in validator.errors:
            print(f"   - {error}")
        sys.exit(EXIT_FAIL)


if __name__ == "__main__":
    main()
