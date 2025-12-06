#!/usr/bin/env python3
"""
First Organism Environment Validator

Validates that .env.first_organism contains all required variables with
sufficient credential strength for secure operation.

Usage:
    python tools/validate_first_organism_env.py [path/to/.env.first_organism]

Exit codes:
    0 - All checks passed
    1 - Validation failed
    2 - File not found or unreadable
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

# Blocklist of weak/default passwords
WEAK_PASSWORDS = frozenset({
    "mlpass",
    "postgres",
    "password",
    "test",
    "admin",
    "root",
    "secret",
    "changeme",
    "12345",
    "123456",
    "qwerty",
    "devkey",
    "default",
    "organism-secret",
    "redis-secret",
    "phoenix-org-1",
})

# Required environment variables with validation rules
REQUIRED_VARS = {
    "DATABASE_URL": {"min_length": 20, "must_contain": ["://", "@"]},
    "REDIS_URL": {"min_length": 15, "must_contain": ["://"]},
    "POSTGRES_USER": {"min_length": 3},
    "POSTGRES_PASSWORD": {"min_length": 12},
    "POSTGRES_DB": {"min_length": 3},
    "REDIS_PASSWORD": {"min_length": 12},
    "LEDGER_API_KEY": {"min_length": 16},
    "CORS_ALLOWED_ORIGINS": {"min_length": 10, "no_wildcard": True},
}

# Optional but recommended variables
OPTIONAL_VARS = {
    "MAX_REQUEST_BODY_BYTES": {"type": "int"},
    "RATE_LIMIT_REQUESTS_PER_MINUTE": {"type": "int"},
    "RATE_LIMIT_WINDOW_SECONDS": {"type": "int"},
    "FIRST_ORGANISM_STRICT_MODE": {"values": ["true", "false", "1", "0"]},
}


@dataclass
class ValidationResult:
    """Result of validating a single variable."""

    name: str
    value: Optional[str]
    passed: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def parse_env_file(path: Path) -> dict[str, str]:
    """Parse a .env file into a dictionary."""
    env_vars = {}
    
    if not path.exists():
        return env_vars
    
    content = path.read_text(encoding="utf-8")
    
    for line in content.splitlines():
        line = line.strip()
        
        # Skip empty lines and comments
        if not line or line.startswith("#"):
            continue
        
        # Handle KEY=VALUE format
        if "=" in line:
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            
            # Remove surrounding quotes
            if (value.startswith('"') and value.endswith('"')) or \
               (value.startswith("'") and value.endswith("'")):
                value = value[1:-1]
            
            # Skip placeholder values
            if value.startswith("<") and value.endswith(">"):
                continue
            
            env_vars[key] = value
    
    return env_vars


def extract_password_from_url(url: str) -> Optional[str]:
    """Extract password from a database or Redis URL."""
    match = re.search(r"://[^:]+:([^@]+)@", url)
    if match:
        return match.group(1)
    return None


def check_weak_password(value: str) -> Optional[str]:
    """Check if value contains a weak password pattern."""
    lowered = value.lower()
    for weak in WEAK_PASSWORDS:
        if weak in lowered:
            return weak
    return None


def validate_variable(
    name: str,
    value: Optional[str],
    rules: dict,
) -> ValidationResult:
    """Validate a single environment variable."""
    result = ValidationResult(name=name, value=value, passed=True)
    
    if value is None or value.strip() == "":
        result.passed = False
        result.errors.append(f"Variable {name} is not set or empty")
        return result
    
    # Check minimum length
    min_length = rules.get("min_length", 0)
    if len(value) < min_length:
        result.passed = False
        result.errors.append(
            f"{name} must be at least {min_length} characters (got {len(value)})"
        )
    
    # Check required substrings
    must_contain = rules.get("must_contain", [])
    for substr in must_contain:
        if substr not in value:
            result.passed = False
            result.errors.append(f"{name} must contain '{substr}'")
    
    # Check for wildcards in CORS
    if rules.get("no_wildcard") and "*" in value:
        result.passed = False
        result.errors.append(f"{name} cannot contain wildcard (*)")
    
    # Check for weak passwords
    weak = check_weak_password(value)
    if weak:
        result.passed = False
        result.errors.append(f"{name} contains weak/default password pattern: {weak!r}")
    
    # For URLs, also check the embedded password
    if "URL" in name:
        password = extract_password_from_url(value)
        if password:
            weak = check_weak_password(password)
            if weak:
                result.passed = False
                result.errors.append(
                    f"{name} contains weak password in URL: {weak!r}"
                )
            if len(password) < 12:
                result.warnings.append(
                    f"{name} password is short ({len(password)} chars), recommend 12+"
                )
    
    return result


def validate_env_file(path: Path) -> Tuple[bool, List[ValidationResult]]:
    """Validate all variables in an env file."""
    env_vars = parse_env_file(path)
    results = []
    all_passed = True
    
    # Check required variables
    for name, rules in REQUIRED_VARS.items():
        value = env_vars.get(name)
        result = validate_variable(name, value, rules)
        results.append(result)
        if not result.passed:
            all_passed = False
    
    # Check optional variables (warnings only)
    for name, rules in OPTIONAL_VARS.items():
        value = env_vars.get(name)
        if value is not None:
            result = ValidationResult(name=name, value=value, passed=True)
            
            if rules.get("type") == "int":
                try:
                    int(value)
                except ValueError:
                    result.warnings.append(f"{name} should be an integer")
            
            allowed_values = rules.get("values")
            if allowed_values and value.lower() not in allowed_values:
                result.warnings.append(
                    f"{name} should be one of: {', '.join(allowed_values)}"
                )
            
            results.append(result)
    
    return all_passed, results


def print_results(results: List[ValidationResult]) -> None:
    """Print validation results to stdout."""
    passed = [r for r in results if r.passed and not r.warnings]
    warnings = [r for r in results if r.passed and r.warnings]
    failed = [r for r in results if not r.passed]
    
    print("\n" + "=" * 60)
    print("FIRST ORGANISM ENVIRONMENT VALIDATION")
    print("=" * 60)
    
    if passed:
        print(f"\n✅ PASSED ({len(passed)} variables):")
        for r in passed:
            print(f"   {r.name}")
    
    if warnings:
        print(f"\n⚠️  WARNINGS ({len(warnings)} variables):")
        for r in warnings:
            print(f"   {r.name}:")
            for w in r.warnings:
                print(f"      - {w}")
    
    if failed:
        print(f"\n❌ FAILED ({len(failed)} variables):")
        for r in failed:
            print(f"   {r.name}:")
            for e in r.errors:
                print(f"      - {e}")
    
    print("\n" + "-" * 60)
    if failed:
        print("RESULT: VALIDATION FAILED")
        print("Fix the errors above before running First Organism tests.")
    else:
        print("RESULT: VALIDATION PASSED")
        if warnings:
            print("Consider addressing the warnings for optimal security.")
    print("-" * 60 + "\n")


def main() -> int:
    """Main entry point."""
    # Determine env file path
    if len(sys.argv) > 1:
        env_path = Path(sys.argv[1])
    else:
        # Default locations
        for candidate in [
            Path(".env.first_organism"),
            Path("ops/first_organism/.env.first_organism"),
        ]:
            if candidate.exists():
                env_path = candidate
                break
        else:
            env_path = Path(".env.first_organism")
    
    print(f"Validating: {env_path.absolute()}")
    
    if not env_path.exists():
        print(f"\n❌ ERROR: File not found: {env_path}")
        print("\nTo create the file:")
        print("  cp ops/first_organism/first_organism.env.template .env.first_organism")
        print("  # Then edit .env.first_organism with your credentials")
        return 2
    
    all_passed, results = validate_env_file(env_path)
    print_results(results)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

