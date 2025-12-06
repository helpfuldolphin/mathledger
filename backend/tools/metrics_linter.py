"""
MathLedger Metrics V1 Linter

Validates metrics data structure and content for consistency and correctness.
Provides ASCII-only error messages for stable CI/CD integration.
"""

import json
import re
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime


class MetricsLinterError(Exception):
    """Base exception for metrics linter errors."""
    pass


class MetricsLinter:
    """Validates metrics data structure and content."""

    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def lint(self, data: Any) -> Tuple[bool, List[str], List[str]]:
        """
        Lint metrics data and return validation results.

        Args:
            data: Metrics data to validate (dict, str, or file path)

        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        self.errors = []
        self.warnings = []

        try:
            # Parse input data
            if isinstance(data, str):
                if data.strip() == "":
                    self.errors.append("EMPTY_FILE: Input file is empty")
                    return False, self.errors, self.warnings

                # Check if it's a file path or JSON string
                if data.startswith('{') or data.startswith('['):
                    metrics = json.loads(data)
                else:
                    with open(data, 'r') as f:
                        content = f.read().strip()
                        if not content:
                            self.errors.append("EMPTY_FILE: Input file is empty")
                            return False, self.errors, self.warnings
                        metrics = json.loads(content)
            elif isinstance(data, dict):
                metrics = data
            else:
                self.errors.append("INVALID_INPUT: Expected dict, JSON string, or file path")
                return False, self.errors, self.warnings

            # Validate structure
            self._validate_structure(metrics)

            # Validate content
            self._validate_content(metrics)

            return len(self.errors) == 0, self.errors, self.warnings

        except json.JSONDecodeError as e:
            self.errors.append(f"JSON_PARSE_ERROR: Invalid JSON - {str(e)}")
            return False, self.errors, self.warnings
        except FileNotFoundError:
            self.errors.append("FILE_NOT_FOUND: Input file does not exist")
            return False, self.errors, self.warnings
        except Exception as e:
            self.errors.append(f"UNEXPECTED_ERROR: {str(e)}")
            return False, self.errors, self.warnings

    def _validate_structure(self, metrics: Dict[str, Any]) -> None:
        """Validate the basic structure of metrics data."""
        required_fields = [
            'statements', 'proofs', 'derivation_rules', 'blocks',
            'lemmas', 'throughput', 'frontier', 'failures_by_class', 'queue'
        ]

        for field in required_fields:
            if field not in metrics:
                self.errors.append(f"MISSING_FIELD: Required field '{field}' is missing")

        # Validate nested structures
        if 'statements' in metrics:
            self._validate_statements_structure(metrics['statements'])

        if 'proofs' in metrics:
            self._validate_proofs_structure(metrics['proofs'])

        if 'blocks' in metrics:
            self._validate_blocks_structure(metrics['blocks'])

    def _validate_statements_structure(self, statements: Dict[str, Any]) -> None:
        """Validate statements metrics structure."""
        required_fields = ['total', 'axioms', 'derived', 'max_depth']

        for field in required_fields:
            if field not in statements:
                self.errors.append(f"MISSING_FIELD: statements.{field} is missing")
            elif not isinstance(statements[field], int):
                self.errors.append(f"TYPE_ERROR: statements.{field} must be integer, got {type(statements[field]).__name__}")

    def _validate_proofs_structure(self, proofs: Dict[str, Any]) -> None:
        """Validate proofs metrics structure."""
        required_fields = ['by_status', 'by_prover', 'recent_hour', 'success_rate']

        for field in required_fields:
            if field not in proofs:
                self.errors.append(f"MISSING_FIELD: proofs.{field} is missing")
            elif field == 'success_rate' and not isinstance(proofs[field], (int, float)):
                self.errors.append(f"TYPE_ERROR: proofs.success_rate must be numeric, got {type(proofs[field]).__name__}")
            elif field in ['recent_hour'] and not isinstance(proofs[field], int):
                self.errors.append(f"TYPE_ERROR: proofs.{field} must be integer, got {type(proofs[field]).__name__}")

    def _validate_blocks_structure(self, blocks: Dict[str, Any]) -> None:
        """Validate blocks metrics structure."""
        required_fields = ['total']

        for field in required_fields:
            if field not in blocks:
                self.errors.append(f"MISSING_FIELD: blocks.{field} is missing")
            elif not isinstance(blocks[field], int):
                self.errors.append(f"TYPE_ERROR: blocks.{field} must be integer, got {type(blocks[field]).__name__}")

    def _validate_content(self, metrics: Dict[str, Any]) -> None:
        """Validate the content and values of metrics data."""
        # Check for legacy format indicators
        self._check_legacy_format(metrics)

        # Validate merkle hashes
        self._validate_merkle_hashes(metrics)

        # Validate numeric ranges
        self._validate_numeric_ranges(metrics)

        # Validate wall_minutes if present
        self._validate_wall_minutes(metrics)

        # Validate seed type if present
        self._validate_seed_type(metrics)

    def _check_legacy_format(self, metrics: Dict[str, Any]) -> None:
        """Check for legacy format indicators."""
        # Look for single legacy line patterns
        if 'legacy_format' in metrics:
            self.warnings.append("LEGACY_FORMAT: Detected legacy format indicator")

        # Check for old field names
        legacy_fields = ['old_metrics', 'deprecated_stats', 'legacy_data', 'old_data']
        for field in legacy_fields:
            if field in metrics:
                self.warnings.append(f"LEGACY_FIELD: Found legacy field '{field}'")

    def _validate_merkle_hashes(self, metrics: Dict[str, Any]) -> None:
        """Validate merkle hash lengths."""
        def check_hash_length(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    current_path = f"{path}.{key}" if path else key
                    if ('hash' in key.lower() or key.lower() in ['merkle_root', 'root_hash', 'block_hash']) and isinstance(value, str):
                        if len(value) != 64:
                            self.errors.append(f"MERKLE_LENGTH_ERROR: {current_path} hash length is {len(value)}, expected 64")
                    elif isinstance(value, (dict, list)):
                        check_hash_length(value, current_path)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    check_hash_length(item, f"{path}[{i}]")

        check_hash_length(metrics)

    def _validate_numeric_ranges(self, metrics: Dict[str, Any]) -> None:
        """Validate numeric values are within reasonable ranges."""
        if 'statements' in metrics and isinstance(metrics['statements'], dict):
            stmt = metrics['statements']
            if 'total' in stmt and isinstance(stmt['total'], int):
                if stmt['total'] < 0:
                    self.errors.append("RANGE_ERROR: statements.total cannot be negative")
                elif stmt['total'] > 1000000:
                    self.warnings.append("RANGE_WARNING: statements.total is very large")

        if 'proofs' in metrics and isinstance(metrics['proofs'], dict):
            proofs = metrics['proofs']
            if 'success_rate' in proofs and isinstance(proofs['success_rate'], (int, float)):
                rate = proofs['success_rate']
                if rate < 0 or rate > 100:
                    self.errors.append("RANGE_ERROR: proofs.success_rate must be between 0 and 100")

    def _validate_wall_minutes(self, metrics: Dict[str, Any]) -> None:
        """Validate wall_minutes field if present."""
        def check_wall_minutes(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    current_path = f"{path}.{key}" if path else key
                    if key == 'wall_minutes':
                        if not isinstance(value, (int, float)) or isinstance(value, bool):
                            self.errors.append(f"WALL_MINUTES_TYPE_ERROR: {current_path} must be numeric, got {type(value).__name__}")
                        elif isinstance(value, (int, float)) and not isinstance(value, bool) and value < 0:
                            self.errors.append(f"WALL_MINUTES_RANGE_ERROR: {current_path} cannot be negative")
                    elif isinstance(value, (dict, list)):
                        check_wall_minutes(value, current_path)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    check_wall_minutes(item, f"{path}[{i}]")

        check_wall_minutes(metrics)

    def _validate_seed_type(self, metrics: Dict[str, Any]) -> None:
        """Validate seed field type if present."""
        def check_seed_type(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    current_path = f"{path}.{key}" if path else key
                    if key == 'seed':
                        if not isinstance(value, (str, int)):
                            self.errors.append(f"SEED_TYPE_ERROR: {current_path} must be string or integer, got {type(value).__name__}")
                    elif isinstance(value, (dict, list)):
                        check_seed_type(value, current_path)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    check_seed_type(item, f"{path}[{i}]")

        check_seed_type(metrics)


def lint_metrics(data: Any) -> Tuple[bool, List[str], List[str]]:
    """
    Convenience function to lint metrics data.

    Args:
        data: Metrics data to validate

    Returns:
        Tuple of (is_valid, errors, warnings)
    """
    linter = MetricsLinter()
    return linter.lint(data)


def main():
    """CLI entry point for metrics linter."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="MathLedger Metrics V1 Linter")
    parser.add_argument('input', help='Metrics JSON file or JSON string')
    parser.add_argument('--warnings-as-errors', action='store_true',
                       help='Treat warnings as errors')

    args = parser.parse_args()

    is_valid, errors, warnings = lint_metrics(args.input)

    # Print errors
    for error in errors:
        print(f"ERROR: {error}")

    # Print warnings
    for warning in warnings:
        print(f"WARNING: {warning}")

    # Determine exit code
    if not is_valid or (args.warnings_as_errors and warnings):
        sys.exit(1)
    else:
        print("SUCCESS: Metrics validation passed")
        sys.exit(0)


if __name__ == "__main__":
    main()
