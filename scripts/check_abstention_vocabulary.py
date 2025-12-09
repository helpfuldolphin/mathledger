#!/usr/bin/env python3
"""
Abstention Vocabulary Drift Checker

Scans the codebase for abstention-related keys and ensures they all map to
known AbstentionType values. This enforces the "no drift" invariant across
all layers of the system.

USAGE:
    # Check all relevant directories
    uv run python scripts/check_abstention_vocabulary.py

    # Check specific directories
    uv run python scripts/check_abstention_vocabulary.py --paths rfl/ derivation/

    # Output JSON report
    uv run python scripts/check_abstention_vocabulary.py --json

    # Strict mode (exit 1 on any unknown key)
    uv run python scripts/check_abstention_vocabulary.py --strict

EXIT CODES:
    0 - All abstention keys are known
    1 - Unknown abstention keys found (--strict mode)
    2 - Error during scanning

PHASE II — VERIFICATION ZONE
Agent B6 (abstention-ops-6)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rfl.verification.abstention_taxonomy import (
    AbstentionType,
    classify_verification_method,
    classify_breakdown_key,
    ABSTENTION_METHOD_STRINGS,
    COMPLETE_MAPPING,
)
from rfl.verification.abstention_semantics import (
    ABSTENTION_TREE,
    categorize,
    get_all_categories,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Directories to scan for abstention vocabulary
DEFAULT_SCAN_PATHS = [
    "rfl/",
    "derivation/",
    "experiments/",
    "backend/metrics/",
    "backend/verification/",
]

# File extensions to scan
SCAN_EXTENSIONS = {".py", ".json", ".yaml", ".yml"}

# Patterns to identify abstention-related strings
ABSTENTION_PATTERNS = [
    # Direct abstention type values
    r'"(abstain_\w+)"',
    r"'(abstain_\w+)'",
    # Verification method strings
    r'"(lean-disabled|lean-timeout|lean-error|truth-table-\w+)"',
    r"'(lean-disabled|lean-timeout|lean-error|truth-table-\w+)'",
    # Legacy breakdown keys
    r'"(engine_failure|timeout|unexpected_error|empty_run|pending_validation)"',
    r"'(engine_failure|timeout|unexpected_error|empty_run|pending_validation)'",
    r'"(no_successful_proofs|zero_throughput|budget_exceeded|candidate_limit)"',
    r"'(no_successful_proofs|zero_throughput|budget_exceeded|candidate_limit)'",
    # Status strings
    r'status\s*[=:]\s*["\'](\w*abstain\w*)["\']',
    r'method\s*[=:]\s*["\']([\w-]+)["\']',
    # Dictionary keys with abstention-like names
    r'\["(\w*abstain\w+)"\]',
    r"\['(\w*abstain\w+)'\]",
    # Histogram/breakdown keys
    r'breakdown\[["\']([\w_]+)["\']\]',
    r'histogram\[["\']([\w_]+)["\']\]',
]

# Known keys that are NOT abstention types (whitelist)
KNOWN_NON_ABSTENTION_KEYS = {
    # General status values
    "success",
    "failed",
    "error",
    "aborted",
    "verified",
    "pattern",
    "truth-table",
    "lean",
    "unknown",
    "none",
    "unverified",
    "method",
    "seed",
    "TIMEOUT",  # Constant name, not value
    # Meta keys (field names, not values)
    "abstention_rate",
    "abstention_count",
    "abstention_mass",
    "abstention_fraction",
    "abstention_breakdown",
    "abstention_histogram",
    "abstention_tolerance",
    "abstention_type",  # Field name, not value
    "abstentions",
    "abstained",
    "abstain_pct",
    "abstain_count",
    "abstain_reason",
    "n_abstained",
    "has_derivation_abstained",
    "abstain",  # Generic abstain string, not a type
    "verification_method",
    # Lean adapter modes
    "lean_adapter_disabled",
    "lean_adapter_scaffold",
    "lean_adapter_simulate",
    # Statistical methods
    "wilson",
    "two_proportion_z",
    # Verification methods (non-abstention)
    "truth-table-only",
    "truth-table-timeout",
    # Partial matches
    "abstain_error",  # Partial/generic, not canonical type
    "abstain_lean_",  # Partial pattern, not full type
    # Test/fixture keys
    "test_abstain",
    "mock_abstain",
}


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class VocabularyMatch:
    """A single match of an abstention-related string in the codebase."""
    file_path: str
    line_number: int
    key: str
    context: str
    is_known: bool
    canonical_type: Optional[str] = None
    category: Optional[str] = None


@dataclass
class ScanResult:
    """Complete result of vocabulary scan."""
    total_files_scanned: int = 0
    total_matches: int = 0
    known_matches: int = 0
    unknown_matches: int = 0
    matches: List[VocabularyMatch] = field(default_factory=list)
    unknown_keys: Set[str] = field(default_factory=set)
    category_coverage: Dict[str, int] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_files_scanned": self.total_files_scanned,
            "total_matches": self.total_matches,
            "known_matches": self.known_matches,
            "unknown_matches": self.unknown_matches,
            "unknown_keys": sorted(self.unknown_keys),
            "category_coverage": self.category_coverage,
            "matches": [asdict(m) for m in self.matches],
            "errors": self.errors,
        }


# ---------------------------------------------------------------------------
# Scanner Implementation
# ---------------------------------------------------------------------------

class AbstentionVocabularyScanner:
    """
    Scans codebase for abstention-related keys and validates them against
    the canonical taxonomy.
    """

    def __init__(self, root_path: Path = PROJECT_ROOT):
        self.root_path = root_path
        self.patterns = [re.compile(p, re.IGNORECASE) for p in ABSTENTION_PATTERNS]

    def scan(self, paths: List[str]) -> ScanResult:
        """
        Scan specified paths for abstention vocabulary.
        
        Args:
            paths: List of relative paths to scan
            
        Returns:
            ScanResult with all findings
        """
        result = ScanResult()
        
        for path_str in paths:
            path = self.root_path / path_str
            if not path.exists():
                result.errors.append(f"Path does not exist: {path_str}")
                continue
            
            if path.is_file():
                self._scan_file(path, result)
            else:
                for file_path in path.rglob("*"):
                    if file_path.is_file() and file_path.suffix in SCAN_EXTENSIONS:
                        self._scan_file(file_path, result)
        
        # Compute category coverage
        for category in get_all_categories():
            cat_key = category.value if hasattr(category, 'value') else str(category)
            result.category_coverage[cat_key] = 0
        
        for match in result.matches:
            if match.category:
                result.category_coverage[match.category] = (
                    result.category_coverage.get(match.category, 0) + 1
                )
        
        return result

    def _scan_file(self, file_path: Path, result: ScanResult) -> None:
        """Scan a single file for abstention vocabulary."""
        result.total_files_scanned += 1
        
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            result.errors.append(f"Error reading {file_path}: {e}")
            return
        
        rel_path = str(file_path.relative_to(self.root_path))
        
        for line_num, line in enumerate(content.split("\n"), start=1):
            # Skip comments
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith("//"):
                continue
            
            for pattern in self.patterns:
                for match in pattern.finditer(line):
                    key = match.group(1)
                    self._process_match(key, rel_path, line_num, line, result)

    def _process_match(
        self, 
        key: str, 
        file_path: str, 
        line_num: int, 
        context: str,
        result: ScanResult
    ) -> None:
        """Process a potential abstention key match."""
        # Skip known non-abstention keys (case-insensitive)
        key_lower = key.lower()
        if key_lower in {k.lower() for k in KNOWN_NON_ABSTENTION_KEYS}:
            return
        
        # Skip if it's a variable/field name pattern
        if key.endswith("_rate") or key.endswith("_count") or key.endswith("_mass"):
            return
        
        result.total_matches += 1
        
        # Try to classify the key
        canonical_type = None
        category = None
        is_known = False
        
        # Check if it's a direct AbstentionType value
        try:
            abst_type = AbstentionType(key)
            canonical_type = abst_type.value
            cat = categorize(abst_type)
            category = cat.value if hasattr(cat, 'value') else str(cat)
            is_known = True
        except ValueError:
            pass
        
        # Check verification method mapping
        if not is_known:
            abst_type = classify_verification_method(key)
            if abst_type:
                canonical_type = abst_type.value
                cat = categorize(abst_type)
                category = cat.value if hasattr(cat, 'value') else str(cat)
                is_known = True
        
        # Check breakdown key mapping
        if not is_known:
            abst_type = classify_breakdown_key(key)
            if abst_type:
                canonical_type = abst_type.value
                cat = categorize(abst_type)
                category = cat.value if hasattr(cat, 'value') else str(cat)
                is_known = True
        
        # Check if in complete mapping
        if not is_known and key in COMPLETE_MAPPING:
            abst_type = COMPLETE_MAPPING[key]
            canonical_type = abst_type.value
            cat = categorize(abst_type)
            category = cat.value if hasattr(cat, 'value') else str(cat)
            is_known = True
        
        # Record the match
        match_record = VocabularyMatch(
            file_path=file_path,
            line_number=line_num,
            key=key,
            context=context.strip()[:100],  # Truncate context
            is_known=is_known,
            canonical_type=canonical_type,
            category=category,
        )
        result.matches.append(match_record)
        
        if is_known:
            result.known_matches += 1
        else:
            result.unknown_matches += 1
            result.unknown_keys.add(key)


# ---------------------------------------------------------------------------
# Validation Functions
# ---------------------------------------------------------------------------

def validate_semantic_tree_coverage() -> Tuple[bool, List[str]]:
    """
    Validate that the semantic tree covers all AbstentionType values.
    
    Returns:
        (is_valid, list_of_errors)
    """
    errors = []
    
    # ABSTENTION_TREE is Dict[AbstentionType, SemanticCategory]
    # Check all AbstentionType values are covered
    tree_types = set(ABSTENTION_TREE.keys())
    
    for abst_type in AbstentionType:
        if abst_type not in tree_types:
            errors.append(f"AbstentionType.{abst_type.name} not in semantic tree")
    
    # Check no extra types in tree (shouldn't happen with enum keys)
    valid_types = set(AbstentionType)
    for abst_type in tree_types:
        if abst_type not in valid_types:
            errors.append(f"Unknown type '{abst_type}' in semantic tree")
    
    return len(errors) == 0, errors


def validate_no_drift(result: ScanResult, strict: bool = False) -> Tuple[bool, List[str]]:
    """
    Validate that no unknown abstention keys were found.
    
    Args:
        result: Scan result to validate
        strict: If True, any unknown key is a failure
        
    Returns:
        (is_valid, list_of_errors)
    """
    errors = []
    
    if result.unknown_keys:
        for key in sorted(result.unknown_keys):
            # Find first occurrence
            first_match = next(
                (m for m in result.matches if m.key == key and not m.is_known),
                None
            )
            location = f"{first_match.file_path}:{first_match.line_number}" if first_match else "unknown"
            errors.append(f"Unknown abstention key '{key}' at {location}")
    
    if strict:
        return len(errors) == 0, errors
    
    # In non-strict mode, only fail on clearly problematic keys
    critical_errors = [e for e in errors if "abstain_" in e.lower()]
    return len(critical_errors) == 0, errors


# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------

def generate_report(result: ScanResult, verbose: bool = False) -> str:
    """Generate human-readable report from scan result."""
    lines = [
        "=" * 70,
        "ABSTENTION VOCABULARY DRIFT CHECK",
        "=" * 70,
        "",
        f"Files scanned: {result.total_files_scanned}",
        f"Total matches: {result.total_matches}",
        f"Known keys:    {result.known_matches}",
        f"Unknown keys:  {result.unknown_matches}",
        "",
    ]
    
    # Category coverage
    lines.append("Category Coverage:")
    for category, count in sorted(result.category_coverage.items()):
        lines.append(f"  {category}: {count}")
    lines.append("")
    
    # Unknown keys
    if result.unknown_keys:
        lines.append("⚠️  UNKNOWN KEYS DETECTED:")
        for key in sorted(result.unknown_keys):
            lines.append(f"  - {key}")
        lines.append("")
    else:
        lines.append("✅ No unknown abstention keys found")
        lines.append("")
    
    # Errors
    if result.errors:
        lines.append("ERRORS:")
        for error in result.errors:
            lines.append(f"  - {error}")
        lines.append("")
    
    # Verbose: show all matches
    if verbose and result.matches:
        lines.append("ALL MATCHES:")
        for match in result.matches:
            status = "✓" if match.is_known else "✗"
            lines.append(
                f"  [{status}] {match.file_path}:{match.line_number} "
                f"key={match.key!r} → {match.canonical_type or 'UNKNOWN'}"
            )
        lines.append("")
    
    # Semantic tree validation
    tree_valid, tree_errors = validate_semantic_tree_coverage()
    if tree_valid:
        lines.append("✅ Semantic tree covers all AbstentionType values")
    else:
        lines.append("⚠️  SEMANTIC TREE COVERAGE ISSUES:")
        for error in tree_errors:
            lines.append(f"  - {error}")
    
    lines.append("")
    lines.append("=" * 70)
    
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main() -> int:
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Check abstention vocabulary for drift from canonical taxonomy"
    )
    parser.add_argument(
        "--paths",
        nargs="+",
        default=DEFAULT_SCAN_PATHS,
        help="Paths to scan (default: rfl/, derivation/, experiments/)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON report instead of human-readable"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 if any unknown keys found"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show all matches in report"
    )
    
    args = parser.parse_args()
    
    # Run scan
    scanner = AbstentionVocabularyScanner()
    result = scanner.scan(args.paths)
    
    # Validate
    drift_valid, drift_errors = validate_no_drift(result, strict=args.strict)
    tree_valid, tree_errors = validate_semantic_tree_coverage()
    
    # Output
    if args.json:
        output = result.to_dict()
        output["validation"] = {
            "no_drift": drift_valid,
            "drift_errors": drift_errors,
            "tree_coverage": tree_valid,
            "tree_errors": tree_errors,
        }
        print(json.dumps(output, indent=2))
    else:
        print(generate_report(result, verbose=args.verbose))
    
    # Exit code
    if args.strict and (not drift_valid or not tree_valid):
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

