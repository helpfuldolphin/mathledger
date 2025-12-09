#!/usr/bin/env python3
"""
Abstention Taxonomy Diff Tool

Compares two taxonomy export files and reports changes between versions.
This enables clear tracking of what changed between taxonomy versions.

USAGE:
    # Human-readable diff
    uv run python scripts/diff_abstention_taxonomy.py --old old.json --new new.json

    # JSON structured diff
    uv run python scripts/diff_abstention_taxonomy.py --old old.json --new new.json --json

    # Exit with code 1 if any changes detected (useful for CI)
    uv run python scripts/diff_abstention_taxonomy.py --old old.json --new new.json --fail-on-diff

EXIT CODES:
    0 - No differences (or differences found without --fail-on-diff)
    1 - Differences found (with --fail-on-diff)
    2 - Error (file not found, invalid JSON, etc.)

PHASE II — VERIFICATION ZONE
Agent B6 (abstention-ops-6)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Set, Any, Optional, Tuple


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class TaxonomyDiff:
    """Complete diff between two taxonomy versions."""
    old_version: str
    new_version: str
    
    # Types
    added_types: List[str] = field(default_factory=list)
    removed_types: List[str] = field(default_factory=list)
    
    # Categories
    added_categories: List[str] = field(default_factory=list)
    removed_categories: List[str] = field(default_factory=list)
    
    # Category changes (type moved from one category to another)
    category_changes: List[Dict[str, str]] = field(default_factory=list)
    
    # Legacy mapping changes
    added_legacy_mappings: Dict[str, str] = field(default_factory=dict)
    removed_legacy_mappings: Dict[str, str] = field(default_factory=dict)
    changed_legacy_mappings: List[Dict[str, str]] = field(default_factory=list)
    
    # Verification method changes
    added_verification_methods: List[str] = field(default_factory=list)
    removed_verification_methods: List[str] = field(default_factory=list)
    
    @property
    def has_changes(self) -> bool:
        """Check if there are any changes."""
        return (
            bool(self.added_types) or
            bool(self.removed_types) or
            bool(self.added_categories) or
            bool(self.removed_categories) or
            bool(self.category_changes) or
            bool(self.added_legacy_mappings) or
            bool(self.removed_legacy_mappings) or
            bool(self.changed_legacy_mappings) or
            bool(self.added_verification_methods) or
            bool(self.removed_verification_methods)
        )
    
    @property
    def has_breaking_changes(self) -> bool:
        """Check if there are breaking changes (removals or moves)."""
        return (
            bool(self.removed_types) or
            bool(self.removed_categories) or
            bool(self.category_changes) or
            bool(self.removed_legacy_mappings) or
            bool(self.changed_legacy_mappings)
        )
    
    @property
    def change_summary(self) -> str:
        """Return a brief summary of changes."""
        parts = []
        if self.added_types:
            parts.append(f"+{len(self.added_types)} types")
        if self.removed_types:
            parts.append(f"-{len(self.removed_types)} types")
        if self.added_categories:
            parts.append(f"+{len(self.added_categories)} categories")
        if self.removed_categories:
            parts.append(f"-{len(self.removed_categories)} categories")
        if self.category_changes:
            parts.append(f"{len(self.category_changes)} category moves")
        if self.added_legacy_mappings:
            parts.append(f"+{len(self.added_legacy_mappings)} legacy mappings")
        if self.removed_legacy_mappings:
            parts.append(f"-{len(self.removed_legacy_mappings)} legacy mappings")
        if self.changed_legacy_mappings:
            parts.append(f"{len(self.changed_legacy_mappings)} changed mappings")
        return ", ".join(parts) if parts else "No changes"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "old_version": self.old_version,
            "new_version": self.new_version,
            "has_changes": self.has_changes,
            "has_breaking_changes": self.has_breaking_changes,
            "summary": self.change_summary,
            "added_types": sorted(self.added_types),
            "removed_types": sorted(self.removed_types),
            "added_categories": sorted(self.added_categories),
            "removed_categories": sorted(self.removed_categories),
            "category_changes": sorted(self.category_changes, key=lambda x: x.get("type", "")),
            "added_legacy_mappings": dict(sorted(self.added_legacy_mappings.items())),
            "removed_legacy_mappings": dict(sorted(self.removed_legacy_mappings.items())),
            "changed_legacy_mappings": sorted(self.changed_legacy_mappings, key=lambda x: x.get("key", "")),
            "added_verification_methods": sorted(self.added_verification_methods),
            "removed_verification_methods": sorted(self.removed_verification_methods),
        }


# ---------------------------------------------------------------------------
# Diff Functions
# ---------------------------------------------------------------------------

def load_taxonomy(path: Path) -> Dict[str, Any]:
    """Load and validate a taxonomy JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"Taxonomy file not found: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Basic validation
    required_keys = ["abstention_types", "categories"]
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Invalid taxonomy file: missing '{key}' key")
    
    return data


def compute_diff(old_data: Dict[str, Any], new_data: Dict[str, Any]) -> TaxonomyDiff:
    """
    Compute the diff between two taxonomy exports.
    
    Args:
        old_data: The old taxonomy export dictionary
        new_data: The new taxonomy export dictionary
        
    Returns:
        TaxonomyDiff with all detected changes
    """
    diff = TaxonomyDiff(
        old_version=old_data.get("taxonomy_version", old_data.get("version", "unknown")),
        new_version=new_data.get("taxonomy_version", new_data.get("version", "unknown")),
    )
    
    # Compare abstention types
    old_types = set(old_data.get("abstention_types", {}).keys())
    new_types = set(new_data.get("abstention_types", {}).keys())
    
    diff.added_types = sorted(new_types - old_types)
    diff.removed_types = sorted(old_types - new_types)
    
    # Compare categories
    old_categories = set(old_data.get("categories", {}).keys())
    new_categories = set(new_data.get("categories", {}).keys())
    
    diff.added_categories = sorted(new_categories - old_categories)
    diff.removed_categories = sorted(old_categories - new_categories)
    
    # Check for category changes (type moved from one category to another)
    old_type_info = old_data.get("abstention_types", {})
    new_type_info = new_data.get("abstention_types", {})
    
    for type_name in old_types & new_types:
        old_cat = old_type_info.get(type_name, {}).get("category")
        new_cat = new_type_info.get(type_name, {}).get("category")
        
        if old_cat != new_cat:
            diff.category_changes.append({
                "type": type_name,
                "old_category": old_cat,
                "new_category": new_cat,
            })
    
    # Compare legacy mappings
    old_mappings = old_data.get("legacy_mappings", {})
    new_mappings = new_data.get("legacy_mappings", {})
    
    old_keys = set(old_mappings.keys())
    new_keys = set(new_mappings.keys())
    
    diff.added_legacy_mappings = {k: new_mappings[k] for k in sorted(new_keys - old_keys)}
    diff.removed_legacy_mappings = {k: old_mappings[k] for k in sorted(old_keys - new_keys)}
    
    # Check for changed mappings (same key, different target)
    for key in old_keys & new_keys:
        if old_mappings[key] != new_mappings[key]:
            diff.changed_legacy_mappings.append({
                "key": key,
                "old_target": old_mappings[key],
                "new_target": new_mappings[key],
            })
    
    # Compare verification methods
    old_methods = set(old_data.get("verification_methods", []))
    new_methods = set(new_data.get("verification_methods", []))
    
    diff.added_verification_methods = sorted(new_methods - old_methods)
    diff.removed_verification_methods = sorted(old_methods - new_methods)
    
    return diff


# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------

def generate_human_report(diff: TaxonomyDiff) -> str:
    """Generate human-readable diff report."""
    lines = [
        "=" * 70,
        "ABSTENTION TAXONOMY DIFF",
        "=" * 70,
        "",
        f"Old version: {diff.old_version}",
        f"New version: {diff.new_version}",
        "",
    ]
    
    if not diff.has_changes:
        lines.append("✅ No changes detected")
        lines.append("")
        lines.append("=" * 70)
        return "\n".join(lines)
    
    lines.append(f"Summary: {diff.change_summary}")
    if diff.has_breaking_changes:
        lines.append("⚠️  BREAKING CHANGES DETECTED")
    lines.append("")
    
    # Types
    if diff.added_types:
        lines.append("ADDED TYPES:")
        for t in diff.added_types:
            lines.append(f"  + {t}")
        lines.append("")
    
    if diff.removed_types:
        lines.append("REMOVED TYPES (BREAKING):")
        for t in diff.removed_types:
            lines.append(f"  - {t}")
        lines.append("")
    
    # Categories
    if diff.added_categories:
        lines.append("ADDED CATEGORIES:")
        for c in diff.added_categories:
            lines.append(f"  + {c}")
        lines.append("")
    
    if diff.removed_categories:
        lines.append("REMOVED CATEGORIES (BREAKING):")
        for c in diff.removed_categories:
            lines.append(f"  - {c}")
        lines.append("")
    
    # Category moves
    if diff.category_changes:
        lines.append("CATEGORY CHANGES (BREAKING):")
        for change in diff.category_changes:
            lines.append(
                f"  ~ {change['type']}: "
                f"{change['old_category']} → {change['new_category']}"
            )
        lines.append("")
    
    # Legacy mappings
    if diff.added_legacy_mappings:
        lines.append("ADDED LEGACY MAPPINGS:")
        for key, target in sorted(diff.added_legacy_mappings.items()):
            lines.append(f"  + {key} → {target}")
        lines.append("")
    
    if diff.removed_legacy_mappings:
        lines.append("REMOVED LEGACY MAPPINGS (BREAKING):")
        for key, target in sorted(diff.removed_legacy_mappings.items()):
            lines.append(f"  - {key} → {target}")
        lines.append("")
    
    if diff.changed_legacy_mappings:
        lines.append("CHANGED LEGACY MAPPINGS (BREAKING):")
        for change in diff.changed_legacy_mappings:
            lines.append(
                f"  ~ {change['key']}: "
                f"{change['old_target']} → {change['new_target']}"
            )
        lines.append("")
    
    # Verification methods
    if diff.added_verification_methods:
        lines.append("ADDED VERIFICATION METHODS:")
        for m in diff.added_verification_methods:
            lines.append(f"  + {m}")
        lines.append("")
    
    if diff.removed_verification_methods:
        lines.append("REMOVED VERIFICATION METHODS:")
        for m in diff.removed_verification_methods:
            lines.append(f"  - {m}")
        lines.append("")
    
    lines.append("=" * 70)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main() -> int:
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Compare two abstention taxonomy export files"
    )
    parser.add_argument(
        "--old",
        type=Path,
        required=True,
        help="Path to old taxonomy JSON file"
    )
    parser.add_argument(
        "--new",
        type=Path,
        required=True,
        help="Path to new taxonomy JSON file"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output structured JSON diff instead of human-readable"
    )
    parser.add_argument(
        "--fail-on-diff",
        action="store_true",
        help="Exit with code 1 if any changes detected"
    )
    parser.add_argument(
        "--fail-on-breaking",
        action="store_true",
        help="Exit with code 1 only if breaking changes detected"
    )
    
    args = parser.parse_args()
    
    try:
        old_data = load_taxonomy(args.old)
        new_data = load_taxonomy(args.new)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    
    diff = compute_diff(old_data, new_data)
    
    # Output
    if args.json:
        print(json.dumps(diff.to_dict(), indent=2))
    else:
        print(generate_human_report(diff))
    
    # Exit code
    if args.fail_on_diff and diff.has_changes:
        return 1
    if args.fail_on_breaking and diff.has_breaking_changes:
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

