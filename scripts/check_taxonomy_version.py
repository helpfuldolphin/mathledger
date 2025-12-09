#!/usr/bin/env python3
"""
Abstention Taxonomy Version Guard

Detects "silent" taxonomy changes - changes to the taxonomy without a
corresponding version bump. This ensures all taxonomy changes are explicit,
trackable, and evidence-pack friendly.

USAGE:
    # Check current code against exported semantics
    uv run python scripts/check_taxonomy_version.py --export artifacts/abstention_semantics.json

    # Generate current semantics to stdout (for debugging)
    uv run python scripts/check_taxonomy_version.py --dump-current

EXIT CODES:
    0 - Taxonomy and version are in sync
    1 - Taxonomy changed but version was not bumped (or export is stale)
    2 - Error (file not found, invalid JSON, etc.)

HOW IT WORKS:
    1. Loads the exported semantics JSON (e.g., from CI artifacts)
    2. Generates current in-memory semantics from the live code
    3. Compares:
       - taxonomy_version
       - Set of AbstentionType keys
       - Category assignments
       - Legacy mappings
    4. If content changed but version stayed the same → FAIL

DEVELOPER WORKFLOW:
    When you intentionally change the taxonomy:
    1. Make your changes to abstention_taxonomy.py / abstention_semantics.py
    2. Bump ABSTENTION_TAXONOMY_VERSION in abstention_semantics.py
    3. Regenerate export:
       uv run python -c "from rfl.verification.abstention_semantics import export_semantics; export_semantics('artifacts/abstention_semantics.json')"
    4. Run diff to verify changes:
       uv run python scripts/diff_abstention_taxonomy.py --old artifacts/abstention_semantics_prev.json --new artifacts/abstention_semantics.json
    5. Commit both the code changes and the updated export

PHASE II — VERIFICATION ZONE
Agent B6 (abstention-ops-6)
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Set, Any, Optional, Tuple

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_current_semantics() -> Dict[str, Any]:
    """
    Generate the current taxonomy semantics from live code.
    
    This mirrors what export_semantics() produces, but returns
    the data in memory rather than writing to file.
    """
    from rfl.verification.abstention_taxonomy import (
        AbstentionType,
        COMPLETE_MAPPING,
        ABSTENTION_METHOD_STRINGS,
    )
    from rfl.verification.abstention_semantics import (
        ABSTENTION_TAXONOMY_VERSION,
        ABSTENTION_TREE,
        SemanticCategory,
        get_types_for_category,
        get_schema_path,
    )
    
    # Build abstention types section
    abstention_types = {}
    for abst_type in AbstentionType:
        category = ABSTENTION_TREE.get(abst_type)
        legacy_keys = [
            key for key, mapped_type in COMPLETE_MAPPING.items()
            if mapped_type == abst_type
        ]
        abstention_types[abst_type.value] = {
            "category": category.value if category else None,
            "lean_specific": abst_type.value.startswith("abstain_lean_"),
            "legacy_keys": sorted(legacy_keys),
        }
    
    # Build categories section
    categories = {}
    for category in SemanticCategory:
        types_in_category = get_types_for_category(category)
        categories[category.value] = [t.value for t in types_in_category]
    
    # Build legacy mappings section
    legacy_mappings = {
        key: mapped_type.value 
        for key, mapped_type in COMPLETE_MAPPING.items()
    }
    
    return {
        "taxonomy_version": ABSTENTION_TAXONOMY_VERSION,
        "abstention_types": abstention_types,
        "categories": categories,
        "legacy_mappings": legacy_mappings,
        "verification_methods": sorted(ABSTENTION_METHOD_STRINGS),
    }


def load_exported_semantics(path: Path) -> Dict[str, Any]:
    """Load the exported semantics JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"Export file not found: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return data


def compare_semantics(
    exported: Dict[str, Any],
    current: Dict[str, Any],
) -> Tuple[bool, List[str]]:
    """
    Compare exported semantics against current code.
    
    Returns:
        (is_in_sync, list_of_differences)
        
    is_in_sync is True if:
        - Both have the same taxonomy_version, OR
        - Content is identical (version difference is acceptable if no content change)
    """
    differences: List[str] = []
    
    exported_version = exported.get("taxonomy_version", exported.get("version", "unknown"))
    current_version = current.get("taxonomy_version", "unknown")
    
    # Compare abstention types
    exported_types = set(exported.get("abstention_types", {}).keys())
    current_types = set(current.get("abstention_types", {}).keys())
    
    added_types = current_types - exported_types
    removed_types = exported_types - current_types
    
    if added_types:
        differences.append(f"Added types: {sorted(added_types)}")
    if removed_types:
        differences.append(f"Removed types: {sorted(removed_types)}")
    
    # Compare category assignments
    for type_name in exported_types & current_types:
        exported_cat = exported.get("abstention_types", {}).get(type_name, {}).get("category")
        current_cat = current.get("abstention_types", {}).get(type_name, {}).get("category")
        if exported_cat != current_cat:
            differences.append(f"Category changed for {type_name}: {exported_cat} → {current_cat}")
    
    # Compare categories
    exported_categories = set(exported.get("categories", {}).keys())
    current_categories = set(current.get("categories", {}).keys())
    
    added_categories = current_categories - exported_categories
    removed_categories = exported_categories - current_categories
    
    if added_categories:
        differences.append(f"Added categories: {sorted(added_categories)}")
    if removed_categories:
        differences.append(f"Removed categories: {sorted(removed_categories)}")
    
    # Compare legacy mappings
    exported_mappings = exported.get("legacy_mappings", {})
    current_mappings = current.get("legacy_mappings", {})
    
    added_mappings = set(current_mappings.keys()) - set(exported_mappings.keys())
    removed_mappings = set(exported_mappings.keys()) - set(current_mappings.keys())
    
    if added_mappings:
        differences.append(f"Added legacy mappings: {sorted(added_mappings)}")
    if removed_mappings:
        differences.append(f"Removed legacy mappings: {sorted(removed_mappings)}")
    
    for key in set(exported_mappings.keys()) & set(current_mappings.keys()):
        if exported_mappings[key] != current_mappings[key]:
            differences.append(
                f"Changed legacy mapping {key}: {exported_mappings[key]} → {current_mappings[key]}"
            )
    
    # Check version vs content consistency
    version_changed = exported_version != current_version
    content_changed = bool(differences)
    
    if content_changed and not version_changed:
        # This is the "silent bump" we want to catch
        return False, differences
    
    # If version changed but content didn't, that's OK (just a documentation bump)
    # If neither changed, all good
    # If both changed, all good (proper version bump)
    return True, differences


# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------

def generate_report(
    is_in_sync: bool,
    exported_version: str,
    current_version: str,
    differences: List[str],
) -> str:
    """Generate human-readable report."""
    lines = [
        "=" * 70,
        "TAXONOMY VERSION CHECK",
        "=" * 70,
        "",
        f"Exported version: {exported_version}",
        f"Current version:  {current_version}",
        "",
    ]
    
    if is_in_sync:
        lines.append("✅ Taxonomy and version are in sync")
    else:
        lines.append("❌ TAXONOMY CHANGED WITHOUT VERSION BUMP")
        lines.append("")
        lines.append("Detected changes:")
        for diff in differences:
            lines.append(f"  - {diff}")
        lines.append("")
        lines.append("To fix this:")
        lines.append("  1. Bump ABSTENTION_TAXONOMY_VERSION in abstention_semantics.py")
        lines.append("  2. Regenerate export:")
        lines.append('     uv run python -c "from rfl.verification.abstention_semantics import export_semantics; export_semantics(\'artifacts/abstention_semantics.json\')"')
        lines.append("  3. Commit both changes")
    
    lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main() -> int:
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Check taxonomy version consistency"
    )
    parser.add_argument(
        "--export",
        type=Path,
        help="Path to exported semantics JSON file to compare against"
    )
    parser.add_argument(
        "--dump-current",
        action="store_true",
        help="Dump current code semantics to stdout (JSON)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON instead of human-readable"
    )
    
    args = parser.parse_args()
    
    # Generate current semantics
    try:
        current = generate_current_semantics()
    except ImportError as e:
        print(f"ERROR: Failed to import taxonomy modules: {e}", file=sys.stderr)
        return 2
    
    # Dump mode - just output current semantics
    if args.dump_current:
        print(json.dumps(current, indent=2))
        return 0
    
    # Check mode - compare against export
    if not args.export:
        print("ERROR: --export is required unless using --dump-current", file=sys.stderr)
        return 2
    
    try:
        exported = load_exported_semantics(args.export)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    
    # Compare
    is_in_sync, differences = compare_semantics(exported, current)
    
    exported_version = exported.get("taxonomy_version", exported.get("version", "unknown"))
    current_version = current.get("taxonomy_version", "unknown")
    
    # Output
    if args.json:
        output = {
            "is_in_sync": is_in_sync,
            "exported_version": exported_version,
            "current_version": current_version,
            "differences": differences,
        }
        print(json.dumps(output, indent=2))
    else:
        print(generate_report(is_in_sync, exported_version, current_version, differences))
    
    return 0 if is_in_sync else 1


if __name__ == "__main__":
    sys.exit(main())

