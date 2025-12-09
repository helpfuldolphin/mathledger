#!/usr/bin/env python3
"""
Taxonomy Governance & Impact Analysis

Provides semantic analysis of taxonomy changes, governance reports,
and CI enforcement for breaking changes.

USAGE:
    # Analyze impact of taxonomy change
    uv run python scripts/taxonomy_governance.py analyze --old old.json --new new.json

    # Generate Markdown governance report
    uv run python scripts/taxonomy_governance.py report --old old.json --new new.json

    # CI check with breaking change acknowledgment
    uv run python scripts/taxonomy_governance.py ci-check --old old.json --new new.json
    
    # CI check with acknowledgment (allows breaking changes)
    UPDATE_TAXONOMY_ACK=1 uv run python scripts/taxonomy_governance.py ci-check --old old.json --new new.json

EXIT CODES:
    0 - No issues (or breaking changes acknowledged)
    1 - Breaking changes require acknowledgment
    2 - Error (file not found, invalid JSON, etc.)

RISK LEVELS:
    LOW    - Non-breaking changes only (additions, description updates)
    MEDIUM - Some breaking changes (legacy mapping changes)
    HIGH   - Major breaking changes (type/category removal, type migration)

PHASE III — GOVERNANCE
PHASE IV — METRICS, DOCS, CURRICULUM IMPACT
Agent B6 (abstention-ops-6)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Risk Level Enum
# ---------------------------------------------------------------------------

class RiskLevel(str, Enum):
    """Risk level for taxonomy changes."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    
    def __str__(self) -> str:
        return self.value


# ---------------------------------------------------------------------------
# Impact Data Structures
# ---------------------------------------------------------------------------

@dataclass
class BreakingChange:
    """Represents a single breaking change."""
    change_type: str  # e.g., "type_removed", "category_changed"
    description: str
    affected_component: str
    downstream_impact: str
    migration_hint: str


@dataclass 
class NonBreakingChange:
    """Represents a single non-breaking change."""
    change_type: str  # e.g., "type_added", "description_updated"
    description: str
    affected_component: str


@dataclass
class TaxonomyImpactAnalysis:
    """Complete impact analysis for taxonomy changes."""
    old_version: str
    new_version: str
    schema_version: str = "1.0.0"
    
    breaking_changes: List[BreakingChange] = field(default_factory=list)
    non_breaking_changes: List[NonBreakingChange] = field(default_factory=list)
    
    risk_level: RiskLevel = RiskLevel.LOW
    
    # Metadata
    analyzed_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    @property
    def has_breaking_changes(self) -> bool:
        return len(self.breaking_changes) > 0
    
    @property
    def has_changes(self) -> bool:
        return len(self.breaking_changes) > 0 or len(self.non_breaking_changes) > 0
    
    @property
    def total_changes(self) -> int:
        return len(self.breaking_changes) + len(self.non_breaking_changes)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "schema_version": self.schema_version,
            "old_version": self.old_version,
            "new_version": self.new_version,
            "risk_level": str(self.risk_level),
            "has_breaking_changes": self.has_breaking_changes,
            "has_changes": self.has_changes,
            "total_changes": self.total_changes,
            "breaking_changes": [
                {
                    "change_type": bc.change_type,
                    "description": bc.description,
                    "affected_component": bc.affected_component,
                    "downstream_impact": bc.downstream_impact,
                    "migration_hint": bc.migration_hint,
                }
                for bc in self.breaking_changes
            ],
            "non_breaking_changes": [
                {
                    "change_type": nc.change_type,
                    "description": nc.description,
                    "affected_component": nc.affected_component,
                }
                for nc in self.non_breaking_changes
            ],
            "analyzed_at": self.analyzed_at,
        }


# ---------------------------------------------------------------------------
# Impact Analysis Functions
# ---------------------------------------------------------------------------

def analyze_taxonomy_change(
    old: Dict[str, Any], 
    new: Dict[str, Any]
) -> TaxonomyImpactAnalysis:
    """
    Analyze the semantic impact of taxonomy changes.
    
    This goes beyond raw diff to explain potential downstream impact
    and provide migration guidance.
    
    Args:
        old: Old taxonomy export dictionary
        new: New taxonomy export dictionary
        
    Returns:
        TaxonomyImpactAnalysis with categorized changes and risk level
    """
    analysis = TaxonomyImpactAnalysis(
        old_version=old.get("taxonomy_version", old.get("version", "unknown")),
        new_version=new.get("taxonomy_version", new.get("version", "unknown")),
    )
    
    # Analyze abstention types
    _analyze_type_changes(old, new, analysis)
    
    # Analyze categories
    _analyze_category_changes(old, new, analysis)
    
    # Analyze legacy mappings
    _analyze_legacy_mapping_changes(old, new, analysis)
    
    # Analyze verification methods
    _analyze_verification_method_changes(old, new, analysis)
    
    # Analyze category assignments
    _analyze_category_assignments(old, new, analysis)
    
    # Determine risk level
    analysis.risk_level = _calculate_risk_level(analysis)
    
    return analysis


def _analyze_type_changes(
    old: Dict[str, Any], 
    new: Dict[str, Any], 
    analysis: TaxonomyImpactAnalysis
) -> None:
    """Analyze changes to AbstentionType values."""
    old_types = set(old.get("abstention_types", {}).keys())
    new_types = set(new.get("abstention_types", {}).keys())
    
    # Added types (non-breaking)
    for type_name in sorted(new_types - old_types):
        type_info = new.get("abstention_types", {}).get(type_name, {})
        category = type_info.get("category", "unknown")
        analysis.non_breaking_changes.append(NonBreakingChange(
            change_type="type_added",
            description=f"New abstention type '{type_name}' added to category '{category}'",
            affected_component=f"AbstentionType.{type_name.upper()}",
        ))
    
    # Removed types (breaking)
    for type_name in sorted(old_types - new_types):
        type_info = old.get("abstention_types", {}).get(type_name, {})
        legacy_keys = type_info.get("legacy_keys", [])
        analysis.breaking_changes.append(BreakingChange(
            change_type="type_removed",
            description=f"Abstention type '{type_name}' has been removed",
            affected_component=f"AbstentionType.{type_name.upper()}",
            downstream_impact=(
                f"Code referencing this type will fail. "
                f"Legacy keys affected: {legacy_keys}"
            ),
            migration_hint=(
                f"Replace references to '{type_name}' with an appropriate "
                f"alternative type, or handle the removal gracefully."
            ),
        ))


def _analyze_category_changes(
    old: Dict[str, Any], 
    new: Dict[str, Any], 
    analysis: TaxonomyImpactAnalysis
) -> None:
    """Analyze changes to SemanticCategory values."""
    old_categories = set(old.get("categories", {}).keys())
    new_categories = set(new.get("categories", {}).keys())
    
    # Added categories (non-breaking)
    for cat_name in sorted(new_categories - old_categories):
        analysis.non_breaking_changes.append(NonBreakingChange(
            change_type="category_added",
            description=f"New semantic category '{cat_name}' added",
            affected_component=f"SemanticCategory.{cat_name.upper()}",
        ))
    
    # Removed categories (breaking)
    for cat_name in sorted(old_categories - new_categories):
        old_types = old.get("categories", {}).get(cat_name, [])
        analysis.breaking_changes.append(BreakingChange(
            change_type="category_removed",
            description=f"Semantic category '{cat_name}' has been removed",
            affected_component=f"SemanticCategory.{cat_name.upper()}",
            downstream_impact=(
                f"Types previously in this category: {old_types}. "
                f"Dashboard aggregations using this category will break."
            ),
            migration_hint=(
                f"Update dashboard queries and aggregations to use "
                f"the new category structure."
            ),
        ))


def _analyze_category_assignments(
    old: Dict[str, Any], 
    new: Dict[str, Any], 
    analysis: TaxonomyImpactAnalysis
) -> None:
    """Analyze changes to type-to-category assignments."""
    old_types = old.get("abstention_types", {})
    new_types = new.get("abstention_types", {})
    
    for type_name in set(old_types.keys()) & set(new_types.keys()):
        old_cat = old_types[type_name].get("category")
        new_cat = new_types[type_name].get("category")
        
        if old_cat != new_cat:
            analysis.breaking_changes.append(BreakingChange(
                change_type="type_category_changed",
                description=(
                    f"Type '{type_name}' moved from category "
                    f"'{old_cat}' to '{new_cat}'"
                ),
                affected_component=f"AbstentionType.{type_name.upper()}",
                downstream_impact=(
                    f"categorize({type_name}) will return different value. "
                    f"Dashboard aggregations and filters may show different results."
                ),
                migration_hint=(
                    f"Update any code that assumes '{type_name}' belongs to "
                    f"'{old_cat}'. Review dashboard filters and reports."
                ),
            ))


def _analyze_legacy_mapping_changes(
    old: Dict[str, Any], 
    new: Dict[str, Any], 
    analysis: TaxonomyImpactAnalysis
) -> None:
    """Analyze changes to legacy key mappings."""
    old_mappings = old.get("legacy_mappings", {})
    new_mappings = new.get("legacy_mappings", {})
    
    old_keys = set(old_mappings.keys())
    new_keys = set(new_mappings.keys())
    
    # Added mappings (non-breaking)
    for key in sorted(new_keys - old_keys):
        target = new_mappings[key]
        analysis.non_breaking_changes.append(NonBreakingChange(
            change_type="legacy_mapping_added",
            description=f"New legacy key '{key}' now maps to '{target}'",
            affected_component=f"classify_breakdown_key('{key}')",
        ))
    
    # Removed mappings (breaking)
    for key in sorted(old_keys - new_keys):
        old_target = old_mappings[key]
        analysis.breaking_changes.append(BreakingChange(
            change_type="legacy_mapping_removed",
            description=f"Legacy key '{key}' (→ {old_target}) has been removed",
            affected_component=f"classify_breakdown_key('{key}')",
            downstream_impact=(
                f"Code using '{key}' will get None from classification. "
                f"Existing logs/data with this key may not be categorized."
            ),
            migration_hint=(
                f"Migrate data using '{key}' to a supported key, or add "
                f"backward compatibility handling."
            ),
        ))
    
    # Changed mappings (breaking)
    for key in sorted(old_keys & new_keys):
        if old_mappings[key] != new_mappings[key]:
            analysis.breaking_changes.append(BreakingChange(
                change_type="legacy_mapping_changed",
                description=(
                    f"Legacy key '{key}' target changed: "
                    f"'{old_mappings[key]}' → '{new_mappings[key]}'"
                ),
                affected_component=f"classify_breakdown_key('{key}')",
                downstream_impact=(
                    f"Historical data with '{key}' will now categorize differently. "
                    f"Dashboards showing trends may show discontinuities."
                ),
                migration_hint=(
                    f"Backfill historical data if needed, or document the "
                    f"classification change in release notes."
                ),
            ))


def _analyze_verification_method_changes(
    old: Dict[str, Any], 
    new: Dict[str, Any], 
    analysis: TaxonomyImpactAnalysis
) -> None:
    """Analyze changes to verification methods."""
    old_methods = set(old.get("verification_methods", []))
    new_methods = set(new.get("verification_methods", []))
    
    # Added methods (non-breaking)
    for method in sorted(new_methods - old_methods):
        analysis.non_breaking_changes.append(NonBreakingChange(
            change_type="verification_method_added",
            description=f"New verification method '{method}' added",
            affected_component=f"classify_verification_method('{method}')",
        ))
    
    # Removed methods (non-breaking, but notable)
    for method in sorted(old_methods - new_methods):
        analysis.non_breaking_changes.append(NonBreakingChange(
            change_type="verification_method_removed",
            description=f"Verification method '{method}' removed from list",
            affected_component=f"ABSTENTION_METHOD_STRINGS",
        ))


def _calculate_risk_level(analysis: TaxonomyImpactAnalysis) -> RiskLevel:
    """Calculate overall risk level based on changes."""
    if not analysis.has_changes:
        return RiskLevel.LOW
    
    # HIGH: Any type/category removal or type migration
    high_risk_types = {"type_removed", "category_removed", "type_category_changed"}
    for bc in analysis.breaking_changes:
        if bc.change_type in high_risk_types:
            return RiskLevel.HIGH
    
    # MEDIUM: Legacy mapping changes
    medium_risk_types = {"legacy_mapping_removed", "legacy_mapping_changed"}
    for bc in analysis.breaking_changes:
        if bc.change_type in medium_risk_types:
            return RiskLevel.MEDIUM
    
    # LOW: Only additions
    return RiskLevel.LOW


# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------

def render_taxonomy_change_report(analysis: TaxonomyImpactAnalysis) -> str:
    """
    Generate a Markdown governance report for taxonomy changes.
    
    This report is designed to be included in PRs when taxonomy changes.
    
    Args:
        analysis: The impact analysis result
        
    Returns:
        Markdown-formatted report string
    """
    lines = [
        "# 📋 Taxonomy Change Report",
        "",
        "## Summary",
        "",
        f"| Property | Value |",
        f"|----------|-------|",
        f"| Old Version | `{analysis.old_version}` |",
        f"| New Version | `{analysis.new_version}` |",
        f"| Risk Level | **{analysis.risk_level}** |",
        f"| Total Changes | {analysis.total_changes} |",
        f"| Breaking Changes | {len(analysis.breaking_changes)} |",
        f"| Non-Breaking Changes | {len(analysis.non_breaking_changes)} |",
        "",
    ]
    
    # Risk level badge/warning
    if analysis.risk_level == RiskLevel.HIGH:
        lines.extend([
            "## ⚠️ HIGH RISK - Breaking Changes Detected",
            "",
            "> **WARNING**: This change includes breaking modifications that may affect:",
            "> - Existing data classification",
            "> - Dashboard aggregations", 
            "> - Downstream code dependencies",
            "> ",
            "> **Explicit acknowledgment required for CI to pass.**",
            "",
        ])
    elif analysis.risk_level == RiskLevel.MEDIUM:
        lines.extend([
            "## ⚡ MEDIUM RISK - Legacy Mapping Changes",
            "",
            "> **CAUTION**: Legacy mapping changes may affect historical data classification.",
            "",
        ])
    
    # Breaking changes section
    if analysis.breaking_changes:
        lines.extend([
            "## 🔴 Breaking Changes",
            "",
            "| Type | Description | Affected Component |",
            "|------|-------------|-------------------|",
        ])
        for bc in analysis.breaking_changes:
            lines.append(
                f"| `{bc.change_type}` | {bc.description} | `{bc.affected_component}` |"
            )
        lines.append("")
        
        # Detailed breaking changes
        lines.extend([
            "### Breaking Change Details",
            "",
        ])
        for i, bc in enumerate(analysis.breaking_changes, 1):
            lines.extend([
                f"#### {i}. {bc.description}",
                "",
                f"- **Type**: `{bc.change_type}`",
                f"- **Component**: `{bc.affected_component}`",
                f"- **Downstream Impact**: {bc.downstream_impact}",
                f"- **Migration Hint**: {bc.migration_hint}",
                "",
            ])
    
    # Non-breaking changes section
    if analysis.non_breaking_changes:
        lines.extend([
            "## 🟢 Non-Breaking Changes",
            "",
            "| Type | Description | Affected Component |",
            "|------|-------------|-------------------|",
        ])
        for nc in analysis.non_breaking_changes:
            lines.append(
                f"| `{nc.change_type}` | {nc.description} | `{nc.affected_component}` |"
            )
        lines.append("")
    
    # No changes
    if not analysis.has_changes:
        lines.extend([
            "## ✅ No Changes Detected",
            "",
            "The taxonomy content is identical between versions.",
            "",
        ])
    
    # CI instructions
    if analysis.risk_level == RiskLevel.HIGH:
        lines.extend([
            "## 🔧 CI Instructions",
            "",
            "To acknowledge this breaking change and allow CI to pass:",
            "",
            "```bash",
            "# Set environment variable before running CI check",
            "export UPDATE_TAXONOMY_ACK=1",
            "",
            "# Or create marker file",
            "touch .taxonomy-breaking-change-ack",
            "```",
            "",
        ])
    
    # Footer
    lines.extend([
        "---",
        f"*Report generated at {analysis.analyzed_at}*",
        "",
    ])
    
    return "\n".join(lines)


def render_taxonomy_change_report_json(analysis: TaxonomyImpactAnalysis) -> str:
    """Render analysis as JSON string."""
    return json.dumps(analysis.to_dict(), indent=2)


# ---------------------------------------------------------------------------
# CI Check Functions
# ---------------------------------------------------------------------------

def check_breaking_changes_acknowledged(analysis: TaxonomyImpactAnalysis) -> tuple[bool, str]:
    """
    Check if breaking changes have been acknowledged.
    
    Acknowledgment can be provided via:
    - Environment variable: UPDATE_TAXONOMY_ACK=1
    - Marker file: .taxonomy-breaking-change-ack
    
    Returns:
        (is_acknowledged, message)
    """
    if analysis.risk_level != RiskLevel.HIGH:
        return True, "No acknowledgment required (risk level is not HIGH)"
    
    # Check environment variable
    if os.environ.get("UPDATE_TAXONOMY_ACK") == "1":
        return True, "Breaking changes acknowledged via UPDATE_TAXONOMY_ACK=1"
    
    # Check marker file
    marker_file = PROJECT_ROOT / ".taxonomy-breaking-change-ack"
    if marker_file.exists():
        return True, f"Breaking changes acknowledged via {marker_file}"
    
    return False, (
        "Breaking changes require explicit acknowledgment. "
        "Set UPDATE_TAXONOMY_ACK=1 or create .taxonomy-breaking-change-ack file."
    )


# ---------------------------------------------------------------------------
# File Loading
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


# ---------------------------------------------------------------------------
# Phase IV: Metrics, Docs, Curriculum Impact Analysis
# ---------------------------------------------------------------------------

def analyze_taxonomy_impact_on_metrics(
    analysis: TaxonomyImpactAnalysis,
    metrics_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Analyze how taxonomy changes affect metrics configuration.
    
    Checks if metrics reference removed types/categories and identifies
    categories that have no metric signals.
    
    Args:
        analysis: The taxonomy change impact analysis
        metrics_config: Metrics configuration dictionary
        
    Returns:
        Dictionary with:
        - affected_metric_kinds: List of metric kinds that reference removed types
        - non_covered_categories: Categories with no metric signals
        - status: "OK" | "PARTIAL" | "MISALIGNED"
    """
    result = {
        "affected_metric_kinds": [],
        "non_covered_categories": [],
        "status": "OK",
    }
    
    # Extract removed type names from analysis
    removed_type_names = []
    removed_category_names = []
    
    for bc in analysis.breaking_changes:
        if bc.change_type == "type_removed":
            # Extract type name from component (e.g., "AbstentionType.ABSTAIN_CRASH" -> "abstain_crash")
            component = bc.affected_component
            if "." in component:
                type_name = component.split(".")[-1].lower()
                # Convert ABSTAIN_CRASH -> abstain_crash
                if type_name.startswith("abstain_"):
                    removed_type_names.append(type_name)
                else:
                    removed_type_names.append(f"abstain_{type_name}")
        elif bc.change_type == "category_removed":
            component = bc.affected_component
            if "." in component:
                cat_name = component.split(".")[-1].lower()
                removed_category_names.append(cat_name)
    
    # Scan metrics_config for references to removed types/categories
    metrics_config_str = json.dumps(metrics_config).lower()
    
    affected_kinds = []
    for metric_kind, metric_def in metrics_config.get("metrics", {}).items():
        metric_str = json.dumps(metric_def).lower()
        
        # Check if metric references removed types
        for removed_type in removed_type_names:
            # Check various forms: "abstain_crash", "abstain-crash", "crash", etc.
            type_variants = [
                removed_type,
                removed_type.replace("_", "-"),
                removed_type.replace("abstain_", ""),
            ]
            if any(variant in metric_str for variant in type_variants):
                affected_kinds.append(metric_kind)
                break
        
        # Check if metric references removed categories
        for removed_cat in removed_category_names:
            cat_variants = [
                removed_cat,
                removed_cat.replace("_", "-"),
                removed_cat.replace("_related", ""),
            ]
            if any(variant in metric_str for variant in cat_variants):
                affected_kinds.append(metric_kind)
                break
    
    result["affected_metric_kinds"] = sorted(set(affected_kinds))
    
    # Identify categories with no metric signals
    # Standard categories from SemanticCategory enum
    all_categories = {
        "timeout_related", "crash_related", "resource_related",
        "oracle_related", "invalid_related"
    }
    
    covered_categories = set()
    for metric_def in metrics_config.get("metrics", {}).values():
        metric_str = json.dumps(metric_def).lower()
        for cat in all_categories:
            # Check for category name in various forms
            cat_variants = [
                cat,
                cat.replace("_", "-"),
                cat.replace("_related", ""),
                cat.replace("_", " "),
            ]
            if any(variant in metric_str for variant in cat_variants):
                covered_categories.add(cat)
    
    non_covered = sorted(all_categories - covered_categories)
    result["non_covered_categories"] = non_covered
    
    # Determine status
    if result["affected_metric_kinds"]:
        result["status"] = "MISALIGNED"
    elif result["non_covered_categories"]:
        result["status"] = "PARTIAL"
    else:
        result["status"] = "OK"
    
    return result


def analyze_taxonomy_alignment_with_docs_and_curriculum(
    analysis: TaxonomyImpactAnalysis,
    vocab_index: Dict[str, Any],
    curriculum_manifest: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Analyze alignment of taxonomy changes with documentation and curriculum.
    
    Checks if docs reference removed/renamed types and if curriculum slices
    use outdated taxonomy nodes.
    
    Args:
        analysis: The taxonomy change impact analysis
        vocab_index: Vocabulary index from documentation (maps terms to locations)
        curriculum_manifest: Curriculum configuration (from curriculum.yaml)
        
    Returns:
        Dictionary with:
        - missing_doc_updates: List of doc patterns needing updates
        - slices_with_outdated_types: Slices referencing removed/renamed types
        - alignment_status: "ALIGNED" | "PARTIAL" | "OUT_OF_DATE"
    """
    result = {
        "missing_doc_updates": [],
        "slices_with_outdated_types": [],
        "alignment_status": "ALIGNED",
    }
    
    # Extract removed/renamed types from analysis
    removed_type_names = []
    for bc in analysis.breaking_changes:
        if bc.change_type == "type_removed":
            # Extract type name from component (e.g., "AbstentionType.ABSTAIN_CRASH")
            component = bc.affected_component
            if "." in component:
                type_name = component.split(".")[-1].lower()
                if type_name.startswith("abstain_"):
                    removed_type_names.append(type_name)
                else:
                    removed_type_names.append(f"abstain_{type_name}")
    
    # Check vocab_index for references to removed types
    doc_updates_needed = []
    if vocab_index:
        for term, locations in vocab_index.items():
            term_lower = term.lower()
            for removed_type in removed_type_names:
                # Check various forms
                type_variants = [
                    removed_type,
                    removed_type.replace("_", "-"),
                    removed_type.replace("abstain_", ""),
                ]
                if any(variant in term_lower for variant in type_variants):
                    doc_updates_needed.append({
                        "term": term,
                        "locations": locations if isinstance(locations, list) else [locations],
                        "reason": f"References removed type: {removed_type}",
                    })
                    break
    
    result["missing_doc_updates"] = doc_updates_needed
    
    # Check curriculum slices for outdated types
    outdated_slices = []
    # Handle both old and new curriculum formats
    systems = curriculum_manifest.get("systems", {})
    if not systems:
        # Try direct slices key
        slices = curriculum_manifest.get("slices", [])
    else:
        # Get slices from pl system (or first system)
        pl_system = systems.get("pl", {})
        if not pl_system:
            pl_system = list(systems.values())[0] if systems else {}
        slices = pl_system.get("slices", [])
    
    for slice_def in slices:
        slice_name = slice_def.get("name", "unknown")
        slice_str = json.dumps(slice_def).lower()
        
        # Check if slice references removed types in gates or params
        for removed_type in removed_type_names:
            type_variants = [
                removed_type,
                removed_type.replace("_", "-"),
                removed_type.replace("abstain_", ""),
            ]
            if any(variant in slice_str for variant in type_variants):
                # Check which field contains it
                location = "unknown"
                if removed_type in json.dumps(slice_def.get("gates", {})).lower():
                    location = "gates"
                elif removed_type in json.dumps(slice_def.get("params", {})).lower():
                    location = "params"
                
                outdated_slices.append({
                    "slice_name": slice_name,
                    "outdated_type": removed_type,
                    "location": location,
                })
                break
    
    result["slices_with_outdated_types"] = outdated_slices
    
    # Determine alignment status
    if result["missing_doc_updates"] or result["slices_with_outdated_types"]:
        if result["missing_doc_updates"] and result["slices_with_outdated_types"]:
            result["alignment_status"] = "OUT_OF_DATE"
        else:
            result["alignment_status"] = "PARTIAL"
    else:
        result["alignment_status"] = "ALIGNED"
    
    return result


def build_taxonomy_director_panel(
    impact_metrics: Dict[str, Any],
    impact_docs_curriculum: Dict[str, Any],
    risk_analysis: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Build a global health taxonomy panel combining all impact analyses.
    
    This provides a single dashboard view of taxonomy health across
    metrics, docs, curriculum, and risk.
    
    Args:
        impact_metrics: Output from analyze_taxonomy_impact_on_metrics()
        impact_docs_curriculum: Output from analyze_taxonomy_alignment_with_docs_and_curriculum()
        risk_analysis: Output from analyze_taxonomy_change() (as dict)
        
    Returns:
        Dictionary with:
        - status_light: "GREEN" | "YELLOW" | "RED"
        - risk_level: "LOW" | "MEDIUM" | "HIGH"
        - alignment_status: Combined alignment status
        - headline: Short neutral summary string
        - requires_ack: Whether acknowledgment is required
    """
    risk_level = risk_analysis.get("risk_level", "LOW")
    metrics_status = impact_metrics.get("status", "OK")
    alignment_status = impact_docs_curriculum.get("alignment_status", "ALIGNED")
    
    # Determine status_light
    if risk_level == "HIGH" or metrics_status == "MISALIGNED" or alignment_status == "OUT_OF_DATE":
        status_light = "RED"
    elif risk_level == "MEDIUM" or metrics_status == "PARTIAL" or alignment_status == "PARTIAL":
        status_light = "YELLOW"
    else:
        status_light = "GREEN"
    
    # Determine requires_ack (from existing CI guard logic)
    requires_ack = risk_level == "HIGH"
    
    # Build headline
    headline_parts = []
    if risk_level != "LOW":
        headline_parts.append(f"Risk: {risk_level}")
    if metrics_status != "OK":
        headline_parts.append(f"Metrics: {metrics_status}")
    if alignment_status != "ALIGNED":
        headline_parts.append(f"Alignment: {alignment_status}")
    
    if headline_parts:
        headline = " | ".join(headline_parts)
    else:
        headline = "Taxonomy stable and aligned"
    
    return {
        "status_light": status_light,
        "risk_level": risk_level,
        "alignment_status": alignment_status,
        "headline": headline,
        "requires_ack": requires_ack,
        "metrics_status": metrics_status,
        "affected_metric_kinds": impact_metrics.get("affected_metric_kinds", []),
        "non_covered_categories": impact_metrics.get("non_covered_categories", []),
        "missing_doc_updates": len(impact_docs_curriculum.get("missing_doc_updates", [])),
        "outdated_slices": len(impact_docs_curriculum.get("slices_with_outdated_types", [])),
    }


# ---------------------------------------------------------------------------
# Phase IV Extension: Runbook & Evidence Capsule
# ---------------------------------------------------------------------------

def build_taxonomy_change_runbook(
    impact_metrics: Dict[str, Any],
    impact_docs_curriculum: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build an ordered runbook for taxonomy change propagation.
    
    This is descriptive only - provides a checklist of steps to follow
    when making taxonomy changes. Does not execute any actions.
    
    Args:
        impact_metrics: Output from analyze_taxonomy_impact_on_metrics()
        impact_docs_curriculum: Output from analyze_taxonomy_alignment_with_docs_and_curriculum()
        
    Returns:
        Dictionary with:
        - schema_version: "1.0.0"
        - steps: List of ordered runbook entries
        - blocking_items: List of items that must be addressed
        - advisory_items: List of recommended items
    """
    steps = []
    blocking_items = []
    advisory_items = []
    
    # Step 1: Update taxonomy code (always first)
    steps.append({
        "id": "update_taxonomy_code",
        "description": "Update abstention_taxonomy.py and abstention_semantics.py with taxonomy changes",
        "status": "PENDING",
        "order": 1,
    })
    
    # Step 2: Bump version
    steps.append({
        "id": "bump_version",
        "description": "Increment ABSTENTION_TAXONOMY_VERSION in abstention_semantics.py",
        "status": "PENDING",
        "order": 2,
    })
    
    # Step 3: Regenerate export
    steps.append({
        "id": "regenerate_export",
        "description": "Run export_semantics() to generate updated artifacts/abstention_semantics.json",
        "status": "PENDING",
        "order": 3,
    })
    
    # Step 4: Update metrics if affected
    affected_metric_kinds = impact_metrics.get("affected_metric_kinds", [])
    if affected_metric_kinds:
        blocking_items.append(f"Metrics configuration references removed types: {', '.join(affected_metric_kinds)}")
        steps.append({
            "id": "update_metrics_config",
            "description": f"Update metrics configuration to remove references to: {', '.join(affected_metric_kinds)}",
            "status": "PENDING",
            "order": 4,
        })
    else:
        steps.append({
            "id": "verify_metrics_config",
            "description": "Verify metrics configuration is compatible with taxonomy changes",
            "status": "PENDING",
            "order": 4,
        })
    
    # Step 5: Add metrics for new categories if needed
    non_covered = impact_metrics.get("non_covered_categories", [])
    if non_covered:
        advisory_items.append(f"Categories without metric signals: {', '.join(non_covered)}")
        steps.append({
            "id": "add_category_metrics",
            "description": f"Consider adding metric signals for categories: {', '.join(non_covered)}",
            "status": "PENDING",
            "order": 5,
        })
    
    # Step 6: Update documentation
    missing_doc_updates = impact_docs_curriculum.get("missing_doc_updates", [])
    if missing_doc_updates:
        blocking_items.append(f"Documentation references removed types: {len(missing_doc_updates)} locations")
        doc_locations = []
        for update in missing_doc_updates[:5]:  # Limit to first 5 for brevity
            doc_locations.extend(update.get("locations", []))
        
        steps.append({
            "id": "update_documentation",
            "description": f"Update documentation at {len(set(doc_locations))} locations referencing removed types",
            "status": "PENDING",
            "order": 6,
            "details": [{"term": u["term"], "locations": u["locations"]} for u in missing_doc_updates[:5]],
        })
    else:
        steps.append({
            "id": "verify_documentation",
            "description": "Verify documentation is aligned with taxonomy changes",
            "status": "PENDING",
            "order": 6,
        })
    
    # Step 7: Update curriculum slices
    outdated_slices = impact_docs_curriculum.get("slices_with_outdated_types", [])
    if outdated_slices:
        blocking_items.append(f"Curriculum slices reference removed types: {len(outdated_slices)} slices")
        slice_names = [s["slice_name"] for s in outdated_slices]
        steps.append({
            "id": "update_curriculum_slices",
            "description": f"Update curriculum.yaml slices: {', '.join(slice_names)}",
            "status": "PENDING",
            "order": 7,
            "details": outdated_slices,
        })
    else:
        steps.append({
            "id": "verify_curriculum_slices",
            "description": "Verify curriculum.yaml slices are aligned with taxonomy changes",
            "status": "PENDING",
            "order": 7,
        })
    
    # Step 8: Run tests
    steps.append({
        "id": "run_tests",
        "description": "Run full test suite: pytest tests/rfl/test_abstention_*.py tests/rfl/test_taxonomy_*.py",
        "status": "PENDING",
        "order": 8,
    })
    
    # Step 9: Generate diff report
    steps.append({
        "id": "generate_diff_report",
        "description": "Run diff_abstention_taxonomy.py to generate change report",
        "status": "PENDING",
        "order": 9,
    })
    
    # Step 10: CI acknowledgment (if needed)
    if blocking_items:
        steps.append({
            "id": "acknowledge_breaking_changes",
            "description": "Set UPDATE_TAXONOMY_ACK=1 or create .taxonomy-breaking-change-ack file for CI",
            "status": "PENDING",
            "order": 10,
        })
    
    # Step 11: Commit
    steps.append({
        "id": "commit_changes",
        "description": "Commit all changes with message: 'taxonomy: bump to vX.Y.Z - <description>'",
        "status": "PENDING",
        "order": 11,
    })
    
    return {
        "schema_version": "1.0.0",
        "steps": sorted(steps, key=lambda x: x["order"]),
        "blocking_items": blocking_items,
        "advisory_items": advisory_items,
        "total_steps": len(steps),
        "blocking_count": len(blocking_items),
        "advisory_count": len(advisory_items),
    }


def summarize_taxonomy_change_for_evidence(
    analysis: Dict[str, Any],
    panel: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Create a compact evidence tile summarizing taxonomy changes.
    
    This provides a neutral, factual summary suitable for audit trails,
    evidence packs, and governance documentation. Uses descriptive language
    only - no normative judgments.
    
    Args:
        analysis: Output from analyze_taxonomy_change() (as dict)
        panel: Output from build_taxonomy_director_panel()
        
    Returns:
        Dictionary with:
        - change_magnitude: "LOW" | "MEDIUM" | "HIGH"
        - alignment_status: From panel
        - metrics_impacted: Count of affected metric kinds
        - docs_impacted: Count of documentation locations needing updates
        - requires_ack: Boolean from panel
        - breaking_changes_count: Number of breaking changes
        - non_breaking_changes_count: Number of non-breaking changes
        - version_from: Old version
        - version_to: New version
    """
    risk_level = panel.get("risk_level", "LOW")
    alignment_status = panel.get("alignment_status", "ALIGNED")
    
    # Determine change_magnitude from risk_level and alignment
    if risk_level == "HIGH" or alignment_status == "OUT_OF_DATE":
        change_magnitude = "HIGH"
    elif risk_level == "MEDIUM" or alignment_status == "PARTIAL":
        change_magnitude = "MEDIUM"
    else:
        change_magnitude = "LOW"
    
    breaking_changes = analysis.get("breaking_changes", [])
    non_breaking_changes = analysis.get("non_breaking_changes", [])
    
    return {
        "change_magnitude": change_magnitude,
        "alignment_status": alignment_status,
        "metrics_impacted": len(panel.get("affected_metric_kinds", [])),
        "docs_impacted": panel.get("missing_doc_updates", 0),
        "requires_ack": panel.get("requires_ack", False),
        "breaking_changes_count": len(breaking_changes),
        "non_breaking_changes_count": len(non_breaking_changes),
        "version_from": analysis.get("old_version", "unknown"),
        "version_to": analysis.get("new_version", "unknown"),
        "risk_level": risk_level,
        "status_light": panel.get("status_light", "GREEN"),
    }


# ---------------------------------------------------------------------------
# Phase V: Semantic Integrity Grid
# ---------------------------------------------------------------------------

def build_taxonomy_integrity_radar(
    metrics_impact: Dict[str, Any],
    docs_alignment: Dict[str, Any],
    curriculum_alignment: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build the Taxonomy Integrity Radar - unified view of taxonomy health.
    
    Aggregates metrics, docs, and curriculum alignment into a single
    integrity status with alignment score.
    
    Args:
        metrics_impact: Output from analyze_taxonomy_impact_on_metrics()
        docs_alignment: Output from analyze_taxonomy_alignment_with_docs_and_curriculum()
        curriculum_alignment: Same as docs_alignment (for clarity)
        
    Returns:
        Dictionary with:
        - schema_version: "1.0.0"
        - integrity_status: "OK" | "WARN" | "BLOCK"
        - affected_metrics: List of affected metric kinds
        - affected_docs: List of doc update requirements
        - affected_slices: List of affected curriculum slices
        - alignment_score: Float 0.0-1.0 (1.0 = perfect alignment)
        - summary: Neutral summary string
    """
    # Extract affected items
    affected_metrics = metrics_impact.get("affected_metric_kinds", [])
    affected_docs = docs_alignment.get("missing_doc_updates", [])
    affected_slices = docs_alignment.get("slices_with_outdated_types", [])
    
    # Determine integrity status
    # BLOCK: Curriculum slices affected (critical - breaks runtime)
    # WARN: Docs or metrics out of date (non-critical but needs attention)
    # OK: Everything aligned
    if affected_slices:
        integrity_status = "BLOCK"
    elif affected_docs or affected_metrics or metrics_impact.get("status") != "OK":
        integrity_status = "WARN"
    else:
        integrity_status = "OK"
    
    # Calculate alignment score (0.0 = misaligned, 1.0 = perfect)
    # Score components:
    # - Metrics: 0.3 weight (OK=1.0, PARTIAL=0.5, MISALIGNED=0.0)
    # - Docs: 0.3 weight (ALIGNED=1.0, PARTIAL=0.5, OUT_OF_DATE=0.0)
    # - Curriculum: 0.4 weight (no affected slices=1.0, affected=0.0)
    
    metrics_status = metrics_impact.get("status", "OK")
    docs_status = docs_alignment.get("alignment_status", "ALIGNED")
    
    metrics_score = {
        "OK": 1.0,
        "PARTIAL": 0.5,
        "MISALIGNED": 0.0,
    }.get(metrics_status, 0.0)
    
    docs_score = {
        "ALIGNED": 1.0,
        "PARTIAL": 0.5,
        "OUT_OF_DATE": 0.0,
    }.get(docs_status, 0.0)
    
    curriculum_score = 1.0 if not affected_slices else 0.0
    
    alignment_score = (
        metrics_score * 0.3 +
        docs_score * 0.3 +
        curriculum_score * 0.4
    )
    
    # Build summary
    summary_parts = []
    if integrity_status == "BLOCK":
        summary_parts.append(f"{len(affected_slices)} curriculum slice(s) require updates")
    if affected_metrics:
        summary_parts.append(f"{len(affected_metrics)} metric(s) affected")
    if affected_docs:
        summary_parts.append(f"{len(affected_docs)} documentation location(s) need updates")
    
    if summary_parts:
        summary = ". ".join(summary_parts) + "."
    else:
        summary = "Taxonomy integrity maintained across all systems."
    
    return {
        "schema_version": "1.0.0",
        "integrity_status": integrity_status,
        "affected_metrics": affected_metrics,
        "affected_docs": [doc.get("term", "unknown") for doc in affected_docs[:10]],  # Limit to 10
        "affected_slices": [slice.get("slice_name", "unknown") for slice in affected_slices],
        "alignment_score": round(alignment_score, 3),
        "summary": summary,
        "metrics_status": metrics_status,
        "docs_status": docs_status,
        "curriculum_status": "ALIGNED" if not affected_slices else "OUT_OF_DATE",
    }


def build_global_console_tile(
    radar: Dict[str, Any],
    risk_analysis: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Create a Global Console Tile for high-level dashboard view.
    
    Provides executive-level summary of taxonomy health.
    
    Args:
        radar: Output from build_taxonomy_integrity_radar()
        risk_analysis: Output from analyze_taxonomy_change() (as dict)
        
    Returns:
        Dictionary with:
        - status_light: "GREEN" | "YELLOW" | "RED"
        - taxonomy_alignment_status: From radar
        - critical_breaks_count: Count of blocking issues
        - headline: Neutral summary string
    """
    integrity_status = radar.get("integrity_status", "OK")
    alignment_score = radar.get("alignment_score", 1.0)
    affected_slices = radar.get("affected_slices", [])
    
    # Determine status_light
    if integrity_status == "BLOCK" or alignment_score < 0.5:
        status_light = "RED"
    elif integrity_status == "WARN" or alignment_score < 0.8:
        status_light = "YELLOW"
    else:
        status_light = "GREEN"
    
    # Count critical breaks (curriculum slices are critical)
    critical_breaks_count = len(affected_slices)
    
    # Build headline
    if critical_breaks_count > 0:
        headline = f"Taxonomy integrity: {critical_breaks_count} critical break(s) detected"
    elif integrity_status == "WARN":
        headline = "Taxonomy integrity: warnings present, alignment partial"
    else:
        headline = "Taxonomy integrity: all systems aligned"
    
    return {
        "status_light": status_light,
        "taxonomy_alignment_status": radar.get("integrity_status", "OK"),
        "critical_breaks_count": critical_breaks_count,
        "headline": headline,
        "alignment_score": alignment_score,
        "affected_metrics_count": len(radar.get("affected_metrics", [])),
        "affected_docs_count": len(radar.get("affected_docs", [])),
    }


def evaluate_taxonomy_for_ci(radar: Dict[str, Any]) -> tuple[int, str]:
    """
    Evaluate taxonomy integrity for CI blocking.
    
    Exit code rules:
    - BLOCK (exit 1): Any curriculum slice referencing removed types
    - WARN (exit 0 with message): Docs or metrics out of date
    - OK (exit 0): Everything aligned
    
    Args:
        radar: Output from build_taxonomy_integrity_radar()
        
    Returns:
        Tuple of (exit_code, message)
        - exit_code: 0 (OK/WARN) or 1 (BLOCK)
        - message: Human-readable status message
    """
    integrity_status = radar.get("integrity_status", "OK")
    affected_slices = radar.get("affected_slices", [])
    affected_metrics = radar.get("affected_metrics", [])
    affected_docs = radar.get("affected_docs", [])
    
    if integrity_status == "BLOCK":
        # BLOCK: Curriculum slices affected
        slice_names = ", ".join(affected_slices)
        message = (
            f"BLOCK: {len(affected_slices)} curriculum slice(s) reference removed types: {slice_names}. "
            f"Update curriculum.yaml before proceeding."
        )
        return 1, message
    
    elif integrity_status == "WARN":
        # WARN: Docs or metrics out of date
        warn_parts = []
        if affected_metrics:
            warn_parts.append(f"{len(affected_metrics)} metric(s) affected")
        if affected_docs:
            warn_parts.append(f"{len(affected_docs)} doc location(s) need updates")
        
        message = f"WARN: {', '.join(warn_parts)}. Review and update as needed."
        return 0, message
    
    else:
        # OK: Everything aligned
        message = "OK: Taxonomy integrity maintained across all systems."
        return 0, message


def build_taxonomy_drift_timeline(
    historical_impacts: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Generate Taxonomy Drift Timeline from historical impact objects.
    
    Analyzes a sequence of taxonomy changes to identify drift patterns.
    
    Args:
        historical_impacts: List of impact analysis dictionaries (from analyze_taxonomy_change)
        
    Returns:
        Dictionary with:
        - drift_band: "STABLE" | "LOW_DRIFT" | "MEDIUM_DRIFT" | "HIGH_DRIFT"
        - change_intensity: Float 0.0-1.0 (cumulative change intensity)
        - first_break_index: Index of first breaking change (or -1 if none)
        - summary: Neutral summary string
    """
    if not historical_impacts:
        return {
            "drift_band": "STABLE",
            "change_intensity": 0.0,
            "first_break_index": -1,
            "summary": "No taxonomy changes recorded.",
        }
    
    # Analyze historical impacts
    total_changes = 0
    breaking_changes = 0
    first_break_index = -1
    
    for i, impact in enumerate(historical_impacts):
        breaking_count = len(impact.get("breaking_changes", []))
        non_breaking_count = len(impact.get("non_breaking_changes", []))
        
        total_changes += breaking_count + non_breaking_count
        breaking_changes += breaking_count
        
        if breaking_count > 0 and first_break_index == -1:
            first_break_index = i
    
    # Calculate change intensity
    # Intensity = (breaking_changes * 2 + non_breaking_changes) / (total_impacts * max_expected_changes)
    # Normalize to 0.0-1.0 range
    non_breaking_changes = total_changes - breaking_changes
    max_expected_changes = len(historical_impacts) * 10  # Assume max 10 changes per impact
    
    if max_expected_changes > 0:
        change_intensity = min(1.0, (breaking_changes * 2 + non_breaking_changes) / max_expected_changes)
    else:
        change_intensity = 0.0
    
    # Determine drift band
    if change_intensity == 0.0:
        drift_band = "STABLE"
    elif change_intensity < 0.2:
        drift_band = "LOW_DRIFT"
    elif change_intensity < 0.5:
        drift_band = "MEDIUM_DRIFT"
    else:
        drift_band = "HIGH_DRIFT"
    
    # Build summary
    if breaking_changes > 0:
        summary = (
            f"Taxonomy drift timeline: {len(historical_impacts)} change(s) analyzed, "
            f"{breaking_changes} breaking change(s), {non_breaking_changes} non-breaking change(s). "
            f"First breaking change at index {first_break_index}."
        )
    else:
        summary = (
            f"Taxonomy drift timeline: {len(historical_impacts)} change(s) analyzed, "
            f"{non_breaking_changes} non-breaking change(s). No breaking changes detected."
        )
    
    return {
        "drift_band": drift_band,
        "change_intensity": round(change_intensity, 3),
        "first_break_index": first_break_index,
        "summary": summary,
        "total_impacts": len(historical_impacts),
        "total_breaking_changes": breaking_changes,
        "total_non_breaking_changes": non_breaking_changes,
    }


# ---------------------------------------------------------------------------
# CLI Commands
# ---------------------------------------------------------------------------

def cmd_analyze(args: argparse.Namespace) -> int:
    """Run impact analysis command."""
    try:
        old_data = load_taxonomy(args.old)
        new_data = load_taxonomy(args.new)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    
    analysis = analyze_taxonomy_change(old_data, new_data)
    
    if args.json:
        print(render_taxonomy_change_report_json(analysis))
    else:
        # Print summary
        print("=" * 70)
        print("TAXONOMY CHANGE IMPACT ANALYSIS")
        print("=" * 70)
        print(f"Old Version: {analysis.old_version}")
        print(f"New Version: {analysis.new_version}")
        print(f"Risk Level: {analysis.risk_level}")
        print(f"Breaking Changes: {len(analysis.breaking_changes)}")
        print(f"Non-Breaking Changes: {len(analysis.non_breaking_changes)}")
        print("=" * 70)
        
        if analysis.breaking_changes:
            print("\nBREAKING CHANGES:")
            for bc in analysis.breaking_changes:
                print(f"  [{bc.change_type}] {bc.description}")
        
        if analysis.non_breaking_changes:
            print("\nNON-BREAKING CHANGES:")
            for nc in analysis.non_breaking_changes:
                print(f"  [{nc.change_type}] {nc.description}")
    
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Generate governance report command."""
    try:
        old_data = load_taxonomy(args.old)
        new_data = load_taxonomy(args.new)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    
    analysis = analyze_taxonomy_change(old_data, new_data)
    
    if args.json:
        print(render_taxonomy_change_report_json(analysis))
    else:
        print(render_taxonomy_change_report(analysis))
    
    # Write to file if specified
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            if args.json:
                f.write(render_taxonomy_change_report_json(analysis))
            else:
                f.write(render_taxonomy_change_report(analysis))
        print(f"\nReport written to: {output_path}", file=sys.stderr)
    
    return 0


def cmd_ci_check(args: argparse.Namespace) -> int:
    """CI check with breaking change acknowledgment."""
    try:
        old_data = load_taxonomy(args.old)
        new_data = load_taxonomy(args.new)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    
    analysis = analyze_taxonomy_change(old_data, new_data)
    
    print("=" * 70)
    print("TAXONOMY CI CHECK")
    print("=" * 70)
    print(f"Old Version: {analysis.old_version}")
    print(f"New Version: {analysis.new_version}")
    print(f"Risk Level: {analysis.risk_level}")
    print(f"Breaking Changes: {len(analysis.breaking_changes)}")
    print()
    
    if analysis.risk_level == RiskLevel.HIGH:
        acknowledged, message = check_breaking_changes_acknowledged(analysis)
        
        if acknowledged:
            print(f"✅ {message}")
            print()
            print("Breaking changes summary:")
            for bc in analysis.breaking_changes:
                print(f"  - [{bc.change_type}] {bc.description}")
            print()
            print("=" * 70)
            return 0
        else:
            print(f"❌ {message}")
            print()
            print("Breaking changes that require acknowledgment:")
            for bc in analysis.breaking_changes:
                print(f"  - [{bc.change_type}] {bc.description}")
                print(f"    Impact: {bc.downstream_impact}")
            print()
            print("To acknowledge, either:")
            print("  1. Set environment variable: UPDATE_TAXONOMY_ACK=1")
            print("  2. Create marker file: .taxonomy-breaking-change-ack")
            print()
            print("=" * 70)
            return 1
    
    print("✅ No breaking changes require acknowledgment")
    print("=" * 70)
    return 0


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Taxonomy governance and impact analysis tools"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # analyze command
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze impact of taxonomy changes"
    )
    analyze_parser.add_argument("--old", type=Path, required=True, help="Old taxonomy JSON")
    analyze_parser.add_argument("--new", type=Path, required=True, help="New taxonomy JSON")
    analyze_parser.add_argument("--json", action="store_true", help="Output JSON")
    
    # report command
    report_parser = subparsers.add_parser(
        "report",
        help="Generate Markdown governance report"
    )
    report_parser.add_argument("--old", type=Path, required=True, help="Old taxonomy JSON")
    report_parser.add_argument("--new", type=Path, required=True, help="New taxonomy JSON")
    report_parser.add_argument("--json", action="store_true", help="Output JSON instead of Markdown")
    report_parser.add_argument("--output", "-o", type=Path, help="Write report to file")
    
    # ci-check command
    ci_parser = subparsers.add_parser(
        "ci-check",
        help="CI check with breaking change acknowledgment"
    )
    ci_parser.add_argument("--old", type=Path, required=True, help="Old taxonomy JSON")
    ci_parser.add_argument("--new", type=Path, required=True, help="New taxonomy JSON")
    
    args = parser.parse_args()
    
    if args.command == "analyze":
        return cmd_analyze(args)
    elif args.command == "report":
        return cmd_report(args)
    elif args.command == "ci-check":
        return cmd_ci_check(args)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

