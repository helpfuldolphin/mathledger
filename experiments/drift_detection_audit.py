"""
PHASE II — NOT USED IN PHASE I

Cross-File Drift Detection and Documentation Linter.

This module extends the semantic consistency auditor with:
1. Cross-file drift detection between all Phase II artifacts
2. Documentation linter for Markdown structure and formula references
3. CI-suitable severity grading (OK / WARNING / FAIL)

Artifacts compared:
- docs/PHASE2_RFL_UPLIFT_PLAN.md: Narrative documentation
- RFL_UPLIFT_THEORY.md: Formal theory definitions
- config/curriculum_uplift_phase2.yaml: Slice definitions
- experiments/slice_success_metrics.py: Metric implementations
- experiments/prereg/PREREG_UPLIFT_U2.yaml: Preregistration templates

Absolute Safeguards:
- Semantic audits must never alter any metrics.
- No modification to snippet semantics or governance.
- This is a DETECTION-ONLY auditor.
"""

from __future__ import annotations

import re
import argparse
import json
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

from experiments.semantic_consistency_audit import (
    MetricKind,
    AuditStatus,
    AuditIssue,
    SliceSemanticSpec,
    AuditReport,
    METRIC_REQUIRED_PARAMS,
    METRIC_OPTIONAL_PARAMS,
    DOC_SLICE_METRIC_MAP,
    METRIC_FUNCTION_MAP,
    load_curriculum_yaml,
    load_prereg_yaml,
    extract_metric_functions,
    extract_slice_specs,
    run_static_checks,
    run_runtime_sanity_checks,
    check_global_consistency,
)


# =============================================================================
# CI SEVERITY GRADING
# =============================================================================

class DriftSeverity(str, Enum):
    """CI-suitable severity grading."""
    OK = "OK"              # All consistent, no drift
    WARNING = "WARNING"    # Non-breaking inconsistency (cosmetic, documentation lag)
    FAIL = "FAIL"          # Breaking inconsistency (metric-doc mismatch, missing refs)


@dataclass
class DriftFinding:
    """Represents a cross-file drift finding."""
    severity: DriftSeverity
    category: str
    source_file: str
    target_file: str
    message: str
    details: Optional[Dict[str, Any]] = None


@dataclass
class DocumentationFinding:
    """Represents a documentation linter finding."""
    severity: DriftSeverity
    category: str
    file: str
    line: Optional[int]
    message: str
    details: Optional[Dict[str, Any]] = None


@dataclass
class DriftReport:
    """Complete drift detection report."""
    drift_findings: List[DriftFinding] = field(default_factory=list)
    doc_findings: List[DocumentationFinding] = field(default_factory=list)
    formula_findings: List[DriftFinding] = field(default_factory=list)
    base_audit: Optional[AuditReport] = None
    
    @property
    def status(self) -> DriftSeverity:
        """Overall drift status."""
        all_findings = (
            self.drift_findings + 
            self.formula_findings + 
            [DocumentationFinding(f.severity, f.category, f.file, f.line, f.message)
             for f in self.doc_findings]
        )
        if any(f.severity == DriftSeverity.FAIL for f in all_findings):
            return DriftSeverity.FAIL
        if any(f.severity == DriftSeverity.WARNING for f in all_findings):
            return DriftSeverity.WARNING
        return DriftSeverity.OK


# =============================================================================
# DOCUMENTATION LINTER
# =============================================================================

# Expected headers in PHASE2_RFL_UPLIFT_PLAN.md
EXPECTED_PLAN_HEADERS = [
    "Phase II: RFL Uplift Plan",
    "Overview",
    "Phase II Uplift Slices",
    "slice_uplift_goal",
    "slice_uplift_sparse", 
    "slice_uplift_tree",
    "slice_uplift_dependency",
    "Success Metric",
    "Parameters",
    "Expected Uplift",
]

# Expected headers in RFL_UPLIFT_THEORY.md
EXPECTED_THEORY_HEADERS = [
    "RFL Uplift Theory",
    "Abstract",
    "Preliminaries",
    "Definition",
    "Slice-Specific Success Metrics",
]

# Regex patterns for extracting elements from documentation
SLICE_NAME_PATTERN = re.compile(r'`(slice_uplift_\w+)`')
METRIC_KIND_PATTERN = re.compile(r'(goal_hit|density|chain_length|multi_goal)', re.IGNORECASE)
HASH_PATTERN = re.compile(r'[a-f0-9]{64}')
FORMULA_PATTERN = re.compile(r'`([^`]+(?:->|/\\|\\\/|~)[^`]+)`')
CODE_BLOCK_METRIC_PATTERN = re.compile(r'success\s*=\s*\(([^)]+)\)', re.DOTALL)


def load_markdown_file(path: Path) -> Tuple[str, List[str]]:
    """Load a markdown file and return content and lines."""
    if not path.exists():
        return "", []
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    return content, content.splitlines()


def extract_headers_from_markdown(content: str) -> List[Tuple[int, str, int]]:
    """
    Extract headers from markdown content.
    Returns list of (level, header_text, line_number).
    """
    headers = []
    for i, line in enumerate(content.splitlines(), 1):
        match = re.match(r'^(#{1,6})\s+(.+)$', line.strip())
        if match:
            level = len(match.group(1))
            text = match.group(2).strip()
            headers.append((level, text, i))
    return headers


def extract_slice_references(content: str) -> Set[str]:
    """Extract all slice name references from markdown."""
    return set(SLICE_NAME_PATTERN.findall(content))


def extract_metric_kind_references(content: str) -> Set[str]:
    """Extract all metric kind references from markdown."""
    return {m.lower() for m in METRIC_KIND_PATTERN.findall(content)}


def extract_hash_references(content: str) -> Set[str]:
    """Extract all SHA256 hash references from markdown."""
    return set(HASH_PATTERN.findall(content))


def extract_formula_references(content: str) -> Set[str]:
    """Extract formula-like strings from markdown code blocks."""
    return set(FORMULA_PATTERN.findall(content))


def lint_documentation(
    plan_path: Path,
    theory_path: Path,
) -> List[DocumentationFinding]:
    """
    Lint documentation files for expected structure.
    """
    findings = []
    
    # Check PHASE2_RFL_UPLIFT_PLAN.md
    if plan_path.exists():
        plan_content, _ = load_markdown_file(plan_path)
        headers = extract_headers_from_markdown(plan_content)
        header_texts = [h[1] for h in headers]
        
        # Check for expected headers
        for expected in EXPECTED_PLAN_HEADERS:
            found = any(expected.lower() in h.lower() for h in header_texts)
            if not found:
                findings.append(DocumentationFinding(
                    severity=DriftSeverity.WARNING,
                    category="missing_header",
                    file=str(plan_path),
                    line=None,
                    message=f"Expected header '{expected}' not found in document",
                ))
        
        # Check for slice section consistency
        for slice_name in DOC_SLICE_METRIC_MAP.keys():
            if slice_name not in plan_content:
                findings.append(DocumentationFinding(
                    severity=DriftSeverity.FAIL,
                    category="missing_slice_section",
                    file=str(plan_path),
                    line=None,
                    message=f"Documented slice '{slice_name}' not mentioned in plan",
                    details={"slice": slice_name},
                ))
    else:
        findings.append(DocumentationFinding(
            severity=DriftSeverity.FAIL,
            category="file_not_found",
            file=str(plan_path),
            line=None,
            message="PHASE2_RFL_UPLIFT_PLAN.md not found",
        ))
    
    # Check RFL_UPLIFT_THEORY.md
    if theory_path.exists():
        theory_content, _ = load_markdown_file(theory_path)
        headers = extract_headers_from_markdown(theory_content)
        header_texts = [h[1] for h in headers]
        
        for expected in EXPECTED_THEORY_HEADERS:
            found = any(expected.lower() in h.lower() for h in header_texts)
            if not found:
                findings.append(DocumentationFinding(
                    severity=DriftSeverity.WARNING,
                    category="missing_header",
                    file=str(theory_path),
                    line=None,
                    message=f"Expected header '{expected}' not found in theory document",
                ))
    else:
        findings.append(DocumentationFinding(
            severity=DriftSeverity.WARNING,
            category="file_not_found", 
            file=str(theory_path),
            line=None,
            message="RFL_UPLIFT_THEORY.md not found (optional)",
        ))
    
    return findings


# =============================================================================
# CROSS-FILE DRIFT DETECTION
# =============================================================================

def detect_metric_kind_drift(
    plan_path: Path,
    curriculum_path: Path,
    prereg_path: Path,
) -> List[DriftFinding]:
    """
    Detect drift in metric kind definitions across files.
    
    Compares:
    - Documentation (PHASE2_RFL_UPLIFT_PLAN.md)
    - YAML config (curriculum_uplift_phase2.yaml)
    - Prereg templates (PREREG_UPLIFT_U2.yaml)
    """
    findings = []
    
    # Load all sources
    plan_content = ""
    if plan_path.exists():
        plan_content, _ = load_markdown_file(plan_path)
    
    curriculum = {}
    if curriculum_path.exists():
        curriculum = load_curriculum_yaml(curriculum_path)
    
    prereg = {}
    if prereg_path.exists():
        prereg = load_prereg_yaml(prereg_path)
    
    # For each documented slice, check consistency
    for slice_name, expected_metric in DOC_SLICE_METRIC_MAP.items():
        # Get metric from YAML
        yaml_metric = None
        slice_config = curriculum.get("slices", {}).get(slice_name, {})
        if isinstance(slice_config, dict):
            success_metric = slice_config.get("success_metric", {})
            if isinstance(success_metric, dict):
                yaml_metric = success_metric.get("kind")
        
        # Get metric from prereg
        prereg_metric = None
        prereg_slice = prereg.get(slice_name, {})
        if isinstance(prereg_slice, dict):
            prereg_sm = prereg_slice.get("success_metric", {})
            if isinstance(prereg_sm, dict):
                prereg_metric = prereg_sm.get("kind")
        
        # Check doc vs YAML
        if yaml_metric and yaml_metric != expected_metric.value:
            findings.append(DriftFinding(
                severity=DriftSeverity.FAIL,
                category="metric_kind_drift",
                source_file="docs/PHASE2_RFL_UPLIFT_PLAN.md",
                target_file="config/curriculum_uplift_phase2.yaml",
                message=f"Slice '{slice_name}': doc says '{expected_metric.value}', YAML has '{yaml_metric}'",
                details={"slice": slice_name, "doc": expected_metric.value, "yaml": yaml_metric},
            ))
        
        # Check doc vs prereg
        if prereg_metric and prereg_metric != expected_metric.value:
            findings.append(DriftFinding(
                severity=DriftSeverity.FAIL,
                category="metric_kind_drift",
                source_file="docs/PHASE2_RFL_UPLIFT_PLAN.md",
                target_file="experiments/prereg/PREREG_UPLIFT_U2.yaml",
                message=f"Slice '{slice_name}': doc says '{expected_metric.value}', prereg has '{prereg_metric}'",
                details={"slice": slice_name, "doc": expected_metric.value, "prereg": prereg_metric},
            ))
        
        # Check YAML vs prereg
        if yaml_metric and prereg_metric and yaml_metric != prereg_metric:
            findings.append(DriftFinding(
                severity=DriftSeverity.FAIL,
                category="metric_kind_drift",
                source_file="config/curriculum_uplift_phase2.yaml",
                target_file="experiments/prereg/PREREG_UPLIFT_U2.yaml", 
                message=f"Slice '{slice_name}': YAML has '{yaml_metric}', prereg has '{prereg_metric}'",
                details={"slice": slice_name, "yaml": yaml_metric, "prereg": prereg_metric},
            ))
    
    return findings


def detect_parameter_drift(
    curriculum_path: Path,
    prereg_path: Path,
) -> List[DriftFinding]:
    """
    Detect drift in metric parameters between YAML and prereg.
    """
    findings = []
    
    curriculum = {}
    if curriculum_path.exists():
        curriculum = load_curriculum_yaml(curriculum_path)
    
    prereg = {}
    if prereg_path.exists():
        prereg = load_prereg_yaml(prereg_path)
    
    for slice_name in DOC_SLICE_METRIC_MAP.keys():
        # Get params from YAML
        yaml_params = set()
        slice_config = curriculum.get("slices", {}).get(slice_name, {})
        if isinstance(slice_config, dict):
            success_metric = slice_config.get("success_metric", {})
            if isinstance(success_metric, dict):
                params = success_metric.get("parameters", {})
                if isinstance(params, dict):
                    yaml_params = set(params.keys())
        
        # Get params from prereg
        prereg_params = set()
        prereg_slice = prereg.get(slice_name, {})
        if isinstance(prereg_slice, dict):
            prereg_sm = prereg_slice.get("success_metric", {})
            if isinstance(prereg_sm, dict):
                params = prereg_sm.get("parameters", {})
                if isinstance(params, dict):
                    prereg_params = set(params.keys())
        
        # Check for parameter drift
        if yaml_params and prereg_params:
            yaml_only = yaml_params - prereg_params
            prereg_only = prereg_params - yaml_params
            
            if yaml_only:
                findings.append(DriftFinding(
                    severity=DriftSeverity.WARNING,
                    category="param_drift",
                    source_file="config/curriculum_uplift_phase2.yaml",
                    target_file="experiments/prereg/PREREG_UPLIFT_U2.yaml",
                    message=f"Slice '{slice_name}': params in YAML but not prereg: {yaml_only}",
                    details={"slice": slice_name, "yaml_only": list(yaml_only)},
                ))
            
            if prereg_only:
                findings.append(DriftFinding(
                    severity=DriftSeverity.WARNING,
                    category="param_drift",
                    source_file="experiments/prereg/PREREG_UPLIFT_U2.yaml",
                    target_file="config/curriculum_uplift_phase2.yaml",
                    message=f"Slice '{slice_name}': params in prereg but not YAML: {prereg_only}",
                    details={"slice": slice_name, "prereg_only": list(prereg_only)},
                ))
    
    return findings


def detect_function_drift() -> List[DriftFinding]:
    """
    Detect drift between documented metric kinds and implemented functions.
    """
    findings = []
    
    functions = extract_metric_functions()
    if "_import_error" in functions:
        findings.append(DriftFinding(
            severity=DriftSeverity.FAIL,
            category="import_error",
            source_file="experiments/slice_success_metrics.py",
            target_file="N/A",
            message=f"Cannot import metric functions: {functions['_import_error']}",
        ))
        return findings
    
    # Check each expected function exists
    for metric_kind, func_name in METRIC_FUNCTION_MAP.items():
        if func_name not in functions:
            findings.append(DriftFinding(
                severity=DriftSeverity.FAIL,
                category="missing_function",
                source_file="experiments/semantic_consistency_audit.py",
                target_file="experiments/slice_success_metrics.py",
                message=f"Expected function '{func_name}' for metric '{metric_kind.value}' not found",
                details={"metric_kind": metric_kind.value, "expected_function": func_name},
            ))
    
    # Check for extra functions not in our map
    expected_funcs = set(METRIC_FUNCTION_MAP.values())
    actual_funcs = {k for k in functions.keys() if not k.startswith("_")}
    extra_funcs = actual_funcs - expected_funcs
    
    if extra_funcs:
        findings.append(DriftFinding(
            severity=DriftSeverity.WARNING,
            category="undocumented_function",
            source_file="experiments/slice_success_metrics.py",
            target_file="experiments/semantic_consistency_audit.py",
            message=f"Undocumented metric functions found: {extra_funcs}",
            details={"extra_functions": list(extra_funcs)},
        ))
    
    return findings


# =============================================================================
# FORMULA REFERENCE VALIDATION
# =============================================================================

def validate_formula_references(
    plan_path: Path,
    curriculum_path: Path,
) -> List[DriftFinding]:
    """
    Validate that formulas referenced in documentation exist in YAML slice pools.
    """
    findings = []
    
    # Load documentation formulas
    plan_content = ""
    if plan_path.exists():
        plan_content, _ = load_markdown_file(plan_path)
    
    # Load YAML formula pools
    curriculum = {}
    if curriculum_path.exists():
        curriculum = load_curriculum_yaml(curriculum_path)
    
    # Extract all hashes from YAML
    yaml_hashes = set()
    yaml_formulas = set()
    
    for slice_name, slice_config in curriculum.get("slices", {}).items():
        if isinstance(slice_config, dict):
            # Get hashes from formula_pool_entries
            pool_entries = slice_config.get("formula_pool_entries", [])
            for entry in pool_entries:
                if isinstance(entry, dict):
                    if "hash" in entry:
                        yaml_hashes.add(entry["hash"])
                    if "formula" in entry:
                        yaml_formulas.add(entry["formula"])
                    if "normalized" in entry:
                        yaml_formulas.add(entry["normalized"])
            
            # Get hashes from success_metric parameters
            success_metric = slice_config.get("success_metric", {})
            if isinstance(success_metric, dict):
                params = success_metric.get("parameters", {})
                if isinstance(params, dict):
                    # target_hashes, required_goal_hashes, etc.
                    for key, value in params.items():
                        if "hash" in key.lower():
                            if isinstance(value, list):
                                yaml_hashes.update(value)
                            elif isinstance(value, str):
                                yaml_hashes.add(value)
            
            # Get hashes from dependency_graph
            dep_graph = slice_config.get("dependency_graph", {})
            if isinstance(dep_graph, dict):
                yaml_hashes.update(dep_graph.keys())
                for deps in dep_graph.values():
                    if isinstance(deps, list):
                        yaml_hashes.update(deps)
    
    # Extract hashes from documentation
    doc_hashes = extract_hash_references(plan_content)
    
    # Check for hashes in docs not in YAML (might be examples)
    doc_only_hashes = doc_hashes - yaml_hashes
    if doc_only_hashes:
        # This is usually fine - docs might have example hashes
        findings.append(DriftFinding(
            severity=DriftSeverity.WARNING,
            category="doc_hash_not_in_yaml",
            source_file=str(plan_path),
            target_file="config/curriculum_uplift_phase2.yaml",
            message=f"Found {len(doc_only_hashes)} hash(es) in docs not in YAML (may be examples)",
            details={"hashes": list(doc_only_hashes)[:5]},  # Show first 5
        ))
    
    return findings


def validate_slice_pool_completeness(
    curriculum_path: Path,
) -> List[DriftFinding]:
    """
    Validate that slices have formula pools when expected.
    """
    findings = []
    
    curriculum = {}
    if curriculum_path.exists():
        curriculum = load_curriculum_yaml(curriculum_path)
    
    for slice_name in DOC_SLICE_METRIC_MAP.keys():
        slice_config = curriculum.get("slices", {}).get(slice_name, {})
        if not isinstance(slice_config, dict):
            continue
        
        # Check for formula_pool_entries
        pool_entries = slice_config.get("formula_pool_entries", [])
        if not pool_entries:
            findings.append(DriftFinding(
                severity=DriftSeverity.WARNING,
                category="empty_formula_pool",
                source_file="config/curriculum_uplift_phase2.yaml",
                target_file="N/A",
                message=f"Slice '{slice_name}' has no formula_pool_entries",
                details={"slice": slice_name},
            ))
            continue
        
        # Check for targets
        targets = [e for e in pool_entries if isinstance(e, dict) and e.get("role") == "target"]
        if not targets:
            findings.append(DriftFinding(
                severity=DriftSeverity.WARNING,
                category="no_target_formulas",
                source_file="config/curriculum_uplift_phase2.yaml",
                target_file="N/A",
                message=f"Slice '{slice_name}' has no formulas with role='target'",
                details={"slice": slice_name},
            ))
        
        # Check for hash consistency
        for entry in pool_entries:
            if isinstance(entry, dict) and "formula" in entry and "hash" not in entry:
                findings.append(DriftFinding(
                    severity=DriftSeverity.WARNING,
                    category="missing_hash",
                    source_file="config/curriculum_uplift_phase2.yaml",
                    target_file="N/A",
                    message=f"Formula '{entry.get('name', 'unnamed')}' in '{slice_name}' has no hash",
                    details={"slice": slice_name, "formula": entry.get("formula")},
                ))
    
    return findings


# =============================================================================
# REPORTING
# =============================================================================

def format_drift_markdown_report(report: DriftReport) -> str:
    """Generate a human-readable Markdown drift report."""
    lines = [
        "# Cross-File Drift Detection Report",
        "",
        "**PHASE II — NOT USED IN PHASE I**",
        "",
        f"**Overall Status: {report.status.value}**",
        "",
    ]
    
    # Summary counts
    total_drift = len(report.drift_findings)
    total_doc = len(report.doc_findings)
    total_formula = len(report.formula_findings)
    
    lines.extend([
        "## Summary",
        "",
        f"- Metric/Parameter Drift Findings: {total_drift}",
        f"- Documentation Linter Findings: {total_doc}",
        f"- Formula Reference Findings: {total_formula}",
        "",
    ])
    
    # Drift findings
    if report.drift_findings:
        lines.append("## Metric & Parameter Drift")
        lines.append("")
        for f in report.drift_findings:
            icon = {"OK": "✅", "WARNING": "⚠️", "FAIL": "❌"}.get(f.severity.value, "❓")
            lines.append(f"- {icon} **[{f.severity.value}]** {f.category}")
            lines.append(f"  - Source: `{f.source_file}`")
            lines.append(f"  - Target: `{f.target_file}`")
            lines.append(f"  - {f.message}")
        lines.append("")
    
    # Documentation findings
    if report.doc_findings:
        lines.append("## Documentation Linter")
        lines.append("")
        for f in report.doc_findings:
            icon = {"OK": "✅", "WARNING": "⚠️", "FAIL": "❌"}.get(f.severity.value, "❓")
            lines.append(f"- {icon} **[{f.severity.value}]** {f.category} in `{f.file}`")
            if f.line:
                lines.append(f"  - Line: {f.line}")
            lines.append(f"  - {f.message}")
        lines.append("")
    
    # Formula findings
    if report.formula_findings:
        lines.append("## Formula Reference Validation")
        lines.append("")
        for f in report.formula_findings:
            icon = {"OK": "✅", "WARNING": "⚠️", "FAIL": "❌"}.get(f.severity.value, "❓")
            lines.append(f"- {icon} **[{f.severity.value}]** {f.category}")
            lines.append(f"  - {f.message}")
        lines.append("")
    
    # Include base audit if available
    if report.base_audit:
        lines.append("## Base Semantic Audit")
        lines.append("")
        lines.append(f"- Status: {report.base_audit.status.value}")
        lines.append(f"- Slices Audited: {len(report.base_audit.slices)}")
        lines.append(f"- Runtime Tests: {report.base_audit.total_tests_passed} passed, "
                    f"{report.base_audit.total_tests_failed} failed")
        lines.append("")
    
    # Final verdict
    lines.append("## Verdict")
    lines.append("")
    if report.status == DriftSeverity.OK:
        lines.append("✅ **All consistency checks passed.** No drift detected.")
    elif report.status == DriftSeverity.WARNING:
        lines.append("⚠️ **Warnings detected.** Non-breaking inconsistencies found.")
    else:
        lines.append("❌ **FAIL.** Breaking inconsistencies detected between configuration and documentation.")
    
    return "\n".join(lines)


def format_drift_json_report(report: DriftReport) -> Dict[str, Any]:
    """Generate a JSON-serializable drift report."""
    return {
        "label": "PHASE II — NOT USED IN PHASE I",
        "overall_status": report.status.value,
        "summary": {
            "drift_findings": len(report.drift_findings),
            "doc_findings": len(report.doc_findings),
            "formula_findings": len(report.formula_findings),
        },
        "drift_findings": [
            {
                "severity": f.severity.value,
                "category": f.category,
                "source_file": f.source_file,
                "target_file": f.target_file,
                "message": f.message,
                "details": f.details,
            }
            for f in report.drift_findings
        ],
        "doc_findings": [
            {
                "severity": f.severity.value,
                "category": f.category,
                "file": f.file,
                "line": f.line,
                "message": f.message,
                "details": f.details,
            }
            for f in report.doc_findings
        ],
        "formula_findings": [
            {
                "severity": f.severity.value,
                "category": f.category,
                "source_file": f.source_file,
                "target_file": f.target_file,
                "message": f.message,
                "details": f.details,
            }
            for f in report.formula_findings
        ],
        "base_audit": {
            "status": report.base_audit.status.value if report.base_audit else None,
            "slices": len(report.base_audit.slices) if report.base_audit else 0,
            "tests_passed": report.base_audit.total_tests_passed if report.base_audit else 0,
            "tests_failed": report.base_audit.total_tests_failed if report.base_audit else 0,
        } if report.base_audit else None,
    }


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run_drift_detection(
    curriculum_path: Optional[Path] = None,
    prereg_path: Optional[Path] = None,
    plan_path: Optional[Path] = None,
    theory_path: Optional[Path] = None,
    include_base_audit: bool = True,
) -> DriftReport:
    """
    Run the full drift detection and documentation lint.
    
    Args:
        curriculum_path: Path to curriculum_uplift_phase2.yaml
        prereg_path: Path to PREREG_UPLIFT_U2.yaml
        plan_path: Path to PHASE2_RFL_UPLIFT_PLAN.md
        theory_path: Path to RFL_UPLIFT_THEORY.md
        include_base_audit: Whether to include the base semantic audit
    
    Returns:
        DriftReport with all findings
    """
    # Default paths
    project_root = Path(__file__).parent.parent
    curriculum_path = curriculum_path or project_root / "config" / "curriculum_uplift_phase2.yaml"
    prereg_path = prereg_path or project_root / "experiments" / "prereg" / "PREREG_UPLIFT_U2.yaml"
    plan_path = plan_path or project_root / "docs" / "PHASE2_RFL_UPLIFT_PLAN.md"
    theory_path = theory_path or project_root / "RFL_UPLIFT_THEORY.md"
    
    report = DriftReport()
    
    # 1. Run base semantic audit
    if include_base_audit:
        from experiments.semantic_consistency_audit import run_audit
        report.base_audit = run_audit(curriculum_path, prereg_path)
    
    # 2. Run documentation linter
    report.doc_findings.extend(lint_documentation(plan_path, theory_path))
    
    # 3. Detect metric kind drift
    report.drift_findings.extend(detect_metric_kind_drift(plan_path, curriculum_path, prereg_path))
    
    # 4. Detect parameter drift
    report.drift_findings.extend(detect_parameter_drift(curriculum_path, prereg_path))
    
    # 5. Detect function drift
    report.drift_findings.extend(detect_function_drift())
    
    # 6. Validate formula references
    report.formula_findings.extend(validate_formula_references(plan_path, curriculum_path))
    report.formula_findings.extend(validate_slice_pool_completeness(curriculum_path))
    
    return report


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="PHASE II Cross-File Drift Detection and Documentation Linter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Absolute Safeguards:
- Semantic audits must never alter any metrics.
- No modification to snippet semantics or governance.
- This is a DETECTION-ONLY auditor.

Files Compared:
- docs/PHASE2_RFL_UPLIFT_PLAN.md
- RFL_UPLIFT_THEORY.md
- config/curriculum_uplift_phase2.yaml
- experiments/slice_success_metrics.py
- experiments/prereg/PREREG_UPLIFT_U2.yaml
        """,
    )
    parser.add_argument(
        "--curriculum",
        type=Path,
        default=None,
        help="Path to curriculum_uplift_phase2.yaml",
    )
    parser.add_argument(
        "--prereg",
        type=Path,
        default=None,
        help="Path to PREREG_UPLIFT_U2.yaml",
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=None,
        help="Path to PHASE2_RFL_UPLIFT_PLAN.md",
    )
    parser.add_argument(
        "--theory",
        type=Path,
        default=None,
        help="Path to RFL_UPLIFT_THEORY.md",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON report instead of Markdown",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write report to file instead of stdout",
    )
    parser.add_argument(
        "--skip-base-audit",
        action="store_true",
        help="Skip the base semantic audit",
    )
    
    args = parser.parse_args()
    
    # Run drift detection
    report = run_drift_detection(
        curriculum_path=args.curriculum,
        prereg_path=args.prereg,
        plan_path=args.plan,
        theory_path=args.theory,
        include_base_audit=not args.skip_base_audit,
    )
    
    # Format output
    if args.json:
        output = json.dumps(format_drift_json_report(report), indent=2)
    else:
        output = format_drift_markdown_report(report)
    
    # Write output
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"Report written to {args.output}")
    else:
        print(output)
    
    # Exit with appropriate code
    if report.status == DriftSeverity.FAIL:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()

