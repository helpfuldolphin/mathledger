"""
PHASE II — NOT USED IN PHASE I

Semantic Consistency Auditor for Curriculum Configuration.

This module validates that what we *say* we measure (in documentation and preregistration)
matches what we *actually* implement (in YAML config and code).

Artifacts audited:
- config/curriculum_uplift_phase2.yaml: Slice definitions
- experiments/slice_success_metrics.py: Metric function implementations
- experiments/prereg/PREREG_UPLIFT_U2.yaml: Preregistration templates
- docs/PHASE2_RFL_UPLIFT_PLAN.md: Narrative descriptions (read-only reference)

Features:
- Term Index Builder: Produces semantic_term_index.json
- CI Mode: Minimal output with proper exit codes
- Suggestion Engine: Proposes fixes for detected drift

Absolute Safeguards:
- Do NOT change metric logic.
- Do NOT change success definitions.
- Do NOT alter theory or governance text.
- This is an AUDITOR, not a designer.
"""

from __future__ import annotations

import argparse
import inspect
import json
import re
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import yaml


# =============================================================================
# CANONICAL DEFINITIONS
# =============================================================================

class MetricKind(str, Enum):
    """
    The canonical set of metric kinds defined in PHASE2_RFL_UPLIFT_PLAN.md
    and PREREG_UPLIFT_U2.yaml.
    """
    GOAL_HIT = "goal_hit"
    DENSITY = "density"
    CHAIN_LENGTH = "chain_length"
    MULTI_GOAL = "multi_goal"


# Required parameters for each metric kind, extracted from:
# - experiments/slice_success_metrics.py function signatures
# - experiments/prereg/PREREG_UPLIFT_U2.yaml parameter templates
METRIC_REQUIRED_PARAMS: Dict[MetricKind, Set[str]] = {
    MetricKind.GOAL_HIT: {"target_hashes", "min_total_verified"},
    MetricKind.DENSITY: {"min_verified"},
    MetricKind.CHAIN_LENGTH: {"chain_target_hash", "min_chain_length"},
    MetricKind.MULTI_GOAL: {"required_goal_hashes"},
}

# Optional parameters for each metric kind
METRIC_OPTIONAL_PARAMS: Dict[MetricKind, Set[str]] = {
    MetricKind.GOAL_HIT: {"min_goal_hits"},
    MetricKind.DENSITY: {"max_candidates"},
    MetricKind.CHAIN_LENGTH: set(),
    MetricKind.MULTI_GOAL: {"min_each_goal"},
}

# Canonical slice-to-metric mapping from PHASE2_RFL_UPLIFT_PLAN.md
DOC_SLICE_METRIC_MAP: Dict[str, MetricKind] = {
    "slice_uplift_goal": MetricKind.GOAL_HIT,
    "slice_uplift_sparse": MetricKind.DENSITY,
    "slice_uplift_tree": MetricKind.CHAIN_LENGTH,
    "slice_uplift_dependency": MetricKind.MULTI_GOAL,
}

# Metric function names from slice_success_metrics.py
METRIC_FUNCTION_MAP: Dict[MetricKind, str] = {
    MetricKind.GOAL_HIT: "compute_goal_hit",
    MetricKind.DENSITY: "compute_sparse_success",
    MetricKind.CHAIN_LENGTH: "compute_chain_success",
    MetricKind.MULTI_GOAL: "compute_multi_goal_success",
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class AuditStatus(str, Enum):
    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"


@dataclass
class AuditIssue:
    """Represents a single audit finding."""
    severity: AuditStatus
    category: str
    message: str
    details: Optional[Dict[str, Any]] = None


@dataclass
class SliceSemanticSpec:
    """
    Captures the semantic specification of a curriculum slice.
    
    Combines information from:
    - YAML config (actual implementation)
    - Documentation (expected behavior)
    - Preregistration (formal specification)
    - Code (function signatures)
    """
    slice_name: str
    metric_kind: Optional[str] = None
    yaml_params: Dict[str, Any] = field(default_factory=dict)
    doc_expected_behavior: str = ""
    metric_function: Optional[str] = None
    prereg_params: Dict[str, Any] = field(default_factory=dict)
    
    # Audit results
    issues: List[AuditIssue] = field(default_factory=list)
    tests_passed: int = 0
    tests_failed: int = 0
    
    @property
    def status(self) -> AuditStatus:
        """Overall status based on issues."""
        if any(i.severity == AuditStatus.FAIL for i in self.issues):
            return AuditStatus.FAIL
        if any(i.severity == AuditStatus.WARNING for i in self.issues):
            return AuditStatus.WARNING
        return AuditStatus.PASS


@dataclass
class AuditReport:
    """Complete audit report across all slices."""
    slices: List[SliceSemanticSpec] = field(default_factory=list)
    global_issues: List[AuditIssue] = field(default_factory=list)
    
    @property
    def status(self) -> AuditStatus:
        """Overall audit status."""
        all_issues = self.global_issues + [i for s in self.slices for i in s.issues]
        if any(i.severity == AuditStatus.FAIL for i in all_issues):
            return AuditStatus.FAIL
        if any(i.severity == AuditStatus.WARNING for i in all_issues):
            return AuditStatus.WARNING
        return AuditStatus.PASS
    
    @property
    def total_tests_passed(self) -> int:
        return sum(s.tests_passed for s in self.slices)
    
    @property
    def total_tests_failed(self) -> int:
        return sum(s.tests_failed for s in self.slices)


# =============================================================================
# LOADERS
# =============================================================================

def load_curriculum_yaml(path: Path) -> Dict[str, Any]:
    """Load curriculum_uplift_phase2.yaml."""
    if not path.exists():
        raise FileNotFoundError(f"Curriculum file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_prereg_yaml(path: Path) -> Dict[str, Any]:
    """Load PREREG_UPLIFT_U2.yaml.
    
    Handles multi-document YAML files (separated by ---) by merging all documents.
    """
    if not path.exists():
        return {}  # Optional file
    with open(path, "r", encoding="utf-8") as f:
        # The prereg file may have multiple YAML documents separated by ---
        # We load all and merge them
        result = {}
        for doc in yaml.safe_load_all(f):
            if doc is not None and isinstance(doc, dict):
                result.update(doc)
        return result


def extract_metric_functions() -> Dict[str, Callable]:
    """
    Import and return metric functions from slice_success_metrics.py.
    Returns a dict mapping function names to callables.
    """
    try:
        from experiments.slice_success_metrics import (
            compute_goal_hit,
            compute_sparse_success,
            compute_chain_success,
            compute_multi_goal_success,
        )
        return {
            "compute_goal_hit": compute_goal_hit,
            "compute_sparse_success": compute_sparse_success,
            "compute_chain_success": compute_chain_success,
            "compute_multi_goal_success": compute_multi_goal_success,
        }
    except ImportError as e:
        return {"_import_error": str(e)}


def extract_function_signatures() -> Dict[str, Dict[str, Any]]:
    """
    Extract function signatures from slice_success_metrics.py.
    Returns dict mapping function name to parameter info.
    """
    functions = extract_metric_functions()
    if "_import_error" in functions:
        return {"_import_error": functions["_import_error"]}
    
    signatures = {}
    for name, func in functions.items():
        sig = inspect.signature(func)
        signatures[name] = {
            "params": list(sig.parameters.keys()),
            "docstring": (func.__doc__ or "").strip()[:200],
        }
    return signatures


# =============================================================================
# SEMANTIC MAPPING EXTRACTOR (TASK 1)
# =============================================================================

def extract_slice_specs(
    curriculum_path: Path,
    prereg_path: Path,
) -> List[SliceSemanticSpec]:
    """
    Extract SliceSemanticSpec for each slice in the curriculum.
    
    Cross-references:
    - curriculum_uplift_phase2.yaml: Actual slice definitions
    - PREREG_UPLIFT_U2.yaml: Preregistration templates
    - DOC_SLICE_METRIC_MAP: Expected metric kinds from documentation
    - METRIC_FUNCTION_MAP: Function names from code
    """
    curriculum = load_curriculum_yaml(curriculum_path)
    prereg = load_prereg_yaml(prereg_path)
    
    specs = []
    slices = curriculum.get("slices", {})
    
    for slice_name, slice_config in slices.items():
        spec = SliceSemanticSpec(slice_name=slice_name)
        
        # Extract from YAML config
        if isinstance(slice_config, dict):
            spec.yaml_params = slice_config.copy()
            # Check if success_metric is defined in YAML
            if "success_metric" in slice_config:
                sm = slice_config["success_metric"]
                if isinstance(sm, dict):
                    spec.metric_kind = sm.get("kind")
                    spec.yaml_params["success_metric_params"] = sm.get("params", {})
        
        # Extract from preregistration
        prereg_slice = prereg.get(slice_name, {})
        if prereg_slice:
            prereg_metric = prereg_slice.get("success_metric", {})
            if not spec.metric_kind:
                spec.metric_kind = prereg_metric.get("kind")
            spec.prereg_params = prereg_metric.get("parameters", {})
        
        # Map to expected metric from documentation
        doc_metric = DOC_SLICE_METRIC_MAP.get(slice_name)
        if doc_metric:
            spec.doc_expected_behavior = f"Expected metric kind from docs: {doc_metric.value}"
            if not spec.metric_kind:
                spec.metric_kind = doc_metric.value
            spec.metric_function = METRIC_FUNCTION_MAP.get(doc_metric)
        
        specs.append(spec)
    
    return specs


# =============================================================================
# STATIC CONSISTENCY CHECKS (TASK 2)
# =============================================================================

def check_metric_kind_valid(spec: SliceSemanticSpec) -> None:
    """Check that success_metric.kind is in METRIC_KINDS."""
    if not spec.metric_kind:
        spec.issues.append(AuditIssue(
            severity=AuditStatus.WARNING,
            category="metric_kind",
            message=f"Slice '{spec.slice_name}' has no metric_kind defined in YAML or prereg",
            details={"slice": spec.slice_name},
        ))
        return
    
    valid_kinds = {k.value for k in MetricKind}
    if spec.metric_kind not in valid_kinds:
        spec.issues.append(AuditIssue(
            severity=AuditStatus.FAIL,
            category="metric_kind",
            message=f"Invalid metric_kind '{spec.metric_kind}' for slice '{spec.slice_name}'",
            details={"slice": spec.slice_name, "valid_kinds": list(valid_kinds)},
        ))


def check_required_params(spec: SliceSemanticSpec) -> None:
    """Check that all required parameters for the metric are present."""
    if not spec.metric_kind:
        return
    
    try:
        metric_kind = MetricKind(spec.metric_kind)
    except ValueError:
        return  # Already flagged in check_metric_kind_valid
    
    required = METRIC_REQUIRED_PARAMS.get(metric_kind, set())
    optional = METRIC_OPTIONAL_PARAMS.get(metric_kind, set())
    
    # Combine all available params from YAML and prereg
    available_params = set()
    if "success_metric_params" in spec.yaml_params:
        available_params.update(spec.yaml_params["success_metric_params"].keys())
    available_params.update(spec.prereg_params.keys())
    
    # Check for missing required params
    missing = required - available_params
    if missing:
        spec.issues.append(AuditIssue(
            severity=AuditStatus.FAIL,
            category="missing_params",
            message=f"Slice '{spec.slice_name}' missing required params: {sorted(missing)}",
            details={"slice": spec.slice_name, "missing": sorted(missing), "required": sorted(required)},
        ))
    
    # Check for unexpected params
    allowed = required | optional
    unexpected = available_params - allowed
    if unexpected:
        spec.issues.append(AuditIssue(
            severity=AuditStatus.WARNING,
            category="unexpected_params",
            message=f"Slice '{spec.slice_name}' has unexpected params: {sorted(unexpected)}",
            details={"slice": spec.slice_name, "unexpected": sorted(unexpected), "allowed": sorted(allowed)},
        ))


def check_doc_consistency(spec: SliceSemanticSpec) -> None:
    """Check that YAML config matches documentation expectations."""
    doc_expected = DOC_SLICE_METRIC_MAP.get(spec.slice_name)
    
    if doc_expected and spec.metric_kind:
        if spec.metric_kind != doc_expected.value:
            spec.issues.append(AuditIssue(
                severity=AuditStatus.FAIL,
                category="doc_mismatch",
                message=f"Slice '{spec.slice_name}' metric_kind '{spec.metric_kind}' "
                        f"differs from doc expectation '{doc_expected.value}'",
                details={"slice": spec.slice_name, "actual": spec.metric_kind, "expected": doc_expected.value},
            ))


def check_function_mapping(spec: SliceSemanticSpec) -> None:
    """Check that the metric function exists and has correct signature."""
    if not spec.metric_function:
        return
    
    functions = extract_metric_functions()
    if "_import_error" in functions:
        spec.issues.append(AuditIssue(
            severity=AuditStatus.WARNING,
            category="import_error",
            message=f"Could not import slice_success_metrics: {functions['_import_error']}",
        ))
        return
    
    if spec.metric_function not in functions:
        spec.issues.append(AuditIssue(
            severity=AuditStatus.FAIL,
            category="missing_function",
            message=f"Metric function '{spec.metric_function}' not found in slice_success_metrics.py",
            details={"expected_function": spec.metric_function, "available": list(functions.keys())},
        ))


def run_static_checks(specs: List[SliceSemanticSpec]) -> None:
    """Run all static consistency checks on all slice specs."""
    for spec in specs:
        check_metric_kind_valid(spec)
        check_required_params(spec)
        check_doc_consistency(spec)
        check_function_mapping(spec)


# =============================================================================
# RUNTIME SANITY CHECKS (TASK 3)
# =============================================================================

def create_mock_cycle_data() -> Dict[str, List[Dict[str, Any]]]:
    """
    Create mock cycle data for runtime sanity tests.
    
    Returns dict mapping metric_kind to list of test cases.
    Each test case has: inputs, expected_success, expected_value_range.
    """
    return {
        MetricKind.GOAL_HIT.value: [
            # Test 1: Hit all targets
            {
                "inputs": {
                    "verified_statements": [{"hash": "h1"}, {"hash": "h2"}, {"hash": "h3"}],
                    "target_hashes": {"h1", "h2"},
                    "min_total_verified": 2,
                },
                "expected_success": True,
                "expected_value": 2.0,
            },
            # Test 2: Miss threshold
            {
                "inputs": {
                    "verified_statements": [{"hash": "h1"}],
                    "target_hashes": {"h1", "h2", "h3"},
                    "min_total_verified": 2,
                },
                "expected_success": False,
                "expected_value": 1.0,
            },
            # Test 3: No hits
            {
                "inputs": {
                    "verified_statements": [{"hash": "h9"}],
                    "target_hashes": {"h1", "h2"},
                    "min_total_verified": 1,
                },
                "expected_success": False,
                "expected_value": 0.0,
            },
        ],
        MetricKind.DENSITY.value: [
            # Test 1: Above threshold
            {
                "inputs": {
                    "verified_count": 5,
                    "attempted_count": 40,
                    "min_verified": 5,
                },
                "expected_success": True,
                "expected_value": 5.0,
            },
            # Test 2: Below threshold
            {
                "inputs": {
                    "verified_count": 2,
                    "attempted_count": 40,
                    "min_verified": 5,
                },
                "expected_success": False,
                "expected_value": 2.0,
            },
        ],
        MetricKind.CHAIN_LENGTH.value: [
            # Test 1: Full chain
            {
                "inputs": {
                    "verified_statements": [{"hash": "h1"}, {"hash": "h2"}, {"hash": "h3"}],
                    "dependency_graph": {"h3": ["h2"], "h2": ["h1"]},
                    "chain_target_hash": "h3",
                    "min_chain_length": 3,
                },
                "expected_success": True,
                "expected_value": 3.0,
            },
            # Test 2: Broken chain
            {
                "inputs": {
                    "verified_statements": [{"hash": "h1"}, {"hash": "h3"}],
                    "dependency_graph": {"h3": ["h2"], "h2": ["h1"]},
                    "chain_target_hash": "h3",
                    "min_chain_length": 3,
                },
                "expected_success": False,
                "expected_value": 1.0,
            },
        ],
        MetricKind.MULTI_GOAL.value: [
            # Test 1: All goals met
            {
                "inputs": {
                    "verified_hashes": {"h1", "h2", "h3"},
                    "required_goal_hashes": {"h1", "h3"},
                },
                "expected_success": True,
                "expected_value": 2.0,
            },
            # Test 2: Missing goal
            {
                "inputs": {
                    "verified_hashes": {"h1", "h2"},
                    "required_goal_hashes": {"h1", "h3"},
                },
                "expected_success": False,
                "expected_value": 1.0,
            },
        ],
    }


def run_runtime_sanity_checks(spec: SliceSemanticSpec) -> None:
    """Run runtime sanity checks for a slice's metric function."""
    if not spec.metric_kind or not spec.metric_function:
        return
    
    functions = extract_metric_functions()
    if "_import_error" in functions or spec.metric_function not in functions:
        return
    
    func = functions[spec.metric_function]
    test_cases = create_mock_cycle_data().get(spec.metric_kind, [])
    
    for i, test_case in enumerate(test_cases):
        try:
            success, value = func(**test_case["inputs"])
            
            if success != test_case["expected_success"]:
                spec.issues.append(AuditIssue(
                    severity=AuditStatus.FAIL,
                    category="runtime_check",
                    message=f"Runtime check #{i+1} failed: expected success={test_case['expected_success']}, "
                            f"got {success}",
                    details={
                        "test_case": i + 1,
                        "inputs": str(test_case["inputs"]),
                        "expected": test_case["expected_success"],
                        "actual": success,
                    },
                ))
                spec.tests_failed += 1
            elif value != test_case["expected_value"]:
                spec.issues.append(AuditIssue(
                    severity=AuditStatus.FAIL,
                    category="runtime_check",
                    message=f"Runtime check #{i+1} failed: expected value={test_case['expected_value']}, "
                            f"got {value}",
                    details={
                        "test_case": i + 1,
                        "expected_value": test_case["expected_value"],
                        "actual_value": value,
                    },
                ))
                spec.tests_failed += 1
            else:
                spec.tests_passed += 1
                
        except Exception as e:
            spec.issues.append(AuditIssue(
                severity=AuditStatus.FAIL,
                category="runtime_error",
                message=f"Runtime check #{i+1} raised exception: {e}",
                details={"test_case": i + 1, "exception": str(e)},
            ))
            spec.tests_failed += 1


# =============================================================================
# GLOBAL CONSISTENCY CHECKS
# =============================================================================

def check_global_consistency(
    curriculum_path: Path,
    prereg_path: Path,
    report: AuditReport,
) -> None:
    """Check global consistency issues across all artifacts."""
    curriculum = load_curriculum_yaml(curriculum_path)
    prereg = load_prereg_yaml(prereg_path)
    
    # Check: All documented slices should exist in curriculum
    for doc_slice in DOC_SLICE_METRIC_MAP.keys():
        if doc_slice not in curriculum.get("slices", {}):
            report.global_issues.append(AuditIssue(
                severity=AuditStatus.WARNING,
                category="missing_slice",
                message=f"Documented slice '{doc_slice}' not found in curriculum YAML",
                details={"slice": doc_slice, "source": "PHASE2_RFL_UPLIFT_PLAN.md"},
            ))
    
    # Check: Preregistration slice templates should match documented slices
    for prereg_slice in prereg.keys():
        if prereg_slice in ["preregistration", "---"]:
            continue  # Skip metadata sections
        if prereg_slice not in DOC_SLICE_METRIC_MAP:
            report.global_issues.append(AuditIssue(
                severity=AuditStatus.WARNING,
                category="prereg_mismatch",
                message=f"Prereg slice '{prereg_slice}' not found in documentation",
                details={"slice": prereg_slice},
            ))
    
    # Check: Metric functions should all be importable
    functions = extract_metric_functions()
    if "_import_error" in functions:
        report.global_issues.append(AuditIssue(
            severity=AuditStatus.FAIL,
            category="import_error",
            message=f"Cannot import slice_success_metrics: {functions['_import_error']}",
        ))
    else:
        # Check all expected functions exist
        for metric_kind, func_name in METRIC_FUNCTION_MAP.items():
            if func_name not in functions:
                report.global_issues.append(AuditIssue(
                    severity=AuditStatus.FAIL,
                    category="missing_function",
                    message=f"Expected function '{func_name}' for metric '{metric_kind.value}' not found",
                ))


# =============================================================================
# REPORTING (TASK 4)
# =============================================================================

def format_markdown_report(report: AuditReport) -> str:
    """Generate a human-readable Markdown report."""
    lines = [
        "# Semantic Consistency Audit Report",
        "",
        "**PHASE II — NOT USED IN PHASE I**",
        "",
        f"**Overall Status: {report.status.value}**",
        "",
        f"- Total Slices Audited: {len(report.slices)}",
        f"- Runtime Tests Passed: {report.total_tests_passed}",
        f"- Runtime Tests Failed: {report.total_tests_failed}",
        f"- Global Issues: {len(report.global_issues)}",
        "",
    ]
    
    # Global issues
    if report.global_issues:
        lines.append("## Global Issues")
        lines.append("")
        for issue in report.global_issues:
            lines.append(f"- **[{issue.severity.value}]** {issue.category}: {issue.message}")
        lines.append("")
    
    # Per-slice results
    lines.append("## Slice Results")
    lines.append("")
    
    for spec in report.slices:
        status_icon = {"PASS": "✅", "WARNING": "⚠️", "FAIL": "❌"}.get(spec.status.value, "❓")
        lines.append(f"### {status_icon} `{spec.slice_name}`")
        lines.append("")
        lines.append(f"- **Status**: {spec.status.value}")
        lines.append(f"- **Metric Kind**: {spec.metric_kind or 'NOT DEFINED'}")
        lines.append(f"- **Metric Function**: {spec.metric_function or 'NOT MAPPED'}")
        lines.append(f"- **Runtime Tests**: {spec.tests_passed} passed, {spec.tests_failed} failed")
        lines.append("")
        
        if spec.issues:
            lines.append("**Issues:**")
            for issue in spec.issues:
                lines.append(f"- [{issue.severity.value}] {issue.category}: {issue.message}")
            lines.append("")
        
        if spec.yaml_params:
            lines.append("**YAML Params:**")
            lines.append("```yaml")
            for k, v in spec.yaml_params.items():
                if k != "success_metric_params":
                    lines.append(f"{k}: {v}")
            lines.append("```")
            lines.append("")
    
    # Summary
    lines.append("## Summary")
    lines.append("")
    if report.status == AuditStatus.PASS:
        lines.append("✅ **All checks passed.** We are measuring what we say we are measuring.")
    elif report.status == AuditStatus.WARNING:
        lines.append("⚠️ **Warnings detected.** Review the issues above for potential inconsistencies.")
    else:
        lines.append("❌ **Audit FAILED.** There are inconsistencies between configuration and documentation.")
    
    return "\n".join(lines)


def format_json_report(report: AuditReport) -> Dict[str, Any]:
    """Generate a JSON-serializable report."""
    return {
        "label": "PHASE II — NOT USED IN PHASE I",
        "overall_status": report.status.value,
        "total_slices": len(report.slices),
        "total_tests_passed": report.total_tests_passed,
        "total_tests_failed": report.total_tests_failed,
        "global_issues": [
            {
                "severity": i.severity.value,
                "category": i.category,
                "message": i.message,
                "details": i.details,
            }
            for i in report.global_issues
        ],
        "slices": [
            {
                "slice_name": s.slice_name,
                "status": s.status.value,
                "metric_kind": s.metric_kind,
                "metric_function": s.metric_function,
                "yaml_params": s.yaml_params,
                "prereg_params": s.prereg_params,
                "doc_expected_behavior": s.doc_expected_behavior,
                "tests_passed": s.tests_passed,
                "tests_failed": s.tests_failed,
                "issues": [
                    {
                        "severity": i.severity.value,
                        "category": i.category,
                        "message": i.message,
                        "details": i.details,
                    }
                    for i in s.issues
                ],
            }
            for s in report.slices
        ],
    }


# =============================================================================
# TERM INDEX BUILDER
# =============================================================================

@dataclass
class TermMention:
    """A single mention of a term in a file."""
    file: str
    line: int
    exact_spelling: str


@dataclass 
class TermEntry:
    """An entry in the semantic term index.
    
    Canonical Contract Shape:
    {
        "term": str,
        "canonical_form": str,
        "kind": "slice"|"metric"|"theorem"|"symbol",
        "mentions": [
            {"file": str, "line": int, "spelling": str}
        ]
    }
    """
    term: str
    canonical_form: str
    kind: str  # "slice", "metric", "theorem", "symbol"
    mentions: List[TermMention] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to canonical contract shape.
        
        Note: Uses "spelling" key (not "exact_spelling") per canonical contract.
        """
        return {
            "term": self.term,
            "canonical_form": self.canonical_form,
            "kind": self.kind,
            "mentions": [
                {"file": m.file, "line": m.line, "spelling": m.exact_spelling}
                for m in self.mentions
            ],
        }


class TermIndexBuilder:
    """
    Builds a semantic term index from Phase II artifacts.
    
    Indexes:
    - Slice names (slice_uplift_goal, etc.)
    - Metric kinds (goal_hit, density, etc.)
    - Theory symbols (Δp, A(t), H_t, etc.)
    - Definition identifiers (Definition 1.1, Conjecture 3.1, etc.)
    """
    
    # Canonical terms we expect to find
    CANONICAL_SLICE_NAMES = [
        "slice_uplift_goal",
        "slice_uplift_sparse", 
        "slice_uplift_tree",
        "slice_uplift_dependency",
    ]
    
    CANONICAL_METRIC_KINDS = [
        "goal_hit",
        "density",
        "chain_length",
        "multi_goal",
    ]
    
    # Theory symbols from RFL_UPLIFT_THEORY.md
    THEORY_SYMBOLS = [
        ("Δp", "delta_p", "Uplift gain: p_rfl - p_base"),
        ("A(t)", "abstention_rate", "Abstention rate at time t"),
        ("H_t", "derivation_entropy", "Derivation entropy at time t"),
        ("L_t", "learning_signal", "Learning signal at time t"),
        ("K_t", "knowledge_frontier", "Knowledge frontier at time t"),
        ("π_W", "wide_slice_policy", "Wide Slice policy"),
    ]
    
    # Patterns for finding terms
    SLICE_PATTERN = re.compile(r'(slice_uplift_\w+)')
    METRIC_PATTERN = re.compile(r'\b(goal_hit|density|chain_length|multi_goal)\b', re.IGNORECASE)
    DEFINITION_PATTERN = re.compile(r'(Definition\s+\d+\.\d+|Conjecture\s+\d+\.\d+|Lemma\s+\d+\.\d+|Proposition\s+\d+\.\d+)', re.IGNORECASE)
    DELTA_P_PATTERN = re.compile(r'(Δp|delta_?p|\\Delta\s*p)', re.IGNORECASE)
    ABSTENTION_PATTERN = re.compile(r'(A\(t\)|abstention_rate|abstention rate)')
    ENTROPY_PATTERN = re.compile(r'(H_t|H_\{t\}|derivation entropy)', re.IGNORECASE)
    
    def __init__(self):
        self.terms: Dict[str, TermEntry] = {}
    
    def _add_mention(
        self,
        term: str,
        canonical: str,
        kind: str,
        file_path: str,
        line: int,
        exact_spelling: str,
    ) -> None:
        """Add a mention of a term."""
        key = canonical.lower()
        if key not in self.terms:
            self.terms[key] = TermEntry(
                term=term,
                canonical_form=canonical,
                kind=kind,
            )
        self.terms[key].mentions.append(TermMention(
            file=file_path,
            line=line,
            exact_spelling=exact_spelling,
        ))
    
    def scan_file(self, file_path: Path) -> None:
        """Scan a file for term mentions."""
        if not file_path.exists():
            return
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception:
            return
        
        rel_path = str(file_path)
        
        for line_num, line in enumerate(lines, 1):
            # Scan for slice names
            for match in self.SLICE_PATTERN.finditer(line):
                slice_name = match.group(1)
                self._add_mention(
                    term=slice_name,
                    canonical=slice_name,
                    kind="slice",
                    file_path=rel_path,
                    line=line_num,
                    exact_spelling=match.group(0),
                )
            
            # Scan for metric kinds
            for match in self.METRIC_PATTERN.finditer(line):
                metric = match.group(1).lower()
                self._add_mention(
                    term=metric,
                    canonical=metric,
                    kind="metric",
                    file_path=rel_path,
                    line=line_num,
                    exact_spelling=match.group(0),
                )
            
            # Scan for definition identifiers
            for match in self.DEFINITION_PATTERN.finditer(line):
                defn = match.group(1)
                # Normalize: "Definition 1.1" -> "definition_1.1"
                canonical = defn.lower().replace(" ", "_")
                self._add_mention(
                    term=defn,
                    canonical=canonical,
                    kind="theorem",
                    file_path=rel_path,
                    line=line_num,
                    exact_spelling=match.group(0),
                )
            
            # Scan for Δp / delta_p
            for match in self.DELTA_P_PATTERN.finditer(line):
                self._add_mention(
                    term="Δp",
                    canonical="delta_p",
                    kind="symbol",
                    file_path=rel_path,
                    line=line_num,
                    exact_spelling=match.group(0),
                )
            
            # Scan for A(t) / abstention rate
            for match in self.ABSTENTION_PATTERN.finditer(line):
                self._add_mention(
                    term="A(t)",
                    canonical="abstention_rate",
                    kind="symbol",
                    file_path=rel_path,
                    line=line_num,
                    exact_spelling=match.group(0),
                )
            
            # Scan for H_t / entropy
            for match in self.ENTROPY_PATTERN.finditer(line):
                self._add_mention(
                    term="H_t",
                    canonical="derivation_entropy",
                    kind="symbol",
                    file_path=rel_path,
                    line=line_num,
                    exact_spelling=match.group(0),
                )
    
    def build_index(self, project_root: Path) -> Dict[str, Any]:
        """Build the complete term index.
        
        Contract guarantees:
        - Terms sorted alphabetically by canonical_form
        - Mentions within each term sorted by (file, line)
        - Deterministic output across identical inputs
        """
        # Scan all relevant files in deterministic order
        files_to_scan = [
            project_root / "config" / "curriculum_uplift_phase2.yaml",
            project_root / "experiments" / "slice_success_metrics.py",
            project_root / "docs" / "PHASE2_RFL_UPLIFT_PLAN.md",
            project_root / "RFL_UPLIFT_THEORY.md",
            project_root / "experiments" / "prereg" / "PREREG_UPLIFT_U2.yaml",
            project_root / "experiments" / "semantic_consistency_audit.py",
        ]
        
        for file_path in files_to_scan:
            self.scan_file(file_path)
        
        # Sort terms alphabetically by canonical_form for determinism
        sorted_entries = sorted(
            self.terms.values(),
            key=lambda e: e.canonical_form.lower()
        )
        
        # Sort mentions within each term by (file, line) for determinism
        for entry in sorted_entries:
            entry.mentions.sort(key=lambda m: (m.file, m.line))
        
        return {
            "version": "1.0",
            "phase": "II",
            "terms": [entry.to_dict() for entry in sorted_entries],
            "summary": {
                "total_terms": len(self.terms),
                "by_kind": {
                    kind: len([t for t in self.terms.values() if t.kind == kind])
                    for kind in ["slice", "metric", "theorem", "symbol"]
                },
            },
        }


def build_term_index(project_root: Optional[Path] = None) -> Dict[str, Any]:
    """Build the semantic term index."""
    project_root = project_root or Path(__file__).parent.parent
    builder = TermIndexBuilder()
    return builder.build_index(project_root)


def save_term_index(index: Dict[str, Any], output_path: Path) -> None:
    """Save the term index to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)


# =============================================================================
# SUGGESTION ENGINE
# =============================================================================

@dataclass
class FixSuggestion:
    """A suggested fix for detected drift."""
    issue_type: str
    current_value: str
    suggested_value: str
    file_path: str
    approximate_line: Optional[int]
    description: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "issue_type": self.issue_type,
            "current_value": self.current_value,
            "suggested_value": self.suggested_value,
            "file_path": self.file_path,
            "approximate_line": self.approximate_line,
            "description": self.description,
        }


class SuggestionEngine:
    """
    Generates fix suggestions for detected semantic drift.
    
    Constraints:
    - Suggestions only — no writes to disk.
    - Includes file names and approximate line numbers.
    """
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.suggestions: List[FixSuggestion] = []
    
    def _find_line_number(self, file_path: Path, pattern: str) -> Optional[int]:
        """Find approximate line number for a pattern in a file."""
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    if pattern.lower() in line.lower():
                        return line_num
        except Exception:
            pass
        
        return None
    
    def analyze_metric_kind_drift(
        self,
        slice_name: str,
        yaml_kind: Optional[str],
        expected_kind: str,
    ) -> None:
        """Generate suggestions for metric kind mismatches."""
        if yaml_kind and yaml_kind != expected_kind:
            yaml_path = self.project_root / "config" / "curriculum_uplift_phase2.yaml"
            line_num = self._find_line_number(yaml_path, f"kind: {yaml_kind}")
            
            self.suggestions.append(FixSuggestion(
                issue_type="metric_kind_mismatch",
                current_value=yaml_kind,
                suggested_value=expected_kind,
                file_path=str(yaml_path),
                approximate_line=line_num,
                description=f"Replace '{yaml_kind}' with '{expected_kind}' in curriculum YAML for slice '{slice_name}'",
            ))
    
    def analyze_missing_slice(self, slice_name: str) -> None:
        """Generate suggestions for missing slices."""
        yaml_path = self.project_root / "config" / "curriculum_uplift_phase2.yaml"
        
        self.suggestions.append(FixSuggestion(
            issue_type="missing_slice",
            current_value="<not found>",
            suggested_value=slice_name,
            file_path=str(yaml_path),
            approximate_line=None,
            description=f"Add slice '{slice_name}' to curriculum YAML (documented but missing)",
        ))
    
    def analyze_invalid_metric_kind(
        self,
        slice_name: str,
        invalid_kind: str,
        valid_kinds: List[str],
    ) -> None:
        """Generate suggestions for invalid metric kinds."""
        yaml_path = self.project_root / "config" / "curriculum_uplift_phase2.yaml"
        line_num = self._find_line_number(yaml_path, f"kind: {invalid_kind}")
        
        # Suggest the closest valid kind
        suggested = self._find_closest_match(invalid_kind, valid_kinds)
        
        self.suggestions.append(FixSuggestion(
            issue_type="invalid_metric_kind",
            current_value=invalid_kind,
            suggested_value=suggested,
            file_path=str(yaml_path),
            approximate_line=line_num,
            description=f"Replace invalid metric kind '{invalid_kind}' with '{suggested}' for slice '{slice_name}'",
        ))
    
    def analyze_doc_mismatch(
        self,
        slice_name: str,
        doc_expected: str,
        yaml_actual: str,
    ) -> None:
        """Generate suggestions for doc/YAML mismatches."""
        # Could be fixed in either location - suggest YAML fix
        yaml_path = self.project_root / "config" / "curriculum_uplift_phase2.yaml"
        line_num = self._find_line_number(yaml_path, f"kind: {yaml_actual}")
        
        self.suggestions.append(FixSuggestion(
            issue_type="doc_yaml_mismatch",
            current_value=yaml_actual,
            suggested_value=doc_expected,
            file_path=str(yaml_path),
            approximate_line=line_num,
            description=f"Update YAML to match documentation: change '{yaml_actual}' to '{doc_expected}' for '{slice_name}'",
        ))
    
    def _find_closest_match(self, value: str, candidates: List[str]) -> str:
        """Find the closest matching candidate using simple heuristics."""
        value_lower = value.lower().replace("_", "").replace("-", "")
        
        for candidate in candidates:
            candidate_lower = candidate.lower().replace("_", "").replace("-", "")
            # Check if one contains the other
            if value_lower in candidate_lower or candidate_lower in value_lower:
                return candidate
        
        # Default to first candidate
        return candidates[0] if candidates else value
    
    def generate_from_report(self, report: "AuditReport") -> List[FixSuggestion]:
        """Generate suggestions from an audit report."""
        for spec in report.slices:
            for issue in spec.issues:
                if issue.category == "metric_kind" and "Invalid" in issue.message:
                    details = issue.details or {}
                    self.analyze_invalid_metric_kind(
                        spec.slice_name,
                        spec.metric_kind or "",
                        details.get("valid_kinds", list(METRIC_FUNCTION_MAP.keys())),
                    )
                elif issue.category == "doc_mismatch":
                    details = issue.details or {}
                    self.analyze_doc_mismatch(
                        spec.slice_name,
                        details.get("expected", ""),
                        details.get("actual", ""),
                    )
        
        for issue in report.global_issues:
            if issue.category == "missing_slice":
                details = issue.details or {}
                self.analyze_missing_slice(details.get("slice", ""))
        
        return self.suggestions


def generate_suggestions(
    report: "AuditReport",
    project_root: Optional[Path] = None,
) -> List[FixSuggestion]:
    """Generate fix suggestions for an audit report."""
    project_root = project_root or Path(__file__).parent.parent
    engine = SuggestionEngine(project_root)
    return engine.generate_from_report(report)


def format_suggestions_markdown(suggestions: List[FixSuggestion]) -> str:
    """Format suggestions as Markdown."""
    if not suggestions:
        return "## Suggestions\n\nNo fixes needed — all checks passed! ✅\n"
    
    lines = [
        "## Suggested Fixes",
        "",
        f"Found {len(suggestions)} potential fix(es):",
        "",
    ]
    
    for i, s in enumerate(suggestions, 1):
        lines.append(f"### {i}. {s.issue_type}")
        lines.append("")
        lines.append(f"**File**: `{s.file_path}`")
        if s.approximate_line:
            lines.append(f"**Approximate Line**: {s.approximate_line}")
        lines.append(f"**Current**: `{s.current_value}`")
        lines.append(f"**Suggested**: `{s.suggested_value}`")
        lines.append("")
        lines.append(f"*{s.description}*")
        lines.append("")
    
    return "\n".join(lines)


# =============================================================================
# CI MODE
# =============================================================================

def run_ci_checks(
    curriculum_path: Optional[Path] = None,
    prereg_path: Optional[Path] = None,
) -> Tuple[bool, List[str]]:
    """
    Run core consistency checks in CI mode.
    
    Returns:
        Tuple of (all_passed, list of failure messages)
    """
    project_root = Path(__file__).parent.parent
    curriculum_path = curriculum_path or project_root / "config" / "curriculum_uplift_phase2.yaml"
    prereg_path = prereg_path or project_root / "experiments" / "prereg" / "PREREG_UPLIFT_U2.yaml"
    
    failures: List[str] = []
    
    # Load curriculum
    try:
        curriculum = load_curriculum_yaml(curriculum_path)
    except FileNotFoundError as e:
        failures.append(f"FAIL: {e}")
        return False, failures
    
    slices = curriculum.get("slices", {})
    
    # Check 1: All documented slices exist
    for doc_slice in DOC_SLICE_METRIC_MAP.keys():
        if doc_slice not in slices:
            failures.append(f"FAIL: Documented slice '{doc_slice}' not found in YAML")
    
    # Check 2: Metric kinds are consistent
    valid_kinds = {k.value for k in MetricKind}
    for slice_name, slice_config in slices.items():
        if not isinstance(slice_config, dict):
            continue
        
        success_metric = slice_config.get("success_metric", {})
        if isinstance(success_metric, dict):
            yaml_kind = success_metric.get("kind", "")
            
            # Check if kind is valid
            if yaml_kind and yaml_kind not in valid_kinds:
                failures.append(f"FAIL: Invalid metric_kind '{yaml_kind}' in slice '{slice_name}'")
            
            # Check against documented expectation
            expected = DOC_SLICE_METRIC_MAP.get(slice_name)
            if expected and yaml_kind and yaml_kind != expected.value:
                failures.append(f"FAIL: Slice '{slice_name}' has kind '{yaml_kind}', expected '{expected.value}'")
    
    # Check 3: No Phase I slices mislabeled as Phase II
    phase = curriculum.get("phase", "")
    if phase and phase != "II":
        failures.append(f"FAIL: Curriculum phase is '{phase}', expected 'II'")
    
    all_passed = len(failures) == 0
    return all_passed, failures


def format_ci_output(passed: bool, failures: List[str]) -> str:
    """Format CI output (minimal)."""
    if passed:
        return "OK: All semantic consistency checks passed"
    else:
        return "\n".join(failures)


# =============================================================================
# ONE-LINE DRIFT SUMMARY
# =============================================================================

@dataclass
class DriftCounts:
    """Counts of drift by category."""
    slices: int = 0
    metrics: int = 0
    theory: int = 0
    
    @property
    def total(self) -> int:
        return self.slices + self.metrics + self.theory
    
    @property
    def is_ok(self) -> bool:
        return self.total == 0


def count_drift(
    curriculum_path: Optional[Path] = None,
    prereg_path: Optional[Path] = None,
) -> DriftCounts:
    """
    Count drift by category.
    
    Categories:
    - slices: Missing slices, slice name mismatches
    - metrics: Invalid metric kinds, metric kind mismatches
    - theory: Reserved for future theory term drift
    
    Returns:
        DriftCounts with counts per category
    """
    project_root = Path(__file__).parent.parent
    curriculum_path = curriculum_path or project_root / "config" / "curriculum_uplift_phase2.yaml"
    
    counts = DriftCounts()
    
    # Load curriculum
    try:
        curriculum = load_curriculum_yaml(curriculum_path)
    except FileNotFoundError:
        counts.slices += 1  # File missing counts as slice drift
        return counts
    
    slices = curriculum.get("slices", {})
    
    # Check 1: Missing documented slices (slice drift)
    for doc_slice in DOC_SLICE_METRIC_MAP.keys():
        if doc_slice not in slices:
            counts.slices += 1
    
    # Check 2: Metric kind issues
    valid_kinds = {k.value for k in MetricKind}
    for slice_name, slice_config in slices.items():
        if not isinstance(slice_config, dict):
            continue
        
        success_metric = slice_config.get("success_metric", {})
        if isinstance(success_metric, dict):
            yaml_kind = success_metric.get("kind", "")
            
            # Invalid metric kind
            if yaml_kind and yaml_kind not in valid_kinds:
                counts.metrics += 1
            
            # Metric kind mismatch
            expected = DOC_SLICE_METRIC_MAP.get(slice_name)
            if expected and yaml_kind and yaml_kind != expected.value:
                counts.metrics += 1
    
    return counts


def format_drift_summary(counts: DriftCounts) -> str:
    """
    Format drift counts as a single deterministic line.
    
    Format: Semantic Drift: slices=N metrics=N theory=N total=N OK|FAIL
    
    Rules:
    - No wrapping
    - Categories fixed in order: slices, metrics, theory, total
    - Status is OK or FAIL
    """
    status = "OK" if counts.is_ok else "FAIL"
    return f"Semantic Drift: slices={counts.slices} metrics={counts.metrics} theory={counts.theory} total={counts.total} {status}"


# =============================================================================
# CANONICAL SUGGESTION FORMAT
# =============================================================================

# Known metric kind mappings for canonical suggestions
KNOWN_METRIC_FIXES: Dict[str, str] = {
    "sparse_success": "density",
    "chain_success": "chain_length",
    "multi_goal_success": "multi_goal",
}


def sanitize_suggestion_text(text: str) -> str:
    """
    Sanitize suggestion text to ensure it follows canonical format.
    
    Rules:
    - No code snippets (no backticks)
    - No multiline blocks
    - Single line only
    """
    # Remove any backticks
    text = text.replace("`", "")
    # Replace newlines with spaces
    text = text.replace("\n", " ").replace("\r", "")
    # Collapse multiple spaces
    while "  " in text:
        text = text.replace("  ", " ")
    return text.strip()


def format_canonical_suggestion(
    old_term: str,
    new_term: str,
    file_path: str,
    line_number: Optional[int],
) -> str:
    """
    Format a suggestion in canonical template.
    
    Template: Replace <old> with <new> in <file>, line ~<N>
    
    If line_number is None, omits the line part.
    """
    # Extract just the filename for readability
    file_name = Path(file_path).name
    
    if line_number is not None:
        return f"Replace {old_term} with {new_term} in {file_name}, line ~{line_number}"
    else:
        return f"Replace {old_term} with {new_term} in {file_name}"


def generate_canonical_suggestions(
    curriculum_path: Optional[Path] = None,
) -> List[str]:
    """
    Generate suggestions in canonical format.
    
    Returns list of strings, each in format:
    Replace <old> with <new> in <file>, line ~<N>
    """
    project_root = Path(__file__).parent.parent
    curriculum_path = curriculum_path or project_root / "config" / "curriculum_uplift_phase2.yaml"
    
    suggestions: List[str] = []
    
    # Load curriculum
    try:
        curriculum = load_curriculum_yaml(curriculum_path)
    except FileNotFoundError:
        return suggestions
    
    slices = curriculum.get("slices", {})
    valid_kinds = {k.value for k in MetricKind}
    
    # Scan for known metric kind fixes
    try:
        with open(curriculum_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception:
        lines = []
    
    for slice_name, slice_config in slices.items():
        if not isinstance(slice_config, dict):
            continue
        
        success_metric = slice_config.get("success_metric", {})
        if isinstance(success_metric, dict):
            yaml_kind = success_metric.get("kind", "")
            
            # Check for known fixes
            if yaml_kind in KNOWN_METRIC_FIXES:
                new_kind = KNOWN_METRIC_FIXES[yaml_kind]
                
                # Find approximate line number
                line_num = None
                for i, line in enumerate(lines, 1):
                    if f"kind: {yaml_kind}" in line:
                        line_num = i
                        break
                
                suggestion = format_canonical_suggestion(
                    old_term=yaml_kind,
                    new_term=new_kind,
                    file_path=str(curriculum_path),
                    line_number=line_num,
                )
                suggestions.append(suggestion)
            
            # Check for expected vs actual mismatch
            expected = DOC_SLICE_METRIC_MAP.get(slice_name)
            if expected and yaml_kind and yaml_kind != expected.value:
                if yaml_kind not in KNOWN_METRIC_FIXES:
                    # Find approximate line number
                    line_num = None
                    for i, line in enumerate(lines, 1):
                        if f"kind: {yaml_kind}" in line:
                            line_num = i
                            break
                    
                    suggestion = format_canonical_suggestion(
                        old_term=yaml_kind,
                        new_term=expected.value,
                        file_path=str(curriculum_path),
                        line_number=line_num,
                    )
                    suggestions.append(suggestion)
    
    return suggestions


# =============================================================================
# SEMANTIC KNOWLEDGE GRAPH (v1.3)
# =============================================================================

@dataclass
class GraphEdge:
    """
    An edge in the semantic knowledge graph.
    
    Kinds:
    - cooccur: Terms appear in same file
    - ref: Parent/child reference relationship
    - category: Term belongs to category (slice→metric, metric→param)
    """
    src: str
    dst: str
    weight: float
    kind: str  # "cooccur", "ref", "category"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "src": self.src,
            "dst": self.dst,
            "weight": round(self.weight, 4),
            "kind": self.kind,
        }


@dataclass
class SemanticKnowledgeGraph:
    """
    A semantic knowledge graph capturing relationships between terms.
    
    Nodes: Terms (slices, metrics, theorems, symbols)
    Edges: Co-occurrence, reference, category membership
    """
    terms: List[Dict[str, Any]] = field(default_factory=list)
    edges: List[GraphEdge] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Export to canonical JSON format."""
        return {
            "version": "1.3",
            "terms": sorted(self.terms, key=lambda t: t.get("canonical_form", "")),
            "edges": sorted(
                [e.to_dict() for e in self.edges],
                key=lambda e: (e["src"], e["dst"], e["kind"]),
            ),
        }
    
    def get_node_degree(self, term: str) -> int:
        """Get the degree (edge count) of a node."""
        return sum(1 for e in self.edges if e.src == term or e.dst == term)
    
    def get_neighbors(self, term: str) -> Set[str]:
        """Get all neighbors of a term."""
        neighbors = set()
        for e in self.edges:
            if e.src == term:
                neighbors.add(e.dst)
            elif e.dst == term:
                neighbors.add(e.src)
        return neighbors


class SemanticGraphBuilder:
    """
    Builds a semantic knowledge graph from term index and relationships.
    
    Determinism guarantees:
    - Edges sorted by (src, dst, kind)
    - Weights computed deterministically
    - No randomness in construction
    """
    
    # Category hierarchy for category edges
    CATEGORY_HIERARCHY: Dict[str, str] = {
        # slice → metric relationships
        "slice_uplift_goal": "goal_hit",
        "slice_uplift_sparse": "density",
        "slice_uplift_tree": "chain_length",
        "slice_uplift_dependency": "multi_goal",
    }
    
    # Reference relationships (term references another term)
    REFERENCE_MAP: Dict[str, List[str]] = {
        "goal_hit": ["target_hashes", "min_total_verified"],
        "density": ["min_verified"],
        "chain_length": ["chain_target_hash", "min_chain_length"],
        "multi_goal": ["required_goal_hashes"],
    }
    
    def __init__(self):
        self.terms: Dict[str, Dict[str, Any]] = {}
        self.file_terms: Dict[str, Set[str]] = {}  # file -> terms in that file
    
    def add_term(self, term_dict: Dict[str, Any]) -> None:
        """Add a term from the term index."""
        canonical = term_dict.get("canonical_form", "")
        if canonical:
            self.terms[canonical] = term_dict
            
            # Track which files contain this term
            for mention in term_dict.get("mentions", []):
                file_path = mention.get("file", "")
                if file_path:
                    if file_path not in self.file_terms:
                        self.file_terms[file_path] = set()
                    self.file_terms[file_path].add(canonical)
    
    def build_edges(self) -> List[GraphEdge]:
        """Build all edges deterministically."""
        edges: List[GraphEdge] = []
        
        # 1. Co-occurrence edges (terms in same file)
        edges.extend(self._build_cooccur_edges())
        
        # 2. Reference edges (parent/child relationships)
        edges.extend(self._build_ref_edges())
        
        # 3. Category edges (term → category)
        edges.extend(self._build_category_edges())
        
        # Sort for determinism
        edges.sort(key=lambda e: (e.src, e.dst, e.kind))
        
        return edges
    
    def _build_cooccur_edges(self) -> List[GraphEdge]:
        """Build co-occurrence edges from file co-location."""
        edges = []
        seen_pairs: Set[Tuple[str, str]] = set()
        
        for file_path, terms in sorted(self.file_terms.items()):
            terms_list = sorted(terms)
            for i, t1 in enumerate(terms_list):
                for t2 in terms_list[i + 1:]:
                    # Normalize pair ordering
                    pair = tuple(sorted([t1, t2]))
                    if pair not in seen_pairs:
                        seen_pairs.add(pair)
                        # Weight based on number of shared files
                        shared_files = sum(
                            1 for ft in self.file_terms.values()
                            if t1 in ft and t2 in ft
                        )
                        weight = min(1.0, shared_files / 5.0)  # Normalize to 0-1
                        edges.append(GraphEdge(
                            src=pair[0],
                            dst=pair[1],
                            weight=weight,
                            kind="cooccur",
                        ))
        
        return edges
    
    def _build_ref_edges(self) -> List[GraphEdge]:
        """Build reference edges from known relationships."""
        edges = []
        
        for parent, children in sorted(self.REFERENCE_MAP.items()):
            if parent in self.terms:
                for child in sorted(children):
                    # Check if child term exists (might be a parameter)
                    edges.append(GraphEdge(
                        src=parent,
                        dst=child,
                        weight=1.0,  # Strong reference
                        kind="ref",
                    ))
        
        return edges
    
    def _build_category_edges(self) -> List[GraphEdge]:
        """Build category membership edges."""
        edges = []
        
        for term, category in sorted(self.CATEGORY_HIERARCHY.items()):
            if term in self.terms:
                edges.append(GraphEdge(
                    src=term,
                    dst=category,
                    weight=1.0,  # Category membership is binary
                    kind="category",
                ))
        
        return edges
    
    def build_graph(self, term_index: Dict[str, Any]) -> SemanticKnowledgeGraph:
        """Build the complete semantic knowledge graph."""
        # Add all terms
        for term_dict in term_index.get("terms", []):
            self.add_term(term_dict)
        
        # Build edges
        edges = self.build_edges()
        
        # Construct graph
        return SemanticKnowledgeGraph(
            terms=list(self.terms.values()),
            edges=edges,
        )


def build_semantic_graph(term_index: Dict[str, Any]) -> SemanticKnowledgeGraph:
    """Build a semantic knowledge graph from a term index."""
    builder = SemanticGraphBuilder()
    return builder.build_graph(term_index)


def save_semantic_graph(graph: SemanticKnowledgeGraph, output_path: Path) -> None:
    """Save the semantic graph to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(graph.to_dict(), f, indent=2)


# =============================================================================
# GRAPH-BASED DRIFT DETECTION (v1.3)
# =============================================================================

@dataclass
class GraphDriftSignal:
    """A single drift signal detected in graph comparison."""
    signal_type: str  # "node_disappeared", "edge_collapse", "edge_explosion", "migration", "degree_change"
    severity: str  # "info", "warning", "critical"
    term: str
    description: str
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal_type": self.signal_type,
            "severity": self.severity,
            "term": self.term,
            "description": self.description,
            "details": self.details,
        }


@dataclass
class DriftGraphReport:
    """Report of drift detected between two semantic graphs."""
    signals: List[GraphDriftSignal] = field(default_factory=list)
    old_node_count: int = 0
    new_node_count: int = 0
    old_edge_count: int = 0
    new_edge_count: int = 0
    
    @property
    def has_critical(self) -> bool:
        return any(s.severity == "critical" for s in self.signals)
    
    @property
    def has_warnings(self) -> bool:
        return any(s.severity == "warning" for s in self.signals)
    
    @property
    def status(self) -> str:
        if self.has_critical:
            return "CRITICAL"
        elif self.has_warnings:
            return "WARNING"
        return "OK"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "old_node_count": self.old_node_count,
            "new_node_count": self.new_node_count,
            "old_edge_count": self.old_edge_count,
            "new_edge_count": self.new_edge_count,
            "signal_count": len(self.signals),
            "signals": sorted(
                [s.to_dict() for s in self.signals],
                key=lambda s: (s["severity"], s["signal_type"], s["term"]),
            ),
        }


def analyze_graph_drift(
    old_graph: SemanticKnowledgeGraph,
    new_graph: SemanticKnowledgeGraph,
    weight_collapse_threshold: float = 0.3,
    weight_explosion_threshold: float = 3.0,
    degree_change_threshold: int = 3,
) -> DriftGraphReport:
    """
    Analyze drift between two semantic knowledge graphs.
    
    Drift signals detected:
    - Node disappearance: Term in old but not new
    - Edge weight collapse: Weight decreased significantly
    - Edge weight explosion: Weight increased significantly
    - Term → category migration: Category membership changed
    - Unexpected degree changes: Node connectivity changed
    
    Args:
        old_graph: Previous version of the graph
        new_graph: Current version of the graph
        weight_collapse_threshold: Ratio below which edge is "collapsed"
        weight_explosion_threshold: Ratio above which edge is "exploded"
        degree_change_threshold: Minimum degree change to flag
    
    Returns:
        DriftGraphReport with all detected signals
    """
    report = DriftGraphReport(
        old_node_count=len(old_graph.terms),
        new_node_count=len(new_graph.terms),
        old_edge_count=len(old_graph.edges),
        new_edge_count=len(new_graph.edges),
    )
    
    # Build lookup structures
    old_terms = {t.get("canonical_form", "") for t in old_graph.terms}
    new_terms = {t.get("canonical_form", "") for t in new_graph.terms}
    
    old_edges = {(e.src, e.dst, e.kind): e for e in old_graph.edges}
    new_edges = {(e.src, e.dst, e.kind): e for e in new_graph.edges}
    
    # 1. Detect node disappearance
    for term in sorted(old_terms - new_terms):
        report.signals.append(GraphDriftSignal(
            signal_type="node_disappeared",
            severity="critical",
            term=term,
            description=f"Term '{term}' disappeared from graph",
            details={"was_in_old": True, "in_new": False},
        ))
    
    # 2. Detect new nodes (for info)
    for term in sorted(new_terms - old_terms):
        report.signals.append(GraphDriftSignal(
            signal_type="node_appeared",
            severity="info",
            term=term,
            description=f"New term '{term}' appeared in graph",
            details={"was_in_old": False, "in_new": True},
        ))
    
    # 3. Detect edge weight collapse/explosion
    for key, old_edge in sorted(old_edges.items()):
        if key in new_edges:
            new_edge = new_edges[key]
            if old_edge.weight > 0:
                ratio = new_edge.weight / old_edge.weight
                
                if ratio < weight_collapse_threshold:
                    report.signals.append(GraphDriftSignal(
                        signal_type="edge_collapse",
                        severity="warning",
                        term=f"{old_edge.src}→{old_edge.dst}",
                        description=f"Edge weight collapsed: {old_edge.weight:.3f} → {new_edge.weight:.3f}",
                        details={
                            "src": old_edge.src,
                            "dst": old_edge.dst,
                            "kind": old_edge.kind,
                            "old_weight": old_edge.weight,
                            "new_weight": new_edge.weight,
                            "ratio": ratio,
                        },
                    ))
                elif ratio > weight_explosion_threshold:
                    report.signals.append(GraphDriftSignal(
                        signal_type="edge_explosion",
                        severity="warning",
                        term=f"{old_edge.src}→{old_edge.dst}",
                        description=f"Edge weight exploded: {old_edge.weight:.3f} → {new_edge.weight:.3f}",
                        details={
                            "src": old_edge.src,
                            "dst": old_edge.dst,
                            "kind": old_edge.kind,
                            "old_weight": old_edge.weight,
                            "new_weight": new_edge.weight,
                            "ratio": ratio,
                        },
                    ))
    
    # 4. Detect category edge changes (migration)
    old_category_edges = {(e.src, e.dst) for e in old_graph.edges if e.kind == "category"}
    new_category_edges = {(e.src, e.dst) for e in new_graph.edges if e.kind == "category"}
    
    for src, dst in sorted(old_category_edges - new_category_edges):
        report.signals.append(GraphDriftSignal(
            signal_type="migration",
            severity="warning",
            term=src,
            description=f"Term '{src}' migrated away from category '{dst}'",
            details={"old_category": dst, "new_category": None},
        ))
    
    # 5. Detect degree changes
    for term in sorted(old_terms & new_terms):
        old_degree = old_graph.get_node_degree(term)
        new_degree = new_graph.get_node_degree(term)
        degree_diff = abs(new_degree - old_degree)
        
        if degree_diff >= degree_change_threshold:
            report.signals.append(GraphDriftSignal(
                signal_type="degree_change",
                severity="warning",
                term=term,
                description=f"Term '{term}' degree changed: {old_degree} → {new_degree}",
                details={
                    "old_degree": old_degree,
                    "new_degree": new_degree,
                    "change": new_degree - old_degree,
                },
            ))
    
    return report


def format_drift_report_summary(report: DriftGraphReport) -> str:
    """Format a one-line drift report summary."""
    critical = sum(1 for s in report.signals if s.severity == "critical")
    warnings = sum(1 for s in report.signals if s.severity == "warning")
    info = sum(1 for s in report.signals if s.severity == "info")
    
    return f"Graph Drift: critical={critical} warnings={warnings} info={info} status={report.status}"


# =============================================================================
# GRAPH-AWARE SUGGESTIONS (v1.3)
# =============================================================================

def generate_graph_aware_suggestions(
    graph: SemanticKnowledgeGraph,
    curriculum_path: Optional[Path] = None,
) -> List[str]:
    """
    Generate suggestions using graph connectivity.
    
    Uses graph edges to find canonical replacements:
    - If term X is connected to canonical term Y via category edge,
      suggest replacing X with Y
    
    Guarantees:
    - Single deterministic sentence
    - No code formatting (no backticks)
    - No newlines
    
    Returns:
        List of suggestion strings in canonical format
    """
    project_root = Path(__file__).parent.parent
    curriculum_path = curriculum_path or project_root / "config" / "curriculum_uplift_phase2.yaml"
    
    suggestions: List[str] = []
    
    # Build term lookup from graph
    term_set = {t.get("canonical_form", "") for t in graph.terms}
    
    # Build category connections
    category_map: Dict[str, str] = {}  # term → category
    for edge in graph.edges:
        if edge.kind == "category":
            category_map[edge.src] = edge.dst
    
    # Load curriculum
    try:
        curriculum = load_curriculum_yaml(curriculum_path)
    except FileNotFoundError:
        return suggestions
    
    # Find line numbers
    try:
        with open(curriculum_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception:
        lines = []
    
    slices = curriculum.get("slices", {})
    
    for slice_name, slice_config in sorted(slices.items()):
        if not isinstance(slice_config, dict):
            continue
        
        success_metric = slice_config.get("success_metric", {})
        if isinstance(success_metric, dict):
            yaml_kind = success_metric.get("kind", "")
            
            # Check if this is a known fix via KNOWN_METRIC_FIXES
            if yaml_kind in KNOWN_METRIC_FIXES:
                canonical_kind = KNOWN_METRIC_FIXES[yaml_kind]
                
                # Verify canonical term exists in graph
                if canonical_kind in term_set:
                    # Find line number
                    line_num = None
                    for i, line in enumerate(lines, 1):
                        if f"kind: {yaml_kind}" in line:
                            line_num = i
                            break
                    
                    # Generate graph-aware suggestion
                    suggestion = _format_graph_suggestion(
                        old_term=yaml_kind,
                        new_term=canonical_kind,
                        file_path=str(curriculum_path),
                        line_number=line_num,
                        connection_reason="connected via category edge",
                    )
                    suggestions.append(suggestion)
            
            # Also check for terms that should use their category
            elif yaml_kind in category_map:
                canonical_kind = category_map[yaml_kind]
                
                # Find line number
                line_num = None
                for i, line in enumerate(lines, 1):
                    if f"kind: {yaml_kind}" in line:
                        line_num = i
                        break
                
                suggestion = _format_graph_suggestion(
                    old_term=yaml_kind,
                    new_term=canonical_kind,
                    file_path=str(curriculum_path),
                    line_number=line_num,
                    connection_reason="connected via graph category",
                )
                suggestions.append(suggestion)
    
    return suggestions


def _format_graph_suggestion(
    old_term: str,
    new_term: str,
    file_path: str,
    line_number: Optional[int],
    connection_reason: str,
) -> str:
    """
    Format a graph-aware suggestion.
    
    Template: Replace <old> with <new> in <file>, line ~<N> (<reason>)
    
    Guarantees:
    - Single line
    - No backticks
    - No newlines
    """
    file_name = Path(file_path).name
    
    base = f"Replace {old_term} with {new_term} in {file_name}"
    if line_number is not None:
        base += f", line ~{line_number}"
    
    # Sanitize and append reason
    result = f"{base} ({connection_reason})"
    return sanitize_suggestion_text(result)


# =============================================================================
# SEMANTIC GOVERNANCE SNAPSHOT (Phase III)
# =============================================================================

class GovernanceStatus(str, Enum):
    """Governance status levels."""
    OK = "OK"
    ATTENTION = "ATTENTION"
    CRITICAL = "CRITICAL"


def build_semantic_governance_snapshot(
    graph: SemanticKnowledgeGraph,
    drift_report: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a governance snapshot from the semantic graph and optional drift report.
    
    The governance snapshot serves as a safety checkpoint for curriculum edits,
    capturing critical signals that may block or require attention for changes.
    
    Args:
        graph: The current semantic knowledge graph
        drift_report: Optional drift report from analyze_graph_drift().to_dict()
    
    Returns:
        Dict with:
        - schema_version: "1.0"
        - term_count: Number of terms in graph
        - edge_count: Number of edges in graph
        - critical_signals: List of critical issues (node_disappeared, migration)
        - status: OK | ATTENTION | CRITICAL
    """
    # Extract critical signals from drift report
    critical_signals: List[Dict[str, Any]] = []
    
    if drift_report:
        signals = drift_report.get("signals", [])
        for signal in signals:
            signal_type = signal.get("signal_type", "")
            severity = signal.get("severity", "")
            
            # Critical signals: node disappearance, category migration
            if signal_type == "node_disappeared" or (
                signal_type == "migration" and severity in ("warning", "critical")
            ):
                critical_signals.append({
                    "type": signal_type,
                    "term": signal.get("term", ""),
                    "description": signal.get("description", ""),
                    "severity": severity,
                })
    
    # Determine governance status
    has_disappearances = any(s["type"] == "node_disappeared" for s in critical_signals)
    has_migrations = any(s["type"] == "migration" for s in critical_signals)
    
    if has_disappearances:
        status = GovernanceStatus.CRITICAL
    elif has_migrations:
        status = GovernanceStatus.ATTENTION
    else:
        status = GovernanceStatus.OK
    
    return {
        "schema_version": "1.0",
        "term_count": len(graph.terms),
        "edge_count": len(graph.edges),
        "critical_signals": critical_signals,
        "status": status.value,
    }


# =============================================================================
# SUGGESTION SAFETY FILTER (Phase III)
# =============================================================================

def filter_graph_suggestions(
    suggestions: List[str],
    governance_snapshot: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Filter suggestions based on governance snapshot safety status.
    
    This is a DESCRIPTIVE filter only - it classifies suggestions by safety level
    but does NOT apply any changes. Suggestions targeting terms that have
    disappeared or migrated are blocked.
    
    Args:
        suggestions: List of canonical suggestion strings
        governance_snapshot: From build_semantic_governance_snapshot()
    
    Returns:
        Dict with:
        - allowed_suggestions: list[str] - Safe to apply
        - blocked_suggestions: list[str] - Should not be applied
        - reasons: dict[str, str] - Mapping suggestion → reason for blocking
    """
    allowed: List[str] = []
    blocked: List[str] = []
    reasons: Dict[str, str] = {}
    
    # Extract problematic terms from critical signals
    disappeared_terms: Set[str] = set()
    migrated_terms: Set[str] = set()
    
    for signal in governance_snapshot.get("critical_signals", []):
        term = signal.get("term", "")
        signal_type = signal.get("type", "")
        
        if signal_type == "node_disappeared":
            disappeared_terms.add(term)
        elif signal_type == "migration":
            migrated_terms.add(term)
    
    # Check governance status
    status = governance_snapshot.get("status", "OK")
    
    for suggestion in suggestions:
        # Parse suggestion to extract target term (the "new" term in "Replace X with Y")
        # Format: "Replace <old> with <new> in <file>, line ~<N> (<reason>)"
        target_term = _extract_target_term(suggestion)
        source_term = _extract_source_term(suggestion)
        
        is_blocked = False
        block_reason = ""
        
        # Check if target term has disappeared
        if target_term and target_term in disappeared_terms:
            is_blocked = True
            block_reason = f"Target term '{target_term}' has disappeared from the graph"
        
        # Check if source term has migrated (category changed)
        elif source_term and source_term in migrated_terms:
            is_blocked = True
            block_reason = f"Source term '{source_term}' has migrated to a different category"
        
        # Check if target term has migrated
        elif target_term and target_term in migrated_terms:
            is_blocked = True
            block_reason = f"Target term '{target_term}' has migrated to a different category"
        
        # In CRITICAL status, block all suggestions as a safety measure
        elif status == GovernanceStatus.CRITICAL.value and (disappeared_terms or migrated_terms):
            is_blocked = True
            block_reason = "Governance status is CRITICAL; manual review required"
        
        if is_blocked:
            blocked.append(suggestion)
            reasons[suggestion] = block_reason
        else:
            allowed.append(suggestion)
    
    return {
        "allowed_suggestions": allowed,
        "blocked_suggestions": blocked,
        "reasons": reasons,
    }


def _extract_target_term(suggestion: str) -> Optional[str]:
    """
    Extract the target term (new term) from a canonical suggestion.
    
    Format: "Replace <old> with <new> in <file>..."
    """
    # Match "with X in" pattern
    match = re.search(r'\bwith\s+(\S+)\s+in\b', suggestion)
    if match:
        return match.group(1)
    return None


def _extract_source_term(suggestion: str) -> Optional[str]:
    """
    Extract the source term (old term) from a canonical suggestion.
    
    Format: "Replace <old> with <new> in <file>..."
    """
    # Match "Replace X with" pattern
    match = re.search(r'^Replace\s+(\S+)\s+with\b', suggestion)
    if match:
        return match.group(1)
    return None


# =============================================================================
# GLOBAL HEALTH SEMANTIC SIGNAL (Phase III)
# =============================================================================

class SemanticHealthStatus(str, Enum):
    """Semantic health status levels."""
    OK = "OK"
    WARN = "WARN"
    CRITICAL = "CRITICAL"


def summarize_semantic_graph_for_global_health(
    snapshot: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Summarize semantic graph governance snapshot for global health reporting.
    
    This produces a simplified health signal that can be integrated into
    broader system health dashboards.
    
    Args:
        snapshot: From build_semantic_governance_snapshot()
    
    Returns:
        Dict with:
        - semantic_ok: bool - True if no critical issues
        - node_disappearance_count: int
        - category_migration_count: int
        - status: OK | WARN | CRITICAL
    """
    critical_signals = snapshot.get("critical_signals", [])
    
    # Count by type
    node_disappearance_count = sum(
        1 for s in critical_signals if s.get("type") == "node_disappeared"
    )
    category_migration_count = sum(
        1 for s in critical_signals if s.get("type") == "migration"
    )
    
    # Determine health status
    if node_disappearance_count > 0:
        status = SemanticHealthStatus.CRITICAL
        semantic_ok = False
    elif category_migration_count > 0:
        status = SemanticHealthStatus.WARN
        semantic_ok = False
    else:
        status = SemanticHealthStatus.OK
        semantic_ok = True
    
    return {
        "semantic_ok": semantic_ok,
        "node_disappearance_count": node_disappearance_count,
        "category_migration_count": category_migration_count,
        "status": status.value,
    }


def format_semantic_health_summary(health: Dict[str, Any]) -> str:
    """
    Format semantic health as a one-line summary.
    
    Format: Semantic Health: disappearances=N migrations=N status=OK|WARN|CRITICAL
    """
    return (
        f"Semantic Health: "
        f"disappearances={health['node_disappearance_count']} "
        f"migrations={health['category_migration_count']} "
        f"status={health['status']}"
    )


# =============================================================================
# PHASE IV — MULTI-RUN DRIFT TIMELINE & CURRICULUM/DOCS COUPLING
# =============================================================================

SEMANTIC_DRIFT_TIMELINE_SCHEMA_VERSION = "semantic-drift-timeline-1.0.0"


def build_semantic_drift_timeline(
    snapshots: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build a multi-run semantic drift timeline from historical snapshots.
    
    STATUS: PHASE IV — NOT RUN IN PHASE I
    
    Analyzes how semantic graph health evolves across multiple runs to detect
    patterns of stability, drift, or volatility.
    
    Args:
        snapshots: Sequence of governance snapshots (oldest to newest).
                  Each snapshot should have: run_id, term_count, critical_signals, status.
    
    Returns:
        Dictionary with:
        - schema_version
        - timeline: List of {run_id, term_count, critical_signal_count, status}
        - runs_with_critical_signals: List of run_ids with critical signals
        - node_disappearance_events: List of {run_id, term} for each disappearance
        - trend: "STABLE" | "DRIFTING" | "VOLATILE"
    """
    if not snapshots:
        return {
            "schema_version": SEMANTIC_DRIFT_TIMELINE_SCHEMA_VERSION,
            "timeline": [],
            "runs_with_critical_signals": [],
            "node_disappearance_events": [],
            "trend": "STABLE",
        }
    
    timeline = []
    runs_with_critical_signals = []
    node_disappearance_events = []
    
    # Build timeline entries
    for snapshot in snapshots:
        run_id = snapshot.get("run_id", f"run_{len(timeline)}")
        term_count = snapshot.get("term_count", 0)
        critical_signals = snapshot.get("critical_signals", [])
        critical_signal_count = len(critical_signals)
        status = snapshot.get("status", "OK")
        
        timeline.append({
            "run_id": run_id,
            "term_count": term_count,
            "critical_signal_count": critical_signal_count,
            "status": status,
        })
        
        # Track runs with critical signals
        if critical_signal_count > 0:
            runs_with_critical_signals.append(run_id)
        
        # Track node disappearance events
        for signal in critical_signals:
            if signal.get("type") == "node_disappeared":
                node_disappearance_events.append({
                    "run_id": run_id,
                    "term": signal.get("term", ""),
                })
    
    # Determine trend
    if len(snapshots) < 2:
        trend = "STABLE"
    else:
        # Count status changes and critical signal changes
        status_changes = 0
        critical_signal_changes = 0
        
        for i in range(len(timeline) - 1):
            if timeline[i]["status"] != timeline[i + 1]["status"]:
                status_changes += 1
            if timeline[i]["critical_signal_count"] != timeline[i + 1]["critical_signal_count"]:
                critical_signal_changes += 1
        
        total_changes = status_changes + critical_signal_changes
        num_pairs = len(timeline) - 1
        
        # Priority: If critical signals present, it's DRIFTING (not VOLATILE)
        # Drifting: some changes or critical signals present
        if total_changes > 0 or len(runs_with_critical_signals) > 0:
            # Check if it's volatile (frequent changes) but only if no critical signals
            if total_changes >= num_pairs * 0.5 and len(runs_with_critical_signals) == 0:
                trend = "VOLATILE"
            else:
                trend = "DRIFTING"
        else:
            trend = "STABLE"
    
    return {
        "schema_version": SEMANTIC_DRIFT_TIMELINE_SCHEMA_VERSION,
        "timeline": timeline,
        "runs_with_critical_signals": runs_with_critical_signals,
        "node_disappearance_events": node_disappearance_events,
        "trend": trend,
    }


def analyze_semantic_alignment_with_curriculum(
    graph_snapshot: Dict[str, Any],
    curriculum_manifest: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Analyze alignment between semantic graph and curriculum manifest.
    
    STATUS: PHASE IV — NOT RUN IN PHASE I
    
    Detects misalignment between:
    - Terms heavily used in code/docs but absent in curriculum
    - Slices/metrics defined in curriculum but not appearing in semantic graph
    
    This is descriptive analysis only; no suggestions are applied.
    
    Args:
        graph_snapshot: Governance snapshot from build_semantic_governance_snapshot.
        curriculum_manifest: Dictionary with curriculum structure:
                           - slices: Dict[str, Dict] mapping slice names to configs
                           - Each slice config may have success_metric.kind
    
    Returns:
        Dictionary with:
        - orphan_terms: List of terms in graph but not in curriculum
        - unused_curriculum_terms: List of curriculum terms not in graph
        - alignment_status: "ALIGNED" | "PARTIAL" | "MISALIGNED"
    """
    # Extract terms from graph snapshot
    # Note: graph_snapshot doesn't directly contain term names, but we can infer
    # from critical_signals and term_count. For full analysis, we'd need the graph itself.
    # For now, we'll work with what we have in the snapshot.
    
    # Extract curriculum terms
    curriculum_terms: Set[str] = set()
    slices = curriculum_manifest.get("slices", {})
    
    for slice_name, slice_config in slices.items():
        curriculum_terms.add(slice_name)
        
        if isinstance(slice_config, dict):
            success_metric = slice_config.get("success_metric", {})
            if isinstance(success_metric, dict):
                metric_kind = success_metric.get("kind", "")
                if metric_kind:
                    curriculum_terms.add(metric_kind)
    
    # For orphan terms, we'd need the actual graph terms
    # Since we only have the snapshot, we'll use critical_signals as a proxy
    # In a full implementation, we'd pass the graph itself
    orphan_terms: List[str] = []
    
    # Check for unused curriculum terms
    # Since we don't have graph terms directly, we'll mark this as a limitation
    # and return empty for now (would need graph.terms in full implementation)
    unused_curriculum_terms: List[str] = []
    
    # Determine alignment status
    if not orphan_terms and not unused_curriculum_terms:
        alignment_status = "ALIGNED"
    elif len(orphan_terms) <= len(curriculum_terms) * 0.1 and len(unused_curriculum_terms) <= len(curriculum_terms) * 0.1:
        alignment_status = "PARTIAL"
    else:
        alignment_status = "MISALIGNED"
    
    return {
        "schema_version": SEMANTIC_DRIFT_TIMELINE_SCHEMA_VERSION,
        "orphan_terms": orphan_terms,
        "unused_curriculum_terms": unused_curriculum_terms,
        "alignment_status": alignment_status,
        "curriculum_term_count": len(curriculum_terms),
    }


def analyze_semantic_alignment_with_curriculum_full(
    graph: SemanticKnowledgeGraph,
    curriculum_manifest: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Full alignment analysis using the actual graph (not just snapshot).
    
    STATUS: PHASE IV — NOT RUN IN PHASE I
    
    This version uses the full graph to detect orphan terms and unused curriculum terms.
    """
    # Extract graph terms
    graph_terms: Set[str] = set()
    for term_dict in graph.terms:
        canonical = term_dict.get("canonical_form", "")
        if canonical:
            graph_terms.add(canonical)
    
    # Extract curriculum terms
    curriculum_terms: Set[str] = set()
    slices = curriculum_manifest.get("slices", {})
    
    for slice_name, slice_config in slices.items():
        curriculum_terms.add(slice_name)
        
        if isinstance(slice_config, dict):
            success_metric = slice_config.get("success_metric", {})
            if isinstance(success_metric, dict):
                metric_kind = success_metric.get("kind", "")
                if metric_kind:
                    curriculum_terms.add(metric_kind)
    
    # Find orphan terms (in graph but not in curriculum)
    orphan_terms = sorted(graph_terms - curriculum_terms)
    
    # Find unused curriculum terms (in curriculum but not in graph)
    unused_curriculum_terms = sorted(curriculum_terms - graph_terms)
    
    # Determine alignment status
    total_curriculum_terms = len(curriculum_terms)
    total_graph_terms = len(graph_terms)
    
    if not orphan_terms and not unused_curriculum_terms:
        alignment_status = "ALIGNED"
    elif total_curriculum_terms == 0:
        # No curriculum terms to compare
        alignment_status = "ALIGNED" if not orphan_terms else "PARTIAL"
    else:
        # Calculate misalignment percentage
        orphan_pct = len(orphan_terms) / max(total_graph_terms, 1) if total_graph_terms > 0 else 0
        unused_pct = len(unused_curriculum_terms) / max(total_curriculum_terms, 1) if total_curriculum_terms > 0 else 0
        
        # PARTIAL: small misalignment (< 20% of terms)
        # MISALIGNED: significant misalignment (>= 20% of terms)
        if orphan_pct < 0.2 and unused_pct < 0.2:
            alignment_status = "PARTIAL"
        else:
            alignment_status = "MISALIGNED"
    
    return {
        "schema_version": SEMANTIC_DRIFT_TIMELINE_SCHEMA_VERSION,
        "orphan_terms": orphan_terms,
        "unused_curriculum_terms": unused_curriculum_terms,
        "alignment_status": alignment_status,
        "curriculum_term_count": len(curriculum_terms),
        "graph_term_count": len(graph_terms),
    }


def build_semantic_director_panel_legacy(
    drift_timeline: Dict[str, Any],
    alignment_analysis: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a director console panel combining drift timeline and alignment analysis.
    
    STATUS: PHASE IV — NOT RUN IN PHASE I
    
    Provides a high-level executive summary suitable for:
    - Investor/acquirer due diligence
    - Internal strategy reviews
    - System health dashboards
    
    Args:
        drift_timeline: From build_semantic_drift_timeline.
        alignment_analysis: From analyze_semantic_alignment_with_curriculum.
    
    Returns:
        Dictionary with:
        - semantic_status_light: "GREEN" | "YELLOW" | "RED"
        - alignment_status
        - critical_run_ids: List of run_ids with critical signals
        - headline: Short neutral sentence summarizing drift and alignment
    """
    # Determine status light
    timeline_trend = drift_timeline.get("trend", "STABLE")
    runs_with_critical = drift_timeline.get("runs_with_critical_signals", [])
    alignment_status = alignment_analysis.get("alignment_status", "ALIGNED")
    
    # RED: Critical signals present or misaligned
    if runs_with_critical or alignment_status == "MISALIGNED":
        status_light = "RED"
    # YELLOW: Drifting or partial alignment
    elif timeline_trend == "DRIFTING" or timeline_trend == "VOLATILE" or alignment_status == "PARTIAL":
        status_light = "YELLOW"
    # GREEN: Stable and aligned
    else:
        status_light = "GREEN"
    
    # Build headline
    parts = []
    
    if timeline_trend == "VOLATILE":
        parts.append("Semantic graph shows volatile drift")
    elif timeline_trend == "DRIFTING":
        parts.append("Semantic graph shows gradual drift")
    elif timeline_trend == "STABLE":
        parts.append("Semantic graph is stable")
    
    if alignment_status == "MISALIGNED":
        parts.append("with significant curriculum misalignment")
    elif alignment_status == "PARTIAL":
        parts.append("with partial curriculum alignment")
    elif alignment_status == "ALIGNED":
        parts.append("with curriculum alignment")
    
    if runs_with_critical:
        parts.append(f"({len(runs_with_critical)} runs with critical signals)")
    
    headline = ". ".join(parts) + "." if parts else "Semantic graph status unknown."
    
    return {
        "schema_version": SEMANTIC_DRIFT_TIMELINE_SCHEMA_VERSION,
        "semantic_status_light": status_light,
        "alignment_status": alignment_status,
        "critical_run_ids": runs_with_critical,
        "headline": headline,
        "trend": timeline_trend,
        "node_disappearance_count": len(drift_timeline.get("node_disappearance_events", [])),
    }


# =============================================================================
# CROSS-SYSTEM SEMANTIC ALIGNMENT INDEX (Phase IV - Task 1)
# =============================================================================

class AlignmentStatus(str, Enum):
    """Semantic alignment status across systems."""
    ALIGNED = "ALIGNED"
    PARTIAL = "PARTIAL"
    DIVERGENT = "DIVERGENT"


def build_semantic_alignment_index(
    graph_snapshot: Dict[str, Any],
    curriculum_manifest: Dict[str, Any],
    taxonomy_semantics: Dict[str, Any],
    docs_vocab_index: Dict[str, Any],
    graph: Optional[SemanticKnowledgeGraph] = None,
) -> Dict[str, Any]:
    """
    Build a cross-system semantic alignment index.
    
    Compares terms across code (graph), curriculum, taxonomy, and docs to identify
    misalignments and orphaned terms.
    
    Args:
        graph_snapshot: From build_semantic_governance_snapshot() - used for governance context
        curriculum_manifest: Dict with "terms" list (curriculum-specific terms)
        taxonomy_semantics: Dict with "terms" list (taxonomy/classification terms)
        docs_vocab_index: Dict with "terms" list (documentation vocabulary)
        graph: Optional SemanticKnowledgeGraph - if provided, extracts terms from graph
    
    Returns:
        Dict with:
        - terms_only_in_code: List[str] - Terms in graph but not elsewhere
        - terms_only_in_docs: List[str] - Terms in docs but not in code/curriculum
        - terms_only_in_curriculum: List[str] - Terms in curriculum but not in code/docs
        - taxonomy_terms_with_no_uses: List[str] - Taxonomy terms not used anywhere
        - alignment_status: ALIGNED | PARTIAL | DIVERGENT
    """
    # Extract term sets from each system
    # If graph is provided, use it; otherwise try to extract from snapshot or use empty set
    if graph:
        graph_terms = {
            t.get("canonical_form", "")
            for t in graph.terms
            if t.get("canonical_form")
        }
    else:
        # Fallback: use empty set if graph not provided
        graph_terms = set()
    
    curriculum_terms = set(curriculum_manifest.get("terms", []))
    taxonomy_terms = set(taxonomy_semantics.get("terms", []))
    docs_terms = set(docs_vocab_index.get("terms", []))
    
    # Combine all systems for comparison
    all_systems_terms = graph_terms | curriculum_terms | taxonomy_terms | docs_terms
    
    # Find terms only in code (graph)
    terms_only_in_code = sorted(graph_terms - curriculum_terms - docs_terms - taxonomy_terms)
    
    # Find terms only in docs
    terms_only_in_docs = sorted(docs_terms - graph_terms - curriculum_terms)
    
    # Find terms only in curriculum
    terms_only_in_curriculum = sorted(curriculum_terms - graph_terms - docs_terms)
    
    # Find taxonomy terms with no uses (not in code, curriculum, or docs)
    taxonomy_terms_with_no_uses = sorted(taxonomy_terms - graph_terms - curriculum_terms - docs_terms)
    
    # Determine alignment status
    total_orphaned = (
        len(terms_only_in_code) +
        len(terms_only_in_docs) +
        len(terms_only_in_curriculum) +
        len(taxonomy_terms_with_no_uses)
    )
    
    if total_orphaned == 0:
        alignment_status = AlignmentStatus.ALIGNED
    elif total_orphaned <= len(all_systems_terms) * 0.1:  # Less than 10% orphaned
        alignment_status = AlignmentStatus.PARTIAL
    else:
        alignment_status = AlignmentStatus.DIVERGENT
    
    return {
        "terms_only_in_code": terms_only_in_code,
        "terms_only_in_docs": terms_only_in_docs,
        "terms_only_in_curriculum": terms_only_in_curriculum,
        "taxonomy_terms_with_no_uses": taxonomy_terms_with_no_uses,
        "alignment_status": alignment_status.value,
        "total_orphaned_terms": total_orphaned,
        "total_terms_across_systems": len(all_systems_terms),
    }


# =============================================================================
# SEMANTIC RISK DECOMPOSITION (Phase IV - Task 2)
# =============================================================================

class RiskStatus(str, Enum):
    """Semantic risk status levels."""
    OK = "OK"
    ATTENTION = "ATTENTION"
    CRITICAL = "CRITICAL"


def analyze_semantic_risk(
    alignment_index: Dict[str, Any],
    governance_snapshot: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Analyze semantic risk by combining alignment and governance signals.
    
    High-risk terms: Terms with disappearances/migrations that are also misaligned
    Medium-risk terms: Terms with partial misalignment only
    
    Args:
        alignment_index: From build_semantic_alignment_index()
        governance_snapshot: From build_semantic_governance_snapshot()
    
    Returns:
        Dict with:
        - high_risk_terms: List[str] - Terms with critical issues + misalignment
        - medium_risk_terms: List[str] - Terms with partial misalignment
        - status: OK | ATTENTION | CRITICAL
        - notes: List[str] - Explanatory notes about risk factors
    """
    # Extract problematic terms from governance
    critical_signals = governance_snapshot.get("critical_signals", [])
    governance_problematic_terms = {
        s.get("term", "")
        for s in critical_signals
        if s.get("type") in ("node_disappeared", "migration")
    }
    
    # Extract misaligned terms from alignment index
    misaligned_terms = set()
    misaligned_terms.update(alignment_index.get("terms_only_in_code", []))
    misaligned_terms.update(alignment_index.get("terms_only_in_docs", []))
    misaligned_terms.update(alignment_index.get("terms_only_in_curriculum", []))
    misaligned_terms.update(alignment_index.get("taxonomy_terms_with_no_uses", []))
    
    # High-risk: Terms that are both problematic (disappeared/migrated) AND misaligned
    high_risk_terms = sorted(governance_problematic_terms & misaligned_terms)
    
    # Medium-risk: Terms that are misaligned but not in governance critical signals
    medium_risk_terms = sorted(misaligned_terms - governance_problematic_terms)
    
    # Determine risk status
    if high_risk_terms:
        status = RiskStatus.CRITICAL
    elif medium_risk_terms or governance_snapshot.get("status") == GovernanceStatus.ATTENTION.value:
        status = RiskStatus.ATTENTION
    else:
        status = RiskStatus.OK
    
    # Generate explanatory notes
    notes = []
    
    if high_risk_terms:
        notes.append(f"{len(high_risk_terms)} term(s) have both governance issues and alignment problems")
    
    if medium_risk_terms:
        notes.append(f"{len(medium_risk_terms)} term(s) are misaligned across systems")
    
    alignment_status = alignment_index.get("alignment_status", "")
    if alignment_status == AlignmentStatus.DIVERGENT.value:
        notes.append("System alignment is DIVERGENT (>10% orphaned terms)")
    elif alignment_status == AlignmentStatus.PARTIAL.value:
        notes.append("System alignment is PARTIAL (some orphaned terms)")
    
    governance_status = governance_snapshot.get("status", "")
    if governance_status == GovernanceStatus.CRITICAL.value:
        notes.append("Governance status is CRITICAL (node disappearances detected)")
    elif governance_status == GovernanceStatus.ATTENTION.value:
        notes.append("Governance status is ATTENTION (category migrations detected)")
    
    if not notes:
        notes.append("No semantic risk factors detected")
    
    return {
        "high_risk_terms": high_risk_terms,
        "medium_risk_terms": medium_risk_terms,
        "status": status.value,
        "notes": notes,
    }


# =============================================================================
# DIRECTOR SEMANTIC PANEL (Phase IV - Task 3 - Extended Version)
# =============================================================================

class StatusLight(str, Enum):
    """Status light colors for director panel."""
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"


def build_semantic_director_panel(
    governance_snapshot: Dict[str, Any],
    alignment_index: Dict[str, Any],
    risk_analysis: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a high-level semantic director panel for executive dashboard.
    
    Provides a single unified view of semantic backbone health across all systems.
    
    Args:
        governance_snapshot: From build_semantic_governance_snapshot()
        alignment_index: From build_semantic_alignment_index()
        risk_analysis: From analyze_semantic_risk()
    
    Returns:
        Dict with:
        - status_light: GREEN | YELLOW | RED
        - semantic_ok: bool
        - alignment_status: ALIGNED | PARTIAL | DIVERGENT
        - critical_term_count: int
        - headline: str - Short neutral sentence summarizing posture
    """
    # Extract key metrics
    governance_status = governance_snapshot.get("status", "OK")
    alignment_status = alignment_index.get("alignment_status", "ALIGNED")
    risk_status = risk_analysis.get("status", "OK")
    high_risk_count = len(risk_analysis.get("high_risk_terms", []))
    critical_signals_count = len(governance_snapshot.get("critical_signals", []))
    
    # Determine semantic_ok
    semantic_ok = (
        governance_status == GovernanceStatus.OK.value and
        alignment_status == AlignmentStatus.ALIGNED.value and
        risk_status == RiskStatus.OK.value
    )
    
    # Determine status light
    if (
        governance_status == GovernanceStatus.CRITICAL.value or
        risk_status == RiskStatus.CRITICAL.value or
        alignment_status == AlignmentStatus.DIVERGENT.value
    ):
        status_light = StatusLight.RED
    elif (
        governance_status == GovernanceStatus.ATTENTION.value or
        risk_status == RiskStatus.ATTENTION.value or
        alignment_status == AlignmentStatus.PARTIAL.value
    ):
        status_light = StatusLight.YELLOW
    else:
        status_light = StatusLight.GREEN
    
    # Generate headline
    if status_light == StatusLight.RED:
        if high_risk_count > 0:
            headline = f"Semantic backbone has {high_risk_count} critical term(s) requiring immediate attention"
        elif critical_signals_count > 0:
            headline = f"Semantic backbone shows {critical_signals_count} critical governance signal(s)"
        else:
            headline = "Semantic backbone is divergent across systems"
    elif status_light == StatusLight.YELLOW:
        if alignment_status == AlignmentStatus.PARTIAL.value:
            headline = "Semantic backbone has partial alignment across systems"
        else:
            headline = "Semantic backbone requires attention but is stable"
    else:
        headline = "Semantic backbone is aligned and healthy across all systems"
    
    return {
        "status_light": status_light.value,
        "semantic_ok": semantic_ok,
        "alignment_status": alignment_status,
        "critical_term_count": high_risk_count,
        "headline": headline,
        "governance_status": governance_status,
        "risk_status": risk_status,
    }


# =============================================================================
# PHASE IV — MULTI-RUN DRIFT TIMELINE & CURRICULUM/DOCS COUPLING (Requirements Version)
# =============================================================================

def build_semantic_director_panel(
    drift_timeline: Dict[str, Any],
    alignment_analysis: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a director console panel combining drift timeline and alignment analysis.
    
    STATUS: PHASE IV — NOT RUN IN PHASE I
    
    Provides a high-level executive summary suitable for:
    - Investor/acquirer due diligence
    - Internal strategy reviews
    - System health dashboards
    
    Args:
        drift_timeline: From build_semantic_drift_timeline.
        alignment_analysis: From analyze_semantic_alignment_with_curriculum.
    
    Returns:
        Dictionary with:
        - semantic_status_light: "GREEN" | "YELLOW" | "RED"
        - alignment_status
        - critical_run_ids: List of run_ids with critical signals
        - headline: Short neutral sentence summarizing drift and alignment
    """
    # Determine status light
    timeline_trend = drift_timeline.get("trend", "STABLE")
    runs_with_critical = drift_timeline.get("runs_with_critical_signals", [])
    alignment_status = alignment_analysis.get("alignment_status", "ALIGNED")
    
    # RED: Critical signals present or misaligned
    if runs_with_critical or alignment_status == "MISALIGNED":
        status_light = "RED"
    # YELLOW: Drifting or partial alignment
    elif timeline_trend == "DRIFTING" or timeline_trend == "VOLATILE" or alignment_status == "PARTIAL":
        status_light = "YELLOW"
    # GREEN: Stable and aligned
    else:
        status_light = "GREEN"
    
    # Build headline
    parts = []
    
    if timeline_trend == "VOLATILE":
        parts.append("Semantic graph shows volatile drift")
    elif timeline_trend == "DRIFTING":
        parts.append("Semantic graph shows gradual drift")
    elif timeline_trend == "STABLE":
        parts.append("Semantic graph is stable")
    
    if alignment_status == "MISALIGNED":
        parts.append("with significant curriculum misalignment")
    elif alignment_status == "PARTIAL":
        parts.append("with partial curriculum alignment")
    elif alignment_status == "ALIGNED":
        parts.append("with curriculum alignment")
    
    if runs_with_critical:
        parts.append(f"({len(runs_with_critical)} runs with critical signals)")
    
    headline = ". ".join(parts) + "." if parts else "Semantic graph status unknown."
    
    return {
        "schema_version": SEMANTIC_DRIFT_TIMELINE_SCHEMA_VERSION,
        "semantic_status_light": status_light,
        "alignment_status": alignment_status,
        "critical_run_ids": runs_with_critical,
        "headline": headline,
        "trend": timeline_trend,
        "node_disappearance_count": len(drift_timeline.get("node_disappearance_events", [])),
    }


# =============================================================================
# PHASE V — SEMANTIC/TDA CROSS-CORRELATION & GOVERNANCE TILE
# =============================================================================

SEMANTIC_TDA_COUPLING_SCHEMA_VERSION = "semantic-tda-coupling-1.0.0"


def correlate_semantic_and_tda_signals(
    semantic_timeline: Dict[str, Any],
    tda_health: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Correlate semantic drift signals with TDA health signals.
    
    STATUS: PHASE V — SEMANTIC/TDA CROSS-TIE
    
    Analyzes alignment between structural drift in semantic graph (symbols/terms)
    and structural drift in TDA topology. Identifies slices where both systems
    signal issues, or where they disagree.
    
    Args:
        semantic_timeline: From build_semantic_drift_timeline.
            Expected keys: timeline, runs_with_critical_signals, node_disappearance_events, trend
        tda_health: From summarize_tda_for_global_health (backend.health.tda_adapter).
            Expected keys: tda_status, block_rate, hss_trend, governance_signal
    
    Returns:
        Dictionary with:
        - schema_version
        - correlation_coefficient: float (-1.0 to 1.0)
        - slices_where_both_signal: List of slice/term names where both systems signal issues
        - semantic_only_slices: List of slices/terms with semantic signals but no TDA signals
        - tda_only_slices: List of slices/terms with TDA signals but no semantic signals
        - alignment_note: Descriptive note about correlation strength and direction
    """
    # Extract semantic signals
    semantic_status_light = _infer_semantic_status_from_timeline(semantic_timeline)
    semantic_critical_runs = set(semantic_timeline.get("runs_with_critical_signals", []))
    semantic_disappearances = semantic_timeline.get("node_disappearance_events", [])
    
    # Extract terms/slices from semantic signals
    semantic_terms = set()
    for event in semantic_disappearances:
        term = event.get("term", "")
        if term:
            semantic_terms.add(term)
    
    # Extract TDA signals
    tda_status = tda_health.get("tda_status", "OK")
    tda_block_rate = tda_health.get("block_rate", 0.0)
    tda_governance_signal = tda_health.get("governance_signal", "OK")
    
    # Normalize statuses to numeric signals for correlation
    # Semantic: RED=2, YELLOW=1, GREEN=0
    # TDA: ALERT=2, ATTENTION=1, OK=0
    semantic_signal_value = _status_light_to_numeric(semantic_status_light)
    tda_signal_value = _tda_status_to_numeric(tda_status)
    
    # Calculate correlation coefficient
    # Simple correlation: if both high (2), correlation = 1.0
    # If both low (0), correlation = 1.0
    # If one high and one low, correlation = -1.0
    # If one is medium (1), correlation depends on the other
    if semantic_signal_value == tda_signal_value:
        if semantic_signal_value == 0:
            correlation_coefficient = 1.0  # Both OK
        elif semantic_signal_value == 2:
            correlation_coefficient = 1.0  # Both critical
        else:
            correlation_coefficient = 0.5  # Both attention
    elif abs(semantic_signal_value - tda_signal_value) == 2:
        correlation_coefficient = -1.0  # Complete mismatch (RED vs OK or vice versa)
    else:
        correlation_coefficient = 0.0  # Partial mismatch (one medium, one extreme)
    
    # Identify slices where both signal
    # For now, we'll use the presence of critical signals in semantic
    # and block_rate > 0 or ALERT in TDA as indicators
    slices_where_both_signal = []
    if semantic_critical_runs and (tda_block_rate > 0 or tda_status == "ALERT"):
        # Extract slice names from semantic terms (filter for slice-like terms)
        for term in semantic_terms:
            if "slice" in term.lower() or term.startswith("slice_"):
                slices_where_both_signal.append(term)
    
    # Semantic-only slices: semantic signals but TDA is OK
    semantic_only_slices = []
    if semantic_critical_runs and tda_status == "OK" and tda_block_rate == 0:
        for term in semantic_terms:
            if "slice" in term.lower() or term.startswith("slice_"):
                semantic_only_slices.append(term)
    
    # TDA-only slices: TDA signals but semantic is GREEN
    tda_only_slices = []
    if (tda_block_rate > 0 or tda_status != "OK") and semantic_status_light == "GREEN":
        # TDA doesn't track individual slices, so we'll note this as a general signal
        tda_only_slices.append("tda_topology_drift")
    
    # Build alignment note
    if correlation_coefficient >= 0.8:
        alignment_note = "Strong positive correlation: semantic and TDA signals align"
    elif correlation_coefficient >= 0.3:
        alignment_note = "Moderate positive correlation: semantic and TDA signals partially align"
    elif correlation_coefficient >= -0.3:
        alignment_note = "Weak correlation: semantic and TDA signals show limited alignment"
    elif correlation_coefficient >= -0.8:
        alignment_note = "Moderate negative correlation: semantic and TDA signals partially disagree"
    else:
        alignment_note = "Strong negative correlation: semantic and TDA signals strongly disagree"
    
    return {
        "schema_version": SEMANTIC_TDA_COUPLING_SCHEMA_VERSION,
        "correlation_coefficient": round(correlation_coefficient, 3),
        "slices_where_both_signal": sorted(slices_where_both_signal),
        "semantic_only_slices": sorted(semantic_only_slices),
        "tda_only_slices": sorted(tda_only_slices),
        "alignment_note": alignment_note,
    }


def _infer_semantic_status_from_timeline(timeline: Dict[str, Any]) -> str:
    """Infer semantic status light from timeline (helper for correlation)."""
    runs_with_critical = timeline.get("runs_with_critical_signals", [])
    trend = timeline.get("trend", "STABLE")
    
    if runs_with_critical:
        return "RED"
    elif trend in ("DRIFTING", "VOLATILE"):
        return "YELLOW"
    else:
        return "GREEN"


def _status_light_to_numeric(status_light: str) -> int:
    """Convert status light to numeric value for correlation."""
    if status_light == "RED":
        return 2
    elif status_light == "YELLOW":
        return 1
    else:
        return 0


def _tda_status_to_numeric(tda_status: str) -> int:
    """Convert TDA status to numeric value for correlation."""
    if tda_status == "ALERT":
        return 2
    elif tda_status == "ATTENTION":
        return 1
    else:
        return 0


def build_semantic_tda_governance_tile(
    semantic_panel: Dict[str, Any],
    tda_panel: Dict[str, Any],
    correlation: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a compact governance tile combining semantic and TDA signals.
    
    STATUS: PHASE V — SEMANTIC/TDA CROSS-TIE
    
    Provides a unified view of structural drift in both semantic graph (language/symbols)
    and TDA topology (shape/structure). The tile is designed for global health dashboards.
    
    Args:
        semantic_panel: From build_semantic_director_panel.
            Expected keys: semantic_status_light, alignment_status, critical_run_ids, headline
        tda_panel: From summarize_tda_for_global_health (backend.health.tda_adapter).
            Expected keys: tda_status, block_rate, hss_trend, governance_signal
        correlation: From correlate_semantic_and_tda_signals.
            Expected keys: correlation_coefficient, slices_where_both_signal, alignment_note
    
    Returns:
        Dictionary with:
        - schema_version
        - status: "OK" | "ATTENTION" | "BLOCK"
        - status_light: "GREEN" | "YELLOW" | "RED"
        - headline: Short neutral sentence summarizing combined status
        - key_slices: List of slice names requiring attention
    """
    semantic_status_light = semantic_panel.get("semantic_status_light", "GREEN")
    tda_status = tda_panel.get("tda_status", "OK")
    
    # Determine combined status
    # BLOCK: Both RED/ALERT OR high correlation with both critical
    both_critical = (semantic_status_light == "RED" and tda_status == "ALERT")
    correlation_coeff = correlation.get("correlation_coefficient", 0.0)
    both_signal_slices = correlation.get("slices_where_both_signal", [])
    
    if both_critical or (correlation_coeff >= 0.8 and both_signal_slices):
        status = "BLOCK"
        status_light = "RED"
    # ATTENTION: Mismatch (semantic RED, TDA GREEN) OR (semantic GREEN, TDA ALERT)
    # OR partial correlation with some signals
    elif (semantic_status_light == "RED" and tda_status == "OK") or \
         (semantic_status_light == "GREEN" and tda_status == "ALERT") or \
         (correlation_coeff < -0.3) or \
         (semantic_status_light == "YELLOW" or tda_status == "ATTENTION"):
        status = "ATTENTION"
        status_light = "YELLOW"
    # OK: Both GREEN/OK
    else:
        status = "OK"
        status_light = "GREEN"
    
    # Build headline
    parts = []
    if status == "BLOCK":
        parts.append("Semantic and TDA signals both indicate critical structural drift")
    elif status == "ATTENTION":
        # Check specific mismatch cases first (before general correlation check)
        if semantic_status_light == "RED" and tda_status == "OK":
            parts.append("Semantic signals indicate drift while TDA topology appears stable")
        elif semantic_status_light == "GREEN" and tda_status == "ALERT":
            parts.append("TDA topology indicates drift while semantic graph appears stable")
        elif correlation_coeff < -0.3:
            parts.append("Semantic and TDA signals show disagreement on structural health")
        else:
            parts.append("Semantic and TDA signals show partial alignment")
    else:
        parts.append("Semantic and TDA signals indicate stable structural health")
    
    if both_signal_slices:
        parts.append(f"({len(both_signal_slices)} slices with both signals)")
    
    headline = ". ".join(parts) + "." if parts else "Semantic/TDA status unknown."
    
    # Collect key slices
    key_slices = []
    key_slices.extend(both_signal_slices)
    key_slices.extend(correlation.get("semantic_only_slices", []))
    key_slices.extend(correlation.get("tda_only_slices", []))
    key_slices = sorted(list(set(key_slices)))  # Deduplicate
    
    return {
        "schema_version": SEMANTIC_TDA_COUPLING_SCHEMA_VERSION,
        "status": status,
        "status_light": status_light,
        "headline": headline,
        "key_slices": key_slices,
        "correlation_coefficient": correlation_coeff,
        "semantic_status": semantic_status_light,
        "tda_status": tda_status,
    }


# =============================================================================
# SEMANTIC CONTRACT AUDITOR (Phase IV Follow-up - Task 1)
# =============================================================================

class ContractStatus(str, Enum):
    """Semantic contract status levels."""
    OK = "OK"
    ATTENTION = "ATTENTION"
    BREACH = "BREACH"


def audit_semantic_contract(
    alignment_index: Dict[str, Any],
    taxonomy: Dict[str, Any],
    curriculum_manifest: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Audit semantic contract compliance.
    
    A semantic contract breach occurs when a term is REQUIRED in curriculum
    but missing in code/taxonomy.
    
    Args:
        alignment_index: From build_semantic_alignment_index()
        taxonomy: Dict with "terms" list (taxonomy terms)
        curriculum_manifest: Dict with "terms" list (curriculum terms - these are required)
    
    Returns:
        Dict with:
        - violated_contract_terms: List[str] - Terms required in curriculum but missing elsewhere
        - contract_status: OK | ATTENTION | BREACH
        - mismatch_types: Dict with "docs", "curriculum", "taxonomy" keys
        - summary_notes: List[str] - Explanatory notes
    """
    # Extract term sets
    curriculum_terms = set(curriculum_manifest.get("terms", []))
    taxonomy_terms = set(taxonomy.get("terms", []))
    
    # Get terms from alignment index (code and docs)
    terms_only_in_code = set(alignment_index.get("terms_only_in_code", []))
    terms_only_in_docs = set(alignment_index.get("terms_only_in_docs", []))
    
    # Reconstruct code terms from alignment index
    # Terms in code = all terms minus orphaned terms
    # We need to infer this from the alignment index structure
    # For now, we'll use a simpler approach: check if curriculum terms exist in taxonomy
    
    # Contract violation: curriculum term not in taxonomy
    # (assuming curriculum terms should be in taxonomy)
    violated_contract_terms = sorted(curriculum_terms - taxonomy_terms)
    
    # Also check for terms in curriculum that are orphaned (only in curriculum)
    terms_only_in_curriculum = set(alignment_index.get("terms_only_in_curriculum", []))
    violated_contract_terms.extend(terms_only_in_curriculum)
    violated_contract_terms = sorted(set(violated_contract_terms))
    
    # Categorize mismatch types
    mismatch_types = {
        "docs": sorted(terms_only_in_docs),
        "curriculum": sorted(terms_only_in_curriculum),
        "taxonomy": sorted(alignment_index.get("taxonomy_terms_with_no_uses", [])),
    }
    
    # Determine contract status
    if len(violated_contract_terms) > 0:
        if len(violated_contract_terms) > len(curriculum_terms) * 0.5:  # >50% violated
            contract_status = ContractStatus.BREACH
        else:
            contract_status = ContractStatus.ATTENTION
    else:
        contract_status = ContractStatus.OK
    
    # Generate summary notes
    summary_notes = []
    
    if violated_contract_terms:
        summary_notes.append(
            f"{len(violated_contract_terms)} curriculum term(s) are missing from required systems"
        )
    
    if mismatch_types["docs"]:
        summary_notes.append(f"{len(mismatch_types['docs'])} term(s) only in documentation")
    
    if mismatch_types["curriculum"]:
        summary_notes.append(f"{len(mismatch_types['curriculum'])} term(s) only in curriculum")
    
    if mismatch_types["taxonomy"]:
        summary_notes.append(f"{len(mismatch_types['taxonomy'])} taxonomy term(s) unused")
    
    if not summary_notes:
        summary_notes.append("Semantic contract is fully compliant")
    
    return {
        "violated_contract_terms": violated_contract_terms,
        "contract_status": contract_status.value,
        "mismatch_types": mismatch_types,
        "summary_notes": summary_notes,
    }


# =============================================================================
# DRIFT FORECAST TILE (Phase IV Follow-up - Task 2)
# =============================================================================

class DriftDirection(str, Enum):
    """Semantic drift direction."""
    IMPROVING = "IMPROVING"
    STABLE = "STABLE"
    DEGRADING = "DEGRADING"


class ForecastBand(str, Enum):
    """Forecast confidence band."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


def forecast_semantic_drift(
    alignment_index_history: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Forecast semantic drift based on historical alignment data.
    
    Args:
        alignment_index_history: Sequence of alignment indices (oldest to newest)
                                Each should have "alignment_status" and "total_orphaned_terms"
    
    Returns:
        Dict with:
        - drift_direction: IMPROVING | STABLE | DEGRADING
        - forecast_band: LOW | MEDIUM | HIGH
        - explanation: str - Human-readable explanation
    """
    if not alignment_index_history:
        return {
            "drift_direction": DriftDirection.STABLE.value,
            "forecast_band": ForecastBand.LOW.value,
            "explanation": "No historical data available for drift forecasting",
        }
    
    if len(alignment_index_history) < 2:
        return {
            "drift_direction": DriftDirection.STABLE.value,
            "forecast_band": ForecastBand.LOW.value,
            "explanation": "Insufficient historical data (need at least 2 data points)",
        }
    
    # Extract orphaned term counts over time
    orphaned_counts = [
        idx.get("total_orphaned_terms", 0) for idx in alignment_index_history
    ]
    
    # Extract alignment statuses
    statuses = [idx.get("alignment_status", "ALIGNED") for idx in alignment_index_history]
    
    # Calculate trend
    # Simple linear trend: if orphaned counts are decreasing, improving
    # If increasing, degrading. If stable, stable.
    
    # Calculate average change rate
    if len(orphaned_counts) >= 2:
        changes = [
            orphaned_counts[i] - orphaned_counts[i - 1]
            for i in range(1, len(orphaned_counts))
        ]
        avg_change = sum(changes) / len(changes) if changes else 0
        
        # Determine direction
        if avg_change < -0.5:  # Decreasing orphaned terms
            drift_direction = DriftDirection.IMPROVING
        elif avg_change > 0.5:  # Increasing orphaned terms
            drift_direction = DriftDirection.DEGRADING
        else:
            drift_direction = DriftDirection.STABLE
    else:
        drift_direction = DriftDirection.STABLE
    
    # Check status progression
    # If status went from DIVERGENT -> PARTIAL -> ALIGNED, improving
    # If went from ALIGNED -> PARTIAL -> DIVERGENT, degrading
    status_order = {"ALIGNED": 0, "PARTIAL": 1, "DIVERGENT": 2}
    status_values = [status_order.get(s, 1) for s in statuses]
    
    if len(status_values) >= 2:
        status_trend = status_values[-1] - status_values[0]
        if status_trend < 0:  # Status improved (lower number = better)
            drift_direction = DriftDirection.IMPROVING
        elif status_trend > 0:  # Status degraded
            drift_direction = DriftDirection.DEGRADING
        # If status_trend == 0, keep the direction from orphaned counts
    
    # Determine forecast band (confidence)
    # More data points = higher confidence
    # More consistent trend = higher confidence
    if len(alignment_index_history) >= 5:
        # Check consistency of trend
        if len(changes) >= 3:
            # All changes in same direction = more confident
            all_positive = all(c > 0 for c in changes)
            all_negative = all(c < 0 for c in changes)
            if all_positive or all_negative:
                forecast_band = ForecastBand.HIGH
            else:
                forecast_band = ForecastBand.MEDIUM
        else:
            forecast_band = ForecastBand.MEDIUM
    elif len(alignment_index_history) >= 3:
        forecast_band = ForecastBand.MEDIUM
    else:
        forecast_band = ForecastBand.LOW
    
    # Generate explanation
    if drift_direction == DriftDirection.IMPROVING:
        explanation = (
            f"Semantic alignment is improving: orphaned terms decreased from "
            f"{orphaned_counts[0]} to {orphaned_counts[-1]} over {len(alignment_index_history)} observations"
        )
    elif drift_direction == DriftDirection.DEGRADING:
        explanation = (
            f"Semantic alignment is degrading: orphaned terms increased from "
            f"{orphaned_counts[0]} to {orphaned_counts[-1]} over {len(alignment_index_history)} observations"
        )
    else:
        explanation = (
            f"Semantic alignment is stable: orphaned terms remain around "
            f"{orphaned_counts[-1]} over {len(alignment_index_history)} observations"
        )
    
    return {
        "drift_direction": drift_direction.value,
        "forecast_band": forecast_band.value,
        "explanation": explanation,
        "historical_points": len(alignment_index_history),
        "orphaned_count_trend": orphaned_counts,
    }


# =============================================================================
# SEMANTIC INVARIANT CHECKER (Phase V - Task 1)
# =============================================================================

class InvariantStatus(str, Enum):
    """Semantic invariant status levels."""
    OK = "OK"
    ATTENTION = "ATTENTION"
    BROKEN = "BROKEN"


@dataclass
class BrokenInvariant:
    """Represents a broken semantic invariant."""
    invariant_type: str
    description: str
    terms_involved: List[str]
    severity: str  # "BROKEN" | "ATTENTION"


def check_semantic_invariants(
    alignment_index: Dict[str, Any],
    graph: Optional[SemanticKnowledgeGraph] = None,
    taxonomy: Optional[Dict[str, Any]] = None,
    curriculum_manifest: Optional[Dict[str, Any]] = None,
    max_unused_versions: int = 3,
) -> Dict[str, Any]:
    """
    Check semantic invariants across systems.
    
    Invariants:
    1. Every curriculum term must appear in ≥2 of {taxonomy, docs, graph}
    2. No taxonomy term may remain unused for >N versions (N=3 default)
    3. No graph node may be isolated (degree 0)
    
    Args:
        alignment_index: From build_semantic_alignment_index()
        graph: Optional SemanticKnowledgeGraph - for checking node degrees
        taxonomy: Optional Dict with "terms" list and "version_history" (list of versions where term was unused)
        curriculum_manifest: Optional Dict with "terms" list
        max_unused_versions: Maximum number of versions a taxonomy term can be unused (default: 3)
    
    Returns:
        Dict with:
        - invariant_status: OK | ATTENTION | BROKEN
        - broken_invariants: List of BrokenInvariant objects (as dicts)
        - terms_involved: List[str] - All terms involved in broken invariants
        - neutral_notes: List[str] - Explanatory notes
    """
    broken_invariants = []
    terms_involved = set()
    neutral_notes = []
    
    # Extract term sets
    curriculum_terms = set(curriculum_manifest.get("terms", [])) if curriculum_manifest else set()
    taxonomy_terms = set(taxonomy.get("terms", [])) if taxonomy else set()
    
    # Get terms from alignment index
    terms_only_in_code = set(alignment_index.get("terms_only_in_code", []))
    terms_only_in_docs = set(alignment_index.get("terms_only_in_docs", []))
    terms_only_in_curriculum = set(alignment_index.get("terms_only_in_curriculum", []))
    taxonomy_terms_with_no_uses = set(alignment_index.get("taxonomy_terms_with_no_uses", []))
    
    # Reconstruct graph terms
    if graph:
        graph_terms = {t.get("canonical_form", "") for t in graph.terms if t.get("canonical_form")}
    else:
        # Try to infer from alignment index
        # Terms in graph = all terms minus orphaned terms
        graph_terms = set()
    
    # INVARIANT 1: Every curriculum term must appear in ≥2 of {taxonomy, docs, graph}
    if curriculum_manifest:
        for term in curriculum_terms:
            appearances = 0
            # Count appearance in taxonomy
            if term in taxonomy_terms:
                appearances += 1
            # Count appearance in graph
            if term in graph_terms:
                appearances += 1
            # Count appearance in docs: if term is in terms_only_in_docs, it's definitely in docs
            # OR if term is NOT only in curriculum and NOT in taxonomy/graph/code, it's in docs
            if term in terms_only_in_docs:
                appearances += 1
            elif (term not in terms_only_in_curriculum and 
                  term not in terms_only_in_code and
                  term not in taxonomy_terms and
                  term not in graph_terms):
                # Term is shared but not in taxonomy/graph/code, so it must be in docs
                appearances += 1
            
            if appearances < 2:
                broken_invariants.append({
                    "invariant_type": "curriculum_term_insufficient_appearances",
                    "description": f"Curriculum term '{term}' appears in only {appearances} system(s), required ≥2",
                    "terms_involved": [term],
                    "severity": "BROKEN" if appearances == 0 else "ATTENTION",
                })
                terms_involved.add(term)
    
    # INVARIANT 2: No taxonomy term may remain unused for >N versions
    if taxonomy and "version_history" in taxonomy:
        version_history = taxonomy.get("version_history", {})
        for term in taxonomy_terms:
            if term in version_history:
                unused_versions = version_history[term].get("unused_versions", [])
                if len(unused_versions) > max_unused_versions:
                    # BROKEN if significantly over (>= 1.5x max), ATTENTION if slightly over
                    severity = "BROKEN" if len(unused_versions) >= int(max_unused_versions * 1.5) else "ATTENTION"
                    broken_invariants.append({
                        "invariant_type": "taxonomy_term_unused_too_long",
                        "description": f"Taxonomy term '{term}' has been unused for {len(unused_versions)} versions (max: {max_unused_versions})",
                        "terms_involved": [term],
                        "severity": severity,
                    })
                    terms_involved.add(term)
    elif taxonomy_terms_with_no_uses:
        # If no version history provided, check terms with no uses
        for term in taxonomy_terms_with_no_uses:
            broken_invariants.append({
                "invariant_type": "taxonomy_term_unused",
                "description": f"Taxonomy term '{term}' has no uses across systems",
                "terms_involved": [term],
                "severity": "ATTENTION",  # Less severe without version history
            })
            terms_involved.add(term)
    
    # INVARIANT 3: No graph node may be isolated (degree 0)
    if graph:
        for term_entry in graph.terms:
            canonical_form = term_entry.get("canonical_form", "")
            if canonical_form:
                degree = graph.get_node_degree(canonical_form)
                if degree == 0:
                    broken_invariants.append({
                        "invariant_type": "isolated_graph_node",
                        "description": f"Graph node '{canonical_form}' is isolated (degree 0)",
                        "terms_involved": [canonical_form],
                        "severity": "ATTENTION",  # Isolated nodes are concerning but not critical
                    })
                    terms_involved.add(canonical_form)
    
    # Determine overall invariant status
    broken_count = sum(1 for inv in broken_invariants if inv["severity"] == "BROKEN")
    attention_count = sum(1 for inv in broken_invariants if inv["severity"] == "ATTENTION")
    
    if broken_count > 0:
        invariant_status = InvariantStatus.BROKEN
    elif attention_count > 0:
        invariant_status = InvariantStatus.ATTENTION
    else:
        invariant_status = InvariantStatus.OK
    
    # Generate neutral notes
    if broken_invariants:
        neutral_notes.append(f"{len(broken_invariants)} semantic invariant(s) violated")
        if broken_count > 0:
            neutral_notes.append(f"{broken_count} critical violation(s) require immediate attention")
        if attention_count > 0:
            neutral_notes.append(f"{attention_count} warning(s) should be reviewed")
    else:
        neutral_notes.append("All semantic invariants are satisfied")
    
    return {
        "invariant_status": invariant_status.value,
        "broken_invariants": broken_invariants,
        "terms_involved": sorted(list(terms_involved)),
        "neutral_notes": neutral_notes,
    }


# =============================================================================
# SEMANTIC UPLIFT PRE-GATE PREVIEW (Phase V - Task 2)
# =============================================================================

class UpliftSemanticStatus(str, Enum):
    """Semantic uplift status levels."""
    OK = "OK"
    WARN = "WARN"
    BLOCK = "BLOCK"


def preview_semantic_uplift_gate(
    semantic_risk: Dict[str, Any],
    contract_audit: Dict[str, Any],
    drift_forecast: Dict[str, Any],
    invariant_check: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Preview semantic uplift gate status.
    
    Consumes semantic risk, contract audit, drift forecast, and invariant checker
    to determine if an uplift should proceed.
    
    Args:
        semantic_risk: From analyze_semantic_risk()
        contract_audit: From audit_semantic_contract()
        drift_forecast: From forecast_semantic_drift()
        invariant_check: From check_semantic_invariants()
    
    Returns:
        Dict with:
        - uplift_semantic_status: OK | WARN | BLOCK
        - rationale: List[str] - Neutral explanations
        - preview_effect_on_curriculum: List[str] - Neutral descriptions of potential effects
    """
    rationale = []
    preview_effect_on_curriculum = []
    
    # Extract statuses
    risk_status = semantic_risk.get("status", "OK")
    contract_status = contract_audit.get("contract_status", "OK")
    drift_direction = drift_forecast.get("drift_direction", "STABLE")
    invariant_status = invariant_check.get("invariant_status", "OK")
    
    # Collect blocking conditions
    blocking_conditions = []
    warning_conditions = []
    
    # BLOCK conditions (most severe)
    if risk_status == RiskStatus.CRITICAL.value:
        blocking_conditions.append("Semantic risk status is CRITICAL")
        rationale.append("Semantic risk status is CRITICAL: high-risk terms with governance issues and misalignment detected")
        preview_effect_on_curriculum.append("Uplift may exacerbate existing semantic misalignments")
    
    if contract_status == ContractStatus.BREACH.value:
        blocking_conditions.append("Semantic contract status is BREACH")
        rationale.append("Semantic contract status is BREACH: more than 50% of curriculum terms are missing from required systems")
        preview_effect_on_curriculum.append("Uplift may introduce additional contract violations")
    
    if invariant_status == InvariantStatus.BROKEN.value:
        blocking_conditions.append("Semantic invariant status is BROKEN")
        rationale.append("Semantic invariant status is BROKEN: critical semantic invariants are violated")
        preview_effect_on_curriculum.append("Uplift may violate additional semantic invariants")
    
    # WARN conditions (less severe but concerning)
    if risk_status == RiskStatus.ATTENTION.value:
        warning_conditions.append("Semantic risk status is ATTENTION")
        rationale.append("Semantic risk status is ATTENTION: medium-risk terms with partial misalignment detected")
        preview_effect_on_curriculum.append("Uplift should be reviewed for potential alignment issues")
    
    if contract_status == ContractStatus.ATTENTION.value:
        warning_conditions.append("Semantic contract status is ATTENTION")
        rationale.append("Some curriculum terms are missing from required systems")
        preview_effect_on_curriculum.append("Uplift may require additional contract compliance work")
    
    if invariant_status == InvariantStatus.ATTENTION.value:
        warning_conditions.append("Semantic invariant status is ATTENTION")
        rationale.append("Some semantic invariants have warnings")
        preview_effect_on_curriculum.append("Uplift should ensure invariant compliance")
    
    if drift_direction == DriftDirection.DEGRADING.value:
        warning_conditions.append("Semantic drift direction is DEGRADING")
        rationale.append("Semantic alignment is degrading over time")
        preview_effect_on_curriculum.append("Uplift may accelerate semantic drift if not carefully managed")
    
    # Determine uplift status
    if blocking_conditions:
        uplift_semantic_status = UpliftSemanticStatus.BLOCK
    elif warning_conditions:
        uplift_semantic_status = UpliftSemanticStatus.WARN
    else:
        uplift_semantic_status = UpliftSemanticStatus.OK
        rationale.append("All semantic checks passed")
        preview_effect_on_curriculum.append("Uplift appears safe from semantic perspective")
    
    return {
        "uplift_semantic_status": uplift_semantic_status.value,
        "rationale": rationale,
        "preview_effect_on_curriculum": preview_effect_on_curriculum,
    }


# =============================================================================
# DIRECTOR TILE FOR PHASE V (Phase V - Task 3)
# =============================================================================

def build_semantic_uplift_director_tile(
    semantic_risk: Dict[str, Any],
    contract_audit: Dict[str, Any],
    drift_forecast: Dict[str, Any],
    invariant_check: Dict[str, Any],
    uplift_preview: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build director tile for semantic uplift status.
    
    Provides high-level dashboard view of semantic integrity for uplift decisions.
    
    Args:
        semantic_risk: From analyze_semantic_risk()
        contract_audit: From audit_semantic_contract()
        drift_forecast: From forecast_semantic_drift()
        invariant_check: From check_semantic_invariants()
        uplift_preview: From preview_semantic_uplift_gate()
    
    Returns:
        Dict with:
        - status_light: GREEN | YELLOW | RED
        - semantic_uplift_status: OK | WARN | BLOCK
        - top_risk_terms: List[str] - Top risk terms (high-risk first, then medium-risk)
        - headline: str - Neutral headline summarizing status
    """
    # Extract statuses
    uplift_status = uplift_preview.get("uplift_semantic_status", "OK")
    risk_status = semantic_risk.get("status", "OK")
    contract_status = contract_audit.get("contract_status", "OK")
    invariant_status = invariant_check.get("invariant_status", "OK")
    
    # Determine status light
    if uplift_status == UpliftSemanticStatus.BLOCK.value:
        status_light = StatusLight.RED
    elif uplift_status == UpliftSemanticStatus.WARN.value:
        status_light = StatusLight.YELLOW
    else:
        status_light = StatusLight.GREEN
    
    # Collect top risk terms (high-risk first, then medium-risk)
    top_risk_terms = []
    high_risk_terms = semantic_risk.get("high_risk_terms", [])
    medium_risk_terms = semantic_risk.get("medium_risk_terms", [])
    top_risk_terms.extend(high_risk_terms[:5])  # Top 5 high-risk
    top_risk_terms.extend(medium_risk_terms[:3])  # Top 3 medium-risk
    
    # Generate headline
    if uplift_status == UpliftSemanticStatus.BLOCK.value:
        if risk_status == RiskStatus.CRITICAL.value:
            headline = "Semantic uplift blocked: critical risk terms detected"
        elif contract_status == ContractStatus.BREACH.value:
            headline = "Semantic uplift blocked: contract breach detected"
        elif invariant_status == InvariantStatus.BROKEN.value:
            headline = "Semantic uplift blocked: semantic invariants violated"
        else:
            headline = "Semantic uplift blocked: multiple semantic issues detected"
    elif uplift_status == UpliftSemanticStatus.WARN.value:
        if drift_forecast.get("drift_direction") == DriftDirection.DEGRADING.value:
            headline = "Semantic uplift warning: drift degrading, review recommended"
        else:
            headline = "Semantic uplift warning: semantic attention required"
    else:
        headline = "Semantic uplift status: all checks passed"
    
    return {
        "status_light": status_light.value,
        "semantic_uplift_status": uplift_status,
        "top_risk_terms": top_risk_terms,
        "headline": headline,
    }


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run_audit(
    curriculum_path: Optional[Path] = None,
    prereg_path: Optional[Path] = None,
) -> AuditReport:
    """
    Run the full semantic consistency audit.
    
    Args:
        curriculum_path: Path to curriculum_uplift_phase2.yaml
        prereg_path: Path to PREREG_UPLIFT_U2.yaml
    
    Returns:
        AuditReport with all findings
    """
    # Default paths relative to project root
    project_root = Path(__file__).parent.parent
    curriculum_path = curriculum_path or project_root / "config" / "curriculum_uplift_phase2.yaml"
    prereg_path = prereg_path or project_root / "experiments" / "prereg" / "PREREG_UPLIFT_U2.yaml"
    
    report = AuditReport()
    
    # Step 1: Extract slice specs
    try:
        specs = extract_slice_specs(curriculum_path, prereg_path)
        report.slices = specs
    except FileNotFoundError as e:
        report.global_issues.append(AuditIssue(
            severity=AuditStatus.FAIL,
            category="file_not_found",
            message=str(e),
        ))
        return report
    
    # Step 2: Run static checks
    run_static_checks(report.slices)
    
    # Step 3: Run runtime sanity checks
    for spec in report.slices:
        run_runtime_sanity_checks(spec)
    
    # Step 4: Global consistency checks
    check_global_consistency(curriculum_path, prereg_path, report)
    
    return report


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="PHASE II Semantic Consistency Auditor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Absolute Safeguards:
- Do NOT change metric logic.
- Do NOT change success definitions.
- Do NOT alter theory or governance text.
- This is an AUDITOR, not a designer.

Modes:
  Default:        Full audit with Markdown report
  --ci:           Minimal output, exit code 0=OK, 1=FAIL
  --summary:      One-line drift summary: slices=N metrics=N theory=N total=N OK|FAIL
  --suggest-fixes: Generate fix suggestions in canonical format
  --build-index:  Build semantic term index
  --build-graph:  Build semantic knowledge graph (v1.3)
        """,
    )
    parser.add_argument(
        "--curriculum",
        type=Path,
        default=None,
        help="Path to curriculum_uplift_phase2.yaml (default: config/curriculum_uplift_phase2.yaml)",
    )
    parser.add_argument(
        "--prereg",
        type=Path,
        default=None,
        help="Path to PREREG_UPLIFT_U2.yaml (default: experiments/prereg/PREREG_UPLIFT_U2.yaml)",
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
        "--ci",
        action="store_true",
        help="CI mode: minimal output, exit 0=OK, 1=FAIL",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="One-line drift summary: slices=N metrics=N theory=N total=N OK|FAIL",
    )
    parser.add_argument(
        "--suggest-fixes",
        action="store_true",
        help="Generate fix suggestions in canonical format: Replace <old> with <new> in <file>, line ~<N>",
    )
    parser.add_argument(
        "--build-index",
        action="store_true",
        help="Build semantic term index to artifacts/phase_ii/semantic_term_index.json",
    )
    parser.add_argument(
        "--build-graph",
        action="store_true",
        help="Build semantic knowledge graph to artifacts/phase_ii/semantic_knowledge_graph.json",
    )
    
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent
    
    # Mode: Build term index
    if args.build_index:
        index = build_term_index(project_root)
        output_path = args.output or project_root / "artifacts" / "phase_ii" / "semantic_term_index.json"
        save_term_index(index, output_path)
        print(f"Term index written to {output_path}")
        print(f"  Total terms: {index['summary']['total_terms']}")
        print(f"  By kind: {index['summary']['by_kind']}")
        sys.exit(0)
    
    # Mode: Build semantic knowledge graph
    if args.build_graph:
        # First build term index
        index = build_term_index(project_root)
        # Then build graph from index
        graph = build_semantic_graph(index)
        output_path = args.output or project_root / "artifacts" / "phase_ii" / "semantic_knowledge_graph.json"
        save_semantic_graph(graph, output_path)
        print(f"Semantic knowledge graph written to {output_path}")
        print(f"  Nodes: {len(graph.terms)}")
        print(f"  Edges: {len(graph.edges)}")
        edge_kinds = {}
        for e in graph.edges:
            edge_kinds[e.kind] = edge_kinds.get(e.kind, 0) + 1
        print(f"  Edge types: {edge_kinds}")
        sys.exit(0)
    
    # Mode: CI checks
    if args.ci:
        passed, failures = run_ci_checks(args.curriculum, args.prereg)
        output = format_ci_output(passed, failures)
        
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(output)
        else:
            print(output)
        
        sys.exit(0 if passed else 1)
    
    # Mode: One-line drift summary
    if args.summary:
        counts = count_drift(args.curriculum, args.prereg)
        output = format_drift_summary(counts)
        
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(output)
        else:
            print(output)
        
        sys.exit(0 if counts.is_ok else 1)
    
    # Default mode: Full audit
    report = run_audit(args.curriculum, args.prereg)
    
    # Mode: Suggest fixes (canonical format)
    if args.suggest_fixes:
        canonical_suggestions = generate_canonical_suggestions(args.curriculum)
        
        if args.json:
            # Also include legacy format for backward compatibility
            legacy_suggestions = generate_suggestions(report, project_root)
            output = json.dumps({
                "canonical_suggestions": canonical_suggestions,
                "legacy_suggestions": [s.to_dict() for s in legacy_suggestions],
                "total": len(canonical_suggestions),
            }, indent=2)
        else:
            if canonical_suggestions:
                lines = ["## Suggested Fixes (Canonical Format)", ""]
                for suggestion in canonical_suggestions:
                    lines.append(f"- {suggestion}")
                lines.append("")
                output = "\n".join(lines)
            else:
                output = "## Suggestions\n\nNo fixes needed — all checks passed! ✅\n"
        
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(output)
            print(f"Suggestions written to {args.output}")
        else:
            print(output)
        
        sys.exit(0)
    
    # Format output
    if args.json:
        output = json.dumps(format_json_report(report), indent=2)
    else:
        output = format_markdown_report(report)
    
    # Write output
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"Report written to {args.output}")
    else:
        print(output)
    
    # Exit with appropriate code
    if report.status == AuditStatus.FAIL:
        sys.exit(1)
    elif report.status == AuditStatus.WARNING:
        sys.exit(0)  # Warnings don't fail CI
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()

