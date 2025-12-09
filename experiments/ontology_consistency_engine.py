"""
PHASE II — NOT USED IN PHASE I

Ontology Consistency Engine: Total Cross-File Coherence Monitoring.

This module provides multi-layer ontology consistency checking:
1. Metric Ontology Extractor - Extract and compare metric definitions from 3 sources
2. Bidirectional Schema Validation - doc↔YAML↔code cycle checking
3. Mathematical Checks - Detect unreachable thresholds and contradictions
4. CI Report Formatter - Machine-readable JSON output with recommendations

Constraints:
- Must not rewrite curriculum
- Must not adjust thresholds
- Must not modify metrics code

Artifacts Monitored:
- experiments/slice_success_metrics.py (CODE)
- docs/PHASE2_RFL_UPLIFT_PLAN.md (DOC)
- config/curriculum_uplift_phase2.yaml (YAML)
"""

from __future__ import annotations

import argparse
import ast
import inspect
import json
import re
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import yaml


# =============================================================================
# SEVERITY AND STATUS
# =============================================================================

class ConsistencyStatus(str, Enum):
    """CI-suitable status codes."""
    OK = "ok"
    WARNING = "warning"
    FAIL = "fail"


# =============================================================================
# ONTOLOGY DATA STRUCTURES
# =============================================================================

@dataclass
class MetricOntologyEntry:
    """
    Represents a metric's ontology extracted from a single source.
    """
    source: str  # "code", "doc", "yaml"
    metric_kind: str
    required_params: Set[str] = field(default_factory=set)
    optional_params: Set[str] = field(default_factory=set)
    success_criteria: str = ""
    description: str = ""
    function_name: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "metric_kind": self.metric_kind,
            "required_params": sorted(self.required_params),
            "optional_params": sorted(self.optional_params),
            "success_criteria": self.success_criteria,
            "description": self.description,
            "function_name": self.function_name,
        }


@dataclass
class OntologyDiff:
    """Represents a difference between two ontology views."""
    metric_kind: str
    source_a: str
    source_b: str
    diff_type: str  # "missing_param", "extra_param", "criteria_mismatch", etc.
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_kind": self.metric_kind,
            "source_a": self.source_a,
            "source_b": self.source_b,
            "diff_type": self.diff_type,
            "details": self.details,
        }


@dataclass
class MathematicalIssue:
    """Represents a mathematical inconsistency."""
    slice_name: str
    issue_type: str  # "unreachable_threshold", "contradictory_params", "degenerate_criteria"
    severity: ConsistencyStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    recommendation: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "slice_name": self.slice_name,
            "issue_type": self.issue_type,
            "severity": self.severity.value,
            "message": self.message,
            "details": self.details,
            "recommendation": self.recommendation,
        }


@dataclass
class SchemaValidationResult:
    """Result of bidirectional schema validation."""
    direction: str  # "doc→yaml", "yaml→code", "code→doc"
    slice_name: str
    status: ConsistencyStatus
    diffs: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "direction": self.direction,
            "slice_name": self.slice_name,
            "status": self.status.value,
            "diffs": self.diffs,
        }


@dataclass
class OntologyReport:
    """Complete ontology consistency report."""
    status: ConsistencyStatus = ConsistencyStatus.OK
    metric_ontology: Dict[str, List[MetricOntologyEntry]] = field(default_factory=dict)
    metric_drift: List[OntologyDiff] = field(default_factory=list)
    slice_drift: List[SchemaValidationResult] = field(default_factory=list)
    formula_inconsistencies: List[Dict[str, Any]] = field(default_factory=list)
    mathematical_issues: List[MathematicalIssue] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    def compute_status(self) -> ConsistencyStatus:
        """Compute overall status from findings."""
        if any(m.severity == ConsistencyStatus.FAIL for m in self.mathematical_issues):
            return ConsistencyStatus.FAIL
        if any(s.status == ConsistencyStatus.FAIL for s in self.slice_drift):
            return ConsistencyStatus.FAIL
        if len(self.metric_drift) > 0:
            return ConsistencyStatus.WARNING
        if any(m.severity == ConsistencyStatus.WARNING for m in self.mathematical_issues):
            return ConsistencyStatus.WARNING
        return ConsistencyStatus.OK
    
    def to_ci_json(self) -> Dict[str, Any]:
        """Format for CI consumption."""
        self.status = self.compute_status()
        return {
            "status": self.status.value,
            "metric_drift": [d.to_dict() for d in self.metric_drift],
            "slice_drift": [s.to_dict() for s in self.slice_drift],
            "formula_inconsistencies": self.formula_inconsistencies,
            "mathematical_issues": [m.to_dict() for m in self.mathematical_issues],
            "recommendations": self.recommendations,
        }


# =============================================================================
# METRIC ONTOLOGY EXTRACTOR
# =============================================================================

def extract_ontology_from_code(metrics_path: Path) -> List[MetricOntologyEntry]:
    """
    Extract metric ontology from slice_success_metrics.py.
    
    Parses function signatures, docstrings, and parameter names.
    """
    entries = []
    
    if not metrics_path.exists():
        return entries
    
    # Import the module
    try:
        from experiments.slice_success_metrics import (
            compute_goal_hit,
            compute_sparse_success,
            compute_chain_success,
            compute_multi_goal_success,
        )
        
        functions = {
            "goal_hit": compute_goal_hit,
            "density": compute_sparse_success,
            "chain_length": compute_chain_success,
            "multi_goal": compute_multi_goal_success,
        }
        
        for metric_kind, func in functions.items():
            sig = inspect.signature(func)
            params = set(sig.parameters.keys())
            
            # Parse docstring for success criteria
            docstring = func.__doc__ or ""
            success_criteria = ""
            if "success" in docstring.lower():
                # Extract success-related lines
                for line in docstring.split("\n"):
                    if "success" in line.lower() or "returns" in line.lower():
                        success_criteria += line.strip() + " "
            
            entries.append(MetricOntologyEntry(
                source="code",
                metric_kind=metric_kind,
                required_params=params,
                optional_params=set(),
                success_criteria=success_criteria.strip(),
                description=docstring.split("\n")[0].strip() if docstring else "",
                function_name=func.__name__,
            ))
    
    except ImportError as e:
        # Return empty if import fails
        pass
    
    return entries


def extract_ontology_from_doc(doc_path: Path) -> List[MetricOntologyEntry]:
    """
    Extract metric ontology from PHASE2_RFL_UPLIFT_PLAN.md.
    
    Parses success metric code blocks and parameter descriptions.
    """
    entries = []
    
    if not doc_path.exists():
        return entries
    
    with open(doc_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Map slice names to metric kinds
    slice_metric_map = {
        "slice_uplift_goal": "goal_hit",
        "slice_uplift_sparse": "density",
        "slice_uplift_tree": "chain_length",
        "slice_uplift_dependency": "multi_goal",
    }
    
    # Parse each slice section
    for slice_name, metric_kind in slice_metric_map.items():
        # Find the section for this slice
        pattern = rf'###\s+\d+\.\s+`{slice_name}`.*?(?=###\s+\d+\.|## |$)'
        match = re.search(pattern, content, re.DOTALL)
        
        if not match:
            continue
        
        section = match.group(0)
        
        # Extract success metric code block
        success_pattern = r'\*\*Success Metric\*\*:\s*```([^`]+)```'
        success_match = re.search(success_pattern, section, re.DOTALL)
        success_criteria = success_match.group(1).strip() if success_match else ""
        
        # Extract parameters mentioned
        params = set()
        # Look for parameter patterns like "min_verified", "target_hashes", etc.
        param_patterns = [
            r'len\(verified\)\s*>=\s*(\d+)',
            r'verified\s*>=\s*(\d+)',
            r'TARGET_HASHES',
            r'REQUIRED_GOALS',
            r'proof_depth\([^)]+\)\s*>=\s*(\d+)',
        ]
        
        if "TARGET_HASHES" in section or "target_hash" in section.lower():
            params.add("target_hashes")
        if "REQUIRED_GOALS" in section:
            params.add("required_goal_hashes")
        if "verified >=" in section or "len(verified)" in section:
            params.add("min_verified")
        if "proof_depth" in section or "chain" in section.lower():
            params.add("min_chain_length")
        
        entries.append(MetricOntologyEntry(
            source="doc",
            metric_kind=metric_kind,
            required_params=params,
            success_criteria=success_criteria,
            description=f"From PHASE2_RFL_UPLIFT_PLAN.md section for {slice_name}",
        ))
    
    return entries


def extract_ontology_from_yaml(yaml_path: Path) -> List[MetricOntologyEntry]:
    """
    Extract metric ontology from curriculum_uplift_phase2.yaml.
    
    Parses success_metric definitions from each slice.
    """
    entries = []
    
    if not yaml_path.exists():
        return entries
    
    with open(yaml_path, "r", encoding="utf-8") as f:
        curriculum = yaml.safe_load(f)
    
    slices = curriculum.get("slices", {})
    
    for slice_name, slice_config in slices.items():
        if not isinstance(slice_config, dict):
            continue
        
        success_metric = slice_config.get("success_metric", {})
        if not isinstance(success_metric, dict):
            continue
        
        metric_kind = success_metric.get("kind", "")
        params = success_metric.get("parameters", {})
        
        if not metric_kind:
            continue
        
        entries.append(MetricOntologyEntry(
            source="yaml",
            metric_kind=metric_kind,
            required_params=set(params.keys()) if isinstance(params, dict) else set(),
            success_criteria=f"kind={metric_kind}, params={list(params.keys()) if params else []}",
            description=slice_config.get("description", "")[:100],
        ))
    
    return entries


def compare_ontologies(
    code_entries: List[MetricOntologyEntry],
    doc_entries: List[MetricOntologyEntry],
    yaml_entries: List[MetricOntologyEntry],
) -> List[OntologyDiff]:
    """
    Compare ontology entries across all three sources.
    """
    diffs = []
    
    # Build lookup by metric_kind
    code_by_kind = {e.metric_kind: e for e in code_entries}
    doc_by_kind = {e.metric_kind: e for e in doc_entries}
    yaml_by_kind = {e.metric_kind: e for e in yaml_entries}
    
    all_kinds = set(code_by_kind.keys()) | set(doc_by_kind.keys()) | set(yaml_by_kind.keys())
    
    for kind in all_kinds:
        code_entry = code_by_kind.get(kind)
        doc_entry = doc_by_kind.get(kind)
        yaml_entry = yaml_by_kind.get(kind)
        
        # Check code vs doc
        if code_entry and doc_entry:
            code_params = code_entry.required_params
            doc_params = doc_entry.required_params
            
            if code_params != doc_params and doc_params:  # Only if doc has params defined
                missing_in_code = doc_params - code_params
                extra_in_code = code_params - doc_params
                
                if missing_in_code:
                    diffs.append(OntologyDiff(
                        metric_kind=kind,
                        source_a="doc",
                        source_b="code",
                        diff_type="missing_param_in_code",
                        details={"missing": list(missing_in_code)},
                    ))
        
        # Check yaml vs code
        if yaml_entry and code_entry:
            yaml_params = yaml_entry.required_params
            # Note: code params include runtime args like verified_statements
            # YAML params are configuration params
            # This comparison may need adjustment
            pass
        
        # Check missing entries
        if code_entry and not doc_entry:
            diffs.append(OntologyDiff(
                metric_kind=kind,
                source_a="code",
                source_b="doc",
                diff_type="missing_in_doc",
                details={"message": f"Metric {kind} exists in code but not documented"},
            ))
        
        if code_entry and not yaml_entry:
            diffs.append(OntologyDiff(
                metric_kind=kind,
                source_a="code",
                source_b="yaml",
                diff_type="missing_in_yaml",
                details={"message": f"Metric {kind} exists in code but no YAML slices use it"},
            ))
    
    return diffs


# =============================================================================
# BIDIRECTIONAL SCHEMA VALIDATION
# =============================================================================

def validate_doc_to_yaml(
    doc_path: Path,
    yaml_path: Path,
) -> List[SchemaValidationResult]:
    """
    Validate doc → YAML direction.
    
    Check that everything documented exists in YAML.
    """
    results = []
    
    if not doc_path.exists() or not yaml_path.exists():
        return results
    
    with open(yaml_path, "r", encoding="utf-8") as f:
        curriculum = yaml.safe_load(f)
    
    slices = curriculum.get("slices", {})
    
    # Expected slices from documentation
    doc_slices = {
        "slice_uplift_goal": "goal_hit",
        "slice_uplift_sparse": "density",
        "slice_uplift_tree": "chain_length",
        "slice_uplift_dependency": "multi_goal",
    }
    
    for slice_name, expected_metric in doc_slices.items():
        diffs = []
        status = ConsistencyStatus.OK
        
        yaml_slice = slices.get(slice_name, {})
        
        if not yaml_slice:
            diffs.append({
                "type": "missing_slice",
                "message": f"Slice {slice_name} documented but not in YAML",
            })
            status = ConsistencyStatus.FAIL
        else:
            yaml_metric = yaml_slice.get("success_metric", {}).get("kind", "")
            if yaml_metric != expected_metric:
                diffs.append({
                    "type": "metric_mismatch",
                    "expected": expected_metric,
                    "actual": yaml_metric,
                })
                status = ConsistencyStatus.FAIL
        
        results.append(SchemaValidationResult(
            direction="doc→yaml",
            slice_name=slice_name,
            status=status,
            diffs=diffs,
        ))
    
    return results


def validate_yaml_to_code(
    yaml_path: Path,
) -> List[SchemaValidationResult]:
    """
    Validate YAML → code direction.
    
    Check that all metrics referenced in YAML have implementations.
    """
    results = []
    
    if not yaml_path.exists():
        return results
    
    with open(yaml_path, "r", encoding="utf-8") as f:
        curriculum = yaml.safe_load(f)
    
    # Get available functions
    try:
        from experiments.slice_success_metrics import (
            compute_goal_hit,
            compute_sparse_success,
            compute_chain_success,
            compute_multi_goal_success,
        )
        available_metrics = {"goal_hit", "density", "chain_length", "multi_goal"}
    except ImportError:
        available_metrics = set()
    
    slices = curriculum.get("slices", {})
    
    for slice_name, slice_config in slices.items():
        if not isinstance(slice_config, dict):
            continue
        
        success_metric = slice_config.get("success_metric", {})
        if not isinstance(success_metric, dict):
            continue
        
        metric_kind = success_metric.get("kind", "")
        diffs = []
        status = ConsistencyStatus.OK
        
        if metric_kind and metric_kind not in available_metrics:
            diffs.append({
                "type": "missing_implementation",
                "metric_kind": metric_kind,
                "message": f"No implementation found for metric '{metric_kind}'",
            })
            status = ConsistencyStatus.FAIL
        
        results.append(SchemaValidationResult(
            direction="yaml→code",
            slice_name=slice_name,
            status=status,
            diffs=diffs,
        ))
    
    return results


def validate_code_to_doc(
    doc_path: Path,
) -> List[SchemaValidationResult]:
    """
    Validate code → doc direction.
    
    Check that all implemented metrics are documented.
    """
    results = []
    
    # Get implemented metrics
    try:
        from experiments.slice_success_metrics import (
            compute_goal_hit,
            compute_sparse_success,
            compute_chain_success,
            compute_multi_goal_success,
        )
        implemented = {
            "goal_hit": "compute_goal_hit",
            "density": "compute_sparse_success",
            "chain_length": "compute_chain_success",
            "multi_goal": "compute_multi_goal_success",
        }
    except ImportError:
        return results
    
    # Check documentation
    if not doc_path.exists():
        for metric_kind in implemented:
            results.append(SchemaValidationResult(
                direction="code→doc",
                slice_name=f"metric_{metric_kind}",
                status=ConsistencyStatus.FAIL,
                diffs=[{"type": "missing_doc", "message": "Documentation file not found"}],
            ))
        return results
    
    with open(doc_path, "r", encoding="utf-8") as f:
        content = f.read().lower()
    
    for metric_kind, func_name in implemented.items():
        diffs = []
        status = ConsistencyStatus.OK
        
        # Check if metric is mentioned
        if metric_kind.replace("_", " ") not in content and metric_kind not in content:
            diffs.append({
                "type": "undocumented_metric",
                "metric_kind": metric_kind,
                "function": func_name,
            })
            status = ConsistencyStatus.WARNING
        
        results.append(SchemaValidationResult(
            direction="code→doc",
            slice_name=f"metric_{metric_kind}",
            status=status,
            diffs=diffs,
        ))
    
    return results


# =============================================================================
# MATHEMATICAL CHECKS
# =============================================================================

def check_unreachable_thresholds(yaml_path: Path) -> List[MathematicalIssue]:
    """
    Detect unreachable thresholds.
    
    Examples:
    - min_verified > formula_pool size
    - min_goal_hits > number of target formulas
    """
    issues = []
    
    if not yaml_path.exists():
        return issues
    
    with open(yaml_path, "r", encoding="utf-8") as f:
        curriculum = yaml.safe_load(f)
    
    slices = curriculum.get("slices", {})
    
    for slice_name, slice_config in slices.items():
        if not isinstance(slice_config, dict):
            continue
        
        params = slice_config.get("params", {})
        success_metric = slice_config.get("success_metric", {})
        formula_pool_entries = slice_config.get("formula_pool_entries", [])
        
        if not isinstance(success_metric, dict):
            continue
        
        metric_kind = success_metric.get("kind", "")
        metric_params = success_metric.get("parameters", {})
        
        if not isinstance(metric_params, dict):
            continue
        
        # Count target formulas
        target_count = sum(
            1 for entry in formula_pool_entries
            if isinstance(entry, dict) and entry.get("role") == "target"
        )
        
        total_formula_count = len(formula_pool_entries)
        formula_pool_param = params.get("formula_pool", total_formula_count)
        
        # Check goal_hit: min_goal_hits <= number of targets
        if metric_kind == "goal_hit":
            min_goal_hits = metric_params.get("min_goal_hits", 0)
            if min_goal_hits > target_count and target_count > 0:
                issues.append(MathematicalIssue(
                    slice_name=slice_name,
                    issue_type="unreachable_threshold",
                    severity=ConsistencyStatus.FAIL,
                    message=f"min_goal_hits ({min_goal_hits}) > target formulas ({target_count})",
                    details={
                        "min_goal_hits": min_goal_hits,
                        "target_count": target_count,
                    },
                    recommendation=f"Reduce min_goal_hits to at most {target_count}",
                ))
        
        # Check density: min_verified should be achievable
        if metric_kind == "density":
            min_verified = metric_params.get("min_verified", 0)
            if min_verified > total_formula_count and total_formula_count > 0:
                issues.append(MathematicalIssue(
                    slice_name=slice_name,
                    issue_type="unreachable_threshold",
                    severity=ConsistencyStatus.WARNING,
                    message=f"min_verified ({min_verified}) > formula pool ({total_formula_count})",
                    details={
                        "min_verified": min_verified,
                        "formula_pool_size": total_formula_count,
                    },
                    recommendation=f"Consider if {min_verified} verified formulas is achievable",
                ))
        
        # Check chain_length: min_chain_length should be reasonable
        if metric_kind == "chain_length":
            min_chain_length = metric_params.get("min_chain_length", 0)
            if min_chain_length > total_formula_count and total_formula_count > 0:
                issues.append(MathematicalIssue(
                    slice_name=slice_name,
                    issue_type="unreachable_threshold",
                    severity=ConsistencyStatus.WARNING,
                    message=f"min_chain_length ({min_chain_length}) > formula pool ({total_formula_count})",
                    details={
                        "min_chain_length": min_chain_length,
                        "formula_pool_size": total_formula_count,
                    },
                    recommendation="Ensure dependency graph supports required chain length",
                ))
        
        # Check multi_goal: required goals should exist
        if metric_kind == "multi_goal":
            required_goals = metric_params.get("required_goal_hashes", [])
            if isinstance(required_goals, list):
                # Check if required goals are in formula pool
                pool_hashes = {
                    entry.get("hash") for entry in formula_pool_entries
                    if isinstance(entry, dict) and "hash" in entry
                }
                missing_goals = set(required_goals) - pool_hashes
                if missing_goals:
                    issues.append(MathematicalIssue(
                        slice_name=slice_name,
                        issue_type="unreachable_threshold",
                        severity=ConsistencyStatus.FAIL,
                        message=f"Required goal hashes not in formula pool: {len(missing_goals)} missing",
                        details={
                            "missing_count": len(missing_goals),
                            "missing_hashes": list(missing_goals)[:3],  # Show first 3
                        },
                        recommendation="Ensure all required_goal_hashes exist in formula_pool_entries",
                    ))
    
    return issues


def check_contradictory_params(yaml_path: Path) -> List[MathematicalIssue]:
    """
    Detect contradictory metric parameters.
    """
    issues = []
    
    if not yaml_path.exists():
        return issues
    
    with open(yaml_path, "r", encoding="utf-8") as f:
        curriculum = yaml.safe_load(f)
    
    slices = curriculum.get("slices", {})
    
    for slice_name, slice_config in slices.items():
        if not isinstance(slice_config, dict):
            continue
        
        params = slice_config.get("params", {})
        success_metric = slice_config.get("success_metric", {})
        
        if not isinstance(success_metric, dict):
            continue
        
        metric_params = success_metric.get("parameters", {})
        
        # Check for zero or negative thresholds
        for param_name, param_value in metric_params.items():
            if isinstance(param_value, (int, float)):
                if "min" in param_name and param_value < 0:
                    issues.append(MathematicalIssue(
                        slice_name=slice_name,
                        issue_type="contradictory_params",
                        severity=ConsistencyStatus.FAIL,
                        message=f"Negative minimum: {param_name}={param_value}",
                        details={"param": param_name, "value": param_value},
                        recommendation=f"Set {param_name} to a non-negative value",
                    ))
        
        # Check that breadth_max <= total_max
        breadth_max = params.get("breadth_max", 0)
        total_max = params.get("total_max", float("inf"))
        if breadth_max > total_max:
            issues.append(MathematicalIssue(
                slice_name=slice_name,
                issue_type="contradictory_params",
                severity=ConsistencyStatus.WARNING,
                message=f"breadth_max ({breadth_max}) > total_max ({total_max})",
                details={"breadth_max": breadth_max, "total_max": total_max},
                recommendation="breadth_max should not exceed total_max",
            ))
    
    return issues


def check_degenerate_criteria(yaml_path: Path) -> List[MathematicalIssue]:
    """
    Detect degenerate success criteria.
    
    Examples:
    - min_verified = 0 (always succeeds)
    - empty required_goal_hashes (vacuous success)
    """
    issues = []
    
    if not yaml_path.exists():
        return issues
    
    with open(yaml_path, "r", encoding="utf-8") as f:
        curriculum = yaml.safe_load(f)
    
    slices = curriculum.get("slices", {})
    
    for slice_name, slice_config in slices.items():
        if not isinstance(slice_config, dict):
            continue
        
        success_metric = slice_config.get("success_metric", {})
        
        if not isinstance(success_metric, dict):
            continue
        
        metric_kind = success_metric.get("kind", "")
        metric_params = success_metric.get("parameters", {})
        
        if not isinstance(metric_params, dict):
            continue
        
        # Check for zero thresholds (trivial success)
        if metric_kind == "density":
            min_verified = metric_params.get("min_verified", 1)
            if min_verified == 0:
                issues.append(MathematicalIssue(
                    slice_name=slice_name,
                    issue_type="degenerate_criteria",
                    severity=ConsistencyStatus.WARNING,
                    message="min_verified=0 makes success trivial",
                    details={"min_verified": min_verified},
                    recommendation="Set min_verified > 0 for meaningful success criteria",
                ))
        
        # Check for empty goals
        if metric_kind == "multi_goal":
            required_goals = metric_params.get("required_goal_hashes", [])
            if isinstance(required_goals, list) and len(required_goals) == 0:
                issues.append(MathematicalIssue(
                    slice_name=slice_name,
                    issue_type="degenerate_criteria",
                    severity=ConsistencyStatus.WARNING,
                    message="Empty required_goal_hashes makes success vacuous",
                    details={"required_goal_hashes": required_goals},
                    recommendation="Specify at least one required goal hash",
                ))
        
        # Check for zero chain length
        if metric_kind == "chain_length":
            min_chain = metric_params.get("min_chain_length", 1)
            if min_chain == 0:
                issues.append(MathematicalIssue(
                    slice_name=slice_name,
                    issue_type="degenerate_criteria",
                    severity=ConsistencyStatus.WARNING,
                    message="min_chain_length=0 makes success trivial",
                    details={"min_chain_length": min_chain},
                    recommendation="Set min_chain_length > 0",
                ))
    
    return issues


# =============================================================================
# FORMULA INCONSISTENCY CHECKS
# =============================================================================

def check_formula_inconsistencies(yaml_path: Path) -> List[Dict[str, Any]]:
    """
    Check for formula-related inconsistencies.
    """
    inconsistencies = []
    
    if not yaml_path.exists():
        return inconsistencies
    
    with open(yaml_path, "r", encoding="utf-8") as f:
        curriculum = yaml.safe_load(f)
    
    slices = curriculum.get("slices", {})
    
    for slice_name, slice_config in slices.items():
        if not isinstance(slice_config, dict):
            continue
        
        formula_pool = slice_config.get("formula_pool_entries", [])
        
        # Check for duplicate hashes
        hashes = []
        for entry in formula_pool:
            if isinstance(entry, dict) and "hash" in entry:
                hashes.append(entry["hash"])
        
        seen = set()
        duplicates = []
        for h in hashes:
            if h in seen:
                duplicates.append(h)
            seen.add(h)
        
        if duplicates:
            inconsistencies.append({
                "slice_name": slice_name,
                "type": "duplicate_hash",
                "message": f"Found {len(duplicates)} duplicate hash(es)",
                "hashes": duplicates[:3],
            })
        
        # Check for formulas without hashes
        for entry in formula_pool:
            if isinstance(entry, dict):
                if "formula" in entry and "hash" not in entry:
                    inconsistencies.append({
                        "slice_name": slice_name,
                        "type": "missing_hash",
                        "formula_name": entry.get("name", "unnamed"),
                        "message": "Formula entry missing hash",
                    })
    
    return inconsistencies


# =============================================================================
# RECOMMENDATIONS GENERATOR
# =============================================================================

def generate_recommendations(report: OntologyReport) -> List[str]:
    """
    Generate actionable recommendations based on findings.
    """
    recommendations = []
    
    # From metric drift
    if report.metric_drift:
        recommendations.append(
            "Review metric definitions across code, docs, and YAML for consistency"
        )
    
    # From slice drift
    fail_slices = [s for s in report.slice_drift if s.status == ConsistencyStatus.FAIL]
    if fail_slices:
        slice_names = [s.slice_name for s in fail_slices]
        recommendations.append(
            f"Fix schema validation failures in slices: {', '.join(slice_names[:3])}"
        )
    
    # From mathematical issues
    for issue in report.mathematical_issues:
        if issue.recommendation and issue.recommendation not in recommendations:
            recommendations.append(issue.recommendation)
    
    # From formula inconsistencies
    if report.formula_inconsistencies:
        recommendations.append(
            "Review formula pool entries for missing hashes and duplicates"
        )
    
    # General
    if not recommendations:
        recommendations.append("No issues found. Configuration is consistent.")
    
    return recommendations


# =============================================================================
# MAIN ENGINE
# =============================================================================

def run_ontology_audit(
    yaml_path: Optional[Path] = None,
    doc_path: Optional[Path] = None,
    metrics_path: Optional[Path] = None,
) -> OntologyReport:
    """
    Run the full ontology consistency audit.
    """
    project_root = Path(__file__).parent.parent
    yaml_path = yaml_path or project_root / "config" / "curriculum_uplift_phase2.yaml"
    doc_path = doc_path or project_root / "docs" / "PHASE2_RFL_UPLIFT_PLAN.md"
    metrics_path = metrics_path or project_root / "experiments" / "slice_success_metrics.py"
    
    report = OntologyReport()
    
    # 1. Extract ontology from all sources
    code_entries = extract_ontology_from_code(metrics_path)
    doc_entries = extract_ontology_from_doc(doc_path)
    yaml_entries = extract_ontology_from_yaml(yaml_path)
    
    # Store for reference
    for entry in code_entries:
        report.metric_ontology.setdefault(entry.metric_kind, []).append(entry)
    for entry in doc_entries:
        report.metric_ontology.setdefault(entry.metric_kind, []).append(entry)
    for entry in yaml_entries:
        report.metric_ontology.setdefault(entry.metric_kind, []).append(entry)
    
    # 2. Compare ontologies
    report.metric_drift = compare_ontologies(code_entries, doc_entries, yaml_entries)
    
    # 3. Bidirectional schema validation
    report.slice_drift.extend(validate_doc_to_yaml(doc_path, yaml_path))
    report.slice_drift.extend(validate_yaml_to_code(yaml_path))
    report.slice_drift.extend(validate_code_to_doc(doc_path))
    
    # 4. Mathematical checks
    report.mathematical_issues.extend(check_unreachable_thresholds(yaml_path))
    report.mathematical_issues.extend(check_contradictory_params(yaml_path))
    report.mathematical_issues.extend(check_degenerate_criteria(yaml_path))
    
    # 5. Formula inconsistencies
    report.formula_inconsistencies = check_formula_inconsistencies(yaml_path)
    
    # 6. Generate recommendations
    report.recommendations = generate_recommendations(report)
    
    # 7. Compute final status
    report.status = report.compute_status()
    
    return report


# =============================================================================
# REPORTING
# =============================================================================

def format_markdown_report(report: OntologyReport) -> str:
    """Generate human-readable Markdown report."""
    lines = [
        "# Ontology Consistency Engine Report",
        "",
        "**PHASE II — NOT USED IN PHASE I**",
        "",
        f"**Overall Status: {report.status.value.upper()}**",
        "",
        "## Summary",
        "",
        f"- Metric Ontology Entries: {sum(len(v) for v in report.metric_ontology.values())}",
        f"- Metric Drift Findings: {len(report.metric_drift)}",
        f"- Schema Validation Results: {len(report.slice_drift)}",
        f"- Mathematical Issues: {len(report.mathematical_issues)}",
        f"- Formula Inconsistencies: {len(report.formula_inconsistencies)}",
        "",
    ]
    
    # Metric Ontology
    lines.append("## Metric Ontology")
    lines.append("")
    for metric_kind, entries in report.metric_ontology.items():
        lines.append(f"### `{metric_kind}`")
        for entry in entries:
            lines.append(f"- **{entry.source}**: params={sorted(entry.required_params)}")
        lines.append("")
    
    # Metric Drift
    if report.metric_drift:
        lines.append("## Metric Drift")
        lines.append("")
        for diff in report.metric_drift:
            lines.append(f"- [{diff.diff_type}] {diff.metric_kind}: {diff.source_a} → {diff.source_b}")
            if diff.details:
                lines.append(f"  - Details: {diff.details}")
        lines.append("")
    
    # Schema Validation
    lines.append("## Bidirectional Schema Validation")
    lines.append("")
    for result in report.slice_drift:
        icon = {"ok": "✅", "warning": "⚠️", "fail": "❌"}.get(result.status.value, "❓")
        lines.append(f"- {icon} `{result.slice_name}` ({result.direction}): {result.status.value}")
        for diff in result.diffs:
            lines.append(f"  - {diff}")
    lines.append("")
    
    # Mathematical Issues
    if report.mathematical_issues:
        lines.append("## Mathematical Issues")
        lines.append("")
        for issue in report.mathematical_issues:
            icon = {"ok": "✅", "warning": "⚠️", "fail": "❌"}.get(issue.severity.value, "❓")
            lines.append(f"- {icon} **{issue.slice_name}** [{issue.issue_type}]: {issue.message}")
            if issue.recommendation:
                lines.append(f"  - Recommendation: {issue.recommendation}")
        lines.append("")
    
    # Formula Inconsistencies
    if report.formula_inconsistencies:
        lines.append("## Formula Inconsistencies")
        lines.append("")
        for inc in report.formula_inconsistencies:
            lines.append(f"- `{inc.get('slice_name', 'unknown')}`: {inc.get('message', '')}")
        lines.append("")
    
    # Recommendations
    lines.append("## Recommendations")
    lines.append("")
    for i, rec in enumerate(report.recommendations, 1):
        lines.append(f"{i}. {rec}")
    lines.append("")
    
    return "\n".join(lines)


# =============================================================================
# CLI
# =============================================================================

def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="PHASE II Ontology Consistency Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Constraints:
- Must not rewrite curriculum
- Must not adjust thresholds
- Must not modify metrics code

This is a DETECTION-ONLY auditor.
        """,
    )
    parser.add_argument(
        "--yaml",
        type=Path,
        default=None,
        help="Path to curriculum_uplift_phase2.yaml",
    )
    parser.add_argument(
        "--doc",
        type=Path,
        default=None,
        help="Path to PHASE2_RFL_UPLIFT_PLAN.md",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=None,
        help="Path to slice_success_metrics.py",
    )
    parser.add_argument(
        "--ci-json",
        action="store_true",
        help="Output CI-compatible JSON report",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write report to file",
    )
    
    args = parser.parse_args()
    
    # Run audit
    report = run_ontology_audit(
        yaml_path=args.yaml,
        doc_path=args.doc,
        metrics_path=args.metrics,
    )
    
    # Format output
    if args.ci_json:
        output = json.dumps(report.to_ci_json(), indent=2)
    else:
        output = format_markdown_report(report)
    
    # Write output
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"Report written to {args.output}")
    else:
        print(output)
    
    # Exit code
    if report.status == ConsistencyStatus.FAIL:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()

