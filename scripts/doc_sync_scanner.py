#!/usr/bin/env python3
"""
Documentation Synchronization Scanner

PHASE II — NOT RUN IN PHASE I
No uplift claims are made.
Deterministic execution guaranteed.

This module scans /docs, /experiments, /backend for documentation consistency,
flags terminology mismatches, and ensures governance alignment with VSD_PHASE_2.md
and PREREG_UPLIFT_U2.yaml.

Author: doc-ops-1 (Governance Synchronization Officer)
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

# PHASE II CONSTANTS
PHASE_II_MARKER = "PHASE II — NOT RUN IN PHASE I"
NO_UPLIFT_MARKER = "No uplift claims are made."
DETERMINISM_MARKER = "Deterministic execution"


@dataclass(frozen=True)
class TermDefinition:
    """Immutable governance term definition."""

    canonical_name: str  # The official governance term
    doc_variants: FrozenSet[str]  # Acceptable variations in documentation
    code_variants: FrozenSet[str]  # Acceptable variations in code
    category: str  # Category: slice, metric, mode, phase, symbol, concept
    description: str
    governance_source: str  # Source document (VSD_PHASE_2.md, PREREG_UPLIFT_U2.yaml, etc.)


@dataclass
class TerminologyViolation:
    """A terminology inconsistency detected in the codebase."""

    file_path: str
    line_number: int
    found_term: str
    expected_term: str
    category: str
    severity: str  # error, warning, info
    context: str  # Surrounding code/text
    suggestion: str


@dataclass
class DocstringComplianceResult:
    """Result of docstring compliance check for a file."""

    file_path: str
    has_phase_marker: bool
    has_no_uplift_marker: bool
    has_determinism_marker: bool
    implements_metric: bool
    implements_loader: bool
    implements_runner: bool
    violations: List[str] = field(default_factory=list)


@dataclass
class OrphanedDocumentation:
    """Documentation that references non-existent code elements."""

    doc_file: str
    line_number: int
    referenced_element: str
    element_type: str  # function, class, file, slice, metric


# ==============================================================================
# GOVERNANCE VOCABULARY REGISTRY
# ==============================================================================

def build_governance_vocabulary() -> Dict[str, TermDefinition]:
    """
    Build the canonical governance vocabulary from VSD_PHASE_2.md and
    PREREG_UPLIFT_U2.yaml specifications.

    Returns:
        Dictionary mapping canonical names to their definitions.
    """
    vocabulary: Dict[str, TermDefinition] = {}

    # ----- SLICE NAMES -----
    slice_terms = [
        TermDefinition(
            canonical_name="slice_debug_uplift",
            doc_variants=frozenset({"slice_debug_uplift", "debug uplift slice", "debug slice"}),
            code_variants=frozenset({"slice_debug_uplift", "SLICE_DEBUG_UPLIFT"}),
            category="slice",
            description="Debug slice for microscopic uplift experiments (atoms=2, depth=2)",
            governance_source="config/curriculum.yaml",
        ),
        TermDefinition(
            canonical_name="slice_easy_fo",
            doc_variants=frozenset({"slice_easy_fo", "easy slice", "FO easy slice", "Easy"}),
            code_variants=frozenset({"slice_easy_fo", "SLICE_EASY_FO"}),
            category="slice",
            description="Easy slice for First Organism testing (atoms=3, depth=3)",
            governance_source="config/curriculum.yaml",
        ),
        TermDefinition(
            canonical_name="slice_uplift_proto",
            doc_variants=frozenset({"slice_uplift_proto", "uplift proto slice", "proto slice"}),
            code_variants=frozenset({"slice_uplift_proto", "SLICE_UPLIFT_PROTO"}),
            category="slice",
            description="Medium-hard slice for uplift experiments (atoms=3, depth=4)",
            governance_source="config/curriculum.yaml",
        ),
        TermDefinition(
            canonical_name="atoms4-depth4",
            doc_variants=frozenset({"atoms4-depth4", "4-atom 4-depth slice"}),
            code_variants=frozenset({"atoms4-depth4", "atoms4_depth4"}),
            category="slice",
            description="Intermediate slice (atoms=4, depth=4)",
            governance_source="config/curriculum.yaml",
        ),
        TermDefinition(
            canonical_name="atoms4-depth5",
            doc_variants=frozenset({"atoms4-depth5", "4-atom 5-depth slice"}),
            code_variants=frozenset({"atoms4-depth5", "atoms4_depth5"}),
            category="slice",
            description="Intermediate slice (atoms=4, depth=5)",
            governance_source="config/curriculum.yaml",
        ),
        TermDefinition(
            canonical_name="atoms5-depth6",
            doc_variants=frozenset({"atoms5-depth6", "5-atom 6-depth slice"}),
            code_variants=frozenset({"atoms5-depth6", "atoms5_depth6"}),
            category="slice",
            description="Intermediate slice (atoms=5, depth=6)",
            governance_source="config/curriculum.yaml",
        ),
        TermDefinition(
            canonical_name="slice_medium",
            doc_variants=frozenset({"slice_medium", "medium slice", "Medium", "Wide Slice"}),
            code_variants=frozenset({"slice_medium", "SLICE_MEDIUM"}),
            category="slice",
            description="Wide Slice for RFL uplift experiments (atoms=5, depth=7)",
            governance_source="config/curriculum.yaml",
        ),
        TermDefinition(
            canonical_name="first_organism_pl2_hard",
            doc_variants=frozenset({"first_organism_pl2_hard", "FO hard slice", "PL2 hard"}),
            code_variants=frozenset({"first_organism_pl2_hard"}),
            category="slice",
            description="First Organism PL2 hard slice (atoms=6, depth=8)",
            governance_source="config/curriculum.yaml",
        ),
        TermDefinition(
            canonical_name="slice_hard",
            doc_variants=frozenset({"slice_hard", "hard slice", "Hard"}),
            code_variants=frozenset({"slice_hard", "SLICE_HARD"}),
            category="slice",
            description="Hard slice for stress testing (atoms=7, depth=12)",
            governance_source="config/curriculum.yaml",
        ),
    ]

    # ----- SUCCESS METRICS (from VSD_PHASE_2.md) -----
    metric_terms = [
        TermDefinition(
            canonical_name="goal_hit",
            doc_variants=frozenset({"goal_hit", "goal hit", "goal-hit", "goal hit rate"}),
            code_variants=frozenset({"goal_hit", "GOAL_HIT", "goalHit"}),
            category="metric",
            description="Rate of achieving the primary objective",
            governance_source="VSD_PHASE_2.md",
        ),
        TermDefinition(
            canonical_name="sparse_density",
            doc_variants=frozenset({"sparse_density", "sparse density", "sparsity"}),
            code_variants=frozenset({"sparse_density", "SPARSE_DENSITY", "sparseDensity"}),
            category="metric",
            description="Measure of the efficiency of the solution path",
            governance_source="VSD_PHASE_2.md",
        ),
        TermDefinition(
            canonical_name="chain_success",
            doc_variants=frozenset({"chain_success", "chain success", "chain-success"}),
            code_variants=frozenset({"chain_success", "CHAIN_SUCCESS", "chainSuccess"}),
            category="metric",
            description="Success rate across a chain of dependent tasks",
            governance_source="VSD_PHASE_2.md",
        ),
        TermDefinition(
            canonical_name="joint_goal",
            doc_variants=frozenset({"joint_goal", "joint goal", "joint-goal"}),
            code_variants=frozenset({"joint_goal", "JOINT_GOAL", "jointGoal"}),
            category="metric",
            description="Rate of achieving a composite goal of multiple objectives",
            governance_source="VSD_PHASE_2.md",
        ),
        TermDefinition(
            canonical_name="abstention_rate",
            doc_variants=frozenset({"abstention_rate", "abstention rate", "α_rate", "alpha_rate"}),
            code_variants=frozenset({"abstention_rate", "abstention_fraction", "alpha_rate"}),
            category="metric",
            description="Abstention rate = abstentions / total_attempts",
            governance_source="docs/RFL_LAW.md",
        ),
        TermDefinition(
            canonical_name="abstention_mass",
            doc_variants=frozenset({"abstention_mass", "abstention mass", "α_mass", "alpha_mass"}),
            code_variants=frozenset({"abstention_mass", "abstention_count", "alpha_mass"}),
            category="metric",
            description="Abstention mass = raw abstention count",
            governance_source="docs/RFL_LAW.md",
        ),
        TermDefinition(
            canonical_name="coverage_rate",
            doc_variants=frozenset({"coverage_rate", "coverage rate", "coverage"}),
            code_variants=frozenset({"coverage_rate", "coverage", "ci_lower_min"}),
            category="metric",
            description="Coverage metric for curriculum gate",
            governance_source="config/curriculum.yaml",
        ),
        TermDefinition(
            canonical_name="throughput",
            doc_variants=frozenset({"throughput", "proofs per hour", "pph", "velocity"}),
            code_variants=frozenset({"throughput", "proofs_per_hour", "min_pph"}),
            category="metric",
            description="Proofs per hour throughput metric",
            governance_source="experiments/METRICS_DEFINITION.md",
        ),
        TermDefinition(
            canonical_name="max_depth",
            doc_variants=frozenset({"max_depth", "maximum depth", "depth_max"}),
            code_variants=frozenset({"max_depth", "depth_max", "MAX_DEPTH"}),
            category="metric",
            description="Maximum derivation depth of verified statements",
            governance_source="experiments/METRICS_DEFINITION.md",
        ),
    ]

    # ----- MODE NAMES -----
    mode_terms = [
        TermDefinition(
            canonical_name="baseline",
            doc_variants=frozenset({"baseline", "Baseline", "random baseline", "BFS baseline"}),
            code_variants=frozenset({"baseline", "BASELINE", "mode_baseline"}),
            category="mode",
            description="Baseline mode using random/BFS derivation",
            governance_source="experiments/METRICS_DEFINITION.md",
        ),
        TermDefinition(
            canonical_name="rfl",
            doc_variants=frozenset({"rfl", "RFL", "Reflexive Formal Learning", "reflexive mode"}),
            code_variants=frozenset({"rfl", "RFL", "mode_rfl"}),
            category="mode",
            description="Reflexive Formal Learning mode with learned policy",
            governance_source="docs/RFL_LAW.md",
        ),
    ]

    # ----- PHASE TERMINOLOGY -----
    phase_terms = [
        TermDefinition(
            canonical_name="PHASE_I",
            doc_variants=frozenset({"PHASE I", "Phase I", "Phase 1", "phase-1"}),
            code_variants=frozenset({"PHASE_I", "phase_1", "PHASE1"}),
            category="phase",
            description="Phase I - foundational experiments (no modification allowed)",
            governance_source="VSD_PHASE_2.md",
        ),
        TermDefinition(
            canonical_name="PHASE_II",
            doc_variants=frozenset({"PHASE II", "Phase II", "Phase 2", "phase-2"}),
            code_variants=frozenset({"PHASE_II", "phase_2", "PHASE2"}),
            category="phase",
            description="Phase II - Operation Asymmetry governance",
            governance_source="VSD_PHASE_2.md",
        ),
        TermDefinition(
            canonical_name="PHASE_III",
            doc_variants=frozenset({"PHASE III", "Phase III", "Phase 3", "phase-3"}),
            code_variants=frozenset({"PHASE_III", "phase_3", "PHASE3"}),
            category="phase",
            description="Phase III - post-audit escalation phase",
            governance_source="VSD_PHASE_2.md",
        ),
    ]

    # ----- CORE SYMBOLS (from RFL_LAW.md) -----
    symbol_terms = [
        TermDefinition(
            canonical_name="H_t",
            doc_variants=frozenset({"H_t", "H(t)", "composite_root", "composite attestation root"}),
            code_variants=frozenset({"H_t", "ht_hash", "composite_root", "attestation_root"}),
            category="symbol",
            description="Composite attestation root = SHA256(R_t || U_t)",
            governance_source="docs/RFL_LAW.md",
        ),
        TermDefinition(
            canonical_name="R_t",
            doc_variants=frozenset({"R_t", "R(t)", "reasoning_root", "Reasoning Merkle root"}),
            code_variants=frozenset({"R_t", "reasoning_root", "reasoning_merkle_root"}),
            category="symbol",
            description="Reasoning Merkle root over proof artifacts",
            governance_source="docs/RFL_LAW.md",
        ),
        TermDefinition(
            canonical_name="U_t",
            doc_variants=frozenset({"U_t", "U(t)", "ui_root", "UI Merkle root"}),
            code_variants=frozenset({"U_t", "ui_root", "ui_merkle_root"}),
            category="symbol",
            description="UI Merkle root over human interaction artifacts",
            governance_source="docs/RFL_LAW.md",
        ),
        TermDefinition(
            canonical_name="symbolic_descent",
            doc_variants=frozenset({"∇_sym", "symbolic descent", "descent gradient", "nabla_sym"}),
            code_variants=frozenset({"symbolic_descent", "nabla_sym", "descent_gradient"}),
            category="symbol",
            description="Symbolic descent = -(α_rate - τ)",
            governance_source="docs/RFL_LAW.md",
        ),
        TermDefinition(
            canonical_name="step_id",
            doc_variants=frozenset({"step_id", "step ID", "deterministic step identifier"}),
            code_variants=frozenset({"step_id", "stepId", "STEP_ID"}),
            category="symbol",
            description="Deterministic step identifier (64-char hex)",
            governance_source="docs/RFL_LAW.md",
        ),
        TermDefinition(
            canonical_name="abstention_tolerance",
            doc_variants=frozenset({"τ", "tau", "abstention tolerance", "tolerance threshold"}),
            code_variants=frozenset({"abstention_tolerance", "tau", "tolerance"}),
            category="symbol",
            description="Abstention tolerance threshold (default 0.10)",
            governance_source="docs/RFL_LAW.md",
        ),
    ]

    # ----- CORE CONCEPTS -----
    concept_terms = [
        TermDefinition(
            canonical_name="First_Organism",
            doc_variants=frozenset({"First Organism", "FO", "first organism", "first-organism"}),
            code_variants=frozenset({"first_organism", "FirstOrganism", "FO"}),
            category="concept",
            description="The first production metabolic cycle harness",
            governance_source="docs/FIRST_ORGANISM.md",
        ),
        TermDefinition(
            canonical_name="curriculum_slice",
            doc_variants=frozenset({"curriculum slice", "slice", "Curriculum Slice"}),
            code_variants=frozenset({"CurriculumSlice", "curriculum_slice", "slice_cfg"}),
            category="concept",
            description="A contiguous run interval with fixed derivation policy",
            governance_source="config/curriculum.yaml",
        ),
        TermDefinition(
            canonical_name="attestation",
            doc_variants=frozenset({"attestation", "Attestation", "attested run"}),
            code_variants=frozenset({"attestation", "AttestedRunContext", "attested"}),
            category="concept",
            description="Cryptographic verification of derivation results",
            governance_source="docs/ATTESTATION_SPEC.md",
        ),
        TermDefinition(
            canonical_name="ledger_entry",
            doc_variants=frozenset({"ledger entry", "RunLedgerEntry", "run entry"}),
            code_variants=frozenset({"RunLedgerEntry", "ledger_entry", "policy_ledger"}),
            category="concept",
            description="Structured curriculum ledger entry for a single RFL run",
            governance_source="docs/RFL_LAW.md",
        ),
        TermDefinition(
            canonical_name="dual_attestation",
            doc_variants=frozenset({"dual attestation", "dual root attestation", "dual-attestation"}),
            code_variants=frozenset({"dual_attestation", "dual_root", "DUAL_ATTESTATION"}),
            category="concept",
            description="Require two independent statistical checks to agree",
            governance_source="VSD_PHASE_2.md",
        ),
    ]

    # Combine all terms
    all_terms = slice_terms + metric_terms + mode_terms + phase_terms + symbol_terms + concept_terms

    for term in all_terms:
        vocabulary[term.canonical_name] = term

    return vocabulary


# ==============================================================================
# SCANNER IMPLEMENTATION
# ==============================================================================


class DocumentationSyncScanner:
    """
    Scans documentation and code for terminology consistency violations.

    PHASE II — NOT RUN IN PHASE I
    No uplift claims are made.
    Deterministic execution guaranteed.
    """

    SCAN_EXTENSIONS = {
        "docs": {".md", ".tex", ".txt", ".rst"},
        "code": {".py", ".lean", ".yaml", ".yml", ".json"},
    }

    IGNORE_DIRS = {
        "__pycache__",
        ".git",
        "node_modules",
        ".venv",
        "venv",
        "dist",
        "build",
        ".pytest_cache",
        ".mypy_cache",
        "MagicMock",
    }

    def __init__(self, root_path: Path, vocabulary: Optional[Dict[str, TermDefinition]] = None):
        """
        Initialize the scanner.

        Args:
            root_path: Root path of the repository
            vocabulary: Optional custom vocabulary (defaults to governance vocabulary)
        """
        self.root_path = root_path
        self.vocabulary = vocabulary or build_governance_vocabulary()
        self.violations: List[TerminologyViolation] = []
        self.docstring_results: List[DocstringComplianceResult] = []
        self.orphaned_docs: List[OrphanedDocumentation] = []
        self._term_usage: Dict[str, List[Tuple[str, int]]] = {}  # term -> [(file, line), ...]

    def scan_all(self) -> Dict[str, Any]:
        """
        Execute full documentation consistency scan.

        Returns:
            Dictionary with scan results
        """
        self.violations.clear()
        self.docstring_results.clear()
        self.orphaned_docs.clear()
        self._term_usage.clear()

        # Scan directories
        scan_dirs = ["docs", "experiments", "backend", "scripts", "tests", "rfl"]
        for dir_name in scan_dirs:
            dir_path = self.root_path / dir_name
            if dir_path.exists():
                self._scan_directory(dir_path)

        # Check for orphaned documentation
        self._detect_orphaned_docs()

        # Generate summary
        return {
            "total_violations": len(self.violations),
            "violations_by_severity": self._count_by_severity(),
            "violations_by_category": self._count_by_category(),
            "docstring_compliance": self._summarize_docstring_compliance(),
            "orphaned_documentation": len(self.orphaned_docs),
            "term_coverage": self._compute_term_coverage(),
            "violations": [self._violation_to_dict(v) for v in self.violations],
            "docstring_results": [self._docstring_to_dict(d) for d in self.docstring_results],
            "orphaned_docs": [self._orphan_to_dict(o) for o in self.orphaned_docs],
        }

    def _scan_directory(self, dir_path: Path) -> None:
        """Recursively scan a directory for consistency issues."""
        for item in dir_path.iterdir():
            if item.name in self.IGNORE_DIRS:
                continue

            if item.is_dir():
                self._scan_directory(item)
            elif item.is_file():
                ext = item.suffix.lower()
                if ext in self.SCAN_EXTENSIONS["docs"]:
                    self._scan_documentation_file(item)
                elif ext in self.SCAN_EXTENSIONS["code"]:
                    self._scan_code_file(item)

    def _scan_documentation_file(self, file_path: Path) -> None:
        """Scan a documentation file for terminology issues."""
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return

        lines = content.split("\n")
        for line_num, line in enumerate(lines, 1):
            self._check_terminology(file_path, line_num, line, is_code=False)

    def _scan_code_file(self, file_path: Path) -> None:
        """Scan a code file for terminology issues and docstring compliance."""
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return

        # Check docstring compliance for Python files
        if file_path.suffix == ".py":
            self._check_docstring_compliance(file_path, content)

        lines = content.split("\n")
        for line_num, line in enumerate(lines, 1):
            self._check_terminology(file_path, line_num, line, is_code=True)

    def _check_terminology(self, file_path: Path, line_num: int, line: str, is_code: bool) -> None:
        """Check a line for terminology consistency."""
        rel_path = str(file_path.relative_to(self.root_path))

        for canonical_name, term_def in self.vocabulary.items():
            # Build regex for this term
            variants = term_def.code_variants if is_code else term_def.doc_variants

            for variant in variants:
                # Skip if variant is the canonical name
                if variant == canonical_name:
                    continue

                # Case-sensitive search for exact matches
                pattern = r'\b' + re.escape(variant) + r'\b'
                matches = list(re.finditer(pattern, line, re.IGNORECASE))

                for match in matches:
                    found_term = match.group()
                    # Only flag if not already canonical
                    if found_term.lower() != canonical_name.lower():
                        # Record usage
                        if found_term not in self._term_usage:
                            self._term_usage[found_term] = []
                        self._term_usage[found_term].append((rel_path, line_num))

            # Check for mismatches between doc and code variants
            if is_code:
                for doc_variant in term_def.doc_variants - term_def.code_variants:
                    if doc_variant in line and doc_variant not in term_def.code_variants:
                        # Documentation term used in code - potential issue
                        self.violations.append(
                            TerminologyViolation(
                                file_path=rel_path,
                                line_number=line_num,
                                found_term=doc_variant,
                                expected_term=list(term_def.code_variants)[0],
                                category=term_def.category,
                                severity="warning",
                                context=line.strip()[:100],
                                suggestion=f"Use code variant '{list(term_def.code_variants)[0]}' instead of doc variant '{doc_variant}'",
                            )
                        )

    def _check_docstring_compliance(self, file_path: Path, content: str) -> None:
        """Check Python file docstring compliance with Phase II requirements."""
        rel_path = str(file_path.relative_to(self.root_path))

        # Determine if file implements metric, loader, or runner
        implements_metric = any(
            kw in content.lower()
            for kw in ["def compute_", "class metric", "def calculate_", "def measure_"]
        )
        implements_loader = any(
            kw in content.lower()
            for kw in ["loader", "def load_", "class loader"]
        )
        implements_runner = any(
            kw in content.lower()
            for kw in ["runner", "def run_", "class runner", "def execute_"]
        )

        # Check for required markers
        has_phase_marker = PHASE_II_MARKER in content
        has_no_uplift_marker = NO_UPLIFT_MARKER in content or "no uplift claim" in content.lower()
        has_determinism_marker = DETERMINISM_MARKER.lower() in content.lower()

        violations: List[str] = []

        # Files that implement metrics/loaders/runners must have Phase II markers
        if implements_metric or implements_loader or implements_runner:
            if not has_phase_marker:
                violations.append(f"Missing '{PHASE_II_MARKER}' marker")
            if not has_no_uplift_marker:
                violations.append(f"Missing '{NO_UPLIFT_MARKER}' marker")
            if not has_determinism_marker:
                violations.append("Missing determinism guarantee statement")

        result = DocstringComplianceResult(
            file_path=rel_path,
            has_phase_marker=has_phase_marker,
            has_no_uplift_marker=has_no_uplift_marker,
            has_determinism_marker=has_determinism_marker,
            implements_metric=implements_metric,
            implements_loader=implements_loader,
            implements_runner=implements_runner,
            violations=violations,
        )

        self.docstring_results.append(result)

        # Add violations to main list
        for violation_text in violations:
            self.violations.append(
                TerminologyViolation(
                    file_path=rel_path,
                    line_number=1,  # Module-level issue
                    found_term="(missing marker)",
                    expected_term="Phase II compliance markers",
                    category="docstring",
                    severity="error",
                    context=violation_text,
                    suggestion=f"Add required docstring markers for Phase II compliance",
                )
            )

    def _detect_orphaned_docs(self) -> None:
        """Detect documentation referencing non-existent code elements."""
        docs_path = self.root_path / "docs"
        if not docs_path.exists():
            return

        # Build set of existing code elements
        code_elements: Set[str] = set()
        for code_dir in ["backend", "experiments", "scripts", "rfl"]:
            dir_path = self.root_path / code_dir
            if dir_path.exists():
                for py_file in dir_path.rglob("*.py"):
                    code_elements.add(py_file.stem)
                    # Parse function/class names
                    try:
                        content = py_file.read_text(encoding="utf-8", errors="replace")
                        # Simple regex for function and class names
                        for match in re.finditer(r"^(?:def|class)\s+(\w+)", content, re.MULTILINE):
                            code_elements.add(match.group(1))
                    except Exception:
                        pass

        # Add slice names from curriculum
        for term in self.vocabulary.values():
            if term.category == "slice":
                code_elements.add(term.canonical_name)

        # Scan documentation for references
        for doc_file in docs_path.rglob("*.md"):
            try:
                content = doc_file.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

            lines = content.split("\n")
            for line_num, line in enumerate(lines, 1):
                # Look for code references (backticks)
                for match in re.finditer(r"`([a-zA-Z_][a-zA-Z0-9_]*)`", line):
                    ref = match.group(1)
                    # Skip common non-code terms
                    if ref.lower() in {"true", "false", "none", "null", "string", "int", "float"}:
                        continue
                    # Check if it looks like a code element but doesn't exist
                    if "_" in ref or ref[0].isupper():
                        if ref not in code_elements and ref.lower() not in code_elements:
                            self.orphaned_docs.append(
                                OrphanedDocumentation(
                                    doc_file=str(doc_file.relative_to(self.root_path)),
                                    line_number=line_num,
                                    referenced_element=ref,
                                    element_type="function" if "_" in ref else "class",
                                )
                            )

    def _count_by_severity(self) -> Dict[str, int]:
        """Count violations by severity level."""
        counts: Dict[str, int] = {"error": 0, "warning": 0, "info": 0}
        for v in self.violations:
            counts[v.severity] = counts.get(v.severity, 0) + 1
        return counts

    def _count_by_category(self) -> Dict[str, int]:
        """Count violations by category."""
        counts: Dict[str, int] = {}
        for v in self.violations:
            counts[v.category] = counts.get(v.category, 0) + 1
        return counts

    def _summarize_docstring_compliance(self) -> Dict[str, Any]:
        """Summarize docstring compliance results."""
        total = len(self.docstring_results)
        compliant = sum(1 for r in self.docstring_results if not r.violations)
        metrics_files = sum(1 for r in self.docstring_results if r.implements_metric)
        loaders_files = sum(1 for r in self.docstring_results if r.implements_loader)
        runners_files = sum(1 for r in self.docstring_results if r.implements_runner)

        return {
            "total_files": total,
            "compliant_files": compliant,
            "compliance_rate": compliant / total if total > 0 else 1.0,
            "metrics_files": metrics_files,
            "loaders_files": loaders_files,
            "runners_files": runners_files,
        }

    def _compute_term_coverage(self) -> Dict[str, Any]:
        """Compute terminology coverage statistics."""
        used_terms = set(self._term_usage.keys())
        canonical_terms = set(self.vocabulary.keys())

        return {
            "total_governance_terms": len(canonical_terms),
            "unique_terms_found": len(used_terms),
            "terms_by_category": {
                cat: len([t for t in self.vocabulary.values() if t.category == cat])
                for cat in {"slice", "metric", "mode", "phase", "symbol", "concept"}
            },
        }

    def _violation_to_dict(self, v: TerminologyViolation) -> Dict[str, Any]:
        """Convert violation to dictionary."""
        return {
            "file": v.file_path,
            "line": v.line_number,
            "found": v.found_term,
            "expected": v.expected_term,
            "category": v.category,
            "severity": v.severity,
            "context": v.context,
            "suggestion": v.suggestion,
        }

    def _docstring_to_dict(self, d: DocstringComplianceResult) -> Dict[str, Any]:
        """Convert docstring result to dictionary."""
        return {
            "file": d.file_path,
            "has_phase_marker": d.has_phase_marker,
            "has_no_uplift_marker": d.has_no_uplift_marker,
            "has_determinism_marker": d.has_determinism_marker,
            "implements_metric": d.implements_metric,
            "implements_loader": d.implements_loader,
            "implements_runner": d.implements_runner,
            "violations": d.violations,
        }

    def _orphan_to_dict(self, o: OrphanedDocumentation) -> Dict[str, Any]:
        """Convert orphan to dictionary."""
        return {
            "doc_file": o.doc_file,
            "line": o.line_number,
            "referenced_element": o.referenced_element,
            "element_type": o.element_type,
        }


# ==============================================================================
# TERM MAPPING TABLE GENERATOR
# ==============================================================================


def generate_term_mapping_table(vocabulary: Dict[str, TermDefinition]) -> str:
    """
    Generate the PHASE2_TERM_MAPPING.md content.

    Args:
        vocabulary: The governance vocabulary

    Returns:
        Markdown content for the mapping table
    """
    lines = [
        "# Phase II Terminology Mapping Table",
        "",
        "PHASE II — NOT RUN IN PHASE I",
        "No uplift claims are made.",
        "",
        "This document provides the canonical mapping between documentation terms,",
        "code terms, and governance terms as defined in VSD_PHASE_2.md and",
        "PREREG_UPLIFT_U2.yaml.",
        "",
        "## Governance Safeguards",
        "",
        "- All terminology must match this canonical mapping",
        "- Violations are flagged by the documentation consistency scanner",
        "- CI gates enforce terminology consistency",
        "",
    ]

    # Group by category
    categories = ["slice", "metric", "mode", "phase", "symbol", "concept"]

    for category in categories:
        terms = [t for t in vocabulary.values() if t.category == category]
        if not terms:
            continue

        lines.append(f"## {category.title()} Terms")
        lines.append("")
        lines.append("| Canonical Name | Doc Variants | Code Variants | Source |")
        lines.append("|----------------|--------------|---------------|--------|")

        for term in sorted(terms, key=lambda t: t.canonical_name):
            doc_vars = ", ".join(sorted(term.doc_variants)[:3])
            code_vars = ", ".join(sorted(term.code_variants)[:3])
            if len(term.doc_variants) > 3:
                doc_vars += ", ..."
            if len(term.code_variants) > 3:
                code_vars += ", ..."

            lines.append(
                f"| `{term.canonical_name}` | {doc_vars} | {code_vars} | {term.governance_source} |"
            )

        lines.append("")

    # Add description section
    lines.extend([
        "## Term Descriptions",
        "",
    ])

    for category in categories:
        terms = [t for t in vocabulary.values() if t.category == category]
        if not terms:
            continue

        lines.append(f"### {category.title()} Descriptions")
        lines.append("")

        for term in sorted(terms, key=lambda t: t.canonical_name):
            lines.append(f"- **{term.canonical_name}**: {term.description}")

        lines.append("")

    # Add footer
    lines.extend([
        "---",
        "",
        "Generated by `scripts/doc_sync_scanner.py`",
        "",
        "Last updated: See git commit history",
    ])

    return "\n".join(lines)


# ==============================================================================
# CLI INTERFACE
# ==============================================================================


def main() -> int:
    """
    Main entry point for the documentation sync scanner.

    Returns:
        Exit code (0 for success, non-zero for violations)
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Documentation Synchronization Scanner - Phase II Governance"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).parent.parent,
        help="Repository root path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON file for scan results",
    )
    parser.add_argument(
        "--generate-mapping",
        action="store_true",
        help="Generate PHASE2_TERM_MAPPING.md",
    )
    parser.add_argument(
        "--ci-mode",
        action="store_true",
        help="CI mode: exit with error code if violations found",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("PHASE II — Documentation Synchronization Scanner")
    print("No uplift claims are made. Deterministic execution guaranteed.")
    print("=" * 70)
    print()

    # Build vocabulary
    vocabulary = build_governance_vocabulary()
    print(f"Loaded {len(vocabulary)} governance terms")

    # Generate mapping table if requested
    if args.generate_mapping:
        mapping_content = generate_term_mapping_table(vocabulary)
        mapping_path = args.root / "docs" / "PHASE2_TERM_MAPPING.md"
        mapping_path.write_text(mapping_content, encoding="utf-8")
        print(f"Generated: {mapping_path}")

    # Run scan
    scanner = DocumentationSyncScanner(args.root, vocabulary)
    results = scanner.scan_all()

    # Output results
    if args.output:
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Results written to: {args.output}")

    # Print summary
    print()
    print("SCAN SUMMARY")
    print("-" * 40)
    print(f"Total violations: {results['total_violations']}")
    print(f"  Errors:   {results['violations_by_severity'].get('error', 0)}")
    print(f"  Warnings: {results['violations_by_severity'].get('warning', 0)}")
    print(f"  Info:     {results['violations_by_severity'].get('info', 0)}")
    print()
    print(f"Orphaned documentation: {results['orphaned_documentation']}")
    print()

    compliance = results['docstring_compliance']
    print("Docstring Compliance:")
    print(f"  Total files scanned: {compliance['total_files']}")
    print(f"  Compliant files:     {compliance['compliant_files']}")
    print(f"  Compliance rate:     {compliance['compliance_rate']:.1%}")
    print()

    if args.verbose and results['violations']:
        print("VIOLATIONS:")
        print("-" * 40)
        for v in results['violations'][:20]:  # Show first 20
            print(f"  [{v['severity'].upper()}] {v['file']}:{v['line']}")
            print(f"    Found: {v['found']}")
            print(f"    Expected: {v['expected']}")
            print(f"    Suggestion: {v['suggestion']}")
            print()

    # CI mode: fail on errors
    if args.ci_mode:
        error_count = results['violations_by_severity'].get('error', 0)
        if error_count > 0:
            print(f"CI GATE FAILED: {error_count} error(s) found")
            return 1

    print("CI GATE PASSED" if args.ci_mode else "Scan complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())

