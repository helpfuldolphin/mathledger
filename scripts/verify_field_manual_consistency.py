#!/usr/bin/env python3
"""
PHASE II — NOT RUN IN PHASE I

Field Manual Consistency Verification Tool

This script ensures fm.tex agrees with:
  - Curriculum YAML (config/curriculum.yaml)
  - Manifest schemas (experiments/manifest.py)
  - Success metric definitions (experiments/slice_success_metrics.py)

Usage:
    python scripts/verify_field_manual_consistency.py [--verbose]

Exit codes:
    0 - All checks passed
    1 - Consistency errors found
    2 - File not found or parse error
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import yaml


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
FM_TEX_PATH = PROJECT_ROOT / "docs" / "fm.tex"
CURRICULUM_PATH = PROJECT_ROOT / "config" / "curriculum.yaml"
MANIFEST_PATH = PROJECT_ROOT / "experiments" / "manifest.py"
METRICS_PATH = PROJECT_ROOT / "experiments" / "slice_success_metrics.py"

# Expected slices documented in fm.tex (canonical set)
DOCUMENTED_SLICES = {
    "slice_debug_uplift",
    "slice_easy_fo",
    "slice_uplift_proto",
    "atoms4-depth4",
    "atoms4-depth5",
    "atoms5-depth6",
    "slice_medium",
    "first_organism_pl2_hard",
    "slice_hard",
}

# Expected success metrics documented in fm.tex
DOCUMENTED_METRICS = {
    "compute_goal_hit",
    "compute_sparse_success",
    "compute_chain_success",
    "compute_multi_goal_success",
}

# Expected LaTeX labels for section consistency
EXPECTED_LABELS = {
    "sec:introduction",
    "sec:u2-slices",
    "sec:u2-runner",
    "sec:determinism",
    "sec:governance",
    "sec:metrics",
    "sec:seed-discipline",
    "sec:workflows",
    "sec:evidence",
}

# Expected invariant labels
EXPECTED_INVARIANTS = {
    "inv:separation",
    "inv:monotonicity",
    "inv:gate-inactive",
    "inv:layers",
}

# Expected definition labels
EXPECTED_DEFINITIONS = {
    "def:slice",
    "def:slice-params",
    "def:gates",
    "def:telemetry",
    "def:hash-identity",
    "def:roots",
    "def:p-base",
    "def:p-rfl",
    "def:delta",
    "def:ci",
    "def:seed-schedule",
}


# ─────────────────────────────────────────────────────────────────────────────
# Utility Functions
# ─────────────────────────────────────────────────────────────────────────────

class ConsistencyError:
    """Represents a consistency violation."""
    
    def __init__(self, category: str, message: str, severity: str = "error"):
        self.category = category
        self.message = message
        self.severity = severity  # "error" or "warning"
    
    def __str__(self) -> str:
        return f"[{self.severity.upper()}] {self.category}: {self.message}"


def load_curriculum() -> Dict[str, Any]:
    """Load and parse curriculum.yaml."""
    if not CURRICULUM_PATH.exists():
        raise FileNotFoundError(f"Curriculum not found: {CURRICULUM_PATH}")
    
    with open(CURRICULUM_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_fm_tex() -> str:
    """Load fm.tex content."""
    if not FM_TEX_PATH.exists():
        raise FileNotFoundError(f"Field manual not found: {FM_TEX_PATH}")
    
    with open(FM_TEX_PATH, "r", encoding="utf-8") as f:
        return f.read()


def load_metrics_module() -> str:
    """Load slice_success_metrics.py content."""
    if not METRICS_PATH.exists():
        raise FileNotFoundError(f"Metrics module not found: {METRICS_PATH}")
    
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        return f.read()


def extract_latex_labels(tex_content: str) -> Set[str]:
    """Extract all \\label{...} from LaTeX content."""
    pattern = r"\\label\{([^}]+)\}"
    matches = re.findall(pattern, tex_content)
    return set(matches)


def extract_slice_names_from_tex(tex_content: str) -> Set[str]:
    """Extract slice names referenced in fm.tex."""
    # Match texttt{slice_name} patterns - handle both escaped and unescaped underscores
    # LaTeX uses \_ for underscores in texttt
    # More permissive pattern to catch all variants
    patterns = [
        r"\\texttt\{(slice[\\_a-z0-9]+)\}",  # slice_* patterns
        r"\\texttt\{(atoms[0-9]+-depth[0-9]+)\}",  # atoms*-depth* patterns  
        r"\\texttt\{(first[\\_a-z0-9]+)\}",  # first_organism_* patterns
    ]
    
    normalized = set()
    for pattern in patterns:
        matches = re.findall(pattern, tex_content)
        for match in matches:
            # Normalize by removing backslashes (LaTeX escapes)
            clean = match.replace("\\_", "_").replace("\\", "")
            normalized.add(clean)
    
    return normalized


def extract_metric_functions_from_tex(tex_content: str) -> Set[str]:
    """Extract metric function names from fm.tex code listings."""
    pattern = r"def (compute_[a-z_]+)\("
    matches = re.findall(pattern, tex_content)
    return set(matches)


def get_curriculum_slices(curriculum: Dict[str, Any]) -> Set[str]:
    """Extract slice names from curriculum.yaml."""
    slices = set()
    for system_data in curriculum.get("systems", {}).values():
        for slice_def in system_data.get("slices", []):
            if "name" in slice_def:
                slices.add(slice_def["name"])
    return slices


def get_metric_functions(metrics_content: str) -> Set[str]:
    """Extract function names from slice_success_metrics.py."""
    pattern = r"^def (compute_[a-z_]+)\("
    matches = re.findall(pattern, metrics_content, re.MULTILINE)
    return set(matches)


# ─────────────────────────────────────────────────────────────────────────────
# Consistency Checks
# ─────────────────────────────────────────────────────────────────────────────

def check_phase_ii_markers(tex_content: str) -> List[ConsistencyError]:
    """Verify PHASE II markers are present in required locations."""
    errors = []
    
    # Check for header marker
    if "PHASE II — NOT RUN IN PHASE I" not in tex_content:
        errors.append(ConsistencyError(
            "PHASE_MARKER",
            "Missing 'PHASE II — NOT RUN IN PHASE I' marker in header"
        ))
    
    # Check for phaseiilabel commands (should appear at start of major sections)
    phaseiilabel_count = tex_content.count("\\phaseiilabel")
    if phaseiilabel_count < 7:
        errors.append(ConsistencyError(
            "PHASE_MARKER",
            f"Expected at least 7 \\phaseiilabel commands, found {phaseiilabel_count}",
            severity="warning"
        ))
    
    return errors


def check_label_consistency(tex_content: str) -> List[ConsistencyError]:
    """Verify all expected LaTeX labels exist."""
    errors = []
    labels = extract_latex_labels(tex_content)
    
    # Check section labels
    missing_sections = EXPECTED_LABELS - labels
    for label in missing_sections:
        errors.append(ConsistencyError(
            "LABEL_MISSING",
            f"Missing section label: {label}"
        ))
    
    # Check invariant labels
    missing_invariants = EXPECTED_INVARIANTS - labels
    for label in missing_invariants:
        errors.append(ConsistencyError(
            "LABEL_MISSING",
            f"Missing invariant label: {label}"
        ))
    
    # Check definition labels
    missing_definitions = EXPECTED_DEFINITIONS - labels
    for label in missing_definitions:
        errors.append(ConsistencyError(
            "LABEL_MISSING",
            f"Missing definition label: {label}"
        ))
    
    return errors


def check_slice_consistency(tex_content: str, curriculum: Dict[str, Any]) -> List[ConsistencyError]:
    """Verify fm.tex documents all curriculum slices."""
    errors = []
    
    curriculum_slices = get_curriculum_slices(curriculum)
    documented_slices = extract_slice_names_from_tex(tex_content)
    
    # Check for undocumented slices
    undocumented = curriculum_slices - documented_slices
    for slice_name in undocumented:
        errors.append(ConsistencyError(
            "SLICE_UNDOCUMENTED",
            f"Slice '{slice_name}' exists in curriculum.yaml but not documented in fm.tex"
        ))
    
    # Check for orphaned documentation (slice in tex but not curriculum)
    orphaned = documented_slices - curriculum_slices
    for slice_name in orphaned:
        # Ignore known historical/special slices
        if slice_name not in {"slice_example", "slice_test"}:
            errors.append(ConsistencyError(
                "SLICE_ORPHANED",
                f"Slice '{slice_name}' documented in fm.tex but not in curriculum.yaml",
                severity="warning"
            ))
    
    return errors


def check_metric_consistency(tex_content: str, metrics_content: str) -> List[ConsistencyError]:
    """Verify fm.tex documents all success metric functions."""
    errors = []
    
    defined_metrics = get_metric_functions(metrics_content)
    documented_metrics = extract_metric_functions_from_tex(tex_content)
    
    # Check for undocumented metrics
    undocumented = defined_metrics - documented_metrics
    for metric_name in undocumented:
        errors.append(ConsistencyError(
            "METRIC_UNDOCUMENTED",
            f"Metric function '{metric_name}' exists in slice_success_metrics.py but not documented"
        ))
    
    # Check for documented metrics that don't exist
    phantom = documented_metrics - defined_metrics
    for metric_name in phantom:
        errors.append(ConsistencyError(
            "METRIC_PHANTOM",
            f"Metric function '{metric_name}' documented but not in slice_success_metrics.py",
            severity="warning"
        ))
    
    return errors


def check_slice_params(tex_content: str, curriculum: Dict[str, Any]) -> List[ConsistencyError]:
    """Verify slice parameters in fm.tex match curriculum.yaml."""
    errors = []
    
    # Extract table rows from fm.tex
    # Table format: name & role & atoms & depth & breadth & total & regime
    table_pattern = r"\\texttt\{([^}]+)\}\s*&\s*[^&]+&\s*(\d+)\s*&\s*(\d+)\s*&"
    table_entries = {}
    
    for match in re.finditer(table_pattern, tex_content):
        raw_name = match.group(1)
        atoms = int(match.group(2))
        depth = int(match.group(3))
        # Normalize name (remove LaTeX escapes)
        name = raw_name.replace("\\_", "_").replace("\\", "")
        table_entries[name] = {"atoms": atoms, "depth": depth}
    
    # Compare with curriculum
    for system_data in curriculum.get("systems", {}).values():
        for slice_def in system_data.get("slices", []):
            name = slice_def.get("name", "")
            params = slice_def.get("params", {})
            
            atoms = params.get("atoms")
            depth = params.get("depth_max")
            
            if atoms and depth and name in DOCUMENTED_SLICES:
                if name in table_entries:
                    doc_params = table_entries[name]
                    if doc_params["atoms"] != atoms:
                        errors.append(ConsistencyError(
                            "PARAMS_MISMATCH",
                            f"Slice '{name}' atoms mismatch: yaml={atoms}, tex={doc_params['atoms']}"
                        ))
                    if doc_params["depth"] != depth:
                        errors.append(ConsistencyError(
                            "PARAMS_MISMATCH",
                            f"Slice '{name}' depth mismatch: yaml={depth}, tex={doc_params['depth']}"
                        ))
    
    return errors


def check_determinism_contract(tex_content: str) -> List[ConsistencyError]:
    """Verify determinism contract elements are documented."""
    errors = []
    
    required_forbidden = [
        "datetime.now",
        "datetime.utcnow",
        "time.time",
        "uuid.uuid4",
        "random",
        "os.urandom",
    ]
    
    for forbidden in required_forbidden:
        if forbidden not in tex_content:
            errors.append(ConsistencyError(
                "DETERMINISM",
                f"Forbidden primitive '{forbidden}' not documented in determinism contract"
            ))
    
    # Check for hash identity formula
    if r"\mathrm{SHA256}" not in tex_content and "SHA256" not in tex_content:
        errors.append(ConsistencyError(
            "DETERMINISM",
            "Hash identity formula (SHA256) not documented"
        ))
    
    return errors


def check_evidence_section_empty(tex_content: str) -> List[ConsistencyError]:
    """Verify evidence interpretation section remains empty."""
    errors = []
    
    # Find the evidence section
    evidence_match = re.search(
        r"\\section\{Evidence Interpretation\}.*?(?=\\section|\\appendix|\\end\{document\})",
        tex_content,
        re.DOTALL
    )
    
    if evidence_match:
        section_content = evidence_match.group(0)
        
        # Should contain the "INTENTIONALLY EMPTY" marker
        if "INTENTIONALLY EMPTY" not in section_content:
            errors.append(ConsistencyError(
                "EVIDENCE_SECTION",
                "Evidence section must contain 'INTENTIONALLY EMPTY' marker"
            ))
        
        # Should NOT contain any actual analysis claims
        forbidden_terms = [
            r"\d+%\s+uplift",  # percentage uplift claims
            "statistically significant",
            "p < 0.05",
            "we found",
            "results show",
        ]
        
        for term in forbidden_terms:
            if re.search(term, section_content, re.IGNORECASE):
                errors.append(ConsistencyError(
                    "EVIDENCE_SECTION",
                    f"Evidence section contains forbidden content: '{term}'"
                ))
    else:
        errors.append(ConsistencyError(
            "EVIDENCE_SECTION",
            "Evidence Interpretation section not found"
        ))
    
    return errors


def check_terminology_consistency(tex_content: str) -> List[ConsistencyError]:
    """Verify canonical terminology is used consistently."""
    errors = []
    
    # Check for "Phase 2" outside of file paths and code listings
    # Remove code listings and file paths first
    text_only = re.sub(r"\\texttt\{[^}]+\}", "", tex_content)
    text_only = re.sub(r"\\lstinline[^{]*\{[^}]+\}", "", text_only)
    text_only = re.sub(r"\\begin\{lstlisting\}.*?\\end\{lstlisting\}", "", text_only, flags=re.DOTALL)
    
    # Now check for "Phase 2" in actual text (not paths)
    if re.search(r"Phase\s+2\b", text_only):
        errors.append(ConsistencyError(
            "TERMINOLOGY",
            "Found 'Phase 2' in text - use 'Phase II' consistently",
            severity="warning"
        ))
    
    return errors


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_all_checks(verbose: bool = False) -> Tuple[List[ConsistencyError], List[ConsistencyError]]:
    """Run all consistency checks."""
    all_errors = []
    all_warnings = []
    
    try:
        tex_content = load_fm_tex()
        curriculum = load_curriculum()
        metrics_content = load_metrics_module()
    except FileNotFoundError as e:
        return [ConsistencyError("FILE", str(e))], []
    
    checks = [
        ("Phase II Markers", lambda: check_phase_ii_markers(tex_content)),
        ("Label Consistency", lambda: check_label_consistency(tex_content)),
        ("Slice Consistency", lambda: check_slice_consistency(tex_content, curriculum)),
        ("Metric Consistency", lambda: check_metric_consistency(tex_content, metrics_content)),
        ("Slice Parameters", lambda: check_slice_params(tex_content, curriculum)),
        ("Determinism Contract", lambda: check_determinism_contract(tex_content)),
        ("Evidence Section", lambda: check_evidence_section_empty(tex_content)),
        ("Terminology", lambda: check_terminology_consistency(tex_content)),
    ]
    
    for check_name, check_fn in checks:
        if verbose:
            print(f"Running: {check_name}...")
        
        results = check_fn()
        for result in results:
            if result.severity == "error":
                all_errors.append(result)
            else:
                all_warnings.append(result)
    
    return all_errors, all_warnings


def main():
    parser = argparse.ArgumentParser(
        description="Verify fm.tex consistency with curriculum and metrics"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("PHASE II — Field Manual Consistency Verification")
    print("=" * 70)
    print()
    
    errors, warnings = run_all_checks(verbose=args.verbose)
    
    # Print warnings
    if warnings:
        print(f"\n⚠️  WARNINGS ({len(warnings)}):")
        print("-" * 40)
        for warning in warnings:
            print(f"  {warning}")
    
    # Print errors
    if errors:
        print(f"\n❌ ERRORS ({len(errors)}):")
        print("-" * 40)
        for error in errors:
            print(f"  {error}")
    
    # Summary
    print("\n" + "=" * 70)
    
    exit_code = 0
    if errors:
        print(f"❌ FAILED: {len(errors)} error(s) found")
        exit_code = 1
    elif args.strict and warnings:
        print(f"❌ FAILED (strict mode): {len(warnings)} warning(s) found")
        exit_code = 1
    elif warnings:
        print(f"✅ PASSED with {len(warnings)} warning(s)")
    else:
        print("✅ PASSED: All consistency checks passed")
    
    print("=" * 70)
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())

