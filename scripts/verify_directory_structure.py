#!/usr/bin/env python3
"""
PHASE II — DIRECTORY STRUCTURE LINTER

Agent: doc-ops-3 — Directory Cartographer
Purpose: Enforce directory boundaries and phase designation rules.

Usage:
    uv run python scripts/verify_directory_structure.py [--strict] [--fix-labels]
"""

import argparse
import ast
import re
import sys
from pathlib import Path
from typing import NamedTuple


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent

# Phase II markers (case-insensitive patterns)
PHASE_II_MARKERS = [
    r"PHASE\s*II",
    r"PHASE\s*2",
    r"Phase\s*II",
    r"Phase\s*2",
]

# Files/modules that are definitively Phase II
PHASE_II_FILES = {
    # Backend Phase II modules
    "backend/metrics/u2_analysis.py",
    "backend/metrics/statistical.py",
    "backend/runner/u2_runner.py",
    "backend/telemetry/u2_schema.py",
    "backend/security/u2_security.py",
    "backend/promotion/u2_evidence.py",
    # Experiment Phase II modules
    "experiments/run_uplift_u2.py",
    "experiments/u2_cross_slice_analysis.py",
    "experiments/u2_pipeline.py",
    "experiments/curriculum_hash_ledger.py",
    "experiments/curriculum_loader_v2.py",
    # Config
    "config/curriculum_uplift_phase2.yaml",
    # Scripts
    "scripts/validate_u2_environment.py",
    "scripts/build_u2_evidence_dossier.py",
    "scripts/proof_dag_u2_audit.py",
    # Analysis
    "analysis/u2_dynamics.py",
}

# Phase II import patterns (what Phase I code should NOT import)
PHASE_II_IMPORT_PATTERNS = [
    r"from\s+backend\.metrics\.u2_analysis",
    r"from\s+backend\.metrics\.statistical",
    r"from\s+backend\.runner\.u2_runner",
    r"from\s+backend\.telemetry\.u2_schema",
    r"from\s+backend\.security\.u2_security",
    r"from\s+backend\.promotion\.u2_evidence",
    r"from\s+experiments\.run_uplift_u2",
    r"from\s+experiments\.u2_",
    r"from\s+analysis\.u2_",
    r"from\s+tests\.phase2",
    r"import\s+backend\.metrics\.u2_analysis",
    r"import\s+backend\.runner\.u2_runner",
    r"import\s+tests\.phase2",
]

# Directories that MUST contain only Phase II content
PHASE_II_ONLY_DIRS = [
    "tests/phase2",
    "artifacts/phase_ii",
    "artifacts/u2",
    "experiments/synthetic_uplift",
]

# Directories that MUST contain only specific content types
DIRECTORY_CONTENT_RULES = {
    "backend/verification": {
        "description": "verification tools only",
        "allowed_patterns": [r".*verify.*\.py$", r".*validation.*\.py$", r"__pycache__"],
        "forbidden_patterns": [r".*runner.*\.py$", r".*experiment.*\.py$"],
    },
    "docs": {
        "description": "non-code artifacts only",
        "allowed_extensions": [
            ".md", ".pdf", ".tex", ".json", ".yaml", ".yml",  # Documents
            ".png", ".jpg", ".svg", ".gif",  # Images
            ".patch", ".diff",  # Patches
            ".txt", ".log",  # Text/logs
            ".jsonl",  # JSON lines (data/logs)
            ".sig",  # Signatures
            ".html", ".css",  # Static web
        ],
        "forbidden_extensions": [".py", ".js", ".ts", ".lean"],
    },
}

# Phase I directories (should NOT import Phase II)
PHASE_I_DIRS = [
    "curriculum",
    "derivation",
    "attestation",
    "rfl",
    "backend/axiom_engine",
    "backend/basis",
    "backend/bridge",
    "backend/causal",
    "backend/consensus",
    "backend/crypto",
    "backend/dag",
    "backend/fol_eq",
    "backend/frontier",
    "backend/generator",
    "backend/governance",
    "backend/ht",
    "backend/integration",
    "backend/ledger",
    "backend/logic",
    "backend/models",
    "backend/orchestrator",
    "backend/phase_ix",
    "backend/repro",
    "backend/rfl",
    "backend/testing",
    "backend/tools",
]


# -----------------------------------------------------------------------------
# Result Types
# -----------------------------------------------------------------------------

class LintError(NamedTuple):
    file: str
    line: int
    code: str
    message: str


class LintResult(NamedTuple):
    errors: list[LintError]
    warnings: list[LintError]
    files_checked: int


# -----------------------------------------------------------------------------
# Linting Functions
# -----------------------------------------------------------------------------

def check_phase_ii_label(filepath: Path) -> list[LintError]:
    """Check that Phase II files contain a phase marker in first 50 lines."""
    errors = []
    rel_path = filepath.relative_to(PROJECT_ROOT).as_posix()
    
    # Only check known Phase II files
    if rel_path not in PHASE_II_FILES:
        # Also check if in Phase II only directories
        in_phase_ii_dir = any(rel_path.startswith(d) for d in PHASE_II_ONLY_DIRS)
        if not in_phase_ii_dir:
            return errors
    
    try:
        content = filepath.read_text(encoding="utf-8", errors="replace")
        lines = content.split("\n")[:50]
        header = "\n".join(lines)
        
        has_marker = any(re.search(pattern, header) for pattern in PHASE_II_MARKERS)
        
        if not has_marker:
            errors.append(LintError(
                file=rel_path,
                line=1,
                code="PH2-001",
                message="Phase II file missing phase marker in first 50 lines. Add '# PHASE II' comment.",
            ))
    except Exception as e:
        errors.append(LintError(
            file=rel_path,
            line=0,
            code="PH2-ERR",
            message=f"Could not read file: {e}",
        ))
    
    return errors


def check_phase_ii_imports(filepath: Path) -> list[LintError]:
    """Check that Phase I files do not import Phase II modules."""
    errors = []
    rel_path = filepath.relative_to(PROJECT_ROOT).as_posix()
    
    # Skip Phase II files
    if rel_path in PHASE_II_FILES:
        return errors
    
    # Skip files in Phase II directories
    if any(rel_path.startswith(d) for d in PHASE_II_ONLY_DIRS):
        return errors
    
    # Only check Phase I directories
    in_phase_i = any(rel_path.startswith(d) for d in PHASE_I_DIRS)
    if not in_phase_i:
        return errors
    
    try:
        content = filepath.read_text(encoding="utf-8", errors="replace")
        
        for line_num, line in enumerate(content.split("\n"), start=1):
            for pattern in PHASE_II_IMPORT_PATTERNS:
                if re.search(pattern, line):
                    errors.append(LintError(
                        file=rel_path,
                        line=line_num,
                        code="PH2-002",
                        message=f"Phase I code importing Phase II module: {line.strip()}",
                    ))
    except Exception:
        pass  # Silently skip unreadable files for import checks
    
    return errors


def check_directory_purity(directory: Path) -> list[LintError]:
    """Check that directories contain only their allowed content."""
    errors = []
    rel_dir = directory.relative_to(PROJECT_ROOT).as_posix()
    
    if rel_dir not in DIRECTORY_CONTENT_RULES:
        return errors
    
    rules = DIRECTORY_CONTENT_RULES[rel_dir]
    
    if not directory.exists():
        return errors
    
    for item in directory.rglob("*"):
        if item.is_dir():
            continue
        
        rel_item = item.relative_to(PROJECT_ROOT).as_posix()
        
        # Check allowed extensions
        if "allowed_extensions" in rules:
            ext = item.suffix.lower()
            if ext and ext not in rules["allowed_extensions"]:
                errors.append(LintError(
                    file=rel_item,
                    line=0,
                    code="DIR-001",
                    message=f"File with extension '{ext}' not allowed in {rel_dir}/ ({rules['description']})",
                ))
        
        # Check forbidden extensions
        if "forbidden_extensions" in rules:
            ext = item.suffix.lower()
            if ext in rules["forbidden_extensions"]:
                errors.append(LintError(
                    file=rel_item,
                    line=0,
                    code="DIR-002",
                    message=f"File with extension '{ext}' forbidden in {rel_dir}/ ({rules['description']})",
                ))
        
        # Check allowed patterns
        if "allowed_patterns" in rules:
            matches_allowed = any(
                re.match(p, item.name) for p in rules["allowed_patterns"]
            )
            if not matches_allowed and rules.get("forbidden_patterns"):
                matches_forbidden = any(
                    re.match(p, item.name) for p in rules["forbidden_patterns"]
                )
                if matches_forbidden:
                    errors.append(LintError(
                        file=rel_item,
                        line=0,
                        code="DIR-003",
                        message=f"File matches forbidden pattern in {rel_dir}/ ({rules['description']})",
                    ))
    
    return errors


def check_phase_ii_directory_contents(directory: Path) -> list[LintError]:
    """Check that Phase II only directories don't contain Phase I code."""
    errors = []
    
    if not directory.exists():
        return errors
    
    rel_dir = directory.relative_to(PROJECT_ROOT).as_posix()
    
    # Check each Python file in Phase II directories
    for py_file in directory.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue
        
        rel_file = py_file.relative_to(PROJECT_ROOT).as_posix()
        
        try:
            content = py_file.read_text(encoding="utf-8", errors="replace")
            lines = content.split("\n")[:50]
            header = "\n".join(lines)
            
            has_marker = any(re.search(pattern, header) for pattern in PHASE_II_MARKERS)
            
            if not has_marker:
                errors.append(LintError(
                    file=rel_file,
                    line=1,
                    code="PH2-003",
                    message=f"File in Phase II directory ({rel_dir}/) missing phase marker",
                ))
        except Exception:
            pass
    
    return errors


def lint_all() -> LintResult:
    """Run all linting checks."""
    errors = []
    warnings = []
    files_checked = 0
    
    # Check Phase II files for labels
    for rel_path in PHASE_II_FILES:
        filepath = PROJECT_ROOT / rel_path
        if filepath.exists():
            errors.extend(check_phase_ii_label(filepath))
            files_checked += 1
    
    # Check Phase I files for forbidden imports
    for phase_i_dir in PHASE_I_DIRS:
        dir_path = PROJECT_ROOT / phase_i_dir
        if not dir_path.exists():
            continue
        
        for py_file in dir_path.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            errors.extend(check_phase_ii_imports(py_file))
            files_checked += 1
    
    # Check directory content rules
    for rel_dir in DIRECTORY_CONTENT_RULES:
        dir_path = PROJECT_ROOT / rel_dir
        errors.extend(check_directory_purity(dir_path))
    
    # Check Phase II only directories
    for rel_dir in PHASE_II_ONLY_DIRS:
        dir_path = PROJECT_ROOT / rel_dir
        if dir_path.exists():
            errors.extend(check_phase_ii_directory_contents(dir_path))
    
    return LintResult(errors=errors, warnings=warnings, files_checked=files_checked)


def add_phase_ii_label(filepath: Path) -> bool:
    """Add Phase II label to a file if missing."""
    try:
        content = filepath.read_text(encoding="utf-8")
        
        # Check if already has marker
        lines = content.split("\n")[:50]
        header = "\n".join(lines)
        if any(re.search(pattern, header) for pattern in PHASE_II_MARKERS):
            return False
        
        # Determine comment style based on extension
        ext = filepath.suffix.lower()
        if ext in [".py"]:
            marker = '# PHASE II — U2 UPLIFT EXPERIMENT\n'
        elif ext in [".yaml", ".yml"]:
            marker = '# PHASE II — U2 UPLIFT EXPERIMENT\n'
        elif ext in [".md"]:
            marker = '<!-- PHASE II — U2 UPLIFT EXPERIMENT -->\n\n'
        else:
            return False
        
        # Handle shebang
        if content.startswith("#!"):
            first_newline = content.index("\n")
            new_content = content[:first_newline + 1] + marker + content[first_newline + 1:]
        else:
            new_content = marker + content
        
        filepath.write_text(new_content, encoding="utf-8")
        return True
    except Exception:
        return False


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Verify MathLedger directory structure and phase boundaries."
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with error code if any issues found",
    )
    parser.add_argument(
        "--fix-labels",
        action="store_true",
        help="Automatically add Phase II labels to files missing them",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only output errors",
    )
    args = parser.parse_args()
    
    if not args.quiet:
        print("=" * 60)
        print("PHASE II — DIRECTORY STRUCTURE LINTER")
        print("Agent: doc-ops-3")
        print("=" * 60)
        print()
    
    result = lint_all()
    
    # Apply fixes if requested
    fixed_count = 0
    if args.fix_labels:
        for error in result.errors:
            if error.code == "PH2-001" or error.code == "PH2-003":
                filepath = PROJECT_ROOT / error.file
                if add_phase_ii_label(filepath):
                    fixed_count += 1
                    if not args.quiet:
                        print(f"  FIXED: {error.file}")
        
        # Re-run lint after fixes
        if fixed_count > 0:
            result = lint_all()
    
    # Report errors
    if result.errors:
        print(f"\n{'ERRORS'.center(60, '-')}\n")
        for error in result.errors:
            loc = f"{error.file}:{error.line}" if error.line else error.file
            print(f"  [{error.code}] {loc}")
            print(f"         {error.message}")
            print()
    
    # Report warnings
    if result.warnings and not args.quiet:
        print(f"\n{'WARNINGS'.center(60, '-')}\n")
        for warning in result.warnings:
            loc = f"{warning.file}:{warning.line}" if warning.line else warning.file
            print(f"  [{warning.code}] {loc}")
            print(f"         {warning.message}")
            print()
    
    # Summary
    if not args.quiet:
        print(f"\n{'SUMMARY'.center(60, '-')}")
        print(f"  Files checked: {result.files_checked}")
        print(f"  Errors: {len(result.errors)}")
        print(f"  Warnings: {len(result.warnings)}")
        if fixed_count > 0:
            print(f"  Fixed: {fixed_count}")
        print()
    
    if result.errors:
        if args.strict:
            print("FAIL: Directory structure violations detected.")
            sys.exit(1)
        else:
            print("WARN: Directory structure violations detected (non-strict mode).")
            sys.exit(0)
    else:
        if not args.quiet:
            print("PASS: Directory structure is compliant.")
        sys.exit(0)


if __name__ == "__main__":
    main()

