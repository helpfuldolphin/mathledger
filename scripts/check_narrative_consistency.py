#!/usr/bin/env python3
"""
check_narrative_consistency.py — Narrative Consistency Checker

MISSION: Ensure narrative consistency across all human-facing documentation.

ABSOLUTE SAFEGUARDS:
  - No uplift claims unless Phase II experiments completed with statistical significance.
  - No deviation from governance vocabulary.
  - No unauthorized conceptual innovation.

This script scans documentation for terminology drift and alignment with canonical definitions.

Usage:
    python scripts/check_narrative_consistency.py [--fix] [--verbose]
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# ==============================================================================
# CANONICAL TERMINOLOGY DEFINITIONS
# ==============================================================================

CANONICAL_DEFINITIONS = {
    "RFL": {
        "full_form": "Reflexive Formal Learning",
        "description": "The core learning framework using verifiable feedback from formal proofs",
        "prohibited_alternatives": ["RLVF", "Reflective Feedback Loop"],
        "required_context": None,
    },
    "Phase_II": {
        "canonical_form": "Phase II",
        "prohibited_alternatives": ["Phase 2", "Phase-II", "phase ii", "PHASE-II"],
        "description": "The experimental phase for uplift measurement (not yet run)",
    },
    "RLHF": {
        "usage": "only_in_contrast",
        "description": "Reinforcement Learning from Human Feedback — referenced only to contrast with RFL",
        "required_context": r"moving from|RLHF\s*→|vs\.?\s*RFL|not.*RLHF",
    },
    "RLPF": {
        "usage": "only_in_contrast",
        "description": "Reinforcement Learning from Process Feedback — referenced only to contrast with RFL",
        "required_context": r"moving from|RLPF\s*→|vs\.?\s*RFL|not.*RLPF|No RLHF|No.*proxy",
    },
    "uplift": {
        "safeguard": True,
        "description": "Measurable improvement — NO CLAIMS until Phase II completed",
        "prohibited_patterns": [
            r"(?<!no )uplift\s+(achieved|demonstrated|proven|confirmed|observed)",
            r"(?<!not )(?<!no )(?<!without )uplift\s+of\s+\d+",
            r"shows?\s+uplift",
            r"uplift\s+result",
        ],
        "allowed_patterns": [
            r"no uplift",
            r"uplift\s+plan",
            r"uplift\s+experiment",
            r"uplift\s+gate",
            r"uplift\s+slice",
            r"measure.*uplift",
            r"test.*uplift",
            r"Phase II.*uplift",
            r"Phase I.*no uplift",
            r"negative control",
        ],
    },
    "slices": {
        "canonical_names": [
            "slice_debug_uplift",
            "slice_easy_fo",
            "slice_uplift_proto",
            "slice_uplift_goal",
            "slice_uplift_sparse",
            "slice_uplift_tree",
            "slice_uplift_dependency",
            "slice_medium",
            "slice_hard",
            "atoms4-depth4",
            "atoms4-depth5",
            "atoms5-depth6",
            "first_organism_pl2_hard",
        ],
        "deprecated_patterns": [
            r"slice\s*[ABCD](?![a-z])",  # Slice A, Slice B, etc.
            r"slices?\s+A\s*/\s*B\s*/\s*C\s*/\s*D",
        ],
        "description": "Curriculum slices are named descriptively, not alphabetically",
    },
}

ABSOLUTE_SAFEGUARDS_TEXT = """Absolute Safeguards:
  - No uplift claims.
  - No deviation from governance vocabulary.
  - No unauthorized conceptual innovation."""

DOCUMENT_PATHS = [
    "paper/main.tex",
    "paper/sections/*.tex",
    "docs/*.md",
    "README.md",
    "AGENTS.md",
    "VSD*.md",
    "governance_verdict.md",
    "RFL_*.md",
    "*_PHASE_*.md",
]

EXCLUDED_PATHS = [
    "node_modules",
    ".git",
    "__pycache__",
    "*.pyc",
    "*.jsonl",
    "results/",
    "artifacts/",
    "ui/",
    "apps/",
    "dist/",
]


@dataclass
class Issue:
    """Represents a narrative consistency issue."""

    file: str
    line: int
    category: str
    severity: Literal["error", "warning", "info"]
    message: str
    context: str
    suggestion: str | None = None


@dataclass
class AuditResult:
    """Results from the narrative consistency audit."""

    issues: list[Issue] = field(default_factory=list)
    files_scanned: int = 0
    errors: int = 0
    warnings: int = 0
    info: int = 0

    def add_issue(self, issue: Issue) -> None:
        self.issues.append(issue)
        if issue.severity == "error":
            self.errors += 1
        elif issue.severity == "warning":
            self.warnings += 1
        else:
            self.info += 1


class NarrativeConsistencyChecker:
    """Checks documentation for narrative consistency with canonical definitions."""

    def __init__(self, repo_root: Path, verbose: bool = False):
        self.repo_root = repo_root
        self.verbose = verbose
        self.result = AuditResult()

    def scan_all(self) -> AuditResult:
        """Scan all relevant documentation files."""
        files_to_scan = self._collect_files()

        for file_path in files_to_scan:
            self._scan_file(file_path)
            self.result.files_scanned += 1

        return self.result

    def _collect_files(self) -> list[Path]:
        """Collect all files matching the document patterns."""
        files = []

        for pattern in DOCUMENT_PATHS:
            if "*" in pattern:
                # Glob pattern
                files.extend(self.repo_root.glob(pattern))
            else:
                # Direct path
                file_path = self.repo_root / pattern
                if file_path.exists():
                    files.append(file_path)

        # Filter out excluded paths
        filtered = []
        for f in files:
            exclude = False
            for excl in EXCLUDED_PATHS:
                if excl.replace("*", "") in str(f):
                    exclude = True
                    break
            if not exclude and f.is_file():
                filtered.append(f)

        return sorted(set(filtered))

    def _scan_file(self, file_path: Path) -> None:
        """Scan a single file for narrative consistency issues."""
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            if self.verbose:
                print(f"  [SKIP] Could not read {file_path}: {e}")
            return

        lines = content.split("\n")
        rel_path = str(file_path.relative_to(self.repo_root))

        for line_num, line in enumerate(lines, start=1):
            self._check_rfl_terminology(rel_path, line_num, line)
            self._check_phase_terminology(rel_path, line_num, line)
            self._check_uplift_claims(rel_path, line_num, line)
            self._check_rlhf_rlpf_context(rel_path, line_num, line, lines, line_num - 1)
            self._check_slice_naming(rel_path, line_num, line)

    def _check_rfl_terminology(
        self, file: str, line_num: int, line: str
    ) -> None:
        """Check RFL terminology usage."""
        rfl_def = CANONICAL_DEFINITIONS["RFL"]

        # Check for prohibited alternatives
        for alt in rfl_def["prohibited_alternatives"]:
            pattern = rf"\b{re.escape(alt)}\b"
            if re.search(pattern, line, re.IGNORECASE):
                # Special case: "Reflective Feedback Loop" in methodology section is
                # sometimes used historically but should be flagged as warning
                severity = "warning" if alt == "Reflective Feedback Loop" else "error"
                self.result.add_issue(
                    Issue(
                        file=file,
                        line=line_num,
                        category="RFL_Terminology",
                        severity=severity,
                        message=f"Prohibited term '{alt}' found",
                        context=line.strip()[:100],
                        suggestion=f"Use 'RFL' or 'Reflexive Formal Learning' instead",
                    )
                )

    def _check_phase_terminology(
        self, file: str, line_num: int, line: str
    ) -> None:
        """Check Phase II terminology consistency."""
        phase_def = CANONICAL_DEFINITIONS["Phase_II"]

        for alt in phase_def["prohibited_alternatives"]:
            # Case-sensitive check for exact prohibited forms
            if alt in line:
                self.result.add_issue(
                    Issue(
                        file=file,
                        line=line_num,
                        category="Phase_Terminology",
                        severity="warning",
                        message=f"Non-canonical phase naming '{alt}' found",
                        context=line.strip()[:100],
                        suggestion="Use 'Phase II' (Roman numerals, capitalized)",
                    )
                )

    def _check_uplift_claims(self, file: str, line_num: int, line: str) -> None:
        """Check for unauthorized uplift claims."""
        uplift_def = CANONICAL_DEFINITIONS["uplift"]
        line_lower = line.lower()

        if "uplift" not in line_lower:
            return

        # Check if line contains allowed patterns (these are OK)
        for allowed in uplift_def["allowed_patterns"]:
            if re.search(allowed, line_lower):
                return

        # Check for prohibited patterns
        for prohibited in uplift_def["prohibited_patterns"]:
            if re.search(prohibited, line_lower):
                self.result.add_issue(
                    Issue(
                        file=file,
                        line=line_num,
                        category="Uplift_Claim",
                        severity="error",
                        message="Potential unauthorized uplift claim detected",
                        context=line.strip()[:100],
                        suggestion="Uplift claims require Phase II completion with statistical significance",
                    )
                )
                return

    def _check_rlhf_rlpf_context(
        self,
        file: str,
        line_num: int,
        line: str,
        all_lines: list[str],
        line_idx: int,
    ) -> None:
        """Check that RLHF/RLPF are only mentioned in contrast context."""
        for term in ["RLHF", "RLPF"]:
            if term not in line:
                continue

            term_def = CANONICAL_DEFINITIONS[term]
            context_pattern = term_def["required_context"]

            # Check current line and surrounding context (±2 lines)
            context_window = ""
            for i in range(max(0, line_idx - 2), min(len(all_lines), line_idx + 3)):
                context_window += all_lines[i] + " "

            if not re.search(context_pattern, context_window, re.IGNORECASE):
                self.result.add_issue(
                    Issue(
                        file=file,
                        line=line_num,
                        category="RLHF_RLPF_Context",
                        severity="warning",
                        message=f"'{term}' mentioned without contrast context",
                        context=line.strip()[:100],
                        suggestion=f"'{term}' should appear in context contrasting with RFL (e.g., 'moving from {term} to RFL')",
                    )
                )

    def _check_slice_naming(self, file: str, line_num: int, line: str) -> None:
        """Check for deprecated slice naming conventions."""
        slice_def = CANONICAL_DEFINITIONS["slices"]

        for pattern in slice_def["deprecated_patterns"]:
            if re.search(pattern, line):
                self.result.add_issue(
                    Issue(
                        file=file,
                        line=line_num,
                        category="Slice_Naming",
                        severity="warning",
                        message="Deprecated alphabetic slice naming found",
                        context=line.strip()[:100],
                        suggestion=f"Use descriptive slice names: {', '.join(slice_def['canonical_names'][:3])}...",
                    )
                )


def generate_report(result: AuditResult, repo_root: Path) -> str:
    """Generate a markdown report of the audit results."""
    report = []
    report.append("# Narrative Consistency Audit Report\n")
    report.append(f"**Generated**: {__import__('datetime').datetime.now().isoformat()}\n")
    report.append(f"**Files Scanned**: {result.files_scanned}\n")
    report.append(f"**Total Issues**: {len(result.issues)}\n")
    report.append(f"  - Errors: {result.errors}\n")
    report.append(f"  - Warnings: {result.warnings}\n")
    report.append(f"  - Info: {result.info}\n")
    report.append("\n---\n")

    # Summary by category
    report.append("## Summary by Category\n")
    categories: dict[str, list[Issue]] = {}
    for issue in result.issues:
        if issue.category not in categories:
            categories[issue.category] = []
        categories[issue.category].append(issue)

    for cat, issues in sorted(categories.items()):
        errors = sum(1 for i in issues if i.severity == "error")
        warnings = sum(1 for i in issues if i.severity == "warning")
        report.append(f"- **{cat}**: {len(issues)} issues ({errors} errors, {warnings} warnings)\n")

    report.append("\n---\n")

    # Detailed issues
    report.append("## Detailed Issues\n")

    if result.errors > 0:
        report.append("### ❌ Errors (Must Fix)\n")
        for issue in result.issues:
            if issue.severity == "error":
                report.append(f"\n**{issue.file}:{issue.line}** — {issue.category}\n")
                report.append(f"> {issue.message}\n")
                report.append(f"```\n{issue.context}\n```\n")
                if issue.suggestion:
                    report.append(f"💡 *Suggestion*: {issue.suggestion}\n")

    if result.warnings > 0:
        report.append("\n### ⚠️ Warnings (Should Review)\n")
        for issue in result.issues:
            if issue.severity == "warning":
                report.append(f"\n**{issue.file}:{issue.line}** — {issue.category}\n")
                report.append(f"> {issue.message}\n")
                report.append(f"```\n{issue.context}\n```\n")
                if issue.suggestion:
                    report.append(f"💡 *Suggestion*: {issue.suggestion}\n")

    report.append("\n---\n")

    # Canonical definitions reference
    report.append("## Canonical Terminology Reference\n")
    report.append("""
| Term | Canonical Form | Notes |
|------|----------------|-------|
| RFL | Reflexive Formal Learning | NOT 'RLVF' or 'Reflective Feedback Loop' |
| Phase II | Phase II | Roman numerals, capitalized. NOT 'Phase 2' |
| RLHF/RLPF | (contrast only) | Only mention when contrasting with RFL |
| Uplift | (no claims) | Requires Phase II completion with statistical significance |
| Slices | Descriptive names | e.g., slice_uplift_goal, NOT 'Slice A' |
""")

    report.append("\n---\n")
    report.append("## Absolute Safeguards\n")
    report.append("```\n")
    report.append(ABSOLUTE_SAFEGUARDS_TEXT)
    report.append("\n```\n")

    return "".join(report)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Check narrative consistency across MathLedger documentation"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Attempt to auto-fix simple issues (not implemented)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="docs/NARRATIVE_CONSISTENCY_REPORT.md",
        help="Output report path",
    )

    args = parser.parse_args()

    # Find repo root
    script_path = Path(__file__).resolve()
    repo_root = script_path.parent.parent

    print("=" * 70)
    print("NARRATIVE CONSISTENCY CHECKER — doc-ops-5")
    print("=" * 70)
    print(f"Repository: {repo_root}")
    print()

    checker = NarrativeConsistencyChecker(repo_root, verbose=args.verbose)
    result = checker.scan_all()

    print(f"Files scanned: {result.files_scanned}")
    print(f"Issues found:  {len(result.issues)}")
    print(f"  - Errors:    {result.errors}")
    print(f"  - Warnings:  {result.warnings}")
    print(f"  - Info:      {result.info}")
    print()

    # Generate and save report
    report = generate_report(result, repo_root)
    output_path = repo_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"Report saved to: {output_path}")

    # Print summary
    if result.errors > 0:
        print()
        print("❌ NARRATIVE DRIFT DETECTED — Errors require attention")
        for issue in result.issues:
            if issue.severity == "error":
                print(f"  • {issue.file}:{issue.line}: {issue.message}")
        return 1
    elif result.warnings > 0:
        print()
        print("⚠️  WARNINGS DETECTED — Review recommended")
        return 0
    else:
        print()
        print("✅ NARRATIVE CONSISTENCY VERIFIED — All documents aligned")
        return 0


if __name__ == "__main__":
    sys.exit(main())

