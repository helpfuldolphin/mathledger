#!/usr/bin/env python3
"""
PHASE II — NOT RUN IN PHASE I

Field Manual Canonicalizer & Intelligence System

This tool extracts structural elements from fm.tex and produces:
  - A canonical JSON representation (fm_canonical.json)
  - A deterministic signature hash
  - Label ordering and uniqueness validation
  - Cross-reference audit
  - Drift detection

Usage:
    python scripts/fm_canonicalize.py [command] [options]

Commands:
    canonicalize    Extract and serialize fm.tex structure to JSON
    audit-refs      Audit cross-references for dangling refs
    drift-check     Check for drift against determinism contract
    full            Run all checks and produce canonical output

ABSOLUTE SAFEGUARDS:
    - This tool is DESCRIPTIVE, not NORMATIVE
    - No modifications to fm.tex contents
    - No inference or claims regarding uplift
"""

import argparse
import hashlib
import json
import re
import sys
from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
FM_TEX_PATH = PROJECT_ROOT / "docs" / "fm.tex"
FM_CANONICAL_PATH = PROJECT_ROOT / "docs" / "fm_canonical.json"
FM_SPEC_CONTRACT_PATH = PROJECT_ROOT / "artifacts" / "spec" / "fm_spec_contract.json"
LABEL_DRIFT_PATH = PROJECT_ROOT / "artifacts" / "spec" / "label_drift_summary.json"

# ─────────────────────────────────────────────────────────────────────────────
# Spec Contract API Definition (Frozen Structure for Other Agents)
# ─────────────────────────────────────────────────────────────────────────────
#
# The Spec Contract (fm_spec_contract.json) is a STABLE API that other agents
# (I, J, N, O) can rely on for:
#   - Label and invariant mapping
#   - Definition lookups
#   - Cross-check drift detection
#
# CONTRACT SHAPE (frozen — changes require explicit human review):
#   {
#     "version": "1.0",                      # Semver-like version
#     "signature_hash": "<sha256>",          # 64-char hex, deterministic
#     "definitions": [                       # All fm.tex definitions
#       {"label": "def:*", "name": "...", "section": "N.M"}
#     ],
#     "invariants": [                        # All fm.tex invariants
#       {"label": "inv:*", "name": "...", "section": "N.M"}
#     ],
#     "metrics_section": "N",                # Section number for metrics
#     "uplift_section": "N.M",               # Section number for uplift/evidence
#     "phase_section": "N",                  # Section number for phase intro
#     "consumers": {                         # Label → external references map
#       "<label>": {"consumers": {"<doc>": <count>}, "total_refs": N}
#     },
#     "label_coverage": {                    # Coverage statistics
#       "labels_with_refs": N,
#       "total_labels": M,
#       "coverage_pct": P.PP
#     }
#   }
#
# CI INTEGRATION:
#   uv run python scripts/fm_canonicalize.py full
#   uv run python scripts/fm_canonicalize.py label-drift --strict
#
# AGENT CONSUMPTION:
#   - Load fm_spec_contract.json to get label/definition mappings
#   - Load cross_check_summary.json to detect doc-level drift
#   - Load label_drift_summary.json to detect orphaned labels
# ─────────────────────────────────────────────────────────────────────────────

# All keys that MUST be present in fm_spec_contract.json
SPEC_CONTRACT_REQUIRED_KEYS = [
    "version",
    "signature_hash",
    "definitions",
    "invariants",
    "metrics_section",
    "uplift_section",
    "phase_section",
    "consumers",
    "label_coverage",
]

# Label drift summary keys (for validation)
LABEL_DRIFT_REQUIRED_KEYS = [
    "timestamp",
    "fm_signature_hash",
    "fm_only_labels",
    "external_only_labels",
    "well_connected_labels",
    "coverage_pct",
]

# Documentation files for cross-checking (read-only)
CROSS_CHECK_DOCS = [
    PROJECT_ROOT / "docs" / "PHASE2_RFL_UPLIFT_PLAN.md",
    PROJECT_ROOT / "experiments" / "PHASE1_RFL_SUMMARY.md",
    PROJECT_ROOT / "RFL_UPLIFT_THEORY.md",
]

# Core terminology that should be consistently used across docs
CORE_TERMINOLOGY = [
    "RFL",
    "H_t",
    "R_t",
    "U_t",
    "abstention",
    "uplift",
    "determinism",
    "Phase II",
    "baseline",
    "policy",
]

# Determinism contract forbidden primitives (from docs/DETERMINISM_CONTRACT.md)
DETERMINISM_FORBIDDEN_PRIMITIVES = [
    "datetime.now",
    "datetime.utcnow", 
    "time.time",
    "uuid.uuid4",
    "os.urandom",
]

# Canonical terminology mappings (incorrect -> correct)
TERMINOLOGY_RULES = {
    r"\bPhase\s+2\b": "Phase II",
    r"\bphase\s+2\b": "Phase II",
    r"\bH_t\b(?![}$])": r"$H_t$",  # Should be in math mode
    r"\bR_t\b(?![}$])": r"$R_t$",
    r"\bU_t\b(?![}$])": r"$U_t$",
}

# Expected section ordering (for structure validation)
EXPECTED_SECTION_ORDER = [
    "introduction",
    "u2-slices",
    "u2-runner",
    "determinism",
    "governance",
    "metrics",
    "seed-discipline",
    "workflows",
    "evidence",
]


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LabelInfo:
    """Information about a LaTeX label."""
    name: str
    line_number: int
    label_type: str  # 'section', 'definition', 'invariant', 'table', 'other'
    context: str  # Surrounding text for disambiguation


@dataclass
class DefinitionInfo:
    """Information about a LaTeX definition environment."""
    label: str
    name: str
    content: str
    line_start: int
    line_end: int


@dataclass 
class InvariantInfo:
    """Information about a LaTeX invariant environment."""
    label: str
    name: str
    content: str
    line_start: int
    line_end: int


@dataclass
class FormulaInfo:
    """Information about a LaTeX formula/equation."""
    formula_type: str  # 'inline', 'display', 'align', 'equation'
    content: str
    line_number: int
    label: Optional[str] = None


@dataclass
class RefInfo:
    """Information about a LaTeX reference."""
    ref_type: str  # 'ref', 'eqref', 'pageref'
    target: str
    line_number: int


@dataclass
class CrossRefAuditResult:
    """Result of cross-reference audit."""
    dangling_refs: List[Tuple[str, int, str]]  # (target, line, ref_type)
    unused_labels: List[Tuple[str, int]]  # (label, line)
    suggestions: List[str]


@dataclass
class DriftCheckResult:
    """Result of drift detection."""
    determinism_violations: List[Tuple[int, str, str]]  # (line, violation, context)
    terminology_issues: List[Tuple[int, str, str, str]]  # (line, found, expected, context)
    structure_issues: List[str]


@dataclass
class CanonicalRepresentation:
    """Canonical JSON-serializable representation of fm.tex."""
    version: str = "1.0.0"
    source_path: str = ""
    source_hash: str = ""
    labels: List[Dict[str, Any]] = field(default_factory=list)
    definitions: List[Dict[str, Any]] = field(default_factory=list)
    invariants: List[Dict[str, Any]] = field(default_factory=list)
    formulas: List[Dict[str, Any]] = field(default_factory=list)
    sections: List[Dict[str, Any]] = field(default_factory=list)
    signature_hash: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SpecContract:
    """Distilled contract from fm_canonical.json for other agents."""
    version: str = "1.0"
    signature_hash: str = ""
    definitions: List[Dict[str, str]] = field(default_factory=list)
    invariants: List[Dict[str, str]] = field(default_factory=list)
    metrics_section: Optional[str] = None
    uplift_section: Optional[str] = None
    phase_section: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "signature_hash": self.signature_hash,
            "definitions": self.definitions,
            "invariants": self.invariants,
            "metrics_section": self.metrics_section,
            "uplift_section": self.uplift_section,
            "phase_section": self.phase_section,
        }


@dataclass
class CrossCheckResult:
    """Result of cross-checking fm.tex against external docs."""
    doc_path: str
    missing_labels: List[str]  # Labels referenced in doc but not in fm.tex
    undefined_terms: List[str]  # Core terms missing from doc
    referenced_definitions: List[str]  # Definitions referenced correctly
    status: str  # "pass", "warn", "fail"
    details: List[str]


@dataclass
class CrossCheckSummary:
    """Summary of all cross-check results."""
    timestamp: str
    fm_signature_hash: str
    docs_checked: int
    docs_passed: int
    docs_warned: int
    docs_failed: int
    results: List[Dict[str, Any]]


@dataclass
class LabelConsumer:
    """Tracks which external documents reference a label."""
    label: str
    consumers: Dict[str, int]  # {doc_path: reference_count}
    total_refs: int


@dataclass
class LabelDriftSummary:
    """Summary of label drift between fm.tex and external docs."""
    timestamp: str
    fm_signature_hash: str
    fm_only_labels: List[str]  # Defined in fm.tex but not referenced externally
    external_only_labels: List[str]  # Referenced externally but not defined in fm.tex
    well_connected_labels: List[str]  # Defined and referenced
    coverage_pct: float  # Percentage of fm labels that are referenced externally


# ─────────────────────────────────────────────────────────────────────────────
# Extraction Functions
# ─────────────────────────────────────────────────────────────────────────────

def load_fm_tex() -> Tuple[str, List[str]]:
    """Load fm.tex content and return both raw string and lines."""
    if not FM_TEX_PATH.exists():
        raise FileNotFoundError(f"Field manual not found: {FM_TEX_PATH}")
    
    with open(FM_TEX_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    
    lines = content.split("\n")
    return content, lines


def compute_file_hash(content: str) -> str:
    """Compute SHA-256 hash of file content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def extract_labels(content: str, lines: List[str]) -> List[LabelInfo]:
    """Extract all \\label{} tags with metadata."""
    labels = []
    pattern = r"\\label\{([^}]+)\}"
    
    for i, line in enumerate(lines, start=1):
        for match in re.finditer(pattern, line):
            label_name = match.group(1)
            
            # Determine label type from prefix
            if label_name.startswith("sec:"):
                label_type = "section"
            elif label_name.startswith("def:"):
                label_type = "definition"
            elif label_name.startswith("inv:"):
                label_type = "invariant"
            elif label_name.startswith("tab:"):
                label_type = "table"
            elif label_name.startswith("eq:"):
                label_type = "equation"
            elif label_name.startswith("fig:"):
                label_type = "figure"
            else:
                label_type = "other"
            
            # Get context (current line stripped)
            context = line.strip()[:100]
            
            labels.append(LabelInfo(
                name=label_name,
                line_number=i,
                label_type=label_type,
                context=context
            ))
    
    return labels


def extract_definitions(content: str, lines: List[str]) -> List[DefinitionInfo]:
    """Extract all definition environments."""
    definitions = []
    
    # Pattern for \begin{definition}[name]...\end{definition}
    pattern = r"\\begin\{definition\}(?:\[([^\]]*)\])?(.*?)\\end\{definition\}"
    
    for match in re.finditer(pattern, content, re.DOTALL):
        name = match.group(1) or "Unnamed"
        body = match.group(2).strip()
        
        # Find line numbers
        start_pos = match.start()
        end_pos = match.end()
        line_start = content[:start_pos].count("\n") + 1
        line_end = content[:end_pos].count("\n") + 1
        
        # Extract label if present
        label_match = re.search(r"\\label\{([^}]+)\}", body)
        label = label_match.group(1) if label_match else ""
        
        # Clean content (remove label from body for cleaner representation)
        clean_body = re.sub(r"\\label\{[^}]+\}\s*", "", body).strip()
        
        definitions.append(DefinitionInfo(
            label=label,
            name=name,
            content=clean_body[:500],  # Truncate for JSON
            line_start=line_start,
            line_end=line_end
        ))
    
    return definitions


def extract_invariants(content: str, lines: List[str]) -> List[InvariantInfo]:
    """Extract all invariant environments."""
    invariants = []
    
    pattern = r"\\begin\{invariant\}(?:\[([^\]]*)\])?(.*?)\\end\{invariant\}"
    
    for match in re.finditer(pattern, content, re.DOTALL):
        name = match.group(1) or "Unnamed"
        body = match.group(2).strip()
        
        start_pos = match.start()
        end_pos = match.end()
        line_start = content[:start_pos].count("\n") + 1
        line_end = content[:end_pos].count("\n") + 1
        
        label_match = re.search(r"\\label\{([^}]+)\}", body)
        label = label_match.group(1) if label_match else ""
        
        clean_body = re.sub(r"\\label\{[^}]+\}\s*", "", body).strip()
        
        invariants.append(InvariantInfo(
            label=label,
            name=name,
            content=clean_body[:500],
            line_start=line_start,
            line_end=line_end
        ))
    
    return invariants


def extract_formulas(content: str, lines: List[str]) -> List[FormulaInfo]:
    """Extract all formula environments."""
    formulas = []
    
    # Inline math: $...$
    for i, line in enumerate(lines, start=1):
        for match in re.finditer(r"\$([^$]+)\$", line):
            if not match.group(1).startswith("$"):  # Avoid $$
                formulas.append(FormulaInfo(
                    formula_type="inline",
                    content=match.group(1),
                    line_number=i
                ))
    
    # Display math: \[...\]
    pattern = r"\\\[(.*?)\\\]"
    for match in re.finditer(pattern, content, re.DOTALL):
        line_num = content[:match.start()].count("\n") + 1
        formulas.append(FormulaInfo(
            formula_type="display",
            content=match.group(1).strip(),
            line_number=line_num
        ))
    
    # Align environments
    pattern = r"\\begin\{align\*?\}(.*?)\\end\{align\*?\}"
    for match in re.finditer(pattern, content, re.DOTALL):
        line_num = content[:match.start()].count("\n") + 1
        label_match = re.search(r"\\label\{([^}]+)\}", match.group(1))
        formulas.append(FormulaInfo(
            formula_type="align",
            content=match.group(1).strip()[:200],
            line_number=line_num,
            label=label_match.group(1) if label_match else None
        ))
    
    # Equation environments
    pattern = r"\\begin\{equation\*?\}(.*?)\\end\{equation\*?\}"
    for match in re.finditer(pattern, content, re.DOTALL):
        line_num = content[:match.start()].count("\n") + 1
        label_match = re.search(r"\\label\{([^}]+)\}", match.group(1))
        formulas.append(FormulaInfo(
            formula_type="equation",
            content=match.group(1).strip()[:200],
            line_number=line_num,
            label=label_match.group(1) if label_match else None
        ))
    
    return formulas


def extract_sections(content: str, lines: List[str]) -> List[Dict[str, Any]]:
    """Extract section structure."""
    sections = []
    
    # Match \section{...} and \subsection{...}
    pattern = r"\\(section|subsection|subsubsection)\{([^}]+)\}"
    
    for match in re.finditer(pattern, content):
        level = match.group(1)
        title = match.group(2)
        line_num = content[:match.start()].count("\n") + 1
        
        # Look for associated label
        # Check next few characters for \label
        post_content = content[match.end():match.end()+100]
        label_match = re.search(r"^\s*\\label\{([^}]+)\}", post_content)
        label = label_match.group(1) if label_match else None
        
        sections.append({
            "level": level,
            "title": title,
            "line_number": line_num,
            "label": label
        })
    
    return sections


def extract_refs(content: str, lines: List[str]) -> List[RefInfo]:
    """Extract all references (\\ref, \\eqref, \\pageref)."""
    refs = []
    
    for i, line in enumerate(lines, start=1):
        # \ref{...}
        for match in re.finditer(r"\\ref\{([^}]+)\}", line):
            refs.append(RefInfo("ref", match.group(1), i))
        
        # \eqref{...}
        for match in re.finditer(r"\\eqref\{([^}]+)\}", line):
            refs.append(RefInfo("eqref", match.group(1), i))
        
        # \pageref{...}
        for match in re.finditer(r"\\pageref\{([^}]+)\}", line):
            refs.append(RefInfo("pageref", match.group(1), i))
        
        # Section~\ref{...} pattern
        for match in re.finditer(r"~\\ref\{([^}]+)\}", line):
            # Already captured by \ref above, skip
            pass
    
    return refs


# ─────────────────────────────────────────────────────────────────────────────
# Validation Functions
# ─────────────────────────────────────────────────────────────────────────────

def validate_label_uniqueness(labels: List[LabelInfo]) -> List[str]:
    """Check for duplicate labels."""
    errors = []
    seen = {}
    
    for label in labels:
        if label.name in seen:
            errors.append(
                f"Duplicate label '{label.name}' at lines {seen[label.name]} and {label.line_number}"
            )
        else:
            seen[label.name] = label.line_number
    
    return errors


def validate_label_ordering(labels: List[LabelInfo]) -> List[str]:
    """Validate that section labels appear in expected order."""
    errors = []
    
    section_labels = [l for l in labels if l.label_type == "section"]
    section_names = [l.name.replace("sec:", "") for l in section_labels]
    
    # Check against expected order
    expected_idx = 0
    for name in section_names:
        if expected_idx < len(EXPECTED_SECTION_ORDER):
            if name == EXPECTED_SECTION_ORDER[expected_idx]:
                expected_idx += 1
            elif name in EXPECTED_SECTION_ORDER:
                actual_idx = EXPECTED_SECTION_ORDER.index(name)
                if actual_idx < expected_idx:
                    errors.append(
                        f"Section '{name}' appears out of order (expected after '{EXPECTED_SECTION_ORDER[expected_idx]}')"
                    )
                expected_idx = actual_idx + 1
    
    return errors


def validate_label_naming(labels: List[LabelInfo]) -> List[str]:
    """Validate label naming conventions."""
    errors = []
    
    for label in labels:
        # Check for proper prefixes
        if ":" not in label.name:
            errors.append(
                f"Label '{label.name}' at line {label.line_number} missing type prefix (sec:, def:, inv:, tab:, etc.)"
            )
        
        # Check for spaces or special characters
        if " " in label.name or "\t" in label.name:
            errors.append(
                f"Label '{label.name}' at line {label.line_number} contains whitespace"
            )
    
    return errors


# ─────────────────────────────────────────────────────────────────────────────
# Cross-Reference Auditor
# ─────────────────────────────────────────────────────────────────────────────

def audit_cross_references(
    labels: List[LabelInfo], 
    refs: List[RefInfo]
) -> CrossRefAuditResult:
    """Audit cross-references for completeness and correctness."""
    
    label_names = {l.name for l in labels}
    label_lines = {l.name: l.line_number for l in labels}
    referenced = set()
    dangling = []
    suggestions = []
    
    for ref in refs:
        referenced.add(ref.target)
        if ref.target not in label_names:
            dangling.append((ref.target, ref.line_number, ref.ref_type))
            
            # Generate suggestion
            # Find similar labels
            similar = find_similar_labels(ref.target, label_names)
            if similar:
                suggestions.append(
                    f"Line {ref.line_number}: \\{ref.ref_type}{{{ref.target}}} - "
                    f"Label not found. Did you mean: {', '.join(similar)}?"
                )
            else:
                suggestions.append(
                    f"Line {ref.line_number}: \\{ref.ref_type}{{{ref.target}}} - "
                    f"Label not found. Create with \\label{{{ref.target}}}"
                )
    
    # Find unused labels
    unused = []
    for label in labels:
        if label.name not in referenced:
            # Sections are often not referenced directly, don't flag them
            if label.label_type != "section":
                unused.append((label.name, label.line_number))
    
    return CrossRefAuditResult(
        dangling_refs=dangling,
        unused_labels=unused,
        suggestions=suggestions
    )


def find_similar_labels(target: str, labels: Set[str], threshold: int = 3) -> List[str]:
    """Find labels similar to target using edit distance."""
    similar = []
    
    for label in labels:
        distance = levenshtein_distance(target, label)
        if distance <= threshold:
            similar.append(label)
    
    return sorted(similar, key=lambda x: levenshtein_distance(target, x))[:3]


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    
    return prev_row[-1]


# ─────────────────────────────────────────────────────────────────────────────
# Drift Detection
# ─────────────────────────────────────────────────────────────────────────────

def check_drift(content: str, lines: List[str]) -> DriftCheckResult:
    """Check for drift against determinism contract and terminology rules."""
    
    determinism_violations = []
    terminology_issues = []
    structure_issues = []
    
    # Check for forbidden primitives in code listings that shouldn't be there
    # (They should only appear in the "forbidden" documentation, not as recommendations)
    in_lstlisting = False
    lstlisting_start = 0
    
    for i, line in enumerate(lines, start=1):
        if r"\begin{lstlisting}" in line:
            in_lstlisting = True
            lstlisting_start = i
            continue
        if r"\end{lstlisting}" in line:
            in_lstlisting = False
            continue
        
        # Check code listings for potential violations
        if in_lstlisting:
            for forbidden in DETERMINISM_FORBIDDEN_PRIMITIVES:
                # Look for usage patterns (not just documentation)
                # Pattern: the forbidden primitive followed by ( suggesting a call
                usage_pattern = rf"{re.escape(forbidden)}\s*\("
                if re.search(usage_pattern, line):
                    # Check if this is in a "forbidden" context or "replacement" context
                    # Look at surrounding context
                    context_start = max(0, i - lstlisting_start)
                    if "Forbidden" not in lines[lstlisting_start - 1] and \
                       "forbidden" not in lines[lstlisting_start - 1].lower() and \
                       "replacement" not in lines[max(0, i-3):i]:
                        determinism_violations.append((
                            i,
                            f"Potential use of forbidden primitive: {forbidden}",
                            line.strip()[:80]
                        ))
    
    # Check terminology in prose (outside code blocks)
    text_content = remove_code_blocks(content)
    text_lines = text_content.split("\n")
    
    for pattern, expected in TERMINOLOGY_RULES.items():
        for i, line in enumerate(text_lines, start=1):
            matches = list(re.finditer(pattern, line))
            for match in matches:
                found = match.group(0)
                # Skip if it's in a \texttt{} or path
                if is_in_code_context(line, match.start()):
                    continue
                terminology_issues.append((
                    i,
                    found,
                    expected,
                    line.strip()[:80]
                ))
    
    # Check structure: Evidence section should be empty
    evidence_section = extract_evidence_section(content)
    if evidence_section:
        # Check for content that shouldn't be there
        forbidden_evidence_patterns = [
            r"\d+%\s*(uplift|improvement)",
            r"statistically\s+significant",
            r"p\s*[<>=]\s*0\.\d+",
            r"results\s+(show|demonstrate|indicate)",
            r"we\s+(found|observed|measured)",
        ]
        for pattern in forbidden_evidence_patterns:
            if re.search(pattern, evidence_section, re.IGNORECASE):
                structure_issues.append(
                    f"Evidence section contains forbidden content matching: {pattern}"
                )
    
    # Check for Phase II markers
    # Count both literal markers and \phaseiilabel macro uses
    literal_count = content.count("PHASE II — NOT RUN IN PHASE I")
    macro_count = content.count(r"\phaseiilabel")
    total_marker_count = literal_count + macro_count
    
    if total_marker_count < 5:
        structure_issues.append(
            f"Insufficient Phase II markers: found {total_marker_count} "
            f"(literal: {literal_count}, macro: {macro_count}), expected at least 5"
        )
    
    return DriftCheckResult(
        determinism_violations=determinism_violations,
        terminology_issues=terminology_issues,
        structure_issues=structure_issues
    )


def remove_code_blocks(content: str) -> str:
    """Remove lstlisting and verbatim blocks from content."""
    # Remove lstlisting
    content = re.sub(
        r"\\begin\{lstlisting\}.*?\\end\{lstlisting\}",
        "",
        content,
        flags=re.DOTALL
    )
    # Remove verbatim
    content = re.sub(
        r"\\begin\{verbatim\}.*?\\end\{verbatim\}",
        "",
        content,
        flags=re.DOTALL
    )
    # Remove \texttt{...}
    content = re.sub(r"\\texttt\{[^}]*\}", "", content)
    # Remove \lstinline
    content = re.sub(r"\\lstinline[^{]*\{[^}]*\}", "", content)
    
    return content


def is_in_code_context(line: str, pos: int) -> bool:
    """Check if position is inside a code context."""
    # Check for \texttt{...}
    for match in re.finditer(r"\\texttt\{[^}]*\}", line):
        if match.start() <= pos < match.end():
            return True
    # Check for \lstinline
    for match in re.finditer(r"\\lstinline[^{]*\{[^}]*\}", line):
        if match.start() <= pos < match.end():
            return True
    return False


def extract_evidence_section(content: str) -> Optional[str]:
    """Extract the Evidence Interpretation section content."""
    match = re.search(
        r"\\section\{Evidence Interpretation\}(.*?)(?=\\section|\\appendix|\\end\{document\})",
        content,
        re.DOTALL
    )
    return match.group(1) if match else None


# ─────────────────────────────────────────────────────────────────────────────
# Canonicalization
# ─────────────────────────────────────────────────────────────────────────────

def build_canonical_representation(
    content: str,
    lines: List[str]
) -> CanonicalRepresentation:
    """Build the canonical JSON representation of fm.tex."""
    
    labels = extract_labels(content, lines)
    definitions = extract_definitions(content, lines)
    invariants = extract_invariants(content, lines)
    formulas = extract_formulas(content, lines)
    sections = extract_sections(content, lines)
    
    # Compute source hash
    source_hash = compute_file_hash(content)
    
    # Build canonical representation
    canon = CanonicalRepresentation(
        version="1.0.0",
        source_path=str(FM_TEX_PATH),
        source_hash=source_hash,
        labels=[asdict(l) for l in labels],
        definitions=[asdict(d) for d in definitions],
        invariants=[asdict(inv) for inv in invariants],
        formulas=[asdict(f) for f in formulas],
        sections=sections
    )
    
    # Compute signature hash (deterministic)
    canon.signature_hash = compute_signature_hash(canon)
    
    return canon


def compute_signature_hash(canon: CanonicalRepresentation) -> str:
    """Compute deterministic signature hash of canonical representation."""
    
    # Create ordered structure for hashing
    # Exclude signature_hash itself to avoid circular dependency
    hashable = OrderedDict([
        ("version", canon.version),
        ("source_hash", canon.source_hash),
        ("label_count", len(canon.labels)),
        ("definition_count", len(canon.definitions)),
        ("invariant_count", len(canon.invariants)),
        ("formula_count", len(canon.formulas)),
        ("section_count", len(canon.sections)),
        # Include sorted label names for content awareness
        ("label_names", sorted([l["name"] for l in canon.labels])),
        ("definition_labels", sorted([d["label"] for d in canon.definitions if d["label"]])),
        ("invariant_labels", sorted([i["label"] for i in canon.invariants if i["label"]])),
    ])
    
    # Serialize deterministically
    canonical_json = json.dumps(hashable, sort_keys=True, separators=(",", ":"))
    
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def save_canonical_json(canon: CanonicalRepresentation, path: Path) -> None:
    """Save canonical representation to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(canon.to_dict(), f, indent=2, sort_keys=True)


# ─────────────────────────────────────────────────────────────────────────────
# Spec Contract Export
# ─────────────────────────────────────────────────────────────────────────────

def find_section_for_label(label: str, sections: List[Dict[str, Any]]) -> Optional[str]:
    """Find the section number containing a label."""
    # Build section numbering
    section_nums = []
    sec_count = 0
    subsec_count = 0
    subsubsec_count = 0
    
    for sec in sections:
        if sec["level"] == "section":
            sec_count += 1
            subsec_count = 0
            subsubsec_count = 0
            section_nums.append((sec.get("label"), f"{sec_count}"))
        elif sec["level"] == "subsection":
            subsec_count += 1
            subsubsec_count = 0
            section_nums.append((sec.get("label"), f"{sec_count}.{subsec_count}"))
        elif sec["level"] == "subsubsection":
            subsubsec_count += 1
            section_nums.append((sec.get("label"), f"{sec_count}.{subsec_count}.{subsubsec_count}"))
    
    # Find matching section
    for sec_label, sec_num in section_nums:
        if sec_label == label:
            return sec_num
    
    return None


def find_section_by_title_pattern(pattern: str, sections: List[Dict[str, Any]]) -> Optional[str]:
    """Find section number by title pattern."""
    sec_count = 0
    subsec_count = 0
    
    for sec in sections:
        if sec["level"] == "section":
            sec_count += 1
            subsec_count = 0
            if re.search(pattern, sec["title"], re.IGNORECASE):
                return f"{sec_count}"
        elif sec["level"] == "subsection":
            subsec_count += 1
            if re.search(pattern, sec["title"], re.IGNORECASE):
                return f"{sec_count}.{subsec_count}"
    
    return None


def export_spec_contract(
    canonical_path: Optional[Path] = None,
    include_consumers: bool = True,
    doc_paths: Optional[List[Path]] = None,
) -> Dict[str, Any]:
    """
    Load fm_canonical.json and produce a distilled spec contract.
    
    The spec contract is the STABLE API for other agents (I, J, N, O).
    It always includes consumers and label_coverage by default.
    
    Args:
        canonical_path: Path to fm_canonical.json (default: FM_CANONICAL_PATH)
        include_consumers: Whether to include consumer mapping (default: True)
        doc_paths: External docs to check for consumer references
    
    Returns:
        Dictionary containing the spec contract with all required keys.
    """
    # Load canonical JSON
    canon_path = canonical_path or FM_CANONICAL_PATH
    if not canon_path.exists():
        raise FileNotFoundError(f"Canonical JSON not found: {canon_path}")
    
    with open(canon_path, "r", encoding="utf-8") as f:
        canon_data = json.load(f)
    
    # Build definitions list
    definitions = []
    for defn in canon_data.get("definitions", []):
        label = defn.get("label", "")
        name = defn.get("name", "Unnamed")
        section = find_section_for_label(label, canon_data.get("sections", []))
        if label:
            definitions.append({
                "label": label,
                "name": name,
                "section": section or "?"
            })
    
    # Build invariants list
    invariants = []
    for inv in canon_data.get("invariants", []):
        label = inv.get("label", "")
        name = inv.get("name", "Unnamed")
        section = find_section_for_label(label, canon_data.get("sections", []))
        if label:
            invariants.append({
                "label": label,
                "name": name,
                "section": section or "?"
            })
    
    # Find key sections
    sections = canon_data.get("sections", [])
    metrics_section = find_section_by_title_pattern(r"metric", sections)
    uplift_section = find_section_by_title_pattern(r"evidence|uplift", sections)
    phase_section = find_section_by_title_pattern(r"introduction|overview", sections)
    
    # Build base contract
    contract = {
        "version": "1.0",
        "signature_hash": canon_data.get("signature_hash", ""),
        "definitions": definitions,
        "invariants": invariants,
        "metrics_section": metrics_section,
        "uplift_section": uplift_section,
        "phase_section": phase_section,
    }
    
    # Always include consumers and label_coverage (frozen API structure)
    if include_consumers:
        fm_labels = {l["name"] for l in canon_data.get("labels", [])}
        consumers_index = build_consumers_index(fm_labels, doc_paths)
        contract["consumers"] = consumers_index
        
        # Compute coverage
        with_refs = sum(1 for data in consumers_index.values() if data["total_refs"] > 0)
        total = len(consumers_index)
        coverage_pct = (with_refs / total * 100) if total > 0 else 0.0
        
        contract["label_coverage"] = {
            "labels_with_refs": with_refs,
            "total_labels": total,
            "coverage_pct": round(coverage_pct, 2)
        }
    else:
        # Even without consumers, include empty structures for API consistency
        contract["consumers"] = {}
        contract["label_coverage"] = {
            "labels_with_refs": 0,
            "total_labels": 0,
            "coverage_pct": 0.0
        }
    
    return contract


def save_spec_contract(contract: Dict[str, Any], path: Optional[Path] = None) -> Path:
    """Save spec contract to JSON file."""
    out_path = path or FM_SPEC_CONTRACT_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(contract, f, indent=2, sort_keys=True)
    
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Cross-Check Against External Docs
# ─────────────────────────────────────────────────────────────────────────────

def extract_label_refs_from_markdown(content: str) -> Set[str]:
    """Extract LaTeX-style label references from markdown content."""
    refs = set()
    
    # Match patterns like `def:slice`, `inv:monotonicity`, `sec:metrics`
    # These might appear in backticks or as plain text references
    patterns = [
        r"`((?:def|inv|sec|tab|eq|fig):[a-z0-9_-]+)`",  # In backticks
        r"\b((?:def|inv|sec|tab|eq|fig):[a-z0-9_-]+)\b",  # Plain text
    ]
    
    for pattern in patterns:
        for match in re.finditer(pattern, content, re.IGNORECASE):
            refs.add(match.group(1).lower())
    
    return refs


def extract_definition_refs_from_markdown(content: str) -> Set[str]:
    """Extract references to fm.tex definitions by name."""
    refs = set()
    
    # Common definition name patterns
    patterns = [
        r"Definition\s+(\d+\.\d+)",
        r"Invariant\s+(\d+\.\d+)",
        r"as defined in (Section|Sec\.?)\s+(\d+\.\d+)",
    ]
    
    for pattern in patterns:
        for match in re.finditer(pattern, content, re.IGNORECASE):
            refs.add(match.group(0))
    
    return refs


def check_core_terminology(content: str) -> Tuple[List[str], List[str]]:
    """Check which core terms are present/missing in content."""
    present = []
    missing = []
    
    for term in CORE_TERMINOLOGY:
        if term.lower() in content.lower():
            present.append(term)
        else:
            missing.append(term)
    
    return present, missing


def cross_check_document(
    doc_path: Path,
    fm_labels: Set[str],
    fm_definitions: List[Dict[str, Any]],
    fm_invariants: List[Dict[str, Any]],
) -> CrossCheckResult:
    """Cross-check a single document against fm.tex labels."""
    
    if not doc_path.exists():
        return CrossCheckResult(
            doc_path=str(doc_path),
            missing_labels=[],
            undefined_terms=[],
            referenced_definitions=[],
            status="skip",
            details=[f"Document not found: {doc_path}"]
        )
    
    with open(doc_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Check label references
    doc_refs = extract_label_refs_from_markdown(content)
    fm_labels_lower = {l.lower() for l in fm_labels}
    missing_labels = [ref for ref in doc_refs if ref not in fm_labels_lower]
    
    # Check which definitions are referenced
    def_labels = {d["label"].lower() for d in fm_definitions if d.get("label")}
    inv_labels = {i["label"].lower() for i in fm_invariants if i.get("label")}
    all_def_labels = def_labels | inv_labels
    
    referenced = [ref for ref in doc_refs if ref in all_def_labels]
    
    # Check terminology
    _, undefined = check_core_terminology(content)
    
    # Determine status
    details = []
    if missing_labels:
        details.append(f"Labels referenced but not in fm.tex: {missing_labels}")
    if undefined:
        details.append(f"Core terms not found: {undefined}")
    if referenced:
        details.append(f"Definitions correctly referenced: {referenced}")
    
    if missing_labels:
        status = "fail"
    elif len(undefined) > 3:  # More than 3 core terms missing is a warning
        status = "warn"
    else:
        status = "pass"
    
    return CrossCheckResult(
        doc_path=str(doc_path),
        missing_labels=missing_labels,
        undefined_terms=undefined,
        referenced_definitions=referenced,
        status=status,
        details=details
    )


def run_cross_check(
    canonical_path: Optional[Path] = None,
    doc_paths: Optional[List[Path]] = None,
) -> CrossCheckSummary:
    """Run cross-check against all specified documents."""
    from datetime import datetime, timezone
    
    # Load canonical JSON
    canon_path = canonical_path or FM_CANONICAL_PATH
    if not canon_path.exists():
        raise FileNotFoundError(f"Canonical JSON not found: {canon_path}")
    
    with open(canon_path, "r", encoding="utf-8") as f:
        canon_data = json.load(f)
    
    # Extract labels
    fm_labels = {l["name"] for l in canon_data.get("labels", [])}
    fm_definitions = canon_data.get("definitions", [])
    fm_invariants = canon_data.get("invariants", [])
    
    # Check each document
    docs = doc_paths or CROSS_CHECK_DOCS
    results = []
    passed = 0
    warned = 0
    failed = 0
    
    for doc_path in docs:
        result = cross_check_document(doc_path, fm_labels, fm_definitions, fm_invariants)
        results.append({
            "doc_path": result.doc_path,
            "status": result.status,
            "missing_labels": result.missing_labels,
            "undefined_terms": result.undefined_terms,
            "referenced_definitions": result.referenced_definitions,
            "details": result.details
        })
        
        if result.status == "pass":
            passed += 1
        elif result.status == "warn":
            warned += 1
        elif result.status == "fail":
            failed += 1
    
    return CrossCheckSummary(
        timestamp=datetime.now(timezone.utc).isoformat(),
        fm_signature_hash=canon_data.get("signature_hash", ""),
        docs_checked=len(docs),
        docs_passed=passed,
        docs_warned=warned,
        docs_failed=failed,
        results=results
    )


def generate_cross_check_report(summary: CrossCheckSummary) -> str:
    """Generate markdown report from cross-check summary."""
    lines = [
        "# Field Manual Cross-Check Report",
        "",
        f"**Timestamp:** {summary.timestamp}",
        f"**FM Signature Hash:** `{summary.fm_signature_hash[:16]}...`",
        "",
        "## Summary",
        "",
        f"| Status | Count |",
        f"|--------|-------|",
        f"| ✅ Passed | {summary.docs_passed} |",
        f"| ⚠️ Warned | {summary.docs_warned} |",
        f"| ❌ Failed | {summary.docs_failed} |",
        f"| **Total** | {summary.docs_checked} |",
        "",
        "## Details",
        "",
    ]
    
    for result in summary.results:
        status_icon = {"pass": "✅", "warn": "⚠️", "fail": "❌", "skip": "⏭️"}.get(result["status"], "?")
        lines.append(f"### {status_icon} `{Path(result['doc_path']).name}`")
        lines.append("")
        
        if result["missing_labels"]:
            lines.append("**Missing Labels:**")
            for label in result["missing_labels"]:
                lines.append(f"- `{label}`")
            lines.append("")
        
        if result["undefined_terms"]:
            lines.append("**Core Terms Not Found:**")
            for term in result["undefined_terms"]:
                lines.append(f"- {term}")
            lines.append("")
        
        if result["referenced_definitions"]:
            lines.append("**Definitions Referenced:**")
            for ref in result["referenced_definitions"]:
                lines.append(f"- `{ref}`")
            lines.append("")
        
        if not result["missing_labels"] and not result["undefined_terms"]:
            lines.append("*All checks passed.*")
            lines.append("")
    
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Label Consumers & Drift Detection
# ─────────────────────────────────────────────────────────────────────────────

def build_consumers_index(
    fm_labels: Set[str],
    doc_paths: Optional[List[Path]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Build an index mapping each fm.tex label to its external consumers.
    
    Returns:
        Dictionary mapping label -> {consumers: {doc_path: count}, total_refs: int}
    """
    docs = doc_paths or CROSS_CHECK_DOCS
    consumers: Dict[str, Dict[str, int]] = {label: {} for label in fm_labels}
    
    for doc_path in docs:
        if not doc_path.exists():
            continue
        
        with open(doc_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        doc_name = doc_path.name
        doc_refs = extract_label_refs_from_markdown(content)
        
        # Count references per label
        for ref in doc_refs:
            ref_lower = ref.lower()
            # Match against fm_labels (case-insensitive)
            for fm_label in fm_labels:
                if fm_label.lower() == ref_lower:
                    if doc_name not in consumers[fm_label]:
                        consumers[fm_label][doc_name] = 0
                    consumers[fm_label][doc_name] += 1
                    break
    
    # Build result with totals
    result = {}
    for label, doc_counts in consumers.items():
        total = sum(doc_counts.values())
        result[label] = {
            "consumers": doc_counts,
            "total_refs": total
        }
    
    return result


def compute_label_drift(
    canonical_path: Optional[Path] = None,
    doc_paths: Optional[List[Path]] = None,
) -> LabelDriftSummary:
    """
    Compute label drift between fm.tex and external documentation.
    
    Returns:
        LabelDriftSummary with fm_only, external_only, and well_connected labels.
    """
    from datetime import datetime, timezone
    
    # Load canonical JSON
    canon_path = canonical_path or FM_CANONICAL_PATH
    if not canon_path.exists():
        raise FileNotFoundError(f"Canonical JSON not found: {canon_path}")
    
    with open(canon_path, "r", encoding="utf-8") as f:
        canon_data = json.load(f)
    
    # Get all fm.tex labels (definitions, invariants, sections, tables)
    fm_labels = set()
    for label_info in canon_data.get("labels", []):
        fm_labels.add(label_info["name"])
    
    # Collect all external references
    docs = doc_paths or CROSS_CHECK_DOCS
    external_refs: Set[str] = set()
    
    for doc_path in docs:
        if not doc_path.exists():
            continue
        
        with open(doc_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        doc_refs = extract_label_refs_from_markdown(content)
        external_refs.update(ref.lower() for ref in doc_refs)
    
    # Normalize fm_labels for comparison
    fm_labels_lower = {label.lower(): label for label in fm_labels}
    
    # Compute drift
    fm_only = []
    well_connected = []
    
    for label_lower, label in fm_labels_lower.items():
        if label_lower in external_refs:
            well_connected.append(label)
        else:
            fm_only.append(label)
    
    # External-only: referenced but not defined
    external_only = []
    for ref in external_refs:
        if ref not in fm_labels_lower:
            external_only.append(ref)
    
    # Compute coverage
    total_fm = len(fm_labels)
    coverage_pct = (len(well_connected) / total_fm * 100) if total_fm > 0 else 0.0
    
    return LabelDriftSummary(
        timestamp=datetime.now(timezone.utc).isoformat(),
        fm_signature_hash=canon_data.get("signature_hash", ""),
        fm_only_labels=sorted(fm_only),
        external_only_labels=sorted(external_only),
        well_connected_labels=sorted(well_connected),
        coverage_pct=round(coverage_pct, 2)
    )


def save_label_drift_summary(summary: LabelDriftSummary, path: Optional[Path] = None) -> Path:
    """Save label drift summary to JSON file."""
    out_path = path or LABEL_DRIFT_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=2, sort_keys=True)
    
    return out_path


def export_spec_contract_with_consumers(
    canonical_path: Optional[Path] = None,
    doc_paths: Optional[List[Path]] = None,
) -> Dict[str, Any]:
    """
    Export spec contract with consumer information.
    
    DEPRECATED: Use export_spec_contract() directly — consumers are now always included.
    
    This function exists for backwards compatibility with tests.
    """
    return export_spec_contract(canonical_path, include_consumers=True, doc_paths=doc_paths)


# ─────────────────────────────────────────────────────────────────────────────
# FM Router API — Fast Lookup & Posture Functions
# ─────────────────────────────────────────────────────────────────────────────
#
# These functions provide a stable, code-level lookup interface over the spec
# contract. They are designed for use by other agents (I, J, N, O) to quickly
# query the Field Manual without parsing JSON directly.
#
# USAGE:
#   contract = json.load(open("artifacts/spec/fm_spec_contract.json"))
#   defn = get_label_definition(contract, "def:slice")
#   consumers = get_label_consumers(contract, "def:slice")
#
# API CONTRACT:
#   - All functions are pure (no side effects)
#   - All functions are deterministic
#   - Unknown labels return None or empty dict
# ─────────────────────────────────────────────────────────────────────────────

def get_label_definition(contract: Dict[str, Any], label: str) -> Optional[Dict[str, Any]]:
    """
    Look up a definition or invariant by label.
    
    Args:
        contract: The fm_spec_contract.json contents
        label: Label to look up (e.g., "def:slice", "inv:monotonicity")
    
    Returns:
        Definition/invariant dict with keys {label, name, section}, or None if not found.
    
    Example:
        >>> get_label_definition(contract, "def:slice")
        {"label": "def:slice", "name": "Slice", "section": "2"}
    """
    # Check definitions
    for defn in contract.get("definitions", []):
        if defn.get("label") == label:
            return defn
    
    # Check invariants
    for inv in contract.get("invariants", []):
        if inv.get("label") == label:
            return inv
    
    return None


def get_label_consumers(contract: Dict[str, Any], label: str) -> Dict[str, int]:
    """
    Get external document references for a label.
    
    Args:
        contract: The fm_spec_contract.json contents
        label: Label to look up
    
    Returns:
        Dict mapping doc names to reference counts, or empty dict if not found.
    
    Example:
        >>> get_label_consumers(contract, "def:slice")
        {"PHASE2_RFL_UPLIFT_PLAN.md": 3, "RFL_UPLIFT_THEORY.md": 1}
    """
    consumers = contract.get("consumers", {})
    label_data = consumers.get(label, {})
    return label_data.get("consumers", {})


def get_label_total_refs(contract: Dict[str, Any], label: str) -> int:
    """
    Get total reference count for a label across all external docs.
    
    Args:
        contract: The fm_spec_contract.json contents
        label: Label to look up
    
    Returns:
        Total reference count, or 0 if not found.
    """
    consumers = contract.get("consumers", {})
    label_data = consumers.get(label, {})
    return label_data.get("total_refs", 0)


def is_label_well_connected(contract: Dict[str, Any], label: str) -> bool:
    """
    Check if a label is well-connected (has external references).
    
    Args:
        contract: The fm_spec_contract.json contents
        label: Label to check
    
    Returns:
        True if the label has at least one external reference.
    """
    return get_label_total_refs(contract, label) > 0


def build_fm_consumers_view(contract: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Build a view of which docs rely most heavily on fm.tex labels.
    
    Args:
        contract: The fm_spec_contract.json contents
    
    Returns:
        List of dicts sorted by total_refs descending:
        [
            {"doc": "PHASE2_RFL_UPLIFT_PLAN.md", "total_refs": 15, "label_count": 8},
            {"doc": "RFL_UPLIFT_THEORY.md", "total_refs": 5, "label_count": 3},
            ...
        ]
    
    Each entry contains:
        - doc: Document filename
        - total_refs: Total number of label references in this doc
        - label_count: Number of distinct labels referenced
    """
    consumers = contract.get("consumers", {})
    
    # Aggregate by document
    doc_stats: Dict[str, Dict[str, int]] = {}
    
    for label, label_data in consumers.items():
        doc_refs = label_data.get("consumers", {})
        for doc, count in doc_refs.items():
            if doc not in doc_stats:
                doc_stats[doc] = {"total_refs": 0, "label_count": 0}
            doc_stats[doc]["total_refs"] += count
            doc_stats[doc]["label_count"] += 1
    
    # Build result list
    result = [
        {
            "doc": doc,
            "total_refs": stats["total_refs"],
            "label_count": stats["label_count"],
        }
        for doc, stats in doc_stats.items()
    ]
    
    # Sort by total_refs descending, then by doc name for stability
    result.sort(key=lambda x: (-x["total_refs"], x["doc"]))
    
    return result


def build_fm_posture(
    contract: Dict[str, Any],
    drift_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a minimal, governance-facing "FM posture" summary.
    
    This provides a quick health check view of the Field Manual's
    connectivity and coverage status.
    
    Args:
        contract: The fm_spec_contract.json contents
        drift_summary: The label_drift_summary.json contents
    
    Returns:
        {
            "schema_version": "1.0.0",
            "signature_hash": "abc123...",
            "total_labels": 84,
            "well_connected_labels": 5,
            "external_only_labels": 0,
            "fm_only_labels": 79,
            "coverage_pct": 5.95,
            "health_status": "healthy" | "warning" | "critical"
        }
    
    Health status:
        - "critical": external_only_labels > 0 (drift detected)
        - "warning": coverage_pct < 10
        - "healthy": otherwise
    """
    label_coverage = contract.get("label_coverage", {})
    
    total_labels = label_coverage.get("total_labels", 0)
    well_connected = len(drift_summary.get("well_connected_labels", []))
    external_only = len(drift_summary.get("external_only_labels", []))
    fm_only = len(drift_summary.get("fm_only_labels", []))
    coverage_pct = drift_summary.get("coverage_pct", 0.0)
    
    # Determine health status
    if external_only > 0:
        health_status = "critical"
    elif coverage_pct < 10:
        health_status = "warning"
    else:
        health_status = "healthy"
    
    return {
        "schema_version": "1.0.0",
        "signature_hash": contract.get("signature_hash", ""),
        "total_labels": total_labels,
        "well_connected_labels": well_connected,
        "external_only_labels": external_only,
        "fm_only_labels": fm_only,
        "coverage_pct": coverage_pct,
        "health_status": health_status,
    }


def load_fm_posture() -> Dict[str, Any]:
    """
    Load and compute FM posture from artifact files.
    
    Convenience function that loads both the spec contract and drift summary,
    then computes the posture.
    
    Returns:
        FM posture dict (see build_fm_posture for schema)
    
    Raises:
        FileNotFoundError: If artifact files are missing
    """
    if not FM_SPEC_CONTRACT_PATH.exists():
        raise FileNotFoundError(f"Spec contract not found: {FM_SPEC_CONTRACT_PATH}")
    if not LABEL_DRIFT_PATH.exists():
        raise FileNotFoundError(f"Drift summary not found: {LABEL_DRIFT_PATH}")
    
    with open(FM_SPEC_CONTRACT_PATH, "r", encoding="utf-8") as f:
        contract = json.load(f)
    
    with open(LABEL_DRIFT_PATH, "r", encoding="utf-8") as f:
        drift_summary = json.load(f)
    
    return build_fm_posture(contract, drift_summary)


def build_fm_governance_snapshot(posture: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build an FM governance snapshot with governance status.
    
    This extends the FM posture with governance-specific status indicators
    for use in governance dashboards and decision-making.
    
    Args:
        posture: The FM posture dict from build_fm_posture()
    
    Returns:
        {
            "schema_version": "1.0.0",
            "total_labels": int,
            "well_connected_labels": int,
            "external_only_labels": int,
            "coverage_pct": float,
            "health_status": "healthy" | "warning" | "critical",
            "governance_status": "OK" | "WARN" | "ATTENTION"
        }
    
    Governance status:
        - "ATTENTION": external_only_labels > 0 (drift requires attention)
        - "WARN": coverage_pct < 10 or health_status is "warning"
        - "OK": otherwise
    """
    governance_status = "OK"
    
    # ATTENTION if external-only labels exist (drift detected)
    if posture.get("external_only_labels", 0) > 0:
        governance_status = "ATTENTION"
    # WARN if low coverage or health warning
    elif posture.get("coverage_pct", 0.0) < 10 or posture.get("health_status") == "warning":
        governance_status = "WARN"
    
    return {
        "schema_version": posture.get("schema_version", "1.0.0"),
        "total_labels": posture.get("total_labels", 0),
        "well_connected_labels": posture.get("well_connected_labels", 0),
        "external_only_labels": posture.get("external_only_labels", 0),
        "fm_only_labels": posture.get("fm_only_labels", 0),
        "coverage_pct": posture.get("coverage_pct", 0.0),
        "health_status": posture.get("health_status", "unknown"),
        "governance_status": governance_status,
    }


def compute_alignment_indicator(consumers_view: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute cross-system alignment indicator from consumers view.
    
    This analyzes the distribution of FM label references across external
    documents to detect concentration patterns and alignment issues.
    
    Args:
        consumers_view: List from build_fm_consumers_view()
    
    Returns:
        {
            "heaviest_consumers": [
                {"doc": "...", "total_refs": N, "label_count": M},
                ...
            ],
            "docs_with_low_label_diversity": [
                {"doc": "...", "total_refs": N, "label_count": M, "diversity_ratio": R},
                ...
            ],
            "alignment_status": "WELL_DISTRIBUTED" | "CONCENTRATED" | "SPARSE"
        }
    
    Alignment status:
        - "SPARSE": No or very few references (total_refs < 5)
        - "CONCENTRATED": Top doc has > 50% of total refs
        - "WELL_DISTRIBUTED": Otherwise
    
    Low diversity: docs with total_refs > 10 but label_count < total_refs * 0.3
    (i.e., many references but few distinct labels)
    """
    if not consumers_view:
        return {
            "heaviest_consumers": [],
            "docs_with_low_label_diversity": [],
            "alignment_status": "SPARSE",
        }
    
    # Top 3 heaviest consumers
    heaviest = consumers_view[:3]
    
    # Calculate total refs for concentration analysis
    total_refs_all = sum(e["total_refs"] for e in consumers_view)
    
    # Detect low label diversity
    low_diversity = []
    for entry in consumers_view:
        total_refs = entry["total_refs"]
        label_count = entry["label_count"]
        
        # High refs but low diversity (many refs to few labels)
        if total_refs > 10 and label_count > 0:
            diversity_ratio = label_count / total_refs
            if diversity_ratio < 0.3:  # Less than 30% diversity
                low_diversity.append({
                    "doc": entry["doc"],
                    "total_refs": total_refs,
                    "label_count": label_count,
                    "diversity_ratio": round(diversity_ratio, 3),
                })
    
    # Determine alignment status
    if total_refs_all < 5:
        alignment_status = "SPARSE"
    elif consumers_view and consumers_view[0]["total_refs"] > total_refs_all * 0.5:
        alignment_status = "CONCENTRATED"
    else:
        alignment_status = "WELL_DISTRIBUTED"
    
    return {
        "heaviest_consumers": heaviest,
        "docs_with_low_label_diversity": low_diversity,
        "alignment_status": alignment_status,
    }


def summarize_fm_for_global_health(
    governance_snapshot: Dict[str, Any],
    alignment_indicator: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Summarize FM status for global health monitoring.
    
    This provides a minimal, actionable summary for integration into
    global system health dashboards and monitoring systems.
    
    Args:
        governance_snapshot: From build_fm_governance_snapshot()
        alignment_indicator: From compute_alignment_indicator()
    
    Returns:
        {
            "fm_ok": bool,
            "coverage_pct": float,
            "external_only_labels": int,
            "alignment_status": "WELL_DISTRIBUTED" | "CONCENTRATED" | "SPARSE",
            "status": "OK" | "WARN" | "BLOCK"
        }
    
    Status:
        - "BLOCK": external_only_labels > 0 or health_status is "critical"
        - "WARN": governance_status is "WARN" or alignment is "CONCENTRATED"
        - "OK": otherwise
    
    fm_ok: True if status is "OK", False otherwise
    """
    external_only = governance_snapshot.get("external_only_labels", 0)
    health_status = governance_snapshot.get("health_status", "unknown")
    governance_status = governance_snapshot.get("governance_status", "OK")
    alignment_status = alignment_indicator.get("alignment_status", "SPARSE")
    
    # Determine global status
    if external_only > 0 or health_status == "critical":
        status = "BLOCK"
    elif governance_status == "WARN" or alignment_status == "CONCENTRATED":
        status = "WARN"
    else:
        status = "OK"
    
    return {
        "fm_ok": status == "OK",
        "coverage_pct": governance_snapshot.get("coverage_pct", 0.0),
        "external_only_labels": external_only,
        "alignment_status": alignment_status,
        "status": status,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Phase IV — Field Manual as API & Cross-Agent Truth Anchor
# ─────────────────────────────────────────────────────────────────────────────


def extract_labels_from_taxonomy(taxonomy_semantics: Dict[str, Any]) -> Set[str]:
    """
    Extract label/term identifiers from taxonomy semantics.
    
    Supports various taxonomy structures:
    - {"terms": [...], "labels": [...], "definitions": {...}}
    - {"entries": [{"term": "...", "label": "..."}]}
    """
    labels: Set[str] = set()
    
    # Direct labels list
    if "labels" in taxonomy_semantics:
        labels.update(taxonomy_semantics["labels"])
    
    # Terms list
    if "terms" in taxonomy_semantics:
        labels.update(taxonomy_semantics["terms"])
    
    # Definitions dict keys
    if "definitions" in taxonomy_semantics:
        labels.update(taxonomy_semantics["definitions"].keys())
    
    # Entries list with term/label fields
    if "entries" in taxonomy_semantics:
        for entry in taxonomy_semantics["entries"]:
            if isinstance(entry, dict):
                if "label" in entry:
                    labels.add(entry["label"])
                if "term" in entry:
                    labels.add(entry["term"])
    
    return labels


def extract_labels_from_curriculum(curriculum_manifest: Dict[str, Any]) -> Set[str]:
    """
    Extract label/term identifiers from curriculum manifest.
    
    Supports various curriculum structures:
    - {"slices": [{"name": "..."}], "parameters": [...]}
    - {"systems": {...}, "labels": [...]}
    """
    labels: Set[str] = set()
    
    # Direct labels list
    if "labels" in curriculum_manifest:
        labels.update(curriculum_manifest["labels"])
    
    # Slices with names
    if "slices" in curriculum_manifest:
        for slice_def in curriculum_manifest["slices"]:
            if isinstance(slice_def, dict) and "name" in slice_def:
                labels.add(slice_def["name"])
    
    # Systems with slices
    if "systems" in curriculum_manifest:
        for system_data in curriculum_manifest["systems"].values():
            if isinstance(system_data, dict) and "slices" in system_data:
                for slice_def in system_data["slices"]:
                    if isinstance(slice_def, dict) and "name" in slice_def:
                        labels.add(slice_def["name"])
    
    # Parameters list
    if "parameters" in curriculum_manifest:
        labels.update(curriculum_manifest["parameters"])
    
    return labels


def build_label_contract_index(
    spec_contract: Dict[str, Any],
    taxonomy_semantics: Dict[str, Any],
    curriculum_manifest: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build cross-agent label contract index.
    
    PHASE IV — NOT RUN IN PHASE I
    No uplift claims are made.
    Deterministic execution guaranteed.
    
    Checks alignment between Field Manual, taxonomy, and curriculum
    to ensure labels are consistently defined across all systems.
    
    Args:
        spec_contract: FM spec contract from export_spec_contract()
        taxonomy_semantics: Taxonomy structure (flexible format)
        curriculum_manifest: Curriculum structure (flexible format)
    
    Returns:
        {
            "schema_version": "1.0.0",
            "label_index": {
                "<label>": {
                    "in_fm": bool,
                    "in_taxonomy": bool,
                    "in_curriculum": bool
                },
                ...
            },
            "labels_missing_in_taxonomy": List[str],
            "labels_missing_in_curriculum": List[str],
            "labels_missing_in_fm": List[str],
            "contract_status": "ALIGNED" | "PARTIAL" | "BROKEN"
        }
    """
    # Extract FM labels
    fm_labels: Set[str] = set()
    for defn in spec_contract.get("definitions", []):
        if "label" in defn:
            fm_labels.add(defn["label"])
    for inv in spec_contract.get("invariants", []):
        if "label" in inv:
            fm_labels.add(inv["label"])
    
    # Extract taxonomy and curriculum labels
    taxonomy_labels = extract_labels_from_taxonomy(taxonomy_semantics)
    curriculum_labels = extract_labels_from_curriculum(curriculum_manifest)
    
    # Build unified label set
    all_labels = fm_labels | taxonomy_labels | curriculum_labels
    
    # Build label index
    label_index: Dict[str, Dict[str, bool]] = {}
    for label in sorted(all_labels):
        label_index[label] = {
            "in_fm": label in fm_labels,
            "in_taxonomy": label in taxonomy_labels,
            "in_curriculum": label in curriculum_labels,
        }
    
    # Find missing labels
    labels_missing_in_taxonomy = sorted(taxonomy_labels - fm_labels)
    labels_missing_in_curriculum = sorted(curriculum_labels - fm_labels)
    labels_missing_in_fm = sorted((taxonomy_labels | curriculum_labels) - fm_labels)
    
    # Determine contract status
    if labels_missing_in_fm:
        contract_status = "BROKEN"
    elif labels_missing_in_taxonomy or labels_missing_in_curriculum:
        contract_status = "PARTIAL"
    else:
        contract_status = "ALIGNED"
    
    return {
        "schema_version": "1.0.0",
        "label_index": label_index,
        "labels_missing_in_taxonomy": labels_missing_in_taxonomy,
        "labels_missing_in_curriculum": labels_missing_in_curriculum,
        "labels_missing_in_fm": labels_missing_in_fm,
        "contract_status": contract_status,
    }


def build_field_manual_integration_contract(
    label_index: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build agent integration contract from label index.
    
    PHASE IV — NOT RUN IN PHASE I
    No uplift claims are made.
    Deterministic execution guaranteed.
    
    Provides a contract that agents can use to understand which label sets
    are required and which are missing or incomplete.
    
    Args:
        label_index: Label contract index from build_label_contract_index()
    
    Returns:
        {
            "contract_version": "1.0.0",
            "required_label_sets": {
                "curriculum": List[str],
                "taxonomy": List[str],
                "docs": List[str]
            },
            "integration_status": "OK" | "WARN" | "BLOCK",
            "notes": str
        }
    """
    contract_status = label_index.get("contract_status", "UNKNOWN")
    missing_in_taxonomy = label_index.get("labels_missing_in_taxonomy", [])
    missing_in_curriculum = label_index.get("labels_missing_in_curriculum", [])
    missing_in_fm = label_index.get("labels_missing_in_fm", [])
    
    # Extract required label sets from index
    label_index_data = label_index.get("label_index", {})
    
    curriculum_labels = sorted([
        label for label, data in label_index_data.items()
        if data.get("in_curriculum", False)
    ])
    
    taxonomy_labels = sorted([
        label for label, data in label_index_data.items()
        if data.get("in_taxonomy", False)
    ])
    
    docs_labels = sorted([
        label for label, data in label_index_data.items()
        if data.get("in_fm", False)
    ])
    
    # Determine integration status
    if contract_status == "BROKEN" or missing_in_fm:
        integration_status = "BLOCK"
    elif contract_status == "PARTIAL" or missing_in_taxonomy or missing_in_curriculum:
        integration_status = "WARN"
    else:
        integration_status = "OK"
    
    # Build neutral notes
    notes_parts = []
    if missing_in_fm:
        notes_parts.append(f"{len(missing_in_fm)} label(s) used in taxonomy/curriculum but not defined in Field Manual")
    if missing_in_taxonomy:
        notes_parts.append(f"{len(missing_in_taxonomy)} label(s) in taxonomy but not in Field Manual")
    if missing_in_curriculum:
        notes_parts.append(f"{len(missing_in_curriculum)} label(s) in curriculum but not in Field Manual")
    
    if not notes_parts:
        notes = "All label sets are aligned across systems."
    else:
        notes = "Integration status: " + "; ".join(notes_parts) + "."
    
    return {
        "contract_version": "1.0.0",
        "required_label_sets": {
            "curriculum": curriculum_labels,
            "taxonomy": taxonomy_labels,
            "docs": docs_labels,
        },
        "integration_status": integration_status,
        "notes": notes,
    }


def build_field_manual_director_panel(
    fm_posture: Dict[str, Any],
    integration_contract: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build Field Manual director panel for governance dashboard.
    
    PHASE IV — NOT RUN IN PHASE I
    No uplift claims are made.
    Deterministic execution guaranteed.
    
    Provides a high-level governance view of Field Manual status,
    suitable for director-level dashboards and decision-making.
    
    Args:
        fm_posture: FM posture from build_fm_posture()
        integration_contract: Integration contract from build_field_manual_integration_contract()
    
    Returns:
        {
            "status_light": "🟢" | "🟡" | "🔴",
            "health_status": "healthy" | "warning" | "critical",
            "coverage_pct": float,
            "integration_status": "OK" | "WARN" | "BLOCK",
            "headline": str
        }
    """
    health_status = fm_posture.get("health_status", "unknown")
    coverage_pct = fm_posture.get("coverage_pct", 0.0)
    integration_status = integration_contract.get("integration_status", "UNKNOWN")
    
    # Determine status light
    if health_status == "critical" or integration_status == "BLOCK":
        status_light = "🔴"
    elif health_status == "warning" or integration_status == "WARN":
        status_light = "🟡"
    else:
        status_light = "🟢"
    
    # Build neutral headline
    if integration_status == "BLOCK":
        headline = f"Field Manual integration blocked: {integration_contract.get('notes', '')}"
    elif integration_status == "WARN":
        headline = f"Field Manual integration partial: {integration_contract.get('notes', '')}"
    elif coverage_pct < 10:
        headline = f"Field Manual coverage at {coverage_pct:.1f}%: limited external references detected."
    elif coverage_pct >= 50:
        headline = f"Field Manual coverage at {coverage_pct:.1f}%: well-connected across documentation."
    else:
        headline = f"Field Manual coverage at {coverage_pct:.1f}%: moderate external reference connectivity."
    
    return {
        "status_light": status_light,
        "health_status": health_status,
        "coverage_pct": coverage_pct,
        "integration_status": integration_status,
        "headline": headline,
    }


def build_field_manual_drift_timeline(
    posture_snapshots: Sequence[Dict[str, Any]],
    label_indexes: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build Field Manual drift timeline from historical snapshots.
    
    PHASE IV — NOT RUN IN PHASE I
    No uplift claims are made.
    Deterministic execution guaranteed.
    
    Analyzes trends in FM coverage and contract alignment over time
    to detect improving, stable, or degrading conditions.
    
    Args:
        posture_snapshots: Sequence of FM posture snapshots (from build_fm_posture())
        label_indexes: Sequence of label contract indexes (from build_label_contract_index())
    
    Returns:
        {
            "schema_version": "1.0.0",
            "version_count": int,
            "coverage_trend": "IMPROVING" | "STABLE" | "DEGRADING",
            "contract_trend": "IMPROVING" | "STABLE" | "DEGRADING",
            "neutral_notes": List[str]
        }
    
    Trend Logic:
        - IMPROVING: Latest value better than earliest (coverage up, contract status improved)
        - DEGRADING: Latest value worse than earliest (coverage down, contract status degraded)
        - STABLE: No significant change
    """
    version_count = len(posture_snapshots)
    
    if version_count < 2:
        return {
            "schema_version": "1.0.0",
            "version_count": version_count,
            "coverage_trend": "STABLE",
            "contract_trend": "STABLE",
            "neutral_notes": ["Insufficient data for trend analysis (need at least 2 snapshots)"],
        }
    
    # Extract coverage percentages
    coverage_values = [
        snapshot.get("coverage_pct", 0.0) for snapshot in posture_snapshots
    ]
    
    # Extract contract statuses (map to numeric for comparison)
    contract_status_map = {
        "ALIGNED": 3,
        "PARTIAL": 2,
        "BROKEN": 1,
    }
    
    contract_values = []
    for label_index in label_indexes:
        status = label_index.get("contract_status", "UNKNOWN")
        contract_values.append(contract_status_map.get(status, 0))
    
    # Determine coverage trend
    if len(coverage_values) >= 2:
        first_coverage = coverage_values[0]
        last_coverage = coverage_values[-1]
        coverage_diff = last_coverage - first_coverage
        
        # Threshold for "significant" change: 5 percentage points
        if coverage_diff > 5.0:
            coverage_trend = "IMPROVING"
        elif coverage_diff < -5.0:
            coverage_trend = "DEGRADING"
        else:
            coverage_trend = "STABLE"
    else:
        coverage_trend = "STABLE"
    
    # Determine contract trend
    if len(contract_values) >= 2:
        first_contract = contract_values[0]
        last_contract = contract_values[-1]
        contract_diff = last_contract - first_contract
        
        if contract_diff > 0:
            contract_trend = "IMPROVING"
        elif contract_diff < 0:
            contract_trend = "DEGRADING"
        else:
            contract_trend = "STABLE"
    else:
        contract_trend = "STABLE"
    
    # Build neutral notes
    neutral_notes: List[str] = []
    neutral_notes.append(f"Analyzed {version_count} snapshot(s)")
    
    if coverage_trend == "IMPROVING":
        neutral_notes.append(f"Coverage trend: improving (from {coverage_values[0]:.1f}% to {coverage_values[-1]:.1f}%)")
    elif coverage_trend == "DEGRADING":
        neutral_notes.append(f"Coverage trend: degrading (from {coverage_values[0]:.1f}% to {coverage_values[-1]:.1f}%)")
    else:
        neutral_notes.append(f"Coverage trend: stable (around {coverage_values[0]:.1f}%)")
    
    # Build contract status names for notes
    reverse_map = {v: k for k, v in contract_status_map.items()}
    first_status = reverse_map.get(contract_values[0], "UNKNOWN") if contract_values else "UNKNOWN"
    last_status = reverse_map.get(contract_values[-1], "UNKNOWN") if contract_values else "UNKNOWN"
    
    if contract_trend == "IMPROVING":
        neutral_notes.append(f"Contract trend: improving (from {first_status} to {last_status})")
    elif contract_trend == "DEGRADING":
        neutral_notes.append(f"Contract trend: degrading (from {first_status} to {last_status})")
    else:
        neutral_notes.append(f"Contract trend: stable ({first_status})")
    
    return {
        "schema_version": "1.0.0",
        "version_count": version_count,
        "coverage_trend": coverage_trend,
        "contract_trend": contract_trend,
        "neutral_notes": neutral_notes,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI Commands
# ─────────────────────────────────────────────────────────────────────────────

def cmd_canonicalize(args) -> int:
    """Run canonicalization and produce JSON output."""
    print("=" * 70)
    print("PHASE II — Field Manual Canonicalizer")
    print("=" * 70)
    
    try:
        content, lines = load_fm_tex()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return 2
    
    print(f"\nSource: {FM_TEX_PATH}")
    print(f"Lines: {len(lines)}")
    
    # Build canonical representation
    canon = build_canonical_representation(content, lines)
    
    print(f"\nExtracted:")
    print(f"  Labels: {len(canon.labels)}")
    print(f"  Definitions: {len(canon.definitions)}")
    print(f"  Invariants: {len(canon.invariants)}")
    print(f"  Formulas: {len(canon.formulas)}")
    print(f"  Sections: {len(canon.sections)}")
    
    print(f"\nSource Hash: {canon.source_hash[:16]}...")
    print(f"Signature Hash: {canon.signature_hash[:16]}...")
    
    # Validate
    labels = extract_labels(content, lines)
    errors = []
    errors.extend(validate_label_uniqueness(labels))
    errors.extend(validate_label_ordering(labels))
    errors.extend(validate_label_naming(labels))
    
    if errors:
        print(f"\n⚠️  Validation Issues ({len(errors)}):")
        for err in errors:
            print(f"  - {err}")
    else:
        print("\n✅ Label validation passed")
    
    # Save
    output_path = Path(args.output) if args.output else FM_CANONICAL_PATH
    save_canonical_json(canon, output_path)
    print(f"\nCanonical JSON written to: {output_path}")
    
    print("=" * 70)
    return 0 if not errors else 1


def cmd_audit_refs(args) -> int:
    """Run cross-reference audit."""
    print("=" * 70)
    print("PHASE II — LaTeX Cross-Reference Auditor")
    print("=" * 70)
    
    try:
        content, lines = load_fm_tex()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return 2
    
    labels = extract_labels(content, lines)
    refs = extract_refs(content, lines)
    
    print(f"\nFound {len(labels)} labels and {len(refs)} references")
    
    result = audit_cross_references(labels, refs)
    
    has_issues = False
    
    if result.dangling_refs:
        has_issues = True
        print(f"\n❌ DANGLING REFERENCES ({len(result.dangling_refs)}):")
        print("-" * 50)
        for target, line, ref_type in result.dangling_refs:
            print(f"  Line {line}: \\{ref_type}{{{target}}} - target not found")
    
    if result.unused_labels:
        print(f"\n⚠️  UNUSED LABELS ({len(result.unused_labels)}):")
        print("-" * 50)
        for label, line in result.unused_labels:
            print(f"  Line {line}: \\label{{{label}}} - never referenced")
    
    if result.suggestions:
        print(f"\n💡 SUGGESTIONS:")
        print("-" * 50)
        for suggestion in result.suggestions:
            print(f"  {suggestion}")
    
    if not result.dangling_refs and not result.unused_labels:
        print("\n✅ All cross-references valid")
    
    print("=" * 70)
    return 1 if has_issues else 0


def cmd_drift_check(args) -> int:
    """Run drift detection."""
    print("=" * 70)
    print("PHASE II — Field Manual Drift Guard")
    print("=" * 70)
    
    try:
        content, lines = load_fm_tex()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return 2
    
    result = check_drift(content, lines)
    
    has_errors = False
    
    if result.determinism_violations:
        has_errors = True
        print(f"\n❌ DETERMINISM VIOLATIONS ({len(result.determinism_violations)}):")
        print("-" * 50)
        for line, violation, context in result.determinism_violations:
            print(f"  Line {line}: {violation}")
            print(f"    Context: {context}")
    
    if result.terminology_issues:
        print(f"\n⚠️  TERMINOLOGY ISSUES ({len(result.terminology_issues)}):")
        print("-" * 50)
        for line, found, expected, context in result.terminology_issues:
            print(f"  Line {line}: Found '{found}', expected '{expected}'")
            print(f"    Context: {context}")
    
    if result.structure_issues:
        has_errors = True
        print(f"\n❌ STRUCTURE ISSUES ({len(result.structure_issues)}):")
        print("-" * 50)
        for issue in result.structure_issues:
            print(f"  - {issue}")
    
    if not result.determinism_violations and not result.terminology_issues and not result.structure_issues:
        print("\n✅ No drift detected")
    
    print("=" * 70)
    return 1 if has_errors else 0


def cmd_export_spec(args) -> int:
    """
    Export spec contract from canonical JSON.
    
    The spec contract is the STABLE API for other agents.
    It always includes consumers and label_coverage.
    
    Output: artifacts/spec/fm_spec_contract.json
    """
    print("=" * 70)
    print("PHASE II — Spec Contract Export")
    print("=" * 70)
    
    try:
        # Ensure canonical JSON exists
        if not FM_CANONICAL_PATH.exists():
            print("Canonical JSON not found. Running canonicalization first...")
            content, lines = load_fm_tex()
            canon = build_canonical_representation(content, lines)
            save_canonical_json(canon, FM_CANONICAL_PATH)
        
        # Export spec contract (always with consumers - frozen API)
        print("\nBuilding consumers index...")
        with_consumers = getattr(args, 'with_consumers', True)  # Default to True
        contract = export_spec_contract(FM_CANONICAL_PATH, include_consumers=with_consumers)
        
        out_path = save_spec_contract(contract, Path(args.output) if args.output else None)
        
        print(f"\nSpec Contract:")
        print(f"  Version: {contract['version']}")
        print(f"  Signature Hash: {contract['signature_hash'][:16]}...")
        print(f"  Definitions: {len(contract['definitions'])}")
        print(f"  Invariants: {len(contract['invariants'])}")
        print(f"  Metrics Section: {contract['metrics_section'] or 'not found'}")
        print(f"  Uplift Section: {contract['uplift_section'] or 'not found'}")
        print(f"  Phase Section: {contract['phase_section'] or 'not found'}")
        
        # Always show consumer analysis (frozen API)
        coverage = contract.get('label_coverage', {})
        print(f"\nConsumer Analysis:")
        print(f"  Labels with refs: {coverage.get('labels_with_refs', 0)}")
        print(f"  Total labels: {coverage.get('total_labels', 0)}")
        print(f"  Coverage: {coverage.get('coverage_pct', 0):.1f}%")
        
        print(f"\n✅ Spec contract written to: {out_path}")
        print("=" * 70)
        return 0
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("=" * 70)
        return 1


def cmd_label_drift(args) -> int:
    """Detect label drift between fm.tex and external docs."""
    print("=" * 70)
    print("PHASE II — Label Drift Detector")
    print("=" * 70)
    
    try:
        # Ensure canonical JSON exists
        if not FM_CANONICAL_PATH.exists():
            print("Canonical JSON not found. Running canonicalization first...")
            content, lines = load_fm_tex()
            canon = build_canonical_representation(content, lines)
            save_canonical_json(canon, FM_CANONICAL_PATH)
        
        # Compute drift
        drift = compute_label_drift(FM_CANONICAL_PATH)
        
        print(f"\nLabel Drift Analysis:")
        print(f"  Signature Hash: {drift.fm_signature_hash[:16]}...")
        print(f"  Coverage: {drift.coverage_pct:.1f}%")
        print(f"\n  Well-connected (in fm + referenced): {len(drift.well_connected_labels)}")
        print(f"  FM-only (defined but not referenced): {len(drift.fm_only_labels)}")
        print(f"  External-only (referenced but not defined): {len(drift.external_only_labels)}")
        
        # Show details
        if drift.external_only_labels:
            print(f"\n⚠️  EXTERNAL-ONLY LABELS (potential drift):")
            for label in drift.external_only_labels[:10]:
                print(f"    - {label}")
            if len(drift.external_only_labels) > 10:
                print(f"    ... and {len(drift.external_only_labels) - 10} more")
        
        if drift.fm_only_labels and args.verbose:
            print(f"\n📋 FM-ONLY LABELS (not externally referenced):")
            for label in drift.fm_only_labels[:20]:
                print(f"    - {label}")
            if len(drift.fm_only_labels) > 20:
                print(f"    ... and {len(drift.fm_only_labels) - 20} more")
        
        # Save summary
        out_path = save_label_drift_summary(drift, Path(args.output) if args.output else None)
        print(f"\n📄 Drift summary written to: {out_path}")
        
        print("=" * 70)
        
        # Return code based on external-only labels (potential issues)
        if drift.external_only_labels:
            print("⚠️  DRIFT DETECTED: Some external references lack fm.tex definitions")
            return 1 if args.strict else 0
        else:
            print("✅ NO DRIFT: All external references have fm.tex definitions")
            return 0
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("=" * 70)
        return 1


def cmd_posture(args) -> int:
    """
    Show FM posture snapshot (governance-facing).
    
    Provides a minimal health check view of Field Manual connectivity.
    """
    print("=" * 70)
    print("PHASE II — FM Posture Snapshot")
    print("=" * 70)
    
    try:
        posture = load_fm_posture()
        
        if getattr(args, 'json', False):
            print(json.dumps(posture, indent=2))
        else:
            # Human-readable format
            status_icon = {
                "healthy": "✅",
                "warning": "⚠️",
                "critical": "❌",
            }.get(posture["health_status"], "?")
            
            print(f"\n{status_icon} Health Status: {posture['health_status'].upper()}")
            print(f"\nSignature Hash: {posture['signature_hash'][:16]}...")
            print(f"\nLabel Statistics:")
            print(f"  Total labels: {posture['total_labels']}")
            print(f"  Well-connected: {posture['well_connected_labels']}")
            print(f"  FM-only: {posture['fm_only_labels']}")
            print(f"  External-only: {posture['external_only_labels']}")
            print(f"\nCoverage: {posture['coverage_pct']:.1f}%")
        
        print("=" * 70)
        
        # Return code based on health status
        if posture["health_status"] == "critical":
            return 1
        return 0
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("Run 'fm_canonicalize.py full' first to generate artifacts.")
        print("=" * 70)
        return 2


def cmd_consumers_view(args) -> int:
    """
    Show cross-doc FM consumers view.
    
    Displays which documents rely most heavily on fm.tex labels.
    """
    print("=" * 70)
    print("PHASE II — Cross-Doc FM Consumers View")
    print("=" * 70)
    
    try:
        if not FM_SPEC_CONTRACT_PATH.exists():
            raise FileNotFoundError(f"Spec contract not found: {FM_SPEC_CONTRACT_PATH}")
        
        with open(FM_SPEC_CONTRACT_PATH, "r", encoding="utf-8") as f:
            contract = json.load(f)
        
        view = build_fm_consumers_view(contract)
        
        if getattr(args, 'json', False):
            print(json.dumps(view, indent=2))
        else:
            if not view:
                print("\nNo external document references found.")
            else:
                print(f"\n{'Document':<40} {'Total Refs':>12} {'Labels':>10}")
                print("-" * 64)
                for entry in view:
                    print(f"{entry['doc']:<40} {entry['total_refs']:>12} {entry['label_count']:>10}")
                
                total_refs = sum(e["total_refs"] for e in view)
                total_labels = sum(e["label_count"] for e in view)
                print("-" * 64)
                print(f"{'TOTAL':<40} {total_refs:>12} {total_labels:>10}")
        
        print("=" * 70)
        return 0
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("Run 'fm_canonicalize.py full' first to generate artifacts.")
        print("=" * 70)
        return 2


def cmd_governance(args) -> int:
    """
    Show FM governance snapshot.
    
    Provides governance-facing status with governance_status indicator.
    """
    print("=" * 70)
    print("PHASE III — FM Governance Snapshot")
    print("=" * 70)
    
    try:
        posture = load_fm_posture()
        governance = build_fm_governance_snapshot(posture)
        
        if getattr(args, 'json', False):
            print(json.dumps(governance, indent=2))
        else:
            # Human-readable format
            gov_icon = {
                "OK": "✅",
                "WARN": "⚠️",
                "ATTENTION": "🔴",
            }.get(governance["governance_status"], "?")
            
            print(f"\n{gov_icon} Governance Status: {governance['governance_status']}")
            print(f"Health Status: {governance['health_status'].upper()}")
            print(f"\nLabel Statistics:")
            print(f"  Total labels: {governance['total_labels']}")
            print(f"  Well-connected: {governance['well_connected_labels']}")
            print(f"  FM-only: {governance['fm_only_labels']}")
            print(f"  External-only: {governance['external_only_labels']}")
            print(f"\nCoverage: {governance['coverage_pct']:.1f}%")
        
        print("=" * 70)
        
        # Return code based on governance status
        if governance["governance_status"] == "ATTENTION":
            return 2
        elif governance["governance_status"] == "WARN":
            return 1
        return 0
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("Run 'fm_canonicalize.py full' first to generate artifacts.")
        print("=" * 70)
        return 2


def cmd_alignment(args) -> int:
    """
    Show cross-system alignment indicator.
    
    Analyzes distribution of FM label references across external docs.
    """
    print("=" * 70)
    print("PHASE III — Cross-System Alignment Indicator")
    print("=" * 70)
    
    try:
        if not FM_SPEC_CONTRACT_PATH.exists():
            raise FileNotFoundError(f"Spec contract not found: {FM_SPEC_CONTRACT_PATH}")
        
        with open(FM_SPEC_CONTRACT_PATH, "r", encoding="utf-8") as f:
            contract = json.load(f)
        
        consumers_view = build_fm_consumers_view(contract)
        alignment = compute_alignment_indicator(consumers_view)
        
        if getattr(args, 'json', False):
            print(json.dumps(alignment, indent=2))
        else:
            status_icon = {
                "WELL_DISTRIBUTED": "✅",
                "CONCENTRATED": "⚠️",
                "SPARSE": "📊",
            }.get(alignment["alignment_status"], "?")
            
            print(f"\n{status_icon} Alignment Status: {alignment['alignment_status']}")
            
            if alignment["heaviest_consumers"]:
                print(f"\nHeaviest Consumers (top 3):")
                for entry in alignment["heaviest_consumers"]:
                    print(f"  {entry['doc']}: {entry['total_refs']} refs, {entry['label_count']} labels")
            
            if alignment["docs_with_low_label_diversity"]:
                print(f"\n⚠️  Docs with Low Label Diversity:")
                for entry in alignment["docs_with_low_label_diversity"]:
                    print(f"  {entry['doc']}: {entry['total_refs']} refs, {entry['label_count']} labels (ratio: {entry['diversity_ratio']:.2f})")
            else:
                print("\n✅ All docs show good label diversity")
        
        print("=" * 70)
        return 0
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("Run 'fm_canonicalize.py full' first to generate artifacts.")
        print("=" * 70)
        return 2


def cmd_global_health(args) -> int:
    """
    Show global health FM signal.
    
    Provides minimal, actionable summary for global system health monitoring.
    """
    print("=" * 70)
    print("PHASE III — Global Health FM Signal")
    print("=" * 70)
    
    try:
        # Load posture and compute governance
        posture = load_fm_posture()
        governance = build_fm_governance_snapshot(posture)
        
        # Load contract and compute alignment
        if not FM_SPEC_CONTRACT_PATH.exists():
            raise FileNotFoundError(f"Spec contract not found: {FM_SPEC_CONTRACT_PATH}")
        
        with open(FM_SPEC_CONTRACT_PATH, "r", encoding="utf-8") as f:
            contract = json.load(f)
        
        consumers_view = build_fm_consumers_view(contract)
        alignment = compute_alignment_indicator(consumers_view)
        
        # Compute global health summary
        global_health = summarize_fm_for_global_health(governance, alignment)
        
        if getattr(args, 'json', False):
            print(json.dumps(global_health, indent=2))
        else:
            status_icon = {
                "OK": "✅",
                "WARN": "⚠️",
                "BLOCK": "🔴",
            }.get(global_health["status"], "?")
            
            print(f"\n{status_icon} FM Status: {global_health['status']}")
            print(f"FM OK: {global_health['fm_ok']}")
            print(f"\nCoverage: {global_health['coverage_pct']:.1f}%")
            print(f"External-only labels: {global_health['external_only_labels']}")
            print(f"Alignment: {global_health['alignment_status']}")
        
        print("=" * 70)
        
        # Return code based on status
        if global_health["status"] == "BLOCK":
            return 2
        elif global_health["status"] == "WARN":
            return 1
        return 0
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("Run 'fm_canonicalize.py full' first to generate artifacts.")
        print("=" * 70)
        return 2


def cmd_label_contract_index(args) -> int:
    """
    Build cross-agent label contract index.
    
    Checks alignment between Field Manual, taxonomy, and curriculum.
    """
    print("=" * 70)
    print("PHASE IV — Cross-Agent Label Contract Index")
    print("=" * 70)
    
    try:
        # Load spec contract
        if not FM_SPEC_CONTRACT_PATH.exists():
            raise FileNotFoundError(f"Spec contract not found: {FM_SPEC_CONTRACT_PATH}")
        
        with open(FM_SPEC_CONTRACT_PATH, "r", encoding="utf-8") as f:
            spec_contract = json.load(f)
        
        # Load taxonomy and curriculum (optional)
        taxonomy_semantics: Dict[str, Any] = {}
        curriculum_manifest: Dict[str, Any] = {}
        
        if getattr(args, 'taxonomy', None):
            taxonomy_path = Path(args.taxonomy)
            if taxonomy_path.exists():
                with open(taxonomy_path, "r", encoding="utf-8") as f:
                    taxonomy_semantics = json.load(f)
            else:
                print(f"WARNING: Taxonomy file not found: {taxonomy_path}")
        
        if getattr(args, 'curriculum', None):
            curriculum_path = Path(args.curriculum)
            if curriculum_path.exists():
                with open(curriculum_path, "r", encoding="utf-8") as f:
                    curriculum_manifest = json.load(f)
            else:
                print(f"WARNING: Curriculum file not found: {curriculum_path}")
        
        # Build label contract index
        label_index = build_label_contract_index(
            spec_contract, taxonomy_semantics, curriculum_manifest
        )
        
        if getattr(args, 'json', False):
            print(json.dumps(label_index, indent=2))
        else:
            status_icon = {
                "ALIGNED": "✅",
                "PARTIAL": "⚠️",
                "BROKEN": "🔴",
            }.get(label_index["contract_status"], "?")
            
            print(f"\n{status_icon} Contract Status: {label_index['contract_status']}")
            print(f"\nLabel Index: {len(label_index['label_index'])} labels")
            print(f"  Missing in taxonomy: {len(label_index['labels_missing_in_taxonomy'])}")
            print(f"  Missing in curriculum: {len(label_index['labels_missing_in_curriculum'])}")
            print(f"  Missing in FM: {len(label_index['labels_missing_in_fm'])}")
            
            if label_index["labels_missing_in_fm"]:
                print(f"\n⚠️  Labels missing in FM:")
                for label in label_index["labels_missing_in_fm"][:10]:
                    print(f"    - {label}")
                if len(label_index["labels_missing_in_fm"]) > 10:
                    print(f"    ... and {len(label_index['labels_missing_in_fm']) - 10} more")
        
        print("=" * 70)
        
        # Return code based on contract status
        if label_index["contract_status"] == "BROKEN":
            return 2
        elif label_index["contract_status"] == "PARTIAL":
            return 1
        return 0
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("Run 'fm_canonicalize.py full' first to generate artifacts.")
        print("=" * 70)
        return 2


def cmd_integration_contract(args) -> int:
    """
    Show agent integration contract.
    
    Provides contract information for agent integration.
    """
    print("=" * 70)
    print("PHASE IV — Agent Integration Contract")
    print("=" * 70)
    
    try:
        # Load label index (from file or build on the fly)
        if getattr(args, 'label_index', None):
            label_index_path = Path(args.label_index)
            if not label_index_path.exists():
                raise FileNotFoundError(f"Label index not found: {label_index_path}")
            with open(label_index_path, "r", encoding="utf-8") as f:
                label_index = json.load(f)
        else:
            # Build minimal index from spec contract only
            if not FM_SPEC_CONTRACT_PATH.exists():
                raise FileNotFoundError(f"Spec contract not found: {FM_SPEC_CONTRACT_PATH}")
            with open(FM_SPEC_CONTRACT_PATH, "r", encoding="utf-8") as f:
                spec_contract = json.load(f)
            # Build with empty taxonomy/curriculum
            label_index = build_label_contract_index(spec_contract, {}, {})
        
        # Build integration contract
        integration_contract = build_field_manual_integration_contract(label_index)
        
        if getattr(args, 'json', False):
            print(json.dumps(integration_contract, indent=2))
        else:
            status_icon = {
                "OK": "✅",
                "WARN": "⚠️",
                "BLOCK": "🔴",
            }.get(integration_contract["integration_status"], "?")
            
            print(f"\n{status_icon} Integration Status: {integration_contract['integration_status']}")
            print(f"\nRequired Label Sets:")
            print(f"  Curriculum: {len(integration_contract['required_label_sets']['curriculum'])} labels")
            print(f"  Taxonomy: {len(integration_contract['required_label_sets']['taxonomy'])} labels")
            print(f"  Docs (FM): {len(integration_contract['required_label_sets']['docs'])} labels")
            print(f"\nNotes: {integration_contract['notes']}")
        
        print("=" * 70)
        
        # Return code based on integration status
        if integration_contract["integration_status"] == "BLOCK":
            return 2
        elif integration_contract["integration_status"] == "WARN":
            return 1
        return 0
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("Run 'fm_canonicalize.py full' first to generate artifacts.")
        print("=" * 70)
        return 2


def cmd_director_panel(args) -> int:
    """
    Show Field Manual director panel.
    
    Provides high-level governance view for director dashboards.
    """
    print("=" * 70)
    print("PHASE IV — Field Manual Director Panel")
    print("=" * 70)
    
    try:
        # Load FM posture
        posture = load_fm_posture()
        
        # Build integration contract (minimal, from spec contract only)
        if not FM_SPEC_CONTRACT_PATH.exists():
            raise FileNotFoundError(f"Spec contract not found: {FM_SPEC_CONTRACT_PATH}")
        with open(FM_SPEC_CONTRACT_PATH, "r", encoding="utf-8") as f:
            spec_contract = json.load(f)
        
        label_index = build_label_contract_index(spec_contract, {}, {})
        integration_contract = build_field_manual_integration_contract(label_index)
        
        # Build director panel
        panel = build_field_manual_director_panel(posture, integration_contract)
        
        if getattr(args, 'json', False):
            print(json.dumps(panel, indent=2))
        else:
            print(f"\n{panel['status_light']} Status Light: {panel['status_light']}")
            print(f"\nHealth Status: {panel['health_status'].upper()}")
            print(f"Coverage: {panel['coverage_pct']:.1f}%")
            print(f"Integration Status: {panel['integration_status']}")
            print(f"\nHeadline: {panel['headline']}")
        
        print("=" * 70)
        
        # Return code based on status
        if panel["integration_status"] == "BLOCK" or panel["health_status"] == "critical":
            return 2
        elif panel["integration_status"] == "WARN" or panel["health_status"] == "warning":
            return 1
        return 0
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("Run 'fm_canonicalize.py full' first to generate artifacts.")
        print("=" * 70)
        return 2


def cmd_cross_check(args) -> int:
    """Cross-check fm.tex against external documentation."""
    print("=" * 70)
    print("PHASE II — Field Manual Cross-Check")
    print("=" * 70)
    
    try:
        # Ensure canonical JSON exists
        if not FM_CANONICAL_PATH.exists():
            print("Canonical JSON not found. Running canonicalization first...")
            content, lines = load_fm_tex()
            canon = build_canonical_representation(content, lines)
            save_canonical_json(canon, FM_CANONICAL_PATH)
        
        # Run cross-check
        summary = run_cross_check(FM_CANONICAL_PATH)
        
        print(f"\nDocuments Checked: {summary.docs_checked}")
        print(f"  ✅ Passed: {summary.docs_passed}")
        print(f"  ⚠️  Warned: {summary.docs_warned}")
        print(f"  ❌ Failed: {summary.docs_failed}")
        
        # Show details
        for result in summary.results:
            status_icon = {"pass": "✅", "warn": "⚠️", "fail": "❌", "skip": "⏭️"}.get(result["status"], "?")
            doc_name = Path(result["doc_path"]).name
            print(f"\n{status_icon} {doc_name}:")
            
            if result["missing_labels"]:
                print(f"    Missing labels: {result['missing_labels']}")
            if result["undefined_terms"]:
                print(f"    Missing terms: {result['undefined_terms'][:5]}{'...' if len(result['undefined_terms']) > 5 else ''}")
            if result["status"] == "pass":
                print("    All checks passed")
        
        # Generate and save report
        if args.output:
            report = generate_cross_check_report(summary)
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(report)
            print(f"\n📄 Report written to: {out_path}")
        
        # Also save JSON summary
        json_out = FM_SPEC_CONTRACT_PATH.parent / "cross_check_summary.json"
        json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": summary.timestamp,
                "fm_signature_hash": summary.fm_signature_hash,
                "docs_checked": summary.docs_checked,
                "docs_passed": summary.docs_passed,
                "docs_warned": summary.docs_warned,
                "docs_failed": summary.docs_failed,
                "results": summary.results
            }, f, indent=2)
        print(f"📄 JSON summary written to: {json_out}")
        
        print("\n" + "=" * 70)
        
        # Return code based on failures
        if summary.docs_failed > 0:
            print("❌ CROSS-CHECK FAILED")
            return 1
        elif summary.docs_warned > 0:
            print("⚠️  CROSS-CHECK PASSED WITH WARNINGS")
            return 0
        else:
            print("✅ CROSS-CHECK PASSED")
            return 0
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("=" * 70)
        return 1


def cmd_full(args) -> int:
    """
    Run all checks and produce canonical output (CI-friendly).
    
    CI INTEGRATION:
        uv run python scripts/fm_canonicalize.py full
    
    This command runs 5 stages:
        1. Canonicalization → docs/fm_canonical.json
        2. Spec Contract Export → artifacts/spec/fm_spec_contract.json
        3. Label Drift Detection → artifacts/spec/label_drift_summary.json
        4. Cross-Reference Audit
        5. Drift Detection (determinism contract)
    
    Exit codes:
        0 = All checks passed
        1 = Issues detected (fails CI)
        2 = Fatal error (missing files, etc.)
    """
    print("=" * 70)
    print("PHASE II — Field Manual Full Analysis")
    print("=" * 70)
    
    exit_code = 0
    
    # Stage 1: Canonicalize
    print("\n[1/5] CANONICALIZATION")
    code = cmd_canonicalize(args)
    if code > exit_code:
        exit_code = code
    
    # Stage 2: Export spec contract (always with consumers)
    print("\n[2/5] SPEC CONTRACT EXPORT (with consumers)")
    spec_args = argparse.Namespace(output=None, with_consumers=True)
    code = cmd_export_spec(spec_args)
    if code > exit_code:
        exit_code = code
    
    # Stage 3: Label drift detection
    print("\n[3/5] LABEL DRIFT DETECTION")
    drift_args = argparse.Namespace(output=None, verbose=False, strict=False)
    code = cmd_label_drift(drift_args)
    if code > exit_code:
        exit_code = code
    
    # Stage 4: Audit refs
    print("\n[4/5] CROSS-REFERENCE AUDIT")
    code = cmd_audit_refs(args)
    if code > exit_code:
        exit_code = code
    
    # Stage 5: Drift check (determinism contract)
    print("\n[5/5] DRIFT DETECTION (determinism contract)")
    code = cmd_drift_check(args)
    if code > exit_code:
        exit_code = code
    
    print("\n" + "=" * 70)
    if exit_code == 0:
        print("✅ ALL CHECKS PASSED")
    else:
        print("❌ ISSUES DETECTED")
    print("=" * 70)
    
    # Summary of generated artifacts
    print("\nGenerated Artifacts:")
    print(f"  📄 {FM_CANONICAL_PATH}")
    print(f"  📄 {FM_SPEC_CONTRACT_PATH}")
    print(f"  📄 {LABEL_DRIFT_PATH}")
    
    return exit_code


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Field Manual Canonicalizer & Intelligence System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # canonicalize command
    p_canon = subparsers.add_parser("canonicalize", help="Extract and serialize fm.tex structure")
    p_canon.add_argument("--output", "-o", help="Output path for canonical JSON")
    
    # audit-refs command
    p_audit = subparsers.add_parser("audit-refs", help="Audit cross-references")
    
    # drift-check command
    p_drift = subparsers.add_parser("drift-check", help="Check for drift")
    
    # export-spec command
    p_spec = subparsers.add_parser("export-spec", help="Export spec contract for other agents")
    p_spec.add_argument("--output", "-o", help="Output path for spec contract JSON")
    p_spec.add_argument("--with-consumers", action="store_true", default=True,
                        help="Include consumer mapping (default: True, always included)")
    p_spec.add_argument("--no-consumers", dest="with_consumers", action="store_false",
                        help="Exclude consumer mapping (not recommended)")
    
    # cross-check command
    p_cross = subparsers.add_parser("cross-check", help="Cross-check against external docs")
    p_cross.add_argument("--output", "-o", help="Output path for markdown report")
    
    # label-drift command
    p_ldrift = subparsers.add_parser("label-drift", help="Detect label drift between fm.tex and docs")
    p_ldrift.add_argument("--output", "-o", help="Output path for drift summary JSON")
    p_ldrift.add_argument("--verbose", "-v", action="store_true",
                          help="Show fm-only labels in output")
    p_ldrift.add_argument("--strict", action="store_true",
                          help="Fail if external-only labels exist")
    
    # posture command (governance-facing summary)
    p_posture = subparsers.add_parser("posture", help="Show FM posture snapshot (governance-facing)")
    p_posture.add_argument("--json", action="store_true", help="Output as JSON")
    
    # consumers-view command
    p_consumers = subparsers.add_parser("consumers-view", help="Show cross-doc FM consumers view")
    p_consumers.add_argument("--json", action="store_true", help="Output as JSON")
    
    # governance command
    p_gov = subparsers.add_parser("governance", help="Show FM governance snapshot")
    p_gov.add_argument("--json", action="store_true", help="Output as JSON")
    
    # alignment command
    p_align = subparsers.add_parser("alignment", help="Show cross-system alignment indicator")
    p_align.add_argument("--json", action="store_true", help="Output as JSON")
    
    # global-health command
    p_health = subparsers.add_parser("global-health", help="Show global health FM signal")
    p_health.add_argument("--json", action="store_true", help="Output as JSON")
    
    # label-contract-index command (Phase IV)
    p_label_index = subparsers.add_parser("label-contract-index", help="Build cross-agent label contract index")
    p_label_index.add_argument("--taxonomy", type=Path, help="Path to taxonomy semantics JSON")
    p_label_index.add_argument("--curriculum", type=Path, help="Path to curriculum manifest JSON")
    p_label_index.add_argument("--json", action="store_true", help="Output as JSON")
    
    # integration-contract command (Phase IV)
    p_integration = subparsers.add_parser("integration-contract", help="Show agent integration contract")
    p_integration.add_argument("--label-index", type=Path, help="Path to label contract index JSON")
    p_integration.add_argument("--json", action="store_true", help="Output as JSON")
    
    # director-panel command (Phase IV)
    p_director = subparsers.add_parser("director-panel", help="Show Field Manual director panel")
    p_director.add_argument("--json", action="store_true", help="Output as JSON")
    
    # full command (CI-friendly)
    p_full = subparsers.add_parser("full", help="Run all checks (CI-friendly)")
    p_full.add_argument("--output", "-o", help="Output path for canonical JSON")
    
    args = parser.parse_args()
    
    if args.command == "canonicalize":
        return cmd_canonicalize(args)
    elif args.command == "audit-refs":
        return cmd_audit_refs(args)
    elif args.command == "drift-check":
        return cmd_drift_check(args)
    elif args.command == "export-spec":
        return cmd_export_spec(args)
    elif args.command == "cross-check":
        return cmd_cross_check(args)
    elif args.command == "label-drift":
        return cmd_label_drift(args)
    elif args.command == "posture":
        return cmd_posture(args)
    elif args.command == "consumers-view":
        return cmd_consumers_view(args)
    elif args.command == "governance":
        return cmd_governance(args)
    elif args.command == "alignment":
        return cmd_alignment(args)
    elif args.command == "global-health":
        return cmd_global_health(args)
    elif args.command == "label-contract-index":
        return cmd_label_contract_index(args)
    elif args.command == "integration-contract":
        return cmd_integration_contract(args)
    elif args.command == "director-panel":
        return cmd_director_panel(args)
    elif args.command == "full":
        return cmd_full(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())

