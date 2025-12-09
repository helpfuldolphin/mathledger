#!/usr/bin/env python3
"""
narrative_consistency_index.py — Narrative Consistency Index (NCI) System

PHASE II — DOC OPS (E5) — MANDATE: NARRATIVE COHESION STABILIZATION

ABSOLUTE SAFEGUARDS:
  - No editing substantive claims in documentation.
  - No modifying success criteria or scientific statements.

This module provides:
  1. Narrative Consistency Index (NCI) computation
  2. Terminology alignment scoring
  3. Phase usage discipline measurement
  4. Uplift claim avoidance verification
  5. Structural coherence analysis
  6. Git-based drift detection (with silent drift detection)
  7. Documentation advisor suggestions
  8. Bucketed NCI report generation
  9. Commit-level narrative delta comparison
  10. Hot spots analysis for prioritized author guidance
  11. Ten-minute fix suggestions (low-effort, high-impact)
  12. CI dashboard summary (GitHub Summary output)
  13. NCI Insight Grid v1.3:
      - Area-based NCI variance view (build_nci_area_view)
      - Time-slice snapshot comparison (compare_nci_snapshots)
      - Dashboard summary JSON (build_nci_insight_summary)
  14. NCI Alerting & Health Dashboard (Phase III):
      - SLO evaluation (evaluate_nci_slo)
      - Alert suggestions (build_nci_alerts)
      - Global health signal (summarize_nci_for_global_health)
  15. NCI as Narrative Health Signal (Phase IV):
      - Work priority view (build_nci_work_priority_view)
      - Doc-weaver integration contract (build_nci_contract_for_doc_tools)
      - Director narrative panel (build_nci_director_panel)
  16. Cross-Run Narrative Stability & Doc-Weaver Feedback Loop:
      - Stability timeline tracking (build_nci_stability_timeline)
      - Enhanced doc-weaver contract v2 (build_nci_contract_for_doc_tools_v2)

Outputs:
  - narrative_index.json: Detailed consistency metrics
  - narrative_heatmap.png: Visual consistency heatmap (deterministic)
  - bucket_report.md: Markdown report by category
  - narrative_delta.json: Commit-level NCI comparison (with silent_drift_files)
  - narrative_hotspots.json: Prioritized inconsistency hot spots
  - quick_fixes.json: Low-effort fix suggestions with priority scores

Usage:
    python scripts/narrative_consistency_index.py [--mode MODE] [--output-dir DIR]
    
Modes:
    index         - Compute NCI and generate outputs (default)
    drift         - Detect narrative drift across commits
    advisor       - Provide documentation improvement suggestions
    bucket-report - Generate bucketed NCI report (Markdown)
    delta-since   - Compute NCI delta between commits (requires --base-commit)
                    Now includes silent_drift_files for unchanged files with NCI shift
    hotspots      - Generate prioritized hot spots analysis
    ci-summary    - Print dashboard summary for GitHub Actions (always exits 0)
    quick-fixes   - Generate ten-minute fix suggestions (sorted by priority)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Sequence

# Optional imports for visualization
try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for determinism
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# ==============================================================================
# CANONICAL DEFINITIONS (from check_narrative_consistency.py)
# ==============================================================================

CANONICAL_TERMS = {
    "RFL": {
        "canonical": "Reflexive Formal Learning",
        "variants": ["RLVF", "Reflective Feedback Loop", "RFL Loop"],
        "weight": 1.0,
    },
    "Phase_II": {
        "canonical": "Phase II",
        "variants": ["Phase 2", "Phase-II", "phase ii", "PHASE-II", "Phase-2"],
        "weight": 0.8,
    },
    "H_t": {
        "canonical": "H_t",
        "variants": ["Ht", "H(t)", "H-t"],
        "weight": 0.6,
    },
    "R_t": {
        "canonical": "R_t",
        "variants": ["Rt", "R(t)", "R-t"],
        "weight": 0.6,
    },
    "U_t": {
        "canonical": "U_t",
        "variants": ["Ut", "U(t)", "U-t"],
        "weight": 0.6,
    },
}

UPLIFT_CLAIM_PATTERNS = [
    r"(?<!no )(?<!not )uplift\s+(achieved|demonstrated|proven|confirmed|observed|shown)",
    r"(?<!no )(?<!not )(?<!without )shows?\s+uplift",
    r"uplift\s+of\s+\d+%",
    r"significant\s+uplift",
    r"measurable\s+uplift\s+(?!plan|experiment|gate|slice)",
]

ALLOWED_UPLIFT_CONTEXTS = [
    r"no uplift",
    r"not.*uplift",
    r"without uplift",
    r"uplift\s+plan",
    r"uplift\s+experiment",
    r"uplift\s+gate",
    r"uplift\s+slice",
    r"measure.*uplift",
    r"test.*uplift",
    r"Phase II.*uplift",
    r"if.*uplift",
    r"when.*uplift",
    r"negative control",
    r"NOT.*RUN",
]

STRUCTURAL_MARKERS = {
    "safeguard_banner": r"ABSOLUTE SAFEGUARD|PHASE II.*NOT.*RUN|NO UPLIFT CLAIM",
    "cross_reference": r"docs/\w+\.md|See\s+\[|See\s+`",
    "version_marker": r"Version:\s*[\d.]+|v\d+\.\d+",
    "status_marker": r"Status:\s*\w+|STATUS:|✅|❌|⚠️",
}

DOCUMENT_CATEGORIES = {
    "paper": ["paper/**/*.tex"],
    "docs": ["docs/**/*.md"],
    "governance": ["governance_verdict.md", "VSD*.md", "*LAW*.md"],
    "config": ["config/*.yaml", "configs/**/*.yaml"],
    "readme": ["README*.md", "AGENTS.md"],
}

# ==============================================================================
# DATA STRUCTURES
# ==============================================================================


@dataclass
class TerminologyScore:
    """Score for terminology alignment in a document."""
    canonical_count: int = 0
    variant_count: int = 0
    total_terms: int = 0
    alignment_ratio: float = 1.0
    violations: list[dict] = field(default_factory=list)


@dataclass
class PhaseScore:
    """Score for Phase terminology discipline."""
    canonical_count: int = 0
    non_canonical_count: int = 0
    discipline_ratio: float = 1.0
    violations: list[dict] = field(default_factory=list)


@dataclass
class UpliftScore:
    """Score for uplift claim avoidance."""
    safe_references: int = 0
    potential_claims: int = 0
    avoidance_ratio: float = 1.0
    violations: list[dict] = field(default_factory=list)


@dataclass
class StructuralScore:
    """Score for structural coherence."""
    has_safeguard_banner: bool = False
    has_cross_references: bool = False
    has_version_marker: bool = False
    has_status_marker: bool = False
    coherence_ratio: float = 0.0


@dataclass
class DocumentMetrics:
    """Complete metrics for a single document."""
    path: str
    category: str
    line_count: int
    terminology: TerminologyScore = field(default_factory=TerminologyScore)
    phase: PhaseScore = field(default_factory=PhaseScore)
    uplift: UpliftScore = field(default_factory=UpliftScore)
    structure: StructuralScore = field(default_factory=StructuralScore)
    nci_score: float = 0.0
    
    def compute_nci(self) -> float:
        """Compute the Narrative Consistency Index for this document."""
        weights = {
            "terminology": 0.30,
            "phase": 0.25,
            "uplift": 0.30,
            "structure": 0.15,
        }
        
        scores = {
            "terminology": self.terminology.alignment_ratio,
            "phase": self.phase.discipline_ratio,
            "uplift": self.uplift.avoidance_ratio,
            "structure": self.structure.coherence_ratio,
        }
        
        self.nci_score = sum(weights[k] * scores[k] for k in weights)
        return self.nci_score


@dataclass
class NarrativeIndex:
    """Global Narrative Consistency Index."""
    timestamp: str
    commit_hash: str
    total_documents: int
    global_nci: float
    category_scores: dict[str, float]
    documents: list[DocumentMetrics]
    drift_detected: bool = False
    advisor_suggestions: list[dict] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "commit_hash": self.commit_hash,
            "total_documents": self.total_documents,
            "global_nci": round(self.global_nci, 4),
            "category_scores": {k: round(v, 4) for k, v in self.category_scores.items()},
            "summary": {
                "terminology_alignment": round(
                    sum(d.terminology.alignment_ratio for d in self.documents) / max(len(self.documents), 1), 4
                ),
                "phase_discipline": round(
                    sum(d.phase.discipline_ratio for d in self.documents) / max(len(self.documents), 1), 4
                ),
                "uplift_avoidance": round(
                    sum(d.uplift.avoidance_ratio for d in self.documents) / max(len(self.documents), 1), 4
                ),
                "structural_coherence": round(
                    sum(d.structure.coherence_ratio for d in self.documents) / max(len(self.documents), 1), 4
                ),
            },
            "documents": [
                {
                    "path": d.path,
                    "category": d.category,
                    "nci_score": round(d.nci_score, 4),
                    "line_count": d.line_count,
                    "terminology": {
                        "canonical": d.terminology.canonical_count,
                        "variants": d.terminology.variant_count,
                        "alignment": round(d.terminology.alignment_ratio, 4),
                    },
                    "phase": {
                        "canonical": d.phase.canonical_count,
                        "non_canonical": d.phase.non_canonical_count,
                        "discipline": round(d.phase.discipline_ratio, 4),
                    },
                    "uplift": {
                        "safe": d.uplift.safe_references,
                        "potential_claims": d.uplift.potential_claims,
                        "avoidance": round(d.uplift.avoidance_ratio, 4),
                    },
                    "structure": {
                        "safeguard_banner": d.structure.has_safeguard_banner,
                        "cross_references": d.structure.has_cross_references,
                        "coherence": round(d.structure.coherence_ratio, 4),
                    },
                }
                for d in self.documents
            ],
            "drift_detected": self.drift_detected,
            "advisor_suggestions": self.advisor_suggestions[:20],  # Top 20
        }


@dataclass
class DriftReport:
    """Report on narrative drift between commits."""
    base_commit: str
    head_commit: str
    timestamp: str
    files_changed: int
    terminology_drift: list[dict]
    definition_drift: list[dict]
    nci_delta: float
    drift_severity: Literal["none", "minor", "moderate", "severe"]


@dataclass
class AdvisorSuggestion:
    """A suggestion for improving narrative cohesion."""
    file: str
    line: int
    category: str
    priority: Literal["high", "medium", "low"]
    suggestion: str
    context: str


@dataclass
class BucketSummary:
    """Summary of a document category bucket."""
    category: str
    document_count: int
    avg_nci: float
    avg_terminology: float
    avg_phase: float
    avg_uplift: float
    avg_structure: float
    worst_files: list[tuple[str, float]]  # (path, nci_score)


@dataclass
class NarrativeDelta:
    """NCI delta between two commits."""
    base_commit: str
    head_commit: str
    timestamp: str
    base_nci: float
    head_nci: float
    delta: float
    changed_files: list[str]
    file_deltas: list[dict]  # Per-file NCI changes
    silent_drift_files: list[dict] = field(default_factory=list)  # Files unchanged but NCI shifted


@dataclass
class HotSpot:
    """A documentation hot spot requiring attention."""
    file: str
    category: str
    nci_score: float
    nci_deficit: float  # How much below 1.0
    contribution_pct: float  # Contribution to global NCI drop
    severity_counts: dict[str, int]  # Per-dimension violation counts
    primary_issue: str  # Main dimension causing issues


@dataclass
class TenMinuteFix:
    """A concrete, low-effort documentation fix suggestion."""
    file: str
    issue_type: str
    hint: str
    estimated_effort: str  # Always "<10m"
    word_count: int
    violation_count: int
    priority_score: float  # Higher = more impactful & easier


@dataclass
class SilentDriftFile:
    """A file that didn't change in git but whose NCI contribution shifted."""
    file: str
    base_nci: float
    head_nci: float
    nci_delta: float
    reason: str  # Why the drift occurred (vocabulary shift, etc.)


# ==============================================================================
# CORE ANALYSIS ENGINE
# ==============================================================================


class NarrativeConsistencyIndexer:
    """Computes the Narrative Consistency Index across documentation."""
    
    def __init__(self, repo_root: Path, verbose: bool = False):
        self.repo_root = repo_root
        self.verbose = verbose
        self.documents: list[DocumentMetrics] = []
        
    def collect_documents(self) -> list[Path]:
        """Collect all documents to analyze."""
        files = []
        
        for category, patterns in DOCUMENT_CATEGORIES.items():
            for pattern in patterns:
                matches = list(self.repo_root.glob(pattern))
                files.extend(matches)
        
        # Filter to existing files, exclude generated/binary
        filtered = []
        exclude_patterns = [
            "node_modules", "__pycache__", ".git", "dist", "build",
            "artifacts", "results", ".png", ".jpg", ".pdf"
        ]
        
        for f in files:
            if not f.is_file():
                continue
            path_str = str(f)
            if any(excl in path_str for excl in exclude_patterns):
                continue
            filtered.append(f)
        
        return sorted(set(filtered))
    
    def categorize_document(self, path: Path) -> str:
        """Determine the category of a document."""
        # Normalize path separators for cross-platform compatibility
        rel_path = str(path.relative_to(self.repo_root)).replace("\\", "/")
        
        if rel_path.startswith("paper/"):
            return "paper"
        elif rel_path.startswith("docs/"):
            return "docs"
        elif "governance" in rel_path.lower() or "VSD" in rel_path or "LAW" in rel_path:
            return "governance"
        elif rel_path.startswith("config"):
            return "config"
        elif "README" in rel_path or "AGENTS" in rel_path:
            return "readme"
        else:
            return "other"
    
    def analyze_terminology(self, content: str, lines: list[str]) -> TerminologyScore:
        """Analyze terminology alignment."""
        score = TerminologyScore()
        
        for term_key, term_def in CANONICAL_TERMS.items():
            canonical = term_def["canonical"]
            variants = term_def["variants"]
            
            # Count canonical uses
            canonical_pattern = re.escape(canonical)
            canonical_matches = len(re.findall(canonical_pattern, content, re.IGNORECASE))
            score.canonical_count += canonical_matches
            
            # Count variant uses
            for variant in variants:
                variant_pattern = rf"\b{re.escape(variant)}\b"
                variant_matches = re.finditer(variant_pattern, content, re.IGNORECASE)
                for match in variant_matches:
                    score.variant_count += 1
                    # Find line number
                    pos = match.start()
                    line_num = content[:pos].count('\n') + 1
                    score.violations.append({
                        "term": term_key,
                        "found": match.group(),
                        "expected": canonical,
                        "line": line_num,
                    })
        
        score.total_terms = score.canonical_count + score.variant_count
        if score.total_terms > 0:
            score.alignment_ratio = score.canonical_count / score.total_terms
        else:
            score.alignment_ratio = 1.0  # No terms = no violations
        
        return score
    
    def analyze_phase_discipline(self, content: str) -> PhaseScore:
        """Analyze Phase terminology discipline."""
        score = PhaseScore()
        
        # Canonical: "Phase II", "Phase I", "Phase III"
        canonical_pattern = r"\bPhase\s+[IVX]+\b"
        canonical_matches = re.findall(canonical_pattern, content)
        score.canonical_count = len(canonical_matches)
        
        # Non-canonical variants
        non_canonical_patterns = [
            (r"\bPhase\s+\d+\b", "Arabic numeral"),
            (r"\bPhase-[IVX]+\b", "Hyphenated"),
            (r"\bphase\s+[ivx]+\b", "Lowercase"),
        ]
        
        for pattern, reason in non_canonical_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                score.non_canonical_count += 1
                pos = match.start()
                line_num = content[:pos].count('\n') + 1
                score.violations.append({
                    "found": match.group(),
                    "reason": reason,
                    "line": line_num,
                })
        
        total = score.canonical_count + score.non_canonical_count
        if total > 0:
            score.discipline_ratio = score.canonical_count / total
        else:
            score.discipline_ratio = 1.0
        
        return score
    
    def analyze_uplift_claims(self, content: str) -> UpliftScore:
        """Analyze uplift claim avoidance."""
        score = UpliftScore()
        
        # Check if document contains "uplift" at all
        if "uplift" not in content.lower():
            score.avoidance_ratio = 1.0
            return score
        
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            line_lower = line.lower()
            
            if "uplift" not in line_lower:
                continue
            
            # Check if it's in an allowed context
            is_allowed = False
            for allowed_pattern in ALLOWED_UPLIFT_CONTEXTS:
                if re.search(allowed_pattern, line_lower):
                    is_allowed = True
                    score.safe_references += 1
                    break
            
            if is_allowed:
                continue
            
            # Check for claim patterns
            for claim_pattern in UPLIFT_CLAIM_PATTERNS:
                if re.search(claim_pattern, line_lower):
                    score.potential_claims += 1
                    score.violations.append({
                        "line": line_num,
                        "content": line.strip()[:80],
                        "pattern": claim_pattern,
                    })
                    break
            else:
                # No claim pattern matched, count as safe
                score.safe_references += 1
        
        total = score.safe_references + score.potential_claims
        if total > 0:
            score.avoidance_ratio = score.safe_references / total
        else:
            score.avoidance_ratio = 1.0
        
        return score
    
    def analyze_structure(self, content: str) -> StructuralScore:
        """Analyze structural coherence."""
        score = StructuralScore()
        
        score.has_safeguard_banner = bool(
            re.search(STRUCTURAL_MARKERS["safeguard_banner"], content, re.IGNORECASE)
        )
        score.has_cross_references = bool(
            re.search(STRUCTURAL_MARKERS["cross_reference"], content)
        )
        score.has_version_marker = bool(
            re.search(STRUCTURAL_MARKERS["version_marker"], content)
        )
        score.has_status_marker = bool(
            re.search(STRUCTURAL_MARKERS["status_marker"], content)
        )
        
        # Compute coherence ratio
        markers = [
            score.has_safeguard_banner,
            score.has_cross_references,
            score.has_version_marker,
            score.has_status_marker,
        ]
        score.coherence_ratio = sum(markers) / len(markers)
        
        return score
    
    def analyze_document(self, path: Path) -> DocumentMetrics:
        """Analyze a single document."""
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            content = ""
        
        lines = content.split('\n')
        rel_path = str(path.relative_to(self.repo_root))
        
        metrics = DocumentMetrics(
            path=rel_path,
            category=self.categorize_document(path),
            line_count=len(lines),
        )
        
        metrics.terminology = self.analyze_terminology(content, lines)
        metrics.phase = self.analyze_phase_discipline(content)
        metrics.uplift = self.analyze_uplift_claims(content)
        metrics.structure = self.analyze_structure(content)
        metrics.compute_nci()
        
        return metrics
    
    def compute_index(self) -> NarrativeIndex:
        """Compute the global Narrative Consistency Index."""
        files = self.collect_documents()
        
        if self.verbose:
            print(f"Analyzing {len(files)} documents...")
        
        self.documents = []
        for f in files:
            metrics = self.analyze_document(f)
            self.documents.append(metrics)
            if self.verbose:
                print(f"  {metrics.path}: NCI={metrics.nci_score:.3f}")
        
        # Compute category scores
        category_scores = {}
        for category in DOCUMENT_CATEGORIES.keys():
            cat_docs = [d for d in self.documents if d.category == category]
            if cat_docs:
                category_scores[category] = sum(d.nci_score for d in cat_docs) / len(cat_docs)
            else:
                category_scores[category] = 1.0
        
        # Global NCI
        if self.documents:
            global_nci = sum(d.nci_score for d in self.documents) / len(self.documents)
        else:
            global_nci = 1.0
        
        # Get current commit hash
        commit_hash = self._get_commit_hash()
        
        index = NarrativeIndex(
            timestamp=datetime.utcnow().isoformat() + "Z",
            commit_hash=commit_hash,
            total_documents=len(self.documents),
            global_nci=global_nci,
            category_scores=category_scores,
            documents=self.documents,
        )
        
        return index
    
    def _get_commit_hash(self) -> str:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=self.repo_root,
            )
            return result.stdout.strip()[:12]
        except Exception:
            return "unknown"


# ==============================================================================
# DRIFT DETECTOR
# ==============================================================================


class DriftDetector:
    """Detects narrative drift across git commits."""
    
    def __init__(self, repo_root: Path, verbose: bool = False):
        self.repo_root = repo_root
        self.verbose = verbose
    
    def get_file_at_commit(self, filepath: str, commit: str) -> str | None:
        """Get file contents at a specific commit."""
        try:
            result = subprocess.run(
                ["git", "show", f"{commit}:{filepath}"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=self.repo_root,
            )
            if result.returncode == 0:
                return result.stdout
            return None
        except Exception:
            return None
    
    def get_changed_files(self, base: str, head: str) -> list[str]:
        """Get list of changed documentation files between commits."""
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only", base, head],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=self.repo_root,
            )
            files = result.stdout.strip().split('\n')
            # Filter to documentation files
            doc_files = [
                f for f in files
                if f.endswith(('.md', '.tex', '.yaml', '.yml'))
                and not any(excl in f for excl in ['node_modules', '__pycache__'])
            ]
            return doc_files
        except Exception:
            return []
    
    def detect_terminology_drift(
        self, old_content: str, new_content: str, filepath: str
    ) -> list[dict]:
        """Detect terminology changes between versions."""
        drifts = []
        
        for term_key, term_def in CANONICAL_TERMS.items():
            canonical = term_def["canonical"]
            variants = term_def["variants"]
            
            # Count canonical in old vs new
            old_canonical = len(re.findall(re.escape(canonical), old_content, re.IGNORECASE))
            new_canonical = len(re.findall(re.escape(canonical), new_content, re.IGNORECASE))
            
            # Count variants in old vs new
            old_variants = 0
            new_variants = 0
            for variant in variants:
                pattern = rf"\b{re.escape(variant)}\b"
                old_variants += len(re.findall(pattern, old_content, re.IGNORECASE))
                new_variants += len(re.findall(pattern, new_content, re.IGNORECASE))
            
            # Detect drift
            if new_variants > old_variants:
                drifts.append({
                    "file": filepath,
                    "term": term_key,
                    "drift_type": "variant_increase",
                    "old_count": old_variants,
                    "new_count": new_variants,
                    "canonical": canonical,
                })
            elif new_canonical < old_canonical and old_canonical > 0:
                drifts.append({
                    "file": filepath,
                    "term": term_key,
                    "drift_type": "canonical_decrease",
                    "old_count": old_canonical,
                    "new_count": new_canonical,
                })
        
        return drifts
    
    def detect_definition_drift(
        self, old_content: str, new_content: str, filepath: str
    ) -> list[dict]:
        """Detect changes to key definitions."""
        drifts = []
        
        # Look for definition patterns
        definition_patterns = [
            (r"RFL\s*(?:=|:|\bis\b)\s*([^\n.]+)", "RFL"),
            (r"H_t\s*(?:=|:|\bis\b)\s*([^\n.]+)", "H_t"),
            (r"Phase II\s*(?:=|:|\bis\b)\s*([^\n.]+)", "Phase_II"),
        ]
        
        for pattern, def_name in definition_patterns:
            old_defs = re.findall(pattern, old_content, re.IGNORECASE)
            new_defs = re.findall(pattern, new_content, re.IGNORECASE)
            
            # Normalize and compare
            old_normalized = {d.strip().lower() for d in old_defs}
            new_normalized = {d.strip().lower() for d in new_defs}
            
            if old_normalized and new_normalized and old_normalized != new_normalized:
                drifts.append({
                    "file": filepath,
                    "definition": def_name,
                    "drift_type": "definition_change",
                    "old_definitions": list(old_defs)[:3],
                    "new_definitions": list(new_defs)[:3],
                })
        
        return drifts
    
    def compute_drift(self, base_commit: str = "HEAD~10", head_commit: str = "HEAD") -> DriftReport:
        """Compute narrative drift between two commits."""
        changed_files = self.get_changed_files(base_commit, head_commit)
        
        if self.verbose:
            print(f"Analyzing drift across {len(changed_files)} changed files...")
        
        all_term_drifts = []
        all_def_drifts = []
        
        for filepath in changed_files:
            old_content = self.get_file_at_commit(filepath, base_commit) or ""
            new_content = self.get_file_at_commit(filepath, head_commit) or ""
            
            if not old_content and not new_content:
                continue
            
            term_drifts = self.detect_terminology_drift(old_content, new_content, filepath)
            def_drifts = self.detect_definition_drift(old_content, new_content, filepath)
            
            all_term_drifts.extend(term_drifts)
            all_def_drifts.extend(def_drifts)
        
        # Compute NCI delta (simplified)
        total_drifts = len(all_term_drifts) + len(all_def_drifts)
        nci_delta = -0.01 * total_drifts  # Each drift reduces NCI slightly
        
        # Determine severity
        if total_drifts == 0:
            severity = "none"
        elif total_drifts <= 5:
            severity = "minor"
        elif total_drifts <= 15:
            severity = "moderate"
        else:
            severity = "severe"
        
        return DriftReport(
            base_commit=base_commit,
            head_commit=head_commit,
            timestamp=datetime.utcnow().isoformat() + "Z",
            files_changed=len(changed_files),
            terminology_drift=all_term_drifts,
            definition_drift=all_def_drifts,
            nci_delta=nci_delta,
            drift_severity=severity,
        )


# ==============================================================================
# DOCUMENTATION ADVISOR
# ==============================================================================


class DocumentationAdvisor:
    """Provides suggestions for improving narrative cohesion."""
    
    def __init__(self, index: NarrativeIndex):
        self.index = index
    
    def generate_suggestions(self) -> list[AdvisorSuggestion]:
        """Generate improvement suggestions without auto-editing."""
        suggestions = []
        
        for doc in self.index.documents:
            # Terminology suggestions
            for violation in doc.terminology.violations:
                suggestions.append(AdvisorSuggestion(
                    file=doc.path,
                    line=violation.get("line", 0),
                    category="terminology",
                    priority="medium",
                    suggestion=f"Consider using '{violation['expected']}' instead of '{violation['found']}'",
                    context=f"Term '{violation['term']}' has canonical form '{violation['expected']}'",
                ))
            
            # Phase discipline suggestions
            for violation in doc.phase.violations:
                suggestions.append(AdvisorSuggestion(
                    file=doc.path,
                    line=violation.get("line", 0),
                    category="phase_discipline",
                    priority="low",
                    suggestion=f"Consider using 'Phase II' format instead of '{violation['found']}'",
                    context=f"Reason: {violation['reason']}",
                ))
            
            # Uplift claim warnings
            for violation in doc.uplift.violations:
                suggestions.append(AdvisorSuggestion(
                    file=doc.path,
                    line=violation.get("line", 0),
                    category="uplift_claim",
                    priority="high",
                    suggestion="Review this line for potential unauthorized uplift claim",
                    context=violation.get("content", "")[:60],
                ))
            
            # Structural suggestions
            if doc.category in ["governance", "docs"] and not doc.structure.has_safeguard_banner:
                if "phase" in doc.path.lower() or "uplift" in doc.path.lower():
                    suggestions.append(AdvisorSuggestion(
                        file=doc.path,
                        line=1,
                        category="structure",
                        priority="medium",
                        suggestion="Consider adding ABSOLUTE SAFEGUARDS banner for Phase II documentation",
                        context="Phase II documents should have safeguard banners",
                    ))
            
            if doc.category in ["docs", "governance"] and not doc.structure.has_cross_references:
                if doc.line_count > 50:
                    suggestions.append(AdvisorSuggestion(
                        file=doc.path,
                        line=1,
                        category="structure",
                        priority="low",
                        suggestion="Consider adding cross-references to related documentation",
                        context="Larger documents benefit from internal linking",
                    ))
        
        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        suggestions.sort(key=lambda s: priority_order.get(s.priority, 3))
        
        return suggestions


# ==============================================================================
# BUCKET REPORT GENERATOR
# ==============================================================================


class BucketReportGenerator:
    """Generates bucketed NCI reports by document category."""
    
    def __init__(self, index: NarrativeIndex):
        self.index = index
    
    def compute_bucket_summaries(self) -> list[BucketSummary]:
        """Compute summary statistics for each bucket/category."""
        summaries = []
        
        # Get all unique categories
        categories = set(d.category for d in self.index.documents)
        
        for category in sorted(categories):
            cat_docs = [d for d in self.index.documents if d.category == category]
            
            if not cat_docs:
                continue
            
            n = len(cat_docs)
            
            # Compute averages
            avg_nci = sum(d.nci_score for d in cat_docs) / n
            avg_terminology = sum(d.terminology.alignment_ratio for d in cat_docs) / n
            avg_phase = sum(d.phase.discipline_ratio for d in cat_docs) / n
            avg_uplift = sum(d.uplift.avoidance_ratio for d in cat_docs) / n
            avg_structure = sum(d.structure.coherence_ratio for d in cat_docs) / n
            
            # Find worst 5 files
            sorted_docs = sorted(cat_docs, key=lambda d: d.nci_score)
            worst_files = [(d.path, d.nci_score) for d in sorted_docs[:5]]
            
            summaries.append(BucketSummary(
                category=category,
                document_count=n,
                avg_nci=avg_nci,
                avg_terminology=avg_terminology,
                avg_phase=avg_phase,
                avg_uplift=avg_uplift,
                avg_structure=avg_structure,
                worst_files=worst_files,
            ))
        
        # Sort by NCI (worst first for attention)
        summaries.sort(key=lambda s: s.avg_nci)
        
        return summaries
    
    def generate_markdown_report(self) -> str:
        """Generate a Markdown bucket report."""
        summaries = self.compute_bucket_summaries()
        
        lines = []
        lines.append("# Narrative Consistency Bucket Report")
        lines.append("")
        lines.append(f"**Generated**: {datetime.utcnow().isoformat()}Z")
        lines.append(f"**Commit**: {self.index.commit_hash}")
        lines.append("")
        lines.append(f"## Global NCI: {self.index.global_nci:.2f}")
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("## Buckets")
        lines.append("")
        
        # Summary table
        lines.append("| Bucket | Documents | NCI | Terminology | Phase | Uplift | Structure |")
        lines.append("|--------|-----------|-----|-------------|-------|--------|-----------|")
        
        for s in summaries:
            lines.append(
                f"| {s.category.title()} | {s.document_count} | "
                f"{s.avg_nci:.2f} | {s.avg_terminology:.2f} | "
                f"{s.avg_phase:.2f} | {s.avg_uplift:.2f} | {s.avg_structure:.2f} |"
            )
        
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("## Top 5 Inconsistent Files Per Bucket")
        lines.append("")
        
        for s in summaries:
            lines.append(f"### {s.category.title()} (Avg NCI: {s.avg_nci:.2f})")
            lines.append("")
            
            if s.worst_files:
                for i, (path, score) in enumerate(s.worst_files, 1):
                    # Neutral language: "High variance" not "Bad doc"
                    variance_label = "High variance" if score < 0.5 else "Moderate variance" if score < 0.8 else "Low variance"
                    lines.append(f"{i}. `{path}` — NCI: {score:.2f} ({variance_label})")
            else:
                lines.append("*No files in this bucket*")
            
            lines.append("")
        
        lines.append("---")
        lines.append("")
        lines.append("## Interpretation Guide")
        lines.append("")
        lines.append("- **NCI ≥ 0.90**: Excellent consistency")
        lines.append("- **NCI 0.70-0.89**: Acceptable, minor improvements possible")
        lines.append("- **NCI 0.50-0.69**: Attention needed, moderate inconsistencies")
        lines.append("- **NCI < 0.50**: High variance, review recommended")
        lines.append("")
        lines.append("*Note: This report identifies variance, not quality. Neutral observations only.*")
        
        return "\n".join(lines)


# ==============================================================================
# NARRATIVE DELTA CALCULATOR
# ==============================================================================


class NarrativeDeltaCalculator:
    """Computes NCI delta between two commits."""
    
    def __init__(self, repo_root: Path, verbose: bool = False):
        self.repo_root = repo_root
        self.verbose = verbose
    
    def _get_commit_hash_full(self, commit_ref: str) -> str:
        """Resolve a commit reference to its full hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", commit_ref],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=self.repo_root,
            )
            return result.stdout.strip()[:12]
        except Exception:
            return commit_ref
    
    def _compute_nci_at_commit(self, commit: str) -> tuple[float, dict[str, float]]:
        """
        Compute NCI at a specific commit by checking out files temporarily.
        Returns (global_nci, {path: nci_score}).
        """
        # Get list of doc files at that commit
        try:
            result = subprocess.run(
                ["git", "ls-tree", "-r", "--name-only", commit],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=self.repo_root,
            )
            all_files = result.stdout.strip().split('\n')
        except Exception:
            return 1.0, {}
        
        # Filter to documentation files
        doc_extensions = ('.md', '.tex', '.yaml', '.yml')
        doc_files = [
            f for f in all_files
            if f.endswith(doc_extensions)
            and not any(excl in f for excl in ['node_modules', '__pycache__', 'dist', 'build'])
        ]
        
        if not doc_files:
            return 1.0, {}
        
        # Analyze each file at that commit
        file_scores = {}
        indexer = NarrativeConsistencyIndexer(self.repo_root, verbose=False)
        
        for filepath in doc_files:
            try:
                result = subprocess.run(
                    ["git", "show", f"{commit}:{filepath}"],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    cwd=self.repo_root,
                )
                if result.returncode != 0:
                    continue
                
                content = result.stdout
                lines = content.split('\n')
                
                # Create a temporary metrics object
                metrics = DocumentMetrics(
                    path=filepath,
                    category=self._categorize_path(filepath),
                    line_count=len(lines),
                )
                
                metrics.terminology = indexer.analyze_terminology(content, lines)
                metrics.phase = indexer.analyze_phase_discipline(content)
                metrics.uplift = indexer.analyze_uplift_claims(content)
                metrics.structure = indexer.analyze_structure(content)
                metrics.compute_nci()
                
                file_scores[filepath] = metrics.nci_score
                
            except Exception:
                continue
        
        if file_scores:
            global_nci = sum(file_scores.values()) / len(file_scores)
        else:
            global_nci = 1.0
        
        return global_nci, file_scores
    
    def _categorize_path(self, filepath: str) -> str:
        """Categorize a file path."""
        filepath = filepath.replace("\\", "/")
        
        if filepath.startswith("paper/"):
            return "paper"
        elif filepath.startswith("docs/"):
            return "docs"
        elif "governance" in filepath.lower() or "VSD" in filepath or "LAW" in filepath:
            return "governance"
        elif filepath.startswith("config"):
            return "config"
        elif "README" in filepath or "AGENTS" in filepath:
            return "readme"
        else:
            return "other"
    
    def _get_changed_files(self, base: str, head: str) -> list[str]:
        """Get list of changed documentation files."""
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only", base, head],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=self.repo_root,
            )
            files = result.stdout.strip().split('\n')
            doc_files = [
                f for f in files
                if f and f.endswith(('.md', '.tex', '.yaml', '.yml'))
                and not any(excl in f for excl in ['node_modules', '__pycache__'])
            ]
            return sorted(doc_files)
        except Exception:
            return []
    
    def compute_delta(self, base_commit: str, head_commit: str = "HEAD") -> NarrativeDelta:
        """Compute NCI delta between two commits, including silent drift detection."""
        if self.verbose:
            print(f"Computing NCI delta: {base_commit} → {head_commit}")
        
        base_hash = self._get_commit_hash_full(base_commit)
        head_hash = self._get_commit_hash_full(head_commit)
        
        # Compute NCI at each commit
        if self.verbose:
            print(f"  Analyzing base commit ({base_hash})...")
        base_nci, base_scores = self._compute_nci_at_commit(base_commit)
        
        if self.verbose:
            print(f"  Analyzing head commit ({head_hash})...")
        head_nci, head_scores = self._compute_nci_at_commit(head_commit)
        
        # Get changed files
        changed_files = self._get_changed_files(base_commit, head_commit)
        changed_files_set = set(changed_files)
        
        # Compute per-file deltas and detect silent drift
        file_deltas = []
        silent_drift_files = []
        all_files = set(base_scores.keys()) | set(head_scores.keys())
        
        for filepath in sorted(all_files):
            base_score = base_scores.get(filepath, None)
            head_score = head_scores.get(filepath, None)
            
            if base_score is not None and head_score is not None:
                delta = head_score - base_score
                if abs(delta) > 0.01:  # Only significant changes
                    file_delta_entry = {
                        "file": filepath,
                        "base_nci": round(base_score, 4),
                        "head_nci": round(head_score, 4),
                        "delta": round(delta, 4),
                    }
                    
                    # Check for silent drift: file didn't change in git but NCI shifted
                    if filepath not in changed_files_set:
                        # This is silent drift!
                        reason = self._infer_drift_reason(delta)
                        silent_drift_files.append({
                            "file": filepath,
                            "base_nci": round(base_score, 4),
                            "head_nci": round(head_score, 4),
                            "nci_delta": round(delta, 4),
                            "reason": reason,
                        })
                        file_delta_entry["silent_drift"] = True
                        file_delta_entry["reason"] = reason
                    
                    file_deltas.append(file_delta_entry)
                    
            elif base_score is None and head_score is not None:
                file_deltas.append({
                    "file": filepath,
                    "base_nci": None,
                    "head_nci": round(head_score, 4),
                    "delta": None,
                    "status": "added",
                })
            elif base_score is not None and head_score is None:
                file_deltas.append({
                    "file": filepath,
                    "base_nci": round(base_score, 4),
                    "head_nci": None,
                    "delta": None,
                    "status": "removed",
                })
        
        if self.verbose and silent_drift_files:
            print(f"  Detected {len(silent_drift_files)} silent drift file(s)")
        
        return NarrativeDelta(
            base_commit=base_hash,
            head_commit=head_hash,
            timestamp=datetime.utcnow().isoformat() + "Z",
            base_nci=round(base_nci, 4),
            head_nci=round(head_nci, 4),
            delta=round(head_nci - base_nci, 4),
            changed_files=changed_files,
            file_deltas=file_deltas,
            silent_drift_files=silent_drift_files,
        )
    
    def _infer_drift_reason(self, delta: float) -> str:
        """Infer the reason for silent drift based on the delta direction.
        
        Uses purely descriptive, neutral language — no judgment.
        """
        if delta > 0:
            return "Relative NCI increased due to global vocabulary composition shift"
        elif delta < 0:
            return "Relative NCI decreased due to global vocabulary composition shift"
        else:
            return "Measurement within expected variance"


# ==============================================================================
# HOT SPOTS ANALYZER
# ==============================================================================


class HotSpotsAnalyzer:
    """Identifies narrative consistency hot spots."""
    
    def __init__(self, index: NarrativeIndex, top_n: int = 20):
        self.index = index
        self.top_n = top_n
    
    def compute_hotspots(self) -> list[HotSpot]:
        """Compute hot spots ranked by inconsistency contribution."""
        hotspots = []
        
        # Compute global deficit (how much total NCI is below 1.0)
        total_deficit = sum(1.0 - d.nci_score for d in self.index.documents)
        
        for doc in self.index.documents:
            nci_deficit = 1.0 - doc.nci_score
            
            if nci_deficit < 0.01:
                continue  # Skip nearly perfect docs
            
            # Compute contribution percentage
            contribution_pct = (nci_deficit / total_deficit * 100) if total_deficit > 0 else 0
            
            # Count severity per dimension
            severity_counts = {
                "terminology": len(doc.terminology.violations),
                "phase": len(doc.phase.violations),
                "uplift": len(doc.uplift.violations),
                "structure": 0,  # Structure doesn't have violations, compute differently
            }
            
            # Structure severity based on missing markers
            if not doc.structure.has_safeguard_banner:
                severity_counts["structure"] += 1
            if not doc.structure.has_cross_references:
                severity_counts["structure"] += 1
            if not doc.structure.has_version_marker:
                severity_counts["structure"] += 1
            if not doc.structure.has_status_marker:
                severity_counts["structure"] += 1
            
            # Determine primary issue (dimension with lowest ratio)
            dimension_scores = {
                "terminology": doc.terminology.alignment_ratio,
                "phase": doc.phase.discipline_ratio,
                "uplift": doc.uplift.avoidance_ratio,
                "structure": doc.structure.coherence_ratio,
            }
            primary_issue = min(dimension_scores, key=dimension_scores.get)
            
            hotspots.append(HotSpot(
                file=doc.path,
                category=doc.category,
                nci_score=doc.nci_score,
                nci_deficit=nci_deficit,
                contribution_pct=contribution_pct,
                severity_counts=severity_counts,
                primary_issue=primary_issue,
            ))
        
        # Sort by contribution (highest first)
        hotspots.sort(key=lambda h: h.contribution_pct, reverse=True)
        
        return hotspots[:self.top_n]
    
    def to_json(self) -> dict:
        """Generate JSON output for hot spots."""
        hotspots = self.compute_hotspots()
        
        return {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "commit_hash": self.index.commit_hash,
            "global_nci": round(self.index.global_nci, 4),
            "total_hotspots": len(hotspots),
            "hotspots": [
                {
                    "rank": i + 1,
                    "file": h.file,
                    "category": h.category,
                    "nci_score": round(h.nci_score, 4),
                    "nci_deficit": round(h.nci_deficit, 4),
                    "contribution_pct": round(h.contribution_pct, 2),
                    "severity_counts": h.severity_counts,
                    "primary_issue": h.primary_issue,
                }
                for i, h in enumerate(hotspots)
            ],
            "summary": {
                "by_category": self._summarize_by_category(hotspots),
                "by_primary_issue": self._summarize_by_issue(hotspots),
            },
        }
    
    def _summarize_by_category(self, hotspots: list[HotSpot]) -> dict:
        """Summarize hot spots by category."""
        summary = defaultdict(lambda: {"count": 0, "total_contribution": 0.0})
        for h in hotspots:
            summary[h.category]["count"] += 1
            summary[h.category]["total_contribution"] += h.contribution_pct
        return {k: {"count": v["count"], "total_contribution": round(v["total_contribution"], 2)} 
                for k, v in summary.items()}
    
    def _summarize_by_issue(self, hotspots: list[HotSpot]) -> dict:
        """Summarize hot spots by primary issue."""
        summary = defaultdict(int)
        for h in hotspots:
            summary[h.primary_issue] += 1
        return dict(summary)


# ==============================================================================
# TEN-MINUTE FIX GENERATOR
# ==============================================================================


def suggest_ten_minute_fixes(
    index: NarrativeIndex,
    hotspots: list[HotSpot],
    max_suggestions: int = 10
) -> list[dict[str, Any]]:
    """
    Returns a small set of concrete, low-effort doc fixes.
    
    Heuristics:
    - Prefer docs with high variance but small word count
    - Files whose issues are mostly terminology/phase consistency
    - Each suggestion includes: file, issue_type, hint, estimated_effort
    
    No edits — suggestions only.
    """
    suggestions = []
    
    # Build lookup for document metrics
    doc_lookup = {d.path: d for d in index.documents}
    
    for hotspot in hotspots:
        doc = doc_lookup.get(hotspot.file)
        if not doc:
            continue
        
        # Calculate word count (proxy from line count * avg words per line)
        word_count_estimate = doc.line_count * 8  # Rough estimate
        
        # Calculate priority score
        # Higher score = more impactful AND easier to fix
        # Prefer: small files, terminology/phase issues (easy to fix), high deficit
        
        # Base score from contribution
        priority = hotspot.contribution_pct
        
        # Boost for smaller files (easier to review)
        if word_count_estimate < 500:
            priority *= 2.0
        elif word_count_estimate < 1000:
            priority *= 1.5
        elif word_count_estimate > 3000:
            priority *= 0.5
        
        # Boost for terminology/phase issues (straightforward fixes)
        easy_issues = {"terminology", "phase"}
        if hotspot.primary_issue in easy_issues:
            priority *= 1.5
        
        # Penalize uplift issues (require careful review, not quick fixes)
        if hotspot.primary_issue == "uplift":
            priority *= 0.3
        
        # Generate specific hint based on issue type
        hint = _generate_fix_hint(hotspot, doc)
        
        suggestions.append({
            "file": hotspot.file,
            "issue_type": hotspot.primary_issue,
            "hint": hint,
            "estimated_effort": "<10m",
            "word_count": word_count_estimate,
            "violation_count": sum(hotspot.severity_counts.values()),
            "priority_score": round(priority, 2),
            "nci_score": round(hotspot.nci_score, 2),
        })
    
    # Sort by priority (highest first)
    suggestions.sort(key=lambda s: s["priority_score"], reverse=True)
    
    return suggestions[:max_suggestions]


def _generate_fix_hint(hotspot: HotSpot, doc: DocumentMetrics) -> str:
    """Generate a concrete, actionable hint for fixing the issue."""
    issue = hotspot.primary_issue
    severity = hotspot.severity_counts
    
    if issue == "terminology":
        if severity["terminology"] == 1:
            return "Single terminology variant found. Search for non-canonical term and replace."
        elif severity["terminology"] <= 5:
            return f"Found {severity['terminology']} terminology variants. Quick find-replace should resolve."
        else:
            return f"Multiple terminology variants ({severity['terminology']}). Consider batch replacement."
    
    elif issue == "phase":
        if severity["phase"] <= 3:
            return f"Found {severity['phase']} Phase notation issue(s). Replace 'Phase 2' with 'Phase II'."
        else:
            return f"Found {severity['phase']} Phase notation issues. Global search for 'Phase [0-9]' recommended."
    
    elif issue == "structure":
        hints = []
        if not doc.structure.has_safeguard_banner:
            hints.append("Add ABSOLUTE SAFEGUARDS banner")
        if not doc.structure.has_cross_references:
            hints.append("Add cross-references to related docs")
        if not doc.structure.has_version_marker:
            hints.append("Add Version: marker")
        if not doc.structure.has_status_marker:
            hints.append("Add Status: marker")
        return "; ".join(hints[:2]) if hints else "Improve document structure markers"
    
    elif issue == "uplift":
        return f"Review {severity['uplift']} potential uplift claim(s). Requires careful context review."
    
    return "Review document for narrative consistency"


class TenMinuteFixGenerator:
    """Generates actionable, low-effort fix suggestions."""
    
    def __init__(self, index: NarrativeIndex, max_suggestions: int = 10):
        self.index = index
        self.max_suggestions = max_suggestions
    
    def generate(self) -> list[dict[str, Any]]:
        """Generate ten-minute fix suggestions."""
        # First compute hotspots
        analyzer = HotSpotsAnalyzer(self.index, top_n=50)  # Get more to filter
        hotspots = analyzer.compute_hotspots()
        
        return suggest_ten_minute_fixes(self.index, hotspots, self.max_suggestions)
    
    def to_json(self) -> dict:
        """Generate JSON output for ten-minute fixes.
        
        Stable API structure:
        {
            "generated_at": "ISO timestamp",
            "global_nci": 0.77,
            "suggestions": [...]
        }
        """
        suggestions = self.generate()
        
        # Normalize priority_score to 0-1 range for API stability
        if suggestions:
            max_priority = max(s["priority_score"] for s in suggestions)
            if max_priority > 0:
                for s in suggestions:
                    s["priority_score"] = round(s["priority_score"] / max_priority, 2)
        
        # Clean suggestions to match API spec (remove internal fields)
        clean_suggestions = []
        for s in suggestions:
            clean_suggestions.append({
                "file": s["file"],
                "issue_type": s["issue_type"],
                "hint": s["hint"],
                "estimated_effort": s["estimated_effort"],
                "priority_score": s["priority_score"],
                "violation_count": s["violation_count"],
            })
        
        return {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "global_nci": round(self.index.global_nci, 4),
            "suggestions": clean_suggestions,
        }


# ==============================================================================
# NCI INSIGHT GRID — Area-Based Variance View
# ==============================================================================

# Schema version for insight grid outputs
NCI_INSIGHT_SCHEMA_VERSION = "1.0.0"

# Conceptual areas for NCI grouping
NCI_AREAS = {
    "terminology": {
        "dimensions": ["terminology"],
        "description": "Term consistency across documents",
    },
    "phase_notation": {
        "dimensions": ["phase"],
        "description": "Phase numbering conventions",
    },
    "uplift_discipline": {
        "dimensions": ["uplift"],
        "description": "Uplift claim avoidance",
    },
    "structural": {
        "dimensions": ["structure"],
        "description": "Document structure markers",
    },
}


def build_nci_area_view(index: NarrativeIndex) -> dict[str, Any]:
    """
    Build an area-based NCI variance view.
    
    Groups NCI observations into conceptual areas (terminology, phase, etc.)
    and computes per-area statistics.
    
    Args:
        index: NarrativeIndex with computed document metrics
        
    Returns:
        Dict with area statistics:
        {
            "schema_version": "1.0.0",
            "global_nci": 0.75,
            "areas": {
                "terminology": {"nci": 0.73, "variance": 0.02, "doc_count": 7},
                ...
            }
        }
    """
    # Collect per-area scores from each document
    area_scores: dict[str, list[float]] = {area: [] for area in NCI_AREAS}
    
    for doc in index.documents:
        # Map each dimension to its area
        if doc.terminology:
            area_scores["terminology"].append(doc.terminology.alignment_ratio)
        if doc.phase:
            area_scores["phase_notation"].append(doc.phase.discipline_ratio)
        if doc.uplift:
            area_scores["uplift_discipline"].append(doc.uplift.avoidance_ratio)
        if doc.structure:
            area_scores["structural"].append(doc.structure.coherence_ratio)
    
    # Compute statistics per area
    areas = {}
    for area_name in sorted(NCI_AREAS.keys()):  # Sorted for determinism
        scores = area_scores[area_name]
        if scores:
            mean_nci = sum(scores) / len(scores)
            # Variance = avg of squared differences from mean
            variance = sum((s - mean_nci) ** 2 for s in scores) / len(scores)
            areas[area_name] = {
                "nci": round(mean_nci, 4),
                "variance": round(variance, 4),
                "doc_count": len(scores),
            }
        else:
            areas[area_name] = {
                "nci": 1.0,
                "variance": 0.0,
                "doc_count": 0,
            }
    
    return {
        "schema_version": NCI_INSIGHT_SCHEMA_VERSION,
        "global_nci": round(index.global_nci, 4),
        "areas": areas,
    }


# ==============================================================================
# NCI INSIGHT GRID — Time-Slice Snapshot Comparison
# ==============================================================================


@dataclass
class NCISnapshot:
    """A point-in-time NCI snapshot for comparison."""
    timestamp: str
    global_nci: float
    area_nci: dict[str, float]  # area_name -> nci score
    doc_count: int


def create_nci_snapshot(index: NarrativeIndex) -> NCISnapshot:
    """Create a snapshot from an NCI index for later comparison."""
    area_view = build_nci_area_view(index)
    
    area_nci = {
        area_name: stats["nci"]
        for area_name, stats in area_view["areas"].items()
    }
    
    return NCISnapshot(
        timestamp=datetime.utcnow().isoformat() + "Z",
        global_nci=round(index.global_nci, 4),
        area_nci=area_nci,
        doc_count=len(index.documents),
    )


def compare_nci_snapshots(
    old_snapshot: NCISnapshot | dict,
    new_snapshot: NCISnapshot | dict,
) -> dict[str, Any]:
    """
    Compare two NCI snapshots and compute deltas.
    
    Uses neutral language — no "better/worse", just numeric deltas.
    
    Args:
        old_snapshot: Previous NCI snapshot (NCISnapshot or dict)
        new_snapshot: Current NCI snapshot (NCISnapshot or dict)
        
    Returns:
        {
            "schema_version": "1.0.0",
            "old_timestamp": "...",
            "new_timestamp": "...",
            "global_nci_delta": float,
            "area_deltas": {
                "terminology": float,
                "phase_notation": float,
                ...
            }
        }
    """
    # Handle both NCISnapshot and dict inputs
    if isinstance(old_snapshot, dict):
        old_ts = old_snapshot.get("timestamp", "unknown")
        old_global = old_snapshot.get("global_nci", 0)
        old_areas = old_snapshot.get("area_nci", {})
    else:
        old_ts = old_snapshot.timestamp
        old_global = old_snapshot.global_nci
        old_areas = old_snapshot.area_nci
    
    if isinstance(new_snapshot, dict):
        new_ts = new_snapshot.get("timestamp", "unknown")
        new_global = new_snapshot.get("global_nci", 0)
        new_areas = new_snapshot.get("area_nci", {})
    else:
        new_ts = new_snapshot.timestamp
        new_global = new_snapshot.global_nci
        new_areas = new_snapshot.area_nci
    
    # Compute global delta
    global_delta = new_global - old_global
    
    # Compute per-area deltas
    all_areas = sorted(set(old_areas.keys()) | set(new_areas.keys()))
    area_deltas = {}
    for area in all_areas:
        old_val = old_areas.get(area, 0.0)
        new_val = new_areas.get(area, 0.0)
        area_deltas[area] = round(new_val - old_val, 4)
    
    return {
        "schema_version": NCI_INSIGHT_SCHEMA_VERSION,
        "old_timestamp": old_ts,
        "new_timestamp": new_ts,
        "global_nci_delta": round(global_delta, 4),
        "area_deltas": area_deltas,
    }


# ==============================================================================
# NCI INSIGHT GRID — Dashboard Summary JSON
# ==============================================================================


def build_nci_insight_summary(
    quick_fixes: dict[str, Any],
    area_view: dict[str, Any],
) -> dict[str, Any]:
    """
    Build a consolidated NCI insight summary for dashboards.
    
    Args:
        quick_fixes: Output from TenMinuteFixGenerator.to_json()
        area_view: Output from build_nci_area_view()
        
    Returns:
        {
            "schema_version": "1.0.0",
            "global_nci": 0.75,
            "top_files": ["docs/...", ...],
            "dominant_area": "terminology",
            "suggestion_count": 10
        }
    """
    # Extract global NCI (prefer from quick_fixes for consistency)
    global_nci = quick_fixes.get("global_nci", area_view.get("global_nci", 0))
    
    # Extract top files from suggestions
    suggestions = quick_fixes.get("suggestions", [])
    top_files = [s["file"] for s in suggestions[:5]]  # Top 5
    
    # Find dominant area (lowest NCI = most attention needed)
    areas = area_view.get("areas", {})
    dominant_area = None
    lowest_nci = 1.0
    for area_name in sorted(areas.keys()):  # Sorted for determinism
        area_nci = areas[area_name].get("nci", 1.0)
        doc_count = areas[area_name].get("doc_count", 0)
        if doc_count > 0 and area_nci < lowest_nci:
            lowest_nci = area_nci
            dominant_area = area_name
    
    return {
        "schema_version": NCI_INSIGHT_SCHEMA_VERSION,
        "global_nci": round(global_nci, 4),
        "top_files": top_files,
        "dominant_area": dominant_area or "none",
        "suggestion_count": len(suggestions),
    }


# ==============================================================================
# NCI ALERTING & HEALTH DASHBOARD — Phase III
# ==============================================================================


def evaluate_nci_slo(
    area_view: dict[str, Any],
    thresholds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Evaluate NCI against Service Level Objectives (SLOs).
    
    Uses neutral language — no judgment, just status indicators.
    
    Args:
        area_view: Output from build_nci_area_view()
        thresholds: Optional dict with:
            - max_global_nci (default: 0.75) — threshold for WARN
            - max_area_nci (default: 0.70) — per-area threshold for WARN
            - min_structural_nci (default: 0.60) — minimum structural NCI
    
    Returns:
        {
            "slo_status": "OK" | "WARN" | "BREACH",
            "violations": [list of neutral violation descriptions],
            "global_nci": float,
            "thresholds_used": {...}
        }
    """
    # Default thresholds (neutral, non-prescriptive)
    default_thresholds = {
        "max_global_nci": 0.75,
        "max_area_nci": 0.70,
        "min_structural_nci": 0.60,
    }
    
    if thresholds is None:
        thresholds = default_thresholds
    else:
        thresholds = {**default_thresholds, **thresholds}
    
    global_nci = area_view.get("global_nci", 1.0)
    areas = area_view.get("areas", {})
    
    violations = []
    status = "OK"
    
    # Check global NCI
    if global_nci < thresholds["max_global_nci"]:
        violations.append(
            f"Global NCI ({global_nci:.4f}) below threshold ({thresholds['max_global_nci']:.4f})"
        )
        status = "WARN"
    
    # Check per-area NCI
    for area_name in sorted(areas.keys()):
        area_data = areas[area_name]
        area_nci = area_data.get("nci", 1.0)
        doc_count = area_data.get("doc_count", 0)
        
        if doc_count > 0 and area_nci < thresholds["max_area_nci"]:
            violations.append(
                f"Area '{area_name}' NCI ({area_nci:.4f}) below threshold ({thresholds['max_area_nci']:.4f})"
            )
            if status == "OK":
                status = "WARN"
    
    # Check structural minimum (if specified)
    if "min_structural_nci" in thresholds:
        structural_nci = areas.get("structural", {}).get("nci", 1.0)
        if structural_nci < thresholds["min_structural_nci"]:
            violations.append(
                f"Structural NCI ({structural_nci:.4f}) below minimum ({thresholds['min_structural_nci']:.4f})"
            )
            status = "BREACH"  # Structural issues are more severe
    
    # Upgrade to BREACH if multiple violations
    if len(violations) >= 3:
        status = "BREACH"
    
    return {
        "slo_status": status,
        "violations": violations,
        "global_nci": round(global_nci, 4),
        "thresholds_used": thresholds,
    }


def build_nci_alerts(
    insight_summary: dict[str, Any],
    slo_result: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Build non-prescriptive alert suggestions based on NCI insights and SLO evaluation.
    
    Language: "may warrant review", "high NCI variance", etc. — no "fix/broken".
    
    Args:
        insight_summary: Output from build_nci_insight_summary()
        slo_result: Output from evaluate_nci_slo()
    
    Returns:
        List of alert dicts:
        [
            {
                "area": "terminology",
                "nci": 0.73,
                "reason": "NCI in this area may warrant review",
                "top_file": "docs/X.md"  # optional
            }
        ]
    """
    alerts = []
    
    # Check SLO violations
    if slo_result["slo_status"] != "OK":
        violations = slo_result.get("violations", [])
        for violation in violations:
            # Extract area from violation message if present
            area = None
            if "area" in violation.lower() or "structural" in violation.lower():
                # Try to extract area name
                for area_name in ["structural", "terminology", "phase_notation", "uplift_discipline"]:
                    if area_name in violation.lower():
                        area = area_name
                        break
            
            alerts.append({
                "area": area or "global",
                "nci": slo_result.get("global_nci", 0),
                "reason": violation,
            })
    
    # Check dominant area if it has high variance
    dominant_area = insight_summary.get("dominant_area", "none")
    if dominant_area != "none":
        top_files = insight_summary.get("top_files", [])
        alerts.append({
            "area": dominant_area,
            "nci": None,  # Will be filled from area_view if needed
            "reason": f"High NCI variance in '{dominant_area}' area may warrant review",
            "top_file": top_files[0] if top_files else None,
        })
    
    # Check suggestion count
    suggestion_count = insight_summary.get("suggestion_count", 0)
    if suggestion_count > 10:
        alerts.append({
            "area": "global",
            "nci": insight_summary.get("global_nci", 0),
            "reason": f"Many suggestion opportunities ({suggestion_count}) available for review",
        })
    
    return alerts


def summarize_nci_for_global_health(
    insight_summary: dict[str, Any],
    slo_result: dict[str, Any],
) -> dict[str, Any]:
    """
    Build a compact global health NCI signal for dashboards and MAAS.
    
    Uses simple status labels: OK | WARN | HOT
    
    Args:
        insight_summary: Output from build_nci_insight_summary()
        slo_result: Output from evaluate_nci_slo()
    
    Returns:
        {
            "nci_status": "OK" | "WARN" | "HOT",
            "global_nci": 0.75,
            "dominant_area": "terminology",
            "suggestion_count": 10
        }
    """
    slo_status = slo_result.get("slo_status", "OK")
    
    # Map SLO status to health status
    if slo_status == "BREACH":
        nci_status = "HOT"
    elif slo_status == "WARN":
        nci_status = "WARN"
    else:
        nci_status = "OK"
    
    # Upgrade to HOT if suggestion count is very high
    suggestion_count = insight_summary.get("suggestion_count", 0)
    if suggestion_count > 20:
        nci_status = "HOT"
    
    return {
        "nci_status": nci_status,
        "global_nci": insight_summary.get("global_nci", 1.0),
        "dominant_area": insight_summary.get("dominant_area", "none"),
        "suggestion_count": suggestion_count,
    }


# ==============================================================================
# NCI AS NARRATIVE HEALTH SIGNAL & ALERTING CONTRACT — Phase IV
# ==============================================================================


def build_nci_work_priority_view(
    insight_summary: dict[str, Any],
    slo_result: dict[str, Any],
    area_view: dict[str, Any] | None = None,
    quick_fixes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build a work priority view for narrative consistency improvement.
    
    Prioritizes areas by NCI variance or SLO BREACH status.
    Maps files to areas for focused work.
    
    Args:
        insight_summary: Output from build_nci_insight_summary()
        slo_result: Output from evaluate_nci_slo()
        area_view: Optional output from build_nci_area_view() for variance data
        quick_fixes: Optional output from TenMinuteFixGenerator.to_json() for file mapping
    
    Returns:
        {
            "priority_areas": ["area1", "area2", ...],  # Ordered by priority
            "files_per_area": {
                "area1": ["file1.md", "file2.md"],
                ...
            },
            "status": "OK" | "ATTENTION" | "BREACH"
        }
    """
    # Map issue_type to area names
    issue_to_area = {
        "terminology": "terminology",
        "phase": "phase_notation",
        "structure": "structural",
        "uplift": "uplift_discipline",
    }
    
    # Determine status from SLO
    slo_status = slo_result.get("slo_status", "OK")
    if slo_status == "BREACH":
        status = "BREACH"
    elif slo_status == "WARN":
        status = "ATTENTION"
    else:
        status = "OK"
    
    # Collect area priorities (highest variance or BREACH first)
    area_priorities = []
    files_per_area = defaultdict(list)
    
    # Get files from quick_fixes if available
    if quick_fixes:
        suggestions = quick_fixes.get("suggestions", [])
        for suggestion in suggestions:
            issue_type = suggestion.get("issue_type", "")
            file_path = suggestion.get("file", "")
            area = issue_to_area.get(issue_type)
            
            if area and file_path:
                files_per_area[area].append(file_path)
    
    # Also use top_files from insight_summary
    top_files = insight_summary.get("top_files", [])
    dominant_area = insight_summary.get("dominant_area", "none")
    if dominant_area != "none" and top_files:
        # Add top files to dominant area
        files_per_area[dominant_area].extend(top_files[:3])
    
    # Build priority list based on area_view variance or SLO violations
    if area_view:
        areas = area_view.get("areas", {})
        area_scores = []
        
        for area_name in sorted(areas.keys()):  # Deterministic ordering
            area_data = areas[area_name]
            area_nci = area_data.get("nci", 1.0)
            variance = area_data.get("variance", 0.0)
            doc_count = area_data.get("doc_count", 0)
            
            if doc_count > 0:
                # Priority score: lower NCI and higher variance = higher priority
                priority_score = variance - area_nci  # Higher score = more priority
                area_scores.append((area_name, priority_score, area_nci, variance))
        
        # Sort by priority score (highest first)
        area_scores.sort(key=lambda x: x[1], reverse=True)
        priority_areas = [area for area, _, _, _ in area_scores]
        
        # If SLO BREACH, prioritize areas mentioned in violations
        if status == "BREACH":
            violations = slo_result.get("violations", [])
            breach_areas = []
            for violation in violations:
                for area_name in priority_areas:
                    if area_name in violation.lower():
                        breach_areas.append(area_name)
            
            # Move breach areas to front
            for area in breach_areas:
                if area in priority_areas:
                    priority_areas.remove(area)
                    priority_areas.insert(0, area)
    else:
        # Fallback: use dominant area if available
        if dominant_area != "none":
            priority_areas = [dominant_area]
        else:
            priority_areas = []
    
    # Deduplicate files per area
    for area in files_per_area:
        files_per_area[area] = list(dict.fromkeys(files_per_area[area]))[:5]  # Top 5 per area
    
    return {
        "priority_areas": priority_areas,
        "files_per_area": dict(files_per_area),
        "status": status,
    }


def build_nci_contract_for_doc_tools(
    priority_view: dict[str, Any],
) -> dict[str, Any]:
    """
    Build a contract for doc-weaver tools indicating which files/areas to focus on.
    
    This contract guides documentation automation tools on work prioritization.
    
    Args:
        priority_view: Output from build_nci_work_priority_view()
    
    Returns:
        {
            "contract_version": "1.0.0",
            "areas_to_focus": ["area1", "area2", ...],
            "max_files_per_area": 5,
            "selection_rule": "description of selection logic"
        }
    """
    priority_areas = priority_view.get("priority_areas", [])
    files_per_area = priority_view.get("files_per_area", {})
    status = priority_view.get("status", "OK")
    
    # Determine max files per area based on status
    if status == "BREACH":
        max_files = 10  # More files for critical status
    elif status == "ATTENTION":
        max_files = 5
    else:
        max_files = 3
    
    # Build selection rule description
    if status == "BREACH":
        rule = "Areas with SLO BREACH prioritized first, then by highest variance (lowest NCI × highest variance)"
    elif status == "ATTENTION":
        rule = "Areas with highest variance prioritized (lowest NCI × highest variance)"
    else:
        rule = "Areas with highest variance prioritized (lowest NCI × highest variance)"
    
    # Filter areas that actually have files
    areas_with_files = [
        area for area in priority_areas
        if area in files_per_area and len(files_per_area[area]) > 0
    ]
    
    return {
        "contract_version": "1.0.0",
        "areas_to_focus": areas_with_files,
        "max_files_per_area": max_files,
        "selection_rule": rule,
        "files_per_area": {
            area: files_per_area[area][:max_files]
            for area in areas_with_files
            if area in files_per_area
        },
    }


def build_nci_director_panel(
    insight_summary: dict[str, Any],
    priority_view: dict[str, Any],
    slo_result: dict[str, Any],
) -> dict[str, Any]:
    """
    Build a high-level narrative panel for the Director dashboard.
    
    Provides status light, key metrics, and a neutral headline about
    narrative consistency posture.
    
    Args:
        insight_summary: Output from build_nci_insight_summary()
        priority_view: Output from build_nci_work_priority_view()
        slo_result: Output from evaluate_nci_slo()
    
    Returns:
        {
            "status_light": "🟢" | "🟡" | "🔴",
            "global_nci": 0.75,
            "dominant_area": "terminology",
            "headline": "neutral sentence about narrative consistency posture"
        }
    """
    global_nci = insight_summary.get("global_nci", 1.0)
    dominant_area = insight_summary.get("dominant_area", "none")
    status = priority_view.get("status", "OK")
    suggestion_count = insight_summary.get("suggestion_count", 0)
    priority_areas = priority_view.get("priority_areas", [])
    
    # Determine status light
    if status == "BREACH":
        status_light = "🔴"
    elif status == "ATTENTION":
        status_light = "🟡"
    else:
        status_light = "🟢"
    
    # Build neutral headline
    if status == "BREACH":
        headline = f"Narrative consistency SLO breach detected. {len(priority_areas)} area(s) require attention."
    elif status == "ATTENTION":
        if dominant_area != "none":
            headline = f"Narrative consistency requires attention. Primary focus area: {dominant_area}."
        else:
            headline = f"Narrative consistency requires attention. {suggestion_count} improvement opportunity(ies) identified."
    else:
        headline = f"Narrative consistency within target. Global NCI: {global_nci:.2f}."
    
    return {
        "status_light": status_light,
        "global_nci": round(global_nci, 4),
        "dominant_area": dominant_area,
        "headline": headline,
    }


# ==============================================================================
# CROSS-RUN NARRATIVE STABILITY & DOC-WEAVER FEEDBACK LOOP
# ==============================================================================


def build_nci_stability_timeline(
    snapshots: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """
    Build a stability timeline from a sequence of NCI snapshots.
    
    Tracks narrative consistency over time and identifies trends.
    Uses neutral language — no judgment, just descriptive trend labels.
    
    Args:
        snapshots: Sequence of snapshot dicts, each should have:
            - "run_id" or "timestamp" (used as identifier)
            - "global_nci" (float)
            - "dominant_area" (str, optional)
            - Or can be output from create_nci_snapshot() or insight_summary
    
    Returns:
        {
            "schema_version": "1.0.0",
            "timeline": [
                {"run_id": "run_1", "global_nci": 0.75, "dominant_area": "terminology"},
                ...
            ],
            "trend": "IMPROVING" | "STABLE" | "DEGRADING",
            "neutral_notes": ["note 1", "note 2", ...]
        }
    """
    if not snapshots:
        return {
            "schema_version": NCI_INSIGHT_SCHEMA_VERSION,
            "timeline": [],
            "trend": "STABLE",
            "neutral_notes": ["No snapshot data available"],
        }
    
    # Build timeline entries
    timeline = []
    for i, snapshot in enumerate(snapshots):
        # Extract run_id (prefer run_id, fallback to timestamp or index)
        run_id = snapshot.get("run_id") or snapshot.get("timestamp") or snapshot.get("generated_at") or f"run_{i+1}"
        
        # Extract global_nci
        global_nci = snapshot.get("global_nci", snapshot.get("head_nci", 0.0))
        
        # Extract dominant_area
        dominant_area = snapshot.get("dominant_area") or snapshot.get("primary_issue") or "none"
        
        timeline.append({
            "run_id": str(run_id),
            "global_nci": round(float(global_nci), 4),
            "dominant_area": str(dominant_area),
        })
    
    # Determine trend based on global_nci slope
    if len(timeline) < 2:
        trend = "STABLE"
        neutral_notes = ["Insufficient data points for trend analysis"]
    else:
        ncis = [entry["global_nci"] for entry in timeline]
        
        # Calculate linear trend (simple slope)
        # Use first and last points for simplicity
        first_nci = ncis[0]
        last_nci = ncis[-1]
        
        # Threshold for considering a change significant
        change_threshold = 0.01  # 1% change threshold
        
        nci_change = last_nci - first_nci
        
        if abs(nci_change) < change_threshold:
            trend = "STABLE"
        elif nci_change > 0:
            trend = "IMPROVING"
        else:
            trend = "DEGRADING"
        
        # Build neutral notes
        neutral_notes = []
        if len(timeline) >= 2:
            neutral_notes.append(f"NCI changed from {first_nci:.4f} to {last_nci:.4f} over {len(timeline)} runs")
        
        if trend == "IMPROVING":
            neutral_notes.append("Global NCI increasing over time")
        elif trend == "DEGRADING":
            neutral_notes.append("Global NCI decreasing over time")
        else:
            neutral_notes.append("Global NCI relatively stable")
        
        # Note dominant area shifts
        dominant_areas = [entry["dominant_area"] for entry in timeline if entry["dominant_area"] != "none"]
        if dominant_areas:
            unique_areas = set(dominant_areas)
            if len(unique_areas) > 1:
                neutral_notes.append(f"Dominant area shifts observed: {', '.join(sorted(unique_areas))}")
    
    return {
        "schema_version": NCI_INSIGHT_SCHEMA_VERSION,
        "timeline": timeline,
        "trend": trend,
        "neutral_notes": neutral_notes,
    }


def build_nci_contract_for_doc_tools_v2(
    priority_view: dict[str, Any],
    stability_timeline: dict[str, Any],
) -> dict[str, Any]:
    """
    Build an enhanced doc-weaver contract (v2) with trend information and workflow suggestions.
    
    Extends v1 contract with stability timeline analysis to suggest appropriate workflows.
    
    Args:
        priority_view: Output from build_nci_work_priority_view()
        stability_timeline: Output from build_nci_stability_timeline()
    
    Returns:
        {
            "contract_version": "2.0.0",
            "areas_to_focus": [...],
            "max_files_per_area": int,
            "selection_rule": "...",
            "files_per_area": {...},
            "trend": "IMPROVING" | "STABLE" | "DEGRADING",
            "suggested_workflow": "stabilize_first" | "expand_coverage" | "maintenance"
        }
    """
    # Get base contract from v1
    base_contract = build_nci_contract_for_doc_tools(priority_view)
    
    # Extract trend and status
    trend = stability_timeline.get("trend", "STABLE")
    status = priority_view.get("status", "OK")
    
    # Determine suggested workflow based on trend and status
    # DEGRADING + BREACH → stabilize_first
    # IMPROVING + OK → expand_coverage
    # STABLE → maintenance
    # Otherwise: balance based on status
    
    if trend == "DEGRADING" and status == "BREACH":
        suggested_workflow = "stabilize_first"
    elif trend == "DEGRADING":
        suggested_workflow = "stabilize_first"
    elif trend == "IMPROVING" and status == "OK":
        suggested_workflow = "expand_coverage"
    elif trend == "IMPROVING":
        suggested_workflow = "maintenance"  # Continue improving but maintain
    elif trend == "STABLE":
        suggested_workflow = "maintenance"
    else:
        # Fallback: use status to determine
        if status == "BREACH":
            suggested_workflow = "stabilize_first"
        elif status == "ATTENTION":
            suggested_workflow = "maintenance"
        else:
            suggested_workflow = "expand_coverage"
    
    # Merge base contract with v2 additions
    return {
        **base_contract,
        "contract_version": "2.0.0",
        "trend": trend,
        "suggested_workflow": suggested_workflow,
    }


# ==============================================================================
# HEATMAP GENERATOR
# ==============================================================================


def generate_heatmap(index: NarrativeIndex, output_path: Path) -> bool:
    """Generate a deterministic consistency heatmap."""
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not available, skipping heatmap generation")
        return False
    
    # Prepare data
    categories = list(DOCUMENT_CATEGORIES.keys()) + ["other"]
    metrics = ["Terminology", "Phase", "Uplift", "Structure", "NCI"]
    
    # Build matrix
    data = []
    for category in categories:
        cat_docs = [d for d in index.documents if d.category == category]
        if not cat_docs:
            data.append([1.0, 1.0, 1.0, 1.0, 1.0])
            continue
        
        row = [
            sum(d.terminology.alignment_ratio for d in cat_docs) / len(cat_docs),
            sum(d.phase.discipline_ratio for d in cat_docs) / len(cat_docs),
            sum(d.uplift.avoidance_ratio for d in cat_docs) / len(cat_docs),
            sum(d.structure.coherence_ratio for d in cat_docs) / len(cat_docs),
            sum(d.nci_score for d in cat_docs) / len(cat_docs),
        ]
        data.append(row)
    
    # Create figure with deterministic settings
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 100
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create heatmap
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "nci", ["#ff4444", "#ffaa00", "#44aa44"], N=256
    )
    
    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=1, aspect='auto')
    
    # Labels
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels([c.title() for c in categories], fontsize=10)
    
    # Add values
    for i in range(len(categories)):
        for j in range(len(metrics)):
            value = data[i][j]
            color = "white" if value < 0.5 else "black"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=color, fontsize=9)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Consistency Score", fontsize=10)
    
    # Title
    ax.set_title(
        f"Narrative Consistency Heatmap — Global NCI: {index.global_nci:.3f}",
        fontsize=12,
        fontweight="bold"
    )
    
    plt.tight_layout()
    plt.savefig(output_path, format='png', bbox_inches='tight')
    plt.close()
    
    return True


# ==============================================================================
# GLOBAL HEALTH ENVELOPE v4 — System Health Synthesizer
# ==============================================================================


def _extract_component_band(envelope: dict[str, Any]) -> str:
    """
    Extract status band from various envelope formats.
    
    Supports multiple formats:
    - Direct band field: {"band": "GREEN"}
    - Status field: {"status": "OK"} -> GREEN
    - Risk band: {"risk_band": "LOW"} -> GREEN
    - NCI status: {"nci_status": "OK"} -> GREEN
    
    Returns: "GREEN" | "YELLOW" | "RED" | "UNKNOWN"
    """
    if not envelope:
        return "UNKNOWN"
    
    # Try direct band field
    if "band" in envelope:
        band = envelope["band"].upper()
        if band in ["GREEN", "YELLOW", "RED"]:
            return band
    
    # Try status fields (map to bands)
    status_mapping = {
        "OK": "GREEN",
        "WARN": "YELLOW",
        "ATTENTION": "YELLOW",
        "HOT": "RED",
        "BREACH": "RED",
    }
    
    for status_field in ["status", "nci_status", "governance_drift_status"]:
        if status_field in envelope:
            status = envelope[status_field].upper()
            if status in status_mapping:
                return status_mapping[status]
    
    # Try risk band
    if "risk_band" in envelope:
        risk = envelope["risk_band"].upper()
        if risk == "LOW":
            return "GREEN"
        elif risk == "MEDIUM":
            return "YELLOW"
        elif risk == "HIGH":
            return "RED"
    
    return "UNKNOWN"


def build_global_health_envelope_v4(
    metric_health: dict[str, Any] | None = None,
    drift_envelope: dict[str, Any] | None = None,
    semantic_envelope: dict[str, Any] | None = None,
    atlas_envelope: dict[str, Any] | None = None,
    telemetry_envelope: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build Global Health Envelope v4 — synthesizes all system health signals.
    
    Combines signals from multiple pillars (E1-E5) into a planetary-scale
    health assessment.
    
    Args:
        metric_health: Health signals from metrics subsystem
        drift_envelope: Governance drift signals (E1)
        semantic_envelope: Semantic consistency signals
        atlas_envelope: Structural/topology signals
        telemetry_envelope: NCI telemetry signals (E5)
    
    Returns:
        {
            "schema_version": "4.0.0",
            "global_band": "GREEN" | "YELLOW" | "RED",
            "envelope_components": {
                "metric_health": {"band": "GREEN", "present": true},
                "drift_envelope": {"band": "YELLOW", "present": true},
                ...
            },
            "cross_signal_hotspots": [
                {"component": "drift_envelope", "issue": "neutral description"}
            ],
            "headline": "neutral headline about global health"
        }
    """
    # Collect all components
    components = {
        "metric_health": metric_health,
        "drift_envelope": drift_envelope,
        "semantic_envelope": semantic_envelope,
        "atlas_envelope": atlas_envelope,
        "telemetry_envelope": telemetry_envelope,
    }
    
    # Extract bands and build component summary
    envelope_components = {}
    bands = []
    critical_count = 0
    risk_count = 0
    warning_count = 0
    
    for component_name, component_data in components.items():
        present = component_data is not None
        band = _extract_component_band(component_data) if present else "UNKNOWN"
        
        envelope_components[component_name] = {
            "band": band,
            "present": present,
        }
        
        if band == "RED":
            critical_count += 1
            bands.append("RED")
        elif band == "YELLOW":
            risk_count += 1
            bands.append("YELLOW")
        elif band == "GREEN":
            bands.append("GREEN")
    
    # Determine global band
    # RED if ≥2 components in critical bands
    # YELLOW if 1 component in risk band or 3+ minor warnings
    # GREEN otherwise
    if critical_count >= 2:
        global_band = "RED"
    elif critical_count >= 1 or risk_count >= 1 or warning_count >= 3:
        global_band = "YELLOW"
    else:
        global_band = "GREEN"
    
    # Identify cross-signal hotspots (mismatched states, etc.)
    cross_signal_hotspots = []
    
    # Check for mismatched states (e.g., metric OK but semantic degraded)
    metric_band = envelope_components.get("metric_health", {}).get("band", "UNKNOWN")
    semantic_band = envelope_components.get("semantic_envelope", {}).get("band", "UNKNOWN")
    drift_band = envelope_components.get("drift_envelope", {}).get("band", "UNKNOWN")
    
    if metric_band == "GREEN" and semantic_band == "RED":
        cross_signal_hotspots.append({
            "component": "semantic_envelope",
            "issue": "Semantic consistency degraded despite metric health appearing stable",
        })
    
    if drift_band == "RED" and metric_band != "RED":
        cross_signal_hotspots.append({
            "component": "drift_envelope",
            "issue": "Governance drift in critical band while other signals stable",
        })
    
    # Build neutral headline
    if global_band == "RED":
        headline = f"Global health envelope: {critical_count} component(s) in critical state"
    elif global_band == "YELLOW":
        if critical_count >= 1:
            headline = f"Global health envelope: {critical_count} component(s) require attention"
        else:
            headline = f"Global health envelope: {risk_count} component(s) in risk state"
    else:
        headline = "Global health envelope: all components within acceptable ranges"
    
    return {
        "schema_version": "4.0.0",
        "global_band": global_band,
        "envelope_components": envelope_components,
        "cross_signal_hotspots": cross_signal_hotspots,
        "headline": headline,
    }


def analyze_system_coherence(
    metric_health: dict[str, Any] | None = None,
    drift_envelope: dict[str, Any] | None = None,
    semantic_envelope: dict[str, Any] | None = None,
    atlas_envelope: dict[str, Any] | None = None,
    telemetry_envelope: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Analyze system coherence — detects mismatched states across pillars.
    
    Identifies situations where different subsystems report conflicting
    health states (e.g., metrics OK but semantic degraded).
    
    Args:
        metric_health: Health signals from metrics subsystem
        drift_envelope: Governance drift signals (E1)
        semantic_envelope: Semantic consistency signals
        atlas_envelope: Structural/topology signals
        telemetry_envelope: NCI telemetry signals (E5)
    
    Returns:
        {
            "schema_version": "1.0.0",
            "coherence_status": "COHERENT" | "MISMATCHED" | "INSUFFICIENT_DATA",
            "mismatches": [
                {
                    "components": ["component1", "component2"],
                    "issue": "neutral description",
                    "severity": "LOW" | "MEDIUM" | "HIGH"
                }
            ],
            "notes": ["neutral note 1", ...]
        }
    """
    components = {
        "metric_health": metric_health,
        "drift_envelope": drift_envelope,
        "semantic_envelope": semantic_envelope,
        "atlas_envelope": atlas_envelope,
        "telemetry_envelope": telemetry_envelope,
    }
    
    # Extract bands for all present components
    component_bands = {}
    for name, data in components.items():
        if data:
            component_bands[name] = _extract_component_band(data)
    
    if len(component_bands) < 2:
        return {
            "schema_version": "1.0.0",
            "coherence_status": "INSUFFICIENT_DATA",
            "mismatches": [],
            "notes": ["Insufficient component data for coherence analysis"],
        }
    
    mismatches = []
    notes = []
    
    # Check for metric/semantic mismatch
    metric_band = component_bands.get("metric_health")
    semantic_band = component_bands.get("semantic_envelope")
    if metric_band and semantic_band:
        if metric_band == "GREEN" and semantic_band == "RED":
            mismatches.append({
                "components": ["metric_health", "semantic_envelope"],
                "issue": "Metrics indicate stable state while semantic consistency degraded",
                "severity": "HIGH",
            })
        elif metric_band == "RED" and semantic_band == "GREEN":
            mismatches.append({
                "components": ["metric_health", "semantic_envelope"],
                "issue": "Metrics indicate degraded state while semantic consistency stable",
                "severity": "MEDIUM",
            })
    
    # Check for drift/telemetry mismatch
    drift_band = component_bands.get("drift_envelope")
    telemetry_band = component_bands.get("telemetry_envelope")
    if drift_band and telemetry_band:
        if drift_band == "RED" and telemetry_band in ["GREEN", "YELLOW"]:
            mismatches.append({
                "components": ["drift_envelope", "telemetry_envelope"],
                "issue": "Governance drift in critical state while narrative telemetry indicates stability",
                "severity": "HIGH",
            })
    
    # Check for atlas/telemetry mismatch (structural vs narrative)
    atlas_band = component_bands.get("atlas_envelope")
    if atlas_band and telemetry_band:
        if atlas_band == "RED" and telemetry_band == "GREEN":
            mismatches.append({
                "components": ["atlas_envelope", "telemetry_envelope"],
                "issue": "Structural topology degraded while narrative consistency stable",
                "severity": "MEDIUM",
            })
    
    # Determine coherence status
    if mismatches:
        high_severity_count = sum(1 for m in mismatches if m["severity"] == "HIGH")
        if high_severity_count > 0:
            coherence_status = "MISMATCHED"
            notes.append(f"{high_severity_count} high-severity coherence mismatch(es) detected")
        else:
            coherence_status = "MISMATCHED"
            notes.append(f"{len(mismatches)} coherence mismatch(es) detected")
    else:
        coherence_status = "COHERENT"
        notes.append("All component signals aligned")
    
    # Add component band summary
    present_bands = ", ".join([f"{k}: {v}" for k, v in component_bands.items()])
    notes.append(f"Component bands: {present_bands}")
    
    return {
        "schema_version": "1.0.0",
        "coherence_status": coherence_status,
        "mismatches": mismatches,
        "notes": notes,
    }


def build_director_mega_panel(
    metric_health: dict[str, Any] | None = None,
    drift_envelope: dict[str, Any] | None = None,
    semantic_envelope: dict[str, Any] | None = None,
    atlas_envelope: dict[str, Any] | None = None,
    telemetry_envelope: dict[str, Any] | None = None,
    nci_director_panel: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build Director Mega-Panel — combines E1-E5 alignment into single release-grade artifact.
    
    Synthesizes all agent signals (E1: Governance, E2-E4: Other, E5: Narrative)
    into a unified executive dashboard.
    
    Args:
        metric_health: Health signals from metrics subsystem
        drift_envelope: Governance drift signals (E1)
        semantic_envelope: Semantic consistency signals
        atlas_envelope: Structural/topology signals
        telemetry_envelope: NCI telemetry signals (E5)
        nci_director_panel: Optional E5 director panel output
    
    Returns:
        {
            "schema_version": "1.0.0",
            "release_ready": true | false,
            "mega_status_light": "🟢" | "🟡" | "🔴",
            "component_summary": {
                "E1_governance": {"status": "OK", "band": "GREEN"},
                "E5_narrative": {"status": "OK", "band": "GREEN"},
                ...
            },
            "global_envelope": {...},  # Full envelope v4 output
            "coherence_analysis": {...},  # Coherence analysis output
            "executive_headline": "neutral executive summary"
        }
    """
    # Build global envelope
    global_envelope = build_global_health_envelope_v4(
        metric_health=metric_health,
        drift_envelope=drift_envelope,
        semantic_envelope=semantic_envelope,
        atlas_envelope=atlas_envelope,
        telemetry_envelope=telemetry_envelope,
    )
    
    # Build coherence analysis
    coherence_analysis = analyze_system_coherence(
        metric_health=metric_health,
        drift_envelope=drift_envelope,
        semantic_envelope=semantic_envelope,
        atlas_envelope=atlas_envelope,
        telemetry_envelope=telemetry_envelope,
    )
    
    # Map components to agent names
    component_summary = {}
    
    # E1: Governance Drift
    if drift_envelope:
        drift_band = _extract_component_band(drift_envelope)
        component_summary["E1_governance"] = {
            "status": "OK" if drift_band == "GREEN" else ("WARN" if drift_band == "YELLOW" else "CRITICAL"),
            "band": drift_band,
        }
    
    # E5: Narrative (NCI Telemetry)
    if telemetry_envelope or nci_director_panel:
        telemetry_band = _extract_component_band(telemetry_envelope or nci_director_panel or {})
        component_summary["E5_narrative"] = {
            "status": "OK" if telemetry_band == "GREEN" else ("WARN" if telemetry_band == "YELLOW" else "CRITICAL"),
            "band": telemetry_band,
        }
    
    # Other components
    if metric_health:
        metric_band = _extract_component_band(metric_health)
        component_summary["metrics"] = {
            "status": "OK" if metric_band == "GREEN" else ("WARN" if metric_band == "YELLOW" else "CRITICAL"),
            "band": metric_band,
        }
    
    if semantic_envelope:
        semantic_band = _extract_component_band(semantic_envelope)
        component_summary["semantic"] = {
            "status": "OK" if semantic_band == "GREEN" else ("WARN" if semantic_band == "YELLOW" else "CRITICAL"),
            "band": semantic_band,
        }
    
    if atlas_envelope:
        atlas_band = _extract_component_band(atlas_envelope)
        component_summary["atlas"] = {
            "status": "OK" if atlas_band == "GREEN" else ("WARN" if atlas_band == "YELLOW" else "CRITICAL"),
            "band": atlas_band,
        }
    
    # Determine mega status light
    global_band = global_envelope.get("global_band", "GREEN")
    if global_band == "RED":
        mega_status_light = "🔴"
    elif global_band == "YELLOW":
        mega_status_light = "🟡"
    else:
        mega_status_light = "🟢"
    
    # Determine release readiness
    # Not ready if: RED band, or coherence mismatches, or multiple critical components
    release_ready = (
        global_band != "RED"
        and coherence_analysis.get("coherence_status") != "MISMATCHED"
        and global_envelope.get("envelope_components", {}).get("drift_envelope", {}).get("band") != "RED"
        and global_envelope.get("envelope_components", {}).get("telemetry_envelope", {}).get("band") != "RED"
    )
    
    # Build executive headline
    if not release_ready:
        if global_band == "RED":
            executive_headline = "System not release-ready: critical health issues detected"
        elif coherence_analysis.get("coherence_status") == "MISMATCHED":
            executive_headline = "System not release-ready: component coherence mismatches detected"
        else:
            executive_headline = "System not release-ready: health checks indicate attention required"
    else:
        executive_headline = "System release-ready: all component signals within acceptable ranges"
    
    return {
        "schema_version": "1.0.0",
        "release_ready": release_ready,
        "mega_status_light": mega_status_light,
        "component_summary": component_summary,
        "global_envelope": global_envelope,
        "coherence_analysis": coherence_analysis,
        "executive_headline": executive_headline,
    }


# ==============================================================================
# CLI INTERFACE
# ==============================================================================


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Narrative Consistency Index (NCI) System"
    )
    parser.add_argument(
        "--mode",
        choices=["index", "drift", "advisor", "bucket-report", "delta-since", "hotspots", "ci-summary", "quick-fixes", "insight-grid"],
        default="index",
        help="Operating mode",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/narrative",
        help="Output directory for generated files",
    )
    parser.add_argument(
        "--base-commit",
        type=str,
        default="HEAD~10",
        help="Base commit for drift/delta detection",
    )
    parser.add_argument(
        "--head-commit",
        type=str,
        default="HEAD",
        help="Head commit for drift/delta detection",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of top hot spots to include",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output",
    )
    
    args = parser.parse_args()
    
    # Find repo root
    script_path = Path(__file__).resolve()
    repo_root = script_path.parent.parent
    
    # Create output directory
    output_dir = repo_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("NARRATIVE CONSISTENCY INDEX (NCI) — doc-ops-5 / E5")
    print("=" * 70)
    print(f"Mode: {args.mode}")
    print(f"Repository: {repo_root}")
    print()
    
    if args.mode == "index":
        # Compute NCI
        indexer = NarrativeConsistencyIndexer(repo_root, verbose=args.verbose)
        index = indexer.compute_index()
        
        # Generate advisor suggestions
        advisor = DocumentationAdvisor(index)
        index.advisor_suggestions = [
            {
                "file": s.file,
                "line": s.line,
                "category": s.category,
                "priority": s.priority,
                "suggestion": s.suggestion,
            }
            for s in advisor.generate_suggestions()
        ]
        
        # Save JSON
        json_path = output_dir / "narrative_index.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(index.to_dict(), f, indent=2)
        print(f"Index saved: {json_path}")
        
        # Generate heatmap
        heatmap_path = output_dir / "narrative_heatmap.png"
        if generate_heatmap(index, heatmap_path):
            print(f"Heatmap saved: {heatmap_path}")
        
        # Print summary
        print()
        print(f"Global NCI: {index.global_nci:.4f}")
        print("Category Scores:")
        for cat, score in index.category_scores.items():
            print(f"  {cat}: {score:.4f}")
        print()
        print(f"Documents analyzed: {index.total_documents}")
        print(f"Advisor suggestions: {len(index.advisor_suggestions)}")
        
        return 0 if index.global_nci >= 0.8 else 1
    
    elif args.mode == "drift":
        # Detect drift
        detector = DriftDetector(repo_root, verbose=args.verbose)
        report = detector.compute_drift(args.base_commit, args.head_commit)
        
        # Save report
        drift_path = output_dir / "drift_report.json"
        with open(drift_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(report), f, indent=2)
        print(f"Drift report saved: {drift_path}")
        
        # Print summary
        print()
        print(f"Base: {report.base_commit} → Head: {report.head_commit}")
        print(f"Files changed: {report.files_changed}")
        print(f"Terminology drifts: {len(report.terminology_drift)}")
        print(f"Definition drifts: {len(report.definition_drift)}")
        print(f"NCI Delta: {report.nci_delta:+.4f}")
        print(f"Severity: {report.drift_severity.upper()}")
        
        return 0 if report.drift_severity in ["none", "minor"] else 1
    
    elif args.mode == "advisor":
        # Generate advisor suggestions
        indexer = NarrativeConsistencyIndexer(repo_root, verbose=args.verbose)
        index = indexer.compute_index()
        
        advisor = DocumentationAdvisor(index)
        suggestions = advisor.generate_suggestions()
        
        # Save suggestions
        advisor_path = output_dir / "advisor_suggestions.json"
        with open(advisor_path, 'w', encoding='utf-8') as f:
            json.dump([asdict(s) for s in suggestions], f, indent=2)
        print(f"Suggestions saved: {advisor_path}")
        
        # Print summary
        print()
        print(f"Total suggestions: {len(suggestions)}")
        
        high = sum(1 for s in suggestions if s.priority == "high")
        medium = sum(1 for s in suggestions if s.priority == "medium")
        low = sum(1 for s in suggestions if s.priority == "low")
        
        print(f"  High priority: {high}")
        print(f"  Medium priority: {medium}")
        print(f"  Low priority: {low}")
        
        if high > 0:
            print()
            print("High Priority Suggestions:")
            for s in suggestions[:10]:
                if s.priority == "high":
                    print(f"  • {s.file}:{s.line} — {s.suggestion}")
        
        return 0
    
    elif args.mode == "bucket-report":
        # Generate bucketed NCI report
        indexer = NarrativeConsistencyIndexer(repo_root, verbose=args.verbose)
        index = indexer.compute_index()
        
        generator = BucketReportGenerator(index)
        report_md = generator.generate_markdown_report()
        
        # Save Markdown report
        report_path = output_dir / "bucket_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_md)
        print(f"Bucket report saved: {report_path}")
        
        # Also save JSON summary
        summaries = generator.compute_bucket_summaries()
        json_path = output_dir / "bucket_summary.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "global_nci": round(index.global_nci, 4),
                "buckets": [
                    {
                        "category": s.category,
                        "document_count": s.document_count,
                        "avg_nci": round(s.avg_nci, 4),
                        "avg_terminology": round(s.avg_terminology, 4),
                        "avg_phase": round(s.avg_phase, 4),
                        "avg_uplift": round(s.avg_uplift, 4),
                        "avg_structure": round(s.avg_structure, 4),
                        "worst_files": [
                            {"path": p, "nci": round(n, 4)} for p, n in s.worst_files
                        ],
                    }
                    for s in summaries
                ],
            }, f, indent=2)
        print(f"Bucket summary JSON saved: {json_path}")
        
        # Print summary
        print()
        print(f"Global NCI: {index.global_nci:.2f}")
        print()
        print("Bucket Summary:")
        for s in summaries:
            print(f"  {s.category.title():12} — NCI: {s.avg_nci:.2f} ({s.document_count} docs)")
        
        return 0 if index.global_nci >= 0.8 else 1
    
    elif args.mode == "delta-since":
        # Compute NCI delta between commits
        calculator = NarrativeDeltaCalculator(repo_root, verbose=args.verbose)
        delta = calculator.compute_delta(args.base_commit, args.head_commit)
        
        # Save JSON
        delta_path = output_dir / "narrative_delta.json"
        with open(delta_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(delta), f, indent=2)
        print(f"Narrative delta saved: {delta_path}")
        
        # Print summary
        print()
        print(f"Base commit: {delta.base_commit}")
        print(f"Head commit: {delta.head_commit}")
        print()
        print(f"Base NCI: {delta.base_nci:.4f}")
        print(f"Head NCI: {delta.head_nci:.4f}")
        print(f"Delta:    {delta.delta:+.4f}")
        print()
        print(f"Changed files: {len(delta.changed_files)}")
        print(f"File-level deltas: {len(delta.file_deltas)}")
        
        if delta.file_deltas:
            print()
            print("Significant file changes:")
            for fd in delta.file_deltas[:10]:
                if fd.get("delta") is not None:
                    print(f"  {fd['file']}: {fd['delta']:+.4f}")
                elif fd.get("status") == "added":
                    print(f"  {fd['file']}: (added)")
                elif fd.get("status") == "removed":
                    print(f"  {fd['file']}: (removed)")
        
        return 0
    
    elif args.mode == "hotspots":
        # Generate hot spots analysis
        indexer = NarrativeConsistencyIndexer(repo_root, verbose=args.verbose)
        index = indexer.compute_index()
        
        analyzer = HotSpotsAnalyzer(index, top_n=args.top_n)
        hotspots_json = analyzer.to_json()
        
        # Save JSON
        hotspots_path = output_dir / "narrative_hotspots.json"
        with open(hotspots_path, 'w', encoding='utf-8') as f:
            json.dump(hotspots_json, f, indent=2)
        print(f"Hot spots saved: {hotspots_path}")
        
        # Print summary
        print()
        print(f"Global NCI: {index.global_nci:.4f}")
        print(f"Total hot spots identified: {hotspots_json['total_hotspots']}")
        print()
        
        if hotspots_json["hotspots"]:
            print(f"Top {min(10, len(hotspots_json['hotspots']))} Hot Spots:")
            for h in hotspots_json["hotspots"][:10]:
                print(f"  #{h['rank']:2} {h['file']}")
                print(f"      NCI: {h['nci_score']:.2f} | Contribution: {h['contribution_pct']:.1f}% | Issue: {h['primary_issue']}")
        
        print()
        print("By Category:")
        for cat, data in hotspots_json["summary"]["by_category"].items():
            print(f"  {cat}: {data['count']} files ({data['total_contribution']:.1f}% contribution)")
        
        print()
        print("By Primary Issue:")
        for issue, count in hotspots_json["summary"]["by_primary_issue"].items():
            print(f"  {issue}: {count} files")
        
        return 0
    
    elif args.mode == "ci-summary":
        # CI-friendly summary output (always exits 0, advisory only)
        # Uses neutral language: no "good/bad/fail/pass"
        indexer = NarrativeConsistencyIndexer(repo_root, verbose=False)
        index = indexer.compute_index()
        
        # Find bucket with highest variance (lowest NCI)
        bucket_scores = {}
        for doc in index.documents:
            if doc.category not in bucket_scores:
                bucket_scores[doc.category] = []
            bucket_scores[doc.category].append(doc.nci_score)
        
        bucket_nci = {
            cat: sum(scores) / len(scores) if scores else 1.0
            for cat, scores in bucket_scores.items()
        }
        highest_variance_bucket = min(bucket_nci, key=bucket_nci.get) if bucket_nci else "none"
        highest_variance_bucket_nci = bucket_nci.get(highest_variance_bucket, 1.0)
        
        # Get top 3 files with highest variance
        analyzer = HotSpotsAnalyzer(index, top_n=3)
        top_variance_files = analyzer.compute_hotspots()
        
        # Print dashboard summary (neutral language)
        print("=" * 60)
        print("NARRATIVE CONSISTENCY INDEX — DASHBOARD")
        print("=" * 60)
        print()
        print(f"Global NCI:            {index.global_nci:.4f}")
        print()
        print(f"Highest Variance:      {highest_variance_bucket} (NCI: {highest_variance_bucket_nci:.4f})")
        print()
        print("Files Needing Attention (by variance contribution):")
        if top_variance_files:
            for i, h in enumerate(top_variance_files, 1):
                print(f"  {i}. {h.file}")
                print(f"     NCI: {h.nci_score:.2f} | Variance: {h.contribution_pct:.1f}% | Area: {h.primary_issue}")
        else:
            print("  (no files with high variance)")
        print()
        print("=" * 60)
        
        # Always exit 0 for CI (advisory only, not for gating)
        return 0
    
    elif args.mode == "quick-fixes":
        # Generate ten-minute fix suggestions
        indexer = NarrativeConsistencyIndexer(repo_root, verbose=args.verbose)
        index = indexer.compute_index()
        
        generator = TenMinuteFixGenerator(index, max_suggestions=args.top_n)
        fixes_json = generator.to_json()
        
        # Save JSON
        fixes_path = output_dir / "quick_fixes.json"
        with open(fixes_path, 'w', encoding='utf-8') as f:
            json.dump(fixes_json, f, indent=2)
        print(f"Quick fixes saved: {fixes_path}")
        
        # Print summary (derive counts from suggestions list)
        suggestions = fixes_json["suggestions"]
        print()
        print(f"Global NCI: {fixes_json['global_nci']:.4f}")
        print(f"Suggestions: {len(suggestions)}")
        print()
        
        if suggestions:
            print("Ten-Minute Fixes (sorted by priority, normalized 0-1):")
            print("-" * 70)
            for s in suggestions:
                print(f"  File: {s['file']}")
                print(f"    Issue: {s['issue_type']} | Priority: {s['priority_score']}")
                print(f"    Hint: {s['hint']}")
                print(f"    Effort: {s['estimated_effort']}")
                print()
            
            # Summarize by issue type (derived)
            issue_counts = defaultdict(int)
            for s in suggestions:
                issue_counts[s["issue_type"]] += 1
            
            print("By Issue Type:")
            for issue, count in sorted(issue_counts.items()):
                print(f"  {issue}: {count}")
        
        return 0
    
    elif args.mode == "insight-grid":
        # Generate NCI Insight Grid — consolidated dashboard view
        indexer = NarrativeConsistencyIndexer(repo_root, verbose=args.verbose)
        index = indexer.compute_index()
        
        # Build area view
        area_view = build_nci_area_view(index)
        
        # Build quick fixes for insight summary
        generator = TenMinuteFixGenerator(index, max_suggestions=args.top_n)
        quick_fixes = generator.to_json()
        
        # Build insight summary
        insight_summary = build_nci_insight_summary(quick_fixes, area_view)
        
        # Save all outputs
        area_view_path = output_dir / "nci_area_view.json"
        with open(area_view_path, 'w', encoding='utf-8') as f:
            json.dump(area_view, f, indent=2)
        
        insight_path = output_dir / "nci_insight_summary.json"
        with open(insight_path, 'w', encoding='utf-8') as f:
            json.dump(insight_summary, f, indent=2)
        
        print(f"Area view saved: {area_view_path}")
        print(f"Insight summary saved: {insight_path}")
        
        # Print summary
        print()
        print("=" * 60)
        print("NCI INSIGHT GRID v1.3")
        print("=" * 60)
        print()
        print(f"Global NCI: {insight_summary['global_nci']:.4f}")
        print(f"Dominant Area: {insight_summary['dominant_area']}")
        print(f"Suggestion Count: {insight_summary['suggestion_count']}")
        print()
        print("Area NCI Breakdown:")
        for area_name, stats in area_view["areas"].items():
            print(f"  {area_name}:")
            print(f"    NCI: {stats['nci']:.4f} | Variance: {stats['variance']:.4f} | Docs: {stats['doc_count']}")
        print()
        print("Top Files Needing Attention:")
        for i, f in enumerate(insight_summary["top_files"][:5], 1):
            print(f"  {i}. {f}")
        print()
        print("=" * 60)
        
        return 0
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

