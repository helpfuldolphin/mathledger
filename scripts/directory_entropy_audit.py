#!/usr/bin/env python3
"""
PHASE II — DIRECTORY ENTROPY AUDITOR

Agent: E3 (doc-ops-3) — Directory Geometry Analyst & Hardening Sentinel
Purpose: Compute directory entropy scores, detect boundary violations,
         classify directory archetypes, track mutation events, and
         compute structure risk indices.

Usage:
    uv run python scripts/directory_entropy_audit.py [OPTIONS]

Options:
    --output FILE           Output JSON report (default: directory_entropy_report.json)
    --guardian              Run Phase Boundary Guardian mode
    --classify              Run Directory Archetype Classifier
    --history               Analyze git history for mutation events
    --remediation           Generate remediation suggestions
    --risk-index            Compute Structure Risk Index and output ranked summary
    --phase PHASE           Filter to phase1 or phase2 directories only
    --compare OLD NEW       Compare two reports to detect structural drift
    --refactor-candidates   Generate refactor candidate shortlist
    --verbose               Verbose output
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent

# Directory archetype definitions
DIRECTORY_ARCHETYPES = {
    "runtime": {
        "description": "Code that executes during system operation",
        "indicators": ["runner", "worker", "server", "api", "app"],
        "expected_extensions": [".py", ".js", ".ts"],
        "directories": [
            "backend/orchestrator",
            "backend/worker.py",
            "backend/runner",
            "backend/api",
        ],
    },
    "analysis": {
        "description": "Data analysis and visualization code",
        "indicators": ["analysis", "analyze", "plot", "visualize", "metrics"],
        "expected_extensions": [".py", ".ipynb"],
        "directories": [
            "analysis",
            "experiments",
            "backend/metrics",
            "backend/causal",
        ],
    },
    "evidence": {
        "description": "Artifacts, outputs, and attestations",
        "indicators": ["artifacts", "results", "output", "attestation", "evidence"],
        "expected_extensions": [".json", ".jsonl", ".csv", ".png", ".pdf"],
        "directories": [
            "artifacts",
            "results",
            "attestation",
        ],
    },
    "schema": {
        "description": "Data definitions and validation",
        "indicators": ["schema", "model", "types", "validation"],
        "expected_extensions": [".py", ".json", ".yaml"],
        "directories": [
            "backend/api/schemas.py",
            "backend/models",
            "backend/telemetry",
            "migrations",
        ],
    },
    "governance": {
        "description": "Policy, rules, and compliance",
        "indicators": ["governance", "policy", "rules", "compliance", "audit"],
        "expected_extensions": [".py", ".md", ".yaml"],
        "directories": [
            "backend/governance",
            "ops",
            "docs/audits",
        ],
    },
    "infrastructure": {
        "description": "Build, deployment, and configuration",
        "indicators": ["infra", "config", "docker", "ci", "deploy"],
        "expected_extensions": [".yaml", ".yml", ".json", ".toml", ".env"],
        "directories": [
            "infra",
            "config",
            ".github",
        ],
    },
    "documentation": {
        "description": "Non-code documentation and specifications",
        "indicators": ["docs", "readme", "spec", "guide"],
        "expected_extensions": [".md", ".pdf", ".tex", ".txt"],
        "directories": [
            "docs",
        ],
    },
    "testing": {
        "description": "Test suites and fixtures",
        "indicators": ["test", "tests", "spec", "fixture"],
        "expected_extensions": [".py"],
        "directories": [
            "tests",
        ],
    },
    "cryptographic": {
        "description": "Cryptographic primitives and verification",
        "indicators": ["crypto", "hash", "sign", "verify", "attestation"],
        "expected_extensions": [".py", ".lean"],
        "directories": [
            "backend/crypto",
            "backend/lean_proj",
        ],
    },
}

# Expected file patterns per directory (for entropy calculation)
DIRECTORY_EXPECTATIONS = {
    "backend": {
        "expected_extensions": [".py"],
        "forbidden_extensions": [".md", ".txt"],
        "naming_pattern": r"^[a-z_]+\.py$",
    },
    "docs": {
        "expected_extensions": [".md", ".pdf", ".tex", ".json", ".yaml", ".png", ".svg"],
        "forbidden_extensions": [".py", ".js", ".ts", ".lean"],
        "naming_pattern": r"^[A-Z_][A-Za-z0-9_\-\.]+\.(md|pdf|tex|json|yaml|png|svg)$",
    },
    "tests": {
        "expected_extensions": [".py"],
        "forbidden_extensions": [".md"],
        "naming_pattern": r"^(test_[a-z_]+|conftest|__init__)\.py$",
    },
    "scripts": {
        "expected_extensions": [".py", ".ps1", ".sh"],
        "forbidden_extensions": [".md"],
        "naming_pattern": r"^[a-z_\-]+\.(py|ps1|sh)$",
    },
    "config": {
        "expected_extensions": [".yaml", ".yml", ".json", ".env"],
        "forbidden_extensions": [".py"],
        "naming_pattern": r"^[a-z_\-\.]+\.(yaml|yml|json|env)$",
    },
    "migrations": {
        "expected_extensions": [".sql"],
        "forbidden_extensions": [".py", ".md"],
        "naming_pattern": r"^\d{3}_[a-z_]+\.sql$",
    },
    "artifacts": {
        "expected_extensions": [".json", ".jsonl", ".csv", ".png", ".pdf", ".tex"],
        "forbidden_extensions": [".py"],
        "naming_pattern": None,  # Flexible naming
    },
    "results": {
        "expected_extensions": [".json", ".jsonl", ".csv", ".txt"],
        "forbidden_extensions": [".py"],
        "naming_pattern": None,
    },
}

# Phase I directories (should not import Phase II)
PHASE_I_DIRECTORIES = [
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
    "backend/verification",
]

# Phase II directories (contain Phase II code/experiments)
PHASE_II_DIRECTORIES = [
    "analysis",
    "artifacts/phase_ii",
    "artifacts/u2",
    "backend/metrics",
    "backend/promotion",
    "backend/runner",
    "backend/security",
    "backend/telemetry",
    "experiments/synthetic_uplift",
    "experiments/u2",
    "tests/phase2",
    "tests/env",
    "tests/metrics",
]

# Risk classification thresholds
RISK_THRESHOLDS = {
    "entropy_high": 2.0,       # Directory entropy above this = HIGH risk
    "entropy_medium": 1.0,     # Directory entropy above this = MEDIUM risk
    "violations_high": 10,     # Violation count above this = HIGH risk
    "violations_medium": 3,    # Violation count above this = MEDIUM risk
    "global_risk_high": 0.6,   # Global risk score above this = concerning
    "global_risk_medium": 0.3, # Global risk score above this = needs attention
}

# Phase II import patterns (forbidden in Phase I)
PHASE_II_IMPORT_PATTERNS = [
    (r"from\s+backend\.metrics\.u2_analysis", "backend.metrics.u2_analysis"),
    (r"from\s+backend\.metrics\.statistical", "backend.metrics.statistical"),
    (r"from\s+backend\.runner\.u2_runner", "backend.runner.u2_runner"),
    (r"from\s+backend\.runner\s+import\s+u2_runner", "backend.runner.u2_runner"),
    (r"from\s+backend\.telemetry\.u2_schema", "backend.telemetry.u2_schema"),
    (r"from\s+backend\.telemetry\s+import\s+u2_schema", "backend.telemetry.u2_schema"),
    (r"from\s+backend\.security\.u2_security", "backend.security.u2_security"),
    (r"from\s+backend\.security\s+import\s+u2_security", "backend.security.u2_security"),
    (r"from\s+backend\.promotion\.u2_evidence", "backend.promotion.u2_evidence"),
    (r"from\s+backend\.promotion\s+import\s+u2_evidence", "backend.promotion.u2_evidence"),
    (r"from\s+experiments\.run_uplift_u2", "experiments.run_uplift_u2"),
    (r"from\s+experiments\.u2_", "experiments.u2_*"),
    (r"from\s+analysis\.u2_", "analysis.u2_*"),
    (r"from\s+tests\.phase2", "tests.phase2"),
    (r"import\s+backend\.metrics\.u2_analysis", "backend.metrics.u2_analysis"),
    (r"import\s+backend\.runner\.u2_runner", "backend.runner.u2_runner"),
    (r"import\s+backend\.telemetry\.u2_schema", "backend.telemetry.u2_schema"),
    (r"import\s+backend\.security\.u2_security", "backend.security.u2_security"),
    (r"import\s+tests\.phase2", "tests.phase2"),
]


# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------

@dataclass
class EntropyScore:
    """Entropy score for a directory."""
    directory: str
    total_entropy: float
    extension_entropy: float
    naming_entropy: float
    violation_count: int
    unexpected_files: list[str]
    forbidden_files: list[str]
    naming_violations: list[str]
    file_count: int
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BoundaryViolation:
    """A Phase I → Phase II import violation."""
    source_file: str
    line_number: int
    import_statement: str
    target_module: str
    severity: str  # "error" or "warning"
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RemediationSuggestion:
    """Suggested fix for a boundary violation."""
    violation: BoundaryViolation
    suggestion_type: str  # "promote", "lazy_import", "dependency_inversion", "relocate"
    description: str
    code_example: str | None = None
    
    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["violation"] = self.violation.to_dict()
        return d


@dataclass
class ArchetypeClassification:
    """Classification of a directory into an archetype."""
    directory: str
    primary_archetype: str
    confidence: float
    secondary_archetypes: list[tuple[str, float]]
    role_violations: list[str]
    naming_consistency: float
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MutationEvent:
    """A directory structure mutation event from git history."""
    commit_hash: str
    timestamp: str
    author: str
    event_type: str  # "file_added", "file_removed", "file_moved", "dir_created"
    path: str
    details: str | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EntropyReport:
    """Complete entropy audit report."""
    timestamp: str
    project_root: str
    report_hash: str
    total_directories: int
    total_files: int
    global_entropy: float
    directory_scores: list[EntropyScore]
    boundary_violations: list[BoundaryViolation]
    remediation_suggestions: list[RemediationSuggestion]
    archetype_classifications: list[ArchetypeClassification]
    mutation_events: list[MutationEvent]
    summary: dict[str, Any]
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "project_root": self.project_root,
            "report_hash": self.report_hash,
            "total_directories": self.total_directories,
            "total_files": self.total_files,
            "global_entropy": self.global_entropy,
            "directory_scores": [s.to_dict() for s in self.directory_scores],
            "boundary_violations": [v.to_dict() for v in self.boundary_violations],
            "remediation_suggestions": [r.to_dict() for r in self.remediation_suggestions],
            "archetype_classifications": [c.to_dict() for c in self.archetype_classifications],
            "mutation_events": [m.to_dict() for m in self.mutation_events],
            "summary": self.summary,
        }


@dataclass
class DirectoryRisk:
    """Risk assessment for a single directory."""
    path: str
    risk_score: float  # 0.0 to 1.0
    risk_level: str    # "LOW", "MEDIUM", "HIGH"
    entropy: float
    violations: int
    contributing_factors: list[str]
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StructureRiskIndex:
    """Complete structure risk index."""
    timestamp: str
    global_risk_score: float  # 0.0 to 1.0
    global_risk_level: str    # "LOW", "MEDIUM", "HIGH"
    top_risky_directories: list[DirectoryRisk]
    phase_filter: str | None  # "phase1", "phase2", or None
    total_directories_analyzed: int
    risk_distribution: dict[str, int]  # {"LOW": n, "MEDIUM": m, "HIGH": k}
    drift_indicators: list[str]
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "global_risk_score": self.global_risk_score,
            "global_risk_level": self.global_risk_level,
            "top_risky_directories": [d.to_dict() for d in self.top_risky_directories],
            "phase_filter": self.phase_filter,
            "total_directories_analyzed": self.total_directories_analyzed,
            "risk_distribution": self.risk_distribution,
            "drift_indicators": self.drift_indicators,
        }


@dataclass
class DirectoryDrift:
    """Drift metrics for a single directory between two reports."""
    path: str
    entropy_old: float
    entropy_new: float
    entropy_delta: float
    violations_old: int
    violations_new: int
    violations_delta: int
    risk_score_old: float
    risk_score_new: float
    risk_delta: float
    risk_level_old: str
    risk_level_new: str
    is_new_directory: bool
    is_removed_directory: bool
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StructureDriftReport:
    """
    Complete drift analysis between two structure reports.
    
    Schema Contract (v1):
    {
      "old_report": "path/to/old.json",
      "new_report": "path/to/new.json",
      "overall_trend": "improving|degrading|stable",
      "max_risk_increase": float,
      "max_risk_decrease": float,
      "directories_with_increased_risk": [...],  # Sorted alphabetically
      "directories_with_decreased_risk": [...],  # Sorted alphabetically
      "new_directories": [...],                   # Sorted alphabetically
      "removed_directories": [...],               # Sorted alphabetically
      "risk_transitions": {...},                  # LOW→HIGH, etc.
      ...extended fields...
    }
    """
    timestamp: str
    old_report_path: str
    new_report_path: str
    global_risk_old: float
    global_risk_new: float
    global_risk_delta: float
    max_risk_increase: float
    max_risk_decrease: float
    directories_with_increased_risk: list[str]
    directories_with_decreased_risk: list[str]
    new_directories: list[str]
    removed_directories: list[str]
    directory_drifts: list[DirectoryDrift]
    summary: dict[str, Any]
    risk_transitions: dict[str, list[str]] | None = None  # e.g., {"LOW_TO_HIGH": [...]}
    
    def to_dict(self) -> dict[str, Any]:
        """
        Serialize to dict with stable, deterministic ordering.
        
        Contract: All directory lists are sorted alphabetically for
        reproducibility and diff-friendliness.
        """
        # Ensure deterministic ordering of all directory lists
        return {
            # Contract-required fields (v1 schema)
            "old_report": self.old_report_path,
            "new_report": self.new_report_path,
            "overall_trend": self.summary.get("overall_trend", "stable"),
            "max_risk_increase": self.max_risk_increase,
            "max_risk_decrease": self.max_risk_decrease,
            "directories_with_increased_risk": sorted(self.directories_with_increased_risk),
            "directories_with_decreased_risk": sorted(self.directories_with_decreased_risk),
            "new_directories": sorted(self.new_directories),
            "removed_directories": sorted(self.removed_directories),
            "risk_transitions": self.risk_transitions or {},
            # Extended fields (for detailed analysis)
            "timestamp": self.timestamp,
            "global_risk_old": self.global_risk_old,
            "global_risk_new": self.global_risk_new,
            "global_risk_delta": self.global_risk_delta,
            "directory_drifts": [d.to_dict() for d in self.directory_drifts],
            "summary": self.summary,
        }


@dataclass
class RefactorCandidate:
    """A directory identified as a candidate for structural refactoring."""
    path: str
    risk_score: float
    risk_level: str
    entropy: float
    violations: int
    is_phase_ii: bool
    risk_increased_recently: bool
    risk_delta: float | None  # Delta from drift analysis if available
    priority: str  # "CRITICAL", "HIGH", "MEDIUM"
    reasons: list[str]
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# -----------------------------------------------------------------------------
# Entropy Calculation
# -----------------------------------------------------------------------------

def compute_shannon_entropy(items: list[str]) -> float:
    """Compute Shannon entropy of a list of items."""
    if not items:
        return 0.0
    
    counts = Counter(items)
    total = len(items)
    
    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    
    return entropy


def compute_directory_entropy(directory: Path, expectations: dict | None = None, base_path: Path | None = None) -> EntropyScore:
    """Compute entropy score for a single directory."""
    if base_path is None:
        base_path = PROJECT_ROOT
    
    try:
        rel_dir = directory.relative_to(base_path).as_posix()
    except ValueError:
        # Directory is outside base_path, use absolute path
        rel_dir = str(directory)
    
    # Get all files in directory (non-recursive for this directory only)
    files = []
    try:
        for item in directory.iterdir():
            if item.is_file() and not item.name.startswith("."):
                files.append(item)
    except PermissionError:
        pass
    
    if not files:
        return EntropyScore(
            directory=rel_dir,
            total_entropy=0.0,
            extension_entropy=0.0,
            naming_entropy=0.0,
            violation_count=0,
            unexpected_files=[],
            forbidden_files=[],
            naming_violations=[],
            file_count=0,
        )
    
    # Get expectations for this directory type
    if expectations is None:
        expectations = {}
        for key, exp in DIRECTORY_EXPECTATIONS.items():
            if rel_dir.startswith(key) or rel_dir == key:
                expectations = exp
                break
    
    # Compute extension entropy
    extensions = [f.suffix.lower() for f in files]
    extension_entropy = compute_shannon_entropy(extensions)
    
    # Compute naming pattern entropy (how varied are the naming patterns)
    name_patterns = []
    for f in files:
        # Classify name pattern
        if f.name.startswith("test_"):
            name_patterns.append("test_prefix")
        elif f.name.startswith("__"):
            name_patterns.append("dunder")
        elif f.name[0].isupper():
            name_patterns.append("capitalized")
        elif "_" in f.name:
            name_patterns.append("snake_case")
        elif "-" in f.name:
            name_patterns.append("kebab-case")
        else:
            name_patterns.append("simple")
    
    naming_entropy = compute_shannon_entropy(name_patterns)
    
    # Check for violations
    unexpected_files = []
    forbidden_files = []
    naming_violations = []
    
    expected_ext = expectations.get("expected_extensions", [])
    forbidden_ext = expectations.get("forbidden_extensions", [])
    naming_pattern = expectations.get("naming_pattern")
    
    for f in files:
        ext = f.suffix.lower()
        
        # Check unexpected extensions
        if expected_ext and ext and ext not in expected_ext:
            unexpected_files.append(f.name)
        
        # Check forbidden extensions
        if ext in forbidden_ext:
            forbidden_files.append(f.name)
        
        # Check naming pattern
        if naming_pattern and not re.match(naming_pattern, f.name):
            naming_violations.append(f.name)
    
    # Compute total entropy (weighted combination)
    violation_count = len(unexpected_files) + len(forbidden_files) + len(naming_violations)
    violation_penalty = violation_count * 0.5  # Each violation adds 0.5 to entropy
    
    total_entropy = (
        extension_entropy * 0.4 +
        naming_entropy * 0.3 +
        violation_penalty * 0.3
    )
    
    return EntropyScore(
        directory=rel_dir,
        total_entropy=round(total_entropy, 4),
        extension_entropy=round(extension_entropy, 4),
        naming_entropy=round(naming_entropy, 4),
        violation_count=violation_count,
        unexpected_files=unexpected_files,
        forbidden_files=forbidden_files,
        naming_violations=naming_violations,
        file_count=len(files),
    )


def compute_all_directory_entropies() -> list[EntropyScore]:
    """Compute entropy for all directories in the project."""
    scores = []
    
    # Walk all directories
    for dirpath in PROJECT_ROOT.rglob("*"):
        if not dirpath.is_dir():
            continue
        
        # Skip hidden directories and common exclusions
        rel_path = dirpath.relative_to(PROJECT_ROOT).as_posix()
        if any(part.startswith(".") for part in rel_path.split("/")):
            continue
        if "__pycache__" in rel_path:
            continue
        if "node_modules" in rel_path:
            continue
        if ".venv" in rel_path:
            continue
        
        score = compute_directory_entropy(dirpath)
        if score.file_count > 0:  # Only include directories with files
            scores.append(score)
    
    # Sort by entropy (highest first)
    scores.sort(key=lambda s: s.total_entropy, reverse=True)
    
    return scores


# -----------------------------------------------------------------------------
# Phase Boundary Guardian
# -----------------------------------------------------------------------------

def scan_for_boundary_violations() -> list[BoundaryViolation]:
    """Scan entire repo for Phase I → Phase II forbidden imports."""
    violations = []
    
    for phase_i_dir in PHASE_I_DIRECTORIES:
        dir_path = PROJECT_ROOT / phase_i_dir
        if not dir_path.exists():
            continue
        
        # Handle both file and directory paths
        if dir_path.is_file():
            py_files = [dir_path] if dir_path.suffix == ".py" else []
        else:
            py_files = list(dir_path.rglob("*.py"))
        
        for py_file in py_files:
            if "__pycache__" in str(py_file):
                continue
            
            try:
                content = py_file.read_text(encoding="utf-8", errors="replace")
                lines = content.split("\n")
                
                for line_num, line in enumerate(lines, start=1):
                    for pattern, target_module in PHASE_II_IMPORT_PATTERNS:
                        if re.search(pattern, line):
                            rel_path = py_file.relative_to(PROJECT_ROOT).as_posix()
                            violations.append(BoundaryViolation(
                                source_file=rel_path,
                                line_number=line_num,
                                import_statement=line.strip(),
                                target_module=target_module,
                                severity="error",
                            ))
            except Exception:
                pass
    
    return violations


def generate_remediation_suggestions(violations: list[BoundaryViolation]) -> list[RemediationSuggestion]:
    """Generate remediation suggestions for boundary violations."""
    suggestions = []
    
    for violation in violations:
        # Determine suggestion type based on the violation
        target = violation.target_module
        source = violation.source_file
        
        # Security modules might need to be promoted to Phase I
        if "security" in target or "u2_security" in target:
            suggestions.append(RemediationSuggestion(
                violation=violation,
                suggestion_type="promote",
                description=(
                    f"Consider promoting core security primitives from {target} to Phase I. "
                    f"If SecurityException, DeterministicPRNG are foundational, they should "
                    f"live in backend/security/core.py (Phase I)."
                ),
                code_example=(
                    "# Option 1: Promote to Phase I\n"
                    "# Move SecurityException, DeterministicPRNG to backend/security/core.py\n"
                    "# Then import from there instead\n"
                    "from backend.security.core import SecurityException, DeterministicPRNG"
                ),
            ))
        
        # Analysis modules should use lazy imports
        elif "analysis" in target or "metrics" in target:
            suggestions.append(RemediationSuggestion(
                violation=violation,
                suggestion_type="lazy_import",
                description=(
                    f"Use lazy/conditional import for {target}. "
                    f"Import inside the function that needs it, not at module level."
                ),
                code_example=(
                    "# Option 2: Lazy import\n"
                    "def analyze_u2_data(data):\n"
                    "    # Import only when Phase II analysis is needed\n"
                    f"    from {target} import analyze\n"
                    "    return analyze(data)"
                ),
            ))
        
        # Runner modules should use dependency inversion
        elif "runner" in target:
            suggestions.append(RemediationSuggestion(
                violation=violation,
                suggestion_type="dependency_inversion",
                description=(
                    f"Use dependency inversion pattern. Define an abstract interface "
                    f"in Phase I, implement in Phase II."
                ),
                code_example=(
                    "# Option 3: Dependency inversion\n"
                    "# In Phase I (backend/runner/base.py):\n"
                    "class BaseRunner(Protocol):\n"
                    "    def run(self, config: dict) -> Result: ...\n\n"
                    "# In Phase II (backend/runner/u2_runner.py):\n"
                    "class U2Runner(BaseRunner):\n"
                    "    def run(self, config: dict) -> Result:\n"
                    "        # Phase II implementation"
                ),
            ))
        
        # Generic suggestion for other cases
        else:
            suggestions.append(RemediationSuggestion(
                violation=violation,
                suggestion_type="relocate",
                description=(
                    f"The import of {target} in {source} violates phase boundaries. "
                    f"Consider: (1) promoting the dependency to Phase I, "
                    f"(2) using lazy import, or (3) restructuring the dependency."
                ),
                code_example=None,
            ))
    
    return suggestions


# -----------------------------------------------------------------------------
# Directory Archetype Classifier
# -----------------------------------------------------------------------------

def classify_directory_archetype(directory: Path) -> ArchetypeClassification:
    """Classify a directory into an archetype category."""
    rel_dir = directory.relative_to(PROJECT_ROOT).as_posix()
    
    # Score each archetype
    archetype_scores: dict[str, float] = {}
    
    for archetype, config in DIRECTORY_ARCHETYPES.items():
        score = 0.0
        
        # Check if directory is in the explicit list
        for listed_dir in config["directories"]:
            if rel_dir.startswith(listed_dir) or rel_dir == listed_dir:
                score += 5.0  # Strong match
                break
        
        # Check indicators in directory name
        for indicator in config["indicators"]:
            if indicator in rel_dir.lower():
                score += 2.0
        
        # Check file extensions
        if directory.is_dir():
            try:
                files = list(directory.iterdir())
                extensions = [f.suffix.lower() for f in files if f.is_file()]
                expected_ext = config["expected_extensions"]
                
                if extensions:
                    matching = sum(1 for e in extensions if e in expected_ext)
                    score += (matching / len(extensions)) * 3.0
            except PermissionError:
                pass
        
        archetype_scores[archetype] = score
    
    # Sort by score
    sorted_archetypes = sorted(archetype_scores.items(), key=lambda x: x[1], reverse=True)
    
    primary = sorted_archetypes[0]
    secondary = [(name, score) for name, score in sorted_archetypes[1:4] if score > 0]
    
    # Compute confidence (how dominant is the primary)
    total_score = sum(s for _, s in sorted_archetypes)
    confidence = primary[1] / total_score if total_score > 0 else 0.0
    
    # Check for role violations
    role_violations = []
    primary_config = DIRECTORY_ARCHETYPES.get(primary[0], {})
    
    if directory.is_dir():
        try:
            files = list(directory.iterdir())
            for f in files:
                if f.is_file():
                    ext = f.suffix.lower()
                    expected = primary_config.get("expected_extensions", [])
                    if expected and ext and ext not in expected:
                        role_violations.append(f"Unexpected extension '{ext}' for {primary[0]} archetype: {f.name}")
        except PermissionError:
            pass
    
    # Compute naming consistency
    naming_consistency = 1.0 - (len(role_violations) / max(len(files) if directory.is_dir() else 1, 1))
    
    return ArchetypeClassification(
        directory=rel_dir,
        primary_archetype=primary[0],
        confidence=round(confidence, 4),
        secondary_archetypes=[(name, round(score, 4)) for name, score in secondary],
        role_violations=role_violations[:5],  # Limit to 5
        naming_consistency=round(max(0, naming_consistency), 4),
    )


def classify_all_directories() -> list[ArchetypeClassification]:
    """Classify all directories into archetypes."""
    classifications = []
    
    # Get top-level directories and important subdirectories
    dirs_to_classify = set()
    
    # Add top-level directories
    for item in PROJECT_ROOT.iterdir():
        if item.is_dir() and not item.name.startswith("."):
            if item.name not in ("__pycache__", "node_modules", ".venv"):
                dirs_to_classify.add(item)
    
    # Add important subdirectories
    important_subdirs = [
        "backend/api",
        "backend/axiom_engine",
        "backend/causal",
        "backend/crypto",
        "backend/dag",
        "backend/frontier",
        "backend/governance",
        "backend/ledger",
        "backend/logic",
        "backend/metrics",
        "backend/models",
        "backend/orchestrator",
        "backend/runner",
        "backend/security",
        "backend/telemetry",
        "tests/integration",
        "tests/phase2",
        "docs/architecture",
        "docs/audits",
        "artifacts/phase_ii",
        "artifacts/u2",
    ]
    
    for subdir in important_subdirs:
        path = PROJECT_ROOT / subdir
        if path.exists() and path.is_dir():
            dirs_to_classify.add(path)
    
    for directory in sorted(dirs_to_classify):
        classification = classify_directory_archetype(directory)
        classifications.append(classification)
    
    # Sort by confidence
    classifications.sort(key=lambda c: c.confidence, reverse=True)
    
    return classifications


# -----------------------------------------------------------------------------
# Git History Analysis
# -----------------------------------------------------------------------------

def analyze_git_mutations(max_commits: int = 100) -> list[MutationEvent]:
    """Analyze git history for directory mutation events."""
    mutations = []
    
    try:
        # Get recent commits with file changes
        result = subprocess.run(
            ["git", "log", f"--max-count={max_commits}", "--name-status", "--pretty=format:%H|%ai|%an"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        
        if result.returncode != 0:
            return mutations
        
        lines = result.stdout.strip().split("\n")
        current_commit = None
        current_timestamp = None
        current_author = None
        
        for line in lines:
            if "|" in line and line.count("|") == 2:
                # Commit header line
                parts = line.split("|")
                current_commit = parts[0][:8]  # Short hash
                current_timestamp = parts[1].split()[0]  # Date only
                current_author = parts[2]
            elif line.strip() and current_commit:
                # File change line
                parts = line.split("\t")
                if len(parts) >= 2:
                    status = parts[0]
                    path = parts[-1]
                    
                    # Skip common non-structural changes
                    if "__pycache__" in path or ".pyc" in path:
                        continue
                    
                    # Determine event type
                    if status == "A":
                        event_type = "file_added"
                    elif status == "D":
                        event_type = "file_removed"
                    elif status.startswith("R"):
                        event_type = "file_moved"
                    else:
                        continue  # Skip modifications
                    
                    # Check if this affects directory structure
                    if "/" in path:
                        mutations.append(MutationEvent(
                            commit_hash=current_commit,
                            timestamp=current_timestamp,
                            author=current_author,
                            event_type=event_type,
                            path=path,
                            details=f"Status: {status}",
                        ))
    
    except FileNotFoundError:
        # Git not available
        pass
    except Exception as e:
        pass
    
    # Limit to most recent mutations
    return mutations[:200]


# -----------------------------------------------------------------------------
# Report Generation
# -----------------------------------------------------------------------------

def compute_report_hash(report_data: dict) -> str:
    """Compute deterministic hash of report content."""
    # Remove timestamp and hash for deterministic hashing
    data_for_hash = {
        k: v for k, v in report_data.items() 
        if k not in ("timestamp", "report_hash")
    }
    
    # Sort and serialize deterministically
    json_str = json.dumps(data_for_hash, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(json_str.encode()).hexdigest()[:16]


def generate_report(
    include_guardian: bool = True,
    include_classify: bool = True,
    include_history: bool = False,
    include_remediation: bool = True,
) -> EntropyReport:
    """Generate complete entropy audit report."""
    
    # Compute directory entropies
    directory_scores = compute_all_directory_entropies()
    
    # Compute global entropy
    if directory_scores:
        global_entropy = sum(s.total_entropy for s in directory_scores) / len(directory_scores)
    else:
        global_entropy = 0.0
    
    # Scan for boundary violations
    boundary_violations = scan_for_boundary_violations() if include_guardian else []
    
    # Generate remediation suggestions
    remediation_suggestions = (
        generate_remediation_suggestions(boundary_violations) 
        if include_remediation and boundary_violations 
        else []
    )
    
    # Classify directories
    archetype_classifications = classify_all_directories() if include_classify else []
    
    # Analyze git history
    mutation_events = analyze_git_mutations() if include_history else []
    
    # Count totals
    total_dirs = len(directory_scores)
    total_files = sum(s.file_count for s in directory_scores)
    
    # Build summary
    summary = {
        "high_entropy_directories": [
            s.directory for s in directory_scores[:5] if s.total_entropy > 1.0
        ],
        "boundary_violation_count": len(boundary_violations),
        "affected_phase_i_files": list(set(v.source_file for v in boundary_violations)),
        "archetype_distribution": Counter(c.primary_archetype for c in archetype_classifications),
        "low_confidence_classifications": [
            c.directory for c in archetype_classifications if c.confidence < 0.3
        ],
        "total_violations": sum(s.violation_count for s in directory_scores),
        "mutation_summary": {
            "file_added": sum(1 for m in mutation_events if m.event_type == "file_added"),
            "file_removed": sum(1 for m in mutation_events if m.event_type == "file_removed"),
            "file_moved": sum(1 for m in mutation_events if m.event_type == "file_moved"),
        } if mutation_events else {},
    }
    
    report = EntropyReport(
        timestamp=datetime.utcnow().isoformat() + "Z",
        project_root=str(PROJECT_ROOT),
        report_hash="",  # Will be computed
        total_directories=total_dirs,
        total_files=total_files,
        global_entropy=round(global_entropy, 4),
        directory_scores=directory_scores,
        boundary_violations=boundary_violations,
        remediation_suggestions=remediation_suggestions,
        archetype_classifications=archetype_classifications,
        mutation_events=mutation_events,
        summary=summary,
    )
    
    # Compute report hash
    report_dict = report.to_dict()
    report.report_hash = compute_report_hash(report_dict)
    
    return report


# -----------------------------------------------------------------------------
# Structure Risk Index
# -----------------------------------------------------------------------------

def classify_directory_risk(score: EntropyScore) -> DirectoryRisk:
    """Classify a directory's risk level based on entropy and violations."""
    contributing_factors = []
    
    # Compute risk components
    entropy_risk = 0.0
    if score.total_entropy >= RISK_THRESHOLDS["entropy_high"]:
        entropy_risk = 1.0
        contributing_factors.append(f"High entropy ({score.total_entropy:.2f})")
    elif score.total_entropy >= RISK_THRESHOLDS["entropy_medium"]:
        entropy_risk = 0.5
        contributing_factors.append(f"Medium entropy ({score.total_entropy:.2f})")
    else:
        entropy_risk = score.total_entropy / RISK_THRESHOLDS["entropy_medium"]
    
    violation_risk = 0.0
    if score.violation_count >= RISK_THRESHOLDS["violations_high"]:
        violation_risk = 1.0
        contributing_factors.append(f"High violations ({score.violation_count})")
    elif score.violation_count >= RISK_THRESHOLDS["violations_medium"]:
        violation_risk = 0.5
        contributing_factors.append(f"Medium violations ({score.violation_count})")
    elif score.violation_count > 0:
        violation_risk = score.violation_count / RISK_THRESHOLDS["violations_medium"]
        contributing_factors.append(f"Minor violations ({score.violation_count})")
    
    # Add specific violation types
    if score.forbidden_files:
        contributing_factors.append(f"Forbidden files: {len(score.forbidden_files)}")
    if score.unexpected_files:
        contributing_factors.append(f"Unexpected files: {len(score.unexpected_files)}")
    
    # Weighted combination
    risk_score = entropy_risk * 0.6 + violation_risk * 0.4
    risk_score = min(1.0, max(0.0, risk_score))
    
    # Classify risk level
    if risk_score >= 0.7:
        risk_level = "HIGH"
    elif risk_score >= 0.3:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"
    
    return DirectoryRisk(
        path=score.directory,
        risk_score=round(risk_score, 4),
        risk_level=risk_level,
        entropy=score.total_entropy,
        violations=score.violation_count,
        contributing_factors=contributing_factors,
    )


def compute_structure_risk_index(
    report: dict | EntropyReport,
    phase_filter: str | None = None,
) -> StructureRiskIndex:
    """
    Compute Structure Risk Index from an entropy report.
    
    Args:
        report: Either a dict (JSON report) or EntropyReport object
        phase_filter: Optional "phase1" or "phase2" to filter directories
    
    Returns:
        StructureRiskIndex with global risk score and ranked directories
    """
    # Handle both dict and EntropyReport inputs
    if isinstance(report, dict):
        directory_scores = [
            EntropyScore(**s) for s in report.get("directory_scores", [])
        ]
        boundary_violations = report.get("boundary_violations", [])
    else:
        directory_scores = report.directory_scores
        boundary_violations = [v.to_dict() for v in report.boundary_violations]
    
    # Apply phase filter
    if phase_filter:
        directory_scores = filter_directories_by_phase(directory_scores, phase_filter)
    
    # Compute risk for each directory
    directory_risks = []
    for score in directory_scores:
        risk = classify_directory_risk(score)
        directory_risks.append(risk)
    
    # Sort by risk score (highest first)
    directory_risks.sort(key=lambda r: r.risk_score, reverse=True)
    
    # Compute global risk score
    if directory_risks:
        # Weighted average with emphasis on high-risk directories
        weights = [r.risk_score ** 2 for r in directory_risks]  # Square for emphasis
        total_weight = sum(weights) if weights else 1
        global_risk = sum(r.risk_score * w for r, w in zip(directory_risks, weights)) / total_weight
        
        # Add penalty for boundary violations
        violation_penalty = len(boundary_violations) * 0.05
        global_risk = min(1.0, global_risk + violation_penalty)
    else:
        global_risk = 0.0
    
    # Classify global risk level
    if global_risk >= RISK_THRESHOLDS["global_risk_high"]:
        global_risk_level = "HIGH"
    elif global_risk >= RISK_THRESHOLDS["global_risk_medium"]:
        global_risk_level = "MEDIUM"
    else:
        global_risk_level = "LOW"
    
    # Compute risk distribution
    risk_distribution = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
    for risk in directory_risks:
        risk_distribution[risk.risk_level] += 1
    
    # Identify drift indicators
    drift_indicators = []
    if risk_distribution["HIGH"] > 0:
        drift_indicators.append(f"{risk_distribution['HIGH']} HIGH-risk directories detected")
    if boundary_violations:
        drift_indicators.append(f"{len(boundary_violations)} phase boundary violation(s)")
    
    high_entropy_count = sum(1 for r in directory_risks if r.entropy > RISK_THRESHOLDS["entropy_high"])
    if high_entropy_count > 3:
        drift_indicators.append(f"{high_entropy_count} directories with concerning entropy")
    
    return StructureRiskIndex(
        timestamp=datetime.utcnow().isoformat() + "Z",
        global_risk_score=round(global_risk, 4),
        global_risk_level=global_risk_level,
        top_risky_directories=directory_risks[:15],  # Top 15
        phase_filter=phase_filter,
        total_directories_analyzed=len(directory_risks),
        risk_distribution=risk_distribution,
        drift_indicators=drift_indicators,
    )


# -----------------------------------------------------------------------------
# Phase Filtering
# -----------------------------------------------------------------------------

def is_phase_i_directory(path: str) -> bool:
    """Check if a directory path belongs to Phase I."""
    for phase_i_dir in PHASE_I_DIRECTORIES:
        if path.startswith(phase_i_dir) or path == phase_i_dir:
            return True
    # Also check common Phase I directories not in explicit list
    phase_i_prefixes = [
        "curriculum", "derivation", "attestation", "rfl",
        "backend/axiom", "backend/basis", "backend/bridge",
        "backend/consensus", "backend/crypto", "backend/dag",
        "backend/fol_eq", "backend/frontier", "backend/generator",
        "backend/governance", "backend/ht", "backend/integration",
        "backend/ledger", "backend/logic", "backend/models",
        "backend/orchestrator", "backend/phase_ix", "backend/repro",
        "backend/rfl", "backend/testing", "backend/tools",
        "backend/verification",
    ]
    for prefix in phase_i_prefixes:
        if path.startswith(prefix):
            return True
    return False


def is_phase_ii_directory(path: str) -> bool:
    """Check if a directory path belongs to Phase II."""
    for phase_ii_dir in PHASE_II_DIRECTORIES:
        if path.startswith(phase_ii_dir) or path == phase_ii_dir:
            return True
    # Also check for u2_ prefix patterns
    if "/u2_" in path or path.startswith("u2_"):
        return True
    if "phase2" in path.lower() or "phase_ii" in path.lower():
        return True
    return False


def filter_directories_by_phase(
    scores: list[EntropyScore],
    phase: str,
) -> list[EntropyScore]:
    """
    Filter directory scores to a specific phase.
    
    Args:
        scores: List of EntropyScore objects
        phase: "phase1" or "phase2"
    
    Returns:
        Filtered list of EntropyScore objects
    """
    filtered = []
    
    for score in scores:
        path = score.directory
        
        if phase == "phase1":
            if is_phase_i_directory(path):
                filtered.append(score)
        elif phase == "phase2":
            if is_phase_ii_directory(path):
                filtered.append(score)
    
    return filtered


def compute_phase_entropy(phase: str) -> tuple[list[EntropyScore], float]:
    """
    Compute entropy scores for a specific phase.
    
    Args:
        phase: "phase1" or "phase2"
    
    Returns:
        Tuple of (filtered scores, average entropy)
    """
    all_scores = compute_all_directory_entropies()
    filtered_scores = filter_directories_by_phase(all_scores, phase)
    
    if filtered_scores:
        avg_entropy = sum(s.total_entropy for s in filtered_scores) / len(filtered_scores)
    else:
        avg_entropy = 0.0
    
    return filtered_scores, avg_entropy


# -----------------------------------------------------------------------------
# Structural Drift Detection
# -----------------------------------------------------------------------------

def load_report_from_file(report_path: str | Path) -> dict:
    """Load a JSON report from file."""
    path = Path(report_path)
    if not path.exists():
        raise FileNotFoundError(f"Report not found: {report_path}")
    
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compare_structure_risk(
    old_report_path: str | Path,
    new_report_path: str | Path,
) -> StructureDriftReport:
    """
    Compute drift in structure risk between two runs.
    
    Args:
        old_report_path: Path to the baseline/old report JSON
        new_report_path: Path to the new/current report JSON
    
    Returns:
        StructureDriftReport with per-directory and global drift metrics
    """
    # Load reports
    old_report = load_report_from_file(old_report_path)
    new_report = load_report_from_file(new_report_path)
    
    # Build directory maps
    old_scores = {s["directory"]: s for s in old_report.get("directory_scores", [])}
    new_scores = {s["directory"]: s for s in new_report.get("directory_scores", [])}
    
    # Compute risk indices for both
    old_risk_index = compute_structure_risk_index(old_report)
    new_risk_index = compute_structure_risk_index(new_report)
    
    # Build risk score maps
    old_risks = {d.path: d for d in old_risk_index.top_risky_directories}
    new_risks = {d.path: d for d in new_risk_index.top_risky_directories}
    
    # For directories not in top list, compute risk from scores
    for path, score in old_scores.items():
        if path not in old_risks:
            entropy_score = EntropyScore(**score)
            old_risks[path] = classify_directory_risk(entropy_score)
    
    for path, score in new_scores.items():
        if path not in new_risks:
            entropy_score = EntropyScore(**score)
            new_risks[path] = classify_directory_risk(entropy_score)
    
    # Compute drift for each directory
    all_directories = set(old_scores.keys()) | set(new_scores.keys())
    directory_drifts = []
    
    max_risk_increase = 0.0
    max_risk_decrease = 0.0
    increased_risk_dirs = []
    decreased_risk_dirs = []
    new_directories = []
    removed_directories = []
    
    # Track risk level transitions
    risk_transitions: dict[str, list[str]] = {
        "LOW_TO_MEDIUM": [],
        "LOW_TO_HIGH": [],
        "MEDIUM_TO_HIGH": [],
        "MEDIUM_TO_LOW": [],
        "HIGH_TO_MEDIUM": [],
        "HIGH_TO_LOW": [],
    }
    
    for path in sorted(all_directories):
        is_new = path not in old_scores
        is_removed = path not in new_scores
        
        if is_new:
            new_directories.append(path)
            # New directory - compare against zero baseline
            new_score = new_scores[path]
            new_risk = new_risks.get(path)
            drift = DirectoryDrift(
                path=path,
                entropy_old=0.0,
                entropy_new=new_score["total_entropy"],
                entropy_delta=new_score["total_entropy"],
                violations_old=0,
                violations_new=new_score["violation_count"],
                violations_delta=new_score["violation_count"],
                risk_score_old=0.0,
                risk_score_new=new_risk.risk_score if new_risk else 0.0,
                risk_delta=new_risk.risk_score if new_risk else 0.0,
                risk_level_old="N/A",
                risk_level_new=new_risk.risk_level if new_risk else "LOW",
                is_new_directory=True,
                is_removed_directory=False,
            )
        elif is_removed:
            removed_directories.append(path)
            # Removed directory
            old_score = old_scores[path]
            old_risk = old_risks.get(path)
            drift = DirectoryDrift(
                path=path,
                entropy_old=old_score["total_entropy"],
                entropy_new=0.0,
                entropy_delta=-old_score["total_entropy"],
                violations_old=old_score["violation_count"],
                violations_new=0,
                violations_delta=-old_score["violation_count"],
                risk_score_old=old_risk.risk_score if old_risk else 0.0,
                risk_score_new=0.0,
                risk_delta=-(old_risk.risk_score if old_risk else 0.0),
                risk_level_old=old_risk.risk_level if old_risk else "LOW",
                risk_level_new="N/A",
                is_new_directory=False,
                is_removed_directory=True,
            )
        else:
            # Existing directory - compute delta
            old_score = old_scores[path]
            new_score = new_scores[path]
            old_risk = old_risks.get(path)
            new_risk = new_risks.get(path)
            
            risk_old = old_risk.risk_score if old_risk else 0.0
            risk_new = new_risk.risk_score if new_risk else 0.0
            risk_delta = risk_new - risk_old
            
            drift = DirectoryDrift(
                path=path,
                entropy_old=old_score["total_entropy"],
                entropy_new=new_score["total_entropy"],
                entropy_delta=round(new_score["total_entropy"] - old_score["total_entropy"], 4),
                violations_old=old_score["violation_count"],
                violations_new=new_score["violation_count"],
                violations_delta=new_score["violation_count"] - old_score["violation_count"],
                risk_score_old=risk_old,
                risk_score_new=risk_new,
                risk_delta=round(risk_delta, 4),
                risk_level_old=old_risk.risk_level if old_risk else "LOW",
                risk_level_new=new_risk.risk_level if new_risk else "LOW",
                is_new_directory=False,
                is_removed_directory=False,
            )
            
            # Track increases/decreases
            if risk_delta > 0.01:
                increased_risk_dirs.append(path)
                if risk_delta > max_risk_increase:
                    max_risk_increase = risk_delta
            elif risk_delta < -0.01:
                decreased_risk_dirs.append(path)
                if risk_delta < max_risk_decrease:
                    max_risk_decrease = risk_delta
            
            # Track risk level transitions
            level_old = drift.risk_level_old
            level_new = drift.risk_level_new
            if level_old != level_new and level_old != "N/A" and level_new != "N/A":
                transition_key = f"{level_old}_TO_{level_new}"
                if transition_key in risk_transitions:
                    risk_transitions[transition_key].append(path)
        
        directory_drifts.append(drift)
    
    # Sort by absolute risk delta
    directory_drifts.sort(key=lambda d: abs(d.risk_delta), reverse=True)
    
    # Compute global delta
    global_risk_delta = round(
        new_risk_index.global_risk_score - old_risk_index.global_risk_score, 4
    )
    
    # Build summary
    summary = {
        "total_directories_compared": len(all_directories),
        "directories_improved": len(decreased_risk_dirs),
        "directories_degraded": len(increased_risk_dirs),
        "directories_stable": len(all_directories) - len(increased_risk_dirs) - len(decreased_risk_dirs) - len(new_directories) - len(removed_directories),
        "new_directories_count": len(new_directories),
        "removed_directories_count": len(removed_directories),
        "overall_trend": "improving" if global_risk_delta < -0.02 else ("degrading" if global_risk_delta > 0.02 else "stable"),
    }
    
    # Sort all directory lists for deterministic output
    return StructureDriftReport(
        timestamp=datetime.utcnow().isoformat() + "Z",
        old_report_path=str(old_report_path),
        new_report_path=str(new_report_path),
        global_risk_old=old_risk_index.global_risk_score,
        global_risk_new=new_risk_index.global_risk_score,
        global_risk_delta=global_risk_delta,
        max_risk_increase=round(max_risk_increase, 4),
        max_risk_decrease=round(max_risk_decrease, 4),
        directories_with_increased_risk=sorted(increased_risk_dirs),
        directories_with_decreased_risk=sorted(decreased_risk_dirs),
        new_directories=sorted(new_directories),
        removed_directories=sorted(removed_directories),
        directory_drifts=directory_drifts[:50],  # Top 50, already sorted by risk delta
        summary=summary,
        risk_transitions={k: sorted(v) for k, v in risk_transitions.items() if v},
    )


# -----------------------------------------------------------------------------
# Refactor Candidate Generation
# -----------------------------------------------------------------------------

def generate_refactor_candidates(
    risk_index: StructureRiskIndex,
    drift_report: StructureDriftReport | None = None,
    top_k: int = 10,
) -> list[RefactorCandidate]:
    """
    Generate a shortlist of directories that are candidates for refactoring.
    
    Criteria:
    - Have HIGH risk
    - Are Phase II
    - Have had risk increased recently (if drift data available)
    
    Args:
        risk_index: Current structure risk index
        drift_report: Optional drift report for recent changes
        top_k: Maximum number of candidates to return
    
    Returns:
        List of RefactorCandidate objects, sorted by priority
    """
    # Build drift lookup
    drift_lookup = {}
    if drift_report:
        for drift in drift_report.directory_drifts:
            drift_lookup[drift.path] = drift
    
    candidates = []
    
    for dir_risk in risk_index.top_risky_directories:
        reasons = []
        priority_score = 0.0
        
        # Check if HIGH risk
        is_high_risk = dir_risk.risk_level == "HIGH"
        if is_high_risk:
            reasons.append("HIGH risk level")
            priority_score += 3.0
        elif dir_risk.risk_level == "MEDIUM" and dir_risk.risk_score >= 0.5:
            reasons.append("Elevated MEDIUM risk")
            priority_score += 1.5
        
        # Check if Phase II
        is_phase_ii = is_phase_ii_directory(dir_risk.path)
        if is_phase_ii:
            reasons.append("Phase II directory")
            priority_score += 1.0
        
        # Check if risk increased recently
        risk_increased = False
        risk_delta = None
        if dir_risk.path in drift_lookup:
            drift = drift_lookup[dir_risk.path]
            risk_delta = drift.risk_delta
            if drift.risk_delta > 0.05:
                risk_increased = True
                reasons.append(f"Risk increased by {drift.risk_delta:.2%}")
                priority_score += 2.0
            elif drift.is_new_directory:
                reasons.append("Newly added directory")
                priority_score += 0.5
        
        # Determine priority level
        if priority_score >= 4.0:
            priority = "CRITICAL"
        elif priority_score >= 2.5:
            priority = "HIGH"
        else:
            priority = "MEDIUM"
        
        # Only include if it has at least one reason
        if reasons and (is_high_risk or is_phase_ii or risk_increased):
            candidates.append(RefactorCandidate(
                path=dir_risk.path,
                risk_score=dir_risk.risk_score,
                risk_level=dir_risk.risk_level,
                entropy=dir_risk.entropy,
                violations=dir_risk.violations,
                is_phase_ii=is_phase_ii,
                risk_increased_recently=risk_increased,
                risk_delta=risk_delta,
                priority=priority,
                reasons=reasons,
            ))
    
    # Sort by priority then risk score
    priority_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2}
    candidates.sort(key=lambda c: (priority_order.get(c.priority, 3), -c.risk_score))
    
    return candidates[:top_k]


def write_refactor_candidates(
    candidates: list[RefactorCandidate],
    output_path: Path | None = None,
) -> Path:
    """
    Write refactor candidates to JSON file.
    
    Args:
        candidates: List of RefactorCandidate objects
        output_path: Output path (default: artifacts/structure/refactor_candidates.json)
    
    Returns:
        Path to the written file
    """
    if output_path is None:
        output_path = PROJECT_ROOT / "artifacts" / "structure" / "refactor_candidates.json"
    
    # Ensure directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "total_candidates": len(candidates),
        "candidates": [c.to_dict() for c in candidates],
        "summary": {
            "critical_count": sum(1 for c in candidates if c.priority == "CRITICAL"),
            "high_count": sum(1 for c in candidates if c.priority == "HIGH"),
            "medium_count": sum(1 for c in candidates if c.priority == "MEDIUM"),
            "phase_ii_count": sum(1 for c in candidates if c.is_phase_ii),
        },
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    
    return output_path


# -----------------------------------------------------------------------------
# Zone-Based Structural Grouping
# -----------------------------------------------------------------------------

def extract_zone(path: str) -> str:
    """
    Extract the zone (top-level directory) from a path.
    
    Args:
        path: Directory path (e.g., "backend/api/v1")
    
    Returns:
        Zone name (e.g., "backend"), or "root" if no prefix
    """
    # Normalize path separators
    normalized = path.replace("\\", "/")
    parts = normalized.split("/")
    
    # Return first non-empty part, or "root" if at project root
    for part in parts:
        if part and part != ".":
            return part
    return "root"


def group_directories_into_zones(
    report: EntropyReport | StructureRiskIndex | dict,
) -> dict[str, Any]:
    """
    Group directories into higher-level zones and aggregate risk counts.
    
    Zones are determined by the top-level path prefix before '/'.
    
    Args:
        report: EntropyReport, StructureRiskIndex, or dict with directory data
    
    Returns:
        Dict with zone-aggregated risk counts:
        {
          "zones": {
            "backend": {"high": 2, "medium": 5, "low": 10},
            "tests": {"high": 1, "medium": 4, "low": 12}
          },
          "total_zones": int,
          "zone_list": ["backend", "docs", "tests", ...]
        }
    """
    # Extract directory risk data
    if isinstance(report, dict):
        # Handle dict from JSON
        if "top_risky_directories" in report:
            # StructureRiskIndex format
            directories = [
                {"path": d["path"], "risk_level": d["risk_level"]}
                for d in report.get("top_risky_directories", [])
            ]
        elif "directory_scores" in report:
            # EntropyReport format - compute risk levels
            directories = []
            for score in report.get("directory_scores", []):
                # Classify based on entropy and violations
                entropy = score.get("total_entropy", 0)
                violations = score.get("violation_count", 0)
                if entropy >= RISK_THRESHOLDS["entropy_high"] or violations >= RISK_THRESHOLDS["violations_high"]:
                    level = "HIGH"
                elif entropy >= RISK_THRESHOLDS["entropy_medium"] or violations >= RISK_THRESHOLDS["violations_medium"]:
                    level = "MEDIUM"
                else:
                    level = "LOW"
                directories.append({"path": score["directory"], "risk_level": level})
        else:
            directories = []
    elif isinstance(report, StructureRiskIndex):
        directories = [
            {"path": d.path, "risk_level": d.risk_level}
            for d in report.top_risky_directories
        ]
    elif isinstance(report, EntropyReport):
        directories = []
        for score in report.directory_scores:
            risk = classify_directory_risk(score)
            directories.append({"path": score.directory, "risk_level": risk.risk_level})
    else:
        directories = []
    
    # Group by zone
    zones: dict[str, dict[str, int]] = {}
    
    for d in directories:
        zone = extract_zone(d["path"])
        if zone not in zones:
            zones[zone] = {"high": 0, "medium": 0, "low": 0}
        
        level = d["risk_level"].lower()
        if level in zones[zone]:
            zones[zone][level] += 1
    
    # Sort zones for deterministic output
    sorted_zones = {k: zones[k] for k in sorted(zones.keys())}
    
    return {
        "zones": sorted_zones,
        "total_zones": len(sorted_zones),
        "zone_list": sorted(zones.keys()),
    }


# -----------------------------------------------------------------------------
# Refactor Candidate Bundles
# -----------------------------------------------------------------------------

def build_refactor_bundles(
    candidates: list[RefactorCandidate] | list[dict],
) -> list[dict[str, Any]]:
    """
    Group refactor candidates into bundles that make sense to address together.
    
    Bundling heuristic: Group by zone + risk band.
    
    Args:
        candidates: List of RefactorCandidate objects or dicts
    
    Returns:
        List of bundle dicts:
        [
          {
            "bundle_id": "zone:backend:HIGH",
            "directories": ["backend/api", "backend/worker"],
            "dominant_risk_level": "HIGH",
            "directory_count": 2
          }
        ]
    """
    # Normalize to dicts
    if candidates and isinstance(candidates[0], RefactorCandidate):
        candidate_list = [c.to_dict() for c in candidates]
    else:
        candidate_list = list(candidates) if candidates else []
    
    # Group by zone + risk level
    bundle_map: dict[str, list[str]] = {}
    
    for c in candidate_list:
        path = c.get("path", "")
        risk_level = c.get("risk_level", "LOW")
        zone = extract_zone(path)
        
        bundle_key = f"zone:{zone}:{risk_level}"
        if bundle_key not in bundle_map:
            bundle_map[bundle_key] = []
        bundle_map[bundle_key].append(path)
    
    # Build bundle list
    bundles = []
    for bundle_id, directories in sorted(bundle_map.items()):
        # Parse risk level from bundle_id
        parts = bundle_id.split(":")
        risk_level = parts[2] if len(parts) > 2 else "LOW"
        
        bundles.append({
            "bundle_id": bundle_id,
            "directories": sorted(directories),  # Deterministic ordering
            "dominant_risk_level": risk_level,
            "directory_count": len(directories),
        })
    
    # Sort bundles: HIGH first, then MEDIUM, then LOW, then by zone name
    risk_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    bundles.sort(key=lambda b: (risk_order.get(b["dominant_risk_level"], 3), b["bundle_id"]))
    
    return bundles


# -----------------------------------------------------------------------------
# Structural Posture Summary
# -----------------------------------------------------------------------------

def build_structural_posture(
    report: StructureDriftReport | StructureRiskIndex | dict,
) -> dict[str, Any]:
    """
    Build a minimal JSON for governance describing overall structural posture.
    
    This provides neutral, count-based data without evaluative language.
    
    Args:
        report: StructureDriftReport, StructureRiskIndex, or dict
    
    Returns:
        Dict with structural posture:
        {
          "schema_version": "1.0.0",
          "high_risk_directories": int,
          "medium_risk_directories": int,
          "low_risk_directories": int,
          "new_directories": int,
          "removed_directories": int
        }
    """
    posture = {
        "schema_version": "1.0.0",
        "high_risk_directories": 0,
        "medium_risk_directories": 0,
        "low_risk_directories": 0,
        "new_directories": 0,
        "removed_directories": 0,
    }
    
    if isinstance(report, dict):
        # Handle StructureDriftReport dict
        if "risk_transitions" in report or "new_directories" in report:
            posture["new_directories"] = len(report.get("new_directories", []))
            posture["removed_directories"] = len(report.get("removed_directories", []))
            
            # Count risk levels from directory_drifts if available
            for drift in report.get("directory_drifts", []):
                level = drift.get("risk_level_new", "LOW")
                if level == "HIGH":
                    posture["high_risk_directories"] += 1
                elif level == "MEDIUM":
                    posture["medium_risk_directories"] += 1
                elif level != "N/A":
                    posture["low_risk_directories"] += 1
        
        # Handle StructureRiskIndex dict
        elif "risk_distribution" in report:
            dist = report.get("risk_distribution", {})
            posture["high_risk_directories"] = dist.get("HIGH", 0)
            posture["medium_risk_directories"] = dist.get("MEDIUM", 0)
            posture["low_risk_directories"] = dist.get("LOW", 0)
    
    elif isinstance(report, StructureDriftReport):
        posture["new_directories"] = len(report.new_directories)
        posture["removed_directories"] = len(report.removed_directories)
        
        # Count from directory drifts
        for drift in report.directory_drifts:
            level = drift.risk_level_new
            if level == "HIGH":
                posture["high_risk_directories"] += 1
            elif level == "MEDIUM":
                posture["medium_risk_directories"] += 1
            elif level != "N/A":
                posture["low_risk_directories"] += 1
    
    elif isinstance(report, StructureRiskIndex):
        posture["high_risk_directories"] = report.risk_distribution.get("HIGH", 0)
        posture["medium_risk_directories"] = report.risk_distribution.get("MEDIUM", 0)
        posture["low_risk_directories"] = report.risk_distribution.get("LOW", 0)
    
    return posture


def write_structural_posture(
    posture: dict[str, Any],
    output_path: Path | None = None,
) -> Path:
    """
    Write structural posture to JSON file.
    
    Args:
        posture: Structural posture dict from build_structural_posture()
        output_path: Output path (default: artifacts/structure/structural_posture.json)
    
    Returns:
        Path to the written file
    """
    if output_path is None:
        output_path = PROJECT_ROOT / "artifacts" / "structure" / "structural_posture.json"
    
    # Ensure directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Add timestamp
    output_data = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        **posture,
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    
    return output_path


# -----------------------------------------------------------------------------
# Phase IV — Structure-Aware Release Gate & Refactor Planning
# -----------------------------------------------------------------------------

def evaluate_structure_for_release(
    drift_report: StructureDriftReport | dict,
) -> dict[str, Any]:
    """
    Evaluate structural health for release gating decisions.
    
    PHASE IV — STRUCTURE-AWARE RELEASE GATE
    No uplift claims are made.
    Deterministic execution guaranteed.
    
    Analyzes structural drift to determine if release should proceed,
    be warned, or be blocked based on risk in core vs peripheral zones.
    
    Args:
        drift_report: StructureDriftReport or dict from compare_structure_risk()
    
    Returns:
        Release evaluation dictionary:
        {
          "release_ok": bool,
          "status": "OK" | "WARN" | "BLOCK",
          "refactor_zones": List[str],  # Directory prefixes needing attention
          "reasons": List[str]  # Neutral, concise explanation strings
        }
    """
    # Normalize to dict
    if isinstance(drift_report, StructureDriftReport):
        drift_dict = drift_report.to_dict()
    else:
        drift_dict = drift_report
    
    # Define core zones (critical for release)
    core_zones = {"backend", "experiments", "tests", "scripts"}
    
    # Define peripheral zones (less critical)
    peripheral_zones = {"docs", "ui", "artifacts", "config", "infra"}
    
    # Extract directory drifts
    directory_drifts = drift_dict.get("directory_drifts", [])
    
    # Analyze risk by zone
    core_high_risk: list[str] = []
    core_medium_risk: list[str] = []
    peripheral_high_risk: list[str] = []
    peripheral_medium_risk: list[str] = []
    
    for drift in directory_drifts:
        path = drift.get("path", "")
        risk_level = drift.get("risk_level_new", "LOW")
        zone = extract_zone(path)
        
        if zone in core_zones:
            if risk_level == "HIGH":
                core_high_risk.append(path)
            elif risk_level == "MEDIUM":
                core_medium_risk.append(path)
        elif zone in peripheral_zones:
            if risk_level == "HIGH":
                peripheral_high_risk.append(path)
            elif risk_level == "MEDIUM":
                peripheral_medium_risk.append(path)
    
    # Determine status
    reasons: list[str] = []
    refactor_zones: set[str] = set()
    
    # BLOCK: HIGH risk in core zones
    if core_high_risk:
        status = "BLOCK"
        release_ok = False
        reasons.append(f"{len(core_high_risk)} directory(ies) in core zones have HIGH risk level.")
        for path in core_high_risk:
            refactor_zones.add(extract_zone(path))
    # WARN: MEDIUM risk in core zones OR HIGH risk in peripheral zones
    elif core_medium_risk or peripheral_high_risk:
        status = "WARN"
        release_ok = True
        if core_medium_risk:
            reasons.append(f"{len(core_medium_risk)} directory(ies) in core zones have MEDIUM risk level.")
            for path in core_medium_risk:
                refactor_zones.add(extract_zone(path))
        if peripheral_high_risk:
            reasons.append(f"{len(peripheral_high_risk)} directory(ies) in peripheral zones have HIGH risk level.")
            for path in peripheral_high_risk:
                refactor_zones.add(extract_zone(path))
    # OK: No elevated risk in critical areas
    else:
        status = "OK"
        release_ok = True
        reasons.append("No elevated structural risk detected in core zones.")
    
    return {
        "release_ok": release_ok,
        "status": status,
        "refactor_zones": sorted(refactor_zones),  # Deterministic ordering
        "reasons": reasons,
    }


def build_structural_refactor_plan(
    drift_report: StructureDriftReport | dict,
    release_eval: dict[str, Any],
) -> dict[str, Any]:
    """
    Build a refactor planning view from drift analysis and release evaluation.
    
    PHASE IV — REFACTOR PLANNING LENS
    No uplift claims are made.
    Deterministic execution guaranteed.
    
    Produces a structured view of refactoring priorities and batching
    suggestions for planning purposes.
    
    Args:
        drift_report: StructureDriftReport or dict
        release_eval: Release evaluation from evaluate_structure_for_release()
    
    Returns:
        Refactor plan dictionary:
        {
          "priority_zones": List[str],  # Ordered by structural concern
          "suggested_batching": List[Dict[str, Any]],  # Groups of directories
          "notes": List[str]  # Neutral planning text
        }
    """
    # Normalize drift report
    if isinstance(drift_report, StructureDriftReport):
        drift_dict = drift_report.to_dict()
    else:
        drift_dict = drift_report
    
    # Extract directory drifts and sort by risk
    directory_drifts = drift_dict.get("directory_drifts", [])
    
    # Group by zone and risk level
    zone_risk_map: dict[str, dict[str, list[str]]] = {}
    
    for drift in directory_drifts:
        path = drift.get("path", "")
        risk_level = drift.get("risk_level_new", "LOW")
        zone = extract_zone(path)
        
        if zone not in zone_risk_map:
            zone_risk_map[zone] = {"HIGH": [], "MEDIUM": [], "LOW": []}
        
        if risk_level in zone_risk_map[zone]:
            zone_risk_map[zone][risk_level].append(path)
    
    # Build priority zones (sorted by risk severity)
    priority_zones: list[str] = []
    risk_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    
    # Sort zones by highest risk first
    zone_priorities: list[tuple[str, int]] = []
    for zone, risks in zone_risk_map.items():
        # Find highest risk level in zone
        max_priority = 3
        if risks["HIGH"]:
            max_priority = 0
        elif risks["MEDIUM"]:
            max_priority = 1
        elif risks["LOW"]:
            max_priority = 2
        
        zone_priorities.append((zone, max_priority))
    
    zone_priorities.sort(key=lambda x: (x[1], x[0]))  # Sort by priority, then alphabetically
    priority_zones = [zone for zone, _ in zone_priorities]
    
    # Build suggested batching (group by zone + risk level)
    suggested_batching: list[dict[str, Any]] = []
    
    for zone in priority_zones:
        risks = zone_risk_map.get(zone, {})
        
        # Create batches by risk level within zone
        for risk_level in ["HIGH", "MEDIUM", "LOW"]:
            dirs = sorted(risks.get(risk_level, []))  # Deterministic ordering
            
            if dirs:
                suggested_batching.append({
                    "batch_id": f"{zone}:{risk_level}",
                    "zone": zone,
                    "risk_level": risk_level,
                    "directories": dirs,
                    "directory_count": len(dirs),
                })
    
    # Build neutral planning notes
    notes: list[str] = []
    
    refactor_zones = release_eval.get("refactor_zones", [])
    if refactor_zones:
        notes.append(f"Zones identified for structural attention: {', '.join(refactor_zones)}.")
    
    high_risk_count = sum(
        len(risks.get("HIGH", []))
        for risks in zone_risk_map.values()
    )
    if high_risk_count > 0:
        notes.append(f"{high_risk_count} directory(ies) currently at HIGH risk level.")
    
    medium_risk_count = sum(
        len(risks.get("MEDIUM", []))
        for risks in zone_risk_map.values()
    )
    if medium_risk_count > 0:
        notes.append(f"{medium_risk_count} directory(ies) currently at MEDIUM risk level.")
    
    if not notes:
        notes.append("No structural refactoring priorities identified at this time.")
    
    return {
        "priority_zones": priority_zones,
        "suggested_batching": suggested_batching,
        "notes": notes,
    }


def build_structure_director_panel(
    release_eval: dict[str, Any],
    refactor_plan: dict[str, Any],
) -> dict[str, Any]:
    """
    Build Director-level structural health dashboard panel.
    
    PHASE IV — DIRECTOR STRUCTURE PANEL
    No uplift claims are made.
    Deterministic execution guaranteed.
    
    Produces a high-level executive summary of structural health
    suitable for Director/MAAS integration.
    
    Args:
        release_eval: Release evaluation from evaluate_structure_for_release()
        refactor_plan: Refactor plan from build_structural_refactor_plan()
    
    Returns:
        Director panel dictionary:
        {
          "status_light": "GREEN" | "YELLOW" | "RED",
          "high_risk_zones": List[str],
          "headline": str  # Short neutral sentence
        }
    """
    # Map release status to traffic light
    status = release_eval.get("status", "OK")
    if status == "BLOCK":
        status_light = "RED"
    elif status == "WARN":
        status_light = "YELLOW"
    else:
        status_light = "GREEN"
    
    # Extract high-risk zones from refactor plan
    high_risk_zones: list[str] = []
    
    for batch in refactor_plan.get("suggested_batching", []):
        if batch.get("risk_level") == "HIGH":
            zone = batch.get("zone", "")
            if zone and zone not in high_risk_zones:
                high_risk_zones.append(zone)
    
    # Sort for deterministic output
    high_risk_zones.sort()
    
    # Build neutral headline
    if status_light == "RED":
        headline = "Structural risk in core zones requires attention before release."
    elif status_light == "YELLOW":
        headline = "Structural risk detected in some zones; review recommended."
    else:
        headline = "Structural health within acceptable parameters."
    
    return {
        "status_light": status_light,
        "high_risk_zones": high_risk_zones,
        "headline": headline,
    }


# -----------------------------------------------------------------------------
# Refactor Sprint Planner (Phase IV)
# -----------------------------------------------------------------------------

def build_refactor_sprint_plan(
    refactor_plan: dict[str, Any],
    *,
    max_batch_size: int = 3,
) -> dict[str, Any]:
    """
    Build a sprint-based refactor plan from batches.
    
    PHASE IV — REFACTOR SPRINT PLANNER
    No uplift claims are made.
    Deterministic execution guaranteed.
    
    Groups refactor batches into sprints, with 2-3 priority batches per sprint.
    
    Args:
        refactor_plan: Refactor plan from build_structural_refactor_plan()
        max_batch_size: Maximum number of batches per sprint (default: 3)
    
    Returns:
        Sprint plan dictionary:
        {
          "sprints": List[Dict[str, Any]],  # Each with sprint_id and batches
          "neutral_notes": List[str]  # Descriptive planning text
        }
    """
    suggested_batching = refactor_plan.get("suggested_batching", [])
    
    if not suggested_batching:
        return {
            "sprints": [],
            "neutral_notes": ["No refactor batches identified for sprint planning."],
        }
    
    # Sort batches by priority (HIGH first, then by batch_id for determinism)
    risk_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    sorted_batches = sorted(
        suggested_batching,
        key=lambda b: (risk_order.get(b.get("risk_level", "LOW"), 3), b.get("batch_id", "")),
    )
    
    # Group batches into sprints
    sprints: list[dict[str, Any]] = []
    current_sprint_batches: list[str] = []
    sprint_number = 1
    
    for batch in sorted_batches:
        batch_id = batch.get("batch_id", "")
        
        # Start new sprint if current one is full
        if len(current_sprint_batches) >= max_batch_size:
            sprints.append({
                "sprint_id": f"sprint_{sprint_number:02d}",
                "batches": sorted(current_sprint_batches),  # Deterministic ordering
            })
            current_sprint_batches = []
            sprint_number += 1
        
        current_sprint_batches.append(batch_id)
    
    # Add final sprint if it has batches
    if current_sprint_batches:
        sprints.append({
            "sprint_id": f"sprint_{sprint_number:02d}",
            "batches": sorted(current_sprint_batches),  # Deterministic ordering
        })
    
    # Build neutral planning notes
    neutral_notes: list[str] = []
    
    total_batches = len(sorted_batches)
    total_sprints = len(sprints)
    
    if total_sprints > 0:
        neutral_notes.append(f"Refactor plan organized into {total_sprints} sprint(s) with {total_batches} batch(es).")
        
        # Count batches by risk level
        high_batches = sum(1 for b in sorted_batches if b.get("risk_level") == "HIGH")
        medium_batches = sum(1 for b in sorted_batches if b.get("risk_level") == "MEDIUM")
        low_batches = sum(1 for b in sorted_batches if b.get("risk_level") == "LOW")
        
        if high_batches > 0:
            neutral_notes.append(f"{high_batches} batch(es) at HIGH risk level identified.")
        if medium_batches > 0:
            neutral_notes.append(f"{medium_batches} batch(es) at MEDIUM risk level identified.")
        if low_batches > 0:
            neutral_notes.append(f"{low_batches} batch(es) at LOW risk level identified.")
    else:
        neutral_notes.append("No sprints generated from refactor plan.")
    
    return {
        "sprints": sprints,
        "neutral_notes": neutral_notes,
    }


# -----------------------------------------------------------------------------
# Phase V — Semantic Drift Sentinel Grid v3
# -----------------------------------------------------------------------------

def build_semantic_drift_tensor(
    semantic_drift_timeline: dict[str, Any],
    causal_chronicle: dict[str, Any],
    drift_multi_axis_view: dict[str, Any],
) -> dict[str, Any]:
    """
    Build a semantic drift tensor combining multiple drift analysis axes.
    
    PHASE V — SEMANTIC DRIFT SENTINEL GRID v3
    No uplift claims are made.
    Deterministic execution guaranteed.
    
    Combines semantic drift timeline, causal chronicle, and multi-axis drift view
    into a unified tensor representation for cross-system safety analysis.
    
    Args:
        semantic_drift_timeline: Dict with semantic drift data per slice over time
        causal_chronicle: Dict with causal relationships and chronicle data
        drift_multi_axis_view: Dict with multi-axis drift metrics
    
    Returns:
        Semantic drift tensor dictionary:
        {
          "drift_components": {
            "<slice>": {
              "semantic": float,
              "causal": float,
              "metric_correlated": float
            }
          },
          "semantic_hotspots": List[str],  # Sorted slice identifiers
          "tensor_norm": float
        }
    """
    # Extract slice data from inputs
    drift_components: dict[str, dict[str, float]] = {}
    
    # Extract semantic drift per slice
    semantic_data = semantic_drift_timeline.get("slices", {})
    if isinstance(semantic_data, dict):
        for slice_id, slice_data in semantic_data.items():
            if slice_id not in drift_components:
                drift_components[slice_id] = {
                    "semantic": 0.0,
                    "causal": 0.0,
                    "metric_correlated": 0.0,
                }
            # Extract semantic drift score (normalize to 0-1 range)
            semantic_score = slice_data.get("drift_score", 0.0)
            if isinstance(semantic_score, (int, float)):
                drift_components[slice_id]["semantic"] = float(semantic_score)
    
    # Extract causal drift per slice
    causal_data = causal_chronicle.get("slice_causality", {})
    if isinstance(causal_data, dict):
        for slice_id, causal_info in causal_data.items():
            if slice_id not in drift_components:
                drift_components[slice_id] = {
                    "semantic": 0.0,
                    "causal": 0.0,
                    "metric_correlated": 0.0,
                }
            # Extract causal drift score
            causal_score = causal_info.get("causal_drift", 0.0)
            if isinstance(causal_score, (int, float)):
                drift_components[slice_id]["causal"] = float(causal_score)
    
    # Extract metric-correlated drift per slice
    multi_axis_data = drift_multi_axis_view.get("slice_metrics", {})
    if isinstance(multi_axis_data, dict):
        for slice_id, metric_info in multi_axis_data.items():
            if slice_id not in drift_components:
                drift_components[slice_id] = {
                    "semantic": 0.0,
                    "causal": 0.0,
                    "metric_correlated": 0.0,
                }
            # Extract metric-correlated drift score
            metric_score = metric_info.get("correlated_drift", 0.0)
            if isinstance(metric_score, (int, float)):
                drift_components[slice_id]["metric_correlated"] = float(metric_score)
    
    # Identify semantic hotspots (slices with high combined drift)
    semantic_hotspots: list[str] = []
    
    for slice_id, components in drift_components.items():
        # Calculate combined drift score (weighted average)
        combined_score = (
            components["semantic"] * 0.4 +
            components["causal"] * 0.3 +
            components["metric_correlated"] * 0.3
        )
        
        # Hotspot threshold: combined score > 0.6
        if combined_score > 0.6:
            semantic_hotspots.append(slice_id)
    
    # Sort hotspots deterministically
    semantic_hotspots.sort()
    
    # Calculate tensor norm (Frobenius norm of drift components)
    tensor_norm = 0.0
    
    for slice_id, components in drift_components.items():
        # L2 norm of component vector
        component_norm = (
            components["semantic"] ** 2 +
            components["causal"] ** 2 +
            components["metric_correlated"] ** 2
        ) ** 0.5
        tensor_norm += component_norm ** 2
    
    tensor_norm = tensor_norm ** 0.5
    
    return {
        "drift_components": drift_components,
        "semantic_hotspots": semantic_hotspots,
        "tensor_norm": tensor_norm,
    }


def analyze_semantic_drift_counterfactual(
    semantic_drift_tensor: dict[str, Any],
    *,
    projection_horizon: int = 3,
    stability_threshold: float = 0.7,
) -> dict[str, Any]:
    """
    Analyze counterfactual scenarios for semantic drift progression.
    
    PHASE V — SEMANTIC DRIFT COUNTERFACTUAL ANALYZER
    No uplift claims are made.
    Deterministic execution guaranteed.
    
    Answers: "If drift continues at current rate, which slices become unstable next?"
    
    Args:
        semantic_drift_tensor: Tensor from build_semantic_drift_tensor()
        projection_horizon: Number of time steps to project forward (default: 3)
        stability_threshold: Threshold above which a slice is considered unstable (default: 0.7)
    
    Returns:
        Counterfactual analysis dictionary:
        {
          "projected_unstable_slices": List[str],  # Sorted by projected instability time
          "stability_timeline": {
            "<slice>": {
              "current_stability": float,
              "projected_stability": List[float],  # Per time step
              "becomes_unstable_at": int | None  # Time step or None if stable
            }
          },
          "neutral_notes": List[str]
        }
    """
    drift_components = semantic_drift_tensor.get("drift_components", {})
    
    stability_timeline: dict[str, dict[str, Any]] = {}
    projected_unstable_slices: list[str] = []
    
    for slice_id, components in drift_components.items():
        # Calculate current combined drift score
        current_drift = (
            components["semantic"] * 0.4 +
            components["causal"] * 0.3 +
            components["metric_correlated"] * 0.3
        )
        
        # Current stability is inverse of drift (1.0 = fully stable, 0.0 = unstable)
        current_stability = max(0.0, 1.0 - current_drift)
        
        # Estimate drift rate (assume linear progression based on current drift)
        # Higher current drift suggests faster progression
        drift_rate = current_drift * 0.1  # Conservative estimate: 10% of current drift per step
        
        # Project stability forward
        projected_stability: list[float] = []
        becomes_unstable_at: int | None = None
        
        for step in range(projection_horizon):
            projected_drift = min(1.0, current_drift + (drift_rate * (step + 1)))
            projected_stability_value = max(0.0, 1.0 - projected_drift)
            projected_stability.append(projected_stability_value)
            
            # Check if becomes unstable at this step
            if becomes_unstable_at is None and projected_stability_value < stability_threshold:
                becomes_unstable_at = step + 1
        
        stability_timeline[slice_id] = {
            "current_stability": current_stability,
            "projected_stability": projected_stability,
            "becomes_unstable_at": becomes_unstable_at,
        }
        
        # Add to unstable slices list if projected to become unstable
        if becomes_unstable_at is not None:
            projected_unstable_slices.append(slice_id)
    
    # Sort unstable slices by when they become unstable (earliest first)
    projected_unstable_slices.sort(
        key=lambda sid: (
            stability_timeline[sid]["becomes_unstable_at"] or float("inf"),
            sid  # Secondary sort by slice ID for determinism
        )
    )
    
    # Build neutral planning notes
    neutral_notes: list[str] = []
    
    total_slices = len(drift_components)
    unstable_count = len(projected_unstable_slices)
    
    if unstable_count > 0:
        neutral_notes.append(
            f"{unstable_count} of {total_slices} slice(s) projected to become unstable "
            f"within {projection_horizon} time step(s) if current drift rate continues."
        )
        
        # List earliest unstable slices
        earliest_unstable = projected_unstable_slices[:3]
        if earliest_unstable:
            neutral_notes.append(
                f"Earliest projected instability: {', '.join(earliest_unstable)}."
            )
    else:
        neutral_notes.append(
            f"No slices projected to become unstable within {projection_horizon} time step(s)."
        )
    
    return {
        "projected_unstable_slices": projected_unstable_slices,
        "stability_timeline": stability_timeline,
        "neutral_notes": neutral_notes,
    }


def build_semantic_drift_director_panel_v3(
    semantic_drift_tensor: dict[str, Any],
    counterfactual_analysis: dict[str, Any],
) -> dict[str, Any]:
    """
    Build Director Panel v3 with semantic-drift gating recommendations.
    
    PHASE V — SEMANTIC DRIFT DIRECTOR PANEL v3
    No uplift claims are made.
    Deterministic execution guaranteed.
    
    Provides semantic-drift gating recommendations for MAAS & D5 integration.
    
    Args:
        semantic_drift_tensor: Tensor from build_semantic_drift_tensor()
        counterfactual_analysis: Analysis from analyze_semantic_drift_counterfactual()
    
    Returns:
        Director panel v3 dictionary:
        {
          "status_light": "GREEN" | "YELLOW" | "RED",
          "semantic_hotspots": List[str],
          "projected_instability_count": int,
          "gating_recommendation": "OK" | "WARN" | "BLOCK",
          "recommendation_reasons": List[str],
          "headline": str
        }
    """
    # Extract key metrics
    semantic_hotspots = semantic_drift_tensor.get("semantic_hotspots", [])
    tensor_norm = semantic_drift_tensor.get("tensor_norm", 0.0)
    projected_unstable = counterfactual_analysis.get("projected_unstable_slices", [])
    projected_instability_count = len(projected_unstable)
    
    # Determine status light based on hotspots and projected instability
    if len(semantic_hotspots) >= 3 or projected_instability_count >= 3:
        status_light = "RED"
        gating_recommendation = "BLOCK"
    elif len(semantic_hotspots) >= 1 or projected_instability_count >= 1:
        status_light = "YELLOW"
        gating_recommendation = "WARN"
    else:
        status_light = "GREEN"
        gating_recommendation = "OK"
    
    # Build recommendation reasons
    recommendation_reasons: list[str] = []
    
    if semantic_hotspots:
        recommendation_reasons.append(
            f"{len(semantic_hotspots)} semantic hotspot(s) identified: "
            f"{', '.join(semantic_hotspots[:3])}"
            + (f" and {len(semantic_hotspots) - 3} more" if len(semantic_hotspots) > 3 else "")
        )
    
    if projected_instability_count > 0:
        recommendation_reasons.append(
            f"{projected_instability_count} slice(s) projected to become unstable."
        )
    
    if tensor_norm > 2.0:
        recommendation_reasons.append(
            f"Tensor norm ({tensor_norm:.2f}) indicates elevated drift across system."
        )
    
    if not recommendation_reasons:
        recommendation_reasons.append("No significant semantic drift indicators detected.")
    
    # Build neutral headline
    if status_light == "RED":
        headline = (
            "Semantic drift analysis indicates system instability risk; "
            "gating recommendation: BLOCK."
        )
    elif status_light == "YELLOW":
        headline = (
            "Semantic drift analysis indicates potential instability; "
            "gating recommendation: WARN."
        )
    else:
        headline = (
            "Semantic drift analysis indicates system stability; "
            "gating recommendation: OK."
        )
    
    return {
        "status_light": status_light,
        "semantic_hotspots": sorted(semantic_hotspots),  # Deterministic ordering
        "projected_instability_count": projected_instability_count,
        "gating_recommendation": gating_recommendation,
        "recommendation_reasons": recommendation_reasons,
        "headline": headline,
    }


# -----------------------------------------------------------------------------
# Phase II Refactor Heatmap
# -----------------------------------------------------------------------------

def print_phase_ii_heatmap(
    risk_index: StructureRiskIndex,
    candidates: list[RefactorCandidate],
    drift_report: StructureDriftReport | None = None,
) -> str:
    """
    Generate and print a compact Phase II refactor heatmap suitable for CI logs.
    
    Args:
        risk_index: Current structure risk index
        candidates: Refactor candidates (should be Phase II filtered)
        drift_report: Optional drift report
    
    Returns:
        The heatmap string for CI summary
    """
    lines = []
    lines.append("")
    lines.append("=" * 60)
    lines.append("  PHASE II STRUCTURAL RISK HEATMAP")
    lines.append("=" * 60)
    
    # Filter to Phase II candidates
    phase_ii_candidates = [c for c in candidates if c.is_phase_ii]
    
    # Risk distribution for Phase II
    phase_ii_dirs = [d for d in risk_index.top_risky_directories 
                     if is_phase_ii_directory(d.path)]
    
    high_count = sum(1 for d in phase_ii_dirs if d.risk_level == "HIGH")
    medium_count = sum(1 for d in phase_ii_dirs if d.risk_level == "MEDIUM")
    low_count = sum(1 for d in phase_ii_dirs if d.risk_level == "LOW")
    
    lines.append("")
    lines.append(f"  📊 Phase II Risk Distribution:")
    lines.append(f"     🔴 HIGH:   {high_count:3d}")
    lines.append(f"     🟡 MEDIUM: {medium_count:3d}")
    lines.append(f"     🟢 LOW:    {low_count:3d}")
    lines.append("")
    
    # Trend indicator if drift available
    if drift_report:
        trend = drift_report.summary.get("overall_trend", "stable")
        trend_icon = {"improving": "📈", "degrading": "📉", "stable": "➡️"}.get(trend, "❓")
        lines.append(f"  📈 Trend: {trend.upper()} {trend_icon}")
        lines.append(f"     Global Risk Delta: {drift_report.global_risk_delta:+.4f}")
        lines.append("")
    
    # Top candidates
    if phase_ii_candidates:
        lines.append("  🎯 Top Refactor Candidates:")
        lines.append("  " + "-" * 56)
        lines.append(f"  {'Path':<35} {'Risk':>6} {'Priority':<10}")
        lines.append("  " + "-" * 56)
        
        for c in phase_ii_candidates[:5]:  # Top 5
            risk_bar = "█" * int(c.risk_score * 5) + "░" * (5 - int(c.risk_score * 5))
            lines.append(f"  {c.path[:35]:<35} [{risk_bar}] {c.priority:<10}")
        
        if len(phase_ii_candidates) > 5:
            lines.append(f"  ... and {len(phase_ii_candidates) - 5} more")
    else:
        lines.append("  ✅ No Phase II refactor candidates identified")
    
    lines.append("")
    lines.append("=" * 60)
    lines.append("")
    
    output = "\n".join(lines)
    print(output)
    return output


def generate_phase_ii_ci_summary(
    risk_index: StructureRiskIndex,
    candidates: list[RefactorCandidate],
    drift_report: StructureDriftReport | None = None,
) -> str:
    """
    Generate a compact CI summary for Phase II structural health.
    
    This produces markdown suitable for $GITHUB_STEP_SUMMARY.
    
    Args:
        risk_index: Current structure risk index
        candidates: Refactor candidates
        drift_report: Optional drift report
    
    Returns:
        Markdown string for GitHub step summary
    """
    lines = []
    lines.append("### 🏗️ Phase II Structural Health")
    lines.append("")
    
    # Global score
    lines.append(f"**Global Risk Score:** `{risk_index.global_risk_score:.4f}` ({risk_index.global_risk_level})")
    lines.append("")
    
    # Phase II specific counts
    phase_ii_candidates = [c for c in candidates if c.is_phase_ii]
    high_risk_phase_ii = sum(1 for c in phase_ii_candidates if c.risk_level == "HIGH")
    
    if high_risk_phase_ii > 0:
        lines.append(f"⚠️ **Phase II HIGH Risk Directories:** {high_risk_phase_ii}")
        lines.append("")
        lines.append("| Directory | Risk | Priority |")
        lines.append("|-----------|------|----------|")
        for c in phase_ii_candidates[:3]:  # Top 3 for brevity
            lines.append(f"| `{c.path}` | {c.risk_score:.2f} | {c.priority} |")
        if len(phase_ii_candidates) > 3:
            lines.append(f"| ... | {len(phase_ii_candidates) - 3} more | |")
    else:
        lines.append("✅ No HIGH-risk Phase II directories detected.")
    
    lines.append("")
    
    # Drift summary
    if drift_report:
        trend = drift_report.summary.get("overall_trend", "stable")
        trend_icon = {"improving": "📈", "degrading": "📉", "stable": "➡️"}.get(trend, "")
        lines.append(f"**Trend:** {trend.upper()} {trend_icon} (Δ `{drift_report.global_risk_delta:+.4f}`)")
        
        if drift_report.risk_transitions:
            low_to_high = drift_report.risk_transitions.get("LOW_TO_HIGH", [])
            if low_to_high:
                lines.append("")
                lines.append(f"🚨 **{len(low_to_high)} directories transitioned LOW→HIGH**")
    
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def print_drift_summary(drift_report: StructureDriftReport) -> None:
    """Print a human-readable summary of structural drift."""
    print()
    print("-" * 70)
    print("STRUCTURAL DRIFT ANALYSIS")
    print("-" * 70)
    
    # Global trend indicator
    trend = drift_report.summary.get("overall_trend", "stable")
    trend_indicator = {
        "improving": "📈 ✅",
        "degrading": "📉 ⚠️ ",
        "stable": "➡️ ",
    }.get(trend, "❓")
    
    print(f"  Overall Trend: {trend.upper()} {trend_indicator}")
    print()
    print(f"  Global Risk: {drift_report.global_risk_old:.4f} → {drift_report.global_risk_new:.4f}")
    delta_sign = "+" if drift_report.global_risk_delta >= 0 else ""
    print(f"  Global Delta: {delta_sign}{drift_report.global_risk_delta:.4f}")
    print()
    
    # Statistics
    print("  Directory Changes:")
    print(f"      Improved:  {len(drift_report.directories_with_decreased_risk):3d}")
    print(f"      Degraded:  {len(drift_report.directories_with_increased_risk):3d}")
    print(f"      New:       {len(drift_report.new_directories):3d}")
    print(f"      Removed:   {len(drift_report.removed_directories):3d}")
    print()
    
    if drift_report.max_risk_increase > 0:
        print(f"  Max Risk Increase: +{drift_report.max_risk_increase:.4f}")
    if drift_report.max_risk_decrease < 0:
        print(f"  Max Risk Decrease: {drift_report.max_risk_decrease:.4f}")
    print()
    
    # Top degraded directories
    if drift_report.directories_with_increased_risk:
        print("  ⚠️  Directories with Increased Risk:")
        for path in drift_report.directories_with_increased_risk[:5]:
            # Find the drift for this path
            drift = next((d for d in drift_report.directory_drifts if d.path == path), None)
            if drift:
                print(f"      {path}: +{drift.risk_delta:.4f}")
        if len(drift_report.directories_with_increased_risk) > 5:
            print(f"      ... and {len(drift_report.directories_with_increased_risk) - 5} more")
        print()


def print_refactor_candidates(candidates: list[RefactorCandidate]) -> None:
    """Print a summary of refactor candidates."""
    print()
    print("-" * 70)
    print("REFACTOR CANDIDATES")
    print("-" * 70)
    
    if not candidates:
        print("  ✅ No high-priority refactor candidates identified")
        print()
        return
    
    # Group by priority
    critical = [c for c in candidates if c.priority == "CRITICAL"]
    high = [c for c in candidates if c.priority == "HIGH"]
    medium = [c for c in candidates if c.priority == "MEDIUM"]
    
    if critical:
        print("  🔴 CRITICAL Priority:")
        for c in critical:
            print(f"      {c.path}")
            print(f"          Risk: {c.risk_score:.4f} | Entropy: {c.entropy:.4f}")
            print(f"          Reasons: {', '.join(c.reasons)}")
        print()
    
    if high:
        print("  🟠 HIGH Priority:")
        for c in high:
            print(f"      {c.path}")
            print(f"          Risk: {c.risk_score:.4f} | Reasons: {', '.join(c.reasons[:2])}")
        print()
    
    if medium:
        print("  🟡 MEDIUM Priority:")
        for c in medium[:3]:
            print(f"      {c.path}: {c.risk_score:.4f}")
        if len(medium) > 3:
            print(f"      ... and {len(medium) - 3} more")
        print()


def print_risk_index_summary(risk_index: StructureRiskIndex) -> None:
    """Print a human-readable summary of the risk index."""
    print()
    print("-" * 70)
    print("STRUCTURE RISK INDEX")
    print("-" * 70)
    
    # Global risk indicator
    risk_indicator = {
        "LOW": "✅",
        "MEDIUM": "⚠️ ",
        "HIGH": "🔴",
    }.get(risk_index.global_risk_level, "❓")
    
    print(f"  Global Risk Score: {risk_index.global_risk_score:.4f} ({risk_index.global_risk_level}) {risk_indicator}")
    
    if risk_index.phase_filter:
        print(f"  Phase Filter: {risk_index.phase_filter.upper()}")
    
    print(f"  Directories Analyzed: {risk_index.total_directories_analyzed}")
    print()
    
    # Risk distribution
    print("  Risk Distribution:")
    print(f"      LOW:    {risk_index.risk_distribution['LOW']:3d} directories")
    print(f"      MEDIUM: {risk_index.risk_distribution['MEDIUM']:3d} directories")
    print(f"      HIGH:   {risk_index.risk_distribution['HIGH']:3d} directories")
    print()
    
    # Drift indicators
    if risk_index.drift_indicators:
        print("  ⚠️  Drift Indicators:")
        for indicator in risk_index.drift_indicators:
            print(f"      • {indicator}")
        print()
    
    # Top risky directories
    if risk_index.top_risky_directories:
        high_risk = [d for d in risk_index.top_risky_directories if d.risk_level == "HIGH"]
        medium_risk = [d for d in risk_index.top_risky_directories if d.risk_level == "MEDIUM"]
        
        if high_risk:
            print("  🔴 HIGH-Risk Directories:")
            for d in high_risk[:5]:
                print(f"      {d.path}")
                print(f"          Score: {d.risk_score:.4f} | Entropy: {d.entropy:.4f} | Violations: {d.violations}")
                if d.contributing_factors:
                    print(f"          Factors: {', '.join(d.contributing_factors[:3])}")
            print()
        
        if medium_risk and len(high_risk) < 5:
            print("  ⚠️  MEDIUM-Risk Directories:")
            for d in medium_risk[:max(0, 5 - len(high_risk))]:
                print(f"      {d.path}: {d.risk_score:.4f}")
            print()


def main():
    parser = argparse.ArgumentParser(
        description="Directory Entropy Auditor — Agent E3 (doc-ops-3)"
    )
    parser.add_argument(
        "--output",
        default="directory_entropy_report.json",
        help="Output JSON report file",
    )
    parser.add_argument(
        "--guardian",
        action="store_true",
        help="Run Phase Boundary Guardian mode",
    )
    parser.add_argument(
        "--classify",
        action="store_true", 
        help="Run Directory Archetype Classifier",
    )
    parser.add_argument(
        "--history",
        action="store_true",
        help="Analyze git history for mutation events",
    )
    parser.add_argument(
        "--remediation",
        action="store_true",
        help="Generate remediation suggestions for violations",
    )
    parser.add_argument(
        "--risk-index",
        action="store_true",
        help="Compute Structure Risk Index and output ranked summary",
    )
    parser.add_argument(
        "--phase",
        choices=["phase1", "phase2"],
        help="Filter analysis to phase1 or phase2 directories only",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("OLD", "NEW"),
        help="Compare two reports to detect structural drift",
    )
    parser.add_argument(
        "--refactor-candidates",
        action="store_true",
        help="Generate refactor candidate shortlist",
    )
    parser.add_argument(
        "--heatmap",
        action="store_true",
        help="Display Phase II refactor heatmap (compact CI-friendly format)",
    )
    parser.add_argument(
        "--ci-summary",
        type=str,
        metavar="FILE",
        help="Write markdown summary to FILE (e.g., $GITHUB_STEP_SUMMARY)",
    )
    parser.add_argument(
        "--release-gate",
        type=str,
        metavar="DRIFT_REPORT",
        help="Evaluate structure for release gating using drift report JSON file",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON format (used with --release-gate)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all analyses",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    args = parser.parse_args()
    
    # Handle --release-gate mode separately (CI hook)
    if args.release_gate:
        drift_report_path = Path(args.release_gate)
        
        if not drift_report_path.exists():
            print(f"Error: Drift report not found: {drift_report_path}", file=sys.stderr)
            sys.exit(3)
        
        try:
            # Load drift report
            with open(drift_report_path, "r", encoding="utf-8") as f:
                drift_dict = json.load(f)
            
            # Evaluate structure for release
            release_eval = evaluate_structure_for_release(drift_dict)
            
            # Output JSON summary if requested
            if args.json:
                output = {
                    "release_ok": release_eval["release_ok"],
                    "status": release_eval["status"],
                    "refactor_zones": release_eval["refactor_zones"],
                    "reasons": release_eval["reasons"],
                }
                print(json.dumps(output, indent=2))
            else:
                # Human-readable output
                print("=" * 70)
                print("STRUCTURE-AWARE RELEASE GATE")
                print("Agent: E3 (doc-ops-3)")
                print("=" * 70)
                print()
                print(f"Status: {release_eval['status']}")
                print(f"Release OK: {release_eval['release_ok']}")
                print()
                if release_eval["refactor_zones"]:
                    print(f"Refactor Zones: {', '.join(release_eval['refactor_zones'])}")
                    print()
                if release_eval["reasons"]:
                    print("Reasons:")
                    for reason in release_eval["reasons"]:
                        print(f"  • {reason}")
                print()
            
            # Return appropriate exit code
            if release_eval["status"] == "BLOCK":
                sys.exit(2)
            elif release_eval["status"] == "WARN":
                sys.exit(1)
            else:
                sys.exit(0)
                
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in drift report: {e}", file=sys.stderr)
            sys.exit(3)
        except Exception as e:
            print(f"Error evaluating release gate: {e}", file=sys.stderr)
            sys.exit(3)
    
    # Handle --compare mode separately
    if args.compare:
        print("=" * 70)
        print("PHASE II — STRUCTURAL DRIFT DETECTOR")
        print("Agent: E3 (doc-ops-3)")
        print("=" * 70)
        print()
        
        old_path, new_path = args.compare
        print(f"Comparing: {old_path} → {new_path}")
        
        try:
            drift_report = compare_structure_risk(old_path, new_path)
            
            # Output drift report
            drift_output_path = PROJECT_ROOT / "structure_drift_report.json"
            with open(drift_output_path, "w", encoding="utf-8") as f:
                json.dump(drift_report.to_dict(), f, indent=2)
            print(f"Drift report written to: structure_drift_report.json")
            
            # Print summary
            print_drift_summary(drift_report)
            
            # Generate refactor candidates if requested
            if args.refactor_candidates:
                # Load current risk index
                new_report = load_report_from_file(new_path)
                risk_index = compute_structure_risk_index(new_report)
                candidates = generate_refactor_candidates(risk_index, drift_report)
                
                if candidates:
                    output_path = write_refactor_candidates(candidates)
                    print(f"  Refactor candidates written to: {output_path.relative_to(PROJECT_ROOT)}")
                    print_refactor_candidates(candidates)
            
            print("=" * 70)
            sys.exit(0)
            
        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Error comparing reports: {e}")
            sys.exit(1)
    
    # Default to all if no specific flags (except risk-index which is separate)
    if not (args.guardian or args.classify or args.history or args.risk_index):
        args.all = True
    
    if args.all:
        args.guardian = True
        args.classify = True
        args.history = True
        args.remediation = True
    
    print("=" * 70)
    print("PHASE II — DIRECTORY ENTROPY AUDITOR")
    print("Agent: E3 (doc-ops-3)")
    if args.phase:
        print(f"Phase Filter: {args.phase.upper()}")
    print("=" * 70)
    print()
    
    # Generate report
    if args.verbose:
        print("Computing directory entropies...")
    
    report = generate_report(
        include_guardian=args.guardian,
        include_classify=args.classify,
        include_history=args.history,
        include_remediation=args.remediation,
    )
    
    # Apply phase filter to the report if specified
    if args.phase:
        filtered_scores = filter_directories_by_phase(report.directory_scores, args.phase)
        # Update report with filtered data
        report.directory_scores = filtered_scores
        report.total_directories = len(filtered_scores)
        report.total_files = sum(s.file_count for s in filtered_scores)
        if filtered_scores:
            report.global_entropy = round(
                sum(s.total_entropy for s in filtered_scores) / len(filtered_scores), 4
            )
        else:
            report.global_entropy = 0.0
    
    # Output report
    output_path = PROJECT_ROOT / args.output
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2, sort_keys=False)
    
    print(f"Report written to: {args.output}")
    print(f"Report hash: {report.report_hash}")
    print()
    
    # Print summary
    print("-" * 70)
    print("SUMMARY")
    print("-" * 70)
    print(f"  Total directories analyzed: {report.total_directories}")
    print(f"  Total files: {report.total_files}")
    print(f"  Global entropy score: {report.global_entropy:.4f}")
    print()
    
    if report.boundary_violations:
        print(f"  ⚠️  Phase boundary violations: {len(report.boundary_violations)}")
        for v in report.boundary_violations[:5]:
            print(f"      {v.source_file}:{v.line_number} → {v.target_module}")
        if len(report.boundary_violations) > 5:
            print(f"      ... and {len(report.boundary_violations) - 5} more")
        print()
    else:
        print("  ✅ No phase boundary violations detected")
        print()
    
    if report.directory_scores:
        high_entropy = [s for s in report.directory_scores if s.total_entropy > 1.5]
        if high_entropy:
            print(f"  ⚠️  High entropy directories ({len(high_entropy)}):")
            for s in high_entropy[:5]:
                print(f"      {s.directory}: {s.total_entropy:.4f}")
        else:
            print("  ✅ No high-entropy directories detected")
        print()
    
    if args.classify and report.archetype_classifications:
        print("  Directory archetypes:")
        archetype_counts = Counter(c.primary_archetype for c in report.archetype_classifications)
        for archetype, count in archetype_counts.most_common():
            print(f"      {archetype}: {count}")
        print()
    
    if args.history and report.mutation_events:
        print(f"  Recent mutations: {len(report.mutation_events)}")
        print(f"      Added: {report.summary['mutation_summary'].get('file_added', 0)}")
        print(f"      Removed: {report.summary['mutation_summary'].get('file_removed', 0)}")
        print(f"      Moved: {report.summary['mutation_summary'].get('file_moved', 0)}")
        print()
    
    # Compute and display risk index if requested
    risk_index = None
    if args.risk_index or args.refactor_candidates:
        risk_index = compute_structure_risk_index(report, phase_filter=args.phase)
        
        if args.risk_index:
            print_risk_index_summary(risk_index)
            
            # Write risk index to separate file
            risk_output_path = PROJECT_ROOT / "structure_risk_index.json"
            with open(risk_output_path, "w", encoding="utf-8") as f:
                json.dump(risk_index.to_dict(), f, indent=2, sort_keys=False)
            print(f"  Risk index written to: structure_risk_index.json")
            print()
    
    # Generate refactor candidates if requested
    candidates = []
    if args.refactor_candidates and risk_index:
        candidates = generate_refactor_candidates(risk_index)
        
        if candidates:
            output_path = write_refactor_candidates(candidates)
            print(f"  Refactor candidates written to: {output_path.relative_to(PROJECT_ROOT)}")
        
        print_refactor_candidates(candidates)
    
    # Display Phase II heatmap if requested or if phase2 filter active
    if args.heatmap or (args.phase == "phase2" and args.refactor_candidates):
        if not risk_index:
            risk_index = compute_structure_risk_index(report, phase_filter=args.phase)
        if not candidates:
            candidates = generate_refactor_candidates(risk_index)
        
        print_phase_ii_heatmap(risk_index, candidates)
    
    # Write CI summary if requested
    if args.ci_summary:
        if not risk_index:
            risk_index = compute_structure_risk_index(report, phase_filter=args.phase)
        if not candidates:
            candidates = generate_refactor_candidates(risk_index)
        
        ci_summary = generate_phase_ii_ci_summary(risk_index, candidates)
        
        # Write to file (typically $GITHUB_STEP_SUMMARY)
        try:
            with open(args.ci_summary, "a", encoding="utf-8") as f:
                f.write(ci_summary)
                f.write("\n\n")
            print(f"  CI summary appended to: {args.ci_summary}")
        except Exception as e:
            print(f"  Warning: Could not write CI summary: {e}")
    
    print("=" * 70)
    
    # Exit with error if boundary violations found
    if report.boundary_violations:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()

