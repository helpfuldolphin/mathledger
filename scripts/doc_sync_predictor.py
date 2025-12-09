#!/usr/bin/env python3
"""
Governance Drift Prediction Engine

PHASE II — NOT RUN IN PHASE I
No uplift claims are made.
Deterministic execution guaranteed.

This module consumes historical doc-sync scan outputs and predicts
terminology drift patterns to identify high-risk files likely to
diverge from governance vocabulary.

Capabilities:
  - Consume last N doc-sync scan outputs
  - Compute drift vectors: term frequency shifts, schema mismatch deltas
  - Predict high-risk terminology regions (files likely to drift next)
  - Produce governance_drift_forecast.json artifact
  - Deterministic output: same inputs → identical forecast

Author: E1 (doc-ops-1) — Governance Synchronization Officer
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

# Deterministic timestamp for reproducibility
DETERMINISTIC_EPOCH = "2025-01-01T00:00:00Z"

# Default scan directories for vocabulary index
DEFAULT_SCAN_DIRS = ["docs", "experiments", "backend", "scripts", "tests", "rfl"]


@dataclass(frozen=True)
class DriftVector:
    """
    Immutable drift vector representing terminology change direction.
    
    Components:
      - term_frequency_delta: Change in term occurrence count
      - violation_delta: Change in violation count
      - severity_weight: Weighted severity score (error=3, warning=1, info=0.1)
      - file_churn: Number of files affected by this term's violations
    """
    term: str
    term_frequency_delta: float
    violation_delta: int
    severity_weight: float
    file_churn: int
    
    @property
    def magnitude(self) -> float:
        """Compute drift vector magnitude (L2 norm of components)."""
        return math.sqrt(
            self.term_frequency_delta ** 2 +
            self.violation_delta ** 2 +
            self.severity_weight ** 2 +
            self.file_churn ** 2
        )
    
    @property
    def risk_score(self) -> float:
        """
        Compute composite risk score.
        
        Higher scores indicate greater likelihood of governance drift.
        Formula: magnitude * log(1 + severity_weight) * (1 + violation_delta/10)
        """
        return self.magnitude * math.log1p(self.severity_weight) * (1 + self.violation_delta / 10)


@dataclass
class FileRiskProfile:
    """Risk profile for a single file's terminology alignment."""
    
    file_path: str
    violation_count: int
    error_count: int
    warning_count: int
    affected_terms: List[str]
    drift_risk_score: float
    historical_violations: List[int]  # Violation counts from previous scans
    trend: str  # "increasing", "decreasing", "stable", "new"
    
    @property
    def is_high_risk(self) -> bool:
        """Determine if file is high-risk based on multiple factors."""
        return (
            self.drift_risk_score > 5.0 or
            self.error_count > 0 or
            self.trend == "increasing" or
            len(self.affected_terms) >= 3
        )


@dataclass
class DriftForecast:
    """
    Complete governance drift forecast.
    
    Deterministic: identical inputs produce identical forecasts.
    """
    
    # Metadata
    forecast_id: str  # SHA256 of input data for reproducibility
    generated_at: str  # ISO 8601 timestamp (deterministic from inputs)
    input_scan_count: int
    
    # Global metrics
    total_drift_magnitude: float
    average_drift_per_term: float
    high_risk_file_count: int
    trending_violations: str  # "increasing", "decreasing", "stable"
    
    # Detailed predictions
    drift_vectors: List[Dict[str, Any]]  # Serialized DriftVector objects
    high_risk_files: List[Dict[str, Any]]  # Serialized FileRiskProfile objects
    term_hotspots: Dict[str, float]  # term -> risk score
    predicted_next_violations: Dict[str, int]  # file -> predicted new violations
    
    # Schema mismatch tracking
    schema_mismatch_deltas: Dict[str, int]  # category -> delta count
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    def to_json(self, indent: int = 2) -> str:
        """Serialize to deterministic JSON string."""
        return json.dumps(
            self.to_dict(),
            indent=indent,
            sort_keys=True,
            ensure_ascii=True
        )


@dataclass
class ScanSnapshot:
    """Parsed snapshot from a single doc-sync scan output."""
    
    scan_id: str
    timestamp: str
    total_violations: int
    violations_by_severity: Dict[str, int]
    violations_by_category: Dict[str, int]
    violations_by_file: Dict[str, int]
    violations_by_term: Dict[str, int]
    docstring_compliance_rate: float
    orphaned_doc_count: int


class GovernanceDriftPredictor:
    """
    Governance Drift Prediction Engine.
    
    PHASE II — NOT RUN IN PHASE I
    No uplift claims are made.
    Deterministic execution guaranteed.
    
    Consumes historical scan data and produces drift forecasts.
    """
    
    # Risk thresholds
    HIGH_RISK_THRESHOLD = 5.0
    INCREASING_TREND_THRESHOLD = 1.2  # 20% increase
    DECREASING_TREND_THRESHOLD = 0.8  # 20% decrease
    
    # Severity weights for risk calculation
    SEVERITY_WEIGHTS = {
        "error": 3.0,
        "warning": 1.0,
        "info": 0.1,
    }
    
    def __init__(self, scan_history_dir: Optional[Path] = None):
        """
        Initialize predictor.
        
        Args:
            scan_history_dir: Directory containing historical scan JSON outputs.
                              If None, uses artifacts/doc_sync_history/
        """
        self.scan_history_dir = scan_history_dir or Path("artifacts/doc_sync_history")
        self.snapshots: List[ScanSnapshot] = []
    
    def load_scan_history(self, max_scans: int = 10) -> int:
        """
        Load historical scan outputs.
        
        Args:
            max_scans: Maximum number of recent scans to load
            
        Returns:
            Number of scans loaded
        """
        self.snapshots.clear()
        
        if not self.scan_history_dir.exists():
            return 0
        
        # Find all scan JSON files, sorted by modification time (oldest first for determinism)
        scan_files = sorted(
            self.scan_history_dir.glob("doc_scan_*.json"),
            key=lambda p: p.stat().st_mtime if p.exists() else 0
        )
        
        # Take most recent N
        scan_files = scan_files[-max_scans:]
        
        for scan_file in scan_files:
            snapshot = self._parse_scan_file(scan_file)
            if snapshot:
                self.snapshots.append(snapshot)
        
        return len(self.snapshots)
    
    def load_from_data(self, scan_data_list: List[Dict[str, Any]]) -> int:
        """
        Load scan data directly from dictionaries (for testing).
        
        Args:
            scan_data_list: List of scan output dictionaries
            
        Returns:
            Number of scans loaded
        """
        self.snapshots.clear()
        
        for i, data in enumerate(scan_data_list):
            snapshot = self._parse_scan_data(data, f"synthetic_{i:04d}")
            if snapshot:
                self.snapshots.append(snapshot)
        
        return len(self.snapshots)
    
    def _parse_scan_file(self, scan_file: Path) -> Optional[ScanSnapshot]:
        """Parse a single scan output file."""
        try:
            with open(scan_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return self._parse_scan_data(data, scan_file.stem)
        except Exception:
            return None
    
    def _parse_scan_data(self, data: Dict[str, Any], scan_id: str) -> Optional[ScanSnapshot]:
        """Parse scan data dictionary into ScanSnapshot."""
        try:
            # Extract violation counts by file
            violations_by_file: Dict[str, int] = defaultdict(int)
            violations_by_term: Dict[str, int] = defaultdict(int)
            
            for violation in data.get("violations", []):
                file_path = violation.get("file", "unknown")
                found_term = violation.get("found", "unknown")
                violations_by_file[file_path] += 1
                violations_by_term[found_term] += 1
            
            # Extract compliance rate
            compliance = data.get("docstring_compliance", {})
            compliance_rate = compliance.get("compliance_rate", 0.0)
            
            return ScanSnapshot(
                scan_id=scan_id,
                timestamp=data.get("timestamp", DETERMINISTIC_EPOCH),
                total_violations=data.get("total_violations", 0),
                violations_by_severity=data.get("violations_by_severity", {}),
                violations_by_category=data.get("violations_by_category", {}),
                violations_by_file=dict(violations_by_file),
                violations_by_term=dict(violations_by_term),
                docstring_compliance_rate=compliance_rate,
                orphaned_doc_count=data.get("orphaned_documentation", 0),
            )
        except Exception:
            return None
    
    def compute_drift_vectors(self) -> List[DriftVector]:
        """
        Compute drift vectors for all terms across scan history.
        
        Returns:
            List of DriftVector objects sorted by risk score (descending)
        """
        if len(self.snapshots) < 2:
            return []
        
        # Aggregate term violations across all snapshots
        term_history: Dict[str, List[int]] = defaultdict(list)
        term_files: Dict[str, Set[str]] = defaultdict(set)
        
        for snapshot in self.snapshots:
            # Track which terms appear in each snapshot
            seen_terms: Set[str] = set()
            for term, count in snapshot.violations_by_term.items():
                term_history[term].append(count)
                seen_terms.add(term)
            
            # Add zeros for terms not seen in this snapshot
            for term in term_history:
                if term not in seen_terms:
                    term_history[term].append(0)
        
        # Track which files each term affects (from most recent snapshot)
        latest = self.snapshots[-1]
        for violation_file, count in latest.violations_by_file.items():
            # Approximate: we don't have per-term-per-file in snapshot
            # Use overall file as proxy
            pass
        
        # Compute drift vectors
        vectors: List[DriftVector] = []
        
        for term, history in term_history.items():
            if len(history) < 2:
                continue
            
            # Compute deltas between first and last observation
            first_count = history[0]
            last_count = history[-1]
            
            freq_delta = last_count - first_count
            violation_delta = last_count - first_count
            
            # Compute severity weight (use warning as default)
            severity_weight = self.SEVERITY_WEIGHTS.get("warning", 1.0)
            
            # Count file churn (unique files affected across history)
            file_churn = sum(1 for h in history if h > 0)
            
            vector = DriftVector(
                term=term,
                term_frequency_delta=float(freq_delta),
                violation_delta=violation_delta,
                severity_weight=severity_weight,
                file_churn=file_churn,
            )
            vectors.append(vector)
        
        # Sort by risk score (descending) for deterministic ordering
        vectors.sort(key=lambda v: (-v.risk_score, v.term))
        
        return vectors
    
    def compute_file_risk_profiles(self) -> List[FileRiskProfile]:
        """
        Compute risk profiles for all files with violations.
        
        Returns:
            List of FileRiskProfile objects sorted by risk score (descending)
        """
        if not self.snapshots:
            return []
        
        # Build file history
        file_history: Dict[str, List[int]] = defaultdict(list)
        
        for snapshot in self.snapshots:
            seen_files: Set[str] = set()
            for file_path, count in snapshot.violations_by_file.items():
                file_history[file_path].append(count)
                seen_files.add(file_path)
            
            # Add zeros for files not seen
            for file_path in file_history:
                if file_path not in seen_files:
                    file_history[file_path].append(0)
        
        # Get latest snapshot for current state
        latest = self.snapshots[-1]
        
        profiles: List[FileRiskProfile] = []
        
        for file_path, history in file_history.items():
            current_violations = history[-1] if history else 0
            
            # Determine trend
            if len(history) < 2:
                trend = "new"
            else:
                first_nonzero = next((h for h in history if h > 0), 0)
                if first_nonzero == 0 and current_violations > 0:
                    trend = "new"
                elif current_violations == 0:
                    trend = "resolved"
                elif current_violations > history[-2] * self.INCREASING_TREND_THRESHOLD:
                    trend = "increasing"
                elif current_violations < history[-2] * self.DECREASING_TREND_THRESHOLD:
                    trend = "decreasing"
                else:
                    trend = "stable"
            
            # Compute risk score
            base_risk = current_violations * 0.5
            trend_multiplier = {
                "increasing": 2.0,
                "new": 1.5,
                "stable": 1.0,
                "decreasing": 0.5,
                "resolved": 0.1,
            }.get(trend, 1.0)
            
            risk_score = base_risk * trend_multiplier
            
            # Extract affected terms (approximate from latest violations)
            affected_terms = self._get_affected_terms_for_file(file_path, latest)
            
            profile = FileRiskProfile(
                file_path=file_path,
                violation_count=current_violations,
                error_count=0,  # Would need per-file severity breakdown
                warning_count=current_violations,
                affected_terms=affected_terms,
                drift_risk_score=risk_score,
                historical_violations=history,
                trend=trend,
            )
            profiles.append(profile)
        
        # Sort by risk score (descending), then by path for determinism
        profiles.sort(key=lambda p: (-p.drift_risk_score, p.file_path))
        
        return profiles
    
    def _get_affected_terms_for_file(
        self, file_path: str, snapshot: ScanSnapshot
    ) -> List[str]:
        """Get terms that caused violations in a specific file."""
        # This is an approximation since we don't have per-file term breakdown
        # Return top terms from overall snapshot
        terms = sorted(
            snapshot.violations_by_term.items(),
            key=lambda x: (-x[1], x[0])
        )
        return [t[0] for t in terms[:5]]
    
    def compute_schema_mismatch_deltas(self) -> Dict[str, int]:
        """
        Compute changes in schema mismatch counts by category.
        
        Returns:
            Dictionary mapping category to violation count delta
        """
        if len(self.snapshots) < 2:
            return {}
        
        first = self.snapshots[0]
        last = self.snapshots[-1]
        
        deltas: Dict[str, int] = {}
        
        # Get all categories
        all_categories = set(first.violations_by_category.keys()) | set(
            last.violations_by_category.keys()
        )
        
        for category in sorted(all_categories):
            first_count = first.violations_by_category.get(category, 0)
            last_count = last.violations_by_category.get(category, 0)
            deltas[category] = last_count - first_count
        
        return deltas
    
    def predict_next_violations(
        self, profiles: List[FileRiskProfile]
    ) -> Dict[str, int]:
        """
        Predict violation counts for next scan based on trends.
        
        Uses simple linear extrapolation with trend-based multipliers.
        
        Args:
            profiles: File risk profiles with historical data
            
        Returns:
            Dictionary mapping file path to predicted violation count
        """
        predictions: Dict[str, int] = {}
        
        for profile in profiles:
            if len(profile.historical_violations) < 2:
                # New file: predict current + 1 if increasing, else current
                predicted = profile.violation_count + (1 if profile.trend == "new" else 0)
            else:
                # Linear extrapolation
                history = profile.historical_violations
                slope = (history[-1] - history[0]) / len(history)
                
                # Apply trend multiplier
                trend_factor = {
                    "increasing": 1.5,
                    "new": 1.2,
                    "stable": 1.0,
                    "decreasing": 0.8,
                    "resolved": 0.0,
                }.get(profile.trend, 1.0)
                
                predicted = max(0, int(history[-1] + slope * trend_factor))
            
            predictions[profile.file_path] = predicted
        
        return predictions
    
    def generate_forecast(self) -> DriftForecast:
        """
        Generate complete governance drift forecast.
        
        Returns:
            DriftForecast object with all predictions and metrics
        """
        # Compute components
        drift_vectors = self.compute_drift_vectors()
        file_profiles = self.compute_file_risk_profiles()
        schema_deltas = self.compute_schema_mismatch_deltas()
        predictions = self.predict_next_violations(file_profiles)
        
        # Compute aggregate metrics
        total_drift = sum(v.magnitude for v in drift_vectors)
        avg_drift = total_drift / len(drift_vectors) if drift_vectors else 0.0
        high_risk_count = sum(1 for p in file_profiles if p.is_high_risk)
        
        # Determine overall trend
        if len(self.snapshots) >= 2:
            first_total = self.snapshots[0].total_violations
            last_total = self.snapshots[-1].total_violations
            if last_total > first_total * self.INCREASING_TREND_THRESHOLD:
                trending = "increasing"
            elif last_total < first_total * self.DECREASING_TREND_THRESHOLD:
                trending = "decreasing"
            else:
                trending = "stable"
        else:
            trending = "insufficient_data"
        
        # Compute term hotspots (top risk terms)
        term_hotspots = {v.term: v.risk_score for v in drift_vectors[:20]}
        
        # Generate deterministic forecast ID
        input_hash = self._compute_input_hash()
        
        # Generate deterministic timestamp from input hash
        timestamp = self._deterministic_timestamp(input_hash)
        
        # Serialize vectors and profiles
        serialized_vectors = [
            {
                "term": v.term,
                "term_frequency_delta": v.term_frequency_delta,
                "violation_delta": v.violation_delta,
                "severity_weight": v.severity_weight,
                "file_churn": v.file_churn,
                "magnitude": v.magnitude,
                "risk_score": v.risk_score,
            }
            for v in drift_vectors
        ]
        
        serialized_profiles = [
            {
                "file_path": p.file_path,
                "violation_count": p.violation_count,
                "error_count": p.error_count,
                "warning_count": p.warning_count,
                "affected_terms": p.affected_terms,
                "drift_risk_score": p.drift_risk_score,
                "trend": p.trend,
                "is_high_risk": p.is_high_risk,
            }
            for p in file_profiles
            if p.violation_count > 0  # Only include files with violations
        ]
        
        return DriftForecast(
            forecast_id=input_hash,
            generated_at=timestamp,
            input_scan_count=len(self.snapshots),
            total_drift_magnitude=total_drift,
            average_drift_per_term=avg_drift,
            high_risk_file_count=high_risk_count,
            trending_violations=trending,
            drift_vectors=serialized_vectors,
            high_risk_files=serialized_profiles[:50],  # Limit to top 50
            term_hotspots=term_hotspots,
            predicted_next_violations=predictions,
            schema_mismatch_deltas=schema_deltas,
        )
    
    def _compute_input_hash(self) -> str:
        """Compute deterministic hash of all input data."""
        # Serialize snapshots deterministically
        snapshot_data = []
        for s in self.snapshots:
            snapshot_data.append({
                "scan_id": s.scan_id,
                "total_violations": s.total_violations,
                "violations_by_severity": s.violations_by_severity,
                "violations_by_category": s.violations_by_category,
            })
        
        content = json.dumps(snapshot_data, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
    
    def _deterministic_timestamp(self, input_hash: str) -> str:
        """Generate deterministic timestamp from input hash."""
        # Use hash prefix to generate a consistent timestamp offset
        # This ensures same inputs → same timestamp
        offset_hours = int(input_hash[:8], 16) % (24 * 365)  # Up to 1 year offset
        
        # Base epoch + deterministic offset
        base_ts = datetime(2025, 1, 1, 0, 0, 0)
        from datetime import timedelta
        result_ts = base_ts + timedelta(hours=offset_hours)
        
        return result_ts.strftime("%Y-%m-%dT%H:%M:%SZ")


# ==============================================================================
# GOVERNANCE VOCABULARY LINTER
# ==============================================================================


class GovernanceVocabularyLinter:
    """
    Linter for PR descriptions and commit messages.
    
    PHASE II — NOT RUN IN PHASE I
    No uplift claims are made.
    Deterministic execution guaranteed.
    
    Enforces canonical terminology in human-written text.
    """
    
    def __init__(self, vocabulary: Optional[Dict[str, Any]] = None):
        """
        Initialize linter.
        
        Args:
            vocabulary: Optional vocabulary dictionary. If None, loads from scanner.
        """
        if vocabulary is None:
            # Import from scanner module
            from scripts.doc_sync_scanner import build_governance_vocabulary
            vocab = build_governance_vocabulary()
            self.vocabulary = {
                name: {
                    "canonical": name,
                    "doc_variants": list(term.doc_variants),
                    "code_variants": list(term.code_variants),
                    "category": term.category,
                }
                for name, term in vocab.items()
            }
        else:
            self.vocabulary = vocabulary
        
        # Build reverse lookup: variant -> canonical
        self.variant_to_canonical: Dict[str, str] = {}
        for name, term in self.vocabulary.items():
            for variant in term.get("doc_variants", []) + term.get("code_variants", []):
                if variant.lower() != name.lower():
                    self.variant_to_canonical[variant.lower()] = name
    
    def lint_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Lint text for terminology violations.
        
        Args:
            text: Text to lint (PR description, commit message, etc.)
            
        Returns:
            List of violation dictionaries
        """
        import re
        
        violations: List[Dict[str, Any]] = []
        lines = text.split("\n")
        
        for line_num, line in enumerate(lines, 1):
            for variant, canonical in self.variant_to_canonical.items():
                # Case-insensitive word boundary search
                pattern = r'\b' + re.escape(variant) + r'\b'
                matches = list(re.finditer(pattern, line, re.IGNORECASE))
                
                for match in matches:
                    found = match.group()
                    # Skip if already canonical
                    if found.lower() == canonical.lower():
                        continue
                    
                    violations.append({
                        "line": line_num,
                        "column": match.start() + 1,
                        "found": found,
                        "canonical": canonical,
                        "category": self.vocabulary[canonical].get("category", "unknown"),
                        "context": line.strip()[:80],
                        "suggestion": f"Use '{canonical}' instead of '{found}'",
                    })
        
        return violations
    
    def lint_pr_description(self, description: str) -> Dict[str, Any]:
        """
        Lint a PR description.
        
        Args:
            description: PR description text
            
        Returns:
            Linting result with violations and pass/fail status
        """
        violations = self.lint_text(description)
        
        return {
            "passed": len(violations) == 0,
            "violation_count": len(violations),
            "violations": violations,
            "summary": (
                "PR description uses canonical terminology"
                if len(violations) == 0
                else f"Found {len(violations)} terminology issue(s) in PR description"
            ),
        }
    
    def lint_commit_message(self, message: str) -> Dict[str, Any]:
        """
        Lint a commit message.
        
        Args:
            message: Commit message text
            
        Returns:
            Linting result with violations and pass/fail status
        """
        violations = self.lint_text(message)
        
        # For commits, only flag errors for critical terms
        critical_violations = [
            v for v in violations
            if v["category"] in {"metric", "phase", "symbol"}
        ]
        
        return {
            "passed": len(critical_violations) == 0,
            "violation_count": len(violations),
            "critical_violation_count": len(critical_violations),
            "violations": violations,
            "summary": (
                "Commit message uses canonical terminology"
                if len(critical_violations) == 0
                else f"Found {len(critical_violations)} critical terminology issue(s)"
            ),
        }


# ==============================================================================
# GOVERNANCE TERM TIMELINE INDEX
# ==============================================================================


@dataclass
class CommitTermSnapshot:
    """Term usage snapshot for a single commit."""
    
    commit_hash: str
    commit_date: str  # ISO 8601, deterministic
    term_count: int
    variant_count: int
    files_touched: int
    variants_found: List[str]


def build_term_timeline(
    term: str,
    history_window: int = 50,
    root_path: Optional[Path] = None,
    scan_dirs: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Build a timeline of term usage across git history.
    
    PHASE II — NOT RUN IN PHASE I
    No uplift claims are made.
    Deterministic execution guaranteed.
    
    Args:
        term: Canonical governance term to track (e.g., "Phase II", "RFL")
        history_window: Number of commits to analyze (default 50)
        root_path: Repository root path (default: current directory)
        scan_dirs: Directories to scan for term occurrences
        
    Returns:
        Deterministic JSON-serializable dict with timeline data
    """
    import re
    import subprocess
    
    root_path = root_path or Path.cwd()
    scan_dirs = scan_dirs or DEFAULT_SCAN_DIRS
    
    # Get governance vocabulary for variant detection
    try:
        from scripts.doc_sync_scanner import build_governance_vocabulary
        vocabulary = build_governance_vocabulary()
    except ImportError:
        vocabulary = {}
    
    # Find variants for this term
    term_lower = term.lower().replace(" ", "_").replace("-", "_")
    variants: Set[str] = {term}
    
    for canonical_name, term_def in vocabulary.items():
        if (canonical_name.lower() == term_lower or
            term.lower() in [v.lower() for v in term_def.doc_variants] or
            term.lower() in [v.lower() for v in term_def.code_variants]):
            variants.update(term_def.doc_variants)
            variants.update(term_def.code_variants)
            break
    
    # Build regex pattern for all variants
    variant_patterns = [re.escape(v) for v in sorted(variants)]
    combined_pattern = r'\b(' + '|'.join(variant_patterns) + r')\b'
    
    # Get git log (deterministic: sorted by commit date)
    try:
        git_log = subprocess.run(
            ["git", "log", f"-{history_window}", "--format=%H|%aI", "--date-order"],
            cwd=root_path,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
        )
        commits = [
            line.split("|") for line in git_log.stdout.strip().split("\n")
            if line and "|" in line
        ]
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        # Git not available or not a repo
        return {
            "term": term,
            "commits_analyzed": 0,
            "timeline": [],
            "error": "Git not available or not a repository",
        }
    
    timeline: List[Dict[str, Any]] = []
    
    for commit_hash, commit_date in commits:
        # Count term occurrences in this commit's tree
        term_count = 0
        variant_counts: Dict[str, int] = defaultdict(int)
        files_with_term: Set[str] = set()
        
        for scan_dir in scan_dirs:
            dir_path = root_path / scan_dir
            if not dir_path.exists():
                continue
            
            # Get files at this commit
            try:
                ls_tree = subprocess.run(
                    ["git", "ls-tree", "-r", "--name-only", commit_hash, scan_dir],
                    cwd=root_path,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    check=True,
                )
                files = ls_tree.stdout.strip().split("\n") if ls_tree.stdout else []
            except subprocess.CalledProcessError:
                continue
            
            for file_path in files:
                if not file_path:
                    continue
                    
                # Get file content at this commit
                try:
                    show = subprocess.run(
                        ["git", "show", f"{commit_hash}:{file_path}"],
                        cwd=root_path,
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        check=True,
                    )
                    content = show.stdout
                    if content is None:
                        continue
                except subprocess.CalledProcessError:
                    continue
                
                # Count term occurrences
                try:
                    matches = re.findall(combined_pattern, content, re.IGNORECASE)
                except (TypeError, re.error):
                    continue
                    
                if matches:
                    files_with_term.add(file_path)
                    for match in matches:
                        term_count += 1
                        variant_counts[match] += 1
        
        # Build snapshot
        snapshot = {
            "commit": commit_hash[:12],  # Short hash for readability
            "date": commit_date,
            "term_count": term_count,
            "variant_count": len(variant_counts),
            "files_touched": len(files_with_term),
            "variants_found": sorted(variant_counts.keys()),
        }
        timeline.append(snapshot)
    
    # Reverse to chronological order (oldest first)
    timeline.reverse()
    
    # Compute deterministic ID from inputs
    input_data = json.dumps({
        "term": term,
        "history_window": history_window,
        "commits": [c[0] for c in commits],
    }, sort_keys=True)
    timeline_id = hashlib.sha256(input_data.encode()).hexdigest()
    
    return {
        "term": term,
        "timeline_id": timeline_id,
        "commits_analyzed": len(commits),
        "variants_tracked": sorted(variants),
        "timeline": timeline,
    }


# ==============================================================================
# GOVERNANCE VOCABULARY INDEX EXPORT
# ==============================================================================


def export_governance_vocabulary_index(
    out_path: str,
    root_path: Optional[Path] = None,
    scan_dirs: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Export complete governance vocabulary index with file occurrences.
    
    PHASE II — NOT RUN IN PHASE I
    No uplift claims are made.
    Deterministic execution guaranteed.
    
    Args:
        out_path: Output file path for JSON export
        root_path: Repository root path
        scan_dirs: Directories to scan for occurrences
        
    Returns:
        The exported index dictionary
    """
    import re
    
    root_path = root_path or Path.cwd()
    scan_dirs = scan_dirs or DEFAULT_SCAN_DIRS
    
    # Load governance vocabulary
    try:
        from scripts.doc_sync_scanner import build_governance_vocabulary
        vocabulary = build_governance_vocabulary()
    except ImportError:
        return {"error": "Could not load governance vocabulary"}
    
    # Build index structure
    index: Dict[str, Any] = {
        "version": "1.0.0",
        "generated_at": DETERMINISTIC_EPOCH,  # Will be updated with deterministic timestamp
        "total_terms": len(vocabulary),
        "terms": {},
    }
    
    # Track all file occurrences
    term_files: Dict[str, Set[str]] = defaultdict(set)
    term_variant_occurrences: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    
    # Scan files for term occurrences
    ignore_dirs = {"__pycache__", ".git", "node_modules", ".venv", "venv", "MagicMock"}
    scan_extensions = {".py", ".md", ".yaml", ".yml", ".json", ".tex", ".txt"}
    
    for scan_dir in scan_dirs:
        dir_path = root_path / scan_dir
        if not dir_path.exists():
            continue
        
        for file_path in dir_path.rglob("*"):
            # Skip ignored directories
            if any(part in ignore_dirs for part in file_path.parts):
                continue
            
            # Skip non-text files
            if file_path.suffix.lower() not in scan_extensions:
                continue
            
            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            
            rel_path = str(file_path.relative_to(root_path))
            
            # Check each term and its variants
            for canonical_name, term_def in vocabulary.items():
                all_variants = set(term_def.doc_variants) | set(term_def.code_variants)
                all_variants.add(canonical_name)
                
                for variant in all_variants:
                    pattern = r'\b' + re.escape(variant) + r'\b'
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    if matches:
                        term_files[canonical_name].add(rel_path)
                        for match in matches:
                            term_variant_occurrences[canonical_name][match] += 1
    
    # Build term entries
    for canonical_name, term_def in sorted(vocabulary.items()):
        files = sorted(term_files.get(canonical_name, set()))
        variant_counts = dict(sorted(
            term_variant_occurrences.get(canonical_name, {}).items()
        ))
        
        index["terms"][canonical_name] = {
            "canonical": canonical_name,
            "category": term_def.category,
            "description": term_def.description,
            "governance_source": term_def.governance_source,
            "doc_variants": sorted(term_def.doc_variants),
            "code_variants": sorted(term_def.code_variants),
            "files": files,
            "file_count": len(files),
            "variant_occurrences": variant_counts,
            "total_occurrences": sum(variant_counts.values()),
        }
    
    # Compute deterministic timestamp from content
    content_hash = hashlib.sha256(
        json.dumps(index["terms"], sort_keys=True).encode()
    ).hexdigest()
    offset_hours = int(content_hash[:8], 16) % (24 * 365)
    from datetime import timedelta
    generated_at = datetime(2025, 1, 1) + timedelta(hours=offset_hours)
    index["generated_at"] = generated_at.strftime("%Y-%m-%dT%H:%M:%SZ")
    index["index_id"] = content_hash
    
    # Write output
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(
        json.dumps(index, indent=2, sort_keys=True, ensure_ascii=True),
        encoding="utf-8"
    )
    
    return index


# ==============================================================================
# DRIFT RADAR SUMMARY
# ==============================================================================


def generate_drift_radar_summary(
    forecast: DriftForecast,
    top_k_files: int = 5,
    top_k_terms: int = 5,
) -> Dict[str, Any]:
    """
    Generate a concise drift radar summary for CI logs.
    
    PHASE II — NOT RUN IN PHASE I
    No uplift claims are made.
    Deterministic execution guaranteed.
    
    Args:
        forecast: DriftForecast object
        top_k_files: Number of top risk files to include
        top_k_terms: Number of top volatile terms to include
        
    Returns:
        Summary dictionary suitable for CI output
    """
    # Extract top files by risk
    top_files = forecast.high_risk_files[:top_k_files]
    
    # Extract top terms by volatility (risk score)
    top_terms = sorted(
        forecast.term_hotspots.items(),
        key=lambda x: (-x[1], x[0])
    )[:top_k_terms]
    
    # Build summary
    summary = {
        "radar_id": forecast.forecast_id[:16],
        "scan_count": forecast.input_scan_count,
        "overall_trend": forecast.trending_violations,
        "high_risk_file_count": forecast.high_risk_file_count,
        "total_drift_magnitude": round(forecast.total_drift_magnitude, 2),
        "top_risk_files": [
            {
                "file": f["file_path"],
                "risk": round(f["drift_risk_score"], 2),
                "trend": f["trend"],
            }
            for f in top_files
        ],
        "top_volatile_terms": [
            {"term": t[0], "volatility": round(t[1], 2)}
            for t in top_terms
        ],
        "message": _build_radar_message(forecast, top_k_files),
    }
    
    return summary


def _build_radar_message(forecast: DriftForecast, top_k: int) -> str:
    """Build human-readable radar message."""
    if forecast.high_risk_file_count == 0:
        return "Governance Drift Radar: No files with elevated drift risk detected."
    
    return (
        f"Governance Drift Radar: {forecast.high_risk_file_count} file(s) with elevated "
        f"drift risk (see artifacts/governance/drift_forecast.json)"
    )


def run_drift_radar(
    history_dir: Path,
    output_dir: Path,
    max_scans: int = 10,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Run drift radar analysis and produce summary.
    
    Args:
        history_dir: Directory with historical scan outputs
        output_dir: Directory for output artifacts
        max_scans: Maximum scans to analyze
        top_k: Number of top items to include in summary
        
    Returns:
        Radar summary dictionary
    """
    predictor = GovernanceDriftPredictor(history_dir)
    scan_count = predictor.load_scan_history(max_scans)
    
    if scan_count < 2:
        # Minimal forecast for insufficient data
        forecast = DriftForecast(
            forecast_id="insufficient_data",
            generated_at=DETERMINISTIC_EPOCH,
            input_scan_count=scan_count,
            total_drift_magnitude=0.0,
            average_drift_per_term=0.0,
            high_risk_file_count=0,
            trending_violations="insufficient_data",
            drift_vectors=[],
            high_risk_files=[],
            term_hotspots={},
            predicted_next_violations={},
            schema_mismatch_deltas={},
        )
    else:
        forecast = predictor.generate_forecast()
    
    # Write full forecast
    output_dir.mkdir(parents=True, exist_ok=True)
    forecast_path = output_dir / "drift_forecast.json"
    forecast_path.write_text(forecast.to_json(), encoding="utf-8")
    
    # Generate and write summary
    summary = generate_drift_radar_summary(forecast, top_k, top_k)
    summary_path = output_dir / "drift_radar_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8"
    )
    
    return summary


# ==============================================================================
# TERM STABILITY PROFILE
# ==============================================================================


@dataclass
class TermStabilityEntry:
    """Stability profile for a single governance term."""
    
    term: str
    stability_score: float  # [0, 1] where 1.0 = stable, 0.0 = unstable
    frequency_variance: float
    variant_count: int
    file_spread: int
    total_occurrences: int
    primary_files: List[str]
    variants: List[str]
    trend: str  # "stable", "increasing", "decreasing", "volatile"


def build_term_stability_profile(
    vocab_index: Dict[str, Any],
    history_window: int = 50,
    root_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Compute stability scores for each governance term over the last N commits.
    
    PHASE II — NOT RUN IN PHASE I
    No uplift claims are made.
    Deterministic execution guaranteed.
    
    Stability is computed based on:
      - frequency_variance: How much term usage varies across commits
      - variant_count: Number of different spellings/formats (more = less stable)
      - file_spread: Number of files using the term (more = more entrenched)
    
    Args:
        vocab_index: Governance vocabulary index from export_governance_vocabulary_index()
        history_window: Number of commits to analyze for stability
        root_path: Repository root path
        
    Returns:
        Dictionary with stability profiles for all terms
    """
    import re
    import subprocess
    import statistics
    
    root_path = root_path or Path.cwd()
    terms_data = vocab_index.get("terms", {})
    
    if not terms_data:
        return {
            "profile_id": "empty",
            "generated_at": DETERMINISTIC_EPOCH,
            "terms_analyzed": 0,
            "history_window": history_window,
            "profiles": {},
            "summary": {
                "stable_count": 0,
                "unstable_count": 0,
                "average_stability": 0.0,
            },
        }
    
    # Get commit history
    try:
        git_log = subprocess.run(
            ["git", "log", f"-{history_window}", "--format=%H", "--date-order"],
            cwd=root_path,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
        )
        commits = [c.strip() for c in git_log.stdout.strip().split("\n") if c.strip()]
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        commits = []
    
    profiles: Dict[str, Dict[str, Any]] = {}
    
    for term_name, term_data in sorted(terms_data.items()):
        # Build variants pattern
        all_variants = set(term_data.get("doc_variants", []))
        all_variants.update(term_data.get("code_variants", []))
        all_variants.add(term_name)
        
        variant_patterns = [re.escape(v) for v in sorted(all_variants)]
        combined_pattern = r'\b(' + '|'.join(variant_patterns) + r')\b'
        
        # Analyze term frequency across commits
        commit_frequencies: List[int] = []
        
        if commits:
            # Sample a subset of commits for performance (every 5th commit)
            sampled_commits = commits[::max(1, len(commits) // 10)] if len(commits) > 10 else commits
            
            for commit_hash in sampled_commits[:10]:  # Max 10 samples for performance
                total_count = 0
                
                # Check term occurrences in tracked files
                for scan_dir in DEFAULT_SCAN_DIRS:
                    try:
                        ls_tree = subprocess.run(
                            ["git", "ls-tree", "-r", "--name-only", commit_hash, scan_dir],
                            cwd=root_path,
                            capture_output=True,
                            text=True,
                            encoding="utf-8",
                            errors="replace",
                            check=True,
                        )
                        files = [f for f in ls_tree.stdout.strip().split("\n") if f]
                    except subprocess.CalledProcessError:
                        continue
                    
                    for file_path in files[:50]:  # Limit files per dir
                        try:
                            show = subprocess.run(
                                ["git", "show", f"{commit_hash}:{file_path}"],
                                cwd=root_path,
                                capture_output=True,
                                text=True,
                                encoding="utf-8",
                                errors="replace",
                                check=True,
                            )
                            content = show.stdout or ""
                            matches = re.findall(combined_pattern, content, re.IGNORECASE)
                            total_count += len(matches)
                        except (subprocess.CalledProcessError, TypeError, re.error):
                            continue
                
                commit_frequencies.append(total_count)
        
        # Compute stability metrics
        variant_count = len(all_variants)
        file_spread = term_data.get("file_count", 0)
        total_occurrences = term_data.get("total_occurrences", 0)
        
        # Compute frequency variance
        if len(commit_frequencies) >= 2:
            try:
                frequency_variance = statistics.variance(commit_frequencies)
                frequency_mean = statistics.mean(commit_frequencies)
                # Coefficient of variation (normalized variance)
                cv = (math.sqrt(frequency_variance) / frequency_mean) if frequency_mean > 0 else 0.0
            except statistics.StatisticsError:
                frequency_variance = 0.0
                cv = 0.0
        else:
            frequency_variance = 0.0
            cv = 0.0
        
        # Compute stability score [0, 1]
        # Factors that increase stability:
        #   - Low variance (cv close to 0)
        #   - High file spread (term is widely used)
        #   - Low variant count (consistent spelling)
        
        # Variance penalty: cv > 0.5 starts penalizing
        variance_factor = max(0.0, 1.0 - cv)
        
        # Variant penalty: more than 3 variants starts penalizing
        variant_factor = max(0.0, 1.0 - (variant_count - 3) * 0.1) if variant_count > 3 else 1.0
        
        # Spread bonus: more files = more stable (entrenched)
        spread_factor = min(1.0, file_spread / 50) if file_spread > 0 else 0.5
        
        # Combined stability score
        stability_score = (variance_factor * 0.4 + variant_factor * 0.3 + spread_factor * 0.3)
        stability_score = max(0.0, min(1.0, stability_score))
        
        # Determine trend
        if len(commit_frequencies) >= 2:
            first_half = commit_frequencies[:len(commit_frequencies)//2]
            second_half = commit_frequencies[len(commit_frequencies)//2:]
            first_avg = sum(first_half) / len(first_half) if first_half else 0
            second_avg = sum(second_half) / len(second_half) if second_half else 0
            
            if second_avg > first_avg * 1.2:
                trend = "increasing"
            elif second_avg < first_avg * 0.8:
                trend = "decreasing"
            elif cv > 0.5:
                trend = "volatile"
            else:
                trend = "stable"
        else:
            trend = "stable"
        
        # Get primary files (top 5 by occurrence)
        primary_files = term_data.get("files", [])[:5]
        
        profiles[term_name] = {
            "term": term_name,
            "stability_score": round(stability_score, 3),
            "frequency_variance": round(frequency_variance, 2),
            "variant_count": variant_count,
            "file_spread": file_spread,
            "total_occurrences": total_occurrences,
            "primary_files": primary_files,
            "variants": sorted(all_variants),
            "trend": trend,
        }
    
    # Compute summary statistics
    scores = [p["stability_score"] for p in profiles.values()]
    stable_count = sum(1 for s in scores if s >= 0.7)
    unstable_count = sum(1 for s in scores if s < 0.5)
    avg_stability = sum(scores) / len(scores) if scores else 0.0
    
    # Compute deterministic ID
    content_hash = hashlib.sha256(
        json.dumps(profiles, sort_keys=True).encode()
    ).hexdigest()
    
    from datetime import timedelta
    offset_hours = int(content_hash[:8], 16) % (24 * 365)
    generated_at = datetime(2025, 1, 1) + timedelta(hours=offset_hours)
    
    return {
        "profile_id": content_hash,
        "generated_at": generated_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "terms_analyzed": len(profiles),
        "history_window": history_window,
        "profiles": profiles,
        "summary": {
            "stable_count": stable_count,
            "unstable_count": unstable_count,
            "average_stability": round(avg_stability, 3),
        },
    }


# ==============================================================================
# GOVERNANCE WATCH LIST
# ==============================================================================


def export_governance_watch_list(
    profile: Dict[str, Any],
    threshold: float = 0.6,
    out_path: str = "artifacts/governance/governance_watch_list.json",
) -> Dict[str, Any]:
    """
    Export a watch-list of terms whose stability_score < threshold.
    
    PHASE II — NOT RUN IN PHASE I
    No uplift claims are made.
    Deterministic execution guaranteed.
    
    Args:
        profile: Term stability profile from build_term_stability_profile()
        threshold: Stability threshold (terms below this are on watch)
        out_path: Output file path
        
    Returns:
        Watch list dictionary
    """
    profiles = profile.get("profiles", {})
    
    # Find unstable terms
    watch_entries: List[Dict[str, Any]] = []
    
    for term_name, term_profile in sorted(profiles.items()):
        stability = term_profile.get("stability_score", 1.0)
        if stability < threshold:
            watch_entries.append({
                "term": term_name,
                "stability_score": stability,
                "variant_count": term_profile.get("variant_count", 0),
                "file_spread": term_profile.get("file_spread", 0),
                "primary_files": term_profile.get("primary_files", []),
                "variants": term_profile.get("variants", []),
                "trend": term_profile.get("trend", "unknown"),
                "risk_level": _compute_risk_level(stability),
            })
    
    # Sort by stability (lowest first = most concerning)
    watch_entries.sort(key=lambda x: (x["stability_score"], x["term"]))
    
    # Build watch list
    watch_list = {
        "watch_list_id": profile.get("profile_id", "unknown")[:16],
        "generated_at": profile.get("generated_at", DETERMINISTIC_EPOCH),
        "threshold": threshold,
        "watch_count": len(watch_entries),
        "entries": watch_entries,
        "top_5_volatile": [e["term"] for e in watch_entries[:5]],
        "message": _build_watch_message(watch_entries),
    }
    
    # Write output
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(
        json.dumps(watch_list, indent=2, sort_keys=True, ensure_ascii=True),
        encoding="utf-8"
    )
    
    return watch_list


def _compute_risk_level(stability: float) -> str:
    """Compute risk level from stability score."""
    if stability < 0.3:
        return "critical"
    elif stability < 0.5:
        return "high"
    elif stability < 0.7:
        return "moderate"
    else:
        return "low"


def _build_watch_message(entries: List[Dict[str, Any]]) -> str:
    """Build human-readable watch list message."""
    if not entries:
        return "Governance Watch List: All terms are stable."
    
    critical = sum(1 for e in entries if e["risk_level"] == "critical")
    high = sum(1 for e in entries if e["risk_level"] == "high")
    
    if critical > 0:
        return f"Governance Watch List: {critical} critical, {high} high-risk terms require attention."
    elif high > 0:
        return f"Governance Watch List: {high} high-risk terms require attention."
    else:
        return f"Governance Watch List: {len(entries)} term(s) below stability threshold."


# ==============================================================================
# DOC AUTHOR GOVERNANCE HINTS
# ==============================================================================


def generate_governance_hints(
    file_path: str,
    stability_profile: Dict[str, Any],
    vocab_index: Dict[str, Any],
    root_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Generate governance hints for a specific documentation file.
    
    PHASE II — NOT RUN IN PHASE I
    No uplift claims are made.
    Deterministic execution guaranteed.
    
    For a given file, identifies unstable or drift-prone terms and suggests:
      - Canonical variants
      - Which other files use the canonical form
    
    This is suggestion-only and never edits files.
    
    Args:
        file_path: Path to the file to analyze
        stability_profile: Term stability profile
        vocab_index: Governance vocabulary index
        root_path: Repository root path
        
    Returns:
        Hints dictionary with suggestions
    """
    import re
    
    root_path = root_path or Path.cwd()
    target_file = root_path / file_path if not Path(file_path).is_absolute() else Path(file_path)
    
    # Read file content
    try:
        content = target_file.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return {
            "file": file_path,
            "error": f"Could not read file: {e}",
            "hints": [],
            "summary": "File could not be analyzed.",
        }
    
    profiles = stability_profile.get("profiles", {})
    terms_data = vocab_index.get("terms", {})
    
    hints: List[Dict[str, Any]] = []
    
    # Check each term for usage in the file
    for term_name, term_profile in profiles.items():
        stability = term_profile.get("stability_score", 1.0)
        
        # Only generate hints for unstable terms (stability < 0.7)
        if stability >= 0.7:
            continue
        
        # Build pattern for this term's variants
        variants = term_profile.get("variants", [])
        if not variants:
            continue
        
        variant_patterns = [re.escape(v) for v in variants]
        combined_pattern = r'\b(' + '|'.join(variant_patterns) + r')\b'
        
        # Find occurrences in file
        try:
            matches = re.findall(combined_pattern, content, re.IGNORECASE)
        except (TypeError, re.error):
            continue
        
        if not matches:
            continue
        
        # Count variant usage
        variant_counts: Dict[str, int] = defaultdict(int)
        for match in matches:
            variant_counts[match] += 1
        
        # Determine canonical form (the term_name itself)
        canonical = term_name
        
        # Find files using canonical form
        term_data = terms_data.get(term_name, {})
        canonical_files = [
            f for f in term_data.get("files", [])
            if f != file_path
        ][:5]
        
        # Check if non-canonical variants are used
        non_canonical_used = [v for v in variant_counts.keys() if v.lower() != canonical.lower()]
        
        if non_canonical_used:
            hint = {
                "term": term_name,
                "stability_score": stability,
                "risk_level": _compute_risk_level(stability),
                "found_variants": dict(variant_counts),
                "canonical_form": canonical,
                "suggestion": f"Consider using canonical form '{canonical}' for consistency.",
                "files_using_canonical": canonical_files,
                "total_occurrences_in_file": sum(variant_counts.values()),
            }
            hints.append(hint)
    
    # Sort hints by stability (lowest first)
    hints.sort(key=lambda x: (x["stability_score"], x["term"]))
    
    # Build summary
    if not hints:
        summary = "No governance hints: all terms used are stable or use canonical forms."
    else:
        summary = f"Found {len(hints)} term(s) that may benefit from using canonical forms."
    
    return {
        "file": file_path,
        "hints_count": len(hints),
        "hints": hints,
        "summary": summary,
    }


# ==============================================================================
# GOVERNANCE DRIFT CHRONICLE (Longitudinal Snapshots)
# ==============================================================================


def compare_watch_lists(
    old: Dict[str, Any],
    new: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compare two watch lists to surface changes over time.
    
    PHASE II — NOT RUN IN PHASE I
    No uplift claims are made.
    Deterministic execution guaranteed.
    
    Produces a chronicle of changes between two watch list snapshots,
    identifying added/removed terms and risk level transitions.
    
    Args:
        old: Previous watch list dictionary (or empty dict if first run)
        new: Current watch list dictionary
        
    Returns:
        Chronicle dictionary with deterministic ordering
    """
    # Extract term sets
    old_entries = {e["term"]: e for e in old.get("entries", [])}
    new_entries = {e["term"]: e for e in new.get("entries", [])}
    
    old_terms = set(old_entries.keys())
    new_terms = set(new_entries.keys())
    
    # Compute additions and removals
    added_terms = sorted(new_terms - old_terms)
    removed_terms = sorted(old_terms - new_terms)
    
    # Compute risk level changes for terms present in both
    common_terms = old_terms & new_terms
    risk_level_upgrades: List[Dict[str, str]] = []
    risk_level_downgrades: List[Dict[str, str]] = []
    
    # Risk level ordering (lower index = more severe)
    risk_order = {"critical": 0, "high": 1, "moderate": 2, "low": 3}
    
    for term in sorted(common_terms):
        old_risk = old_entries[term].get("risk_level", "low")
        new_risk = new_entries[term].get("risk_level", "low")
        
        if old_risk != new_risk:
            old_severity = risk_order.get(old_risk, 3)
            new_severity = risk_order.get(new_risk, 3)
            
            change_entry = {"term": term, "from": old_risk, "to": new_risk}
            
            if new_severity < old_severity:
                # Risk increased (e.g., moderate -> high)
                risk_level_upgrades.append(change_entry)
            else:
                # Risk decreased (e.g., high -> moderate)
                risk_level_downgrades.append(change_entry)
    
    # Compute deterministic chronicle ID
    chronicle_content = json.dumps({
        "old_id": old.get("watch_list_id"),
        "new_id": new.get("watch_list_id"),
        "added": added_terms,
        "removed": removed_terms,
        "upgrades": risk_level_upgrades,
        "downgrades": risk_level_downgrades,
    }, sort_keys=True)
    chronicle_id = hashlib.sha256(chronicle_content.encode()).hexdigest()
    
    # Compute deterministic timestamp
    from datetime import timedelta
    offset_hours = int(chronicle_id[:8], 16) % (24 * 365)
    generated_at = datetime(2025, 1, 1) + timedelta(hours=offset_hours)
    
    return {
        "schema_version": "1.0.0",
        "chronicle_id": chronicle_id,
        "generated_at": generated_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "previous_watch_list_id": old.get("watch_list_id") if old else None,
        "current_watch_list_id": new.get("watch_list_id"),
        "added_terms": added_terms,
        "removed_terms": removed_terms,
        "risk_level_upgrades": risk_level_upgrades,
        "risk_level_downgrades": risk_level_downgrades,
        "summary": _build_chronicle_summary(
            added_terms, removed_terms, risk_level_upgrades, risk_level_downgrades
        ),
    }


def _build_chronicle_summary(
    added: List[str],
    removed: List[str],
    upgrades: List[Dict[str, str]],
    downgrades: List[Dict[str, str]],
) -> str:
    """Build human-readable chronicle summary."""
    parts = []
    
    if not added and not removed and not upgrades and not downgrades:
        return "No changes detected between watch list snapshots."
    
    if added:
        parts.append(f"{len(added)} term(s) added to watch list")
    if removed:
        parts.append(f"{len(removed)} term(s) removed from watch list")
    if upgrades:
        parts.append(f"{len(upgrades)} term(s) with increased risk")
    if downgrades:
        parts.append(f"{len(downgrades)} term(s) with decreased risk")
    
    return "Changes: " + "; ".join(parts) + "."


# ==============================================================================
# GOVERNANCE RISK SNAPSHOT (Dashboard Contract)
# ==============================================================================


def build_governance_risk_snapshot(
    profile: Dict[str, Any],
    watch_list: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a compact governance risk snapshot for CI dashboards.
    
    PHASE II — NOT RUN IN PHASE I
    No uplift claims are made.
    Deterministic execution guaranteed.
    
    Produces a small JSON object summarizing current governance risk
    using only numerical metrics (no evaluative language).
    
    Args:
        profile: Term stability profile from build_term_stability_profile()
        watch_list: Watch list from export_governance_watch_list()
        
    Returns:
        Snapshot dictionary with counts and term lists
    """
    profiles = profile.get("profiles", {})
    entries = watch_list.get("entries", [])
    
    # Count by risk level
    risk_counts = {"critical": 0, "high": 0, "moderate": 0, "low": 0}
    for entry in entries:
        risk = entry.get("risk_level", "low")
        if risk in risk_counts:
            risk_counts[risk] += 1
    
    # Get top volatile terms (already sorted by stability in watch list)
    top_volatile = watch_list.get("top_5_volatile", [])
    
    # Compute deterministic snapshot ID
    snapshot_content = json.dumps({
        "profile_id": profile.get("profile_id"),
        "watch_list_id": watch_list.get("watch_list_id"),
    }, sort_keys=True)
    snapshot_id = hashlib.sha256(snapshot_content.encode()).hexdigest()
    
    # Compute deterministic timestamp
    from datetime import timedelta
    offset_hours = int(snapshot_id[:8], 16) % (24 * 365)
    generated_at = datetime(2025, 1, 1) + timedelta(hours=offset_hours)
    
    return {
        "schema_version": "1.0.0",
        "snapshot_id": snapshot_id,
        "generated_at": generated_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_terms": profile.get("terms_analyzed", 0),
        "watch_count": watch_list.get("watch_count", 0),
        "critical_count": risk_counts["critical"],
        "high_count": risk_counts["high"],
        "moderate_count": risk_counts["moderate"],
        "low_count": risk_counts["low"],
        "top_volatile_terms": top_volatile,
    }


# ==============================================================================
# GOVERNANCE RISK EVALUATION (Phase III)
# ==============================================================================


def evaluate_governance_risk(
    snapshot: Dict[str, Any],
    chronicle: Optional[Dict[str, Any]] = None,
    watch_list: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Evaluate governance risk level from snapshot and chronicle.
    
    PHASE III — NOT RUN IN PHASE I
    No uplift claims are made.
    Deterministic execution guaranteed.
    
    Aggregates risk indicators to produce a risk band (LOW, MEDIUM, HIGH)
    suitable for risk console dashboards.
    
    Args:
        snapshot: Governance risk snapshot from build_governance_risk_snapshot()
        chronicle: Optional chronicle from compare_watch_lists()
        watch_list: Optional watch list to check risk levels of added terms
    
    Returns:
        {
            "schema_version": "1.0.0",
            "risk_band": "LOW" | "MEDIUM" | "HIGH",
            "new_critical_terms": List[str],
            "risk_upgrades_count": int,
            "risk_downgrades_count": int,
            "summary": str
        }
    
    Risk band rules:
        - HIGH: any new critical terms OR many upgrades (>= 3)
        - MEDIUM: some high/moderate terms but no new critical
        - LOW: no critical/high in watch list
    """
    # Extract current risk counts from snapshot
    critical_count = snapshot.get("critical_count", 0)
    high_count = snapshot.get("high_count", 0)
    moderate_count = snapshot.get("moderate_count", 0)
    
    # Extract chronicle data if available
    new_critical_terms: List[str] = []
    risk_upgrades_count = 0
    risk_downgrades_count = 0
    
    if chronicle:
        added_terms = chronicle.get("added_terms", [])
        upgrades = chronicle.get("risk_level_upgrades", [])
        downgrades = chronicle.get("risk_level_downgrades", [])
        risk_upgrades_count = len(upgrades)
        risk_downgrades_count = len(downgrades)
        
        # Find new critical terms: terms in added_terms with risk_level == "critical"
        if watch_list:
            # Use watch list to check risk levels
            watch_entries = {e["term"]: e for e in watch_list.get("entries", [])}
            for term in added_terms:
                entry = watch_entries.get(term)
                if entry and entry.get("risk_level") == "critical":
                    new_critical_terms.append(term)
        else:
            # Heuristic: check if added terms upgraded to critical
            added_terms_set = set(added_terms)
            for upgrade in upgrades:
                term = upgrade.get("term", "")
                if upgrade.get("to") == "critical" and term in added_terms_set:
                    new_critical_terms.append(term)
    
    # Determine risk band
    # HIGH: any new critical terms OR many upgrades (>=3) OR current critical_count > 0
    # MEDIUM: some high/moderate terms but no new critical and fewer upgrades
    # LOW: no critical/high in watch list
    
    if new_critical_terms or risk_upgrades_count >= 3 or critical_count > 0:
        risk_band = "HIGH"
    elif high_count > 0 or risk_upgrades_count > 0:
        risk_band = "MEDIUM"
    else:
        risk_band = "LOW"
    
    # Build neutral summary
    summary_parts = []
    if new_critical_terms:
        summary_parts.append(f"{len(new_critical_terms)} new critical term(s) added")
    if risk_upgrades_count > 0:
        summary_parts.append(f"{risk_upgrades_count} term(s) with increased risk level")
    if risk_downgrades_count > 0:
        summary_parts.append(f"{risk_downgrades_count} term(s) with decreased risk level")
    if critical_count > 0:
        summary_parts.append(f"{critical_count} critical term(s) in watch list")
    if high_count > 0:
        summary_parts.append(f"{high_count} high-risk term(s) in watch list")
    
    if summary_parts:
        summary = "Risk indicators: " + "; ".join(summary_parts) + "."
    else:
        summary = "No significant risk indicators detected."
    
    return {
        "schema_version": "1.0.0",
        "risk_band": risk_band,
        "new_critical_terms": sorted(new_critical_terms),
        "risk_upgrades_count": risk_upgrades_count,
        "risk_downgrades_count": risk_downgrades_count,
        "summary": summary,
    }


def extract_governance_alerts(
    chronicle: Dict[str, Any],
    risk_eval: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Extract governance alerts from chronicle and risk evaluation.
    
    PHASE III — ALERT SURFACE
    Deterministic execution guaranteed.
    
    Args:
        chronicle: Chronicle from compare_watch_lists()
        risk_eval: Risk evaluation from evaluate_governance_risk()
        
    Returns:
        List of alert dictionaries, each with:
        - term: term name
        - old_risk: previous risk level (or None)
        - new_risk: current risk level (or None)
        - alert_kind: "new_critical" | "upgraded" | "removed"
        - message: short neutral string
    """
    alerts: List[Dict[str, Any]] = []
    
    # Check for new critical terms
    new_critical_terms = risk_eval.get("new_critical_terms", [])
    for term in sorted(new_critical_terms):
        alerts.append({
            "term": term,
            "old_risk": None,
            "new_risk": "critical",
            "alert_kind": "new_critical",
            "message": f"Term '{term}' added to watch list with critical risk level.",
        })
    
    # Check for risk level upgrades
    risk_level_upgrades = chronicle.get("risk_level_upgrades", [])
    for upgrade in sorted(risk_level_upgrades, key=lambda x: (x.get("term", ""), x.get("from", ""))):
        term = upgrade.get("term", "")
        old_risk = upgrade.get("from", "")
        new_risk = upgrade.get("to", "")
        alerts.append({
            "term": term,
            "old_risk": old_risk,
            "new_risk": new_risk,
            "alert_kind": "upgraded",
            "message": f"Risk level for '{term}' increased from {old_risk} to {new_risk}.",
        })
    
    # Check for removed terms
    removed_terms = chronicle.get("removed_terms", [])
    for term in removed_terms:
        alerts.append({
            "term": term,
            "old_risk": None,  # We don't track old risk for removed terms
            "new_risk": None,
            "alert_kind": "removed",
            "message": f"Term '{term}' removed from watch list",
        })
    
    # Sort by alert kind priority, then term name for deterministic ordering
    kind_priority = {"new_critical": 0, "upgraded": 1, "removed": 2}
    alerts.sort(key=lambda a: (kind_priority.get(a["alert_kind"], 99), a["term"]))
    
    return alerts


def summarize_governance_drift_for_global_health(
    risk_eval: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Summarize governance drift for global health monitoring.
    
    PHASE III — GLOBAL HEALTH SIGNAL
    Deterministic execution guaranteed.
    
    Provides a compact signal suitable for MAAS and Director's console.
    
    Args:
        risk_eval: Risk evaluation from evaluate_governance_risk()
        
    Returns:
        Compact summary dictionary with:
        - governance_drift_status: "OK" | "ATTENTION" | "HOT"
        - critical_watch_terms_count: number of critical terms
        - risk_band: "LOW" | "MEDIUM" | "HIGH"
    """
    risk_band = risk_eval.get("risk_band", "LOW")
    new_critical_terms = risk_eval.get("new_critical_terms", [])
    risk_upgrades_count = risk_eval.get("risk_upgrades_count", 0)
    
    # Map risk band to status
    # HOT: HIGH risk band OR new critical terms OR many upgrades
    # ATTENTION: MEDIUM risk band OR some upgrades
    # OK: LOW risk band and no significant changes
    if risk_band == "HIGH" or new_critical_terms or risk_upgrades_count >= 3:
        status = "HOT"
    elif risk_band == "MEDIUM" or risk_upgrades_count > 0:
        status = "ATTENTION"
    else:
        status = "OK"
    
    critical_watch_terms_count = len(new_critical_terms)
    
    # Compute deterministic summary ID
    summary_content = json.dumps({
        "evaluation_id": risk_eval.get("evaluation_id"),
        "governance_drift_status": status,
    }, sort_keys=True)
    summary_id = hashlib.sha256(summary_content.encode()).hexdigest()
    
    # Compute deterministic timestamp
    from datetime import timedelta
    offset_hours = int(summary_id[:8], 16) % (24 * 365)
    generated_at = datetime(2025, 1, 1) + timedelta(hours=offset_hours)
    
    return {
        "schema_version": "1.0.0",
        "summary_id": summary_id,
        "generated_at": generated_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "governance_drift_status": status,
        "critical_watch_terms_count": critical_watch_terms_count,
        "risk_band": risk_band,
    }


# ==============================================================================
# GOVERNANCE DRIFT RADAR TIMELINE (Phase IV)
# ==============================================================================


def build_governance_drift_radar(
    risk_evaluations: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build governance drift radar timeline from sequence of risk evaluations.
    
    PHASE IV — NOT RUN IN PHASE I
    No uplift claims are made.
    Deterministic execution guaranteed.
    
    Analyzes a sequence of risk evaluations to produce trend analysis
    suitable for continuous monitoring dashboards.
    
    Args:
        risk_evaluations: Sequence of risk evaluation dictionaries from
                         evaluate_governance_risk()
        
    Returns:
        Radar timeline dictionary with trend_status, metrics, and summary
    """
    if not risk_evaluations:
        return {
            "schema_version": "1.0.0",
            "radar_id": "empty",
            "generated_at": DETERMINISTIC_EPOCH,
            "total_runs": 0,
            "runs_with_high_risk": 0,
            "runs_with_new_critical_terms": 0,
            "trend_status": "STABLE",
            "max_consecutive_high_runs": 0,
            "summary": "No risk evaluation data available.",
        }
    
    total_runs = len(risk_evaluations)
    runs_with_high_risk = sum(1 for e in risk_evaluations if e.get("risk_band") == "HIGH")
    runs_with_new_critical_terms = sum(
        1 for e in risk_evaluations if e.get("new_critical_terms", [])
    )
    
    # Compute trend status
    # IMPROVING: risk bands trending from HIGH → MEDIUM → LOW
    # DEGRADING: risk bands trending from LOW → MEDIUM → HIGH
    # STABLE: no clear trend or mixed pattern
    
    if total_runs < 2:
        trend_status = "STABLE"
    else:
        # Extract risk bands in chronological order
        risk_bands = [e.get("risk_band", "LOW") for e in risk_evaluations]
        
        # Map to numeric values for trend analysis
        band_values = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
        numeric_bands = [band_values.get(band, 0) for band in risk_bands]
        
        # Simple linear trend: compare first half vs second half
        mid_point = len(numeric_bands) // 2
        first_half_avg = sum(numeric_bands[:mid_point]) / len(numeric_bands[:mid_point]) if mid_point > 0 else 0
        second_half_avg = sum(numeric_bands[mid_point:]) / len(numeric_bands[mid_point:]) if len(numeric_bands[mid_point:]) > 0 else 0
        
        if second_half_avg < first_half_avg - 0.3:  # Significant decrease
            trend_status = "IMPROVING"
        elif second_half_avg > first_half_avg + 0.3:  # Significant increase
            trend_status = "DEGRADING"
        else:
            trend_status = "STABLE"
    
    # Compute max consecutive high runs
    max_consecutive_high = 0
    current_consecutive = 0
    for e in risk_evaluations:
        if e.get("risk_band") == "HIGH":
            current_consecutive += 1
            max_consecutive_high = max(max_consecutive_high, current_consecutive)
        else:
            current_consecutive = 0
    
    # Build neutral summary
    summary_parts = []
    if runs_with_high_risk > 0:
        summary_parts.append(f"{runs_with_high_risk} run(s) with HIGH risk band")
    if runs_with_new_critical_terms > 0:
        summary_parts.append(f"{runs_with_new_critical_terms} run(s) with new critical terms")
    if max_consecutive_high > 0:
        summary_parts.append(f"Maximum {max_consecutive_high} consecutive HIGH risk run(s)")
    
    if summary_parts:
        summary = "Radar analysis: " + "; ".join(summary_parts) + "."
    else:
        summary = "Radar analysis: No elevated risk indicators across evaluation timeline."
    
    # Compute deterministic radar ID
    radar_content = json.dumps({
        "total_runs": total_runs,
        "runs_with_high_risk": runs_with_high_risk,
        "trend_status": trend_status,
    }, sort_keys=True)
    radar_id = hashlib.sha256(radar_content.encode()).hexdigest()
    
    # Compute deterministic timestamp
    from datetime import timedelta
    offset_hours = int(radar_id[:8], 16) % (24 * 365)
    generated_at = datetime(2025, 1, 1) + timedelta(hours=offset_hours)
    
    return {
        "schema_version": "1.0.0",
        "radar_id": radar_id,
        "generated_at": generated_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_runs": total_runs,
        "runs_with_high_risk": runs_with_high_risk,
        "runs_with_new_critical_terms": runs_with_new_critical_terms,
        "trend_status": trend_status,
        "max_consecutive_high_runs": max_consecutive_high,
        "summary": summary,
    }


# ==============================================================================
# POLICY FEED FOR MAAS / GLOBAL HEALTH (Phase IV)
# ==============================================================================


def summarize_governance_radar_for_policy(
    radar: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Summarize governance radar for policy dashboard (MAAS / Global Health).
    
    PHASE IV — NOT RUN IN PHASE I
    No uplift claims are made.
    Deterministic execution guaranteed.
    
    Produces a compact policy feed suitable for governance dashboards
    and automated policy decision systems.
    
    Args:
        radar: Radar timeline from build_governance_drift_radar()
        
    Returns:
        Policy summary dictionary with attention flags, status, and review terms
    """
    total_runs = radar.get("total_runs", 0)
    runs_with_high_risk = radar.get("runs_with_high_risk", 0)
    runs_with_new_critical_terms = radar.get("runs_with_new_critical_terms", 0)
    trend_status = radar.get("trend_status", "STABLE")
    max_consecutive_high = radar.get("max_consecutive_high_runs", 0)
    
    # Determine if policy attention is required
    # Attention required if: high risk runs OR degrading trend OR consecutive high runs
    policy_attention_required = (
        runs_with_high_risk > 0 or
        trend_status == "DEGRADING" or
        max_consecutive_high >= 2
    )
    
    # Determine status
    if runs_with_new_critical_terms > 0 or max_consecutive_high >= 3 or trend_status == "DEGRADING":
        status = "HOT"
    elif policy_attention_required:
        status = "ATTENTION"
    else:
        status = "OK"
    
    # Extract key terms to review (from recent high-risk evaluations)
    # Note: This would ideally come from the risk_evaluations, but we only have
    # the aggregated radar. For now, we'll note that terms should be reviewed
    # if there were new critical terms or high risk runs.
    key_terms_to_review: List[str] = []
    # In a full implementation, we'd extract terms from the risk_evaluations
    # For now, we provide a placeholder that indicates review is needed
    
    # Build neutral notes
    notes: List[str] = []
    if total_runs > 0:
        notes.append(f"Analyzed {total_runs} risk evaluation run(s)")
    if runs_with_high_risk > 0:
        notes.append(f"{runs_with_high_risk} run(s) classified as HIGH risk")
    if trend_status != "STABLE":
        notes.append(f"Trend status: {trend_status}")
    if max_consecutive_high > 0:
        notes.append(f"Maximum {max_consecutive_high} consecutive HIGH risk run(s)")
    
    if not notes:
        notes.append("No significant governance risk indicators detected")
    
    # Compute deterministic policy summary ID
    policy_content = json.dumps({
        "radar_id": radar.get("radar_id"),
        "policy_attention_required": policy_attention_required,
        "status": status,
    }, sort_keys=True)
    policy_id = hashlib.sha256(policy_content.encode()).hexdigest()
    
    # Compute deterministic timestamp
    from datetime import timedelta
    offset_hours = int(policy_id[:8], 16) % (24 * 365)
    generated_at = datetime(2025, 1, 1) + timedelta(hours=offset_hours)
    
    return {
        "schema_version": "1.0.0",
        "policy_summary_id": policy_id,
        "generated_at": generated_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "policy_attention_required": policy_attention_required,
        "status": status,
        "key_terms_to_review": sorted(key_terms_to_review),
        "notes": notes,
    }


# ==============================================================================
# PR REVIEW HOOKS (Phase IV)
# ==============================================================================


def build_governance_review_hints_for_pr(
    risk_eval: Dict[str, Any],
    alerts: List[Dict[str, Any]],
    files_touched: List[str],
) -> Dict[str, Any]:
    """
    Build governance review hints for PR documentation and terminology changes.
    
    PHASE IV — NOT RUN IN PHASE I
    No uplift claims are made.
    Deterministic execution guaranteed.
    
    Provides reviewer guidance by matching governance alerts against
    files touched in a PR, highlighting terms that need attention.
    
    Args:
        risk_eval: Risk evaluation from evaluate_governance_risk()
        alerts: List of alerts from extract_governance_alerts()
        files_touched: List of file paths modified in the PR
        
    Returns:
        Review hints dictionary with highlight_terms, sections_to_review, summary_hint
    """
    # Extract terms from alerts
    alert_terms = {a.get("term", "") for a in alerts if a.get("term")}
    
    # Normalize file paths for matching (handle both forward and backslash)
    normalized_files = {
        f.replace("\\", "/") for f in files_touched if f
    }
    
    # Find terms that appear in both alerts and files touched
    # We'll check if any alert term appears in any file path
    highlight_terms: List[str] = []
    for term in sorted(alert_terms):
        # Check if term appears in any file path (simple substring match)
        # In practice, you'd want to check file contents, but for PR hooks,
        # matching against file paths is a reasonable heuristic
        term_lower = term.lower()
        for file_path in normalized_files:
            file_lower = file_path.lower()
            if term_lower in file_lower or term_lower.replace("_", "-") in file_lower:
                highlight_terms.append(term)
                break
    
    # Build sections to review
    sections_to_review: List[Dict[str, str]] = []
    
    # For each alert, if the term might be in touched files, add a review section
    for alert in alerts:
        term = alert.get("term", "")
        alert_kind = alert.get("alert_kind", "")
        
        # Find files that might contain this term
        matching_files = [
            f for f in normalized_files
            if term.lower() in f.lower() or term.lower().replace("_", "-") in f.lower()
        ]
        
        if matching_files:
            reason = _build_review_reason(alert_kind, alert)
            for file_path in sorted(matching_files)[:3]:  # Limit to 3 files per term
                sections_to_review.append({
                    "file": file_path,
                    "term": term,
                    "reason": reason,
                })
    
    # Build summary hint
    if highlight_terms:
        summary_hint = (
            f"This PR touches {len(highlight_terms)} governance term(s) "
            f"that may require terminology review: {', '.join(highlight_terms[:3])}."
        )
    elif risk_eval.get("risk_band") == "HIGH":
        summary_hint = "This PR may affect governance terminology; review recommended."
    else:
        summary_hint = "No governance terminology concerns detected for this PR."
    
    # Compute deterministic review hints ID
    review_content = json.dumps({
        "evaluation_id": risk_eval.get("evaluation_id"),
        "highlight_terms": sorted(highlight_terms),
        "files_count": len(files_touched),
    }, sort_keys=True)
    review_id = hashlib.sha256(review_content.encode()).hexdigest()
    
    # Compute deterministic timestamp
    from datetime import timedelta
    offset_hours = int(review_id[:8], 16) % (24 * 365)
    generated_at = datetime(2025, 1, 1) + timedelta(hours=offset_hours)
    
    return {
        "schema_version": "1.0.0",
        "review_hints_id": review_id,
        "generated_at": generated_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "highlight_terms": sorted(highlight_terms),
        "sections_to_review": sections_to_review,
        "summary_hint": summary_hint,
    }


def _build_review_reason(alert_kind: str, alert: Dict[str, Any]) -> str:
    """Build neutral review reason from alert kind."""
    term = alert.get("term", "")
    
    if alert_kind == "new_critical":
        return f"Term '{term}' recently added to watch list with critical risk level."
    elif alert_kind == "upgraded":
        old_risk = alert.get("old_risk", "")
        new_risk = alert.get("new_risk", "")
        return f"Term '{term}' risk level increased from {old_risk} to {new_risk}."
    elif alert_kind == "removed":
        return f"Term '{term}' recently removed from watch list."
    else:
        return f"Term '{term}' flagged for governance review."


# ==============================================================================
# GOVERNANCE RISK BUDGETING (Phase V)
# ==============================================================================


def build_governance_risk_budget(
    radar: Dict[str, Any],
    *,
    max_high_runs: int = 3,
    max_new_critical_terms: int = 5,
) -> Dict[str, Any]:
    """
    Build governance risk budget from radar timeline.
    
    PHASE V — NOT RUN IN PHASE I
    No uplift claims are made.
    Deterministic execution guaranteed.
    
    Provides a budget view tracking whether governance risk indicators
    are within acceptable limits. Purely descriptive with no side effects.
    
    Args:
        radar: Radar timeline from build_governance_drift_radar()
        max_high_runs: Maximum allowed high-risk runs (default: 3)
        max_new_critical_terms: Maximum allowed new critical terms (default: 5)
        
    Returns:
        Budget dictionary with status, remaining budget, and neutral notes
    """
    runs_with_high_risk = radar.get("runs_with_high_risk", 0)
    runs_with_new_critical_terms = radar.get("runs_with_new_critical_terms", 0)
    max_consecutive_high = radar.get("max_consecutive_high_runs", 0)
    
    # Compute remaining budget
    remaining_high_runs = max(0, max_high_runs - runs_with_high_risk)
    remaining_new_critical_terms = max(0, max_new_critical_terms - runs_with_new_critical_terms)
    
    # Determine budget status
    # EXCEEDED: over either limit
    # NEARING_LIMIT: at 80% of either limit
    # OK: within limits
    
    high_runs_ratio = runs_with_high_risk / max_high_runs if max_high_runs > 0 else 0
    critical_terms_ratio = runs_with_new_critical_terms / max_new_critical_terms if max_new_critical_terms > 0 else 0
    max_ratio = max(high_runs_ratio, critical_terms_ratio)
    
    if runs_with_high_risk > max_high_runs or runs_with_new_critical_terms > max_new_critical_terms:
        status = "EXCEEDED"
        budget_ok = False
    elif max_ratio >= 0.8:  # At 80% of limit
        status = "NEARING_LIMIT"
        budget_ok = True
    else:
        status = "OK"
        budget_ok = True
    
    # Build neutral notes
    notes: List[str] = []
    notes.append(f"High-risk runs: {runs_with_high_risk} of {max_high_runs} limit")
    notes.append(f"New critical terms runs: {runs_with_new_critical_terms} of {max_new_critical_terms} limit")
    if max_consecutive_high > 0:
        notes.append(f"Maximum consecutive high-risk runs: {max_consecutive_high}")
    
    # Compute deterministic budget ID
    budget_content = json.dumps({
        "radar_id": radar.get("radar_id"),
        "max_high_runs": max_high_runs,
        "max_new_critical_terms": max_new_critical_terms,
        "status": status,
    }, sort_keys=True)
    budget_id = hashlib.sha256(budget_content.encode()).hexdigest()
    
    # Compute deterministic timestamp
    from datetime import timedelta
    offset_hours = int(budget_id[:8], 16) % (24 * 365)
    generated_at = datetime(2025, 1, 1) + timedelta(hours=offset_hours)
    
    return {
        "schema_version": "1.0.0",
        "budget_id": budget_id,
        "generated_at": generated_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "budget_ok": budget_ok,
        "remaining_high_runs": remaining_high_runs,
        "remaining_new_critical_terms": remaining_new_critical_terms,
        "status": status,
        "neutral_notes": notes,
    }


# ==============================================================================
# BRANCH PROTECTION ADAPTER (Phase V)
# ==============================================================================


def evaluate_governance_for_branch_protection(
    radar: Dict[str, Any],
    policy_summary: Dict[str, Any],
    risk_budget: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Evaluate governance status for branch protection decisions.
    
    PHASE V — NOT RUN IN PHASE I
    No uplift claims are made.
    Deterministic execution guaranteed.
    
    Translates radar, policy, and budget views into branch protection signals
    suitable for CI pipelines on main/release branches.
    
    Args:
        radar: Radar timeline from build_governance_drift_radar()
        policy_summary: Policy summary from summarize_governance_radar_for_policy()
        risk_budget: Risk budget from build_governance_risk_budget()
        
    Returns:
        Branch protection evaluation with status, blocking reasons, and advisory notes
    """
    policy_status = policy_summary.get("status", "OK")
    budget_status = risk_budget.get("status", "OK")
    budget_ok = risk_budget.get("budget_ok", True)
    
    # Determine branch protection status
    # BLOCK: policy HOT OR budget EXCEEDED
    # WARN: policy ATTENTION OR budget NEARING_LIMIT
    # OK: all clear
    
    blocking_reasons: List[str] = []
    advisory_notes: List[str] = []
    
    if policy_status == "HOT" or budget_status == "EXCEEDED":
        status = "BLOCK"
        branch_safe = False
        
        if policy_status == "HOT":
            blocking_reasons.append("Policy status is HOT")
        if budget_status == "EXCEEDED":
            blocking_reasons.append("Risk budget exceeded")
    elif policy_status == "ATTENTION" or budget_status == "NEARING_LIMIT":
        status = "WARN"
        branch_safe = True
        
        if policy_status == "ATTENTION":
            advisory_notes.append("Policy status is ATTENTION")
        if budget_status == "NEARING_LIMIT":
            advisory_notes.append("Risk budget nearing limit")
    else:
        status = "OK"
        branch_safe = True
        advisory_notes.append("No governance blocking conditions detected")
    
    # Add radar context to advisory notes
    trend_status = radar.get("trend_status", "STABLE")
    if trend_status != "STABLE":
        advisory_notes.append(f"Radar trend: {trend_status}")
    
    # Compute deterministic branch protection ID
    branch_content = json.dumps({
        "radar_id": radar.get("radar_id"),
        "policy_summary_id": policy_summary.get("policy_summary_id"),
        "budget_id": risk_budget.get("budget_id"),
        "status": status,
    }, sort_keys=True)
    branch_id = hashlib.sha256(branch_content.encode()).hexdigest()
    
    # Compute deterministic timestamp
    from datetime import timedelta
    offset_hours = int(branch_id[:8], 16) % (24 * 365)
    generated_at = datetime(2025, 1, 1) + timedelta(hours=offset_hours)
    
    return {
        "schema_version": "1.0.0",
        "branch_protection_id": branch_id,
        "generated_at": generated_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "branch_safe": branch_safe,
        "status": status,
        "blocking_reasons": blocking_reasons,
        "advisory_notes": advisory_notes,
    }


# ==============================================================================
# PR COMMENT RENDERER (Phase V)
# ==============================================================================


def render_governance_pr_comment(
    review_hints: Dict[str, Any],
    radar_summary: Dict[str, Any],
) -> str:
    """
    Render governance review hints as a PR comment in Markdown.
    
    PHASE V — NOT RUN IN PHASE I
    No uplift claims are made.
    Deterministic execution guaranteed.
    
    Produces a Markdown string suitable for posting as a PR comment
    with radar status, highlighted terms, and sections to review.
    
    Args:
        review_hints: Review hints from build_governance_review_hints_for_pr()
        radar_summary: Radar timeline from build_governance_drift_radar()
        
    Returns:
        Markdown-formatted string for PR comment
    """
    lines: List[str] = []
    
    # Header
    lines.append("## 📋 Governance Terminology Review")
    lines.append("")
    
    # Radar status summary
    trend_status = radar_summary.get("trend_status", "STABLE")
    trend_emoji = {
        "STABLE": "🟢",
        "IMPROVING": "📈",
        "DEGRADING": "📉",
    }.get(trend_status, "⚪")
    
    lines.append(f"**Radar Status:** {trend_emoji} {trend_status}")
    lines.append("")
    
    # Summary hint
    summary_hint = review_hints.get("summary_hint", "")
    if summary_hint:
        lines.append(summary_hint)
        lines.append("")
    
    # Highlighted terms
    highlight_terms = review_hints.get("highlight_terms", [])
    if highlight_terms:
        lines.append("### ⚠️ Terms Requiring Attention")
        lines.append("")
        for term in highlight_terms:
            lines.append(f"- `{term}`")
        lines.append("")
    
    # Sections to review
    sections_to_review = review_hints.get("sections_to_review", [])
    if sections_to_review:
        lines.append("### 📝 Sections to Review")
        lines.append("")
        lines.append("| File | Term | Reason |")
        lines.append("|------|------|--------|")
        
        for section in sections_to_review:
            file_path = section.get("file", "")
            term = section.get("term", "")
            reason = section.get("reason", "")
            # Escape pipe characters in reason for table
            reason_escaped = reason.replace("|", "\\|")
            lines.append(f"| `{file_path}` | `{term}` | {reason_escaped} |")
        
        lines.append("")
    
    # Footer
    lines.append("---")
    lines.append("*Generated by E1 — Governance Drift Cartographer*")
    
    return "\n".join(lines)


# ==============================================================================
# EPISTEMIC ALIGNMENT TENSOR (Phase VI)
# ==============================================================================


def build_epistemic_alignment_tensor(
    semantic_panel: Dict[str, Any],
    curriculum_panel: Dict[str, Any],
    metric_readiness_matrix: Dict[str, Any],
    drift_multi_axis_view: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build unified epistemic alignment tensor from multi-domain panels.
    
    PHASE VI — NOT RUN IN PHASE I
    No uplift claims are made.
    Deterministic execution guaranteed.
    
    Combines semantic, curriculum, metric, and drift views into a unified
    alignment tensor for cross-domain stability analysis.
    
    Args:
        semantic_panel: Semantic director panel with status_light, alignment_status, etc.
        curriculum_panel: Curriculum alignment panel (expected structure similar to semantic)
        metric_readiness_matrix: Metric readiness matrix with slice → metric → status mappings
        drift_multi_axis_view: Multi-axis drift view from governance drift radar
        
    Returns:
        Alignment tensor dictionary with slice_axis, system_axes, tensor norm, and hotspots
    """
    # Extract semantic axis score [0, 1] (higher = healthier)
    semantic_status_light = semantic_panel.get("semantic_status_light", "RED")
    semantic_scores = {"GREEN": 1.0, "YELLOW": 0.5, "RED": 0.0}
    semantic_axis = semantic_scores.get(semantic_status_light, 0.0)
    
    # Extract curriculum axis score [0, 1]
    curriculum_status = curriculum_panel.get("status_light", curriculum_panel.get("curriculum_status", "RED"))
    curriculum_scores = {"GREEN": 1.0, "YELLOW": 0.5, "RED": 0.0, "ALIGNED": 1.0, "PARTIAL": 0.5, "DIVERGENT": 0.0}
    curriculum_axis = curriculum_scores.get(curriculum_status, 0.0)
    
    # Extract metrics axis score [0, 1]
    # Aggregate from metric_readiness_matrix: count ready vs total
    matrix_data = metric_readiness_matrix.get("matrix", {})
    total_metrics = 0
    ready_metrics = 0
    
    for slice_name, slice_metrics in matrix_data.items():
        for metric_kind, metric_status in slice_metrics.items():
            total_metrics += 1
            status = metric_status.get("status", "BLOCKED")
            if status == "READY":
                ready_metrics += 1
    
    metrics_axis = ready_metrics / total_metrics if total_metrics > 0 else 0.0
    
    # Extract drift axis score [0, 1] (invert: lower drift = higher score)
    trend_status = drift_multi_axis_view.get("trend_status", "DEGRADING")
    drift_scores = {"IMPROVING": 1.0, "STABLE": 0.7, "DEGRADING": 0.3}
    drift_axis = drift_scores.get(trend_status, 0.3)
    
    # Build slice_axis: per-slice alignment scores
    slice_axis: Dict[str, float] = {}
    
    # Get slices from metric_readiness_matrix
    slices = set(matrix_data.keys())
    
    # Also check semantic_panel for slice information
    semantic_slices = semantic_panel.get("slice_alignment", {})
    if isinstance(semantic_slices, dict):
        slices.update(semantic_slices.keys())
    
    # Compute per-slice scores
    for slice_name in sorted(slices):
        # Semantic score for this slice
        slice_semantic = semantic_slices.get(slice_name, {}) if isinstance(semantic_slices, dict) else {}
        slice_semantic_score = slice_semantic.get("score", 0.5) if isinstance(slice_semantic, dict) else 0.5
        
        # Metric score for this slice
        slice_metrics = matrix_data.get(slice_name, {})
        slice_total = len(slice_metrics)
        slice_ready = sum(1 for m in slice_metrics.values() if m.get("status") == "READY")
        slice_metric_score = slice_ready / slice_total if slice_total > 0 else 0.5
        
        # Drift score for this slice (use overall drift for now)
        slice_drift_score = drift_axis
        
        # Curriculum score (use overall for now)
        slice_curriculum_score = curriculum_axis
        
        # Combined slice score (weighted average)
        slice_axis[slice_name] = (
            slice_semantic_score * 0.3 +
            slice_metric_score * 0.3 +
            slice_drift_score * 0.2 +
            slice_curriculum_score * 0.2
        )
    
    # Compute tensor norm (L2 norm of system axes)
    alignment_tensor_norm = math.sqrt(
        semantic_axis ** 2 +
        curriculum_axis ** 2 +
        metrics_axis ** 2 +
        drift_axis ** 2
    ) / 2.0  # Normalize by max possible norm (sqrt(4) = 2)
    
    # Identify misalignment hotspots
    # Hotspots = slices with low semantic & low metric & high drift simultaneously
    hotspot_threshold = 0.4  # Low alignment threshold
    misalignment_hotspots: List[str] = []
    
    for slice_name, slice_score in slice_axis.items():
        slice_semantic = semantic_slices.get(slice_name, {}) if isinstance(semantic_slices, dict) else {}
        slice_semantic_val = slice_semantic.get("score", 0.5) if isinstance(slice_semantic, dict) else 0.5
        
        slice_metrics = matrix_data.get(slice_name, {})
        slice_total = len(slice_metrics)
        slice_ready = sum(1 for m in slice_metrics.values() if m.get("status") == "READY")
        slice_metric_val = slice_ready / slice_total if slice_total > 0 else 0.5
        
        # Check if low semantic AND low metric AND high drift (low drift_axis means high drift)
        if (slice_semantic_val < hotspot_threshold and
            slice_metric_val < hotspot_threshold and
            drift_axis < 0.5):  # High drift (low drift_axis)
            misalignment_hotspots.append(slice_name)
    
    # Sort hotspots deterministically
    misalignment_hotspots.sort()
    
    # Compute deterministic tensor ID
    tensor_content = json.dumps({
        "semantic_axis": round(semantic_axis, 3),
        "curriculum_axis": round(curriculum_axis, 3),
        "metrics_axis": round(metrics_axis, 3),
        "drift_axis": round(drift_axis, 3),
        "tensor_norm": round(alignment_tensor_norm, 3),
    }, sort_keys=True)
    tensor_id = hashlib.sha256(tensor_content.encode()).hexdigest()
    
    # Compute deterministic timestamp
    from datetime import timedelta
    offset_hours = int(tensor_id[:8], 16) % (24 * 365)
    generated_at = datetime(2025, 1, 1) + timedelta(hours=offset_hours)
    
    return {
        "schema_version": "1.0.0",
        "tensor_id": tensor_id,
        "generated_at": generated_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "slice_axis": {k: round(v, 3) for k, v in sorted(slice_axis.items())},
        "system_axes": {
            "semantic": round(semantic_axis, 3),
            "curriculum": round(curriculum_axis, 3),
            "metrics": round(metrics_axis, 3),
            "drift": round(drift_axis, 3),
        },
        "alignment_tensor_norm": round(alignment_tensor_norm, 3),
        "misalignment_hotspots": misalignment_hotspots,
    }


# ==============================================================================
# PREDICTIVE MISALIGNMENT FORECASTER (Phase VI)
# ==============================================================================


def forecast_epistemic_misalignment(
    alignment_tensor: Dict[str, Any],
    historical_alignment: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Forecast epistemic misalignment from current tensor and historical data.
    
    PHASE VI — NOT RUN IN PHASE I
    No uplift claims are made.
    Deterministic execution guaranteed.
    
    Predicts future misalignment risk based on tensor norm trends,
    multi-axis variance, and hotspot clustering.
    
    Args:
        alignment_tensor: Current alignment tensor from build_epistemic_alignment_tensor()
        historical_alignment: Optional list of previous alignment tensors for trend analysis
        
    Returns:
        Forecast dictionary with predicted_band, confidence, time_to_drift_event, explanation
    """
    current_norm = alignment_tensor.get("alignment_tensor_norm", 0.5)
    system_axes = alignment_tensor.get("system_axes", {})
    hotspots = alignment_tensor.get("misalignment_hotspots", [])
    
    # Compute multi-axis variance
    axis_values = [
        system_axes.get("semantic", 0.0),
        system_axes.get("curriculum", 0.0),
        system_axes.get("metrics", 0.0),
        system_axes.get("drift", 0.0),
    ]
    
    if len(axis_values) >= 2:
        import statistics
        try:
            axis_variance = statistics.variance(axis_values)
            axis_mean = statistics.mean(axis_values)
            # Coefficient of variation
            axis_cv = (math.sqrt(axis_variance) / axis_mean) if axis_mean > 0 else 0.0
        except statistics.StatisticsError:
            axis_cv = 0.0
    else:
        axis_cv = 0.0
    
    # Analyze tensor norm trend from history
    norm_trend = "stable"
    if historical_alignment and len(historical_alignment) >= 2:
        historical_norms = [t.get("alignment_tensor_norm", 0.5) for t in historical_alignment]
        if len(historical_norms) >= 2:
            first_half = historical_norms[:len(historical_norms)//2]
            second_half = historical_norms[len(historical_norms)//2:]
            first_avg = sum(first_half) / len(first_half) if first_half else 0.5
            second_avg = sum(second_half) / len(second_half) if second_half else 0.5
            
            if second_avg < first_avg - 0.1:
                norm_trend = "decreasing"
            elif second_avg > first_avg + 0.1:
                norm_trend = "increasing"
    
    # Hotspot clustering: count and density
    hotspot_count = len(hotspots)
    hotspot_density = hotspot_count / max(1, len(alignment_tensor.get("slice_axis", {})))
    
    # Predict misalignment band
    # HIGH: low norm OR decreasing trend OR high variance OR many hotspots
    # MEDIUM: moderate norm with some variance or hotspots
    # LOW: high norm, stable trend, low variance, few hotspots
    
    if (current_norm < 0.4 or
        norm_trend == "decreasing" or
        axis_cv > 0.5 or
        hotspot_density > 0.3):
        predicted_band = "HIGH"
    elif (current_norm < 0.6 or
          axis_cv > 0.3 or
          hotspot_density > 0.15):
        predicted_band = "MEDIUM"
    else:
        predicted_band = "LOW"
    
    # Compute confidence [0, 1]
    # Higher confidence with more historical data and consistent signals
    base_confidence = 0.5
    if historical_alignment:
        history_bonus = min(0.3, len(historical_alignment) * 0.05)
        base_confidence += history_bonus
    
    # Consistency bonus: if all signals point same direction
    signal_consistency = 1.0
    if (current_norm < 0.4 and norm_trend == "decreasing" and axis_cv > 0.5):
        signal_consistency = 1.2  # All signals agree
    elif (current_norm > 0.7 and norm_trend == "increasing" and axis_cv < 0.2):
        signal_consistency = 1.2
    
    confidence = min(1.0, base_confidence * signal_consistency)
    
    # Estimate time to drift event (in evaluation cycles)
    # Based on current norm and trend
    if predicted_band == "HIGH":
        if norm_trend == "decreasing":
            time_to_drift_event = 2  # Near-term risk
        else:
            time_to_drift_event = 5
    elif predicted_band == "MEDIUM":
        time_to_drift_event = 10
    else:
        time_to_drift_event = 20  # Low risk, far horizon
    
    # Build neutral explanation
    explanation: List[str] = []
    explanation.append(f"Current alignment tensor norm: {current_norm:.2f}")
    if historical_alignment:
        explanation.append(f"Trend analysis based on {len(historical_alignment)} historical evaluations")
    if norm_trend != "stable":
        explanation.append(f"Tensor norm trend: {norm_trend}")
    if axis_cv > 0.3:
        explanation.append(f"Multi-axis variance: {axis_cv:.2f}")
    if hotspot_count > 0:
        explanation.append(f"Misalignment hotspots detected: {hotspot_count} slice(s)")
    
    # Compute deterministic forecast ID
    forecast_content = json.dumps({
        "tensor_id": alignment_tensor.get("tensor_id"),
        "predicted_band": predicted_band,
        "current_norm": round(current_norm, 3),
    }, sort_keys=True)
    forecast_id = hashlib.sha256(forecast_content.encode()).hexdigest()
    
    # Compute deterministic timestamp
    from datetime import timedelta
    offset_hours = int(forecast_id[:8], 16) % (24 * 365)
    generated_at = datetime(2025, 1, 1) + timedelta(hours=offset_hours)
    
    return {
        "schema_version": "1.0.0",
        "forecast_id": forecast_id,
        "generated_at": generated_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "predicted_band": predicted_band,
        "confidence": round(confidence, 3),
        "time_to_drift_event": time_to_drift_event,
        "neutral_explanation": explanation,
    }


# ==============================================================================
# EPISTEMIC DIRECTOR PANEL (Phase VI)
# ==============================================================================


def build_epistemic_director_panel(
    alignment_tensor: Dict[str, Any],
    forecast: Dict[str, Any],
    structural_view: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build unified epistemic director panel for executive dashboard.
    
    PHASE VI — NOT RUN IN PHASE I
    No uplift claims are made.
    Deterministic execution guaranteed.
    
    Combines alignment tensor, forecast, and structural view into a single
    director panel suitable for high-level decision making.
    
    Args:
        alignment_tensor: Alignment tensor from build_epistemic_alignment_tensor()
        forecast: Forecast from forecast_epistemic_misalignment()
        structural_view: Structural governance view (expected with status_light or similar)
        
    Returns:
        Director panel with status_light, alignment_band, forecast_band, headline, flags
    """
    tensor_norm = alignment_tensor.get("alignment_tensor_norm", 0.5)
    predicted_band = forecast.get("predicted_band", "MEDIUM")
    structural_status = structural_view.get("status_light", structural_view.get("status", "YELLOW"))
    
    # Determine alignment band from tensor norm
    if tensor_norm >= 0.7:
        alignment_band = "HIGH"
    elif tensor_norm >= 0.4:
        alignment_band = "MEDIUM"
    else:
        alignment_band = "LOW"
    
    # Determine structural band
    structural_scores = {"GREEN": "HIGH", "YELLOW": "MEDIUM", "RED": "LOW"}
    structural_band = structural_scores.get(structural_status, "MEDIUM")
    
    # Determine overall status light
    # RED: low alignment OR high predicted misalignment OR structural RED
    # YELLOW: medium alignment OR medium predicted misalignment OR structural YELLOW
    # GREEN: high alignment AND low predicted misalignment AND structural GREEN
    
    if (alignment_band == "LOW" or
        predicted_band == "HIGH" or
        structural_status == "RED"):
        status_light = "RED"
    elif (alignment_band == "MEDIUM" or
          predicted_band == "MEDIUM" or
          structural_status == "YELLOW"):
        status_light = "YELLOW"
    else:
        status_light = "GREEN"
    
    # Build neutral headline
    if status_light == "RED":
        headline = "Epistemic alignment requires attention across multiple domains."
    elif status_light == "YELLOW":
        headline = "Epistemic alignment shows mixed signals across domains."
    else:
        headline = "Epistemic alignment indicators are within expected ranges."
    
    # Build flags (neutral, descriptive)
    flags: List[str] = []
    
    if tensor_norm < 0.4:
        flags.append(f"Alignment tensor norm below threshold: {tensor_norm:.2f}")
    
    hotspots = alignment_tensor.get("misalignment_hotspots", [])
    if hotspots:
        flags.append(f"Misalignment hotspots detected: {len(hotspots)} slice(s)")
    
    if predicted_band == "HIGH":
        flags.append(f"Forecast indicates elevated misalignment risk: {predicted_band}")
    
    confidence = forecast.get("confidence", 0.5)
    if confidence < 0.6:
        flags.append(f"Forecast confidence below threshold: {confidence:.2f}")
    
    if structural_status == "RED":
        flags.append("Structural governance status: RED")
    
    if not flags:
        flags.append("No elevated epistemic alignment concerns detected")
    
    # Compute deterministic panel ID
    panel_content = json.dumps({
        "tensor_id": alignment_tensor.get("tensor_id"),
        "forecast_id": forecast.get("forecast_id"),
        "status_light": status_light,
    }, sort_keys=True)
    panel_id = hashlib.sha256(panel_content.encode()).hexdigest()
    
    # Compute deterministic timestamp
    from datetime import timedelta
    offset_hours = int(panel_id[:8], 16) % (24 * 365)
    generated_at = datetime(2025, 1, 1) + timedelta(hours=offset_hours)
    
    return {
        "schema_version": "1.0.0",
        "panel_id": panel_id,
        "generated_at": generated_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status_light": status_light,
        "alignment_band": alignment_band,
        "forecast_band": predicted_band,
        "structural_band": structural_band,
        "headline": headline,
        "flags": flags,
    }


# ==============================================================================
# MARKDOWN HINTS OUTPUT
# ==============================================================================


def format_hints_as_markdown(hints_result: Dict[str, Any]) -> str:
    """
    Format governance hints as markdown for authors and reviewers.
    
    PHASE II — NOT RUN IN PHASE I
    No uplift claims are made.
    Deterministic execution guaranteed.
    
    Produces a markdown document with a table and suggestion list.
    Uses neutral language only (no "fix", "wrong", "error").
    
    Args:
        hints_result: Result from generate_governance_hints()
        
    Returns:
        Markdown-formatted string
    """
    lines: List[str] = []
    
    # Header
    lines.append("## Governance Terminology Hints")
    lines.append("")
    lines.append(f"**File:** `{hints_result.get('file', 'unknown')}`")
    lines.append("")
    
    # Handle error case
    if hints_result.get("error"):
        lines.append(f"⚠️ {hints_result['error']}")
        return "\n".join(lines)
    
    hints = hints_result.get("hints", [])
    
    if not hints:
        lines.append("✅ No terminology alignment suggestions for this file.")
        lines.append("")
        lines.append("All governance terms used are stable or already use canonical forms.")
        return "\n".join(lines)
    
    lines.append(f"Found **{len(hints)}** term(s) that may benefit from alignment with canonical forms.")
    lines.append("")
    
    # Summary table
    lines.append("### Summary")
    lines.append("")
    lines.append("| Term | Stability | Risk | Occurrences | Canonical | Example Files |")
    lines.append("|------|-----------|------|-------------|-----------|---------------|")
    
    for hint in hints:
        term = hint.get("term", "")
        stability = hint.get("stability_score", 0.0)
        risk = hint.get("risk_level", "unknown")
        occurrences = hint.get("total_occurrences_in_file", 0)
        canonical = hint.get("canonical_form", "")
        
        # Get example files (max 2 for table brevity)
        example_files = hint.get("files_using_canonical", [])[:2]
        files_str = ", ".join(f"`{f}`" for f in example_files) if example_files else "—"
        
        # Risk badge
        risk_badge = {
            "critical": "🔴",
            "high": "🟠",
            "moderate": "🟡",
            "low": "🟢",
        }.get(risk, "⚪")
        
        lines.append(
            f"| `{term}` | {stability:.2f} | {risk_badge} {risk} | {occurrences} | `{canonical}` | {files_str} |"
        )
    
    lines.append("")
    
    # Detailed suggestions
    lines.append("### Suggestions")
    lines.append("")
    
    for hint in hints:
        term = hint.get("term", "")
        canonical = hint.get("canonical_form", "")
        found_variants = hint.get("found_variants", {})
        
        lines.append(f"#### `{term}`")
        lines.append("")
        
        # List found variants
        if found_variants:
            variant_items = sorted(found_variants.items(), key=lambda x: (-x[1], x[0]))
            lines.append("**Variants found in file:**")
            for variant, count in variant_items:
                lines.append(f"- `{variant}` ({count} occurrence{'s' if count != 1 else ''})")
            lines.append("")
        
        # Neutral suggestion
        lines.append(f"**Suggestion:** Consider aligning with canonical form `{canonical}` for consistency across the codebase.")
        lines.append("")
        
        # Reference files
        ref_files = hint.get("files_using_canonical", [])
        if ref_files:
            lines.append("**Reference files using canonical form:**")
            for f in ref_files[:3]:
                lines.append(f"- `{f}`")
            lines.append("")
    
    # Footer
    lines.append("---")
    lines.append("*Generated by E1 — Governance Drift Cartographer*")
    
    return "\n".join(lines)


# ==============================================================================
# CLI INTERFACE
# ==============================================================================


def main() -> int:
    """
    Main entry point for governance drift predictor.
    
    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Governance Drift Prediction Engine - Phase II"
    )
    parser.add_argument(
        "--history-dir",
        type=Path,
        default=Path("artifacts/doc_sync_history"),
        help="Directory containing historical scan outputs",
    )
    parser.add_argument(
        "--max-scans",
        type=int,
        default=10,
        help="Maximum number of historical scans to analyze",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("governance_drift_forecast.json"),
        help="Output file for forecast JSON",
    )
    parser.add_argument(
        "--lint-pr",
        type=str,
        help="Lint a PR description file",
    )
    parser.add_argument(
        "--lint-commit",
        type=str,
        help="Lint a commit message file",
    )
    parser.add_argument(
        "--term-timeline",
        type=str,
        metavar="TERM",
        help="Build timeline for a governance term (e.g., 'Phase II', 'RFL')",
    )
    parser.add_argument(
        "--history",
        type=int,
        default=50,
        help="Number of commits to analyze for term timeline (default: 50)",
    )
    parser.add_argument(
        "--export-vocab-index",
        action="store_true",
        help="Export governance vocabulary index to artifacts/governance/",
    )
    parser.add_argument(
        "--radar",
        action="store_true",
        help="Run drift radar summary mode (informational, non-blocking)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top items to show in radar summary (default: 5)",
    )
    parser.add_argument(
        "--term-stability",
        action="store_true",
        help="Compute term stability profiles for all governance terms",
    )
    parser.add_argument(
        "--watch-list",
        action="store_true",
        help="Export governance watch list of unstable terms",
    )
    parser.add_argument(
        "--stability-threshold",
        type=float,
        default=0.6,
        help="Stability threshold for watch list (default: 0.6)",
    )
    parser.add_argument(
        "--hints",
        action="store_true",
        help="Generate governance hints for a specific file",
    )
    parser.add_argument(
        "--file",
        type=str,
        help="File path for --hints mode",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Output hints in markdown format (use with --hints)",
    )
    parser.add_argument(
        "--chronicle",
        action="store_true",
        help="Compare current watch list against previous snapshot",
    )
    parser.add_argument(
        "--previous-watch-list",
        type=str,
        help="Path to previous watch list JSON (for --chronicle)",
    )
    parser.add_argument(
        "--risk-snapshot",
        action="store_true",
        help="Generate governance risk snapshot for dashboards",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON only (for scripting)",
    )
    parser.add_argument(
        "--ci-mode",
        action="store_true",
        help="CI mode: exit with error only on errors, not warnings",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--epistemic-tensor",
        action="store_true",
        help="Build epistemic alignment tensor (Phase VI)",
    )
    parser.add_argument(
        "--semantic-panel",
        type=str,
        help="Path to semantic panel JSON (for --epistemic-tensor)",
    )
    parser.add_argument(
        "--curriculum-panel",
        type=str,
        help="Path to curriculum panel JSON (for --epistemic-tensor)",
    )
    parser.add_argument(
        "--metric-matrix",
        type=str,
        help="Path to metric readiness matrix JSON (for --epistemic-tensor)",
    )
    parser.add_argument(
        "--drift-view",
        type=str,
        help="Path to drift multi-axis view JSON (for --epistemic-tensor)",
    )
    parser.add_argument(
        "--forecast-misalignment",
        action="store_true",
        help="Forecast epistemic misalignment (Phase VI)",
    )
    parser.add_argument(
        "--alignment-tensor",
        type=str,
        help="Path to alignment tensor JSON (for --forecast-misalignment)",
    )
    parser.add_argument(
        "--historical-alignment",
        type=str,
        help="Path to historical alignment JSON array (for --forecast-misalignment)",
    )
    parser.add_argument(
        "--director-panel",
        action="store_true",
        help="Build epistemic director panel (Phase VI)",
    )
    parser.add_argument(
        "--forecast",
        type=str,
        help="Path to forecast JSON (for --director-panel)",
    )
    parser.add_argument(
        "--structural-view",
        type=str,
        help="Path to structural view JSON (for --director-panel)",
    )
    
    args = parser.parse_args()
    
    # JSON mode suppresses banner
    if not args.json:
        print("=" * 70)
        print("PHASE II — Governance Drift Prediction Engine")
        print("No uplift claims are made. Deterministic execution guaranteed.")
        print("=" * 70)
        print()
    
    # Handle term timeline
    if args.term_timeline:
        return _run_term_timeline(args.term_timeline, args.history, args.json)
    
    # Handle vocabulary index export
    if args.export_vocab_index:
        return _run_vocab_index_export(args.json)
    
    # Handle drift radar summary
    if args.radar:
        return _run_drift_radar(args.history_dir, args.max_scans, args.top_k, args.json)
    
    # Handle term stability profile
    if args.term_stability:
        return _run_term_stability(args.history, args.json)
    
    # Handle watch list export
    if args.watch_list:
        return _run_watch_list(args.history, args.stability_threshold, args.json)
    
    # Handle governance hints
    if args.hints:
        if not args.file:
            print("ERROR: --hints requires --file argument")
            return 1
        return _run_governance_hints(args.file, args.history, args.json, args.markdown)
    
    # Handle chronicle comparison
    if args.chronicle:
        return _run_chronicle(args.history, args.stability_threshold, args.previous_watch_list, args.json)
    
    # Handle risk snapshot
    if args.risk_snapshot:
        return _run_risk_snapshot(args.history, args.stability_threshold, args.json)
    
    # Handle epistemic alignment tensor (Phase VI)
    if args.epistemic_tensor:
        return _run_epistemic_tensor(
            args.semantic_panel,
            args.curriculum_panel,
            args.metric_matrix,
            args.drift_view,
            args.json,
        )
    
    # Handle forecast misalignment (Phase VI)
    if args.forecast_misalignment:
        return _run_forecast_misalignment(
            args.alignment_tensor,
            args.historical_alignment,
            args.json,
        )
    
    # Handle director panel (Phase VI)
    if args.director_panel:
        return _run_director_panel(
            args.alignment_tensor,
            args.forecast,
            args.structural_view,
            args.json,
        )
    
    # Handle PR/commit linting
    if args.lint_pr:
        return _lint_pr_file(args.lint_pr, args.ci_mode)
    
    if args.lint_commit:
        return _lint_commit_file(args.lint_commit, args.ci_mode)
    
    # Run drift prediction
    predictor = GovernanceDriftPredictor(args.history_dir)
    
    # Load history
    scan_count = predictor.load_scan_history(args.max_scans)
    print(f"Loaded {scan_count} historical scans from {args.history_dir}")
    
    if scan_count < 2:
        print("WARNING: Insufficient scan history for drift prediction")
        print("Need at least 2 scans to compute drift vectors")
        
        # Generate minimal forecast
        forecast = DriftForecast(
            forecast_id="insufficient_data",
            generated_at=DETERMINISTIC_EPOCH,
            input_scan_count=scan_count,
            total_drift_magnitude=0.0,
            average_drift_per_term=0.0,
            high_risk_file_count=0,
            trending_violations="insufficient_data",
            drift_vectors=[],
            high_risk_files=[],
            term_hotspots={},
            predicted_next_violations={},
            schema_mismatch_deltas={},
        )
    else:
        forecast = predictor.generate_forecast()
    
    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(forecast.to_json(), encoding="utf-8")
    print(f"Forecast written to: {args.output}")
    
    # Print summary
    print()
    print("DRIFT FORECAST SUMMARY")
    print("-" * 40)
    print(f"Forecast ID: {forecast.forecast_id[:16]}...")
    print(f"Input scans: {forecast.input_scan_count}")
    print(f"Total drift magnitude: {forecast.total_drift_magnitude:.2f}")
    print(f"Average drift per term: {forecast.average_drift_per_term:.2f}")
    print(f"High-risk files: {forecast.high_risk_file_count}")
    print(f"Violation trend: {forecast.trending_violations}")
    print()
    
    if args.verbose and forecast.drift_vectors:
        print("TOP DRIFT VECTORS:")
        for v in forecast.drift_vectors[:10]:
            print(f"  {v['term']}: risk={v['risk_score']:.2f}, delta={v['violation_delta']}")
        print()
    
    if args.verbose and forecast.high_risk_files:
        print("HIGH-RISK FILES:")
        for f in forecast.high_risk_files[:10]:
            print(f"  {f['file_path']}: risk={f['drift_risk_score']:.2f}, trend={f['trend']}")
        print()
    
    # CI mode: only fail on errors, not predictions
    if args.ci_mode:
        # Predictions are non-blocking
        print("CI GATE: PASSED (drift prediction is non-blocking)")
    
    return 0


def _lint_pr_file(pr_file: str, ci_mode: bool) -> int:
    """Lint a PR description file."""
    try:
        with open(pr_file, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"ERROR: Could not read PR file: {e}")
        return 1
    
    linter = GovernanceVocabularyLinter()
    result = linter.lint_pr_description(content)
    
    print(f"PR Description Lint: {result['summary']}")
    
    if result["violations"]:
        print("\nViolations:")
        for v in result["violations"]:
            print(f"  Line {v['line']}: '{v['found']}' -> '{v['canonical']}'")
            print(f"    Context: {v['context']}")
    
    if ci_mode and not result["passed"]:
        return 1
    
    return 0


def _lint_commit_file(commit_file: str, ci_mode: bool) -> int:
    """Lint a commit message file."""
    try:
        with open(commit_file, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"ERROR: Could not read commit file: {e}")
        return 1
    
    linter = GovernanceVocabularyLinter()
    result = linter.lint_commit_message(content)
    
    print(f"Commit Message Lint: {result['summary']}")
    
    if result["violations"]:
        print("\nViolations:")
        for v in result["violations"]:
            critical = "(CRITICAL)" if v["category"] in {"metric", "phase", "symbol"} else ""
            print(f"  Line {v['line']}: '{v['found']}' -> '{v['canonical']}' {critical}")
    
    if ci_mode and not result["passed"]:
        return 1
    
    return 0


def _run_term_timeline(term: str, history: int, json_output: bool) -> int:
    """Run term timeline analysis."""
    timeline = build_term_timeline(term, history_window=history)
    
    if json_output:
        print(json.dumps(timeline, indent=2, sort_keys=True))
    else:
        print(f"Term Timeline: '{term}'")
        print("-" * 40)
        print(f"Timeline ID: {timeline.get('timeline_id', 'N/A')[:16]}...")
        print(f"Commits analyzed: {timeline.get('commits_analyzed', 0)}")
        print(f"Variants tracked: {', '.join(timeline.get('variants_tracked', [])[:5])}")
        print()
        
        if timeline.get("error"):
            print(f"ERROR: {timeline['error']}")
            return 1
        
        entries = timeline.get("timeline", [])
        if entries:
            print("Recent commits (newest first):")
            # Show last 10 entries
            for entry in reversed(entries[-10:]):
                print(f"  {entry['commit']} ({entry['date'][:10]}): "
                      f"count={entry['term_count']}, files={entry['files_touched']}")
    
    return 0


def _run_vocab_index_export(json_output: bool) -> int:
    """Export governance vocabulary index."""
    out_path = "artifacts/governance/governance_vocabulary_index.json"
    
    index = export_governance_vocabulary_index(out_path)
    
    if json_output:
        print(json.dumps(index, indent=2, sort_keys=True))
    else:
        if index.get("error"):
            print(f"ERROR: {index['error']}")
            return 1
        
        print(f"Governance Vocabulary Index exported to: {out_path}")
        print("-" * 40)
        print(f"Index ID: {index.get('index_id', 'N/A')[:16]}...")
        print(f"Total terms: {index.get('total_terms', 0)}")
        
        # Show summary by category
        terms = index.get("terms", {})
        categories: Dict[str, int] = defaultdict(int)
        for term_data in terms.values():
            categories[term_data.get("category", "unknown")] += 1
        
        print("\nTerms by category:")
        for cat, count in sorted(categories.items()):
            print(f"  {cat}: {count}")
        
        # Show top 5 terms by occurrence
        top_terms = sorted(
            terms.items(),
            key=lambda x: -x[1].get("total_occurrences", 0)
        )[:5]
        
        print("\nTop 5 terms by occurrence:")
        for name, data in top_terms:
            print(f"  {name}: {data.get('total_occurrences', 0)} occurrences in {data.get('file_count', 0)} files")
    
    return 0


def _run_drift_radar(history_dir: Path, max_scans: int, top_k: int, json_output: bool) -> int:
    """Run drift radar summary."""
    output_dir = Path("artifacts/governance")
    
    summary = run_drift_radar(history_dir, output_dir, max_scans, top_k)
    
    if json_output:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print("GOVERNANCE DRIFT RADAR")
        print("=" * 50)
        print()
        print(summary["message"])
        print()
        print(f"Radar ID: {summary['radar_id']}")
        print(f"Scans analyzed: {summary['scan_count']}")
        print(f"Overall trend: {summary['overall_trend']}")
        print(f"High-risk files: {summary['high_risk_file_count']}")
        print(f"Total drift magnitude: {summary['total_drift_magnitude']}")
        print()
        
        if summary["top_risk_files"]:
            print(f"Top {top_k} Risk Files:")
            for f in summary["top_risk_files"]:
                print(f"  [{f['trend']}] {f['file']}: risk={f['risk']}")
            print()
        
        if summary["top_volatile_terms"]:
            print(f"Top {top_k} Volatile Terms:")
            for t in summary["top_volatile_terms"]:
                print(f"  {t['term']}: volatility={t['volatility']}")
            print()
        
        print(f"Full forecast: {output_dir / 'drift_forecast.json'}")
        print(f"Radar summary: {output_dir / 'drift_radar_summary.json'}")
    
    # Radar never fails CI (informational only)
    return 0


def _run_term_stability(history: int, json_output: bool) -> int:
    """Run term stability profile generation."""
    # First export vocab index
    vocab_index = export_governance_vocabulary_index(
        "artifacts/governance/governance_vocabulary_index.json"
    )
    
    if vocab_index.get("error"):
        print(f"ERROR: {vocab_index['error']}")
        return 1
    
    # Build stability profile
    profile = build_term_stability_profile(vocab_index, history_window=history)
    
    # Write to file
    out_path = Path("artifacts/governance/term_stability_profile.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(profile, indent=2, sort_keys=True, ensure_ascii=True),
        encoding="utf-8"
    )
    
    if json_output:
        print(json.dumps(profile, indent=2, sort_keys=True))
    else:
        print("TERM STABILITY PROFILE")
        print("=" * 50)
        print()
        print(f"Profile ID: {profile.get('profile_id', 'N/A')[:16]}...")
        print(f"Terms analyzed: {profile.get('terms_analyzed', 0)}")
        print(f"History window: {profile.get('history_window', 0)} commits")
        print()
        
        summary = profile.get("summary", {})
        print(f"Stable terms (≥0.7): {summary.get('stable_count', 0)}")
        print(f"Unstable terms (<0.5): {summary.get('unstable_count', 0)}")
        print(f"Average stability: {summary.get('average_stability', 0):.3f}")
        print()
        
        # Show top 5 most unstable terms
        profiles = profile.get("profiles", {})
        unstable = sorted(
            profiles.items(),
            key=lambda x: (x[1].get("stability_score", 1.0), x[0])
        )[:5]
        
        if unstable:
            print("Top 5 Unstable Terms:")
            for name, data in unstable:
                print(f"  {name}: stability={data.get('stability_score', 0):.3f}, "
                      f"variants={data.get('variant_count', 0)}, trend={data.get('trend', 'unknown')}")
        
        print()
        print(f"Output: {out_path}")
    
    return 0


def _run_watch_list(history: int, threshold: float, json_output: bool) -> int:
    """Run governance watch list export."""
    # First export vocab index
    vocab_index = export_governance_vocabulary_index(
        "artifacts/governance/governance_vocabulary_index.json"
    )
    
    if vocab_index.get("error"):
        print(f"ERROR: {vocab_index['error']}")
        return 1
    
    # Build stability profile
    profile = build_term_stability_profile(vocab_index, history_window=history)
    
    # Export watch list
    watch_list = export_governance_watch_list(
        profile,
        threshold=threshold,
        out_path="artifacts/governance/governance_watch_list.json"
    )
    
    if json_output:
        print(json.dumps(watch_list, indent=2, sort_keys=True))
    else:
        print("GOVERNANCE WATCH LIST")
        print("=" * 50)
        print()
        print(watch_list.get("message", ""))
        print()
        print(f"Watch List ID: {watch_list.get('watch_list_id', 'N/A')}")
        print(f"Stability threshold: {threshold}")
        print(f"Terms on watch: {watch_list.get('watch_count', 0)}")
        print()
        
        entries = watch_list.get("entries", [])
        if entries:
            print("Watch List Entries:")
            for entry in entries[:10]:  # Show top 10
                risk = entry.get("risk_level", "unknown")
                risk_badge = {"critical": "🔴", "high": "🟠", "moderate": "🟡", "low": "🟢"}.get(risk, "⚪")
                print(f"  {risk_badge} {entry['term']}: stability={entry['stability_score']:.3f}, "
                      f"variants={entry['variant_count']}")
            
            if len(entries) > 10:
                print(f"  ... and {len(entries) - 10} more")
        
        print()
        top_5 = watch_list.get("top_5_volatile", [])
        if top_5:
            print(f"Top 5 Volatile: {', '.join(top_5)}")
        
        print()
        print("Output: artifacts/governance/governance_watch_list.json")
    
    return 0


def _run_governance_hints(file_path: str, history: int, json_output: bool, markdown_output: bool = False) -> int:
    """Run governance hints for a specific file."""
    # First export vocab index
    vocab_index = export_governance_vocabulary_index(
        "artifacts/governance/governance_vocabulary_index.json"
    )
    
    if vocab_index.get("error"):
        print(f"ERROR: {vocab_index['error']}")
        return 1
    
    # Build stability profile
    profile = build_term_stability_profile(vocab_index, history_window=history)
    
    # Generate hints
    hints = generate_governance_hints(file_path, profile, vocab_index)
    
    if json_output:
        print(json.dumps(hints, indent=2, sort_keys=True))
    elif markdown_output:
        # Output markdown format
        markdown = format_hints_as_markdown(hints)
        print(markdown)
    else:
        print("GOVERNANCE HINTS")
        print("=" * 50)
        print()
        print(f"File: {hints.get('file', 'unknown')}")
        print()
        
        if hints.get("error"):
            print(f"ERROR: {hints['error']}")
            return 1
        
        print(hints.get("summary", ""))
        print()
        
        hint_list = hints.get("hints", [])
        if hint_list:
            print("Suggestions:")
            print("-" * 40)
            for hint in hint_list:
                risk = hint.get("risk_level", "unknown")
                risk_badge = {"critical": "🔴", "high": "🟠", "moderate": "🟡", "low": "🟢"}.get(risk, "⚪")
                print(f"\n{risk_badge} Term: {hint['term']} (stability: {hint['stability_score']:.3f})")
                print(f"   Found variants: {hint['found_variants']}")
                print(f"   Suggestion: {hint['suggestion']}")
                if hint.get("files_using_canonical"):
                    print(f"   Files using canonical form:")
                    for f in hint["files_using_canonical"][:3]:
                        print(f"     - {f}")
        else:
            print("✓ No suggestions - all terms are stable or use canonical forms.")
    
    return 0


def _run_chronicle(history: int, threshold: float, previous_path: Optional[str], json_output: bool) -> int:
    """Run governance drift chronicle comparison."""
    # First export vocab index
    vocab_index = export_governance_vocabulary_index(
        "artifacts/governance/governance_vocabulary_index.json"
    )
    
    if vocab_index.get("error"):
        print(f"ERROR: {vocab_index['error']}")
        return 1
    
    # Build stability profile
    profile = build_term_stability_profile(vocab_index, history_window=history)
    
    # Generate current watch list
    current_watch_list = export_governance_watch_list(
        profile,
        threshold=threshold,
        out_path="artifacts/governance/governance_watch_list.json"
    )
    
    # Load previous watch list if provided
    previous_watch_list: Dict[str, Any] = {}
    if previous_path:
        try:
            previous_watch_list = json.loads(Path(previous_path).read_text(encoding="utf-8"))
        except Exception as e:
            print(f"WARNING: Could not load previous watch list: {e}")
            previous_watch_list = {}
    
    # Generate chronicle
    chronicle = compare_watch_lists(previous_watch_list, current_watch_list)
    
    # Write chronicle to file
    chronicle_path = Path("artifacts/governance/governance_drift_chronicle.json")
    chronicle_path.parent.mkdir(parents=True, exist_ok=True)
    chronicle_path.write_text(
        json.dumps(chronicle, indent=2, sort_keys=True, ensure_ascii=True),
        encoding="utf-8"
    )
    
    if json_output:
        print(json.dumps(chronicle, indent=2, sort_keys=True))
    else:
        print("GOVERNANCE DRIFT CHRONICLE")
        print("=" * 50)
        print()
        print(chronicle.get("summary", ""))
        print()
        print(f"Chronicle ID: {chronicle.get('chronicle_id', 'N/A')[:16]}...")
        print(f"Previous Watch List: {chronicle.get('previous_watch_list_id') or 'None (first run)'}")
        print(f"Current Watch List: {chronicle.get('current_watch_list_id', 'N/A')[:16]}...")
        print()
        
        added = chronicle.get("added_terms", [])
        if added:
            print(f"Added to watch ({len(added)}):")
            for term in added[:5]:
                print(f"  + {term}")
            if len(added) > 5:
                print(f"  ... and {len(added) - 5} more")
            print()
        
        removed = chronicle.get("removed_terms", [])
        if removed:
            print(f"Removed from watch ({len(removed)}):")
            for term in removed[:5]:
                print(f"  - {term}")
            if len(removed) > 5:
                print(f"  ... and {len(removed) - 5} more")
            print()
        
        upgrades = chronicle.get("risk_level_upgrades", [])
        if upgrades:
            print(f"Risk increased ({len(upgrades)}):")
            for u in upgrades[:5]:
                print(f"  ⬆️ {u['term']}: {u['from']} → {u['to']}")
            print()
        
        downgrades = chronicle.get("risk_level_downgrades", [])
        if downgrades:
            print(f"Risk decreased ({len(downgrades)}):")
            for d in downgrades[:5]:
                print(f"  ⬇️ {d['term']}: {d['from']} → {d['to']}")
            print()
        
        print(f"Output: {chronicle_path}")
    
    return 0


def _run_risk_snapshot(history: int, threshold: float, json_output: bool) -> int:
    """Run governance risk snapshot generation."""
    # First export vocab index
    vocab_index = export_governance_vocabulary_index(
        "artifacts/governance/governance_vocabulary_index.json"
    )
    
    if vocab_index.get("error"):
        print(f"ERROR: {vocab_index['error']}")
        return 1
    
    # Build stability profile
    profile = build_term_stability_profile(vocab_index, history_window=history)
    
    # Generate watch list
    watch_list = export_governance_watch_list(
        profile,
        threshold=threshold,
        out_path="artifacts/governance/governance_watch_list.json"
    )
    
    # Build risk snapshot
    snapshot = build_governance_risk_snapshot(profile, watch_list)
    
    # Write snapshot to file
    snapshot_path = Path("artifacts/governance/governance_risk_snapshot.json")
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(
        json.dumps(snapshot, indent=2, sort_keys=True, ensure_ascii=True),
        encoding="utf-8"
    )
    
    if json_output:
        print(json.dumps(snapshot, indent=2, sort_keys=True))
    else:
        print("GOVERNANCE RISK SNAPSHOT")
        print("=" * 50)
        print()
        print(f"Snapshot ID: {snapshot.get('snapshot_id', 'N/A')[:16]}...")
        print()
        print(f"Total Terms: {snapshot.get('total_terms', 0)}")
        print(f"Watch Count: {snapshot.get('watch_count', 0)}")
        print()
        print("Risk Distribution:")
        print(f"  🔴 Critical: {snapshot.get('critical_count', 0)}")
        print(f"  🟠 High:     {snapshot.get('high_count', 0)}")
        print(f"  🟡 Moderate: {snapshot.get('moderate_count', 0)}")
        print(f"  🟢 Low:      {snapshot.get('low_count', 0)}")
        print()
        
        top_volatile = snapshot.get("top_volatile_terms", [])
        if top_volatile:
            print(f"Top Volatile: {', '.join(top_volatile)}")
        
        print()
        print(f"Output: {snapshot_path}")
    
    return 0


def _run_epistemic_tensor(
    semantic_panel_path: Optional[str],
    curriculum_panel_path: Optional[str],
    metric_matrix_path: Optional[str],
    drift_view_path: Optional[str],
    json_output: bool,
) -> int:
    """Run epistemic alignment tensor construction."""
    if not all([semantic_panel_path, curriculum_panel_path, metric_matrix_path, drift_view_path]):
        print("ERROR: --epistemic-tensor requires --semantic-panel, --curriculum-panel, --metric-matrix, --drift-view")
        return 1
    
    try:
        semantic_panel = json.loads(Path(semantic_panel_path).read_text(encoding="utf-8"))
        curriculum_panel = json.loads(Path(curriculum_panel_path).read_text(encoding="utf-8"))
        metric_matrix = json.loads(Path(metric_matrix_path).read_text(encoding="utf-8"))
        drift_view = json.loads(Path(drift_view_path).read_text(encoding="utf-8"))
    except Exception as e:
        print(f"ERROR: Could not load input files: {e}")
        return 1
    
    tensor = build_epistemic_alignment_tensor(
        semantic_panel,
        curriculum_panel,
        metric_matrix,
        drift_view,
    )
    
    if json_output:
        print(json.dumps(tensor, indent=2, sort_keys=True))
    else:
        print("EPISTEMIC ALIGNMENT TENSOR")
        print("=" * 50)
        print()
        print(f"Tensor ID: {tensor.get('tensor_id', 'N/A')[:16]}...")
        print()
        print("System Axes:")
        system_axes = tensor.get("system_axes", {})
        for axis, value in sorted(system_axes.items()):
            print(f"  {axis}: {value:.3f}")
        print()
        print(f"Tensor Norm: {tensor.get('alignment_tensor_norm', 0.0):.3f}")
        print()
        hotspots = tensor.get("misalignment_hotspots", [])
        if hotspots:
            print(f"Misalignment Hotspots: {', '.join(hotspots)}")
        else:
            print("Misalignment Hotspots: None")
        print()
        slice_axis = tensor.get("slice_axis", {})
        if slice_axis:
            print("Slice Alignment Scores:")
            for slice_name, score in sorted(slice_axis.items()):
                print(f"  {slice_name}: {score:.3f}")
    
    return 0


def _run_forecast_misalignment(
    alignment_tensor_path: Optional[str],
    historical_alignment_path: Optional[str],
    json_output: bool,
) -> int:
    """Run epistemic misalignment forecast."""
    if not alignment_tensor_path:
        print("ERROR: --forecast-misalignment requires --alignment-tensor")
        return 1
    
    try:
        alignment_tensor = json.loads(Path(alignment_tensor_path).read_text(encoding="utf-8"))
        historical_alignment = None
        if historical_alignment_path:
            historical_alignment = json.loads(Path(historical_alignment_path).read_text(encoding="utf-8"))
    except Exception as e:
        print(f"ERROR: Could not load input files: {e}")
        return 1
    
    forecast = forecast_epistemic_misalignment(alignment_tensor, historical_alignment)
    
    if json_output:
        print(json.dumps(forecast, indent=2, sort_keys=True))
    else:
        print("EPISTEMIC MISALIGNMENT FORECAST")
        print("=" * 50)
        print()
        print(f"Forecast ID: {forecast.get('forecast_id', 'N/A')[:16]}...")
        print()
        print(f"Predicted Band: {forecast.get('predicted_band', 'N/A')}")
        print(f"Confidence: {forecast.get('confidence', 0.0):.3f}")
        print(f"Time to Drift Event: {forecast.get('time_to_drift_event', 0)} cycles")
        print()
        explanation = forecast.get("neutral_explanation", [])
        if explanation:
            print("Explanation:")
            for line in explanation:
                print(f"  - {line}")
    
    return 0


def _run_director_panel(
    alignment_tensor_path: Optional[str],
    forecast_path: Optional[str],
    structural_view_path: Optional[str],
    json_output: bool,
) -> int:
    """Run epistemic director panel construction."""
    if not all([alignment_tensor_path, forecast_path, structural_view_path]):
        print("ERROR: --director-panel requires --alignment-tensor, --forecast, --structural-view")
        return 1
    
    try:
        alignment_tensor = json.loads(Path(alignment_tensor_path).read_text(encoding="utf-8"))
        forecast = json.loads(Path(forecast_path).read_text(encoding="utf-8"))
        structural_view = json.loads(Path(structural_view_path).read_text(encoding="utf-8"))
    except Exception as e:
        print(f"ERROR: Could not load input files: {e}")
        return 1
    
    panel = build_epistemic_director_panel(alignment_tensor, forecast, structural_view)
    
    if json_output:
        print(json.dumps(panel, indent=2, sort_keys=True))
    else:
        print("EPISTEMIC DIRECTOR PANEL")
        print("=" * 50)
        print()
        print(f"Panel ID: {panel.get('panel_id', 'N/A')[:16]}...")
        print()
        status_light = panel.get("status_light", "UNKNOWN")
        status_emoji = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴"}.get(status_light, "⚪")
        print(f"Status Light: {status_emoji} {status_light}")
        print()
        print(f"Alignment Band: {panel.get('alignment_band', 'N/A')}")
        print(f"Forecast Band: {panel.get('forecast_band', 'N/A')}")
        print(f"Structural Band: {panel.get('structural_band', 'N/A')}")
        print()
        print(f"Headline: {panel.get('headline', 'N/A')}")
        print()
        flags = panel.get("flags", [])
        if flags:
            print("Flags:")
            for flag in flags:
                print(f"  - {flag}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

