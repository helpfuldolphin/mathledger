"""P5 Patterns Calibration Panel.

STATUS: PHASE X — P5 PATTERN QUALITY CONTROL

Provides per-experiment snapshot generation and cross-experiment panel aggregation
for P5 divergence pattern classification across CAL-EXP-1/2/3.

SHADOW MODE CONTRACT:
- All functions are read-only and side-effect free (except file writes)
- Pattern snapshots are observational only
- Panel aggregation is advisory, not a hard gate
- No control flow depends on pattern values
- No governance writes (no solo hard block)

See:
- docs/system_law/Real_Telemetry_Topology_Spec.md Section 3
- docs/system_law/Global_Governance_Fusion_PhaseX.md Section 11
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from backend.topology.first_light.p5_pattern_classifier import (
    DivergencePattern,
    PatternClassification,
    P5_PATTERN_SCHEMA_VERSION,
)

__all__ = [
    "P5_PATTERNS_SNAPSHOT_SCHEMA_VERSION",
    "P5_PATTERNS_PANEL_SCHEMA_VERSION",
    "EXTRACTION_SOURCE_MANIFEST",
    "EXTRACTION_SOURCE_EVIDENCE_JSON",
    "EXTRACTION_SOURCE_MISSING",
    "MAX_DRIVERS_CAP",
    "P5PatternSnapshot",
    "P5PatternsPanel",
    "build_p5_patterns_snapshot",
    "persist_p5_patterns_snapshot",
    "load_p5_patterns_snapshot",
    "build_p5_patterns_panel",
    "persist_p5_patterns_panel",
    "attach_p5_patterns_panel_to_evidence",
    "extract_p5_patterns_panel_status",
    "extract_p5_patterns_panel_signal_for_status",
    "extract_p5_patterns_panel_warnings",
    "p5_patterns_panel_for_alignment_view",
    "load_and_build_panel_from_directory",
]

# Schema versions
P5_PATTERNS_SNAPSHOT_SCHEMA_VERSION = "1.0.0"
P5_PATTERNS_PANEL_SCHEMA_VERSION = "1.0.0"

# High-severity patterns that require tracking across experiments
HIGH_SEVERITY_PATTERNS = frozenset([
    DivergencePattern.STRUCTURAL_BREAK.value,
    DivergencePattern.ATTRACTOR_MISS.value,
])


@dataclass
class P5PatternSnapshot:
    """
    Per-experiment P5 patterns snapshot.

    Captures dominant pattern, counts, max streak, and high-confidence events
    for a single calibration experiment.

    SHADOW MODE: Observational only.
    """
    schema_version: str = P5_PATTERNS_SNAPSHOT_SCHEMA_VERSION
    cal_id: str = ""
    timestamp: str = ""
    mode: str = "SHADOW"

    # Pattern summary
    dominant_pattern: str = DivergencePattern.NOMINAL.value
    pattern_counts: Dict[str, int] = field(default_factory=dict)
    max_streak: int = 0
    max_streak_pattern: str = DivergencePattern.NOMINAL.value

    # High-confidence events
    high_confidence_events: List[Dict[str, Any]] = field(default_factory=list)
    high_confidence_threshold: float = 0.85

    # Totals
    total_classifications: int = 0
    cycles_analyzed: int = 0

    # Recalibration tracking
    recalibration_triggered: bool = False

    # Content hash for determinism verification
    content_hash: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()
        if not self.content_hash:
            self.content_hash = self._compute_content_hash()

    def _compute_content_hash(self) -> str:
        """Compute deterministic content hash."""
        hashable = {
            "cal_id": self.cal_id,
            "dominant_pattern": self.dominant_pattern,
            "pattern_counts": dict(sorted(self.pattern_counts.items())),
            "max_streak": self.max_streak,
            "total_classifications": self.total_classifications,
        }
        content = json.dumps(hashable, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "schema_version": self.schema_version,
            "cal_id": self.cal_id,
            "timestamp": self.timestamp,
            "mode": self.mode,
            "dominant_pattern": self.dominant_pattern,
            "pattern_counts": dict(sorted(self.pattern_counts.items())),
            "max_streak": self.max_streak,
            "max_streak_pattern": self.max_streak_pattern,
            "high_confidence_events": self.high_confidence_events,
            "high_confidence_threshold": self.high_confidence_threshold,
            "total_classifications": self.total_classifications,
            "cycles_analyzed": self.cycles_analyzed,
            "recalibration_triggered": self.recalibration_triggered,
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "P5PatternSnapshot":
        """Create from dictionary."""
        return cls(
            schema_version=data.get("schema_version", P5_PATTERNS_SNAPSHOT_SCHEMA_VERSION),
            cal_id=data.get("cal_id", ""),
            timestamp=data.get("timestamp", ""),
            mode=data.get("mode", "SHADOW"),
            dominant_pattern=data.get("dominant_pattern", DivergencePattern.NOMINAL.value),
            pattern_counts=data.get("pattern_counts", {}),
            max_streak=data.get("max_streak", 0),
            max_streak_pattern=data.get("max_streak_pattern", DivergencePattern.NOMINAL.value),
            high_confidence_events=data.get("high_confidence_events", []),
            high_confidence_threshold=data.get("high_confidence_threshold", 0.85),
            total_classifications=data.get("total_classifications", 0),
            cycles_analyzed=data.get("cycles_analyzed", 0),
            recalibration_triggered=data.get("recalibration_triggered", False),
            content_hash=data.get("content_hash", ""),
        )


@dataclass
class P5PatternsPanel:
    """
    Cross-experiment P5 patterns panel.

    Aggregates pattern data across CAL-EXP-1/2/3 for quality control.

    SHADOW MODE CONTRACT:
    - Advisory only, no hard blocks
    - Observational, does not gate decisions
    """
    schema_version: str = P5_PATTERNS_PANEL_SCHEMA_VERSION
    timestamp: str = ""
    mode: str = "SHADOW"

    # Aggregated counts across all experiments
    aggregated_pattern_counts: Dict[str, int] = field(default_factory=dict)

    # Per-experiment summary
    experiment_summaries: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # High-severity pattern tracking
    structural_break_experiments: List[str] = field(default_factory=list)
    attractor_miss_experiments: List[str] = field(default_factory=list)

    # Panel-level metrics
    total_experiments: int = 0
    experiments_with_high_severity: int = 0
    max_streak_across_experiments: int = 0
    max_streak_experiment: str = ""
    max_streak_pattern: str = DivergencePattern.NOMINAL.value

    # Content hash for determinism verification
    content_hash: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()
        if not self.content_hash:
            self.content_hash = self._compute_content_hash()

    def _compute_content_hash(self) -> str:
        """Compute deterministic content hash."""
        hashable = {
            "aggregated_pattern_counts": dict(sorted(self.aggregated_pattern_counts.items())),
            "structural_break_experiments": sorted(self.structural_break_experiments),
            "attractor_miss_experiments": sorted(self.attractor_miss_experiments),
            "total_experiments": self.total_experiments,
            "max_streak_across_experiments": self.max_streak_across_experiments,
        }
        content = json.dumps(hashable, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "schema_version": self.schema_version,
            "timestamp": self.timestamp,
            "mode": self.mode,
            "aggregated_pattern_counts": dict(sorted(self.aggregated_pattern_counts.items())),
            "experiment_summaries": {
                k: v for k, v in sorted(self.experiment_summaries.items())
            },
            "structural_break_experiments": sorted(self.structural_break_experiments),
            "attractor_miss_experiments": sorted(self.attractor_miss_experiments),
            "total_experiments": self.total_experiments,
            "experiments_with_high_severity": self.experiments_with_high_severity,
            "max_streak_across_experiments": self.max_streak_across_experiments,
            "max_streak_experiment": self.max_streak_experiment,
            "max_streak_pattern": self.max_streak_pattern,
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "P5PatternsPanel":
        """Create from dictionary."""
        return cls(
            schema_version=data.get("schema_version", P5_PATTERNS_PANEL_SCHEMA_VERSION),
            timestamp=data.get("timestamp", ""),
            mode=data.get("mode", "SHADOW"),
            aggregated_pattern_counts=data.get("aggregated_pattern_counts", {}),
            experiment_summaries=data.get("experiment_summaries", {}),
            structural_break_experiments=data.get("structural_break_experiments", []),
            attractor_miss_experiments=data.get("attractor_miss_experiments", []),
            total_experiments=data.get("total_experiments", 0),
            experiments_with_high_severity=data.get("experiments_with_high_severity", 0),
            max_streak_across_experiments=data.get("max_streak_across_experiments", 0),
            max_streak_experiment=data.get("max_streak_experiment", ""),
            max_streak_pattern=data.get("max_streak_pattern", DivergencePattern.NOMINAL.value),
            content_hash=data.get("content_hash", ""),
        )


def build_p5_patterns_snapshot(
    cal_id: str,
    classifications: List[PatternClassification],
    cycles_analyzed: int = 0,
    recalibration_triggered: bool = False,
    high_confidence_threshold: float = 0.85,
) -> P5PatternSnapshot:
    """
    Build P5 patterns snapshot for a single calibration experiment.

    SHADOW MODE CONTRACT:
    - Non-mutating (returns new object)
    - Observational only

    Args:
        cal_id: Calibration experiment identifier (e.g., "CAL-EXP-1")
        classifications: List of PatternClassification from the experiment
        cycles_analyzed: Number of cycles in the experiment
        recalibration_triggered: Whether recalibration was triggered
        high_confidence_threshold: Threshold for high-confidence events

    Returns:
        P5PatternSnapshot with aggregated pattern data
    """
    if not classifications:
        return P5PatternSnapshot(
            cal_id=cal_id,
            cycles_analyzed=cycles_analyzed,
            recalibration_triggered=recalibration_triggered,
            high_confidence_threshold=high_confidence_threshold,
        )

    # Count patterns
    pattern_counter: Counter[str] = Counter()
    for c in classifications:
        pattern_counter[c.pattern.value] += 1

    # Find dominant pattern
    dominant_pattern = pattern_counter.most_common(1)[0][0] if pattern_counter else DivergencePattern.NOMINAL.value

    # Compute max streak
    max_streak = 0
    max_streak_pattern = DivergencePattern.NOMINAL.value
    current_streak = 0
    current_pattern: Optional[str] = None

    for c in classifications:
        pattern_val = c.pattern.value
        if pattern_val == current_pattern:
            current_streak += 1
        else:
            current_pattern = pattern_val
            current_streak = 1

        if current_streak > max_streak:
            max_streak = current_streak
            max_streak_pattern = pattern_val

    # Collect high-confidence events
    high_confidence_events: List[Dict[str, Any]] = []
    for i, c in enumerate(classifications):
        if c.confidence >= high_confidence_threshold and c.pattern != DivergencePattern.NOMINAL:
            high_confidence_events.append({
                "cycle_index": i,
                "pattern": c.pattern.value,
                "confidence": round(c.confidence, 4),
                "timestamp": c.timestamp,
            })

    # Limit to top 10 by confidence
    high_confidence_events.sort(key=lambda x: -x["confidence"])
    high_confidence_events = high_confidence_events[:10]

    snapshot = P5PatternSnapshot(
        cal_id=cal_id,
        dominant_pattern=dominant_pattern,
        pattern_counts=dict(pattern_counter),
        max_streak=max_streak,
        max_streak_pattern=max_streak_pattern,
        high_confidence_events=high_confidence_events,
        high_confidence_threshold=high_confidence_threshold,
        total_classifications=len(classifications),
        cycles_analyzed=cycles_analyzed if cycles_analyzed > 0 else len(classifications),
        recalibration_triggered=recalibration_triggered,
    )

    # Recompute hash after setting all fields
    snapshot.content_hash = snapshot._compute_content_hash()

    return snapshot


def persist_p5_patterns_snapshot(
    snapshot: P5PatternSnapshot,
    output_dir: Path,
) -> Path:
    """
    Persist P5 patterns snapshot to disk.

    SHADOW MODE CONTRACT:
    - File write is observational only
    - Does not gate any decisions

    Args:
        snapshot: P5PatternSnapshot to persist
        output_dir: Directory to write snapshot (e.g., calibration/)

    Returns:
        Path to written snapshot file
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    cal_id = snapshot.cal_id.replace("-", "_").lower()
    output_path = output_dir / f"p5_patterns_{cal_id}.json"

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(snapshot.to_dict(), f, indent=2, sort_keys=True)

    return output_path


def load_p5_patterns_snapshot(
    snapshot_path: Path,
) -> Optional[P5PatternSnapshot]:
    """
    Load P5 patterns snapshot from disk.

    Args:
        snapshot_path: Path to snapshot JSON file

    Returns:
        P5PatternSnapshot or None if file doesn't exist
    """
    if not snapshot_path.exists():
        return None

    with open(snapshot_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return P5PatternSnapshot.from_dict(data)


def build_p5_patterns_panel(
    snapshots: List[P5PatternSnapshot],
) -> P5PatternsPanel:
    """
    Build cross-experiment P5 patterns panel.

    Aggregates pattern data from multiple calibration experiments.

    SHADOW MODE CONTRACT:
    - Non-mutating (returns new object)
    - Advisory only, no hard blocks
    - Observational, does not gate decisions

    Args:
        snapshots: List of P5PatternSnapshot from CAL-EXP-1/2/3

    Returns:
        P5PatternsPanel with aggregated cross-experiment data
    """
    if not snapshots:
        return P5PatternsPanel()

    # Aggregate pattern counts
    aggregated_counts: Counter[str] = Counter()
    experiment_summaries: Dict[str, Dict[str, Any]] = {}

    structural_break_experiments: List[str] = []
    attractor_miss_experiments: List[str] = []

    max_streak_across = 0
    max_streak_exp = ""
    max_streak_pat = DivergencePattern.NOMINAL.value

    for snap in snapshots:
        # Aggregate counts
        for pattern, count in snap.pattern_counts.items():
            aggregated_counts[pattern] += count

        # Build experiment summary
        experiment_summaries[snap.cal_id] = {
            "dominant_pattern": snap.dominant_pattern,
            "total_classifications": snap.total_classifications,
            "max_streak": snap.max_streak,
            "recalibration_triggered": snap.recalibration_triggered,
            "high_confidence_count": len(snap.high_confidence_events),
        }

        # Track high-severity patterns
        if snap.pattern_counts.get(DivergencePattern.STRUCTURAL_BREAK.value, 0) > 0:
            structural_break_experiments.append(snap.cal_id)

        if snap.pattern_counts.get(DivergencePattern.ATTRACTOR_MISS.value, 0) > 0:
            attractor_miss_experiments.append(snap.cal_id)

        # Track max streak
        if snap.max_streak > max_streak_across:
            max_streak_across = snap.max_streak
            max_streak_exp = snap.cal_id
            max_streak_pat = snap.max_streak_pattern

    # Count experiments with high-severity patterns
    high_severity_exps = set(structural_break_experiments) | set(attractor_miss_experiments)

    panel = P5PatternsPanel(
        aggregated_pattern_counts=dict(aggregated_counts),
        experiment_summaries=experiment_summaries,
        structural_break_experiments=sorted(structural_break_experiments),
        attractor_miss_experiments=sorted(attractor_miss_experiments),
        total_experiments=len(snapshots),
        experiments_with_high_severity=len(high_severity_exps),
        max_streak_across_experiments=max_streak_across,
        max_streak_experiment=max_streak_exp,
        max_streak_pattern=max_streak_pat,
    )

    # Recompute hash after setting all fields
    panel.content_hash = panel._compute_content_hash()

    return panel


def persist_p5_patterns_panel(
    panel: P5PatternsPanel,
    output_dir: Path,
) -> Path:
    """
    Persist P5 patterns panel to disk.

    SHADOW MODE CONTRACT:
    - File write is observational only
    - Does not gate any decisions

    Args:
        panel: P5PatternsPanel to persist
        output_dir: Directory to write panel (e.g., calibration/)

    Returns:
        Path to written panel file
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "p5_patterns_panel.json"

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(panel.to_dict(), f, indent=2, sort_keys=True)

    return output_path


def attach_p5_patterns_panel_to_evidence(
    evidence: Dict[str, Any],
    panel: P5PatternsPanel,
) -> Dict[str, Any]:
    """
    Attach P5 patterns panel to evidence pack.

    SHADOW MODE CONTRACT:
    - Non-mutating (returns new dict)
    - Panel is advisory only
    - No solo hard block

    Args:
        evidence: Evidence dictionary
        panel: P5PatternsPanel from build_p5_patterns_panel

    Returns:
        New evidence dict with p5_patterns_panel under evidence["governance"]
    """
    # Create new dict (non-mutating)
    updated_evidence = dict(evidence)

    # Ensure governance structure exists
    if "governance" not in updated_evidence:
        updated_evidence["governance"] = {}
    else:
        updated_evidence["governance"] = dict(updated_evidence["governance"])

    # Attach panel
    updated_evidence["governance"]["p5_patterns_panel"] = panel.to_dict()

    return updated_evidence


def extract_p5_patterns_panel_status(
    panel: P5PatternsPanel,
) -> Dict[str, Any]:
    """
    Extract status summary for governance reporting.

    SHADOW MODE CONTRACT:
    - Non-mutating (returns new dict)
    - Observational only

    Args:
        panel: P5PatternsPanel from build_p5_patterns_panel

    Returns:
        Status summary with:
        - total_experiments: int
        - experiments_with_high_severity: int
        - dominant_pattern_overall: str
        - has_structural_breaks: bool
        - has_attractor_misses: bool
    """
    # Find overall dominant pattern
    if panel.aggregated_pattern_counts:
        dominant = max(
            panel.aggregated_pattern_counts.items(),
            key=lambda x: x[1]
        )[0]
    else:
        dominant = DivergencePattern.NOMINAL.value

    return {
        "total_experiments": panel.total_experiments,
        "experiments_with_high_severity": panel.experiments_with_high_severity,
        "dominant_pattern_overall": dominant,
        "has_structural_breaks": len(panel.structural_break_experiments) > 0,
        "has_attractor_misses": len(panel.attractor_miss_experiments) > 0,
        "max_streak": panel.max_streak_across_experiments,
    }


def load_and_build_panel_from_directory(
    calibration_dir: Path,
    cal_ids: Optional[List[str]] = None,
) -> Tuple[P5PatternsPanel, List[P5PatternSnapshot]]:
    """
    Load snapshots from directory and build panel.

    Convenience function that scans calibration directory for p5_patterns_*.json
    files and builds a panel from found snapshots.

    Args:
        calibration_dir: Directory containing p5_patterns_*.json files
        cal_ids: Optional list of specific cal_ids to include

    Returns:
        Tuple of (P5PatternsPanel, List[P5PatternSnapshot])
    """
    snapshots: List[P5PatternSnapshot] = []

    if not calibration_dir.exists():
        return P5PatternsPanel(), snapshots

    # Find all p5_patterns_*.json files
    pattern_files = list(calibration_dir.glob("p5_patterns_*.json"))

    for pf in pattern_files:
        # Skip the panel file itself
        if pf.name == "p5_patterns_panel.json":
            continue

        snapshot = load_p5_patterns_snapshot(pf)
        if snapshot:
            # Filter by cal_id if specified
            if cal_ids is None or snapshot.cal_id in cal_ids:
                snapshots.append(snapshot)

    panel = build_p5_patterns_panel(snapshots)
    return panel, snapshots


# =============================================================================
# Status Integration (Manifest-First, Evidence-Fallback)
# =============================================================================

# Extraction source constants
EXTRACTION_SOURCE_MANIFEST = "MANIFEST"
EXTRACTION_SOURCE_EVIDENCE_JSON = "EVIDENCE_JSON"
EXTRACTION_SOURCE_MISSING = "MISSING"

# Maximum drivers cap (enforced everywhere)
MAX_DRIVERS_CAP = 3


def extract_p5_patterns_panel_signal_for_status(
    pack_manifest: Optional[Dict[str, Any]] = None,
    evidence_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Extract compact P5 patterns panel signal for status display.

    STATUS: PHASE X — P5 PATTERNS PANEL STATUS INTEGRATION (STABILIZED + PROVENANCE)

    Manifest-first extraction contract:
    - Primary: pack_manifest["governance"]["p5_patterns_panel"]
    - Fallback: evidence_data["governance"]["p5_patterns_panel"]

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - Signal is purely observational (advisory only)
    - No control flow depends on the signal
    - Neutral phrasing only
    - mode="SHADOW" always
    - weight_hint="LOW" always
    - Can NEVER produce conflict=true
    - Can NEVER be sole cause of BLOCK decision

    NO SOLO HARD BLOCK ASSERTION:
    - This signal can only contribute to escalation when combined with other signals
    - Alone, it can produce at most "warn" status, never "block"
    - The GGFL adapter enforces conflict=false invariant

    DRIVER FORMAT (reason codes only, no prose):
    - DRIVER_DOMINANT_{pattern} - non-NOMINAL dominant pattern
    - DRIVER_STREAK_{n} - max streak >= 5
    - DRIVER_HIGH_SEVERITY_{n} - high severity count > 0
    - Cap: MAX_DRIVERS_CAP (3)

    Args:
        pack_manifest: Optional manifest dictionary (preferred source).
        evidence_data: Optional evidence.json dictionary (fallback source).

    Returns:
        Stabilized status signal with:
        - schema_version: str (passthrough from panel)
        - mode: "SHADOW" (always)
        - weight_hint: "LOW" (always)
        - extraction_source: "MANIFEST" | "EVIDENCE_JSON" | "MISSING"
        - dominant_pattern: str (overall dominant pattern)
        - max_streak: int (max streak across experiments)
        - experiments_with_high_severity: int
        - status: "ok" | "warn" (warn if structural breaks exist, never "block")
        - top_drivers: List[str] (max 3, reason codes only, deterministic ordering)
    """
    panel_data = None
    extraction_source = EXTRACTION_SOURCE_MISSING

    # Manifest-first extraction
    if pack_manifest:
        governance = pack_manifest.get("governance", {})
        panel_data = governance.get("p5_patterns_panel")
        if panel_data:
            extraction_source = EXTRACTION_SOURCE_MANIFEST

    # Fallback to evidence.json
    if not panel_data and evidence_data:
        governance = evidence_data.get("governance", {})
        panel_data = governance.get("p5_patterns_panel")
        if panel_data:
            extraction_source = EXTRACTION_SOURCE_EVIDENCE_JSON

    if not panel_data:
        return {
            "schema_version": P5_PATTERNS_PANEL_SCHEMA_VERSION,
            "mode": "SHADOW",
            "weight_hint": "LOW",
            "extraction_source": EXTRACTION_SOURCE_MISSING,
            "dominant_pattern": DivergencePattern.NOMINAL.value,
            "max_streak": 0,
            "experiments_with_high_severity": 0,
            "status": "ok",
            "top_drivers": [],
        }

    # Extract compact signal
    aggregated_counts = panel_data.get("aggregated_pattern_counts", {})
    structural_break_exps = panel_data.get("structural_break_experiments", [])
    max_streak = panel_data.get("max_streak_across_experiments", 0)
    high_severity_count = panel_data.get("experiments_with_high_severity", 0)

    # Find dominant pattern
    if aggregated_counts:
        dominant = max(aggregated_counts.items(), key=lambda x: x[1])[0]
    else:
        dominant = DivergencePattern.NOMINAL.value

    # Determine status: warn if any STRUCTURAL_BREAK experiments exist
    # NOTE: Status can NEVER be "block" - only "ok" or "warn" (NO SOLO HARD BLOCK)
    status = "warn" if len(structural_break_exps) > 0 else "ok"

    # Build deterministic top_drivers (max 3, reason codes only)
    # Order: dominant_pattern > max_streak > high_severity_count
    # FORMAT: DRIVER_{CATEGORY}_{VALUE} - NO PROSE
    top_drivers: List[str] = []

    # Driver 1: Dominant pattern (if not NOMINAL)
    if dominant != DivergencePattern.NOMINAL.value:
        top_drivers.append(f"DRIVER_DOMINANT_{dominant}")

    # Driver 2: Max streak (if >= 5)
    if max_streak >= 5:
        top_drivers.append(f"DRIVER_STREAK_{max_streak}")

    # Driver 3: High severity count (if > 0)
    if high_severity_count > 0:
        top_drivers.append(f"DRIVER_HIGH_SEVERITY_{high_severity_count}")

    # ENFORCE: Cap to MAX_DRIVERS_CAP (3)
    top_drivers = top_drivers[:MAX_DRIVERS_CAP]

    return {
        "schema_version": panel_data.get("schema_version", P5_PATTERNS_PANEL_SCHEMA_VERSION),
        "mode": "SHADOW",  # INVARIANT: Always SHADOW - observational only
        "weight_hint": "LOW",  # INVARIANT: Always LOW - advisory only
        "extraction_source": extraction_source,  # Provenance tracking
        "dominant_pattern": dominant,
        "max_streak": max_streak,
        "experiments_with_high_severity": high_severity_count,
        "status": status,
        "top_drivers": top_drivers,
    }


def extract_p5_patterns_panel_warnings(
    pack_manifest: Optional[Dict[str, Any]] = None,
    evidence_data: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """
    Extract advisory warnings from P5 patterns panel.

    STATUS: PHASE X — P5 PATTERNS PANEL WARNINGS

    Manifest-first extraction contract:
    - Primary: pack_manifest["governance"]["p5_patterns_panel"]
    - Fallback: evidence_data["governance"]["p5_patterns_panel"]

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from list construction)
    - Warnings are purely observational (advisory only)
    - No control flow depends on the warnings
    - Neutral phrasing only
    - Returns exactly 1 warning max (warning hygiene cap)

    Args:
        pack_manifest: Optional manifest dictionary (preferred source).
        evidence_data: Optional evidence.json dictionary (fallback source).

    Returns:
        List of advisory warning strings (empty if no warnings, max 1 warning).
    """
    warnings: List[str] = []

    panel_data = None

    # Manifest-first extraction
    if pack_manifest:
        governance = pack_manifest.get("governance", {})
        panel_data = governance.get("p5_patterns_panel")

    # Fallback to evidence.json
    if not panel_data and evidence_data:
        governance = evidence_data.get("governance", {})
        panel_data = governance.get("p5_patterns_panel")

    if not panel_data:
        return warnings

    # Check for STRUCTURAL_BREAK experiments
    structural_break_exps = panel_data.get("structural_break_experiments", [])

    if structural_break_exps:
        exp_count = len(structural_break_exps)
        # Limit to top 3 experiment IDs for brevity
        top_exps = sorted(structural_break_exps)[:3]
        top_exps_str = ", ".join(top_exps)

        warnings.append(
            f"P5 patterns panel: {exp_count} experiment(s) contain STRUCTURAL_BREAK "
            f"classifications (experiments: {top_exps_str})"
        )

    # Warning hygiene cap: max 1 warning
    return warnings[:1]


# =============================================================================
# GGFL Adapter (SIG-PAT - Low Weight)
# =============================================================================

def p5_patterns_panel_for_alignment_view(
    signal: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build P5 patterns panel signal for Global Governance First Light (GGFL) alignment view.

    STATUS: PHASE X — SIG-PAT GGFL ADAPTER (LOW WEIGHT, STABILIZED)

    This function converts the P5 patterns panel status signal into a format
    suitable for the GGFL alignment view. It provides a low-weight, non-conflicting
    signal that indicates P5 pattern quality across calibration experiments.

    SHADOW MODE CONTRACT:
    - This function is read-only (aside from dict construction)
    - The returned signal is purely observational
    - No control flow depends on the signal contents
    - Advisory only; neutral phrasing
    - LOW weight, no conflict (no solo hard block)

    NO SOLO HARD BLOCK INVARIANT:
    - conflict is ALWAYS False - this is a compile-time invariant
    - This signal can NEVER be the sole cause of a BLOCK decision
    - The signal can only contribute to escalation when combined with other signals

    Args:
        signal: P5 patterns panel signal from extract_p5_patterns_panel_signal_for_status().

    Returns:
        GGFL alignment view signal dictionary with:
        - signal_type: "SIG-PAT" (P5 Patterns)
        - schema_version: str (passthrough from signal)
        - mode: "SHADOW" (always)
        - status: "ok" | "warn" (warn if high severity events present, never "block")
        - conflict: false (ALWAYS - no solo hard block invariant)
        - weight_hint: "LOW" (always)
        - drivers: List[str] (from signal.top_drivers or computed, max 3)
        - summary: str (neutral sentence)
    """
    if not signal:
        return {
            "signal_type": "SIG-PAT",
            "schema_version": P5_PATTERNS_PANEL_SCHEMA_VERSION,
            "mode": "SHADOW",
            "status": "ok",
            "conflict": False,  # INVARIANT: Always False
            "weight_hint": "LOW",
            "drivers": [],
            "summary": "P5 patterns panel not available",
        }

    dominant_pattern = signal.get("dominant_pattern", DivergencePattern.NOMINAL.value)
    max_streak = signal.get("max_streak", 0)
    high_severity_count = signal.get("experiments_with_high_severity", 0)
    status = signal.get("status", "ok")

    # Use top_drivers from stabilized signal if available, otherwise compute
    # DRIVER FORMAT: Reason codes only (DRIVER_{CATEGORY}_{VALUE}), no prose
    drivers = signal.get("top_drivers")
    if drivers is None:
        # Fallback: compute drivers (for backward compatibility)
        drivers = []
        if dominant_pattern != DivergencePattern.NOMINAL.value:
            drivers.append(f"DRIVER_DOMINANT_{dominant_pattern}")
        if max_streak >= 5:
            drivers.append(f"DRIVER_STREAK_{max_streak}")
        if high_severity_count > 0:
            drivers.append(f"DRIVER_HIGH_SEVERITY_{high_severity_count}")

    # ENFORCE: Cap to MAX_DRIVERS_CAP (3) - applies to both computed and passthrough
    drivers = drivers[:MAX_DRIVERS_CAP]

    # Build summary (single neutral sentence)
    if status == "warn":
        summary = (
            f"P5 patterns panel: dominant pattern {dominant_pattern}, "
            f"max streak {max_streak}, {high_severity_count} experiment(s) with high-severity patterns."
        )
    else:
        summary = (
            f"P5 patterns panel: dominant pattern {dominant_pattern}, "
            f"max streak {max_streak}, no high-severity patterns detected."
        )

    # ==========================================================================
    # NO SOLO HARD BLOCK INVARIANT
    # ==========================================================================
    # conflict is ALWAYS False - this is a non-negotiable invariant
    # This signal can NEVER be the sole cause of a BLOCK decision in fusion
    # ==========================================================================

    return {
        "signal_type": "SIG-PAT",
        "schema_version": signal.get("schema_version", P5_PATTERNS_PANEL_SCHEMA_VERSION),
        "mode": "SHADOW",  # INVARIANT: Always SHADOW
        "status": status,
        "conflict": False,  # INVARIANT: Always False - NO SOLO HARD BLOCK
        "weight_hint": "LOW",  # INVARIANT: Always LOW
        "drivers": drivers,
        "summary": summary,
    }
