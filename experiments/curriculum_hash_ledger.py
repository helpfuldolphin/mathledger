# PHASE II — NOT USED IN PHASE I
# File: experiments/curriculum_hash_ledger.py
"""
Curriculum Hash Ledger — Drift detection system for Phase II curricula.

Records curriculum snapshots over time and detects configuration drift
(changes to slices, formulas, parameters) as a first-class signal.

Usage:
    # Record a snapshot
    uv run python experiments/curriculum_hash_ledger.py --snapshot \
        --config config/curriculum_uplift_phase2.yaml --origin=ci

    # Compare two snapshots
    uv run python experiments/curriculum_hash_ledger.py --diff \
        --from 0 --to -1

    # Fail if any drift detected between last two snapshots
    uv run python experiments/curriculum_hash_ledger.py --diff --fail-on-drift

Drift Severity Categories:
    NONE        — identical slices
    COSMETIC    — whitespace, ordering changes (hash unchanged)
    PARAMETRIC_MINOR — small number/value changes
    PARAMETRIC_MAJOR — large differences or removed params
    SEMANTIC    — change in metric kind, target hashes, formula pool
    STRUCTURAL  — slice added/removed

Risk Levels:
    INFO  — NONE, COSMETIC (proceed)
    WARN  — PARAMETRIC_MINOR (proceed with warning)
    BLOCK — PARAMETRIC_MAJOR, SEMANTIC, STRUCTURAL (fail CI)
"""

import argparse
import base64
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


# Domain separation prefix for curriculum hashing (follows HT_INVARIANT_SPEC_v1.md conventions)
DOMAIN_CURRICULUM = b"MathLedger:Curriculum:v2:"
DOMAIN_SLICE = b"MathLedger:CurriculumSlice:v2:"
DOMAIN_SIGNATURE = b"MathLedger:LedgerSignature:v2:"

# Environment variable to enable signing
LEDGER_SIGNING_ENV = "LEDGER_SIGNING"

# Default test key paths (for development/testing only)
DEFAULT_KEY_DIR = Path("artifacts/keys/ledger")
DEFAULT_PRIVATE_KEY_PATH = DEFAULT_KEY_DIR / "ledger_signing.key"
DEFAULT_PUBLIC_KEY_PATH = DEFAULT_KEY_DIR / "ledger_signing.pub"


class DriftType(str, Enum):
    """Drift severity categories for curriculum changes."""
    NONE = "NONE"                       # Identical slices
    COSMETIC = "COSMETIC"               # Whitespace, ordering changes
    PARAMETRIC_MINOR = "PARAMETRIC_MINOR"  # Small number/value changes
    PARAMETRIC_MAJOR = "PARAMETRIC_MAJOR"  # Large differences or removed params
    SEMANTIC = "SEMANTIC"               # Change in metric kind, target hashes, formula pool
    STRUCTURAL = "STRUCTURAL"           # Slice added/removed


class RiskLevel(str, Enum):
    """Risk levels for CI gating decisions."""
    INFO = "INFO"    # Proceed normally
    WARN = "WARN"    # Proceed with warning
    BLOCK = "BLOCK"  # Fail CI


@dataclass
class DriftEvent:
    """
    A single drift event in a slice's timeline.
    
    Represents a change detected between two consecutive snapshots.
    """
    timestamp: str                  # When the drift was detected
    old_hash: str                   # Hash before change
    new_hash: str                   # Hash after change
    drift_type: DriftType           # Classification of the change
    risk_level: RiskLevel           # CI gating level
    snapshot_index: int             # Index in ledger (for reference)
    git_commit: str                 # Git commit at detection
    change_type: str = "modified"   # "added", "removed", or "modified"
    changed_keys: List[str] = field(default_factory=list)
    notes: str = ""                 # Any notes from the snapshot
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "timestamp": self.timestamp,
            "old_hash": self.old_hash,
            "new_hash": self.new_hash,
            "drift_type": self.drift_type.value,
            "risk_level": self.risk_level.value,
            "snapshot_index": self.snapshot_index,
            "git_commit": self.git_commit,
            "change_type": self.change_type,
            "changed_keys": self.changed_keys,
            "notes": self.notes
        }
    
    def format_line(self) -> str:
        """Format as a single timeline line."""
        risk_icon = {"INFO": "✓", "WARN": "⚠", "BLOCK": "✗"}.get(self.risk_level.value, "?")
        return (
            f"[{self.snapshot_index:3d}] {self.timestamp} "
            f"{risk_icon} {self.drift_type.value:18s} "
            f"{self.old_hash[:8]}→{self.new_hash[:8]} "
            f"({self.change_type})"
        )


# Mapping from drift type to risk level
DRIFT_RISK_MAP: Dict[DriftType, RiskLevel] = {
    DriftType.NONE: RiskLevel.INFO,
    DriftType.COSMETIC: RiskLevel.INFO,
    DriftType.PARAMETRIC_MINOR: RiskLevel.WARN,
    DriftType.PARAMETRIC_MAJOR: RiskLevel.BLOCK,
    DriftType.SEMANTIC: RiskLevel.BLOCK,
    DriftType.STRUCTURAL: RiskLevel.BLOCK,
}


# =============================================================================
# TASK 2: Drift Intent Annotation
# =============================================================================

class DriftIntent(str, Enum):
    """
    Advisory annotation for the likely intent behind a drift event.
    
    This is purely informational and NEVER impacts exit codes or CI decisions.
    """
    MANUAL_EDIT = "MANUAL_EDIT"           # Human edited the config directly
    AUTO_REWRITE = "AUTO_REWRITE"         # Automated tool rewrote the config
    CURRICULUM_REFRESH = "CURRICULUM_REFRESH"  # Scheduled curriculum update
    UNKNOWN = "UNKNOWN"                   # Default when no heuristics match


# Heuristic patterns for detecting drift intent
INTENT_HEURISTICS = {
    # Notes patterns that suggest manual edits
    DriftIntent.MANUAL_EDIT: [
        r"manual",
        r"hand[\s-]?edit",
        r"human[\s-]?approved",
        r"reviewed",
        r"intentional",
    ],
    # Notes patterns that suggest automated rewrites
    DriftIntent.AUTO_REWRITE: [
        r"auto[\s-]?generated",
        r"script[\s-]?update",
        r"automated",
        r"bot",
        r"ci[\s-]?update",
        r"migration[\s-]?script",
    ],
    # Notes patterns that suggest curriculum refresh
    DriftIntent.CURRICULUM_REFRESH: [
        r"refresh",
        r"curriculum[\s-]?update",
        r"scheduled[\s-]?update",
        r"periodic",
        r"nightly",
        r"weekly",
    ],
}


def detect_drift_intent(notes: str, origin: str = "") -> DriftIntent:
    """
    Detect the likely intent behind a drift event using heuristics.
    
    This function analyzes snapshot notes and origin to classify
    the likely intent. It NEVER impacts CI decisions — advisory only.
    
    Args:
        notes: Free-text notes from the snapshot.
        origin: Origin field (e.g., "manual", "ci", "pre-commit").
    
    Returns:
        DriftIntent classification, defaults to UNKNOWN.
    """
    import re
    
    # Combine notes and origin for analysis
    text = f"{notes} {origin}".lower()
    
    if not text.strip():
        return DriftIntent.UNKNOWN
    
    # Check each intent's heuristics
    for intent, patterns in INTENT_HEURISTICS.items():
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return intent
    
    return DriftIntent.UNKNOWN


# =============================================================================
# TASK 1: Timeline Compression
# =============================================================================

@dataclass
class CompressedEventGroup:
    """
    A group of consecutive DriftEvents with the same compression key.
    
    For INFO-level events, multiple events may be merged into one group.
    For BLOCK/WARN events, each event is its own group (no merging).
    """
    slice_name: str
    change_type: str
    risk_level: RiskLevel
    events: List[DriftEvent]
    
    # Summary fields for compressed display
    first_timestamp: str = ""
    last_timestamp: str = ""
    first_snapshot_index: int = 0
    last_snapshot_index: int = 0
    event_count: int = 0
    drift_intents: List[DriftIntent] = field(default_factory=list)
    
    def __post_init__(self):
        """Compute summary fields from events."""
        if self.events:
            self.first_timestamp = self.events[0].timestamp
            self.last_timestamp = self.events[-1].timestamp
            self.first_snapshot_index = self.events[0].snapshot_index
            self.last_snapshot_index = self.events[-1].snapshot_index
            self.event_count = len(self.events)
            # Detect intent for each event
            self.drift_intents = [
                detect_drift_intent(e.notes, "") for e in self.events
            ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "slice_name": self.slice_name,
            "change_type": self.change_type,
            "risk_level": self.risk_level.value,
            "first_timestamp": self.first_timestamp,
            "last_timestamp": self.last_timestamp,
            "first_snapshot_index": self.first_snapshot_index,
            "last_snapshot_index": self.last_snapshot_index,
            "event_count": self.event_count,
            "drift_intents": [i.value for i in self.drift_intents],
            "events": [e.to_dict() for e in self.events]
        }


@dataclass
class CompressedTimeline:
    """
    A compressed representation of a drift timeline.
    
    Compression merges consecutive INFO-only drifts while preserving
    exact BLOCK and WARN boundaries for audit purposes.
    """
    slice_name: str
    groups: List[CompressedEventGroup]
    original_event_count: int
    compressed_group_count: int
    block_count: int
    warn_count: int
    info_count: int
    timeline_hash: str = ""
    
    def __post_init__(self):
        """Compute timeline hash for integrity checking."""
        if not self.timeline_hash:
            self.timeline_hash = self._compute_hash()
    
    def _compute_hash(self) -> str:
        """Compute deterministic hash of the compressed timeline."""
        # Use canonical JSON for determinism
        data = {
            "slice_name": self.slice_name,
            "groups": [g.to_dict() for g in self.groups],
            "original_event_count": self.original_event_count,
        }
        json_bytes = json.dumps(data, sort_keys=True, separators=(',', ':')).encode('utf-8')
        return hashlib.sha256(json_bytes).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "slice_name": self.slice_name,
            "groups": [g.to_dict() for g in self.groups],
            "original_event_count": self.original_event_count,
            "compressed_group_count": self.compressed_group_count,
            "block_count": self.block_count,
            "warn_count": self.warn_count,
            "info_count": self.info_count,
            "timeline_hash": self.timeline_hash
        }


def compress_drift_timeline(
    events: List[DriftEvent],
    slice_name: str = ""
) -> CompressedTimeline:
    """
    Compress a drift timeline by merging consecutive INFO-only events.
    
    Compression rules:
    - Merge consecutive INFO-only drifts into a single group
    - Preserve exact BLOCK and WARN boundaries (never merge)
    - Deterministic grouping keys: same slice, same change_type, same risk_level
    
    Args:
        events: List of DriftEvent objects (should be chronologically sorted).
        slice_name: Name of the slice for this timeline.
    
    Returns:
        CompressedTimeline with grouped events.
    """
    if not events:
        return CompressedTimeline(
            slice_name=slice_name,
            groups=[],
            original_event_count=0,
            compressed_group_count=0,
            block_count=0,
            warn_count=0,
            info_count=0
        )
    
    groups: List[CompressedEventGroup] = []
    current_group_events: List[DriftEvent] = []
    current_group_key: Optional[Tuple[str, RiskLevel]] = None
    
    block_count = 0
    warn_count = 0
    info_count = 0
    
    for event in events:
        # Count by risk level
        if event.risk_level == RiskLevel.BLOCK:
            block_count += 1
        elif event.risk_level == RiskLevel.WARN:
            warn_count += 1
        else:
            info_count += 1
        
        # Grouping key: (change_type, risk_level)
        event_key = (event.change_type, event.risk_level)
        
        # BLOCK and WARN events are never merged - each gets its own group
        if event.risk_level in (RiskLevel.BLOCK, RiskLevel.WARN):
            # Flush any pending INFO group
            if current_group_events and current_group_key:
                groups.append(CompressedEventGroup(
                    slice_name=slice_name,
                    change_type=current_group_key[0],
                    risk_level=current_group_key[1],
                    events=current_group_events
                ))
                current_group_events = []
                current_group_key = None
            
            # Create single-event group for BLOCK/WARN
            groups.append(CompressedEventGroup(
                slice_name=slice_name,
                change_type=event.change_type,
                risk_level=event.risk_level,
                events=[event]
            ))
        else:
            # INFO events can be merged if consecutive with same key
            if current_group_key == event_key:
                current_group_events.append(event)
            else:
                # Flush previous group if exists
                if current_group_events and current_group_key:
                    groups.append(CompressedEventGroup(
                        slice_name=slice_name,
                        change_type=current_group_key[0],
                        risk_level=current_group_key[1],
                        events=current_group_events
                    ))
                # Start new group
                current_group_events = [event]
                current_group_key = event_key
    
    # Flush final group
    if current_group_events and current_group_key:
        groups.append(CompressedEventGroup(
            slice_name=slice_name,
            change_type=current_group_key[0],
            risk_level=current_group_key[1],
            events=current_group_events
        ))
    
    return CompressedTimeline(
        slice_name=slice_name,
        groups=groups,
        original_event_count=len(events),
        compressed_group_count=len(groups),
        block_count=block_count,
        warn_count=warn_count,
        info_count=info_count
    )


def decompress_drift_timeline(compressed: CompressedTimeline) -> List[DriftEvent]:
    """
    Decompress a CompressedTimeline back to a list of DriftEvents.
    
    This is the inverse of compress_drift_timeline().
    Round-trip: decompress(compress(x)) ≈ x (events preserved, ordering maintained).
    
    Args:
        compressed: CompressedTimeline to decompress.
    
    Returns:
        List of DriftEvent objects in chronological order.
    """
    events: List[DriftEvent] = []
    for group in compressed.groups:
        events.extend(group.events)
    return events


# =============================================================================
# TASK 3: Chronicle Export Contract
# =============================================================================

CHRONICLE_SCHEMA_VERSION = "1.0"


@dataclass
class Chronicle:
    """
    Complete chronicle export of a slice's drift history.
    
    Provides a deterministic, self-authenticating record suitable
    for archival and CI integration.
    """
    schema_version: str
    slice_name: str
    events: CompressedTimeline
    timeline_hash: str
    block_count: int
    warnings: List[str]
    drift_intents_histogram: Dict[str, int]
    export_timestamp: str = ""
    
    def __post_init__(self):
        if not self.export_timestamp:
            self.export_timestamp = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict with deterministic ordering."""
        return {
            "schema_version": self.schema_version,
            "slice_name": self.slice_name,
            "events": self.events.to_dict(),
            "timeline_hash": self.timeline_hash,
            "block_count": self.block_count,
            "warnings": self.warnings,
            "drift_intents_histogram": self.drift_intents_histogram,
            "export_timestamp": self.export_timestamp
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Export as deterministic JSON string."""
        return json.dumps(self.to_dict(), sort_keys=True, indent=indent)


def export_chronicle(
    events: List[DriftEvent],
    slice_name: str,
    warnings: Optional[List[str]] = None
) -> Chronicle:
    """
    Export a complete chronicle for a slice's drift history.
    
    Contract fields:
    - schema_version: Version of the chronicle format
    - slice_name: Name of the slice
    - events: Compressed timeline
    - timeline_hash: Deterministic hash of the timeline
    - block_count: Number of BLOCK-level events
    - warnings: List of warning messages
    - drift_intents_histogram: Count of each DriftIntent type
    
    Args:
        events: List of DriftEvent objects for the slice.
        slice_name: Name of the slice.
        warnings: Optional list of warning messages.
    
    Returns:
        Chronicle object with all required fields.
    """
    # Compress the timeline
    compressed = compress_drift_timeline(events, slice_name)
    
    # Build drift intents histogram
    intent_histogram: Dict[str, int] = {intent.value: 0 for intent in DriftIntent}
    for group in compressed.groups:
        for intent in group.drift_intents:
            intent_histogram[intent.value] += 1
    
    return Chronicle(
        schema_version=CHRONICLE_SCHEMA_VERSION,
        slice_name=slice_name,
        events=compressed,
        timeline_hash=compressed.timeline_hash,
        block_count=compressed.block_count,
        warnings=warnings or [],
        drift_intents_histogram=intent_histogram
    )


# =============================================================================
# PHASE III: Cross-Slice Chronicle Index & Governance Lens
# =============================================================================

CHRONICLE_INDEX_SCHEMA_VERSION = "1.0"


class GovernanceStatus(str, Enum):
    """Governance status categories for curriculum-wide assessment."""
    STABLE = "STABLE"    # No BLOCK events, minimal churn
    MIXED = "MIXED"      # Some BLOCK events or varied intent mix
    VOLATILE = "VOLATILE"  # High BLOCK event count or high churn


class HealthStatus(str, Enum):
    """Global health status for curriculum chronicles."""
    OK = "OK"            # Low activity, no BLOCK events
    ATTENTION = "ATTENTION"  # Medium activity or some concerns
    HOT = "HOT"          # High activity or BLOCK events present


class ActivityLevel(str, Enum):
    """Activity level classification based on event count."""
    LOW = "LOW"          # Few events
    MEDIUM = "MEDIUM"    # Moderate events
    HIGH = "HIGH"        # Many events


# Activity level thresholds (total event groups across all slices)
ACTIVITY_THRESHOLD_LOW = 5
ACTIVITY_THRESHOLD_MEDIUM = 20


def build_chronicle_index(chronicles: List[Chronicle]) -> Dict[str, Any]:
    """
    Build a curriculum-wide chronicle index over all slices.
    
    Aggregates chronicle data across all slices into a unified index
    with deterministic ordering.
    
    Args:
        chronicles: Sequence of Chronicle objects for all slices.
    
    Returns:
        Dict with:
        - schema_version: Version of the index format
        - slice_count: Number of slices indexed
        - total_event_groups: Sum of compressed groups across all slices
        - slices_with_block_events: List of slice names with BLOCK events
        - intent_histogram_global: Aggregate drift intent histogram
        - slices_with_refresh_intent: Slices where CURRICULUM_REFRESH appears
    """
    # Sort chronicles by slice name for deterministic output
    sorted_chronicles = sorted(chronicles, key=lambda c: c.slice_name)
    
    # Initialize aggregates
    total_event_groups = 0
    slices_with_block_events: List[str] = []
    intent_histogram_global: Dict[str, int] = {intent.value: 0 for intent in DriftIntent}
    slices_with_refresh_intent: List[str] = []
    slices_with_manual_edits: List[str] = []
    slices_with_auto_rewrite: List[str] = []
    
    for chronicle in sorted_chronicles:
        # Count event groups
        total_event_groups += chronicle.events.compressed_group_count
        
        # Track slices with BLOCK events
        if chronicle.block_count > 0:
            slices_with_block_events.append(chronicle.slice_name)
        
        # Aggregate intent histogram
        for intent_name, count in chronicle.drift_intents_histogram.items():
            intent_histogram_global[intent_name] += count
            
            # Track slices by intent type
            if count > 0:
                if intent_name == DriftIntent.CURRICULUM_REFRESH.value:
                    if chronicle.slice_name not in slices_with_refresh_intent:
                        slices_with_refresh_intent.append(chronicle.slice_name)
                elif intent_name == DriftIntent.MANUAL_EDIT.value:
                    if chronicle.slice_name not in slices_with_manual_edits:
                        slices_with_manual_edits.append(chronicle.slice_name)
                elif intent_name == DriftIntent.AUTO_REWRITE.value:
                    if chronicle.slice_name not in slices_with_auto_rewrite:
                        slices_with_auto_rewrite.append(chronicle.slice_name)
    
    return {
        "schema_version": CHRONICLE_INDEX_SCHEMA_VERSION,
        "slice_count": len(chronicles),
        "total_event_groups": total_event_groups,
        "slices_with_block_events": sorted(slices_with_block_events),
        "intent_histogram_global": intent_histogram_global,
        "slices_with_refresh_intent": sorted(slices_with_refresh_intent),
        # Extra fields for governance lens (pre-computed)
        "_slices_with_manual_edits": sorted(slices_with_manual_edits),
        "_slices_with_auto_rewrite": sorted(slices_with_auto_rewrite),
    }


def summarize_chronicles_for_governance(index: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a governance/audit lens summary from the chronicle index.
    
    Provides a high-level view suitable for governance review.
    This is descriptive only; no CI behavior change.
    
    Args:
        index: Chronicle index from build_chronicle_index().
    
    Returns:
        Dict with:
        - has_block_level_drift: bool indicating any BLOCK events
        - slices_with_manual_edits: list of slice names
        - slices_with_auto_rewrite: list of slice names
        - status: "STABLE" | "MIXED" | "VOLATILE"
    """
    slices_with_block = index.get("slices_with_block_events", [])
    slices_with_manual = index.get("_slices_with_manual_edits", [])
    slices_with_auto = index.get("_slices_with_auto_rewrite", [])
    intent_histogram = index.get("intent_histogram_global", {})
    
    has_block_level_drift = len(slices_with_block) > 0
    
    # Determine governance status
    status = GovernanceStatus.STABLE
    
    # Count non-UNKNOWN intents to assess mix
    non_unknown_intents = sum(
        count for intent, count in intent_histogram.items()
        if intent != DriftIntent.UNKNOWN.value and count > 0
    )
    
    # Multiple intent types present = MIXED
    intent_types_present = sum(
        1 for intent, count in intent_histogram.items()
        if intent != DriftIntent.UNKNOWN.value and count > 0
    )
    
    if has_block_level_drift and len(slices_with_block) >= 3:
        # Many BLOCK events = VOLATILE
        status = GovernanceStatus.VOLATILE
    elif has_block_level_drift or intent_types_present >= 2:
        # Some BLOCK events or mixed intents = MIXED
        status = GovernanceStatus.MIXED
    else:
        status = GovernanceStatus.STABLE
    
    return {
        "has_block_level_drift": has_block_level_drift,
        "slices_with_manual_edits": slices_with_manual,
        "slices_with_auto_rewrite": slices_with_auto,
        "status": status.value,
    }


def summarize_chronicles_for_global_health(index: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a global health chronicle signal from the index.
    
    Provides a quick health status suitable for dashboards and monitoring.
    
    Args:
        index: Chronicle index from build_chronicle_index().
    
    Returns:
        Dict with:
        - curriculum_change_activity_level: "LOW" | "MEDIUM" | "HIGH"
        - any_block_events: bool
        - dominant_intent: The DriftIntent that appears most often
        - status: "OK" | "ATTENTION" | "HOT"
    """
    total_event_groups = index.get("total_event_groups", 0)
    slices_with_block = index.get("slices_with_block_events", [])
    intent_histogram = index.get("intent_histogram_global", {})
    
    # Determine activity level
    if total_event_groups <= ACTIVITY_THRESHOLD_LOW:
        activity_level = ActivityLevel.LOW
    elif total_event_groups <= ACTIVITY_THRESHOLD_MEDIUM:
        activity_level = ActivityLevel.MEDIUM
    else:
        activity_level = ActivityLevel.HIGH
    
    any_block_events = len(slices_with_block) > 0
    
    # Find dominant intent
    dominant_intent = DriftIntent.UNKNOWN.value
    max_count = 0
    for intent_name, count in sorted(intent_histogram.items()):  # sorted for determinism
        if count > max_count:
            max_count = count
            dominant_intent = intent_name
    
    # Determine health status
    if any_block_events:
        status = HealthStatus.HOT
    elif activity_level == ActivityLevel.HIGH:
        status = HealthStatus.ATTENTION
    elif activity_level == ActivityLevel.MEDIUM and max_count > 10:
        status = HealthStatus.ATTENTION
    else:
        status = HealthStatus.OK
    
    return {
        "curriculum_change_activity_level": activity_level.value,
        "any_block_events": any_block_events,
        "dominant_intent": dominant_intent,
        "status": status.value,
    }


# =============================================================================
# PHASE IV: Cross-Slice Chronicle Governance & Narrative Feed
# =============================================================================

class AlignmentStatus(str, Enum):
    """Alignment status for curriculum chronicle view."""
    STABLE = "STABLE"    # Low churn, aligned
    DRIFTY = "DRIFTY"    # Some misalignment or churn
    VOLATILE = "VOLATILE"  # High churn or significant misalignment


# Threshold for "high edit churn" (event groups per slice)
HIGH_CHURN_THRESHOLD = 10


def build_chronicle_alignment_view(
    index: Dict[str, Any],
    curriculum_timeline: Optional[Dict[str, Any]] = None,
    drift_classifications: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Build a cross-slice chronicle alignment view.
    
    Combines chronicle index data with curriculum timeline and drift
    classifications to identify alignment patterns.
    
    Args:
        index: Chronicle index from build_chronicle_index().
        curriculum_timeline: Optional dict mapping slice_name -> timeline data.
            Expected keys: event_count, last_change_timestamp, etc.
        drift_classifications: Optional dict mapping slice_name -> drift_type.
            Expected keys: drift_type (e.g., "STRUCTURAL", "SEMANTIC").
    
    Returns:
        Dict with:
        - slices_with_high_edit_churn: List of slice names with high event counts
        - slices_with_block_drift_and_block_events: Slices with both block drift and block events
        - alignment_status: "STABLE" | "DRIFTY" | "VOLATILE"
    """
    slices_with_high_churn: List[str] = []
    slices_with_block_drift_and_events: List[str] = []
    
    # Get slices with BLOCK events from index
    slices_with_block_events = set(index.get("slices_with_block_events", []))
    
    # Identify high churn slices
    # If curriculum_timeline provided, use it; otherwise estimate from chronicles
    if curriculum_timeline:
        for slice_name, timeline_data in curriculum_timeline.items():
            event_count = timeline_data.get("event_count", 0)
            if event_count >= HIGH_CHURN_THRESHOLD:
                slices_with_high_churn.append(slice_name)
    else:
        # Estimate from index: slices with many event groups relative to average
        total_groups = index.get("total_event_groups", 0)
        slice_count = index.get("slice_count", 1)
        avg_groups_per_slice = total_groups / slice_count if slice_count > 0 else 0
        
        # Mark slices with significantly above-average groups as high churn
        # This is a heuristic when detailed timeline data isn't available
        for slice_name in index.get("slices_with_block_events", []):
            # If a slice has BLOCK events, it's likely high churn
            if slice_name not in slices_with_high_churn:
                slices_with_high_churn.append(slice_name)
    
    # Identify slices with both block drift classification AND block events
    if drift_classifications:
        for slice_name, drift_data in drift_classifications.items():
            drift_type = drift_data.get("drift_type", "")
            # Check if drift type is BLOCK-level (STRUCTURAL, SEMANTIC, PARAMETRIC_MAJOR)
            is_block_drift = drift_type in [
                DriftType.STRUCTURAL.value,
                DriftType.SEMANTIC.value,
                DriftType.PARAMETRIC_MAJOR.value
            ]
            
            if is_block_drift and slice_name in slices_with_block_events:
                slices_with_block_drift_and_events.append(slice_name)
    
    # Determine alignment status
    has_high_churn = len(slices_with_high_churn) > 0
    has_block_drift_mismatch = len(slices_with_block_drift_and_events) > 0
    total_slices = index.get("slice_count", 0)
    churn_ratio = len(slices_with_high_churn) / total_slices if total_slices > 0 else 0
    
    if churn_ratio >= 0.5 or len(slices_with_block_drift_and_events) >= 3:
        alignment_status = AlignmentStatus.VOLATILE
    elif has_high_churn or has_block_drift_mismatch:
        alignment_status = AlignmentStatus.DRIFTY
    else:
        alignment_status = AlignmentStatus.STABLE
    
    return {
        "slices_with_high_edit_churn": sorted(slices_with_high_churn),
        "slices_with_block_drift_and_block_events": sorted(slices_with_block_drift_and_events),
        "alignment_status": alignment_status.value,
    }


def render_chronicle_governance_narrative(
    index: Dict[str, Any],
    alignment_view: Dict[str, Any]
) -> str:
    """
    Render a governance-grade narrative in Markdown format.
    
    Provides a neutral, descriptive summary suitable for governance review.
    Highlights key patterns without judgmental language.
    
    Args:
        index: Chronicle index from build_chronicle_index().
        alignment_view: Alignment view from build_chronicle_alignment_view().
    
    Returns:
        Markdown-formatted string with governance narrative.
    """
    lines = []
    lines.append("# Curriculum Chronicle Governance Summary")
    lines.append("")
    
    # Activity level
    health_summary = summarize_chronicles_for_global_health(index)
    activity_level = health_summary.get("curriculum_change_activity_level", "UNKNOWN")
    lines.append(f"## Change Activity Level: {activity_level}")
    lines.append("")
    
    total_groups = index.get("total_event_groups", 0)
    slice_count = index.get("slice_count", 0)
    lines.append(f"The curriculum has {total_groups} recorded change event groups across {slice_count} slices.")
    lines.append("")
    
    # Manual edits with BLOCK events
    slices_with_manual = set(index.get("_slices_with_manual_edits", []))
    slices_with_block = set(index.get("slices_with_block_events", []))
    slices_manual_and_block = sorted(slices_with_manual & slices_with_block)
    
    if slices_manual_and_block:
        lines.append("## Slices with Manual Edits and Block-Level Events")
        lines.append("")
        lines.append("The following slices have both manual edit intent and block-level drift events:")
        lines.append("")
        for slice_name in slices_manual_and_block:
            lines.append(f"- `{slice_name}`")
        lines.append("")
    
    # Curriculum refresh intents
    slices_with_refresh = index.get("slices_with_refresh_intent", [])
    if slices_with_refresh:
        lines.append("## Curriculum Refresh Activity")
        lines.append("")
        lines.append(f"{len(slices_with_refresh)} slice(s) show curriculum refresh intent:")
        lines.append("")
        for slice_name in slices_with_refresh:
            lines.append(f"- `{slice_name}`")
        lines.append("")
    
    # High churn slices
    high_churn_slices = alignment_view.get("slices_with_high_edit_churn", [])
    if high_churn_slices:
        lines.append("## High Edit Churn Slices")
        lines.append("")
        lines.append(f"{len(high_churn_slices)} slice(s) show elevated edit frequency:")
        lines.append("")
        for slice_name in high_churn_slices:
            lines.append(f"- `{slice_name}`")
        lines.append("")
    
    # Alignment status
    alignment_status = alignment_view.get("alignment_status", "UNKNOWN")
    lines.append(f"## Alignment Status: {alignment_status}")
    lines.append("")
    
    if alignment_status == AlignmentStatus.STABLE.value:
        lines.append("The curriculum shows stable alignment with minimal churn.")
    elif alignment_status == AlignmentStatus.DRIFTY.value:
        lines.append("The curriculum shows some drift patterns with moderate churn.")
    else:
        lines.append("The curriculum shows volatile alignment patterns with significant churn.")
    lines.append("")
    
    # Intent distribution
    intent_histogram = index.get("intent_histogram_global", {})
    non_zero_intents = {
        intent: count for intent, count in intent_histogram.items()
        if count > 0
    }
    if non_zero_intents:
        lines.append("## Intent Distribution")
        lines.append("")
        for intent, count in sorted(non_zero_intents.items()):
            lines.append(f"- {intent}: {count} occurrence(s)")
        lines.append("")
    
    return "\n".join(lines)


def build_chronicle_summary_for_acquisition(
    index: Dict[str, Any],
    alignment_view: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Build an acquisition-facing chronicle summary.
    
    Provides a high-level, neutral summary suitable for due diligence
    and acquisition discussions.
    
    Args:
        index: Chronicle index from build_chronicle_index().
        alignment_view: Alignment view from build_chronicle_alignment_view().
    
    Returns:
        Dict with:
        - change_activity_band: "LOW" | "MEDIUM" | "HIGH"
        - governance_status: "STABLE" | "MIXED" | "VOLATILE"
        - headline: Neutral summary text
    """
    # Get activity level from health summary
    health_summary = summarize_chronicles_for_global_health(index)
    activity_band = health_summary.get("curriculum_change_activity_level", "LOW")
    
    # Get governance status
    governance_summary = summarize_chronicles_for_governance(index)
    governance_status = governance_summary.get("status", "STABLE")
    
    # Build headline
    total_groups = index.get("total_event_groups", 0)
    slice_count = index.get("slice_count", 0)
    has_block_events = health_summary.get("any_block_events", False)
    
    if activity_band == "LOW":
        frequency_desc = "infrequent"
    elif activity_band == "MEDIUM":
        frequency_desc = "moderate"
    else:
        frequency_desc = "frequent"
    
    if governance_status == "STABLE":
        governance_desc = "governed"
    elif governance_status == "MIXED":
        governance_desc = "partially governed"
    else:
        governance_desc = "under active change control"
    
    headline = (
        f"The curriculum has {frequency_desc} change activity "
        f"({total_groups} recorded changes across {slice_count} slices) "
        f"and appears {governance_desc}."
    )
    
    if has_block_events:
        block_count = len(index.get("slices_with_block_events", []))
        headline += f" {block_count} slice(s) have block-level drift events requiring review."
    
    return {
        "change_activity_band": activity_band,
        "governance_status": governance_status,
        "headline": headline,
    }


# =============================================================================
# FOLLOW-UP: Chronicle Causality Map & Multi-Axis Stability Estimator
# =============================================================================

class StabilityBand(str, Enum):
    """Stability band classification."""
    LOW = "LOW"          # Low stability, high volatility
    MEDIUM = "MEDIUM"    # Moderate stability
    HIGH = "HIGH"        # High stability, low volatility


# Causality inference parameters
CAUSAL_TIME_WINDOW_HOURS = 24  # Events within 24 hours are considered potentially causal
CAUSAL_STRENGTH_THRESHOLD = 0.5  # Minimum strength for causal link


def build_chronicle_causality_map(
    chronicle_index: Dict[str, Any],
    drift_events: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Build a causality map from chronicle index and drift events.
    
    Infers causal relationships between drift events based on:
    - Timestamp ordering (temporal adjacency)
    - Drift category relationships (e.g., STRUCTURAL → SEMANTIC)
    - Slice relationships (events in same slice more likely causal)
    
    Args:
        chronicle_index: Chronicle index from build_chronicle_index().
        drift_events: Dict mapping event_id -> event_data with:
            - timestamp: ISO timestamp string
            - drift_type: DriftType value
            - slice_name: str
            - snapshot_index: int
            - risk_level: RiskLevel value
    
    Returns:
        Dict with:
        - causal_links: List of tuples (event_a_id, event_b_id) representing causal relationships
        - likely_root_causes: List of event_ids that appear as root causes
        - causality_strength_score: float [0.0, 1.0] indicating overall causality strength
        - neutral_notes: List of descriptive notes about causality patterns
    """
    from datetime import datetime, timedelta
    
    causal_links: List[Tuple[str, str]] = []
    likely_root_causes: List[str] = []
    neutral_notes: List[str] = []
    
    if not drift_events:
        return {
            "causal_links": [],
            "likely_root_causes": [],
            "causality_strength_score": 0.0,
            "neutral_notes": ["No drift events available for causality analysis."]
        }
    
    # Sort events by timestamp for temporal analysis
    event_items = list(drift_events.items())
    sorted_events = sorted(
        event_items,
        key=lambda x: x[1].get("timestamp", "")
    )
    
    # Track which events have incoming causal links (to identify root causes)
    has_incoming_link: set = set()
    
    # Analyze temporal adjacency and drift type relationships
    for i, (event_a_id, event_a) in enumerate(sorted_events):
        timestamp_a_str = event_a.get("timestamp", "")
        drift_type_a = event_a.get("drift_type", "")
        slice_a = event_a.get("slice_name", "")
        risk_level_a = event_a.get("risk_level", "")
        
        if not timestamp_a_str:
            continue
        
        try:
            timestamp_a = datetime.fromisoformat(timestamp_a_str.replace('Z', '+00:00'))
        except (ValueError, AttributeError):
            continue
        
        # Look ahead for potentially causal events
        for j in range(i + 1, len(sorted_events)):
            event_b_id, event_b = sorted_events[j]
            timestamp_b_str = event_b.get("timestamp", "")
            drift_type_b = event_b.get("drift_type", "")
            slice_b = event_b.get("slice_name", "")
            
            if not timestamp_b_str:
                continue
            
            try:
                timestamp_b = datetime.fromisoformat(timestamp_b_str.replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                continue
            
            # Check temporal adjacency
            time_diff = timestamp_b - timestamp_a
            if time_diff.total_seconds() > CAUSAL_TIME_WINDOW_HOURS * 3600:
                break  # Events too far apart, stop looking ahead
            
            # Determine if causal relationship likely
            is_causal = False
            causal_reason = ""
            
            # Same slice with temporal proximity
            if slice_a == slice_b and time_diff.total_seconds() < 3600:  # Within 1 hour
                is_causal = True
                causal_reason = f"Temporal adjacency in same slice ({slice_a})"
            
            # Drift type causality patterns
            elif drift_type_a == DriftType.STRUCTURAL.value:
                # STRUCTURAL changes often cause SEMANTIC or PARAMETRIC changes
                if drift_type_b in [DriftType.SEMANTIC.value, DriftType.PARAMETRIC_MAJOR.value]:
                    is_causal = True
                    causal_reason = f"STRUCTURAL change likely caused {drift_type_b} change"
            
            elif drift_type_a == DriftType.SEMANTIC.value:
                # SEMANTIC changes might cause PARAMETRIC adjustments
                if drift_type_b == DriftType.PARAMETRIC_MAJOR.value:
                    is_causal = True
                    causal_reason = f"SEMANTIC change likely caused PARAMETRIC_MAJOR adjustment"
            
            # BLOCK-level events often cascade
            elif risk_level_a == RiskLevel.BLOCK.value:
                if drift_type_b in [DriftType.SEMANTIC.value, DriftType.STRUCTURAL.value]:
                    is_causal = True
                    causal_reason = f"BLOCK-level event likely triggered {drift_type_b} change"
            
            if is_causal:
                causal_links.append((event_a_id, event_b_id))
                has_incoming_link.add(event_b_id)
                if causal_reason:
                    neutral_notes.append(f"Link {event_a_id} → {event_b_id}: {causal_reason}")
    
    # Identify root causes (events with no incoming links)
    for event_id, _ in sorted_events:
        if event_id not in has_incoming_link:
            likely_root_causes.append(event_id)
    
    # Calculate causality strength score
    total_events = len(drift_events)
    if total_events == 0:
        causality_strength_score = 0.0
    else:
        # Score based on ratio of causal links to events
        link_ratio = len(causal_links) / total_events if total_events > 0 else 0.0
        # Normalize to [0.0, 1.0] with threshold
        causality_strength_score = min(1.0, link_ratio / CAUSAL_STRENGTH_THRESHOLD)
    
    # Add summary notes
    if causal_links:
        neutral_notes.append(
            f"Identified {len(causal_links)} potential causal relationships "
            f"across {total_events} drift events."
        )
    else:
        neutral_notes.append("No clear causal relationships detected in drift event sequence.")
    
    if likely_root_causes:
        neutral_notes.append(
            f"{len(likely_root_causes)} event(s) identified as likely root causes."
        )
    
    return {
        "causal_links": causal_links,
        "likely_root_causes": sorted(likely_root_causes),
        "causality_strength_score": round(causality_strength_score, 3),
        "neutral_notes": neutral_notes,
    }


def estimate_multi_axis_chronicle_stability(
    alignment_view: Dict[str, Any],
    causality_map: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Estimate multi-axis stability from alignment view and causality map.
    
    Combines multiple stability axes to provide a comprehensive stability assessment:
    - Alignment axis: alignment_status (STABLE/DRIFTY/VOLATILE)
    - Drift axis: presence and frequency of drift events
    - Churn axis: high edit churn patterns
    - Causality axis: causal link density and root cause count
    
    Args:
        alignment_view: Alignment view from build_chronicle_alignment_view().
        causality_map: Causality map from build_chronicle_causality_map().
    
    Returns:
        Dict with:
        - stability_band: "LOW" | "MEDIUM" | "HIGH"
        - axes_contributing: List of axis names contributing to stability assessment
        - headline: Neutral summary describing stability across axes
        - evidence_fields: Dict with evidence from each axis
    """
    axes_contributing: List[str] = []
    evidence_fields: Dict[str, Any] = {}
    
    # Axis 1: Alignment
    alignment_status = alignment_view.get("alignment_status", "UNKNOWN")
    high_churn_slices = len(alignment_view.get("slices_with_high_edit_churn", []))
    block_drift_slices = len(alignment_view.get("slices_with_block_drift_and_block_events", []))
    
    alignment_score = 1.0  # HIGH stability default
    if alignment_status == AlignmentStatus.VOLATILE.value:
        alignment_score = 0.0  # LOW stability
        axes_contributing.append("alignment")
    elif alignment_status == AlignmentStatus.DRIFTY.value:
        alignment_score = 0.5  # MEDIUM stability
        axes_contributing.append("alignment")
    
    evidence_fields["alignment"] = {
        "status": alignment_status,
        "high_churn_slices": high_churn_slices,
        "block_drift_slices": block_drift_slices,
        "score": alignment_score
    }
    
    # Axis 2: Causality
    causal_links = causality_map.get("causal_links", [])
    root_causes = len(causality_map.get("likely_root_causes", []))
    causality_strength = causality_map.get("causality_strength_score", 0.0)
    
    # High causality strength suggests instability (many cascading changes)
    causality_score = 1.0 - causality_strength  # Invert: high strength = low stability
    if causality_strength > 0.5:
        axes_contributing.append("causality")
    if root_causes > 3:
        axes_contributing.append("causality")
        causality_score = min(causality_score, 0.3)
    
    evidence_fields["causality"] = {
        "causal_link_count": len(causal_links),
        "root_cause_count": root_causes,
        "strength_score": causality_strength,
        "score": causality_score
    }
    
    # Axis 3: Churn
    churn_score = 1.0  # HIGH stability default
    if high_churn_slices > 0:
        churn_ratio = high_churn_slices / max(1, high_churn_slices + 1)  # Simplified
        churn_score = 1.0 - min(1.0, churn_ratio)
        if high_churn_slices >= 2:
            axes_contributing.append("churn")
            churn_score = max(0.0, churn_score - 0.3)
    
    evidence_fields["churn"] = {
        "high_churn_slice_count": high_churn_slices,
        "score": churn_score
    }
    
    # Axis 4: Drift
    drift_score = 1.0  # HIGH stability default
    if block_drift_slices > 0:
        drift_score = max(0.0, 1.0 - (block_drift_slices * 0.3))
        if block_drift_slices >= 2:
            axes_contributing.append("drift")
    
    evidence_fields["drift"] = {
        "block_drift_slice_count": block_drift_slices,
        "score": drift_score
    }
    
    # Calculate composite stability score
    axis_scores = [
        evidence_fields["alignment"]["score"],
        evidence_fields["causality"]["score"],
        evidence_fields["churn"]["score"],
        evidence_fields["drift"]["score"]
    ]
    composite_score = sum(axis_scores) / len(axis_scores)
    
    # Determine stability band
    if composite_score >= 0.7:
        stability_band = StabilityBand.HIGH
    elif composite_score >= 0.4:
        stability_band = StabilityBand.MEDIUM
    else:
        stability_band = StabilityBand.LOW
    
    # Build headline
    if stability_band == StabilityBand.HIGH:
        stability_desc = "high stability"
    elif stability_band == StabilityBand.MEDIUM:
        stability_desc = "moderate stability"
    else:
        stability_desc = "low stability"
    
    axes_desc = ", ".join(axes_contributing) if axes_contributing else "none"
    headline = (
        f"The curriculum shows {stability_desc} across multiple axes. "
        f"Contributing factors: {axes_desc if axes_desc != 'none' else 'minimal concerns'}. "
        f"Stability score: {composite_score:.2f}."
    )
    
    return {
        "stability_band": stability_band.value,
        "axes_contributing": sorted(list(set(axes_contributing))),
        "headline": headline,
        "evidence_fields": evidence_fields,
        "composite_score": round(composite_score, 3),
    }


# =============================================================================
# STRATCOM FOLLOW-UP: Dynamical Causality Grid & Multi-Phase Recurrence Projection
# =============================================================================

class RecurrenceBand(str, Enum):
    """Recurrence likelihood band classification."""
    LOW = "LOW"          # Low recurrence likelihood
    MEDIUM = "MEDIUM"    # Moderate recurrence likelihood
    HIGH = "HIGH"        # High recurrence likelihood


class InvariantStatus(str, Enum):
    """Invariant validation status."""
    OK = "OK"            # All invariants satisfied
    VIOLATED = "VIOLATED"  # One or more invariants violated


class StatusLight(str, Enum):
    """Director tile status light."""
    GREEN = "GREEN"      # All systems nominal
    YELLOW = "YELLOW"    # Attention required
    RED = "RED"          # Critical issues


# Recurrence projection parameters
RECURRENCE_HORIZON_BASE = 30  # Base horizon in days
RECURRENCE_DENSITY_THRESHOLD = 0.3  # Threshold for high event density


def build_recurrence_projection_engine(
    causality_map: Dict[str, Any],
    drift_events: Dict[str, Any],
    stability_scores: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Build recurrence projection from causality map, drift events, and stability scores.
    
    Projects likelihood of drift recurrence based on:
    - Causality strength (stronger causality → higher recurrence)
    - Drift event density (more events → higher recurrence)
    - Churn ratio (higher churn → higher recurrence)
    
    Formula: Recurrence ∝ causality_strength × drift_event_density × churn_ratio
    
    Args:
        causality_map: Causality map from build_chronicle_causality_map().
        drift_events: Dict of drift events (event_id -> event_data).
        stability_scores: Dict with stability evidence (from estimate_multi_axis_chronicle_stability).
    
    Returns:
        Dict with:
        - recurrence_likelihood: float [0.0, 1.0]
        - drivers: List of factors driving recurrence
        - projected_recurrence_horizon: int (days)
        - neutral_explanation: str
    """
    # Extract key metrics
    causality_strength = causality_map.get("causality_strength_score", 0.0)
    causal_links = causality_map.get("causal_links", [])
    root_causes = len(causality_map.get("likely_root_causes", []))
    
    total_events = len(drift_events)
    
    # Calculate drift event density
    # Density = events per unit time (normalized)
    if total_events > 0:
        # Estimate time span from events
        timestamps = [
            event.get("timestamp", "")
            for event in drift_events.values()
            if event.get("timestamp")
        ]
        if timestamps:
            from datetime import datetime
            try:
                sorted_timestamps = sorted(timestamps)
                if len(sorted_timestamps) > 1:
                    start = datetime.fromisoformat(sorted_timestamps[0].replace('Z', '+00:00'))
                    end = datetime.fromisoformat(sorted_timestamps[-1].replace('Z', '+00:00'))
                    time_span_days = (end - start).total_seconds() / 86400
                    if time_span_days > 0:
                        drift_event_density = total_events / time_span_days
                    else:
                        drift_event_density = total_events  # All events same day
                else:
                    drift_event_density = 1.0  # Single event
            except (ValueError, AttributeError):
                drift_event_density = total_events / 30.0  # Default: assume 30 days
        else:
            drift_event_density = total_events / 30.0
    else:
        drift_event_density = 0.0
    
    # Normalize density to [0, 1] range
    # High density = >1 event per day
    normalized_density = min(1.0, drift_event_density / RECURRENCE_DENSITY_THRESHOLD)
    
    # Calculate churn ratio from stability scores
    churn_evidence = stability_scores.get("evidence_fields", {}).get("churn", {})
    high_churn_slices = churn_evidence.get("high_churn_slice_count", 0)
    # Estimate churn ratio (simplified)
    churn_ratio = min(1.0, high_churn_slices / 5.0) if high_churn_slices > 0 else 0.0
    
    # Calculate recurrence likelihood
    # Recurrence ∝ causality_strength × drift_event_density × churn_ratio
    recurrence_likelihood = min(1.0, causality_strength * normalized_density * (1.0 + churn_ratio))
    
    # Identify drivers
    drivers = []
    if causality_strength > 0.5:
        drivers.append(f"High causality strength ({causality_strength:.2f})")
    if normalized_density > 0.5:
        drivers.append(f"Elevated drift event density ({normalized_density:.2f})")
    if churn_ratio > 0.3:
        drivers.append(f"High churn ratio ({churn_ratio:.2f})")
    if root_causes > 2:
        drivers.append(f"Multiple root causes ({root_causes})")
    if len(causal_links) > 5:
        drivers.append(f"Extensive causal links ({len(causal_links)})")
    
    if not drivers:
        drivers.append("Low recurrence indicators")
    
    # Project recurrence horizon
    # Higher likelihood → shorter horizon
    if recurrence_likelihood >= 0.7:
        horizon = int(RECURRENCE_HORIZON_BASE * 0.5)  # 15 days
    elif recurrence_likelihood >= 0.4:
        horizon = int(RECURRENCE_HORIZON_BASE * 0.75)  # 22 days
    else:
        horizon = RECURRENCE_HORIZON_BASE  # 30 days
    
    # Build neutral explanation
    if recurrence_likelihood >= 0.7:
        likelihood_desc = "high"
    elif recurrence_likelihood >= 0.4:
        likelihood_desc = "moderate"
    else:
        likelihood_desc = "low"
    
    explanation = (
        f"Recurrence likelihood is {likelihood_desc} ({recurrence_likelihood:.2f}). "
        f"Projected recurrence horizon: {horizon} days. "
        f"Primary drivers: {', '.join(drivers[:3])}."
    )
    
    return {
        "recurrence_likelihood": round(recurrence_likelihood, 3),
        "drivers": drivers,
        "projected_recurrence_horizon": horizon,
        "neutral_explanation": explanation,
    }


def build_phase_transition_drift_invariant_checker(
    drift_events: Dict[str, Any],
    causality_map: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Check phase-transition drift invariants.
    
    Invariants:
    1. No slice may show BLOCK drift twice within a 3-event window
    2. No STRUCTURAL event may cascade to PARAMETRIC_MAJOR > once per window
    3. No chronological inversion violations (events must be time-ordered)
    
    Args:
        drift_events: Dict of drift events (event_id -> event_data).
        causality_map: Causality map from build_chronicle_causality_map().
    
    Returns:
        Dict with:
        - invariant_status: "OK" | "VIOLATED"
        - broken_invariants: List of violated invariant descriptions
        - explanations: List of neutral explanations for violations
    """
    from datetime import datetime
    
    broken_invariants: List[str] = []
    explanations: List[str] = []
    
    if not drift_events:
        return {
            "invariant_status": InvariantStatus.OK.value,
            "broken_invariants": [],
            "explanations": ["No drift events to validate."]
        }
    
    # Sort events by timestamp for window analysis
    event_items = list(drift_events.items())
    sorted_events = sorted(
        event_items,
        key=lambda x: x[1].get("timestamp", "")
    )
    
    # Invariant 1: No slice may show BLOCK drift twice within a 3-event window
    window_size = 3
    for i in range(len(sorted_events) - window_size + 1):
        window_events = sorted_events[i:i + window_size]
        
        # Group by slice
        slice_block_counts: Dict[str, int] = {}
        for event_id, event_data in window_events:
            slice_name = event_data.get("slice_name", "")
            risk_level = event_data.get("risk_level", "")
            
            if risk_level == RiskLevel.BLOCK.value:
                slice_block_counts[slice_name] = slice_block_counts.get(slice_name, 0) + 1
        
        # Check for violations
        for slice_name, count in slice_block_counts.items():
            if count >= 2:
                broken_invariants.append(
                    f"BLOCK drift twice in 3-event window for slice '{slice_name}'"
                )
                explanations.append(
                    f"Slice '{slice_name}' shows {count} BLOCK-level drift events "
                    f"within a 3-event window, violating single-BLOCK-per-window invariant."
                )
    
    # Invariant 2: No STRUCTURAL event may cascade to PARAMETRIC_MAJOR > once per window
    causal_links = causality_map.get("causal_links", [])
    
    # Group causal links by source event
    structural_to_parametric: Dict[str, List[str]] = {}
    for event_a_id, event_b_id in causal_links:
        event_a = drift_events.get(event_a_id, {})
        event_b = drift_events.get(event_b_id, {})
        
        if (event_a.get("drift_type") == DriftType.STRUCTURAL.value and
            event_b.get("drift_type") == DriftType.PARAMETRIC_MAJOR.value):
            if event_a_id not in structural_to_parametric:
                structural_to_parametric[event_a_id] = []
            structural_to_parametric[event_a_id].append(event_b_id)
    
    # Check for violations (STRUCTURAL → PARAMETRIC_MAJOR > once per window)
    for struct_event_id, param_events in structural_to_parametric.items():
        if len(param_events) > 1:
            # Check if they're within a window
            struct_timestamp = drift_events.get(struct_event_id, {}).get("timestamp", "")
            param_timestamps = [
                drift_events.get(param_id, {}).get("timestamp", "")
                for param_id in param_events
            ]
            
            # If all within reasonable time window, it's a violation
            try:
                if struct_timestamp and all(param_timestamps):
                    struct_time = datetime.fromisoformat(struct_timestamp.replace('Z', '+00:00'))
                    param_times = [
                        datetime.fromisoformat(ts.replace('Z', '+00:00'))
                        for ts in param_timestamps if ts
                    ]
                    
                    if param_times:
                        max_time_diff = max((pt - struct_time).total_seconds() for pt in param_times)
                        if max_time_diff < 86400 * 7:  # Within 7 days
                            broken_invariants.append(
                                f"STRUCTURAL event '{struct_event_id}' cascades to "
                                f"PARAMETRIC_MAJOR {len(param_events)} times"
                            )
                            explanations.append(
                                f"STRUCTURAL event '{struct_event_id}' triggers multiple "
                                f"PARAMETRIC_MAJOR changes ({len(param_events)}), "
                                f"violating single-cascade-per-window invariant."
                            )
            except (ValueError, AttributeError):
                pass  # Skip if timestamp parsing fails
    
    # Invariant 3: No chronological inversion violations
    # Events in causality links should respect temporal ordering
    for event_a_id, event_b_id in causal_links:
        event_a = drift_events.get(event_a_id, {})
        event_b = drift_events.get(event_b_id, {})
        
        timestamp_a = event_a.get("timestamp", "")
        timestamp_b = event_b.get("timestamp", "")
        
        if timestamp_a and timestamp_b:
            try:
                time_a = datetime.fromisoformat(timestamp_a.replace('Z', '+00:00'))
                time_b = datetime.fromisoformat(timestamp_b.replace('Z', '+00:00'))
                
                if time_a > time_b:
                    broken_invariants.append(
                        f"Chronological inversion: {event_a_id} ({timestamp_a}) "
                        f"→ {event_b_id} ({timestamp_b})"
                    )
                    explanations.append(
                        f"Causal link {event_a_id} → {event_b_id} violates chronological "
                        f"ordering: source event occurs after target event."
                    )
            except (ValueError, AttributeError):
                pass  # Skip if timestamp parsing fails
    
    # Determine overall status
    invariant_status = InvariantStatus.VIOLATED if broken_invariants else InvariantStatus.OK
    
    if invariant_status == InvariantStatus.OK:
        explanations.append("All phase-transition drift invariants satisfied.")
    
    return {
        "invariant_status": invariant_status.value,
        "broken_invariants": sorted(broken_invariants),
        "explanations": explanations,
    }


def build_director_tile(
    recurrence_projection: Dict[str, Any],
    invariant_check: Dict[str, Any],
    stability_estimate: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Build director tile for high-level governance dashboard.
    
    Combines recurrence projection, invariant status, and stability estimate
    into a single executive summary tile.
    
    Args:
        recurrence_projection: From build_recurrence_projection_engine().
        invariant_check: From build_phase_transition_drift_invariant_checker().
        stability_estimate: From estimate_multi_axis_chronicle_stability().
    
    Returns:
        Dict with:
        - status_light: "GREEN" | "YELLOW" | "RED"
        - recurrence_band: "LOW" | "MEDIUM" | "HIGH"
        - invariants_ok: bool
        - highlighted_cases: List of notable cases
        - headline: str (neutral summary)
    """
    # Determine recurrence band
    recurrence_likelihood = recurrence_projection.get("recurrence_likelihood", 0.0)
    if recurrence_likelihood >= 0.7:
        recurrence_band = RecurrenceBand.HIGH
    elif recurrence_likelihood >= 0.4:
        recurrence_band = RecurrenceBand.MEDIUM
    else:
        recurrence_band = RecurrenceBand.LOW
    
    # Check invariants
    invariants_ok = invariant_check.get("invariant_status") == InvariantStatus.OK.value
    broken_invariants = invariant_check.get("broken_invariants", [])
    
    # Determine status light
    stability_band = stability_estimate.get("stability_band", StabilityBand.MEDIUM.value)
    
    if not invariants_ok:
        status_light = StatusLight.RED
    elif recurrence_band == RecurrenceBand.HIGH or stability_band == StabilityBand.LOW.value:
        status_light = StatusLight.YELLOW
    else:
        status_light = StatusLight.GREEN
    
    # Build highlighted cases
    highlighted_cases: List[str] = []
    
    if recurrence_band == RecurrenceBand.HIGH:
        drivers = recurrence_projection.get("drivers", [])
        if drivers:
            highlighted_cases.append(f"High recurrence likelihood: {', '.join(drivers[:2])}")
    
    if broken_invariants:
        highlighted_cases.append(f"{len(broken_invariants)} invariant violation(s) detected")
    
    stability_axes = stability_estimate.get("axes_contributing", [])
    if stability_axes:
        highlighted_cases.append(f"Stability concerns: {', '.join(stability_axes)}")
    
    if not highlighted_cases:
        highlighted_cases.append("No notable concerns")
    
    # Build headline
    status_desc = status_light.value.lower()
    recurrence_desc = recurrence_likelihood
    horizon = recurrence_projection.get("projected_recurrence_horizon", 30)
    
    headline = (
        f"Chronicle status: {status_desc}. "
        f"Recurrence likelihood: {recurrence_desc:.2f} "
        f"(projected horizon: {horizon} days). "
        f"Invariants: {'satisfied' if invariants_ok else 'violated'}."
    )
    
    return {
        "status_light": status_light.value,
        "recurrence_band": recurrence_band.value,
        "invariants_ok": invariants_ok,
        "highlighted_cases": highlighted_cases,
        "headline": headline,
    }


# Semantic keys that trigger SEMANTIC drift when changed
SEMANTIC_KEYS = frozenset([
    "success_metric", "kind", "target_hash", "target_hashes",
    "formula_pool", "formula_pool_entries", "prereg_hash"
])

# Thresholds for parametric classification
PARAMETRIC_MINOR_THRESHOLD = 0.1  # 10% change is minor
PARAMETRIC_MAJOR_THRESHOLD = 0.5  # 50% change is major


class LedgerSigner:
    """
    Cryptographic signing for ledger snapshots using Ed25519.
    
    Signing is optional and controlled by LEDGER_SIGNING=1 environment variable.
    Uses deterministic Ed25519 signatures for reproducibility.
    """
    
    def __init__(
        self,
        private_key_path: Optional[Path] = None,
        public_key_path: Optional[Path] = None
    ):
        """
        Initialize the signer with key paths.
        
        Args:
            private_key_path: Path to Ed25519 private key (PEM format).
            public_key_path: Path to Ed25519 public key (PEM format).
        """
        self.private_key_path = private_key_path or DEFAULT_PRIVATE_KEY_PATH
        self.public_key_path = public_key_path or DEFAULT_PUBLIC_KEY_PATH
        self._private_key = None
        self._public_key = None
    
    @staticmethod
    def is_signing_enabled() -> bool:
        """Check if signing is enabled via environment variable."""
        return os.environ.get(LEDGER_SIGNING_ENV, "0") == "1"
    
    def _load_private_key(self):
        """Load the private key from file."""
        if self._private_key is not None:
            return self._private_key
        
        try:
            from cryptography.hazmat.primitives import serialization
            
            with open(self.private_key_path, 'rb') as f:
                self._private_key = serialization.load_pem_private_key(
                    f.read(),
                    password=None
                )
            return self._private_key
        except ImportError:
            raise ImportError(
                "cryptography package required for signing. "
                "Install with: pip install cryptography"
            )
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Private key not found: {self.private_key_path}. "
                f"Generate with: python -c \"from experiments.curriculum_hash_ledger import LedgerSigner; LedgerSigner.generate_test_keypair()\""
            )
    
    def _load_public_key(self):
        """Load the public key from file."""
        if self._public_key is not None:
            return self._public_key
        
        try:
            from cryptography.hazmat.primitives import serialization
            
            with open(self.public_key_path, 'rb') as f:
                self._public_key = serialization.load_pem_public_key(f.read())
            return self._public_key
        except ImportError:
            raise ImportError(
                "cryptography package required for verification. "
                "Install with: pip install cryptography"
            )
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Public key not found: {self.public_key_path}"
            )
    
    @classmethod
    def generate_test_keypair(
        cls,
        private_key_path: Optional[Path] = None,
        public_key_path: Optional[Path] = None
    ) -> Tuple[Path, Path]:
        """
        Generate a test Ed25519 keypair for development.
        
        WARNING: These keys are for testing only. Do not use in production
        without proper key management.
        
        Returns:
            Tuple of (private_key_path, public_key_path)
        """
        try:
            from cryptography.hazmat.primitives import serialization
            from cryptography.hazmat.primitives.asymmetric import ed25519
        except ImportError:
            raise ImportError(
                "cryptography package required. Install with: pip install cryptography"
            )
        
        priv_path = private_key_path or DEFAULT_PRIVATE_KEY_PATH
        pub_path = public_key_path or DEFAULT_PUBLIC_KEY_PATH
        
        # Ensure directory exists
        priv_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate keypair
        private_key = ed25519.Ed25519PrivateKey.generate()
        public_key = private_key.public_key()
        
        # Serialize and save private key
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        with open(priv_path, 'wb') as f:
            f.write(private_pem)
        
        # Serialize and save public key
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        with open(pub_path, 'wb') as f:
            f.write(public_pem)
        
        print(f"Generated test keypair:")
        print(f"  Private key: {priv_path}")
        print(f"  Public key:  {pub_path}")
        print(f"  WARNING: These keys are for testing only!")
        
        return priv_path, pub_path
    
    def sign_snapshot(self, snapshot: Dict[str, Any]) -> str:
        """
        Sign a snapshot and return the base64-encoded signature.
        
        The signature is computed over:
        DOMAIN_SIGNATURE || SHA256(canonical_json(snapshot))
        
        Args:
            snapshot: The snapshot dict to sign.
        
        Returns:
            Base64-encoded Ed25519 signature string.
        """
        private_key = self._load_private_key()
        
        # Create deterministic message to sign
        canonical = canonical_json(snapshot)
        message_hash = hashlib.sha256(DOMAIN_SIGNATURE + canonical).digest()
        
        # Sign the hash
        signature = private_key.sign(message_hash)
        
        return base64.b64encode(signature).decode('ascii')
    
    def verify_signature(
        self,
        snapshot: Dict[str, Any],
        signature: str
    ) -> bool:
        """
        Verify a snapshot signature.
        
        Args:
            snapshot: The snapshot dict that was signed.
            signature: Base64-encoded signature string.
        
        Returns:
            True if signature is valid, False otherwise.
        """
        try:
            from cryptography.exceptions import InvalidSignature
        except ImportError:
            raise ImportError(
                "cryptography package required for verification."
            )
        
        public_key = self._load_public_key()
        
        # Recreate the signed message
        canonical = canonical_json(snapshot)
        message_hash = hashlib.sha256(DOMAIN_SIGNATURE + canonical).digest()
        
        # Decode signature
        try:
            sig_bytes = base64.b64decode(signature)
        except Exception:
            return False
        
        # Verify
        try:
            public_key.verify(sig_bytes, message_hash)
            return True
        except InvalidSignature:
            return False


def canonical_json(obj: Any) -> bytes:
    """
    Canonical JSON representation for deterministic hashing.
    Follows INV-CANON-1 from HT_INVARIANT_SPEC_v1.md.
    """
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(',', ':'),
        ensure_ascii=False
    ).encode('utf-8')


def get_git_commit() -> str:
    """Get current git commit SHA, or 'unknown' if not in a git repo."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


class CurriculumHashLedger:
    """
    Records and compares curriculum hash snapshots over time.
    
    The ledger is an append-only JSONL file that tracks:
    - Global curriculum hash
    - Per-slice hashes
    - Timestamps, git commits, and provenance
    
    This enables drift detection: if any slice definition, formula pool,
    or target hash changes between snapshots, it will be surfaced.
    """
    
    DEFAULT_LEDGER_PATH = Path("artifacts/phase_ii/curriculum_hash_ledger.jsonl")
    
    def __init__(self, ledger_path: Optional[Path] = None):
        """
        Initialize the ledger.
        
        Args:
            ledger_path: Path to the JSONL ledger file. Defaults to
                         artifacts/phase_ii/curriculum_hash_ledger.jsonl
        """
        self.ledger_path = ledger_path or self.DEFAULT_LEDGER_PATH
    
    def _load_config(self, config_path: Path) -> Dict[str, Any]:
        """Load a curriculum config file (YAML)."""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def compute_slice_hash(self, slice_name: str, slice_config: Dict[str, Any]) -> str:
        """
        Compute SHA256 hash of a single slice configuration.
        
        Uses domain separation per HT_INVARIANT_SPEC_v1.md conventions.
        """
        # Include slice name in the hash to ensure different slices with
        # identical configs produce different hashes
        payload = {
            "name": slice_name,
            "config": slice_config
        }
        canonical = canonical_json(payload)
        return hashlib.sha256(DOMAIN_SLICE + canonical).hexdigest()
    
    def compute_curriculum_hash(self, config_path: str) -> Tuple[str, Dict[str, str]]:
        """
        Compute the global curriculum hash and per-slice hashes.
        
        Args:
            config_path: Path to the curriculum YAML file.
        
        Returns:
            Tuple of (global_hash, slice_hashes_dict)
            where slice_hashes_dict maps slice_name -> hash
        """
        config = self._load_config(Path(config_path))
        
        # Handle different curriculum file structures
        # Structure 1: Has 'slices' key (like curriculum_uplift_phase2.yaml with dict slices)
        # Structure 2: Has 'systems' with nested slices (like config/curriculum.yaml)
        # Structure 3: Is a list (like curriculum_uplift_phase2.yaml root list)
        
        slice_hashes: Dict[str, str] = {}
        
        if isinstance(config, list):
            # Root is a list (like curriculum_uplift_phase2.yaml)
            for i, item in enumerate(config):
                item_name = item.get('curriculum_id', f'item_{i}')
                slice_hashes[item_name] = self.compute_slice_hash(item_name, item)
        elif 'slices' in config and isinstance(config['slices'], dict):
            # Dict-style slices
            for slice_name, slice_config in config['slices'].items():
                slice_hashes[slice_name] = self.compute_slice_hash(slice_name, slice_config)
        elif 'slices' in config and isinstance(config['slices'], list):
            # List-style slices
            for slice_config in config['slices']:
                slice_name = slice_config.get('name', f'unnamed_{len(slice_hashes)}')
                slice_hashes[slice_name] = self.compute_slice_hash(slice_name, slice_config)
        elif 'systems' in config:
            # Systems structure (like config/curriculum.yaml)
            for system_name, system_config in config['systems'].items():
                if 'slices' in system_config:
                    for slice_config in system_config['slices']:
                        slice_name = slice_config.get('name', f'{system_name}_unnamed')
                        full_name = f"{system_name}/{slice_name}"
                        slice_hashes[full_name] = self.compute_slice_hash(full_name, slice_config)
        else:
            # Fallback: hash the entire config as a single "slice"
            slice_hashes["_root"] = self.compute_slice_hash("_root", config)
        
        # Compute global hash from sorted slice hashes
        # This ensures the global hash is deterministic
        sorted_slice_data = sorted(slice_hashes.items())
        global_payload = {
            "config_path": str(config_path),
            "slice_hashes": dict(sorted_slice_data)
        }
        global_hash = hashlib.sha256(
            DOMAIN_CURRICULUM + canonical_json(global_payload)
        ).hexdigest()
        
        return global_hash, slice_hashes
    
    def record_snapshot(
        self,
        config_path: str,
        timestamp: Optional[str] = None,
        git_commit: Optional[str] = None,
        origin: str = "manual",
        notes: str = "",
        signer: Optional[LedgerSigner] = None
    ) -> Dict[str, Any]:
        """
        Record a curriculum snapshot to the ledger.
        
        Args:
            config_path: Path to the curriculum YAML file.
            timestamp: ISO8601 timestamp (auto-generated if None).
            git_commit: Git commit SHA (auto-detected if None).
            origin: Source of the snapshot ("manual", "ci", "pre-commit").
            notes: Free-text notes about this snapshot.
            signer: Optional LedgerSigner for cryptographic signing.
                    If None and LEDGER_SIGNING=1, auto-creates signer.
        
        Returns:
            The snapshot entry that was recorded.
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        
        if git_commit is None:
            git_commit = get_git_commit()
        
        curriculum_hash, slice_hashes = self.compute_curriculum_hash(config_path)
        
        entry = {
            "timestamp": timestamp,
            "config_path": str(config_path),
            "curriculum_hash": curriculum_hash,
            "git_commit": git_commit,
            "slice_hashes": slice_hashes,
            "origin": origin,
            "notes": notes
        }
        
        # Sign if enabled
        if signer is None and LedgerSigner.is_signing_enabled():
            try:
                signer = LedgerSigner()
            except Exception as e:
                print(f"Warning: Signing enabled but failed to initialize: {e}", file=sys.stderr)
        
        if signer is not None:
            try:
                signature = signer.sign_snapshot(entry)
                entry["signature"] = signature
                entry["signed"] = True
            except Exception as e:
                print(f"Warning: Failed to sign snapshot: {e}", file=sys.stderr)
                entry["signed"] = False
        else:
            entry["signed"] = False
        
        # Ensure ledger directory exists
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Append to ledger (append-only)
        with open(self.ledger_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, sort_keys=True) + '\n')
        
        return entry
    
    def verify_snapshot_signature(
        self,
        snapshot: Dict[str, Any],
        signer: Optional[LedgerSigner] = None
    ) -> Tuple[bool, str]:
        """
        Verify the signature on a snapshot.
        
        Args:
            snapshot: The snapshot dict to verify.
            signer: Optional LedgerSigner (auto-creates if None).
        
        Returns:
            Tuple of (is_valid, message).
        """
        if not snapshot.get("signed", False):
            return False, "Snapshot is not signed"
        
        signature = snapshot.get("signature")
        if not signature:
            return False, "Snapshot marked as signed but no signature present"
        
        # Create a copy without the signature field for verification
        snapshot_copy = {k: v for k, v in snapshot.items() if k not in ("signature", "signed")}
        
        if signer is None:
            signer = LedgerSigner()
        
        try:
            if signer.verify_signature(snapshot_copy, signature):
                return True, "Signature valid"
            else:
                return False, "Invalid signature"
        except Exception as e:
            return False, f"Verification error: {e}"
    
    def load_snapshots(self) -> List[Dict[str, Any]]:
        """Load all snapshots from the ledger."""
        if not self.ledger_path.exists():
            return []
        
        snapshots = []
        with open(self.ledger_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    snapshots.append(json.loads(line))
        return snapshots
    
    def get_snapshot(self, index_or_timestamp: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific snapshot by index or timestamp.
        
        Args:
            index_or_timestamp: Either an integer index (supports negative),
                               or an ISO8601 timestamp string.
        
        Returns:
            The snapshot dict, or None if not found.
        """
        snapshots = self.load_snapshots()
        if not snapshots:
            return None
        
        # Try to interpret as integer index
        try:
            idx = int(index_or_timestamp)
            if -len(snapshots) <= idx < len(snapshots):
                return snapshots[idx]
            return None
        except ValueError:
            pass
        
        # Search by timestamp
        for snap in snapshots:
            if snap.get("timestamp") == index_or_timestamp:
                return snap
        
        return None
    
    def compare_snapshots(
        self,
        old_snapshot: Dict[str, Any],
        new_snapshot: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare two snapshots and produce a drift report.
        
        Args:
            old_snapshot: The older snapshot dict.
            new_snapshot: The newer snapshot dict.
        
        Returns:
            A dict containing:
            - has_drift: bool
            - global_hash_changed: bool
            - added_slices: list of slice names
            - removed_slices: list of slice names
            - changed_slices: list of slice names whose hash changed
            - old_timestamp, new_timestamp
            - old_git_commit, new_git_commit
        """
        old_slices = old_snapshot.get("slice_hashes", {})
        new_slices = new_snapshot.get("slice_hashes", {})
        
        old_keys = set(old_slices.keys())
        new_keys = set(new_slices.keys())
        
        added = sorted(new_keys - old_keys)
        removed = sorted(old_keys - new_keys)
        
        # Check for hash changes in common slices
        common = old_keys & new_keys
        changed = []
        for name in sorted(common):
            if old_slices[name] != new_slices[name]:
                changed.append(name)
        
        global_hash_changed = (
            old_snapshot.get("curriculum_hash") != new_snapshot.get("curriculum_hash")
        )
        
        has_drift = bool(added or removed or changed)
        
        return {
            "has_drift": has_drift,
            "global_hash_changed": global_hash_changed,
            "added_slices": added,
            "removed_slices": removed,
            "changed_slices": changed,
            "old_timestamp": old_snapshot.get("timestamp"),
            "new_timestamp": new_snapshot.get("timestamp"),
            "old_git_commit": old_snapshot.get("git_commit"),
            "new_git_commit": new_snapshot.get("git_commit"),
            "old_curriculum_hash": old_snapshot.get("curriculum_hash"),
            "new_curriculum_hash": new_snapshot.get("curriculum_hash")
        }
    
    def classify_slice_drift(
        self,
        old_config: Dict[str, Any],
        new_config: Dict[str, Any]
    ) -> Tuple[DriftType, List[str]]:
        """
        Classify the type of drift between two slice configurations.
        
        Args:
            old_config: The old slice configuration dict.
            new_config: The new slice configuration dict.
        
        Returns:
            Tuple of (DriftType, list of changed keys).
        """
        if old_config == new_config:
            return DriftType.NONE, []
        
        # Check for identical canonical representation (COSMETIC)
        old_canonical = canonical_json(old_config)
        new_canonical = canonical_json(new_config)
        if old_canonical == new_canonical:
            return DriftType.COSMETIC, []
        
        # Find all changed keys recursively
        changed_keys = self._find_changed_keys(old_config, new_config)
        
        # Check for semantic changes (high priority)
        semantic_changes = [k for k in changed_keys if self._is_semantic_key(k)]
        if semantic_changes:
            return DriftType.SEMANTIC, changed_keys
        
        # Check parametric changes
        param_severity = self._classify_parametric_changes(old_config, new_config, changed_keys)
        if param_severity == "major":
            return DriftType.PARAMETRIC_MAJOR, changed_keys
        elif param_severity == "minor":
            return DriftType.PARAMETRIC_MINOR, changed_keys
        
        # Default to parametric minor for any other changes
        return DriftType.PARAMETRIC_MINOR, changed_keys
    
    def _is_semantic_key(self, key: str) -> bool:
        """Check if a key is considered semantic (affects experiment meaning)."""
        # Check the leaf key name
        leaf = key.split('.')[-1] if '.' in key else key
        return leaf in SEMANTIC_KEYS
    
    def _find_changed_keys(
        self,
        old: Any,
        new: Any,
        prefix: str = ""
    ) -> List[str]:
        """Recursively find all keys that differ between old and new."""
        changed = []
        
        if type(old) != type(new):
            return [prefix] if prefix else ["_root"]
        
        if isinstance(old, dict):
            all_keys = set(old.keys()) | set(new.keys())
            for k in all_keys:
                key_path = f"{prefix}.{k}" if prefix else k
                if k not in old:
                    changed.append(f"+{key_path}")
                elif k not in new:
                    changed.append(f"-{key_path}")
                elif old[k] != new[k]:
                    if isinstance(old[k], dict) and isinstance(new[k], dict):
                        changed.extend(self._find_changed_keys(old[k], new[k], key_path))
                    else:
                        changed.append(key_path)
        elif isinstance(old, list):
            if old != new:
                changed.append(prefix if prefix else "_list")
        elif old != new:
            changed.append(prefix if prefix else "_value")
        
        return changed
    
    def _classify_parametric_changes(
        self,
        old: Dict[str, Any],
        new: Dict[str, Any],
        changed_keys: List[str]
    ) -> str:
        """
        Classify parametric changes as 'minor' or 'major'.
        
        Returns 'major' if any removed params or >50% value changes.
        Returns 'minor' if changes are <10% value differences.
        """
        # Removed keys are major
        if any(k.startswith('-') for k in changed_keys):
            return "major"
        
        # Check numeric value changes
        max_change_ratio = 0.0
        for key in changed_keys:
            if key.startswith('+') or key.startswith('-'):
                continue
            
            old_val = self._get_nested_value(old, key)
            new_val = self._get_nested_value(new, key)
            
            if isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
                if old_val != 0:
                    ratio = abs(new_val - old_val) / abs(old_val)
                    max_change_ratio = max(max_change_ratio, ratio)
                elif new_val != 0:
                    max_change_ratio = 1.0  # Changed from 0 to non-zero
        
        if max_change_ratio >= PARAMETRIC_MAJOR_THRESHOLD:
            return "major"
        elif max_change_ratio >= PARAMETRIC_MINOR_THRESHOLD:
            return "minor"
        
        # Default: if any keys changed, it's minor
        return "minor" if changed_keys else "none"
    
    def _get_nested_value(self, obj: Dict[str, Any], key_path: str) -> Any:
        """Get a nested value from a dict using dot notation."""
        keys = key_path.split('.')
        current = obj
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return None
        return current
    
    def classify_drift(
        self,
        old_snapshot: Dict[str, Any],
        new_snapshot: Dict[str, Any],
        old_config: Optional[Dict[str, Any]] = None,
        new_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Produce a comprehensive drift classification report.
        
        Args:
            old_snapshot: The older snapshot dict.
            new_snapshot: The newer snapshot dict.
            old_config: Optional loaded old config (for deep comparison).
            new_config: Optional loaded new config (for deep comparison).
        
        Returns:
            Extended diff dict with drift_type, affected_slices, risk_level.
        """
        # Get basic diff
        diff = self.compare_snapshots(old_snapshot, new_snapshot)
        
        # Initialize classification fields
        affected_slices: Dict[str, Dict[str, Any]] = {}
        overall_drift_type = DriftType.NONE
        
        # Structural changes (added/removed slices)
        if diff["added_slices"] or diff["removed_slices"]:
            overall_drift_type = DriftType.STRUCTURAL
            for name in diff["added_slices"]:
                affected_slices[name] = {
                    "drift_type": DriftType.STRUCTURAL.value,
                    "change": "added",
                    "changed_keys": []
                }
            for name in diff["removed_slices"]:
                affected_slices[name] = {
                    "drift_type": DriftType.STRUCTURAL.value,
                    "change": "removed",
                    "changed_keys": []
                }
        
        # Changed slices (need config for deep analysis)
        if diff["changed_slices"] and old_config and new_config:
            old_slices_map = self._extract_slices(old_config)
            new_slices_map = self._extract_slices(new_config)
            
            for name in diff["changed_slices"]:
                old_slice = old_slices_map.get(name, {})
                new_slice = new_slices_map.get(name, {})
                
                drift_type, changed_keys = self.classify_slice_drift(old_slice, new_slice)
                affected_slices[name] = {
                    "drift_type": drift_type.value,
                    "change": "modified",
                    "changed_keys": changed_keys
                }
                
                # Update overall drift type (take highest severity)
                if self._drift_severity(drift_type) > self._drift_severity(overall_drift_type):
                    overall_drift_type = drift_type
        elif diff["changed_slices"]:
            # No configs provided, classify as unknown/semantic
            for name in diff["changed_slices"]:
                affected_slices[name] = {
                    "drift_type": DriftType.SEMANTIC.value,
                    "change": "modified",
                    "changed_keys": ["_unknown_without_config"]
                }
            overall_drift_type = DriftType.SEMANTIC
        
        # Determine risk level
        risk_level = DRIFT_RISK_MAP.get(overall_drift_type, RiskLevel.BLOCK)
        
        # Add classification to diff
        diff["drift_type"] = overall_drift_type.value
        diff["affected_slices"] = affected_slices
        diff["risk_level"] = risk_level.value
        
        return diff
    
    def _extract_slices(self, config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Extract slice configs from various curriculum formats."""
        slices_map: Dict[str, Dict[str, Any]] = {}
        
        if isinstance(config, list):
            for i, item in enumerate(config):
                name = item.get('curriculum_id', f'item_{i}')
                slices_map[name] = item
        elif 'slices' in config and isinstance(config['slices'], dict):
            slices_map = dict(config['slices'])
        elif 'slices' in config and isinstance(config['slices'], list):
            for sc in config['slices']:
                name = sc.get('name', f'unnamed_{len(slices_map)}')
                slices_map[name] = sc
        elif 'systems' in config:
            for system_name, system_config in config['systems'].items():
                if 'slices' in system_config:
                    for sc in system_config['slices']:
                        name = sc.get('name', f'{system_name}_unnamed')
                        full_name = f"{system_name}/{name}"
                        slices_map[full_name] = sc
        else:
            slices_map["_root"] = config
        
        return slices_map
    
    def _drift_severity(self, drift_type: DriftType) -> int:
        """Return numeric severity for drift type comparison."""
        severity_order = [
            DriftType.NONE,
            DriftType.COSMETIC,
            DriftType.PARAMETRIC_MINOR,
            DriftType.PARAMETRIC_MAJOR,
            DriftType.SEMANTIC,
            DriftType.STRUCTURAL,
        ]
        return severity_order.index(drift_type) if drift_type in severity_order else 0
    
    def build_drift_timeline(
        self,
        slice_name: str,
        include_no_change: bool = False
    ) -> List[DriftEvent]:
        """
        Build a chronological timeline of drift events for a specific slice.
        
        Analyzes all consecutive snapshot pairs and records drift events
        where the specified slice changed.
        
        Args:
            slice_name: Name of the slice to track (e.g., "slice_uplift_goal").
            include_no_change: If True, include NONE events (no change).
        
        Returns:
            List of DriftEvent objects sorted by timestamp (oldest first).
        """
        snapshots = self.load_snapshots()
        if len(snapshots) < 2:
            return []
        
        timeline: List[DriftEvent] = []
        
        for i in range(1, len(snapshots)):
            old_snap = snapshots[i - 1]
            new_snap = snapshots[i]
            
            old_slices = old_snap.get("slice_hashes", {})
            new_slices = new_snap.get("slice_hashes", {})
            
            old_hash = old_slices.get(slice_name)
            new_hash = new_slices.get(slice_name)
            
            # Initialize for each iteration
            changed_keys_list: List[str] = []
            
            # Determine change type
            if old_hash is None and new_hash is None:
                # Slice doesn't exist in either snapshot
                continue
            elif old_hash is None and new_hash is not None:
                # Slice was added
                change_type = "added"
                drift_type = DriftType.STRUCTURAL
                old_hash = "N/A"
            elif old_hash is not None and new_hash is None:
                # Slice was removed
                change_type = "removed"
                drift_type = DriftType.STRUCTURAL
                new_hash = "N/A"
            elif old_hash == new_hash:
                # No change
                if not include_no_change:
                    continue
                change_type = "unchanged"
                drift_type = DriftType.NONE
            else:
                # Slice was modified - try to load configs for detailed classification
                change_type = "modified"
                
                try:
                    old_config_path = Path(old_snap.get("config_path", ""))
                    new_config_path = Path(new_snap.get("config_path", ""))
                    
                    if old_config_path.exists() and new_config_path.exists():
                        old_config = self._load_config(old_config_path)
                        new_config = self._load_config(new_config_path)
                        
                        old_slices_cfg = self._extract_slices(old_config)
                        new_slices_cfg = self._extract_slices(new_config)
                        
                        old_slice_config = old_slices_cfg.get(slice_name, {})
                        new_slice_config = new_slices_cfg.get(slice_name, {})
                        
                        drift_type, changed_keys_list = self.classify_slice_drift(
                            old_slice_config, new_slice_config
                        )
                    else:
                        # Config files not available, default to SEMANTIC
                        drift_type = DriftType.SEMANTIC
                except Exception:
                    # On any error, default to SEMANTIC
                    drift_type = DriftType.SEMANTIC
            
            risk_level = DRIFT_RISK_MAP.get(drift_type, RiskLevel.INFO)
            
            event = DriftEvent(
                timestamp=new_snap.get("timestamp", ""),
                old_hash=old_hash or "N/A",
                new_hash=new_hash or "N/A",
                drift_type=drift_type,
                risk_level=risk_level,
                snapshot_index=i,
                git_commit=new_snap.get("git_commit", "unknown"),
                change_type=change_type,
                changed_keys=changed_keys_list,
                notes=new_snap.get("notes", "")
            )
            timeline.append(event)
        
        # Already sorted by snapshot index (chronological)
        return timeline
    
    def build_all_slices_timeline(
        self,
        include_no_change: bool = False
    ) -> Dict[str, List[DriftEvent]]:
        """
        Build drift timelines for all slices in the ledger.
        
        Returns:
            Dict mapping slice_name -> list of DriftEvent.
        """
        snapshots = self.load_snapshots()
        if not snapshots:
            return {}
        
        # Collect all slice names across all snapshots
        all_slices: set = set()
        for snap in snapshots:
            all_slices.update(snap.get("slice_hashes", {}).keys())
        
        # Build timeline for each slice
        result = {}
        for slice_name in sorted(all_slices):
            timeline = self.build_drift_timeline(slice_name, include_no_change)
            if timeline or include_no_change:
                result[slice_name] = timeline
        
        return result
    
    def format_timeline(
        self,
        timeline: List[DriftEvent],
        slice_name: str
    ) -> str:
        """
        Format a drift timeline as human-readable text.
        
        Args:
            timeline: List of DriftEvent objects.
            slice_name: Name of the slice for the header.
        
        Returns:
            Formatted string representation.
        """
        lines = []
        lines.append(f"# Drift Timeline: {slice_name}")
        lines.append("")
        
        if not timeline:
            lines.append("No drift events recorded for this slice.")
            return "\n".join(lines)
        
        # Summary stats
        block_count = sum(1 for e in timeline if e.risk_level == RiskLevel.BLOCK)
        warn_count = sum(1 for e in timeline if e.risk_level == RiskLevel.WARN)
        info_count = sum(1 for e in timeline if e.risk_level == RiskLevel.INFO)
        
        lines.append(f"**Total Events**: {len(timeline)}")
        lines.append(f"**Risk Summary**: {block_count} BLOCK, {warn_count} WARN, {info_count} INFO")
        lines.append("")
        
        # Header
        lines.append("```")
        lines.append("IDX   TIMESTAMP                    RISK DRIFT_TYPE         HASH_CHANGE          TYPE")
        lines.append("-" * 95)
        
        for event in timeline:
            lines.append(event.format_line())
        
        lines.append("```")
        lines.append("")
        
        # Details for BLOCK events
        block_events = [e for e in timeline if e.risk_level == RiskLevel.BLOCK]
        if block_events:
            lines.append("## BLOCK-Level Events (Require Review)")
            lines.append("")
            for event in block_events:
                lines.append(f"- **[{event.snapshot_index}]** {event.timestamp}")
                lines.append(f"  - Type: {event.drift_type.value} ({event.change_type})")
                lines.append(f"  - Git: `{event.git_commit[:12]}...`")
                if event.notes:
                    lines.append(f"  - Notes: {event.notes}")
                lines.append("")
        
        return "\n".join(lines)
    
    def format_diff_report(self, diff: Dict[str, Any]) -> str:
        """
        Format a diff report as a human-readable Markdown summary.
        """
        lines = []
        lines.append("# Curriculum Drift Report")
        lines.append("")
        lines.append(f"**Comparison**: {diff['old_timestamp']} → {diff['new_timestamp']}")
        lines.append(f"**Git commits**: `{diff['old_git_commit'][:8]}...` → `{diff['new_git_commit'][:8]}...`")
        lines.append("")
        
        # Show classification if present
        drift_type = diff.get('drift_type', 'UNKNOWN')
        risk_level = diff.get('risk_level', 'UNKNOWN')
        
        if not diff['has_drift']:
            lines.append("✅ **No drift detected** — curriculum unchanged.")
            lines.append("")
            lines.append(f"- **Drift Type**: `{drift_type}`")
            lines.append(f"- **Risk Level**: `{risk_level}`")
        else:
            # Format based on risk level
            if risk_level == RiskLevel.BLOCK.value:
                lines.append("🚫 **BLOCKING DRIFT DETECTED** — curriculum has critical changes!")
            elif risk_level == RiskLevel.WARN.value:
                lines.append("⚠️ **Drift detected** — curriculum has minor changes.")
            else:
                lines.append("ℹ️ **Drift detected** — curriculum has cosmetic changes.")
            
            lines.append("")
            lines.append(f"- **Drift Type**: `{drift_type}`")
            lines.append(f"- **Risk Level**: `{risk_level}`")
            lines.append("")
            
            if diff['global_hash_changed']:
                lines.append(f"Global hash: `{diff['old_curriculum_hash'][:16]}...` → `{diff['new_curriculum_hash'][:16]}...`")
                lines.append("")
            
            # Show affected slices with details
            affected_slices = diff.get('affected_slices', {})
            
            if diff['added_slices']:
                lines.append("## Added Slices (STRUCTURAL)")
                for name in diff['added_slices']:
                    lines.append(f"- `{name}`")
                lines.append("")
            
            if diff['removed_slices']:
                lines.append("## Removed Slices (STRUCTURAL)")
                for name in diff['removed_slices']:
                    lines.append(f"- `{name}`")
                lines.append("")
            
            if diff['changed_slices']:
                lines.append("## Changed Slices")
                for name in diff['changed_slices']:
                    slice_info = affected_slices.get(name, {})
                    slice_drift = slice_info.get('drift_type', 'UNKNOWN')
                    changed_keys = slice_info.get('changed_keys', [])
                    
                    lines.append(f"### `{name}` ({slice_drift})")
                    if changed_keys:
                        for key in changed_keys[:10]:  # Limit to 10 keys
                            lines.append(f"- `{key}`")
                        if len(changed_keys) > 10:
                            lines.append(f"- ... and {len(changed_keys) - 10} more")
                    lines.append("")
        
        return '\n'.join(lines)
    
    def export_slice_chronicle(
        self,
        slice_name: str,
        warnings: Optional[List[str]] = None
    ) -> Chronicle:
        """
        Export a complete chronicle for a specific slice's drift history.
        
        Args:
            slice_name: Name of the slice to export.
            warnings: Optional list of warning messages to include.
        
        Returns:
            Chronicle object with complete drift history.
        """
        events = self.build_drift_timeline(slice_name)
        return export_chronicle(events, slice_name, warnings)
    
    def export_all_chronicles(
        self,
        output_dir: Optional[Path] = None
    ) -> Dict[str, Chronicle]:
        """
        Export chronicles for all slices in the ledger.
        
        Args:
            output_dir: Optional directory to write JSON files.
        
        Returns:
            Dict mapping slice_name -> Chronicle.
        """
        all_timelines = self.build_all_slices_timeline()
        chronicles = {}
        
        for slice_name, events in all_timelines.items():
            chronicle = export_chronicle(events, slice_name)
            chronicles[slice_name] = chronicle
            
            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"{slice_name}_chronicle.json"
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(chronicle.to_json())
        
        return chronicles
    
    def build_curriculum_index(self) -> Dict[str, Any]:
        """
        Build a curriculum-wide chronicle index over all slices.
        
        Returns:
            Chronicle index dict with aggregated metrics.
        """
        chronicles = list(self.export_all_chronicles().values())
        return build_chronicle_index(chronicles)
    
    def get_governance_summary(self) -> Dict[str, Any]:
        """
        Get a governance/audit lens summary for the curriculum.
        
        Returns:
            Governance summary dict.
        """
        index = self.build_curriculum_index()
        return summarize_chronicles_for_governance(index)
    
    def get_health_summary(self) -> Dict[str, Any]:
        """
        Get a global health summary for the curriculum.
        
        Returns:
            Health summary dict.
        """
        index = self.build_curriculum_index()
        return summarize_chronicles_for_global_health(index)
    
    def get_alignment_view(
        self,
        curriculum_timeline: Optional[Dict[str, Any]] = None,
        drift_classifications: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get a chronicle alignment view for the curriculum.
        
        Args:
            curriculum_timeline: Optional dict with slice timeline data.
            drift_classifications: Optional dict with slice drift classifications.
        
        Returns:
            Alignment view dict.
        """
        index = self.build_curriculum_index()
        return build_chronicle_alignment_view(index, curriculum_timeline, drift_classifications)
    
    def get_governance_narrative(self) -> str:
        """
        Get a governance narrative in Markdown format.
        
        Returns:
            Markdown-formatted governance narrative.
        """
        index = self.build_curriculum_index()
        alignment_view = self.get_alignment_view()
        return render_chronicle_governance_narrative(index, alignment_view)
    
    def get_acquisition_summary(self) -> Dict[str, Any]:
        """
        Get an acquisition-facing chronicle summary.
        
        Returns:
            Acquisition summary dict.
        """
        index = self.build_curriculum_index()
        alignment_view = self.get_alignment_view()
        return build_chronicle_summary_for_acquisition(index, alignment_view)
    
    def get_causality_map(
        self,
        drift_events: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get a causality map for the curriculum chronicles.
        
        Args:
            drift_events: Optional dict of drift events. If not provided,
                will extract from chronicles.
        
        Returns:
            Causality map dict.
        """
        index = self.build_curriculum_index()
        
        # If drift_events not provided, extract from chronicles
        if drift_events is None:
            drift_events = {}
            all_timelines = self.build_all_slices_timeline()
            event_counter = 0
            
            for slice_name, events in all_timelines.items():
                for event in events:
                    event_id = f"event_{event_counter}"
                    drift_events[event_id] = {
                        "timestamp": event.timestamp,
                        "drift_type": event.drift_type.value,
                        "slice_name": slice_name,
                        "snapshot_index": event.snapshot_index,
                        "risk_level": event.risk_level.value,
                    }
                    event_counter += 1
        
        return build_chronicle_causality_map(index, drift_events)
    
    def get_multi_axis_stability(self) -> Dict[str, Any]:
        """
        Get multi-axis stability estimate for the curriculum.
        
        Returns:
            Multi-axis stability dict.
        """
        alignment_view = self.get_alignment_view()
        causality_map = self.get_causality_map()
        return estimate_multi_axis_chronicle_stability(alignment_view, causality_map)
    
    def get_recurrence_projection(
        self,
        drift_events: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get recurrence projection for the curriculum.
        
        Args:
            drift_events: Optional dict of drift events. If not provided,
                will extract from chronicles.
        
        Returns:
            Recurrence projection dict.
        """
        causality_map = self.get_causality_map(drift_events)
        stability_estimate = self.get_multi_axis_stability()
        
        # Extract drift events if not provided
        if drift_events is None:
            drift_events = {}
            all_timelines = self.build_all_slices_timeline()
            event_counter = 0
            
            for slice_name, events in all_timelines.items():
                for event in events:
                    event_id = f"event_{event_counter}"
                    drift_events[event_id] = {
                        "timestamp": event.timestamp,
                        "drift_type": event.drift_type.value,
                        "slice_name": slice_name,
                        "snapshot_index": event.snapshot_index,
                        "risk_level": event.risk_level.value,
                    }
                    event_counter += 1
        
        return build_recurrence_projection_engine(
            causality_map, drift_events, stability_estimate
        )
    
    def get_invariant_check(
        self,
        drift_events: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get phase-transition drift invariant check for the curriculum.
        
        Args:
            drift_events: Optional dict of drift events. If not provided,
                will extract from chronicles.
        
        Returns:
            Invariant check dict.
        """
        causality_map = self.get_causality_map(drift_events)
        
        # Extract drift events if not provided
        if drift_events is None:
            drift_events = {}
            all_timelines = self.build_all_slices_timeline()
            event_counter = 0
            
            for slice_name, events in all_timelines.items():
                for event in events:
                    event_id = f"event_{event_counter}"
                    drift_events[event_id] = {
                        "timestamp": event.timestamp,
                        "drift_type": event.drift_type.value,
                        "slice_name": slice_name,
                        "snapshot_index": event.snapshot_index,
                        "risk_level": event.risk_level.value,
                    }
                    event_counter += 1
        
        return build_phase_transition_drift_invariant_checker(drift_events, causality_map)
    
    def get_director_tile(self) -> Dict[str, Any]:
        """
        Get director tile for executive governance dashboard.
        
        Returns:
            Director tile dict.
        """
        recurrence_projection = self.get_recurrence_projection()
        invariant_check = self.get_invariant_check()
        stability_estimate = self.get_multi_axis_stability()
        
        return build_director_tile(recurrence_projection, invariant_check, stability_estimate)


def main():
    parser = argparse.ArgumentParser(
        description="Curriculum Hash Ledger — Track curriculum drift over time."
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--snapshot",
        action="store_true",
        help="Record a new snapshot of the curriculum."
    )
    mode_group.add_argument(
        "--diff",
        action="store_true",
        help="Compare two snapshots for drift."
    )
    mode_group.add_argument(
        "--list",
        action="store_true",
        help="List all recorded snapshots."
    )
    mode_group.add_argument(
        "--hash",
        action="store_true",
        help="Compute and print the curriculum hash (no recording)."
    )
    
    # Snapshot options
    parser.add_argument(
        "--config",
        type=str,
        default="config/curriculum_uplift_phase2.yaml",
        help="Path to the curriculum config file."
    )
    parser.add_argument(
        "--origin",
        type=str,
        choices=["manual", "ci", "pre-commit"],
        default="manual",
        help="Source of the snapshot."
    )
    parser.add_argument(
        "--notes",
        type=str,
        default="",
        help="Free-text notes for the snapshot."
    )
    
    # Diff options
    parser.add_argument(
        "--from",
        dest="from_ref",
        type=str,
        default="0",
        help="Older snapshot (index or timestamp). Default: 0 (first)."
    )
    parser.add_argument(
        "--to",
        dest="to_ref",
        type=str,
        default="-1",
        help="Newer snapshot (index or timestamp). Default: -1 (last)."
    )
    parser.add_argument(
        "--fail-on-drift",
        action="store_true",
        help="Exit with non-zero code if drift is detected."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output diff as JSON instead of Markdown."
    )
    
    # Ledger path override
    parser.add_argument(
        "--ledger",
        type=str,
        default=None,
        help="Path to the ledger JSONL file (overrides default)."
    )
    
    args = parser.parse_args()
    
    ledger_path = Path(args.ledger) if args.ledger else None
    ledger = CurriculumHashLedger(ledger_path=ledger_path)
    
    if args.snapshot:
        # Record a new snapshot
        if not Path(args.config).exists():
            print(f"Error: Config file not found: {args.config}", file=sys.stderr)
            sys.exit(1)
        
        entry = ledger.record_snapshot(
            config_path=args.config,
            origin=args.origin,
            notes=args.notes
        )
        print(f"Snapshot recorded:")
        print(f"  Timestamp:  {entry['timestamp']}")
        print(f"  Config:     {entry['config_path']}")
        print(f"  Hash:       {entry['curriculum_hash']}")
        print(f"  Git commit: {entry['git_commit']}")
        print(f"  Origin:     {entry['origin']}")
        print(f"  Slices:     {len(entry['slice_hashes'])}")
        
    elif args.diff:
        # Compare snapshots
        snapshots = ledger.load_snapshots()
        if len(snapshots) < 2:
            print("Error: Need at least 2 snapshots to compare.", file=sys.stderr)
            sys.exit(1)
        
        old_snap = ledger.get_snapshot(args.from_ref)
        new_snap = ledger.get_snapshot(args.to_ref)
        
        if old_snap is None:
            print(f"Error: Snapshot not found: {args.from_ref}", file=sys.stderr)
            sys.exit(1)
        if new_snap is None:
            print(f"Error: Snapshot not found: {args.to_ref}", file=sys.stderr)
            sys.exit(1)
        
        diff = ledger.compare_snapshots(old_snap, new_snap)
        
        if args.json:
            print(json.dumps(diff, indent=2))
        else:
            print(ledger.format_diff_report(diff))
        
        if args.fail_on_drift and diff['has_drift']:
            sys.exit(1)
    
    elif args.list:
        # List all snapshots
        snapshots = ledger.load_snapshots()
        if not snapshots:
            print("No snapshots recorded yet.")
        else:
            print(f"Recorded snapshots ({len(snapshots)} total):")
            for i, snap in enumerate(snapshots):
                print(f"  [{i}] {snap['timestamp']} — {snap['curriculum_hash'][:16]}... ({snap['origin']})")
    
    elif args.hash:
        # Just compute and print hash (no recording)
        if not Path(args.config).exists():
            print(f"Error: Config file not found: {args.config}", file=sys.stderr)
            sys.exit(1)
        
        global_hash, slice_hashes = ledger.compute_curriculum_hash(args.config)
        print(f"Config: {args.config}")
        print(f"Global hash: {global_hash}")
        print(f"Slice hashes ({len(slice_hashes)}):")
        for name, h in sorted(slice_hashes.items()):
            print(f"  {name}: {h}")


if __name__ == "__main__":
    main()

