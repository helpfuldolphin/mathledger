#!/usr/bin/env python3
"""
Consensus Polygraph v2 — Multi-Agent Consensus & Conflict Detection Engine

Fuses all consensus-adjacent systems (semantic, metric, drift, topology, curriculum)
into a unified conflict map. Detects emerging cross-system divergences before they manifest.

Author: Agent E4 (doc-ops-4) — Migration Council Orchestrator
Date: 2025-12-06

ABSOLUTE SAFEGUARDS:
- Read-only analysis — no mutations to production state
- Deterministic output — same panels → same conflict map
- Neutral language — no prescriptive terms
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

# Standard panel status values
PANEL_STATUS_OK = "OK"
PANEL_STATUS_ATTENTION = "ATTENTION"
PANEL_STATUS_BLOCK = "BLOCK"

# Consensus band thresholds
CONSENSUS_BAND_HIGH = 0.8  # 80%+ agreement
CONSENSUS_BAND_MEDIUM = 0.5  # 50-80% agreement
CONSENSUS_BAND_LOW = 0.0  # <50% agreement


@dataclass
class SystemConflict:
    """Represents a conflict between systems for a given slice/component."""
    slice_id: str
    component: str | None
    conflicting_systems: list[str]
    statuses: dict[str, str]  # system -> status
    severity: str  # "HIGH" | "MEDIUM" | "LOW"
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "slice_id": self.slice_id,
            "component": self.component,
            "conflicting_systems": self.conflicting_systems,
            "statuses": self.statuses,
            "severity": self.severity,
        }


@dataclass
class ConsensusPolygraphResult:
    """Result of consensus polygraph analysis."""
    system_conflicts: list[SystemConflict]
    agreement_rate: float
    consensus_band: str  # "HIGH" | "MEDIUM" | "LOW"
    neutral_notes: list[str]
    total_slices: int
    agreeing_slices: int
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "system_conflicts": [c.to_dict() for c in self.system_conflicts],
            "agreement_rate": self.agreement_rate,
            "consensus_band": self.consensus_band,
            "neutral_notes": self.neutral_notes,
            "total_slices": self.total_slices,
            "agreeing_slices": self.agreeing_slices,
        }


def extract_slice_statuses(
    panel: dict[str, Any],
    system_name: str,
) -> dict[str, str]:
    """
    Extract slice-level statuses from a panel.
    
    Panels may have different structures. This function handles common formats:
    - panels with "slices" array
    - panels with "components" dict
    - panels with slice_id -> status mapping
    
    Args:
        panel: The panel dictionary
        system_name: Name of the system (for error messages)
        
    Returns:
        Dictionary mapping slice_id -> status
    """
    statuses: dict[str, str] = {}
    
    # Format 1: "slices" array with slice_id and status
    if "slices" in panel:
        for slice_data in panel["slices"]:
            if isinstance(slice_data, dict):
                slice_id = slice_data.get("slice_id") or slice_data.get("id") or slice_data.get("name")
                status = slice_data.get("status") or slice_data.get("signal") or slice_data.get("verdict")
                if slice_id and status:
                    statuses[str(slice_id)] = str(status)
    
    # Format 2: "components" dict with slice_id keys
    elif "components" in panel:
        for slice_id, comp_data in panel["components"].items():
            if isinstance(comp_data, dict):
                status = comp_data.get("status") or comp_data.get("signal") or comp_data.get("verdict")
                if status:
                    statuses[str(slice_id)] = str(status)
            elif isinstance(comp_data, str):
                statuses[str(slice_id)] = comp_data
    
    # Format 3: Direct slice_id -> status mapping
    elif "slice_statuses" in panel:
        statuses = {str(k): str(v) for k, v in panel["slice_statuses"].items()}
    
    # Format 4: Try to find any dict-like structure with status fields
    else:
        # Look for common keys that might indicate slice data
        for key, value in panel.items():
            if isinstance(value, dict) and ("status" in value or "signal" in value or "verdict" in value):
                status = value.get("status") or value.get("signal") or value.get("verdict")
                if status:
                    statuses[str(key)] = str(status)
    
    return statuses


def normalize_status(status: str) -> str:
    """
    Normalize panel status to standard values (OK/ATTENTION/BLOCK).
    
    Args:
        status: Raw status from panel
        
    Returns:
        Normalized status: OK, ATTENTION, or BLOCK
    """
    status_upper = str(status).upper()
    
    # OK variants
    if status_upper in ("OK", "PASS", "GREEN", "SUCCESS", "VALID", "GOOD"):
        return PANEL_STATUS_OK
    
    # ATTENTION variants
    if status_upper in ("ATTENTION", "WARN", "WARNING", "YELLOW", "CAUTION", "REVIEW"):
        return PANEL_STATUS_ATTENTION
    
    # BLOCK variants
    if status_upper in ("BLOCK", "FAIL", "RED", "ERROR", "INVALID", "BAD", "CRITICAL"):
        return PANEL_STATUS_BLOCK
    
    # Default to ATTENTION for unknown statuses
    return PANEL_STATUS_ATTENTION


def build_consensus_polygraph(
    semantic_panel: dict[str, Any] | None = None,
    metric_panel: dict[str, Any] | None = None,
    topology_panel: dict[str, Any] | None = None,
    drift_panel: dict[str, Any] | None = None,
    curriculum_panel: dict[str, Any] | None = None,
) -> ConsensusPolygraphResult:
    """
    Build a consensus polygraph by analyzing multiple system panels.
    
    Detects conflicts where systems disagree on slice/component statuses.
    Calculates agreement rate and consensus band.
    
    Args:
        semantic_panel: Semantic consistency panel
        metric_panel: Metric conformance panel
        topology_panel: Topology analysis panel
        drift_panel: Governance drift panel
        curriculum_panel: Curriculum analysis panel
        
    Returns:
        ConsensusPolygraphResult with conflicts, agreement rate, and consensus band
    """
    # Collect all panels
    panels = {
        "semantic": semantic_panel,
        "metric": metric_panel,
        "topology": topology_panel,
        "drift": drift_panel,
        "curriculum": curriculum_panel,
    }
    
    # Filter out None panels
    active_panels = {name: panel for name, panel in panels.items() if panel is not None}
    
    if not active_panels:
        return ConsensusPolygraphResult(
            system_conflicts=[],
            agreement_rate=1.0,
            consensus_band="HIGH",
            neutral_notes=["No panels provided for consensus analysis"],
            total_slices=0,
            agreeing_slices=0,
        )
    
    # Extract slice statuses from each panel
    all_slice_statuses: dict[str, dict[str, str]] = {}
    # Format: {slice_id: {system_name: normalized_status}}
    
    for system_name, panel in active_panels.items():
        statuses = extract_slice_statuses(panel, system_name)
        for slice_id, status in statuses.items():
            if slice_id not in all_slice_statuses:
                all_slice_statuses[slice_id] = {}
            all_slice_statuses[slice_id][system_name] = normalize_status(status)
    
    # Find all unique slice IDs
    all_slice_ids = set(all_slice_statuses.keys())
    
    if not all_slice_ids:
        return ConsensusPolygraphResult(
            system_conflicts=[],
            agreement_rate=1.0,
            consensus_band="HIGH",
            neutral_notes=["No slices found in provided panels"],
            total_slices=0,
            agreeing_slices=0,
        )
    
    # Analyze each slice for conflicts
    conflicts: list[SystemConflict] = []
    agreeing_slices = 0
    
    for slice_id in all_slice_ids:
        slice_statuses = all_slice_statuses[slice_id]
        
        # Get unique normalized statuses for this slice
        unique_statuses = set(slice_statuses.values())
        
        # If all systems agree (only one unique status), count as agreeing
        if len(unique_statuses) == 1:
            agreeing_slices += 1
        else:
            # Conflict detected - systems disagree
            conflicting_systems = list(slice_statuses.keys())
            
            # Determine severity based on status combination
            if PANEL_STATUS_BLOCK in unique_statuses:
                severity = "HIGH"
            elif PANEL_STATUS_ATTENTION in unique_statuses:
                severity = "MEDIUM"
            else:
                severity = "LOW"
            
            conflict = SystemConflict(
                slice_id=slice_id,
                component=None,  # Could be extracted from panel if available
                conflicting_systems=conflicting_systems,
                statuses=slice_statuses.copy(),
                severity=severity,
            )
            conflicts.append(conflict)
    
    # Calculate agreement rate
    total_slices = len(all_slice_ids)
    agreement_rate = agreeing_slices / total_slices if total_slices > 0 else 1.0
    
    # Determine consensus band
    if agreement_rate >= CONSENSUS_BAND_HIGH:
        consensus_band = "HIGH"
    elif agreement_rate >= CONSENSUS_BAND_MEDIUM:
        consensus_band = "MEDIUM"
    else:
        consensus_band = "LOW"
    
    # Build neutral notes
    neutral_notes: list[str] = []
    neutral_notes.append(f"Analyzed {total_slices} slices across {len(active_panels)} systems")
    neutral_notes.append(f"Agreement rate: {agreement_rate:.1%}")
    if conflicts:
        neutral_notes.append(f"{len(conflicts)} conflicts detected")
        high_conflicts = sum(1 for c in conflicts if c.severity == "HIGH")
        if high_conflicts > 0:
            neutral_notes.append(f"{high_conflicts} high-severity conflicts")
    else:
        neutral_notes.append("No conflicts detected")
    
    return ConsensusPolygraphResult(
        system_conflicts=conflicts,
        agreement_rate=agreement_rate,
        consensus_band=consensus_band,
        neutral_notes=neutral_notes,
        total_slices=total_slices,
        agreeing_slices=agreeing_slices,
    )


def detect_predictive_conflicts(
    current_polygraph: ConsensusPolygraphResult,
    historical_polygraph: ConsensusPolygraphResult | None = None,
) -> dict[str, Any]:
    """
    Detect emerging cross-system divergences before they manifest.
    
    Analyzes trends in conflict patterns to predict future conflicts.
    
    Args:
        current_polygraph: Current consensus polygraph result
        historical_polygraph: Previous polygraph result for trend analysis
        
    Returns:
        Dictionary with predictive conflict information
    """
    predictions: list[dict[str, Any]] = []
    
    # Analyze current conflicts for predictive signals
    for conflict in current_polygraph.system_conflicts:
        # High-severity conflicts are more likely to escalate
        if conflict.severity == "HIGH":
            predictions.append({
                "slice_id": conflict.slice_id,
                "risk": "HIGH",
                "reason": f"Multiple systems show BLOCK status for {conflict.slice_id}",
                "systems_involved": conflict.conflicting_systems,
            })
        # Medium-severity conflicts with many systems involved
        elif conflict.severity == "MEDIUM" and len(conflict.conflicting_systems) >= 3:
            predictions.append({
                "slice_id": conflict.slice_id,
                "risk": "MEDIUM",
                "reason": f"Multiple systems show ATTENTION status for {conflict.slice_id}",
                "systems_involved": conflict.conflicting_systems,
            })
    
    # Trend analysis if historical data available
    if historical_polygraph:
        # Detect increasing conflict rate
        if current_polygraph.agreement_rate < historical_polygraph.agreement_rate:
            rate_delta = historical_polygraph.agreement_rate - current_polygraph.agreement_rate
            if rate_delta > 0.1:  # 10% drop
                predictions.append({
                    "slice_id": "GLOBAL",
                    "risk": "MEDIUM",
                    "reason": f"Agreement rate decreased by {rate_delta:.1%}",
                    "systems_involved": [],
                })
        
        # Detect new conflicts (conflicts that weren't present before)
        current_conflict_slices = {c.slice_id for c in current_polygraph.system_conflicts}
        historical_conflict_slices = {c.slice_id for c in historical_polygraph.system_conflicts}
        new_conflicts = current_conflict_slices - historical_conflict_slices
        
        if new_conflicts:
            predictions.append({
                "slice_id": "MULTIPLE",
                "risk": "LOW",
                "reason": f"{len(new_conflicts)} new conflicts emerged",
                "systems_involved": [],
            })
    
    # Consensus band degradation
    if current_polygraph.consensus_band == "LOW":
        predictions.append({
            "slice_id": "GLOBAL",
            "risk": "HIGH",
            "reason": "Consensus band is LOW - systems show significant disagreement",
            "systems_involved": [],
        })
    
    return {
        "predictive_conflicts": predictions,
        "total_predictions": len(predictions),
        "high_risk_predictions": sum(1 for p in predictions if p["risk"] == "HIGH"),
    }


def build_consensus_director_panel(
    polygraph_result: ConsensusPolygraphResult,
    predictive_conflicts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build a consensus director panel for D1 autopilot + D4 telemetry integration.
    
    Provides a high-level view of consensus state across all systems.
    
    Args:
        polygraph_result: The consensus polygraph result
        predictive_conflicts: Optional predictive conflict analysis
        
    Returns:
        Dictionary with director panel data
    """
    # Determine overall status light
    if polygraph_result.consensus_band == "HIGH":
        status_light = "GREEN"
    elif polygraph_result.consensus_band == "MEDIUM":
        status_light = "YELLOW"
    else:
        status_light = "RED"
    
    # Build headline
    headline_parts: list[str] = []
    
    if polygraph_result.agreement_rate >= 0.9:
        headline_parts.append("High consensus across systems")
    elif polygraph_result.agreement_rate >= 0.7:
        headline_parts.append("Moderate consensus with some divergence")
    else:
        headline_parts.append("Low consensus - significant system disagreement")
    
    if polygraph_result.system_conflicts:
        headline_parts.append(f"{len(polygraph_result.system_conflicts)} conflicts detected")
    
    if predictive_conflicts and predictive_conflicts.get("high_risk_predictions", 0) > 0:
        headline_parts.append(f"{predictive_conflicts['high_risk_predictions']} high-risk predictions")
    
    headline = ". ".join(headline_parts) + "."
    
    # Build summary
    summary = {
        "status_light": status_light,
        "consensus_band": polygraph_result.consensus_band,
        "agreement_rate": polygraph_result.agreement_rate,
        "headline": headline,
        "total_slices": polygraph_result.total_slices,
        "conflicts": len(polygraph_result.system_conflicts),
        "predictive_risks": predictive_conflicts.get("total_predictions", 0) if predictive_conflicts else 0,
    }
    
    return summary


def main():
    """Main entry point for consensus polygraph CLI."""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(
        description="Consensus Polygraph v2 — Multi-Agent Consensus & Conflict Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This tool analyzes multiple system panels (semantic, metric, topology, drift, curriculum)
to detect conflicts and calculate consensus metrics.

Example:
  python tools/consensus_polygraph.py \\
    --semantic semantic_panel.json \\
    --metric metric_panel.json \\
    --topology topology_panel.json

Output:
  - system_conflicts: List of conflicts where systems disagree
  - agreement_rate: Proportion of slices where all systems agree
  - consensus_band: HIGH (>=80%), MEDIUM (50-80%), LOW (<50%)
  - neutral_notes: Factual summary of analysis
        """
    )
    parser.add_argument(
        "--semantic",
        type=str,
        help="Semantic panel JSON file path",
    )
    parser.add_argument(
        "--metric",
        type=str,
        help="Metric panel JSON file path",
    )
    parser.add_argument(
        "--topology",
        type=str,
        help="Topology panel JSON file path",
    )
    parser.add_argument(
        "--drift",
        type=str,
        help="Governance drift panel JSON file path",
    )
    parser.add_argument(
        "--curriculum",
        type=str,
        help="Curriculum panel JSON file path",
    )
    parser.add_argument(
        "--historical",
        type=str,
        help="Historical polygraph JSON file path for trend analysis",
    )
    parser.add_argument(
        "--predictive",
        action="store_true",
        help="Include predictive conflict detection",
    )
    parser.add_argument(
        "--director-panel",
        action="store_true",
        help="Output director panel format",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON format",
    )
    
    args = parser.parse_args()
    
    # Load panels
    def load_panel(file_path):
        if file_path is None:
            return None
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Error: Panel file not found: {file_path}", file=sys.stderr)
            return None
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in {file_path}: {e}", file=sys.stderr)
            return None
        except Exception as e:
            print(f"Error loading panel from {file_path}: {e}", file=sys.stderr)
            return None
    
    semantic_panel = load_panel(args.semantic)
    metric_panel = load_panel(args.metric)
    topology_panel = load_panel(args.topology)
    drift_panel = load_panel(args.drift)
    curriculum_panel = load_panel(args.curriculum)
    
    # Build polygraph
    polygraph = build_consensus_polygraph(
        semantic_panel=semantic_panel,
        metric_panel=metric_panel,
        topology_panel=topology_panel,
        drift_panel=drift_panel,
        curriculum_panel=curriculum_panel,
    )
    
    # Load historical if provided
    historical_polygraph = None
    if args.historical:
        try:
            with open(args.historical, "r", encoding="utf-8") as f:
                historical_data = json.load(f)
            # Reconstruct from dict (simplified - in practice would need full reconstruction)
            historical_polygraph = ConsensusPolygraphResult(
                system_conflicts=[
                    SystemConflict(**c) for c in historical_data.get("system_conflicts", [])
                ],
                agreement_rate=historical_data.get("agreement_rate", 1.0),
                consensus_band=historical_data.get("consensus_band", "HIGH"),
                neutral_notes=historical_data.get("neutral_notes", []),
                total_slices=historical_data.get("total_slices", 0),
                agreeing_slices=historical_data.get("agreeing_slices", 0),
            )
        except Exception as e:
            print(f"Error loading historical polygraph: {e}", file=sys.stderr)
    
    # Predictive conflicts
    predictive_conflicts = None
    if args.predictive:
        predictive_conflicts = detect_predictive_conflicts(
            polygraph,
            historical_polygraph,
        )
    
    # Director panel
    if args.director_panel:
        panel = build_consensus_director_panel(polygraph, predictive_conflicts)
        if args.json:
            print(json.dumps(panel, indent=2))
        else:
            print(f"Status Light: {panel['status_light']}")
            print(f"Consensus Band: {panel['consensus_band']}")
            print(f"Agreement Rate: {panel['agreement_rate']:.1%}")
            print(f"Headline: {panel['headline']}")
            print(f"Conflicts: {panel['conflicts']}")
            if panel.get("predictive_risks", 0) > 0:
                print(f"Predictive Risks: {panel['predictive_risks']}")
        return
    
    # Standard output
    if args.json:
        output = polygraph.to_dict()
        if predictive_conflicts:
            output["predictive_conflicts"] = predictive_conflicts
        print(json.dumps(output, indent=2))
    else:
        print("=" * 60)
        print("    CONSENSUS POLYGRAPH v2")
        print("=" * 60)
        print()
        print(f"Agreement Rate: {polygraph.agreement_rate:.1%}")
        print(f"Consensus Band: {polygraph.consensus_band}")
        print(f"Total Slices: {polygraph.total_slices}")
        print(f"Agreeing Slices: {polygraph.agreeing_slices}")
        print(f"Conflicts: {len(polygraph.system_conflicts)}")
        print()
        
        if polygraph.system_conflicts:
            print("System Conflicts:")
            for conflict in polygraph.system_conflicts:
                print(f"  - {conflict.slice_id}: {conflict.severity}")
                print(f"    Systems: {', '.join(conflict.conflicting_systems)}")
                print(f"    Statuses: {conflict.statuses}")
        print()
        
        if predictive_conflicts:
            print("Predictive Conflicts:")
            for pred in predictive_conflicts.get("predictive_conflicts", []):
                print(f"  - {pred['slice_id']}: {pred['risk']} risk")
                print(f"    {pred['reason']}")
        print()
        
        print("Notes:")
        for note in polygraph.neutral_notes:
            print(f"  - {note}")


if __name__ == "__main__":
    main()

