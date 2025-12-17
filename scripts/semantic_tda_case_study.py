"""Case study harness for semantic-TDA integration.

STATUS: PHASE V — SEMANTIC/TDA CROSS-TIE

Creates a synthetic scenario demonstrating semantic-TDA correlation and
governance tile generation. This serves as a "First Light" example.

Usage:
    python scripts/semantic_tda_case_study.py
"""

import json
from pathlib import Path
from typing import Any, Dict

from backend.health.semantic_tda_adapter import (
    attach_semantic_tda_to_evidence,
    build_semantic_tda_tile_for_global_health,
)
from experiments.semantic_consistency_audit import (
    correlate_semantic_and_tda_signals,
)


def build_synthetic_drift_scenario() -> Dict[str, Any]:
    """
    Build a synthetic scenario where semantic and TDA both signal drift.
    
    Scenario: Uplift slice shows semantic drift (terms disappearing) and
    TDA shows degraded HSS and elevated block rate on the same slice.
    
    Returns:
        Dictionary with semantic_timeline, tda_health, semantic_panel, tda_panel
    """
    # Semantic timeline: DRIFTING with critical signals
    semantic_timeline = {
        "timeline": [
            {"run_id": "run_uplift_001", "status": "CRITICAL"},
            {"run_id": "run_uplift_002", "status": "CRITICAL"},
        ],
        "runs_with_critical_signals": ["run_uplift_001", "run_uplift_002"],
        "node_disappearance_events": [
            {"run_id": "run_uplift_001", "term": "slice_uplift_goal"},
            {"run_id": "run_uplift_001", "term": "uplift_success_metric"},
            {"run_id": "run_uplift_002", "term": "curriculum_alignment_term"},
        ],
        "trend": "DRIFTING",
        "semantic_status_light": "RED",
    }

    # TDA health: ALERT with degraded metrics
    tda_health = {
        "tda_status": "ALERT",
        "block_rate": 0.28,  # Exceeds 20% threshold
        "mean_hss": 0.45,  # Degraded from baseline
        "hss_trend": "DEGRADING",
        "governance_signal": "BLOCK",
        "notes": [
            "block_rate=28.00% exceeds 20% threshold",
            "hss_trend classified as DEGRADING over 100 cycles",
            "Topological structure shows significant deviation from baseline",
        ],
    }

    # Semantic panel: Director-level summary
    semantic_panel = {
        "semantic_status_light": "RED",
        "alignment_status": "MISALIGNED",
        "critical_run_ids": ["run_uplift_001", "run_uplift_002"],
        "headline": (
            "Semantic graph shows critical drift with significant curriculum "
            "misalignment (2 runs with critical signals, 3 terms disappeared)"
        ),
        "trend": "DRIFTING",
        "node_disappearance_count": 3,
    }

    # TDA panel: Same as tda_health (for clarity in integration)
    tda_panel = tda_health.copy()

    return {
        "semantic_timeline": semantic_timeline,
        "tda_health": tda_health,
        "semantic_panel": semantic_panel,
        "tda_panel": tda_panel,
    }


def run_case_study() -> Dict[str, Any]:
    """
    Run the case study: correlate signals and build governance tile.
    
    Returns:
        Complete case study results including correlation and tile
    """
    scenario = build_synthetic_drift_scenario()

    # Step 1: Correlate semantic and TDA signals
    correlation = correlate_semantic_and_tda_signals(
        scenario["semantic_timeline"],
        scenario["tda_health"],
    )

    # Step 2: Build governance tile using adapter (includes tile_type and notes)
    tile = build_semantic_tda_tile_for_global_health(
        scenario["semantic_panel"],
        scenario["tda_panel"],
        scenario["semantic_timeline"],
    )

    # Assemble results
    results = {
        "scenario": scenario,
        "correlation": correlation,
        "governance_tile": tile,
        "summary": {
            "correlation_coefficient": correlation["correlation_coefficient"],
            "tile_status": tile["status"],
            "tile_status_light": tile["status_light"],
            "key_slices": tile["key_slices"],
            "headline": tile["headline"],
        },
    }

    return results


def build_evidence_bundle(correlation: Dict[str, Any], tile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build evidence bundle format matching Phase X evidence pack structure.
    
    Returns:
        Evidence bundle with governance.semantic_tda attached
    """
    # Create minimal evidence pack structure
    evidence = {
        "timestamp": "2024-01-01T00:00:00Z",
        "run_id": "case_study_synthetic",
        "evidence_type": "semantic_tda_case_study",
        "data": {
            "correlation_record": correlation,
        },
    }
    
    # Attach semantic-TDA tile to evidence pack
    bundle = attach_semantic_tda_to_evidence(evidence, tile)
    
    return bundle


def main() -> None:
    """Main entry point: run case study and save results."""
    print("Semantic-TDA Case Study: First Light")
    print("=" * 60)

    # Run case study
    results = run_case_study()

    # Print summary
    print("\nCorrelation Results:")
    print(f"  Coefficient: {results['correlation']['correlation_coefficient']:.3f}")
    print(f"  Alignment: {results['correlation']['alignment_note']}")
    print(f"  Slices where both signal: {results['correlation']['slices_where_both_signal']}")

    print("\nGovernance Tile:")
    print(f"  Status: {results['governance_tile']['status']}")
    print(f"  Status Light: {results['governance_tile']['status_light']}")
    print(f"  Headline: {results['governance_tile']['headline']}")
    print(f"  Key Slices: {results['governance_tile']['key_slices']}")

    # Save detailed results
    artifacts_dir = Path("artifacts") / "semantic_tda"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed case study
    case_study_path = artifacts_dir / "case_study.json"
    with open(case_study_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: {case_study_path}")

    # Build and save evidence bundle format
    bundle = build_evidence_bundle(results["correlation"], results["governance_tile"])
    bundle_path = artifacts_dir / "semantic_tda_case_study_bundle.json"
    with open(bundle_path, "w") as f:
        json.dump(bundle, f, indent=2)
    print(f"Evidence bundle saved to: {bundle_path}")

    # Verify bundle structure
    assert "governance" in bundle
    assert "semantic_tda" in bundle["governance"]
    assert bundle["governance"]["semantic_tda"]["tile_type"] == "semantic_tda"
    print("\nBundle structure verified: governance.semantic_tda present")

    print("\nCase study complete!")


if __name__ == "__main__":
    main()

