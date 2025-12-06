#!/usr/bin/env python3
"""
Example: Using the Docs Governance Layer

This script demonstrates how to use the docs governance layer
to validate documentation and integrate with Evidence Packs.
"""

import json
import sys
from pathlib import Path

# Add repo root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from docs.docs_governance import (
    build_docs_governance_snapshot,
    evaluate_uplift_safety,
    build_docs_section_for_evidence_pack
)


def example_1_clean_docs():
    """Example 1: Clean documentation with no issues."""
    print("=" * 70)
    print("EXAMPLE 1: Clean Documentation")
    print("=" * 70)
    
    # Simulate clean reports
    snippet_report = {"invalid_files": []}
    phase_marker_report = {
        "docs_missing_markers": [],
        "docs_with_uplift_mentions_without_disclaimer": []
    }
    toc_index = {"evidence_docs": ["docs/README.md"]}
    
    # Build snapshot
    snapshot = build_docs_governance_snapshot(
        snippet_report, phase_marker_report, toc_index
    )
    
    # Evaluate safety
    safety = evaluate_uplift_safety(snapshot)
    
    # Build evidence section
    evidence = build_docs_section_for_evidence_pack(snapshot, safety)
    
    print(f"\nStatus: {safety['status']}")
    print(f"Uplift Safe: {safety['uplift_safe']}")
    print(f"Issues: {len(safety['issues'])}")
    print(f"\nEvidence Pack Section:")
    print(json.dumps(evidence, indent=2))
    print()


def example_2_warning_issues():
    """Example 2: Documentation with warning-level issues."""
    print("=" * 70)
    print("EXAMPLE 2: Documentation with Warnings")
    print("=" * 70)
    
    # Simulate reports with warnings
    snippet_report = {
        "invalid_files": ["docs/example1.md", "docs/example2.md"]
    }
    phase_marker_report = {
        "docs_missing_markers": ["docs/phase2_doc.md"],
        "docs_with_uplift_mentions_without_disclaimer": []
    }
    toc_index = {"evidence_docs": ["docs/README.md", "docs/example1.md"]}
    
    # Build snapshot
    snapshot = build_docs_governance_snapshot(
        snippet_report, phase_marker_report, toc_index
    )
    
    # Evaluate safety
    safety = evaluate_uplift_safety(snapshot)
    
    # Build evidence section
    evidence = build_docs_section_for_evidence_pack(snapshot, safety)
    
    print(f"\nStatus: {safety['status']}")
    print(f"Uplift Safe: {safety['uplift_safe']}")
    print(f"Issues Found: {len(safety['issues'])}")
    for issue in safety['issues']:
        print(f"  - {issue}")
    print(f"\nSnapshot Details:")
    print(f"  Total Docs: {snapshot['doc_count']}")
    print(f"  Invalid Snippets: {len(snapshot['docs_with_invalid_snippets'])}")
    print(f"  Missing Markers: {len(snapshot['docs_missing_phase_markers'])}")
    print()


def example_3_blocking_issues():
    """Example 3: Documentation with blocking issues."""
    print("=" * 70)
    print("EXAMPLE 3: Documentation with BLOCKING Issues")
    print("=" * 70)
    
    # Simulate reports with blocking issues
    snippet_report = {"invalid_files": []}
    phase_marker_report = {
        "docs_missing_markers": [],
        "docs_with_uplift_mentions_without_disclaimer": [
            "docs/U2_PORT_PLAN.md",
            "docs/PHASE2_CLAIMS.md"
        ]
    }
    toc_index = {"evidence_docs": ["docs/README.md"]}
    
    # Build snapshot
    snapshot = build_docs_governance_snapshot(
        snippet_report, phase_marker_report, toc_index
    )
    
    # Evaluate safety
    safety = evaluate_uplift_safety(snapshot)
    
    # Build evidence section
    evidence = build_docs_section_for_evidence_pack(snapshot, safety)
    
    print(f"\n⚠️  Status: {safety['status']}")
    print(f"⚠️  Uplift Safe: {safety['uplift_safe']}")
    print(f"\n⚠️  CRITICAL ISSUES:")
    for issue in safety['issues']:
        print(f"  ❌ {issue}")
    print(f"\nThese issues MUST be resolved before publication!")
    print()


def example_4_evidence_pack_integration():
    """Example 4: Integrating with Evidence Pack v2."""
    print("=" * 70)
    print("EXAMPLE 4: Evidence Pack v2 Integration")
    print("=" * 70)
    
    # Simulate mixed reports
    snippet_report = {"invalid_files": ["docs/draft.md"]}
    phase_marker_report = {
        "docs_missing_markers": [],
        "docs_with_uplift_mentions_without_disclaimer": []
    }
    toc_index = {
        "evidence_docs": [
            "docs/RFL_PHASE_I_TRUTH_SOURCE.md",
            "docs/PHASE2_RFL_UPLIFT_PLAN.md",
            "docs/VSD_PHASE_2.md"
        ]
    }
    
    # Build full governance report
    snapshot = build_docs_governance_snapshot(
        snippet_report, phase_marker_report, toc_index
    )
    safety = evaluate_uplift_safety(snapshot)
    evidence_section = build_docs_section_for_evidence_pack(snapshot, safety)
    
    # Create evidence pack structure
    evidence_pack = {
        "version": "2.0",
        "timestamp": "2025-12-06T22:00:00Z",
        "experiments": {
            "phase_i_rfl": {
                "status": "complete",
                "gates_passed": ["G1", "G2", "G3", "G4", "G5"]
            }
        },
        "docs_governance": evidence_section,
        "metadata": {
            "total_docs_checked": snapshot["doc_count"],
            "evidence_docs_covered": snapshot["evidence_docs_covered"],
            "governance_status": safety["status"]
        }
    }
    
    print("\nEvidence Pack v2 Structure:")
    print(json.dumps(evidence_pack, indent=2))
    print(f"\nGovernance Status: {evidence_pack['metadata']['governance_status']}")
    print(f"Uplift Safe: {'✅ YES' if evidence_section['uplift_safe'] else '❌ NO'}")
    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("DOCS GOVERNANCE LAYER - EXAMPLES")
    print("=" * 70 + "\n")
    
    example_1_clean_docs()
    example_2_warning_issues()
    example_3_blocking_issues()
    example_4_evidence_pack_integration()
    
    print("=" * 70)
    print("All examples completed successfully!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
