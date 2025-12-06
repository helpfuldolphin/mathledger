#!/usr/bin/env python3
"""
Docs Governance Layer for MathLedger

Provides risk signals, evidence pack integration, and uplift-safe narrative management.

Core functions:
- build_docs_governance_snapshot: Aggregate lint/validation reports into governance snapshot
- evaluate_uplift_safety: Pure function to assess documentation uplift safety
- build_docs_section_for_evidence_pack: Format governance data for Evidence Pack v2

All outputs are deterministic and JSON-safe.
"""

from pathlib import Path
from typing import Dict, Any, List, Set
import json


def build_docs_governance_snapshot(
    snippet_report: Dict[str, Any],
    phase_marker_report: Dict[str, Any],
    toc_index: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Build a governance snapshot from documentation validation reports.
    
    Args:
        snippet_report: Output from snippet_check.py with code snippet validation
        phase_marker_report: Output from phase_marker_lint.py with phase marker validation
        toc_index: Output from generate_evidence_pack_toc.py with TOC metadata
    
    Returns:
        Dict with deterministic governance metrics:
        - schema_version: str
        - doc_count: int
        - docs_with_invalid_snippets: List[str]
        - docs_missing_phase_markers: List[str]
        - docs_with_uplift_mentions_without_disclaimer: List[str]
        - evidence_docs_covered: int (docs in evidence_pack_config.yaml that exist)
    """
    snapshot = {
        "schema_version": "1.0.0",
        "doc_count": 0,
        "docs_with_invalid_snippets": [],
        "docs_missing_phase_markers": [],
        "docs_with_uplift_mentions_without_disclaimer": [],
        "evidence_docs_covered": 0
    }
    
    # Process snippet report
    if snippet_report:
        invalid_snippets = snippet_report.get("invalid_files", [])
        snapshot["docs_with_invalid_snippets"] = sorted(invalid_snippets)
    
    # Process phase marker report
    if phase_marker_report:
        missing_markers = phase_marker_report.get("docs_missing_markers", [])
        snapshot["docs_missing_phase_markers"] = sorted(missing_markers)
        
        uplift_without_disclaimer = phase_marker_report.get(
            "docs_with_uplift_mentions_without_disclaimer", []
        )
        snapshot["docs_with_uplift_mentions_without_disclaimer"] = sorted(
            uplift_without_disclaimer
        )
    
    # Process TOC index to count evidence docs
    if toc_index:
        evidence_docs = toc_index.get("evidence_docs", [])
        # Count how many evidence docs actually exist on disk
        existing_docs = 0
        for doc_path in evidence_docs:
            if Path(doc_path).exists():
                existing_docs += 1
        snapshot["evidence_docs_covered"] = existing_docs
        
        # Total doc count from all sources
        all_docs = set()
        all_docs.update(snapshot["docs_with_invalid_snippets"])
        all_docs.update(snapshot["docs_missing_phase_markers"])
        all_docs.update(snapshot["docs_with_uplift_mentions_without_disclaimer"])
        all_docs.update(evidence_docs)
        snapshot["doc_count"] = len(all_docs)
    
    return snapshot


def evaluate_uplift_safety(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate documentation uplift safety based on governance snapshot.
    
    This is a pure function that only interprets documentation structure
    and disclaimers, NOT experimental results.
    
    Args:
        snapshot: Output from build_docs_governance_snapshot
    
    Returns:
        Dict with:
        - uplift_safe: bool (True if no uplift safety issues found)
        - issues: List[str] (short neutral descriptors of problems)
        - status: str ("OK" | "WARN" | "BLOCK")
    """
    issues = []
    
    # Check for documents with invalid snippets
    invalid_snippets = snapshot.get("docs_with_invalid_snippets", [])
    for doc in invalid_snippets:
        issues.append(f"{doc} contains invalid code snippets")
    
    # Check for documents missing phase markers
    missing_markers = snapshot.get("docs_missing_phase_markers", [])
    for doc in missing_markers:
        issues.append(f"{doc} missing required phase markers")
    
    # Check for uplift mentions without disclaimer (critical)
    uplift_no_disclaimer = snapshot.get("docs_with_uplift_mentions_without_disclaimer", [])
    for doc in uplift_no_disclaimer:
        issues.append(f"{doc} mentions uplift but lacks disclaimer")
    
    # Determine status
    uplift_safe = len(uplift_no_disclaimer) == 0
    
    if len(issues) == 0:
        status = "OK"
    elif len(uplift_no_disclaimer) > 0:
        # Uplift mentions without disclaimer is a blocking issue
        status = "BLOCK"
    else:
        # Other issues are warnings
        status = "WARN"
    
    return {
        "uplift_safe": uplift_safe,
        "issues": sorted(issues),  # Sort for determinism
        "status": status
    }


def build_docs_section_for_evidence_pack(
    snapshot: Dict[str, Any],
    uplift_safety: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Build a documentation section for Evidence Pack v2 integration.
    
    Args:
        snapshot: Output from build_docs_governance_snapshot
        uplift_safety: Output from evaluate_uplift_safety
    
    Returns:
        Dict suitable for inclusion in Evidence Pack with:
        - docs_governance_ok: bool
        - docs_with_phase_issues: List[str]
        - docs_with_snippet_issues: List[str]
        - uplift_safe: bool
    """
    return {
        "docs_governance_ok": uplift_safety["status"] == "OK",
        "docs_with_phase_issues": sorted(
            snapshot.get("docs_missing_phase_markers", [])
        ),
        "docs_with_snippet_issues": sorted(
            snapshot.get("docs_with_invalid_snippets", [])
        ),
        "uplift_safe": uplift_safety["uplift_safe"]
    }


def main():
    """CLI entry point for docs governance validation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="MathLedger Docs Governance Layer"
    )
    parser.add_argument(
        "--snippet-report",
        type=Path,
        help="Path to snippet_check.py output JSON"
    )
    parser.add_argument(
        "--phase-marker-report",
        type=Path,
        help="Path to phase_marker_lint.py output JSON"
    )
    parser.add_argument(
        "--toc-index",
        type=Path,
        help="Path to generate_evidence_pack_toc.py output JSON"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs_governance_report.json"),
        help="Output path for governance report"
    )
    
    args = parser.parse_args()
    
    # Load input reports
    snippet_report = {}
    if args.snippet_report and args.snippet_report.exists():
        with open(args.snippet_report) as f:
            snippet_report = json.load(f)
    
    phase_marker_report = {}
    if args.phase_marker_report and args.phase_marker_report.exists():
        with open(args.phase_marker_report) as f:
            phase_marker_report = json.load(f)
    
    toc_index = {}
    if args.toc_index and args.toc_index.exists():
        with open(args.toc_index) as f:
            toc_index = json.load(f)
    
    # Build governance snapshot
    snapshot = build_docs_governance_snapshot(
        snippet_report, phase_marker_report, toc_index
    )
    
    # Evaluate uplift safety
    uplift_safety = evaluate_uplift_safety(snapshot)
    
    # Build evidence pack section
    evidence_section = build_docs_section_for_evidence_pack(
        snapshot, uplift_safety
    )
    
    # Combine into full report
    report = {
        "snapshot": snapshot,
        "uplift_safety": uplift_safety,
        "evidence_pack_section": evidence_section
    }
    
    # Write output
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    
    print(f"Docs governance report written to {args.output}")
    print(f"Status: {uplift_safety['status']}")
    if uplift_safety["issues"]:
        print(f"Issues found: {len(uplift_safety['issues'])}")
        for issue in uplift_safety["issues"]:
            print(f"  - {issue}")
    
    # Exit with appropriate code
    if uplift_safety["status"] == "BLOCK":
        return 2
    elif uplift_safety["status"] == "WARN":
        return 1
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
