"""
PHASE II — TDA-Aware Evidence Fusion

This module implements multi-run evidence fusion with TDA (Timeout/Depth Analysis)
governance integration. It detects conflicts between uplift decisions and TDA outcomes,
flagging hidden instabilities for advisory review.

Key Features:
- Extended evidence summary schema with uplift and TDA fields
- Inconsistency detection (uplift/TDA conflicts, hidden instability)
- Alignment status computation (OK/WARN/BLOCK)
- Advisory-only blocking (does not claim "uplift achieved")

Schema Extension:
Each run in a multi-run summary carries:
{
  "run_id": "...",
  "uplift": {
    "delta_p": float,
    "abstention_rate": float,
    "promotion_decision": "PASS" | "WARN" | "BLOCK"
  },
  "tda": {
    "HSS": float,  # Hidden State Score
    "block_rate": float,
    "tda_outcome": "OK" | "ATTENTION" | "BLOCK"
  }
}

Alignment Rules:
- BLOCK if any run has PASS uplift but BLOCK TDA (conflict)
- WARN if any run has PASS uplift but low HSS (hidden instability)
- OK otherwise
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class PromotionDecision(str, Enum):
    """Promotion decision for uplift evidence."""
    PASS = "PASS"
    WARN = "WARN"
    BLOCK = "BLOCK"


class TDAOutcome(str, Enum):
    """TDA analysis outcome."""
    OK = "OK"
    ATTENTION = "ATTENTION"
    BLOCK = "BLOCK"


class AlignmentStatus(str, Enum):
    """Alignment status between uplift and TDA."""
    OK = "OK"
    WARN = "WARN"
    BLOCK = "BLOCK"


@dataclass
class UpliftMetrics:
    """Uplift metrics for a single run."""
    delta_p: float
    abstention_rate: float
    promotion_decision: PromotionDecision


@dataclass
class TDAMetrics:
    """TDA metrics for a single run."""
    HSS: float  # Hidden State Score
    block_rate: float
    tda_outcome: TDAOutcome


@dataclass
class RunEvidence:
    """Evidence summary for a single run."""
    run_id: str
    uplift: UpliftMetrics
    tda: TDAMetrics
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "run_id": self.run_id,
            "uplift": {
                "delta_p": self.uplift.delta_p,
                "abstention_rate": self.uplift.abstention_rate,
                "promotion_decision": self.uplift.promotion_decision.value,
            },
            "tda": {
                "HSS": self.tda.HSS,
                "block_rate": self.tda.block_rate,
                "tda_outcome": self.tda.tda_outcome.value,
            },
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> RunEvidence:
        """Reconstruct from dict."""
        uplift = UpliftMetrics(
            delta_p=d["uplift"]["delta_p"],
            abstention_rate=d["uplift"]["abstention_rate"],
            promotion_decision=PromotionDecision(d["uplift"]["promotion_decision"]),
        )
        tda = TDAMetrics(
            HSS=d["tda"]["HSS"],
            block_rate=d["tda"]["block_rate"],
            tda_outcome=TDAOutcome(d["tda"]["tda_outcome"]),
        )
        return cls(
            run_id=d["run_id"],
            uplift=uplift,
            tda=tda,
            metadata=d.get("metadata", {}),
        )


@dataclass
class TDAAlignment:
    """TDA alignment analysis results."""
    conflicted_runs: List[str] = field(default_factory=list)
    hidden_instability_runs: List[str] = field(default_factory=list)
    alignment_status: AlignmentStatus = AlignmentStatus.OK

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "conflicted_runs": self.conflicted_runs,
            "hidden_instability_runs": self.hidden_instability_runs,
            "alignment_status": self.alignment_status.value,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> TDAAlignment:
        """Reconstruct from dict."""
        return cls(
            conflicted_runs=d.get("conflicted_runs", []),
            hidden_instability_runs=d.get("hidden_instability_runs", []),
            alignment_status=AlignmentStatus(d["alignment_status"]),
        )


@dataclass
class FusedEvidence:
    """Fused evidence summary from multiple runs."""
    runs: List[RunEvidence]
    tda_alignment: TDAAlignment
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "runs": [r.to_dict() for r in self.runs],
            "tda_alignment": self.tda_alignment.to_dict(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> FusedEvidence:
        """Reconstruct from dict."""
        runs = [RunEvidence.from_dict(r) for r in d["runs"]]
        tda_alignment = TDAAlignment.from_dict(d["tda_alignment"])
        return cls(
            runs=runs,
            tda_alignment=tda_alignment,
            metadata=d.get("metadata", {}),
        )


def detect_conflicts(runs: List[RunEvidence], hss_threshold: float = 0.7) -> TDAAlignment:
    """
    Detect conflicts between uplift decisions and TDA outcomes.

    Rules:
    1. Conflict: PASS uplift + BLOCK TDA → alignment BLOCK
    2. Hidden instability: PASS uplift + low HSS → alignment WARN
    3. Otherwise: alignment OK

    Args:
        runs: List of run evidence summaries
        hss_threshold: Minimum HSS value to avoid hidden instability warning

    Returns:
        TDAAlignment with conflict detection results
    """
    conflicted_runs = []
    hidden_instability_runs = []

    for run in runs:
        # Check for uplift/TDA conflict
        if (run.uplift.promotion_decision == PromotionDecision.PASS and
                run.tda.tda_outcome == TDAOutcome.BLOCK):
            conflicted_runs.append(run.run_id)

        # Check for hidden instability (PASS with low HSS)
        if (run.uplift.promotion_decision == PromotionDecision.PASS and
                run.tda.HSS < hss_threshold):
            hidden_instability_runs.append(run.run_id)

    # Determine alignment status
    if conflicted_runs:
        alignment_status = AlignmentStatus.BLOCK
    elif hidden_instability_runs:
        alignment_status = AlignmentStatus.WARN
    else:
        alignment_status = AlignmentStatus.OK

    return TDAAlignment(
        conflicted_runs=conflicted_runs,
        hidden_instability_runs=hidden_instability_runs,
        alignment_status=alignment_status,
    )


def fuse_evidence_summaries(
    runs: List[RunEvidence],
    hss_threshold: float = 0.7,
    metadata: Optional[Dict[str, Any]] = None,
) -> FusedEvidence:
    """
    Fuse multiple run evidence summaries with TDA-aware conflict detection.

    This function:
    1. Takes multiple run evidence summaries (each with uplift + TDA metrics)
    2. Detects conflicts between uplift decisions and TDA outcomes
    3. Flags hidden instability (PASS with low HSS)
    4. Produces a fused summary with alignment status

    Args:
        runs: List of run evidence summaries
        hss_threshold: Minimum HSS value to avoid hidden instability warning
        metadata: Optional metadata to attach to fused summary

    Returns:
        FusedEvidence with runs, TDA alignment, and metadata
    """
    if not runs:
        # Empty runs list - return OK alignment (no conflicts)
        return FusedEvidence(
            runs=[],
            tda_alignment=TDAAlignment(alignment_status=AlignmentStatus.OK),
            metadata=metadata or {},
        )

    # Detect conflicts and hidden instability
    tda_alignment = detect_conflicts(runs, hss_threshold)

    return FusedEvidence(
        runs=runs,
        tda_alignment=tda_alignment,
        metadata=metadata or {},
    )


def load_evidence_summaries(path: Path) -> List[RunEvidence]:
    """
    Load evidence summaries from a JSON file.

    Expected format:
    {
      "runs": [
        {
          "run_id": "...",
          "uplift": {...},
          "tda": {...}
        },
        ...
      ]
    }

    Args:
        path: Path to JSON file

    Returns:
        List of RunEvidence objects
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "runs" not in data:
        raise ValueError(f"Invalid evidence file: missing 'runs' key in {path}")

    return [RunEvidence.from_dict(r) for r in data["runs"]]


def save_fused_evidence(fused: FusedEvidence, path: Path) -> None:
    """
    Save fused evidence to a JSON file.

    Args:
        fused: FusedEvidence to save
        path: Path to output JSON file
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(fused.to_dict(), f, indent=2, sort_keys=True)


def main() -> None:
    """CLI entry point for evidence fusion."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Fuse multiple run evidence summaries with TDA-aware conflict detection"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to input JSON file with run evidence summaries",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Path to output JSON file for fused evidence",
    )
    parser.add_argument(
        "--hss-threshold",
        type=float,
        default=0.7,
        help="Minimum HSS value to avoid hidden instability warning (default: 0.7)",
    )

    args = parser.parse_args()

    # Load evidence summaries
    print(f"Loading evidence summaries from {args.input}...")
    runs = load_evidence_summaries(args.input)
    print(f"Loaded {len(runs)} run evidence summaries")

    # Fuse evidence with TDA awareness
    print("Fusing evidence summaries with TDA conflict detection...")
    fused = fuse_evidence_summaries(runs, hss_threshold=args.hss_threshold)

    # Print summary
    print(f"\nFusion complete:")
    print(f"  Runs: {len(fused.runs)}")
    print(f"  Conflicted runs: {len(fused.tda_alignment.conflicted_runs)}")
    print(f"  Hidden instability runs: {len(fused.tda_alignment.hidden_instability_runs)}")
    print(f"  Alignment status: {fused.tda_alignment.alignment_status.value}")

    if fused.tda_alignment.conflicted_runs:
        print(f"\n⚠️  CONFLICT DETECTED:")
        for run_id in fused.tda_alignment.conflicted_runs:
            print(f"    - {run_id}: PASS uplift but BLOCK TDA")

    if fused.tda_alignment.hidden_instability_runs:
        print(f"\n⚠️  HIDDEN INSTABILITY:")
        for run_id in fused.tda_alignment.hidden_instability_runs:
            print(f"    - {run_id}: PASS uplift but low HSS")

    # Save fused evidence
    print(f"\nSaving fused evidence to {args.output}...")
    save_fused_evidence(fused, args.output)
    print("Done.")


if __name__ == "__main__":
    main()
