"""
RFL Multi-Run Evidence Fusion with TDA Integration

Fuses evidence from multiple experimental runs while integrating TDA (Topological
Data Analysis) governance signals. Provides inconsistency detection and pre-check
gates for promotion decisions.

PHASE II — U2 Uplift Experiments Extension

Constraints:
- Advisory only (no uplift claims)
- Deterministic ordering guarantees
- TDA Hard Gate awareness (SHADOW mode support)
- Phase-safe (respects Phase I vs Phase II boundaries)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class TDAOutcome(Enum):
    """TDA Hard Gate decision outcome."""
    PASS = "pass"
    WARN = "warn"
    BLOCK = "block"
    SHADOW = "shadow"  # SHADOW mode - log but don't block
    UNKNOWN = "unknown"


class InconsistencyType(Enum):
    """Types of inconsistencies between uplift and TDA signals."""
    UPLIFT_WITHOUT_QUALITY = "uplift_without_quality"  # Uplift but poor TDA metrics
    DEGRADATION_WITH_GOOD_TDA = "degradation_with_good_tda"  # Regression but good TDA
    HIGH_BLOCK_RATE = "high_block_rate"  # Excessive event blocking
    TDA_STRUCTURAL_RISK = "tda_structural_risk"  # TDA Hard Gate detected risk
    MISSING_TDA_DATA = "missing_tda_data"  # TDA fields not populated
    NONE = "none"


@dataclass
class TDAFields:
    """
    TDA governance signal fields for a single run.
    
    Attributes:
        HSS: Hash Stability Score or similar governance metric
        block_rate: Rate of events blocked by verification gate
        tda_outcome: TDA Hard Gate decision (PASS/WARN/BLOCK/SHADOW)
    """
    HSS: Optional[float] = None
    block_rate: Optional[float] = None
    tda_outcome: Optional[TDAOutcome] = None
    
    def __post_init__(self):
        """Validate TDA fields."""
        if self.block_rate is not None:
            if not (0.0 <= self.block_rate <= 1.0):
                raise ValueError(f"block_rate must be in [0, 1], got {self.block_rate}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "HSS": self.HSS,
            "block_rate": self.block_rate,
            "tda_outcome": self.tda_outcome.value if self.tda_outcome else None,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TDAFields:
        """Deserialize from dictionary."""
        outcome = data.get("tda_outcome")
        if outcome and isinstance(outcome, str):
            try:
                outcome = TDAOutcome(outcome)
            except ValueError:
                outcome = TDAOutcome.UNKNOWN
        
        return cls(
            HSS=data.get("HSS"),
            block_rate=data.get("block_rate"),
            tda_outcome=outcome,
        )


@dataclass
class RunEntry:
    """
    Evidence entry for a single experimental run with TDA fields.
    
    Extends run ledger entries to include TDA governance signals.
    """
    run_id: str
    experiment_id: str
    slice_name: str
    mode: str  # "baseline" or "rfl"
    
    # Performance metrics
    coverage_rate: float
    novelty_rate: float
    throughput: float
    success_rate: float
    abstention_fraction: float
    
    # TDA governance fields
    tda: TDAFields = field(default_factory=lambda: TDAFields())
    
    # Additional metadata
    timestamp: Optional[str] = None
    cycle_count: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize timestamp if not set."""
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc).isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary with deterministic ordering."""
        return {
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "slice_name": self.slice_name,
            "mode": self.mode,
            "coverage_rate": self.coverage_rate,
            "novelty_rate": self.novelty_rate,
            "throughput": self.throughput,
            "success_rate": self.success_rate,
            "abstention_fraction": self.abstention_fraction,
            "tda": self.tda.to_dict(),
            "timestamp": self.timestamp,
            "cycle_count": self.cycle_count,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RunEntry:
        """Deserialize from dictionary."""
        tda_data = data.get("tda", {})
        if isinstance(tda_data, dict):
            tda = TDAFields.from_dict(tda_data)
        else:
            tda = TDAFields()
        
        return cls(
            run_id=data["run_id"],
            experiment_id=data["experiment_id"],
            slice_name=data["slice_name"],
            mode=data["mode"],
            coverage_rate=data["coverage_rate"],
            novelty_rate=data["novelty_rate"],
            throughput=data["throughput"],
            success_rate=data["success_rate"],
            abstention_fraction=data["abstention_fraction"],
            tda=tda,
            timestamp=data.get("timestamp"),
            cycle_count=data.get("cycle_count"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class InconsistencyReport:
    """Report of inconsistencies between uplift and TDA signals."""
    inconsistency_type: InconsistencyType
    severity: str  # "info", "warning", "error"
    message: str
    affected_runs: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "inconsistency_type": self.inconsistency_type.value,
            "severity": self.severity,
            "message": self.message,
            "affected_runs": self.affected_runs,
            "details": self.details,
        }


@dataclass
class FusedEvidenceSummary:
    """
    Summary of fused evidence from multiple runs with TDA integration.
    
    Provides aggregate metrics, inconsistency reports, and TDA governance signals.
    """
    experiment_id: str
    slice_name: str
    
    # Run entries (baseline and RFL)
    baseline_runs: List[RunEntry] = field(default_factory=list)
    rfl_runs: List[RunEntry] = field(default_factory=list)
    
    # Aggregate metrics
    baseline_mean_coverage: float = 0.0
    rfl_mean_coverage: float = 0.0
    baseline_mean_abstention: float = 0.0
    rfl_mean_abstention: float = 0.0
    
    # TDA aggregate signals
    mean_block_rate: float = 0.0
    tda_pass_rate: float = 0.0
    tda_hard_gate_blocks: int = 0
    
    # Inconsistency reports
    inconsistencies: List[InconsistencyReport] = field(default_factory=list)
    
    # Promotion eligibility
    promotion_blocked: bool = False
    promotion_block_reason: Optional[str] = None
    
    # Metadata
    fused_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    fusion_hash: Optional[str] = None
    
    def compute_fusion_hash(self) -> str:
        """
        Compute deterministic hash of fused evidence.
        
        Ensures reproducibility by sorting run entries and using canonical JSON.
        """
        # Sort runs by run_id for determinism
        sorted_baseline = sorted(self.baseline_runs, key=lambda r: r.run_id)
        sorted_rfl = sorted(self.rfl_runs, key=lambda r: r.run_id)
        
        # Build canonical payload
        payload = {
            "experiment_id": self.experiment_id,
            "slice_name": self.slice_name,
            "baseline_runs": [r.to_dict() for r in sorted_baseline],
            "rfl_runs": [r.to_dict() for r in sorted_rfl],
        }
        
        # Canonical JSON encoding
        canonical = json.dumps(payload, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(canonical.encode('utf-8')).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "slice_name": self.slice_name,
            "baseline_runs": [r.to_dict() for r in self.baseline_runs],
            "rfl_runs": [r.to_dict() for r in self.rfl_runs],
            "baseline_mean_coverage": self.baseline_mean_coverage,
            "rfl_mean_coverage": self.rfl_mean_coverage,
            "baseline_mean_abstention": self.baseline_mean_abstention,
            "rfl_mean_abstention": self.rfl_mean_abstention,
            "mean_block_rate": self.mean_block_rate,
            "tda_pass_rate": self.tda_pass_rate,
            "tda_hard_gate_blocks": self.tda_hard_gate_blocks,
            "inconsistencies": [inc.to_dict() for inc in self.inconsistencies],
            "promotion_blocked": self.promotion_blocked,
            "promotion_block_reason": self.promotion_block_reason,
            "fused_at": self.fused_at,
            "fusion_hash": self.fusion_hash,
        }


def fuse_evidence_summaries(
    baseline_runs: List[RunEntry],
    rfl_runs: List[RunEntry],
    experiment_id: str,
    slice_name: str,
    tda_hard_gate_mode: str = "SHADOW",
) -> FusedEvidenceSummary:
    """
    Fuse evidence from multiple runs with TDA governance integration.
    
    Aggregates metrics, detects inconsistencies between uplift and TDA signals,
    and determines promotion eligibility based on TDA Hard Gate.
    
    Args:
        baseline_runs: List of baseline run entries
        rfl_runs: List of RFL run entries
        experiment_id: Experiment identifier
        slice_name: Curriculum slice name
        tda_hard_gate_mode: "SHADOW" (log only) or "ENFORCE" (block)
    
    Returns:
        FusedEvidenceSummary with aggregate metrics and inconsistency reports
    """
    logger.info(f"Fusing evidence for experiment {experiment_id}, slice {slice_name}")
    logger.info(f"  Baseline runs: {len(baseline_runs)}")
    logger.info(f"  RFL runs: {len(rfl_runs)}")
    logger.info(f"  TDA Hard Gate mode: {tda_hard_gate_mode}")
    
    summary = FusedEvidenceSummary(
        experiment_id=experiment_id,
        slice_name=slice_name,
        baseline_runs=baseline_runs,
        rfl_runs=rfl_runs,
    )
    
    # Compute aggregate metrics
    if baseline_runs:
        summary.baseline_mean_coverage = sum(r.coverage_rate for r in baseline_runs) / len(baseline_runs)
        summary.baseline_mean_abstention = sum(r.abstention_fraction for r in baseline_runs) / len(baseline_runs)
    
    if rfl_runs:
        summary.rfl_mean_coverage = sum(r.coverage_rate for r in rfl_runs) / len(rfl_runs)
        summary.rfl_mean_abstention = sum(r.abstention_fraction for r in rfl_runs) / len(rfl_runs)
    
    # Compute TDA aggregate signals
    all_runs = baseline_runs + rfl_runs
    if all_runs:
        # Mean block rate across all runs
        block_rates = [r.tda.block_rate for r in all_runs if r.tda.block_rate is not None]
        if block_rates:
            summary.mean_block_rate = sum(block_rates) / len(block_rates)
        
        # TDA pass rate
        tda_outcomes = [r.tda.tda_outcome for r in all_runs if r.tda.tda_outcome is not None]
        if tda_outcomes:
            pass_count = sum(1 for o in tda_outcomes if o == TDAOutcome.PASS)
            summary.tda_pass_rate = pass_count / len(tda_outcomes)
            summary.tda_hard_gate_blocks = sum(1 for o in tda_outcomes if o == TDAOutcome.BLOCK)
    
    # Detect inconsistencies
    summary.inconsistencies = _detect_inconsistencies(baseline_runs, rfl_runs)
    
    # Evaluate TDA Hard Gate
    promotion_blocked, block_reason = _evaluate_tda_hard_gate(
        summary,
        mode=tda_hard_gate_mode,
    )
    
    summary.promotion_blocked = promotion_blocked
    summary.promotion_block_reason = block_reason
    
    # Compute fusion hash for deterministic reproduction
    summary.fusion_hash = summary.compute_fusion_hash()
    
    logger.info(f"Fusion complete. Hash: {summary.fusion_hash}")
    logger.info(f"  Inconsistencies detected: {len(summary.inconsistencies)}")
    logger.info(f"  Promotion blocked: {summary.promotion_blocked}")
    if summary.promotion_blocked:
        logger.warning(f"  Block reason: {summary.promotion_block_reason}")
    
    return summary


def _detect_inconsistencies(
    baseline_runs: List[RunEntry],
    rfl_runs: List[RunEntry],
) -> List[InconsistencyReport]:
    """
    Detect inconsistencies between uplift signals and TDA governance signals.
    
    Checks for:
    1. Uplift without quality (high coverage but poor TDA metrics)
    2. Degradation with good TDA (regression but good TDA signals)
    3. High block rates (excessive event blocking)
    4. TDA structural risks
    5. Missing TDA data
    """
    inconsistencies: List[InconsistencyReport] = []
    
    # Check for missing TDA data
    all_runs = baseline_runs + rfl_runs
    missing_tda = [r for r in all_runs if r.tda.tda_outcome is None]
    if missing_tda:
        inconsistencies.append(InconsistencyReport(
            inconsistency_type=InconsistencyType.MISSING_TDA_DATA,
            severity="warning",
            message=f"{len(missing_tda)} runs missing TDA governance data",
            affected_runs=[r.run_id for r in missing_tda],
            details={"count": len(missing_tda)},
        ))
    
    # Check for high block rates
    high_block_runs = [
        r for r in all_runs
        if r.tda.block_rate is not None and r.tda.block_rate > 0.5
    ]
    if high_block_runs:
        inconsistencies.append(InconsistencyReport(
            inconsistency_type=InconsistencyType.HIGH_BLOCK_RATE,
            severity="error",
            message=f"{len(high_block_runs)} runs with excessive event blocking (>50%)",
            affected_runs=[r.run_id for r in high_block_runs],
            details={
                "mean_block_rate": sum(r.tda.block_rate for r in high_block_runs) / len(high_block_runs),
            },
        ))
    
    # Check for TDA structural risks
    tda_risk_runs = [
        r for r in all_runs
        if r.tda.tda_outcome == TDAOutcome.BLOCK or r.tda.tda_outcome == TDAOutcome.WARN
    ]
    if tda_risk_runs:
        severity = "error" if any(r.tda.tda_outcome == TDAOutcome.BLOCK for r in tda_risk_runs) else "warning"
        inconsistencies.append(InconsistencyReport(
            inconsistency_type=InconsistencyType.TDA_STRUCTURAL_RISK,
            severity=severity,
            message=f"{len(tda_risk_runs)} runs flagged by TDA Hard Gate",
            affected_runs=[r.run_id for r in tda_risk_runs],
            details={
                "block_count": sum(1 for r in tda_risk_runs if r.tda.tda_outcome == TDAOutcome.BLOCK),
                "warn_count": sum(1 for r in tda_risk_runs if r.tda.tda_outcome == TDAOutcome.WARN),
            },
        ))
    
    # Compare baseline vs RFL for uplift inconsistencies
    if baseline_runs and rfl_runs:
        baseline_mean_coverage = sum(r.coverage_rate for r in baseline_runs) / len(baseline_runs)
        rfl_mean_coverage = sum(r.coverage_rate for r in rfl_runs) / len(rfl_runs)
        
        baseline_mean_block_rate = sum(
            r.tda.block_rate for r in baseline_runs if r.tda.block_rate is not None
        ) / max(1, len([r for r in baseline_runs if r.tda.block_rate is not None]))
        
        rfl_mean_block_rate = sum(
            r.tda.block_rate for r in rfl_runs if r.tda.block_rate is not None
        ) / max(1, len([r for r in rfl_runs if r.tda.block_rate is not None]))
        
        # Uplift without quality: RFL coverage improved but block rate also increased significantly
        if rfl_mean_coverage > baseline_mean_coverage and rfl_mean_block_rate > baseline_mean_block_rate * 1.5:
            inconsistencies.append(InconsistencyReport(
                inconsistency_type=InconsistencyType.UPLIFT_WITHOUT_QUALITY,
                severity="warning",
                message="RFL shows coverage uplift but increased event blocking",
                details={
                    "baseline_coverage": baseline_mean_coverage,
                    "rfl_coverage": rfl_mean_coverage,
                    "baseline_block_rate": baseline_mean_block_rate,
                    "rfl_block_rate": rfl_mean_block_rate,
                },
            ))
        
        # Degradation with good TDA: RFL coverage decreased but block rate improved
        if rfl_mean_coverage < baseline_mean_coverage and rfl_mean_block_rate < baseline_mean_block_rate * 0.5:
            inconsistencies.append(InconsistencyReport(
                inconsistency_type=InconsistencyType.DEGRADATION_WITH_GOOD_TDA,
                severity="info",
                message="RFL shows coverage degradation but improved event quality",
                details={
                    "baseline_coverage": baseline_mean_coverage,
                    "rfl_coverage": rfl_mean_coverage,
                    "baseline_block_rate": baseline_mean_block_rate,
                    "rfl_block_rate": rfl_mean_block_rate,
                },
            ))
    
    return inconsistencies


def _evaluate_tda_hard_gate(
    summary: FusedEvidenceSummary,
    mode: str = "SHADOW",
) -> Tuple[bool, Optional[str]]:
    """
    Evaluate TDA Hard Gate to determine promotion eligibility.
    
    In SHADOW mode: Log issues but don't block promotion
    In ENFORCE mode: Block promotion on structural risks
    
    Returns:
        Tuple of (promotion_blocked, block_reason)
    """
    if mode.upper() not in ["SHADOW", "ENFORCE"]:
        logger.warning(f"Unknown TDA Hard Gate mode: {mode}, defaulting to SHADOW")
        mode = "SHADOW"
    
    # Check for critical TDA failures
    critical_inconsistencies = [
        inc for inc in summary.inconsistencies
        if inc.severity == "error"
    ]
    
    if not critical_inconsistencies:
        return False, None
    
    # In SHADOW mode, log but don't block
    if mode.upper() == "SHADOW":
        logger.warning(f"TDA Hard Gate (SHADOW): {len(critical_inconsistencies)} critical issues detected")
        for inc in critical_inconsistencies:
            logger.warning(f"  - {inc.inconsistency_type.value}: {inc.message}")
        return False, None
    
    # In ENFORCE mode, block promotion
    reasons = [f"{inc.inconsistency_type.value}: {inc.message}" for inc in critical_inconsistencies]
    block_reason = f"TDA Hard Gate blocked promotion. Critical issues: {'; '.join(reasons)}"
    
    logger.error(f"TDA Hard Gate (ENFORCE): Blocking promotion")
    logger.error(f"  Reason: {block_reason}")
    
    return True, block_reason


def save_fused_evidence(
    summary: FusedEvidenceSummary,
    output_path: Path,
) -> None:
    """
    Save fused evidence summary to JSON file.
    
    Uses canonical JSON encoding for deterministic output.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(
            summary.to_dict(),
            f,
            indent=2,
            sort_keys=True,
        )
    
    logger.info(f"Fused evidence saved to: {output_path}")


def load_fused_evidence(input_path: Path) -> FusedEvidenceSummary:
    """Load fused evidence summary from JSON file."""
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    baseline_runs = [RunEntry.from_dict(r) for r in data.get("baseline_runs", [])]
    rfl_runs = [RunEntry.from_dict(r) for r in data.get("rfl_runs", [])]
    
    inconsistencies = []
    for inc_data in data.get("inconsistencies", []):
        inconsistencies.append(InconsistencyReport(
            inconsistency_type=InconsistencyType(inc_data["inconsistency_type"]),
            severity=inc_data["severity"],
            message=inc_data["message"],
            affected_runs=inc_data.get("affected_runs", []),
            details=inc_data.get("details", {}),
        ))
    
    summary = FusedEvidenceSummary(
        experiment_id=data["experiment_id"],
        slice_name=data["slice_name"],
        baseline_runs=baseline_runs,
        rfl_runs=rfl_runs,
        baseline_mean_coverage=data.get("baseline_mean_coverage", 0.0),
        rfl_mean_coverage=data.get("rfl_mean_coverage", 0.0),
        baseline_mean_abstention=data.get("baseline_mean_abstention", 0.0),
        rfl_mean_abstention=data.get("rfl_mean_abstention", 0.0),
        mean_block_rate=data.get("mean_block_rate", 0.0),
        tda_pass_rate=data.get("tda_pass_rate", 0.0),
        tda_hard_gate_blocks=data.get("tda_hard_gate_blocks", 0),
        inconsistencies=inconsistencies,
        promotion_blocked=data.get("promotion_blocked", False),
        promotion_block_reason=data.get("promotion_block_reason"),
        fused_at=data.get("fused_at", datetime.now(timezone.utc).isoformat()),
        fusion_hash=data.get("fusion_hash"),
    )
    
    return summary


__all__ = [
    "TDAOutcome",
    "InconsistencyType",
    "TDAFields",
    "RunEntry",
    "InconsistencyReport",
    "FusedEvidenceSummary",
    "fuse_evidence_summaries",
    "save_fused_evidence",
    "load_fused_evidence",
]
