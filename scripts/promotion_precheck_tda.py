#!/usr/bin/env python3
"""
Phase-Safe Promotion Pre-Check with TDA Integration

Extends the promotion pre-check pipeline with TDA (Topological Data Analysis)
governance signals. Evaluates fused evidence from multiple runs and blocks
promotion if TDA Hard Gate detects structural risks.

Phase Safety:
- Phase I: Basic precheck (SPARK hermetic tests + attestation)
- Phase II: Extended with TDA fusion and multi-run evidence

Usage:
    python scripts/promotion_precheck_tda.py --experiment-id EXP_001 --mode SHADOW
    python scripts/promotion_precheck_tda.py --evidence-pack path/to/fused_evidence.json --mode ENFORCE
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rfl.evidence_fusion import (
    FusedEvidenceSummary,
    load_fused_evidence,
    fuse_evidence_summaries,
    RunEntry,
    TDAFields,
    TDAOutcome,
)
from rfl.hard_gate import HardGateMode

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_ERROR = 2
EXIT_ABSTAIN = 3


class PromotionPrecheckTDA:
    """Promotion pre-check with TDA integration."""
    
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.results_dir = repo_root / "results"
        self.artifacts_dir = repo_root / "artifacts"
    
    def run_precheck(
        self,
        experiment_id: str,
        slice_name: str,
        mode: str = "SHADOW",
        evidence_pack_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Run promotion pre-check with TDA integration.
        
        Args:
            experiment_id: Experiment identifier
            slice_name: Curriculum slice name
            mode: TDA Hard Gate mode ("SHADOW" or "ENFORCE")
            evidence_pack_path: Optional path to pre-computed evidence pack
        
        Returns:
            Pre-check report dictionary
        """
        logger.info("=" * 80)
        logger.info("PROMOTION PRE-CHECK WITH TDA INTEGRATION")
        logger.info("=" * 80)
        logger.info(f"Experiment: {experiment_id}")
        logger.info(f"Slice: {slice_name}")
        logger.info(f"TDA Hard Gate Mode: {mode}")
        logger.info("")
        
        report = {
            "status": "fail",
            "experiment_id": experiment_id,
            "slice_name": slice_name,
            "tda_mode": mode,
            "phase": "II",  # Phase II precheck with TDA
            "checks": {},
            "tda_summary": {},
            "promotion_decision": {},
        }
        
        try:
            # Load or compute fused evidence
            if evidence_pack_path and evidence_pack_path.exists():
                logger.info(f"Loading pre-computed evidence pack: {evidence_pack_path}")
                fused_evidence = load_fused_evidence(evidence_pack_path)
            else:
                logger.info("Computing fused evidence from run logs...")
                fused_evidence = self._load_and_fuse_evidence(experiment_id, slice_name, mode)
            
            # Validate fused evidence
            validation_result = self._validate_fused_evidence(fused_evidence)
            report["checks"]["fused_evidence_validation"] = validation_result
            
            if not validation_result["passed"]:
                report["status"] = "fail"
                report["promotion_decision"] = {
                    "allowed": False,
                    "reason": "Fused evidence validation failed",
                    "details": validation_result,
                }
                return report
            
            # Check TDA governance signals
            tda_check = self._check_tda_signals(fused_evidence, mode)
            report["checks"]["tda_governance"] = tda_check
            report["tda_summary"] = self._summarize_tda(fused_evidence)
            
            # Determine promotion decision
            promotion_allowed = not fused_evidence.promotion_blocked
            
            if not promotion_allowed:
                report["status"] = "blocked"
                report["promotion_decision"] = {
                    "allowed": False,
                    "reason": fused_evidence.promotion_block_reason,
                    "mode": mode,
                }
            else:
                report["status"] = "pass"
                report["promotion_decision"] = {
                    "allowed": True,
                    "reason": "All TDA checks passed",
                    "mode": mode,
                    "advisory": mode == "SHADOW",
                }
            
            # Log summary
            self._log_summary(report)
            
            return report
            
        except Exception as e:
            logger.exception("Pre-check failed with exception")
            report["status"] = "error"
            report["error"] = str(e)
            return report
    
    def _load_and_fuse_evidence(
        self,
        experiment_id: str,
        slice_name: str,
        mode: str,
    ) -> FusedEvidenceSummary:
        """
        Load run logs and fuse evidence.
        
        Looks for:
        - results/{experiment_id}_baseline_*.jsonl
        - results/{experiment_id}_rfl_*.jsonl
        """
        baseline_runs = []
        rfl_runs = []
        
        # Search for baseline run logs
        baseline_pattern = f"{experiment_id}_baseline_*.jsonl"
        for log_path in self.results_dir.glob(baseline_pattern):
            runs = self._load_run_entries(log_path, mode="baseline")
            baseline_runs.extend(runs)
            logger.info(f"Loaded {len(runs)} baseline runs from {log_path.name}")
        
        # Search for RFL run logs
        rfl_pattern = f"{experiment_id}_rfl_*.jsonl"
        for log_path in self.results_dir.glob(rfl_pattern):
            runs = self._load_run_entries(log_path, mode="rfl")
            rfl_runs.extend(runs)
            logger.info(f"Loaded {len(runs)} RFL runs from {log_path.name}")
        
        if not baseline_runs and not rfl_runs:
            raise ValueError(f"No run logs found for experiment {experiment_id}")
        
        logger.info(f"Total baseline runs: {len(baseline_runs)}")
        logger.info(f"Total RFL runs: {len(rfl_runs)}")
        
        # Fuse evidence
        fused = fuse_evidence_summaries(
            baseline_runs=baseline_runs,
            rfl_runs=rfl_runs,
            experiment_id=experiment_id,
            slice_name=slice_name,
            tda_hard_gate_mode=mode,
        )
        
        return fused
    
    def _load_run_entries(self, log_path: Path, mode: str) -> List[RunEntry]:
        """Load run entries from JSONL log file."""
        runs = []
        
        with open(log_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    entry = json.loads(line.strip())
                    
                    # Extract TDA fields if present
                    tda_data = entry.get("tda", {})
                    if isinstance(tda_data, dict):
                        tda = TDAFields.from_dict(tda_data)
                    else:
                        # Mock TDA fields if not present
                        tda = TDAFields(
                            HSS=0.8,  # Default reasonable value
                            block_rate=entry.get("block_rate", 0.0),
                            tda_outcome=TDAOutcome.UNKNOWN,
                        )
                    
                    run = RunEntry(
                        run_id=entry.get("run_id", f"{log_path.stem}_{line_num}"),
                        experiment_id=entry.get("experiment_id", "unknown"),
                        slice_name=entry.get("slice_name", "unknown"),
                        mode=mode,
                        coverage_rate=entry.get("coverage_rate", 0.0),
                        novelty_rate=entry.get("novelty_rate", 0.0),
                        throughput=entry.get("throughput", 0.0),
                        success_rate=entry.get("success_rate", 0.0),
                        abstention_fraction=entry.get("abstention_fraction", 1.0),
                        tda=tda,
                        timestamp=entry.get("timestamp"),
                        cycle_count=entry.get("cycle_count"),
                        metadata=entry.get("metadata", {}),
                    )
                    
                    runs.append(run)
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line {line_num} in {log_path.name}: {e}")
                    continue
                except Exception as e:
                    logger.warning(f"Failed to create RunEntry from line {line_num} in {log_path.name}: {e}")
                    continue
        
        return runs
    
    def _validate_fused_evidence(self, fused: FusedEvidenceSummary) -> Dict[str, Any]:
        """Validate fused evidence for completeness and consistency."""
        issues = []
        
        # Check for minimum run count
        if len(fused.baseline_runs) < 1:
            issues.append("No baseline runs found")
        
        if len(fused.rfl_runs) < 1:
            issues.append("No RFL runs found")
        
        # Check fusion hash
        if not fused.fusion_hash:
            issues.append("Fusion hash not computed")
        
        # Check for critical inconsistencies
        critical_inconsistencies = [
            inc for inc in fused.inconsistencies
            if inc.severity == "error"
        ]
        
        if critical_inconsistencies:
            issues.append(f"{len(critical_inconsistencies)} critical inconsistencies detected")
        
        passed = len(issues) == 0
        
        return {
            "passed": passed,
            "issues": issues,
            "baseline_count": len(fused.baseline_runs),
            "rfl_count": len(fused.rfl_runs),
            "fusion_hash": fused.fusion_hash,
        }
    
    def _check_tda_signals(
        self,
        fused: FusedEvidenceSummary,
        mode: str,
    ) -> Dict[str, Any]:
        """Check TDA governance signals."""
        checks = {
            "mean_block_rate": {
                "value": fused.mean_block_rate,
                "threshold": 0.8,
                "passed": fused.mean_block_rate < 0.8,
            },
            "tda_pass_rate": {
                "value": fused.tda_pass_rate,
                "threshold": 0.5,
                "passed": fused.tda_pass_rate >= 0.5,
            },
            "hard_gate_blocks": {
                "value": fused.tda_hard_gate_blocks,
                "threshold": 0,
                "passed": fused.tda_hard_gate_blocks == 0 or mode == "SHADOW",
            },
        }
        
        all_passed = all(check["passed"] for check in checks.values())
        
        return {
            "passed": all_passed,
            "checks": checks,
            "mode": mode,
        }
    
    def _summarize_tda(self, fused: FusedEvidenceSummary) -> Dict[str, Any]:
        """Summarize TDA governance signals."""
        return {
            "mean_block_rate": fused.mean_block_rate,
            "tda_pass_rate": fused.tda_pass_rate,
            "tda_hard_gate_blocks": fused.tda_hard_gate_blocks,
            "inconsistency_count": len(fused.inconsistencies),
            "critical_inconsistencies": len([
                inc for inc in fused.inconsistencies if inc.severity == "error"
            ]),
            "promotion_blocked": fused.promotion_blocked,
            "fusion_hash": fused.fusion_hash,
        }
    
    def _log_summary(self, report: Dict[str, Any]) -> None:
        """Log pre-check summary."""
        logger.info("")
        logger.info("=" * 80)
        logger.info("PRE-CHECK SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Status: {report['status'].upper()}")
        logger.info(f"Experiment: {report['experiment_id']}")
        logger.info(f"Slice: {report['slice_name']}")
        logger.info(f"TDA Mode: {report['tda_mode']}")
        logger.info("")
        
        tda_summary = report.get("tda_summary", {})
        logger.info("TDA Summary:")
        logger.info(f"  Mean block rate: {tda_summary.get('mean_block_rate', 0):.2%}")
        logger.info(f"  TDA pass rate: {tda_summary.get('tda_pass_rate', 0):.2%}")
        logger.info(f"  Hard gate blocks: {tda_summary.get('tda_hard_gate_blocks', 0)}")
        logger.info(f"  Inconsistencies: {tda_summary.get('inconsistency_count', 0)}")
        logger.info(f"  Fusion hash: {tda_summary.get('fusion_hash', 'N/A')}")
        logger.info("")
        
        decision = report.get("promotion_decision", {})
        if decision.get("allowed"):
            logger.info("✓ PROMOTION ALLOWED")
            if decision.get("advisory"):
                logger.info("  (Advisory mode - TDA checks informational only)")
        else:
            logger.info("✗ PROMOTION BLOCKED")
            logger.info(f"  Reason: {decision.get('reason', 'Unknown')}")
        
        logger.info("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Phase-safe promotion pre-check with TDA integration"
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        required=True,
        help="Experiment identifier",
    )
    parser.add_argument(
        "--slice-name",
        type=str,
        default="unknown",
        help="Curriculum slice name",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["SHADOW", "ENFORCE"],
        default="SHADOW",
        help="TDA Hard Gate mode (SHADOW=log only, ENFORCE=block)",
    )
    parser.add_argument(
        "--evidence-pack",
        type=Path,
        help="Path to pre-computed fused evidence pack",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path for pre-check report",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root directory",
    )
    
    args = parser.parse_args()
    
    # Run pre-check
    precheck = PromotionPrecheckTDA(args.repo_root)
    
    try:
        report = precheck.run_precheck(
            experiment_id=args.experiment_id,
            slice_name=args.slice_name,
            mode=args.mode,
            evidence_pack_path=args.evidence_pack,
        )
        
        # Save report if output path specified
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, 'w') as f:
                json.dump(report, f, indent=2, sort_keys=True)
            logger.info(f"Pre-check report saved to: {args.output}")
        
        # Determine exit code
        status = report.get("status", "error")
        if status == "pass":
            sys.exit(EXIT_PASS)
        elif status == "blocked":
            # Blocked is different from fail - TDA hard gate decision
            if args.mode == "SHADOW":
                # In shadow mode, log but allow
                logger.warning("SHADOW mode: promotion would be blocked in ENFORCE mode")
                sys.exit(EXIT_PASS)
            else:
                sys.exit(EXIT_FAIL)
        elif status == "fail":
            sys.exit(EXIT_FAIL)
        elif status == "error":
            sys.exit(EXIT_ERROR)
        else:
            sys.exit(EXIT_ERROR)
    
    except Exception as e:
        logger.exception("Pre-check failed")
        sys.exit(EXIT_ERROR)


if __name__ == "__main__":
    main()
