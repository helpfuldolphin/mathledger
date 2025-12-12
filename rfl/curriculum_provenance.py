# REAL-READY
"""
RFL Curriculum Provenance Integration

Provides helper functions for integrating curriculum fingerprinting and drift
detection into the RFL runner. This module extends RunLedgerEntry with curriculum
provenance fields and provides drift checking utilities.

Author: MANUS-E, Curriculum Integrity Engineer
Status: OPERATIONAL
"""

from typing import Dict, Any
from pathlib import Path

from backend.frontier.curriculum import load as load_curriculum
from backend.frontier.curriculum_drift_enforcement import (
    check_curriculum_drift,
    stamp_run_ledger_with_fingerprint
)


def initialize_curriculum_baseline(curriculum_slug: str = "pl") -> Dict[str, Any]:
    """
    Initialize curriculum baseline for drift detection.
    
    Args:
        curriculum_slug: Curriculum system slug (default: "pl")
    
    Returns:
        Dictionary containing baseline_fingerprint, baseline_version, baseline_slice_count
    """
    system = load_curriculum(curriculum_slug)
    
    return {
        "curriculum_slug": curriculum_slug,
        "baseline_fingerprint": system.fingerprint(),
        "baseline_version": system.version,
        "baseline_slice_count": len(system.slices)
    }


def check_drift_before_run(
    curriculum_slug: str,
    baseline_fingerprint: str,
    artifact_dir: Path,
    fail_on_drift: bool = True
) -> bool:
    """
    Check for curriculum drift before starting an RFL run.
    
    Args:
        curriculum_slug: Curriculum system slug
        baseline_fingerprint: Expected curriculum fingerprint
        artifact_dir: Directory to write drift_report.json
        fail_on_drift: If True, raise CurriculumDriftError on drift (BLOCK mode)
                      If False, log warning and continue (WARN mode)
    
    Returns:
        True if no drift detected, False if drift detected (WARN mode only)
    
    Raises:
        CurriculumDriftError: If drift detected and fail_on_drift=True
    """
    signal_mode = "BLOCK" if fail_on_drift else "WARN"
    
    return check_curriculum_drift(
        curriculum_slug=curriculum_slug,
        baseline_fingerprint=baseline_fingerprint,
        artifact_dir=artifact_dir,
        signal_mode=signal_mode
    )


def extend_run_ledger_entry(
    ledger_entry_dict: Dict[str, Any],
    curriculum_slug: str
) -> Dict[str, Any]:
    """
    Extend a RunLedgerEntry dictionary with curriculum provenance fields.
    
    This function adds:
    - curriculum_slug: The curriculum system slug (e.g., "pl")
    - curriculum_fingerprint: The SHA-256 fingerprint of the curriculum
    
    Args:
        ledger_entry_dict: Dictionary representing a RunLedgerEntry
        curriculum_slug: Curriculum system slug
    
    Returns:
        Updated ledger_entry_dict with provenance fields
    """
    return stamp_run_ledger_with_fingerprint(ledger_entry_dict, curriculum_slug)


# Example integration pattern for RFLRunner
# 
# # DEMO-SCAFFOLD: This is illustrative code showing how to integrate into RFLRunner
# # Actual integration requires modifying rfl/runner.py
# 
# class RFLRunner:
#     def __init__(self, config: RFLConfig):
#         self.config = config
#         
#         # Initialize curriculum baseline
#         self.curriculum_baseline = initialize_curriculum_baseline("pl")
#         
#         logger.info(
#             f"Curriculum baseline: {self.curriculum_baseline['baseline_fingerprint'][:12]}..."
#         )
#     
#     def run_all(self):
#         # Check for drift before starting experiments
#         artifact_dir = Path(self.config.artifacts_dir)
#         
#         check_drift_before_run(
#             curriculum_slug=self.curriculum_baseline["curriculum_slug"],
#             baseline_fingerprint=self.curriculum_baseline["baseline_fingerprint"],
#             artifact_dir=artifact_dir,
#             fail_on_drift=True  # BLOCK mode
#         )
#         
#         # Continue with existing run logic...
#     
#     def _log_run_to_ledger(self, run_result: ExperimentResult, ...):
#         # Create base ledger entry
#         ledger_entry = {
#             "run_id": run_result.run_id,
#             "slice_name": run_result.slice_name,
#             # ... other fields
#         }
#         
#         # Extend with curriculum provenance
#         ledger_entry = extend_run_ledger_entry(
#             ledger_entry,
#             curriculum_slug=self.curriculum_baseline["curriculum_slug"]
#         )
#         
#         # Write to ledger...
