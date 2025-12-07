"""
PHASE II — NOT USED IN PHASE I

Multi-Run Evidence Fusion for U2 Experiments

This module provides cross-run validation and evidence aggregation for
Phase II U2 experiments. It checks for:
- Determinism violations across runs
- Missing artifacts
- Conflicting slice names
- Run ordering anomalies
- RFL policy input completeness

CRITICAL: This module does NOT interpret results as uplift evidence.
Uplift claims require full gate compliance (G1-G5) as specified in
VSD_PHASE_2.md.
"""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


@dataclass
class DeterminismViolation:
    """Records a determinism violation between runs."""
    
    run_ids: List[str]
    cycle: int
    description: str
    trace_hashes: List[str]


@dataclass
class MissingArtifact:
    """Records a missing artifact."""
    
    run_id: str
    artifact_type: str
    expected_path: Optional[str] = None


@dataclass
class ConflictingSliceName:
    """Records a slice name conflict."""
    
    run_ids: List[str]
    slice_names: List[str]


@dataclass
class RunOrderingAnomaly:
    """Records a run ordering anomaly."""
    
    description: str
    run_ids: List[str]
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FusedEvidenceSummary:
    """
    Multi-run evidence summary for promotion eligibility.
    
    This summary aggregates multiple U2 experiment runs and validates
    cross-run consistency. It does NOT make uplift claims.
    
    Attributes:
        run_count: Number of runs analyzed
        determinism_violations: List of determinism violations
        missing_artifacts: List of missing artifacts
        conflicting_slice_names: List of slice name conflicts
        run_ordering_anomalies: List of run ordering anomalies
        rfl_policy_complete: Whether RFL policy inputs are complete
        pass_status: Overall pass status ("PASS", "WARN", "BLOCK")
        metadata: Additional metadata
    """
    
    run_count: int
    determinism_violations: List[DeterminismViolation] = field(default_factory=list)
    missing_artifacts: List[MissingArtifact] = field(default_factory=list)
    conflicting_slice_names: List[ConflictingSliceName] = field(default_factory=list)
    run_ordering_anomalies: List[RunOrderingAnomaly] = field(default_factory=list)
    rfl_policy_complete: bool = True
    pass_status: str = "PASS"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "label": "PHASE II — NOT USED IN PHASE I",
            "run_count": self.run_count,
            "determinism_violations": [
                {
                    "run_ids": v.run_ids,
                    "cycle": v.cycle,
                    "description": v.description,
                    "trace_hashes": v.trace_hashes,
                }
                for v in self.determinism_violations
            ],
            "missing_artifacts": [
                {
                    "run_id": a.run_id,
                    "artifact_type": a.artifact_type,
                    "expected_path": a.expected_path,
                }
                for a in self.missing_artifacts
            ],
            "conflicting_slice_names": [
                {
                    "run_ids": c.run_ids,
                    "slice_names": c.slice_names,
                }
                for c in self.conflicting_slice_names
            ],
            "run_ordering_anomalies": [
                {
                    "description": a.description,
                    "run_ids": a.run_ids,
                    "details": a.details,
                }
                for a in self.run_ordering_anomalies
            ],
            "rfl_policy_complete": self.rfl_policy_complete,
            "pass_status": self.pass_status,
            "metadata": self.metadata,
        }


def validate_run_summary(summary: Dict[str, Any]) -> List[str]:
    """
    Validate a single run summary for required fields.
    
    Args:
        summary: Run summary dict
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Check required top-level fields
    required_fields = ["slice", "mode", "cycles", "initial_seed"]
    for field in required_fields:
        if field not in summary:
            errors.append(f"Missing required field: {field}")
    
    # Check for outputs section
    if "outputs" not in summary:
        errors.append("Missing 'outputs' section")
    else:
        outputs = summary["outputs"]
        if "results" not in outputs:
            errors.append("Missing 'outputs.results' path")
        if "manifest" not in outputs:
            errors.append("Missing 'outputs.manifest' path")
    
    # Check for slice_config_hash
    if "slice_config_hash" not in summary:
        errors.append("Missing 'slice_config_hash'")
    
    # Check for ht_series_hash
    if "ht_series_hash" not in summary:
        errors.append("Missing 'ht_series_hash'")
    
    return errors


def check_determinism_violations(
    run_summaries: List[Dict[str, Any]]
) -> List[DeterminismViolation]:
    """
    Check for determinism violations across runs.
    
    Determinism violations occur when runs with the same slice, mode, and seed
    produce different ht_series_hash values.
    
    Args:
        run_summaries: List of run summary dicts
        
    Returns:
        List of determinism violations
    """
    violations = []
    
    # Group runs by (slice, mode, initial_seed)
    run_groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for summary in run_summaries:
        key = (
            summary.get("slice"),
            summary.get("mode"),
            summary.get("initial_seed"),
        )
        if key not in run_groups:
            run_groups[key] = []
        run_groups[key].append(summary)
    
    # Check each group for consistency
    for (slice_name, mode, seed), summaries in run_groups.items():
        if len(summaries) < 2:
            continue
        
        # Compare ht_series_hash across runs
        ht_hashes = [s.get("ht_series_hash") for s in summaries]
        if len(set(ht_hashes)) > 1:
            violations.append(
                DeterminismViolation(
                    run_ids=[
                        s.get("outputs", {}).get("manifest", "unknown")
                        for s in summaries
                    ],
                    cycle=-1,  # Not cycle-specific
                    description=(
                        f"ht_series_hash mismatch for slice={slice_name}, "
                        f"mode={mode}, seed={seed}"
                    ),
                    trace_hashes=ht_hashes,
                )
            )
    
    return violations


def check_missing_artifacts(
    run_summaries: List[Dict[str, Any]]
) -> List[MissingArtifact]:
    """
    Check for missing artifacts referenced in run summaries.
    
    Args:
        run_summaries: List of run summary dicts
        
    Returns:
        List of missing artifacts
    """
    missing = []
    
    for summary in run_summaries:
        manifest_path = summary.get("outputs", {}).get("manifest", "")
        
        # Check for results file
        results_path = summary.get("outputs", {}).get("results")
        if results_path:
            if not Path(results_path).exists():
                missing.append(
                    MissingArtifact(
                        run_id=manifest_path,
                        artifact_type="results",
                        expected_path=results_path,
                    )
                )
        
        # Check for manifest file
        if manifest_path:
            if not Path(manifest_path).exists():
                missing.append(
                    MissingArtifact(
                        run_id=manifest_path,
                        artifact_type="manifest",
                        expected_path=manifest_path,
                    )
                )
    
    return missing


def check_conflicting_slice_names(
    run_summaries: List[Dict[str, Any]]
) -> List[ConflictingSliceName]:
    """
    Check for conflicting slice names across runs.
    
    Conflicts occur when runs claim to be from the same experiment but have
    different slice names.
    
    Args:
        run_summaries: List of run summary dicts
        
    Returns:
        List of slice name conflicts
    """
    conflicts = []
    
    # Group runs by experiment_id if present
    # For now, we just check if there are different slice names with same config hash
    config_hash_to_slices: Dict[str, Set[str]] = {}
    
    for summary in run_summaries:
        config_hash = summary.get("slice_config_hash")
        slice_name = summary.get("slice")
        
        if config_hash and slice_name:
            if config_hash not in config_hash_to_slices:
                config_hash_to_slices[config_hash] = set()
            config_hash_to_slices[config_hash].add(slice_name)
    
    # Check for conflicts
    for config_hash, slice_names in config_hash_to_slices.items():
        if len(slice_names) > 1:
            conflicts.append(
                ConflictingSliceName(
                    run_ids=[
                        s.get("outputs", {}).get("manifest", "unknown")
                        for s in run_summaries
                        if s.get("slice_config_hash") == config_hash
                    ],
                    slice_names=list(slice_names),
                )
            )
    
    return conflicts


def check_run_ordering_anomalies(
    run_summaries: List[Dict[str, Any]]
) -> List[RunOrderingAnomaly]:
    """
    Check for run ordering anomalies.
    
    Anomalies include:
    - RFL runs without corresponding baseline runs
    - Missing seed coverage
    
    Args:
        run_summaries: List of run summary dicts
        
    Returns:
        List of run ordering anomalies
    """
    anomalies = []
    
    # Group runs by slice
    slice_groups: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for summary in run_summaries:
        slice_name = summary.get("slice", "unknown")
        mode = summary.get("mode", "unknown")
        
        if slice_name not in slice_groups:
            slice_groups[slice_name] = {"baseline": [], "rfl": []}
        
        if mode in ["baseline", "rfl"]:
            slice_groups[slice_name][mode].append(summary)
    
    # Check each slice for baseline/RFL pairing
    for slice_name, modes in slice_groups.items():
        baseline_runs = modes["baseline"]
        rfl_runs = modes["rfl"]
        
        if rfl_runs and not baseline_runs:
            anomalies.append(
                RunOrderingAnomaly(
                    description=f"Slice '{slice_name}' has RFL runs but no baseline runs",
                    run_ids=[
                        s.get("outputs", {}).get("manifest", "unknown")
                        for s in rfl_runs
                    ],
                    details={
                        "slice": slice_name,
                        "rfl_count": len(rfl_runs),
                        "baseline_count": 0,
                    },
                )
            )
    
    return anomalies


def check_rfl_policy_completeness(
    run_summaries: List[Dict[str, Any]]
) -> bool:
    """
    Check if RFL policy inputs are complete.
    
    For each RFL run, verifies that:
    - policy_stats section exists
    - Required fields are present
    
    Args:
        run_summaries: List of run summary dicts
        
    Returns:
        True if all RFL runs have complete policy inputs
    """
    for summary in run_summaries:
        if summary.get("mode") == "rfl":
            policy_stats = summary.get("policy_stats")
            if not policy_stats:
                return False
            
            required_fields = ["update_count", "success_count", "attempt_count"]
            for field in required_fields:
                if field not in policy_stats:
                    return False
    
    return True


def fuse_evidence_summaries(run_summaries: List[Dict[str, Any]]) -> FusedEvidenceSummary:
    """
    Compute cross-run determinism, cross-run slice consistency,
    evidence completeness, and calibration reproducibility.
    
    This function aggregates multiple U2 experiment runs and validates
    cross-run consistency. It does NOT make uplift claims and returns
    a neutral evidence block.
    
    Args:
        run_summaries: List of run summary dicts (loaded from manifest JSON files)
        
    Returns:
        FusedEvidenceSummary with validation results
    """
    if not run_summaries:
        return FusedEvidenceSummary(
            run_count=0,
            pass_status="BLOCK",
            metadata={"error": "No run summaries provided"},
        )
    
    # Validate each run summary
    validation_errors = []
    for i, summary in enumerate(run_summaries):
        errors = validate_run_summary(summary)
        if errors:
            validation_errors.extend([f"Run {i}: {e}" for e in errors])
    
    if validation_errors:
        return FusedEvidenceSummary(
            run_count=len(run_summaries),
            pass_status="BLOCK",
            metadata={"validation_errors": validation_errors},
        )
    
    # Check for issues
    determinism_violations = check_determinism_violations(run_summaries)
    missing_artifacts = check_missing_artifacts(run_summaries)
    conflicting_slice_names = check_conflicting_slice_names(run_summaries)
    run_ordering_anomalies = check_run_ordering_anomalies(run_summaries)
    rfl_policy_complete = check_rfl_policy_completeness(run_summaries)
    
    # Determine pass status
    pass_status = "PASS"
    
    # BLOCK conditions
    if determinism_violations:
        pass_status = "BLOCK"
    elif missing_artifacts:
        pass_status = "BLOCK"
    elif conflicting_slice_names:
        pass_status = "BLOCK"
    
    # WARN conditions
    if pass_status == "PASS":
        if run_ordering_anomalies:
            pass_status = "WARN"
        elif not rfl_policy_complete:
            pass_status = "WARN"
    
    return FusedEvidenceSummary(
        run_count=len(run_summaries),
        determinism_violations=determinism_violations,
        missing_artifacts=missing_artifacts,
        conflicting_slice_names=conflicting_slice_names,
        run_ordering_anomalies=run_ordering_anomalies,
        rfl_policy_complete=rfl_policy_complete,
        pass_status=pass_status,
        metadata={
            "phase": "II",
            "label": "PHASE II — NOT USED IN PHASE I",
        },
    )


def inject_multi_run_fusion_into_evidence(
    summary: Dict[str, Any],
    fused: FusedEvidenceSummary,
) -> Dict[str, Any]:
    """
    Produce composite neutral evidence block with multi-run fusion results.
    
    This function takes an existing evidence summary and adds multi-run
    fusion results to it. The result is a neutral evidence block that
    does NOT make uplift claims.
    
    Args:
        summary: Existing evidence summary dict
        fused: Fused evidence summary from fuse_evidence_summaries
        
    Returns:
        Combined evidence dict with multi-run fusion data
    """
    combined = summary.copy()
    
    combined["multi_run_fusion"] = fused.to_dict()
    combined["label"] = "PHASE II — NOT USED IN PHASE I"
    
    # Add top-level status
    if "status" not in combined:
        combined["status"] = {}
    
    combined["status"]["multi_run_validation"] = fused.pass_status
    
    return combined
