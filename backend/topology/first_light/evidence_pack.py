"""
Evidence Pack Builder for Phase X Whitepaper Artifacts

This module implements the Evidence Pack generator that bundles P3/P4 artifacts
into a cryptographically verifiable package for whitepaper submission and
compliance demonstration.

SHADOW MODE CONTRACT:
- All governance checks are ADVISORY ONLY
- No enforcement or blocking occurs
- All outputs are observational

See docs/system_law/Evidence_Pack_Spec_PhaseX.md for specification.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from scripts.first_light_proof_hash_snapshot import generate_snapshot

# CTRPK detection - optional
try:
    from backend.topology.first_light.ctrpk_detection import CTRPKReference
    HAS_CTRPK = True
except ImportError:
    CTRPKReference = None  # type: ignore
    HAS_CTRPK = False

# Noise vs Reality integration - optional
try:
    from backend.topology.first_light.noise_vs_reality import (
        build_from_harness_and_divergence,
        validate_noise_vs_reality_summary,
        SCHEMA_VERSION as NOISE_VS_REALITY_SCHEMA_VERSION,
    )
    HAS_NOISE_VS_REALITY = True
except ImportError:
    HAS_NOISE_VS_REALITY = False
    NOISE_VS_REALITY_SCHEMA_VERSION = "noise-vs-reality/1.0.0"

# Schema validation - optional dependency
try:
    import jsonschema
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False


# =============================================================================
# Constants
# =============================================================================

EVIDENCE_PACK_VERSION = "1.0.0"
SCHEMA_VERSION = "1.0.0"

# Required artifacts for each category
REQUIRED_P3_ARTIFACTS = [
    "synthetic_raw.jsonl",
    "stability_report.json",
    "red_flag_matrix.json",
    "metrics_windows.json",
]

REQUIRED_P4_ARTIFACTS = [
    "divergence_log.jsonl",
    "twin_trajectory.jsonl",
    "calibration_report.json",
]

OPTIONAL_P3_ARTIFACTS = [
    "tda_metrics.json",
]

OPTIONAL_P4_ARTIFACTS = [
    "tda_metrics.json",
    "divergence_distribution.json",
]

# P3 Pathology Annotation (SHADOW-ONLY metadata)
# Optional pathology annotation artifact for P3 First-Light stress-test runs.
P3_PATHOLOGY_ARTIFACT = "p3_pathology.json"
P3_PATHOLOGY_SCHEMA = "evidence_pack/p3_pathology.schema.json"

# P5 Real Telemetry (RESERVED - Phase X Shadow Mode)
# These artifacts are reserved for future P5 real telemetry integration.
# Currently: detection and schema validation only, no enforcement.
P5_DIVERGENCE_ARTIFACT = "p5_divergence_real.json"
P5_DIVERGENCE_SCHEMA = "p5/p5_divergence_real.schema.json"

# P5 Topology Auditor Report (Phase X Shadow Mode)
# Provides topology/bundle validation for P5 real telemetry runs.
P5_TOPOLOGY_AUDITOR_REPORT_ARTIFACT = "p5_topology_auditor_report.json"
P5_TOPOLOGY_AUDITOR_REPORT_SCHEMA = "topology/p5_topology_auditor_report.schema.json"

# P5 Replay Safety (Phase X Shadow Mode)
# Replay logs artifact for P5 real-telemetry replay safety validation.
P5_REPLAY_LOGS_ARTIFACT = "p5_replay_logs.jsonl"
P5_REPLAY_LOGS_DIR = "p5_replay_logs"

# P5 Structural Drill (Phase X Shadow Mode — CAL-EXP-3 Optional Stress Test)
# Structural drill artifacts for STRUCTURAL_BREAK event simulation.
# See docs/system_law/P5_Structural_Drill_Package.md
STRUCTURAL_DRILL_ARTIFACT = "structural_drill_artifact.json"
STRUCTURAL_DRILL_MANIFEST = "structural_drill_manifest.json"
STRUCTURAL_DRILL_SCHEMA = "structural/structural_drill_artifact.schema.json"

# P5 Pattern Tags (Phase X Shadow Mode)
# TDA pattern classification tags for GGFL integration.
# See docs/system_law/GGFL_P5_Pattern_Test_Plan.md
P5_PATTERN_TAGS_ARTIFACT = "p5_pattern_tags.json"
P5_PATTERN_TAGS_SCHEMA = "p5/p5_pattern_tags.schema.json"
STRUCTURAL_DRILL_DIR = "p5_structural_drill"

# Budget Calibration (Phase X Shadow Mode)
# Budget calibration artifacts for FP/FN analysis per Budget_PhaseX_Doctrine.md Section 7.3
BUDGET_CALIBRATION_SUMMARY_ARTIFACT = "budget_calibration_summary.json"
BUDGET_CALIBRATION_LOG_ARTIFACT = "budget_calibration_log.jsonl"
BUDGET_CALIBRATION_SUBDIR = "calibration"

# P5 Divergence Diagnostic (Phase X Shadow Mode)
# P5 diagnostic panel output from P5DivergenceInterpreter.
# See docs/system_law/P5_Divergence_Diagnostic_Panel_Spec.md
P5_DIVERGENCE_DIAGNOSTIC_ARTIFACT = "p5_divergence_diagnostic.json"
P5_DIVERGENCE_DIAGNOSTIC_SCHEMA = "p5/p5_divergence_diagnostic.schema.json"

# NCI P5 Narrative Consistency Index (Phase X Shadow Mode)
# NCI P5 artifacts for documentation consistency evaluation.
# See docs/system_law/NCI_PhaseX_Spec.md Section 11
NCI_P5_RESULT_ARTIFACT = "nci_p5_result.json"
NCI_P5_SIGNAL_ARTIFACT = "nci_p5_signal.json"
NCI_P5_SUBDIR = "calibration"  # Also search in calibration/ subdirectory

REQUIRED_VISUALIZATIONS = [
    "delta_p_trendline.svg",
    "rsi_trajectory.svg",
    "omega_occupancy.svg",
]

OPTIONAL_VISUALIZATIONS = [
    "twin_vs_reality.svg",
    "red_flag_heatmap.svg",
    "tda_dashboard.svg",
]

# Schema file mapping
ARTIFACT_SCHEMA_MAP = {
    "synthetic_raw.jsonl": "first_light/first_light_synthetic_raw.schema.json",
    "stability_report.json": "first_light/first_light_stability_report.schema.json",
    "red_flag_matrix.json": "first_light/first_light_red_flag_matrix.schema.json",
    "metrics_windows.json": "first_light/first_light_metrics_windows.schema.json",
    "divergence_log.jsonl": "phase_x_p4/p4_divergence_log.schema.json",
    "twin_trajectory.jsonl": "phase_x_p4/p4_twin_trajectory.schema.json",
    "calibration_report.json": "phase_x_p4/p4_calibration_report.schema.json",
    "divergence_distribution.json": "phase_x_p4/p4_divergence_distribution.schema.json",
    "tda_metrics.json": "evidence_pack/tda_metrics.schema.json",
    # P3 Pathology Annotation (SHADOW-ONLY)
    P3_PATHOLOGY_ARTIFACT: P3_PATHOLOGY_SCHEMA,
    # P5 Real Telemetry Divergence (RESERVED)
    P5_DIVERGENCE_ARTIFACT: P5_DIVERGENCE_SCHEMA,
}

PROOF_SNAPSHOT_ENV_FLAG = "FIRST_LIGHT_INCLUDE_PROOF_SNAPSHOT"
PROOF_LOG_ENV_VAR = "FIRST_LIGHT_PROOF_LOG"
DEFAULT_PROOF_SNAPSHOT_REL_PATH = Path("compliance") / "proof_log_snapshot.json"
DEFAULT_PROOF_LOG_RELATIVE_CANDIDATES = [
    Path("proofs.jsonl"),
    Path("proof_log.jsonl"),
    Path("proof_logs.jsonl"),
    Path("p3_synthetic") / "proofs.jsonl",
    Path("p3_synthetic") / "proof_log.jsonl",
    Path("p4_shadow") / "proofs.jsonl",
    Path("p4_shadow") / "proof_log.jsonl",
]


def _truthy_env(value: Optional[str]) -> bool:
    if not value:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class ArtifactInfo:
    """Information about a single artifact in the evidence pack."""
    path: str
    sha256: str
    size_bytes: int
    category: str
    schema_ref: Optional[str] = None
    format: str = "json"
    description: str = ""
    required: bool = True
    validation_passed: bool = True
    validation_errors: List[str] = field(default_factory=list)


@dataclass
class CompletenessCheck:
    """Result of completeness validation."""
    p3_artifacts: Dict[str, bool] = field(default_factory=dict)
    p4_artifacts: Dict[str, bool] = field(default_factory=dict)
    visualizations: Dict[str, bool] = field(default_factory=dict)
    compliance: Dict[str, bool] = field(default_factory=dict)
    all_required_present: bool = False
    missing_artifacts: List[str] = field(default_factory=list)


@dataclass
class GovernanceAdvisory:
    """
    SHADOW MODE: Advisory governance check result.

    These advisories are for analysis only and do not block pack generation.
    """
    check_name: str
    passed: bool
    severity: str  # INFO, WARN, CRITICAL
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StatusReference:
    """Reference to first_light_status.json if present."""
    path: str
    sha256: str
    schema_version: Optional[str] = None
    shadow_mode_ok: Optional[bool] = None


@dataclass
class P5ReplayGovernanceReference:
    """
    Reference to P5 replay governance signal in evidence pack.

    SHADOW MODE CONTRACT:
    - This is observational only, does NOT gate any operations
    - Detection and validation occur but do not affect pack generation
    - Provides P5 real-telemetry replay safety signal for auditors

    See docs/system_law/Replay_Safety_P5_Engineering_Plan.md
    """
    path: str
    sha256: str
    schema_version: str = "1.0.0"
    status: Optional[str] = None  # "ok" | "warn" | "block"
    determinism_rate: Optional[float] = None
    determinism_band: Optional[str] = None  # "GREEN" | "YELLOW" | "RED"
    p5_grade: bool = False
    telemetry_source: Optional[str] = None  # "real" | "shadow" | "synthetic"
    production_run_id: Optional[str] = None
    hash_match_count: int = 0
    hash_mismatch_count: int = 0
    replay_latency_ms: Optional[float] = None
    # Robustness fields (v1.1.0)
    schema_ok: bool = True  # False if schema version mismatch or missing P5 fields
    advisory_warnings: List[str] = field(default_factory=list)
    skipped_gz_count: int = 0  # Number of .gz files skipped (gzip not supported)
    malformed_line_count: int = 0  # Number of malformed lines skipped


@dataclass
class P5DivergenceReference:
    """
    Reference to P5 real telemetry divergence report if present.

    SHADOW MODE: This is observational only. Detection and validation
    occur but do not affect pack generation.

    See docs/system_law/schemas/p5/p5_divergence_real.schema.json
    """
    path: str
    sha256: str
    schema_version: Optional[str] = None
    telemetry_source: Optional[str] = None  # "real" or "mock"
    validation_status: Optional[str] = None  # VALIDATED_REAL, SUSPECTED_MOCK, UNVALIDATED
    divergence_rate: Optional[float] = None
    mode: Optional[str] = None  # Must be "SHADOW" for Phase X
    schema_valid: bool = False
    validation_errors: List[str] = field(default_factory=list)


@dataclass
class P5DiagnosticReference:
    """
    Reference to P5 divergence diagnostic panel output if present.

    SHADOW MODE: This is observational only. Detection and validation
    occur but do not affect pack generation.

    See docs/system_law/P5_Divergence_Diagnostic_Panel_Spec.md
    """
    path: str
    sha256: str
    schema_version: Optional[str] = None
    root_cause_hypothesis: Optional[str] = None
    action: Optional[str] = None
    headline: Optional[str] = None
    cycle: Optional[int] = None
    run_id: Optional[str] = None


@dataclass
class P5IdentityPreflightReference:
    """
    Reference to p5_identity_preflight.json artifact if present.

    SHADOW MODE CONTRACT:
    - This is observational only, does NOT gate any operations
    - Detection and validation occur but do not affect pack generation
    - Provides P5 identity pre-flight check for slice configuration alignment

    See docs/system_law/Identity_Preflight_Precedence_Law.md
    """
    path: str
    sha256: str
    schema_version: Optional[str] = None
    status: Optional[str] = None  # "OK" | "INVESTIGATE" | "BLOCK"
    fingerprint_match: Optional[bool] = None
    mode: Optional[str] = None  # Must be "SHADOW" for Phase X


@dataclass
class P5TopologyAuditorReference:
    """
    Reference to P5 topology auditor report if present.

    SHADOW MODE: This is observational only. Detection and validation
    occur but do not affect pack generation.

    PARTIAL JSON EXTRACTION CONTRACT:
    - If the report JSON is invalid (malformed), schema_ok=False is set
    - If schema_ok=False, scenario is NOT fabricated (remains None)
    - Advisory warning is returned via validation_passed=False
    - sha256 is still computed for the raw file bytes
    - This ensures no false scenario claims from corrupted data

    See docs/system_law/Topology_Bundle_PhaseX_Requirements.md Section 10.
    """
    path: str
    sha256: str
    schema_version: Optional[str] = None
    scenario: Optional[str] = None  # MOCK_BASELINE, HEALTHY, MISMATCH, XCOR_ANOMALY
    scenario_confidence: Optional[float] = None
    joint_status: Optional[str] = None  # ALIGNED, TENSION, DIVERGENT
    shadow_mode_invariant_ok: bool = True
    validation_passed: bool = False
    mode: Optional[str] = None  # Must be "SHADOW" for Phase X
    schema_ok: bool = True  # False if JSON parsing failed or schema invalid
    advisory_warning: Optional[str] = None  # Warning message for partial extraction


@dataclass
class P5PatternTagsReference:
    """
    Reference to P5 pattern tags artifact if present.

    SHADOW MODE CONTRACT:
    - Pattern tags are observational only
    - Detection does not affect pack generation
    - Provides TDA pattern classification for GGFL integration

    See docs/system_law/GGFL_P5_Pattern_Test_Plan.md
    """
    path: str
    sha256: str
    schema_version: Optional[str] = None
    final_pattern: Optional[str] = None  # DRIFT, NOISE_AMPLIFICATION, etc.
    final_streak: int = 0
    cycles_analyzed: int = 0
    recalibration_triggered: bool = False
    mode: Optional[str] = None  # Must be "SHADOW" for Phase X
    shadow_mode_invariants_ok: bool = True


@dataclass
class EvidencePackResult:
    """Result of evidence pack generation."""
    success: bool
    bundle_id: str
    manifest_path: Optional[str]
    merkle_root: Optional[str]
    artifacts: List[ArtifactInfo]
    completeness: CompletenessCheck
    governance_advisories: List[GovernanceAdvisory]
    errors: List[str]
    warnings: List[str]
    status_reference: Optional[StatusReference] = None
    p5_divergence_reference: Optional[P5DivergenceReference] = None
    p5_replay_governance_reference: Optional[P5ReplayGovernanceReference] = None
    p5_topology_auditor_reference: Optional[P5TopologyAuditorReference] = None
    p5_pattern_tags_reference: Optional[P5PatternTagsReference] = None
    p5_diagnostic_reference: Optional[P5DiagnosticReference] = None
    structural_drill_reference: Optional["StructuralDrillReference"] = None
    ctrpk_reference: Optional["CTRPKReference"] = None
    nci_p5_reference: Optional["NciP5Reference"] = None
    p5_identity_preflight_reference: Optional[P5IdentityPreflightReference] = None


@dataclass
class StructuralDrillReference:
    """
    Reference to P5 structural drill artifact if present.

    SHADOW MODE: This is observational only. Drill is an optional
    stress diagnostic for CAL-EXP-3 regime-change testing.

    See docs/system_law/P5_Structural_Drill_Package.md
    """
    path: str
    sha256: str
    drill_id: Optional[str] = None
    scenario_id: Optional[str] = None
    drill_success: bool = False
    max_streak: int = 0
    break_events: List[int] = field(default_factory=list)
    pattern_counts: Dict[str, int] = field(default_factory=dict)
    mode: str = "SHADOW"
    schema_valid: bool = False
    validation_errors: List[str] = field(default_factory=list)


@dataclass
class CTRPKReference:
    """
    Reference to CTRPK (Curriculum Transition Requests Per 1K Cycles) compact block.

    SHADOW MODE CONTRACT:
    - CTRPK is purely observational
    - Detection does not affect pack generation success/failure
    - Provides curriculum churn metrics for calibration era monitoring

    See docs/system_law/Curriculum_PhaseX_Invariants.md Section 12.
    """
    path: str
    sha256: str
    value: float
    status: str  # "OK" | "WARN" | "BLOCK"
    trend: str  # "IMPROVING" | "STABLE" | "DEGRADING"
    window_cycles: int
    transition_requests: int
    mode: str = "SHADOW"


@dataclass
class BudgetCalibrationReference:
    """
    Reference to Budget Calibration artifact if present.

    SHADOW MODE CONTRACT:
    - Budget calibration is purely observational
    - Detection does not affect pack generation success/failure
    - Provides FP/FN calibration metrics for budget drift analysis
    - enablement_recommendation is advisory only, not gating

    See docs/system_law/Budget_PhaseX_Doctrine.md Section 7.3.
    """
    path: str
    sha256: str
    schema_version: Optional[str] = None
    enablement_recommendation: Optional[str] = None  # "ENABLE" | "DEFER"
    fp_rate: Optional[float] = None
    fn_rate: Optional[float] = None
    overall_pass: Optional[bool] = None
    log_path: Optional[str] = None  # Optional log JSONL path
    log_sha256: Optional[str] = None
    mode: str = "SHADOW"


@dataclass
class NciP5Reference:
    """
    Reference to NCI P5 (Narrative Consistency Index) artifact if present.

    SHADOW MODE CONTRACT:
    - NCI P5 is purely observational
    - Detection does not affect pack generation success/failure
    - Provides documentation consistency metrics for governance signals
    - Attaches at manifest.signals.nci_p5

    See docs/system_law/NCI_PhaseX_Spec.md Section 11.
    """
    path: str
    sha256: str
    schema_version: str = "1.0.0"
    mode: Optional[str] = None  # "DOC_ONLY" | "TELEMETRY_CHECKED" | "FULLY_BOUND"
    global_nci: Optional[float] = None
    confidence: Optional[float] = None
    slo_status: Optional[str] = None  # "OK" | "WARN" | "BREACH"
    recommendation: Optional[str] = None  # "NONE" | "WARNING" | "REVIEW"
    tcl_aligned: bool = True
    sic_aligned: bool = True
    tcl_violation_count: int = 0
    sic_violation_count: int = 0
    warning_count: int = 0
    shadow_mode: bool = True
    # Artifact provenance: detection_path indicates where artifact was found
    detection_path: str = "root"  # "root" | "calibration"
    # Extraction source: how the NCI data was obtained
    extraction_source: str = "MISSING"  # "MANIFEST_SIGNAL" | "MANIFEST_RESULT" | "EVIDENCE_JSON" | "MISSING"
    # Optional: path to full result file if signal file was used
    result_path: Optional[str] = None
    result_sha256: Optional[str] = None


# =============================================================================
# Merkle Tree Implementation
# =============================================================================

def compute_sha256(data: bytes) -> str:
    """Compute SHA-256 hash of data."""
    return hashlib.sha256(data).hexdigest()


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_merkle_root(hashes: List[str]) -> str:
    """
    Compute Merkle root from a list of leaf hashes.

    Uses SHA-256 binary Merkle tree construction:
    - Leaves are sorted lexicographically for determinism
    - Parent = SHA256(left || right)
    - Odd nodes are promoted unchanged
    """
    if not hashes:
        # Empty tree - hash of empty string
        return compute_sha256(b"")

    # Sort for determinism
    leaves = sorted(hashes)

    # Convert hex strings to bytes
    current_level = [bytes.fromhex(h) for h in leaves]

    while len(current_level) > 1:
        next_level = []
        for i in range(0, len(current_level), 2):
            if i + 1 < len(current_level):
                # Hash pair
                combined = current_level[i] + current_level[i + 1]
                next_level.append(hashlib.sha256(combined).digest())
            else:
                # Odd node - promote unchanged
                next_level.append(current_level[i])
        current_level = next_level

    return current_level[0].hex()


# =============================================================================
# Schema Validation
# =============================================================================

def load_schema(schema_path: Path) -> Optional[Dict[str, Any]]:
    """Load a JSON schema from file."""
    if not schema_path.exists():
        return None
    try:
        with open(schema_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def validate_json_against_schema(
    data: Dict[str, Any],
    schema: Dict[str, Any],
) -> Tuple[bool, List[str]]:
    """
    Validate JSON data against a JSON schema.

    Returns (is_valid, list_of_errors).
    """
    if not HAS_JSONSCHEMA:
        # Schema validation not available - skip with warning
        return True, ["jsonschema not installed - validation skipped"]

    errors = []
    try:
        jsonschema.validate(instance=data, schema=schema)
        return True, []
    except jsonschema.ValidationError as e:
        errors.append(f"Validation error: {e.message}")
        return False, errors
    except jsonschema.SchemaError as e:
        errors.append(f"Schema error: {e.message}")
        return False, errors


def validate_jsonl_against_schema(
    file_path: Path,
    schema: Dict[str, Any],
    max_lines: int = 100,
) -> Tuple[bool, List[str]]:
    """
    Validate JSONL file against schema (samples first N lines).

    Returns (is_valid, list_of_errors).
    """
    if not HAS_JSONSCHEMA:
        return True, ["jsonschema not installed - validation skipped"]

    errors = []
    lines_checked = 0

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    is_valid, record_errors = validate_json_against_schema(record, schema)
                    if not is_valid:
                        errors.extend([f"Line {i+1}: {e}" for e in record_errors])
                    lines_checked += 1
                except json.JSONDecodeError as e:
                    errors.append(f"Line {i+1}: JSON decode error: {e}")
    except OSError as e:
        errors.append(f"File read error: {e}")
        return False, errors

    if errors:
        return False, errors
    return True, [f"Validated {lines_checked} lines"]


# =============================================================================
# Governance Checks (SHADOW MODE - Advisory Only)
# =============================================================================

def check_shadow_mode_compliance(artifacts: List[ArtifactInfo], run_dir: Path) -> GovernanceAdvisory:
    """
    SHADOW MODE: Check that all artifacts contain mode: SHADOW markers.

    Advisory only - does not block pack generation.
    """
    violations = []

    for artifact in artifacts:
        if artifact.format not in ("json", "jsonl"):
            continue

        file_path = run_dir / artifact.path
        if not file_path.exists():
            continue

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read(4096)  # Check first 4KB
                # Look for mode field
                if '"mode"' in content and '"SHADOW"' not in content:
                    violations.append(artifact.path)
        except OSError:
            continue

    if violations:
        return GovernanceAdvisory(
            check_name="shadow_mode_compliance",
            passed=False,
            severity="WARN",
            message=f"SHADOW MODE marker not found in {len(violations)} artifacts",
            details={"artifacts": violations},
        )

    return GovernanceAdvisory(
        check_name="shadow_mode_compliance",
        passed=True,
        severity="INFO",
        message="All artifacts comply with SHADOW MODE contract",
        details={},
    )


def check_schema_compliance(artifacts: List[ArtifactInfo]) -> GovernanceAdvisory:
    """
    SHADOW MODE: Check schema validation results.

    Advisory only - does not block pack generation.
    """
    failed = [a for a in artifacts if not a.validation_passed]

    if failed:
        return GovernanceAdvisory(
            check_name="schema_compliance",
            passed=False,
            severity="WARN",
            message=f"{len(failed)} artifacts failed schema validation",
            details={
                "failed_artifacts": [
                    {"path": a.path, "errors": a.validation_errors}
                    for a in failed
                ]
            },
        )

    return GovernanceAdvisory(
        check_name="schema_compliance",
        passed=True,
        severity="INFO",
        message="All artifacts passed schema validation",
        details={},
    )


def check_completeness(completeness: CompletenessCheck) -> GovernanceAdvisory:
    """
    SHADOW MODE: Check artifact completeness.

    Advisory only - does not block pack generation.
    """
    if completeness.all_required_present:
        return GovernanceAdvisory(
            check_name="completeness",
            passed=True,
            severity="INFO",
            message="All required artifacts present",
            details={},
        )

    return GovernanceAdvisory(
        check_name="completeness",
        passed=False,
        severity="CRITICAL",
        message=f"Missing {len(completeness.missing_artifacts)} required artifacts",
        details={"missing": completeness.missing_artifacts},
    )


def check_stability_thresholds(run_dir: Path) -> GovernanceAdvisory:
    """
    SHADOW MODE: Check stability metrics against thresholds.

    Advisory only - does not block pack generation.
    """
    stability_path = run_dir / "stability_report.json"
    if not stability_path.exists():
        return GovernanceAdvisory(
            check_name="stability_thresholds",
            passed=False,
            severity="WARN",
            message="Stability report not found",
            details={},
        )

    try:
        with open(stability_path, "r", encoding="utf-8") as f:
            report = json.load(f)
    except (json.JSONDecodeError, OSError):
        return GovernanceAdvisory(
            check_name="stability_thresholds",
            passed=False,
            severity="WARN",
            message="Failed to parse stability report",
            details={},
        )

    # Check criteria evaluation
    criteria = report.get("criteria_evaluation", {})
    all_passed = criteria.get("all_passed", False)

    if all_passed:
        return GovernanceAdvisory(
            check_name="stability_thresholds",
            passed=True,
            severity="INFO",
            message="All stability criteria passed",
            details=criteria,
        )

    # Extract failed criteria
    failed_criteria = [
        c for c in criteria.get("criteria", [])
        if not c.get("passed", True)
    ]

    return GovernanceAdvisory(
        check_name="stability_thresholds",
        passed=False,
        severity="WARN",
        message=f"{len(failed_criteria)} stability criteria not met",
        details={"failed_criteria": failed_criteria},
    )


def run_governance_checks(
    artifacts: List[ArtifactInfo],
    completeness: CompletenessCheck,
    run_dir: Path,
) -> List[GovernanceAdvisory]:
    """
    Run all SHADOW MODE governance checks.

    All checks are advisory only - they do not block pack generation.
    """
    advisories = []

    advisories.append(check_shadow_mode_compliance(artifacts, run_dir))
    advisories.append(check_schema_compliance(artifacts))
    advisories.append(check_completeness(completeness))
    advisories.append(check_stability_thresholds(run_dir))

    return advisories


def detect_status_file(run_dir: Path) -> Optional[StatusReference]:
    """
    Detect and extract reference to first_light_status.json if present.

    The status file provides a cross-link between the Evidence Pack and
    the First Light run status summary. External verifiers should treat
    both artifacts as a paired set.

    Args:
        run_dir: Path to the run directory or evidence pack directory.

    Returns:
        StatusReference if first_light_status.json exists, None otherwise.
    """
    status_path = run_dir / "first_light_status.json"
    if not status_path.exists():
        return None

    try:
        # Compute hash
        file_hash = compute_file_hash(status_path)

        # Extract key fields from status
        with open(status_path, "r", encoding="utf-8") as f:
            status_data = json.load(f)

        schema_version = status_data.get("schema_version")
        shadow_mode_ok = status_data.get("shadow_mode_ok")

        return StatusReference(
            path="first_light_status.json",
            sha256=file_hash,
            schema_version=schema_version,
            shadow_mode_ok=shadow_mode_ok,
        )
    except (json.JSONDecodeError, OSError):
        # Status file exists but couldn't be parsed - still reference it
        try:
            file_hash = compute_file_hash(status_path)
            return StatusReference(
                path="first_light_status.json",
                sha256=file_hash,
            )
        except OSError:
            return None


def detect_p5_identity_preflight_file(run_dir: Path) -> Optional[P5IdentityPreflightReference]:
    """
    Detect and extract reference to p5_identity_preflight.json if present.

    SHADOW MODE CONTRACT:
    - This function is purely observational
    - Detection does not affect pack generation success/failure
    - Provides identity pre-flight reference for manifest attachment

    Per Identity_Preflight_Precedence_Law.md, the dedicated artifact file
    takes precedence over legacy identity_preflight.json and run_config.json.

    Args:
        run_dir: Path to the run directory or evidence pack directory.

    Returns:
        P5IdentityPreflightReference if p5_identity_preflight.json exists, None otherwise.
    """
    preflight_path = run_dir / "p5_identity_preflight.json"
    if not preflight_path.exists():
        return None

    try:
        # Compute hash
        file_hash = compute_file_hash(preflight_path)

        # Extract key fields from artifact
        with open(preflight_path, "r", encoding="utf-8") as f:
            preflight_data = json.load(f)

        schema_version = preflight_data.get("schema_version")
        status = preflight_data.get("status")
        fingerprint_match = preflight_data.get("fingerprint_match")
        mode = preflight_data.get("mode")

        return P5IdentityPreflightReference(
            path="p5_identity_preflight.json",
            sha256=file_hash,
            schema_version=schema_version,
            status=status,
            fingerprint_match=fingerprint_match,
            mode=mode,
        )
    except (json.JSONDecodeError, OSError):
        # File exists but couldn't be parsed - still reference it with hash only
        try:
            file_hash = compute_file_hash(preflight_path)
            return P5IdentityPreflightReference(
                path="p5_identity_preflight.json",
                sha256=file_hash,
            )
        except OSError:
            return None


def detect_p5_divergence_file(run_dir: Path) -> Optional[P5DivergenceReference]:
    """
    Detect and extract reference to P5 real telemetry divergence report if present.

    SHADOW MODE: This function provides observational detection only.
    Detection does not affect pack generation success/failure.

    The P5 divergence report provides real telemetry validation results
    linking to RTTS (manifold validation), TDA (metric comparison),
    and GGFL (governance signals).

    Expected location: p4_shadow/p5_divergence_real.json

    Args:
        run_dir: Path to the run directory or evidence pack directory.

    Returns:
        P5DivergenceReference if p5_divergence_real.json exists, None otherwise.
    """
    # Check expected location per Evidence_Pack_Spec_PhaseX.md Section 5.6
    p5_path = run_dir / "p4_shadow" / P5_DIVERGENCE_ARTIFACT
    if not p5_path.exists():
        # Also check root for flexibility
        p5_path = run_dir / P5_DIVERGENCE_ARTIFACT
        if not p5_path.exists():
            return None

    try:
        # Compute hash
        file_hash = compute_file_hash(p5_path)

        # Extract key fields
        with open(p5_path, "r", encoding="utf-8") as f:
            p5_data = json.load(f)

        # Validate against schema if available
        schema_valid = False
        validation_errors: List[str] = []

        if HAS_JSONSCHEMA:
            schema_path = (
                Path(__file__).parent.parent.parent.parent
                / "docs"
                / "system_law"
                / "schemas"
                / P5_DIVERGENCE_SCHEMA
            )
            if schema_path.exists():
                try:
                    with open(schema_path, "r", encoding="utf-8") as sf:
                        schema = json.load(sf)
                    jsonschema.validate(instance=p5_data, schema=schema)
                    schema_valid = True
                except jsonschema.ValidationError as ve:
                    validation_errors.append(str(ve.message))
                except Exception as e:
                    validation_errors.append(f"Schema validation error: {e}")

        # Determine relative path for reference
        rel_path = (
            f"p4_shadow/{P5_DIVERGENCE_ARTIFACT}"
            if (run_dir / "p4_shadow" / P5_DIVERGENCE_ARTIFACT).exists()
            else P5_DIVERGENCE_ARTIFACT
        )

        return P5DivergenceReference(
            path=rel_path,
            sha256=file_hash,
            schema_version=p5_data.get("schema_version"),
            telemetry_source=p5_data.get("telemetry_source"),
            validation_status=p5_data.get("validation_status"),
            divergence_rate=p5_data.get("divergence_rate"),
            mode=p5_data.get("mode"),
            schema_valid=schema_valid,
            validation_errors=validation_errors,
        )
    except (json.JSONDecodeError, OSError) as e:
        # File exists but couldn't be parsed - still reference it with error
        try:
            file_hash = compute_file_hash(p5_path)
            rel_path = (
                f"p4_shadow/{P5_DIVERGENCE_ARTIFACT}"
                if (run_dir / "p4_shadow" / P5_DIVERGENCE_ARTIFACT).exists()
                else P5_DIVERGENCE_ARTIFACT
            )
            return P5DivergenceReference(
                path=rel_path,
                sha256=file_hash,
                schema_valid=False,
                validation_errors=[f"Parse error: {e}"],
            )
        except OSError:
            return None


def detect_p5_diagnostic_file(run_dir: Path) -> Optional[P5DiagnosticReference]:
    """
    Detect and extract reference to P5 divergence diagnostic if present.

    SHADOW MODE CONTRACT:
    - This function is purely observational
    - Detection does not affect pack generation success/failure
    - The diagnostic is advisory only, no gating

    Expected locations:
    - p5_divergence_diagnostic.json (root)
    - p4_shadow/p5_divergence_diagnostic.json

    Args:
        run_dir: Path to the run directory or evidence pack directory.

    Returns:
        P5DiagnosticReference if p5_divergence_diagnostic.json exists, None otherwise.
    """
    # Check expected locations
    p5_path = run_dir / P5_DIVERGENCE_DIAGNOSTIC_ARTIFACT
    if not p5_path.exists():
        p5_path = run_dir / "p4_shadow" / P5_DIVERGENCE_DIAGNOSTIC_ARTIFACT
        if not p5_path.exists():
            return None

    try:
        # Compute hash
        file_hash = compute_file_hash(p5_path)

        # Extract key fields
        with open(p5_path, "r", encoding="utf-8") as f:
            diag_data = json.load(f)

        # Determine relative path
        rel_path = P5_DIVERGENCE_DIAGNOSTIC_ARTIFACT
        if (run_dir / "p4_shadow" / P5_DIVERGENCE_DIAGNOSTIC_ARTIFACT).exists():
            rel_path = f"p4_shadow/{P5_DIVERGENCE_DIAGNOSTIC_ARTIFACT}"

        return P5DiagnosticReference(
            path=rel_path,
            sha256=file_hash,
            schema_version=diag_data.get("schema_version"),
            root_cause_hypothesis=diag_data.get("root_cause_hypothesis"),
            action=diag_data.get("action"),
            headline=diag_data.get("headline"),
            cycle=diag_data.get("cycle"),
            run_id=diag_data.get("run_id"),
        )
    except (json.JSONDecodeError, OSError):
        # File exists but couldn't be parsed - still reference it
        try:
            file_hash = compute_file_hash(p5_path)
            rel_path = P5_DIVERGENCE_DIAGNOSTIC_ARTIFACT
            if (run_dir / "p4_shadow" / P5_DIVERGENCE_DIAGNOSTIC_ARTIFACT).exists():
                rel_path = f"p4_shadow/{P5_DIVERGENCE_DIAGNOSTIC_ARTIFACT}"
            return P5DiagnosticReference(
                path=rel_path,
                sha256=file_hash,
            )
        except OSError:
            return None


def detect_p5_replay_logs(
    run_dir: Path,
    expected_hashes: Optional[Dict[str, str]] = None,
    explicit_path: Optional[Path] = None,
) -> Optional[P5ReplayGovernanceReference]:
    """
    Detect and process P5 replay logs if present, extracting replay governance signal.

    SHADOW MODE CONTRACT:
    - This function is purely observational
    - Detection does not affect pack generation success/failure
    - The replay governance signal is advisory only, no gating

    ROBUSTNESS (v1.1.0):
    - Supports rotated JSONL segments (*.jsonl in directory)
    - Skips .jsonl.gz files with advisory warning (gzip not required)
    - Accepts explicit_path for absolute paths outside run_dir
    - Schema guard: preserves extraction on schema mismatch, surfaces schema_ok=false
    - Malformed lines are skipped with counter, pipeline completes
    - Deterministic ordering via sorted file/line processing

    Expected locations:
    - p4_shadow/p5_replay_logs.jsonl (single JSONL file)
    - p4_shadow/p5_replay_logs/ (directory with *.json or *.jsonl files)
    - p5_replay_logs.jsonl (root)
    - p5_replay_logs/ (root directory)

    Args:
        run_dir: Path to the run directory or evidence pack directory.
        expected_hashes: Optional dict mapping cycle_id -> expected_hash
            for determinism verification.
        explicit_path: Optional explicit path to JSONL file or directory.
            Allows absolute paths outside run_dir (for CLI usage).

    Returns:
        P5ReplayGovernanceReference if replay logs found and processed, None otherwise.
    """
    # Try to import P5 replay governance functions
    try:
        from backend.health.replay_governance_adapter import (
            extract_p5_replay_safety_from_logs,
        )
    except ImportError:
        # P5 replay governance module not available
        return None

    # Advisory warnings for robustness issues
    advisory_warnings: List[str] = []
    skipped_gz_count = 0
    malformed_line_count = 0

    # Determine replay logs path
    replay_logs_path: Optional[Path] = None
    replay_logs_dir: Optional[Path] = None

    if explicit_path is not None:
        # Use explicit path (can be absolute, outside run_dir)
        explicit_path = Path(explicit_path)
        if explicit_path.exists():
            if explicit_path.is_file():
                replay_logs_path = explicit_path
            elif explicit_path.is_dir():
                replay_logs_dir = explicit_path
        else:
            return None
    else:
        # Check standard locations
        for subdir in ["p4_shadow", ""]:
            base = run_dir / subdir if subdir else run_dir
            jsonl_path = base / P5_REPLAY_LOGS_ARTIFACT
            dir_path = base / P5_REPLAY_LOGS_DIR
            if jsonl_path.exists() and jsonl_path.is_file():
                replay_logs_path = jsonl_path
                break
            if dir_path.exists() and dir_path.is_dir():
                replay_logs_dir = dir_path
                break

    if replay_logs_path is None and replay_logs_dir is None:
        return None

    # Load replay logs with robustness
    replay_logs: List[Dict[str, Any]] = []

    if replay_logs_path is not None and replay_logs_path.is_file():
        # Single JSONL file
        try:
            with open(replay_logs_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            replay_logs.append(json.loads(line))
                        except json.JSONDecodeError:
                            malformed_line_count += 1
                            advisory_warnings.append(
                                f"Malformed JSON at {replay_logs_path.name}:{line_num}"
                            )
        except OSError as e:
            advisory_warnings.append(f"Failed to read {replay_logs_path.name}: {e}")
            return None

        # Compute relative path if possible
        try:
            rel_path = str(replay_logs_path.relative_to(run_dir))
        except ValueError:
            # Absolute path outside run_dir
            rel_path = str(replay_logs_path)
    else:
        # Directory with logs (supports rotation: *.json, *.jsonl)
        assert replay_logs_dir is not None

        # Collect all log files with deterministic ordering
        log_files: List[Path] = []
        for pattern in ["*.json", "*.jsonl"]:
            log_files.extend(replay_logs_dir.glob(pattern))

        # Check for .gz files and skip with warning
        gz_files = list(replay_logs_dir.glob("*.jsonl.gz"))
        if gz_files:
            skipped_gz_count = len(gz_files)
            advisory_warnings.append(
                f"Skipped {skipped_gz_count} .jsonl.gz file(s): gzip decompression not supported"
            )

        # Sort for deterministic ordering
        log_files = sorted(set(log_files), key=lambda p: p.name)

        for log_file in log_files:
            if log_file.suffix == ".jsonl":
                # JSONL file (rotated segment)
                try:
                    with open(log_file, "r", encoding="utf-8") as f:
                        for line_num, line in enumerate(f, 1):
                            line = line.strip()
                            if line:
                                try:
                                    replay_logs.append(json.loads(line))
                                except json.JSONDecodeError:
                                    malformed_line_count += 1
                                    advisory_warnings.append(
                                        f"Malformed JSON at {log_file.name}:{line_num}"
                                    )
                except OSError as e:
                    advisory_warnings.append(f"Failed to read {log_file.name}: {e}")
            else:
                # Single JSON file
                try:
                    with open(log_file, "r", encoding="utf-8") as f:
                        replay_logs.append(json.load(f))
                except json.JSONDecodeError:
                    malformed_line_count += 1
                    advisory_warnings.append(f"Malformed JSON in {log_file.name}")
                except OSError as e:
                    advisory_warnings.append(f"Failed to read {log_file.name}: {e}")

        # Compute relative path if possible
        try:
            rel_path = str(replay_logs_dir.relative_to(run_dir))
        except ValueError:
            # Absolute path outside run_dir
            rel_path = str(replay_logs_dir)

    if not replay_logs:
        # No valid logs loaded - but we may still have warnings to report
        if advisory_warnings:
            # Return reference with warnings but no signal
            return P5ReplayGovernanceReference(
                path=rel_path,
                sha256="",
                schema_version="1.0.0",
                status="warn",
                schema_ok=False,
                advisory_warnings=sorted(advisory_warnings),  # Deterministic order
                skipped_gz_count=skipped_gz_count,
                malformed_line_count=malformed_line_count,
            )
        return None

    # Extract production_run_id from logs or use default
    production_run_id = "unknown"
    if replay_logs and isinstance(replay_logs[0], dict):
        production_run_id = replay_logs[0].get(
            "run_id", replay_logs[0].get("production_run_id", "unknown")
        )

    # Schema/version guard: check for required P5 fields
    schema_ok = True
    required_p5_fields = {"cycle_id", "trace_hash", "timestamp"}
    for log in replay_logs:
        if not isinstance(log, dict):
            schema_ok = False
            advisory_warnings.append("Log entry is not a dict")
            break
        missing = required_p5_fields - set(log.keys())
        if missing:
            schema_ok = False
            advisory_warnings.append(f"Missing P5 fields: {sorted(missing)}")
            break

    # Extract P5 replay safety signal
    try:
        p5_signal = extract_p5_replay_safety_from_logs(
            replay_logs=replay_logs,
            production_run_id=production_run_id,
            expected_hashes=expected_hashes,
            telemetry_source="real",
        )
    except Exception as e:
        # Signal extraction failed - preserve with schema_ok=false
        advisory_warnings.append(f"Signal extraction failed: {e}")
        return P5ReplayGovernanceReference(
            path=rel_path,
            sha256="",
            schema_version="1.0.0",
            status="warn",
            schema_ok=False,
            advisory_warnings=sorted(advisory_warnings),
            skipped_gz_count=skipped_gz_count,
            malformed_line_count=malformed_line_count,
        )

    # Check schema version compatibility
    signal_schema = p5_signal.get("schema_version", "1.0.0")
    if signal_schema not in ("1.0.0", "1.1.0"):
        schema_ok = False
        advisory_warnings.append(f"Unknown schema version: {signal_schema}")

    # Compute hash of the signal JSON for manifest
    signal_json = json.dumps(p5_signal, sort_keys=True, separators=(",", ":"))
    signal_hash = compute_sha256(signal_json.encode("utf-8"))

    return P5ReplayGovernanceReference(
        path=rel_path,
        sha256=signal_hash,
        schema_version=p5_signal.get("schema_version", "1.0.0"),
        status=p5_signal.get("status"),
        determinism_rate=p5_signal.get("determinism_rate"),
        determinism_band=p5_signal.get("determinism_band"),
        p5_grade=p5_signal.get("p5_grade", False),
        telemetry_source=p5_signal.get("telemetry_source"),
        production_run_id=p5_signal.get("production_run_id"),
        hash_match_count=p5_signal.get("hash_match_count", 0),
        hash_mismatch_count=p5_signal.get("hash_mismatch_count", 0),
        replay_latency_ms=p5_signal.get("replay_latency_ms"),
        schema_ok=schema_ok,
        advisory_warnings=sorted(advisory_warnings),  # Deterministic order
        skipped_gz_count=skipped_gz_count,
        malformed_line_count=malformed_line_count,
    )


def detect_p5_pattern_tags(run_dir: Path) -> Optional[P5PatternTagsReference]:
    """
    Detect and extract reference to P5 pattern tags artifact if present.

    SHADOW MODE CONTRACT:
    - This function provides observational detection only
    - Detection does not affect pack generation success/failure
    - Provides TDA pattern classification for GGFL integration

    Expected locations:
    - p4_shadow/p5_pattern_tags.json
    - p5_pattern_tags.json (root)

    Args:
        run_dir: Path to the run directory or evidence pack directory.

    Returns:
        P5PatternTagsReference if p5_pattern_tags.json exists, None otherwise.
    """
    # Check expected location
    tags_path = run_dir / "p4_shadow" / P5_PATTERN_TAGS_ARTIFACT
    if not tags_path.exists():
        tags_path = run_dir / P5_PATTERN_TAGS_ARTIFACT
        if not tags_path.exists():
            return None

    try:
        # Compute hash
        file_hash = compute_file_hash(tags_path)

        # Extract key fields
        with open(tags_path, "r", encoding="utf-8") as f:
            tags_data = json.load(f)

        # Determine relative path for reference
        rel_path = (
            f"p4_shadow/{P5_PATTERN_TAGS_ARTIFACT}"
            if (run_dir / "p4_shadow" / P5_PATTERN_TAGS_ARTIFACT).exists()
            else P5_PATTERN_TAGS_ARTIFACT
        )

        # Extract classification summary
        summary = tags_data.get("classification_summary", {})
        shadow_invariants = tags_data.get("shadow_mode_invariants", {})

        return P5PatternTagsReference(
            path=rel_path,
            sha256=file_hash,
            schema_version=tags_data.get("schema_version"),
            final_pattern=summary.get("final_pattern"),
            final_streak=summary.get("final_streak", 0),
            cycles_analyzed=tags_data.get("cycles_analyzed", 0),
            recalibration_triggered=summary.get("recalibration_triggered", False),
            mode=tags_data.get("mode"),
            shadow_mode_invariants_ok=(
                shadow_invariants.get("no_enforcement", False) and
                shadow_invariants.get("logged_only", False) and
                shadow_invariants.get("observation_only", False)
            ),
        )
    except (json.JSONDecodeError, OSError):
        # File exists but couldn't be parsed - still reference it
        try:
            file_hash = compute_file_hash(tags_path)
            rel_path = (
                f"p4_shadow/{P5_PATTERN_TAGS_ARTIFACT}"
                if (run_dir / "p4_shadow" / P5_PATTERN_TAGS_ARTIFACT).exists()
                else P5_PATTERN_TAGS_ARTIFACT
            )
            return P5PatternTagsReference(
                path=rel_path,
                sha256=file_hash,
                shadow_mode_invariants_ok=False,
            )
        except OSError:
            return None


def detect_p5_topology_auditor_report(run_dir: Path) -> Optional[P5TopologyAuditorReference]:
    """
    Detect and extract reference to P5 topology auditor report if present.

    SHADOW MODE CONTRACT:
    - This function is purely observational
    - Detection does not affect pack generation success/failure
    - The topology auditor report is advisory only, no gating

    PARTIAL JSON EXTRACTION CONTRACT:
    - If report JSON is invalid (malformed), schema_ok=False is set
    - If schema_ok=False, scenario is NOT fabricated (remains None)
    - Advisory warning is returned via advisory_warning field
    - sha256 is still computed for the raw file bytes
    - This ensures no false scenario claims from corrupted data

    Expected locations:
    - p5_topology_auditor_report.json (root)
    - p4_shadow/p5_topology_auditor_report.json

    Args:
        run_dir: Path to the run directory or evidence pack directory.

    Returns:
        P5TopologyAuditorReference if report found, None otherwise.
    """
    # Check expected locations
    report_path = run_dir / P5_TOPOLOGY_AUDITOR_REPORT_ARTIFACT
    if not report_path.exists():
        report_path = run_dir / "p4_shadow" / P5_TOPOLOGY_AUDITOR_REPORT_ARTIFACT
        if not report_path.exists():
            return None

    # Determine relative path for reference
    rel_path = (
        f"p4_shadow/{P5_TOPOLOGY_AUDITOR_REPORT_ARTIFACT}"
        if (run_dir / "p4_shadow" / P5_TOPOLOGY_AUDITOR_REPORT_ARTIFACT).exists()
        else P5_TOPOLOGY_AUDITOR_REPORT_ARTIFACT
    )

    # Always compute hash first (for raw file bytes)
    try:
        file_hash = compute_file_hash(report_path)
    except OSError:
        return None

    # Attempt JSON parsing
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            report_data = json.load(f)
        schema_ok = True
        advisory_warning = None
    except json.JSONDecodeError as e:
        # PARTIAL EXTRACTION CONTRACT: Invalid JSON → schema_ok=False, no scenario fabrication
        return P5TopologyAuditorReference(
            path=rel_path,
            sha256=file_hash,
            schema_version=None,
            scenario=None,  # NOT fabricated when JSON invalid
            scenario_confidence=None,
            joint_status=None,
            shadow_mode_invariant_ok=True,
            validation_passed=False,
            mode=None,
            schema_ok=False,
            advisory_warning=f"JSON parse error: {str(e)[:100]}",
        )
    except OSError as e:
        # File read error
        return P5TopologyAuditorReference(
            path=rel_path,
            sha256=file_hash,
            schema_version=None,
            scenario=None,
            scenario_confidence=None,
            joint_status=None,
            shadow_mode_invariant_ok=True,
            validation_passed=False,
            mode=None,
            schema_ok=False,
            advisory_warning=f"File read error: {str(e)[:100]}",
        )

    # Extract scenario match - only if present in valid JSON
    scenario_match = report_data.get("scenario_match", {})
    scenario = scenario_match.get("scenario")
    scenario_confidence = scenario_match.get("confidence")

    # Extract P5 summary for joint_status
    p5_summary = report_data.get("p5_summary", {})
    joint_status = p5_summary.get("joint_status")

    # Extract smoke validation
    smoke_validation = report_data.get("smoke_validation", {})
    validation_passed = smoke_validation.get("validation_passed", False)
    shadow_mode_invariant_ok = smoke_validation.get("shadow_mode_invariant_ok", True)

    # Check for missing required fields and set advisory warning
    missing_fields = []
    if not scenario_match:
        missing_fields.append("scenario_match")
    if not p5_summary:
        missing_fields.append("p5_summary")
    if not smoke_validation:
        missing_fields.append("smoke_validation")

    if missing_fields:
        advisory_warning = f"Missing fields: {', '.join(missing_fields)}"
        # Note: We don't set schema_ok=False for missing fields, only for invalid JSON
        # This preserves the distinction between malformed JSON and incomplete data

    return P5TopologyAuditorReference(
        path=rel_path,
        sha256=file_hash,
        schema_version=report_data.get("schema_version"),
        scenario=scenario,
        scenario_confidence=scenario_confidence,
        joint_status=joint_status,
        shadow_mode_invariant_ok=shadow_mode_invariant_ok,
        validation_passed=validation_passed,
        mode=report_data.get("mode"),
        schema_ok=schema_ok,
        advisory_warning=advisory_warning,
    )


def detect_structural_drill_artifact(run_dir: Path) -> Optional[StructuralDrillReference]:
    """
    Detect and extract reference to P5 structural drill artifact if present.

    SHADOW MODE CONTRACT:
    - This function provides observational detection only
    - Detection does not affect pack generation success/failure
    - Provides CAL-EXP-3 stress test results for governance reporting

    Expected locations:
    - p5_structural_drill/structural_drill_artifact.json
    - structural_drill_artifact.json (root)

    Args:
        run_dir: Path to the run directory or evidence pack directory.

    Returns:
        StructuralDrillReference if structural_drill_artifact.json exists, None otherwise.
    """
    # Check expected locations
    drill_path = run_dir / STRUCTURAL_DRILL_DIR / STRUCTURAL_DRILL_ARTIFACT
    if not drill_path.exists():
        drill_path = run_dir / STRUCTURAL_DRILL_ARTIFACT
        if not drill_path.exists():
            # Also check for manifest
            manifest_path = run_dir / STRUCTURAL_DRILL_DIR / STRUCTURAL_DRILL_MANIFEST
            if not manifest_path.exists():
                manifest_path = run_dir / STRUCTURAL_DRILL_MANIFEST
            if manifest_path.exists():
                drill_path = manifest_path
            else:
                return None

    try:
        # Compute hash
        file_hash = compute_file_hash(drill_path)

        # Extract key fields
        with open(drill_path, "r", encoding="utf-8") as f:
            drill_data = json.load(f)

        # Validate against schema if available
        schema_valid = False
        validation_errors: List[str] = []

        if HAS_JSONSCHEMA:
            schema_path = (
                Path(__file__).parent.parent.parent.parent
                / "docs"
                / "system_law"
                / "schemas"
                / STRUCTURAL_DRILL_SCHEMA
            )
            if schema_path.exists():
                try:
                    with open(schema_path, "r", encoding="utf-8") as sf:
                        schema = json.load(sf)
                    jsonschema.validate(instance=drill_data, schema=schema)
                    schema_valid = True
                except jsonschema.ValidationError as ve:
                    validation_errors.append(str(ve.message))
                except Exception as e:
                    validation_errors.append(f"Schema validation error: {e}")

        # Determine relative path for reference
        rel_path = str(drill_path.relative_to(run_dir))

        # Extract summary fields
        summary = drill_data.get("summary", {})

        return StructuralDrillReference(
            path=rel_path,
            sha256=file_hash,
            drill_id=drill_data.get("drill_id"),
            scenario_id=drill_data.get("scenario_id"),
            drill_success=summary.get("drill_success", False),
            max_streak=summary.get("max_streak", 0),
            break_events=summary.get("break_events", []),
            pattern_counts=summary.get("pattern_counts", {}),
            mode=drill_data.get("metadata", {}).get("shadow_mode", True) and "SHADOW" or "UNKNOWN",
            schema_valid=schema_valid,
            validation_errors=validation_errors,
        )
    except (json.JSONDecodeError, OSError):
        # File exists but couldn't be parsed - still reference it with error
        try:
            file_hash = compute_file_hash(drill_path)
            rel_path = str(drill_path.relative_to(run_dir))
            return StructuralDrillReference(
                path=rel_path,
                sha256=file_hash,
                schema_valid=False,
                validation_errors=["Failed to parse drill artifact JSON"],
            )
        except OSError:
            return None


def detect_budget_calibration(run_dir: Path) -> Optional[BudgetCalibrationReference]:
    """
    Detect and extract reference to Budget Calibration artifact if present.

    SHADOW MODE CONTRACT:
    - This function provides observational detection only
    - Detection does not affect pack generation success/failure
    - Provides FP/FN calibration metrics for budget drift analysis
    - enablement_recommendation is advisory only, not gating

    Expected locations (checked in order):
    - budget_calibration_summary.json (root)
    - calibration/budget_calibration_summary.json

    Optional log file:
    - budget_calibration_log.jsonl (root)
    - calibration/budget_calibration_log.jsonl

    Args:
        run_dir: Path to the run directory or evidence pack directory.

    Returns:
        BudgetCalibrationReference if budget_calibration_summary.json exists, None otherwise.

    Reference: docs/system_law/Budget_PhaseX_Doctrine.md Section 7.3
    """
    # Check expected locations for summary
    summary_path = run_dir / BUDGET_CALIBRATION_SUMMARY_ARTIFACT
    if not summary_path.exists():
        summary_path = run_dir / BUDGET_CALIBRATION_SUBDIR / BUDGET_CALIBRATION_SUMMARY_ARTIFACT
        if not summary_path.exists():
            return None

    try:
        # Compute hash
        file_hash = compute_file_hash(summary_path)

        # Extract key fields
        with open(summary_path, "r", encoding="utf-8") as f:
            summary_data = json.load(f)

        # Determine relative path for reference
        rel_path = str(summary_path.relative_to(run_dir))

        # Extract compact summary fields
        compact = summary_data.get("compact_summary", {})
        schema_version = compact.get("schema_version") or summary_data.get("schema_version")
        enablement_recommendation = compact.get("enablement_recommendation")
        overall_pass = compact.get("overall_pass")
        fp_rate = compact.get("fp_rate")
        fn_rate = compact.get("fn_rate")

        # Check for optional log file
        log_path: Optional[str] = None
        log_sha256: Optional[str] = None

        # Check same directory as summary for log
        log_candidate = summary_path.parent / BUDGET_CALIBRATION_LOG_ARTIFACT
        if log_candidate.exists():
            log_path = str(log_candidate.relative_to(run_dir))
            log_sha256 = compute_file_hash(log_candidate)

        return BudgetCalibrationReference(
            path=rel_path,
            sha256=file_hash,
            schema_version=schema_version,
            enablement_recommendation=enablement_recommendation,
            fp_rate=fp_rate,
            fn_rate=fn_rate,
            overall_pass=overall_pass,
            log_path=log_path,
            log_sha256=log_sha256,
            mode="SHADOW",
        )
    except (json.JSONDecodeError, OSError):
        # File exists but couldn't be parsed - still reference it
        try:
            file_hash = compute_file_hash(summary_path)
            rel_path = str(summary_path.relative_to(run_dir))
            return BudgetCalibrationReference(
                path=rel_path,
                sha256=file_hash,
                mode="SHADOW",
            )
        except OSError:
            return None


# =============================================================================
# Evidence Pack Builder
# =============================================================================

class EvidencePackBuilder:
    """
    Builder for Evidence Pack bundles.

    Walks a First Light run directory, validates artifacts, computes
    cryptographic hashes, and generates a bundle manifest.

    SHADOW MODE CONTRACT:
    - All governance checks are advisory only
    - Missing optional artifacts do not fail the build
    - Pack is generated even if governance checks fail
    """

    def __init__(
        self,
        schemas_dir: Optional[Path] = None,
        validate_schemas: bool = True,
    ):
        """
        Initialize the builder.

        Args:
            schemas_dir: Path to schema files. If None, uses default location.
            validate_schemas: Whether to validate artifacts against schemas.
        """
        self.schemas_dir = schemas_dir or self._default_schemas_dir()
        self.validate_schemas = validate_schemas and HAS_JSONSCHEMA
        self._schema_cache: Dict[str, Dict[str, Any]] = {}

    def _default_schemas_dir(self) -> Path:
        """Get default schemas directory."""
        # Navigate from backend/topology/first_light to docs/system_law/schemas
        module_dir = Path(__file__).parent
        return module_dir.parent.parent.parent / "docs" / "system_law" / "schemas"

    def _load_schema(self, schema_name: str) -> Optional[Dict[str, Any]]:
        """Load and cache a schema."""
        if schema_name in self._schema_cache:
            return self._schema_cache[schema_name]

        schema_path = self.schemas_dir / schema_name
        schema = load_schema(schema_path)
        if schema:
            self._schema_cache[schema_name] = schema
        return schema

    def _detect_format(self, filename: str) -> str:
        """Detect file format from extension."""
        if filename.endswith(".jsonl"):
            return "jsonl"
        elif filename.endswith(".json"):
            return "json"
        elif filename.endswith(".svg"):
            return "svg"
        elif filename.endswith(".md"):
            return "md"
        else:
            return "txt"

    def _detect_category(self, rel_path: str) -> str:
        """Detect artifact category from path."""
        if rel_path.startswith("p3_synthetic/") or rel_path.startswith("p3_synthetic\\"):
            return "p3_synthetic"
        elif rel_path.startswith("p4_shadow/") or rel_path.startswith("p4_shadow\\"):
            return "p4_shadow"
        elif rel_path.startswith("visualizations/") or rel_path.startswith("visualizations\\"):
            return "visualization"
        elif rel_path.startswith("compliance/") or rel_path.startswith("compliance\\"):
            return "compliance"
        else:
            return "other"

    def _validate_artifact(
        self,
        file_path: Path,
        artifact_name: str,
    ) -> Tuple[bool, List[str]]:
        """Validate an artifact against its schema."""
        if not self.validate_schemas:
            return True, []

        schema_name = ARTIFACT_SCHEMA_MAP.get(artifact_name)
        if not schema_name:
            return True, ["No schema defined for this artifact"]

        schema = self._load_schema(schema_name)
        if not schema:
            return True, [f"Schema not found: {schema_name}"]

        file_format = self._detect_format(artifact_name)

        if file_format == "jsonl":
            return validate_jsonl_against_schema(file_path, schema)
        elif file_format == "json":
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return validate_json_against_schema(data, schema)
            except (json.JSONDecodeError, OSError) as e:
                return False, [f"Failed to load JSON: {e}"]
        else:
            return True, ["Non-JSON format - schema validation skipped"]

    def _resolve_proof_log_path(
        self,
        run_dir: Path,
        proof_log_path: Optional[str | Path],
    ) -> Tuple[Optional[Path], List[str]]:
        """
        Resolve the proof log path using explicit parameters, env vars, or defaults.
        Returns (resolved_path, checked_paths).
        """
        search_paths: List[Path] = []
        checked: List[str] = []
        seen: set[str] = set()

        def register(path: Path) -> None:
            key = str(path)
            if key not in seen:
                search_paths.append(path)
                seen.add(key)

        if proof_log_path:
            candidate = Path(proof_log_path)
            register(candidate)
            if not candidate.is_absolute():
                register(run_dir / candidate)

        env_candidate = os.environ.get(PROOF_LOG_ENV_VAR)
        if env_candidate:
            candidate = Path(env_candidate)
            register(candidate)
            if not candidate.is_absolute():
                register(run_dir / candidate)

        for relative in DEFAULT_PROOF_LOG_RELATIVE_CANDIDATES:
            register(run_dir / relative)

        for candidate in search_paths:
            checked.append(str(candidate))
            if candidate.exists():
                return candidate, checked

        return None, checked

    def _maybe_generate_proof_snapshot(
        self,
        run_dir: Path,
        proof_log_path: Optional[str | Path],
        proof_snapshot_output: Optional[str | Path],
    ) -> Tuple[Optional[Dict[str, Any]], List[str]]:
        """
        Generate proof snapshot JSON if configured.

        Returns (manifest_entry, warnings).
        """
        warnings: List[str] = []
        proof_log_file, checked = self._resolve_proof_log_path(run_dir, proof_log_path)

        if proof_log_file is None:
            if checked:
                warnings.append(
                    "Proof snapshot requested but proof log not found. "
                    "Checked: " + ", ".join(checked)
                )
            else:
                warnings.append(
                    "Proof snapshot requested but no proof log path was provided."
                )
            return None, warnings

        snapshot_rel = (
            Path(proof_snapshot_output)
            if proof_snapshot_output
            else DEFAULT_PROOF_SNAPSHOT_REL_PATH
        )
        snapshot_abs = snapshot_rel if snapshot_rel.is_absolute() else run_dir / snapshot_rel
        snapshot_abs.parent.mkdir(parents=True, exist_ok=True)

        try:
            snapshot_payload = generate_snapshot(
                str(proof_log_file),
                str(snapshot_abs),
            )
        except Exception as exc:  # pragma: no cover - best effort logging
            warnings.append(f"Failed to generate proof snapshot: {exc}")
            return None, warnings

        file_hash = compute_file_hash(snapshot_abs)
        try:
            manifest_path = snapshot_abs.relative_to(run_dir).as_posix()
        except ValueError:
            manifest_path = snapshot_abs.as_posix()

        manifest_entry = {
            "path": manifest_path,
            "sha256": file_hash,
            "schema_version": snapshot_payload.get("schema_version"),
            "canonical_hash_algorithm": snapshot_payload.get("canonical_hash_algorithm"),
            "canonicalization_version": snapshot_payload.get("canonicalization_version"),
            "canonical_hash": snapshot_payload.get("canonical_hash"),
            "entry_count": snapshot_payload.get("entry_count"),
            "source": snapshot_payload.get("source"),
        }
        return manifest_entry, warnings

    def _collect_artifacts(
        self,
        run_dir: Path,
    ) -> Tuple[List[ArtifactInfo], CompletenessCheck]:
        """
        Collect all artifacts from run directory.

        Returns (artifacts, completeness_check).
        """
        artifacts = []
        completeness = CompletenessCheck()

        # Track found artifacts
        found_p3 = set()
        found_p4 = set()
        found_viz = set()
        found_compliance = set()

        # Walk directory
        for root, dirs, files in os.walk(run_dir):
            root_path = Path(root)
            for filename in files:
                file_path = root_path / filename
                rel_path = str(file_path.relative_to(run_dir))

                # Compute hash and size
                file_hash = compute_file_hash(file_path)
                file_size = file_path.stat().st_size

                # Detect category
                category = self._detect_category(rel_path)
                file_format = self._detect_format(filename)

                # Validate against schema
                is_valid, errors = self._validate_artifact(file_path, filename)

                # Determine if required
                is_required = (
                    filename in REQUIRED_P3_ARTIFACTS or
                    filename in REQUIRED_P4_ARTIFACTS or
                    filename in REQUIRED_VISUALIZATIONS
                )

                # Get schema reference
                schema_ref = ARTIFACT_SCHEMA_MAP.get(filename)

                artifact = ArtifactInfo(
                    path=rel_path,
                    sha256=file_hash,
                    size_bytes=file_size,
                    category=category,
                    schema_ref=schema_ref,
                    format=file_format,
                    description=f"{category} artifact: {filename}",
                    required=is_required,
                    validation_passed=is_valid,
                    validation_errors=errors,
                )
                artifacts.append(artifact)

                # Track for completeness
                if category == "p3_synthetic":
                    found_p3.add(filename)
                elif category == "p4_shadow":
                    found_p4.add(filename)
                elif category == "visualization":
                    found_viz.add(filename)
                elif category == "compliance":
                    found_compliance.add(filename)

        # Build completeness check
        for name in REQUIRED_P3_ARTIFACTS + OPTIONAL_P3_ARTIFACTS:
            completeness.p3_artifacts[name] = name in found_p3

        for name in REQUIRED_P4_ARTIFACTS + OPTIONAL_P4_ARTIFACTS:
            completeness.p4_artifacts[name] = name in found_p4

        for name in REQUIRED_VISUALIZATIONS + OPTIONAL_VISUALIZATIONS:
            completeness.visualizations[name] = name in found_viz

        # Check compliance artifacts
        completeness.compliance["compliance_narrative.md"] = "compliance_narrative.md" in found_compliance
        completeness.compliance["audit_attestation.json"] = "audit_attestation.json" in found_compliance

        # Determine missing required
        missing = []
        for name in REQUIRED_P3_ARTIFACTS:
            if name not in found_p3:
                missing.append(f"p3_synthetic/{name}")
        for name in REQUIRED_P4_ARTIFACTS:
            if name not in found_p4:
                missing.append(f"p4_shadow/{name}")
        for name in REQUIRED_VISUALIZATIONS:
            if name not in found_viz:
                missing.append(f"visualizations/{name}")

        completeness.missing_artifacts = missing
        completeness.all_required_present = len(missing) == 0

        return artifacts, completeness

    def _generate_manifest(
        self,
        bundle_id: str,
        artifacts: List[ArtifactInfo],
        completeness: CompletenessCheck,
        merkle_root: str,
        run_dir: Path,
        p3_run_id: Optional[str] = None,
        p4_run_id: Optional[str] = None,
        status_reference: Optional[StatusReference] = None,
        pathology_annotation: Optional[Dict[str, Any]] = None,
        p5_replay_governance_reference: Optional[P5ReplayGovernanceReference] = None,
        p5_pattern_tags_reference: Optional[P5PatternTagsReference] = None,
        structural_drill_reference: Optional[StructuralDrillReference] = None,
        p5_topology_auditor_reference: Optional[P5TopologyAuditorReference] = None,
        p5_diagnostic_reference: Optional[P5DiagnosticReference] = None,
        budget_calibration_reference: Optional[BudgetCalibrationReference] = None,
        p5_divergence_reference: Optional[P5DivergenceReference] = None,
        nci_p5_reference: Optional[NciP5Reference] = None,
        p5_identity_preflight_reference: Optional[P5IdentityPreflightReference] = None,
    ) -> Dict[str, Any]:
        """Generate the bundle manifest."""
        now = datetime.now(timezone.utc).isoformat()

        manifest: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "bundle_id": bundle_id,
            "bundle_version": EVIDENCE_PACK_VERSION,
            "generated_at": now,
            "p3_run_id": p3_run_id or "unknown",
            "p4_run_id": p4_run_id or "unknown",
            "artifacts": [
                {
                    "path": a.path,
                    "sha256": a.sha256,
                    "size_bytes": a.size_bytes,
                    "category": a.category,
                    "schema_ref": a.schema_ref,
                    "format": a.format,
                    "required": a.required,
                }
                for a in artifacts
            ],
            "completeness": {
                "p3_artifacts": completeness.p3_artifacts,
                "p4_artifacts": completeness.p4_artifacts,
                "visualizations": completeness.visualizations,
                "compliance": completeness.compliance,
                "all_required_present": completeness.all_required_present,
                "missing_artifacts": completeness.missing_artifacts,
            },
            "cryptographic_root": f"sha256:{merkle_root}",
            "validation_status": {
                "all_artifacts_present": completeness.all_required_present,
                "all_hashes_verified": True,
                "all_schemas_valid": all(a.validation_passed for a in artifacts),
                "completeness_check_passed": completeness.all_required_present,
                "validation_timestamp": now,
            },
        }

        # Add status reference if first_light_status.json exists
        # External verifiers should treat Evidence Pack + status as paired artifacts
        if status_reference is not None:
            manifest["status_reference"] = {
                "path": status_reference.path,
                "sha256": status_reference.sha256,
                "schema_version": status_reference.schema_version,
                "shadow_mode_ok": status_reference.shadow_mode_ok,
            }

        # Add P5 divergence reference if p5_divergence_real.json exists (SHADOW MODE)
        # Reference: docs/system_law/schemas/p5/p5_divergence_real.schema.json
        if p5_divergence_reference is not None:
            manifest["p5_divergence_reference"] = {
                "path": p5_divergence_reference.path,
                "sha256": p5_divergence_reference.sha256,
                "schema_version": p5_divergence_reference.schema_version,
                "telemetry_source": p5_divergence_reference.telemetry_source,
                "validation_status": p5_divergence_reference.validation_status,
                "divergence_rate": p5_divergence_reference.divergence_rate,
                "mode": p5_divergence_reference.mode,
                "schema_valid": p5_divergence_reference.schema_valid,
            }

        # Annotate pathology injection used (SHADOW-ONLY metadata)
        if pathology_annotation is not None:
            manifest["evidence"] = {
                "data": {
                    "p3_pathology": pathology_annotation,
                }
            }

            governance_pathology = self._build_pathology_governance(pathology_annotation)
            if governance_pathology:
                manifest["governance"] = manifest.get("governance", {})
                manifest["governance"]["p3_pathology"] = governance_pathology

        # P5 BASELINE: Include mock vs real comparison if available (SHADOW ONLY)
        p5_comparison = self._load_p5_mock_vs_real_comparison(run_dir)
        if p5_comparison:
            manifest["governance"] = manifest.get("governance", {})
            manifest["governance"]["p5_mock_vs_real_comparison"] = p5_comparison

        # P5 CALIBRATION: Include CAL-EXP-1 summary if present (SHADOW ONLY)
        cal_exp1 = self._load_cal_exp1_summary(run_dir)
        if cal_exp1:
            manifest["governance"] = manifest.get("governance", {})
            manifest["governance"].setdefault("p5_calibration", {})
            manifest["governance"]["p5_calibration"]["cal_exp1"] = cal_exp1

        # P5 REPLAY SAFETY: Include replay governance signal if present (SHADOW ONLY)
        # Reference: docs/system_law/Replay_Safety_P5_Engineering_Plan.md
        if p5_replay_governance_reference is not None:
            manifest["governance"] = manifest.get("governance", {})
            manifest["governance"]["replay_p5"] = {
                "schema_version": p5_replay_governance_reference.schema_version,
                "path": p5_replay_governance_reference.path,
                "sha256": p5_replay_governance_reference.sha256,
                "status": p5_replay_governance_reference.status,
                "determinism_rate": p5_replay_governance_reference.determinism_rate,
                "determinism_band": p5_replay_governance_reference.determinism_band,
                "p5_grade": p5_replay_governance_reference.p5_grade,
                "telemetry_source": p5_replay_governance_reference.telemetry_source,
                "production_run_id": p5_replay_governance_reference.production_run_id,
                "hash_match_count": p5_replay_governance_reference.hash_match_count,
                "hash_mismatch_count": p5_replay_governance_reference.hash_mismatch_count,
                "replay_latency_ms": p5_replay_governance_reference.replay_latency_ms,
                # SHADOW MODE CONTRACT marker
                "mode": "SHADOW",
                "shadow_mode_contract": {
                    "observational_only": True,
                    "no_control_flow_influence": True,
                    "no_governance_modification": True,
                },
            }

        # P5 PATTERN TAGS: Include TDA pattern classification if present (SHADOW ONLY)
        # Reference: docs/system_law/GGFL_P5_Pattern_Test_Plan.md
        if p5_pattern_tags_reference is not None:
            manifest["governance"] = manifest.get("governance", {})
            manifest["governance"]["p5_patterns"] = {
                "schema_version": p5_pattern_tags_reference.schema_version,
                "path": p5_pattern_tags_reference.path,
                "sha256": p5_pattern_tags_reference.sha256,
                "final_pattern": p5_pattern_tags_reference.final_pattern,
                "final_streak": p5_pattern_tags_reference.final_streak,
                "cycles_analyzed": p5_pattern_tags_reference.cycles_analyzed,
                "recalibration_triggered": p5_pattern_tags_reference.recalibration_triggered,
                # SHADOW MODE CONTRACT marker
                "mode": "SHADOW",
                "shadow_mode_invariants_ok": p5_pattern_tags_reference.shadow_mode_invariants_ok,
                "shadow_mode_contract": {
                    "observational_only": True,
                    "no_control_flow_influence": True,
                    "no_governance_modification": True,
                },
            }

        # P5 STRUCTURAL DRILL: Include CAL-EXP-3 stress test if present (SHADOW ONLY)
        # Reference: docs/system_law/P5_Structural_Drill_Package.md
        if structural_drill_reference is not None:
            manifest["governance"] = manifest.get("governance", {})
            manifest["governance"]["structure"] = manifest["governance"].get("structure", {})
            manifest["governance"]["structure"]["drill"] = {
                "path": structural_drill_reference.path,
                "sha256": structural_drill_reference.sha256,
                "drill_id": structural_drill_reference.drill_id,
                "scenario_id": structural_drill_reference.scenario_id,
                "drill_success": structural_drill_reference.drill_success,
                "max_streak": structural_drill_reference.max_streak,
                "break_events": structural_drill_reference.break_events,
                "pattern_counts": structural_drill_reference.pattern_counts,
                "schema_valid": structural_drill_reference.schema_valid,
                # SHADOW MODE CONTRACT marker
                "mode": "SHADOW",
                "shadow_mode_contract": {
                    "observational_only": True,
                    "no_control_flow_influence": True,
                    "no_governance_modification": True,
                },
            }

        # P5 TOPOLOGY AUDITOR: Include topology/bundle validation if present (SHADOW ONLY)
        # Reference: docs/system_law/Topology_Bundle_PhaseX_Requirements.md Section 10
        if p5_topology_auditor_reference is not None:
            manifest["governance"] = manifest.get("governance", {})
            manifest["governance"]["topology_p5"] = {
                "schema_version": p5_topology_auditor_reference.schema_version,
                "path": p5_topology_auditor_reference.path,
                "sha256": p5_topology_auditor_reference.sha256,
                "scenario": p5_topology_auditor_reference.scenario,
                "scenario_confidence": p5_topology_auditor_reference.scenario_confidence,
                "joint_status": p5_topology_auditor_reference.joint_status,
                "validation_passed": p5_topology_auditor_reference.validation_passed,
                # SHADOW MODE CONTRACT marker
                "mode": "SHADOW",
                "shadow_mode_invariant_ok": p5_topology_auditor_reference.shadow_mode_invariant_ok,
                "shadow_mode_contract": {
                    "observational_only": True,
                    "no_control_flow_influence": True,
                    "no_governance_modification": True,
                },
            }

        # P5 Divergence Diagnostic (SHADOW MODE)
        # Reference: docs/system_law/P5_Divergence_Diagnostic_Panel_Spec.md
        if p5_diagnostic_reference is not None:
            manifest["governance"] = manifest.get("governance", {})
            manifest["governance"]["diagnostic_p5"] = {
                "schema_version": p5_diagnostic_reference.schema_version,
                "path": p5_diagnostic_reference.path,
                "sha256": p5_diagnostic_reference.sha256,
                "root_cause_hypothesis": p5_diagnostic_reference.root_cause_hypothesis,
                "action": p5_diagnostic_reference.action,
                "headline": p5_diagnostic_reference.headline,
                "cycle": p5_diagnostic_reference.cycle,
                "run_id": p5_diagnostic_reference.run_id,
                # SHADOW MODE CONTRACT marker
                "mode": "SHADOW",
                "shadow_mode_contract": {
                    "observational_only": True,
                    "no_control_flow_influence": True,
                    "no_governance_modification": True,
                },
            }

        # Budget Calibration (SHADOW MODE - Phase X Budget Doctrine)
        # Reference: docs/system_law/Budget_PhaseX_Doctrine.md Section 7.3
        if budget_calibration_reference is not None:
            manifest["governance"] = manifest.get("governance", {})
            manifest["governance"]["budget_risk"] = manifest["governance"].get("budget_risk", {})
            calibration_ref: Dict[str, Any] = {
                "path": budget_calibration_reference.path,
                "sha256": budget_calibration_reference.sha256,
                "schema_version": budget_calibration_reference.schema_version,
                "enablement_recommendation": budget_calibration_reference.enablement_recommendation,
                "overall_pass": budget_calibration_reference.overall_pass,
                # Round FP/FN rates for auditor readability
                "fp_rate": (
                    round(budget_calibration_reference.fp_rate, 4)
                    if budget_calibration_reference.fp_rate is not None
                    else None
                ),
                "fn_rate": (
                    round(budget_calibration_reference.fn_rate, 4)
                    if budget_calibration_reference.fn_rate is not None
                    else None
                ),
                # SHADOW MODE CONTRACT marker
                "mode": "SHADOW",
                "shadow_mode_contract": {
                    "observational_only": True,
                    "no_control_flow_influence": True,
                    "no_governance_modification": True,
                },
            }
            # Include optional log reference if present
            if budget_calibration_reference.log_path:
                calibration_ref["log_path"] = budget_calibration_reference.log_path
                calibration_ref["log_sha256"] = budget_calibration_reference.log_sha256
            manifest["governance"]["budget_risk"]["calibration_reference"] = calibration_ref

        # NCI P5 Narrative Consistency Index (SHADOW MODE)
        # Reference: docs/system_law/NCI_PhaseX_Spec.md Section 11
        # Surfaces at: manifest.signals.nci_p5
        if nci_p5_reference is not None:
            manifest["signals"] = manifest.get("signals", {})
            nci_p5_entry: Dict[str, Any] = {
                "schema_version": nci_p5_reference.schema_version,
                "path": nci_p5_reference.path,
                # Artifact provenance: explicit signal/result sha256 and detection location
                "signal_sha256": nci_p5_reference.sha256,
                "detection_path": nci_p5_reference.detection_path,
                "extraction_source": nci_p5_reference.extraction_source,
                "mode": nci_p5_reference.mode,
                "global_nci": nci_p5_reference.global_nci,
                "confidence": nci_p5_reference.confidence,
                "slo_status": nci_p5_reference.slo_status,
                "recommendation": nci_p5_reference.recommendation,
                "tcl_aligned": nci_p5_reference.tcl_aligned,
                "sic_aligned": nci_p5_reference.sic_aligned,
                "tcl_violation_count": nci_p5_reference.tcl_violation_count,
                "sic_violation_count": nci_p5_reference.sic_violation_count,
                "warning_count": nci_p5_reference.warning_count,
                # SHADOW MODE CONTRACT marker
                "shadow_mode": True,
                "shadow_mode_contract": {
                    "observational_only": True,
                    "no_control_flow_influence": True,
                    "no_governance_modification": True,
                },
            }
            # Include result file cross-reference if signal file was primary
            if nci_p5_reference.result_path:
                nci_p5_entry["result_path"] = nci_p5_reference.result_path
                nci_p5_entry["result_sha256"] = nci_p5_reference.result_sha256
            manifest["signals"]["nci_p5"] = nci_p5_entry

        # P5 IDENTITY PREFLIGHT: Include artifact reference if present (SHADOW MODE)
        # Reference: docs/system_law/Identity_Preflight_Precedence_Law.md
        # Surfaces at: manifest.governance.slice_identity.p5_preflight_reference
        if p5_identity_preflight_reference is not None:
            manifest["governance"] = manifest.get("governance", {})
            manifest["governance"]["slice_identity"] = manifest["governance"].get("slice_identity", {})
            manifest["governance"]["slice_identity"]["p5_preflight_reference"] = {
                "path": p5_identity_preflight_reference.path,
                "sha256": p5_identity_preflight_reference.sha256,
                "schema_version": p5_identity_preflight_reference.schema_version,
                "status": p5_identity_preflight_reference.status,
                "fingerprint_match": p5_identity_preflight_reference.fingerprint_match,
                "mode": p5_identity_preflight_reference.mode,
                # SHADOW MODE CONTRACT marker
                "shadow_mode_contract": {
                    "observational_only": True,
                    "no_control_flow_influence": True,
                    "no_governance_modification": True,
                },
            }

        return manifest

    def _load_pathology_annotation(self, run_dir: Path) -> Optional[Dict[str, Any]]:
        """
        Load pathology annotation from stability_report.json if present.

        Returns:
            Dict with pathology metadata or None if not available/none.
        """
        report_path = run_dir / "stability_report.json"
        if not report_path.exists():
            return None

        try:
            with report_path.open("r", encoding="utf-8") as f:
                report = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

        pathology = report.get("pathology")
        if not pathology or pathology == "none":
            return None

        return {
            "pathology": pathology,
            "pathology_params": report.get("pathology_params", {}) or {},
        }

    def _load_p5_mock_vs_real_comparison(self, run_dir: Path) -> Optional[Dict[str, Any]]:
        """
        Load P5 mock vs real comparison data from P4 artifacts.

        SHADOW MODE CONTRACT:
        - This is purely observational data extraction
        - It does not gate or modify any governance decisions
        - The comparison is advisory only for P5 baseline characterization

        Returns:
            Dict with comparison metrics or None if not a P5 run
        """
        # Check for P4 run config with P5 adapter
        p4_config_path = run_dir / "run_config.json"
        if not p4_config_path.exists():
            return None

        try:
            with p4_config_path.open("r", encoding="utf-8") as f:
                config = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

        # Check if this is a P5 run (real adapter)
        telemetry_source = config.get("telemetry_source", "mock")
        if not telemetry_source.startswith("real"):
            return None

        # Load P4 summary for divergence metrics
        summary_path = run_dir / "p4_summary.json"
        if not summary_path.exists():
            return None

        try:
            with summary_path.open("r", encoding="utf-8") as f:
                summary = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

        divergence = summary.get("divergence_analysis", {})
        twin_acc = summary.get("twin_accuracy", {})

        return {
            "schema_version": "1.0.0",
            "mode": "SHADOW",
            "telemetry_source": telemetry_source,
            "run_id": config.get("run_id"),
            "metrics": {
                "divergence_rate": divergence.get("divergence_rate"),
                "twin_success_accuracy": twin_acc.get("success_prediction_accuracy"),
                "twin_omega_accuracy": twin_acc.get("omega_prediction_accuracy"),
                "twin_blocked_accuracy": twin_acc.get("blocked_prediction_accuracy"),
                "max_divergence_streak": divergence.get("max_divergence_streak"),
            },
            # P5 baseline: observational advisory, no gating
            "advisory": {
                "status": "SHADOW_OBSERVATION",
                "note": "P5 mock vs real comparison for baseline characterization. No gating.",
            },
        }

    def _build_pathology_governance(self, pathology_annotation: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Build governance annotation for pathology stress-test (P5 narrative).
        """
        pathology_type = pathology_annotation.get("pathology")
        if not pathology_type:
            return None

        params = pathology_annotation.get("pathology_params", {}) or {}
        if pathology_type == "spike":
            magnitude = params.get("magnitude")
            expected = "Sharp H spike injected as P5 stress probe"
        elif pathology_type == "drift":
            magnitude = params.get("slope")
            expected = "Linear H drift to probe slow-roll degradation"
        elif pathology_type == "oscillation":
            magnitude = params.get("amplitude")
            expected = "Sinusoidal H oscillation to probe stability under periodic stress"
        else:
            magnitude = None
            expected = "Unknown pathology type"

        entry: Dict[str, Any] = {
            "type": pathology_type,
            "magnitude": magnitude,
            "expected_effects": expected,
        }
        if params:
            entry["params"] = params
        return entry

    def _load_cal_exp1_summary(self, run_dir: Path) -> Optional[Dict[str, Any]]:
        """
        Load CAL-EXP-1 warm-start summary (SHADOW advisory only).
        """
        report_path = run_dir / "cal_exp1_report.json"
        if not report_path.exists():
            return None

        try:
            with report_path.open("r", encoding="utf-8") as f:
                report = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

        windows = report.get("windows", [])
        if windows:
            mean_divergence = sum(w.get("divergence_rate", 0.0) for w in windows) / len(windows)
        else:
            mean_divergence = None

        summary = report.get("summary", {})
        return {
            "schema_version": "1.0.0",
            "final_divergence_rate": summary.get("final_divergence_rate"),
            "final_delta_bias": summary.get("final_delta_bias"),
            "mean_divergence_over_windows": mean_divergence,
            "pattern_tag": windows[-1].get("pattern_tag") if windows else None,
            "note": "SHADOW advisory only; no gating.",
        }

    def build_evidence_pack(
        self,
        run_dir: Path,
        output_dir: Optional[Path] = None,
        p3_run_id: Optional[str] = None,
        p4_run_id: Optional[str] = None,
        include_proof_snapshot: bool = False,
        proof_log_path: Optional[str | Path] = None,
        proof_snapshot_output: Optional[str | Path] = None,
    ) -> EvidencePackResult:
        """
        Build an evidence pack from a First Light run directory.

        Args:
            run_dir: Path to the run directory containing artifacts.
            output_dir: Optional output directory. If None, uses run_dir.
            p3_run_id: Optional P3 run identifier.
            p4_run_id: Optional P4 run identifier.
            include_proof_snapshot: Whether to emit compliance/proof_log_snapshot.json
                (can also be enabled via FIRST_LIGHT_INCLUDE_PROOF_SNAPSHOT env var).
            proof_log_path: Optional explicit path to the proof log JSONL.
            proof_snapshot_output: Optional override for snapshot output path.

        Returns:
            EvidencePackResult with build status and details.
        """
        run_dir = Path(run_dir)
        output_dir = Path(output_dir) if output_dir else run_dir

        errors = []
        warnings = []

        # Validate run directory exists
        if not run_dir.exists():
            return EvidencePackResult(
                success=False,
                bundle_id="",
                manifest_path=None,
                merkle_root=None,
                artifacts=[],
                completeness=CompletenessCheck(),
                governance_advisories=[],
                errors=[f"Run directory does not exist: {run_dir}"],
                warnings=[],
            )

        proof_snapshot_manifest: Optional[Dict[str, Any]] = None
        include_snapshot_flag = include_proof_snapshot or _truthy_env(
            os.environ.get(PROOF_SNAPSHOT_ENV_FLAG)
        )
        if include_snapshot_flag:
            snapshot_entry, snapshot_warnings = self._maybe_generate_proof_snapshot(
                run_dir=run_dir,
                proof_log_path=proof_log_path,
                proof_snapshot_output=proof_snapshot_output,
            )
            if snapshot_entry:
                proof_snapshot_manifest = snapshot_entry
            if snapshot_warnings:
                warnings.extend(snapshot_warnings)

        # Generate bundle ID
        bundle_id = str(uuid.uuid4())

        # Collect artifacts
        artifacts, completeness = self._collect_artifacts(run_dir)

        if not artifacts:
            return EvidencePackResult(
                success=False,
                bundle_id=bundle_id,
                manifest_path=None,
                merkle_root=None,
                artifacts=[],
                completeness=completeness,
                governance_advisories=[],
                errors=["No artifacts found in run directory"],
                warnings=[],
            )

        # Compute Merkle root
        artifact_hashes = [a.sha256 for a in artifacts]
        merkle_root = compute_merkle_root(artifact_hashes)

        # Run governance checks (SHADOW MODE - advisory only)
        advisories = run_governance_checks(artifacts, completeness, run_dir)

        # Track warnings from failed advisories
        for advisory in advisories:
            if not advisory.passed:
                warnings.append(f"[{advisory.severity}] {advisory.check_name}: {advisory.message}")

        # Detect status file cross-link
        # first_light_status.json provides a machine-readable summary that
        # external verifiers should treat as paired with this Evidence Pack
        status_reference = detect_status_file(run_dir)

        pathology_annotation = self._load_pathology_annotation(run_dir)

        # Detect P5 replay logs and extract governance signal (SHADOW MODE)
        # Reference: docs/system_law/Replay_Safety_P5_Engineering_Plan.md
        p5_replay_governance_reference = detect_p5_replay_logs(run_dir)

        # Detect P5 pattern tags and extract TDA classification (SHADOW MODE)
        # Reference: docs/system_law/GGFL_P5_Pattern_Test_Plan.md
        p5_pattern_tags_reference = detect_p5_pattern_tags(run_dir)

        # Detect P5 structural drill artifact (SHADOW MODE - CAL-EXP-3 optional)
        # Reference: docs/system_law/P5_Structural_Drill_Package.md
        structural_drill_reference = detect_structural_drill_artifact(run_dir)

        # Detect P5 topology auditor report (SHADOW MODE)
        # Reference: docs/system_law/Topology_Bundle_PhaseX_Requirements.md Section 10
        p5_topology_auditor_reference = detect_p5_topology_auditor_report(run_dir)

        # Detect P5 divergence diagnostic (SHADOW MODE)
        # Reference: docs/system_law/P5_Divergence_Diagnostic_Panel_Spec.md
        p5_diagnostic_reference = detect_p5_diagnostic_file(run_dir)

        # Detect Budget Calibration artifact (SHADOW MODE - Phase X Budget Doctrine)
        # Reference: docs/system_law/Budget_PhaseX_Doctrine.md Section 7.3
        budget_calibration_reference = detect_budget_calibration(run_dir)

        # Detect P5 divergence real report (SHADOW MODE)
        # Reference: docs/system_law/schemas/p5/p5_divergence_real.schema.json
        p5_divergence_reference = detect_p5_divergence_file(run_dir)

        # Detect CTRPK compact artifact (SHADOW MODE)
        # Reference: docs/system_law/Curriculum_PhaseX_Invariants.md Section 12
        from backend.topology.first_light.ctrpk_detection import (
            detect_ctrpk_artifact,
            attach_ctrpk_to_manifest,
        )
        ctrpk_reference = detect_ctrpk_artifact(run_dir)

        # Detect NCI P5 artifacts (SHADOW MODE)
        # Reference: docs/system_law/NCI_PhaseX_Spec.md Section 11
        nci_p5_reference = detect_nci_p5_artifacts(run_dir)

        # Detect P5 identity preflight artifact (SHADOW MODE)
        # Reference: docs/system_law/Identity_Preflight_Precedence_Law.md
        p5_identity_preflight_reference = detect_p5_identity_preflight_file(run_dir)

        # Generate manifest
        manifest = self._generate_manifest(
            bundle_id=bundle_id,
            artifacts=artifacts,
            completeness=completeness,
            merkle_root=merkle_root,
            run_dir=run_dir,
            p3_run_id=p3_run_id,
            p4_run_id=p4_run_id,
            status_reference=status_reference,
            pathology_annotation=pathology_annotation,
            p5_replay_governance_reference=p5_replay_governance_reference,
            p5_pattern_tags_reference=p5_pattern_tags_reference,
            structural_drill_reference=structural_drill_reference,
            p5_topology_auditor_reference=p5_topology_auditor_reference,
            p5_diagnostic_reference=p5_diagnostic_reference,
            budget_calibration_reference=budget_calibration_reference,
            nci_p5_reference=nci_p5_reference,
            p5_divergence_reference=p5_divergence_reference,
            p5_identity_preflight_reference=p5_identity_preflight_reference,
        )

        if proof_snapshot_manifest:
            manifest["proof_log_snapshot"] = proof_snapshot_manifest

        # Attach CTRPK to manifest if detected (SHADOW MODE)
        if ctrpk_reference:
            manifest = attach_ctrpk_to_manifest(manifest, ctrpk_reference)

        # Write manifest
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / "manifest.json"

        try:
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
        except OSError as e:
            errors.append(f"Failed to write manifest: {e}")
            return EvidencePackResult(
                success=False,
                bundle_id=bundle_id,
                manifest_path=None,
                merkle_root=merkle_root,
                artifacts=artifacts,
                completeness=completeness,
                governance_advisories=advisories,
                errors=errors,
                warnings=warnings,
            )

        return EvidencePackResult(
            success=True,
            bundle_id=bundle_id,
            manifest_path=str(manifest_path),
            merkle_root=f"sha256:{merkle_root}",
            artifacts=artifacts,
            completeness=completeness,
            governance_advisories=advisories,
            errors=errors,
            warnings=warnings,
            status_reference=status_reference,
            p5_replay_governance_reference=p5_replay_governance_reference,
            p5_pattern_tags_reference=p5_pattern_tags_reference,
            structural_drill_reference=structural_drill_reference,
            p5_topology_auditor_reference=p5_topology_auditor_reference,
            p5_diagnostic_reference=p5_diagnostic_reference,
            ctrpk_reference=ctrpk_reference,
            nci_p5_reference=nci_p5_reference,
        )


# =============================================================================
# Public API
# =============================================================================

def build_evidence_pack(
    run_dir: str | Path,
    output_dir: Optional[str | Path] = None,
    schemas_dir: Optional[str | Path] = None,
    validate_schemas: bool = True,
    p3_run_id: Optional[str] = None,
    p4_run_id: Optional[str] = None,
    include_proof_snapshot: bool = False,
    proof_log_path: Optional[str | Path] = None,
    proof_snapshot_output: Optional[str | Path] = None,
) -> EvidencePackResult:
    """
    Build an evidence pack from a First Light run directory.

    This is the main entry point for evidence pack generation.

    Args:
        run_dir: Path to the run directory containing artifacts.
        output_dir: Optional output directory for manifest. Defaults to run_dir.
        schemas_dir: Optional path to schema files.
        validate_schemas: Whether to validate against JSON schemas.
        p3_run_id: Optional P3 run identifier for manifest.
        p4_run_id: Optional P4 run identifier for manifest.
        include_proof_snapshot: Whether to emit compliance/proof_log_snapshot.json (also
            toggleable via FIRST_LIGHT_INCLUDE_PROOF_SNAPSHOT env var).
        proof_log_path: Optional explicit proof log path override.
        proof_snapshot_output: Optional explicit snapshot path override.

    Returns:
        EvidencePackResult with build status, manifest path, and diagnostics.

    Example:
        >>> result = build_evidence_pack("results/first_light/run_123")
        >>> if result.success:
        ...     print(f"Bundle ID: {result.bundle_id}")
        ...     print(f"Merkle root: {result.merkle_root}")
    """
    builder = EvidencePackBuilder(
        schemas_dir=Path(schemas_dir) if schemas_dir else None,
        validate_schemas=validate_schemas,
    )

    return builder.build_evidence_pack(
        run_dir=Path(run_dir),
        output_dir=Path(output_dir) if output_dir else None,
        p3_run_id=p3_run_id,
        p4_run_id=p4_run_id,
        include_proof_snapshot=include_proof_snapshot,
        proof_log_path=proof_log_path,
        proof_snapshot_output=proof_snapshot_output,
    )


def attach_identity_preflight_to_evidence(
    evidence: Dict[str, Any],
    identity_preflight: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Attach P5 identity pre-flight check results to evidence pack.

    SHADOW MODE CONTRACT:
    - This function is non-mutating and observational only
    - It creates a copy with identity_preflight attached
    - Identity check results are advisory; they do not gate evidence validity

    Args:
        evidence: Evidence pack dict (manifest or run_config)
        identity_preflight: P5 identity pre-flight check result from
            check_p5_identity_alignment()

    Returns:
        New dict with identity_preflight attached under governance.slice_identity.p5_preflight

    Example:
        >>> evidence = {"governance": {}}
        >>> preflight = {"status": "OK", "fingerprint_match": True}
        >>> updated = attach_identity_preflight_to_evidence(evidence, preflight)
        >>> updated["governance"]["slice_identity"]["p5_preflight"]["status"]
        'OK'
    """
    import copy
    result = copy.deepcopy(evidence)

    if "governance" not in result:
        result["governance"] = {}

    if "slice_identity" not in result["governance"]:
        result["governance"]["slice_identity"] = {}

    preflight_summary = {
        k: v for k, v in identity_preflight.items()
        if k != "full_report"
    }

    preflight_summary["mode"] = "SHADOW"
    preflight_summary["shadow_mode_contract"] = {
        "observational_only": True,
        "no_control_flow_influence": True,
        "no_governance_modification": True,
        "advisory_status": preflight_summary.get("status", "UNKNOWN"),
    }

    result["governance"]["slice_identity"]["p5_preflight"] = preflight_summary
    return result


def load_identity_preflight_from_run_config(run_config_path: str | Path) -> Optional[Dict[str, Any]]:
    """
    Load identity_preflight from a run_config.json file.

    SHADOW MODE CONTRACT:
    - This function is purely observational
    - It extracts identity preflight data for evidence attachment

    Args:
        run_config_path: Path to run_config.json

    Returns:
        identity_preflight dict if present, None otherwise
    """
    config_path = Path(run_config_path)
    if not config_path.exists():
        return None

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return config.get("identity_preflight")
    except (json.JSONDecodeError, OSError):
        return None


def detect_identity_preflight(run_dir: Path) -> Optional[Dict[str, Any]]:
    """
    Detect identity_preflight from dedicated file or run_config.json.

    SHADOW MODE CONTRACT:
    - This function is purely observational
    - It searches for identity preflight data to include in evidence pack

    Detection Priority (prefer dedicated file for reproducible audits):
    1. p5_identity_preflight.json (dedicated artifact file - preferred)
    2. identity_preflight.json (legacy separate file)
    3. run_config.json (identity_preflight field - embedded)

    Args:
        run_dir: Path to run directory

    Returns:
        identity_preflight dict if found, None otherwise
    """
    # Priority 1: Dedicated p5_identity_preflight.json artifact (preferred)
    p5_preflight_path = run_dir / "p5_identity_preflight.json"
    if p5_preflight_path.exists():
        try:
            with open(p5_preflight_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    # Priority 2: Legacy identity_preflight.json
    preflight_path = run_dir / "identity_preflight.json"
    if preflight_path.exists():
        try:
            with open(preflight_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    # Priority 3: Embedded in run_config.json
    run_config_path = run_dir / "run_config.json"
    preflight = load_identity_preflight_from_run_config(run_config_path)
    if preflight:
        return preflight

    return None


def detect_nci_p5_artifacts(run_dir: Path) -> Optional[NciP5Reference]:
    """
    Detect and extract reference to NCI P5 artifacts if present.

    SHADOW MODE CONTRACT:
    - This function is purely observational
    - Detection does not affect pack generation success/failure
    - Provides documentation consistency metrics for governance signals
    - All outputs are advisory only

    Detection Priority (prefer signal file for compact GGFL consumption):
    1. nci_p5_signal.json (compact signal - preferred)
    2. nci_p5_result.json (full result - fallback)

    Expected locations:
    - Root: nci_p5_signal.json, nci_p5_result.json
    - Calibration subdirectory: calibration/nci_p5_signal.json, calibration/nci_p5_result.json

    Args:
        run_dir: Path to the run directory or evidence pack directory.

    Returns:
        NciP5Reference if any NCI P5 artifact exists, None otherwise.

    See docs/system_law/NCI_PhaseX_Spec.md Section 11.
    """
    # Search paths for signal file (preferred - compact)
    signal_paths = [
        run_dir / NCI_P5_SIGNAL_ARTIFACT,
        run_dir / NCI_P5_SUBDIR / NCI_P5_SIGNAL_ARTIFACT,
    ]

    # Search paths for result file (full result - fallback)
    result_paths = [
        run_dir / NCI_P5_RESULT_ARTIFACT,
        run_dir / NCI_P5_SUBDIR / NCI_P5_RESULT_ARTIFACT,
    ]

    # Try to find signal file first (compact, preferred for GGFL)
    signal_path: Optional[Path] = None
    for sp in signal_paths:
        if sp.exists():
            signal_path = sp
            break

    # Also check for result file (may exist alongside or instead of signal)
    result_path: Optional[Path] = None
    for rp in result_paths:
        if rp.exists():
            result_path = rp
            break

    # No artifacts found
    if signal_path is None and result_path is None:
        return None

    # Use signal file as primary source if available
    primary_path = signal_path if signal_path else result_path
    assert primary_path is not None  # We know at least one exists

    try:
        # Compute hash of primary file
        file_hash = compute_file_hash(primary_path)

        # Extract key fields from primary file
        with open(primary_path, "r", encoding="utf-8") as f:
            nci_data = json.load(f)

        # Handle non-dict JSON (null, array, etc.)
        if not isinstance(nci_data, dict):
            nci_data = {}

        # Determine relative path
        try:
            rel_path = str(primary_path.relative_to(run_dir)).replace("\\", "/")
        except ValueError:
            rel_path = primary_path.name

        # Extract fields from signal or result
        # Signal file has flat structure; result file has nested structure
        if signal_path:
            # Signal file (flat structure)
            nci_mode = nci_data.get("mode")
            global_nci = nci_data.get("global_nci")
            confidence = nci_data.get("confidence")
            slo_status = nci_data.get("slo_status")
            recommendation = nci_data.get("recommendation")
            tcl_aligned = nci_data.get("tcl_aligned", True)
            sic_aligned = nci_data.get("sic_aligned", True)
            tcl_violation_count = nci_data.get("tcl_violation_count", 0)
            sic_violation_count = nci_data.get("sic_violation_count", 0)
            warning_count = nci_data.get("warning_count", 0)
            shadow_mode = nci_data.get("shadow_mode", True)
            schema_version = nci_data.get("schema_version", "1.0.0")
        else:
            # Result file (nested structure)
            nci_mode = nci_data.get("mode")
            global_nci = nci_data.get("global_nci")
            confidence = nci_data.get("confidence")
            slo_eval = nci_data.get("slo_evaluation", {})
            slo_status = slo_eval.get("status")
            gov_signal = nci_data.get("governance_signal", {})
            recommendation = gov_signal.get("recommendation")
            tcl_result = nci_data.get("tcl_result", {})
            sic_result = nci_data.get("sic_result", {})
            tcl_aligned = tcl_result.get("aligned", True)
            sic_aligned = sic_result.get("aligned", True)
            tcl_violation_count = len(tcl_result.get("violations", []))
            sic_violation_count = len(sic_result.get("violations", []))
            warning_count = len(nci_data.get("warnings", []))
            shadow_mode = nci_data.get("shadow_mode", True)
            schema_version = nci_data.get("schema_version", "1.0.0")

        # Determine detection_path from relative path
        detection_path_value = "calibration" if NCI_P5_SUBDIR in rel_path else "root"

        # Determine extraction_source based on which file was used
        # MANIFEST_SIGNAL: signal file was found and used
        # MANIFEST_RESULT: result file was found (no signal file)
        if signal_path:
            extraction_source_value = "MANIFEST_SIGNAL"
        else:
            extraction_source_value = "MANIFEST_RESULT"

        # Build reference with optional result file cross-reference
        ref = NciP5Reference(
            path=rel_path,
            sha256=file_hash,
            schema_version=schema_version,
            mode=nci_mode,
            global_nci=global_nci,
            confidence=confidence,
            slo_status=slo_status,
            recommendation=recommendation,
            tcl_aligned=tcl_aligned,
            sic_aligned=sic_aligned,
            tcl_violation_count=tcl_violation_count,
            sic_violation_count=sic_violation_count,
            warning_count=warning_count,
            shadow_mode=shadow_mode,
            detection_path=detection_path_value,
            extraction_source=extraction_source_value,
        )

        # If signal file used, cross-reference result file if it also exists
        if signal_path and result_path:
            try:
                result_rel_path = str(result_path.relative_to(run_dir)).replace("\\", "/")
            except ValueError:
                result_rel_path = result_path.name
            ref.result_path = result_rel_path
            ref.result_sha256 = compute_file_hash(result_path)

        return ref

    except (json.JSONDecodeError, OSError):
        # File exists but couldn't be parsed - still reference it with error
        try:
            file_hash = compute_file_hash(primary_path)
            try:
                rel_path = str(primary_path.relative_to(run_dir)).replace("\\", "/")
            except ValueError:
                rel_path = primary_path.name
            # Determine extraction source for malformed file
            extraction_src = "MANIFEST_SIGNAL" if signal_path else "MANIFEST_RESULT"
            detection_path_val = "calibration" if NCI_P5_SUBDIR in rel_path else "root"
            return NciP5Reference(
                path=rel_path,
                sha256=file_hash,
                shadow_mode=True,
                extraction_source=extraction_src,
                detection_path=detection_path_val,
            )
        except OSError:
            return None


def verify_merkle_root(manifest_path: str | Path) -> Tuple[bool, str]:
    """
    Verify the Merkle root in a manifest matches the artifacts.

    Args:
        manifest_path: Path to manifest.json file.

    Returns:
        (is_valid, message) tuple.
    """
    manifest_path = Path(manifest_path)

    if not manifest_path.exists():
        return False, f"Manifest not found: {manifest_path}"

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return False, f"Failed to load manifest: {e}"

    # Extract artifact hashes
    artifacts = manifest.get("artifacts", [])
    artifact_hashes = [a["sha256"] for a in artifacts]

    # Recompute Merkle root
    computed_root = compute_merkle_root(artifact_hashes)

    # Compare with stored root
    stored_root = manifest.get("cryptographic_root", "")
    if stored_root.startswith("sha256:"):
        stored_root = stored_root[7:]

    if computed_root == stored_root:
        return True, "Merkle root verified successfully"
    else:
        return False, f"Merkle root mismatch: computed={computed_root}, stored={stored_root}"
