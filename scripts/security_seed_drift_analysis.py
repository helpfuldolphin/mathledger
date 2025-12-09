#!/usr/bin/env python3
"""
security_seed_drift_analysis.py - Seed Drift vs Substrate Nondeterminism Diagnosis

PHASE II -- NOT RUN IN PHASE I

Implements differential diagnosis between SEED_DRIFT and SUBSTRATE_NONDETERMINISM
per U2_SECURITY_PLAYBOOK.md specifications.

Given manifests and seed schedules, this tool classifies replay disagreements
and outputs structured diagnosis reports.

This tool is diagnostic only - no automatic repair.
"""

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional


class Classification(Enum):
    SEED_DRIFT = "SEED_DRIFT"
    SUBSTRATE_NONDETERMINISM = "SUBSTRATE_NONDETERMINISM"
    UNKNOWN = "UNKNOWN"
    NO_DISAGREEMENT = "NO_DISAGREEMENT"


class SeedDriftCause(Enum):
    EXPLICIT_RESEED = "explicit_reseed"
    ENV_VAR_OVERRIDE = "env_var_override"
    MANIFEST_CORRUPTION = "manifest_corruption"
    IMPORT_SIDE_EFFECT = "import_side_effect"
    UNKNOWN = "unknown"


class SubstrateNondetCause(Enum):
    FLOATING_POINT = "floating_point"
    HASH_RANDOMIZATION = "hash_randomization"
    THREAD_SCHEDULING = "thread_scheduling"
    MEMORY_ALLOCATION = "memory_allocation"
    LIBRARY_NONDET = "library_nondet"
    UNKNOWN = "unknown"


@dataclass
class DimensionAnalysis:
    """
    Five-dimension comparison per playbook table.

    | Characteristic | Seed Drift | Substrate Nondeterminism |
    |----------------|------------|--------------------------|
    | Definition | PRNG seed changed | Same seed, different outputs |
    | Root Cause | Config error, bug, tampering | FP variance, threading, hash |
    | Detectability | Seed snapshots differ | Seeds match, outputs differ |
    | Reproducibility | Fixing seed restores | Same seed still varies |
    | Scope | Usually total | Often partial |
    """
    seeds_match: bool
    outputs_match: bool
    scope: str  # "total", "partial", "unknown"
    reproducibility: str  # "deterministic", "variant", "unknown"
    detectability: str  # Description of how detected


@dataclass
class SeedDriftDiagnosisReport:
    """Structured diagnosis report per playbook specification."""

    incident_id: str
    run_id: str
    detected_at: str

    # Classification result
    classification: str
    confidence: str  # "HIGH", "MEDIUM", "LOW"

    # Five-dimension analysis
    dimensions: dict

    # Seed comparison
    original_seed: Optional[str]
    replay_seed: Optional[str]
    master_seed_original: Optional[str]
    master_seed_replay: Optional[str]

    # Divergence details
    divergence_cycle: Optional[int]
    divergence_type: Optional[str]

    # Probable cause (if determinable)
    probable_cause: Optional[str]
    cause_category: str
    evidence: list

    # Resolution guidance
    resolution_steps: list
    preventive_measures: list

    # Metadata
    analyzer_version: str = "1.0.0"
    playbook_reference: str = "U2_SECURITY_PLAYBOOK.md#seed-drift-vs-replay-disagreement"


def generate_incident_id(run_id: str, timestamp: datetime) -> str:
    """Generate deterministic incident ID."""
    ts_str = timestamp.strftime("%Y%m%d_%H%M%S")
    return f"DRIFT_{ts_str}_{run_id[:8]}"


def load_manifest(filepath: Path) -> dict:
    """Load manifest file (YAML or JSON)."""
    if not filepath.exists():
        raise FileNotFoundError(f"Manifest not found: {filepath}")

    content = filepath.read_text()

    # Try JSON first
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Try YAML
    try:
        import yaml
        return yaml.safe_load(content)
    except ImportError:
        # YAML not available, try simple key: value parsing
        result = {}
        for line in content.splitlines():
            if ':' in line and not line.strip().startswith('#'):
                key, _, value = line.partition(':')
                result[key.strip()] = value.strip()
        return result


def load_seed_schedule(filepath: Path) -> dict:
    """Load seed schedule file."""
    if not filepath.exists():
        return {}

    content = filepath.read_text()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        try:
            import yaml
            return yaml.safe_load(content)
        except ImportError:
            return {}


def extract_seed_values(manifest: dict) -> tuple[Optional[str], Optional[str]]:
    """Extract PRNG seed and master seed from manifest."""
    prng_seed = manifest.get("prng_seed") or manifest.get("prng_seed_start")
    master_seed = manifest.get("u2_master_seed") or manifest.get("master_seed")

    # Convert to string for comparison
    if prng_seed is not None:
        prng_seed = str(prng_seed)
    if master_seed is not None:
        master_seed = str(master_seed)

    return prng_seed, master_seed


def compare_seeds(original_manifest: dict, replay_manifest: dict) -> tuple[bool, bool]:
    """
    Compare seeds between original and replay manifests.
    Returns (prng_seeds_match, master_seeds_match).
    """
    orig_prng, orig_master = extract_seed_values(original_manifest)
    replay_prng, replay_master = extract_seed_values(replay_manifest)

    prng_match = orig_prng == replay_prng
    master_match = orig_master == replay_master

    return prng_match, master_match


def analyze_divergence_scope(replay_receipt: Optional[dict]) -> str:
    """
    Analyze scope of divergence.

    Per playbook:
    - Seed drift: "Usually total (affects all random choices)"
    - Substrate nondet: "Often partial (specific operations)"
    """
    if replay_receipt is None:
        return "unknown"

    cycles_total = replay_receipt.get("cycles_replayed", 0)
    cycles_matched = replay_receipt.get("cycles_matched", 0)
    divergence_point = replay_receipt.get("divergence_point")

    if cycles_total == 0:
        return "unknown"

    if cycles_matched == cycles_total:
        return "none"  # No divergence

    if divergence_point == 0:
        return "total"

    match_ratio = cycles_matched / cycles_total
    if match_ratio < 0.2:
        return "total"
    elif match_ratio < 0.8:
        return "partial"
    else:
        return "minimal"


def determine_classification(
    seeds_match: bool,
    outputs_match: bool,
    master_seeds_match: bool
) -> tuple[Classification, str]:
    """
    Apply playbook decision matrix.

    | Seeds Match? | Outputs Match? | Diagnosis | Severity |
    |--------------|----------------|-----------|----------|
    | Yes | Yes | Valid replay | N/A |
    | Yes | No | SUBSTRATE_NONDET | HIGH |
    | No | No | SEED_DRIFT | CRITICAL |
    | No | Yes | Coincidental (investigate) | MEDIUM |
    """
    if outputs_match:
        if seeds_match:
            return Classification.NO_DISAGREEMENT, "HIGH"
        else:
            # Seeds differ but outputs match - suspicious
            return Classification.UNKNOWN, "MEDIUM"

    # Outputs don't match
    if seeds_match and master_seeds_match:
        return Classification.SUBSTRATE_NONDETERMINISM, "HIGH"
    elif not seeds_match or not master_seeds_match:
        return Classification.SEED_DRIFT, "HIGH"
    else:
        return Classification.UNKNOWN, "LOW"


def identify_seed_drift_cause(
    original_manifest: dict,
    replay_manifest: dict,
    seed_schedule: Optional[dict]
) -> tuple[SeedDriftCause, list[str]]:
    """
    Attempt to identify the cause of seed drift.
    Returns (cause, evidence).
    """
    evidence = []

    orig_prng, orig_master = extract_seed_values(original_manifest)
    replay_prng, replay_master = extract_seed_values(replay_manifest)

    # Check for explicit reseed
    if original_manifest.get("prng_seed_start") != original_manifest.get("prng_seed_end"):
        evidence.append("Original manifest shows seed changed during run (start != end)")
        return SeedDriftCause.EXPLICIT_RESEED, evidence

    # Check for env var differences
    orig_env = original_manifest.get("environment", {})
    replay_env = replay_manifest.get("environment", {})

    if orig_env.get("PYTHONHASHSEED") != replay_env.get("PYTHONHASHSEED"):
        evidence.append(f"PYTHONHASHSEED differs: {orig_env.get('PYTHONHASHSEED')} vs {replay_env.get('PYTHONHASHSEED')}")
        return SeedDriftCause.ENV_VAR_OVERRIDE, evidence

    # Check for manifest hash mismatch (indicates corruption/substitution)
    orig_hash = original_manifest.get("manifest_hash")
    replay_hash = replay_manifest.get("manifest_hash")

    if orig_hash and replay_hash and orig_hash != replay_hash:
        evidence.append(f"Manifest hash mismatch: {orig_hash[:16]}... vs {replay_hash[:16]}...")
        return SeedDriftCause.MANIFEST_CORRUPTION, evidence

    # Check seed schedule
    if seed_schedule:
        scheduled_seed = seed_schedule.get("prng_seed")
        if scheduled_seed and str(scheduled_seed) != orig_prng:
            evidence.append(f"Original seed {orig_prng} differs from scheduled {scheduled_seed}")
            return SeedDriftCause.MANIFEST_CORRUPTION, evidence

    # No specific cause identified
    evidence.append("Seed values differ but specific cause not determined")
    evidence.append(f"Original PRNG seed: {orig_prng}")
    evidence.append(f"Replay PRNG seed: {replay_prng}")

    return SeedDriftCause.UNKNOWN, evidence


def identify_substrate_nondet_cause(
    original_manifest: dict,
    replay_manifest: dict,
    replay_receipt: Optional[dict]
) -> tuple[SubstrateNondetCause, list[str]]:
    """
    Attempt to identify the cause of substrate nondeterminism.
    Returns (cause, evidence).
    """
    evidence = []

    divergence_type = None
    if replay_receipt:
        divergence_type = replay_receipt.get("divergence_type")

    # Check based on divergence type hints
    if divergence_type:
        if "float" in divergence_type.lower() or "numeric" in divergence_type.lower():
            evidence.append(f"Divergence type indicates floating-point: {divergence_type}")
            return SubstrateNondetCause.FLOATING_POINT, evidence

        if "order" in divergence_type.lower():
            evidence.append(f"Divergence type indicates ordering: {divergence_type}")
            return SubstrateNondetCause.HASH_RANDOMIZATION, evidence

        if "timing" in divergence_type.lower() or "thread" in divergence_type.lower():
            evidence.append(f"Divergence type indicates timing: {divergence_type}")
            return SubstrateNondetCause.THREAD_SCHEDULING, evidence

    # Check for PYTHONHASHSEED
    orig_env = original_manifest.get("environment", {})
    replay_env = replay_manifest.get("environment", {})

    orig_hashseed = orig_env.get("PYTHONHASHSEED")
    replay_hashseed = replay_env.get("PYTHONHASHSEED")

    if orig_hashseed != "0" or replay_hashseed != "0":
        evidence.append(f"PYTHONHASHSEED not fixed to 0: original={orig_hashseed}, replay={replay_hashseed}")
        return SubstrateNondetCause.HASH_RANDOMIZATION, evidence

    # Check for platform differences
    orig_platform = original_manifest.get("platform", {})
    replay_platform = replay_manifest.get("platform", {})

    if orig_platform and replay_platform:
        if orig_platform.get("python_version") != replay_platform.get("python_version"):
            evidence.append(f"Python version differs: {orig_platform.get('python_version')} vs {replay_platform.get('python_version')}")
            return SubstrateNondetCause.LIBRARY_NONDET, evidence

        if orig_platform.get("cpu") != replay_platform.get("cpu"):
            evidence.append(f"CPU differs: may cause floating-point variance")
            return SubstrateNondetCause.FLOATING_POINT, evidence

    evidence.append("Seeds match but outputs differ - substrate nondeterminism suspected")
    return SubstrateNondetCause.UNKNOWN, evidence


def get_resolution_steps(classification: Classification, cause_category: str) -> list[str]:
    """Get resolution steps per playbook guidance."""

    if classification == Classification.SEED_DRIFT:
        cause = SeedDriftCause(cause_category) if cause_category in [e.value for e in SeedDriftCause] else SeedDriftCause.UNKNOWN

        if cause == SeedDriftCause.EXPLICIT_RESEED:
            return [
                "Identify and fix code that calls random.seed() after initialization",
                "Add reseed detection assertion to runner",
                "Re-run experiment with fixed code"
            ]
        elif cause == SeedDriftCause.ENV_VAR_OVERRIDE:
            return [
                "Restore correct environment variables",
                "Lock env vars at runner init",
                "Re-run experiment with correct environment"
            ]
        elif cause == SeedDriftCause.MANIFEST_CORRUPTION:
            return [
                "Restore manifest from version control",
                "Verify manifest hash before run",
                "Re-run experiment with correct manifest"
            ]
        elif cause == SeedDriftCause.IMPORT_SIDE_EFFECT:
            return [
                "Audit import chain for random state initialization",
                "Isolate problematic imports",
                "Re-run experiment with imports controlled"
            ]
        else:
            return [
                "Investigate seed value sources",
                "Check all configuration files",
                "Re-run with explicit seed verification"
            ]

    elif classification == Classification.SUBSTRATE_NONDETERMINISM:
        cause = SubstrateNondetCause(cause_category) if cause_category in [e.value for e in SubstrateNondetCause] else SubstrateNondetCause.UNKNOWN

        if cause == SubstrateNondetCause.FLOATING_POINT:
            return [
                "Use tolerant comparison for numeric values",
                "Normalize floating-point to fixed precision",
                "Consider deterministic math libraries"
            ]
        elif cause == SubstrateNondetCause.HASH_RANDOMIZATION:
            return [
                "Set PYTHONHASHSEED=0 in runner init",
                "Use OrderedDict instead of dict where order matters",
                "Sort collections before processing"
            ]
        elif cause == SubstrateNondetCause.THREAD_SCHEDULING:
            return [
                "Force sequential execution in critical path",
                "Remove concurrency from determinism-sensitive code",
                "Add synchronization barriers"
            ]
        elif cause == SubstrateNondetCause.MEMORY_ALLOCATION:
            return [
                "Pin allocator behavior if possible",
                "Use deterministic allocator",
                "Avoid address-dependent comparisons"
            ]
        elif cause == SubstrateNondetCause.LIBRARY_NONDET:
            return [
                "Audit and approve all dependencies",
                "Patch or replace nondeterministic libraries",
                "Pin library versions exactly"
            ]
        else:
            return [
                "Investigate divergence patterns",
                "Profile execution to identify variance source",
                "Consider platform standardization"
            ]

    else:
        return [
            "Collect additional diagnostic information",
            "Manual investigation required",
            "Review both manifests in detail"
        ]


def get_preventive_measures(classification: Classification, cause_category: str) -> list[str]:
    """Get preventive measures per playbook guidance."""

    if classification == Classification.SEED_DRIFT:
        return [
            "Add reseed detection assertion to FrozenRandom wrapper",
            "Lock environment variables at runner initialization",
            "Compute and verify manifest hash before every run",
            "Audit dependency chain for random state side effects"
        ]
    elif classification == Classification.SUBSTRATE_NONDETERMINISM:
        return [
            "Set PYTHONHASHSEED=0 in all execution environments",
            "Use deterministic data structures (sorted, OrderedDict)",
            "Avoid concurrency in determinism-critical paths",
            "Standardize execution platform (Docker, pinned versions)"
        ]
    else:
        return [
            "Implement comprehensive seed tracking",
            "Add replay validation to CI pipeline",
            "Document all sources of randomness"
        ]


def analyze_seed_drift(
    original_manifest: dict,
    replay_manifest: dict,
    replay_receipt: Optional[dict] = None,
    seed_schedule: Optional[dict] = None,
    run_id: Optional[str] = None
) -> SeedDriftDiagnosisReport:
    """
    Main analysis function implementing playbook diagnostic logic.
    """
    # Extract run ID
    if run_id is None:
        run_id = original_manifest.get("run_id", "UNKNOWN")

    # Generate deterministic timestamp and incident ID
    timestamp = datetime.now(timezone.utc)
    incident_id = generate_incident_id(run_id, timestamp)

    # Extract seed values
    orig_prng, orig_master = extract_seed_values(original_manifest)
    replay_prng, replay_master = extract_seed_values(replay_manifest)

    # Compare seeds
    prng_match, master_match = compare_seeds(original_manifest, replay_manifest)
    seeds_match = prng_match and master_match

    # Determine if outputs match (from replay receipt)
    outputs_match = True
    divergence_cycle = None
    divergence_type = None

    if replay_receipt:
        status = replay_receipt.get("status", "").upper()
        outputs_match = status == "PASS"
        divergence_cycle = replay_receipt.get("divergence_point")
        divergence_type = replay_receipt.get("divergence_type")

    # Analyze scope
    scope = analyze_divergence_scope(replay_receipt)

    # Build dimension analysis
    dimensions = DimensionAnalysis(
        seeds_match=seeds_match,
        outputs_match=outputs_match,
        scope=scope,
        reproducibility="deterministic" if outputs_match else "variant",
        detectability="Seed comparison" if not seeds_match else "Output comparison"
    )

    # Classify per playbook decision matrix
    classification, confidence = determine_classification(seeds_match, outputs_match, master_match)

    # Identify probable cause
    evidence = []
    if classification == Classification.SEED_DRIFT:
        cause_enum, evidence = identify_seed_drift_cause(original_manifest, replay_manifest, seed_schedule)
        cause_category = cause_enum.value
    elif classification == Classification.SUBSTRATE_NONDETERMINISM:
        cause_enum, evidence = identify_substrate_nondet_cause(original_manifest, replay_manifest, replay_receipt)
        cause_category = cause_enum.value
    else:
        cause_category = "unknown"
        evidence = ["Insufficient information for cause determination"]

    # Get resolution and prevention guidance
    resolution_steps = get_resolution_steps(classification, cause_category)
    preventive_measures = get_preventive_measures(classification, cause_category)

    return SeedDriftDiagnosisReport(
        incident_id=incident_id,
        run_id=run_id,
        detected_at=timestamp.isoformat(),
        classification=classification.value,
        confidence=confidence,
        dimensions=asdict(dimensions),
        original_seed=orig_prng,
        replay_seed=replay_prng,
        master_seed_original=orig_master,
        master_seed_replay=replay_master,
        divergence_cycle=divergence_cycle,
        divergence_type=divergence_type,
        probable_cause=cause_category if cause_category != "unknown" else None,
        cause_category=cause_category,
        evidence=evidence,
        resolution_steps=resolution_steps,
        preventive_measures=preventive_measures,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Seed Drift vs Substrate Nondeterminism Diagnosis Tool",
        epilog="See U2_SECURITY_PLAYBOOK.md for diagnostic procedures."
    )
    parser.add_argument(
        "--original-manifest",
        type=Path,
        required=True,
        help="Path to original run manifest"
    )
    parser.add_argument(
        "--replay-manifest",
        type=Path,
        required=True,
        help="Path to replay run manifest"
    )
    parser.add_argument(
        "--replay-receipt",
        type=Path,
        help="Path to replay receipt JSON (optional)"
    )
    parser.add_argument(
        "--seed-schedule",
        type=Path,
        help="Path to seed schedule file (optional)"
    )
    parser.add_argument(
        "--run-id",
        help="Override run ID (optional)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("seed_drift_analysis.json"),
        help="Output path for diagnosis report"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console output"
    )

    args = parser.parse_args()

    try:
        # Load inputs
        original_manifest = load_manifest(args.original_manifest)
        replay_manifest = load_manifest(args.replay_manifest)

        replay_receipt = None
        if args.replay_receipt and args.replay_receipt.exists():
            with open(args.replay_receipt) as f:
                replay_receipt = json.load(f)

        seed_schedule = None
        if args.seed_schedule:
            seed_schedule = load_seed_schedule(args.seed_schedule)

        # Analyze
        report = analyze_seed_drift(
            original_manifest=original_manifest,
            replay_manifest=replay_manifest,
            replay_receipt=replay_receipt,
            seed_schedule=seed_schedule,
            run_id=args.run_id
        )

        # Output
        report_dict = asdict(report)

        with open(args.output, 'w') as f:
            json.dump(report_dict, f, indent=2)

        if not args.quiet:
            print(f"Seed Drift Analysis Report Generated: {args.output}")
            print(f"  Incident ID: {report.incident_id}")
            print(f"  Classification: {report.classification}")
            print(f"  Confidence: {report.confidence}")
            print(f"  Seeds Match: {report.dimensions['seeds_match']}")
            print(f"  Outputs Match: {report.dimensions['outputs_match']}")
            if report.probable_cause:
                print(f"  Probable Cause: {report.probable_cause}")
            if report.divergence_cycle is not None:
                print(f"  Divergence at cycle: {report.divergence_cycle}")

        # Exit code based on classification
        if report.classification == Classification.SEED_DRIFT.value:
            sys.exit(2)
        elif report.classification == Classification.SUBSTRATE_NONDETERMINISM.value:
            sys.exit(1)
        elif report.classification == Classification.UNKNOWN.value:
            sys.exit(3)
        else:
            sys.exit(0)

    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(4)
    except Exception as e:
        print(f"ERROR: Unexpected error - {e}", file=sys.stderr)
        sys.exit(4)


if __name__ == "__main__":
    main()
