"""
Noise vs Reality Integration for Evidence Pack

This module provides integration between the noise_vs_reality generator
and the Evidence Pack builder.

SHADOW MODE CONTRACT:
- All operations are observational only
- No governance modification or gating
- Optional: only generated if both P3 noise and P5 divergence are available

INPUT PREFERENCE ORDER:
1. p5_divergence_real.json (preferred - rich P5 report)
2. divergence_log.jsonl (fallback - raw cycle data)

P5_SOURCE ENUM (FROZEN):
- p5_real_validated: Real telemetry, RTTS-validated
- p5_suspected_mock: Real telemetry, suspected mock data
- p5_real_adapter: Real telemetry, unvalidated
- p5_jsonl_fallback: Fallback to divergence_log.jsonl
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# =============================================================================
# Extraction Source Enum (FROZEN)
# =============================================================================

class ExtractionSource(str, Enum):
    """
    FROZEN enumeration of signal extraction sources.

    Indicates where the noise_vs_reality signal was extracted from.
    """
    MANIFEST = "MANIFEST"
    EVIDENCE_JSON = "EVIDENCE_JSON"
    MISSING = "MISSING"


# Set of valid extraction_source values
VALID_EXTRACTION_SOURCES = frozenset(s.value for s in ExtractionSource)


# =============================================================================
# P5 Source Enum (FROZEN)
# =============================================================================

class P5Source(str, Enum):
    """
    FROZEN enumeration of valid p5_source values.

    Any unknown source values are coerced to JSONL_FALLBACK with advisory note.
    """
    REAL_VALIDATED = "p5_real_validated"
    SUSPECTED_MOCK = "p5_suspected_mock"
    REAL_ADAPTER = "p5_real_adapter"
    JSONL_FALLBACK = "p5_jsonl_fallback"


# Set of valid p5_source values for validation
VALID_P5_SOURCES = frozenset(s.value for s in P5Source)


def normalize_p5_source(raw_source: Optional[str]) -> Tuple[str, Optional[str]]:
    """
    Normalize and validate p5_source value.

    Args:
        raw_source: Raw source identifier (may be None or unknown)

    Returns:
        Tuple of (normalized_source, advisory_note or None)
        - If valid: (source, None)
        - If unknown/None: (p5_jsonl_fallback, advisory note)
    """
    if raw_source is None:
        return P5Source.JSONL_FALLBACK.value, "p5_source was None; coerced to fallback"

    if raw_source in VALID_P5_SOURCES:
        return raw_source, None

    # Unknown value - coerce to fallback with advisory
    return (
        P5Source.JSONL_FALLBACK.value,
        f"Unknown p5_source '{raw_source}' coerced to {P5Source.JSONL_FALLBACK.value}",
    )

# Import noise_vs_reality generator
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


def compute_summary_sha256(summary: Dict[str, Any]) -> str:
    """Compute SHA256 hash of noise_vs_reality_summary for manifest reference."""
    # Serialize with sorted keys for deterministic hashing
    serialized = json.dumps(summary, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _load_p5_divergence_real_report(run_dir: Path) -> Optional[Dict[str, Any]]:
    """
    Load p5_divergence_real.json if it exists.

    This is the PREFERRED source for P5 divergence data as it contains
    rich metadata including pattern classification, validation status,
    and divergence decomposition.

    Returns:
        Parsed report dict or None if not found/invalid
    """
    # Check multiple possible locations
    search_paths = [
        run_dir / "p5_divergence_real.json",
        run_dir / "p4_shadow" / "p5_divergence_real.json",
        run_dir / "governance" / "p5_divergence_real.json",
    ]

    for report_path in search_paths:
        if report_path.exists():
            try:
                with report_path.open("r", encoding="utf-8") as f:
                    report = json.load(f)
                # Validate required fields
                if (
                    report.get("schema_version")
                    and report.get("total_cycles")
                    and "divergence_rate" in report
                ):
                    return report
            except (OSError, json.JSONDecodeError):
                continue
    return None


def _extract_divergence_from_real_report(
    report: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str]:
    """
    Extract divergence series and red flags from p5_divergence_real.json.

    Since p5_divergence_real.json is a summary report (not cycle-level data),
    we synthesize the P5 summary input directly from it.

    Returns:
        (divergence_series, red_flags, telemetry_provider)
    """
    total_cycles = report.get("total_cycles", 100)
    divergence_rate = report.get("divergence_rate", 0.0)
    pattern = report.get("pattern_classification", "NOMINAL")

    # Synthesize representative divergence series from summary stats
    # This allows the comparison metrics to be computed accurately
    divergence_series: List[Dict[str, Any]] = []
    red_flags: List[Dict[str, Any]] = []

    # Extract decomposition if available for richer synthesis
    decomposition = report.get("divergence_decomposition", {})
    bias = decomposition.get("bias", 0.0)

    # Calculate how many divergent cycles based on rate
    divergent_count = int(total_cycles * divergence_rate)

    for i in range(total_cycles):
        is_divergent = i < divergent_count
        point = {
            "cycle": i,
            "twin_delta_p": 0.01,  # Nominal twin prediction
            "real_delta_p": 0.01 + (bias if is_divergent else 0.0),
            "divergence_magnitude": bias if is_divergent else 0.0,
            "is_red_flag": False,
        }

        # Mark red flags for pattern-specific cycles
        if pattern in ["STRUCTURAL_BREAK", "ATTRACTOR_MISS"] and is_divergent and i % 10 == 0:
            point["is_red_flag"] = True
            red_flags.append({
                "cycle": i,
                "type": f"PATTERN_{pattern}",
                "severity": "WARN",
                "description": f"Divergence pattern: {pattern}",
            })

        divergence_series.append(point)

    # Determine telemetry provider from validation status (uses frozen enum)
    validation_status = report.get("validation_status", "UNVALIDATED")
    if validation_status == "VALIDATED_REAL":
        telemetry_provider = P5Source.REAL_VALIDATED.value
    elif validation_status == "SUSPECTED_MOCK":
        telemetry_provider = P5Source.SUSPECTED_MOCK.value
    else:
        telemetry_provider = P5Source.REAL_ADAPTER.value

    return divergence_series, red_flags, telemetry_provider


def generate_noise_vs_reality_for_evidence(
    run_dir: Path,
) -> Optional[Dict[str, Any]]:
    """
    Generate noise_vs_reality_summary if P3 noise summary and P5 divergence data exist.

    SHADOW MODE CONTRACT:
    - This is purely observational comparison
    - No governance modification or gating
    - Optional: only generated if both P3 noise and P5 divergence are available

    INPUT PREFERENCE ORDER for P5 divergence:
    1. p5_divergence_real.json (preferred - rich P5 report with pattern classification)
    2. divergence_log.jsonl (fallback - raw cycle-level data)

    Args:
        run_dir: Path to the run directory containing artifacts

    Returns:
        noise_vs_reality_summary dict or None if prerequisites not met
    """
    if not HAS_NOISE_VS_REALITY:
        return None

    # Look for P3 noise summary in stability_report.json or governance/noise artifact
    noise_summary = None

    # Try stability_report.json first (embedded noise_summary)
    stability_path = run_dir / "stability_report.json"
    if stability_path.exists():
        try:
            with stability_path.open("r", encoding="utf-8") as f:
                stability = json.load(f)
            noise_summary = stability.get("noise_summary")
        except (OSError, json.JSONDecodeError):
            pass

    # Try dedicated noise_summary.json artifact
    if not noise_summary:
        noise_path = run_dir / "noise_summary.json"
        if noise_path.exists():
            try:
                with noise_path.open("r", encoding="utf-8") as f:
                    noise_summary = json.load(f)
            except (OSError, json.JSONDecodeError):
                pass

    if not noise_summary:
        return None

    # Look for P5 divergence data
    # PREFERENCE ORDER: p5_divergence_real.json > divergence_log.jsonl
    divergence_series: List[Dict[str, Any]] = []
    red_flags: List[Dict[str, Any]] = []
    telemetry_provider = P5Source.REAL_ADAPTER.value
    p5_source: Optional[str] = None  # Track which source was used
    p5_source_advisory: Optional[str] = None  # Advisory note for coercion

    # PREFERRED: Try p5_divergence_real.json first
    p5_real_report = _load_p5_divergence_real_report(run_dir)
    if p5_real_report:
        divergence_series, red_flags, telemetry_provider = _extract_divergence_from_real_report(
            p5_real_report
        )
        # telemetry_provider already set from validation_status (enum value)
        p5_source = telemetry_provider

    # FALLBACK: Try divergence_log.jsonl if no real report
    if not divergence_series:
        divergence_log_path = run_dir / "p4_shadow" / "divergence_log.jsonl"
        if not divergence_log_path.exists():
            divergence_log_path = run_dir / "divergence_log.jsonl"

        if divergence_log_path.exists():
            try:
                with divergence_log_path.open("r", encoding="utf-8") as f:
                    for i, line in enumerate(f):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                            # Convert to scatter point format
                            twin_pred = entry.get("twin_prediction", {})
                            real_obs = entry.get("real_observation", {})
                            point = {
                                "cycle": entry.get("cycle", i),
                                "twin_delta_p": entry.get("twin_delta_p", twin_pred.get("delta_p", 0.0)),
                                "real_delta_p": entry.get("real_delta_p", real_obs.get("delta_p", 0.0)),
                                "divergence_magnitude": entry.get("divergence_magnitude", entry.get("divergence", 0.0)),
                                "is_red_flag": entry.get("is_red_flag", entry.get("red_flag", False)),
                            }
                            divergence_series.append(point)

                            # Collect red flags
                            if point["is_red_flag"]:
                                red_flags.append({
                                    "cycle": point["cycle"],
                                    "type": entry.get("red_flag_type", "DIVERGENCE"),
                                    "severity": entry.get("severity", "WARN"),
                                    "description": entry.get("description", ""),
                                })
                        except json.JSONDecodeError:
                            continue
                # Use frozen enum value for JSONL fallback
                p5_source = P5Source.JSONL_FALLBACK.value
                telemetry_provider = P5Source.JSONL_FALLBACK.value
            except OSError:
                pass

    if not divergence_series:
        return None

    # Generate noise_vs_reality_summary
    try:
        # Determine experiment ID from run config
        experiment_id = None
        config_path = run_dir / "run_config.json"
        if config_path.exists():
            try:
                with config_path.open("r", encoding="utf-8") as f:
                    config = json.load(f)
                experiment_id = config.get("run_id") or config.get("experiment_id")
            except (OSError, json.JSONDecodeError):
                pass

        nvr_summary = build_from_harness_and_divergence(
            noise_summary=noise_summary,
            divergence_series=divergence_series,
            red_flags=red_flags if red_flags else None,
            experiment_id=experiment_id,
            telemetry_provider=telemetry_provider,
        )

        # Normalize and validate p5_source (frozen enum)
        normalized_source, source_advisory = normalize_p5_source(p5_source)
        nvr_summary["_p5_source"] = normalized_source
        if source_advisory:
            nvr_summary["_p5_source_advisory"] = source_advisory

        # Validate the summary
        is_valid, errors = validate_noise_vs_reality_summary(nvr_summary)
        if not is_valid:
            # Still return but mark validation failed
            nvr_summary["_validation_errors"] = errors

        return nvr_summary

    except Exception:
        # Generation failed - return None, don't block pack creation
        return None


def attach_noise_vs_reality_to_manifest(
    manifest: Dict[str, Any],
    nvr_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Attach noise_vs_reality_summary to evidence pack manifest.

    Stores under manifest["governance"]["noise_vs_reality"].
    Includes sha256 hash of the full summary for integrity verification.

    SHADOW MODE CONTRACT:
    - This is purely observational attachment
    - Does not modify any governance decisions

    Args:
        manifest: Evidence pack manifest dict (modified in place)
        nvr_summary: noise_vs_reality_summary from generate_noise_vs_reality_for_evidence

    Returns:
        Modified manifest dict
    """
    if "governance" not in manifest:
        manifest["governance"] = {}

    # Compute sha256 of the full summary for integrity reference
    summary_sha256 = compute_summary_sha256(nvr_summary)

    # Get p5_source and normalize if needed
    raw_p5_source = nvr_summary.get("_p5_source")
    normalized_source, source_advisory = normalize_p5_source(raw_p5_source)

    # Extract top_factor fields from governance_advisory
    gov_advisory = nvr_summary.get("governance_advisory", {})

    manifest["governance"]["noise_vs_reality"] = {
        "schema_version": nvr_summary.get("schema_version", NOISE_VS_REALITY_SCHEMA_VERSION),
        "mode": nvr_summary.get("mode", "SHADOW"),
        "experiment_id": nvr_summary.get("experiment_id"),
        "verdict": nvr_summary.get("coverage_assessment", {}).get("verdict"),
        "coverage_ratio": nvr_summary.get("comparison_metrics", {}).get("coverage_ratio"),
        "advisory_severity": gov_advisory.get("severity"),
        "advisory_message": gov_advisory.get("message"),
        "top_factor": gov_advisory.get("top_factor"),
        "top_factor_value": gov_advisory.get("top_factor_value"),
        "p3_noise_rate": nvr_summary.get("p3_summary", {}).get("noise_event_rate"),
        "p5_divergence_rate": nvr_summary.get("p5_summary", {}).get("divergence_rate"),
        "p5_source": normalized_source,
        "p5_source_advisory": source_advisory,  # None if valid, note if coerced
        "summary_sha256": summary_sha256,
    }

    return manifest


def format_advisory_warning(
    verdict: str,
    top_factor: Optional[str],
    top_factor_value: Optional[float],
    p5_source: str,
) -> Optional[str]:
    """
    Format single-line advisory warning with (top_factor, top_factor_value, p5_source).

    STABLE FORMAT: "verdict: factor=value [source]"

    Args:
        verdict: Coverage verdict (ADEQUATE/MARGINAL/INSUFFICIENT)
        top_factor: Driving factor name (coverage_ratio or exceedance_rate)
        top_factor_value: Numeric value of top factor
        p5_source: P5 source enum value

    Returns:
        Single-line warning string or None if ADEQUATE
    """
    if verdict == "ADEQUATE":
        return None

    # Format factor value
    if top_factor == "coverage_ratio" and top_factor_value is not None:
        factor_str = f"coverage_ratio={top_factor_value:.2f}"
    elif top_factor == "exceedance_rate" and top_factor_value is not None:
        factor_str = f"exceedance_rate={top_factor_value * 100:.1f}%"
    else:
        factor_str = "unknown_factor"

    # Abbreviate p5_source for display
    source_abbrev = {
        P5Source.REAL_VALIDATED.value: "real",
        P5Source.SUSPECTED_MOCK.value: "mock?",
        P5Source.REAL_ADAPTER.value: "adapter",
        P5Source.JSONL_FALLBACK.value: "jsonl",
    }.get(p5_source, "unk")

    return f"{verdict}: {factor_str} [{source_abbrev}]"


def extract_noise_vs_reality_signal(
    manifest: Dict[str, Any],
    evidence_json_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Extract noise_vs_reality signal from evidence pack manifest for status generator.

    Returns signal suitable for signals.noise_vs_reality in first_light_status.json.

    EXTRACTION SOURCE PRIORITY:
    1. manifest["governance"]["noise_vs_reality"] -> MANIFEST
    2. evidence_json_path (if provided) -> EVIDENCE_JSON
    3. No data available -> MISSING

    Signal includes:
    - extraction_source: MANIFEST/EVIDENCE_JSON/MISSING
    - verdict: ADEQUATE/MARGINAL/INSUFFICIENT
    - advisory_severity: INFO/WARN
    - advisory_warning: Single-line format with (top_factor, top_factor_value, p5_source)
    - top_factor: coverage_ratio or exceedance_rate (driving factor for WARN)
    - top_factor_value: numeric value of top factor
    - p5_source: enum value (p5_real_validated, p5_suspected_mock, p5_real_adapter, p5_jsonl_fallback)
    - p5_source_advisory: coercion note if p5_source was normalized
    - summary_sha256: hash for integrity verification

    Args:
        manifest: Evidence pack manifest dict
        evidence_json_path: Optional path to evidence.json file for fallback

    Returns:
        Signal dict with full provenance (always returns dict, never None)
    """
    # Try manifest first
    governance = manifest.get("governance", {})
    nvr = governance.get("noise_vs_reality")
    extraction_source = ExtractionSource.MISSING.value

    if nvr:
        extraction_source = ExtractionSource.MANIFEST.value
    elif evidence_json_path and evidence_json_path.exists():
        # Fallback to evidence.json
        try:
            import json
            with evidence_json_path.open("r", encoding="utf-8") as f:
                evidence = json.load(f)
            nvr = evidence.get("governance", {}).get("noise_vs_reality")
            if nvr:
                extraction_source = ExtractionSource.EVIDENCE_JSON.value
        except (OSError, json.JSONDecodeError):
            pass

    # Return MISSING signal if no data
    if not nvr:
        return {
            "extraction_source": ExtractionSource.MISSING.value,
            "verdict": None,
            "advisory_severity": None,
            "advisory_warning": None,
            "advisory_message": None,
            "top_factor": None,
            "top_factor_value": None,
            "coverage_ratio": None,
            "p3_noise_rate": None,
            "p5_divergence_rate": None,
            "p5_source": None,
            "p5_source_advisory": None,
            "summary_sha256": None,
        }

    # Extract fields
    verdict = nvr.get("verdict")
    top_factor = nvr.get("top_factor")
    top_factor_value = nvr.get("top_factor_value")
    p5_source = nvr.get("p5_source", P5Source.JSONL_FALLBACK.value)

    # Generate single-line advisory_warning
    advisory_warning = format_advisory_warning(
        verdict=verdict,
        top_factor=top_factor,
        top_factor_value=top_factor_value,
        p5_source=p5_source,
    )

    return {
        "extraction_source": extraction_source,
        "verdict": verdict,
        "advisory_severity": nvr.get("advisory_severity"),
        "advisory_warning": advisory_warning,
        "advisory_message": nvr.get("advisory_message"),
        "top_factor": top_factor,
        "top_factor_value": top_factor_value,
        "coverage_ratio": nvr.get("coverage_ratio"),
        "p3_noise_rate": nvr.get("p3_noise_rate"),
        "p5_divergence_rate": nvr.get("p5_divergence_rate"),
        "p5_source": p5_source,
        "p5_source_advisory": nvr.get("p5_source_advisory"),
        "summary_sha256": nvr.get("summary_sha256"),
    }
