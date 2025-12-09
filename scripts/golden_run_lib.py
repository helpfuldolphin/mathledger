"""
Shared helpers for Golden Run Recorder/Validator.

This module centralizes hashing, environment snapshotting, and diff logic so
both the recorder and validator operate on the same primitives. The functions
are intentionally deterministic: every helper avoids global mutable state and
hashes inputs using canonical encodings.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from typing import Sequence as TypingSequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SCHEMA_VERSION = "1.0.0"

# Seed-related variables that we always snapshot for replay.
DEFAULT_ENV_KEYS: Tuple[str, ...] = (
    "PYTHONHASHSEED",
    "RFL_MASTER_SEED",
    "RFL_PRNG_MASTER_SEED",
    "ML_GOLDEN_SEED",
    "ML_GOLDEN_RUN_ID",
    "CUDA_VISIBLE_DEVICES",
    "UV_PROJECT_ENV",
)


@dataclass(frozen=True)
class ArtifactSummary:
    """Hashes and metadata for run artifacts."""

    ht_series_hash: str
    ht_series_metadata: Dict[str, Any]
    trace_hash: str
    metrics_snapshot: Dict[str, Any]
    metrics_snapshot_hash: Optional[str]


def sha256_file(path: Path) -> str:
    """Compute SHA-256 for a binary file."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_hash(payload: Any) -> str:
    """
    Hash arbitrary JSON-serializable payload using canonical ordering.

    Sorting keys plus compact separators ensures byte-for-byte determinism.
    """
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def compute_ht_series_hash(ht_log: Path) -> Tuple[str, Dict[str, Any]]:
    """
    Hash the sequence of h_t roots emitted in a JSONL log.

    Each discovered roots.h_t is appended with a newline prior to hashing so the
    digest is sensitive to ordering and cycle count.
    """
    if not ht_log.exists():
        raise FileNotFoundError(f"HT log not found: {ht_log}")

    digest = hashlib.sha256()
    cycles = 0
    first_cycle: Optional[int] = None
    last_cycle: Optional[int] = None

    with ht_log.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Line {line_no}: invalid JSON in {ht_log}: {exc}") from exc
            roots = record.get("roots") or {}
            h_t = roots.get("h_t")
            if not h_t:
                continue
            digest.update(h_t.encode("utf-8"))
            digest.update(b"\n")
            cycle = record.get("cycle")
            if isinstance(cycle, int):
                first_cycle = cycle if first_cycle is None else min(first_cycle, cycle)
                last_cycle = cycle if last_cycle is None else max(last_cycle, cycle)
            cycles += 1

    if cycles == 0:
        raise ValueError(f"No h_t entries found in {ht_log}")

    metadata: Dict[str, Any] = {"cycles_included": cycles}
    if first_cycle is not None and last_cycle is not None:
        metadata["cycle_range"] = [first_cycle, last_cycle]
    return digest.hexdigest(), metadata


def load_metrics_snapshot(path: Optional[Path]) -> Dict[str, Any]:
    """Load metrics JSON if provided; otherwise return an empty dict."""
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _git_info(project_root: Path) -> Dict[str, Any]:
    """Collect git commit, branch, and dirty flag."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": "unknown", "branch": "unknown", "is_dirty": False}

    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=project_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        branch = "unknown"

    try:
        status_out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=project_root,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        is_dirty = bool(status_out.strip())
        diff_hash = None
        if is_dirty:
            diff_bytes = subprocess.check_output(
                ["git", "diff", "HEAD"],
                cwd=project_root,
                stderr=subprocess.DEVNULL,
            )
            diff_hash = hashlib.sha256(diff_bytes).hexdigest()
    except (subprocess.CalledProcessError, FileNotFoundError):
        is_dirty = False
        diff_hash = None

    info: Dict[str, Any] = {"commit": commit, "branch": branch, "is_dirty": is_dirty}
    if diff_hash:
        info["diff_sha256"] = diff_hash
    return info


def _pip_freeze_hash() -> str:
    """Hash the current pip environment for quick comparisons."""
    try:
        output = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return "unavailable"
    return hashlib.sha256(output.encode("utf-8")).hexdigest()


def collect_environment_snapshot(
    env: Mapping[str, str],
    env_keys: Sequence[str],
    extra_env_keys: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Capture python/system/git info plus the requested env vars."""
    keys: List[str] = list(env_keys)
    if extra_env_keys:
        keys.extend(extra_env_keys)
    unique_keys = sorted({k for k in keys if k})
    variables = {k: env.get(k) for k in unique_keys if k in env}

    return {
        "captured_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "system": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "git": _git_info(PROJECT_ROOT),
        "pip_freeze_sha256": _pip_freeze_hash(),
        "variables": variables,
    }


def rel_path(path: Path, base: Path) -> str:
    """Return a repo-relative path when possible."""
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def build_artifact_snapshot(
    ht_log: Path,
    trace_log: Path,
    metrics_path: Optional[Path],
) -> ArtifactSummary:
    """Compute hashes for the core replay artifacts."""
    ht_hash, ht_meta = compute_ht_series_hash(ht_log)
    trace_hash = sha256_file(trace_log)
    metrics_snapshot = load_metrics_snapshot(metrics_path)
    metrics_hash = canonical_json_hash(metrics_snapshot) if metrics_snapshot else None
    return ArtifactSummary(
        ht_series_hash=ht_hash,
        ht_series_metadata=ht_meta,
        trace_hash=trace_hash,
        metrics_snapshot=metrics_snapshot,
        metrics_snapshot_hash=metrics_hash,
    )


def diff_artifacts(
    golden: Mapping[str, Any],
    observed: Mapping[str, Any],
) -> List[str]:
    """
    Compare top-level hashes between golden record and observed snapshot.

    Returns a list of human-readable drift descriptions, empty if identical.
    """
    diffs: List[str] = []
    if golden.get("ht_series_hash") != observed.get("ht_series_hash"):
        golden_meta = golden.get("ht_series_metadata", {})
        observed_meta = observed.get("ht_series_metadata", {})
        diffs.append(
            "ht_series_hash mismatch "
            f"(expected={golden.get('ht_series_hash')}, actual={observed.get('ht_series_hash')}, "
            f"expected_meta={golden_meta}, actual_meta={observed_meta})"
        )
    if golden.get("trace_hash") != observed.get("trace_hash"):
        diffs.append(
            "trace_hash mismatch "
            f"(expected={golden.get('trace_hash')}, actual={observed.get('trace_hash')})"
        )
    golden_metrics_hash = golden.get("metrics_snapshot_hash")
    observed_metrics_hash = observed.get("metrics_snapshot_hash")
    if golden_metrics_hash != observed_metrics_hash:
        diffs.append(
            "metrics_snapshot_hash mismatch "
            f"(expected={golden_metrics_hash}, actual={observed_metrics_hash})"
        )
    return diffs


def summarize_golden_runs_for_global_health(
    results: TypingSequence[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Produce a compact status summary for golden run replay outcomes.

    Args:
        results: Sequence of per-run dictionaries that include a "status" key.
                 Expected statuses: "OK", "MISMATCH", "ERROR".

    Returns:
        Dict[str, Any]: JSON-safe summary containing:
            - status: "OK" when all runs passed, otherwise "DEGRADED"
            - mismatch_count: number of runs that were not OK
            - headline: human-readable string
    """
    total = len(results)
    mismatch_count = sum(
        1 for r in results if r.get("status") not in ("OK", "SKIPPED")
    )
    if total == 0:
        status = "EMPTY"
        headline = "No golden runs evaluated"
    elif mismatch_count == 0:
        status = "OK"
        headline = f"All {total} golden runs matched"
    else:
        status = "DEGRADED"
        matched = total - mismatch_count
        headline = f"{matched}/{total} golden runs matched"

    return {
        "status": status,
        "mismatch_count": mismatch_count,
        "headline": headline,
    }
