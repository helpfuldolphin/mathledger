#!/usr/bin/env python3
"""
Golden Run Recorder - Deterministic experiment capture.

Usage:
    uv run python scripts/record_golden_run.py \
        --run-id u2-baseline \
        --ht-log results/uplift_u2_arithmetic_simple_baseline.jsonl \
        --trace-log traces/uplift_u2_arithmetic_simple_baseline.jsonl \
        --metrics artifacts/uplift_u2_arithmetic_simple_baseline_metrics.json \
        -- python experiments/run_uplift_u2.py --slice arithmetic_simple ...
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from rfl.prng import DEFAULT_MASTER_SEED

from scripts.golden_run_lib import (
    ArtifactSummary,
    DEFAULT_ENV_KEYS,
    PROJECT_ROOT,
    SCHEMA_VERSION,
    build_artifact_snapshot,
    collect_environment_snapshot,
    rel_path,
    sha256_file,
)


def _parse_env_override(values: Optional[List[str]]) -> Dict[str, str]:
    overrides: Dict[str, str] = {}
    if not values:
        return overrides
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"Invalid env override '{raw}'. Expected KEY=VALUE.")
        key, value = raw.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid env override '{raw}': empty key.")
        overrides[key] = value
    return overrides


def _generate_run_id(prefix: str = "golden") -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{ts}"


def _default_output_path(run_id: str) -> Path:
    return PROJECT_ROOT / "artifacts" / "golden_runs" / f"{run_id}.json"


def _command_to_string(command: List[str]) -> str:
    return subprocess.list2cmdline(command)


def _build_env(run_id: str, seed: str, extra_env: Dict[str, str]) -> Dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONHASHSEED", "0")
    env["ML_GOLDEN_RUN_ID"] = run_id
    env.setdefault("ML_GOLDEN_SEED", seed)
    env.setdefault("RFL_MASTER_SEED", seed)
    env.setdefault("RFL_PRNG_MASTER_SEED", seed)
    env.update(extra_env)
    return env


def record_golden_run(args: argparse.Namespace) -> Path:
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("No experiment command provided. Use `--` before the command.")

    run_id = args.run_id or _generate_run_id()
    seed = args.seed or DEFAULT_MASTER_SEED
    env_overrides = _parse_env_override(args.set_env)
    env = _build_env(run_id, seed, env_overrides)
    ht_log = Path(args.ht_log)
    trace_log = Path(args.trace_log)
    metrics_path = Path(args.metrics) if args.metrics else None
    trace_log.parent.mkdir(parents=True, exist_ok=True)
    ht_log.parent.mkdir(parents=True, exist_ok=True)
    if metrics_path:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)

    workdir = Path(args.workdir).resolve()

    print(f"[GoldenRun] run_id={run_id}")
    print(f"[GoldenRun] command={' '.join(command)}")
    print(f"[GoldenRun] workdir={workdir}")

    completed = subprocess.run(
        command,
        cwd=workdir,
        env=env,
        text=True,
    )
    if completed.returncode != 0:
        raise SystemExit(
            f"Experiment command failed with exit code {completed.returncode}"
        )

    artifact_summary: ArtifactSummary = build_artifact_snapshot(
        ht_log=ht_log,
        trace_log=trace_log,
        metrics_path=metrics_path,
    )

    env_snapshot = collect_environment_snapshot(
        env=env,
        env_keys=DEFAULT_ENV_KEYS,
        extra_env_keys=args.extra_env_keys,
    )

    record = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "command": _command_to_string(command),
        "command_args": command,
        "env": env_snapshot,
        "ht_series_hash": artifact_summary.ht_series_hash,
        "ht_series_metadata": artifact_summary.ht_series_metadata,
        "trace_hash": artifact_summary.trace_hash,
        "metrics_snapshot": artifact_summary.metrics_snapshot,
        "metrics_snapshot_hash": artifact_summary.metrics_snapshot_hash,
    }

    artifacts = {
        "ht_log": {
            "path": rel_path(ht_log, PROJECT_ROOT),
            "file_sha256": sha256_file(ht_log),
        },
        "trace_log": {
            "path": rel_path(trace_log, PROJECT_ROOT),
            "file_sha256": sha256_file(trace_log),
        },
    }
    if metrics_path:
        artifacts["metrics"] = {
            "path": rel_path(metrics_path, PROJECT_ROOT),
            "file_sha256": sha256_file(metrics_path),
        }
    record["artifacts"] = artifacts

    output_path = Path(args.output) if args.output else _default_output_path(run_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)

    print(f"[GoldenRun] Wrote record to {output_path}")
    print(f"[GoldenRun] ht_series_hash={artifact_summary.ht_series_hash}")
    print(f"[GoldenRun] trace_hash={artifact_summary.trace_hash}")
    if artifact_summary.metrics_snapshot_hash:
        print(
            f"[GoldenRun] metrics_snapshot_hash={artifact_summary.metrics_snapshot_hash}"
        )
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Golden Run Recorder (deterministic experiment capture)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run-id",
        help="Identifier for the run (default: golden-<timestamp>)",
    )
    parser.add_argument(
        "--seed",
        help="Master seed applied via env if experiment does not override it.",
        default=DEFAULT_MASTER_SEED,
    )
    parser.add_argument(
        "--ht-log",
        required=True,
        help="Path to JSONL log containing per-cycle roots.h_t entries.",
    )
    parser.add_argument(
        "--trace-log",
        required=True,
        help="Path to the structured trace JSONL file emitted by the command.",
    )
    parser.add_argument(
        "--metrics",
        help="Optional JSON file to embed in metrics_snapshot.",
    )
    parser.add_argument(
        "--output",
        help="Where to write the golden run receipt.",
    )
    parser.add_argument(
        "--workdir",
        default=str(PROJECT_ROOT),
        help="Working directory for the experiment command.",
    )
    parser.add_argument(
        "--set-env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Additional environment variables to export while running the experiment.",
    )
    parser.add_argument(
        "--extra-env-keys",
        action="append",
        default=[],
        help="Additional environment keys to include in the recorded snapshot.",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Experiment command to run (prefix with -- to separate).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    record_golden_run(args)


if __name__ == "__main__":
    main()
