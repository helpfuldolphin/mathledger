#!/usr/bin/env python3
"""
Golden Run Replay Validator.

Loads a golden run receipt, optionally replays the recorded command, and
compares newly hashed artifacts against the golden record. Any drift produces a
non-zero exit code together with explicit diagnostics.

Policy Schema (YAML):
    schema_version: 1.0.0
    golden_runs:
      - name: string
        runs_required: int
        runs:
          - receipt: path/to/receipt.json
            ht_log: path/to/ht.jsonl
            trace_log: path/to/trace.jsonl
            metrics: optional/path/to/metrics.json
            skip_command: bool (optional)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

from scripts.golden_run_lib import (
    ArtifactSummary,
    PROJECT_ROOT,
    build_artifact_snapshot,
    summarize_golden_runs_for_global_health,
    diff_artifacts,
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


def _load_record(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_artifact_path(record: Dict[str, Any], key: str) -> Optional[Path]:
    info = (record.get("artifacts") or {}).get(key)
    if not info:
        return None
    stored = info.get("path")
    if not stored:
        return None
    path = Path(stored)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _summary_to_mapping(summary: ArtifactSummary) -> Dict[str, Any]:
    return {
        "ht_series_hash": summary.ht_series_hash,
        "ht_series_metadata": summary.ht_series_metadata,
        "trace_hash": summary.trace_hash,
        "metrics_snapshot_hash": summary.metrics_snapshot_hash,
    }


def _sanitize_command(command: Optional[List[str]]) -> List[str]:
    if not command:
        return []
    if command[0] == "--":
        command = command[1:]
    return command


def _normalize_path(value: Optional[str]) -> Optional[Path]:
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def _run_single_validation(
    golden_path: Path,
    ht_log: Optional[Path],
    trace_log: Optional[Path],
    metrics_path: Optional[Path],
    env_overrides: Dict[str, str],
    skip_command: bool,
    command_override: Sequence[str],
) -> Dict[str, Any]:
    record = _load_record(golden_path)
    env = os.environ.copy()
    recorded_env = (record.get("env") or {}).get("variables") or {}
    env.update(recorded_env)
    env.update(env_overrides)

    command = _sanitize_command(list(command_override))
    fallback_command = record.get("command_args")
    if not command and fallback_command:
        command = fallback_command
    shell_command: Optional[str] = None
    if not command:
        shell_command = record.get("command")

    ht_log = ht_log or _resolve_artifact_path(record, "ht_log")
    trace_log = trace_log or _resolve_artifact_path(record, "trace_log")
    metrics_path = metrics_path or _resolve_artifact_path(record, "metrics")

    if ht_log is None or trace_log is None:
        raise ValueError("Golden record is missing ht_log/trace_log artifact metadata.")

    if not skip_command:
        if command:
            print(f"[GoldenRun] Replaying command: {' '.join(command)}")
            completed = subprocess.run(command, cwd=PROJECT_ROOT, env=env, text=True)
        elif shell_command:
            print(f"[GoldenRun] Replaying shell command: {shell_command}")
            completed = subprocess.run(
                shell_command,
                cwd=PROJECT_ROOT,
                env=env,
                text=True,
                shell=True,
            )
        else:
            raise ValueError(
                "Golden record does not include command_args and no command override provided."
            )

        if completed.returncode != 0:
            raise RuntimeError(
                f"Replay command failed with exit code {completed.returncode}"
            )
    else:
        print("[GoldenRun] --skip-command enabled; skipping replay execution.")

    observed: ArtifactSummary = build_artifact_snapshot(
        ht_log=ht_log,
        trace_log=trace_log,
        metrics_path=metrics_path,
    )

    diffs = diff_artifacts(record, _summary_to_mapping(observed))
    return {
        "record": record,
        "diffs": diffs,
        "ht_log": str(ht_log),
        "trace_log": str(trace_log),
        "metrics": str(metrics_path) if metrics_path else None,
    }


def validate_replay(args: argparse.Namespace) -> int:
    record_path = Path(args.golden)
    env_overrides = _parse_env_override(args.set_env)
    try:
        result = _run_single_validation(
            golden_path=record_path,
            ht_log=Path(args.ht_log) if args.ht_log else None,
            trace_log=Path(args.trace_log) if args.trace_log else None,
            metrics_path=Path(args.metrics) if args.metrics else None,
            env_overrides=env_overrides,
            skip_command=args.skip_command,
            command_override=args.command,
        )
    except Exception as exc:
        raise SystemExit(str(exc)) from exc

    if result["diffs"]:
        print("[GoldenRun] Replay drift detected:", file=sys.stderr)
        for diff in result["diffs"]:
            print(f"  - {diff}", file=sys.stderr)
        return 1

    print("[GoldenRun] Replay verified: hashes match golden record.")
    return 0


def _load_policy(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def validate_policy(args: argparse.Namespace) -> int:
    policy = _load_policy(Path(args.policy))
    golden_runs = policy.get("golden_runs", [])
    env_overrides = _parse_env_override(args.set_env)
    results: List[Dict[str, Any]] = []

    for golden in golden_runs:
        name = golden.get("name", "unnamed_golden")
        runs = golden.get("runs") or []
        required = golden.get("runs_required") or len(runs)
        if len(runs) < required:
            results.append(
                {
                    "name": name,
                    "status": "ERROR",
                    "diffs": [
                        f"Policy requires {required} runs but only {len(runs)} configured"
                    ],
                }
            )
            continue
        for idx, run_cfg in enumerate(runs, start=1):
            receipt_path = _normalize_path(run_cfg.get("receipt"))
            if receipt_path is None:
                results.append(
                    {
                        "name": f"{name}#run{idx}",
                        "status": "ERROR",
                        "diffs": ["Missing receipt path in policy"],
                    }
                )
                continue

            ht_override = _normalize_path(run_cfg.get("ht_log"))
            trace_override = _normalize_path(run_cfg.get("trace_log"))
            metrics_override = _normalize_path(run_cfg.get("metrics"))
            skip_cmd = args.skip_command or run_cfg.get("skip_command", False)
            try:
                result = _run_single_validation(
                    golden_path=receipt_path,
                    ht_log=ht_override,
                    trace_log=trace_override,
                    metrics_path=metrics_override,
                    env_overrides=env_overrides,
                    skip_command=skip_cmd,
                    command_override=args.command,
                )
                status = "OK" if not result["diffs"] else "MISMATCH"
                diffs = result["diffs"]
            except Exception as exc:
                status = "ERROR"
                diffs = [str(exc)]

            results.append(
                {
                    "name": f"{name}#run{idx}",
                    "status": status,
                    "diffs": diffs,
                }
            )

    summary = summarize_golden_runs_for_global_health(results)
    print(json.dumps(summary, indent=2))
    if summary["status"] == "OK":
        return 0
    print("[GoldenRun] Failures detected in golden run policy validation:", file=sys.stderr)
    for entry in results:
        if entry["status"] != "OK":
            print(f"  - {entry['name']}: {entry['status']}", file=sys.stderr)
            for diff in entry.get("diffs", []):
                print(f"      {diff}", file=sys.stderr)
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Golden Run Replay Validator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--golden",
        help="Path to the golden run receipt emitted by record_golden_run.py",
    )
    target.add_argument(
        "--policy",
        help="Path to a golden run policy YAML file describing multiple runs.",
    )
    parser.add_argument(
        "--ht-log",
        help="Override path to HT-series log for replay validation.",
    )
    parser.add_argument(
        "--trace-log",
        help="Override path to trace log for replay validation.",
    )
    parser.add_argument(
        "--metrics",
        help="Override path to metrics snapshot JSON.",
    )
    parser.add_argument(
        "--set-env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Environment overrides applied prior to running the replay command.",
    )
    parser.add_argument(
        "--skip-command",
        action="store_true",
        help="Skip rerunning the recorded command; only recompute hashes from files.",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Optional command override (prefix with -- to separate).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.policy:
        exit_code = validate_policy(args)
    else:
        exit_code = validate_replay(args)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
