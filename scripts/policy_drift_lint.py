#!/usr/bin/env python3
"""
Policy Drift Lint CLI.

Canonical workflow profiles:
    * strict (default): WARN → exit 1. Use in CI so any soft drift pauses the pipeline.
    * advisory (--warn-exit-zero): WARN → exit 0. Ideal for pre-commit/dev hooks that nudge
      without blocking.
    * silent (omit --text): emit JSON only so dashboards ingest the tile without extra logs.

Global health integration example::

    import json, subprocess
    from scripts.policy_drift_lint import summarize_policy_drift_for_global_health

    proc = subprocess.run(
        [
            "uv", "run", "python", "scripts/policy_drift_lint.py",
            "--old", "config/policy_baseline.yaml",
            "--new", "config/policy_candidate.yaml",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(proc.stdout)
    policy_tile = summarize_policy_drift_for_global_health(report)
    global_health = {"policy_drift": policy_tile}
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import yaml

SCHEMA_VERSION = "1.0.0"
SAFE_LEARNING_RATE_MAX = 0.5

CATEGORY_ORDER = [
    "learning_rates",
    "clipping_thresholds",
    "abstention_rewards",
    "abstention_controls",
    "promotion_thresholds",
]

CATEGORY_LABELS = {
    "learning_rates": "learning rate schedule",
    "clipping_thresholds": "clipping thresholds",
    "abstention_rewards": "abstention reward weights",
    "abstention_controls": "abstention gating controls",
    "promotion_thresholds": "promotion thresholds",
}

BREAKING_CATEGORIES = {"promotion_thresholds"}


@dataclass
class DiffEntry:
    path: str
    old: Any
    new: Any
    change: str


def load_policy(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Policy file not found: {path}")
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return {}
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as exc:  # pragma: no cover
        raise ValueError(f"Failed to parse policy file {path}: {exc}") from exc


def classify_path(path_segments: List[str]) -> Optional[str]:
    if not path_segments:
        return None
    lowered = [segment.lower() for segment in path_segments]

    def contains(keyword: str) -> bool:
        return any(keyword in segment for segment in lowered)

    def equals(keyword: str) -> bool:
        return any(segment == keyword for segment in lowered)

    learning_keywords = (
        "learning_rate",
        "learningrate",
        "learning-rate",
        "lr",
        "lr_schedule",
        "eta",
        "step_size",
        "stepsize",
        "scheduler",
    )
    if any(contains(keyword) for keyword in learning_keywords):
        return "learning_rates"

    if contains("clip") or contains("clamp"):
        return "clipping_thresholds"

    if contains("abstention") and (contains("gate") or contains("toggle") or contains("enabled")):
        return "abstention_controls"

    if contains("abstention") and (contains("reward") or contains("weight")):
        return "abstention_rewards"

    promotion_keywords = ("promotion_threshold", "promotion-cutoff", "promotionlimit")
    if any(contains(keyword) for keyword in promotion_keywords):
        return "promotion_thresholds"
    if contains("promotion") and contains("threshold"):
        return "promotion_thresholds"
    if equals("required_level") or equals("promotion_threshold"):
        return "promotion_thresholds"

    return None


def extract_tracked_values(data: Any) -> Dict[str, Dict[str, Any]]:
    tracked: Dict[str, Dict[str, Any]] = {category: {} for category in CATEGORY_ORDER}

    def visit(node: Any, path: List[str]) -> None:
        if isinstance(node, Mapping):
            for key in sorted(node.keys()):
                visit(node[key], path + [str(key)])
        elif isinstance(node, list):
            for idx, value in enumerate(node):
                visit(value, path + [str(idx)])
        else:
            category = classify_path(path)
            if category:
                tracked[category][".".join(path)] = node

    visit(data, [])
    return tracked


def build_diff(
    old_values: Dict[str, Dict[str, Any]],
    new_values: Dict[str, Dict[str, Any]],
) -> Dict[str, List[DiffEntry]]:
    diffs: Dict[str, List[DiffEntry]] = {category: [] for category in CATEGORY_ORDER}
    for category in CATEGORY_ORDER:
        old_map = old_values.get(category, {})
        new_map = new_values.get(category, {})
        all_paths = sorted(set(old_map.keys()) | set(new_map.keys()))
        for path in all_paths:
            if path not in old_map:
                diffs[category].append(DiffEntry(path, None, new_map[path], "added"))
            elif path not in new_map:
                diffs[category].append(DiffEntry(path, old_map[path], None, "removed"))
            else:
                old_val = old_map[path]
                new_val = new_map[path]
                if old_val != new_val:
                    diffs[category].append(DiffEntry(path, old_val, new_val, "changed"))
    return diffs


def _to_float(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _is_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        lowered = value.strip().lower()
        return lowered not in ("", "0", "false", "off", "disabled", "no")
    return value is not None


def _is_learning_rate_spike(entry: DiffEntry) -> bool:
    if entry.change == "removed":
        return False
    new_val = _to_float(entry.new)
    if new_val is None or new_val <= SAFE_LEARNING_RATE_MAX:
        return False
    old_val = _to_float(entry.old)
    if old_val is None:
        return True
    return new_val > old_val


def _is_abstention_gate_disabled(entry: DiffEntry) -> bool:
    if entry.change == "removed":
        return True
    if entry.change == "changed":
        return not _is_truthy(entry.new) and _is_truthy(entry.old)
    if entry.change == "added":
        return not _is_truthy(entry.new)
    return False


def build_report(diffs: Dict[str, List[DiffEntry]]) -> Dict[str, Any]:
    def serialize(category: str, entry: DiffEntry) -> Dict[str, Any]:
        return {
            "category": category,
            "path": entry.path,
            "change": entry.change,
            "old": entry.old,
            "new": entry.new,
        }

    breaking: List[Dict[str, Any]] = []
    soft: List[Dict[str, Any]] = []

    for category in CATEGORY_ORDER:
        entries = diffs[category]
        if not entries:
            continue
        for entry in entries:
            is_breaking = category in BREAKING_CATEGORIES
            if category == "learning_rates" and _is_learning_rate_spike(entry):
                is_breaking = True
            if category == "clipping_thresholds" and entry.change == "removed":
                is_breaking = True
            if category == "abstention_controls" and _is_abstention_gate_disabled(entry):
                is_breaking = True
            target = breaking if is_breaking else soft
            target.append(serialize(category, entry))

    if breaking:
        status = "BLOCK"
    elif soft:
        status = "WARN"
    else:
        status = "OK"

    return {
        "schema_version": SCHEMA_VERSION,
        "breaking_changes": breaking,
        "soft_changes": soft,
        "status": status,
        "summary": {
            "total_changes": sum(len(entries) for entries in diffs.values()),
            "by_category": {category: len(entries) for category, entries in diffs.items()},
        },
    }


def format_value(value: Any) -> str:
    if value is None:
        return "∅"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def print_text_report(report: Dict[str, Any]) -> None:
    status = report["status"]
    if status == "OK":
        print("Policy stable; no tracked drift.")
        return

    print(f"Status: {status}")
    if report["breaking_changes"]:
        print("\n[Breaking changes]")
        for entry in report["breaking_changes"]:
            label = CATEGORY_LABELS.get(entry["category"], entry["category"])
            print(
                f"  - {label}: {entry['path']} {entry['change'].upper()} "
                f"({format_value(entry['old'])} -> {format_value(entry['new'])})"
            )
    if report["soft_changes"]:
        print("\n[Soft changes]")
        for entry in report["soft_changes"]:
            label = CATEGORY_LABELS.get(entry["category"], entry["category"])
            print(
                f"  - {label}: {entry['path']} {entry['change'].upper()} "
                f"({format_value(entry['old'])} -> {format_value(entry['new'])})"
            )


def summarize_policy_drift_for_global_health(report: Dict[str, Any]) -> Dict[str, Any]:
    status = report.get("status", "OK")
    breaking_count = len(report.get("breaking_changes", []))
    soft_count = len(report.get("soft_changes", []))

    if status == "OK":
        headline = "Policy stable; no tracked drift."
        policy_ok = True
    elif status == "WARN":
        categories = sorted(
            {
                CATEGORY_LABELS.get(entry["category"], entry["category"])
                for entry in report.get("soft_changes", [])
            }
        )
        category_str = ", ".join(categories) if categories else "unknown areas"
        headline = f"{soft_count} soft change(s) detected in {category_str}."
        policy_ok = False
    else:
        categories = sorted(
            {
                CATEGORY_LABELS.get(entry["category"], entry["category"])
                for entry in report.get("breaking_changes", [])
            }
        )
        category_str = ", ".join(categories) if categories else "policy parameters"
        headline = f"{breaking_count} blocking change(s) detected in {category_str}."
        policy_ok = False

    return {
        "schema_version": SCHEMA_VERSION,
        "policy_ok": policy_ok,
        "status": status,
        "headline": headline,
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Policy Drift Linter")
    parser.add_argument("--old", required=True, type=Path, help="Path to baseline policy (JSON/YAML)")
    parser.add_argument("--new", required=True, type=Path, help="Path to candidate policy (JSON/YAML)")
    parser.add_argument(
        "--text",
        action="store_true",
        help="Emit supplemental human-readable summary (JSON always printed)",
    )
    parser.add_argument(
        "--warn-exit-zero",
        action="store_true",
        help="Return exit code 0 on WARN status (JSON still reports WARN).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    try:
        old_policy = load_policy(args.old)
        new_policy = load_policy(args.new)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[policy-drift-lint] ERROR: {exc}", file=sys.stderr)
        return 2

    old_values = extract_tracked_values(old_policy or {})
    new_values = extract_tracked_values(new_policy or {})
    report = build_report(build_diff(old_values, new_values))

    print(json.dumps(report, indent=2, sort_keys=True))
    if args.text:
        print()
        print_text_report(report)

    status = report["status"]
    if status == "OK":
        return 0
    if status == "WARN" and args.warn_exit_zero:
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
