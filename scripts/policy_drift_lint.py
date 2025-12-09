#!/usr/bin/env python3
"""
Policy Drift Lint.

Compares two policy configuration files (JSON or YAML) and reports structural
differences for the governance-critical parameters:
  * Learning rate / step size schedules
  * Clipping thresholds
  * Abstention reward weights
  * Promotion thresholds

Exit code is 0 when no tracked drift is detected, and 1 when differences exist.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple

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
    "learning_rates": "Learning rates / step size schedules",
    "clipping_thresholds": "Clipping thresholds",
    "abstention_rewards": "Abstention reward weights",
    "abstention_controls": "Abstention gating controls",
    "promotion_thresholds": "Promotion thresholds",
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
        raise ValueError(f"Failed to parse {path}: {exc}") from exc


def classify_path(path_segments: List[str]) -> Optional[str]:
    if not path_segments:
        return None
    lowered = [segment.lower() for segment in path_segments]

    def contains(keyword: str) -> bool:
        return any(keyword in segment for segment in lowered)

    def equals(keyword: str) -> bool:
        return any(segment == keyword for segment in lowered)

    # Learning rate / step size
    learning_keywords = (
        "learning_rate",
        "learning-rate",
        "learningrates",
        "learningrate",
        "lr_schedule",
        "lr",
        "eta",
        "step_size",
        "stepsize",
        "eta_schedule",
        "scheduler",
    )
    if any(contains(keyword) for keyword in learning_keywords):
        return "learning_rates"

    # Clipping thresholds
    if contains("clip") or contains("clamp"):
        return "clipping_thresholds"

    # Abstention gating controls
    if contains("abstention") and (contains("gate") or contains("gating") or contains("enabled") or contains("toggle")):
        return "abstention_controls"

    # Abstention reward weights (need abstention + reward/weight context)
    if contains("abstention") and (contains("reward") or contains("weight")):
        return "abstention_rewards"

    # Promotion thresholds
    promotion_keywords = (
        "promotion_threshold",
        "promotion-cutoff",
        "promotionlimit",
        "promotionlimit",
    )
    if any(contains(keyword) for keyword in promotion_keywords):
        return "promotion_thresholds"
    if contains("promotion") and contains("threshold"):
        return "promotion_thresholds"
    if equals("required_level") or equals("promotion_threshold"):
        return "promotion_thresholds"

    return None


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


def has_diffs(diffs: Dict[str, List[DiffEntry]]) -> bool:
    return any(diffs[category] for category in CATEGORY_ORDER)


def format_value(value: Any) -> str:
    if value is None:
        return "∅"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def build_report(diffs: Dict[str, List[DiffEntry]]) -> Dict[str, Any]:
    def serialize_entry(category: str, entry: DiffEntry) -> Dict[str, Any]:
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
            target.append(serialize_entry(category, entry))

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


def print_text_report(report: Dict[str, Any]) -> None:
    status = report["status"]
    breaking = report["breaking_changes"]
    soft = report["soft_changes"]
    if status == "OK":
        print("No policy drift detected for tracked parameters.")
        return

    print(f"Status: {status}")
    if breaking:
        print("\n[Breaking changes]")
        for entry in breaking:
            old_str = format_value(entry["old"])
            new_str = format_value(entry["new"])
            label = CATEGORY_LABELS.get(entry["category"], entry["category"])
            print(f"  - {label}: {entry['path']} {entry['change'].upper()} ({old_str} -> {new_str})")
    if soft:
        print("\n[Soft changes]")
        for entry in soft:
            old_str = format_value(entry["old"])
            new_str = format_value(entry["new"])
            label = CATEGORY_LABELS.get(entry["category"], entry["category"])
            print(f"  - {label}: {entry['path']} {entry['change'].upper()} ({old_str} -> {new_str})")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Policy Drift Linter")
    parser.add_argument("--old", required=True, type=Path, help="Path to the baseline policy file (JSON or YAML)")
    parser.add_argument("--new", required=True, type=Path, help="Path to the candidate policy file (JSON or YAML)")
    parser.add_argument(
        "--text",
        action="store_true",
        help="Emit a supplemental human-readable summary (JSON is always printed)",
    )
    parser.add_argument(
        "--warn-exit-zero",
        action="store_true",
        help="Exit code 0 for WARN status (JSON still reports WARN)",
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
    diffs = build_diff(old_values, new_values)

    report = build_report(diffs)
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


def summarize_policy_drift_for_global_health(report: Dict[str, Any]) -> Dict[str, Any]:
    status = report.get("status", "OK")

    if status == "OK":
        headline = "Policy stable; no tracked drift."
        policy_ok = True
    elif status == "WARN":
        headline = "Policy drift detected; review required."
        policy_ok = False
    else:
        headline = "Policy drift violates contract."
        policy_ok = False

    return {
        "schema_version": "1.0.0",
        "policy_ok": policy_ok,
        "status": status,
        "headline": headline,
    }


if __name__ == "__main__":
    sys.exit(main())
