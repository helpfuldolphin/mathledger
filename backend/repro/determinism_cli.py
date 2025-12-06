"""Determinism check CLI required for Wave-1 promotion."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict


DEFAULT_RUNS = 3


def calculate_dir_hash(directory: Path) -> str:
    """Compute a deterministic SHA-256 digest over all files (sorted)."""
    sha256 = hashlib.sha256()
    for root, _, files in sorted(os.walk(directory)):
        for name in sorted(files):
            filepath = Path(root) / name
            if "__pycache__" in filepath.parts:
                continue
            try:
                with open(filepath, "rb") as handle:
                    while True:
                        chunk = handle.read(65536)
                        if not chunk:
                            break
                        sha256.update(chunk)
                sha256.update(str(filepath.relative_to(directory)).encode("utf-8"))
            except OSError:
                continue
    return sha256.hexdigest()


def calculate_file_hash(path: Path) -> str:
    """Hash an individual file (used for manifest stability)."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(65536)
            if not chunk:
                break
            sha256.update(chunk)
    return sha256.hexdigest()


def repeat_hash(func, target: Path, runs: int) -> str:
    """Verify that repeated hash executions produce the same digest."""
    baseline: str | None = None
    for _ in range(runs):
        current = func(target)
        if baseline is None:
            baseline = current
        elif current != baseline:
            raise ValueError(f"Determinism violation: {current} != {baseline}")
    assert baseline is not None
    return baseline


def run_checks(directory: Path, runs: int, manifest: Path | None) -> Dict[str, Any]:
    """Execute the deterministic hashing checks."""
    result: Dict[str, Any] = {
        "directory": str(directory),
        "runs": runs,
        "directory_hash": repeat_hash(calculate_dir_hash, directory, runs),
    }
    if manifest:
        result["manifest"] = str(manifest)
        result["manifest_hash"] = repeat_hash(calculate_file_hash, manifest, runs)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Wave-1 determinism gate")
    parser.add_argument("--directory", type=Path, default=Path("basis"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    args = parser.parse_args()

    if not args.directory.exists():
        print(json.dumps({"status": "FAIL", "error": "directory missing"}))
        sys.exit(1)
    if args.manifest and not args.manifest.exists():
        print(json.dumps({"status": "FAIL", "error": "manifest missing"}))
        sys.exit(1)

    try:
        summary = run_checks(args.directory, args.runs, args.manifest)
        print(json.dumps({"status": "PASS", "summary": summary}, indent=2))
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}))
        sys.exit(1)


if __name__ == "__main__":
    main()

