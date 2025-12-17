#!/usr/bin/env python3
"""Advisory CI check: detect accidental legacy P3 wrapper invocations.

This repo retains the legacy-only P3 wrapper `first_light_p3_harness.py`
(under `scripts/`). New orchestration must invoke the canonical harness
`usla_first_light_harness.py` instead.

This script scans the repository for invocations of the legacy wrapper path
outside the allowed locations (the wrapper itself and tests).
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

LEGACY_WRAPPER_REL_PATH = Path("scripts") / "first_light_p3_harness.py"
LEGACY_WRAPPER_NEEDLES = (
    LEGACY_WRAPPER_REL_PATH.as_posix().lower(),
    LEGACY_WRAPPER_REL_PATH.as_posix().replace("/", "\\").lower(),
)

ALLOWED_PREFIXES = (
    Path("tests"),
)
ALLOWED_EXACT = (
    LEGACY_WRAPPER_REL_PATH,
)

SCAN_EXTENSIONS = {
    ".py",
    ".ps1",
    ".psm1",
    ".sh",
    ".yml",
    ".yaml",
    ".bat",
    ".cmd",
}

SKIP_DIR_NAMES = {
    ".git",
    ".venv",
    ".pytest_cache",
    ".uv-cache",
    "__pycache__",
    "artifacts",
    "metrics",
    "archive",
    "analysis",
    "allblue_archive",
    "bootstrap_output",
    "ci_verification",
}


@dataclass(frozen=True)
class LegacyWrapperInvocation:
    rel_path: str
    line_number: int
    line: str


def _repo_root(default: Path | None = None) -> Path:
    if default is not None:
        return default.resolve()
    return Path(__file__).resolve().parents[1]


def _is_allowed(rel_path: Path) -> bool:
    if rel_path.as_posix() in {path.as_posix() for path in ALLOWED_EXACT}:
        return True
    return any(rel_path.parts[:1] == prefix.parts for prefix in ALLOWED_PREFIXES)


def _iter_candidate_files(root: Path) -> Iterable[Path]:
    def onerror(_: OSError) -> None:
        return

    for dirpath, dirnames, filenames in os.walk(root, onerror=onerror):
        dirnames[:] = [name for name in dirnames if name not in SKIP_DIR_NAMES]
        for filename in filenames:
            path = Path(dirpath) / filename
            if path.suffix.lower() not in SCAN_EXTENSIONS:
                continue
            yield path


def _scan_file(root: Path, path: Path) -> list[LegacyWrapperInvocation]:
    rel_path = path.relative_to(root)
    if _is_allowed(rel_path):
        return []

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    violations: list[LegacyWrapperInvocation] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        lowered = line.lower()
        if any(needle in lowered for needle in LEGACY_WRAPPER_NEEDLES):
            violations.append(
                LegacyWrapperInvocation(
                    rel_path=rel_path.as_posix(),
                    line_number=line_number,
                    line=line.strip(),
                )
            )
    return violations


def find_legacy_wrapper_invocations(root: Path) -> list[LegacyWrapperInvocation]:
    invocations: list[LegacyWrapperInvocation] = []
    for path in _iter_candidate_files(root):
        invocations.extend(_scan_file(root, path))
    invocations.sort(key=lambda item: (item.rel_path, item.line_number))
    return invocations


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Advisory CI check for accidental legacy P3 wrapper invocations.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Repo root to scan (default: auto-detect).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Always exit 0 (useful for non-gating CI advisory mode).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    root = _repo_root(Path(args.root) if args.root else None)
    invocations = find_legacy_wrapper_invocations(root)
    if not invocations:
        print("legacy_p3_wrapper_ci_check: OK (no legacy wrapper invocations found)")
        return 0

    print(
        f"legacy_p3_wrapper_ci_check: FOUND {len(invocations)} legacy wrapper invocation(s) "
        f"outside allowed locations under {root}"
    )
    for item in invocations:
        print(f"- {item.rel_path}:{item.line_number}: {item.line}")

    if args.dry_run:
        print("legacy_p3_wrapper_ci_check: DRY RUN (non-gating) — would fail in strict mode.")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
