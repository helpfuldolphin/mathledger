#!/usr/bin/env python3
"""Legacy compatibility wrapper for the real USLA First Light harness.

This script kept the original P3 stub CLI alive but now defers to
``scripts/usla_first_light_harness.py`` so callers do not need to update their
entry points when the real implementation lands. Core arguments (cycles,
output dir, seed, slice, runner type, etc.) are parsed and forwarded, and the
primary synthetic artifact is mirrored back to the user-provided output
directory for backwards compatibility.

New tooling SHOULD invoke ``scripts/usla_first_light_harness.py`` directly.
This script exists solely to keep older orchestration code working while
callers migrate to the single source of truth harness.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Sequence
import warnings

from scripts import usla_first_light_harness

OUTPUT_FILENAME = "first_light_synthetic_raw.jsonl"
REAL_PRIMARY_FILENAME = "synthetic_raw.jsonl"
REAL_PROG_NAME = "usla_first_light_harness.py"
CANONICAL_COMMAND = (
    "uv run python scripts/usla_first_light_harness.py "
    "--cycles <n> --output-dir <path> [--seed <seed> ...]"
)
BANNER = (
    "\n=== LEGACY ONLY — DO NOT USE FOR NEW RUNS ===\n"
    "Canonical harness: "
    f"{CANONICAL_COMMAND}\n"
    "This wrapper exists for backwards compatibility only.\n"
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Wrapper for the real USLA First Light harness.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=1000,
        help="Number of cycles to run (default: 1000).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory that should receive artifacts (required).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed forwarded to the real harness.",
    )
    parser.add_argument(
        "--slice",
        type=str,
        default=None,
        help="Slice to run; defaults to arithmetic_simple if omitted.",
    )
    parser.add_argument(
        "--runner-type",
        type=str,
        choices=["u2", "rfl"],
        default=None,
        help="Runner type to use; defaults to u2 if omitted.",
    )
    parser.add_argument(
        "--tau-0",
        type=float,
        dest="tau_0",
        default=None,
        help="Initial tau_0 value (forwarded verbatim).",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=None,
        help="Metrics window size override.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without executing the run.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging inside the real harness.",
    )
    return parser


def _compose_forward_args(
    parsed: argparse.Namespace,
    extras: Sequence[str],
) -> list[str]:
    """Translate parsed args + passthrough extras into the real CLI argv."""
    forward: list[str] = [
        "--cycles",
        str(parsed.cycles),
        "--output-dir",
        parsed.output_dir,
    ]
    if parsed.seed is not None:
        forward.extend(["--seed", str(parsed.seed)])
    if parsed.slice:
        forward.extend(["--slice", parsed.slice])
    if parsed.runner_type:
        forward.extend(["--runner-type", parsed.runner_type])
    if parsed.tau_0 is not None:
        forward.extend(["--tau-0", str(parsed.tau_0)])
    if parsed.window_size is not None:
        forward.extend(["--window-size", str(parsed.window_size)])
    if parsed.dry_run:
        forward.append("--dry-run")
    if parsed.verbose:
        forward.append("--verbose")
    forward.extend(extras)
    return forward


def _delegate_to_real_harness(forward_args: Sequence[str]) -> int:
    """Call the real harness with a synthetic argv."""
    original_argv = sys.argv[:]
    try:
        sys.argv = [REAL_PROG_NAME, *forward_args]
        return usla_first_light_harness.main()
    finally:
        sys.argv = original_argv


def _mirror_primary_artifact(output_root: Path) -> Path | None:
    """Copy the latest synthetic JSONL into the legacy location."""
    if not output_root.exists():
        return None

    try:
        candidates: list[Path] = []
        for name in {REAL_PRIMARY_FILENAME, OUTPUT_FILENAME}:
            candidates.extend(
                output_root.rglob(name),
            )
        candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    except OSError:
        return None

    if not candidates:
        return None

    latest = candidates[0]
    dest = output_root / OUTPUT_FILENAME
    if latest.resolve() == dest.resolve():
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(latest, dest)
    return dest


def main(argv: Sequence[str] | None = None) -> int:
    """Parse args, delegate to the real harness, and mirror artifacts."""
    warnings.warn(
        "scripts/first_light_p3_harness.py is deprecated; call "
        "scripts/usla_first_light_harness.py directly for all new tooling.",
        FutureWarning,
        stacklevel=2,
    )
    print(BANNER, flush=True)
    parser = _build_parser()
    parsed, extras = parser.parse_known_args(list(argv) if argv is not None else None)

    forward_args = _compose_forward_args(parsed, extras)
    exit_code = _delegate_to_real_harness(forward_args)

    # Best-effort compatibility shim so older tooling still finds the JSONL.
    _mirror_primary_artifact(Path(parsed.output_dir))

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
