#!/usr/bin/env python3
"""
Smoke test helper for containerized First Light harnesses.

Runs the P3 and P4 docker-compose services if Docker is available, reporting pass/fail
and emitting a JSON summary that external auditors can archive.
Intended for air-gapped or clean-environment verification.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


COMPOSE_FILE = "docker-compose.p3.yml"
COMMANDS: Dict[str, List[str]] = {
    "p3": ["docker", "compose", "-f", COMPOSE_FILE, "run", "--rm", "p3-harness"],
    "p4": ["docker", "compose", "-f", COMPOSE_FILE, "run", "--rm", "p4-harness"],
}

# Expected artifact roots for parity with external verification steps.
# These paths are relative to the container working dir (/app).
OUTPUT_PATHS = {
    "p3": None,  # P3 docker smoke is pytest-only and uses temp dirs.
    "p4": "results/p4_compose",
}

# In-container artifact directories for audit traceability.
P3_ARTIFACT_DIR = "/tmp/pytest-of-root"
P4_ARTIFACT_DIR = "/app/results/p4_compose"
P3_MOUNT_DIR = "/app/results/p3_pytest"

REASON_DOCKER_NOT_FOUND = "DOCKER_NOT_FOUND"
REASON_BIND_MOUNT_INVALID = "BIND_MOUNT_INVALID"
REASON_COPY_FAILED = "COPY_FAILED"
REASON_CODE_ORDER = (
    REASON_DOCKER_NOT_FOUND,
    REASON_BIND_MOUNT_INVALID,
    REASON_COPY_FAILED,
)

P3_PYTEST_AND_COPY_COMMAND = (
    "pytest tests/first_light/test_p3_harness_skeleton.py; "
    "pytest_status=$?; "
    f"mkdir -p {P3_MOUNT_DIR}; "
    f"cp -a {P3_ARTIFACT_DIR}/. {P3_MOUNT_DIR}/ 2>/dev/null; "
    "copy_status=$?; "
    'if [ "$copy_status" -eq 0 ]; then echo P3_COPY_OK=1; else echo P3_COPY_OK=0; fi; '
    "exit $pytest_status"
)

HOST_EXTRACTION_INSTRUCTIONS = {
    "p3": (
        "P3 docker smoke runs a pytest skeleton. Artifacts are written to "
        f"ephemeral tmp_path directories under {P3_ARTIFACT_DIR} inside the container. "
        "To persist them to the host, re-run this script with "
        "--p3-bind-mount <HOST_DIR>, which bind-mounts the host dir to "
        f"{P3_MOUNT_DIR} and copies pytest artifacts there. The JSON field "
        "p3_host_dir records the resolved host path actually mounted."
    ),
    "p4": (
        "Bind-mount a host directory to the container artifact dir to retain outputs. "
        "Example:\n"
        "  docker compose -f docker-compose.p3.yml run --rm "
        "-v <HOST_DIR>:/app/results/p4_compose p4-harness\n"
        "Artifacts will appear on the host under <HOST_DIR>/."
    ),
}

CONTAINER_IMAGE_TAGS = {
    "p3": "docker-compose.p3.yml#p3-harness",
    "p4": "docker-compose.p3.yml#p4-harness",
}


def docker_available() -> bool:
    """Return True if docker CLI is available in PATH."""
    return shutil.which("docker") is not None


def _excerpt(text: str, limit: int = 500) -> str:
    """Return a shortened excerpt for stderr logging."""
    snippet = text.strip()
    if len(snippet) > limit:
        return snippet[: limit - 3] + "..."
    return snippet


def run_command(cmd: List[str]) -> Tuple[bool, subprocess.CompletedProcess]:
    """Run a shell command and return success flag with process result."""
    print(f"[INFO] Running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"[ERROR] Exit code {proc.returncode} for: {' '.join(cmd)}")
    return proc.returncode == 0, proc


def _normalize_bind_mount_dir(raw: str) -> str:
    """Normalize a host path for docker bind-mount specs (portable + absolute)."""
    expanded = os.path.expandvars(raw)
    resolved = Path(expanded).expanduser().resolve()
    return resolved.as_posix()


def _canonicalize_reason_codes(codes: Sequence[str]) -> list[str]:
    unique = {code for code in codes}
    ordered = [code for code in REASON_CODE_ORDER if code in unique]
    extras = sorted(unique.difference(REASON_CODE_ORDER))
    return [*ordered, *extras]


def _looks_like_bind_mount_error(stderr: str) -> bool:
    lowered = (stderr or "").lower()
    patterns = (
        "invalid volume specification",
        "invalid mount",
        "mount denied",
        "bind source path does not exist",
        "error while creating mount source path",
        "is not shared from osx and is not known to docker",
        "the system cannot find the path specified",
    )
    return any(pattern in lowered for pattern in patterns)


def _format_one_line_summary(summary: dict) -> str:
    parts = [
        f"docker_found={json.dumps(summary.get('docker_found'), ensure_ascii=True)}",
        f"p3_ok={json.dumps(summary.get('p3_ok'), ensure_ascii=True)}",
        f"p4_ok={json.dumps(summary.get('p4_ok'), ensure_ascii=True)}",
        f"reason_codes={json.dumps(summary.get('reason_codes', []), ensure_ascii=True)}",
        f"p3_host_dir={json.dumps(summary.get('p3_host_dir'), ensure_ascii=True)}",
    ]
    return "SMOKE_SUMMARY " + " ".join(parts)


def _build_commands(p3_bind_mount: str | None) -> Dict[str, List[str]]:
    commands: Dict[str, List[str]] = {key: list(cmd) for key, cmd in COMMANDS.items()}
    if p3_bind_mount:
        commands["p3"] = [
            "docker",
            "compose",
            "-f",
            COMPOSE_FILE,
            "run",
            "--rm",
            "-v",
            f"{p3_bind_mount}:{P3_MOUNT_DIR}",
            "p3-harness",
            "sh",
            "-c",
            P3_PYTEST_AND_COPY_COMMAND,
        ]
    return commands


def run_smoke(p3_bind_mount: str | None = None) -> Tuple[dict, bool]:
    """Execute smoke tests and produce a summary dictionary."""
    reason_codes: list[str] = []
    normalized_p3_bind_mount: str | None = None
    if p3_bind_mount:
        try:
            normalized_p3_bind_mount = _normalize_bind_mount_dir(p3_bind_mount)
            Path(normalized_p3_bind_mount).mkdir(parents=True, exist_ok=True)
        except (OSError, ValueError):
            normalized_p3_bind_mount = None
            reason_codes.extend([REASON_BIND_MOUNT_INVALID, REASON_COPY_FAILED])

    summary = {
        "p3_ok": None,
        "p4_ok": None,
        "p3_artifact_dir": P3_ARTIFACT_DIR,
        "p4_artifact_dir": P4_ARTIFACT_DIR,
        "p3_host_dir": normalized_p3_bind_mount,
        "output_paths": OUTPUT_PATHS,
        "container_image_tags": CONTAINER_IMAGE_TAGS,
        "stderr_excerpts": {"p3": None, "p4": None},
        "host_extraction_instructions": HOST_EXTRACTION_INSTRUCTIONS,
        "reason_codes": [],
        "docker_found": False,
    }

    if not docker_available():
        print("[WARN] Docker CLI not found in PATH; skipping container smoke tests.")
        reason_codes.append(REASON_DOCKER_NOT_FOUND)
        summary["reason_codes"] = _canonicalize_reason_codes(reason_codes)
        return summary, True

    summary["docker_found"] = True
    overall_success = True
    commands = _build_commands(normalized_p3_bind_mount)
    for key, cmd in commands.items():
        success, proc = run_command(cmd)
        summary[f"{key}_ok"] = success
        summary["stderr_excerpts"][key] = _excerpt(proc.stderr or "")
        if key == "p3" and normalized_p3_bind_mount:
            stdout = proc.stdout or ""
            if "P3_COPY_OK=0" in stdout:
                reason_codes.append(REASON_COPY_FAILED)
            if not success and _looks_like_bind_mount_error(proc.stderr or ""):
                reason_codes.extend([REASON_BIND_MOUNT_INVALID, REASON_COPY_FAILED])
        status = "PASS" if success else "FAIL"
        print(f"[RESULT] {' '.join(cmd)} -> {status}")
        overall_success = overall_success and success

    summary["reason_codes"] = _canonicalize_reason_codes(reason_codes)
    return summary, overall_success


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run containerized First Light harness smoke tests.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--p3-bind-mount",
        dest="p3_bind_mount",
        type=str,
        default=None,
        help="Host directory to bind-mount for persisting P3 pytest artifacts.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    parsed = parser.parse_args(list(argv) if argv is not None else None)
    summary, overall_success = run_smoke(p3_bind_mount=parsed.p3_bind_mount)
    print(json.dumps(summary, indent=2))
    if overall_success:
        print("[INFO] Docker harness smoke tests completed successfully or were skipped.")
        print(_format_one_line_summary(summary))
        return 0

    print("[WARN] One or more docker harness smoke tests failed.")
    print(_format_one_line_summary(summary))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
