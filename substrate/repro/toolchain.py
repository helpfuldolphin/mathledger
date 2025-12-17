"""
Toolchain Snapshot and Fingerprint Computation

This module provides deterministic toolchain fingerprinting for experiment reproducibility.
It computes a single SHA-256 hash from all toolchain lock files, enabling verification
that two runs used identical toolchain configurations.

SAVE TO REPO: YES
Rationale: Core reproducibility infrastructure. Required for CAL-EXP-1 and successors.
"""

import hashlib
import json
import platform
import socket
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


@dataclass
class PythonToolchain:
    """Python toolchain version information."""
    version: str
    uv_version: str
    uv_lock_hash: str


@dataclass
class LeanToolchain:
    """Lean toolchain version information."""
    version: str
    toolchain_hash: str
    lake_manifest_hash: str
    lakefile_hash: str


@dataclass
class PlatformInfo:
    """Platform information for audit (not enforced for reproducibility)."""
    os: str
    arch: str
    hostname: str


@dataclass
class ToolchainSnapshot:
    """Complete toolchain snapshot with fingerprint."""
    schema_version: str
    fingerprint: str
    python: PythonToolchain
    lean: LeanToolchain
    platform: PlatformInfo

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "schema_version": self.schema_version,
            "fingerprint": self.fingerprint,
            "python": asdict(self.python),
            "lean": asdict(self.lean),
            "platform": asdict(self.platform),
        }


def _hash_file(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    if not path.exists():
        raise FileNotFoundError(f"Toolchain file not found: {path}")
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def _get_uv_version() -> str:
    """Get uv package manager version."""
    try:
        result = subprocess.run(
            ["uv", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            # Output format: "uv 0.8.16 (2de677b0d 2025-09-09)"
            # We want just the version number (second token)
            parts = result.stdout.strip().split()
            if len(parts) >= 2:
                return parts[1]  # "0.8.16"
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return "unknown"


def _get_python_version() -> str:
    """Get Python version string."""
    return platform.python_version()


def _get_lean_version(lean_toolchain_path: Path) -> str:
    """Read Lean version from toolchain file."""
    if lean_toolchain_path.exists():
        return lean_toolchain_path.read_text().strip()
    return "unknown"


def _get_platform_info() -> PlatformInfo:
    """Collect platform information for audit."""
    return PlatformInfo(
        os=platform.platform(),
        arch=platform.machine(),
        hostname=socket.gethostname(),
    )


def compute_toolchain_fingerprint(
    uv_lock_hash: str,
    lean_toolchain_hash: str,
    lake_manifest_hash: str,
    lakefile_hash: str,
) -> str:
    """
    Compute the canonical toolchain fingerprint.

    The fingerprint is SHA-256 of the concatenation of component hashes
    in canonical order: uv.lock, lean-toolchain, lake-manifest.json, lakefile.lean
    """
    combined = uv_lock_hash + lean_toolchain_hash + lake_manifest_hash + lakefile_hash
    return hashlib.sha256(combined.encode()).hexdigest()


def capture_toolchain_snapshot(repo_root: Optional[Path] = None) -> ToolchainSnapshot:
    """
    Capture a complete toolchain snapshot.

    Args:
        repo_root: Repository root directory. If None, attempts to find it
                   by walking up from current file location.

    Returns:
        ToolchainSnapshot with all version info and fingerprint.

    Raises:
        FileNotFoundError: If any required toolchain file is missing.
    """
    if repo_root is None:
        # Walk up from this file to find repo root (where uv.lock is)
        current = Path(__file__).resolve()
        for parent in current.parents:
            if (parent / "uv.lock").exists():
                repo_root = parent
                break
        if repo_root is None:
            raise FileNotFoundError("Cannot find repository root (no uv.lock found)")

    repo_root = Path(repo_root)

    # Define paths
    uv_lock_path = repo_root / "uv.lock"
    lean_toolchain_path = repo_root / "backend" / "lean_proj" / "lean-toolchain"
    lake_manifest_path = repo_root / "backend" / "lean_proj" / "lake-manifest.json"
    lakefile_path = repo_root / "backend" / "lean_proj" / "lakefile.lean"

    # Compute hashes
    uv_lock_hash = _hash_file(uv_lock_path)
    lean_toolchain_hash = _hash_file(lean_toolchain_path)
    lake_manifest_hash = _hash_file(lake_manifest_path)
    lakefile_hash = _hash_file(lakefile_path)

    # Compute fingerprint
    fingerprint = compute_toolchain_fingerprint(
        uv_lock_hash,
        lean_toolchain_hash,
        lake_manifest_hash,
        lakefile_hash,
    )

    # Build snapshot
    python_toolchain = PythonToolchain(
        version=_get_python_version(),
        uv_version=_get_uv_version(),
        uv_lock_hash=uv_lock_hash,
    )

    lean_toolchain = LeanToolchain(
        version=_get_lean_version(lean_toolchain_path),
        toolchain_hash=lean_toolchain_hash,
        lake_manifest_hash=lake_manifest_hash,
        lakefile_hash=lakefile_hash,
    )

    return ToolchainSnapshot(
        schema_version="1.0",
        fingerprint=fingerprint,
        python=python_toolchain,
        lean=lean_toolchain,
        platform=_get_platform_info(),
    )


def verify_toolchain_match(
    snapshot1: ToolchainSnapshot,
    snapshot2: ToolchainSnapshot,
    strict: bool = True,
) -> tuple[bool, list[str]]:
    """
    Verify two toolchain snapshots match.

    Args:
        snapshot1: First snapshot to compare.
        snapshot2: Second snapshot to compare.
        strict: If True, all hashes must match. If False, only fingerprint.

    Returns:
        Tuple of (match: bool, differences: list[str])
    """
    differences = []

    if snapshot1.fingerprint != snapshot2.fingerprint:
        differences.append(
            f"Fingerprint mismatch: {snapshot1.fingerprint[:16]}... vs {snapshot2.fingerprint[:16]}..."
        )

    if strict:
        if snapshot1.python.uv_lock_hash != snapshot2.python.uv_lock_hash:
            differences.append("uv.lock hash mismatch")
        if snapshot1.lean.toolchain_hash != snapshot2.lean.toolchain_hash:
            differences.append("lean-toolchain hash mismatch")
        if snapshot1.lean.lake_manifest_hash != snapshot2.lean.lake_manifest_hash:
            differences.append("lake-manifest.json hash mismatch")
        if snapshot1.lean.lakefile_hash != snapshot2.lean.lakefile_hash:
            differences.append("lakefile.lean hash mismatch")

    return len(differences) == 0, differences


def save_toolchain_snapshot(snapshot: ToolchainSnapshot, output_path: Path) -> None:
    """Save toolchain snapshot to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(snapshot.to_dict(), f, indent=2)


def load_toolchain_snapshot(input_path: Path) -> ToolchainSnapshot:
    """Load toolchain snapshot from JSON file."""
    with open(input_path) as f:
        data = json.load(f)

    return ToolchainSnapshot(
        schema_version=data["schema_version"],
        fingerprint=data["fingerprint"],
        python=PythonToolchain(**data["python"]),
        lean=LeanToolchain(**data["lean"]),
        platform=PlatformInfo(**data["platform"]),
    )


if __name__ == "__main__":
    # CLI: capture and print current toolchain snapshot
    import argparse

    parser = argparse.ArgumentParser(description="Capture toolchain snapshot")
    parser.add_argument("--output", "-o", type=Path, help="Output JSON file")
    parser.add_argument("--verify", "-v", type=Path, help="Verify against baseline snapshot")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    snapshot = capture_toolchain_snapshot()

    if args.verify:
        baseline = load_toolchain_snapshot(args.verify)
        match, diffs = verify_toolchain_match(baseline, snapshot)
        if match:
            print("PASS: Toolchain matches baseline")
            sys.exit(0)
        else:
            print("FAIL: Toolchain mismatch")
            for diff in diffs:
                print(f"  - {diff}")
            sys.exit(1)

    if args.output:
        save_toolchain_snapshot(snapshot, args.output)
        print(f"Snapshot saved to {args.output}")
    elif args.json:
        print(json.dumps(snapshot.to_dict(), indent=2))
    else:
        print("Toolchain Snapshot")
        print("=" * 60)
        print(f"Fingerprint: {snapshot.fingerprint}")
        print()
        print("Python:")
        print(f"  Version:      {snapshot.python.version}")
        print(f"  uv Version:   {snapshot.python.uv_version}")
        print(f"  uv.lock Hash: {snapshot.python.uv_lock_hash[:16]}...")
        print()
        print("Lean:")
        print(f"  Version:           {snapshot.lean.version}")
        print(f"  Toolchain Hash:    {snapshot.lean.toolchain_hash[:16]}...")
        print(f"  Lake Manifest Hash: {snapshot.lean.lake_manifest_hash[:16]}...")
        print(f"  Lakefile Hash:     {snapshot.lean.lakefile_hash[:16]}...")
        print()
        print("Platform (audit only):")
        print(f"  OS:       {snapshot.platform.os}")
        print(f"  Arch:     {snapshot.platform.arch}")
        print(f"  Hostname: {snapshot.platform.hostname}")
