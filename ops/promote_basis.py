import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# --- Configuration ---
BASIS_DIR = Path("basis")
DOCS_DIR = Path("docs")
OPS_LOGS_DIR = Path("ops/logs")
EXPORTS_DIR = Path("exports")
PROMOTION_LOG_FILE = OPS_LOGS_DIR / "basis_promotions.jsonl"
MANIFEST_FILE = Path("spanning_set_manifest.json")
REPO_URL = "git@github.com:helpfuldolphin/mathledger.git"

TARGET_WHITELIST = [
    "basis",
    "docs",
    "tests",
    "pyproject.toml",
    "README.md",
    "LICENSE",
    ".gitignore",
    ".git",
]

SECURITY_PATTERNS = [
    "-----BEGIN PRIVATE KEY-----",
    "-----BEGIN RSA PRIVATE KEY-----",
    "-----BEGIN EC PRIVATE KEY-----",
    "-----BEGIN OPENSSH PRIVATE KEY-----",
    "sk_live_",
    "sk_test_",
    "AKIA",
    "ssh-rsa AAAA",
]

SLOP_DIR_NAMES = {
    "tmp",
    "temp",
    "artifacts",
    "build",
    "dist",
    ".cache",
    "__pycache__",
}

SLOP_FILE_SUFFIXES = {".bak", ".tmp", ".swp", ".old", ".orig"}
GATE_ORDER = ["first_organism", "determinism", "security", "lean"]
MAX_SECRET_SCAN_SIZE = 512 * 1024


def summarize_output(text: str, max_lines: int = 6) -> str:
    if not text:
        return ""
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return ""
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "\n".join(lines[-max_lines:])


def flatten_detail(text: str) -> str:
    if not text:
        return ""
    parts = [line.strip() for line in text.strip().splitlines() if line.strip()]
    return " | ".join(parts[:4])


def run_command(cmd: List[str], cwd: Optional[Path] = None, capture: bool = False) -> subprocess.CompletedProcess:
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            check=True,
            text=True,
            capture_output=capture,
        )
        return result
    except subprocess.CalledProcessError as err:
        if capture and err.stderr:
            print(f" [!] Command failed: {' '.join(cmd)}")
            print(f"     Error: {summarize_output(err.stderr)}")
        raise err


def calculate_dir_hash(directory: Path) -> str:
    sha256 = hashlib.sha256()
    if not directory.exists():
        return "0" * 64

    for root, _, files in sorted(os.walk(directory)):
        for names in sorted(files):
            filepath = Path(root) / names
            if "__pycache__" in filepath.parts:
                continue
            try:
                with open(filepath, "rb") as f:
                    while True:
                        data = f.read(65536)
                        if not data:
                            break
                        sha256.update(data)
                sha256.update(str(filepath.relative_to(directory)).encode("utf-8"))
            except OSError:
                pass
    return sha256.hexdigest()


def detect_slop(paths: List[Path]) -> List[str]:
    issues: List[str] = []
    for base in paths:
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if "__pycache__" in path.parts:
                continue
            name = path.name.lower()
            if path.is_dir() and name in SLOP_DIR_NAMES:
                issues.append(f"directory {path} is flagged as navigational slop")
            elif path.is_file():
                if path.suffix.lower() in SLOP_FILE_SUFFIXES or name.endswith("~"):
                    issues.append(f"file {path} carries slop suffix '{path.suffix}'")
    return issues


def scan_for_secrets(paths: List[Path]) -> List[str]:
    issues: List[str] = []
    for base in paths:
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if size > MAX_SECRET_SCAN_SIZE:
                continue
            try:
                content = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for pattern in SECURITY_PATTERNS:
                if pattern in content:
                    issues.append(f"{path}: contains fingerprint {pattern}")
    return issues


def summarize_list(items: List[str], limit: int = 3) -> str:
    if not items:
        return ""
    if len(items) <= limit:
        return "; ".join(items)
    return f"{'; '.join(items[:limit])} (+{len(items) - limit} more)"


def run_first_organism_test() -> Tuple[bool, str]:
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/integration/test_first_organism.py",
        "-v",
    ]
    try:
        completed = run_command(cmd, capture=True)
        detail = summarize_output(completed.stdout or completed.stderr)
        return True, detail or "First Organism tests passed."
    except subprocess.CalledProcessError as exc:
        detail = summarize_output(exc.stdout or exc.stderr or str(exc))
        return False, detail or "First Organism tests failed."


def run_determinism_gate(runs: int = 3) -> Tuple[bool, str]:
    if not MANIFEST_FILE.exists():
        return False, "Missing spanning_set_manifest.json"
    script_path = Path("backend") / "repro" / "determinism_cli.py"
    if not script_path.exists():
        return False, "Determinism CLI is missing."
    cmd = [
        sys.executable,
        str(script_path),
        "--directory",
        str(BASIS_DIR),
        "--runs",
        str(runs),
        "--manifest",
        str(MANIFEST_FILE),
    ]
    try:
        completed = run_command(cmd, capture=True)
        detail = summarize_output(completed.stdout or completed.stderr)
        return True, detail or "Determinism gate passed."
    except subprocess.CalledProcessError as exc:
        detail = summarize_output(exc.stdout or exc.stderr or str(exc))
        return False, detail or "Determinism gate failed."


def run_security_gate(paths: List[Path]) -> Tuple[bool, str]:
    slop = detect_slop(paths)
    secrets = scan_for_secrets(paths)
    details: List[str] = []
    if slop:
        details.append(f"Slop: {summarize_list(slop, 3)}")
    if secrets:
        details.append(f"Secrets: {summarize_list(secrets, 3)}")
    if details:
        return False, " | ".join(details)
    return True, "No slop or secret patterns detected."


def run_lean_verification() -> Tuple[bool, str]:
    lean_dir = Path("backend/lean_proj")
    if not lean_dir.exists():
        return False, "Lean project directory missing."
    lake_cmd = shutil.which("lake")
    if not lake_cmd:
        return False, "Lean CLI 'lake' not found in PATH."
    try:
        completed = run_command([lake_cmd, "build"], cwd=lean_dir, capture=True)
        detail = summarize_output(completed.stdout or completed.stderr)
        return True, detail or "Lean build succeeded."
    except subprocess.CalledProcessError as exc:
        detail = summarize_output(exc.stdout or exc.stderr or str(exc))
        return False, detail or "Lean build failed."


def mark_skipped_gates(statuses: Dict[str, Dict[str, str]], failed_gate: str) -> None:
    index = GATE_ORDER.index(failed_gate)
    for gate in GATE_ORDER[index + 1 :]:
        statuses[gate] = {
            "status": "NOT RUN",
            "detail": f"Skipped after {failed_gate} failure.",
        }


def check_gates() -> Tuple[bool, Dict[str, Dict[str, str]]]:
    statuses: Dict[str, Dict[str, str]] = {
        gate: {"status": "NOT RUN", "detail": "Awaiting execution."}
        for gate in GATE_ORDER
    }

    print("[-] Checking gates...")

    first_ok, first_detail = run_first_organism_test()
    statuses["first_organism"] = {
        "status": "PASS" if first_ok else "FAIL",
        "detail": first_detail,
    }
    print(f" [*] First Organism Integration: {'PASS' if first_ok else 'FAIL'}")
    if not first_ok:
        mark_skipped_gates(statuses, "first_organism")
        return False, statuses

    det_ok, det_detail = run_determinism_gate()
    statuses["determinism"] = {
        "status": "PASS" if det_ok else "FAIL",
        "detail": det_detail,
    }
    print(f" [*] Determinism CLI: {'PASS' if det_ok else 'FAIL'}")
    if not det_ok:
        mark_skipped_gates(statuses, "determinism")
        return False, statuses

    security_ok, security_detail = run_security_gate([BASIS_DIR, DOCS_DIR])
    statuses["security"] = {
        "status": "PASS" if security_ok else "FAIL",
        "detail": security_detail,
    }
    print(f" [*] Security Scan: {'PASS' if security_ok else 'FAIL'}")
    if not security_ok:
        mark_skipped_gates(statuses, "security")
        return False, statuses

    lean_ok, lean_detail = run_lean_verification()
    statuses["lean"] = {
        "status": "PASS" if lean_ok else "FAIL",
        "detail": lean_detail,
    }
    print(f" [*] Lean Build: {'PASS' if lean_ok else 'FAIL'}")
    if not lean_ok:
        mark_skipped_gates(statuses, "lean")
        return False, statuses

    return True, statuses


def create_snapshot(source_hash: str) -> Path:
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"basis_{timestamp}_{source_hash[:8]}.tar.gz"
    filepath = EXPORTS_DIR / filename

    print(f"[-] Creating snapshot: {filepath}")
    with tarfile.open(filepath, "w:gz") as tar:
        tar.add(BASIS_DIR, arcname="basis")
        if DOCS_DIR.exists():
            tar.add(DOCS_DIR, arcname="docs")
        for asset in ["pyproject.toml", "README.md"]:
            path = Path(asset)
            if path.exists():
                tar.add(path)
    return filepath


def emit_promotion_plan(
    target_dir: Path,
    snapshot_path: Path,
    source_hash: str,
    gate_statuses: Dict[str, Dict[str, str]],
) -> None:
    print("[-] Promotion Plan")
    print(f" [.] Source hash: {source_hash}")
    print(f" [.] Snapshot artifact: {snapshot_path.name}")
    print(f" [.] Target repository: {target_dir}")
    print(" [.] Whitelisted paths that will survive the wipe:")
    for entry in TARGET_WHITELIST:
        print(f"     - {entry}")
    print(" [.] Gates summary:")
    for gate, info in gate_statuses.items():
        print(f"     - {gate}: {info['status']} ({flatten_detail(info['detail'])})")


def prepare_target_repo(target_dir: Path) -> None:
    print(f"[-] Preparing target repository at {target_dir}...")
    if target_dir.exists():
        if (target_dir / ".git").exists():
            print(" [.] Updating existing repository...")
            run_command(["git", "fetch", "origin"], cwd=target_dir)
        else:
            print(" [!] Target directory exists but is not a git repo. Aborting.")
            sys.exit(1)
    else:
        print(f" [.] Cloning from {REPO_URL}...")
        run_command(["git", "clone", REPO_URL, str(target_dir)])


def materialize(snapshot_path: Path, target_dir: Path) -> None:
    print(f"[-] Materializing into {target_dir}...")
    print(" [.] Wiping non-whitelisted files...")
    for item in target_dir.iterdir():
        if item.name not in TARGET_WHITELIST:
            if item.name == ".git":
                continue
            print(f"  - Removing {item.name}")
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
    print(" [.] Unpacking snapshot...")
    with tarfile.open(snapshot_path, "r:gz") as tar:
        tar.extractall(target_dir)
    print("  + Materialization complete.")


def generate_report(
    snapshot_path: Path,
    source_hash: str,
    target_dir: Path,
    gate_statuses: Dict[str, Dict[str, str]],
) -> None:
    report_path = target_dir / "basis_promotion_release_notes.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Basis Promotion Report\n\n")
        f.write(f"**Date:** {datetime.now(timezone.utc).isoformat()}\n")
        f.write(f"**Source Hash:** {source_hash}\n")
        f.write(f"**Snapshot:** {snapshot_path.name}\n\n")
        f.write("## Gate Results\n")
        for gate, info in gate_statuses.items():
            detail = flatten_detail(info["detail"])
            f.write(f"- **{gate.replace('_', ' ').title()}**: {info['status']} — {detail}\n")
        f.write("\n## Artifacts\n")
        f.write(f"- Snapshot: `{snapshot_path.name}`\n")
        f.write(f"- Manifest: `{MANIFEST_FILE.name}`\n\n")
        f.write("## Promotion Plan\n")
        f.write("- Reset target repo to the snapshot content.\n")
        f.write("- Remove non-whitelisted files before unpacking.\n")
        f.write("- Create a deterministic commit referencing the gate results.\n")
    print(f"[-] Report written to {report_path}")


def git_commit(target_dir: Path, source_hash: str) -> None:
    print("[-] Creating git commit...")
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    msg = f"promote: basis sync {source_hash[:8]} ({timestamp})"

    run_command(["git", "add", "."], cwd=target_dir)
    status = run_command(["git", "status", "--porcelain"], cwd=target_dir, capture=True)
    if not status.stdout.strip():
        print(" [!] No changes to commit.")
        return
    run_command(["git", "commit", "-m", msg], cwd=target_dir)
    print(f" [x] Commit created: {msg}")


def log_promotion(
    source_hash: str,
    snapshot_path: Optional[Path],
    success: bool,
    gate_statuses: Dict[str, Dict[str, str]],
    target_dir: Path,
) -> None:
    OPS_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_hash": source_hash,
        "snapshot": str(snapshot_path) if snapshot_path else None,
        "target_dir": str(target_dir),
        "status": "PASS" if success else "FAIL",
        "gates": {gate: info["status"] for gate, info in gate_statuses.items()},
        "gate_details": {gate: flatten_detail(info["detail"]) for gate, info in gate_statuses.items()},
    }
    with open(PROMOTION_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote Basis to Vault")
    parser.add_argument("--target-dir", required=True, type=Path, help="Path to the target git repository")
    parser.add_argument("--dry-run", action="store_true", help="Do not modify target")
    parser.add_argument("--skip-gates", action="store_true", help="Skip validation gates (DANGEROUS - DEV ONLY)")
    args = parser.parse_args()

    if not BASIS_DIR.exists():
        print(f"Error: {BASIS_DIR} not found.")
        sys.exit(1)

    source_hash = calculate_dir_hash(BASIS_DIR)
    print(f"[-] Basis Source Hash: {source_hash}")

    if args.skip_gates:
        gate_statuses = {
            gate: {"status": "SKIPPED", "detail": "Gate skipped by operator."}
            for gate in GATE_ORDER
        }
        gate_ok = True
    else:
        gate_ok, gate_statuses = check_gates()

    if not gate_ok:
        log_promotion(source_hash, None, False, gate_statuses, args.target_dir)
        print("Error: Gates failed.")
        sys.exit(1)

    snapshot = create_snapshot(source_hash)

    if args.dry_run:
        print("[DRY RUN] Skipping materialization.")
        log_promotion(source_hash, snapshot, False, gate_statuses, args.target_dir)
        sys.exit(0)

    prepare_target_repo(args.target_dir)
    emit_promotion_plan(args.target_dir, snapshot, source_hash, gate_statuses)
    materialize(snapshot, args.target_dir)
    generate_report(snapshot, source_hash, args.target_dir, gate_statuses)
    git_commit(args.target_dir, source_hash)
    log_promotion(source_hash, snapshot, True, gate_statuses, args.target_dir)
    print("[-] Promotion Sequence Complete. Ready for Human Review & Push.")


if __name__ == "__main__":
    main()

