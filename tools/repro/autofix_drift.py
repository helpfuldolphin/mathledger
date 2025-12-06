#!/usr/bin/env python3
"""
Autofix Drift - Automated determinism patch application and verification.

Applies docs/patches/determinism-fixes.diff and verifies with drift_sentinel.py
and seed_replay_guard.py. On failure, prints violation locations and recommended
replacements.

Usage:
    python tools/repro/autofix_drift.py --apply --verify
    python tools/repro/autofix_drift.py --verify-only
    python tools/repro/autofix_drift.py --dry-run

Exit Codes:
    0: Success (patch applied and verified, or already applied)
    1: Patch application failed
    2: Verification failed (drift detected or nondeterminism)
    3: Invalid arguments or missing files
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def get_repo_root() -> Path:
    """Get repository root directory."""
    current = Path(__file__).resolve()
    while current.parent != current:
        if (current / ".git").exists():
            return current
        current = current.parent
    raise RuntimeError("Could not find repository root (no .git directory)")


def check_patch_exists(repo_root: Path) -> bool:
    """Check if determinism-fixes.diff exists."""
    patch_file = repo_root / "docs" / "patches" / "determinism-fixes.diff"
    return patch_file.exists()


def check_already_applied(repo_root: Path) -> bool:
    """Check if patch is already applied by looking for deterministic imports."""
    derive_file = repo_root / "backend" / "axiom_engine" / "derive.py"
    if not derive_file.exists():
        return False
    
    content = derive_file.read_text()
    return "from backend.repro.determinism import" in content


def apply_patch(repo_root: Path, dry_run: bool = False) -> Tuple[bool, str]:
    """
    Apply determinism-fixes.diff patch.
    
    Returns:
        (success, message)
    """
    patch_file = repo_root / "docs" / "patches" / "determinism-fixes.diff"
    
    if not patch_file.exists():
        return False, f"Patch file not found: {patch_file}"
    
    if check_already_applied(repo_root):
        return True, "Patch already applied (deterministic imports detected)"
    
    cmd = ["git", "apply"]
    if dry_run:
        cmd.append("--check")
    cmd.append(str(patch_file))
    
    try:
        result = subprocess.run(
            cmd,
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            if dry_run:
                return True, "Patch can be applied (dry-run)"
            else:
                return True, "Patch applied successfully"
        else:
            error_msg = result.stderr or result.stdout
            return False, f"Patch application failed: {error_msg}"
    
    except subprocess.TimeoutExpired:
        return False, "Patch application timed out"
    except Exception as e:
        return False, f"Patch application error: {e}"


def run_drift_sentinel(repo_root: Path) -> Tuple[bool, List[Dict]]:
    """
    Run drift_sentinel.py to detect nondeterministic operations.
    
    Returns:
        (success, violations)
    """
    sentinel_script = repo_root / "tools" / "repro" / "drift_sentinel.py"
    whitelist_file = repo_root / "artifacts" / "repro" / "drift_whitelist.json"
    
    if not sentinel_script.exists():
        return False, [{"error": f"drift_sentinel.py not found: {sentinel_script}"}]
    
    cmd = [
        sys.executable,
        str(sentinel_script),
        "--all",
        "--whitelist", str(whitelist_file)
    ]
    
    try:
        result = subprocess.run(
            cmd,
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=120
        )
        
        violations = []
        current_file = None
        current_line = None
        current_pattern = None
        
        for line in result.stdout.split("\n"):
            line = line.strip()
            if line.startswith("backend/") and ":" in line:
                parts = line.split(":")
                if len(parts) >= 2:
                    current_file = parts[0]
                    current_line = parts[1]
            elif line.startswith("Pattern:"):
                current_pattern = line.replace("Pattern:", "").strip()
            elif line.startswith("Fix:"):
                fix = line.replace("Fix:", "").strip()
                if current_file and current_line and current_pattern:
                    violations.append({
                        "file": current_file,
                        "line": current_line,
                        "pattern": current_pattern,
                        "fix": fix
                    })
                    current_file = None
                    current_line = None
                    current_pattern = None
        
        success = result.returncode == 0
        return success, violations
    
    except subprocess.TimeoutExpired:
        return False, [{"error": "drift_sentinel.py timed out"}]
    except Exception as e:
        return False, [{"error": f"drift_sentinel.py error: {e}"}]


def run_replay_guard(repo_root: Path) -> Tuple[bool, str]:
    """
    Run seed_replay_guard.py to verify byte-identical runs.
    
    Returns:
        (success, message)
    """
    guard_script = repo_root / "tools" / "repro" / "seed_replay_guard.py"
    artifacts_path = repo_root / "artifacts" / "repro"
    
    if not guard_script.exists():
        return False, f"seed_replay_guard.py not found: {guard_script}"
    
    cmd = [
        sys.executable,
        str(guard_script),
        "--seed", "0",
        "--runs", "3",
        "--path", str(artifacts_path)
    ]
    
    try:
        result = subprocess.run(
            cmd,
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        success = result.returncode == 0
        message = result.stdout + result.stderr
        
        return success, message
    
    except subprocess.TimeoutExpired:
        return False, "seed_replay_guard.py timed out"
    except Exception as e:
        return False, f"seed_replay_guard.py error: {e}"


def print_violations(violations: List[Dict]):
    """Print violation details with recommended fixes."""
    if not violations:
        return
    
    print("\n" + "=" * 80)
    print("VIOLATIONS DETECTED")
    print("=" * 80)
    
    for v in violations:
        if "error" in v:
            print(f"\nERROR: {v['error']}")
            continue
        
        print(f"\nFile: {v['file']}:{v['line']}")
        print(f"  Pattern: {v['pattern']}")
        print(f"  Fix: {v['fix']}")
    
    print("\n" + "=" * 80)
    print("RECOMMENDED ACTION")
    print("=" * 80)
    print("\nApply determinism-fixes.diff to fix all violations:")
    print("  git apply docs/patches/determinism-fixes.diff")
    print("\nOr use autofix assistant:")
    print("  uv run python tools/repro/autofix_drift.py --apply --verify")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Autofix drift - Apply determinism patch and verify",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/repro/autofix_drift.py --apply --verify
  
  python tools/repro/autofix_drift.py --verify-only
  
  python tools/repro/autofix_drift.py --dry-run
        """
    )
    
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply determinism-fixes.diff patch"
    )
    
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run drift_sentinel.py and seed_replay_guard.py"
    )
    
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only run verification (no patch application)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check if patch can be applied without applying it"
    )
    
    args = parser.parse_args()
    
    try:
        repo_root = get_repo_root()
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 3
    
    if not any([args.apply, args.verify, args.verify_only, args.dry_run]):
        print("ERROR: Must specify --apply, --verify, --verify-only, or --dry-run", file=sys.stderr)
        parser.print_help()
        return 3
    
    if not check_patch_exists(repo_root):
        print("ERROR: Patch file not found: docs/patches/determinism-fixes.diff", file=sys.stderr)
        return 3
    
    exit_code = 0
    
    if args.apply or args.dry_run:
        print("=" * 80)
        print("APPLYING DETERMINISM PATCH")
        print("=" * 80)
        
        success, message = apply_patch(repo_root, dry_run=args.dry_run)
        print(f"\n{message}\n")
        
        if not success and not args.dry_run:
            exit_code = 1
            return exit_code
    
    if args.verify or args.verify_only:
        print("=" * 80)
        print("RUNNING DRIFT SENTINEL")
        print("=" * 80)
        
        success, violations = run_drift_sentinel(repo_root)
        
        if success:
            print("\n[PASS] Drift Sentinel: 0 violations\n")
        else:
            print("\n[FAIL] Drift Sentinel: violations detected\n")
            print_violations(violations)
            exit_code = 2
        
        print("=" * 80)
        print("RUNNING DETERMINISM GUARD")
        print("=" * 80)
        
        success, message = run_replay_guard(repo_root)
        print(f"\n{message}")
        
        if success:
            print("[PASS] Determinism Guard: 3/3 byte-identical runs\n")
        else:
            print("[FAIL] Determinism Guard: nondeterminism detected\n")
            exit_code = 2
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
