#!/usr/bin/env python3
"""
Autofix Drift V3.2 - File-local patches with confirm list for core modules.

Extends V3 with:
- File-local patch generation per function
- Confirm list: auto-apply only to core derivation modules
- Safe patch generation for crypto/auth modules (no auto-apply)
- RFC8785 signed manifest

Usage:
    python tools/repro/autofix_drift_v3_2.py --generate-patches
    python tools/repro/autofix_drift_v3_2.py --apply-confirmed --verify
    python tools/repro/autofix_drift_v3_2.py --dry-run

Exit Codes:
    0: Success (patches generated/applied and verified)
    1: Patch generation/application failed
    2: Verification failed (drift detected or nondeterminism)
    3: Invalid arguments or missing files
"""

import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

from backend.repro.determinism import deterministic_isoformat

CONFIRM_LIST = {
    "backend/axiom_engine/derive.py",
    "backend/axiom_engine/policy.py",
    "backend/ledger/blocking.py",
    "backend/axiom_engine/model.py"
}

MANUAL_REVIEW_PATTERNS = [
    r"backend/crypto/.*",
    r"backend/.*auth.*"
]

REPAIR_PATTERNS = {
    "datetime.utcnow": {
        "pattern": r"datetime\.(?:datetime\.)?utcnow\(\)",
        "replacement": "deterministic_timestamp(_GLOBAL_SEED)",
        "import": "from backend.repro.determinism import deterministic_timestamp",
        "requires_seed": True
    },
    "datetime.now": {
        "pattern": r"datetime\.(?:datetime\.)?now\(\)",
        "replacement": "deterministic_timestamp(_GLOBAL_SEED)",
        "import": "from backend.repro.determinism import deterministic_timestamp",
        "requires_seed": True
    },
    "time.time": {
        "pattern": r"time\.time\(\)",
        "replacement": "deterministic_unix_timestamp(_GLOBAL_SEED)",
        "import": "from backend.repro.determinism import deterministic_unix_timestamp",
        "requires_seed": True
    },
    "uuid.uuid4": {
        "pattern": r"uuid\.uuid4\(\)",
        "replacement": "deterministic_uuid(str(content))",
        "import": "from backend.repro.determinism import deterministic_uuid",
        "requires_seed": False,
        "note": "Replace 'content' with appropriate content-based identifier"
    },
    "np.random.random": {
        "pattern": r"np\.random\.random\(",
        "replacement": "SeededRNG(_GLOBAL_SEED).random(",
        "import": "from backend.repro.determinism import SeededRNG",
        "requires_seed": True
    },
    "np.random.rand": {
        "pattern": r"np\.random\.rand\(",
        "replacement": "SeededRNG(_GLOBAL_SEED).rand(",
        "import": "from backend.repro.determinism import SeededRNG",
        "requires_seed": True
    },
    "random.random": {
        "pattern": r"random\.random\(\)",
        "replacement": "SeededRNG(_GLOBAL_SEED).random()",
        "import": "from backend.repro.determinism import SeededRNG",
        "requires_seed": True
    },
    "random.randint": {
        "pattern": r"random\.randint\(",
        "replacement": "SeededRNG(_GLOBAL_SEED).randint(",
        "import": "from backend.repro.determinism import SeededRNG",
        "requires_seed": True
    },
    "os.urandom": {
        "pattern": r"os\.urandom\(",
        "replacement": "SeededRNG(_GLOBAL_SEED).bytes(",
        "import": "from backend.repro.determinism import SeededRNG",
        "requires_seed": True,
        "note": "OS entropy replaced with seeded RNG - REVIEW FOR CRYPTO USAGE"
    }
}


def get_repo_root() -> Path:
    """Get repository root directory."""
    current = Path(__file__).resolve()
    while current.parent != current:
        if (current / ".git").exists():
            return current
        current = current.parent
    raise RuntimeError("Could not find repository root (no .git directory)")


def rfc8785_canonicalize(data: Dict) -> str:
    """RFC 8785 canonical JSON serialization."""
    return json.dumps(data, sort_keys=True, separators=(',', ':'), ensure_ascii=True)


def compute_sha256(content: str) -> str:
    """Compute SHA-256 hash of content."""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def is_manual_review_required(file_path: str) -> bool:
    """Check if file requires manual review (crypto/auth)."""
    for pattern in MANUAL_REVIEW_PATTERNS:
        if re.match(pattern, file_path):
            return True
    return False


def is_confirmed_module(file_path: str) -> bool:
    """Check if file is in confirm list for auto-apply."""
    return file_path in CONFIRM_LIST


def scan_file_for_violations(file_path: Path, whitelist: List[str]) -> List[Dict]:
    """Scan a Python file for nondeterministic operations."""
    rel_path = str(file_path).replace(str(get_repo_root()) + "/", "")
    if rel_path in whitelist:
        return []
    
    violations = []
    
    try:
        content = file_path.read_text()
        lines = content.split("\n")
        
        for line_num, line in enumerate(lines, start=1):
            for pattern_name, pattern_info in REPAIR_PATTERNS.items():
                if re.search(pattern_info["pattern"], line):
                    violations.append({
                        "file": rel_path,
                        "line": line_num,
                        "pattern": pattern_name,
                        "original": line.strip(),
                        "replacement": pattern_info["replacement"],
                        "import": pattern_info["import"],
                        "requires_seed": pattern_info.get("requires_seed", False),
                        "note": pattern_info.get("note", "")
                    })
    
    except Exception as e:
        print(f"Warning: Could not scan {file_path}: {e}", file=sys.stderr)
    
    return violations


def generate_file_local_patch(file_path: Path, violations: List[Dict], repo_root: Path) -> Optional[str]:
    """Generate file-local patch for violations."""
    if not violations:
        return None
    
    try:
        content = file_path.read_text()
        lines = content.split("\n")
        
        required_imports = set()
        requires_seed = False
        
        for violation in violations:
            line_num = violation["line"] - 1
            if line_num < len(lines):
                pattern = REPAIR_PATTERNS[violation["pattern"]]["pattern"]
                replacement = violation["replacement"]
                lines[line_num] = re.sub(pattern, replacement, lines[line_num])
                
                required_imports.add(violation["import"])
                if violation["requires_seed"]:
                    requires_seed = True
        
        import_line = 0
        in_docstring = False
        for i, line in enumerate(lines):
            stripped = line.strip()
            if i == 0 and (stripped.startswith('"""') or stripped.startswith("'''")):
                in_docstring = True
            if in_docstring and (stripped.endswith('"""') or stripped.endswith("'''")):
                in_docstring = False
                import_line = i + 1
                break
            if not in_docstring and (stripped.startswith("import ") or stripped.startswith("from ")):
                import_line = i
                break
        
        for imp in sorted(required_imports):
            if imp not in content:
                lines.insert(import_line, imp)
                import_line += 1
        
        if requires_seed and "_GLOBAL_SEED" not in content:
            lines.insert(import_line, "")
            lines.insert(import_line + 1, "_GLOBAL_SEED = 0")
            lines.insert(import_line + 2, "")
        
        new_content = "\n".join(lines)
        
        rel_path = str(file_path.relative_to(repo_root))
        
        import difflib
        diff_lines = list(difflib.unified_diff(
            content.splitlines(keepends=False),
            new_content.splitlines(keepends=False),
            fromfile=f"a/{rel_path}",
            tofile=f"b/{rel_path}",
            lineterm=""
        ))
        
        if not diff_lines:
            return None
        
        return "\n".join(diff_lines) + "\n"
    
    except Exception as e:
        print(f"Error generating patch for {file_path}: {e}", file=sys.stderr)
        return None


def generate_patches(repo_root: Path, whitelist: List[str], dry_run: bool = False) -> Tuple[bool, Dict]:
    """Generate file-local patches with confirm list classification."""
    backend_dir = repo_root / "backend"
    if not backend_dir.exists():
        return False, {"error": "backend directory not found"}
    
    all_violations = {}
    for py_file in backend_dir.rglob("*.py"):
        violations = scan_file_for_violations(py_file, whitelist)
        if violations:
            all_violations[str(py_file)] = violations
    
    if not all_violations:
        return True, {
            "status": "clean",
            "message": "No violations detected",
            "files_scanned": len(list(backend_dir.rglob("*.py"))),
            "violations": 0
        }
    
    patch_dir = repo_root / "artifacts" / "repro" / "patches"
    patch_dir.mkdir(parents=True, exist_ok=True)
    
    confirmed_patches = {}
    manual_review_patches = {}
    
    for file_path_str, violations in all_violations.items():
        file_path = Path(file_path_str)
        rel_path = str(file_path).replace(str(repo_root) + "/", "")
        
        patch_content = generate_file_local_patch(file_path, violations, repo_root)
        
        if patch_content:
            patch_filename = rel_path.replace("/", "_") + ".patch"
            patch_file = patch_dir / patch_filename
            
            if not dry_run:
                patch_file.write_text(patch_content)
            
            patch_info = {
                "patch_file": str(patch_file),
                "violations": len(violations),
                "hash": compute_sha256(patch_content),
                "manual_review": is_manual_review_required(rel_path),
                "confirmed": is_confirmed_module(rel_path)
            }
            
            if is_manual_review_required(rel_path):
                manual_review_patches[rel_path] = patch_info
            elif is_confirmed_module(rel_path):
                confirmed_patches[rel_path] = patch_info
            else:
                manual_review_patches[rel_path] = patch_info
    
    manifest_timestamp = deterministic_isoformat(
        "autofix_manifest_v3_2",
        sorted(all_violations.keys()),
        sum(len(v) for v in all_violations.values()),
        dry_run
    )

    manifest = {
        "version": "3.2.0",
        "timestamp": manifest_timestamp,
        "repo_root": str(repo_root),
        "files_scanned": len(list(backend_dir.rglob("*.py"))),
        "files_with_violations": len(all_violations),
        "total_violations": sum(len(v) for v in all_violations.values()),
        "confirmed_patches": confirmed_patches,
        "manual_review_patches": manual_review_patches,
        "confirm_list": list(CONFIRM_LIST),
        "dry_run": dry_run
    }
    
    canonical_json = rfc8785_canonicalize(manifest)
    manifest["signature"] = compute_sha256(canonical_json)
    
    if not dry_run:
        manifest_file = repo_root / "artifacts" / "repro" / "autofix_manifest.json"
        manifest_file.write_text(rfc8785_canonicalize(manifest))
    
    return True, manifest


def apply_confirmed_patches(repo_root: Path, manifest_path: Path) -> Tuple[bool, str]:
    """Apply patches from confirm list only."""
    if not manifest_path.exists():
        return False, f"Manifest not found: {manifest_path}"
    
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    
    confirmed_patches = manifest.get("confirmed_patches", {})
    
    if not confirmed_patches:
        return True, "No confirmed patches to apply"
    
    applied = []
    failed = []
    
    for file_path, patch_info in confirmed_patches.items():
        patch_file = Path(patch_info["patch_file"])
        
        if not patch_file.exists():
            failed.append(f"{file_path}: patch file not found")
            continue
        
        cmd = ["git", "apply", str(patch_file)]
        
        try:
            result = subprocess.run(
                cmd,
                cwd=repo_root,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                applied.append(file_path)
            else:
                failed.append(f"{file_path}: {result.stderr}")
        
        except Exception as e:
            failed.append(f"{file_path}: {e}")
    
    if failed:
        return False, f"Applied {len(applied)}, failed {len(failed)}: {', '.join(failed)}"
    else:
        return True, f"Applied {len(applied)} confirmed patches successfully"


def load_whitelist(repo_root: Path) -> List[str]:
    """Load whitelist from drift_whitelist.json."""
    whitelist_file = repo_root / "artifacts" / "repro" / "drift_whitelist.json"
    
    if not whitelist_file.exists():
        return []
    
    try:
        with open(whitelist_file, "r") as f:
            data = json.load(f)
            return data.get("whitelist", [])
    except Exception as e:
        print(f"Warning: Could not load whitelist: {e}", file=sys.stderr)
        return []


def run_drift_sentinel(repo_root: Path) -> Tuple[bool, str]:
    """Run drift_sentinel.py."""
    sentinel_script = repo_root / "tools" / "repro" / "drift_sentinel.py"
    whitelist_file = repo_root / "artifacts" / "repro" / "drift_whitelist.json"
    
    if not sentinel_script.exists():
        return False, f"drift_sentinel.py not found: {sentinel_script}"
    
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
        
        return result.returncode == 0, result.stdout + result.stderr
    
    except Exception as e:
        return False, f"drift_sentinel.py error: {e}"


def run_replay_guard(repo_root: Path) -> Tuple[bool, str]:
    """Run seed_replay_guard.py."""
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
        
        return result.returncode == 0, result.stdout + result.stderr
    
    except Exception as e:
        return False, f"seed_replay_guard.py error: {e}"


def main():
    parser = argparse.ArgumentParser(
        description="Autofix Drift V3.2 - File-local patches with confirm list",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/repro/autofix_drift_v3_2.py --generate-patches
  
  python tools/repro/autofix_drift_v3_2.py --apply-confirmed --verify
  
  python tools/repro/autofix_drift_v3_2.py --dry-run
        """
    )
    
    parser.add_argument(
        "--generate-patches",
        action="store_true",
        help="Generate file-local patches for all violations"
    )
    
    parser.add_argument(
        "--apply-confirmed",
        action="store_true",
        help="Apply patches from confirm list only"
    )
    
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run drift_sentinel.py and seed_replay_guard.py"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be patched without applying"
    )
    
    args = parser.parse_args()
    
    try:
        repo_root = get_repo_root()
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 3
    
    whitelist = load_whitelist(repo_root)
    
    exit_code = 0
    
    if args.generate_patches or args.dry_run:
        print("=" * 80)
        print("GENERATING FILE-LOCAL PATCHES")
        print("=" * 80)
        
        success, manifest = generate_patches(repo_root, whitelist, dry_run=args.dry_run)
        
        if success:
            if manifest.get("status") == "clean":
                print(f"\n[PASS] {manifest['message']}")
                print(f"Files scanned: {manifest['files_scanned']}")
            else:
                print(f"\nFiles scanned: {manifest['files_scanned']}")
                print(f"Files with violations: {manifest['files_with_violations']}")
                print(f"Total violations: {manifest['total_violations']}")
                print(f"\nConfirmed patches (auto-apply): {len(manifest['confirmed_patches'])}")
                print(f"Manual review patches: {len(manifest['manual_review_patches'])}")
                print(f"Manifest signature: {manifest['signature']}")
                
                if not args.dry_run:
                    print(f"\nManifest saved: artifacts/repro/autofix_manifest.json")
                    print(f"Patches saved: artifacts/repro/patches/")
                    
                    print("\nConfirmed modules (auto-apply):")
                    for file_path in manifest['confirmed_patches'].keys():
                        print(f"  - {file_path}")
                    
                    print("\nManual review required:")
                    for file_path in manifest['manual_review_patches'].keys():
                        print(f"  - {file_path}")
        else:
            print(f"\n[FAIL] Patch generation failed: {manifest.get('error', 'Unknown error')}")
            exit_code = 1
    
    if args.apply_confirmed:
        print("\n" + "=" * 80)
        print("APPLYING CONFIRMED PATCHES")
        print("=" * 80)
        
        manifest_path = repo_root / "artifacts" / "repro" / "autofix_manifest.json"
        success, message = apply_confirmed_patches(repo_root, manifest_path)
        print(f"\n{message}")
        
        if not success:
            exit_code = 1
    
    if args.verify:
        print("\n" + "=" * 80)
        print("RUNNING DRIFT SENTINEL")
        print("=" * 80)
        
        success, output = run_drift_sentinel(repo_root)
        print(output)
        
        if success:
            print("[PASS] Drift Sentinel: 0 violations")
        else:
            print("[FAIL] Drift Sentinel: violations detected")
            exit_code = 2
        
        print("\n" + "=" * 80)
        print("RUNNING DETERMINISM GUARD")
        print("=" * 80)
        
        success, output = run_replay_guard(repo_root)
        print(output)
        
        if success:
            print("[PASS] Determinism Guard: 3/3 byte-identical runs")
        else:
            print("[FAIL] Determinism Guard: nondeterminism detected")
            exit_code = 2
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
