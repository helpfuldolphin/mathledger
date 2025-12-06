#!/usr/bin/env python3
"""
Generate Determinism Attestation - Produce RFC8785 signed attestation after clean run.

Creates artifacts/repro/determinism_attestation.json with:
- Counts by pattern and module
- Whitelist summary
- Seed replay hash
- RFC 8785 + SHA256 seal

Usage:
    python tools/repro/generate_attestation.py --seed 0
    python tools/repro/generate_attestation.py --verify

Exit Codes:
    0: Success (attestation generated and verified)
    1: Attestation generation failed
    2: Verification failed
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from backend.repro.determinism import (
    compute_sha256,
    deterministic_isoformat,
    deterministic_slug,
    rfc8785_canonicalize,
)

def get_repo_root() -> Path:
    """Get repository root directory."""
    current = Path(__file__).resolve()
    while current.parent != current:
        if (current / ".git").exists():
            return current
        current = current.parent
    raise RuntimeError("Could not find repository root (no .git directory)")


def load_drift_report(repo_root: Path) -> Dict:
    """Load drift report if it exists."""
    report_path = repo_root / "artifacts" / "repro" / "drift_report.json"
    
    if not report_path.exists():
        return {"status": "CLEAN", "violation_count": 0, "violations": []}
    
    with open(report_path, "r") as f:
        return json.load(f)


def load_whitelist(repo_root: Path) -> Dict:
    """Load whitelist configuration."""
    whitelist_path = repo_root / "artifacts" / "repro" / "drift_whitelist.json"
    
    if not whitelist_path.exists():
        return {"whitelist": [], "function_whitelist": []}
    
    with open(whitelist_path, "r") as f:
        return json.load(f)


def run_replay_guard(repo_root: Path, seed: int, runs: int) -> Tuple[bool, str]:
    """Run seed replay guard and capture hash."""
    guard_script = repo_root / "tools" / "repro" / "seed_replay_guard.py"
    artifacts_path = repo_root / "artifacts" / "repro"
    
    if not guard_script.exists():
        return False, "N/A"
    
    cmd = [
        sys.executable,
        str(guard_script),
        "--seed", str(seed),
        "--runs", str(runs),
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
        
        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if "sha256=" in line:
                    hash_part = line.split("sha256=")[1].strip()
                    return True, hash_part
            return True, "PASS"
        else:
            return False, "FAIL"
    
    except Exception as e:
        return False, f"ERROR: {e}"


def count_violations_by_pattern(violations: List[Dict]) -> Dict[str, int]:
    """Count violations by pattern type."""
    counts = {}
    for v in violations:
        pattern = v.get("pattern", "unknown")
        counts[pattern] = counts.get(pattern, 0) + 1
    return counts


def count_violations_by_module(violations: List[Dict]) -> Dict[str, int]:
    """Count violations by module."""
    counts = {}
    for v in violations:
        file_path = v.get("file", "unknown")
        module = file_path.split("/")[0] if "/" in file_path else file_path
        counts[module] = counts.get(module, 0) + 1
    return counts


def load_previous_attestation(repo_root: Path) -> str:
    """Load previous attestation signature for chaining."""
    attestation_path = repo_root / "artifacts" / "repro" / "determinism_attestation.json"
    
    if not attestation_path.exists():
        return "N/A"
    
    try:
        with open(attestation_path, "r") as f:
            prev_attestation = json.load(f)
            return prev_attestation.get("signature", "N/A")
    except Exception:
        return "N/A"


def generate_attestation(repo_root: Path, seed: int, runs: int) -> Dict:
    """Generate determinism attestation with chain linkage."""
    drift_report = load_drift_report(repo_root)
    whitelist_config = load_whitelist(repo_root)
    
    violations = drift_report.get("violations", [])
    
    replay_success, replay_hash = run_replay_guard(repo_root, seed, runs)
    
    prev_signature = load_previous_attestation(repo_root)
    
    attestation_timestamp = deterministic_isoformat(
        "determinism_attestation",
        seed,
        runs,
        replay_hash,
        drift_report.get("violation_count", 0),
    )

    attestation = {
        "version": "1.0.0",
        "timestamp": attestation_timestamp,
        "status": "CLEAN" if drift_report.get("violation_count", 0) == 0 else "DRIFT_DETECTED",
        "seed": seed,
        "replay_runs": runs,
        "replay_hash": replay_hash,
        "replay_success": replay_success,
        "prev_signature": prev_signature,
        "violation_summary": {
            "total": len(violations),
            "by_pattern": count_violations_by_pattern(violations),
            "by_module": count_violations_by_module(violations)
        },
        "whitelist_summary": {
            "file_count": len(whitelist_config.get("whitelist", [])),
            "function_count": len(whitelist_config.get("function_whitelist", [])),
            "files": whitelist_config.get("whitelist", [])
        },
        "determinism_score": 100 if len(violations) == 0 else max(0, 100 - len(violations)),
        "proof_or_abstain": "PROOF" if replay_success and len(violations) == 0 else "ABSTAIN"
    }
    
    canonical_json = rfc8785_canonicalize(attestation)
    attestation["signature"] = compute_sha256(canonical_json)
    
    return attestation


def verify_attestation(repo_root: Path) -> Tuple[bool, str]:
    """Verify existing attestation signature."""
    attestation_path = repo_root / "artifacts" / "repro" / "determinism_attestation.json"
    
    if not attestation_path.exists():
        return False, "Attestation file not found"
    
    with open(attestation_path, "r") as f:
        attestation = json.load(f)
    
    stored_signature = attestation.pop("signature", None)
    
    if not stored_signature:
        return False, "No signature found in attestation"
    
    canonical_json = rfc8785_canonicalize(attestation)
    computed_signature = compute_sha256(canonical_json)
    
    if stored_signature == computed_signature:
        return True, f"Signature verified: {stored_signature}"
    else:
        return False, f"Signature mismatch: stored={stored_signature}, computed={computed_signature}"


def main():
    parser = argparse.ArgumentParser(
        description="Generate determinism attestation with RFC8785 signature",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/repro/generate_attestation.py --seed 0
  
  python tools/repro/generate_attestation.py --verify
        """
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for replay guard (default: 0)"
    )
    
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of replay runs (default: 3)"
    )
    
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify existing attestation signature"
    )
    
    args = parser.parse_args()
    
    try:
        repo_root = get_repo_root()
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    
    if args.verify:
        print("=" * 80)
        print("VERIFYING DETERMINISM ATTESTATION")
        print("=" * 80)
        
        success, message = verify_attestation(repo_root)
        print(f"\n{message}")
        
        if success:
            print("\n[PASS] Determinism Attestation: signature verified")
            return 0
        else:
            print("\n[FAIL] Determinism Attestation: signature verification failed")
            return 2
    
    else:
        print("=" * 80)
        print("GENERATING DETERMINISM ATTESTATION")
        print("=" * 80)
        
        attestation = generate_attestation(repo_root, args.seed, args.runs)
        
        # Save current attestation
        attestation_path = repo_root / "artifacts" / "repro" / "determinism_attestation.json"
        attestation_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(attestation_path, "w") as f:
            f.write(rfc8785_canonicalize(attestation))
        
        history_dir = repo_root / "artifacts" / "repro" / "attestation_history"
        history_dir.mkdir(parents=True, exist_ok=True)
        
        history_slug = deterministic_slug("attestation", attestation["signature"])
        history_path = history_dir / f"attestation_{history_slug}.json"
        
        with open(history_path, "w") as f:
            f.write(rfc8785_canonicalize(attestation))
        
        print(f"\nAttestation generated: {attestation_path}")
        print(f"History saved: {history_path}")
        print(f"Status: {attestation['status']}")
        print(f"Determinism score: {attestation['determinism_score']}%")
        print(f"Replay hash: {attestation['replay_hash']}")
        print(f"Signature: {attestation['signature']}")
        print(f"Previous signature: {attestation['prev_signature']}")
        print(f"Proof-or-Abstain: {attestation['proof_or_abstain']}")
        
        if attestation['status'] == "CLEAN" and attestation['replay_success']:
            print("\n[PASS] Determinism Attestation (chained): " + attestation['signature'])
            return 0
        else:
            print("\n[ABSTAIN] Determinism Attestation: violations detected or replay failed")
            return 2


if __name__ == "__main__":
    sys.exit(main())
