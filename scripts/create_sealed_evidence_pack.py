#!/usr/bin/env python3
"""
Sealed Evidence Pack Creator
Collects artifacts and creates a sealed evidence pack for CI.
"""
import os
import sys
import json
import shutil
import subprocess
from datetime import datetime
from typing import Dict, Any, List

def run_command(cmd: List[str]) -> tuple[int, str, str]:
    """Run a command and return (returncode, stdout, stderr)."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 1, "", "Command timed out"
    except Exception as e:
        return 1, "", str(e)

def collect_artifacts() -> Dict[str, Any]:
    """Collect all required artifacts for the sealed evidence pack."""
    artifacts = {
        "timestamp": datetime.now().isoformat() + 'Z',
        "collected_files": [],
        "missing_files": [],
        "generation_results": {}
    }

    required_artifacts = [
        "artifacts/attestation/dual_attestation.json",
        "artifacts/badges/coverage_badge.json",
        "artifacts/wpv5/EVIDENCE.md"
    ]

    optional_directories = [
        "figures/exports",
        "artifacts/figures",
        "artifacts/exports"
    ]

    if not os.path.exists("artifacts/attestation/dual_attestation.json"):
        print("Generating dual attestation...")
        rc, stdout, stderr = run_command(["python", "scripts/gen_dual_attestation.py"])
        artifacts["generation_results"]["dual_attestation"] = {
            "return_code": rc,
            "stdout": stdout.strip(),
            "stderr": stderr.strip()
        }

    if not os.path.exists("artifacts/badges/coverage_badge.json"):
        print("Generating coverage badge...")
        rc, stdout, stderr = run_command(["python", "scripts/gen_coverage_badge.py"])
        artifacts["generation_results"]["coverage_badge"] = {
            "return_code": rc,
            "stdout": stdout.strip(),
            "stderr": stderr.strip()
        }

    if not os.path.exists("artifacts/wpv5/EVIDENCE.md"):
        print("Generating evidence pack...")
        rc, stdout, stderr = run_command(["python", "scripts/mk_evidence_pack.py"])
        artifacts["generation_results"]["evidence_pack"] = {
            "return_code": rc,
            "stdout": stdout.strip(),
            "stderr": stderr.strip()
        }

    for artifact_path in required_artifacts:
        if os.path.exists(artifact_path):
            artifacts["collected_files"].append(artifact_path)
        else:
            artifacts["missing_files"].append(artifact_path)

    for dir_path in optional_directories:
        if os.path.exists(dir_path):
            for root, dirs, files in os.walk(dir_path):
                for file in files:
                    full_path = os.path.join(root, file)
                    artifacts["collected_files"].append(full_path)

    return artifacts

def create_sealed_pack(artifacts: Dict[str, Any]) -> str:
    """Create the sealed evidence pack directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pack_dir = f"artifacts/sealed_evidence_pack_{timestamp}"

    os.makedirs(pack_dir, exist_ok=True)

    for file_path in artifacts["collected_files"]:
        if os.path.exists(file_path):
            rel_path = file_path
            if rel_path.startswith("artifacts/"):
                rel_path = rel_path[10:]

            dest_path = os.path.join(pack_dir, rel_path)
            dest_dir = os.path.dirname(dest_path)
            os.makedirs(dest_dir, exist_ok=True)

            try:
                shutil.copy2(file_path, dest_path)
                print(f"Copied: {file_path} -> {dest_path}")
            except Exception as e:
                print(f"Failed to copy {file_path}: {e}")

    manifest = {
        "sealed_evidence_pack": {
            "created": artifacts["timestamp"],
            "pack_directory": pack_dir,
            "collected_files": artifacts["collected_files"],
            "missing_files": artifacts["missing_files"],
            "generation_results": artifacts["generation_results"],
            "git_commit": get_git_commit(),
            "ci_run_id": os.getenv("GITHUB_RUN_ID", "local"),
            "ci_run_number": os.getenv("GITHUB_RUN_NUMBER", "0")
        }
    }

    manifest_path = os.path.join(pack_dir, "MANIFEST.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=True)

    print(f"Sealed evidence pack created: {pack_dir}")
    print(f"Manifest: {manifest_path}")

    return pack_dir

def get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"

def main():
    """Main function to create sealed evidence pack."""
    print("Creating Sealed Evidence Pack...")

    artifacts = collect_artifacts()

    pack_dir = create_sealed_pack(artifacts)

    print("\n" + "="*50)
    print("SEALED EVIDENCE PACK SUMMARY:")
    print("="*50)
    print(f"Pack Directory: {pack_dir}")
    print(f"Collected Files: {len(artifacts['collected_files'])}")
    print(f"Missing Files: {len(artifacts['missing_files'])}")

    if artifacts['missing_files']:
        print("Missing files:")
        for missing in artifacts['missing_files']:
            print(f"  - {missing}")

    print("="*50)

    if os.getenv("GITHUB_ACTIONS"):
        with open(os.environ.get("GITHUB_OUTPUT", "/dev/null"), "a") as f:
            f.write(f"sealed_pack_dir={pack_dir}\n")
            f.write(f"collected_count={len(artifacts['collected_files'])}\n")
            f.write(f"missing_count={len(artifacts['missing_files'])}\n")

    return 0 if not artifacts['missing_files'] else 1

if __name__ == "__main__":
    sys.exit(main())
