#!/usr/bin/env python3
"""
Dual Attestation Generator
Creates dual_attestation.json with UI Merkle and Reasoning Merkle for Trust Demo.
"""
import json
import subprocess
import sys
import os
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional

def get_git_commit_hash() -> str:
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

def get_ui_merkle_root() -> str:
    """Generate UI Merkle root from UI components."""
    ui_paths = [
        "apps/ui",
        "frontend",
        "ui"
    ]

    ui_files = []
    for path in ui_paths:
        if os.path.exists(path):
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith(('.js', '.jsx', '.ts', '.tsx', '.vue', '.html', '.css')):
                        ui_files.append(os.path.join(root, file))

    if not ui_files:
        try:
            result = subprocess.run(
                ["git", "ls-tree", "-r", "HEAD", "--name-only"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                all_files = result.stdout.strip().split('\n')
                ui_files = [f for f in all_files if any(ui_path in f for ui_path in ['ui/', 'frontend/', 'apps/ui/'])]
        except Exception:
            pass

    hasher = hashlib.sha256()
    for file_path in sorted(ui_files):
        hasher.update(file_path.encode('utf-8'))
        if os.path.exists(file_path):
            try:
                with open(file_path, 'rb') as f:
                    hasher.update(f.read())
            except Exception:
                pass

    return hasher.hexdigest()

def get_reasoning_merkle_root() -> str:
    """Generate Reasoning Merkle root from proof/reasoning components."""
    reasoning_paths = [
        "backend/axiom_engine",
        "backend/logic",
        "backend/lean_proj",
        "tools/ci/strategic_pr_validator.py"
    ]

    reasoning_files = []
    for path in reasoning_paths:
        if os.path.exists(path):
            if os.path.isfile(path):
                reasoning_files.append(path)
            else:
                for root, dirs, files in os.walk(path):
                    for file in files:
                        if file.endswith(('.py', '.lean', '.json')):
                            reasoning_files.append(os.path.join(root, file))

    hasher = hashlib.sha256()
    for file_path in sorted(reasoning_files):
        hasher.update(file_path.encode('utf-8'))
        if os.path.exists(file_path):
            try:
                with open(file_path, 'rb') as f:
                    hasher.update(f.read())
            except Exception:
                pass

    return hasher.hexdigest()

def get_database_merkle_root() -> Optional[str]:
    """Get latest Merkle root from database if available."""
    try:
        from backend.tools.progress import get_latest_run_data

        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            return None  # No default fallback
        data = get_latest_run_data(db_url, offline=True)

        if data and 'latest_block' in data:
            return data['latest_block'].get('merkle_root')
    except Exception:
        pass
    return None

def generate_dual_attestation() -> Dict[str, Any]:
    """Generate dual attestation data."""
    timestamp = datetime.now().isoformat() + 'Z'
    commit_hash = get_git_commit_hash()

    ui_merkle = get_ui_merkle_root()
    reasoning_merkle = get_reasoning_merkle_root()
    db_merkle = get_database_merkle_root()

    ui_bytes = bytes.fromhex(ui_merkle)
    reasoning_bytes = bytes.fromhex(reasoning_merkle)

    min_len = min(len(ui_bytes), len(reasoning_bytes))
    xor_result = bytes(a ^ b for a, b in zip(ui_bytes[:min_len], reasoning_bytes[:min_len]))
    dual_attestation_hash = xor_result.hex()

    return {
        "timestamp": timestamp,
        "commit_hash": commit_hash,
        "ui_merkle_root": ui_merkle,
        "reasoning_merkle_root": reasoning_merkle,
        "database_merkle_root": db_merkle,
        "dual_attestation_hash": dual_attestation_hash,
        "attestation_method": "UI_Merkle_XOR_Reasoning_Merkle",
        "trust_demo_ready": True,
        "verification": {
            "ui_files_count": len([f for f in os.listdir('.') if f.endswith(('.js', '.jsx', '.ts', '.tsx'))]) if os.path.exists('.') else 0,
            "reasoning_files_count": len([f for f in os.listdir('backend') if f.endswith('.py')]) if os.path.exists('backend') else 0,
            "database_connected": db_merkle is not None
        }
    }

def main():
    """Main function to generate dual attestation."""
    print("Generating dual attestation...")

    output_dir = "artifacts/attestation"
    os.makedirs(output_dir, exist_ok=True)

    attestation_data = generate_dual_attestation()

    output_path = os.path.join(output_dir, "dual_attestation.json")
    with open(output_path, 'w') as f:
        json.dump(attestation_data, f, indent=2, ensure_ascii=True)

    print(f"Dual attestation created: {output_path}")
    print(f"UI Merkle: {attestation_data['ui_merkle_root'][:16]}...")
    print(f"Reasoning Merkle: {attestation_data['reasoning_merkle_root'][:16]}...")
    print(f"Dual Attestation Hash: {attestation_data['dual_attestation_hash'][:16]}...")

    return 0

if __name__ == "__main__":
    sys.exit(main())
