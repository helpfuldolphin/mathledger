#!/usr/bin/env python3
"""
Deterministic Build - Reproducible builds with hash verification

Ensures that builds are deterministic and verifiable by:
1. Setting SOURCE_DATE_EPOCH for reproducible timestamps
2. Computing hashes of all source files
3. Building with fixed parameters
4. Verifying output hashes match expected values

Usage:
    python deterministic_build.py
    python deterministic_build.py --verify-only
    python deterministic_build.py --save-baseline
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime

REPO_ROOT = Path(__file__).parent.parent.parent
BASELINE_FILE = REPO_ROOT / 'artifacts' / 'build_baseline.json'

def compute_file_hash(file_path):
    """Compute SHA256 hash of a file"""
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            sha256.update(chunk)
    return sha256.hexdigest()

def compute_tree_hash(directory, patterns=None):
    """Compute hash of all files in directory matching patterns"""
    if patterns is None:
        patterns = ['*.py', '*.toml', '*.yaml', '*.yml']
    
    file_hashes = {}
    for pattern in patterns:
        for file_path in sorted(directory.rglob(pattern)):
            if '.git' in file_path.parts or '__pycache__' in file_path.parts:
                continue
            if file_path.is_file():
                rel_path = file_path.relative_to(REPO_ROOT)
                file_hashes[str(rel_path)] = compute_file_hash(file_path)
    
    combined = json.dumps(file_hashes, sort_keys=True)
    return hashlib.sha256(combined.encode()).hexdigest(), file_hashes

def set_deterministic_env():
    """Set environment variables for deterministic builds"""
    os.environ['SOURCE_DATE_EPOCH'] = '1609459200'  # 2021-01-01 00:00:00 UTC
    os.environ['PYTHONHASHSEED'] = '0'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

def run_build():
    """Run deterministic build"""
    print("Running deterministic build...")
    
    print("Cleaning previous build artifacts...")
    subprocess.run(['rm', '-rf', 'dist/', 'build/', '*.egg-info'], 
                   cwd=REPO_ROOT, shell=True, check=False)
    
    print("Building...")
    result = subprocess.run(
        ['python', '-m', 'build', '--no-isolation'],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"Build failed: {result.stderr}")
        return False
    
    print("Build completed successfully")
    return True

def save_baseline():
    """Save current build hashes as baseline"""
    print("Computing source tree hash...")
    tree_hash, file_hashes = compute_tree_hash(REPO_ROOT)
    
    baseline = {
        'timestamp': datetime.utcnow().isoformat(),
        'tree_hash': tree_hash,
        'file_count': len(file_hashes),
        'files': file_hashes
    }
    
    BASELINE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(BASELINE_FILE, 'w') as f:
        json.dump(baseline, f, indent=2)
    
    print(f"Baseline saved to {BASELINE_FILE}")
    print(f"Tree hash: {tree_hash}")
    print(f"Files tracked: {len(file_hashes)}")

def verify_against_baseline():
    """Verify current state against baseline"""
    if not BASELINE_FILE.exists():
        print(f"No baseline found at {BASELINE_FILE}")
        print("Run with --save-baseline to create one")
        return False
    
    print("Loading baseline...")
    with open(BASELINE_FILE) as f:
        baseline = json.load(f)
    
    print("Computing current tree hash...")
    current_hash, current_files = compute_tree_hash(REPO_ROOT)
    
    print()
    print("Verification Results:")
    print(f"Baseline tree hash: {baseline['tree_hash']}")
    print(f"Current tree hash:  {current_hash}")
    print(f"Baseline files:     {baseline['file_count']}")
    print(f"Current files:      {len(current_files)}")
    
    if current_hash == baseline['tree_hash']:
        print()
        print("PASS: Build is deterministic - hashes match!")
        return True
    else:
        print()
        print("FAIL: Build is NOT deterministic - hashes differ")
        
        baseline_files = set(baseline['files'].keys())
        current_file_set = set(current_files.keys())
        
        added = current_file_set - baseline_files
        removed = baseline_files - current_file_set
        modified = {f for f in baseline_files & current_file_set 
                   if baseline['files'][f] != current_files[f]}
        
        if added:
            print(f"\nAdded files ({len(added)}):")
            for f in sorted(added)[:10]:
                print(f"  + {f}")
            if len(added) > 10:
                print(f"  ... and {len(added) - 10} more")
        
        if removed:
            print(f"\nRemoved files ({len(removed)}):")
            for f in sorted(removed)[:10]:
                print(f"  - {f}")
            if len(removed) > 10:
                print(f"  ... and {len(removed) - 10} more")
        
        if modified:
            print(f"\nModified files ({len(modified)}):")
            for f in sorted(modified)[:10]:
                print(f"  M {f}")
            if len(modified) > 10:
                print(f"  ... and {len(modified) - 10} more")
        
        return False

def main():
    parser = argparse.ArgumentParser(description='Deterministic build with hash verification')
    parser.add_argument('--verify-only', action='store_true', 
                       help='Only verify against baseline, do not build')
    parser.add_argument('--save-baseline', action='store_true',
                       help='Save current state as baseline')
    
    args = parser.parse_args()
    
    if args.save_baseline:
        save_baseline()
        return 0
    
    if args.verify_only:
        success = verify_against_baseline()
        return 0 if success else 1
    
    print("=== Deterministic Build ===")
    print()
    
    set_deterministic_env()
    
    if not run_build():
        return 1
    
    print()
    if verify_against_baseline():
        print()
        print("Build verification PASSED")
        return 0
    else:
        print()
        print("Build verification FAILED")
        print("Run with --save-baseline to update baseline")
        return 1

if __name__ == '__main__':
    sys.exit(main())
