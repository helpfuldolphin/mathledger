#!/usr/bin/env python3
"""
CI Velocity Measurement - Measure and seal CI performance metrics

Measures CI velocity by running key operations and recording timing data.
Produces RFC 8785 canonical JSON artifact with sealed hash.

Usage:
    python ci_velocity.py --measure    # Measure CI velocity
    python ci_velocity.py --verify     # Verify sealed artifact
"""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

REPO_ROOT = Path(__file__).parent.parent.parent
PERF_DIR = REPO_ROOT / 'artifacts' / 'perf'
PERF_LOG = PERF_DIR / 'perf_log.json'

def canonical_json(obj: Any) -> str:
    """RFC 8785 canonical JSON serialization"""
    return json.dumps(
        obj,
        ensure_ascii=True,
        sort_keys=True,
        separators=(',', ':'),
        indent=None
    )

def compute_hash(data: str) -> str:
    """Compute SHA256 hash"""
    return hashlib.sha256(data.encode('utf-8')).hexdigest()

def measure_operation(name: str, cmd: str) -> Dict:
    """Measure single operation timing"""
    start = time.time()
    result = subprocess.run(
        cmd,
        shell=True,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True
    )
    elapsed = time.time() - start
    
    return {
        'name': name,
        'command': cmd,
        'elapsed_seconds': round(elapsed, 3),
        'return_code': result.returncode,
        'success': result.returncode == 0
    }

def measure_ci_velocity() -> Dict:
    """Measure CI velocity across key operations"""
    print("Measuring CI velocity...")
    print()
    
    operations = [
        ('ascii_check', 'python tools/check_ascii.py'),
        ('unit_tests', 'NO_NETWORK=true python -m unittest discover -s tests -p "test_*.py" -q'),
        ('import_check', 'python -c "import backend.axiom_engine.derive"'),
    ]
    
    measurements = []
    total_time = 0
    
    for name, cmd in operations:
        print(f"Measuring: {name}...")
        measurement = measure_operation(name, cmd)
        measurements.append(measurement)
        total_time += measurement['elapsed_seconds']
        
        status = 'PASS' if measurement['success'] else 'FAIL'
        print(f"  {status}: {measurement['elapsed_seconds']}s")
    
    print()
    
    velocity_data = {
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'total_elapsed_seconds': round(total_time, 3),
        'measurements': measurements,
        'success_count': sum(1 for m in measurements if m['success']),
        'total_count': len(measurements)
    }
    
    return velocity_data

def seal_artifact(data: Dict) -> str:
    """Seal artifact with RFC 8785 canonical hash"""
    canonical = canonical_json(data)
    return compute_hash(canonical)

def save_perf_log(data: Dict, signature: str) -> None:
    """Save performance log with signature"""
    PERF_DIR.mkdir(parents=True, exist_ok=True)
    
    sealed_data = dict(data)
    sealed_data['signature'] = signature
    
    with open(PERF_LOG, 'w') as f:
        json.dump(sealed_data, f, indent=2)
    
    canonical_file = PERF_DIR / 'perf_log_canonical.json'
    with open(canonical_file, 'w') as f:
        f.write(canonical_json(data))

def verify_perf_log() -> bool:
    """Verify sealed performance log"""
    if not PERF_LOG.exists():
        print("No performance log found")
        return False
    
    with open(PERF_LOG) as f:
        data = json.load(f)
    
    stored_signature = data.pop('signature', None)
    if not stored_signature:
        print("No signature found in performance log")
        return False
    
    canonical = canonical_json(data)
    computed_signature = compute_hash(canonical)
    
    if stored_signature == computed_signature:
        print(f"[PASS] CI Velocity: {data['total_elapsed_seconds']}s")
        print(f"  Signature: {stored_signature}")
        print(f"  Timestamp: {data['timestamp']}")
        print(f"  Success: {data['success_count']}/{data['total_count']}")
        return True
    else:
        print("[FAIL] Signature mismatch")
        print(f"  Stored: {stored_signature}")
        print(f"  Computed: {computed_signature}")
        return False

def main():
    parser = argparse.ArgumentParser(description='CI velocity measurement')
    parser.add_argument('--measure', action='store_true',
                       help='Measure CI velocity')
    parser.add_argument('--verify', action='store_true',
                       help='Verify sealed artifact')
    
    args = parser.parse_args()
    
    if args.measure:
        velocity_data = measure_ci_velocity()
        signature = seal_artifact(velocity_data)
        save_perf_log(velocity_data, signature)
        
        print(f"[PASS] CI Velocity: {velocity_data['total_elapsed_seconds']}s")
        print(f"  Artifact: {PERF_LOG}")
        print(f"  Signature: {signature}")
        print(f"  Success: {velocity_data['success_count']}/{velocity_data['total_count']}")
        return 0
    
    elif args.verify:
        success = verify_perf_log()
        return 0 if success else 1
    
    else:
        parser.print_help()
        return 1

if __name__ == '__main__':
    sys.exit(main())
