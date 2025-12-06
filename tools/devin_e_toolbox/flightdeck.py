#!/usr/bin/env python3
"""
Flight Deck - Consolidated operator flow execution

Runs all operator flows in sequence:
1. velocity seal - CI velocity measurement
2. audit chain - Audit trail sync+verify
3. allblue freeze - Fleet state archival

Produces consolidated RFC 8785 sealed report.

Usage:
    python flightdeck.py --run              # Execute flight deck
    python flightdeck.py --run --dry-run    # Dry-run mode (no network)
    python flightdeck.py --verify           # Verify sealed report
    python flightdeck.py --preflight        # Run preflight checks only
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from backend.repro.determinism import deterministic_isoformat, deterministic_timestamp_from_content

REPO_ROOT = Path(__file__).parent.parent.parent
OPS_DIR = REPO_ROOT / 'artifacts' / 'ops'
FLIGHTDECK_REPORT = OPS_DIR / 'flightdeck.json'
FLIGHTDECK_PLAN = OPS_DIR / 'flightdeck_plan.json'
FLIGHTDECK_BUNDLE = OPS_DIR / 'flightdeck_bundle.zip'
FLIGHTDECK_BUNDLE_SIG = OPS_DIR / 'flightdeck_bundle.sig'

PREFLIGHT_CACHE: Dict[str, Tuple[bool, List[Dict]]] = {}


def check_python_version() -> Dict:
    """Check Python version >= 3.9"""
    major, minor = sys.version_info.major, sys.version_info.minor
    version_str = f"{major}.{minor}.{sys.version_info.micro}"
    required = (3, 9)
    
    if (major, minor) >= required:
        return {
            'check': 'Python >= 3.9',
            'status': 'PASS',
            'details': version_str,
            'severity': 'ERROR'
        }
    else:
        return {
            'check': 'Python >= 3.9',
            'status': 'FAIL',
            'details': version_str,
            'severity': 'ERROR',
            'remediation': f'Upgrade Python to >= 3.9 (current: {version_str})'
        }

def check_node_version() -> Dict:
    """Check Node version >= 16 (optional)"""
    try:
        result = subprocess.run(
            ['node', '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            version_str = result.stdout.strip()
            match = re.search(r'v?(\d+)\.', version_str)
            if match:
                major = int(match.group(1))
                if major >= 16:
                    return {
                        'check': 'Node >= 16',
                        'status': 'PASS',
                        'details': version_str,
                        'severity': 'WARN'
                    }
                else:
                    return {
                        'check': 'Node >= 16',
                        'status': 'WARN',
                        'details': version_str,
                        'severity': 'WARN',
                        'remediation': f'Upgrade Node to >= 16 (current: {version_str})'
                    }
        return {
            'check': 'Node >= 16',
            'status': 'WARN',
            'details': 'Not found',
            'severity': 'WARN',
            'remediation': 'Install Node.js >= 16 (optional for MathLedger)'
        }
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {
            'check': 'Node >= 16',
            'status': 'WARN',
            'details': 'Not found',
            'severity': 'WARN',
            'remediation': 'Install Node.js >= 16 (optional for MathLedger)'
        }

def check_port_availability(port: int, service: str) -> Dict:
    """Check if port is available (optional service)"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('localhost', port))
        sock.close()
        
        if result == 0:
            return {
                'check': f'{service}:{port}',
                'status': 'PASS',
                'details': 'Connected',
                'severity': 'WARN'
            }
        else:
            return {
                'check': f'{service}:{port}',
                'status': 'WARN',
                'details': 'Connection refused',
                'severity': 'WARN',
                'remediation': f'Start {service}: docker-compose up -d {service.lower()}'
            }
    except Exception as e:
        return {
            'check': f'{service}:{port}',
            'status': 'WARN',
            'details': str(e),
            'severity': 'WARN',
            'remediation': f'Start {service}: docker-compose up -d {service.lower()}'
        }

def check_write_permissions() -> Dict:
    """Check write permissions to artifacts directories"""
    dirs = [
        REPO_ROOT / 'artifacts' / 'ops',
        REPO_ROOT / 'artifacts' / 'perf',
        REPO_ROOT / 'artifacts' / 'audit',
        REPO_ROOT / 'artifacts' / 'allblue'
    ]
    
    failed_dirs = []
    for d in dirs:
        try:
            d.mkdir(parents=True, exist_ok=True)
            test_file = d / '.write_test'
            test_file.touch()
            test_file.unlink()
        except (PermissionError, OSError) as e:
            failed_dirs.append(str(d.relative_to(REPO_ROOT)))
    
    if not failed_dirs:
        return {
            'check': 'artifacts/ write',
            'status': 'PASS',
            'details': 'All directories writable',
            'severity': 'ERROR'
        }
    else:
        return {
            'check': 'artifacts/ write',
            'status': 'FAIL',
            'details': f'Cannot write: {", ".join(failed_dirs)}',
            'severity': 'ERROR',
            'remediation': f'Fix permissions: chmod -R u+w {" ".join(failed_dirs)}'
        }

def check_disk_space() -> Dict:
    """Check available disk space"""
    stat = shutil.disk_usage(REPO_ROOT)
    free_gb = stat.free / (1024**3)
    
    if free_gb >= 1.0:
        return {
            'check': 'Disk space',
            'status': 'PASS',
            'details': f'{free_gb:.1f} GB free',
            'severity': 'WARN'
        }
    else:
        return {
            'check': 'Disk space',
            'status': 'WARN',
            'details': f'{free_gb:.1f} GB free',
            'severity': 'WARN',
            'remediation': 'Free up disk space (< 1GB available)'
        }

def run_preflight_checks(use_cache: bool = True) -> Tuple[bool, List[Dict]]:
    """Run all preflight checks with optional caching"""
    cache_key = 'preflight_checks'
    
    if use_cache and cache_key in PREFLIGHT_CACHE:
        return PREFLIGHT_CACHE[cache_key]
    
    checks = [
        check_python_version(),
        check_node_version(),
        check_port_availability(5432, 'PostgreSQL'),
        check_port_availability(6379, 'Redis'),
        check_write_permissions(),
        check_disk_space()
    ]
    
    # Only ERROR severity failures block execution
    error_failures = [c for c in checks if c['severity'] == 'ERROR' and c['status'] == 'FAIL']
    all_pass = len(error_failures) == 0
    
    result = (all_pass, checks)
    PREFLIGHT_CACHE[cache_key] = result
    
    return result

def print_preflight_table(checks: List[Dict]) -> None:
    """Print preflight checks as ASCII table"""
    print("=== PREFLIGHT CHECKS ===")
    print()
    print("| Check             | Status | Details                          |")
    print("|-------------------|--------|----------------------------------|")
    
    for check in checks:
        check_name = check['check'].ljust(17)
        status = check['status'].ljust(6)
        details = check['details'][:32].ljust(32)
        print(f"| {check_name} | {status} | {details} |")
    
    print()


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

def run_command(cmd: str, dry_run: bool = False) -> Dict:
    """Run command and capture output"""
    env = os.environ.copy()
    if dry_run:
        env['NO_NETWORK'] = 'true'
    
    result = subprocess.run(
        cmd,
        shell=True,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=env
    )
    return {
        'command': cmd,
        'return_code': result.returncode,
        'stdout': result.stdout,
        'stderr': result.stderr,
        'success': result.returncode == 0
    }

def extract_hash(text: str, pattern: str) -> str:
    """Extract hash from command output"""
    import re
    match = re.search(pattern, text)
    return match.group(1) if match else 'unknown'

def run_velocity_seal(dry_run: bool = False) -> Dict:
    """Run velocity seal and extract results"""
    print("Running velocity seal...")
    result = run_command("python tools/devin_e_toolbox/ci_velocity.py --measure", dry_run=dry_run)
    
    perf_log = REPO_ROOT / 'artifacts' / 'perf' / 'perf_log.json'
    signature = 'unknown'
    total_time = 0
    
    if perf_log.exists():
        with open(perf_log) as f:
            data = json.load(f)
            signature = data.get('signature', 'unknown')
            total_time = data.get('total_elapsed_seconds', 0)
    
    return {
        'operation': 'velocity_seal',
        'success': result['success'],
        'signature': signature,
        'total_elapsed_seconds': total_time,
        'artifact': str(perf_log.relative_to(REPO_ROOT))
    }

def run_audit_chain(dry_run: bool = False) -> Dict:
    """Run audit chain and extract results"""
    print("Running audit chain...")
    
    sync_result = run_command("python tools/devin_e_toolbox/audit_sync.py --collect", dry_run=dry_run)
    
    verify_result = run_command("python tools/devin_e_toolbox/audit_sync.py --verify", dry_run=dry_run)
    
    chain_hash = extract_hash(verify_result['stdout'], r'Latest hash: ([a-f0-9]{64})')
    
    audit_log = REPO_ROOT / 'artifacts' / 'audit' / 'audit_trail.jsonl'
    
    return {
        'operation': 'audit_chain',
        'success': sync_result['success'] and verify_result['success'],
        'chain_hash': chain_hash,
        'artifact': str(audit_log.relative_to(REPO_ROOT))
    }

def run_allblue_freeze(dry_run: bool = False) -> Dict:
    """Run allblue freeze and extract results"""
    print("Running allblue freeze...")
    result = run_command("python tools/devin_e_toolbox/allblue_archive.py --archive --force", dry_run=dry_run)
    
    signature = extract_hash(result['stdout'], r'Signature: ([a-f0-9]{64})')
    
    state_file = REPO_ROOT / 'artifacts' / 'allblue' / 'fleet_state.json'
    
    return {
        'operation': 'allblue_freeze',
        'success': result['success'],
        'signature': signature,
        'artifact': str(state_file.relative_to(REPO_ROOT))
    }

def run_flightdeck(dry_run: bool = False, parallel: bool = False) -> Dict:
    """Execute full flight deck sequence"""
    mode_parts = []
    if dry_run:
        mode_parts.append("DRY-RUN")
    if parallel:
        mode_parts.append("PARALLEL")
    mode_str = f" ({', '.join(mode_parts)})" if mode_parts else ""
    print(f"=== FLIGHT DECK{mode_str} ===")
    print()
    
    if parallel:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_velocity = executor.submit(run_velocity_seal, dry_run)
            future_audit = executor.submit(run_audit_chain, dry_run)
            future_allblue = executor.submit(run_allblue_freeze, dry_run)
            
            velocity = future_velocity.result()
            audit = future_audit.result()
            allblue = future_allblue.result()
    else:
        velocity = run_velocity_seal(dry_run=dry_run)
        audit = run_audit_chain(dry_run=dry_run)
        allblue = run_allblue_freeze(dry_run=dry_run)
    
    operations = [velocity, audit, allblue]
    operations_fingerprint = canonical_json(operations)
    timestamp = deterministic_isoformat(
        'flightdeck',
        'parallel' if parallel else 'serial',
        'dry_run' if dry_run else 'normal',
        operations_fingerprint
    ).replace('+00:00', 'Z')
    
    print(f"  velocity seal: {'PASS' if velocity['success'] else 'FAIL'}")
    print(f"  audit chain: {'PASS' if audit['success'] else 'FAIL'}")
    print(f"  allblue freeze: {'PASS' if allblue['success'] else 'FAIL'}")
    print()
    
    report = {
        'timestamp': timestamp,
        'mode': 'dry-run' if dry_run else 'normal',
        'parallel': parallel,
        'operations': operations,
        'success_count': sum(1 for op in operations if op['success']),
        'total_count': len(operations),
        'all_pass': all(op['success'] for op in operations)
    }
    
    return report

def seal_report(report: Dict) -> str:
    """Seal report with RFC 8785 canonical hash"""
    canonical = canonical_json(report)
    return compute_hash(canonical)

def save_flightdeck_report(report: Dict, signature: str) -> None:
    """Save flight deck report with signature"""
    OPS_DIR.mkdir(parents=True, exist_ok=True)
    
    sealed_report = dict(report)
    sealed_report['signature'] = signature
    
    with open(FLIGHTDECK_REPORT, 'w', encoding='ascii') as f:
        f.write(canonical_json(sealed_report))
    
    canonical_file = OPS_DIR / 'flightdeck_canonical.json'
    with open(canonical_file, 'w') as f:
        f.write(canonical_json(report))

def verify_flightdeck_report() -> bool:
    """Verify sealed flight deck report"""
    if not FLIGHTDECK_REPORT.exists():
        print("No flight deck report found")
        return False
    
    with open(FLIGHTDECK_REPORT) as f:
        data = json.load(f)
    
    stored_signature = data.pop('signature', None)
    if not stored_signature:
        print("No signature found in report")
        return False
    
    canonical = canonical_json(data)
    computed_signature = compute_hash(canonical)
    
    if stored_signature == computed_signature:
        print(f"[PASS] Flight Deck: {stored_signature}")
        print(f"  Timestamp: {data['timestamp']}")
        print(f"  Success: {data['success_count']}/{data['total_count']}")
        print(f"  All Pass: {data['all_pass']}")
        return True
    else:
        print("[FAIL] Signature mismatch")
        print(f"  Stored: {stored_signature}")
        print(f"  Computed: {computed_signature}")
        return False


def generate_verify_txt(report: Dict, signature: str) -> str:
    """Generate VERIFY.txt content for evidence bundle"""
    timestamp = report.get('timestamp', 'unknown')
    success = f"{report.get('success_count', 0)}/{report.get('total_count', 0)}"
    
    return f"""Flight Deck Evidence Bundle Verification
=========================================

This bundle contains RFC 8785-sealed artifacts from a Flight Deck run.

Timestamp: {timestamp}
Signature: {signature}
Success: {success}

Verification Commands:

1. Extract bundle:
   unzip flightdeck_bundle.zip -d /tmp/verify

2. Verify Flight Deck report:
   python tools/devin_e_toolbox/flightdeck.py --verify

3. Verify signature manually:
   python -c "import json, hashlib; data=json.load(open('flightdeck.json')); sig=data.pop('signature'); canonical=json.dumps(data, ensure_ascii=True, sort_keys=True, separators=(',',':')); print('Expected:', sig); print('Computed:', hashlib.sha256(canonical.encode()).hexdigest())"

4. Verify audit chain:
   python tools/devin_e_toolbox/audit_sync.py --verify

5. Verify Merkle roots (if database available):
   python tools/devin_e_toolbox/artifact_verifier.py --verify-merkle

Expected Signature: {signature}

All artifacts use RFC 8785 canonical JSON for deterministic hashing.

Operations Executed:
- velocity seal: CI velocity measurement
- audit chain: Audit trail sync+verify
- allblue freeze: Fleet state archival

For more information, see docs/workflows/WEEKLY_PROOF_OF_BUILD.md
"""

def sign_bundle(bundle_path: Path) -> Optional[str]:
    """Create detached signature for bundle using SHA256"""
    if not bundle_path.exists():
        return None
    
    with open(bundle_path, 'rb') as f:
        bundle_data = f.read()
        bundle_hash = hashlib.sha256(bundle_data).hexdigest()
    
    sig_timestamp = deterministic_isoformat('flightdeck_bundle_sig', bundle_hash).replace('+00:00', 'Z')
    sig_content = f"""Flight Deck Bundle Signature
=============================

Bundle: {bundle_path.name}
SHA256: {bundle_hash}
Timestamp: {sig_timestamp}

This is a detached signature file for the Flight Deck evidence bundle.
Verify with: sha256sum {bundle_path.name}
"""
    
    with open(FLIGHTDECK_BUNDLE_SIG, 'w') as f:
        f.write(sig_content)
    
    return bundle_hash

def verify_bundle_signature() -> bool:
    """Verify bundle signature"""
    if not FLIGHTDECK_BUNDLE.exists() or not FLIGHTDECK_BUNDLE_SIG.exists():
        print("Bundle or signature file not found")
        return False
    
    with open(FLIGHTDECK_BUNDLE, 'rb') as f:
        bundle_hash = hashlib.sha256(f.read()).hexdigest()
    
    with open(FLIGHTDECK_BUNDLE_SIG) as f:
        sig_content = f.read()
        match = re.search(r'SHA256: ([a-f0-9]{64})', sig_content)
        if not match:
            print("No SHA256 found in signature file")
            return False
        stored_hash = match.group(1)
    
    if bundle_hash == stored_hash:
        print(f"[PASS] Bundle Signature: verified")
        print(f"  SHA256: {bundle_hash}")
        return True
    else:
        print("[FAIL] Bundle signature mismatch")
        print(f"  Stored: {stored_hash}")
        print(f"  Computed: {bundle_hash}")
        return False

def create_evidence_bundle(report: Dict, signature: str, sign: bool = False) -> str:
    """Create deterministic evidence bundle ZIP"""
    OPS_DIR.mkdir(parents=True, exist_ok=True)
    
    artifacts = [
        ('flightdeck.json', FLIGHTDECK_REPORT),
        ('perf_log.json', REPO_ROOT / 'artifacts' / 'perf' / 'perf_log.json'),
        ('audit_trail.jsonl', REPO_ROOT / 'artifacts' / 'audit' / 'audit_trail.jsonl'),
        ('fleet_state.json', REPO_ROOT / 'artifacts' / 'allblue' / 'fleet_state.json')
    ]
    
    verify_content = generate_verify_txt(report, signature)
    if sign:
        verify_content += f"""
Bundle Signing:

To verify the bundle signature:
   python tools/devin_e_toolbox/flightdeck.py --verify-signature

Or manually:
   sha256sum {FLIGHTDECK_BUNDLE.name}
   cat {FLIGHTDECK_BUNDLE_SIG.name}
"""
    
    def _zip_timestamp_tuple(seed: Any) -> tuple[int, int, int, int, int, int]:
        ts = deterministic_timestamp_from_content(seed, signature or "unsigned")
        return (ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second)

    with zipfile.ZipFile(FLIGHTDECK_BUNDLE, 'w', zipfile.ZIP_DEFLATED) as zf:
        for arcname, filepath in artifacts:
            if filepath.exists():
                zinfo = zipfile.ZipInfo(arcname)
                zinfo.date_time = _zip_timestamp_tuple(("artifact", arcname))
                zinfo.compress_type = zipfile.ZIP_DEFLATED
                with open(filepath, 'rb') as f:
                    zf.writestr(zinfo, f.read())
        
        zinfo = zipfile.ZipInfo('VERIFY.txt')
        zinfo.date_time = _zip_timestamp_tuple(("verify", signature))
        zinfo.compress_type = zipfile.ZIP_DEFLATED
        zf.writestr(zinfo, verify_content.encode('utf-8'))
    
    bundle_contents = []
    with zipfile.ZipFile(FLIGHTDECK_BUNDLE, 'r') as zf:
        for name in sorted(zf.namelist()):
            bundle_contents.append(name)
            bundle_contents.append(zf.read(name).decode('utf-8', errors='replace'))
    
    bundle_data = '\n'.join(bundle_contents)
    bundle_hash = compute_hash(bundle_data)
    
    if sign:
        sign_bundle(FLIGHTDECK_BUNDLE)
    
    return bundle_hash

def main():
    parser = argparse.ArgumentParser(description='Flight deck operator flows')
    parser.add_argument('--run', action='store_true',
                       help='Execute flight deck')
    parser.add_argument('--dry-run', action='store_true',
                       help='Dry-run mode with NO_NETWORK')
    parser.add_argument('--preflight', action='store_true',
                       help='Run preflight checks only')
    parser.add_argument('--force-preflight', action='store_true',
                       help='Force preflight checks (bypass cache)')
    parser.add_argument('--parallel', action='store_true',
                       help='Run operations in parallel')
    parser.add_argument('--sign', action='store_true',
                       help='Sign evidence bundle')
    parser.add_argument('--verify', action='store_true',
                       help='Verify sealed report')
    parser.add_argument('--verify-signature', action='store_true',
                       help='Verify bundle signature')
    
    args = parser.parse_args()
    
    if args.verify_signature:
        success = verify_bundle_signature()
        return 0 if success else 1
    
    if args.preflight:
        use_cache = not args.force_preflight
        all_pass, checks = run_preflight_checks(use_cache=use_cache)
        print_preflight_table(checks)
        
        if not all_pass:
            error_failures = [c for c in checks if c['severity'] == 'ERROR' and c['status'] == 'FAIL']
            failed_deps = ', '.join([c['check'] for c in error_failures])
            print(f"ABSTAIN: Preflight check failed (dep={failed_deps})")
            print()
            print("Remediation:")
            for check in error_failures:
                if 'remediation' in check:
                    print(f"  - {check['remediation']}")
            return 2
        else:
            print("[PASS] Preflight checks")
            return 0
    
    elif args.run:
        use_cache = not args.force_preflight
        all_pass, checks = run_preflight_checks(use_cache=use_cache)
        print_preflight_table(checks)
        
        if not all_pass:
            error_failures = [c for c in checks if c['severity'] == 'ERROR' and c['status'] == 'FAIL']
            failed_deps = ', '.join([c['check'] for c in error_failures])
            print(f"ABSTAIN: Preflight check failed (dep={failed_deps})")
            print()
            print("Remediation:")
            for check in error_failures:
                if 'remediation' in check:
                    print(f"  - {check['remediation']}")
            return 2
        
        # Run flight deck
        dry_run = args.dry_run
        parallel = args.parallel
        report = run_flightdeck(dry_run=dry_run, parallel=parallel)
        signature = seal_report(report)
        
        if dry_run:
            OPS_DIR.mkdir(parents=True, exist_ok=True)
            plan_report = dict(report)
            plan_report['signature'] = signature
            with open(FLIGHTDECK_PLAN, 'w') as f:
                json.dump(plan_report, f, indent=2)
            
            status = 'PASS' if report['all_pass'] else 'FAIL'
            print(f"[{status}] Flight Deck (DRY-RUN): {signature}")
            print(f"  Plan: {FLIGHTDECK_PLAN}")
            print(f"  Success: {report['success_count']}/{report['total_count']}")
        else:
            save_flightdeck_report(report, signature)
            
            status = 'PASS' if report['all_pass'] else 'FAIL'
            print(f"[{status}] Flight Deck: {signature}")
            print(f"  Report: {FLIGHTDECK_REPORT}")
            print(f"  Success: {report['success_count']}/{report['total_count']}")
            
            bundle_hash = create_evidence_bundle(report, signature, sign=args.sign)
            print()
            print(f"[PASS] Evidence Bundle: {bundle_hash}")
            print(f"  Bundle: {FLIGHTDECK_BUNDLE}")
            print(f"  Size: {FLIGHTDECK_BUNDLE.stat().st_size} bytes")
            print(f"  Artifacts: 4 sealed files + VERIFY.txt")
            
            if args.sign:
                print()
                print(f"[PASS] Bundle Signature: created")
                print(f"  Signature: {FLIGHTDECK_BUNDLE_SIG}")
                print(f"  Verify: python tools/devin_e_toolbox/flightdeck.py --verify-signature")
        
        return 0 if report['all_pass'] else 1
    
    elif args.verify:
        success = verify_flightdeck_report()
        return 0 if success else 1
    
    else:
        parser.print_help()
        return 1

if __name__ == '__main__':
    sys.exit(main())
