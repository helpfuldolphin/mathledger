#!/usr/bin/env python3
"""
Network Sandbox Mode Script

Enables network-free testing and CI execution for MathLedger.
Provides environment setup, validation, and execution wrapper.

Usage:
    python scripts/network-sandbox.py pytest tests/
    
    python scripts/network-sandbox.py python backend/axiom_engine/derive.py --smoke-pl
    
    python scripts/network-sandbox.py --validate
    
    python scripts/network-sandbox.py --simulate
"""

import os
import sys
import subprocess
import argparse
from datetime import datetime
from pathlib import Path


# ============================================================================
# ============================================================================

REPO_ROOT = Path(__file__).parent.parent
ARTIFACTS_DIR = REPO_ROOT / 'artifacts' / 'no_network'
SIMULATION_LOG = ARTIFACTS_DIR / 'simulation.log'

NO_NETWORK_ENV = {
    'NO_NETWORK': 'true',
    'DISABLE_DB_STARTUP': '1',
    'DATABASE_URL': 'mock://testing',
    'REDIS_URL': 'mock://testing',
}


# ============================================================================
# ============================================================================

def log(message: str, level: str = 'INFO'):
    """Log message to stdout and simulation log."""
    timestamp = datetime.utcnow().isoformat()
    log_line = f"[{timestamp}] [{level}] {message}"
    print(log_line)
    
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(SIMULATION_LOG, 'a', encoding='ascii') as f:
        ascii_line = log_line.encode('ascii', errors='replace').decode('ascii')
        f.write(ascii_line + '\n')


# ============================================================================
# ============================================================================

def validate_sandbox():
    """Validate network sandbox configuration."""
    log("Validating network sandbox configuration...", "INFO")
    
    issues = []
    
    if sys.version_info < (3, 8):
        issues.append(f"Python 3.8+ required, found {sys.version}")
    else:
        log(f"Python version: {sys.version.split()[0]}", "OK")
    
    required_modules = ['psycopg', 'redis', 'fastapi']
    for module in required_modules:
        try:
            __import__(module)
            log(f"Module {module}: available", "OK")
        except ImportError:
            log(f"Module {module}: not available (optional in simulation mode)", "WARN")
    
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from backend.testing import no_network
        log("Network isolation module: available", "OK")
        
        if not callable(no_network.mock_psycopg_connect):
            issues.append("mock_psycopg_connect not callable")
        if not callable(no_network.mock_redis_from_url):
            issues.append("mock_redis_from_url not callable")
            
        log("Mock functions: validated", "OK")
    except ImportError as e:
        issues.append(f"Cannot import backend.testing.no_network: {e}")
    
    if not ARTIFACTS_DIR.exists():
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        log(f"Created artifacts directory: {ARTIFACTS_DIR}", "INFO")
    else:
        log(f"Artifacts directory: {ARTIFACTS_DIR}", "OK")
    
    if issues:
        log("Validation FAILED", "ERROR")
        for issue in issues:
            log(f"  - {issue}", "ERROR")
        return False
    else:
        log("Validation PASSED", "OK")
        return True


# ============================================================================
# ============================================================================

def run_simulation():
    """Run network isolation simulation."""
    log("Starting network isolation simulation...", "INFO")
    
    if SIMULATION_LOG.exists():
        SIMULATION_LOG.unlink()
    
    log("Simulation: Database mock", "TEST")
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from backend.testing.no_network import mock_psycopg_connect
        
        connect = mock_psycopg_connect()
        conn = connect()
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM proofs")
            result = cur.fetchone()
            log(f"Database mock query result: {result}", "OK")
        conn.close()
        log("Database mock: PASS", "OK")
    except Exception as e:
        log(f"Database mock: FAIL - {e}", "ERROR")
    
    log("Simulation: Redis mock", "TEST")
    try:
        from backend.testing.no_network import MockRedis
        
        redis_client = MockRedis()
        redis_client.lpush('test:queue', 'job1', 'job2')
        length = redis_client.llen('test:queue')
        log(f"Redis mock queue length: {length}", "OK")
        
        job = redis_client.rpop('test:queue')
        log(f"Redis mock pop result: {job}", "OK")
        log("Redis mock: PASS", "OK")
    except Exception as e:
        log(f"Redis mock: FAIL - {e}", "ERROR")
    
    log("Simulation: HTTP recorder", "TEST")
    try:
        from backend.testing.no_network import HTTPRecorder
        
        recorder = HTTPRecorder(str(ARTIFACTS_DIR / 'recordings'))
        recorder.record(
            'GET', 'http://example.com/api/test',
            None, 200, '{"status": "ok"}', {'Content-Type': 'application/json'}
        )
        
        response = recorder.replay('GET', 'http://example.com/api/test')
        if response and response['status'] == 200:
            log("HTTP recorder: PASS", "OK")
        else:
            log("HTTP recorder: FAIL - replay mismatch", "ERROR")
    except Exception as e:
        log(f"HTTP recorder: FAIL - {e}", "ERROR")
    
    log("Simulation: Network sandbox", "TEST")
    try:
        from backend.testing.no_network import network_sandbox, is_no_network_mode
        
        os.environ['NO_NETWORK'] = 'true'
        
        if is_no_network_mode():
            log("NO_NETWORK mode detected: true", "OK")
        else:
            log("NO_NETWORK mode detected: false", "ERROR")
        
        with network_sandbox(strict=False) as sandbox:
            log("Network sandbox context: entered", "OK")
        
        log("Network sandbox: PASS", "OK")
    except Exception as e:
        log(f"Network sandbox: FAIL - {e}", "ERROR")
    
    log("Simulation complete", "INFO")
    log(f"Simulation log written to: {SIMULATION_LOG}", "INFO")


# ============================================================================
# ============================================================================

def run_command(args: list):
    """Run command in network sandbox."""
    log(f"Running command in network sandbox: {' '.join(args)}", "INFO")
    
    env = os.environ.copy()
    env.update(NO_NETWORK_ENV)
    
    pythonpath = env.get('PYTHONPATH', '')
    if pythonpath:
        env['PYTHONPATH'] = f"{REPO_ROOT}:{pythonpath}"
    else:
        env['PYTHONPATH'] = str(REPO_ROOT)
    
    log(f"Environment: NO_NETWORK={env['NO_NETWORK']}", "INFO")
    log(f"Environment: DATABASE_URL={env['DATABASE_URL']}", "INFO")
    log(f"Environment: REDIS_URL={env['REDIS_URL']}", "INFO")
    
    try:
        result = subprocess.run(
            args,
            env=env,
            cwd=REPO_ROOT,
            capture_output=False,
            text=True
        )
        
        if result.returncode == 0:
            log(f"Command completed successfully", "OK")
        else:
            log(f"Command failed with exit code {result.returncode}", "ERROR")
        
        return result.returncode
    except Exception as e:
        log(f"Command execution failed: {e}", "ERROR")
        return 1


# ============================================================================
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Network Sandbox Mode - Run MathLedger tests without network',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/network-sandbox.py --validate
  
  python scripts/network-sandbox.py --simulate
  
  python scripts/network-sandbox.py pytest tests/
  
  python scripts/network-sandbox.py python backend/axiom_engine/derive.py --smoke-pl
        """
    )
    
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate sandbox configuration'
    )
    
    parser.add_argument(
        '--simulate',
        action='store_true',
        help='Run network isolation simulation'
    )
    
    parser.add_argument(
        'command',
        nargs='*',
        help='Command to run in sandbox'
    )
    
    args = parser.parse_args()
    
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    log("=" * 80, "INFO")
    log("MathLedger Network Sandbox Mode", "INFO")
    log("=" * 80, "INFO")
    
    if args.simulate:
        run_simulation()
        sys.exit(0)
    
    if args.validate or not args.command:
        if not validate_sandbox():
            sys.exit(1)
        if not args.command:
            sys.exit(0)
    
    if args.command:
        exit_code = run_command(args.command)
        sys.exit(exit_code)
    
    parser.print_help()
    sys.exit(1)


if __name__ == '__main__':
    main()
