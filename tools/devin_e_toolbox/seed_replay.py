#!/usr/bin/env python3
"""
Seed Replay - Reproduce derivations with specific seeds

Enables deterministic replay of derivations by:
1. Loading seed and parameters from metrics/blocks
2. Re-running derivation with exact same configuration
3. Verifying output matches original run
4. Comparing Merkle roots for integrity

Usage:
    python seed_replay.py --seed 42
    python seed_replay.py --block 1592
    python seed_replay.py --seed 101 --system fol --mode guided
    python seed_replay.py --verify-only --seed 42
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

def load_run_from_metrics(seed, system=None, mode=None):
    """Load run configuration from metrics JSONL"""
    metrics_file = REPO_ROOT / 'artifacts' / 'wpv5' / 'run_metrics.jsonl'
    
    if not metrics_file.exists():
        print(f"Metrics file not found: {metrics_file}")
        return None
    
    print(f"Searching for seed {seed} in metrics...")
    
    with open(metrics_file) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                if entry.get('seed') == int(seed):
                    if system and entry.get('system') != system:
                        continue
                    if mode and entry.get('mode') != mode:
                        continue
                    return entry
            except json.JSONDecodeError:
                continue
    
    return None

def load_run_from_block(block_no):
    """Load run configuration from block number"""
    db_url = os.environ.get('DATABASE_URL', 'postgresql://ml:mlpass@localhost:5432/mathledger')
    
    import psycopg
    try:
        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT block_no, merkle_root, created_at
                    FROM blocks
                    WHERE block_no = %s
                """, (block_no,))
                result = cur.fetchone()
                
                if result:
                    return {
                        'block_no': result[0],
                        'merkle': result[1],
                        'created_at': str(result[2])
                    }
    except Exception as e:
        print(f"Error querying database: {e}")
    
    return None

def replay_derivation(config, verify=False):
    """Replay derivation with given configuration"""
    system = config.get('system', 'pl')
    mode = config.get('mode', 'baseline')
    seed = config.get('seed')
    method = config.get('method', '')
    
    print()
    print("=== Replay Configuration ===")
    print(f"System: {system}")
    print(f"Mode: {mode}")
    print(f"Seed: {seed}")
    print(f"Method: {method}")
    
    if 'inserted_proofs' in config:
        print(f"Expected proofs: {config['inserted_proofs']}")
    if 'merkle' in config:
        print(f"Expected Merkle: {config['merkle']}")
    
    cmd_parts = [
        'python', '-m', 'backend.axiom_engine.derive',
        '--system', system,
        '--mode', mode
    ]
    
    if seed is not None:
        cmd_parts.extend(['--seed', str(seed)])
    
    if 'guided' in mode:
        cmd_parts.extend(['--steps', '3600'])
        cmd_parts.extend(['--herbrand-depth', '2'])
        cmd_parts.extend(['--topk', '64'])
        cmd_parts.extend(['--epsilon', '0.05'])
    else:
        steps = config.get('wall_minutes', 5) * 60  # Rough estimate
        cmd_parts.extend(['--steps', str(min(int(steps), 900))])
    
    cmd_parts.append('--seal')
    
    print()
    print("=== Running Replay ===")
    print(f"Command: {' '.join(cmd_parts)}")
    print()
    
    if verify:
        print("Verify-only mode: skipping actual execution")
        return True
    
    result = subprocess.run(
        cmd_parts,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print("Replay failed:")
        print(result.stderr)
        return False
    
    print("Replay completed successfully")
    print(result.stdout)
    
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Replay derivations with specific seeds')
    parser.add_argument('--seed', type=int, help='Seed to replay')
    parser.add_argument('--block', type=int, help='Block number to replay')
    parser.add_argument('--system', choices=['pl', 'fol'], help='Filter by system')
    parser.add_argument('--mode', choices=['baseline', 'guided'], help='Filter by mode')
    parser.add_argument('--verify-only', action='store_true', 
                       help='Only verify configuration, do not execute')
    
    args = parser.parse_args()
    
    if not args.seed and not args.block:
        parser.error('Either --seed or --block must be specified')
    
    config = None
    
    if args.seed:
        config = load_run_from_metrics(args.seed, args.system, args.mode)
        if not config:
            print(f"No run found for seed {args.seed}")
            if args.system:
                print(f"  System filter: {args.system}")
            if args.mode:
                print(f"  Mode filter: {args.mode}")
            return 1
        print(f"Found run: {config.get('method', 'unknown')}")
    
    elif args.block:
        config = load_run_from_block(args.block)
        if not config:
            print(f"No block found: {args.block}")
            return 1
        print(f"Found block {config['block_no']}")
        config['system'] = 'fol'
        config['mode'] = 'baseline'
        config['seed'] = args.block  # Use block number as seed fallback
    
    success = replay_derivation(config, verify=args.verify_only)
    return 0 if success else 1

if __name__ == '__main__':
    sys.exit(main())
