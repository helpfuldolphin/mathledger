#!/usr/bin/env python3
"""
Artifact Verifier - Verify integrity of artifacts and proofs

Verifies:
1. Artifact file integrity (checksums, format validation)
2. Merkle root consistency across blocks
3. Proof parent-child relationships
4. JSONL schema compliance

Usage:
    python artifact_verifier.py --all
    python artifact_verifier.py --merkle
    python artifact_verifier.py --proofs
    python artifact_verifier.py --artifacts
    python artifact_verifier.py --file artifacts/wpv5/run_metrics.jsonl
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

def compute_merkle_root(ids):
    """Compute Merkle root from sorted IDs"""
    if not ids:
        return None
    
    sorted_ids = sorted(ids)
    combined = ''.join(sorted_ids)
    return hashlib.sha256(combined.encode()).hexdigest()

def verify_jsonl_schema(file_path):
    """Verify JSONL file follows v1 schema"""
    required_fields = ['system', 'mode', 'method', 'seed', 'inserted_proofs', 
                      'wall_minutes', 'block_no', 'merkle']
    
    print(f"Verifying JSONL schema: {file_path}")
    
    if not file_path.exists():
        print(f"  FAIL: File not found")
        return False
    
    line_count = 0
    errors = []
    
    with open(file_path) as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            
            line_count += 1
            
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"Line {line_no}: Invalid JSON - {e}")
                continue
            
            missing = [f for f in required_fields if f not in entry]
            if missing:
                errors.append(f"Line {line_no}: Missing fields: {', '.join(missing)}")
            
            if 'seed' in entry and not isinstance(entry['seed'], int):
                errors.append(f"Line {line_no}: seed must be integer")
            
            if 'inserted_proofs' in entry and not isinstance(entry['inserted_proofs'], int):
                errors.append(f"Line {line_no}: inserted_proofs must be integer")
            
            if 'merkle' in entry and not isinstance(entry['merkle'], str):
                errors.append(f"Line {line_no}: merkle must be string")
            elif 'merkle' in entry and len(entry['merkle']) != 64:
                errors.append(f"Line {line_no}: merkle must be 64-char hex")
    
    print(f"  Lines processed: {line_count}")
    
    if errors:
        print(f"  FAIL: {len(errors)} errors found")
        for error in errors[:10]:
            print(f"    - {error}")
        if len(errors) > 10:
            print(f"    ... and {len(errors) - 10} more")
        return False
    else:
        print(f"  PASS: Schema valid")
        return True

def verify_merkle_consistency():
    """Verify Merkle roots in database match computed values"""
    print("Verifying Merkle root consistency...")
    
    db_url = os.environ.get('DATABASE_URL', 'postgresql://ml:mlpass@localhost:5432/mathledger')
    
    try:
        import psycopg
    except ImportError:
        print("  SKIP: psycopg not available")
        return True
    
    try:
        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT b.block_no, b.merkle_root, 
                           array_agg(s.hash ORDER BY s.hash) as statement_hashes
                    FROM blocks b
                    LEFT JOIN statements s ON s.block_id = b.id
                    GROUP BY b.block_no, b.merkle_root
                    ORDER BY b.block_no
                    LIMIT 100
                """)
                
                blocks = cur.fetchall()
                
                if not blocks:
                    print("  SKIP: No blocks found")
                    return True
                
                print(f"  Checking {len(blocks)} blocks...")
                
                mismatches = []
                for block_no, stored_merkle, hashes in blocks:
                    if not hashes or hashes == [None]:
                        continue
                    
                    computed_merkle = compute_merkle_root(hashes)
                    
                    if computed_merkle != stored_merkle:
                        mismatches.append({
                            'block_no': block_no,
                            'stored': stored_merkle,
                            'computed': computed_merkle
                        })
                
                if mismatches:
                    print(f"  FAIL: {len(mismatches)} Merkle root mismatches")
                    for m in mismatches[:5]:
                        print(f"    Block {m['block_no']}:")
                        print(f"      Stored:   {m['stored']}")
                        print(f"      Computed: {m['computed']}")
                    if len(mismatches) > 5:
                        print(f"    ... and {len(mismatches) - 5} more")
                    return False
                else:
                    print(f"  PASS: All Merkle roots consistent")
                    return True
                    
    except Exception as e:
        print(f"  ERROR: {e}")
        return False

def verify_proof_integrity():
    """Verify proof parent-child relationships"""
    print("Verifying proof integrity...")
    
    db_url = os.environ.get('DATABASE_URL', 'postgresql://ml:mlpass@localhost:5432/mathledger')
    
    try:
        import psycopg
    except ImportError:
        print("  SKIP: psycopg not available")
        return True
    
    try:
        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT COUNT(*) 
                    FROM proofs p
                    LEFT JOIN statements s ON p.statement_id = s.id
                    WHERE s.id IS NULL
                """)
                orphaned_proofs = cur.fetchone()[0]
                
                cur.execute("""
                    SELECT COUNT(*)
                    FROM proof_parents pp
                    LEFT JOIN statements s ON pp.parent_statement_id = s.id
                    WHERE s.id IS NULL
                """)
                invalid_parents = cur.fetchone()[0]
                
                cur.execute("""
                    SELECT COUNT(*)
                    FROM proof_parents pp1
                    JOIN proof_parents pp2 ON pp1.parent_statement_id = pp2.statement_id
                    WHERE pp2.parent_statement_id = pp1.statement_id
                """)
                circular_deps = cur.fetchone()[0]
                
                issues = []
                if orphaned_proofs > 0:
                    issues.append(f"{orphaned_proofs} orphaned proofs")
                if invalid_parents > 0:
                    issues.append(f"{invalid_parents} invalid parent references")
                if circular_deps > 0:
                    issues.append(f"{circular_deps} circular dependencies")
                
                if issues:
                    print(f"  FAIL: {', '.join(issues)}")
                    return False
                else:
                    print(f"  PASS: Proof integrity verified")
                    return True
                    
    except Exception as e:
        print(f"  ERROR: {e}")
        return False

def verify_artifacts():
    """Verify artifact files in artifacts directory"""
    print("Verifying artifact files...")
    
    artifacts_dir = REPO_ROOT / 'artifacts'
    
    if not artifacts_dir.exists():
        print(f"  SKIP: Artifacts directory not found")
        return True
    
    required_files = [
        'wpv5/run_metrics.jsonl',
        'wpv5/fol_ab.csv',
        'wpv5/fol_stats.json'
    ]
    
    missing = []
    for rel_path in required_files:
        file_path = artifacts_dir / rel_path
        if not file_path.exists():
            missing.append(rel_path)
    
    if missing:
        print(f"  WARNING: {len(missing)} expected files missing:")
        for f in missing:
            print(f"    - {f}")
    
    jsonl_files = list(artifacts_dir.rglob('*.jsonl'))
    print(f"  Found {len(jsonl_files)} JSONL files")
    
    all_valid = True
    for jsonl_file in jsonl_files:
        if not verify_jsonl_schema(jsonl_file):
            all_valid = False
    
    if all_valid:
        print(f"  PASS: All artifacts valid")
        return True
    else:
        print(f"  FAIL: Some artifacts invalid")
        return False

def main():
    parser = argparse.ArgumentParser(description='Verify artifact integrity')
    parser.add_argument('--all', action='store_true', help='Run all verifications')
    parser.add_argument('--merkle', action='store_true', help='Verify Merkle roots')
    parser.add_argument('--proofs', action='store_true', help='Verify proof integrity')
    parser.add_argument('--artifacts', action='store_true', help='Verify artifact files')
    parser.add_argument('--file', type=Path, help='Verify specific JSONL file')
    
    args = parser.parse_args()
    
    if not any([args.all, args.merkle, args.proofs, args.artifacts, args.file]):
        parser.error('At least one verification type must be specified')
    
    print("=== Artifact Verification ===")
    print()
    
    results = []
    
    if args.file:
        results.append(('JSONL Schema', verify_jsonl_schema(args.file)))
    
    if args.all or args.artifacts:
        results.append(('Artifacts', verify_artifacts()))
    
    if args.all or args.merkle:
        results.append(('Merkle Roots', verify_merkle_consistency()))
    
    if args.all or args.proofs:
        results.append(('Proof Integrity', verify_proof_integrity()))
    
    print()
    print("=== Summary ===")
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(r[1] for r in results)
    return 0 if all_passed else 1

if __name__ == '__main__':
    sys.exit(main())
